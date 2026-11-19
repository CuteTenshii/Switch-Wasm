//! Containers, updates, add-on content, control data and keys.

use switch_core::json;
use switch_core::nca::{ContentType, Nca};
use switch_core::source::Window;
use switch_core::trace::Level;

use wasm_bindgen::prelude::wasm_bindgen;

use crate::{container, nsp_file_source, session, Dlc, HostSource, Session, Update};

/// Open `file`, named `name`, read by range: an update or add-on content is held for the
/// title it belongs to, anything else becomes the open container. Returns `{ kind, .. }` as
/// JSON: `update` with `title` and `version`, `dlc` with `packs`, or `container` with
/// `files`. Fails with the reason when it is not a container.
#[wasm_bindgen]
pub fn switch_open_container(file: &web_sys::Blob, name: &str) -> Result<String, String> {
    let index = crate::hostfiles::add(file);
    let src = HostSource {
        file: index,
        len: file.size() as u64,
    };
    let mut held = true;
    let opened = open_file(session(), src, name, || {
        held = false;
        HostSource {
            file: 0,
            len: crate::hostfiles::open_container(file),
        }
    });
    if !held || opened.is_err() {
        crate::hostfiles::forget(index);
    }
    opened
}

/// [`switch_open_container`] over `src`. `promote` makes it the open container and returns
/// the source it is then read through.
pub(crate) fn open_file(
    s: &mut Session,
    src: HostSource,
    name: &str,
    promote: impl FnOnce() -> HostSource,
) -> Result<String, String> {
    let files = switch_core::xci::read_container(&src)
        .map_err(|e| e.to_string())?
        .files;
    let ncas = switch_core::nca::parse_ncas(&files, &src, &s.keys);
    if let Some(json) = hold_update(s, src, &files, &ncas, name) {
        return Ok(json);
    }
    if let Some(json) = hold_dlc(s, src, &files, &ncas, name) {
        return Ok(json);
    }
    s.container = Some(promote());
    s.control = None;
    s.container_nacp = None;
    let json = json::object(|o| {
        o.field("kind", "container");
        o.array("files", |a| {
            for f in &files {
                a.object(|o| {
                    o.field("name", &f.name)
                        .field("offset", f.offset)
                        .field("size", f.size);
                });
            }
        });
    });
    s.nsp_files = files;
    s.ncas = ncas;
    Ok(json)
}

/// Open `file`, named `name`, as a single standalone `.nca`. Returns 0.
#[wasm_bindgen]
pub fn switch_open_nca(file: &web_sys::Blob, name: &str) -> i32 {
    let s = session();
    s.container = Some(HostSource {
        file: 0,
        len: crate::hostfiles::open_container(file),
    });
    s.nsp_files = Vec::new();
    s.ncas = Vec::new();
    s.control = None;
    s.container_nacp = None;
    s.last_error.clear();
    s.cpu.log(
        Level::Info,
        &format!("[io] opened {name} as a standalone NCA"),
    );
    0
}

/// Register `src` as a system data archive for `OpenDataStorageByDataId`. Returns 0, or -1
/// if it is not a readable data archive.
#[cfg(target_arch = "wasm32")]
pub(crate) fn add_archive(s: &mut Session, src: HostSource) -> i32 {
    let nca = match Nca::parse_source(&src, Some(&s.keys)) {
        Ok(nca) => nca,
        Err(e) => {
            s.last_error = e.to_string();
            return -1;
        }
    };
    if !matches!(
        nca.content_type,
        ContentType::Data | ContentType::PublicData
    ) {
        s.last_error = format!(
            "not a data archive (content type {})",
            nca.content_type.name()
        );
        return -1;
    }
    let Some(index) = nca.romfs_section_index() else {
        s.last_error = "data archive has no RomFS section".into();
        return -1;
    };
    match nca.romfs_source(src, &s.keys, index) {
        Ok(romfs) => {
            s.cpu.add_data_archive(nca.title_id, Box::new(romfs));
            s.last_error.clear();
            0
        }
        Err(e) => {
            s.last_error = e.to_string();
            -1
        }
    }
}

/// Hold `src` as the update it is, for the launches of the title it patches: `{ kind, title,
/// version }` as JSON, the base program id and its NACP display version ("1.0.1", or empty).
/// `None` if it is no update.
fn hold_update(
    s: &mut Session,
    src: HostSource,
    files: &[switch_core::nsp::Pfs0File],
    ncas: &[(usize, Nca)],
    name: &str,
) -> Option<String> {
    let (index, nca) = ncas
        .iter()
        .find(|(_, nca)| nca.content_type == ContentType::Program)?;
    if !nca.is_update() {
        return None;
    }
    // An update carries its own ticket.
    let _ = switch_core::ticket::load_bundled_title_key(&mut s.keys, nca, files, &src);
    let f = &files[*index];
    let program_id = nca.program_id;
    let version = display_version(files, ncas, &src, &s.keys);
    s.update = Some(Update {
        nca: nca.clone(),
        src,
        program: (f.offset, f.size),
    });
    let shown = if version.is_empty() {
        String::new()
    } else {
        format!(" {version}")
    };
    s.cpu.log(
        Level::Info,
        &format!("[io] opened {name} as update{shown} for {program_id:016x}"),
    );
    Some(json::object(|o| {
        o.field("kind", "update")
            .field("title", format!("{program_id:016x}"))
            .field("version", version);
    }))
}

/// The NACP display version of the Control NCA among `ncas`, or empty.
fn display_version(
    files: &[switch_core::nsp::Pfs0File],
    ncas: &[(usize, Nca)],
    src: &HostSource,
    keys: &switch_core::keys::KeySet,
) -> String {
    let control = || {
        let (index, _) = ncas
            .iter()
            .find(|(_, nca)| nca.content_type == ContentType::Control)?;
        let f = files.get(*index)?;
        let window = Window::new(*src, f.offset, f.size, &f.name).ok()?;
        switch_core::control::Control::from_source(window, keys).ok()
    };
    control().map_or_else(String::new, |control| control.nacp.display_version)
}

/// Hold the add-on content `src` holds, for the launches of its title: `{ kind, packs }` as
/// JSON, each piece with its content id, base title and index. `None` if it holds none.
fn hold_dlc(
    s: &mut Session,
    src: HostSource,
    files: &[switch_core::nsp::Pfs0File],
    ncas: &[(usize, Nca)],
    name: &str,
) -> Option<String> {
    // A container with a Program NCA is a game or an update, not DLC.
    if ncas
        .iter()
        .any(|(_, nca)| nca.content_type == ContentType::Program)
    {
        return None;
    }
    let mut found = Vec::new();
    for (index, nca) in ncas {
        if !matches!(
            nca.content_type,
            ContentType::Data | ContentType::PublicData
        ) || !is_add_on_content_id(nca.title_id)
        {
            continue;
        }
        // Each piece carries its own ticket.
        let _ = switch_core::ticket::load_bundled_title_key(&mut s.keys, nca, files, &src);
        if nca.romfs_section_index().is_none() {
            continue;
        }
        let f = &files[*index];
        s.dlc.retain(|held| held.content_id != nca.title_id);
        s.dlc.push(Dlc {
            content_id: nca.title_id,
            src,
            nca: (f.offset, f.size),
        });
        found.push(nca.title_id);
    }
    let first = found.first()?;
    let packs = match found.len() {
        1 => "1 add-on content pack".to_owned(),
        n => format!("{n} add-on content packs"),
    };
    s.cpu.log(
        Level::Info,
        &format!("[io] opened {name} as {packs} for {:016x}", first & !0x1FFF),
    );
    Some(json::object(|o| {
        o.field("kind", "dlc");
        o.array("packs", |a| add_on_content_entries(a, &found));
    }))
}

/// Each add-on content id with its base title and index.
fn add_on_content_entries(a: &mut json::Array, content_ids: &[u64]) {
    for id in content_ids {
        a.object(|o| {
            o.field("id", format!("{id:016x}"))
                .field("title_id", format!("{:016x}", id & !0x1FFF))
                .field("index", id & 0x7FF);
        });
    }
}

/// Whether a title id is add-on content: the base plus 0x1000 and an 11-bit index.
fn is_add_on_content_id(title_id: u64) -> bool {
    (0x1000..0x1800).contains(&(title_id & 0x1FFF))
}

/// A JSON summary of the header of NSP file `index`'s NCA, or of the container's when it is
/// a standalone NCA, decrypting with loaded keys.
#[wasm_bindgen]
pub fn switch_inspect_nca(index: Option<u32>) -> String {
    let s = session();
    let parsed = match index {
        Some(index) => nsp_file_source(s, index)
            .and_then(|f| Nca::parse_source(&f, Some(&s.keys)).map_err(|e| e.to_string())),
        None => container(s)
            .and_then(|c| Nca::parse_source(&c, Some(&s.keys)).map_err(|e| e.to_string())),
    };
    match parsed {
        Ok(nca) => json::object(|o| {
            o.field("title_id", format!("{:016x}", nca.title_id))
                .field("content_type", nca.content_type.name())
                .field("sdk_version", format!("{:08x}", nca.sdk_version))
                .field("crypto_type", nca.crypto_type)
                .field("encrypted", nca.is_encrypted())
                .field("file_size", nca.file_size);
            o.array("sections", |a| {
                for (i, sec) in nca.sections.iter().enumerate() {
                    // The filesystem type lives in the FS header, which needs the full header to decrypt.
                    let fs_type = match nca.fs_headers.get(i).and_then(|o| o.as_ref()) {
                        Some(fs) if sec.media_size > 0 => {
                            if fs.partition_type == 1 {
                                "PFS0"
                            } else {
                                "ROMFS"
                            }
                        }
                        _ => "?",
                    };
                    a.object(|o| {
                        o.field("offset", sec.media_offset)
                            .field("size", sec.media_size)
                            .field("fs_type", fs_type);
                    });
                }
            });
        }),
        // Return the raw error; the frontend adds friendly context.
        Err(e) => json::object(|o| {
            o.field("error", e);
        }),
    }
}

/// Read and cache the open container's Control NCA (the one in its file table, or the
/// container itself when it was opened as a standalone NCA), keep its NACP for the
/// container's launches, store it for the next session's `ns`, and return it as
/// [`switch_control_json`] does. Fails with the reason when there is none readable
/// (including when no `prod.keys` are loaded).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn switch_load_control() -> Result<String, String> {
    let s = session();
    s.control = None;
    let container = container(s)?;
    let control = if s.nsp_files.is_empty() {
        switch_core::control::Control::from_source(container, &s.keys)
    } else {
        let (index, nca) = s
            .ncas
            .iter()
            .find(|(_, nca)| nca.content_type == ContentType::Control)
            .ok_or(
                "no Control NCA in this container (or its header couldn't be decrypted; load prod.keys)",
            )?;
        let _ =
            switch_core::ticket::load_bundled_title_key(&mut s.keys, nca, &s.nsp_files, &container);
        let file = nsp_file_source(s, *index as u32)?;
        switch_core::control::Control::from_source(file, &s.keys)
    }
    .map_err(|e| e.to_string())?;
    if control.title_id != 0 {
        s.cpu
            .add_application_control(control.title_id, &control.nacp_raw, &control.icon);
    }
    s.container_nacp = Some(control.nacp.clone());
    s.control = Some(control);
    crate::idb::controls::remember().await;
    Ok(switch_control_json())
}

/// The cached control data as JSON, or `{}`. `icon_size` is the length of `switch_control_icon`.
#[wasm_bindgen]
pub fn switch_control_json() -> String {
    let s = session();
    let Some(control) = &s.control else {
        return "{}".to_owned();
    };
    let nacp = &control.nacp;
    json::object(|o| {
        o.field("title_id", format!("{:016x}", control.title_id))
            .field("name", &control.name)
            .field("publisher", &control.publisher)
            .field("language", control.language)
            .field("version", &nacp.display_version)
            .field("isbn", &nacp.isbn)
            .field("error_code_category", &nacp.application_error_code_category)
            .field("startup_user_account", nacp.startup_user_account.name())
            .field("screenshot", nacp.screenshot.name())
            .field("video_capture", nacp.video_capture.name())
            .field("demo", nacp.is_demo);
        o.array("languages", |a| {
            a.extend(nacp.titles.iter().map(|title| title.language));
        });
        o.array("ratings", |a| {
            for rating in &nacp.ratings {
                a.object(|r| {
                    r.field("organisation", rating.organisation)
                        .field("age", rating.age);
                });
            }
        });
        o.field(
            "add_on_content_base_id",
            format!("{:016x}", nacp.add_on_content_base_id),
        )
        .field(
            "save_data_owner_id",
            format!("{:016x}", nacp.save_data_owner_id),
        )
        .field("user_save_size", nacp.user_account_save_data_size)
        .field(
            "user_save_journal_size",
            nacp.user_account_save_data_journal_size,
        )
        .field("device_save_size", nacp.device_save_data_size)
        .field(
            "device_save_journal_size",
            nacp.device_save_data_journal_size,
        )
        .field("bcat_storage_size", nacp.bcat_delivery_cache_storage_size)
        .field("icon_mime", control.icon_mime())
        .field("icon_size", control.icon.len());
    })
}

/// The cached icon, or empty.
#[wasm_bindgen]
pub fn switch_control_icon() -> Vec<u8> {
    session()
        .control
        .as_ref()
        .map_or_else(Vec::new, |control| control.icon.clone())
}

/// Replace the session's keys with the `prod.keys` and `title.keys` text given; either may
/// be absent.
#[wasm_bindgen]
pub fn switch_load_keys(prod: Option<Vec<u8>>, title: Option<Vec<u8>>) {
    let s = session();
    let parse = |text: Option<Vec<u8>>| {
        switch_core::keys::parse_keys_file(&String::from_utf8_lossy(&text.unwrap_or_default()))
    };
    let prod_entries = parse(prod);
    let title_entries = parse(title);
    let mut ks = switch_core::keys::keyset_from_prod(&prod_entries);
    ks.title_keys = switch_core::keys::keyset_from_title(&title_entries);
    s.keys = ks;
    // The header key decides which NCAs parse.
    if let Some(container) = s.container {
        s.ncas = switch_core::nca::parse_ncas(&s.nsp_files, &container, &s.keys);
    }
    s.last_error.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_on_content_is_listed_with_its_base_title_and_index() {
        assert_eq!(
            json::array(|a| add_on_content_entries(
                a,
                &[0x0100_0000_0000_1001, 0x0100_0000_0000_17ff]
            )),
            concat!(
                r#"[{"id":"0100000000001001","title_id":"0100000000000000","index":1},"#,
                r#"{"id":"01000000000017ff","title_id":"0100000000000000","index":2047}]"#
            )
        );
        assert_eq!(json::array(|a| add_on_content_entries(a, &[])), "[]");
    }
}
