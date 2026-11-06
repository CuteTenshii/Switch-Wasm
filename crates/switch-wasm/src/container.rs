//! Containers, updates, add-on content, control data and keys.

use switch_core::json;
use switch_core::nca::Nca;
use switch_core::nsp::Pfs0;
use switch_core::source::{ByteSource, Window};

use wasm_bindgen::prelude::wasm_bindgen;

use crate::{container, nsp_file_source, session, Dlc, HostSource, Session, Update};

/// Open the `size`-byte container (NSP or XCI) the host has ready, read through
/// `host_read`. Returns its file table as JSON, or fails with the reason.
#[wasm_bindgen]
pub fn switch_open_nsp(size: u64) -> Result<String, String> {
    let s = session();
    let container = HostSource { file: 0, len: size };
    s.container = Some(container);
    s.nsp_files = Vec::new();
    s.control = None;
    match switch_core::xci::read_container(&container) {
        Ok(pfs0) => {
            s.nsp_files = pfs0.files;
            Ok(json::array(|a| {
                for f in &s.nsp_files {
                    a.object(|o| {
                        o.field("name", &f.name)
                            .field("offset", f.offset)
                            .field("size", f.size);
                    });
                }
            }))
        }
        Err(e) => Err(e.to_string()),
    }
}

/// Open the host's container as a single standalone `.nca`. Returns 0.
#[wasm_bindgen]
pub fn switch_open_nca(size: u64) -> i32 {
    let s = session();
    s.container = Some(HostSource { file: 0, len: size });
    s.nsp_files = Vec::new();
    s.control = None;
    s.last_error.clear();
    0
}

/// Register `src` as a system data archive for `OpenDataStorageByDataId`. Returns 0, or -1
/// if it is not a readable data archive.
#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
pub(crate) fn add_archive(s: &mut Session, src: HostSource) -> i32 {
    let nca = match Nca::parse_source(&src, Some(&s.keys)) {
        Ok(nca) => nca,
        Err(e) => {
            s.last_error = e.to_string();
            return -1;
        }
    };
    use switch_core::nca::ContentType;
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

/// Register the host's update container for the title about to run. Returns `{ title,
/// version }` as JSON, the base program id it patches and its NACP display version
/// ("1.0.1", or empty), or nothing with the reason in `switch_last_error`. Pairing is
/// checked at boot.
#[wasm_bindgen]
pub fn switch_add_update(file: u32, size: u64) -> String {
    let s = session();
    let src = HostSource { file, len: size };
    let files = match Pfs0::read_from(&src) {
        Ok(pfs0) => pfs0.files,
        Err(e) => {
            s.last_error = format!("an update has to be an NSP: {e}");
            return String::new();
        }
    };
    let Some((index, nca)) = switch_core::nca::find_nca_by_type(
        &files,
        &src,
        &s.keys,
        switch_core::nca::ContentType::Program,
    ) else {
        s.last_error =
            "no Program NCA in this container (or its header couldn't be decrypted; load prod.keys)"
                .into();
        return String::new();
    };
    // An update carries its own ticket.
    let _ = switch_core::ticket::load_bundled_title_key(&mut s.keys, &nca, &files, &src);
    if !nca.is_update() {
        s.last_error =
            "this container is a title in its own right, not an update: its RomFS is its own"
                .into();
        return String::new();
    }
    let f = &files[index];
    let program = (f.offset, f.size);
    let program_id = nca.program_id;
    let version = display_version(&files, &src, &s.keys);
    s.update = Some(Update { nca, src, program });
    s.last_error.clear();
    json::object(|o| {
        o.field("title", format!("{program_id:016x}"))
            .field("version", version);
    })
}

/// The NACP display version of the Control NCA among `files`, or empty.
fn display_version(
    files: &[switch_core::nsp::Pfs0File],
    src: &HostSource,
    keys: &switch_core::keys::KeySet,
) -> String {
    let control = || {
        let (index, _) = switch_core::control::find_control_nca(files, src, keys)?;
        let f = files.get(index)?;
        let window = Window::new(*src, f.offset, f.size, &f.name).ok()?;
        switch_core::control::Control::from_source(window, keys).ok()
    };
    control().map_or_else(String::new, |control| control.nacp.display_version)
}

/// Register a container of add-on content. Returns the pieces it holds as JSON, each with its
/// content id, base title and index; none, with the reason in `switch_last_error`, if it is
/// not add-on content. Pairing is checked at boot.
#[wasm_bindgen]
pub fn switch_add_dlc(file: u32, size: u64) -> String {
    let s = session();
    let src = HostSource { file, len: size };
    let files = match Pfs0::read_from(&src) {
        Ok(pfs0) => pfs0.files,
        Err(e) => {
            s.last_error = format!("add-on content has to be an NSP: {e}");
            return dlc_json([].into_iter());
        }
    };
    // A container with a Program NCA is a game or an update, not DLC.
    if switch_core::nca::find_nca_by_type(
        &files,
        &src,
        &s.keys,
        switch_core::nca::ContentType::Program,
    )
    .is_some()
    {
        s.last_error = "this container holds a program; add-on content is data only".into();
        return dlc_json([].into_iter());
    }

    let mut found = Vec::new();
    for f in &files {
        if !f.name.to_ascii_lowercase().ends_with(".nca") {
            continue;
        }
        let Ok(window) = Window::new(src, f.offset, f.size, &f.name) else {
            continue;
        };
        let Ok(nca) = Nca::parse_source(&window, Some(&s.keys)) else {
            continue;
        };
        use switch_core::nca::ContentType;
        if !matches!(
            nca.content_type,
            ContentType::Data | ContentType::PublicData
        ) || !is_add_on_content_id(nca.title_id)
        {
            continue;
        }
        // Each piece carries its own ticket.
        let _ = switch_core::ticket::load_bundled_title_key(&mut s.keys, &nca, &files, &src);
        if nca.romfs_section_index().is_none() {
            continue;
        }
        s.dlc.retain(|held| held.content_id != nca.title_id);
        s.dlc.push(Dlc {
            content_id: nca.title_id,
            src,
            nca: (f.offset, f.size),
        });
        found.push(nca.title_id);
    }
    if found.is_empty() {
        s.last_error = "no add-on content in this container".into();
    } else {
        s.last_error.clear();
    }
    dlc_json(found.into_iter())
}

/// Each add-on content id with its base title and index.
fn dlc_json(content_ids: impl Iterator<Item = u64>) -> String {
    json::array(|a| {
        for id in content_ids {
            a.object(|o| {
                o.field("id", format!("{id:016x}"))
                    .field("title_id", format!("{:016x}", id & !0x1FFF))
                    .field("index", id & 0x7FF);
            });
        }
    })
}

#[wasm_bindgen]
pub fn switch_clear_dlc() {
    session().dlc.clear();
}

#[wasm_bindgen]
pub fn switch_clear_update() {
    session().update = None;
}

/// Whether a title id is add-on content: the base plus 0x1000 and an 11-bit index.
fn is_add_on_content_id(title_id: u64) -> bool {
    (0x1000..0x1800).contains(&(title_id & 0x1FFF))
}

/// Read up to `len` bytes of NSP file `index` from `file_offset`; fails with the reason.
#[wasm_bindgen]
pub fn switch_read_file(index: u32, file_offset: u64, len: u32) -> Result<Vec<u8>, String> {
    let file = nsp_file_source(session(), index)?;
    if len == 0 || file_offset >= file.len() {
        return Ok(Vec::new());
    }
    let n = u64::from(len).min(file.len() - file_offset) as usize;
    let mut out = vec![0u8; n];
    let got = file
        .read_at(file_offset, &mut out)
        .map_err(|e| e.to_string())?;
    out.truncate(got);
    Ok(out)
}

/// A JSON summary of the NCA whose header is `data`, decrypting with loaded keys.
#[wasm_bindgen]
pub fn switch_parse_nca(data: &[u8]) -> String {
    let s = session();
    match Nca::parse_with_keys(data, Some(&s.keys)) {
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
                            if fs.fs_type == 1 {
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
            o.field("error", e.to_string());
        }),
    }
}

/// Cache control data and pass its NACP figures (save data, add-on base id) to the CPU.
fn cache_control(s: &mut Session, control: switch_core::control::Control) {
    if control.title_id != 0 {
        s.cpu
            .add_application_control(control.title_id, &control.nacp_raw, &control.icon);
    }
    s.cpu
        .set_save_data_quota(switch_core::cpu::SaveDataQuota::from(&control.nacp));
    s.cpu
        .set_add_on_content_base_id(control.nacp.add_on_content_base_id);
    s.control = Some(control);
}

/// Read and cache the open container's Control NCA: the one in its file table, or the
/// container itself when it was opened as a standalone NCA (which has no file table).
/// Fails with the reason when it has none readable (including when no `prod.keys` are loaded).
#[wasm_bindgen]
pub fn switch_load_control() -> Result<(), String> {
    let s = session();
    s.control = None;
    let container = container(s)?;
    let control = if s.nsp_files.is_empty() {
        switch_core::control::Control::from_source(container, &s.keys)
    } else {
        let (index, nca) =
            switch_core::control::find_control_nca(&s.nsp_files, &container, &s.keys).ok_or(
                "no Control NCA in this container (or its header couldn't be decrypted; load prod.keys)",
            )?;
        let _ = switch_core::ticket::load_bundled_title_key(
            &mut s.keys,
            &nca,
            &s.nsp_files,
            &container,
        );
        let file = nsp_file_source(s, index as u32)?;
        switch_core::control::Control::from_source(file, &s.keys)
    };
    cache_control(s, control.map_err(|e| e.to_string())?);
    Ok(())
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
    s.last_error.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_on_content_is_listed_with_its_base_title_and_index() {
        assert_eq!(
            dlc_json([0x0100_0000_0000_1001u64, 0x0100_0000_0000_17ff].into_iter()),
            concat!(
                r#"[{"id":"0100000000001001","title_id":"0100000000000000","index":1},"#,
                r#"{"id":"01000000000017ff","title_id":"0100000000000000","index":2047}]"#
            )
        );
        assert_eq!(dlc_json(std::iter::empty()), "[]");
    }
}
