//! Containers, updates, add-on content, control data and keys.

use switch_core::json;
use switch_core::nca::Nca;
use switch_core::nsp::Pfs0;
use switch_core::source::{ByteSource, Window};

use crate::{container, nsp_file_source, session, write_into, Dlc, HostSource, Session, Update};

/// Open the `size`-byte container (NSP or XCI) the host has ready, read through
/// `host_read`. Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn switch_open_nsp(handle: u32, size: u64) -> i32 {
    let s = session(handle);
    let container = HostSource { file: 0, len: size };
    s.container = Some(container);
    s.nsp_files = Vec::new();
    s.control = None;
    match switch_core::xci::read_container(&container) {
        Ok(pfs0) => {
            s.nsp_files = pfs0.files;
            s.last_error.clear();
            0
        }
        Err(e) => {
            s.last_error = e.to_string();
            -1
        }
    }
}

/// Open the host's container as a single standalone `.nca`. Returns 0.
#[no_mangle]
pub extern "C" fn switch_open_nca(handle: u32, size: u64) -> i32 {
    let s = session(handle);
    s.container = Some(HostSource { file: 0, len: size });
    s.nsp_files = Vec::new();
    s.control = None;
    s.last_error.clear();
    0
}

/// Register host file `file` as a system data archive for `OpenDataStorageByDataId`.
/// Returns 0, or -1 if it is not a readable data archive.
#[no_mangle]
pub extern "C" fn switch_add_archive(handle: u32, file: u32, size: u64) -> i32 {
    let s = session(handle);
    let src = HostSource { file, len: size };
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

/// Register the host's update container for the title about to run. Returns the
/// base program id it patches, or 0 with the reason in `switch_last_error`.
/// Pairing is checked at boot.
#[no_mangle]
pub extern "C" fn switch_add_update(handle: u32, file: u32, size: u64) -> u64 {
    let s = session(handle);
    let src = HostSource { file, len: size };
    let files = match Pfs0::read_from(&src) {
        Ok(pfs0) => pfs0.files,
        Err(e) => {
            s.last_error = format!("an update has to be an NSP: {e}");
            return 0;
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
        return 0;
    };
    // An update carries its own ticket.
    let _ = switch_core::ticket::load_bundled_title_key(&mut s.keys, &nca, &files, &src);
    if !nca.is_update() {
        s.last_error =
            "this container is a title in its own right, not an update: its RomFS is its own"
                .into();
        return 0;
    }
    let f = &files[index];
    let program = (f.offset, f.size);
    let program_id = nca.program_id;
    s.update = Some(Update {
        nca,
        src,
        program,
        files,
    });
    s.last_error.clear();
    program_id
}

/// The update's NACP display version ("1.0.1") into `buf`, or empty.
#[no_mangle]
pub extern "C" fn switch_update_version(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    let version = s
        .update
        .as_ref()
        .and_then(|update| {
            let (index, _) =
                switch_core::control::find_control_nca(&update.files, &update.src, &s.keys)?;
            let f = update.files.get(index)?;
            let window = Window::new(update.src, f.offset, f.size, &f.name).ok()?;
            let control = switch_core::control::Control::from_source(window, &s.keys).ok()?;
            Some(control.nacp.display_version)
        })
        .unwrap_or_default();
    write_into(buf, maxlen, version.as_bytes())
}

/// Register a container of add-on content. Returns how many pieces it holds, or 0
/// with the reason in `switch_last_error`. Pairing is checked at boot.
#[no_mangle]
pub extern "C" fn switch_add_dlc(handle: u32, file: u32, size: u64) -> u32 {
    let s = session(handle);
    let src = HostSource { file, len: size };
    let files = match Pfs0::read_from(&src) {
        Ok(pfs0) => pfs0.files,
        Err(e) => {
            s.last_error = format!("add-on content has to be an NSP: {e}");
            return 0;
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
        return 0;
    }

    let mut found = 0;
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
        found += 1;
    }
    if found == 0 {
        s.last_error = "no add-on content in this container".into();
    } else {
        s.last_error.clear();
    }
    found
}

/// The session's add-on content as JSON: content id, index and base title.
#[no_mangle]
pub extern "C" fn switch_dlc_json(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    let json = dlc_json(s.dlc.iter().map(|dlc| dlc.content_id));
    write_into(buf, maxlen, json.as_bytes())
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

#[no_mangle]
pub extern "C" fn switch_clear_dlc(handle: u32) {
    session(handle).dlc.clear();
}

#[no_mangle]
pub extern "C" fn switch_clear_update(handle: u32) {
    session(handle).update = None;
}

/// Whether a title id is add-on content: the base plus 0x1000 and an 11-bit index.
fn is_add_on_content_id(title_id: u64) -> bool {
    (0x1000..0x1800).contains(&(title_id & 0x1FFF))
}

/// Parse the file table of the current NSP and return it as JSON.
/// Writes up to `maxlen` bytes into `buf`; returns bytes written.
#[no_mangle]
pub extern "C" fn switch_nsp_files_json(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    let json = json::array(|a| {
        for f in &s.nsp_files {
            a.object(|o| {
                o.field("name", &f.name)
                    .field("offset", f.offset)
                    .field("size", f.size);
            });
        }
    });
    write_into(buf, maxlen, json.as_bytes())
}

/// Read a slice of NSP file `index` from `file_offset` into `buf`. Returns bytes
/// copied or -1.
#[no_mangle]
pub extern "C" fn switch_read_file(
    handle: u32,
    index: u32,
    file_offset: u64,
    buf: *mut u8,
    maxlen: u32,
) -> i64 {
    let s = session(handle);
    let Some(file) = nsp_file_source(s, index) else {
        return -1;
    };
    if buf.is_null() || maxlen == 0 || file_offset >= file.len() {
        return 0;
    }
    let n = (maxlen as u64).min(file.len() - file_offset) as usize;
    // SAFETY: JS allocated `maxlen` bytes at `buf`, and `n` is no larger.
    let out = unsafe { std::slice::from_raw_parts_mut(buf, n) };
    match file.read_at(file_offset, out) {
        Ok(got) => got as i64,
        Err(e) => {
            s.last_error = e.to_string();
            -1
        }
    }
}

/// Parse an NCA from `ptr`/`len` and return a JSON summary, decrypting with loaded keys.
#[no_mangle]
pub extern "C" fn switch_parse_nca(
    handle: u32,
    ptr: *const u8,
    len: u32,
    buf: *mut u8,
    maxlen: u32,
) -> u32 {
    let s = session(handle);
    let data = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    let json = match Nca::parse_with_keys(data, Some(&s.keys)) {
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
    };
    write_into(buf, maxlen, json.as_bytes())
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

/// Read and cache the open container's Control NCA. Returns 0, or -1 when it has
/// none readable (including when no `prod.keys` are loaded).
#[no_mangle]
pub extern "C" fn switch_load_control_from_nsp(handle: u32) -> i32 {
    let s = session(handle);
    s.control = None;
    let Some(container) = container(s) else {
        return -1;
    };
    let found = switch_core::control::find_control_nca(&s.nsp_files, &container, &s.keys);
    let Some((index, nca)) = found else {
        s.last_error =
            "no Control NCA in this container (or its header couldn't be decrypted; load prod.keys)"
                .into();
        return -1;
    };
    let _ =
        switch_core::ticket::load_bundled_title_key(&mut s.keys, &nca, &s.nsp_files, &container);
    let Some(file) = nsp_file_source(s, index as u32) else {
        return -1;
    };
    match switch_core::control::Control::from_source(file, &s.keys) {
        Ok(control) => {
            cache_control(s, control);
            s.last_error.clear();
            0
        }
        Err(e) => {
            s.last_error = e.to_string();
            -1
        }
    }
}

/// Same, for a container opened as a standalone Control NCA.
#[no_mangle]
pub extern "C" fn switch_load_control_from_nca(handle: u32) -> i32 {
    let s = session(handle);
    s.control = None;
    let Some(container) = container(s) else {
        return -1;
    };
    match switch_core::control::Control::from_source(container, &s.keys) {
        Ok(control) => {
            cache_control(s, control);
            s.last_error.clear();
            0
        }
        Err(e) => {
            s.last_error = e.to_string();
            -1
        }
    }
}

/// Record another title's `control.nacp` and icon for `ns`'s `GetApplicationControlData`.
#[no_mangle]
pub extern "C" fn switch_add_application_control(
    handle: u32,
    application_id: u64,
    nacp_ptr: *const u8,
    nacp_len: u32,
    icon_ptr: *const u8,
    icon_len: u32,
) -> i32 {
    let nacp = unsafe { std::slice::from_raw_parts(nacp_ptr, nacp_len as usize) };
    let icon = unsafe { std::slice::from_raw_parts(icon_ptr, icon_len as usize) };
    session(handle)
        .cpu
        .add_application_control(application_id, nacp, icon);
    0
}

/// The cached `control.nacp` as stored; returns its length, or -1 with none cached.
#[no_mangle]
pub extern "C" fn switch_control_nacp(handle: u32, buf: *mut u8, maxlen: u32) -> i64 {
    let s = session(handle);
    let Some(control) = &s.control else {
        return -1;
    };
    let n = control.nacp_raw.len().min(maxlen as usize);
    if n > 0 && !buf.is_null() {
        unsafe {
            std::ptr::copy_nonoverlapping(control.nacp_raw.as_ptr(), buf, n);
        }
    }
    n as i64
}

/// The cached control data as JSON, or `{}`. `icon_size` sizes `switch_control_icon`'s buffer.
#[no_mangle]
pub extern "C" fn switch_control_json(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    let Some(control) = &s.control else {
        return write_into(buf, maxlen, b"{}");
    };
    let nacp = &control.nacp;
    let json = json::object(|o| {
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
    });
    write_into(buf, maxlen, json.as_bytes())
}

/// Copy the cached icon into `buf`. Returns bytes copied, or -1.
#[no_mangle]
pub extern "C" fn switch_control_icon(handle: u32, buf: *mut u8, maxlen: u32) -> i64 {
    let s = session(handle);
    let Some(control) = &s.control else {
        return -1;
    };
    let n = control.icon.len().min(maxlen as usize);
    if n > 0 && !buf.is_null() {
        unsafe {
            std::ptr::copy_nonoverlapping(control.icon.as_ptr(), buf, n);
        }
    }
    n as i64
}

/// Load `prod.keys` / `title.keys` text files into the session. Either pointer
/// may be NULL with length 0. Returns 0 on success, -1 on parse failure.
#[no_mangle]
pub extern "C" fn switch_load_keys(
    handle: u32,
    prod_ptr: *const u8,
    prod_len: u32,
    title_ptr: *const u8,
    title_len: u32,
) -> i32 {
    let s = session(handle);
    let prod = if !prod_ptr.is_null() && prod_len > 0 {
        unsafe { std::slice::from_raw_parts(prod_ptr, prod_len as usize) }
    } else {
        &[]
    };
    let title = if !title_ptr.is_null() && title_len > 0 {
        unsafe { std::slice::from_raw_parts(title_ptr, title_len as usize) }
    } else {
        &[]
    };
    let prod_text = String::from_utf8_lossy(prod);
    let title_text = String::from_utf8_lossy(title);
    let prod_entries = switch_core::keys::parse_keys_file(&prod_text);
    let title_entries = switch_core::keys::parse_keys_file(&title_text);
    let mut ks = switch_core::keys::keyset_from_prod(&prod_entries);
    ks.title_keys = switch_core::keys::keyset_from_title(&title_entries);
    s.keys = ks;
    s.last_error.clear();
    0
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
