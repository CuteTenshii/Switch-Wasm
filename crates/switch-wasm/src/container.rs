//! Containers, updates, add-on content, control data and keys.

use switch_core::nca::Nca;
use switch_core::nsp::Pfs0;
use switch_core::source::{ByteSource, Window};

use crate::{
    container, json_escape, nsp_file_source, session, write_into, Dlc, HostSource, Session, Update,
};

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
            "no Program NCA in this container (or its header couldn't be decrypted — load prod.keys)"
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
        s.last_error = "this container holds a program — add-on content is data only".into();
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
    let mut out = Vec::new();
    out.push(b'[');
    for (i, dlc) in s.dlc.iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        out.extend_from_slice(
            format!(
                "{{\"id\":\"{:016x}\",\"title_id\":\"{:016x}\",\"index\":{}}}",
                dlc.content_id,
                dlc.content_id & !0x1FFF,
                dlc.content_id & 0x7FF
            )
            .as_bytes(),
        );
    }
    out.push(b']');
    write_into(buf, maxlen, &out)
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
    let mut out = Vec::new();
    out.extend_from_slice(b"[");
    for (i, f) in s.nsp_files.iter().enumerate() {
        if i > 0 {
            out.extend_from_slice(b",");
        }
        out.extend_from_slice(b"{\"name\":\"");
        json_escape(&f.name, &mut out);
        out.extend_from_slice(b"\",\"offset\":");
        out.extend_from_slice(f.offset.to_string().as_bytes());
        out.extend_from_slice(b",\"size\":");
        out.extend_from_slice(f.size.to_string().as_bytes());
        out.extend_from_slice(b"}");
    }
    out.extend_from_slice(b"]");
    write_into(buf, maxlen, &out)
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
    let mut out = Vec::new();
    match Nca::parse_with_keys(data, Some(&s.keys)) {
        Ok(nca) => {
            out.extend_from_slice(b"{\"title_id\":\"");
            out.extend_from_slice(format!("{:016x}", nca.title_id).as_bytes());
            out.extend_from_slice(b"\",\"content_type\":\"");
            out.extend_from_slice(nca.content_type.name().as_bytes());
            out.extend_from_slice(b"\",\"sdk_version\":\"");
            out.extend_from_slice(format!("{:08x}", nca.sdk_version).as_bytes());
            out.extend_from_slice(b"\",\"crypto_type\":");
            out.extend_from_slice(nca.crypto_type.to_string().as_bytes());
            out.extend_from_slice(b",\"encrypted\":");
            out.extend_from_slice(if nca.is_encrypted() {
                b"true"
            } else {
                b"false"
            });
            out.extend_from_slice(b",\"file_size\":");
            out.extend_from_slice(nca.file_size.to_string().as_bytes());
            out.extend_from_slice(b",\"sections\":[");
            for (i, sec) in nca.sections.iter().enumerate() {
                if i > 0 {
                    out.extend_from_slice(b",");
                }
                out.extend_from_slice(b"{\"offset\":");
                out.extend_from_slice(sec.media_offset.to_string().as_bytes());
                out.extend_from_slice(b",\"size\":");
                out.extend_from_slice(sec.media_size.to_string().as_bytes());
                out.extend_from_slice(b",\"fs_type\":\"");
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
                out.extend_from_slice(fs_type.as_bytes());
                out.extend_from_slice(b"\"}");
            }
            out.extend_from_slice(b"]}");
        }
        Err(e) => {
            // Return the raw error; the frontend adds friendly context.
            out.extend_from_slice(b"{\"error\":\"");
            json_escape(&e.to_string(), &mut out);
            out.extend_from_slice(b"\"}");
        }
    }
    write_into(buf, maxlen, &out)
}

/// Cache control data and pass its NACP figures (save data, add-on base id) to the CPU.
fn cache_control(s: &mut Session, control: switch_core::control::Control) {
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
            "no Control NCA in this container (or its header couldn't be decrypted — load prod.keys)"
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

/// The cached control data as JSON, or `{}`. `icon_size` sizes `switch_control_icon`'s buffer.
#[no_mangle]
pub extern "C" fn switch_control_json(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let s = session(handle);
    let mut out = Vec::new();
    let Some(control) = &s.control else {
        out.extend_from_slice(b"{}");
        return write_into(buf, maxlen, &out);
    };
    let nacp = &control.nacp;
    out.extend_from_slice(b"{\"title_id\":\"");
    out.extend_from_slice(format!("{:016x}", control.title_id).as_bytes());
    out.extend_from_slice(b"\",\"name\":\"");
    json_escape(&control.name, &mut out);
    out.extend_from_slice(b"\",\"publisher\":\"");
    json_escape(&control.publisher, &mut out);
    out.extend_from_slice(b"\",\"language\":\"");
    json_escape(control.language, &mut out);
    out.extend_from_slice(b"\",\"version\":\"");
    json_escape(&nacp.display_version, &mut out);
    out.extend_from_slice(b"\",\"isbn\":\"");
    json_escape(&nacp.isbn, &mut out);
    out.extend_from_slice(b"\",\"error_code_category\":\"");
    json_escape(&nacp.application_error_code_category, &mut out);
    out.extend_from_slice(b"\",\"startup_user_account\":\"");
    out.extend_from_slice(nacp.startup_user_account.name().as_bytes());
    out.extend_from_slice(b"\",\"screenshot\":\"");
    out.extend_from_slice(nacp.screenshot.name().as_bytes());
    out.extend_from_slice(b"\",\"video_capture\":\"");
    out.extend_from_slice(nacp.video_capture.name().as_bytes());
    out.extend_from_slice(b"\",\"demo\":");
    out.extend_from_slice(if nacp.is_demo { b"true" } else { b"false" });
    out.extend_from_slice(b",\"languages\":[");
    for (i, title) in nacp.titles.iter().enumerate() {
        if i > 0 {
            out.extend_from_slice(b",");
        }
        out.extend_from_slice(b"\"");
        json_escape(title.language, &mut out);
        out.extend_from_slice(b"\"");
    }
    out.extend_from_slice(b"],\"ratings\":[");
    for (i, rating) in nacp.ratings.iter().enumerate() {
        if i > 0 {
            out.extend_from_slice(b",");
        }
        out.extend_from_slice(b"{\"organisation\":\"");
        json_escape(rating.organisation, &mut out);
        out.extend_from_slice(b"\",\"age\":");
        out.extend_from_slice(rating.age.to_string().as_bytes());
        out.extend_from_slice(b"}");
    }
    out.extend_from_slice(b"],\"add_on_content_base_id\":\"");
    out.extend_from_slice(format!("{:016x}", nacp.add_on_content_base_id).as_bytes());
    out.extend_from_slice(b"\",\"save_data_owner_id\":\"");
    out.extend_from_slice(format!("{:016x}", nacp.save_data_owner_id).as_bytes());
    out.extend_from_slice(b"\",\"user_save_size\":");
    out.extend_from_slice(nacp.user_account_save_data_size.to_string().as_bytes());
    out.extend_from_slice(b",\"user_save_journal_size\":");
    out.extend_from_slice(
        nacp.user_account_save_data_journal_size
            .to_string()
            .as_bytes(),
    );
    out.extend_from_slice(b",\"device_save_size\":");
    out.extend_from_slice(nacp.device_save_data_size.to_string().as_bytes());
    out.extend_from_slice(b",\"device_save_journal_size\":");
    out.extend_from_slice(nacp.device_save_data_journal_size.to_string().as_bytes());
    out.extend_from_slice(b",\"bcat_storage_size\":");
    out.extend_from_slice(nacp.bcat_delivery_cache_storage_size.to_string().as_bytes());
    out.extend_from_slice(b",\"icon_mime\":\"");
    out.extend_from_slice(control.icon_mime().as_bytes());
    out.extend_from_slice(b"\",\"icon_size\":");
    out.extend_from_slice(control.icon.len().to_string().as_bytes());
    out.extend_from_slice(b"}");
    write_into(buf, maxlen, &out)
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
