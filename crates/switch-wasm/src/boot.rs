//! Loading and booting programs.

use switch_core::cpu::Cpu;
use switch_core::nca::Nca;
use switch_core::source::{ByteSource, Window};
use switch_core::trace::Level;
use wasm_bindgen::prelude::*;

use crate::{container, nsp_file_source, session, Added, Dlc, Session};

/// End the running title and its applets, leaving the console ready for the next one.
#[wasm_bindgen]
pub fn switch_end_title() {
    let s = session();
    s.cpu.end_title();
    crate::audio::drop_queued();
    crate::input::reset();
    s.control = None;
    s.last_error.clear();
}

/// Load an NRO homebrew image into the CPU, told the SD card path it came from when there
/// is one. Returns the entry address.
fn load_nro_bytes(s: &mut Session, data: &[u8], sd_path: Option<&str>) -> Result<u64, String> {
    // Booting empties the output and trace buffers.
    crate::log::collect(&mut s.cpu);
    let booted = match sd_path {
        None => s.cpu.boot_homebrew(data),
        Some(path) => s.cpu.boot_homebrew_at(data, path),
    };
    s.control = None;
    let loaded = booted.map_err(|e| e.to_string())?;
    // Cached for display only: homebrew runs in another title's process.
    s.control = switch_core::control::Control::from_nro(data);
    let what = match &s.control {
        Some(c) if !c.publisher.is_empty() => format!("{} - {}", c.name, c.publisher),
        Some(c) if !c.name.is_empty() => c.name.clone(),
        _ => "an NRO".to_owned(),
    };
    s.cpu.log(
        Level::Info,
        &format!("Loaded {what}, entry {:#010x}", loaded.entry),
    );
    crate::log::collect(&mut s.cpu);
    s.cpu.halted = false;
    Ok(loaded.entry as u64)
}

/// Load the program in `file`, an ELF or an NRO by its magic, read here. Returns the entry
/// address, or fails with the reason.
#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
#[wasm_bindgen]
pub async fn switch_load_program(file: web_sys::Blob) -> Result<f64, String> {
    let data = crate::blob::read_vec(&file)
        .await
        .map_err(|e| format!("the file could not be read: {e:?}"))?;
    let s = session();
    let entry = if data.starts_with(b"\x7fELF") {
        load_elf_bytes(s, &data)
    } else {
        load_nro_bytes(s, &data, None)
    };
    entry.map(|entry| entry as f64)
}

/// Load the NRO at `path` on the SD card, which is told its path. Returns the entry
/// address, or fails with the reason.
#[wasm_bindgen]
pub fn switch_load_sd_nro(path: &str) -> Result<f64, String> {
    let s = session();
    let Some(size) = s.cpu.fs.size(path) else {
        return Err(format!("{path} is not on the SD card"));
    };
    let mut data = vec![0u8; size as usize];
    s.cpu.fs.read(path, 0, &mut data);
    load_nro_bytes(s, &data, Some(path)).map(|entry| entry as f64)
}

/// Decrypt a Program NCA's ExeFS from `nca_src`, load it and boot. Returns the entry address.
/// `nca_src` stays alive as the title's RomFS source. `added` holds the update and DLC.
pub(crate) fn load_and_boot_nca<S: ByteSource + 'static>(
    keys: &switch_core::keys::KeySet,
    cpu: &mut Cpu,
    nca_src: S,
    added: Added<'_>,
) -> Result<u64, String> {
    let nca = Nca::parse_source(&nca_src, Some(keys)).map_err(|e| e.to_string())?;
    // Refuse an update for a different title.
    let update = match added.update {
        Some(u) if u.nca.program_id == nca.program_id => Some(u),
        Some(u) => {
            return Err(format!(
                "the update added to this session is for title {:016x}, but this container is {:016x}",
                u.nca.program_id, nca.program_id
            ));
        }
        None => None,
    };
    // An update's ExeFS is a complete replacement set of modules.
    let program = update.map_or(&nca, |u| &u.nca);
    let exefs_index = program
        .exefs_section_index()
        .ok_or("no ExeFS (PFS0) section in this NCA")?;
    let exefs = match update {
        Some(u) => u
            .program_window()
            .and_then(|window| program.read_pfs0_section(window, keys, exefs_index)),
        None => program.read_pfs0_section(&nca_src, keys, exefs_index),
    }
    .map_err(|e| e.to_string())?;
    if update.is_some() {
        cpu.log(
            Level::Info,
            &format!(
                "[update] booting the update's modules for {:016x}, over this container's RomFS",
                nca.program_id
            ),
        );
    }
    // Report whether the ExeFS hash coverage was checked.
    match program.pfs0_hash_coverage(exefs_index) {
        Some((block, blocks)) => cpu.log(
            Level::Info,
            &format!(
                "[exefs] {:#x} bytes, {} blocks of {:#x} verified against the section hash table",
                exefs.len(),
                blocks,
                block
            ),
        ),
        None => cpu.log(
            Level::Warn,
            &format!(
                "[exefs] {:#x} bytes; hash table geometry unrecognised, contents NOT verified",
                exefs.len()
            ),
        ),
    }

    let program =
        switch_core::cpu::Program::parse(program.program_id, exefs).map_err(|e| e.to_string())?;

    // RomFS is optional and failures do not block booting; it is read by range.
    match update {
        // An update's RomFS is a patch, readable only over the base's.
        Some(u) => {
            let patched = u.program_window().and_then(|window| {
                switch_core::bktr::patched_romfs_source(&u.nca, window, &nca, nca_src, keys)
            });
            match patched {
                Ok(romfs) => cpu.set_romfs_source(Box::new(romfs)),
                Err(e) => cpu.log(
                    Level::Error,
                    &format!("the update's RomFS is unreadable: {}", e),
                ),
            }
        }
        None => {
            if let Some(romfs_index) = nca.romfs_section_index() {
                match nca.romfs_source(nca_src, keys, romfs_index) {
                    Ok(romfs) => cpu.set_romfs_source(Box::new(romfs)),
                    Err(e) => cpu.log(Level::Error, &format!("romfs unavailable: {}", e)),
                }
            }
        }
    }

    // Booting empties the output and trace buffers.
    crate::log::collect(cpu);
    let loaded = cpu.boot_program(&program).map_err(|e| e.to_string())?;
    // After the modules, so its lines follow the boot's.
    mount_add_on_content(cpu, keys, added.dlc);
    Ok(u64::from(loaded[0].entry))
}

/// Mount the added DLC whose ids belong to this title; report and skip the rest.
fn mount_add_on_content(cpu: &mut Cpu, keys: &switch_core::keys::KeySet, dlc: &[Dlc]) {
    for entry in dlc {
        let romfs = entry.window().and_then(|window| {
            let nca = Nca::parse_source(&window, Some(keys))?;
            let index = nca
                .romfs_section_index()
                .ok_or_else(|| switch_core::Error::Nca("no RomFS in this archive".into()))?;
            nca.romfs_source(window, keys, index)
        });
        match romfs {
            Ok(romfs) => {
                let size = romfs.len();
                match cpu.add_add_on_content(entry.content_id, Box::new(romfs)) {
                    Some(index) => cpu.log(
                        Level::Info,
                        &format!(
                            "[aoc] {:016x} mounted as add-on content {index}, {size:#x} bytes",
                            entry.content_id
                        ),
                    ),
                    None => cpu.log(
                        Level::Warn,
                        &format!(
                            "[aoc] {:016x} is not this title's add-on content; not mounted",
                            entry.content_id
                        ),
                    ),
                }
            }
            Err(e) => cpu.log(
                Level::Error,
                &format!("[aoc] {:016x} could not be read: {e}", entry.content_id),
            ),
        }
    }
}

/// Boot the open container as a standalone Program NCA. Returns the entry address, or fails
/// with the reason.
#[wasm_bindgen]
pub fn switch_load_nca() -> Result<f64, String> {
    let s = session();
    let container = container(s)?;
    let added = Added {
        update: s.update.as_ref(),
        dlc: &s.dlc,
    };
    let entry = load_and_boot_nca(&s.keys, &mut s.cpu, container, added)?;
    s.cpu.log(Level::Info, "[io] booted the standalone NCA");
    Ok(entry as f64)
}

/// The index of the Program NCA in the open container, or why there is none (including that
/// no `prod.keys` are loaded). Pass it to `switch_load_nca_from_nsp`.
#[wasm_bindgen]
pub fn switch_program_nca_index() -> Result<u32, String> {
    let s = session();
    let container = container(s)?;
    switch_core::nca::find_nca_by_type(
        &s.nsp_files,
        &container,
        &s.keys,
        switch_core::nca::ContentType::Program,
    )
    .map(|(index, _)| index as u32)
    .ok_or_else(|| {
        "no Program NCA in this container (or its header couldn't be decrypted; load prod.keys)"
            .to_owned()
    })
}

/// Boot NSP file `index` as a Program NCA. Returns the entry address, or fails with the
/// reason.
#[wasm_bindgen]
pub fn switch_load_nca_from_nsp(index: u32) -> Result<f64, String> {
    let s = session();
    let container = container(s)?;
    let nca_src = nsp_file_source(s, index)?;

    // Try the bundled ticket before external title.keys.
    if let Ok(nca) = Nca::parse_source(&nca_src, Some(&s.keys)) {
        let _ = switch_core::ticket::load_bundled_title_key(
            &mut s.keys,
            &nca,
            &s.nsp_files,
            &container,
        );
    }

    let added = Added {
        update: s.update.as_ref(),
        dlc: &s.dlc,
    };
    let entry = load_and_boot_nca(&s.keys, &mut s.cpu, nca_src, added)?;
    let line = format!("[io] booted {}", s.nsp_files[index as usize].name);
    s.cpu.log(Level::Info, &line);
    mount_manual(s, &container);
    Ok(entry as f64)
}

/// Give the booted title the RomFS of its `HtmlDocument` NCA, which its web applets open.
fn mount_manual(s: &mut crate::Session, container: &crate::HostSource) {
    let Some((index, nca)) = switch_core::nca::find_html_document(&s.nsp_files, container, &s.keys)
    else {
        return;
    };
    let f = &s.nsp_files[index];
    let romfs = Window::new(*container, f.offset, f.size, &f.name).and_then(|window| {
        let section = nca
            .romfs_section_index()
            .ok_or_else(|| switch_core::Error::Nca("no RomFS in this Manual".into()))?;
        nca.romfs_source(window, &s.keys, section)
    });
    match romfs {
        Ok(romfs) => s.cpu.set_manual_source(Box::new(romfs)),
        Err(e) => s
            .cpu
            .log(Level::Warn, &format!("the Manual is unreadable: {e}")),
    }
}

/// Load an AArch64 ELF into the CPU. Returns the entry address.
#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
fn load_elf_bytes(s: &mut Session, data: &[u8]) -> Result<u64, String> {
    // An ELF carries no control data.
    s.control = None;
    let elf = switch_core::elf::load_elf(&s.cpu.mem, data).map_err(|e| e.to_string())?;
    s.cpu.set_pc(elf.entry as u32);
    boot_entry_regs(&mut s.cpu, 0);
    s.cpu.log(
        Level::Info,
        &format!("Loaded an ELF, entry {:#010x}", elf.entry),
    );
    crate::log::collect(&mut s.cpu);
    s.cpu.halted = false;
    Ok(elf.entry as u64)
}

/// Reset the integer registers and set the entry convention: `x0 = env`,
/// `x1 = UINT64_MAX` for the homebrew ABI, `x0 = 0` for NSO, `x0 = 0, x1 = 1` otherwise.
#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
fn boot_entry_regs(cpu: &mut Cpu, env_addr: u32) {
    for i in 0..=30u8 {
        cpu.set_reg(i, 0);
    }
    cpu.set_reg(0, env_addr as u64);
    cpu.set_reg(1, if env_addr != 0 { u64::MAX } else { 1 });
    // LR to the exit trampoline, so a returning `main` exits.
    cpu.set_reg(30, switch_core::cpu::SELF_RETURN_TRAMPOLINE as u64);
}
