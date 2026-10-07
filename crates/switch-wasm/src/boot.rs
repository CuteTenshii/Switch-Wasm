//! Loading and booting programs.

use switch_core::cpu::Cpu;
use switch_core::elf::load_elf;
use switch_core::nca::Nca;
use switch_core::nsp::Pfs0;
use switch_core::source::ByteSource;
use switch_core::trace::Level;

use crate::{container, nsp_file_source, session, Added, Dlc};

/// Load an NRO homebrew image into the CPU. Returns entry address or -1.
#[no_mangle]
pub extern "C" fn switch_load_nro(handle: u32, ptr: *const u8, len: u32) -> i64 {
    let s = session(handle);
    let data = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    s.control = None;
    match s.cpu.boot_homebrew(data) {
        Ok(loaded) => {
            // Cached for display only: homebrew runs in another title's process.
            s.control = switch_core::control::Control::from_nro(data);
            s.cpu.out.clear();
            s.cpu.trace.clear();
            s.cpu.halted = false;
            s.last_error.clear();
            loaded.entry as i64
        }
        Err(e) => {
            s.last_error = e.to_string();
            -1
        }
    }
}

/// ExeFS modules in load order (`rtld`, `main`, `subsdk0..9`, `sdk`), skipping absent ones.
fn collect_modules<'a>(pfs0: &Pfs0, exefs: &'a [u8]) -> Vec<(&'static str, &'a [u8])> {
    const MODULE_ORDER: &[&str] = &[
        "rtld", "main", "subsdk0", "subsdk1", "subsdk2", "subsdk3", "subsdk4", "subsdk5",
        "subsdk6", "subsdk7", "subsdk8", "subsdk9", "sdk",
    ];
    MODULE_ORDER
        .iter()
        .filter_map(|&name| {
            let f = pfs0.find(name)?;
            let start = f.offset as usize;
            let end = start + f.size as usize; // Pfs0::parse already bounds-checked every entry
            Some((name, &exefs[start..end]))
        })
        .collect()
}

/// Decrypt a Program NCA's ExeFS from `nca_src`, load it and boot. Returns entry or -1.
/// `nca_src` stays alive as the title's RomFS source. `added` holds the update and DLC.
pub(crate) fn load_and_boot_nca<S: ByteSource + 'static>(
    keys: &switch_core::keys::KeySet,
    cpu: &mut Cpu,
    last_error: &mut String,
    nca_src: S,
    added: Added<'_>,
) -> i64 {
    let nca = match Nca::parse_source(&nca_src, Some(keys)) {
        Ok(nca) => nca,
        Err(e) => {
            *last_error = e.to_string();
            return -1;
        }
    };
    // Refuse an update for a different title.
    let update = match added.update {
        Some(u) if u.nca.program_id == nca.program_id => Some(u),
        Some(u) => {
            *last_error = format!(
                "the update added to this session is for title {:016x}, but this container is {:016x}",
                u.nca.program_id, nca.program_id
            );
            return -1;
        }
        None => None,
    };
    // An update's ExeFS is a complete replacement set of modules.
    let program = update.map_or(&nca, |u| &u.nca);
    let exefs_index = match program.exefs_section_index() {
        Some(i) => i,
        None => {
            *last_error = "no ExeFS (PFS0) section in this NCA".into();
            return -1;
        }
    };
    let exefs = match update {
        Some(u) => u
            .program_window()
            .and_then(|window| program.read_pfs0_section(window, keys, exefs_index)),
        None => program.read_pfs0_section(&nca_src, keys, exefs_index),
    };
    let exefs = match exefs {
        Ok(v) => v,
        Err(e) => {
            *last_error = e.to_string();
            return -1;
        }
    };
    if update.is_some() {
        cpu.diagnostic(
            Level::Info,
            &format!(
                "[update] booting the update's modules for {:016x}, over this container's RomFS",
                nca.program_id
            ),
        );
    }
    // Report whether the ExeFS hash coverage was checked.
    match program.pfs0_hash_coverage(exefs_index) {
        Some((block, blocks)) => cpu.diagnostic(
            Level::Info,
            &format!(
                "[exefs] {:#x} bytes, {} blocks of {:#x} verified against the section hash table",
                exefs.len(),
                blocks,
                block
            ),
        ),
        None => cpu.diagnostic(
            Level::Warn,
            &format!(
                "[exefs] {:#x} bytes; hash table geometry unrecognised, contents NOT verified",
                exefs.len()
            ),
        ),
    }

    let pfs0 = match Pfs0::parse(&exefs) {
        Ok(p) => p,
        Err(e) => {
            *last_error = e.to_string();
            return -1;
        }
    };
    // Reported by `pm`; applets derive their `AppletId` from it.
    cpu.set_program_id(program.program_id);

    let modules = collect_modules(&pfs0, &exefs);
    // Report what the ExeFS holds next to what was loaded.
    cpu.diagnostic(
        Level::Info,
        &format!(
            "[exefs] entries: {}, loading: {}",
            pfs0.files
                .iter()
                .map(|f| f.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            modules
                .iter()
                .map(|(name, _)| *name)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    );
    if !modules.iter().any(|(name, _)| *name == "main") {
        *last_error = "no 'main' executable in this NCA's ExeFS".into();
        return -1;
    }

    // RomFS is optional and failures do not block booting; it is read by range.
    match update {
        // An update's RomFS is a patch, readable only over the base's.
        Some(u) => {
            let patched = u.program_window().and_then(|window| {
                switch_core::bktr::patched_romfs_source(&u.nca, window, &nca, nca_src, keys)
            });
            match patched {
                Ok(romfs) => cpu.set_romfs_source(Box::new(romfs)),
                Err(e) => cpu.diagnostic(
                    Level::Error,
                    &format!("the update's RomFS is unreadable: {}", e),
                ),
            }
        }
        None => {
            if let Some(romfs_index) = nca.romfs_section_index() {
                match nca.romfs_source(nca_src, keys, romfs_index) {
                    Ok(romfs) => cpu.set_romfs_source(Box::new(romfs)),
                    Err(e) => cpu.diagnostic(Level::Error, &format!("romfs unavailable: {}", e)),
                }
            }
        }
    }

    // The address space layout from the NPDM system resource size; must precede the boot.
    let system_resource = switch_core::npdm::Npdm::system_resource_size_of(&pfs0, &exefs);
    cpu.diagnostic(
        Level::Info,
        &format!(
            "[npdm] system resource {system_resource:#x}: {}",
            if system_resource == 0 {
                "plain heap"
            } else {
                "virtual address memory"
            }
        ),
    );
    cpu.set_system_resource_size(system_resource);

    // Main thread priority.
    if let Some(priority) = switch_core::npdm::Npdm::main_thread_priority_of(&pfs0, &exefs) {
        cpu.set_main_thread_priority(priority);
    }
    // Main thread core.
    if let Some(core) = switch_core::npdm::Npdm::main_thread_core_of(&pfs0, &exefs) {
        cpu.set_main_thread_core(core);
    }
    // Allowed cores.
    if let Some(mask) = switch_core::npdm::Npdm::core_mask_of(&pfs0, &exefs) {
        cpu.set_process_core_mask(mask);
    }

    // 32- or 64-bit, from the NPDM flags; the entry ABI differs.
    if !switch_core::npdm::Npdm::is_64_bit_of(&pfs0, &exefs) {
        cpu.diagnostic(
            Level::Info,
            "[npdm] AArch32 title: running the A32 interpreter",
        );
        cpu.set_mode(switch_core::cpu::ExecMode::A32);
    }

    match cpu.boot_retail_program(&modules) {
        Ok(loaded) => {
            // After the modules: booting clears the diagnostic buffer.
            mount_add_on_content(cpu, keys, added.dlc);
            last_error.clear();
            loaded[0].entry as i64
        }
        Err(e) => {
            *last_error = e.to_string();
            -1
        }
    }
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
                    Some(index) => cpu.diagnostic(
                        Level::Info,
                        &format!(
                            "[aoc] {:016x} mounted as add-on content {index}, {size:#x} bytes",
                            entry.content_id
                        ),
                    ),
                    None => cpu.diagnostic(
                        Level::Warn,
                        &format!(
                            "[aoc] {:016x} is not this title's add-on content; not mounted",
                            entry.content_id
                        ),
                    ),
                }
            }
            Err(e) => cpu.diagnostic(
                Level::Error,
                &format!("[aoc] {:016x} could not be read: {e}", entry.content_id),
            ),
        }
    }
}

/// Boot the open container as a standalone Program NCA. Returns entry or -1;
/// check `switch_last_error`, since 0 can be a valid entry.
#[no_mangle]
pub extern "C" fn switch_load_nca(handle: u32) -> i64 {
    let s = session(handle);
    let Some(container) = container(s) else {
        return -1;
    };
    let added = Added {
        update: s.update.as_ref(),
        dlc: &s.dlc,
    };
    load_and_boot_nca(&s.keys, &mut s.cpu, &mut s.last_error, container, added)
}

/// The index of the Program NCA in the open container, or -1 (including when no
/// `prod.keys` are loaded). Pass it to `switch_load_nca_from_nsp`.
#[no_mangle]
pub extern "C" fn switch_program_nca_index(handle: u32) -> i32 {
    let s = session(handle);
    let Some(container) = container(s) else {
        return -1;
    };
    let found = switch_core::nca::find_nca_by_type(
        &s.nsp_files,
        &container,
        &s.keys,
        switch_core::nca::ContentType::Program,
    );
    match found {
        Some((index, _)) => {
            s.last_error.clear();
            index as i32
        }
        None => {
            s.last_error =
                "no Program NCA in this container (or its header couldn't be decrypted; load prod.keys)"
                    .into();
            -1
        }
    }
}

/// Boot NSP file `index` as a Program NCA. Returns entry or -1; check
/// `switch_last_error`, since 0 can be a valid entry.
#[no_mangle]
pub extern "C" fn switch_load_nca_from_nsp(handle: u32, index: u32) -> i64 {
    let s = session(handle);
    let Some(container) = container(s) else {
        return -1;
    };
    let Some(nca_src) = nsp_file_source(s, index) else {
        return -1;
    };

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
    load_and_boot_nca(&s.keys, &mut s.cpu, &mut s.last_error, nca_src, added)
}

/// Load an AArch64 ELF into the CPU. Returns entry address or -1.
#[no_mangle]
pub extern "C" fn switch_load_elf(handle: u32, ptr: *const u8, len: u32) -> i64 {
    let s = session(handle);
    let data = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    // An ELF carries no control data.
    s.control = None;
    match load_elf(&mut s.cpu.mem, data) {
        Ok(elf) => {
            s.cpu.set_pc(elf.entry as u32);
            boot_entry_regs(&mut s.cpu, 0);
            s.cpu.out.clear();
            s.cpu.trace.clear();
            s.cpu.halted = false;
            s.last_error.clear();
            elf.entry as i64
        }
        Err(e) => {
            s.last_error = e.to_string();
            -1
        }
    }
}

/// Reset the integer registers and set the entry convention: `x0 = env`,
/// `x1 = UINT64_MAX` for the homebrew ABI, `x0 = 0` for NSO, `x0 = 0, x1 = 1` otherwise.
fn boot_entry_regs(cpu: &mut Cpu, env_addr: u32) {
    for i in 0..=30u8 {
        cpu.set_reg(i, 0);
    }
    cpu.set_reg(0, env_addr as u64);
    cpu.set_reg(1, if env_addr != 0 { u64::MAX } else { 1 });
    // LR to the exit trampoline, so a returning `main` exits.
    cpu.set_reg(30, switch_core::cpu::SELF_RETURN_TRAMPOLINE as u64);
}
