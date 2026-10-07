//! Booting homebrew and retail programs.

use crate::cpu::*;
use crate::trace::Level;
use crate::{Error, Result};

impl Cpu {
    /// Map a runtime environment and point SP at a stack, as the loader does before
    /// jumping to a program's entry point. Only hosts booting real homebrew call this.
    pub fn bootstrap(&mut self) {
        // The whole guest space is lazily mapped: reads see zeros, writes allocate on first touch.
        self.mem.soft_map_zero(0, GUEST_SPACE_END);
        let _ = self
            .mem
            .map_zero((STACK_TOP - STACK_SIZE) as u32, STACK_SIZE as usize);
        self.regs[SP_SLOT] = STACK_TOP;
        // TLS base, clear of the heap, the stack and the GPU driver's own allocations.
        self.tpidr = u64::from(MAIN_THREAD_TLS_BASE);
        // LR points at a stub that calls ExitProcess (svc 0x07), so returning from main exits cleanly.
        let _ = self.mem.map_zero(SELF_RETURN_TRAMPOLINE, 0x10);
        self.mem.write_u32(SELF_RETURN_TRAMPOLINE, 0xD400_00E1).ok(); // svc #7
        self.mem
            .write_u32(SELF_RETURN_TRAMPOLINE + 4, 0x1400_0000)
            .ok(); // b .
                   // Returning from a thread entry point is `svcExitThread` (svc 0x0A).
        let _ = self.mem.map_zero(THREAD_EXIT_TRAMPOLINE, 0x10);
        self.mem.write_u32(THREAD_EXIT_TRAMPOLINE, 0xD400_0141).ok(); // svc #0xa
        self.mem
            .write_u32(THREAD_EXIT_TRAMPOLINE + 4, 0x1400_0000)
            .ok(); // b .
    }

    /// Boot a homebrew NRO as HBL does: run the crt0 up to `main`, then the `.init_array`
    /// and main `ThreadVars` setup the skipped `__libnx_init` would provide.
    pub fn boot_homebrew(&mut self, data: &[u8]) -> Result<crate::nro::LoadedNro> {
        self.mem.clear_modules();
        self.module_names.clear();
        let loaded = crate::nro::load_nro(&mut self.mem, data)?;
        let end = loaded
            .data
            .mem_addr
            .wrapping_add(loaded.data.file_size)
            .wrapping_add(loaded.bss_size);
        self.record_module_name(loaded.base, end, "homebrew");
        // Expose the NRO at argv[0] on the SD card for `romfsMountSelf`.
        self.fs
            .write_file(crate::nro::HOMEBREW_NRO_PATH, data.to_vec());
        self.out.clear();
        self.trace.clear();
        self.halted = false;
        self.guest_fatal = None;
        self.trace_enabled = false;
        for i in 0..=30u8 {
            self.set_reg(i, 0);
        }
        self.set_reg(0, loaded.env_addr as u64);
        self.set_reg(1, if loaded.env_addr != 0 { u64::MAX } else { 1 });
        self.set_reg(30, SELF_RETURN_TRAMPOLINE as u64);

        let init = crate::nro::init_array_entries(data);
        if !init.is_empty() && loaded.env_addr != 0 {
            // The crt0 calls main at entry+0xc0; BSS is zeroed and relocations applied by then.
            let main_call = loaded.entry.wrapping_add(0xc0);
            let main_insn = self.mem.fetch(main_call).ok();
            let is_bl = matches!(main_insn, Some(i) if (i & 0xFC00_0000) == 0x9400_0000);
            if is_bl {
                self.set_pc(loaded.entry);
                for _ in 0..5_000_000u64 {
                    if self.halted || self.get_pc() == main_call {
                        break;
                    }
                    self.step()?;
                }
                // ThreadVars at TLS+0x1E0: magic, handle, thread_ptr, _REENT (zeroed; lazily set up), tls_tp.
                const TV_MAGIC: u32 = 0x2154_5624; // "!TV$"
                const REENT_ADDR: u32 = 0x1FF1_0000;
                let tls = self.tls_base();
                let _ = self.mem.map_zero(REENT_ADDR, 0x400);
                let _ = self.mem.write_u32(tls + 0x1E0, TV_MAGIC);
                let _ = self.mem.write_u32(tls + 0x1E4, 0x100);
                let _ = self.mem.write_u32(tls + 0x1E8, 0);
                let _ = self.mem.write_u32(tls + 0x1F0, REENT_ADDR);
                let _ = self.mem.write_u32(tls + 0x1F8, tls);
                // Run the constructors; each returns via x30.
                const SENTINEL: u32 = 0x1FF0_0000;
                for &entry in &init {
                    if self.halted {
                        break;
                    }
                    for i in 0..=29u8 {
                        self.set_reg(i, 0);
                    }
                    self.set_reg(30, SENTINEL as u64);
                    self.set_pc(entry);
                    for _ in 0..20_000_000u64 {
                        if self.halted || self.get_pc() == SENTINEL {
                            break;
                        }
                        self.step()?;
                    }
                }
                // Restore the entry registers and resume at the crt0's call; x2 is the loader's
                // return address, which `__nx_exit` jumps to.
                for i in 0..=30u8 {
                    self.set_reg(i, 0);
                }
                self.set_reg(0, loaded.env_addr as u64);
                self.set_reg(1, if loaded.env_addr != 0 { u64::MAX } else { 1 });
                self.set_reg(2, SELF_RETURN_TRAMPOLINE as u64);
                self.set_reg(30, SELF_RETURN_TRAMPOLINE as u64);
                self.set_pc(main_call);
                return Ok(loaded);
            }
        }
        self.set_pc(loaded.entry);
        Ok(loaded)
    }

    /// Boot a retail title's modules (`rtld`, `main`, `subsdk*`, `sdk`, in that order)
    /// back to back in one address space and enter `rtld`, which relocates the rest.
    pub fn boot_retail_program(
        &mut self,
        modules: &[(&str, &[u8])],
    ) -> Result<Vec<crate::nso::LoadedNso>> {
        self.out.clear();
        self.trace.clear();
        self.halted = false;
        self.guest_fatal = None;
        self.trace_enabled = false;
        self.mem.clear_modules();
        self.module_names.clear();
        for i in 0..=30u8 {
            self.set_reg(i, 0);
        }
        // Horizon's entry ABI: X0 is 0 for a normal launch, X1 the main thread handle
        // (`nnSdk` compares `SdkMutex` lock words against it).
        self.set_reg(1, MAIN_THREAD_HANDLE);
        if self.mode == ExecMode::A32 {
            // A32 keeps SP in r13; restore it after clearing the registers.
            self.regs[13] = self.regs[SP_SLOT];
            self.regs[14] = SELF_RETURN_TRAMPOLINE as u64;
        } else {
            self.set_reg(30, SELF_RETURN_TRAMPOLINE as u64);
        }

        const MODULE_ALIGN: u32 = 0x1000;
        let mut base = crate::nso::NSO_BASE;
        let mut loaded = Vec::with_capacity(modules.len());
        // Name the title, so fault addresses are read against the right binary.
        self.diagnostic(
            Level::Info,
            &format!("[loader] program {:#018x}", self.program_id),
        );
        for (name, data) in modules {
            let module = crate::nso::load_nso(&mut self.mem, data, base).map_err(|e| {
                Error::Cpu(format!("loading module {:?} at {:#x}: {}", name, base, e))
            })?;
            let image_end = module
                .data
                .mem_addr
                .wrapping_add(module.data.file_size)
                .wrapping_add(module.bss_size);
            // Where each module landed; `rtld` finds them itself via `svcQueryMemory`.
            self.diagnostic(Level::Info, &format!(
                "[loader] {} at {:#010x}: text {:#010x}..{:#010x}, rodata {:#010x}..{:#010x}, data {:#010x}..{:#010x}, bss {:#010x}..{:#010x}",
                name,
                module.base,
                module.text.mem_addr,
                module.text.mem_addr.wrapping_add(module.text.file_size),
                module.ro.mem_addr,
                module.ro.mem_addr.wrapping_add(module.ro.file_size),
                module.data.mem_addr,
                module.data.mem_addr.wrapping_add(module.data.file_size),
                module.data.mem_addr.wrapping_add(module.data.file_size),
                image_end,
            ));
            self.record_module_name(module.base, image_end, name);
            base = image_end.wrapping_add(MODULE_ALIGN - 1) & !(MODULE_ALIGN - 1);
            loaded.push(module);
        }
        let entry = loaded
            .first()
            .ok_or_else(|| Error::Cpu("no modules to boot".into()))?
            .entry;
        self.set_pc(entry);
        self.seed_applet_launch_arguments();
        self.seed_launch_parameters();
        Ok(loaded)
    }

    /// Seed `am`'s launch parameters as a console's launcher would, including the
    /// `PreselectedUser` that `nn::account::OpenPreselectedUser` requires.
    pub(crate) fn seed_launch_parameters(&mut self) {
        self.am_launch_parameters.clear();
        if crate::services::am::is_library_applet(self.program_id) {
            return;
        }
        self.am_launch_parameters.insert(
            crate::services::am::LAUNCH_PARAMETER_PRESELECTED_USER,
            crate::services::am::preselected_user_parameter(self.current_user().uid),
        );
    }

    /// Queue what a library applet's caller would push: `LibAppletCommonArguments`, then
    /// the applet's own launch structs (see [`crate::services::am::applet_launch_storages`]).
    pub(crate) fn seed_applet_launch_arguments(&mut self) {
        self.am_in_data.clear();
        self.am_out_data.clear();
        self.am_interactive_in.clear();
        self.am_interactive_out.clear();
        if !crate::services::am::is_library_applet(self.program_id) {
            return;
        }
        const COMMON_ARGS_VERSION: u32 = 1;
        const COMMON_ARGS_SIZE: u32 = 0x20;
        let mut args = Vec::with_capacity(COMMON_ARGS_SIZE as usize);
        args.extend_from_slice(&COMMON_ARGS_VERSION.to_le_bytes());
        args.extend_from_slice(&COMMON_ARGS_SIZE.to_le_bytes());
        // LaVersion: the applet's own interface revision.
        args.extend_from_slice(
            &crate::services::am::applet_interface_version(self.program_id).to_le_bytes(),
        );
        // ExpectedThemeColor: 0 is the basic white theme.
        args.extend_from_slice(&0u32.to_le_bytes());
        // PlayStartupSound, then padding out to the tick field.
        args.resize(0x18, 0);
        // The tick the caller started the applet at.
        args.extend_from_slice(&0u64.to_le_bytes());
        self.am_in_data.push_back(args);
        // Then the applet's own launch structs.
        let user = self.current_user().uid;
        for storage in crate::services::am::applet_launch_storages(self.program_id, user) {
            self.am_in_data.push_back(storage);
        }
    }
}
