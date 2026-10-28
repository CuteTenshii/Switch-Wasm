//! Booting homebrew and retail programs.

use crate::cpu::*;
use crate::trace::Level;
use crate::{Error, Result};

impl Cpu {
    /// Map a runtime environment and point SP at a stack, as the loader does before
    /// jumping to a program's entry point. Only hosts booting real homebrew call this.
    pub fn bootstrap(&mut self) {
        self.map_runtime();
        self.regs[SP_SLOT] = STACK_TOP;
        // TLS base, clear of the heap, the stack and the GPU driver's own allocations.
        self.tpidr = u64::from(MAIN_THREAD_TLS_BASE);
    }

    /// The address-space half of [`Cpu::bootstrap`]: the stack and the exit trampolines.
    pub(crate) fn map_runtime(&mut self) {
        // The whole guest space is lazily mapped: reads see zeros, writes allocate on first touch.
        self.mem.soft_map_zero(0, GUEST_SPACE_END);
        let _ = self
            .mem
            .map_zero((STACK_TOP - STACK_SIZE) as u32, STACK_SIZE as usize);
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

    /// Boot a homebrew NRO as HBL does: enter its crt0 with the environment block, which
    /// leaves the heap, `ThreadVars` and the `.init_array` to `__libnx_init`.
    pub fn boot_homebrew(&mut self, data: &[u8]) -> Result<crate::nro::LoadedNro> {
        // Expose the NRO at argv[0] on the SD card for `romfsMountSelf`.
        self.fs
            .write_file(crate::nro::HOMEBREW_NRO_PATH, data.to_vec());
        self.boot_nro(data, crate::nro::HOMEBREW_NRO_PATH)
    }

    /// Boot `data`, the NRO at `path` on the SD card, telling it `sdmc:<path>` as
    /// `argv[0]` as hbmenu does, so it finds the files beside it.
    pub fn boot_homebrew_at(&mut self, data: &[u8], path: &str) -> Result<crate::nro::LoadedNro> {
        let path = path.strip_prefix("sdmc:").unwrap_or(path);
        self.boot_nro(data, &format!("sdmc:{path}"))
    }

    fn boot_nro(&mut self, data: &[u8], argv0: &str) -> Result<crate::nro::LoadedNro> {
        self.mem.clear_modules();
        self.process_mut().module_names.clear();
        let loaded = crate::nro::load_nro(&self.mem, data)?;
        let end = loaded
            .data
            .mem_addr
            .wrapping_add(loaded.data.file_size)
            .wrapping_add(loaded.bss_size);
        self.record_module_name(loaded.base, end, "homebrew");
        if loaded.env_addr != 0 {
            crate::nro::write_argv0(&self.mem, argv0);
        }
        self.out.clear();
        self.trace.clear();
        self.halted = false;
        self.process_mut().guest_fatal = None;
        self.trace_enabled = false;
        for i in 0..=30u8 {
            self.set_reg(i, 0);
        }
        self.set_reg(0, loaded.env_addr as u64);
        self.set_reg(1, if loaded.env_addr != 0 { u64::MAX } else { 1 });
        self.set_reg(30, SELF_RETURN_TRAMPOLINE as u64);

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
        self.process_mut().guest_fatal = None;
        self.trace_enabled = false;
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

        let loaded = self.load_modules(modules)?;
        self.set_pc(loaded[0].entry);
        self.seed_applet_launch_arguments();
        self.seed_launch_parameters();
        Ok(loaded)
    }

    /// Load `modules` back to back from [`crate::nso::NSO_BASE`] into the running process.
    pub(crate) fn load_modules(
        &mut self,
        modules: &[(&str, &[u8])],
    ) -> Result<Vec<crate::nso::LoadedNso>> {
        self.mem.clear_modules();
        self.process_mut().module_names.clear();
        const MODULE_ALIGN: u32 = 0x1000;
        let mut base = crate::nso::NSO_BASE;
        let mut loaded = Vec::with_capacity(modules.len());
        // Name the title, so fault addresses are read against the right binary.
        self.log(
            Level::Info,
            &format!("[loader] program {:#018x}", self.process().program_id),
        );
        for (name, data) in modules {
            let module = crate::nso::load_nso(&self.mem, data, base).map_err(|e| {
                Error::Cpu(format!("loading module {:?} at {:#x}: {}", name, base, e))
            })?;
            let image_end = module
                .data
                .mem_addr
                .wrapping_add(module.data.file_size)
                .wrapping_add(module.bss_size);
            // Where each module landed; `rtld` finds them itself via `svcQueryMemory`.
            self.log(Level::Debug, &format!(
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
        if loaded.is_empty() {
            return Err(Error::Cpu("no modules to boot".into()));
        }
        Ok(loaded)
    }

    /// Seed `am`'s launch parameters as a console's launcher would, including the
    /// `PreselectedUser` that `nn::account::OpenPreselectedUser` requires.
    pub(crate) fn seed_launch_parameters(&mut self) {
        self.process_mut().am_launch_parameters.clear();
        if crate::services::am::is_library_applet(self.process().program_id) {
            return;
        }
        let user = crate::services::am::preselected_user_parameter(self.current_user().uid);
        self.process_mut()
            .am_launch_parameters
            .insert(crate::services::am::LAUNCH_PARAMETER_PRESELECTED_USER, user);
    }

    /// Queue what a library applet's caller would push: `LibAppletCommonArguments`, then
    /// the applet's own launch structs (see [`crate::services::am::applet_launch_storages`]).
    pub(crate) fn seed_applet_launch_arguments(&mut self) {
        self.process_mut().am_in_data.clear();
        self.process_mut().am_out_data.clear();
        self.process_mut().am_interactive_in.clear();
        self.process_mut().am_interactive_out.clear();
        if !crate::services::am::is_library_applet(self.process().program_id) {
            return;
        }
        const COMMON_ARGS_VERSION: u32 = 1;
        const COMMON_ARGS_SIZE: u32 = 0x20;
        let mut args = Vec::with_capacity(COMMON_ARGS_SIZE as usize);
        args.extend_from_slice(&COMMON_ARGS_VERSION.to_le_bytes());
        args.extend_from_slice(&COMMON_ARGS_SIZE.to_le_bytes());
        // LaVersion: the applet's own interface revision.
        args.extend_from_slice(
            &crate::services::am::applet_interface_version(self.process().program_id).to_le_bytes(),
        );
        // ExpectedThemeColor: 0 is the basic white theme.
        args.extend_from_slice(&0u32.to_le_bytes());
        // PlayStartupSound, then padding out to the tick field.
        args.resize(0x18, 0);
        // The tick the caller started the applet at.
        args.extend_from_slice(&0u64.to_le_bytes());
        self.process_mut().am_in_data.push_back(args);
        // Then the applet's own launch structs.
        let user = self.current_user().uid;
        for storage in crate::services::am::applet_launch_storages(self.process().program_id, user)
        {
            self.process_mut().am_in_data.push_back(storage);
        }
    }
}
