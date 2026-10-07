//! `ldr:ro`: run-time NRO loading into [`crate::cpu::RO_MODULE_REGION_ADDR`].

use crate::cpu::Cpu;
use crate::trace::Level;
use crate::Result;

const RO_RESULT_MODULE: u32 = 22;

/// One NRO `ldr:ro` has mapped into the process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RoModule {
    /// The caller's NRO buffer, so an unload naming the source still finds the module.
    source: u32,
    /// The image at `base` followed by its zero-filled BSS, `size` bytes in total.
    base: u32,
    size: u32,
    text: (u32, u32),
}

impl Cpu {
    /// `ldr:ro` (`nn::ro::detail::IRoInterface`). `LoadModule` maps the caller's
    /// NRO; relocation is the caller's job. NRR registrations are recorded but not
    /// verified.
    pub(crate) fn ldr_ro_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "ldr:ro");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, "ldr:ro-control", cmd_id),
            };
        }
        // The pid placeholder shifts each argument one word in.
        let data = self.ipc_request_data(tls);
        let mut args = [0u64; 4];
        for (index, arg) in args.iter_mut().enumerate() {
            *arg = self
                .mem
                .read_u64(data.wrapping_add(8 * (index as u32 + 1)))
                .unwrap_or(0);
        }
        match cmd_id {
            // LoadModule(pid, nro_address, nro_size, bss_address, bss_size) -> u64 address.
            Some(0) => self.ldr_ro_load_module(tls, args[0], args[1], args[2], args[3]),
            // UnloadModule(pid, address).
            Some(1) => self.ldr_ro_unload_module(tls, args[0]),
            // RegisterModuleInfo(pid, nrr_address, nrr_size), and 7.0.0+ RegisterProcessModuleInfo.
            Some(2) | Some(10) => self.ldr_ro_register_module_info(tls, args[0], args[1]),
            // UnregisterModuleInfo(pid, nrr_address).
            Some(3) => {
                const NOT_REGISTERED: u32 = RO_RESULT_MODULE | (1029 << 9);
                let nrr_address = args[0] as u32;
                match self.ro_registrations.remove(&nrr_address) {
                    Some(_) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                    None => self.write_ipc_response(tls, NOT_REGISTERED, &[], &[], &[]),
                }
            }
            // RegisterProcessHandle [3.0.0+]: `nn::ro::Initialize`'s first call.
            Some(4) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            _ => self.unimplemented_command(tls, "ldr:ro", cmd_id),
        }
    }

    /// Map a copy of the caller's NRO and return its address. Unlike a real
    /// kernel this is a copy, not an alias; guests never write the source after.
    fn ldr_ro_load_module(
        &mut self,
        tls: u32,
        nro_address: u64,
        nro_size: u64,
        bss_address: u64,
        bss_size: u64,
    ) -> Result<()> {
        const OUT_OF_ADDRESS_SPACE: u32 = RO_RESULT_MODULE | (2 << 9);
        const INVALID_NRO: u32 = RO_RESULT_MODULE | (4 << 9);
        const INVALID_ADDRESS: u32 = RO_RESULT_MODULE | (1025 << 9);
        const INVALID_SIZE: u32 = RO_RESULT_MODULE | (1026 << 9);

        // Reject rather than truncate addresses beyond 32 bits.
        if nro_address > u64::from(u32::MAX) || bss_address > u64::from(u32::MAX) {
            return self.write_ipc_response(tls, INVALID_ADDRESS, &[], &[], &[]);
        }
        if !Self::ro_is_page_aligned(nro_address) || !Self::ro_is_page_aligned(bss_address) {
            return self.write_ipc_response(tls, INVALID_ADDRESS, &[], &[], &[]);
        }
        // Also keeps a nonsense size from becoming a huge host allocation.
        let too_big = u64::from(crate::cpu::RO_MODULE_REGION_SIZE);
        if nro_size == 0
            || !Self::ro_is_page_aligned(nro_size)
            || !Self::ro_is_page_aligned(bss_size)
            || nro_size > too_big
            || bss_size > too_big
        {
            return self.write_ipc_response(tls, INVALID_SIZE, &[], &[], &[]);
        }

        let image = self.read_bytes(nro_address as u32, nro_size as u32);
        let header = match crate::nro::NroHeader::parse(&image) {
            Ok(header) => header,
            Err(e) => {
                self.diagnostic(
                    Level::Warn,
                    &format!("[ro] refusing the module at {nro_address:#010x}: {e}"),
                );
                return self.write_ipc_response(tls, INVALID_NRO, &[], &[], &[]);
            }
        };
        // A short BSS would spill the module's data onto the next mapping.
        if bss_size < u64::from(header.bss_size) {
            self.diagnostic(
                Level::Warn,
                &format!(
                    "[ro] refusing the module at {nro_address:#010x}: it needs {:#x} bytes of bss \
                 and the caller supplied {bss_size:#x}",
                    header.bss_size
                ),
            );
            return self.write_ipc_response(tls, INVALID_SIZE, &[], &[], &[]);
        }

        let size = (nro_size + bss_size) as u32;
        let Some(base) = self.ro_free_region(size) else {
            self.diagnostic(
                Level::Warn,
                &format!(
                    "[ro] no room for a {size:#x}-byte module: {} already mapped",
                    self.ro_modules.len()
                ),
            );
            return self.write_ipc_response(tls, OUT_OF_ADDRESS_SPACE, &[], &[], &[]);
        };

        // One write so the BSS is zero, not leftovers from an unloaded module.
        let mut mapped = image;
        mapped.resize(size as usize, 0);
        self.mem.map(base, &mapped)?;
        let text = (
            base.wrapping_add(header.text_offset),
            base.wrapping_add(header.text_offset)
                .wrapping_add(header.text_size),
        );
        // `.text` is read-execute; `UnloadModule` must undo this.
        self.mem.mark_readonly(text.0, text.1);
        // Run-time modules carry the `Alias*` memory states; see `Memory::mark_module`.
        self.mem.mark_module(
            (text.0, base.wrapping_add(header.data_offset)),
            (
                base.wrapping_add(header.data_offset),
                base.wrapping_add(size),
            ),
            true,
        );
        self.record_module_name(
            base,
            base.wrapping_add(size),
            &format!("ro@{nro_address:#x}"),
        );
        self.ro_modules.insert(
            base,
            RoModule {
                source: nro_address as u32,
                base,
                size,
                text,
            },
        );
        self.diagnostic(
            Level::Info,
            &format!(
                "[ro] mapped the module at {nro_address:#010x} to {base:#010x}: text \
             {:#010x}..{:#010x}, rodata {:#010x}..{:#010x}, data {:#010x}..{:#010x}, bss \
             {:#010x}..{:#010x}",
                text.0,
                text.1,
                base.wrapping_add(header.ro_offset),
                base.wrapping_add(header.ro_offset)
                    .wrapping_add(header.ro_size),
                base.wrapping_add(header.data_offset),
                base.wrapping_add(header.data_offset)
                    .wrapping_add(header.data_size),
                base.wrapping_add(nro_size as u32),
                base.wrapping_add(size),
            ),
        );
        self.write_ipc_response(tls, 0, &[], &u64::from(base).to_le_bytes(), &[])
    }

    /// Unmap a module by the address `LoadModule` returned, or by its source buffer.
    fn ldr_ro_unload_module(&mut self, tls: u32, address: u64) -> Result<()> {
        const NOT_LOADED: u32 = RO_RESULT_MODULE | (1028 << 9);
        let address = address as u32;
        let base = if self.ro_modules.contains_key(&address) {
            Some(address)
        } else {
            self.ro_modules
                .values()
                .find(|m| m.source == address)
                .map(|m| m.base)
        };
        let Some(module) = base.and_then(|base| self.ro_modules.remove(&base)) else {
            return self.write_ipc_response(tls, NOT_LOADED, &[], &[], &[]);
        };
        self.mem.unmark_readonly(module.text.0, module.text.1);
        self.mem
            .unmark_module(module.base, module.base.wrapping_add(module.size));
        self.mem.unmap(module.base, module.size as usize);
        self.forget_module_name(module.base);
        self.diagnostic(
            Level::Info,
            &format!(
                "[ro] unmapped the module at {:#010x} ({:#x} bytes)",
                module.base, module.size
            ),
        );
        self.write_ipc_response(tls, 0, &[], &[], &[])
    }

    /// Record an NRR registration. Only the magic is checked; the signature
    /// chain cannot be verified without console keys.
    fn ldr_ro_register_module_info(
        &mut self,
        tls: u32,
        nrr_address: u64,
        nrr_size: u64,
    ) -> Result<()> {
        /// "NRR0".
        const NRR0_MAGIC: u32 = 0x3052_524E;
        const INVALID_NRR: u32 = RO_RESULT_MODULE | (6 << 9);
        const INVALID_ADDRESS: u32 = RO_RESULT_MODULE | (1025 << 9);
        const INVALID_SIZE: u32 = RO_RESULT_MODULE | (1026 << 9);

        if nrr_address > u64::from(u32::MAX) || !Self::ro_is_page_aligned(nrr_address) {
            return self.write_ipc_response(tls, INVALID_ADDRESS, &[], &[], &[]);
        }
        if nrr_size == 0 || !Self::ro_is_page_aligned(nrr_size) || nrr_size > u64::from(u32::MAX) {
            return self.write_ipc_response(tls, INVALID_SIZE, &[], &[], &[]);
        }
        if self.mem.read_u32(nrr_address as u32).unwrap_or(0) != NRR0_MAGIC {
            self.diagnostic(Level::Warn, &format!("[ro] no NRR at {nrr_address:#010x}"));
            return self.write_ipc_response(tls, INVALID_NRR, &[], &[], &[]);
        }
        self.ro_registrations
            .insert(nrr_address as u32, nrr_size as u32);
        self.write_ipc_response(tls, 0, &[], &[], &[])
    }

    /// First fit over live mappings, so load/unload cycles do not exhaust the region.
    fn ro_free_region(&self, size: u32) -> Option<u32> {
        let region_end =
            crate::cpu::RO_MODULE_REGION_ADDR.wrapping_add(crate::cpu::RO_MODULE_REGION_SIZE);
        let mut candidate = crate::cpu::RO_MODULE_REGION_ADDR;
        for module in self.ro_modules.values() {
            if size <= module.base.saturating_sub(candidate) {
                return Some(candidate);
            }
            candidate = candidate.max(module.base.wrapping_add(module.size));
        }
        (size <= region_end.saturating_sub(candidate)).then_some(candidate)
    }

    fn ro_is_page_aligned(value: u64) -> bool {
        value.is_multiple_of(crate::mem::PAGE_SIZE as u64)
    }
}
