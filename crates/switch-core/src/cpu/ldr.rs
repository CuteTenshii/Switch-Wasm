//! `ldr:ro`: run-time module loading.
//!
//! A title that links against an NRO maps it here rather than at startup, so
//! this is a real loader — it relocates into [`super::RO_MODULE_REGION_ADDR`]
//! and hands back where it landed.

use super::Cpu;
use crate::trace::Level;
use crate::Result;

/// The module number every result `ro` reports carries. A caller that acts on
/// a failure at all switches on the description beside it, and a description
/// under the wrong module names a different service's error entirely.
const RO_RESULT_MODULE: u32 = 22;

/// One NRO that `ldr:ro` has mapped into the process. See
/// [`Cpu::ldr_ro_request`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RoModule {
    /// Where the caller's own copy of the NRO file lives — the address it
    /// passed to `LoadModule`, kept so an unload naming the source rather than
    /// the mapping still finds the module.
    source: u32,
    /// The mapping: the image at `base`, its zero-filled BSS behind it, `size`
    /// bytes in total.
    base: u32,
    size: u32,
    /// The `.text` range, read-only for as long as the module is mapped.
    text: (u32, u32),
}

impl Cpu {
    /// `ldr:ro` — `nn::ro::detail::IRoInterface`, the half of dynamic module
    /// loading that cannot happen inside the process.
    ///
    /// A title that loads code at run time (`nn::ro::LoadModule`, and libnx's
    /// `ldrRoLoadNro` under `roDlopen`) holds the NRO file in its own memory
    /// and asks this service to **map** it: a copy of the image at an address
    /// the service picks, `.text` executable and unwritable, with the caller's
    /// zero-filled BSS directly behind it. Everything after that —
    /// relocations, symbol resolution, the module list — is the caller's own
    /// work, done against the address returned here. So `LoadModule` is the
    /// one command that has to do something real, and what it does is the
    /// mapping.
    ///
    /// The rest of the interface is authorization. An NRR carries signed
    /// hashes of the NROs a title is permitted to load, and
    /// `RegisterModuleInfo` is the caller presenting one; `ro` checks the
    /// signature chain against a key a console has and this emulator does not.
    /// A registration is therefore accepted rather than verified — but it is
    /// *recorded*, so unregistering one is not a blind success and a caller
    /// that never registered anything is visible in the state rather than
    /// indistinguishable from one that did.
    ///
    /// Nothing implemented this at all, which is how it announced itself:
    /// `RegisterProcessHandle` is the first call `nn::ro::Initialize` makes,
    /// and the fallback answered it with a fabricated object id. Had the title
    /// got as far as `LoadModule`, that same fallback would have handed it an
    /// object id to jump to.
    pub(super) fn ldr_ro_request(
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
        // Every command on this interface opens with the `u64` placeholder the
        // pid descriptor really fills in, so each argument below sits one word
        // further in than its position in the signature suggests.
        let data = self.ipc_request_data(tls);
        let mut args = [0u64; 4];
        for (index, arg) in args.iter_mut().enumerate() {
            *arg = self
                .mem
                .read_u64(data.wrapping_add(8 * (index as u32 + 1)))
                .unwrap_or(0);
        }
        match cmd_id {
            // LoadModule(pid, nro_address, nro_size, bss_address, bss_size)
            // -> u64 mapped address. Called `LoadNro` before 3.0.0.
            Some(0) => self.ldr_ro_load_module(tls, args[0], args[1], args[2], args[3]),
            // UnloadModule(pid, address).
            Some(1) => self.ldr_ro_unload_module(tls, args[0]),
            // RegisterModuleInfo(pid, nrr_address, nrr_size), and the 7.0.0+
            // `RegisterProcessModuleInfo`, which is the same call with the
            // process handle passed explicitly instead of through the pid
            // descriptor. One process here, so they are the same work.
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
            // RegisterProcessHandle(pid, process handle) [3.0.0+]: the caller
            // telling `ro` which process the modules it is about to load
            // belong to. There is one process here and every module maps into
            // it, so the handle names something already known — but the call
            // is `nn::ro::Initialize`'s first, and refusing it stops a title
            // before it loads anything.
            Some(4) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            _ => self.unimplemented_command(tls, "ldr:ro", cmd_id),
        }
    }

    /// `IRoInterface::LoadModule`: map the NRO the caller is holding, and hand
    /// back the address it now lives at.
    ///
    /// The mapping is a **copy**, not the alias a real kernel makes. Page
    /// storage here is not shareable — [`crate::mem::Memory::copy_range`] has
    /// the same constraint for `svcMapMemory` — so writes through the caller's
    /// original buffer do not reach the loaded module. That is a difference a
    /// guest could observe, and nothing does: the source buffer is a file
    /// image the caller read and stops touching, and every write that matters
    /// (the relocations `nn::ro` applies) goes to the returned address.
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

        // A `u64` argument in a 32-bit address space: anything that does not
        // fit is not an address the guest can have meant, and truncating it
        // would map a module over whatever lives at the bottom 32 bits.
        if nro_address > u64::from(u32::MAX) || bss_address > u64::from(u32::MAX) {
            return self.write_ipc_response(tls, INVALID_ADDRESS, &[], &[], &[]);
        }
        if !Self::ro_is_page_aligned(nro_address) || !Self::ro_is_page_aligned(bss_address) {
            return self.write_ipc_response(tls, INVALID_ADDRESS, &[], &[], &[]);
        }
        // The region is the ceiling on any one module, and checking against it
        // here is also what keeps a nonsense size from becoming a
        // multi-gigabyte host allocation two lines further down.
        let too_big = u64::from(super::RO_MODULE_REGION_SIZE);
        if nro_size == 0
            || !Self::ro_is_page_aligned(nro_size)
            || !Self::ro_is_page_aligned(bss_size)
            || nro_size > too_big
            || bss_size > too_big
        {
            return self.write_ipc_response(tls, INVALID_SIZE, &[], &[], &[]);
        }

        // The whole image, because that is what validating it takes:
        // `NroHeader::parse` checks each segment against the size the header
        // declares, and it can only do that with the bytes in hand.
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
        // The BSS is mapped behind the image and nowhere else, so a caller
        // that sized it short would have the module's zero-initialized data
        // land outside the mapping — on somebody else's module, once the
        // region has more than one.
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

        // Image and BSS in one write, so the BSS is genuinely zero rather than
        // whatever an unloaded module left in a page this run reuses.
        let mut mapped = image;
        mapped.resize(size as usize, 0);
        self.mem.map(base, &mapped)?;
        let text = (
            base.wrapping_add(header.text_offset),
            base.wrapping_add(header.text_offset)
                .wrapping_add(header.text_size),
        );
        // `.text` is never a relocation target, and a real kernel maps an
        // NRO's code segment read-execute — so a write into it is a bug worth
        // faulting on rather than one to absorb. Undone by `UnloadModule`, or
        // the protection would outlive the mapping and fault whatever is
        // mapped over it next.
        self.mem.mark_readonly(text.0, text.1);
        // A module mapped after the process started carries the `Alias*`
        // memory states rather than the process image's. See
        // `Memory::mark_module`.
        self.mem.mark_module(
            (text.0, base.wrapping_add(header.data_offset)),
            (
                base.wrapping_add(header.data_offset),
                base.wrapping_add(size),
            ),
            true,
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

    /// `IRoInterface::UnloadModule`: drop a mapping made by
    /// [`Cpu::ldr_ro_load_module`], freeing both the pages and the address
    /// space for the next load.
    ///
    /// The address is the one `LoadModule` returned — that is what `nn::ro`
    /// keeps — but the source buffer is accepted too. The two are a `u64`
    /// named `nro_address` in both directions of this interface, they are easy
    /// to confuse from the outside, and unmapping the wrong module is a far
    /// worse answer than accepting either.
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
        self.diagnostic(
            Level::Info,
            &format!(
                "[ro] unmapped the module at {:#010x} ({:#x} bytes)",
                module.base, module.size
            ),
        );
        self.write_ipc_response(tls, 0, &[], &[], &[])
    }

    /// `IRoInterface::RegisterModuleInfo`: the caller presenting the NRR that
    /// says which NROs it may load.
    ///
    /// Only the magic is checked. The rest of an NRR is a signature chain over
    /// a table of NRO hashes, verified on a console against a key that is not
    /// here — and a hash check without the signature behind it authorizes
    /// nothing, it only decides which NROs to refuse for a reason the caller
    /// cannot distinguish from a real one. So this records the registration
    /// and accepts it, which is also what a console with the check patched out
    /// does.
    fn ldr_ro_register_module_info(
        &mut self,
        tls: u32,
        nrr_address: u64,
        nrr_size: u64,
    ) -> Result<()> {
        /// "NRR0", the first four bytes of an NRR.
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

    /// The lowest free run of `size` bytes in the module region, or `None`
    /// when nothing there can hold one.
    ///
    /// First fit over the live mappings rather than a bump allocator: a title
    /// that loads and unloads plugins as it goes would otherwise walk the
    /// region and run out of address space it is not using. `ro_modules` is
    /// keyed by base address, so iterating it walks the mappings in order and
    /// the gaps fall out of the walk.
    fn ro_free_region(&self, size: u32) -> Option<u32> {
        let region_end = super::RO_MODULE_REGION_ADDR.wrapping_add(super::RO_MODULE_REGION_SIZE);
        let mut candidate = super::RO_MODULE_REGION_ADDR;
        for module in self.ro_modules.values() {
            if size <= module.base.saturating_sub(candidate) {
                return Some(candidate);
            }
            candidate = candidate.max(module.base.wrapping_add(module.size));
        }
        (size <= region_end.saturating_sub(candidate)).then_some(candidate)
    }

    /// Whether an address or size is a whole number of pages, which every
    /// argument `ro` takes has to be — it is mapping memory, and half a page
    /// of a module is not something to map.
    fn ro_is_page_aligned(value: u64) -> bool {
        value.is_multiple_of(crate::mem::PAGE_SIZE as u64)
    }
}
