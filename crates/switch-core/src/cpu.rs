//! AArch64 (A64) interpreter core.
//!
//! Implements a from-scratch decode + execute loop for the A64 instruction
//! set covering the integer core that compiled Switch homebrew actually uses:
//! integer ALU, shifts, bitfield ops, multiplies/divides, conditional selects
//! and compares, loads/stores (immediate, register-offset, literal, paired,
//! exclusive), PC-relative addressing, and the branch/subroutine family.
//!
//! System instructions (MRS/MSR/barriers/hints) are handled minimally, and
//! `SVC` drives a small, explicit syscall ABI used by the bundled demo
//! payload. Floating point, SIMD and the Horizon OS are out of scope for
//! Phase 1 and raise [`Error::Cpu`] if encountered.
//!
//! Encoding references are taken from the ARMv8 architecture and cross-checked
//! against QEMU's `target/arm/tcg/a64.decode`.

use crate::mem::Memory;
use crate::{Error, Result};
use std::collections::HashMap;

/// Where SVC traps are routed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SyscallMode {
    /// `SVC #0` halts the machine; anything else faults.
    #[default]
    None,
    /// Real libnx/libtransistor syscall numbers, best-effort stubs so homebrew
    /// built for the Switch can boot single-threaded: console logging, sleeps,
    /// handles and process/timing info are faked, and unsupported calls fault.
    Horizon,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunReport {
    /// Number of instructions executed this run.
    pub steps: u64,
    /// True if the machine reached a halt trap rather than exhausting the
    /// step budget.
    pub halted: bool,
}

/// Host-provided stack for [`Cpu::bootstrap`]: 1 MiB full-descending, top at
/// `STACK_TOP`. Must be clear of the NRO image (0x08000000+), the libnx TLS
/// (0x0FF00000) and the heap homebrew puts at 0x10000000 — hbmenu's deko3d
/// memblocks are allocated there and zeroing them would stomp a stack placed
/// in the same region, corrupting return addresses.
pub const STACK_SIZE: u64 = 0x0010_0000;
pub const STACK_TOP: u64 = 0x1010_0000;

/// Address of the return-address trampoline for direct-entered homebrew.
pub const SELF_RETURN_TRAMPOLINE: u32 = 0x1F00_0000;

#[derive(Debug)]
pub struct Cpu {
    pub mem: Memory,
    /// X0..=X30 (X31 is the stack pointer).
    regs: [u64; 31],
    /// The stack pointer register (X31).
    sp: u64,
    pc: u32,
    /// NZCV, packed as ARM PSTATE does: N=31, Z=30, C=29, V=28.
    nzcv: u32,
    /// SIMD vector registers Q0..=Q31 (128-bit). Only the handful of
    /// instructions libnx's `memset`/`memcpy` rely on are implemented;
    /// full NEON is out of scope for Phase 1.
    vregs: [u128; 32],
    /// Console output accumulated by the UART syscall mode.
    pub out: Vec<u8>,
    /// Debug trace: per-instruction disassembly (when enabled) plus fault
    /// context with a register snapshot.
    pub trace: Vec<u8>,
    /// When true, each executed instruction is appended to `trace`.
    pub trace_enabled: bool,
    /// Safety cap on the trace buffer to avoid unbounded growth.
    trace_cap: usize,
    pub syscall_mode: SyscallMode,
    pub halted: bool,
    /// Instructions executed in total.
    pub cycles: u64,
    /// Ring buffer of the most recent `RECENT_LEN` `(pc, insn)` pairs, dumped
    /// on fault so the path into a crash is visible without full tracing.
    recent: [(u32, u32); RECENT_LEN],
    /// Total instructions recorded into [`Cpu::recent`].
    recent_len: usize,
    /// Thread-local-storage base (TPIDR_EL0). libnx reads and writes this to
    /// find per-thread globals; stubbing it as always-zero makes those
    /// accesses land near address 0.
    tpidr: u64,
    /// Monotonic id handed out for domain IPC out-objects, so each synthesized
    /// subservice gets a distinct non-zero object id.
    next_object_id: u32,
    /// Maps fake handles returned by `ConnectToNamedPort` or sm:`GetService`
    /// to the named port / service they represent, so `SendSyncRequest` can
    /// dispatch IPC commands to the right stub.
    service_handles: HashMap<u64, String>,
    /// Monotonic fake-handle allocator. 0 is invalid, so we start above the
    /// earlier hard-coded `FAKE_HANDLE`.
    next_handle: u32,
    /// Monotonic domain object id allocator for services that use IPC domains
    /// (e.g. `vi:m`).
    next_domain_object_id: u32,
    /// Maps (session handle, domain object id) to the interface name used for
    /// dispatching domain commands.
    domain_objects: HashMap<(u64, u32), String>,
    /// Maps non-domain vi session handles to their sub-interface (vi:iads,
    /// vi:ihosbd, ...) so the display stub can dispatch binder vs. display
    /// commands on the right session.
    vi_ifaces: HashMap<u64, String>,
    /// Synthetic SD-card directory state for the fsp-srv stub: maps an open
    /// directory handle to the entries it has not yielded yet
    /// (name bytes, entry type, file size). Lets `fsFsOpenDirectory` /
    /// `fsDirRead` hand NX-Shell's `FS::GetDirList` a real listing.
    fs_dirs: HashMap<u64, Vec<(Vec<u8>, u8, u64)>>,
    /// Address the guest mapped its hid shared memory to (via `MapSharedMemory`
    /// on the handle hid's IPC returned). The host writes gamepad state into
    /// the libnx `HidSharedMemory` layout there so `padUpdate` sees it; 0 means
    /// hid hasn't been initialized yet.
    hid_shmem_addr: u32,
    /// Monotonic sampling number for the hid shared-memory LIFO entries.
    sample_counter: u64,
}

/// How many recently-executed instructions the fault trace shows.
pub const RECENT_LEN: usize = 64;

impl Default for Cpu {
    fn default() -> Self {
        Cpu::new()
    }
}

impl Cpu {
    pub fn new() -> Cpu {
        let mut cpu = Cpu {
            mem: Memory::new(),
            regs: [0; 31],
            sp: 0,
            pc: 0,
            nzcv: 0,
            vregs: [0; 32],
            out: Vec::new(),
            trace: Vec::new(),
            trace_enabled: false,
            trace_cap: 512 * 1024,
            syscall_mode: SyscallMode::None,
            halted: false,
            cycles: 0,
            recent: [(0, 0); RECENT_LEN],
            recent_len: 0,
            tpidr: 0,
            next_object_id: 1,
            service_handles: HashMap::new(),
            next_handle: 0x1000,
            next_domain_object_id: 1,
            domain_objects: HashMap::new(),
            vi_ifaces: HashMap::new(),
            fs_dirs: HashMap::new(),
            hid_shmem_addr: 0,
            sample_counter: 0,
        };
        // The framebuffer and input registers are fixed hardware-mapped
        // regions: pre-map them so reads never fault and programs (or the
        // host) can touch them before writing.
        let _ = cpu
            .mem
            .map_zero(crate::FB_BASE, (crate::FB_WIDTH * crate::FB_HEIGHT * 4) as usize);
        let _ = cpu.mem.map_zero(crate::INPUT_ADDR, 4096);
        cpu
    }

    /// Map a host-provided runtime environment and point SP at a stack, the
    /// way the real loader does before jumping to a program's entry point.
    ///
    /// Without this, libnx-style crt0 writes to low memory (applet/env
    /// metadata, null-relative globals) fault on the unmapped zeropage and
    /// there is no stack to push to. The demo never touches the stack, so the
    /// unit tests keep SP at 0; only hosts that want to boot real homebrew
    /// should call this.
    pub fn bootstrap(&mut self) {
        // Present the low 2 GiB address space (everything below the old
        // 2 GiB NRO base) as lazily mapped: reads return zeros, writes
        // allocate a page on first touch, so nothing is reserved up front.
        // This lets libnx-style code read heap/init globals without faulting
        // even when a baked-in pointer is stale.
        self.mem.soft_map_zero(0, 0x8000_0000);
        // 1 MiB full-descending stack; SP starts at the top.
        let _ = self.mem.map_zero((STACK_TOP - STACK_SIZE) as u32, STACK_SIZE as usize);
        self.sp = STACK_TOP;
        // libnx reads TPIDR_EL0 expecting the loader (HBL/kernel) to have set
        // the thread-local-storage base. Point it at a writable region clear of
        // both the heap and the stack: the stack mapping is [0x10000000,
        // 0x10100000) and svcSetHeapSize hands out 0x20000000. If TPIDR overlaps
        // either, the app's IPC code writes its CMIF request over the heap's
        // first chunk header (and malloc stomps the TLS), corrupting the
        // allocator.
        self.tpidr = 0x0FF0_0000;
        // A return-address trampoline: the loader enters homebrew's `main`
        // directly, so LR is 0 and any early return would branch to NULL.
        // Point LR at a stub that calls ExitProcess (svc 0x07), so main's
        // return surfaces as a clean exit code instead of a NULL jump.
        let _ = self.mem.map_zero(SELF_RETURN_TRAMPOLINE, 0x10);
        self.mem.write_u32(SELF_RETURN_TRAMPOLINE, 0xD400_00E1).ok(); // svc #7
        self.mem.write_u32(SELF_RETURN_TRAMPOLINE + 4, 0x1400_0000).ok(); // b .
    }

    /// Boot a homebrew NRO the way HBL does: load the image, let the crt0's
    /// relocation pass run up to the point it calls `main`, run the `.init_array`
    /// (C++ static constructors) and set up the main thread's `ThreadVars`
    /// (newlib reentrancy) that the skipped `__libnx_init` would normally
    /// provide, then leave the CPU ready to enter `main`.
    ///
    /// The libnx "HOME BREW" crt0 runs the relocation pass itself and then
    /// jumps to main; when it omits the `__libnx_init` step, every std::string
    /// global is left empty and NX-Shell's `FS::GetDirList` resolves its SD
    /// path to "" and exits. Running the constructors fixes that.
    pub fn boot_homebrew(&mut self, data: &[u8]) -> Result<crate::nro::LoadedNro> {
        let loaded = crate::nro::load_nro(&mut self.mem, data)?;
        self.out.clear();
        self.trace.clear();
        self.halted = false;
        self.trace_enabled = false;
        for i in 0..=30u8 {
            self.set_reg(i, 0);
        }
        self.set_reg(0, loaded.env_addr as u64);
        self.set_reg(1, if loaded.env_addr != 0 { u64::MAX } else { 1 });
        self.set_reg(30, SELF_RETURN_TRAMPOLINE as u64);

        let init = crate::nro::init_array_entries(data);
        if !init.is_empty() && loaded.env_addr != 0 {
            // The crt0 calls main with `bl` at entry+0xc0 (libnx switch_crt0
            // layout); by then BSS is zeroed and RELR relocations are applied.
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
                // ThreadVars at TLS+0x1E0: magic, handle, thread_ptr, _REENT,
                // tls_tp. The _REENT can be zeroed; newlib's malloc lazily
                // initializes it.
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
                // The constructors clobber the entry registers; restore them
                // and resume at the crt0's `bl main` so main is entered with
                // the normal calling convention (x30 = the crt0's return path).
                for i in 0..=30u8 {
                    self.set_reg(i, 0);
                }
                self.set_reg(0, loaded.env_addr as u64);
                self.set_reg(1, if loaded.env_addr != 0 { u64::MAX } else { 1 });
                self.set_reg(30, SELF_RETURN_TRAMPOLINE as u64);
                self.set_pc(main_call);
                return Ok(loaded);
            }
        }
        self.set_pc(loaded.entry);
        Ok(loaded)
    }

    // ---- register access ----

    #[inline]
    pub fn get_pc(&self) -> u32 {
        self.pc
    }

    #[inline]
    pub fn sp(&self) -> u64 {
        self.sp
    }

    pub fn set_pc(&mut self, pc: u32) {
        self.pc = pc;
    }

    /// Debug: dump the fake-handle -> service-name map.
    pub fn service_handles_snapshot(&self) -> Vec<(u64, String)> {
        let mut v: Vec<(u64, String)> = self
            .service_handles
            .iter()
            .map(|(&h, s)| (h, s.clone()))
            .collect();
        v.sort();
        v
    }

    /// Read X0..=X30 (X31 reads as zero / is the stack pointer).
    #[inline]
    pub fn read_x(&self, idx: u8) -> u64 {
        match idx {
            0..=30 => self.regs[idx as usize],
            31 => self.sp,
            _ => 0,
        }
    }

    pub fn read_reg(&self, idx: u8) -> u64 {
        self.read_x(idx)
    }

    /// Read the 128-bit SIMD&FP register Qn.
    pub fn read_vreg(&self, idx: u8) -> u128 {
        self.vregs[idx as usize]
    }

    /// Base of the libnx TLS (thread-local storage) region.
    pub fn tls_base(&self) -> u32 {
        self.tpidr as u32
    }

    /// Write the 128-bit SIMD&FP register Qn.
    pub fn set_vreg(&mut self, idx: u8, val: u128) {
        self.vregs[idx as usize] = val;
    }

    /// Read X0..=X30 where X31 is ZR (always zero).
    #[inline]
    fn read_zr(&self, idx: u8) -> u64 {
        if idx == 31 { 0 } else { self.regs[idx as usize] }
    }

    #[inline]
    fn write_zr(&mut self, idx: u8, val: u64) {
        if idx != 31 {
            self.regs[idx as usize] = val;
        }
    }

    /// Write X0..=X30 where X31 is SP.
    #[inline]
    fn write_x(&mut self, idx: u8, val: u64) {
        match idx {
            0..=30 => self.regs[idx as usize] = val,
            31 => self.sp = val,
            _ => {}
        }
    }

    pub fn set_reg(&mut self, idx: u8, val: u64) {
        self.write_zr(idx, val);
    }

    pub fn read_u32_reg(&self, idx: u8) -> u32 {
        self.read_zr(idx) as u32
    }

    pub fn set_pc_and_sp(&mut self, pc: u32, sp: u64) {
        self.pc = pc;
        self.sp = sp;
    }

    /// Write the host gamepad state so the guest can see it. The button
    /// bitmask goes to the memory-mapped [`crate::INPUT_ADDR`] (simple polling
    /// mechanism); when libnx has mapped its hid shared memory, the same state
    /// is mirrored into the player-1 `HidNpadInternalState` layout that
    /// `padUpdate` reads, so real homebrew (padInitialize/padUpdate) works too.
    ///
    /// `buttons` is a bitfield of `HidNpadButton` (A=1<<0, B=1<<1, X=1<<2,
    /// Y=1<<3, L=1<<4, R=1<<5, ZL=1<<6, ZR=1<<7, Plus=1<<8, Minus=1<<9,
    /// DpadLeft=1<<10, DpadUp=1<<11, DpadRight=1<<12, DpadDown=1<<13,
    /// StickL=1<<14, StickR=1<<15). Sticks are signed -32768..32767.
    pub fn set_gamepad_state(&mut self, buttons: u64, stick_lx: i32, stick_ly: i32, stick_rx: i32, stick_ry: i32) {
        // Simple host→guest register: a u64 mask, then two analog sticks.
        let _ = self.mem.write_u64(crate::INPUT_ADDR, buttons);
        let _ = self.mem.write_u32(crate::INPUT_ADDR + 8, stick_lx as u32);
        let _ = self.mem.write_u32(crate::INPUT_ADDR + 12, stick_ly as u32);
        let _ = self.mem.write_u32(crate::INPUT_ADDR + 16, stick_rx as u32);
        let _ = self.mem.write_u32(crate::INPUT_ADDR + 20, stick_ry as u32);

        if self.hid_shmem_addr == 0 {
            return;
        }
        self.write_hid_gamepad_state(buttons, stick_lx, stick_ly, stick_rx, stick_ry);
    }

    /// Mirror the gamepad state into libnx's `HidSharedMemory` for player 1.
    /// The `npad` section sits at offset 0x3D7C0; each entry's `internal_state`
    /// holds `style_set` at +0 and `full_key_lifo` at +0x20. A LIFO with one
    /// entry, a monotonic sampling number, the connected attribute and the
    /// button/stick state is enough for `padUpdate` to report input.
    fn write_hid_gamepad_state(&mut self, buttons: u64, lx: i32, ly: i32, rx: i32, ry: i32) {
        const NPAD_OFF: u32 = 0x3D7C0; // offsetof(HidSharedMemory, npad)
        const STYLE_FULL_KEY_HANDHELD: u32 = 1 | 4; // NpadFullKey | NpadHandheld
        const ATTR_CONNECTED: u32 = 1;
        let base = self.hid_shmem_addr.wrapping_add(NPAD_OFF);
        self.sample_counter = self.sample_counter.wrapping_add(1);
        // internal_state
        let _ = self.mem.write_u32(base, STYLE_FULL_KEY_HANDHELD);        // style_set
        let _ = self.mem.write_u32(base + 4, 0);                          // joy_assignment_mode
        // full_key_lifo at internal_state + 0x20
        let lifo = base.wrapping_add(0x20);
        let _ = self.mem.write_u64(lifo + 0x08, 1);       // header.buffer_count
        let _ = self.mem.write_u64(lifo + 0x10, 0);       // header.tail
        let _ = self.mem.write_u64(lifo + 0x18, 1);       // header.count
        let _ = self.mem.write_u64(lifo + 0x20, self.sample_counter); // storage[0].sampling_number
        let _ = self.mem.write_u64(lifo + 0x30, buttons); // storage[0].state.buttons
        let _ = self.mem.write_u32(lifo + 0x38, lx as u32);
        let _ = self.mem.write_u32(lifo + 0x3C, ly as u32);
        let _ = self.mem.write_u32(lifo + 0x40, rx as u32);
        let _ = self.mem.write_u32(lifo + 0x44, ry as u32);
        let _ = self.mem.write_u32(lifo + 0x48, ATTR_CONNECTED);          // attributes
    }

    #[inline]
    pub fn nzcv(&self) -> u32 {
        self.nzcv
    }

    #[inline]
    fn condition_holds(&self, cond: u8) -> bool {
        let n = (self.nzcv >> 31) & 1;
        let z = (self.nzcv >> 30) & 1;
        let c = (self.nzcv >> 29) & 1;
        let v = (self.nzcv >> 28) & 1;
        match cond & 0xF {
            0x0 => z == 1,                 // EQ
            0x1 => z == 0,                 // NE
            0x2 => c == 1,                 // CS
            0x3 => c == 0,                 // CC
            0x4 => n == 1,                 // MI
            0x5 => n == 0,                 // PL
            0x6 => v == 1,                 // VS
            0x7 => v == 0,                 // VC
            0x8 => c == 1 && z == 0,       // HI
            0x9 => c == 0 || z == 1,       // LS
            0xA => n == v,                 // GE
            0xB => n != v,                 // LT
            0xC => z == 0 && n == v,       // GT
            0xD => z == 1 || n != v,       // LE
            _ => true,                     // AL / NV
        }
    }

    #[inline]
    fn mask(sf: bool) -> u64 {
        if sf { u64::MAX } else { u32::MAX as u64 }
    }

    /// Compute `a + b + carry_in`, returning (result, carry-out, overflow).
    /// Operands are masked to the operation size first: callers pass `b` as
    /// the already-inverted subtrahend for SUB, whose 64-bit `!` would
    /// otherwise pollute the 32-bit carry/overflow computation.
    #[inline]
    fn add_carry_overflow(a: u64, b: u64, carry_in: u64, sf: bool) -> (u64, u32, u32) {
        let size = if sf { 64 } else { 32 };
        let mask = Self::mask(sf);
        let a = a & mask;
        let b = b & mask;
        let base = 1u128 << size;
        let sum = (a as u128) + (b as u128) + (carry_in as u128);
        let result = (sum & (base - 1)) as u64;
        let carry = ((sum >> size) & 1) as u32;
        let sign = 1u64 << (size - 1);
        let overflow = (((a & b & !result) | (!a & !b & result)) & sign != 0) as u32;
        (result, carry, overflow)
    }

    fn set_nzcv_from_alu(&mut self, result: u64, sf: bool, carry: u32, overflow: u32) {
        let n = ((result >> (if sf { 63 } else { 31 })) & 1) as u32;
        let z = (result == 0) as u32;
        self.nzcv = (n << 31) | (z << 30) | (carry << 29) | (overflow << 28);
    }

    fn set_nzcv_from_compare(&mut self, a: u64, b: u64, sub: bool, carry_in: u64, sf: bool) {
        let (result, carry, overflow) = if sub {
            Self::add_carry_overflow(a, !b, carry_in, sf)
        } else {
            Self::add_carry_overflow(a, b, carry_in, sf)
        };
        self.set_nzcv_from_alu(result, sf, carry, overflow);
    }

    fn add_sub(&mut self, rd: u8, rn: u8, rhs: u64, set_flags: bool, sub: bool, sf: bool) {
        let a = self.read_x(rn) & Self::mask(sf);
        let (result, carry, overflow) = if sub {
            Self::add_carry_overflow(a, !rhs, 1, sf)
        } else {
            Self::add_carry_overflow(a, rhs, 0, sf)
        };
        if set_flags {
            self.set_nzcv_from_alu(result, sf, carry, overflow);
        }
        // Rd=31 is SP only for the plain ADD/SUB forms. For ADDS/SUBS it is
        // XZR, so the result must be discarded — writing it to SP corrupts
        // the stack pointer on every CMP/CMN.
        if rd != 31 || !set_flags {
            self.write_x(rd, result);
        }
    }

    // ---- main execution ----

    /// Execute a single instruction. Returns `Ok(())` on success.
    pub fn step(&mut self) -> Result<()> {
        if self.halted {
            return Err(Error::Cpu("attempted to step a halted CPU".into()));
        }
        let pc = self.pc;
        let insn = match self.mem.fetch(pc) {
            Ok(i) => i,
            Err(e) => {
                self.record_fault(&e, pc, 0);
                return Err(e);
            }
        };
        let next_pc = pc.wrapping_add(4);
        self.recent[self.recent_len % RECENT_LEN] = (pc, insn);
        self.recent_len = self.recent_len.saturating_add(1);
        let result = self.execute(insn, next_pc);
        if self.trace_enabled {
            self.trace_line(&format!("{:08x}: {:08x}  {}\n", pc, insn, crate::disasm::disassemble(insn)));
        }
        if let Err(e) = &result {
            self.record_fault(e, pc, insn);
        }
        self.cycles += 1;
        result
    }

    fn record_fault(&mut self, e: &Error, pc: u32, insn: u32) {
        self.trace_line(&format!(
            "\n=== FAULT ===\n{}\n  at pc={:#010x} insn={:#010x}  {}\n",
            e,
            pc,
            insn,
            if insn == 0 {
                String::new()
            } else {
                crate::disasm::disassemble(insn)
            }
        ));
        self.trace_regs(pc);
        // Show the run-up to the fault so the crash path is readable without
        // full tracing enabled.
        let n = self.recent_len.min(RECENT_LEN);
        if n > 0 {
            let start = self.recent_len.wrapping_sub(n) % RECENT_LEN;
            self.trace_line(&format!("--- last {} instructions ---\n", n));
            for i in 0..n {
                let (ipc, iinsn) = self.recent[(start + i) % RECENT_LEN];
                self.trace_line(&format!(
                    "{:08x}: {:08x}  {}\n",
                    ipc,
                    iinsn,
                    crate::disasm::disassemble(iinsn)
                ));
            }
        }
    }

    fn trace_line(&mut self, line: &str) {
        if self.trace.len() >= self.trace_cap {
            if !self.trace.ends_with(b"\n[TRACE TRUNCATED]\n") {
                self.trace
                    .extend_from_slice(b"\n[TRACE TRUNCATED]\n");
            }
            return;
        }
        self.trace.extend_from_slice(line.as_bytes());
    }

    fn trace_regs(&mut self, pc: u32) {
        let dump = self.reg_dump();
        self.trace_line(&dump);
        let _ = pc;
    }

    /// Format a full register snapshot for debugging.
    pub fn reg_dump(&self) -> String {
        use std::fmt::Write;
        let mut s = String::with_capacity(1024);
        let n = (self.nzcv >> 31) & 1;
        let z = (self.nzcv >> 30) & 1;
        let c = (self.nzcv >> 29) & 1;
        let v = (self.nzcv >> 28) & 1;
        let _ = writeln!(
            s,
            "pc={:#010x}  sp={:#018x}  nzcv=N:{n} Z:{z} C:{c} V:{v}",
            self.pc, self.sp
        );
        for i in 0..31 {
            let _ = write!(s, "x{:<2}={:#018x}  ", i, self.regs[i]);
            if i % 4 == 3 {
                let _ = writeln!(s);
            }
        }
        let _ = writeln!(s);
        s
    }

    /// Run up to `max_steps` instructions, stopping early on halt or error.
    pub fn run(&mut self, max_steps: u64) -> Result<RunReport> {
        let mut steps = 0u64;
        while steps < max_steps && !self.halted {
            self.step()?;
            steps += 1;
        }
        Ok(RunReport {
            steps,
            halted: self.halted,
        })
    }

    #[inline]
    fn b_imm(&mut self, next_pc: &mut u32, imm: i64) {
        *next_pc = (self.pc as i64).wrapping_add(imm) as u32;
    }

    fn execute(&mut self, insn: u32, mut next_pc: u32) -> Result<()> {
        // ---------------- unconditional branches ----------------
        let op26 = (insn >> 26) & 0x3F;
        if op26 == 0b000101 {
            // B #imm
            let imm = sext_u64((insn & 0x3FF_FFFF) as u64, 26) << 2;
            self.b_imm(&mut next_pc, imm as i64);
            self.pc = next_pc;
            return Ok(());
        }
        if op26 == 0b100101 {
            // BL #imm
            let imm = sext_u64((insn & 0x3FF_FFFF) as u64, 26) << 2;
            self.write_zr(30, next_pc as u64);
            self.b_imm(&mut next_pc, imm as i64);
            self.pc = next_pc;
            return Ok(());
        }

        // ---------------- load literal ----------------
        if ((insn >> 27) & 0b111) == 0b011 && ((insn >> 26) & 1) == 0 && ((insn >> 24) & 0b11) == 0b00 {
            let rt = (insn & 0x1F) as u8;
            let imm = sext_u64((insn >> 5) & 0x7_FFFF, 19) << 2;
            let addr = (self.pc as i64).wrapping_add(imm as i64) as u32;
            let sz = (insn >> 30) & 0b11;
            let (val, width, sign) = match sz {
                0b00 => (self.mem.read_u32(addr)? as u64, 32, false),
                0b01 => (self.mem.read_u64(addr)?, 64, false),
                0b10 => (self.mem.read_u32(addr)? as u64, 32, true),
                _ => {
                    // PRFM: prefetch hint, treat as NOP
                    self.pc = next_pc;
                    return Ok(());
                }
            };
            let val = if sign {
                sext_u64(val, width)
            } else if width == 32 {
                val & u32::MAX as u64
            } else {
                val
            };
            self.write_zr(rt, val);
            self.pc = next_pc;
            return Ok(());
        }

        // ---------------- branch register ----------------
        if ((insn >> 25) & 0x7F) == 0b1101011 {
            let opc = (insn >> 21) & 0xF;
            let op2 = (insn >> 16) & 0x1F;
            let op3 = (insn >> 10) & 0x3F;
            if op2 == 0x1F && op3 == 0 {
                let rn = ((insn >> 5) & 0x1F) as u8;
                match opc {
                    0b0000 => {
                        // BR
                        self.pc = self.read_zr(rn) as u32;
                        return Ok(());
                    }
                    0b0001 => {
                        // BLR
                        self.write_zr(30, next_pc as u64);
                        self.pc = self.read_zr(rn) as u32;
                        return Ok(());
                    }
                    0b0010 => {
                        // RET. A `ret` that returns to address 0 is a homebrew
                        // exit path whose return-address convention the boot
                        // model doesn't provide (the crt0 stashes the loader's
                        // LR in x27, but the atexit table runner returns via
                        // x30 = 0). Redirect to the exit trampoline so it
                        // surfaces as a clean ExitProcess instead of a NULL
                        // fetch.
                        let tgt = self.read_zr(rn) as u32;
                        self.pc = if tgt == 0 { SELF_RETURN_TRAMPOLINE } else { tgt };
                        return Ok(());
                    }
                    _ => {
                        return Err(Error::Cpu(format!(
                            "unimplemented branch-register opc {:#b} at {:#x}",
                            opc,
                            self.pc
                        )))
                    }
                }
            }
        }

        // ---------------- exceptions ----------------
        if ((insn >> 24) & 0xFF) == 0b11010100 {
            let kind = (insn >> 21) & 0b111;
            return match kind {
                0b000 => {
                    // SVC/HVC/SMC; SVC has low bits 0b00001
                    if (insn & 0x1F) == 0b00001 {
                        let imm = ((insn >> 5) & 0xFFFF) as u16;
                        self.syscall(imm)?;
                        self.pc = next_pc;
                        Ok(())
                    } else {
                        Err(Error::Cpu(format!(
                            "unimplemented HVC/SMC at {:#x}",
                            self.pc
                        )))
                    }
                }
                0b001 => {
                    // BRK
                    let imm = ((insn >> 5) & 0xFFFF) as u16;
                    Err(Error::Cpu(format!(
                        "BRK #{} at {:#x}",
                        imm, self.pc
                    )))
                }
                _ => Err(Error::Cpu(format!(
                    "unimplemented exception instruction at {:#x}",
                    self.pc
                ))),
            };
        }

        // ---------------- system ----------------
        if ((insn >> 22) & 0x3FF) == 0b1101010100 {
            return self.system(insn, next_pc);
        }

        // ---------------- conditional branch ----------------
        if ((insn >> 24) & 0xFF) == 0b01010100 {
            let imm = sext_u64((insn >> 5) & 0x7_FFFF, 19) << 2;
            let cond = (insn & 0xF) as u8;
            if self.condition_holds(cond) {
                self.b_imm(&mut next_pc, imm as i64);
            }
            self.pc = next_pc;
            return Ok(());
        }

        // ---------------- compare & branch ----------------
        if ((insn >> 25) & 0x3F) == 0b011010 {
            let rt = (insn & 0x1F) as u8;
            let nz = ((insn >> 24) & 1) == 1;
            let imm = sext_u64((insn >> 5) & 0x7_FFFF, 19) << 2;
            let val = self.read_zr(rt);
            let is_zero = if (insn >> 31) & 1 == 1 {
                val == 0
            } else {
                (val as u32) == 0
            };
            if is_zero == !nz {
                self.b_imm(&mut next_pc, imm as i64);
            }
            self.pc = next_pc;
            return Ok(());
        }

        // ---------------- test bit & branch ----------------
        if ((insn >> 25) & 0x3F) == 0b011011 {
            let rt = (insn & 0x1F) as u8;
            let nz = ((insn >> 24) & 1) == 1;
            let bit = ((insn >> 31) & 1) << 5 | ((insn >> 19) & 0x1F);
            let imm = sext_u64((insn >> 5) & 0x3FFF, 14) << 2;
            let val = self.read_zr(rt);
            let bit_val = (val >> bit) & 1 == 1;
            if bit_val == nz {
                self.b_imm(&mut next_pc, imm as i64);
            }
            self.pc = next_pc;
            return Ok(());
        }

        // ---------------- loads & stores ----------------
        if self.try_load_store(insn, &mut next_pc)? {
            self.pc = next_pc;
            return Ok(());
        }

        // ---------------- minimal SIMD (vector registers) ----------------
        if self.try_simd(insn)? {
            self.pc = next_pc;
            return Ok(());
        }

        // ---------------- scalar floating point ----------------
        if self.try_fp(insn)? {
            self.pc = next_pc;
            return Ok(());
        }

        // ---------------- PC-relative addressing ----------------
        // ADR/ADRP: fixed bits[28:24] == 10000; bits[30:29] are immlo (not
        // zero in general, so the older check that required them to be 0
        // silently dropped real ADRP instructions).
        if ((insn >> 24) & 0x1F) == 0b10000 {
            let rd = (insn & 0x1F) as u8;
            let immhi = ((insn >> 5) & 0x7_FFFF) as u64;
            let immlo = ((insn >> 29) & 0b11) as u64;
            let imm = sext_u64((immhi << 2) | immlo, 21);
            let page = (insn >> 31) & 1 == 1;
            let base = if page {
                (self.pc & !0xFFF) as u64
            } else {
                self.pc as u64
            };
            let target = if page {
                base.wrapping_add(imm.wrapping_shl(12))
            } else {
                base.wrapping_add(imm)
            };
            self.write_zr(rd, target);
            self.pc = next_pc;
            return Ok(());
        }

        // ---------------- data processing: immediate ----------------
        if self.try_data_proc_imm(insn, &mut next_pc)? {
            self.pc = next_pc;
            return Ok(());
        }

        // ---------------- data processing: register ----------------
        if self.try_data_proc_reg(insn, &mut next_pc)? {
            self.pc = next_pc;
            return Ok(());
        }

        Err(Error::Cpu(format!(
            "unimplemented instruction 0x{:08x} at pc={:#x}",
            insn, self.pc
        )))
    }

    /// Minimal SIMD data-processing for the vector registers — just enough for
    /// the libnx `memset`/`memcpy` hot path (Phase 1 keeps NEON out of scope).
    ///
    /// Handled forms (fixed bits `[30:23] == 10_011100`, `[22:20] == 000`):
    /// * `DUP <Vd>.<T>, <Rn>`  — replicate the low element of a GPR across the
    ///   vector (`bits[15:10] == 000011`).
    /// * `MOV <Xd>, <Vn>.D[<m>]` (UMOV) — copy a 64-bit lane to a GPR
    ///   (`bits[15:10] == 001111`, 64-bit lane form `imm5 == 01000 | m`).
    fn try_simd(&mut self, insn: u32) -> Result<bool> {
        // ---- narrowing shifts (SHRN / RSHRN / SQSHRN / ...) ----
        // bits[31]=0, bits[28:24]=01111, bit23=0. These share the group with
        // MOVI (bits[28:23]=011110), so they must be checked first; the
        // opcode lives in bits[15:11] and the shift in bits[22:16].
        if ((insn >> 31) & 1) == 0
            && ((insn >> 24) & 0x1F) == 0b01111
            && ((insn >> 23) & 1) == 0
            && ((insn >> 10) & 1) == 1
        {
            let op = (insn >> 11) & 0x1F;
            if matches!(op, 0b10000 | 0b10001 | 0b10010 | 0b10011) {
                let q = (insn >> 30) & 1 == 1;
                let u = (insn >> 29) & 1;
                let rd = (insn & 0x1F) as u8;
                let rn = ((insn >> 5) & 0x1F) as u8;
                let size = (insn >> 22) & 1;
                let dest_esize = 8u32 << size;
                let shift_field = (insn >> 16) & 0x7F;
                let shift = (2 * dest_esize).saturating_sub(shift_field as u32);
                if shift > 0 && shift <= dest_esize {
                    let rounding = op & 1 == 1; // RSHRN/SQRSHRN/UQRSHRN round
                    let (signed_src, to_unsigned) = match (u, op) {
                        (0, 0b10000) => (false, false), // SHRN
                        (0, 0b10001) => (false, false), // RSHRN
                        (1, 0b10000) => (true, true),   // SQSHRUN
                        (1, 0b10001) => (true, true),   // SQRSHRUN
                        (0, 0b10010) => (true, false),  // SQSHRN
                        (0, 0b10011) => (true, false),  // SQRSHRN
                        (1, 0b10010) => (false, false), // UQSHRN
                        (1, 0b10011) => (false, false), // UQRSHRN
                        _ => unreachable!(),
                    };
                    let saturating = op & 0b10 != 0 || to_unsigned;
                    self.simd_shrn(rd, rn, q, dest_esize, shift, rounding, signed_src, to_unsigned, saturating);
                    return Ok(true);
                }
            }
        }

        // MOVI/MVNI (modified immediate): bits[28:23] == 011110, bits[22:19]==0.
        // The 8-bit immediate is NOT contiguous: `abcdefgh` sits at bits 18:16
        // (a:b:c) and 9:5 (d:e:f:g:h), with bits 15:12 = cmode, bit 29 = op
        // (0 = MOVI, 1 = MVNI/bitwise). Cross-checked against QEMU.
        if ((insn >> 31) & 1) == 0 && ((insn >> 23) & 0x3F) == 0b011110 && ((insn >> 19) & 0b1111) == 0b0000 {
            let q = (insn >> 30) & 1;
            let op = (insn >> 29) & 1;
            let rd = (insn & 0x1F) as u8;
            let imm8 = (((insn >> 16) & 0b111) << 5) | ((insn >> 5) & 0x1F);
            let cmode = (insn >> 12) & 0b1111;
            let imm64 = simd_imm_const(imm8, cmode, op);
            // q=0 writes only the low 64 bits (upper half cleared).
            self.vregs[rd as usize] = if q == 1 {
                imm64 as u128 | ((imm64 as u128) << 64)
            } else {
                imm64 as u128
            };
            return Ok(true);
        }

        // ---- permute (ZIP/UZP/TRN) ----
        // bit31=0, q=bit30, bits[29:24]=001110, bit21=0, opcode in bits[15:10]
        // (UZP1/TRN1/ZIP1 = 000110/001010/001110, UZP2/TRN2/ZIP2 = 010110/
        // 011010/011110). The copy-group guard above must not swallow these.
        let perm = (insn >> 10) & 0b111111;
        if ((insn >> 31) & 1) == 0
            && ((insn >> 24) & 0x1F) == 0b01110
            && ((insn >> 21) & 1) == 0
            && matches!(
                perm,
                0b000110 | 0b010110 | 0b001010 | 0b011010 | 0b001110 | 0b011110
            )
        {
            let q = (insn >> 30) & 1 == 1;
            let rd = (insn & 0x1F) as u8;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rm = ((insn >> 16) & 0x1F) as u8;
            let esize = 8u32 << ((insn >> 22) & 0b11);
            self.simd_permute(rd, rn, rm, q, esize, perm);
            return Ok(true);
        }

        // ---- integer three-same / compare / logical (Advanced SIMD) ----
        // bit31=0 with bits[29:24]=001110 (signed group, bit29=0) or 011110
        // (unsigned group, bit29=1). The opcode is in bits[15:11] with
        // bit10=1; the only bit10=0 form handled here is CMEQ #0.
        let grp = (insn >> 24) & 0x1F;
        // Vector three-same always has bits[28:24] == 01110 (bit28=0);
        // bits[28:24] == 11110 is the scalar-FP group, handled by try_fp.
        // Copy group (DUP/INS/UMOV/SMOV, 0{q}00 1110 000): q (bit30) is free,
        // and bit20 is part of imm5 (so it may be set for 64-bit lanes).
        let copy_group = ((insn >> 21) & 0x1FF) == 0b001110000 && ((insn >> 31) & 1) == 0;
        if ((insn >> 31) & 1) == 0 && grp == 0b01110 && !copy_group {
            // (copy_group == the DUP/MOV/INS encodings, which also live in the
            // 0x4e group with bits[23:21] == 000 and are handled below.)
            let q = (insn >> 30) & 1 == 1;
            let rd = (insn & 0x1F) as u8;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rm = ((insn >> 16) & 0x1F) as u8;
            let sz = (insn >> 22) & 0b11;
            let u = (insn >> 29) & 1; // 0 → 0x4e group, 1 → 0x6e group
            let op = (insn >> 11) & 0x1F;
            let b10 = (insn >> 10) & 1;
            let esize = match sz {
                0 => 8u32,
                1 => 16,
                2 => 32,
                _ => 64,
            };
            if b10 == 1 {
                match op {
                    0b00000 => {
                        // SHADD (signed group) / UHADD (unsigned group):
                        // halving add, (a+b) >> 1.
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            if u == 0 {
                                ((a as i128 + b as i128) >> 1) as u64
                            } else {
                                a.wrapping_add(b) >> 1
                            }
                        });
                        return Ok(true);
                    }
                    0b00001 => {
                        // SQADD (signed group) / UQADD (unsigned group).
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            saturating_add(a, b, esize, u != 0)
                        });
                        return Ok(true);
                    }
                    0b00010 => {
                        // SRHADD / URHADD: rounding halving add.
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            if u == 0 {
                                ((a as i128 + b as i128 + 1) >> 1) as u64
                            } else {
                                a.wrapping_add(b).wrapping_add(1) >> 1
                            }
                        });
                        return Ok(true);
                    }
                    0b00100 => {
                        // SHSUB / UHSUB: halving subtract.
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            if u == 0 {
                                ((a as i128 - b as i128) >> 1) as u64
                            } else {
                                a.wrapping_sub(b) >> 1
                            }
                        });
                        return Ok(true);
                    }
                    0b00101 => {
                        // SQSUB / UQSUB: saturating subtract.
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            saturating_sub(a, b, esize, u != 0)
                        });
                        return Ok(true);
                    }
                    0b01000 => {
                        // SSHL / USHL: shift left by register (negative shift
                        // amounts shift right).
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            shift_by_reg(a, b, esize, u != 0)
                        });
                        return Ok(true);
                    }
                    0b10000 => {
                        // ADD (signed group) / SUB (unsigned group).
                                    self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            if u == 0 {
                                a.wrapping_add(b)
                            } else {
                                a.wrapping_sub(b)
                            }
                        });
                        return Ok(true);
                    }
                    0b10001 => {
                        // CMTST (signed group) / CMEQ (unsigned group).
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            if u == 0 {
                                if a & b != 0 { u64::MAX } else { 0 }
                            } else if a == b {
                                u64::MAX
                            } else {
                                0
                            }
                        });
                        return Ok(true);
                    }
                    0b00111 => {
                        // CMGE (signed group) / CMHS (unsigned group).
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            let ge = if u == 0 {
                                Self::sge(a, b, esize)
                            } else {
                                a >= b
                            };
                            if ge { u64::MAX } else { 0 }
                        });
                        return Ok(true);
                    }
                    0b10111 if u == 0 => {
                        // ADDP: pairwise addition.
                        self.simd_pairwise(rd, rn, rm, q, esize, |a, b| a.wrapping_add(b));
                        return Ok(true);
                    }
                    0b10100 => {
                        // SMAXP (signed group) / UMAXP (unsigned group).
                        self.simd_pairwise(rd, rn, rm, q, esize, |a, b| {
                            if u == 0 {
                                if Self::sge(a, b, esize) { a } else { b }
                            } else {
                                a.max(b)
                            }
                        });
                        return Ok(true);
                    }
                    0b10101 => {
                        // SMINP (signed group) / UMINP (unsigned group).
                        self.simd_pairwise(rd, rn, rm, q, esize, |a, b| {
                            if u == 0 {
                                if Self::sge(a, b, esize) { b } else { a }
                            } else {
                                a.min(b)
                            }
                        });
                        return Ok(true);
                    }
                    0b00011 => {
                        // Bitwise logicals (the selector lives in bits[23:21];
                        // it doubles as `sz`, so no sz guard here).
                        let sub = (insn >> 21) & 0b111;
                        let a = self.vregs[rn as usize];
                        let b = self.vregs[rm as usize];
                        let full = if q { u128::MAX } else { (1u128 << 64) - 1 };
                        let d = self.vregs[rd as usize];
                        let r = match (u, sub) {
                            (0, 0b001) => a & b,        // AND
                            (0, 0b011) => a & !b,        // BIC
                            (0, 0b101) => a | b,         // ORR
                            (0, 0b111) => a | !b,        // ORN
                            (1, 0b001) => a ^ b,         // EOR
                            (1, 0b011) => (d & a) | (b & !a), // BSL: mask = Vn
                            (1, 0b101) => (b & a) | (d & !a), // BIT: mask = Vn
                            (1, 0b111) => (b & d) | (a & !d), // BIF: mask = Vd
                            _ => return Ok(false),
                        };
                        self.vregs[rd as usize] = r & full;
                        return Ok(true);
                    }
                    _ => {}
                }
            } else if op == 0b10011 && rm == 0 && u == 0 {
                // CMEQ <Vd>.<T>, <Vn>.<T>, #0 (compare against zero).
                self.simd_elem(rd, rn, rm, q, esize, |a, _| {
                    if a == 0 { u64::MAX } else { 0 }
                });
                return Ok(true);
            }
            return Err(Error::Cpu(format!(
                "unimplemented SIMD three-same u={} op={:#b} sz={} at {:#x}",
                u, op, sz, self.pc
            )));
        }

        // ---- copy / element moves ----
        if ((insn >> 21) & 0x1FF) != 0b001110000 || ((insn >> 31) & 1) != 0 {
            return Ok(false);
        }
        let q = (insn >> 30) & 1;
        let rd = (insn & 0x1F) as u8;
        let rn = ((insn >> 5) & 0x1F) as u8;
        let imm5 = (insn >> 16) & 0x1F;
        match (insn >> 10) & 0b111111 {
            0b000111 => {
                // INS <Vd>.<T>[<index>], <Rn> — insert a GPR lane. Same
                // imm5 → (esize, index) scheme as UMOV/SMOV: esize = 8<<ctz,
                // index = imm5 >> (ctz+1).
                let lsb = imm5.trailing_zeros();
                if lsb > 3 {
                    return Ok(false);
                }
                let esize = 8u32 << lsb;
                let index = imm5 >> (lsb + 1);
                let shift = (index as u32) * esize;
                let mask = (1u128 << esize) - 1;
                let v = self.vregs[rd as usize];
                let val = (self.read_zr(rn) as u128) & mask;
                self.vregs[rd as usize] = (v & !(mask << shift)) | (val << shift);
                Ok(true)
            }
            0b000011 if imm5 != 0 => {
                // DUP <Vd>.<T>, <Rn>: element size is `8 << ctz(imm5)` (imm5 =
                // 1/2/4/8 for 8/16/32/64-bit; the low bits hold the element
                // index, which the general-register form ignores).
                let esize = 8u32 << imm5.trailing_zeros();
                let elements = if q == 1 { 128 / esize } else { 64 / esize };
                let val = (self.read_zr(rn) as u128) & ((1u128 << esize) - 1);
                let mut v: u128 = 0;
                for i in 0..elements {
                    v |= (val as u128) << (i as u32 * esize);
                }
                self.vregs[rd as usize] = v;
                Ok(true)
            }
            0b001111 => {
                let lsb = imm5.trailing_zeros();
                let esize = 8u32 << lsb;
                let index = imm5 >> (lsb + 1);
                let shift = (index as u32) * esize;
                let val = (self.vregs[rn as usize] >> shift) & ((1u128 << esize) - 1);
                self.write_zr(rd, val as u64);
                Ok(true)
            }
            0b001011 => {
                // SMOV <Xd/Wd>, <Vn>.B/H/S[<index>] — extract a lane,
                // sign-extended (8/16-bit → Wd, 32-bit → Xd).
                let lsb = imm5.trailing_zeros();
                let esize = 8u32 << lsb;
                let index = imm5 >> (lsb + 1);
                let shift = (index as u32) * esize;
                let val = (self.vregs[rn as usize] >> shift) & ((1u128 << esize) - 1);
                let val = sext_u64(val as u64, esize);
                self.write_zr(rd, val);
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    // ---------------- scalar floating point ----------------
    //
    // The scalar FP subset hbmenu's UI/drawing code needs: FMOV, the common
    // arithmetic (FADD/FSUB/FMUL/FDIV/FNMUL/FMAX/FMIN/FMAXNM/FMINNM), the
    // unary ops (FABS/FNEG/FSQRT/FRINTx/FCVT between single and double),
    // fused multiply-add (FMADD/FMSUB/FNMADD/FNMSUB), compares (FCMP/
    // FCMPE/FCCMP), FCSEL, and the integer<->float conversions. NaN, infinity
    // and rounding come straight from Rust's IEEE f32/f64 (round-to-nearest,
    // the FPCR default); FP exception flags are not modelled.

    #[inline]
    fn fp_get_f32(&self, r: u8) -> f32 {
        f32::from_bits(self.vregs[r as usize] as u32)
    }

    #[inline]
    fn fp_get_f64(&self, r: u8) -> f64 {
        f64::from_bits(self.vregs[r as usize] as u64)
    }

    #[inline]
    fn fp_set_f32(&mut self, r: u8, v: f32) {
        let bits = v.to_bits() as u128;
        self.vregs[r as usize] = (self.vregs[r as usize] & !0xFFFF_FFFF) | bits;
    }

    #[inline]
    fn fp_set_f64(&mut self, r: u8, v: f64) {
        self.vregs[r as usize] = v.to_bits() as u128;
    }

    fn try_fp(&mut self, insn: u32) -> Result<bool> {
        let sf = (insn >> 31) & 1;
        // FMOV (immediate): bits[31:24] = 00011110, bit21 = 1,
        // bits[12:10] = 100, bits[9:5] = 0, imm8 = bits[20:13], type in
        // bits[23:22] (00 = S, 01 = D). The value is VFPExpandImm() —
        // `fmov s0, #1.0` = 0x1E2E1002 (sdl-hello's float env-var helper).
        if ((insn >> 24) & 0xFF) == 0b00011110
            && ((insn >> 21) & 1) == 1
            && ((insn >> 10) & 0b111) == 0b100
            && ((insn >> 5) & 0x1F) == 0
        {
            let imm8 = ((insn >> 13) & 0xFF) as u8;
            let ftype = (insn >> 22) & 0b11;
            let rd = (insn & 0x1F) as u8;
            let sign = if (imm8 >> 7) & 1 == 1 { 0x8000u32 } else { 0 };
            return match ftype {
                0b00 => {
                    let imm = (sign
                        | if (imm8 >> 6) & 1 == 1 { 0x3E00 } else { 0x4000 }
                        | (((imm8 & 0x3F) as u32) << 3))
                        << 16;
                    self.fp_set_f32(rd, f32::from_bits(imm));
                    Ok(true)
                }
                0b01 => {
                    let imm = (sign as u64
                        | if (imm8 >> 6) & 1 == 1 { 0x3FC0 } else { 0x4000 }
                        | (imm8 & 0x3F) as u64)
                        << 48;
                    self.fp_set_f64(rd, f64::from_bits(imm));
                    Ok(true)
                }
                _ => Ok(false), // half precision: out of scope
            };
        }

        // FCVT (float <-> float): bits[31:24] = 00011110, bit21 = 1,
        // bits[20:15] = 000100 (Dn -> Sd) / 000101 (Sn -> Dd),
        // bits[14:10] = 10000. `fcvt s0, d0` = 0x1E624000.
        if ((insn >> 24) & 0xFF) == 0b00011110
            && ((insn >> 21) & 1) == 1
            && ((insn >> 10) & 0x1F) == 0b10000
        {
            let op = (insn >> 15) & 0x3F;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rd = (insn & 0x1F) as u8;
            return match op {
                0b000100 => {
                    // double -> single
                    self.fp_set_f32(rd, self.fp_get_f64(rn) as f32);
                    Ok(true)
                }
                0b000101 => {
                    // single -> double
                    self.fp_set_f64(rd, self.fp_get_f32(rn) as f64);
                    Ok(true)
                }
                _ => Ok(false),
            };
        }
        // FMOV (register): move between GPR and a vector lane. bits[30:24] =
        // 0011110, bits[15:10] = 000000, bits[21:16] select direction/size.
        if ((insn >> 24) & 0x7F) == 0b0011110
            && ((insn >> 10) & 0x3F) == 0
            && matches!((insn >> 16) & 0x3F, 0b100110 | 0b100111)
        {
            let sel = (insn >> 16) & 0x3F;
            let double = ((insn >> 22) & 1) == 1;
            let rd = (insn & 0x1F) as u8;
            let rn = ((insn >> 5) & 0x1F) as u8;
            match sel {
                0b100110 => {
                    // FMOV Xd/Wd, Dn/Sn — move the FP bit pattern to a GPR.
                    let val = if double {
                        self.fp_get_f64(rn).to_bits()
                    } else {
                        self.fp_get_f32(rn).to_bits() as u64
                    };
                    self.write_zr(rd, val);
                }
                0b100111 => {
                    // FMOV Vd.D/S, Xn/Wn — move a GPR bit pattern to FP.
                    if double {
                        self.fp_set_f64(rd, f64::from_bits(self.read_zr(rn)));
                    } else {
                        self.fp_set_f32(rd, f32::from_bits(self.read_zr(rn) as u32));
                    }
                }
                _ => return Ok(false),
            }
            return Ok(true);
        }

        // Integer <-> float conversions: bits[30:24] = 0011110 (sf at bit31),
        // `type` in bits[23:22] picks single/double, opc in bits[21:16].
        // The pure integer forms have bits[15:10] = 0; non-zero there means a
        // fixed-point scale (or, with bit21=1, a 2-source FP op — FADD etc.).
        if ((insn >> 24) & 0x7F) == 0b0011110 && ((insn >> 10) & 0x3F) == 0 {
            let ftype = (insn >> 22) & 0b11; // 00 → S, 01 → D
            let opc = (insn >> 16) & 0x3F;
            let rd = (insn & 0x1F) as u8;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let fbits = (insn >> 10) & 0x3F;
            // type 00 = single, 01 = double (10/11 = half, out of scope). The
            // float size is independent of `sf`; the old `&& sf == 1` clause
            // made `fcvtzs w, d1` / `scvtf d0, w1` pick the wrong register size.
            let use_double = ftype == 0b01;
            return match opc {
                0b000010 | 0b000011 => {
                    // SCVTF / UCVTF: integer → float.
                    let signed = opc == 0b000010;
                    let v = self.read_zr(rn);
                    if use_double {
                        let f = if signed {
                            (v as i64) as f64
                        } else {
                            v as f64
                        };
                        self.fp_set_f64(rd, f);
                    } else {
                        let f = if signed {
                            (v as i32) as f32
                        } else {
                            v as f32
                        };
                        self.fp_set_f32(rd, f);
                    }
                    Ok(true)
                }
                 0b011000 | 0b011001 | 0b101000 | 0b111000 | 0b101001 | 0b111001 => {
                    // FCVTZS / FCVTZU: float → integer, round toward zero.
                    // opc bit16 = 0 for ZS (signed), 1 for ZU (unsigned);
                    // bit20 (and `type`) selects single vs double source.
                    // `fcvtzs w24, d1` = 0x1e780038, `fcvtzu x8, d1` = 0x9e790028.
                    let signed = (opc & 1) == 0;
                    let f = if use_double {
                        self.fp_get_f64(rn)
                    } else {
                        self.fp_get_f32(rn) as f64
                    };
                    let r = if signed {
                        if sf != 0 {
                            f as i64 as u64
                        } else {
                            f as i32 as u32 as u64
                        }
                    } else if sf != 0 {
                        f as u64
                    } else {
                        f as u32 as u64
                    };
                    self.write_zr(rd, r);
                    Ok(true)
                }
                0b100000..=0b100111 if fbits == 0 => {
                    // Float → integer with explicit rounding mode (opc 1000xx).
                    let (signed, rounding) = match opc {
                        0b100000 => (true, Rounding::TiesEven),
                        0b100001 => (false, Rounding::TiesEven),
                        0b100010 => (true, Rounding::TowardNeg),
                        0b100011 => (false, Rounding::TowardNeg),
                        0b100100 => (true, Rounding::TowardPos),
                        0b100101 => (false, Rounding::TowardPos),
                        0b100110 => (true, Rounding::TiesAway),
                        0b100111 => (false, Rounding::TiesAway),
                        _ => unreachable!(),
                    };
                    let f = if use_double {
                        self.fp_get_f64(rn)
                    } else {
                        self.fp_get_f32(rn) as f64
                    };
                    let r = round_to_int(f, rounding, signed);
                    self.write_zr(rd, r & Self::mask(sf != 0));
                    Ok(true)
                }
                _ => Ok(false),
            }
        }

        // Scalar integer compare-to-zero: CMGE/CMGT/CMLE/CMLT <Dd>, <Dn>, #0.
        // bits[31:30] = 01 (D), bit29 = U, bits[28:25] = 1110,
        // bits[24:21] = 0111, bits[20:16] = 00000 (the zero operand),
        // op = bits[15:10]. The result is an all-ones/all-zeros mask (used as
        // a predicate by NX-Shell: `cmge d31, d31, #0` then `fmov x2, d31`).
        if ((insn >> 31) & 1) == 0
            && ((insn >> 30) & 0b11) == 0b01
            && ((insn >> 25) & 0b1111) == 0b1111
            && ((insn >> 21) & 0xF) == 0b0111
            && ((insn >> 16) & 0x1F) == 0
        {
            let u = (insn >> 29) & 1;
            let op = (insn >> 10) & 0x3F;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rd = (insn & 0x1F) as u8;
            let v = (self.vregs[rn as usize] as u64) as i64;
            let cond = match (u, op) {
                (1, 0b100010) => v >= 0, // CMGE
                (0, 0b100010) => v > 0,  // CMGT
                (1, 0b100110) => v <= 0, // CMLE
                (0, 0b101010) => v < 0,  // CMLT
                _ => return Ok(false),
            };
            self.fp_set_f64(rd, f64::from_bits(if cond { u64::MAX } else { 0 }));
            return Ok(true);
        }

        // Scalar FP data processing: bits[31:24] = 00011110 (single/double;
        // bit23 = 1 selects half precision, which is out of scope).
        if ((insn >> 24) & 0xFF) != 0b00011110 || ((insn >> 23) & 1) == 1 {
            return Ok(false);
        }
        let double = ((insn >> 22) & 1) == 1;
        let rn = ((insn >> 5) & 0x1F) as u8;
        let rd = (insn & 0x1F) as u8;
        let rm = ((insn >> 16) & 0x1F) as u8;
        // 3-source fused ops: bits[31:24] = 00011111.
        if ((insn >> 24) & 0xFF) == 0b00011111 {
            let ra = ((insn >> 10) & 0x1F) as u8;
            let o3 = (insn >> 15) & 1;
            let o1 = (insn >> 21) & 1;
            // o1/o3 → negate-accumulator / negate-product (QEMU do_fmadd):
            // 00 FMADD, 01 FMSUB, 10 FNMADD, 11 FNMSUB.
            let neg_a = o1 == 1;
            let neg_n = o1 != o3;
            let fa = if double {
                self.fp_get_f64(ra)
            } else {
                self.fp_get_f32(ra) as f64
            };
            let fn_ = if double {
                self.fp_get_f64(rn)
            } else {
                self.fp_get_f32(rn) as f64
            };
            let fm = if double {
                self.fp_get_f64(rm)
            } else {
                self.fp_get_f32(rm) as f64
            };
            let fa = if neg_a { -fa } else { fa };
            let fn_ = if neg_n { -fn_ } else { fn_ };
            let r = fn_ * fm + fa;
            if double {
                self.fp_set_f64(rd, r);
            } else {
                self.fp_set_f32(rd, r as f32);
            }
            return Ok(true);
        }
        // bit21 == 0 → the compare/select encodings (FCSEL/FCCMP live here,
        // distinguished by bits[11:10]).
        if ((insn >> 21) & 1) == 0 {
            let sel_lo = (insn >> 10) & 0b11;
            let cond = ((insn >> 12) & 0xF) as u8;
            if sel_lo == 0b01 {
                // FCCMP: compare, else set nzcv from the immediate.
                let nzcv = (insn & 0xF) as u32;
                if self.condition_holds(cond) {
                    self.fp_cmp(rn, rm, double);
                } else {
                    self.nzcv = nzcv << 28;
                }
                return Ok(true);
            }
            if sel_lo == 0b11 {
                // FCSEL: select Vn or Vm on the condition.
                let v = if self.condition_holds(cond) { rn } else { rm };
                if double {
                    let f = self.fp_get_f64(v);
                    self.fp_set_f64(rd, f);
                } else {
                    let f = self.fp_get_f32(v);
                    self.fp_set_f32(rd, f);
                }
                return Ok(true);
            }
            return Ok(false);
        }
        // bit21 == 1: 1-source (bits[14:10] == 10000), 2-source, FCMP.
        let fixed = (insn >> 10) & 0x3F;
        if fixed == 0b100000 {
            // 1-source: opcode in bits[20:15].
            let op = (insn >> 15) & 0x3F;
            let a = if double { self.fp_get_f64(rn) } else { self.fp_get_f32(rn) as f64 };
            let r = match op {
                1 => a.abs(),
                2 => -a,
                3 => a.sqrt(),
                4 if double => a as f32 as f64, // FCVT Sd←Dn
                4 => a,                          // FCVT Sh←Sn (half, unsupported)
                5 if !double => a as f64,        // FCVT Dd←Sn
                5 => a as f32 as f64,            // FCVT Hd←Dn (half, unsupported)
                8 => a.round_ties_even(),        // FRINTN
                9 => a.ceil(),                   // FRINTP
                10 => a.floor(),                 // FRINTM
                11 => a.trunc(),                 // FRINTZ
                12 => a.round(),                 // FRINTA (ties away)
                14 | 15 => a,                    // FRINTX/I (already rounded)
                _ => return Ok(false),
            };
            if double {
                self.fp_set_f64(rd, r);
            } else if op == 4 && !double {
                // FCVT to half — out of scope.
                return Ok(false);
            } else {
                self.fp_set_f32(rd, r as f32);
            }
            return Ok(true);
        }
        if fixed == 0b001000 {
            // FCMP / FCMPE (with or without zero).
            let e = (insn >> 9) & 1;
            let z = (insn >> 8) & 1;
            let _ = e;
            if z == 1 {
                self.fp_cmp_zero(rn, double);
            } else {
                self.fp_cmp(rn, rm, double);
            }
            return Ok(true);
        }
        // 2-source: opcode in bits[15:11].
        let op = (insn >> 11) & 0x1F;
        let a = if double { self.fp_get_f64(rn) } else { self.fp_get_f32(rn) as f64 };
        let b = if double { self.fp_get_f64(rm) } else { self.fp_get_f32(rm) as f64 };
        let r = match op {
            1 => a * b,    // FMUL
            3 => a / b,    // FDIV
            5 => a + b,    // FADD
            7 => a - b,    // FSUB
            9 => fp_max(a, b),     // FMAX
            11 => fp_min(a, b),    // FMIN
            13 => fp_maxnum(a, b), // FMAXNM
            15 => fp_minnum(a, b), // FMINNM
            17 => -(a * b),        // FNMUL
            _ => return Ok(false),
        };
        if double {
            self.fp_set_f64(rd, r);
        } else {
            self.fp_set_f32(rd, r as f32);
        }
        Ok(true)
    }

    /// Compare two FP values and set NZCV.
    fn fp_cmp(&mut self, rn: u8, rm: u8, double: bool) {
        let a = if double {
            self.fp_get_f64(rn)
        } else {
            self.fp_get_f32(rn) as f64
        };
        let b = if double {
            self.fp_get_f64(rm)
        } else {
            self.fp_get_f32(rm) as f64
        };
        self.set_fp_flags(a, b);
    }

    fn fp_cmp_zero(&mut self, rn: u8, double: bool) {
        let a = if double {
            self.fp_get_f64(rn)
        } else {
            self.fp_get_f32(rn) as f64
        };
        self.set_fp_flags(a, 0.0);
    }

    fn set_fp_flags(&mut self, a: f64, b: f64) {
        let (n, z, c, v) = if a.is_nan() || b.is_nan() {
            (0, 0, 1, 1)
        } else if a < b {
            (1, 0, 0, 0)
        } else if a == b {
            (0, 1, 1, 0)
        } else {
            (0, 0, 1, 0)
        };
        self.nzcv = (n << 31) | (z << 30) | (c << 29) | (v << 28);
    }

    /// Element-wise SIMD binary op over `esize`-bit lanes (little-endian lane
    /// order), `q` selects 128-bit vs 64-bit registers.
    fn simd_elem<F: Fn(u64, u64) -> u64>(&mut self, rd: u8, rn: u8, rm: u8, q: bool, esize: u32, f: F) {
        let lanes = if q { 128 / esize } else { 64 / esize };
        let mask = (1u128 << esize) - 1;
        let a = self.vregs[rn as usize];
        let b = self.vregs[rm as usize];
        let mut out: u128 = 0;
        for i in 0..lanes {
            let av = ((a >> (esize * i)) & mask) as u64;
            let bv = ((b >> (esize * i)) & mask) as u64;
            out |= (f(av, bv) as u128 & mask) << (esize * i);
        }
        self.vregs[rd as usize] = out;
    }

    /// ZIP1/ZIP2/UZP1/UZP2/TRN1/TRN2 over `esize`-bit lanes.
    fn simd_permute(&mut self, rd: u8, rn: u8, rm: u8, q: bool, esize: u32, op: u32) {
        let lanes = if q { 128 / esize } else { 64 / esize };
        let half = lanes / 2;
        let mask = (1u128 << esize) - 1;
        let a = self.vregs[rn as usize];
        let b = self.vregs[rm as usize];
        let get = |r: u128, i: u32| ((r >> (esize * i)) & mask) as u64;
        let mut out: u128 = 0;
        for i in 0..half {
            let (n0, m0) = match op {
                // ZIP1: interleave the low halves; ZIP2: the high halves.
                0b001110 => (get(a, i), get(b, i)),
                0b011110 => (get(a, half + i), get(b, half + i)),
                // UZP1: even lanes; UZP2: odd lanes.
                0b000110 => (get(a, 2 * i), get(b, 2 * i)),
                0b010110 => (get(a, 2 * i + 1), get(b, 2 * i + 1)),
                // TRN1/TRN2: transpose even/odd lanes.
                0b001010 => (get(a, 2 * i), get(b, 2 * i + 1)),
                _ => (get(a, 2 * i + 1), get(b, 2 * i)),
            };
            out |= ((n0 as u128) & mask) << (esize * 2 * i);
            out |= ((m0 as u128) & mask) << (esize * (2 * i + 1));
        }
        self.vregs[rd as usize] = out;
    }

    /// Pairwise SIMD binary op (ADDP/SMAXP/UMAXP): the destination's first
    /// half pairs up Vn's lanes, the second half Vm's.
    fn simd_pairwise<F: Fn(u64, u64) -> u64>(&mut self, rd: u8, rn: u8, rm: u8, q: bool, esize: u32, f: F) {
        let lanes = if q { 128 / esize } else { 64 / esize };
        let half = lanes / 2;
        let mask = (1u128 << esize) - 1;
        let a = self.vregs[rn as usize];
        let b = self.vregs[rm as usize];
        let mut out: u128 = 0;
        for i in 0..half {
            let a0 = ((a >> (esize * 2 * i)) & mask) as u64;
            let a1 = ((a >> (esize * (2 * i + 1))) & mask) as u64;
            let b0 = ((b >> (esize * 2 * i)) & mask) as u64;
            let b1 = ((b >> (esize * (2 * i + 1))) & mask) as u64;
            out |= (f(a0, a1) as u128 & mask) << (esize * i);
            out |= (f(b0, b1) as u128 & mask) << (esize * (i + half));
        }
        self.vregs[rd as usize] = out;
    }
    /// Signed `a >= b` for `bits`-wide lanes.
    fn sge(a: u64, b: u64, bits: u32) -> bool {
        let shift = 64 - bits;
        ((a << shift) as i64) >= ((b << shift) as i64)
    }

    /// Shift-right-and-narrow a vector (SHRN/RSHRN and the saturating forms).
    /// Every `2*dest_esize`-bit lane of `Vn` is shifted right by `shift`
    /// (optionally rounding) and narrowed to `dest_esize` bits; `Q=1` targets
    /// the high half (`SHRN2`), `Q=0` the low half.
    fn simd_shrn(
        &mut self,
        rd: u8,
        rn: u8,
        q: bool,
        dest_esize: u32,
        shift: u32,
        rounding: bool,
        signed_src: bool,
        to_unsigned: bool,
        saturating: bool,
    ) {
        let src_esize = 2 * dest_esize;
        let src_elements = 128 / src_esize;
        let src_mask = (1u128 << src_esize) - 1;
        let dest_mask = (1u128 << dest_esize) - 1;
        let src = self.vregs[rn as usize];
        let round_add = if rounding { 1i64 << (shift - 1) } else { 0 };
        let mut narrowed = [0u64; 16];
        for i in 0..src_elements {
            let raw = ((src >> (src_esize * i)) & src_mask) as u64;
            let shifted = if signed_src {
                let v = (raw as i64) << (64 - src_esize) >> (64 - src_esize);
                v.wrapping_add(round_add) >> shift
            } else {
                (raw.wrapping_add(round_add as u64) >> shift) as i64
            };
            let mut v = shifted;
            if saturating || to_unsigned {
                let (min, max) = if to_unsigned || !signed_src {
                    (0i64, dest_mask as i64)
                } else {
                    (-(1i64 << (dest_esize - 1)), (1i64 << (dest_esize - 1)) - 1)
                };
                v = v.clamp(min, max);
            }
            narrowed[i as usize] = (v as u64) & (dest_mask as u64);
        }
        let mut out: u128 = 0;
        for i in 0..src_elements {
            out |= (narrowed[i as usize] as u128) << (dest_esize * i);
        }
        if q {
            self.vregs[rd as usize] = (self.vregs[rd as usize] & ((1u128 << 64) - 1)) | (out << 64);
        } else {
            self.vregs[rd as usize] = out & ((1u128 << 64) - 1);
        }
    }

    /// Scan the TLS IPC buffer for the "SFCI" request-header magic and return
    /// the command id that follows it (`CmifInHeader::command_id`). Returns
    /// `None` when the buffer doesn't look like a CMIF request (the domain and
    /// non-domain layouts place the header at different offsets, so we search
    /// rather than hard-code one).
    fn ipc_command_id(&self, tls: u32) -> Option<u32> {
        for i in 0..0x40u32 {
            if self.mem.read_u32(tls.wrapping_add(i)).ok()? == 0x4943_4653 {
                return self.mem.read_u32(tls.wrapping_add(i + 8)).ok();
            }
        }
        // Older libnx (pre-CMIF) sessions — e.g. the FsDir object NX-Shell's
        // `fsDirRead` uses — marshal requests as {type=2, object_id, cmd_id,
        // ...} with no SFCI magic. Fall back to reading the command there.
        let start = self.ipc_reply_start(tls);
        if self.mem.read_u32(tls.wrapping_add(start)).unwrap_or(0) == 2 {
            return self.mem.read_u32(tls.wrapping_add(start + 8)).ok();
        }
        None
    }

    /// Compute where the reply starts in the TLS IPC buffer, mirroring libnx's
    /// `cmifGetAlignedDataStart`: walk the request's hipc header (16-byte
    /// message header, optional special header + pid, then buffer descriptors)
    /// to the data area, and round up to 16 bytes.
    fn ipc_reply_start(&self, tls: u32) -> u32 {
        let hdr1 = self.mem.read_u32(tls).unwrap_or(0);
        let hdr2 = self.mem.read_u32(tls.wrapping_add(4)).unwrap_or(0);
        let num_send_statics = (hdr1 >> 16) & 0xf;
        let num_send_buffers = (hdr1 >> 20) & 0xf;
        let num_recv_buffers = (hdr1 >> 24) & 0xf;
        let num_exch_buffers = (hdr1 >> 28) & 0xf;
        let has_special = (hdr2 >> 31) & 1;
        let mut data_off = 8u32;
        if has_special != 0 {
            data_off += 4;
            let sphdr = self.mem.read_u32(tls.wrapping_add(8)).unwrap_or(0);
            if sphdr & 1 != 0 {
                data_off += 8; // pid
            }
        }
        data_off += 8 * num_send_statics;
        data_off += 12 * (num_send_buffers + num_recv_buffers + num_exch_buffers);
        (data_off + 15) & !15
    }

    fn alloc_handle(&mut self) -> u64 {
        let h = self.next_handle as u64;
        self.next_handle = self.next_handle.wrapping_add(1);
        h
    }

    /// Address of the `index`-th receive buffer in a hipc request, from its
    /// buffer descriptors (walk the same layout as [`Cpu::ipc_reply_start`]).
    /// Returns `None` if the request has no such buffer.
    fn ipc_recv_buffer_addr(&self, tls: u32, index: u32) -> Option<u32> {
        let hdr1 = self.mem.read_u32(tls).unwrap_or(0);
        let hdr2 = self.mem.read_u32(tls.wrapping_add(4)).unwrap_or(0);
        let num_send_statics = (hdr1 >> 16) & 0xf;
        let num_send_buffers = (hdr1 >> 20) & 0xf;
        let num_recv_buffers = (hdr1 >> 24) & 0xf;
        let _num_exch_buffers = (hdr1 >> 28) & 0xf;
        if index >= num_recv_buffers {
            return None;
        }
        let has_special = (hdr2 >> 31) & 1;
        let mut off = 8u32;
        if has_special != 0 {
            off += 4;
            let sphdr = self.mem.read_u32(tls.wrapping_add(8)).unwrap_or(0);
            if sphdr & 1 != 0 {
                off += 8; // pid
            }
        }
        off += 8 * num_send_statics;
        off += 12 * (num_send_buffers + index);
        let address_low = self.mem.read_u32(tls.wrapping_add(off + 4)).unwrap_or(0);
        let desc = self.mem.read_u32(tls.wrapping_add(off + 8)).unwrap_or(0);
        let addr_mid = (desc >> 4) & 0xf;
        let addr_high = (desc >> 6) & 0x3F_FFFF;
        let addr = (address_low as u64) | ((addr_mid as u64) << 32) | ((addr_high as u64) << 36);
        Some(addr as u32)
    }

    fn record_handle(&mut self, handle: u64, name: &str) {
        self.service_handles.insert(handle, name.to_owned());
    }

    fn service_name(&self, handle: u64) -> Option<&str> {
        self.service_handles.get(&handle).map(|s| s.as_str())
    }

    fn read_port_name(&self, ptr: u32) -> String {
        let mut name = Vec::new();
        for i in 0..16u32 {
            match self.mem.read_u8(ptr.wrapping_add(i)) {
                Ok(0) | Err(_) => break,
                Ok(b) => name.push(b),
            }
        }
        String::from_utf8_lossy(&name).into_owned()
    }

    fn u64_to_service_name(&self, value: u64) -> String {
        let bytes = value.to_le_bytes();
        let len = bytes.iter().position(|&b| b == 0).unwrap_or(8);
        String::from_utf8_lossy(&bytes[..len]).into_owned()
    }

    fn ipc_message_type(&self, tls: u32) -> u32 {
        self.mem.read_u32(tls).unwrap_or(0) & 0xFFFF
    }

    /// Whether the request is a domain message. Domain-ness is NOT encoded in
    /// the hipc type field: a domain request still carries type 4
    /// (`CmifCommandType_Request`), and the domain header (`CmifDomainInHeader`)
    /// lives at the start of the data area with its `type` byte set to
    /// `CmifDomainRequestType_SendMessage` (1). Reading it from where
    /// [`Cpu::ipc_reply_start`] puts the data area avoids misfiring on plain
    /// requests that happen to have a nonzero word in the same spot.
    fn ipc_is_domain_request(&self, tls: u32) -> bool {
        let start = self.ipc_reply_start(tls);
        self.mem.read_u8(tls.wrapping_add(start)).unwrap_or(0) == 1
    }

    fn ipc_domain_object_id(&self, tls: u32) -> u32 {
        let start = self.ipc_reply_start(tls);
        self.mem.read_u32(tls.wrapping_add(start + 4)).unwrap_or(0xFFFFFFFF)
    }

    fn alloc_domain_object(&mut self) -> u32 {
        let id = self.next_domain_object_id;
        self.next_domain_object_id = id.wrapping_add(1);
        if id == 0 { 1 } else { id }
    }

    fn record_domain_object(&mut self, handle: u64, object_id: u32, name: &str) {
        self.domain_objects.insert((handle, object_id), name.to_owned());
    }

    fn domain_interface(&self, handle: u64, object_id: u32) -> Option<&str> {
        self.domain_objects.get(&(handle, object_id)).map(|s| s.as_str())
    }

    /// Write a complete HIPC response into the TLS IPC buffer. `move_handles`
    /// are emitted in the handle descriptor, `raw_data` lands after the
    /// SFCO/result header, and `domain_objects` produces a domain response
    /// (message type 4) with the given out-object ids.
    fn write_ipc_response(
        &mut self,
        tls: u32,
        result: u32,
        move_handles: &[u64],
        raw_data: &[u8],
        domain_objects: &[u32],
    ) -> Result<()> {
        let is_domain = self.ipc_is_domain_request(tls);
        // Real Horizon reply types: 0x40 (Reply) and 0x44 (DomainReply).
        let msg_type = if is_domain { 0x44u32 } else { 0x40 };
        self.mem.write_u32(tls, msg_type)?;
        let has_handles = !move_handles.is_empty();
        let handle_desc = (move_handles.len() as u32) << 5;
        let raw_data_words = ((raw_data.len() as u32) + 3) / 4;
        let object_words = ((domain_objects.len() as u32) * 4 + 3) / 4;
        // SFCO header (4 words) + raw data + domain header/objects when needed,
        // padded so pre+post = 4 words.
        let mut raw_section_words = 4 + raw_data_words + 4;
        if is_domain {
            raw_section_words += 4 + object_words;
        }
        let mut header1 = raw_section_words;
        if has_handles {
            header1 |= 1 << 31;
        }
        self.mem.write_u32(tls.wrapping_add(4), header1)?;
        let mut off = 8u32;
        if has_handles {
            self.mem.write_u32(tls.wrapping_add(off), handle_desc)?;
            off += 4;
            for &h in move_handles {
                self.mem.write_u32(tls.wrapping_add(off), h as u32)?;
                off += 4;
            }
        }
        // Align to 16 bytes.
        let pre = (16 - (off % 16)) % 16;
        off += pre;
        if is_domain {
            self.mem.write_u32(tls.wrapping_add(off), domain_objects.len() as u32)?;
            self.mem.write_u32(tls.wrapping_add(off + 4), 0)?;
            self.mem.write_u32(tls.wrapping_add(off + 8), 0)?;
            self.mem.write_u32(tls.wrapping_add(off + 12), 0)?;
            off += 16;
        }
        // SFCO header.
        self.mem.write_u32(tls.wrapping_add(off), 0x4F43_4653)?;
        self.mem.write_u32(tls.wrapping_add(off + 4), 0)?;
        self.mem.write_u32(tls.wrapping_add(off + 8), result)?;
        self.mem.write_u32(tls.wrapping_add(off + 12), 0)?;
        off += 16;
        // Raw data.
        for (i, &b) in raw_data.iter().enumerate() {
            self.mem.write_u8(tls.wrapping_add(off + i as u32), b)?;
        }
        off += raw_data_words * 4;
        // Domain object ids.
        for (i, &obj) in domain_objects.iter().enumerate() {
            self.mem.write_u32(tls.wrapping_add(off + (i as u32) * 4), obj)?;
        }
        Ok(())
    }

    fn stub_sm(&mut self, tls: u32, cmd_id: Option<u32>, _handle: u64) -> Result<()> {
        match cmd_id {
            Some(0) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            Some(1) => {
                // GetService: raw data is the 8-byte service name.
                let start = self.ipc_reply_start(tls);
                let name_raw = self.mem.read_u64(tls.wrapping_add(start + 0x10)).unwrap_or(0);
                let name = self.u64_to_service_name(name_raw);
                let handle = self.alloc_handle();
                self.record_handle(handle, &name);
                self.write_ipc_response(tls, 0, &[handle], &[], &[])
            }
            Some(2) => {
                let handle = self.alloc_handle();
                self.write_ipc_response(tls, 0, &[handle], &[], &[])
            }
            Some(3) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    fn stub_fsp_srv(&mut self, tls: u32, cmd_id: Option<u32>, handle: u64) -> Result<()> {
        match cmd_id {
            // 0 = ConvertToDomain: hand back a fresh domain object id so the
            // session becomes a domain (libnx's serviceConvertToDomain reads it
            // from the out data). All later fsp-srv requests then carry the
            // object id in the CmifDomainInHeader.
            Some(0) => {
                let obj = self.alloc_domain_object();
                self.record_domain_object(handle, obj, "fsp-srv");
                self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
            }
            Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // 18 = fsOpenSdCardFileSystem, 11 = fsOpenBisFileSystem: both hand
            // out an FsFileSystem session as a domain out-object.
            Some(18) | Some(11) => {
                let obj = self.alloc_domain_object();
                self.record_domain_object(handle, obj, "fsp-srv-fs");
                self.write_ipc_response(tls, 0, &[], &[], &[obj])
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// fsp-srv FsFileSystem object: `fsFsOpenDirectory` (9) hands out an FsDir
    /// object backed by the synthetic SD card; everything else answers success
    /// so the file-manager's setup checks pass.
    fn stub_fs(&mut self, tls: u32, cmd_id: Option<u32>, handle: u64) -> Result<()> {
        if std::env::var("TRACE_IPC").is_ok() {
            eprintln!("[fs] pc={:#x} cmd={:?} h={}", self.pc, cmd_id, handle);
        }
        match cmd_id {
            Some(9) => {
                let obj = self.alloc_domain_object();
                self.record_domain_object(handle, obj, "fsp-srv-fs-dir");
                let entries = self.synthetic_sd_root();
                self.fs_dirs.insert(obj as u64, entries);
                self.write_ipc_response(tls, 0, &[], &[], &[obj])
            }
            // GetEntryType: report the entry as a directory so `stat` on the
            // SD paths succeeds and the file manager stops re-creating them.
            Some(7) => {
                let ty = 0u64; // FsDirEntryType_Dir
                self.write_ipc_response(tls, 0, &[], &ty.to_le_bytes(), &[])
            }
            // The remaining fs object commands (CreateFile/DeleteFile/
            // CreateDirectory/OpenFile/...) answer success so the setup
            // sequence completes without a concrete backing store.
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// A small stand-in SD card so `FS::GetDirList` (opendir/readdir over
    /// fsdev) sees real entries and NX-Shell's `main` reaches its render loop
    /// instead of returning early. `(name, type, file_size)` with type 0 = dir,
    /// 1 = file.
    fn synthetic_sd_root(&self) -> Vec<(Vec<u8>, u8, u64)> {
        let mut v = Vec::new();
        for (name, ty, size) in [
            ("Switch", 0u8, 0u64),
            ("hbmenu.nro", 1u8, 0x120_000),
            ("switch.nro", 1u8, 0x200_000),
            ("README.txt", 1u8, 0x40),
            ("sdmc", 0u8, 0),
        ] {
            v.push((name.as_bytes().to_vec(), ty, size));
        }
        v
    }

    /// fsp-srv FsDir object: cmd 0 = fsDirRead (fill the out buffer with
    /// `FsDirectoryEntry` structs), cmd 1 = fsDirGetEntryCount.
    fn stub_fs_dir(&mut self, tls: u32, cmd_id: Option<u32>, object_id: u32) -> Result<()> {
        if std::env::var("TRACE_IPC").is_ok() {
            eprintln!("[dir] pc={:#x} cmd={:?} obj={}", self.pc, cmd_id, object_id);
        }
        match cmd_id {
            Some(0) => {
                let remaining = self.fs_dirs.entry(object_id as u64).or_default();
                let to_yield = remaining.clone();
                remaining.clear();
                if let Some(buf) = self.ipc_recv_buffer_addr(tls, 0) {
                    for (i, (name, ty, size)) in to_yield.iter().enumerate() {
                        let off = buf.wrapping_add(i as u32 * 0x310);
                        for j in 0..0x301u32 {
                            let b = if (j as usize) < name.len() { name[j as usize] } else { 0 };
                            let _ = self.mem.write_u8(off.wrapping_add(j), b);
                        }
                        // pad[3] (offsets 0x301..0x304) stays zero.
                        let _ = self.mem.write_u8(off.wrapping_add(0x304), *ty);
                        // pad2[3] stays zero; file_size at 0x308 (8-aligned).
                        let _ = self.mem.write_u64(off.wrapping_add(0x308), *size);
                    }
                }
                let n = to_yield.len() as u64;
                self.write_ipc_response(tls, 0, &[], &n.to_le_bytes(), &[])
            }
            Some(1) => {
                let n = self.fs_dirs.get(&(object_id as u64)).map(|v| v.len() as u64).unwrap_or(0);
                self.write_ipc_response(tls, 0, &[], &n.to_le_bytes(), &[])
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    fn stub_vi(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        let req_type = self.ipc_message_type(tls);
        // Control (type 5) requests: cmd 0 = ConvertToDomain, cmd 3 =
        // QueryPointerBufferSize. Older libnx always converts the session to a
        // domain before dispatching; hbmenu's libnx (NX_SERVICE_ASSUME_NON_DOMAIN)
        // instead sends cmd 3 and then uses raw non-domain requests.
        if req_type == 5 {
            return match cmd_id {
                Some(0) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "vi:root");
                    let raw = obj.to_le_bytes();
                    self.write_ipc_response(tls, 0, &[], &raw, &[])
                }
                // QueryPointerBufferSize: the server pointer buffer size (u16).
                Some(3) => self.write_ipc_response(tls, 0, &[], &0x200u16.to_le_bytes(), &[]),
                _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
            };
        }
        let object_id = if self.ipc_is_domain_request(tls) {
            self.ipc_domain_object_id(tls)
        } else {
            0xFFFFFFFF
        };
        let is_domain = object_id != 0xFFFFFFFF;
        if !is_domain {
            // Non-domain (NX_SERVICE_ASSUME_NON_DOMAIN) sessions marshal output
            // objects as move handles. Dispatch on the sub-interface (tracked per
            // handle); unknown handles default to the vi root.
            let iface = self.vi_ifaces.get(&handle).map(|s| s.as_str()).unwrap_or("vi:root");
            match iface {
                // IHOSBinderDriverRelay: libnx binder protocol — AdjustRefcount
                // (1), GetNativeHandle (2), TransactParcel (3).
                "vi:ihosbd" => match cmd_id {
                    Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                    Some(2) => {
                        let h = self.alloc_handle();
                        self.write_ipc_response(tls, 0, &[h], &[], &[])
                    }
                    Some(3) => self.vi_transact_parcel(tls),
                    _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
                },
                // vi root: cmd 2 hands out the IApplicationDisplayService.
                "vi:root" => match cmd_id {
                    Some(2) => self.vi_out_session(tls, "vi:iads"),
                    _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
                },
                // IApplicationDisplayService and the other display services.
                _ => match cmd_id {
                    Some(100) => self.vi_out_session(tls, "vi:ihosbd"),
                    Some(101) => self.vi_out_session(tls, "vi:isds"),
                    Some(102) => self.vi_out_session(tls, "vi:imds"),
                    Some(103) => self.vi_out_session(tls, "vi:ihosbdind"),
                    Some(1010) => self.write_ipc_response(tls, 0, &[], &1u64.to_le_bytes(), &[]),
                    Some(5202) => {
                        let h = self.alloc_handle();
                        self.write_ipc_response(tls, 0, &[h], &[], &[])
                    }
                    // _viOpenLayer (2020) / _viCreateStrayLayer (2030 / 2012 / 2312):
                    // fill the native-window receive buffer with a Binder parcel whose
                    // payload[2] is the IGraphicBufferProducer binder id, and return the
                    // parcel size. viCreateLayer parses exactly that.
                    Some(2020) => self.vi_native_window(tls, 8),
                    Some(2030) | Some(2012) | Some(2312) => self.vi_native_window(tls, 16),
                    _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
                },
            }
        } else {
            match self.domain_interface(handle, object_id) {
                Some("vi:root") => match cmd_id {
                    Some(2) => {
                        let obj = self.alloc_domain_object();
                        self.record_domain_object(handle, obj, "vi:iads");
                        self.write_ipc_response(tls, 0, &[], &[], &[obj])
                    }
                    _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
                },
                Some("vi:iads") => match cmd_id {
                    Some(100) => {
                        let obj = self.alloc_domain_object();
                        self.record_domain_object(handle, obj, "vi:ihosbd");
                        self.write_ipc_response(tls, 0, &[], &[], &[obj])
                    }
                    Some(101) => {
                        let obj = self.alloc_domain_object();
                        self.record_domain_object(handle, obj, "vi:isds");
                        self.write_ipc_response(tls, 0, &[], &[], &[obj])
                    }
                    Some(102) => {
                        let obj = self.alloc_domain_object();
                        self.record_domain_object(handle, obj, "vi:imds");
                        self.write_ipc_response(tls, 0, &[], &[], &[obj])
                    }
                    // OpenDisplay: return a display id of 1.
                    Some(1010) => {
                        let raw = 1u64.to_le_bytes();
                        self.write_ipc_response(tls, 0, &[], &raw, &[])
                    }
                    // GetDisplayVsyncEvent: return a copy handle.
                    Some(5202) => {
                        let h = self.alloc_handle();
                        self.write_ipc_response(tls, 0, &[h], &[], &[])
                    }
                    _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
                },
                Some("vi:ihosbd") | Some("vi:isds") | Some("vi:imds") => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
            }
        }
    }

    /// Non-domain vi session hand-out: allocate a fresh handle, record it as a
    /// vi session so later SendSyncRequests route back here, and return it as a
    /// move handle (how NX_SERVICE_ASSUME_NON_DOMAIN marshals output objects).
    fn vi_out_session(&mut self, tls: u32, iface: &str) -> Result<()> {
        let h = self.alloc_handle();
        self.record_handle(h, "vi:m");
        self.vi_ifaces.insert(h, iface.to_owned());
        self.write_ipc_response(tls, 0, &[h], &[], &[])
    }

    /// IHOSBinderDriverRelay TransactParcel: fill the out-buffer with a reply
    /// parcel. The transaction code lives at `ipc_reply_start + 0x14` (the
    /// {session_id, code, flags} request struct). bqConnect (code 0xA) expects a
    /// BqBufferOutput {width, height, transform, numPendingBuffers} followed by
    /// a binder error code; other transactions get an empty reply parcel.
    fn vi_transact_parcel(&mut self, tls: u32) -> Result<()> {
        let start = self.ipc_reply_start(tls);
        let code = self.mem.read_u32(tls.wrapping_add(start + 0x14)).unwrap_or(0);
        let mut payload: Vec<u8> = Vec::new();
        if code == 0xA {
            // CONNECT: BqBufferOutput + binder error (0 = success).
            for v in [1280u32, 720, 0, 0, 0] {
                payload.extend_from_slice(&v.to_le_bytes());
            }
        }
        let mut parcel = Vec::with_capacity(16 + payload.len());
        parcel.extend_from_slice(&(payload.len() as u32).to_le_bytes()); // payload_size
        parcel.extend_from_slice(&16u32.to_le_bytes()); // payload_off
        parcel.extend_from_slice(&0u32.to_le_bytes()); // objects_size
        parcel.extend_from_slice(&((16 + payload.len()) as u32).to_le_bytes()); // objects_off
        parcel.extend_from_slice(&payload);
        if let Some(buf) = self.ipc_recv_buffer_addr(tls, 0) {
            for (i, &b) in parcel.iter().enumerate() {
                let _ = self.mem.write_u8(buf.wrapping_add(i as u32), b);
            }
        }
        self.write_ipc_response(tls, 0, &[], &[], &[])
    }

    /// viOpenLayer / viCreateStrayLayer reply: fill the request's native-window
    /// receive buffer with a Binder parcel (ParcelHeader + payload whose third
    /// word is the IGraphicBufferProducer binder object id), then return the
    /// parcel size. `out_size` is the number of reply data words (8 for 2020's
    /// single u64, 16 for 2030's layer_id+size pair).
    fn vi_native_window(&mut self, tls: u32, out_size: usize) -> Result<()> {
        let payload: [u8; 12] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]; // payload[2] = binder id 1
        let parcel_size = 16 + payload.len() as u32;
        let mut parcel = Vec::with_capacity(parcel_size as usize);
        parcel.extend_from_slice(&payload.len().to_le_bytes()); // payload_size
        parcel.extend_from_slice(&16u32.to_le_bytes()); // payload_off
        parcel.extend_from_slice(&0u32.to_le_bytes()); // objects_size
        parcel.extend_from_slice(&parcel_size.to_le_bytes()); // objects_off
        parcel.extend_from_slice(&payload);

        let mut raw = Vec::with_capacity(out_size);
        if out_size >= 16 {
            // 2030: { layer_id, native_window_size }
            raw.extend_from_slice(&1u64.to_le_bytes());
            raw.extend_from_slice(&(parcel_size as u64).to_le_bytes());
        } else {
            // 2020: native_window_size
            raw.extend_from_slice(&(parcel_size as u64).to_le_bytes());
        }

        if let Some(buf) = self.ipc_recv_buffer_addr(tls, 0) {
            for (i, &b) in parcel.iter().enumerate() {
                let _ = self.mem.write_u8(buf.wrapping_add(i as u32), b);
            }
        }
        self.write_ipc_response(tls, 0, &[], &raw, &[])
    }

    fn stub_nvdrv(&mut self, tls: u32, cmd_id: Option<u32>, _handle: u64) -> Result<()> {
        match cmd_id {
            // Open: fd, error
            Some(0) => {
                let mut raw = [0u8; 8];
                raw[..4].copy_from_slice(&1i32.to_le_bytes());
                self.write_ipc_response(tls, 0, &[], &raw, &[])
            }
            // Ioctl: result
            Some(1) | Some(2) => self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[]),
            // Initialize: just the result (old libnx CMIF replies carry no raw
            // data; a 4-byte payload shifts its reply parser and makes
            // nvInitialize bail before nvGpuInit/nvOpen).
            Some(3) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    fn syscall(&mut self, imm: u16) -> Result<()> {
        match self.syscall_mode {
            SyscallMode::None => {                if imm == 0 {
                    self.halted = true;
                    Ok(())
                } else {
                    Err(Error::Cpu(format!("unimplemented syscall #{}", imm)))
                }
            }
            SyscallMode::Horizon => self.horizon_syscall(imm),
        }
    }

    /// Permissive stubs for the Horizon syscall numbers libnx homebrew hits
    /// during startup and normal single-threaded operation. The syscall
    /// numbers follow the real Switch ABI as emitted by libnx
    /// (`nx/source/kernel/svc.s`). There are no real services or threads, so
    /// service/IPC calls return success with a fake handle and waits complete
    /// immediately; this lets the app's `main()` run as far as it can before
    /// it needs real hardware.
    ///
    /// Results follow the real ABI: X0 carries the Result (success is 0,
    /// errors have bit 31 set), out-handles come back in X1, and
    /// value-returning syscalls put their result in X1 so the libnx wrapper
    /// (`str x0; svc; ldr x2; str x1, [x2]`) stores it into the caller's out
    /// pointer.
    fn horizon_syscall(&mut self, imm: u16) -> Result<()> {
        const RESULT_OK: u64 = 0;
        // Non-zero handle handed out by handle-returning syscalls (libnx
        // stores X1 into the caller's output pointer).
        const FAKE_HANDLE: u64 = 0x1000;
        match imm {
            0x01 => {
                // SetHeapSize: report a heap at a soft-mapped address.
                self.write_zr(0, RESULT_OK);
                self.write_zr(1, 0x3000_0000);
                Ok(())
            }
            0x02 | 0x03 | 0x04 | 0x14 => {
                // SetMemoryPermission / SetMemoryAttribute / MapMemory /
                // UnmapSharedMemory
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x13 => {
                // MapSharedMemory(handle, addr, size, perm): libnx maps the
                // hid service's shared memory this way; back it with a real
                // zeroed buffer and remember where so the host can write
                // gamepad state into the HidSharedMemory layout.
                let addr = self.read_zr(1) as u32;
                let size = self.read_zr(2) as u32;
                self.mem.map_zero(addr, size as usize)?;
                self.hid_shmem_addr = addr;
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x05 => {
                // UnmapMemory(addr, size). hbmenu detects the process address
                // space by unmapping the very top of the 64-bit range and
                // reading the failure code: an out-of-range unmap returns a
                // kernel error whose low bits are 0xd401 (39-bit AArch64) or
                // 0xdc01 (36-bit). Report 39-bit; anything in-range is a no-op
                // success.
                let addr = self.read_zr(0);
                if (addr >> 48) == 0xFFFF {
                    self.write_zr(0, 0x8000_D401);
                } else {
                    self.write_zr(0, RESULT_OK);
                }
                Ok(())
            }
             0x06 => {
                 // QueryMemory(info, pageInfo, addr): report the contiguous
                 // run of pages in the same state (allocated vs untouched) as
                 // the queried page. Real pages (image, stack, heap, anything
                 // the app has written to) come back as RWX; untouched
                 // soft-mapped pages come back as unmapped so libnx virtmem
                 // address-space walks and reservations see free space. The
                 // old stub reported the whole low 2 GiB as one RWX region,
                 // which made deko3d's AS reservation fail.
                 let out = self.read_zr(0) as u32;
                 let addr = self.read_zr(2) as u32;
                 let state = |a: u32| self.mem.page_mapped(a & !0xFFF);
                 let mut base = addr & !0xFFF;
                 while base >= 0x1000 && state(base - 0x1000) == state(base) {
                     base -= 0x1000;
                 }
                 let mapped = state(base);
                 let mut end = base + 0x1000;
                 while end < 0x8000_0000 && state(end) == mapped {
                     end += 0x1000;
                 }
                 let mut info = Vec::with_capacity(40);
                 info.extend_from_slice(&(base as u64).to_le_bytes());
                 info.extend_from_slice(&((end - base) as u64).to_le_bytes());
                 for v in [
                     if mapped { 3u32 } else { 0 }, // type (CodeStatic / Unmapped)
                     0,
                     if mapped { 0b111 } else { 0 }, // perm (RWX / none)
                     0,
                     0,
                     0,
                 ] {
                     info.extend_from_slice(&v.to_le_bytes());
                 }
                 for (i, &b) in info.iter().enumerate() {
                     self.mem.write_u8(out.wrapping_add(i as u32), b)?;
                 }
                 self.write_zr(0, RESULT_OK);
                 self.write_zr(1, if mapped { 0x1000 } else { 0 }); // page info
                 Ok(())
             }
            0x07 | 0x0A => {
                // ExitProcess / ExitThread
                self.halted = true;
                Ok(())
            }
            0x08 => {
                // CreateThread: hand out a fake handle; StartThread is a no-op
                // so the main thread keeps running and waits "complete".
                self.write_zr(0, RESULT_OK);
                self.write_zr(1, FAKE_HANDLE);
                Ok(())
            }
            0x09 | 0x0B | 0x0C | 0x0D | 0x0E | 0x0F | 0x16 | 0x17 | 0x19 | 0x1A
            | 0x1B | 0x1C | 0x1D | 0x28 => {
                // StartThread / SleepThread / get-set thread priority+core
                // mask / CloseHandle / ResetSignal / CancelSynchronization /
                // ArbitrateLock+Unlock / Wait+SignalProcessWideKey /
                // ReturnFromException
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x10 => {
                // GetCurrentProcessorNumber
                self.write_zr(0, 0);
                Ok(())
            }
            0x11 | 0x12 => {
                // SignalEvent / ClearEvent
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x15 => {
                // CreateTransferMemory
                self.write_zr(0, RESULT_OK);
                self.write_zr(1, FAKE_HANDLE);
                Ok(())
            }
            0x18 => {
                // WaitSynchronization: report the wait as immediately satisfied.
                // X1 is the "number of signaled handles" that the libnx wrapper
                // stores to the caller's out pointer; deko3d indexes its waiter
                // array by it, so garbage here makes the fence wait retry forever.
                self.write_zr(0, RESULT_OK);
                self.write_zr(1, 1);
                Ok(())
            }
            0x1E => {
                // GetSystemTick (ns scale, arbitrary)
                self.write_zr(0, self.cycles * 1000);
                Ok(())
            }
            0x1F => {
                // ConnectToNamedPort: read the port name so later IPC can be
                // dispatched to the right stub service.
                let name_ptr = self.read_zr(1) as u32;
                let name = if name_ptr != 0 {
                    self.read_port_name(name_ptr)
                } else {
                    String::new()
                };
                let handle = self.alloc_handle();
                self.record_handle(handle, &name);
                self.write_zr(0, RESULT_OK);
                self.write_zr(1, handle);
                Ok(())
            }
            0x20 | 0x21 | 0x22 | 0x23 => {
                // SendSyncRequest[Light|WithUserBuffer] / async variant.
                // If we recognize the target handle as a named service, dispatch
                // to a small stub implementation. Otherwise fall back to the
                // libnx applet-style generic reply so hbmenu/applet init still
                // progresses.
                let tls = self.tpidr as u32;
                let handle = self.read_zr(0);
                let cmd_id = self.ipc_command_id(tls);
                let svc_name = self.service_name(handle).map(|s| s.to_string());
                if std::env::var("TRACE_IPC").is_ok() {
                    eprintln!(
                        "[ipc] pc={:#x} h={} svc={:?} cmd={:?}",
                        self.pc, handle, svc_name, cmd_id
                    );
                }
                if let Some(name) = svc_name {
                    let name = name;
                    match name.as_str() {
                        "sm:" | "sm" => self.stub_sm(tls, cmd_id, handle)?,
                        "fsp-srv" | "fsp-srv:" => {
                            // fsp-srv converts to a domain, so the fs/dir
                            // sub-sessions come in as object ids on the same
                            // session handle. Route on the recorded object
                            // interface; unknown objects hit the root stub.
                            let object_id = self.ipc_domain_object_id(tls);
                            match self.domain_interface(handle, object_id) {
                                Some("fsp-srv-fs") => self.stub_fs(tls, cmd_id, handle)?,
                                Some("fsp-srv-fs-dir") => {
                                    self.stub_fs_dir(tls, cmd_id, object_id)?
                                }
                                _ => self.stub_fsp_srv(tls, cmd_id, handle)?,
                            }
                        }
                        "vi:m" | "vi:m:" => self.stub_vi(tls, handle, cmd_id)?,
                        "nvdrv" | "nvdrv:" | "nvdrv:a" | "nvdrv:a:" => {
                            self.stub_nvdrv(tls, cmd_id, handle)?
                        }
                         _ => {
                             // Known service, no dedicated stub: answer like
                             // the anonymous-handle fallback below so applet
                             // state commands (ReceiveMessage → the applet
                             // message, Get*Mode/GetCurrentFocusState → the
                             // state value) return real data instead of an
                             // empty reply.
                             let data = match cmd_id {
                                 Some(1) => 15, // ReceiveMessage → FocusStateChanged
                                 Some(5) => 1,  // GetOperationMode → Handheld
                                 Some(6) => 0,  // GetPerformanceMode → Normal
                                 Some(9) => 1,  // GetCurrentFocusState → InFocus
                                 _ => {
                                     let obj = self.next_object_id;
                                     self.next_object_id = obj.wrapping_add(1);
                                     obj
                                 }
                             };
                             self.write_ipc_response(tls, 0, &[], &data.to_le_bytes(), &[])?
                         }
                    }
                } else {
                    // Unrecognized session handle. The display service's session
                    // handles come from generic object-id replies and aren't in
                    // service_handles, so try the vi stub first; fall back to the
                    // old applet-style generic reply if the request isn't a vi
                    // command (e.g. hid/time sessions).
                    if let Some(cmd) = cmd_id {
                        // vi commands: GetIApplicationDisplayService (2) and the
                        // display/session commands (100+). The generic reply already
                        // answers ConvertToDomain (0) with a valid object id, and the
                        // small applet state commands (1 = ReceiveMessage, 5/6/9)
                        // must keep the generic reply.
                        if cmd == 2 || cmd >= 100 {
                            return self.stub_vi(tls, handle, cmd_id);
                        }
                    }
                    let start = self.ipc_reply_start(tls);
                    let is_domain = self
                        .mem
                        .read_u32(tls.wrapping_add(start + 0x10))
                        .unwrap_or(0)
                        == 0x4943_4653;
                    let data = match cmd_id {
                        Some(1) => 15,
                        Some(5) => 1,
                        Some(6) => 0,
                        Some(9) => 1,
                        _ => {
                            let obj = self.next_object_id;
                            self.next_object_id = obj.wrapping_add(1);
                            obj
                        }
                    };
                    if is_domain {
                        for i in 0..4u32 {
                            let _ = self.mem.write_u32(tls.wrapping_add(start + i * 4), 0);
                        }
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x10), 0x4F43_4653);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x14), 0);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x18), 0);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x1C), 0);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x20), data);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x24), 0);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x28), data);
                    } else {
                        let _ = self.mem.write_u32(tls.wrapping_add(start), 0x4F43_4653);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x04), 0);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x08), 0);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x0C), 0);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x10), data);
                    }
                }
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x24 => {
                // GetProcessId
                self.write_zr(0, 1);
                Ok(())
            }
            0x25 => {
                // GetThreadId
                self.write_zr(0, 1);
                Ok(())
            }
            0x26 => {
                // Break: fatal debugger trap — surface and stop.
                self.out.extend_from_slice(b"[svcBreak]\n");
                self.halted = true;
                Ok(())
            }
            0x27 => {
                // OutputDebugString(ptr, size) — log to the console.
                let ptr = self.read_zr(0) as u32;
                let len = (self.read_zr(1) as i64).clamp(0, 4096) as u32;
                if ptr != 0 && len > 0 {
                    for i in 0..len {
                        match self.mem.read_u8(ptr.wrapping_add(i)) {
                            Ok(b) => self.out.push(b),
                            Err(_) => break,
                        }
                    }
                }
                Ok(())
            }
             0x29 => {
                 // GetInfo(out, infoType, handle, infoSubValue): report the
                 // value in X1 (the libnx wrapper stores it to the out
                 // pointer). The InfoType numbering here matches the libnx
                 // build hbmenu is compiled against: 2/3 Alias, 4/5 Heap,
                 // 6/7 Total/Used memory, 12/13 Aslr, 14/15 Stack.
                 let info_type = self.read_zr(1);
                 let value = match info_type {
                     2 => 0x0000_0010_0000_0000, // AliasRegionAddress
                     3 => 0x0000_0000_2000_0000, // AliasRegionSize
                     4 => 0x0000_0002_0000_0000, // HeapRegionAddress
                     5 => 0x0000_0000_2000_0000, // HeapRegionSize
                     6 => 0x1E00_0000, // TotalMemorySize
                     7 => 0,         // UsedMemorySize
                     8 => 0,         // DebuggerAttached
                     9 => 0,         // ResourceLimit
                     12 => 0x0800_0000, // AslrRegionAddress
                     13 => 0x1F00_0000, // AslrRegionSize
                     14 => 0x1000_0000, // StackRegionAddress
                     15 => 0x0010_0000, // StackRegionSize
                     20 => 0,        // UserExceptionContextAddress
                     28 => 0,        // AliasRegionExtraSize
                     _ => 0,
                 };
                 self.write_zr(1, value);
                 self.write_zr(0, RESULT_OK);
                 Ok(())
             }
            0x6F => {
                // GetSystemInfo(out, handle, infoType): value in X1, as above.
                let info_type = self.read_zr(2);
                let value = match info_type {
                    2 => 0x1000_0000, // TotalMemorySize
                    3 => 0,           // UsedMemorySize
                    _ => 0,
                };
                self.write_zr(1, value);
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            _ => Err(Error::Cpu(format!("unimplemented Horizon syscall #{:#x}", imm))),
        }
    }

    fn system(&mut self, insn: u32, next_pc: u32) -> Result<()> {
        let l = (insn >> 21) & 1;
        let op0 = (insn >> 19) & 0b11;
        let op1 = (insn >> 16) & 0b111;
        let crn = (insn >> 12) & 0xF;
        let crm = (insn >> 8) & 0xF;
        let op2 = (insn >> 5) & 0b111;
        let rt = (insn & 0x1F) as u8;

        if (insn >> 16) & 0xFFFF == 0xD503 {
            // HINT (incl. NOP), barriers, MSR immediate etc all no-op here.
            self.pc = next_pc;
            return Ok(());
        }

        if l == 1 {
            // MRS Xt, <sysreg>
            let sysreg = (op0 << 14) | (op1 << 11) | (crn << 7) | (crm << 3) | op2;
            let val = match sysreg {
                // NZCV: 3:3:4:2:0
                0b11_011_0100_0010_000 => self.nzcv as u64,
                // TPIDR_EL0 (3:3:13:0:2) and TPIDRRO_EL0 (3:3:13:0:3):
                // libnx reads the TLS base from one of these, set by the
                // loader. Both are backed by the same value here.
                0b11_011_1101_0000_010 | 0b11_011_1101_0000_011 => self.tpidr,
                // FPCR: 3:0:4:4:0
                0b11_000_0100_0100_000 => 0,
                // DCZID_EL0: 3:3:0:0:7 — report the Cortex-A57 DC ZVA block
                // size (BS=4 → 64 bytes). musl/newlib memset strides the
                // cache-zero loop with `4 << BS`; BS=0 makes it run away.
                0b11_011_0000_0000_111 => 4,
                // CTR_EL0 etc: report 0
                _ => 0,
            };
            self.write_zr(rt, val);
            self.pc = next_pc;
            return Ok(());
        }

        if op0 == 0 {
            // MSR (immediate) or MSR (register) to PSTATE/special
            match (op1, crn, crm, op2) {
                // MSR NZCV, #imm (best-effort; also DAIFSET/CLEAR as no-op)
                (0b010, 0b0100, 0b0010, 0b000) | (0b011, 0b0100, 0b0010, 0b000) => {
                    let imm = (insn >> 8) & 0xF;
                    self.nzcv = imm;
                }
                _ => {
                    // All other MSR immediate forms (DAIF, SPSel, ...) are no-ops.
                }
            }
            self.pc = next_pc;
            return Ok(());
        }

        if l == 0 && op0 == 1 && op1 == 3 && crn == 7 && crm == 4 && op2 == 1 {
            // DC ZVA Xt: zero the 64-byte block at Xt (A57 dczid BS=4).
            // musl/newlib memset uses this to clear aligned blocks.
            let addr = self.read_zr(rt) as u32 & !0x3F;
            for i in 0..0x40u32 {
                self.mem.write_u8(addr.wrapping_add(i), 0)?;
            }
            self.pc = next_pc;
            return Ok(());
        }

        if l == 0 && (op0 == 2 || op0 == 3) {
            // MSR (register): write sysreg from Xt. We observe NZCV and the
            // libnx TLS base.
            let sysreg = (op0 << 14) | (op1 << 11) | (crn << 7) | (crm << 3) | op2;
            match sysreg {
                0b11_011_0100_0010_000 => self.nzcv = self.read_zr(rt) as u32,
                0b11_011_1101_0000_010 | 0b11_011_1101_0000_011 => {
                    self.tpidr = self.read_zr(rt)
                }
                _ => {}
            }
            self.pc = next_pc;
            return Ok(());
        }

        Err(Error::Cpu(format!(
            "unimplemented system instruction 0x{:08x} at {:#x}",
            insn, self.pc
        )))
    }

    // ---------- loads & stores ----------

    /// SIMD (V=1) memory ops: the Q-register (128-bit) subset libnx's
    /// `memset`/`memcpy` uses. Handles unsigned-immediate and unscaled
    /// STR/LDR Q, plus STP/LDP Q (signed-offset / pre-index). Everything else
    /// that sets V=1 is left unimplemented.
    fn try_simd_load_store(&mut self, insn: u32) -> Result<bool> {
        let grp = (insn >> 27) & 0b111;
        // Scalar SIMD LDR/STR (V=1): bits[29:27] = 111, bit26 = 1. The size/opc
        // pairs select the width — opc 00/01 are STR/LDR of B/H/S/D (size
        // 00/01/10/11, byte offset scaled 1/2/4/8), opc 10/11 are STR/LDR Q
        // (128-bit, size must be 00, offset scaled 16). mode=01 is the
        // unsigned-offset form (imm12), mode=00 the unscaled STUR/LDUR (imm9).
        // `ldr b29, [x0, #0x280]` = 0x3D4A001D, `stur q17, [x0, #0x8]` = 0x3C808011.
        if grp == 0b111 {
            let sz = (insn >> 30) & 0b11;
            let opc = (insn >> 22) & 0b11;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rt = (insn & 0x1F) as u8;
            let (bytes, load) = match (sz, opc) {
                (0b00, 0b00) => (1u64, false),
                (0b00, 0b01) => (1, true),
                (0b01, 0b00) => (2, false),
                (0b01, 0b01) => (2, true),
                (0b10, 0b00) => (4, false),
                (0b10, 0b01) => (4, true),
                (0b11, 0b00) => (8, false),
                (0b11, 0b01) => (8, true),
                (0b00, 0b10) => (16, false),
                (0b00, 0b11) => (16, true),
                _ => return Ok(false),
            };
            let mode = (insn >> 24) & 0b11;
            let addr = match mode {
                // Unsigned offset (immediate). imm12 occupies bits[21:10], so
                // bit 21 must NOT be treated as a register-offset flag here —
                // `ldr b29, [x0, #0xc80]` was being misread as a register load
                // using a garbage Rm.
                0b01 => {
                    let scale = if bytes == 16 { 16u64 } else { bytes };
                    let imm = ((insn >> 10) & 0xFFF) as u64;
                    (self.read_x(rn).wrapping_add(imm * scale)) as u32
                }
                // Register offset (mode 0b00, bit 21 set): addr = Xn + (Rm << LSL).
                0b00 if ((insn >> 21) & 1) == 1 => {
                    let rm = ((insn >> 16) & 0x1F) as u8;
                    let s = (insn >> 12) & 1;
                    let shift = if s == 1 { bytes.trailing_zeros() } else { 0 };
                    (self.read_x(rn).wrapping_add(self.read_x(rm).wrapping_shl(shift))) as u32
                }
                // Unscaled offset (STUR/LDUR).
                0b00 => {
                    let imm = sext_u64((insn >> 12) & 0x1FF, 9) as i64;
                    (self.read_x(rn) as i64).wrapping_add(imm) as u32
                }
                _ => return Ok(false),
            };
            if load {
                self.vregs[rt as usize] = match bytes {
                    1 => self.mem.read_u8(addr)? as u128,
                    2 => self.mem.read_u16(addr)? as u128,
                    4 => self.mem.read_u32(addr)? as u128,
                    8 => self.mem.read_u64(addr)? as u128,
                    _ => self.load_q(addr)?,
                };
            } else {
                match bytes {
                    1 => self.mem.write_u8(addr, self.vregs[rt as usize] as u8)?,
                    2 => self.mem.write_u16(addr, self.vregs[rt as usize] as u16)?,
                    4 => self.mem.write_u32(addr, self.vregs[rt as usize] as u32)?,
                    8 => self.mem.write_u64(addr, self.vregs[rt as usize] as u64)?,
                    _ => self.store_q(addr, self.vregs[rt as usize])?,
                }
            }
            return Ok(true);
        }
        // Single structure (LD1/ST1 into one lane): bit31=0, q=bit30,
        // bits[29:24]=001101, p=bit23 (post-index), bit22=1 (load) / 0
        // (store), scale in bits[15:14], selem in bits[13]/[21], lane index in
        // bits[12:10]/[12]/[13] depending on scale.
        if ((insn >> 31) & 1) == 0 && ((insn >> 24) & 0x3F) == 0b001101 {
            let q = (insn >> 30) & 1 == 1;
            let p = (insn >> 23) & 1 == 1;
            let load = (insn >> 22) & 1 == 1;
            let rm = ((insn >> 16) & 0x1F) as u8;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rt = (insn & 0x1F) as u8;
            let scale = (insn >> 14) & 0b11;
            let esize = 1u32 << scale;
            let selem = (((insn >> 21) & 1) << 1 | (insn >> 13) & 1) as u32 + 1;
            let qbit = if q { 1u32 } else { 0 };
            let index = match scale {
                0 => ((insn >> 10) & 0b111) | (qbit << 3),
                1 => ((insn >> 10) & 0b11) | (qbit << 2),
                2 => ((insn >> 12) & 1) | (qbit << 1),
                _ => qbit,
            };
            let mut addr = self.read_x(rn);
            let esize_u = esize as u64;
            for xs in 0..selem {
                let reg = (rt as u32 + xs) % 32;
                let shift = (index as u32) * esize;
                let mask = (1u128 << esize) - 1;
                if load {
                    let val = self.load_by_size(addr as u32, scale, false)?;
                    self.vregs[reg as usize] =
                        (self.vregs[reg as usize] & !(mask << shift)) | ((val as u128) << shift);
                } else {
                    let val = ((self.vregs[reg as usize] >> shift) & mask) as u64;
                    self.store_by_size(addr as u32, scale, val)?;
                }
                addr = addr.wrapping_add(esize_u);
            }
            if p {
                if rm == 31 {
                    self.write_x(rn, addr);
                } else {
                    self.write_x(rn, self.read_x(rn).wrapping_add(self.read_x(rm)));
                }
            }
            return Ok(true);
        }
        // Multiple structures (LD1/LD2/LD3/LD4, ST1/ST2/ST3/ST4): bit31=0,
        // bits[29:24]=001100, bit23=p (post-index), bit22=1 (load) / 0
        // (store), L=bit21, opcode=bits[15:12] selects (rpt, selem), sz in
        // bits[11:10]. Only the `selem==1` forms are contiguous (each
        // register is a plain 64/128-bit chunk); selem>1 interleaves, which
        // isn't needed yet.
        if ((insn >> 31) & 1) == 0
            && ((insn >> 28) & 0b11) == 0b00
            && ((insn >> 24) & 0x0F) == 0b1100
        {
            let q = (insn >> 30) & 1 == 1;
            let load = (insn >> 22) & 1 == 1;
            let rm = ((insn >> 16) & 0x1F) as u8;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rt = (insn & 0x1F) as u8;
            let opcode = (insn >> 12) & 0b1111;
            let (rpt, selem) = match opcode {
                0b0000 => (1, 4),
                0b0010 => (4, 1),
                0b0100 => (1, 3),
                0b0110 => (3, 1),
                0b0111 => (1, 1),
                0b1000 => (1, 2),
                0b1010 => (2, 1),
                _ => return Ok(false),
            };
            if selem != 1 {
                return Ok(false);
            }
            let vec_bytes: u32 = if q { 16 } else { 8 };
            let addr = self.read_x(rn);
            for i in 0..rpt {
                let a = addr.wrapping_add((i as u64) * vec_bytes as u64) as u32;
                if load {
                    self.vregs[(rt + i) as usize % 32] = self.load_q(a)?;
                } else {
                    self.store_q(a, self.vregs[(rt + i) as usize % 32])?;
                }
            }
            if rm != 31 {
                let total = (rpt as u64) * (vec_bytes as u64);
                self.write_x(rn, addr.wrapping_add(total));
            }
            return Ok(true);
        }
        // SIMD&FP register-offset form (V=1): bits[29:27]==111, bits[25:24]==00,
        // bit21==1. Same B/H/S/D/Q size mapping as the immediate forms.
        if grp == 0b111
            && ((insn >> 25) & 1) == 0
            && ((insn >> 24) & 1) == 0
            && ((insn >> 21) & 1) == 1
        {
            let size = (insn >> 30) & 0b11;
            let opc = (insn >> 22) & 0b11;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rt = (insn & 0x1F) as u8;
            let rm = ((insn >> 16) & 0x1F) as u8;
            let opt = ((insn >> 13) & 0b111) as u8;
            let s = (insn >> 12) & 1;
            let is_q = size == 0 && (opc == 0b10 || opc == 0b11);
            let is_b = size == 0 && (opc == 0b00 || opc == 0b01);
            let is_h = size == 1 && (opc == 0b00 || opc == 0b01);
            let is_s = size == 2 && (opc == 0b00 || opc == 0b01);
            let is_d = size == 3 && (opc == 0b00 || opc == 0b01);
            if !is_q && !is_b && !is_h && !is_s && !is_d {
                return Ok(false);
            }
            let off_sz = if is_q { 4 } else { size as u8 };
            let offset = self.offset_from_reg(rm, opt, s, off_sz)?;
            let addr = (self.read_x(rn) as i64).wrapping_add(offset) as u32;
            let elem_bytes: u32 = if is_q {
                16
            } else if is_d {
                8
            } else if is_s {
                4
            } else if is_h {
                2
            } else {
                1
            };
            let load = if is_q { opc == 0b11 } else { opc == 0b01 };
            if load {
                self.vregs[rt as usize] = match elem_bytes {
                    16 => self.load_q(addr)?,
                    8 => self.mem.read_u64(addr)? as u128,
                    4 => self.mem.read_u32(addr)? as u128,
                    2 => self.mem.read_u16(addr)? as u128,
                    _ => self.mem.read_u8(addr)? as u128,
                };
            } else {
                match elem_bytes {
                    16 => self.store_q(addr, self.vregs[rt as usize])?,
                    8 => self.mem.write_u64(addr, self.vregs[rt as usize] as u64)?,
                    4 => self.mem.write_u32(addr, self.vregs[rt as usize] as u32)?,
                    2 => self.mem.write_u16(addr, self.vregs[rt as usize] as u16)?,
                    _ => self.mem.write_u8(addr, self.vregs[rt as usize] as u8)?,
                }
            }
            return Ok(true);
        }
        if grp == 0b111 {
            // Unsigned immediate (mode 01) and unscaled (mode 00) forms for
            // SIMD&FP registers. The 128-bit Q form reuses size=00 with
            // opc=10 (STR) / 11 (LDR); size=00 opc=00/01 is S (32-bit) and
            // size=01 opc=00/01 is D (64-bit).
            let mode = (insn >> 24) & 0b11;
            let opc = (insn >> 22) & 0b11;
            let size = (insn >> 30) & 0b11;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rt = (insn & 0x1F) as u8;
            let is_q = size == 0 && (opc == 0b10 || opc == 0b11);
            let is_b = size == 0 && (opc == 0b00 || opc == 0b01);
            let is_h = size == 1 && (opc == 0b00 || opc == 0b01);
            let is_s = size == 2 && (opc == 0b00 || opc == 0b01);
            let is_d = size == 3 && (opc == 0b00 || opc == 0b01);
            if !is_q && !is_b && !is_h && !is_s && !is_d {
                return Ok(false);
            }
            // B/H/S/D/Q use size 00/01/10/11/00(opc 10/11); the imm is scaled
            // by the element byte count.
            let elem_bytes: u32 = if is_q {
                16
            } else if is_d {
                8
            } else if is_s {
                4
            } else if is_h {
                2
            } else {
                1
            };
            let shift = elem_bytes.trailing_zeros();
            let base = self.read_x(rn);
            let (addr, writeback, wb_val) = if mode == 0b01 {
                // Unsigned immediate, no writeback.
                let imm = (((insn >> 10) & 0xFFF) as u64) << shift;
                (base.wrapping_add(imm) as u32, false, 0)
            } else if mode == 0b00 && ((insn >> 21) & 1) == 0 {
                // Unscaled/pre/post-indexed immediate: imm9 is a byte offset,
                // and bits[11:10] select the addressing mode.
                let imm = sext_u64((insn >> 12) & 0x1FF, 9) as i64;
                let idx = (insn >> 10) & 0b11;
                match idx {
                    0b00 => (base.wrapping_add(imm as u64) as u32, false, 0),
                    0b01 => (base as u32, true, base.wrapping_add(imm as u64)),
                    0b11 => {
                        let addr = base.wrapping_add(imm as u64);
                        (addr as u32, true, addr)
                    }
                    _ => return Ok(false),
                }
            } else {
                return Ok(false);
            };
            let load = if is_q { opc == 0b11 } else { opc == 0b01 };
            if load {
                // Loads zero the destination register above the element.
                self.vregs[rt as usize] = match elem_bytes {
                    16 => self.load_q(addr)?,
                    8 => self.mem.read_u64(addr)? as u128,
                    4 => self.mem.read_u32(addr)? as u128,
                    2 => self.mem.read_u16(addr)? as u128,
                    _ => self.mem.read_u8(addr)? as u128,
                };
            } else {
                match elem_bytes {
                    16 => self.store_q(addr, self.vregs[rt as usize])?,
                    8 => self.mem.write_u64(addr, self.vregs[rt as usize] as u64)?,
                    4 => self.mem.write_u32(addr, self.vregs[rt as usize] as u32)?,
                    2 => self.mem.write_u16(addr, self.vregs[rt as usize] as u16)?,
                    _ => self.mem.write_u8(addr, self.vregs[rt as usize] as u8)?,
                }
            }
            if writeback {
                self.write_x(rn, wb_val);
            }
            return Ok(true);
        }
        if grp == 0b101 && ((insn >> 25) & 1) == 0 {
            // STP/LDP SIMD&FP: size 00/01/10 → S/D/Q pairs, imm scaled by
            // 4<<size (4/8/16 bytes). size=11 is unallocated.
            let size = (insn >> 30) & 0b11;
            if size == 0b11 {
                return Ok(false);
            }
            let bytes: u32 = 4 << size;
            let l = (insn >> 22) & 1;
            let mode = (insn >> 23) & 0b11;
            let imm = sext_u64((insn >> 15) & 0x7F, 7) as i64;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rt = (insn & 0x1F) as u8;
            let rt2 = ((insn >> 10) & 0x1F) as u8;
            let base = self.read_x(rn);
            let scaled = (imm as u64).wrapping_mul(bytes as u64);
            let (addr, writeback, wb) = match mode {
                0b00 => (base.wrapping_add(scaled), false, 0),
                0b01 => (base, true, base.wrapping_add(scaled)),
                0b10 => (base.wrapping_add(scaled), false, 0),
                _ => (base.wrapping_add(scaled), true, base.wrapping_add(scaled)),
            };
            let addr = addr as u32;
            if l == 1 {
                let (v0, v1) = match size {
                    0 => (
                        self.mem.read_u32(addr)? as u128,
                        self.mem.read_u32(addr.wrapping_add(bytes))? as u128,
                    ),
                    1 => (
                        self.mem.read_u64(addr)? as u128,
                        self.mem.read_u64(addr.wrapping_add(bytes))? as u128,
                    ),
                    _ => (
                        self.load_q(addr)?,
                        self.load_q(addr.wrapping_add(bytes))?,
                    ),
                };
                self.vregs[rt as usize] = v0;
                self.vregs[rt2 as usize] = v1;
            } else {
                match size {
                    0 => {
                        self.mem.write_u32(addr, self.vregs[rt as usize] as u32)?;
                        self.mem.write_u32(addr.wrapping_add(bytes), self.vregs[rt2 as usize] as u32)?;
                    }
                    1 => {
                        self.mem.write_u64(addr, self.vregs[rt as usize] as u64)?;
                        self.mem.write_u64(addr.wrapping_add(bytes), self.vregs[rt2 as usize] as u64)?;
                    }
                    _ => {
                        self.store_q(addr, self.vregs[rt as usize])?;
                        self.store_q(addr.wrapping_add(bytes), self.vregs[rt2 as usize])?;
                    }
                }
            }
            if writeback {
                self.write_x(rn, wb);
            }
            return Ok(true);
        }
        Ok(false)
    }

    #[inline]
    fn load_q(&self, addr: u32) -> Result<u128> {
        Ok((self.mem.read_u64(addr)? as u128) | ((self.mem.read_u64(addr.wrapping_add(8))? as u128) << 64))
    }

    #[inline]
    fn store_q(&mut self, addr: u32, v: u128) -> Result<()> {
        self.mem.write_u64(addr, v as u64)?;
        self.mem.write_u64(addr.wrapping_add(8), (v >> 64) as u64)
    }

    #[allow(clippy::too_many_lines)]
    fn try_load_store(&mut self, insn: u32, _next_pc: &mut u32) -> Result<bool> {
        // Exclusive accessors.
        let grp_excl = (insn >> 21) & 0x1FF;
        if (0b001000000..=0b001000011).contains(&grp_excl) || grp_excl == 0b001000100 || grp_excl == 0b001000110 {
            let sz = (insn >> 30) & 0b11;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rt = (insn & 0x1F) as u8;
            let rt2 = ((insn >> 10) & 0x1F) as u8;
            let base = self.read_x(rn);
            match grp_excl {
                0b001000000 => {
                    // STXR Ws, Xt, [Xn]
                    let val = self.read_zr(rt);
                    self.store_by_size(base as u32, sz, val)?;
                    self.write_zr(((insn >> 16) & 0x1F) as u8, 0); // success
                }
                0b001000010 => {
                    // LDXR Xt, [Xn]
                    let val = self.load_by_size(base as u32, sz, false)?;
                    self.write_zr(rt, val);
                }
                0b001000001 => {
                    // STXP: 64-bit pair store
                    let v0 = self.read_zr(rt);
                    let v1 = self.read_zr(rt2);
                    self.mem.write_u64(base as u32, v0)?;
                    self.mem.write_u64(base.wrapping_add(8) as u32, v1)?;
                    self.write_zr(((insn >> 16) & 0x1F) as u8, 0);
                }
                0b001000011 => {
                    // LDXP: 64-bit pair load
                    let v0 = self.mem.read_u64(base as u32)?;
                    let v1 = self.mem.read_u64(base.wrapping_add(8) as u32)?;
                    self.write_zr(rt, v0);
                    self.write_zr(rt2, v1);
                }
                0b001000100 => {
                    // STLR: store-release
                    self.store_by_size(base as u32, sz, self.read_zr(rt))?;
                }
                0b001000110 => {
                    // LDAR: load-acquire
                    let val = self.load_by_size(base as u32, sz, false)?;
                    self.write_zr(rt, val);
                }
                _ => unreachable!(),
            }
            return Ok(true);
        }

        // SIMD (V=1) memory ops: minimal Q-register subset for libnx memset.
        if ((insn >> 26) & 1) == 1 {
            return self.try_simd_load_store(insn);
        }

        // Register-offset form: bit21 == 1 (any size — the previous
        // bits[31:27]==11111 test only matched the 64-bit forms, so 8/16/32-bit
        // register-offset loads/stores fell through as "unimplemented").
        if ((insn >> 27) & 0b111) == 0b111
            && ((insn >> 26) & 1) == 0
            && ((insn >> 24) & 0b11) == 0b00
            && ((insn >> 21) & 1) == 1
        {
            let sz = (insn >> 30) & 0b11;
            let opc = (insn >> 22) & 0b11;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rt = (insn & 0x1F) as u8;
            let rm = ((insn >> 16) & 0x1F) as u8;
            let opt = ((insn >> 13) & 0b111) as u8;
            let s = (insn >> 12) & 1;
            let offset = self.offset_from_reg(rm, opt, s, sz as u8)?;
            let addr = (self.read_x(rn) as i64).wrapping_add(offset) as u32;
            self.ld_st_opc(addr, rt, sz, opc)?;
            return Ok(true);
        }

        // Immediate offset forms: bits[29:27] == 111, V=0
        if ((insn >> 27) & 0b111) == 0b111 && ((insn >> 26) & 1) == 0 {
            let mode = (insn >> 24) & 0b11;
            let sz = (insn >> 30) & 0b11;
            let opc = (insn >> 22) & 0b11;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rt = (insn & 0x1F) as u8;
            if mode == 0b01 {
                // Unsigned offset
                let imm = ((insn >> 10) & 0xFFF) as u64;
                let scale = match sz {
                    0b00 => 1,
                    0b01 => 2,
                    0b10 => 4,
                    _ => 8,
                };
                let addr = self
                    .read_x(rn)
                    .wrapping_add(imm.wrapping_mul(scale))
                    as u32;
                self.ld_st_opc(addr, rt, sz, opc)?;
                return Ok(true);
            }
            if mode == 0b00 && ((insn >> 21) & 1) == 0 {
                // Unscaled / pre / post index
                let idx = (insn >> 10) & 0b11;
                let imm = sext_u64((insn >> 12) & 0x1FF, 9) as i64;
                let base = self.read_x(rn);
                let (addr, writeback) = match idx {
                    0b00 | 0b10 => (base.wrapping_add(imm as u64), false),
                    0b01 => (base, true),                                  // post-index
                    _ => (base.wrapping_add(imm as u64), true),            // pre-index
                };
                self.ld_st_opc(addr as u32, rt, sz, opc)?;
                if writeback {
                    let new_base = if idx == 0b01 {
                        base.wrapping_add(imm as u64)
                    } else {
                        addr
                    };
                    self.write_x(rn, new_base);
                }
                return Ok(true);
            }
        }

        // Paired load/store: bits[29:27] == 101, V=0. The bit25==0 check
        // distinguishes pairs from the SUBS-shifted-register space (which has
        // bits[29:27]=101 too but bit25=1).
        if ((insn >> 27) & 0b111) == 0b101
            && ((insn >> 26) & 1) == 0
            && ((insn >> 25) & 1) == 0
        {
            return self.try_pair(insn);
        }

        Ok(false)
    }

    fn ld_st_opc(&mut self, addr: u32, rt: u8, sz: u32, opc: u32) -> Result<()> {
        // opc selects the access: 00 = STR, 01 = LDR, 10/11 = sign-extending
        // loads (LDRSB/LDRSH/LDRSW). The load bit is NOT opc&1 — treating
        // opc=10 as a store silently corrupted the target (observed as a
        // bogus `ldrsw` index in NX-Shell's tokenizer).
        //
        // size=11 with opc=10/11 is PRFM (prefetch hint) — it must NOT be
        // executed as a sign-extending load, or the load clobbers a register
        // (libtransistor's memcpy starts with `prfm pldl1keep, [x1]`, and
        // treating it as `ldrsw x0, [x1]` made memcpy write to the source
        // magic value as an address, zero-filling the real destination).
        if sz == 0b11 && opc >= 0b10 {
            return Ok(());
        }
        let load = opc != 0b00;
        let sign = (opc >> 1) == 1;
        if load {
            let val = self.load_by_size(addr, sz, sign)?;
            self.write_zr(rt, val);
        } else {
            self.store_by_size(addr, sz, self.read_zr(rt))?;
        }
        Ok(())
    }

    fn load_by_size(&self, addr: u32, sz: u32, sign: bool) -> Result<u64> {
        let raw = match sz {
            0b00 => self.mem.read_u8(addr)? as u64,
            0b01 => self.mem.read_u16(addr)? as u64,
            0b10 => self.mem.read_u32(addr)? as u64,
            _ => self.mem.read_u64(addr)?,
        };
        Ok(if sign {
            let width = match sz {
                0b00 => 8,
                0b01 => 16,
                0b10 => 32,
                _ => 64,
            };
            sext_u64(raw, width)
        } else {
            raw
        })
    }

    fn store_by_size(&mut self, addr: u32, sz: u32, val: u64) -> Result<()> {
        match sz {
            0b00 => self.mem.write_u8(addr, val as u8),
            0b01 => self.mem.write_u16(addr, val as u16),
            0b10 => self.mem.write_u32(addr, val as u32),
            _ => self.mem.write_u64(addr, val),
        }
    }

    fn offset_from_reg(&self, rm: u8, opt: u8, s: u32, sz: u8) -> Result<i64> {
        // Register-offset loads/stores shift Rm by `LSL #scale` where scale is
        // log2(size) (2 for word, 3 for doubleword), NOT the byte count — the
        // byte count over-shifted table indices (e.g. `ldrsw x8,[x9,x8,lsl#2]`
        // read entry 4x too far, loading 0 and jumping into the table itself).
        let shift = if s == 1 { sz as u32 } else { 0 };
        let v = self.read_zr(rm);
        let ext = match opt {
            0b011 => v,                                       // LSL / UXTX
            0b010 => (v as u32) as i64 as u64,                // UXTW
            0b111 => sext_u64(v, 32),                         // SXTW
            0b000 => (v as u8) as u64,                        // UXTB
            0b001 => (v as u16) as u64,                       // UXTH
            0b100 => sext_u64(v, 8),                          // SXTB
            0b101 => sext_u64(v, 16),                         // SXTH
            0b110 => v,                                       // SXTX
            _ => return Err(Error::Cpu(format!("bad register offset option {}", opt))),
        };
        Ok((ext.wrapping_shl(shift)) as i64)
    }

    fn try_pair(&mut self, insn: u32) -> Result<bool> {
        let opc = (insn >> 30) & 0b11;
        let l = (insn >> 22) & 1;
        // opc=01 is the LDP-signed / STGP space; only loads make sense for us.
        if opc == 0b01 && l == 0 {
            return Err(Error::Cpu(format!(
                "unimplemented tagged store-pair at {:#x}",
                self.pc
            )));
        }
        if opc == 0b11 {
            return Err(Error::Cpu(format!(
                "unimplemented pair addressing mode at {:#x}",
                self.pc
            )));
        }
        let sz = if opc == 0b10 { 0b11 } else { 0b10 };
        let scale = if sz == 0b11 { 8u64 } else { 4 };
        let mode = (insn >> 23) & 0b11;
        let imm = sext_u64((insn >> 15) & 0x7F, 7) as i64;
        let rn = ((insn >> 5) & 0x1F) as u8;
        let rt = (insn & 0x1F) as u8;
        let rt2 = ((insn >> 10) & 0x1F) as u8;
        let scaled = (imm as u64).wrapping_mul(scale);

        let base = self.read_x(rn);
        let (addr, writeback, wb_val) = match mode {
            0b00 => (base.wrapping_add(scaled), false, 0),        // signed offset
            0b01 => (base, true, base.wrapping_add(scaled)),      // post-index
            0b10 => (base.wrapping_add(scaled), false, 0),        // offset
            _ => (base.wrapping_add(scaled), true, base.wrapping_add(scaled)), // pre-index
        };
        let addr = addr as u32;

        if l == 1 {
            // LDP: load rt, rt2
            let v0 = if sz == 0b11 {
                self.mem.read_u64(addr)?
            } else {
                let w = self.mem.read_u32(addr)?;
                if opc == 0b01 {
                    sext_u64(w as u64, 32)
                } else {
                    w as u64
                }
            };
            let v1 = if sz == 0b11 {
                self.mem.read_u64(addr.wrapping_add(scale as u32))?
            } else {
                let w = self.mem.read_u32(addr.wrapping_add(scale as u32))?;
                if opc == 0b01 {
                    sext_u64(w as u64, 32)
                } else {
                    w as u64
                }
            };
            self.write_zr(rt, v0);
            self.write_zr(rt2, v1);
        } else {
            // STP: store rt, rt2
            if sz == 0b11 {
                self.mem.write_u64(addr, self.read_zr(rt))?;
                self.mem.write_u64(addr.wrapping_add(8), self.read_zr(rt2))?;
            } else {
                self.mem.write_u32(addr, self.read_zr(rt) as u32)?;
                self.mem.write_u32(addr.wrapping_add(4), self.read_zr(rt2) as u32)?;
            }
        }
        if writeback {
            self.write_x(rn, wb_val);
        }
        Ok(true)
    }

    // ---------- data processing: immediate ----------

    fn try_data_proc_imm(&mut self, insn: u32, _next_pc: &mut u32) -> Result<bool> {
        let grp = (insn >> 24) & 0x1F;
        let sf = (insn >> 31) & 1 == 1;
        match grp {
            0b10000 => {
                // ADR/ADRP (handled earlier, defensive)
                Ok(true)
            }
            0b10001 => {
                // ADD/SUB immediate
                if ((insn >> 23) & 1) == 1 {
                    return Err(Error::Cpu(format!(
                        "unimplemented ADDG/SUBG at {:#x}",
                        self.pc
                    )));
                }
                let op = (insn >> 29) & 0b11;
                let sh = (insn >> 22) & 1;
                let imm12 = ((insn >> 10) & 0xFFF) as u64;
                let rn = ((insn >> 5) & 0x1F) as u8;
                let rd = (insn & 0x1F) as u8;
                let imm = if sh == 1 { imm12 << 12 } else { imm12 };
                // op bit1 selects ADD/SUB, bit0 selects the S (flags) form:
                // ADD=00, ADDS=01, SUB=10, SUBS=11.
                let sub = (op >> 1) == 1;
                let set_flags = (op & 1) == 1;
                self.add_sub(rd, rn, imm, set_flags, sub, sf);
                Ok(true)
            }
            0b10010 => {
                if ((insn >> 23) & 1) == 1 {
                    // MOVN/MOVZ/MOVK
                    let opc = (insn >> 29) & 0b11;
                    let rd = (insn & 0x1F) as u8;
                    let imm16 = ((insn >> 5) & 0xFFFF) as u64;
                    let hw = if sf {
                        (insn >> 21) & 0b11
                    } else {
                        // 32-bit forms encode the shift in bit 21 (bit 22 is
                        // part of the fixed 100101 pattern).
                        (insn >> 21) & 1
                    };
                    let shift = hw * 16;
                    match opc {
                        0b00 => {
                            // MOVN
                            let v = !(imm16 << shift) & Self::mask(sf);
                            self.write_zr(rd, v);
                        }
                        0b10 => {
                            // MOVZ
                            self.write_zr(rd, (imm16 << shift) & Self::mask(sf));
                        }
                        0b11 => {
                            // MOVK
                            let mask = (0xFFFFu64 << shift) & Self::mask(sf);
                            let cur = self.read_zr(rd) & !mask;
                            self.write_zr(rd, cur | ((imm16 << shift) & mask));
                        }
                        _ => {
                            return Err(Error::Cpu(format!(
                                "unimplemented MOV wide opc {} at {:#x}",
                                opc, self.pc
                            )))
                        }
                    }
                    Ok(true)
                } else {
                    // Logical immediate
                    let opc = (insn >> 29) & 0b11;
                    let n = (insn >> 22) & 1;
                    let immr = (insn >> 16) & 0x3F;
                    let imms = (insn >> 10) & 0x3F;
                    let rn = ((insn >> 5) & 0x1F) as u8;
                    let rd = (insn & 0x1F) as u8;
                    let mask = decode_bit_mask(sf, n, immr, imms).ok_or_else(|| {
                        Error::Cpu(format!("unallocated logical immediate at {:#x}", self.pc))
                    })?;
                    let a = self.read_zr(rn) & Self::mask(sf);
                    let r = match opc {
                        0b00 => a & mask,         // AND
                        0b01 => a | mask,         // ORR
                        0b10 => a ^ mask,         // EOR
                        _ => {
                            let r = a & mask;
                            let nbit = (r >> (if sf { 63 } else { 31 })) & 1;
                            let z = (r == 0) as u64;
                            let c = (self.nzcv >> 29) & 1;
                            let v = (self.nzcv >> 28) & 1;
                            self.nzcv =
                                ((nbit as u32) << 31) | ((z as u32) << 30) | (c << 29) | (v << 28);
                            r
                        }
                    };
                    self.write_zr(rd, r);
                    Ok(true)
                }
            }
            0b10011 => {
                if ((insn >> 23) & 1) == 0 {
                    // Bitfield move
                    let opc = (insn >> 29) & 0b11;
                    let rn = ((insn >> 5) & 0x1F) as u8;
                    let rd = (insn & 0x1F) as u8;
                    let (immr, imms) = if sf {
                        if ((insn >> 22) & 1) != 1 {
                            return Err(Error::Cpu(format!(
                                "unallocated bitfield N at {:#x}",
                                self.pc
                            )));
                        }
                        (((insn >> 16) & 0x3F), ((insn >> 10) & 0x3F))
                    } else {
                        if ((insn >> 21) & 1) == 1 || ((insn >> 15) & 1) == 1 {
                            return Err(Error::Cpu(format!(
                                "unallocated 32-bit bitfield at {:#x}",
                                self.pc
                            )));
                        }
                        (((insn >> 16) & 0x1F), ((insn >> 10) & 0x1F))
                    };
                    let val = self.read_zr(rn) & Self::mask(sf);
                    let cur = self.read_zr(rd) & Self::mask(sf);
                    let r = bitfield_apply(opc, val, cur, immr, imms, sf);
                    self.write_zr(rd, r);
                    Ok(true)
                } else {
                    // EXTR
                    let rn = ((insn >> 5) & 0x1F) as u8;
                    let rd = (insn & 0x1F) as u8;
                    let rm = ((insn >> 16) & 0x1F) as u8;
                    let (imm, ok) = if sf {
                        if ((insn >> 22) & 1) != 1 || ((insn >> 21) & 1) == 1 {
                            (0, false)
                        } else {
                            (((insn >> 10) & 0x3F), true)
                        }
                    } else {
                        if ((insn >> 22) & 1) == 1 || ((insn >> 21) & 1) == 1 || ((insn >> 15) & 1) == 1
                        {
                            (0, false)
                        } else {
                            (((insn >> 10) & 0x1F), true)
                        }
                    };
                    if !ok {
                        return Err(Error::Cpu(format!("unallocated EXTR at {:#x}", self.pc)));
                    }
                    let size = if sf { 64 } else { 32 };
                    let a = self.read_zr(rn) & Self::mask(sf);
                    let b = self.read_zr(rm) & Self::mask(sf);
                    let r = if imm == 0 {
                        a
                    } else {
                        ((a >> imm) | (b.wrapping_shl((size as u32).wrapping_sub(imm)))) & Self::mask(sf)
                    };
                    self.write_zr(rd, r);
                    Ok(true)
                }
            }
            _ => Ok(false),
        }
    }

    // ---------- data processing: register ----------

    #[allow(clippy::too_many_lines)]
    fn try_data_proc_reg(&mut self, insn: u32, _next_pc: &mut u32) -> Result<bool> {
        let grp = (insn >> 24) & 0x1F;
        let sf = (insn >> 31) & 1 == 1;
        match grp {
            0b01010 => {
                // Logical shifted register
                let opc = (insn >> 29) & 0b11;
                let st = (insn >> 22) & 0b11;
                let invert = ((insn >> 21) & 1) == 1;
                let rm = ((insn >> 16) & 0x1F) as u8;
                let sa = (insn >> 10) & 0x3F;
                let rn = ((insn >> 5) & 0x1F) as u8;
                let rd = (insn & 0x1F) as u8;
                let a = self.read_zr(rn) & Self::mask(sf);
                let mut b = self.read_zr(rm) & Self::mask(sf);
                if invert {
                    b = !b & Self::mask(sf);
                }
                let b = shift_reg(b, st, sa, sf);
                let r = match opc {
                    0b00 => a & b,
                    0b01 => a | b,
                    0b10 => a ^ b,
                    _ => {
                        let r = a & b;
                        let nbit = (r >> (if sf { 63 } else { 31 })) & 1;
                        let z = (r == 0) as u64;
                        let c = (self.nzcv >> 29) & 1;
                        let v = (self.nzcv >> 28) & 1;
                        self.nzcv = ((nbit as u32) << 31) | ((z as u32) << 30) | (c << 29) | (v << 28);
                        r
                    }
                };
                self.write_zr(rd, r);
                Ok(true)
            }
            0b01011 => {
                // ADD/SUB shifted or extended. op bit1 selects ADD/SUB,
                // bit0 the S (flags) form: ADD=00, ADDS=01, SUB=10, SUBS=11.
                let op = (insn >> 29) & 0b11;
                let rn = ((insn >> 5) & 0x1F) as u8;
                let rd = (insn & 0x1F) as u8;
                let rm = ((insn >> 16) & 0x1F) as u8;
                let sub = (op >> 1) == 1;
                let set_flags = (op & 1) == 1;
                if ((insn >> 21) & 0b111) == 0b001 {
                    // Extended register
                    let option = ((insn >> 13) & 0b111) as u8;
                    let shift = (insn >> 10) & 0b111;
                    let v = extend_reg(self.read_zr(rm), option, sf) & Self::mask(sf);
                    let v = v.wrapping_shl(shift) & Self::mask(sf);
                    self.add_sub(rd, rn, v, set_flags, sub, sf);
                } else {
                    // Shifted register
                    let st = (insn >> 22) & 0b11;
                    let sa = (insn >> 10) & 0x3F;
                    let v = shift_reg(self.read_zr(rm) & Self::mask(sf), st, sa, sf);
                    self.add_sub(rd, rn, v, set_flags, sub, sf);
                }
                Ok(true)
            }
            0b11010 => {
                if ((insn >> 22) & 1) == 1 {
                    if ((insn >> 23) & 1) == 1 {
                        // 2-source or 1-source (bits[28:21]=11010110)
                        let opcode2 = (insn >> 10) & 0x3F;
                        let rn = ((insn >> 5) & 0x1F) as u8;
                        let rd = (insn & 0x1F) as u8;
                        let rm = ((insn >> 16) & 0x1F) as u8;
                        if ((insn >> 29) & 0b11) == 0b00 {
                            // 2-source (bits[30:29]=00)
                            let a = self.read_zr(rn) & Self::mask(sf);
                            let b = self.read_zr(rm) & Self::mask(sf);
                            let r = match opcode2 {
                                0b000010 => {
                                    // UDIV
                                    a.checked_div(b).unwrap_or(0)
                                }
                                0b000011 => {
                                    // SDIV
                                    let (x, y) = (a as i64, b as i64);
                                    (x.checked_div(y).unwrap_or(0)) as u64
                                }
                                0b001000 => shift_var(a, b, 0, sf),
                                0b001001 => shift_var(a, b, 1, sf),
                                0b001010 => shift_var(a, b, 2, sf),
                                0b001011 => {
                                    // RORV
                                    let size = if sf { 64 } else { 32 };
                                    let amt = (b % size) as u32;
                                    if sf {
                                        a.rotate_right(amt)
                                    } else {
                                        (a as u32).rotate_right(amt) as u64
                                    }
                                }
                                _ => {
                                    return Err(Error::Cpu(format!(
                                        "unimplemented 2-source opcode {} at {:#x}",
                                        opcode2, self.pc
                                    )))
                                }
                            };
                            self.write_zr(rd, r);
                        } else if ((insn >> 29) & 0b11) == 0b10 {
                            // 1-source (bits[30:29]=10)
                            let a = self.read_zr(rn) & Self::mask(sf);
                            let size = if sf { 64 } else { 32 };
                            let r = match opcode2 {
                                0b000000 => reverse_bits(a, size),   // RBIT
                                0b000001 => reverse_16_lanes(a, size), // REV16
                                0b000010 => reverse_32_lanes(a, size), // REV32
                                0b000011 => {
                                    // REV64 (64-bit only)
                                    a.swap_bytes()
                                }
                                0b000100 => clz(a, size),
                                0b000101 => cls(a, size),
                                0b000110 => ctz(a, size),
                                _ => {
                                    return Err(Error::Cpu(format!(
                                        "unimplemented 1-source opcode {} at {:#x}",
                                        opcode2, self.pc
                                    )))
                                }
                            };
                            self.write_zr(rd, r & Self::mask(sf));
                        } else {
                            return Err(Error::Cpu(format!(
                                "unimplemented data-processing op at {:#x}",
                                self.pc
                            )));
                        }
                        Ok(true)
                    } else {
                        // CCMP / CCMN
                        let op = (insn >> 30) & 1;
                        let imm_flag = (insn >> 11) & 1;
                        let cond = ((insn >> 12) & 0xF) as u8;
                        let nzcv = insn & 0xF;
                        let rn = ((insn >> 5) & 0x1F) as u8;
                        if !self.condition_holds(cond) {
                            self.nzcv = nzcv << 28;
                        } else {
                            let a = self.read_zr(rn) & Self::mask(sf);
                            // Bit 30: 1 = CCMP (subtract), 0 = CCMN (add). The
                            // 5-bit immediate is unsigned for both forms
                            // (QEMU-verified). The subtract needs the
                            // +carry_in the borrow implies, so carry_in is 1
                            // for CCMP and 0 for CCMN.
                            let b = if imm_flag == 1 {
                                ((insn >> 16) & 0x1F) as u64
                            } else {
                                self.read_zr(((insn >> 16) & 0x1F) as u8)
                            };
                            self.set_nzcv_from_compare(a, b, op == 1, op as u64, sf);
                        }
                        Ok(true)
                    }
                } else {
                    if ((insn >> 23) & 1) == 1 {
                        // CSEL family: csel / csinc / csinv / csneg.
                        // The invert/increment are part of the *else* value,
                        // not applied to the selected value.
                        let else_inv = ((insn >> 30) & 1) == 1;
                        let else_inc = ((insn >> 10) & 1) == 1;
                        let cond = ((insn >> 12) & 0xF) as u8;
                        let rn = ((insn >> 5) & 0x1F) as u8;
                        let rd = (insn & 0x1F) as u8;
                        let rm = ((insn >> 16) & 0x1F) as u8;
                        let a = self.read_zr(rn) & Self::mask(sf);
                        let b = self.read_zr(rm) & Self::mask(sf);
                        let take_a = self.condition_holds(cond);
                        let mut else_val = b;
                        if else_inv {
                            else_val = !else_val;
                        }
                        if else_inc {
                            else_val = else_val.wrapping_add(1);
                        }
                        let r = if take_a { a } else { else_val };
                        self.write_zr(rd, r & Self::mask(sf));
                    } else {
                        // ADC / ADCS / SBC / SBCS
                        let op = (insn >> 29) & 0b11;
                        let rn = ((insn >> 5) & 0x1F) as u8;
                        let rd = (insn & 0x1F) as u8;
                        let rm = ((insn >> 16) & 0x1F) as u8;
                        let carry_in = ((self.nzcv >> 29) & 1) as u64;
                        let a = self.read_zr(rn) & Self::mask(sf);
                        let b = self.read_zr(rm) & Self::mask(sf);
                        let sub = (op & 1) == 1;
                        let set_flags = (op & 2) == 2;
                        let (result, carry, overflow) = if sub {
                            Self::add_carry_overflow(a, !b, carry_in, sf)
                        } else {
                            Self::add_carry_overflow(a, b, carry_in, sf)
                        };
                        if set_flags {
                            self.set_nzcv_from_alu(result, sf, carry, overflow);
                        }
                        self.write_zr(rd, result);
                    }
                    Ok(true)
                }
            }
             0b11011 => {
                // Data processing (3-source) / multiply.
                let rn = ((insn >> 5) & 0x1F) as u8;
                let rd = (insn & 0x1F) as u8;
                let rm = ((insn >> 16) & 0x1F) as u8;
                let ra = ((insn >> 10) & 0x1F) as u8;
                let o0 = ((insn >> 15) & 1) == 1;
                let a = self.read_zr(rn);
                let b = self.read_zr(rm);
                match (insn >> 21) & 0xFF {
                    // MADD / MSUB (bits[28:21] == 11011000), 32- and 64-bit.
                    0b11011000 => {
                        let sf = (insn >> 31) & 1;
                        let mask = Self::mask(sf != 0);
                        let a = a & mask;
                        let b = b & mask;
                        let c = self.read_zr(ra) & mask;
                        let product = a.wrapping_mul(b);
                        let r = if o0 {
                            c.wrapping_sub(product)
                        } else {
                            c.wrapping_add(product)
                        };
                        self.write_zr(rd, r & mask);
                    }
                    // SMADDL / SMSUBL: signed 64x64 multiply-add/subtract.
                    0b11011001 => {
                        let product = ((a as i64 as i128) * (b as i64 as i128)) as u64;
                        let c = self.read_zr(ra);
                        let r = if o0 {
                            c.wrapping_sub(product)
                        } else {
                            c.wrapping_add(product)
                        };
                        self.write_zr(rd, r);
                    }
                    // UMADDL / UMSUBL: unsigned 64x64 multiply-add/subtract.
                    0b11011101 => {
                        let product = (a as u128 * b as u128) as u64;
                        let c = self.read_zr(ra);
                        let r = if o0 {
                            c.wrapping_sub(product)
                        } else {
                            c.wrapping_add(product)
                        };
                        self.write_zr(rd, r);
                    }
                    // SMULH: top 64 bits of the signed 128-bit product.
                    0b11011010 => {
                        let product = ((a as i64 as i128) * (b as i64 as i128)) >> 64;
                        self.write_zr(rd, product as u64);
                    }
                    // UMULH: top 64 bits of the unsigned 128-bit product.
                    0b11011110 => {
                        let product = (a as u128 * b as u128) >> 64;
                        self.write_zr(rd, product as u64);
                    }
                    _ => {
                        return Err(Error::Cpu(format!(
                            "unimplemented multiply-long at {:#x}",
                            self.pc
                        )));
                    }
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}

// ---------------- free-standing helpers ----------------

/// Rounding mode for the float-to-integer conversion instructions.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Rounding {
    /// Round to nearest, ties to even.
    TiesEven,
    /// Round toward +infinity.
    TowardPos,
    /// Round toward -infinity.
    TowardNeg,
    /// Round to nearest, ties away from zero.
    TiesAway,
}

/// Convert a float to a (possibly signed) integer using an explicit rounding
/// mode, then truncate to the destination size. NaN → 0, out-of-range results
/// saturate, matching the default FPCR behavior the emulator assumes.
fn round_to_int(f: f64, r: Rounding, signed: bool) -> u64 {
    if f.is_nan() {
        return 0;
    }
    let rounded = match r {
        Rounding::TiesEven => f.round_ties_even(),
        Rounding::TowardPos => f.ceil(),
        Rounding::TowardNeg => f.floor(),
        Rounding::TiesAway => f.round(),
    };
    let clipped = rounded.clamp(
        i64::MIN as f64,
        if signed { i64::MAX as f64 } else { u64::MAX as f64 },
    );
    if signed {
        (clipped as i64) as u64
    } else {
        clipped.max(0.0) as u64
    }
}

/// Saturating add of two `bits`-wide lanes (`signed` selects SQADD/UQADD).
fn saturating_add(a: u64, b: u64, bits: u32, signed: bool) -> u64 {
    let sum = (a as i128) + (b as i128);
    if signed {
        let (min, max) = (i64::MIN >> (64 - bits), (1i64 << (bits - 1)) - 1);
        sum.clamp(min as i128, max as i128) as u64
    } else {
        let max = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
        (sum as u128).min(max as u128) as u64
    }
}

/// Saturating subtract of two `bits`-wide lanes (`signed` selects SQSUB/UQSUB).
fn saturating_sub(a: u64, b: u64, bits: u32, signed: bool) -> u64 {
    let diff = (a as i128) - (b as i128);
    if signed {
        let (min, max) = (i64::MIN >> (64 - bits), (1i64 << (bits - 1)) - 1);
        diff.clamp(min as i128, max as i128) as u64
    } else {
        let max = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
        diff.clamp(0, max as i128) as u64
    }
}

/// Shift a lane left by the amount in `b`'s low bits; negative shifts shift
/// right (arithmetically for SSHL, logically for USHL).
fn shift_by_reg(a: u64, b: u64, bits: u32, unsigned: bool) -> u64 {
    let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let a = a & mask;
    let shift = (b & mask) as i64;
    if shift >= 0 {
        let sh = shift as u32;
        if sh >= bits { 0 } else { (a << sh) & mask }
    } else {
        let sh = (-shift) as u32;
        if unsigned {
            if sh >= bits { 0 } else { a >> sh }
        } else {
            // Sign-extend the lane before the arithmetic shift.
            let sa = if bits == 64 {
                a as i64
            } else {
                ((a << (64 - bits)) as i64) >> (64 - bits)
            };
            if sh >= bits {
                if sa < 0 { mask } else { 0 }
            } else {
                ((sa >> sh) as u64) & mask
            }
        }
    }
}

/// FP max/min with ARM semantics: if either operand is NaN the NaN operand is
/// returned (Rust's `f64::max` would discard it).
fn fp_max(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        a
    } else if b.is_nan() {
        b
    } else {
        a.max(b)
    }
}

fn fp_min(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        a
    } else if b.is_nan() {
        b
    } else {
        a.min(b)
    }
}

/// FMAXNM/FMINNM: same NaN handling as the plain max/min.
fn fp_maxnum(a: f64, b: f64) -> f64 {
    fp_max(a, b)
}

fn fp_minnum(a: f64, b: f64) -> f64 {
    fp_min(a, b)
}

#[inline]
fn sext_u64<T: Into<u64>>(v: T, bits: u32) -> u64 {
    let v = v.into();
    if bits >= 64 {
        return v;
    }
    let sign = 1u64 << (bits - 1);
    let mask = (1u64 << bits) - 1;
    let v = v & mask;
    if v & sign != 0 {
        v | !mask
    } else {
        v
    }
}

/// Shift `v` left/right logically or arithmetically, or rotate, by `sa`.
fn shift_reg(v: u64, st: u32, sa: u32, sf: bool) -> u64 {
    let size = if sf { 64 } else { 32 };
    let mask = if sf { u64::MAX } else { u32::MAX as u64 };
    let v = v & mask;
    match st {
        0 => {
            // LSL
            if sa >= size {
                0
            } else if sa == 0 {
                v
            } else {
                (v << sa) & mask
            }
        }
        1 => {
            // LSR
            if sa >= size {
                0
            } else if sa == 0 {
                v
            } else {
                v >> sa
            }
        }
        2 => {
            // ASR
            if sa == 0 {
                v
            } else if sa >= size {
                if v & (1 << (size - 1)) != 0 {
                    mask
                } else {
                    0
                }
            } else {
                ((v as i64) >> sa) as u64 & mask
            }
        }
        _ => {
            // ROR
            if sa == 0 {
                v
            } else if sf {
                v.rotate_right(sa % 64)
            } else {
                ((v as u32).rotate_right(sa % 32)) as u64
            }
        }
    }
}

/// Variable shift by register amount (LSLV/LSRV/ASRV).
fn shift_var(v: u64, amt: u64, kind: u32, sf: bool) -> u64 {
    let size = if sf { 64 } else { 32 };
    let amt = (amt % size) as u32;
    shift_reg(v, kind, amt, sf)
}

/// Extend a register value for the ADD/SUB extended-register form.
fn extend_reg(v: u64, option: u8, sf: bool) -> u64 {
    match option {
        0b000 => v as u8 as u64,        // UXTB
        0b001 => v as u16 as u64,       // UXTH
        0b010 => v as u32 as u64,       // UXTW
        0b011 => v,                     // UXTX / LSL
        0b100 => sext_u64(v, 8),        // SXTB
        0b101 => sext_u64(v, 16),       // SXTH
        0b110 => sext_u64(v, 32),       // SXTW
        0b111 => v,                     // SXTX
        _ => v,
    }
    .min(if sf { u64::MAX } else { u32::MAX as u64 })
}

/// Decode the rotated-element bitmask of the logical-immediate encoding.
/// Decode a logical-immediate (AND/ORR/EOR/ANDS) bitmask, per ARM ARM
/// `DecodeBitMasks`. Matches QEMU `logic_imm_decode_wmask`: the element size
/// is derived from `N:NOT(imms)` and bits of `imms` above the element size
/// are ignored (e.g. `mov w20, #0x80808080`).
/// Expand a MOVI/MVNI 8-bit immediate per ARM `AdvSIMDExpandImm` (mirrors
/// QEMU's `asimd_imm_const`). Returns the 64-bit lane value; the caller
/// replicates it over the 128-bit register for Q=1.
fn simd_imm_const(imm: u32, cmode: u32, op: u32) -> u64 {
    let mut imm = imm;
    match cmode {
        0 | 1 => {}
        2 | 3 => imm <<= 8,
        4 | 5 => imm <<= 16,
        6 | 7 => imm <<= 24,
        8 | 9 => imm |= imm << 16,
        10 | 11 => imm = (imm << 8) | (imm << 24),
        12 => imm = (imm << 8) | 0xff,
        13 => imm = (imm << 16) | 0xffff,
        14 => {
            if op == 1 {
                // Byte-mask form: imm's set bits select 0xff bytes.
                let mut imm64 = 0u64;
                for n in 0..8 {
                    if imm & (1 << n) != 0 {
                        imm64 |= 0xffu64 << (n * 8);
                    }
                }
                return imm64;
            }
            imm |= (imm << 8) | (imm << 16) | (imm << 24);
        }
        15 => {
            if op == 1 {
                // 64-bit float immediate (valid for AArch64).
                let mut imm64 = ((imm & 0x3f) as u64) << 48;
                if imm & 0x80 != 0 {
                    imm64 |= 0x8000_0000_0000_0000;
                }
                if imm & 0x40 != 0 {
                    imm64 |= 0x3fc0_0000_0000_0000;
                } else {
                    imm64 |= 0x4000_0000_0000_0000;
                }
                return imm64;
            }
            imm = ((imm & 0x80) << 24)
                | ((imm & 0x3f) << 19)
                | if imm & 0x40 != 0 { 0x1f << 25 } else { 1 << 30 };
        }
        _ => {}
    }
    if op != 0 {
        imm = !imm;
    }
    (imm as u64) | ((imm as u64) << 32)
}

pub(crate) fn decode_bit_mask(sf: bool, n: u32, immr: u32, imms: u32) -> Option<u64> {
    if !sf && n != 0 {
        return None;
    }
    let combined = ((n & 1) << 6) | ((!imms) & 0x3F);
    if combined == 0 {
        return None;
    }
    let len = 32 - combined.leading_zeros() - 1;
    let e = 1u64 << len;
    let levels = e - 1;
    let s = imms as u64 & levels;
    let r = immr as u64 & levels;
    if s == levels {
        return None;
    }
    let mut welem = (1u64 << (s + 1)) - 1;
    if r != 0 {
        welem = (welem >> r) | (welem << (e - r));
        if e < 64 {
            welem &= (1u64 << e) - 1;
        }
    }
    let datasize = if sf { 64 } else { 32 };
    let mut wmask = 0u64;
    let mut shift = 0u32;
    while shift < datasize {
        wmask |= welem.wrapping_shl(shift);
        shift += e as u32;
    }
    Some(wmask)
}

/// SBFM / BFM / UBFM semantics.
fn bitfield_apply(opc: u32, val: u64, cur: u64, immr: u32, imms: u32, sf: bool) -> u64 {
    let datasize = if sf { 64 } else { 32 };
    let lsb = immr as u64;
    let msb = imms as u64;

    match opc {
        // UBFM
        0b10 => {
            if msb >= lsb {
                let width = (msb - lsb + 1) as u32;
                (val >> lsb) & mask_of_width(width, sf)
            } else {
                // UBFIZ: field at the bottom, shifted up
                let shift = datasize - lsb;
                ((val & mask_of_width((msb + 1) as u32, sf)).wrapping_shl(shift as u32))
                    & mask_of_width(64, sf)
            }
        }
        // SBFM
        0b00 => {
            if msb >= lsb {
                let width = (msb - lsb + 1) as u32;
                sext_u64(val >> lsb, width)
            } else {
                let shift = datasize - lsb;
                let field = val & mask_of_width((msb + 1) as u32, sf);
                let shifted = field.wrapping_shl(shift as u32);
                // sign extend from bit (msb) after the shift
                let sign_bit = msb as u32;
                if shifted & (1u64 << sign_bit) != 0 {
                    shifted | !mask_of_width((shift + msb + 1) as u32, sf)
                } else {
                    shifted & mask_of_width((shift + msb + 1) as u32, sf)
                }
            }
        }
        // BFM — merges Rn into the ORIGINAL Rd (BFI / BFXIL). The old decoder
        // used `cur = val` (Rn) and never read the destination register, so
        // `bfi` zeroed the bits it was meant to preserve. libtransistor's
        // squashfs `swab_super` relies on this.
        0b01 => {
            if msb >= lsb {
                let width = (msb - lsb + 1) as u32;
                let field = (val >> lsb) & mask_of_width(width, sf);
                (cur & !mask_of_width(width, sf)) | field
            } else {
                let field = val & mask_of_width((msb + 1) as u32, sf);
                let shift = (datasize - lsb) as u32;
                let m = mask_of_width((msb + 1) as u32, sf).wrapping_shl(shift);
                (cur & !m) | (field << shift)
            }
        }
        _ => 0,
    }
}

fn mask_of_width(width: u32, _sf: bool) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

fn reverse_bits(v: u64, size: u32) -> u64 {
    let r = v.reverse_bits();
    if size == 64 {
        r
    } else {
        (r >> 32) as u32 as u64
    }
}

fn reverse_16_lanes(v: u64, size: u32) -> u64 {
    let mut out = 0u64;
    let lanes = size / 16;
    for i in 0..lanes {
        let lane = ((v >> (i * 16)) & 0xFFFF) as u16;
        out |= ((lane.swap_bytes() as u64) & 0xFFFF) << (i * 16);
    }
    out
}

fn reverse_32_lanes(v: u64, size: u32) -> u64 {
    let mut out = 0u64;
    let lanes = size / 32;
    for i in 0..lanes {
        let lane = ((v >> (i * 32)) & 0xFFFF_FFFF) as u32;
        out |= ((lane.swap_bytes() as u64) & 0xFFFF_FFFF) << (i * 32);
    }
    out
}

fn clz(v: u64, size: u32) -> u64 {
    let v = if size == 32 { (v as u32) as u64 } else { v };
    (if size == 64 {
        v.leading_zeros()
    } else {
        (v as u32).leading_zeros()
    }) as u64
}

fn cls(v: u64, size: u32) -> u64 {
    if size == 32 {
        let v = v as i32;
        if v == 0 {
            return 31;
        }
        if v < 0 {
            (!v).leading_zeros() as u64
        } else {
            v.leading_zeros() as u64
        }
    } else {
        let v = v as i64;
        if v == 0 {
            return 63;
        }
        if v < 0 {
            (!v).leading_zeros() as u64
        } else {
            v.leading_zeros() as u64
        }
    }
}

fn ctz(v: u64, size: u32) -> u64 {
    let v = if size == 32 { (v as u32) as u64 } else { v };
    (if size == 64 {
        v.trailing_zeros()
    } else {
        (v as u32).trailing_zeros()
    }) as u64
}
