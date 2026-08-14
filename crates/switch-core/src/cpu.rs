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

/// Where SVC traps are routed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SyscallMode {
    /// `SVC #0` halts the machine; anything else faults.
    #[default]
    None,
    /// Demo ABI: `SVC #0` halts, `SVC #1` writes the byte in X0 to the
    /// console, `SVC #2` writes the NUL-terminated string at X0.
    Uart,
    /// Real libnx syscall numbers, best-effort stubs so homebrew built for
    /// the Switch can boot single-threaded: console logging, sleeps, handles
    /// and process/timing info are faked, and unsupported calls fault.
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
/// `STACK_TOP` (below the NRO load base and clear of the framebuffer).
pub const STACK_SIZE: u64 = 0x0010_0000;
pub const STACK_TOP: u64 = 0x1010_0000;

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
        // the thread-local-storage base. Point it at a writable low-memory
        // region as a fallback; the crt0 may override it via MSR, which the
        // interpreter honors.
        self.tpidr = 0x2000_0000;
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
    #[inline]
    fn add_carry_overflow(a: u64, b: u64, carry_in: u64, sf: bool) -> (u64, u32, u32) {
        let size = if sf { 64 } else { 32 };
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
                        // RET
                        self.pc = self.read_zr(rn) as u32;
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

    fn syscall(&mut self, imm: u16) -> Result<()> {
        match self.syscall_mode {
            SyscallMode::None => {
                if imm == 0 {
                    self.halted = true;
                    Ok(())
                } else {
                    Err(Error::Cpu(format!("unimplemented syscall #{}", imm)))
                }
            }
            SyscallMode::Uart => match imm {
                0 => {
                    self.halted = true;
                    Ok(())
                }
                1 => {
                    self.out.push((self.read_zr(0) & 0xFF) as u8);
                    Ok(())
                }
                2 => {
                    let mut addr = self.read_zr(0) as u32;
                    loop {
                        let c = self.mem.read_u8(addr)?;
                        if c == 0 {
                            break;
                        }
                        self.out.push(c);
                        addr = addr.wrapping_add(1);
                    }
                    Ok(())
                }
                _ => Err(Error::Cpu(format!("unknown UART syscall #{}", imm))),
            },
            SyscallMode::Horizon => self.horizon_syscall(imm),
        }
    }

    /// Permissive stubs for the Horizon syscall numbers libnx homebrew hits
    /// during startup and normal single-threaded operation. There are no real
    /// services or threads, so service/IPC calls return success with a fake
    /// handle and waits complete immediately; this lets the app's `main()`
    /// run as far as it can before it needs real hardware. Results follow the
    /// real ABI: X0 carries the Result (or value), success is 0, errors have
    /// bit 31 set, and out-handles come back in X1.
    fn horizon_syscall(&mut self, imm: u16) -> Result<()> {
        const RESULT_OK: u64 = 0;
        // Non-zero handle handed out by handle-returning syscalls (libnx
        // stores X1 into the caller's output pointer).
        const FAKE_HANDLE: u64 = 0x1000;
        match imm {
            0x00 => {
                // SetHeapSize: report a heap at a soft-mapped address.
                self.write_zr(0, RESULT_OK);
                self.write_zr(1, 0x2000_0000);
                Ok(())
            }
            0x03 | 0x04 | 0x12 | 0x13 => {
                // MapMemory / UnmapMemory / MapSharedMemory / UnmapSharedMemory
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x06 | 0x09 => {
                // ExitProcess / ExitThread
                self.halted = true;
                Ok(())
            }
            0x07 => {
                // CreateThread: hand out a fake handle; StartThread is a no-op
                // so the main thread keeps running and waits "complete".
                self.write_zr(0, RESULT_OK);
                self.write_zr(1, FAKE_HANDLE);
                Ok(())
            }
            0x08 | 0x0A | 0x0B | 0x0C | 0x0D | 0x0E => {
                // StartThread / SleepThread / get-set thread priority / core mask
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x0F => {
                // GetCurrentProcessorNumber
                self.write_zr(0, 0);
                Ok(())
            }
            0x10 | 0x11 | 0x16 => {
                // SignalEvent / ClearEvent / ResetSignal
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x14 => {
                // CreateTransferMemory
                self.write_zr(0, RESULT_OK);
                self.write_zr(1, FAKE_HANDLE);
                Ok(())
            }
            0x15 => {
                // CloseHandle
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x17 | 0x18 => {
                // WaitSynchronization / CancelSynchronization: waits complete
                // immediately.
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x19 | 0x1A | 0x1B | 0x1C => {
                // ArbitrateLock/Unlock, Wait/SignalProcessWideKey
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x1D => {
                // GetSystemTick (ns scale, arbitrary)
                self.write_zr(0, self.cycles * 1000);
                Ok(())
            }
            0x1E => {
                // ConnectToNamedPort: return a fake handle so sm/service init
                // proceeds instead of aborting.
                self.write_zr(0, RESULT_OK);
                self.write_zr(1, FAKE_HANDLE);
                Ok(())
            }
            0x1F | 0x20 | 0x21 | 0x22 => {
                // SendSyncRequest[Light|WithUserBuffer] / async variant:
                // pretend the request succeeded. The IPC buffer is left as-is,
                // so whatever handles libnx parses out of it are fake too.
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x23 => {
                // GetProcessId
                self.write_zr(0, 1);
                Ok(())
            }
            0x24 => {
                // GetThreadId
                self.write_zr(0, 1);
                Ok(())
            }
            0x25 => {
                // Break: fatal debugger trap — surface and stop.
                self.out.extend_from_slice(b"[svcBreak]\n");
                self.halted = true;
                Ok(())
            }
            0x26 => {
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
            0x28 | 0x29 => {
                // GetInfo / GetSystemInfo — report 0.
                self.write_zr(0, 0);
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
                // CTR_EL0, DCZID_EL0 etc: report 0
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

        // Register-offset form: bit21 == 1
        if ((insn >> 27) & 0x1F) == 0b11111
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
        let load = (opc & 1) == 1;
        let sign = (opc >> 1) == 1;
        let val = self.load_by_size(addr, sz, sign)?;
        if load {
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
        let scale = match sz {
            0 => 1,
            1 => 2,
            2 => 4,
            _ => 8,
        };
        let shift = if s == 1 { scale } else { 0 };
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
                        (insn >> 22) & 1
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
                    let r = bitfield_apply(opc, val, immr, imms, sf);
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
                            let (b, carry_in) = if imm_flag == 1 {
                                let imm = (insn >> 16) & 0x1F;
                                let v = if op == 1 {
                                    sext_u64(imm as u64, 5)
                                } else {
                                    imm as u64
                                };
                                (v, 0u64)
                            } else {
                                (self.read_zr(((insn >> 16) & 0x1F) as u8), 0u64)
                            };
                            self.set_nzcv_from_compare(a, b, op == 0, carry_in, sf);
                        }
                        Ok(true)
                    }
                } else {
                    if ((insn >> 23) & 1) == 1 {
                        // CSEL family
                        let else_inv = ((insn >> 30) & 1) == 1;
                        let else_inc = ((insn >> 10) & 1) == 1;
                        let cond = ((insn >> 12) & 0xF) as u8;
                        let rn = ((insn >> 5) & 0x1F) as u8;
                        let rd = (insn & 0x1F) as u8;
                        let rm = ((insn >> 16) & 0x1F) as u8;
                        let a = self.read_zr(rn) & Self::mask(sf);
                        let b = self.read_zr(rm) & Self::mask(sf);
                        let take_a = self.condition_holds(cond);
                        let mut r = if take_a { a } else { b };
                        if else_inc {
                            r = r.wrapping_add(1);
                        }
                        if else_inv {
                            r = !r;
                        }
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
                // MADD / MSUB (bits[28:21] == 11011000)
                if ((insn >> 21) & 0xFF) == 0b11011000 {
                    let o0 = ((insn >> 15) & 1) == 1;
                    let rn = ((insn >> 5) & 0x1F) as u8;
                    let rd = (insn & 0x1F) as u8;
                    let rm = ((insn >> 16) & 0x1F) as u8;
                    let ra = ((insn >> 10) & 0x1F) as u8;
                    let a = self.read_zr(rn) & Self::mask(sf);
                    let b = self.read_zr(rm) & Self::mask(sf);
                    let c = self.read_zr(ra) & Self::mask(sf);
                    let product = a.wrapping_mul(b);
                    let r = if o0 {
                        c.wrapping_sub(product)
                    } else {
                        c.wrapping_add(product)
                    };
                    self.write_zr(rd, r & Self::mask(sf));
                } else {
                    return Err(Error::Cpu(format!(
                        "unimplemented multiply-long at {:#x}",
                        self.pc
                    )));
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}

// ---------------- free-standing helpers ----------------

#[inline]
fn sext_u64<T: Into<u64>>(v: T, bits: u32) -> u64 {
    let v = v.into();
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
pub(crate) fn decode_bit_mask(sf: bool, n: u32, immr: u32, imms: u32) -> Option<u64> {
    if !sf && n != 0 {
        return None;
    }
    let not_imms = (!imms) & 0x3F;
    let combined = ((n & 1) << 6) | not_imms;
    let len = (0..=6).rev().find(|&i| combined & (1 << i) != 0)?;
    let levels = (1u64 << len) - 1;
    if !sf && (imms as u64 & !levels) != 0 {
        return None;
    }
    let s = imms as u64 & levels;
    let r = immr as u64 & levels;
    let esize = 1u64 << len;
    let welem = if s == esize - 1 { u64::MAX } else { (1u64 << (s + 1)) - 1 };
    let wmask_elem = rotate_right(welem, r as u32, esize as u32);
    let datasize = if sf { 64 } else { 32 };
    let mut wmask = 0u64;
    let mut shift = 0u32;
    while shift < datasize {
        wmask |= wmask_elem.wrapping_shl(shift);
        shift += esize as u32;
    }
    Some(wmask)
}

fn rotate_right(v: u64, r: u32, esize: u32) -> u64 {
    if esize == 64 {
        return v.rotate_right(r % 64);
    }
    let m = (1u64 << esize) - 1;
    let v = v & m;
    let r = r % esize;
    if r == 0 {
        v
    } else {
        ((v >> r) | (v << (esize - r))) & m
    }
}

/// SBFM / BFM / UBFM semantics.
fn bitfield_apply(opc: u32, val: u64, immr: u32, imms: u32, sf: bool) -> u64 {
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
        // BFM
        0b01 => {
            let cur = val;
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
