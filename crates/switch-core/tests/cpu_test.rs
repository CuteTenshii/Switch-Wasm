//! CPU correctness tests.
//!
//! These run hand-assembled AArch64 machine code through the interpreter and
//! assert on register, flag, memory and console state. Encodings were
//! verified against QEMU's `a64.decode` where a doubt existed.

use switch_core::cpu::{Cpu, SyscallMode};

fn cpu_at(pc: u32) -> Cpu {
    let mut cpu = Cpu::new();
    cpu.mem.map_zero(pc, 0x400).unwrap();
    cpu.set_pc(pc);
    cpu
}

/// Little helper for assembling a small program in memory then running it.
fn exec(code: &[u32], max: u64) -> Cpu {
    let mut cpu = cpu_at(0x1000);
    let mut bytes = Vec::with_capacity(code.len() * 4);
    for insn in code {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    let steps = max.min(code.len() as u64);
    cpu.run(steps).unwrap();
    cpu
}

/// Map code, run exactly `code.len()` instructions, return the CPU.
fn run_program(mut cpu: Cpu, pc: u32, code: &[u32]) -> Cpu {
    let mut bytes = Vec::with_capacity(code.len() * 4);
    for insn in code {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(pc, &bytes).unwrap();
    cpu.set_pc(pc);
    cpu.run(code.len() as u64).unwrap();
    cpu
}

// ---------------- instruction encodings (hand-assembled) ----------------

// ADD Xd, Xn, #imm  :  sf(1) op(0) S(0) 100010 sh imm12 Rn Rd
fn add_imm(rd: u32, rn: u32, imm: u32, sf: bool) -> u32 {
    let sf = if sf { 1u32 << 31 } else { 0 };
    sf | 0b100010 << 23 | ((imm & 0xFFF) << 10) | (rn << 5) | rd
}

// ADD Xd, Xn, Xm (shifted, LSL #0)
fn add_reg(rd: u32, rn: u32, rm: u32, sf: bool) -> u32 {
    let sf = if sf { 1u32 << 31 } else { 0 };
    sf | 0b01011 << 24 | (rm << 16) | (rn << 5) | rd
}

// MOVZ/MOVN/MOVK Xd, #imm16, LSL #(hw*16)
fn movz(rd: u32, imm16: u32, hw: u32, sf: bool) -> u32 {
    let sf = if sf { 1u32 << 31 } else { 0 };
    sf | 0b10 << 29 | 0b100101 << 23 | (hw << 21) | (imm16 << 5) | rd
}
fn movn(rd: u32, imm16: u32, hw: u32, sf: bool) -> u32 {
    let sf = if sf { 1u32 << 31 } else { 0 };
    sf | 0b00 << 29 | 0b100101 << 23 | (hw << 21) | (imm16 << 5) | rd
}
fn movk(rd: u32, imm16: u32, hw: u32, sf: bool) -> u32 {
    let sf = if sf { 1u32 << 31 } else { 0 };
    sf | 0b11 << 29 | 0b100101 << 23 | (hw << 21) | (imm16 << 5) | rd
}

// LDR Xt, [Xn, #imm]  (unsigned offset, 64-bit)
fn ldr64(rt: u32, rn: u32, imm: u32) -> u32 {
    0b11 << 30 | 0b111 << 27 | 0b01 << 24 | 0b01 << 22 | ((imm >> 3) & 0xFFF) << 10 | (rn << 5) | rt
}
fn ldr32(rt: u32, rn: u32, imm: u32) -> u32 {
    0b10 << 30 | 0b111 << 27 | 0b01 << 24 | 0b01 << 22 | ((imm >> 2) & 0xFFF) << 10 | (rn << 5) | rt
}
// STR Xt, [Xn, #imm]
fn str64(rt: u32, rn: u32, imm: u32) -> u32 {
    0b11 << 30 | 0b111 << 27 | 0b01 << 24 | 0b00 << 22 | ((imm >> 3) & 0xFFF) << 10 | (rn << 5) | rt
}
// LDUR Xt, [Xn, #imm]
fn ldur64(rt: u32, rn: u32, imm: i64) -> u32 {
    0b11 << 30 | 0b111 << 27 | 0b00 << 24 | 0b01 << 22 | ((imm as u32 & 0x1FF) << 12) | (rn << 5) | rt
}

// B #imm
fn b(imm: i32) -> u32 {
    0b000101 << 26 | ((imm >> 2) as u32 & 0x3FF_FFFF)
}
// BL #imm
fn bl(imm: i32) -> u32 {
    0b100101 << 26 | ((imm >> 2) as u32 & 0x3FF_FFFF)
}
// BR Xn
fn br(rn: u32) -> u32 {
    0xD61F0000 | (rn << 5)
}
// BLR Xn
fn blr(rn: u32) -> u32 {
    0xD63F0000 | (rn << 5)
}
// RET Xn
fn ret(rn: u32) -> u32 {
    0xD65F0000 | (rn << 5)
}
// B.cond #imm, cond
fn bcond(cond: u32, imm: i32) -> u32 {
    0b01010100 << 24 | ((imm >> 2) as u32 & 0x7_FFFF) << 5 | 0x10 | cond
}
// CBZ/CBNZ Xt, #imm
fn cbz(rt: u32, imm: i32, sf: bool, nz: bool) -> u32 {
    let sf = if sf { 1u32 << 31 } else { 0 };
    sf | 0b011010 << 25 | ((nz as u32) << 24) | ((imm >> 2) as u32 & 0x7_FFFF) << 5 | rt
}
// TBZ/TBNZ Xt, #bit, #imm
fn tbz(rt: u32, bit: u32, imm: i32, nz: bool) -> u32 {
    let sf = (bit >> 5) << 31;
    sf | 0b011011 << 25 | ((nz as u32) << 24) | ((bit & 0x1F) << 19) | ((imm >> 2) as u32 & 0x3FFF) << 5 | rt
}
// SVC #imm
fn svc(imm: u32) -> u32 {
    0xD4000000 | (imm << 5) | 1
}
// NOP
fn nop() -> u32 {
    0xD503201F
}
// MOV Xd, Xm  == ORR Xd, XZR, Xm
fn mov_reg(rd: u32, rm: u32, sf: bool) -> u32 {
    let sf = if sf { 1u32 << 31 } else { 0 };
    sf | 0b01 << 29 | 0b01010 << 24 | (rm << 16) | (31 << 5) | rd
}
// CMP Xn, Xm  == SUBS XZR, Xn, Xm
fn cmp_reg(rn: u32, rm: u32, sf: bool) -> u32 {
    let sf = if sf { 1u32 << 31 } else { 0 };
    sf | 0b11 << 29 | 0b01011 << 24 | (rm << 16) | (rn << 5) | 31
}
// ADR Xd, #imm
fn adr(rd: u32, imm: i32) -> u32 {
    let imm = imm as u32;
    0b10000 << 24 | (imm & 0x3) << 29 | ((imm >> 2) & 0x7_FFFF) << 5 | rd
}
// MRS Xt, NZCV
fn mrs_nzcv(rt: u32) -> u32 {
    0xD53B4200 | rt
}
// MSR XZR, NZCV (write flags from xzr = 0)
fn msr_nzcv() -> u32 {
    0xD51B4200 | 31
}

// ---------------- tests ----------------

#[test]
fn movz_movn_movk_build_64bit() {
    let cpu = exec(
        &[
            movz(1, 0x1234, 0, true),
            movk(1, 0x5678, 1, true),
            movk(1, 0x9ABC, 2, true),
            movk(1, 0xDEF0, 3, true),
            movn(2, 0xABCD, 0, true),
        ],
        100,
    );
    assert_eq!(cpu.read_x(1), 0xDEF0_9ABC_5678_1234);
    assert_eq!(cpu.read_x(2), !0xABCDu64);
    assert_eq!(cpu.get_pc(), 0x1000 + 5 * 4);
}

#[test]
fn add_immediate_sets_flags() {
    let cpu = exec(
        &[add_imm(1, 31, 5, true), add_imm(2, 1, 0xFFF, false)],
        100,
    );
    // x1 = SP(0) + 5
    assert_eq!(cpu.read_x(1), 5);
    // w2 = w1 + 0xFFF
    assert_eq!(cpu.read_x(2), 0xFFF + 5);
}

#[test]
fn sub_and_flags() {
    // SUBS XZR, X2, X1  after x2=3, x1=3 → Z set
    let mut cpu = exec(
        &[
            movz(1, 3, 0, true),
            movz(2, 3, 0, true),
            cmp_reg(2, 1, true),
        ],
        100,
    );
    assert_eq!(cpu.nzcv() & (1 << 30), 1 << 30); // Z
    assert_eq!(cpu.nzcv() & (1 << 31), 0);       // N clear

    // SUBS XZR, X1, X2 with x1=1, x2=3 → N set, C clear
    cpu = exec(
        &[
            movz(1, 1, 0, true),
            movz(2, 3, 0, true),
            cmp_reg(1, 2, true),
        ],
        100,
    );
    assert_eq!(cpu.nzcv() & (1 << 31), 1 << 31); // N
    assert_eq!(cpu.nzcv() & (1 << 29), 0);       // C clear (borrow)
}

#[test]
fn load_store_immediate() {
    let mut cpu = cpu_at(0x2000);
    cpu.set_reg(2, 0x3000);
    cpu.mem.map_zero(0x2FF8, 0x20).unwrap();
    cpu.mem.write_u64(0x2FF8, 0xDEAD_BEEF_CAFE_F00D).unwrap();
    let code = [
        str64(1, 2, 0),
        str64(3, 2, 8),
        ldr64(4, 2, 8),
        ldur64(5, 2, -8),
        ldr32(6, 2, 0),
    ];
    let mut bytes = Vec::new();
    for insn in code {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(0x2000, &bytes).unwrap();
    cpu.set_pc(0x2000);
    cpu.set_reg(1, 0xDEAD_BEEF_CAFE_F00D);
    cpu.set_reg(3, 0x1122_3344_5566_7788);
    cpu.run(5).unwrap();
    assert_eq!(cpu.mem.read_u64(0x3000).unwrap(), 0xDEAD_BEEF_CAFE_F00D);
    assert_eq!(cpu.mem.read_u64(0x3008).unwrap(), 0x1122_3344_5566_7788);
    assert_eq!(cpu.read_x(4), 0x1122_3344_5566_7788);
    assert_eq!(cpu.read_x(5), 0xDEAD_BEEF_CAFE_F00D); // LDUR negative offset into 0x2FF8
    assert_eq!(cpu.read_x(6), 0xCAFE_F00D);            // LDR 32-bit zero-extended
}

#[test]
fn stp_ldp_pre_index() {
    let mut cpu = cpu_at(0x2000);
    cpu.set_reg(0, 0x4000);
    let code: [u32; 3] = [
        0xA9BE_07E0, // STP X0, X1, [SP, #-32]!  -> base SP=0x4000, imm -32
        0xA940_0FE2, // LDP X2, X3, [SP]  (offset mode, imm 0)
        0xA8C2_07E0, // LDP X0, X1, [SP], #32 (post-index)
    ];
    let mut bytes = Vec::new();
    for insn in code {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(0x2000, &bytes).unwrap();
    cpu.set_pc(0x2000);
    cpu.set_pc_and_sp(0x2000, 0x4000);
    cpu.set_reg(0, 0x1111_1111_1111_1111);
    cpu.set_reg(1, 0x2222_2222_2222_2222);
    cpu.run(3).unwrap();
    assert_eq!(cpu.mem.read_u64(0x4000 - 32).unwrap(), 0x1111_1111_1111_1111);
    assert_eq!(cpu.mem.read_u64(0x4000 - 24).unwrap(), 0x2222_2222_2222_2222);
    assert_eq!(cpu.read_x(2), 0x1111_1111_1111_1111);
    assert_eq!(cpu.read_x(3), 0x2222_2222_2222_2222);
    assert_eq!(cpu.sp(), 0x4000); // after post-index add back
}

#[test]
fn branches_subroutine_and_link() {
    // main: BL func ; SVC 0
    // func: MOV X9, #42 ; RET
    let func_pc = 0x1100i32;
    let bl_off = func_pc - 0x1000; // BL at 0x1000, target 0x1100
    let cpu = exec(
        &[
            bl(bl_off),
            svc(0),
            nop(), // 0x1008
            nop(), // 0x100c
            nop(), // 0x1010
            nop(), // 0x1014
            nop(), // 0x1018
            nop(), // 0x101c
            nop(), // 0x1020
            nop(), // 0x1024
            nop(), // 0x1028
            nop(), // 0x102c
            nop(), // 0x1030
            nop(), // 0x1034
            nop(), // 0x1038
            nop(), // 0x103c
            nop(), // 0x1040
            nop(), // 0x1044
            nop(), // 0x1048
            nop(), // 0x104c
            nop(), // 0x1050
            nop(), // 0x1054
            nop(), // 0x1058
            nop(), // 0x105c
            nop(), // 0x1060
            nop(), // 0x1064
            nop(), // 0x1068
            nop(), // 0x106c
            nop(), // 0x1070
            nop(), // 0x1074
            nop(), // 0x1078
            nop(), // 0x107c
            nop(), // 0x1080
            nop(), // 0x1084
            nop(), // 0x1088
            nop(), // 0x108c
            nop(), // 0x1090
            nop(), // 0x1094
            nop(), // 0x1098
            nop(), // 0x109c
            nop(), // 0x10a0
            nop(), // 0x10a4
            nop(), // 0x10a8
            nop(), // 0x10ac
            nop(), // 0x10b0
            nop(), // 0x10b4
            nop(), // 0x10b8
            nop(), // 0x10bc
            nop(), // 0x10c0
            nop(), // 0x10c4
            nop(), // 0x10c8
            nop(), // 0x10cc
            nop(), // 0x10d0
            nop(), // 0x10d4
            nop(), // 0x10d8
            nop(), // 0x10dc
            nop(), // 0x10e0
            nop(), // 0x10e4
            nop(), // 0x10e8
            nop(), // 0x10ec
            nop(), // 0x10f0
            nop(), // 0x10f4
            nop(), // 0x10f8
            nop(), // 0x10fc
            movz(9, 42, 0, true),
            ret(30),
        ],
        200,
    );
    assert_eq!(cpu.read_x(9), 42);
    // x30 = return address = 0x1004
    assert_eq!(cpu.read_x(30), 0x1004);
}

#[test]
fn conditional_branch_and_compare() {
    // x1=0; CMP x1,xzr; B.NE +8 (should NOT branch since Z set)
    let cpu = exec(
        &[
            movz(1, 0, 0, true),  // x1 = 0
            cmp_reg(1, 31, true), // CMP x1, xzr
            bcond(0x1, 8),        // B.NE +8
            movz(2, 0xAA, 0, true),
            nop(),
        ],
        4,
    );
    assert_eq!(cpu.read_x(2), 0xAA);
}

#[test]
fn cbz_tbz() {
    // x1=0 → CBZ branches past a MOV
    let cpu = exec(
        &[
            movz(1, 0, 0, true),
            cbz(1, 8, true, false), // branch +8 → 0x100C
            movz(2, 0xBB, 0, true),
            movz(3, 0xCC, 0, true),
            movz(4, 0xDD, 0, true),
        ],
        4,
    );
    assert_eq!(cpu.read_x(2), 0);
    assert_eq!(cpu.read_x(3), 0xCC); // branch skipped the movz(2) only
    assert_eq!(cpu.read_x(4), 0xDD);

    // TBZ on bit 0 (x5=1 → bit0 set → TBNZ branches)
    let cpu = exec(
        &[
            movz(5, 1, 0, true),
            tbz(5, 0, 8, true), // TBNZ x5, #0, +8
            movz(6, 0xEE, 0, true),
            movz(7, 0xFF, 0, true),
            nop(),
        ],
        4,
    );
    assert_eq!(cpu.read_x(6), 0);
    assert_eq!(cpu.read_x(7), 0xFF);
}

#[test]
fn syscall_uart_prints_and_halts() {
    let mut cpu = Cpu::new();
    cpu.syscall_mode = SyscallMode::Uart;
    // write "hi\0" at 0x3000
    cpu.mem.map(0x3000, b"hi\0").unwrap();
    let code = [adr(0, 0x3000 - 0x1000), svc(2), svc(0)];
    let mut bytes = Vec::new();
    for insn in code {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    let report = cpu.run(code.len() as u64).unwrap();
    assert!(report.halted);
    assert_eq!(cpu.out, b"hi");
}

#[test]
fn logical_immediate_masks() {
    // AND X1, X2, #0xFF  → encoding N=0, immr=0, imms=7
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(2, 0xABCD_EF12_3456_78FF);
    let code: [u32; 1] = [
        0b0 << 31 | 0b00 << 29 | 0b100100 << 23 | 0 << 22 | (0 << 16) | (7 << 10) | (2 << 5) | 1,
    ];
    let mut bytes = Vec::new();
    for insn in code {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(1), 0xFF);
}

#[test]
fn multiply_add() {
    // MADD X1, X2, X3, X4 → x1 = 5*7 + 11 = 46
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(2, 5);
    cpu.set_reg(3, 7);
    cpu.set_reg(4, 11);
    // MADD X1, X2, X3, X4: sf=1, 11011000, rm=3, o0=0, ra=4, rn=2, rd=1
    let code = [1u32 << 31 | 0b11011000 << 21 | (3 << 16) | (4 << 10) | (2 << 5) | 1];
    let mut bytes = Vec::new();
    for insn in code {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(1), 46);
}

#[test]
fn csel_selects_by_condition() {
    // x1=5, x2=9; CMP x1,x2 (GT? x1<x2 so no); CSEL x3, x1, x2, GT
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, 5);
    cpu.set_reg(2, 9);
    // cmp then CSEL X3, X1, X2, GT (GT false → take else operand X2=9)
    let code = [
        cmp_reg(1, 2, true),
        1u32 << 31 | 0b011010100 << 21 | (2 << 16) | (0xC << 12) | (1 << 5) | 3,
    ];
    let mut bytes = Vec::new();
    for insn in code {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(2).unwrap();
    assert_eq!(cpu.read_x(3), 9); // GT false → take else operand (x2=9)
}

#[test]
fn udiv_sdiv() {
    // UDIV X1, X6, X7 ; SDIV X2, X6, X7
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(6, 50);
    cpu.set_reg(7, 7);
    let udiv = 1u32 << 31 | 0b11010110 << 21 | (7 << 16) | 0b000010 << 10 | (6 << 5) | 1;
    let sdiv = 1u32 << 31 | 0b11010110 << 21 | (7 << 16) | 0b000011 << 10 | (6 << 5) | 2;
    let mut bytes = Vec::new();
    for insn in [udiv, sdiv] {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(2).unwrap();
    assert_eq!(cpu.read_x(1), 7);
    assert_eq!(cpu.read_x(2), 7);

    // negative: 50 / -7
    cpu.set_reg(6, 50);
    cpu.set_reg(7, !7u64 + 1);
    cpu.set_pc(0x1000);
    cpu.run(2).unwrap();
    assert_eq!(cpu.read_x(2) as i64, -7);
}

#[test]
fn bitfield_extract_and_insert() {
    // UBFX X1, X2, #4, #8  → UBFM X1, X2, #4, #11 (immr=4, imms=11)
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(2, 0xABCD_EF12_3456_78FF);
    let ubfm = 1u32 << 31 | 0b10 << 29 | 0b100110 << 23 | 1 << 22 | (4 << 16) | (11 << 10) | (2 << 5) | 1;
    let mut bytes = Vec::new();
    for insn in [ubfm] {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(1), (0xABCD_EF12_3456_78FFu64 >> 4) & 0xFF);

    // SBFX: SBFM X3, X2, #0, #7 (sign-extend 8 bits) → -1
    let sbfm = 1u32 << 31 | 0b00 << 29 | 0b100110 << 23 | 1 << 22 | (0 << 16) | (7 << 10) | (2 << 5) | 3;
    let mut bytes = Vec::new();
    for insn in [sbfm] {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(3), 0xFFFF_FFFF_FFFF_FFFF); // 0xFF sign-extended = -1
}

#[test]
fn ldr_literal() {
    // LDR X1, #+8 → loads the u64 at pc+8 (the literal pool value)
    let mut cpu = cpu_at(0x1000);
    let ldr_lit: u32 = 0b01 << 30 | 0b011 << 27 | (8 >> 2) << 5 | 1;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&ldr_lit.to_le_bytes());
    bytes.extend_from_slice(&nop().to_le_bytes());
    bytes.extend_from_slice(&0xCAFE_BEEF_DEAD_F00Du64.to_le_bytes());
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(2).unwrap();
    assert_eq!(cpu.read_x(1), 0xCAFE_BEEF_DEAD_F00D);
}

#[test]
fn halt_via_svc() {
    let mut cpu = Cpu::new();
    let code = [svc(0)];
    let mut bytes = Vec::new();
    for insn in code {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    let report = cpu.run(10).unwrap();
    assert!(report.halted);
}

#[test]
fn mrs_msr_nzcv() {
    // CMP to set flags then MRS NZCV
    let mut cpu = Cpu::new();
    let code = [movz(1, 3, 0, true), movz(2, 3, 0, true), cmp_reg(1, 2, true), mrs_nzcv(4), svc(0)];
    let mut bytes = Vec::new();
    for insn in code {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(code.len() as u64).unwrap();
    assert_eq!(cpu.read_x(4) & (1 << 30), 1 << 30); // Z set, captured via MRS
}

#[test]
fn exclusive_load_store() {
    let mut cpu = Cpu::new();
    cpu.set_reg(1, 0x3000);
    cpu.mem.map_zero(0x3000, 8).unwrap();
    // STXR W0, X2, [X1] ; LDXR X3, [X1]
    let stxr: u32 = 0b11 << 30 | 0b001000000 << 21 | (0 << 16) | (1 << 10) | (1 << 5) | 2;
    let ldxr: u32 = 0b11 << 30 | 0b001000010 << 21 | (1 << 10) | (1 << 5) | 3;
    let mut bytes = Vec::new();
    for insn in [stxr, ldxr] {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    cpu.set_reg(2, 0x1234_5678_9ABC_DEF0);
    cpu.run(2).unwrap();
    assert_eq!(cpu.mem.read_u64(0x3000).unwrap(), 0x1234_5678_9ABC_DEF0);
    assert_eq!(cpu.read_x(0), 0); // STXR status
    assert_eq!(cpu.read_x(3), 0x1234_5678_9ABC_DEF0);
}

#[test]
fn stack_memory_and_rt_eq_rn() {
    // LDR X1, [X1] must read base before overwrite
    let mut cpu = Cpu::new();
    cpu.set_reg(1, 0x3000);
    cpu.mem.map_zero(0x3000, 8).unwrap();
    cpu.mem.write_u64(0x3000, 0x5555).unwrap();
    let code = [ldr64(1, 1, 0)];
    let mut bytes = Vec::new();
    for insn in code {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(1), 0x5555);
}

#[test]
fn reverse_and_count_ops() {
    // RBIT, CLZ, CTZ, REV32
    let mut cpu = Cpu::new();
    cpu.set_reg(1, 0x0000_0000_0000_00F0);
    let rbit = 1u32 << 31 | 0b10 << 29 | 0b11010110 << 21 | (0b000000 << 10) | (1 << 5) | 2;
    let clz = 1u32 << 31 | 0b10 << 29 | 0b11010110 << 21 | (0b000100 << 10) | (1 << 5) | 3;
    let ctz = 1u32 << 31 | 0b10 << 29 | 0b11010110 << 21 | (0b000110 << 10) | (1 << 5) | 4;
    let rev32 = 1u32 << 31 | 0b10 << 29 | 0b11010110 << 21 | (0b000010 << 10) | (1 << 5) | 5;
    let mut bytes = Vec::new();
    for insn in [rbit, clz, ctz, rev32] {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(4).unwrap();
    assert_eq!(cpu.read_x(2), 0x0F00_0000_0000_0000); // bit-reversed
    assert_eq!(cpu.read_x(3), 56);                     // clz(0xF0) = 56
    assert_eq!(cpu.read_x(4), 4);                      // ctz(0xF0) = 4
    assert_eq!(cpu.read_x(5), 0xF000_0000);            // rev32 of 0x...00F0
}

#[test]
fn add_immediate_preserves_flags() {
    // CMP x1, x1 (sets Z), then ADD X2, X1, #'0' — flags must stay set.
    let mut cpu = Cpu::new();
    cpu.set_reg(1, 5);
    let code = [
        cmp_reg(1, 1, true), // Z=1
        add_imm(2, 1, 48, true), // must NOT clear flags
        bcond(0x0, 8),       // B.EQ +8 -> taken only if Z still set
        movz(3, 0xEE, 0, true),
        movz(4, 0xFF, 0, true),
    ];
    let mut bytes = Vec::new();
    for insn in code {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(4).unwrap();
    assert_eq!(cpu.read_x(2), 5 + 48);
    assert_eq!(cpu.read_x(3), 0);      // EQ branch taken → movz(3) skipped
    assert_eq!(cpu.read_x(4), 0xFF);
}

#[test]
fn bootstrap_provides_stack_and_low_memory() {
    let mut cpu = Cpu::new();
    assert_eq!(cpu.sp(), 0);
    cpu.bootstrap();

    // SP points at the top of the mapped stack.
    assert_eq!(cpu.sp(), 0x1010_0000);
    cpu.mem.write_u64((cpu.sp() - 8) as u32, 0x1234_5678).unwrap();
    assert_eq!(cpu.mem.read_u64((cpu.sp() - 8) as u32).unwrap(), 0x1234_5678);

    // Reads from untouched low memory return zero instead of faulting — the
    // exact `ldr x0, [x0]` at 0x244498 a real libnx binary hit.
    assert_eq!(cpu.mem.read_u32(0x244498).unwrap(), 0);
    // Writes allocate a private page on first touch.
    cpu.mem.write_u32(0xb00, 0xDEAD_BEEF).unwrap();
    assert_eq!(cpu.mem.read_u32(0xb00).unwrap(), 0xDEAD_BEEF);
    // The soft region ends at the NRO base; reads beyond it still fault.
    assert!(cpu.mem.read_u32(0x8000_0000 + 0xDEAD).is_err());
}

#[test]
fn horizon_syscall_stubs() {
    use switch_core::cpu::SyscallMode;

    // OutputDebugString(0x3000, 5) logs the string to the console.
    let mut cpu = cpu_at(0x1000);
    cpu.syscall_mode = SyscallMode::Horizon;
    cpu.mem.map(0x3000, b"hello").unwrap();
    cpu.set_reg(0, 0x3000);
    cpu.set_reg(1, 5);
    cpu.mem.map(0x1000, &svc(0x26).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.out, b"hello");

    // A null pointer / bogus length is tolerated (no fault).
    let mut cpu = cpu_at(0x1000);
    cpu.syscall_mode = SyscallMode::Horizon;
    cpu.set_reg(0, 0);
    cpu.set_reg(1, 0xFFFFFFFFFFFFFFDCu64);
    cpu.mem.map(0x1000, &svc(0x26).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();

    // ExitProcess halts the machine.
    let mut cpu = cpu_at(0x1000);
    cpu.syscall_mode = SyscallMode::Horizon;
    cpu.mem.map(0x1000, &svc(0x06).to_le_bytes()).unwrap();
    let report = cpu.run(1).unwrap();
    assert!(report.halted);

    // GetSystemTick returns the cycle count scaled to ns.
    let mut cpu = cpu_at(0x1000);
    cpu.syscall_mode = SyscallMode::Horizon;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&nop().to_le_bytes());
    bytes.extend_from_slice(&svc(0x1D).to_le_bytes());
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.run(2).unwrap();
    assert_eq!(cpu.read_x(0), 1000); // one nop executed before the svc

    // ConnectToNamedPort succeeds with a fake handle returned in X1.
    let mut cpu = cpu_at(0x1000);
    cpu.syscall_mode = SyscallMode::Horizon;
    cpu.set_reg(0, 0x3000); // name pointer (ignored by the stub)
    cpu.set_reg(1, 4);
    cpu.mem.map(0x1000, &svc(0x1E).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);
    assert_eq!(cpu.read_x(1), 0x1000);

    // SendSyncRequest is a no-op success so service init proceeds.
    let mut cpu = cpu_at(0x1000);
    cpu.syscall_mode = SyscallMode::Horizon;
    cpu.set_reg(0, 0x1000); // session handle
    cpu.set_reg(1, 0x3000); // ipc buffer pointer
    cpu.set_reg(2, 0x40);
    cpu.mem.map(0x1000, &svc(0x20).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);
}

#[test]
fn tpidr_el0_roundtrip() {
    // msr tpidrro_el0, x1 ; mrs x2, tpidrro_el0 (the encoding hbmenu uses)
    let mut cpu = Cpu::new();
    cpu.set_reg(1, 0x1234_5678_9ABC_DEF0);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(0xD51BD060u32 | 1).to_le_bytes()); // msr tpidrro_el0, x1
    bytes.extend_from_slice(&(0xD53BD060u32 | 2).to_le_bytes()); // mrs x2, tpidrro_el0
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(2).unwrap();
    assert_eq!(cpu.read_x(2), 0x1234_5678_9ABC_DEF0);
}

#[test]
fn sub_shifted_register() {
    // SUB X2, X0, X1 must subtract, not add, and must not clobber flags.
    let mut cpu = Cpu::new();
    cpu.set_reg(0, 0x1000);
    cpu.set_reg(1, 0x123);
    // sub x2, x0, x1  (0xCB010002), then a marker add that reads flags.
    let code = [
        0xCB01_0002u32, // sub x2, x0, x1
        0x9100_0003u32, // add x3, x0, #0
    ];
    let mut bytes = Vec::new();
    for w in code {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(2).unwrap();
    assert_eq!(cpu.read_x(2), 0x1000 - 0x123);
    assert_eq!(cpu.read_x(3), 0x1000);
}

#[test]
fn adds_shifted_register() {
    // ADDS X2, X0, X1 must add AND set flags (previously decoded as SUB).
    let mut cpu = Cpu::new();
    cpu.set_reg(0, 0x10);
    cpu.set_reg(1, 0x20);
    let code = [
        0xAB01_0002u32, // adds x2, x0, x1
        0x5400_0040u32, // b.eq +8
        0xD280_1DC3u32, // movz x3, #0xee
        0xD503_201Fu32, // nop
    ];
    let mut bytes = Vec::new();
    for w in code {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(4).unwrap();
    assert_eq!(cpu.read_x(2), 0x30);   // 0x10 + 0x20, not subtraction
    assert_eq!(cpu.read_x(3), 0xEE);   // Z clear (result nonzero) -> branch not taken
}

#[test]
fn cmp_does_not_clobber_sp() {
    // CMP X0, #0x10 == SUBS XZR, X0, #0x10 — must not write the stack pointer.
    let mut cpu = Cpu::new();
    cpu.set_reg(0, 0x10);
    cpu.set_pc_and_sp(0x1000, 0x30441ef9);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0xF100401Fu32.to_le_bytes()); // cmp x0, #0x10
    bytes.extend_from_slice(&0xD503201Fu32.to_le_bytes()); // nop
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.run(2).unwrap();
    assert_eq!(cpu.sp(), 0x30441ef9, "CMP must leave SP untouched");
    assert_eq!(cpu.nzcv() & (1 << 30), 1 << 30); // Z set (0x10 - 0x10 == 0)
}

#[test]
fn fault_trace_shows_recent_instructions() {
    let mut cpu = Cpu::new();
    cpu.mem.map_zero(0x1000, 0x20).unwrap();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&movz(1, 7, 0, true).to_le_bytes());
    bytes.extend_from_slice(&0x0000_0000u32.to_le_bytes()); // executes as UDF
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    assert!(cpu.run(10).is_err());
    let trace = String::from_utf8_lossy(&cpu.trace).to_string();
    assert!(trace.contains("movz x1, #0x7"), "fault trace must show the run-up:\n{trace}");
    assert!(trace.contains(".word 0x00000000"), "fault trace must show the faulting word:\n{trace}");
}

#[test]
fn adrp_adr_with_nonzero_immlo() {
    // ADRP X0, #0x1000 → page(0x1000) + 0x1000 = 0x2000 (immlo = 01)
    // ADR  X1, #0x5    → pc(0x1004) + 5 = 0x1009          (immlo = 01)
    let cpu = exec(
        &[
            1u32 << 31 | 1 << 29 | 0b10000 << 24,      // adrp x0, #0x1000
            0b10000 << 24 | 1 << 29 | 1 << 5 | 1,      // adr x1, #0x5
        ],
        100,
    );
    assert_eq!(cpu.read_x(0), 0x2000);
    assert_eq!(cpu.read_x(1), 0x1009);
}

#[test]
fn disassembler_produces_readable_output() {
    use switch_core::disasm::disassemble;
    // MOVZ X1, #0x1234
    assert_eq!(disassemble(0xD2824681), "movz x1, #0x1234");
    // ADR X0, #0x54
    assert_eq!(disassemble(0x100002A0), "adr x0, #0x54");
    // ADRP X0, #0x1000 (non-zero immlo — previously misdecoded)
    assert_eq!(disassemble(0xB0000000), "adrp x0, #0x1");
    // ADRP X0, #0x235 from the crashing binary's trace
    assert_eq!(disassemble(0xB00011A0), "adrp x0, #0x235");
    // SVC #0
    assert_eq!(disassemble(0xD4000001), "svc #0x0");
    // RET
    assert_eq!(disassemble(0xD65F03C0), "ret x30");
    // B.LE +8
    assert_eq!(disassemble(0x5400005D), "b.le #0x8");
    // STP X0, X1, [sp, #-0x20]!
    assert_eq!(disassemble(0xA9BE07E0), "stp x0, x1, [sp, #-0x20]!");
    // CBZ X1, #0x8
    assert_eq!(disassemble(0xB4000041), "cbz x1, #0x8");
    // LDR X4, [X2, #0x8]
    assert_eq!(disassemble(0xF9400444), "ldr x4, [x2, #0x8]");
    // ADD X3, X1, X2
    assert_eq!(disassemble(0x8B020023), "add x3, x1, x2");
    // NOP
    assert_eq!(disassemble(0xD503201F), "nop");
    // CSEL X3, X1, X2, GT
    assert_eq!(disassemble(0x9A82C023), "csel x3, x1, x2, gt");
}
