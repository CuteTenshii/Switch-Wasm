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
    cpu.mem.map(0x1000, &svc(0x27).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.out, b"hello");

    // A null pointer / bogus length is tolerated (no fault).
    let mut cpu = cpu_at(0x1000);
    cpu.syscall_mode = SyscallMode::Horizon;
    cpu.set_reg(0, 0);
    cpu.set_reg(1, 0xFFFFFFFFFFFFFFDCu64);
    cpu.mem.map(0x1000, &svc(0x27).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();

    // ExitProcess halts the machine.
    let mut cpu = cpu_at(0x1000);
    cpu.syscall_mode = SyscallMode::Horizon;
    cpu.mem.map(0x1000, &svc(0x07).to_le_bytes()).unwrap();
    let report = cpu.run(1).unwrap();
    assert!(report.halted);

    // GetSystemTick returns the cycle count scaled to ns.
    let mut cpu = cpu_at(0x1000);
    cpu.syscall_mode = SyscallMode::Horizon;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&nop().to_le_bytes());
    bytes.extend_from_slice(&svc(0x1E).to_le_bytes());
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.run(2).unwrap();
    assert_eq!(cpu.read_x(0), 1000); // one nop executed before the svc

    // ConnectToNamedPort succeeds with a fake handle returned in X1.
    let mut cpu = cpu_at(0x1000);
    cpu.syscall_mode = SyscallMode::Horizon;
    cpu.set_reg(0, 0x3000); // name pointer (ignored by the stub)
    cpu.set_reg(1, 4);
    cpu.mem.map(0x1000, &svc(0x1F).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);
    assert_eq!(cpu.read_x(1), 0x1000);

    // SendSyncRequest is a no-op success so service init proceeds.
    let mut cpu = cpu_at(0x1000);
    cpu.syscall_mode = SyscallMode::Horizon;
    cpu.set_reg(0, 0x1000); // session handle
    cpu.set_reg(1, 0x3000); // ipc buffer pointer
    cpu.set_reg(2, 0x40);
    cpu.mem.map(0x1000, &svc(0x21).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);
}

#[test]
fn horizon_query_memory_and_get_info() {
    use switch_core::cpu::SyscallMode;

    // QueryMemory writes a MemoryInfo struct to the out pointer and returns
    // the page info in X1.
    let mut cpu = cpu_at(0x1000);
    cpu.syscall_mode = SyscallMode::Horizon;
    cpu.set_reg(0, 0x3000); // MemoryInfo out
    cpu.set_reg(1, 0x4000); // PageInfo out
    cpu.set_reg(2, 0x0800_1000); // queried address
    cpu.mem.map(0x1000, &svc(0x06).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);
    assert_eq!(cpu.read_x(1), 0x1000);
    assert_eq!(cpu.mem.read_u64(0x3000).unwrap(), 0x0800_1000);
    assert_eq!(cpu.mem.read_u64(0x3008).unwrap(), 0x8000_0000);

    // GetInfo returns the requested value in X1 (the libnx wrapper stores it
    // to the out pointer). InfoType 4 = TotalMemorySize.
    let mut cpu = cpu_at(0x1000);
    cpu.syscall_mode = SyscallMode::Horizon;
    cpu.set_reg(1, 4); // infoType
    cpu.set_reg(2, 0xffff_8001); // CUR_PROCESS_HANDLE
    cpu.mem.map(0x1000, &svc(0x29).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);
    assert_eq!(cpu.read_x(1), 0x1E00_0000);
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

#[test]
fn movk32_shift_is_bit21() {
    // movz w1, #0x4653 ; movk w1, #0x4f43, lsl #16 → 0x4f434653.
    // The 32-bit hw/shift bit is bit 21 (not bit 22) — a regression from the
    // hbmenu MOD0-magic check that miscomputed w1 as 0x4f43.
    let cpu = exec(
        &[movz(1, 0x4653, 0, false), movk(1, 0x4f43, 1, false)],
        2,
    );
    assert_eq!(cpu.read_x(1), 0x4f43_4653);
}

#[test]
fn add_carry_overflow_masks_to_operand_size() {
    // CMP W0, #4 with W0 = 0 must set C=0 (borrow), so B.CC is taken.
    // A 64-bit `!rhs` leaking into the 32-bit carry computation previously set
    // C=1 and mis-branched in hbmenu's applet init.
    let cpu = exec(
        &[
            movz(0, 0, 0, false),
            // cmp w0, #4  (SUBS WZR, W0, #4)
            0x7100_101F,
            bcond(0x3, 8), // B.CC +8 (taken iff C=0)
            movz(5, 0xAA, 0, false),
            movz(6, 0xBB, 0, false),
            nop(),
            nop(),
        ],
        6,
    );
    assert_eq!(cpu.read_x(5), 0); // B.CC taken → movz(5) skipped
    assert_eq!(cpu.read_x(6), 0xBB);
}

#[test]
fn register_offset_load_32bit() {
    // ldr w2, [x19, x0] — the 0xb8606a62 form hbmenu hit; previously only the
    // 64-bit register-offset encodings (bits[31:27]==11111) were decoded.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(19, 0x2000);
    cpu.set_reg(0, 0x30);
    cpu.mem.write_u32(0x2030, 0xDEAD_BEEF).unwrap();
    let code = [0xB860_6A62u32, nop()];
    let mut bytes = Vec::new();
    for insn in code {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(2), 0xDEAD_BEEF);
}

#[test]
fn simd_dup_umov_and_q_store() {
    // dup v0.16b, w1 ; mov x2, v0.d[0] ; str q0, [x3]
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, 0x5A);
    cpu.set_reg(3, 0x3000);
    cpu.mem.map_zero(0x3000, 0x40).unwrap();
    let code = [
        0x4E01_0C20u32, // dup v0.16b, w1
        0x4E08_3C02u32, // mov x2, v0.d[0]  (umov, rd=2)
        0x3D80_0060u32, // str q0, [x3]
        nop(),
    ];
    let mut bytes = Vec::new();
    for insn in code {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.run(3).unwrap();
    assert_eq!(cpu.read_x(2), 0x5A5A_5A5A_5A5A_5A5A);
    assert_eq!(cpu.mem.read_u64(0x3000).unwrap(), 0x5A5A_5A5A_5A5A_5A5A);
    assert_eq!(cpu.mem.read_u64(0x3008).unwrap(), 0x5A5A_5A5A_5A5A_5A5A);
}

// ---------------- Advanced SIMD (three-same / logical / permute) ----------------

// dup <Vd>.16B, <Wn>
fn dup16(rd: u32, rn: u32) -> u32 {
    0x4E01_0C00u32 | (rd & 0x1F) | ((rn & 0x1F) << 5)
}
// mov <Xd>, <Vn>.D[0]  (umov)
fn umov_d0(rd: u32, rn: u32) -> u32 {
    0x4E08_3C00u32 | rd | (rn << 5)
}
// SUB <Vd>.4S, <Vn>.4S, <Vm>.4S
fn sub4s(rd: u32, rn: u32, rm: u32) -> u32 {
    (1u32 << 30) | (1u32 << 29) | (0b1110 << 24) | (0b10 << 22) | (1 << 21)
        | (rm << 16) | (0b10000 << 11) | (1 << 10) | (rn << 5) | rd
}
// CMEQ <Vd>.16B, <Vn>.16B, <Vm>.16B
fn cmeq16(rd: u32, rn: u32, rm: u32) -> u32 {
    (1u32 << 30) | (1u32 << 29) | (0b1110 << 24) | (1 << 21) | (rm << 16)
        | (0b10001 << 11) | (1 << 10) | (rn << 5) | rd
}
// UHADD <Vd>.16B, <Vn>.16B, <Vm>.16B
fn uhadd16(rd: u32, rn: u32, rm: u32) -> u32 {
    (1u32 << 30) | (1u32 << 29) | (0b1110 << 24) | (rm << 16) | (1 << 10) | (rn << 5) | rd
}
// ADDP <Vd>.16B, <Vn>.16B, <Vm>.16B
fn addp16(rd: u32, rn: u32, rm: u32) -> u32 {
    (1u32 << 30) | (0b1110 << 24) | (1 << 21) | (rm << 16) | (0b10111 << 11) | (1 << 10) | (rn << 5) | rd
}
// ZIP1 <Vd>.16B, <Vn>.16B, <Vm>.16B
fn zip1_16(rd: u32, rn: u32, rm: u32) -> u32 {
    (1u32 << 30) | (0b1110 << 24) | (rm << 16) | (0b001110 << 10) | (rn << 5) | rd
}

#[test]
fn simd_three_same_add_sub_compare() {
    // dup v0.16b, w1 (0x3d) ; dup v1.16b, w2 (0x3d) ; sub v2.4s, v0.4s, v1.4s
    // → lanes of 0x3d3d3d3d - 0x3d3d3d3d = 0 ; mov x3, v2.d[0]
    let code = [dup16(0, 1), dup16(1, 2), sub4s(2, 0, 1), umov_d0(3, 2), nop()];
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, 0x3d);
    cpu.set_reg(2, 0x3d);
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_x(3), 0);

    // cmeq v4.16b, v0.16b, v1.16b → all-ones since equal ; mov x5, v4.d[0]
    let code = [dup16(0, 1), dup16(1, 2), cmeq16(4, 0, 1), umov_d0(5, 4), nop()];
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, 0x3d);
    cpu.set_reg(2, 0x3d);
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_x(5), u64::MAX);

    // uhadd v6.16b, v0.16b, v1.16b with unequal bytes (1 + 3) >> 1 = 2
    let code = [dup16(0, 1), dup16(1, 2), uhadd16(6, 0, 1), umov_d0(7, 6), nop()];
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, 1);
    cpu.set_reg(2, 3);
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_x(7), 0x0202_0202_0202_0202);
}

#[test]
fn simd_pairwise_addp() {
    // v1 = {1,1,...}, v2 = {2,2,...}; addp v3.16b, v1.16b, v2.16b →
    // v3[0..7] = v1 pairwise (2), v3[8..15] = v2 pairwise (4).
    let code = [dup16(1, 1), dup16(2, 2), addp16(3, 1, 2), umov_d0(4, 3), nop()];
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, 1);
    cpu.set_reg(2, 2);
    let cpu = run_program(cpu, 0x1000, &code);
    // lane0 = v1[0]+v1[1] = 2, lane7 = v1[14]+v1[15] = 2.
    assert_eq!(cpu.read_x(4) & 0xFF, 2);
    assert_eq!((cpu.read_x(4) >> 56) & 0xFF, 2);
}

#[test]
fn simd_zip1_interleave() {
    // v0 = {0..15} (bytes), v1 = {0x10..0x1f}; zip1 v2.16b, v0.16b, v1.16b
    // → v2[2i] = v0[i], v2[2i+1] = v1[i] for i in 0..8.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x3000);
    cpu.mem.map_zero(0x3000, 0x40).unwrap();
    // ldr q0, [x0] ; ldr q1, [x0, #0x10] ; zip1 v2.16b, v0.16b, v1.16b ;
    // str q2, [x0, #0x20]
    let ldr_q = |rt: u32, imm: u32| {
        0x3DC0_0000u32 | rt | ((imm >> 4) << 10)
    };
    let str_q = |rt: u32, imm: u32| {
        0x3D80_0000u32 | rt | ((imm >> 4) << 10)
    };
    let code = [
        ldr_q(0, 0),
        ldr_q(1, 0x10),
        zip1_16(2, 0, 1),
        str_q(2, 0x20),
        nop(),
    ];
    for i in 0..16u32 {
        cpu.mem.write_u8(0x3000 + i, i as u8).unwrap();
        cpu.mem.write_u8(0x3010 + i, (0x10 + i) as u8).unwrap();
    }
    let cpu = run_program(cpu, 0x1000, &code);
    for i in 0..8u32 {
        assert_eq!(cpu.mem.read_u8(0x3020 + 2 * i).unwrap(), i as u8);
        assert_eq!(cpu.mem.read_u8(0x3020 + 2 * i + 1).unwrap(), 0x10u8 + i as u8);
    }
}

// ---------------- scalar floating point ----------------

// fmov <Vd>.D, <Xn>
fn fmov_dx(rd: u32, rn: u32) -> u32 {
    (1u32 << 31) | (0b0011110 << 24) | (0b01 << 22) | (0b100111 << 16) | (rn << 5) | rd
}
// fmov <Xd>, <Vn>.D
fn fmov_xd(rd: u32, rn: u32) -> u32 {
    (1u32 << 31) | (0b0011110 << 24) | (0b01 << 22) | (0b100110 << 16) | (rn << 5) | rd
}
// fadd <Dd>, <Dn>, <Dm>
fn fadd_d(rd: u32, rn: u32, rm: u32) -> u32 {
    (0b00011110 << 24) | (0b01 << 22) | (1 << 21) | (rm << 16) | (0b00101 << 11) | (rn << 5) | rd
}
#[test]
fn scalar_fp_fadd_fmov() {
    // fmov d0, x1 (2.0) ; fadd d2, d0, d0 (4.0) ; fmov x3, d2
    let code = [fmov_dx(0, 1), fadd_d(2, 0, 0), fmov_xd(3, 2), nop()];
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, 0x4000_0000_0000_0000); // 2.0
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_x(3), 0x4010_0000_0000_0000); // 4.0
}

#[test]
fn scalar_fp_fcvtzs() {
    // scvtf s0, w1 (1000 → 1000.0f) ; fcvtzs w2, s0 (round trip)
    let scvtf_ws = |rd: u32, rn: u32| {
        (0b0011110 << 24) | (0b000010 << 16) | (rn << 5) | rd
    };
    let fcvtzs_ws = |rd: u32, rn: u32| {
        (0b0011110 << 24) | (0b011000 << 16) | (rn << 5) | rd
    };
    let code = [scvtf_ws(0, 1), fcvtzs_ws(2, 0), nop()];
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, 1000);
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_x(2), 1000);
}

// ---------------- multiply-long ----------------

#[test]
fn multiply_long_ops() {
    // UMADDL X2, X0, X1, X3: X2 = X3 + X0*X1
    let umaddl = |rd: u32, rn: u32, rm: u32, ra: u32| {
        (1u32 << 31) | (0b11011101 << 21) | (rm << 16) | (ra << 10) | (rn << 5) | rd
    };
    // SMULH X4, X0, X1: high 64 of signed product
    let smulh = |rd: u32, rn: u32, rm: u32| {
        (1u32 << 31) | (0b11011010 << 21) | (rm << 16) | (31 << 10) | (rn << 5) | rd
    };
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x1000);
    cpu.set_reg(1, 0x200);
    cpu.set_reg(3, 0x42);
    let code = [umaddl(2, 0, 1, 3), nop()];
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_x(2), 0x42 + 0x1000 * 0x200);

    // SMULH of i64::MIN * 2 → -1
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x8000_0000_0000_0000);
    cpu.set_reg(1, 2);
    let code = [smulh(4, 0, 1), nop()];
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_x(4), u64::MAX);
}

// ---------------- Horizon IPC reply synthesis ----------------

#[test]
fn horizon_ipc_reply_success_not_busy() {
    use switch_core::cpu::SyscallMode;
    // A domain IPC request (ICommonStateGetter::ReceiveMessage, cmd 1) must get
    // a success reply: the "SFCO" marker, Result 0 — NOT the AM_BUSY error
    // (0x19280) that wedges hbmenu in its "wait for applet" sleep loop — and
    // the FocusStateChanged applet message (15) in the reply payload.
    let mut cpu = cpu_at(0x1000);
    cpu.syscall_mode = SyscallMode::Horizon;
    let tls = 0x3000u32;
    cpu.mem.map_zero(tls, 0x100).unwrap();
    // msr tpidr_el0, x0 (x0 = tls base)
    cpu.set_reg(0, tls as u64);
    // hipc header: type=4 (request), 12 data words, no special header
    cpu.mem.write_u32(tls, 0x0000_0004).unwrap();
    cpu.mem.write_u32(tls + 4, 0x0c).unwrap();
    // domain header
    cpu.mem.write_u32(tls + 0x10, 0x0010_0001).unwrap(); // type=1, data_size=0x10
    cpu.mem.write_u32(tls + 0x14, 4).unwrap(); // object id
    // CmifInHeader: "SFCI", version 0, command id 1 (ReceiveMessage)
    cpu.mem.write_u32(tls + 0x20, 0x4943_4653).unwrap();
    cpu.mem.write_u32(tls + 0x24, 0).unwrap();
    cpu.mem.write_u32(tls + 0x28, 1).unwrap();
    cpu.mem.write_u32(tls + 0x2c, 0).unwrap();

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(0xD51B_D040u32).to_le_bytes()); // msr tpidr_el0, x0
    bytes.extend_from_slice(&svc(0x21).to_le_bytes());
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.run(2).unwrap();

    assert_eq!(cpu.read_x(0), 0); // Result 0 (success)
    // The reply is coherent: SFCO at the domain out header, Result 0 after it.
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 0x4F43_4653); // "SFCO"
    assert_eq!(cpu.mem.read_u32(tls + 0x24).unwrap(), 0);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0); // Result
    assert_eq!(cpu.mem.read_u32(tls + 0x30).unwrap(), 15); // FocusStateChanged
    assert_ne!(cpu.mem.read_u32(tls + 0x30).unwrap(), 0x19280);
}

#[test]
fn horizon_ipc_reply_focus_state() {
    use switch_core::cpu::SyscallMode;
    // ICommonStateGetter::GetCurrentFocusState (cmd 9) must report InFocus (1)
    // so libnx's applet-mainloop wait loop terminates.
    let mut cpu = cpu_at(0x1000);
    cpu.syscall_mode = SyscallMode::Horizon;
    let tls = 0x3000u32;
    cpu.mem.map_zero(tls, 0x100).unwrap();
    cpu.set_reg(0, tls as u64);
    cpu.mem.write_u32(tls, 0x0000_0004).unwrap();
    cpu.mem.write_u32(tls + 4, 0x0c).unwrap();
    cpu.mem.write_u32(tls + 0x10, 0x0010_0001).unwrap();
    cpu.mem.write_u32(tls + 0x14, 4).unwrap();
    cpu.mem.write_u32(tls + 0x20, 0x4943_4653).unwrap();
    cpu.mem.write_u32(tls + 0x28, 9).unwrap(); // cmd 9 = GetCurrentFocusState

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(0xD51B_D040u32).to_le_bytes());
    bytes.extend_from_slice(&svc(0x21).to_le_bytes());
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.run(2).unwrap();

    assert_eq!(cpu.read_x(0), 0);
    assert_eq!(cpu.mem.read_u32(tls + 0x30).unwrap(), 1); // InFocus
}

#[test]
fn ldrsw_sign_extends_and_loads() {
    // LDRSW must LOAD a 32-bit value and sign-extend it — not be decoded as a
    // store (opc=10 was misread as store, corrupting the target). Store
    // 0xFFFF8001 at [x0] then `ldrsw w1, [x0, #0]` must yield 0xFFFFFFFF_FFFF8001.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x3000);
    cpu.mem.map_zero(0x3000, 0x10).unwrap();
    cpu.mem.write_u32(0x3000, 0xFFFF_8001).unwrap();
    // LDRSW W1, [X0, #0] : size=10, V=0, opc=10, mode=01, imm=0, rn=0, rt=1
    let ldrsw = 0b10u32 << 30 | 0b111 << 27 | 0b01 << 24 | 0b10 << 22 | (0 << 10) | (0 << 5) | 1;
    let code = [ldrsw, nop()];
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_x(1), 0xFFFF_FFFF_FFFF_8001);
}

#[test]
fn gamepad_input_writes_input_reg_and_hid_shmem() {
    use switch_core::cpu::SyscallMode;
    // MapSharedMemory (svc 0x13) with x1=addr, x2=size must back the region
    // with real memory and record it; set_gamepad_state then mirrors the
    // button mask into INPUT_ADDR and the libnx HidSharedMemory player-1
    // layout (npad at 0x3D7C0, full_key_lifo at +0x20).
    let mut cpu = cpu_at(0x1000);
    cpu.syscall_mode = SyscallMode::Horizon;
    cpu.set_reg(1, 0x3000_0000);
    cpu.set_reg(2, 0x40000);
    cpu.mem.map(0x1000, &svc(0x13).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);

    cpu.set_gamepad_state(0x3 /* A|B */, 1000, -2000, 0, 0);

    assert_eq!(cpu.mem.read_u64(switch_core::INPUT_ADDR).unwrap(), 0x3);
    // style_set = FullKey|Handheld
    assert_eq!(cpu.mem.read_u32(0x3000_0000 + 0x3D7C0).unwrap(), 5);
    // header.count == 1, one LIFO entry with buttons + connected attribute
    let lifo = 0x3000_0000 + 0x3D7C0 + 0x20;
    assert_eq!(cpu.mem.read_u64(lifo + 0x18).unwrap(), 1);
    assert_eq!(cpu.mem.read_u64(lifo + 0x30).unwrap(), 0x3);
    assert_eq!(cpu.mem.read_u32(lifo + 0x38).unwrap(), 1000);
    assert_eq!(cpu.mem.read_u32(lifo + 0x3C).unwrap(), 2000u32.wrapping_neg());
    assert_eq!(cpu.mem.read_u32(lifo + 0x48).unwrap(), 1); // IsConnected
}

#[test]
fn logical_immediate_mask_80808080() {
    // 0x3201c3f4 = `mov w20, #0x80808080`. The old decoder rejected it because
    // imms=48 with N=0 has bits above the element size — those are ignored, not
    // "unallocated" (QEMU logic_imm_decode_wmask). This was the first bug that
    // stopped sdl-hello.nro from booting.
    let code = [0x3201c3f4, nop()];
    let cpu = run_program(cpu_at(0x1000), 0x1000, &code);
    assert_eq!(cpu.read_x(20), 0x8080_8080);
}

#[test]
fn ldrsw_register_offset_shifts_by_log2() {
    // `ldrsw x8, [x9, x8, lsl #2]` = 0xb8a87928. The offset is Rm<<2 (the
    // encoded LSL is log2(size), NOT the byte count); shifting by 4 read the
    // wrong jump-table slot, loaded 0 and branched into the table itself.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(8, 0x27);
    cpu.set_reg(9, 0x3000);
    cpu.mem.map_zero(0x3000, 0x200).unwrap();
    cpu.mem.write_u32(0x3000 + 0x27 * 4, 0xFFFF_F25A).unwrap();
    let code = [0xb8a87928, nop()];
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_x(8), 0xFFFF_FFFF_FFFF_F25A);
}

#[test]
fn movi_modified_immediate_cmodes() {
    // MOVI cmode semantics + the split imm8 field (bits 18:16 ++ 9:5), values
    // verified under qemu-aarch64 (real A57 semantics).
    // movi v0.8b, #0x1c  → every byte 0x1c (q=0: upper half cleared)
    let cpu = run_program(cpu_at(0x1000), 0x1000, &[0x0f00e780, nop()]);
    assert_eq!(cpu.read_vreg(0), 0x1c1c1c1c_1c1c1c1c);
    // movi v4.4h, #0x1c  → every halfword 0x001c
    let cpu = run_program(cpu_at(0x1000), 0x1000, &[0x4f008783, nop()]);
    assert_eq!(cpu.read_vreg(3), 0x001c001c_001c001c_001c001c_001c001c);
    // movi v4.4s, #0x1c  → every word 0x1c
    let cpu = run_program(cpu_at(0x1000), 0x1000, &[0x4f000784, nop()]);
    assert_eq!(cpu.read_vreg(4), 0x0000001c_0000001c_0000001c_0000001c);
    // movi v5.4s, #0x1c, lsl #8
    let cpu = run_program(cpu_at(0x1000), 0x1000, &[0x4f002785, nop()]);
    assert_eq!(cpu.read_vreg(5), 0x00001c00_00001c00_00001c00_00001c00);
    // mvni v6.4s, #0x1c  → ~0x1c per word
    let cpu = run_program(cpu_at(0x1000), 0x1000, &[0x6f000786, nop()]);
    assert_eq!(cpu.read_vreg(6), 0xffffffe3_ffffffe3_ffffffe3_ffffffe3);
    // movi v7.2d, #0  (the encoding sdl-hello hit) → zero
    let cpu = run_program(cpu_at(0x1000), 0x1000, &[0x2f00e408, nop()]);
    assert_eq!(cpu.read_vreg(8), 0);
}

#[test]
fn dczid_el0_reports_a57_block_size() {
    // mrs x5, dczid_el0 must return BS=4 (64-byte DC ZVA). musl/newlib memset
    // strides its cache-zero loop by `4 << BS`; BS=0 made it run away forever.
    let cpu = run_program(cpu_at(0x1000), 0x1000, &[0xd53b00e5, nop()]);
    assert_eq!(cpu.read_x(5), 4);
}

#[test]
fn dc_zva_zeroes_64_bytes() {
    // dc zva, x3 = 0xd50b7423: zeroes the 64-byte block at x3 (A57 block size).
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(3, 0x3040);
    cpu.mem.map_zero(0x3000, 0x100).unwrap();
    for i in 0..0x40u32 {
        cpu.mem.write_u8(0x3040 + i, 0xAA).unwrap();
    }
    cpu.mem.write_u8(0x3080, 0xBB).unwrap();
    let code = [0xd50b7423, nop()];
    let cpu = run_program(cpu, 0x1000, &code);
    for i in 0..0x40u32 {
        assert_eq!(cpu.mem.read_u8(0x3040 + i).unwrap(), 0);
    }
    assert_eq!(cpu.mem.read_u8(0x3080).unwrap(), 0xBB);
}

#[test]
fn fpr_load_store_pairs_and_scalar() {
    // str d0, [x8, x10] (register offset, FP 64-bit) = 0xfc2a6900
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(8, 0x3000);
    cpu.set_reg(10, 0x80);
    cpu.set_vreg(0, 0x1122_3344_5566_7788);
    cpu.mem.map_zero(0x3000, 0x200).unwrap();
    let code = [0xfc2a6900, nop()];
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.mem.read_u64(0x3080).unwrap(), 0x1122_3344_5566_7788);

    // stp d8, d9, [x0, #0x70] (FP store pair, D) = 0x6d072408
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x3000);
    cpu.set_vreg(8, 0x0102_0304_0506_0708);
    cpu.set_vreg(9, 0x1112_1314_1516_1718);
    cpu.mem.map_zero(0x3000, 0x200).unwrap();
    let code = [0x6d072408, nop()];
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.mem.read_u64(0x3070).unwrap(), 0x0102_0304_0506_0708);
    assert_eq!(cpu.mem.read_u64(0x3078).unwrap(), 0x1112_1314_1516_1718);
}

#[test]
fn ccmp_eq_sets_carry_for_unsigned_ge() {
    // ccmp x21, x1, #0, eq  with x21=0x20, x1=0x18 should leave C=1,
    // so a following b.hs is taken. Regression caught by libtransistor malloc.
    let code: [u32; 5] = [
        0xd2800a95, // mov x21, #0x20  (0x20 << 5? actually movz x21,#0x20)
        0xd2800301, // mov x1, #0x18
        0xf10000ff, // cmp x7, #0   (sets Z=1; x7 is zero)
        0xfa4102a0, // ccmp x21, x1, #0, eq  (exact instruction from sdl-hello)
        0x540000a2, // b.hs #+20
    ];
    let mut cpu = cpu_at(0x1000);
    let mut bytes = Vec::new();
    for insn in &code { bytes.extend_from_slice(&insn.to_le_bytes()); }
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(code.len() as u64).unwrap();
    assert_eq!(cpu.get_pc(), 0x1010 + 20,
        "b.hs should have been taken; ccmp produced wrong flags");
}
