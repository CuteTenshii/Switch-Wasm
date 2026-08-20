//! CPU correctness tests.
//!
//! These run hand-assembled AArch64 machine code through the interpreter and
//! assert on register, flag, memory and console state. Encodings were
//! verified against QEMU's `a64.decode` where a doubt existed.

use switch_core::cpu::Cpu;

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
fn csel_family_else_ops() {
    // csinv/csinc/csneg must apply invert/increment only to the ELSE operand.
    // x1=5, x2=9. EQ true (cmp x1,x1) → all three select x1. EQ false
    // (cmp x1,x2) → csinv=~9=-10, csinc=10, csneg=-9.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, 5);
    cpu.set_reg(2, 9);
    let eq_true = [
        cmp_reg(1, 1, true),
        0xda820023, // csinv x3, x1, x2, eq
        0x9a820424, // csinc x4, x1, x2, eq
        0xda820425, // csneg x5, x1, x2, eq
    ];
    cpu = run_program(cpu, 0x1000, &eq_true);
    assert_eq!(cpu.read_x(3), 5);
    assert_eq!(cpu.read_x(4), 5);
    assert_eq!(cpu.read_x(5), 5);

    let eq_false = [
        cmp_reg(1, 2, true),
        0xda820023, // csinv x3, x1, x2, eq
        0x9a820424, // csinc x4, x1, x2, eq
        0xda820425, // csneg x5, x1, x2, eq
    ];
    cpu = run_program(cpu, 0x1000, &eq_false);
    assert_eq!(cpu.read_x(3) as i64, -10);
    assert_eq!(cpu.read_x(4), 10);
    assert_eq!(cpu.read_x(5) as i64, -9);
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

    // OutputDebugString(0x3000, 5) logs the string to the console.
    let mut cpu = cpu_at(0x1000);
    cpu.mem.map(0x3000, b"hello").unwrap();
    cpu.set_reg(0, 0x3000);
    cpu.set_reg(1, 5);
    cpu.mem.map(0x1000, &svc(0x27).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.out, b"hello");

    // A null pointer / bogus length is tolerated (no fault).
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0);
    cpu.set_reg(1, 0xFFFFFFFFFFFFFFDCu64);
    cpu.mem.map(0x1000, &svc(0x27).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();

    // ExitProcess halts the machine.
    let mut cpu = cpu_at(0x1000);
    cpu.mem.map(0x1000, &svc(0x07).to_le_bytes()).unwrap();
    let report = cpu.run(1).unwrap();
    assert!(report.halted);

    // GetSystemTick returns the cycle count scaled to ns.
    let mut cpu = cpu_at(0x1000);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&nop().to_le_bytes());
    bytes.extend_from_slice(&svc(0x1E).to_le_bytes());
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.run(2).unwrap();
    assert_eq!(cpu.read_x(0), 1000); // one nop executed before the svc

    // ConnectToNamedPort succeeds with a fake handle returned in X1.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x3000); // name pointer (ignored by the stub)
    cpu.set_reg(1, 4);
    cpu.mem.map(0x1000, &svc(0x1F).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);
    assert_eq!(cpu.read_x(1), 0x1000);

    // SendSyncRequest is a no-op success so service init proceeds.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x1000); // session handle
    cpu.set_reg(1, 0x3000); // ipc buffer pointer
    cpu.set_reg(2, 0x40);
    cpu.mem.map(0x1000, &svc(0x21).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);
}

#[test]
fn horizon_query_memory_and_get_info() {

    // QueryMemory writes a MemoryInfo struct to the out pointer and returns
    // the page info in X1. It reports the contiguous run of pages in the same
    // state as the queried address.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x3000); // MemoryInfo out
    cpu.set_reg(1, 0x4000); // PageInfo out
    cpu.set_reg(2, 0x0800_1000); // queried address
    cpu.mem.map(0x1000, &svc(0x06).to_le_bytes()).unwrap();
    cpu.mem.map_zero(0x0800_0000, 0x1_0000).unwrap(); // mapped 64 KiB run
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);
    assert_eq!(cpu.read_x(1), 0x1000); // mapped page info
    assert_eq!(cpu.mem.read_u64(0x3000).unwrap(), 0x0800_0000); // run base
    assert_eq!(cpu.mem.read_u64(0x3008).unwrap(), 0x1_0000); // run size
    assert_eq!(cpu.mem.read_u32(0x3010).unwrap(), 3); // type
    assert_eq!(cpu.mem.read_u32(0x3018).unwrap(), 0b111); // perm (RWX)

    // An untouched soft-mapped page reports as unmapped (type 0, no perm),
    // which is what lets libnx virtmem find free address space.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x3000);
    cpu.set_reg(1, 0x3040);
    cpu.set_reg(2, 0x1234000);
    cpu.mem.map_zero(0x3000, 0x60).unwrap();
    cpu.mem.map(0x1000, &svc(0x06).to_le_bytes()).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(1).unwrap();
    assert_eq!(cpu.mem.read_u32(0x3010).unwrap(), 0);   // type (unmapped)
    assert_eq!(cpu.mem.read_u32(0x3018).unwrap(), 0);   // perm

    // GetInfo returns the requested value in X1 (the libnx wrapper stores it
    // to the out pointer). InfoType 4 = HeapRegionAddress. Every region this
    // reports has to be an address the emulator can actually represent: guest
    // memory is addressed with a `u32`, and reporting Horizon's real
    // out-of-range region bases had `nnSdk` asking `svcMapPhysicalMemory` to
    // back 0x10_0000_0000.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, 4); // infoType
    cpu.set_reg(2, 0xffff_8001); // CUR_PROCESS_HANDLE
    cpu.mem.map(0x1000, &svc(0x29).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);
    assert_eq!(cpu.read_x(1), u64::from(switch_core::cpu::GUEST_HEAP_REGION_ADDR));
    assert!(cpu.read_x(1) <= u64::from(u32::MAX));

    // InfoType 21/22 = Total/UsedNonSystemMemorySize, which is what `nnSdk`
    // sizes the application heap from — it hands the difference straight to
    // `nn::mem::StandardAllocator::Initialize`, which asserts on a span under
    // 16 KiB. Answering 0 (the old `_ => 0` default) made that difference 0.
    for (info_type, expected) in [(21u64, 0x1E00_0000u64), (22, 0)] {
        let mut cpu = cpu_at(0x1000);
        cpu.set_reg(1, info_type);
        cpu.set_reg(2, 0xffff_8001);
        cpu.mem.map(0x1000, &svc(0x29).to_le_bytes()).unwrap();
        cpu.run(1).unwrap();
        assert_eq!(cpu.read_x(0), 0);
        assert_eq!(cpu.read_x(1), expected);
    }

    // InfoType 16 = SystemResourceSizeTotal, deliberately 0 — see the note on
    // it in `svc.rs`. A non-zero answer switches `nnSdk` onto its virtual
    // address memory manager, which needs kernel machinery this emulator does
    // not have, and `nn::os::AllocateAddressRegion` then fails outright.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, 16);
    cpu.set_reg(2, 0xffff_8001);
    cpu.mem.map(0x1000, &svc(0x29).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(1), 0);

    // InfoType 6 = TotalMemorySize.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, 6);
    cpu.set_reg(2, 0xffff_8001);
    cpu.mem.map(0x1000, &svc(0x29).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);
    assert_eq!(cpu.read_x(1), 0x1E00_0000);

    // InfoType 12 = AslrRegionAddress.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, 12);
    cpu.set_reg(2, 0xffff_8001);
    cpu.mem.map(0x1000, &svc(0x29).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(1), 0x0800_0000);
}

#[test]
fn horizon_map_physical_memory() {
    use switch_core::cpu::GUEST_ALIAS_REGION_ADDR;
    // MapPhysicalMemory(address, size) is how an application built for the
    // 39-bit address space grows its heap — it picks the address itself out of
    // the alias region rather than calling svcSetHeapSize, which is why a
    // retail title never issues syscall 0x01 at all.
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.mem.map(0x1000, &svc(0x2c).to_le_bytes()).unwrap();
    cpu.set_reg(0, u64::from(GUEST_ALIAS_REGION_ADDR));
    cpu.set_reg(1, 0x10_0000);
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);
    // The pages are demand-allocated, so the range reads as zeros and costs
    // nothing until it is written to.
    assert_eq!(cpu.mem.read_u32(GUEST_ALIAS_REGION_ADDR).unwrap(), 0);

    // An unaligned or empty range is rejected, and so is one the emulator
    // cannot address: guest memory is indexed with a `u32`, and silently
    // truncating Horizon's real alias base (0x10_0000_0000) to 0 would map the
    // heap over the null page.
    for (addr, size) in [
        (u64::from(GUEST_ALIAS_REGION_ADDR), 0u64),
        (u64::from(GUEST_ALIAS_REGION_ADDR) + 1, 0x1000),
        (u64::from(GUEST_ALIAS_REGION_ADDR), 0x800),
        (0x10_0000_0000, 0x1000),
    ] {
        let mut cpu = cpu_at(0x1000);
        cpu.bootstrap();
        cpu.set_pc(0x1000);
        cpu.mem.map(0x1000, &svc(0x2c).to_le_bytes()).unwrap();
        cpu.set_reg(0, addr);
        cpu.set_reg(1, size);
        cpu.run(1).unwrap();
        assert_ne!(cpu.read_x(0), 0, "{addr:#x}+{size:#x} should be rejected");
    }
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
    // bit21 is what separates three-same from the copy/permute/table space —
    // every other helper here sets it, and this one used to leave it clear.
    // The decoder ignored bit21, so the malformed encoding still reached
    // UHADD; it is really an INS (element) opcode.
    (1u32 << 30) | (1u32 << 29) | (0b1110 << 24) | (1 << 21) | (rm << 16) | (1 << 10) | (rn << 5) | rd
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

// AdvSIMD table lookup: `0 Q 001110 00 0 Rm 0 len op 00 Rn Rd`. `op` picks
// TBX (1) over TBL (0); `len` is the table size in registers, minus one.
fn tbl_insn(q: u32, len: u32, op: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    (q << 30) | (0b001110 << 24) | (rm << 16) | (len << 13) | (op << 12) | (rn << 5) | rd
}

#[test]
fn simd_table_lookup_gathers_bytes_and_zeroes_misses() {
    let ldr_q = |rt: u32, imm: u32| 0x3DC0_0000u32 | rt | ((imm >> 4) << 10);
    let str_q = |rt: u32, imm: u32| 0x3D80_0000u32 | rt | ((imm >> 4) << 10);
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x3000);
    cpu.mem.map_zero(0x3000, 0x100).unwrap();
    // Two table registers: v0 = 0x10..0x1f, v1 = 0x20..0x2f.
    for i in 0..16u32 {
        cpu.mem.write_u8(0x3000 + i, 0x10 + i as u8).unwrap();
        cpu.mem.write_u8(0x3010 + i, 0x20 + i as u8).unwrap();
    }
    // v2: the low half reverses the first eight table bytes, the high half
    // is 0x40 — past the end of any table this test builds.
    for i in 0..8u32 {
        cpu.mem.write_u8(0x3020 + i, 7 - i as u8).unwrap();
        cpu.mem.write_u8(0x3028 + i, 0x40).unwrap();
    }
    // v3 starts as 0xaa so TBX keeping a byte is distinguishable from TBL
    // zeroing it.
    for i in 0..16u32 {
        cpu.mem.write_u8(0x3030 + i, 0xaa).unwrap();
    }
    let code = [
        ldr_q(0, 0x00),
        ldr_q(1, 0x10),
        ldr_q(2, 0x20),
        ldr_q(3, 0x30),
        tbl_insn(1, 0, 0, 4, 0, 2), // tbl v4.16b, {v0.16b}, v2.16b
        tbl_insn(1, 0, 1, 3, 0, 2), // tbx v3.16b, {v0.16b}, v2.16b
        tbl_insn(0, 0, 0, 5, 0, 2), // tbl v5.8b,  {v0.16b}, v2.8b
        str_q(4, 0x40),
        str_q(3, 0x50),
        str_q(5, 0x60),
        nop(),
    ];
    let cpu = run_program(cpu, 0x1000, &code);
    for i in 0..8u32 {
        // Indices in range gather from the table...
        assert_eq!(cpu.mem.read_u8(0x3040 + i).unwrap(), 0x17 - i as u8);
        assert_eq!(cpu.mem.read_u8(0x3050 + i).unwrap(), 0x17 - i as u8);
        assert_eq!(cpu.mem.read_u8(0x3060 + i).unwrap(), 0x17 - i as u8);
        // ...and out-of-range ones read zero from TBL, but leave TBX's
        // destination byte as it was.
        assert_eq!(cpu.mem.read_u8(0x3048 + i).unwrap(), 0);
        assert_eq!(cpu.mem.read_u8(0x3058 + i).unwrap(), 0xaa);
        // The 8-byte form zeroes the top half of the destination.
        assert_eq!(cpu.mem.read_u8(0x3068 + i).unwrap(), 0);
    }
}

// AdvSIMD copy, element form: `0 1 1 01110000 imm5 0 imm4 1 Rn Rd`.
// `imm5` carries the element size and the destination lane, `imm4` the source
// lane (both shifted down by log2 of the element size).
fn ins_elem_b(rd: u32, dst_index: u32, rn: u32, src_index: u32) -> u32 {
    let imm5 = (dst_index << 1) | 1; // size = B → imm5<0> = 1
    0x6E00_0400u32 | (imm5 << 16) | (src_index << 11) | (rn << 5) | rd
}

#[test]
fn simd_ins_element_moves_one_lane_and_leaves_the_rest() {
    // `INS <Vd>.<Ts>[<i1>], <Vn>.<Ts>[<i2>]` is the `op == 1` half of the
    // AdvSIMD copy group. The group was matched on bits[29:21], which pins
    // `op` to 0, so every one of these fell through to the three-same integer
    // decoder and executed as an unrelated arithmetic op — silently
    // overwriting the whole destination register instead of one lane.
    //
    // The regression this comes from: libnx's `smEncodeName` builds an 8-byte
    // `SmServiceName` in a vector register with one `ldr b<n>, [s, #i]` per
    // character followed by a chain of `ins v31.b[i], v<n>.b[0]`, then
    // `umov x1, v31.d[0]`. Checkpoint asked `sm` for `ns:am2` this way and the
    // request went out with an all-zero name; the session it got back was
    // filed under "", answered nothing, and the guest panicked.
    let mut cpu = cpu_at(0x1000);
    // v31 holds 'n' (as `ldr b31, [x0]` would leave it); one source register
    // per remaining character, each with the byte in lane 0.
    cpu.set_vreg(31, 0x6E);
    for (i, ch) in [b's', b':', b'a', b'm', b'2'].into_iter().enumerate() {
        cpu.set_vreg(29 - i as u8, u128::from(ch));
    }
    let code = [
        ins_elem_b(31, 1, 29, 0),
        ins_elem_b(31, 2, 28, 0),
        ins_elem_b(31, 3, 27, 0),
        ins_elem_b(31, 4, 26, 0),
        ins_elem_b(31, 5, 25, 0),
        0x4E08_3FE1, // umov x1, v31.d[0]
        nop(),
    ];
    assert_eq!(code[0], 0x6E03_07BF); // ins v31.b[1], v29.b[0], as clang emits
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_vreg(31), 0x326D_613A_736E);
    assert_eq!(cpu.read_x(1), u64::from_le_bytes(*b"ns:am2\0\0"));

    // A non-zero source lane, and lanes wider than a byte. INS touches only
    // the destination lane — the top half of Vd is left alone, unlike almost
    // every other AdvSIMD encoding.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, 0x1122_3344_5566_7788_99AA_BBCC_DDEE_FF00);
    cpu.set_vreg(1, 0xFFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF);
    let code = [
        ins_elem_b(1, 15, 0, 8), // ins v1.b[15], v0.b[8]
        // ins v1.s[1], v0.s[3]: imm5 = 0b00100 | (1 << 3), imm4 = 3 << 2
        0x6E00_0400u32 | (0b01100 << 16) | (0b1100 << 11) | (0 << 5) | 1,
        // ins v1.d[0], v0.d[1]: imm5 = 0b01000, imm4 = 1 << 3
        0x6E00_0400u32 | (0b01000 << 16) | (0b1000 << 11) | (0 << 5) | 1,
        nop(),
    ];
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_vreg(1), 0x88FF_FFFF_FFFF_FFFF_1122_3344_5566_7788);
}

#[test]
fn simd_table_lookup_spans_several_registers() {
    let ldr_q = |rt: u32, imm: u32| 0x3DC0_0000u32 | rt | ((imm >> 4) << 10);
    let str_q = |rt: u32, imm: u32| 0x3D80_0000u32 | rt | ((imm >> 4) << 10);
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x3000);
    cpu.mem.map_zero(0x3000, 0x100).unwrap();
    for i in 0..16u32 {
        cpu.mem.write_u8(0x3000 + i, 0x10 + i as u8).unwrap();
        cpu.mem.write_u8(0x3010 + i, 0x20 + i as u8).unwrap();
    }
    // A {v0, v1} table is 32 bytes, so 31 is the last index that hits and 32
    // is the first that misses.
    for (i, idx) in [0u8, 15, 16, 31, 32].into_iter().enumerate() {
        cpu.mem.write_u8(0x3020 + i as u32, idx).unwrap();
    }
    let code = [
        ldr_q(0, 0x00),
        ldr_q(1, 0x10),
        ldr_q(2, 0x20),
        tbl_insn(1, 1, 0, 3, 0, 2), // tbl v3.16b, {v0.16b, v1.16b}, v2.16b
        str_q(3, 0x40),
        nop(),
    ];
    let cpu = run_program(cpu, 0x1000, &code);
    let out: Vec<u8> = (0..5).map(|i| cpu.mem.read_u8(0x3040 + i).unwrap()).collect();
    assert_eq!(out, vec![0x10, 0x1f, 0x20, 0x2f, 0x00]);
}

// AdvSIMD across lanes: `0 Q U 01110 size 11000 opcode(5) 10 Rn Rd`.
fn across_lanes(q: u32, u: u32, size: u32, opcode: u32, rd: u32, rn: u32) -> u32 {
    (q << 30) | (u << 29) | (0b01110 << 24) | (size << 22) | (0b11000 << 17)
        | (opcode << 12) | (0b10 << 10) | (rn << 5) | rd
}

#[test]
fn simd_across_lanes_reduce() {
    // v0.4s = {3, 7, 2, -1(0xFFFFFFFF)}.
    let ldr_q = |rt: u32, imm: u32| 0x3DC0_0000u32 | rt | ((imm >> 4) << 10);
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x3000);
    cpu.mem.map_zero(0x3000, 0x20).unwrap();
    for (i, v) in [3u32, 7, 2, 0xFFFF_FFFF].into_iter().enumerate() {
        cpu.mem.write_u32(0x3000 + 4 * i as u32, v).unwrap();
    }
    // smaxv s1, v0.4s = 0x4eb0a801 ; sminv s2, v0.4s = 0x4eb1a802
    let smaxv = across_lanes(1, 0, 0b10, 0b01010, 1, 0);
    let sminv = across_lanes(1, 0, 0b10, 0b11010, 2, 0);
    let code = [ldr_q(0, 0), smaxv, sminv, umov_d0(3, 1), umov_d0(4, 2), nop()];
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_x(3) as u32, 7); // signed max ignores the -1 lane.
    assert_eq!(cpu.read_x(4) as u32, 0xFFFF_FFFF); // signed min picks it.

    // v0.4s = {1, 2, 3, 4}; uaddlv d5, v0.4s = 10, widened into a 64-bit lane.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x3000);
    cpu.mem.map_zero(0x3000, 0x20).unwrap();
    for (i, v) in [1u32, 2, 3, 4].into_iter().enumerate() {
        cpu.mem.write_u32(0x3000 + 4 * i as u32, v).unwrap();
    }
    let uaddlv = across_lanes(1, 1, 0b10, 0b00011, 5, 0);
    let code = [ldr_q(0, 0), uaddlv, umov_d0(6, 5), nop()];
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_x(6), 10);
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
    // `scvtf s0, w1` = 0x1e220020 (1000 → 1000.0f) then `fcvtzs w2, s0` =
    // 0x1e380002 back again. Bit 21 is a fixed 1 in this encoding class, so the
    // operation is rmode (bits[20:19]) and opcode (bits[18:16]); the old
    // hand-assembled encodings here had bit 21 clear, which is not a valid
    // instruction at all.
    let code = [0x1e22_0020u32, 0x1e38_0002, nop()];
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

/// A domain request carrying raw arguments after the CmifInHeader. The reply
/// overwrites the request in TLS, so the payload has to go in before the
/// request runs rather than by re-running it.
fn ipc_request_with_payload(cpu: &mut Cpu, handle: u64, object_id: u32, cmd: u32, payload: &[u8]) {
    build_ipc_request(cpu, 4, Some(object_id), cmd);
    // No buffer descriptors, so the data area starts at 0x10: the domain
    // header, then the CmifInHeader at 0x20, then the arguments at 0x30.
    let tls = cpu.tls_base();
    for (i, &b) in payload.iter().enumerate() {
        cpu.mem.write_u8(tls + 0x30 + i as u32, b).unwrap();
    }
    run_ipc_request(cpu, handle);
}

/// Drive one IPC request at `handle` and return the CPU. The request is built
/// in the guest's own TLS buffer the way `libnx` marshals a CMIF message:
/// hipc header, an optional `CmifDomainInHeader`, then the `SFCI` in-header
/// carrying the command id.
fn ipc_request(cpu: &mut Cpu, handle: u64, msg_type: u32, object_id: Option<u32>, cmd: u32) {
    build_ipc_request(cpu, msg_type, object_id, cmd);
    run_ipc_request(cpu, handle);
}

/// Marshal a request into TLS without sending it.
fn build_ipc_request(cpu: &mut Cpu, msg_type: u32, object_id: Option<u32>, cmd: u32) {
    let tls = cpu.tls_base();
    for i in (0..0x100u32).step_by(4) {
        cpu.mem.write_u32(tls + i, 0).unwrap();
    }
    cpu.mem.write_u32(tls, msg_type).unwrap();
    cpu.mem.write_u32(tls + 4, 0x0c).unwrap();
    let cmif = match object_id {
        Some(obj) => {
            // CmifDomainRequestType_SendMessage, 0x10 bytes of data.
            cpu.mem.write_u32(tls + 0x10, 0x0010_0001).unwrap();
            cpu.mem.write_u32(tls + 0x14, obj).unwrap();
            tls + 0x20
        }
        None => tls + 0x10,
    };
    cpu.mem.write_u32(cmif, 0x4943_4653).unwrap(); // "SFCI"
    cpu.mem.write_u32(cmif + 8, cmd).unwrap();
}

/// Send whatever is marshalled in TLS to `handle`.
fn run_ipc_request(cpu: &mut Cpu, handle: u64) {
    cpu.set_reg(0, handle);
    let pc = cpu.get_pc();
    cpu.mem.map(pc, &svc(0x21).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    cpu.set_pc(pc);
}

/// A bootstrapped Horizon CPU with `appletOE` already bound to a handle and
/// converted to a domain, plus the object ids of the `IApplicationProxy` and
/// `ICommonStateGetter` opened through it.
fn applet_chain() -> (Cpu, u64, u32, u32) {
    const APPLET: u64 = 0x1000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(APPLET, "appletOE");
    let tls = cpu.tls_base();

    // Control::ConvertToDomain on the root session -> IApplicationProxyService.
    ipc_request(&mut cpu, APPLET, 5, None, 0);
    let proxy_service = cpu.mem.read_u32(tls + 0x20).unwrap();
    // IApplicationProxyService::OpenApplicationProxy -> IApplicationProxy.
    ipc_request(&mut cpu, APPLET, 4, Some(proxy_service), 0);
    let proxy = cpu.mem.read_u32(tls + 0x30).unwrap();
    // IApplicationProxy::GetCommonStateGetter.
    ipc_request(&mut cpu, APPLET, 4, Some(proxy), 0);
    let state_getter = cpu.mem.read_u32(tls + 0x30).unwrap();
    (cpu, APPLET, proxy, state_getter)
}

/// Build an IPC request carrying one map-alias send buffer, and run it.
fn ipc_request_with_buffer(
    cpu: &mut Cpu,
    handle: u64,
    object_id: u32,
    cmd: u32,
    buf: u32,
    len: u32,
    recv: bool,
    payload: &[u8],
) {
    let tls = cpu.tls_base();
    for i in (0..0x100u32).step_by(4) {
        cpu.mem.write_u32(tls + i, 0).unwrap();
    }
    // hdr1: type 4 (Request), one buffer — send buffers count in bits 23:20,
    // receive buffers in 27:24. Either way it is one 12-byte descriptor, so
    // the aligned data area lands at 0x20.
    cpu.mem.write_u32(tls, 4 | (1 << if recv { 24 } else { 20 })).unwrap();
    cpu.mem.write_u32(tls + 4, 0x0c).unwrap();
    // HipcBufferDescriptor: size, address, then the high bits (all zero for a
    // 32-bit guest address).
    cpu.mem.write_u32(tls + 0x08, len).unwrap();
    cpu.mem.write_u32(tls + 0x0c, buf).unwrap();
    cpu.mem.write_u32(tls + 0x10, 0).unwrap();
    // One descriptor pushes the aligned data area out to 0x20.
    cpu.mem.write_u32(tls + 0x20, 0x0010_0001).unwrap();
    cpu.mem.write_u32(tls + 0x24, object_id).unwrap();
    cpu.mem.write_u32(tls + 0x30, 0x4943_4653).unwrap(); // "SFCI"
    cpu.mem.write_u32(tls + 0x38, cmd).unwrap();
    // The command's own arguments follow the 16-byte CmifInHeader.
    for (i, &b) in payload.iter().enumerate() {
        cpu.mem.write_u8(tls + 0x40 + i as u32, b).unwrap();
    }
    cpu.set_reg(0, handle);
    let pc = cpu.get_pc();
    cpu.mem.map(pc, &svc(0x21).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    cpu.set_pc(pc);
}

/// Write one `lm` LogPacket at `addr` and return its total length.
fn write_log_packet(cpu: &mut Cpu, addr: u32, flags: u8, severity: u8, tlvs: &[(u8, &[u8])]) -> u32 {
    let mut payload = Vec::new();
    for &(key, data) in tlvs {
        payload.push(key);
        payload.push(data.len() as u8);
        payload.extend_from_slice(data);
    }
    for i in 0..0x18u32 {
        cpu.mem.write_u8(addr + i, 0).unwrap();
    }
    cpu.mem.write_u8(addr + 0x10, flags).unwrap();
    cpu.mem.write_u8(addr + 0x12, severity).unwrap();
    cpu.mem.write_u32(addr + 0x14, payload.len() as u32).unwrap();
    for (i, &b) in payload.iter().enumerate() {
        cpu.mem.write_u8(addr + 0x18 + i as u32, b).unwrap();
    }
    0x18 + payload.len() as u32
}

/// Run one `svcWaitSynchronization` over `handles`, returning (result, index).
fn wait_sync(cpu: &mut Cpu, handles: &[u32], timeout: i64) -> (u64, u64) {
    const ARRAY: u32 = 0x7000;
    for (i, &h) in handles.iter().enumerate() {
        cpu.mem.write_u32(ARRAY + (i as u32) * 4, h).unwrap();
    }
    cpu.set_reg(1, u64::from(ARRAY));
    cpu.set_reg(2, handles.len() as u64);
    cpu.set_reg(3, timeout as u64);
    let pc = cpu.get_pc();
    cpu.mem.map(pc, &svc(0x18).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    cpu.set_pc(pc);
    (cpu.read_x(0), cpu.read_x(1))
}

#[test]
fn ssl_keeps_context_state_and_refuses_connections() {
    // ssl is the system TLS stack: a title asks the OS to build connections
    // rather than bringing its own implementation. The local half -- contexts
    // and their options -- is real here; the connection half is not, because
    // there is no socket layer under it.
    const SSL: u64 = 0x9000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(SSL, "ssl");
    let tls = cpu.tls_base();
    ipc_request(&mut cpu, SSL, 5, None, 0); // ConvertToDomain
    let service = cpu.mem.read_u32(tls + 0x20).unwrap();

    // SetInterfaceVersion is the only ssl command an offline retail title
    // issues, because ssl is in its NPDM service list and nnSdk initialises it
    // at startup regardless.
    ipc_request_with_payload(&mut cpu, SSL, service, 5, &4u32.to_le_bytes());
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);

    // CreateContext -> ISslContext, and the count follows it.
    ipc_request(&mut cpu, SSL, 4, Some(service), 1);
    assert_eq!(cpu.mem.read_u32(tls + 0x30).unwrap(), 0);
    ipc_request(&mut cpu, SSL, 4, Some(service), 0);
    let context = cpu.mem.read_u32(tls + 0x30).unwrap();
    assert_ne!(context, service);
    ipc_request(&mut cpu, SSL, 4, Some(service), 1);
    assert_eq!(cpu.mem.read_u32(tls + 0x30).unwrap(), 1);

    // Options are per-context state a caller reads back.
    let mut args = Vec::new();
    args.extend_from_slice(&2u32.to_le_bytes()); // option
    args.extend_from_slice(&1u32.to_le_bytes()); // value
    ipc_request_with_payload(&mut cpu, SSL, context, 0, &args);
    ipc_request_with_payload(&mut cpu, SSL, context, 1, &2u32.to_le_bytes());
    assert_eq!(cpu.mem.read_u32(tls + 0x30).unwrap(), 1);
    // An option never set reads as 0 rather than as another option's value.
    ipc_request_with_payload(&mut cpu, SSL, context, 1, &7u32.to_le_bytes());
    assert_eq!(cpu.mem.read_u32(tls + 0x30).unwrap(), 0);

    // CreateConnection reports itself rather than handing back a connection
    // that can never connect.
    const UNKNOWN_COMMAND_ID: u32 = 10 | (221 << 9);
    ipc_request(&mut cpu, SSL, 4, Some(context), 2);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), UNKNOWN_COMMAND_ID);
}

/// Open `hid` and convert it to a domain: (cpu, session handle, IHidServer).
fn hid_server() -> (Cpu, u64, u32) {
    const HID: u64 = 0x9000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(HID, "hid");
    let tls = cpu.tls_base();
    // A control reply is not a domain reply: SFCO lands at 0x10 and the raw
    // data (here the new domain object id) at 0x20.
    ipc_request(&mut cpu, HID, 5, None, 0); // ConvertToDomain
    let server = cpu.mem.read_u32(tls + 0x20).unwrap();
    (cpu, HID, server)
}

#[test]
fn hid_hands_over_the_input_shared_memory() {
    // The input *data* lives in a shared memory region the guest reads
    // directly; hid's IPC is the negotiation that hands it over. libnx got
    // working input out of the old fabricated reply only because it maps that
    // region by size and this emulator recognises it that way -- nnSdk calls a
    // method on the IAppletResource it is given, and an object id is not one.
    let (mut cpu, hid, server) = hid_server();
    let tls = cpu.tls_base();

    ipc_request(&mut cpu, hid, 4, Some(server), 0); // CreateAppletResource
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);
    let resource = cpu.mem.read_u32(tls + 0x30).unwrap();
    assert_ne!(resource, server);

    // GetSharedMemoryHandle -> a copy handle, not a move handle.
    ipc_request(&mut cpu, hid, 4, Some(resource), 0);
    assert_eq!(cpu.mem.read_u32(tls + 0x08).unwrap(), 1 << 1);
    assert_ne!(cpu.mem.read_u32(tls + 0x0c).unwrap(), 0);

    // QueryPointerBufferSize has to be non-zero: nn::hid::SetSupportedNpadIdType
    // marshals its id array as a pointer buffer, and nnSdk checks the
    // negotiated size before it sends.
    ipc_request(&mut cpu, hid, 5, None, 3);
    assert_ne!(cpu.mem.read_u16(tls + 0x20).unwrap(), 0);
}

#[test]
fn hid_reads_back_what_the_guest_configured() {
    // A caller that sets a controller style set and reads back something else
    // decides the pad it wanted is not there -- which is what the generic
    // reply's incrementing object id looked like.
    let (mut cpu, hid, server) = hid_server();
    let tls = cpu.tls_base();
    const STYLE_SET: u32 = 0b1101;

    ipc_request_with_payload(&mut cpu, hid, server, 100, &STYLE_SET.to_le_bytes());
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);
    ipc_request(&mut cpu, hid, 4, Some(server), 101);
    assert_eq!(cpu.mem.read_u32(tls + 0x30).unwrap(), STYLE_SET);

    // Set/GetNpadJoyHoldType: the hold type follows the aruid.
    let mut args = [0u8; 16];
    args[8..].copy_from_slice(&1u64.to_le_bytes());
    ipc_request_with_payload(&mut cpu, hid, server, 120, &args);
    ipc_request(&mut cpu, hid, 4, Some(server), 121);
    assert_eq!(cpu.mem.read_u64(tls + 0x30).unwrap(), 1);
}

#[test]
fn hid_vibration_reaches_the_host() {
    // SendVibrationValue(handle, HidVibrationValue, aruid): the value is four
    // floats, so the two band amplitudes sit at +4 and +0xc after the u32
    // handle. The frontend maps them onto dual-rumble's magnitudes.
    let (mut cpu, hid, server) = hid_server();
    let tls = cpu.tls_base();

    let mut args = Vec::new();
    args.extend_from_slice(&0u32.to_le_bytes()); // device handle
    args.extend_from_slice(&0.75f32.to_bits().to_le_bytes()); // amp_low
    args.extend_from_slice(&160.0f32.to_bits().to_le_bytes()); // freq_low
    args.extend_from_slice(&0.25f32.to_bits().to_le_bytes()); // amp_high
    args.extend_from_slice(&320.0f32.to_bits().to_le_bytes()); // freq_high
    ipc_request_with_payload(&mut cpu, hid, server, 201, &args);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);
    assert_eq!(cpu.vibration(), (0.75, 0.25));

    // GetActualVibrationValue reports what is playing.
    ipc_request(&mut cpu, hid, 4, Some(server), 202);
    assert_eq!(f32::from_bits(cpu.mem.read_u32(tls + 0x30).unwrap()), 0.75);
    assert_eq!(f32::from_bits(cpu.mem.read_u32(tls + 0x38).unwrap()), 0.25);

    // Out of range or not finite is clamped rather than handed to the browser.
    let mut args = vec![0u8; 4];
    args.extend_from_slice(&5.0f32.to_bits().to_le_bytes());
    args.extend_from_slice(&0u32.to_le_bytes());
    args.extend_from_slice(&f32::NAN.to_bits().to_le_bytes());
    args.extend_from_slice(&0u32.to_le_bytes());
    ipc_request_with_payload(&mut cpu, hid, server, 201, &args);
    assert_eq!(cpu.vibration(), (1.0, 0.0));
}

#[test]
fn hid_sys_is_its_own_interface_and_answers_before_any_command() {
    // `libnx` opens hid:sys in hidsysInitialize and records the session's
    // pointer buffer size on it before sending anything, so for a title that
    // never calls a hid:sys command -- Checkpoint is one -- opening the
    // service *is* the only traffic there ever is. With hid:sys unrouted that
    // control request fell through to the generic reply and was answered with
    // a fabricated object id, exactly the way ns:am2 was.
    const HIDSYS: u64 = 0x9100;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(HIDSYS, "hid:sys");
    let tls = cpu.tls_base();

    ipc_request(&mut cpu, HIDSYS, 5, None, 3); // QueryPointerBufferSize
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0);
    assert_eq!(cpu.mem.read_u16(tls + 0x20).unwrap(), 0x1000);

    ipc_request(&mut cpu, HIDSYS, 5, None, 0); // ConvertToDomain
    let server = cpu.mem.read_u32(tls + 0x20).unwrap();

    // EnableAppletToGetInput: a setter over state this emulator does not have.
    ipc_request(&mut cpu, HIDSYS, 4, Some(server), 503);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);

    // GetUniquePadIds -> an s64 count. A unique pad is a *detachable*
    // controller and the one here is the built-in handheld pad, so there are
    // none and the pointer buffer is left alone.
    ipc_request(&mut cpu, HIDSYS, 4, Some(server), 703);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);
    assert_eq!(cpu.mem.read_u64(tls + 0x30).unwrap(), 0);

    // AcquireHomeButtonEventHandle -> a copy handle. There is no Home button,
    // so it is handed out and never signalled.
    ipc_request(&mut cpu, HIDSYS, 4, Some(server), 101);
    assert_eq!(cpu.mem.read_u32(tls + 0x08).unwrap(), 1 << 1);
    assert_ne!(cpu.mem.read_u32(tls + 0x0c).unwrap(), 0);

    // Converting the session to a domain must not quietly turn it into
    // IHidServer: command 0 there is CreateAppletResource, and hid:sys has no
    // command 0 at all.
    const UNKNOWN_COMMAND_ID: u32 = 10 | (221 << 9);
    ipc_request(&mut cpu, HIDSYS, 4, Some(server), 0);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), UNKNOWN_COMMAND_ID);
}

#[test]
fn events_are_copy_handles_and_start_unsignalled() {
    // Every event a service hands out is a **copy** handle: a move handle
    // transfers ownership and lives in a different field of the handle
    // descriptor, so an event sent in the move slot is read back as 0. That is
    // why nnSdk spent whole boots waiting on handle 0 after asking for
    // GetGpuErrorDetectedSystemEvent.
    const APPLET: u64 = 0x9000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(APPLET, "appletOE");
    let tls = cpu.tls_base();

    ipc_request(&mut cpu, APPLET, 5, None, 0);
    let proxy_service = cpu.mem.read_u32(tls + 0x20).unwrap();
    ipc_request(&mut cpu, APPLET, 4, Some(proxy_service), 0);
    let proxy = cpu.mem.read_u32(tls + 0x30).unwrap();
    ipc_request(&mut cpu, APPLET, 4, Some(proxy), 20); // IApplicationFunctions
    let functions = cpu.mem.read_u32(tls + 0x30).unwrap();

    // GetGpuErrorDetectedSystemEvent.
    ipc_request(&mut cpu, APPLET, 4, Some(functions), 130);
    // { send_pid:1, num_copy:4, num_move:4 } -- one copy handle, no move ones.
    assert_eq!(cpu.mem.read_u32(tls + 0x08).unwrap(), 1 << 1);
    let event = cpu.mem.read_u32(tls + 0x0c).unwrap();
    assert_ne!(event, 0, "the guest must receive a real handle");

    // Nothing has fired it, so a poll times out. Reporting the wait satisfied
    // is what told nn::oe::GpuErrorHandler that the GPU had faulted.
    const RESULT_TIMED_OUT: u64 = 0xEA01;
    let (result, _) = wait_sync(&mut cpu, &[event], 0);
    assert_eq!(result, RESULT_TIMED_OUT);

    // A second event, left unsignalled, so the index below is a real position
    // rather than "the first handle".
    ipc_request(&mut cpu, APPLET, 4, Some(proxy), 0); // ICommonStateGetter
    let state_getter = cpu.mem.read_u32(tls + 0x30).unwrap();
    ipc_request(&mut cpu, APPLET, 4, Some(state_getter), 0); // GetEventHandle
    let quiet = cpu.mem.read_u32(tls + 0x0c).unwrap();
    assert_ne!(quiet, event);

    // Once signalled it reports the index that fired, and consumes it: these
    // are auto-clear events, so a second poll times out again.
    cpu.signal_event(u64::from(event));
    let (result, index) = wait_sync(&mut cpu, &[quiet, event], 0);
    assert_eq!(result, 0);
    assert_eq!(index, 1, "the index of the handle that fired, not a count");
    let (result, _) = wait_sync(&mut cpu, &[event], 0);
    assert_eq!(result, RESULT_TIMED_OUT);

    // A handle this emulator does not model as an event is still treated as
    // ready, which is what keeps thread handles and unmodelled service handles
    // behaving as they always have.
    let (result, index) = wait_sync(&mut cpu, &[0x1234], 0);
    assert_eq!(result, 0);
    assert_eq!(index, 0);
}

#[test]
fn control_clone_hands_back_a_working_session() {
    // CloneCurrentObject (control command 2) duplicates a session, and the
    // reply has to carry a **new session handle as a move handle**. Answering
    // it with a bare success and no handle left nnSdk -- which clones fsp-srv
    // before mounting anything -- talking to handle 0, so nn::fs::MountRom
    // failed without ever issuing a filesystem command.
    // Clear of `alloc_handle`'s own range, which starts at 0x1000 -- a real
    // session handle always comes from there, but this one is hand-registered.
    const FS: u64 = 0x9000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(FS, "fsp-srv");
    let tls = cpu.tls_base();

    // Convert to a domain first, so the clone has objects to inherit.
    ipc_request(&mut cpu, FS, 5, None, 0);
    let object = cpu.mem.read_u32(tls + 0x20).unwrap();

    ipc_request(&mut cpu, FS, 5, None, 2); // CloneCurrentObject
    assert_eq!(cpu.read_x(0), 0);
    // Move handles land right after the 8-byte hipc header: a descriptor word
    // then the handles themselves.
    let clone = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    assert_ne!(clone, 0, "clone must hand back a real handle, not 0");
    assert_ne!(clone, FS, "the clone is a separate session");

    // The clone reaches the same service, holding the same domain objects.
    let handles = cpu.service_handles_snapshot();
    assert!(handles.iter().any(|(h, name)| *h == clone && name == "fsp-srv"));
    ipc_request(&mut cpu, clone, 4, Some(object), 1); // SetCurrentProcess
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);
}

#[test]
fn storage_read_uses_the_istorage_field_layout() {
    // IStorage::Read is (s64 offset, u64 size) -- *not* IFile::Read, which
    // leads with a u32 option and pads to 8, putting its offset at +8 and its
    // size at +0x10. Reading those two fields here meant every RomFS read came
    // back as "0 bytes at offset 0x50": the guest mounted its RomFS, parsed an
    // empty header, and found none of its own files.
    const FS: u64 = 0x1000;
    const OUT: u32 = 0x6000;
    let romfs: Vec<u8> = (0..64u8).collect();
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.set_romfs(romfs.clone());
    cpu.register_service_handle(FS, "fsp-srv-storage");
    let tls = cpu.tls_base();

    // Read(offset = 4, size = 8).
    let mut args = Vec::new();
    args.extend_from_slice(&4u64.to_le_bytes()); // offset
    args.extend_from_slice(&8u64.to_le_bytes()); // size
    ipc_request_with_buffer(&mut cpu, FS, 1, 0, OUT, 16, true, &args);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);
    for i in 0..8u32 {
        assert_eq!(cpu.mem.read_u8(OUT + i).unwrap(), romfs[4 + i as usize], "byte {i}");
    }
    // Nothing past the requested size is touched.
    assert_eq!(cpu.mem.read_u8(OUT + 8).unwrap(), 0);

    // A read that runs off the end is clamped rather than faulting.
    let mut args = Vec::new();
    args.extend_from_slice(&(romfs.len() as u64 - 2).to_le_bytes());
    args.extend_from_slice(&64u64.to_le_bytes());
    ipc_request_with_buffer(&mut cpu, FS, 1, 0, OUT, 64, true, &args);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);
    assert_eq!(cpu.mem.read_u8(OUT).unwrap(), romfs[romfs.len() - 2]);

    // GetSize reports the whole RomFS.
    ipc_request(&mut cpu, FS, 4, Some(1), 4);
    assert_eq!(cpu.mem.read_u64(tls + 0x30).unwrap(), romfs.len() as u64);
}

#[test]
fn lm_writes_the_guests_own_log_to_the_console() {
    const LM: u64 = 0x1000;
    const PACKET: u32 = 0x5000;
    const KEY_TEXT: u8 = 2;
    const KEY_MODULE: u8 = 6;
    const HEAD: u8 = 1;
    const TAIL: u8 = 2;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(LM, "lm");
    let tls = cpu.tls_base();

    ipc_request(&mut cpu, LM, 5, None, 0); // Control::ConvertToDomain
    let service = cpu.mem.read_u32(tls + 0x20).unwrap();
    ipc_request(&mut cpu, LM, 4, Some(service), 0); // OpenLogger
    let logger = cpu.mem.read_u32(tls + 0x30).unwrap();

    // One whole message in a single packet: severity 3 is Error, and the
    // module name comes from key 6, the text from key 2.
    let len = write_log_packet(
        &mut cpu,
        PACKET,
        HEAD | TAIL,
        3,
        &[(KEY_MODULE, b"Game"), (KEY_TEXT, b"hello world")],
    );
    ipc_request_with_buffer(&mut cpu, LM, logger, 0, PACKET, len, false, &[]);
    assert_eq!(String::from_utf8_lossy(&cpu.out), "[lm/ERROR/Game] hello world\n");

    // A message split across packets: only the head carries the prefix and
    // only the tail ends the line, so the two halves join into one message.
    cpu.out.clear();
    let len = write_log_packet(&mut cpu, PACKET, HEAD, 1, &[(KEY_TEXT, b"split ")]);
    ipc_request_with_buffer(&mut cpu, LM, logger, 0, PACKET, len, false, &[]);
    let len = write_log_packet(&mut cpu, PACKET, TAIL, 1, &[(KEY_TEXT, b"message")]);
    ipc_request_with_buffer(&mut cpu, LM, logger, 0, PACKET, len, false, &[]);
    assert_eq!(String::from_utf8_lossy(&cpu.out), "[lm/INFO] split message\n");

    // A packet claiming more payload than the buffer holds is trusted only as
    // far as the buffer goes, rather than walking off the end of the mapping.
    cpu.out.clear();
    let len = write_log_packet(&mut cpu, PACKET, HEAD | TAIL, 0, &[(KEY_TEXT, b"truncated")]);
    cpu.mem.write_u32(PACKET + 0x14, 0xFFFF).unwrap();
    ipc_request_with_buffer(&mut cpu, LM, logger, 0, PACKET, len, false, &[]);
    assert_eq!(String::from_utf8_lossy(&cpu.out), "[lm/TRACE] truncated\n");
}

#[test]
fn pctl_reports_parental_controls_off() {
    const PCTL: u64 = 0x1000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(PCTL, "pctl");
    let tls = cpu.tls_base();

    // Control::ConvertToDomain -> IParentalControlServiceFactory, then
    // CreateServiceWithoutInitialize -> IParentalControlService.
    ipc_request(&mut cpu, PCTL, 5, None, 0);
    let factory = cpu.mem.read_u32(tls + 0x20).unwrap();
    ipc_request(&mut cpu, PCTL, 4, Some(factory), 1);
    let service = cpu.mem.read_u32(tls + 0x30).unwrap();

    // A permission check answers with a bare Result: success *is* "permitted",
    // and a restriction is an error the caller checks for by value.
    for cmd in [1001u32, 1004, 1013, 1017] {
        ipc_request(&mut cpu, PCTL, 4, Some(service), cmd);
        assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0, "cmd {cmd}");
    }

    // The two query families read in opposite directions, and answering both
    // the same way would report free communication as unavailable.
    for cmd in [1031u32, 1010, 1453, 1455] {
        ipc_request(&mut cpu, PCTL, 4, Some(service), cmd);
        assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0, "cmd {cmd}");
        assert_eq!(cpu.mem.read_u8(tls + 0x30).unwrap(), 0, "cmd {cmd} restricted");
    }
    for cmd in [1018u32, 1065] {
        ipc_request(&mut cpu, PCTL, 4, Some(service), cmd);
        assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0, "cmd {cmd}");
        assert_eq!(cpu.mem.read_u8(tls + 0x30).unwrap(), 1, "cmd {cmd} allowed");
    }

    // Anything else still reports honestly rather than fabricating a success.
    const UNKNOWN_COMMAND_ID: u32 = 10 | (221 << 9);
    ipc_request(&mut cpu, PCTL, 4, Some(service), 1203); // SetPinCode
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), UNKNOWN_COMMAND_ID);
}

#[test]
fn applet_common_state_getter_reports_focus_once() {
    // ICommonStateGetter::ReceiveMessage (cmd 1) must hand out the startup
    // FocusStateChanged (15) exactly once and then report "no message" — NOT
    // the AM_BUSY error (0x19280) that wedges hbmenu in its "wait for applet"
    // sleep loop, and not a fresh focus change on every poll, which made JKSV
    // treat every frame as a new focus transition.
    let (mut cpu, handle, _proxy, state_getter) = applet_chain();
    let tls = cpu.tls_base();

    ipc_request(&mut cpu, handle, 4, Some(state_getter), 1);
    assert_eq!(cpu.read_x(0), 0); // svc result
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 0x4F43_4653); // "SFCO"
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0); // Result: success
    assert_ne!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0x19280);
    assert_eq!(cpu.mem.read_u32(tls + 0x30).unwrap(), 15); // FocusStateChanged

    ipc_request(&mut cpu, handle, 4, Some(state_getter), 1);
    const NO_MESSAGES: u32 = 128 | (3 << 9);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), NO_MESSAGES);

    // GetCurrentFocusState (cmd 9) reports InFocus so libnx's applet-mainloop
    // wait loop terminates.
    ipc_request(&mut cpu, handle, 4, Some(state_getter), 9);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);
    assert_eq!(cpu.mem.read_u32(tls + 0x30).unwrap(), 1);
}

#[test]
fn applet_unimplemented_command_is_an_error_not_a_fake_success() {
    // An `am` command with no implementation behind it must report cmif's
    // "unknown command id" rather than a bare success. Everything `am` returns
    // is a live handle or a piece of applet state the caller then acts on, so
    // a fabricated success is a wrong answer the guest believes: answering
    // IApplicationFunctions::GetGpuErrorDetectedSystemEvent that way left
    // nnSdk's system worker waiting on handle 0.
    const UNKNOWN_COMMAND_ID: u32 = 10 | (221 << 9);
    let (mut cpu, handle, proxy, _state_getter) = applet_chain();
    let tls = cpu.tls_base();

    // IApplicationProxy::GetDisplayController, then a command it does not have.
    ipc_request(&mut cpu, handle, 4, Some(proxy), 4);
    let display_controller = cpu.mem.read_u32(tls + 0x30).unwrap();
    ipc_request(&mut cpu, handle, 4, Some(display_controller), 8);
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 0x4F43_4653); // "SFCO"
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), UNKNOWN_COMMAND_ID);
}

#[test]
fn applet_control_command_with_context_is_not_a_normal_command() {
    // nnSdk sends every message in the "with context" encoding —
    // ControlWithContext (7) rather than Control (5). Reading only type 5 as a
    // control message turned `appletOE`'s opening QueryPointerBufferSize into
    // IApplicationProxyService command 3, which does not exist, and the applet
    // chain died before it ever opened.
    const APPLET: u64 = 0x1000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(APPLET, "appletOE");
    let tls = cpu.tls_base();

    ipc_request(&mut cpu, APPLET, 7, None, 3); // QueryPointerBufferSize
    assert_eq!(cpu.mem.read_u32(tls + 0x10).unwrap(), 0x4F43_4653); // "SFCO"
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0); // success, not an error
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
fn prfm_is_a_noop_not_ldrsw() {
    // `prfm pldl1keep, [x1]` = 0xF9800020 (size=11, V=0, opc=10). It is a
    // prefetch HINT and must not write a register. libtransistor's memcpy
    // starts with it; decoding it as `ldrsw x0, [x1]` clobbered the
    // destination register and made memcpy copy to `[source_magic_value]`,
    // leaving the real destination zeroed.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x1234_5678_9ABC_DEF0);
    cpu.set_reg(1, 0x3000);
    cpu.mem.map_zero(0x3000, 0x10).unwrap();
    cpu.mem.write_u32(0x3000, 0x7371_7368).unwrap();
    let prfm = 0b11u32 << 30 | 0b111 << 27 | 0b01 << 24 | 0b10 << 22 | (0 << 10) | (1 << 5) | 0;
    assert_eq!(prfm, 0xF980_0020);
    let code = [prfm, nop()];
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_x(0), 0x1234_5678_9ABC_DEF0);
}

#[test]
fn bfi_merges_into_destination_register() {
    // `bfi w0, w1, #8, #24` = BFM w0, w1, #8, #31 must insert w1's low 24
    // bits into w0 bits [31:8] and keep w0 bits [7:0]. The old decoder never
    // read the destination register and shifted the wrong field (verified
    // against qemu-aarch64: w0=0x5, w1=0xAB -> 0x0000AB05). libtransistor's
    // squashfs `swab_super` depends on this.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x5);
    cpu.set_reg(1, 0xAB);
    let code = [0x3318_5C20, nop()]; // bfi w0, w1, #8, #24 (clang-encoded)
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_x(0), 0x0000_AB05);
}

#[test]
fn bfxil_extracts_field_into_low_bits() {
    // `bfxil w0, w1, #16, #8` = BFM w0, w1, #16, #7 copies w1 bits [23:16]
    // into w0 bits [7:0], keeping w0's upper bits. qemu-aarch64: w1=0x00430000
    // -> w0 = (old_w0 & ~0xFF) | 0x43.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x1234_5678);
    cpu.set_reg(1, 0x0043_0000);
    let code = [0x3310_5C20, nop()]; // bfxil w0, w1, #16, #8 (clang-encoded)
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_x(0), 0x1234_5678 & !0xFF | 0x43);
}

#[test]
fn gamepad_input_writes_input_reg_and_hid_shmem() {
    // MapSharedMemory (svc 0x13) with x1=addr, x2=size must back the region
    // with real memory and, for a region hid's size, record it; set_gamepad_state
    // then mirrors the pad into INPUT_ADDR and into the two npad slots libnx
    // reads. The offsets are `HidSharedMemory`'s: npad at 0x9A00, 0x5000 per
    // controller, `full_key_lifo` at +0x28 and `handheld_lifo` at +0x378 within
    // `HidNpadInternalState`, each LIFO holding a 0x20-byte header then storage
    // entries of {sampling_number, HidNpadCommonState}.
    const SHMEM: u32 = 0x3000_0000;
    const NPAD: u32 = SHMEM + 0x9A00;
    const HANDHELD: u32 = NPAD + 8 * 0x5000;
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, SHMEM as u64);
    cpu.set_reg(2, 0x40000);
    cpu.mem.map(0x1000, &svc(0x13).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);

    // A|B held, left stick pushed fully up and slightly left.
    cpu.set_gamepad_state(0x3, -1000, 30000, 0, 0);

    // The mask handed to the guest gains HidNpadButton_StickLUp (1 << 17); the
    // small horizontal deflection stays below the pseudo-button threshold.
    let expected_buttons = 0x3 | (1 << 17);
    assert_eq!(
        cpu.mem.read_u64(switch_core::INPUT_ADDR).unwrap(),
        expected_buttons
    );

    for (base, lifo_off, style, device) in [
        (NPAD, 0x28, 1 << 0, 1 << 0),                    // player 1, Pro Controller
        (HANDHELD, 0x378, 1 << 1, (1 << 2) | (1 << 3)),  // handheld
    ] {
        assert_eq!(cpu.mem.read_u32(base).unwrap(), style, "style_set");
        assert_eq!(cpu.mem.read_u32(base + 0x4188).unwrap(), device, "device_type");
        let lifo = base + lifo_off;
        assert_eq!(cpu.mem.read_u64(lifo + 0x08).unwrap(), 17, "buffer_count");
        assert_eq!(cpu.mem.read_u64(lifo + 0x10).unwrap(), 0, "tail");
        assert_eq!(cpu.mem.read_u64(lifo + 0x18).unwrap(), 1, "count");
        let entry = lifo + 0x20;
        let sample = cpu.mem.read_u64(entry).unwrap();
        assert!(sample > 0, "sampling number must advance");
        assert_eq!(cpu.mem.read_u64(entry + 0x08).unwrap(), sample);
        assert_eq!(cpu.mem.read_u64(entry + 0x10).unwrap(), expected_buttons);
        assert_eq!(cpu.mem.read_u32(entry + 0x18).unwrap(), 1000u32.wrapping_neg());
        assert_eq!(cpu.mem.read_u32(entry + 0x1C).unwrap(), 30000);
        // IsConnected, whatever else the controller reports about its halves.
        assert_eq!(cpu.mem.read_u32(entry + 0x28).unwrap() & 1, 1);
    }
}

#[test]
fn mapping_pl_shared_memory_delivers_the_shared_font() {
    use switch_core::cpu::PL_SHMEM_SIZE;
    // `plInitialize` maps pl's shared memory and homebrew then reads the font
    // out of it at the offset pl reported (0), so the bytes have to be there by
    // the time the mapping syscall returns.
    const ADDR: u32 = 0x2000_0000;
    let font: Vec<u8> = (0..=255u8).cycle().take(0x2000).collect();
    let mut cpu = cpu_at(0x1000);
    cpu.set_shared_font(font.clone());
    cpu.set_reg(1, ADDR as u64);
    cpu.set_reg(2, u64::from(PL_SHMEM_SIZE));
    cpu.mem.map(0x1000, &svc(0x13).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);
    assert_eq!(cpu.mem.dump(ADDR, font.len()).unwrap(), font);

    // A font handed over after the guest mapped the region still reaches it:
    // the guest is holding a pointer into memory it already mapped.
    let replacement: Vec<u8> = vec![0xAB; 0x1000];
    cpu.set_shared_font(replacement.clone());
    assert_eq!(cpu.mem.dump(ADDR, replacement.len()).unwrap(), replacement);
}

#[test]
fn map_memory_backs_the_destination_and_unmap_frees_it() {
    // svcMapMemory(dst, src, size) aliases a range; libnx uses it to mirror a
    // thread's stack and then finds the *next* thread's mirror by looking for an
    // unmapped range, so the destination has to read back as mapped memory
    // holding the source's bytes. svcUnmapMemory hands them back and frees it.
    const SRC: u32 = 0x3000_0000;
    const DST: u32 = 0x1800_0000;
    let mut cpu = cpu_at(0x1000);
    cpu.mem.map(SRC, &0xDEAD_BEEFu32.to_le_bytes()).unwrap();
    cpu.mem.map(0x1000, &svc(0x04).to_le_bytes()).unwrap();
    cpu.mem.map(0x1004, &svc(0x05).to_le_bytes()).unwrap();

    cpu.set_reg(0, DST as u64);
    cpu.set_reg(1, SRC as u64);
    cpu.set_reg(2, 0x2000);
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);
    assert!(cpu.mem.page_mapped(DST));
    assert!(cpu.mem.page_mapped(DST + 0x1000));
    assert_eq!(cpu.mem.read_u32(DST).unwrap(), 0xDEAD_BEEF);

    cpu.mem.write_u32(DST, 0x1234_5678).unwrap();
    cpu.set_reg(0, DST as u64);
    cpu.set_reg(1, SRC as u64);
    cpu.set_reg(2, 0x2000);
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);
    assert_eq!(cpu.mem.read_u32(SRC).unwrap(), 0x1234_5678);
    assert!(!cpu.mem.page_mapped(DST));
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

#[test]
fn ccmp_subtract_carry_in() {
    // CCMP (op=1) computes Rn - imm, which needs the +carry_in the borrow
    // implies. With Rn=0, imm=0 the result is 0: Z and C must both be set.
    // Without carry_in, 0 + !0 + 0 = u64::MAX, corrupting N/Z/C. This exact
    // instruction was what sent NX-Shell's crt0 relocator into svcBreak.
    let code: [u32; 4] = [
        0xd2800102, // mov x2, #8
        0xd2800000, // mov x0, #0
        0xf100005f, // cmp x2, #0      (8-0=8 → Z clear, C set → NE holds)
        0xfa401800, // ccmp x0, #0, #0, ne   (0 - 0 = 0 → Z set, C set)
    ];
    let cpu = exec(&code, 100);
    assert_eq!(cpu.nzcv() & (1 << 30), 1 << 30, "Z must be set for 0-0");
    assert_eq!(cpu.nzcv() & (1 << 29), 1 << 29, "C must be set for 0-0 (no borrow)");
    assert_eq!(cpu.nzcv() & (1 << 31), 0, "N must be clear for 0-0");
}

#[test]
fn ccmp_ccmn_immediate_is_unsigned() {
    // The 5-bit immediate is unsigned for both CCMP and CCMN (QEMU-verified).
    // CCMP x0, #0x10 with x0=1 → 1-16 = -15 → N set.
    let c1: [u32; 3] = [
        0xd2800020, // mov x0, #1
        0xf100001f, // cmp x0, #0      (Z clear → NE holds)
        0xfa501800, // ccmp x0, #0x10, #0, ne
    ];
    let cpu = exec(&c1, 100);
    assert_eq!(cpu.nzcv() & (1 << 31), 1 << 31, "CCMP 1-16 = -15 → N set");
    assert_eq!(cpu.nzcv() & (1 << 30), 0, "CCMP 1-16 ≠ 0 → Z clear");

    // CCMN x0, #0x10 with x0=1 → 1+16 = 17 → all flags clear.
    let c2: [u32; 3] = [
        0xd2800020, // mov x0, #1
        0xf100001f, // cmp x0, #0
        0xba501800, // ccmn x0, #0x10, #0, ne
    ];
    let cpu = exec(&c2, 100);
    assert_eq!(cpu.nzcv() & 0xF000_0000, 0, "CCMN 1+16 = 17 → no flags set");
}

#[test]
fn simd_scalar_byte_load_and_stur_q() {
    // hbmenu / NX-Shell both faulted on `ldr b29, [x0, #0x280]` = 0x3d4a001d
    // (SIMD scalar 8-bit load). Also covers `stur q17, [x0, #0x8]` = 0x3c808011
    // (SIMD scalar STUR, unscaled offset) which the same libnx init loop uses.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x3000);
    cpu.mem.map_zero(0x3000, 0x300).unwrap();
    cpu.mem.write_u8(0x3280, 0xAB).unwrap();
    // ldr b29, [x0, #0x280]: size=00, V=1, mode=01, opc=01, imm12=0x280, rn=0, rt=29
    let ldr_b = 0b00u32 << 30 | 0b111 << 27 | (1 << 26) | (0b01 << 24) | (0b01 << 22) | (0x280 << 10) | 29;
    assert_eq!(ldr_b, 0x3d4a_001d);
    let code = [ldr_b, nop()];
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_vreg(29), 0xAB);

    // stur q17, [x0, #0x8] = 0x3c808011: size=00, V=1, mode=00 (unscaled),
    // opc=10 (STR Q), imm9=8, rn=0, rt=17.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x3000);
    cpu.mem.map_zero(0x3000, 0x40).unwrap();
    cpu.set_vreg(17, 0x1122_3344_5566_7788_99AA_BBCC_DDEE_FF00);
    let code = [0x3c80_8011u32, nop()];
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.mem.read_into(0x3008, &mut [0u8; 16]).and_then(|_| Ok(u128::from_le_bytes(cpu.mem.dump(0x3008, 16).unwrap().try_into().unwrap()))).unwrap(), 0x1122_3344_5566_7788_99AA_BBCC_DDEE_FF00);
}

#[test]
fn query_memory_writes_40_byte_memoryinfo() {
    // svc 0x06 (QueryMemory) writes a 40-byte MemoryInfo
    // {base(u64), size(u64), type/attr/perm/device/ipc/padding(u32 each)} —
    // NOT 8 x u64. The old stub wrote 64 bytes, overflowing the struct by 24
    // bytes; when the app's info pointer sat near the top of its stack this
    // clobbered main's saved LR and made NX-Shell's main "return" to 0.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x3000);   // info out pointer
    cpu.set_reg(1, 0x3040);   // page info out pointer
    cpu.set_reg(2, 0x1234000); // address
    cpu.mem.map_zero(0x1234000, 0x1000).unwrap();
    cpu.mem.map_zero(0x3000, 0x60).unwrap();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&svc(0x06).to_le_bytes());
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(1).unwrap();
    // Only the first 40 bytes are written; byte 40+ must stay untouched (0).
    assert_eq!(cpu.mem.read_u64(0x3000).unwrap(), 0x1234000);
    assert_eq!(cpu.mem.read_u64(0x3008).unwrap(), 0x1000);
    assert_eq!(cpu.mem.read_u32(0x3010).unwrap(), 3);   // type (mapped)
    assert_eq!(cpu.mem.read_u32(0x3014).unwrap(), 0);   // attr
    assert_eq!(cpu.mem.read_u32(0x3018).unwrap(), 0b111); // perm (RWX)
    assert_eq!(cpu.mem.read_u32(0x301c).unwrap(), 0);   // device_refcount
    assert_eq!(cpu.mem.read_u32(0x3020).unwrap(), 0);   // ipc_refcount
    assert_eq!(cpu.mem.read_u32(0x3024).unwrap(), 0);   // padding

    // An untouched soft-mapped page reports as unmapped (type 0, no perm),
    // which is what lets libnx virtmem find free address space.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x3000);
    cpu.set_reg(1, 0x3040);
    cpu.set_reg(2, 0x1234000);
    cpu.mem.map_zero(0x3000, 0x60).unwrap();
    cpu.mem.map(0x1000, &svc(0x06).to_le_bytes()).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(1).unwrap();
    // 0x1234000 is inside the soft-mapped range but never written -> unmapped.
    assert_eq!(cpu.mem.read_u32(0x3010).unwrap(), 0);   // type (unmapped)
    assert_eq!(cpu.mem.read_u32(0x3018).unwrap(), 0);   // perm
    // The old bug wrote 24 more bytes here; 0x3028+ must be untouched zeros.
    assert_eq!(cpu.mem.read_u64(0x3028).unwrap(), 0);
    assert_eq!(cpu.mem.read_u64(0x3040).unwrap(), 0); // pageinfo written via x1? no, x1 holds it
    assert_eq!(cpu.read_x(1), 0); // unmapped soft page -> page info 0
}

#[test]
fn scalar_cmge_with_zero_masks_predicate() {
    // NX-Shell faulted on `cmge d31, d31, #0` = 0x7ee08bff (scalar integer
    // compare-to-zero: Dd = all-ones if Dn >= 0, else 0), used as a predicate
    // mask via `fmov x2, d31` in a string-layout loop.
    // Encoding: bits[31:30] = 01 (D), U (bit29) = 1, bits[28:25] = 1111,
    // bits[24:21] = 0111, bits[20:16] = 00000 (zero operand), op = bits[15:10]
    // = 100010 (GE), Rn = bits[9:5], Rd = bits[4:0].
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(31, 0); // 0 >= 0 → all-ones
    let cpu = run_program(cpu, 0x1000, &[0x7ee0_8bff, nop()]);
    assert_eq!(cpu.read_vreg(31), u64::MAX as u128);

    // cmgt d4, d5, #0 = 0x5ee088a4 (U=0, op=100010): negative operand → 0.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(5, 0x8000_0000_0000_0000);
    let cpu = run_program(cpu, 0x1000, &[0x5ee0_88a4, nop()]);
    assert_eq!(cpu.read_vreg(4), 0);
}

#[test]
fn sub_shifted_register_reads_xzr_not_sp() {
    // `neg x1, x0` assembles as `sub x1, xzr, x0` (0xcb0003e1). In the
    // shifted-register form register 31 is XZR; only the immediate and
    // extended forms name SP. Reading SP here made newlib's `aligned_alloc`
    // compute a garbage rounded size, so every aligned allocation failed.
    let mut cpu = cpu_at(0x1000);
    cpu.set_pc_and_sp(0x1000, 0x1000_0000);
    cpu.set_reg(0, 0x1000);
    let cpu = run_program(cpu, 0x1000, &[0xcb00_03e1, nop()]);
    assert_eq!(cpu.read_x(1), (-0x1000i64) as u64);

    // The same instruction with a destination of 31 writes XZR, not SP.
    let mut cpu = cpu_at(0x1000);
    cpu.set_pc_and_sp(0x1000, 0x1000_0000);
    cpu.set_reg(0, 1);
    cpu.set_reg(1, 2);
    // sub sp-encoding: `sub x31, x1, x0` shifted-register = 0xcb00003f.
    let cpu = run_program(cpu, 0x1000, &[0xcb00_003f, nop()]);
    assert_eq!(cpu.sp(), 0x1000_0000, "shifted-register Rd=31 is XZR");
}

#[test]
fn add_immediate_still_uses_sp_for_register_31() {
    // `add sp, sp, #0x10` (0x910043ff) must keep naming SP: the immediate
    // form is the one where register 31 really is the stack pointer.
    let mut cpu = cpu_at(0x1000);
    cpu.set_pc_and_sp(0x1000, 0x1000_0000);
    let cpu = run_program(cpu, 0x1000, &[0x9100_43ff, nop()]);
    assert_eq!(cpu.sp(), 0x1000_0010);
}

#[test]
fn simd_post_index_store_writes_back_the_base() {
    // `str q27, [x2], #0x10` = 0x3c81045b. Without the write-back the base
    // never advances, so the vectorised table-fill loops it appears in never
    // terminate.
    let mut cpu = cpu_at(0x1000);
    cpu.mem.map_zero(0x4000, 0x100).unwrap();
    cpu.set_reg(2, 0x4000);
    cpu.set_vreg(27, 0x1122_3344_5566_7788_99AA_BBCC_DDEE_FF00);
    let cpu = run_program(cpu, 0x1000, &[0x3c81_045b, nop()]);
    assert_eq!(cpu.read_x(2), 0x4010, "post-index base must advance");
    assert_eq!(cpu.mem.read_u64(0x4000).unwrap(), 0x99AA_BBCC_DDEE_FF00);
    assert_eq!(cpu.mem.read_u64(0x4008).unwrap(), 0x1122_3344_5566_7788);
}

#[test]
fn simd_pre_index_load_uses_the_updated_base() {
    // `ldr q0, [x1, #0x10]!` = 0x3cc10c20: the base updates first, then the
    // access uses it.
    let mut cpu = cpu_at(0x1000);
    cpu.mem.map_zero(0x4000, 0x100).unwrap();
    cpu.mem.write_u64(0x4010, 0xDEAD_BEEF_CAFE_F00D).unwrap();
    cpu.set_reg(1, 0x4000);
    let cpu = run_program(cpu, 0x1000, &[0x3cc1_0c20, nop()]);
    assert_eq!(cpu.read_x(1), 0x4010);
    assert_eq!(cpu.read_vreg(0) as u64, 0xDEAD_BEEF_CAFE_F00D);
}

#[test]
fn simd_shift_right_immediate() {
    // `ushr v27.4s, v27.4s, #1` = 0x6f3f077b.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(27, 0x0000_0004_0000_0008_0000_0010_0000_0020);
    let cpu = run_program(cpu, 0x1000, &[0x6f3f_077b, nop()]);
    assert_eq!(cpu.read_vreg(27), 0x0000_0002_0000_0004_0000_0008_0000_0010);

    // `sshr v0.4s, v1.4s, #1` = 0x4f3f0420 keeps the sign.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(1, 0xFFFF_FFFE_0000_0004_0000_0000_0000_0000);
    let cpu = run_program(cpu, 0x1000, &[0x4f3f_0420, nop()]);
    assert_eq!(cpu.read_vreg(0) >> 96, 0xFFFF_FFFF);
}

#[test]
fn simd_shift_left_immediate() {
    // `shl v0.4s, v1.4s, #1` = 0x4f215420.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(1, 0x0000_0001_0000_0002_0000_0003_0000_0004);
    let cpu = run_program(cpu, 0x1000, &[0x4f21_5420, nop()]);
    assert_eq!(cpu.read_vreg(0), 0x0000_0002_0000_0004_0000_0006_0000_0008);
}

#[test]
fn simd_multiply_and_multiply_accumulate() {
    // `mul v26.4s, v26.4s, v28.4s` = 0x4ebc9f5a.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(26, 0x0000_0002_0000_0003_0000_0004_0000_0005);
    cpu.set_vreg(28, 0x0000_0002_0000_0002_0000_0002_0000_0002);
    let cpu = run_program(cpu, 0x1000, &[0x4ebc_9f5a, nop()]);
    assert_eq!(cpu.read_vreg(26), 0x0000_0004_0000_0006_0000_0008_0000_000A);

    // `mla v0.4s, v1.4s, v2.4s` = 0x4ea29420 accumulates into Vd.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, 0x0000_0001_0000_0001_0000_0001_0000_0001);
    cpu.set_vreg(1, 0x0000_0002_0000_0002_0000_0002_0000_0002);
    cpu.set_vreg(2, 0x0000_0003_0000_0003_0000_0003_0000_0003);
    let cpu = run_program(cpu, 0x1000, &[0x4ea2_9420, nop()]);
    assert_eq!(cpu.read_vreg(0), 0x0000_0007_0000_0007_0000_0007_0000_0007);
}

/// 64 bytes of 0x00..0x3F at `addr`, for the structure load/store tests.
fn map_ramp(cpu: &mut Cpu, addr: u32, len: u32) {
    cpu.mem.map_zero(addr, len as usize).unwrap();
    for i in 0..len {
        cpu.mem.write_u8(addr + i, i as u8).unwrap();
    }
}

/// The 16 bytes at `addr` as the u128 a `ld1 {Vt.16b}` would produce.
fn mem_u128(cpu: &Cpu, addr: u32) -> u128 {
    u128::from_le_bytes(cpu.mem.dump(addr, 16).unwrap().try_into().unwrap())
}

#[test]
fn ld1_multiple_structures_writes_back_only_when_post_indexed() {
    // `ld1 {v1.16b, v2.16b}, [x2], #32` = 0x4cdfa041. The immediate post-index
    // form has Rm == 31, which the old decode read as "no writeback" — newlib's
    // strrchr then computed its result from a base 32 bytes too low and
    // PHYSFS_init failed on a garbage argv[0] directory.
    let mut cpu = cpu_at(0x1000);
    map_ramp(&mut cpu, 0x3000, 64);
    cpu.set_reg(2, 0x3000);
    let cpu = run_program(cpu, 0x1000, &[0x4cdf_a041, nop()]);
    assert_eq!(cpu.read_vreg(1), mem_u128(&cpu, 0x3000));
    assert_eq!(cpu.read_vreg(2), mem_u128(&cpu, 0x3010));
    assert_eq!(cpu.read_reg(2), 0x3020);

    // `ld1 {v3.16b}, [x0]` = 0x4c407003 has no writeback at all: Rm reads as 0
    // there, and the old decode wrote the incremented base into x0.
    let mut cpu = cpu_at(0x1000);
    map_ramp(&mut cpu, 0x3000, 64);
    cpu.set_reg(0, 0x3000);
    let cpu = run_program(cpu, 0x1000, &[0x4c40_7003, nop()]);
    assert_eq!(cpu.read_vreg(3), mem_u128(&cpu, 0x3000));
    assert_eq!(cpu.read_reg(0), 0x3000);

    // `ld1 {v4.8b}, [x0]` = 0x0c407004 moves 8 bytes, not 16, and zeroes the
    // register's top half.
    let mut cpu = cpu_at(0x1000);
    map_ramp(&mut cpu, 0x3000, 64);
    cpu.set_reg(0, 0x3000);
    cpu.set_vreg(4, u128::MAX);
    let cpu = run_program(cpu, 0x1000, &[0x0c40_7004, nop()]);
    assert_eq!(cpu.read_vreg(4), 0x0706_0504_0302_0100);

    // `ld1 {v5.16b, v6.16b, v7.16b, v8.16b}, [x1], #64` = 0x4cdf2025.
    let mut cpu = cpu_at(0x1000);
    map_ramp(&mut cpu, 0x3000, 64);
    cpu.set_reg(1, 0x3000);
    let cpu = run_program(cpu, 0x1000, &[0x4cdf_2025, nop()]);
    for (i, reg) in (5u8..=8).enumerate() {
        assert_eq!(cpu.read_vreg(reg), mem_u128(&cpu, 0x3000 + 16 * i as u32));
    }
    assert_eq!(cpu.read_reg(1), 0x3040);

    // Register post-index: `ld1 {v0.2d, v1.2d}, [x0], x5` = 0x4cc5ac00 advances
    // the base by Xm, not by the transfer size.
    let mut cpu = cpu_at(0x1000);
    map_ramp(&mut cpu, 0x3000, 64);
    cpu.set_reg(0, 0x3000);
    cpu.set_reg(5, 8);
    let cpu = run_program(cpu, 0x1000, &[0x4cc5_ac00, nop()]);
    assert_eq!(cpu.read_reg(0), 0x3008);
}

#[test]
fn ld1r_replicates_one_element_to_every_lane() {
    // `ld1r {v9.16b}, [x0]` = 0x4d40c009 and `ld1r {v10.4s}, [x0]` = 0x4d40c80a.
    // The replicate group is `scale == 0b11`, which the old decode treated as a
    // doubleword lane insert.
    let mut cpu = cpu_at(0x1000);
    cpu.mem.map_zero(0x3000, 0x40).unwrap();
    cpu.mem.write_u32(0x3000, 0x1122_33AB).unwrap();
    cpu.set_reg(0, 0x3000);
    let cpu = run_program(cpu, 0x1000, &[0x4d40_c009, 0x4d40_c80a, nop()]);
    assert_eq!(cpu.read_vreg(9), u128::from_le_bytes([0xAB; 16]));
    assert_eq!(cpu.read_vreg(10), 0x1122_33AB_1122_33AB_1122_33AB_1122_33AB);
}

#[test]
fn ld2_and_st2_interleave_lanes() {
    // `ld2 {v11.16b, v12.16b}, [x0]` = 0x4c40800b splits the block into even
    // and odd bytes; `st2 {v11.16b, v12.16b}, [x3]` = 0x4c00806b puts it back.
    let mut cpu = cpu_at(0x1000);
    map_ramp(&mut cpu, 0x3000, 32);
    cpu.mem.map_zero(0x3100, 0x40).unwrap();
    cpu.set_reg(0, 0x3000);
    cpu.set_reg(3, 0x3100);
    let cpu = run_program(cpu, 0x1000, &[0x4c40_800b, 0x4c00_806b, nop()]);
    let evens: Vec<u8> = (0..32u8).filter(|b| b % 2 == 0).collect();
    let odds: Vec<u8> = (0..32u8).filter(|b| b % 2 == 1).collect();
    assert_eq!(cpu.read_vreg(11), u128::from_le_bytes(evens.try_into().unwrap()));
    assert_eq!(cpu.read_vreg(12), u128::from_le_bytes(odds.try_into().unwrap()));
    assert_eq!(cpu.mem.dump(0x3100, 32).unwrap(), (0..32u8).collect::<Vec<u8>>());

    // `ld4 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64` = 0x4cdf0810 takes
    // every fourth word into each register.
    let mut cpu = cpu_at(0x1000);
    cpu.mem.map_zero(0x3000, 0x40).unwrap();
    for i in 0..16u32 {
        cpu.mem.write_u32(0x3000 + i * 4, i).unwrap();
    }
    cpu.set_reg(0, 0x3000);
    let cpu = run_program(cpu, 0x1000, &[0x4cdf_0810, nop()]);
    assert_eq!(cpu.read_vreg(16), 0x0000_000C_0000_0008_0000_0004_0000_0000);
    assert_eq!(cpu.read_vreg(17), 0x0000_000D_0000_0009_0000_0005_0000_0001);
    assert_eq!(cpu.read_vreg(18), 0x0000_000E_0000_000A_0000_0006_0000_0002);
    assert_eq!(cpu.read_vreg(19), 0x0000_000F_0000_000B_0000_0007_0000_0003);
    assert_eq!(cpu.read_reg(0), 0x3040);
}

#[test]
fn ld1_single_lane_addresses_the_whole_element() {
    // `ld1 {v13.s}[1], [x0]` = 0x0d40900d replaces bits 32..63 and nothing
    // else; the old decode shifted by the element's byte count and masked to
    // that many bits, so it rewrote 4 bits at bit 4.
    let mut cpu = cpu_at(0x1000);
    cpu.mem.map_zero(0x3000, 0x40).unwrap();
    cpu.mem.map_zero(0x3100, 0x40).unwrap();
    cpu.mem.write_u32(0x3000, 0xDEAD_BEEF).unwrap();
    cpu.set_reg(0, 0x3000);
    cpu.set_reg(3, 0x3100);
    cpu.set_vreg(13, 0x1111_1111_2222_2222_3333_3333_4444_4444);
    let cpu = run_program(cpu, 0x1000, &[0x0d40_900d, 0x0d00_906d, nop()]);
    assert_eq!(cpu.read_vreg(13), 0x1111_1111_2222_2222_DEAD_BEEF_4444_4444);
    assert_eq!(cpu.mem.read_u32(0x3100).unwrap(), 0xDEAD_BEEF);
}

#[test]
fn bsl_bit_and_bif_take_their_mask_from_the_right_register() {
    // BSL selects with Vd, BIT and BIF with Vm. All three had the mask wrong,
    // which broke newlib's vectorised strchr (it uses `bif` to fold the
    // "matched" and "end of string" predicates together).
    let mut cpu = cpu_at(0x1000);
    let byte = |b: u8| u128::from_le_bytes([b; 16]);
    cpu.set_vreg(20, byte(0xF0));
    cpu.set_vreg(21, byte(0xAA));
    cpu.set_vreg(22, byte(0x55));
    cpu.set_vreg(23, byte(0x55));
    cpu.set_vreg(24, byte(0xAA));
    cpu.set_vreg(25, byte(0xF0));
    cpu.set_vreg(26, byte(0x55));
    cpu.set_vreg(27, byte(0xAA));
    cpu.set_vreg(28, byte(0xF0));
    // bsl v20, v21, v22 / bit v23, v24, v25 / bif v26, v27, v28
    let cpu = run_program(cpu, 0x1000, &[0x6e76_1eb4, 0x6eb9_1f17, 0x6efc_1f7a, nop()]);
    assert_eq!(cpu.read_vreg(20), byte(0xA5));
    assert_eq!(cpu.read_vreg(23), byte(0xA5));
    assert_eq!(cpu.read_vreg(26), byte(0x5A));
}

#[test]
fn register_offset_load_sign_extends_a_32_bit_index() {
    // `ldr x4, [x1, w2, sxtw #3]` = 0xf862d824 with w2 = -2 reads 16 bytes
    // below the base.
    let mut cpu = cpu_at(0x1000);
    cpu.mem.map_zero(0x3000, 0x200).unwrap();
    cpu.mem.write_u64(0x30F0, 0xAAAA_AAAA_AAAA_AAAA).unwrap();
    cpu.set_reg(1, 0x3100);
    cpu.set_reg(2, 0xFFFF_FFFE);
    let cpu = run_program(cpu, 0x1000, &[0xf862_d824, nop()]);
    assert_eq!(cpu.read_reg(4), 0xAAAA_AAAA_AAAA_AAAA);

    // `ldr x5, [x1, x2, sxtx #3]` = 0xf862f825 takes the index as a full 64-bit
    // value.
    let mut cpu = cpu_at(0x1000);
    cpu.mem.map_zero(0x3000, 0x200).unwrap();
    cpu.mem.write_u64(0x30F0, 0xBBBB_BBBB_BBBB_BBBB).unwrap();
    cpu.set_reg(1, 0x3100);
    cpu.set_reg(2, 0xFFFF_FFFF_FFFF_FFFE);
    let cpu = run_program(cpu, 0x1000, &[0xf862_f825, nop()]);
    assert_eq!(cpu.read_reg(5), 0xBBBB_BBBB_BBBB_BBBB);
}

#[test]
fn scalar_fp_one_source_and_fused_multiply_add() {
    // `fmov s0, s15` = 0x1e2041e0 is opcode 0 of the 1-source group, whose low
    // opcode bit sits in bits[15] — matching bits[15:10] as a unit missed the
    // whole group, so NX-Shell faulted here. FMOV is a bit-exact copy, so a
    // signalling NaN payload survives it.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(15, 0x1111_1111_1111_1111_1111_1111_7FA0_1234);
    let cpu = run_program(cpu, 0x1000, &[0x1e20_41e0, nop()]);
    assert_eq!(cpu.read_vreg(0), 0x7FA0_1234);

    // `fmov d1, d2` = 0x1e604041 keeps 64 bits and zeroes the rest.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(2, 0x9999_9999_9999_9999_4008_0000_0000_0000);
    let cpu = run_program(cpu, 0x1000, &[0x1e60_4041, nop()]);
    assert_eq!(cpu.read_vreg(1), 0x4008_0000_0000_0000);

    // FMADD/FMSUB/FNMADD/FNMSUB (`fmadd d3, d4, d5, d6` = 0x1f451883 and
    // friends): the 3-source group has its own top byte, so it was unreachable.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(4, 3.0f64.to_bits() as u128);
    cpu.set_vreg(5, 4.0f64.to_bits() as u128);
    cpu.set_vreg(6, 5.0f64.to_bits() as u128);
    let cpu = run_program(
        cpu,
        0x1000,
        &[0x1f45_1883, 0x1f45_9887, 0x1f65_1888, 0x1f65_9889, nop()],
    );
    assert_eq!(f64::from_bits(cpu.read_vreg(3) as u64), 17.0);
    assert_eq!(f64::from_bits(cpu.read_vreg(7) as u64), -7.0);
    assert_eq!(f64::from_bits(cpu.read_vreg(8) as u64), -17.0);
    assert_eq!(f64::from_bits(cpu.read_vreg(9) as u64), 7.0);

    // FABS/FNEG/FSQRT/FRINTM/FRINTP and the two FCVT directions.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(12, (-1.5f32).to_bits() as u128);
    cpu.set_vreg(14, (2.5f64).to_bits() as u128);
    cpu.set_vreg(15, (9.0f32).to_bits() as u128);
    cpu.set_vreg(16, (-1.5f64).to_bits() as u128);
    cpu.set_vreg(17, (1.25f32).to_bits() as u128);
    cpu.set_vreg(18, (0.5f64).to_bits() as u128);
    cpu.set_vreg(19, (0.25f32).to_bits() as u128);
    let cpu = run_program(
        cpu,
        0x1000,
        &[
            0x1e20_c18b, // fabs s11, s12
            0x1e61_41cd, // fneg d13, d14
            0x1e21_c1ee, // fsqrt s14, s15
            0x1e65_420f, // frintm d15, d16
            0x1e24_c230, // frintp s16, s17
            0x1e62_4251, // fcvt s17, d18
            0x1e22_c272, // fcvt d18, s19
            nop(),
        ],
    );
    assert_eq!(f32::from_bits(cpu.read_vreg(11) as u32), 1.5);
    assert_eq!(f64::from_bits(cpu.read_vreg(13) as u64), -2.5);
    assert_eq!(f32::from_bits(cpu.read_vreg(14) as u32), 3.0);
    assert_eq!(f64::from_bits(cpu.read_vreg(15) as u64), -2.0);
    assert_eq!(f32::from_bits(cpu.read_vreg(16) as u32), 2.0);
    assert_eq!(f32::from_bits(cpu.read_vreg(17) as u32), 0.5);
    assert_eq!(f64::from_bits(cpu.read_vreg(18) as u64), 0.25);
}

#[test]
fn ctr_el0_reports_64_byte_cache_lines() {
    // `mrs x7, ctr_el0` = 0xd53b0027. Cache-flush loops stride by
    // `4 << DminLine`; reporting 0 walked NX-Shell's buffers 4 bytes at a time.
    let cpu = run_program(cpu_at(0x1000), 0x1000, &[0xd53b_0027, nop()]);
    assert_eq!(cpu.read_reg(7), 0x8444_C004);
    assert_eq!((cpu.read_reg(7) >> 16) & 0xF, 4);
}

/// Pack four 32-bit lanes, lane 0 in the low bits.
fn u32x4(lanes: [u32; 4]) -> u128 {
    lanes.iter().rev().fold(0u128, |acc, &l| (acc << 32) | u128::from(l))
}

fn f32x4(lanes: [f32; 4]) -> u128 {
    u32x4([lanes[0].to_bits(), lanes[1].to_bits(), lanes[2].to_bits(), lanes[3].to_bits()])
}

fn u64x2(lanes: [u64; 2]) -> u128 {
    u128::from(lanes[0]) | (u128::from(lanes[1]) << 64)
}

fn f64x2(lanes: [f64; 2]) -> u128 {
    u64x2([lanes[0].to_bits(), lanes[1].to_bits()])
}

fn lanes_f32(v: u128) -> [f32; 4] {
    [0, 1, 2, 3].map(|i| f32::from_bits((v >> (32 * i)) as u32))
}

fn lanes_u32(v: u128) -> [u32; 4] {
    [0, 1, 2, 3].map(|i| (v >> (32 * i)) as u32)
}

#[test]
fn vector_integer_float_conversions() {
    // `scvtf v28.4s, v31.4s` = 0x4e21dbfc is where NX-Shell faulted: the
    // two-register misc group was falling through to the three-same decode.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(31, u32x4([1, 2, 3, 0xFFFF_FFFF]));
    let cpu = run_program(cpu, 0x1000, &[0x4e21_dbfc, nop()]);
    assert_eq!(lanes_f32(cpu.read_vreg(28)), [1.0, 2.0, 3.0, -1.0]);

    // UCVTF reads the same lanes as unsigned.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(3, u32x4([1, 2, 3, 0xFFFF_FFFF]));
    let cpu = run_program(cpu, 0x1000, &[0x6e21_d862, nop()]);
    assert_eq!(lanes_f32(cpu.read_vreg(2)), [1.0, 2.0, 3.0, 4_294_967_295.0]);

    // FCVTZS truncates toward zero and saturates at the lane width.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(5, f32x4([1.9, -1.9, 1.0e10, -1.0e10]));
    let cpu = run_program(cpu, 0x1000, &[0x4ea1_b8a4, nop()]);
    assert_eq!(
        lanes_u32(cpu.read_vreg(4)),
        [1, (-1i32) as u32, i32::MAX as u32, i32::MIN as u32]
    );

    // FCVTZU clamps negatives to zero.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(7, f32x4([2.7, -2.7, 5.0, 0.0]));
    let cpu = run_program(cpu, 0x1000, &[0x6ea1_b8e6, nop()]);
    assert_eq!(lanes_u32(cpu.read_vreg(6)), [2, 0, 5, 0]);

    // The rounding modes: FCVTNS ties to even, FCVTPS up, FCVTMS down.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(13, f32x4([0.5, 1.5, 2.5, -1.5]));
    cpu.set_vreg(11, f32x4([1.1, -1.1, 0.0, 3.9]));
    cpu.set_vreg(9, f64x2([-1.5, 2.5]));
    let cpu = run_program(cpu, 0x1000, &[0x4e21_a9ac, 0x4ea1_a96a, 0x4e61_b928, nop()]);
    assert_eq!(lanes_u32(cpu.read_vreg(12)), [0, 2, 2, (-2i32) as u32]);
    assert_eq!(lanes_u32(cpu.read_vreg(10)), [2, (-1i32) as u32, 0, 4]);
    assert_eq!(cpu.read_vreg(8), u64x2([(-2i64) as u64, 2]));
}

#[test]
fn vector_floating_point_arithmetic() {
    // `fdiv v28.4s, v28.4s, v30.4s` = 0x6e3eff9c — the FP three-same group
    // (opcodes from 0b11000 up) was being decoded as integer ops.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(28, f32x4([1.0, 3.0, 5.0, 9.0]));
    cpu.set_vreg(30, f32x4([2.0, 4.0, 5.0, 3.0]));
    let cpu = run_program(cpu, 0x1000, &[0x6e3e_ff9c, nop()]);
    assert_eq!(lanes_f32(cpu.read_vreg(28)), [0.5, 0.75, 1.0, 3.0]);

    // FADD / FSUB (2D) / FMUL.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(1, f32x4([1.0, 2.0, 3.0, 4.0]));
    cpu.set_vreg(2, f32x4([0.5, 0.5, 0.5, 0.5]));
    cpu.set_vreg(4, f32x4([2.0, 3.0, 4.0, 5.0]));
    cpu.set_vreg(5, f32x4([2.0, 2.0, 2.0, 2.0]));
    let cpu = run_program(cpu, 0x1000, &[0x4e22_d420, 0x6e25_dc83, nop()]);
    assert_eq!(lanes_f32(cpu.read_vreg(0)), [1.5, 2.5, 3.5, 4.5]);
    assert_eq!(lanes_f32(cpu.read_vreg(3)), [4.0, 6.0, 8.0, 10.0]);

    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(1, f64x2([1.0, 2.0]));
    cpu.set_vreg(2, f64x2([0.25, 0.5]));
    let cpu = run_program(cpu, 0x1000, &[0x4ee2_d420, nop()]);
    assert_eq!(cpu.read_vreg(0), f64x2([0.75, 1.5]));

    // FMLA and FMLS accumulate into Vd.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(6, f32x4([1.0, 1.0, 1.0, 1.0]));
    cpu.set_vreg(7, f32x4([2.0, 2.0, 2.0, 2.0]));
    cpu.set_vreg(8, f32x4([3.0, 3.0, 3.0, 3.0]));
    cpu.set_vreg(9, f32x4([1.0, 1.0, 1.0, 1.0]));
    cpu.set_vreg(10, f32x4([2.0, 2.0, 2.0, 2.0]));
    cpu.set_vreg(11, f32x4([3.0, 3.0, 3.0, 3.0]));
    let cpu = run_program(cpu, 0x1000, &[0x4e28_cce6, 0x4eab_cd49, nop()]);
    assert_eq!(lanes_f32(cpu.read_vreg(6)), [7.0; 4]);
    assert_eq!(lanes_f32(cpu.read_vreg(9)), [-5.0; 4]);

    // FMAX, FMINNM (2D), the compares and FADDP.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(13, f32x4([1.0, 5.0, -1.0, 0.0]));
    cpu.set_vreg(14, f32x4([2.0, 4.0, -2.0, 0.0]));
    cpu.set_vreg(16, f64x2([1.0, 8.0]));
    cpu.set_vreg(17, f64x2([2.0, 4.0]));
    cpu.set_vreg(19, f32x4([1.0, 2.0, 3.0, 4.0]));
    cpu.set_vreg(20, f32x4([1.0, 0.0, 3.0, 0.0]));
    cpu.set_vreg(22, f32x4([1.0, 2.0, 3.0, 4.0]));
    cpu.set_vreg(23, f32x4([2.0, 2.0, 1.0, 5.0]));
    cpu.set_vreg(25, f32x4([-3.0, 1.0, -1.0, 2.0]));
    cpu.set_vreg(26, f32x4([2.0, -2.0, 1.0, 2.0]));
    cpu.set_vreg(28, f32x4([1.0, 2.0, 3.0, 4.0]));
    cpu.set_vreg(29, f32x4([10.0, 20.0, 30.0, 40.0]));
    let code = [
        0x4e2e_f5ac, // fmax v12.4s, v13.4s, v14.4s
        0x4ef1_c60f, // fminnm v15.2d, v16.2d, v17.2d
        0x4e34_e672, // fcmeq v18.4s, v19.4s, v20.4s
        0x6e37_e6d5, // fcmge v21.4s, v22.4s, v23.4s
        0x6eba_ef38, // facgt v24.4s, v25.4s, v26.4s
        0x6e3d_d79b, // faddp v27.4s, v28.4s, v29.4s
        nop(),
    ];
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(lanes_f32(cpu.read_vreg(12)), [2.0, 5.0, -1.0, 0.0]);
    assert_eq!(cpu.read_vreg(15), f64x2([1.0, 4.0]));
    assert_eq!(lanes_u32(cpu.read_vreg(18)), [u32::MAX, 0, u32::MAX, 0]);
    assert_eq!(lanes_u32(cpu.read_vreg(21)), [0, u32::MAX, u32::MAX, 0]);
    assert_eq!(lanes_u32(cpu.read_vreg(24)), [u32::MAX, 0, 0, 0]);
    assert_eq!(lanes_f32(cpu.read_vreg(27)), [3.0, 7.0, 30.0, 70.0]);

    // FABS / FNEG / FSQRT / FRINTM / FRINTP and the compares against zero.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(15, f32x4([-1.5, 2.5, -0.0, 3.0]));
    cpu.set_vreg(17, f64x2([1.5, -2.5]));
    cpu.set_vreg(19, f32x4([4.0, 9.0, 16.0, 25.0]));
    cpu.set_vreg(21, f32x4([1.5, -1.5, 2.0, -2.5]));
    cpu.set_vreg(23, f32x4([1.5, -1.5, 2.0, -2.5]));
    cpu.set_vreg(25, f32x4([1.0, -1.0, 0.0, 2.0]));
    cpu.set_vreg(27, f32x4([1.0, -1.0, 0.0, 2.0]));
    let code = [
        0x4ea0_f9ee, // fabs v14.4s, v15.4s
        0x6ee0_fa30, // fneg v16.2d, v17.2d
        0x6ea1_fa72, // fsqrt v18.4s, v19.4s
        0x4e21_9ab4, // frintm v20.4s, v21.4s
        0x4ea1_8af6, // frintp v22.4s, v23.4s
        0x4ea0_cb38, // fcmgt v24.4s, v25.4s, #0.0
        0x6ea0_db7a, // fcmle v26.4s, v27.4s, #0.0
        nop(),
    ];
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(lanes_f32(cpu.read_vreg(14)), [1.5, 2.5, 0.0, 3.0]);
    assert_eq!(cpu.read_vreg(16), f64x2([-1.5, 2.5]));
    assert_eq!(lanes_f32(cpu.read_vreg(18)), [2.0, 3.0, 4.0, 5.0]);
    assert_eq!(lanes_f32(cpu.read_vreg(20)), [1.0, -2.0, 2.0, -3.0]);
    assert_eq!(lanes_f32(cpu.read_vreg(22)), [2.0, -1.0, 2.0, -2.0]);
    assert_eq!(lanes_u32(cpu.read_vreg(24)), [u32::MAX, 0, 0, u32::MAX]);
    assert_eq!(lanes_u32(cpu.read_vreg(26)), [0, u32::MAX, u32::MAX, 0]);
}

#[test]
fn vector_two_register_misc_integer_ops() {
    // CNT / CLZ / NOT / RBIT.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(1, u32x4([0x0000_00FF, 0x0101_0101, 0, 0x8000_0001]));
    cpu.set_vreg(3, u32x4([1, 0x8000_0000, 0, 0x0000_FFFF]));
    cpu.set_vreg(5, u32x4([0, u32::MAX, 0x1234_5678, 0]));
    cpu.set_vreg(7, u32x4([0x0000_0001, 0, 0, 0]));
    let code = [
        0x4e20_5820, // cnt v0.16b, v1.16b
        0x6ea0_4862, // clz v2.4s, v3.4s
        0x6e20_58a4, // not v4.16b, v5.16b
        0x6e60_58e6, // rbit v6.16b, v7.16b
        nop(),
    ];
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(lanes_u32(cpu.read_vreg(0)), [8, 0x0101_0101, 0, 0x0100_0001]);
    assert_eq!(lanes_u32(cpu.read_vreg(2)), [31, 0, 32, 16]);
    assert_eq!(lanes_u32(cpu.read_vreg(4)), [u32::MAX, 0, 0xEDCB_A987, u32::MAX]);
    assert_eq!(lanes_u32(cpu.read_vreg(6)), [0x0000_0080, 0, 0, 0]);

    // ABS / NEG / CMGT #0 / CMLT #0 / CLS.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(9, u32x4([1, (-2i32) as u32, 0, i32::MIN as u32]));
    cpu.set_vreg(11, u32x4([1, (-2i32) as u32, 0, 5]));
    cpu.set_vreg(1, u32x4([1, (-1i32) as u32, 0, 7]));
    cpu.set_vreg(3, u32x4([1, (-1i32) as u32, 0, 7]));
    cpu.set_vreg(7, u32x4([0x0000_0001, 0xFFFF_FFFF, 0x4000_0000, 0]));
    let code = [
        0x4ea0_b928, // abs v8.4s, v9.4s
        0x6ea0_b96a, // neg v10.4s, v11.4s
        0x4ea0_8820, // cmgt v0.4s, v1.4s, #0
        0x4ea0_a862, // cmlt v2.4s, v3.4s, #0
        0x4ea0_48e6, // cls v6.4s, v7.4s
        nop(),
    ];
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(lanes_u32(cpu.read_vreg(8)), [1, 2, 0, i32::MIN as u32]);
    assert_eq!(lanes_u32(cpu.read_vreg(10)), [(-1i32) as u32, 2, 0, (-5i32) as u32]);
    assert_eq!(lanes_u32(cpu.read_vreg(0)), [u32::MAX, 0, 0, u32::MAX]);
    assert_eq!(lanes_u32(cpu.read_vreg(2)), [0, u32::MAX, 0, 0]);
    assert_eq!(lanes_u32(cpu.read_vreg(6)), [30, 31, 0, 31]);

    // REV64 / REV32 / REV16 reverse bytes within their container.
    let mut cpu = cpu_at(0x1000);
    let ramp = u128::from_le_bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    cpu.set_vreg(13, ramp);
    cpu.set_vreg(15, ramp);
    cpu.set_vreg(17, ramp);
    let code = [
        0x4e20_09ac, // rev64 v12.16b, v13.16b
        0x6e60_09ee, // rev32 v14.8h, v15.8h
        0x4e20_1a30, // rev16 v16.16b, v17.16b
        nop(),
    ];
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(
        cpu.read_vreg(12).to_le_bytes(),
        [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8]
    );
    assert_eq!(
        cpu.read_vreg(14).to_le_bytes(),
        [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13]
    );
    assert_eq!(
        cpu.read_vreg(16).to_le_bytes(),
        [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]
    );

    // XTN / SQXTN / UQXTN narrow, SHLL widens, UADDLP folds lane pairs.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(19, u64x2([0x0004_0003_0002_0001, 0]));
    cpu.set_vreg(21, u64x2([0x7FFF_8000_0002_FFFF, 0]));
    cpu.set_vreg(23, u32x4([0x0000_00FF, 0x0001_0000, 0, 0]));
    cpu.set_vreg(25, u64x2([0x0004_0003_0002_0001, 0]));
    cpu.set_vreg(5, u64x2([0x0004_0003_0002_0001, 0x0008_0007_0006_0005]));
    let code = [
        0x0e21_2a72, // xtn v18.8b, v19.8h
        0x0e21_4ab4, // sqxtn v20.8b, v21.8h
        0x2e61_4af6, // uqxtn v22.4h, v23.4s
        0x2e61_3b38, // shll v24.4s, v25.4h, #16
        0x6e60_28a4, // uaddlp v4.4s, v5.8h
        nop(),
    ];
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_vreg(18), 0x0403_0201);
    // Signed saturation: 0x7FFF -> 0x7F, 0x8000 -> 0x80, 0x0002 stays,
    // 0xFFFF is -1.
    assert_eq!(cpu.read_vreg(20), 0x7F80_02FF);
    // Unsigned saturation: 0x0001_0000 -> 0xFFFF, 0xFF stays.
    assert_eq!(cpu.read_vreg(22), 0xFFFF_00FF);
    assert_eq!(cpu.read_vreg(24), u64x2([0x0002_0000_0001_0000, 0x0004_0000_0003_0000]));
    assert_eq!(lanes_u32(cpu.read_vreg(4)), [3, 7, 11, 15]);

    // FCVTL widens 2S to 2D and FCVTN narrows back.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(27, f32x4([1.5, -2.5, 0.0, 0.0]));
    cpu.set_vreg(29, f64x2([3.5, -4.5]));
    let cpu = run_program(cpu, 0x1000, &[0x0e61_7b7a, 0x0e61_6bbc, nop()]);
    assert_eq!(cpu.read_vreg(26), f64x2([1.5, -2.5]));
    assert_eq!(lanes_f32(cpu.read_vreg(28)), [3.5, -4.5, 0.0, 0.0]);
}

#[test]
fn scalar_integer_float_conversions_and_rounding_modes() {
    // `ucvtf d0, x1` = 0x9e630020 is where NX-Shell died: rmode/opcode were
    // read as one 6-bit field including the fixed bit 21, so this decoded as
    // FCVTMU and wrote x0 — clobbering a live pointer.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0xDEAD_BEEF);
    cpu.set_reg(1, 5);
    let cpu = run_program(cpu, 0x1000, &[0x9e63_0020, nop()]);
    assert_eq!(f64::from_bits(cpu.read_vreg(0) as u64), 5.0);
    assert_eq!(cpu.read_reg(0), 0xDEAD_BEEF, "x0 must be untouched");

    // UCVTF reads the source as unsigned, SCVTF as signed, and `sf` gives the
    // source width independently of the destination's.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, u64::MAX);
    cpu.set_reg(3, -3i64 as u64);
    cpu.set_reg(5, 0xFFFF_FFFF);
    let code = [
        0x9e63_0020, // ucvtf d0, x1
        0x9e62_0062, // scvtf d2, x3
        0x1e23_00a4, // ucvtf s4, w5
        nop(),
    ];
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(f64::from_bits(cpu.read_vreg(0) as u64), 18_446_744_073_709_551_615.0);
    assert_eq!(f64::from_bits(cpu.read_vreg(2) as u64), -3.0);
    assert_eq!(f32::from_bits(cpu.read_vreg(4) as u32), 4_294_967_295.0);

    // The float → integer forms: rmode picks the rounding and the result
    // saturates at the destination width.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, (2.5f32).to_bits() as u128);
    let code = [
        0x1e20_0008, // fcvtns w8, s0  (ties to even → 2)
        0x1e28_0009, // fcvtps w9, s0  (toward +inf → 3)
        0x1e30_000a, // fcvtms w10, s0 (toward -inf → 2)
        0x1e24_000b, // fcvtas w11, s0 (ties away → 3)
        nop(),
    ];
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_reg(8), 2);
    assert_eq!(cpu.read_reg(9), 3);
    assert_eq!(cpu.read_reg(10), 2);
    assert_eq!(cpu.read_reg(11), 3);

    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, (-2.7f64).to_bits() as u128);
    let code = [
        0x9e79_0006, // fcvtzu x6, d0 → negative clamps to 0
        0x1e78_0007, // fcvtzs w7, d0 → -2 (truncated)
        nop(),
    ];
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_reg(6), 0);
    assert_eq!(cpu.read_reg(7) as u32, -2i32 as u32);

    // A 32-bit destination saturates rather than wrapping.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, (1.0e18f64).to_bits() as u128);
    let cpu = run_program(cpu, 0x1000, &[0x1e78_0007, nop()]);
    assert_eq!(cpu.read_reg(7) as u32, i32::MAX as u32);
}

#[test]
fn fcmp_against_zero_uses_the_opcode2_bit() {
    // `fcmp d8, #0.0` = 0x1e602108. The compare-with-zero flag is bit 3 of
    // opcode2 (bits[4:0]); reading it from bits[9:8] took it out of Rn, so this
    // compared d8 against v0 instead of zero.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(8, (1.5f64).to_bits() as u128);
    cpu.set_vreg(0, (1.5f64).to_bits() as u128);
    let cpu = run_program(cpu, 0x1000, &[0x1e60_2108, nop()]);
    // 1.5 > 0 → N=0 Z=0 C=1 V=0.
    assert_eq!(cpu.nzcv() >> 28, 0b0010);

    // `fcmp d0, #0.0` = 0x1e602008 with a non-zero d0 must not read d0 as the
    // second operand (which would always compare equal).
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, (-4.0f64).to_bits() as u128);
    let cpu = run_program(cpu, 0x1000, &[0x1e60_2008, nop()]);
    // -4 < 0 → N=1 Z=0 C=0 V=0.
    assert_eq!(cpu.nzcv() >> 28, 0b1000);

    // The register form still compares Vn with Vm.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(1, (2.0f64).to_bits() as u128);
    cpu.set_vreg(2, (2.0f64).to_bits() as u128);
    let cpu = run_program(cpu, 0x1000, &[0x1e62_2020, nop()]);
    // equal → N=0 Z=1 C=1 V=0.
    assert_eq!(cpu.nzcv() >> 28, 0b0110);
}

#[test]
fn ext_extracts_across_a_vector_pair() {
    // `ext v0.16b, v1.16b, v2.16b, #4` = 0x6e022020 takes the top 12 bytes of
    // Vn followed by the low 4 of Vm. NX-Shell faulted on the `#8` form.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(1, 0x1F1E_1D1C_1B1A_1918_1716_1514_1312_1110);
    cpu.set_vreg(2, 0x2F2E_2D2C_2B2A_2928_2726_2524_2322_2120);
    let cpu = run_program(cpu, 0x1000, &[0x6e02_2020, nop()]);
    assert_eq!(cpu.read_vreg(0), 0x2322_2120_1F1E_1D1C_1B1A_1918_1716_1514);

    // #8 on a single register rotates it by 8 bytes.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(31, 0x1122_3344_5566_7788_99AA_BBCC_DDEE_FF00);
    let cpu = run_program(cpu, 0x1000, &[0x6e1f_43ff, nop()]);
    assert_eq!(cpu.read_vreg(31), 0x99AA_BBCC_DDEE_FF00_1122_3344_5566_7788);

    // #0 is a plain move of Vn.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(7, 0xAAAA);
    cpu.set_vreg(8, 0xBBBB);
    let cpu = run_program(cpu, 0x1000, &[0x6e08_00e6, nop()]);
    assert_eq!(cpu.read_vreg(6), 0xAAAA);

    // The 64-bit form works on the low halves only and zeroes the top.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(4, 0xFFFF_FFFF_FFFF_FFFF_0807_0605_0403_0201);
    cpu.set_vreg(5, 0xFFFF_FFFF_FFFF_FFFF_1817_1615_1413_1211);
    let cpu = run_program(cpu, 0x1000, &[0x2e05_1883, nop()]);
    assert_eq!(cpu.read_vreg(3), 0x1312_1108_0706_0504);
}

#[test]
fn scalar_shift_by_immediate() {
    // `ushr d30, d31, #32` = 0x7f6007fe. The scalar forms differ from the
    // vector ones only in bit 28 and always operate on one 64-bit lane; only
    // the vector encodings were decoded, so NX-Shell faulted here.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(31, 0xFFFF_FFFF_FFFF_FFFF_1122_3344_5566_7788);
    let cpu = run_program(cpu, 0x1000, &[0x7f60_07fe, nop()]);
    assert_eq!(cpu.read_vreg(30), 0x1122_3344);

    // `shl d0, d1, #4` = 0x5f445420 and `sshr d2, d3, #63` = 0x5f410462.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(1, 0x0000_0000_0000_0123);
    cpu.set_vreg(3, 0x8000_0000_0000_0000);
    let cpu = run_program(cpu, 0x1000, &[0x5f44_5420, 0x5f41_0462, nop()]);
    assert_eq!(cpu.read_vreg(0), 0x1230);
    assert_eq!(cpu.read_vreg(2), 0xFFFF_FFFF_FFFF_FFFF);
}

#[test]
fn fcsel_fccmp_and_fixed_point_conversions() {
    // `fcsel s30, s31, s30, gt` = 0x1e3ecffe. FCSEL and FCCMP have bit 21 set;
    // they were guarded on bit 21 being clear, so neither was reachable.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(31, (1.5f32).to_bits() as u128);
    cpu.set_vreg(30, (2.5f32).to_bits() as u128);
    // Set flags with `fcmp s31, s30` (1.5 < 2.5 → not GT) then select.
    cpu.set_reg(0, 0);
    let cpu = run_program(cpu, 0x1000, &[0x1e3e_23e0, 0x1e3e_cffe, nop()]);
    assert_eq!(f32::from_bits(cpu.read_vreg(30) as u32), 2.5, "GT false → Vm");

    // With the condition true it takes Vn.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(1, (7.0f64).to_bits() as u128);
    cpu.set_vreg(2, (9.0f64).to_bits() as u128);
    cpu.set_vreg(3, (1.0f64).to_bits() as u128);
    // `fcmp d3, d3` sets Z (equal), so EQ holds → `fcsel d0, d1, d2, eq` = d1.
    let cpu = run_program(cpu, 0x1000, &[0x1e63_2060, 0x1e62_0c20, nop()]);
    assert_eq!(f64::from_bits(cpu.read_vreg(0) as u64), 7.0);

    // FCCMP with a failing condition installs its NZCV immediate instead of
    // comparing: `fccmp d1, d2, #5, ne` = 0x1e621425 after `fcmp d3, d3` (Z set,
    // so NE fails) leaves NZCV = 5 (Z and V).
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(1, (1.0f64).to_bits() as u128);
    cpu.set_vreg(2, (2.0f64).to_bits() as u128);
    cpu.set_vreg(3, (1.0f64).to_bits() as u128);
    let cpu = run_program(cpu, 0x1000, &[0x1e63_2060, 0x1e62_1425, nop()]);
    assert_eq!(cpu.nzcv() >> 28, 0b0101);

    // The fixed-point conversions (bit 21 clear) used to land in the branch
    // those conditionals occupied: `scvtf s0, w1, #8` = 0x1e02e020 scales by
    // 2^-8, and `fcvtzs w2, s0, #4` = 0x1e18f002 scales back up by 2^4.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, 256);
    cpu.set_reg(4, 3);
    let code = [
        0x1e02_e020, // scvtf s0, w1, #8   → 256 / 256 = 1.0
        0x1e18_f002, // fcvtzs w2, s0, #4  → 1.0 * 16 = 16
        0x9e43_c083, // ucvtf d3, x4, #16  → 3 / 65536
        nop(),
    ];
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(f32::from_bits(cpu.read_vreg(0) as u32), 1.0);
    assert_eq!(cpu.read_reg(2), 16);
    assert_eq!(f64::from_bits(cpu.read_vreg(3) as u64), 3.0 / 65536.0);
}

#[test]
fn scalar_two_register_misc_converts_one_lane() {
    // `ucvtf s13, s13` = 0x7e21d9ad — the scalar form of the two-register misc
    // group (bits[31:30] = 01, bits[28:24] = 11110). Only the vector encodings
    // were decoded, so NX-Shell faulted here.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(13, 0xFFFF_FFFF_FFFF_FFFF_FFFF_FFFF_0000_0007);
    let cpu = run_program(cpu, 0x1000, &[0x7e21_d9ad, nop()]);
    // One lane converted, everything above it zeroed.
    assert_eq!(cpu.read_vreg(13), u128::from((7.0f32).to_bits()));
}

#[test]
fn blr_reads_its_target_before_linking() {
    // `blr x30` is a return-and-relink: BLR reads the target register first,
    // then writes x30. Linking first made it branch to itself+4 — hbmenu's NEON
    // JPEG decoder ends its IDCT that way, so its icon never finished decoding.
    let mut cpu = cpu_at(0x1000);
    cpu.mem.map_zero(0x2000, 0x100).unwrap();
    cpu.set_reg(30, 0x2000);
    // blr x30 = 0xd63f03c0
    let cpu = run_program(cpu, 0x1000, &[0xd63f_03c0u32]);
    assert_eq!(cpu.get_pc(), 0x2000, "branched to the old x30");
    assert_eq!(cpu.read_reg(30), 0x1004, "and linked to the next instruction");
}

#[test]
fn permutes_follow_the_zip_uzp_trn_definitions() {
    // Values cross-checked against qemu-aarch64. Vn = 0x1000,0x2000,..,0x8000
    // and Vm = 1,2,4,..,0x80 as halfwords.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(2, u64x2([0x4000_3000_2000_1000, 0x8000_7000_6000_5000]));
    cpu.set_vreg(3, u64x2([0x0008_0004_0002_0001, 0x0080_0040_0020_0010]));
    // trn1 v28.8h, v2.8h, v3.8h / trn2 v16.8h / zip1 v10.8h / zip2 v11.8h /
    // uzp1 v12.8h / uzp2 v13.8h
    let cpu = run_program(
        cpu,
        0x1000,
        &[0x4e43_285c, 0x4e43_6850, 0x4e43_384a, 0x4e43_784b, 0x4e43_184c, 0x4e43_584d, nop()],
    );
    // TRN1 takes the even elements of both, interleaved.
    assert_eq!(cpu.read_vreg(28), u64x2([0x0004_3000_0001_1000, 0x0040_7000_0010_5000]));
    // TRN2 the odd ones.
    assert_eq!(cpu.read_vreg(16), u64x2([0x0008_4000_0002_2000, 0x0080_8000_0020_6000]));
    // ZIP1 interleaves the low halves, ZIP2 the high halves.
    assert_eq!(cpu.read_vreg(10), u64x2([0x0002_2000_0001_1000, 0x0008_4000_0004_3000]));
    assert_eq!(cpu.read_vreg(11), u64x2([0x0020_6000_0010_5000, 0x0080_8000_0040_7000]));
    // UZP1 packs Vn's even elements then Vm's; UZP2 the odd ones.
    assert_eq!(cpu.read_vreg(12), u64x2([0x7000_5000_3000_1000, 0x0040_0010_0004_0001]));
    assert_eq!(cpu.read_vreg(13), u64x2([0x8000_6000_4000_2000, 0x0080_0020_0008_0002]));
}

#[test]
fn widening_and_by_element_multiplies() {
    // `smull v18.4s, v18.4h, v0.h[2]` = 0x4f60a252-ish: every lane times one
    // selected lane, widened. Cross-checked against qemu-aarch64.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(2, u64x2([0x0004_0003_0002_0001, 0x0008_0007_0006_0005]));
    cpu.set_vreg(0, u64x2([0x0000_0003_0000_0000, 0]));
    // smull v4.4s, v2.4h, v0.h[2] (v0.h[2] = 3)
    let cpu = run_program(cpu, 0x1000, &[0x0f60_a044, nop()]);
    assert_eq!(lanes_u32(cpu.read_vreg(4)), [3, 6, 9, 12]);

    // The `2` form reads the high half of the source.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(2, u64x2([0x0004_0003_0002_0001, 0x0008_0007_0006_0005]));
    cpu.set_vreg(0, u64x2([0x0000_0003_0000_0000, 0]));
    let cpu = run_program(cpu, 0x1000, &[0x4f60_a044, nop()]);
    assert_eq!(lanes_u32(cpu.read_vreg(4)), [15, 18, 21, 24]);

    // The vector (three-different) forms: smull, smlal, saddl, addhn.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(2, u64x2([0x0004_0003_0002_0001, 0]));
    cpu.set_vreg(3, u64x2([0xFFFF_0002_0003_0004, 0]));
    cpu.set_vreg(5, u64x2([0x0000_000A_0000_000A, 0x0000_000A_0000_000A]));
    // smull v5.4s, v2.4h, v3.4h
    let cpu = run_program(cpu, 0x1000, &[0x0e63_c045, nop()]);
    // 1*4, 2*3, 3*2, 4*-1
    assert_eq!(
        lanes_u32(cpu.read_vreg(5)),
        [4, 6, 6, (-4i32) as u32]
    );
}

#[test]
fn guest_threads_run_and_hand_over_at_blocking_syscalls() {
    // A guest program that creates a thread, starts it, and waits for it to set
    // a flag. Thread creation used to hand out a fake handle and never run
    // anything, so the wait spun forever — which is where hbmenu stopped.
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.mem.map_zero(0x4000, 0x2000).unwrap(); // the child's stack
    cpu.mem.map_zero(0x6000, 0x1000).unwrap(); // the flag and the arg it saw

    // main: svcCreateThread(entry = 0x2000, arg = 0x1234, stack_top = 0x5000),
    // svcStartThread, then sleep until the flag is set and exit.
    let main = [
        0xd284_0001u32, // mov x1, #0x2000  (entry)
        0xd282_4682,    // mov x2, #0x1234  (arg)
        0xd28a_0003,    // mov x3, #0x5000  (stack top)
        0x5280_0764,    // mov w4, #0x3b    (priority)
        0x1280_0005,    // mov w5, #-1      (core)
        0xd400_0101,    // svc #8           (CreateThread → handle in x1)
        0xaa01_03e0,    // mov x0, x1
        0xd400_0121,    // svc #9           (StartThread)
        0xd400_0161,    // svc #0xb         (SleepThread → yields)
        0xd28c_0009,    // mov x9, #0x6000
        0xb940_0122,    // ldr w2, [x9]
        0x34ff_ffa2,    // cbz w2, -12      (back to the sleep)
        0xd400_00e1,    // svc #7           (ExitProcess)
    ];
    // child: record the argument it was passed, set the flag, exit.
    let child = [
        0xd28c_0009u32, // mov x9, #0x6000
        0xb900_0520,    // str w0, [x9, #4]
        0x5280_0aa1,    // mov w1, #0x55
        0xb900_0121,    // str w1, [x9]
        0xd400_0141,    // svc #0xa         (ExitThread)
    ];
    let bytes = |code: &[u32]| -> Vec<u8> {
        code.iter().flat_map(|i| i.to_le_bytes()).collect()
    };
    cpu.mem.map_zero(0x1000, 0x100).unwrap();
    cpu.mem.map(0x1000, &bytes(&main)).unwrap();
    cpu.mem.map_zero(0x2000, 0x100).unwrap();
    cpu.mem.map(0x2000, &bytes(&child)).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(10_000).unwrap();

    assert!(cpu.halted, "main should reach ExitProcess once the child ran");
    assert_eq!(cpu.mem.read_u32(0x6000).unwrap(), 0x55, "the child set the flag");
    assert_eq!(cpu.mem.read_u32(0x6004).unwrap(), 0x1234, "with its argument in x0");
    assert_eq!(cpu.thread_count(), 2);
}

#[test]
fn a_thread_polling_an_idle_socket_does_not_starve_the_others() {
    // NXpotify's Zeroconf listener is `if (poll(&pfd, 1, 200) <= 0) continue;`
    // around an idle socket. Nothing here will ever be ready, so the answer is
    // always zero — but a poll that was given a timeout is a *wait*, and
    // returning it instantly left that thread looping with no blocking syscall
    // in it. Threads only hand over at those, so the loop owned the CPU
    // forever and the main thread never drew another frame.
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.register_service_handle(0x30, "bsd:u");
    cpu.mem.map_zero(0x4000, 0x2000).unwrap(); // the child's stack
    cpu.mem.map_zero(0x6000, 0x1000).unwrap(); // the flag main sets

    // main: start the poller, sleep once to hand it the CPU, and then — only
    // if it hands the CPU back — set the flag and exit.
    let main = [
        0xd284_0001u32, // mov x1, #0x2000  (entry)
        0xd280_0002,    // mov x2, #0       (arg)
        0xd28a_0003,    // mov x3, #0x5000  (stack top)
        0x5280_0764,    // mov w4, #0x3b    (priority)
        0x1280_0005,    // mov w5, #-1      (core)
        0xd400_0101,    // svc #8           (CreateThread → handle in x1)
        0xaa01_03e0,    // mov x0, x1
        0xd400_0121,    // svc #9           (StartThread)
        0xd400_0161,    // svc #0xb         (SleepThread → over to the poller)
        0xd28c_0009,    // mov x9, #0x6000
        0x5280_0aa1,    // mov w1, #0x55
        0xb900_0121,    // str w1, [x9]
        0xd400_00e1,    // svc #7           (ExitProcess)
    ];
    // The poller: build a `Poll(nfds = 1, timeout = 200)` CMIF request in its
    // own TLS block and send it, forever. Threads get their TLS at
    // THREAD_TLS_BASE + index * stride, and this is thread 1.
    let child = [
        0xd282_0009u32, // mov x9, #0x1000
        0xf2a3_fc29,    // movk x9, #0x1fe1, lsl #16   (= THREAD_TLS_BASE + stride)
        0x5280_0081,    // mov w1, #4                  (message type: Request)
        0xb900_0121,    // str w1, [x9]
        0x5280_0101,    // mov w1, #8                  (data words)
        0xb900_0521,    // str w1, [x9, #4]
        0x5288_ca61,    // mov w1, #0x4653             ("SFCI")
        0x72a9_2861,    // movk w1, #0x4943, lsl #16
        0xb900_1121,    // str w1, [x9, #0x10]
        0x5280_00c1,    // mov w1, #6                  (command: Poll)
        0xb900_1921,    // str w1, [x9, #0x18]
        0x5280_0021,    // mov w1, #1                  (nfds)
        0xb900_2121,    // str w1, [x9, #0x20]
        0x5280_1901,    // mov w1, #200                (timeout, ms)
        0xb900_2521,    // str w1, [x9, #0x24]
        0xd280_0600,    // mov x0, #0x30               (the bsd:u handle)
        0xd400_0421,    // svc #0x21                   (SendSyncRequest)
        0x17ff_ffef,    // b -0x44                     (round again)
    ];
    let bytes = |code: &[u32]| -> Vec<u8> { code.iter().flat_map(|i| i.to_le_bytes()).collect() };
    cpu.mem.map_zero(0x1000, 0x100).unwrap();
    cpu.mem.map(0x1000, &bytes(&main)).unwrap();
    cpu.mem.map_zero(0x2000, 0x100).unwrap();
    cpu.mem.map(0x2000, &bytes(&child)).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(10_000).unwrap();

    assert!(cpu.halted, "main never got the CPU back from the polling thread");
    assert_eq!(cpu.mem.read_u32(0x6000).unwrap(), 0x55);
}

#[test]
fn set_thread_activity_takes_a_thread_out_of_the_rotation() {
    // `svcSetThreadActivity` is `nn::os::SuspendThread`/`ResumeThread`. A
    // suspended thread keeps whatever it was doing and simply stops being
    // scheduled; Horizon refuses to suspend the caller, and reports a thread
    // already in the requested state rather than treating the call as a no-op.
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.mem.map_zero(0x4000, 0x2000).unwrap(); // the child's stack
    cpu.mem.map_zero(0x6000, 0x1000).unwrap(); // the flag it sets

    // main: create and start a child, suspend it, yield a few times, then
    // record whether it ever ran, resume it, yield again, and exit.
    let main = [
        0xd284_0001u32, // mov x1, #0x2000  (entry)
        0xd280_0002,    // mov x2, #0        (arg)
        0xd28a_0003,    // mov x3, #0x5000   (stack top)
        0xd400_0101,    // svc #8            (CreateThread -> handle in x1)
        0xaa01_03ea,    // mov x10, x1       (keep the handle)
        0xaa01_03e0,    // mov x0, x1
        0xd400_0121,    // svc #9            (StartThread)
        0xaa0a_03e0,    // mov x0, x10
        0xd280_0021,    // mov x1, #1        (Paused)
        0xd400_0641,    // svc #0x32         (SetThreadActivity)
        0xd400_0161,    // svc #0xb          (SleepThread -> yields)
        0xd400_0161,    // svc #0xb
        0xd28c_0009,    // mov x9, #0x6000
        0xb940_0122,    // ldr w2, [x9]
        0xb900_0522,    // str w2, [x9, #4]  (what it saw while suspended)
        0xaa0a_03e0,    // mov x0, x10
        0xd280_0001,    // mov x1, #0        (Runnable)
        0xd400_0641,    // svc #0x32         (SetThreadActivity)
        0xd400_0161,    // svc #0xb
        0xd400_00e1,    // svc #7            (ExitProcess)
    ];
    let child = [
        0xd28c_0009u32, // mov x9, #0x6000
        0x5280_0aa1,    // mov w1, #0x55
        0xb900_0121,    // str w1, [x9]
        0xd400_0141,    // svc #0xa          (ExitThread)
    ];
    let bytes = |code: &[u32]| -> Vec<u8> { code.iter().flat_map(|i| i.to_le_bytes()).collect() };
    cpu.mem.map_zero(0x1000, 0x100).unwrap();
    cpu.mem.map(0x1000, &bytes(&main)).unwrap();
    cpu.mem.map_zero(0x2000, 0x100).unwrap();
    cpu.mem.map(0x2000, &bytes(&child)).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(10_000).unwrap();

    assert!(cpu.halted, "main should reach ExitProcess");
    assert_eq!(cpu.mem.read_u32(0x6004).unwrap(), 0, "a suspended thread must not run");
    assert_eq!(cpu.mem.read_u32(0x6000).unwrap(), 0x55, "and must run once resumed");
}

#[test]
fn arbitrate_lock_hands_the_mutex_to_a_waiter() {
    // Horizon keeps the lock word in guest memory: it holds the owner's handle,
    // plus bit30 when someone is queued. `svcArbitrateUnlock` has to move
    // ownership, or libnx's mutexLock re-reads the word and spins forever.
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.mem.map_zero(0x4000, 0x2000).unwrap();
    cpu.mem.map_zero(0x6000, 0x1000).unwrap();
    const MUTEX: u32 = 0x6100;

    // main: create + start a thread, then hold the mutex and unlock it. The
    // child blocks in ArbitrateLock and must come out owning the word.
    let main = [
        0xd284_0001u32, // mov x1, #0x2000
        0xd280_0002,    // mov x2, #0
        0xd28a_0003,    // mov x3, #0x5000
        0x5280_0764,    // mov w4, #0x3b
        0x1280_0005,    // mov w5, #-1
        0xd400_0101,    // svc #8
        0xaa01_03e0,    // mov x0, x1
        0xd400_0121,    // svc #9
        0xd400_0161,    // svc #0xb   (yield: the child blocks on the mutex)
        0xd28c_2009,    // mov x9, #0x6100
        0xaa0903e0u32,  // mov x0, x9  (ArbitrateUnlock takes the address in x0)
        0xd400_0361,    // svc #0x1b  (ArbitrateUnlock → hand it to the child)
        0xd400_0161,    // svc #0xb   (yield so the child can finish)
        0xd400_00e1,    // svc #7
    ];
    // child: ask the kernel to arbitrate a mutex main owns, then record the
    // word it ended up with.
    let child = [
        0xd28c_2009u32, // mov x9, #0x6100
        0xb940_0120,    // ldr w0, [x9]     (current owner)
        0xaa09_03e1,    // mov x1, x9       (the mutex address)
        0xd280_0022,    // mov x2, #1       (our handle, unused by the stub)
        0xd400_0341,    // svc #0x1a        (ArbitrateLock → blocks)
        0xb940_0122,    // ldr w2, [x9]
        0xd28c_0009,    // mov x9, #0x6000
        0xb900_0122,    // str w2, [x9]
        0xd400_0141,    // svc #0xa
    ];
    let bytes = |code: &[u32]| -> Vec<u8> {
        code.iter().flat_map(|i| i.to_le_bytes()).collect()
    };
    cpu.mem.map_zero(0x1000, 0x100).unwrap();
    cpu.mem.map(0x1000, &bytes(&main)).unwrap();
    cpu.mem.map_zero(0x2000, 0x100).unwrap();
    cpu.mem.map(0x2000, &bytes(&child)).unwrap();
    // main "owns" the mutex to begin with (handle 1 = the main thread).
    cpu.mem.write_u32(MUTEX, 1).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(10_000).unwrap();

    assert!(cpu.halted);
    // The child saw itself as the owner after the unlock handed it over.
    let observed = cpu.mem.read_u32(0x6000).unwrap();
    assert_ne!(observed, 0, "the child ran after the unlock");
    assert_ne!(observed, 1, "and the word no longer names the main thread");
}

#[test]
fn thirty_two_bit_writes_clear_the_upper_half() {
    // Every write to a W register zeroes bits 63:32. SBFM's sign extension was
    // filling them instead, so `asr w0, w0, #31` produced
    // 0xFFFF_FFFF_FFFF_FFFF and any later 64-bit use of that register saw a
    // huge value.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0xFFFF_FF00); // negative as a word, clear upper half
    cpu.set_reg(5, 0x0000_0F00);
    cpu.set_reg(6, 0x0000_8001);
    let code = [
        0x131f_7c00u32, // asr w0, w0, #31
        0x1304_2ca5,    // sbfx w5, w5, #4, #8
        0x1300_3cc6,    // sxth w6, w6
        nop(),
    ];
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_reg(0), 0x0000_0000_FFFF_FFFF);
    assert_eq!(cpu.read_reg(5), 0x0000_0000_FFFF_FFF0);
    assert_eq!(cpu.read_reg(6), 0x0000_0000_FFFF_8001);

    // The 64-bit forms still fill the whole register.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(4, 0xFFFF_FFFF_FFFF_FF00);
    let cpu = run_program(cpu, 0x1000, &[0x937f_fc84, nop()]); // asr x4, x4, #63
    assert_eq!(cpu.read_reg(4), u64::MAX);
}

#[test]
fn asr_by_register_sign_extends_from_the_operand_width() {
    // `asr w2, w2, w3` on a negative word must give all-ones. The operand was
    // masked to 32 bits and then shifted as a positive i64, yielding 1 — which
    // is how libjpeg-turbo's HUFF_EXTEND lost the sign of every DC difference
    // and hbmenu's icon decoded with the wrong luma and chroma.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(2, 0xFFFF_FF00);
    cpu.set_reg(3, 31);
    let cpu = run_program(cpu, 0x1000, &[0x1ac3_2842, nop()]);
    assert_eq!(cpu.read_reg(2), 0x0000_0000_FFFF_FFFF);

    // A positive word still shifts to zero, and the 64-bit form is unaffected.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(2, 0x7FFF_FF00);
    cpu.set_reg(3, 31);
    let cpu = run_program(cpu, 0x1000, &[0x1ac3_2842, nop()]);
    assert_eq!(cpu.read_reg(2), 0);
}

#[test]
fn scalar_integer_forms_verified_against_qemu() {
    // Each of these was wrong until `tools/difftest.py --scalar` compared it
    // with qemu-aarch64; the values here are qemu's.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(10, 0xFFFF_FF00);
    cpu.set_reg(11, 0x1F);
    cpu.set_reg(12, 0x8000_0000_0000_0001);

    // EXTR takes the low bits of Rn:Rm >> imm, so Rn is the *high* half.
    let cpu = run_program(cpu, 0x1000, &[0x138b_1d41, nop()]); // extr w1, w10, w11, #7
    assert_eq!(cpu.read_reg(1), 0);

    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(10, 0xFFFF_FF00);
    cpu.set_reg(11, 0x1F);
    let cpu = run_program(cpu, 0x1000, &[0x93cb_8542, nop()]); // extr x2, x10, x11, #33
    assert_eq!(cpu.read_reg(2), 0x7FFF_FF80_0000_0000);

    // ADCS adds with carry: bit30 selects subtract and bit29 sets the flags, and
    // having them swapped made `adcs` subtract.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(10, 0xFFFF_FF00);
    cpu.set_reg(11, 0x1F);
    // `cmp w11, w11` sets C, then `adcs w5, w10, w11`.
    let cpu = run_program(cpu, 0x1000, &[0x6b0b_017f, 0x3a0b_0145, nop()]);
    assert_eq!(cpu.read_reg(5), 0xFFFF_FF20);

    // SDIV sign-extends its operands from their own width.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(10, 0xFFFF_FF00); // -256 as a word
    cpu.set_reg(11, 0x1F);
    let cpu = run_program(cpu, 0x1000, &[0x1acb_0d49, nop()]); // sdiv w9, w10, w11
    assert_eq!(cpu.read_reg(9), 0xFFFF_FFF8); // -256 / 31 = -8

    // SMADDL multiplies the low 32 bits, sign-extended.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(10, 0xFFFF_FF00);
    cpu.set_reg(11, 0x1F);
    cpu.set_reg(12, 0x8000_0000_0000_0001);
    let cpu = run_program(cpu, 0x1000, &[0x9b2b_3145, nop()]); // smaddl x5, w10, w11, x12
    assert_eq!(cpu.read_reg(5), 0x7FFF_FFFF_FFFF_E101);

    // CLS counts the sign bits *after* the sign bit.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(10, 0xFFFF_FF00);
    let cpu = run_program(cpu, 0x1000, &[0x5ac0_1549, nop()]); // cls w9, w10
    assert_eq!(cpu.read_reg(9), 23);
}

/// Marshal a non-domain request (an `SFCI` header straight after the hipc
/// header) with `payload` as its arguments, and send it. `nnSdk` keeps
/// `audout` as a plain session rather than converting it to a domain, so this
/// is the shape those commands actually arrive in.
fn ipc_request_plain(cpu: &mut Cpu, handle: u64, cmd: u32, payload: &[u8]) {
    build_ipc_request(cpu, 4, None, cmd);
    let tls = cpu.tls_base();
    for (i, &b) in payload.iter().enumerate() {
        cpu.mem.write_u8(tls + 0x20 + i as u32, b).unwrap();
    }
    run_ipc_request(cpu, handle);
}

/// The same, carrying one map-alias buffer in the direction `recv` asks for.
fn ipc_request_plain_with_buffer(
    cpu: &mut Cpu,
    handle: u64,
    cmd: u32,
    buf: u32,
    len: u32,
    recv: bool,
    payload: &[u8],
) {
    let tls = cpu.tls_base();
    for i in (0..0x100u32).step_by(4) {
        cpu.mem.write_u32(tls + i, 0).unwrap();
    }
    // Send buffers count in bits 23:20, receive buffers in 27:24.
    cpu.mem.write_u32(tls, 4 | (1 << if recv { 24 } else { 20 })).unwrap();
    cpu.mem.write_u32(tls + 4, 0x0c).unwrap();
    cpu.mem.write_u32(tls + 0x08, len).unwrap();
    cpu.mem.write_u32(tls + 0x0c, buf).unwrap();
    cpu.mem.write_u32(tls + 0x10, 0).unwrap();
    // One descriptor pushes the aligned data area out to 0x20.
    cpu.mem.write_u32(tls + 0x20, 0x4943_4653).unwrap(); // "SFCI"
    cpu.mem.write_u32(tls + 0x28, cmd).unwrap();
    for (i, &b) in payload.iter().enumerate() {
        cpu.mem.write_u8(tls + 0x30 + i as u32, b).unwrap();
    }
    run_ipc_request(cpu, handle);
}

#[test]
fn audout_plays_the_buffers_the_guest_hands_it() {
    // `audout` is the plain PCM-out device, and the whole interface is the
    // buffer protocol: append a buffer, wait on the event, collect the tags of
    // the buffers the device is done with. A device that accepts buffers and
    // never releases them hangs the guest's audio thread forever.
    const AUDOUT: u64 = 0xA000;
    const DESC: u32 = 0x8000; // the AudioOutBuffer struct
    const PCM: u32 = 0x8100; // its samples
    const TAGS: u32 = 0x8200; // where released tags come back
    const TAG: u64 = 0xFEED_0001;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(AUDOUT, "audout:u");
    let tls = cpu.tls_base();

    // OpenAudioOut(48 kHz, stereo) -> { rate, channels, format, state } and an
    // IAudioOut as a *move* handle.
    let mut args = Vec::new();
    args.extend_from_slice(&48_000u32.to_le_bytes());
    args.extend_from_slice(&2u32.to_le_bytes());
    args.extend_from_slice(&0u64.to_le_bytes()); // aruid
    ipc_request_plain(&mut cpu, AUDOUT, 1, &args);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0, "OpenAudioOut failed");
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 48_000);
    assert_eq!(cpu.mem.read_u32(tls + 0x24).unwrap(), 2);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 2, "PcmFormat::Int16");
    assert_eq!(cpu.mem.read_u32(tls + 0x2c).unwrap(), 1, "a device opens stopped");
    // { send_pid:1, num_copy:4, num_move:4 }: one move handle, no copy ones.
    assert_eq!(cpu.mem.read_u32(tls + 0x08).unwrap(), 1 << 5);
    let device = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    assert_ne!(device, 0, "no IAudioOut came back");

    // RegisterBufferEvent: an event, and events are *copy* handles.
    ipc_request_plain(&mut cpu, device, 4, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x08).unwrap(), 1 << 1);
    let event = cpu.mem.read_u32(tls + 0x0c).unwrap();
    assert_ne!(event, 0);
    // Nothing has been played, so nothing has been released.
    assert_eq!(wait_sync(&mut cpu, &[event], 0).0, 0xEA01, "event fired early");

    // StartAudioOut, then hand over one buffer of four stereo frames.
    ipc_request_plain(&mut cpu, device, 1, &[]);
    ipc_request_plain(&mut cpu, device, 0, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 0, "started");

    let samples: [i16; 8] = [1, -1, 2, -2, 3, -3, 4, -4];
    for (i, &s) in samples.iter().enumerate() {
        cpu.mem.write_u16(PCM + i as u32 * 2, s as u16).unwrap();
    }
    // AudioOutBuffer { next, buffer, buffer_size, data_size, data_offset }.
    cpu.mem.write_u64(DESC, 0).unwrap();
    cpu.mem.write_u64(DESC + 8, u64::from(PCM)).unwrap();
    cpu.mem.write_u64(DESC + 16, 16).unwrap();
    cpu.mem.write_u64(DESC + 24, 16).unwrap();
    cpu.mem.write_u64(DESC + 32, 0).unwrap();
    ipc_request_plain_with_buffer(&mut cpu, device, 3, DESC, 40, false, &TAG.to_le_bytes());
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0, "AppendAudioOutBuffer failed");

    // The samples reached the host, unchanged, at full volume.
    let mut played = [0i16; 8];
    assert_eq!(cpu.take_audio(&mut played), 8);
    assert_eq!(played, samples);

    // The buffer came back: the event fired and its tag is collectable.
    assert_eq!(wait_sync(&mut cpu, &[event], 0).0, 0, "the released buffer did not fire");
    ipc_request_plain_with_buffer(&mut cpu, device, 5, TAGS, 16, true, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 1, "no tag released");
    assert_eq!(cpu.mem.read_u64(TAGS).unwrap(), TAG);

    // GetAudioOutPlayedSampleCount counts frames, not samples: four stereo
    // frames, not eight.
    ipc_request_plain(&mut cpu, device, 10, &[]);
    assert_eq!(cpu.mem.read_u64(tls + 0x20).unwrap(), 4);

    // And the host is told what to play it at.
    assert_eq!(cpu.audio_format(), (48_000, 2));
}

#[test]
fn audout_reads_the_channel_count_as_sixteen_bits() {
    // `OpenAudioOut` takes the channel count as a 16-bit field, and the two
    // bytes above it are padding the caller never writes. Reading the whole
    // word and echoing it back handed `nnSdk` a channel count of 0xcafe0002 --
    // negative, so its own "channelCount > 0" check failed and the title tore
    // its audio down and re-opened, which aborts.
    const AUDOUT: u64 = 0xA000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(AUDOUT, "audout:u");
    let tls = cpu.tls_base();

    let mut args = Vec::new();
    args.extend_from_slice(&0u32.to_le_bytes()); // sample rate: device default
    args.extend_from_slice(&0xcafe_0002u32.to_le_bytes()); // stereo, plus junk
    args.extend_from_slice(&0u64.to_le_bytes()); // aruid
    ipc_request_plain(&mut cpu, AUDOUT, 1, &args);
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 48_000, "device default rate");
    assert_eq!(cpu.mem.read_u32(tls + 0x24).unwrap(), 2, "the padding leaked through");
}

#[test]
fn audout_does_not_play_a_stopped_device() {
    // A device that has not been started is not playing. Its buffers still
    // come back -- the memory is the guest's -- but nothing is queued for the
    // host, because nothing was heard.
    const AUDOUT: u64 = 0xA000;
    const DESC: u32 = 0x8000;
    const PCM: u32 = 0x8100;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(AUDOUT, "audout:u");
    let tls = cpu.tls_base();

    let mut args = Vec::new();
    args.extend_from_slice(&48_000u32.to_le_bytes());
    args.extend_from_slice(&2u32.to_le_bytes());
    args.extend_from_slice(&0u64.to_le_bytes());
    ipc_request_plain(&mut cpu, AUDOUT, 1, &args);
    let device = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());

    cpu.mem.write_u16(PCM, 0x1234).unwrap();
    cpu.mem.write_u64(DESC + 8, u64::from(PCM)).unwrap();
    cpu.mem.write_u64(DESC + 16, 2).unwrap();
    cpu.mem.write_u64(DESC + 24, 2).unwrap();
    cpu.mem.write_u64(DESC + 32, 0).unwrap();
    ipc_request_plain_with_buffer(&mut cpu, device, 3, DESC, 40, false, &7u64.to_le_bytes());

    let mut played = [0i16; 4];
    assert_eq!(cpu.take_audio(&mut played), 0, "a stopped device played something");
    // The tag still comes back.
    ipc_request_plain(&mut cpu, device, 9, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 1);
}

#[test]
fn vi_native_window_names_the_binder_interface() {
    // `OpenLayer` answers with an Android parcel holding one flattened binder
    // object. libnx only reads the binder id out of it; nnSdk also checks the
    // interface name, and rejected the whole layer -- vi result 114-1, an
    // abort inside nn::vi::CreateLayer -- while the parcel carried a bare id
    // and nothing else.
    const VI: u64 = 0xB000;
    const WINDOW: u32 = 0x8000;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(VI, "vi:m");
    let tls = cpu.tls_base();

    // GetDisplayService first: OpenLayer lives on the
    // IApplicationDisplayService, not on the vi root.
    ipc_request_plain(&mut cpu, VI, 2, &[]);
    let display = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    assert_ne!(display, 0, "no IApplicationDisplayService");

    // OpenLayer, with the 0x100-byte native-window receive buffer the caller
    // always provides.
    ipc_request_plain_with_buffer(&mut cpu, display, 2020, WINDOW, 0x100, true, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0, "OpenLayer failed");
    let size = cpu.mem.read_u64(tls + 0x20).unwrap() as u32;

    // Parcel header: { payload_size, payload_off, objects_size, objects_off }.
    let payload_size = cpu.mem.read_u32(WINDOW).unwrap();
    let payload_off = cpu.mem.read_u32(WINDOW + 4).unwrap();
    let objects_size = cpu.mem.read_u32(WINDOW + 8).unwrap();
    let objects_off = cpu.mem.read_u32(WINDOW + 12).unwrap();
    assert_eq!(payload_size, 0x28, "a flat_binder_object is 0x28 bytes");
    assert_eq!(payload_off, 0x10);
    assert_eq!(objects_size, 4, "one object in the offset table");
    assert_eq!(objects_off, payload_off + payload_size);
    assert_eq!(size, objects_off + objects_size, "the reported size must cover it all");

    let payload = WINDOW + payload_off;
    assert_eq!(cpu.mem.read_u32(payload).unwrap(), 2, "flat_binder_object type");
    let binder = cpu.mem.read_u64(payload + 8).unwrap();
    assert_ne!(binder, 0, "no IGraphicBufferProducer id");
    let mut name = [0u8; 8];
    for (i, slot) in name.iter_mut().enumerate() {
        *slot = cpu.mem.read_u8(payload + 0x18 + i as u32).unwrap();
    }
    assert_eq!(&name, b"dispdrv\0", "the interface has to name itself");
}

