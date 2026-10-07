//! The A64 integer core: ALU, addressing modes, branches, system registers, and disassembler.

mod cpu;

use cpu::*;

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
fn add_immediate_reads_register_31_as_sp() {
    let cpu = exec(&[add_imm(1, 31, 5, true), add_imm(2, 1, 0xFFF, false)], 100);
    // x1 = SP(0) + 5
    assert_eq!(cpu.read_x(1), 5);
    // w2 = w1 + 0xFFF
    assert_eq!(cpu.read_x(2), 0xFFF + 5);
}

#[test]
fn sub_and_flags() {
    // SUBS XZR, X2, X1 after x2=3, x1=3: Z set
    let mut cpu = exec(
        &[
            movz(1, 3, 0, true),
            movz(2, 3, 0, true),
            cmp_reg(2, 1, true),
        ],
        100,
    );
    assert_eq!(cpu.nzcv() & (1 << 30), 1 << 30); // Z
    assert_eq!(cpu.nzcv() & (1 << 31), 0); // N clear

    // SUBS XZR, X1, X2 with x1=1, x2=3: N set, C clear
    cpu = exec(
        &[
            movz(1, 1, 0, true),
            movz(2, 3, 0, true),
            cmp_reg(1, 2, true),
        ],
        100,
    );
    assert_eq!(cpu.nzcv() & (1 << 31), 1 << 31); // N
    assert_eq!(cpu.nzcv() & (1 << 29), 0); // C clear (borrow)
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
    assert_eq!(cpu.read_x(6), 0xCAFE_F00D); // LDR 32-bit zero-extended
}

#[test]
fn stp_ldp_pre_index() {
    let mut cpu = cpu_at(0x2000);
    cpu.set_reg(0, 0x4000);
    let code: [u32; 3] = [
        0xA9BE_07E0, // STP X0, X1, [SP, #-32]!
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
    assert_eq!(
        cpu.mem.read_u64(0x4000 - 32).unwrap(),
        0x1111_1111_1111_1111
    );
    assert_eq!(
        cpu.mem.read_u64(0x4000 - 24).unwrap(),
        0x2222_2222_2222_2222
    );
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
    let mut code = vec![bl(bl_off), svc(0)];
    code.resize((bl_off / 4) as usize, nop());
    code.extend([movz(9, 42, 0, true), ret(30)]);
    let cpu = exec(&code, 200);
    assert_eq!(cpu.read_x(9), 42);
    // x30 = return address = 0x1004
    assert_eq!(cpu.read_x(30), 0x1004);
}

#[test]
fn conditional_branch_and_compare() {
    // x1=0; CMP x1,xzr; B.NE +8 (not taken since Z set)
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
    // x1=0: CBZ branches past a MOV
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

    // TBNZ on bit 0, with x5=1
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
    // AND X1, X2, #0xFF: N=0, immr=0, imms=7
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(2, 0xABCD_EF12_3456_78FF);
    let code: [u32; 1] = [0b0 << 31
        | 0b00 << 29
        | 0b100100 << 23
        | 0 << 22
        | (0 << 16)
        | (7 << 10)
        | (2 << 5)
        | 1];
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
    // MADD X1, X2, X3, X4: x1 = 5*7 + 11 = 46
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
    // x1=5, x2=9; CMP x1,x2; CSEL x3, x1, x2, GT
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, 5);
    cpu.set_reg(2, 9);
    // GT is false, so the else operand X2=9 is taken.
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
    // csinv/csinc/csneg modify only the else operand. EQ true selects x1 in all three;
    // EQ false gives csinv=~9=-10, csinc=10, csneg=-9.
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
    // UBFX X1, X2, #4, #8 = UBFM X1, X2, #4, #11
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(2, 0xABCD_EF12_3456_78FF);
    let ubfm =
        1u32 << 31 | 0b10 << 29 | 0b100110 << 23 | 1 << 22 | (4 << 16) | (11 << 10) | (2 << 5) | 1;
    let mut bytes = Vec::new();
    {
        let insn = ubfm;
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(1), (0xABCD_EF12_3456_78FFu64 >> 4) & 0xFF);

    // SBFX: SBFM X3, X2, #0, #7 (sign-extend 8 bits) gives -1
    let sbfm =
        1u32 << 31 | 0b00 << 29 | 0b100110 << 23 | 1 << 22 | (0 << 16) | (7 << 10) | (2 << 5) | 3;
    let mut bytes = Vec::new();
    {
        let insn = sbfm;
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(3), 0xFFFF_FFFF_FFFF_FFFF); // 0xFF sign-extended = -1
}

#[test]
fn ldr_literal() {
    // LDR X1, #+8 loads the u64 literal at pc+8
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
    let code = [
        movz(1, 3, 0, true),
        movz(2, 3, 0, true),
        cmp_reg(1, 2, true),
        mrs_nzcv(4),
        0xD51B_4205, // msr nzcv, x5
        mrs_nzcv(6),
        svc(0),
    ];
    let mut bytes = Vec::new();
    for insn in code {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    cpu.set_reg(5, 0x9000_0000);
    cpu.run(code.len() as u64).unwrap();
    assert_eq!(cpu.read_x(4), 0x6000_0000, "Z and C after an equal compare");
    assert_eq!(cpu.nzcv(), 0x9000_0000, "MSR wrote N and V");
    assert_eq!(cpu.read_x(6), 0x9000_0000, "and MRS reads them back");
}

#[test]
fn exclusive_load_store() {
    // A bare STXR with no prior LDXR fails and stores nothing.
    let stxr: u32 = 0b11 << 30 | 0b001000000 << 21 | (0 << 16) | (1 << 10) | (1 << 5) | 2;
    let ldxr: u32 = 0b11 << 30 | 0b001000010 << 21 | (1 << 10) | (1 << 5) | 3;
    let bytes = |code: &[u32]| -> Vec<u8> { code.iter().flat_map(|i| i.to_le_bytes()).collect() };

    let mut cpu = Cpu::new();
    cpu.set_reg(1, 0x3000);
    cpu.mem.map_zero(0x3000, 8).unwrap();
    cpu.mem.map(0x1000, &bytes(&[stxr])).unwrap();
    cpu.set_pc(0x1000);
    cpu.set_reg(2, 0x1234_5678_9ABC_DEF0);
    cpu.run(1).unwrap();
    assert_eq!(
        cpu.mem.read_u64(0x3000).unwrap(),
        0,
        "a bare STXR stored anyway"
    );
    assert_eq!(cpu.read_x(0), 1, "a bare STXR reported success");

    // LDXR then STXR to the same address is the pair, and it goes through.
    let mut cpu = Cpu::new();
    cpu.set_reg(1, 0x3000);
    cpu.mem.map_zero(0x3000, 8).unwrap();
    cpu.mem.map(0x1000, &bytes(&[ldxr, stxr])).unwrap();
    cpu.set_pc(0x1000);
    cpu.set_reg(2, 0x1234_5678_9ABC_DEF0);
    cpu.run(2).unwrap();
    assert_eq!(cpu.read_x(3), 0, "LDXR read the wrong value");
    assert_eq!(cpu.mem.read_u64(0x3000).unwrap(), 0x1234_5678_9ABC_DEF0);
    assert_eq!(cpu.read_x(0), 0, "the pair failed");

    // And the monitor is one-shot: a second STXR after it fails.
    let mut cpu = Cpu::new();
    cpu.set_reg(1, 0x3000);
    cpu.mem.map_zero(0x3000, 8).unwrap();
    cpu.mem.map(0x1000, &bytes(&[ldxr, stxr, stxr])).unwrap();
    cpu.set_pc(0x1000);
    cpu.set_reg(2, 0x1234_5678_9ABC_DEF0);
    cpu.run(3).unwrap();
    assert_eq!(cpu.read_x(0), 1, "the monitor survived its own STXR");
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
    assert_eq!(cpu.read_x(3), 56); // clz(0xF0) = 56
    assert_eq!(cpu.read_x(4), 4); // ctz(0xF0) = 4
    assert_eq!(cpu.read_x(5), 0xF000_0000); // rev32 of 0x...00F0
}

#[test]
fn add_immediate_preserves_flags() {
    // CMP x1, x1 (sets Z), then ADD X2, X1, #'0': flags must stay set.
    let mut cpu = Cpu::new();
    cpu.set_reg(1, 5);
    let code = [
        cmp_reg(1, 1, true),     // Z=1
        add_imm(2, 1, 48, true), // must NOT clear flags
        bcond(0x0, 8),           // B.EQ +8 -> taken only if Z still set
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
    assert_eq!(cpu.read_x(3), 0); // EQ branch taken → movz(3) skipped
    assert_eq!(cpu.read_x(4), 0xFF);
}

#[test]
fn tpidr_el0_roundtrips_and_tpidrro_el0_does_not() {
    // TPIDRRO_EL0 is read-only at EL0: a `msr` to it is ignored.
    const VALUE: u64 = 0x1234_5678_9ABC_DEF0;
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    let tls = cpu.tls_base();
    cpu.set_reg(1, VALUE);
    let code = [
        0xD51B_D041u32, // msr tpidr_el0, x1
        0xD53B_D042,    // mrs x2, tpidr_el0
        0xD51B_D061,    // msr tpidrro_el0, x1
        0xD53B_D063,    // mrs x3, tpidrro_el0
    ];
    let mut bytes = Vec::new();
    for insn in code {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(4).unwrap();
    assert_eq!(cpu.read_x(2), VALUE, "TPIDR_EL0 is the guest's to write");
    assert_eq!(
        cpu.read_x(3),
        u64::from(tls),
        "a guest write to TPIDRRO_EL0 must not displace the thread's TLS base"
    );
}

#[test]
fn the_generic_timer_counts_and_reports_its_own_rate() {
    // `nn::os::GetSystemTick` reads CNTPCT_EL0, so it must advance.
    const NOPS: usize = 2000;
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    let mut code = vec![0xD53B_E021u32]; // mrs x1, cntpct_el0
    code.extend(std::iter::repeat_n(0xD503_201Fu32, NOPS)); // nop
    code.push(0xD53B_E022); // mrs x2, cntpct_el0
    code.push(0xD53B_E043); // mrs x3, cntvct_el0
    code.push(0xD53B_E004); // mrs x4, cntfrq_el0
    let mut bytes = Vec::new();
    for insn in &code {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(code.len() as u64).unwrap();
    assert_eq!(cpu.read_x(1), 0, "read at cycle 0");
    // 2001 cycles of the 1.02 GHz clock are 37.66 ticks of the 19.2 MHz one.
    assert_eq!(
        cpu.read_x(2),
        37,
        "CNTPCT_EL0 must advance as the guest runs, at the tick rate"
    );
    assert_eq!(cpu.read_x(3), 37, "CNTVCT_EL0 reads the same counter");
    assert_eq!(cpu.read_x(4), 19_200_000, "CNTFRQ_EL0 is the 19.2 MHz rate");
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
    assert_eq!(cpu.nzcv(), 0, "SUBS would have set C");
}

#[test]
fn adds_shifted_register() {
    // ADDS X2, X0, X1 must add and set flags.
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
    assert_eq!(cpu.read_x(2), 0x30); // 0x10 + 0x20, not subtraction
    assert_eq!(cpu.read_x(3), 0xEE); // Z clear (result nonzero) -> branch not taken
}

#[test]
fn cmp_does_not_clobber_sp() {
    // CMP X0, #0x10 == SUBS XZR, X0, #0x10: must not write the stack pointer.
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
    assert!(
        trace.contains("movz x1, #0x7"),
        "fault trace must show the run-up:\n{trace}"
    );
    assert!(
        trace.contains(".word 0x00000000"),
        "fault trace must show the faulting word:\n{trace}"
    );
}

#[test]
fn adrp_adr_with_nonzero_immlo() {
    // ADRP X0, #0x1000: page(0x1000) + 0x1000 = 0x2000 (immlo = 01)
    // ADR X1, #0x5: pc(0x1004) + 5 = 0x1009 (immlo = 01)
    let cpu = exec(
        &[
            1u32 << 31 | 1 << 29 | 0b10000 << 24, // adrp x0, #0x1000
            0b10000 << 24 | 1 << 29 | 1 << 5 | 1, // adr x1, #0x5
        ],
        100,
    );
    assert_eq!(cpu.read_x(0), 0x2000);
    assert_eq!(cpu.read_x(1), 0x1009);
}

#[test]
fn disassembler_agrees_with_llvm_on_load_store_size_and_addressing() {
    use switch_core::disasm::disassemble;
    // Encodings and expected strings are from `llvm-mc --show-encoding` and `--disassemble`.
    for (insn, want) in [
        // Sizes on the indexed forms.
        (0x38001408u32, "strb w8, [x0], #1"),
        (0x78002c29, "strh w9, [x1, #2]!"),
        // Unscaled (stur/ldur): a signed 9-bit byte offset.
        (0xb81fc062, "stur w2, [x3, #-4]"),
        (0x381ff0a4, "sturb w4, [x5, #-1]"),
        (0xf85f80e6, "ldur x6, [x7, #-8]"),
        (0x385fe128, "ldurb w8, [x9, #-2]"),
        (0xb89fc16a, "ldursw x10, [x11, #-4]"),
        // Unprivileged.
        (0x380039ac, "sttrb w12, [x13, #3]"),
        // Register offset, narrower than 64-bit.
        (0x38214913, "strb w19, [x8, w1, uxtw]"),
        (0x78647862, "ldrh w2, [x3, x4, lsl #1]"),
        // Acquire/release and exclusive: size, and `o0`.
        (0x089ffd09, "stlrb w9, [x8]"),
        (0x48dffcc7, "ldarh w7, [x6]"),
        (0x0801fc62, "stlxrb w1, w2, [x3]"),
        (0xc85ffca4, "ldaxr x4, [x5]"),
        // SIMD&FP loads and stores.
        (0x3d800420, "str q0, [x1, #0x10]"),
        (0xbc5fc062, "ldur s2, [x3, #-4]"),
        (0x6d0127e8, "stp d8, d9, [sp, #0x10]"),
        (0xacc12c4a, "ldp q10, q11, [x2], #0x20"),
        // LDPSW is not a 32-bit LDP: two signed words into 64-bit registers.
        (0x694001b2, "ldpsw x18, x0, [x13, #0x0]"),
        (0x294114c4, "ldp w4, w5, [x6, #0x8]"),
    ] {
        assert_eq!(disassemble(insn), want, "for {insn:#010x}");
    }
}

#[test]
fn disassembler_names_rev_ccmp_and_the_barriers() {
    use switch_core::disasm::disassemble;
    // The 32-bit form reverses the whole register, so it is REV; REV32 is 64-bit only.
    assert_eq!(disassemble(0x5ac00808), "rev w8, w0");
    assert_eq!(disassemble(0xdac00829), "rev32 x9, x1");
    assert_eq!(disassemble(0xdac00c4a), "rev x10, x2");

    // CCMP subtracts and CCMN adds; bit 30 chooses.
    assert_eq!(disassemble(0x7a400804), "ccmp w0, #0x0, #0x4, eq");
    assert_eq!(disassemble(0x3a411822), "ccmn w1, #0x1, #0x2, ne");

    // The barriers share the hint encoding space but are their own instructions.
    assert_eq!(disassemble(0xd5033bbf), "dmb #0xb");
    assert_eq!(disassemble(0xd5033fdf), "isb #0xf");
    assert_eq!(disassemble(0xd5033f5f), "clrex");
}

#[test]
fn disassembler_produces_readable_output() {
    use switch_core::disasm::disassemble;
    // MOVZ X1, #0x1234
    assert_eq!(disassemble(0xD2824681), "movz x1, #0x1234");
    // ADR X0, #0x54
    assert_eq!(disassemble(0x100002A0), "adr x0, #0x54");
    // ADRP X0, #0x1000 (non-zero immlo)
    assert_eq!(disassemble(0xB0000000), "adrp x0, #0x1");
    // ADRP X0, #0x235
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
    // movz w1, #0x4653 ; movk w1, #0x4f43, lsl #16 gives 0x4f434653; the 32-bit hw field starts at bit 21.
    let cpu = exec(&[movz(1, 0x4653, 0, false), movk(1, 0x4f43, 1, false)], 2);
    assert_eq!(cpu.read_x(1), 0x4f43_4653);
}

#[test]
fn add_carry_overflow_masks_to_operand_size() {
    // CMP W0, #4 with W0 = 0 must clear C (borrow), so B.CC is taken.
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
    // ldr w2, [x19, x0]: a 32-bit register-offset load.
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

    // SMULH of i64::MIN * 2 is -1
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x8000_0000_0000_0000);
    cpu.set_reg(1, 2);
    let code = [smulh(4, 0, 1), nop()];
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_x(4), u64::MAX);
}

#[test]
fn ldrsw_sign_extends_and_loads() {
    // LDRSW loads and sign-extends: 0xFFFF8001 yields 0xFFFFFFFF_FFFF8001.
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
    // `prfm pldl1keep, [x1]` = 0xF9800020 is a hint and must not write a register.
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
    // `bfi w0, w1, #8, #24` inserts w1's low 24 bits into w0[31:8] (qemu: w0=0x5, w1=0xAB -> 0x0000AB05).
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x5);
    cpu.set_reg(1, 0xAB);
    let code = [0x3318_5C20, nop()]; // bfi w0, w1, #8, #24 (clang-encoded)
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_x(0), 0x0000_AB05);
}

#[test]
fn bfxil_extracts_field_into_low_bits() {
    // `bfxil w0, w1, #16, #8` copies w1[23:16] into w0[7:0], keeping w0's upper bits.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x1234_5678);
    cpu.set_reg(1, 0x0043_0000);
    let code = [0x3310_5C20, nop()]; // bfxil w0, w1, #16, #8 (clang-encoded)
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_x(0), 0x1234_5678 & !0xFF | 0x43);
}

#[test]
fn logical_immediate_mask_80808080() {
    // 0x3201c3f4 = `mov w20, #0x80808080`: imms bits above the element size are ignored.
    let code = [0x3201c3f4, nop()];
    let cpu = run_program(cpu_at(0x1000), 0x1000, &code);
    assert_eq!(cpu.read_x(20), 0x8080_8080);
}

#[test]
fn ldrsw_register_offset_shifts_by_log2() {
    // `ldrsw x8, [x9, x8, lsl #2]` = 0xb8a87928: the shift is log2(size), not the byte count.
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
fn dczid_el0_reports_a57_block_size() {
    // mrs x5, dczid_el0 must return BS=4 (64-byte DC ZVA).
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
fn ccmp_eq_sets_carry_for_unsigned_ge() {
    // ccmp x21, x1, #0, eq with x21=0x20, x1=0x18 leaves C=1, so b.hs is taken.
    let code: [u32; 5] = [
        0xd2800a95, // mov x21, #0x20
        0xd2800301, // mov x1, #0x18
        0xf10000ff, // cmp x7, #0   (sets Z=1; x7 is zero)
        0xfa4102a0, // ccmp x21, x1, #0, eq
        0x540000a2, // b.hs #+20
    ];
    let mut cpu = cpu_at(0x1000);
    let mut bytes = Vec::new();
    for insn in &code {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(code.len() as u64).unwrap();
    assert_eq!(
        cpu.get_pc(),
        0x1010 + 20,
        "b.hs should have been taken; ccmp produced wrong flags"
    );
}

#[test]
fn ccmp_subtract_carry_in() {
    // CCMP computes Rn - imm with the implied carry in: 0 - 0 sets Z and C.
    let code: [u32; 4] = [
        0xd2800102, // mov x2, #8
        0xd2800000, // mov x0, #0
        0xf100005f, // cmp x2, #0 (Z clear, C set)
        0xfa401800, // ccmp x0, #0, #0, ne (Z set, C set)
    ];
    let cpu = exec(&code, 100);
    assert_eq!(cpu.nzcv() & (1 << 30), 1 << 30, "Z must be set for 0-0");
    assert_eq!(
        cpu.nzcv() & (1 << 29),
        1 << 29,
        "C must be set for 0-0 (no borrow)"
    );
    assert_eq!(cpu.nzcv() & (1 << 31), 0, "N must be clear for 0-0");
}

#[test]
fn ccmp_ccmn_immediate_is_unsigned() {
    // The 5-bit immediate is unsigned for both CCMP and CCMN.
    // CCMP x0, #0x10 with x0=1: 1-16 = -15, N set.
    let c1: [u32; 3] = [
        0xd2800020, // mov x0, #1
        0xf100001f, // cmp x0, #0      (Z clear → NE holds)
        0xfa501800, // ccmp x0, #0x10, #0, ne
    ];
    let cpu = exec(&c1, 100);
    assert_eq!(cpu.nzcv() & (1 << 31), 1 << 31, "CCMP 1-16 = -15 → N set");
    assert_eq!(cpu.nzcv() & (1 << 30), 0, "CCMP 1-16 ≠ 0 → Z clear");

    // CCMN x0, #0x10 with x0=1: 1+16 = 17, all flags clear.
    let c2: [u32; 3] = [
        0xd2800020, // mov x0, #1
        0xf100001f, // cmp x0, #0
        0xba501800, // ccmn x0, #0x10, #0, ne
    ];
    let cpu = exec(&c2, 100);
    assert_eq!(cpu.nzcv() & 0xF000_0000, 0, "CCMN 1+16 = 17 → no flags set");
}

#[test]
fn sub_shifted_register_reads_xzr_not_sp() {
    // `neg x1, x0` = `sub x1, xzr, x0` (0xcb0003e1): register 31 is XZR in the shifted-register form.
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
    // `add sp, sp, #0x10` (0x910043ff): in the immediate form register 31 is SP.
    let mut cpu = cpu_at(0x1000);
    cpu.set_pc_and_sp(0x1000, 0x1000_0000);
    let cpu = run_program(cpu, 0x1000, &[0x9100_43ff, nop()]);
    assert_eq!(cpu.sp(), 0x1000_0010);
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
fn ctr_el0_reports_64_byte_cache_lines() {
    // `mrs x7, ctr_el0` = 0xd53b0027; cache-flush loops stride by `4 << DminLine`.
    let cpu = run_program(cpu_at(0x1000), 0x1000, &[0xd53b_0027, nop()]);
    assert_eq!(cpu.read_reg(7), 0x8444_C004);
}

#[test]
fn blr_reads_its_target_before_linking() {
    // `blr x30` reads the target before writing x30.
    let mut cpu = cpu_at(0x1000);
    cpu.mem.map_zero(0x2000, 0x100).unwrap();
    cpu.set_reg(30, 0x2000);
    // blr x30 = 0xd63f03c0
    let cpu = run_program(cpu, 0x1000, &[0xd63f_03c0u32]);
    assert_eq!(cpu.get_pc(), 0x2000, "branched to the old x30");
    assert_eq!(
        cpu.read_reg(30),
        0x1004,
        "and linked to the next instruction"
    );
}

#[test]
fn a_logical_immediate_writes_sp_but_ands_writes_the_zero_register() {
    // `AND`, `ORR` and `EOR` (immediate) write SP for register 31; only `ANDS` (`TST`) writes XZR.
    // LLVM aligns stack frames with `and sp, x9, #imm`.
    let cpu = exec(
        &[
            0xd2800fe9, // mov x9, #0x7f
            0x927ae53f, // and sp, x9, #0xffffffffffffffc0
        ],
        8,
    );
    assert_eq!(
        cpu.sp(),
        0x40,
        "and (immediate) has to write SP, not discard"
    );

    // `mov sp, x9` is `orr sp, x9, #0`.
    let cpu = exec(
        &[
            0xd2801fe9, // mov x9, #0xff
            0xb2400d3f, // orr sp, x9, #0xf
        ],
        8,
    );
    assert_eq!(cpu.sp(), 0xff);

    // ANDS with Rd 31 is TST: the flags move, SP does not.
    let cpu = exec(
        &[
            0xd28001e9, // mov x9, #0xf
            0xf240053f, // ands xzr, x9, #0x3
        ],
        8,
    );
    assert_eq!(cpu.sp(), 0, "ands must leave SP alone");
    assert_eq!(cpu.nzcv() >> 30 & 1, 0, "0xf & 0x3 is not zero");
}

#[test]
fn thirty_two_bit_writes_clear_the_upper_half() {
    // Every write to a W register zeroes bits 63:32, including SBFM's sign extension.
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
    // `asr w2, w2, w3` on a negative word must give all-ones.
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
    // Expected values are from qemu-aarch64 via `tools/difftest.py --scalar`.
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

    // ADCS: bit 30 selects subtract and bit 29 sets the flags.
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

#[test]
fn the_sysreg_move_helper_encodes_what_the_assembler_does() {
    assert_eq!(mrs(0, 3, 4, 4, 0), 0xD53B_4400, "mrs x0, fpcr");
    assert_eq!(msr(0, 3, 4, 4, 0), 0xD51B_4400, "msr fpcr, x0");
    assert_eq!(mrs(1, 3, 4, 4, 1), 0xD53B_4421, "mrs x1, fpsr");
}

/// `ExtendReg` truncates to the operation width rather than clamping.
#[test]
fn a_signed_32_bit_extend_keeps_its_low_bits() {
    let code = &[
        0x0b21_8040, // add w0, w2, w1, sxtb
        0x0b21_a047, // add w7, w2, w1, sxth
        0x0b21_c049, // add w9, w2, w1, sxtw
    ];
    // Negative in all three widths, with a different value in each.
    let (jit, interp) = both_engines(&[(1, 0xFFFF_FFFF_8000_8080), (2, 0)], code);
    for cpu in [&jit, &interp] {
        assert_eq!(
            cpu.read_x(0),
            0xFFFF_FF80,
            "sxtb clamped instead of truncating"
        );
        assert_eq!(
            cpu.read_x(7),
            0xFFFF_8080,
            "sxth clamped instead of truncating"
        );
        assert_eq!(
            cpu.read_x(9),
            0x8000_8080,
            "sxtw clamped instead of truncating"
        );
    }
}

/// `BIC`/`ORN`/`EON` invert the shifted operand (shift first, then invert).
#[test]
fn bic_and_friends_invert_after_shifting() {
    let code = &[
        0x8a22_1020, // bic x0, x1, x2, lsl #4
        0x2a25_2083, // orn w3, w4, w5, lsl #8
        0xca6a_1128, // eon x8, x9, x10, lsr #4
    ];
    let setup = &[(1, 0xFF), (2, 0x0F), (4, 0), (5, 0xFF), (9, 0), (10, 0xFF)];
    let (jit, interp) = both_engines(setup, code);
    for cpu in [&jit, &interp] {
        assert_eq!(cpu.read_x(0), 0x0F, "bic inverted before shifting");
        assert_eq!(cpu.read_x(3), 0xFFFF_00FF, "orn inverted before shifting");
        assert_eq!(
            cpu.read_x(8),
            0xFFFF_FFFF_FFFF_FFF0,
            "eon inverted before shifting"
        );
    }
}

/// A sign-extending load into a W register zeroes the top half.
#[test]
fn a_signed_load_into_a_w_register_narrows_like_every_other_w_write() {
    let code = &[
        0x79c0_0001, // ldrsh w1, [x0]
        0x7980_0002, // ldrsh x2, [x0]
        0x39c0_0803, // ldrsb w3, [x0, #2]
        0x3980_0804, // ldrsb x4, [x0, #2]
    ];
    for jit in [true, false] {
        let mut cpu = cpu_at(0x1000);
        cpu.set_jit_enabled(jit);
        cpu.mem.map_zero(0x2000, 0x100).unwrap();
        cpu.mem.write_u16(0x2000, 0xFF00).unwrap();
        cpu.mem.write_u8(0x2002, 0x80).unwrap();
        cpu.set_reg(0, 0x2000);
        let cpu = run_program(cpu, 0x1000, code);
        assert_eq!(cpu.read_x(1), 0x0000_0000_FFFF_FF00, "ldrsh w (jit={jit})");
        assert_eq!(cpu.read_x(2), 0xFFFF_FFFF_FFFF_FF00, "ldrsh x (jit={jit})");
        assert_eq!(cpu.read_x(3), 0x0000_0000_FFFF_FF80, "ldrsb w (jit={jit})");
        assert_eq!(cpu.read_x(4), 0xFFFF_FFFF_FFFF_FF80, "ldrsb x (jit={jit})");
    }
}

/// `CLREX` clears the local monitor, so a following `STXR` fails.
#[test]
fn clrex_clears_the_reservation_a_store_exclusive_needs() {
    let mut cpu = cpu_at(0x1000);
    cpu.mem.map_zero(0x2000, 0x100).unwrap();
    cpu.set_reg(0, 0x2000);
    cpu.set_reg(2, 0xAAAA);
    let code = &[
        0xc85f7c01, // ldxr x1, [x0]
        0xd5033f5f, // clrex
        0xc8037c02, // stxr w3, x2, [x0]
        0xf9400004, // ldr x4, [x0]
    ];
    let cpu = run_program(cpu, 0x1000, code);
    assert_eq!(cpu.read_x(3), 1, "the store-exclusive reported success");
    assert_eq!(cpu.read_x(4), 0, "it stored anyway");
}

/// A 32-bit `LDXP`/`STXP` pair moves two words, not two doublewords.
#[test]
fn a_32_bit_exclusive_pair_moves_words() {
    let mut cpu = cpu_at(0x1000);
    cpu.mem.map_zero(0x2000, 0x100).unwrap();
    cpu.mem.write_u64(0x2008, 0xDEAD_BEEF_DEAD_BEEF).unwrap();
    cpu.set_reg(0, 0x2000);
    cpu.set_reg(1, 0x1111_1111_1111_1111);
    cpu.set_reg(2, 0x2222_2222_2222_2222);
    let code = &[
        0x887f0c03, // ldxp w3, w3, [x0]  (both halves, discarded)
        0x88240801, // stxp w4, w1, w2, [x0]
        0xf9400405, // ldr x5, [x0, #8]
        0xb9400006, // ldr w6, [x0]
        0xb9400407, // ldr w7, [x0, #4]
    ];
    let cpu = run_program(cpu, 0x1000, code);
    assert_eq!(cpu.read_x(4), 0, "the pair store failed");
    assert_eq!(cpu.read_x(6), 0x1111_1111, "first word");
    assert_eq!(cpu.read_x(7), 0x2222_2222, "second word");
    assert_eq!(
        cpu.read_x(5),
        0xDEAD_BEEF_DEAD_BEEF,
        "a word pair wrote over its neighbour"
    );
}
