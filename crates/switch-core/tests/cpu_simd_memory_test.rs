//! SIMD and FP loads and stores.

mod cpu;

use cpu::*;

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
fn simd_scalar_byte_load_and_stur_q() {
    // `ldr b29, [x0, #0x280]` = 0x3d4a001d (SIMD scalar 8-bit load) and
    // `stur q17, [x0, #0x8]` = 0x3c808011 (SIMD scalar STUR).
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x3000);
    cpu.mem.map_zero(0x3000, 0x300).unwrap();
    cpu.mem.write_u8(0x3280, 0xAB).unwrap();
    // ldr b29, [x0, #0x280]: size=00, V=1, mode=01, opc=01, imm12=0x280, rn=0, rt=29
    let ldr_b =
        0b00u32 << 30 | 0b111 << 27 | (1 << 26) | (0b01 << 24) | (0b01 << 22) | (0x280 << 10) | 29;
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
    let stored = cpu.mem.dump(0x3008, 16).unwrap();
    assert_eq!(
        u128::from_le_bytes(stored.try_into().unwrap()),
        0x1122_3344_5566_7788_99AA_BBCC_DDEE_FF00
    );
}

#[test]
fn simd_post_index_store_writes_back_the_base() {
    // `str q27, [x2], #0x10` = 0x3c81045b writes back the base.
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
fn ld1_multiple_structures_writes_back_only_when_post_indexed() {
    // `ld1 {v1.16b, v2.16b}, [x2], #32` = 0x4cdfa041: immediate post-index has Rm == 31.
    let mut cpu = cpu_at(0x1000);
    map_ramp(&mut cpu, 0x3000, 64);
    cpu.set_reg(2, 0x3000);
    let cpu = run_program(cpu, 0x1000, &[0x4cdf_a041, nop()]);
    assert_eq!(cpu.read_vreg(1), mem_u128(&cpu, 0x3000));
    assert_eq!(cpu.read_vreg(2), mem_u128(&cpu, 0x3010));
    assert_eq!(cpu.read_reg(2), 0x3020);

    // `ld1 {v3.16b}, [x0]` = 0x4c407003 has no writeback.
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
    // `ld1r {v9.16b}, [x0]` = 0x4d40c009 and `ld1r {v10.4s}, [x0]` = 0x4d40c80a
    // (the replicate group, `scale == 0b11`).
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
    assert_eq!(
        cpu.read_vreg(11),
        u128::from_le_bytes(evens.try_into().unwrap())
    );
    assert_eq!(
        cpu.read_vreg(12),
        u128::from_le_bytes(odds.try_into().unwrap())
    );
    assert_eq!(
        cpu.mem.dump(0x3100, 32).unwrap(),
        (0..32u8).collect::<Vec<u8>>()
    );

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
    // `ld1 {v13.s}[1], [x0]` = 0x0d40900d replaces bits 32..63 and nothing else.
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
