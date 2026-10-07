//! Advanced SIMD integer vector operations.

mod cpu;

use cpu::*;

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

#[test]
fn simd_three_same_add_sub_compare() {
    // dup v0.16b, w1 (0x5a) ; dup v1.16b, w2 (0x3d) ; sub v2.4s, v0.4s, v1.4s
    // → lanes of 0x5a5a5a5a - 0x3d3d3d3d ; mov x3, v2.d[0]
    let code = [
        dup16(0, 1),
        dup16(1, 2),
        sub4s(2, 0, 1),
        umov_d0(3, 2),
        nop(),
    ];
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, 0x5a);
    cpu.set_reg(2, 0x3d);
    cpu.set_reg(3, 0xA5A5);
    cpu.set_vreg(2, u128::MAX);
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_vreg(2), 0x1d1d1d1d_1d1d1d1d_1d1d1d1d_1d1d1d1d);
    assert_eq!(cpu.read_x(3), 0x1d1d1d1d_1d1d1d1d);

    // cmeq v4.16b, v0.16b, v1.16b → all-ones since equal ; mov x5, v4.d[0]
    let code = [
        dup16(0, 1),
        dup16(1, 2),
        cmeq16(4, 0, 1),
        umov_d0(5, 4),
        nop(),
    ];
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, 0x3d);
    cpu.set_reg(2, 0x3d);
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_x(5), u64::MAX);

    // uhadd v6.16b, v0.16b, v1.16b with unequal bytes (1 + 3) >> 1 = 2
    let code = [
        dup16(0, 1),
        dup16(1, 2),
        uhadd16(6, 0, 1),
        umov_d0(7, 6),
        nop(),
    ];
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, 1);
    cpu.set_reg(2, 3);
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_x(7), 0x0202_0202_0202_0202);
}

#[test]
fn simd_saturating_add_sub_clamp_by_signedness() {
    let code = [
        0x6E21_0C02, // uqadd v2.16b, v0.16b, v1.16b
        0x4E21_0C03, // sqadd v3.16b, v0.16b, v1.16b
        0x6E21_2C04, // uqsub v4.16b, v0.16b, v1.16b
        0x4E21_2C05, // sqsub v5.16b, v0.16b, v1.16b
        0x4EE7_0CC8, // sqadd v8.2d, v6.2d, v7.2d
        nop(),
    ];
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, 0x01ff807f_01ff807f_01ff807f_01ff807f);
    cpu.set_vreg(1, 0x8001ff01_8001ff01_8001ff01_8001ff01);
    cpu.set_vreg(6, 0xffffffffffffffff_7fffffffffffffff);
    cpu.set_vreg(7, 0x8000000000000000_0000000000000001);
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_vreg(2), 0x81ffff80_81ffff80_81ffff80_81ffff80);
    assert_eq!(cpu.read_vreg(3), 0x8100807f_8100807f_8100807f_8100807f);
    assert_eq!(cpu.read_vreg(4), 0x00fe007e_00fe007e_00fe007e_00fe007e);
    assert_eq!(cpu.read_vreg(5), 0x7ffe817e_7ffe817e_7ffe817e_7ffe817e);
    assert_eq!(cpu.read_vreg(8), 0x8000000000000000_7fffffffffffffff);
}

#[test]
fn simd_pairwise_addp() {
    // v1 = {0..15}, v2 = {0x10..0x1f}; addp v3.16b, v1.16b, v2.16b puts v1's
    // pairwise sums in the low half and v2's in the high half.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(1, u128::from_le_bytes(std::array::from_fn(|i| i as u8)));
    cpu.set_vreg(
        2,
        u128::from_le_bytes(std::array::from_fn(|i| 0x10 + i as u8)),
    );
    cpu.set_vreg(3, u128::MAX);
    let cpu = run_program(cpu, 0x1000, &[addp16(3, 1, 2), nop()]);
    assert_eq!(cpu.read_vreg(3), 0x3d393531_2d292521_1d191511_0d090501);
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
    let ldr_q = |rt: u32, imm: u32| 0x3DC0_0000u32 | rt | ((imm >> 4) << 10);
    let str_q = |rt: u32, imm: u32| 0x3D80_0000u32 | rt | ((imm >> 4) << 10);
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
        assert_eq!(
            cpu.mem.read_u8(0x3020 + 2 * i + 1).unwrap(),
            0x10u8 + i as u8
        );
    }
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
    // v2: the low half reverses the first eight table bytes; the high half (0x40) is out of range.
    for i in 0..8u32 {
        cpu.mem.write_u8(0x3020 + i, 7 - i as u8).unwrap();
        cpu.mem.write_u8(0x3028 + i, 0x40).unwrap();
    }
    // v3 starts as 0xaa so TBX keeping a byte differs from TBL zeroing it.
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

#[test]
fn simd_ins_element_moves_one_lane_and_leaves_the_rest() {
    // `INS <Vd>.<Ts>[<i1>], <Vn>.<Ts>[<i2>]` is the `op == 1` half of the
    // AdvSIMD copy group; libnx's `smEncodeName` builds service names with it.
    let mut cpu = cpu_at(0x1000);
    // v31 holds 'n' (as `ldr b31, [x0]` would leave it); one source register
    // per remaining character, each with the byte in lane 0.
    cpu.set_vreg(31, 0x6E);
    for (i, ch) in b"s:am2".iter().enumerate() {
        cpu.set_vreg(29 - i as u8, u128::from(*ch));
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

    // A non-zero source lane and wider lanes; INS leaves the rest of Vd alone.
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
    let out: Vec<u8> = (0..5)
        .map(|i| cpu.mem.read_u8(0x3040 + i).unwrap())
        .collect();
    assert_eq!(out, vec![0x10, 0x1f, 0x20, 0x2f, 0x00]);
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
    let code = [
        ldr_q(0, 0),
        smaxv,
        sminv,
        umov_d0(3, 1),
        umov_d0(4, 2),
        nop(),
    ];
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

#[test]
fn movi_modified_immediate_cmodes() {
    // MOVI cmode and the split imm8 field (bits 18:16 ++ 9:5), checked under qemu-aarch64.
    // movi v0.8b, #0x1c  → every byte 0x1c (q=0: upper half cleared)
    let cpu = run_program(cpu_at(0x1000), 0x1000, &[0x0f00e780, nop()]);
    assert_eq!(cpu.read_vreg(0), 0x1c1c1c1c_1c1c1c1c);
    // movi v3.8h, #0x1c  → every halfword 0x001c
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
    // movi d8, #0  (the encoding sdl-hello hit) → zero, upper half included
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(8, u128::MAX);
    let cpu = run_program(cpu, 0x1000, &[0x2f00e408, nop()]);
    assert_eq!(cpu.read_vreg(8), 0);
}

#[test]
fn scalar_cmge_with_zero_masks_predicate() {
    // `cmge d31, d31, #0` = 0x7ee08bff: scalar compare-to-zero.
    // Encoding: bits[31:30] = 01 (D), U (bit29) = 1, bits[28:25] = 1111,
    // bits[24:21] = 0111, bits[20:16] = 00000 (zero operand), op = bits[15:10]
    // = 100010 (GE), Rn = bits[9:5], Rd = bits[4:0].
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(31, 0); // 0 >= 0 → all-ones
    let cpu = run_program(cpu, 0x1000, &[0x7ee0_8bff, nop()]);
    assert_eq!(cpu.read_vreg(31), u64::MAX as u128);

    // cmgt d4, d5, #0 = 0x5ee088a4 (U=0, op=100010): negative operand → 0.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(4, u128::MAX);
    cpu.set_vreg(5, 0x8000_0000_0000_0000);
    let cpu = run_program(cpu, 0x1000, &[0x5ee0_88a4, nop()]);
    assert_eq!(cpu.read_vreg(4), 0);
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

#[test]
fn bsl_bit_and_bif_take_their_mask_from_the_right_register() {
    // BSL selects with Vd, BIT and BIF with Vm.
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
    assert_eq!(
        lanes_u32(cpu.read_vreg(0)),
        [8, 0x0101_0101, 0, 0x0100_0001]
    );
    assert_eq!(lanes_u32(cpu.read_vreg(2)), [31, 0, 32, 16]);
    assert_eq!(
        lanes_u32(cpu.read_vreg(4)),
        [u32::MAX, 0, 0xEDCB_A987, u32::MAX]
    );
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
    assert_eq!(
        lanes_u32(cpu.read_vreg(10)),
        [(-1i32) as u32, 2, 0, (-5i32) as u32]
    );
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
    assert_eq!(
        cpu.read_vreg(24),
        u64x2([0x0002_0000_0001_0000, 0x0004_0000_0003_0000])
    );
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
fn ext_extracts_across_a_vector_pair() {
    // `ext v0.16b, v1.16b, v2.16b, #4` = 0x6e022020: the top 12 bytes of Vn,
    // then the low 4 of Vm.
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
        &[
            0x4e43_285c,
            0x4e43_6850,
            0x4e43_384a,
            0x4e43_784b,
            0x4e43_184c,
            0x4e43_584d,
            nop(),
        ],
    );
    // TRN1 takes the even elements of both, interleaved.
    assert_eq!(
        cpu.read_vreg(28),
        u64x2([0x0004_3000_0001_1000, 0x0040_7000_0010_5000])
    );
    // TRN2 the odd ones.
    assert_eq!(
        cpu.read_vreg(16),
        u64x2([0x0008_4000_0002_2000, 0x0080_8000_0020_6000])
    );
    // ZIP1 interleaves the low halves, ZIP2 the high halves.
    assert_eq!(
        cpu.read_vreg(10),
        u64x2([0x0002_2000_0001_1000, 0x0008_4000_0004_3000])
    );
    assert_eq!(
        cpu.read_vreg(11),
        u64x2([0x0020_6000_0010_5000, 0x0080_8000_0040_7000])
    );
    // UZP1 packs Vn's even elements then Vm's; UZP2 the odd ones.
    assert_eq!(
        cpu.read_vreg(12),
        u64x2([0x7000_5000_3000_1000, 0x0040_0010_0004_0001])
    );
    assert_eq!(
        cpu.read_vreg(13),
        u64x2([0x8000_6000_4000_2000, 0x0080_0020_0008_0002])
    );
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

    // The vector (three-different) form.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(2, u64x2([0x0004_0003_0002_0001, 0]));
    cpu.set_vreg(3, u64x2([0xFFFF_0002_0003_0004, 0]));
    cpu.set_vreg(5, u64x2([0x0000_000A_0000_000A, 0x0000_000A_0000_000A]));
    // smull v5.4s, v2.4h, v3.4h
    let cpu = run_program(cpu, 0x1000, &[0x0e63_c045, nop()]);
    // 1*4, 2*3, 3*2, 4*-1
    assert_eq!(lanes_u32(cpu.read_vreg(5)), [4, 6, 6, (-4i32) as u32]);
}

#[test]
fn advsimd_scalar_by_element_multiplies() {
    // `01 U 11111 size L M Rm opcode H 0 Rn Rd`: scalar by-element, the bottom
    // element written and the rest zeroed. Encodings are LLVM's.

    // fmul s3, s4, v3.s[0]: Rd == Rm, so Vm must be read before Vd is written.
    let cpu = simd1(
        0x5f839083,
        &[(4, f32b(3.0)), (3, f32b(2.0) | (f32b(9.0) << 32))],
    );
    assert_eq!(cpu.read_vreg(3), f32b(6.0), "fmul s3, s4, v3.s[0]");

    // fmul d0, d1, v2.d[1]: the high half of Vm; Vd's top half is cleared.
    let cpu = simd1(
        0x5fc29820,
        &[
            (1, f64b(1.5)),
            (2, f64b(1.0) | (f64b(4.0) << 64)),
            (0, u128::MAX),
        ],
    );
    assert_eq!(cpu.read_vreg(0), f64b(6.0), "fmul d0, d1, v2.d[1]");

    // fmla s0, s1, v2.s[3]: Vd is the accumulator, so 1 + 2*3.
    let cpu = simd1(
        0x5fa21820,
        &[(0, f32b(1.0)), (1, f32b(2.0)), (2, f32b(3.0) << 96)],
    );
    assert_eq!(cpu.read_vreg(0), f32b(7.0), "fmla s0, s1, v2.s[3]");

    // fmls d5, d6, v7.d[0]: 10 - 2*3.
    let cpu = simd1(
        0x5fc750c5,
        &[(5, f64b(10.0)), (6, f64b(2.0)), (7, f64b(3.0))],
    );
    assert_eq!(cpu.read_vreg(5), f64b(4.0), "fmls d5, d6, v7.d[0]");

    // fmulx s0, s1, v2.s[0]: an ordinary multiply...
    let cpu = simd1(0x7f829020, &[(1, f32b(2.0)), (2, f32b(3.0))]);
    assert_eq!(cpu.read_vreg(0), f32b(6.0), "fmulx s0, s1, v2.s[0]");
    // ...except zero times infinity is 2.0 rather than a NaN.
    let cpu = simd1(0x7f829020, &[(1, f32b(0.0)), (2, f32b(f32::INFINITY))]);
    assert_eq!(cpu.read_vreg(0), f32b(2.0), "fmulx 0 * inf");
    let cpu = simd1(0x7f829020, &[(1, f32b(-0.0)), (2, f32b(f32::INFINITY))]);
    assert_eq!(
        cpu.read_vreg(0),
        f32b(-2.0),
        "fmulx -0 * inf keeps the sign"
    );

    // sqdmulh h0, h1, v2.h[3]: the doubled product's high half. 2*0x4000*0x4000
    // is 0x2000_0000, and the top 16 bits of that are 0x2000.
    let cpu = simd1(0x5f72c020, &[(1, 0x4000), (2, 0x4000 << 48)]);
    assert_eq!(cpu.read_vreg(0), 0x2000, "sqdmulh h0, h1, v2.h[3]");
    // The one input pair that saturates: the most negative value squared.
    let cpu = simd1(0x5f72c020, &[(1, 0x8000), (2, 0x8000 << 48)]);
    assert_eq!(cpu.read_vreg(0), 0x7FFF, "sqdmulh saturates at -min * -min");

    // sqrdmulh s0, s1, v2.s[2]: the same, rounded rather than truncated.
    let cpu = simd1(0x5f82d820, &[(1, 1 << 30), (2, (1u128 << 30) << 64)]);
    assert_eq!(cpu.read_vreg(0), 1 << 29, "sqrdmulh s0, s1, v2.s[2]");

    // sqdmull s0, h1, v2.h[0]: doubled at twice the width, giving 0x2000_0000.
    let cpu = simd1(0x5f42b020, &[(1, 0x4000), (2, 0x4000)]);
    assert_eq!(cpu.read_vreg(0), 0x2000_0000, "sqdmull s0, h1, v2.h[0]");
    let cpu = simd1(0x5f42b020, &[(1, 0x8000), (2, 0x8000)]);
    assert_eq!(cpu.read_vreg(0), 0x7FFF_FFFF, "sqdmull saturates");

    // sqdmlal s0, h1, v2.h[5]: 100 + 2*3*4.
    let cpu = simd1(0x5f523820, &[(0, 100), (1, 3), (2, 4 << 80)]);
    assert_eq!(cpu.read_vreg(0), 124, "sqdmlal s0, h1, v2.h[5]");

    // sqdmlsl d0, s1, v2.s[1]: 1000 - 2*5*7.
    let cpu = simd1(0x5fa27020, &[(0, 1000), (1, 5), (2, 7 << 32)]);
    assert_eq!(cpu.read_vreg(0), 930, "sqdmlsl d0, s1, v2.s[1]");

    // sqrdmlah s0, s1, v2.s[1]: SQRDMULH accumulated into Vd.
    let cpu = simd1(
        0x7fa2d020,
        &[(0, 7), (1, 1 << 30), (2, (1u128 << 30) << 32)],
    );
    assert_eq!(cpu.read_vreg(0), (1 << 29) + 7, "sqrdmlah s0, s1, v2.s[1]");

    // sqrdmlsh h0, h1, v2.h[2]: and subtracted from it.
    let cpu = simd1(0x7f62f020, &[(0, 0x2007), (1, 0x4000), (2, 0x4000 << 32)]);
    assert_eq!(cpu.read_vreg(0), 7, "sqrdmlsh h0, h1, v2.h[2]");
}

#[test]
fn dup_element_to_a_scalar_takes_the_lane_and_zeroes_the_rest() {
    // The AdvSIMD scalar copy group: `DUP (element)` (alias `mov s1, v0.s[1]`).
    // Unlike vector DUP, the lane goes to the bottom and the rest is zeroed.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, 0x4444_4444_3333_3333_2222_2222_1111_1111u128);
    cpu.mem.map(0x1000, &0x5e0c0401u32.to_le_bytes()).unwrap(); // dup s1, v0.s[1]
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_vreg(1), 0x2222_2222);

    // Lane size from imm5's lowest set bit, index from the bits above:
    // `dup d1, v0.d[1]` = 0x5e180401.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, 0x4444_4444_3333_3333_2222_2222_1111_1111u128);
    cpu.mem.map(0x1000, &0x5e180401u32.to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_vreg(1), 0x4444_4444_3333_3333);

    // `dup b1, v0.b[3]` = 0x5e070401.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, 0x4444_4444_3333_3333_2222_2222_1111_1111u128);
    cpu.mem.map(0x1000, &0x5e070401u32.to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_vreg(1), 0x11);

    // The vector form of DUP shares the imm5 encoding and must still
    // replicate rather than zero: `dup v1.4s, v0.s[1]` = 0x4e0c0401.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, 0x4444_4444_3333_3333_2222_2222_1111_1111u128);
    cpu.mem.map(0x1000, &0x4e0c0401u32.to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(
        cpu.read_vreg(1),
        0x2222_2222_2222_2222_2222_2222_2222_2222u128
    );
}

/// The doubleword-only scalar integer compares and ADD/SUB; -1 against 1
/// separates signed from unsigned.
#[test]
fn simd_scalar_compares_and_add_sub_on_a_doubleword() {
    const ONES: u64 = u64::MAX;
    let code = [
        0x7ee4_8e64u32, // cmeq d4, d19, d4
        0x5ee2_8c20,    // cmtst d0, d1, d2
        0x5ee5_3483,    // cmgt d3, d4, d5
        0x7ee8_34e6,    // cmhi d6, d7, d8
        0x5eeb_3d49,    // cmge d9, d10, d11
        0x7eee_3dac,    // cmhs d12, d13, d14
        0x5ef1_860f,    // add d15, d16, d17
        0x7ef4_8672,    // sub d18, d19, d20
        nop(),
    ];
    let mut cpu = cpu_at(0x1000);
    let bytes: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
    cpu.mem.map(0x1000, &bytes).unwrap();
    let minus_one = u128::from(ONES) | (7u128 << 64);
    cpu.set_vreg(19, 5);
    cpu.set_vreg(4, 5 | (9u128 << 64));
    cpu.set_vreg(1, 0b1010);
    cpu.set_vreg(2, 0b0101);
    // cmgt's first operand is d4, which cmeq sets to all ones first.
    cpu.set_vreg(5, 1);
    for (a, b) in [(7, 8), (10, 11), (13, 14)] {
        cpu.set_vreg(a, minus_one);
        cpu.set_vreg(b, 1);
    }
    cpu.set_vreg(16, u128::from(ONES));
    cpu.set_vreg(17, 2);
    cpu.set_vreg(20, 6);
    cpu.run(8).unwrap();

    assert_eq!(
        cpu.read_vreg(4),
        u128::from(ONES),
        "cmeq: equal, and the upper half cleared"
    );
    assert_eq!(cpu.read_vreg(0), 0, "cmtst: no bit in common");
    assert_eq!(cpu.read_vreg(3), 0, "cmgt: -1 is not greater than 1");
    assert_eq!(cpu.read_vreg(6), u128::from(ONES), "cmhi: 2^64-1 is");
    assert_eq!(cpu.read_vreg(9), 0, "cmge: signed");
    assert_eq!(cpu.read_vreg(12), u128::from(ONES), "cmhs: unsigned");
    assert_eq!(cpu.read_vreg(15), 1, "add wraps");
    assert_eq!(cpu.read_vreg(18), u128::from(ONES), "sub: 5 - 6 wraps");
}
