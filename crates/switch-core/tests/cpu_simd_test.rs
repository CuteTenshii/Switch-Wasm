//! Advanced SIMD, scalar floating point, and the crypto extension.

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
    // dup v0.16b, w1 (0x3d) ; dup v1.16b, w2 (0x3d) ; sub v2.4s, v0.4s, v1.4s
    // → lanes of 0x3d3d3d3d - 0x3d3d3d3d = 0 ; mov x3, v2.d[0]
    let code = [
        dup16(0, 1),
        dup16(1, 2),
        sub4s(2, 0, 1),
        umov_d0(3, 2),
        nop(),
    ];
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, 0x3d);
    cpu.set_reg(2, 0x3d);
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_x(3), 0);

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
fn simd_pairwise_addp() {
    // v1 = {1,1,...}, v2 = {2,2,...}; addp v3.16b, v1.16b, v2.16b →
    // v3[0..7] = v1 pairwise (2), v3[8..15] = v2 pairwise (4).
    let code = [
        dup16(1, 1),
        dup16(2, 2),
        addp16(3, 1, 2),
        umov_d0(4, 3),
        nop(),
    ];
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
    // v2: the low half reverses the first eight table bytes, the high half
    // is 0x40: past the end of any table this test builds.
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

#[test]
fn simd_ins_element_moves_one_lane_and_leaves_the_rest() {
    // `INS <Vd>.<Ts>[<i1>], <Vn>.<Ts>[<i2>]` is the `op == 1` half of the
    // AdvSIMD copy group. The group was matched on bits[29:21], which pins
    // `op` to 0, so every one of these fell through to the three-same integer
    // decoder and executed as an unrelated arithmetic op, silently
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

    // A non-zero source lane, and lanes wider than a byte. INS touches only
    // the destination lane: the top half of Vd is left alone, unlike almost
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
    // hbmenu / NX-Shell both faulted on `ldr b29, [x0, #0x280]` = 0x3d4a001d
    // (SIMD scalar 8-bit load). Also covers `stur q17, [x0, #0x8]` = 0x3c808011
    // (SIMD scalar STUR, unscaled offset) which the same libnx init loop uses.
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
    assert_eq!(
        cpu.mem
            .read_into(0x3008, &mut [0u8; 16])
            .map(|_| u128::from_le_bytes(cpu.mem.dump(0x3008, 16).unwrap().try_into().unwrap()))
            .unwrap(),
        0x1122_3344_5566_7788_99AA_BBCC_DDEE_FF00
    );
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

#[test]
fn ld1_multiple_structures_writes_back_only_when_post_indexed() {
    // `ld1 {v1.16b, v2.16b}, [x2], #32` = 0x4cdfa041. The immediate post-index
    // form has Rm == 31, which the old decode read as "no writeback", newlib's
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
fn scalar_fp_one_source_and_fused_multiply_add() {
    // `fmov s0, s15` = 0x1e2041e0 is opcode 0 of the 1-source group, whose low
    // opcode bit sits in bits[15], matching bits[15:10] as a unit missed the
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
    assert_eq!(
        lanes_f32(cpu.read_vreg(2)),
        [1.0, 2.0, 3.0, 4_294_967_295.0]
    );

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
    // `fdiv v28.4s, v28.4s, v30.4s` = 0x6e3eff9c, the FP three-same group
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
fn scalar_integer_float_conversions_and_rounding_modes() {
    // `ucvtf d0, x1` = 0x9e630020 is where NX-Shell died: rmode/opcode were
    // read as one 6-bit field including the fixed bit 21, so this decoded as
    // FCVTMU and wrote x0: clobbering a live pointer.
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
    assert_eq!(
        f64::from_bits(cpu.read_vreg(0) as u64),
        18_446_744_073_709_551_615.0
    );
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
    assert_eq!(
        f32::from_bits(cpu.read_vreg(30) as u32),
        2.5,
        "GT false → Vm"
    );

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
    // `ucvtf s13, s13` = 0x7e21d9ad, the scalar form of the two-register misc
    // group (bits[31:30] = 01, bits[28:24] = 11110). Only the vector encodings
    // were decoded, so NX-Shell faulted here.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(13, 0xFFFF_FFFF_FFFF_FFFF_FFFF_FFFF_0000_0007);
    let cpu = run_program(cpu, 0x1000, &[0x7e21_d9ad, nop()]);
    // One lane converted, everything above it zeroed.
    assert_eq!(cpu.read_vreg(13), u128::from((7.0f32).to_bits()));
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

    // The vector (three-different) forms: smull, smlal, saddl, addhn.
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
    // `01 U 11111 size L M Rm opcode H 0 Rn Rd`: one lane of Vn times one
    // selected lane of Vm, written as a *scalar* -- bottom element, everything
    // above it zeroed. It differs from the vector-by-element form only in
    // bits[28:24], 11111 against 01111, so the whole group was falling through
    // that check. "A Short Hike" reaches `fmul s3, s4, v3.s[0]` two billion
    // instructions in, having got that far only once every earlier fix landed.
    //
    // Every encoding below is what LLVM assembles the named instruction to,
    // not one derived by hand from the manual.

    // fmul s3, s4, v3.s[0] -- Rd and Rm are the same register here, which is
    // the real instruction from the title, so it also pins that reading Vm
    // happens before writing Vd.
    let cpu = simd1(
        0x5f839083,
        &[(4, f32b(3.0)), (3, f32b(2.0) | (f32b(9.0) << 32))],
    );
    assert_eq!(cpu.read_vreg(3), f32b(6.0), "fmul s3, s4, v3.s[0]");

    // fmul d0, d1, v2.d[1] -- the index picks the *high* half of Vm, and the
    // 64-bit result must not leave the old top half of Vd behind.
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
    // ...except that zero times infinity is 2.0 rather than a NaN, which is
    // the only reason the instruction exists apart from FMUL.
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

    // sqdmull s0, h1, v2.h[0]: doubled, kept at twice the width rather than
    // shifted back down -- so the same inputs give the whole 0x2000_0000.
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
    // The AdvSIMD *scalar* copy group holds exactly one instruction: `DUP
    // (element)`, which lifts one lane of a vector into a scalar register.
    // `mov s1, v0.s[1]` is an alias for it, and it is what a vectorised hash
    // reaches for to fold its accumulator lanes together -- "A Short Hike"
    // stops on one 1.29 billion instructions in.
    //
    // It differs from the vector copy group only in bits[28:21], so it was
    // being rejected along with everything else that is not 0111 0000. Unlike
    // the vector DUP the lane is *not* replicated: it goes at the bottom and
    // the rest of the register is zeroed.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, 0x4444_4444_3333_3333_2222_2222_1111_1111u128);
    cpu.mem.map(0x1000, &0x5e0c0401u32.to_le_bytes()).unwrap(); // dup s1, v0.s[1]
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_vreg(1), 0x2222_2222);

    // The lane size comes from the lowest set bit of imm5, the index from the
    // bits above it: `dup d1, v0.d[1]` = 0x5e180401.
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

#[test]
fn crc32_accumulates_over_the_bytes_of_its_operand() {
    // The check value of "123456789" for both polynomials, fed in as one
    // doubleword plus a trailing byte. The instructions accumulate without
    // the final inversion, so these are the complements of the usual
    // 0xCBF43926 and 0xE3069283.
    let mut cpu = Cpu::new();
    cpu.set_reg(0, u64::from_le_bytes(*b"12345678"));
    cpu.set_reg(1, 0xFFFF_FFFF);
    cpu.set_reg(3, u64::from(b'9'));
    cpu.set_reg(6, 0x3231);
    cpu.set_reg(8, 0x3433_3231);
    let cpu = run_program(
        cpu,
        0x1000,
        &[
            crc32(2, 1, 0, false, 0b11),
            crc32(2, 2, 3, false, 0b00),
            crc32(4, 1, 0, true, 0b11),
            crc32(4, 4, 3, true, 0b00),
            crc32(5, 31, 6, false, 0b01),
            crc32(7, 31, 8, false, 0b10),
        ],
    );
    assert_eq!(cpu.read_x(2), 0x340B_C6D9);
    assert_eq!(cpu.read_x(4), 0x1CF9_6D7C);
    assert_eq!(cpu.read_x(5), 0x0E8A_5632);
    assert_eq!(cpu.read_x(7), 0xBAA7_3FBF);
}

/// AESE then AESMC is one AES round bar the key schedule, so FIPS-197's own
/// round-1 vector pins both instructions and the decoder that reaches them.
#[test]
fn the_aes_instructions_run_a_fips_197_round() {
    let mut cpu = cpu_at(0x1000);
    // The round input, and a zero key so AESE's XOR leaves it alone.
    cpu.set_vreg(
        0,
        u128::from_le_bytes([
            0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0,
            0xe0, 0xf0,
        ]),
    );
    cpu.set_vreg(1, 0);
    let cpu = run_program(cpu, 0x1000, &[aes(0b00100, 0, 1), aes(0b00110, 0, 0)]);
    assert_eq!(
        cpu.read_vreg(0),
        u128::from_le_bytes([
            0x5f, 0x72, 0x64, 0x15, 0x57, 0xf5, 0xbc, 0x92, 0xf7, 0xbe, 0x3b, 0x29, 0x1d, 0xb9,
            0xf9, 0x1a,
        ]),
        "aese/aesmc did not produce the FIPS-197 round-1 state"
    );
}

#[test]
fn aesd_and_aesimc_invert_the_encrypting_pair() {
    let state = 0x0f0e_0d0c_0b0a_0908_0706_0504_0302_0100u128;
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, state);
    cpu.set_vreg(1, 0);
    let cpu = run_program(
        cpu,
        0x1000,
        &[
            aes(0b00100, 0, 1), // AESE v0, v1  (SubBytes/ShiftRows)
            aes(0b00110, 0, 0), // AESMC v0, v0
            aes(0b00111, 0, 0), // AESIMC v0, v0
            aes(0b00101, 0, 1), // AESD v0, v1
        ],
    );
    assert_eq!(
        cpu.read_vreg(0),
        state,
        "the decrypting pair did not invert"
    );
}

/// SHA1H is a bare rotate, and SHA1SU0's three-way XOR is the schedule step,
/// both cheap enough to state the expected value outright.
#[test]
fn the_sha1_instructions_decode_and_compute() {
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, 0x1234_5678);
    let cpu = run_program(cpu, 0x1000, &[sha2(0b00000, 1, 0)]);
    assert_eq!(
        cpu.read_vreg(1),
        u128::from(0x1234_5678u32.rotate_left(30)),
        "sha1h is a 30-bit rotate of the low word into a cleared register"
    );

    let (d, n, m) = (0x1111u128, 0x2222u128 << 64, 0x3333u128);
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, d);
    cpu.set_vreg(1, n);
    cpu.set_vreg(2, m);
    let cpu = run_program(cpu, 0x1000, &[sha3(0b011, 0, 1, 2)]);
    assert_eq!(cpu.read_vreg(0), ((d >> 64) | (n << 64)) ^ d ^ m);
}

/// SHA256H and SHA256H2 keep opposite halves of the same four rounds, so a
/// state where both are driven from the same inputs must not come back equal.
#[test]
fn the_sha256_round_instructions_keep_opposite_halves() {
    let (x, y, w) = (
        0x0000_0004_0000_0003_0000_0002_0000_0001u128,
        0x0000_0008_0000_0007_0000_0006_0000_0005u128,
        0x0000_000c_0000_000b_0000_000a_0000_0009u128,
    );
    let mut cpu = cpu_at(0x1000);
    for (i, v) in [(0u8, x), (1, y), (2, w)] {
        cpu.set_vreg(i, v);
    }
    cpu.set_vreg(3, y);
    cpu.set_vreg(4, x);
    let cpu = run_program(
        cpu,
        0x1000,
        &[
            sha3(0b100, 0, 1, 2), // SHA256H  q0, q1, v2.4s
            sha3(0b101, 3, 4, 2), // SHA256H2 q3, q4, v2.4s
        ],
    );
    let (part1, part2) = (cpu.read_vreg(0), cpu.read_vreg(3));
    assert_ne!(part1, x, "sha256h did not advance the state");
    assert_ne!(part2, y, "sha256h2 did not advance the state");
    assert_ne!(part1, part2, "the two halves came back identical");
}

#[test]
fn pmull_multiplies_without_carrying() {
    let mut cpu = cpu_at(0x1000);
    // 8-bit lanes: 0b11 * 0b11 is 0b101, not 0b1001.
    cpu.set_vreg(0, 0x03);
    cpu.set_vreg(1, 0x03);
    let cpu = run_program(cpu, 0x1000, &[pmull(0, 0b00, 2, 0, 1)]);
    assert_eq!(cpu.read_vreg(2) & 0xFFFF, 0b101);

    // 64-bit lanes, and PMULL2 reads the top half of each source.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, (1u128 << 63) << 64);
    cpu.set_vreg(1, (1u128 << 63) << 64);
    let cpu = run_program(cpu, 0x1000, &[pmull(1, 0b11, 2, 0, 1)]);
    assert_eq!(cpu.read_vreg(2), 1u128 << 126);
}

/// `fcvt` to and from a half is ARMv8.0 baseline, the half-precision
/// *arithmetic* the A57 lacks is a separate thing.
#[test]
fn fcvt_converts_to_and_from_half_precision() {
    // 1.0 as a half is 0x3C00; as a single 0x3F800000, as a double 0x3FF0...
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, 0x3C00);
    let cpu = run_program(
        cpu,
        0x1000,
        &[
            fcvt(0b11, 0b00, 1, 0), // FCVT s1, h0
            fcvt(0b11, 0b01, 2, 0), // FCVT d2, h0
        ],
    );
    assert_eq!(cpu.read_vreg(1), u128::from(1.0f32.to_bits()));
    assert_eq!(cpu.read_vreg(2), u128::from(1.0f64.to_bits()));

    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, u128::from((-2.5f32).to_bits()));
    cpu.set_vreg(1, u128::from(65504.0f64.to_bits())); // the largest half
    let cpu = run_program(
        cpu,
        0x1000,
        &[
            fcvt(0b00, 0b11, 2, 0), // FCVT h2, s0
            fcvt(0b01, 0b11, 3, 1), // FCVT h3, d1
        ],
    );
    assert_eq!(cpu.read_vreg(2), 0xC100, "-2.5 is not the expected half");
    assert_eq!(
        cpu.read_vreg(3),
        0x7BFF,
        "65504 should be the largest finite half"
    );
}

#[test]
fn narrowing_to_half_saturates_rounds_and_flushes_at_the_edges() {
    let cases: [(f64, u16); 7] = [
        (0.0, 0x0000),
        (-0.0, 0x8000),
        (70000.0, 0x7C00),          // beyond the range: infinity
        (65520.0, 0x7C00),          // rounds up past the largest finite half
        (65519.0, 0x7BFF),          // still rounds down to it
        (6.0e-8, 0x0001),           // above half the smallest subnormal
        (2.0f64.powi(-25), 0x0000), // exactly a tie, so down to zero
    ];
    for (input, expect) in cases {
        let mut cpu = cpu_at(0x1000);
        cpu.set_vreg(0, u128::from(input.to_bits()));
        let cpu = run_program(cpu, 0x1000, &[fcvt(0b01, 0b11, 1, 0)]);
        assert_eq!(
            cpu.read_vreg(1) as u16,
            expect,
            "fcvt h, d of {input} gave {:#06x}",
            cpu.read_vreg(1) as u16
        );
    }
}

#[test]
fn widening_from_half_handles_subnormals_and_infinities() {
    let cases: [(u16, f32); 5] = [
        (0x0001, 5.960_464_5e-8), // the smallest subnormal, 2^-24
        (0x03FF, 6.097_555e-5),   // the largest subnormal
        (0x0400, 6.103_515_6e-5), // the smallest normal, 2^-14
        (0x7C00, f32::INFINITY),
        (0xFC00, f32::NEG_INFINITY),
    ];
    for (input, expect) in cases {
        let mut cpu = cpu_at(0x1000);
        cpu.set_vreg(0, u128::from(input));
        let cpu = run_program(cpu, 0x1000, &[fcvt(0b11, 0b00, 1, 0)]);
        let got = f32::from_bits(cpu.read_vreg(1) as u32);
        assert_eq!(got, expect, "fcvt s, h of {input:#06x} gave {got}");
    }
}

/// Every half a single can represent exactly must survive the round trip.
#[test]
fn every_half_survives_a_round_trip_through_single() {
    let code = [fcvt(0b11, 0b00, 1, 0), fcvt(0b00, 0b11, 2, 1)];
    let mut cpu = cpu_at(0x1000);
    let mut bytes = Vec::new();
    for insn in &code {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(0x1000, &bytes).unwrap();
    for raw in 0u32..=0xFFFF {
        let half = raw as u16;
        if (half >> 10) & 0x1F == 0x1F {
            continue; // infinities and NaNs have no single canonical form
        }
        cpu.set_vreg(0, u128::from(half));
        cpu.set_pc(0x1000);
        cpu.run(code.len() as u64).unwrap();
        assert_eq!(
            cpu.read_vreg(2) as u16,
            half,
            "half {half:#06x} did not survive the round trip"
        );
    }
}

/// FCVTL/FCVTN move a whole vector of halves, and are the form the vectorised
/// half-float packing in shader and texture code actually uses.
#[test]
fn the_vector_half_conversions_move_four_lanes() {
    // FCVTL v1.4s, v0.4h : 0 Q 0 01110 size 10000 10111 10 Rn Rd, size = 00
    let fcvtl = |q: u32, rd: u32, rn: u32| {
        (q << 30) | 0b01110 << 24 | 0b10000 << 17 | 0b10111 << 12 | 0b10 << 10 | (rn << 5) | rd
    };
    let fcvtn = |q: u32, rd: u32, rn: u32| {
        (q << 30) | 0b01110 << 24 | 0b10000 << 17 | 0b10110 << 12 | 0b10 << 10 | (rn << 5) | rd
    };
    // 1.0, 2.0, -1.0, 0.5 as halves.
    let halves: u64 = 0x3800_BC00_4000_3C00;
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, u128::from(halves));
    let cpu = run_program(cpu, 0x1000, &[fcvtl(0, 1, 0), fcvtn(0, 2, 1)]);
    let widened = cpu.read_vreg(1);
    for (i, expect) in [1.0f32, 2.0, -1.0, 0.5].into_iter().enumerate() {
        let lane = f32::from_bits((widened >> (32 * i)) as u32);
        assert_eq!(lane, expect, "fcvtl lane {i}");
    }
    assert_eq!(
        cpu.read_vreg(2),
        u128::from(halves),
        "fcvtn did not narrow back to the halves it started from"
    );
}

/// The shift amount is the low **byte** of the lane sign-extended, so a
/// negative one shifts right. Reading the whole lane instead made that
/// impossible below 64 bits: `sshl` could only ever shift left.
#[test]
fn sshl_shifts_right_on_a_negative_amount_in_every_lane_width() {
    for (size, esize) in [(0u32, 8u32), (1, 16), (2, 32), (3, 64)] {
        let lane = |v: u64| {
            let mut out = 0u128;
            for i in 0..(128 / esize) {
                out |= u128::from(v & (u64::MAX >> (64 - esize))) << (esize * i);
            }
            out
        };
        let mut cpu = cpu_at(0x1000);
        cpu.set_vreg(0, lane(0x40));
        cpu.set_vreg(1, lane(0xFE)); // -2 as a signed byte
        let cpu = run_program(cpu, 0x1000, &[simd_shift_reg(1, 0, size, SSHL, 2, 0, 1)]);
        assert_eq!(
            cpu.read_vreg(2),
            lane(0x10),
            "sshl by -2 with {esize}-bit lanes did not divide by four"
        );
    }
}

#[test]
fn ushl_shifts_in_zeros_and_sshl_shifts_in_the_sign() {
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, 0xF000_0000u32 as u128); // one 32-bit lane, negative
    cpu.set_vreg(1, 0xFC); // -4
    let cpu = run_program(
        cpu,
        0x1000,
        &[
            simd_shift_reg(0, 0, 0b10, SSHL, 2, 0, 1),
            simd_shift_reg(0, 1, 0b10, SSHL, 3, 0, 1),
        ],
    );
    assert_eq!(cpu.read_vreg(2) as u32, 0xFF00_0000, "sshl is arithmetic");
    assert_eq!(cpu.read_vreg(3) as u32, 0x0F00_0000, "ushl is logical");
}

#[test]
fn the_saturating_shift_clamps_instead_of_dropping_bits() {
    // Signed 32-bit: 0x40000000 << 2 overflows to INT_MAX rather than wrapping.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, 0x4000_0000);
    cpu.set_vreg(1, 2);
    let cpu = run_program(cpu, 0x1000, &[simd_shift_reg(0, 0, 0b10, SQSHL, 2, 0, 1)]);
    assert_eq!(
        cpu.read_vreg(2) as u32,
        0x7FFF_FFFF,
        "sqshl did not saturate high"
    );

    // And the negative direction saturates to INT_MIN.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, 0xC000_0000);
    cpu.set_vreg(1, 2);
    let cpu = run_program(cpu, 0x1000, &[simd_shift_reg(0, 0, 0b10, SQSHL, 2, 0, 1)]);
    assert_eq!(
        cpu.read_vreg(2) as u32,
        0x8000_0000,
        "sqshl did not saturate low"
    );

    // Unsigned saturates to UINT_MAX.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, 0x4000_0000);
    cpu.set_vreg(1, 2);
    let cpu = run_program(cpu, 0x1000, &[simd_shift_reg(0, 1, 0b10, SQSHL, 2, 0, 1)]);
    assert_eq!(
        cpu.read_vreg(2) as u32,
        0xFFFF_FFFF,
        "uqshl did not saturate"
    );
}

#[test]
fn the_rounding_shift_rounds_the_bits_it_drops() {
    // 0b110 >> 1 is 3 exactly; 0b111 >> 1 is 3.5, which rounds away to 4.
    for (input, expect) in [(0b110u32, 3u32), (0b111, 4), (0b101, 3), (0b100, 2)] {
        let mut cpu = cpu_at(0x1000);
        cpu.set_vreg(0, u128::from(input));
        cpu.set_vreg(1, 0xFF); // -1
        let cpu = run_program(cpu, 0x1000, &[simd_shift_reg(0, 0, 0b10, SRSHL, 2, 0, 1)]);
        assert_eq!(cpu.read_vreg(2) as u32, expect, "srshl of {input:#b}");
        // The plain shift truncates instead.
        let mut cpu = cpu_at(0x1000);
        cpu.set_vreg(0, u128::from(input));
        cpu.set_vreg(1, 0xFF);
        let cpu = run_program(cpu, 0x1000, &[simd_shift_reg(0, 0, 0b10, SSHL, 2, 0, 1)]);
        assert_eq!(
            cpu.read_vreg(2) as u32,
            input >> 1,
            "sshl of {input:#b} truncates"
        );
    }
}

#[test]
fn a_shift_past_the_lane_width_empties_or_saturates_it() {
    // Left past the width: everything is gone.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, 0x1234);
    cpu.set_vreg(1, 40);
    let cpu = run_program(cpu, 0x1000, &[simd_shift_reg(0, 0, 0b10, SSHL, 2, 0, 1)]);
    assert_eq!(cpu.read_vreg(2) as u32, 0);

    // The saturating form clamps rather than emptying.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, 0x1234);
    cpu.set_vreg(1, 40);
    let cpu = run_program(cpu, 0x1000, &[simd_shift_reg(0, 0, 0b10, SQSHL, 2, 0, 1)]);
    assert_eq!(cpu.read_vreg(2) as u32, 0x7FFF_FFFF);

    // Right past the width: zero unsigned, the sign signed.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, 0xFFFF_FFFF);
    cpu.set_vreg(1, 0x80); // -128
    let cpu = run_program(
        cpu,
        0x1000,
        &[
            simd_shift_reg(0, 0, 0b10, SSHL, 2, 0, 1),
            simd_shift_reg(0, 1, 0b10, SSHL, 3, 0, 1),
        ],
    );
    assert_eq!(
        cpu.read_vreg(2) as u32,
        0xFFFF_FFFF,
        "the sign fills a signed lane"
    );
    assert_eq!(cpu.read_vreg(3) as u32, 0, "zero fills an unsigned one");
}

#[test]
fn the_scalar_variable_shifts_decode_and_clear_the_rest_of_the_register() {
    // SSHL/SRSHL are doubleword-only; the saturating pair carries a size.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, u128::MAX);
    cpu.set_vreg(1, 0xF8); // -8
    let cpu = run_program(cpu, 0x1000, &[scalar_shift_reg(1, 0b11, SSHL, 2, 0, 1)]);
    assert_eq!(
        cpu.read_vreg(2),
        u128::from(u64::MAX >> 8),
        "ushl d2 should shift one 64-bit lane and clear the top half"
    );

    // SQSHL at 16-bit scalar width saturates to a halfword.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, 0x4000);
    cpu.set_vreg(1, 4);
    let cpu = run_program(cpu, 0x1000, &[scalar_shift_reg(0, 0b01, SQSHL, 2, 0, 1)]);
    assert_eq!(cpu.read_vreg(2), 0x7FFF);

    // SQRSHL rounds and saturates together.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, 0b111);
    cpu.set_vreg(1, 0xFF); // -1
    let cpu = run_program(cpu, 0x1000, &[scalar_shift_reg(0, 0b11, SQRSHL, 2, 0, 1)]);
    assert_eq!(cpu.read_vreg(2), 4);
}

/// `fegetround`/`fesetround` are an MRS/MSR pair on FPCR, so the register has
/// to be real storage before either can mean anything.
#[test]
fn fpcr_and_fpsr_round_trip_through_mrs_and_msr() {
    let (a, b, c, d) = FPCR_REG;
    let cpu = run_program(
        cpu_at(0x1000),
        0x1000,
        &[
            movz(0, 0x00C0, 1, true),
            msr(0, a, b, c, d),
            mrs(1, a, b, c, d),
        ],
    );
    assert_eq!(
        cpu.read_x(1),
        0x00C0_0000,
        "fpcr did not read back what was written"
    );

    let (a, b, c, d) = FPSR_REG;
    let cpu = run_program(
        cpu_at(0x1000),
        0x1000,
        &[
            movz(0, 0x001F, 0, true),
            msr(0, a, b, c, d),
            mrs(1, a, b, c, d),
        ],
    );
    assert_eq!(
        cpu.read_x(1),
        0x1F,
        "fpsr did not read back the exception flags"
    );
}

/// FRINTX and FRINTI are the two that round to whatever mode FPCR names, so
/// changing the mode has to change their answer.
#[test]
fn frinti_follows_the_rounding_mode_in_fpcr() {
    let (a, b, c, d) = FPCR_REG;
    let frinti = |rd: u32, rn: u32| {
        0x1E << 24 | 1 << 22 | 1 << 21 | 0b001111 << 15 | 0b10000 << 10 | (rn << 5) | rd
    };
    for (rmode, input, expect) in [
        (0b00u32, 2.5f64, 2.0f64), // nearest, ties to even
        (0b00, 3.5, 4.0),
        (0b01, 2.1, 3.0), // toward +inf
        (0b10, 2.9, 2.0), // toward -inf
        (0b10, -2.1, -3.0),
        (0b11, 2.9, 2.0), // toward zero
        (0b11, -2.9, -2.0),
    ] {
        let mut cpu = cpu_at(0x1000);
        cpu.set_vreg(5, u128::from(input.to_bits()));
        let cpu = run_program(
            cpu,
            0x1000,
            &[
                movz(0, rmode << 6, 1, true),
                msr(0, a, b, c, d),
                frinti(6, 5),
            ],
        );
        let got = f64::from_bits(cpu.read_vreg(6) as u64);
        assert_eq!(got, expect, "frinti of {input} in mode {rmode:#b}");
    }
}

#[test]
fn dividing_by_zero_raises_the_divide_by_zero_flag() {
    let (a, b, c, d) = FPSR_REG;
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, u128::from(1.0f64.to_bits()));
    cpu.set_vreg(1, 0);
    let cpu = run_program(cpu, 0x1000, &[fdiv_d(2, 0, 1), mrs(3, a, b, c, d)]);
    assert_eq!(
        cpu.read_x(3) & 0b10,
        0b10,
        "DZC was not raised by 1.0 / 0.0"
    );
    assert_eq!(cpu.read_x(3) & 1, 0, "and 1.0 / 0.0 is not Invalid");

    // 0/0 has no answer at all, which is Invalid rather than divide-by-zero.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, 0);
    cpu.set_vreg(1, 0);
    let cpu = run_program(cpu, 0x1000, &[fdiv_d(2, 0, 1), mrs(3, a, b, c, d)]);
    assert_eq!(cpu.read_x(3) & 1, 1, "IOC was not raised by 0.0 / 0.0");
    assert_eq!(cpu.read_x(3) & 0b10, 0, "and it is not divide-by-zero");
}

#[test]
fn a_convert_that_cannot_fit_or_loses_a_fraction_says_so() {
    let (a, b, c, d) = FPSR_REG;
    // FCVTZS Wd, Dn
    let fcvtzs = |rd: u32, rn: u32| 0x1E << 24 | 1 << 22 | 1 << 21 | 0b11 << 19 | (rn << 5) | rd;

    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, u128::from(4.0f64.to_bits()));
    let cpu = run_program(cpu, 0x1000, &[fcvtzs(1, 0), mrs(2, a, b, c, d)]);
    assert_eq!(cpu.read_x(1), 4);
    assert_eq!(cpu.read_x(2) & 0x1F, 0, "an exact convert raised a flag");

    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, u128::from(4.5f64.to_bits()));
    let cpu = run_program(cpu, 0x1000, &[fcvtzs(1, 0), mrs(2, a, b, c, d)]);
    assert_eq!(cpu.read_x(1), 4);
    assert_eq!(
        cpu.read_x(2) & 0b10000,
        0b10000,
        "IXC was not raised by 4.5"
    );

    for input in [f64::NAN, 1.0e30] {
        let mut cpu = cpu_at(0x1000);
        cpu.set_vreg(0, u128::from(input.to_bits()));
        let cpu = run_program(cpu, 0x1000, &[fcvtzs(1, 0), mrs(2, a, b, c, d)]);
        assert_eq!(
            cpu.read_x(2) & 1,
            1,
            "IOC was not raised converting {input}"
        );
    }
}

/// The flags are sticky: only a write clears them.
#[test]
fn the_exception_flags_are_sticky_until_written() {
    let (a, b, c, d) = FPSR_REG;
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, u128::from(1.0f64.to_bits()));
    cpu.set_vreg(1, 0);
    cpu.set_vreg(3, u128::from(6.0f64.to_bits()));
    cpu.set_vreg(4, u128::from(2.0f64.to_bits()));
    let cpu = run_program(
        cpu,
        0x1000,
        &[
            fdiv_d(2, 0, 1), // raises DZC
            fdiv_d(5, 3, 4), // a clean divide must not clear it
            mrs(6, a, b, c, d),
            movz(7, 0, 0, true),
            msr(7, a, b, c, d),
            mrs(8, a, b, c, d),
        ],
    );
    assert_eq!(
        cpu.read_x(6) & 0b10,
        0b10,
        "a later clean divide cleared DZC"
    );
    assert_eq!(cpu.read_x(8), 0, "writing FPSR did not clear the flags");
}
