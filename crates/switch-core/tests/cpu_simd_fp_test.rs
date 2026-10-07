//! Scalar and vector floating point: arithmetic, conversions, rounding and FPCR/FPSR.

mod cpu;

use cpu::*;

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
    // `scvtf s0, w1` = 0x1e220020 then `fcvtzs w2, s0` = 0x1e380002. Bit 21 is
    // a fixed 1; rmode is bits[20:19], opcode bits[18:16].
    let code = [0x1e22_0020u32, 0x1e38_0002, nop()];
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, 1000);
    let cpu = run_program(cpu, 0x1000, &code);
    assert_eq!(cpu.read_x(2), 1000);
}

#[test]
fn scalar_fp_one_source_and_fused_multiply_add() {
    // `fmov s0, s15` = 0x1e2041e0, opcode 0 of the 1-source group (low opcode
    // bit in bit 15). FMOV is bit-exact, so a signalling NaN survives.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(15, 0x1111_1111_1111_1111_1111_1111_7FA0_1234);
    let cpu = run_program(cpu, 0x1000, &[0x1e20_41e0, nop()]);
    assert_eq!(cpu.read_vreg(0), 0x7FA0_1234);

    // `fmov d1, d2` = 0x1e604041 keeps 64 bits and zeroes the rest.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(2, 0x9999_9999_9999_9999_4008_0000_0000_0000);
    let cpu = run_program(cpu, 0x1000, &[0x1e60_4041, nop()]);
    assert_eq!(cpu.read_vreg(1), 0x4008_0000_0000_0000);

    // FMADD/FMSUB/FNMADD/FNMSUB (`fmadd d3, d4, d5, d6` = 0x1f451883 and friends).
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
    // `scvtf v28.4s, v31.4s` = 0x4e21dbfc, the two-register misc group.
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
    // `fdiv v28.4s, v28.4s, v30.4s` = 0x6e3eff9c, FP three-same (opcodes from 0b11000).
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
fn scalar_integer_float_conversions_and_rounding_modes() {
    // `ucvtf d0, x1` = 0x9e630020: rmode/opcode exclude the fixed bit 21.
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
    // `fcmp d8, #0.0` = 0x1e602108: the zero flag is bit 3 of opcode2 (bits[4:0]).
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
fn fcsel_fccmp_and_fixed_point_conversions() {
    // `fcsel s29, s31, s30, gt` = 0x1e3ecffd. FCSEL and FCCMP have bit 21 set.
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(31, (1.5f32).to_bits() as u128);
    cpu.set_vreg(30, (2.5f32).to_bits() as u128);
    cpu.set_vreg(29, u128::MAX);
    // Set flags with `fcmp s31, s30` (1.5 < 2.5 → not GT) then select.
    let cpu = run_program(cpu, 0x1000, &[0x1e3e_23e0, 0x1e3e_cffd, nop()]);
    assert_eq!(
        cpu.read_vreg(29),
        u128::from((2.5f32).to_bits()),
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

    // Fixed-point conversions (bit 21 clear): `scvtf s0, w1, #8` = 0x1e02e020
    // scales by 2^-8, `fcvtzs w2, s0, #4` = 0x1e18f002 by 2^4.
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
    // `ucvtf s13, s13` = 0x7e21d9ad, the scalar two-register misc group
    // (bits[31:30] = 01, bits[28:24] = 11110).
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(13, 0xFFFF_FFFF_FFFF_FFFF_FFFF_FFFF_0000_0007);
    let cpu = run_program(cpu, 0x1000, &[0x7e21_d9ad, nop()]);
    // One lane converted, everything above it zeroed.
    assert_eq!(cpu.read_vreg(13), u128::from((7.0f32).to_bits()));
}

/// `fcvt` to and from a half is ARMv8.0 baseline, unlike half-precision arithmetic.
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

/// FCVTL/FCVTN on a whole vector of halves.
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

/// `fegetround`/`fesetround` are MRS/MSR on FPCR, so it must be real storage.
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

/// FRINTX and FRINTI round in FPCR's mode.
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

/// Fixed-point conversions: shift-by-immediate encodings counting fraction bits.
#[test]
fn simd_fixed_point_converts_in_both_directions() {
    let f32s = |lanes: [f32; 4]| -> u128 {
        lanes
            .iter()
            .enumerate()
            .map(|(i, f)| u128::from(f.to_bits()) << (32 * i))
            .sum()
    };
    let words = |lanes: [u32; 4]| -> u128 {
        lanes
            .iter()
            .enumerate()
            .map(|(i, &w)| u128::from(w) << (32 * i))
            .sum()
    };
    let f64s = |lanes: [f64; 2]| -> u128 {
        u128::from(lanes[0].to_bits()) | (u128::from(lanes[1].to_bits()) << 64)
    };
    let code = [
        0x4f31_fc00u32, // fcvtzs v0.4s, v0.4s, #15
        0x6f78_fc41,    // fcvtzu v1.2d, v2.2d, #8
        0x4f31_e483,    // scvtf v3.4s, v4.4s, #15
        0x2f3f_e4c5,    // ucvtf v5.2s, v6.2s, #1
        0x5f31_fd07,    // fcvtzs s7, s8, #15
        0x5f7d_e549,    // scvtf d9, d10, #3
        nop(),
    ];
    let mut cpu = cpu_at(0x1000);
    let bytes: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_vreg(0, f32s([0.5, -1.0, 1e10, f32::NAN]));
    cpu.set_vreg(2, f64s([-3.0, 2.75]));
    cpu.set_vreg(4, words([0x4000, 0xFFFF_8000, 1, 0]));
    cpu.set_vreg(5, u128::MAX);
    cpu.set_vreg(6, words([3, 0xFFFF_FFFF, 7, 7]));
    cpu.set_vreg(8, f32s([0.25, 7.0, 7.0, 7.0]));
    cpu.set_vreg(10, u128::from((-8i64) as u64));
    cpu.run(6).unwrap();

    assert_eq!(
        cpu.read_vreg(0),
        words([0x4000, 0xFFFF_8000, 0x7FFF_FFFF, 0]),
        "toward zero, saturating, NaN to 0"
    );
    assert_eq!(
        cpu.read_vreg(1),
        704u128 << 64,
        "unsigned: a negative saturates to 0"
    );
    assert_eq!(cpu.read_vreg(3), f32s([0.5, -1.0, 1.0 / 32768.0, 0.0]));
    assert_eq!(
        cpu.read_vreg(5),
        u128::from(1.5f32.to_bits()) | (u128::from(2_147_483_648f32.to_bits()) << 32),
        "a 64-bit vector clears the upper half"
    );
    assert_eq!(cpu.read_vreg(7), 8192, "the scalar form converts one lane");
    assert_eq!(cpu.read_vreg(9), u128::from((-1.0f64).to_bits()));
}

/// The floating-point reductions across four lanes.
#[test]
fn simd_fp_reductions_across_lanes() {
    let f32s = |lanes: [f32; 4]| -> u128 {
        lanes
            .iter()
            .enumerate()
            .map(|(i, f)| u128::from(f.to_bits()) << (32 * i))
            .sum()
    };
    let code = [
        0x6e30_c800u32, // fmaxnmv s0, v0.4s
        0x6eb0_c841,    // fminnmv s1, v2.4s
        0x6e30_f883,    // fmaxv s3, v4.4s
        0x6eb0_f8c5,    // fminv s5, v6.4s
        nop(),
    ];
    let mut cpu = cpu_at(0x1000);
    let bytes: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
    cpu.mem.map(0x1000, &bytes).unwrap();
    let lanes = f32s([1.5, -7.0, 42.0, 0.25]);
    for reg in [0, 2, 4, 6] {
        cpu.set_vreg(reg, lanes);
    }
    cpu.run(4).unwrap();
    // One lane written, the rest of each register cleared.
    assert_eq!(cpu.read_vreg(0), u128::from(42.0f32.to_bits()));
    assert_eq!(cpu.read_vreg(1), u128::from((-7.0f32).to_bits()));
    assert_eq!(cpu.read_vreg(3), u128::from(42.0f32.to_bits()));
    assert_eq!(cpu.read_vreg(5), u128::from((-7.0f32).to_bits()));
}

/// The scalar pairwise reductions.
#[test]
fn simd_scalar_pairwise_reduces_two_lanes_into_one() {
    let singles = |a: f32, b: f32| u128::from(a.to_bits()) | (u128::from(b.to_bits()) << 32);
    let doubles = |a: f64, b: f64| u128::from(a.to_bits()) | (u128::from(b.to_bits()) << 64);
    let code = [
        0x7e30_f800u32, // fmaxp s0, v0.2s
        0x7e30_d822,    // faddp s2, v1.2s
        0x7e70_d864,    // faddp d4, v3.2d
        0x5ef1_b8a6,    // addp d6, v5.2d
        0x7ef0_c8e8,    // fminnmp d8, v7.2d
        nop(),
    ];
    let mut cpu = cpu_at(0x1000);
    let bytes: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
    cpu.mem.map(0x1000, &bytes).unwrap();
    // Lanes above the pair hold something, which the result must not keep.
    cpu.set_vreg(0, singles(-2.5, 9.0) | (u128::MAX << 64));
    cpu.set_vreg(1, singles(1.25, 2.5));
    cpu.set_vreg(3, doubles(0.5, 0.25));
    cpu.set_vreg(5, u128::from(u64::MAX) | (3u128 << 64));
    cpu.set_vreg(7, doubles(4.0, -8.0));
    cpu.run(5).unwrap();
    assert_eq!(cpu.read_vreg(0), u128::from(9.0f32.to_bits()));
    assert_eq!(cpu.read_vreg(2), u128::from(3.75f32.to_bits()));
    assert_eq!(cpu.read_vreg(4), u128::from(0.75f64.to_bits()));
    assert_eq!(cpu.read_vreg(6), 2, "addp wraps");
    assert_eq!(cpu.read_vreg(8), u128::from((-8.0f64).to_bits()));
}
