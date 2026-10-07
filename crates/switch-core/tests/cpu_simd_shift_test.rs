//! SIMD shifts by immediate and by register.

mod cpu;

use cpu::*;

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
fn scalar_shift_by_immediate() {
    // `ushr d30, d31, #32` = 0x7f6007fe: scalar forms differ from vector ones in bit 28.
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

/// The shift amount is the lane's low byte sign-extended, so negative shifts right.
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
