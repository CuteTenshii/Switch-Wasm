//! The A32 data-processing core: the shifter's carry, the condition codes,
//! the multiplies and the ARMv6 media group.

mod a32;

use a32::{r, run, BASE, HALT};

/// The shifter's carry out *is* `C` for a logical operation, and `ADC` reads
/// it back. `lsls r1, r0, #1` on 0x00008001 shifts a zero out, so the `adc`
/// adds nothing.
#[test]
fn a_logical_operation_takes_its_carry_from_the_shifter() {
    let cpu = run(&[
        0xE308_0001, // movw r0, #0x8001
        0xE1B0_1080, // lsls r1, r0, #1
        0xE2A2_2000, // adc  r2, r2, #0
    ]);
    assert_eq!(r(&cpu, 1), 0x0001_0002);
    assert_eq!(r(&cpu, 2), 0);

    // The same shift out of bit 31 sets it.
    let cpu = run(&[
        0xE3A0_0102, // mov  r0, #0x80000000
        0xE1B0_1080, // lsls r1, r0, #1
        0xE2A2_2000, // adc  r2, r2, #0
    ]);
    assert_eq!(r(&cpu, 1), 0);
    assert_eq!(r(&cpu, 2), 1);
}

/// `RRX` is `ROR` with an immediate amount of zero: one place right through
/// the carry, which is not a rotate at all.
#[test]
fn rrx_shifts_through_the_carry_rather_than_rotating() {
    let cpu = run(&[
        0xE3A0_0001, // mov  r0, #1
        0xE1B0_1060, // movs r1, r0, rrx
        0xE1B0_2061, // movs r2, r1, rrx
    ]);
    // C started clear, so the first RRX brings in a zero and shifts the 1 out.
    assert_eq!(r(&cpu, 1), 0);
    // That 1 is now the carry, and comes back as bit 31.
    assert_eq!(r(&cpu, 2), 0x8000_0000);
}

/// ARM's `C` after a subtraction is *not* a borrow: it is the adder's carry
/// out, so a subtraction that does not borrow sets it.
#[test]
fn subtraction_sets_carry_when_it_does_not_borrow() {
    let cpu = run(&[
        0xE3B0_0005, // movs r0, #5
        0xE250_1003, // subs r1, r0, #3
        0xE2A3_3000, // adc  r3, r3, #0
    ]);
    assert_eq!(r(&cpu, 1), 2);
    assert_eq!(r(&cpu, 3), 1, "5 - 3 does not borrow, so C is set");

    let cpu = run(&[
        0xE3B0_0005, // movs r0, #5
        0xE250_2007, // subs r2, r0, #7
        0xE2A3_3000, // adc  r3, r3, #0
    ]);
    assert_eq!(r(&cpu, 2), (-2i32) as u32);
    assert_eq!(r(&cpu, 3), 0, "5 - 7 borrows, so C is clear");
}

#[test]
fn a_condition_that_fails_costs_only_the_advance() {
    let cpu = run(&[
        0xE3A0_0000, // mov  r0, #0
        0xE350_0000, // cmp  r0, #0
        0x03A0_1007, // moveq r1, #7
        0x13A0_2009, // movne r2, #9
    ]);
    assert_eq!(r(&cpu, 1), 7);
    assert_eq!(r(&cpu, 2), 0);
}

/// Reading r15 yields the instruction's own address plus 8, the pipeline
/// offset the architecture made visible.
#[test]
fn r15_reads_as_the_instruction_address_plus_eight() {
    let cpu = run(&[
        0xE1A0_000F, // mov r0, pc
        0xE28F_1000, // add r1, pc, #0
    ]);
    assert_eq!(r(&cpu, 0), BASE + 8);
    assert_eq!(r(&cpu, 1), BASE + 4 + 8);
}

/// The register-shifted form takes its amount from the bottom byte only, and
/// an amount of zero leaves both the value and the carry alone.
#[test]
fn a_register_shift_of_zero_changes_nothing() {
    let cpu = run(&[
        0xE3A0_00FF, // mov  r0, #0xff
        0xE3A0_1000, // mov  r1, #0
        0xE1B0_2110, // movs r2, r0, lsl r1
        0xE3A0_3B01, // mov  r3, #0x400
        0xE1B0_4310, // movs r4, r0, lsl r3   (amount 0x00 after masking)
    ]);
    assert_eq!(r(&cpu, 2), 0xFF);
    assert_eq!(
        r(&cpu, 4),
        0xFF,
        "only the bottom byte of the amount counts"
    );
}

#[test]
fn the_long_multiplies_accumulate_into_a_register_pair() {
    let cpu = run(&[
        0xE3A0_2102, // mov r2, #0x80000000
        0xE3A0_3004, // mov r3, #4
        0xE081_0392, // umull r0, r1, r2, r3
    ]);
    assert_eq!(r(&cpu, 0), 0);
    assert_eq!(r(&cpu, 1), 2);

    // The signed form of the same product is negative.
    let cpu = run(&[
        0xE3A0_2102, // mov r2, #0x80000000
        0xE3A0_3004, // mov r3, #4
        0xE0C1_0392, // smull r0, r1, r2, r3
    ]);
    assert_eq!(r(&cpu, 0), 0);
    assert_eq!(r(&cpu, 1), 0xFFFF_FFFE);
}

#[test]
fn mls_subtracts_its_product() {
    let cpu = run(&[
        0xE3A0_9003, // mov r9, #3
        0xE3A0_A004, // mov r10, #4
        0xE3A0_B064, // mov r11, #100
        0xE068_BA99, // mls r8, r9, r10, r11
    ]);
    assert_eq!(r(&cpu, 8), 100 - 12);
}

#[test]
fn the_extends_and_the_reverses() {
    let cpu = run(&[
        0xE3A0_10FF, // mov   r1, #0xff
        0xE6EF_0071, // uxtb  r0, r1
        0xE3E0_3000, // mvn   r3, #0        -> 0xffffffff
        0xE6BF_2073, // sxth  r2, r3
        0xE59F_5014, // ldr   r5, [pc, #20] -> the literal below
        0xE6BF_4F35, // rev   r4, r5
        0xE6BF_6FB5, // rev16 r6, r5
        0xE6FF_AF35, // rbit  r10, r5
        0xE3A0_1001, // mov   r1, #1
        0xE6E1_0072, // uxtab r0, r1, r2
        HALT,        // stop before the literal
        0x1122_3344,
    ]);
    assert_eq!(r(&cpu, 0), 1 + 0xFF, "uxtab adds the extended byte");
    assert_eq!(r(&cpu, 2), 0xFFFF_FFFF, "sxth of -1 is -1");
    assert_eq!(r(&cpu, 4), 0x4433_2211, "rev reverses the whole word");
    assert_eq!(
        r(&cpu, 6),
        0x2211_4433,
        "rev16 reverses inside each halfword, and not the halfwords"
    );
    assert_eq!(r(&cpu, 10), 0x22CC_4488, "rbit reverses the bits");
}

#[test]
fn the_bitfield_instructions() {
    let cpu = run(&[
        0xE59F_1020, // ldr  r1, [pc, #32] -> the literal below
        0xE1A0_3001, // mov  r3, r1
        0xE1A0_5001, // mov  r5, r1
        0xE7E7_0251, // ubfx r0, r1, #4, #8
        0xE7A7_2253, // sbfx r2, r3, #4, #8
        0xE3A0_400F, // mov  r4, #15
        0xE7CB_4415, // bfi  r4, r5, #8, #4
        0xE3E0_6000, // mvn  r6, #0
        0xE7CB_641F, // bfc  r6, #8, #4
        HALT,
        0xDEAD_BEEF,
    ]);
    assert_eq!(r(&cpu, 0), 0xEE, "0xdeadbeef bits 11:4");
    assert_eq!(r(&cpu, 2), 0xFFFF_FFEE, "the same field, sign extended");
    assert_eq!(r(&cpu, 4), 0x0000_0F0F, "bits 11:8 replaced by 0xef & 0xf");
    assert_eq!(r(&cpu, 6), 0xFFFF_F0FF, "bfc clears bits 11:8");
}

/// `SSAT` shares its `op1` with the extends and is told apart by bit 5 of
/// `op2`, so a decoder that keys on the opcode field alone runs one as the
/// other.
#[test]
fn saturating_arithmetic_clamps_to_the_edge_it_overflowed_towards() {
    let cpu = run(&[
        0xE3A0_0102, // mov  r0, #0x80000000
        0xE6A7_0010, // ssat r0, #8, r0
        0xE3A0_1CFF, // mov  r1, #0xff00
        0xE6E8_2011, // usat r2, #8, r1
    ]);
    assert_eq!(r(&cpu, 0), (-128i32) as u32, "clamped to the signed floor");
    assert_eq!(r(&cpu, 2), 0xFF, "clamped to the unsigned ceiling");

    // Two large negatives saturate downward rather than wrapping positive.
    let cpu = run(&[
        0xE3A0_5102, // mov  r5, #0x80000000
        0xE1A0_6005, // mov  r6, r5
        0xE106_4055, // qadd r4, r5, r6
    ]);
    assert_eq!(r(&cpu, 4), i32::MIN as u32);
}
