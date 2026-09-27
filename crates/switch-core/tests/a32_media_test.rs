//! The ARMv6 media instructions a compiler reaches for in packed-pixel and
//! fixed-point code: the parallel adds and subtracts, `SEL`, the halfword
//! packs, the two-lane saturations, the dual multiply-accumulates into 64
//! bits and the sums of absolute differences.
//!
//! The expected values are not worked out by hand. They are what `qemu-arm`
//! computed for the same instructions on the same operands, so a misreading
//! of the ARM ARM here would have to be made twice, once by qemu.
//! `tools/a32_media_reference.py` regenerates the table.

mod a32;

use a32::{r, run};

const MVN_R8_0: u32 = 0xE3E0_8000;
const MOV_R7_0: u32 = 0xE3A0_7000;
const MOV_R0_0: u32 = 0xE3A0_0000;
/// `usub8 r9, r7, r7`: nothing borrows, so every GE flag starts set.
const USUB8_R9_R7_R7: u32 = 0xE657_9FF7;
/// `sel r6, r8, r7`: 0xff in each byte whose GE flag is set, which is how
/// the flags are read back without reading the status register.
const SEL_R6_R8_R7: u32 = 0xE688_6FB7;

fn movw(rd: u32, imm: u32) -> u32 {
    0xE300_0000 | (imm >> 12 & 0xF) << 16 | rd << 12 | imm & 0xFFF
}

fn movt(rd: u32, imm: u32) -> u32 {
    0xE340_0000 | (imm >> 12 & 0xF) << 16 | rd << 12 | imm & 0xFFF
}

/// Each case: the instructions, `r1`..`r4` going in, and `r0`, `r3`, `r4`
/// and the GE mask coming out.
const CASES: &[(&[u32], [u32; 4], [u32; 4])] = &[
    // sadd16 r0, r1, r2
    (
        &[0xE6110F12],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x00000000, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // sadd16 r0, r1, r2
    (
        &[0xE6110F12],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0xACF03568, 0x80000000, 0x7FFFFFFF, 0x0000FFFF],
    ),
    // sadd16 r0, r1, r2
    (
        &[0xE6110F12],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0x0000FFFF, 0x00000000, 0x00000000, 0xFFFF0000],
    ),
    // sasx r0, r1, r2
    (
        &[0xE6110F32],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x01FE0000, 0x0001FFFF, 0xFFFFFFFF, 0x0000FFFF],
    ),
    // sasx r0, r1, r2
    (
        &[0xE6110F32],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0xF124BBBC, 0x80000000, 0x7FFFFFFF, 0x0000FFFF],
    ),
    // sasx r0, r1, r2
    (
        &[0xE6110F32],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0xFFFEFFFF, 0x00000000, 0x00000000, 0x00000000],
    ),
    // ssax r0, r1, r2
    (
        &[0xE6110F52],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x0000FE02, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // ssax r0, r1, r2
    (
        &[0xE6110F52],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x3344F134, 0x80000000, 0x7FFFFFFF, 0xFFFF0000],
    ),
    // ssax r0, r1, r2
    (
        &[0xE6110F52],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0x00000001, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // ssub16 r0, r1, r2
    (
        &[0xE6110F72],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x01FEFE02, 0x0001FFFF, 0xFFFFFFFF, 0x0000FFFF],
    ),
    // ssub16 r0, r1, r2
    (
        &[0xE6110F72],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x77787788, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // ssub16 r0, r1, r2
    (
        &[0xE6110F72],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0xFFFE0001, 0x00000000, 0x00000000, 0x0000FFFF],
    ),
    // sadd8 r0, r1, r2
    (
        &[0xE6110F92],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0xFF00FF00, 0x0001FFFF, 0xFFFFFFFF, 0x00FF00FF],
    ),
    // sadd8 r0, r1, r2
    (
        &[0xE6110F92],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0xACF03468, 0x80000000, 0x7FFFFFFF, 0x0000FFFF],
    ),
    // sadd8 r0, r1, r2
    (
        &[0xE6110F92],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0xFF00FFFF, 0x00000000, 0x00000000, 0x00FF0000],
    ),
    // ssub8 r0, r1, r2
    (
        &[0xE6110FF2],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x01FEFF02, 0x0001FFFF, 0xFFFFFFFF, 0x0000FFFF],
    ),
    // ssub8 r0, r1, r2
    (
        &[0xE6110FF2],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x78787888, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // ssub8 r0, r1, r2
    (
        &[0xE6110FF2],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0xFFFE0101, 0x00000000, 0x00000000, 0x0000FFFF],
    ),
    // qadd16 r0, r1, r2
    (
        &[0xE6210F12],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x00000000, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // qadd16 r0, r1, r2
    (
        &[0xE6210F12],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0xACF03568, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // qadd16 r0, r1, r2
    (
        &[0xE6210F12],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0x0000FFFF, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // qasx r0, r1, r2
    (
        &[0xE6210F32],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x80000000, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // qasx r0, r1, r2
    (
        &[0xE6210F32],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0xF1247FFF, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // qasx r0, r1, r2
    (
        &[0xE6210F32],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0xFFFEFFFF, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // qsax r0, r1, r2
    (
        &[0xE6210F52],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x00007FFF, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // qsax r0, r1, r2
    (
        &[0xE6210F52],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x3344F134, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // qsax r0, r1, r2
    (
        &[0xE6210F52],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0x00000001, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // qsub16 r0, r1, r2
    (
        &[0xE6210F72],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x80007FFF, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // qsub16 r0, r1, r2
    (
        &[0xE6210F72],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x77787788, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // qsub16 r0, r1, r2
    (
        &[0xE6210F72],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0xFFFE0001, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // qadd8 r0, r1, r2
    (
        &[0xE6210F92],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0xFF00FF00, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // qadd8 r0, r1, r2
    (
        &[0xE6210F92],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0xACF03468, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // qadd8 r0, r1, r2
    (
        &[0xE6210F92],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0xFF00FFFF, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // qsub8 r0, r1, r2
    (
        &[0xE6210FF2],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x80FE7F02, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // qsub8 r0, r1, r2
    (
        &[0xE6210FF2],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x7878787F, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // qsub8 r0, r1, r2
    (
        &[0xE6210FF2],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0xFFFE0101, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // shadd16 r0, r1, r2
    (
        &[0xE6310F12],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x00000000, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // shadd16 r0, r1, r2
    (
        &[0xE6310F12],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0xD6781AB4, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // shadd16 r0, r1, r2
    (
        &[0xE6310F12],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0x0000FFFF, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // shasx r0, r1, r2
    (
        &[0xE6310F32],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x80FF0000, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // shasx r0, r1, r2
    (
        &[0xE6310F32],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0xF8925DDE, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // shasx r0, r1, r2
    (
        &[0xE6310F32],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0xFFFFFFFF, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // shsax r0, r1, r2
    (
        &[0xE6310F52],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x00007F01, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // shsax r0, r1, r2
    (
        &[0xE6310F52],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x19A2F89A, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // shsax r0, r1, r2
    (
        &[0xE6310F52],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // shsub16 r0, r1, r2
    (
        &[0xE6310F72],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x80FF7F01, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // shsub16 r0, r1, r2
    (
        &[0xE6310F72],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x3BBC3BC4, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // shsub16 r0, r1, r2
    (
        &[0xE6310F72],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0xFFFF0000, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // shadd8 r0, r1, r2
    (
        &[0xE6310F92],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0xFF00FF00, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // shadd8 r0, r1, r2
    (
        &[0xE6310F92],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0xD6F81A34, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // shadd8 r0, r1, r2
    (
        &[0xE6310F92],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0xFF00FFFF, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // shsub8 r0, r1, r2
    (
        &[0xE6310FF2],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x80FF7F01, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // shsub8 r0, r1, r2
    (
        &[0xE6310FF2],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x3C3C3C44, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // shsub8 r0, r1, r2
    (
        &[0xE6310FF2],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0xFFFF0000, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // uadd16 r0, r1, r2
    (
        &[0xE6510F12],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x00000000, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // uadd16 r0, r1, r2
    (
        &[0xE6510F12],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0xACF03568, 0x80000000, 0x7FFFFFFF, 0x0000FFFF],
    ),
    // uadd16 r0, r1, r2
    (
        &[0xE6510F12],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0x0000FFFF, 0x00000000, 0x00000000, 0xFFFF0000],
    ),
    // uasx r0, r1, r2
    (
        &[0xE6510F32],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x01FE0000, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // uasx r0, r1, r2
    (
        &[0xE6510F32],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0xF124BBBC, 0x80000000, 0x7FFFFFFF, 0x00000000],
    ),
    // uasx r0, r1, r2
    (
        &[0xE6510F32],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0xFFFEFFFF, 0x00000000, 0x00000000, 0xFFFF0000],
    ),
    // usax r0, r1, r2
    (
        &[0xE6510F52],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x0000FE02, 0x0001FFFF, 0xFFFFFFFF, 0xFFFF0000],
    ),
    // usax r0, r1, r2
    (
        &[0xE6510F52],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x3344F134, 0x80000000, 0x7FFFFFFF, 0x00000000],
    ),
    // usax r0, r1, r2
    (
        &[0xE6510F52],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0x00000001, 0x00000000, 0x00000000, 0xFFFF0000],
    ),
    // usub16 r0, r1, r2
    (
        &[0xE6510F72],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x01FEFE02, 0x0001FFFF, 0xFFFFFFFF, 0xFFFF0000],
    ),
    // usub16 r0, r1, r2
    (
        &[0xE6510F72],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x77787788, 0x80000000, 0x7FFFFFFF, 0x00000000],
    ),
    // usub16 r0, r1, r2
    (
        &[0xE6510F72],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0xFFFE0001, 0x00000000, 0x00000000, 0xFFFF0000],
    ),
    // uadd8 r0, r1, r2
    (
        &[0xE6510F92],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0xFF00FF00, 0x0001FFFF, 0xFFFFFFFF, 0x00FF00FF],
    ),
    // uadd8 r0, r1, r2
    (
        &[0xE6510F92],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0xACF03468, 0x80000000, 0x7FFFFFFF, 0x0000FFFF],
    ),
    // uadd8 r0, r1, r2
    (
        &[0xE6510F92],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0xFF00FFFF, 0x00000000, 0x00000000, 0x00FF0000],
    ),
    // usub8 r0, r1, r2
    (
        &[0xE6510FF2],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x01FEFF02, 0x0001FFFF, 0xFFFFFFFF, 0xFFFF0000],
    ),
    // usub8 r0, r1, r2
    (
        &[0xE6510FF2],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x78787888, 0x80000000, 0x7FFFFFFF, 0x00000000],
    ),
    // usub8 r0, r1, r2
    (
        &[0xE6510FF2],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0xFFFE0101, 0x00000000, 0x00000000, 0xFFFF0000],
    ),
    // uqadd16 r0, r1, r2
    (
        &[0xE6610F12],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0xFFFFFFFF, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // uqadd16 r0, r1, r2
    (
        &[0xE6610F12],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0xACF0FFFF, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // uqadd16 r0, r1, r2
    (
        &[0xE6610F12],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0xFFFFFFFF, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // uqasx r0, r1, r2
    (
        &[0xE6610F32],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0xFFFF0000, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // uqasx r0, r1, r2
    (
        &[0xE6610F32],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0xF1240000, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // uqasx r0, r1, r2
    (
        &[0xE6610F32],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0xFFFF0000, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // uqsax r0, r1, r2
    (
        &[0xE6610F52],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x0000FE02, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // uqsax r0, r1, r2
    (
        &[0xE6610F52],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x0000F134, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // uqsax r0, r1, r2
    (
        &[0xE6610F52],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0x00000001, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // uqsub16 r0, r1, r2
    (
        &[0xE6610F72],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x01FE0000, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // uqsub16 r0, r1, r2
    (
        &[0xE6610F72],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x00000000, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // uqsub16 r0, r1, r2
    (
        &[0xE6610F72],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0xFFFE0000, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // uqadd8 r0, r1, r2
    (
        &[0xE6610F92],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0xFFFFFFFF, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // uqadd8 r0, r1, r2
    (
        &[0xE6610F92],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0xACF0FFFF, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // uqadd8 r0, r1, r2
    (
        &[0xE6610F92],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0xFFFFFFFF, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // uqsub8 r0, r1, r2
    (
        &[0xE6610FF2],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x01FE0000, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // uqsub8 r0, r1, r2
    (
        &[0xE6610FF2],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x00000000, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // uqsub8 r0, r1, r2
    (
        &[0xE6610FF2],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0xFFFE0000, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // uhadd16 r0, r1, r2
    (
        &[0xE6710F12],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x80008000, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // uhadd16 r0, r1, r2
    (
        &[0xE6710F12],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x56789AB4, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // uhadd16 r0, r1, r2
    (
        &[0xE6710F12],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0x80007FFF, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // uhasx r0, r1, r2
    (
        &[0xE6710F32],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x80FF0000, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // uhasx r0, r1, r2
    (
        &[0xE6710F32],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x7892DDDE, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // uhasx r0, r1, r2
    (
        &[0xE6710F32],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0xFFFFFFFF, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // uhsax r0, r1, r2
    (
        &[0xE6710F52],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x00007F01, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // uhsax r0, r1, r2
    (
        &[0xE6710F52],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x99A2789A, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // uhsax r0, r1, r2
    (
        &[0xE6710F52],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // uhsub16 r0, r1, r2
    (
        &[0xE6710F72],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x00FFFF01, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // uhsub16 r0, r1, r2
    (
        &[0xE6710F72],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0xBBBCBBC4, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // uhsub16 r0, r1, r2
    (
        &[0xE6710F72],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0x7FFF8000, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // uhadd8 r0, r1, r2
    (
        &[0xE6710F92],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x7F807F80, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // uhadd8 r0, r1, r2
    (
        &[0xE6710F92],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x56789AB4, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // uhadd8 r0, r1, r2
    (
        &[0xE6710F92],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0x7F807F7F, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // uhsub8 r0, r1, r2
    (
        &[0xE6710FF2],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x007FFF81, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // uhsub8 r0, r1, r2
    (
        &[0xE6710FF2],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0xBCBCBCC4, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // uhsub8 r0, r1, r2
    (
        &[0xE6710FF2],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0x7F7F8080, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // sadd8 r9, r1, r2; sel r0, r1, r2
    (
        &[0xE6119F92, 0xE6810FB2],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x7FFF8001, 0x0001FFFF, 0xFFFFFFFF, 0x00FF00FF],
    ),
    // sadd8 r9, r1, r2; sel r0, r1, r2
    (
        &[0xE6119F92, 0xE6810FB2],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x9ABC5678, 0x80000000, 0x7FFFFFFF, 0x0000FFFF],
    ),
    // sadd8 r9, r1, r2; sel r0, r1, r2
    (
        &[0xE6119F92, 0xE6810FB2],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0x00FFFFFF, 0x00000000, 0x00000000, 0x00FF0000],
    ),
    // usub16 r9, r1, r2; sel r0, r1, r2
    (
        &[0xE6519F72, 0xE6810FB2],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x80FF80FF, 0x0001FFFF, 0xFFFFFFFF, 0xFFFF0000],
    ),
    // usub16 r9, r1, r2; sel r0, r1, r2
    (
        &[0xE6519F72, 0xE6810FB2],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x9ABCDEF0, 0x80000000, 0x7FFFFFFF, 0x00000000],
    ),
    // usub16 r9, r1, r2; sel r0, r1, r2
    (
        &[0xE6519F72, 0xE6810FB2],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0xFFFFFFFF, 0x00000000, 0x00000000, 0xFFFF0000],
    ),
    // pkhbt r0, r1, r2
    (
        &[0xE6810012],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x7F017F01, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // pkhbt r0, r1, r2
    (
        &[0xE6810012],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x9ABC5678, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // pkhbt r0, r1, r2
    (
        &[0xE6810012],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0x00010000, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // pkhbt r0, r1, r2, lsl #8
    (
        &[0xE6810412],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x01807F01, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // pkhbt r0, r1, r2, lsl #8
    (
        &[0xE6810412],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0xBCDE5678, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // pkhbt r0, r1, r2, lsl #8
    (
        &[0xE6810412],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0x01FF0000, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // pkhtb r0, r1, r2, asr #8
    (
        &[0xE6810452],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x80FF0180, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // pkhtb r0, r1, r2, asr #8
    (
        &[0xE6810452],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x1234BCDE, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // pkhtb r0, r1, r2, asr #8
    (
        &[0xE6810452],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0xFFFF01FF, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // pkhtb r0, r1, r2, asr #32
    (
        &[0xE6810052],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x80FF0000, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // pkhtb r0, r1, r2, asr #32
    (
        &[0xE6810052],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x1234FFFF, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // pkhtb r0, r1, r2, asr #32
    (
        &[0xE6810052],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0xFFFF0000, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // ssat16 r0, #9, r1
    (
        &[0xE6A80F31],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0xFF0000FF, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // ssat16 r0, #9, r1
    (
        &[0xE6A80F31],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x00FF00FF, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // ssat16 r0, #9, r1
    (
        &[0xE6A80F31],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0xFFFF0000, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // ssat16 r0, #16, r2
    (
        &[0xE6AF0F32],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // ssat16 r0, #16, r2
    (
        &[0xE6AF0F32],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x9ABCDEF0, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // ssat16 r0, #16, r2
    (
        &[0xE6AF0F32],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0x0001FFFF, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // usat16 r0, #7, r1
    (
        &[0xE6E70F31],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x0000007F, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // usat16 r0, #7, r1
    (
        &[0xE6E70F31],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x007F007F, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // usat16 r0, #7, r1
    (
        &[0xE6E70F31],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // usat16 r0, #0, r2
    (
        &[0xE6E00F32],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x00000000, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // usat16 r0, #0, r2
    (
        &[0xE6E00F32],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x00000000, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // usat16 r0, #0, r2
    (
        &[0xE6E00F32],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // usat16 r0, #15, r1
    (
        &[0xE6EF0F31],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x00007F01, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // usat16 r0, #15, r1
    (
        &[0xE6EF0F31],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x12345678, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // usat16 r0, #15, r1
    (
        &[0xE6EF0F31],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // smlald r3, r4, r1, r2
    (
        &[0xE7443211],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x00000000, 0x81FE03FD, 0xFFFFFFFE, 0xFFFFFFFF],
    ),
    // smlald r3, r4, r1, r2
    (
        &[0xE7443211],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x00000000, 0x6DA1C6B0, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // smlald r3, r4, r1, r2
    (
        &[0xE7443211],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // smlaldx r3, r4, r1, r2
    (
        &[0xE7443231],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x00000000, 0x7E05FC01, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // smlaldx r3, r4, r1, r2
    (
        &[0xE7443231],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x00000000, 0x5B71D8E0, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // smlaldx r3, r4, r1, r2
    (
        &[0xE7443231],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0x00000000, 0x00000001, 0x00000000, 0xFFFFFFFF],
    ),
    // smlsld r3, r4, r1, r2
    (
        &[0xE7443251],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x00000000, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // smlsld r3, r4, r1, r2
    (
        &[0xE7443251],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x00000000, 0x7C087A50, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // smlsld r3, r4, r1, r2
    (
        &[0xE7443251],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0x00000000, 0x00000001, 0x00000000, 0xFFFFFFFF],
    ),
    // smlsldx r3, r4, r1, r2
    (
        &[0xE7443271],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x00000000, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // smlsldx r3, r4, r1, r2
    (
        &[0xE7443271],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x00000000, 0x60258760, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // smlsldx r3, r4, r1, r2
    (
        &[0xE7443271],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // usad8 r0, r1, r2
    (
        &[0xE780F211],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x000001FE, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // usad8 r0, r1, r2
    (
        &[0xE780F211],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x00000210, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // usad8 r0, r1, r2
    (
        &[0xE780F211],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0x000003FB, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
    // usada8 r0, r1, r2, r3
    (
        &[0xE7803211],
        [0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF],
        [0x000201FD, 0x0001FFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    ),
    // usada8 r0, r1, r2, r3
    (
        &[0xE7803211],
        [0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF],
        [0x80000210, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF],
    ),
    // usada8 r0, r1, r2, r3
    (
        &[0xE7803211],
        [0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000],
        [0x000003FB, 0x00000000, 0x00000000, 0xFFFFFFFF],
    ),
];

#[test]
fn the_media_instructions_agree_with_qemu() {
    for (words, inputs, expected) in CASES {
        let mut code = Vec::new();
        for (reg, &value) in (1..=4).zip(inputs) {
            code.push(movw(reg, value & 0xFFFF));
            code.push(movt(reg, value >> 16));
        }
        code.extend([MVN_R8_0, MOV_R7_0, MOV_R0_0, USUB8_R9_R7_R7]);
        code.extend_from_slice(words);
        code.push(SEL_R6_R8_R7);
        let cpu = run(&code);
        let got = [r(&cpu, 0), r(&cpu, 3), r(&cpu, 4), r(&cpu, 6)];
        assert_eq!(got, *expected, "{words:08x?} on {inputs:08x?}");
    }
}
