//! VFP: the AArch32 floating-point coprocessor.
//!
//! The register file is A64's seen differently — `D(2n)` is the bottom half of
//! `V(n)` and `D(2n+1)` the top — so several of these check the aliasing
//! directly rather than only the arithmetic.

mod a32;

use a32::{cpu, r, BASE, HALT, SCRATCH};

use switch_core::cpu::Cpu;

/// Load a 32-bit constant into `rd` with `movw`/`movt`.
fn load(rd: u32, value: u32) -> [u32; 2] {
    let lo = value & 0xFFFF;
    let hi = value >> 16;
    [
        0xE300_0000 | ((lo & 0xF000) << 4) | (rd << 12) | (lo & 0xFFF),
        0xE340_0000 | ((hi & 0xF000) << 4) | (rd << 12) | (hi & 0xFFF),
    ]
}

fn run(code: &[u32]) -> Cpu {
    let mut cpu = cpu();
    let mut bytes = Vec::new();
    for insn in code.iter().chain(std::iter::once(&HALT)) {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(BASE, &bytes).unwrap();
    cpu.run(code.len() as u64 + 1).unwrap();
    cpu
}

/// Assemble `movw/movt` for each value, move it into `s0`/`s1`, then run `ops`.
fn with_singles(a: f32, b: f32, ops: &[u32]) -> Cpu {
    let mut code = Vec::new();
    code.extend_from_slice(&load(0, a.to_bits()));
    code.extend_from_slice(&load(1, b.to_bits()));
    code.push(0xEE00_0A10); // vmov s0, r0
    code.push(0xEE00_1A90); // vmov s1, r1
    code.extend_from_slice(ops);
    run(&code)
}

/// `s3` read back as a float.
fn s3(cpu: &Cpu) -> f32 {
    // vmov r3, s3 has already run in these programs.
    f32::from_bits(r(cpu, 3))
}

#[test]
fn a_core_register_moves_to_and_from_a_single() {
    let cpu = run(&[
        load(0, 0x4048_F5C3)[0], // 3.14
        load(0, 0x4048_F5C3)[1],
        0xEE00_0A10, // vmov s0, r0
        0xEE11_3A90, // vmov r3, s3   (s3 is untouched, so zero)
    ]);
    assert_eq!(r(&cpu, 3), 0, "s3 was never written");
    assert_eq!(cpu.read_vreg(0) as u32, 0x4048_F5C3, "s0 is V0's low word");
}

/// `D1` is the *top* half of `V0`, and `S2`/`S3` are its two words. A file
/// that gave each its own storage would pass the arithmetic tests and fail
/// this one.
#[test]
fn the_single_double_and_vector_views_are_one_register_file() {
    let cpu = run(&[
        load(0, 0x1111_1111)[0],
        load(0, 0x1111_1111)[1],
        load(1, 0x2222_2222)[0],
        load(1, 0x2222_2222)[1],
        0xEC41_0B11, // vmov d1, r0, r1
    ]);
    // D1 is the high half of V0.
    assert_eq!(
        cpu.read_vreg(0) >> 64,
        0x2222_2222_1111_1111,
        "d1 is the top half of v0"
    );
    assert_eq!(cpu.read_vreg(0) as u64, 0, "and d0 was not touched");
}

#[test]
fn the_arithmetic_on_singles() {
    let ops = |op: u32| [op, 0xEE11_3A90];
    assert_eq!(s3(&with_singles(7.0, 3.0, &ops(0xEE70_1A20))), 10.0, "vadd");
    assert_eq!(s3(&with_singles(7.0, 3.0, &ops(0xEE70_1A60))), 4.0, "vsub");
    assert_eq!(s3(&with_singles(7.0, 3.0, &ops(0xEE60_1A20))), 21.0, "vmul");
    assert_eq!(s3(&with_singles(6.0, 3.0, &ops(0xEEC0_1A20))), 2.0, "vdiv");
    assert_eq!(
        s3(&with_singles(7.0, 3.0, &ops(0xEE60_1A60))),
        -21.0,
        "vnmul negates the product"
    );
}

/// The accumulating forms read their destination as a third operand, so a
/// decoder that only fetched two would multiply into a register it never read.
#[test]
fn the_multiply_accumulate_forms_read_their_destination() {
    /// `s0 = 7, s1 = 3, s3 = 100`, then `op`, then `r3 = s3`.
    fn accumulate(op: u32) -> f32 {
        let mut code: Vec<u32> = Vec::new();
        code.extend_from_slice(&load(0, 7.0f32.to_bits()));
        code.extend_from_slice(&load(1, 3.0f32.to_bits()));
        code.extend_from_slice(&load(3, 100.0f32.to_bits()));
        code.push(0xEE00_0A10); // vmov s0, r0
        code.push(0xEE00_1A90); // vmov s1, r1
        code.push(0xEE01_3A90); // vmov s3, r3
        code.push(op);
        code.push(0xEE11_3A90); // vmov r3, s3
        f32::from_bits(r(&run(&code), 3))
    }
    assert_eq!(accumulate(0xEE40_1A20), 100.0 + 21.0, "vmla");
    assert_eq!(accumulate(0xEE40_1A60), 100.0 - 21.0, "vmls");
}

#[test]
fn the_one_operand_arithmetic() {
    let ops = |op: u32| [op, 0xEE11_3A90];
    assert_eq!(s3(&with_singles(-5.0, 0.0, &ops(0xEEF0_1AC0))), 5.0, "vabs");
    assert_eq!(s3(&with_singles(5.0, 0.0, &ops(0xEEF1_1A40))), -5.0, "vneg");
    assert_eq!(s3(&with_singles(9.0, 0.0, &ops(0xEEF1_1AC0))), 3.0, "vsqrt");
}

/// `VABS` and `VNEG` are bit operations on the sign, not `f32::abs`: the sign
/// of a NaN is architectural.
#[test]
fn abs_and_neg_move_the_sign_bit_of_a_nan() {
    let nan = 0xFFC0_0000u32; // a negative quiet NaN
    let cpu = with_singles(f32::from_bits(nan), 0.0, &[0xEEF0_1AC0, 0xEE11_3A90]);
    assert_eq!(r(&cpu, 3), 0x7FC0_0000, "vabs cleared the sign and kept the payload");
}

#[test]
fn the_immediate_moves_expand_the_vfp_encoding() {
    let cpu = run(&[
        0xEEB7_0A00, // vmov.f32 s0, #1.0
        0xEE10_3A10, // vmov r3, s0
    ]);
    assert_eq!(f32::from_bits(r(&cpu, 3)), 1.0);

    let cpu = run(&[
        0xEEB8_0B00, // vmov.f64 d0, #-2.0
        0xEC53_2B10, // vmov r2, r3, d0
    ]);
    let bits = u64::from(r(&cpu, 2)) | (u64::from(r(&cpu, 3)) << 32);
    assert_eq!(f64::from_bits(bits), -2.0);
}

/// A comparison writes FPSCR, not the condition flags. `VMRS APSR_nzcv` is
/// what moves it across — without which every `vcmp` would be invisible to the
/// branch that follows it.
#[test]
fn a_comparison_reaches_the_condition_flags_only_through_vmrs() {
    let mut code: Vec<u32> = Vec::new();
    code.extend_from_slice(&load(0, 1.0f32.to_bits()));
    code.extend_from_slice(&load(1, 2.0f32.to_bits()));
    code.push(0xEE00_0A10); // vmov s0, r0
    code.push(0xEE00_1A90); // vmov s1, r1
    code.push(0xEEB4_0A60); // vcmp.f32 s0, s1
    code.push(0xE3A0_2001); // mov r2, #1     (flags untouched by vcmp)
    code.push(0x43A0_2002); // movmi r2, #2   -- not taken yet
    code.push(0xEEF1_FA10); // vmrs APSR_nzcv, fpscr
    code.push(0x43A0_3003); // movmi r3, #3   -- now taken: 1.0 < 2.0 sets N
    let cpu = run(&code);
    assert_eq!(r(&cpu, 2), 1, "vcmp alone does not move the condition flags");
    assert_eq!(r(&cpu, 3), 3, "vmrs does");
}

/// The conversions cross precisions, so the destination is numbered by the
/// *other* register rule than the source. Using one rule for both writes a
/// register sixteen away.
#[test]
fn the_precision_conversions_number_their_destination_by_the_other_rule() {
    // A single widened into d1.
    let mut code: Vec<u32> = Vec::new();
    code.extend_from_slice(&load(0, 2.5f32.to_bits()));
    code.push(0xEE00_0A10); // vmov s0, r0
    code.push(0xEEB7_1AC0); // vcvt.f64.f32 d1, s0
    code.push(0xEC53_2B11); // vmov r2, r3, d1
    let cpu = run(&code);
    let bits = u64::from(r(&cpu, 2)) | (u64::from(r(&cpu, 3)) << 32);
    assert_eq!(f64::from_bits(bits), 2.5, "widened into d1");

    // And a double narrowed back into s2.
    let mut code: Vec<u32> = Vec::new();
    code.extend_from_slice(&load(0, (2.5f64.to_bits() & 0xFFFF_FFFF) as u32));
    code.extend_from_slice(&load(1, (2.5f64.to_bits() >> 32) as u32));
    code.push(0xEC41_0B11); // vmov d1, r0, r1
    code.push(0xEEB7_1BC1); // vcvt.f32.f64 s2, d1
    code.push(0xEE11_3A10); // vmov r3, s2
    let cpu = run(&code);
    assert_eq!(f32::from_bits(r(&cpu, 3)), 2.5, "narrowed into s2");
}

/// The integer conversions always use an `S` register for the integer, even
/// when the float half is a double — so that operand is numbered `Vm:M` and
/// not `M:Vm`.
#[test]
fn the_integer_conversions_keep_the_integer_in_a_single() {
    let mut code: Vec<u32> = Vec::new();
    code.extend_from_slice(&load(0, (-7i32) as u32));
    code.push(0xEE00_0A10); // vmov s0, r0
    code.push(0xEEB8_1BC0); // vcvt.f64.s32 d1, s0
    code.push(0xEC53_2B11); // vmov r2, r3, d1
    let cpu = run(&code);
    let bits = u64::from(r(&cpu, 2)) | (u64::from(r(&cpu, 3)) << 32);
    assert_eq!(f64::from_bits(bits), -7.0);

    // And a float back to a signed integer.
    let mut code: Vec<u32> = Vec::new();
    code.extend_from_slice(&load(0, (-7.9f32).to_bits()));
    code.push(0xEE00_0A10); // vmov s0, r0
    code.push(0xEEFD_0AC0); // vcvt.s32.f32 s1, s0
    code.push(0xEE10_3A90); // vmov r3, s1
    let cpu = run(&code);
    assert_eq!(r(&cpu, 3) as i32, -7, "toward zero, which is the default");
}

#[test]
fn the_loads_and_stores_move_singles_and_doubles() {
    let mut code: Vec<u32> = Vec::new();
    code.extend_from_slice(&load(0, SCRATCH));
    code.extend_from_slice(&load(1, 0x1111_1111));
    code.extend_from_slice(&load(2, 0x2222_2222));
    code.push(0xEC42_1B10); // vmov d0, r1, r2
    code.push(0xED80_0B00); // vstr d0, [r0]
    code.push(0xED90_1B00); // vldr d1, [r0]
    code.push(0xEC53_2B11); // vmov r2, r3, d1
    let cpu = run(&code);
    assert_eq!(cpu.mem.read_u32(SCRATCH).unwrap(), 0x1111_1111);
    assert_eq!(cpu.mem.read_u32(SCRATCH + 4).unwrap(), 0x2222_2222);
    assert_eq!((r(&cpu, 2), r(&cpu, 3)), (0x1111_1111, 0x2222_2222));
}

/// `vpush {d0-d7}` is the first VFP instruction Mario Kart 8 Deluxe's `rtld`
/// reaches, 8192 instructions in.
#[test]
fn vpush_and_vpop_round_trip_eight_doubles() {
    let mut code: Vec<u32> = Vec::new();
    // Fill d0 with a known value, push the bank, clobber d0, pop it back.
    code.extend_from_slice(&load(1, 0xAAAA_AAAA));
    code.extend_from_slice(&load(2, 0xBBBB_BBBB));
    code.push(0xEC42_1B10); // vmov d0, r1, r2
    code.push(0xED2D_0B10); // vpush {d0-d7}
    code.extend_from_slice(&load(1, 0));
    code.extend_from_slice(&load(2, 0));
    code.push(0xEC42_1B10); // vmov d0, r1, r2  -- clobber
    code.push(0xECBD_0B10); // vpop {d0-d7}
    code.push(0xEC53_2B10); // vmov r2, r3, d0
    let cpu = run(&code);
    assert_eq!((r(&cpu, 2), r(&cpu, 3)), (0xAAAA_AAAA, 0xBBBB_BBBB));
    assert_eq!(cpu.sp(), 0x9000, "the stack came back to where it started");
}

#[test]
fn a_block_transfer_writes_its_base_back() {
    let mut code: Vec<u32> = Vec::new();
    code.extend_from_slice(&load(0, SCRATCH));
    code.push(0xECA0_0B04); // vstmia r0!, {d0-d1}
    let cpu = run(&code);
    assert_eq!(r(&cpu, 0), SCRATCH + 16, "two doubles is sixteen bytes");
}
