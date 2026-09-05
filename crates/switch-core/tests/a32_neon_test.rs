//! Advanced SIMD (NEON).
//!
//! The classes covered here are the ones Mario Kart 8 Deluxe's modules
//! actually contain, measured by canonicalising every NEON encoding in their
//! 4.8M instruction words. `Qn` is `V(n)` whole and `Dn` its halves, so the
//! vector register file is checked through `read_vreg` directly rather than
//! only through the arithmetic.

mod a32;

use a32::{cpu as new_cpu, load, r, SCRATCH};

use switch_core::cpu::Cpu;

/// A quad register holding four `f32` lanes.
fn quad(values: [f32; 4]) -> u128 {
    values.iter().enumerate().fold(0u128, |acc, (i, v)| {
        acc | (u128::from(v.to_bits()) << (32 * i))
    })
}

fn lanes(cpu: &Cpu, q: u8) -> [f32; 4] {
    let v = cpu.read_vreg(q);
    let mut out = [0.0; 4];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = f32::from_bits((v >> (32 * i)) as u32);
    }
    out
}

/// Run `ops` with `q0` and `q1` preloaded.
fn with_quads(a: [f32; 4], b: [f32; 4], ops: &[u32]) -> Cpu {
    let mut cpu = new_cpu();
    cpu.set_vreg(0, quad(a));
    cpu.set_vreg(1, quad(b));
    load(&mut cpu, ops);
    cpu.run(ops.len() as u64 + 1).unwrap();
    cpu
}

#[test]
fn the_immediate_move_expands_its_pattern_across_the_register() {
    let mut cpu = new_cpu();
    cpu.set_vreg(0, u128::MAX);
    load(&mut cpu, &[0xF280_0050u32]);
    cpu.run(2).unwrap();
    assert_eq!(cpu.read_vreg(0), 0, "vmov.i32 q0, #0 clears the whole quad");

    // A byte pattern replicates through a D register and leaves the quad's
    // other half alone.
    let mut cpu = new_cpu();
    cpu.set_vreg(0, u128::MAX);
    load(&mut cpu, &[0xF387_0E1Fu32]);
    cpu.run(2).unwrap();
    assert_eq!(cpu.read_vreg(0), u128::MAX, "vmov.i8 d0, #0xff");
}

#[test]
fn the_floating_point_arithmetic_runs_lane_by_lane() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [10.0, 20.0, 30.0, 40.0];
    assert_eq!(
        lanes(&with_quads(a, b, &[0xF200_4D42]), 2),
        [11.0, 22.0, 33.0, 44.0],
        "vadd.f32"
    );
    assert_eq!(
        lanes(&with_quads(a, b, &[0xF300_4D52]), 2),
        [10.0, 40.0, 90.0, 160.0],
        "vmul.f32"
    );
    assert_eq!(
        lanes(&with_quads(b, a, &[0xF220_4D42]), 2),
        [9.0, 18.0, 27.0, 36.0],
        "vsub.f32"
    );
    assert_eq!(
        lanes(&with_quads(a, b, &[0xF200_4F42]), 2),
        [10.0, 20.0, 30.0, 40.0],
        "vmax.f32"
    );
    assert_eq!(
        lanes(&with_quads(a, b, &[0xF220_4F42]), 2),
        [1.0, 2.0, 3.0, 4.0],
        "vmin.f32"
    );
}

#[test]
fn the_integer_add_wraps_within_each_lane() {
    let mut cpu = new_cpu();
    cpu.set_vreg(0, (u128::from(u32::MAX) << 96) | 5);
    cpu.set_vreg(1, (1u128 << 96) | 7);
    load(&mut cpu, &[0xF220_4842u32]);
    cpu.run(2).unwrap();
    let v = cpu.read_vreg(2);
    assert_eq!(v & 0xFFFF_FFFF, 12, "the low lane added");
    assert_eq!(v >> 96, 0, "and the top lane wrapped inside itself");
}

/// The bitwise operations take their operation from the `size` field rather
/// than the opcode, which is the one place in the three-register group where
/// `size` is not an element width.
#[test]
fn the_bitwise_operations_are_selected_by_the_size_field() {
    let mut cpu = new_cpu();
    cpu.set_vreg(0, 0xF0F0_F0F0);
    cpu.set_vreg(1, 0x00FF_00FF);
    load(&mut cpu, &[0xF220_4152u32, 0xF200_4152]);
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_vreg(2), 0xF0F0_F0F0 | 0x00FF_00FF, "vorr");
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_vreg(2), 0xF0F0_F0F0 & 0x00FF_00FF, "vand");
}

/// `VBSL` takes its mask from the *destination*, which is what distinguishes
/// it from `VBIT` and `VBIF` beside it.
#[test]
fn bit_select_takes_its_mask_from_the_destination() {
    let mut cpu = new_cpu();
    cpu.set_vreg(0, 0xAAAA_AAAA); // the "true" source
    cpu.set_vreg(1, 0x5555_5555); // the "false" source
    cpu.set_vreg(2, 0x0000_FFFF); // the mask
    load(&mut cpu, &[0xF310_4152u32]);
    cpu.run(2).unwrap();
    assert_eq!(cpu.read_vreg(2), 0x5555_AAAA);
}

/// The float multiply-accumulate by element is the single most common NEON
/// encoding in the title, 2,898 of them.
#[test]
fn the_multiply_by_element_broadcasts_one_lane() {
    let mut cpu = new_cpu();
    cpu.set_vreg(0, quad([1.0, 2.0, 3.0, 4.0]));
    // d2 is the low half of q1; its two lanes are the scalar candidates.
    cpu.set_vreg(1, quad([10.0, 100.0, 0.0, 0.0]));
    cpu.set_vreg(2, quad([1000.0, 1000.0, 1000.0, 1000.0]));
    load(&mut cpu, &[0xF3A0_4142u32]);
    cpu.run(2).unwrap();
    assert_eq!(
        lanes(&cpu, 2),
        [1010.0, 1020.0, 1030.0, 1040.0],
        "vmla.f32 q2, q0, d2[0] accumulates lane 0 of d2"
    );

    // Lane 1 of the same D register.
    let mut cpu = new_cpu();
    cpu.set_vreg(0, quad([1.0, 2.0, 3.0, 4.0]));
    cpu.set_vreg(1, quad([10.0, 100.0, 0.0, 0.0]));
    load(&mut cpu, &[0xF3A0_4962u32]);
    cpu.run(2).unwrap();
    assert_eq!(
        lanes(&cpu, 2),
        [100.0, 200.0, 300.0, 400.0],
        "vmul by d2[1]"
    );
}

/// `VEXT` slides a window across the pair, and shares its `1011` opcode field
/// with the two-register group: bit 24 is all that tells them apart.
#[test]
fn ext_slides_a_window_across_the_register_pair() {
    let mut cpu = new_cpu();
    cpu.set_vreg(0, 0x0F0E_0D0C_0B0A_0908_0706_0504_0302_0100);
    cpu.set_vreg(1, 0x1F1E_1D1C_1B1A_1918_1716_1514_1312_1110);
    load(&mut cpu, &[0xF2B0_4442u32]);
    cpu.run(2).unwrap();
    // Four bytes in: the window starts at byte 4 of q0 and runs into q1.
    assert_eq!(cpu.read_vreg(2), 0x1312_1110_0F0E_0D0C_0B0A_0908_0706_0504);
}

#[test]
fn the_table_lookup_gathers_bytes_and_zeroes_the_misses() {
    let mut cpu = new_cpu();
    // d0/d1 are the two halves of q0: a table of sixteen bytes.
    cpu.set_vreg(0, 0x0F0E_0D0C_0B0A_0908_0706_0504_0302_0100);
    // d2 is the low half of q1: the indices, one of them out of range.
    cpu.set_vreg(1, 0x0000_0000_0000_0000 | 0xFF03_0201u128);
    load(&mut cpu, &[0xF3B0_4902u32]);
    cpu.run(2).unwrap();
    // d4 is the low half of q2.
    let out = (cpu.read_vreg(2) as u64).to_le_bytes();
    assert_eq!(out[0], 0x01);
    assert_eq!(out[1], 0x02);
    assert_eq!(out[2], 0x03);
    assert_eq!(out[3], 0x00, "index 0xff is out of range, so zero");
}

#[test]
fn dup_broadcasts_one_lane_to_every_lane() {
    let mut cpu = new_cpu();
    cpu.set_vreg(0, quad([7.0, 9.0, 0.0, 0.0]));
    load(&mut cpu, &[0xF3BC_4C40u32]);
    cpu.run(2).unwrap();
    assert_eq!(lanes(&cpu, 2), [9.0, 9.0, 9.0, 9.0], "d0[1] to every lane");
}

#[test]
fn the_two_register_miscellaneous_operations() {
    let mut cpu = new_cpu();
    cpu.set_vreg(0, 0x0F0E_0D0C_0B0A_0908_0706_0504_0302_0100);
    load(&mut cpu, &[0xF3B0_40C0u32]);
    cpu.run(2).unwrap();
    assert_eq!(
        cpu.read_vreg(2),
        0x0C0D_0E0F_0809_0A0B_0405_0607_0001_0203,
        "vrev32.8 reverses the bytes inside each word"
    );

    let mut cpu = new_cpu();
    cpu.set_vreg(0, 0xFF0F_0301);
    load(&mut cpu, &[0xF3B0_4500u32]);
    cpu.run(2).unwrap();
    let out = (cpu.read_vreg(2) as u64).to_le_bytes();
    assert_eq!([out[0], out[1], out[2], out[3]], [1, 2, 4, 8], "vcnt.8");
}

/// `VLD1`/`VST1` moving two `D` registers is the commonest NEON memory
/// instruction in the title, 14,168 across its modules, mostly this shape.
#[test]
fn the_vector_loads_and_stores_move_two_registers_and_advance_the_base() {
    let mut cpu = new_cpu();
    cpu.set_vreg(1, 0x1122_3344_5566_7788_99AA_BBCC_DDEE_FF00);
    // r1 = SCRATCH, then store q1 through it, then load it back into q0.
    let code = [
        0xE3A0_1A08u32, // mov r1, #0x8000
        0xE1A0_0001,    // mov r0, r1
        0xF401_2A8D,    // vst1.32 {d2, d3}, [r1]!
        0xF420_0A8D,    // vld1.32 {d0, d1}, [r0]!
    ];
    load(&mut cpu, &code);
    cpu.run(code.len() as u64 + 1).unwrap();
    assert_eq!(
        cpu.read_vreg(0),
        0x1122_3344_5566_7788_99AA_BBCC_DDEE_FF00,
        "the quad round-tripped through memory"
    );
    assert_eq!(r(&cpu, 1), SCRATCH + 16, "the store advanced its base");
    assert_eq!(r(&cpu, 0), SCRATCH + 16, "and so did the load");
}

/// `VSEL` carries its own condition rather than the instruction's, which is
/// why it lives in the unconditional encoding space.
#[test]
fn vsel_reads_the_condition_flags_it_names() {
    // s0 = 1.0, s1 = 2.0; compare 1 against 2 so GT is false and the second
    // operand is taken.
    let mut cpu = new_cpu();
    let code = [
        0xE3A0_0001u32, // mov r0, #1  -> sets no flags
        0xE350_0002,    // cmp r0, #2  -> 1 < 2, so GT is false
        0xFE30_1A20,    // vselgt.f32 s2, s0, s1
    ];
    cpu.set_vreg(0, quad([1.0, 2.0, 0.0, 0.0])); // s0 = 1.0, s1 = 2.0
    load(&mut cpu, &code);
    cpu.run(code.len() as u64 + 1).unwrap();
    assert_eq!(lanes(&cpu, 0)[1], 2.0, "s1 is unchanged");
    assert_eq!(lanes(&cpu, 0)[2], 2.0, "s2 took the false operand");
}
