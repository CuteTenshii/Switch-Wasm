//! A32 memory access: the addressing modes, the block transfers, and the
//! exclusive pairs.

mod a32;

use a32::{r, run, HALT, SCRATCH};

/// Every addressing mode moves the lowest register to the lowest address,
/// whichever direction the base walks.
#[test]
fn the_block_transfers_all_run_lowest_register_to_lowest_address() {
    let cpu = run(&[
        0xE3A0_0A08, // mov r0, #0x8000
        0xE3A0_1001, // mov r1, #1
        0xE3A0_2002, // mov r2, #2
        0xE3A0_3003, // mov r3, #3
        0xE880_000E, // stmia r0, {r1, r2, r3}
        0xE890_0070, // ldmia r0, {r4, r5, r6}
    ]);
    assert_eq!(cpu.mem.read_u32(SCRATCH).unwrap(), 1);
    assert_eq!(cpu.mem.read_u32(SCRATCH + 4).unwrap(), 2);
    assert_eq!(cpu.mem.read_u32(SCRATCH + 8).unwrap(), 3);
    assert_eq!((r(&cpu, 4), r(&cpu, 5), r(&cpu, 6)), (1, 2, 3));
}

#[test]
fn a_decrementing_store_writes_below_its_base_and_leaves_it_there() {
    let cpu = run(&[
        0xE3A0_0A08, // mov r0, #0x8000
        0xE3A0_5005, // mov r5, #5
        0xE3A0_6006, // mov r6, #6
        0xE920_0060, // stmdb r0!, {r5, r6}
    ]);
    assert_eq!(cpu.mem.read_u32(SCRATCH - 8).unwrap(), 5);
    assert_eq!(cpu.mem.read_u32(SCRATCH - 4).unwrap(), 6);
    assert_eq!(r(&cpu, 0), SCRATCH - 8);
}

/// A store transfers the base's *original* value, so the writeback cannot
/// happen before the loop that reads the registers.
#[test]
fn a_store_of_its_own_base_transfers_the_value_it_started_with() {
    let cpu = run(&[
        0xE3A0_0A08, // mov r0, #0x8000
        0xE3A0_1001, // mov r1, #1
        0xE8A0_0003, // stmia r0!, {r0, r1}
    ]);
    assert_eq!(cpu.mem.read_u32(SCRATCH).unwrap(), SCRATCH);
    assert_eq!(
        r(&cpu, 0),
        SCRATCH + 8,
        "and the base still walks past both"
    );
}

/// `push {..., lr}` / `pop {..., pc}` is how every A32 function returns, so a
/// load of r15 out of a block transfer has to branch.
#[test]
fn a_block_load_of_r15_branches() {
    let cpu = run(&[
        0xE3A0_0A08, // mov  r0, #0x8000
        0xE59F_100C, // ldr  r1, [pc, #12]  -> the address of `mov r4, #4`
        0xE580_1000, // str  r1, [r0]
        0xE890_8000, // ldmia r0, {pc}
        0xE3A0_2001, // mov  r2, #1         (skipped)
        HALT,        // (skipped)
        0x0000_101C, // literal: the address of the mov below
        0xE3A0_4004, // mov  r4, #4         <- landed on
    ]);
    assert_eq!(r(&cpu, 4), 4);
    assert_eq!(r(&cpu, 2), 0, "the branch skipped this");
}

/// A post-indexed load writes the base back *after* the value lands, so
/// `ldr r0, [r0], #4` keeps what it loaded.
#[test]
fn a_post_indexed_load_into_its_own_base_keeps_the_value() {
    let cpu = run(&[
        0xE3A0_0A08, // mov r0, #0x8000
        0xE3A0_1063, // mov r1, #0x63
        0xE580_1000, // str r1, [r0]
        0xE490_0004, // ldr r0, [r0], #4
    ]);
    assert_eq!(r(&cpu, 0), 0x63);
}

#[test]
fn the_halfword_and_doubleword_forms() {
    let cpu = run(&[
        0xE3A0_1A08, // mov   r1, #0x8000
        0xE3E0_0000, // mvn   r0, #0        -> 0xffffffff
        0xE1C1_00B0, // strh  r0, [r1]
        0xE1D1_00B0, // ldrh  r0, [r1]
        0xE1D1_20D0, // ldrsb r2, [r1]
        0xE1D1_30F0, // ldrsh r3, [r1]
        0xE3A0_4004, // mov   r4, #4
        0xE3A0_5005, // mov   r5, #5
        0xE1C1_40F8, // strd  r4, r5, [r1, #8]
        0xE1C1_60D8, // ldrd  r6, r7, [r1, #8]
    ]);
    assert_eq!(r(&cpu, 0), 0xFFFF, "a halfword load zero-extends");
    assert_eq!(r(&cpu, 2), 0xFFFF_FFFF, "a signed byte load sign-extends");
    assert_eq!(r(&cpu, 3), 0xFFFF_FFFF, "and so does a signed halfword");
    assert_eq!(
        (r(&cpu, 6), r(&cpu, 7)),
        (4, 5),
        "a doubleword moves a pair"
    );
}

#[test]
fn a_scaled_register_offset_subtracts_when_the_encoding_says_down() {
    let cpu = run(&[
        0xE3A0_0A08, // mov r0, #0x8000
        0xE3A0_1002, // mov r1, #2
        0xE3A0_2007, // mov r2, #7
        0xE780_2101, // str r2, [r0, r1, lsl #2]
        0xE280_0020, // add r0, r0, #32
        0xE710_3101, // ldr r3, [r0, -r1, lsl #2]
    ]);
    assert_eq!(cpu.mem.read_u32(SCRATCH + 8).unwrap(), 7);
    assert_eq!(r(&cpu, 3), 0, "0x8020 - 8 is not where the 7 went");
    assert_eq!(cpu.mem.read_u32(SCRATCH + 0x18).unwrap(), 0);
}

#[test]
fn an_exclusive_store_fails_without_a_matching_load() {
    // With the monitor armed by the load, the store succeeds and reports 0.
    let cpu = run(&[
        0xE3A0_1A08, // mov   r1, #0x8000
        0xE3A0_3007, // mov   r3, #7
        0xE191_0F9F, // ldrex r0, [r1]
        0xE181_2F93, // strex r2, r3, [r1]
    ]);
    assert_eq!(r(&cpu, 2), 0);
    assert_eq!(cpu.mem.read_u32(SCRATCH).unwrap(), 7);

    // Without it, the store reports failure and writes nothing.
    let cpu = run(&[
        0xE3A0_1A08, // mov   r1, #0x8000
        0xE3A0_3007, // mov   r3, #7
        0xE181_2F93, // strex r2, r3, [r1]
    ]);
    assert_eq!(r(&cpu, 2), 1);
    assert_eq!(cpu.mem.read_u32(SCRATCH).unwrap(), 0);
}

#[test]
fn a_byte_swap_exchanges_memory_and_a_register() {
    let cpu = run(&[
        0xE3A0_1A08, // mov r1, #0x8000
        0xE3A0_2063, // mov r2, #0x63
        0xE581_2000, // str r2, [r1]
        0xE3A0_3007, // mov r3, #7
        0xE101_0093, // swp r0, r3, [r1]
    ]);
    assert_eq!(r(&cpu, 0), 0x63, "the old value comes back");
    assert_eq!(cpu.mem.read_u32(SCRATCH).unwrap(), 7);
}
