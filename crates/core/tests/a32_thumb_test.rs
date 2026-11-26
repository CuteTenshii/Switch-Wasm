//! Thumb-2 execution and ARM/Thumb interworking. Encodings come from
//! `llvm-mc -triple=thumbv7-none-eabi` and results match `qemu-system-arm` (`tools/thumb_difftest.py`).

mod a32;

use a32::{r, BASE, HALT};

/// An ARM prologue that branches into Thumb code at `BASE + 8`.
const ENTER_THUMB: [u32; 2] = [0xE28F_C001, 0xE12F_FF1C]; // add r12, pc, #1; bx r12

fn run_thumb(code: &[u8]) -> switch_core::cpu::Cpu {
    let mut cpu = a32::cpu();
    let mut bytes = Vec::new();
    for insn in ENTER_THUMB {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    bytes.extend_from_slice(code);
    bytes.extend_from_slice(&HALT.to_le_bytes());
    cpu.mem.map(u64::from(BASE), &bytes).unwrap();
    cpu.run(1000).unwrap();
    cpu
}

#[test]
fn it_block_table_branch_interworking_and_wide_ops() {
    // movs r0,#5; cmp r0,#5; ite eq; moveq r1,#1; movne r1,#2; movs r4,#1; tbb [pc,r4]; ...
    let code: [u8; 64] = [
        0x05, 0x20, 0x05, 0x28, 0x0c, 0xbf, 0x01, 0x21, 0x02, 0x21, 0x01, 0x24, 0xdf, 0xe8, 0x04,
        0xf0, 0x01, 0x02, 0x0a, 0x22, 0x14, 0x23, 0x01, 0xa5, 0x00, 0x35, 0x28, 0x47, 0x07, 0x60,
        0xa0, 0xe3, 0x04, 0x70, 0x8f, 0xe2, 0x01, 0x70, 0x87, 0xe3, 0x17, 0xff, 0x2f, 0xe1, 0xdf,
        0xf8, 0x0c, 0x80, 0x03, 0xb4, 0xbd, 0xe8, 0x00, 0x06, 0xa8, 0xfb, 0x08, 0xbc, 0x00, 0xdf,
        0x78, 0x56, 0x34, 0x12,
    ];
    let cpu = run_thumb(&code);
    assert_eq!(r(&cpu, 0), 5);
    assert_eq!(r(&cpu, 1), 1);
    assert_eq!(r(&cpu, 2), 0);
    assert_eq!(r(&cpu, 3), 20);
    assert_eq!(r(&cpu, 6), 7);
    assert_eq!(r(&cpu, 8), 0x1234_5678);
    assert_eq!(r(&cpu, 9), 5);
    assert_eq!(r(&cpu, 10), 1);
    let product = 0x1234_5678u64 * 0x1234_5678;
    assert_eq!(r(&cpu, 11), product as u32);
    assert_eq!(r(&cpu, 12), (product >> 32) as u32);
}
