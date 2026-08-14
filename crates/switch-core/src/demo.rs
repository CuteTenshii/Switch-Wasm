//! The bundled demo homebrew payload.
//!
//! A small hand-assembled AArch64 program that exercises the CPU core:
//! PC-relative addressing, subroutines, arithmetic, bitfield logic, string
//! printing through the UART syscall ABI, a hex dump and a framebuffer paint
//! (a horizontal gradient across the top of the memory-mapped display at
//! [`crate::FB_BASE`]). It is a genuine NRO image (version 1) that the
//! in-browser loader can boot.
//!
//! Program layout (text base maps to [`crate::nro::NRO_BASE`]):
//!
//! ```text
//! 0x00  adr  x0, msg1
//! 0x04  svc  #2                ; print string at x0
//! 0x08  movz x1, #0x1234
//! 0x0c  movz x2, #0x5678
//! 0x10  add  x3, x1, x2        ; 0x68ac
//! 0x14  mov  x0, x3
//! 0x18  bl   print_hex
//! 0x1c  bl   paint             ; fill the framebuffer with a gradient
//! 0x20  svc  #0                ; halt
//! 0x24  print_hex: mov  x9, x0 ; save value (x0 is the syscall arg)
//! 0x28  movz x4, #60
//! 0x2c  loop:   lsrv x5, x9, x4
//! 0x30  and   x5, x5, #0xf
//! 0x34  subs  xzr, x5, #9
//! 0x38  add   x6, x5, #'0'
//! 0x3c  b.le  put
//! 0x40  add   x6, x5, #('a'-0xa)
//! 0x44  put:    mov  x0, x6
//! 0x48  svc   #1               ; putchar
//! 0x4c  subs  x4, x4, #4
//! 0x50  b.pl  loop
//! 0x54  ret
//! 0x58  paint:  movz x8, #0x3F00, lsl 16   ; x8 = FB_BASE
//! 0x5c  movz x9, #80                        ; 80 rows remaining (80..1)
//! 0x60  row:    movz x16, #320              ; 640px / 2px per store
//! 0x64  movz x10, #80
//! 0x68  subs x10, x10, x9                   ; row index = 80 - x9
//! 0x6c  movz x17, #3
//! 0x70  subs x11, x9, #1
//! 0x74  mul   x12, x10, x17                 ; green = row * 3
//! 0x78  mul   x11, x11, x17                 ; red   = (x9-1) * 3
//! 0x7c  movz x13, #128                      ; blue
//! 0x80  movz x14, #0xFF00, lsl 16           ; alpha
//! 0x84  orr   x14, x14, x13, lsl 16         ; | blue << 16
//! 0x88  orr   x14, x14, x12, lsl 8          ; | green << 8
//! 0x8c  orr   x14, x14, x11                 ; | red
//! 0x90  add   x15, x14, x14, lsl 32         ; two identical pixels
//! 0x94  px:     str   x15, [x8], #8
//! 0x98  subs  x16, x16, #1
//! 0x9c  b.ne  px
//! 0xa0  subs  x9, x9, #1
//! 0xa4  b.ne  row
//! 0xa8  ret
//! 0xac  msg1: .asciz "switch-wasm demo\nvalue = "
//! ```
//!
//! The hex dump and gradient paint are cross-checked by unit tests; the
//! console output contract (`demo_runs_and_prints_hex`) must not change.

use crate::nro::{NRO0_MAGIC, NRO_BASE};

const TEXT_OFF: u32 = 0x50;
const MSG_OFF: u32 = 0xac;
const TEXT_SIZE: u32 = 0x100;
const MSG: &[u8] = b"switch-wasm demo\nvalue = \0";

const GRADIENT_ROWS: u32 = 80;
const FB_COLS: u32 = crate::FB_WIDTH / 2; // 2 pixels per 64-bit store

// ---- tiny A64 encoders (only what the demo needs) ----

fn adr(rd: u32, imm: i64) -> u32 {
    let imm = imm as u32;
    0b10000 << 24 | ((imm & 0b11) << 29) | (((imm >> 2) & 0x7_FFFF) << 5) | rd
}

fn svc(imm: u32) -> u32 {
    0xD400_0000 | (imm << 5) | 1
}

fn movz(rd: u32, imm16: u32) -> u32 {
    0xD280_0000 | (imm16 << 5) | rd
}

/// MOVZ with a logical-shift-left of `shift` bits (16, 32 or 48).
fn movz_lsl(rd: u32, imm16: u32, shift: u32) -> u32 {
    0xD280_0000 | ((shift / 16) << 21) | (imm16 << 5) | rd
}

fn add_imm(rd: u32, rn: u32, imm12: u32) -> u32 {
    0x9100_0000 | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd
}

fn add_reg(rd: u32, rn: u32, rm: u32) -> u32 {
    0x8B00_0000 | (rm << 16) | (rn << 5) | rd
}

/// ADD Xd, Xn, Xm, LSL #sh.
fn add_shift(rd: u32, rn: u32, rm: u32, sh: u32) -> u32 {
    0x8B00_0000 | (rm << 16) | (sh << 10) | (rn << 5) | rd
}

/// SUBS Xd, Xn, Xm (shifted register, LSL #0).
fn subs_reg(rd: u32, rn: u32, rm: u32) -> u32 {
    0xEB00_0000 | (rm << 16) | (rn << 5) | rd
}

fn mov_reg(rd: u32, rm: u32) -> u32 {
    0xAA00_0000 | (rm << 16) | 0x3E0 | rd
}

/// ORR Xd, Xn, Xm, LSL #sh.
fn orr_shift(rd: u32, rn: u32, rm: u32, sh: u32) -> u32 {
    0xAA00_0000 | (rm << 16) | (sh << 10) | (rn << 5) | rd
}

/// MUL Xd, Xn, Xm  ==  MADD Xd, Xn, Xm, XZR.
fn mul(rd: u32, rn: u32, rm: u32) -> u32 {
    0x9B00_0000 | (rm << 16) | (31 << 10) | (rn << 5) | rd
}

/// STR Xt, [Xn], #imm (post-index).
fn str_x_post(rt: u32, rn: u32, imm: u32) -> u32 {
    0xF800_0000 | (imm << 12) | (0b01 << 10) | (rn << 5) | rt
}

fn bl(off: i64) -> u32 {
    0x9400_0000 | (((off >> 2) as u32) & 0x3FF_FFFF)
}

fn b_cond(cond: u32, off: i64) -> u32 {
    0x5400_0000 | ((((off >> 2) as u32) & 0x7_FFFF) << 5) | 0x10 | cond
}

fn lsrv(rd: u32, rn: u32, rm: u32) -> u32 {
    0x9AC4_0000 | (rm << 16) | (0b001001 << 10) | (rn << 5) | rd
}

/// AND Xd, Xn, #mask — N=1 form so the mask is a single 64-bit element
/// (e.g. `#0xf` rather than the replicated `#0xf0000000f`).
fn and_imm(rd: u32, rn: u32, imms: u32) -> u32 {
    0x9240_0000 | (imms << 10) | (rn << 5) | rd
}

fn subs_imm(rd: u32, rn: u32, imm12: u32) -> u32 {
    0xF100_0000 | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd
}

fn ret() -> u32 {
    0xD65F_03C0
}

/// Assemble the demo program and wrap it in an NRO v1 container.
pub fn demo_nro() -> Vec<u8> {
    let mut text = Vec::with_capacity(TEXT_SIZE as usize);

    let msg_addr = NRO_BASE.wrapping_add(MSG_OFF);
    text.extend_from_slice(&adr(0, (msg_addr - NRO_BASE) as i64).to_le_bytes()); // 0x00
    text.extend_from_slice(&svc(2).to_le_bytes());                              // 0x04
    text.extend_from_slice(&movz(1, 0x1234).to_le_bytes());                     // 0x08
    text.extend_from_slice(&movz(2, 0x5678).to_le_bytes());                     // 0x0c
    text.extend_from_slice(&add_reg(3, 1, 2).to_le_bytes());                    // 0x10
    text.extend_from_slice(&mov_reg(0, 3).to_le_bytes());                       // 0x14
    text.extend_from_slice(&bl(0x24 - 0x18).to_le_bytes());                     // 0x18 bl print_hex
    text.extend_from_slice(&bl(0x58 - 0x1c).to_le_bytes());                     // 0x1c bl paint
    text.extend_from_slice(&svc(0).to_le_bytes());                              // 0x20

    // print_hex at 0x24
    text.extend_from_slice(&mov_reg(9, 0).to_le_bytes());                      // 0x24
    text.extend_from_slice(&movz(4, 60).to_le_bytes());                        // 0x28
    // loop at 0x2c
    text.extend_from_slice(&lsrv(5, 9, 4).to_le_bytes());                      // 0x2c
    text.extend_from_slice(&and_imm(5, 5, 3).to_le_bytes());                   // 0x30 (mask 0xf)
    text.extend_from_slice(&subs_imm(31, 5, 9).to_le_bytes());                 // 0x34
    text.extend_from_slice(&add_imm(6, 5, b'0' as u32).to_le_bytes());         // 0x38
    text.extend_from_slice(&b_cond(0xD, 0x44 - 0x3c).to_le_bytes());           // 0x3c (LE)
    text.extend_from_slice(&add_imm(6, 5, (b'a' - 0xA) as u32).to_le_bytes()); // 0x40
    // put at 0x44
    text.extend_from_slice(&mov_reg(0, 6).to_le_bytes());                      // 0x44
    text.extend_from_slice(&svc(1).to_le_bytes());                             // 0x48
    text.extend_from_slice(&subs_imm(4, 4, 4).to_le_bytes());                  // 0x4c
    text.extend_from_slice(&b_cond(0x5, 0x2c - 0x50).to_le_bytes());           // 0x50 (PL)
    text.extend_from_slice(&ret().to_le_bytes());                              // 0x54

    // paint at 0x58: horizontal gradient across the top of the framebuffer.
    // x9 counts the remaining rows (80..1); the painted row index is 80 - x9.
    text.extend_from_slice(&movz_lsl(8, 0x3F00, 16).to_le_bytes());            // 0x58 x8 = FB_BASE
    text.extend_from_slice(&movz(9, GRADIENT_ROWS).to_le_bytes());             // 0x5c
    // row_loop at 0x60
    text.extend_from_slice(&movz(16, FB_COLS).to_le_bytes());                  // 0x60
    text.extend_from_slice(&movz(10, GRADIENT_ROWS).to_le_bytes());            // 0x64
    text.extend_from_slice(&subs_reg(10, 10, 9).to_le_bytes());                // 0x68 row = 80 - x9
    text.extend_from_slice(&movz(17, 3).to_le_bytes());                        // 0x6c
    text.extend_from_slice(&subs_imm(11, 9, 1).to_le_bytes());                 // 0x70 x9 - 1
    text.extend_from_slice(&mul(12, 10, 17).to_le_bytes());                    // 0x74 green = row * 3
    text.extend_from_slice(&mul(11, 11, 17).to_le_bytes());                    // 0x78 red = (x9-1) * 3
    text.extend_from_slice(&movz(13, 128).to_le_bytes());                      // 0x7c blue
    text.extend_from_slice(&movz_lsl(14, 0xFF00, 16).to_le_bytes());           // 0x80 alpha
    text.extend_from_slice(&orr_shift(14, 14, 13, 16).to_le_bytes());          // 0x84 | blue << 16
    text.extend_from_slice(&orr_shift(14, 14, 12, 8).to_le_bytes());           // 0x88 | green << 8
    text.extend_from_slice(&orr_shift(14, 14, 11, 0).to_le_bytes());           // 0x8c | red
    text.extend_from_slice(&add_shift(15, 14, 14, 32).to_le_bytes());          // 0x90 pair of pixels
    // px at 0x94
    text.extend_from_slice(&str_x_post(15, 8, 8).to_le_bytes());               // 0x94
    text.extend_from_slice(&subs_imm(16, 16, 1).to_le_bytes());                // 0x98
    text.extend_from_slice(&b_cond(0x1, 0x94 - 0x9c).to_le_bytes());           // 0x9c (NE)
    text.extend_from_slice(&subs_imm(9, 9, 1).to_le_bytes());                  // 0xa0
    text.extend_from_slice(&b_cond(0x1, 0x60 - 0xa4).to_le_bytes());           // 0xa4 (NE)
    text.extend_from_slice(&ret().to_le_bytes());                              // 0xa8

    // message at 0xac
    text.extend_from_slice(MSG);
    text.resize(TEXT_SIZE as usize, 0);

    build_nro_v1(&text)
}

fn build_nro_v1(text: &[u8]) -> Vec<u8> {
    let data_off = TEXT_OFF + text.len() as u32;
    let mut out = vec![0u8; data_off as usize];
    out[0..4].copy_from_slice(&NRO0_MAGIC.to_le_bytes());
    out[4..8].copy_from_slice(&1u32.to_le_bytes()); // version 1
    let total = out.len() as u32;
    out[8..12].copy_from_slice(&total.to_le_bytes());
    out[0x10..0x14].copy_from_slice(&TEXT_OFF.to_le_bytes());
    out[0x14..0x18].copy_from_slice(&(text.len() as u32).to_le_bytes());
    out[0x20..0x24].copy_from_slice(&data_off.to_le_bytes());
    out[0x24..0x28].copy_from_slice(&0u32.to_le_bytes());
    out[0x28..0x2C].copy_from_slice(&0x1000u32.to_le_bytes()); // bss
    out[TEXT_OFF as usize..].copy_from_slice(text);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::{Cpu, SyscallMode};
    use crate::nro::load_nro;

    #[test]
    fn demo_runs_and_prints_hex() {
        let nro = demo_nro();
        let mut cpu = Cpu::new();
        cpu.syscall_mode = SyscallMode::Uart;
        let loaded = load_nro(&mut cpu.mem, &nro).unwrap();
        cpu.set_pc(loaded.entry);
        let report = cpu.run(100_000).unwrap();
        assert!(report.halted);
        let out = String::from_utf8_lossy(&cpu.out);
        assert!(out.starts_with("switch-wasm demo\nvalue = "));
        assert!(out.ends_with("00000000000068ac"), "out = {out:?}");
    }

    #[test]
    fn demo_nro_parses() {
        let nro = demo_nro();
        let mut cpu = Cpu::new();
        let loaded = load_nro(&mut cpu.mem, &nro).unwrap();
        assert_eq!(loaded.entry, crate::nro::NRO_BASE);
        assert!(loaded.is_64bit);
    }

    #[test]
    fn demo_paints_framebuffer_gradient() {
        let nro = demo_nro();
        let mut cpu = Cpu::new();
        cpu.syscall_mode = SyscallMode::Uart;
        let loaded = load_nro(&mut cpu.mem, &nro).unwrap();
        cpu.set_pc(loaded.entry);
        let report = cpu.run(100_000).unwrap();
        assert!(report.halted);

        // The painted band is the top GRADIENT_ROWS rows, fully opaque.
        let mut buf = vec![0u8; (crate::FB_WIDTH * GRADIENT_ROWS * 4) as usize];
        cpu.mem.read_into(crate::FB_BASE, &mut buf).unwrap();
        for px in buf.chunks(4) {
            assert_eq!(px[3], 0xFF, "every painted pixel must be opaque");
        }

        // First painted row (x9=80): red=(80-1)*3=237, green=0, blue=128.
        let mut first = [0u8; 4];
        cpu.mem.read_into(crate::FB_BASE, &mut first).unwrap();
        assert_eq!(first, [237, 0, 128, 0xFF]);

        // Last painted row (x9=1): red=0, green=(80-1)*3=237, blue=128.
        let last_row = crate::FB_BASE + (GRADIENT_ROWS - 1) * crate::FB_STRIDE;
        let mut last = [0u8; 4];
        cpu.mem.read_into(last_row, &mut last).unwrap();
        assert_eq!(last, [0, 237, 128, 0xFF]);

        // The row below the band is untouched (still zeroed).
        let below = crate::FB_BASE + GRADIENT_ROWS * crate::FB_STRIDE;
        let mut zero = [0u8; 4];
        cpu.mem.read_into(below, &mut zero).unwrap();
        assert_eq!(zero, [0, 0, 0, 0]);
    }
}
