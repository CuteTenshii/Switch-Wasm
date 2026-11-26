//! CRC32, AES, SHA and polynomial multiply.

mod cpu;

use cpu::*;

#[test]
fn crc32_accumulates_over_the_bytes_of_its_operand() {
    // The check value of "123456789", as one doubleword plus a byte. Without the
    // final inversion these are the complements of 0xCBF43926 and 0xE3069283.
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

/// AESE then AESMC is one AES round bar the key schedule; FIPS-197's round-1 vector.
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

/// Operands with data in every lane; expected values from an Apple M-series core.
const SHA_A: u128 = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210;
const SHA_B: u128 = 0x0f1e_2d3c_4b5a_6978_8796_a5b4_c3d2_e1f0;
const SHA_C: u128 = 0x1357_9bdf_2468_ace0_dead_beef_cafe_f00d;

#[test]
fn the_sha1_instructions_decode_and_compute() {
    let mut cpu = cpu_at(0x1000);
    cpu.set_vreg(0, SHA_A);
    cpu.set_vreg(1, u128::MAX);
    let cpu = run_program(cpu, 0x1000, &[sha2(0b00000, 1, 0)]);
    assert_eq!(
        cpu.read_vreg(1),
        0x1d95_0c84,
        "sha1h is a 30-bit rotate of the low word into a cleared register"
    );

    let mut cpu = cpu_at(0x1000);
    for (i, v) in [(0u8, SHA_A), (1, SHA_B), (2, SHA_C)] {
        cpu.set_vreg(i, v);
    }
    let cpu = run_program(cpu, 0x1000, &[sha3(0b011, 0, 1, 2)]);
    assert_eq!(
        cpu.read_vreg(0),
        0x95e2_7b0c_6e11_80ff_2152_4110_3501_0ff2,
        "sha1su0"
    );
}

/// SHA256H and SHA256H2 keep opposite halves of the same four rounds, and
/// take the two halves of the state in opposite operand order.
#[test]
fn the_sha256_round_instructions_keep_opposite_halves() {
    let mut cpu = cpu_at(0x1000);
    for (i, v) in [(0u8, SHA_A), (1, SHA_B), (2, SHA_C), (3, SHA_B), (4, SHA_A)] {
        cpu.set_vreg(i, v);
    }
    let cpu = run_program(
        cpu,
        0x1000,
        &[
            sha3(0b100, 0, 1, 2), // SHA256H  q0, q1, v2.4s
            sha3(0b101, 3, 4, 2), // SHA256H2 q3, q4, v2.4s
        ],
    );
    assert_eq!(
        cpu.read_vreg(0),
        0x56db_4b4f_3866_0fc4_3cc9_10bc_c623_274d,
        "sha256h"
    );
    assert_eq!(
        cpu.read_vreg(3),
        0x7b42_d622_6854_4a49_dd48_6ff0_511a_fafd,
        "sha256h2"
    );
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
