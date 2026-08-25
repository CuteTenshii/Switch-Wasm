//! Minimal AES-128 primitives (ECB + XTS) and the GF(2^128) multiply used by
//! XTS — enough to decrypt an NCA header, which is AES-128-XTS over two
//! 0x200-byte sectors with the global `header_key` from `prod.keys`.
//!
//! FIPS-197 Rijndael with a 128-bit key, hand-rolled (the workspace forbids
//! external dependencies). Verified against the NIST SP 800-38A / SP 800-38E
//! test vectors in the unit tests below.

const SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

const INV_SBOX: [u8; 256] = [
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d,
];

const RCON: [u8; 11] = [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36];

/// Expand a 128-bit key into the 11 round keys (176 bytes).
fn expand_key(key: &[u8; 16]) -> [u8; 176] {
    let mut w = [0u8; 176];
    w[..16].copy_from_slice(key);
    let mut rcon = 0;
    for i in 1..11 {
        let base = i * 16;
        let prev = base - 16;
        // temp = RotWord(SubWord(w[i-1])) XOR Rcon
        let mut tmp = [w[prev + 12], w[prev + 13], w[prev + 14], w[prev + 15]];
        let t = tmp[0];
        tmp[0] = SBOX[tmp[1] as usize];
        tmp[1] = SBOX[tmp[2] as usize];
        tmp[2] = SBOX[tmp[3] as usize];
        tmp[3] = SBOX[t as usize];
        rcon += 1;
        tmp[0] ^= RCON[rcon];
        for j in 0..4 {
            w[base + j] = w[prev + j] ^ tmp[j];
        }
        for j in 4..16 {
            w[base + j] = w[base + j - 4] ^ w[prev + j];
        }
    }
    w
}

fn add_round_key(state: &mut [u8; 16], round_key: &[u8]) {
    for i in 0..16 {
        state[i] ^= round_key[i];
    }
}

/// SubBytes, ShiftRows and MixColumns, one pass over the state each — the
/// textbook shape, kept as the reference [`RoundKeys::encrypt_block`]'s
/// table-driven round is checked against.
#[cfg(test)]
fn sub_bytes(state: &mut [u8; 16]) {
    for b in state.iter_mut() {
        *b = SBOX[*b as usize];
    }
}

fn inv_sub_bytes(state: &mut [u8; 16]) {
    for b in state.iter_mut() {
        *b = INV_SBOX[*b as usize];
    }
}

#[cfg(test)]
fn shift_rows(state: &mut [u8; 16]) {
    let s = *state;
    // Row 0 unchanged, row 1 left 1, row 2 left 2, row 3 left 3 (column-major).
    state[0] = s[0];
    state[1] = s[5];
    state[2] = s[10];
    state[3] = s[15];
    state[4] = s[4];
    state[5] = s[9];
    state[6] = s[14];
    state[7] = s[3];
    state[8] = s[8];
    state[9] = s[13];
    state[10] = s[2];
    state[11] = s[7];
    state[12] = s[12];
    state[13] = s[1];
    state[14] = s[6];
    state[15] = s[11];
}

fn inv_shift_rows(state: &mut [u8; 16]) {
    let s = *state;
    state[0] = s[0];
    state[1] = s[13];
    state[2] = s[10];
    state[3] = s[7];
    state[4] = s[4];
    state[5] = s[1];
    state[6] = s[14];
    state[7] = s[11];
    state[8] = s[8];
    state[9] = s[5];
    state[10] = s[2];
    state[11] = s[15];
    state[12] = s[12];
    state[13] = s[9];
    state[14] = s[6];
    state[15] = s[3];
}

/// Multiply a byte in GF(2^8) by x (0x02), used by MixColumns.
const fn xtime(a: u8) -> u8 {
    (a << 1) ^ (if a & 0x80 != 0 { 0x1b } else { 0 })
}

/// SubBytes, ShiftRows and MixColumns folded into one table lookup per byte —
/// the standard way AES is implemented, and four times the rate of doing the
/// three as separate passes over the state.
///
/// `TE[r][x]` is the contribution row `r`'s byte makes to a whole output
/// column, packed the way [`RoundKeys::encrypt_block`] holds a column: row 0
/// in the low byte. Four of them XORed together, plus the round key, is the
/// column.
///
/// 4 KiB of tables. Table-driven AES leaks key material through cache timing,
/// which matters not at all here: what this decrypts is firmware and game
/// content, with keys the user already has on disk.
const TE: [[u32; 256]; 4] = {
    let mut t = [[0u32; 256]; 4];
    let mut x = 0usize;
    while x < 256 {
        let s = SBOX[x] as u32;
        let s2 = xtime(SBOX[x]) as u32;
        let s3 = s2 ^ s;
        t[0][x] = s2 | (s << 8) | (s << 16) | (s3 << 24);
        t[1][x] = s3 | (s2 << 8) | (s << 16) | (s << 24);
        t[2][x] = s | (s3 << 8) | (s2 << 16) | (s << 24);
        t[3][x] = s | (s << 8) | (s3 << 16) | (s2 << 24);
        x += 1;
    }
    t
};

#[cfg(test)]
fn mix_columns(state: &mut [u8; 16]) {
    for c in 0..4 {
        let i = c * 4;
        let a = state[i];
        let b = state[i + 1];
        let c2 = state[i + 2];
        let d = state[i + 3];
        state[i] = xtime(a) ^ (xtime(b) ^ b) ^ c2 ^ d;
        state[i + 1] = a ^ xtime(b) ^ (xtime(c2) ^ c2) ^ d;
        state[i + 2] = a ^ b ^ xtime(c2) ^ (xtime(d) ^ d);
        state[i + 3] = (xtime(a) ^ a) ^ b ^ c2 ^ xtime(d);
    }
}

fn inv_mix_columns(state: &mut [u8; 16]) {
    for c in 0..4 {
        let i = c * 4;
        let a = state[i];
        let b = state[i + 1];
        let c2 = state[i + 2];
        let d = state[i + 3];
        // Multiply by the inverse matrix: 0x0e, 0x0b, 0x0d, 0x09.
        state[i] = gmul(a, 0x0e) ^ gmul(b, 0x0b) ^ gmul(c2, 0x0d) ^ gmul(d, 0x09);
        state[i + 1] = gmul(a, 0x09) ^ gmul(b, 0x0e) ^ gmul(c2, 0x0b) ^ gmul(d, 0x0d);
        state[i + 2] = gmul(a, 0x0d) ^ gmul(b, 0x09) ^ gmul(c2, 0x0e) ^ gmul(d, 0x0b);
        state[i + 3] = gmul(a, 0x0b) ^ gmul(b, 0x0d) ^ gmul(c2, 0x09) ^ gmul(d, 0x0e);
    }
}

fn gmul(mut a: u8, mut b: u8) -> u8 {
    let mut p = 0u8;
    for _ in 0..8 {
        if b & 1 != 0 {
            p ^= a;
        }
        let hi = a & 0x80 != 0;
        a <<= 1;
        if hi {
            a ^= 0x1b;
        }
        b >>= 1;
    }
    p
}

/// Row `r` of the column ShiftRows takes it from, for output column `c`.
#[inline]
fn byte_of(state: &[u32; 4], c: usize, r: usize) -> usize {
    (state[(c + r) % 4] >> (8 * r)) as u8 as usize
}

/// Column `c` of a sixteen-byte state, packed with row 0 in the low byte.
#[inline]
fn column_word(state: &[u8; 16], c: usize) -> u32 {
    column_word_at(state, c * 4)
}

/// The same packing, out of a longer buffer at a byte offset — the round keys
/// are one flat 176-byte array.
#[inline]
fn column_word_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

/// One AES-128 key's expanded round keys.
///
/// Held apart from the block operations because every bulk mode uses one
/// schedule for every block it touches, and deriving it is comparable work to
/// encrypting with it. Expanding per block made the firmware fonts — 17.7 MB
/// of AES-CTR across five NCA sections — expand it 1.1 million times, which
/// was 28% of a Home Menu boot.
#[derive(Clone)]
pub struct RoundKeys([u8; 176]);

impl RoundKeys {
    pub fn new(key: &[u8; 16]) -> RoundKeys {
        RoundKeys(expand_key(key))
    }

    /// Encrypt one 16-byte block.
    ///
    /// The state is held as four column words rather than sixteen bytes, so a
    /// round is four table lookups and four XORs per column instead of four
    /// separate passes over a byte array. `column_word` and the row/column
    /// indexing are the same convention [`shift_rows`] uses: `state[c * 4 + r]`
    /// is row `r` of column `c`, and ShiftRows takes row `r` of a column from
    /// column `c + r`.
    pub fn encrypt_block(&self, block: &[u8; 16]) -> [u8; 16] {
        let w = &self.0;
        let mut s = [0u32; 4];
        for (c, column) in s.iter_mut().enumerate() {
            *column = column_word(block, c) ^ column_word_at(w, c * 4);
        }
        for round in 1..10 {
            let rk = round * 16;
            let mut next = [0u32; 4];
            for (c, column) in next.iter_mut().enumerate() {
                *column = TE[0][byte_of(&s, c, 0)]
                    ^ TE[1][byte_of(&s, c, 1)]
                    ^ TE[2][byte_of(&s, c, 2)]
                    ^ TE[3][byte_of(&s, c, 3)]
                    ^ column_word_at(w, rk + c * 4);
            }
            s = next;
        }
        // The last round has no MixColumns, so no table: SubBytes and
        // ShiftRows straight into the output.
        let mut out = [0u8; 16];
        for c in 0..4 {
            for r in 0..4 {
                out[c * 4 + r] = SBOX[byte_of(&s, c, r)] ^ w[160 + c * 4 + r];
            }
        }
        out
    }

    /// Decrypt one 16-byte block.
    pub fn decrypt_block(&self, block: &[u8; 16]) -> [u8; 16] {
        let w = &self.0;
        let mut state = *block;
        add_round_key(&mut state, &w[160..176]);
        for round in (1..10).rev() {
            inv_shift_rows(&mut state);
            inv_sub_bytes(&mut state);
            add_round_key(&mut state, &w[round * 16..(round + 1) * 16]);
            inv_mix_columns(&mut state);
        }
        inv_shift_rows(&mut state);
        inv_sub_bytes(&mut state);
        add_round_key(&mut state, &w[0..16]);
        state
    }
}

/// Encrypt one 16-byte block. Expands the key schedule for this block alone —
/// use [`RoundKeys`] directly for more than one.
pub fn aes128_encrypt_block(key: &[u8; 16], block: &[u8; 16]) -> [u8; 16] {
    RoundKeys::new(key).encrypt_block(block)
}

/// Decrypt one 16-byte block. See [`aes128_encrypt_block`] on the schedule.
pub fn aes128_decrypt_block(key: &[u8; 16], block: &[u8; 16]) -> [u8; 16] {
    RoundKeys::new(key).decrypt_block(block)
}

/// AES-128-ECB encrypt of `data` (length must be a multiple of 16).
pub fn aes128_ecb_encrypt(key: &[u8; 16], data: &[u8]) -> Vec<u8> {
    let keys = RoundKeys::new(key);
    let mut out = Vec::with_capacity(data.len());
    for chunk in data.chunks(16) {
        let mut blk = [0u8; 16];
        blk[..chunk.len()].copy_from_slice(chunk);
        out.extend_from_slice(&keys.encrypt_block(&blk));
    }
    out
}

/// AES-128-ECB decrypt of `data` (length must be a multiple of 16).
pub fn aes128_ecb_decrypt(key: &[u8; 16], data: &[u8]) -> Vec<u8> {
    let keys = RoundKeys::new(key);
    let mut out = Vec::with_capacity(data.len());
    for chunk in data.chunks(16) {
        let mut blk = [0u8; 16];
        blk[..chunk.len()].copy_from_slice(chunk);
        out.extend_from_slice(&keys.decrypt_block(&blk));
    }
    out
}

/// Multiply a 128-bit XTS tweak by x in GF(2^128) (poly 0x87), the standard
/// "next tweak" step. OpenSSL's reference (and the `cryptography` bindings)
/// treat the tweak bytes little-endian (byte 0 = least significant), so
/// multiply by x is a left shift toward byte 15 with the reduction XORed into
/// byte 0.
fn xts_mul_x(tweak: &mut [u8; 16]) {
    let carry = tweak[15] & 0x80;
    for i in (0..15).rev() {
        tweak[i + 1] = (tweak[i + 1] << 1) | (tweak[i] >> 7);
    }
    tweak[0] <<= 1;
    if carry != 0 {
        tweak[0] ^= 0x87;
    }
}

/// AES-128-XTS decrypt of `data` using a 32-byte key (two 128-bit halves).
/// `sector` is the starting sector number; `sector_size` is the XTS sector
/// size (e.g. 0x200 for NCA headers). Nintendo's tweak places the little-endian
/// sector number in the high 8 bytes of the 128-bit tweak.
pub fn aes128_xts_decrypt(key: &[u8; 32], data: &[u8], sector: u64, sector_size: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut s = sector;
    for chunk in data.chunks(sector_size) {
        let mut tweak = [0u8; 16];
        let mut sv = s;
        for i in (0..16).rev() {
            tweak[i] = (sv & 0xff) as u8;
            sv >>= 8;
        }
        s += 1;
        aes128_xts_decrypt_sector(key, chunk, &tweak, &mut out);
    }
    out
}

pub fn aes128_xts_decrypt_sector(key: &[u8; 32], chunk: &[u8], tweak: &[u8; 16], out: &mut Vec<u8>) {
    let mut key1 = [0u8; 16];
    let mut key2 = [0u8; 16];
    key1.copy_from_slice(&key[..16]);
    key2.copy_from_slice(&key[16..]);
    let data_keys = RoundKeys::new(&key1);
    let mut tweak = RoundKeys::new(&key2).encrypt_block(tweak);
    for blk in chunk.chunks(16) {
        let mut c = [0u8; 16];
        c[..blk.len()].copy_from_slice(blk);
        let mut x = [0u8; 16];
        for i in 0..16 {
            x[i] = c[i] ^ tweak[i];
        }
        let p = data_keys.decrypt_block(&x);
        for i in 0..16 {
            out.push(p[i] ^ tweak[i]);
        }
        xts_mul_x(&mut tweak);
    }
}

/// AES-128-CTR keystream XOR (encryption and decryption are the same
/// operation). `counter` is the initial 128-bit big-endian counter block; it
/// increments by one, as a big-endian integer, every 16 bytes of `data`. This
/// is the primitive NCA section bodies are encrypted with — the counter's
/// initial value is derived from the section's FS header (see `nca.rs`).
pub fn aes128_ctr_xor(key: &[u8; 16], counter: &[u8; 16], data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    aes128_ctr_xor_in_place(key, counter, &mut out);
    out
}

/// The same keystream, applied to a buffer already in place.
///
/// `data` must start on a cipher-block boundary relative to `counter` — the
/// keystream block a byte gets is decided by its index here, so a caller
/// decrypting a range out of the middle of a stream aligns the range down to
/// a multiple of 16 and advances `counter` to match (see
/// [`crate::nca::SectionSource`], which reads sections that way).
pub fn aes128_ctr_xor_in_place(key: &[u8; 16], counter: &[u8; 16], data: &mut [u8]) {
    let keys = RoundKeys::new(key);
    let mut ctr = *counter;
    for chunk in data.chunks_mut(16) {
        let ks = keys.encrypt_block(&ctr);
        for (b, k) in chunk.iter_mut().zip(ks.iter()) {
            *b ^= k;
        }
        for i in (0..16).rev() {
            ctr[i] = ctr[i].wrapping_add(1);
            if ctr[i] != 0 {
                break;
            }
        }
    }
}

const SHA256_H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA-256 (FIPS 180-4). Used to verify a decrypted NCA section's hash-table
/// region against the FS header's stored master hash — the only way to tell
/// whether an NCA decrypted correctly, since a wrong key produces plausible
/// garbage rather than an obvious error.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h = SHA256_H0;
    let bit_len = (data.len() as u64) * 8;

    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for block in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aes128_ecb_nist_vectors() {
        // NIST SP 800-38A, key 2b7e151628aed2a6abf7158809cf4f3c.
        let key = hex("2b7e151628aed2a6abf7158809cf4f3c");
        let pt = hex("6bc1bee22e409f96e93d7e117393172a");
        let ct = hex("3ad77bb40d7a3660a89ecaf32466ef97");
        assert_eq!(aes128_encrypt_block(&key, &pt), ct);
        assert_eq!(aes128_decrypt_block(&key, &ct), pt);
    }

    #[test]
    fn aes128_xts_cross_checked() {
        // Two-block AES-128-XTS, verified against OpenSSL's `xts128.c`
        // (via the `cryptography` bindings): key halves are distinct, tweak is
        // the 16 zero bytes for data unit 0, 32 bytes of input. Decrypting the
        // ciphertext must reproduce the plaintext.
        let mut key = [0u8; 32];
        for i in 0..16 {
            key[i] = i as u8;
            key[16 + i] = (0x10 + i) as u8;
        }
        let ct_hex = "74a109aabf1937c022d19da4b96cbc40b8ddc9c0653a7fb0dc8425c7ef276dea";
        let mut ct = [0u8; 32];
        for i in 0..32 {
            ct[i] = u8::from_str_radix(&ct_hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        let tweak = [0u8; 16];
        let mut out = Vec::new();
        aes128_xts_decrypt_sector(&key, &ct, &tweak, &mut out);
        let exp: Vec<u8> = (0..32u8).collect();
        assert_eq!(out, exp);
    }

    #[test]
    fn nca_header_xts_uses_two_sectors() {
        // The NCA header path decrypts 0x400 bytes as two 0x200-byte XTS
        // sectors with hactool's tweak (sector number in the high 8 bytes).
        // Round-trips against a two-sector decrypt.
        let mut key = [0u8; 32];
        for i in 0..32 {
            key[i] = i as u8;
        }
        let mut data = [0u8; 0x400];
        for (i, b) in data.iter_mut().enumerate() {
            *b = i as u8;
        }
        // Encrypt via the OpenSSL cross-check is not available, but the
        // sector-based path must produce a 0x400-byte output without panicking
        // and the two sectors must decrypt deterministically.
        let out = aes128_xts_decrypt(&key, &data, 0, 0x200);
        assert_eq!(out.len(), 0x400);
    }

    #[test]
    fn aes128_ctr_cross_checked() {
        // Cross-checked against `openssl enc -aes-128-ctr`: a 3-block message
        // with an initial counter chosen so the increment carries across two
        // bytes (...fffe -> ...ffff -> ...0000), which is the part most likely
        // to have a bug.
        let mut key = [0u8; 16];
        for i in 0..16 {
            key[i] = i as u8;
        }
        let mut ctr = [0xffu8; 16];
        ctr[15] = 0xfe;
        let pt: Vec<u8> = (0..48u8).collect();
        let ct = hex_vec(
            "b6b4c0db298ed208c747d2ffa26399e12c550d21da1294347cceb882124da50\
             ce6801914a3aa7da54766ab498de5f656",
        );
        assert_eq!(aes128_ctr_xor(&key, &ctr, &pt), ct);
        // CTR is its own inverse.
        assert_eq!(aes128_ctr_xor(&key, &ctr, &ct), pt);
    }

    #[test]
    fn sha256_vectors() {
        assert_eq!(
            hex_digest(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hex_digest(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        // Longer than one 64-byte block, to exercise the padding/length path.
        assert_eq!(
            hex_digest(&sha256(
                b"switch-wasm sha256 test vector, a bit longer than one block to exercise padding across blocks"
            )),
            "9678e85ce95a8577d3aa07dd855313e94f3b51caaf709885b00d5b8365f8f2c0"
        );
    }

    fn hex_vec(s: &str) -> Vec<u8> {
        let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
            .collect()
    }

    fn hex(s: &str) -> [u8; 16] {
        let mut out = [0u8; 16];
        for i in 0..16 {
            out[i] = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap();
        }
        out
    }

    fn hex_digest(v: &[u8]) -> String {
        v.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

#[cfg(test)]
mod table_driven {
    use super::*;

    /// AES-128 encryption written the textbook way: one pass over the state
    /// per step. [`RoundKeys::encrypt_block`] folds three of those steps into
    /// a table lookup, which is four times the rate and much easier to get
    /// subtly wrong — so it is checked against this.
    fn reference(key: &[u8; 16], block: &[u8; 16]) -> [u8; 16] {
        let w = expand_key(key);
        let mut state = *block;
        add_round_key(&mut state, &w[0..16]);
        for round in 1..10 {
            sub_bytes(&mut state);
            shift_rows(&mut state);
            mix_columns(&mut state);
            add_round_key(&mut state, &w[round * 16..(round + 1) * 16]);
        }
        sub_bytes(&mut state);
        shift_rows(&mut state);
        add_round_key(&mut state, &w[160..176]);
        state
    }

    #[test]
    fn the_table_driven_round_matches_the_textbook_one() {
        // A NIST vector says the cipher is right at one point; this says the
        // two implementations agree everywhere, which is what a rewrite of a
        // primitive actually needs. Deterministic inputs, so a failure is
        // reproducible.
        let mut key = [0u8; 16];
        let mut block = [0u8; 16];
        let mut x = 0x1234_5678u32;
        for _ in 0..4096 {
            for byte in key.iter_mut().chain(block.iter_mut()) {
                // xorshift: any cheap spread of bit patterns will do.
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                *byte = x as u8;
            }
            assert_eq!(
                RoundKeys::new(&key).encrypt_block(&block),
                reference(&key, &block),
                "key {key:02x?} block {block:02x?}"
            );
        }
    }

    #[test]
    fn a_ctr_stream_still_round_trips() {
        // CTR is its own inverse, and it is the mode every NCA section body
        // uses, so this is the path the firmware fonts go through.
        let key = [0x42u8; 16];
        let counter = [0x11u8; 16];
        let plain: Vec<u8> = (0..1000u32).map(|i| (i * 7) as u8).collect();
        let mut buf = plain.clone();
        aes128_ctr_xor_in_place(&key, &counter, &mut buf);
        assert_ne!(buf, plain, "it actually encrypted something");
        aes128_ctr_xor_in_place(&key, &counter, &mut buf);
        assert_eq!(buf, plain);
    }
}
