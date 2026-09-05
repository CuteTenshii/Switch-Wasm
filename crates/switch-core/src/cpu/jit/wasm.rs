//! Writing a wasm module, byte by byte.
//!
//! [`super::emit`] turns a translated block into wasm; this is what it writes
//! that wasm *with*. Nothing here knows anything about AArch64 or about this
//! emulator: it is the binary format and nothing else, so it can be read
//! against the spec on its own.
//!
//! The core has no dependencies (see AGENTS.md), so there is no `wasm-encoder`
//! to reach for, and there could not be: this code has to compile *to* wasm as
//! well as run on the host, and it is on the path of every first visit to a
//! block, so it has to be small.
//!
//! # What is deliberately missing
//!
//! Only the instructions and sections an emitted block needs. There are no
//! floats, no SIMD, no globals and no multi-memory: a block that needs
//! something not here does not get emitted at all and stays with the
//! interpreter, which is the same "slower, never wrong" rule the translator
//! already works to. Adding an opcode is a line; guessing at one that is never
//! emitted is dead code that no test can reach.

/// Value types, as they appear in a signature or a local declaration.
pub(super) const I32: u8 = 0x7F;
pub(super) const I64: u8 = 0x7E;

/// Append `v` as an unsigned LEB128, the encoding every length, index and
/// memory offset in the format uses.
pub(super) fn uleb(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let mut byte = (v & 0x7F) as u8;
        v >>= 7;
        if v != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if v == 0 {
            return;
        }
    }
}

/// Append `v` as a signed LEB128, which is what `i32.const` and `i64.const`
/// take.
///
/// Not interchangeable with [`uleb`]: the two agree only while the value fits
/// in six bits, so a constant of 64 written the unsigned way decodes as -64.
/// That is a silent wrong answer rather than a validation error, which is why
/// the two have separate names here rather than one function with a flag.
pub(super) fn sleb(out: &mut Vec<u8>, mut v: i64) {
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        let sign_bit = byte & 0x40 != 0;
        if (v == 0 && !sign_bit) || (v == -1 && sign_bit) {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// A section, length-prefixed as the format requires.
fn section(out: &mut Vec<u8>, id: u8, body: &[u8]) {
    if body.is_empty() {
        return;
    }
    out.push(id);
    uleb(out, body.len() as u64);
    out.extend_from_slice(body);
}

/// A vector: a count followed by the elements, which is how every list in the
/// format is written.
fn vec_header(out: &mut Vec<u8>, count: usize) {
    uleb(out, count as u64);
}

/// One function's locals and body.
///
/// The body is built by calling the instruction methods in order; each appends
/// its own encoding, so the `Vec` *is* the instruction stream and there is no
/// separate list of instructions to lower.
#[derive(Default)]
pub(super) struct Func {
    /// Run-length encoded local declarations, `(count, type)`, as the format
    /// stores them.
    locals: Vec<(u32, u8)>,
    code: Vec<u8>,
}

impl Func {
    pub(super) fn new() -> Func {
        Func::default()
    }

    /// Declare `count` locals of type `ty`, returning the index of the first.
    ///
    /// Indices continue from the parameters, so a function's first local is
    /// numbered after its last parameter; the caller passes `params` because
    /// this type never sees the signature.
    pub(super) fn locals(&mut self, params: u32, count: u32, ty: u8) -> u32 {
        let first = params + self.locals.iter().map(|&(n, _)| n).sum::<u32>();
        self.locals.push((count, ty));
        first
    }

    fn op(&mut self, opcode: u8) {
        self.code.push(opcode);
    }

    fn op_idx(&mut self, opcode: u8, idx: u32) {
        self.code.push(opcode);
        uleb(&mut self.code, u64::from(idx));
    }

    pub(super) fn i32_const(&mut self, v: i32) {
        self.code.push(0x41);
        sleb(&mut self.code, i64::from(v));
    }

    pub(super) fn i64_const(&mut self, v: i64) {
        self.code.push(0x42);
        sleb(&mut self.code, v);
    }

    pub(super) fn local_get(&mut self, i: u32) {
        self.op_idx(0x20, i);
    }

    pub(super) fn local_set(&mut self, i: u32) {
        self.op_idx(0x21, i);
    }

    /// A load or store's immediates are an alignment *hint* (log2 of the
    /// assumed alignment) and a static byte offset. The offset is what makes
    /// guest state cheap to reach: the address operand stays dynamic and the
    /// base of the register file or the guest arena is folded into the
    /// instruction.
    fn mem(&mut self, opcode: u8, align: u8, offset: u32) {
        self.code.push(opcode);
        uleb(&mut self.code, u64::from(align));
        uleb(&mut self.code, u64::from(offset));
    }

    pub(super) fn i32_load(&mut self, offset: u32) {
        self.mem(0x28, 2, offset);
    }

    pub(super) fn i64_load(&mut self, offset: u32) {
        self.mem(0x29, 3, offset);
    }

    pub(super) fn i32_store(&mut self, offset: u32) {
        self.mem(0x36, 2, offset);
    }

    pub(super) fn i64_store(&mut self, offset: u32) {
        self.mem(0x37, 3, offset);
    }

    pub(super) fn i32_and(&mut self) {
        self.op(0x71);
    }

    pub(super) fn i32_or(&mut self) {
        self.op(0x72);
    }

    pub(super) fn i32_shl(&mut self) {
        self.op(0x74);
    }

    pub(super) fn i64_add(&mut self) {
        self.op(0x7C);
    }

    pub(super) fn i64_and(&mut self) {
        self.op(0x83);
    }

    pub(super) fn i64_or(&mut self) {
        self.op(0x84);
    }

    pub(super) fn i64_xor(&mut self) {
        self.op(0x85);
    }

    pub(super) fn i64_shr_u(&mut self) {
        self.op(0x88);
    }

    pub(super) fn i64_eqz(&mut self) {
        self.op(0x50);
    }

    pub(super) fn i64_ne(&mut self) {
        self.op(0x52);
    }

    pub(super) fn i64_lt_u(&mut self) {
        self.op(0x54);
    }

    /// Narrow an i64 to i32. Every guest address is computed in 64 bits and
    /// then used as a wasm address, which is 32-bit, so this is on the path of
    /// every guest load and store.
    pub(super) fn i32_wrap_i64(&mut self) {
        self.op(0xA7);
    }

    pub(super) fn end(&mut self) {
        self.op(0x0B);
    }

    /// How many bytes the body has reached. The emitter uses this to decide a
    /// block has grown past what is worth compiling.
    pub(super) fn len(&self) -> usize {
        self.code.len()
    }

    /// The function as the code section holds it: size, locals, body.
    fn encode(&self) -> Vec<u8> {
        let mut inner = Vec::with_capacity(self.code.len() + 8);
        vec_header(&mut inner, self.locals.len());
        for &(count, ty) in &self.locals {
            uleb(&mut inner, u64::from(count));
            inner.push(ty);
        }
        inner.extend_from_slice(&self.code);
        let mut out = Vec::with_capacity(inner.len() + 5);
        uleb(&mut out, inner.len() as u64);
        out.extend_from_slice(&inner);
        out
    }
}

/// A function signature.
pub(super) struct Type {
    pub(super) params: Vec<u8>,
    pub(super) results: Vec<u8>,
}

/// A module under construction.
///
/// The emitted module imports its memory rather than defining one, because the
/// memory it has to address is the emulator's own linear memory: the guest
/// register file and the guest arena both live there, so an emitted block
/// reaches guest state with a plain `i64.load` at a static offset instead of
/// calling back into the host.
#[derive(Default)]
pub(super) struct Module {
    types: Vec<Type>,
    /// Type index per defined function.
    funcs: Vec<u32>,
    bodies: Vec<Func>,
    /// `(name, function index)`.
    exports: Vec<(String, u32)>,
}

impl Module {
    pub(super) fn new() -> Module {
        Module::default()
    }

    pub(super) fn add_type(&mut self, params: Vec<u8>, results: Vec<u8>) -> u32 {
        self.types.push(Type { params, results });
        self.types.len() as u32 - 1
    }

    pub(super) fn add_func(&mut self, ty: u32, body: Func) -> u32 {
        self.funcs.push(ty);
        self.bodies.push(body);
        self.funcs.len() as u32 - 1
    }

    pub(super) fn export(&mut self, name: &str, func: u32) {
        self.exports.push((name.to_string(), func));
    }

    pub(super) fn finish(&self) -> Vec<u8> {
        let mut out = vec![0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00];

        let mut body = Vec::new();
        vec_header(&mut body, self.types.len());
        for ty in &self.types {
            body.push(0x60);
            vec_header(&mut body, ty.params.len());
            body.extend_from_slice(&ty.params);
            vec_header(&mut body, ty.results.len());
            body.extend_from_slice(&ty.results);
        }
        section(&mut out, 1, &body);

        // The memory import is written here rather than being a field, because
        // there is exactly one and it is not optional: an emitted module with
        // its own memory could not see guest state at all.
        body.clear();
        vec_header(&mut body, 1);
        name_bytes(&mut body, "e");
        name_bytes(&mut body, "m");
        body.push(0x02);
        // A minimum of zero pages: the host's memory is already as large as it
        // is, and asking for more here would only fail an instantiation the
        // host has already sized correctly.
        body.push(0x00);
        uleb(&mut body, 0);
        section(&mut out, 2, &body);

        body.clear();
        vec_header(&mut body, self.funcs.len());
        for &ty in &self.funcs {
            uleb(&mut body, u64::from(ty));
        }
        section(&mut out, 3, &body);

        body.clear();
        vec_header(&mut body, self.exports.len());
        for (name, func) in &self.exports {
            name_bytes(&mut body, name);
            body.push(0x00);
            uleb(&mut body, u64::from(*func));
        }
        section(&mut out, 7, &body);

        body.clear();
        vec_header(&mut body, self.bodies.len());
        for func in &self.bodies {
            body.extend_from_slice(&func.encode());
        }
        section(&mut out, 10, &body);

        out
    }
}

/// A name, as the format writes one: a length and its UTF-8 bytes.
fn name_bytes(out: &mut Vec<u8>, name: &str) {
    uleb(out, name.len() as u64);
    out.extend_from_slice(name.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(v: u64) -> Vec<u8> {
        let mut out = Vec::new();
        uleb(&mut out, v);
        out
    }

    fn s(v: i64) -> Vec<u8> {
        let mut out = Vec::new();
        sleb(&mut out, v);
        out
    }

    #[test]
    fn unsigned_leb128_matches_the_spec_examples() {
        assert_eq!(u(0), vec![0x00]);
        assert_eq!(u(1), vec![0x01]);
        assert_eq!(u(127), vec![0x7F]);
        assert_eq!(u(128), vec![0x80, 0x01]);
        assert_eq!(u(624485), vec![0xE5, 0x8E, 0x26]);
        assert_eq!(u(u32::MAX as u64), vec![0xFF, 0xFF, 0xFF, 0xFF, 0x0F]);
    }

    /// The encoding a constant needs is the *signed* one, and the two agree
    /// only up to 63. A guest immediate of 64 written unsigned decodes as -64,
    /// which validates and computes the wrong answer, so this is the check
    /// that the two never got swapped.
    #[test]
    fn signed_leb128_is_not_the_unsigned_one() {
        assert_eq!(s(0), vec![0x00]);
        assert_eq!(s(63), vec![0x3F]);
        assert_eq!(s(-1), vec![0x7F]);
        assert_eq!(s(-64), vec![0x40]);
        assert_eq!(s(64), vec![0xC0, 0x00]);
        assert_eq!(s(-123456), vec![0xC0, 0xBB, 0x78]);
        assert_eq!(s(i64::from(i32::MIN)), vec![0x80, 0x80, 0x80, 0x80, 0x78]);

        // Where they differ is the whole point: 64 and every value with bit 6
        // set in its last byte.
        assert_ne!(s(64), u(64));
        assert_eq!(s(63), u(63));
    }

    /// A module with one function that adds its two parameters, checked byte
    /// for byte. If the section framing is wrong this is where it shows,
    /// rather than in a browser with a `CompileError` and no offset.
    #[test]
    fn a_minimal_module_encodes_to_the_expected_bytes() {
        let mut m = Module::new();
        let ty = m.add_type(vec![I64, I64], vec![I64]);
        let mut f = Func::new();
        f.local_get(0);
        f.local_get(1);
        f.i64_add();
        f.end();
        let idx = m.add_func(ty, f);
        m.export("add", idx);
        let bytes = m.finish();

        assert_eq!(
            &bytes[..8],
            &[0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00]
        );
        #[rustfmt::skip]
        let expected: Vec<u8> = vec![
            0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00,
            0x01, 0x07, 0x01, 0x60, 0x02, 0x7E, 0x7E, 0x01, 0x7E,
            0x02, 0x08, 0x01, 0x01, b'e', 0x01, b'm', 0x02, 0x00, 0x00,
            0x03, 0x02, 0x01, 0x00,
            0x07, 0x07, 0x01, 0x03, b'a', b'd', b'd', 0x00, 0x00,
            0x0A, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x7C, 0x0B,
        ];
        assert_eq!(bytes, expected);
    }

    /// Locals are numbered after the parameters, and a second group continues
    /// from the first. Getting this wrong reads a parameter as a local, which
    /// validates whenever the types happen to line up.
    #[test]
    fn local_indices_continue_from_the_parameters() {
        let mut f = Func::new();
        assert_eq!(f.locals(2, 3, I64), 2);
        assert_eq!(f.locals(2, 1, I32), 5);
    }
}
