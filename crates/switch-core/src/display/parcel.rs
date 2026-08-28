//! Android `Parcel` serialization, as the Switch's binder uses it.
//!
//! A parcel is a 16-byte header followed by a payload of 4-byte-aligned
//! fields. Every `IGraphicBufferProducer` request begins with an "interface
//! token" (a magic word plus the UTF-16 interface name), and structs travel as
//! "flattened objects" (`{ i32 length, i32 fd_count, bytes }`).

/// `ParcelHeader` is four words: payload size/offset then objects size/offset.
pub const HEADER_SIZE: u32 = 16;

/// Reads fields out of a parcel payload, tracking the cursor the way
/// `parcelReadData` does.
#[derive(Debug)]
pub struct ParcelReader<'a> {
    payload: &'a [u8],
    pos: usize,
}

impl<'a> ParcelReader<'a> {
    /// Parse the header of a raw parcel and expose its payload.
    pub fn new(raw: &'a [u8]) -> ParcelReader<'a> {
        let payload_size = read_u32(raw, 0) as usize;
        let payload_off = read_u32(raw, 4) as usize;
        let end = payload_off.saturating_add(payload_size).min(raw.len());
        let payload = raw.get(payload_off.min(raw.len())..end).unwrap_or(&[]);
        ParcelReader { payload, pos: 0 }
    }

    pub fn remaining(&self) -> usize {
        self.payload.len().saturating_sub(self.pos)
    }

    pub fn read_u32(&mut self) -> u32 {
        let v = read_u32(self.payload, self.pos);
        self.pos += 4;
        v
    }

    pub fn read_i32(&mut self) -> i32 {
        self.read_u32() as i32
    }

    /// Read `len` bytes, advancing by the 4-byte-aligned length.
    pub fn read_bytes(&mut self, len: usize) -> &'a [u8] {
        let start = self.pos.min(self.payload.len());
        let end = (start + len).min(self.payload.len());
        self.pos += (len + 3) & !3;
        &self.payload[start..end]
    }

    /// Skip the `{ magic, String16(interface) }` token every request carries.
    pub fn skip_interface_token(&mut self) {
        let _magic = self.read_u32();
        let len = self.read_u32() as usize;
        // String16 stores `len + 1` UTF-16 code units (the NUL included).
        self.pos += ((len + 1) * 2 + 3) & !3;
    }

    /// Read a flattened object (`{ i32 length, i32 fd_count, bytes }`).
    /// Returns `None` when it carries file descriptors, which the Switch's
    /// binder never does.
    pub fn read_flattened(&mut self) -> Option<&'a [u8]> {
        let len = self.read_i32();
        let fd_count = self.read_i32();
        if fd_count != 0 || len < 0 {
            return None;
        }
        Some(self.read_bytes(len as usize))
    }
}

/// Builds a parcel payload and emits the complete parcel with its header.
#[derive(Debug, Default)]
pub struct ParcelWriter {
    payload: Vec<u8>,
}

impl ParcelWriter {
    pub fn new() -> ParcelWriter {
        ParcelWriter {
            payload: Vec::new(),
        }
    }

    pub fn write_u32(&mut self, value: u32) {
        self.payload.extend_from_slice(&value.to_le_bytes());
    }

    pub fn write_i32(&mut self, value: i32) {
        self.write_u32(value as u32);
    }

    /// Append raw bytes, padded to a 4-byte boundary.
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        self.payload.extend_from_slice(bytes);
        while self.payload.len() % 4 != 0 {
            self.payload.push(0);
        }
    }

    /// Append a flattened object (`{ i32 length, i32 fd_count, bytes }`).
    pub fn write_flattened(&mut self, bytes: &[u8]) {
        self.write_i32(bytes.len() as i32);
        self.write_i32(0);
        self.write_bytes(bytes);
    }

    /// Serialize the parcel: header followed by the payload.
    pub fn finish(self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_SIZE as usize + self.payload.len());
        out.extend_from_slice(&(self.payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&HEADER_SIZE.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // objects_size
        out.extend_from_slice(&(HEADER_SIZE + self.payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.payload);
        out
    }
}

fn read_u32(data: &[u8], at: usize) -> u32 {
    let mut v = 0u32;
    for i in 0..4 {
        v |= (data.get(at + i).copied().unwrap_or(0) as u32) << (8 * i);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a request parcel the way libnx does, for the reader tests.
    fn request(payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&HEADER_SIZE.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&(HEADER_SIZE + payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    fn interface_token(name: &str) -> Vec<u8> {
        let mut w = ParcelWriter::new();
        w.write_u32(0x100);
        w.write_i32(name.len() as i32);
        let mut utf16 = Vec::new();
        for c in name.chars().chain(std::iter::once('\0')) {
            utf16.extend_from_slice(&(c as u16).to_le_bytes());
        }
        w.write_bytes(&utf16);
        w.payload
    }

    #[test]
    fn reader_skips_the_interface_token() {
        let mut payload = interface_token("android.gui.IGraphicBufferProducer");
        payload.extend_from_slice(&7u32.to_le_bytes());
        payload.extend_from_slice(&9u32.to_le_bytes());
        let raw = request(&payload);

        let mut r = ParcelReader::new(&raw);
        r.skip_interface_token();
        assert_eq!(r.read_i32(), 7);
        assert_eq!(r.read_i32(), 9);
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn flattened_object_roundtrip() {
        let mut w = ParcelWriter::new();
        w.write_i32(3);
        w.write_flattened(&[1, 2, 3, 4, 5]);
        let raw = w.finish();

        let mut r = ParcelReader::new(&raw);
        assert_eq!(r.read_i32(), 3);
        assert_eq!(r.read_flattened(), Some(&[1u8, 2, 3, 4, 5][..]));
    }

    #[test]
    fn writer_header_describes_the_payload() {
        let mut w = ParcelWriter::new();
        w.write_i32(-1);
        let raw = w.finish();
        assert_eq!(read_u32(&raw, 0), 4); // payload_size
        assert_eq!(read_u32(&raw, 4), HEADER_SIZE); // payload_off
        assert_eq!(read_u32(&raw, 8), 0); // objects_size
        assert_eq!(read_u32(&raw, 12), HEADER_SIZE + 4); // objects_off
        assert_eq!(read_u32(&raw, 16), 0xFFFF_FFFF);
    }

    #[test]
    fn a_flattened_object_with_fds_is_rejected() {
        let mut w = ParcelWriter::new();
        w.write_i32(4);
        w.write_i32(1); // fd_count
        w.write_bytes(&[0; 4]);
        let raw = w.finish();
        let mut r = ParcelReader::new(&raw);
        assert_eq!(r.read_flattened(), None);
    }

    #[test]
    fn reading_past_the_payload_yields_zeroes() {
        let raw = request(&[]);
        let mut r = ParcelReader::new(&raw);
        assert_eq!(r.read_i32(), 0);
        assert!(r.read_bytes(8).is_empty());
    }
}
