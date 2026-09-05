//! Random-access byte sources: how a container larger than memory is read.
//!
//! A retail `.nsp` does not fit in memory on the target this emulator was
//! built for. wasm32 linear memory tops out at 4 GiB, Rust's allocator on
//! that target rejects any single request above `isize::MAX` (2 GiB), and a
//! modern title's container is bigger than both, so nothing that reads a
//! container may assume it can hold one.
//!
//! Everything that reads container bytes therefore goes through
//! [`ByteSource`]: a `u64`-addressed window that yields ranges on demand. The
//! browser leaves the file on disk and serves ranges out of it; the host
//! test suite and the native examples wrap a slice they already have. The
//! composable pieces below ([`Window`], plus [`crate::nca::SectionSource`])
//! stack into "the RomFS inside the section inside the NCA inside the NSP"
//! without a single copy of anything but the bytes actually asked for.

use crate::Error;

/// The largest buffer this target can allocate at once, `isize::MAX`, which
/// is what `Layout` (and therefore every allocation) is limited to. On wasm32
/// that is 2 GiB, so it is a real ceiling and not a theoretical one: a
/// request past it used to reach `Layout::from_size_align(..).unwrap()` and
/// trap the whole module with `unreachable`.
pub const MAX_ALLOC: u64 = isize::MAX as u64;

/// Turn a `u64` length into one this target can actually allocate, or say why
/// it can't. Callers that are about to build a `Vec` of guest-controlled size
/// go through here so an oversized container is an error message rather than
/// a trap.
pub fn alloc_len(len: u64, what: &str) -> Result<usize, Error> {
    if len > MAX_ALLOC {
        return Err(Error::TooLarge {
            what: what.to_string(),
            len,
            max: MAX_ALLOC,
        });
    }
    Ok(len as usize)
}

/// A `u64`-addressed, read-only, random-access byte range.
///
/// `Debug` is a supertrait because [`crate::cpu::Cpu`] stores one and derives
/// `Debug` itself.
pub trait ByteSource: std::fmt::Debug {
    /// Total number of readable bytes.
    fn len(&self) -> u64;

    /// Read into `out`, returning how many bytes were filled.
    ///
    /// A short fill means end-of-source and nothing else: implementations
    /// that can fail (a host file that moved out from under us) report that
    /// as an error rather than as a short read, so a caller can tell "there
    /// is no more data" from "the data could not be read".
    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<usize, Error>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fill `out` completely, or fail. Used for headers and structures whose
    /// declared size is the whole point.
    fn read_exact_at(&self, offset: u64, out: &mut [u8]) -> Result<(), Error> {
        let want = out.len();
        let got = self.read_at(offset, out)?;
        if got != want {
            return Err(Error::Truncated {
                what: format!("read at {:#x}", offset),
                expected: want,
                got,
            });
        }
        Ok(())
    }

    /// Copy `len` bytes out into a fresh buffer.
    ///
    /// Reserves before filling so a length this target cannot allocate is an
    /// `Err`, not an abort: the global allocator's out-of-memory path is
    /// `handle_alloc_error`, which on wasm is an `unreachable` trap that
    /// takes the module down with no message.
    fn read_vec(&self, offset: u64, len: u64) -> Result<Vec<u8>, Error> {
        let n = alloc_len(len, "buffer")?;
        let mut buf = Vec::new();
        buf.try_reserve_exact(n).map_err(|_| Error::TooLarge {
            what: "buffer".into(),
            len,
            max: MAX_ALLOC,
        })?;
        buf.resize(n, 0);
        self.read_exact_at(offset, &mut buf)?;
        Ok(buf)
    }
}

impl<T: ByteSource + ?Sized> ByteSource for &T {
    fn len(&self) -> u64 {
        (**self).len()
    }
    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<usize, Error> {
        (**self).read_at(offset, out)
    }
}

impl<T: ByteSource + ?Sized> ByteSource for Box<T> {
    fn len(&self) -> u64 {
        (**self).len()
    }
    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<usize, Error> {
        (**self).read_at(offset, out)
    }
}

/// A source over bytes already in memory. The native examples and the test
/// suite read whole files this way; the browser never does.
#[derive(Debug, Clone, Copy)]
pub struct SliceSource<'a>(pub &'a [u8]);

impl ByteSource for SliceSource<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<usize, Error> {
        // `offset` is u64 and this target's usize may be 32-bit: compare
        // before narrowing, or a 4 GiB offset wraps to a valid index.
        if offset >= self.len() {
            return Ok(0);
        }
        let start = offset as usize;
        let n = out.len().min(self.0.len() - start);
        out[..n].copy_from_slice(&self.0[start..start + n]);
        Ok(n)
    }
}

/// An owned in-memory source, for the same cases as [`SliceSource`] where the
/// bytes have to outlive the caller's frame (the RomFS a native loader
/// decrypted up front, say).
#[derive(Debug, Clone)]
pub struct MemSource(pub Vec<u8>);

impl ByteSource for MemSource {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<usize, Error> {
        SliceSource(&self.0).read_at(offset, out)
    }
}

/// A source over a file on disk, for hosts that have one, the native
/// counterpart of the browser's `host_read`.
///
/// Reads the range asked for and nothing else, so a multi-gigabyte container
/// costs a few seeks rather than its own size in memory.
#[derive(Debug)]
pub struct FileSource {
    file: std::cell::RefCell<std::fs::File>,
    len: u64,
}

impl FileSource {
    pub fn open(path: impl AsRef<std::path::Path>) -> std::io::Result<FileSource> {
        let file = std::fs::File::open(path)?;
        let len = file.metadata()?.len();
        Ok(FileSource {
            file: std::cell::RefCell::new(file),
            len,
        })
    }
}

impl ByteSource for FileSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<usize, Error> {
        use std::io::{Read, Seek, SeekFrom};
        if offset >= self.len {
            return Ok(0);
        }
        let want = ((out.len() as u64).min(self.len - offset)) as usize;
        let mut file = self.file.borrow_mut();
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| Error::Io(format!("seek to {offset:#x}: {e}")))?;
        file.read_exact(&mut out[..want])
            .map_err(|e| Error::Io(format!("read {want} bytes at {offset:#x}: {e}")))?;
        Ok(want)
    }
}

/// A sub-range of another source, addressed from 0.
///
/// This is what turns "the whole NSP" into "the NCA at entry 3" and "the
/// decrypted section" into "the RomFS image inside it", without either step
/// copying or bounding-checking the layer below it again.
#[derive(Debug, Clone)]
pub struct Window<S> {
    inner: S,
    base: u64,
    len: u64,
}

impl<S: ByteSource> Window<S> {
    /// A window over `base..base + len` of `inner`, which must lie inside it.
    pub fn new(inner: S, base: u64, len: u64, what: &str) -> Result<Window<S>, Error> {
        let end = base.checked_add(len).ok_or(Error::Overflow)?;
        if end > inner.len() {
            return Err(Error::OutOfRange {
                what: what.to_string(),
                start: base,
                end,
                available: inner.len(),
            });
        }
        Ok(Window { inner, base, len })
    }

    /// The window's start within the source it was cut from.
    pub fn base(&self) -> u64 {
        self.base
    }

    /// Give the wrapped source back.
    pub fn into_inner(self) -> S {
        self.inner
    }
}

impl<S: ByteSource> ByteSource for Window<S> {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<usize, Error> {
        if offset >= self.len {
            return Ok(0);
        }
        let avail = self.len - offset;
        let want = (out.len() as u64).min(avail) as usize;
        self.inner.read_at(self.base + offset, &mut out[..want])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_slice_source_reads_and_stops_at_the_end() {
        let data: Vec<u8> = (0..64u8).collect();
        let src = SliceSource(&data);
        let mut out = [0u8; 16];
        assert_eq!(src.read_at(8, &mut out).unwrap(), 16);
        assert_eq!(out[0], 8);
        // A read straddling the end fills what exists and says how much.
        assert_eq!(src.read_at(56, &mut out).unwrap(), 8);
        assert_eq!(src.read_at(64, &mut out).unwrap(), 0);
        assert_eq!(src.read_at(1 << 40, &mut out).unwrap(), 0);
    }

    #[test]
    fn a_window_is_addressed_from_zero_and_cannot_escape() {
        let data: Vec<u8> = (0..64u8).collect();
        let w = Window::new(SliceSource(&data), 32, 16, "test").unwrap();
        assert_eq!(w.len(), 16);
        let mut out = [0u8; 32];
        // Asking for more than the window holds stops at its end, even though
        // the source below it has more.
        assert_eq!(w.read_at(0, &mut out).unwrap(), 16);
        assert_eq!(out[0], 32);
        assert_eq!(out[15], 47);
        assert_eq!(out[16], 0);
        assert!(matches!(
            Window::new(SliceSource(&data), 60, 16, "test"),
            Err(Error::OutOfRange { .. })
        ));
    }

    #[test]
    fn an_unallocatable_length_is_an_error_not_a_trap() {
        let data = vec![0u8; 16];
        let src = SliceSource(&data);
        // The failure this whole module exists to prevent: a container-sized
        // length reaching the allocator.
        assert!(matches!(
            src.read_vec(0, MAX_ALLOC + 1),
            Err(Error::TooLarge { .. })
        ));
    }
}
