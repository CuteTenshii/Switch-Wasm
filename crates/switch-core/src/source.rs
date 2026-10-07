//! Random-access byte sources: how a container larger than memory is read.
//! Everything that reads container bytes goes through [`ByteSource`], which
//! yields `u64`-addressed ranges on demand; [`Window`] stacks sub-ranges without copying.

use crate::Error;

/// The largest single allocation (`isize::MAX`, 2 GiB on wasm32).
pub const MAX_ALLOC: u64 = isize::MAX as u64;

/// Check that a `u64` length can be allocated, so oversized input errors instead of trapping.
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
pub trait ByteSource: std::fmt::Debug {
    fn len(&self) -> u64;

    /// Read into `out`, returning how many bytes were filled.
    /// A short fill means end-of-source; read failures are errors, not short reads.
    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<usize, Error>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fill `out` completely, or fail.
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

    /// Copy `len` bytes into a fresh buffer; an unallocatable length is an `Err`, not a trap.
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

/// A source over bytes already in memory (native examples and tests only).
#[derive(Debug, Clone, Copy)]
pub struct SliceSource<'a>(pub &'a [u8]);

impl ByteSource for SliceSource<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<usize, Error> {
        // Compare before narrowing, or an offset above 4 GiB wraps on wasm32.
        if offset >= self.len() {
            return Ok(0);
        }
        let start = offset as usize;
        let n = out.len().min(self.0.len() - start);
        out[..n].copy_from_slice(&self.0[start..start + n]);
        Ok(n)
    }
}

/// An owned in-memory source.
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

/// A source over a file on disk, reading only the requested ranges.
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
        crate::trace!(
            crate::trace::Trace::Io,
            "[io] file read {want:#x} bytes at {offset:#x}"
        );
        let mut file = self.file.borrow_mut();
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| Error::Io(format!("seek to {offset:#x}: {e}")))?;
        file.read_exact(&mut out[..want])
            .map_err(|e| Error::Io(format!("read {want} bytes at {offset:#x}: {e}")))?;
        Ok(want)
    }
}

/// A sub-range of another source, addressed from 0.
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

    pub fn base(&self) -> u64 {
        self.base
    }

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
        assert!(matches!(
            src.read_vec(0, MAX_ALLOC + 1),
            Err(Error::TooLarge { .. })
        ));
    }
}
