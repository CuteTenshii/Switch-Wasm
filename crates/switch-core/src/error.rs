//! Error type shared by all parsers.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A buffer ended before a declared structure was complete.
    Truncated {
        what: String,
        expected: usize,
        got: usize,
    },
    /// A magic value did not match the expected format magic.
    BadMagic { what: String, found: u32 },
    /// A PFS0 string-table entry pointed outside the table.
    BadStringTable { index: usize, offset: usize },
    /// A PFS0 file entry claimed an extent beyond the image.
    FileOutOfBounds {
        index: usize,
        name: String,
        offset: u64,
        size: u64,
        image_size: usize,
    },
    /// Arithmetic overflow while computing an address or extent.
    Overflow,
    /// The file is not an ELF we can load (bad class, machine, etc).
    Elf(String),
    /// The file is not an NRO we can load.
    Nro(String),
    /// A CPU fault (bad memory access, invalid state, unreachable).
    Cpu(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Truncated { what, expected, got } => write!(
                f,
                "{}: expected at least {} bytes, got {}",
                what, expected, got
            ),
            Error::BadMagic { what, found } => {
                write!(f, "{}: bad magic 0x{:08x}", what, found)
            }
            Error::BadStringTable { index, offset } => write!(
                f,
                "PFS0 string table: entry {} name offset {} out of range",
                index, offset
            ),
            Error::FileOutOfBounds { index, name, offset, size, image_size } => write!(
                f,
                "PFS0 file {} ('{}'): range [{:#x}, {:#x}) exceeds image size {}",
                index, name, offset, offset + size, image_size
            ),
            Error::Overflow => write!(f, "arithmetic overflow"),
            Error::Elf(msg) => write!(f, "ELF: {}", msg),
            Error::Nro(msg) => write!(f, "NRO: {}", msg),
            Error::Cpu(msg) => write!(f, "CPU: {}", msg),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
