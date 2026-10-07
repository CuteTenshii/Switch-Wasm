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
        image_size: u64,
    },
    /// Like `Truncated`, but with `u64` offsets for containers past 4 GiB.
    OutOfRange {
        what: String,
        start: u64,
        end: u64,
        available: u64,
    },
    /// An allocation larger than this target allows (`isize::MAX` on wasm32).
    TooLarge { what: String, len: u64, max: u64 },
    /// The host's copy of a container file could not be read.
    Io(String),
    /// Arithmetic overflow while computing an address or extent.
    Overflow,
    /// The file is not an ELF we can load (bad class, machine, etc).
    Elf(String),
    /// The file is not an NRO we can load.
    Nro(String),
    /// An NCA body couldn't be decrypted or extracted.
    Nca(String),
    /// The file is not an NSO we can load.
    Nso(String),
    /// A RomFS image couldn't be walked.
    RomFs(String),
    /// An ES ticket couldn't be parsed or its title key decrypted.
    Ticket(String),
    /// A cartridge image's partitions could not be walked.
    Xci(String),
    /// A CPU fault (bad memory access, invalid state, unreachable).
    Cpu(String),
    /// A GPU fault: an unmapped address, a malformed command stream, or an
    /// unimplemented class or format.
    Gpu(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Truncated {
                what,
                expected,
                got,
            } => write!(
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
            Error::FileOutOfBounds {
                index,
                name,
                offset,
                size,
                image_size,
            } => write!(
                f,
                "PFS0 file {} ('{}'): range [{:#x}, {:#x}) exceeds image size {}",
                index,
                name,
                offset,
                offset + size,
                image_size
            ),
            Error::OutOfRange {
                what,
                start,
                end,
                available,
            } => write!(
                f,
                "{}: range [{:#x}, {:#x}) exceeds the {} bytes available",
                what, start, end, available
            ),
            Error::TooLarge { what, len, max } => write!(
                f,
                "{} is {} bytes, larger than the {} bytes this build can hold in memory at once",
                what, len, max
            ),
            Error::Io(msg) => write!(f, "read failed: {}", msg),
            Error::Overflow => write!(f, "arithmetic overflow"),
            Error::Elf(msg) => write!(f, "ELF: {}", msg),
            Error::Nro(msg) => write!(f, "NRO: {}", msg),
            Error::Nca(msg) => write!(f, "NCA: {}", msg),
            Error::Xci(msg) => write!(f, "XCI: {}", msg),
            Error::Nso(msg) => write!(f, "NSO: {}", msg),
            Error::RomFs(msg) => write!(f, "RomFS: {}", msg),
            Error::Ticket(msg) => write!(f, "ticket: {}", msg),
            Error::Cpu(msg) => write!(f, "CPU: {}", msg),
            Error::Gpu(msg) => write!(f, "GPU: {}", msg),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
