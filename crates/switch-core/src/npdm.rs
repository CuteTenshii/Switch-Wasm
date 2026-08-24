//! NPDM (`main.npdm`) — the process manifest an ExeFS carries beside its
//! executables.
//!
//! Horizon reads this before it creates the process, and one field in it
//! decides how the address space is laid out: `system_resource_size`, the
//! slice of the application's memory pool the kernel keeps for its own
//! per-process bookkeeping. A title that declares one gets virtual address
//! memory and runs its heap through `nn::os::detail::VammManager`; a title
//! that declares zero gets the plain heap and never touches the manager. The
//! two want quite different things from the address space, which is why
//! [`crate::cpu::MemoryLayout`] is chosen from this rather than fixed.
//!
//! META header (offsets from the start of the file):
//!
//! ```text
//! 0x00  magic "META" (u32)
//! 0x04  signature key generation (u32)
//! 0x08  reserved
//! 0x0C  flags (u8)
//! 0x0E  main thread priority (u8)
//! 0x0F  main thread core number (u8)
//! 0x14  system resource size (u32) — [7.0.0+], 0 on older titles
//! 0x18  version (u32)
//! 0x1C  main thread stack size (u32)
//! 0x20  name (0x10 bytes, NUL-padded)
//! ```

use crate::Error;

pub const NPDM_MAGIC: u32 = 0x4154_454d; // "META", little-endian
/// Bytes needed to read every field parsed here.
pub const NPDM_HEADER_SIZE: usize = 0x30;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Npdm {
    /// The kernel's per-process bookkeeping reservation, carved out of the
    /// application pool. Non-zero means this title expects virtual address
    /// memory; `nnSdk` decides that by asking `svcGetInfo` for the same
    /// figure, so reporting anything else is telling the title something its
    /// own manifest contradicts.
    pub system_resource_size: u32,
    /// The stack the main thread is created with.
    pub main_thread_stack_size: u32,
    /// The manifest's name field, for diagnostics — "Application" on a retail
    /// game.
    pub name: String,
}

impl Npdm {
    /// Parse a `main.npdm`.
    pub fn parse(data: &[u8]) -> Result<Npdm, Error> {
        if data.len() < NPDM_HEADER_SIZE {
            return Err(Error::Truncated {
                what: "NPDM header".into(),
                expected: NPDM_HEADER_SIZE,
                got: data.len(),
            });
        }
        let magic = crate::nsp::read_u32(data, 0);
        if magic != NPDM_MAGIC {
            return Err(Error::BadMagic { what: "NPDM".into(), found: magic });
        }
        let name_bytes = &data[0x20..0x30];
        let end = name_bytes.iter().position(|&b| b == 0).unwrap_or(name_bytes.len());
        Ok(Npdm {
            system_resource_size: crate::nsp::read_u32(data, 0x14),
            main_thread_stack_size: crate::nsp::read_u32(data, 0x1C),
            name: String::from_utf8_lossy(&name_bytes[..end]).into_owned(),
        })
    }

    /// The `system_resource_size` of an ExeFS's `main.npdm`, or 0 when the
    /// container has no manifest or one that cannot be read.
    ///
    /// Zero is the right answer for both of those: it is what a title without
    /// a manifest gets on hardware, and it selects the plain heap, which is
    /// the layout that works without knowing anything about the title.
    pub fn system_resource_size_of(exefs: &crate::nsp::Pfs0, data: &[u8]) -> u32 {
        let Some(file) = exefs.find("main.npdm") else {
            return 0;
        };
        let start = file.offset as usize;
        let Some(end) = start.checked_add(file.size as usize) else {
            return 0;
        };
        if end > data.len() {
            return 0;
        }
        Npdm::parse(&data[start..end])
            .map(|n| n.system_resource_size)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npdm(system_resource_size: u32) -> Vec<u8> {
        let mut data = vec![0u8; NPDM_HEADER_SIZE];
        data[0..4].copy_from_slice(&NPDM_MAGIC.to_le_bytes());
        data[0x0C] = 0x37;
        data[0x14..0x18].copy_from_slice(&system_resource_size.to_le_bytes());
        data[0x1C..0x20].copy_from_slice(&0x0010_0000u32.to_le_bytes());
        data[0x20..0x2B].copy_from_slice(b"Application");
        data
    }

    #[test]
    fn parses_a_manifest() {
        let parsed = Npdm::parse(&npdm(0x0100_0000)).unwrap();
        assert_eq!(parsed.system_resource_size, 0x0100_0000);
        assert_eq!(parsed.main_thread_stack_size, 0x0010_0000);
        assert_eq!(parsed.name, "Application");
    }

    /// Just Dance 2019 declares zero here and Just Dance 2023 declares 16 MiB,
    /// and that difference is the whole of what decides which address space
    /// each one gets — so a zero has to survive parsing as a real answer
    /// rather than being confused with a missing one.
    #[test]
    fn zero_is_a_real_answer() {
        assert_eq!(Npdm::parse(&npdm(0)).unwrap().system_resource_size, 0);
    }

    #[test]
    fn rejects_a_file_that_is_not_a_manifest() {
        let mut data = npdm(0);
        data[0] = b'X';
        assert!(matches!(Npdm::parse(&data), Err(Error::BadMagic { .. })));
        assert!(matches!(Npdm::parse(&data[..8]), Err(Error::Truncated { .. })));
    }
}
