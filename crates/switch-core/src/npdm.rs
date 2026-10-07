//! NPDM (`main.npdm`): the process manifest beside an ExeFS's executables.
//!
//! `system_resource_size` selects [`crate::cpu::MemoryLayout`]: non-zero means
//! virtual address memory via `nn::os::detail::VammManager`, zero the plain heap.
//!
//! META header (offsets from the start of the file):
//!
//! ```text
//! 0x00  magic "META" (u32)
//! 0x04  signature key generation (u32)
//! 0x08  reserved
//! 0x0C  flags (u8): bit 0 is `Is64BitInstruction`, bits 1:3 the
//!       address-space type
//! 0x0E  main thread priority (u8)
//! 0x0F  main thread core number (u8)
//! 0x14  system resource size (u32), [7.0.0+], 0 on older titles
//! 0x18  version (u32)
//! 0x1C  main thread stack size (u32)
//! 0x20  name (0x10 bytes, NUL-padded)
//! 0x70  ACI0 offset (u32), 0x74 ACI0 size (u32)
//! ```
//!
//! The ACI0 kernel capabilities (offset and size at ACI0+0x30 and +0x34) hold
//! the `ThreadInfo` descriptor naming the allowed cores.

use crate::Error;

pub const NPDM_MAGIC: u32 = 0x4154_454d; // "META", little-endian
/// Bytes needed to read every field parsed here.
pub const NPDM_HEADER_SIZE: usize = 0x30;
const ACI0_MAGIC: u32 = 0x3049_4341; // "ACI0", little-endian
/// The fallback core mask: cores 0 to 2, as retail applications get. Core 3 is the system's.
pub const APPLICATION_CORE_MASK: u64 = 0b0111;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Npdm {
    /// Whether the process runs in AArch64 (Mario Kart 8 Deluxe does not).
    pub is_64_bit: bool,
    /// The kernel's per-process reservation; non-zero means virtual address
    /// memory, and `svcGetInfo` must report the same figure.
    pub system_resource_size: u32,
    /// 0 (most urgent) to 63.
    pub main_thread_priority: u8,
    /// Also where threads created with the default core (-2) go.
    pub main_thread_core: u8,
    pub main_thread_stack_size: u32,
    /// For diagnostics; "Application" on a retail game.
    pub name: String,
    /// From the `ThreadInfo` capability. Applications get 0..=2; some system
    /// applets get core 3 alone.
    pub core_mask: u64,
}

impl Npdm {
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
            return Err(Error::BadMagic {
                what: "NPDM".into(),
                found: magic,
            });
        }
        let name_bytes = &data[0x20..0x30];
        let end = name_bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(name_bytes.len());
        Ok(Npdm {
            is_64_bit: data[0x0C] & 1 != 0,
            system_resource_size: crate::nsp::read_u32(data, 0x14),
            main_thread_priority: data[0x0E],
            main_thread_core: data[0x0F],
            main_thread_stack_size: crate::nsp::read_u32(data, 0x1C),
            name: String::from_utf8_lossy(&name_bytes[..end]).into_owned(),
            core_mask: thread_info_core_mask(data).unwrap_or(APPLICATION_CORE_MASK),
        })
    }

    /// Whether an ExeFS's `main.npdm` declares AArch64; missing or unreadable means yes.
    pub fn is_64_bit_of(exefs: &crate::nsp::Pfs0, data: &[u8]) -> bool {
        Npdm::of(exefs, data).map(|n| n.is_64_bit).unwrap_or(true)
    }

    pub fn of(exefs: &crate::nsp::Pfs0, data: &[u8]) -> Option<Npdm> {
        let file = exefs.find("main.npdm")?;
        let start = file.offset as usize;
        let end = start.checked_add(file.size as usize)?;
        if end > data.len() {
            return None;
        }
        Npdm::parse(&data[start..end]).ok()
    }

    /// `None` when there is no parseable manifest.
    pub fn main_thread_priority_of(exefs: &crate::nsp::Pfs0, data: &[u8]) -> Option<u8> {
        Npdm::of(exefs, data).map(|n| n.main_thread_priority)
    }

    /// `None` when there is no parseable manifest.
    pub fn main_thread_core_of(exefs: &crate::nsp::Pfs0, data: &[u8]) -> Option<u8> {
        Npdm::of(exefs, data).map(|n| n.main_thread_core)
    }

    /// `None` when there is no parseable manifest.
    pub fn core_mask_of(exefs: &crate::nsp::Pfs0, data: &[u8]) -> Option<u64> {
        Npdm::of(exefs, data).map(|n| n.core_mask)
    }

    /// 0 (the plain heap, as on hardware) when there is no readable manifest.
    pub fn system_resource_size_of(exefs: &crate::nsp::Pfs0, data: &[u8]) -> u32 {
        Npdm::of(exefs, data)
            .map(|n| n.system_resource_size)
            .unwrap_or(0)
    }
}

/// The core mask from the ACI0 `ThreadInfo` capability (type = three trailing
/// one bits; lowest and highest core at bits 16..24 and 24..32).
fn thread_info_core_mask(data: &[u8]) -> Option<u64> {
    let read = |at: usize| -> Option<u32> {
        Some(u32::from_le_bytes(
            data.get(at..at.checked_add(4)?)?.try_into().ok()?,
        ))
    };
    let aci0 = read(0x70)? as usize;
    if read(aci0)? != ACI0_MAGIC {
        return None;
    }
    let caps = aci0.checked_add(read(aci0 + 0x30)? as usize)?;
    let count = read(aci0 + 0x34)? as usize / 4;
    (0..count).find_map(|i| {
        let cap = read(caps + 4 * i)?;
        if cap.trailing_ones() != 3 {
            return None;
        }
        let (min, max) = ((cap >> 16) & 0xFF, cap >> 24);
        (min <= max && max <= 3).then(|| (min..=max).fold(0, |mask, core| mask | 1u64 << core))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npdm(system_resource_size: u32) -> Vec<u8> {
        let mut data = vec![0u8; NPDM_HEADER_SIZE];
        data[0..4].copy_from_slice(&NPDM_MAGIC.to_le_bytes());
        data[0x0C] = 0x37;
        data[0x0E] = 0x2C;
        data[0x0F] = 1;
        data[0x14..0x18].copy_from_slice(&system_resource_size.to_le_bytes());
        data[0x1C..0x20].copy_from_slice(&0x0010_0000u32.to_le_bytes());
        data[0x20..0x2B].copy_from_slice(b"Application");
        data
    }

    /// A manifest whose ACI0 holds `caps` as its kernel capabilities.
    fn with_capabilities(caps: &[u32]) -> Vec<u8> {
        let mut data = npdm(0);
        data.resize(0x80, 0);
        data[0x70..0x74].copy_from_slice(&0x80u32.to_le_bytes());
        let mut aci0 = vec![0u8; 0x40];
        aci0[0..4].copy_from_slice(&ACI0_MAGIC.to_le_bytes());
        aci0[0x30..0x34].copy_from_slice(&0x40u32.to_le_bytes());
        aci0[0x34..0x38].copy_from_slice(&(4 * caps.len() as u32).to_le_bytes());
        for cap in caps {
            aci0.extend_from_slice(&cap.to_le_bytes());
        }
        data.extend_from_slice(&aci0);
        data
    }

    /// `ThreadInfo` with priorities 28..=59 on cores `min..=max`.
    fn thread_info(min: u32, max: u32) -> u32 {
        0b0111 | 59 << 4 | 28 << 10 | min << 16 | max << 24
    }

    #[test]
    fn the_core_mask_comes_from_thread_info() {
        // An unrelated descriptor (one trailing one) ahead of it is skipped.
        let applet = with_capabilities(&[0b0001, thread_info(3, 3)]);
        assert_eq!(Npdm::parse(&applet).unwrap().core_mask, 0b1000);
        let game = with_capabilities(&[thread_info(0, 2)]);
        assert_eq!(Npdm::parse(&game).unwrap().core_mask, 0b0111);
    }

    #[test]
    fn a_manifest_without_thread_info_gets_the_application_cores() {
        assert_eq!(
            Npdm::parse(&npdm(0)).unwrap().core_mask,
            APPLICATION_CORE_MASK
        );
        let other = with_capabilities(&[0b0001]);
        assert_eq!(
            Npdm::parse(&other).unwrap().core_mask,
            APPLICATION_CORE_MASK
        );
    }

    #[test]
    fn parses_a_manifest() {
        let parsed = Npdm::parse(&npdm(0x0100_0000)).unwrap();
        assert!(parsed.is_64_bit);
        assert_eq!(parsed.system_resource_size, 0x0100_0000);
        assert_eq!(parsed.main_thread_priority, 0x2C);
        assert_eq!(parsed.main_thread_core, 1);
        assert_eq!(parsed.main_thread_stack_size, 0x0010_0000);
        assert_eq!(parsed.name, "Application");
    }

    /// A declared zero must parse as a real value (Just Dance 2019 vs 2023's 16 MiB).
    #[test]
    fn zero_is_a_real_answer() {
        assert_eq!(Npdm::parse(&npdm(0)).unwrap().system_resource_size, 0);
    }

    /// Mario Kart 8 Deluxe's flags byte: bit 0 clear means AArch32.
    #[test]
    fn a_thirty_two_bit_title_says_so_in_bit_zero() {
        let mut data = npdm(0);
        data[0x0C] = 0x04;
        assert!(!Npdm::parse(&data).unwrap().is_64_bit);
    }

    #[test]
    fn rejects_a_file_that_is_not_a_manifest() {
        let mut data = npdm(0);
        data[0] = b'X';
        assert!(matches!(Npdm::parse(&data), Err(Error::BadMagic { .. })));
        assert!(matches!(
            Npdm::parse(&data[..8]),
            Err(Error::Truncated { .. })
        ));
    }
}
