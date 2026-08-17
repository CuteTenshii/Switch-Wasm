//! switch-core: a from-scratch Nintendo Switch emulation core targeting the
//! browser (WASM) and the host for testing.
//!
//! Phase 0 provides container/format parsers: PFS0 (`.nsp`), NCA headers and
//! the homebrew NRO/ELF loaders.
//!
//! Phase 1 provides a full AArch64 integer interpreter ([`cpu::Cpu`]) that can
//! boot hand-assembled and simple compiled homebrew.
//!
//! Commercial game content is encrypted and requires console keys; that is
//! deliberately out of scope.

pub mod cpu;
pub mod crypto;
pub mod disasm;
pub mod display;
pub mod elf;
pub mod error;
pub mod gpu;
pub mod keys;
pub mod mem;
pub mod nca;
pub mod nro;
pub mod nsp;
pub mod vfs;

pub use error::{Error, Result};

/// Memory-mapped framebuffer (modelled on the Switch GPU's): fixed-size,
/// little-endian RGBA. Homebrew writes pixels here and the host renders it.
pub const FB_BASE: u32 = 0x3F00_0000;
pub const FB_WIDTH: u32 = 640;
pub const FB_HEIGHT: u32 = 360;
pub const FB_STRIDE: u32 = FB_WIDTH * 4;
/// Memory-mapped input register: the host writes an ASCII key here and
/// homebrew acknowledges (writes 0) when consumed.
pub const INPUT_ADDR: u32 = 0x3F10_0000;
