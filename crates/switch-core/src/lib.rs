//! switch-core: a from-scratch Nintendo Switch emulation core targeting the
//! browser (WASM) and the host for testing.
//!
//! Phase 0 provides container/format parsers: PFS0 (`.nsp`), NCA headers and
//! the homebrew NRO/ELF loaders.
//!
//! Phase 1 provides a full AArch64 integer interpreter ([`cpu::Cpu`]) that can
//! boot hand-assembled and simple compiled homebrew.
//!
//! Commercial game content (NCA) can be decrypted and its main executable
//! (NSO0) loaded when the caller supplies `prod.keys`/`title.keys` — see
//! [`nca`] and [`nso`]. That only gets a real title as far as its own crt0;
//! actually running one needs the Horizon service surface a retail SDK
//! program expects, which is a much larger undertaking than homebrew ever
//! needed and is tracked separately in `PROGRESS.md`.

pub mod control;
pub mod cpu;
pub mod crypto;
pub mod disasm;
pub mod display;
pub mod elf;
pub mod error;
pub mod gpu;
pub mod keys;
pub mod lz4;
pub mod mem;
pub mod nca;
pub mod nro;
pub mod nso;
pub mod nsp;
pub mod romfs;
pub mod source;
pub mod ticket;
pub mod vfs;

pub use error::{Error, Result};

/// Memory-mapped framebuffer (modelled on the Switch GPU's): fixed-size,
/// little-endian RGBA. Homebrew writes pixels here and the host renders it.
///
/// It and [`INPUT_ADDR`] sit above every region a Horizon process is given —
/// see `cpu::GUEST_SPACE_END`. They used to live at 0x3F00_0000, immediately
/// after a 240 MiB heap region; the heap now needs the address space they
/// were standing in.
pub const FB_BASE: u32 = 0xF200_0000;
pub const FB_WIDTH: u32 = 640;
pub const FB_HEIGHT: u32 = 360;
pub const FB_STRIDE: u32 = FB_WIDTH * 4;
/// Memory-mapped input register: the host writes an ASCII key here and
/// homebrew acknowledges (writes 0) when consumed.
pub const INPUT_ADDR: u32 = 0xF210_0000;
