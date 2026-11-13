//! A VP9 decoder: profile 0 (8-bit 4:2:0), bit-exact with libvpx.

// Ported from the reference decoder's index arithmetic, which these lints would obscure.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_memcpy,
    dead_code
)]

mod block;
mod bool_decoder;
pub mod decoder;
mod frame;
mod header;
pub mod idct;
mod inter;
pub mod intra;
mod loopfilter;
mod modeinfo;
pub mod nvdec;
mod probs;
mod tables;
mod tokens;
mod transforms_1d;
mod types;

pub use decoder::Decoder;
pub use frame::{Frame, Plane};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Truncated,
    Corrupt(&'static str),
    Unsupported(&'static str),
}
