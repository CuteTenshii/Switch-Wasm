//! A block-translating JIT for the A64 core.
//!
//! [`decode`] translates guest code into cached [`ir::Block`]s that [`exec`]
//! runs without re-decoding. Hot blocks are written out as wasm by [`emit`]
//! and installed through [`host`]; anything without an op falls back to
//! [`ir::Op::Interpret`]. [`cache`] drops blocks whose pages a store touched.

mod cache;
mod decode;
mod emit;
mod exec;
mod host;
pub(in crate::cpu) mod ir;
mod wasm;

pub use cache::JitStats;
pub use decode::translates;
pub use emit::{defers, emits, tail_call_probe, Layout, Refused, LEFT};
pub use exec::HOT;
pub use host::{set_jit_host, Entry, JitHost};

pub(in crate::cpu) use cache::Jit;
