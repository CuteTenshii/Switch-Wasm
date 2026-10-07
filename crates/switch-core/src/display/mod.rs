//! The display stack: `vi` binder sessions, [`buffer_queue`] and [`parcel`].

pub mod buffer_queue;
pub mod parcel;

pub use buffer_queue::{Action, BufferQueue};
