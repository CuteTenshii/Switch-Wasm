//! The display stack between an app and the screen.
//!
//! An app talks to `vi` to get an `IHOSBinderDriver` session, then drives an
//! `IGraphicBufferProducer` over Android's binder protocol: parcels in,
//! parcels out. [`buffer_queue`] implements that producer and [`parcel`] the
//! serialization it rides on.

pub mod buffer_queue;
pub mod parcel;

pub use buffer_queue::{Action, BufferQueue};
