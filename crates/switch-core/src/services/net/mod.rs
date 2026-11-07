//! The networking services: an empty network by default. Nothing off the console answers,
//! but `bsd` loopback between sockets of this process works. See [`netbridge`](super::netbridge)
//! for how a host can carry traffic outside.

mod bsd;
mod dns;
mod nifm;
mod ssl;

pub(crate) use bsd::{BsdSocket, BSD_FIRST_EPHEMERAL_PORT};
pub(crate) use ssl::{SslCertificate, SslConnection};
