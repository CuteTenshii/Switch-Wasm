use std::fmt;

/// Why a program could not be decoded, translated or run.
#[derive(Debug)]
pub enum Error {
    /// The program, or something it did, is malformed or not modelled.
    Fault(String),
    /// An error a host's constant, memory or texture source returned, kept whole so the host
    /// can take it back unchanged.
    Host(Box<dyn std::error::Error + Send + Sync>),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Fault(message) => f.write_str(message),
            Error::Host(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Fault(_) => None,
            // Displayed as the host error itself, so its source is the host error's.
            Error::Host(error) => error.source(),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
