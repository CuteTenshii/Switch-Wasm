//! The Horizon kernel: supervisor calls, IPC, threads, scheduling and synchronization.

pub(crate) mod boot;
pub(crate) mod cores;
pub(crate) mod events;
pub(crate) mod handles;
pub(crate) mod ipc;
pub(crate) mod layout;
pub(crate) mod machine;
pub(crate) mod memory_use;
pub(crate) mod process;
pub(crate) mod program;
pub(crate) mod sched;
pub(crate) mod svc;
pub(crate) mod sync;
pub(crate) mod thread;
pub(crate) mod thread_report;
