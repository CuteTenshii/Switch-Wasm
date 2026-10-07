//! Horizon's services: one module per domain, dispatched from `kernel::svc` over `kernel::ipc`.

pub(crate) mod acc;
pub(crate) mod am;
pub(crate) mod audio;
pub(crate) mod audout;
pub(crate) mod audren;
pub(crate) mod erpt;
pub(crate) mod fonts;
pub(crate) mod fs;
pub(crate) mod hid;
pub(crate) mod hwopus;
pub(crate) mod input;
pub(crate) mod ldr;
pub(crate) mod log;
pub(crate) mod mii;
pub(crate) mod net;
pub(crate) mod ns;
pub(crate) mod nv;
pub(crate) mod online;
pub(crate) mod pl;
pub(crate) mod power;
pub(crate) mod settings;
pub(crate) mod storage;
pub(crate) mod time;
pub(crate) mod vi;
