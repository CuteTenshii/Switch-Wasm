//! Library applets installed on the NAND.

use switch_core::nca::{ContentType, Nca};

use crate::{session, HostSource};

/// Register a Program NCA from the NAND as an installed library applet. Returns 0,
/// or -1 with the reason in `switch_last_error`.
#[no_mangle]
pub extern "C" fn switch_add_applet(handle: u32, file: u32, size: u64) -> i32 {
    let s = session(handle);
    let src = HostSource { file, len: size };
    match Nca::parse_source(&src, Some(&s.keys)) {
        Ok(nca) if nca.content_type == ContentType::Program => {
            s.applets.insert(nca.program_id, src);
            s.cpu.set_launchable_applets(s.applets.keys().copied());
            s.last_error.clear();
            0
        }
        Ok(_) => {
            s.last_error = "not a Program NCA".into();
            -1
        }
        Err(e) => {
            s.last_error = e.to_string();
            -1
        }
    }
}
