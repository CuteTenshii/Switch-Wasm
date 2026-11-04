//! Firmware NCAs from the NAND.

use switch_core::nca::Nca;

use crate::boot::load_and_boot_nca;
use crate::{session, Added, HostSource};

/// Identify a firmware NCA from its header: writes the content type to `kind_out`
/// (0 program, 1 data archive, 2 other) and returns the title id, or 0.
#[no_mangle]
pub extern "C" fn switch_nand_identify(
    handle: u32,
    file: u32,
    size: u64,
    kind_out: *mut u32,
) -> u64 {
    let s = session(handle);
    let src = HostSource { file, len: size };
    let nca = match Nca::parse_source(&src, Some(&s.keys)) {
        Ok(nca) => nca,
        Err(e) => {
            s.last_error = e.to_string();
            return 0;
        }
    };
    use switch_core::nca::ContentType;
    let kind = match nca.content_type {
        ContentType::Program => 0,
        ContentType::Data | ContentType::PublicData => 1,
        _ => 2,
    };
    if !kind_out.is_null() {
        unsafe { *kind_out = kind };
    }
    s.last_error.clear();
    nca.title_id
}

/// Boot a Program NCA from the NAND (an applet). Returns the entry address, or -1
/// with the reason in `switch_last_error`.
#[no_mangle]
pub extern "C" fn switch_nand_launch(handle: u32, ptr: *const u8, len: u32) -> i64 {
    let s = session(handle);
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) }.to_vec();
    load_and_boot_nca(
        &s.keys,
        &mut s.cpu,
        &mut s.last_error,
        switch_core::source::MemSource(bytes),
        Added::default(),
    )
}
