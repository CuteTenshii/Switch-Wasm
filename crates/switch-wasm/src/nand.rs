//! Firmware NCAs from the NAND.

use switch_core::json;
use switch_core::nca::Nca;
use wasm_bindgen::prelude::wasm_bindgen;

use crate::session;
use crate::HostSource;

/// Identify a firmware NCA from its header as `{ id, kind }` JSON, where `kind` is 0 for a
/// program, 1 for a data archive and 2 for anything else; empty when it cannot be read.
#[wasm_bindgen]
pub fn switch_nand_identify(file: u32, size: u64) -> String {
    let s = session();
    let src = HostSource { file, len: size };
    let nca = match Nca::parse_source(&src, Some(&s.keys)) {
        Ok(nca) => nca,
        Err(e) => {
            s.last_error = e.to_string();
            return String::new();
        }
    };
    use switch_core::nca::ContentType;
    let kind = match nca.content_type {
        ContentType::Program => 0,
        ContentType::Data | ContentType::PublicData => 1,
        _ => 2,
    };
    s.last_error.clear();
    json::object(|o| {
        o.field("id", format!("{:016x}", nca.title_id))
            .field("kind", kind);
    })
}

/// Boot a Program NCA from the NAND (an applet), read by range from the browser's copy
/// rather than copied into linear memory. Returns the entry address, or fails with the reason.
#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
#[wasm_bindgen]
pub fn switch_nand_launch(content: &web_sys::Blob) -> Result<f64, String> {
    use crate::boot::load_and_boot_nca;
    use crate::Added;
    let s = session();
    let src = HostSource {
        file: crate::add_host_file(content),
        len: content.size() as u64,
    };
    load_and_boot_nca(&s.keys, &mut s.cpu, src, Added::default()).map(|entry| entry as f64)
}
