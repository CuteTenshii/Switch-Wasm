//! Library applets installed on the NAND.

use std::rc::Rc;

use switch_core::cpu::Installed;
use switch_core::nca::{ContentType, Nca};
use switch_core::source::ByteSource;

use crate::{HostSource, Session};

/// Install a Program NCA from the NAND, so the kernel can start it as a library applet.
/// Returns 0, or -1 with the reason in `switch_last_error`.
pub(crate) fn add_applet(s: &mut Session, src: HostSource) -> i32 {
    let installed = Nca::parse_source(&src, Some(&s.keys)).and_then(|nca| {
        if nca.content_type != ContentType::Program {
            return Err(switch_core::Error::Nca("not a Program NCA".into()));
        }
        let exefs_index = nca
            .exefs_section_index()
            .ok_or_else(|| switch_core::Error::Nca("no ExeFS section".into()))?;
        let exefs: Rc<dyn ByteSource> = Rc::new(nca.section_source(src, &s.keys, exefs_index)?);
        let romfs = match nca.romfs_section_index() {
            Some(index) => {
                Some(Rc::new(nca.romfs_source(src, &s.keys, index)?) as Rc<dyn ByteSource>)
            }
            None => None,
        };
        Ok(Installed {
            nca,
            exefs,
            exefs_index,
            romfs,
        })
    });
    match installed {
        Ok(installed) => {
            s.cpu.install_program(installed);
            s.last_error.clear();
            0
        }
        Err(e) => {
            s.last_error = e.to_string();
            -1
        }
    }
}
