//! The NAND's stored firmware, registered with the session: system data archives, library
//! applets, and the SYSTEM partition's listing.

use std::collections::HashMap;

use js_sys::{Array, ArrayBuffer, Reflect};
use switch_core::trace::Level;
use wasm_bindgen::prelude::*;
use web_sys::{Blob, File};

use super::{self as idb, NAND_CONTENT, NAND_DB, NAND_STORES, NAND_TITLES, NAND_VERSION};
use crate::{add_host_file, applets, container, session_if_live, HostSource};

/// Program ids of the library applets a title or the Home Menu can start.
const LIBRARY_APPLETS: [u64; 15] = [
    0x0100_0000_0000_1001,
    0x0100_0000_0000_1002,
    0x0100_0000_0000_1003,
    0x0100_0000_0000_1004,
    0x0100_0000_0000_1005,
    0x0100_0000_0000_1006,
    0x0100_0000_0000_1007,
    0x0100_0000_0000_1008,
    0x0100_0000_0000_1009,
    0x0100_0000_0000_100a,
    0x0100_0000_0000_100b,
    0x0100_0000_0000_100d,
    0x0100_0000_0000_100f,
    0x0100_0000_0000_1010,
    0x0100_0000_0000_1011,
];

/// `NandEntry.kind`: a program, or a data archive.
const KIND_PROGRAM: f64 = 0.0;
const KIND_ARCHIVE: f64 = 1.0;

struct Title {
    id: u64,
    name: String,
    kind: f64,
}

/// Register what the NAND holds with the session. Returns how many system data archives
/// were registered.
#[wasm_bindgen]
pub async fn switch_restore_nand(handle: u32) -> u32 {
    match restore(handle).await {
        Ok(registered) => registered,
        Err(err) => {
            if let Some(s) = session_if_live(handle) {
                s.cpu.log(
                    Level::Error,
                    &format!("NAND: the archives could not be registered ({err:?})"),
                );
            }
            0
        }
    }
}

async fn restore(handle: u32) -> Result<u32, JsValue> {
    let db = idb::open(NAND_DB, NAND_VERSION, &NAND_STORES).await?;
    let titles = idb::get_all(&db, NAND_TITLES).await;
    let content = idb::get_all(&db, NAND_CONTENT).await;
    db.close();
    let titles: Vec<Title> = titles?
        .into_iter()
        .filter_map(|(id, entry)| {
            Some(Title {
                id: u64::from_str_radix(&id, 16).ok()?,
                name: Reflect::get(&entry, &"name".into()).ok()?.as_string()?,
                kind: Reflect::get(&entry, &"kind".into()).ok()?.as_f64()?,
            })
        })
        .collect();
    let content: HashMap<String, JsValue> = content?.into_iter().collect();
    let Some(s) = session_if_live(handle) else {
        return Ok(0);
    };

    // One host file per stored NCA, shared by whatever is registered from it.
    let mut hosts: HashMap<&str, HostSource> = HashMap::new();
    for title in &titles {
        if let Some(blob) = content
            .get(&title.name)
            .and_then(|c| as_file(c, &title.name))
        {
            let len = blob.size() as u64;
            hosts.insert(
                &title.name,
                HostSource {
                    file: add_host_file(&blob),
                    len,
                },
            );
        }
    }

    let (mut registered, mut bytes) = (0, 0);
    for title in titles.iter().filter(|t| t.kind == KIND_ARCHIVE) {
        let Some(&src) = hosts.get(title.name.as_str()) else {
            continue;
        };
        if container::add_archive(s, src) < 0 {
            let reason = s.last_error.clone();
            s.cpu.log(
                Level::Warn,
                &format!("[io] system archive \"{}\": refused ({reason})", title.name),
            );
        } else {
            registered += 1;
            bytes += src.len;
        }
    }
    if registered > 0 {
        s.cpu.log(
            Level::Debug,
            &format!(
                "[io] registered {registered} system archives ({:.2} MiB)",
                bytes as f64 / f64::from(1 << 20)
            ),
        );
    }
    for title in titles
        .iter()
        .filter(|t| t.kind == KIND_PROGRAM && LIBRARY_APPLETS.contains(&t.id))
    {
        let Some(&src) = hosts.get(title.name.as_str()) else {
            continue;
        };
        if applets::add_applet(s, src) < 0 {
            let reason = s.last_error.clone();
            s.cpu.log(
                Level::Warn,
                &format!("[io] applet \"{}\": refused ({reason})", title.name),
            );
        }
    }
    for title in &titles {
        if let Some(&src) = hosts.get(title.name.as_str()) {
            s.cpu.add_nand_content(&title.name, std::rc::Rc::new(src));
        }
    }
    Ok(registered)
}

/// Stored content as a `File` named `name`: a `Blob`, or an `ArrayBuffer` from older builds.
fn as_file(content: &JsValue, name: &str) -> Option<Blob> {
    if let Some(file) = content.dyn_ref::<File>() {
        return Some(file.clone().into());
    }
    let parts = Array::of1(content);
    if content.is_instance_of::<Blob>() || content.is_instance_of::<ArrayBuffer>() {
        return File::new_with_blob_sequence(&parts, name)
            .ok()
            .map(Into::into);
    }
    None
}
