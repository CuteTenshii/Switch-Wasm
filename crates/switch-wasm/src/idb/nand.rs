//! The NAND's stored firmware, registered with the session: system data archives, library
//! applets, and the SYSTEM partition's listing.

use std::collections::HashMap;

use js_sys::{Array, Reflect};
use switch_core::trace::Level;
use wasm_bindgen::prelude::*;
use web_sys::{Blob, File};

use super::{self as idb, content, NAND_DB, NAND_STORES, NAND_TITLES, NAND_VERSION};
use crate::hostfiles::add as add_host_file;
use crate::post::post;
use crate::{applets, container, session_if_live, HostSource};

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
pub async fn switch_restore_nand() -> u32 {
    match restore().await {
        Ok(registered) => {
            if registered > 0 {
                if let Some(s) = session_if_live() {
                    s.cpu.log(
                        Level::Info,
                        &format!("NAND: {registered} system data archive(s) restored."),
                    );
                }
            }
            registered
        }
        Err(err) => {
            if let Some(s) = session_if_live() {
                s.cpu.log(
                    Level::Error,
                    &format!("NAND: the archives could not be registered ({err:?})"),
                );
            }
            0
        }
    }
}

async fn restore() -> Result<u32, JsValue> {
    let db = idb::open(NAND_DB, NAND_VERSION, &NAND_STORES).await?;
    let titles = idb::get_all(&db, NAND_TITLES).await;
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
    let mut content: HashMap<String, Blob> = HashMap::new();
    for title in &titles {
        if content.contains_key(&title.name) {
            continue;
        }
        match content::lend(&title.name).await {
            Ok(Some(blob)) => {
                if let Some(file) = as_file(&blob, &title.name) {
                    content.insert(title.name.clone(), file);
                }
            }
            Ok(None) => {}
            Err(err) => {
                if let Some(s) = session_if_live() {
                    s.cpu.log(
                        Level::Warn,
                        &format!("[io] NAND content \"{}\": unreadable ({err:?})", title.name),
                    );
                }
            }
        }
    }
    let Some(s) = session_if_live() else {
        return Ok(0);
    };

    // One host file per stored NCA, shared by whatever is registered from it.
    let mut hosts: HashMap<&str, HostSource> = HashMap::new();
    for title in &titles {
        if let Some(blob) = content.get(&title.name) {
            let len = blob.size() as u64;
            hosts.insert(
                &title.name,
                HostSource {
                    file: add_host_file(blob),
                    len,
                },
            );
        }
    }

    let (mut registered, mut bytes) = (0, 0);
    let archives: Vec<&Title> = titles.iter().filter(|t| t.kind == KIND_ARCHIVE).collect();
    post_progress(0, archives.len());
    for (done, title) in archives.iter().enumerate() {
        let Some(&src) = hosts.get(title.name.as_str()) else {
            post_progress(done + 1, archives.len());
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
        post_progress(done + 1, archives.len());
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

/// Tell the page how many archives are registered so far.
fn post_progress(done: usize, total: usize) {
    post(
        "nand_progress",
        &[
            ("done", JsValue::from_f64(done as f64)),
            ("total", JsValue::from_f64(total as f64)),
        ],
    );
}

/// Stored content as a `File` named `name`, so host file reports name it.
fn as_file(content: &Blob, name: &str) -> Option<Blob> {
    if content.is_instance_of::<File>() {
        return Some(content.clone());
    }
    File::new_with_blob_sequence(&Array::of1(content), name)
        .ok()
        .map(Into::into)
}
