//! Each opened title's NACP and icon, so `ns` can describe it to other programs.

use js_sys::{Object, Reflect, Uint8Array};
use switch_core::trace::Level;
use wasm_bindgen::prelude::*;

use super::{self as idb, NAND_CONTROLS, NAND_DB, NAND_STORES, NAND_VERSION};
use crate::session_if_live;

/// Hand every stored control to the session.
pub(crate) async fn restore() {
    let result = async {
        let db = idb::open(NAND_DB, NAND_VERSION, &NAND_STORES).await?;
        let entries = idb::get_all(&db, NAND_CONTROLS).await;
        db.close();
        entries
    }
    .await;
    let Some(s) = session_if_live() else {
        return;
    };
    match result {
        Ok(entries) => {
            for (id, control) in entries {
                let bytes = |name: &str| {
                    Reflect::get(&control, &name.into())
                        .ok()
                        .and_then(|data| data.dyn_into::<Uint8Array>().ok())
                        .map_or_else(Vec::new, |data| data.to_vec())
                };
                if let Ok(id) = u64::from_str_radix(&id, 16) {
                    s.cpu
                        .add_application_control(id, &bytes("nacp"), &bytes("icon"));
                }
            }
        }
        Err(err) => s.cpu.log(
            Level::Error,
            &format!("Title details: could not be read ({err:?})"),
        ),
    }
}

/// Store the control data the session has just read, for the next session's `ns`.
#[wasm_bindgen]
pub async fn switch_remember_control() {
    let Some(s) = session_if_live() else {
        return;
    };
    let Some(control) = &s.control else {
        return;
    };
    if control.title_id == 0 || control.nacp_raw.is_empty() {
        return;
    }
    let (id, nacp, icon) = (
        format!("{:016x}", control.title_id),
        Uint8Array::from(&control.nacp_raw[..]),
        Uint8Array::from(&control.icon[..]),
    );
    let entry = Object::new();
    Reflect::set(&entry, &"nacp".into(), &nacp).ok();
    Reflect::set(&entry, &"icon".into(), &icon).ok();
    let outcome = write(&id, entry.into()).await;
    if let (Err(err), Some(s)) = (outcome, session_if_live()) {
        s.cpu.log(
            Level::Error,
            &format!("Title details: could not be stored ({err:?})"),
        );
    }
}

async fn write(id: &str, entry: JsValue) -> Result<(), JsValue> {
    let db = idb::open(NAND_DB, NAND_VERSION, &NAND_STORES).await?;
    let outcome = idb::apply(&db, NAND_CONTROLS, &[(id.to_owned(), Some(entry))]).await;
    db.close();
    outcome
}
