//! Profiles the guest edited through `acc`, stored in the page's profile store.

use js_sys::{Reflect, Uint8Array};
use switch_core::trace::Level;
use wasm_bindgen::prelude::*;

use super::{self as idb, NAND_DB, NAND_STORES, NAND_USERS, NAND_VERSION};
use crate::session_if_live;

/// A user as the core holds it: uid in hex, nickname, edit time, picture.
type Edited = (String, String, i64, Option<Vec<u8>>);

/// If the guest edited a profile since the last call, store the core's nickname, edit time
/// and picture over the stored profiles of the same uid. Returns whether it did.
#[wasm_bindgen]
pub async fn switch_store_profile_edits() -> bool {
    let Some(s) = session_if_live() else {
        return false;
    };
    if !s.cpu.take_profile_edits() {
        return false;
    }
    let users: Vec<Edited> = s
        .cpu
        .users()
        .iter()
        .map(|user| {
            let uid = user.uid.iter().map(|byte| format!("{byte:02x}")).collect();
            (
                uid,
                user.nickname.clone(),
                user.edited_at,
                user.picture.clone(),
            )
        })
        .collect();
    match store(&users).await {
        Ok(()) => true,
        Err(err) => {
            if let Some(s) = session_if_live() {
                s.cpu.log(
                    Level::Error,
                    &format!("Profiles: the guest's edits could not be saved ({err:?})"),
                );
            }
            false
        }
    }
}

async fn store(users: &[Edited]) -> Result<(), JsValue> {
    let db = idb::open(NAND_DB, NAND_VERSION, &NAND_STORES).await?;
    let outcome = write(&db, users).await;
    db.close();
    outcome
}

async fn write(db: &web_sys::IdbDatabase, users: &[Edited]) -> Result<(), JsValue> {
    let stored = idb::get_all(db, NAND_USERS).await?;
    let mut changes = Vec::new();
    for (uid, nickname, edited_at, picture) in users {
        let Some((_, profile)) = stored.iter().find(|(key, _)| key == uid) else {
            continue;
        };
        Reflect::set(profile, &"nickname".into(), &nickname.into())?;
        Reflect::set(profile, &"editedAt".into(), &(*edited_at as f64).into())?;
        if let Some(picture) = picture {
            Reflect::set(profile, &"picture".into(), &Uint8Array::from(&picture[..]))?;
        }
        changes.push((uid.clone(), Some(profile.clone())));
    }
    idb::apply(db, NAND_USERS, &changes).await
}
