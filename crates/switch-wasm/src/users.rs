//! The console's user accounts.

use wasm_bindgen::prelude::wasm_bindgen;

use crate::session;

// The console's users: staged one at a time, committed whole before the title starts.

/// A uid from its 32 hex digits in memory byte order; zero, which a commit refuses, if it is
/// not that.
fn uid_from_hex(text: &str) -> [u8; 16] {
    let mut uid = [0u8; 16];
    if text.len() == 32 {
        for (byte, pair) in uid.iter_mut().zip(text.as_bytes().chunks(2)) {
            let parsed = std::str::from_utf8(pair)
                .ok()
                .and_then(|pair| u8::from_str_radix(pair, 16).ok());
            match parsed {
                Some(value) => *byte = value,
                None => return [0; 16],
            }
        }
    }
    uid
}

/// Stage one user. `edited_at` is POSIX seconds; no picture means a generated one. Pictures
/// are baseline JPEG.
#[wasm_bindgen]
pub fn switch_user_stage(uid: &str, nickname: &str, edited_at: f64, picture: Option<Vec<u8>>) {
    let picture = picture.filter(|picture| !picture.is_empty());
    let mut user = switch_core::cpu::UserAccount::new(uid_from_hex(uid), nickname, picture);
    user.edited_at = edited_at as i64;
    session().staged_users.push(user);
}

/// Install the staged users with the given one playing. Returns 0, or (changing
/// nothing) 1 for no users or more than eight, 2 for a zero uid, 3 for a duplicate
/// uid, 4 for a playing user not in the list. The staged list is emptied either way.
#[wasm_bindgen]
pub fn switch_users_commit(current: &str) -> u32 {
    use switch_core::cpu::UsersRefused;
    let s = session();
    let users = std::mem::take(&mut s.staged_users);
    match s.cpu.set_users(users, uid_from_hex(current)) {
        Ok(()) => 0,
        Err(UsersRefused::Count) => 1,
        Err(UsersRefused::ZeroUid) => 2,
        Err(UsersRefused::DuplicateUid) => 3,
        Err(UsersRefused::UnknownCurrent) => 4,
    }
}
