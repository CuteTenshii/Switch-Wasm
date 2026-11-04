//! The console's user accounts.

use crate::session;
use crate::storage::sd_path;

// The console's users: staged one at a time, committed whole before the title starts.
// A uid travels as two little-endian halves.

fn uid_from_halves(lo: u64, hi: u64) -> [u8; 16] {
    let mut uid = [0u8; 16];
    uid[..8].copy_from_slice(&lo.to_le_bytes());
    uid[8..].copy_from_slice(&hi.to_le_bytes());
    uid
}

/// Stage one user. `edited_at` is POSIX seconds; `picture_len` 0 means a generated
/// picture. Pictures are baseline JPEG.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn switch_user_stage(
    handle: u32,
    uid_lo: u64,
    uid_hi: u64,
    name_ptr: *const u8,
    name_len: u32,
    edited_at: i64,
    picture_ptr: *const u8,
    picture_len: u32,
) {
    let name = sd_path(name_ptr, name_len);
    let picture = (picture_len > 0)
        .then(|| unsafe { std::slice::from_raw_parts(picture_ptr, picture_len as usize) }.to_vec());
    let mut user =
        switch_core::cpu::UserAccount::new(uid_from_halves(uid_lo, uid_hi), &name, picture);
    user.edited_at = edited_at;
    session(handle).staged_users.push(user);
}

/// Install the staged users with the given one playing. Returns 0, or (changing
/// nothing) 1 for no users or more than eight, 2 for a zero uid, 3 for a duplicate
/// uid, 4 for a playing user not in the list. The staged list is emptied either way.
#[no_mangle]
pub extern "C" fn switch_users_commit(handle: u32, current_lo: u64, current_hi: u64) -> u32 {
    use switch_core::cpu::UsersRefused;
    let s = session(handle);
    let users = std::mem::take(&mut s.staged_users);
    match s
        .cpu
        .set_users(users, uid_from_halves(current_lo, current_hi))
    {
        Ok(()) => 0,
        Err(UsersRefused::Count) => 1,
        Err(UsersRefused::ZeroUid) => 2,
        Err(UsersRefused::DuplicateUid) => 3,
        Err(UsersRefused::UnknownCurrent) => 4,
    }
}
