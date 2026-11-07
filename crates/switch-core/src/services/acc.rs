//! `acc`: the console's user accounts and their profile pictures. Users are
//! set by the host ([`Cpu::set_users`]) and always signed in; a user without a
//! picture gets a synthesized JPEG.

use crate::cpu::Cpu;
use crate::Result;

/// A new user's uid: 128 random bits, never all zero, which titles read as "nobody".
pub fn random_uid() -> [u8; 16] {
    use std::hash::{BuildHasher, Hasher};
    loop {
        // A `RandomState` carries keys the process picked at random.
        let keys = std::collections::hash_map::RandomState::new();
        let mut uid = [0u8; 16];
        for (half, tag) in uid.as_chunks_mut::<8>().0.iter_mut().zip(0u64..) {
            let mut hasher = keys.build_hasher();
            hasher.write_u64(tag);
            half.copy_from_slice(&hasher.finish().to_le_bytes());
        }
        if uid != [0; 16] {
            return uid;
        }
    }
}

/// The uid of the console's first user, made the first time any console is and the same for
/// every console in the process: two consoles run in lockstep must not differ in who plays.
/// A page replaces that user with its own profiles, whose uids it makes when they are created.
pub(crate) fn default_uid() -> [u8; 16] {
    static UID: std::sync::OnceLock<[u8; 16]> = std::sync::OnceLock::new();
    *UID.get_or_init(random_uid)
}

pub const MAX_USERS: usize = 8;

/// Mixed into a picture's id so it never equals the uid.
const PROFILE_IMAGE_ID: [u8; 16] = *b"switch-wasm icon";

/// `nn::account::ProfileBase`: uid, last-edit timestamp, then the nickname.
const PROFILE_BASE_LEN: usize = 0x38;

/// `nn::account::UserData` (icon id, background colour, mii id).
const ACCOUNT_USER_DATA_LEN: usize = 0x80;

/// acc's "that user does not exist" (module 124, description 100).
const ACCOUNT_USER_NOT_EXIST: u32 = 124 | (100 << 9);
const ACCOUNT_REGISTRATION_REQUIRED: u32 = 124 | (200 << 9);

/// Arbitrary but nonzero: zero means "no account".
const NETWORK_SERVICE_ACCOUNT_ID: u64 = 0x0000_0001_0000_0001;

pub const NICKNAME_LEN: usize = 0x20;

pub(crate) const DEFAULT_NICKNAME: &str = "Player";

/// Real profile icons are 256x256.
const PROFILE_IMAGE_SIZE: u16 = 256;

/// No-picture colours, picked by uid (the page picks the same way).
const PROFILE_IMAGE_COLORS: [(u8, u8, u8); 8] = [
    (0x4B, 0x50, 0x5A),
    (0x2F, 0x6F, 0xB5),
    (0xC0, 0x4A, 0x3C),
    (0x3E, 0x8E, 0x5A),
    (0xB0, 0x7A, 0x1E),
    (0x7A, 0x4F, 0xA8),
    (0x1F, 0x8A, 0x8C),
    (0xB4, 0x4C, 0x86),
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserAccount {
    pub uid: [u8; 16],
    /// At most [`NICKNAME_LEN`] - 1 bytes of UTF-8.
    pub nickname: String,
    /// Last edit as POSIX seconds, 0 if never edited.
    pub edited_at: i64,
    /// A baseline JPEG; `None` for a synthesized one.
    pub picture: Option<Vec<u8>>,
}

impl UserAccount {
    pub fn new(uid: [u8; 16], nickname: &str, picture: Option<Vec<u8>>) -> UserAccount {
        UserAccount {
            uid,
            nickname: fit_nickname(nickname),
            edited_at: 0,
            picture,
        }
    }

    fn image(&self) -> Vec<u8> {
        match &self.picture {
            Some(picture) => picture.clone(),
            None => profile_image(self.uid),
        }
    }

    /// The `IProfile::GetImageId` uuid: two seeded FNV-1a hashes of the uid and
    /// picture, so it changes exactly when the picture does.
    fn image_id(&self) -> [u8; 16] {
        let picture = self.picture.as_deref().unwrap_or(&[]);
        let hash = |seed: &[u8]| {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for &byte in seed.iter().chain(&self.uid).chain(picture) {
                h ^= u64::from(byte);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
            h
        };
        let mut id = [0u8; 16];
        id[..8].copy_from_slice(&hash(&PROFILE_IMAGE_ID[..8]).to_le_bytes());
        id[8..].copy_from_slice(&hash(&PROFILE_IMAGE_ID[8..]).to_le_bytes());
        id
    }
}

/// Cut to the 0x1F bytes `nn::account::Nickname` holds, on a char boundary.
fn fit_nickname(nickname: &str) -> String {
    let mut end = nickname.len().min(NICKNAME_LEN - 1);
    while end > 0 && !nickname.is_char_boundary(end) {
        end -= 1;
    }
    nickname[..end].to_owned()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsersRefused {
    /// No users, or more than [`MAX_USERS`].
    Count,
    /// A zero uid, which titles read as "nobody".
    ZeroUid,
    DuplicateUid,
    UnknownCurrent,
}

impl std::fmt::Display for UsersRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            UsersRefused::Count => "a console holds one to eight users",
            UsersRefused::ZeroUid => "a zero uid means nobody",
            UsersRefused::DuplicateUid => "two users share a uid",
            UsersRefused::UnknownCurrent => "the user playing is not one of them",
        })
    }
}

const JPEG_SOI: u8 = 0xD8;

const JPEG_APP0: u8 = 0xE0;

const JPEG_DQT: u8 = 0xDB;

const JPEG_SOF0: u8 = 0xC0;

const JPEG_DHT: u8 = 0xC4;

const JPEG_SOS: u8 = 0xDA;

const JPEG_EOI: u8 = 0xD9;

/// 8 makes a constant block's DC coefficient (`8x`) quantize to exactly `x`.
const JPEG_QUANT: u8 = 8;

const JPEG_EOB: u8 = 0x00;

/// DC Huffman table: twelve categories, four in three bits and eight in four.
const JPEG_DC_BITS: [u8; 16] = [0, 0, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

const JPEG_DC_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// AC Huffman table: EOB and ZRL, one bit each (a complete code).
const JPEG_AC_BITS: [u8; 16] = [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

const JPEG_AC_VALUES: [u8; 2] = [JPEG_EOB, 0xF0];

/// The picture of a user who has none: a solid field in the user's colour.
fn profile_image(uid: [u8; 16]) -> Vec<u8> {
    solid_jpeg(PROFILE_IMAGE_SIZE, picture_color(uid))
}

/// u32 so the pick is the same on wasm32 and 64-bit hosts.
fn picture_color(uid: [u8; 16]) -> (u8, u8, u8) {
    let spread = uid.iter().fold(0u32, |sum, &byte| {
        sum.wrapping_mul(31).wrapping_add(u32::from(byte))
    });
    PROFILE_IMAGE_COLORS[spread as usize % PROFILE_IMAGE_COLORS.len()]
}

/// A baseline JPEG of a single solid colour, `size` x `size` pixels. Each
/// block is a DC difference and EOB; with a quantization table of 8 the colour
/// round-trips exactly. The Huffman tables are minimal but complete.
fn solid_jpeg(size: u16, rgb: (u8, u8, u8)) -> Vec<u8> {
    let (red, green, blue) = (f32::from(rgb.0), f32::from(rgb.1), f32::from(rgb.2));
    let round = |value: f32| value.round().clamp(0.0, 255.0) as i32;
    // JFIF RGB -> YCbCr (BT.601).
    let components = [
        round(0.299 * red + 0.587 * green + 0.114 * blue),
        round(-0.168_736 * red - 0.331_264 * green + 0.5 * blue + 128.0),
        round(0.5 * red - 0.418_688 * green - 0.081_312 * blue + 128.0),
    ];

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(&[0xFF, JPEG_SOI]);
    segment(
        &mut out,
        JPEG_APP0,
        b"JFIF\0\x01\x01\x00\x00\x01\x00\x01\x00\x00",
    );
    // One quantization table (id 0), 8-bit precision, for all components.
    let mut quant = vec![0u8];
    quant.extend_from_slice(&[JPEG_QUANT; 64]);
    segment(&mut out, JPEG_DQT, &quant);
    // SOF0: 8-bit samples, `size` square, three components sampled 1x1.
    let mut frame = vec![8];
    frame.extend_from_slice(&size.to_be_bytes());
    frame.extend_from_slice(&size.to_be_bytes());
    frame.push(3);
    for id in 1..=3u8 {
        frame.extend_from_slice(&[id, 0x11, 0]);
    }
    segment(&mut out, JPEG_SOF0, &frame);
    for (class, bits, values) in [
        (0x00u8, &JPEG_DC_BITS, &JPEG_DC_VALUES[..]),
        (0x10u8, &JPEG_AC_BITS, &JPEG_AC_VALUES[..]),
    ] {
        let mut table = vec![class];
        table.extend_from_slice(bits);
        table.extend_from_slice(values);
        segment(&mut out, JPEG_DHT, &table);
    }
    // SOS: all three components with table pair 0, baseline sequential scan.
    let mut scan = vec![3];
    for id in 1..=3u8 {
        scan.extend_from_slice(&[id, 0x00]);
    }
    scan.extend_from_slice(&[0, 63, 0]);
    segment(&mut out, JPEG_SOS, &scan);

    let dc_codes = huffman_codes(&JPEG_DC_BITS, &JPEG_DC_VALUES);
    let ac_codes = huffman_codes(&JPEG_AC_BITS, &JPEG_AC_VALUES);
    let code_for = |codes: &[(u8, u16, u8)], symbol: u8| -> (u32, u32) {
        let (_, code, length) = codes
            .iter()
            .find(|&&(candidate, _, _)| candidate == symbol)
            .expect("the tables above cover every symbol this emits");
        (u32::from(*code), u32::from(*length))
    };

    let mcus = u32::from(size).div_ceil(8) * u32::from(size).div_ceil(8);
    let mut bits = JpegBits::default();
    for mcu in 0..mcus {
        for &component in &components {
            // Level shift; only the first block of each component has a nonzero DC difference.
            let diff = if mcu == 0 { component - 128 } else { 0 };
            let category = if diff == 0 {
                0
            } else {
                32 - diff.unsigned_abs().leading_zeros()
            };
            let (code, length) = code_for(&dc_codes, category as u8);
            bits.push(code, length);
            if category > 0 {
                // A negative difference is sent as its one's complement.
                let value = if diff > 0 {
                    diff
                } else {
                    diff + (1 << category) - 1
                };
                bits.push(value as u32, category);
            }
            let (code, length) = code_for(&ac_codes, JPEG_EOB);
            bits.push(code, length);
        }
    }
    out.extend_from_slice(&bits.finish());
    out.extend_from_slice(&[0xFF, JPEG_EOI]);
    out
}

/// A marker segment: `FF <marker>`, the length including itself, the payload.
fn segment(out: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    out.extend_from_slice(&[0xFF, marker]);
    out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(payload);
}

/// Canonical Huffman codes from `BITS`/`HUFFVAL` as `(symbol, code, length)`,
/// per Annex C.
fn huffman_codes(bits: &[u8; 16], values: &[u8]) -> Vec<(u8, u16, u8)> {
    let mut codes = Vec::with_capacity(values.len());
    let mut code = 0u16;
    let mut next = 0usize;
    for (index, &count) in bits.iter().enumerate() {
        for _ in 0..count {
            codes.push((values[next], code, index as u8 + 1));
            code += 1;
            next += 1;
        }
        code <<= 1;
    }
    codes
}

#[derive(Default)]
struct JpegBits {
    out: Vec<u8>,
    accumulator: u32,
    filled: u32,
}

impl JpegBits {
    fn push(&mut self, code: u32, length: u32) {
        for shift in (0..length).rev() {
            self.accumulator = (self.accumulator << 1) | ((code >> shift) & 1);
            self.filled += 1;
            if self.filled == 8 {
                let byte = self.accumulator as u8;
                self.out.push(byte);
                // Byte stuffing: 0xFF is followed by 0x00.
                if byte == 0xFF {
                    self.out.push(0x00);
                }
                self.accumulator = 0;
                self.filled = 0;
            }
        }
    }

    /// Pad the final partial byte with 1 bits.
    fn finish(mut self) -> Vec<u8> {
        while self.filled != 0 {
            self.push(1, 1);
        }
        self.out
    }
}

impl Cpu {
    /// `acc:u0`, `acc:u1` and `acc:su`. Commands 0..=51 are shared; from 100 up
    /// the same id means different things per service, so those arms
    /// dispatch on the service the session was opened under.
    pub(crate) fn acc_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    // The domain object inherits the service name, which decides its 100+ commands.
                    let name = self.service_name(handle).unwrap_or("acc:u0").to_string();
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, &name);
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, "acc:control", cmd_id),
            };
        }
        let object_id = self.ipc_domain_object_id(tls);
        let iface = if self.ipc_is_domain_request(tls) {
            self.domain_interface(handle, object_id)
                .unwrap_or("acc:u0")
                .to_string()
        } else {
            match self.service_name(handle) {
                Some(name) => name.to_string(),
                None => "acc:u0".to_string(),
            }
        };
        match iface.as_str() {
            "acc:u0" | "acc:u1" | "acc:su" => {
                self.acc_user_service_request(tls, handle, &iface, cmd_id)
            }
            "acc:profile" | "acc:profile-editor" => {
                self.acc_profile_request(tls, handle, &iface, cmd_id)
            }
            "acc:manager" => self.acc_manager_request(tls, handle, cmd_id),
            "acc:administrator" => match cmd_id {
                // IsRegistered: the network service account `GetAccountId` reports.
                Some(200) => self.write_ipc_response(tls, 0, &[], &[1u8], &[]),
                // IsLinkedWithNintendoAccount: no Nintendo Account here.
                Some(250) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                _ => self.acc_manager_request(tls, handle, cmd_id),
            },
            "acc:async-context" => self.acc_async_context_request(tls, cmd_id),
            // `INotifier::GetSystemEvent`: a real event that is never signalled.
            "acc:notifier" => match cmd_id {
                Some(0) => {
                    let event = self.alloc_event("acc:notifier", false);
                    self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }

    /// Commands on the account service itself; `iface` is the service opened.
    fn acc_user_service_request(
        &mut self,
        tls: u32,
        handle: u64,
        iface: &str,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        let application = iface == "acc:u0";
        match cmd_id {
            // GetUserCount -> s32.
            Some(0) => {
                let count = self.users.len() as i32;
                self.write_ipc_response(tls, 0, &[], &count.to_le_bytes(), &[])
            }
            // GetUserExistence(AccountUid) -> bool.
            Some(1) => {
                let exists = self.user(self.acc_requested_uid(tls)).is_some();
                self.write_ipc_response(tls, 0, &[], &[u8::from(exists)], &[])
            }
            // ListAllUsers / ListQualifiedUsers: every user.
            Some(2) | Some(141) => {
                let uids: Vec<[u8; 16]> = self.users.iter().map(|user| user.uid).collect();
                self.acc_write_user_list(tls, &uids)
            }
            // ListOpenUsers / ListOpenContextStoredUsers: the user playing.
            Some(3) | Some(60) => {
                let uid = self.current_user().uid;
                self.acc_write_user_list(tls, &[uid])
            }
            // GetLastOpenedUser -> AccountUid.
            Some(4) => {
                let uid = self.current_user().uid;
                self.write_ipc_response(tls, 0, &[], &uid, &[])
            }
            // GetProfile(AccountUid) -> IProfile.
            Some(5) => self.acc_open_profile(tls, handle, "acc:profile"),
            // IsUserRegistrationRequestPermitted -> false: there is no account applet.
            Some(50) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
            // TrySelectUserWithoutInteraction(bool) -> AccountUid: the user playing.
            Some(51) | Some(52) => {
                let uid = self.current_user().uid;
                self.write_ipc_response(tls, 0, &[], &uid, &[])
            }
            // DebugActivateOpenContextRetention: retention is unconditional.
            Some(99) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // InitializeApplicationInfo: 100, 140 (6.0.0+) and 160 are the same call.
            Some(100) | Some(140) | Some(160) if application => {
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // GetBaasAccountManagerForApplication(AccountUid) -> IManagerForApplication.
            Some(101) if application => {
                self.reply_with_interface(tls, handle, "acc:manager")?;
                Ok(())
            }
            // AuthenticateApplicationAsync / CheckNetworkServiceAvailabilityAsync -> IAsyncContext.
            Some(102) | Some(103) if application => {
                self.reply_with_interface(tls, handle, "acc:async-context")?;
                Ok(())
            }
            // `acc:u1`/`acc:su` from here down. The notifier getters -> INotifier.
            Some(100) | Some(101) | Some(103) | Some(104) | Some(106) => {
                self.reply_with_interface(tls, handle, "acc:notifier")?;
                Ok(())
            }
            // GetBaasAccountManagerForSystemService(AccountUid) -> IManagerForSystemService.
            Some(102) => {
                self.reply_with_interface(tls, handle, "acc:manager")?;
                Ok(())
            }
            // CheckNetworkServiceAvailabilityAsync -> IAsyncContext.
            Some(105) => {
                self.reply_with_interface(tls, handle, "acc:async-context")?;
                Ok(())
            }
            // StoreSaveDataThumbnail / ClearSaveDataThumbnail / LoadSaveDataThumbnail
            // (AccountUid, u64 application id): the image is kept per user and title.
            Some(cmd @ 110..=112) => {
                let at = self.ipc_request_data(tls);
                let uid: [u8; 16] = self.read_bytes(at, 16).try_into().unwrap_or_default();
                let key = (uid, self.mem.read_u64(at.wrapping_add(16))?);
                match cmd {
                    110 => {
                        let image = match self.ipc_send_buffer(tls, 0) {
                            Some((addr, size)) if addr != 0 => self.read_bytes(addr, size),
                            _ => Vec::new(),
                        };
                        self.save_thumbnails.insert(key, image);
                        self.write_ipc_response(tls, 0, &[], &[], &[])
                    }
                    111 => {
                        self.save_thumbnails.remove(&key);
                        self.write_ipc_response(tls, 0, &[], &[], &[])
                    }
                    _ => {
                        let image = self.save_thumbnails.get(&key).cloned().unwrap_or_default();
                        let (addr, room) = self.ipc_output_buffer(tls, 0).unwrap_or((0, 0));
                        let n = image.len().min(room as usize);
                        if n != 0 {
                            self.mem.write_bytes(addr, &image[..n])?;
                        }
                        self.write_ipc_response(tls, 0, &[], &(n as u32).to_le_bytes(), &[])
                    }
                }
            }
            // IsUserAccountSwitchLocked -> true: users are switched on the page.
            Some(150) => self.write_ipc_response(tls, 0, &[], &[1u8], &[]),
            // GetProfileEditor(AccountUid) -> IProfileEditor.
            Some(205) if iface == "acc:su" => {
                self.acc_open_profile(tls, handle, "acc:profile-editor")
            }
            // GetBaasAccountAdministrator(AccountUid) -> IAdministrator.
            Some(250) if iface == "acc:su" => {
                self.reply_with_interface(tls, handle, "acc:administrator")?;
                Ok(())
            }
            _ => self.unimplemented_command(tls, iface, cmd_id),
        }
    }

    /// `IProfile` and `IProfileEditor` (the same plus the store commands).
    fn acc_profile_request(
        &mut self,
        tls: u32,
        handle: u64,
        iface: &str,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        // An object this service did not hand out is taken to be the playing user's.
        let key = self.ipc_object_key(tls, handle);
        let uid = self
            .process()
            .acc_profiles
            .get(&key)
            .copied()
            .unwrap_or(self.current_user().uid);
        let Some(index) = self.users.iter().position(|user| user.uid == uid) else {
            return self.write_ipc_response(tls, ACCOUNT_USER_NOT_EXIST, &[], &[], &[]);
        };
        match cmd_id {
            // Get -> ProfileBase, plus AccountUserData zeroed in the caller's buffer.
            Some(0) => {
                if let Some((addr, size)) = self.ipc_output_buffer(tls, 0) {
                    if addr != 0 {
                        for i in 0..(size as usize).min(ACCOUNT_USER_DATA_LEN) as u32 {
                            self.mem.write_u8(addr.wrapping_add(i), 0)?;
                        }
                    }
                }
                let base = profile_base(&self.users[index]);
                self.write_ipc_response(tls, 0, &[], &base, &[])
            }
            // GetBase -> ProfileBase.
            Some(1) => {
                let base = profile_base(&self.users[index]);
                self.write_ipc_response(tls, 0, &[], &base, &[])
            }
            // GetImageSize / GetLargeImageSize [18.0.0+] -> u32, the one picture's size.
            Some(10) | Some(20) => {
                let size = self.users[index].image().len() as u32;
                self.write_ipc_response(tls, 0, &[], &size.to_le_bytes(), &[])
            }
            // LoadImage / LoadLargeImage [18.0.0+] (out buffer) -> u32 bytes written.
            Some(11) | Some(21) => {
                let image = self.users[index].image();
                let mut written = 0u32;
                if let Some((addr, size)) = self.ipc_output_buffer(tls, 0) {
                    if addr != 0 {
                        let len = image.len().min(size as usize);
                        for (i, &byte) in image[..len].iter().enumerate() {
                            self.mem.write_u8(addr.wrapping_add(i as u32), byte)?;
                        }
                        written = len as u32;
                    }
                }
                self.write_ipc_response(tls, 0, &[], &written.to_le_bytes(), &[])
            }
            // GetImageId [18.0.0+] -> Uuid; must be stable and nonzero.
            Some(30) => {
                let id = self.users[index].image_id();
                self.write_ipc_response(tls, 0, &[], &id, &[])
            }
            // Store, StoreWithImage, StoreWithLargeImage [18.0.0+]: persist the
            // nickname and picture, and tell the host.
            Some(100) | Some(101) | Some(110) if iface == "acc:profile-editor" => {
                let at = self.ipc_request_data(tls);
                let nickname = self.read_string(at.wrapping_add(0x18), NICKNAME_LEN as u32);
                let picture = match (cmd_id, self.ipc_send_buffer(tls, 0)) {
                    (Some(101) | Some(110), Some((addr, size))) if addr != 0 && size != 0 => {
                        Some(self.read_bytes(addr, size))
                    }
                    _ => None,
                };
                let edited_at = self.unix_time;
                let user = &mut self.users[index];
                user.nickname = fit_nickname(&nickname);
                user.edited_at = edited_at;
                if picture.is_some() {
                    user.picture = picture;
                }
                self.profiles_edited = true;
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            _ => self.unimplemented_command(tls, iface, cmd_id),
        }
    }

    /// `IManagerForApplication`/`IManagerForSystemService`, and what `IAdministrator` shares.
    fn acc_manager_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        match cmd_id {
            // CheckAvailability -> success, as `nifm` reports a link; tokens are empty.
            Some(0) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // GetAccountId -> u64 NetworkServiceAccountId.
            Some(1) => {
                let id = NETWORK_SERVICE_ACCOUNT_ID.to_le_bytes();
                self.write_ipc_response(tls, 0, &[], &id, &[])
            }
            // EnsureIdTokenCacheAsync -> IAsyncContext.
            Some(2) => {
                self.reply_with_interface(tls, handle, "acc:async-context")?;
                Ok(())
            }
            // LoadIdTokenCache(out buffer) -> u32 size: always empty.
            Some(3) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            // GetNetworkServiceLicenseCacheEx (15.0.0+) -> u32 license, s64 expiry, zeroed.
            Some(143) => self.write_ipc_response(tls, 0, &[], &[0u8; 0x10], &[]),
            // GetNintendoAccountUserResourceCache (130, and 136 from 19.0.0): no Nintendo Account
            // is linked here. Shop skips the whole block on this result, and parses a date out of
            // any success.
            Some(130 | 136) => {
                self.write_ipc_response(tls, ACCOUNT_REGISTRATION_REQUIRED, &[], &[], &[])
            }
            _ => self.unimplemented_command(tls, "acc:manager", cmd_id),
        }
    }

    /// `IAsyncContext`, already finished: event signalled, `HasDone` true, success.
    fn acc_async_context_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        match cmd_id {
            // GetSystemEvent.
            Some(0) => {
                let event = self.alloc_event("acc:async-context", false);
                self.signal_event(event);
                self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
            }
            // Cancel.
            Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // HasDone -> bool.
            Some(2) => self.write_ipc_response(tls, 0, &[], &[1u8], &[]),
            // GetResult -> Result.
            Some(3) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            _ => self.unimplemented_command(tls, "acc:async-context", cmd_id),
        }
    }

    fn acc_requested_uid(&self, tls: u32) -> [u8; 16] {
        let at = self.ipc_request_data(tls);
        let mut uid = [0u8; 16];
        for (index, byte) in uid.iter_mut().enumerate() {
            *byte = self.mem.read_u8(at.wrapping_add(index as u32)).unwrap_or(0);
        }
        uid
    }

    /// Hand out an `IProfile` or `IProfileEditor` for the requested user.
    fn acc_open_profile(&mut self, tls: u32, handle: u64, iface: &str) -> Result<()> {
        let uid = self.acc_requested_uid(tls);
        if self.user(uid).is_none() {
            return self.write_ipc_response(tls, ACCOUNT_USER_NOT_EXIST, &[], &[], &[]);
        }
        let key = self.reply_with_interface(tls, handle, iface)?;
        self.process_mut().acc_profiles.insert(key, uid);
        Ok(())
    }

    /// Write `uids` into a list command's output buffer and zero the rest.
    fn acc_write_user_list(&mut self, tls: u32, uids: &[[u8; 16]]) -> Result<()> {
        // Callers count users up to the first all-zero uid, so every slot is written.
        if let Some((addr, size)) = self.ipc_output_buffer(tls, 0) {
            if addr != 0 {
                for offset in 0..size {
                    self.mem.write_u8(addr.wrapping_add(offset), 0)?;
                }
                let fits = size as usize / 16;
                for (slot, uid) in uids.iter().take(fits).enumerate() {
                    self.mem
                        .write_bytes(addr.wrapping_add(slot as u32 * 16), uid)?;
                }
            }
        }
        self.write_ipc_response(tls, 0, &[], &[], &[])
    }

    fn user(&self, uid: [u8; 16]) -> Option<&UserAccount> {
        self.users.iter().find(|user| user.uid == uid)
    }

    pub fn users(&self) -> &[UserAccount] {
        &self.users
    }

    pub fn current_user(&self) -> &UserAccount {
        &self.users[self.current_user]
    }

    /// Replace the console's users and set the playing one. Also rewrites the
    /// preselected-user launch parameter, which may already be seeded.
    pub fn set_users(
        &mut self,
        users: Vec<UserAccount>,
        current: [u8; 16],
    ) -> std::result::Result<(), UsersRefused> {
        if users.is_empty() || users.len() > MAX_USERS {
            return Err(UsersRefused::Count);
        }
        if users.iter().any(|user| user.uid == [0; 16]) {
            return Err(UsersRefused::ZeroUid);
        }
        for (index, user) in users.iter().enumerate() {
            if users[..index].iter().any(|other| other.uid == user.uid) {
                return Err(UsersRefused::DuplicateUid);
            }
        }
        let Some(current) = users.iter().position(|user| user.uid == current) else {
            return Err(UsersRefused::UnknownCurrent);
        };
        self.users = users
            .into_iter()
            .map(|user| UserAccount {
                nickname: fit_nickname(&user.nickname),
                ..user
            })
            .collect();
        self.current_user = current;
        let preselected = super::am::LAUNCH_PARAMETER_PRESELECTED_USER;
        if self
            .process()
            .am_launch_parameters
            .contains_key(&preselected)
        {
            let uid = self.current_user().uid;
            self.process_mut()
                .am_launch_parameters
                .insert(preselected, super::am::preselected_user_parameter(uid));
        }
        Ok(())
    }

    /// Whether the guest stored a profile through `IProfileEditor` since the last call.
    pub fn take_profile_edits(&mut self) -> bool {
        std::mem::take(&mut self.profiles_edited)
    }

    pub fn set_user_nickname(&mut self, nickname: &str) {
        let current = self.current_user;
        self.users[current].nickname = fit_nickname(nickname);
    }

    pub fn user_nickname(&self) -> &str {
        &self.current_user().nickname
    }
}

/// `nn::account::ProfileBase`: uid, last edit, and a NUL-padded 0x20-byte nickname.
fn profile_base(user: &UserAccount) -> [u8; PROFILE_BASE_LEN] {
    let mut base = [0u8; PROFILE_BASE_LEN];
    base[..0x10].copy_from_slice(&user.uid);
    base[0x10..0x18].copy_from_slice(&user.edited_at.to_le_bytes());
    let nickname = user.nickname.as_bytes();
    let len = nickname.len().min(NICKNAME_LEN - 1);
    base[0x18..0x18 + len].copy_from_slice(&nickname[..len]);
    base
}

#[cfg(test)]
mod tests {
    use crate::kernel::ipc::testing::*;

    #[test]
    fn a_generated_uid_is_never_nobody_and_never_repeats() {
        let uids: Vec<_> = (0..64).map(|_| super::random_uid()).collect();
        assert!(uids.iter().all(|uid| *uid != [0; 16]));
        for (index, uid) in uids.iter().enumerate() {
            assert!(!uids[..index].contains(uid), "{uid:?} was made twice");
        }
        // Both halves vary, so it is 128 bits and not 64 repeated.
        assert!(uids.iter().any(|uid| uid[..8] != uid[8..]));
    }

    #[test]
    fn every_console_in_a_process_starts_with_the_same_generated_user() {
        let (first, second) = (Cpu::new(), Cpu::new());
        assert_eq!(first.current_user().uid, second.current_user().uid);
        assert_eq!(first.current_user().uid, super::default_uid());
        assert_ne!(super::default_uid(), [0; 16]);
        assert_ne!(
            super::default_uid(),
            super::random_uid(),
            "a new user is not the default"
        );
    }

    #[test]
    fn a_profile_image_id_changes_with_the_picture_and_only_then() {
        let user =
            |picture: Option<Vec<u8>>| super::UserAccount::new(super::default_uid(), "A", picture);
        let plain = user(None).image_id();
        assert_ne!(plain, user(Some(vec![1, 2, 3])).image_id());
        assert_eq!(
            plain,
            user(None).image_id(),
            "the same picture, the same id"
        );
    }

    type HuffmanTable = (u8, Vec<(u8, u16, u8)>);
    use crate::cpu::Cpu;

    fn acc(cpu: &mut Cpu, service: &str, command_id: u32) {
        cpu.register_service_handle(9, service);
        cpu.acc_request(TLS, 9, Some(command_id)).unwrap();
    }

    #[test]
    fn acc_reports_one_user_who_is_signed_in() {
        let mut cpu = request(false, 0, &[]);
        acc(&mut cpu, "acc:u0", 0);
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 1);

        // GetLastOpenedUser: the uid, never zero.
        let mut cpu = request(false, 4, &[]);
        acc(&mut cpu, "acc:u0", 4);
        let uid = cpu.read_bytes(TLS + 0x20, 16);
        assert_eq!(uid, super::default_uid().to_vec());

        // TrySelectUserWithoutInteraction (51, and 52 from 19.0.0) hands back the same one.
        for command in [51, 52] {
            let mut cpu = request(false, command, &[0, 0, 0, 0]);
            acc(&mut cpu, "acc:su", command);
            assert_eq!(
                cpu.read_bytes(TLS + 0x20, 16),
                super::default_uid().to_vec()
            );
        }
    }

    #[test]
    fn acc_list_all_users_zeroes_the_slots_it_has_no_user_for() {
        // Every slot must be written; the caller scans for the first zero uid.
        const BUFFER: u32 = 0x4000;
        let mut cpu = request_with_recv_buffer(2, &[], BUFFER, 0x40);
        cpu.mem.map_zero(BUFFER, 0x100).unwrap();
        for offset in 0..0x40 {
            cpu.mem.write_u8(BUFFER + offset, 0xAA).unwrap();
        }
        acc(&mut cpu, "acc:u0", 2);

        assert_eq!(cpu.read_bytes(BUFFER, 16), super::default_uid().to_vec());
        assert_eq!(
            cpu.read_bytes(BUFFER + 16, 0x30),
            vec![0u8; 0x30],
            "stale uids left behind"
        );
    }

    #[test]
    fn save_data_thumbnails_are_kept_per_user_and_title() {
        const SEND: u32 = 0x4000;
        const RECV: u32 = 0x5000;
        const TITLE: u64 = 0x0100_7EF0_0011_E000;
        let mut payload = super::default_uid().to_vec();
        payload.extend_from_slice(&TITLE.to_le_bytes());
        let mut cpu = request(false, 110, &payload);
        cpu.mem.map_zero(SEND, 0x2000).unwrap();
        cpu.mem.write_bytes(SEND, &[1, 2, 3, 4]).unwrap();

        let load = |cpu: &mut Cpu, payload: &[u8]| {
            write_map_buffer_request(cpu, 112, payload, RECV, 8, false);
            acc(cpu, "acc:su", 112);
            (
                cpu.mem.read_u32(TLS + 0x18).unwrap(),
                cpu.mem.read_u32(TLS + 0x20).unwrap(),
            )
        };

        write_map_buffer_request(&mut cpu, 110, &payload, SEND, 4, true);
        acc(&mut cpu, "acc:su", 110);
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "store");

        assert_eq!(load(&mut cpu, &payload), (0, 4));
        assert_eq!(cpu.read_bytes(RECV, 4), [1, 2, 3, 4]);

        let mut other = payload.clone();
        other[16] ^= 1;
        assert_eq!(load(&mut cpu, &other), (0, 0), "another title");

        write_request(&mut cpu, 111, &payload);
        acc(&mut cpu, "acc:su", 111);
        assert_eq!(load(&mut cpu, &payload), (0, 0), "cleared");
    }

    #[test]
    fn the_administrator_is_registered_and_not_linked() {
        let mut cpu = request(false, 250, &super::default_uid());
        acc(&mut cpu, "acc:su", 250);
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0);
        let administrator = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
        assert_eq!(cpu.service_name(administrator), Some("acc:administrator"));

        for (command, answer) in [(200, 1), (250, 0)] {
            write_request(&mut cpu, command, &[]);
            cpu.acc_request(TLS, administrator, Some(command)).unwrap();
            assert_eq!(
                cpu.mem.read_u8(TLS + 0x20).unwrap(),
                answer,
                "command {command}"
            );
        }
    }

    #[test]
    fn acc_knows_only_its_own_uid() {
        let mut cpu = request(false, 1, &super::default_uid());
        acc(&mut cpu, "acc:u0", 1);
        assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 1);

        let mut cpu = request(false, 1, &[0xAB; 16]);
        acc(&mut cpu, "acc:u0", 1);
        assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 0);

        // GetProfile for an invented uid fails.
        let mut cpu = request(false, 5, &[0xAB; 16]);
        acc(&mut cpu, "acc:u0", 5);
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x18).unwrap(),
            super::ACCOUNT_USER_NOT_EXIST
        );
    }

    #[test]
    fn acc_profile_get_writes_the_userdata_into_its_pointer_buffer() {
        const BUFFER: u32 = 0x4000;
        let mut cpu = request_with_recv_static(0, &[], BUFFER, 0x80);
        cpu.mem.map_zero(BUFFER, 0x100).unwrap();
        // Stale stack contents the reply must overwrite.
        for offset in 0..0x80 {
            cpu.mem.write_u8(BUFFER + offset, 0xAA).unwrap();
        }
        assert_eq!(cpu.ipc_recv_static_buffers(TLS), vec![(BUFFER, 0x80)]);

        cpu.register_service_handle(9, "acc:profile");
        cpu.acc_request(TLS, 9, Some(0)).unwrap();

        assert_eq!(
            cpu.read_bytes(BUFFER, 0x80),
            vec![0u8; 0x80],
            "userdata zeroed, not left as stack garbage"
        );
        assert_eq!(
            cpu.read_bytes(TLS + 0x20, 16),
            super::default_uid().to_vec()
        );
        assert_eq!(cpu.mem.read_u64(TLS + 0x30).unwrap(), 0);
        assert_eq!(cpu.read_string(TLS + 0x38, 0x20), "Player");
    }

    #[test]
    fn acc_profile_editor_stores_a_nickname_that_reads_back() {
        let mut store = [0u8; super::PROFILE_BASE_LEN];
        store[..16].copy_from_slice(&super::default_uid());
        store[0x18..0x18 + 5].copy_from_slice(b"Yuuto");
        let mut cpu = request(false, 100, &store);
        cpu.set_unix_time(1_700_000_000);
        cpu.register_service_handle(9, "acc:profile-editor");
        cpu.acc_request(TLS, 9, Some(100)).unwrap();
        assert_eq!(cpu.user_nickname(), "Yuuto");

        // GetBase reports what was stored, timestamp included.
        write_request(&mut cpu, 1, &[]);
        cpu.register_service_handle(9, "acc:profile");
        cpu.acc_request(TLS, 9, Some(1)).unwrap();
        assert_eq!(cpu.read_string(TLS + 0x38, 0x20), "Yuuto");
        assert_eq!(cpu.mem.read_u64(TLS + 0x30).unwrap(), 1_700_000_000);
    }

    #[test]
    fn acc_initialize_application_info_answers_every_id_it_has_had() {
        // InitializeApplicationInfo under all three command ids.
        for command in [100u32, 140, 160] {
            let mut cpu = request(false, command, &0u64.to_le_bytes());
            cpu.register_service_handle(9, "acc:u0");
            cpu.acc_request(TLS, 9, Some(command)).unwrap();
            assert_eq!(
                cpu.mem.read_u32(TLS + 0x18).unwrap(),
                0,
                "command {command}"
            );
        }

        // 141 is `ListQualifiedUsers`.
        const BUFFER: u32 = 0x4000;
        let mut cpu = request_with_recv_buffer(141, &[], BUFFER, 0x40);
        cpu.mem.map_zero(BUFFER, 0x100).unwrap();
        cpu.register_service_handle(9, "acc:u0");
        cpu.acc_request(TLS, 9, Some(141)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0);
        assert_eq!(cpu.read_bytes(BUFFER, 16), super::default_uid().to_vec());
        assert_eq!(
            cpu.read_bytes(BUFFER + 16, 0x30),
            vec![0u8; 0x30],
            "one user listed"
        );
    }

    #[test]
    fn acc_the_same_command_id_means_different_things_on_u0_and_u1() {
        // 101 differs per service; tell them apart by what the returned session answers.
        for (service, iface) in [("acc:u0", "acc:manager"), ("acc:u1", "acc:notifier")] {
            let mut cpu = request(false, 101, &super::default_uid());
            acc(&mut cpu, service, 101);
            let session = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;
            assert_eq!(cpu.service_name(session), Some(iface), "{service} cmd 101");
        }
    }

    #[test]
    fn acc_async_contexts_report_work_that_is_already_finished() {
        // CheckNetworkServiceAvailabilityAsync, then HasDone on the context.
        let mut cpu = request(false, 103, &[]);
        acc(&mut cpu, "acc:u0", 103);
        let session = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;
        assert_eq!(cpu.service_name(session), Some("acc:async-context"));

        let mut cpu = request(false, 2, &[]);
        cpu.register_service_handle(session, "acc:async-context");
        cpu.acc_request(TLS, session, Some(2)).unwrap();
        assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 1);
    }

    #[test]
    fn acc_load_image_writes_exactly_the_size_it_advertised() {
        const BUFFER: u32 = 0x4000;
        let mut cpu = request(false, 10, &[]);
        cpu.register_service_handle(9, "acc:profile");
        cpu.acc_request(TLS, 9, Some(10)).unwrap();
        let advertised = cpu.mem.read_u32(TLS + 0x20).unwrap();
        assert!(advertised > 0, "an icon of no bytes is nothing to decode");

        let mut cpu = request_with_recv_buffer(11, &[], BUFFER, advertised);
        cpu.mem
            .map_zero(BUFFER, advertised as usize + 0x100)
            .unwrap();
        cpu.register_service_handle(9, "acc:profile");
        cpu.acc_request(TLS, 9, Some(11)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), advertised);
        assert_eq!(
            cpu.read_bytes(BUFFER, advertised),
            super::profile_image(super::default_uid())
        );

        // GetLargeImageSize and LoadLargeImage [18.0.0+] answer for the same icon.
        let mut cpu = request(false, 20, &[]);
        cpu.register_service_handle(9, "acc:profile");
        cpu.acc_request(TLS, 9, Some(20)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), advertised);

        let mut cpu = request_with_recv_buffer(21, &[], BUFFER, advertised);
        cpu.mem
            .map_zero(BUFFER, advertised as usize + 0x100)
            .unwrap();
        cpu.register_service_handle(9, "acc:profile");
        cpu.acc_request(TLS, 9, Some(21)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), advertised);
        assert_eq!(
            cpu.read_bytes(BUFFER, advertised),
            super::profile_image(super::default_uid())
        );
    }

    #[test]
    fn acc_names_the_icon_with_an_id_that_is_not_the_uid() {
        // GetImageId [18.0.0+]: stable across calls and nonzero.
        let mut ids = Vec::new();
        for _ in 0..2 {
            let mut cpu = request(false, 30, &[]);
            cpu.register_service_handle(9, "acc:profile");
            cpu.acc_request(TLS, 9, Some(30)).unwrap();
            assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "refused");
            ids.push(cpu.read_bytes(TLS + 0x20, 0x10));
        }
        assert_eq!(ids[0], ids[1], "a cache key that changes never hits");
        assert_ne!(ids[0], vec![0u8; 0x10], "zero means there is no icon");
        assert_ne!(
            ids[0],
            super::default_uid().to_vec(),
            "the icon, not the user"
        );
    }

    const ANN: [u8; 16] = *b"ann-uid-00000001";
    const BEN: [u8; 16] = *b"ben-uid-00000002";

    fn two_users(cpu: &mut Cpu) {
        let ann = super::UserAccount::new(ANN, "Ann", Some(vec![0xFF, 0xD8, 0x01, 0xFF, 0xD9]));
        let ben = super::UserAccount::new(BEN, "Ben", None);
        cpu.set_users(vec![ann, ben], BEN).unwrap();
    }

    /// Open `iface` for `uid` through `command` on `acc:su`, returning its handle.
    fn open_profile(uid: [u8; 16], command: u32) -> (Cpu, u64) {
        let mut cpu = request(false, command, &uid);
        two_users(&mut cpu);
        acc(&mut cpu, "acc:su", command);
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x18).unwrap(),
            0,
            "opening the profile"
        );
        let object = u64::from(cpu.mem.read_u32(TLS + 0x0C).unwrap());
        (cpu, object)
    }

    #[test]
    fn acc_lists_every_user_and_answers_with_the_one_playing() {
        let mut cpu = request(false, 0, &[]);
        two_users(&mut cpu);
        acc(&mut cpu, "acc:u0", 0);
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 2, "GetUserCount");

        const BUFFER: u32 = 0x4000;
        let mut cpu = request_with_recv_buffer(2, &[], BUFFER, 0x80);
        cpu.mem.map_zero(BUFFER, 0x100).unwrap();
        two_users(&mut cpu);
        acc(&mut cpu, "acc:u0", 2);
        assert_eq!(cpu.read_bytes(BUFFER, 16), ANN.to_vec());
        assert_eq!(cpu.read_bytes(BUFFER + 16, 16), BEN.to_vec());
        assert_eq!(cpu.read_bytes(BUFFER + 32, 0x60), vec![0u8; 0x60]);

        let mut cpu = request_with_recv_buffer(3, &[], BUFFER, 0x80);
        cpu.mem.map_zero(BUFFER, 0x100).unwrap();
        two_users(&mut cpu);
        acc(&mut cpu, "acc:u0", 3);
        assert_eq!(cpu.read_bytes(BUFFER, 16), BEN.to_vec());
        assert_eq!(cpu.read_bytes(BUFFER + 16, 16), vec![0u8; 16]);

        // GetLastOpenedUser and TrySelectUserWithoutInteraction: the one playing.
        for command in [4u32, 51] {
            let mut cpu = request(false, command, &[0; 4]);
            two_users(&mut cpu);
            acc(&mut cpu, "acc:u0", command);
            assert_eq!(
                cpu.read_bytes(TLS + 0x20, 16),
                BEN.to_vec(),
                "command {command}"
            );
        }

        for (uid, exists) in [(ANN, 1u8), (BEN, 1), ([0xAB; 16], 0)] {
            let mut cpu = request(false, 1, &uid);
            two_users(&mut cpu);
            acc(&mut cpu, "acc:u0", 1);
            assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), exists);
        }
    }

    #[test]
    fn a_profile_is_the_profile_of_the_user_it_was_opened_for() {
        let (mut cpu, profile) = open_profile(ANN, 5);
        write_request(&mut cpu, 1, &[]);
        cpu.acc_request(TLS, profile, Some(1)).unwrap();
        assert_eq!(cpu.read_bytes(TLS + 0x20, 16), ANN.to_vec());
        assert_eq!(cpu.read_string(TLS + 0x38, 0x20), "Ann");

        write_request(&mut cpu, 10, &[]);
        cpu.acc_request(TLS, profile, Some(10)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x20).unwrap(),
            5,
            "her own picture's size"
        );

        write_request(&mut cpu, 30, &[]);
        cpu.acc_request(TLS, profile, Some(30)).unwrap();
        let ann_image = cpu.read_bytes(TLS + 0x20, 16);

        // Ben has no picture, so one is made in his colour.
        let (mut cpu, profile) = open_profile(BEN, 5);
        write_request(&mut cpu, 30, &[]);
        cpu.acc_request(TLS, profile, Some(30)).unwrap();
        assert_ne!(
            cpu.read_bytes(TLS + 0x20, 16),
            ann_image,
            "one cache key for two pictures"
        );
        assert_ne!(
            super::profile_image(BEN),
            super::profile_image(ANN),
            "two users drawn alike"
        );

        let mut cpu = request(false, 5, &[0xAB; 16]);
        two_users(&mut cpu);
        acc(&mut cpu, "acc:u0", 5);
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x18).unwrap(),
            super::ACCOUNT_USER_NOT_EXIST
        );
    }

    #[test]
    fn an_edit_lands_on_the_user_it_was_opened_for_and_is_reported() {
        let (mut cpu, editor) = open_profile(ANN, 205);
        assert!(!cpu.take_profile_edits());
        let mut store = [0u8; super::PROFILE_BASE_LEN];
        store[..16].copy_from_slice(&ANN);
        store[0x18..0x18 + 4].copy_from_slice(b"Anna");
        write_request(&mut cpu, 100, &store);
        cpu.acc_request(TLS, editor, Some(100)).unwrap();
        assert_eq!(cpu.users()[0].nickname, "Anna");
        assert_eq!(
            cpu.users()[1].nickname,
            "Ben",
            "the one playing is untouched"
        );
        assert!(cpu.take_profile_edits());
        assert!(!cpu.take_profile_edits(), "taken, not read");
    }

    #[test]
    fn a_user_list_that_cannot_be_a_console_is_refused() {
        use super::{UserAccount, UsersRefused, MAX_USERS};
        let mut cpu = Cpu::new();
        let user = |uid: [u8; 16]| UserAccount::new(uid, "x", None);
        assert_eq!(cpu.set_users(vec![], ANN), Err(UsersRefused::Count));
        let crowd = (0..=MAX_USERS as u8).map(|n| user([n + 1; 16])).collect();
        assert_eq!(cpu.set_users(crowd, [1; 16]), Err(UsersRefused::Count));
        assert_eq!(
            cpu.set_users(vec![user([0; 16])], [0; 16]),
            Err(UsersRefused::ZeroUid)
        );
        assert_eq!(
            cpu.set_users(vec![user(ANN), user(ANN)], ANN),
            Err(UsersRefused::DuplicateUid)
        );
        assert_eq!(
            cpu.set_users(vec![user(ANN)], BEN),
            Err(UsersRefused::UnknownCurrent)
        );
        assert_eq!(
            cpu.user_nickname(),
            "Player",
            "a refused list changes nothing"
        );

        let long = "é".repeat(20);
        cpu.set_users(vec![UserAccount::new(ANN, &long, None)], ANN)
            .unwrap();
        assert_eq!(cpu.user_nickname(), "é".repeat(15));
    }

    /// Decode the whole profile icon with tables rebuilt from its DHT segments:
    /// every block must be the same colour and the scan must end at EOI.
    #[test]
    fn the_profile_icon_is_a_jpeg_that_decodes_to_one_colour() {
        let jpeg = super::profile_image(super::default_uid());
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "SOI");
        assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9], "EOI");

        let mut quant = [0u8; 64];
        let mut tables: Vec<HuffmanTable> = Vec::new();
        let (mut width, mut height) = (0u32, 0u32);
        let mut components = 0usize;
        let mut scan_start = 0usize;
        let mut at = 2usize;
        while at + 4 <= jpeg.len() {
            assert_eq!(jpeg[at], 0xFF, "a segment starts with a marker");
            let marker = jpeg[at + 1];
            let length = u16::from_be_bytes([jpeg[at + 2], jpeg[at + 3]]) as usize;
            let payload = &jpeg[at + 4..at + 2 + length];
            match marker {
                super::JPEG_DQT => {
                    assert_eq!(payload[0], 0, "8-bit precision, table 0");
                    quant.copy_from_slice(&payload[1..65]);
                }
                super::JPEG_SOF0 => {
                    assert_eq!(payload[0], 8, "8-bit samples");
                    height = u32::from(u16::from_be_bytes([payload[1], payload[2]]));
                    width = u32::from(u16::from_be_bytes([payload[3], payload[4]]));
                    components = payload[5] as usize;
                    for index in 0..components {
                        // 1x1 sampling: no subsampling to undo.
                        assert_eq!(payload[7 + index * 3], 0x11);
                    }
                }
                super::JPEG_DHT => {
                    let bits: [u8; 16] = payload[1..17].try_into().unwrap();
                    let count: usize = bits.iter().map(|&b| b as usize).sum();
                    let mut codes = Vec::new();
                    let (mut code, mut next) = (0u16, 0usize);
                    for (index, &in_this_length) in bits.iter().enumerate() {
                        for _ in 0..in_this_length {
                            codes.push((payload[17 + next], code, index as u8 + 1));
                            code += 1;
                            next += 1;
                        }
                        code <<= 1;
                    }
                    assert_eq!(next, count);
                    tables.push((payload[0], codes));
                }
                super::JPEG_SOS => {
                    for index in 0..components {
                        assert_eq!(payload[2 + index * 2], 0x00, "both tables are id 0");
                    }
                    scan_start = at + 2 + length;
                    break;
                }
                _ => {}
            }
            at += 2 + length;
        }
        assert_eq!((width, height), (256, 256));
        assert_eq!(components, 3);

        // The entropy-coded segment, up to the EOI. 0xFF00 is a stuffed 0xFF.
        let mut scan = Vec::new();
        let mut at = scan_start;
        while at < jpeg.len() {
            if jpeg[at] == 0xFF {
                match jpeg[at + 1] {
                    0x00 => {
                        scan.push(0xFFu8);
                        at += 2;
                        continue;
                    }
                    super::JPEG_EOI => break,
                    other => panic!("unexpected marker {other:#x} inside the scan"),
                }
            }
            scan.push(jpeg[at]);
            at += 1;
        }
        assert_eq!(at, jpeg.len() - 2, "the scan runs right up to the EOI");

        /// A cursor over the scan's bits, MSB first.
        struct Reader<'a> {
            data: &'a [u8],
            bit: usize,
        }
        impl Reader<'_> {
            fn bit(&mut self) -> u32 {
                let value = u32::from(self.data[self.bit / 8] >> (7 - self.bit % 8)) & 1;
                self.bit += 1;
                value
            }

            fn symbol(&mut self, codes: &[(u8, u16, u8)]) -> u8 {
                let (mut code, mut length) = (0u16, 0u8);
                for _ in 0..16 {
                    code = (code << 1) | self.bit() as u16;
                    length += 1;
                    let found = codes.iter().find(|&&(_, candidate, candidate_length)| {
                        candidate == code && candidate_length == length
                    });
                    if let Some(&(symbol, _, _)) = found {
                        return symbol;
                    }
                }
                panic!("no Huffman code matched");
            }
        }
        let dc_table = &tables.iter().find(|(id, _)| *id == 0x00).unwrap().1;
        let ac_table = &tables.iter().find(|(id, _)| *id == 0x10).unwrap().1;
        let mut reader = Reader {
            data: &scan,
            bit: 0,
        };

        // Every block in MCU order: a DC difference then EOB.
        let blocks = width.div_ceil(8) * height.div_ceil(8);
        let mut predictor = [0i32; 3];
        for mcu in 0..blocks {
            for (component, pred) in predictor.iter_mut().enumerate() {
                let category = reader.symbol(dc_table);
                let mut diff = 0i32;
                if category > 0 {
                    let mut value = 0i32;
                    for _ in 0..category {
                        value = (value << 1) | reader.bit() as i32;
                    }
                    // A leading zero bit means a negative one's-complement value.
                    diff = if value >= 1 << (category - 1) {
                        value
                    } else {
                        value - (1 << category) + 1
                    };
                }
                *pred += diff;
                assert_eq!(
                    reader.symbol(ac_table),
                    super::JPEG_EOB,
                    "AC of a flat block"
                );

                // Dequantize and undo the level shift; a lone DC's IDCT is DC/8.
                let value = *pred * i32::from(quant[0]) / 8 + 128;
                let (red, green, blue) = super::picture_color(super::default_uid());
                let (red, green, blue) = (f32::from(red), f32::from(green), f32::from(blue));
                let expected = match component {
                    0 => 0.299 * red + 0.587 * green + 0.114 * blue,
                    1 => -0.168_736 * red - 0.331_264 * green + 0.5 * blue + 128.0,
                    _ => 0.5 * red - 0.418_688 * green - 0.081_312 * blue + 128.0,
                };
                assert_eq!(
                    value,
                    expected.round() as i32,
                    "mcu {mcu} component {component}"
                );
            }
        }
        assert!(
            scan.len() * 8 - reader.bit < 8,
            "the scan decodes to exactly the blocks the frame declares"
        );
    }
}
