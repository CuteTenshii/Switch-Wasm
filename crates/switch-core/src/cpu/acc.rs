//! `acc`: the console's one user account, and the profile picture it is
//! drawn with.
//!
//! There is exactly one user here ([`ACCOUNT_UID`]) and it is always signed
//! in. The icon is synthesized rather than stored — `IProfile::LoadImage`
//! hands out an encoded JPEG and callers feed what they get straight to a
//! decoder, so there has to be a real one to decode.

use super::Cpu;
use crate::Result;

pub(super) const ACCOUNT_UID: [u8; 16] = *b"switch-wasm user";

/// The `Uuid` `IProfile::GetImageId` reports for the icon below.
///
/// Callers cache the icon and re-read it only when this changes. The icon is
/// synthesized from a constant, so it never does.
const PROFILE_IMAGE_ID: [u8; 16] = *b"switch-wasm icon";

/// `nn::account::ProfileBase`: uid, last-edit timestamp, then the nickname.
const PROFILE_BASE_LEN: usize = 0x38;

/// `nn::account::UserData`, the block `IProfile::Get` fills in beside the base
/// (icon id, background colour, mii id).
const ACCOUNT_USER_DATA_LEN: usize = 0x80;

/// acc's "that user does not exist" (module 124, description 100).
///
/// Only a caller that invented a uid can reach this: the only uid this service
/// ever hands out is [`ACCOUNT_UID`], so anything else was not obtained from
/// here.
const ACCOUNT_USER_NOT_EXIST: u32 = 124 | (100 << 9);

/// The `NetworkServiceAccountId` `IManagerForApplication::GetAccountId`
/// reports. Nonzero, since zero is that field's "no account" sentinel; the
/// value itself is arbitrary and nothing derives anything from it.
const NETWORK_SERVICE_ACCOUNT_ID: u64 = 0x0000_0001_0000_0001;

/// `nn::account::Nickname`: a fixed NUL-terminated field inside `ProfileBase`.
pub(super) const NICKNAME_LEN: usize = 0x20;

/// The nickname the console's user has until a host or the guest changes it.
pub(super) const DEFAULT_NICKNAME: &str = "Player";

/// Real profile icons are 256x256.
const PROFILE_IMAGE_SIZE: u16 = 256;

/// The icon's colour, a neutral slate. Nothing derives anything from it.
const PROFILE_IMAGE_COLOR: (u8, u8, u8) = (0x4B, 0x50, 0x5A);

/// JPEG markers, for the profile icon [`solid_jpeg`] encodes.
const JPEG_SOI: u8 = 0xD8;

const JPEG_APP0: u8 = 0xE0;

const JPEG_DQT: u8 = 0xDB;

const JPEG_SOF0: u8 = 0xC0;

const JPEG_DHT: u8 = 0xC4;

const JPEG_SOS: u8 = 0xDA;

const JPEG_EOI: u8 = 0xD9;

/// Every entry of the quantization table. 8 is what makes a constant block's
/// DC coefficient (`8x`) quantize to exactly `x`.
const JPEG_QUANT: u8 = 8;

/// The AC symbol for "end of block": the rest of this block is zeros.
const JPEG_EOB: u8 = 0x00;

/// The DC Huffman table: the twelve magnitude categories, four coded in three
/// bits and eight in four, which is a complete code.
const JPEG_DC_BITS: [u8; 16] = [0, 0, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

const JPEG_DC_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// The AC Huffman table: end-of-block and run-of-sixteen-zeros, one bit each.
/// A constant image only ever emits the first, but a two-symbol table is a
/// complete code where a one-symbol table would not be.
const JPEG_AC_BITS: [u8; 16] = [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

const JPEG_AC_VALUES: [u8; 2] = [JPEG_EOB, 0xF0];

/// The user's profile picture, as the JPEG `IProfile::LoadImage` is defined to
/// return.
///
/// `acc` hands the icon out as an encoded JPEG and callers feed what they get
/// straight to a decoder, so answering with zero bytes leaves anything that
/// draws the user's picture with nothing to decode. There is no icon on this
/// console to hand over, so one is made: a plain field of colour, which is
/// what an account with no picture set should look like.
fn profile_image() -> Vec<u8> {
    solid_jpeg(PROFILE_IMAGE_SIZE, PROFILE_IMAGE_COLOR)
}

/// A baseline JPEG of a single solid colour, `size` x `size` pixels.
///
/// Encoding a constant image needs no DCT and no zig-zag: the transform of a
/// block of constant level-shifted value `x` is one DC coefficient of `8x`
/// with every AC coefficient zero. So each block is a Huffman-coded DC
/// *difference* — nonzero only in the first block of each component, since the
/// predictor is the previous block's DC and every block is the same — followed
/// by end-of-block. With a quantization table of 8 throughout, `8x`
/// quantizes to exactly `x` and dequantizes back to `8x`, so the colour
/// survives the round trip unchanged.
///
/// The Huffman tables are minimal rather than Annex K's: an encoder that emits
/// only DC categories and EOB needs no other symbols, and the tables travel in
/// the file anyway. Both are complete codes (their Kraft sums are 1), which is
/// what a decoder building a derived table expects.
fn solid_jpeg(size: u16, rgb: (u8, u8, u8)) -> Vec<u8> {
    let (red, green, blue) = (f32::from(rgb.0), f32::from(rgb.1), f32::from(rgb.2));
    let round = |value: f32| value.round().clamp(0.0, 255.0) as i32;
    // JFIF's RGB -> YCbCr (BT.601), the colour space a baseline JPEG's three
    // components are in.
    let components = [
        round(0.299 * red + 0.587 * green + 0.114 * blue),
        round(-0.168_736 * red - 0.331_264 * green + 0.5 * blue + 128.0),
        round(0.5 * red - 0.418_688 * green - 0.081_312 * blue + 128.0),
    ];

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(&[0xFF, JPEG_SOI]);
    // APP0/JFIF: version 1.1, no density units, no thumbnail.
    segment(
        &mut out,
        JPEG_APP0,
        b"JFIF\0\x01\x01\x00\x00\x01\x00\x01\x00\x00",
    );
    // One quantization table (id 0), 8-bit precision, used by all three
    // components.
    let mut quant = vec![0u8];
    quant.extend_from_slice(&[JPEG_QUANT; 64]);
    segment(&mut out, JPEG_DQT, &quant);
    // SOF0: 8-bit samples, `size` square, three components each sampled 1x1
    // (no chroma subsampling, so one block per component per MCU).
    let mut frame = vec![8];
    frame.extend_from_slice(&size.to_be_bytes());
    frame.extend_from_slice(&size.to_be_bytes());
    frame.push(3);
    for id in 1..=3u8 {
        frame.extend_from_slice(&[id, 0x11, 0]);
    }
    segment(&mut out, JPEG_SOF0, &frame);
    // The two Huffman tables: class 0 (DC) id 0, then class 1 (AC) id 0.
    for (class, bits, values) in [
        (0x00u8, &JPEG_DC_BITS, &JPEG_DC_VALUES[..]),
        (0x10u8, &JPEG_AC_BITS, &JPEG_AC_VALUES[..]),
    ] {
        let mut table = vec![class];
        table.extend_from_slice(bits);
        table.extend_from_slice(values);
        segment(&mut out, JPEG_DHT, &table);
    }
    // SOS: all three components, each using table pair 0, full spectral
    // selection (a baseline sequential scan).
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
            // The level shift, and the DC predictor: the first block of each
            // component carries the whole value, every later one differs from
            // its predecessor by nothing.
            let diff = if mcu == 0 { component - 128 } else { 0 };
            let category = if diff == 0 {
                0
            } else {
                32 - diff.unsigned_abs().leading_zeros()
            };
            let (code, length) = code_for(&dc_codes, category as u8);
            bits.push(code, length);
            if category > 0 {
                // A negative difference is sent as its one's complement in
                // `category` bits, which is what makes the leading bit the
                // sign.
                let value = if diff > 0 {
                    diff
                } else {
                    diff + (1 << category) - 1
                };
                bits.push(value as u32, category);
            }
            // Every AC coefficient of a constant block is zero.
            let (code, length) = code_for(&ac_codes, JPEG_EOB);
            bits.push(code, length);
        }
    }
    out.extend_from_slice(&bits.finish());
    out.extend_from_slice(&[0xFF, JPEG_EOI]);
    out
}

/// A marker segment: `FF <marker>`, the payload length including its own two
/// bytes, then the payload.
fn segment(out: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    out.extend_from_slice(&[0xFF, marker]);
    out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(payload);
}

/// Canonical Huffman codes from a JPEG `BITS`/`HUFFVAL` pair, as
/// `(symbol, code, length)` — the generation procedure from the spec's Annex
/// C, and the same walk a decoder makes to rebuild them from the DHT segment.
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

/// The entropy-coded segment's bit stream, most significant bit first.
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
                // Byte stuffing: an 0xFF inside the entropy stream is followed
                // by a 0x00 so a decoder cannot mistake it for a marker.
                if byte == 0xFF {
                    self.out.push(0x00);
                }
                self.accumulator = 0;
                self.filled = 0;
            }
        }
    }

    /// Pad the final partial byte with 1 bits, which is what the spec calls
    /// for — a 1-filled tail cannot be confused with the start of a marker.
    fn finish(mut self) -> Vec<u8> {
        while self.filled != 0 {
            self.push(1, 1);
        }
        self.out
    }
}

impl Cpu {
    /// `acc:u0` (`IAccountServiceForApplication`), `acc:u1`
    /// (`IAccountServiceForSystemService`) and `acc:su`
    /// (`IAccountServiceForAdministrator`): the console's user accounts.
    ///
    /// There is **one** user here and it is always signed in. That is not a
    /// placeholder for a user database — it is what this console is: no
    /// account applet to register a second user with, no profile UI, and
    /// nowhere to persist one to. So every "who is the current user" question
    /// has a determinate answer ([`ACCOUNT_UID`]), and every list is one entry
    /// long.
    ///
    /// A title asks early and does not proceed without an answer:
    /// `nn::account::Initialize` runs before save data is mounted, and
    /// `GetLastOpenedUser`/`TrySelectUserWithoutInteraction` are how it picks
    /// whose save to open. A zero uid is the "nobody is signed in" sentinel,
    /// which is what the generic fabricated-object-id fallback was effectively
    /// answering with before this existed.
    ///
    /// The three services share commands 0..=51 and diverge from 100 up, where
    /// the *same* command id means different things (100 is
    /// `InitializeApplicationInfo` on `acc:u0` but `GetUserRegistrationNotifier`
    /// on `acc:u1`), so those arms dispatch on which service the session was
    /// opened under rather than on the command alone.
    pub(super) fn acc_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        const QUERY_POINTER_BUFFER_SIZE: u32 = 3;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    // Which of the three services this session is decides what
                    // its 100+ commands mean, so the domain object inherits the
                    // name rather than being recorded as a generic "acc".
                    let name = self.service_name(handle).unwrap_or("acc:u0").to_string();
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, &name);
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                // `IProfile::Get` returns its `AccountUserData` through a
                // receive-static ("pointer") buffer, and a client told the
                // server has no room for one sends no descriptor at all — then
                // reads the icon id and background colour back out of its own
                // uninitialized stack. Same reasoning as `hid`'s.
                Some(QUERY_POINTER_BUFFER_SIZE) => {
                    self.write_ipc_response(tls, 0, &[], &0x1000u16.to_le_bytes(), &[])
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
            "acc:profile" | "acc:profile-editor" => self.acc_profile_request(tls, &iface, cmd_id),
            "acc:manager" => self.acc_manager_request(tls, handle, cmd_id),
            "acc:async-context" => self.acc_async_context_request(tls, cmd_id),
            // `INotifier::GetSystemEvent`, for the several notifiers `acc:u1`
            // hands out. The event is real and stays **unsignalled**: nothing
            // here ever registers a user, changes one's state, or syncs a
            // profile, so a notifier that never fires is the truthful model of
            // this console rather than a gap. (An event reported signalled
            // sends `nnSdk`'s system worker looking for a callback that was
            // never registered — see `am:applet-message`.)
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

    /// The commands on the account service itself, as opposed to the objects
    /// it hands out. `iface` is the service the session was opened under.
    fn acc_user_service_request(
        &mut self,
        tls: u32,
        handle: u64,
        iface: &str,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        // `acc:u0` is the application-facing service, `acc:u1`/`acc:su` the
        // system-facing ones.
        let application = iface == "acc:u0";
        match cmd_id {
            // GetUserCount -> s32.
            Some(0) => self.write_ipc_response(tls, 0, &[], &1i32.to_le_bytes(), &[]),
            // GetUserExistence(AccountUid) -> bool.
            Some(1) => {
                let exists = self.acc_requested_uid(tls) == ACCOUNT_UID;
                self.write_ipc_response(tls, 0, &[], &[u8::from(exists)], &[])
            }
            // ListAllUsers / ListOpenUsers / ListOpenContextStoredUsers /
            // ListQualifiedUsers: the same one-entry list each time. The user
            // exists, is signed in, has an open context, and qualifies for
            // whatever the title is about to do — there is no sign-out, and no
            // second account, to make those four lists differ.
            Some(2) | Some(3) | Some(60) | Some(141) => self.acc_write_user_list(tls),
            // GetLastOpenedUser -> AccountUid.
            Some(4) => self.write_ipc_response(tls, 0, &[], &ACCOUNT_UID, &[]),
            // GetProfile(AccountUid) -> IProfile.
            Some(5) => {
                if self.acc_requested_uid(tls) != ACCOUNT_UID {
                    return self.write_ipc_response(tls, ACCOUNT_USER_NOT_EXIST, &[], &[], &[]);
                }
                self.reply_with_interface(tls, handle, "acc:profile")?;
                Ok(())
            }
            // IsUserRegistrationRequestPermitted(u64) -> bool. Registering a
            // user means running the account applet, which does not exist
            // here — the one permission query on this console that is honestly
            // "no".
            Some(50) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
            // TrySelectUserWithoutInteraction(bool network_account_required)
            // -> AccountUid. This is how a title gets a user without putting
            // up the selector applet, and with one account it is also the
            // right answer: there is nothing to choose between.
            Some(51) => self.write_ipc_response(tls, 0, &[], &ACCOUNT_UID, &[]),
            // DebugActivateOpenContextRetention: retention is unconditional
            // here, since the one user's context is never dropped.
            Some(99) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // InitializeApplicationInfo: the title naming itself to `acc`,
            // which it does before asking `acc` anything else. Three command
            // ids for one call — 100 is the original, 140 replaced it in
            // 6.0.0, and 160 is what a current SDK sends. All three marshal
            // the same way and answer with a bare Result, so they share an
            // arm; nothing here varies by application.
            //
            // 160 is the one Tomodachi Life sends, and reading its request off
            // the wire is what identified it: a domain request of type 6
            // (RequestWithContext) with the pid flag set, carrying one u64 of
            // payload — zero, the placeholder the kernel overwrites — and no
            // buffers, no receive list, nothing for a reply to fill. Its
            // caller reads nothing back and aborts unless the Result is
            // success, which is what refusing the command did.
            //
            // 140 used to be answered with a user list, on the reading that it
            // was `ListQualifiedUsers`. That is 141. 140 only ever looked
            // right because a list reply is also a success.
            Some(100) | Some(140) | Some(160) if application => {
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // GetBaasAccountManagerForApplication(AccountUid) ->
            // IManagerForApplication.
            Some(101) if application => {
                self.reply_with_interface(tls, handle, "acc:manager")?;
                Ok(())
            }
            // AuthenticateApplicationAsync / CheckNetworkServiceAvailabilityAsync
            // -> IAsyncContext.
            Some(102) | Some(103) if application => {
                self.reply_with_interface(tls, handle, "acc:async-context")?;
                Ok(())
            }
            // From here down the session is `acc:u1`/`acc:su`, since every
            // application arm above is guarded and matches first.
            //
            // GetUserRegistrationNotifier / GetUserStateChangeNotifier /
            // GetBaasUserAvailabilityChangeNotifier / GetProfileUpdateNotifier
            // / GetProfileSyncNotifier -> INotifier.
            Some(100) | Some(101) | Some(103) | Some(104) | Some(106) => {
                self.reply_with_interface(tls, handle, "acc:notifier")?;
                Ok(())
            }
            // GetBaasAccountManagerForSystemService(AccountUid) ->
            // IManagerForSystemService — the same interface `acc:u0`'s command
            // 101 hands an application.
            Some(102) => {
                self.reply_with_interface(tls, handle, "acc:manager")?;
                Ok(())
            }
            // CheckNetworkServiceAvailabilityAsync -> IAsyncContext.
            Some(105) => {
                self.reply_with_interface(tls, handle, "acc:async-context")?;
                Ok(())
            }
            // StoreSaveDataThumbnail(AccountUid, buffer) /
            // ClearSaveDataThumbnail(AccountUid): the picture the home menu
            // shows beside a save. There is no home menu and no thumbnail
            // store, so the thumbnail is accepted and dropped — failing a call
            // a title makes on every save would be the larger lie.
            Some(110) | Some(111) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // IsUserAccountSwitchLocked -> bool. Locked: with one account
            // there is nothing to switch to, so a title that offers the
            // switch would be offering a dead end.
            Some(150) => self.write_ipc_response(tls, 0, &[], &[1u8], &[]),
            // IAccountServiceForAdministrator::GetProfileEditor(AccountUid) ->
            // IProfileEditor, the only route by which the nickname can be
            // changed from inside the guest.
            Some(205) if iface == "acc:su" => {
                if self.acc_requested_uid(tls) != ACCOUNT_UID {
                    return self.write_ipc_response(tls, ACCOUNT_USER_NOT_EXIST, &[], &[], &[]);
                }
                self.reply_with_interface(tls, handle, "acc:profile-editor")?;
                Ok(())
            }
            _ => self.unimplemented_command(tls, iface, cmd_id),
        }
    }

    /// `IProfile`, and `IProfileEditor` — the same interface plus the two
    /// store commands, which is why they share an arm.
    fn acc_profile_request(&mut self, tls: u32, iface: &str, cmd_id: Option<u32>) -> Result<()> {
        match cmd_id {
            // Get -> ProfileBase, with the AccountUserData in an output
            // buffer. The userdata is written even though every field of it is
            // zero here: the buffer belongs to the caller, and left untouched
            // it reads back as whatever was on that stack — an icon id and a
            // background colour chosen out of garbage.
            Some(0) => {
                if let Some((addr, size)) = self.ipc_output_buffer(tls, 0) {
                    if addr != 0 {
                        for i in 0..(size as usize).min(ACCOUNT_USER_DATA_LEN) as u32 {
                            self.mem.write_u8(addr.wrapping_add(i), 0)?;
                        }
                    }
                }
                let base = self.acc_profile_base();
                self.write_ipc_response(tls, 0, &[], &base, &[])
            }
            // GetBase -> ProfileBase.
            Some(1) => {
                let base = self.acc_profile_base();
                self.write_ipc_response(tls, 0, &[], &base, &[])
            }
            // GetImageSize / GetLargeImageSize [18.0.0+] -> u32, which has
            // to be the exact length 11 and 21 then write: a caller sizes its
            // buffer from this. There is one icon and no larger variant of
            // it, so both report the size of that one.
            Some(10) | Some(20) => {
                let size = profile_image().len() as u32;
                self.write_ipc_response(tls, 0, &[], &size.to_le_bytes(), &[])
            }
            // LoadImage / LoadLargeImage [18.0.0+] (out buffer) -> u32 bytes
            // written.
            Some(11) | Some(21) => {
                let image = profile_image();
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
            // GetImageId [18.0.0+] -> a Uuid naming the icon, which callers
            // hold onto to decide whether the copy they cached is still the
            // current one. Zero is that field's "no icon" sentinel, and the
            // generic reply left it reading whatever the caller's own stack
            // held -- a fresh id every call, so a cache that never hits.
            Some(30) => self.write_ipc_response(tls, 0, &[], &PROFILE_IMAGE_ID, &[]),
            // IProfileEditor::Store(ProfileBase, userdata), StoreWithImage
            // and StoreWithLargeImage [18.0.0+]: the nickname is the one part
            // of this profile that is real state, so a store writes it back
            // and a later GetBase reads out what was stored. Accepting an
            // edit and then reporting the old name is the failure mode a
            // `Set`/`Get` pair always has.
            Some(100) | Some(101) | Some(110) if iface == "acc:profile-editor" => {
                let at = self.ipc_request_data(tls);
                let nickname = self.read_string(at.wrapping_add(0x18), NICKNAME_LEN as u32);
                self.set_user_nickname(&nickname);
                self.account_edited_at = self.unix_time;
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            _ => self.unimplemented_command(tls, iface, cmd_id),
        }
    }

    /// `IManagerForApplication`/`IManagerForSystemService`: the Nintendo
    /// Account linked to the user, as far as a title can see it.
    fn acc_manager_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        match cmd_id {
            // CheckAvailability -> Result, where success means "this user has
            // a network service account".
            //
            // There is no Nintendo Account behind this user and no network
            // stack to authenticate one against, so this is the same trade
            // `nifm`'s permanently-connected ethernet link makes: reporting
            // the account unavailable sends a title down its offline path
            // (or into an error dialog) rather than letting it start. What it
            // still cannot get is a *token* — command 3 hands back an empty
            // one — so anything that genuinely authenticates fails there,
            // where the missing piece actually is.
            Some(0) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // GetAccountId -> u64 NetworkServiceAccountId. Nonzero, since zero
            // is this field's "no account" sentinel.
            Some(1) => {
                let id = NETWORK_SERVICE_ACCOUNT_ID.to_le_bytes();
                self.write_ipc_response(tls, 0, &[], &id, &[])
            }
            // EnsureIdTokenCacheAsync -> IAsyncContext.
            Some(2) => {
                self.reply_with_interface(tls, handle, "acc:async-context")?;
                Ok(())
            }
            // LoadIdTokenCache(out buffer) -> u32 size. There is no token to
            // cache, and an empty one is what an unlinked account has.
            Some(3) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            _ => self.unimplemented_command(tls, "acc:manager", cmd_id),
        }
    }

    /// `IAsyncContext`: the object an `*Async` command hands back so the
    /// caller can wait for the work.
    ///
    /// Every one of those commands here answered from state that was already
    /// in hand, so the context it returns is one that has already finished:
    /// its event is signalled the moment the guest asks for it, `HasDone` is
    /// true, and the result is success. A context that never completes hangs
    /// whatever is waiting on it.
    fn acc_async_context_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        match cmd_id {
            // GetSystemEvent.
            Some(0) => {
                let event = self.alloc_event("acc:async-context", false);
                self.signal_event(event);
                self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
            }
            // Cancel: nothing is running to cancel.
            Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // HasDone -> bool.
            Some(2) => self.write_ipc_response(tls, 0, &[], &[1u8], &[]),
            // GetResult -> Result.
            Some(3) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            _ => self.unimplemented_command(tls, "acc:async-context", cmd_id),
        }
    }

    /// The `AccountUid` an acc command carries as the first 16 bytes of its
    /// payload.
    fn acc_requested_uid(&self, tls: u32) -> [u8; 16] {
        let at = self.ipc_request_data(tls);
        let mut uid = [0u8; 16];
        for (index, byte) in uid.iter_mut().enumerate() {
            *byte = self.mem.read_u8(at.wrapping_add(index as u32)).unwrap_or(0);
        }
        uid
    }

    /// `nn::account::ProfileBase`: the uid, when the profile was last edited,
    /// and the nickname as a NUL-padded 0x20-byte field.
    fn acc_profile_base(&self) -> [u8; PROFILE_BASE_LEN] {
        let mut base = [0u8; PROFILE_BASE_LEN];
        base[..0x10].copy_from_slice(&ACCOUNT_UID);
        base[0x10..0x18].copy_from_slice(&self.account_edited_at.to_le_bytes());
        let nickname = self.account_nickname.as_bytes();
        let len = nickname.len().min(NICKNAME_LEN - 1);
        base[0x18..0x18 + len].copy_from_slice(&nickname[..len]);
        base
    }

    /// Write the console's one uid into a list command's output buffer, and
    /// answer with how many were written.
    ///
    /// The count goes in the reply whether or not the buffer had room for the
    /// uid: the client reads a fixed-size `s32` out of the raw data, and a
    /// reply too short for it fails in its CMIF parse rather than in the
    /// command that was actually asked.
    fn acc_write_user_list(&mut self, tls: u32) -> Result<()> {
        // The reply is a bare `Result`: these commands carry no count, and the
        // caller works out how many users there are by reading its own array
        // and stopping at the first all-zero uid. So the **whole** buffer has
        // to be written, not just the entry that exists — a server that fills
        // one slot and leaves the other seven is a console with one user and
        // seven made of whatever was on the caller's stack. That is what the
        // Home Menu found: it enumerated three accounts, asked `acc:su` for a
        // profile editor for each, and aborted when the third uid turned out
        // to be a pair of pointers.
        if let Some((addr, size)) = self.ipc_output_buffer(tls, 0) {
            if addr != 0 {
                for offset in 0..size {
                    self.mem.write_u8(addr.wrapping_add(offset), 0)?;
                }
                if size as usize >= ACCOUNT_UID.len() {
                    for (index, &byte) in ACCOUNT_UID.iter().enumerate() {
                        self.mem.write_u8(addr.wrapping_add(index as u32), byte)?;
                    }
                }
            }
        }
        self.write_ipc_response(tls, 0, &[], &[], &[])
    }
}

#[cfg(test)]
mod tests {
    use crate::cpu::ipc::testing::*;
    use crate::cpu::Cpu;

    /// Drive one acc command on a session opened under `service`.
    fn acc(cpu: &mut Cpu, service: &str, command_id: u32) {
        cpu.register_service_handle(9, service);
        cpu.acc_request(TLS, 9, Some(command_id)).unwrap();
    }

    #[test]
    fn acc_reports_one_user_who_is_signed_in() {
        // GetUserCount.
        let mut cpu = request(false, 0, &[]);
        acc(&mut cpu, "acc:u0", 0);
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 1);

        // GetLastOpenedUser: the uid, and not the zero that means "nobody is
        // signed in".
        let mut cpu = request(false, 4, &[]);
        acc(&mut cpu, "acc:u0", 4);
        let uid = cpu.read_bytes(TLS + 0x20, 16);
        assert_eq!(uid, super::ACCOUNT_UID.to_vec());
        assert_ne!(uid, vec![0u8; 16]);

        // TrySelectUserWithoutInteraction hands back the same one, since there
        // is nothing to choose between.
        let mut cpu = request(false, 51, &[0, 0, 0, 0]);
        acc(&mut cpu, "acc:u0", 51);
        assert_eq!(cpu.read_bytes(TLS + 0x20, 16), super::ACCOUNT_UID.to_vec());
    }

    #[test]
    fn acc_list_all_users_zeroes_the_slots_it_has_no_user_for() {
        // These commands carry no count: the caller passes a fixed array and
        // works out how many users there are by scanning for the first all-zero
        // uid. So every slot has to be written, not just the one that exists —
        // and the Home Menu is what proved it, enumerating three accounts out
        // of an array with one user and two of the caller's own stack in it.
        const BUFFER: u32 = 0x4000;
        let mut cpu = request_with_recv_buffer(2, &[], BUFFER, 0x40);
        cpu.mem.map_zero(BUFFER, 0x100).unwrap();
        for offset in 0..0x40 {
            cpu.mem.write_u8(BUFFER + offset, 0xAA).unwrap();
        }
        acc(&mut cpu, "acc:u0", 2);

        assert_eq!(cpu.read_bytes(BUFFER, 16), super::ACCOUNT_UID.to_vec());
        assert_eq!(
            cpu.read_bytes(BUFFER + 16, 0x30),
            vec![0u8; 0x30],
            "stale uids left behind"
        );
    }

    #[test]
    fn acc_knows_only_its_own_uid() {
        // GetUserExistence for the one user, then for a uid nothing handed out.
        let mut cpu = request(false, 1, &super::ACCOUNT_UID);
        acc(&mut cpu, "acc:u0", 1);
        assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 1);

        let mut cpu = request(false, 1, &[0xAB; 16]);
        acc(&mut cpu, "acc:u0", 1);
        assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 0);

        // GetProfile for that same invented uid fails rather than handing back
        // the one user's profile under someone else's id.
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
        // Whatever was on the caller's stack. Left alone, this is what it
        // would read back as its icon id and background colour.
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
        // ProfileBase: the uid, then the never-edited timestamp, then the
        // nickname.
        assert_eq!(cpu.read_bytes(TLS + 0x20, 16), super::ACCOUNT_UID.to_vec());
        assert_eq!(cpu.mem.read_u64(TLS + 0x30).unwrap(), 0);
        assert_eq!(cpu.read_string(TLS + 0x38, 0x20), "Player");
    }

    #[test]
    fn acc_profile_editor_stores_a_nickname_that_reads_back() {
        let mut store = [0u8; super::PROFILE_BASE_LEN];
        store[..16].copy_from_slice(&super::ACCOUNT_UID);
        store[0x18..0x18 + 5].copy_from_slice(b"Yuuto");
        let mut cpu = request(false, 100, &store);
        cpu.set_unix_time(1_700_000_000);
        cpu.register_service_handle(9, "acc:profile-editor");
        cpu.acc_request(TLS, 9, Some(100)).unwrap();
        assert_eq!(cpu.user_nickname(), "Yuuto");

        // GetBase reports what was stored, timestamp included — a store the
        // service accepts and then forgets is the failure mode every
        // Set/Get pair has.
        write_request(&mut cpu, 1, &[]);
        cpu.register_service_handle(9, "acc:profile");
        cpu.acc_request(TLS, 9, Some(1)).unwrap();
        assert_eq!(cpu.read_string(TLS + 0x38, 0x20), "Yuuto");
        assert_eq!(cpu.mem.read_u64(TLS + 0x30).unwrap(), 1_700_000_000);
    }

    #[test]
    fn acc_initialize_application_info_answers_every_id_it_has_had() {
        // The title naming itself to `acc`, which it does before asking `acc`
        // anything else. The command id moved with the SDK — 100, then 140 in
        // 6.0.0, then 160 — and all three are the same call: the pid and a u64
        // placeholder go out, a bare Result comes back. Tomodachi Life sends
        // 160, and refusing it aborted `nnSdk` before the title drew anything.
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

        // 141 is `ListQualifiedUsers`, and it is the one of these that answers
        // with the user list. 140 used to, on a misreading that only ever
        // looked right because a list reply is also a success.
        const BUFFER: u32 = 0x4000;
        let mut cpu = request_with_recv_buffer(141, &[], BUFFER, 0x40);
        cpu.mem.map_zero(BUFFER, 0x100).unwrap();
        cpu.register_service_handle(9, "acc:u0");
        cpu.acc_request(TLS, 9, Some(141)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0);
        assert_eq!(cpu.read_bytes(BUFFER, 16), super::ACCOUNT_UID.to_vec());
        assert_eq!(
            cpu.read_bytes(BUFFER + 16, 0x30),
            vec![0u8; 0x30],
            "one user listed"
        );
    }

    #[test]
    fn acc_the_same_command_id_means_different_things_on_u0_and_u1() {
        // 101 is GetBaasAccountManagerForApplication on acc:u0 and
        // GetUserStateChangeNotifier on acc:u1. Both hand back a session, so
        // the only way to tell them apart is what that session then answers:
        // the notifier has a GetSystemEvent, the manager has a GetAccountId.
        for (service, iface) in [("acc:u0", "acc:manager"), ("acc:u1", "acc:notifier")] {
            let mut cpu = request(false, 101, &super::ACCOUNT_UID);
            acc(&mut cpu, service, 101);
            let session = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;
            assert_eq!(cpu.service_name(session), Some(iface), "{service} cmd 101");
        }
    }

    #[test]
    fn acc_async_contexts_report_work_that_is_already_finished() {
        // CheckNetworkServiceAvailabilityAsync, then HasDone on what it
        // returned. A context that never completes hangs its waiter.
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
        assert_eq!(cpu.read_bytes(BUFFER, advertised), super::profile_image());

        // GetLargeImageSize and LoadLargeImage [18.0.0+] answer for the same
        // icon: there is no larger variant of it to report a second size for.
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
        assert_eq!(cpu.read_bytes(BUFFER, advertised), super::profile_image());
    }

    #[test]
    fn acc_names_the_icon_with_an_id_that_is_not_the_uid() {
        // GetImageId [18.0.0+] is what a caller caches its copy of the icon
        // against, so it has to be the same every call -- and nonzero, which
        // is that field's "no icon" sentinel.
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
            super::ACCOUNT_UID.to_vec(),
            "the icon, not the user"
        );
    }

    /// Decode the profile icon: walk its markers, rebuild the Huffman tables
    /// out of the DHT segments the file itself carries, and run the whole
    /// entropy-coded scan.
    ///
    /// A constant image is the strongest thing to assert against — every one
    /// of the 3072 blocks has to decode to the same colour, and the bit stream
    /// has to run out exactly at the EOI marker. That covers the tables, the
    /// canonical code generation, the DC prediction and the byte stuffing,
    /// none of which can be checked by eye.
    #[test]
    fn the_profile_icon_is_a_jpeg_that_decodes_to_one_colour() {
        let jpeg = super::profile_image();
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "SOI");
        assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9], "EOI");

        // Walk the marker segments, keeping what the scan needs.
        let mut quant = [0u8; 64];
        let mut tables: Vec<(u8, Vec<(u8, u16, u8)>)> = Vec::new();
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
                        // 1x1 sampling: one block per component per MCU, so no
                        // subsampling to undo below.
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

        /// A cursor over the scan's bits, most significant bit of each byte
        /// first, which is the order an entropy-coded segment is packed in.
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

            /// Read one Huffman-coded symbol: extend the code a bit at a time
            /// until it matches one the table defines, which is unambiguous
            /// because no code is a prefix of another.
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

        // Every block, in MCU order: a DC difference then an immediate
        // end-of-block, with the DC predictor carried per component.
        let blocks = width.div_ceil(8) * height.div_ceil(8);
        let mut predictor = [0i32; 3];
        for mcu in 0..blocks {
            for component in 0..3usize {
                let category = reader.symbol(dc_table);
                let mut diff = 0i32;
                if category > 0 {
                    let mut value = 0i32;
                    for _ in 0..category {
                        value = (value << 1) | reader.bit() as i32;
                    }
                    // The sign convention: a leading zero bit means the value
                    // is negative and stored as its one's complement.
                    diff = if value >= 1 << (category - 1) {
                        value
                    } else {
                        value - (1 << category) + 1
                    };
                }
                predictor[component] += diff;
                assert_eq!(
                    reader.symbol(ac_table),
                    super::JPEG_EOB,
                    "AC of a flat block"
                );

                // Dequantize and undo the level shift: the inverse DCT of a
                // lone DC coefficient is that coefficient over 8, everywhere.
                let value = predictor[component] * i32::from(quant[0]) / 8 + 128;
                let (red, green, blue) = super::PROFILE_IMAGE_COLOR;
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
        // Only the 1-padding of the last byte may be left over.
        assert!(
            scan.len() * 8 - reader.bit < 8,
            "the scan decodes to exactly the blocks the frame declares"
        );
    }
}
