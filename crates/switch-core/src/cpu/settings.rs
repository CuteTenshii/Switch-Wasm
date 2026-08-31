//! The console's settings services: `set`/`set:sys` (system settings and the
//! firmware version), `lbl` (the backlight), `notif` (scheduled alarms) and
//! `pctl` (parental controls).
//!
//! These are **stored, not answered**. One caller writes a setting and another
//! reads it back, so a value that is not kept is a setting that silently
//! reverts — which is a different bug from one that is simply unimplemented.
//!
//! `set:sys` is stored twice over: [`SystemSettings`] is what the running
//! console reads, and it lives in system save data
//! ([`SYSTEM_SETTINGS_SAVE`]), which the host writes back to the browser and
//! restores into the next session. A setting that does not survive a reload
//! is the same bug one step further out.

use super::Cpu;
use crate::trace::Level;
use crate::Result;

/// Language codes in `SetLanguage` order, as NUL-padded ASCII in a `u64`.
///
/// One table for the whole module: `set` lists these and maps an index to
/// one, and the console's own language is an entry in it that `set:sys`'s
/// `SetLanguageCode` can replace.
const LANGUAGE_CODES: [&str; 18] = [
    "ja", "en-US", "fr", "de", "it", "es", "zh-CN", "ko", "nl", "pt", "ru", "zh-TW", "en-GB",
    "fr-CA", "es-419", "zh-Hans", "zh-Hant", "pt-BR",
];

/// The language a console with no stored setting boots in (`SetLanguage_ENUS`).
const DEFAULT_LANGUAGE: usize = 1;

/// `SetRegion_USA`, the region that goes with it.
const DEFAULT_REGION: u32 = 1;

/// One language code, packed the way `nn::settings::LanguageCode` is.
fn language_code(index: usize) -> u64 {
    let mut packed = [0u8; 8];
    let name = LANGUAGE_CODES[index.min(LANGUAGE_CODES.len() - 1)].as_bytes();
    packed[..name.len()].copy_from_slice(name);
    u64::from_le_bytes(packed)
}

/// The system version `set:sys` reports, as major/minor/micro.
///
/// libnx seeds `hosversionGet` from this and branches on it everywhere, so the
/// number is load-bearing rather than decorative.
///
/// It sat at 12.1.0 for a long time, chosen to clear the gates the services
/// here implement (6.0.0 for `acc`'s qualified-user list) while staying below
/// the ones they did not — 17.0.0, where `ts` moves its measurement onto a
/// per-device `ISession`. Both of those are now implemented: `ts` routes
/// `OpenSession` to `ts:session-internal`/`ts:session-external`, and `acc`
/// answers `ListQualifiedUsers`. The ceiling the old number was avoiding is
/// gone, and staying under it meant claiming to be four years older than the
/// titles being run — Tomodachi Life alone reaches for `am` and `hid`
/// commands added in 18.0.0 and 20.0.0.
///
/// So this reports the current firmware. Nothing here implements everything a
/// 22.5.0 console does, and it never did at 12.1.0 either — the number says
/// which side of a feature gate to take, not what is finished behind it.
const FIRMWARE_VERSION: (u8, u8, u8) = (22, 5, 0);

/// Where the console keeps what `set:sys` serves.
///
/// System save data `8000000000000050` is the id hardware files these under,
/// and the id a guest mounting them through `fsp-srv` would name. Putting
/// them there rather than in a store of this service's own is what makes them
/// persist: the host already writes back every save it has been handed and
/// restores them into the next session, so a setting written here survives a
/// reload with no plumbing of its own.
pub(super) const SYSTEM_SETTINGS_SAVE: u64 = 0x8000_0000_0000_0050;

/// The file inside that save. Eden writes `settings` in the same place;
/// nothing on the guest side reads the name, but agreeing costs nothing.
const SYSTEM_SETTINGS_FILE: &str = "/settings";

/// What a stored block starts with. A file that does not open with these is
/// one this build cannot read — a newer layout, or something else entirely —
/// and the defaults are used rather than a half-parsed console.
const SYSTEM_SETTINGS_MAGIC: &[u8; 8] = b"swsetsys";
const SYSTEM_SETTINGS_VERSION: u32 = 1;

/// The widths of the settings blocks a caller hands over whole.
const TV_SETTINGS_SIZE: usize = 0x20;
const NOTIFICATION_SETTINGS_SIZE: usize = 0x18;
const SLEEP_SETTINGS_SIZE: usize = 0xc;
const INITIAL_LAUNCH_SETTINGS_SIZE: usize = 0x20;
const DEVICE_NICK_NAME_SIZE: usize = 0x80;
const LOCATION_NAME_SIZE: usize = 0x24;
const STEADY_CLOCK_TIME_POINT_SIZE: usize = 0x10;
const SYSTEM_CLOCK_CONTEXT_SIZE: usize = 0x20;
const CLOCK_SOURCE_ID_SIZE: usize = 0x10;
const EULA_VERSION_SIZE: usize = 0x30;
const ACCOUNT_NOTIFICATION_SETTINGS_SIZE: usize = 0x18;

/// How many `nn::settings::system::AudioOutputModeTarget`s there are: None,
/// Hdmi, Speaker, Headphone, and the two unnamed ones after them. Each keeps
/// its own mode, which is the entire point of the target argument.
const AUDIO_OUTPUT_TARGETS: usize = 6;

/// `AudioOutputMode_ch_2`, stereo — what this console's mixer produces, on
/// every output it could produce it on.
const AUDIO_OUTPUT_STEREO: u32 = 1;

/// The system settings, as one block.
///
/// Every field is something a caller can write and read back. Before this
/// existed each was a constant in the command table, and the ~40 `Set*`
/// commands beside them all fell through to the stub: the settings applet
/// wrote a colour set, a nickname or a keyboard layout, was told it had
/// worked, and read back the constant it would have got anyway.
///
/// The defaults are what those constants said, so a console that has never
/// been through the settings applet answers exactly as it used to.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct SystemSettings {
    /// The console's language, as the packed `nn::settings::LanguageCode`
    /// `set`'s `GetLanguageCode` reports, and the region beside it.
    pub(super) language_code: u64,
    region: u32,
    /// `KeyboardLayout`, which the software keyboard lays out its keys from.
    /// Kept separate from the language: hardware lets them disagree, and the
    /// applet that changes one does not touch the other.
    pub(super) keyboard_layout: u32,
    color_set: u32,
    account_settings: u32,
    applet_launch_flags: u32,
    chinese_traditional_input_method: u32,
    error_report_share_permission: u32,
    primary_album_storage: u32,
    push_notification_activity_mode_on_sleep: i32,
    platform_region: i32,
    panel_crc_mode: i32,
    touch_screen_mode: u32,
    quest_flag: u8,
    vibration_master_volume: f32,
    /// One mode per `AudioOutputModeTarget`, indexed by it.
    audio_output_mode: [u32; AUDIO_OUTPUT_TARGETS],
    lock_screen: bool,
    console_information_upload: bool,
    automatic_application_download: bool,
    speaker_auto_mute: bool,
    usb30_enable: bool,
    /// The two radios. `nfc:sys` and `btm:sys` answer out of these rather
    /// than out of state of their own: the switch in the settings applet and
    /// the switch a service reads are one switch.
    pub(super) nfc_enable: bool,
    pub(super) bluetooth_enable: bool,
    wireless_lan_enable: bool,
    auto_update_enable: bool,
    battery_percentage: bool,
    field_testing: bool,
    user_clock_automatic_correction: bool,
    /// The blocks a caller hands over whole and reads back whole. Their
    /// fields belong to the caller, so they are stored as the bytes they
    /// arrived as rather than picked apart into something this console would
    /// have to put back together.
    tv_settings: [u8; TV_SETTINGS_SIZE],
    notification_settings: [u8; NOTIFICATION_SETTINGS_SIZE],
    sleep_settings: [u8; SLEEP_SETTINGS_SIZE],
    initial_launch_settings: [u8; INITIAL_LAUNCH_SETTINGS_SIZE],
    device_nick_name: [u8; DEVICE_NICK_NAME_SIZE],
    pub(super) device_time_zone_location_name: [u8; LOCATION_NAME_SIZE],
    device_time_zone_updated_time: [u8; STEADY_CLOCK_TIME_POINT_SIZE],
    user_clock_correction_updated_time: [u8; STEADY_CLOCK_TIME_POINT_SIZE],
    user_clock_context: [u8; SYSTEM_CLOCK_CONTEXT_SIZE],
    network_clock_context: [u8; SYSTEM_CLOCK_CONTEXT_SIZE],
    external_steady_clock_source_id: [u8; CLOCK_SOURCE_ID_SIZE],
    external_steady_clock_internal_offset: i64,
    /// The agreements this console has accepted and the per-account
    /// notification overrides: two lists a caller replaces wholesale.
    eula_versions: Vec<[u8; EULA_VERSION_SIZE]>,
    account_notification_settings: Vec<[u8; ACCOUNT_NOTIFICATION_SETTINGS_SIZE]>,
}

impl Default for SystemSettings {
    fn default() -> SystemSettings {
        // `EulaVersion { u32 version; SystemRegionCode region;
        // EulaVersionClockType clock_type; pad[4]; SystemClockContext; }`.
        // A console that has accepted none has not finished first-time setup,
        // and the Home Menu hands over to `starter` for that — which nothing
        // here can launch. When it was accepted is not tracked, so the clock
        // context stays zero; callers gate on the version and the region.
        const EULA_VERSION: u32 = 0x1_0000;
        const EULA_STEADY_CLOCK: u32 = 1;
        let mut eula = [0u8; EULA_VERSION_SIZE];
        eula[0x00..0x04].copy_from_slice(&EULA_VERSION.to_le_bytes());
        eula[0x04..0x08].copy_from_slice(&DEFAULT_REGION.to_le_bytes());
        eula[0x08..0x0c].copy_from_slice(&EULA_STEADY_CLOCK.to_le_bytes());

        // `TvSettings`: CEC and burn-in prevention on, resolution and RGB
        // range Auto, no colour transform. The tail past the first four words
        // is what the reply padding used to leave stale — two floats, so a
        // NaN gamma was a reachable answer rather than merely a wrong one.
        const ALLOWS_CEC: u32 = 1 << 2;
        const PREVENTS_SCREEN_BURN_IN: u32 = 1 << 3;
        const HDMI_CONTENT_TYPE_GAME: u32 = 4;
        let mut tv = [0u8; TV_SETTINGS_SIZE];
        tv[0x00..0x04].copy_from_slice(&(ALLOWS_CEC | PREVENTS_SCREEN_BURN_IN).to_le_bytes());
        tv[0x08..0x0c].copy_from_slice(&HDMI_CONTENT_TYPE_GAME.to_le_bytes());
        tv[0x18..0x1c].copy_from_slice(&1.0f32.to_le_bytes());
        tv[0x1c..0x20].copy_from_slice(&0.5f32.to_le_bytes());

        // `NotificationSettings { flags; volume; start_time; stop_time; }`,
        // quiet from nine in the evening to nine in the morning.
        const ENABLES_NEWS: u32 = 1 << 8;
        const INCOMING_LAMP: u32 = 1 << 9;
        const VOLUME_HIGH: u32 = 2;
        let mut notification = [0u8; NOTIFICATION_SETTINGS_SIZE];
        notification[0x00..0x04].copy_from_slice(&(ENABLES_NEWS | INCOMING_LAMP).to_le_bytes());
        notification[0x04..0x08].copy_from_slice(&VOLUME_HIGH.to_le_bytes());
        notification[0x08..0x0c].copy_from_slice(&9u32.to_le_bytes());
        notification[0x10..0x14].copy_from_slice(&21u32.to_le_bytes());

        // `SleepSettings { flags; handheld_plan; console_plan; }`. Both plans
        // are `Never` (5), and the zero the stub used to leave was not a
        // duration but a plan *index* — it said "sleep after one minute".
        // Nothing here dims a screen this emulator does not own.
        const SLEEP_NEVER: u32 = 5;
        let mut sleep = [0u8; SLEEP_SETTINGS_SIZE];
        sleep[0x04..0x08].copy_from_slice(&SLEEP_NEVER.to_le_bytes());
        sleep[0x08..0x0c].copy_from_slice(&SLEEP_NEVER.to_le_bytes());

        // `InitialLaunchSettings { InitialLaunchFlag; pad[4]; timestamp; }`.
        // The flags say the console has been through first-time setup; a
        // console that has not is one the Home Menu will not draw a menu for.
        const LAUNCH_COMPLETION: u32 = 1;
        const LAUNCH_USER_ADDITION: u32 = 1 << 8;
        const LAUNCH_TIMESTAMP: u32 = 1 << 16;
        let mut initial_launch = [0u8; INITIAL_LAUNCH_SETTINGS_SIZE];
        initial_launch[..4].copy_from_slice(
            &(LAUNCH_COMPLETION | LAUNCH_USER_ADDITION | LAUNCH_TIMESTAMP).to_le_bytes(),
        );

        let mut nick_name = [0u8; DEVICE_NICK_NAME_SIZE];
        nick_name[..DEVICE_NICK_NAME.len()].copy_from_slice(DEVICE_NICK_NAME);

        // The one zone there is: `time` has no TZif database to resolve any
        // other, so a name here that it cannot convert against would be a
        // console whose clock disagrees with its own settings screen.
        let mut location = [0u8; LOCATION_NAME_SIZE];
        location[..DEVICE_TIME_ZONE.len()].copy_from_slice(DEVICE_TIME_ZONE);

        SystemSettings {
            language_code: language_code(DEFAULT_LANGUAGE),
            region: DEFAULT_REGION,
            // `KeyboardLayout_EnglishUs`. Zero is `Japanese` — a real layout,
            // but not this console's.
            keyboard_layout: 1,
            // `ColorSet_BasicWhite`, the light theme this menu is drawn in.
            color_set: 0,
            account_settings: 0,
            applet_launch_flags: 0,
            chinese_traditional_input_method: 0,
            // `ErrorReportSharePermission_NotConfirmed`, which is the truth:
            // nothing has asked.
            error_report_share_permission: 0,
            // `PrimaryAlbumStorage_Nand`. There is no album on the card here.
            primary_album_storage: 0,
            push_notification_activity_mode_on_sleep: 0,
            // `PlatformRegion_Global`, which has no zero — see the command.
            platform_region: 1,
            panel_crc_mode: 0,
            // `TouchScreenMode_Standard`. The zero the stub left is `Stylus`,
            // which is a real mode and the wrong one for a console driven by
            // a finger on a browser canvas.
            touch_screen_mode: 1,
            // `QuestFlag_Retail`; a kiosk unit runs a different Home Menu.
            quest_flag: 0,
            vibration_master_volume: 1.0,
            audio_output_mode: [AUDIO_OUTPUT_STEREO; AUDIO_OUTPUT_TARGETS],
            lock_screen: false,
            console_information_upload: false,
            automatic_application_download: false,
            speaker_auto_mute: false,
            usb30_enable: false,
            // No reader is attached, so nothing scans; the switch still reads
            // back off, which is what a console with NFC turned off says.
            nfc_enable: false,
            // A console boots with its Bluetooth radio on: that is how it
            // finds the Joy-Cons it is already paired to.
            bluetooth_enable: true,
            // And with wireless on — there is a network stack behind this one.
            wireless_lan_enable: true,
            auto_update_enable: false,
            battery_percentage: false,
            field_testing: false,
            // Nothing here corrects a clock against a network time server, so
            // saying it does would be a console that never catches up.
            user_clock_automatic_correction: false,
            tv_settings: tv,
            notification_settings: notification,
            sleep_settings: sleep,
            initial_launch_settings: initial_launch,
            device_nick_name: nick_name,
            device_time_zone_location_name: location,
            device_time_zone_updated_time: [0; STEADY_CLOCK_TIME_POINT_SIZE],
            user_clock_correction_updated_time: [0; STEADY_CLOCK_TIME_POINT_SIZE],
            user_clock_context: [0; SYSTEM_CLOCK_CONTEXT_SIZE],
            network_clock_context: [0; SYSTEM_CLOCK_CONTEXT_SIZE],
            external_steady_clock_source_id: [0; CLOCK_SOURCE_ID_SIZE],
            external_steady_clock_internal_offset: 0,
            eula_versions: vec![eula],
            account_notification_settings: Vec::new(),
        }
    }
}

/// What this console calls itself, until something renames it.
const DEVICE_NICK_NAME: &[u8] = b"switch-wasm";

/// The zone `time` resolves every calendar conversion against.
pub(super) const DEVICE_TIME_ZONE: &[u8] = b"UTC";

/// The `Uuid` every Mii made on this console is stamped with — ASCII, so a
/// Mii dumped out of here says where it came from. Eden's is the same trick
/// ("Eden Default UID").
///
/// Fixed rather than generated: a Mii made in one session has to still be
/// this console's in the next, and a fresh id each boot would disown all of
/// them. Nothing here has been observed reading it.
const MII_AUTHOR_ID: [u8; 0x10] = [
    0x73, 0x77, 0x69, 0x74, 0x63, 0x68, 0x2d, 0x77, 0x61, 0x73, 0x6d, 0x00, 0x00, 0x00, 0x00, 0x01,
];

/// `HomeMenuScheme`: the main, back, sub, bezel and extra colours the Home
/// Menu tints itself with, as ARGB.
///
/// These are **not measured from hardware**. They are Eden's, which its own
/// `GetHomeMenuScheme` marks stubbed — a dark grey on grey with white
/// accents. What matters here is that the five words are a coherent scheme
/// and that `extra` is opaque black rather than a colour picked to look like
/// something: a menu that draws with these gets a plausible theme, and one
/// that draws with the stale reply padding gets whatever was in TLS.
const HOME_MENU_SCHEME: [u32; 5] = [
    0xff32_3232,
    0xff32_3232,
    0xffff_ffff,
    0xffff_ffff,
    0xff00_0000,
];

/// The firmware's settings items, as `GetSettingsItemValue` serves them.
///
/// These are not the settings above: nothing writes them, and they are not
/// per-console. They are the compiled-in constants firmware components read
/// out of `set:sys` instead of hard-coding — a heap reservation, a clock
/// interval, whether the platform has a rail — and a component that cannot
/// read one does not carry on with a default of its own.
///
/// The set is Eden's, minus its `hid_debug` block: `hid` here is emulated
/// rather than the sysmodule those items configure, so answering them would
/// be describing a component that is not running.
fn settings_item(category: &str, name: &str) -> Option<Vec<u8>> {
    let value = match (category, name) {
        // `hbloader`, which reads how much heap to leave an applet.
        ("hbloader", "applet_heap_size") => 0u64.to_le_bytes().to_vec(),
        ("hbloader", "applet_heap_reservation_size") => 0x860_0000u64.to_le_bytes().to_vec(),
        // `time`'s intervals and the year an unset clock starts at.
        ("time", "notify_time_to_fs_interval_seconds") => 600i32.to_le_bytes().to_vec(),
        ("time", "standard_network_clock_sufficient_accuracy_minutes") => {
            43_200i32.to_le_bytes().to_vec()
        }
        ("time", "standard_steady_clock_rtc_update_interval_minutes") => {
            5i32.to_le_bytes().to_vec()
        }
        ("time", "standard_steady_clock_test_offset_minutes") => 0i32.to_le_bytes().to_vec(),
        ("time", "standard_user_clock_initial_year") => 2023i32.to_le_bytes().to_vec(),
        // What the platform is wired with: a Joy-Con rail, and the
        // microcontroller behind it.
        ("hid", "has_rail_interface") => vec![1],
        ("hid", "has_sio_mcu") => vec![1],
        ("mii", "is_db_test_mode_enabled") => vec![0],
        // Read by `GetDebugModeFlag` as well as by name.
        ("settings_debug", "is_debug_mode_enabled") => vec![0],
        // Whether the error applet closes itself. It does not: an error that
        // vanishes before it is read is one nobody can report.
        ("err", "applet_auto_close") => vec![0],
        _ => return None,
    };
    Some(value)
}

impl SystemSettings {
    /// The stored form: a header, then one record per setting.
    ///
    /// Each record is tagged with the `set:sys` command id that carries the
    /// setting, so the tags need no namespace of their own and a record can
    /// be traced back to the command that wrote it. A reader keeps its
    /// default for a tag that is absent and skips one it does not know, so a
    /// build that adds a setting still reads a file written before it.
    fn serialize(&self) -> Vec<u8> {
        fn record(out: &mut Vec<u8>, tag: u32, bytes: &[u8]) {
            out.extend_from_slice(&tag.to_le_bytes());
            out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(bytes);
        }

        let mut out = Vec::new();
        out.extend_from_slice(SYSTEM_SETTINGS_MAGIC);
        out.extend_from_slice(&SYSTEM_SETTINGS_VERSION.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());

        record(&mut out, 0, &self.language_code.to_le_bytes());
        record(&mut out, 7, &[u8::from(self.lock_screen)]);
        record(&mut out, 13, &self.external_steady_clock_source_id);
        record(&mut out, 15, &self.user_clock_context);
        record(&mut out, 17, &self.account_settings.to_le_bytes());
        let eula: Vec<u8> = self.eula_versions.concat();
        record(&mut out, 21, &eula);
        record(&mut out, 23, &self.color_set.to_le_bytes());
        record(&mut out, 25, &[u8::from(self.console_information_upload)]);
        record(
            &mut out,
            27,
            &[u8::from(self.automatic_application_download)],
        );
        record(&mut out, 29, &self.notification_settings);
        let account_notifications: Vec<u8> = self.account_notification_settings.concat();
        record(&mut out, 31, &account_notifications);
        record(&mut out, 35, &self.vibration_master_volume.to_le_bytes());
        record(&mut out, 39, &self.tv_settings);
        let modes: Vec<u8> = self
            .audio_output_mode
            .iter()
            .flat_map(|mode| mode.to_le_bytes())
            .collect();
        record(&mut out, 43, &modes);
        record(&mut out, 45, &[u8::from(self.speaker_auto_mute)]);
        record(&mut out, 47, &[self.quest_flag]);
        record(&mut out, 53, &self.device_time_zone_location_name);
        record(&mut out, 57, &self.region.to_le_bytes());
        record(&mut out, 58, &self.network_clock_context);
        record(
            &mut out,
            60,
            &[u8::from(self.user_clock_automatic_correction)],
        );
        record(&mut out, 63, &self.primary_album_storage.to_le_bytes());
        record(&mut out, 65, &[u8::from(self.usb30_enable)]);
        record(&mut out, 69, &[u8::from(self.nfc_enable)]);
        record(&mut out, 71, &self.sleep_settings);
        record(&mut out, 73, &[u8::from(self.wireless_lan_enable)]);
        record(&mut out, 75, &self.initial_launch_settings);
        record(&mut out, 77, &self.device_nick_name);
        record(&mut out, 88, &[u8::from(self.bluetooth_enable)]);
        record(&mut out, 95, &[u8::from(self.auto_update_enable)]);
        record(&mut out, 99, &[u8::from(self.battery_percentage)]);
        record(
            &mut out,
            106,
            &self.external_steady_clock_internal_offset.to_le_bytes(),
        );
        record(
            &mut out,
            120,
            &self.push_notification_activity_mode_on_sleep.to_le_bytes(),
        );
        record(
            &mut out,
            124,
            &self.error_report_share_permission.to_le_bytes(),
        );
        record(&mut out, 126, &self.applet_launch_flags.to_le_bytes());
        record(&mut out, 136, &self.keyboard_layout.to_le_bytes());
        record(&mut out, 150, &self.device_time_zone_updated_time);
        record(&mut out, 152, &self.user_clock_correction_updated_time);
        record(
            &mut out,
            170,
            &self.chinese_traditional_input_method.to_le_bytes(),
        );
        record(&mut out, 183, &self.platform_region.to_le_bytes());
        record(&mut out, 187, &self.touch_screen_mode.to_le_bytes());
        record(&mut out, 201, &[u8::from(self.field_testing)]);
        record(&mut out, 203, &self.panel_crc_mode.to_le_bytes());
        out
    }

    /// Read a block back, or `None` when the bytes are not one this build
    /// wrote — in which case the caller keeps the defaults rather than a
    /// console assembled out of whatever the file did contain.
    fn parse(stored: &[u8]) -> Option<SystemSettings> {
        const HEADER: usize = 0x10;
        if stored.len() < HEADER || &stored[..8] != SYSTEM_SETTINGS_MAGIC {
            return None;
        }
        if u32::from_le_bytes(stored[8..12].try_into().ok()?) != SYSTEM_SETTINGS_VERSION {
            return None;
        }
        let mut settings = SystemSettings::default();
        let mut at = HEADER;
        while at + 8 <= stored.len() {
            let tag = u32::from_le_bytes(stored[at..at + 4].try_into().ok()?);
            let len = u32::from_le_bytes(stored[at + 4..at + 8].try_into().ok()?) as usize;
            at += 8;
            // A record that runs past the end is a truncated file: what has
            // been read so far stands, and there is nothing after it.
            let Some(value) = stored.get(at..at + len) else {
                break;
            };
            at += len;
            settings.restore(tag, value);
        }
        Some(settings)
    }

    /// One record back into its field. A value of the wrong width is left
    /// out — the default is a setting this console can answer with, and a
    /// half-written block is not.
    fn restore(&mut self, tag: u32, value: &[u8]) {
        fn u32_at(value: &[u8]) -> Option<u32> {
            Some(u32::from_le_bytes(value.get(..4)?.try_into().ok()?))
        }
        fn u64_at(value: &[u8]) -> Option<u64> {
            Some(u64::from_le_bytes(value.get(..8)?.try_into().ok()?))
        }
        fn flag(value: &[u8]) -> Option<bool> {
            Some(value.first()? != &0)
        }
        fn block<const N: usize>(value: &[u8]) -> Option<[u8; N]> {
            value.get(..N)?.try_into().ok()
        }
        fn list<const N: usize>(value: &[u8]) -> Vec<[u8; N]> {
            value
                .chunks_exact(N)
                .filter_map(|entry| entry.try_into().ok())
                .collect()
        }

        match tag {
            0 => self.language_code = u64_at(value).unwrap_or(self.language_code),
            7 => self.lock_screen = flag(value).unwrap_or(self.lock_screen),
            13 => {
                self.external_steady_clock_source_id =
                    block(value).unwrap_or(self.external_steady_clock_source_id)
            }
            15 => self.user_clock_context = block(value).unwrap_or(self.user_clock_context),
            17 => self.account_settings = u32_at(value).unwrap_or(self.account_settings),
            21 => self.eula_versions = list(value),
            23 => self.color_set = u32_at(value).unwrap_or(self.color_set),
            25 => {
                self.console_information_upload =
                    flag(value).unwrap_or(self.console_information_upload)
            }
            27 => {
                self.automatic_application_download =
                    flag(value).unwrap_or(self.automatic_application_download)
            }
            29 => self.notification_settings = block(value).unwrap_or(self.notification_settings),
            31 => self.account_notification_settings = list(value),
            35 => {
                self.vibration_master_volume = u32_at(value)
                    .map(f32::from_bits)
                    .unwrap_or(self.vibration_master_volume)
            }
            39 => self.tv_settings = block(value).unwrap_or(self.tv_settings),
            43 => {
                for (target, mode) in value.chunks_exact(4).enumerate() {
                    if let (Some(slot), Some(mode)) =
                        (self.audio_output_mode.get_mut(target), u32_at(mode))
                    {
                        *slot = mode;
                    }
                }
            }
            45 => self.speaker_auto_mute = flag(value).unwrap_or(self.speaker_auto_mute),
            47 => self.quest_flag = value.first().copied().unwrap_or(self.quest_flag),
            53 => {
                self.device_time_zone_location_name =
                    block(value).unwrap_or(self.device_time_zone_location_name)
            }
            57 => self.region = u32_at(value).unwrap_or(self.region),
            58 => self.network_clock_context = block(value).unwrap_or(self.network_clock_context),
            60 => {
                self.user_clock_automatic_correction =
                    flag(value).unwrap_or(self.user_clock_automatic_correction)
            }
            63 => self.primary_album_storage = u32_at(value).unwrap_or(self.primary_album_storage),
            65 => self.usb30_enable = flag(value).unwrap_or(self.usb30_enable),
            69 => self.nfc_enable = flag(value).unwrap_or(self.nfc_enable),
            71 => self.sleep_settings = block(value).unwrap_or(self.sleep_settings),
            73 => self.wireless_lan_enable = flag(value).unwrap_or(self.wireless_lan_enable),
            75 => {
                self.initial_launch_settings = block(value).unwrap_or(self.initial_launch_settings)
            }
            77 => self.device_nick_name = block(value).unwrap_or(self.device_nick_name),
            88 => self.bluetooth_enable = flag(value).unwrap_or(self.bluetooth_enable),
            95 => self.auto_update_enable = flag(value).unwrap_or(self.auto_update_enable),
            99 => self.battery_percentage = flag(value).unwrap_or(self.battery_percentage),
            106 => {
                self.external_steady_clock_internal_offset = u64_at(value)
                    .map(|raw| raw as i64)
                    .unwrap_or(self.external_steady_clock_internal_offset)
            }
            120 => {
                self.push_notification_activity_mode_on_sleep = u32_at(value)
                    .map(|raw| raw as i32)
                    .unwrap_or(self.push_notification_activity_mode_on_sleep)
            }
            124 => {
                self.error_report_share_permission =
                    u32_at(value).unwrap_or(self.error_report_share_permission)
            }
            126 => self.applet_launch_flags = u32_at(value).unwrap_or(self.applet_launch_flags),
            136 => self.keyboard_layout = u32_at(value).unwrap_or(self.keyboard_layout),
            150 => {
                self.device_time_zone_updated_time =
                    block(value).unwrap_or(self.device_time_zone_updated_time)
            }
            152 => {
                self.user_clock_correction_updated_time =
                    block(value).unwrap_or(self.user_clock_correction_updated_time)
            }
            170 => {
                self.chinese_traditional_input_method =
                    u32_at(value).unwrap_or(self.chinese_traditional_input_method)
            }
            183 => {
                self.platform_region = u32_at(value)
                    .map(|raw| raw as i32)
                    .unwrap_or(self.platform_region)
            }
            187 => self.touch_screen_mode = u32_at(value).unwrap_or(self.touch_screen_mode),
            201 => self.field_testing = flag(value).unwrap_or(self.field_testing),
            203 => {
                self.panel_crc_mode = u32_at(value)
                    .map(|raw| raw as i32)
                    .unwrap_or(self.panel_crc_mode)
            }
            // A record this build has no field for: a setting added later,
            // read back by a build that has it again.
            _ => {}
        }
    }
}

/// `lbl`'s view of the panel backlight — everything a caller set, kept so
/// that it reads back.
///
/// None of it reaches a panel: there is no PWM behind this and the host
/// decides its own window brightness. What matters is that the settings
/// agree with each other, because the system settings applet writes one and
/// then reads the *other* — it sets a brightness and asks what is applied to
/// the backlight, and a console that answers those two independently is a
/// console whose brightness slider does not move.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Backlight {
    /// The brightness setting, 0.0–1.0, and the copy `SaveCurrentSetting`
    /// took of it for `LoadCurrentSetting` to put back.
    setting: f32,
    saved: f32,
    /// The separate brightness VR mode runs at.
    vr_setting: f32,
    /// Whether the panel is lit at all. This is not a brightness of zero:
    /// `SwitchBacklightOff` leaves the setting alone, and the applet that
    /// turns the screen off expects to find its slider where it left it.
    on: bool,
    dimming: bool,
    auto_brightness: bool,
    vr_mode: bool,
    /// The ambient light sensor's last reading, in lux. There is no sensor
    /// here, so this is only ever what `SetAmbientLightSensorValue` put there.
    lux: f32,
    /// The three-point mappings and the reflection delay. Nothing reads these
    /// but their own getters, which is the entire reason they are stored.
    brightness_mapping: [f32; 3],
    lux_mapping: [f32; 3],
    reflection_delay: f32,
}

impl Default for Backlight {
    fn default() -> Backlight {
        Backlight {
            setting: 1.0,
            saved: 1.0,
            vr_setting: 1.0,
            on: true,
            // A retail console dims an idle screen and does not use the light
            // sensor unless the user turns auto-brightness on.
            dimming: true,
            auto_brightness: false,
            vr_mode: false,
            lux: 0.0,
            brightness_mapping: [0.0; 3],
            lux_mapping: [0.0; 3],
            reflection_delay: 0.0,
        }
    }
}

/// One alarm `notif` has been asked to keep, as the caller gave it.
///
/// The setting is the caller's own 0x40-byte `nn::notification::AlarmSetting`
/// with the id the system assigned written into it, and the parameter is the
/// opaque blob a title attaches for its own use when the alarm fires. Both
/// come back verbatim, which is the whole contract: `notif` stores these, it
/// does not interpret them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct AlarmSetting {
    id: u16,
    setting: Vec<u8>,
    parameter: Vec<u8>,
}

/// The size of `nn::notification::AlarmSetting`, and where its id sits.
const ALARM_SETTING_SIZE: usize = 0x40;

const ALARM_SETTING_ID: usize = 0;

/// The largest `ApplicationParameter` an alarm may carry.
const ALARM_PARAMETER_MAX: u32 = 0x400;

impl Cpu {
    pub(super) fn set_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        // The control interface every session carries, which has to be
        // answered before the service's own table is consulted — and `set`
        // is the case that shows why. It has a command **3** of its own,
        // `GetAvailableLanguageCodeCount`, so without this the two collided:
        // `nnSdk` opened the session, asked how large a pointer buffer it may
        // send through, and was told 18 — the number of language codes this
        // console has. It will not marshal a command whose buffer does not
        // fit, so Just Dance 2017 closed the session again without ever
        // sending a settings command and aborted inside
        // `nn::settings::LanguageCode::Make`, 660 million instructions in.
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "set");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, "set:control", cmd_id),
            };
        }
        // How many of the codes the pre-4.0.0 pair of commands reports. The cap is
        // that command's own 15-entry array, not the twelve languages that
        // predate 4.0.0: `nn::settings::detail::MakeLanguageCode` asks command
        // 1 for 15 codes and indexes the answer by `SetLanguage` directly, so
        // reporting twelve aborted Minecraft on `fr-CA` (13). Only `zh-Hans`,
        // `zh-Hant` and `pt-BR` are out of the legacy pair's reach.
        const LEGACY_LANGUAGE_CODES: usize = 15;

        let code = language_code;

        match cmd_id {
            // GetRegionCode -> SetRegion, and GetLanguageCode -> the packed
            // code. Both come out of the stored system settings rather than
            // out of a constant: `set:sys`'s SetRegionCode and SetLanguageCode
            // write those two fields, and a console that answers here from a
            // constant is one whose language reverts the moment it is read.
            Some(4) => {
                let region = self.system_settings().region;
                self.write_ipc_response(tls, 0, &[], &region.to_le_bytes(), &[])
            }
            Some(0) => {
                let raw = self.system_settings().language_code.to_le_bytes();
                self.write_ipc_response(tls, 0, &[], &raw, &[])
            }
            // MakeLanguageCode(SetLanguage) -> u64 code. This one is a lookup
            // in the table above and not a question about this console, so it
            // answers for whichever language it was handed.
            Some(2) => {
                let language = self.mem.read_u32(self.ipc_request_data(tls)).unwrap_or(0);
                let index = (language as usize).min(LANGUAGE_CODES.len() - 1);
                self.write_ipc_response(tls, 0, &[], &code(index).to_le_bytes(), &[])
            }
            // GetAvailableLanguageCodes (1 = pre-4.0.0) and
            // GetAvailableLanguageCodes2 (5 = current): fill the out buffer
            // with the codes and return how many were written. The only
            // difference between them is how the buffer arrives — 1 offers a
            // receive-static ("pointer") one, 5 a map-alias one, and
            // `ipc_output_buffer` takes either.
            //
            // 1 was left to the catch-all, which answers with success and no
            // data at all: `nn::settings::LanguageCode::Make` read the count
            // back as zero, found no code for the language it had been asked
            // for, and aborted. That is where Just Dance 2017 stopped once it
            // had a RomFS to read — it is a pre-4.0.0 title, and 1 is the only
            // one of the two it knows.
            Some(1) | Some(5) => {
                let available = match cmd_id {
                    Some(1) => LEGACY_LANGUAGE_CODES,
                    _ => LANGUAGE_CODES.len(),
                };
                let mut written = 0usize;
                if let Some((addr, size)) = self.ipc_output_buffer(tls, 0) {
                    if addr != 0 {
                        written = (size as usize / 8).min(available);
                        for index in 0..written {
                            self.mem
                                .write_u64(addr.wrapping_add((index * 8) as u32), code(index))?;
                        }
                    }
                }
                self.write_ipc_response(tls, 0, &[], &(written as u32).to_le_bytes(), &[])
            }
            // GetAvailableLanguageCodeCount (3 = pre-4.0.0, 6 = current). Each
            // has to agree with the command that fills the buffer beside it: a
            // count larger than what `GetAvailableLanguageCodes` writes is a
            // caller indexing past the codes it was given.
            Some(3) | Some(6) => {
                let total = match cmd_id {
                    Some(3) => LEGACY_LANGUAGE_CODES,
                    _ => LANGUAGE_CODES.len(),
                } as u32;
                self.write_ipc_response(tls, 0, &[], &total.to_le_bytes(), &[])
            }
            _ => {
                self.warn_stub(
                    "set",
                    cmd_id,
                    "an empty success, so the caller reads its own buffer as the answer",
                );
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
        }
    }

    /// The system settings, loaded the first time anything asks for them.
    ///
    /// They live in save data, and a save is only in a session once the host
    /// has restored it — which happens after the session is built and before
    /// a title runs. Reading them lazily is what makes that ordering
    /// irrelevant: the first command to touch a setting finds whatever the
    /// last session left, and a console that has never had one written finds
    /// the defaults.
    pub(super) fn system_settings(&mut self) -> &mut SystemSettings {
        if self.system_settings.is_none() {
            let stored = self
                .save_data(SYSTEM_SETTINGS_SAVE)
                .and_then(|save| save.file(SYSTEM_SETTINGS_FILE))
                .and_then(SystemSettings::parse);
            self.system_settings = Some(stored.unwrap_or_default());
        }
        self.system_settings
            .as_mut()
            .expect("filled in immediately above")
    }

    /// Change a setting and write the block back to the save it lives in, so
    /// the host's next flush carries it out to storage.
    ///
    /// The whole block goes back rather than the one field that changed: it
    /// is a few hundred bytes, and a partial write is a file that has to be
    /// merged with what was already there before it can be read.
    pub(super) fn store_system_settings(&mut self, edit: impl FnOnce(&mut SystemSettings)) {
        edit(self.system_settings());
        let blob = self.system_settings().serialize();
        self.save_data_mut(SYSTEM_SETTINGS_SAVE)
            .guest_write_file(SYSTEM_SETTINGS_FILE, blob);
    }

    /// `set:sys` — the console's system settings.
    ///
    /// Two kinds of command live here. The `Get*`/`Set*` pairs are the
    /// service proper: they read and write [`SystemSettings`], which is
    /// stored in save data and so survives the session. The rest describe
    /// hardware this console does not have a choice about — its firmware
    /// version, its model, its serial — and answer with constants.
    ///
    /// The pairs are what this service is *for*. Before they were stored,
    /// every setter fell through to the stub: the settings applet wrote a
    /// value, was told it had worked, and read back the constant beside it.
    pub(super) fn set_sys_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return self.write_ipc_response(tls, 0, &[], &[], &[]);
        }
        // Every setter is the same three steps — read the argument, put it in
        // the block, write the block back out to the save — and answers with
        // nothing. Spelled out forty times over, the shape is what a reader
        // has to check rather than what the setter actually stores.
        macro_rules! stored {
            ($field:ident = $value:expr) => {{
                let value = $value;
                self.store_system_settings(|settings| settings.$field = value);
                return self.write_ipc_response(tls, 0, &[], &[], &[]);
            }};
        }
        match cmd_id {
            // ---- the settings themselves, setter then getter ----
            // SetLanguageCode(u64) / `set`'s GetLanguageCode reads it back.
            Some(0) => stored!(language_code = self.ipc_arg_u64(tls, 0)),
            // SetRegionCode(SystemRegionCode), likewise read back by `set`.
            Some(57) => stored!(region = self.ipc_arg_u32(tls, 0)),
            // Get/SetLockScreenFlag(bool).
            Some(7) => {
                let flag = u8::from(self.system_settings().lock_screen);
                self.write_ipc_response(tls, 0, &[], &[flag], &[])
            }
            Some(8) => stored!(lock_screen = self.ipc_arg_u8(tls, 0) != 0),
            // Get/SetExternalSteadyClockSourceId(Uuid): which clock the
            // steady clock is counting from. Nothing here reads it but its
            // own getter, which is why it has to be kept.
            Some(13) => {
                let id = self.system_settings().external_steady_clock_source_id;
                self.write_ipc_response(tls, 0, &[], &id, &[])
            }
            Some(14) => {
                stored!(external_steady_clock_source_id = self.request_block(tls))
            }
            // Get/SetUserSystemClockContext(SystemClockContext) and the
            // network one below: the offset and epoch a clock reading is
            // interpreted against. `time` keeps its own; these are the copies
            // the settings service files for whoever asks it instead.
            Some(15) => {
                let context = self.system_settings().user_clock_context;
                self.write_ipc_response(tls, 0, &[], &context, &[])
            }
            Some(16) => stored!(user_clock_context = self.request_block(tls)),
            Some(58) => {
                let context = self.system_settings().network_clock_context;
                self.write_ipc_response(tls, 0, &[], &context, &[])
            }
            Some(59) => stored!(network_clock_context = self.request_block(tls)),
            // Get/SetAccountSettings -> AccountSettings { u32 flags }.
            Some(17) => {
                let flags = self.system_settings().account_settings;
                self.write_ipc_response(tls, 0, &[], &flags.to_le_bytes(), &[])
            }
            Some(18) => stored!(account_settings = self.ipc_arg_u32(tls, 0)),
            // GetEulaVersions -> s32 count, with the agreements themselves in
            // an output buffer; SetEulaVersions replaces the list from an
            // input one. A console that has accepted none has not finished
            // first-time setup, and the Home Menu hands over to `starter` for
            // that — which nothing here can launch. The count has to be what
            // *fits*: naming an entry the caller has nowhere to read is worse
            // than reporting a short list.
            Some(21) => {
                let eula: Vec<u8> = self.system_settings().eula_versions.concat();
                let count = self.write_whole_entries(tls, &eula, EULA_VERSION_SIZE) as i32;
                self.write_ipc_response(tls, 0, &[], &count.to_le_bytes(), &[])
            }
            Some(22) => {
                let eula = self.request_list::<EULA_VERSION_SIZE>(tls);
                self.store_system_settings(|settings| settings.eula_versions = eula);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // Get/SetColorSetId -> ColorSet, the light or dark theme the Home
            // Menu draws itself in.
            Some(23) => {
                let color_set = self.system_settings().color_set;
                self.write_ipc_response(tls, 0, &[], &color_set.to_le_bytes(), &[])
            }
            Some(24) => stored!(color_set = self.ipc_arg_u32(tls, 0)),
            // Get/SetConsoleInformationUploadFlag(bool) and
            // Get/SetAutomaticApplicationDownloadFlag(bool).
            Some(25) => {
                let flag = u8::from(self.system_settings().console_information_upload);
                self.write_ipc_response(tls, 0, &[], &[flag], &[])
            }
            Some(26) => stored!(console_information_upload = self.ipc_arg_u8(tls, 0) != 0),
            Some(27) => {
                let flag = u8::from(self.system_settings().automatic_application_download);
                self.write_ipc_response(tls, 0, &[], &[flag], &[])
            }
            Some(28) => stored!(automatic_application_download = self.ipc_arg_u8(tls, 0) != 0),
            // Get/SetNotificationSettings -> NotificationSettings, 0x18
            // bytes: wider than the four padding words a reply zeroes, so
            // before this was answered `stop_time` was whatever the caller's
            // own request had left in TLS — a quiet period ending at an
            // arbitrary hour.
            Some(29) => {
                let settings = self.system_settings().notification_settings;
                self.write_ipc_response(tls, 0, &[], &settings, &[])
            }
            Some(30) => stored!(notification_settings = self.request_block(tls)),
            // Get/SetAccountNotificationSettings: a count and a buffer of
            // per-account overrides, the same shape as the EULA pair.
            Some(31) => {
                let overrides: Vec<u8> = self
                    .system_settings()
                    .account_notification_settings
                    .concat();
                let count =
                    self.write_whole_entries(tls, &overrides, ACCOUNT_NOTIFICATION_SETTINGS_SIZE)
                        as i32;
                self.write_ipc_response(tls, 0, &[], &count.to_le_bytes(), &[])
            }
            Some(32) => {
                let overrides = self.request_list::<ACCOUNT_NOTIFICATION_SETTINGS_SIZE>(tls);
                self.store_system_settings(|settings| {
                    settings.account_notification_settings = overrides
                });
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // Get/SetVibrationMasterVolume(float). `hid`'s rumble is scaled
            // by this on hardware; here it is a setting the applet's slider
            // moves and reads back.
            Some(35) => {
                let volume = self.system_settings().vibration_master_volume;
                self.write_ipc_response(tls, 0, &[], &volume.to_le_bytes(), &[])
            }
            Some(36) => stored!(vibration_master_volume = self.ipc_arg_f32(tls, 0)),
            // GetSettingsItemValueSize / GetSettingsItemValue: the firmware's
            // own key/value table, addressed by a category and a name in two
            // pointer buffers. This is not a settings *pair* — nothing writes
            // it — but it is the part of `set:sys` system components read
            // rather than the settings applet, and it is answered from a
            // table rather than stubbed because a caller that asks for an
            // item reads the size it is given and then that many bytes.
            Some(37) | Some(38) => self.set_sys_item_request(tls, cmd_id == Some(38)),
            // Get/SetTvSettings -> TvSettings, 0x20 bytes. Past the padding
            // again: `tv_gama` and `contrast_ratio` are floats, so a NaN
            // gamma used to be a reachable answer and not merely a wrong one.
            Some(39) => {
                let settings = self.system_settings().tv_settings;
                self.write_ipc_response(tls, 0, &[], &settings, &[])
            }
            Some(40) => stored!(tv_settings = self.request_block(tls)),
            // GetAudioOutputMode(AudioOutputModeTarget) /
            // SetAudioOutputMode(target, mode). Each output keeps its own
            // mode, so the target is which one is being asked about — a
            // service that answered them all alike would report the
            // headphones set to whatever was last chosen for the dock.
            Some(43) => {
                let target = self.ipc_arg_u32(tls, 0) as usize;
                let mode = self
                    .system_settings()
                    .audio_output_mode
                    .get(target)
                    .copied()
                    .unwrap_or(AUDIO_OUTPUT_STEREO);
                self.write_ipc_response(tls, 0, &[], &mode.to_le_bytes(), &[])
            }
            Some(44) => {
                let target = self.ipc_arg_u32(tls, 0) as usize;
                let mode = self.ipc_arg_u32(tls, 4);
                self.store_system_settings(|settings| {
                    if let Some(slot) = settings.audio_output_mode.get_mut(target) {
                        *slot = mode;
                    }
                });
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // Get/SetSpeakerAutoMuteFlag(bool): whether the speakers cut out
            // when something is plugged into the headphone socket.
            Some(45) => {
                let flag = u8::from(self.system_settings().speaker_auto_mute);
                self.write_ipc_response(tls, 0, &[], &[flag], &[])
            }
            Some(46) => stored!(speaker_auto_mute = self.ipc_arg_u8(tls, 0) != 0),
            // Get/SetQuestFlag -> QuestFlag, a u8. Zero is Retail; a kiosk
            // unit runs a different Home Menu entirely.
            Some(47) => {
                let flag = self.system_settings().quest_flag;
                self.write_ipc_response(tls, 0, &[], &[flag], &[])
            }
            Some(48) => stored!(quest_flag = self.ipc_arg_u8(tls, 0)),
            // Get/SetDeviceTimeZoneLocationName(LocationName): the zone the
            // console is set to. `time` reports the same name from the same
            // field, so the two services cannot disagree about where this
            // console is — though `time` still converts against UTC, having
            // no TZif database to resolve any other zone with.
            Some(53) => {
                let name = self.system_settings().device_time_zone_location_name;
                self.write_ipc_response(tls, 0, &[], &name, &[])
            }
            Some(54) => {
                stored!(device_time_zone_location_name = self.request_block(tls))
            }
            // IsUserSystemClockAutomaticCorrectionEnabled /
            // SetUserSystemClockAutomaticCorrectionEnabled(bool).
            Some(60) => {
                let flag = u8::from(self.system_settings().user_clock_automatic_correction);
                self.write_ipc_response(tls, 0, &[], &[flag], &[])
            }
            Some(61) => stored!(user_clock_automatic_correction = self.ipc_arg_u8(tls, 0) != 0),
            // GetDebugModeFlag -> bool, which real `set:sys` answers out of
            // the settings-item table rather than from a field of its own.
            // This one does the same, so the two cannot disagree.
            Some(62) => {
                let debug = settings_item("settings_debug", "is_debug_mode_enabled")
                    .and_then(|value| value.first().copied())
                    .unwrap_or(0);
                self.write_ipc_response(tls, 0, &[], &[debug], &[])
            }
            // Get/SetPrimaryAlbumStorage -> PrimaryAlbumStorage.
            Some(63) => {
                let storage = self.system_settings().primary_album_storage;
                self.write_ipc_response(tls, 0, &[], &storage.to_le_bytes(), &[])
            }
            Some(64) => stored!(primary_album_storage = self.ipc_arg_u32(tls, 0)),
            // Get/SetUsb30EnableFlag(bool).
            Some(65) => {
                let flag = u8::from(self.system_settings().usb30_enable);
                self.write_ipc_response(tls, 0, &[], &[flag], &[])
            }
            Some(66) => stored!(usb30_enable = self.ipc_arg_u8(tls, 0) != 0),
            // Get/SetNfcEnableFlag(bool) and Get/SetBluetoothEnableFlag(bool).
            // `nfc:sys` and `btm:sys` answer their own "is it on" commands out
            // of these same two fields: the switch in the settings applet and
            // the switch a service reads are one switch, and a console that
            // kept them apart is one whose radio turns itself back on.
            Some(69) => {
                let flag = u8::from(self.system_settings().nfc_enable);
                self.write_ipc_response(tls, 0, &[], &[flag], &[])
            }
            Some(70) => stored!(nfc_enable = self.ipc_arg_u8(tls, 0) != 0),
            Some(88) => {
                let flag = u8::from(self.system_settings().bluetooth_enable);
                self.write_ipc_response(tls, 0, &[], &[flag], &[])
            }
            Some(89) => stored!(bluetooth_enable = self.ipc_arg_u8(tls, 0) != 0),
            // Get/SetSleepSettings -> SleepSettings, 0xc bytes. The plans are
            // *indices* rather than durations, so the zeroes the stub left
            // said "sleep after one minute" rather than "do not sleep".
            Some(71) => {
                let settings = self.system_settings().sleep_settings;
                self.write_ipc_response(tls, 0, &[], &settings, &[])
            }
            Some(72) => stored!(sleep_settings = self.request_block(tls)),
            // Get/SetWirelessLanEnableFlag(bool).
            Some(73) => {
                let flag = u8::from(self.system_settings().wireless_lan_enable);
                self.write_ipc_response(tls, 0, &[], &[flag], &[])
            }
            Some(74) => stored!(wireless_lan_enable = self.ipc_arg_u8(tls, 0) != 0),
            // Get/SetInitialLaunchSettings -> InitialLaunchSettings. The
            // flags say whether the console has been through first-time
            // setup, and the Home Menu will not draw a menu for one that has
            // not — it waits to hand over to `starter` instead.
            Some(75) => {
                let settings = self.system_settings().initial_launch_settings;
                self.write_ipc_response(tls, 0, &[], &settings, &[])
            }
            Some(76) => stored!(initial_launch_settings = self.request_block(tls)),
            // Get/SetDeviceNickName: 0x80 bytes through a buffer either way.
            // This is the name the console calls itself on a local network
            // and in the settings applet's own title bar.
            Some(77) => {
                let name = self.system_settings().device_nick_name;
                self.write_output_buffer(tls, 0, &name);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(78) => {
                let name = self.input_block::<DEVICE_NICK_NAME_SIZE>(tls);
                self.store_system_settings(|settings| settings.device_nick_name = name);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // Get/SetAutoUpdateEnableFlag(bool) and
            // Get/SetBatteryPercentageFlag(bool).
            Some(95) => {
                let flag = u8::from(self.system_settings().auto_update_enable);
                self.write_ipc_response(tls, 0, &[], &[flag], &[])
            }
            Some(96) => stored!(auto_update_enable = self.ipc_arg_u8(tls, 0) != 0),
            Some(99) => {
                let flag = u8::from(self.system_settings().battery_percentage);
                self.write_ipc_response(tls, 0, &[], &[flag], &[])
            }
            Some(100) => stored!(battery_percentage = self.ipc_arg_u8(tls, 0) != 0),
            // SetExternalSteadyClockInternalOffset(s64) / Get. Note which way
            // round these two are: the setter has the lower id.
            Some(105) => {
                stored!(external_steady_clock_internal_offset = self.ipc_arg_u64(tls, 0) as i64)
            }
            Some(106) => {
                let offset = self.system_settings().external_steady_clock_internal_offset;
                self.write_ipc_response(tls, 0, &[], &offset.to_le_bytes(), &[])
            }
            // Get/SetPushNotificationActivityModeOnSleep(s32).
            Some(120) => {
                let mode = self
                    .system_settings()
                    .push_notification_activity_mode_on_sleep;
                self.write_ipc_response(tls, 0, &[], &mode.to_le_bytes(), &[])
            }
            Some(121) => {
                stored!(push_notification_activity_mode_on_sleep = self.ipc_arg_u32(tls, 0) as i32)
            }
            // Get/SetErrorReportSharePermission -> ErrorReportSharePermission.
            // Zero is NotConfirmed, which is the truth: nothing has asked.
            Some(124) => {
                let permission = self.system_settings().error_report_share_permission;
                self.write_ipc_response(tls, 0, &[], &permission.to_le_bytes(), &[])
            }
            Some(125) => stored!(error_report_share_permission = self.ipc_arg_u32(tls, 0)),
            // Get/SetAppletLaunchFlags(u32).
            Some(126) => {
                let flags = self.system_settings().applet_launch_flags;
                self.write_ipc_response(tls, 0, &[], &flags.to_le_bytes(), &[])
            }
            Some(127) => stored!(applet_launch_flags = self.ipc_arg_u32(tls, 0)),
            // Get/SetKeyboardLayout -> KeyboardLayout. Zero is `Japanese`, a
            // real layout but not this console's, and the software keyboard
            // reads this to lay out its keys.
            Some(136) => {
                let layout = self.system_settings().keyboard_layout;
                self.write_ipc_response(tls, 0, &[], &layout.to_le_bytes(), &[])
            }
            Some(137) => stored!(keyboard_layout = self.ipc_arg_u32(tls, 0)),
            // Get/SetDeviceTimeZoneLocationUpdatedTime and the same pair for
            // the clock's automatic correction: a SteadyClockTimePoint each,
            // saying when the setting beside it last moved.
            Some(150) => {
                let when = self.system_settings().device_time_zone_updated_time;
                self.write_ipc_response(tls, 0, &[], &when, &[])
            }
            Some(151) => stored!(device_time_zone_updated_time = self.request_block(tls)),
            Some(152) => {
                let when = self.system_settings().user_clock_correction_updated_time;
                self.write_ipc_response(tls, 0, &[], &when, &[])
            }
            Some(153) => {
                stored!(user_clock_correction_updated_time = self.request_block(tls))
            }
            // Get/SetChineseTraditionalInputMethod -> its own enum.
            Some(170) => {
                let method = self.system_settings().chinese_traditional_input_method;
                self.write_ipc_response(tls, 0, &[], &method.to_le_bytes(), &[])
            }
            Some(171) => stored!(chinese_traditional_input_method = self.ipc_arg_u32(tls, 0)),
            // Get/SetPlatformRegion -> s32 PlatformRegion, which is Global
            // (1) or Terra (2) — the Chinese console — and has no zero. So
            // the generic empty-success reply left the caller reading a value
            // that is not a member of the enum, and `nn::settings` aborts on
            // that: the error applet took an svcBreak with no message here,
            // one command into its own start.
            Some(183) => {
                let region = self.system_settings().platform_region;
                self.write_ipc_response(tls, 0, &[], &region.to_le_bytes(), &[])
            }
            Some(184) => stored!(platform_region = self.ipc_arg_u32(tls, 0) as i32),
            // Get/SetTouchScreenMode -> TouchScreenMode. Standard, not the
            // Stylus its zero means.
            Some(187) => {
                let mode = self.system_settings().touch_screen_mode;
                self.write_ipc_response(tls, 0, &[], &mode.to_le_bytes(), &[])
            }
            Some(188) => stored!(touch_screen_mode = self.ipc_arg_u32(tls, 0)),
            // Get/SetFieldTestingFlag(bool).
            Some(201) => {
                let flag = u8::from(self.system_settings().field_testing);
                self.write_ipc_response(tls, 0, &[], &[flag], &[])
            }
            Some(202) => stored!(field_testing = self.ipc_arg_u8(tls, 0) != 0),
            // Get/SetPanelCrcMode(s32).
            Some(203) => {
                let mode = self.system_settings().panel_crc_mode;
                self.write_ipc_response(tls, 0, &[], &mode.to_le_bytes(), &[])
            }
            Some(204) => stored!(panel_crc_mode = self.ipc_arg_u32(tls, 0) as i32),

            // ---- what this console has no choice about ----
            // GetFirmwareVersion / GetFirmwareVersion2 -> a
            // `SetSysFirmwareVersion` in an output buffer.
            //
            // This is not cosmetic. libnx's `__appInit` seeds `hosversionGet`
            // from it, and everything version-gated downstream branches on
            // that: which `acc` commands exist, which `ts` interface carries
            // the temperature, which audio-renderer revision is negotiated.
            // The generic empty-success answer left the caller reading its own
            // uninitialized buffer as the version — NX-Fetch reported "Horizon
            // OS 115.119.105", which is the ASCII of `switch-wasm user`, the
            // uid this emulator had left in that same buffer earlier.
            Some(3) | Some(4) => {
                let version = Self::firmware_version();
                if let Some((addr, size)) = self.ipc_output_buffer(tls, 0) {
                    if addr != 0 {
                        for (index, &byte) in version.iter().take(size as usize).enumerate() {
                            self.mem.write_u8(addr.wrapping_add(index as u32), byte)?;
                        }
                    }
                }
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // GetBatteryLot -> SetSysBatteryLot { char lot[0x18] } and
            // GetSerialNumber -> SetSysSerialNumber { char number[0x18] }.
            // Both are burned in at manufacturing and unique per console;
            // these are fixed placeholders, not real numbers.
            Some(67) | Some(68) => {
                const BATTERY_LOT: &[u8] = b"0000000000000000";
                const SERIAL: &[u8] = b"XAW00000000000";
                let text = if cmd_id == Some(67) {
                    BATTERY_LOT
                } else {
                    SERIAL
                };
                let mut raw = [0u8; 0x18];
                raw[..text.len()].copy_from_slice(text);
                self.write_ipc_response(tls, 0, &[], &raw, &[])
            }
            // GetProductModel -> u32 ProductModel, which starts at 1 (Nx).
            // Zero is not a model, so the generic empty-success reply sat
            // outside the enum the same way GetPlatformRegion's did.
            Some(79) => self.write_ipc_response(tls, 0, &[], &1u32.to_le_bytes(), &[]),
            // GetMiiAuthorId -> the Uuid every Mii made on this console is
            // stamped with. It has to be the same one every session or a Mii
            // made yesterday is not this console's today, so it is fixed
            // rather than generated — and it is stored with the settings for
            // exactly the reason the settings are stored.
            Some(90) => {
                let id = MII_AUTHOR_ID;
                self.write_ipc_response(tls, 0, &[], &id, &[])
            }
            // GetRebootlessSystemUpdateVersion -> { u32 version;
            // reserved[0x1c]; char display_version[0x20]; }. No update has
            // been applied over the running firmware, which is version zero
            // and an empty display string.
            Some(149) => self.write_ipc_response(tls, 0, &[], &[0u8; 0x40], &[]),
            // GetHomeMenuScheme -> HomeMenuScheme, the five colours the Home
            // Menu tints itself with, and GetHomeMenuSchemeModel -> u32,
            // which scheme a console of this model uses. Zero for the model
            // is the standard one; the colours are a plausible scheme rather
            // than a measured one — see [`HOME_MENU_SCHEME`].
            Some(174) => {
                let mut scheme = Vec::with_capacity(0x14);
                for color in HOME_MENU_SCHEME {
                    scheme.extend_from_slice(&color.to_le_bytes());
                }
                self.write_ipc_response(tls, 0, &[], &scheme, &[])
            }
            Some(185) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            _ => {
                self.warn_stub(
                    "set:sys",
                    cmd_id,
                    "an empty success, so the caller reads its own buffer as the answer",
                );
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
        }
    }

    /// `GetSettingsItemValueSize` (37) and `GetSettingsItemValue` (38): the
    /// firmware's key/value table, addressed by a category and a name that
    /// arrive as two separate input buffers.
    ///
    /// An item this console does not have is refused rather than answered
    /// with a zero: a caller reads the size back and then that many bytes, so
    /// a fabricated success hands it a value it never stored. The refusal
    /// names what was asked for, which is the only way to find out what a
    /// title wanted.
    fn set_sys_item_request(&mut self, tls: u32, with_value: bool) -> Result<()> {
        /// `nn::settings::ResultSettingsItemNotFound` — module 105,
        /// description 11, as Atmosphère's `settings_results.hpp` names it.
        const SETTINGS_ITEM_NOT_FOUND: u32 = 105 | (11 << 9);
        /// `nn::settings::SettingItemName`, the width of each name buffer.
        const NAME_SIZE: u32 = 0x48;

        let name_at = |cpu: &Cpu, index: u32| -> String {
            match cpu.ipc_input_buffer(tls, index) {
                Some((addr, size)) if addr != 0 => cpu.read_string(addr, size.min(NAME_SIZE)),
                _ => String::new(),
            }
        };
        let category = name_at(self, 0);
        let name = name_at(self, 1);

        let Some(value) = settings_item(&category, &name) else {
            self.warn_missing_settings_item(&category, &name);
            return self.write_ipc_response(tls, SETTINGS_ITEM_NOT_FOUND, &[], &[], &[]);
        };
        // The size reported is the item's own, not how much of it fit: a
        // caller sizes its buffer from command 37 and would read the
        // difference as a short item rather than as a buffer it undersized.
        let size = value.len() as u64;
        if with_value {
            self.write_output_buffer(tls, 0, &value);
        }
        self.write_ipc_response(tls, 0, &[], &size.to_le_bytes(), &[])
    }

    /// Say once which settings item was asked for and not found. Once per
    /// item rather than per call: `nnSdk` retries, and the interesting part
    /// is the name.
    fn warn_missing_settings_item(&mut self, category: &str, name: &str) {
        if self
            .missing_settings_items
            .insert(format!("{category}!{name}"))
        {
            self.diagnostic(
                Level::Warn,
                &format!("[set:sys] no settings item {category}!{name}"),
            );
        }
    }

    /// Write as many whole `size`-byte entries as the request's first output
    /// buffer has room for, and say how many that was.
    ///
    /// A partial entry is not one. The caller reads the count and stops
    /// there, so the bytes past it are bytes it never looks at — and it sized
    /// that buffer itself, so a list that does not fit is a list it asked for
    /// less of.
    fn write_whole_entries(&mut self, tls: u32, entries: &[u8], size: usize) -> usize {
        let room = self.ipc_output_buffer(tls, 0).map_or(0, |(addr, len)| {
            if addr == 0 {
                0
            } else {
                len as usize / size
            }
        });
        let count = room.min(entries.len() / size);
        self.write_output_buffer(tls, 0, &entries[..count * size]);
        count
    }

    /// A fixed-width struct out of a request's raw data — the shape every
    /// `Set*` that takes a settings block arrives in.
    fn request_block<const N: usize>(&self, tls: u32) -> [u8; N] {
        let mut block = [0u8; N];
        let data = self.ipc_request_data(tls);
        for (offset, byte) in block.iter_mut().enumerate() {
            *byte = self
                .mem
                .read_u8(data.wrapping_add(offset as u32))
                .unwrap_or(0);
        }
        block
    }

    /// A fixed-width struct out of a request's first input buffer, for the
    /// setters whose argument is too wide to travel in the raw data.
    fn input_block<const N: usize>(&self, tls: u32) -> [u8; N] {
        let mut block = [0u8; N];
        if let Some((addr, size)) = self.ipc_input_buffer(tls, 0) {
            if addr != 0 {
                let bytes = self.read_bytes(addr, size.min(N as u32));
                block[..bytes.len()].copy_from_slice(&bytes);
            }
        }
        block
    }

    /// The list of fixed-width entries in a request's first input buffer —
    /// what `SetEulaVersions` and `SetAccountNotificationSettings` replace
    /// their whole list from. A trailing partial entry is not one.
    fn request_list<const N: usize>(&self, tls: u32) -> Vec<[u8; N]> {
        let Some((addr, size)) = self.ipc_input_buffer(tls, 0) else {
            return Vec::new();
        };
        if addr == 0 {
            return Vec::new();
        }
        self.read_bytes(addr, size)
            .chunks_exact(N)
            .filter_map(|entry| entry.try_into().ok())
            .collect()
    }

    /// `SetSysFirmwareVersion`, the 0x100-byte block `set:sys` reports the
    /// system version in: the numeric version, then the platform, the build
    /// hash, and the two display strings the settings applet shows.
    fn firmware_version() -> [u8; 0x100] {
        let mut version = [0u8; 0x100];
        version[0] = FIRMWARE_VERSION.0;
        version[1] = FIRMWARE_VERSION.1;
        version[2] = FIRMWARE_VERSION.2;
        version[4] = 1; // revision_major
        let mut write = |offset: usize, text: &str, room: usize| {
            let bytes = text.as_bytes();
            let len = bytes.len().min(room - 1);
            version[offset..offset + len].copy_from_slice(&bytes[..len]);
        };
        write(0x08, "NX", 0x20);
        write(0x28, "switch-wasm", 0x40);
        let display = format!(
            "{}.{}.{}",
            FIRMWARE_VERSION.0, FIRMWARE_VERSION.1, FIRMWARE_VERSION.2
        );
        write(0x68, &display, 0x18);
        write(
            0x80,
            &format!("NintendoSDK Firmware for NX {display}-1.0"),
            0x80,
        );
        version
    }

    /// `pctl` and its aliases (`pctl:s`, `pctl:a`, `pctl:r`): parental
    /// controls, reported as **switched off**.
    ///
    /// There is nobody to restrict here — no accounts, no PIN, no play timer,
    /// no linked guardian — so "off" is not a placeholder, it is the true
    /// state of this console. That makes every answer determinate: a
    /// permission check succeeds (a real denial is an error `Result`, not a
    /// `false`), an "is this restricted" query is `false`, and an "is this
    /// still allowed" query is `true`. Note which way round those go — the two
    /// families read in opposite directions, and a blanket `false` would have
    /// reported free communication as *unavailable*.
    ///
    /// A retail title asks for this early: "A Short Hike" opens all four
    /// aliases before it touches the filesystem, and `nnSdk` will not start an
    /// application it believes is restricted.
    pub(super) fn pctl_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "pctl:factory");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, "pctl:control", cmd_id),
            };
        }
        let object_id = self.ipc_domain_object_id(tls);
        let iface = if self.ipc_is_domain_request(tls) {
            self.domain_interface(handle, object_id)
                .unwrap_or("pctl:factory")
                .to_string()
        } else {
            match self.service_name(handle) {
                // The root session is IParentalControlServiceFactory itself.
                Some("pctl") | Some("pctl:s") | Some("pctl:a") | Some("pctl:r") | None => {
                    "pctl:factory".to_string()
                }
                Some(name) => name.to_string(),
            }
        };
        match iface.as_str() {
            // IParentalControlServiceFactory::CreateService /
            // CreateServiceWithoutInitialize. The difference is whether the
            // returned interface arrives already initialized; with no settings
            // to load, both hand back the same thing.
            "pctl:factory" => match cmd_id {
                Some(0) | Some(1) => {
                    self.reply_with_interface(tls, handle, "pctl:service")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "pctl:service" => match cmd_id {
                // Initialize.
                Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // The permission checks: CheckFreeCommunicationPermission,
                // ConfirmLaunchApplicationPermission,
                // ConfirmResumeApplicationPermission,
                // ConfirmSnsPostPermission,
                // ConfirmSystemSettingsPermission,
                // ConfirmStereoVisionPermission, ConfirmShowNewsPermission,
                // EndFreeCommunication,
                // ResetConfirmedStereoVisionPermission.
                //
                // These answer with a bare `Result`: success *is* "permitted",
                // and a restriction shows up as an error the caller checks for
                // by value. Nothing is restricted, so they all succeed.
                Some(1001..=1005) | Some(1013) | Some(1016) | Some(1017) | Some(1064) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // IsRestrictionTemporaryUnlocked /
                // IsRestrictedSystemSettingsEntered / IsRestrictionEnabled /
                // IsPlayTimerEnabled / IsRestrictedByPlayTimer: "is something
                // restricting you" — all false.
                Some(1006) | Some(1010) | Some(1031) | Some(1453) | Some(1455) => {
                    self.write_ipc_response(tls, 0, &[], &0u8.to_le_bytes(), &[])
                }
                // IsFreeCommunicationAvailable / IsStereoVisionPermitted: "is
                // something still allowed" — the opposite sense, so both true.
                Some(1018) | Some(1065) => {
                    self.write_ipc_response(tls, 0, &[], &1u8.to_le_bytes(), &[])
                }
                // IsPairingActive / IsPlayTimerAlarmDisabled: no guardian is
                // paired and there is no timer to sound an alarm. The second
                // reads the other way round again — "disabled" is the
                // unrestricted answer, so it is true where the first is false.
                Some(1403) => self.write_ipc_response(tls, 0, &[], &0u8.to_le_bytes(), &[]),
                Some(1458) => self.write_ipc_response(tls, 0, &[], &1u8.to_le_bytes(), &[]),
                // GetRestrictedFeatures / GetSafetyLevel /
                // GetFreeCommunicationApplicationListCount / GetPinCodeLength
                // / GetAccountState / GetPostEventInterval: nothing set, no
                // list, no PIN, no linked account — zero in every one of them.
                Some(1012) | Some(1032) | Some(1039) | Some(1206) | Some(1424) | Some(1426) => {
                    self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[])
                }
                // GetCurrentSettings -> nn::pctl::RestrictionSettings: a
                // rating age, and the two "may not post / may not talk" flags.
                // Nothing is restricted here, and an age of -1 is how that is
                // said: any rating passes it.
                Some(1035) => self.write_ipc_response(tls, 0, &[], &[0xffu8, 0, 0], &[]),
                // GenerateInquiryCode -> char[0x20], the "%02d%08llu" of 11
                // and eight digits a guardian reads out to have a forgotten
                // PIN reset. No PIN is set here, so the digits are a fixed
                // placeholder; the width and the format are not.
                Some(1204) => {
                    const INQUIRY_CODE: &[u8] = b"1100000000";
                    let mut code = [0u8; 0x20];
                    code[..INQUIRY_CODE.len()].copy_from_slice(INQUIRY_CODE);
                    self.write_ipc_response(tls, 0, &[], &code, &[])
                }
                // The event getters: GetPinCodeChangedEvent (1207),
                // GetSynchronizationEvent (1432),
                // GetPlayTimerEventToRequestSuspension (1457) and
                // GetUnlinkedEvent (1473).
                //
                // Each is a **copy** handle, and each stays unsignalled for
                // the life of the process — the PIN never changes, there is no
                // guardian account to synchronise with, no timer to ask for a
                // suspension and no link to break. A caller waits on all four
                // forever, which is the correct thing for it to do. Refusing
                // instead is what took the Home Menu down: `nnSdk` aborts on
                // an unknown command id rather than carry on without the
                // handle.
                Some(1207) | Some(1432) | Some(1457) | Some(1473) => {
                    let name = match cmd_id {
                        Some(1207) => "pctl:pin-changed",
                        Some(1432) => "pctl:synchronization",
                        Some(1457) => "pctl:play-timer-suspend",
                        _ => "pctl:unlinked",
                    };
                    let h = self.alloc_event(name, true);
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                // GetPlayTimerRemainingTime -> s32. There is no timer running
                // (1453 says so), and a timer that is not running has no
                // deadline: zero here would read as *time is up*, which is the
                // restricted answer 1455 already denies.
                Some(1454) => self.write_ipc_response(tls, 0, &[], &i32::MAX.to_le_bytes(), &[]),
                // GetPlayTimerRemainingTimeDisplayInfo -> 0x18 bytes, whose
                // fields nobody has named: Eden's `parental_control_service.cpp`
                // records the width and writes none of it. Zeroed, like the
                // settings block below, rather than guessed at field by field.
                Some(1459) => self.write_ipc_response(tls, 0, &[], &[0u8; 0x18], &[]),
                // GetPlayTimerSettings: an unset settings block. Zeroed and
                // sized past `nn::pctl::PlayTimerSettings` so that a wider
                // struct still reads as unset rather than as reply padding —
                // a reply may be longer than the caller needs, never shorter.
                Some(1456) => self.write_ipc_response(tls, 0, &[], &[0u8; 0x40], &[]),
                // The 18.0.0+ id for the same thing, which turned 1456 into
                // the `Old` form. Its block widened to 0x44 bytes in 21.0.0,
                // and that is the width answered here for the reason above.
                Some(145601) => self.write_ipc_response(tls, 0, &[], &[0u8; 0x44], &[]),
                // StartPlayTimer / StopPlayTimer / RequestPostEvents /
                // ClearUnlinkedEvent / DisableFeaturesForReset /
                // NotifyApplicationDownloadStarted /
                // NotifyNetworkProfileCreated: void, and there is no state
                // here for any of them to change.
                Some(1046..=1048) | Some(1425) | Some(1451) | Some(1452) | Some(1474) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }

    /// `lbl` — "nn::lbl::detail::ILblController", the panel backlight.
    ///
    /// This is one interface with no sub-objects, and almost all of it is a
    /// setter/getter pair over [`Backlight`]. That is the whole reason it
    /// needs an implementation at all: the generic fallback answered
    /// `LoadCurrentSetting` with a fabricated object id, and every getter
    /// beside it with a value that had nothing to do with what the matching
    /// setter had just been told.
    ///
    /// The one thing this console genuinely does not have is the ambient
    /// light sensor, so `IsAmbientLightSensorAvailable` and
    /// `IsAutoBrightnessControlSupported` say no — and a caller that believes
    /// them never turns auto-brightness on, which is the state the rest of
    /// the answers here describe.
    pub(super) fn lbl_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        /// `LblBacklightSwitchStatus`.
        const BACKLIGHT_DISABLED: u32 = 0;
        const BACKLIGHT_ENABLED: u32 = 1;
        if self.ipc_answer_control(tls, handle, "lbl", cmd_id)? {
            return Ok(());
        }
        match cmd_id {
            // SaveCurrentSetting / LoadCurrentSetting: the applet stashes the
            // brightness before it changes it for a preview, and puts it back
            // when the user backs out.
            Some(0) => {
                self.backlight.saved = self.backlight.setting;
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(1) => {
                self.backlight.setting = self.backlight.saved;
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // SetCurrentBrightnessSetting(float) / GetCurrentBrightnessSetting.
            Some(2) => {
                self.backlight.setting = self.ipc_arg_f32(tls, 0);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(3) => {
                let value = self.backlight.setting;
                self.write_ipc_response(tls, 0, &[], &value.to_bits().to_le_bytes(), &[])
            }
            // ApplyCurrentBrightnessSettingToBacklight: there is no panel to
            // apply it to, and the setting is already what
            // GetBrightnessSettingAppliedToBacklight reports.
            Some(4) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // GetBrightnessSettingAppliedToBacklight -> what the backlight is
            // actually running at, which is the setting while the panel is on
            // and nothing at all while it is off.
            Some(5) => {
                let applied = if self.backlight.on {
                    self.backlight.setting
                } else {
                    0.0
                };
                self.write_ipc_response(tls, 0, &[], &applied.to_bits().to_le_bytes(), &[])
            }
            // SwitchBacklightOn / SwitchBacklightOff, each taking the fade
            // time to get there. Nothing fades, so the switch is immediate.
            Some(6) | Some(7) => {
                self.backlight.on = cmd_id == Some(6);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(8) => {
                let status = if self.backlight.on {
                    BACKLIGHT_ENABLED
                } else {
                    BACKLIGHT_DISABLED
                };
                self.write_ipc_response(tls, 0, &[], &status.to_le_bytes(), &[])
            }
            // EnableDimming / DisableDimming / IsDimmingEnabled.
            Some(9) | Some(10) => {
                self.backlight.dimming = cmd_id == Some(9);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(11) => {
                let enabled = u8::from(self.backlight.dimming);
                self.write_ipc_response(tls, 0, &[], &[enabled], &[])
            }
            // EnableAutoBrightnessControl / Disable / IsEnabled.
            Some(12) | Some(13) => {
                self.backlight.auto_brightness = cmd_id == Some(12);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(14) => {
                let enabled = u8::from(self.backlight.auto_brightness);
                self.write_ipc_response(tls, 0, &[], &[enabled], &[])
            }
            // SetAmbientLightSensorValue(float lux). There is no sensor, so
            // the only lux this console ever sees is the one a debug caller
            // injects here.
            Some(15) => {
                self.backlight.lux = self.ipc_arg_f32(tls, 0);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // GetAmbientLightSensorValue -> { u32 over_limit, float lux }.
            // Over-limit means the reading saturated, which an absent sensor
            // never does.
            Some(16) => {
                let mut raw = Vec::with_capacity(8);
                raw.extend_from_slice(&0u32.to_le_bytes());
                raw.extend_from_slice(&self.backlight.lux.to_bits().to_le_bytes());
                self.write_ipc_response(tls, 0, &[], &raw, &[])
            }
            // SetBrightnessReflectionDelayLevel(float, float) /
            // GetBrightnessReflectionDelayLevel(float) -> float: how long the
            // panel takes to follow a change. The getter takes a float of its
            // own that selects which level it is asking about; there is one
            // level here, so it is ignored.
            Some(17) => {
                self.backlight.reflection_delay = self.ipc_arg_f32(tls, 0);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(18) => {
                let level = self.backlight.reflection_delay;
                self.write_ipc_response(tls, 0, &[], &level.to_bits().to_le_bytes(), &[])
            }
            // SetCurrentBrightnessMapping(float, float, float) / Get, and the
            // same pair for the ambient light sensor's mapping. These are the
            // curve the firmware drives the panel through; nothing here reads
            // them but their own getters.
            Some(19) | Some(21) => {
                let mut mapping = [0.0f32; 3];
                for (index, value) in mapping.iter_mut().enumerate() {
                    *value = self.ipc_arg_f32(tls, 4 * index as u32);
                }
                if cmd_id == Some(19) {
                    self.backlight.brightness_mapping = mapping;
                } else {
                    self.backlight.lux_mapping = mapping;
                }
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(20) | Some(22) => {
                let mapping = if cmd_id == Some(20) {
                    self.backlight.brightness_mapping
                } else {
                    self.backlight.lux_mapping
                };
                let mut raw = Vec::with_capacity(12);
                for value in mapping {
                    raw.extend_from_slice(&value.to_bits().to_le_bytes());
                }
                self.write_ipc_response(tls, 0, &[], &raw, &[])
            }
            // IsAmbientLightSensorAvailable, and the 7.0.0+
            // IsAutoBrightnessControlSupported that follows from it: no
            // sensor, so neither.
            Some(23) | Some(29) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
            // SetCurrentBrightnessSettingForVrMode(float) / Get.
            Some(24) => {
                self.backlight.vr_setting = self.ipc_arg_f32(tls, 0);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(25) => {
                let value = self.backlight.vr_setting;
                self.write_ipc_response(tls, 0, &[], &value.to_bits().to_le_bytes(), &[])
            }
            // EnableVrMode / DisableVrMode / IsVrModeEnabled. `am`'s
            // SetVrModeEnabled is the caller.
            Some(26) | Some(27) => {
                self.backlight.vr_mode = cmd_id == Some(26);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(28) => {
                let enabled = u8::from(self.backlight.vr_mode);
                self.write_ipc_response(tls, 0, &[], &[enabled], &[])
            }
            _ => self.unimplemented_command(tls, "lbl", cmd_id),
        }
    }

    /// `notif:s` / `notif:a` — "nn::notification::server::INotificationServices"
    /// and the application-facing interface beside it: the alarms a title
    /// asks the system to wake it for, and the notifications the Home Menu
    /// shows.
    ///
    /// The alarm store is real. A caller registers an `AlarmSetting`, is
    /// given the id the system filed it under, and lists, reloads and deletes
    /// it by that id — a round trip a fabricated success cannot fake, because
    /// the id it hands back names nothing and the list that follows disagrees
    /// with it. What is *not* modelled is an alarm ever firing: they are
    /// scheduled against a clock the console keeps while it sleeps, and this
    /// console does not sleep.
    pub(super) fn notif_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        let root = if self.service_name(handle) == Some("notif:a") {
            "notif:a"
        } else {
            "notif:s"
        };
        if self.ipc_answer_control(tls, handle, root, cmd_id)? {
            return Ok(());
        }
        let iface = self.ipc_interface(tls, handle, root);
        // INotificationSystemEventAccessor: GetSystemEvent, the one command
        // it has. The event fires when a notification is posted.
        if iface == "notif:event-accessor" {
            return match cmd_id {
                Some(0) => {
                    let event = self.kept_event("notif:system", handle);
                    self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            };
        }
        match cmd_id {
            // RegisterAlarmSetting(AlarmSetting, ApplicationParameter) -> the
            // id the system filed it under. The id is the server's to assign,
            // and it is written back into the stored copy so the listing
            // agrees with what the caller was told.
            Some(500) => {
                let setting = self
                    .ipc_input_buffer(tls, 0)
                    .map(|(addr, size)| self.read_bytes(addr, size.min(ALARM_SETTING_SIZE as u32)));
                let mut setting = setting.unwrap_or_default();
                setting.resize(ALARM_SETTING_SIZE, 0);
                let parameter = self
                    .ipc_input_buffer(tls, 1)
                    .map(|(addr, size)| self.read_bytes(addr, size.min(ALARM_PARAMETER_MAX)))
                    .unwrap_or_default();
                let id = self.notif_next_alarm_id;
                self.notif_next_alarm_id = self.notif_next_alarm_id.wrapping_add(1);
                setting[ALARM_SETTING_ID..ALARM_SETTING_ID + 2].copy_from_slice(&id.to_le_bytes());
                self.notif_alarms.push(AlarmSetting {
                    id,
                    setting,
                    parameter,
                });
                self.write_ipc_response(tls, 0, &[], &id.to_le_bytes(), &[])
            }
            // UpdateAlarmSetting(AlarmSetting, ApplicationParameter): the
            // setting carries the id of the alarm it replaces.
            Some(510) => {
                let setting = self
                    .ipc_input_buffer(tls, 0)
                    .map(|(addr, size)| self.read_bytes(addr, size.min(ALARM_SETTING_SIZE as u32)))
                    .unwrap_or_default();
                let parameter = self
                    .ipc_input_buffer(tls, 1)
                    .map(|(addr, size)| self.read_bytes(addr, size.min(ALARM_PARAMETER_MAX)))
                    .unwrap_or_default();
                let id = u16::from_le_bytes([
                    setting.first().copied().unwrap_or(0),
                    setting.get(1).copied().unwrap_or(0),
                ]);
                if let Some(alarm) = self.notif_alarms.iter_mut().find(|alarm| alarm.id == id) {
                    alarm.setting = setting;
                    alarm.setting.resize(ALARM_SETTING_SIZE, 0);
                    alarm.parameter = parameter;
                }
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // ListAlarmSettings -> the settings themselves in an output
            // buffer, and how many were written.
            Some(520) => {
                let mut entries = Vec::new();
                for alarm in &self.notif_alarms {
                    entries.extend_from_slice(&alarm.setting);
                }
                let written = self.write_output_buffer(tls, 0, &entries);
                let count = (written as usize / ALARM_SETTING_SIZE) as i32;
                self.write_ipc_response(tls, 0, &[], &count.to_le_bytes(), &[])
            }
            // LoadApplicationParameter(AlarmSettingId) -> the blob the title
            // attached, and its real length. A caller reads the length, not
            // the buffer size, so this has to be the *stored* size.
            Some(530) => {
                let id = self.ipc_arg_u32(tls, 0) as u16;
                let parameter = self
                    .notif_alarms
                    .iter()
                    .find(|alarm| alarm.id == id)
                    .map(|alarm| alarm.parameter.clone())
                    .unwrap_or_default();
                let written = self.write_output_buffer(tls, 0, &parameter);
                self.write_ipc_response(tls, 0, &[], &written.to_le_bytes(), &[])
            }
            // DeleteAlarmSetting(AlarmSettingId).
            Some(540) => {
                let id = self.ipc_arg_u32(tls, 0) as u16;
                self.notif_alarms.retain(|alarm| alarm.id != id);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // 1000 is two different commands: `notif:a`'s Initialize, which
            // takes a pid and answers with nothing, and `notif:s`'s
            // GetNotificationCount, which answers with a number. Answering
            // the wrong one hands a void command a count, or a count nothing.
            Some(1000) if root == "notif:a" => self.write_ipc_response(tls, 0, &[], &[], &[]),
            Some(1000) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            // ListNotifications -> entries into a buffer, and how many.
            Some(1010) => self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[]),
            // DeleteNotification / ClearNotifications: nothing is queued.
            Some(1020) | Some(1030) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // GetNotificationSendingNotifier ->
            // INotificationSystemEventAccessor, which holds the event that
            // fires when a notification is posted.
            Some(1040) => {
                self.reply_with_interface(tls, handle, "notif:event-accessor")?;
                Ok(())
            }
            // SetNotificationPresentationSetting /
            // GetNotificationPresentationSetting(NotificationChannel) -> a
            // 0x10-byte setting. All zero is "present it the default way".
            Some(1500) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            Some(1510) => self.write_ipc_response(tls, 0, &[], &[0u8; 0x10], &[]),
            // GetAlarmSetting(AlarmSettingId) -> the 0x40-byte setting.
            Some(2000) => {
                let id = self.ipc_arg_u32(tls, 0) as u16;
                let setting = self
                    .notif_alarms
                    .iter()
                    .find(|alarm| alarm.id == id)
                    .map(|alarm| alarm.setting.clone())
                    .unwrap_or_else(|| vec![0; ALARM_SETTING_SIZE]);
                self.write_ipc_response(tls, 0, &[], &setting, &[])
            }
            // SetAlarmSettingIsMuted(AlarmSettingId, bool).
            Some(2010) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // IsAlarmSettingDeletable(AlarmSettingId) -> bool. Every alarm
            // registered here belongs to the caller and can go.
            Some(2020) => self.write_ipc_response(tls, 0, &[], &[1u8], &[]),
            // RegisterAppletResourceUserId / UnregisterAppletResourceUserId:
            // which applet an alarm belongs to. There is one process here.
            Some(8000) | Some(8010) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // GetCurrentTime -> the PosixTime the alarms are scheduled
            // against, which has to be the same clock `time` reports or an
            // alarm set for "an hour from now" lands in the wrong century.
            Some(8999) => {
                let now = self.unix_time();
                self.write_ipc_response(tls, 0, &[], &now.to_le_bytes(), &[])
            }
            // GetAlarmSettingNextNotificationTime(AlarmSettingId) -> whether
            // the alarm is scheduled, and when. Nothing here schedules one.
            Some(9000) => self.write_ipc_response(tls, 0, &[], &[0u8; 0x10], &[]),
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Cpu;
    use crate::cpu::ipc::testing::*;

    #[test]
    fn set_sys_reports_a_platform_region_that_is_in_the_enum() {
        // `nn::settings::PlatformRegion` is Global (1) or Terra (2) -- the
        // Chinese console -- and has no zero. The generic empty-success reply
        // left the caller reading a value that is in neither, and
        // `nn::settings` aborts on that with no message: it is where the
        // error applet took an svcBreak, one command into its own start.
        let mut cpu = request(false, 183, &[]);
        cpu.set_sys_request(TLS, Some(183)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        let region = cpu.mem.read_u32(TLS + 0x20).unwrap();
        assert_eq!(region, 1, "this console is the Global one");
    }

    #[test]
    fn set_sys_reports_an_accepted_eula() {
        // A console that has accepted no agreement has not finished
        // first-time setup, and the Home Menu hands over to `starter` for
        // that -- which nothing here can launch.
        const BUFFER: u32 = 0x4000;
        const ENTRY: u32 = 0x30;

        let mut cpu = request_with_recv_buffer(21, &[], BUFFER, 4 * ENTRY);
        cpu.mem.map_zero(BUFFER, 0x200).unwrap();
        cpu.set_sys_request(TLS, Some(21)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 1, "one agreement");
        assert_ne!(cpu.mem.read_u32(BUFFER).unwrap(), 0, "a version was set");
        assert_eq!(cpu.mem.read_u32(BUFFER + 4).unwrap(), 1, "SetRegion_USA");

        // A buffer with no room for an entry gets a count of zero, not one
        // naming an entry the caller has nowhere to read.
        write_map_buffer_request(&mut cpu, 21, &[], BUFFER, ENTRY - 1, false);
        cpu.set_sys_request(TLS, Some(21)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "nothing fits");
    }

    #[test]
    fn set_sys_get_serial_number_returns_a_nul_padded_placeholder() {
        let mut cpu = request(false, 68, &[]);
        cpu.set_sys_request(TLS, Some(68)).unwrap();
        let mut got = [0u8; 0x18];
        for (i, byte) in got.iter_mut().enumerate() {
            *byte = cpu.mem.read_u8(TLS + 0x20 + i as u32).unwrap();
        }
        assert!(got.starts_with(b"XAW00000000000"));
        assert_eq!(got[b"XAW00000000000".len()], 0, "NUL-padded, not garbage");
    }

    #[test]
    fn set_sys_fills_the_settings_blocks_that_outrun_the_reply_padding() {
        // A reply zeroes four padding words, which covers an out parameter up
        // to 16 bytes. `TvSettings` is 0x20 and `NotificationSettings` 0x18,
        // so past that a caller read the tail of its own request back --
        // `tv_gama` and `contrast_ratio` are floats, so a NaN gamma was
        // reachable and not merely a wrong number. The scribble is where
        // those tails land.
        const STALE: u8 = 0xa5;

        let mut cpu = request(false, 39, &[]);
        for offset in 0x30..0x40 {
            cpu.mem.write_u8(TLS + offset, STALE).unwrap();
        }
        cpu.set_sys_request(TLS, Some(39)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0xc, "TvFlag");
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x24).unwrap(),
            0,
            "TvResolution_Auto"
        );
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x28).unwrap(),
            4,
            "HdmiContentType_Game"
        );
        assert_eq!(
            f32::from_bits(cpu.mem.read_u32(TLS + 0x38).unwrap()),
            1.0,
            "tv_gama, past the padding"
        );
        assert_eq!(
            f32::from_bits(cpu.mem.read_u32(TLS + 0x3c).unwrap()),
            0.5,
            "contrast_ratio, past the padding"
        );

        let mut cpu = request(false, 29, &[]);
        for offset in 0x30..0x40 {
            cpu.mem.write_u8(TLS + offset, STALE).unwrap();
        }
        cpu.set_sys_request(TLS, Some(29)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x20).unwrap(),
            0x300,
            "NotificationFlag"
        );
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x24).unwrap(),
            2,
            "NotificationVolume_High"
        );
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x28).unwrap(),
            9,
            "quiet hours start"
        );
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x30).unwrap(),
            21,
            "and they stop, past the padding"
        );
        assert_eq!(cpu.mem.read_u32(TLS + 0x34).unwrap(), 0, "on the hour");
    }

    #[test]
    fn set_sys_answers_the_enums_whose_zero_is_wrong() {
        // `ProductModel` starts at 1, so zero is outside it -- the same shape
        // as GetPlatformRegion above. `KeyboardLayout`'s zero is `Japanese`,
        // a real layout but not the one a console reporting en-US everywhere
        // else should hand the software keyboard.
        let mut cpu = request(false, 79, &[]);
        cpu.set_sys_request(TLS, Some(79)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 1, "ProductModel_Nx");

        let mut cpu = request(false, 136, &[]);
        cpu.set_sys_request(TLS, Some(136)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x20).unwrap(),
            1,
            "KeyboardLayout_EnglishUs"
        );
    }

    #[test]
    fn set_sys_sleeps_never_rather_than_at_an_arbitrary_hour() {
        // The one setting here whose zero is actively wrong. `SleepSettings`
        // is { SleepFlag; HandheldSleepPlan; ConsoleSleepPlan }, and the plans
        // are *indices* rather than durations — so the zeroes the empty-success
        // stub left behind said "sleep after one minute", not "do not sleep".
        // Nothing here dims a screen this emulator does not own.
        const NEVER: u32 = 5;
        let mut cpu = request(false, 71, &[]);
        cpu.set_sys_request(TLS, Some(71)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "flags");
        assert_eq!(cpu.mem.read_u32(TLS + 0x24).unwrap(), NEVER, "handheld");
        assert_eq!(cpu.mem.read_u32(TLS + 0x28).unwrap(), NEVER, "console");
    }

    #[test]
    fn set_sys_names_the_settings_it_used_to_leave_to_the_reply_padding() {
        // These are answered rather than stubbed now. The values match the
        // zeroes the padding already supplied, so this is not a fix for
        // anything the guest saw — it is the difference between a console that
        // says it is retail and one that merely never said otherwise.
        for (cmd, want) in [
            (17u32, 0u32), // GetAccountSettings
            (23, 0),       // GetColorSetId, BasicWhite
            (31, 0),       // GetAccountNotificationSettings, no overrides
            (63, 0),       // GetPrimaryAlbumStorage, Nand
            (124, 0),      // GetErrorReportSharePermission, NotConfirmed
            (126, 0),      // GetAppletLaunchFlags
            (170, 0),      // GetChineseTraditionalInputMethod
        ] {
            let mut cpu = request(false, cmd, &[]);
            cpu.set_sys_request(TLS, Some(cmd)).unwrap();
            assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "cmd {cmd} result");
            assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), want, "cmd {cmd}");
        }
        // QuestFlag is a `u8`: Retail, not Kiosk.
        for cmd in [7u32, 47, 95, 99, 201] {
            let mut cpu = request(false, cmd, &[]);
            cpu.set_sys_request(TLS, Some(cmd)).unwrap();
            assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 0, "cmd {cmd}");
        }
    }

    #[test]
    fn set_sys_reports_a_real_firmware_version_into_its_pointer_buffer() {
        // libnx seeds `hosversionGet` from this, and everything version-gated
        // branches on it. Answering with an empty success left the caller
        // reading its own stale buffer: NX-Fetch showed "Horizon OS
        // 115.119.105", the ASCII of the uid this emulator had left there.
        const BUFFER: u32 = 0x4000;
        let mut cpu = request_with_recv_static(3, &[], BUFFER, 0x100);
        cpu.mem.map_zero(BUFFER, 0x200).unwrap();
        for offset in 0..0x100 {
            cpu.mem.write_u8(BUFFER + offset, b'x').unwrap();
        }
        cpu.set_sys_request(TLS, Some(3)).unwrap();

        let (major, minor, micro) = super::FIRMWARE_VERSION;
        assert_eq!(cpu.mem.read_u8(BUFFER).unwrap(), major);
        assert_eq!(cpu.mem.read_u8(BUFFER + 1).unwrap(), minor);
        assert_eq!(cpu.mem.read_u8(BUFFER + 2).unwrap(), micro);
        assert_eq!(cpu.read_string(BUFFER + 0x08, 0x20), "NX");
        // The display strings agree with the numbers above them.
        let display = format!("{major}.{minor}.{micro}");
        assert_eq!(cpu.read_string(BUFFER + 0x68, 0x18), display);
        assert!(cpu
            .read_string(BUFFER + 0x80, 0x80)
            .ends_with(&format!("{display}-1.0")));
    }

    #[test]
    fn set_sys_reads_back_what_its_setters_were_given() {
        // The whole point of the service. Every one of these used to fall
        // through to the stub: the caller was told the write had worked and
        // then read back the constant that had always been there.
        const COLOR_SET: u32 = 24;
        const KEYBOARD_LAYOUT: u32 = 137;
        const LOCK_SCREEN: u32 = 8;
        const VIBRATION_VOLUME: u32 = 36;

        let mut cpu = request(false, COLOR_SET, &1u32.to_le_bytes());
        cpu.set_sys_request(TLS, Some(COLOR_SET)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        write_request(&mut cpu, 23, &[]);
        cpu.set_sys_request(TLS, Some(23)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x20).unwrap(),
            1,
            "ColorSet_BasicBlack"
        );

        write_request(&mut cpu, KEYBOARD_LAYOUT, &4u32.to_le_bytes());
        cpu.set_sys_request(TLS, Some(KEYBOARD_LAYOUT)).unwrap();
        write_request(&mut cpu, 136, &[]);
        cpu.set_sys_request(TLS, Some(136)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x20).unwrap(),
            4,
            "KeyboardLayout_French"
        );

        write_request(&mut cpu, LOCK_SCREEN, &[1]);
        cpu.set_sys_request(TLS, Some(LOCK_SCREEN)).unwrap();
        write_request(&mut cpu, 7, &[]);
        cpu.set_sys_request(TLS, Some(7)).unwrap();
        assert_eq!(
            cpu.mem.read_u8(TLS + 0x20).unwrap(),
            1,
            "the lock screen is on"
        );

        write_request(&mut cpu, VIBRATION_VOLUME, &0.25f32.to_le_bytes());
        cpu.set_sys_request(TLS, Some(VIBRATION_VOLUME)).unwrap();
        write_request(&mut cpu, 35, &[]);
        cpu.set_sys_request(TLS, Some(35)).unwrap();
        assert_eq!(
            f32::from_bits(cpu.mem.read_u32(TLS + 0x20).unwrap()),
            0.25,
            "a float setting, not its bit pattern read as an integer"
        );
    }

    #[test]
    fn set_sys_keeps_a_settings_block_whole_past_the_reply_padding() {
        // `TvSettings` is 0x20 bytes, wider than the four padding words a
        // reply zeroes, and every byte of it belongs to the caller. A setter
        // that reads only the first four words is one whose contrast ratio
        // reverts.
        const SET_TV_SETTINGS: u32 = 40;
        let mut block = [0u8; 0x20];
        block[0x00..0x04].copy_from_slice(&1u32.to_le_bytes()); // Allows4k
        block[0x04..0x08].copy_from_slice(&2u32.to_le_bytes()); // 720p
        block[0x18..0x1c].copy_from_slice(&2.2f32.to_le_bytes()); // tv_gama
        block[0x1c..0x20].copy_from_slice(&0.75f32.to_le_bytes()); // contrast

        let mut cpu = request(false, SET_TV_SETTINGS, &block);
        cpu.set_sys_request(TLS, Some(SET_TV_SETTINGS)).unwrap();
        write_request(&mut cpu, 39, &[]);
        cpu.set_sys_request(TLS, Some(39)).unwrap();
        for (index, &byte) in block.iter().enumerate() {
            assert_eq!(
                cpu.mem.read_u8(TLS + 0x20 + index as u32).unwrap(),
                byte,
                "byte {index} of TvSettings"
            );
        }
    }

    #[test]
    fn set_sys_keeps_one_audio_mode_per_output() {
        // The target argument is which output is being asked about. A service
        // that answered them all alike would report the headphones set to
        // whatever was last chosen for the dock.
        const SET: u32 = 44;
        const GET: u32 = 43;
        const HEADPHONE: u32 = 3;
        const HDMI: u32 = 1;
        const CH_5_1: u32 = 2;

        let mut payload = Vec::new();
        payload.extend_from_slice(&HEADPHONE.to_le_bytes());
        payload.extend_from_slice(&CH_5_1.to_le_bytes());
        let mut cpu = request(false, SET, &payload);
        cpu.set_sys_request(TLS, Some(SET)).unwrap();

        write_request(&mut cpu, GET, &HEADPHONE.to_le_bytes());
        cpu.set_sys_request(TLS, Some(GET)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x20).unwrap(),
            CH_5_1,
            "the headphones"
        );

        write_request(&mut cpu, GET, &HDMI.to_le_bytes());
        cpu.set_sys_request(TLS, Some(GET)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x20).unwrap(),
            super::AUDIO_OUTPUT_STEREO,
            "and the dock, which nothing changed"
        );
    }

    #[test]
    fn set_sys_replaces_the_eula_list_from_the_buffer_it_was_handed() {
        // SetEulaVersions is not "add one": it is the list, and a console
        // handed two agreements has two rather than three.
        const SET: u32 = 22;
        const GET: u32 = 21;
        const ENTRY: u32 = 0x30;
        const IN: u32 = 0x4000;
        const OUT: u32 = 0x5000;

        let mut cpu = request(false, SET, &[]);
        cpu.mem.map_zero(IN, 0x200).unwrap();
        cpu.mem.map_zero(OUT, 0x200).unwrap();
        // Two agreements, distinguishable by their version words.
        cpu.mem.write_u32(IN, 0x2_0000).unwrap();
        cpu.mem.write_u32(IN + ENTRY, 0x3_0000).unwrap();
        write_map_buffer_request(&mut cpu, SET, &[], IN, 2 * ENTRY, true);
        cpu.set_sys_request(TLS, Some(SET)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");

        write_map_buffer_request(&mut cpu, GET, &[], OUT, 4 * ENTRY, false);
        cpu.set_sys_request(TLS, Some(GET)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 2, "two agreements");
        assert_eq!(cpu.mem.read_u32(OUT).unwrap(), 0x2_0000);
        assert_eq!(cpu.mem.read_u32(OUT + ENTRY).unwrap(), 0x3_0000);
    }

    #[test]
    fn set_sys_device_nick_name_round_trips_through_its_buffers() {
        // 0x80 bytes each way, and through a buffer rather than the raw data
        // in both directions — the setter reading the raw data would store
        // the descriptor words instead of the name.
        const SET: u32 = 78;
        const GET: u32 = 77;
        const IN: u32 = 0x4000;
        const OUT: u32 = 0x5000;
        const NAME: &[u8] = b"the console in the tab";

        let mut cpu = request(false, SET, &[]);
        cpu.mem.map_zero(IN, 0x200).unwrap();
        cpu.mem.map_zero(OUT, 0x200).unwrap();
        for (index, &byte) in NAME.iter().enumerate() {
            cpu.mem.write_u8(IN + index as u32, byte).unwrap();
        }
        write_map_buffer_request(&mut cpu, SET, &[], IN, 0x80, true);
        cpu.set_sys_request(TLS, Some(SET)).unwrap();

        write_map_buffer_request(&mut cpu, GET, &[], OUT, 0x80, false);
        cpu.set_sys_request(TLS, Some(GET)).unwrap();
        assert_eq!(cpu.read_string(OUT, 0x80), "the console in the tab");
    }

    #[test]
    fn a_setting_survives_the_session_that_wrote_it() {
        // The settings live in system save data `8000000000000050`, which the
        // host writes back to the browser and restores into the next session.
        // A colour set chosen in the settings applet that reverted on a
        // reload would be the same bug the stub had, one step further out.
        const SET_COLOR_SET: u32 = 24;
        const GET_COLOR_SET: u32 = 23;

        let mut cpu = request(false, SET_COLOR_SET, &1u32.to_le_bytes());
        cpu.set_sys_request(TLS, Some(SET_COLOR_SET)).unwrap();

        let save = cpu
            .save_data(super::SYSTEM_SETTINGS_SAVE)
            .expect("the setter files the settings in their save");
        // The write is queued for the host, or it never leaves the tab.
        assert!(save.pending_changes() > 0, "the host is told to persist it");
        let stored = save
            .file(super::SYSTEM_SETTINGS_FILE)
            .expect("and it is a file in that save")
            .to_vec();

        // A fresh session, handed the save back the way `saveRestore` does.
        let mut next = request(false, GET_COLOR_SET, &[]);
        next.save_data_mut(super::SYSTEM_SETTINGS_SAVE)
            .write_file(super::SYSTEM_SETTINGS_FILE, stored);
        assert_eq!(
            next.save_data(super::SYSTEM_SETTINGS_SAVE)
                .unwrap()
                .pending_changes(),
            0,
            "restoring a save is not a change to write straight back"
        );
        next.set_sys_request(TLS, Some(GET_COLOR_SET)).unwrap();
        assert_eq!(
            next.mem.read_u32(TLS + 0x20).unwrap(),
            1,
            "the colour set the session before it chose"
        );
    }

    #[test]
    fn settings_that_were_never_stored_keep_their_defaults() {
        // A stored block names the settings it was written with and no more,
        // so a build that adds one reads a file written before it existed.
        // The rest of the console has to come back as its default rather than
        // as a zero.
        let settings = super::SystemSettings {
            color_set: 1,
            ..Default::default()
        };
        let mut stored = settings.serialize();
        // Drop the last record, as a build that did not have it would.
        stored.truncate(stored.len() - 8);
        let read = super::SystemSettings::parse(&stored).expect("a block this build wrote");
        assert_eq!(read.color_set, 1, "what was stored");
        assert_eq!(
            read.panel_crc_mode,
            super::SystemSettings::default().panel_crc_mode,
            "and the default for what was not"
        );

        // Bytes this build did not write are not a settings block at all.
        assert!(super::SystemSettings::parse(b"not a settings block").is_none());
        assert!(super::SystemSettings::parse(&[]).is_none());
    }

    #[test]
    fn the_language_set_sys_is_given_is_the_one_set_reports() {
        // `set:sys`'s SetLanguageCode and `set`'s GetLanguageCode are two
        // halves of one setting, and they used to be a setter that dropped
        // its argument and a getter that answered with a constant.
        const SET_LANGUAGE_CODE: u32 = 0;
        const SET_REGION_CODE: u32 = 57;
        const REGION_EUROPE: u32 = 2;
        let french = super::language_code(2);

        let mut cpu = request(false, SET_LANGUAGE_CODE, &french.to_le_bytes());
        cpu.register_service_handle(9, "set");
        cpu.set_sys_request(TLS, Some(SET_LANGUAGE_CODE)).unwrap();
        write_request(&mut cpu, 0, &[]);
        cpu.set_request(TLS, 9, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), french, "fr");

        write_request(&mut cpu, SET_REGION_CODE, &REGION_EUROPE.to_le_bytes());
        cpu.set_sys_request(TLS, Some(SET_REGION_CODE)).unwrap();
        write_request(&mut cpu, 4, &[]);
        cpu.set_request(TLS, 9, Some(4)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), REGION_EUROPE);

        // MakeLanguageCode is a lookup rather than a question about this
        // console, so it still answers for whichever language it was handed.
        write_request(&mut cpu, 2, &0u32.to_le_bytes());
        cpu.set_request(TLS, 9, Some(2)).unwrap();
        assert_eq!(
            cpu.mem.read_u64(TLS + 0x20).unwrap(),
            super::language_code(0)
        );
    }

    #[test]
    fn the_nfc_switch_is_one_switch() {
        // `set:sys`'s NfcEnableFlag and `nfc:sys`'s IsNfcEnabled are the same
        // setting asked about by two services. Kept apart, a reader turned off
        // in the settings applet is one `nfc` still reports as on.
        const SET_NFC_ENABLE_FLAG: u32 = 70;
        const IS_NFC_ENABLED: u32 = 403;

        let mut cpu = request(false, SET_NFC_ENABLE_FLAG, &[1]);
        cpu.set_sys_request(TLS, Some(SET_NFC_ENABLE_FLAG)).unwrap();
        cpu.register_service_handle(9, "nfc:system");
        write_request(&mut cpu, IS_NFC_ENABLED, &[]);
        cpu.nfc_request(TLS, 9, Some(IS_NFC_ENABLED)).unwrap();
        assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 1, "nfc:sys sees it");

        // And the other way round: `nfc:sys`'s own setter writes the setting
        // the settings applet reads.
        write_request(&mut cpu, 500, &[0]);
        cpu.nfc_request(TLS, 9, Some(500)).unwrap();
        write_request(&mut cpu, 69, &[]);
        cpu.set_sys_request(TLS, Some(69)).unwrap();
        assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 0, "set:sys sees that");
    }

    #[test]
    fn set_sys_serves_a_settings_item_and_refuses_one_it_has_not_got() {
        // The firmware's key/value table. A caller reads the size back and
        // then that many bytes, so an item this console does not have has to
        // be refused rather than answered with a zero it would read as one.
        const SIZE_OF: u32 = 37;
        const VALUE_OF: u32 = 38;
        const CATEGORY: u32 = 0x4000;
        const NAME: u32 = 0x4100;
        const OUT: u32 = 0x5000;

        let mut cpu = request(false, SIZE_OF, &[]);
        cpu.mem.map_zero(CATEGORY, 0x200).unwrap();
        cpu.mem.map_zero(OUT, 0x200).unwrap();
        let write_name = |cpu: &mut Cpu, at: u32, text: &str| {
            for (index, &byte) in text.as_bytes().iter().enumerate() {
                cpu.mem.write_u8(at + index as u32, byte).unwrap();
            }
            cpu.mem.write_u8(at + text.len() as u32, 0).unwrap();
        };
        write_name(&mut cpu, CATEGORY, "hbloader");
        write_name(&mut cpu, NAME, "applet_heap_reservation_size");

        let names = [(CATEGORY, 0x48), (NAME, 0x48)];
        write_buffer_request(&mut cpu, SIZE_OF, &[], &names, &[]);
        cpu.set_sys_request(TLS, Some(SIZE_OF)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), 8, "a u64 item");

        write_buffer_request(&mut cpu, VALUE_OF, &[], &names, &[(OUT, 8)]);
        cpu.set_sys_request(TLS, Some(VALUE_OF)).unwrap();
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), 8, "and its size");
        assert_eq!(
            cpu.mem.read_u64(OUT).unwrap(),
            0x860_0000,
            "the reservation hbloader reads"
        );

        // An item that is not in the table.
        write_name(&mut cpu, NAME, "applet_heap_size_in_bananas");
        write_buffer_request(&mut cpu, VALUE_OF, &[], &names, &[(OUT, 8)]);
        cpu.set_sys_request(TLS, Some(VALUE_OF)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x18).unwrap(),
            105 | (11 << 9),
            "ResultSettingsItemNotFound, not a fabricated zero"
        );
    }

    #[test]
    fn set_get_region_code_reports_usa() {
        let mut cpu = request(false, 4, &[]);
        cpu.register_service_handle(9, "set");
        cpu.set_request(TLS, 9, Some(4)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 1, "SetRegion_USA");
    }

    #[test]
    fn set_writes_the_language_codes_into_a_pointer_buffer() {
        // `GetAvailableLanguageCodes`, the pre-4.0.0 form: the codes come back
        // through a receive-static buffer rather than a map-alias one, and the
        // count beside them is what a caller indexes with. Answered with no
        // data at all, the count read as zero and `nn::settings::LanguageCode::
        // Make` aborted rather than return a code it had not been given.
        const BUFFER: u32 = 0x4000;
        let mut cpu = request_with_recv_static(1, &[], BUFFER, 0x80);
        cpu.mem.map_zero(BUFFER, 0x100).unwrap();
        cpu.register_service_handle(9, "set");
        cpu.set_request(TLS, 9, Some(1)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x20).unwrap(),
            15,
            "the pre-4.0.0 count"
        );
        assert_eq!(&cpu.read_bytes(BUFFER, 2), b"ja");
        assert_eq!(&cpu.read_bytes(BUFFER + 8, 5), b"en-US");
        assert_eq!(&cpu.read_bytes(BUFFER + 13 * 8, 5), b"fr-CA");
        // And nothing past the fifteen it reported: this command's array stops
        // there, so `zh-Hans` is only ever reachable through the 4.0.0 pair.
        assert_eq!(cpu.read_bytes(BUFFER + 15 * 8, 8), vec![0u8; 8]);
    }

    #[test]
    fn set_answers_the_pointer_buffer_size_and_not_its_language_count() {
        // Control command 3 is `QueryPointerBufferSize`; `set`'s own command 3
        // is `GetAvailableLanguageCodeCount`. Answering the first with the
        // second told `nnSdk` a session would take 18 bytes, which is smaller
        // than anything `nn::settings` sends, so it stopped before sending.
        // Answered before dispatch, so the request goes in through the
        // syscall rather than to `set_request`: that interception is what the
        // collision now depends on.
        let mut cpu = control_request(3);
        cpu.tpidr = u64::from(TLS);
        cpu.register_service_handle(9, "set");
        cpu.write_zr(0, 9);
        cpu.horizon_syscall(0x20).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(
            cpu.mem.read_u16(TLS + 0x20).unwrap(),
            super::super::ipc::POINTER_BUFFER_SIZE
        );
        assert_ne!(
            cpu.mem.read_u16(TLS + 0x20).unwrap(),
            18,
            "the language-code count"
        );
    }
}
