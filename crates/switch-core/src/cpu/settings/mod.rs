//! Settings services: `set`/`set:sys`, `lbl`, `notif` and `pctl`.
//!
//! Settings are stored, not answered: [`SystemSettings`] persists in system
//! save data ([`SYSTEM_SETTINGS_SAVE`]) so writes read back across sessions.

use super::Cpu;
use crate::trace::Level;
use crate::Result;

mod lbl;
mod notif;
mod pctl;
mod sys;

/// Language codes in `SetLanguage` order, as NUL-padded ASCII in a `u64`.
const LANGUAGE_CODES: [&str; 18] = [
    "ja", "en-US", "fr", "de", "it", "es", "zh-CN", "ko", "nl", "pt", "ru", "zh-TW", "en-GB",
    "fr-CA", "es-419", "zh-Hans", "zh-Hant", "pt-BR",
];

/// `SetLanguage_ENUS`.
const DEFAULT_LANGUAGE: usize = 1;

/// `SetRegion_USA`.
const DEFAULT_REGION: u32 = 1;

/// Packed like `nn::settings::LanguageCode`.
fn language_code(index: usize) -> u64 {
    let mut packed = [0u8; 8];
    let name = LANGUAGE_CODES[index.min(LANGUAGE_CODES.len() - 1)].as_bytes();
    packed[..name.len()].copy_from_slice(name);
    u64::from_le_bytes(packed)
}

/// The firmware version `set:sys` reports; libnx's `hosversionGet` branches on it.
const FIRMWARE_VERSION: (u8, u8, u8) = (22, 5, 0);

/// System save data `8000000000000050`, where hardware keeps these settings.
pub(super) const SYSTEM_SETTINGS_SAVE: super::SaveKey =
    super::SaveKey::shared(0x8000_0000_0000_0050);

const SYSTEM_SETTINGS_FILE: &str = "/settings";

/// A file without this magic is ignored in favour of the defaults.
const SYSTEM_SETTINGS_MAGIC: &[u8; 8] = b"swsetsys";
const SYSTEM_SETTINGS_VERSION: u32 = 1;

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

/// `AudioOutputModeTarget`s: None, Hdmi, Speaker, Headphone and two unnamed.
const AUDIO_OUTPUT_TARGETS: usize = 6;

/// `AudioOutputMode_ch_2` (stereo).
const AUDIO_OUTPUT_STEREO: u32 = 1;

/// The `set:sys` settings block. Every field can be written and read back.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct SystemSettings {
    /// Packed `nn::settings::LanguageCode`, with the region beside it.
    pub(super) language_code: u64,
    region: u32,
    /// Independent of the language, as on hardware.
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
    audio_output_mode: [u32; AUDIO_OUTPUT_TARGETS],
    lock_screen: bool,
    console_information_upload: bool,
    automatic_application_download: bool,
    speaker_auto_mute: bool,
    usb30_enable: bool,
    /// `nfc:sys` and `btm:sys` read the radio switches from here.
    pub(super) nfc_enable: bool,
    pub(super) bluetooth_enable: bool,
    wireless_lan_enable: bool,
    auto_update_enable: bool,
    battery_percentage: bool,
    field_testing: bool,
    user_clock_automatic_correction: bool,
    /// Caller-defined blocks stored as raw bytes.
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
    /// Lists a caller replaces wholesale.
    eula_versions: Vec<[u8; EULA_VERSION_SIZE]>,
    account_notification_settings: Vec<[u8; ACCOUNT_NOTIFICATION_SETTINGS_SIZE]>,
}

impl Default for SystemSettings {
    fn default() -> SystemSettings {
        // `EulaVersion { u32 version; SystemRegionCode region;
        // EulaVersionClockType clock_type; pad[4]; SystemClockContext; }`.
        // With none accepted the Home Menu launches `starter`, which is unsupported.
        const EULA_VERSION: u32 = 0x1_0000;
        const EULA_STEADY_CLOCK: u32 = 1;
        let mut eula = [0u8; EULA_VERSION_SIZE];
        eula[0x00..0x04].copy_from_slice(&EULA_VERSION.to_le_bytes());
        eula[0x04..0x08].copy_from_slice(&DEFAULT_REGION.to_le_bytes());
        eula[0x08..0x0c].copy_from_slice(&EULA_STEADY_CLOCK.to_le_bytes());

        // `TvSettings`: CEC and burn-in prevention on, resolution and RGB range Auto.
        const ALLOWS_CEC: u32 = 1 << 2;
        const PREVENTS_SCREEN_BURN_IN: u32 = 1 << 3;
        const HDMI_CONTENT_TYPE_GAME: u32 = 4;
        let mut tv = [0u8; TV_SETTINGS_SIZE];
        tv[0x00..0x04].copy_from_slice(&(ALLOWS_CEC | PREVENTS_SCREEN_BURN_IN).to_le_bytes());
        tv[0x08..0x0c].copy_from_slice(&HDMI_CONTENT_TYPE_GAME.to_le_bytes());
        tv[0x18..0x1c].copy_from_slice(&1.0f32.to_le_bytes());
        tv[0x1c..0x20].copy_from_slice(&0.5f32.to_le_bytes());

        // `NotificationSettings { flags; volume; start_time; stop_time; }`, quiet 21:00 to 09:00.
        const ENABLES_NEWS: u32 = 1 << 8;
        const INCOMING_LAMP: u32 = 1 << 9;
        const VOLUME_HIGH: u32 = 2;
        let mut notification = [0u8; NOTIFICATION_SETTINGS_SIZE];
        notification[0x00..0x04].copy_from_slice(&(ENABLES_NEWS | INCOMING_LAMP).to_le_bytes());
        notification[0x04..0x08].copy_from_slice(&VOLUME_HIGH.to_le_bytes());
        notification[0x08..0x0c].copy_from_slice(&9u32.to_le_bytes());
        notification[0x10..0x14].copy_from_slice(&21u32.to_le_bytes());

        // `SleepSettings { flags; handheld_plan; console_plan; }`, both plans `Never` (5).
        const SLEEP_NEVER: u32 = 5;
        let mut sleep = [0u8; SLEEP_SETTINGS_SIZE];
        sleep[0x04..0x08].copy_from_slice(&SLEEP_NEVER.to_le_bytes());
        sleep[0x08..0x0c].copy_from_slice(&SLEEP_NEVER.to_le_bytes());

        // `InitialLaunchSettings { InitialLaunchFlag; pad[4]; timestamp; }`.
        // First-time setup is marked complete so the Home Menu draws.
        const LAUNCH_COMPLETION: u32 = 1;
        const LAUNCH_USER_ADDITION: u32 = 1 << 8;
        const LAUNCH_TIMESTAMP: u32 = 1 << 16;
        let mut initial_launch = [0u8; INITIAL_LAUNCH_SETTINGS_SIZE];
        initial_launch[..4].copy_from_slice(
            &(LAUNCH_COMPLETION | LAUNCH_USER_ADDITION | LAUNCH_TIMESTAMP).to_le_bytes(),
        );

        let mut nick_name = [0u8; DEVICE_NICK_NAME_SIZE];
        nick_name[..DEVICE_NICK_NAME.len()].copy_from_slice(DEVICE_NICK_NAME);

        // `time` has no TZif database, so UTC is the only zone.
        let mut location = [0u8; LOCATION_NAME_SIZE];
        location[..DEVICE_TIME_ZONE.len()].copy_from_slice(DEVICE_TIME_ZONE);

        SystemSettings {
            language_code: language_code(DEFAULT_LANGUAGE),
            region: DEFAULT_REGION,
            // `KeyboardLayout_EnglishUs`.
            keyboard_layout: 1,
            // `ColorSet_BasicWhite`.
            color_set: 0,
            account_settings: 0,
            applet_launch_flags: 0,
            chinese_traditional_input_method: 0,
            // `ErrorReportSharePermission_NotConfirmed`.
            error_report_share_permission: 0,
            // `PrimaryAlbumStorage_Nand`.
            primary_album_storage: 0,
            push_notification_activity_mode_on_sleep: 0,
            // `PlatformRegion_Global`.
            platform_region: 1,
            panel_crc_mode: 0,
            // `TouchScreenMode_Standard`.
            touch_screen_mode: 1,
            // `QuestFlag_Retail`.
            quest_flag: 0,
            vibration_master_volume: 1.0,
            audio_output_mode: [AUDIO_OUTPUT_STEREO; AUDIO_OUTPUT_TARGETS],
            lock_screen: false,
            console_information_upload: false,
            automatic_application_download: false,
            speaker_auto_mute: false,
            usb30_enable: false,
            nfc_enable: false,
            bluetooth_enable: true,
            wireless_lan_enable: true,
            auto_update_enable: false,
            battery_percentage: false,
            field_testing: false,
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

const DEVICE_NICK_NAME: &[u8] = b"switch-wasm";

/// The zone `time` resolves calendar conversions against.
pub(super) const DEVICE_TIME_ZONE: &[u8] = b"UTC";

/// The `Uuid` Miis made on this console are stamped with; fixed so they stay this console's.
const MII_AUTHOR_ID: [u8; 0x10] = [
    0x73, 0x77, 0x69, 0x74, 0x63, 0x68, 0x2d, 0x77, 0x61, 0x73, 0x6d, 0x00, 0x00, 0x00, 0x00, 0x01,
];

/// `HomeMenuScheme` ARGB colours (main, back, sub, bezel, extra), taken from
/// Eden's stub rather than hardware.
const HOME_MENU_SCHEME: [u32; 5] = [
    0xff32_3232,
    0xff32_3232,
    0xffff_ffff,
    0xffff_ffff,
    0xff00_0000,
];

/// The firmware's settings items for `GetSettingsItemValue`. Eden's set,
/// minus `hid_debug` since `hid` is emulated rather than the sysmodule.
fn settings_item(category: &str, name: &str) -> Option<Vec<u8>> {
    let value = match (category, name) {
        ("hbloader", "applet_heap_size") => 0u64.to_le_bytes().to_vec(),
        ("hbloader", "applet_heap_reservation_size") => 0x860_0000u64.to_le_bytes().to_vec(),
        ("time", "notify_time_to_fs_interval_seconds") => 600i32.to_le_bytes().to_vec(),
        ("time", "standard_network_clock_sufficient_accuracy_minutes") => {
            43_200i32.to_le_bytes().to_vec()
        }
        ("time", "standard_steady_clock_rtc_update_interval_minutes") => {
            5i32.to_le_bytes().to_vec()
        }
        ("time", "standard_steady_clock_test_offset_minutes") => 0i32.to_le_bytes().to_vec(),
        ("time", "standard_user_clock_initial_year") => 2023i32.to_le_bytes().to_vec(),
        ("hid", "has_rail_interface") => vec![1],
        ("hid", "has_sio_mcu") => vec![1],
        ("mii", "is_db_test_mode_enabled") => vec![0],
        // Read by `GetDebugModeFlag` as well as by name.
        ("settings_debug", "is_debug_mode_enabled") => vec![0],
        // The error applet does not close itself.
        ("err", "applet_auto_close") => vec![0],
        _ => return None,
    };
    Some(value)
}

impl SystemSettings {
    /// A header, then records tagged with the `set:sys` command id that carries
    /// each setting. Readers skip unknown tags and keep defaults for missing ones.
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

    /// `None` when the bytes are not a block this build wrote.
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
            // A truncated record ends the read; earlier records stand.
            let Some(value) = stored.get(at..at + len) else {
                break;
            };
            at += len;
            settings.restore(tag, value);
        }
        Some(settings)
    }

    /// A value of the wrong width is skipped, keeping the default.
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
            value.as_chunks::<N>().0.to_vec()
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
                for (target, mode) in value.as_chunks::<4>().0.iter().enumerate() {
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
            // A setting from a newer build.
            _ => {}
        }
    }
}

/// `lbl`'s backlight state. Nothing reaches a panel, but the settings must
/// agree: the settings applet sets a brightness and reads the applied one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Backlight {
    /// The brightness, 0.0 to 1.0, and the copy `SaveCurrentSetting` took.
    setting: f32,
    saved: f32,
    /// The brightness used in VR mode.
    vr_setting: f32,
    /// Separate from brightness: `SwitchBacklightOff` leaves the setting alone.
    on: bool,
    dimming: bool,
    auto_brightness: bool,
    vr_mode: bool,
    /// Only ever what `SetAmbientLightSensorValue` stored.
    lux: f32,
    /// Stored only so their getters read them back.
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

/// One `notif` alarm: the caller's `AlarmSetting` (with its assigned id) and
/// opaque parameter, both returned verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct AlarmSetting {
    id: u16,
    setting: Vec<u8>,
    parameter: Vec<u8>,
}

/// The size of `nn::notification::AlarmSetting`, and where its id sits.
const ALARM_SETTING_SIZE: usize = 0x40;

const ALARM_SETTING_ID: usize = 0;

const ALARM_PARAMETER_MAX: u32 = 0x400;

impl Cpu {
    pub(super) fn set_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        // Control requests first: `set` has its own command 3, which collided
        // with `QueryPointerBufferSize`.
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
        // The pre-4.0.0 commands use a 15-entry array indexed by `SetLanguage`.
        const LEGACY_LANGUAGE_CODES: usize = 15;

        let code = language_code;

        match cmd_id {
            // GetRegionCode -> SetRegion, GetLanguageCode -> packed code, from the stored settings.
            Some(4) => {
                let region = self.system_settings().region;
                self.write_ipc_response(tls, 0, &[], &region.to_le_bytes(), &[])
            }
            Some(0) => {
                let raw = self.system_settings().language_code.to_le_bytes();
                self.write_ipc_response(tls, 0, &[], &raw, &[])
            }
            // MakeLanguageCode(SetLanguage) -> u64 code.
            Some(2) => {
                let language = self.mem.read_u32(self.ipc_request_data(tls)).unwrap_or(0);
                let index = (language as usize).min(LANGUAGE_CODES.len() - 1);
                self.write_ipc_response(tls, 0, &[], &code(index).to_le_bytes(), &[])
            }
            // GetAvailableLanguageCodes (1 = pre-4.0.0, receive-static buffer) and
            // GetAvailableLanguageCodes2 (5, map-alias buffer) -> count written.
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
            // GetAvailableLanguageCodeCount (3 = pre-4.0.0, 6 = current); must
            // match what the fill command writes.
            Some(3) | Some(6) => {
                let total = match cmd_id {
                    Some(3) => LEGACY_LANGUAGE_CODES,
                    _ => LANGUAGE_CODES.len(),
                } as u32;
                self.write_ipc_response(tls, 0, &[], &total.to_le_bytes(), &[])
            }
            // GetDeviceNickName -> 0x80 bytes, the same field `set:sys` serves.
            Some(11) => {
                let name = self.system_settings().device_nick_name;
                self.write_output_buffer(tls, 0, &name);
                self.write_ipc_response(tls, 0, &[], &[], &[])
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

    /// Loaded lazily, since the host restores saves after the session is built.
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

    /// Edit a setting and write the whole block back to its save.
    pub(super) fn store_system_settings(&mut self, edit: impl FnOnce(&mut SystemSettings)) {
        edit(self.system_settings());
        let blob = self.system_settings().serialize();
        self.save_data_mut(SYSTEM_SETTINGS_SAVE)
            .guest_write_file(SYSTEM_SETTINGS_FILE, blob);
    }
}

#[cfg(test)]
mod tests;
