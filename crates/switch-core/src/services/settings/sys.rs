//! `set:sys` and its settings item helpers.

use super::*;

impl Cpu {
    /// `set:sys`. `Get*`/`Set*` pairs read and write [`SystemSettings`]; the
    /// rest (firmware version, model, serial) are constants.
    pub(crate) fn set_sys_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return self.write_ipc_response(tls, 0, &[], &[], &[]);
        }
        macro_rules! stored {
            ($field:ident = $value:expr) => {{
                let value = $value;
                self.store_system_settings(|settings| settings.$field = value);
                return self.write_ipc_response(tls, 0, &[], &[], &[]);
            }};
        }
        match cmd_id {
            // ---- the settings themselves, setter then getter ----
            // SetLanguageCode(u64); `set`'s GetLanguageCode reads it back.
            Some(0) => stored!(language_code = self.ipc_arg_u64(tls, 0)),
            // SetRegionCode(SystemRegionCode); read back by `set`.
            Some(57) => stored!(region = self.ipc_arg_u32(tls, 0)),
            // Get/SetLockScreenFlag(bool).
            Some(7) => {
                let flag = u8::from(self.system_settings().lock_screen);
                self.write_ipc_response(tls, 0, &[], &[flag], &[])
            }
            Some(8) => stored!(lock_screen = self.ipc_arg_u8(tls, 0) != 0),
            // Get/SetExternalSteadyClockSourceId(Uuid).
            Some(13) => {
                let id = self.system_settings().external_steady_clock_source_id;
                self.write_ipc_response(tls, 0, &[], &id, &[])
            }
            Some(14) => {
                stored!(external_steady_clock_source_id = self.request_block(tls))
            }
            // Get/SetUserSystemClockContext(SystemClockContext) and the network one below.
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
            // GetEulaVersions -> s32 count plus an output buffer (clamped to what
            // fits); SetEulaVersions replaces the list.
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
            // Get/SetColorSetId -> ColorSet.
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
            // Get/SetNotificationSettings -> NotificationSettings, 0x18 bytes.
            Some(29) => {
                let settings = self.system_settings().notification_settings;
                self.write_ipc_response(tls, 0, &[], &settings, &[])
            }
            Some(30) => stored!(notification_settings = self.request_block(tls)),
            // Get/SetAccountNotificationSettings: a count and a buffer, like the EULA pair.
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
            // Get/SetVibrationMasterVolume(float).
            Some(35) => {
                let volume = self.system_settings().vibration_master_volume;
                self.write_ipc_response(tls, 0, &[], &volume.to_le_bytes(), &[])
            }
            Some(36) => stored!(vibration_master_volume = self.ipc_arg_f32(tls, 0)),
            // GetSettingsItemValueSize / GetSettingsItemValue: the firmware's key/value table.
            Some(37) | Some(38) => self.set_sys_item_request(tls, cmd_id == Some(38)),
            // Get/SetTvSettings -> TvSettings, 0x20 bytes.
            Some(39) => {
                let settings = self.system_settings().tv_settings;
                self.write_ipc_response(tls, 0, &[], &settings, &[])
            }
            Some(40) => stored!(tv_settings = self.request_block(tls)),
            // GetAudioOutputMode(AudioOutputModeTarget) /
            // SetAudioOutputMode(target, mode); each target keeps its own mode.
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
            // Get/SetSpeakerAutoMuteFlag(bool).
            Some(45) => {
                let flag = u8::from(self.system_settings().speaker_auto_mute);
                self.write_ipc_response(tls, 0, &[], &[flag], &[])
            }
            Some(46) => stored!(speaker_auto_mute = self.ipc_arg_u8(tls, 0) != 0),
            // Get/SetQuestFlag -> QuestFlag (u8, 0 = Retail).
            Some(47) => {
                let flag = self.system_settings().quest_flag;
                self.write_ipc_response(tls, 0, &[], &[flag], &[])
            }
            Some(48) => stored!(quest_flag = self.ipc_arg_u8(tls, 0)),
            // Get/SetDeviceTimeZoneLocationName(LocationName); `time` reports the same field.
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
            // GetDebugModeFlag -> bool, answered from the settings-item table as hardware does.
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
            // Get/SetNfcEnableFlag(bool) and Get/SetBluetoothEnableFlag(bool),
            // shared with `nfc:sys` and `btm:sys`.
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
            // Get/SetSleepSettings -> SleepSettings, 0xc bytes.
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
            // Get/SetInitialLaunchSettings -> InitialLaunchSettings.
            Some(75) => {
                let settings = self.system_settings().initial_launch_settings;
                self.write_ipc_response(tls, 0, &[], &settings, &[])
            }
            Some(76) => stored!(initial_launch_settings = self.request_block(tls)),
            // Get/SetDeviceNickName: 0x80 bytes through a buffer either way.
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
            // SetExternalSteadyClockInternalOffset(s64) / Get; the setter has the lower id.
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
            // Get/SetKeyboardLayout -> KeyboardLayout.
            Some(136) => {
                let layout = self.system_settings().keyboard_layout;
                self.write_ipc_response(tls, 0, &[], &layout.to_le_bytes(), &[])
            }
            Some(137) => stored!(keyboard_layout = self.ipc_arg_u32(tls, 0)),
            // Get/SetDeviceTimeZoneLocationUpdatedTime and the automatic
            // correction pair: a SteadyClockTimePoint each.
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
            // Get/SetPlatformRegion -> s32, Global (1) or Terra (2); there is no zero.
            Some(183) => {
                let region = self.system_settings().platform_region;
                self.write_ipc_response(tls, 0, &[], &region.to_le_bytes(), &[])
            }
            Some(184) => stored!(platform_region = self.ipc_arg_u32(tls, 0) as i32),
            // Get/SetTouchScreenMode -> TouchScreenMode.
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
            // GetFirmwareVersion / GetFirmwareVersion2 -> `SetSysFirmwareVersion` in a buffer.
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
            // GetBatteryLot / GetSerialNumber -> char[0x18] placeholders.
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
            // GetProductModel -> u32 ProductModel, starting at 1 (Nx).
            Some(79) => self.write_ipc_response(tls, 0, &[], &1u32.to_le_bytes(), &[]),
            // GetMiiAuthorId -> Uuid.
            Some(90) => {
                let id = MII_AUTHOR_ID;
                self.write_ipc_response(tls, 0, &[], &id, &[])
            }
            // GetRebootlessSystemUpdateVersion -> { u32 version;
            // reserved[0x1c]; char display_version[0x20]; }, all zero.
            Some(149) => self.write_ipc_response(tls, 0, &[], &[0u8; 0x40], &[]),
            // GetHomeMenuScheme -> HomeMenuScheme, and GetHomeMenuSchemeModel -> u32 (0).
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

    /// `GetSettingsItemValueSize` (37) and `GetSettingsItemValue` (38). Unknown
    /// items are refused, since callers read back the size and then that many bytes.
    fn set_sys_item_request(&mut self, tls: u32, with_value: bool) -> Result<()> {
        /// `nn::settings::ResultSettingsItemNotFound`.
        const SETTINGS_ITEM_NOT_FOUND: u32 = 105 | (11 << 9);
        /// `nn::settings::SettingItemName`.
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
        // The item's full size, not how much fit.
        let size = value.len() as u64;
        if with_value {
            self.write_output_buffer(tls, 0, &value);
        }
        self.write_ipc_response(tls, 0, &[], &size.to_le_bytes(), &[])
    }

    /// Warn once per item name, since `nnSdk` retries.
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

    /// Write as many whole `size`-byte entries as the first output buffer fits,
    /// and return how many.
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

    /// A fixed-width block from a request's raw data.
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

    /// A fixed-width block from a request's first input buffer.
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

    /// Whole fixed-width entries from a request's first input buffer.
    fn request_list<const N: usize>(&self, tls: u32) -> Vec<[u8; N]> {
        let Some((addr, size)) = self.ipc_input_buffer(tls, 0) else {
            return Vec::new();
        };
        if addr == 0 {
            return Vec::new();
        }
        self.read_bytes(addr, size).as_chunks::<N>().0.to_vec()
    }

    /// `SetSysFirmwareVersion`: version, platform, build hash and display strings.
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
}
