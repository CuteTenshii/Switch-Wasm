//! The console's settings services: `set`/`set:sys` (system settings and the
//! firmware version), `lbl` (the backlight), `notif` (scheduled alarms) and
//! `pctl` (parental controls).
//!
//! These are **stored, not answered**. One caller writes a setting and another
//! reads it back, so a value that is not kept is a setting that silently
//! reverts — which is a different bug from one that is simply unimplemented.

use super::Cpu;
use crate::Result;

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
        // Language codes in `SetLanguage` order, as NUL-padded ASCII in a u64.
        const LANGUAGE_CODES: [&str; 18] = [
            "ja", "en-US", "fr", "de", "it", "es", "zh-CN", "ko", "nl", "pt", "ru", "zh-TW",
            "en-GB", "fr-CA", "es-419", "zh-Hans", "zh-Hant", "pt-BR",
        ];
        // The language the emulated console is set to (`SetLanguage_ENUS`).
        const SYSTEM_LANGUAGE: usize = 1;
        // How many of those the pre-4.0.0 pair of commands reports. The cap is
        // that command's own 15-entry array, not the twelve languages that
        // predate 4.0.0: `nn::settings::detail::MakeLanguageCode` asks command
        // 1 for 15 codes and indexes the answer by `SetLanguage` directly, so
        // reporting twelve aborted Minecraft on `fr-CA` (13). Only `zh-Hans`,
        // `zh-Hant` and `pt-BR` are out of the legacy pair's reach.
        const LEGACY_LANGUAGE_CODES: usize = 15;

        let code = |index: usize| -> u64 {
            let mut packed = [0u8; 8];
            let name = LANGUAGE_CODES[index].as_bytes();
            packed[..name.len()].copy_from_slice(name);
            u64::from_le_bytes(packed)
        };

        match cmd_id {
            // GetRegionCode -> SetRegion (SetRegion_USA), paired with
            // SYSTEM_LANGUAGE (en-US) above rather than a separate constant.
            Some(4) => self.write_ipc_response(tls, 0, &[], &1u32.to_le_bytes(), &[]),
            // GetLanguageCode
            Some(0) => {
                let raw = code(SYSTEM_LANGUAGE).to_le_bytes();
                self.write_ipc_response(tls, 0, &[], &raw, &[])
            }
            // MakeLanguageCode(SetLanguage) -> u64 code
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

    /// `set:sys` — system settings not covered by the plain `set` service
    /// above (language codes). The commands below answer with real values;
    /// every other one falls through to a generic empty-success reply, same
    /// as `set_request`'s default arm.
    pub(super) fn set_sys_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return self.write_ipc_response(tls, 0, &[], &[], &[]);
        }
        match cmd_id {
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
            // GetInitialLaunchSettings -> nn::settings::system::
            // InitialLaunchSettings { InitialLaunchFlag flags; pad[4];
            // SteadyClockTimePoint timestamp; }.
            //
            // The flags say whether the console has ever been through
            // first-time setup: bit 0 completion, bit 8 the user was added,
            // bit 16 the clock was set. Answering zero means "brand new, never
            // set up", and the Home Menu will not draw a menu for a console
            // that has not finished its setup wizard — it waits to hand over
            // to `starter` instead, which nothing here can launch. This
            // console has been set up.
            Some(75) => {
                const COMPLETION: u32 = 1;
                const USER_ADDITION: u32 = 1 << 8;
                const TIMESTAMP: u32 = 1 << 16;
                let mut settings = [0u8; 0x20];
                settings[..4]
                    .copy_from_slice(&(COMPLETION | USER_ADDITION | TIMESTAMP).to_le_bytes());
                self.write_ipc_response(tls, 0, &[], &settings, &[])
            }
            // GetEulaVersions -> s32 count, with the agreements themselves
            // in a map-alias out buffer: `EulaVersion { u32 version;
            // SystemRegionCode region; EulaVersionClockType clock_type;
            // pad[4]; SystemClockContext accepted_at; }`.
            //
            // A console that has accepted none has not finished first-time
            // setup, and the Home Menu hands over to `starter` for that --
            // the same handover `GetInitialLaunchSettings` above is written
            // to avoid. This one has accepted the agreement for the region
            // it is set to. When it was accepted is not tracked, so the
            // clock context stays zero; callers gate on version and region.
            Some(21) => {
                const EULA_VERSION: u32 = 0x1_0000;
                const STEADY_CLOCK: u32 = 1;
                const ENTRY_SIZE: usize = 0x30;

                let mut eula = [0u8; ENTRY_SIZE];
                eula[..4].copy_from_slice(&EULA_VERSION.to_le_bytes());
                // SetRegion_USA, the region `set`'s GetRegionCode reports.
                eula[4..8].copy_from_slice(&1u32.to_le_bytes());
                eula[8..12].copy_from_slice(&STEADY_CLOCK.to_le_bytes());

                let (addr, size) = self.ipc_output_buffer(tls, 0).unwrap_or((0, 0));
                let count = if addr == 0 {
                    0
                } else {
                    (size as usize / ENTRY_SIZE).min(1)
                };
                if count == 1 {
                    for (index, &byte) in eula.iter().enumerate() {
                        self.mem.write_u8(addr.wrapping_add(index as u32), byte)?;
                    }
                }
                self.write_ipc_response(tls, 0, &[], &(count as i32).to_le_bytes(), &[])
            }
            // GetPlatformRegion -> s32 `nn::settings::PlatformRegion`, which
            // is Global (1) or Terra (2) — the Chinese console — and has no
            // zero. So the generic empty-success reply below left the caller
            // reading a value that is not a member of the enum, and
            // `nn::settings` aborts on that: the error applet took an
            // svcBreak with no message here, one command into its own start.
            // This console is the Global one, which is what `set`'s
            // SetRegion_USA already says.
            Some(183) => self.write_ipc_response(tls, 0, &[], &1i32.to_le_bytes(), &[]),
            // GetSerialNumber -> SetSysSerialNumber { char number[0x18] }.
            // Real hardware's is burned in at manufacturing and unique per
            // console; this is a fixed placeholder, not a real serial.
            Some(68) => {
                const SERIAL: &[u8] = b"XAW00000000000";
                let mut number = [0u8; 0x18];
                number[..SERIAL.len()].copy_from_slice(SERIAL);
                self.write_ipc_response(tls, 0, &[], &number, &[])
            }
            // GetProductModel -> u32 `nn::settings::system::ProductModel`,
            // which starts at 1 (Nx). Zero is not a model, so the generic
            // empty-success reply sat outside the enum the same way
            // GetPlatformRegion's did.
            Some(79) => self.write_ipc_response(tls, 0, &[], &1u32.to_le_bytes(), &[]),
            // GetKeyboardLayout -> u32 `nn::settings::system::KeyboardLayout`.
            // Zero is `Japanese`, which is a real layout but not this
            // console's: `set`'s GetRegionCode and GetLanguageCode both say
            // en-US, and the software keyboard reads this to lay out its keys.
            Some(136) => self.write_ipc_response(tls, 0, &[], &1u32.to_le_bytes(), &[]),
            // GetTvSettings -> nn::settings::system::TvSettings, 0x20 bytes —
            // wider than the four padding words a reply zeroes. The tail,
            // `tv_gama` and `contrast_ratio`, came back as whatever the
            // caller's own request had left in TLS: two floats, so a NaN gamma
            // is a reachable answer rather than merely a wrong one.
            Some(39) => {
                const ALLOWS_CEC: u32 = 1 << 2;
                const PREVENTS_SCREEN_BURN_IN: u32 = 1 << 3;
                const HDMI_CONTENT_TYPE_GAME: u32 = 4;
                let mut settings = [0u8; 0x20];
                settings[0x00..0x04]
                    .copy_from_slice(&(ALLOWS_CEC | PREVENTS_SCREEN_BURN_IN).to_le_bytes());
                settings[0x08..0x0c].copy_from_slice(&HDMI_CONTENT_TYPE_GAME.to_le_bytes());
                // tv_resolution Auto, rgb_range Auto, cmu_mode None and
                // tv_underscan 0 are the zero the array already holds.
                settings[0x18..0x1c].copy_from_slice(&1.0f32.to_le_bytes());
                settings[0x1c..0x20].copy_from_slice(&0.5f32.to_le_bytes());
                self.write_ipc_response(tls, 0, &[], &settings, &[])
            }
            // GetNotificationSettings -> nn::settings::system::
            // NotificationSettings { NotificationFlag flags; NotificationVolume
            // volume; NotificationTime start_time, stop_time; }, 0x18 bytes.
            // Also wider than the padding, so `stop_time` was stale — a quiet
            // period that ends at an arbitrary hour.
            Some(29) => {
                const ENABLES_NEWS: u32 = 1 << 8;
                const INCOMING_LAMP: u32 = 1 << 9;
                const VOLUME_HIGH: u32 = 2;
                let mut settings = [0u8; 0x18];
                settings[0x00..0x04].copy_from_slice(&(ENABLES_NEWS | INCOMING_LAMP).to_le_bytes());
                settings[0x04..0x08].copy_from_slice(&VOLUME_HIGH.to_le_bytes());
                settings[0x08..0x0c].copy_from_slice(&9u32.to_le_bytes());
                settings[0x10..0x14].copy_from_slice(&21u32.to_le_bytes());
                self.write_ipc_response(tls, 0, &[], &settings, &[])
            }
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
