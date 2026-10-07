//! `lbl`: the backlight controller.

use super::*;

impl Cpu {
    /// `lbl` (`nn::lbl::detail::ILblController`): setter/getter pairs over
    /// [`Backlight`]. No ambient light sensor, so no auto-brightness.
    pub(crate) fn lbl_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        /// `LblBacklightSwitchStatus`.
        const BACKLIGHT_DISABLED: u32 = 0;
        const BACKLIGHT_ENABLED: u32 = 1;
        if self.ipc_answer_control(tls, handle, "lbl", cmd_id)? {
            return Ok(());
        }
        match cmd_id {
            // SaveCurrentSetting / LoadCurrentSetting.
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
            // ApplyCurrentBrightnessSettingToBacklight.
            Some(4) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // GetBrightnessSettingAppliedToBacklight: the setting while on, zero while off.
            Some(5) => {
                let applied = if self.backlight.on {
                    self.backlight.setting
                } else {
                    0.0
                };
                self.write_ipc_response(tls, 0, &[], &applied.to_bits().to_le_bytes(), &[])
            }
            // SwitchBacklightOn / SwitchBacklightOff(fade time); the switch is immediate.
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
            // SetAmbientLightSensorValue(float lux).
            Some(15) => {
                self.backlight.lux = self.ipc_arg_f32(tls, 0);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // GetAmbientLightSensorValue -> { u32 over_limit, float lux }.
            Some(16) => {
                let mut raw = Vec::with_capacity(8);
                raw.extend_from_slice(&0u32.to_le_bytes());
                raw.extend_from_slice(&self.backlight.lux.to_bits().to_le_bytes());
                self.write_ipc_response(tls, 0, &[], &raw, &[])
            }
            // SetBrightnessReflectionDelayLevel(float, float) /
            // GetBrightnessReflectionDelayLevel(float) -> float; one level, selector ignored.
            Some(17) => {
                self.backlight.reflection_delay = self.ipc_arg_f32(tls, 0);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(18) => {
                let level = self.backlight.reflection_delay;
                self.write_ipc_response(tls, 0, &[], &level.to_bits().to_le_bytes(), &[])
            }
            // SetCurrentBrightnessMapping(float, float, float) / Get, and the
            // ambient light sensor's mapping pair.
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
            // IsAmbientLightSensorAvailable / IsAutoBrightnessControlSupported [7.0.0+].
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
            // EnableVrMode / DisableVrMode / IsVrModeEnabled.
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
}
