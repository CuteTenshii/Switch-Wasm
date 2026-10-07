//! `notif`: the alarm store.

use super::*;

impl Cpu {
    /// `notif:s` / `notif:a`: a real alarm store (register, list, reload,
    /// delete by id). Alarms never fire.
    pub(crate) fn notif_request(
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
        // INotificationSystemEventAccessor::GetSystemEvent.
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
            // RegisterAlarmSetting(AlarmSetting, ApplicationParameter) -> id,
            // also written into the stored copy.
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
            // UpdateAlarmSetting(AlarmSetting, ApplicationParameter), by the setting's id.
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
            // ListAlarmSettings -> settings in a buffer, and the count.
            Some(520) => {
                let mut entries = Vec::new();
                for alarm in &self.notif_alarms {
                    entries.extend_from_slice(&alarm.setting);
                }
                let written = self.write_output_buffer(tls, 0, &entries);
                let count = (written as usize / ALARM_SETTING_SIZE) as i32;
                self.write_ipc_response(tls, 0, &[], &count.to_le_bytes(), &[])
            }
            // LoadApplicationParameter(AlarmSettingId) -> blob and its stored length.
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
            // 1000 is `notif:a`'s Initialize (void) but `notif:s`'s GetNotificationCount.
            Some(1000) if root == "notif:a" => self.write_ipc_response(tls, 0, &[], &[], &[]),
            Some(1000) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            // ListNotifications -> entries into a buffer, and how many.
            Some(1010) => self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[]),
            // DeleteNotification / ClearNotifications: nothing is queued.
            Some(1020) | Some(1030) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // GetNotificationSendingNotifier -> INotificationSystemEventAccessor.
            Some(1040) => {
                self.reply_with_interface(tls, handle, "notif:event-accessor")?;
                Ok(())
            }
            // SetNotificationPresentationSetting /
            // GetNotificationPresentationSetting(NotificationChannel) -> 0x10 bytes.
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
            // IsAlarmSettingDeletable(AlarmSettingId) -> bool.
            Some(2020) => self.write_ipc_response(tls, 0, &[], &[1u8], &[]),
            // RegisterAppletResourceUserId / UnregisterAppletResourceUserId.
            Some(8000) | Some(8010) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // GetCurrentTime -> PosixTime, the same clock `time` reports.
            Some(8999) => {
                let now = self.unix_time();
                self.write_ipc_response(tls, 0, &[], &now.to_le_bytes(), &[])
            }
            // GetAlarmSettingNextNotificationTime(AlarmSettingId) -> unscheduled.
            Some(9000) => self.write_ipc_response(tls, 0, &[], &[0u8; 0x10], &[]),
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }
}
