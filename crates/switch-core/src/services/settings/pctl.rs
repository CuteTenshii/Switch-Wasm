//! `pctl`: parental controls.

use super::*;

impl Cpu {
    /// `pctl` and its aliases: parental controls, switched off. Permission
    /// checks succeed, "is restricted" queries are false and "is allowed"
    /// queries are true.
    pub(crate) fn pctl_request(
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
            // CreateServiceWithoutInitialize.
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
                // CheckFreeCommunicationPermission, ConfirmLaunchApplicationPermission,
                // ConfirmResumeApplicationPermission, ConfirmSnsPostPermission,
                // ConfirmSystemSettingsPermission, ConfirmStereoVisionPermission,
                // ConfirmShowNewsPermission, EndFreeCommunication,
                // ResetConfirmedStereoVisionPermission: success means permitted.
                Some(1001..=1005) | Some(1013) | Some(1016) | Some(1017) | Some(1064) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // IsRestrictionTemporaryUnlocked / IsRestrictedSystemSettingsEntered /
                // IsRestrictionEnabled / IsPlayTimerEnabled / IsRestrictedByPlayTimer.
                Some(1006) | Some(1010) | Some(1031) | Some(1453) | Some(1455) => {
                    self.write_ipc_response(tls, 0, &[], &0u8.to_le_bytes(), &[])
                }
                // IsFreeCommunicationAvailable / IsStereoVisionPermitted.
                Some(1018) | Some(1065) => {
                    self.write_ipc_response(tls, 0, &[], &1u8.to_le_bytes(), &[])
                }
                // IsPairingActive -> false; IsPlayTimerAlarmDisabled -> true.
                Some(1403) => self.write_ipc_response(tls, 0, &[], &0u8.to_le_bytes(), &[]),
                Some(1458) => self.write_ipc_response(tls, 0, &[], &1u8.to_le_bytes(), &[]),
                // GetRestrictedFeatures / GetSafetyLevel /
                // GetFreeCommunicationApplicationListCount / GetPinCodeLength /
                // GetAccountState / GetPostEventInterval.
                Some(1012) | Some(1032) | Some(1039) | Some(1206) | Some(1424) | Some(1426) => {
                    self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[])
                }
                // GetCurrentSettings -> RestrictionSettings; rating age -1 passes everything.
                Some(1035) => self.write_ipc_response(tls, 0, &[], &[0xffu8, 0, 0], &[]),
                // GenerateInquiryCode -> char[0x20] ("%02d%08llu"), a fixed placeholder.
                Some(1204) => {
                    const INQUIRY_CODE: &[u8] = b"1100000000";
                    let mut code = [0u8; 0x20];
                    code[..INQUIRY_CODE.len()].copy_from_slice(INQUIRY_CODE);
                    self.write_ipc_response(tls, 0, &[], &code, &[])
                }
                // GetPinCodeChangedEvent, GetSynchronizationEvent,
                // GetPlayTimerEventToRequestSuspension, GetUnlinkedEvent: copy
                // handles, never signalled. `nnSdk` aborts if these are refused.
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
                // GetPlayTimerRemainingTime -> s32; zero would mean time is up.
                Some(1454) => self.write_ipc_response(tls, 0, &[], &i32::MAX.to_le_bytes(), &[]),
                // GetPlayTimerRemainingTimeDisplayInfo -> 0x18 undocumented bytes.
                Some(1459) => self.write_ipc_response(tls, 0, &[], &[0u8; 0x18], &[]),
                // GetPlayTimerSettings: zeroed, sized past the struct.
                Some(1456) => self.write_ipc_response(tls, 0, &[], &[0u8; 0x40], &[]),
                // 18.0.0+ id for the same; 0x44 bytes since 21.0.0.
                Some(145601) => self.write_ipc_response(tls, 0, &[], &[0u8; 0x44], &[]),
                // StartPlayTimer / StopPlayTimer / RequestPostEvents /
                // ClearUnlinkedEvent / DisableFeaturesForReset /
                // NotifyApplicationDownloadStarted / NotifyNetworkProfileCreated.
                Some(1046..=1048) | Some(1425) | Some(1451) | Some(1452) | Some(1474) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }
}
