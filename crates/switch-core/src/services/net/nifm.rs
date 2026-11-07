//! `nifm`: a Wi-Fi link that is up, with internet.

use crate::cpu::Cpu;
use crate::Result;

/// The console's address, as reported by `nifm` and for unbound sockets.
pub(super) const NIFM_LOCAL_IP: [u8; 4] = [192, 168, 1, 100];

impl Cpu {
    /// `nifm`'s root session (`nifm:u`, `nifm:s`, `nifm:a`), handing out `IGeneralService`.
    pub(crate) fn nifm_request(
        &mut self,
        tls: u32,
        cmd_id: Option<u32>,
        handle: u64,
    ) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let name = self.service_name(handle).unwrap_or("nifm:u").to_string();
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, &name);
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
            };
        }
        const CREATE_GENERAL_SERVICE_OLD: u32 = 4;
        const CREATE_GENERAL_SERVICE: u32 = 5;
        match cmd_id {
            Some(CREATE_GENERAL_SERVICE_OLD) | Some(CREATE_GENERAL_SERVICE) => {
                self.reply_with_interface(tls, handle, "nifm:general-service")?;
                Ok(())
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// `IGeneralService`: a Wi-Fi link that is up with internet access.
    pub(crate) fn nifm_general_service_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        match cmd_id {
            // GetClientId.
            Some(1) => self.write_ipc_response(tls, 0, &[], &1u32.to_le_bytes(), &[]),
            // CreateScanRequest / CreateRequest / CreateTemporaryNetworkProfile.
            Some(2) | Some(4) | Some(14) => {
                self.reply_with_interface(tls, handle, "nifm:request")?;
                Ok(())
            }
            // EnumerateNetworkInterfaces / EnumerateNetworkProfiles: zero.
            Some(6) | Some(7) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            // GetCurrentNetworkProfile / GetCurrentAccessPoint (wired, so no access point): the
            // buffer must still be written.
            Some(5) | Some(36) => {
                if let Some((addr, len)) = self.ipc_output_buffer(tls, 0) {
                    for i in 0..len {
                        self.mem.write_u8(addr.wrapping_add(i), 0)?;
                    }
                }
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // GetCurrentIpAddress.
            Some(12) => self.write_ipc_response(tls, 0, &[], &NIFM_LOCAL_IP, &[]),
            // GetCurrentIpConfigInfo -> IpAddressSetting { bool is_automatic; address; subnet;
            // gateway } then a DnsSetting.
            Some(15) => {
                let mut raw = Vec::with_capacity(0x18);
                raw.push(1); // is_automatic
                raw.extend_from_slice(&NIFM_LOCAL_IP);
                raw.extend_from_slice(&[255, 255, 255, 0]);
                raw.extend_from_slice(&[NIFM_LOCAL_IP[0], NIFM_LOCAL_IP[1], NIFM_LOCAL_IP[2], 1]);
                raw.resize(0x18, 0); // the DnsSetting, which resolves nothing
                self.write_ipc_response(tls, 0, &[], &raw, &[])
            }
            // SetWirelessCommunicationEnabled / IsWirelessCommunicationEnabled: the setting, whose
            // radios off together are Airplane Mode in the home menu.
            Some(16) => {
                let on = self.ipc_arg_u8(tls, 0) != 0;
                self.store_system_settings(|settings| settings.wireless_lan_enable = on);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(17) => {
                let enabled = u8::from(self.system_settings().wireless_lan_enable);
                self.write_ipc_response(tls, 0, &[], &[enabled], &[])
            }
            // GetInternetConnectionStatus -> { type, wifi strength, status }: Ieee80211, 3, connected.
            Some(18) => self.write_ipc_response(tls, 0, &[], &[1u8, 3u8, 2u8], &[]),
            // IsEthernetCommunicationEnabled.
            Some(20) => self.write_ipc_response(tls, 0, &[], &[1u8], &[]),
            // IsAnyInternetRequestAccepted / IsAnyForegroundRequestAccepted.
            Some(21) | Some(22) => self.write_ipc_response(tls, 0, &[], &[1u8], &[]),
            _ => {
                self.warn_no_implementation("nifm:general-service", cmd_id);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
        }
    }

    /// `IRequest`: accepted from the moment it exists; its events start signalled.
    pub(crate) fn nifm_request_object_request(
        &mut self,
        tls: u32,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        match cmd_id {
            // GetRequestState -> NifmRequestState_Accepted.
            Some(0) => self.write_ipc_response(tls, 0, &[], &3u32.to_le_bytes(), &[]),
            // GetSystemEventReadableHandles -> two copy handles: state change and completion.
            Some(2) => {
                let state = self.alloc_event("nifm:request-state", true);
                let done = self.alloc_event("nifm:request-done", true);
                self.signal_event(state);
                self.signal_event(done);
                self.write_ipc_reply(tls, 0, &[state, done], &[], &[], &[])
            }
            // GetRevision.
            Some(20) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            // GetResult, Cancel, Submit, SubmitAndWait and the requirement setters.
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }
}
