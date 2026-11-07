//! `nifm`: a Wi-Fi link that is up, with internet, on one saved network.

use crate::cpu::Cpu;
use crate::uuid::{self, Uuid};
use crate::Result;

/// The console's address, as reported by `nifm` and for unbound sockets.
pub(super) const NIFM_LOCAL_IP: [u8; 4] = [192, 168, 1, 100];

/// The network's mask, a /24.
const NIFM_SUBNET_MASK: [u8; 4] = [255, 255, 255, 0];

const NIFM_GATEWAY: [u8; 4] = [192, 168, 1, 1];

/// The DNS servers it reports: the gateway first, as a home router hands out, then a public one.
const NIFM_DNS: [[u8; 4]; 2] = [NIFM_GATEWAY, [1, 1, 1, 1]];

/// The one saved network, which is also the one the console is joined to: its name is its SSID.
const WIFI_NAME: &str = "switch-wasm";

/// The profile's id, fixed so every question about it gets the same answer.
const WIFI_PROFILE_ID: Uuid = uuid::parse("531e8558-b186-4424-ad40-1d1d21006036");
const _: () = assert!(uuid::is_valid(&WIFI_PROFILE_ID));

/// `NetworkProfileType::User` (a bit), `NetworkInterfaceType::Ieee80211`, `Authentication::Wpa2Psk`
/// and `Encryption::Aes`.
const PROFILE_USER: u8 = 1;
const INTERFACE_WIFI: u8 = 1;
const AUTHENTICATION_WPA2_PSK: u8 = 6;
const ENCRYPTION_AES: u8 = 4;

/// The size of a `SfNetworkProfileData` and of a `SfNetworkProfileBasicInfo`.
const PROFILE_DATA_SIZE: usize = 0x17C;
const PROFILE_BASIC_INFO_SIZE: usize = 0x75;

/// The `IpSettingData` of the network: the address, mask, gateway, DNS and MTU, with no proxy.
fn ip_setting() -> Vec<u8> {
    let mut raw = vec![0u8; 0xC2];
    raw[0] = 1; // is_automatic: the router hands the address out
    raw[1..5].copy_from_slice(&NIFM_LOCAL_IP);
    raw[5..9].copy_from_slice(&NIFM_SUBNET_MASK);
    raw[9..13].copy_from_slice(&NIFM_GATEWAY);
    raw[0xD] = 1;
    raw[0xE..0x12].copy_from_slice(&NIFM_DNS[0]);
    raw[0x12..0x16].copy_from_slice(&NIFM_DNS[1]);
    raw[0xC0..0xC2].copy_from_slice(&1500u16.to_le_bytes());
    raw
}

/// An `Ssid`: a length and the name, 0x21 bytes.
fn ssid() -> [u8; 0x21] {
    let mut raw = [0u8; 0x21];
    raw[0] = WIFI_NAME.len() as u8;
    raw[1..=WIFI_NAME.len()].copy_from_slice(WIFI_NAME.as_bytes());
    raw
}

/// The saved profile, as `GetCurrentNetworkProfile` and `GetNetworkProfile` return it.
fn network_profile_data() -> Vec<u8> {
    let mut raw = vec![0u8; PROFILE_DATA_SIZE];
    raw[..0xC2].copy_from_slice(&ip_setting());
    raw[0xC2..0xD2].copy_from_slice(&WIFI_PROFILE_ID);
    raw[0xD2..0xD2 + WIFI_NAME.len()].copy_from_slice(WIFI_NAME.as_bytes());
    raw[0x112] = PROFILE_USER;
    raw[0x113] = INTERFACE_WIFI;
    raw[0x114] = 1; // joins by itself
                    // The wireless setting: the SSID, whether it is hidden, how it is secured, and no key kept.
    raw[0x116..0x116 + 0x21].copy_from_slice(&ssid());
    raw[0x116 + 0x22] = AUTHENTICATION_WPA2_PSK;
    raw[0x116 + 0x23] = ENCRYPTION_AES;
    raw
}

/// The same profile as one entry of `EnumerateNetworkProfiles`.
fn network_profile_basic_info() -> Vec<u8> {
    let mut raw = vec![0u8; PROFILE_BASIC_INFO_SIZE];
    raw[..0x10].copy_from_slice(&WIFI_PROFILE_ID);
    raw[0x10..0x10 + WIFI_NAME.len()].copy_from_slice(WIFI_NAME.as_bytes());
    raw[0x50] = PROFILE_USER;
    raw[0x51] = INTERFACE_WIFI;
    raw[0x52..0x52 + 0x21].copy_from_slice(&ssid());
    raw[0x73] = AUTHENTICATION_WPA2_PSK;
    raw[0x74] = ENCRYPTION_AES;
    raw
}

/// `NifmRequestState`: free until submitted, accepted after.
const REQUEST_FREE: u32 = 1;
const REQUEST_ACCEPTED: u32 = 3;

/// A `nifm` request a guest created.
#[derive(Debug, Clone)]
pub(crate) struct NifmRequest {
    /// The state-change and completion events, both auto-clear.
    events: [u64; 2],
    submitted: bool,
}

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
            // EnumerateNetworkInterfaces: none listed (the entry's layout is not documented).
            Some(6) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            // EnumerateNetworkProfiles(u8 type) -> s32 count: the saved network, for the user's.
            Some(7) => {
                let wanted = self.ipc_arg_u8(tls, 0) & PROFILE_USER != 0;
                let count = u32::from(wanted);
                if wanted {
                    self.write_output_buffer(tls, 0, &network_profile_basic_info());
                }
                self.write_ipc_response(tls, 0, &[], &count.to_le_bytes(), &[])
            }
            // GetCurrentNetworkProfile / GetNetworkProfile(Uuid): the one saved network.
            Some(5) | Some(8) => {
                self.write_output_buffer(tls, 0, &network_profile_data());
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // GetCurrentAccessPoint: the buffer must be written; its layout is not documented, so
            // it stays zero.
            Some(36) => {
                self.zero_output_buffer(tls, 0);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // GetCurrentIpAddress.
            Some(12) => self.write_ipc_response(tls, 0, &[], &NIFM_LOCAL_IP, &[]),
            // GetCurrentIpConfigInfo -> IpAddressSetting { bool is_automatic; address; subnet;
            // gateway } then a DnsSetting.
            Some(15) => {
                let mut raw = ip_setting()[..0x16].to_vec();
                raw.resize(0x18, 0);
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
            // GetInternetConnectionStatus -> { type, wifi strength, status }: Wi-Fi, 3, Connected.
            Some(18) => {
                const WIFI: u8 = 1;
                const CONNECTED: u8 = 4;
                self.write_ipc_response(tls, 0, &[], &[WIFI, 3, CONNECTED], &[])
            }
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

    /// `IRequest`: accepted from the moment it exists. Its events start signalled, and are
    /// signalled again by `Submit`: a guest that submits waits on them for the request to settle,
    /// and they are auto-clear, so the first signal is gone by then.
    pub(crate) fn nifm_request_object_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        match cmd_id {
            // GetRequestState: accepted once submitted, free before.
            Some(0) => {
                let key = self.ipc_object_key(tls, handle);
                let submitted = self
                    .process()
                    .nifm_requests
                    .get(&key)
                    .is_some_and(|request| request.submitted);
                let state = if submitted {
                    REQUEST_ACCEPTED
                } else {
                    REQUEST_FREE
                };
                self.write_ipc_response(tls, 0, &[], &state.to_le_bytes(), &[])
            }
            // GetSystemEventReadableHandles -> two copy handles: state change and completion.
            Some(2) => {
                let state = self.alloc_event("nifm:request-state", true);
                let done = self.alloc_event("nifm:request-done", true);
                self.signal_event(state);
                self.signal_event(done);
                let key = self.ipc_object_key(tls, handle);
                self.process_mut().nifm_requests.insert(
                    key,
                    NifmRequest {
                        events: [state, done],
                        submitted: false,
                    },
                );
                self.write_ipc_reply(tls, 0, &[state, done], &[], &[], &[])
            }
            // Submit.
            Some(4) => {
                let key = self.ipc_object_key(tls, handle);
                let mut events = [0; 2];
                if let Some(request) = self.process_mut().nifm_requests.get_mut(&key) {
                    request.submitted = true;
                    events = request.events;
                }
                for event in events {
                    self.signal_event(event);
                }
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // GetRevision.
            Some(20) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            // GetResult, Cancel, Submit, SubmitAndWait and the requirement setters.
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::WIFI_PROFILE_ID;
    use crate::cpu::Cpu;
    use crate::kernel::ipc::testing::*;

    const KEY: u64 = 9 << 32;

    /// A request object whose events a guest has asked for.
    fn request_with_events() -> (Cpu, [u64; 2]) {
        let mut cpu = request(false, 2, &[]);
        cpu.register_service_handle(9, "nifm:request");
        cpu.nifm_request_object_request(TLS, 9, Some(2)).unwrap();
        let events = cpu.process().nifm_requests[&KEY].events;
        (cpu, events)
    }

    fn signalled(cpu: &Cpu, event: u64) -> bool {
        cpu.events[&event].signaled
    }

    #[test]
    fn a_requests_events_start_signalled_and_submit_signals_them_again() {
        let (mut cpu, events) = request_with_events();
        assert!(events.iter().all(|&event| signalled(&cpu, event)));

        // A guest waiting on an auto-clear event takes the signal.
        for event in events {
            cpu.events.get_mut(&event).unwrap().signaled = false;
        }
        write_request(&mut cpu, 4, &[]);
        cpu.nifm_request_object_request(TLS, 9, Some(4)).unwrap();
        assert!(
            events.iter().all(|&event| signalled(&cpu, event)),
            "the submitted request settles and says so"
        );
    }

    #[test]
    fn a_request_is_free_until_it_is_submitted_and_accepted_after() {
        let (mut cpu, _) = request_with_events();
        let state = |cpu: &mut Cpu| {
            write_request(cpu, 0, &[]);
            cpu.nifm_request_object_request(TLS, 9, Some(0)).unwrap();
            cpu.mem.read_u32(TLS + 0x20).unwrap()
        };
        assert_eq!(state(&mut cpu), 1, "Free");
        write_request(&mut cpu, 4, &[]);
        cpu.nifm_request_object_request(TLS, 9, Some(4)).unwrap();
        assert_eq!(state(&mut cpu), 3, "Accepted");
    }

    #[test]
    fn requests_signal_only_their_own_events_and_forget_them_with_their_session() {
        let (mut cpu, events) = request_with_events();
        cpu.register_service_handle(10, "nifm:request");
        for event in events {
            cpu.events.get_mut(&event).unwrap().signaled = false;
        }
        cpu.nifm_request_object_request(TLS, 10, Some(4)).unwrap();
        assert!(
            events.iter().all(|&event| !signalled(&cpu, event)),
            "another request's Submit"
        );

        cpu.forget_handle(9);
        assert!(cpu.process().nifm_requests.is_empty());
        cpu.nifm_request_object_request(TLS, 9, Some(4)).unwrap();
    }

    #[test]
    fn the_internet_connection_status_says_wifi_strength_three_and_connected() {
        let mut cpu = request(false, 18, &[]);
        cpu.register_service_handle(9, "nifm:general-service");
        cpu.nifm_general_service_request(TLS, 9, Some(18)).unwrap();
        let word = cpu.mem.read_u32(TLS + 0x20).unwrap().to_le_bytes();
        assert_eq!(
            word[..3],
            [1, 3, 4],
            "Wi-Fi, strength 3, Connected (not 2, which is still connecting)"
        );
    }

    const OUT: u32 = 0x3000;

    /// A console whose `IGeneralService` is session 9, with an output buffer.
    fn console() -> Cpu {
        let mut cpu = request(false, 5, &[]);
        cpu.mem.map_zero(OUT, 0x400).unwrap();
        cpu.register_service_handle(9, "nifm:general-service");
        cpu
    }

    /// Run `cmd` with `payload`, and the `len` bytes it wrote to the output buffer.
    fn ask(cpu: &mut Cpu, cmd: u32, payload: &[u8], len: u32) -> Vec<u8> {
        write_buffer_request(cpu, cmd, payload, &[], &[(OUT, len)]);
        cpu.nifm_general_service_request(TLS, 9, Some(cmd)).unwrap();
        cpu.read_bytes(OUT, len)
    }

    fn reply_word(cpu: &Cpu) -> u32 {
        cpu.mem.read_u32(TLS + 0x20).unwrap()
    }

    #[test]
    fn the_current_network_profile_is_the_saved_wifi_network() {
        let mut cpu = console();
        let profile = ask(&mut cpu, 5, &[], 0x17C);
        // The IpSettingData: automatic, address, mask, gateway; DNS; no proxy; MTU.
        assert_eq!(
            profile[..13],
            [1, 192, 168, 1, 100, 255, 255, 255, 0, 192, 168, 1, 1]
        );
        assert_eq!(
            profile[0xE..0x16],
            [192, 168, 1, 1, 1, 1, 1, 1],
            "gateway, then 1.1.1.1"
        );
        assert_eq!(profile[0x16], 0, "no proxy");
        assert_eq!(u16::from_le_bytes([profile[0xC0], profile[0xC1]]), 1500);
        // Who it is, and what kind of network.
        assert_eq!(profile[0xC2..0xD2], WIFI_PROFILE_ID);
        assert_eq!(&profile[0xD2..0xD2 + 11], b"switch-wasm");
        assert_eq!(profile[0xDD], 0, "a NUL-terminated name");
        assert_eq!(
            profile[0x112..0x116],
            [1, 1, 1, 0],
            "user, Wi-Fi, auto-connect, not large capacity"
        );
        // Its wireless setting: the SSID, visible, WPA2-PSK with AES, and no key kept.
        let wireless = &profile[0x116..0x116 + 0x65];
        assert_eq!((wireless[0], &wireless[1..12]), (11, &b"switch-wasm"[..]));
        assert_eq!(wireless[0x21], 0, "broadcast");
        assert_eq!(wireless[0x22..0x24], [6, 4]);
        assert_eq!(wireless[0x24], 0, "no key");
    }

    #[test]
    fn the_saved_networks_are_listed_for_the_user_and_fetched_by_id() {
        let mut cpu = console();
        let listed = ask(&mut cpu, 7, &[1, 0, 0, 0], 0x75 * 4);
        assert_eq!(reply_word(&cpu), 1, "one saved network");
        assert_eq!(listed[..0x10], WIFI_PROFILE_ID);
        assert_eq!(&listed[0x10..0x1B], b"switch-wasm");
        assert_eq!(listed[0x50..0x52], [1, 1], "user, Wi-Fi");
        assert_eq!(
            (listed[0x52], &listed[0x53..0x5E]),
            (11, &b"switch-wasm"[..])
        );
        assert_eq!(listed[0x73..0x75], [6, 4]);
        assert!(listed[0x75..].iter().all(|&b| b == 0), "only one entry");

        // The hotspot list and temporary profiles hold nothing.
        ask(&mut cpu, 7, &[2, 0, 0, 0], 0x75);
        assert_eq!(reply_word(&cpu), 0);
        ask(&mut cpu, 7, &[4, 0, 0, 0], 0x75);
        assert_eq!(reply_word(&cpu), 0);

        let current = ask(&mut cpu, 5, &[], 0x17C);
        let by_id = ask(&mut cpu, 8, &WIFI_PROFILE_ID, 0x17C);
        assert_eq!(by_id, current);
    }

    #[test]
    fn the_address_the_ip_config_and_the_profile_report_agree() {
        let mut cpu = console();
        let profile = ask(&mut cpu, 5, &[], 0x17C);
        write_request(&mut cpu, 12, &[]);
        cpu.nifm_general_service_request(TLS, 9, Some(12)).unwrap();
        let address = cpu.mem.read_u32(TLS + 0x20).unwrap().to_le_bytes();
        assert_eq!(address, profile[1..5]);

        write_request(&mut cpu, 15, &[]);
        cpu.nifm_general_service_request(TLS, 9, Some(15)).unwrap();
        let config = cpu.read_bytes(TLS + 0x20, 0x16);
        assert_eq!(config, profile[..0x16], "IpAddressSetting then DnsSetting");
    }
}
