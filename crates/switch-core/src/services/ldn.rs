//! `ldn:u`, local wireless, as a console alone in the room: it can host a network that
//! nobody joins, and its scans find nothing.

use crate::cpu::Cpu;
use crate::Result;

const fn ldn_result(description: u32) -> u32 {
    203 | (description << 9)
}

/// 2203-0029.
const INVALID_PROTOCOL: u32 = ldn_result(29);
/// 2203-0037, for a session not yet initialized; 2203-0039 for any other wrong state.
const NOT_INITIALIZED: u32 = ldn_result(37);
const INVALID_STATE: u32 = ldn_result(39);
/// 2203-0065.
const NETWORK_NOT_FOUND: u32 = ldn_result(65);
/// 2203-0096.
const INVALID_INPUT: u32 = ldn_result(96);
/// 2203-0097: the LocalCommunicationId isn't one of the title's.
const UNKNOWN_LOCAL_COMMUNICATION_ID: u32 = ldn_result(97);

const NETWORK_INFO_SIZE: usize = 0x480;
const NETWORK_CONFIG_SIZE: usize = 0x20;
const SECURITY_PARAMETER_SIZE: usize = 0x20;
const USER_NAME_SIZE: usize = 0x21;
const ADVERTISE_DATA_MAX: usize = 0x180;
const NODE_COUNT_MAX: i8 = 8;
/// `NodeLatestUpdate` entries `GetNetworkInfoAndHistory` fills, one per node.
const NODE_UPDATE_SIZE: usize = 8;

/// Where `CreateNetwork`'s arguments sit: `SecurityConfig`, `UserConfig`, padding, `NetworkConfig`.
/// `CreateNetworkPrivate` puts a `SecurityParameter` after the `SecurityConfig`.
const SECURITY_CONFIG_SIZE: usize = 0x44;
const USER_CONFIG_SIZE: usize = 0x30;
const PASSPHRASE_SIZE_RANGE: std::ops::RangeInclusive<u16> = 0x10..=0x40;

/// The host this console is: a locally administered MAC and a link-local /24.
const HOST_MAC: [u8; 6] = [0x02, 0x00, 0x00, 0x00, 0x00, 0x01];
const HOST_IP: u32 = u32::from_be_bytes([169, 254, 1, 1]);
const HOST_SUBNET_MASK: u32 = u32::from_be_bytes([255, 255, 255, 0]);
/// The channel a retail console picks when `CreateNetwork` overrides the requested one.
const HOST_CHANNEL: i16 = 6;
const LINK_LEVEL_EXCELLENT: i8 = 3;
/// `NetworkType::Ldn`.
const NETWORK_TYPE_LDN: u8 = 2;
/// `SecurityMode::Product`, which `CreateNetwork` writes over the guest's choice.
const SECURITY_MODE_PRODUCT: u16 = 1;

/// `nn::ldn::DisconnectReason`.
const DISCONNECT_REASON_NONE: i16 = 0;
const DISCONNECT_REASON_DESTROYED_BY_USER: i16 = 3;

/// `nn::ldn::Protocol` values an NX console accepts.
const PROTOCOLS: [u32; 2] = [1, 3];

/// Where the title's LocalCommunicationIds sit in its `control.nacp`.
const NACP_LOCAL_COMMUNICATION_IDS: usize = 0x30B0;
const NACP_LOCAL_COMMUNICATION_ID_COUNT: usize = 8;

/// `nn::ldn::State`. `StationConnected` is never reached: there is no network to join.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum LdnState {
    #[default]
    None = 0,
    Initialized = 1,
    AccessPoint = 2,
    AccessPointCreated = 3,
    Station = 4,
}

/// The network a session hosts once `CreateNetwork` succeeds.
#[derive(Debug, Clone)]
struct HostedNetwork {
    /// Server random, then session id.
    security_parameter: [u8; SECURITY_PARAMETER_SIZE],
    user_name: [u8; USER_NAME_SIZE],
    config: [u8; NETWORK_CONFIG_SIZE],
}

/// One `IUserLocalCommunicationService`.
#[derive(Debug, Default)]
pub(crate) struct LdnSession {
    state: LdnState,
    /// Auto-clear, made on the first `GetStateChangeEvent`.
    state_change_event: Option<u64>,
    disconnect_reason: i16,
    accept_policy: u8,
    advertise_data: Vec<u8>,
    network: Option<HostedNetwork>,
}

impl LdnSession {
    /// The `NetworkInfo` of the hosted network, with this console as its only node.
    fn network_info(&self) -> Option<[u8; NETWORK_INFO_SIZE]> {
        let network = self.network.as_ref()?;
        let config = &network.config;
        let (server_random, session_id) = network.security_parameter.split_at(0x10);
        let mut raw = [0u8; NETWORK_INFO_SIZE];
        // NetworkId: the IntentId from the config, then the session id.
        raw[..0x10].copy_from_slice(&config[..0x10]);
        raw[0x10..0x20].copy_from_slice(session_id);
        // CommonNetworkInfo.
        raw[0x20..0x26].copy_from_slice(&HOST_MAC);
        let ssid = ssid_of(session_id);
        raw[0x26] = ssid.len() as u8;
        raw[0x27..0x27 + ssid.len()].copy_from_slice(&ssid);
        raw[0x48..0x4A].copy_from_slice(&config[0x10..0x12]);
        raw[0x4A] = LINK_LEVEL_EXCELLENT as u8;
        raw[0x4B] = NETWORK_TYPE_LDN;
        raw[0x50..0x60].copy_from_slice(server_random);
        raw[0x60..0x62].copy_from_slice(&SECURITY_MODE_PRODUCT.to_le_bytes());
        raw[0x62] = self.accept_policy;
        raw[0x66] = config[0x12];
        raw[0x67] = 1;
        // The access point's NodeInfo, node 0.
        let node = &mut raw[0x68..0xA8];
        node[..4].copy_from_slice(&HOST_IP.to_le_bytes());
        node[4..10].copy_from_slice(&HOST_MAC);
        node[0xB] = 1;
        node[0xC..0xC + USER_NAME_SIZE].copy_from_slice(&network.user_name);
        node[0x2E..0x30].copy_from_slice(&config[0x14..0x16]);
        // AdvertiseData.
        let advertised = self.advertise_data.len();
        raw[0x26A..0x26C].copy_from_slice(&(advertised as u16).to_le_bytes());
        raw[0x26C..0x26C + advertised].copy_from_slice(&self.advertise_data);
        Some(raw)
    }
}

/// An SSID of 32 hex digits from the session id, so each network gets its own.
fn ssid_of(session_id: &[u8]) -> Vec<u8> {
    session_id
        .iter()
        .flat_map(|byte| format!("{byte:02x}").into_bytes())
        .collect()
}

impl Cpu {
    /// `ldn:u` (`IUserServiceCreator`) and the interfaces it hands out.
    pub(crate) fn ldn_user_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        if self.ipc_answer_control(tls, handle, "ldn:u", cmd_id)? {
            return Ok(());
        }
        let iface = self.ipc_interface(tls, handle, "ldn:u");
        match iface.as_str() {
            "ldn:user" => self.ldn_session_request(tls, handle, cmd_id, &iface),
            // IClientProcessMonitor: RegisterClient(pid).
            "ldn:client-monitor" => match cmd_id {
                Some(0) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // CreateUserLocalCommunicationService / CreateClientProcessMonitor.
            _ => match cmd_id {
                Some(0) => self.reply_with_interface(tls, handle, "ldn:user").map(drop),
                Some(1) => self
                    .reply_with_interface(tls, handle, "ldn:client-monitor")
                    .map(drop),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
        }
    }

    /// `IUserLocalCommunicationService`.
    fn ldn_session_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
        iface: &str,
    ) -> Result<()> {
        use LdnState::{AccessPoint, AccessPointCreated, Initialized, Station};
        let key = self.ipc_object_key(tls, handle);
        let state = self.ldn_session(key).state;
        let refusal = |allowed: &[LdnState]| match state {
            _ if allowed.contains(&state) => None,
            LdnState::None => Some(NOT_INITIALIZED),
            _ => Some(INVALID_STATE),
        };
        let required = match cmd_id {
            Some(1 | 2 | 4 | 5 | 101 | 204 | 205) => refusal(&[AccessPointCreated]),
            Some(102 | 103) => refusal(&[AccessPointCreated, Station]),
            Some(104 | 106 | 200 | 300 | 403) => refusal(&[Initialized]),
            Some(201 | 206..=209) => refusal(&[AccessPoint, AccessPointCreated]),
            Some(202 | 203) => refusal(&[AccessPoint]),
            Some(301..=303) => refusal(&[Station]),
            // Disconnect: never connected.
            Some(304) => refusal(&[]),
            Some(401) => refusal(&[Initialized, AccessPoint, AccessPointCreated, Station]),
            _ => None,
        };
        if let Some(result) = required {
            return self.write_ipc_response(tls, result, &[], &[], &[]);
        }
        match cmd_id {
            // GetState.
            Some(0) => self.write_ipc_response(tls, 0, &[], &(state as u32).to_le_bytes(), &[]),
            // GetNetworkInfo / GetNetworkInfoAndHistory: nobody has joined or left.
            Some(1 | 101) => {
                let info = self
                    .ldn_session(key)
                    .network_info()
                    .unwrap_or([0; NETWORK_INFO_SIZE]);
                self.write_output_buffer(tls, 0, &info);
                if cmd_id == Some(101) {
                    self.write_output_buffer(
                        tls,
                        1,
                        &[0; NODE_UPDATE_SIZE * NODE_COUNT_MAX as usize],
                    );
                }
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // GetIpv4Address -> address, subnet mask.
            Some(2) => {
                let mut out = [0u8; 8];
                out[..4].copy_from_slice(&HOST_IP.to_le_bytes());
                out[4..].copy_from_slice(&HOST_SUBNET_MASK.to_le_bytes());
                self.write_ipc_response(tls, 0, &[], &out, &[])
            }
            // GetDisconnectReason -> s16.
            Some(3) => {
                let reason = self.ldn_session(key).disconnect_reason;
                self.write_ipc_response(tls, 0, &[], &i32::from(reason).to_le_bytes(), &[])
            }
            // GetSecurityParameter / GetNetworkConfig.
            Some(4 | 5) => {
                let network = self.ldn_session(key).network.clone();
                let out = match (cmd_id, network) {
                    (Some(4), Some(network)) => network.security_parameter,
                    (_, Some(network)) => network.config,
                    (_, None) => [0; 0x20],
                };
                self.write_ipc_response(tls, 0, &[], &out, &[])
            }
            // GetStateChangeEvent.
            Some(100) => {
                let event = match self.ldn_session(key).state_change_event {
                    Some(event) => event,
                    None => {
                        let event = self.alloc_event("ldn:state-change", true);
                        self.ldn_session(key).state_change_event = Some(event);
                        event
                    }
                };
                self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
            }
            // Scan / ScanPrivate -> s16 count, of networks found: none.
            Some(102 | 103) => {
                let fits = self
                    .ipc_output_buffer(tls, 0)
                    .map_or(0, |(_, size)| size as usize / NETWORK_INFO_SIZE);
                let result = if fits == 0 { INVALID_INPUT } else { 0 };
                self.write_ipc_response(tls, result, &[], &0u32.to_le_bytes(), &[])
            }
            // SetWirelessControllerRestriction / SetWirelessAudioPolicy / SetOperationMode.
            Some(104 | 105 | 403) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // SetProtocol.
            Some(106) => {
                let protocol = self.ipc_arg_u32(tls, 0);
                let result = if PROTOCOLS.contains(&protocol) {
                    0
                } else {
                    INVALID_PROTOCOL
                };
                self.write_ipc_response(tls, result, &[], &[], &[])
            }
            // OpenAccessPoint / OpenStation.
            Some(200 | 300) => {
                self.ldn_session(key).disconnect_reason = DISCONNECT_REASON_NONE;
                let next = if cmd_id == Some(200) {
                    AccessPoint
                } else {
                    Station
                };
                self.ldn_set_state(key, next);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // CloseAccessPoint: drops the network and what it advertised.
            Some(201) => {
                self.ldn_destroy_network(key);
                self.ldn_session(key).advertise_data.clear();
                self.ldn_set_state(key, Initialized);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // CreateNetwork / CreateNetworkPrivate.
            Some(202 | 203) => {
                let result = self.ldn_create_network(tls, key, cmd_id == Some(203));
                if result == 0 {
                    self.ldn_set_state(key, AccessPointCreated);
                }
                self.write_ipc_response(tls, result, &[], &[], &[])
            }
            // DestroyNetwork.
            Some(204) => {
                self.ldn_destroy_network(key);
                self.ldn_set_state(key, AccessPoint);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // Reject(Ipv4Address): no station to drop. AddAcceptFilterEntry / ClearAcceptFilter.
            Some(205 | 208 | 209) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // SetAdvertiseData.
            Some(206) => {
                let (address, size) = self.ipc_input_buffer(tls, 0).unwrap_or((0, 0));
                if size as usize > ADVERTISE_DATA_MAX {
                    return self.write_ipc_response(tls, INVALID_INPUT, &[], &[], &[]);
                }
                let data = self.read_bytes(address, size);
                self.ldn_session(key).advertise_data = data;
                self.ldn_network_info_changed(key);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // SetStationAcceptPolicy(u8).
            Some(207) => {
                let policy = self.ipc_arg_u8(tls, 0);
                self.ldn_session(key).accept_policy = policy;
                self.ldn_network_info_changed(key);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // CloseStation.
            Some(301) => {
                self.ldn_set_state(key, Initialized);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // Connect / ConnectPrivate: no network to find.
            Some(302 | 303) => self.write_ipc_response(tls, NETWORK_NOT_FOUND, &[], &[], &[]),
            // Initialize(pid) / InitializeWithVersion(pid, s32 version).
            Some(400 | 402) => {
                if cmd_id == Some(402) && (self.ipc_arg_u32(tls, 0) as i32) < 0 {
                    return self.write_ipc_response(tls, INVALID_INPUT, &[], &[], &[]);
                }
                if state == LdnState::None {
                    self.ldn_set_state(key, Initialized);
                }
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // Finalize: closes whatever was open.
            Some(401) => {
                self.ldn_destroy_network(key);
                self.ldn_session(key).advertise_data.clear();
                self.ldn_set_state(key, LdnState::None);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            _ => self.unimplemented_command(tls, iface, cmd_id),
        }
    }

    fn ldn_session(&mut self, key: u64) -> &mut LdnSession {
        self.process_mut().ldn_sessions.entry(key).or_default()
    }

    /// Move to `state`, signalling the state-change event if it changed.
    fn ldn_set_state(&mut self, key: u64, state: LdnState) {
        let session = self.ldn_session(key);
        if session.state == state {
            return;
        }
        session.state = state;
        if let Some(event) = session.state_change_event {
            self.signal_event(event);
        }
    }

    /// Signal the state-change event if a hosted network's `NetworkInfo` changed.
    fn ldn_network_info_changed(&mut self, key: u64) {
        let session = self.ldn_session(key);
        if let (Some(_), Some(event)) = (&session.network, session.state_change_event) {
            self.signal_event(event);
        }
    }

    fn ldn_destroy_network(&mut self, key: u64) {
        let session = self.ldn_session(key);
        if session.network.take().is_some() {
            session.disconnect_reason = DISCONNECT_REASON_DESTROYED_BY_USER;
        }
    }

    /// Check and keep `CreateNetwork`'s arguments; returns the result to reply with.
    fn ldn_create_network(&mut self, tls: u32, key: u64, private: bool) -> u32 {
        let parameter_size = if private { SECURITY_PARAMETER_SIZE } else { 0 };
        let user_at = SECURITY_CONFIG_SIZE + parameter_size;
        let config_at = user_at + USER_CONFIG_SIZE + 4;
        let args = self.read_bytes(
            self.ipc_request_data(tls),
            (config_at + NETWORK_CONFIG_SIZE) as u32,
        );
        if args.len() < config_at + NETWORK_CONFIG_SIZE {
            return INVALID_INPUT;
        }
        let passphrase_size = u16::from_le_bytes([args[2], args[3]]);
        let user_name = &args[user_at..user_at + USER_NAME_SIZE];
        let config_bytes = &args[config_at..config_at + NETWORK_CONFIG_SIZE];
        let node_count_max = config_bytes[0x12] as i8;
        let version = i16::from_le_bytes([config_bytes[0x14], config_bytes[0x15]]);
        if !PASSPHRASE_SIZE_RANGE.contains(&passphrase_size)
            || user_name[USER_NAME_SIZE - 1] != 0
            || !(1..=NODE_COUNT_MAX).contains(&node_count_max)
            || version < 0
        {
            return INVALID_INPUT;
        }

        // The tmp copy the service works from keeps only the fields it knows.
        let mut config = [0u8; NETWORK_CONFIG_SIZE];
        config[..8].copy_from_slice(&config_bytes[..8]);
        config[0xA..0xC].copy_from_slice(&config_bytes[0xA..0xC]);
        config[0x10..0x13].copy_from_slice(&config_bytes[0x10..0x13]);
        config[0x14..0x16].copy_from_slice(&config_bytes[0x14..0x16]);
        let requested = i64::from_le_bytes(config[..8].try_into().unwrap_or_default());
        let Some(local_communication_id) = self.ldn_local_communication_id(requested) else {
            return UNKNOWN_LOCAL_COMMUNICATION_ID;
        };
        config[..8].copy_from_slice(&local_communication_id.to_le_bytes());
        let channel = i16::from_le_bytes([config[0x10], config[0x11]]);
        if !private || channel == 0 {
            config[0x10..0x12].copy_from_slice(&HOST_CHANNEL.to_le_bytes());
        }

        let mut security_parameter = [0u8; SECURITY_PARAMETER_SIZE];
        if private {
            security_parameter.copy_from_slice(
                &args[SECURITY_CONFIG_SIZE..SECURITY_CONFIG_SIZE + SECURITY_PARAMETER_SIZE],
            );
        } else {
            for chunk in security_parameter.chunks_exact_mut(8) {
                chunk.copy_from_slice(&self.next_random_u64().to_le_bytes());
            }
        }
        let mut name = [0u8; USER_NAME_SIZE];
        name.copy_from_slice(user_name);
        self.ldn_session(key).network = Some(HostedNetwork {
            security_parameter,
            user_name: name,
            config,
        });
        0
    }

    /// The LocalCommunicationId a network is made with: -1 picks the title's first, and any
    /// other must be one of the title's. Without a `control.nacp` any id goes, and -1 is 0.
    fn ldn_local_communication_id(&self, requested: i64) -> Option<i64> {
        let ids: Vec<i64> = match self.application_controls.get(&self.process().program_id) {
            Some(nacp) => nacp
                .get(NACP_LOCAL_COMMUNICATION_IDS..)
                .unwrap_or_default()
                .chunks_exact(8)
                .take(NACP_LOCAL_COMMUNICATION_ID_COUNT)
                .map(|id| i64::from_le_bytes(id.try_into().unwrap_or_default()))
                .collect(),
            None => return Some(if requested == -1 { 0 } else { requested }),
        };
        match requested {
            -1 => Some(ids.first().copied().unwrap_or(0)),
            id => ids.contains(&id).then_some(id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::ipc::testing::*;

    const SESSION: u64 = 9;
    const KEY: u64 = SESSION << 32;

    fn session() -> Cpu {
        let mut cpu = request(false, 0, &[]);
        cpu.register_service_handle(SESSION, "ldn:user");
        cpu
    }

    /// Send `command` with `payload`; returns the result and the first reply word.
    fn call(cpu: &mut Cpu, command: u32, payload: &[u8]) -> (u32, u32) {
        marshal(cpu, false, command, payload);
        cpu.ldn_user_request(TLS, SESSION, Some(command)).unwrap();
        (
            cpu.mem.read_u32(TLS + 0x18).unwrap(),
            cpu.mem.read_u32(TLS + 0x20).unwrap(),
        )
    }

    fn state(cpu: &mut Cpu) -> u32 {
        call(cpu, 0, &[]).1
    }

    /// `CreateNetwork`'s arguments: a 16-byte passphrase, "host", one LocalCommunicationId.
    fn create_network_args(local_communication_id: i64, node_count_max: i8) -> Vec<u8> {
        let mut args = vec![0u8; 0x98];
        args[2..4].copy_from_slice(&0x10u16.to_le_bytes());
        args[0x44..0x48].copy_from_slice(b"host");
        args[0x78..0x80].copy_from_slice(&local_communication_id.to_le_bytes());
        args[0x78 + 0x12] = node_count_max as u8;
        args[0x78 + 0x14..0x78 + 0x16].copy_from_slice(&7i16.to_le_bytes());
        args
    }

    fn hosting() -> Cpu {
        let mut cpu = session();
        call(&mut cpu, 400, &[]);
        call(&mut cpu, 200, &[]);
        assert_eq!(
            call(&mut cpu, 202, &create_network_args(0x0100_ABCD, 4)).0,
            0
        );
        cpu
    }

    #[test]
    fn a_session_is_not_initialized_until_initialize() {
        let mut cpu = session();
        assert_eq!(state(&mut cpu), 0);
        assert_eq!(
            call(&mut cpu, 200, &[]).0,
            NOT_INITIALIZED,
            "OpenAccessPoint"
        );
        assert_eq!(call(&mut cpu, 401, &[]).0, NOT_INITIALIZED, "Finalize");
        assert_eq!(call(&mut cpu, 400, &[]).0, 0);
        assert_eq!(state(&mut cpu), 1);
    }

    #[test]
    fn hosting_walks_the_access_point_states_and_back() {
        let mut cpu = hosting();
        assert_eq!(state(&mut cpu), 3, "AccessPointCreated");
        assert_eq!(call(&mut cpu, 204, &[]).0, 0, "DestroyNetwork");
        assert_eq!(state(&mut cpu), 2, "AccessPoint");
        assert_eq!(
            call(&mut cpu, 3, &[]).1 as i16,
            DISCONNECT_REASON_DESTROYED_BY_USER
        );
        assert_eq!(call(&mut cpu, 201, &[]).0, 0, "CloseAccessPoint");
        assert_eq!(state(&mut cpu), 1);
        assert_eq!(call(&mut cpu, 401, &[]).0, 0, "Finalize");
        assert_eq!(state(&mut cpu), 0);
    }

    #[test]
    fn a_station_cannot_open_an_access_point_or_find_a_network() {
        let mut cpu = session();
        call(&mut cpu, 400, &[]);
        assert_eq!(call(&mut cpu, 300, &[]).0, 0, "OpenStation");
        assert_eq!(state(&mut cpu), 4);
        assert_eq!(call(&mut cpu, 200, &[]).0, INVALID_STATE, "OpenAccessPoint");
        assert_eq!(call(&mut cpu, 302, &[]).0, NETWORK_NOT_FOUND, "Connect");
        assert_eq!(call(&mut cpu, 304, &[]).0, INVALID_STATE, "Disconnect");
        assert_eq!(call(&mut cpu, 1, &[]).0, INVALID_STATE, "GetNetworkInfo");
        assert_eq!(call(&mut cpu, 301, &[]).0, 0, "CloseStation");
        assert_eq!(state(&mut cpu), 1);
    }

    #[test]
    fn a_scan_finds_no_networks() {
        const BUFFER: u32 = 0x3000;
        let mut cpu = auto_select_request(102, &[], BUFFER, NETWORK_INFO_SIZE as u32 * 2, false);
        cpu.register_service_handle(SESSION, "ldn:user");
        cpu.ldn_session(KEY).state = LdnState::Station;
        cpu.ldn_user_request(TLS, SESSION, Some(102)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "networks found");
    }

    #[test]
    fn the_state_change_event_fires_on_each_transition_only() {
        let mut cpu = session();
        call(&mut cpu, 100, &[]);
        let event = cpu.process().ldn_sessions[&KEY].state_change_event.unwrap();
        call(&mut cpu, 400, &[]);
        assert!(cpu.events[&event].signaled, "Initialize");
        cpu.events.get_mut(&event).unwrap().signaled = false;
        call(&mut cpu, 400, &[]);
        assert!(
            !cpu.events[&event].signaled,
            "Initialize again changes nothing"
        );
        call(&mut cpu, 200, &[]);
        assert!(cpu.events[&event].signaled, "OpenAccessPoint");
    }

    #[test]
    fn create_network_checks_its_arguments() {
        let mut cpu = session();
        call(&mut cpu, 400, &[]);
        call(&mut cpu, 200, &[]);
        let mut bad_passphrase = create_network_args(1, 4);
        bad_passphrase[2] = 0x0F;
        for (args, what) in [
            (create_network_args(1, 0), "no nodes"),
            (create_network_args(1, 9), "too many nodes"),
            (bad_passphrase, "short passphrase"),
        ] {
            assert_eq!(call(&mut cpu, 202, &args).0, INVALID_INPUT, "{what}");
            assert_eq!(state(&mut cpu), 2, "{what}");
        }
    }

    #[test]
    fn the_local_communication_id_comes_from_the_titles_nacp() {
        const PROGRAM: u64 = 0x0100_0000_0000_1000;
        let mut cpu = session();
        cpu.process_mut().program_id = PROGRAM;
        let mut nacp = vec![0u8; 0x4000];
        nacp[0x30B0..0x30B8].copy_from_slice(&0x0100_2222i64.to_le_bytes());
        nacp[0x30B8..0x30C0].copy_from_slice(&0x0100_3333i64.to_le_bytes());
        cpu.add_application_control(PROGRAM, &nacp, &[]);
        call(&mut cpu, 400, &[]);
        call(&mut cpu, 200, &[]);

        assert_eq!(
            call(&mut cpu, 202, &create_network_args(0x0100_4444, 2)).0,
            UNKNOWN_LOCAL_COMMUNICATION_ID
        );
        assert_eq!(call(&mut cpu, 202, &create_network_args(-1, 2)).0, 0);
        let info = cpu.process().ldn_sessions[&KEY].network_info().unwrap();
        assert_eq!(
            i64::from_le_bytes(info[..8].try_into().unwrap()),
            0x0100_2222
        );
    }

    #[test]
    fn the_hosted_network_lists_this_console_and_what_it_advertises() {
        let mut cpu = hosting();
        cpu.ldn_session(KEY).advertise_data = b"lobby".to_vec();
        let session = &cpu.process().ldn_sessions[&KEY];
        let info = session.network_info().unwrap();
        let parameter = session.network.as_ref().unwrap().security_parameter;

        assert_eq!(
            i64::from_le_bytes(info[..8].try_into().unwrap()),
            0x0100_ABCD
        );
        assert_eq!(info[0x10..0x20], parameter[0x10..], "session id");
        assert_eq!(info[0x26], 32, "SSID length");
        assert_eq!(i16::from_le_bytes([info[0x48], info[0x49]]), HOST_CHANNEL);
        assert_eq!(info[0x4B], NETWORK_TYPE_LDN);
        assert_eq!(info[0x50..0x60], parameter[..0x10], "server random");
        assert_eq!(
            (info[0x66], info[0x67]),
            (4, 1),
            "node count max, node count"
        );
        assert_eq!(
            u32::from_le_bytes(info[0x68..0x6C].try_into().unwrap()),
            HOST_IP
        );
        assert_eq!(info[0x73], 1, "connected");
        assert_eq!(&info[0x74..0x78], b"host");
        assert_eq!(i16::from_le_bytes([info[0x96], info[0x97]]), 7, "version");
        assert_eq!(u16::from_le_bytes([info[0x26A], info[0x26B]]), 5);
        assert_eq!(&info[0x26C..0x271], b"lobby");
    }

    #[test]
    fn a_closed_session_forgets_its_network() {
        let mut cpu = hosting();
        cpu.forget_handle(SESSION);
        assert!(cpu.process().ldn_sessions.is_empty());
    }
}
