//! `sfdnsres` and `nsd`: name resolution. With the network off nothing resolves.

use super::bsd::{BSD_AF_INET, BSD_SOCKADDR_IN_LEN, BSD_SOCK_DGRAM};
use crate::cpu::Cpu;
use crate::trace::Level;
use crate::Result;

/// `sfdnsres` failures: the definitive ones, not try-again, in FreeBSD's positive numbering.
const SFDNSRES_EAI_NONAME: i32 = 8;

const SFDNSRES_HOST_NOT_FOUND: i32 = 1;

/// `EAI_SERVICE`.
const SFDNSRES_EAI_SERVICE: i32 = 9;

/// `AI_NUMERICHOST`, `SOCK_STREAM`, `IPPROTO_TCP` and `IPPROTO_UDP` in FreeBSD's numbering.
const AI_NUMERICHOST: u32 = 0x4;

const SOCK_STREAM: u32 = 1;

const IPPROTO_TCP: u32 = 6;

const IPPROTO_UDP: u32 = 17;

/// First word of every serialized `addrinfo`.
const ADDRINFO_MAGIC: u32 = 0xBEEF_CAFE;

/// The fields of a serialized `addrinfo` that `getaddrinfo`'s hints carry.
struct AddrInfoHints {
    flags: u32,
    family: u32,
    socktype: u32,
}

impl AddrInfoHints {
    /// All zero, which asks for anything, when there are no hints or they are malformed.
    fn parse(raw: &[u8]) -> AddrInfoHints {
        let word = |index: usize| {
            raw.get(index * 4..index * 4 + 4)
                .map_or(0, |b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
        };
        if word(0) != ADDRINFO_MAGIC {
            return AddrInfoHints {
                flags: 0,
                family: 0,
                socktype: 0,
            };
        }
        AddrInfoHints {
            flags: word(1),
            family: word(2),
            socktype: word(3),
        }
    }
}

/// A numeric port, or one of the few service names a title passes; `None` for any other.
fn service_port(service: &str) -> Option<u16> {
    match service {
        "" => Some(0),
        "http" => Some(80),
        "https" => Some(443),
        _ => service.parse().ok(),
    }
}

/// A DNS name the host could resolve: dot-separated labels of letters, digits, `-` and `_`.
fn is_hostname(name: &str) -> bool {
    let name = name.strip_suffix('.').unwrap_or(name);
    !name.is_empty()
        && name.len() <= 253
        && name.split('.').all(|label| {
            (1..=63).contains(&label.len())
                && label
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        })
}

/// Append one `addrinfo` entry to the `sfdnsres` wire format: big-endian words, then a
/// `sockaddr_in` and an empty canonical name. libnx copies that `sockaddr_in` into a FreeBSD
/// one, `{ u8 len, u8 family, u16 port, u32 address, 8 zero bytes }`, then byte-swaps the port
/// and address ("Nintendo just byteswaps everything recursively"), so those two go on the wire
/// little-endian. The family is a byte at offset 1, not a 16-bit word: written as one, a guest
/// connected with family 0.
fn serialize_addrinfo(out: &mut Vec<u8>, socktype: u32, protocol: u32, ip: [u8; 4], port: u16) {
    for word in [
        ADDRINFO_MAGIC,
        0,
        u32::from(BSD_AF_INET),
        socktype,
        protocol,
        BSD_SOCKADDR_IN_LEN as u32,
    ] {
        out.extend_from_slice(&word.to_be_bytes());
    }
    out.extend_from_slice(&[BSD_SOCKADDR_IN_LEN as u8, BSD_AF_INET]);
    out.extend_from_slice(&port.to_le_bytes());
    out.extend_from_slice(&u32::from_be_bytes(ip).to_le_bytes());
    out.extend_from_slice(&[0; 8]);
    out.push(0);
}

/// The order of a lookup's three reply words, which depends on the request.
#[derive(Clone, Copy)]
enum Reply {
    /// `GetHostByName`/`GetHostByAddr`: `{ h_errno, errno, size }`.
    Host,
    /// `GetAddrInfo`/`GetNameInfo`: `{ errno, gai error, size }`. Minecraft's SDK reads word 1
    /// as the result, so an error in word 0 read as success with an empty list.
    AddrInfo,
    /// The WithOptions forms: `{ size, error, errno }`. Asphalt 9's
    /// `GetHostByNameRequestWithOptions` took a plain order as a failure every time and retried.
    WithOptions,
}

impl Reply {
    fn host(with_options: bool) -> Reply {
        if with_options {
            Reply::WithOptions
        } else {
            Reply::Host
        }
    }

    fn addr_info(with_options: bool) -> Reply {
        if with_options {
            Reply::WithOptions
        } else {
            Reply::AddrInfo
        }
    }
}

/// A lookup's reply words, errno always 0.
fn results(reply: Reply, error: u32, size: u32) -> [u8; 12] {
    let words = match reply {
        Reply::Host => [error, 0, size],
        Reply::AddrInfo => [0, error, size],
        Reply::WithOptions => [size, error, 0],
    };
    let mut out = [0u8; 12];
    for (slot, word) in out.chunks_mut(4).zip(words) {
        slot.copy_from_slice(&word.to_le_bytes());
    }
    out
}

impl Cpu {
    /// `sfdnsres` (`IResolver`): nothing resolves. Every lookup fails definitively,
    /// `EAI_NONAME` or `HOST_NOT_FOUND`, including numeric addresses.
    pub(crate) fn sfdnsres_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return self.write_ipc_response(tls, 0, &[], &0x1000u16.to_le_bytes(), &[]);
        }
        match cmd_id {
            // GetHostByNameRequest / GetHostByAddrRequest and WithOptions forms, once the host
            // carries traffic; `h_errno` otherwise.
            Some(2) | Some(10) if self.net.enabled => {
                self.sfdnsres_host_by_name(tls, cmd_id == Some(10))
            }
            Some(3) | Some(11) if self.net.enabled => {
                self.sfdnsres_host_by_address(tls, cmd_id == Some(11))
            }
            Some(2) | Some(3) | Some(10) | Some(11) => {
                let with_options = matches!(cmd_id, Some(10 | 11));
                self.sfdnsres_failure(tls, SFDNSRES_HOST_NOT_FOUND, Reply::host(with_options))
            }
            // GetAddrInfoRequest and its WithOptions form, once the host carries traffic.
            Some(6) | Some(12) if self.net.enabled => {
                self.sfdnsres_addrinfo(tls, cmd_id == Some(12))
            }
            // GetAddrInfoRequest / GetNameInfoRequest and WithOptions forms (`gai` error).
            Some(6) | Some(7) | Some(12) | Some(13) => {
                let with_options = matches!(cmd_id, Some(12 | 13));
                self.sfdnsres_failure(tls, SFDNSRES_EAI_NONAME, Reply::addr_info(with_options))
            }
            // GetHostStringErrorRequest / GetGaiStringErrorRequest.
            Some(4) | Some(5) => {
                let message: &[u8] = b"Name or service not known\0";
                if let Some((addr, size)) = self.ipc_output_buffer(tls, 0) {
                    if addr != 0 {
                        for (index, &byte) in message.iter().take(size as usize).enumerate() {
                            self.mem.write_u8(addr.wrapping_add(index as u32), byte)?;
                        }
                    }
                }
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // RequestCancelHandleRequest -> u32 token.
            Some(8) => {
                let handle = self.next_object_id;
                self.next_object_id = handle.wrapping_add(1);
                self.write_ipc_response(tls, 0, &[], &handle.to_le_bytes(), &[])
            }
            // CancelRequest, and the resolver options.
            Some(9) | Some(14) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            Some(15) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            _ => self.unimplemented_command(tls, "sfdnsres", cmd_id),
        }
    }

    /// `nsd:a`/`nsd:u`: name resolution for the production environment, `lp1`.
    pub(crate) fn nsd_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const FQDN_SIZE: u32 = 0x100;
        if self.ipc_answer_control(tls, handle, "nsd", cmd_id)? {
            return Ok(());
        }
        match cmd_id {
            // Resolve / ResolveEx(buffer FQDN in, buffer FQDN out); Ex adds an inner result.
            Some(20) | Some(21) => {
                let fqdn = self
                    .ipc_input_buffer(tls, 0)
                    .map(|(addr, size)| self.read_string(addr, size.min(FQDN_SIZE)))
                    .unwrap_or_default();
                let mut resolved = fqdn.replace('%', "lp1").into_bytes();
                resolved.resize(FQDN_SIZE as usize, 0);
                self.write_output_buffer(tls, 0, &resolved);
                let data: &[u8] = if cmd_id == Some(21) { &[0; 4] } else { &[] };
                self.write_ipc_response(tls, 0, &[], data, &[])
            }
            // SetChangeEnvironmentIdentifierDisabled: no environment here changes.
            Some(15) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            _ => self.unimplemented_command(tls, "nsd", cmd_id),
        }
    }

    /// `GetAddrInfoRequest` with a network: a literal address answers as itself, a name as
    /// its synthetic address, which `bsd` maps back to the name when the guest connects.
    fn sfdnsres_addrinfo(&mut self, tls: u32, with_options: bool) -> Result<()> {
        let read = |cpu: &Cpu, index: u32| {
            cpu.ipc_input_buffer(tls, index)
                .map(|(addr, size)| cpu.read_bytes(addr, size))
                .unwrap_or_default()
        };
        let text = |bytes: Vec<u8>| {
            let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
            String::from_utf8_lossy(&bytes[..end]).into_owned()
        };
        let host = text(read(self, 0));
        let service = text(read(self, 1));
        let hints = AddrInfoHints::parse(&read(self, 2));

        let asked = format!(
            "{host}:{service} (family {}, type {})",
            hints.family, hints.socktype
        );
        let Some(port) = service_port(&service) else {
            self.log(
                Level::Info,
                &format!("[net] getaddrinfo {asked}: unknown service"),
            );
            return self.sfdnsres_failure(
                tls,
                SFDNSRES_EAI_SERVICE,
                Reply::addr_info(with_options),
            );
        };
        if hints.family != 0 && hints.family != u32::from(BSD_AF_INET) {
            self.log(Level::Info, &format!("[net] getaddrinfo {asked}: not IPv4"));
            return self.sfdnsres_failure(tls, SFDNSRES_EAI_NONAME, Reply::addr_info(with_options));
        }
        let literal = host
            .parse::<std::net::Ipv4Addr>()
            .ok()
            .map(|ip| ip.octets());
        let address = match literal {
            Some(address) => Some(address),
            None if hints.flags & AI_NUMERICHOST != 0 || !is_hostname(&host) => None,
            None => self.net.synthesize(&host),
        };
        let Some(address) = address else {
            self.log(
                Level::Info,
                &format!("[net] getaddrinfo {asked}: not a name"),
            );
            return self.sfdnsres_failure(tls, SFDNSRES_EAI_NONAME, Reply::addr_info(with_options));
        };
        let [a, b, c, d] = address;
        self.log(
            Level::Info,
            &format!("[net] getaddrinfo {asked} -> {a}.{b}.{c}.{d}"),
        );

        let mut serialized = Vec::new();
        for (socktype, protocol) in [(SOCK_STREAM, IPPROTO_TCP), (BSD_SOCK_DGRAM, IPPROTO_UDP)] {
            if hints.socktype == 0 || hints.socktype == socktype {
                serialize_addrinfo(&mut serialized, socktype, protocol, address, port);
            }
        }
        serialized.extend_from_slice(&0u32.to_be_bytes());
        let fits = self
            .ipc_output_buffer(tls, 0)
            .is_some_and(|(_, size)| serialized.len() <= size as usize);
        if !fits {
            return self.sfdnsres_failure(tls, SFDNSRES_EAI_NONAME, Reply::addr_info(with_options));
        }
        self.write_output_buffer(tls, 0, &serialized);
        let results = results(Reply::addr_info(with_options), 0, serialized.len() as u32);
        self.write_ipc_response(tls, 0, &[], &results, &[])
    }

    /// `GetHostByNameRequest`, and its WithOptions form, with a network: a literal address
    /// answers as itself, a name as its synthetic address, as `GetAddrInfoRequest` does. The
    /// name is the first input buffer (the options, when there are any, the second), the
    /// serialized `hostent` the first output buffer.
    fn sfdnsres_host_by_name(&mut self, tls: u32, with_options: bool) -> Result<()> {
        let name = self
            .ipc_input_buffer(tls, 0)
            .map(|(addr, size)| self.read_bytes(addr, size))
            .unwrap_or_default();
        let end = name.iter().position(|&b| b == 0).unwrap_or(name.len());
        let name = String::from_utf8_lossy(&name[..end]).into_owned();
        let literal = name
            .parse::<std::net::Ipv4Addr>()
            .ok()
            .map(|ip| ip.octets());
        let address = match literal {
            Some(address) => Some(address),
            None if is_hostname(&name) => self.net.synthesize(&name),
            None => None,
        };
        let Some(address) = address else {
            self.log(
                Level::Info,
                &format!("[net] gethostbyname {name}: not a name"),
            );
            return self.sfdnsres_host_failure(tls, with_options);
        };
        let [a, b, c, d] = address;
        self.log(
            Level::Info,
            &format!("[net] gethostbyname {name} -> {a}.{b}.{c}.{d}"),
        );
        self.sfdnsres_hostent(tls, with_options, &name, address)
    }

    /// `GetHostByAddrRequest`, and its WithOptions form, with a network: a synthetic address
    /// names the host it stands for. Anything else would need a reverse lookup the host does
    /// not offer, and is not found.
    fn sfdnsres_host_by_address(&mut self, tls: u32, with_options: bool) -> Result<()> {
        let raw = self
            .ipc_input_buffer(tls, 0)
            .map(|(addr, size)| self.read_bytes(addr, size.min(4)))
            .unwrap_or_default();
        let address: Option<[u8; 4]> = raw.try_into().ok();
        let name = address.and_then(|ip| self.net.name_of(ip).map(str::to_owned));
        let (Some(address), Some(name)) = (address, name) else {
            return self.sfdnsres_host_failure(tls, with_options);
        };
        self.sfdnsres_hostent(tls, with_options, &name, address)
    }

    /// Answer a host lookup with one address. The serialized `hostent` is libnx's
    /// `_resolverDeserializeHostent` format: the name and its NUL, a big-endian alias count (no
    /// aliases), the big-endian 16-bit family and address length, a big-endian address count,
    /// then each address as a little-endian word, since libnx byte-swaps it into `s_addr`.
    fn sfdnsres_hostent(
        &mut self,
        tls: u32,
        with_options: bool,
        name: &str,
        address: [u8; 4],
    ) -> Result<()> {
        let mut serialized = name.as_bytes().to_vec();
        serialized.push(0);
        serialized.extend_from_slice(&0u32.to_be_bytes());
        serialized.extend_from_slice(&u16::from(BSD_AF_INET).to_be_bytes());
        serialized.extend_from_slice(&4u16.to_be_bytes());
        serialized.extend_from_slice(&1u32.to_be_bytes());
        serialized.extend_from_slice(&u32::from_be_bytes(address).to_le_bytes());
        let fits = self
            .ipc_output_buffer(tls, 0)
            .is_some_and(|(_, size)| serialized.len() <= size as usize);
        if !fits {
            return self.sfdnsres_host_failure(tls, with_options);
        }
        self.write_output_buffer(tls, 0, &serialized);
        let results = results(Reply::host(with_options), 0, serialized.len() as u32);
        self.write_ipc_response(tls, 0, &[], &results, &[])
    }

    fn sfdnsres_host_failure(&mut self, tls: u32, with_options: bool) -> Result<()> {
        self.sfdnsres_failure(tls, SFDNSRES_HOST_NOT_FOUND, Reply::host(with_options))
    }

    /// A failed lookup: the error in its word, errno 0, nothing written. See [`results`].
    fn sfdnsres_failure(&mut self, tls: u32, error: i32, reply: Reply) -> Result<()> {
        let results = results(reply, error as u32, 0);
        self.write_ipc_response(tls, 0, &[], &results, &[])
    }
}

#[cfg(test)]
mod tests {
    use crate::cpu::Cpu;
    use crate::kernel::ipc::testing::*;

    #[test]
    fn nsd_resolves_the_environment_placeholder() {
        const IN: u32 = 0x1000;
        const OUT: u32 = 0x1200;
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        cpu.mem.map_zero(IN, 0x400).unwrap();
        cpu.mem.write_bytes(IN, b"%.nintendo.net\0").unwrap();
        write_buffer_request(&mut cpu, 21, &[], &[(IN, 0x100)], &[(OUT, 0x100)]);
        cpu.register_service_handle(9, "nsd:a");
        cpu.nsd_request(TLS, 9, Some(21)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0);
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "inner result");
        assert_eq!(cpu.read_string(OUT, 0x100), "lp1.nintendo.net");
    }

    #[test]
    fn sfdnsres_fails_every_lookup_definitively() {
        // getaddrinfo: EAI_NONAME, not EAI_AGAIN, in the second word; a zero there with an empty
        // list is success with no addresses, which Minecraft's resolve handler dereferences.
        for cmd in [6, 7] {
            let mut cpu = request(false, cmd, &[]);
            cpu.sfdnsres_request(TLS, Some(cmd)).unwrap();
            assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "errno");
            assert_eq!(
                cpu.mem.read_u32(TLS + 0x24).unwrap() as i32,
                super::SFDNSRES_EAI_NONAME
            );
            assert_eq!(cpu.mem.read_u32(TLS + 0x28).unwrap(), 0, "serialized size");
        }

        let mut cpu = request(false, 12, &[]);
        cpu.sfdnsres_request(TLS, Some(12)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "serialized size");
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x24).unwrap() as i32,
            super::SFDNSRES_EAI_NONAME
        );

        // gethostbyname reports through h_errno, which has its own numbering.
        let mut cpu = request(false, 2, &[]);
        cpu.sfdnsres_request(TLS, Some(2)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x20).unwrap() as i32,
            super::SFDNSRES_HOST_NOT_FOUND
        );
    }

    #[test]
    fn sfdnsres_explains_the_failure_it_reports() {
        const BUFFER: u32 = 0x4000;
        let mut cpu = request_with_recv_buffer(5, &[], BUFFER, 0x40);
        cpu.mem.map_zero(BUFFER, 0x100).unwrap();
        cpu.sfdnsres_request(TLS, Some(5)).unwrap();
        assert_eq!(cpu.read_string(BUFFER, 0x40), "Name or service not known");
    }

    const LOOKUP_HOST: u32 = 0x5000;

    const LOOKUP_PORT: u32 = 0x5100;

    const LOOKUP_HINTS: u32 = 0x5200;

    const LOOKUP_OUT: u32 = 0x5800;

    /// `getaddrinfo(host, port, hints)` with the network on; the reply words and the output.
    fn lookup(host: &str, port: &str, socktype: Option<u32>) -> ((i32, u32), Vec<u8>) {
        let mut cpu = Cpu::new();
        cpu.net.enabled = true;
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        cpu.mem.map_zero(LOOKUP_HOST, 0x2000).unwrap();
        cpu.mem
            .write_bytes(LOOKUP_HOST, format!("{host}\0").as_bytes())
            .unwrap();
        cpu.mem
            .write_bytes(LOOKUP_PORT, format!("{port}\0").as_bytes())
            .unwrap();
        let mut hints = Vec::new();
        if let Some(socktype) = socktype {
            for word in [super::ADDRINFO_MAGIC, 0, 0, socktype, 0, 0] {
                hints.extend_from_slice(&word.to_be_bytes());
            }
        }
        cpu.mem.write_bytes(LOOKUP_HINTS, &hints).unwrap();
        let send = [
            (LOOKUP_HOST, 0x100),
            (LOOKUP_PORT, 0x100),
            (LOOKUP_HINTS, 0x400),
        ];
        write_buffer_request(&mut cpu, 6, &[0; 16], &send, &[(LOOKUP_OUT, 0x1000)]);
        cpu.sfdnsres_request(TLS, Some(6)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "errno");
        let ret = cpu.mem.read_u32(TLS + 0x24).unwrap() as i32;
        let len = cpu.mem.read_u32(TLS + 0x28).unwrap();
        ((ret, len), cpu.read_bytes(LOOKUP_OUT, len))
    }

    /// The `(socktype, protocol, address, port)` of each entry in a serialized chain.
    fn entries(chain: &[u8]) -> Vec<(u32, u32, [u8; 4], u16)> {
        let word = |at: usize| u32::from_be_bytes(chain[at..at + 4].try_into().unwrap());
        let mut at = 0;
        let mut found = Vec::new();
        while word(at) != 0 {
            assert_eq!(word(at), super::ADDRINFO_MAGIC);
            assert_eq!(word(at + 8), 2, "AF_INET");
            assert_eq!(word(at + 20), 16, "ai_addrlen");
            let addr = at + 24;
            assert_eq!(chain[addr..addr + 2], [16, 2], "sin_len and sin_family");
            let port = u16::from_le_bytes([chain[addr + 2], chain[addr + 3]]);
            let ip =
                u32::from_le_bytes(chain[addr + 4..addr + 8].try_into().unwrap()).to_be_bytes();
            found.push((word(at + 12), word(at + 16), ip, port));
            assert_eq!(chain[addr + 16], 0, "empty canonical name");
            at = addr + 17;
        }
        assert_eq!(at + 4, chain.len(), "the list ends with its sentinel");
        found
    }

    #[test]
    fn sfdnsres_answers_a_name_with_the_same_synthetic_address_each_time() {
        let ((ret, _), chain) = lookup("Example.com", "443", None);
        assert_eq!(ret, 0);
        let found = entries(&chain);
        assert_eq!(found.len(), 2, "stream and datagram");
        assert_eq!((found[0].0, found[0].1, found[0].3), (1, 6, 443));
        assert_eq!((found[1].0, found[1].1, found[1].3), (2, 17, 443));
        assert_eq!(found[0].2[..2], [198, 18]);
        assert_eq!(found[0].2, found[1].2);
        let (_, again) = lookup("example.com", "443", None);
        assert_eq!(entries(&again)[0].2, found[0].2);
    }

    #[test]
    fn sfdnsres_honours_the_hinted_socket_type_and_literal_addresses() {
        let (_, chain) = lookup("1.2.3.4", "http", Some(1));
        assert_eq!(entries(&chain), vec![(1, 6, [1, 2, 3, 4], 80)]);
        let (_, chain) = lookup("1.2.3.4", "", Some(2));
        assert_eq!(entries(&chain), vec![(2, 17, [1, 2, 3, 4], 0)]);
    }

    #[test]
    fn sfdnsres_refuses_what_is_not_a_name_or_a_service() {
        assert_eq!(lookup("", "80", None).0 .0, super::SFDNSRES_EAI_NONAME);
        assert_eq!(
            lookup("bad name", "80", None).0 .0,
            super::SFDNSRES_EAI_NONAME
        );
        assert_eq!(lookup("a..b", "80", None).0 .0, super::SFDNSRES_EAI_NONAME);
        assert_eq!(
            lookup("example.com", "gopher", None).0 .0,
            super::SFDNSRES_EAI_SERVICE
        );
    }

    /// `gethostbyname(name)` with the network on, as command `cmd` (2, or 10 with an options
    /// buffer); the three reply words and the serialized `hostent`.
    fn host_by_name(cpu: &mut Cpu, name: &str, cmd: u32) -> ([u32; 3], Vec<u8>) {
        cpu.mem
            .write_bytes(LOOKUP_HOST, format!("{name}\0").as_bytes())
            .unwrap();
        let mut send = vec![(LOOKUP_HOST, 0x100)];
        if cmd == 10 {
            send.push((LOOKUP_HINTS, 0x10));
        }
        write_buffer_request(cpu, cmd, &[0; 16], &send, &[(LOOKUP_OUT, 0x100)]);
        cpu.sfdnsres_request(TLS, Some(cmd)).unwrap();
        let word = |at: u32| cpu.mem.read_u32(TLS + at).unwrap();
        let words = [word(0x20), word(0x24), word(0x28)];
        let size = if cmd == 10 { words[0] } else { words[2] };
        (words, cpu.read_bytes(LOOKUP_OUT, size))
    }

    fn networked() -> Cpu {
        let mut cpu = Cpu::new();
        cpu.net.enabled = true;
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        cpu.mem.map_zero(LOOKUP_HOST, 0x2000).unwrap();
        cpu
    }

    /// The name and the single IPv4 address of a serialized `hostent`, read as libnx's
    /// `_resolverDeserializeHostent` does.
    fn hostent(raw: &[u8]) -> (String, [u8; 4]) {
        let end = raw.iter().position(|&b| b == 0).unwrap();
        let name = String::from_utf8(raw[..end].to_vec()).unwrap();
        let mut at = end + 1;
        let be32 = |at: usize| u32::from_be_bytes(raw[at..at + 4].try_into().unwrap());
        assert_eq!(be32(at), 0, "no aliases");
        at += 4;
        assert_eq!(raw[at..at + 4], [0, 2, 0, 4], "AF_INET, 4 bytes");
        at += 4;
        assert_eq!(be32(at), 1, "one address");
        at += 4;
        // libnx reads each address as a native word and byte-swaps it into `s_addr`.
        let word = u32::from_le_bytes(raw[at..at + 4].try_into().unwrap());
        assert_eq!(at + 4, raw.len());
        (name, word.to_be_bytes())
    }

    #[test]
    fn gethostbyname_answers_a_name_with_its_synthetic_address() {
        let mut cpu = networked();
        let expected = cpu.net.synthesize("server.example").unwrap();
        let (words, raw) = host_by_name(&mut cpu, "server.example", 2);
        assert_eq!(words[..2], [0, 0], "h_errno and errno");
        assert_eq!(hostent(&raw), ("server.example".into(), expected));

        let (_, raw) = host_by_name(&mut cpu, "1.2.3.4", 2);
        assert_eq!(hostent(&raw).1, [1, 2, 3, 4], "a literal is itself");

        let (words, _) = host_by_name(&mut cpu, "not a name", 2);
        assert_eq!(words, [super::SFDNSRES_HOST_NOT_FOUND as u32, 0, 0]);
    }

    #[test]
    fn the_with_options_forms_put_the_size_first() {
        let mut cpu = networked();
        let (words, raw) = host_by_name(&mut cpu, "server.example", 10);
        assert_eq!(words, [raw.len() as u32, 0, 0]);
        assert_eq!(hostent(&raw).0, "server.example");

        let (words, _) = host_by_name(&mut cpu, "not a name", 10);
        assert_eq!(words, [0, super::SFDNSRES_HOST_NOT_FOUND as u32, 0]);
    }

    #[test]
    fn gethostbyaddr_names_a_synthetic_address_and_nothing_else() {
        let mut cpu = networked();
        let address = cpu.net.synthesize("server.example").unwrap();
        for (ip, found) in [(address, true), ([1, 2, 3, 4], false)] {
            cpu.mem.write_bytes(LOOKUP_HOST, &ip).unwrap();
            write_buffer_request(
                &mut cpu,
                3,
                &[0; 24],
                &[(LOOKUP_HOST, 4)],
                &[(LOOKUP_OUT, 0x100)],
            );
            cpu.sfdnsres_request(TLS, Some(3)).unwrap();
            let h_errno = cpu.mem.read_u32(TLS + 0x20).unwrap();
            if found {
                let size = cpu.mem.read_u32(TLS + 0x28).unwrap();
                assert_eq!(h_errno, 0);
                let raw = cpu.read_bytes(LOOKUP_OUT, size);
                assert_eq!(hostent(&raw), ("server.example".into(), address));
            } else {
                assert_eq!(h_errno, super::SFDNSRES_HOST_NOT_FOUND as u32);
            }
        }
    }
}
