//! `sfdnsres` and `nsd`: name resolution. With the network off nothing resolves.

use super::bsd::{BSD_AF_INET, BSD_SOCKADDR_IN_LEN, BSD_SOCK_DGRAM};
use crate::cpu::Cpu;
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
/// `sockaddr_in` of family (16 bits), port, address and 8 zero bytes, then an empty canonical name.
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
    out.extend_from_slice(&u16::from(BSD_AF_INET).to_be_bytes());
    out.extend_from_slice(&port.to_be_bytes());
    out.extend_from_slice(&ip);
    out.extend_from_slice(&[0; 8]);
    out.push(0);
}

impl Cpu {
    /// `sfdnsres` (`IResolver`): nothing resolves. Every lookup fails definitively,
    /// `EAI_NONAME` or `HOST_NOT_FOUND`, including numeric addresses.
    pub(crate) fn sfdnsres_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return self.write_ipc_response(tls, 0, &[], &0x1000u16.to_le_bytes(), &[]);
        }
        match cmd_id {
            // GetHostByNameRequest / GetHostByAddrRequest and WithOptions forms (`h_errno`).
            Some(2) | Some(3) | Some(10) | Some(11) => {
                self.sfdnsres_failure(tls, SFDNSRES_HOST_NOT_FOUND)
            }
            // GetAddrInfoRequest and its WithOptions form, once the host carries traffic.
            Some(6) | Some(12) if self.net.enabled => self.sfdnsres_addrinfo(tls),
            // GetAddrInfoRequest / GetNameInfoRequest and WithOptions forms (`gai` error).
            Some(6) | Some(7) | Some(12) | Some(13) => {
                self.sfdnsres_failure(tls, SFDNSRES_EAI_NONAME)
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
    fn sfdnsres_addrinfo(&mut self, tls: u32) -> Result<()> {
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

        let Some(port) = service_port(&service) else {
            return self.sfdnsres_failure(tls, SFDNSRES_EAI_SERVICE);
        };
        if hints.family != 0 && hints.family != u32::from(BSD_AF_INET) {
            return self.sfdnsres_failure(tls, SFDNSRES_EAI_NONAME);
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
            return self.sfdnsres_failure(tls, SFDNSRES_EAI_NONAME);
        };

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
            return self.sfdnsres_failure(tls, SFDNSRES_EAI_NONAME);
        }
        self.write_output_buffer(tls, 0, &serialized);
        let mut results = [0u8; 12];
        results[8..].copy_from_slice(&(serialized.len() as u32).to_le_bytes());
        self.write_ipc_response(tls, 0, &[], &results, &[])
    }

    /// A failed lookup. `SfdnsresRequestResults` is { return value, errno, bytes
    /// written }; the error goes in the first word, errno stays 0.
    fn sfdnsres_failure(&mut self, tls: u32, error: i32) -> Result<()> {
        let mut results = [0u8; 12];
        results[..4].copy_from_slice(&error.to_le_bytes());
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
        // getaddrinfo: EAI_NONAME, not EAI_AGAIN.
        let mut cpu = request(false, 6, &[]);
        cpu.sfdnsres_request(TLS, Some(6)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x20).unwrap() as i32,
            super::SFDNSRES_EAI_NONAME
        );
        assert_eq!(cpu.mem.read_u32(TLS + 0x24).unwrap(), 0, "errno");
        assert_eq!(cpu.mem.read_u32(TLS + 0x28).unwrap(), 0, "serialized size");

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
        let ret = cpu.mem.read_u32(TLS + 0x20).unwrap() as i32;
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
            assert_eq!(u16::from_be_bytes([chain[addr], chain[addr + 1]]), 2);
            let port = u16::from_be_bytes([chain[addr + 2], chain[addr + 3]]);
            let ip = chain[addr + 4..addr + 8].try_into().unwrap();
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
}
