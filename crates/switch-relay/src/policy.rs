//! Who may use the relay and where it may connect.

use std::collections::HashMap;
use std::io::BufRead;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

use tokio::sync::Mutex;

/// Origins that may use the relay without being asked: the deployed emulator and its dev server.
pub const TRUSTED_ORIGINS: [&str; 2] = ["http://localhost:8000", "https://tenshii.moe"];

/// What to do with an origin that is not trusted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unknown {
    /// Ask on the terminal, once per origin until the relay exits.
    Ask,
    Deny,
}

pub struct Policy {
    trusted: Vec<String>,
    unknown: Unknown,
    allow_private: bool,
    /// Answers given to unknown origins. Held across the question, so one is asked at a time.
    decided: Mutex<HashMap<String, bool>>,
}

impl Policy {
    pub fn new(extra_origins: Vec<String>, unknown: Unknown, allow_private: bool) -> Policy {
        let trusted = TRUSTED_ORIGINS
            .iter()
            .map(|origin| (*origin).to_owned())
            .chain(extra_origins)
            .map(|origin| origin.to_ascii_lowercase())
            .collect();
        Policy {
            trusted,
            unknown,
            allow_private,
            decided: Mutex::new(HashMap::new()),
        }
    }

    /// Whether a page with this `Origin` header may connect. A client with none is not a browser.
    pub async fn origin_allowed(&self, origin: Option<&str>) -> bool {
        let Some(origin) = origin.map(str::to_ascii_lowercase) else {
            return false;
        };
        if self.trusted.contains(&origin) {
            return true;
        }
        let mut decided = self.decided.lock().await;
        if let Some(&answer) = decided.get(&origin) {
            return answer;
        }
        let answer = match self.unknown {
            Unknown::Ask => ask(&origin).await,
            Unknown::Deny => false,
        };
        decided.insert(origin, answer);
        answer
    }

    /// Whether the relay may reach a resolved address.
    pub fn address_allowed(&self, address: impl Into<IpAddr>) -> bool {
        self.allow_private
            || match address.into() {
                IpAddr::V4(address) => is_public(address),
                IpAddr::V6(address) => is_public_v6(address),
            }
    }
}

async fn ask(origin: &str) -> bool {
    eprint!("Allow {origin} to use this relay? [y/N] ");
    let answer = tokio::task::spawn_blocking(|| {
        let mut line = String::new();
        std::io::stdin().lock().read_line(&mut line).ok()?;
        Some(line)
    })
    .await
    .ok()
    .flatten();
    answer.is_some_and(|line| matches!(line.trim(), "y" | "Y" | "yes"))
}

/// An address the Internet routes: not this host, not a local network, not reserved.
fn is_public(address: Ipv4Addr) -> bool {
    let [a, b, c, _] = address.octets();
    !(address.is_unspecified()
        || address.is_loopback()
        || address.is_private()
        || address.is_link_local()
        || address.is_multicast()
        || address.is_broadcast()
        || a == 0
        // Carrier-grade NAT, 100.64.0.0/10.
        || (a == 100 && (64..128).contains(&b))
        // IETF protocol assignments, 192.0.0.0/24.
        || (a == 192 && b == 0 && c == 0)
        // Benchmarking, 198.18.0.0/15: the console's synthetic addresses live here.
        || (a == 198 && (18..20).contains(&b))
        // Reserved, 240.0.0.0/4.
        || a >= 240)
}

/// The IPv6 counterpart of [`is_public`]. An address that carries an IPv4 one (mapped, NAT64,
/// 6to4) is judged by that address, since that is where the packets end up.
fn is_public_v6(address: Ipv6Addr) -> bool {
    if let Some(v4) = address.to_ipv4_mapped() {
        return is_public(v4);
    }
    let segments = address.segments();
    let octets = address.octets();
    let embedded =
        |at: usize| Ipv4Addr::new(octets[at], octets[at + 1], octets[at + 2], octets[at + 3]);
    match segments {
        // NAT64's well-known prefix, 64:ff9b::/96.
        [0x64, 0xff9b, 0, 0, 0, 0, _, _] => return is_public(embedded(12)),
        // 6to4, 2002::/16, with the IPv4 address in the next 32 bits.
        [0x2002, ..] => return is_public(embedded(2)),
        _ => {}
    }
    !(address.is_unspecified()
        || address.is_loopback()
        || address.is_multicast()
        // IPv4-compatible, ::/96, deprecated.
        || segments[..6] == [0; 6]
        // Discard-only, 100::/64.
        || segments[..4] == [0x100, 0, 0, 0]
        // Unique local, fc00::/7.
        || (segments[0] & 0xfe00) == 0xfc00
        // Link-local, fe80::/10, and the deprecated site-local, fec0::/10.
        || (segments[0] & 0xffc0) == 0xfe80
        || (segments[0] & 0xffc0) == 0xfec0
        // IETF protocol assignments, 2001::/23, except Teredo (2001::/32), which is routed.
        || (segments[0] == 0x2001 && segments[1] < 0x200 && segments[1] != 0)
        // Documentation, 2001:db8::/32 and 3fff::/20.
        || (segments[0] == 0x2001 && segments[1] == 0xdb8)
        || (segments[0] & 0xfff0) == 0x3ff0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn closed() -> Policy {
        Policy::new(vec!["https://Example.org".into()], Unknown::Deny, false)
    }

    #[tokio::test]
    async fn the_emulators_own_origins_are_trusted() {
        let policy = closed();
        assert!(policy.origin_allowed(Some("https://tenshii.moe")).await);
        assert!(policy.origin_allowed(Some("http://localhost:8000")).await);
        assert!(policy.origin_allowed(Some("HTTPS://TENSHII.MOE")).await);
    }

    #[tokio::test]
    async fn an_origin_given_on_the_command_line_is_trusted_too() {
        assert!(closed().origin_allowed(Some("https://example.org")).await);
    }

    #[tokio::test]
    async fn other_origins_and_clients_without_one_are_refused() {
        let policy = closed();
        assert!(!policy.origin_allowed(Some("https://evil.example")).await);
        assert!(!policy.origin_allowed(Some("http://localhost:8001")).await);
        assert!(
            !policy
                .origin_allowed(Some("https://tenshii.moe.evil.example"))
                .await
        );
        assert!(!policy.origin_allowed(None).await);
    }

    #[test]
    fn only_routable_addresses_are_allowed() {
        let policy = closed();
        for allowed in [
            [1, 1, 1, 1],
            [8, 8, 8, 8],
            [93, 184, 216, 34],
            [172, 32, 0, 1],
        ] {
            assert!(
                policy.address_allowed(Ipv4Addr::from(allowed)),
                "{allowed:?}"
            );
        }
        for refused in [
            [127, 0, 0, 1],
            [0, 0, 0, 0],
            [10, 1, 2, 3],
            [172, 16, 0, 1],
            [172, 31, 255, 255],
            [192, 168, 1, 1],
            [169, 254, 1, 1],
            [100, 64, 0, 1],
            [100, 127, 255, 255],
            [192, 0, 0, 1],
            [198, 18, 0, 1],
            [198, 19, 255, 255],
            [224, 0, 0, 1],
            [240, 0, 0, 1],
            [255, 255, 255, 255],
        ] {
            assert!(
                !policy.address_allowed(Ipv4Addr::from(refused)),
                "{refused:?}"
            );
        }
    }

    #[test]
    fn only_routable_ipv6_addresses_are_allowed() {
        let policy = closed();
        for allowed in [
            "2606:4700:4700::1111",
            "2a00:1450:4001:82a::200e",
            "2001:0:4136:e378::1",
            "64:ff9b::808:808",
            "2002:0808:0808::1",
            "::ffff:8.8.8.8",
        ] {
            let address: Ipv6Addr = allowed.parse().unwrap();
            assert!(policy.address_allowed(address), "{allowed}");
        }
        for refused in [
            "::",
            "::1",
            "fe80::1",
            "fec0::1",
            "fc00::1",
            "fd12:3456::1",
            "ff02::1",
            "2001:db8::1",
            "3fff::1",
            "100::1",
            "2001:2::1",
            "::ffff:127.0.0.1",
            "::ffff:192.168.1.1",
            "64:ff9b::a00:1",
            "2002:c0a8:0101::1",
            "::127.0.0.1",
        ] {
            let address: Ipv6Addr = refused.parse().unwrap();
            assert!(!policy.address_allowed(address), "{refused}");
        }
    }

    #[test]
    fn private_addresses_can_be_allowed_on_request() {
        let policy = Policy::new(Vec::new(), Unknown::Deny, true);
        assert!(policy.address_allowed(Ipv4Addr::LOCALHOST));
    }
}
