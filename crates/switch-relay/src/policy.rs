//! Who may use the relay and where it may connect.

use std::collections::HashMap;
use std::io::BufRead;
use std::net::Ipv4Addr;

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

    /// Whether the relay may connect to a resolved address.
    pub fn address_allowed(&self, address: Ipv4Addr) -> bool {
        self.allow_private || is_public(address)
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
            assert!(policy.address_allowed(allowed.into()), "{allowed:?}");
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
            assert!(!policy.address_allowed(refused.into()), "{refused:?}");
        }
    }

    #[test]
    fn private_addresses_can_be_allowed_on_request() {
        let policy = Policy::new(Vec::new(), Unknown::Deny, true);
        assert!(policy.address_allowed([127, 0, 0, 1].into()));
    }
}
