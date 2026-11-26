//! Who may use the relay and where it may connect.

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::sync::{Mutex, RwLock};

use tokio::sync::watch;

use crate::config::Approval;

/// Origins that may use the relay without being asked: the deployed emulator and its dev server.
pub const TRUSTED_ORIGINS: [&str; 2] = ["http://localhost:8000", "https://tenshii.moe"];

/// Most connections from the origin being asked about that wait for the answer together.
pub const MAX_WAITING: usize = 16;

pub struct Policy {
    /// The built-in origins and those given on the command line, which no page can remove.
    fixed: Vec<String>,
    open: RwLock<Open>,
    /// Who answers for an origin nobody trusted, once per origin until the relay exits.
    unknown: Approval,
    decided: Mutex<HashMap<String, bool>>,
    /// The one question open, if any.
    asking: Mutex<Option<Question>>,
}

/// An origin being asked about, and the connections from it waiting for the answer.
struct Question {
    origin: String,
    waiting: usize,
    answer: watch::Receiver<Option<bool>>,
}

enum Turn {
    Decided(bool),
    /// Another connection from the origin is asking; this one hears its answer.
    Wait(watch::Receiver<Option<bool>>),
    /// This connection asks, and answers the others through the sender.
    Ask(watch::Sender<Option<bool>>),
}

/// Ends the open question however its asker ends, so a dropped one leaves no question behind.
struct Asked<'a> {
    asking: &'a Mutex<Option<Question>>,
    answer: watch::Sender<Option<bool>>,
}

impl Drop for Asked<'_> {
    fn drop(&mut self) {
        *self.asking.lock().unwrap() = None;
        // Waiters that hear nothing take it as no.
        self.answer.send_if_modified(|answer| {
            answer.get_or_insert(false);
            true
        });
    }
}

/// What a page may change while the relay runs.
struct Open {
    origins: Vec<String>,
    allow_private: bool,
    allow_inbound: bool,
}

impl Policy {
    pub fn new(
        extra_origins: Vec<String>,
        unknown: Approval,
        allow_private: bool,
        allow_inbound: bool,
    ) -> Policy {
        let fixed = TRUSTED_ORIGINS
            .iter()
            .map(|origin| (*origin).to_owned())
            .chain(extra_origins)
            .map(|origin| origin.to_ascii_lowercase())
            .collect();
        Policy {
            fixed,
            open: RwLock::new(Open {
                origins: Vec::new(),
                allow_private,
                allow_inbound,
            }),
            unknown,
            decided: Mutex::new(HashMap::new()),
            asking: Mutex::new(None),
        }
    }

    /// The origins trusted for the whole run, whatever the settings say.
    pub fn fixed_origins(&self) -> &[String] {
        &self.fixed
    }

    /// Replace what the settings decide. Streams already open keep what they started with.
    pub fn set(&self, origins: Vec<String>, allow_private: bool, allow_inbound: bool) {
        *self.open.write().unwrap() = Open {
            origins,
            allow_private,
            allow_inbound,
        };
    }

    /// Whether a page with this `Origin` header may connect. A client with none is not a browser.
    ///
    /// One unknown origin is asked about at a time. While it is, more connections from it wait
    /// for the same answer, up to [`MAX_WAITING`], and any other unknown origin is refused
    /// without being remembered, so no page can pile connections up behind a question or queue
    /// question after question. A question nobody answers is no, and is not remembered either.
    pub async fn origin_allowed(&self, origin: Option<&str>) -> bool {
        let Some(origin) = origin.map(str::to_ascii_lowercase) else {
            return false;
        };
        if self.fixed.contains(&origin) || self.open.read().unwrap().origins.contains(&origin) {
            return true;
        }
        let sender = match self.turn(&origin) {
            Turn::Decided(answer) => return answer,
            Turn::Wait(mut answer) => {
                let heard = answer.wait_for(Option::is_some).await.map(|answer| *answer);
                return heard.ok().flatten().unwrap_or(false);
            }
            Turn::Ask(sender) => sender,
        };
        let asked = Asked {
            asking: &self.asking,
            answer: sender,
        };
        let question = format!("Allow pages from {origin} to use this relay?");
        let answer = self.unknown.ask(question).await;
        if let Some(answer) = answer {
            self.decided.lock().unwrap().insert(origin, answer);
        }
        let answer = answer.unwrap_or(false);
        asked.answer.send_replace(Some(answer));
        answer
    }

    /// What a connection from `origin`, which nobody trusts outright, does next.
    fn turn(&self, origin: &str) -> Turn {
        // Taken first, so an answer cannot land between reading `decided` and opening a question.
        let mut asking = self.asking.lock().unwrap();
        if let Some(&answer) = self.decided.lock().unwrap().get(origin) {
            return Turn::Decided(answer);
        }
        match asking.as_mut() {
            Some(question) if question.origin == origin && question.waiting < MAX_WAITING => {
                question.waiting += 1;
                Turn::Wait(question.answer.clone())
            }
            Some(_) => Turn::Decided(false),
            None => {
                let (sender, receiver) = watch::channel(None);
                *asking = Some(Question {
                    origin: origin.to_owned(),
                    waiting: 0,
                    answer: receiver,
                });
                Turn::Ask(sender)
            }
        }
    }

    /// Whether guests may listen for connections and hear from hosts they never sent to.
    pub fn inbound(&self) -> bool {
        self.open.read().unwrap().allow_inbound
    }

    /// Whether guests may reach loopback, local networks and reserved addresses.
    pub fn private(&self) -> bool {
        self.open.read().unwrap().allow_private
    }

    /// Whether the relay may reach a resolved address.
    pub fn address_allowed(&self, address: impl Into<IpAddr>) -> bool {
        self.private()
            || match address.into() {
                IpAddr::V4(address) => is_public(address),
                IpAddr::V6(address) => is_public_v6(address),
            }
    }
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
    use std::sync::Arc;

    use super::*;

    fn closed() -> Policy {
        Policy::new(
            vec!["https://Example.org".into()],
            Approval::Deny,
            false,
            false,
        )
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

    /// Open a question about `origin` as a connection would, and keep it open.
    fn ask_about(policy: &Policy, origin: &str) -> watch::Sender<Option<bool>> {
        match policy.turn(origin) {
            Turn::Ask(sender) => sender,
            _ => panic!("{origin} should be asked about"),
        }
    }

    #[tokio::test]
    async fn while_one_origin_is_asked_about_others_are_refused_and_not_remembered() {
        let policy = closed();
        let open = ask_about(&policy, "https://a.example");
        assert!(!policy.origin_allowed(Some("https://b.example")).await);

        *policy.asking.lock().unwrap() = None;
        drop(open);
        assert!(
            matches!(policy.turn("https://b.example"), Turn::Ask(_)),
            "the refusal was remembered"
        );
    }

    #[tokio::test]
    async fn connections_from_the_origin_asked_about_share_its_answer_up_to_a_limit() {
        let policy = Arc::new(closed());
        let open = ask_about(&policy, "https://a.example");
        let waiting: Vec<_> = (0..MAX_WAITING)
            .map(|_| {
                let policy = policy.clone();
                tokio::spawn(async move { policy.origin_allowed(Some("https://a.example")).await })
            })
            .collect();
        tokio::task::yield_now().await;
        while policy.asking.lock().unwrap().as_ref().unwrap().waiting < MAX_WAITING {
            tokio::task::yield_now().await;
        }
        assert!(
            !policy.origin_allowed(Some("https://a.example")).await,
            "one past the limit waits"
        );
        open.send_replace(Some(true));
        for waiter in waiting {
            assert!(waiter.await.unwrap());
        }
    }

    #[tokio::test]
    async fn an_answer_is_remembered_and_frees_the_question() {
        let policy = Policy::new(Vec::new(), Approval::Allow, false, false);
        assert!(policy.origin_allowed(Some("https://a.example")).await);
        assert!(policy.asking.lock().unwrap().is_none());
        assert!(matches!(
            policy.turn("https://a.example"),
            Turn::Decided(true)
        ));
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
            "2600:1:2::3",
            "2a01:2:3::4",
            "2001:0:102:304::1",
            "64:ff9b::102:304",
            "2002:102:304::1",
            "::ffff:1.2.3.4",
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

    #[tokio::test]
    async fn settings_add_origins_and_open_the_policy_up_while_it_runs() {
        let policy = closed();
        assert!(!policy.origin_allowed(Some("https://added.example")).await);
        policy.set(vec!["https://added.example".into()], true, true);
        assert!(policy.origin_allowed(Some("https://added.example")).await);
        assert!(policy.address_allowed(Ipv4Addr::LOCALHOST));
        assert!(policy.inbound());
        policy.set(Vec::new(), false, false);
        assert!(!policy.address_allowed(Ipv4Addr::LOCALHOST));
        assert!(!policy.inbound());
        assert!(
            policy.origin_allowed(Some("https://example.org")).await,
            "an origin from the command line stays"
        );
    }

    #[test]
    fn private_addresses_can_be_allowed_on_request() {
        let policy = Policy::new(Vec::new(), Approval::Deny, true, false);
        assert!(policy.address_allowed(Ipv4Addr::LOCALHOST));
    }
}
