//! The relay's settings: saved in a file, pinned for one run by flags, and changed from a page.

use std::net::Ipv6Addr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use switch_relay_proto::{Settings as Report, SettingsChange, SettingsRefusal as Refusal};
use tokio::sync::Mutex;

use crate::policy::{Policy, Unknown};
use crate::{dialog, terminal};

/// The port the relay listens on when nothing says otherwise.
pub const DEFAULT_PORT: u16 = 18787;

/// Most origins the settings may trust.
pub const MAX_ORIGINS: usize = 64;

/// What the settings file holds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    pub port: u16,
    pub allow_private: bool,
    pub allow_inbound: bool,
    pub origins: Vec<String>,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            port: DEFAULT_PORT,
            allow_private: false,
            allow_inbound: false,
            origins: Vec::new(),
        }
    }
}

/// What the command line sets for this run, over the file; no page can change it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Overrides {
    pub port: Option<u16>,
    pub allow_private: bool,
    pub allow_inbound: bool,
    pub origins: Vec<String>,
}

/// Where the settings file lives: `switch-relay/switch-relay.toml` in the user's config
/// directory. `None` where the system has no such directory.
pub fn default_path() -> Option<PathBuf> {
    Some(
        dirs::config_dir()?
            .join("switch-relay")
            .join("switch-relay.toml"),
    )
}

/// The settings in `path`, or the defaults when there is no file yet.
pub fn load(path: &Path) -> Result<Settings, String> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(Settings::default())
        }
        Err(error) => return Err(format!("cannot read {}: {error}", path.display())),
    };
    let mut settings: Settings =
        toml::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    if settings.port == 0 {
        return Err(format!("{}: port cannot be 0", path.display()));
    }
    settings.origins = origins(&settings.origins)
        .map_err(|origin| format!("{}: {origin} is not an origin", path.display()))?;
    Ok(settings)
}

/// Write the settings through a file beside `path`, so a crash never leaves half of one.
fn save(path: &Path, settings: &Settings) -> std::io::Result<()> {
    if let Some(directory) = path.parent() {
        std::fs::create_dir_all(directory)?;
    }
    let text = toml::to_string(settings).map_err(std::io::Error::other)?;
    let partial = path.with_extension("toml.partial");
    std::fs::write(&partial, text)?;
    std::fs::rename(&partial, path)
}

/// `origin` as a browser sends it in `Origin`: `http` or `https`, a lowercase host, and a port
/// only when it is not the scheme's own. `None` for anything else.
pub fn origin(origin: &str) -> Option<String> {
    let origin = origin.trim().to_ascii_lowercase();
    let (scheme, rest) = origin.split_once("://")?;
    let default_port = match scheme {
        "http" => 80,
        "https" => 443,
        _ => return None,
    };
    let name = |label: &str| {
        !label.is_empty()
            && label.len() <= 63
            && label
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
    };
    let (host, port) = match rest.strip_prefix('[') {
        Some(bracketed) => {
            let (address, after) = bracketed.split_once(']')?;
            // Written the way a browser writes it, so `[::0:1]` matches `[::1]`.
            let address: Ipv6Addr = address.parse().ok()?;
            let port = match after {
                "" => None,
                after => Some(after.strip_prefix(':')?),
            };
            (format!("[{address}]"), port)
        }
        None => {
            let (host, port) = match rest.split_once(':') {
                Some((host, port)) => (host, Some(port)),
                None => (rest, None),
            };
            if host.len() > 253 || !host.split('.').all(name) {
                return None;
            }
            (host.to_owned(), port)
        }
    };
    let port = match port {
        None => None,
        Some(digits) if digits.bytes().all(|byte| byte.is_ascii_digit()) => {
            Some(digits.parse::<u16>().ok().filter(|&port| port != 0)?)
        }
        Some(_) => return None,
    };
    let origin = match port {
        Some(port) if port != default_port => format!("{scheme}://{host}:{port}"),
        _ => format!("{scheme}://{host}"),
    };
    // The protocol carries an origin after a length byte.
    (origin.len() <= 255).then_some(origin)
}

/// Each origin in its canonical form, without repeats; the first that is not one is the error.
fn origins(list: &[String]) -> Result<Vec<String>, String> {
    let mut out: Vec<String> = Vec::new();
    for given in list {
        let canonical = origin(given).ok_or_else(|| given.clone())?;
        if !out.contains(&canonical) {
            out.push(canonical);
        }
    }
    Ok(out)
}

/// Who answers when a page asks to open the relay up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Approval {
    /// A window on the relay's desktop; with none, the terminal when there is one.
    Ask {
        terminal: bool,
    },
    /// Yes without asking. For tests.
    Allow,
    Deny,
}

impl Approval {
    async fn ask(self, text: String) -> bool {
        match self {
            Approval::Allow => true,
            Approval::Deny => false,
            Approval::Ask { terminal } => {
                let question = text.clone();
                let shown = tokio::task::spawn_blocking(move || dialog::ask(&question))
                    .await
                    .ok()
                    .flatten();
                match shown {
                    Some(answer) => answer,
                    None if terminal => terminal::ask(&text).await,
                    None => false,
                }
            }
        }
    }
}

/// The settings of a running relay, and the one way to change them.
pub struct Config {
    policy: Arc<Policy>,
    overrides: Overrides,
    approval: Approval,
    path: Option<PathBuf>,
    listening: u16,
    /// What the file says. Held across a change, question included, so changes go one by one.
    saved: Mutex<Settings>,
}

impl Config {
    /// `settings` from the file at `path` (or none, so changes last until exit), with the
    /// flags' `overrides` on top, for a relay listening on `listening`.
    pub fn new(
        settings: Settings,
        overrides: Overrides,
        unknown: Unknown,
        approval: Approval,
        path: Option<PathBuf>,
        listening: u16,
    ) -> Config {
        let policy = Policy::new(overrides.origins.clone(), unknown, false, false);
        let config = Config {
            policy: Arc::new(policy),
            overrides,
            approval,
            path,
            listening,
            saved: Mutex::new(Settings::default()),
        };
        config.apply(&settings);
        *config.saved.try_lock().unwrap() = settings;
        config
    }

    pub fn policy(&self) -> &Arc<Policy> {
        &self.policy
    }

    /// The origins the settings trust, less those trusted anyway.
    fn effective_origins(&self, settings: &Settings) -> Vec<String> {
        let fixed = self.policy.fixed_origins();
        settings
            .origins
            .iter()
            .filter(|origin| !fixed.contains(origin))
            .cloned()
            .collect()
    }

    fn apply(&self, settings: &Settings) {
        self.policy.set(
            self.effective_origins(settings),
            settings.allow_private || self.overrides.allow_private,
            settings.allow_inbound || self.overrides.allow_inbound,
        );
    }

    fn report_of(&self, settings: &Settings) -> Report {
        let overrides = &self.overrides;
        Report {
            listening: self.listening,
            port: overrides.port.unwrap_or(settings.port),
            allow_private: settings.allow_private || overrides.allow_private,
            allow_inbound: settings.allow_inbound || overrides.allow_inbound,
            saved: self.path.is_some(),
            port_locked: overrides.port.is_some(),
            private_locked: overrides.allow_private,
            inbound_locked: overrides.allow_inbound,
            origins: self.effective_origins(settings),
            fixed_origins: self.policy.fixed_origins().to_vec(),
        }
    }

    pub async fn report(&self) -> Report {
        self.report_of(&*self.saved.lock().await)
    }

    /// Make the settings what `from` asks. Anything that opens the relay up is asked about
    /// first, after `asking` is called; the file is written before the relay changes.
    pub async fn change(
        &self,
        from: &str,
        wanted: SettingsChange,
        asking: impl FnOnce(),
    ) -> Result<Report, Refusal> {
        let Ok(mut saved) = self.saved.try_lock() else {
            return Err(Refusal::Busy);
        };
        let now = self.report_of(&saved);
        let origins = origins(&wanted.origins).map_err(|_| Refusal::Invalid)?;
        let origins: Vec<String> = origins
            .into_iter()
            .filter(|origin| !now.fixed_origins.contains(origin))
            .collect();
        if wanted.port == 0 || origins.len() > MAX_ORIGINS {
            return Err(Refusal::Invalid);
        }
        let overrides = &self.overrides;
        if overrides.port.is_some_and(|port| port != wanted.port)
            || (overrides.allow_private && !wanted.allow_private)
            || (overrides.allow_inbound && !wanted.allow_inbound)
        {
            return Err(Refusal::Locked);
        }

        let mut opening = Vec::new();
        if wanted.allow_private && !now.allow_private {
            opening.push("let the console reach this computer and your local network".to_owned());
        }
        if wanted.allow_inbound && !now.allow_inbound {
            opening.push("let other computers connect to the console".to_owned());
        }
        for origin in origins
            .iter()
            .filter(|origin| !now.origins.contains(origin))
        {
            opening.push(format!("let pages from {origin} use this relay"));
        }
        if !opening.is_empty() {
            asking();
            let list: String = opening.iter().map(|line| format!("\n  - {line}")).collect();
            let question = format!("{from} wants switch-relay to:\n{list}\n\nAllow this?");
            if !self.approval.ask(question).await {
                eprintln!("refused a settings change from {from}");
                return Err(Refusal::Denied);
            }
        }

        let next = Settings {
            port: if overrides.port.is_some() {
                saved.port
            } else {
                wanted.port
            },
            allow_private: if overrides.allow_private {
                saved.allow_private
            } else {
                wanted.allow_private
            },
            allow_inbound: if overrides.allow_inbound {
                saved.allow_inbound
            } else {
                wanted.allow_inbound
            },
            origins,
        };
        if let Some(path) = &self.path {
            if let Err(error) = save(path, &next) {
                eprintln!("cannot save the settings to {}: {error}", path.display());
                return Err(Refusal::Unsaved);
            }
        }
        self.apply(&next);
        *saved = next;
        eprintln!("{from} changed the settings");
        Ok(self.report_of(&saved))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origins_come_out_as_a_browser_sends_them() {
        for (given, canonical) in [
            ("https://Example.org", "https://example.org"),
            ("https://example.org:443", "https://example.org"),
            ("http://localhost:80", "http://localhost"),
            ("http://localhost:8000", "http://localhost:8000"),
            ("https://[::1]:8443", "https://[::1]:8443"),
            ("http://[2001:DB8:0::1]", "http://[2001:db8::1]"),
            (" https://a-b.example ", "https://a-b.example"),
        ] {
            assert_eq!(origin(given).as_deref(), Some(canonical), "{given}");
        }
    }

    #[test]
    fn anything_else_is_not_an_origin() {
        for given in [
            "",
            "example.org",
            "ftp://example.org",
            "https://",
            "https://example.org/",
            "https://example.org/path",
            "https://user@example.org",
            "https://exa mple.org",
            "https://<b>x</b>",
            "https://example..org",
            "https://example.org:0",
            "https://example.org:65536",
            "https://example.org:+1",
            // A 253-byte host is a valid name, but the origin is too long for a length byte.
            &format!(
                "https://{}.{}",
                vec!["a".repeat(63); 3].join("."),
                "b".repeat(61)
            ),
            "https://[::1",
            "https://[nope]",
            "https://[::1]x",
            "null",
        ] {
            assert_eq!(origin(given), None, "{given}");
        }
    }

    fn config(approval: Approval, overrides: Overrides, path: Option<PathBuf>) -> Config {
        Config::new(
            Settings::default(),
            overrides,
            Unknown::Deny,
            approval,
            path,
            DEFAULT_PORT,
        )
    }

    fn wanted(allow_private: bool, origins: &[&str]) -> SettingsChange {
        SettingsChange {
            port: DEFAULT_PORT,
            allow_private,
            allow_inbound: false,
            origins: origins.iter().map(|origin| (*origin).to_owned()).collect(),
        }
    }

    #[tokio::test]
    async fn opening_up_asks_and_a_no_changes_nothing() {
        let config = config(Approval::Deny, Overrides::default(), None);
        let mut asked = false;
        let refused = config
            .change("https://tenshii.moe", wanted(true, &[]), || asked = true)
            .await;
        assert_eq!(refused, Err(Refusal::Denied));
        assert!(asked);
        assert!(!config.policy().private());
        let mut asked = false;
        let refused = config
            .change(
                "https://tenshii.moe",
                wanted(false, &["https://a.example"]),
                || asked = true,
            )
            .await;
        assert_eq!(refused, Err(Refusal::Denied));
        assert!(asked, "a new origin opens the relay up too");
    }

    #[tokio::test]
    async fn closing_down_does_not_ask() {
        let config = Config::new(
            Settings {
                allow_private: true,
                origins: vec!["https://a.example".into()],
                ..Settings::default()
            },
            Overrides::default(),
            Unknown::Deny,
            Approval::Deny,
            None,
            DEFAULT_PORT,
        );
        assert!(config.policy().private());
        let report = config
            .change("https://tenshii.moe", wanted(false, &[]), || {
                panic!("nothing to ask")
            })
            .await
            .unwrap();
        assert!(!report.allow_private);
        assert!(report.origins.is_empty());
        assert!(!config.policy().private());
    }

    #[tokio::test]
    async fn a_yes_applies_and_saves_the_change() {
        let directory =
            std::env::temp_dir().join(format!("switch-relay-test-{}", std::process::id()));
        let path = directory.join("nested").join("switch-relay.toml");
        let config = config(Approval::Allow, Overrides::default(), Some(path.clone()));
        let report = config
            .change(
                "https://tenshii.moe",
                SettingsChange {
                    port: 9000,
                    ..wanted(true, &["https://A.example:443", "https://tenshii.moe"])
                },
                || {},
            )
            .await
            .unwrap();
        assert_eq!(
            report.origins,
            ["https://a.example"],
            "trusted anyway, so left out"
        );
        assert_eq!((report.port, report.listening), (9000, DEFAULT_PORT));
        assert!(config.policy().private());
        assert!(
            config
                .policy()
                .origin_allowed(Some("https://a.example"))
                .await
        );
        let saved = load(&path).unwrap();
        std::fs::remove_dir_all(&directory).unwrap();
        assert_eq!(
            saved,
            Settings {
                port: 9000,
                allow_private: true,
                allow_inbound: false,
                origins: vec!["https://a.example".into()],
            }
        );
    }

    #[tokio::test]
    async fn what_a_flag_sets_cannot_be_undone_from_a_page() {
        let overrides = Overrides {
            port: Some(5000),
            allow_private: true,
            ..Overrides::default()
        };
        let config = config(Approval::Allow, overrides, None);
        let report = config.report().await;
        assert!(report.port_locked && report.private_locked && !report.inbound_locked);
        assert_eq!(report.port, 5000);
        assert!(report.allow_private);
        let turn_off = config
            .change("https://tenshii.moe", wanted(false, &[]), || {})
            .await;
        assert_eq!(turn_off, Err(Refusal::Locked));
        let move_port = config
            .change(
                "https://tenshii.moe",
                SettingsChange {
                    port: 1,
                    ..wanted(true, &[])
                },
                || {},
            )
            .await;
        assert_eq!(move_port, Err(Refusal::Locked));
        let unchanged = config
            .change(
                "https://tenshii.moe",
                SettingsChange {
                    port: 5000,
                    ..wanted(true, &[])
                },
                || {},
            )
            .await;
        assert!(unchanged.is_ok());
    }

    #[tokio::test]
    async fn nonsense_is_refused() {
        let config = config(Approval::Allow, Overrides::default(), None);
        let bad_origin = config
            .change(
                "https://tenshii.moe",
                wanted(false, &["https://x/path"]),
                || {},
            )
            .await;
        assert_eq!(bad_origin, Err(Refusal::Invalid));
        let no_port = config
            .change(
                "https://tenshii.moe",
                SettingsChange {
                    port: 0,
                    ..wanted(false, &[])
                },
                || {},
            )
            .await;
        assert_eq!(no_port, Err(Refusal::Invalid));
        let many: Vec<String> = (0..=MAX_ORIGINS)
            .map(|index| format!("https://{index}.example"))
            .collect();
        let many: Vec<&str> = many.iter().map(String::as_str).collect();
        let too_many = config
            .change("https://tenshii.moe", wanted(false, &many), || {})
            .await;
        assert_eq!(too_many, Err(Refusal::Invalid));
    }

    #[test]
    fn a_settings_file_is_read_strictly() {
        let directory =
            std::env::temp_dir().join(format!("switch-relay-load-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("switch-relay.toml");
        assert_eq!(
            load(&path),
            Ok(Settings::default()),
            "no file is the defaults"
        );
        std::fs::write(
            &path,
            "allow_inbound = true\norigins = [\"HTTPS://B.example\"]\n",
        )
        .unwrap();
        assert_eq!(
            load(&path),
            Ok(Settings {
                allow_inbound: true,
                origins: vec!["https://b.example".into()],
                ..Settings::default()
            })
        );
        std::fs::write(&path, "alow_private = true\n").unwrap();
        assert!(
            load(&path).is_err(),
            "a misspelt key is an error, not ignored"
        );
        std::fs::write(&path, "origins = [\"b.example\"]\n").unwrap();
        assert!(load(&path).is_err());
        std::fs::write(&path, "port = 0\n").unwrap();
        assert!(load(&path).is_err());
        std::fs::remove_dir_all(&directory).unwrap();
    }
}
