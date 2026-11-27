//! The relay's settings frames, which travel outside the game loop: the socket hands each one
//! to the page as it arrives, and sends the page's changes at once. `docs/relay.md`, Settings.

use switch_core::json;
use switch_relay_proto::{ClientFrame, ServerFrame, SettingsChange, SettingsRefusal};

/// What the page is told for a settings frame, as JSON; `None` for any other frame.
pub(crate) fn report(frame: &ServerFrame) -> Option<String> {
    Some(match frame {
        ServerFrame::Settings(settings) => json::object(|o| {
            o.field("type", "settings")
                .field("listening", settings.listening)
                .field("port", settings.port)
                .field("allowPrivate", settings.allow_private)
                .field("allowInbound", settings.allow_inbound)
                .field("saved", settings.saved)
                .array("locked", |a| {
                    let locked = [
                        ("port", settings.port_locked),
                        ("allowPrivate", settings.private_locked),
                        ("allowInbound", settings.inbound_locked),
                    ];
                    a.extend(locked.iter().filter(|(_, set)| *set).map(|(name, _)| name));
                })
                .array("origins", |a| {
                    a.extend(&settings.origins);
                })
                .array("fixedOrigins", |a| {
                    a.extend(&settings.fixed_origins);
                });
        }),
        ServerFrame::Asking => json::object(|o| {
            o.field("type", "asking");
        }),
        ServerFrame::SettingsRefused(reason) => json::object(|o| {
            let reason = match reason {
                SettingsRefusal::Denied => "denied",
                SettingsRefusal::Busy => "busy",
                SettingsRefusal::Locked => "locked",
                SettingsRefusal::Invalid => "invalid",
                SettingsRefusal::Unsaved => "unsaved",
            };
            o.field("type", "refused").field("reason", reason);
        }),
        _ => return None,
    })
}

/// What the page is told when the relay is too old to have settings.
pub(crate) fn too_old() -> String {
    json::object(|o| {
        o.field("type", "old");
    })
}

/// The frame asking the relay for these settings.
pub(crate) fn change(
    port: u16,
    allow_private: bool,
    allow_inbound: bool,
    origins: Vec<String>,
) -> Vec<u8> {
    ClientFrame::SetSettings(SettingsChange {
        port,
        allow_private,
        allow_inbound,
        origins,
    })
    .encode()
}

#[cfg(test)]
mod tests {
    use super::*;
    use switch_relay_proto::Settings;

    #[test]
    fn the_page_gets_the_settings_as_it_names_them() {
        let frame = ServerFrame::Settings(Settings {
            listening: 18787,
            port: 9000,
            allow_private: true,
            allow_inbound: false,
            saved: true,
            port_locked: false,
            private_locked: true,
            inbound_locked: false,
            origins: vec!["https://a.example".into()],
            fixed_origins: vec!["https://tenshii.moe".into()],
        });
        assert_eq!(
            report(&frame).unwrap(),
            r#"{"type":"settings","listening":18787,"port":9000,"allowPrivate":true,"allowInbound":false,"saved":true,"locked":["allowPrivate"],"origins":["https://a.example"],"fixedOrigins":["https://tenshii.moe"]}"#
        );
        assert_eq!(
            report(&ServerFrame::Asking).unwrap(),
            r#"{"type":"asking"}"#
        );
        assert_eq!(
            report(&ServerFrame::SettingsRefused(SettingsRefusal::Busy)).unwrap(),
            r#"{"type":"refused","reason":"busy"}"#
        );
        assert_eq!(report(&ServerFrame::Alive), None);
    }

    #[test]
    fn a_change_is_the_set_settings_frame() {
        let frame = change(18787, true, false, vec!["https://a.example".into()]);
        assert_eq!(
            ClientFrame::decode(&frame),
            Ok(ClientFrame::SetSettings(SettingsChange {
                port: 18787,
                allow_private: true,
                allow_inbound: false,
                origins: vec!["https://a.example".into()],
            }))
        );
    }
}
