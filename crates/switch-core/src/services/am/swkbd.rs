//! The software keyboard, answered by the host: the guest waits while the page asks.

use super::{LibraryApplet, POP_OUT_DATA_EVENT, STATE_CHANGED_EVENT};
use crate::cpu::Cpu;
use crate::trace::Level;

/// What the keyboard shows and accepts, from `SwkbdConfigCommon`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KeyboardRequest {
    pub header: String,
    pub sub: String,
    pub guide: String,
    /// The submit button's label; empty for the default.
    pub ok: String,
    /// In UTF-16 units.
    pub max_length: u32,
    pub min_length: u32,
    pub password: bool,
}

/// A request and how to answer it.
#[derive(Debug)]
pub(crate) struct PendingKeyboard {
    accessor: u64,
    request: KeyboardRequest,
    utf8: bool,
}

/// The longest text the 0x7D4-byte result holds as UTF-16 with a terminator.
const MAX_TEXT_LENGTH: u32 = 500;

const RESULT_SIZE: usize = 0x7D8;

/// `SwkbdResult`.
const RESULT_OK: u32 = 0;
const RESULT_CANCEL: u32 = 1;

/// `SwkbdConfigCommon` offsets.
const OK_TEXT: (usize, usize) = (0x004, 9);
const HEADER_TEXT: (usize, usize) = (0x024, 65);
const SUB_TEXT: (usize, usize) = (0x0A6, 129);
const GUIDE_TEXT: (usize, usize) = (0x1A8, 257);
const MAX_LENGTH: usize = 0x3AC;
const MIN_LENGTH: usize = 0x3B0;
const PASSWORD_MODE: usize = 0x3B4;
const USE_UTF8: usize = 0x3BD;
const CONFIG_COMMON_SIZE: usize = 0x3D4;

fn utf16_field(config: &[u8], (at, units): (usize, usize)) -> String {
    let text: Vec<u16> = config[at..at + units * 2]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .take_while(|&unit| unit != 0)
        .collect();
    String::from_utf16_lossy(&text)
}

fn word(config: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(config[at..at + 4].try_into().unwrap())
}

/// Parse a `KeyboardConfig`; `None` if it is too short to hold the common part.
fn parse_config(config: &[u8]) -> Option<(KeyboardRequest, bool)> {
    if config.len() < CONFIG_COMMON_SIZE {
        return None;
    }
    let max = word(config, MAX_LENGTH);
    let max_length = if max == 0 || max > MAX_TEXT_LENGTH {
        MAX_TEXT_LENGTH
    } else {
        max
    };
    let request = KeyboardRequest {
        header: utf16_field(config, HEADER_TEXT),
        sub: utf16_field(config, SUB_TEXT),
        guide: utf16_field(config, GUIDE_TEXT),
        ok: utf16_field(config, OK_TEXT),
        max_length,
        min_length: word(config, MIN_LENGTH).min(max_length),
        password: word(config, PASSWORD_MODE) != 0,
    };
    Some((request, config[USE_UTF8] != 0))
}

/// The keyboard's output storage: `u32 SwkbdResult`, then the text, NUL-terminated.
fn result_storage(text: Option<&str>, max_length: u32, utf8: bool) -> Vec<u8> {
    let mut data = vec![0u8; RESULT_SIZE];
    let Some(text) = text else {
        data[..4].copy_from_slice(&RESULT_CANCEL.to_le_bytes());
        return data;
    };
    data[..4].copy_from_slice(&RESULT_OK.to_le_bytes());
    let mut units = 0;
    let text: String = text
        .chars()
        .take_while(|c| {
            units += c.len_utf16();
            units <= max_length as usize
        })
        .collect();
    let bytes: Vec<u8> = if utf8 {
        text.into_bytes()
    } else {
        text.encode_utf16().flat_map(u16::to_le_bytes).collect()
    };
    let room = RESULT_SIZE - 4 - if utf8 { 1 } else { 2 };
    let len = bytes.len().min(room);
    data[4..4 + len].copy_from_slice(&bytes[..len]);
    data
}

impl LibraryApplet {
    /// Whether this is a keyboard the host can answer: swkbd in the foreground, not inline.
    pub(super) fn is_host_keyboard(&self) -> bool {
        self.id == super::APPLET_SWKBD && self.mode == 0
    }
}

impl Cpu {
    /// The keyboard waiting for text, if any.
    pub fn keyboard_request(&self) -> Option<&KeyboardRequest> {
        self.am_keyboard.as_ref().map(|pending| &pending.request)
    }

    /// Answer the waiting keyboard with `text`, or cancel it with `None`. Returns
    /// whether a keyboard was waiting.
    pub fn answer_keyboard(&mut self, text: Option<&str>) -> bool {
        let Some(pending) = self.am_keyboard.take() else {
            return false;
        };
        let storage = result_storage(text, pending.request.max_length, pending.utf8);
        let Some(applet) = self.am_applets.get_mut(&pending.accessor) else {
            return false;
        };
        applet.answer = Some(storage);
        applet.finish();
        self.signal_library_applet_event(pending.accessor, STATE_CHANGED_EVENT);
        self.signal_library_applet_event(pending.accessor, POP_OUT_DATA_EVENT);
        let outcome = if text.is_some() {
            "submitted"
        } else {
            "cancelled"
        };
        self.diagnostic(Level::Info, &format!("[am] swkbd: the text was {outcome}"));
        true
    }

    /// Start a host keyboard from its pushed `KeyboardConfig`; false if it has none.
    pub(super) fn start_keyboard(&mut self, accessor: u64) -> bool {
        let parsed = self
            .am_applets
            .get(&accessor)
            .and_then(|applet| applet.in_data.get(1))
            .and_then(|config| parse_config(config));
        let Some((request, utf8)) = parsed else {
            return false;
        };
        self.diagnostic(Level::Info, "[am] swkbd: waiting for the page to send text");
        self.am_keyboard = Some(PendingKeyboard {
            accessor,
            request,
            utf8,
        });
        true
    }

    /// Drop a waiting keyboard whose applet is being ended.
    pub(super) fn forget_keyboard(&mut self, accessor: u64) {
        if self
            .am_keyboard
            .as_ref()
            .is_some_and(|pending| pending.accessor == accessor)
        {
            self.am_keyboard = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_utf16(config: &mut [u8], (at, _): (usize, usize), text: &str) {
        for (index, unit) in text.encode_utf16().enumerate() {
            config[at + index * 2..at + index * 2 + 2].copy_from_slice(&unit.to_le_bytes());
        }
    }

    #[test]
    fn the_config_names_what_the_keyboard_shows() {
        let mut config = vec![0u8; 0x4C8];
        put_utf16(&mut config, HEADER_TEXT, "Your name?");
        put_utf16(&mut config, GUIDE_TEXT, "Enter a name");
        put_utf16(&mut config, OK_TEXT, "Done");
        config[MAX_LENGTH..MAX_LENGTH + 4].copy_from_slice(&10u32.to_le_bytes());
        config[MIN_LENGTH..MIN_LENGTH + 4].copy_from_slice(&1u32.to_le_bytes());
        config[USE_UTF8] = 1;
        let (request, utf8) = parse_config(&config).unwrap();
        assert_eq!(request.header, "Your name?");
        assert_eq!(request.guide, "Enter a name");
        assert_eq!(request.ok, "Done");
        assert_eq!((request.max_length, request.min_length), (10, 1));
        assert!(!request.password);
        assert!(utf8);
    }

    #[test]
    fn no_length_limit_means_what_the_result_can_hold() {
        let (request, _) = parse_config(&[0u8; CONFIG_COMMON_SIZE]).unwrap();
        assert_eq!(request.max_length, MAX_TEXT_LENGTH);
        assert!(parse_config(&[0u8; CONFIG_COMMON_SIZE - 1]).is_none());
    }

    #[test]
    fn the_result_is_cut_to_the_limit_in_the_encoding_asked_for() {
        let data = result_storage(Some("Zelda"), 3, false);
        assert_eq!(data.len(), RESULT_SIZE);
        assert_eq!(word(&data, 0), RESULT_OK);
        assert_eq!(&data[4..12], &[b'Z', 0, b'e', 0, b'l', 0, 0, 0]);

        let data = result_storage(Some("é"), 10, true);
        assert_eq!(&data[4..7], &[0xC3, 0xA9, 0]);

        let data = result_storage(None, 10, false);
        assert_eq!(word(&data, 0), RESULT_CANCEL);
    }
}
