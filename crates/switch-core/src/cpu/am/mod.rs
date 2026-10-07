//! `am`: the applet framework (`appletOE`/`appletAE`), the proxies and channels
//! they hand out, and the library applets a title can launch.

use super::{AppletMessage, Cpu, SHARED_BUFFER_USABLE_SLOTS};
use crate::trace::Level;
use crate::Result;

mod request;

/// `am` 2, NoDataInChannel.
const AM_NO_DATA_IN_CHANNEL: u32 = 128 | (2 << 9);

/// `LaunchParameterKind::PreselectedUser`.
pub(super) const LAUNCH_PARAMETER_PRESELECTED_USER: u32 = 2;

/// The `PreselectedUser` launch parameter: magic, version, then the uid, in 0x88 bytes.
pub(super) fn preselected_user_parameter(user: [u8; 16]) -> Vec<u8> {
    const MAGIC: u32 = 0xC794_97CA;
    const VERSION: u8 = 1;
    const LEN: usize = 0x88;
    const UID_OFFSET: usize = 0x8;
    let mut data = Vec::with_capacity(LEN);
    data.extend_from_slice(&MAGIC.to_le_bytes());
    data.push(VERSION);
    data.resize(UID_OFFSET, 0);
    data.extend_from_slice(&user);
    data.resize(LEN, 0);
    data
}

/// A library applet created through `ILibraryAppletCreator`. No process runs behind it.
#[derive(Debug, Default)]
pub(crate) struct LibraryApplet {
    id: u32,
    /// `LibraryAppletMode`.
    mode: u32,
    finished: bool,
    events: [Option<u64>; 3],
}

impl LibraryApplet {
    fn new(id: u32, mode: u32) -> Self {
        Self {
            id,
            mode,
            ..Self::default()
        }
    }

    fn finish(&mut self) {
        self.finished = true;
    }

    fn is_finished(&self) -> bool {
        self.finished
    }
}

const STATE_CHANGED_EVENT: usize = 0;

const POP_OUT_DATA_EVENT: usize = 1;

const POP_INTERACTIVE_OUT_DATA_EVENT: usize = 2;

const LIBRARY_APPLET_EVENT_NAMES: [&str; 3] = [
    "am:library-applet-state",
    "am:library-applet-out-data",
    "am:library-applet-interactive-out-data",
];

/// The two queues a library applet pops storages from.
#[derive(Clone, Copy)]
enum AppletQueue {
    InData,
    InteractiveInData,
}

impl AppletQueue {
    const ALL: [Self; 2] = [Self::InData, Self::InteractiveInData];

    fn slot(self) -> usize {
        match self {
            Self::InData => 0,
            Self::InteractiveInData => 1,
        }
    }

    fn event_name(self) -> &'static str {
        match self {
            Self::InData => "am:applet-in-data",
            Self::InteractiveInData => "am:applet-interactive-in-data",
        }
    }

    fn empty_message(self) -> &'static str {
        match self {
            Self::InData => {
                "[am] PopInData: the applet has popped every storage seeded for it and asked \
                 for another"
            }
            Self::InteractiveInData => {
                "[am] PopInteractiveInData: the applet is waiting on an answer from the caller \
                 that launched it, and nothing here launched it"
            }
        }
    }
}

/// How many interactive messages are kept for the host; the oldest are dropped.
const MAX_INTERACTIVE_MESSAGES: usize = 8;

/// The firmware applet an `AppletId` names, the inverse of [`applet_id_for`].
fn applet_name(applet_id: u32) -> &'static str {
    match applet_id {
        0x01 => "application",
        0x02 => "overlayDisp",
        0x03 => "qlaunch",
        0x04 => "system application",
        0x0A => "auth",
        0x0B => "cabinet",
        0x0C => "controller",
        0x0D => "dataErase",
        0x0E => "error",
        0x0F => "netConnect",
        0x10 => "playerSelect",
        0x11 => "swkbd",
        0x12 => "miiEdit",
        0x13 => "web",
        0x14 => "shop",
        0x15 => "photoViewer",
        0x16 => "set",
        0x17 => "offlineWeb",
        0x18 => "loginShare",
        0x19 => "wifiWebAuth",
        0x1A => "myPage",
        _ => "unknown applet",
    }
}

/// Whether a title id is one of the firmware's library applets.
pub(crate) fn is_library_applet(program_id: u64) -> bool {
    matches!(applet_id_for(program_id), 0x0A..=0x1A)
}

/// The `LibAppletCommonArguments::LaVersion` an applet expects.
pub(crate) fn applet_interface_version(program_id: u64) -> u32 {
    match applet_id_for(program_id) {
        0x12 => 3, // miiEdit
        // swkbd: 6.0.0+.
        APPLET_SWKBD => 0x8_000D,
        // Controller: 11.0.0+, the 0x430-byte `ControllerSupportArg`.
        APPLET_CONTROLLER => 8,
        // myPage: 9.0.0+.
        APPLET_MY_PAGE => 0x1_0000,
        // Web: 8.0.0+.
        APPLET_WEB => 0x8_0000,
        _ => 1,
    }
}

const APPLET_SWKBD: u32 = 0x11;
const APPLET_CONTROLLER: u32 = 0x0C;
const APPLET_MY_PAGE: u32 = 0x1A;
const APPLET_WEB: u32 = 0x13;

/// The launch storages a library applet's caller pushes after the common arguments.
pub(crate) fn applet_launch_storages(program_id: u64, user: [u8; 16]) -> Vec<Vec<u8>> {
    const GENERIC_SIZE: usize = 0x100;
    match applet_id_for(program_id) {
        APPLET_SWKBD => vec![swkbd_config(), vec![0u8; SWKBD_WORK_BUFFER_SIZE]],
        APPLET_CONTROLLER => vec![controller_support_arg_private(), controller_support_arg()],
        APPLET_MY_PAGE => vec![my_page_arg(user)],
        APPLET_WEB => vec![web_arg()],
        _ => vec![vec![0u8; GENERIC_SIZE]],
    }
}

/// `nn::swkbd::KeyboardConfig`: `SwkbdConfigCommon` then `SwkbdConfigNew`.
fn swkbd_config() -> Vec<u8> {
    const CONFIG_SIZE: usize = 0x4C8;
    const OK_TEXT: usize = 0x004;
    const MAX_TEXT_LENGTH: usize = 0x3AC;
    const MIN_TEXT_LENGTH: usize = 0x3B0;
    let mut config = vec![0u8; CONFIG_SIZE];
    // SwkbdType_Normal.
    config[0..4].copy_from_slice(&0u32.to_le_bytes());
    for (index, unit) in "OK".encode_utf16().enumerate() {
        let at = OK_TEXT + index * 2;
        config[at..at + 2].copy_from_slice(&unit.to_le_bytes());
    }
    config[MAX_TEXT_LENGTH..MAX_TEXT_LENGTH + 4].copy_from_slice(&32u32.to_le_bytes());
    config[MIN_TEXT_LENGTH..MIN_TEXT_LENGTH + 4].copy_from_slice(&0u32.to_le_bytes());
    config
}

/// The keyboard's third storage: its initial-string and dictionary buffer.
const SWKBD_WORK_BUFFER_SIZE: usize = 0x1000;

/// The friend-list applet's argument: the page to open and the user it belongs to.
fn my_page_arg(user: [u8; 16]) -> Vec<u8> {
    const ARG_SIZE: usize = 0x10A8;
    const USER_ID: usize = 0x8;
    let mut arg = vec![0u8; ARG_SIZE];
    // ShowFriendList.
    arg[..4].copy_from_slice(&0u32.to_le_bytes());
    arg[USER_ID..USER_ID + 16].copy_from_slice(&user);
    arg
}

/// The browser's argument: a `WebArgHeader`, then `{u16 type, u16 size, u32 pad}` entries.
fn web_arg() -> Vec<u8> {
    /// `ShimKind_Web`.
    const SHIM_WEB: u32 = 5;
    const TLV_INITIAL_URL: u16 = 1;
    const WEB_URL_SIZE: usize = 0xC00;
    const WEB_ARG_SIZE: usize = 0x2000;
    const HEADER_SIZE: usize = 8;
    const TLV_SIZE: usize = 8;
    /// A deliberately unreachable start page.
    const DEFAULT_URL: &str = "http://localhost/";

    let mut arg = vec![0u8; WEB_ARG_SIZE];
    arg[0..2].copy_from_slice(&1u16.to_le_bytes());
    arg[4..8].copy_from_slice(&SHIM_WEB.to_le_bytes());
    arg[8..10].copy_from_slice(&TLV_INITIAL_URL.to_le_bytes());
    arg[10..12].copy_from_slice(&(WEB_URL_SIZE as u16).to_le_bytes());
    let url = DEFAULT_URL.as_bytes();
    let at = HEADER_SIZE + TLV_SIZE;
    arg[at..at + url.len()].copy_from_slice(url);
    arg
}

/// A library applet's result storage, summarised for the log.
fn applet_result_summary(program_id: u64, data: &[u8]) -> String {
    let word = |at: usize| -> u32 {
        data.get(at..at + 4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .unwrap_or(0)
    };
    match applet_id_for(program_id) {
        // `ControllerSupportResultInfo { s8 player_count, pad[3], u32 selected_id, u32 result }`.
        APPLET_CONTROLLER if data.len() >= 0xC => {
            let outcome = match word(8) {
                0 => "confirmed".to_owned(),
                2 => "cancelled".to_owned(),
                other => format!("result {other:#x}"),
            };
            format!("{outcome}, {} player(s), npad {}", data[0] as i8, word(4))
        }
        // `u32 SwkbdResult`, then the UTF-16 text.
        APPLET_SWKBD if data.len() >= 4 => {
            let text: Vec<u16> = data[4..]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
                .take_while(|&unit| unit != 0)
                .collect();
            let outcome = if word(0) == 0 {
                "submitted"
            } else {
                "cancelled"
            };
            format!("{outcome} {:?}", String::from_utf16_lossy(&text))
        }
        _ => format!("{} bytes, starting {:#010x}", data.len(), word(0)),
    }
}

/// `AppletIdentityInfo { AppletId, pad, u64 title_id }` for the home menu.
fn home_menu_identity() -> [u8; 16] {
    const QLAUNCH_TITLE_ID: u64 = 0x0100_0000_0000_1000;
    const SYSTEM_APPLET_MENU: u32 = 3;
    let mut info = [0u8; 16];
    info[..4].copy_from_slice(&SYSTEM_APPLET_MENU.to_le_bytes());
    info[8..].copy_from_slice(&QLAUNCH_TITLE_ID.to_le_bytes());
    info
}

/// `nn::hid::system::ControllerSupportArgPrivate`.
fn controller_support_arg_private() -> Vec<u8> {
    const SIZE: u32 = 0x14;
    let mut arg = Vec::with_capacity(SIZE as usize);
    arg.extend_from_slice(&SIZE.to_le_bytes());
    arg.extend_from_slice(&(CONTROLLER_SUPPORT_ARG_SIZE as u32).to_le_bytes());
    // Flag0, Flag1, ShowControllerSupport, caller Application.
    arg.extend_from_slice(&[0, 0, 0, 0]);
    // `GetSupportedNpadStyleSet`, see `NPAD_PRESENTATIONS`.
    arg.extend_from_slice(&super::supported_npad_style_set().to_le_bytes());
    // `GetNpadJoyHoldType`: Vertical.
    arg.extend_from_slice(&0u32.to_le_bytes());
    arg
}

/// `nn::hid::ControllerSupportArg`, the 0x430-byte shape.
const CONTROLLER_SUPPORT_ARG_SIZE: usize = 0x430;

fn controller_support_arg() -> Vec<u8> {
    let mut arg = vec![0u8; CONTROLLER_SUPPORT_ARG_SIZE];
    // sdknso's defaults; the next byte permits dual Joy-Con.
    arg[..4].copy_from_slice(&0x0101_0400u32.to_le_bytes());
    arg[4] = 1;
    // enableSingleMode, needed for handheld to be an allowed answer.
    arg[5] = 1;
    arg
}

/// The `AppletId` a system applet reports for itself, from its title id.
fn applet_id_for(program_id: u64) -> u32 {
    if program_id & !0xFFFF != 0x0100_0000_0000_0000 {
        return 0x01; // AppletId_Application
    }
    match program_id & 0xFFFF {
        0x1000 => 0x03, // qlaunch -> SystemAppletMenu
        0x100C => 0x02, // overlayDisp -> OverlayApplet
        // auth, cabinet, controller, dataErase, error, netConnect,
        // playerSelect, swkbd, miiEdit, web, shop.
        low @ 0x1001..=0x100B => 0x0A + (low as u32 - 0x1001),
        low @ 0x100D..=0x1011 => 0x15 + (low as u32 - 0x100D),
        // `starter` sits in the range but is not a library applet.
        0x1012 => 0x04, // starter -> SystemApplication
        0x1013 => 0x1A, // myPage
        _ => 0x01,
    }
}

impl Cpu {
    /// The event an `ILibraryAppletAccessor` hands out for `slot`, allocated once.
    fn library_applet_event(&mut self, key: u64, slot: usize) -> u64 {
        if let Some(event) = self
            .am_applets
            .get(&key)
            .and_then(|applet| applet.events[slot])
        {
            return event;
        }
        // Not auto-clearing: an ended applet stays ended.
        let event = self.alloc_event(LIBRARY_APPLET_EVENT_NAMES[slot], false);
        self.am_applets.entry(key).or_default().events[slot] = Some(event);
        event
    }

    /// Pop the front of a library applet's queue as an `IStorage`, or 2128-0003 if empty.
    fn pop_applet_storage(&mut self, tls: u32, handle: u64, queue: AppletQueue) -> Result<()> {
        let data = match queue {
            AppletQueue::InData => self.am_in_data.pop_front(),
            AppletQueue::InteractiveInData => self.am_interactive_in.pop_front(),
        };
        self.refresh_applet_pop_events();
        match data {
            Some(data) => {
                let key = self.reply_with_interface(tls, handle, "am:storage")?;
                self.am_storages.insert(key, data);
                Ok(())
            }
            None => {
                const NO_DATA: u32 = 128 | (3 << 9);
                if self
                    .unimplemented_ipc
                    .insert((queue.event_name().to_string(), None))
                {
                    self.diagnostic(Level::Warn, queue.empty_message());
                }
                self.write_ipc_response(tls, NO_DATA, &[], &[], &[])
            }
        }
    }

    /// The event a pop queue hands out, allocated once and not auto-clearing.
    fn applet_queue_event(&mut self, queue: AppletQueue) -> u64 {
        if let Some(event) = self.am_pop_events[queue.slot()] {
            return event;
        }
        let event = self.alloc_event(queue.event_name(), false);
        self.am_pop_events[queue.slot()] = Some(event);
        event
    }

    /// Signal each pop event whose queue is non-empty and clear the rest.
    pub(super) fn refresh_applet_pop_events(&mut self) {
        for queue in AppletQueue::ALL {
            let Some(event) = self.am_pop_events[queue.slot()] else {
                continue;
            };
            let waiting = match queue {
                AppletQueue::InData => !self.am_in_data.is_empty(),
                AppletQueue::InteractiveInData => !self.am_interactive_in.is_empty(),
            };
            if waiting {
                self.signal_event(event);
            } else {
                self.clear_event(event);
            }
        }
    }

    /// Fire one of an applet's events, if the caller has taken it.
    fn signal_library_applet_event(&mut self, key: u64, slot: usize) {
        if let Some(event) = self
            .am_applets
            .get(&key)
            .and_then(|applet| applet.events[slot])
        {
            self.signal_event(event);
        }
    }

    /// Whether the applet behind an accessor has ended.
    fn library_applet_finished(&self, key: u64) -> bool {
        self.am_applets
            .get(&key)
            .is_some_and(LibraryApplet::is_finished)
    }

    /// The `ILockAccessor` event, created signalled and manual-reset.
    fn am_lock_accessor_event(&mut self) -> u64 {
        match self.lock_accessor_event {
            Some(h) => h,
            None => {
                let h = self.alloc_event("am:lock-accessor", false);
                self.signal_event(h);
                self.lock_accessor_event = Some(h);
                h
            }
        }
    }
}

#[cfg(test)]
mod tests;
