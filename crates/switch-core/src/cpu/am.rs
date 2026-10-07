//! `am`: the applet framework (`appletOE`/`appletAE`), the proxies and channels
//! they hand out, and the library applets a title can launch.

use super::Cpu;
use crate::trace::Level;
use crate::Result;

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

    /// `appletOE`/`appletAE` and the sub-interfaces they hand out.
    pub(super) fn applet_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "am:proxy-service");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, "am:control", cmd_id),
            };
        }
        // Which `am` sub-interface this request is for: by domain object id, or by session handle.
        let object_id = self.ipc_domain_object_id(tls);
        let iface = if self.ipc_is_domain_request(tls) {
            self.domain_interface(handle, object_id)
                .unwrap_or("am:unknown")
                .to_string()
        } else {
            match self.service_name(handle) {
                Some("appletOE") | Some("appletAE") | None => "am:proxy-service".to_string(),
                Some(name) => name.to_string(),
            }
        };
        match iface.as_str() {
            "am:proxy-service" => match cmd_id {
                Some(0) => {
                    self.set_applet_is_application(true);
                    self.reply_with_interface(tls, handle, "am:application-proxy")?;
                    Ok(())
                }
                // OpenLibraryAppletProxy, and OpenLibraryAppletProxyOld.
                Some(200) | Some(201) => {
                    self.set_applet_is_application(false);
                    self.reply_with_interface(tls, handle, "am:library-applet-proxy")?;
                    Ok(())
                }
                // OpenSystemAppletProxy (and Ex at 110), opened by the Home Menu.
                Some(100) | Some(110) => {
                    self.set_applet_is_application(false);
                    self.reply_with_interface(tls, handle, "am:system-applet-proxy")?;
                    Ok(())
                }
                // OpenSystemApplicationProxy.
                Some(350) => {
                    self.set_applet_is_application(true);
                    self.reply_with_interface(tls, handle, "am:application-proxy")?;
                    Ok(())
                }
                // OpenOverlayAppletProxy.
                Some(300) => {
                    self.set_applet_is_application(false);
                    self.reply_with_interface(tls, handle, "am:library-applet-proxy")?;
                    Ok(())
                }
                // GetSystemProcessCommonFunctions / GetAppletAlternativeFunctions: no proxy, so the applet kind is unchanged.
                Some(450) => {
                    self.reply_with_interface(tls, handle, "am:system-process-common-functions")?;
                    Ok(())
                }
                Some(460) => {
                    self.reply_with_interface(tls, handle, "am:applet-alternative-functions")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ISystemAppletProxy's Get* accessors.
            "am:system-applet-proxy" => {
                let sub = match cmd_id {
                    Some(0) => Some("am:common-state-getter"),
                    Some(1) => Some("am:self-controller"),
                    Some(2) => Some("am:window-controller"),
                    Some(3) => Some("am:audio-controller"),
                    Some(4) => Some("am:display-controller"),
                    Some(10) => Some("am:process-winding-controller"),
                    Some(11) => Some("am:library-applet-creator"),
                    Some(20) => Some("am:home-menu-functions"),
                    Some(21) => Some("am:global-state-controller"),
                    Some(22) => Some("am:application-creator"),
                    // GetAppletCommonFunctions (10.0.0+).
                    Some(23) => Some("am:applet-common-functions"),
                    Some(1000) => Some("am:debug-functions"),
                    _ => None,
                };
                match sub {
                    Some(name) => {
                        self.reply_with_interface(tls, handle, name)?;
                        Ok(())
                    }
                    None => self.unimplemented_command(tls, &iface, cmd_id),
                }
            }
            // ILibraryAppletProxy's Get* accessors.
            "am:library-applet-proxy" => {
                let sub = match cmd_id {
                    Some(0) => Some("am:common-state-getter"),
                    Some(1) => Some("am:self-controller"),
                    Some(2) => Some("am:window-controller"),
                    Some(3) => Some("am:audio-controller"),
                    Some(4) => Some("am:display-controller"),
                    Some(10) => Some("am:process-winding-controller"),
                    Some(11) => Some("am:library-applet-creator"),
                    Some(20) => Some("am:library-applet-self-accessor"),
                    Some(21) => Some("am:applet-common-functions"),
                    Some(22) => Some("am:home-menu-functions"),
                    Some(23) => Some("am:global-state-controller"),
                    Some(1000) => Some("am:debug-functions"),
                    _ => None,
                };
                match sub {
                    Some(name) => {
                        self.reply_with_interface(tls, handle, name)?;
                        Ok(())
                    }
                    None => self.unimplemented_command(tls, &iface, cmd_id),
                }
            }
            // IApplicationProxy's Get* accessors.
            "am:application-proxy" => {
                let sub = match cmd_id {
                    Some(0) => Some("am:common-state-getter"),
                    Some(1) => Some("am:self-controller"),
                    Some(2) => Some("am:window-controller"),
                    Some(3) => Some("am:audio-controller"),
                    Some(4) => Some("am:display-controller"),
                    Some(11) => Some("am:library-applet-creator"),
                    Some(20) => Some("am:application-functions"),
                    Some(1000) => Some("am:debug-functions"),
                    _ => None,
                };
                match sub {
                    Some(name) => {
                        self.reply_with_interface(tls, handle, name)?;
                        Ok(())
                    }
                    None => self.unimplemented_command(tls, &iface, cmd_id),
                }
            }
            // ICommonStateGetter.
            "am:common-state-getter" => match cmd_id {
                // GetSettingsPlatformRegion: 1 is Global.
                Some(300) => self.write_ipc_response(tls, 0, &[], &1u8.to_le_bytes(), &[]),
                // GetOperationModeSystemInfo.
                Some(200) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
                // GetHomeButtonReaderLockAccessor / GetReaderLockAccessorEx / GetWriterLockAccessorEx.
                Some(30) | Some(31) | Some(32) => {
                    self.reply_with_interface(tls, handle, "am:lock-accessor")?;
                    Ok(())
                }
                // GetEventHandle: signalled while a message is queued.
                Some(0) => {
                    let h = match self.applet_event {
                        Some(h) => h,
                        None => {
                            let h = self.alloc_event("am:applet-message", true);
                            self.applet_event = Some(h);
                            h
                        }
                    };
                    if self.has_applet_message() {
                        self.signal_event(h);
                    }
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                // ReceiveMessage.
                Some(1) => {
                    const NO_MESSAGES: u32 = 128 | (3 << 9); // am, "no message"
                    match self.next_applet_message() {
                        Some(message) => {
                            self.write_ipc_response(tls, 0, &[], &message.to_le_bytes(), &[])
                        }
                        None => self.write_ipc_response(tls, NO_MESSAGES, &[], &[], &[]),
                    }
                }
                // GetOperationMode: Handheld is 0, Console is 1.
                Some(5) => {
                    let mode = self.operation_mode() as u32;
                    self.write_ipc_response(tls, 0, &[], &mode.to_le_bytes(), &[])
                }
                // GetPerformanceMode.
                Some(6) => {
                    let mode = self.operation_mode().performance_mode();
                    self.write_ipc_response(tls, 0, &[], &mode.to_le_bytes(), &[])
                }
                Some(9) => self.write_ipc_response(tls, 0, &[], &1u32.to_le_bytes(), &[]), // GetCurrentFocusState: InFocus
                // GetBootMode: Normal.
                Some(8) => self.write_ipc_response(tls, 0, &[], &0u8.to_le_bytes(), &[]),
                // GetAcquiredSleepLockEvent: never signalled.
                Some(13) => {
                    let h = match self.sleep_lock_event {
                        Some(h) => h,
                        None => {
                            let h = self.alloc_event("am:sleep-lock", false);
                            self.sleep_lock_event = Some(h);
                            h
                        }
                    };
                    if self.sleep_lock_acquired {
                        self.signal_event(h);
                    }
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                // GetDefaultDisplayResolutionChangeEvent: fired on dock or undock.
                Some(61) => {
                    let h = match self.display_resolution_event {
                        Some(h) => h,
                        None => {
                            let h = self.alloc_event("am:display-resolution-changed", true);
                            self.display_resolution_event = Some(h);
                            h
                        }
                    };
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                // GetDefaultDisplayResolution.
                Some(60) => {
                    let (width, height) = self.operation_mode().display_size();
                    let mut raw = Vec::with_capacity(8);
                    raw.extend_from_slice(&width.to_le_bytes());
                    raw.extend_from_slice(&height.to_le_bytes());
                    self.write_ipc_response(tls, 0, &[], &raw, &[])
                }
                // RequestToAcquireSleepLock: granted at once.
                Some(10) => {
                    self.sleep_lock_acquired = true;
                    if let Some(h) = self.sleep_lock_event {
                        self.signal_event(h);
                    }
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // ReleaseSleepLock / ReleaseSleepLockTransiently.
                Some(11) | Some(12) => {
                    self.sleep_lock_acquired = false;
                    if let Some(h) = self.sleep_lock_event {
                        self.clear_event(h);
                    }
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // SetCpuBoostMode: no clock governor to move.
                Some(66) => {
                    self.warn_stub(&iface, cmd_id, "accepted; there is no clock to move");
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // SetRequestExitToLibraryAppletAtExecuteNextProgramEnabled.
                Some(900) => {
                    self.warn_stub(&iface, cmd_id, "the exit-request latch is not recorded");
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "am:application-functions" => match cmd_id {
                // PopLaunchParameter(u32 kind) -> IStorage, handed over once.
                Some(1) => {
                    const LAUNCH_PARAMETER_NOT_FOUND: u32 = 128 | (2 << 9);
                    let kind = self.mem.read_u32(self.ipc_request_data(tls))?;
                    match self.am_launch_parameters.remove(&kind) {
                        Some(data) => {
                            let key = self.reply_with_interface(tls, handle, "am:storage")?;
                            self.am_storages.insert(key, data);
                            Ok(())
                        }
                        None => {
                            self.write_ipc_response(tls, LAUNCH_PARAMETER_NOT_FOUND, &[], &[], &[])
                        }
                    }
                }
                // EnsureSaveData.
                Some(20) => {
                    self.warn_stub(&iface, cmd_id, "0 bytes ensured; no save was created");
                    self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[])
                }
                // ExtendSaveData(u8 type, u128 uid, s64 size, s64 journal): granted and remembered.
                Some(25) => {
                    let data = self.ipc_request_data(tls);
                    self.save_data_quota.size = self.mem.read_u64(data.wrapping_add(0x18))? as i64;
                    self.save_data_quota.journal_size =
                        self.mem.read_u64(data.wrapping_add(0x20))? as i64;
                    self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[])
                }
                // GetSaveDataSize(u8 type, u128 uid) -> two s64s.
                Some(26) => {
                    let quota = self.save_data_quota;
                    self.write_save_data_pair(tls, quota.size, quota.journal_size)
                }
                // GetSaveDataSizeMax / GetDeviceSaveDataSizeMax.
                Some(28) => {
                    let quota = self.save_data_quota;
                    self.write_save_data_pair(tls, quota.size_max, quota.journal_size_max)
                }
                Some(35) => {
                    let quota = self.save_data_quota;
                    self.write_save_data_pair(
                        tls,
                        quota.device_size_max,
                        quota.device_journal_size_max,
                    )
                }
                // GetCacheStorageMax -> s32, then s64 at +8.
                Some(29) => {
                    let quota = self.save_data_quota;
                    let mut out = Vec::with_capacity(16);
                    out.extend_from_slice(&quota.cache_storage_index_max.to_le_bytes());
                    out.extend_from_slice(&[0u8; 4]);
                    out.extend_from_slice(&quota.cache_storage_size_max.to_le_bytes());
                    self.write_ipc_response(tls, 0, &[], &out, &[])
                }
                // CreateCacheStorage(u16 index, s64 size, s64 journal).
                Some(27) => {
                    let mut out = Vec::with_capacity(16);
                    out.extend_from_slice(&1u32.to_le_bytes());
                    out.extend_from_slice(&[0u8; 4]);
                    out.extend_from_slice(&0u64.to_le_bytes());
                    self.write_ipc_response(tls, 0, &[], &out, &[])
                }
                // GetDesiredLanguage -> `nn::settings::LanguageCode`.
                Some(21) => {
                    let code = self.system_settings().language_code;
                    self.write_ipc_response(tls, 0, &[], &code.to_le_bytes(), &[])
                }
                // GetDisplayVersion -> a 16-byte version string.
                Some(23) => {
                    let mut version = [0u8; 16];
                    version[..5].copy_from_slice(b"1.0.0");
                    self.write_ipc_response(tls, 0, &[], &version, &[])
                }
                // BeginBlockingHomeButton{ShortAndLongPressed,} and End.
                Some(30) | Some(31) | Some(32) | Some(33) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // NotifyRunning.
                Some(40) => self.write_ipc_response(tls, 0, &[], &1u8.to_le_bytes(), &[]),
                // GetPseudoDeviceId -> 16 bytes.
                Some(50) => {
                    self.warn_stub(&iface, cmd_id, "an all-zero device id");
                    self.write_ipc_response(tls, 0, &[], &[0u8; 16], &[])
                }
                // GetGpuErrorDetectedSystemEvent: never signalled.
                Some(130) => {
                    self.warn_stub(&iface, cmd_id, "an event nothing here ever signals");
                    let h = self.alloc_event("am:gpu-error", true);
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                // SetTerminateResult(Result).
                Some(22) => {
                    let result = self.mem.read_u32(self.ipc_request_data(tls))?;
                    self.am_terminate_result = result;
                    if result != 0 {
                        let (module, description) = (result & 0x1ff, result >> 9);
                        self.diagnostic(
                            Level::Warn,
                            &format!(
                                "[am] the title set a terminate result of {result:#x} \
                                 ({module}-{description:04})"
                            ),
                        );
                    }
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetLastApplicationExitReason -> u32.
                Some(200) => {
                    let reason = self.am_terminate_result;
                    self.write_ipc_response(tls, 0, &[], &reason.to_le_bytes(), &[])
                }
                // InitializeGamePlayRecording / SetGamePlayRecordingState / SetDelayTimeToAbortOnGpuError.
                Some(66) | Some(67) | Some(131) => {
                    self.warn_stub(&iface, cmd_id, "accepted and not recorded");
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // Copyright frame buffer, image and visibility.
                Some(100) | Some(101) | Some(102) => {
                    self.warn_stub(&iface, cmd_id, "accepted, and no capture draws it");
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // 210 (20.0.0+, unnamed): one out event, never signalled.
                Some(210) => {
                    self.warn_stub(&iface, cmd_id, "an event nothing here ever signals");
                    let h = match self.application_functions_210_event {
                        Some(h) => h,
                        None => {
                            let h = self.alloc_event("am:application-functions-210", true);
                            self.application_functions_210_event = Some(h);
                            h
                        }
                    };
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ISelfController.
            "am:self-controller" => match cmd_id {
                // Setters and notifiers whose whole reply is a Result.
                Some(0..=4) | Some(10..=16) | Some(19) | Some(51) | Some(60) | Some(64)
                | Some(65) | Some(72) | Some(100) | Some(110) | Some(130) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // Set/GetIdleTimeDetectionExtension, SetAutoSleepDisabled / IsAutoSleepDisabled.
                Some(62) => {
                    let data = self.ipc_request_data(tls);
                    self.idle_time_detection_extension = self.mem.read_u32(data).unwrap_or(0);
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                Some(63) => {
                    let extension = self.idle_time_detection_extension;
                    self.write_ipc_response(tls, 0, &[], &extension.to_le_bytes(), &[])
                }
                Some(68) => {
                    let data = self.ipc_request_data(tls);
                    self.auto_sleep_disabled = self.mem.read_u8(data).unwrap_or(0) != 0;
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                Some(69) => {
                    let disabled = u8::from(self.auto_sleep_disabled);
                    self.write_ipc_response(tls, 0, &[], &[disabled], &[])
                }
                // SetHandlesRequestToDisplay: queues `RequestToDisplay`.
                Some(50) => {
                    let data = self.ipc_request_data(tls);
                    if self.mem.read_u8(data).unwrap_or(0) != 0 {
                        self.queue_applet_message(super::AppletMessage::RequestToDisplay);
                    }
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetLibraryAppletLaunchableEvent: signalled.
                Some(9) => {
                    let h = self.kept_event("am:library-applet-launchable", handle);
                    self.signal_event(h);
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                // GetAccumulatedSuspendedTickChangedEvent.
                Some(91) => {
                    self.warn_stub(&iface, cmd_id, "an event nothing here ever signals");
                    let h = self.kept_event("am:accumulated-suspended-tick-changed", handle);
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                // Unknown230(u32) -> u16.
                Some(230) => {
                    self.warn_stub(&iface, cmd_id, "an unknown command, answered 0");
                    self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[])
                }
                // GetAccumulatedSuspendedTickValue.
                Some(90) => self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[]),
                // IsSystemBufferSharingEnabled: false.
                Some(41) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetSystemSharedBufferHandle -> buffer id;
                // GetSystemSharedLayerHandle -> buffer id + layer id.
                Some(43) => self.write_ipc_response(tls, 0, &[], &1u64.to_le_bytes(), &[]),
                Some(42) => {
                    let mut raw = Vec::with_capacity(16);
                    raw.extend_from_slice(&1u64.to_le_bytes());
                    raw.extend_from_slice(&1u64.to_le_bytes());
                    self.write_ipc_response(tls, 0, &[], &raw, &[])
                }
                // CreateManagedDisplayLayer: `vi` models one layer, id 1.
                Some(40) => self.write_ipc_response(tls, 0, &[], &1u64.to_le_bytes(), &[]),
                // CreateManagedDisplaySeparableLayer.
                Some(44) => {
                    // The recording layer is 0 so the caller does not open layer 1 twice.
                    let mut raw = Vec::with_capacity(16);
                    raw.extend_from_slice(&1u64.to_le_bytes());
                    raw.extend_from_slice(&0u64.to_le_bytes());
                    self.write_ipc_response(tls, 0, &[], &raw, &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IWindowController and IDisplayController.
            "am:display-controller" => match cmd_id {
                // Acquire*CaptureSharedBuffer: a real, never-written slot past the two framebuffers.
                Some(22) | Some(24) | Some(26) => {
                    let mut raw = Vec::with_capacity(8);
                    raw.extend_from_slice(&[1u8, 0, 0, 0]); // was_written = true
                    raw.extend_from_slice(
                        &(super::SHARED_BUFFER_USABLE_SLOTS as i32).to_le_bytes(),
                    );
                    self.write_ipc_response(tls, 0, &[], &raw, &[])
                }
                // Capture releases and clears.
                Some(8) | Some(20) | Some(21) | Some(23) | Some(25) | Some(27) | Some(28) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // Update{LastForeground,CallerApplet}CaptureImage.
                Some(1) | Some(4) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // Get*CaptureImageEx: a black 1280x720 RGBA8888 image into the out buffer.
                Some(5) | Some(6) | Some(7) => {
                    if let Some((addr, size)) = self.ipc_output_buffer(tls, 0) {
                        let page = crate::mem::PAGE_SIZE as u32;
                        let end = addr.saturating_add(size);
                        let mut at = addr;
                        while at < end {
                            let run = (page - at % page).min(end - at);
                            if self.mem.fill_le(at, 1, 0, run).is_err() {
                                break;
                            }
                            at += run;
                        }
                    }
                    self.write_ipc_response(tls, 0, &[], &1u8.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "am:window-controller" => match cmd_id {
                // GetAppletResourceUserId / GetAppletResourceUserIdOfCallerApplet.
                Some(1) | Some(2) => self.write_ipc_response(tls, 0, &[], &1u64.to_le_bytes(), &[]),
                // AcquireForegroundRights / ReleaseForegroundRights / RejectToChangeIntoBackground.
                Some(10) | Some(11) | Some(12) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IAudioController.
            "am:audio-controller" => match cmd_id {
                // SetExpectedMasterVolume / ChangeMainAppletMasterVolume /
                // SetTransparentVolumeRate.
                Some(0) | Some(3) | Some(4) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // Get{Main,Library}AppletExpectedMasterVolume -> an f32.
                Some(1) | Some(2) => {
                    self.write_ipc_response(tls, 0, &[], &1.0f32.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IAppletCommonFunctions.
            "am:applet-common-functions" => match cmd_id {
                // Set/GetHomeButtonDoubleClickEnabled.
                Some(50) => {
                    let data = self.ipc_request_data(tls);
                    self.home_button_double_click_enabled =
                        self.mem.read_u8(data).unwrap_or(0) != 0;
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                Some(51) => {
                    let enabled = u8::from(self.home_button_double_click_enabled);
                    self.write_ipc_response(tls, 0, &[], &[enabled], &[])
                }
                // SetCpuBoostRequestPriority.
                Some(70) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // 20.0.0+, unnamed: one u16 out (per Eden).
                Some(350) => self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ISystemProcessCommonFunctions: hands back an IApplicationObserver.
            "am:system-process-common-functions" => match cmd_id {
                Some(1) => {
                    self.reply_with_interface(tls, handle, "am:application-observer")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IHomeMenuFunctions.
            "am:home-menu-functions" => match cmd_id {
                // RequestToGetForeground / LockForeground / UnlockForeground.
                Some(10) | Some(11) | Some(12) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // PopFromGeneralChannel: always empty.
                Some(20) => self.write_ipc_response(tls, AM_NO_DATA_IN_CHANNEL, &[], &[], &[]),
                // GetPopFromGeneralChannelEvent: never signalled.
                Some(21) => {
                    let h = match self.general_channel_event {
                        Some(h) => h,
                        None => {
                            let h = self.alloc_event("am:general-channel", true);
                            self.general_channel_event = Some(h);
                            h
                        }
                    };
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                // GetHomeButtonWriterLockAccessor / GetWriterLockAccessorEx.
                Some(30) | Some(31) => {
                    self.reply_with_interface(tls, handle, "am:lock-accessor")?;
                    Ok(())
                }
                // IsSleepEnabled / IsRebootEnabled.
                Some(40) | Some(41) => self.write_ipc_response(tls, 0, &[], &[1u8], &[]),
                // IsForceTerminateApplicationDisabledForDebug -> bool.
                Some(110) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                // SetLastApplicationExitReason.
                Some(1000) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ILockAccessor.
            "am:lock-accessor" => match cmd_id {
                // TryLock(bool return_handle) -> (bool locked, event).
                Some(1) => {
                    let want_handle =
                        self.mem.read_u8(self.ipc_request_data(tls)).unwrap_or(0) != 0;
                    let h = self.am_lock_accessor_event();
                    if want_handle {
                        self.write_ipc_reply(tls, 0, &[h], &[], &[1u8], &[])
                    } else {
                        self.write_ipc_response(tls, 0, &[], &[1u8], &[])
                    }
                }
                // Unlock.
                Some(2) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetEvent -> the event that says the lock is free.
                Some(3) => {
                    let h = self.am_lock_accessor_event();
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                // IsLocked -> bool.
                Some(4) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IGlobalStateController. Sleep, shutdown and reboot (0-4) are not implemented.
            "am:global-state-controller" => match cmd_id {
                // IsAutoPowerDownRequested.
                Some(9) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                // Idle policy, CEC, HOME long press and display resolution settings.
                Some(10) | Some(11) | Some(12) | Some(13) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // ShouldSleepOnBoot.
                Some(14) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                // GetHdcpAuthenticationFailedEvent.
                Some(15) => {
                    let h = self.alloc_event("am:hdcp-failed", true);
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IProcessWindingController.
            "am:process-winding-controller" => match cmd_id {
                // GetLaunchReason: all zero is a normal start.
                Some(0) => self.write_ipc_response(tls, 0, &[], &[0u8; 4], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ILibraryAppletSelfAccessor.
            "am:library-applet-self-accessor" => match cmd_id {
                // GetLibraryAppletInfo: AllForeground.
                Some(11) => {
                    let mut info = [0u8; 8];
                    info[..4].copy_from_slice(&applet_id_for(self.program_id()).to_le_bytes());
                    self.write_ipc_response(tls, 0, &[], &info, &[])
                }
                // ShouldSetGpuTimeSliceManually.
                Some(150) => self.write_ipc_response(tls, 0, &[], &0u8.to_le_bytes(), &[]),
                // GetMainAppletIdentityInfo / GetCallerAppletIdentityInfo: the home menu.
                Some(12) | Some(14) => {
                    let info = home_menu_identity();
                    self.write_ipc_response(tls, 0, &[], &info, &[])
                }
                // CanUseApplicationCore.
                Some(13) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                // GetMainAppletApplicationDesiredLanguage.
                Some(60) => {
                    let code = self.system_settings().language_code;
                    self.write_ipc_response(tls, 0, &[], &code.to_le_bytes(), &[])
                }
                // GetCallerAppletIdentityInfoStack: the home menu as the single entry.
                Some(17) => {
                    let info = home_menu_identity();
                    let (addr, size) = self.ipc_output_buffer(tls, 0).unwrap_or((0, 0));
                    let room = if addr == 0 {
                        0
                    } else {
                        size as usize / info.len()
                    };
                    let count = room.min(1);
                    if count == 1 {
                        for (index, &byte) in info.iter().enumerate() {
                            self.mem.write_u8(addr.wrapping_add(index as u32), byte)?;
                        }
                    }
                    self.write_ipc_response(tls, 0, &[], &(count as i32).to_le_bytes(), &[])
                }
                // GetDesirableKeyboardLayout: the console's layout.
                Some(19) => {
                    let layout = self.system_settings().keyboard_layout;
                    self.write_ipc_response(tls, 0, &[], &layout.to_le_bytes(), &[])
                }
                // PushOutData(IStorage): kept for [`Cpu::library_applet_results`].
                Some(1) => {
                    let data = self
                        .ipc_input_object_key(tls, handle, 0)
                        .and_then(|key| self.am_storages.get(&key))
                        .cloned();
                    let summary = match &data {
                        Some(data) => applet_result_summary(self.program_id(), data),
                        None => "a storage this session never handed out".to_owned(),
                    };
                    self.diagnostic(
                        Level::Info,
                        &format!(
                            "[am] the {} applet finished: {summary}",
                            applet_name(applet_id_for(self.program_id()))
                        ),
                    );
                    self.am_out_data.push(data.unwrap_or_default());
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // PushInteractiveOutData(IStorage): kept for the host.
                Some(3) => {
                    let data = self
                        .ipc_input_object_key(tls, handle, 0)
                        .and_then(|key| self.am_storages.get(&key))
                        .cloned()
                        .unwrap_or_default();
                    if self
                        .unimplemented_ipc
                        .insert(("am:applet-interactive-output".to_string(), cmd_id))
                    {
                        self.diagnostic(
                            Level::Warn,
                            &format!(
                                "[am] the applet is talking to its caller ({} bytes); nothing \
                                 here launched it, so only the host can answer",
                                data.len()
                            ),
                        );
                    }
                    self.am_interactive_out.push(data);
                    if self.am_interactive_out.len() > MAX_INTERACTIVE_MESSAGES {
                        self.am_interactive_out.remove(0);
                    }
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetPopInDataEvent / GetPopInteractiveInDataEvent.
                Some(5) | Some(6) => {
                    let queue = if cmd_id == Some(5) {
                        AppletQueue::InData
                    } else {
                        AppletQueue::InteractiveInData
                    };
                    let event = self.applet_queue_event(queue);
                    self.refresh_applet_pop_events();
                    self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                }
                // ExitProcessAndReturn.
                Some(10) => {
                    self.halted = true;
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // An unnamed init-time setter taking 16 bytes.
                Some(160) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // PopInData -> IStorage.
                Some(0) => self.pop_applet_storage(tls, handle, AppletQueue::InData),
                // PopInteractiveInData -> IStorage.
                Some(2) => self.pop_applet_storage(tls, handle, AppletQueue::InteractiveInData),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ILibraryAppletCreator.
            "am:library-applet-creator" => match cmd_id {
                // CreateLibraryApplet / CreateLibraryAppletEx (adds a thread id).
                Some(0) | Some(3) => {
                    let at = self.ipc_request_data(tls);
                    let id = self.mem.read_u32(at)?;
                    let mode = self.mem.read_u32(at.wrapping_add(4))?;
                    self.diagnostic(
                        Level::Warn,
                        &format!(
                            "[am] CreateLibraryApplet: {} (mode {mode}) — nothing here runs it, \
                         so it will report itself cancelled",
                            applet_name(id)
                        ),
                    );
                    let key =
                        self.reply_with_interface(tls, handle, "am:library-applet-accessor")?;
                    self.am_applets.insert(key, LibraryApplet::new(id, mode));
                    Ok(())
                }
                // TerminateAllLibraryApplets / AreAnyLibraryAppletsLeft.
                Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                Some(2) => self.write_ipc_response(tls, 0, &[], &0u8.to_le_bytes(), &[]),
                // CreateStorage(s64 size) -> IStorage.
                Some(10) => {
                    const MAX_STORAGE: u64 = 64 * 1024 * 1024;
                    /// `KERNELRESULT(OutOfMemory)`.
                    const OUT_OF_MEMORY: u32 = 1 | (104 << 9);
                    let size = self.mem.read_u64(self.ipc_request_data(tls))?;
                    if size > MAX_STORAGE {
                        return self.write_ipc_response(tls, OUT_OF_MEMORY, &[], &[], &[]);
                    }
                    let key = self.reply_with_interface(tls, handle, "am:storage")?;
                    self.am_storages.insert(key, vec![0u8; size as usize]);
                    Ok(())
                }
                // CreateTransferMemoryStorage / CreateHandleStorage: no backing memory to read.
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ILibraryAppletAccessor: the applet finishes, cancelled, as soon as it starts.
            "am:library-applet-accessor" => {
                let key = self.ipc_object_key(tls, handle);
                match cmd_id {
                    // GetAppletStateChangedEvent.
                    Some(0) => {
                        let event = self.library_applet_event(key, STATE_CHANGED_EVENT);
                        if self.library_applet_finished(key) {
                            self.signal_event(event);
                        }
                        self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                    }
                    // IsCompleted.
                    Some(1) => {
                        let done = u8::from(self.library_applet_finished(key));
                        self.write_ipc_response(tls, 0, &[], &done.to_le_bytes(), &[])
                    }
                    // Start / RequestExit / Terminate.
                    Some(10) | Some(20) | Some(25) => {
                        if let Some(applet) = self.am_applets.get_mut(&key) {
                            applet.finish();
                        }
                        self.signal_library_applet_event(key, STATE_CHANGED_EVENT);
                        self.write_ipc_response(tls, 0, &[], &[], &[])
                    }
                    // GetResult: cancelled.
                    Some(30) => {
                        /// `am` description 22, `LibAppletExitReason_Canceled`.
                        const CANCELLED: u32 = 128 | (22 << 9);
                        self.write_ipc_response(tls, CANCELLED, &[], &[], &[])
                    }
                    // PushInData / PushExtraStorage / PushInteractiveInData: dropped.
                    Some(100) | Some(102) | Some(103) => {
                        self.write_ipc_response(tls, 0, &[], &[], &[])
                    }
                    // PopOutData / PopInteractiveOutData: nothing produced.
                    Some(101) | Some(104) => {
                        const NO_DATA: u32 = 128 | (3 << 9);
                        self.write_ipc_response(tls, NO_DATA, &[], &[], &[])
                    }
                    // GetPopOutDataEvent / GetPopInteractiveOutDataEvent: never signalled.
                    Some(105) => {
                        let event = self.library_applet_event(key, POP_OUT_DATA_EVENT);
                        self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                    }
                    Some(106) => {
                        let event = self.library_applet_event(key, POP_INTERACTIVE_OUT_DATA_EVENT);
                        self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                    }
                    // NeedsToExitProcess.
                    Some(110) => self.write_ipc_response(tls, 0, &[], &0u8.to_le_bytes(), &[]),
                    // GetLibraryAppletInfo.
                    Some(120) => {
                        let mut info = [0u8; 8];
                        if let Some(applet) = self.am_applets.get(&key) {
                            info[..4].copy_from_slice(&applet.id.to_le_bytes());
                            info[4..].copy_from_slice(&applet.mode.to_le_bytes());
                        }
                        self.write_ipc_response(tls, 0, &[], &info, &[])
                    }
                    // RequestForAppletToGetForeground.
                    Some(150) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                    // GetIndirectLayerConsumerHandle.
                    _ => self.unimplemented_command(tls, &iface, cmd_id),
                }
            }
            // `am`'s IStorage, distinct from `fsp-srv`'s.
            "am:storage" => match cmd_id {
                // Open -> IStorageAccessor.
                Some(0) => {
                    let storage = self.ipc_object_key(tls, handle);
                    let accessor = self.reply_with_interface(tls, handle, "am:storage-accessor")?;
                    self.am_storage_of.insert(accessor, storage);
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "am:storage-accessor" => {
                let storage = self
                    .am_storage_of
                    .get(&self.ipc_object_key(tls, handle))
                    .copied()
                    .unwrap_or(0);
                match cmd_id {
                    // GetSize -> s64.
                    Some(0) => {
                        let size = self.am_storages.get(&storage).map_or(0, |d| d.len()) as u64;
                        self.write_ipc_response(tls, 0, &[], &size.to_le_bytes(), &[])
                    }
                    // Write(s64 offset, buffer<in>) / Read(s64 offset, buffer<out>).
                    Some(10) => {
                        let offset = self.mem.read_u64(self.ipc_request_data(tls))? as usize;
                        let Some((addr, len)) = self.ipc_input_buffer(tls, 0) else {
                            return self.write_ipc_response(tls, 0, &[], &[], &[]);
                        };
                        let mut bytes = Vec::with_capacity(len as usize);
                        for i in 0..len {
                            bytes.push(self.mem.read_u8(addr.wrapping_add(i))?);
                        }
                        let data = self.am_storages.entry(storage).or_default();
                        if data.len() < offset + bytes.len() {
                            data.resize(offset + bytes.len(), 0);
                        }
                        data[offset..offset + bytes.len()].copy_from_slice(&bytes);
                        self.write_ipc_response(tls, 0, &[], &[], &[])
                    }
                    Some(11) => {
                        let offset = self.mem.read_u64(self.ipc_request_data(tls))? as usize;
                        let data = self.am_storages.get(&storage).cloned().unwrap_or_default();
                        if let Some((addr, len)) = self.ipc_output_buffer(tls, 0) {
                            let end = data.len().min(offset.saturating_add(len as usize));
                            let chunk = if offset < end {
                                &data[offset..end]
                            } else {
                                &[][..]
                            };
                            for (i, &b) in chunk.iter().enumerate() {
                                self.mem.write_u8(addr.wrapping_add(i as u32), b)?;
                            }
                        }
                        self.write_ipc_response(tls, 0, &[], &[], &[])
                    }
                    _ => self.unimplemented_command(tls, &iface, cmd_id),
                }
            }
            // IDisplayController, IDebugFunctions, and unnamed sessions.
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
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
mod tests {
    use crate::cpu::ipc::testing::*;
    use crate::cpu::Cpu;

    /// `AppletId_LibraryAppletWeb`.
    const APPLET_WEB: u32 = 0x13;

    #[test]
    fn the_system_process_common_functions_chain_hands_back_real_sessions() {
        let mut cpu = request(false, 450, &[]);
        cpu.register_service_handle(9, "appletAE");
        cpu.set_applet_is_application(false);
        cpu.applet_request(TLS, 9, Some(450)).unwrap();
        let functions = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
        assert_ne!(
            functions, 0,
            "GetSystemProcessCommonFunctions moved no session back"
        );
        assert_eq!(
            cpu.service_name(functions),
            Some("am:system-process-common-functions")
        );

        marshal(&mut cpu, false, 1, &[]);
        cpu.applet_request(TLS, functions, Some(1)).unwrap();
        let observer = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
        assert_ne!(observer, 0, "cmd 1 moved no observer back");
        assert_eq!(cpu.service_name(observer), Some("am:application-observer"));

        marshal(&mut cpu, false, 460, &[]);
        cpu.applet_request(TLS, 9, Some(460)).unwrap();
        let alternative = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
        assert_ne!(
            alternative, 0,
            "GetAppletAlternativeFunctions moved no session back"
        );
        assert_eq!(
            cpu.service_name(alternative),
            Some("am:applet-alternative-functions")
        );

        // Neither opens a proxy, so the applet-kind flag is unchanged.
        assert!(!cpu.applet_is_application);
    }

    /// Marshal a `PushOutData`-shaped request carrying one object.
    fn push_storage(cpu: &mut Cpu, command_id: u32, storage: u64) {
        for i in (0..0x200u32).step_by(4) {
            cpu.mem.write_u32(TLS + i, 0).unwrap();
        }
        cpu.mem.write_u32(TLS, 4).unwrap();
        cpu.mem.write_u32(TLS + 4, 8 | (1 << 31)).unwrap();
        cpu.mem.write_u32(TLS + 8, 1 << 5).unwrap();
        cpu.mem.write_u32(TLS + 12, storage as u32).unwrap();
        cpu.mem.write_u32(TLS + 0x10, SFCI).unwrap();
        cpu.mem.write_u32(TLS + 0x18, command_id).unwrap();
    }

    #[test]
    fn the_applet_result_is_kept_rather_than_dropped() {
        const CONTROLLER: u64 = 0x0100_0000_0000_1003;
        const PUSH_OUT_DATA: u32 = 1;
        // { s8 player_count = 1, pad[3], u32 selected_id = 0, u32 result = 0 }.
        let mut result = vec![0u8; 0xC];
        result[0] = 1;

        const STORAGE: u64 = 0x21;
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        cpu.set_program_id(CONTROLLER);
        cpu.register_service_handle(9, "am:library-applet-self-accessor");
        cpu.register_service_handle(STORAGE, "am:storage");
        cpu.am_storages
            .insert(Cpu::object_key(STORAGE, 0), result.clone());
        push_storage(&mut cpu, PUSH_OUT_DATA, STORAGE);
        cpu.applet_request(TLS, 9, Some(PUSH_OUT_DATA)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "push refused");
        assert_eq!(cpu.library_applet_results(), [result.clone()]);

        const STORAGE_OBJECT: u32 = 5;
        let mut cpu = request(true, PUSH_OUT_DATA, &[]);
        cpu.set_program_id(CONTROLLER);
        cpu.record_domain_object(9, 7, "am:library-applet-self-accessor");
        cpu.record_domain_object(9, STORAGE_OBJECT, "am:storage");
        cpu.am_storages
            .insert(Cpu::object_key(9, STORAGE_OBJECT), result.clone());
        // num_in_objects, then a `data_size` of just the `CmifInHeader`.
        cpu.mem.write_u8(TLS + 0x11, 1).unwrap();
        cpu.mem.write_u16(TLS + 0x12, 0x10).unwrap();
        cpu.mem.write_u32(TLS + 0x30, STORAGE_OBJECT).unwrap();
        cpu.applet_request(TLS, 9, Some(PUSH_OUT_DATA)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x28).unwrap(), 0, "push refused");
        assert_eq!(cpu.library_applet_results(), [result.clone()]);

        assert_eq!(
            super::applet_result_summary(CONTROLLER, &result),
            "confirmed, 1 player(s), npad 0"
        );
        result[8] = 2;
        assert_eq!(
            super::applet_result_summary(CONTROLLER, &result),
            "cancelled, 1 player(s), npad 0"
        );
    }

    #[test]
    fn the_applet_can_be_answered_by_the_host_that_started_it() {
        const SWKBD: u64 = 0x0100_0000_0000_1008;
        const PUSH_INTERACTIVE_OUT_DATA: u32 = 3;
        const POP_INTERACTIVE_IN_DATA: u32 = 2;
        const GET_POP_INTERACTIVE_IN_DATA_EVENT: u32 = 6;
        const NO_DATA: u32 = 128 | (3 << 9);
        const STORAGE: u64 = 0x21;

        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        cpu.set_program_id(SWKBD);
        cpu.register_service_handle(9, "am:library-applet-self-accessor");
        cpu.register_service_handle(STORAGE, "am:storage");
        // `u64 size` then the text.
        let mut message = 4u64.to_le_bytes().to_vec();
        message.extend_from_slice(&[0x68, 0, 0x69, 0]);
        cpu.am_storages
            .insert(Cpu::object_key(STORAGE, 0), message.clone());
        push_storage(&mut cpu, PUSH_INTERACTIVE_OUT_DATA, STORAGE);
        cpu.applet_request(TLS, 9, Some(PUSH_INTERACTIVE_OUT_DATA))
            .unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "push refused");
        assert_eq!(cpu.library_applet_interactive_messages(), [message]);

        marshal(&mut cpu, false, GET_POP_INTERACTIVE_IN_DATA_EVENT, &[]);
        cpu.applet_request(TLS, 9, Some(GET_POP_INTERACTIVE_IN_DATA_EVENT))
            .unwrap();
        let event = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
        assert_ne!(event, 0, "no event handed back");
        assert_eq!(cpu.event_name(event), Some("am:applet-interactive-in-data"));
        assert_eq!(cpu.event_signaled(event), Some(false), "answered already");

        marshal(&mut cpu, false, POP_INTERACTIVE_IN_DATA, &[]);
        cpu.applet_request(TLS, 9, Some(POP_INTERACTIVE_IN_DATA))
            .unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), NO_DATA, "pop");

        // `SwkbdTextCheckResult` Success and an empty message.
        let answer = vec![0u8; 0x8];
        cpu.push_applet_interactive_in_data(answer.clone());
        assert_eq!(cpu.event_signaled(event), Some(true), "answer unannounced");

        marshal(&mut cpu, false, POP_INTERACTIVE_IN_DATA, &[]);
        cpu.applet_request(TLS, 9, Some(POP_INTERACTIVE_IN_DATA))
            .unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "pop refused");
        let storage = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
        assert_eq!(cpu.service_name(storage), Some("am:storage"));
        assert_eq!(cpu.am_storages[&Cpu::object_key(storage, 0)], answer);

        assert_eq!(cpu.event_signaled(event), Some(false), "still signalled");
    }

    #[test]
    fn the_in_data_event_is_signalled_while_there_is_something_to_pop() {
        const SWKBD: u64 = 0x0100_0000_0000_1008;
        const GET_POP_IN_DATA_EVENT: u32 = 5;
        const POP_IN_DATA: u32 = 0;

        let mut cpu = request(false, GET_POP_IN_DATA_EVENT, &[]);
        cpu.set_program_id(SWKBD);
        cpu.seed_applet_launch_arguments();
        cpu.register_service_handle(9, "am:library-applet-self-accessor");
        cpu.applet_request(TLS, 9, Some(GET_POP_IN_DATA_EVENT))
            .unwrap();
        let event = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
        assert_eq!(cpu.event_name(event), Some("am:applet-in-data"));
        assert_eq!(cpu.event_signaled(event), Some(true), "storages waiting");

        for _ in 0..3 {
            marshal(&mut cpu, false, POP_IN_DATA, &[]);
            cpu.applet_request(TLS, 9, Some(POP_IN_DATA)).unwrap();
            assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "pop refused");
        }
        assert_eq!(cpu.event_signaled(event), Some(false), "queue not empty");
    }

    #[test]
    fn the_idle_detection_setters_are_accepted_rather_than_refused() {
        for (cmd, payload) in [
            (60u32, &[0u8; 0x10][..]),
            (64, &[0u8; 4][..]),
            (65, &[][..]),
            (72, &[0u8; 4][..]),
        ] {
            let mut cpu = request(false, cmd, payload);
            cpu.register_service_handle(9, "am:self-controller");
            cpu.applet_request(TLS, 9, Some(cmd)).unwrap();
            assert_eq!(
                cpu.mem.read_u32(TLS + 0x18).unwrap(),
                0,
                "cmd {cmd} refused"
            );
        }
    }

    #[test]
    fn the_auto_sleep_settings_read_back_what_was_set() {
        const EXTENSION: u32 = 3;
        let mut cpu = request(false, 62, &EXTENSION.to_le_bytes());
        cpu.register_service_handle(9, "am:self-controller");
        cpu.applet_request(TLS, 9, Some(62)).unwrap();

        marshal(&mut cpu, false, 63, &[]);
        cpu.applet_request(TLS, 9, Some(63)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "refused");
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), EXTENSION);

        marshal(&mut cpu, false, 68, &[1u8]);
        cpu.applet_request(TLS, 9, Some(68)).unwrap();
        marshal(&mut cpu, false, 69, &[]);
        cpu.applet_request(TLS, 9, Some(69)).unwrap();
        assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 1, "auto sleep off");

        marshal(&mut cpu, false, 68, &[0u8]);
        cpu.applet_request(TLS, 9, Some(68)).unwrap();
        marshal(&mut cpu, false, 69, &[]);
        cpu.applet_request(TLS, 9, Some(69)).unwrap();
        assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 0, "auto sleep on");
    }

    #[test]
    fn the_home_button_double_click_setting_reads_back_what_was_set() {
        let mut cpu = request(false, 50, &[1u8]);
        cpu.register_service_handle(9, "am:applet-common-functions");
        cpu.applet_request(TLS, 9, Some(50)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "set refused");

        marshal(&mut cpu, false, 51, &[]);
        cpu.applet_request(TLS, 9, Some(51)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "get refused");
        assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 1, "double click on");

        marshal(&mut cpu, false, 50, &[0u8]);
        cpu.applet_request(TLS, 9, Some(50)).unwrap();
        marshal(&mut cpu, false, 51, &[]);
        cpu.applet_request(TLS, 9, Some(51)).unwrap();
        assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 0, "double click off");
    }

    #[test]
    fn every_button_lock_accessor_hands_back_a_lock() {
        const HOME_BUTTON: u32 = 0;
        for cmd in [30u32, 31, 32] {
            let mut cpu = request(false, cmd, &HOME_BUTTON.to_le_bytes());
            cpu.register_service_handle(9, "am:common-state-getter");
            cpu.applet_request(TLS, 9, Some(cmd)).unwrap();
            assert_eq!(
                cpu.mem.read_u32(TLS + 0x18).unwrap(),
                0,
                "cmd {cmd} refused"
            );
            let lock = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
            assert_ne!(lock, 0, "cmd {cmd} moved no accessor back");
            assert_eq!(cpu.service_name(lock), Some("am:lock-accessor"));
        }
    }

    #[test]
    fn am_reports_the_handheld_operation_mode_it_always_claimed_to() {
        let mut cpu = request(false, 5, &[]);
        cpu.register_service_handle(9, "am:common-state-getter");
        cpu.applet_request(TLS, 9, Some(5)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "Handheld");
    }

    #[test]
    fn the_copyright_notice_for_captures_is_accepted() {
        for cmd in [100, 101, 102] {
            let mut cpu = request(false, cmd, &[]);
            cpu.register_service_handle(9, "am:application-functions");
            cpu.applet_request(TLS, 9, Some(cmd)).unwrap();
            assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "command {cmd}");
        }
    }

    #[test]
    fn each_applet_event_is_named_after_the_interface_that_hands_it_out() {
        let mut cpu = request(false, 130, &[]);
        cpu.register_service_handle(9, "am:application-functions");
        cpu.applet_request(TLS, 9, Some(130)).unwrap();
        let event = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
        assert_ne!(
            event, 0,
            "GetGpuErrorDetectedSystemEvent handed back no handle"
        );
        assert_eq!(cpu.event_name(event), Some("am:gpu-error"));

        let mut cpu = request(false, 91, &[]);
        cpu.register_service_handle(9, "am:self-controller");
        cpu.applet_request(TLS, 9, Some(91)).unwrap();
        let event = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
        assert_ne!(event, 0);
        assert_eq!(
            cpu.event_name(event),
            Some("am:accumulated-suspended-tick-changed")
        );
        assert_eq!(cpu.event_signaled(event), Some(false));

        let mut cpu = request(false, 9, &[]);
        cpu.register_service_handle(9, "am:self-controller");
        cpu.applet_request(TLS, 9, Some(9)).unwrap();
        let event = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
        assert_ne!(event, 0);
        assert_eq!(cpu.event_name(event), Some("am:library-applet-launchable"));
        assert_eq!(cpu.event_signaled(event), Some(true));

        cpu.applet_request(TLS, 9, Some(9)).unwrap();
        assert_eq!(u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap()), event);
    }

    #[test]
    fn am_gives_back_the_terminate_result_the_title_set() {
        const TERMINATE_RESULT: u32 = 202 | (30 << 9);
        let mut cpu = request(false, 22, &TERMINATE_RESULT.to_le_bytes());
        cpu.register_service_handle(9, "am:application-functions");
        cpu.applet_request(TLS, 9, Some(22)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.am_terminate_result, TERMINATE_RESULT);

        marshal(&mut cpu, false, 200, &[]);
        cpu.applet_request(TLS, 9, Some(200)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), TERMINATE_RESULT);
    }

    #[test]
    fn the_preselected_user_is_handed_over_once_and_then_it_is_gone() {
        const LAUNCH_PARAMETER_NOT_FOUND: u32 = 128 | (2 << 9);
        const SFCO: u32 = 0x4F43_4653;

        let kind = super::LAUNCH_PARAMETER_PRESELECTED_USER.to_le_bytes();
        let mut cpu = request(false, 1, &kind);
        cpu.seed_launch_parameters();
        cpu.register_service_handle(9, "am:application-functions");
        cpu.applet_request(TLS, 9, Some(1)).unwrap();

        let storage = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
        assert_ne!(storage, 0, "PopLaunchParameter moved no storage back");
        assert_eq!(cpu.service_name(storage), Some("am:storage"));

        // Magic, version, and a non-zero uid at offset 8.
        let data = cpu.am_storages[&Cpu::object_key(storage, 0)].clone();
        assert_eq!(data.len(), 0x88);
        assert_eq!(
            u32::from_le_bytes(data[..4].try_into().unwrap()),
            0xC794_97CA
        );
        assert_eq!(data[4], 1, "layout version");
        assert_eq!(&data[8..0x18], &crate::cpu::acc::DEFAULT_USER_UID[..]);

        marshal(&mut cpu, false, 1, &kind);
        cpu.applet_request(TLS, 9, Some(1)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x10).unwrap(), SFCO);
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x18).unwrap(),
            LAUNCH_PARAMETER_NOT_FOUND
        );
    }

    #[test]
    fn a_launch_parameter_nobody_left_is_still_refused() {
        const USER_CHANNEL: u32 = 1;
        const LAUNCH_PARAMETER_NOT_FOUND: u32 = 128 | (2 << 9);

        let mut cpu = request(false, 1, &USER_CHANNEL.to_le_bytes());
        cpu.seed_launch_parameters();
        cpu.register_service_handle(9, "am:application-functions");
        cpu.applet_request(TLS, 9, Some(1)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x18).unwrap(),
            LAUNCH_PARAMETER_NOT_FOUND
        );
    }

    #[test]
    fn application_functions_210_hands_out_one_event_and_keeps_handing_out_that_one() {
        let mut cpu = request(false, 210, &[]);
        cpu.register_service_handle(9, "am:application-functions");
        cpu.applet_request(TLS, 9, Some(210)).unwrap();
        let event = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
        assert_ne!(event, 0, "command 210 handed back no event handle");
        assert_eq!(cpu.event_name(event), Some("am:application-functions-210"));

        assert_eq!(cpu.event_signaled(event), Some(false));

        marshal(&mut cpu, false, 210, &[]);
        cpu.applet_request(TLS, 9, Some(210)).unwrap();
        assert_eq!(u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap()), event);
    }

    #[test]
    fn a_stubbed_answer_names_itself_once_and_still_succeeds() {
        let mut cpu = request(false, 66, &[]);
        cpu.register_service_handle(9, "am:application-functions");
        cpu.applet_request(TLS, 9, Some(66)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "Result");
        let trace = String::from_utf8_lossy(&cpu.trace).into_owned();
        assert!(
            trace.contains("[ipc] stub: am:application-functions cmd=Some(66)"),
            "the stub went unreported: {trace:?}"
        );

        // Once per (interface, command).
        marshal(&mut cpu, false, 66, &[]);
        cpu.applet_request(TLS, 9, Some(66)).unwrap();
        let repeated = String::from_utf8_lossy(&cpu.trace)
            .matches("[ipc] stub:")
            .count();
        assert_eq!(repeated, 1, "the stub was reported on every call");
    }

    #[test]
    fn get_save_data_size_reports_the_quota_the_title_was_actually_allotted() {
        let mut payload = [0u8; 0x18];
        payload[0] = 1; // SaveDataType::Account
        payload[8..].copy_from_slice(&crate::cpu::acc::DEFAULT_USER_UID);

        // Tomodachi Life's NACP figures.
        const SAVE: i64 = 56_623_104;
        const JOURNAL: i64 = 10_485_760;

        let mut cpu = request(false, 26, &payload);
        cpu.set_save_data_quota(crate::cpu::fs::SaveDataQuota {
            size: SAVE,
            journal_size: JOURNAL,
            ..Default::default()
        });
        cpu.register_service_handle(9, "am:application-functions");
        cpu.applet_request(TLS, 9, Some(26)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "Result");
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap() as i64, SAVE);
        assert_eq!(cpu.mem.read_u64(TLS + 0x28).unwrap() as i64, JOURNAL);
    }

    #[test]
    fn extending_a_save_grants_it_and_the_size_read_back_is_the_extended_one() {
        const SIZE: i64 = 0x1200_0000;
        const JOURNAL: i64 = 0x0100_0000;
        let mut payload = [0u8; 0x28];
        payload[0] = 1; // SaveDataType::Account
        payload[8..0x18].copy_from_slice(&crate::cpu::acc::DEFAULT_USER_UID);
        payload[0x18..0x20].copy_from_slice(&SIZE.to_le_bytes());
        payload[0x20..].copy_from_slice(&JOURNAL.to_le_bytes());

        let mut cpu = request(false, 25, &payload);
        cpu.register_service_handle(9, "am:application-functions");
        cpu.applet_request(TLS, 9, Some(25)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "Result");

        marshal(&mut cpu, false, 26, &[0u8; 0x18]);
        cpu.applet_request(TLS, 9, Some(26)).unwrap();
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap() as i64, SIZE);
        assert_eq!(cpu.mem.read_u64(TLS + 0x28).unwrap() as i64, JOURNAL);
    }

    #[test]
    fn the_save_data_ceilings_are_reported_apart_from_the_sizes() {
        let quota = crate::cpu::fs::SaveDataQuota {
            size: 1,
            journal_size: 2,
            size_max: 3,
            journal_size_max: 4,
            device_size_max: 5,
            device_journal_size_max: 6,
            ..Default::default()
        };
        for (command, expected) in [(26, (1i64, 2i64)), (28, (3, 4)), (35, (5, 6))] {
            let mut cpu = request(false, command, &[0u8; 0x18]);
            cpu.set_save_data_quota(quota);
            cpu.register_service_handle(9, "am:application-functions");
            cpu.applet_request(TLS, 9, Some(command)).unwrap();
            assert_eq!(
                cpu.mem.read_u32(TLS + 0x18).unwrap(),
                0,
                "Result of {command}"
            );
            let got = (
                cpu.mem.read_u64(TLS + 0x20).unwrap() as i64,
                cpu.mem.read_u64(TLS + 0x28).unwrap() as i64,
            );
            assert_eq!(got, expected, "command {command}");
        }
    }

    #[test]
    fn a_declared_ceiling_of_zero_is_reported_as_zero() {
        let mut cpu = request(false, 28, &[]);
        cpu.set_save_data_quota(crate::cpu::fs::SaveDataQuota {
            size: 56_623_104,
            journal_size: 10_485_760,
            size_max: 0,
            journal_size_max: 0,
            ..Default::default()
        });
        cpu.register_service_handle(9, "am:application-functions");
        cpu.applet_request(TLS, 9, Some(28)).unwrap();
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), 0);
        assert_eq!(cpu.mem.read_u64(TLS + 0x28).unwrap(), 0);
    }

    #[test]
    fn get_cache_storage_max_aligns_its_size_after_its_count() {
        // s32 then s64 at +8; +4 is padding.
        let mut cpu = request(false, 29, &[]);
        cpu.set_save_data_quota(crate::cpu::fs::SaveDataQuota {
            cache_storage_index_max: 3,
            cache_storage_size_max: 0x40_0000,
            ..Default::default()
        });
        cpu.register_service_handle(9, "am:application-functions");
        cpu.applet_request(TLS, 9, Some(29)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 3, "index max");
        assert_eq!(cpu.mem.read_u64(TLS + 0x28).unwrap(), 0x40_0000, "size max");
    }

    #[test]
    fn a_title_whose_nacp_nobody_read_still_gets_room_to_save() {
        let mut payload = [0u8; 0x18];
        payload[0] = 1;
        let mut cpu = request(false, 26, &payload);
        cpu.register_service_handle(9, "am:application-functions");
        cpu.applet_request(TLS, 9, Some(26)).unwrap();
        let size = cpu.mem.read_u64(TLS + 0x20).unwrap() as i64;
        let journal = cpu.mem.read_u64(TLS + 0x28).unwrap() as i64;
        assert_eq!(size, crate::cpu::fs::DEFAULT_SAVE_DATA_SIZE);
        assert_eq!(journal, crate::cpu::fs::DEFAULT_SAVE_DATA_JOURNAL_SIZE);
        assert!(
            size >= 56_623_104,
            "default quota is smaller than a real title's save"
        );
        assert!(
            journal >= 10_485_760,
            "default journal is smaller than a real title's"
        );
    }

    fn library_applet(applet_id: u32) -> (Cpu, u64) {
        let mut payload = [0u8; 8];
        payload[..4].copy_from_slice(&applet_id.to_le_bytes());
        let mut cpu = request(false, 0, &payload);
        cpu.register_service_handle(9, "am:library-applet-creator");
        cpu.applet_request(TLS, 9, Some(0)).unwrap();
        let accessor = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
        assert_ne!(accessor, 0, "CreateLibraryApplet moved no object back");
        assert_eq!(
            cpu.service_name(accessor),
            Some("am:library-applet-accessor")
        );
        (cpu, accessor)
    }

    #[test]
    fn the_keyboard_and_the_controller_applet_pop_three_storages() {
        const SWKBD: u64 = 0x0100_0000_0000_1008;
        const CONTROLLER: u64 = 0x0100_0000_0000_1003;
        const NO_DATA: u32 = 128 | (3 << 9);

        for program_id in [SWKBD, CONTROLLER] {
            let mut cpu = request(false, 0, &[]);
            cpu.set_program_id(program_id);
            cpu.seed_applet_launch_arguments();
            cpu.register_service_handle(9, "am:library-applet-self-accessor");

            let mut sizes = Vec::new();
            for pop in 0..3 {
                write_request(&mut cpu, 0, &[]);
                cpu.applet_request(TLS, 9, Some(0)).unwrap();
                assert_eq!(
                    cpu.mem.read_u32(TLS + 0x18).unwrap(),
                    0,
                    "{program_id:#x} pop {pop} was refused"
                );
                let storage = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
                assert_eq!(cpu.service_name(storage), Some("am:storage"));
                write_request(&mut cpu, 0, &[]);
                cpu.applet_request(TLS, storage, Some(0)).unwrap();
                let accessor = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
                write_request(&mut cpu, 0, &[]);
                cpu.applet_request(TLS, accessor, Some(0)).unwrap();
                sizes.push(cpu.mem.read_u64(TLS + 0x20).unwrap());
            }

            let expected: [u64; 3] = match program_id {
                SWKBD => [0x20, 0x4C8, 0x1000],
                _ => [0x20, 0x14, 0x430],
            };
            assert_eq!(sizes, expected, "{program_id:#x} storage sizes");

            write_request(&mut cpu, 0, &[]);
            cpu.applet_request(TLS, 9, Some(0)).unwrap();
            assert_eq!(
                cpu.mem.read_u32(TLS + 0x18).unwrap(),
                NO_DATA,
                "{program_id:#x} was handed a fourth storage"
            );
        }
    }

    #[test]
    fn every_applet_title_id_maps_to_the_id_switchbrew_gives_it() {
        for (program_id, applet_id) in [
            (0x0100_0000_0000_1000u64, 0x03u32), // qlaunch
            (0x0100_0000_0000_1003, 0x0C),       // controller
            (0x0100_0000_0000_1008, 0x11),       // swkbd
            (0x0100_0000_0000_100C, 0x02),       // overlayDisp
            (0x0100_0000_0000_100D, 0x15),       // photoViewer
            (0x0100_0000_0000_1011, 0x19),       // wifiWebAuth
            (0x0100_0000_0000_1012, 0x04),       // starter, a SystemApplication
            (0x0100_0000_0000_1013, 0x1A),       // myPage
        ] {
            assert_eq!(
                super::applet_id_for(program_id),
                applet_id,
                "{program_id:#x}"
            );
        }
        assert!(super::is_library_applet(0x0100_0000_0000_1013));
        assert!(!super::is_library_applet(0x0100_0000_0000_1012));

        assert_eq!(
            super::applet_interface_version(0x0100_0000_0000_1013),
            0x1_0000
        );
        let arg = &super::applet_launch_storages(
            0x0100_0000_0000_1013,
            crate::cpu::acc::DEFAULT_USER_UID,
        )[0];
        assert_eq!(arg.len(), 0x10A8);
        assert_eq!(arg[8..24], crate::cpu::acc::DEFAULT_USER_UID);
    }

    #[test]
    fn an_applet_that_pushes_its_result_and_exits_stops_the_process() {
        let mut cpu = request(false, 1, &[]);
        cpu.register_service_handle(9, "am:library-applet-self-accessor");

        cpu.applet_request(TLS, 9, Some(1)).unwrap(); // PushOutData
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "PushOutData");
        assert!(!cpu.halted, "the applet has not asked to exit yet");

        write_request(&mut cpu, 10, &[]);
        cpu.applet_request(TLS, 9, Some(10)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x18).unwrap(),
            0,
            "ExitProcessAndReturn"
        );
        assert!(cpu.halted, "the applet asked to exit and kept running");
    }

    #[test]
    fn the_controller_applet_is_told_what_it_may_offer() {
        const CONTROLLER: u64 = 0x0100_0000_0000_1003;
        let storages = super::applet_launch_storages(CONTROLLER, crate::cpu::acc::DEFAULT_USER_UID);
        let private = &storages[0];
        assert_eq!(
            u32::from_le_bytes(private[..4].try_into().unwrap()),
            private.len() as u32
        );
        assert_eq!(
            u32::from_le_bytes(private[4..8].try_into().unwrap()) as usize,
            storages[1].len()
        );
        assert_eq!(super::applet_interface_version(CONTROLLER), 8);

        let styles = u32::from_le_bytes(private[0x0C..0x10].try_into().unwrap());
        assert_ne!(
            styles & crate::cpu::hid_shmem::STYLE_HANDHELD,
            0,
            "handheld is not on offer"
        );
        for pad in crate::cpu::NPAD_PRESENTATIONS {
            assert_ne!(
                styles & pad.style,
                0,
                "style {:#x} is not on offer",
                pad.style
            );
        }
    }

    #[test]
    fn a_library_applet_ends_the_moment_it_is_started() {
        let (mut cpu, accessor) = library_applet(APPLET_WEB);

        write_request(&mut cpu, 0, &[]);
        cpu.applet_request(TLS, accessor, Some(0)).unwrap();
        let event = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
        assert_eq!(cpu.event_name(event), Some("am:library-applet-state"));
        assert_eq!(
            cpu.event_signaled(event),
            Some(false),
            "nothing has started it yet"
        );

        write_request(&mut cpu, 10, &[]); // Start
        cpu.applet_request(TLS, accessor, Some(10)).unwrap();
        assert_eq!(cpu.event_signaled(event), Some(true));

        write_request(&mut cpu, 1, &[]); // IsCompleted
        cpu.applet_request(TLS, accessor, Some(1)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap() & 0xff, 1);

        write_request(&mut cpu, 30, &[]);
        cpu.applet_request(TLS, accessor, Some(30)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x18).unwrap(),
            128 | (22 << 9),
            "cancelled"
        );
    }

    #[test]
    fn the_applet_state_event_is_signalled_when_it_is_asked_for_after_the_start() {
        let (mut cpu, accessor) = library_applet(APPLET_WEB);
        write_request(&mut cpu, 10, &[]);
        cpu.applet_request(TLS, accessor, Some(10)).unwrap();

        write_request(&mut cpu, 0, &[]);
        cpu.applet_request(TLS, accessor, Some(0)).unwrap();
        let event = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
        assert_eq!(cpu.event_signaled(event), Some(true));
    }

    #[test]
    fn an_applet_that_never_ran_has_no_output_to_pop() {
        let (mut cpu, accessor) = library_applet(APPLET_WEB);
        write_request(&mut cpu, 10, &[]);
        cpu.applet_request(TLS, accessor, Some(10)).unwrap();

        write_request(&mut cpu, 101, &[]); // PopOutData
        cpu.applet_request(TLS, accessor, Some(101)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x18).unwrap(),
            128 | (3 << 9),
            "no data"
        );
        assert_eq!(cpu.mem.read_u32(TLS + 0x0c).unwrap(), 0, "and no storage");
    }

    #[test]
    fn created_storage_is_as_long_as_the_caller_asked_for() {
        let mut cpu = request(false, 10, &0x1000u64.to_le_bytes());
        cpu.register_service_handle(9, "am:library-applet-creator");
        cpu.applet_request(TLS, 9, Some(10)).unwrap();
        let storage = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
        assert_eq!(cpu.service_name(storage), Some("am:storage"));
        assert_eq!(cpu.am_storages[&Cpu::object_key(storage, 0)].len(), 0x1000);

        write_request(&mut cpu, 10, &u64::MAX.to_le_bytes());
        cpu.applet_request(TLS, 9, Some(10)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x18).unwrap(),
            1 | (104 << 9),
            "out of memory"
        );
    }

    #[test]
    fn the_ex_form_of_a_creation_is_the_same_creation() {
        const CALLER_THREAD: u64 = 0x2a;
        const FOREGROUND: u32 = 1;

        let mut payload = [0u8; 16];
        payload[..4].copy_from_slice(&APPLET_WEB.to_le_bytes());
        payload[4..8].copy_from_slice(&FOREGROUND.to_le_bytes());
        payload[8..].copy_from_slice(&CALLER_THREAD.to_le_bytes());
        let mut cpu = request(false, 3, &payload);
        cpu.register_service_handle(9, "am:library-applet-creator");
        cpu.applet_request(TLS, 9, Some(3)).unwrap();

        let accessor = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
        assert_ne!(accessor, 0, "CreateLibraryAppletEx moved no object back");
        assert_eq!(
            cpu.service_name(accessor),
            Some("am:library-applet-accessor")
        );
        let applet = &cpu.am_applets[&Cpu::object_key(accessor, 0)];
        assert_eq!(applet.id, APPLET_WEB);
        assert_eq!(applet.mode, FOREGROUND);
    }
}
