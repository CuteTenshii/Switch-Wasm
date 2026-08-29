//! `am`: the applet framework — `appletOE`/`appletAE` and the proxies,
//! functions and channels they hand out, plus the library applets a title can
//! launch.
//!
//! This is the largest service on the console and the one a title talks to
//! most: focus, operation mode, the applet message queue, save-data quotas and
//! every `ILibraryAppletAccessor` go through here. Nothing here *runs* a
//! library applet — see [`LibraryApplet`] — but a caller that launches one
//! still has to see it start, finish and hand back its result.

use super::acc::ACCOUNT_UID;
use super::Cpu;
use crate::Result;

/// The uid of the console's one user account.
///
/// Any 128-bit value does as long as it is **not zero**: zero is what
/// `AccountUid` means by "no user", and a title handed it back from
/// `GetLastOpenedUser` concludes nobody is signed in. Spelling it in ASCII
/// makes it recognisable in a trace, and it is exactly the 16 bytes a uid is.
/// `am` 2, NoDataInChannel: the general channel has nothing queued.
const AM_NO_DATA_IN_CHANNEL: u32 = 128 | (2 << 9);

/// `am`'s `LaunchParameterKind::PreselectedUser`: the user the launcher had
/// already chosen when it started the application.
pub(super) const LAUNCH_PARAMETER_PRESELECTED_USER: u32 = 2;

/// The `PreselectedUser` launch parameter, as `nn::account` reads it: a magic,
/// a version, then the uid, in a block of a fixed 0x88 bytes.
///
/// The HOME menu picks the user before it starts a title and leaves the choice
/// here; `nn::account::Initialize` pops it and caches the uid, and
/// `nn::account::OpenPreselectedUser` hands that cached uid back. There is no
/// menu here, but there is exactly one user ([`ACCOUNT_UID`]) and it is the
/// one every other `acc` answer names, so it is also the one that was
/// "selected".
///
/// `nn::account::detail::TryPopPreselectedUser` reads the block strictly: a
/// storage shorter than 0x88 bytes is an assertion, and a magic or version it
/// does not recognise means no preselected user at all — which it reports as a
/// zero uid, and which `OpenPreselectedUser` then asserts on.
pub(super) fn preselected_user_parameter() -> Vec<u8> {
    /// What the block says it is. `nn::account` compares the first word
    /// against this and ignores anything else.
    const MAGIC: u32 = 0xC794_97CA;
    /// The only layout revision `nn::account` accepts.
    const VERSION: u8 = 1;
    /// The size it insists on before it reads a byte.
    const LEN: usize = 0x88;
    /// Where the uid sits, past the magic, the version and its padding.
    const UID_OFFSET: usize = 0x8;
    let mut data = Vec::with_capacity(LEN);
    data.extend_from_slice(&MAGIC.to_le_bytes());
    data.push(VERSION);
    data.resize(UID_OFFSET, 0);
    data.extend_from_slice(&ACCOUNT_UID);
    data.resize(LEN, 0);
    data
}

/// A library applet created through `ILibraryAppletCreator`, from the
/// caller's side: what it asked for, and how far the applet got.
///
/// There is no process behind this. The applet exists only as the answers its
/// accessor gives — it is created, started, and finished in that one moment,
/// having produced nothing.
#[derive(Debug, Default)]
pub(crate) struct LibraryApplet {
    /// `AppletId`, the firmware applet the caller asked to launch.
    id: u32,
    /// `LibraryAppletMode`: whether it takes the whole screen or composes
    /// into the caller's own display.
    mode: u32,
    finished: bool,
    /// The three events an accessor hands out, by the slot constants below.
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

/// `GetAppletStateChangedEvent`, `GetPopOutDataEvent` and
/// `GetPopInteractiveOutDataEvent`, as slots in [`LibraryApplet::events`].
const STATE_CHANGED_EVENT: usize = 0;

const POP_OUT_DATA_EVENT: usize = 1;

const POP_INTERACTIVE_OUT_DATA_EVENT: usize = 2;

const LIBRARY_APPLET_EVENT_NAMES: [&str; 3] = [
    "am:library-applet-state",
    "am:library-applet-out-data",
    "am:library-applet-interactive-out-data",
];

/// The firmware applet an `AppletId` names — the inverse of
/// [`applet_id_for`], for saying out loud what a caller asked to launch.
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

/// Whether a title id is one of the firmware's library applets — the ones
/// launched *by* another applet rather than from the menu.
pub(crate) fn is_library_applet(program_id: u64) -> bool {
    matches!(applet_id_for(program_id), 0x0A..=0x1A)
}

/// The revision of its own launch interface an applet expects its caller to
/// speak, as `LibAppletCommonArguments::LaVersion`.
pub(crate) fn applet_interface_version(program_id: u64) -> u32 {
    match applet_id_for(program_id) {
        0x12 => 3, // miiEdit
        // swkbd numbers its interface with the firmware it shipped in rather
        // than from one upwards: 0x8000D is 6.0.0 and later, which is what an
        // 18.0.1 keyboard understands. Claiming version 1 describes a 1.0.0
        // caller, whose launch struct is a different shape and half the size.
        APPLET_SWKBD => 0x8_000D,
        // The controller applet picks the shape of its second storage by this
        // number, and 0x8 is what an 11.0.0-and-later one speaks: the 0x430
        // `ControllerSupportArg` with room for eight players, rather than the
        // 0x21C one with room for four.
        APPLET_CONTROLLER => 8,
        // myPage numbers its own the same way the keyboard does: 0x10000 is
        // 9.0.0 and later, whose argument is 0x10A8 bytes against the 0xB0 a
        // version 1 caller sends.
        APPLET_MY_PAGE => 0x1_0000,
        _ => 1,
    }
}

/// `AppletId_LibraryAppletSwkbd`.
const APPLET_SWKBD: u32 = 0x11;
/// `AppletId_LibraryAppletController`.
const APPLET_CONTROLLER: u32 = 0x0C;
/// `AppletId_LibraryAppletMyPage`.
const APPLET_MY_PAGE: u32 = 0x1A;

/// The launch storages a library applet's caller pushes after the common
/// arguments — the ones only its caller could fill in.
///
/// An applet pops these in order and gets no further than the first one that
/// is not there: `PopInData` answers `2128-0003` and `nnSdk` aborts on it.
/// That is one storage for most applets, and **two** for the keyboard and the
/// controller applet, which both take a private struct and then the argument
/// it describes.
///
/// Zeroes are the ordinary entry point for the applets whose struct starts
/// with a mode selector, so they stay the default. The two named here are the
/// ones whose contents say something: the keyboard's configuration is what
/// says how long the text may be and what the confirm button reads, and the
/// controller applet's says which controllers this console can offer.
pub(crate) fn applet_launch_storages(program_id: u64) -> Vec<Vec<u8>> {
    /// Enough of any other applet's struct for it to read the prefix it knows.
    const GENERIC_SIZE: usize = 0x100;
    match applet_id_for(program_id) {
        APPLET_SWKBD => vec![swkbd_config(), vec![0u8; SWKBD_WORK_BUFFER_SIZE]],
        APPLET_CONTROLLER => vec![controller_support_arg_private(), controller_support_arg()],
        APPLET_MY_PAGE => vec![my_page_arg()],
        _ => vec![vec![0u8; GENERIC_SIZE]],
    }
}

/// `nn::swkbd::KeyboardConfig`: `SwkbdConfigCommon` followed by
/// `SwkbdConfigNew`, the 6.0.0+ shape [`applet_interface_version`] claims.
fn swkbd_config() -> Vec<u8> {
    const CONFIG_SIZE: usize = 0x4C8;
    const OK_TEXT: usize = 0x004;
    const MAX_TEXT_LENGTH: usize = 0x3AC;
    const MIN_TEXT_LENGTH: usize = 0x3B0;
    let mut config = vec![0u8; CONFIG_SIZE];
    // SwkbdType_Normal, the full alphanumeric keyboard.
    config[0..4].copy_from_slice(&0u32.to_le_bytes());
    for (index, unit) in "OK".encode_utf16().enumerate() {
        let at = OK_TEXT + index * 2;
        config[at..at + 2].copy_from_slice(&unit.to_le_bytes());
    }
    config[MAX_TEXT_LENGTH..MAX_TEXT_LENGTH + 4].copy_from_slice(&32u32.to_le_bytes());
    config[MIN_TEXT_LENGTH..MIN_TEXT_LENGTH + 4].copy_from_slice(&0u32.to_le_bytes());
    config
}

/// The keyboard's third storage: the buffer its initial string and its user
/// dictionary would live in, at the offsets the configuration names. That
/// configuration names neither, so the applet reads nothing out of it — but
/// the storage still has to be there, and 0x1000 is the size a caller passes.
const SWKBD_WORK_BUFFER_SIZE: usize = 0x1000;

/// The friend-list applet's argument: which of its pages to open on, and the
/// user it is the page *of*. The fields past the uid belong to the types that
/// name another account — a friend request, an invitation — and are cleared
/// for the rest, which is every type this can be launched with here.
fn my_page_arg() -> Vec<u8> {
    /// The 9.0.0+ width, the one [`applet_interface_version`] claims.
    const ARG_SIZE: usize = 0x10A8;
    const USER_ID: usize = 0x8;
    let mut arg = vec![0u8; ARG_SIZE];
    // Type ShowFriendList, which is where the applet opens with no caller to
    // have asked for one of its other pages.
    arg[..4].copy_from_slice(&0u32.to_le_bytes());
    arg[USER_ID..USER_ID + 16].copy_from_slice(&super::acc::ACCOUNT_UID);
    arg
}

/// `nn::hid::system::ControllerSupportArgPrivate`: which of the controller
/// applet's screens to show, and the controller state its caller had when it
/// asked for one.
fn controller_support_arg_private() -> Vec<u8> {
    const SIZE: u32 = 0x14;
    let mut arg = Vec::with_capacity(SIZE as usize);
    arg.extend_from_slice(&SIZE.to_le_bytes());
    arg.extend_from_slice(&(CONTROLLER_SUPPORT_ARG_SIZE as u32).to_le_bytes());
    // Flag0 and Flag1, which sdknso leaves clear outside its *ForSystem
    // entry points, then ControllerSupportMode::ShowControllerSupport and
    // ControllerSupportCaller::Application.
    arg.extend_from_slice(&[0, 0, 0, 0]);
    // What `GetSupportedNpadStyleSet` answered the caller. Every style this
    // console's one pad can be published in, so the applet offers exactly the
    // controllers `hid` will then present — see `NPAD_PRESENTATIONS`.
    let styles = super::NPAD_PRESENTATIONS
        .iter()
        .fold(super::NPAD_HANDHELD.style, |set, pad| set | pad.style);
    arg.extend_from_slice(&styles.to_le_bytes());
    // `GetNpadJoyHoldType`, which is `hid`'s own default here: Vertical.
    arg.extend_from_slice(&0u32.to_le_bytes());
    arg
}

/// `nn::hid::ControllerSupportArg`, the 0x430-byte version 0x7-and-later
/// shape: eight identification colours and eight explain-text entries.
const CONTROLLER_SUPPORT_ARG_SIZE: usize = 0x430;

fn controller_support_arg() -> Vec<u8> {
    let mut arg = vec![0u8; CONTROLLER_SUPPORT_ARG_SIZE];
    // sdknso's own default for the struct, which it writes as a word: no
    // minimum player count, four of them at most, and take-over-connection
    // and left-justify on. The byte after it permits a dual Joy-Con.
    arg[..4].copy_from_slice(&0x0101_0400u32.to_le_bytes());
    arg[4] = 1;
    // enableSingleMode, which the default leaves clear. This console is in
    // handheld mode with one pad, and handheld is not an allowed answer to
    // the applet unless this says a single player will do.
    arg[5] = 1;
    arg
}

/// The `AppletId` a system applet reports for itself, from its title id.
///
/// The firmware's own applets are `0100000000001000`..`0100000000001013`, and
/// their ids run in the same order with two breaks in it: the menu and the
/// overlay applet are not library applets at all and have their own ids.
/// Anything else is an ordinary application.
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
        // photoViewer, set, offlineWeb, loginShare, wifiWebAuth.
        low @ 0x100D..=0x1011 => 0x15 + (low as u32 - 0x100D),
        // `starter` breaks the run: it is a SystemApplication rather than a
        // library applet, and it has a title id in the middle of the range
        // rather than past it. Counting through it put myPage one id too far
        // along — on `gift`, whose id is not a library applet's here at all,
        // so nothing seeded myPage's launch storages and its first
        // `PopInData` was refused.
        0x1012 => 0x04, // starter -> SystemApplication
        0x1013 => 0x1A, // myPage
        _ => 0x01,
    }
}

impl Cpu {
    /// The event an `ILibraryAppletAccessor` hands out for `slot`, allocated
    /// on first ask and kept: a caller that asks twice has to be given the
    /// same object, and one that waits on a handle it was handed a second
    /// copy of would wait on the wrong one.
    fn library_applet_event(&mut self, key: u64, slot: usize) -> u64 {
        if let Some(event) = self
            .am_applets
            .get(&key)
            .and_then(|applet| applet.events[slot])
        {
            return event;
        }
        // Not auto-clearing. What these report is an applet that has ended,
        // which does not un-end, and `libnx` waits on the state-changed one
        // in a loop — an auto-clearing event would be consumed by the first
        // wait and leave the second one hanging.
        let event = self.alloc_event(LIBRARY_APPLET_EVENT_NAMES[slot], false);
        self.am_applets.entry(key).or_default().events[slot] = Some(event);
        event
    }

    /// Fire one of an applet's events, if the caller has taken it. Allocating
    /// it here instead would make an event nothing is waiting on.
    fn signal_library_applet_event(&mut self, key: u64, slot: usize) {
        if let Some(event) = self
            .am_applets
            .get(&key)
            .and_then(|applet| applet.events[slot])
        {
            self.signal_event(event);
        }
    }

    /// Whether the applet behind an accessor has ended. One that was never
    /// created has not: an accessor with no applet is not an applet that ran.
    fn library_applet_finished(&self, key: u64) -> bool {
        self.am_applets
            .get(&key)
            .is_some_and(LibraryApplet::is_finished)
    }

    /// `IApplicationProxyService`/`IApplicationProxy`: the applet-lifecycle
    /// chain homebrew opens as `appletOE` (or `appletAE`, for a non-application
    /// applet). `appletMainLoop` polls `ICommonStateGetter` every frame — the
    /// event handle, then `ReceiveMessage`/`GetOperationMode`/
    /// `GetCurrentFocusState` — to decide whether to keep running; an earlier
    /// generic stub answered every one of those the same way regardless of
    /// which sub-interface actually made the call (and re-sent the initial
    /// "focus changed" message on every single poll), which made at least one
    /// real homebrew (JKSV) treat every frame as a fresh focus transition and
    /// give up after a handful of them.
    ///
    /// Only the commands listed below are implemented. Everything else goes to
    /// [`Cpu::unimplemented_command`] rather than a fabricated success — see there
    /// for why.
    pub(super) fn applet_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        const QUERY_POINTER_BUFFER_SIZE: u32 = 3;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "am:proxy-service");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                Some(QUERY_POINTER_BUFFER_SIZE) => {
                    self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, "am:control", cmd_id),
            };
        }
        // Which `am` sub-interface this request is actually for. A caller that
        // converted the session to a domain (`libnx`) addresses each one by
        // object id on the one `appletOE` handle; a caller that did not
        // (`nnSdk`) got a separate session handle per interface out of
        // [`Cpu::reply_with_interface`], and the name is recorded against the
        // handle instead. Resolving only the domain case left every `nnSdk`
        // request answered as `am:unknown`.
        let object_id = self.ipc_domain_object_id(tls);
        let iface = if self.ipc_is_domain_request(tls) {
            self.domain_interface(handle, object_id)
                .unwrap_or("am:unknown")
                .to_string()
        } else {
            match self.service_name(handle) {
                // The root session before any ConvertToDomain *is*
                // IApplicationProxyService.
                Some("appletOE") | Some("appletAE") | None => "am:proxy-service".to_string(),
                Some(name) => name.to_string(),
            }
        };
        match iface.as_str() {
            // The root session, which is `IApplicationProxyService` on
            // `appletOE` and `IAllSystemAppletProxiesService` on `appletAE`.
            // Which proxy a process opens is how it declares what kind of
            // applet it is: an application opens cmd 0, a library applet
            // (`miiEdit`, `swkbd`, `playerSelect` — every one of the system's
            // own applets) opens cmd 201.
            "am:proxy-service" => match cmd_id {
                // IApplicationProxyService::OpenApplicationProxy.
                Some(0) => {
                    self.set_applet_is_application(true);
                    self.reply_with_interface(tls, handle, "am:application-proxy")?;
                    Ok(())
                }
                // IAllSystemAppletProxiesService::OpenLibraryAppletProxy, and
                // the pre-3.0.0 `OpenLibraryAppletProxyOld` that differs only
                // in not taking the applet attribute buffer.
                Some(200) | Some(201) => {
                    self.set_applet_is_application(false);
                    self.reply_with_interface(tls, handle, "am:library-applet-proxy")?;
                    Ok(())
                }
                // IAllSystemAppletProxiesService::OpenSystemAppletProxy, and
                // the `Ex` form at 110 that differs only in taking an applet
                // attribute. This is what the *Home Menu* opens. qlaunch is
                // neither an application nor a library applet — it is the one
                // process that outlives every title and launches the rest —
                // and it aborts on the spot if this is refused.
                Some(100) | Some(110) => {
                    self.set_applet_is_application(false);
                    self.reply_with_interface(tls, handle, "am:system-applet-proxy")?;
                    Ok(())
                }
                // OpenSystemApplicationProxy. A system application is still an
                // application — it gets the same `IApplicationProxy` and the
                // same focus message — it just ships with the firmware rather
                // than being installed. `starter`, the applet that runs the
                // first-boot sequence, opens this and nothing else: refused,
                // it aborted with `nnSdk`'s unknown-command-id straight into
                // `fatal:u`.
                Some(350) => {
                    self.set_applet_is_application(true);
                    self.reply_with_interface(tls, handle, "am:application-proxy")?;
                    Ok(())
                }
                // OpenOverlayAppletProxy: `overlayDisp`, which draws over
                // whatever is running. It has the same lifecycle and window
                // controls as a library applet does.
                Some(300) => {
                    self.set_applet_is_application(false);
                    self.reply_with_interface(tls, handle, "am:library-applet-proxy")?;
                    Ok(())
                }
                // GetSystemProcessCommonFunctions (19.0.0+) and
                // GetAppletAlternativeFunctions (20.0.0+). Unlike every
                // command above them these open no proxy, so they say nothing
                // about which kind of applet the caller is and must leave that
                // flag alone.
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
            // ISystemAppletProxy's Get* accessors. The first seven are the
            // same ones `ILibraryAppletProxy` hands out; where a library applet
            // has its self-accessor and common functions at 20/21, the system
            // applet has the two interfaces it drives the console with — the
            // Home Menu's own functions and the global power/sleep state — and
            // an IApplicationCreator at 22, which is how the Home Menu starts
            // a game.
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
                    // GetAppletCommonFunctions, added at 23 here in 10.0.0 —
                    // the same interface a library applet fetches at 21.
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
            // ILibraryAppletProxy's Get* accessors. The first five are the
            // same interfaces `IApplicationProxy` hands out — a library applet
            // has the same lifecycle, window and audio controls as an
            // application does — and the rest are its own.
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
                    // A library applet fetches these two as well: the same
                    // pair `ISystemAppletProxy` exposes at 20/21, at the ids
                    // left over once the self-accessor and common functions
                    // have taken 20 and 21 here.
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
            // IApplicationProxy's Get* accessors, each handing back one of the
            // sub-interfaces below.
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
            // ICommonStateGetter: the state `appletMainLoop` polls every frame.
            "am:common-state-getter" => match cmd_id {
                // GetSettingsPlatformRegion -> SetSysPlatformRegion. 1 is
                // Global; 2 is the Chinese console, which has a different set
                // of services and stores behind it.
                Some(300) => self.write_ipc_response(tls, 0, &[], &1u8.to_le_bytes(), &[]),
                // GetHomeButtonReaderLockAccessor /
                // GetReaderLockAccessorEx(u32 button_type) -> ILockAccessor:
                // the read side of the HOME and capture button locks, the
                // counterpart to `IHomeMenuFunctions` 30/31. The Home Menu
                // takes one per button before it will run a transition.
                Some(30) | Some(31) => {
                    self.reply_with_interface(tls, handle, "am:lock-accessor")?;
                    Ok(())
                }
                // GetEventHandle: the copy handle the guest waits on before
                // polling ReceiveMessage.
                //
                // It starts **signalled** exactly when a message is waiting,
                // which at startup means once: AM queues one FocusStateChanged
                // and ReceiveMessage below hands it out. The event is
                // auto-clearing, so the first successful wait consumes it and
                // every later poll times out — which is the whole protocol.
                //
                // An applet does not draw until it has been told it is in
                // focus, and it asks by *polling this event with a zero
                // timeout* rather than by calling ReceiveMessage. Leaving the
                // event dark meant the message was there and nothing ever came
                // to collect it: the Mii editor sat in `appletMainLoop`
                // polling an event that would never fire, one dequeued buffer
                // in hand and not a single draw behind it.
                //
                // (It used to be left dark on purpose, because firing it sent
                // `nnSdk`'s system worker into a handler that did not exist.
                // That was `WaitSynchronization` reporting index 1 for a
                // one-handle wait, and it is fixed where it belongs.)
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
                // ReceiveMessage: real AM enqueues one FocusStateChanged at
                // startup and then reports "no message" until the state
                // actually changes; answering every poll with a fresh message
                // is what made JKSV think focus kept changing.
                Some(1) => {
                    const NO_MESSAGES: u32 = 128 | (3 << 9); // am, "no message"
                    match self.next_applet_message() {
                        Some(message) => {
                            self.write_ipc_response(tls, 0, &[], &message.to_le_bytes(), &[])
                        }
                        None => self.write_ipc_response(tls, NO_MESSAGES, &[], &[], &[]),
                    }
                }
                // GetOperationMode -> AppletOperationMode. **Handheld is 0**
                // and Console (docked) is 1; this answered 1 while its comment
                // said Handheld, so NX-Fetch printed "Docked" beside a 720p
                // handheld framebuffer, and a title that picks its resolution
                // by operation mode was being told to render at 1080p. Both
                // now come from the one switch, so they cannot disagree again:
                // see [`super::OperationMode`].
                Some(5) => {
                    let mode = self.operation_mode() as u32;
                    self.write_ipc_response(tls, 0, &[], &mode.to_le_bytes(), &[])
                }
                // GetPerformanceMode -> ApmPerformanceMode: Normal handheld,
                // Boost docked.
                Some(6) => {
                    let mode = self.operation_mode().performance_mode();
                    self.write_ipc_response(tls, 0, &[], &mode.to_le_bytes(), &[])
                }
                Some(9) => self.write_ipc_response(tls, 0, &[], &1u32.to_le_bytes(), &[]), // GetCurrentFocusState: InFocus
                // GetBootMode: Normal.
                Some(8) => self.write_ipc_response(tls, 0, &[], &0u8.to_le_bytes(), &[]),
                // GetAcquiredSleepLockEvent / GetDefaultDisplayResolutionChangeEvent:
                // handles the caller waits on. Nothing here ever sleeps or
                // changes resolution, so they are handed out and never
                // signalled — see the note on GetEventHandle above for why a
                // wait on them still returns.
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
                // GetDefaultDisplayResolutionChangeEvent: fired when the
                // console is docked or undocked, which is the only thing that
                // changes the resolution. It used to be handed out dark on the
                // grounds that the resolution never changed — true when there
                // was one — and one object per caller, so nothing could have
                // signalled the one being waited on anyway.
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
                // GetDefaultDisplayResolution: the display's, which is the
                // dock's — 1280x720 handheld, 1920x1080 docked. Hard-coding
                // 720p here is what put "1280x720 @ 60Hz [Docked]" on
                // NX-Fetch's screen: the mode and the resolution beside it
                // came from two different places.
                Some(60) => {
                    let (width, height) = self.operation_mode().display_size();
                    let mut raw = Vec::with_capacity(8);
                    raw.extend_from_slice(&width.to_le_bytes());
                    raw.extend_from_slice(&height.to_le_bytes());
                    self.write_ipc_response(tls, 0, &[], &raw, &[])
                }
                // RequestToAcquireSleepLock: nothing else here contends the
                // lock, so it is granted at once — and the event that says so
                // fires with it. Handing that event out dark left an applet
                // waiting for permission to keep the console awake that was
                // never going to come.
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
                // SetRequestExitToLibraryAppletAtExecuteNextProgramEnabled:
                // a title that is about to hand the console to another
                // program (`nn::oe::ExecuteProgram`) asks AM to send it an
                // Exit message when that happens, so it shuts down instead of
                // sitting behind the program it launched.
                //
                // It is a latch with no argument and no reply — asking is
                // setting it — and nothing is recorded because nothing here
                // ever executes a next program, so the message it arms could
                // never be sent. Tomodachi Life asks for it during startup,
                // and refusing it aborted `nnSdk` before the title reached a
                // service.
                Some(900) => {
                    self.warn_stub(&iface, cmd_id, "the exit-request latch is not recorded");
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "am:application-functions" => match cmd_id {
                // PopLaunchParameter(u32 kind) -> IStorage: what the launcher
                // left for this program. It is a *pop* — `am` hands the
                // storage over once and forgets it, so a second ask finds
                // nothing, and `nn::account` relies on that to only ever cache
                // the preselected user once.
                //
                // Only the kinds [`Cpu::seed_launch_parameters`] filled in are
                // here. Everything else fails the way it does on hardware for
                // a program nobody left anything for: an earlier stub's
                // success-with-an-unrelated-object-id left callers treating
                // that id as a launch-parameter storage that was never
                // registered as one.
                Some(1) => {
                    /// `am` description 2: no launch parameter of that kind.
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
                // EnsureSaveData -> the save data size it ensured.
                Some(20) => {
                    self.warn_stub(&iface, cmd_id, "0 bytes ensured; no save was created");
                    self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[])
                }
                // ExtendSaveData(u8 SaveDataType, u128 userId, s64 size,
                // s64 journal) -> an s64 the caller discards. Same argument
                // shape as GetSaveDataSize below with the two sizes appended,
                // so they sit at 0x18 and 0x20.
                //
                // There is no NAND quota here to grant the extension out of,
                // so it is granted as asked and *remembered*: a title that
                // reads its size back must be told what it just set rather
                // than the NACP figure it has already moved past. Refusing it
                // is where Minecraft stopped — `nn::fs::ExtendSaveData` aborts
                // on any error, and an unknown command id is one.
                Some(25) => {
                    let data = self.ipc_request_data(tls);
                    self.save_data_quota.size = self.mem.read_u64(data.wrapping_add(0x18))? as i64;
                    self.save_data_quota.journal_size =
                        self.mem.read_u64(data.wrapping_add(0x20))? as i64;
                    self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[])
                }
                // GetSaveDataSize(u8 SaveDataType, u128 userId) -> two s64s,
                // the save's size and its journal's. The request confirms that
                // shape: its `CmifDomainInHeader` declares data_size=0x28, so
                // 0x18 bytes follow the `CmifInHeader` — a type padded to
                // eight, then the uid.
                //
                // Neither input changes the answer. There is one user here and
                // one save behind it, and the emulated NAND has no quota to
                // divide between save data types — so what a title is told is
                // simply what it was allotted, which is its own NACP's figure
                // once anything has read it (see `Cpu::set_save_data_sizes`).
                //
                // Refusing this is where Tomodachi Life stopped once `am` 210
                // let it through: `nnSdk` answers an unknown command id with an
                // svcBreak, 452M steps in, with the title's RomFS mounted and
                // its first assets already decompressing.
                Some(26) => {
                    let quota = self.save_data_quota;
                    self.write_save_data_pair(tls, quota.size, quota.journal_size)
                }
                // GetSaveDataSizeMax / GetDeviceSaveDataSizeMax: the same two
                // s64s, for how far each save may be *extended* rather than
                // what it was created at. Both take no input.
                //
                // A NACP commonly declares a size and no ceiling, and that 0
                // is reported as it stands: it is the title's own statement
                // that it never grows this save. Inventing headroom would
                // answer a question the title did not ask, and the failure it
                // causes — a title extending a save the system never agreed to
                // — surfaces nowhere near here.
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
                // GetCacheStorageMax -> an s32 and an s64: how many cache
                // storages the title may address, and the ceiling on one
                // storage's data and journal together.
                //
                // The two are laid out the way `sf` marshals a pair of
                // outputs, each aligned to its own width — so the s64 is at +8
                // and +4 is padding, not the second half of a packed struct.
                Some(29) => {
                    let quota = self.save_data_quota;
                    let mut out = Vec::with_capacity(16);
                    out.extend_from_slice(&quota.cache_storage_index_max.to_le_bytes());
                    out.extend_from_slice(&[0u8; 4]);
                    out.extend_from_slice(&quota.cache_storage_size_max.to_le_bytes());
                    self.write_ipc_response(tls, 0, &[], &out, &[])
                }
                // CreateCacheStorage(u16 index, s64 size, s64 journal) -> the
                // storage it was put on, and how much room that took.
                //
                // Cache storage is scratch: a title asks for some, the system
                // may delete it again between runs, and a title that finds it
                // gone rebuilds it. Here there is one storage and it has no
                // quota, so the request is granted as asked and nothing is
                // reserved — `fsp-srv` will create the save the first time the
                // title mounts it, exactly as it does for any other.
                //
                // Unlike its neighbours this command's *output* shape is not
                // documented on switchbrew; it is `libnx`'s (a u32 target
                // followed by a u64 required size). The reply is a full 0x10
                // either way, which is the shape that survives being wrong —
                // a reply may be longer than a caller expects, never shorter.
                Some(27) => {
                    let mut out = Vec::with_capacity(16);
                    // The one storage this console has.
                    out.extend_from_slice(&1u32.to_le_bytes());
                    out.extend_from_slice(&[0u8; 4]);
                    out.extend_from_slice(&0u64.to_le_bytes());
                    self.write_ipc_response(tls, 0, &[], &out, &[])
                }
                // GetDesiredLanguage -> an `nn::settings::LanguageCode`, which
                // is the null-padded BCP-47 tag as eight raw bytes.
                Some(21) => {
                    let mut code = [0u8; 8];
                    code[..5].copy_from_slice(b"en-US");
                    self.write_ipc_response(tls, 0, &[], &code, &[])
                }
                // GetDisplayVersion -> a 16-byte version string.
                Some(23) => {
                    let mut version = [0u8; 16];
                    version[..5].copy_from_slice(b"1.0.0");
                    self.write_ipc_response(tls, 0, &[], &version, &[])
                }
                // BeginBlockingHomeButtonShortAndLongPressed(s64 timeout) and
                // its End, then the same pair for the plain home button.
                //
                // A title asks for this before doing something it must not be
                // interrupted in the middle of — JKSV blocks the home button
                // while it writes a save. There is no home button here and no
                // home menu to return to, so nothing *can* interrupt it: the
                // request is granted because it is already true.
                Some(30) | Some(31) | Some(32) | Some(33) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // NotifyRunning -> whether the notification was the first one.
                Some(40) => self.write_ipc_response(tls, 0, &[], &1u8.to_le_bytes(), &[]),
                // GetPseudoDeviceId -> a 16-byte per-console, per-title id.
                // Zero is a legitimate value and nothing here derives anything
                // from it, but it must be the right *size* — a caller copies
                // 16 bytes out of the reply either way.
                Some(50) => {
                    self.warn_stub(&iface, cmd_id, "an all-zero device id");
                    self.write_ipc_response(tls, 0, &[], &[0u8; 16], &[])
                }
                // GetGpuErrorDetectedSystemEvent: the event `nn::oe::
                // SetupGpuErrorHandler` registers with the SDK's system
                // worker, so that a GPU fault wakes a handler instead of
                // hanging the title. It is the first thing a retail `nnSdk`
                // asks `am` for that it cannot start without — answering it
                // with anything but a copy handle aborts `nn::oe::Initialize`.
                // Nothing here ever faults the GPU, so the event is handed out
                // and never signalled.
                Some(130) => {
                    self.warn_stub(&iface, cmd_id, "an event nothing here ever signals");
                    let h = self.alloc_event("am:gpu-error", true);
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                // SetTerminateResult / InitializeGamePlayRecording /
                // SetGamePlayRecordingState / SetDelayTimeToAbortOnGpuError:
                // nothing to record, nothing to fault, nothing to report back.
                Some(22) | Some(66) | Some(67) | Some(131) => {
                    self.warn_stub(&iface, cmd_id, "accepted and not recorded");
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // Command 210, added in 20.0.0 and still unnamed on
                // switchbrew: no input, one out **event**. It sits between
                // GetLastApplicationExitReason (200) and SetAudioOutputPolicy
                // (220), beside the exit-request flow the same firmware added
                // at 310 — so what fires it is the system asking a running
                // title to quit. Nothing here can ask, so it is handed out and
                // never signalled, which is a wait that genuinely never
                // finishes rather than one answered wrongly.
                //
                // The name is unknown; the shape is not, and the shape is what
                // a caller acts on. Tomodachi Life asks for this immediately
                // after its account setup, and `nnSdk` answers an unknown
                // command id with an svcBreak — so refusing it ended the boot
                // there.
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
            // ISelfController: the applet's own lifecycle knobs.
            "am:self-controller" => match cmd_id {
                // Exit / LockExit / UnlockExit / EnterFatalSection /
                // LeaveFatalSection / SetScreenShotPermission /
                // Set{Operation,Performance}ModeChangedNotification /
                // SetFocusHandlingMode / SetRestartMessageEnabled /
                // SetScreenShotAppletIdentityInfo /
                // SetOutOfFocusSuspendingEnabled /
                // SetScreenShotImageOrientation / SetHandlesRequestToDisplay /
                // SetIdleTimeDetectionExtension / SetAutoSleepDisabled /
                // SetAlbumImageTakenNotificationEnabled /
                // SetApplicationAlbumUserData / SetRecordVolumeMuted.
                //
                // Every one of these is a setter or a notifier whose whole
                // reply is a Result. There is no suspend, screenshot, album or
                // exit-lock behaviour behind them to change, so accepting the
                // setting really is the complete implementation — unlike the
                // commands below it, a bare success here is the truth.
                Some(0..=4) | Some(10..=16) | Some(19) | Some(51) | Some(62) | Some(68)
                | Some(100) | Some(110) | Some(130) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // SetHandlesRequestToDisplay(bool): the applet is taking
                // responsibility for when it appears. AM answers by queueing
                // `RequestToDisplay`, and the applet draws its first frame
                // only once it has read that message and called
                // `ApproveToDisplay` (51, accepted above -- it used to reach
                // `unimplemented_command`, which `nnSdk` aborts on).
                //
                // Without the message the Home Menu waits for permission that
                // never comes: it finishes its layer, preallocates both
                // swapchain buffers, and then runs its frame loop for thirty
                // seconds of console time without ever dequeuing one.
                Some(50) => {
                    let data = self.ipc_request_data(tls);
                    if self.mem.read_u8(data).unwrap_or(0) != 0 {
                        self.queue_applet_message(super::AppletMessage::RequestToDisplay);
                    }
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetLibraryAppletLaunchableEvent /
                // GetAccumulatedSuspendedTickChangedEvent: copy handles the
                // caller stores and later waits on. `libnx`'s `appletInitialize`
                // asks for the second one on 6.0.0+ and keeps whatever handle
                // came back, so answering with success and *no* handle left it
                // holding 0 — the same shape of bug that had `nnSdk`'s system
                // worker waiting on handle 0.
                Some(9) | Some(91) => {
                    self.warn_stub(&iface, cmd_id, "an event nothing here ever signals");
                    let h = self.alloc_event("am:self-controller", true);
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                // GetAccumulatedSuspendedTickValue: nothing has ever been
                // suspended.
                Some(90) => self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[]),
                // IsSystemBufferSharingEnabled: whether this applet draws
                // into a buffer the system shares between applets rather than
                // a layer of its own.
                //
                // It does not, and saying so is what sends it down the
                // CreateManagedDisplayLayer path below — the one `vi` here
                // actually models. Reporting it enabled would commit the
                // caller to asking for a shared buffer handle that nothing
                // can produce.
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
                // CreateManagedDisplayLayer -> the layer id the caller then
                // passes to `vi`'s OpenLayer. The display stub only models one
                // layer and calls it 1 (see [`Cpu::vi_native_window`]), so this
                // has to agree with it.
                Some(40) => self.write_ipc_response(tls, 0, &[], &1u64.to_le_bytes(), &[]),
                // CreateManagedDisplaySeparableLayer -> the same layer plus a
                // recording layer, which nothing here records from.
                Some(44) => {
                    // The recording layer is reported as 0, not as the layer
                    // itself. `vi` here models one layer and calls it 1, and
                    // handing the same id back twice invites the caller to
                    // open it a second time and rebind the binder underneath
                    // its own swapchain.
                    let mut raw = Vec::with_capacity(16);
                    raw.extend_from_slice(&1u64.to_le_bytes());
                    raw.extend_from_slice(&0u64.to_le_bytes());
                    self.write_ipc_response(tls, 0, &[], &raw, &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IWindowController: foreground rights and the applet resource id
            // every other service tags this process's requests with.
            // IDisplayController: the applet's view of the *capture* buffers —
            // the screenshot of whatever was on screen before it, which an
            // applet composites behind itself so the menu or the game shows
            // through.
            //
            // There is nothing behind any applet here, so every acquire
            // reports that no capture was written and names no layer. What it
            // must not do is refuse: `miiEdit` asks for command 26 and
            // `nnSdk` answers an unknown command id with an svcBreak, which
            // killed it before it drew anything.
            "am:display-controller" => match cmd_id {
                // Acquire{LastApplication,LastForeground,CallerApplet}
                // CaptureSharedBuffer -> bool written, s32 shared-buffer slot.
                //
                // The applet is asking for the screen of whatever was on
                // display before it — the Album draws its gallery over a
                // frozen shot of the Home Menu. Nothing was: an applet booted
                // here is booted alone, so the honest capture is a black one.
                //
                // Answering "nothing written, slot -1" is not the way to say
                // that. `nnSdk` treats it as *not ready yet* and asks again:
                // the Album applet spent every frame of a 300M-instruction run
                // in that retry loop and never got as far as a draw. So the
                // reply names a real slot, and the slot named is the first one
                // past the two `AcquireSharedFrameBuffer` hands out, which
                // nothing renders into and nothing has written — its pages are
                // soft-mapped and read as the zeros this claims they are.
                Some(22) | Some(24) | Some(26) => {
                    let mut raw = Vec::with_capacity(8);
                    raw.extend_from_slice(&[1u8, 0, 0, 0]); // was_written = true
                    raw.extend_from_slice(
                        &(super::SHARED_BUFFER_USABLE_SLOTS as i32).to_le_bytes(),
                    );
                    self.write_ipc_response(tls, 0, &[], &raw, &[])
                }
                // The matching releases, ClearCaptureBuffer,
                // ClearAppletTransitionBuffer, and the two screenshot
                // commands: nothing to release, clear or capture.
                Some(8) | Some(20) | Some(21) | Some(23) | Some(25) | Some(27) | Some(28) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // Update{LastForeground,CallerApplet}CaptureImage: the same,
                // and they answer with a bare Result.
                Some(1) | Some(4) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // Get{LastForeground,LastApplication,CallerApplet}
                // CaptureImageEx -> bool written, with the image itself going
                // into a map-alias out buffer of 0x384000 bytes: 1280x720
                // RGBA8888. The black capture the three `Acquire`s above hand
                // out as a slot, handed over as pixels instead — so it is
                // cleared here rather than left as whatever the caller's
                // buffer held, which is what it would then have drawn.
                Some(5) | Some(6) | Some(7) => {
                    if let Some((addr, size)) = self.ipc_output_buffer(tls, 0) {
                        let page = crate::mem::PAGE_SIZE as u32;
                        let end = addr.saturating_add(size);
                        let mut at = addr;
                        while at < end {
                            // `fill_le` looks the page up once for a run that
                            // stays inside one, and writes a byte at a time
                            // for a run that does not.
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
                // There is one process here, so it gets one id; the `vi` and
                // `hid` stubs ignore which id a request carries.
                Some(1) | Some(2) => self.write_ipc_response(tls, 0, &[], &1u64.to_le_bytes(), &[]),
                // AcquireForegroundRights / ReleaseForegroundRights /
                // RejectToChangeIntoBackground: nothing else is competing for
                // the foreground.
                Some(10) | Some(11) | Some(12) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IAudioController: the applet's volume relative to the system's.
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
            // IAppletCommonFunctions: knobs an applet sets on itself that
            // are not specific to being an application or a library applet.
            "am:applet-common-functions" => match cmd_id {
                // SetCpuBoostRequestPriority: where this applet sits in the
                // queue when several ask the system to boost the CPU. There
                // is one process here and no governor to ask.
                Some(70) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // 20.0.0+, and unnamed: switchbrew's table stops at 341. Eden's
                // `am/service/applet_common_functions.cpp` reads it as one u16
                // out and answers 0, which is the only account of its shape
                // there is — and a scalar is the one kind of answer that cannot
                // leave the caller holding an object that was never handed over.
                Some(350) => self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ISystemProcessCommonFunctions: one command, which hands back an
            // IApplicationObserver — the interface a system process watches a
            // running application through. The observer's own commands (1, 2,
            // 10, 20, 30, 40) have no published names or signatures, so they
            // stop at [`Cpu::unimplemented_command`] and name themselves there
            // rather than being guessed at.
            "am:system-process-common-functions" => match cmd_id {
                Some(1) => {
                    self.reply_with_interface(tls, handle, "am:application-observer")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IHomeMenuFunctions: what only the Home Menu can do. qlaunch
            // opens this before it draws anything, so every one of these runs
            // during boot rather than on a user action.
            "am:home-menu-functions" => match cmd_id {
                // RequestToGetForeground / LockForeground / UnlockForeground:
                // who owns the screen. There is one applet here and it always
                // owns it.
                Some(10) | Some(11) | Some(12) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // PopFromGeneralChannel -> IStorage. The channel is what
                // another process pushes a message onto — an nfc tag scan, a
                // Joy-Con pairing. Nothing here pushes one, and an empty
                // channel is reported as `am` 2, NoDataInChannel: the Home
                // Menu drains this until it gets that error, so a *refusal* to
                // answer at all is what stops it (`nnSdk` aborts on an unknown
                // command id rather than carrying on).
                Some(20) => self.write_ipc_response(tls, AM_NO_DATA_IN_CHANNEL, &[], &[], &[]),
                // GetPopFromGeneralChannelEvent: the event that fires when a
                // message lands on that channel. Handed out and never
                // signalled, because nothing here ever pushes one — but the
                // same event each time, since the menu keeps a waiter on it.
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
                // GetHomeButtonWriterLockAccessor / GetWriterLockAccessorEx:
                // the lock the menu takes to suppress the HOME button while a
                // transition is running.
                Some(30) | Some(31) => {
                    self.reply_with_interface(tls, handle, "am:lock-accessor")?;
                    Ok(())
                }
                // IsSleepEnabled / IsRebootEnabled -> bool. Both are what a
                // retail console with no parental or demo restriction reports;
                // neither actually happens here, but the menu greys the
                // entries out when they are false.
                Some(40) | Some(41) => self.write_ipc_response(tls, 0, &[], &[1u8], &[]),
                // IsForceTerminateApplicationDisabledForDebug -> bool.
                Some(110) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                // SetLastApplicationExitReason: recorded for the next crash
                // report, which nothing here writes.
                Some(1000) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ILockAccessor: one of those HOME-button locks. Only the Home
            // Menu holds one, and nothing else here contends for it.
            "am:lock-accessor" => match cmd_id {
                // TryLock(bool return_handle) -> (bool locked, event). Nothing
                // else here holds the lock, so it is always taken; the handle
                // only comes back when the caller asked for it.
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
            // IGlobalStateController: the console-wide power state, which the
            // Home Menu owns rather than shares. The sequences at 0-4 (sleep,
            // shutdown, reboot) are deliberately not implemented: they are
            // user actions, and a console that answers "done" to a shutdown it
            // did not perform is worse than one that refuses.
            "am:global-state-controller" => match cmd_id {
                // IsAutoPowerDownRequested -> bool. The idle timer has not
                // fired, because there is no idle timer.
                Some(9) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                // LoadAndApplyIdlePolicySettings / NotifyCecSettingsChanged /
                // SetDefaultHomeButtonLongPressTime /
                // UpdateDefaultDisplayResolution: settings applied to hardware
                // that is not here.
                Some(10) | Some(11) | Some(12) | Some(13) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // ShouldSleepOnBoot -> bool. A console that was put to sleep
                // rather than shut down resumes straight back to sleep; this
                // one always boots awake.
                Some(14) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                // GetHdcpAuthenticationFailedEvent: fires when a dock refuses
                // to authenticate. There is no dock.
                Some(15) => {
                    let h = self.alloc_event("am:hdcp-failed", true);
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IProcessWindingController: how an applet is resumed after
            // "winding" — being paused so another applet can run in front of
            // it and then unwound back. Nothing here can wind anything: there
            // is one process and nowhere for it to go.
            "am:process-winding-controller" => match cmd_id {
                // GetLaunchReason -> AppletProcessLaunchReason { u8 flag, u8
                // pad[2], u8 unknown }. All-zero is "started normally", which
                // is the only way anything starts here — the nonzero flags
                // mean the process was resumed from a wind or restarted by the
                // menu.
                Some(0) => self.write_ipc_response(tls, 0, &[], &[0u8; 4], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ILibraryAppletSelfAccessor: what a library applet asks about
            // *itself*, and how it is handed its caller's arguments.
            "am:library-applet-self-accessor" => match cmd_id {
                // GetLibraryAppletInfo -> LibraryAppletInfo { AppletId,
                // LibraryAppletMode }.
                //
                // AllForeground (0) is the mode: this applet owns the screen.
                // There is no home menu behind it, nothing else drawing, and
                // no indirect-display path to hand its frames to — so of the
                // five modes it is the only one that is true here.
                Some(11) => {
                    let mut info = [0u8; 8];
                    info[..4].copy_from_slice(&applet_id_for(self.program_id()).to_le_bytes());
                    self.write_ipc_response(tls, 0, &[], &info, &[])
                }
                // ShouldSetGpuTimeSliceManually -> bool. An applet that owns
                // the screen outright does not have to divide the GPU with a
                // running application, so it has no time slice to set.
                //
                // Refusing it is what killed `swkbd`: it aborted into
                // `fatal:u` two million instructions in, before any of the
                // rendering the rest of this is about.
                Some(150) => self.write_ipc_response(tls, 0, &[], &0u8.to_le_bytes(), &[]),
                // GetMainAppletIdentityInfo / GetCallerAppletIdentityInfo ->
                // AppletIdentityInfo { AppletId, pad, u64 title_id }.
                //
                // Both are the home menu. A library applet is launched by
                // whatever is in the foreground, and the only thing that ever
                // launches one from a standing start is the menu — which is
                // also the applet that would be behind it on the stack.
                Some(12) | Some(14) => {
                    const QLAUNCH_TITLE_ID: u64 = 0x0100_0000_0000_1000;
                    let mut info = [0u8; 16];
                    info[..4].copy_from_slice(&3u32.to_le_bytes()); // SystemAppletMenu
                    info[8..].copy_from_slice(&QLAUNCH_TITLE_ID.to_le_bytes());
                    self.write_ipc_response(tls, 0, &[], &info, &[])
                }
                // GetDesirableKeyboardLayout -> nn::settings::KeyboardLayout,
                // the layout the applet's caller asked it to open with.
                // Hardware errors when no caller set one; there is no caller
                // here, so this answers with the layout that goes with the
                // language the console is set to — `set`'s SetLanguage_ENUS.
                Some(19) => {
                    const ENGLISH_US: u32 = 1;
                    self.write_ipc_response(tls, 0, &[], &ENGLISH_US.to_le_bytes(), &[])
                }
                // A setter the applet calls during init, carrying 16 bytes
                // of arguments and expecting nothing back but a Result.
                // Whatever it is configuring has no equivalent here.
                Some(160) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // PopInData -> IStorage: the arguments the applet was
                // launched with, which its caller pushed before starting it.
                //
                // There is no caller here — a library applet is being run
                // directly — so the storage every caller pushes first is
                // synthesized instead: `LibAppletCommonArguments`, the
                // 0x20-byte block carrying the API version the two sides
                // agreed on and the theme to draw in. What the applet pops
                // after that is its own launch struct, which only its caller
                // could know; there is no second storage to hand over.
                Some(0) => match self.am_in_data.pop_front() {
                    Some(data) => {
                        let key = self.reply_with_interface(tls, handle, "am:storage")?;
                        self.am_storages.insert(key, data);
                        Ok(())
                    }
                    None => {
                        /// `am` description 3: the applet asked for a storage
                        /// that was never pushed.
                        const NO_DATA: u32 = 128 | (3 << 9);
                        // Named once, because `nnSdk` aborts on this and the
                        // fatal it raises carries the code and nothing about
                        // where it came from — 2128-0003 is also what an
                        // empty `ReceiveMessage` answers, which is routine.
                        if self
                            .unimplemented_ipc
                            .insert(("am:pop-in-data".to_string(), None))
                        {
                            self.diagnostic(
                                "[am] PopInData: the applet has popped every storage seeded for \
                                 it and asked for another",
                            );
                        }
                        self.write_ipc_response(tls, NO_DATA, &[], &[], &[])
                    }
                },
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ILibraryAppletCreator: how one applet launches another —
            // a game asking for the keyboard, a homebrew handing the screen
            // to the browser.
            //
            // The applet itself is a separate process, and this emulator
            // hosts one. So the accessor handed back here drives an applet
            // that starts and immediately gives up: see
            // `am:library-applet-accessor` for why that is more useful than
            // refusing the creation outright.
            "am:library-applet-creator" => match cmd_id {
                // CreateLibraryApplet(u32 AppletId, u32 LibraryAppletMode)
                // -> ILibraryAppletAccessor.
                Some(0) => {
                    let at = self.ipc_request_data(tls);
                    let id = self.mem.read_u32(at)?;
                    let mode = self.mem.read_u32(at.wrapping_add(4))?;
                    self.diagnostic(&format!(
                        "[am] CreateLibraryApplet: {} (mode {mode}) — nothing here runs it, \
                         so it will report itself cancelled",
                        applet_name(id)
                    ));
                    let key =
                        self.reply_with_interface(tls, handle, "am:library-applet-accessor")?;
                    self.am_applets.insert(key, LibraryApplet::new(id, mode));
                    Ok(())
                }
                // TerminateAllLibraryApplets, and AreAnyLibraryAppletsLeft ->
                // bool. Nothing was ever left running to terminate.
                Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                Some(2) => self.write_ipc_response(tls, 0, &[], &0u8.to_le_bytes(), &[]),
                // CreateStorage(s64 size) -> IStorage: the buffer a caller
                // fills with an applet's launch arguments before pushing it.
                // The bytes are the storage's own, so it starts as `size`
                // zeroes and the caller writes through an IStorageAccessor.
                Some(10) => {
                    /// Past this, the size is not a launch argument any
                    /// caller actually sends, and allocating what it asks for
                    /// is how a bad size becomes an abort.
                    const MAX_STORAGE: u64 = 64 * 1024 * 1024;
                    /// `KERNELRESULT(OutOfMemory)`: what a console answers
                    /// when it cannot allocate the storage.
                    const OUT_OF_MEMORY: u32 = 1 | (104 << 9);
                    let size = self.mem.read_u64(self.ipc_request_data(tls))?;
                    if size > MAX_STORAGE {
                        return self.write_ipc_response(tls, OUT_OF_MEMORY, &[], &[], &[]);
                    }
                    let key = self.reply_with_interface(tls, handle, "am:storage")?;
                    self.am_storages.insert(key, vec![0u8; size as usize]);
                    Ok(())
                }
                // CreateTransferMemoryStorage and CreateHandleStorage take
                // the memory their bytes live in as a handle.
                // `svcCreateTransferMemory` here hands back one fixed handle
                // and records no address, so there is nothing to read the
                // contents from: such a storage would be zeroes claiming to
                // be the caller's data, which is worse than a refusal.
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ILibraryAppletAccessor: the object a caller drives a launched
            // applet through — start it, wait for it to finish, read back
            // what it produced.
            //
            // Nothing runs, so the applet finishes the moment it is started
            // and reports the one honest outcome available: cancelled, the
            // result a console gives when the user backs out of an applet
            // without it producing anything. `libnx` reads that as
            // `LibAppletExitReason_Canceled` and fails the caller's
            // `libappletStart`, which is a path callers are written to
            // survive — unlike the alternatives, which are a success whose
            // output storage is empty (the caller reads zeroes as though the
            // user had typed them) or a refused command (`nnSdk` treats an
            // unknown command id as fatal).
            //
            // The one thing that must not happen is silence: the caller waits
            // on the state-changed event forever, so it is signalled here
            // whether it was fetched before the start or after it.
            "am:library-applet-accessor" => {
                let key = self.ipc_object_key(tls, handle);
                match cmd_id {
                    // GetAppletStateChangedEvent -> event. A **copy** handle,
                    // like every other event a service hands out.
                    Some(0) => {
                        let event = self.library_applet_event(key, STATE_CHANGED_EVENT);
                        if self.library_applet_finished(key) {
                            self.signal_event(event);
                        }
                        self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                    }
                    // IsCompleted -> bool. What the caller polls between
                    // waits on the event above; an applet that has been
                    // started here has already finished.
                    Some(1) => {
                        let done = u8::from(self.library_applet_finished(key));
                        self.write_ipc_response(tls, 0, &[], &done.to_le_bytes(), &[])
                    }
                    // Start, RequestExit and Terminate: all three end the
                    // applet, because it was over before it began.
                    Some(10) | Some(20) | Some(25) => {
                        if let Some(applet) = self.am_applets.get_mut(&key) {
                            applet.finish();
                        }
                        self.signal_library_applet_event(key, STATE_CHANGED_EVENT);
                        self.write_ipc_response(tls, 0, &[], &[], &[])
                    }
                    // GetResult: why the applet ended. See the note above the
                    // interface for why this is a cancellation rather than a
                    // success.
                    Some(30) => {
                        /// `am` description 22, which `libnx` maps to
                        /// `LibAppletExitReason_Canceled`.
                        const CANCELLED: u32 = 128 | (22 << 9);
                        self.write_ipc_response(tls, CANCELLED, &[], &[], &[])
                    }
                    // PushInData / PushExtraStorage / PushInteractiveInData:
                    // the storages the caller hands the applet. Accepted and
                    // dropped — there is no applet to read them, and the
                    // caller keeps its own reference to each one.
                    Some(100) | Some(102) | Some(103) => {
                        self.write_ipc_response(tls, 0, &[], &[], &[])
                    }
                    // PopOutData / PopInteractiveOutData -> IStorage: what
                    // the applet produced. An applet that never ran produced
                    // nothing, which is a real answer rather than an empty
                    // storage — a caller reading a zeroed reply struct
                    // believes every field in it.
                    Some(101) | Some(104) => {
                        /// `am` description 3: no storage to pop.
                        const NO_DATA: u32 = 128 | (3 << 9);
                        self.write_ipc_response(tls, NO_DATA, &[], &[], &[])
                    }
                    // GetPopOutDataEvent / GetPopInteractiveOutDataEvent:
                    // fired when there is something to pop, which there never
                    // is. Handed out and left dark, so a caller that waits on
                    // one times out instead of reading a handle of 0.
                    Some(105) => {
                        let event = self.library_applet_event(key, POP_OUT_DATA_EVENT);
                        self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                    }
                    Some(106) => {
                        let event = self.library_applet_event(key, POP_INTERACTIVE_OUT_DATA_EVENT);
                        self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                    }
                    // NeedsToExitProcess -> bool: whether the caller has to
                    // exit for the applet to run. Nothing here does.
                    Some(110) => self.write_ipc_response(tls, 0, &[], &0u8.to_le_bytes(), &[]),
                    // GetLibraryAppletInfo -> LibraryAppletInfo { AppletId,
                    // LibraryAppletMode }: what this accessor was created for.
                    Some(120) => {
                        let mut info = [0u8; 8];
                        if let Some(applet) = self.am_applets.get(&key) {
                            info[..4].copy_from_slice(&applet.id.to_le_bytes());
                            info[4..].copy_from_slice(&applet.mode.to_le_bytes());
                        }
                        self.write_ipc_response(tls, 0, &[], &info, &[])
                    }
                    // RequestForAppletToGetForeground: the caller offering
                    // the screen to an applet that is not there.
                    Some(150) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                    // GetIndirectLayerConsumerHandle, for the modes that
                    // compose the applet's frames into the caller's own
                    // display. There are no frames to compose.
                    _ => self.unimplemented_command(tls, &iface, cmd_id),
                }
            }
            // `am`'s IStorage: a byte buffer passed between applets. Distinct
            // from `fsp-srv`'s IStorage (the process's RomFS) — same name,
            // different interface — and reached only through an accessor.
            "am:storage" => match cmd_id {
                // Open -> IStorageAccessor. Both objects address the same
                // bytes, so the accessor records which storage it belongs to
                // rather than taking a copy that could then diverge.
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
                    // Write(s64 offset, buffer<in>) / Read(s64 offset,
                    // buffer<out>). The offset is the request's only raw
                    // argument; the bytes travel in a buffer.
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
            // IDisplayController (capture buffers), IDebugFunctions, and any
            // session that never named itself. Nothing here can answer those
            // honestly: a capture buffer has no contents.
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }

    /// The event an `ILockAccessor` hands out, created **signalled** and
    /// manual-reset.
    ///
    /// That is not a shortcut: hardware hands out a lock nobody is holding,
    /// and the Home Menu takes the event as proof of that before it will run a
    /// transition. It polls the event with `nn::os::TryWaitSystemEvent` and
    /// **aborts** when it comes back clear, which is where qlaunch stopped —
    /// one `ICommonStateGetter::GetReaderLockAccessorEx` and one `GetEvent`
    /// after its first scene started.
    ///
    /// One object, because nothing here contends for a HOME button.
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

    /// `AppletId_LibraryAppletWeb`, which is what lennytube asks for when it
    /// hands the screen to the browser.
    const APPLET_WEB: u32 = 0x13;

    #[test]
    fn the_system_process_common_functions_chain_hands_back_real_sessions() {
        // `appletAE` 450 and the observer behind it both return an interface,
        // and `nnSdk` reads one as a move handle: answering either with a bare
        // success leaves the client constructing a null `SharedPointer` and
        // faulting on its first virtual call, rather than failing here.
        let mut cpu = request(false, 450, &[]);
        cpu.register_service_handle(9, "appletAE");
        // Set against the default, so a 450 that wrongly reached for
        // `set_applet_is_application` could not pass by matching it.
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

        // GetAppletAlternativeFunctions, the command the same caller reaches
        // next, has the same shape and the same failure when refused.
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

        // Neither opens a proxy, so the flag every Open*Proxy beside them sets
        // — which decides whether the applet is told `FocusStateChanged` or
        // `ChangeIntoForeground` — stays where it was.
        assert!(!cpu.applet_is_application);
    }

    #[test]
    fn am_reports_the_handheld_operation_mode_it_always_claimed_to() {
        // AppletOperationMode_Handheld is 0 and Console is 1. This answered 1
        // under a comment saying Handheld, so NX-Fetch printed "Docked" beside
        // a 720p framebuffer.
        let mut cpu = request(false, 5, &[]);
        cpu.register_service_handle(9, "am:common-state-getter");
        cpu.applet_request(TLS, 9, Some(5)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "Handheld");
    }

    #[test]
    fn each_applet_event_is_named_after_the_interface_that_hands_it_out() {
        // These two names were swapped: `IApplicationFunctions`'s
        // GetGpuErrorDetectedSystemEvent handed out an event called
        // "am:self-controller", and `ISelfController`'s two events were called
        // "am:gpu-error". Only `TRACE_WAIT` reads these names, which is
        // exactly why it mattered -- a wait trace showing "A Short Hike"
        // blocked on "am:self-controller" sent a debugging session looking for
        // an applet focus event that was never involved. The event it was
        // really waiting on is the GPU-error one, which nothing here ever
        // fires because nothing here ever faults the GPU.
        let mut cpu = request(false, 130, &[]);
        cpu.register_service_handle(9, "am:application-functions");
        cpu.applet_request(TLS, 9, Some(130)).unwrap();
        let event = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
        assert_ne!(
            event, 0,
            "GetGpuErrorDetectedSystemEvent handed back no handle"
        );
        assert_eq!(cpu.event_name(event), Some("am:gpu-error"));

        // ISelfController::GetAccumulatedSuspendedTickChangedEvent.
        let mut cpu = request(false, 91, &[]);
        cpu.register_service_handle(9, "am:self-controller");
        cpu.applet_request(TLS, 9, Some(91)).unwrap();
        let event = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
        assert_ne!(event, 0);
        assert_eq!(cpu.event_name(event), Some("am:self-controller"));
    }

    #[test]
    fn the_preselected_user_is_handed_over_once_and_then_it_is_gone() {
        // The HOME menu picks the user before it starts a title and leaves the
        // choice as a `PreselectedUser` launch parameter.
        // `nn::account::Initialize` pops it and caches the uid;
        // `nn::account::OpenPreselectedUser` asserts when that cached uid is
        // zero. Refusing every kind of launch parameter is what aborted Just
        // Dance 2019 inside `nn::init::Start`, before it had asked `sm` for a
        // single service.
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

        // What `nn::account::detail::TryPopPreselectedUser` reads: it refuses
        // anything shorter than 0x88 bytes, checks the magic and the version,
        // and copies the uid out of offset 8. A uid of zero is what it means
        // by "nobody", so the one thing this must never hand over is zeroes.
        let data = cpu.am_storages[&Cpu::object_key(storage, 0)].clone();
        assert_eq!(data.len(), 0x88);
        assert_eq!(
            u32::from_le_bytes(data[..4].try_into().unwrap()),
            0xC794_97CA
        );
        assert_eq!(data[4], 1, "layout version");
        assert_eq!(&data[8..0x18], &super::ACCOUNT_UID[..]);
        assert_ne!(super::ACCOUNT_UID, [0u8; 16]);

        // `am` hands each launch parameter over once and forgets it, which is
        // what stops a second `nn::account::Initialize` caching a user the
        // launcher never chose.
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
        // Only the kinds a launcher actually fills in are here. `UserChannel`
        // (1) is application-to-application data that nothing here writes, and
        // answering it with the preselected user's block — or with any
        // success — would hand the caller bytes it would then parse as its own.
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
        // Command 210 is unnamed on switchbrew, but its shape is documented:
        // no input, one out event. An out-event is one of the two things a
        // caller cannot invent for itself, so a bare success here is a caller
        // waiting on handle 0 -- and `nnSdk` answers a *refusal* with an
        // svcBreak, which is where Tomodachi Life stopped.
        let mut cpu = request(false, 210, &[]);
        cpu.register_service_handle(9, "am:application-functions");
        cpu.applet_request(TLS, 9, Some(210)).unwrap();
        let event = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
        assert_ne!(event, 0, "command 210 handed back no event handle");
        assert_eq!(cpu.event_name(event), Some("am:application-functions-210"));

        // Nothing here can ask a title to exit, so the event never fires. A
        // wait on it is a wait for something that genuinely never happens.
        assert_eq!(cpu.event_signaled(event), Some(false));

        // Asking again has to give back the event the caller is already
        // waiting on, not a fresh one nothing will ever signal either.
        marshal(&mut cpu, false, 210, &[]);
        cpu.applet_request(TLS, 9, Some(210)).unwrap();
        assert_eq!(u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap()), event);
    }

    #[test]
    fn a_stubbed_answer_names_itself_once_and_still_succeeds() {
        // SetTerminateResult is accepted and thrown away: the reply is a bare
        // success, which is the right *shape* and an answer with nothing
        // behind it. Neither existing warning covers that -- the command is
        // neither missing nor refused -- so the guest believes it and the
        // consequence surfaces somewhere else entirely. The marker is what
        // makes the belief visible; it must not change the reply.
        let mut cpu = request(false, 22, &4u32.to_le_bytes());
        cpu.register_service_handle(9, "am:application-functions");
        cpu.applet_request(TLS, 9, Some(22)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "Result");
        let trace = String::from_utf8_lossy(&cpu.trace).into_owned();
        assert!(
            trace.contains("[ipc] stub: am:application-functions cmd=Some(22)"),
            "the stub went unreported: {trace:?}"
        );

        // Once per (interface, command): a title that polls one every frame
        // would otherwise bury every other line in the trace the browser
        // drains.
        marshal(&mut cpu, false, 22, &4u32.to_le_bytes());
        cpu.applet_request(TLS, 9, Some(22)).unwrap();
        let repeated = String::from_utf8_lossy(&cpu.trace)
            .matches("[ipc] stub:")
            .count();
        assert_eq!(repeated, 1, "the stub was reported on every call");
    }

    #[test]
    fn get_save_data_size_reports_the_quota_the_title_was_actually_allotted() {
        // GetSaveDataSize(u8 SaveDataType, u128 userId) -> two s64s. The
        // payload is 0x18 bytes: the type padded out to eight, then the uid.
        // Neither changes the answer -- there is one user and one save behind
        // it -- so the request is marshalled the way a title sends it and the
        // reply is checked, not the parse.
        let mut payload = [0u8; 0x18];
        payload[0] = 1; // SaveDataType::Account
        payload[8..].copy_from_slice(&super::ACCOUNT_UID);

        // Tomodachi Life's own NACP figures, which is what a console reads out
        // of the Control NCA.
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
        // ExtendSaveData(u8 SaveDataType, u128 userId, s64 size, s64 journal):
        // GetSaveDataSize's 0x18-byte payload with the two sizes appended.
        // `nn::fs::ExtendSaveData` aborts on any error, so refusing this is an
        // svcBreak — which is where Minecraft stopped, 259M steps in.
        const SIZE: i64 = 0x1200_0000;
        const JOURNAL: i64 = 0x0100_0000;
        let mut payload = [0u8; 0x28];
        payload[0] = 1; // SaveDataType::Account
        payload[8..0x18].copy_from_slice(&super::ACCOUNT_UID);
        payload[0x18..0x20].copy_from_slice(&SIZE.to_le_bytes());
        payload[0x20..].copy_from_slice(&JOURNAL.to_le_bytes());

        let mut cpu = request(false, 25, &payload);
        cpu.register_service_handle(9, "am:application-functions");
        cpu.applet_request(TLS, 9, Some(25)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "Result");

        // And what 26 reports afterwards is what the title just set, not the
        // NACP figure it has already moved past.
        marshal(&mut cpu, false, 26, &[0u8; 0x18]);
        cpu.applet_request(TLS, 9, Some(26)).unwrap();
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap() as i64, SIZE);
        assert_eq!(cpu.mem.read_u64(TLS + 0x28).unwrap() as i64, JOURNAL);
    }

    #[test]
    fn the_save_data_ceilings_are_reported_apart_from_the_sizes() {
        // 26 reports what the save was created at, 28 how far it may be
        // extended, 35 the same for the console-wide save. Three commands
        // reporting the same two s64s is exactly the shape where a mixed-up
        // pair goes unnoticed, so each is checked against a distinct number.
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
        // A NACP commonly gives a save a size and no ceiling. That 0 is the
        // title's own statement that it never extends this save, so it is
        // reported rather than quietly replaced with headroom the system never
        // agreed to — the failure that would cause surfaces nowhere near here.
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
        // An s32 then an s64, each aligned to its own width the way `sf`
        // marshals a pair of outputs — so the size is at +8 and +4 is padding.
        // Packing the two would put the size where the caller reads padding
        // and report a ceiling of zero.
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
        // Nothing has called `set_save_data_sizes` -- a bare Program NCA has no
        // NACP to read one out of. The fallback has to be a *quota*, not zero:
        // a title told it has nowhere to put its save is a title that does not
        // write one, and that failure looks nothing like a missing command.
        let mut payload = [0u8; 0x18];
        payload[0] = 1;
        let mut cpu = request(false, 26, &payload);
        cpu.register_service_handle(9, "am:application-functions");
        cpu.applet_request(TLS, 9, Some(26)).unwrap();
        let size = cpu.mem.read_u64(TLS + 0x20).unwrap() as i64;
        let journal = cpu.mem.read_u64(TLS + 0x28).unwrap() as i64;
        assert_eq!(size, crate::cpu::fs::DEFAULT_SAVE_DATA_SIZE);
        assert_eq!(journal, crate::cpu::fs::DEFAULT_SAVE_DATA_JOURNAL_SIZE);
        // Above what a large retail title asks for: Tomodachi Life's NACP
        // declares 54 MiB of save and 10 MiB of journal.
        assert!(
            size >= 56_623_104,
            "default quota is smaller than a real title's save"
        );
        assert!(
            journal >= 10_485_760,
            "default journal is smaller than a real title's"
        );
    }

    /// Create a library applet on a non-domain session and return the `Cpu`
    /// and the accessor handle the reply moved back.
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
        // Both pop the common arguments, then a private struct, then the
        // argument that struct describes -- and stop dead on the pop that is
        // not there, because `PopInData` answers 2128-0003 and `nnSdk`
        // aborts on it rather than carry on. Seeding only the first two is
        // what took both of them down.
        const SWKBD: u64 = 0x0100_0000_0000_1008;
        const CONTROLLER: u64 = 0x0100_0000_0000_1003;
        /// `am` description 3, what a pop past the last storage answers.
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
                // The applet reads the size before the bytes, and a struct
                // of the wrong width is one it will not read at all.
                write_request(&mut cpu, 0, &[]);
                cpu.applet_request(TLS, storage, Some(0)).unwrap();
                let accessor = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
                write_request(&mut cpu, 0, &[]);
                cpu.applet_request(TLS, accessor, Some(0)).unwrap();
                sizes.push(cpu.mem.read_u64(TLS + 0x20).unwrap());
            }

            // LibAppletCommonArguments, then the pair the applet's own
            // interface version describes.
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
        // `starter` sits in the middle of the library applets' title ids and
        // is not one of them, so the run cannot be counted straight through:
        // doing that put myPage on `gift`'s id, which is not a library applet
        // here -- so nothing seeded its launch storages and its first
        // `PopInData` was refused with 2128-0003.
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
        // And the two that are not library applets are not treated as ones:
        // a library applet is handed launch storages and no preselected user,
        // and `starter` is handed the opposite.
        assert!(super::is_library_applet(0x0100_0000_0000_1013));
        assert!(!super::is_library_applet(0x0100_0000_0000_1012));

        // myPage's argument is the 9.0.0+ width its interface version claims,
        // and it names the one user this console has -- a zero uid is "no
        // user", which is not a page the applet can show.
        assert_eq!(
            super::applet_interface_version(0x0100_0000_0000_1013),
            0x1_0000
        );
        let arg = &super::applet_launch_storages(0x0100_0000_0000_1013)[0];
        assert_eq!(arg.len(), 0x10A8);
        assert_eq!(arg[8..24], crate::cpu::acc::ACCOUNT_UID);
    }

    #[test]
    fn the_controller_applet_is_told_what_it_may_offer() {
        // `ControllerSupportArgPrivate` names its own size and the size of
        // the argument behind it, and the applet picks the struct it reads by
        // them -- so both have to match what is actually pushed, and the
        // interface version the common arguments claim has to be the one
        // whose argument shape that is.
        const CONTROLLER: u64 = 0x0100_0000_0000_1003;
        let storages = super::applet_launch_storages(CONTROLLER);
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

        // The styles it may offer are the ones `hid` can actually publish:
        // an applet that offers a controller the console then never presents
        // is one the user cannot get past. Handheld is the one that matters
        // here, since that is the mode this console reports.
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
        // Nothing here can run the applet a caller asks for, so the useful
        // answer is one that ends: the caller waits on the state-changed
        // event before it does anything else, and an event that never fires
        // is a hang rather than a failure it can report.
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

        // GetResult: `libnx` reads this one as LibAppletExitReason_Canceled,
        // which is a path callers survive. A success here would instead have
        // them read an empty output storage as real input.
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
        // The caller usually takes the event before starting the applet, but
        // it does not have to, and an applet that has already ended has to
        // hand back an event that is already fired.
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
        // An empty storage would be worse than a refusal: a caller reads its
        // reply struct field by field and believes the zeroes.
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
        // The caller writes its launch arguments into this through an
        // IStorageAccessor, so the bytes have to be there to be written over.
        let mut cpu = request(false, 10, &0x1000u64.to_le_bytes());
        cpu.register_service_handle(9, "am:library-applet-creator");
        cpu.applet_request(TLS, 9, Some(10)).unwrap();
        let storage = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
        assert_eq!(cpu.service_name(storage), Some("am:storage"));
        assert_eq!(cpu.am_storages[&Cpu::object_key(storage, 0)].len(), 0x1000);

        // A size no caller sends is refused rather than allocated.
        write_request(&mut cpu, 10, &u64::MAX.to_le_bytes());
        cpu.applet_request(TLS, 9, Some(10)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x18).unwrap(),
            1 | (104 << 9),
            "out of memory"
        );
    }
}
