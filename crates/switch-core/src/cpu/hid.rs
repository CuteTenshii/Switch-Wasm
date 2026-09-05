//! `hid`: the controllers, and the shared memory their state is published in.
//!
//! Almost nothing is answered here, the guest maps hid's shared memory once
//! and reads it directly every frame after that, so what this service mostly
//! does is hand that mapping over and agree about which controllers exist.
//! The layout itself is [`super::hid_shmem`].

use super::Cpu;
use crate::Result;

impl Cpu {
    /// `hid`: the input service.
    ///
    /// Input arrives on Switch in two halves, and only one of them is IPC. The
    /// **data** (buttons, sticks, touch points) lives in a 256 KiB shared
    /// memory region the `hid` sysmodule writes continuously and the
    /// application reads directly, with no IPC per frame; this emulator
    /// already fills it from [`Cpu::set_gamepad_state`]. What `IHidServer`
    /// does is the **negotiation** around it: which controller styles and
    /// player slots the app supports, turning the npads and touchscreen on,
    /// and handing over the shared memory in the first place:
    ///
    /// ```text
    /// IHidServer::CreateAppletResource(aruid) -> IAppletResource
    /// IAppletResource::GetSharedMemoryHandle() -> a copy handle
    /// svcMapSharedMemory(handle, addr, 0x40000)
    /// ```
    ///
    /// None of that existed. `libnx` survived it because it maps the region by
    /// size and this emulator recognises it that way, so homebrew got working
    /// input out of a fabricated reply, but `nnSdk` calls a method on the
    /// `IAppletResource` it was handed, and a fabricated object id is not one.
    /// Which `hid` interface a session handle stands for.
    ///
    /// `hid` and `hid:dbg` are both `IHidServer`, the debug service is the
    /// same interface at higher privilege, and nothing here enforces
    /// privilege. `hid:sys` is a *different* interface (`IHidSystemServer`),
    /// so it keeps its own name and gets its own dispatch below.
    fn hid_interface_for(name: Option<&str>) -> String {
        match name {
            Some("hid") | Some("hid:dbg") | None => "hid:server".to_string(),
            Some(name) => name.to_string(),
        }
    }

    pub(super) fn hid_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let name = Self::hid_interface_for(self.service_name(handle));
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, &name);
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, "hid:control", cmd_id),
            };
        }
        let object_id = self.ipc_domain_object_id(tls);
        let iface = if self.ipc_is_domain_request(tls) {
            self.domain_interface(handle, object_id)
                .unwrap_or("hid:server")
                .to_string()
        } else {
            Self::hid_interface_for(self.service_name(handle))
        };
        let data = self.ipc_request_data(tls);
        match iface.as_str() {
            "hid:server" => match cmd_id {
                // CreateAppletResource(aruid) -> IAppletResource.
                Some(0) => {
                    self.reply_with_interface(tls, handle, "hid:applet-resource")?;
                    Ok(())
                }
                // Activate{DebugPad,TouchScreen,Mouse,Keyboard,Npad},
                // ActivateNpadWithRevision, DeactivateNpad, DisconnectNpad,
                // Start/StopSixAxisSensor, the joy-assignment modes,
                // SetNpadHandheldActivationMode, and the Set* half of the
                // style/id negotiation below.
                //
                // Every one of these is a setter: the shared memory this
                // emulator publishes always carries one connected handheld
                // pad, whatever the caller asks to activate, so accepting the
                // request is the whole implementation.
                // The 1000-range commands are the same shape: per-title
                // configuration of how input is delivered (communication mode,
                // touch-screen configuration, vibration style), each carrying
                // a small argument and expecting nothing back. None of it
                // changes what the shared memory here publishes.
                Some(1) | Some(11) | Some(21) | Some(31) | Some(66) | Some(67) | Some(91)
                | Some(103) | Some(104) | Some(107) | Some(109) | Some(122..=125) | Some(128)
                | Some(1000..=1004) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // SetGestureOutputRanges(u32 width, u32 height, u64 aruid),
                // added in 18.0.0 and named but not described on switchbrew.
                // The shape is the request Tomodachi Life sends, which is
                // unambiguous: `00 05 00 00 | d0 02 00 00 | 01 00 …`, 1280,
                // 720, and the aruid `am`'s window controller handed out.
                //
                // It is the coordinate space the gesture engine reports in.
                // A title sets it to the resolution it is drawing at so that a
                // swipe comes back in the same units as its own geometry
                // rather than in the panel's, which is why what arrives here
                // is exactly the handheld display size, the one `vi` and `am`
                // already agree on (see [`super::OperationMode::display_size`]).
                //
                // Nothing here synthesises gestures, so there is no engine to
                // point at a range and accepting is the whole implementation.
                // Refusing was not free: `nnSdk` answers an unknown command id
                // with an svcBreak, and this is where the title stopped once
                // its save data was answered, 454,291,947 steps in, the first
                // blocker outside `am` in a long while.
                //
                // ActivateGesture (91) joins the void setters above for the
                // same reason. This title does not call it: it sets the range
                // and never turns the engine on, but the two are one pair in
                // every SDK that uses either, and the one that is missing when
                // the other is answered is the one that aborts.
                Some(92) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // SetSupportedNpadStyleSet(u32 style_set, aruid) and its
                // readback. A caller that sets a style set and reads back
                // something else decides the pad it wants does not exist,
                // which is what the generic reply's incrementing object id
                // looked like.
                // It also decides which style the pad is *published* in: see
                // `NPAD_PRESENTATIONS`, so a set that changes changes what is
                // in shared memory, which is precisely what 106's event
                // reports.
                Some(100) => {
                    let styles = self.mem.read_u32(data)?;
                    if styles != self.npad_style_set {
                        self.npad_style_set = styles;
                        if let Some(event) = self.npad_style_update_event {
                            self.signal_event(event);
                        }
                    }
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                Some(101) => {
                    let styles = self.npad_style_set;
                    self.write_ipc_response(tls, 0, &[], &styles.to_le_bytes(), &[])
                }
                // SetSupportedNpadIdType: the id list arrives in a buffer, and
                // there is one pad here regardless of which slots are asked
                // for.
                Some(102) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // AcquireNpadStyleSetUpdateEventHandle(npad_id, aruid, u64):
                // fires when a controller is connected or its style changes.
                // Nothing here hot-plugs, but the style does change, 100
                // above signals it, and the pad is already published by the
                // time anyone asks, so it starts **signalled**: a caller
                // waiting to be told the pad has settled is waiting for
                // something that has already happened.
                //
                // One object, for the reason every other kept event here is
                // one: handed a second copy, a caller waits on a handle that
                // 100 does not signal.
                Some(106) => {
                    let event = match self.npad_style_update_event {
                        Some(event) => event,
                        None => {
                            let event = self.alloc_event("hid:npad-style-update", true);
                            self.npad_style_update_event = Some(event);
                            event
                        }
                    };
                    self.signal_event(event);
                    self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                }
                // GetPlayerLedPattern(npad_id) -> the four player LEDs. One
                // pad, so player 1: the first LED.
                Some(108) => self.write_ipc_response(tls, 0, &[], &1u64.to_le_bytes(), &[]),
                // Set/GetNpadJoyHoldType(aruid, u64).
                Some(120) => {
                    self.npad_joy_hold_type = self.mem.read_u64(data.wrapping_add(8))?;
                    // `GetNpadJoyHoldType` reads this out of shared memory
                    // rather than asking for it, so setting it here and not
                    // there would leave the two answers disagreeing.
                    self.write_npad_condition();
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                Some(121) => {
                    let hold = self.npad_joy_hold_type;
                    self.write_ipc_response(tls, 0, &[], &hold.to_le_bytes(), &[])
                }
                // GetNpadHandheldActivationMode.
                Some(129) => self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[]),
                // ---- vibration ----
                //
                // A `HidVibrationValue` is four floats: amplitude and
                // frequency for a low band and a high band. Switch rumble is
                // two linear resonant actuators driven independently, which is
                // also what the browser's Gamepad API exposes as
                // `dual-rumble`'s strong and weak magnitudes, so the two
                // amplitudes are kept and [`Cpu::vibration`] hands them to the
                // page.
                //
                // GetVibrationDeviceInfo -> { device_type, position }: a
                // linear resonant actuator (1) on the left (0).
                Some(200) => {
                    let mut info = Vec::with_capacity(8);
                    info.extend_from_slice(&1u32.to_le_bytes());
                    info.extend_from_slice(&0u32.to_le_bytes());
                    self.write_ipc_response(tls, 0, &[], &info, &[])
                }
                // SendVibrationValue(handle, HidVibrationValue, aruid): the
                // value follows the u32 handle, so the amplitudes are at +4
                // and +0xc.
                Some(201) => {
                    let low = f32::from_bits(self.mem.read_u32(data.wrapping_add(4))?);
                    let high = f32::from_bits(self.mem.read_u32(data.wrapping_add(0xc))?);
                    self.set_vibration(low, high);
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetActualVibrationValue -> what is actually playing.
                Some(202) => {
                    let (low, high) = self.vibration();
                    let mut value = Vec::with_capacity(16);
                    value.extend_from_slice(&low.to_bits().to_le_bytes());
                    value.extend_from_slice(&160.0f32.to_bits().to_le_bytes());
                    value.extend_from_slice(&high.to_bits().to_le_bytes());
                    value.extend_from_slice(&320.0f32.to_bits().to_le_bytes());
                    self.write_ipc_response(tls, 0, &[], &value, &[])
                }
                // CreateActiveVibrationDeviceList -> IActiveVibrationDeviceList.
                Some(203) => {
                    self.reply_with_interface(tls, handle, "hid:vibration-devices")?;
                    Ok(())
                }
                // PermitVibration / Begin/EndPermitVibrationSession.
                Some(204) | Some(209) | Some(210) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // IsVibrationPermitted / IsVibrationDeviceMounted: there is a
                // pad and the page decides whether it can actually rumble.
                Some(205) | Some(211) => {
                    self.write_ipc_response(tls, 0, &[], &1u8.to_le_bytes(), &[])
                }
                // SendVibrationValues(handles[], values[]): the arrays arrive
                // as buffers. Only the first value is kept, this emulator
                // drives one actuator pair, not one per device.
                Some(206) => {
                    if let Some((addr, size)) = self.ipc_input_buffer(tls, 1) {
                        if size >= 16 {
                            let low = f32::from_bits(self.mem.read_u32(addr)?);
                            let high = f32::from_bits(self.mem.read_u32(addr.wrapping_add(8))?);
                            self.set_vibration(low, high);
                        }
                    }
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // ---- what a pad is and how it is attached ----
                //
                // Each of these takes a `u32 npad_id` and answers about that
                // one pad, so they all start by asking which of the two this
                // emulator publishes is being asked about: the handheld slot,
                // or the Pro Controller in player 1's.
                //
                // HasBattery / HasLeftRightBattery -> bool, or one bool per
                // half. Every controller here runs on a battery; only the
                // handheld pad's is in two halves, which is the same split
                // `device_type` already draws between `HandheldLeft|Right` and
                // a single `FullKey`.
                //
                // GetNpadInterfaceType -> u8, and GetNpadLeftRightInterfaceType
                // -> u8 left, u8 right. `HidNpadInterfaceType` says how the
                // controller reaches the console: Bluetooth (1), rail (2) or
                // USB (3). A title reads it to decide what a pad is capable of
                //: whether it can be told to sleep, how much of a rumble
                // budget it has, which glyphs to draw for it.
                //
                // The shared memory already said which is which: slot 0 is a
                // Pro Controller carrying `ATTR_WIRED`, and a wired Pro
                // Controller is one on its USB cable; the handheld slot
                // carries left- and right-wired, which is what a pair of
                // Joy-Con report while they sit on the rails. Nothing here is
                // ever Bluetooth: there is no radio for one to be on the far
                // end of.
                Some(403..=406) => {
                    /// `HidNpadIdType_Handheld`; players 1-8 are 0-7.
                    const HANDHELD: u32 = 0x20;
                    /// `HidNpadInterfaceType_Rail` and `_USB`.
                    const RAIL: u8 = 2;
                    const USB: u8 = 3;
                    let handheld = self.mem.read_u32(data)? == HANDHELD;
                    let reply: &[u8] = match cmd_id {
                        Some(403) => &[1],
                        Some(404) if handheld => &[1, 1],
                        Some(404) => &[0, 0],
                        // One pad reaches the console one way, so both of its
                        // halves report the same interface.
                        Some(406) if handheld => &[RAIL, RAIL],
                        Some(406) => &[USB, USB],
                        _ if handheld => &[RAIL],
                        _ => &[USB],
                    };
                    self.write_ipc_response(tls, 0, &[], reply, &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // `IHidSystemServer`: the privileged half of hid, opened as
            // `hid:sys`. Nothing an application needs for *input* is here,
            // that is `IHidServer` above, but homebrew reaches for this one
            // to take the console's own buttons over, and `libnx` opens the
            // session during `hidsysInitialize` whether or not it ever sends
            // a command.
            //
            // Which is exactly how this went wrong: opening the session is
            // itself the traffic. `libnx` records the pointer buffer size on
            // it before any command, and with `hid:sys` not routed here that
            // control request fell through to the generic reply and was
            // answered with a fabricated object id, the same failure `ns:am2`
            // had. The command ids below come from `libnx`'s `hidsys.c`.
            "hid:sys" => match cmd_id {
                // Acquire{Home,Sleep,Capture}ButtonEventHandle -> a copy
                // handle. This console has no Home, Sleep or Capture button,
                // so each event is handed out and never signalled, the
                // caller waits on a press that cannot arrive, which is the
                // truth rather than a fabricated one.
                Some(101) => {
                    let event = self.alloc_event("hid:sys-home-button", true);
                    self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                }
                Some(121) => {
                    let event = self.alloc_event("hid:sys-sleep-button", true);
                    self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                }
                Some(141) => {
                    let event = self.alloc_event("hid:sys-capture-button", true);
                    self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                }
                // AcquireJoyDetachOnBluetoothOffEventHandle -> a copy handle.
                // It fires when Joy-Cons are detached because Bluetooth went
                // off; there is no Bluetooth radio here, so it never does.
                Some(751) => {
                    let event = self.alloc_event("hid:sys-joy-detach", true);
                    self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                }
                // AcquireConnectionTriggerTimeoutEvent (544) and
                // AcquireDeviceRegisteredEventForControllerSupport (546): the
                // pair the controller-support applet waits on while it asks
                // for a button press on a controller to pair. Nothing here
                // pairs, so neither the registration nor the timeout arrives.
                Some(544) | Some(546) => {
                    let name = if cmd_id == Some(544) {
                        "hid:sys-connection-trigger-timeout"
                    } else {
                        "hid:sys-device-registered"
                    };
                    let event = self.alloc_event(name, true);
                    self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                }
                // Activate{Home,Sleep,Capture}Button, and
                // EnableAppletToGetInput. Every one is a setter over state
                // this emulator does not have: there is one pad, it is
                // always connected, and the caller is always the foreground
                // applet, so accepting the request is the whole
                // implementation.
                Some(111) | Some(131) | Some(151) | Some(301) | Some(304) | Some(305)
                | Some(503) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // ApplyNpadSystemCommonPolicy and its `Full` form (308), which
                // differs only in a status bit nothing here reads.
                //
                // **This is how a system applet asks for controllers**: it
                // never sends `SetSupportedNpadStyleSet`, so accepting this
                // and discarding it left `npad_style_set` at zero for a whole
                // run. Eden's `SetNpadSystemCommonPolicy` grants every style
                // there is; the honest equivalent is every style this console
                // can publish. The hold type it also assigns is already this
                // field's value, so there is nothing to write for it.
                Some(303) | Some(308) => {
                    let styles = super::supported_npad_style_set();
                    if styles != self.npad_style_set {
                        self.npad_style_set = styles;
                        if let Some(event) = self.npad_style_update_event {
                            self.signal_event(event);
                        }
                    }
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetMaskedSupportedNpadStyleSet(u64 aruid) -> NpadStyleSet:
                // what the *system* permits the caller, rather than what the
                // caller asked for at 101. Every style this console's pad can
                // be published in, which is the same set the controller
                // applet is handed in its launch struct.
                Some(310) => {
                    let styles = super::supported_npad_style_set();
                    self.write_ipc_response(tls, 0, &[], &styles.to_le_bytes(), &[])
                }
                // GetNpadCaptureButtonAssignment(u64 aruid) -> u64 count,
                // with the buttons themselves in an out buffer: which button
                // each pad has been assigned as its capture button. None of
                // them has one, so the count is zero and the buffer is left
                // alone, which is what Eden's `hid` does for an empty list.
                Some(313) => self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[]),
                // SetNpadSystemExtStateEnabled(bool, u64 aruid): whether the
                // caller may be handed pads in the SystemExt style. There is
                // one process here, and every slot carries that style
                // already, so the permission has nothing left to gate.
                Some(322) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // InitializeFirmwareUpdate (1000), its USB form
                // InitializeUsbFirmwareUpdateWithoutMemory (1135), and
                // SetFirmwareHotfixUpdateSkipEnabled (1120), which says
                // whether to skip the hotfix that update would apply. The pad
                // here is the console's own and has no firmware to flash, so
                // there is nothing to refuse, nothing to skip and nothing to
                // do.
                // FinalizeUsbFirmwareUpdate (1131) closes the same session.
                Some(1000) | Some(1120) | Some(1131) | Some(1135) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // CheckUsbFirmwareUpdateRequired -> bool. The pad here is the
                // console's own and has no firmware to flash, so none is.
                Some(1132) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                // IsJoyConRailEnabled (523) and IsJoyConAttachedOnAllRail
                // (525) -> bool: whether the console's rails are live, and
                // whether both Joy-Cons are actually seated in them.
                //
                // Both are true. Handheld play is exactly the state where the
                // pads are on the rails, and this console reports handheld
                // operation mode and publishes a handheld npad -- answering
                // otherwise would describe a console that cannot be played
                // the only way this one can.
                Some(523) | Some(525) => self.write_ipc_response(tls, 0, &[], &[1u8], &[]),
                // IsUsbFullKeyControllerEnabled -> bool. There is no USB
                // controller here, wired or otherwise.
                Some(850) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                // SetTouchScreenMagnification / SetTouchScreenDefaultConfiguration
                // / SetForceHandheldStyleVibration: settings on a panel this
                // emulator does not model.
                Some(1150) | Some(1152) | Some(1155) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetTouchScreenDefaultConfiguration ->
                // nn::hid::TouchScreenConfigurationForNx, 0x10 bytes whose
                // first is the mode. Zero is `UseSystemSetting`, which is what
                // a console reports when nothing has overridden the panel.
                Some(1153) => self.write_ipc_response(tls, 0, &[], &[0u8; 0x10], &[]),
                // GetLastActiveNpad -> which controller was used last. There
                // is one pad and it is Player 1, so it is always npad 0.
                //
                // Refusing this is what stopped the Mii editor once
                // `IDisplayController` let it through: `nnSdk` answers an
                // unknown command id with an svcBreak.
                Some(306) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
                // GetNpadFullKeyGripColor -> two RGBA colours for the grips of
                // a Pro Controller. There is no Pro Controller here; black is
                // what an unpainted grip reports.
                Some(309) => self.write_ipc_response(tls, 0, &[], &[0u8; 8], &[]),
                // GetUniquePadIds -> the ids of the physically attached
                // controllers, written into a pointer buffer, with the count
                // returned as an s64. A "unique pad" is a detachable
                // controller; the pad here is the built-in handheld one, so
                // there are none and the buffer is left alone.
                Some(703) => self.write_ipc_response(tls, 0, &[], &0i64.to_le_bytes(), &[]),
                // SetNotificationLedPattern and its timeout form: the
                // breathing pattern for a Joy-Con's player LEDs. No LEDs, and
                // `hid`'s GetPlayerLedPattern above already reports the fixed
                // one.
                Some(830) | Some(831) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IActiveVibrationDeviceList::InitializeVibrationDevice.
            "hid:vibration-devices" => match cmd_id {
                Some(0) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IAppletResource: the handover of the shared memory the input
            // data actually lives in.
            "hid:applet-resource" => match cmd_id {
                Some(0) => {
                    let shmem = self.alloc_handle();
                    self.hid_shmem_handle = Some(shmem);
                    self.write_ipc_reply(tls, 0, &[shmem], &[], &[], &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::cpu::ipc::testing::*;

    #[test]
    fn the_gesture_pair_is_accepted_rather_than_refused() {
        // SetGestureOutputRanges(u32 width, u32 height, u64 aruid) and
        // ActivateGesture. Both are void, and `nnSdk` answers a refusal with
        // an svcBreak, so what is being pinned here is that neither reaches
        // `unimplemented_command`, which replies `cmif`'s unknown-command-id.
        const UNKNOWN_COMMAND_ID: u32 = 10 | (221 << 9);
        // What the title actually sends: the display size both `vi` and `am`
        // report, then the aruid `am`'s window controller hands out.
        let mut payload = Vec::new();
        let (width, height) = super::super::OperationMode::Handheld.display_size();
        payload.extend_from_slice(&width.to_le_bytes());
        payload.extend_from_slice(&height.to_le_bytes());
        payload.extend_from_slice(&1u64.to_le_bytes());
        assert_eq!(
            payload[..4],
            [0x00, 0x05, 0x00, 0x00],
            "1280, as the request carries it"
        );

        for command in [92u32, 91] {
            let mut cpu = request(false, command, &payload);
            cpu.register_service_handle(9, "hid");
            cpu.hid_request(TLS, 9, Some(command)).unwrap();
            let result = cpu.mem.read_u32(TLS + 0x18).unwrap();
            assert_ne!(result, UNKNOWN_COMMAND_ID, "command {command} was refused");
            assert_eq!(result, 0, "command {command}");
        }
    }

    #[test]
    fn the_pad_is_published_in_a_style_the_title_actually_asked_for() {
        // A title that accepts a pair of Joy-Cons and not a Pro Controller is
        // an ordinary thing to be. This used to publish FullKey and Handheld
        // whatever the title supported, so such a title found every slot in a
        // style it had not asked for and `nnSdk` aborted in the npad layer
        // with 2202-0710, one description along from the out-of-range npad id
        // it sits beside. `SetSupportedNpadStyleSet` was stored and read back
        // by its own getter and used for nothing else.
        use crate::cpu::hid_shmem as h;
        const SHMEM: u32 = 0x3000_0000;

        let mut cpu = request(false, 100, &h::STYLE_JOY_DUAL.to_le_bytes());
        cpu.mem.map_zero(SHMEM, 0x40000).unwrap();
        cpu.hid_shmem_addr = SHMEM;
        cpu.register_service_handle(9, "hid");
        cpu.hid_request(TLS, 9, Some(100)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0);

        cpu.set_gamepad_state(0x3, 0, 0, 0, 0);

        // Player 1 is a dual pair: the style it asked for, both halves as the
        // device type, and the state in the *joy_dual* LIFO. A style's states
        // are only ever read out of its own LIFO.
        let slot = SHMEM + h::NPAD;
        assert_eq!(
            cpu.mem.read_u32(slot + h::STYLE_SET).unwrap(),
            h::STYLE_JOY_DUAL
        );
        assert_eq!(
            cpu.mem.read_u32(slot + h::DEVICE_TYPE).unwrap(),
            h::DEVICE_JOY_LEFT | h::DEVICE_JOY_RIGHT
        );
        let entry = slot + h::JOY_DUAL_LIFO + h::LIFO_STORAGE;
        assert_eq!(cpu.mem.read_u64(entry + h::STATE_BUTTONS).unwrap(), 0x3);
        assert_eq!(
            cpu.mem.read_u32(entry + h::STATE_ATTRIBUTES).unwrap() & 1,
            1
        );
        // And nothing was left in the Pro Controller's LIFO for a reader that
        // asked for FullKey to find.
        let full_key = slot + h::FULL_KEY_LIFO + h::LIFO_STORAGE;
        assert_eq!(cpu.mem.read_u64(full_key + h::STATE_BUTTONS).unwrap(), 0);

        // The handheld slot is still published, as it always was: a title
        // that did not name the style just never reads it, and withholding it
        // could only cost a title that works today.
        let handheld = SHMEM + h::NPAD + h::HANDHELD_SLOT * h::ENTRY_SIZE;
        assert_eq!(
            cpu.mem.read_u32(handheld + h::STYLE_SET).unwrap(),
            h::STYLE_HANDHELD
        );
    }

    #[test]
    fn every_slot_carries_the_system_ext_lifo_the_home_menu_reads() {
        // A read watchpoint on all three candidate LIFOs through a boot of
        // 18.0.1's qlaunch finds twelve reads of SystemExt and none at all of
        // FullKey or handheld. It is a second copy every pad carries, not an
        // alternative presentation, so it goes out whatever style was asked
        // for.
        use crate::cpu::hid_shmem as h;
        const SHMEM: u32 = 0x3000_0000;

        let mut cpu = request(false, 100, &h::STYLE_JOY_DUAL.to_le_bytes());
        cpu.mem.map_zero(SHMEM, 0x40000).unwrap();
        cpu.hid_shmem_addr = SHMEM;
        cpu.register_service_handle(9, "hid");
        cpu.hid_request(TLS, 9, Some(100)).unwrap();
        cpu.set_gamepad_state(0x3, 1000, -2000, 3000, -4000);

        for (name, slot) in [
            ("player 1", SHMEM + h::NPAD),
            (
                "handheld",
                SHMEM + h::NPAD + h::HANDHELD_SLOT * h::ENTRY_SIZE,
            ),
        ] {
            let entry = slot + h::SYSTEM_EXT_LIFO + h::LIFO_STORAGE;
            assert_eq!(
                cpu.mem.read_u64(entry + h::STATE_BUTTONS).unwrap(),
                0x3,
                "{name} publishes buttons in the SystemExt LIFO"
            );
            assert_eq!(
                cpu.mem.read_u32(entry + h::STATE_STICK_L).unwrap() as i32,
                1000,
                "{name} publishes sticks there too"
            );
            assert_eq!(
                cpu.mem
                    .read_u64(slot + h::SYSTEM_EXT_LIFO + h::LIFO_COUNT)
                    .unwrap(),
                1,
                "{name}'s SystemExt LIFO header says it holds an entry"
            );
        }

        // And the style tag still names the physical controller only, the way
        // Eden leaves it: the reader above does not consult it.
        assert_eq!(
            cpu.mem.read_u32(SHMEM + h::NPAD + h::STYLE_SET).unwrap(),
            h::STYLE_JOY_DUAL
        );
    }

    #[test]
    fn apply_npad_system_common_policy_is_how_an_applet_asks_for_controllers() {
        // A system applet configures input with this rather than with
        // `SetSupportedNpadStyleSet`, which 18.0.1's qlaunch never sends.
        let mut cpu = request(false, 303, &[]);
        cpu.register_service_handle(9, "hid:sys");
        cpu.hid_request(TLS, 9, Some(303)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0);
        assert_eq!(
            cpu.npad_style_set,
            crate::cpu::supported_npad_style_set(),
            "the policy grants every style this console can publish"
        );

        // Which is also the answer 310 gives, and an applet handed a style by
        // one and refused it by the other is one the user cannot get past.
        marshal(&mut cpu, false, 310, &[]);
        cpu.hid_request(TLS, 9, Some(310)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x20).unwrap(),
            crate::cpu::supported_npad_style_set()
        );
    }

    #[test]
    fn the_npad_condition_is_published_so_the_hold_type_can_be_read_at_all() {
        // `nn::hid::GetNpadJoyHoldType` does not ask `hid` for the hold type,
        // it reads `nn::hid::NpadCondition` straight out of shared memory and
        // refuses the value unless `is_valid` is set. Left at its mapped
        // zeroes the region is not a default hold type, it is one that was
        // never published, and `nnSdk` answers that with an abort: 2202-0710,
        // which is where the 21.2.0 Home Menu stopped with no service request
        // anywhere near the fault.
        use crate::cpu::hid_shmem as h;
        const SHMEM: u32 = 0x3000_0000;
        const HORIZONTAL: u64 = 1;

        let mut cpu = request(false, 120, &[]);
        cpu.mem.map_zero(SHMEM, 0x40000).unwrap();
        cpu.hid_shmem_addr = SHMEM;
        cpu.register_service_handle(9, "hid");
        cpu.set_gamepad_state(0, 0, 0, 0, 0);

        let at = SHMEM + h::NPAD_CONDITION;
        assert_eq!(
            cpu.mem.read_u32(at + h::NPAD_CONDITION_VALID).unwrap(),
            1,
            "is_valid"
        );
        assert_eq!(
            cpu.mem
                .read_u32(at + h::NPAD_CONDITION_INITIALIZED)
                .unwrap(),
            1,
            "is_initialized"
        );

        // SetNpadJoyHoldType(aruid, u64) puts the hold type at +8 of the
        // request. Shared memory has to follow it: the same fact is published
        // here and answered by command 121, and the two cannot disagree.
        marshal(&mut cpu, false, 120, &[]);
        let data = cpu.ipc_request_data(TLS);
        cpu.mem.write_u64(data + 8, HORIZONTAL).unwrap();
        cpu.hid_request(TLS, 9, Some(120)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(at + h::NPAD_CONDITION_HOLD_TYPE).unwrap() as u64,
            HORIZONTAL
        );

        marshal(&mut cpu, false, 121, &[]);
        cpu.hid_request(TLS, 9, Some(121)).unwrap();
        assert_eq!(
            cpu.mem.read_u64(TLS + 0x20).unwrap(),
            HORIZONTAL,
            "command 121 agrees"
        );
    }

    #[test]
    fn a_title_that_names_no_styles_still_gets_the_pair_that_always_worked() {
        // `libnx` homebrew never calls SetSupportedNpadStyleSet and relies on
        // the defaults, so a style set of zero has to keep meaning what it
        // meant before any of this: a Pro Controller in slot 0 and a handheld
        // in slot 8.
        use crate::cpu::hid_shmem as h;
        assert_eq!(
            super::super::npad_presentation_for(0).style,
            h::STYLE_FULL_KEY
        );
        // A set naming only styles this console cannot be has to resolve to
        // something rather than to nothing.
        assert_eq!(
            super::super::npad_presentation_for(1 << 10).style,
            h::STYLE_FULL_KEY
        );
        // And the order is best-first: a title taking both gets the Pro
        // Controller rather than the pair.
        assert_eq!(
            super::super::npad_presentation_for(h::STYLE_FULL_KEY | h::STYLE_JOY_DUAL).style,
            h::STYLE_FULL_KEY
        );
        assert_eq!(
            super::super::npad_presentation_for(h::STYLE_JOY_RIGHT).style,
            h::STYLE_JOY_RIGHT
        );
    }

    /// Drive one `hid` command carrying a single `u32 npad_id` and hand back
    /// the bytes it answered with.
    fn hid_npad_query(command_id: u32, npad_id: u32, len: u32) -> Vec<u8> {
        let mut cpu = request(false, command_id, &npad_id.to_le_bytes());
        cpu.register_service_handle(9, "hid");
        cpu.hid_request(TLS, 9, Some(command_id)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x18).unwrap(),
            0,
            "Result for command {command_id}, npad {npad_id}"
        );
        (0..len)
            .map(|i| cpu.mem.read_u8(TLS + 0x20 + i).unwrap())
            .collect()
    }

    #[test]
    fn the_handheld_pad_is_on_its_rails_and_the_pro_controller_on_its_cable() {
        // GetNpadInterfaceType(u32 npad_id) and its left/right form. The two
        // pads published into the shared memory are attached differently, and
        // a single answer for both would contradict the attributes already
        // written there.
        const HANDHELD: u32 = 0x20;
        const PLAYER_1: u32 = 0;
        const RAIL: u8 = 2;
        const USB: u8 = 3;

        assert_eq!(hid_npad_query(405, HANDHELD, 1), [RAIL]);
        assert_eq!(hid_npad_query(406, HANDHELD, 2), [RAIL, RAIL]);
        assert_eq!(hid_npad_query(405, PLAYER_1, 1), [USB]);
        assert_eq!(hid_npad_query(406, PLAYER_1, 2), [USB, USB]);
    }

    #[test]
    fn every_pad_has_a_battery_and_only_the_handheld_one_has_two() {
        // HasBattery / HasLeftRightBattery. Both pads run on a battery; the
        // halves are the handheld pad's alone, which is the split
        // `device_type` already draws between `HandheldLeft|Right` and a
        // single `FullKey`.
        const HANDHELD: u32 = 0x20;
        const PLAYER_1: u32 = 0;

        assert_eq!(hid_npad_query(403, HANDHELD, 1), [1]);
        assert_eq!(hid_npad_query(404, HANDHELD, 2), [1, 1]);
        assert_eq!(hid_npad_query(403, PLAYER_1, 1), [1]);
        assert_eq!(hid_npad_query(404, PLAYER_1, 2), [0, 0]);
    }
}
