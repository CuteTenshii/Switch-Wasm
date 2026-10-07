//! `hid`: controller negotiation and the shared memory input state is published in
//! ([`crate::cpu::hid_shmem`]).

use crate::cpu::Cpu;
use crate::Result;

impl Cpu {
    /// Which `hid` interface a session stands for; `hid:dbg` is `IHidServer`, `hid:sys` is not.
    fn hid_interface_for(name: Option<&str>) -> String {
        match name {
            Some("hid") | Some("hid:dbg") | None => "hid:server".to_string(),
            Some(name) => name.to_string(),
        }
    }

    pub(crate) fn hid_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
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
                // Setters with no effect: shared memory always publishes the same connected pads.
                Some(1) | Some(11) | Some(21) | Some(31) | Some(66) | Some(67) | Some(91)
                | Some(103) | Some(104) | Some(107) | Some(109) | Some(122..=125) | Some(128)
                | Some(1000..=1004) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // SetGestureOutputRanges(u32 width, u32 height, u64 aruid): no gestures are synthesised.
                Some(92) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // SetSupportedNpadStyleSet and its readback; the set also picks how the pad is published.
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
                // SetSupportedNpadIdType: there is one pad regardless of the slots asked for.
                Some(102) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // AcquireNpadStyleSetUpdateEventHandle: one event, starting signalled since the pad is already published.
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
                // GetPlayerLedPattern: player 1.
                Some(108) => self.write_ipc_response(tls, 0, &[], &1u64.to_le_bytes(), &[]),
                // Set/GetNpadJoyHoldType(aruid, u64).
                Some(120) => {
                    self.npad_joy_hold_type = self.mem.read_u64(data.wrapping_add(8))?;
                    // `GetNpadJoyHoldType` reads shared memory, so keep it in sync.
                    self.write_npad_condition();
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                Some(121) => {
                    let hold = self.npad_joy_hold_type;
                    self.write_ipc_response(tls, 0, &[], &hold.to_le_bytes(), &[])
                }
                // GetNpadHandheldActivationMode.
                Some(129) => self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[]),
                // Vibration: the low and high band amplitudes map to the Gamepad API's dual-rumble.
                // GetVibrationDeviceInfo -> { device_type, position }: an LRA (1) on the left (0).
                Some(200) => {
                    let mut info = Vec::with_capacity(8);
                    info.extend_from_slice(&1u32.to_le_bytes());
                    info.extend_from_slice(&0u32.to_le_bytes());
                    self.write_ipc_response(tls, 0, &[], &info, &[])
                }
                // SendVibrationValue(handle, HidVibrationValue, aruid): amplitudes at +4 and +0xc.
                Some(201) => {
                    let low = f32::from_bits(self.mem.read_u32(data.wrapping_add(4))?);
                    let high = f32::from_bits(self.mem.read_u32(data.wrapping_add(0xc))?);
                    self.set_vibration(low, high);
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetActualVibrationValue.
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
                // IsVibrationPermitted / IsVibrationDeviceMounted.
                Some(205) | Some(211) => {
                    self.write_ipc_response(tls, 0, &[], &1u8.to_le_bytes(), &[])
                }
                // SendVibrationValues(handles[], values[]): only the first value is kept.
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
                // HasBattery, HasLeftRightBattery, GetNpadInterfaceType, GetNpadLeftRightInterfaceType:
                // the handheld pad is on the rails, the Pro Controller in slot 0 is wired (USB).
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
                        Some(406) if handheld => &[RAIL, RAIL],
                        Some(406) => &[USB, USB],
                        _ if handheld => &[RAIL],
                        _ => &[USB],
                    };
                    self.write_ipc_response(tls, 0, &[], reply, &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // `IHidSystemServer`; command ids from libnx `hidsys.c`.
            "hid:sys" => match cmd_id {
                // Acquire{Home,Sleep,Capture}ButtonEventHandle: never signalled.
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
                // AcquireJoyDetachOnBluetoothOffEventHandle: never signalled.
                Some(751) => {
                    let event = self.alloc_event("hid:sys-joy-detach", true);
                    self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                }
                // AcquireConnectionTriggerTimeoutEvent / AcquireDeviceRegisteredEventForControllerSupport: never signalled.
                Some(544) | Some(546) => {
                    let name = if cmd_id == Some(544) {
                        "hid:sys-connection-trigger-timeout"
                    } else {
                        "hid:sys-device-registered"
                    };
                    let event = self.alloc_event(name, true);
                    self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                }
                // Activate{Home,Sleep,Capture}Button and EnableAppletToGetInput: no state to set.
                Some(111) | Some(131) | Some(151) | Some(301) | Some(304) | Some(305)
                | Some(503) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // ApplyNpadSystemCommonPolicy(Full): how system applets ask for controllers; grants every publishable style.
                Some(303) | Some(308) => {
                    let styles = crate::cpu::supported_npad_style_set();
                    if styles != self.npad_style_set {
                        self.npad_style_set = styles;
                        if let Some(event) = self.npad_style_update_event {
                            self.signal_event(event);
                        }
                    }
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetMaskedSupportedNpadStyleSet: every style the pad can be published in.
                Some(310) => {
                    let styles = crate::cpu::supported_npad_style_set();
                    self.write_ipc_response(tls, 0, &[], &styles.to_le_bytes(), &[])
                }
                // GetNpadCaptureButtonAssignment: no capture buttons assigned.
                Some(313) => self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[]),
                // SetNpadSystemExtStateEnabled: every slot already carries SystemExt.
                Some(322) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // Firmware update setup and teardown: no pad firmware to flash.
                Some(1000) | Some(1120) | Some(1131) | Some(1135) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // CheckUsbFirmwareUpdateRequired: no.
                Some(1132) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                // IsJoyConRailEnabled / IsJoyConAttachedOnAllRail: true, matching handheld mode.
                Some(523) | Some(525) => self.write_ipc_response(tls, 0, &[], &[1u8], &[]),
                // IsUsbFullKeyControllerEnabled: no.
                Some(850) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                // SetTouchScreenMagnification / SetTouchScreenDefaultConfiguration / SetForceHandheldStyleVibration.
                Some(1150) | Some(1152) | Some(1155) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetTouchScreenDefaultConfiguration: mode 0 is `UseSystemSetting`.
                Some(1153) => self.write_ipc_response(tls, 0, &[], &[0u8; 0x10], &[]),
                // GetLastActiveNpad: always npad 0.
                Some(306) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
                // GetNpadFullKeyGripColor: black.
                Some(309) => self.write_ipc_response(tls, 0, &[], &[0u8; 8], &[]),
                // GetUniquePadIds: none.
                Some(703) => self.write_ipc_response(tls, 0, &[], &0i64.to_le_bytes(), &[]),
                // SetNotificationLedPattern and its timeout form.
                Some(830) | Some(831) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IActiveVibrationDeviceList::InitializeVibrationDevice.
            "hid:vibration-devices" => match cmd_id {
                Some(0) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IAppletResource: hands over the input shared memory.
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
    use crate::kernel::ipc::testing::*;

    #[test]
    fn the_gesture_pair_is_accepted_rather_than_refused() {
        // Both void; neither may reach `unimplemented_command`.
        let mut payload = Vec::new();
        let (width, height) = crate::cpu::OperationMode::Handheld.display_size();
        payload.extend_from_slice(&width.to_le_bytes());
        payload.extend_from_slice(&height.to_le_bytes());
        payload.extend_from_slice(&1u64.to_le_bytes());

        for command in [92u32, 91] {
            let mut cpu = request(false, command, &payload);
            cpu.register_service_handle(9, "hid");
            cpu.hid_request(TLS, 9, Some(command)).unwrap();
            assert_eq!(
                cpu.mem.read_u32(TLS + 0x18).unwrap(),
                0,
                "command {command}"
            );
        }
    }

    #[test]
    fn the_pad_is_published_in_a_style_the_title_actually_asked_for() {
        // A title that supports Joy-Con pairs but not Pro Controllers must find a style it asked for.
        use crate::cpu::hid_shmem as h;
        const SHMEM: u32 = 0x3000_0000;

        let mut cpu = request(false, 100, &h::STYLE_JOY_DUAL.to_le_bytes());
        cpu.mem.map_zero(SHMEM, 0x40000).unwrap();
        cpu.hid_shmem_addr = SHMEM;
        cpu.register_service_handle(9, "hid");
        cpu.hid_request(TLS, 9, Some(100)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0);

        cpu.set_gamepad_state(0x3, 0, 0, 0, 0);

        // Player 1 is a dual pair, with its state in the joy_dual LIFO.
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
        // Nothing in the Pro Controller's LIFO.
        let full_key = slot + h::FULL_KEY_LIFO + h::LIFO_STORAGE;
        assert_eq!(cpu.mem.read_u64(full_key + h::STATE_BUTTONS).unwrap(), 0);

        // The handheld slot is still published.
        let handheld = SHMEM + h::NPAD + h::HANDHELD_SLOT * h::ENTRY_SIZE;
        assert_eq!(
            cpu.mem.read_u32(handheld + h::STYLE_SET).unwrap(),
            h::STYLE_HANDHELD
        );
    }

    #[test]
    fn every_slot_carries_the_system_ext_lifo_the_home_menu_reads() {
        // SystemExt is a second copy every pad carries, whatever style was asked for.
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

        // The style tag names only the physical controller.
        assert_eq!(
            cpu.mem.read_u32(SHMEM + h::NPAD + h::STYLE_SET).unwrap(),
            h::STYLE_JOY_DUAL
        );
    }

    #[test]
    fn apply_npad_system_common_policy_is_how_an_applet_asks_for_controllers() {
        // System applets configure input with this instead of `SetSupportedNpadStyleSet`.
        let mut cpu = request(false, 303, &[]);
        cpu.register_service_handle(9, "hid:sys");
        cpu.hid_request(TLS, 9, Some(303)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0);
        assert_eq!(
            cpu.npad_style_set,
            crate::cpu::supported_npad_style_set(),
            "the policy grants every style this console can publish"
        );

        // 310 must agree with 303.
        marshal(&mut cpu, false, 310, &[]);
        cpu.hid_request(TLS, 9, Some(310)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x20).unwrap(),
            crate::cpu::supported_npad_style_set()
        );
    }

    #[test]
    fn the_npad_condition_is_published_so_the_hold_type_can_be_read_at_all() {
        // `GetNpadJoyHoldType` reads `NpadCondition` from shared memory and aborts unless it is valid.
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

        // The hold type is at +8 of the request, and shared memory must follow it.
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
        // A style set of zero (libnx defaults) means a Pro Controller in slot 0 and a handheld in slot 8.
        use crate::cpu::hid_shmem as h;
        assert_eq!(
            crate::cpu::npad_presentation_for(0).style,
            h::STYLE_FULL_KEY
        );
        // Only unsupported styles still resolves to something.
        assert_eq!(
            crate::cpu::npad_presentation_for(1 << 10).style,
            h::STYLE_FULL_KEY
        );
        // Best first: a title taking both gets the Pro Controller.
        assert_eq!(
            crate::cpu::npad_presentation_for(h::STYLE_FULL_KEY | h::STYLE_JOY_DUAL).style,
            h::STYLE_FULL_KEY
        );
        assert_eq!(
            crate::cpu::npad_presentation_for(h::STYLE_JOY_RIGHT).style,
            h::STYLE_JOY_RIGHT
        );
    }

    /// Run one `hid` command with a single `u32 npad_id` and return its reply bytes.
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
        // The two published pads are attached differently.
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
        // Both pads have a battery; only the handheld pad's is split.
        const HANDHELD: u32 = 0x20;
        const PLAYER_1: u32 = 0;

        assert_eq!(hid_npad_query(403, HANDHELD, 1), [1]);
        assert_eq!(hid_npad_query(404, HANDHELD, 2), [1, 1]);
        assert_eq!(hid_npad_query(403, PLAYER_1, 1), [1]);
        assert_eq!(hid_npad_query(404, PLAYER_1, 2), [0, 0]);
    }
}
