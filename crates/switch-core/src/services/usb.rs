//! `usb:ds` and `usb:hs`: no cable is attached and no device is plugged in.
//!
//! Command ids are the 11.0.0+ layout, where `usb:ds` hands out `IDsService`.

use crate::cpu::Cpu;
use crate::Result;

/// `UsbState_Detached`.
const STATE_DETACHED: u32 = 0;
/// `UsbDeviceSpeed_None`.
const SPEED_NONE: u32 = 0;

impl Cpu {
    /// `usb:ds`, its `IDsService`, `IDsInterface` and `IDsEndpoint`.
    pub(crate) fn usb_ds_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        if self.ipc_answer_control(tls, handle, "usb:ds", cmd_id)? {
            return Ok(());
        }
        let object = self.ipc_object_key(tls, handle);
        let iface = self.ipc_interface(tls, handle, "usb:ds");
        match iface.as_str() {
            "usb:ds-service" => match cmd_id {
                // Bind / SetUsbDeviceDescriptor / SetBinaryObjectStore / Enable / Disable.
                Some(0) | Some(7) | Some(8) | Some(9) | Some(10) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // RegisterInterface(u8) -> IDsInterface.
                Some(1) => {
                    self.reply_with_interface(tls, handle, "usb:ds-interface")?;
                    Ok(())
                }
                // GetStateChangeEvent: never signalled, the cable stays out.
                Some(2) => {
                    let event = self.kept_event("usb:ds-state", object);
                    self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                }
                Some(3) => self.write_ipc_response(tls, 0, &[], &STATE_DETACHED.to_le_bytes(), &[]),
                // ClearDeviceData.
                Some(4) => {
                    self.usb_ds_strings = 0;
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // AddUsbStringDescriptor -> u8 index, the lowest free slot.
                Some(5) => {
                    let index = self.usb_ds_strings.trailing_ones();
                    if index == u64::BITS {
                        return self.unimplemented_command(tls, &iface, cmd_id);
                    }
                    self.usb_ds_strings |= 1 << index;
                    self.write_ipc_response(tls, 0, &[], &[index as u8], &[])
                }
                // DeleteUsbStringDescriptor(u8).
                Some(6) => {
                    let index = u32::from(self.ipc_arg_u8(tls, 0));
                    if index < u64::BITS {
                        self.usb_ds_strings &= !(1 << index);
                    }
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                Some(11) => self.write_ipc_response(tls, 0, &[], &SPEED_NONE.to_le_bytes(), &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "usb:ds-interface" => match cmd_id {
                // RegisterEndpoint(u8) -> IDsEndpoint.
                Some(0) => {
                    self.reply_with_interface(tls, handle, "usb:ds-endpoint")?;
                    Ok(())
                }
                // GetSetupEvent / GetCtrlInCompletionEvent / GetCtrlOutCompletionEvent.
                Some(1) | Some(5) | Some(7) => {
                    let purpose = match cmd_id {
                        Some(1) => "usb:ds-setup",
                        Some(5) => "usb:ds-ctrl-in",
                        _ => "usb:ds-ctrl-out",
                    };
                    let event = self.kept_event(purpose, object);
                    self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                }
                // AppendConfigurationData.
                Some(10) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // Transfers need a host on the other end.
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "usb:ds-endpoint" => match cmd_id {
                // Cancel: nothing is in flight. SetZlt(bool).
                Some(1) | Some(5) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetCompletionEvent.
                Some(2) => {
                    let event = self.kept_event("usb:ds-completion", object);
                    self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IDsRootSession::OpenDsService -> IDsService.
            _ => match cmd_id {
                Some(0) => {
                    self.reply_with_interface(tls, handle, "usb:ds-service")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
        }
    }

    /// `usb:hs` (`IClientRootSession`): no device is ever plugged in.
    pub(crate) fn usb_hs_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        if self.ipc_answer_control(tls, handle, "usb:hs", cmd_id)? {
            return Ok(());
        }
        match cmd_id {
            // BindClientProcess / DestroyInterfaceAvailableEvent(u8).
            Some(0) | Some(5) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // QueryAllInterfaces / QueryAvailableInterfaces / QueryAcquiredInterfaces -> s32 0.
            Some(1) | Some(2) | Some(3) => {
                self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[])
            }
            // CreateInterfaceAvailableEvent(u8 index, filter): one event per index.
            Some(4) => {
                let index = u64::from(self.ipc_arg_u8(tls, 0));
                let event = self.kept_event("usb:hs-available", (handle << 8) | index);
                self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
            }
            // GetInterfaceStateChangeEvent.
            Some(6) => {
                let event = self.kept_event("usb:hs-state", handle);
                self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
            }
            // AcquireUsbIf: there is no interface to acquire.
            _ => self.unimplemented_command(tls, "usb:hs", cmd_id),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::cpu::Cpu;
    use crate::kernel::ipc::testing::*;

    /// Send `cmd` to `usb:ds` domain object 7, named `iface`, and return the reply's first word.
    fn ds(cpu: &mut Cpu, iface: &str, cmd: u32, payload: &[u8]) -> u32 {
        cpu.record_domain_object(9, 7, iface);
        marshal(cpu, true, cmd, payload);
        if payload.is_empty() {
            cpu.mem.write_u32(TLS + 0x30, 0xffff_ffff).unwrap();
        }
        cpu.usb_ds_request(TLS, 9, Some(cmd)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x28).unwrap(),
            0,
            "{iface} {cmd}: result"
        );
        cpu.mem.read_u32(TLS + 0x30).unwrap()
    }

    /// AddUsbStringDescriptor's index.
    fn add(cpu: &mut Cpu) -> u32 {
        ds(cpu, "usb:ds-service", 5, &[]) & 0xff
    }

    #[test]
    fn the_root_session_opens_a_ds_service() {
        let mut cpu = request(true, 0, &[]);
        let object = ds(&mut cpu, "usb:ds", 0, &[]);
        assert_eq!(cpu.domain_interface(9, object), Some("usb:ds-service"));
    }

    #[test]
    fn the_device_reports_no_cable() {
        let mut cpu = request(true, 3, &[]);
        assert_eq!(
            ds(&mut cpu, "usb:ds-service", 3, &[]),
            super::STATE_DETACHED
        );
        assert_eq!(ds(&mut cpu, "usb:ds-service", 11, &[]), super::SPEED_NONE);
    }

    #[test]
    fn string_descriptors_take_the_lowest_free_index() {
        let mut cpu = request(true, 5, &[]);
        assert_eq!(add(&mut cpu), 0);
        assert_eq!(add(&mut cpu), 1);
        assert_eq!(add(&mut cpu), 2);
        ds(&mut cpu, "usb:ds-service", 6, &[1]);
        assert_eq!(add(&mut cpu), 1, "a deleted index is reused");
        ds(&mut cpu, "usb:ds-service", 4, &[]);
        assert_eq!(add(&mut cpu), 0, "ClearDeviceData frees every index");
    }

    #[test]
    fn the_host_side_lists_no_interfaces() {
        for cmd in [1, 2, 3] {
            let mut cpu = request(false, cmd, &[]);
            cpu.mem.write_u32(TLS + 0x20, 0xffff_ffff).unwrap();
            cpu.usb_hs_request(TLS, 9, Some(cmd)).unwrap();
            assert_eq!(
                cpu.mem.read_u32(TLS + 0x18).unwrap(),
                0,
                "query {cmd}: result"
            );
            assert_eq!(
                cpu.mem.read_u32(TLS + 0x20).unwrap(),
                0,
                "query {cmd}: count"
            );
        }
    }
}
