//! `nvdrv`, the session the GPU devices in [`crate::gpu`] are reached through.

use super::Cpu;
use crate::trace::Level;
use crate::Result;

impl Cpu {
    /// `INvDrvServices`; command ids follow libnx's `services/nv.c`.
    pub(super) fn nvdrv_request(
        &mut self,
        tls: u32,
        cmd_id: Option<u32>,
        _handle: u64,
    ) -> Result<()> {
        // Control requests are session management, not the nv interface.
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                // CloneCurrentObject(Ex): the clone routes to the same driver.
                Some(2) | Some(4) => {
                    let clone = self.alloc_handle();
                    self.record_handle(clone, "nvdrv");
                    self.write_ipc_response(tls, 0, &[clone], &[], &[])
                }
                _ => {
                    self.warn_stub("nvdrv:control", cmd_id, "accepted with no reply data");
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
            };
        }
        let data = self.ipc_request_data(tls);
        let (send, recv) = self.ipc_buffers(tls);
        if crate::trace::enabled(crate::trace::Trace::Nv) {
            crate::traceln!(
                "[nv] cmd={:?} send={:x?} recv={:x?} from {:#x?}",
                cmd_id,
                send,
                recv,
                self.backtrace(6)
            );
        }
        match cmd_id {
            // Open(path buffer) -> { u32 fd, u32 error }
            Some(0) => {
                let path = match send.first() {
                    Some(&(addr, size)) => self.read_string(addr, size),
                    None => String::new(),
                };
                let (fd, error) = self.nv.open(&path)?;
                let mut raw = [0u8; 8];
                raw[..4].copy_from_slice(&fd.to_le_bytes());
                raw[4..].copy_from_slice(&error.to_le_bytes());
                self.write_ipc_response(tls, 0, &[], &raw, &[])
            }
            // Ioctl / Ioctl2 / Ioctl3 { u32 fd, u32 request } -> u32 error.
            // Ioctl2 adds an inline input buffer; Ioctl3 an extra output buffer.
            Some(1) | Some(11) | Some(12) => {
                let fd = self.mem.read_u32(data)?;
                let request = self.mem.read_u32(data.wrapping_add(4))?;
                let inline_in: Vec<u8> = match cmd_id {
                    Some(11) => match send.get(1) {
                        Some(&(addr, size)) => self.read_bytes(addr, size),
                        None => Vec::new(),
                    },
                    _ => Vec::new(),
                };
                let size = crate::gpu::nvdrv::ioctl_size(request) as usize;
                let mut argp = match send.first() {
                    Some(&(addr, len)) if len > 0 => self.read_bytes(addr, len),
                    _ => Vec::new(),
                };
                // Pass the whole buffer: `SUBMIT` sends payloads past its declared size.
                argp.resize(size.max(argp.len()), 0);
                let mut inline_out = Vec::new();
                let error = self.nv.ioctl(
                    &mut self.mem,
                    fd,
                    request,
                    &mut argp,
                    &inline_in,
                    &mut inline_out,
                )?;
                if error != 0 {
                    if crate::trace::enabled(crate::trace::Trace::Nv) {
                        crate::traceln!("[nv] ioctl fd={fd} request={request:#x} -> error {error}");
                    }
                    // An unset config variable is an ordinary answer, not a failure.
                    if error != crate::gpu::nvdrv::NV_CONFIG_VAR_NOT_FOUND {
                        let node = self.nv.device_name(fd).to_owned();
                        self.count_nv_error(&node, request, error);
                    }
                }
                // Report unhandled ioctls once per (node, command).
                use crate::gpu::nvdrv::{NV_NOT_IMPLEMENTED, NV_NOT_SUPPORTED};
                if matches!(error, NV_NOT_IMPLEMENTED | NV_NOT_SUPPORTED) {
                    let node = self.nv.device_name(fd).to_owned();
                    let nr = request & 0xFF;
                    self.count_gap(super::GapKind::Ioctl, &node, Some(nr));
                    if self.unimplemented_ipc.insert((node.clone(), Some(nr))) {
                        let ioc_type = (request >> 8) & 0xFF;
                        let pc = self.pc;
                        self.diagnostic(
                            Level::Warn,
                            &format!(
                                "[nv] unimplemented: {node} ioctl type={ioc_type:#04x} \
                             nr={nr:#04x} ({size} bytes, pc={pc:#x})"
                            ),
                        );
                    }
                }
                if let Some(&(addr, len)) = recv.first() {
                    for (i, &byte) in argp.iter().take(len as usize).enumerate() {
                        self.mem.write_u8(addr.wrapping_add(i as u32), byte)?;
                    }
                }
                // `nvIoctl3`'s second receive buffer, for out-of-line payloads.
                if let Some(&(addr, len)) = recv.get(1) {
                    for (i, &byte) in inline_out.iter().take(len as usize).enumerate() {
                        self.mem.write_u8(addr.wrapping_add(i as u32), byte)?;
                    }
                }
                self.write_ipc_response(tls, 0, &[], &error.to_le_bytes(), &[])
            }
            // Close(u32 fd) -> u32 error
            Some(2) => {
                let fd = self.mem.read_u32(data)?;
                let error = self.nv.close(fd);
                self.write_ipc_response(tls, 0, &[], &error.to_le_bytes(), &[])
            }
            // Initialize(u32 transfer_mem_size, handles) -> u32 error.
            Some(3) => {
                self.nv.transfer_mem_size = self.mem.read_u32(data).unwrap_or(0);
                self.nv.initialized = true;
                self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[])
            }
            // QueryEvent(u32 fd, u32 event_id) -> u32 error + a copy handle
            Some(4) => {
                let fd = self.mem.read_u32(data)?;
                let event_id = self.mem.read_u32(data.wrapping_add(4))?;
                let error = self.nv.query_event(fd, event_id);
                // Named by node: a ctrl event is a syncpoint, a ctrl-gpu event a fault.
                let node = match self.nv.file(fd) {
                    Some(crate::gpu::nvdrv::NvFile::NvHostCtrl) => "nvdrv:nvhost-ctrl",
                    Some(crate::gpu::nvdrv::NvFile::NvHostCtrlGpu) => "nvdrv:nvhost-ctrl-gpu",
                    Some(crate::gpu::nvdrv::NvFile::Channel { .. }) => "nvdrv:nvhost-gpu",
                    Some(crate::gpu::nvdrv::NvFile::AddressSpace { .. }) => "nvdrv:nvhost-as-gpu",
                    Some(crate::gpu::nvdrv::NvFile::NvMap) => "nvdrv:nvmap",
                    _ => "nvdrv:unknown-node",
                };
                if crate::trace::enabled(crate::trace::Trace::Nv) {
                    crate::traceln!("[nv] QueryEvent fd={fd} event={event_id} -> {node}");
                }
                // Syncpoint events are pre-signalled and manual-reset, since submissions run
                // to completion inside the ioctl. The GPU fault event stays dark.
                let fault = matches!(
                    self.nv.file(fd),
                    Some(crate::gpu::nvdrv::NvFile::NvHostCtrlGpu)
                );
                let handle = self.alloc_event(node, fault);
                if !fault {
                    self.signal_event(handle);
                }
                self.write_ipc_reply(tls, 0, &[handle], &[], &error.to_le_bytes(), &[])
            }
            // SetAruid / SetAruidForTest (u64 AppletResourceUserId) -> u32 error.
            Some(7) | Some(8) => {
                self.nv.applet_resource_user_id = self.mem.read_u64(data).unwrap_or(0);
                self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[])
            }
            // GetStatus -> u32 error.
            Some(6) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            // DumpGraphicsMemoryInfo: no input, no output.
            Some(9) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // SetGraphicsFirmwareMemoryMarginEnabled(u32 enabled) -> Result.
            Some(13) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // Everything else: acknowledge with no out data.
            _ => {
                self.warn_stub("nvdrv", cmd_id, "accepted with no reply data");
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::cpu::ipc::testing::*;
    use crate::cpu::Cpu;

    #[test]
    fn set_aruid_answers_with_the_error_word_its_callers_read() {
        // `SetAruid` must reply with a `u32` NvError, not an empty raw section.
        let aruid = 0x0123_4567_89ab_cdefu64;
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        marshal(&mut cpu, false, 8, &aruid.to_le_bytes());
        cpu.nvdrv_request(TLS, Some(8), 9).unwrap();

        // 4 words of SFCO header, one of out data, four of padding.
        assert_eq!(cpu.mem.read_u32(TLS + 4).unwrap(), 9);
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "NvError_Success");
        assert_eq!(cpu.nv.applet_resource_user_id, aruid);
    }

    #[test]
    fn get_status_answers_with_the_error_word_as_well() {
        // `GetStatus` must reply with its NvError word.
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        marshal(&mut cpu, false, 6, &[]);
        cpu.nvdrv_request(TLS, Some(6), 9).unwrap();

        assert_eq!(
            cpu.mem.read_u32(TLS + 4).unwrap(),
            9,
            "one word of out data"
        );
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "NvError_Success");
    }
}
