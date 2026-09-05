//! `nvdrv`, the driver node interface the GPU is reached through.
//!
//! The devices behind it (`/dev/nvmap`, `nvhost-ctrl`, `nvhost-as-gpu`,
//! `nvhost-gpu`) and the ioctls they answer live in [`crate::gpu`]; this is
//! only the session that carries them.

use super::Cpu;
use crate::trace::Level;
use crate::Result;

impl Cpu {
    /// The `INvDrvServices` interface: the guest's door to the GPU.
    ///
    /// Command ids follow libnx's `services/nv.c`: 0 Open, 1 Ioctl, 2 Close,
    /// 3 Initialize, 4 QueryEvent, 8 SetAruid (libnx calls it SetClientPID),
    /// 11 Ioctl2, 12 Ioctl3.
    /// Every one of them answers with a `u32` NvError (Open also returns the
    /// fd), and the ioctl argument struct travels as a map-alias buffer in
    /// each direction.
    pub(super) fn nvdrv_request(
        &mut self,
        tls: u32,
        cmd_id: Option<u32>,
        _handle: u64,
    ) -> Result<()> {
        // Control requests are session management, not the nv interface.
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                // CloneCurrentObject(Ex): libnx clones the nvdrv session and
                // sends SubmitGpfifo/KickoffPb down the clone, so the new
                // handle has to route back to the same driver.
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
            // Ioctl2 adds an inline input buffer between the argument buffers;
            // Ioctl3 adds an extra output buffer after them.
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
                argp.resize(size, 0);
                let mut inline_out = Vec::new();
                let error = self.nv.ioctl(
                    &mut self.mem,
                    fd,
                    request,
                    &mut argp,
                    &inline_in,
                    &mut inline_out,
                )?;
                if error != 0 && crate::trace::enabled(crate::trace::Trace::Nv) {
                    crate::traceln!("[nv] ioctl fd={fd} request={request:#x} -> error {error}");
                }
                // An ioctl the model has no handler for is a gap in the same
                // sense an unimplemented service command is, and it was
                // reaching stderr only, which does not exist in the browser,
                // where the whole GPU stack runs. Reported once per (node,
                // command), because a driver that is refused usually asks
                // again every frame.
                use crate::gpu::nvdrv::{NV_NOT_IMPLEMENTED, NV_NOT_SUPPORTED};
                if matches!(error, NV_NOT_IMPLEMENTED | NV_NOT_SUPPORTED) {
                    let node = self.nv.device_name(fd).to_owned();
                    let nr = request & 0xFF;
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
                // `nvIoctl3`'s second receive buffer: where a caller that
                // asked for its payload out-of-line reads it from. Leaving it
                // untouched is how a retail title ended up with a zeroed GPU
                // characteristics struct.
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
            // Initialize(u32 transfer_mem_size, handles) -> u32 error. libnx
            // ignores the out word, but libtransistor checks the reply's raw
            // size, so omitting it failed sdl-hello's nv init.
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
                // Named by the node it came from, because what the event means
                // is a property of the node and not of nvdrv: a
                // `/dev/nvhost-ctrl` event fires when a syncpoint retires,
                // while a `/dev/nvhost-ctrl-gpu` one fires when the GPU
                // faults. One of those a stalled guest wants signalled and the
                // other it very much does not, and a single `query-event` name
                // for both hid which was which in every trace.
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
                // A syncpoint event stands for work that has already
                // finished. This emulator runs each submission to completion
                // inside the ioctl that carried it, so by the time the guest
                // can ask about the fence, the syncpoint has retired -- hand
                // the event over already signalled, and manual-reset so every
                // poll succeeds rather than only the first.
                //
                // Left dark, these never fired at all, and a guest polling one
                // with a zero timeout got "not yet" forever: the Home Menu
                // asked 22,949 times in two seconds of console time and never
                // dequeued the buffer it was waiting to draw into.
                //
                // The GPU *fault* event is the exception and stays dark and
                // auto-clearing. It does not mean "your work is done", it
                // means the channel died, and a guest told that tears down its
                // renderer.
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
            // SetAruid(u64 AppletResourceUserId) -> u32 error: which
            // applet's nvmap handles and address spaces an fd belongs to.
            // There is one applet here, so the id is recorded and nothing
            // reads it -- but the out word is not optional, and answering
            // with an empty reply is the short-reply bug fixed at Initialize.
            // SetAruidForTest takes and answers the same thing.
            Some(7) | Some(8) => {
                self.nv.applet_resource_user_id = self.mem.read_u64(data).unwrap_or(0);
                self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[])
            }
            // GetStatus -> u32 error. Its name suggests more and it returns
            // only that word, which is the same short-reply trap SetAruid
            // above was in: the catch-all answered it with an empty raw
            // section, and a caller reading the word got whatever the reply's
            // padding held.
            Some(6) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            // DumpGraphicsMemoryInfo: no input, no output. On hardware it
            // writes the driver's memory map to the system log, so a console
            // with no such log does the whole of what it does by returning.
            Some(9) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // SetGraphicsFirmwareMemoryMarginEnabled(u32 enabled) [8.0.0+]:
            // whether the driver holds a slice of video memory back for the
            // graphics firmware. There is no firmware here to hold it back
            // for and no budget to take it from, so the margin is neither
            // kept nor refused -- and unlike its neighbours this one really
            // does answer with a Result and nothing else, which is why it can
            // be spelled out here without changing what it replies.
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
        // `SetAruid` (libnx's `nvSetClientPID`) returns a `u32` NvError like
        // every other nv command. It used to fall through the catch-all and
        // reply with an empty raw section, which read as success only because
        // the reply's padding is zeroed -- a caller that checks the declared
        // size instead, as libtransistor does, sees a short reply.
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
        // The same short reply, from the same catch-all: `GetStatus`'s whole
        // output is one NvError word, and answering it with an empty raw
        // section leaves a caller reading the reply's padding for it.
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
