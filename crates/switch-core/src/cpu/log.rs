//! What the guest says about itself: `lm` (its log stream) and `fatal` (the
//! way it says it is aborting).
//!
//! Neither is answered so much as *listened to*. A title's own diagnostics are
//! often the only account of why it stopped, so `lm` reassembles the packets
//! it is sent and prints them, and `fatal` reports the error the guest was
//! about to die of rather than swallowing it.

use super::Cpu;
use crate::Result;

impl Cpu {
    /// `fatal:u` — a process reporting that it cannot continue.
    ///
    /// Every one of its commands carries the `Result` that caused it, and that
    /// value is the only account a guest ever gives of why it stopped.
    /// Answering the call generically threw it away and left a process that
    /// had *said* what was wrong looking like one that simply went quiet: the
    /// Mii editor gives up here, 135 million instructions in.
    ///
    /// The report is a diagnostic, not a policy. Nothing here reboots into an
    /// error screen, so the call succeeds and the guest carries on into
    /// whatever it does after asking to die.
    pub(super) fn fatal_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        let result = self.mem.read_u32(self.ipc_request_data(tls)).unwrap_or(0);
        let module = result & 0x1FF;
        let description = (result >> 9) & 0x1FFF;
        let trace = self.backtrace(10);
        self.diagnostic(&format!(
            "[fatal] {result:#010x} = {module}-{description:04} (cmd {cmd_id:?}) bt={trace:x?}"
        ));
        self.write_ipc_response(tls, 0, &[], &[], &[])
    }

    /// `lm`: the log manager, which is where a title's own diagnostic output
    /// goes. `nnSdk`'s `NN_LOG` and everything built on it ends up here rather
    /// than at `svcOutputDebugString`, so without this a retail title's
    /// logging is simply thrown away — which is exactly the information that
    /// makes the next failure legible.
    ///
    /// `ILogService::OpenLogger` hands back an `ILogger`, whose `Log` command
    /// carries one **LogPacket** in a send buffer: a 0x18-byte header
    /// (`pid`, `thread id`, `flags`, `severity`, `verbosity`, `payload_size`)
    /// followed by TLV chunks keyed by field. The text of the message is key
    /// 2; the rest are context the guest may or may not attach. A long message
    /// is split across packets, with `flags` bit 0 marking the first and bit 1
    /// the last, so the text is accumulated and only terminated with a newline
    /// on the tail packet.
    pub(super) fn lm_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "lm:service");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, "lm:control", cmd_id),
            };
        }
        let object_id = self.ipc_domain_object_id(tls);
        let iface = if self.ipc_is_domain_request(tls) {
            self.domain_interface(handle, object_id)
                .unwrap_or("lm:service")
                .to_string()
        } else {
            match self.service_name(handle) {
                Some("lm") | None => "lm:service".to_string(),
                Some(name) => name.to_string(),
            }
        };
        match iface.as_str() {
            // ILogService::OpenLogger(pid) -> ILogger.
            "lm:service" => match cmd_id {
                Some(0) => {
                    self.reply_with_interface(tls, handle, "lm:logger")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "lm:logger" => match cmd_id {
                Some(0) => {
                    // Log(buffer). `logSend` marks the buffer AutoSelect, and
                    // this service answers QueryPointerBufferSize with 0, so it
                    // always arrives as a map-alias send buffer rather than a
                    // send-static.
                    if let Some((addr, size)) = self.ipc_send_buffer(tls, 0) {
                        self.absorb_log_packet(addr, size);
                    }
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // SetDestination(u32): which of the console's log sinks to use.
                // There is one sink here and the guest is already using it.
                Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }

    /// Parse one `lm` LogPacket and append its text to the guest's console
    /// output, so a title's own logging lands in the same place its
    /// `svcOutputDebugString` writes do.
    fn absorb_log_packet(&mut self, addr: u32, size: u32) {
        const HEADER_LEN: u32 = 0x18;
        const FLAG_HEAD: u8 = 1 << 0;
        const FLAG_TAIL: u8 = 1 << 1;
        // TLV keys. Only the ones worth putting in front of a human are read;
        // the rest (line number, file, function, drop count, timestamps) are
        // skipped by length like any other chunk.
        const KEY_TEXT: u8 = 2;
        const KEY_MODULE: u8 = 6;
        if size < HEADER_LEN {
            return;
        }
        let flags = self.mem.read_u8(addr.wrapping_add(0x10)).unwrap_or(0);
        let severity = self.mem.read_u8(addr.wrapping_add(0x12)).unwrap_or(0);
        let payload_size = self.mem.read_u32(addr.wrapping_add(0x14)).unwrap_or(0);
        // Trust the smaller of the declared payload and the buffer: a caller
        // that got either wrong should not walk off the end of the mapping.
        let end = payload_size.min(size - HEADER_LEN);

        let mut module = String::new();
        let mut text = String::new();
        let mut off = 0u32;
        while off + 2 <= end {
            let key = self
                .mem
                .read_u8(addr.wrapping_add(HEADER_LEN + off))
                .unwrap_or(0);
            let len = u32::from(
                self.mem
                    .read_u8(addr.wrapping_add(HEADER_LEN + off + 1))
                    .unwrap_or(0),
            );
            off += 2;
            if off + len > end {
                break;
            }
            if key == KEY_TEXT || key == KEY_MODULE {
                let mut chunk = String::with_capacity(len as usize);
                for i in 0..len {
                    match self.mem.read_u8(addr.wrapping_add(HEADER_LEN + off + i)) {
                        Ok(0) => break,
                        Ok(b) => chunk.push(b as char),
                        Err(_) => break,
                    }
                }
                if key == KEY_TEXT {
                    text.push_str(&chunk);
                } else {
                    module = chunk;
                }
            }
            off += len;
        }
        if text.is_empty() {
            return;
        }
        // A message longer than one packet is split, head to tail; only the
        // first carries the prefix and only the last ends the line.
        if flags & FLAG_HEAD != 0 {
            let level = match severity {
                0 => "TRACE",
                1 => "INFO",
                2 => "WARN",
                3 => "ERROR",
                _ => "FATAL",
            };
            let prefix = if module.is_empty() {
                format!("[lm/{level}] ")
            } else {
                format!("[lm/{level}/{module}] ")
            };
            self.out.extend_from_slice(prefix.as_bytes());
        }
        self.out.extend_from_slice(text.as_bytes());
        if flags & FLAG_TAIL != 0 && !self.out.ends_with(b"\n") {
            self.out.push(b'\n');
        }
    }
}
