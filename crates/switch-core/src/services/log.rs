//! What the guest says about itself: `lm` (its log stream) and `fatal` (its abort report).

use crate::cpu::Cpu;
use crate::trace::Level;
use crate::Result;

impl Cpu {
    /// `fatal:u`: report the guest's `Result` as a diagnostic and let it carry on.
    pub(crate) fn fatal_request(&mut self, tls: u64, cmd_id: Option<u32>) -> Result<()> {
        let result = self.mem.read_u32(self.ipc_request_data(tls)).unwrap_or(0);
        let module = result & 0x1FF;
        let description = (result >> 9) & 0x1FFF;
        let trace = self.backtrace(10);
        let report = format!(
            "[fatal] {result:#010x} = {module}-{description:04} (cmd {cmd_id:?}) bt={trace:x?}"
        );
        self.process_mut().guest_fatal = Some(report.clone());
        self.log(Level::Error, &report);
        self.write_ipc_response(tls, 0, &[], &[], &[])
    }

    pub fn guest_fatal(&self) -> Option<&str> {
        self.process().guest_fatal.as_deref()
    }

    /// `lm`, the log manager behind `NN_LOG`. `ILogger::Log` carries one LogPacket:
    /// a 0x18-byte header (`pid`, `thread id`, `flags`, `severity`, `verbosity`,
    /// `payload_size`) then TLV chunks; the text is key 2. `flags` bit 0 marks the
    /// first packet of a message and bit 1 the last.
    pub(crate) fn lm_request(&mut self, tls: u64, handle: u64, cmd_id: Option<u32>) -> Result<()> {
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
                    // Log(AutoSelect buffer): a pointer or a map-alias send buffer.
                    if let Some((addr, size)) = self.ipc_input_buffer(tls, 0) {
                        self.absorb_log_packet(addr, size);
                    }
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // SetDestination(u32).
                Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }

    /// Parse one LogPacket and append its text to the guest's console output.
    fn absorb_log_packet(&mut self, addr: u64, size: u32) {
        const HEADER_LEN: u32 = 0x18;
        const FLAG_HEAD: u8 = 1 << 0;
        const FLAG_TAIL: u8 = 1 << 1;
        // TLV keys; the rest are skipped by length.
        const KEY_TEXT: u8 = 2;
        const KEY_MODULE: u8 = 6;
        if size < HEADER_LEN {
            return;
        }
        let flags = self.mem.read_u8(addr.wrapping_add(0x10)).unwrap_or(0);
        let severity = self.mem.read_u8(addr.wrapping_add(0x12)).unwrap_or(0);
        let payload_size = self.mem.read_u32(addr.wrapping_add(0x14)).unwrap_or(0);
        let end = payload_size.min(size - HEADER_LEN);

        let mut module = String::new();
        let mut text = String::new();
        let mut off = 0u32;
        while off + 2 <= end {
            let key = self
                .mem
                .read_u8(addr.wrapping_add(u64::from(HEADER_LEN + off)))
                .unwrap_or(0);
            let len = u32::from(
                self.mem
                    .read_u8(addr.wrapping_add(u64::from(HEADER_LEN + off + 1)))
                    .unwrap_or(0),
            );
            off += 2;
            if off + len > end {
                break;
            }
            if key == KEY_TEXT || key == KEY_MODULE {
                let mut chunk = String::with_capacity(len as usize);
                for i in 0..len {
                    match self
                        .mem
                        .read_u8(addr.wrapping_add(u64::from(HEADER_LEN + off + i)))
                    {
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

#[cfg(test)]
mod tests {
    use crate::kernel::ipc::testing::*;

    #[test]
    fn a_log_packet_sent_as_a_pointer_buffer_is_printed() {
        const PACKET: u64 = 0x3000;
        let text = b"hello";
        let mut packet = vec![0u8; 0x18];
        packet[0x10] = 0b11; // head and tail
        packet[0x12] = 2; // WARN
        packet[0x14] = 2 + text.len() as u8;
        packet.extend_from_slice(&[2, text.len() as u8]);
        packet.extend_from_slice(text);

        let mut cpu = auto_select_request(0, &[], PACKET, packet.len() as u32, true);
        cpu.mem.map_zero(PACKET, 0x100).unwrap();
        for (offset, &byte) in packet.iter().enumerate() {
            cpu.mem.write_u8(PACKET + offset as u64, byte).unwrap();
        }
        cpu.record_handle(9, "lm:logger");
        cpu.lm_request(TLS, 9, Some(0)).unwrap();
        assert_eq!(cpu.out, b"[lm/WARN] hello\n");
    }
}
