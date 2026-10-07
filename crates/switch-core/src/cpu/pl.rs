//! `pl:u`: the shared system fonts, built by [`super::Cpu::build_shared_fonts`].

use super::{Cpu, FontRegion};
use crate::Result;

impl Cpu {
    /// `pl:u` (`IPlatformServiceManager`): every font type is answered.
    pub(super) fn pl_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        // Which font a per-type command is asking about.
        let font_type = self.mem.read_u32(self.ipc_request_data(tls)).unwrap_or(0);
        let region = self
            .shared_font_regions()
            .get(font_type as usize)
            .copied()
            .unwrap_or(FontRegion { offset: 0, size: 0 });
        match cmd_id {
            // RequestLoad(u32 SharedFontType)
            Some(0) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // GetLoadState(u32) -> u32 (1 = Loaded)
            Some(1) => self.write_ipc_response(tls, 0, &[], &1u32.to_le_bytes(), &[]),
            // GetSize(u32) -> u32
            Some(2) => self.write_ipc_response(tls, 0, &[], &region.size.to_le_bytes(), &[]),
            // GetSharedMemoryAddressOffset(u32) -> u32
            Some(3) => self.write_ipc_response(tls, 0, &[], &region.offset.to_le_bytes(), &[]),
            // GetSharedMemoryNativeHandle -> a shared memory handle.
            Some(4) => {
                let handle = self.alloc_handle();
                self.write_ipc_response(tls, 0, &[handle], &[], &[])
            }
            // GetSharedFontInOrderOfPriority (5, and 6 for the system) ->
            // { u8 Loaded, u8 pad[3], s32 total_fonts }, plus types, offsets and sizes
            // in three output buffers. Every font is reported in `PlSharedFontType` order.
            Some(5) | Some(6) => {
                let (_, recv) = self.ipc_buffers(tls);
                let regions = self.shared_font_regions().to_vec();
                // The smallest of the three buffers bounds the entry count.
                let room = recv.iter().map(|(_, size)| size / 4).min().unwrap_or(0);
                let count = (regions.len() as u32).min(room);
                for (i, region) in regions.iter().enumerate().take(count as usize) {
                    let values = [i as u32, region.offset, region.size];
                    for (buffer, value) in recv.iter().zip(values) {
                        self.mem.write_u32(buffer.0 + i as u32 * 4, value)?;
                    }
                }
                let mut raw = [0u8; 8];
                raw[0] = 1; // Loaded
                raw[4..].copy_from_slice(&count.to_le_bytes());
                self.write_ipc_response(tls, 0, &[], &raw, &[])
            }
            _ => self.unimplemented_command(tls, "pl:u", cmd_id),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::cpu::ipc::testing::*;
    use crate::cpu::Cpu;

    #[test]
    fn pl_serves_the_host_font_from_its_shared_memory() {
        // With no firmware fonts, the host font stands in for every type.
        const BUFFERS: u32 = 0x3000;
        const TYPES: u32 = 7;
        // 40 bytes, a whole number of words, so no padding is in play.
        let font = b"not really a font, but bytes are bytes!!".to_vec();

        // Each font sits behind an eight-byte header the offsets skip.
        for i in 0..TYPES {
            let mut cpu = request(false, 2, &i.to_le_bytes());
            cpu.set_shared_font(font.clone());
            cpu.pl_request(TLS, Some(2)).unwrap();
            let size = cpu.mem.read_u32(TLS + 0x20).unwrap();
            assert_eq!(size as usize, font.len(), "GetSize for type {i}");

            let mut cpu = request(false, 3, &i.to_le_bytes());
            cpu.set_shared_font(font.clone());
            cpu.pl_request(TLS, Some(3)).unwrap();
            let expected = (font.len() as u32 + 8) * i + 8;
            assert_eq!(
                cpu.mem.read_u32(TLS + 0x20).unwrap(),
                expected,
                "offset for type {i}"
            );
        }

        // GetSharedFontInOrderOfPriority: one request, three out map-aliases.
        let mut cpu = Cpu::new();
        cpu.set_shared_font(font.clone());
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        cpu.mem.map_zero(BUFFERS, 0x100).unwrap();
        cpu.mem.write_u32(TLS, 4 | (3 << 24)).unwrap(); // 3 recv buffers
        cpu.mem.write_u32(TLS + 4, 8).unwrap();
        for i in 0..3u32 {
            let at = TLS + 8 + 12 * i;
            cpu.mem.write_u32(at, 4 * TYPES).unwrap();
            cpu.mem.write_u32(at + 4, BUFFERS + 0x20 * i).unwrap();
        }
        let data_area = cpu.ipc_reply_start(TLS);
        cpu.mem.write_u32(TLS + data_area, SFCI).unwrap();
        cpu.mem.write_u32(TLS + data_area + 8, 5).unwrap();
        cpu.pl_request(TLS, Some(5)).unwrap();
        // { u8 fonts_loaded, u8 pad[3], s32 total_fonts }
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 1);
        assert_eq!(cpu.mem.read_u32(TLS + 0x24).unwrap(), TYPES);
        for i in 0..TYPES {
            assert_eq!(cpu.mem.read_u32(BUFFERS + 4 * i).unwrap(), i, "type {i}");
            let offset = cpu.mem.read_u32(BUFFERS + 0x20 + 4 * i).unwrap() as usize;
            let size = cpu.mem.read_u32(BUFFERS + 0x40 + 4 * i).unwrap() as usize;
            assert_eq!(size, font.len(), "font {i}'s size");
            // Each font's data really is at the offset it was promised at.
            let image = cpu.shared_font_image();
            assert_eq!(&image[offset..offset + size], &font[..], "font {i}'s data");
        }
    }

    #[test]
    fn a_bfttf_round_trips_through_the_header_a_console_stores_it_behind() {
        use crate::cpu::{decode_bfttf, encode_bfttf};
        // Four-byte aligned, so nothing is padded and the comparison is exact.
        let ttf: Vec<u8> = (0..64u8).collect();
        let blob = decode_bfttf(&encode_bfttf(&ttf)).expect("its own encoding decodes");
        assert_eq!(&blob[8..], &ttf[..], "the font follows the header");
        assert_eq!(
            &blob[..4],
            &[0x7f, 0x9a, 0x02, 0x18],
            "and the header opens with the magic a decoded bfttf carries"
        );
    }

    #[test]
    fn a_bfttf_shorter_than_a_word_is_padded_not_trimmed() {
        use crate::cpu::{decode_bfttf, encode_bfttf};
        // Trimming would cut into whatever table the directory says is last.
        let ttf = b"abcde".to_vec();
        let blob = decode_bfttf(&encode_bfttf(&ttf)).unwrap();
        assert_eq!(&blob[8..8 + ttf.len()], &ttf[..]);
        assert_eq!(blob.len(), 8 + 8, "rounded up to a whole number of words");
    }

    #[test]
    fn only_a_real_bfttf_decodes() {
        use crate::cpu::decode_bfttf;
        // A plain TrueType file has no magic and must be refused.
        assert!(decode_bfttf(&[0x00, 0x01, 0x00, 0x00, 0, 0, 0, 0]).is_none());
        assert!(
            decode_bfttf(&[0x36, 0xf8, 0x1a]).is_none(),
            "shorter than a header"
        );
    }

    #[test]
    fn pl_reports_only_as_many_fonts_as_the_caller_left_room_for() {
        // The reply count must match the entries written.
        const BUFFERS: u32 = 0x3000;
        let mut cpu = Cpu::new();
        cpu.set_shared_font(b"font".to_vec());
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        cpu.mem.map_zero(BUFFERS, 0x100).unwrap();
        cpu.mem.write_u32(TLS, 4 | (3 << 24)).unwrap();
        cpu.mem.write_u32(TLS + 4, 8).unwrap();
        for i in 0..3u32 {
            let at = TLS + 8 + 12 * i;
            cpu.mem.write_u32(at, 4 * 2).unwrap(); // room for two entries
            cpu.mem.write_u32(at + 4, BUFFERS + 0x20 * i).unwrap();
        }
        let data_area = cpu.ipc_reply_start(TLS);
        cpu.mem.write_u32(TLS + data_area, SFCI).unwrap();
        cpu.mem.write_u32(TLS + data_area + 8, 5).unwrap();
        cpu.pl_request(TLS, Some(5)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x24).unwrap(), 2);
        assert_eq!(
            cpu.mem.read_u32(BUFFERS + 8).unwrap(),
            0,
            "nothing past the second"
        );
    }

    #[test]
    fn without_a_font_pl_reports_an_empty_set() {
        // A well-formed "no fonts" answer, not a spin in `_plRequestLoadWait`.
        const BUFFERS: u32 = 0x3000;
        let mut cpu = request(false, 1, &[]);
        cpu.pl_request(TLS, Some(1)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x20).unwrap(),
            1,
            "reported as loaded"
        );

        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        cpu.mem.map_zero(BUFFERS, 0x100).unwrap();
        cpu.mem.write_u32(TLS, 4 | (3 << 24)).unwrap();
        cpu.mem.write_u32(TLS + 4, 8).unwrap();
        for i in 0..3u32 {
            let at = TLS + 8 + 12 * i;
            cpu.mem.write_u32(at, 0x20).unwrap(); // room for eight entries
            cpu.mem.write_u32(at + 4, BUFFERS + 0x20 * i).unwrap();
        }
        let data_area = cpu.ipc_reply_start(TLS);
        cpu.mem.write_u32(TLS + data_area, SFCI).unwrap();
        cpu.mem.write_u32(TLS + data_area + 8, 5).unwrap();
        cpu.pl_request(TLS, Some(5)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 1, "loaded");
        assert_eq!(cpu.mem.read_u32(TLS + 0x24).unwrap(), 0, "no fonts");
    }
}
