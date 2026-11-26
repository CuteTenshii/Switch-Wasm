//! `pl:u`: the shared system fonts, built by [`crate::cpu::Cpu::build_shared_fonts`].

use crate::cpu::{Cpu, FontRegion};
use crate::Result;

impl Cpu {
    /// `pl:u` (`IPlatformServiceManager`): every font type is answered.
    pub(crate) fn pl_request(&mut self, tls: u64, cmd_id: Option<u32>) -> Result<()> {
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
            // GetSharedFontInOrderOfPriority (5, and 6 for the system), given a u64
            // LanguageCode -> { u8 Loaded, u8 pad[3], s32 total_fonts }, plus types,
            // offsets and sizes in three output buffers.
            Some(cmd @ (5 | 6)) => {
                let language = self.ipc_arg_u64(tls, 0);
                let (_, recv) = self.ipc_buffers(tls);
                let regions = self.shared_font_regions().to_vec();
                // The smallest of the three buffers bounds the entry count.
                let room = recv.iter().map(|(_, size)| size / 4).min().unwrap_or(0);
                let fonts: Vec<(u32, FontRegion)> = fonts_in_priority(cmd == 6, language)
                    .filter_map(|font_type| {
                        let region = *regions.get(font_type as usize)?;
                        (region.size != 0).then_some((font_type, region))
                    })
                    .take(room as usize)
                    .collect();
                let count = fonts.len() as u32;
                for (i, (font_type, region)) in fonts.into_iter().enumerate() {
                    let values = [font_type, region.offset, region.size];
                    for (buffer, value) in recv.iter().zip(values) {
                        self.mem.write_u32(buffer.0 + i as u64 * 4, value)?;
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

/// The shared font types `GetSharedFontInOrderOfPriority` reports, most preferred
/// first, as 21.2.0's glue orders them for `language` (a NUL-padded ASCII code).
///
/// Each command reports six of the seven types: the application one leaves out
/// `nintendo_ext2` (6) and the system one `nintendo_ext` (5). The 21.2.0 offlineWeb
/// applet sizes its font texture cache per font, and an eighth font overflows its heap.
fn fonts_in_priority(for_system: bool, language: u64) -> impl Iterator<Item = u32> {
    let order: [u32; 7] = match &language.to_le_bytes() {
        b"zh-CN\0\0\0" | b"zh-Hans\0" => [6, 5, 2, 1, 3, 4, 0],
        b"zh-TW\0\0\0" | b"zh-Hant\0" => [6, 5, 3, 2, 1, 4, 0],
        b"ko\0\0\0\0\0\0" => [6, 5, 4, 2, 1, 3, 0],
        _ => [6, 5, 0, 2, 1, 3, 4],
    };
    let left_out = if for_system { 5 } else { 6 };
    order
        .into_iter()
        .filter(move |&font_type| font_type != left_out)
}

#[cfg(test)]
mod tests {
    use crate::cpu::Cpu;
    use crate::kernel::ipc::testing::*;

    #[test]
    fn pl_serves_the_host_font_from_its_shared_memory() {
        // With no firmware fonts, the host font stands in for every type.
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

        // Every reported font's data really is at the offset it was promised at.
        let mut cpu = Cpu::new();
        cpu.set_shared_font(font.clone());
        let (count, fonts) = priority_request(&mut cpu, 5, b"ja", TYPES);
        assert_eq!(count as usize, fonts.len());
        assert!(!fonts.is_empty());
        for (font_type, offset, size) in fonts {
            assert_eq!(size as usize, font.len(), "font {font_type}'s size");
            let image = cpu.shared_font_image();
            let (offset, size) = (offset as usize, size as usize);
            assert_eq!(
                &image[offset..offset + size],
                &font[..],
                "font {font_type}'s data"
            );
        }
    }

    /// Sends GetSharedFontInOrderOfPriority (`cmd`) with three buffers of `room`
    /// entries; returns the reply's font count and the (type, offset, size) entries.
    fn priority_request(
        cpu: &mut Cpu,
        cmd: u32,
        language: &[u8],
        room: u32,
    ) -> (u32, Vec<(u32, u32, u32)>) {
        const BUFFERS: u64 = 0x3000;
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        cpu.mem.map_zero(BUFFERS, 0x100).unwrap();
        cpu.mem.write_u32(TLS, 4 | (3 << 24)).unwrap(); // 3 recv buffers
        cpu.mem.write_u32(TLS + 4, 10).unwrap();
        for i in 0..3u64 {
            let at = TLS + 8 + 12 * i;
            for (word, value) in map_descriptor(BUFFERS + 0x20 * i, 4 * room)
                .into_iter()
                .enumerate()
            {
                cpu.mem.write_u32(at + 4 * word as u64, value).unwrap();
            }
        }
        let data = TLS + u64::from(cpu.ipc_reply_start(TLS));
        cpu.mem.write_u32(data, SFCI).unwrap();
        cpu.mem.write_u32(data + 8, cmd).unwrap();
        let mut code = [0u8; 8];
        code[..language.len()].copy_from_slice(language);
        cpu.mem
            .write_u64(data + 0x10, u64::from_le_bytes(code))
            .unwrap();
        cpu.pl_request(TLS, Some(cmd)).unwrap();
        // { u8 fonts_loaded, u8 pad[3], s32 total_fonts }
        assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 1, "loaded");
        let count = cpu.mem.read_u32(TLS + 0x24).unwrap();
        let entries = (0..u64::from(count.min(room)))
            .map(|i| {
                let word = |buffer: u64| cpu.mem.read_u32(BUFFERS + 0x20 * buffer + 4 * i).unwrap();
                (word(0), word(1), word(2))
            })
            .collect();
        (count, entries)
    }

    fn types(fonts: &[(u32, u32, u32)]) -> Vec<u32> {
        fonts.iter().map(|&(font_type, _, _)| font_type).collect()
    }

    #[test]
    fn each_priority_command_reports_six_of_the_seven_fonts() {
        // Seven distinct fonts, so a type is told apart by the region it names.
        let mut cpu = Cpu::new();
        for i in 0..7u8 {
            let font = vec![i; 16 + usize::from(i) * 4];
            let offset = cpu.pl_shmem_image.len();
            cpu.push_shared_font(
                &crate::cpu::decode_bfttf(&crate::cpu::encode_bfttf(&font)).unwrap(),
            );
            assert_eq!(
                cpu.shared_font_regions[usize::from(i)].offset as usize,
                offset + 8
            );
        }
        let regions = cpu.shared_font_regions.clone();

        for (cmd, left_out) in [(5, 6), (6, 5)] {
            let (count, fonts) = priority_request(&mut cpu, cmd, b"en-US", 7);
            assert_eq!(count, 6, "cmd {cmd} count");
            assert_eq!(fonts.len(), 6);
            let mut seen = types(&fonts);
            assert!(
                !seen.contains(&left_out),
                "cmd {cmd} leaves out type {left_out}"
            );
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(seen.len(), 6, "cmd {cmd} reports each type once");
            for (font_type, offset, size) in fonts {
                let region = regions[font_type as usize];
                assert_eq!(
                    (offset, size),
                    (region.offset, region.size),
                    "type {font_type}"
                );
            }
        }
    }

    #[test]
    fn the_language_puts_its_own_font_first_among_the_scripts() {
        let mut cpu = Cpu::new();
        cpu.set_shared_font(b"font".to_vec());
        let position = |fonts: &[(u32, u32, u32)], font_type: u32| {
            types(fonts).iter().position(|&t| t == font_type).unwrap()
        };
        const STANDARD: u32 = 0;
        const KOREAN: u32 = 4;
        let (_, korean) = priority_request(&mut cpu, 5, b"ko", 7);
        assert!(position(&korean, KOREAN) < position(&korean, STANDARD));
        let (_, english) = priority_request(&mut cpu, 5, b"en-US", 7);
        assert!(position(&english, STANDARD) < position(&english, KOREAN));
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
        const BUFFERS: u64 = 0x3000;
        let mut cpu = Cpu::new();
        cpu.set_shared_font(b"font".to_vec());
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        cpu.mem.map_zero(BUFFERS, 0x100).unwrap();
        cpu.mem.write_u32(TLS, 4 | (3 << 24)).unwrap();
        cpu.mem.write_u32(TLS + 4, 8).unwrap();
        for i in 0..3u64 {
            let at = TLS + 8 + 12 * i;
            // Room for two entries.
            for (word, value) in map_descriptor(BUFFERS + 0x20 * i, 4 * 2)
                .into_iter()
                .enumerate()
            {
                cpu.mem.write_u32(at + 4 * word as u64, value).unwrap();
            }
        }
        let data_area = cpu.ipc_reply_start(TLS);
        cpu.mem.write_u32(TLS + u64::from(data_area), SFCI).unwrap();
        cpu.mem
            .write_u32(TLS + u64::from(data_area) + 8, 5)
            .unwrap();
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
        const BUFFERS: u64 = 0x3000;
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
        for i in 0..3u64 {
            let at = TLS + 8 + 12 * i;
            // Room for eight entries.
            for (word, value) in map_descriptor(BUFFERS + 0x20 * i, 0x20)
                .into_iter()
                .enumerate()
            {
                cpu.mem.write_u32(at + 4 * word as u64, value).unwrap();
            }
        }
        let data_area = cpu.ipc_reply_start(TLS);
        cpu.mem.write_u32(TLS + u64::from(data_area), SFCI).unwrap();
        cpu.mem
            .write_u32(TLS + u64::from(data_area) + 8, 5)
            .unwrap();
        cpu.pl_request(TLS, Some(5)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 1, "loaded");
        assert_eq!(cpu.mem.read_u32(TLS + 0x24).unwrap(), 0, "no fonts");
    }
}
