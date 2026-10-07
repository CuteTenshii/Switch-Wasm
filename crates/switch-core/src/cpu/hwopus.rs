//! `hwopus`, the console's Opus decoder service.
//!
//! Packets carry an eight-byte big-endian `{u32 size, u32 final_range}` header.

use super::Cpu;
use crate::opus;
use crate::Result;

/// Description 1001: unsupported sample rate.
const OPUS_INVALID_SAMPLE_RATE: u32 = 111 | (1001 << 9);

/// Description 1002: unsupported channel count.
const OPUS_INVALID_CHANNEL_COUNT: u32 = 111 | (1002 << 9);

/// Description 8: input shorter than the header.
const OPUS_INPUT_TOO_SMALL: u32 = 111 | (8 << 9);

/// Description 3: input shorter than the header says.
const OPUS_BUFFER_TOO_SMALL: u32 = 111 | (3 << 9);

/// Description 17: packet is not decodable Opus.
const OPUS_INVALID_PACKET: u32 = 111 | (17 << 9);

const PACKET_HEADER_LEN: u32 = 8;

/// Per-channel-count decoder object size, rounded up from the reference library.
const DECODER_STATE_SIZE: [u32; 2] = [0x4A00, 0x6C00];

const MAX_STREAMS: u32 = 255;

struct MultiStreamParams {
    sample_rate: u32,
    channels: u32,
    total_streams: u32,
    stereo_streams: u32,
    large_frame: bool,
    /// Stream and half each output channel plays.
    mapping: Vec<u8>,
}

pub(crate) struct HwOpus {
    decoder: Decoder,
    channels: usize,
    /// Most samples per channel one packet can produce.
    max_frame: usize,
}

enum Decoder {
    Single(Box<opus::Decoder>),
    Multi(opus::MultiStreamDecoder),
}

impl core::fmt::Debug for HwOpus {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let kind = match self.decoder {
            Decoder::Single(_) => "single",
            Decoder::Multi(_) => "multistream",
        };
        write!(
            f,
            "HwOpus({kind}, {} channels, max {} samples)",
            self.channels, self.max_frame
        )
    }
}

fn align_up(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

fn valid_sample_rate(rate: u32) -> bool {
    matches!(rate, 8000 | 12000 | 16000 | 24000 | 48000)
}

fn work_buffer_size(
    sample_rate: u32,
    channels: u32,
    large_frame: bool,
) -> core::result::Result<u32, u32> {
    if !matches!(channels, 1 | 2) {
        return Err(OPUS_INVALID_CHANNEL_COUNT);
    }
    if !valid_sample_rate(sample_rate) {
        return Err(OPUS_INVALID_SAMPLE_RATE);
    }
    let frame = if large_frame { 5760 } else { 1920 };
    let scratch = align_up((frame * channels) / (48000 / sample_rate), 64);
    Ok(DECODER_STATE_SIZE[channels as usize - 1] + scratch + 0x600)
}

fn work_buffer_size_multistream(
    sample_rate: u32,
    channels: u32,
    total_streams: u32,
    stereo_streams: u32,
    large_frame: bool,
) -> core::result::Result<u32, u32> {
    if channels == 0 || channels > MAX_STREAMS {
        return Err(OPUS_INVALID_CHANNEL_COUNT);
    }
    if !valid_sample_rate(sample_rate) {
        return Err(OPUS_INVALID_SAMPLE_RATE);
    }
    // The console reports a bad stream count as a sample-rate error.
    if total_streams == 0
        || stereo_streams > total_streams
        || total_streams + stereo_streams > channels
    {
        return Err(OPUS_INVALID_SAMPLE_RATE);
    }
    let mono_streams = total_streams - stereo_streams;
    let base =
        0x100 + stereo_streams * DECODER_STATE_SIZE[1] + mono_streams * DECODER_STATE_SIZE[0];
    let frame = if large_frame { 5760 } else { 1920 };
    let scratch = align_up(1500 * total_streams, 64)
        + align_up((frame * channels) / (48000 / sample_rate), 64);
    Ok(base + scratch)
}

impl Cpu {
    pub(super) fn hwopus_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        if self.ipc_answer_control(tls, handle, "hwopus", cmd_id)? {
            return Ok(());
        }
        let iface = self.ipc_interface(tls, handle, "hwopus");
        if iface == "hwopus:decoder" {
            return self.hwopus_decoder_request(tls, handle, cmd_id);
        }

        let data = self.ipc_request_data(tls);
        match cmd_id {
            // OpenHardwareOpusDecoder
            Some(0) => {
                let sample_rate = self.mem.read_u32(data).unwrap_or(0);
                let channels = self.mem.read_u32(data.wrapping_add(4)).unwrap_or(0);
                self.hwopus_open(tls, handle, sample_rate, channels, false)
            }
            // GetWorkBufferSize
            Some(1) => {
                let sample_rate = self.mem.read_u32(data).unwrap_or(0);
                let channels = self.mem.read_u32(data.wrapping_add(4)).unwrap_or(0);
                self.hwopus_reply_size(tls, work_buffer_size(sample_rate, channels, false))
            }
            // OpenHardwareOpusDecoderForMultiStream
            Some(2) => {
                let params = self.hwopus_multistream_params(tls, false);
                self.hwopus_open_multistream(tls, handle, params)
            }
            // GetWorkBufferSizeForMultiStream
            Some(3) => {
                let p = self.hwopus_multistream_params(tls, false);
                let size = work_buffer_size_multistream(
                    p.sample_rate,
                    p.channels,
                    p.total_streams,
                    p.stereo_streams,
                    p.large_frame,
                );
                self.hwopus_reply_size(tls, size)
            }
            // OpenHardwareOpusDecoderEx
            Some(4) => {
                let sample_rate = self.mem.read_u32(data).unwrap_or(0);
                let channels = self.mem.read_u32(data.wrapping_add(4)).unwrap_or(0);
                let large = self.mem.read_u8(data.wrapping_add(8)).unwrap_or(0) != 0;
                self.hwopus_open(tls, handle, sample_rate, channels, large)
            }
            // GetWorkBufferSizeEx / GetWorkBufferSizeExEx
            Some(5) | Some(8) => {
                let sample_rate = self.mem.read_u32(data).unwrap_or(0);
                let channels = self.mem.read_u32(data.wrapping_add(4)).unwrap_or(0);
                let large = self.mem.read_u8(data.wrapping_add(8)).unwrap_or(0) != 0;
                self.hwopus_reply_size(tls, work_buffer_size(sample_rate, channels, large))
            }
            // OpenHardwareOpusDecoderForMultiStreamEx
            Some(6) => {
                let params = self.hwopus_multistream_params(tls, true);
                self.hwopus_open_multistream(tls, handle, params)
            }
            // GetWorkBufferSizeForMultiStreamEx / ExEx
            Some(7) | Some(9) => {
                let p = self.hwopus_multistream_params(tls, true);
                let size = work_buffer_size_multistream(
                    p.sample_rate,
                    p.channels,
                    p.total_streams,
                    p.stereo_streams,
                    p.large_frame,
                );
                self.hwopus_reply_size(tls, size)
            }
            _ => self.unimplemented_command(tls, "hwopus", cmd_id),
        }
    }

    fn hwopus_reply_size(&mut self, tls: u32, size: core::result::Result<u32, u32>) -> Result<()> {
        match size {
            Ok(size) => self.write_ipc_response(tls, 0, &[], &size.to_le_bytes(), &[]),
            Err(error) => self.write_ipc_response(tls, error, &[], &0u32.to_le_bytes(), &[]),
        }
    }

    fn hwopus_open(
        &mut self,
        tls: u32,
        handle: u64,
        sample_rate: u32,
        channels: u32,
        large: bool,
    ) -> Result<()> {
        if let Err(error) = work_buffer_size(sample_rate, channels, large) {
            return self.write_ipc_response(tls, error, &[], &[], &[]);
        }
        let Ok(decoder) = opus::Decoder::new(sample_rate, channels as usize) else {
            return self.write_ipc_response(tls, OPUS_INVALID_SAMPLE_RATE, &[], &[], &[]);
        };
        let key = self.reply_with_interface(tls, handle, "hwopus:decoder")?;
        self.opus_decoders.insert(
            key,
            HwOpus {
                decoder: Decoder::Single(Box::new(decoder)),
                channels: channels as usize,
                max_frame: max_frame(sample_rate, large),
            },
        );
        Ok(())
    }

    fn hwopus_multistream_params(&self, tls: u32, extended: bool) -> MultiStreamParams {
        let Some((addr, size)) = self.ipc_input_buffer(tls, 0) else {
            return MultiStreamParams {
                sample_rate: 0,
                channels: 0,
                total_streams: 0,
                stereo_streams: 0,
                large_frame: false,
                mapping: Vec::new(),
            };
        };
        let read = |offset: u32| self.mem.read_u32(addr.wrapping_add(offset)).unwrap_or(0);
        let channels = read(4);
        let (large_frame, mapping_at) = if extended {
            (
                self.mem.read_u8(addr.wrapping_add(16)).unwrap_or(0) != 0,
                0x18,
            )
        } else {
            (false, 0x10)
        };
        let mapping = if mapping_at < size {
            self.read_bytes(
                addr.wrapping_add(mapping_at),
                channels.min(size - mapping_at),
            )
        } else {
            Vec::new()
        };
        MultiStreamParams {
            sample_rate: read(0),
            channels,
            total_streams: read(8),
            stereo_streams: read(12),
            large_frame,
            mapping,
        }
    }

    fn hwopus_open_multistream(
        &mut self,
        tls: u32,
        handle: u64,
        params: MultiStreamParams,
    ) -> Result<()> {
        let sizing = work_buffer_size_multistream(
            params.sample_rate,
            params.channels,
            params.total_streams,
            params.stereo_streams,
            params.large_frame,
        );
        if let Err(error) = sizing {
            return self.write_ipc_response(tls, error, &[], &[], &[]);
        }
        if params.mapping.len() < params.channels as usize {
            return self.write_ipc_response(tls, OPUS_INVALID_CHANNEL_COUNT, &[], &[], &[]);
        }
        let decoder = opus::MultiStreamDecoder::new(
            params.sample_rate,
            params.channels as usize,
            params.total_streams as usize,
            params.stereo_streams as usize,
            &params.mapping,
        );
        let Ok(decoder) = decoder else {
            return self.write_ipc_response(tls, OPUS_INVALID_CHANNEL_COUNT, &[], &[], &[]);
        };
        let key = self.reply_with_interface(tls, handle, "hwopus:decoder")?;
        self.opus_decoders.insert(
            key,
            HwOpus {
                decoder: Decoder::Multi(decoder),
                channels: params.channels as usize,
                max_frame: max_frame(params.sample_rate, params.large_frame),
            },
        );
        Ok(())
    }

    fn hwopus_decoder_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const WITH_PERF: [bool; 10] = [
            false, false, false, false, true, true, true, true, true, true,
        ];
        const WITH_RESET: [bool; 10] = [
            false, false, false, false, false, false, true, true, true, true,
        ];

        let key = self.ipc_object_key(tls, handle);
        match cmd_id {
            // SetContext / SetContextForMultiStream: the hardware context is ignored.
            Some(1) | Some(3) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            Some(cmd @ 0..=9) => {
                let index = cmd as usize;
                let reset = WITH_RESET[index] && self.ipc_arg_u8(tls, 0) != 0;
                self.hwopus_decode(tls, key, reset, WITH_PERF[index])
            }
            _ => self.unimplemented_command(tls, "hwopus:decoder", cmd_id),
        }
    }

    fn hwopus_decode(&mut self, tls: u32, key: u64, reset: bool, with_perf: bool) -> Result<()> {
        let Some((input, input_len)) = self.ipc_input_buffer(tls, 0) else {
            return self.write_ipc_response(tls, OPUS_INPUT_TOO_SMALL, &[], &[], &[]);
        };
        if input_len <= PACKET_HEADER_LEN {
            return self.write_ipc_response(tls, OPUS_INPUT_TOO_SMALL, &[], &[], &[]);
        }
        // The header is big-endian.
        let size = self.read_bytes(input, 4);
        let size = u32::from_be_bytes([size[0], size[1], size[2], size[3]]);
        if size == 0 || size > input_len - PACKET_HEADER_LEN {
            return self.write_ipc_response(tls, OPUS_BUFFER_TOO_SMALL, &[], &[], &[]);
        }
        let packet = self.read_bytes(input.wrapping_add(PACKET_HEADER_LEN), size);
        let output_room = self
            .ipc_output_buffer(tls, 0)
            .map_or(0, |(_, size)| size as usize);

        let Some(decoder) = self.opus_decoders.get_mut(&key) else {
            return self.write_ipc_response(tls, OPUS_INVALID_PACKET, &[], &[], &[]);
        };
        if reset {
            match &mut decoder.decoder {
                Decoder::Single(decoder) => decoder.reset(),
                Decoder::Multi(decoder) => decoder.reset(),
            }
        }
        let channels = decoder.channels;
        let frame = decoder.max_frame.min(output_room / (2 * channels));
        let mut pcm = vec![0i16; frame * channels];
        let decoded = match &mut decoder.decoder {
            Decoder::Single(decoder) => decoder.decode(Some(&packet), &mut pcm, frame),
            Decoder::Multi(decoder) => decoder.decode(Some(&packet), &mut pcm, frame),
        };
        let samples = match decoded {
            Ok(samples) => samples,
            Err(opus::Error::BufferTooSmall) => {
                return self.write_ipc_response(tls, OPUS_BUFFER_TOO_SMALL, &[], &[], &[]);
            }
            Err(_) => return self.write_ipc_response(tls, OPUS_INVALID_PACKET, &[], &[], &[]),
        };

        let bytes: Vec<u8> = pcm[..samples * channels]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        self.write_output_buffer(tls, 0, &bytes);

        let mut raw = Vec::with_capacity(16);
        raw.extend_from_slice(&(size + PACKET_HEADER_LEN).to_le_bytes());
        raw.extend_from_slice(&(samples as u32).to_le_bytes());
        if with_perf {
            // Decode time in microseconds, as the DSP would take (real-time factor 8).
            let micros = (samples as u64 * 1_000_000) / 48_000 / 8;
            raw.extend_from_slice(&micros.to_le_bytes());
        }
        self.write_ipc_response(tls, 0, &[], &raw, &[])
    }
}

/// 120 ms at 48 kHz, or 60 ms without large frames.
fn max_frame(sample_rate: u32, large: bool) -> usize {
    let millis = if large { 120 } else { 60 };
    (sample_rate as usize * millis) / 1000
}
