//! `hwopus`, the console's Opus decoder.
//!
//! On hardware this is a service in front of the audio DSP: the caller hands
//! it a work buffer as transfer memory, the DSP decodes into it, and the PCM
//! comes back through an output buffer. Here the decode happens in
//! [`crate::opus`] on the emulator's own side, so the work buffer is sized
//! and never read — [`work_buffer_size`] still has to answer, because the
//! caller allocates from it before it opens anything.
//!
//! The packets are not bare Opus. Each one arrives behind an eight-byte
//! header of `{u32 size, u32 final_range}`, **big-endian**, and the reply's
//! "bytes consumed" counts that header. `final_range` is the range coder
//! state the encoder finished the packet with; a decoder that stayed in step
//! ends with the same value, which is what
//! [`crate::opus::Decoder::final_range`] reports.

use super::Cpu;
use crate::opus;
use crate::Result;

/// `hwopus` (module 111) description 1001: a sample rate Opus does not have.
const OPUS_INVALID_SAMPLE_RATE: u32 = 111 | (1001 << 9);

/// Description 1002: a channel count this decoder cannot open.
const OPUS_INVALID_CHANNEL_COUNT: u32 = 111 | (1002 << 9);

/// Description 8: the input buffer is too short to hold even the header.
const OPUS_INPUT_TOO_SMALL: u32 = 111 | (8 << 9);

/// Description 3: the input buffer is shorter than the header says.
const OPUS_BUFFER_TOO_SMALL: u32 = 111 | (3 << 9);

/// Description 17: the packet is not decodable Opus.
const OPUS_INVALID_PACKET: u32 = 111 | (17 << 9);

/// The eight-byte header in front of every packet.
const PACKET_HEADER_LEN: u32 = 8;

/// The DSP's own decoder object, per channel count. On hardware this is
/// `opus_decoder_get_size` plus the object around it; here it is only a
/// number the caller sizes an allocation with, since nothing reads that
/// allocation. Erring small would be the dangerous direction, so these are
/// the reference library's own sizes rounded up.
const DECODER_STATE_SIZE: [u32; 2] = [0x4A00, 0x6C00];

/// The largest number of streams Opus multi-stream allows.
const MAX_STREAMS: u32 = 255;

/// `OpusMultiStreamParameters`, which is too wide for a request's raw data
/// and so arrives in a buffer of its own.
struct MultiStreamParams {
    sample_rate: u32,
    channels: u32,
    total_streams: u32,
    stereo_streams: u32,
    large_frame: bool,
    /// Which stream, and which half of it, each output channel plays.
    mapping: Vec<u8>,
}

/// One open `IHardwareOpusDecoder`.
pub(crate) struct HwOpus {
    decoder: Decoder,
    channels: usize,
    /// The largest number of samples per channel one packet can produce,
    /// which bounds how much a decode may write into the caller's buffer.
    max_frame: usize,
}

/// Either a plain decoder or a multi-stream one — the two are opened by
/// different commands and decoded by different ones, and a decoder never
/// changes from one to the other.
enum Decoder {
    /// Boxed because a single decoder carries far more state than a
    /// multi-stream one's list of them, and these live in a map.
    Single(Box<opus::Decoder>),
    Multi(opus::MultiStreamDecoder),
}

impl core::fmt::Debug for HwOpus {
    /// The decoder's own state is a megabyte of filter history that says
    /// nothing useful in a dump; what a reader wants is how it was opened.
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

/// How large a work buffer a decoder with these parameters needs.
///
/// The base is the decoder object itself; on top of it goes one frame of
/// scratch at the requested rate, and a fixed allowance for the rest of the
/// object's bookkeeping.
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

/// The same for a multi-stream decoder, which also needs room to hold each
/// stream's sub-packet while it is unpicked from the one it arrived in.
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
    // The sample-rate error for a bad stream count is the console's own
    // answer, not a slip: the real service checks all three against one
    // result.
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
    /// `hwopus`: the decoder factory, and every `IHardwareOpusDecoder` it
    /// hands out.
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
            // OpenHardwareOpusDecoder: in { OpusParameters, u32 work size },
            // the work buffer as transfer memory, out the decoder.
            Some(0) => {
                let sample_rate = self.mem.read_u32(data).unwrap_or(0);
                let channels = self.mem.read_u32(data.wrapping_add(4)).unwrap_or(0);
                self.hwopus_open(tls, handle, sample_rate, channels, false)
            }
            // GetWorkBufferSize: in OpusParameters, out u32.
            Some(1) => {
                let sample_rate = self.mem.read_u32(data).unwrap_or(0);
                let channels = self.mem.read_u32(data.wrapping_add(4)).unwrap_or(0);
                self.hwopus_reply_size(tls, work_buffer_size(sample_rate, channels, false))
            }
            // OpenHardwareOpusDecoderForMultiStream: the parameters are too
            // wide for the raw data, so they arrive in a pointer buffer.
            Some(2) => {
                let params = self.hwopus_multistream_params(tls, false);
                self.hwopus_open_multistream(tls, handle, params)
            }
            // GetWorkBufferSizeForMultiStream.
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
            // OpenHardwareOpusDecoderEx: as command 0, plus the large-frame
            // flag that doubles the longest packet the decoder will take.
            Some(4) => {
                let sample_rate = self.mem.read_u32(data).unwrap_or(0);
                let channels = self.mem.read_u32(data.wrapping_add(4)).unwrap_or(0);
                let large = self.mem.read_u8(data.wrapping_add(8)).unwrap_or(0) != 0;
                self.hwopus_open(tls, handle, sample_rate, channels, large)
            }
            // GetWorkBufferSizeEx / GetWorkBufferSizeExEx.
            Some(5) | Some(8) => {
                let sample_rate = self.mem.read_u32(data).unwrap_or(0);
                let channels = self.mem.read_u32(data.wrapping_add(4)).unwrap_or(0);
                let large = self.mem.read_u8(data.wrapping_add(8)).unwrap_or(0) != 0;
                self.hwopus_reply_size(tls, work_buffer_size(sample_rate, channels, large))
            }
            // OpenHardwareOpusDecoderForMultiStreamEx.
            Some(6) => {
                let params = self.hwopus_multistream_params(tls, true);
                self.hwopus_open_multistream(tls, handle, params)
            }
            // GetWorkBufferSizeForMultiStreamEx / …ExEx.
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

    /// `OpusMultiStreamParameters`, from the pointer buffer the caller sent
    /// it in: rate, channels, stream counts, the large-frame flag on the `Ex`
    /// forms, and the channel mapping.
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

    /// `IHardwareOpusDecoder`. Every command but the two `SetContext`s is a
    /// decode; they differ only in whether they report how long the decode
    /// took, whether they reset the decoder first, and whether the stream is
    /// multi-stream.
    fn hwopus_decoder_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        /// Whether the command's reply carries the decode time.
        const WITH_PERF: [bool; 10] = [
            false, false, false, false, true, true, true, true, true, true,
        ];
        /// Whether the command's request carries a reset flag.
        const WITH_RESET: [bool; 10] = [
            false, false, false, false, false, false, true, true, true, true,
        ];

        let key = self.ipc_object_key(tls, handle);
        match cmd_id {
            // SetContext / SetContextForMultiStream. The context is the
            // hardware decoder's own memory image, which says nothing about
            // the state of this one; a caller only uses it to resume a stream
            // it has already been feeding, and feeding it is what actually
            // carries the state here.
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
        // The header is big-endian, unlike everything else on the wire here.
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
        // The caller's buffer bounds the decode as much as the packet does:
        // a frame longer than it left room for cannot be handed back.
        let frame = decoder.max_frame.min(output_room / (2 * channels));
        let mut pcm = vec![0i16; frame * channels];
        let decoded = match &mut decoder.decoder {
            Decoder::Single(decoder) => decoder.decode(Some(&packet), &mut pcm, frame),
            Decoder::Multi(decoder) => decoder.decode(Some(&packet), &mut pcm, frame),
        };
        let samples = match decoded {
            Ok(samples) => samples,
            // A buffer too short for the frame is the caller's mistake and
            // says so; anything else means the packet itself was not Opus.
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
            // How long the decode took, in microseconds. Reporting zero would
            // be a lie a caller can act on — `nn::codec` uses it to decide
            // how far ahead to decode — so report the time the hardware would
            // have taken, which is the samples' own duration over the DSP's
            // real-time factor.
            let micros = (samples as u64 * 1_000_000) / 48_000 / 8;
            raw.extend_from_slice(&micros.to_le_bytes());
        }
        self.write_ipc_response(tls, 0, &[], &raw, &[])
    }
}

/// The most samples per channel one packet can decode to: 120 ms at 48 kHz,
/// or 60 ms for a decoder that was not opened for large frames.
fn max_frame(sample_rate: u32, large: bool) -> usize {
    let millis = if large { 120 } else { 60 };
    (sample_rate as usize * millis) / 1000
}
