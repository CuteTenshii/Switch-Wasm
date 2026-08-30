//! An Opus decoder (RFC 6716).
//!
//! Opus is two codecs behind one container. SILK is a linear-prediction coder
//! that carries speech efficiently at low rates and internally runs at 8, 12
//! or 16 kHz; CELT is an MDCT transform coder that carries everything else at
//! 48 kHz. A packet is coded by one of them, or — in hybrid mode — by both at
//! once, SILK below 8 kHz and CELT above it.
//!
//! Which of the three a packet used is in its first byte, and a stream may
//! change from one packet to the next. That is the awkward part of decoding
//! Opus: the two codecs have different internal rates, different frame
//! lengths and entirely separate state, so a switch has to be faded through
//! rather than simply taken, and both decoders have to be kept warm across
//! frames that did not use them.
//!
//! What is here is a decoder only. Nothing on this console encodes Opus.

// RFC 6716's decoder, transcribed rather than rewritten: the tables are
// normative, the integer arithmetic has to round the way the encoder's did,
// and a loop that carries its index into the arithmetic reads as the
// reference does only while it stays a loop. Clippy's suggestions here range
// from noise to actively wrong — shortening a table constant or swapping in
// `FRAC_1_SQRT_2` desynchronises the range decoder — so they are off for the
// ported modules and on everywhere else.
macro_rules! ported {
    ($($item:item)*) => {
        $(
            #[allow(
                clippy::approx_constant,
                clippy::assign_op_pattern,
                clippy::excessive_precision,
                clippy::explicit_counter_loop,
                clippy::int_plus_one,
                clippy::manual_is_multiple_of,
                clippy::needless_range_loop,
                clippy::too_many_arguments
            )]
            $item
        )*
    };
}

ported! {
    pub(crate) mod celt;
    mod mdct;
    mod silk;
    mod tables_celt;
    mod tables_silk;
}

mod range;

use celt::CeltDecoder;
use range::RangeDecoder;
use silk::SilkDecoder;

/// Why a packet could not be decoded. A caller that gets one of these should
/// conceal the frame rather than stop: a damaged packet in a stream is not a
/// damaged stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The packet is malformed — a length that does not fit, a frame count of
    /// zero, a duration no mode can produce.
    InvalidPacket,
    /// The arguments do not describe something this decoder can do.
    BadArgument,
    /// The output buffer is shorter than the packet's own duration.
    BufferTooSmall,
}

pub type Result<T> = core::result::Result<T, Error>;

/// Which codec, or both, coded a packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    SilkOnly,
    Hybrid,
    CeltOnly,
}

/// The coded audio bandwidth, which is what sets CELT's top band and SILK's
/// internal rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bandwidth {
    Narrow,
    Medium,
    Wide,
    SuperWide,
    Full,
}

impl Bandwidth {
    /// The first CELT band above this bandwidth.
    fn end_band(self) -> usize {
        match self {
            Bandwidth::Narrow => 13,
            Bandwidth::Medium | Bandwidth::Wide => 17,
            Bandwidth::SuperWide => 19,
            Bandwidth::Full => 21,
        }
    }

    /// The rate SILK codes at internally for this bandwidth.
    fn silk_rate(self) -> u32 {
        match self {
            Bandwidth::Narrow => 8000,
            Bandwidth::Medium => 12000,
            _ => 16000,
        }
    }
}

/// What a packet's first byte says about it, before any of it is decoded.
struct Toc {
    mode: Mode,
    bandwidth: Bandwidth,
    /// Samples per frame at the decoder's output rate.
    frame_size: usize,
    channels: usize,
}

fn parse_toc(byte: u8, fs: u32) -> Toc {
    let mode = if byte & 0x80 != 0 {
        Mode::CeltOnly
    } else if byte & 0x60 == 0x60 {
        Mode::Hybrid
    } else {
        Mode::SilkOnly
    };
    let bandwidth = match mode {
        Mode::CeltOnly => match (byte >> 5) & 0x3 {
            0 => Bandwidth::Narrow,
            1 => Bandwidth::Wide,
            2 => Bandwidth::SuperWide,
            _ => Bandwidth::Full,
        },
        Mode::Hybrid => {
            if byte & 0x10 != 0 {
                Bandwidth::Full
            } else {
                Bandwidth::SuperWide
            }
        }
        Mode::SilkOnly => match (byte >> 5) & 0x3 {
            0 => Bandwidth::Narrow,
            1 => Bandwidth::Medium,
            _ => Bandwidth::Wide,
        },
    };
    let frame_size = match mode {
        Mode::CeltOnly => ((fs << ((byte >> 3) & 0x3)) / 400) as usize,
        Mode::Hybrid => {
            if byte & 0x08 != 0 {
                (fs / 50) as usize
            } else {
                (fs / 100) as usize
            }
        }
        Mode::SilkOnly => match (byte >> 3) & 0x3 {
            3 => (fs * 60 / 1000) as usize,
            n => ((fs << n) / 100) as usize,
        },
    };
    Toc {
        mode,
        bandwidth,
        frame_size,
        channels: if byte & 0x4 != 0 { 2 } else { 1 },
    }
}

/// A frame length, coded as one byte below 252 and two above it.
fn parse_size(data: &[u8]) -> Result<(usize, usize)> {
    match data.first() {
        None => Err(Error::InvalidPacket),
        Some(&first) if first < 252 => Ok((1, usize::from(first))),
        Some(&first) => match data.get(1) {
            None => Err(Error::InvalidPacket),
            Some(&second) => Ok((2, 4 * usize::from(second) + usize::from(first))),
        },
    }
}

/// The frames a packet holds, as `(offset, length)` into the packet, and how
/// far into `data` the packet actually ends.
///
/// Four framings share the two low bits of the first byte: one frame, two of
/// equal size, two of coded sizes, and an arbitrary run with its own count,
/// optional padding and either equal or coded sizes.
///
/// `self_delimited` is the variant used inside a multi-stream packet, where
/// each stream's sub-packet has to say where it ends because another follows
/// it: the last frame's length is coded rather than implied by what is left.
fn parse_frames(data: &[u8], self_delimited: bool) -> Result<(Vec<(usize, usize)>, usize)> {
    if data.is_empty() {
        return Err(Error::InvalidPacket);
    }
    let toc = data[0];
    let mut at = 1usize;
    let mut len = data.len() - 1;
    let mut sizes: Vec<usize> = Vec::new();
    let mut last_size = len;
    let mut cbr = false;
    let mut pad = 0usize;
    let count;

    match toc & 0x3 {
        0 => count = 1,
        1 => {
            count = 2;
            cbr = true;
            if !self_delimited {
                if len & 1 != 0 {
                    return Err(Error::InvalidPacket);
                }
                last_size = len / 2;
                sizes.push(last_size);
            }
        }
        2 => {
            count = 2;
            let (bytes, size) = parse_size(&data[at..])?;
            len -= bytes;
            if size > len {
                return Err(Error::InvalidPacket);
            }
            at += bytes;
            sizes.push(size);
            last_size = len - size;
        }
        _ => {
            if len < 1 {
                return Err(Error::InvalidPacket);
            }
            let ch = data[at];
            at += 1;
            len -= 1;
            count = usize::from(ch & 0x3F);
            if count == 0 {
                return Err(Error::InvalidPacket);
            }
            // Padding is coded as a run of 255s and then a final count.
            if ch & 0x40 != 0 {
                loop {
                    if len == 0 {
                        return Err(Error::InvalidPacket);
                    }
                    let p = data[at];
                    at += 1;
                    len -= 1;
                    let skip = if p == 255 { 254 } else { usize::from(p) };
                    if skip > len {
                        return Err(Error::InvalidPacket);
                    }
                    len -= skip;
                    pad += skip;
                    if p != 255 {
                        break;
                    }
                }
            }
            cbr = ch & 0x80 == 0;
            if !cbr {
                last_size = len;
                for _ in 0..count - 1 {
                    let (bytes, size) = parse_size(&data[at..])?;
                    if bytes > len {
                        return Err(Error::InvalidPacket);
                    }
                    len -= bytes;
                    if size > len {
                        return Err(Error::InvalidPacket);
                    }
                    at += bytes;
                    sizes.push(size);
                    if bytes + size > last_size {
                        return Err(Error::InvalidPacket);
                    }
                    last_size -= bytes + size;
                }
            } else if !self_delimited {
                if !len.is_multiple_of(count) {
                    return Err(Error::InvalidPacket);
                }
                last_size = len / count;
                for _ in 0..count - 1 {
                    sizes.push(last_size);
                }
            }
        }
    }

    if self_delimited {
        let (bytes, size) = parse_size(&data[at..])?;
        if bytes > len {
            return Err(Error::InvalidPacket);
        }
        len -= bytes;
        if size > len {
            return Err(Error::InvalidPacket);
        }
        at += bytes;
        if cbr {
            if size * count > len {
                return Err(Error::InvalidPacket);
            }
            sizes.clear();
            for _ in 0..count - 1 {
                sizes.push(size);
            }
        } else if bytes + size > last_size {
            return Err(Error::InvalidPacket);
        }
        last_size = size;
    } else if last_size > 1275 {
        // Not coded explicitly, so nothing above stopped it being too large.
        return Err(Error::InvalidPacket);
    }
    sizes.push(last_size);

    let mut frames = Vec::with_capacity(sizes.len());
    for size in sizes {
        if at + size > data.len() {
            return Err(Error::InvalidPacket);
        }
        frames.push((at, size));
        at += size;
    }
    Ok((frames, pad + at))
}

/// One Opus stream's decoder: one SILK decoder, one CELT decoder, and the
/// state needed to cross between them.
pub struct Decoder {
    fs: u32,
    channels: usize,
    stream_channels: usize,
    silk: SilkDecoder,
    celt: CeltDecoder,
    /// What the last packet used, so a mode change can be faded.
    prev_mode: Option<Mode>,
    prev_redundancy: bool,
    mode: Option<Mode>,
    bandwidth: Option<Bandwidth>,
    frame_size: usize,
    last_packet_duration: usize,
    final_range: u32,
    /// What SILK was last told about the stream. A lost packet carries no
    /// description of itself, so these carry over.
    silk_channels_internal: usize,
    silk_internal_rate: u32,
}

impl Decoder {
    /// A decoder producing `channels` channels at `fs` Hz. Opus decodes at
    /// 48 kHz internally and decimates, so only the rates that divide it are
    /// available.
    pub fn new(fs: u32, channels: usize) -> Result<Self> {
        if !matches!(fs, 8000 | 12000 | 16000 | 24000 | 48000) || !matches!(channels, 1 | 2) {
            return Err(Error::BadArgument);
        }
        let mut celt = CeltDecoder::new(channels);
        celt.downsample = (48000 / fs) as usize;
        Ok(Decoder {
            fs,
            channels,
            stream_channels: channels,
            silk: SilkDecoder::new(),
            celt,
            prev_mode: None,
            prev_redundancy: false,
            mode: None,
            bandwidth: None,
            frame_size: (fs / 400) as usize,
            last_packet_duration: 0,
            final_range: 0,
            silk_channels_internal: channels,
            silk_internal_rate: 16000,
        })
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn sample_rate(&self) -> u32 {
        self.fs
    }

    /// The range coder's state after the last packet. Comparing it with the
    /// encoder's is how two implementations prove they stayed in step.
    pub fn final_range(&self) -> u32 {
        self.final_range
    }

    pub fn reset(&mut self) {
        self.silk = SilkDecoder::new();
        self.celt.reset();
        self.prev_mode = None;
        self.prev_redundancy = false;
        self.mode = None;
        self.bandwidth = None;
        self.frame_size = (self.fs / 400) as usize;
        self.last_packet_duration = 0;
        self.final_range = 0;
        self.silk_channels_internal = self.channels;
        self.silk_internal_rate = 16000;
    }

    /// Decode one packet into interleaved samples in `-1.0..=1.0`, returning
    /// how many samples per channel were written. `None` conceals a lost
    /// packet of `frame_size` samples.
    pub fn decode_float(
        &mut self,
        packet: Option<&[u8]>,
        pcm: &mut [f32],
        frame_size: usize,
    ) -> Result<usize> {
        self.decode_native(packet, pcm, frame_size, false)
            .map(|(samples, _)| samples)
    }

    /// The form [`MultiStreamDecoder`] needs: it also reports where the
    /// packet ended, because the next stream's starts there.
    fn decode_native(
        &mut self,
        packet: Option<&[u8]>,
        pcm: &mut [f32],
        frame_size: usize,
        self_delimited: bool,
    ) -> Result<(usize, usize)> {
        let data = match packet {
            Some(data) if !data.is_empty() => data,
            _ => {
                // A loss is concealed in whole 2.5 ms steps, because that is
                // the shortest thing either codec can synthesise.
                if !frame_size.is_multiple_of(self.fs as usize / 400) {
                    return Err(Error::BadArgument);
                }
                let mut done = 0usize;
                while done < frame_size {
                    let got = self.decode_frame(None, pcm, done, frame_size - done)?;
                    done += got;
                }
                self.last_packet_duration = done;
                return Ok((done, 0));
            }
        };

        let toc = parse_toc(data[0], self.fs);
        let (frames, packet_offset) = parse_frames(data, self_delimited)?;
        if frames.len() * toc.frame_size > frame_size {
            return Err(Error::BufferTooSmall);
        }

        // Only commit the packet's parameters once it has parsed, so a
        // damaged packet does not leave the decoder describing itself wrong.
        self.mode = Some(toc.mode);
        self.bandwidth = Some(toc.bandwidth);
        self.frame_size = toc.frame_size;
        self.stream_channels = toc.channels;

        let mut done = 0usize;
        for &(offset, size) in &frames {
            let got = self.decode_frame(
                Some(&data[offset..offset + size]),
                pcm,
                done,
                frame_size - done,
            )?;
            done += got;
        }
        self.last_packet_duration = done;
        Ok((done, packet_offset))
    }

    /// Decode one packet into interleaved 16-bit samples.
    pub fn decode(
        &mut self,
        packet: Option<&[u8]>,
        pcm: &mut [i16],
        frame_size: usize,
    ) -> Result<usize> {
        let mut float = vec![0.0f32; frame_size * self.channels];
        let got = self.decode_float(packet, &mut float, frame_size)?;
        for (out, &v) in pcm.iter_mut().zip(float[..got * self.channels].iter()) {
            *out = float_to_i16(v);
        }
        Ok(got)
    }

    /// Decode one frame of a packet — the unit both codecs actually work in.
    /// `at` is where in `pcm` it goes, in samples per channel.
    fn decode_frame(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [f32],
        at: usize,
        room: usize,
    ) -> Result<usize> {
        let f20 = self.fs as usize / 50;
        let f10 = f20 >> 1;
        let f5 = f10 >> 1;
        let f2_5 = f5 >> 1;
        if room < f2_5 {
            return Err(Error::BufferTooSmall);
        }
        let out = &mut pcm[at * self.channels..];

        // A payload of one byte carries no audio: it is a "this frame is
        // silent" marker, and is concealed rather than decoded.
        let data = data.filter(|d| d.len() > 1);

        let (mode, bandwidth, mut audiosize) = match data {
            Some(_) => (self.mode.unwrap(), self.bandwidth, self.frame_size),
            None => {
                let mode = if self.prev_redundancy {
                    Some(Mode::CeltOnly)
                } else {
                    self.prev_mode
                };
                let Some(mode) = mode else {
                    // Nothing has been decoded yet, so there is nothing to
                    // conceal from.
                    let n = room.min(self.frame_size);
                    out[..n * self.channels].fill(0.0);
                    return Ok(n);
                };
                (mode, None, room.min(self.frame_size))
            }
        };

        if data.is_none() {
            // Conceal only in the lengths the codecs have: 2.5, 5, 10 or 20
            // ms, never 12.5 or 30.
            if audiosize > f20 {
                let mut done = 0usize;
                while done < audiosize {
                    let step = (audiosize - done).min(f20);
                    let got = self.decode_frame(None, pcm, at + done, step)?;
                    done += got;
                }
                return Ok(done);
            } else if audiosize < f20 {
                if audiosize > f10 {
                    audiosize = f10;
                } else if mode != Mode::SilkOnly && audiosize > f5 && audiosize < f10 {
                    audiosize = f5;
                }
            }
        }
        if audiosize > room {
            return Err(Error::BadArgument);
        }
        let frame_size = audiosize;

        // Crossing between the two codecs is faded, because their overlap
        // windows do not line up and a hard cut is audible as a click.
        let transition = data.is_some()
            && self.prev_mode.is_some()
            && ((mode == Mode::CeltOnly
                && self.prev_mode != Some(Mode::CeltOnly)
                && !self.prev_redundancy)
                || (mode != Mode::CeltOnly && self.prev_mode == Some(Mode::CeltOnly)));
        let mut pcm_transition: Option<Vec<f32>> = None;
        if transition && mode == Mode::CeltOnly {
            let n = f5.min(audiosize);
            let mut buf = vec![0.0f32; f5 * self.channels];
            self.decode_frame(None, &mut buf, 0, n)?;
            pcm_transition = Some(buf);
        }

        let mut dec = data.map(RangeDecoder::new);
        let mut len = data.map_or(0, |d| d.len());

        let mut silk_pcm = vec![0i16; f10.max(frame_size) * self.channels];
        if mode != Mode::CeltOnly {
            if self.prev_mode == Some(Mode::CeltOnly) {
                self.silk = SilkDecoder::new();
            }
            // A concealed frame keeps the last packet's channel count and
            // internal rate: nothing in a lost packet says what they were,
            // and SILK's state describes a signal at that rate.
            if data.is_some() {
                self.silk_channels_internal = self.stream_channels;
                self.silk_internal_rate = match mode {
                    Mode::SilkOnly => bandwidth.map_or(16000, Bandwidth::silk_rate),
                    _ => 16000,
                };
            }
            let control = silk::Control {
                api_sample_rate: self.fs,
                channels_api: self.channels,
                channels_internal: self.silk_channels_internal,
                internal_sample_rate: self.silk_internal_rate,
                // SILK's own concealment cannot make anything shorter than
                // 10 ms, whatever the packet claimed.
                payload_size_ms: 10.max(1000 * audiosize / self.fs as usize),
            };
            let mut decoded = 0usize;
            while decoded < frame_size {
                let first = decoded == 0;
                let got = match dec.as_mut() {
                    Some(dec) => self.silk.decode(
                        &control,
                        false,
                        first,
                        Some(dec),
                        &mut silk_pcm[decoded * self.channels..],
                    ),
                    None => self.silk.decode(
                        &control,
                        true,
                        first,
                        None,
                        &mut silk_pcm[decoded * self.channels..],
                    ),
                };
                match got {
                    Ok(n) => decoded += n,
                    Err(_) => {
                        // A concealment failure is not fatal; silence is.
                        for v in
                            silk_pcm[decoded * self.channels..frame_size * self.channels].iter_mut()
                        {
                            *v = 0;
                        }
                        decoded = frame_size;
                    }
                }
            }
        }

        // A packet may carry a 5 ms CELT frame beside its SILK, to cover the
        // band SILK does not reach across a mode change.
        let mut start_band = 0usize;
        let mut redundancy = false;
        let mut redundancy_bytes = 0usize;
        let mut celt_to_silk = false;
        if mode != Mode::CeltOnly {
            if let Some(dec) = dec.as_mut() {
                if dec.tell() + 17 + 20 * i32::from(mode == Mode::Hybrid) <= 8 * len as i32 {
                    redundancy = if mode == Mode::Hybrid {
                        dec.decode_bit_logp(12)
                    } else {
                        true
                    };
                    if redundancy {
                        celt_to_silk = dec.decode_bit_logp(1);
                        redundancy_bytes = if mode == Mode::Hybrid {
                            dec.decode_uint(256) as usize + 2
                        } else {
                            len - ((dec.tell() as usize + 7) >> 3)
                        };
                        len -= redundancy_bytes;
                        if (len as i32) * 8 < dec.tell() {
                            len = 0;
                            redundancy_bytes = 0;
                            redundancy = false;
                        }
                        // The redundant frame's bytes are at the end, where
                        // this frame's raw bits would otherwise be read from.
                        dec.shrink(redundancy_bytes);
                    }
                }
            }
            start_band = 17;
        }

        let transition = transition && !redundancy;
        if transition && mode != Mode::CeltOnly {
            let n = f5.min(audiosize);
            let mut buf = vec![0.0f32; f5 * self.channels];
            self.decode_frame(None, &mut buf, 0, n)?;
            pcm_transition = Some(buf);
        }

        if let Some(bandwidth) = bandwidth {
            self.celt.end = bandwidth.end_band();
        }
        self.celt.set_channels(self.stream_channels);

        let mut redundant_audio = vec![0.0f32; if redundancy { f5 * self.channels } else { 0 }];
        let mut redundant_rng = 0u32;
        if redundancy && celt_to_silk {
            // Decoded even when the CELT state is stale and the audio will go
            // unused, because the range coder's final state depends on it.
            self.celt.start = 0;
            let bytes = data.unwrap();
            let mut rdec = RangeDecoder::new(&bytes[len..len + redundancy_bytes]);
            self.celt
                .decode(Some(&mut rdec), redundancy_bytes, &mut redundant_audio, f5);
            redundant_rng = self.celt.rng;
        }

        self.celt.start = start_band;
        if mode != Mode::SilkOnly {
            let celt_frame_size = f20.min(frame_size);
            // A mode change leaves the CELT state describing a different
            // signal; keeping it would ring.
            if Some(mode) != self.prev_mode && self.prev_mode.is_some() && !self.prev_redundancy {
                self.celt.reset();
            }
            match dec.as_mut() {
                Some(dec) => {
                    self.celt.decode(Some(dec), len, out, celt_frame_size);
                }
                None => {
                    self.celt.decode(None, 0, out, celt_frame_size);
                }
            }
        } else {
            out[..frame_size * self.channels].fill(0.0);
            // Coming out of hybrid, let the MDCT fade the CELT half out
            // rather than dropping it.
            if self.prev_mode == Some(Mode::Hybrid)
                && !(redundancy && celt_to_silk && self.prev_redundancy)
            {
                self.celt.start = 0;
                let silence = [0xFFu8, 0xFF];
                let mut sdec = RangeDecoder::new(&silence);
                self.celt.decode(Some(&mut sdec), 2, out, f2_5);
            }
        }

        if mode != Mode::CeltOnly {
            for i in 0..frame_size * self.channels {
                out[i] += (1.0 / 32768.0) * f32::from(silk_pcm[i]);
            }
        }

        if redundancy && !celt_to_silk {
            self.celt.reset();
            self.celt.start = 0;
            let bytes = data.unwrap();
            let mut rdec = RangeDecoder::new(&bytes[len..len + redundancy_bytes]);
            self.celt
                .decode(Some(&mut rdec), redundancy_bytes, &mut redundant_audio, f5);
            redundant_rng = self.celt.rng;
            let tail = self.channels * (frame_size - f2_5);
            fade_in_over(
                &redundant_audio[self.channels * f2_5..],
                &mut out[tail..],
                f2_5,
                self.channels,
                self.fs,
            );
        }
        if redundancy
            && celt_to_silk
            && (self.prev_mode != Some(Mode::SilkOnly) || self.prev_redundancy)
        {
            out[..self.channels * f2_5].copy_from_slice(&redundant_audio[..self.channels * f2_5]);
            fade_out_under(
                &redundant_audio[self.channels * f2_5..],
                &mut out[self.channels * f2_5..],
                f2_5,
                self.channels,
                self.fs,
            );
        }
        if transition {
            if let Some(prev) = pcm_transition {
                if audiosize >= f5 {
                    out[..self.channels * f2_5].copy_from_slice(&prev[..self.channels * f2_5]);
                    fade_out_under(
                        &prev[self.channels * f2_5..],
                        &mut out[self.channels * f2_5..],
                        f2_5,
                        self.channels,
                        self.fs,
                    );
                } else {
                    fade_out_under(&prev, out, f2_5, self.channels, self.fs);
                }
            }
        }

        self.final_range = match dec.as_ref() {
            Some(dec) if len > 1 => dec.rng() ^ redundant_rng,
            _ => 0,
        };
        self.prev_mode = Some(mode);
        self.prev_redundancy = redundancy && !celt_to_silk;
        Ok(audiosize)
    }
}

/// Fade `other` in over what `out` already holds, using the squared CELT
/// window so the two halves sum to unity.
fn fade_in_over(other: &[f32], out: &mut [f32], overlap: usize, channels: usize, fs: u32) {
    let inc = (48000 / fs) as usize;
    for c in 0..channels {
        for i in 0..overlap {
            let w = celt::window_at(i * inc);
            let w = w * w;
            let j = i * channels + c;
            out[j] = w * other[j] + (1.0 - w) * out[j];
        }
    }
}

/// The mirror of [`fade_in_over`]: `out` is what fades in, `other` what fades
/// out under it.
fn fade_out_under(other: &[f32], out: &mut [f32], overlap: usize, channels: usize, fs: u32) {
    let inc = (48000 / fs) as usize;
    for c in 0..channels {
        for i in 0..overlap {
            let w = celt::window_at(i * inc);
            let w = w * w;
            let j = i * channels + c;
            out[j] = w * out[j] + (1.0 - w) * other[j];
        }
    }
}

/// Convert one sample to 16-bit, rounding to nearest and clipping.
fn float_to_i16(x: f32) -> i16 {
    let v = (x * 32768.0).round();
    if v > 32767.0 {
        32767
    } else if v < -32768.0 {
        -32768
    } else {
        v as i16
    }
}

/// A multi-stream Opus decoder: several Opus streams in one packet, mapped
/// onto more channels than one stream can carry.
///
/// Each stream is either mono or a coupled stereo pair, and the mapping says
/// which output channel takes which half of which stream. A channel mapped to
/// 255 is silent — surround layouts use that for the ones a particular mix
/// does not fill.
pub struct MultiStreamDecoder {
    decoders: Vec<Decoder>,
    channels: usize,
    streams: usize,
    coupled: usize,
    mapping: Vec<u8>,
}

impl MultiStreamDecoder {
    /// `mapping[c]` selects what output channel `c` plays: `2*s` and `2*s+1`
    /// for the two halves of coupled stream `s`, `coupled + s` for mono
    /// stream `s`, and 255 for silence.
    pub fn new(
        fs: u32,
        channels: usize,
        streams: usize,
        coupled: usize,
        mapping: &[u8],
    ) -> Result<Self> {
        if streams == 0 || coupled > streams || streams + coupled > 255 || mapping.len() < channels
        {
            return Err(Error::BadArgument);
        }
        let mut decoders = Vec::with_capacity(streams);
        for s in 0..streams {
            decoders.push(Decoder::new(fs, if s < coupled { 2 } else { 1 })?);
        }
        Ok(MultiStreamDecoder {
            decoders,
            channels,
            streams,
            coupled,
            mapping: mapping[..channels].to_vec(),
        })
    }

    pub fn reset(&mut self) {
        for decoder in self.decoders.iter_mut() {
            decoder.reset();
        }
    }

    /// The range coder state of the last stream decoded, which is what a
    /// caller comparing against an encoder checks.
    pub fn final_range(&self) -> u32 {
        self.decoders.iter().fold(0, |acc, d| acc ^ d.final_range())
    }

    /// Decode one packet into `channels` interleaved channels.
    pub fn decode_float(
        &mut self,
        packet: Option<&[u8]>,
        pcm: &mut [f32],
        frame_size: usize,
    ) -> Result<usize> {
        // Every stream but the last is self-delimited, because another
        // follows it in the same packet.
        let mut at = 0usize;
        let mut produced = frame_size;
        let mut buf = vec![0.0f32; 2 * frame_size];

        for s in 0..self.streams {
            let sub = packet.map(|data| &data[at.min(data.len())..]);
            let self_delimited = s != self.streams - 1;
            let (got, offset) =
                self.decoders[s].decode_native(sub, &mut buf, produced, self_delimited)?;
            at += offset;
            produced = got;

            let stride = if s < self.coupled { 2 } else { 1 };
            for half in 0..stride {
                let tag = if s < self.coupled {
                    (2 * s + half) as u8
                } else {
                    (self.coupled + s) as u8
                };
                for c in 0..self.channels {
                    if self.mapping[c] != tag {
                        continue;
                    }
                    for i in 0..got {
                        pcm[i * self.channels + c] = buf[i * stride + half];
                    }
                }
            }
        }

        for c in 0..self.channels {
            if self.mapping[c] == 255 {
                for i in 0..produced {
                    pcm[i * self.channels + c] = 0.0;
                }
            }
        }
        Ok(produced)
    }

    /// Decode one packet into interleaved 16-bit samples.
    pub fn decode(
        &mut self,
        packet: Option<&[u8]>,
        pcm: &mut [i16],
        frame_size: usize,
    ) -> Result<usize> {
        let mut float = vec![0.0f32; frame_size * self.channels];
        let got = self.decode_float(packet, &mut float, frame_size)?;
        for (out, &v) in pcm.iter_mut().zip(float[..got * self.channels].iter()) {
            *out = float_to_i16(v);
        }
        Ok(got)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three 20 ms CELT-only packets at 48 kHz mono, and what libopus makes of
    /// them: the range coder state it ends each one with, and the RMS of the
    /// samples it produced.
    const CELT_PACKETS: [&[u8]; 3] = [
        &[
            0xf8, 0x9f, 0xf7, 0xda, 0x9b, 0x32, 0x2b, 0xce, 0x91, 0xf2, 0x50, 0x86, 0xd0, 0xbe,
            0x88, 0x91, 0xe5, 0xfc, 0xff, 0xd1, 0xb8, 0x45, 0x4f, 0x82, 0x93, 0xbc, 0xa6, 0x61,
            0x9e, 0x76, 0x03, 0x86, 0x83, 0xf1, 0x65, 0x96, 0x94, 0xab, 0x3a, 0x3a, 0xaa, 0xb0,
            0x12, 0x91, 0x97, 0xb5, 0x53, 0xd8, 0x2f, 0x4d, 0xf4, 0x71, 0xc9, 0xdc, 0x90, 0xc5,
            0x89, 0xdd, 0x76, 0xf2, 0xf0, 0x6d, 0xd1, 0x23, 0x1a, 0xe6, 0x16, 0xcb, 0x37, 0x81,
            0x53, 0xe9, 0x70, 0x84, 0x65, 0xc0, 0x8b, 0xb0, 0x29, 0x32, 0x7b, 0xf2, 0x56, 0x71,
            0x16, 0xc0, 0xc9, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0d, 0xb2,
            0x65, 0x69, 0x55, 0x44, 0x5f, 0x55, 0x62, 0xab, 0xe7, 0x8b, 0x74, 0x5b, 0x50, 0x1f,
            0x0f, 0xb4, 0x32, 0x07, 0xa3, 0xe8, 0x5d, 0x0a, 0xbe, 0x45, 0xd7, 0x13, 0xc8, 0xc5,
            0x34, 0x08, 0x98, 0x83, 0xc0, 0x29, 0x72, 0xf6, 0x33, 0xd6, 0xe2, 0x01, 0x31, 0x70,
            0xfc, 0x4e, 0x5b, 0x77, 0xf3, 0x98, 0x1c, 0x97, 0x17, 0xf6, 0xa4, 0xe7, 0x70, 0x96,
            0x5b, 0x3e, 0x09, 0x53, 0x28, 0x6a, 0xd9, 0xe3, 0xaa, 0x85,
        ],
        &[
            0xf8, 0xc0, 0x34, 0x74, 0x3a, 0xfa, 0xca, 0x52, 0x08, 0x96, 0x87, 0x23, 0x53, 0x28,
            0xd4, 0x2a, 0xb5, 0xb9, 0x1c, 0x29, 0xc2, 0xef, 0x78, 0x78, 0x05, 0x86, 0x82, 0xd7,
            0x26, 0xb3, 0x74, 0x34, 0x65, 0x6e, 0xd8, 0x00, 0x3c, 0x57, 0x7e, 0xf1, 0x2e, 0x0f,
            0xd8, 0x3a, 0x9c, 0x0b, 0x90, 0xa7, 0x53, 0x1a, 0x23, 0x04, 0xeb, 0x36, 0x96, 0x56,
            0xd4, 0xc3, 0x5a, 0xed, 0x32, 0x46, 0x68, 0xdc, 0xf3, 0x61, 0xff, 0x7a, 0x7d, 0x49,
            0x89, 0x01, 0xdd, 0xe0, 0xc4, 0x08, 0x2e, 0x85, 0x51, 0xef, 0xbb, 0xc7, 0x3d, 0x94,
            0x40, 0x2b, 0x73, 0x6f, 0x19, 0x8e, 0xac, 0xb1, 0xc2, 0x6d, 0xa8, 0x08, 0x9e, 0xb2,
            0x97, 0x4c, 0xfb, 0x82, 0x50, 0x4b, 0xb0, 0x0b, 0x47, 0x7d, 0xe6, 0x42, 0x82, 0x33,
            0xf5, 0x8b, 0x32, 0x2b, 0x13, 0x3f, 0xc6, 0x16, 0x66, 0x02, 0x72, 0x0f, 0xa6, 0xaf,
            0x7b, 0x53, 0x90, 0x7d, 0x32, 0xc1, 0x41, 0xbf, 0x4c, 0x77, 0x25, 0xd2, 0x14, 0xec,
            0xee, 0x77, 0x1d, 0x5a, 0x10, 0x26, 0x70, 0x67, 0xbd, 0x4e, 0x93, 0x36, 0xd6, 0xf6,
            0x81, 0xb9, 0x3d, 0xd6, 0x4a, 0x63, 0xdf, 0x5d, 0x20, 0x02, 0x43, 0xca, 0x3a, 0xa4,
            0x80, 0xf2, 0x9c, 0x5f, 0x5d,
        ],
        &[
            0xf8, 0xc4, 0xa2, 0x3b, 0xbe, 0xe3, 0x1a, 0x90, 0xd5, 0x1a, 0x77, 0xec, 0xbc, 0xe7,
            0xdb, 0x00, 0xfe, 0xba, 0x23, 0xb8, 0x8c, 0x08, 0x7c, 0x5f, 0xc4, 0xe7, 0xf4, 0xa9,
            0x4d, 0x64, 0xe3, 0x1e, 0x6d, 0xd1, 0x69, 0x8c, 0x66, 0x60, 0xc9, 0x8a, 0xb0, 0x81,
            0x30, 0x9c, 0xad, 0xc0, 0x32, 0x4b, 0x58, 0x02, 0x7b, 0xa0, 0x04, 0x8f, 0x2c, 0xfb,
            0xdb, 0x1c, 0xcc, 0x7a, 0xe4, 0x2a, 0x8e, 0x05, 0x67, 0x75, 0x0d, 0xda, 0xbf, 0xc9,
            0xd7, 0xe4, 0xf4, 0xa9, 0x2d, 0xa8, 0xef, 0xcb, 0x0f, 0x4d, 0x17, 0x8a, 0x3b, 0xbd,
            0x06, 0x84, 0x66, 0xf8, 0x0f, 0x04, 0x69, 0x81, 0x9d, 0xff, 0x2f, 0x20, 0x75, 0x25,
            0x05, 0xe7, 0x50, 0x8b, 0xe9, 0x7c, 0xe2, 0xb9, 0xec, 0x34, 0x8b, 0xfc, 0xfd, 0xc7,
            0xe7, 0x6f, 0x51, 0xec, 0x18, 0xd2, 0x21, 0x8d, 0x23, 0x92, 0x53, 0x65, 0x18, 0x45,
            0x74, 0xfb, 0x58, 0x31, 0xb9, 0x6b, 0xe5, 0x03, 0x11, 0x9f, 0x5d,
        ],
    ];

    const CELT_PACKETS_RANGES: [u32; 3] = [0x0e010400, 0x00f07200, 0x170a6e00];
    const CELT_PACKETS_RMS: [f32; 3] = [0.304529, 0.317325, 0.324115];

    /// Three 20 ms SILK-only wideband packets, same shape.
    const SILK_PACKETS: [&[u8]; 3] = [
        &[
            0x48, 0x83, 0x9c, 0x2d, 0xb5, 0xa7, 0xe7, 0xf6, 0x4c, 0x00, 0x00, 0x1f, 0x4e, 0xce,
            0xfe, 0x00, 0x94, 0xa3, 0x57, 0x91, 0x64, 0x45, 0x0a, 0xaf, 0x36, 0xbc, 0xff, 0x7f,
            0x2c, 0x37, 0xa9, 0xc6, 0x90, 0x34, 0xb4, 0xb7, 0x97, 0xbb, 0x5b, 0x76, 0x80, 0x67,
            0x49, 0xe2, 0xfd, 0x88, 0x38, 0x2b, 0x56, 0x7a, 0xf0, 0xa6, 0x87, 0x2b, 0xc3, 0x19,
            0xd8, 0x6f, 0x1e, 0x6d, 0x45, 0x8d, 0x8d, 0xb2, 0x99, 0xa2, 0x06, 0xf9, 0xbe, 0x05,
            0x9d, 0x80,
        ],
        &[
            0x48, 0xaf, 0x97, 0x1d, 0x57, 0xbb, 0xb6, 0x2c, 0xfe, 0xe5, 0x8d, 0x5f, 0x16, 0xa4,
            0x76, 0xd4, 0x96, 0x22, 0xe4, 0x4f, 0xc2, 0xa9, 0x0a, 0x5b, 0x0f, 0x91, 0x2f, 0x9d,
            0x9b, 0x0a, 0x86, 0xd2, 0x79, 0xeb, 0x96, 0xdb, 0x31, 0xe3, 0x22, 0x2c, 0xca, 0x2e,
            0x43, 0xcf, 0x14, 0xa8, 0xf5, 0xc3, 0xd6, 0xb1, 0x91, 0x28, 0x4a, 0xc4, 0x86, 0x36,
            0x0c, 0xa7,
        ],
        &[
            0x48, 0xa5, 0x26, 0x7a, 0x8f, 0x85, 0xd5, 0x49, 0x03, 0xf3, 0x28, 0x5d, 0x2b, 0x5a,
            0x62, 0x31, 0x58, 0xa8, 0xab, 0xf7, 0x30, 0x62, 0x2c, 0xb5, 0x97, 0x2c, 0x09, 0x84,
            0xba, 0xc3, 0x6c, 0x1a, 0x9d, 0xe9, 0xfb, 0xf1, 0x76, 0xc0, 0x57, 0xeb, 0x96, 0x0a,
            0xef, 0x57, 0x08, 0x90, 0x0b, 0x50, 0xbc, 0xba, 0xb4, 0xc9, 0x38, 0x33, 0x83, 0x06,
            0x77, 0x9b, 0xf8,
        ],
    ];

    const SILK_PACKETS_RANGES: [u32; 3] = [0x00a5f670, 0x07a53e8d, 0x071f1754];
    const SILK_PACKETS_RMS: [f32; 3] = [0.233532, 0.319418, 0.317738];

    /// Three 20 ms hybrid fullband stereo packets, same shape.
    const HYBRID_PACKETS: [&[u8]; 3] = [
        &[
            0x7c, 0x8a, 0x0e, 0x85, 0xb2, 0x6c, 0xa8, 0x9f, 0xba, 0x3c, 0x02, 0xc0, 0x01, 0x75,
            0x5d, 0xf7, 0xc5, 0x29, 0x88, 0x44, 0x1f, 0x10, 0xec, 0xaa, 0x68, 0x4a, 0xa9, 0xf3,
            0xaf, 0x30, 0x29, 0x76, 0x85, 0xf9, 0xb1, 0xa4, 0x55, 0xa1, 0x10, 0xed, 0x92, 0xd5,
            0xb8, 0xec, 0x90, 0x5a, 0xe5, 0xa9, 0xec, 0x21, 0x04, 0x6c, 0xa2, 0x42, 0xc4, 0x34,
            0x1d, 0x96, 0x2c, 0x79, 0xca, 0x7a, 0xe0, 0x9f, 0x50, 0x4b, 0x22, 0xb8, 0xb7, 0x58,
            0x5e, 0x6e, 0xca, 0xae, 0x21, 0xeb, 0x16, 0x0c, 0x7e, 0x4c, 0xe6, 0xca, 0x0b, 0x3f,
            0xb3, 0xd0, 0x2c, 0x1b, 0x7e, 0x46, 0xa1, 0x7b, 0x9c, 0xff, 0xde, 0xcf, 0xcd, 0x50,
            0x40, 0x6e, 0x5c, 0xcf, 0x25, 0x82, 0x6f, 0xf6, 0x2b, 0x12, 0xa3, 0xd3, 0x87, 0x35,
            0x3f, 0x26, 0xc9, 0x65, 0x04, 0x88, 0xc2, 0x33, 0xca, 0x57, 0xed, 0xb2, 0xf8, 0xd0,
            0x90, 0x79, 0x05, 0xa5, 0x2d, 0xde, 0xe5, 0xb9, 0xd7, 0xdb, 0xfd, 0x85, 0x21, 0x5c,
            0x96, 0x8a, 0x38, 0xda, 0xd7, 0xd3, 0x66, 0xe2,
        ],
        &[
            0x7c, 0x8a, 0x0f, 0x71, 0xf0, 0xd8, 0x6c, 0x69, 0x5d, 0xd9, 0x5f, 0x41, 0x37, 0x75,
            0xd7, 0x5c, 0x0f, 0x2a, 0x7b, 0xf1, 0xe6, 0x2a, 0x19, 0xee, 0x41, 0x37, 0xa0, 0xf5,
            0x3e, 0x73, 0x33, 0x76, 0x9d, 0x22, 0xeb, 0x52, 0x32, 0xd1, 0xc1, 0x95, 0xaf, 0xa5,
            0x20, 0x75, 0xf6, 0x33, 0xf2, 0xb2, 0xc5, 0x17, 0x0e, 0xc0, 0xa3, 0xe4, 0x9c, 0x12,
            0xcb, 0x90, 0x34, 0xb3, 0x76, 0x19, 0x9e, 0xae, 0xa7, 0xa1, 0x11, 0x85, 0xf8, 0xec,
            0x11, 0xe2, 0xda, 0x0b, 0x83, 0xbe, 0xc6, 0xdd, 0x15, 0xb0, 0x52, 0x45, 0x30, 0x3e,
            0x51, 0x08, 0xb0, 0xb9, 0xe1, 0x50, 0x99, 0xd7, 0xcc, 0x1a, 0x44, 0xec, 0x69, 0x45,
            0x4c, 0x11, 0xbe, 0x72, 0x2a, 0x34, 0xa5, 0x4a, 0x8b, 0x09, 0xbe, 0x5d, 0x40, 0x9b,
            0x90, 0xa7, 0xfb, 0x54,
        ],
        &[
            0x7c, 0x8a, 0x0f, 0x39, 0xf3, 0xdd, 0x2e, 0xa7, 0x2c, 0x83, 0x5d, 0x59, 0x61, 0xe7,
            0xaf, 0xd5, 0xa4, 0xbd, 0x28, 0xe3, 0xac, 0xf2, 0xbc, 0xa8, 0x09, 0x0e, 0x98, 0xf8,
            0x47, 0xdb, 0xb6, 0xfd, 0x86, 0xe9, 0x2d, 0x38, 0x63, 0x0f, 0xbe, 0xeb, 0x03, 0x07,
            0x8a, 0xa0, 0x0e, 0xfd, 0x77, 0x2f, 0xa8, 0x44, 0x5c, 0xf7, 0xce, 0xf4, 0x00, 0xc6,
            0x8e, 0x84, 0xe9, 0x5a, 0x49, 0x8c, 0xab, 0x78, 0x5e, 0xd8, 0x1c, 0xdd, 0xd0, 0x1f,
            0x83, 0x3a, 0xb8, 0x86, 0x5e, 0x4a, 0xdd, 0xbb, 0x2b, 0x48, 0xa0, 0x77, 0x72, 0x14,
            0x80, 0xd9, 0x6d, 0x04, 0x59, 0x61, 0x50, 0xbf, 0xc5, 0xc1, 0x50, 0xfa, 0xb9, 0xbe,
            0x1a, 0x3a, 0xee, 0x22, 0xce, 0xf5, 0x64, 0xf6, 0xff, 0xa7, 0x1e, 0xd4, 0xe0, 0x10,
            0x8d, 0xad,
        ],
    ];

    const HYBRID_PACKETS_RANGES: [u32; 3] = [0x0f578500, 0x045d0a00, 0x04447100];
    const HYBRID_PACKETS_RMS: [f32; 3] = [0.202872, 0.267704, 0.261691];

    /// Decode `packets` and check every frame against libopus: the range
    /// coder must end each packet in the same state, which proves the same
    /// symbols were read, and the samples must have the same energy, which
    /// proves they were turned into the same signal.
    fn check(packets: &[&[u8]], ranges: &[u32], rms: &[f32], channels: usize) {
        let mut decoder = Decoder::new(48000, channels).unwrap();
        let mut pcm = vec![0.0f32; 960 * channels];
        for (frame, &packet) in packets.iter().enumerate() {
            let got = decoder.decode_float(Some(packet), &mut pcm, 960).unwrap();
            assert_eq!(got, 960, "frame {frame} decoded {got} samples");
            assert_eq!(
                decoder.final_range(),
                ranges[frame],
                "frame {frame} ended the range coder at {:#010x}, not {:#010x}",
                decoder.final_range(),
                ranges[frame]
            );
            let energy: f32 = pcm[..got * channels].iter().map(|v| v * v).sum();
            let got_rms = (energy / (got * channels) as f32).sqrt();
            assert!(
                (got_rms - rms[frame]).abs() < 1e-3,
                "frame {frame} rms {got_rms}, expected {}",
                rms[frame]
            );
        }
    }

    #[test]
    fn decodes_celt_packets_the_way_the_reference_does() {
        check(&CELT_PACKETS, &CELT_PACKETS_RANGES, &CELT_PACKETS_RMS, 1);
    }

    #[test]
    fn decodes_silk_packets_the_way_the_reference_does() {
        check(&SILK_PACKETS, &SILK_PACKETS_RANGES, &SILK_PACKETS_RMS, 1);
    }

    #[test]
    fn decodes_hybrid_packets_the_way_the_reference_does() {
        check(
            &HYBRID_PACKETS,
            &HYBRID_PACKETS_RANGES,
            &HYBRID_PACKETS_RMS,
            2,
        );
    }

    /// A lost packet is concealed rather than refused, and produces exactly
    /// the frame the caller asked for.
    #[test]
    fn conceals_a_lost_packet() {
        let mut decoder = Decoder::new(48000, 1).unwrap();
        let mut pcm = vec![0.0f32; 960];
        decoder
            .decode_float(Some(CELT_PACKETS[0]), &mut pcm, 960)
            .unwrap();
        assert_eq!(decoder.decode_float(None, &mut pcm, 960).unwrap(), 960);
        // Concealment extrapolates the signal, so it is neither silence nor
        // a repeat of the frame before it.
        let energy: f32 = pcm.iter().map(|v| v * v).sum();
        assert!(energy > 0.0, "concealment produced silence");
        assert_eq!(
            decoder.final_range(),
            0,
            "a concealed frame read no symbols"
        );
    }

    /// The output rate is chosen at open time, and everything below 48 kHz
    /// is the same decode decimated.
    #[test]
    fn decodes_at_every_supported_output_rate() {
        for (rate, samples) in [
            (8000, 160),
            (12000, 240),
            (16000, 320),
            (24000, 480),
            (48000, 960),
        ] {
            let mut decoder = Decoder::new(rate, 1).unwrap();
            let mut pcm = vec![0.0f32; samples];
            let got = decoder
                .decode_float(Some(CELT_PACKETS[0]), &mut pcm, samples)
                .unwrap();
            assert_eq!(
                got, samples,
                "{rate} Hz decoded {got} samples, not {samples}"
            );
        }
        assert_eq!(Decoder::new(44100, 1).err(), Some(Error::BadArgument));
        assert_eq!(Decoder::new(48000, 3).err(), Some(Error::BadArgument));
    }

    /// The four framings of the first byte's low two bits, each built by
    /// hand so the parse is checked against the format rather than against
    /// whatever an encoder happened to emit.
    #[test]
    fn parses_every_packet_framing() {
        // Code 0: one frame, the rest of the packet.
        let (frames, offset) = parse_frames(&[0x00, 1, 2, 3], false).unwrap();
        assert_eq!(frames, vec![(1, 3)]);
        assert_eq!(offset, 4);

        // Code 1: two frames of equal size.
        let (frames, _) = parse_frames(&[0x01, 1, 2, 3, 4], false).unwrap();
        assert_eq!(frames, vec![(1, 2), (3, 2)]);
        // An odd remainder cannot be halved.
        assert_eq!(
            parse_frames(&[0x01, 1, 2, 3], false).err(),
            Some(Error::InvalidPacket)
        );

        // Code 2: the first frame's length is coded, the second gets the rest.
        let (frames, _) = parse_frames(&[0x02, 2, 1, 2, 3, 4], false).unwrap();
        assert_eq!(frames, vec![(2, 2), (4, 2)]);

        // Code 3, CBR: a count byte, then frames of equal size.
        let (frames, _) = parse_frames(&[0x03, 3, 1, 2, 3, 4, 5, 6], false).unwrap();
        assert_eq!(frames, vec![(2, 2), (4, 2), (6, 2)]);

        // Code 3, VBR: a count byte with the top bit set, then a length each.
        let (frames, _) = parse_frames(&[0x03, 0x82, 2, 1, 2, 3, 4], false).unwrap();
        assert_eq!(frames, vec![(3, 2), (5, 2)]);

        // Code 3 with padding: the padding is counted in the packet's length
        // but is not part of any frame.
        let (frames, offset) = parse_frames(&[0x03, 0x41, 2, 1, 2, 3, 4], false).unwrap();
        assert_eq!(frames, vec![(3, 2)]);
        assert_eq!(offset, 7);
    }

    /// Inside a multi-stream packet every stream but the last says where it
    /// ends, because the next one starts there.
    #[test]
    fn parses_self_delimited_framing() {
        // Code 0 self-delimited: one coded length, then that many bytes.
        let (frames, offset) = parse_frames(&[0x00, 2, 1, 2, 0xff, 0xff], true).unwrap();
        assert_eq!(frames, vec![(2, 2)]);
        assert_eq!(
            offset, 4,
            "the bytes after the frame belong to the next stream"
        );

        // Code 1 self-delimited: one length, used for both frames.
        let (frames, offset) = parse_frames(&[0x01, 2, 1, 2, 3, 4, 0xff], true).unwrap();
        assert_eq!(frames, vec![(2, 2), (4, 2)]);
        assert_eq!(offset, 6);
    }

    #[test]
    fn refuses_a_packet_that_is_not_one() {
        assert_eq!(parse_frames(&[], false).err(), Some(Error::InvalidPacket));
        // A frame count of zero.
        assert_eq!(
            parse_frames(&[0x03, 0x00, 1], false).err(),
            Some(Error::InvalidPacket)
        );
        // A coded length longer than what is left.
        assert_eq!(
            parse_frames(&[0x02, 9, 1, 2], false).err(),
            Some(Error::InvalidPacket)
        );
    }

    /// A multi-stream decoder plays each stream out to the channels its
    /// mapping names, and leaves a channel mapped to 255 silent.
    #[test]
    fn multistream_maps_streams_onto_channels() {
        // Two mono streams, the second duplicated onto two channels and one
        // channel muted.
        let mapping = [0u8, 1, 1, 255];
        let mut decoder = MultiStreamDecoder::new(48000, 4, 2, 0, &mapping).unwrap();
        // A packet holding both streams: the first self-delimited, then the
        // second. Both are the same CELT frame.
        let stream = CELT_PACKETS[0];
        let mut packet = vec![stream[0]];
        let len = stream.len() - 1;
        if len < 252 {
            packet.push(len as u8);
        } else {
            packet.push(252 + (len % 4) as u8);
            packet.push((len / 4) as u8);
        }
        packet.extend_from_slice(&stream[1..]);
        packet.extend_from_slice(stream);

        let mut pcm = vec![0.0f32; 960 * 4];
        assert_eq!(
            decoder.decode_float(Some(&packet), &mut pcm, 960).unwrap(),
            960
        );
        for i in 0..960 {
            assert_eq!(
                pcm[i * 4 + 1],
                pcm[i * 4 + 2],
                "both copies of stream 1 differ"
            );
            assert_eq!(pcm[i * 4 + 3], 0.0, "a channel mapped to 255 is not silent");
        }
    }

    /// A guest can hand this decoder anything at all, so nothing it is handed
    /// may panic. This throws random bytes and corrupted real packets at it
    /// and only asks that it come back — with samples or with an error, but
    /// without taking the emulator down.
    #[test]
    fn survives_arbitrary_input() {
        let mut seed = 0x1234_5678u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed
        };
        let mut decoder = Decoder::new(48000, 2).unwrap();
        let mut pcm = vec![0.0f32; 5760 * 2];

        for _ in 0..400 {
            let len = (next() as usize % 300) + 1;
            let packet: Vec<u8> = (0..len).map(|_| (next() >> 16) as u8).collect();
            let _ = decoder.decode_float(Some(&packet), &mut pcm, 5760);
        }

        // Real packets with one byte flipped, which is the case that gets
        // furthest into the decoder before anything looks wrong.
        for &original in CELT_PACKETS
            .iter()
            .chain(SILK_PACKETS.iter())
            .chain(HYBRID_PACKETS.iter())
        {
            for _ in 0..200 {
                let mut packet = original.to_vec();
                let at = next() as usize % packet.len();
                packet[at] ^= (next() >> 16) as u8;
                let _ = decoder.decode_float(Some(&packet), &mut pcm, 5760);
            }
        }

        // And truncations, which is what a stream cut mid-packet looks like.
        for &original in CELT_PACKETS.iter().chain(HYBRID_PACKETS.iter()) {
            for len in 1..original.len() {
                let _ = decoder.decode_float(Some(&original[..len]), &mut pcm, 5760);
            }
        }

        // A multi-stream decoder has more to get wrong: the sub-packets are
        // sized from the bytes themselves.
        let mut multi = MultiStreamDecoder::new(48000, 6, 4, 2, &[0, 1, 2, 3, 4, 5]).unwrap();
        let mut pcm = vec![0.0f32; 5760 * 6];
        for _ in 0..400 {
            let len = (next() as usize % 600) + 1;
            let packet: Vec<u8> = (0..len).map(|_| (next() >> 16) as u8).collect();
            let _ = multi.decode_float(Some(&packet), &mut pcm, 5760);
        }
    }
}
