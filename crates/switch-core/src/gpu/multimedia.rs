//! The video engines nvdec (decode) and VIC (compose/scale), behind host1x via
//! `/dev/nvhost-nvdec` and `/dev/nvhost-vic`. Engine methods go through THI's
//! `METHOD0`/`METHOD1`. Work retires as the stream reaches each syncpoint increment.

use crate::gpu::h264;
use crate::gpu::nvmap::NvMap;
use crate::gpu::syncpt::Host1x;
use crate::mem::Memory;
use crate::trace::{enabled, Trace};
use crate::vp9;
use crate::Result;
use std::collections::HashMap;
use std::sync::Arc;

/// host1x's own class, for syncpoint waits and loads.
const CLASS_HOST1X: u32 = 0x01;

/// THI registers every engine class shares, in words.
const THI_INCR_SYNCPT: u32 = 0x00;
const THI_METHOD0: u32 = 0x10;
const THI_METHOD1: u32 = 0x11;

/// VIC's methods reach byte offset 0x1000.
const METHOD_FILE_WORDS: usize = 0x400;

/// nvdec `EXECUTE`: decode the frame described by the methods before it.
const NVDEC_EXECUTE: u32 = 0xC0;

const NVDEC_CODEC_H264: u32 = 3;
const NVDEC_CODEC_VP9: u32 = 9;

/// nvdec frame methods, as word offsets; buffer addresses are device addresses >> 8.
const NVDEC_SET_APPLICATION_ID: u32 = 0x80;
const NVDEC_SET_DRV_PIC_SETUP_OFFSET: u32 = 0x101;
const NVDEC_SET_IN_BUF_BASE_OFFSET: u32 = 0x102;
/// Seventeen surfaces each: for VP9 last, golden, alternate, then the current frame.
const NVDEC_SET_PICTURE_LUMA_OFFSET: u32 = 0x10C;
const NVDEC_SET_PICTURE_CHROMA_OFFSET: u32 = 0x11D;
const NVDEC_VP9_SET_PROB_TAB_BUF_OFFSET: u32 = 0x170;
const NVDEC_VP9_SET_CTX_COUNTER_BUF_OFFSET: u32 = 0x171;

const VP9_PICTURE_BITSTREAM_SIZE: u32 = 0x30;
const PICTURE_SETUP_TRACE_BYTES: u32 = 0x100;

const VIC_EXECUTE: u32 = 0xC0;
const VIC_SET_CONFIG_STRUCT_OFFSET: u32 = 0x1C2;
const VIC_SET_OUTPUT_LUMA_OFFSET: u32 = 0x1C8;
const VIC_SET_OUTPUT_CHROMA_OFFSET: u32 = 0x1C9;
const VIC_SLOT0_LUMA_OFFSET: u32 = 0x100;
/// Where the output surface's description sits in VIC's configuration struct.
const VIC_CONFIG_OUTPUT_SURFACE: u32 = 0x20;
/// `Y8___U8V8_N420`: two planes, the second an interleaved chroma plane at half size.
const VIC_PIXEL_FORMAT_NV12: u32 = 0x44;

/// Sizes of the records after `SUBMIT`'s header, in bytes.
const SUBMIT_HEADER: usize = 0x10;
const SUBMIT_CMDBUF: usize = 12;
const SUBMIT_RELOC: usize = 16;
const SUBMIT_RELOC_SHIFT: usize = 4;
const SUBMIT_SYNCPT_INCR: usize = 8;
const SUBMIT_FENCE: usize = 4;

/// `MAP_BUFFER`'s header and entry sizes, in bytes.
const MAP_HEADER: usize = 12;
const MAP_ENTRY: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    Nvdec,
    Vic,
}

impl Engine {
    pub fn from_node(path: &str) -> Option<Engine> {
        match path {
            "/dev/nvhost-nvdec" => Some(Engine::Nvdec),
            "/dev/nvhost-vic" => Some(Engine::Vic),
            _ => None,
        }
    }

    pub fn node(self) -> &'static str {
        match self {
            Engine::Nvdec => "/dev/nvhost-nvdec",
            Engine::Vic => "/dev/nvhost-vic",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Engine::Nvdec => "nvdec",
            Engine::Vic => "vic",
        }
    }

    fn class(self) -> u32 {
        match self {
            Engine::Nvdec => 0xF0,
            Engine::Vic => 0x5D,
        }
    }

    fn execute_method(self) -> u32 {
        match self {
            Engine::Nvdec => NVDEC_EXECUTE,
            Engine::Vic => VIC_EXECUTE,
        }
    }
}

#[derive(Debug)]
pub struct MmChannel {
    pub engine: Engine,
    pub syncpt: u32,
    method: u32,
    methods: Vec<u32>,
    pub submits: u64,
    pub executes: u64,
    vp9: vp9::nvdec::Channel,
}

/// Decoded frames by the address of their luma surface: what nvdec "writes" and VIC reads.
type Surfaces = HashMap<u32, Arc<vp9::Frame>>;

/// A host's H.264 decoder. Pictures go in by the surface nvdec writes and come back in any
/// order, `None` for one it could not decode.
pub trait H264Decoder: std::fmt::Debug {
    fn decode(&mut self, surface: u32, access_unit: &[u8], key: bool);
    fn take(&mut self) -> Vec<(u32, Option<vp9::Frame>)>;
}

/// How many times VIC is asked again for one frame before it composes without it.
const WAIT_LIMIT: u32 = 1000;

/// What nvdec has written, or is still writing, as VIC finds it.
#[derive(Debug, Default)]
struct Frames {
    surfaces: Surfaces,
    h264: Option<Box<dyn H264Decoder>>,
    /// Surfaces the H.264 decoder has yet to hand back, and how often VIC has waited for each.
    pending: HashMap<u32, u32>,
    /// The parameter sets the decoder was last given.
    parameter_sets: Vec<u8>,
}

impl Frames {
    fn land(&mut self) {
        let Some(decoder) = self.h264.as_mut() else {
            return;
        };
        for (surface, frame) in decoder.take() {
            self.pending.remove(&surface);
            match frame {
                Some(frame) => self.surfaces.insert(surface, Arc::new(frame)),
                None => self.surfaces.remove(&surface),
            };
        }
    }

    /// Whether VIC should wait for `surface`, counting the wait.
    fn waiting_for(&mut self, surface: u32) -> bool {
        let Some(waits) = self.pending.get_mut(&surface) else {
            return false;
        };
        *waits += 1;
        if *waits <= WAIT_LIMIT {
            return true;
        }
        self.pending.remove(&surface);
        crate::traceln!("[video] the H.264 picture for surface {surface:#x} never came back");
        false
    }
}

/// How a submit went: run, refused as malformed, or to be asked again once a frame it reads lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Submitted {
    Done,
    Refused,
    Waiting,
}

impl MmChannel {
    fn new(engine: Engine, syncpt: u32) -> MmChannel {
        MmChannel {
            engine,
            syncpt,
            method: 0,
            methods: vec![0; METHOD_FILE_WORDS],
            submits: 0,
            executes: 0,
            vp9: Default::default(),
        }
    }

    pub fn method(&self, method: u32) -> u32 {
        self.methods.get(method as usize).copied().unwrap_or(0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Write {
    class: u32,
    offset: u32,
    value: u32,
}

/// Every write a host1x stream makes and the first opcode it can't follow.
/// `GATHER` is cut short and reported.
fn decode_stream(words: &[u32], class: u32) -> (Vec<Write>, Option<u32>) {
    let mut writes = Vec::new();
    let mut class = class;
    let mut at = 0;
    let next = |at: &mut usize| {
        let word = words.get(*at).copied();
        *at += 1;
        word
    };
    while let Some(word) = next(&mut at) {
        let offset = (word >> 16) & 0xFFF;
        match word >> 28 {
            // SETCL
            0x0 => {
                class = (word >> 6) & 0x3FF;
                let mask = word & 0x3F;
                for bit in 0..6 {
                    if mask & (1 << bit) != 0 {
                        let Some(value) = next(&mut at) else { break };
                        writes.push(Write {
                            class,
                            offset: offset + bit,
                            value,
                        });
                    }
                }
            }
            // INCR / NONINCR
            0x1 | 0x2 => {
                let incrementing = word >> 28 == 0x1;
                for i in 0..word & 0xFFFF {
                    let Some(value) = next(&mut at) else { break };
                    let offset = if incrementing { offset + i } else { offset };
                    writes.push(Write {
                        class,
                        offset,
                        value,
                    });
                }
            }
            // MASK
            0x3 => {
                for bit in 0..16 {
                    if word & (1 << bit) != 0 {
                        let Some(value) = next(&mut at) else { break };
                        writes.push(Write {
                            class,
                            offset: offset + bit,
                            value,
                        });
                    }
                }
            }
            // IMM
            0x4 => writes.push(Write {
                class,
                offset,
                value: word & 0xFFFF,
            }),
            // RESTART: nothing to restart in a buffer.
            0x5 => {}
            // EXTEND: MLOCK acquire/release.
            0xE => {}
            _ => return (writes, Some(word)),
        }
    }
    (writes, None)
}

#[derive(Debug, Default)]
pub struct Video {
    channels: HashMap<u32, MmChannel>,
    next_channel: u32,
    frames: Frames,
}

impl Video {
    pub fn open(&mut self, engine: Engine, host1x: &mut Host1x) -> Result<u32> {
        let syncpt = host1x.allocate()?;
        let id = self.next_channel;
        self.next_channel += 1;
        self.channels.insert(id, MmChannel::new(engine, syncpt));
        if enabled(Trace::Video) {
            crate::traceln!(
                "[video] {} channel {id} opened, syncpoint {syncpt}",
                engine.name()
            );
        }
        Ok(id)
    }

    pub fn close(&mut self, id: u32, host1x: &mut Host1x) {
        if let Some(channel) = self.channels.remove(&id) {
            host1x.release(channel.syncpt);
        }
    }

    pub fn channel(&self, id: u32) -> Option<&MmChannel> {
        self.channels.get(&id)
    }

    pub fn set_h264(&mut self, decoder: Box<dyn H264Decoder>) {
        self.frames.h264 = Some(decoder);
    }

    /// `SUBMIT`: run the command buffers and write each increment's fence threshold
    /// back. `args` includes every record after the header.
    pub fn submit(
        &mut self,
        id: u32,
        args: &mut [u8],
        mem: &Memory,
        nvmap: &NvMap,
        host1x: &mut Host1x,
    ) -> Result<Submitted> {
        let Video {
            channels, frames, ..
        } = self;
        let Some(channel) = channels.get_mut(&id) else {
            return Ok(Submitted::Refused);
        };
        let count = |at: usize| read_u32(args, at) as usize;
        let (cmdbufs, relocs, incrs, fences) = (count(0), count(4), count(8), count(12));
        let relocs_at = SUBMIT_HEADER + cmdbufs * SUBMIT_CMDBUF;
        let shifts_at = relocs_at + relocs * SUBMIT_RELOC;
        let incrs_at = shifts_at + relocs * SUBMIT_RELOC_SHIFT;
        let fences_at = incrs_at + incrs * SUBMIT_SYNCPT_INCR;
        let end = fences_at + fences * SUBMIT_FENCE;
        if end > args.len() {
            crate::traceln!(
                "[video] {} submit names {cmdbufs} command buffers, {relocs} relocations, \
                 {incrs} increments and {fences} fences, {end} bytes, in {} bytes of arguments",
                channel.engine.name(),
                args.len()
            );
            return Ok(Submitted::Refused);
        }

        let mut streams = Vec::with_capacity(cmdbufs);
        for i in 0..cmdbufs {
            let at = SUBMIT_HEADER + i * SUBMIT_CMDBUF;
            let (handle, offset, words) = (
                read_u32(args, at),
                read_u32(args, at + 4),
                read_u32(args, at + 8),
            );
            let Some(base) = handle_address(nvmap, handle) else {
                crate::traceln!(
                    "[video] {} command buffer {i} names nvmap handle {handle}, which is not allocated",
                    channel.engine.name()
                );
                continue;
            };
            let mut stream = Vec::with_capacity(words as usize);
            for word in 0..words {
                stream.push(mem.read_u32(base.wrapping_add(offset).wrapping_add(word * 4))?);
            }
            for r in 0..relocs {
                let at = relocs_at + r * SUBMIT_RELOC;
                let (buffer, buffer_offset, target, target_offset) = (
                    read_u32(args, at),
                    read_u32(args, at + 4),
                    read_u32(args, at + 8),
                    read_u32(args, at + 12),
                );
                let shift = read_u32(args, shifts_at + r * SUBMIT_RELOC_SHIFT);
                let word = buffer_offset.wrapping_sub(offset) / 4;
                if buffer != handle || buffer_offset < offset || word >= words {
                    continue;
                }
                if let Some(address) = handle_address(nvmap, target) {
                    stream[word as usize] = address.wrapping_add(target_offset) >> shift;
                }
            }
            streams.push(stream);
        }

        if channel.engine == Engine::Vic {
            frames.land();
            let source = streams
                .iter()
                .find_map(|stream| slot_luma(stream, channel.method));
            if source.is_some_and(|surface| frames.waiting_for(surface)) {
                return Ok(Submitted::Waiting);
            }
        }
        channel.submits += 1;

        // Reserve every increment first so each fence reports the final threshold.
        let mut thresholds = Vec::with_capacity(incrs);
        for i in 0..incrs {
            let at = incrs_at + i * SUBMIT_SYNCPT_INCR;
            let (syncpt, increments) = (read_u32(args, at), read_u32(args, at + 4));
            thresholds.push((syncpt, host1x.incr_max(syncpt, increments)?));
        }
        for stream in &streams {
            channel.run(stream, mem, host1x, frames)?;
        }

        // Force each syncpoint to its threshold so no fence is left unreachable.
        for (i, &(syncpt, threshold)) in thresholds.iter().enumerate() {
            if !host1x.is_expired(syncpt, threshold)? {
                if enabled(Trace::Video) {
                    crate::traceln!(
                        "[video] {} submit left syncpoint {syncpt} short of {threshold}; retiring it",
                        channel.engine.name()
                    );
                }
                host1x.set(syncpt, threshold)?;
            }
            if i < fences {
                write_u32(args, fences_at + i * SUBMIT_FENCE, threshold);
            }
        }
        Ok(Submitted::Done)
    }
}

/// The surface a VIC stream composes from slot 0, if it sets one; `method` is the method a
/// `METHOD1` names before the stream sets another.
fn slot_luma(stream: &[u32], mut method: u32) -> Option<u32> {
    let (writes, _) = decode_stream(stream, Engine::Vic.class());
    let mut surface = None;
    for write in writes.iter().filter(|w| w.class == Engine::Vic.class()) {
        match write.offset {
            THI_METHOD0 => method = write.value,
            THI_METHOD1 if method == VIC_SLOT0_LUMA_OFFSET => surface = Some(write.value << 8),
            _ => {}
        }
    }
    surface
}

impl MmChannel {
    fn run(
        &mut self,
        stream: &[u32],
        mem: &Memory,
        host1x: &mut Host1x,
        frames: &mut Frames,
    ) -> Result<()> {
        let (writes, stopped) = decode_stream(stream, self.engine.class());
        for write in writes {
            match (write.class, write.offset) {
                (_, THI_INCR_SYNCPT) => {
                    host1x.increment(write.value & 0xFF)?;
                }
                (class, THI_METHOD0) if class == self.engine.class() => self.method = write.value,
                (class, THI_METHOD1) if class == self.engine.class() => {
                    self.write_method(self.method, write.value, mem, frames)
                }
                (CLASS_HOST1X, offset) => {
                    if enabled(Trace::Video) {
                        crate::traceln!(
                            "[video] {} host1x register {offset:#x} = {:#x}",
                            self.engine.name(),
                            write.value
                        );
                    }
                }
                (class, offset) => {
                    if enabled(Trace::Video) {
                        crate::traceln!(
                            "[video] {} class {class:#x} register {offset:#x} = {:#x}",
                            self.engine.name(),
                            write.value
                        );
                    }
                }
            }
        }
        if let Some(word) = stopped {
            crate::traceln!(
                "[video] {} command stream stopped at opcode word {word:#010x}",
                self.engine.name()
            );
        }
        Ok(())
    }

    fn write_method(&mut self, method: u32, value: u32, mem: &Memory, frames: &mut Frames) {
        if let Some(slot) = self.methods.get_mut(method as usize) {
            *slot = value;
        }
        if enabled(Trace::Video) {
            crate::traceln!(
                "[video] {} method {method:#x} = {value:#x}",
                self.engine.name()
            );
        }
        if method == self.engine.execute_method() {
            self.executes += 1;
            if self.engine == Engine::Nvdec && enabled(Trace::Video) {
                self.trace_decode(mem);
            }
            match self.engine {
                Engine::Nvdec if self.method(NVDEC_SET_APPLICATION_ID) == NVDEC_CODEC_VP9 => {
                    self.decode_vp9(mem, &mut frames.surfaces)
                }
                Engine::Nvdec if self.method(NVDEC_SET_APPLICATION_ID) == NVDEC_CODEC_H264 => {
                    self.decode_h264(mem, frames)
                }
                Engine::Vic => self.compose(mem, &frames.surfaces),
                _ => {}
            }
        }
    }

    /// Decodes the VP9 frame an nvdec `EXECUTE` describes and keeps it for VIC under its surface.
    fn decode_vp9(&mut self, mem: &Memory, surfaces: &mut Surfaces) {
        const SETUP_BYTES: usize = 0x100;
        const PROBABILITY_BYTES: usize = 3744;
        let read = |at: u32, len: usize| {
            let mut buf = vec![0u8; len];
            let _ = mem.read_into(at, &mut buf);
            buf
        };
        let setup_at = self.address(NVDEC_SET_DRV_PIC_SETUP_OFFSET);
        let setup = read(setup_at, SETUP_BYTES);
        let size = u32::from_le_bytes(
            setup[VP9_PICTURE_BITSTREAM_SIZE as usize..VP9_PICTURE_BITSTREAM_SIZE as usize + 4]
                .try_into()
                .unwrap(),
        ) as usize;
        let bitstream = read(self.address(NVDEC_SET_IN_BUF_BASE_OFFSET), size);
        let probs = read(
            self.address(NVDEC_VP9_SET_PROB_TAB_BUF_OFFSET),
            PROBABILITY_BYTES,
        );
        let slots = [0, 1, 2, 3].map(|i| self.address(NVDEC_SET_PICTURE_LUMA_OFFSET + i));
        let refs = [0, 1, 2].map(|i| surfaces.get(&slots[i]).cloned());
        match self.vp9.decode(&setup, &bitstream, &probs, refs) {
            Ok(frame) => {
                surfaces.insert(slots[3], frame);
            }
            Err(e) => {
                if enabled(Trace::Video) {
                    crate::traceln!(
                        "[video] nvdec VP9 frame {} not decoded: {e:?}",
                        self.executes
                    );
                }
            }
        }
    }

    /// Hands the H.264 picture an nvdec `EXECUTE` describes to the host's decoder, behind the
    /// parameter sets its setup was filled from.
    fn decode_h264(&mut self, mem: &Memory, frames: &mut Frames) {
        /// Past any picture's size, so a corrupt length reads no further.
        const MAX_STREAM: u32 = 16 << 20;
        let Frames {
            h264: Some(decoder),
            pending,
            surfaces,
            parameter_sets,
        } = frames
        else {
            return;
        };
        let read = |at: u32, len: usize| {
            let mut buf = vec![0u8; len];
            let _ = mem.read_into(at, &mut buf);
            buf
        };
        let picture = h264::Picture::parse(&read(
            self.address(NVDEC_SET_DRV_PIC_SETUP_OFFSET),
            h264::SETUP_BYTES,
        ));
        let bitstream = read(
            self.address(NVDEC_SET_IN_BUF_BASE_OFFSET),
            picture.stream_len.min(MAX_STREAM) as usize,
        );
        let surface = self.address(NVDEC_SET_PICTURE_LUMA_OFFSET + picture.current_index);
        let refused = |why: &str| {
            if enabled(Trace::Video) {
                crate::traceln!(
                    "[video] nvdec H.264 picture {} not decoded: {why}",
                    self.executes
                );
            }
        };
        let Some(unit) = h264::access_unit(&bitstream) else {
            return refused("no slice in its bitstream");
        };
        let sets = match h264::parameter_sets(&picture, unit.pps_id) {
            Ok(sets) => sets,
            Err(why) => return refused(&why),
        };
        let mut bytes = Vec::with_capacity(sets.len() + unit.bytes.len());
        if unit.key || sets != *parameter_sets {
            bytes.extend_from_slice(&sets);
            *parameter_sets = sets;
        }
        bytes.extend_from_slice(&unit.bytes);
        surfaces.remove(&surface);
        pending.insert(surface, 0);
        decoder.decode(surface, &bytes, unit.key);
    }

    /// VIC reduced to what the game asks of it: the decoded frame in slot 0 written to the output
    /// surfaces as pitch-linear NV12, scaled if the sizes differ.
    fn compose(&mut self, mem: &Memory, surfaces: &Surfaces) {
        let config = self.address(VIC_SET_CONFIG_STRUCT_OFFSET);
        let word = |at: u32| mem.read_u32(config.wrapping_add(at)).unwrap_or(0);
        let out = VIC_CONFIG_OUTPUT_SURFACE;
        let format = word(out) & 0x7f;
        let block_kind = word(out) >> 11 & 0xf;
        let width = (word(out + 8) & 0x3fff) + 1;
        let height = (word(out + 8) >> 14 & 0x3fff) + 1;
        let Some(frame) = surfaces.get(&self.address(VIC_SLOT0_LUMA_OFFSET)) else {
            return;
        };
        if format != VIC_PIXEL_FORMAT_NV12 || block_kind != 0 {
            if enabled(Trace::Video) {
                crate::traceln!(
                    "[video] vic output format {format:#x} kind {block_kind} is not composed"
                );
            }
            return;
        }
        let (width, height) = (width as usize, height as usize);
        let luma_pitch = width.next_multiple_of(VIC_PITCH_ALIGNMENT);
        let chroma_pitch = (width.div_ceil(2) * 2).next_multiple_of(VIC_PITCH_ALIGNMENT);
        let (luma, chroma) = nv12(frame, width, height, luma_pitch, chroma_pitch);
        let _ = mem.write_bytes(self.address(VIC_SET_OUTPUT_LUMA_OFFSET), &luma);
        let _ = mem.write_bytes(self.address(VIC_SET_OUTPUT_CHROMA_OFFSET), &chroma);
    }

    fn address(&self, method: u32) -> u32 {
        self.method(method) << 8
    }

    /// Trace one nvdec `EXECUTE`'s codec, buffers, picture setup and bitstream head.
    fn trace_decode(&self, mem: &Memory) {
        let setup = self.address(NVDEC_SET_DRV_PIC_SETUP_OFFSET);
        let bitstream = self.address(NVDEC_SET_IN_BUF_BASE_OFFSET);
        let size = mem
            .read_u32(setup.wrapping_add(VP9_PICTURE_BITSTREAM_SIZE))
            .unwrap_or(0);
        let surfaces: Vec<String> = (0..4)
            .map(|i| {
                format!(
                    "{:#x}/{:#x}",
                    self.address(NVDEC_SET_PICTURE_LUMA_OFFSET + i),
                    self.address(NVDEC_SET_PICTURE_CHROMA_OFFSET + i)
                )
            })
            .collect();
        crate::traceln!(
            "[video] nvdec decode {}: codec {}, setup {setup:#x}, bitstream {bitstream:#x} \
             ({size} bytes), surfaces {}, probabilities {:#x}, counters {:#x}",
            self.executes,
            self.method(NVDEC_SET_APPLICATION_ID),
            surfaces.join(" "),
            self.address(NVDEC_VP9_SET_PROB_TAB_BUF_OFFSET),
            self.address(NVDEC_VP9_SET_CTX_COUNTER_BUF_OFFSET),
        );
        let words = |at: u32, count: u32| -> String {
            (0..count)
                .map(|i| format!("{:08x}", mem.read_u32(at.wrapping_add(4 * i)).unwrap_or(0)))
                .collect::<Vec<_>>()
                .join(" ")
        };
        for row in (0..PICTURE_SETUP_TRACE_BYTES).step_by(32) {
            crate::traceln!(
                "[video]   setup +{row:#04x}: {}",
                words(setup.wrapping_add(row), 8)
            );
        }
        crate::traceln!("[video]   bitstream: {}", words(bitstream, 8));
        crate::traceln!(
            "[video]   before it: {}",
            words(bitstream.wrapping_sub(32), 8)
        );
    }
}

const VIC_PITCH_ALIGNMENT: usize = 64;

/// The frame as two NV12 planes of `width`x`height`, bilinearly scaled when it is another size.
fn nv12(
    frame: &vp9::Frame,
    width: usize,
    height: usize,
    luma_pitch: usize,
    chroma_pitch: usize,
) -> (Vec<u8>, Vec<u8>) {
    let sample = |plane: &vp9::Plane, x: f32, y: f32| -> u8 {
        let x = x.clamp(0.0, (plane.width - 1) as f32);
        let y = y.clamp(0.0, (plane.height - 1) as f32);
        let (x0, y0) = (x as usize, y as usize);
        let (x1, y1) = (
            (x0 + 1).min(plane.width - 1),
            (y0 + 1).min(plane.height - 1),
        );
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let at = |x: usize, y: usize| f32::from(plane.data[y * plane.stride + x]);
        let top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
        let bottom = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
        (top * (1.0 - fy) + bottom * fy + 0.5) as u8
    };
    let scaled = |plane: &vp9::Plane, x: usize, y: usize, w: usize, h: usize| -> u8 {
        if plane.width == w && plane.height == h {
            return plane.data[y * plane.stride + x];
        }
        let sx = (x as f32 + 0.5) * plane.width as f32 / w as f32 - 0.5;
        let sy = (y as f32 + 0.5) * plane.height as f32 / h as f32 - 0.5;
        sample(plane, sx, sy)
    };
    let (chroma_w, chroma_h) = (width.div_ceil(2), height.div_ceil(2));
    let mut luma = vec![0u8; luma_pitch * height];
    for y in 0..height {
        for x in 0..width {
            luma[y * luma_pitch + x] = scaled(&frame.planes[0], x, y, width, height);
        }
    }
    let mut chroma = vec![0u8; chroma_pitch * chroma_h];
    for y in 0..chroma_h {
        for x in 0..chroma_w {
            chroma[y * chroma_pitch + 2 * x] = scaled(&frame.planes[1], x, y, chroma_w, chroma_h);
            chroma[y * chroma_pitch + 2 * x + 1] =
                scaled(&frame.planes[2], x, y, chroma_w, chroma_h);
        }
    }
    (luma, chroma)
}

/// `MAP_BUFFER`: engines address guest memory directly, so each handle maps to its own address.
pub fn map_buffer(args: &mut [u8], nvmap: &NvMap) -> bool {
    let entries = read_u32(args, 0) as usize;
    if MAP_HEADER + entries * MAP_ENTRY > args.len() {
        return false;
    }
    for i in 0..entries {
        let at = MAP_HEADER + i * MAP_ENTRY;
        let address = handle_address(nvmap, read_u32(args, at)).unwrap_or(0);
        write_u32(args, at + 4, address);
    }
    true
}

/// An nvmap handle's address. A driver may pass a buffer id instead; they never collide.
fn handle_address(nvmap: &NvMap, handle: u32) -> Option<u32> {
    nvmap
        .get(handle)
        .or_else(|| nvmap.by_id(handle))
        .filter(|h| h.cpu_addr != 0)
        .map(|h| h.cpu_addr)
}

fn read_u32(data: &[u8], at: usize) -> u32 {
    data.get(at..at + 4)
        .map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn write_u32(data: &mut [u8], at: usize, value: u32) {
    if let Some(slot) = data.get_mut(at..at + 4) {
        slot.copy_from_slice(&value.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nv12_packs_two_planes_with_the_requested_pitches() {
        let mut frame = vp9::Frame::new(4, 2);
        frame.planes[0].data[..4].copy_from_slice(&[10, 20, 30, 40]);
        frame.planes[0].data[frame.planes[0].stride..][..4].copy_from_slice(&[50, 60, 70, 80]);
        frame.planes[1].data[..2].copy_from_slice(&[1, 2]);
        frame.planes[2].data[..2].copy_from_slice(&[3, 4]);
        let (luma, chroma) = nv12(&frame, 4, 2, 8, 8);
        assert_eq!(&luma[..4], &[10, 20, 30, 40]);
        assert_eq!(&luma[8..12], &[50, 60, 70, 80]);
        assert_eq!(&chroma[..4], &[1, 3, 2, 4]);
    }

    #[test]
    fn nv12_scales_to_the_output_size() {
        let mut frame = vp9::Frame::new(2, 2);
        for y in 0..2 {
            frame.planes[0].data[y * frame.planes[0].stride..][..2].copy_from_slice(&[0, 100]);
        }
        let (luma, _) = nv12(&frame, 4, 2, 4, 4);
        assert_eq!(luma[0], 0);
        assert!(luma[1] > 0 && luma[1] < luma[2] && luma[3] == 100);
    }

    #[test]
    fn a_stream_selects_a_class_and_writes_its_methods() {
        let words = [
            // SETCL nvdec, no masked writes.
            (0xF0 << 6),
            // INCR at METHOD0, two words: METHOD0 then METHOD1.
            0x1000_0000 | (THI_METHOD0 << 16) | 2,
            0x80,
            9,
            // IMM an increment of syncpoint 12.
            0x4000_0000 | (THI_INCR_SYNCPT << 16) | 12,
            // NONINCR METHOD1 twice.
            0x2000_0000 | (THI_METHOD1 << 16) | 2,
            1,
            2,
            // MASK bits 0 and 1 from METHOD0.
            0x3000_0000 | (THI_METHOD0 << 16) | 0b11,
            NVDEC_EXECUTE,
            0,
        ];
        let (writes, stopped) = decode_stream(&words, CLASS_HOST1X);
        assert_eq!(stopped, None);
        let at = |offset, value| Write {
            class: 0xF0,
            offset,
            value,
        };
        assert_eq!(
            writes,
            [
                at(THI_METHOD0, 0x80),
                at(THI_METHOD1, 9),
                at(THI_INCR_SYNCPT, 12),
                at(THI_METHOD1, 1),
                at(THI_METHOD1, 2),
                at(THI_METHOD0, NVDEC_EXECUTE),
                at(THI_METHOD1, 0),
            ]
        );
    }

    #[test]
    fn a_stream_is_cut_short_at_an_opcode_it_cannot_follow() {
        let gather = 0x6000_0000;
        let (writes, stopped) = decode_stream(&[0x4000_0005, gather, 0x4000_0006], 0xF0);
        assert_eq!(writes.len(), 1);
        assert_eq!(stopped, Some(gather));
    }

    #[test]
    fn a_submit_runs_its_stream_and_reports_where_its_fence_lands() {
        let mem = Memory::new();
        mem.map_zero(0x1000_0000, 0x1000).unwrap();
        let mut nvmap = NvMap::new();
        let handle = nvmap.create(0x1000);
        nvmap.alloc(handle, 0, 0, 0x1000, 0, 0x1000_0000).unwrap();
        let mut host1x = Host1x::new();
        let mut video = Video::default();
        let id = video.open(Engine::Nvdec, &mut host1x).unwrap();
        let syncpt = video.channel(id).unwrap().syncpt;

        let stream = [
            0x1000_0000 | (THI_METHOD0 << 16) | 2,
            NVDEC_EXECUTE,
            0,
            0x4000_0000 | (THI_INCR_SYNCPT << 16) | syncpt,
        ];
        for (i, word) in stream.iter().enumerate() {
            mem.write_u32(0x1000_0000 + 4 * i as u32, *word).unwrap();
        }
        let mut args = vec![0u8; SUBMIT_HEADER + SUBMIT_CMDBUF + SUBMIT_SYNCPT_INCR + SUBMIT_FENCE];
        for (at, value) in [
            (0, 1),
            (4, 0),
            (8, 1),
            (12, 1),
            (0x10, handle),
            (0x14, 0),
            (0x18, stream.len() as u32),
            (0x1C, syncpt),
            (0x20, 1),
        ] {
            write_u32(&mut args, at, value);
        }

        assert_eq!(
            video
                .submit(id, &mut args, &mem, &nvmap, &mut host1x)
                .unwrap(),
            Submitted::Done
        );
        let threshold = read_u32(&args, 0x24);
        assert_eq!(threshold, 1);
        assert!(host1x.is_expired(syncpt, threshold).unwrap());
        assert_eq!(
            host1x.read(syncpt).unwrap(),
            1,
            "incremented once, by the stream"
        );
        let channel = video.channel(id).unwrap();
        assert_eq!(channel.executes, 1);
        assert_eq!(channel.method(NVDEC_EXECUTE), 0);
    }

    type Shared<T> = std::rc::Rc<std::cell::RefCell<Vec<T>>>;

    /// A decoder whose pictures come back only when the test lands them.
    #[derive(Debug, Default)]
    struct Deferred {
        given: Shared<(u32, Vec<u8>, bool)>,
        ready: Shared<(u32, Option<vp9::Frame>)>,
    }

    impl H264Decoder for Deferred {
        fn decode(&mut self, surface: u32, access_unit: &[u8], key: bool) {
            self.given
                .borrow_mut()
                .push((surface, access_unit.to_vec(), key));
        }

        fn take(&mut self) -> Vec<(u32, Option<vp9::Frame>)> {
            std::mem::take(&mut self.ready.borrow_mut())
        }
    }

    /// Methods through THI, then the channel's syncpoint once done.
    fn method_stream(methods: &[(u32, u32)], syncpt: u32) -> Vec<u32> {
        let mut stream = Vec::new();
        for &(method, value) in methods {
            stream.extend([0x1000_0000 | (THI_METHOD0 << 16) | 2, method, value]);
        }
        stream.push(0x4000_0000 | (THI_INCR_SYNCPT << 16) | 0x100 | syncpt);
        stream
    }

    /// Submit `stream` from guest memory at `at` with one increment, returning how it went and
    /// the fence it reported.
    fn submit_stream(
        video: &mut Video,
        id: u32,
        stream: &[u32],
        (mem, nvmap, handle): (&Memory, &NvMap, u32),
        host1x: &mut Host1x,
    ) -> (Submitted, u32) {
        for (i, word) in stream.iter().enumerate() {
            mem.write_u32(0x1000_0000 + 4 * i as u32, *word).unwrap();
        }
        let syncpt = video.channel(id).unwrap().syncpt;
        let mut args = vec![0u8; SUBMIT_HEADER + SUBMIT_CMDBUF + SUBMIT_SYNCPT_INCR + SUBMIT_FENCE];
        for (at, value) in [
            (0, 1),
            (8, 1),
            (12, 1),
            (0x10, handle),
            (0x18, stream.len() as u32),
            (0x1C, syncpt),
            (0x20, 1),
        ] {
            write_u32(&mut args, at, value);
        }
        let submitted = video.submit(id, &mut args, mem, nvmap, host1x).unwrap();
        (submitted, read_u32(&args, 0x24))
    }

    /// An H.264 picture reaches the host's decoder behind its parameter sets, and VIC waits for
    /// it, without spending an increment, until the decoder hands it back.
    #[test]
    fn vic_waits_for_the_h264_picture_it_composes() {
        const BASE: u32 = 0x1000_0000;
        const SETUP: u32 = BASE + 0x1000;
        const BITSTREAM: u32 = BASE + 0x2000;
        const CONFIG: u32 = BASE + 0x3000;
        const OUTPUT: u32 = BASE + 0x4000;
        const SURFACE: u32 = BASE + 0x8000;
        let mem = Memory::new();
        mem.map_zero(BASE, 0x10000).unwrap();
        let mut nvmap = NvMap::new();
        let handle = nvmap.create(0x10000);
        nvmap.alloc(handle, 0, 0, 0x10000, 0, BASE).unwrap();
        let mut host1x = Host1x::new();
        let mut video = Video::default();
        let decoder = Deferred::default();
        let (given, ready) = (decoder.given.clone(), decoder.ready.clone());
        video.set_h264(Box::new(decoder));
        let nvdec = video.open(Engine::Nvdec, &mut host1x).unwrap();
        let vic = video.open(Engine::Vic, &mut host1x).unwrap();

        // A 16x32 frame from picture index 0, with a CABAC IDR slice and the driver's tail.
        let slice = [0u8, 0, 1, 0x65, 0x88, 0x84, 0x21];
        let tail = [0u8, 0, 1, 0x0b, 0, 0, 0, 0, 0, 0, 1, 0x0b, 0, 0, 0, 0];
        for (at, value) in [
            (0x48, (slice.len() + tail.len()) as u32),
            (0x60, 1),
            (0x64, 1),
            (0x68, 2),
            (0x70, 1),
            (0xb0, 1 << 12),
        ] {
            mem.write_u32(SETUP + at, value).unwrap();
        }
        mem.write_bytes(BITSTREAM, &slice).unwrap();
        mem.write_bytes(BITSTREAM + slice.len() as u32, &tail)
            .unwrap();
        let syncpt = video.channel(nvdec).unwrap().syncpt;
        let decode = method_stream(
            &[
                (NVDEC_SET_APPLICATION_ID, NVDEC_CODEC_H264),
                (NVDEC_SET_DRV_PIC_SETUP_OFFSET, SETUP >> 8),
                (NVDEC_SET_IN_BUF_BASE_OFFSET, BITSTREAM >> 8),
                (NVDEC_SET_PICTURE_LUMA_OFFSET, SURFACE >> 8),
                (NVDEC_EXECUTE, 0x100),
            ],
            syncpt,
        );
        let memory = (&mem, &nvmap, handle);
        assert_eq!(
            submit_stream(&mut video, nvdec, &decode, memory, &mut host1x).0,
            Submitted::Done
        );
        {
            let given = given.borrow();
            let [(surface, unit, key)] = given.as_slice() else {
                panic!("{} pictures reached the decoder", given.len());
            };
            assert_eq!((*surface, *key), (SURFACE, true));
            assert_eq!(&unit[..5], &[0, 0, 0, 1, 0x67], "the SPS first");
            assert!(unit.ends_with(&slice), "then the slice, without the tail");
        }

        mem.write_u32(CONFIG + VIC_CONFIG_OUTPUT_SURFACE, VIC_PIXEL_FORMAT_NV12)
            .unwrap();
        mem.write_u32(CONFIG + VIC_CONFIG_OUTPUT_SURFACE + 8, 15 | 31 << 14)
            .unwrap();
        let syncpt = video.channel(vic).unwrap().syncpt;
        let compose = method_stream(
            &[
                (VIC_SET_CONFIG_STRUCT_OFFSET, CONFIG >> 8),
                (VIC_SLOT0_LUMA_OFFSET, SURFACE >> 8),
                (VIC_SET_OUTPUT_LUMA_OFFSET, OUTPUT >> 8),
                (VIC_SET_OUTPUT_CHROMA_OFFSET, (OUTPUT + 0x800) >> 8),
                (VIC_EXECUTE, 0),
            ],
            syncpt,
        );
        assert_eq!(
            submit_stream(&mut video, vic, &compose, memory, &mut host1x).0,
            Submitted::Waiting
        );
        let mut frame = vp9::Frame::new(16, 32);
        frame.planes[0].data[0] = 0x5a;
        ready.borrow_mut().push((SURFACE, Some(frame)));
        let (submitted, fence) = submit_stream(&mut video, vic, &compose, memory, &mut host1x);
        assert_eq!(submitted, Submitted::Done);
        assert_eq!(fence, 1, "the wait reserved no increment");
        assert_eq!(mem.read_u8(OUTPUT).unwrap(), 0x5a);
    }

    #[test]
    fn a_submit_whose_records_overrun_its_arguments_is_refused() {
        let mut host1x = Host1x::new();
        let mut video = Video::default();
        let id = video.open(Engine::Vic, &mut host1x).unwrap();
        let mut args = vec![0u8; SUBMIT_HEADER];
        write_u32(&mut args, 0, 3);
        assert_eq!(
            video
                .submit(id, &mut args, &Memory::new(), &NvMap::new(), &mut host1x)
                .unwrap(),
            Submitted::Refused
        );
    }

    #[test]
    fn a_mapped_buffer_is_reached_at_its_own_address() {
        let mut nvmap = NvMap::new();
        let handle = nvmap.create(0x1000);
        nvmap.alloc(handle, 0, 0, 0x1000, 0, 0x2000_0000).unwrap();
        let mut args = vec![0u8; MAP_HEADER + 2 * MAP_ENTRY];
        write_u32(&mut args, 0, 2);
        write_u32(&mut args, MAP_HEADER, handle);
        write_u32(&mut args, MAP_HEADER + MAP_ENTRY, 0xDEAD);
        assert!(map_buffer(&mut args, &nvmap));
        assert_eq!(read_u32(&args, MAP_HEADER + 4), 0x2000_0000);
        assert_eq!(
            read_u32(&args, MAP_HEADER + MAP_ENTRY + 4),
            0,
            "no such handle"
        );
    }
}
