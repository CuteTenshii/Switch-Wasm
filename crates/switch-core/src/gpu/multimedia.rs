//! The video engines: nvdec, which decodes compressed frames, and VIC, the
//! video image compositor that converts and scales them. A title's movie
//! player drives both through `/dev/nvhost-nvdec` and `/dev/nvhost-vic`.
//!
//! Neither is a GPU engine. Each sits behind host1x, which feeds it a command
//! stream of register writes out of buffers the guest names by nvmap handle.
//! An engine's own methods are reached through two registers of its host
//! interface (THI): `METHOD0` selects a method and `METHOD1` writes it.
//!
//! Work is retired as the stream reaches each syncpoint increment, so a fence
//! a submission returns has already passed by the time the guest waits on it.

use crate::gpu::nvmap::NvMap;
use crate::gpu::syncpt::Host1x;
use crate::mem::Memory;
use crate::trace::{enabled, Trace};
use crate::Result;
use std::collections::HashMap;

/// The host1x class of host1x itself, which a stream selects to wait on or
/// load syncpoints rather than to reach an engine.
const CLASS_HOST1X: u32 = 0x01;

/// THI registers every engine class shares, in words.
const THI_INCR_SYNCPT: u32 = 0x00;
const THI_METHOD0: u32 = 0x10;
const THI_METHOD1: u32 = 0x11;

/// How many methods an engine's method file holds. VIC's reach furthest, to
/// byte offset 0x1000 and below.
const METHOD_FILE_WORDS: usize = 0x400;

/// nvdec's `EXECUTE` method: everything written before it describes one
/// frame, and this decodes it.
const NVDEC_EXECUTE: u32 = 0xC0;

/// The nvdec methods a frame is described by, as word offsets into its method
/// file. Every buffer address among them is a device address shifted right
/// by 8.
const NVDEC_SET_APPLICATION_ID: u32 = 0x80;
const NVDEC_SET_DRV_PIC_SETUP_OFFSET: u32 = 0x101;
const NVDEC_SET_IN_BUF_BASE_OFFSET: u32 = 0x102;
/// Seventeen surfaces each: for VP9 the last, golden and alternate
/// references, then the frame being decoded.
const NVDEC_SET_PICTURE_LUMA_OFFSET: u32 = 0x10C;
const NVDEC_SET_PICTURE_CHROMA_OFFSET: u32 = 0x11D;
const NVDEC_VP9_SET_PROB_TAB_BUF_OFFSET: u32 = 0x170;
const NVDEC_VP9_SET_CTX_COUNTER_BUF_OFFSET: u32 = 0x171;

/// Where in a VP9 picture setup the size of the frame's bitstream is.
const VP9_PICTURE_BITSTREAM_SIZE: u32 = 0x30;
/// How much of a picture setup the trace shows: all of VP9's.
const PICTURE_SETUP_TRACE_BYTES: u32 = 0x100;

/// VIC's `EXECUTE` method.
const VIC_EXECUTE: u32 = 0xC0;

/// The sizes of the records `SUBMIT` carries after its header, in bytes.
const SUBMIT_HEADER: usize = 0x10;
const SUBMIT_CMDBUF: usize = 12;
const SUBMIT_RELOC: usize = 16;
const SUBMIT_RELOC_SHIFT: usize = 4;
const SUBMIT_SYNCPT_INCR: usize = 8;
const SUBMIT_FENCE: usize = 4;

/// `MAP_BUFFER`'s header and each of its entries, in bytes.
const MAP_HEADER: usize = 12;
const MAP_ENTRY: usize = 8;

/// Which engine a channel reaches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    Nvdec,
    Vic,
}

impl Engine {
    /// The engine a device node opens, or `None` for a node that is not one
    /// of these.
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

    /// The engine's host1x class, which a stream's `SETCL` selects it by.
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

/// One open `/dev/nvhost-nvdec` or `/dev/nvhost-vic`.
#[derive(Debug)]
pub struct MmChannel {
    pub engine: Engine,
    /// The syncpoint `GET_SYNCPOINT` hands out and the stream increments.
    pub syncpt: u32,
    /// The method `METHOD0` last selected.
    method: u32,
    /// Every method's last value. What an `EXECUTE` does is read out of here.
    methods: Vec<u32>,
    pub submits: u64,
    pub executes: u64,
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
        }
    }

    /// A method's last value.
    pub fn method(&self, method: u32) -> u32 {
        self.methods.get(method as usize).copied().unwrap_or(0)
    }
}

/// A register write a command stream made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Write {
    class: u32,
    offset: u32,
    value: u32,
}

/// Every write a host1x command stream makes, in order, and the first opcode
/// it could not follow, if any.
///
/// `GATHER` is the one opcode that reads from anywhere but the stream itself,
/// and the one an engine's user-space driver has no reason to emit, so a
/// stream that uses it is cut short and reported rather than guessed at.
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
            // SETCL: select a class, then write the masked registers of it.
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
            // INCR and NONINCR: `count` words, to consecutive registers or
            // all to the one.
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
            // MASK: one word for each set bit, to the register that far on.
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
            // IMM: a 16-bit value carried in the opcode word itself.
            0x4 => writes.push(Write {
                class,
                offset,
                value: word & 0xFFFF,
            }),
            // RESTART belongs to a ring the stream is read out of; one read
            // out of a buffer has nothing to restart.
            0x5 => {}
            // EXTEND: acquiring and releasing an MLOCK, which serialises
            // channels on hardware that runs several at once.
            0xE => {}
            _ => return (writes, Some(word)),
        }
    }
    (writes, None)
}

/// The two video engines' state, one channel per open device node.
#[derive(Debug, Default)]
pub struct Video {
    channels: HashMap<u32, MmChannel>,
    next_channel: u32,
}

impl Video {
    /// Open a channel to `engine`, with a syncpoint of its own.
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

    /// `SUBMIT`: run the command buffers the arguments name and write each
    /// syncpoint increment's fence threshold back over the arguments.
    ///
    /// `args` is the header and every record after it, which is why it is
    /// longer than the size the ioctl number declares.
    pub fn submit(
        &mut self,
        id: u32,
        args: &mut [u8],
        mem: &Memory,
        nvmap: &NvMap,
        host1x: &mut Host1x,
    ) -> Result<bool> {
        let Some(channel) = self.channels.get_mut(&id) else {
            return Ok(false);
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
            return Ok(false);
        }
        channel.submits += 1;

        // Reserve every increment before running anything, so each fence
        // reports the threshold the whole submission will reach.
        let mut thresholds = Vec::with_capacity(incrs);
        for i in 0..incrs {
            let at = incrs_at + i * SUBMIT_SYNCPT_INCR;
            let (syncpt, increments) = (read_u32(args, at), read_u32(args, at + 4));
            thresholds.push((syncpt, host1x.incr_max(syncpt, increments)?));
        }

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
            // A relocation patches a word of this buffer with the address of
            // another, which the driver could not know when it wrote it.
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
            channel.run(&stream, mem, host1x)?;
        }

        // An increment the stream makes under a condition this does not
        // model would leave its fence unreachable, and the guest waiting on
        // it forever. Every submission here has finished by now, so each
        // threshold is where its syncpoint stands.
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
        Ok(true)
    }
}

impl MmChannel {
    /// Carry out one command buffer's writes.
    fn run(&mut self, stream: &[u32], mem: &Memory, host1x: &mut Host1x) -> Result<()> {
        let (writes, stopped) = decode_stream(stream, self.engine.class());
        for write in writes {
            match (write.class, write.offset) {
                (_, THI_INCR_SYNCPT) => {
                    host1x.increment(write.value & 0xFF)?;
                }
                (class, THI_METHOD0) if class == self.engine.class() => self.method = write.value,
                (class, THI_METHOD1) if class == self.engine.class() => {
                    self.write_method(self.method, write.value, mem)
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

    fn write_method(&mut self, method: u32, value: u32, mem: &Memory) {
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
        }
    }

    /// A buffer address a method holds.
    fn address(&self, method: u32) -> u32 {
        self.method(method) << 8
    }

    /// Everything one nvdec `EXECUTE` was given: the codec, the buffers, the
    /// picture setup and the start of the bitstream.
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

/// `MAP_BUFFER`: the address each entry's nvmap handle can be reached at by
/// the engine. Both engines address guest memory directly here, so that is
/// the handle's own address.
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

/// Where an nvmap handle's memory is. Command buffers and relocations name
/// buffers by handle; a driver that has only the buffer's id passes that
/// instead, and the two never collide.
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
        let mut mem = Memory::new();
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

        assert!(video
            .submit(id, &mut args, &mem, &nvmap, &mut host1x)
            .unwrap());
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

    #[test]
    fn a_submit_whose_records_overrun_its_arguments_is_refused() {
        let mut host1x = Host1x::new();
        let mut video = Video::default();
        let id = video.open(Engine::Vic, &mut host1x).unwrap();
        let mut args = vec![0u8; SUBMIT_HEADER];
        write_u32(&mut args, 0, 3);
        assert!(!video
            .submit(id, &mut args, &Memory::new(), &NvMap::new(), &mut host1x)
            .unwrap());
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
