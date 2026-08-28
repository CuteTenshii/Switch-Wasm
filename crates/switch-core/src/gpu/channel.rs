//! A GPU channel: the unit of work submission.
//!
//! Userspace pushes 64-bit GPFIFO entries, each pointing at a pushbuffer in
//! GPU memory. The channel's command processor (PFIFO) walks a pushbuffer,
//! decodes its method headers, and routes each method write to whichever class
//! is bound to that header's subchannel.
//!
//! The pushbuffers of a channel are one continuous stream, not a sequence of
//! self-contained programs: a method group's data words may run past the end
//! of the pushbuffer that declared them and be finished by the next.

use crate::gpu::engine::compute::EngineCompute;
use crate::gpu::engine::copy::EngineCopy;
use crate::gpu::engine::threed::Engine3D;
use crate::gpu::engine::twod::Engine2D;
use crate::gpu::engine::{field, Registers, CLASS_2D, CLASS_3D, CLASS_COMPUTE, CLASS_COPY,
    CLASS_GPFIFO, CLASS_INLINE};
use crate::gpu::exec::ExecCtx;
use crate::gpu::syncpt::NvFence;
use crate::{Error, Result};

/// Method 0 of every class binds that class to the header's subchannel.
const SET_OBJECT: u32 = 0x000;

/// Number of subchannels a channel has.
pub const SUBCHANNEL_COUNT: usize = 8;

/// Subchannel the channel's own `MAXWELL_CHANNEL_GPFIFO_A` class answers on.
/// nvhost binds it when the channel is created, so userspace never issues a
/// `SetObject` for it — deko3d writes its syncpoint increments and cache-flush
/// operations straight to subchannel 6, and without the pre-binding those
/// methods land on an unbound subchannel and the fence never signals.
pub const SUBCHANNEL_GPFIFO: usize = 6;

/// Maximum pushbuffer length (in dwords) the GPFIFO entry can express.
const MAX_PUSHBUFFER_WORDS: u32 = 0x1F_FFFF;

/// Methods below this belong to the channel itself on *every* subchannel:
/// PFIFO answers them, and no engine class defines a method under it.
const HOST_METHOD_COUNT: u32 = 0x40;

// Host methods (MAXWELL_CHANNEL_GPFIFO_A).
const GPFIFO_SEMAPHORE_OFFSET: u32 = 0x004;
const GPFIFO_SEMAPHORE_PAYLOAD: u32 = 0x006;
const GPFIFO_SEMAPHORE: u32 = 0x007;
const GPFIFO_SEMAPHORE_ACQUIRE: u32 = 0x01A;
const GPFIFO_SEMAPHORE_RELEASE: u32 = 0x01B;
const GPFIFO_SYNCPOINT: u32 = 0x01D;

/// A `SetObject` argument may carry the engine the class runs on above its
/// class id; only the class id names the object to bind.
const BIND_CLASS_MASK: u32 = 0xFFFF;
const BIND_ENGINE_MASK: u32 = 0x1F_0000;

/// What a header word tells PFIFO to do. `DMA_SEC_OP` (bits 29..31) picks the
/// form; two of the forms pick again with `DMA_TERT_OP`, which sits in the low
/// two bits of the count field and so leaves their groups eleven bits of count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Command {
    /// Start a method group: `count` data words follow the header.
    Methods {
        method: u32,
        subchannel: u32,
        count: u32,
        non_incrementing: bool,
        increment_once: bool,
    },
    /// One method write whose argument is carried in the header itself.
    Immediate { method: u32, subchannel: u32, arg: u32 },
    /// Sub-device mask bookkeeping: which GPUs of an SLI set run what follows.
    /// There is exactly one GPU here, so every mask selects it.
    SubDeviceMask,
    /// Nothing after this header belongs to the stream.
    EndSegment,
}

impl Command {
    fn decode(header: u32) -> Result<Command> {
        let method = field(header, 0, 12);
        let subchannel = field(header, 13, 15);
        let count = field(header, 16, 28);
        let short_count = field(header, 18, 28);
        let tert = field(header, 16, 17);
        match field(header, 29, 31) {
            0 if tert == 0 => Ok(Command::Methods {
                method,
                subchannel,
                count: short_count,
                non_incrementing: false,
                increment_once: false,
            }),
            0 => Ok(Command::SubDeviceMask),
            1 => Ok(Command::Methods {
                method,
                subchannel,
                count,
                non_incrementing: false,
                increment_once: false,
            }),
            2 if tert == 0 => Ok(Command::Methods {
                method,
                subchannel,
                count: short_count,
                non_incrementing: true,
                increment_once: false,
            }),
            3 => Ok(Command::Methods {
                method,
                subchannel,
                count,
                non_incrementing: true,
                increment_once: false,
            }),
            4 => Ok(Command::Immediate { method, subchannel, arg: count }),
            5 => Ok(Command::Methods {
                method,
                subchannel,
                count,
                non_incrementing: false,
                increment_once: true,
            }),
            7 => Ok(Command::EndSegment),
            op => Err(Error::Gpu(format!(
                "pfifo: unsupported submission mode {} in header {:#010x}",
                op, header
            ))),
        }
    }
}

/// The command processor's decode state. It lives on the channel because the
/// stream outlives any one pushbuffer -- and any one submission.
#[derive(Debug, Clone, Copy, Default)]
struct Pfifo {
    method: u32,
    subchannel: u32,
    /// Data words still owed to the group in flight.
    remaining: u32,
    non_incrementing: bool,
    increment_once: bool,
}

#[derive(Debug)]
pub struct Channel {
    pub id: u32,
    /// The address space bound with `NVGPU_AS_IOCTL_BIND_CHANNEL`.
    pub as_id: Option<u32>,
    /// The channel's host1x syncpoint.
    pub syncpt: u32,
    /// Class bound to each subchannel, 0 when unbound.
    pub subchannel_class: [u32; SUBCHANNEL_COUNT],
    pub three_d: Engine3D,
    pub two_d: Engine2D,
    pub copy: EngineCopy,
    pub compute: EngineCompute,
    /// MAXWELL_CHANNEL_GPFIFO_A's own register file.
    pub gpfifo_regs: Registers,
    /// Size of the GPFIFO ring the guest allocated.
    pub gpfifo_entries: u32,
    pfifo: Pfifo,
}

impl Channel {
    pub fn new(id: u32, syncpt: u32) -> Channel {
        let mut subchannel_class = [0; SUBCHANNEL_COUNT];
        subchannel_class[SUBCHANNEL_GPFIFO] = CLASS_GPFIFO;
        Channel {
            id,
            as_id: None,
            syncpt,
            subchannel_class,
            three_d: Engine3D::new(),
            two_d: Engine2D::new(),
            copy: EngineCopy::new(),
            compute: EngineCompute::new(),
            gpfifo_regs: Registers::new(),
            gpfifo_entries: 0,
            pfifo: Pfifo::default(),
        }
    }

    /// Run every pushbuffer referenced by `entries`, then retire the channel's
    /// syncpoint to `fence`.
    pub fn submit(
        &mut self,
        entries: &[u64],
        fence: NvFence,
        ctx: &mut ExecCtx,
    ) -> Result<()> {
        ctx.stats.submissions += 1;
        for &entry in entries {
            self.run_gpfifo_entry(entry, ctx)?;
        }
        if fence.is_valid() {
            ctx.host1x.set(fence.id, fence.value)?;
        }
        Ok(())
    }

    fn run_gpfifo_entry(&mut self, entry: u64, ctx: &mut ExecCtx) -> Result<()> {
        let address = entry & 0xFF_FFFF_FFFC;
        let words = ((entry >> 42) & MAX_PUSHBUFFER_WORDS as u64) as u32;
        if words == 0 {
            return Ok(());
        }
        if ctx.trace {
            eprintln!("[gpu] pushbuffer {:#x} ({} words)", address, words);
        }
        let mut pushbuffer = vec![0u8; words as usize * 4];
        ctx.vmm.read_into(ctx.mem, address, &mut pushbuffer)?;
        self.run_pushbuffer(&pushbuffer, ctx)
    }

    /// Decode and execute one pushbuffer.
    ///
    /// A group left unfinished here is finished by the next pushbuffer: the
    /// guest is free to cut its command stream anywhere, and a 64-word texture
    /// upload landing across two GPFIFO entries used to fault the channel.
    pub fn run_pushbuffer(&mut self, bytes: &[u8], ctx: &mut ExecCtx) -> Result<()> {
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for &word in &words {
            if self.pfifo.remaining > 0 {
                self.data_word(word, ctx)?;
                continue;
            }
            if word == 0 {
                // Pushbuffer padding.
                continue;
            }
            match Command::decode(word)? {
                Command::Methods { method, subchannel, count, non_incrementing, increment_once } => {
                    self.pfifo =
                        Pfifo { method, subchannel, remaining: count, non_incrementing, increment_once };
                }
                Command::Immediate { method, subchannel, arg } => {
                    self.method(subchannel, method, arg, true, ctx)?;
                }
                Command::SubDeviceMask => {}
                Command::EndSegment => break,
            }
        }
        Ok(())
    }

    /// Consume one data word of the group in flight.
    fn data_word(&mut self, word: u32, ctx: &mut ExecCtx) -> Result<()> {
        let method = self.pfifo.method;
        let subchannel = self.pfifo.subchannel;
        self.pfifo.remaining -= 1;
        if !self.pfifo.non_incrementing {
            self.pfifo.method += 1;
        }
        // Increase-once increments after the first word and never again.
        self.pfifo.non_incrementing |= self.pfifo.increment_once;
        let last_call = self.pfifo.remaining == 0;
        self.method(subchannel, method, word, last_call, ctx)
    }

    /// Route one method write to the class bound to `subchannel`.
    pub fn method(
        &mut self,
        subchannel: u32,
        method: u32,
        arg: u32,
        last_call: bool,
        ctx: &mut ExecCtx,
    ) -> Result<()> {
        ctx.stats.methods += 1;
        let slot = subchannel as usize;
        if slot >= SUBCHANNEL_COUNT {
            return Err(Error::Gpu(format!("pfifo: subchannel {} out of range", subchannel)));
        }
        if method < HOST_METHOD_COUNT {
            return self.host_method(slot, method, arg, ctx);
        }
        let class = self.subchannel_class[slot];
        if ctx.trace {
            eprintln!(
                "[gpu] subch{} class={:#x} method={:#05x} arg={:#010x}",
                subchannel, class, method, arg
            );
        }
        match class {
            CLASS_3D => {
                self.three_d.write(method, arg, last_call, ctx)
            }
            // The standalone class and the 3D class's own methods are one
            // unit sharing one register file, so they share one instance.
            CLASS_INLINE => self.three_d.inline.write(method, arg, ctx),
            // Both of these read guest memory, and the wgpu backend keeps a
            // render target on the device until it is flushed — so a copy out
            // of one reads whatever was in memory before it was drawn into.
            // It is not a hypothetical: Just Dance 2019 resolves its
            // multisampled colour target with a 2D blit, once a frame.
            CLASS_2D => {
                if method == crate::gpu::engine::twod::Engine2D::LAUNCHES_BLIT {
                    self.three_d.flush_renderer(ctx)?;
                }
                self.two_d.write(method, arg, ctx)
            }
            CLASS_COPY => {
                if method == crate::gpu::engine::copy::LAUNCH_DMA {
                    self.three_d.flush_renderer(ctx)?;
                }
                self.copy.write(method, arg, ctx)
            }
            CLASS_COMPUTE => {
                // A dispatch reads and writes guest memory, and the wgpu
                // backend keeps a render target on the device until it is
                // flushed — so a kernel reading one would read stale bytes.
                if method == crate::gpu::engine::compute::SEND_SIGNALING_PCAS_B {
                    self.three_d.flush_renderer(ctx)?;
                }
                self.compute.write(method, arg, ctx)
            }
            // Only the host methods handled above exist on this class; a
            // write past them is state with nothing reading it.
            CLASS_GPFIFO => {
                self.gpfifo_regs.set(method, arg);
                Ok(())
            }
            0 => Err(Error::Gpu(format!(
                "pfifo: method {:#x} on subchannel {} before any class was bound",
                method, subchannel
            ))),
            other => Err(Error::Gpu(format!(
                "pfifo: class {:#x} bound to subchannel {} is not implemented",
                other, subchannel
            ))),
        }
    }

    /// The channel's own methods -- class binding, semaphores, syncpoints --
    /// which PFIFO answers on whichever subchannel they arrive on.
    fn host_method(
        &mut self,
        slot: usize,
        method: u32,
        arg: u32,
        ctx: &mut ExecCtx,
    ) -> Result<()> {
        self.gpfifo_regs.set(method, arg);
        match method {
            SET_OBJECT => {
                // The argument may name the engine above the class id, and it
                // is the class id that names the object.
                let class = if arg & !BIND_CLASS_MASK != 0
                    && arg & !(BIND_ENGINE_MASK | BIND_CLASS_MASK) == 0
                {
                    arg & BIND_CLASS_MASK
                } else {
                    arg
                };
                self.subchannel_class[slot] = class;
                if ctx.trace {
                    eprintln!("[gpu] subchannel {} bound to class {:#x}", slot, class);
                }
                Ok(())
            }
            GPFIFO_SEMAPHORE => {
                const OPERATION_RELEASE: u32 = 2;
                const RELEASE_SIZE_4_BYTES: u32 = 1;
                if field(arg, 0, 4) != OPERATION_RELEASE {
                    // Acquires are already satisfied: a submission runs to
                    // completion before the ioctl returns.
                    return Ok(());
                }
                let addr = self.gpfifo_regs.iova(GPFIFO_SEMAPHORE_OFFSET);
                let payload = self.gpfifo_regs.get(GPFIFO_SEMAPHORE_PAYLOAD);
                if field(arg, 24, 24) == RELEASE_SIZE_4_BYTES {
                    ctx.write_u32(addr, payload)?;
                } else {
                    ctx.write_u64(addr, payload as u64)?;
                    ctx.write_u64(addr + 8, ctx.stats.submissions)?;
                }
                Ok(())
            }
            // The long-form release of the same semaphore, payload in the
            // argument rather than in a register.
            GPFIFO_SEMAPHORE_RELEASE => {
                let addr = self.gpfifo_regs.iova(GPFIFO_SEMAPHORE_OFFSET);
                ctx.write_u32(addr, arg)
            }
            GPFIFO_SEMAPHORE_ACQUIRE => Ok(()),
            GPFIFO_SYNCPOINT => {
                const OPERATION_INCR: u32 = 1;
                let id = field(arg, 8, 15);
                if field(arg, 0, 0) == OPERATION_INCR {
                    ctx.host1x.increment(id)?;
                }
                Ok(())
            }
            // Nop, cache maintenance and reference counts: the engines write
            // guest memory as they go, so there is no cache to maintain.
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::exec::GpuStats;
    use crate::gpu::syncpt::Host1x;
    use crate::gpu::vmm::{AddressSpace, SMALL_PAGE_SIZE};
    use crate::mem::Memory;

    struct Harness {
        mem: Memory,
        vmm: AddressSpace,
        host1x: Host1x,
        stats: GpuStats,
        base: u64,
    }

    impl Harness {
        fn new() -> Harness {
            let mut mem = Memory::new();
            mem.map_zero(0x3000_0000, 0x4000).unwrap();
            let mut vmm = AddressSpace::new();
            let base = vmm.map(0x3000_0000, 0x4000, 1, 0, SMALL_PAGE_SIZE, 0, 0).unwrap();
            Harness { mem, vmm, host1x: Host1x::new(), stats: GpuStats::default(), base }
        }

        fn ctx(&mut self) -> ExecCtx<'_> {
            ExecCtx {
                mem: &mut self.mem,
                vmm: &self.vmm,
                host1x: &mut self.host1x,
                stats: &mut self.stats,
                trace: false,
            }
        }
    }

    fn header(mode: u32, arg: u32, subchannel: u32, method: u32) -> u32 {
        (method & 0x1FFF) | ((subchannel & 7) << 13) | ((arg & 0x1FFF) << 16) | (mode << 29)
    }

    fn pushbuffer(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    #[test]
    fn the_gpfifo_subchannel_is_bound_without_a_set_object() {
        // nvhost binds the channel's own class, so deko3d writes its syncpoint
        // increments to subchannel 6 with no SetObject. Requiring one made the
        // pushbuffer fault ("method 0xb on subchannel 6 before any class was
        // bound") and hbmenu's frame fence never signalled.
        let chan = Channel::new(1, 8);
        assert_eq!(chan.subchannel_class[SUBCHANNEL_GPFIFO], CLASS_GPFIFO);
        for (index, &class) in chan.subchannel_class.iter().enumerate() {
            if index != SUBCHANNEL_GPFIFO {
                assert_eq!(class, 0, "subchannel {index} must start unbound");
            }
        }
    }

    #[test]
    fn set_object_binds_a_class_to_a_subchannel() {
        let mut h = Harness::new();
        let mut chan = Channel::new(1, 8);
        let pb = pushbuffer(&[header(3, 1, 0, 0), CLASS_3D]);
        let mut ctx = h.ctx();
        chan.run_pushbuffer(&pb, &mut ctx).unwrap();
        assert_eq!(chan.subchannel_class[0], CLASS_3D);
    }

    #[test]
    fn increasing_mode_walks_consecutive_methods() {
        let mut h = Harness::new();
        let mut chan = Channel::new(1, 8);
        let pb = pushbuffer(&[
            header(3, 1, 0, 0),
            CLASS_3D,
            // Increasing write of four clear-colour registers.
            header(1, 4, 0, 0x360),
            1,
            2,
            3,
            4,
        ]);
        let mut ctx = h.ctx();
        chan.run_pushbuffer(&pb, &mut ctx).unwrap();
        assert_eq!(chan.three_d.regs.get(0x360), 1);
        assert_eq!(chan.three_d.regs.get(0x361), 2);
        assert_eq!(chan.three_d.regs.get(0x362), 3);
        assert_eq!(chan.three_d.regs.get(0x363), 4);
    }

    #[test]
    fn non_increasing_mode_rewrites_one_method() {
        let mut h = Harness::new();
        let mut chan = Channel::new(1, 8);
        let pb = pushbuffer(&[header(3, 1, 0, 0), CLASS_3D, header(3, 3, 0, 0x360), 7, 8, 9]);
        let mut ctx = h.ctx();
        chan.run_pushbuffer(&pb, &mut ctx).unwrap();
        assert_eq!(chan.three_d.regs.get(0x360), 9);
        assert_eq!(chan.three_d.regs.get(0x361), 0);
    }

    #[test]
    fn increase_once_advances_only_after_the_first_word() {
        let mut h = Harness::new();
        let mut chan = Channel::new(1, 8);
        let pb = pushbuffer(&[header(3, 1, 0, 0), CLASS_3D, header(5, 3, 0, 0x360), 7, 8, 9]);
        let mut ctx = h.ctx();
        chan.run_pushbuffer(&pb, &mut ctx).unwrap();
        assert_eq!(chan.three_d.regs.get(0x360), 7);
        assert_eq!(chan.three_d.regs.get(0x361), 9);
    }

    #[test]
    fn inline_mode_carries_its_argument_in_the_header() {
        let mut h = Harness::new();
        let mut chan = Channel::new(1, 8);
        let pb = pushbuffer(&[header(3, 1, 0, 0), CLASS_3D, header(4, 0x123, 0, 0x360)]);
        let mut ctx = h.ctx();
        chan.run_pushbuffer(&pb, &mut ctx).unwrap();
        assert_eq!(chan.three_d.regs.get(0x360), 0x123);
    }

    #[test]
    fn a_method_before_set_object_is_an_error() {
        let mut h = Harness::new();
        let mut chan = Channel::new(1, 8);
        let pb = pushbuffer(&[header(1, 1, 0, 0x360), 1]);
        let mut ctx = h.ctx();
        assert!(chan.run_pushbuffer(&pb, &mut ctx).is_err());
    }

    #[test]
    fn a_method_group_finishes_in_the_next_pushbuffer() {
        // The stream is continuous. A 64-word LOAD_INLINE_DATA upload split
        // across two GPFIFO entries used to fault the channel with
        // "header 0x6040406d claims 64 words but only 0 remain".
        let mut h = Harness::new();
        let mut chan = Channel::new(1, 8);
        let mut ctx = h.ctx();
        chan.run_pushbuffer(&pushbuffer(&[header(3, 1, 0, 0), CLASS_3D]), &mut ctx).unwrap();
        chan.run_pushbuffer(&pushbuffer(&[header(1, 4, 0, 0x360), 1, 2]), &mut ctx).unwrap();
        assert_eq!(chan.three_d.regs.get(0x362), 0, "the group is not done yet");
        chan.run_pushbuffer(&pushbuffer(&[3, 4]), &mut ctx).unwrap();
        assert_eq!(chan.three_d.regs.get(0x360), 1);
        assert_eq!(chan.three_d.regs.get(0x361), 2);
        assert_eq!(chan.three_d.regs.get(0x362), 3);
        assert_eq!(chan.three_d.regs.get(0x363), 4);
    }

    #[test]
    fn a_split_group_keeps_its_increment_mode() {
        let mut h = Harness::new();
        let mut chan = Channel::new(1, 8);
        let mut ctx = h.ctx();
        chan.run_pushbuffer(&pushbuffer(&[header(3, 1, 0, 0), CLASS_3D]), &mut ctx).unwrap();
        chan.run_pushbuffer(&pushbuffer(&[header(5, 3, 0, 0x360), 7]), &mut ctx).unwrap();
        chan.run_pushbuffer(&pushbuffer(&[8, 9]), &mut ctx).unwrap();
        assert_eq!(chan.three_d.regs.get(0x360), 7);
        assert_eq!(chan.three_d.regs.get(0x361), 9);
    }

    #[test]
    fn a_data_word_is_never_read_as_a_header() {
        // A payload that happens to look like an end-of-segment header must
        // still reach the method it belongs to.
        let mut h = Harness::new();
        let mut chan = Channel::new(1, 8);
        let pb = pushbuffer(&[header(3, 1, 0, 0), CLASS_3D, header(1, 2, 0, 0x360), 0xE000_0000, 7]);
        let mut ctx = h.ctx();
        chan.run_pushbuffer(&pb, &mut ctx).unwrap();
        assert_eq!(chan.three_d.regs.get(0x360), 0xE000_0000);
        assert_eq!(chan.three_d.regs.get(0x361), 7);
    }

    #[test]
    fn end_of_segment_stops_the_pushbuffer() {
        let mut h = Harness::new();
        let mut chan = Channel::new(1, 8);
        let pb = pushbuffer(&[
            header(3, 1, 0, 0),
            CLASS_3D,
            header(4, 0x11, 0, 0x360),
            header(7, 0, 0, 0),
            header(4, 0x55, 0, 0x360),
        ]);
        let mut ctx = h.ctx();
        chan.run_pushbuffer(&pb, &mut ctx).unwrap();
        assert_eq!(chan.three_d.regs.get(0x360), 0x11);
    }

    #[test]
    fn the_old_method_forms_count_in_eleven_bits() {
        // Modes 0 and 2 spend the low two bits of the count field on a second
        // opcode, so their counts start two bits higher up.
        let mut h = Harness::new();
        let mut chan = Channel::new(1, 8);
        let old_increasing = 0x360 | (2 << 18);
        let old_non_increasing = 0x360 | (2 << 18) | (2 << 29);
        let pb = pushbuffer(&[
            header(3, 1, 0, 0),
            CLASS_3D,
            old_increasing,
            1,
            2,
            old_non_increasing,
            3,
            4,
        ]);
        let mut ctx = h.ctx();
        chan.run_pushbuffer(&pb, &mut ctx).unwrap();
        assert_eq!(chan.three_d.regs.get(0x360), 4);
        assert_eq!(chan.three_d.regs.get(0x361), 2);
    }

    #[test]
    fn a_sub_device_mask_header_carries_no_data_words() {
        let mut h = Harness::new();
        let mut chan = Channel::new(1, 8);
        // `use_sub_dev_mask`, then a real command that must still be decoded.
        let pb = pushbuffer(&[3 << 16, header(3, 1, 0, 0), CLASS_3D, header(4, 0x55, 0, 0x360)]);
        let mut ctx = h.ctx();
        chan.run_pushbuffer(&pb, &mut ctx).unwrap();
        assert_eq!(chan.three_d.regs.get(0x360), 0x55);
    }

    #[test]
    fn a_reserved_submission_mode_is_an_error() {
        let mut h = Harness::new();
        let mut chan = Channel::new(1, 8);
        let pb = pushbuffer(&[header(6, 1, 0, 0x360), 1]);
        let mut ctx = h.ctx();
        assert!(chan.run_pushbuffer(&pb, &mut ctx).is_err());
    }

    #[test]
    fn host_methods_are_answered_on_any_subchannel() {
        // The low 0x40 of every class's method space belongs to the channel,
        // so a syncpoint increment works on the 3D subchannel too.
        let mut h = Harness::new();
        let mut chan = Channel::new(1, 8);
        let pb = pushbuffer(&[
            header(3, 1, 0, 0),
            CLASS_3D,
            header(1, 1, 0, GPFIFO_SYNCPOINT),
            1 | (9 << 8),
        ]);
        let mut ctx = h.ctx();
        chan.run_pushbuffer(&pb, &mut ctx).unwrap();
        assert_eq!(h.host1x.read(9).unwrap(), 1);
        assert_eq!(chan.three_d.regs.get(GPFIFO_SYNCPOINT), 0);
    }

    #[test]
    fn set_object_ignores_the_engine_id_above_the_class() {
        let mut h = Harness::new();
        let mut chan = Channel::new(1, 8);
        let pb = pushbuffer(&[header(3, 1, 0, 0), (1 << 16) | CLASS_3D]);
        let mut ctx = h.ctx();
        chan.run_pushbuffer(&pb, &mut ctx).unwrap();
        assert_eq!(chan.subchannel_class[0], CLASS_3D);
    }

    #[test]
    fn the_long_form_semaphore_release_writes_its_argument() {
        let mut h = Harness::new();
        let base = h.base;
        let mut chan = Channel::new(1, 8);
        let pb = pushbuffer(&[
            header(1, 2, 6, GPFIFO_SEMAPHORE_OFFSET),
            (base >> 32) as u32,
            base as u32,
            header(1, 1, 6, GPFIFO_SEMAPHORE_RELEASE),
            0x1234_5678,
        ]);
        let mut ctx = h.ctx();
        chan.run_pushbuffer(&pb, &mut ctx).unwrap();
        assert_eq!(h.mem.read_u32(0x3000_0000).unwrap(), 0x1234_5678);
    }

    #[test]
    fn gpfifo_syncpoint_increment_bumps_host1x() {
        let mut h = Harness::new();
        let mut chan = Channel::new(1, 8);
        let pb = pushbuffer(&[
            header(3, 1, 6, 0),
            CLASS_GPFIFO,
            header(1, 1, 6, GPFIFO_SYNCPOINT),
            1 | (9 << 8),
        ]);
        let mut ctx = h.ctx();
        chan.run_pushbuffer(&pb, &mut ctx).unwrap();
        assert_eq!(h.host1x.read(9).unwrap(), 1);
    }

    #[test]
    fn gpfifo_semaphore_release_writes_memory() {
        let mut h = Harness::new();
        let base = h.base;
        let mut chan = Channel::new(1, 8);
        let pb = pushbuffer(&[
            header(3, 1, 6, 0),
            CLASS_GPFIFO,
            header(1, 3, 6, GPFIFO_SEMAPHORE_OFFSET),
            (base >> 32) as u32,
            base as u32,
            0xCAFE_F00D, // payload (method 0x006)
            header(1, 1, 6, GPFIFO_SEMAPHORE),
            2 | (1 << 24), // Release, four-byte
        ]);
        let mut ctx = h.ctx();
        chan.run_pushbuffer(&pb, &mut ctx).unwrap();
        assert_eq!(h.mem.read_u32(0x3000_0000).unwrap(), 0xCAFE_F00D);
    }

    #[test]
    fn submit_runs_a_pushbuffer_through_a_gpfifo_entry() {
        let mut h = Harness::new();
        let base = h.base;
        let pb = pushbuffer(&[header(3, 1, 0, 0), CLASS_3D, header(4, 0x55, 0, 0x360)]);
        for (i, b) in pb.iter().enumerate() {
            h.mem.write_u8(0x3000_0000 + i as u32, *b).unwrap();
        }
        let words = (pb.len() / 4) as u64;
        let entry = (base & 0xFF_FFFF_FFFC) | (words << 42);

        let mut chan = Channel::new(1, 8);
        let fence = NvFence { id: 8, value: 5 };
        let mut ctx = h.ctx();
        chan.submit(&[entry], fence, &mut ctx).unwrap();

        assert_eq!(chan.three_d.regs.get(0x360), 0x55);
        assert_eq!(h.host1x.read(8).unwrap(), 5);
        assert_eq!(h.stats.submissions, 1);
    }
}
