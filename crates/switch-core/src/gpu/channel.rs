//! A GPU channel: PFIFO walks the pushbuffers that GPFIFO entries point at and routes
//! each method to the class bound to its subchannel. A channel's pushbuffers form
//! one continuous stream.

use crate::gpu::engine::compute::EngineCompute;
use crate::gpu::engine::copy::EngineCopy;
use crate::gpu::engine::threed::Engine3D;
use crate::gpu::engine::twod::Engine2D;
use crate::gpu::engine::{
    field, Registers, CLASS_2D, CLASS_3D, CLASS_COMPUTE, CLASS_COPY, CLASS_GPFIFO, CLASS_INLINE,
};
use crate::gpu::exec::ExecCtx;
use crate::gpu::renderer::{BlitRoute, Flush};
use crate::gpu::syncpt::NvFence;
use crate::{Error, Result};

/// Method 0 of every class binds that class to the header's subchannel.
const SET_OBJECT: u32 = 0x000;

pub const SUBCHANNEL_COUNT: usize = 8;

/// Subchannel of `MAXWELL_CHANNEL_GPFIFO_A`, pre-bound by nvhost at channel creation.
pub const SUBCHANNEL_GPFIFO: usize = 6;

/// Maximum pushbuffer length (in dwords) the GPFIFO entry can express.
const MAX_PUSHBUFFER_WORDS: u32 = 0x1F_FFFF;

/// Methods below this are host methods on every subchannel.
const HOST_METHOD_COUNT: u32 = 0x40;

// Host methods (MAXWELL_CHANNEL_GPFIFO_A).
const GPFIFO_SEMAPHORE_OFFSET: u32 = 0x004;
const GPFIFO_SEMAPHORE_PAYLOAD: u32 = 0x006;
const GPFIFO_SEMAPHORE: u32 = 0x007;
const GPFIFO_SEMAPHORE_ACQUIRE: u32 = 0x01A;
const GPFIFO_SEMAPHORE_RELEASE: u32 = 0x01B;
const GPFIFO_SYNCPOINT: u32 = 0x01D;

/// A `SetObject` argument may carry the engine above the class id.
const BIND_CLASS_MASK: u32 = 0xFFFF;
const BIND_ENGINE_MASK: u32 = 0x1F_0000;

/// A PFIFO header word's command. `DMA_SEC_OP` (bits 29..31) picks the form; two forms
/// sub-select with `DMA_TERT_OP` in the low two bits of the count.
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
    Immediate {
        method: u32,
        subchannel: u32,
        arg: u32,
    },
    /// Sub-device mask bookkeeping; the single GPU is always selected.
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
            4 => Ok(Command::Immediate {
                method,
                subchannel,
                arg: count,
            }),
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

/// The command processor's decode state, which outlives any one pushbuffer.
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
    /// The process that opened it, whose memory its address space maps.
    pub owner: u64,
    /// The address space bound with `NVGPU_AS_IOCTL_BIND_CHANNEL`.
    pub as_id: Option<u32>,
    pub syncpt: u32,
    /// Class bound to each subchannel, 0 when unbound.
    pub subchannel_class: [u32; SUBCHANNEL_COUNT],
    pub three_d: Engine3D,
    pub two_d: Engine2D,
    pub copy: EngineCopy,
    pub compute: EngineCompute,
    /// MAXWELL_CHANNEL_GPFIFO_A's own register file.
    pub gpfifo_regs: Registers,
    pub gpfifo_entries: u32,
    pfifo: Pfifo,
    /// A submission stopped at a method waiting for the backend; see [`Channel::resume`].
    parked: Option<Parked>,
    /// Set by the method that stopped, until the submission is parked.
    resume: Option<Resume>,
    /// Set by the pushbuffer that stopped: its words and the one after the method.
    stopped_at: Option<(Vec<u32>, usize)>,
}

/// Where a submission stopped, and what it had left to run.
#[derive(Debug)]
struct Parked {
    resume: Resume,
    words: Vec<u32>,
    next: usize,
    entries: Vec<u64>,
    fence: NvFence,
}

/// The work a parked submission runs first.
#[derive(Debug, Clone, Copy)]
enum Resume {
    /// The method that stopped, from the start.
    Method {
        subchannel: u32,
        method: u32,
        arg: u32,
        last_call: bool,
    },
    /// The software dispatch of the compute launch already recorded.
    SoftwareCompute,
}

/// How far a submission got.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Submitted {
    Done,
    /// Parked at a method waiting for the backend's readbacks.
    Parked,
}

impl Channel {
    pub fn new(id: u32, syncpt: u32) -> Channel {
        let mut subchannel_class = [0; SUBCHANNEL_COUNT];
        subchannel_class[SUBCHANNEL_GPFIFO] = CLASS_GPFIFO;
        Channel {
            id,
            owner: 0,
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
            parked: None,
            resume: None,
            stopped_at: None,
        }
    }

    /// Run every pushbuffer in `entries`, then retire the channel's syncpoint to `fence`; or
    /// park at a method waiting for the backend, for [`Channel::resume`] to finish.
    pub fn submit(
        &mut self,
        entries: &[u64],
        fence: NvFence,
        ctx: &mut ExecCtx,
    ) -> Result<Submitted> {
        ctx.stats.submissions += 1;
        self.run_entries(entries, fence, ctx)
    }

    /// Run on from where a submission parked, parking again if the backend is still not done.
    pub fn resume(&mut self, ctx: &mut ExecCtx) -> Result<Submitted> {
        let Some(parked) = self.parked.take() else {
            return Ok(Submitted::Done);
        };
        let first = match parked.resume {
            Resume::Method {
                subchannel,
                method,
                arg,
                last_call,
            } => self.method(subchannel, method, arg, last_call, ctx),
            Resume::SoftwareCompute => self.software_compute(ctx),
        };
        match first {
            Err(Error::GpuSyncPending) => {
                let resume = self.resume.take().unwrap_or(parked.resume);
                self.parked = Some(Parked { resume, ..parked });
                return Ok(Submitted::Parked);
            }
            other => other?,
        }
        match self.run_words(parked.words, parked.next, ctx) {
            Err(Error::GpuSyncPending) => {
                self.park(parked.entries, parked.fence);
                return Ok(Submitted::Parked);
            }
            other => other?,
        }
        self.run_entries(&parked.entries, parked.fence, ctx)
    }

    /// Whether a submission is parked.
    pub fn is_parked(&self) -> bool {
        self.parked.is_some()
    }

    fn run_entries(
        &mut self,
        entries: &[u64],
        fence: NvFence,
        ctx: &mut ExecCtx,
    ) -> Result<Submitted> {
        for (i, &entry) in entries.iter().enumerate() {
            match self.run_gpfifo_entry(entry, ctx) {
                Err(Error::GpuSyncPending) => {
                    self.park(entries[i + 1..].to_vec(), fence);
                    return Ok(Submitted::Parked);
                }
                other => other?,
            }
        }
        if fence.is_valid() {
            ctx.host1x.set(fence.id, fence.value)?;
        }
        Ok(Submitted::Done)
    }

    fn park(&mut self, entries: Vec<u64>, fence: NvFence) {
        let (words, next) = self.stopped_at.take().unwrap_or_default();
        let resume = self
            .resume
            .take()
            .expect("a method stopped without saying how to resume");
        self.parked = Some(Parked {
            resume,
            words,
            next,
            entries,
            fence,
        });
    }

    fn run_gpfifo_entry(&mut self, entry: u64, ctx: &mut ExecCtx) -> Result<()> {
        let address = entry & 0xFF_FFFF_FFFC;
        let words = ((entry >> 42) & MAX_PUSHBUFFER_WORDS as u64) as u32;
        if words == 0 {
            return Ok(());
        }
        if ctx.trace {
            crate::traceln!("[gpu] pushbuffer {:#x} ({} words)", address, words);
        }
        let mut pushbuffer = vec![0u8; words as usize * 4];
        ctx.vmm
            .read_into(ctx.mem, address, &mut pushbuffer)
            .map_err(|e| match e {
                Error::Gpu(why) => Error::Gpu(format!("pushbuffer of {words} words: {why}")),
                other => other,
            })?;
        self.run_pushbuffer(&pushbuffer, ctx)
    }

    /// Decode and execute one pushbuffer. An unfinished group continues in the next.
    pub fn run_pushbuffer(&mut self, bytes: &[u8], ctx: &mut ExecCtx) -> Result<()> {
        let words: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        self.run_words(words, 0, ctx)
    }

    /// Execute `words` from `from`; a method that has to wait keeps them for the resume.
    fn run_words(&mut self, words: Vec<u32>, from: usize, ctx: &mut ExecCtx) -> Result<()> {
        for at in from..words.len() {
            match self.run_word(words[at], ctx) {
                Ok(true) => {}
                Ok(false) => break,
                Err(Error::GpuSyncPending) => {
                    self.stopped_at = Some((words, at + 1));
                    return Err(Error::GpuSyncPending);
                }
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// One pushbuffer word; `false` at the end of the segment.
    fn run_word(&mut self, word: u32, ctx: &mut ExecCtx) -> Result<bool> {
        if self.pfifo.remaining > 0 {
            self.data_word(word, ctx)?;
            return Ok(true);
        }
        if word == 0 {
            // Pushbuffer padding.
            return Ok(true);
        }
        match Command::decode(word)? {
            Command::Methods {
                method,
                subchannel,
                count,
                non_incrementing,
                increment_once,
            } => {
                self.pfifo = Pfifo {
                    method,
                    subchannel,
                    remaining: count,
                    non_incrementing,
                    increment_once,
                };
            }
            Command::Immediate {
                method,
                subchannel,
                arg,
            } => {
                self.method(subchannel, method, arg, true, ctx)?;
            }
            Command::SubDeviceMask => {}
            Command::EndSegment => return Ok(false),
        }
        Ok(true)
    }

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

    /// Run one method, naming it in the error if it fails.
    pub fn method(
        &mut self,
        subchannel: u32,
        method: u32,
        arg: u32,
        last_call: bool,
        ctx: &mut ExecCtx,
    ) -> Result<()> {
        self.run_method(subchannel, method, arg, last_call, ctx)
            .map_err(|e| match e {
                Error::Gpu(why) => {
                    let class = self
                        .subchannel_class
                        .get(subchannel as usize)
                        .copied()
                        .unwrap_or(0);
                    Error::Gpu(format!(
                        "class {class:#x} method {method:#x} (arg {arg:#x}): {why}"
                    ))
                }
                other => other,
            })
    }

    fn run_method(
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
            return Err(Error::Gpu(format!(
                "pfifo: subchannel {} out of range",
                subchannel
            )));
        }
        if method < HOST_METHOD_COUNT {
            return self.host_method(slot, method, arg, ctx);
        }
        let class = self.subchannel_class[slot];
        if ctx.trace {
            crate::traceln!(
                "[gpu] subch{} class={:#x} method={:#05x} arg={:#010x}",
                subchannel,
                class,
                method,
                arg
            );
        }
        match class {
            CLASS_3D => self.three_d.write(method, arg, last_call, ctx),
            // The standalone class shares the 3D class's register file.
            CLASS_INLINE => self.three_d.inline.write(method, arg, ctx),
            // Flush device-resident render targets before a blit reads guest memory.
            CLASS_2D => {
                if method == crate::gpu::engine::twod::Engine2D::LAUNCHES_BLIT {
                    self.two_d.regs.set(method, arg);
                    // A blit the registers cannot describe reports that from `write`.
                    let request = self.two_d.request().ok();
                    let route = match &request {
                        Some(blit) => self.three_d.blit_on_renderer(blit, ctx)?,
                        None => BlitRoute::Flush,
                    };
                    match (route, request) {
                        (BlitRoute::Done, Some(blit)) => {
                            self.two_d.note(&blit, ctx);
                            ctx.stats.copies += 1;
                            return Ok(());
                        }
                        (BlitRoute::GuestMemory, _) => {}
                        (_, Some(blit)) => {
                            let span = |s: &crate::gpu::surface::Surface| {
                                s.addr..s.addr + u64::from(s.size())
                            };
                            let synced = self.three_d.sync_renderer_ranges(
                                &[span(&blit.src)],
                                &[span(&blit.dst)],
                                ctx,
                            )?;
                            self.wait_for(synced, subchannel, method, arg, last_call)?;
                        }
                        (_, None) => {
                            let flushed = self.three_d.flush_renderer(ctx)?;
                            self.wait_for(flushed, subchannel, method, arg, last_call)?;
                        }
                    }
                }
                self.two_d.write(method, arg, ctx)
            }
            CLASS_COPY => {
                if method == crate::gpu::engine::copy::LAUNCH_DMA {
                    self.copy.regs.set(method, arg);
                    // Titles upload constants this way, hundreds of small runs a frame.
                    let current = self
                        .copy
                        .plain_ranges(arg)
                        .is_some_and(|ranges| self.three_d.guest_memory_is_current(&ranges, ctx));
                    if !current {
                        let flushed = self.three_d.flush_renderer(ctx)?;
                        self.wait_for(flushed, subchannel, method, arg, last_call)?;
                    }
                }
                self.copy.write(method, arg, ctx)
            }
            CLASS_COMPUTE => {
                if self.compute.write(method, arg, ctx)?
                    && !self.three_d.dispatch_on_device(&self.compute, ctx)?
                {
                    self.software_compute(ctx)?;
                }
                Ok(())
            }
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

    /// Device-resident memory goes back before the CPU reads guest memory for a launch.
    fn software_compute(&mut self, ctx: &mut ExecCtx) -> Result<()> {
        if self.three_d.flush_renderer(ctx)? == Flush::Pending {
            self.resume = Some(Resume::SoftwareCompute);
            return Err(Error::GpuSyncPending);
        }
        self.compute.run_software(ctx);
        Ok(())
    }

    /// Stop at this method until a sync the backend has not finished lands.
    fn wait_for(
        &mut self,
        synced: Flush,
        subchannel: u32,
        method: u32,
        arg: u32,
        last_call: bool,
    ) -> Result<()> {
        if synced == Flush::Done {
            return Ok(());
        }
        self.resume = Some(Resume::Method {
            subchannel,
            method,
            arg,
            last_call,
        });
        Err(Error::GpuSyncPending)
    }

    /// The channel's host methods, answered on any subchannel.
    fn host_method(&mut self, slot: usize, method: u32, arg: u32, ctx: &mut ExecCtx) -> Result<()> {
        self.gpfifo_regs.set(method, arg);
        match method {
            SET_OBJECT => {
                let class = if arg & !BIND_CLASS_MASK != 0
                    && arg & !(BIND_ENGINE_MASK | BIND_CLASS_MASK) == 0
                {
                    arg & BIND_CLASS_MASK
                } else {
                    arg
                };
                self.subchannel_class[slot] = class;
                if ctx.trace {
                    crate::traceln!("[gpu] subchannel {} bound to class {:#x}", slot, class);
                }
                Ok(())
            }
            GPFIFO_SEMAPHORE => {
                const OPERATION_RELEASE: u32 = 2;
                const RELEASE_SIZE_4_BYTES: u32 = 1;
                if field(arg, 0, 4) != OPERATION_RELEASE {
                    // Acquires are satisfied: a submission completes before the ioctl returns.
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
            // The long-form release, payload in the argument.
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
            // Nop, cache maintenance and reference counts.
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
            let mem = Memory::new();
            mem.map_zero(0x3000_0000, 0x4000).unwrap();
            let mut vmm = AddressSpace::new();
            let base = vmm
                .map(0x3000_0000, 0x4000, 1, 0, SMALL_PAGE_SIZE, 0, 0)
                .unwrap();
            Harness {
                mem,
                vmm,
                host1x: Host1x::new(),
                stats: GpuStats::default(),
                base,
            }
        }

        fn ctx(&mut self) -> ExecCtx<'_> {
            ExecCtx {
                mem: &self.mem,
                vmm: &self.vmm,
                as_id: 0,
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
        // Subchannel 6 is pre-bound to the GPFIFO class.
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
        let pb = pushbuffer(&[
            header(3, 1, 0, 0),
            CLASS_3D,
            header(3, 3, 0, 0x360),
            7,
            8,
            9,
        ]);
        let mut ctx = h.ctx();
        chan.run_pushbuffer(&pb, &mut ctx).unwrap();
        assert_eq!(chan.three_d.regs.get(0x360), 9);
        assert_eq!(chan.three_d.regs.get(0x361), 0);
    }

    #[test]
    fn increase_once_advances_only_after_the_first_word() {
        let mut h = Harness::new();
        let mut chan = Channel::new(1, 8);
        let pb = pushbuffer(&[
            header(3, 1, 0, 0),
            CLASS_3D,
            header(5, 3, 0, 0x360),
            7,
            8,
            9,
        ]);
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
        // The stream is continuous across GPFIFO entries.
        let mut h = Harness::new();
        let mut chan = Channel::new(1, 8);
        let mut ctx = h.ctx();
        chan.run_pushbuffer(&pushbuffer(&[header(3, 1, 0, 0), CLASS_3D]), &mut ctx)
            .unwrap();
        chan.run_pushbuffer(&pushbuffer(&[header(1, 4, 0, 0x360), 1, 2]), &mut ctx)
            .unwrap();
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
        chan.run_pushbuffer(&pushbuffer(&[header(3, 1, 0, 0), CLASS_3D]), &mut ctx)
            .unwrap();
        chan.run_pushbuffer(&pushbuffer(&[header(5, 3, 0, 0x360), 7]), &mut ctx)
            .unwrap();
        chan.run_pushbuffer(&pushbuffer(&[8, 9]), &mut ctx).unwrap();
        assert_eq!(chan.three_d.regs.get(0x360), 7);
        assert_eq!(chan.three_d.regs.get(0x361), 9);
    }

    #[test]
    fn a_data_word_is_never_read_as_a_header() {
        // A payload that looks like an end-of-segment header still reaches its method.
        let mut h = Harness::new();
        let mut chan = Channel::new(1, 8);
        let pb = pushbuffer(&[
            header(3, 1, 0, 0),
            CLASS_3D,
            header(1, 2, 0, 0x360),
            0xE000_0000,
            7,
        ]);
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
        // Modes 0 and 2 use the low two count bits for a second opcode.
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
        let pb = pushbuffer(&[
            3 << 16,
            header(3, 1, 0, 0),
            CLASS_3D,
            header(4, 0x55, 0, 0x360),
        ]);
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
        // Host methods work on the 3D subchannel too.
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
            h.mem
                .write_u8(u64::from(0x3000_0000 + i as u32), *b)
                .unwrap();
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
