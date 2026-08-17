//! A GPU channel: the unit of work submission.
//!
//! Userspace pushes 64-bit GPFIFO entries, each pointing at a pushbuffer in
//! GPU memory. The channel's command processor (PFIFO) walks a pushbuffer,
//! decodes its method headers, and routes each method write to whichever class
//! is bound to that header's subchannel.

use crate::gpu::engine::compute::EngineCompute;
use crate::gpu::engine::copy::EngineCopy;
use crate::gpu::engine::inline::{EngineInline, METHOD_RANGE as INLINE_METHODS};
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

/// Maximum pushbuffer length (in dwords) the GPFIFO entry can express.
const MAX_PUSHBUFFER_WORDS: u32 = 0x1F_FFFF;

// MAXWELL_CHANNEL_GPFIFO_A methods.
const GPFIFO_SEMAPHORE_OFFSET: u32 = 0x004;
const GPFIFO_SEMAPHORE_PAYLOAD: u32 = 0x006;
const GPFIFO_SEMAPHORE: u32 = 0x007;
const GPFIFO_SYNCPOINT: u32 = 0x01D;

/// Header submission modes (`SubmissionMode` in deko3d's `command.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Increasing,
    NonIncreasing,
    Inline,
    IncreaseOnce,
}

impl Mode {
    fn from_bits(bits: u32) -> Option<Mode> {
        match bits {
            1 => Some(Mode::Increasing),
            3 => Some(Mode::NonIncreasing),
            4 => Some(Mode::Inline),
            5 => Some(Mode::IncreaseOnce),
            _ => None,
        }
    }
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
    pub inline: EngineInline,
    pub compute: EngineCompute,
    /// MAXWELL_CHANNEL_GPFIFO_A's own register file.
    pub gpfifo_regs: Registers,
    /// Size of the GPFIFO ring the guest allocated.
    pub gpfifo_entries: u32,
}

impl Channel {
    pub fn new(id: u32, syncpt: u32) -> Channel {
        Channel {
            id,
            as_id: None,
            syncpt,
            subchannel_class: [0; SUBCHANNEL_COUNT],
            three_d: Engine3D::new(),
            two_d: Engine2D::new(),
            copy: EngineCopy::new(),
            inline: EngineInline::new(),
            compute: EngineCompute::new(),
            gpfifo_regs: Registers::new(),
            gpfifo_entries: 0,
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
    pub fn run_pushbuffer(&mut self, bytes: &[u8], ctx: &mut ExecCtx) -> Result<()> {
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut i = 0usize;
        while i < words.len() {
            let header = words[i];
            i += 1;
            if header == 0 {
                // Pushbuffer padding.
                continue;
            }
            let method = field(header, 0, 12);
            let subchannel = field(header, 13, 15);
            let arg = field(header, 16, 28);
            let mode = Mode::from_bits(field(header, 29, 31)).ok_or_else(|| {
                Error::Gpu(format!(
                    "pfifo: unsupported submission mode {} in header {:#010x}",
                    field(header, 29, 31),
                    header
                ))
            })?;

            if mode == Mode::Inline {
                self.method(subchannel, method, arg, true, ctx)?;
                continue;
            }
            let count = arg as usize;
            if i + count > words.len() {
                return Err(Error::Gpu(format!(
                    "pfifo: header {:#010x} claims {} words but only {} remain",
                    header,
                    count,
                    words.len() - i
                )));
            }
            for n in 0..count {
                let target = match mode {
                    Mode::Increasing => method + n as u32,
                    Mode::NonIncreasing => method,
                    Mode::IncreaseOnce => method + (n > 0) as u32,
                    Mode::Inline => unreachable!("handled above"),
                };
                self.method(subchannel, target, words[i + n], n + 1 == count, ctx)?;
            }
            i += count;
        }
        Ok(())
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
        if method == SET_OBJECT {
            self.subchannel_class[slot] = arg;
            if ctx.trace {
                eprintln!("[gpu] subchannel {} bound to class {:#x}", subchannel, arg);
            }
            return Ok(());
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
                // The 3D class also exposes the inline-to-memory methods, and
                // deko3d sends them on the 3D subchannel.
                if INLINE_METHODS.contains(&method) {
                    self.inline.write(method, arg, ctx)?;
                }
                self.three_d.write(method, arg, last_call, ctx)
            }
            CLASS_INLINE => self.inline.write(method, arg, ctx),
            CLASS_2D => self.two_d.write(method, arg, ctx),
            CLASS_COPY => self.copy.write(method, arg, ctx),
            CLASS_COMPUTE => self.compute.write(method, arg, ctx),
            CLASS_GPFIFO => self.gpfifo_method(method, arg, ctx),
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

    /// MAXWELL_CHANNEL_GPFIFO_A: the channel's own semaphore and syncpoint
    /// operations.
    fn gpfifo_method(&mut self, method: u32, arg: u32, ctx: &mut ExecCtx) -> Result<()> {
        self.gpfifo_regs.set(method, arg);
        match method {
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
            GPFIFO_SYNCPOINT => {
                const OPERATION_INCR: u32 = 1;
                let id = field(arg, 8, 15);
                if field(arg, 0, 0) == OPERATION_INCR {
                    ctx.host1x.increment(id)?;
                }
                Ok(())
            }
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
    fn truncated_method_group_is_rejected() {
        let mut h = Harness::new();
        let mut chan = Channel::new(1, 8);
        let pb = pushbuffer(&[header(3, 1, 0, 0), CLASS_3D, header(1, 4, 0, 0x360), 1]);
        let mut ctx = h.ctx();
        assert!(chan.run_pushbuffer(&pb, &mut ctx).is_err());
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
