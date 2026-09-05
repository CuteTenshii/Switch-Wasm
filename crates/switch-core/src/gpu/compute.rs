//! Running a compute dispatch.
//!
//! What [`crate::gpu::raster`] is to a draw, this is to a launch: the software
//! reference that turns the engine's state into the memory a kernel was
//! supposed to write. It reads the [`Qmd`], decodes the program the way a draw
//! decodes its shaders, and runs one [`Invocation`] per thread of the grid.
//!
//! The interpreter is scalar, so a CTA's threads run one after another rather
//! than in lockstep. That is exact for everything except a barrier and a warp
//! shuffle, the two places a thread's progress depends on the others':
//! threads run to the next `bar`, and only once every one of them has arrived
//! does any of them continue; a `shfl` suspends the same way and is answered
//! once its warp has caught up. Since nothing runs concurrently, an atomic needs no locking
//! and a race cannot be observed: a kernel whose result depends on one gets a
//! valid answer here and a different one on hardware.

use crate::gpu::engine::compute::EngineCompute;
use crate::gpu::exec::ExecCtx;
use crate::gpu::qmd::{ConstantBuffer, Qmd, Release, CONSTANT_BUFFERS, QMD_WORDS};
use crate::gpu::shader::compiled::Compiled;
use crate::gpu::shader::interp::{
    resolve_shuffles, ConstCache, ConstantSource, Env, GlobalMemory, Halt, Invocation,
    ShaderResult, SharedMemory, TextureSource, WARP_LANES,
};
use crate::gpu::shader::{decode_program_from_memory, Op};
use crate::gpu::texture::{self, BlockCache, Descriptors};
use crate::{Error, Result};
use std::cell::RefCell;

/// The most threads one dispatch may run.
///
/// Not a hardware limit: hardware would run a grid this size in microseconds.
/// It is a liveness guard for the browser, where the whole GPU stack runs on
/// one worker thread: a grid of a million interpreted threads is a tab that
/// stops answering, and a refused dispatch that says so is worth more than
/// that.
pub const MAX_DISPATCH_THREADS: u64 = 1 << 20;

/// Run `engine.last_dispatch`.
pub fn dispatch(engine: &EngineCompute, ctx: &mut ExecCtx) -> Result<()> {
    let launch = engine
        .last_dispatch
        .ok_or_else(|| Error::Gpu("compute: a dispatch with no QMD address".into()))?;

    let mut words = [0u32; QMD_WORDS];
    for (i, word) in words.iter_mut().enumerate() {
        *word = ctx.read_u32(launch.qmd_addr + i as u64 * 4)?;
    }
    let qmd = Qmd::parse(&words)?;
    if ctx.trace {
        crate::traceln!(
            "[gpu] compute grid {:?} block {:?} shared={:#x} program={:#x}",
            qmd.cta_raster,
            qmd.cta_threads,
            qmd.shared_memory_size,
            qmd.program_offset
        );
    }

    if qmd.is_empty() {
        return release(&qmd, ctx);
    }
    let threads = qmd.cta_count() * u64::from(qmd.threads_per_cta());
    if threads > MAX_DISPATCH_THREADS {
        return Err(Error::Gpu(format!(
            "compute: a grid of {:?} CTAs of {:?} is {threads} threads, past the \
             {MAX_DISPATCH_THREADS} this runs in software",
            qmd.cta_raster, qmd.cta_threads
        )));
    }

    let program_addr = engine.program_region() + u64::from(qmd.program_offset);
    let memory = DispatchMemory {
        ctx: RefCell::new(ctx),
        banks: qmd.constant_buffers,
        consts: RefCell::new(ConstCache::default()),
        tex_header_pool: engine.tex_header_pool(),
        tex_sampler_pool: engine.tex_sampler_pool(),
        descriptors: RefCell::new(crate::IdMap::default()),
        blocks: RefCell::new(BlockCache::default()),
    };

    let program = {
        let ctx = memory.ctx.borrow();
        decode_program_from_memory(&ctx, program_addr, &|bank: u8| memory.bank(bank))?
    };
    let program = Compiled::with_constants(&program, &memory);

    let shared: SharedMemory = RefCell::new(vec![0u8; qmd.shared_memory_size as usize]);
    let mut env = Env::with_tex_cb_index(&memory, &memory, engine.tex_cb_index());
    env.memory = Some(&memory);
    env.shared = Some(&shared);
    env.special.shared_size = qmd.shared_memory_size;
    env.special.local_size = qmd.local_memory_size;

    // A program with neither a barrier nor a warp shuffle needs no scheduler
    // and no per-thread state kept alive, which is the common case and much
    // the cheaper one. Both are places a thread's progress depends on the
    // others', and nothing else is.
    let cooperative = program
        .ops()
        .iter()
        .any(|op| matches!(op, Op::Bar { .. } | Op::Shfl { .. }));
    let mut threads = Threads::new(&qmd, cooperative);

    for z in 0..qmd.cta_raster[2] {
        for y in 0..qmd.cta_raster[1] {
            for x in 0..qmd.cta_raster[0] {
                shared.borrow_mut().fill(0);
                env.special.ctaid = [x, y, z];
                threads.run_cta(&program, &mut env, &qmd)?;
            }
        }
    }

    let mut ctx = memory.ctx.borrow_mut();
    release(&qmd, &mut ctx)
}

/// The invocations a CTA is run with.
enum Threads {
    /// One invocation reused for every thread, run to completion in turn.
    Serial(Box<Invocation>),
    /// One invocation per thread, all of them live at once because a barrier
    /// suspends a thread in the middle of its program.
    Cooperative(Vec<Invocation>),
}

impl Threads {
    fn new(qmd: &Qmd, cooperative: bool) -> Threads {
        let local = qmd.local_memory_size as usize;
        let fresh = || {
            let mut invocation = Invocation::new();
            invocation.set_local_bytes(local);
            invocation
        };
        if cooperative {
            Threads::Cooperative((0..qmd.threads_per_cta()).map(|_| fresh()).collect())
        } else {
            Threads::Serial(Box::new(fresh()))
        }
    }

    fn run_cta(&mut self, program: &Compiled, env: &mut Env, qmd: &Qmd) -> Result<()> {
        match self {
            Threads::Serial(invocation) => {
                for thread in 0..qmd.threads_per_cta() {
                    env.special.tid = tid(thread, qmd);
                    invocation.reset();
                    invocation.execute(program, env)?;
                }
                Ok(())
            }
            Threads::Cooperative(invocations) => {
                for invocation in invocations.iter_mut() {
                    invocation.reset();
                }
                let mut waiting = vec![true; invocations.len()];
                // Each pass runs every thread that is still going until it
                // exits, reaches a barrier, or reaches a shuffle. A pass that
                // ends with nothing waiting is the barrier every thread
                // arrived at, released.
                loop {
                    let mut arrived = false;
                    let mut shuffled = false;
                    for (thread, invocation) in invocations.iter_mut().enumerate() {
                        if !waiting[thread] {
                            continue;
                        }
                        env.special.lane = thread as u32 % WARP_LANES as u32;
                        env.special.tid = tid(thread as u32, qmd);
                        match invocation.resume(program, env)? {
                            Halt::Exited => waiting[thread] = false,
                            Halt::Barrier => arrived = true,
                            Halt::Shuffle => shuffled = true,
                        }
                    }
                    // A shuffle reaches across one warp, not the whole CTA:
                    // threads are numbered in the order they were launched,
                    // so a warp is a run of [`WARP_LANES`] of them.
                    if shuffled {
                        for warp in invocations.chunks_mut(WARP_LANES) {
                            resolve_shuffles(warp);
                        }
                    }
                    if !arrived && !shuffled {
                        return Ok(());
                    }
                }
            }
        }
    }
}

/// A thread's position in its CTA, x fastest.
fn tid(thread: u32, qmd: &Qmd) -> [u32; 3] {
    let [width, height, _] = qmd.cta_threads;
    let plane = width * height;
    [thread % width, (thread / width) % height, thread / plane]
}

/// Write whichever release semaphores the launch asked for.
fn release(qmd: &Qmd, ctx: &mut ExecCtx) -> Result<()> {
    for release in qmd.releases.into_iter().flatten() {
        let Release {
            addr,
            payload,
            one_word,
        } = release;
        if one_word {
            ctx.write_u32(addr, payload)?;
        } else {
            ctx.write_u64(addr, u64::from(payload))?;
            ctx.write_u64(addr + 8, ctx.stats.submissions)?;
        }
    }
    Ok(())
}

/// Everything a dispatch's threads read and write, over one borrow of the
/// execution context.
///
/// The interpreter reads constants and textures and writes global memory from
/// the same instruction stream, and a write needs the context mutably, so the
/// one mutable borrow lives here and every access goes through it. Nothing
/// re-enters, so no two of those borrows overlap.
struct DispatchMemory<'a, 'b> {
    ctx: RefCell<&'a mut ExecCtx<'b>>,
    /// The QMD's constant buffers. The bind slot is the index: entry `i` is
    /// what the program reads as `c[i]`.
    banks: [Option<ConstantBuffer>; CONSTANT_BUFFERS],
    consts: RefCell<ConstCache>,
    tex_header_pool: u64,
    tex_sampler_pool: u64,
    descriptors: RefCell<crate::IdMap<u32, Descriptors>>,
    blocks: RefCell<BlockCache>,
}

impl DispatchMemory<'_, '_> {
    fn bank(&self, bank: u8) -> Option<(u64, u32)> {
        self.banks
            .get(bank as usize)
            .copied()
            .flatten()
            .map(|c| (c.addr, c.size))
    }
}

impl ConstantSource for DispatchMemory<'_, '_> {
    fn read_const(&self, bank: u8, offset: u16) -> ShaderResult<u32> {
        let key = ConstCache::key(bank, offset);
        if let Some(value) = self.consts.borrow().get(key) {
            return Ok(value);
        }
        let (addr, size) = self.bank(bank).ok_or_else(|| {
            Box::new(Error::Gpu(format!(
                "compute: read from constant bank {bank}, which the QMD did not bind"
            )))
        })?;
        if u32::from(offset) + 4 > size {
            return Err(Box::new(Error::Gpu(format!(
                "compute: constant read c{bank}[{offset:#x}] is past the bound \
                 buffer's size {size:#x}"
            ))));
        }
        let value = self.ctx.borrow().read_u32(addr + u64::from(offset))?;
        self.consts.borrow_mut().insert(key, value);
        Ok(value)
    }
}

impl GlobalMemory for DispatchMemory<'_, '_> {
    fn read_u32(&self, addr: u64) -> ShaderResult<u32> {
        Ok(self.ctx.borrow().read_u32(addr)?)
    }

    fn read_u8(&self, addr: u64) -> ShaderResult<u8> {
        Ok(self.ctx.borrow().vmm_read_u8(addr)?)
    }

    fn write_u32(&self, addr: u64, value: u32) -> ShaderResult<()> {
        Ok(self.ctx.borrow_mut().write_u32(addr, value)?)
    }

    fn write_u8(&self, addr: u64, value: u8) -> ShaderResult<()> {
        Ok(self.ctx.borrow_mut().vmm_write_u8(addr, value)?)
    }
}

impl TextureSource for DispatchMemory<'_, '_> {
    fn sample(&self, handle: u32, u: f32, v: f32, layer: u32) -> ShaderResult<[f32; 4]> {
        let cached = self.descriptors.borrow().get(&handle).copied();
        let ctx = self.ctx.borrow();
        let descriptors = match cached {
            Some(d) => d,
            None => {
                let d = texture::read_descriptors(
                    &ctx,
                    self.tex_header_pool,
                    self.tex_sampler_pool,
                    handle,
                )?;
                self.descriptors.borrow_mut().insert(handle, d);
                d
            }
        };
        Ok(texture::sample_with(
            &ctx,
            &descriptors,
            f64::from(u),
            f64::from(v),
            layer,
            &self.blocks,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::exec::GpuStats;
    use crate::gpu::shader::isa::{self, MemSize, Operand};
    use crate::gpu::syncpt::Host1x;
    use crate::gpu::vmm::{AddressSpace, SMALL_PAGE_SIZE};
    use crate::mem::Memory;

    const SR_TIDX: u8 = 0x21;
    const SR_CTAIDX: u8 = 0x25;
    const RZ: u8 = isa::RZ;

    /// The guard-predicate field holding `PT`, which is every instruction
    /// here: none of them is predicated.
    const PT: u64 = 7 << 16;

    /// A control-flow instruction's condition-code test field holding `T`.
    const FLOW_TEST_T: u64 = 0xF;

    /// Assemble one instruction, and check it against the decoder, which is
    /// what makes these encodings trustworthy rather than a second guess at
    /// the same tables.
    fn encode(word: u64, expected: Op) -> u64 {
        assert_eq!(isa::decode(word).op, expected, "encoded {word:#018x}");
        word
    }

    fn mov32i(dst: u8, imm: u32) -> u64 {
        encode(
            0x0100_0000_0000_0000 | u64::from(imm) << 20 | 0xF << 12 | PT | u64::from(dst),
            Op::Mov32i { dst, imm },
        )
    }

    fn s2r(dst: u8, sr: u8) -> u64 {
        encode(
            0xf0c8u64 << 48 | u64::from(sr) << 20 | PT | u64::from(dst),
            Op::S2r { dst, sr },
        )
    }

    /// `iscadd dst, a, b, shift`: `(a << shift) + b`, which is every
    /// "index into an array" one of these kernels does.
    fn iscadd(dst: u8, a: u8, b: u8, shift: u8) -> u64 {
        encode(
            0x5c18u64 << 48
                | u64::from(shift) << 39
                | u64::from(b) << 20
                | PT
                | u64::from(a) << 8
                | u64::from(dst),
            Op::Iscadd {
                dst,
                a,
                aneg: false,
                b: Operand::Reg(b),
                bneg: false,
                shift,
            },
        )
    }

    /// The size field lives in the three opcode bits the group's mask leaves
    /// free, so `b32` is the base opcode `| 4`.
    fn stg(addr: u8, offset: u32, src: u8) -> u64 {
        encode(
            (0xeed8u64 | 4) << 48
                | (u64::from(offset) & 0xFF_FFFF) << 20
                | PT
                | u64::from(addr) << 8
                | u64::from(src),
            Op::Stg {
                addr,
                offset: offset as i32,
                src,
                size: MemSize::B32,
            },
        )
    }

    fn sts(addr: u8, offset: u32, src: u8) -> u64 {
        encode(
            (0xef58u64 | 4) << 48
                | (u64::from(offset) & 0xFF_FFFF) << 20
                | PT
                | u64::from(addr) << 8
                | u64::from(src),
            Op::Sts {
                addr,
                offset: offset as i32,
                src,
                size: MemSize::B32,
            },
        )
    }

    /// `shfl.bfly p0, dst, src, index, mask`, the lane whose number differs
    /// from this one's in the bits `index` names.
    fn shfl_bfly(dst: u8, src: u8, index: u32, mask: u32) -> u64 {
        encode(
            0xef10u64 << 48
                | u64::from(mask) << 34
                | 3 << 30
                | 1 << 29
                | 1 << 28
                | u64::from(index) << 20
                | PT
                | u64::from(src) << 8
                | u64::from(dst),
            Op::Shfl {
                dst,
                pred: 0,
                src,
                index: Operand::Imm(index),
                mask: Operand::Imm(mask),
                mode: isa::ShflMode::Bfly,
            },
        )
    }

    fn lds(dst: u8, addr: u8, offset: u32) -> u64 {
        encode(
            (0xef48u64 | 4) << 48
                | (u64::from(offset) & 0xFF_FFFF) << 20
                | PT
                | u64::from(addr) << 8
                | u64::from(dst),
            Op::Lds {
                dst,
                addr,
                offset: offset as i32,
                size: MemSize::B32,
            },
        )
    }

    fn bar_sync() -> u64 {
        encode(
            0xf0a8u64 << 48 | 1 << 39 | PT,
            Op::Bar {
                mode: isa::BarMode::Sync,
            },
        )
    }

    fn atom_add_u32(dst: u8, addr: u8, src: u8) -> u64 {
        encode(
            0xed00u64 << 48 | u64::from(src) << 20 | PT | u64::from(addr) << 8 | u64::from(dst),
            Op::Atom {
                dst,
                addr,
                offset: 0,
                src,
                op: isa::AtomOp::Add,
                ty: isa::AtomType::U32,
                space: isa::AtomSpace::Global,
            },
        )
    }

    /// `exit` with the condition-code test a compiler emits: `T`. The field
    /// left at zero is `F`, which is an `exit` that never fires.
    fn exit() -> u64 {
        encode(0xe300_0000_0000_0000 | PT | FLOW_TEST_T, Op::Exit)
    }

    /// Lay instructions out the way a real binary does: one `sched` control
    /// word then three instructions, per 32-byte block.
    fn blocks(insns: &[u64]) -> Vec<u64> {
        let mut out = Vec::new();
        for chunk in insns.chunks(3) {
            out.push(0);
            out.extend_from_slice(chunk);
        }
        out
    }

    /// Where each piece of a test launch lives, as offsets from the mapping.
    const QMD_AT: u64 = 0;
    const PROGRAM_AT: u64 = 0x1000;
    const OUTPUT_AT: u64 = 0x2000;

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
            mem.map_zero(0x3000_0000, 0x1_0000).unwrap();
            let mut vmm = AddressSpace::new();
            let base = vmm
                .map(0x3000_0000, 0x1_0000, 1, 0, SMALL_PAGE_SIZE, 0, 0)
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
                mem: &mut self.mem,
                vmm: &self.vmm,
                host1x: &mut self.host1x,
                stats: &mut self.stats,
                trace: false,
            }
        }

        fn write_words(&mut self, at: u64, words: &[u32]) {
            let base = self.base;
            let mut ctx = self.ctx();
            for (i, word) in words.iter().enumerate() {
                ctx.write_u32(base + at + i as u64 * 4, *word).unwrap();
            }
        }

        fn write_program(&mut self, insns: &[u64]) {
            let words: Vec<u32> = blocks(insns)
                .iter()
                .flat_map(|w| [*w as u32, (*w >> 32) as u32])
                .collect();
            self.write_words(PROGRAM_AT, &words);
        }

        fn read_output(&mut self, count: usize) -> Vec<u32> {
            let base = self.base;
            let ctx = self.ctx();
            (0..count)
                .map(|i| ctx.read_u32(base + OUTPUT_AT + i as u64 * 4).unwrap())
                .collect()
        }

        /// An engine whose program region and QMD address point at this
        /// harness's memory, ready for the trigger.
        fn engine(&self) -> EngineCompute {
            let mut engine = EngineCompute::new();
            let program = self.base + PROGRAM_AT;
            engine.regs.set(0x582, (program >> 32) as u32);
            engine.regs.set(0x583, program as u32);
            engine.last_dispatch = Some(crate::gpu::engine::compute::Dispatch {
                qmd_addr: self.base + QMD_AT,
            });
            engine
        }
    }

    /// A QMD as words, with only the fields these tests vary.
    #[derive(Default)]
    struct Launch {
        grid: [u32; 3],
        block: [u32; 3],
        shared: u32,
        release: Option<(u64, u32)>,
    }

    impl Launch {
        fn words(&self) -> Vec<u32> {
            let mut words = [0u32; QMD_WORDS];
            let mut set = |lo: u32, hi: u32, value: u32| {
                for bit in 0..=(hi - lo) {
                    let at = (lo + bit) as usize;
                    if value >> bit & 1 != 0 {
                        words[at / 32] |= 1 << (at % 32);
                    }
                }
            };
            set(576, 579, 6);
            set(384, 415, self.grid[0]);
            set(416, 431, self.grid[1]);
            set(432, 447, self.grid[2]);
            set(592, 607, self.block[0]);
            set(608, 623, self.block[1]);
            set(624, 639, self.block[2]);
            set(544, 561, self.shared);
            if let Some((addr, payload)) = self.release {
                set(202, 202, 1);
                set(736, 767, addr as u32);
                set(768, 775, (addr >> 32) as u32);
                set(799, 799, 1);
                set(800, 831, payload);
            }
            words.to_vec()
        }
    }

    /// Run a launch and hand back the harness it wrote into.
    fn run(harness: &mut Harness, launch: &Launch, insns: &[u64]) -> Result<()> {
        let words = launch.words();
        harness.write_words(QMD_AT, &words);
        harness.write_program(insns);
        let engine = harness.engine();
        let mut ctx = harness.ctx();
        dispatch(&engine, &mut ctx)
    }

    #[test]
    fn every_thread_of_every_cta_writes_its_own_slot() {
        // The whole point of a dispatch: a grid of threads that differ only
        // in what `s2r` tells them. If the thread and CTA registers were
        // still the zero they used to read, all eight would write slot 0.
        let mut h = Harness::new();
        let out = h.base + OUTPUT_AT;
        let program = [
            s2r(0, SR_TIDX),
            s2r(1, SR_CTAIDX),
            // The block is four wide, so the linear index is ctaid*4 + tid.
            iscadd(0, 1, 0, 2),
            mov32i(4, out as u32),
            iscadd(2, 0, 4, 2),
            mov32i(3, (out >> 32) as u32),
            stg(2, 0, 0),
            exit(),
        ];
        let launch = Launch {
            grid: [2, 1, 1],
            block: [4, 1, 1],
            ..Launch::default()
        };
        run(&mut h, &launch, &program).unwrap();
        assert_eq!(h.read_output(8), vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn a_thread_sees_what_another_thread_of_its_cta_wrote_before_the_barrier() {
        // Each thread publishes its own id into shared memory and then reads
        // thread 3's slot. Threads run one at a time here, so without the
        // barrier releasing them together thread 0 would read a slot nothing
        // had written yet and the answer would be [0, 0, 0, 3].
        let mut h = Harness::new();
        let out = h.base + OUTPUT_AT;
        let program = [
            s2r(0, SR_TIDX),
            iscadd(5, 0, RZ, 2),
            sts(5, 0, 0),
            bar_sync(),
            lds(6, RZ, 12),
            mov32i(4, out as u32),
            iscadd(2, 0, 4, 2),
            mov32i(3, (out >> 32) as u32),
            stg(2, 0, 6),
            exit(),
        ];
        let launch = Launch {
            grid: [1, 1, 1],
            block: [4, 1, 1],
            shared: 64,
            ..Launch::default()
        };
        run(&mut h, &launch, &program).unwrap();
        assert_eq!(h.read_output(4), vec![3, 3, 3, 3]);
    }

    #[test]
    fn a_thread_reads_another_lane_of_its_warp_through_a_shuffle() {
        // Each thread puts its own id in r0 and then reads the id of the
        // lane beside it. Threads run one at a time here, so the exchange
        // only has an answer because a shuffle suspends the thread the same
        // way a barrier does.
        let mut h = Harness::new();
        let out = h.base + OUTPUT_AT;
        let program = [
            s2r(0, SR_TIDX),
            shfl_bfly(1, 0, 1, 0x1f),
            mov32i(4, out as u32),
            iscadd(2, 0, 4, 2),
            mov32i(3, (out >> 32) as u32),
            stg(2, 0, 1),
            exit(),
        ];
        let launch = Launch {
            grid: [1, 1, 1],
            block: [4, 1, 1],
            ..Launch::default()
        };
        run(&mut h, &launch, &program).unwrap();
        assert_eq!(h.read_output(4), vec![1, 0, 3, 2]);
    }

    #[test]
    fn shared_memory_does_not_carry_from_one_cta_to_the_next() {
        // Same kernel, two CTAs, and the second must publish its own slot 3
        // rather than inherit the first's, which it would if the block were
        // allocated once for the grid.
        let mut h = Harness::new();
        let out = h.base + OUTPUT_AT;
        let program = [
            s2r(0, SR_TIDX),
            iscadd(5, 0, RZ, 2),
            // Publish tid + 1 so an unwritten slot (zero) is distinguishable.
            iscadd(7, 0, RZ, 0),
            mov32i(6, 1),
            iscadd(7, 7, 6, 0),
            sts(5, 0, 7),
            bar_sync(),
            lds(6, RZ, 4),
            s2r(1, SR_CTAIDX),
            iscadd(1, 1, 0, 1),
            mov32i(4, out as u32),
            iscadd(2, 1, 4, 2),
            mov32i(3, (out >> 32) as u32),
            stg(2, 0, 6),
            exit(),
        ];
        let launch = Launch {
            grid: [2, 1, 1],
            block: [2, 1, 1],
            shared: 32,
            ..Launch::default()
        };
        run(&mut h, &launch, &program).unwrap();
        // Every thread reads slot 1, which its own CTA's thread 1 wrote as 2.
        assert_eq!(h.read_output(4), vec![2, 2, 2, 2]);
    }

    #[test]
    fn an_atomic_accumulates_every_thread_of_the_grid() {
        let mut h = Harness::new();
        let out = h.base + OUTPUT_AT;
        let program = [
            mov32i(2, out as u32),
            mov32i(3, (out >> 32) as u32),
            mov32i(1, 1),
            atom_add_u32(0, 2, 1),
            exit(),
        ];
        let launch = Launch {
            grid: [3, 2, 1],
            block: [4, 1, 1],
            ..Launch::default()
        };
        run(&mut h, &launch, &program).unwrap();
        assert_eq!(h.read_output(1), vec![24]);
    }

    #[test]
    fn the_release_semaphore_is_written_once_the_grid_has_run() {
        let mut h = Harness::new();
        let at = h.base + OUTPUT_AT + 0x100;
        let launch = Launch {
            grid: [1, 1, 1],
            block: [1, 1, 1],
            release: Some((at, 0xABCD)),
            ..Launch::default()
        };
        run(&mut h, &launch, &[exit()]).unwrap();
        assert_eq!(h.ctx().read_u32(at).unwrap(), 0xABCD);
    }

    #[test]
    fn a_launch_of_nothing_still_releases_its_semaphore() {
        // A zero-width grid is a legal launch, and a guest waiting on its
        // fence waits forever if "no work" means "no release".
        let mut h = Harness::new();
        let at = h.base + OUTPUT_AT + 0x100;
        let launch = Launch {
            grid: [0, 1, 1],
            block: [1, 1, 1],
            release: Some((at, 0x99)),
            ..Launch::default()
        };
        run(&mut h, &launch, &[exit()]).unwrap();
        assert_eq!(h.ctx().read_u32(at).unwrap(), 0x99);
    }

    #[test]
    fn a_grid_past_what_this_runs_in_software_is_refused_rather_than_started() {
        let mut h = Harness::new();
        let launch = Launch {
            grid: [0x10_0000, 4, 1],
            block: [64, 1, 1],
            ..Launch::default()
        };
        let err = run(&mut h, &launch, &[exit()]).unwrap_err();
        assert!(format!("{err:?}").contains("past the"), "got {err:?}");
    }

    #[test]
    fn a_kernel_the_interpreter_cannot_follow_fails_the_dispatch_and_writes_nothing() {
        let mut h = Harness::new();
        let out = h.base + OUTPUT_AT;
        // Not an instruction this decoder knows, with a `PT` guard, or the
        // predicate would skip it and the kernel would run clean.
        let unknown = 0xffff_ffff_ff00_0000 | PT;
        assert!(matches!(isa::decode(unknown).op, Op::Unimplemented { .. }));
        let program = [
            mov32i(2, out as u32),
            mov32i(3, (out >> 32) as u32),
            mov32i(0, 0x1234),
            stg(2, 0, 0),
            unknown,
            exit(),
        ];
        let launch = Launch {
            grid: [1, 1, 1],
            block: [1, 1, 1],
            ..Launch::default()
        };
        let err = run(&mut h, &launch, &program).unwrap_err();
        assert!(format!("{err:?}").contains("unimplemented"), "got {err:?}");
    }

    #[test]
    fn a_thread_index_walks_x_fastest() {
        let qmd = Qmd::parse(&{
            let mut words = [0u32; QMD_WORDS];
            let launch = Launch {
                block: [2, 3, 2],
                ..Launch::default()
            };
            words.copy_from_slice(&launch.words());
            words
        })
        .unwrap();
        assert_eq!(tid(0, &qmd), [0, 0, 0]);
        assert_eq!(tid(1, &qmd), [1, 0, 0]);
        assert_eq!(tid(2, &qmd), [0, 1, 0]);
        assert_eq!(tid(5, &qmd), [1, 2, 0]);
        assert_eq!(tid(6, &qmd), [0, 0, 1]);
        assert_eq!(tid(11, &qmd), [1, 2, 1]);
    }
}
