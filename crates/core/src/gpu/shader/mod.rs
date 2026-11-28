//! Shader programs as the guest leaves them in GPU memory. Decoding, interpreting and
//! translating them is [`switch_shader`]'s; this reads them out of memory and gives the
//! interpreter memory-backed [`sources`].

pub mod sources;

use crate::gpu::exec::ExecCtx;
use crate::{Error, Result};
use switch_shader::{decode_program_with_consts, isa, uses, Op, Program, ProgramHeader};

/// Words to scan for `exit` before giving up.
pub const MAX_PROGRAM_WORDS: u64 = 8192;

/// Mesa-compiled binaries carry a 0x50-byte header; if slot 1 does not decode,
/// the decode is retried past it.
const MESA_SHADER_HEADER_BYTES: u64 = 0x50;

/// Decode a bound shader program out of guest memory, stripping `sched` words.
pub fn decode_program_from_memory(
    ctx: &ExecCtx,
    addr: u64,
    bindings: &dyn Fn(u8) -> Option<(u64, u32)>,
) -> Result<Program> {
    decode_program_from_memory_recording(ctx, addr, bindings).map(|(program, _)| program)
}

/// Guest memory a decode read, so a cached result can be validated.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct DecodeReads {
    /// Pages of the program's words, ascending and deduplicated, as GPU virtual
    /// addresses (translate through `ExecCtx::vmm` before watching).
    pub pages: Vec<u64>,
    /// Constant-buffer words read (only for `brx` jump tables), with their values.
    pub consts: Vec<(u64, u32)>,
}

impl DecodeReads {
    /// Whether guest memory still holds what this decode was built from. Only
    /// constant words are re-read; program pages are expected to be watched.
    pub fn constants_unchanged(&self, ctx: &ExecCtx) -> bool {
        self.consts
            .iter()
            .all(|&(at, value)| ctx.read_u32(at).is_ok_and(|now| now == value))
    }
}

/// [`decode_program_from_memory`], also returning [`DecodeReads`].
pub fn decode_program_from_memory_recording(
    ctx: &ExecCtx,
    addr: u64,
    bindings: &dyn Fn(u8) -> Option<(u64, u32)>,
) -> Result<(Program, DecodeReads)> {
    let mut pages: Vec<u64> = Vec::new();
    let mut consts: Vec<(u64, u32)> = Vec::new();
    let note = |pages: &mut Vec<u64>, at: u64| {
        let page = at >> crate::mem::PAGE_BITS;
        if !pages.contains(&page) {
            pages.push(page);
        }
    };
    let first_real_word = ctx.read_u64(addr + 8)?;
    note(&mut pages, addr + 8);
    let header_at =
        matches!(isa::decode(first_real_word).op, Op::Unimplemented { .. }).then_some(addr);
    let header = match header_at {
        Some(at) => {
            let target_at = at + ProgramHeader::OMAP_TARGET_WORD * 4;
            let target = ctx.read_u32(target_at)?;
            let flags = ctx.read_u32(target_at + 4)?;
            note(&mut pages, target_at);
            note(&mut pages, target_at + 4);
            Some(ProgramHeader::from_omap(target, flags))
        }
        None => None,
    };
    let addr = match header_at {
        Some(at) => at + MESA_SHADER_HEADER_BYTES,
        None => addr,
    };
    let limit = MAX_PROGRAM_WORDS * 8;
    let mut program = decode_program_with_consts(
        &mut |offset: u32| {
            if u64::from(offset) >= limit {
                return Err(switch_shader::Error::Fault(format!(
                    "shader: program read at {offset:#x} is past the {limit:#x}-byte cap"
                )));
            }
            note(&mut pages, addr + u64::from(offset));
            Ok(ctx.read_u64(addr + u64::from(offset))?)
        },
        &mut |bank: u8, offset: u32| {
            let (base, size) = bindings(bank).ok_or_else(|| {
                switch_shader::Error::Fault(format!("shader: constant bank {bank} is unbound"))
            })?;
            if offset + 4 > size {
                return Err(switch_shader::Error::Fault(format!(
                    "shader: read of c{bank}[{offset:#x}] is past its {size:#x}-byte end"
                )));
            }
            let at = base + u64::from(offset);
            let value = ctx.read_u32(at)?;
            consts.push((at, value));
            Ok(value)
        },
    )
    .inspect(|program| {
        // `--trace shader` prints every decoded program in walk order.
        if crate::trace::enabled(crate::trace::Trace::Shader) {
            crate::traceln!(
                "[shader] program at {addr:#x}, {} instructions",
                program.offsets.len()
            );
            for (i, &off) in program.offsets.iter().enumerate() {
                let raw = ctx.read_u64(addr + u64::from(off)).unwrap_or(0);
                crate::traceln!("  {off:#06x}: {raw:016x} {:?}", program.insns[i]);
            }
        }
    })
    .map_err(Error::from)?;
    program.header = header;
    if crate::trace::enabled(crate::trace::Trace::Sph) {
        crate::traceln!("[sph] program at {addr:#x} {header:?}");
    }
    pages.sort_unstable();
    uses::note_decoded(addr, &program);
    Ok((program, DecodeReads { pages, consts }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::exec::GpuStats;
    use crate::gpu::syncpt::Host1x;
    use crate::gpu::vmm::{AddressSpace, SMALL_PAGE_SIZE};
    use crate::mem::Memory;
    use switch_shader::testing::solid_fragment_shader;

    fn decode_recording() -> (u64, DecodeReads) {
        let mut mem = Memory::new();
        mem.map_zero(0x7000_0000, 0x4000).unwrap();
        let mut vmm = AddressSpace::new();
        let at = vmm
            .map(0x7000_0000, 0x4000, 1, 0, SMALL_PAGE_SIZE, 0, 0)
            .unwrap();
        let mut host1x = Host1x::new();
        let mut stats = GpuStats::default();
        let mut ctx = ExecCtx {
            mem: &mut mem,
            vmm: &vmm,
            as_id: 0,
            host1x: &mut host1x,
            stats: &mut stats,
            trace: false,
        };
        for (i, chunk) in solid_fragment_shader()
            .as_chunks::<4>()
            .0
            .iter()
            .enumerate()
        {
            let word = u32::from_le_bytes(*chunk);
            ctx.write_u32(at + i as u64 * 4, word).unwrap();
        }
        let (program, reads) = decode_program_from_memory_recording(&ctx, at, &|_| None).unwrap();
        assert!(!program.insns.is_empty(), "decoded something");
        (at, reads)
    }

    #[test]
    fn a_decode_reports_the_pages_it_read_so_a_cache_can_watch_them() {
        // The pages read are GPU virtual pages.
        let (at, reads) = decode_recording();
        assert_eq!(
            reads.pages,
            vec![at >> crate::mem::PAGE_BITS],
            "the one page this program lives on"
        );
        // No `brx`, so no constant words to re-check.
        assert!(reads.consts.is_empty(), "no constant words consumed");
    }

    #[test]
    fn a_memory_fault_under_a_decode_comes_back_as_the_error_memory_gave() {
        // Zero words fall through, so the walk reads on until it leaves the one mapped page.
        let mut mem = Memory::new();
        mem.map_zero(0x7000_0000, 0x1000).unwrap();
        let mut vmm = AddressSpace::new();
        let at = vmm
            .map(0x7000_0000, 0x1000, 1, 0, SMALL_PAGE_SIZE, 0, 0)
            .unwrap();
        let mut host1x = Host1x::new();
        let mut stats = GpuStats::default();
        let ctx = ExecCtx {
            mem: &mut mem,
            vmm: &vmm,
            as_id: 0,
            host1x: &mut host1x,
            stats: &mut stats,
            trace: false,
        };
        let direct = ctx.read_u64(at + 0x1000).unwrap_err();
        let decoded = decode_program_from_memory(&ctx, at, &|_| None).unwrap_err();
        assert_eq!(decoded, direct);
    }
}
