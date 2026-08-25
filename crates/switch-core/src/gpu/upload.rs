//! What a draw reads out of guest memory.
//!
//! [`crate::gpu::shader::wgsl`] says how to shade a draw and
//! [`crate::gpu::pipeline`] says what state it runs under. Neither says what
//! it draws. That is in guest memory — vertices at a GPU virtual address,
//! indices at another, constants in whatever the driver bound — and a device
//! cannot read guest memory. Somebody has to translate the addresses, bound
//! the ranges and hand over bytes, and this is that.
//!
//! The software rasterizer never needed this. It reads a vertex attribute at
//! a time, through the GPU MMU, exactly when a vertex shader asks for it, and
//! a buffer that a draw does not touch costs nothing. A GPU backend has to
//! decide up front what to upload, which turns "read this word" into "how
//! much of this buffer is this draw actually going to look at" — a question
//! the register file does not answer directly.
//!
//! # Bounding what a draw touches
//!
//! A vertex array says where it starts and where it ends, and the end is
//! often the end of a heap rather than the end of the mesh. What bounds an
//! upload is the draw: `first` and `count` for a sequential draw, and for an
//! indexed one the lowest and highest index in the index buffer, which has to
//! be read to be known. Doing that here is not wasted work — the indices have
//! to be uploaded anyway.
//!
//! # It has a ceiling, on purpose
//!
//! A stride and a count that multiply to something absurd are not a reason to
//! allocate it. [`MAX_UPLOAD`] is the point at which this reports rather than
//! tries, because the failure mode of not having one is a machine in swap.

use crate::gpu::engine::threed::{Engine3D, ShaderStage};
use crate::gpu::exec::ExecCtx;
use crate::gpu::pipeline::{Pipeline, StepMode};
use crate::{Error, Result};

/// Which constant banks to resolve.
///
/// The distinction is not fussiness. A bank is up to 64 KiB and the Home Menu
/// binds eight of them per draw while its shaders read two, so the difference
/// between these two answers is 190 KiB a draw and 60 KiB a draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Banks<'a> {
    /// Every bank the draw has bound, which is a fact about the draw.
    Bound,
    /// Only these, which is a fact about the shaders: the `const_banks` of
    /// each stage's [`crate::gpu::shader::wgsl::Translation`], paired with
    /// the stage it came from.
    Read(&'a [(ShaderStage, u32)]),
}

/// The most one buffer will be read into memory: 64 MiB.
///
/// Larger than any mesh a draw addresses and far smaller than the heap a
/// vertex array's limit usually points at the end of.
pub const MAX_UPLOAD: u64 = 64 << 20;

/// How many constant banks a bind slot has.
const CONSTBUF_BANKS: u32 = 32;

/// The index width a backend is handed. Maxwell also has an 8-bit form and
/// WebGPU does not, so [`Uploads::of`] widens that to 16.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexFormat {
    Uint16,
    Uint32,
}

/// One vertex array's bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexUpload {
    /// Which of Maxwell's vertex arrays this is, matching
    /// [`crate::gpu::pipeline::VertexBuffer::index`].
    pub array: u32,
    /// The element these bytes start at. A backend either offsets the buffer
    /// binding by `first * stride` or adds `first` to its base vertex; what
    /// it must not do is assume element zero, since a draw that starts at
    /// vertex 900 uploads from there.
    pub first: u32,
    pub stride: u32,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexUpload {
    pub format: IndexFormat,
    pub bytes: Vec<u8>,
    /// The lowest and highest index the draw uses, which is what bounds the
    /// vertex uploads.
    pub lowest: u32,
    pub highest: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstantUpload {
    pub stage: ShaderStage,
    pub bank: u32,
    pub bytes: Vec<u8>,
}

/// Everything a draw reads, resolved.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Uploads {
    pub vertex: Vec<VertexUpload>,
    pub index: Option<IndexUpload>,
    pub constants: Vec<ConstantUpload>,
}

impl Uploads {
    /// How many bytes this draw would move to a device.
    pub fn len(&self) -> usize {
        self.vertex.iter().map(|v| v.bytes.len()).sum::<usize>()
            + self.index.as_ref().map_or(0, |i| i.bytes.len())
            + self.constants.iter().map(|c| c.bytes.len()).sum::<usize>()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Resolve what [`Engine3D::last_draw`] reads.
    ///
    /// `pipeline` supplies the vertex layout, so that this and the pipeline
    /// description cannot disagree about which arrays a draw binds or how
    /// they step.
    pub fn of(
        engine: &Engine3D,
        pipeline: &Pipeline,
        ctx: &ExecCtx,
        banks: Banks<'_>,
    ) -> Result<Uploads> {
        let call = engine.last_draw;
        let index = if call.indexed {
            Some(read_indices(
                ctx,
                engine.index_array_start(),
                call.first,
                call.count,
                call.index_format,
            )?)
        } else {
            None
        };

        let mut vertex = Vec::new();
        for buffer in &pipeline.vertex_buffers {
            let array = engine.vertex_array(buffer.index);
            // What the draw reaches: an instanced array advances once per
            // instance, and the engine issues one instance per draw, so the
            // element is the instance id and there is exactly one of it.
            let (first, count) = match buffer.step {
                StepMode::Instance => (engine.instance_id(), 1),
                StepMode::Vertex => match &index {
                    Some(index) => (index.lowest, index.highest - index.lowest + 1),
                    None => (call.first, call.count),
                },
            };
            if count == 0 || buffer.stride == 0 {
                continue;
            }
            let length = u64::from(count) * u64::from(buffer.stride);
            let start = array.start + u64::from(first) * u64::from(buffer.stride);
            // The array's own limit is the real end of the mapping, and it is
            // the address of the *last valid byte* rather than one past it —
            // a 32-byte array at `0x204730000` has a limit of `0x20473001f`.
            // A draw that runs past it is reading something else's memory,
            // and saying so is better than uploading it.
            if array.limit != 0 && start + length > array.limit + 1 {
                return Err(Error::Gpu(format!(
                    "upload: vertex array {} reads {start:#x}..{:#x}, past its limit {:#x}",
                    buffer.index,
                    start + length,
                    array.limit
                )));
            }
            vertex.push(VertexUpload {
                array: buffer.index,
                first,
                stride: buffer.stride,
                bytes: read_range(ctx, start, length, "vertex array")?,
            });
        }

        let mut constants = Vec::new();
        for stage in [ShaderStage::VertexB, ShaderStage::Fragment] {
            for bank in 0..CONSTBUF_BANKS {
                if let Banks::Read(wanted) = banks {
                    if !wanted.contains(&(stage, bank)) {
                        continue;
                    }
                }
                let Some((addr, size)) = engine.bound_constbuf(stage, bank) else {
                    continue;
                };
                if size == 0 {
                    continue;
                }
                constants.push(ConstantUpload {
                    stage,
                    bank,
                    bytes: read_range(ctx, addr, u64::from(size), "constant bank")?,
                });
            }
        }

        Ok(Uploads { vertex, index, constants })
    }
}

/// Read a draw's indices, widening the 8-bit form WebGPU does not have.
fn read_indices(
    ctx: &ExecCtx,
    base: u64,
    first: u32,
    count: u32,
    format: u32,
) -> Result<IndexUpload> {
    let (width, out_format) = match format {
        // Widened, not passed through: a backend has nowhere to put an 8-bit
        // index, and the alternative is every backend widening it itself.
        0 => (1u64, IndexFormat::Uint16),
        1 => (2, IndexFormat::Uint16),
        2 => (4, IndexFormat::Uint32),
        other => return Err(Error::Gpu(format!("upload: unknown index format {other}"))),
    };
    let out_width = if out_format == IndexFormat::Uint16 { 2 } else { 4 };
    if u64::from(count) * out_width > MAX_UPLOAD {
        return Err(Error::Gpu(format!(
            "upload: {count} indices is past the {MAX_UPLOAD}-byte cap"
        )));
    }

    let mut bytes = Vec::with_capacity(count as usize * out_width as usize);
    let mut lowest = u32::MAX;
    let mut highest = 0u32;
    for ordinal in 0..count {
        let at = base + u64::from(first + ordinal) * width;
        let value = match width {
            1 => u32::from(ctx.vmm_read_u8(at)?),
            2 => u32::from(ctx.vmm_read_u8(at)?) | (u32::from(ctx.vmm_read_u8(at + 1)?) << 8),
            _ => ctx.read_u32(at)?,
        };
        lowest = lowest.min(value);
        highest = highest.max(value);
        match out_format {
            IndexFormat::Uint16 => bytes.extend_from_slice(&(value as u16).to_le_bytes()),
            IndexFormat::Uint32 => bytes.extend_from_slice(&value.to_le_bytes()),
        }
    }
    if count == 0 {
        lowest = 0;
    }
    Ok(IndexUpload { format: out_format, bytes, lowest, highest })
}

/// `len` bytes from a GPU virtual address.
///
/// A word at a time where the range allows it: the address translation and
/// the page lookup are per access, not per byte, and a mesh read a byte at a
/// time pays for both eight times over.
fn read_range(ctx: &ExecCtx, gpu_va: u64, len: u64, what: &str) -> Result<Vec<u8>> {
    if len > MAX_UPLOAD {
        return Err(Error::Gpu(format!(
            "upload: {what} at {gpu_va:#x} is {len} bytes, past the {MAX_UPLOAD}-byte cap"
        )));
    }
    let mut out = Vec::with_capacity(len as usize);
    let mut at = gpu_va;
    let end = gpu_va + len;
    while at < end {
        if at.is_multiple_of(4) && end - at >= 4 {
            out.extend_from_slice(&ctx.read_u32(at)?.to_le_bytes());
            at += 4;
        } else {
            out.push(ctx.vmm_read_u8(at)?);
            at += 1;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::vmm::AddressSpace;
    use crate::gpu::{GpuStats, Host1x};
    use crate::mem::Memory;

    /// Guest memory with one page mapped, and the GPU address it is at.
    struct Harness {
        mem: Memory,
        vmm: AddressSpace,
        host1x: Host1x,
        stats: GpuStats,
        base: u64,
    }

    impl Harness {
        fn new(size: u32) -> Harness {
            let mut mem = Memory::new();
            mem.map_zero(0x3000_0000, size as usize).unwrap();
            let mut vmm = AddressSpace::new();
            let base = vmm.map(0x3000_0000, size as u64, 1, 0, 0x1000, 0, 0).unwrap();
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

        fn write(&mut self, offset: u64, bytes: &[u8]) {
            for (i, &byte) in bytes.iter().enumerate() {
                self.mem.write_u8(0x3000_0000 + offset as u32 + i as u32, byte).unwrap();
            }
        }
    }

    #[test]
    fn an_eight_bit_index_is_widened_because_webgpu_has_no_such_format() {
        let mut h = Harness::new(0x1000);
        h.write(0, &[3, 1, 2]);
        let base = h.base;
        let indices = read_indices(&h.ctx(), base, 0, 3, 0).unwrap();
        assert_eq!(indices.format, IndexFormat::Uint16);
        assert_eq!(indices.bytes, vec![3, 0, 1, 0, 2, 0]);
    }

    #[test]
    fn the_index_range_is_what_bounds_a_vertex_upload() {
        // Nothing else says how much of a vertex array an indexed draw
        // reaches: the array's own limit is usually the end of a heap.
        let mut h = Harness::new(0x1000);
        h.write(0, &[9, 0, 5, 0, 7, 0]);
        let base = h.base;
        let indices = read_indices(&h.ctx(), base, 0, 3, 1).unwrap();
        assert_eq!((indices.lowest, indices.highest), (5, 9));
    }

    #[test]
    fn a_thirty_two_bit_index_is_passed_through() {
        let mut h = Harness::new(0x1000);
        h.write(0, &1u32.to_le_bytes());
        h.write(4, &0x1234_5678u32.to_le_bytes());
        let base = h.base;
        let indices = read_indices(&h.ctx(), base, 0, 2, 2).unwrap();
        assert_eq!(indices.format, IndexFormat::Uint32);
        assert_eq!(indices.lowest, 1);
        assert_eq!(indices.highest, 0x1234_5678);
    }

    #[test]
    fn the_first_index_is_an_offset_into_the_index_buffer() {
        // For an indexed draw `first` counts indices, not vertices — the
        // vertex it lands on is whatever the index there says.
        let mut h = Harness::new(0x1000);
        h.write(0, &[0, 0, 0, 42, 0, 0]);
        let base = h.base;
        let indices = read_indices(&h.ctx(), base, 3, 1, 0).unwrap();
        assert_eq!((indices.lowest, indices.highest), (42, 42));
    }

    #[test]
    fn an_index_count_past_the_ceiling_is_reported_rather_than_allocated() {
        // The failure mode of having no ceiling is a machine in swap.
        let mut h = Harness::new(0x1000);
        let base = h.base;
        assert!(read_indices(&h.ctx(), base, 0, u32::MAX, 2).is_err());
    }

    #[test]
    fn a_range_past_the_ceiling_is_reported_before_it_is_read() {
        let mut h = Harness::new(0x1000);
        let base = h.base;
        assert!(read_range(&h.ctx(), base, MAX_UPLOAD + 1, "test").is_err());
    }

    #[test]
    fn a_range_reads_the_same_bytes_however_it_is_aligned() {
        // Words where the range allows and bytes at the edges: a mesh read a
        // byte at a time pays for an address translation eight times over,
        // and the two paths have to agree.
        let mut h = Harness::new(0x1000);
        let bytes: Vec<u8> = (0..32u8).collect();
        h.write(0, &bytes);
        let base = h.base;
        let ctx = h.ctx();
        assert_eq!(read_range(&ctx, base, 32, "test").unwrap(), bytes);
        assert_eq!(read_range(&ctx, base + 1, 30, "test").unwrap(), bytes[1..31]);
        assert_eq!(read_range(&ctx, base + 3, 5, "test").unwrap(), bytes[3..8]);
        assert_eq!(read_range(&ctx, base, 0, "test").unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn a_drawing_with_nothing_in_it_moves_no_bytes() {
        let uploads = Uploads::default();
        assert!(uploads.is_empty());
        assert_eq!(uploads.len(), 0);
    }

    #[test]
    fn the_total_is_every_buffer_a_draw_would_move() {
        let uploads = Uploads {
            vertex: vec![VertexUpload { array: 0, first: 0, stride: 8, bytes: vec![0; 32] }],
            index: Some(IndexUpload {
                format: IndexFormat::Uint16,
                bytes: vec![0; 12],
                lowest: 0,
                highest: 3,
            }),
            constants: vec![ConstantUpload {
                stage: ShaderStage::VertexB,
                bank: 1,
                bytes: vec![0; 256],
            }],
        };
        assert_eq!(uploads.len(), 300);
    }
}
