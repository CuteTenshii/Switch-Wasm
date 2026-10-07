//! Vertex attribute fetch tests.

use super::super::attrib::{
    ATTRIB_SIZE_10_10_10_2, ATTRIB_TYPE_FLOAT, ATTRIB_TYPE_SINT, ATTRIB_TYPE_SNORM,
    ATTRIB_TYPE_SSCALED, ATTRIB_TYPE_UINT, ATTRIB_TYPE_UNORM, ATTRIB_TYPE_USCALED,
};
use super::*;
use crate::gpu::engine::threed::{VertexArray, VertexAttrib};
use crate::gpu::exec::ExecCtx;

pub(super) fn harness() -> (Memory, AddressSpace, u64) {
    let mut mem = Memory::new();
    mem.map_zero(0x6000_0000, 0x1000).unwrap();
    let mut vmm = AddressSpace::new();
    let gpu_va = vmm
        .map(0x6000_0000, 0x1000, 1, 0, SMALL_PAGE_SIZE, 0, 0)
        .unwrap();
    (mem, vmm, gpu_va)
}

#[test]
fn fetch_attribute_reads_a_float32_vec4_by_stride() {
    let (mut mem, vmm, base) = harness();
    // Vertex 1's position at base + stride*1 + offset(0).
    let stride = 16u32;
    let addr = base + stride as u64;
    for (i, v) in [1.0f32, 2.0, 3.0, 4.0].iter().enumerate() {
        vmm.write_u32(&mut mem, addr + i as u64 * 4, v.to_bits())
            .unwrap();
    }
    let mut stats = Default::default();
    let mut host1x = Host1x::new();
    let ctx = ExecCtx {
        mem: &mut mem,
        vmm: &vmm,
        host1x: &mut host1x,
        stats: &mut stats,
        trace: false,
    };

    let attrib = VertexAttrib {
        buffer_id: 0,
        is_fixed: false,
        offset: 0,
        size: 0x01,
        ty: ATTRIB_TYPE_FLOAT,
        is_bgra: false,
    };
    let array = VertexArray {
        enabled: true,
        stride,
        start: base,
        limit: base + 0x1000,
        divisor: 0,
    };

    let v = fetch_attribute(attrib, array, 1, &ctx).unwrap();
    assert_eq!(v, [1.0, 2.0, 3.0, 4.0]);
}

/// Red all ones, green the largest positive ten-bit value, blue zero, and
/// alpha `10` (-2, clamped to -1).
#[test]
fn fetch_attribute_unpacks_10_10_10_2() {
    let (mut mem, vmm, base) = harness();
    let word = 0x3ff | 0x1ff << 10 | 0b10 << 30;
    vmm.write_u32(&mut mem, base, word).unwrap();
    let mut stats = Default::default();
    let mut host1x = Host1x::new();
    let ctx = ExecCtx {
        mem: &mut mem,
        vmm: &vmm,
        host1x: &mut host1x,
        stats: &mut stats,
        trace: false,
    };
    let array = VertexArray {
        enabled: true,
        stride: 4,
        start: base,
        limit: base + 0x1000,
        divisor: 0,
    };
    let fetch = |ty| {
        let attrib = VertexAttrib {
            buffer_id: 0,
            is_fixed: false,
            offset: 0,
            size: ATTRIB_SIZE_10_10_10_2,
            ty,
            is_bgra: false,
        };
        fetch_attribute(attrib, array, 0, &ctx).unwrap()
    };
    assert_eq!(fetch(ATTRIB_TYPE_SNORM), [-1.0 / 511.0, 1.0, 0.0, -1.0]);
    assert_eq!(
        fetch(ATTRIB_TYPE_UNORM),
        [1.0, 511.0 / 1023.0, 0.0, 2.0 / 3.0]
    );
    let bits = |v: [i32; 4]| v.map(|x| f32::from_bits(x as u32));
    assert_eq!(
        fetch(ATTRIB_TYPE_SINT).map(f32::to_bits),
        bits([-1, 511, 0, -2]).map(f32::to_bits)
    );
    assert_eq!(
        fetch(ATTRIB_TYPE_UINT).map(f32::to_bits),
        bits([1023, 511, 0, 2]).map(f32::to_bits)
    );
}

#[test]
fn fetch_attribute_past_the_limit_reads_zeros() {
    let (mut mem, vmm, base) = harness();
    let stride = 16u32;
    for vertex in 0..2u64 {
        for i in 0..4u64 {
            let at = base + vertex * u64::from(stride) + i * 4;
            vmm.write_u32(&mut mem, at, 7.0f32.to_bits()).unwrap();
        }
    }
    let mut stats = Default::default();
    let mut host1x = Host1x::new();
    let ctx = ExecCtx {
        mem: &mut mem,
        vmm: &vmm,
        host1x: &mut host1x,
        stats: &mut stats,
        trace: false,
    };
    let attrib = |size| VertexAttrib {
        buffer_id: 0,
        is_fixed: false,
        offset: 0,
        size,
        ty: ATTRIB_TYPE_FLOAT,
        is_bgra: false,
    };
    // One vertex's worth: its last valid byte is the fifteenth.
    let array = VertexArray {
        enabled: true,
        stride,
        start: base,
        limit: base + u64::from(stride) - 1,
        divisor: 0,
    };
    assert_eq!(
        fetch_attribute(attrib(0x01), array, 0, &ctx).unwrap(),
        [7.0; 4]
    );
    assert_eq!(
        fetch_attribute(attrib(0x01), array, 1, &ctx).unwrap(),
        [0.0; 4],
        "memory past the limit holds sevens, and is not read"
    );
    // Three components: the fourth keeps its default.
    assert_eq!(
        fetch_attribute(attrib(0x02), array, 1, &ctx).unwrap(),
        [0.0, 0.0, 0.0, 1.0]
    );
}

#[test]
fn fetch_attribute_unpacks_unorm8_and_honours_is_bgra() {
    let (mut mem, vmm, base) = harness();
    // Packed BGRA8: B=0x40, G=0x80, R=0xff, A=0x00 (little-endian word).
    let packed = 0x00u32 << 24 | 0xffu32 << 16 | 0x80u32 << 8 | 0x40u32;
    vmm.write_u32(&mut mem, base, packed).unwrap();
    let mut stats = Default::default();
    let mut host1x = Host1x::new();
    let ctx = ExecCtx {
        mem: &mut mem,
        vmm: &vmm,
        host1x: &mut host1x,
        stats: &mut stats,
        trace: false,
    };

    let attrib = VertexAttrib {
        buffer_id: 0,
        is_fixed: false,
        offset: 0,
        size: 0x0a,
        ty: ATTRIB_TYPE_UNORM,
        is_bgra: true,
    };
    let array = VertexArray {
        enabled: true,
        stride: 4,
        start: base,
        limit: base + 0x1000,
        divisor: 0,
    };

    let v = fetch_attribute(attrib, array, 0, &ctx).unwrap();
    // Decoded as BGRA then swapped to RGBA: R=0xff, G=0x80, B=0x40, A=0x00.
    assert_eq!(v, [1.0, 0x80 as f32 / 255.0, 0x40 as f32 / 255.0, 0.0]);
}

#[test]
fn fetch_attribute_unpacks_the_eight_bit_integer_and_normalised_types() {
    let (mut mem, vmm, base) = harness();
    // 0x7F, 0x80, 0x01, 0xFF as four bytes: signed 127, -128, 1, -1.
    let packed = 0xFFu32 << 24 | 0x01u32 << 16 | 0x80u32 << 8 | 0x7Fu32;
    vmm.write_u32(&mut mem, base, packed).unwrap();
    let mut stats = Default::default();
    let mut host1x = Host1x::new();
    let ctx = ExecCtx {
        mem: &mut mem,
        vmm: &vmm,
        host1x: &mut host1x,
        stats: &mut stats,
        trace: false,
    };
    let array = VertexArray {
        enabled: true,
        stride: 4,
        start: base,
        limit: base + 0x1000,
        divisor: 0,
    };
    let fetch = |ty| {
        let attrib = VertexAttrib {
            buffer_id: 0,
            is_fixed: false,
            offset: 0,
            size: 0x0a,
            ty,
            is_bgra: false,
        };
        fetch_attribute(attrib, array, 0, &ctx).unwrap()
    };

    let sint = fetch(ATTRIB_TYPE_SINT);
    let as_int = |v: f32| v.to_bits() as i32;
    assert_eq!(
        [
            as_int(sint[0]),
            as_int(sint[1]),
            as_int(sint[2]),
            as_int(sint[3])
        ],
        [127, -128, 1, -1],
        "sint8 sign-extends, and keeps its bits rather than its value"
    );

    let uint = fetch(ATTRIB_TYPE_UINT);
    assert_eq!(
        [
            uint[0].to_bits(),
            uint[1].to_bits(),
            uint[2].to_bits(),
            uint[3].to_bits()
        ],
        [0x7F, 0x80, 0x01, 0xFF],
        "uint8 is zero-extended"
    );

    let snorm = fetch(ATTRIB_TYPE_SNORM);
    assert_eq!(snorm[0], 1.0);
    assert_eq!(snorm[1], -1.0, "-128 clamps onto -1 rather than past it");
    assert_eq!(snorm[3], -1.0 / 127.0);
}

#[test]
fn fetch_attribute_unpacks_the_sixteen_bit_types() {
    let (mut mem, vmm, base) = harness();
    // 1.0, -2.0, 0.5, 65504 (the largest finite half) as four halves.
    let halves: [u16; 4] = [0x3C00, 0xC000, 0x3800, 0x7BFF];
    let packed = halves
        .iter()
        .enumerate()
        .fold(0u64, |acc, (i, &h)| acc | u64::from(h) << (i * 16));
    vmm.write_u64(&mut mem, base, packed).unwrap();
    // The signed pattern eight bytes on: 0x8000 is -1, as is 0x8001; 0x7FFF is +1.
    vmm.write_u64(&mut mem, base + 8, 0x0001_7FFF_8000_8001)
        .unwrap();
    let mut stats = Default::default();
    let mut host1x = Host1x::new();
    let ctx = ExecCtx {
        mem: &mut mem,
        vmm: &vmm,
        host1x: &mut host1x,
        stats: &mut stats,
        trace: false,
    };
    let array = VertexArray {
        enabled: true,
        stride: 16,
        start: base,
        limit: base + 0x1000,
        divisor: 0,
    };
    let fetch = |size, ty, offset| {
        let attrib = VertexAttrib {
            buffer_id: 0,
            is_fixed: false,
            offset,
            size,
            ty,
            is_bgra: false,
        };
        fetch_attribute(attrib, array, 0, &ctx).unwrap()
    };

    assert_eq!(fetch(0x03, ATTRIB_TYPE_FLOAT, 0), [1.0, -2.0, 0.5, 65504.0]);
    // Fewer than four components pads `(0, 0, 0, 1)`.
    assert_eq!(fetch(0x0f, ATTRIB_TYPE_FLOAT, 0), [1.0, -2.0, 0.0, 1.0]);
    assert_eq!(fetch(0x05, ATTRIB_TYPE_FLOAT, 0), [1.0, -2.0, 0.5, 1.0]);
    assert_eq!(fetch(0x1b, ATTRIB_TYPE_FLOAT, 0), [1.0, 0.0, 0.0, 1.0]);

    let sint = fetch(0x03, ATTRIB_TYPE_SINT, 0);
    let as_int = |v: f32| v.to_bits() as i32;
    assert_eq!(
        [
            as_int(sint[0]),
            as_int(sint[1]),
            as_int(sint[2]),
            as_int(sint[3])
        ],
        [0x3C00, -0x4000, 0x3800, 0x7BFF],
        "sint16 sign-extends, and keeps its bits rather than its value"
    );

    let uint = fetch(0x03, ATTRIB_TYPE_UINT, 0);
    assert_eq!(
        [
            uint[0].to_bits(),
            uint[1].to_bits(),
            uint[2].to_bits(),
            uint[3].to_bits()
        ],
        [0x3C00, 0xC000, 0x3800, 0x7BFF],
        "uint16 is zero-extended"
    );

    let unorm = fetch(0x03, ATTRIB_TYPE_UNORM, 0);
    assert_eq!(unorm[0], 0x3C00 as f32 / 65535.0);

    let snorm = fetch(0x03, ATTRIB_TYPE_SNORM, 8);
    assert_eq!(snorm[0], -1.0, "-32767 is -1");
    assert_eq!(snorm[1], -1.0, "-32768 clamps onto -1 rather than past it");
    assert_eq!(snorm[2], 1.0);
    assert_eq!(snorm[3], 1.0 / 32767.0);
}

/// Narrow integer attributes (sizes `0x1d` and `0x18`): `w` is integer
/// one, and a scaled one is its value.
#[test]
fn fetch_attribute_unpacks_the_narrow_eight_bit_shapes() {
    let (mut mem, vmm, base) = harness();
    // At the mapping's end, so a whole-word read would fault.
    let at = base + 0xffd;
    // 0x80, 0x7f, 0xff in the three bytes from `at`.
    vmm.write_u32(&mut mem, at - 1, 0xff7f_8000).unwrap();
    let mut stats = Default::default();
    let mut host1x = Host1x::new();
    let ctx = ExecCtx {
        mem: &mut mem,
        vmm: &vmm,
        host1x: &mut host1x,
        stats: &mut stats,
        trace: false,
    };
    let array = VertexArray {
        enabled: true,
        stride: 3,
        start: at,
        limit: 0,
        divisor: 0,
    };
    let fetch = |size, ty| {
        let attrib = VertexAttrib {
            buffer_id: 0,
            is_fixed: false,
            offset: 0,
            size,
            ty,
            is_bgra: false,
        };
        fetch_attribute(attrib, array, 0, &ctx).unwrap()
    };
    let bits = |v: [f32; 4]| v.map(f32::to_bits);

    assert_eq!(bits(fetch(0x1d, ATTRIB_TYPE_UINT)), [0x80, 0, 0, 1]);
    assert_eq!(bits(fetch(0x18, ATTRIB_TYPE_UINT)), [0x80, 0x7f, 0, 1]);
    assert_eq!(
        bits(fetch(0x13, ATTRIB_TYPE_SINT)),
        [(-128i32) as u32, 0x7f, u32::MAX, 1]
    );
    assert_eq!(
        fetch(0x18, ATTRIB_TYPE_UNORM),
        [128.0 / 255.0, 127.0 / 255.0, 0.0, 1.0]
    );
    assert_eq!(fetch(0x1d, ATTRIB_TYPE_SNORM), [-1.0, 0.0, 0.0, 1.0]);
    assert_eq!(fetch(0x13, ATTRIB_TYPE_USCALED), [128.0, 127.0, 255.0, 1.0]);
    assert_eq!(fetch(0x13, ATTRIB_TYPE_SSCALED), [-128.0, 127.0, -1.0, 1.0]);
}

#[test]
fn fetch_attribute_from_a_disabled_buffer_is_an_error() {
    let (mut mem, vmm, base) = harness();
    let mut stats = Default::default();
    let mut host1x = Host1x::new();
    let ctx = ExecCtx {
        mem: &mut mem,
        vmm: &vmm,
        host1x: &mut host1x,
        stats: &mut stats,
        trace: false,
    };
    let attrib = VertexAttrib {
        buffer_id: 0,
        is_fixed: false,
        offset: 0,
        size: 0x01,
        ty: ATTRIB_TYPE_FLOAT,
        is_bgra: false,
    };
    let array = VertexArray {
        enabled: false,
        stride: 16,
        start: base,
        limit: base,
        divisor: 0,
    };
    assert!(fetch_attribute(attrib, array, 0, &ctx).is_err());
}

#[test]
fn a_fixed_attribute_reads_the_vec4_default_instead_of_failing_the_draw() {
    // A fixed attribute the shader never reads must not drop the draw.
    let (mut mem, vmm, base) = harness();
    let mut stats = Default::default();
    let mut host1x = Host1x::new();
    let ctx = ExecCtx {
        mem: &mut mem,
        vmm: &vmm,
        host1x: &mut host1x,
        stats: &mut stats,
        trace: false,
    };
    let attrib = VertexAttrib {
        buffer_id: 0,
        is_fixed: true,
        offset: 0,
        size: 0x12,
        ty: ATTRIB_TYPE_FLOAT,
        is_bgra: false,
    };
    // A disabled array: a fixed attribute is not fetched from it.
    let array = VertexArray {
        enabled: false,
        stride: 0,
        start: base,
        limit: base,
        divisor: 0,
    };
    assert_eq!(
        fetch_attribute(attrib, array, 0, &ctx).unwrap(),
        [0.0, 0.0, 0.0, 1.0]
    );
}
