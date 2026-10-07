//! Texture and surface instructions.

use super::*;

/// `suld`/`sust`, fields as Eden's `surface_load_store.cpp` reads them.
pub(super) fn decode_surface(insn: u64) -> Option<Op> {
    const IGN: u64 = 0;
    if field(insn, 49, 2) != IGN {
        return None;
    }
    let raw = field(insn, 52, 1) != 0;
    if raw && field(insn, 23, 1) != 0 {
        return None;
    }
    let dim = match field(insn, 33, 3) {
        0 => SurfaceDim::D1,
        1 => SurfaceDim::Buffer1d,
        2 => SurfaceDim::Array1d,
        3 => SurfaceDim::D2,
        4 => SurfaceDim::Array2d,
        5 => SurfaceDim::D3,
        _ => return None,
    };
    let data = if raw {
        SurfaceData::Raw(match field(insn, 20, 3) {
            0 => SurfaceSize::U8,
            1 => SurfaceSize::S8,
            2 => SurfaceSize::U16,
            3 => SurfaceSize::S16,
            4 => SurfaceSize::B32,
            5 => SurfaceSize::B64,
            6 => SurfaceSize::B128,
            _ => return None,
        })
    } else {
        let swizzle = field(insn, 20, 4);
        SurfaceData::Formatted(std::array::from_fn(|i| swizzle >> i & 1 != 0))
    };
    let store = field(insn, 53, 1) != 0;
    match data {
        SurfaceData::Formatted(mask) if store && mask != [true; 4] => return None,
        SurfaceData::Formatted([false, false, false, false]) => return None,
        _ => {}
    }
    let bound = field(insn, 51, 1) != 0;
    let (handle, handle_reg) = if bound {
        (field(insn, 36, 13) as u16, None)
    } else {
        (0, Some(reg(insn, 39, 8)))
    };
    let coords = reg(insn, 8, 8);
    Some(if store {
        Op::Sust {
            src: reg(insn, 0, 8),
            coords,
            handle,
            handle_reg,
            dim,
            data,
        }
    } else {
        Op::Suld {
            dst: reg(insn, 0, 8),
            coords,
            handle,
            handle_reg,
            dim,
            data,
        }
    })
}

/// `d000_1`/`d200_1`-shared 4-bit field.
pub(super) fn texs_encoding(bits: u64, a: u8, b: u8) -> Option<(TexDim, [u8; 3], Option<u8>)> {
    let next = a.wrapping_add(1);
    let after_b = b.wrapping_add(1);
    Some(match bits {
        // 1D.LZ
        0 => (TexDim::T1d, [a, RZ, RZ], None),
        // 2D, 2D.LZ
        1 | 2 => (TexDim::T2d, [a, b, RZ], None),
        // 2D.LL: `b` is the level, not a coordinate.
        3 => (TexDim::T2d, [a, next, RZ], None),
        // 2D.DC, 2D.LZ.DC: the reference is `b`.
        4 | 6 => (TexDim::T2d, [a, next, RZ], Some(b)),
        // 2D.LL.DC.
        5 => (TexDim::T2d, [a, next, RZ], Some(after_b)),
        // ARRAY_2D, ARRAY_2D.LZ.
        7 | 8 => (TexDim::T2dArray, [next, b, a], None),
        // ARRAY_2D.LZ.DC
        9 => (TexDim::T2dArray, [next, b, a], Some(after_b)),
        // 3D, 3D.LZ
        10 | 11 => (TexDim::T3d, [a, next, b], None),
        // CUBE, CUBE.LL
        12 | 13 => (TexDim::TCube, [a, next, b], None),
        _ => return None,
    })
}

/// Which colour channels a `texs` writes.
pub(super) fn decode_tex_mask(selector: u64, dst: u8, dst2: u8) -> Option<[bool; 4]> {
    const ONE_DEST: [u8; 8] = [0x1, 0x2, 0x4, 0x8, 0x3, 0x9, 0xa, 0xc];
    const TWO_DEST: [u8; 8] = [0x7, 0xb, 0xd, 0xe, 0xf, 0x0, 0x0, 0x0];
    let row = match (dst != RZ, dst2 != RZ) {
        (false, false) => return None, // a sample with nowhere to land
        (true, true) => TWO_DEST,
        _ => ONE_DEST,
    };
    let bits = row[selector as usize & 7];
    if bits == 0 {
        return None;
    }
    Some([bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0])
}

pub(super) fn decode_tex(insn: u64, bindless: bool) -> Op {
    let un = Op::Unimplemented { raw: insn };
    let (aoffi_at, blod_at, lc_at) = if bindless { (36, 37, 40) } else { (54, 55, 58) };
    if field(insn, lc_at, 1) != 0 {
        return un;
    }
    // The dimensionalities `TexDim` names.
    let dim = match field(insn, 28, 3) {
        0 => TexDim::T1d,
        2 => TexDim::T2d,
        3 => TexDim::T2dArray,
        4 => TexDim::T3d,
        6 => TexDim::TCube,
        7 => TexDim::TCubeArray,
        _ => return un,
    };
    let dst = reg(insn, 0, 8);
    let bits = field(insn, 31, 4);
    if bits == 0 || dst == RZ {
        return un; // a sample with nowhere to land
    }
    let coord = reg(insn, 8, 8);
    let (layer, first) = match dim {
        TexDim::T2dArray | TexDim::TCubeArray => (Some(coord), coord.wrapping_add(1)),
        _ => (None, coord),
    };
    let mut meta = reg(insn, 20, 8);
    let mut take = || {
        let r = meta;
        meta = meta.wrapping_add(1);
        r
    };
    // The handle comes first, ahead of everything else the meta register chain carries.
    let handle_reg = bindless.then(&mut take);
    // `blod`.
    let lod = match field(insn, blod_at, 3) {
        0 | 1 => None,
        2 | 3 | 6 | 7 => Some(take()),
        _ => return un,
    };
    let offset = (field(insn, aoffi_at, 1) != 0).then(&mut take);
    let dref = (field(insn, 50, 1) != 0).then(&mut take);
    Op::Tex {
        dst,
        coords: [first, first.wrapping_add(1), first.wrapping_add(2)],
        layer,
        dref,
        offset,
        lod,
        handle: if bindless {
            0
        } else {
            field(insn, 36, 13) as u16
        },
        handle_reg,
        dim,
        mask: [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0],
    }
}

/// The four-bit channel mask a texture instruction keeps at `[31, 35)`.
fn texture_mask(insn: u64) -> [bool; 4] {
    let bits = field(insn, 31, 4);
    [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0]
}

/// `txq`.
pub(super) fn decode_txq(insn: u64) -> Op {
    const DIMENSION: u64 = 1;
    let dst = reg(insn, 0, 8);
    let mask = texture_mask(insn);
    if dst == RZ || field(insn, 22, 3) != DIMENSION || mask == [false; 4] {
        return Op::Unimplemented { raw: insn };
    }
    Op::Txq {
        dst,
        lod: reg(insn, 8, 8),
        handle: field(insn, 36, 13) as u16,
        mask,
    }
}

/// `tld4`, bound, of a 2D image or array.
pub(super) fn decode_tld4(insn: u64) -> Op {
    let un = Op::Unimplemented { raw: insn };
    // A cube's gather picks its face out of a direction first, which the gather here does not do.
    let dim = match field(insn, 28, 3) {
        2 => TexDim::T2d,
        3 => TexDim::T2dArray,
        _ => return un,
    };
    let dst = reg(insn, 0, 8);
    let mask = texture_mask(insn);
    if dst == RZ || mask == [false; 4] || field(insn, 50, 1) != 0 {
        return un;
    }
    let coord = reg(insn, 8, 8);
    let (layer, first) = match dim {
        TexDim::T2dArray | TexDim::TCubeArray => (Some(coord), coord.wrapping_add(1)),
        _ => (None, coord),
    };
    let offset = match field(insn, 54, 2) {
        0 => None,
        1 => Some(reg(insn, 20, 8)),
        _ => return un,
    };
    Op::Tld4 {
        dst,
        coords: [first, first.wrapping_add(1), first.wrapping_add(2)],
        layer,
        offset,
        handle: field(insn, 36, 13) as u16,
        dim,
        component: field(insn, 56, 2) as u8,
        mask,
    }
}

/// What one of a `texs`'s destination registers ends up holding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TexsStore {
    /// The whole register is one channel, as an `f32`.
    Float(usize),
    /// Two channels packed as halves, low first.
    Halves(usize, Option<usize>),
}

/// Where a `texs`'s enabled colour channels land, as `(channel, register)`.
pub fn texs_destinations(dst: u8, dst2: u8, mask: [bool; 4], f16: bool) -> Vec<(u8, TexsStore)> {
    let enabled: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter(|(_, &on)| on)
        .map(|(channel, _)| channel)
        .collect();
    if !f16 {
        return enabled
            .into_iter()
            .enumerate()
            .map(|(n, channel)| {
                let reg = if n < 2 {
                    dst.wrapping_add(n as u8)
                } else {
                    dst2.wrapping_add(n as u8 - 2)
                };
                (reg, TexsStore::Float(channel))
            })
            .collect();
    }
    enabled
        .chunks(2)
        .enumerate()
        .map(|(n, pair)| {
            let reg = if n == 0 { dst } else { dst2 };
            (reg, TexsStore::Halves(pair[0], pair.get(1).copied()))
        })
        .collect()
}
