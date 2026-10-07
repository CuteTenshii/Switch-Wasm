//! Which registers an op reads.

use super::*;

/// One register per set `mask` bit, consecutive from `dst`.
pub(super) fn consecutive_destinations(dst: u8, mask: [bool; 4]) -> Vec<(u8, isa::TexsStore)> {
    mask.iter()
        .enumerate()
        .filter(|(_, &wanted)| wanted)
        .zip(0u8..)
        .map(|((channel, _), n)| (dst.wrapping_add(n), isa::TexsStore::Float(channel)))
        .collect()
}

/// Where `reg`'s pending write lands: before its next reader, `None` if overwritten
/// first, or before the last instruction if never touched.
pub(super) fn first_use_after(ops: &[Op], start: usize, reg: u8) -> Option<usize> {
    for (idx, op) in ops.iter().enumerate().skip(start) {
        if reads(op).contains(&reg) {
            return Some(idx);
        }
        if writes(op).contains(&reg) {
            return None;
        }
    }
    ops.len().checked_sub(1)
}

fn operand_reg(op: Operand) -> Option<u8> {
    match op {
        Operand::Reg(r) if r != RZ => Some(r),
        _ => None,
    }
}

/// The destination, when a half op's merge mode reads it back.
fn half_merge_reads(dst: u8, merge: HMerge) -> Option<u8> {
    match merge {
        HMerge::MrgH0 | HMerge::MrgH1 if dst != RZ => Some(dst),
        _ => None,
    }
}

/// Registers `op` reads (never [`RZ`]).
pub(super) fn reads(op: &Op) -> Vec<u8> {
    let mut out: Vec<u8> = match *op {
        Op::St { src, size, idx, .. } => {
            let mut v: Vec<u8> = (0..size.regs()).map(|i| src.wrapping_add(i)).collect();
            v.push(idx);
            v
        }
        Op::Ld { idx, .. } => vec![idx],
        Op::Ipa { mul: Some(m), .. } => vec![m],
        Op::Mufu { src, .. } => vec![src],
        Op::Rro { src, .. } => operand_reg(src).into_iter().collect(),
        Op::Fadd { a, b, .. } | Op::Fmul { a, b, .. } | Op::Fmnmx { a, b, .. } => {
            let mut v = vec![a];
            v.extend(operand_reg(b));
            v
        }
        Op::Ffma { a, b, c, .. } => {
            let mut v = vec![a];
            v.extend(operand_reg(b));
            v.extend(operand_reg(c));
            v
        }
        // A merging half op also reads its destination.
        Op::Hfma2 {
            dst,
            a,
            b,
            c,
            merge,
            ..
        } => {
            let mut v = vec![a];
            v.extend(operand_reg(b));
            v.extend(operand_reg(c));
            v.extend(half_merge_reads(dst, merge));
            v
        }
        Op::Hadd2 {
            dst, a, b, merge, ..
        }
        | Op::Hmul2 {
            dst, a, b, merge, ..
        } => {
            let mut v = vec![a];
            v.extend(operand_reg(b));
            v.extend(half_merge_reads(dst, merge));
            v
        }
        Op::Hset2 { a, b, .. } | Op::Hsetp2 { a, b, .. } => {
            let mut v = vec![a];
            v.extend(operand_reg(b));
            v
        }
        Op::Iadd { a, b, .. }
        | Op::Imnmx { a, b, .. }
        | Op::Imul { a, b, .. }
        | Op::Lop { a, b, .. }
        | Op::Shl { a, b, .. }
        | Op::Shr { a, b, .. }
        | Op::Bfe { a, b, .. }
        | Op::Sel { a, b, .. }
        | Op::Iset { a, b, .. }
        | Op::Isetp { a, b, .. }
        | Op::Fset { a, b, .. }
        | Op::Fsetp { a, b, .. }
        | Op::Iscadd { a, b, .. } => {
            let mut v = vec![a];
            v.extend(operand_reg(b));
            v
        }
        Op::Iadd3 { a, b, c, .. } | Op::Xmad { a, b, c, .. } => {
            let mut v = vec![a];
            v.extend(operand_reg(b));
            v.extend(operand_reg(c));
            v
        }
        Op::Lop3 { a, b, c, .. } => {
            let mut v = vec![a];
            v.extend(operand_reg(b));
            v.extend(operand_reg(c));
            v
        }
        Op::Vmnmx { a, b, c, .. } => vec![a, b, c],
        Op::Icmp { a, b, c, .. } => {
            let mut v = vec![a, c];
            v.extend(operand_reg(b));
            v
        }
        Op::Shf { lo, shift, hi, .. } => {
            let mut v = vec![lo, hi];
            v.extend(operand_reg(shift));
            v
        }
        Op::Popc { b, .. } | Op::Flo { b, .. } => operand_reg(b).into_iter().collect(),
        Op::Mov { src, .. } => operand_reg(src).into_iter().collect(),
        Op::I2f { src, .. } | Op::F2i { src, .. } | Op::F2f { src, .. } | Op::I2i { src, .. } => {
            operand_reg(src).into_iter().collect()
        }
        Op::Ldc { idx, .. } => vec![idx],
        Op::Ldg { addr, .. } | Op::Ldl { addr, .. } => vec![addr, addr.wrapping_add(1)],
        Op::Stg {
            addr, src, size, ..
        }
        | Op::Stl {
            addr, src, size, ..
        } => {
            let mut v = vec![addr, addr.wrapping_add(1)];
            v.extend((0..size.regs()).map(|i| src.wrapping_add(i)));
            v
        }
        Op::Texs { coords, .. } => coords.to_vec(),
        // Every operand register, so earlier queued samples land first.
        Op::Tex {
            coords,
            layer,
            dref,
            offset,
            lod,
            handle_reg,
            dim,
            ..
        } => {
            let used = match dim {
                TexDim::T1d => 1,
                TexDim::T2d | TexDim::T2dArray => 2,
                TexDim::T3d | TexDim::TCube | TexDim::TCubeArray => 3,
            };
            let mut v = coords[..used].to_vec();
            v.extend([layer, dref, offset, lod, handle_reg].into_iter().flatten());
            v
        }
        Op::Txq { lod, .. } => vec![lod],
        Op::Tld4 {
            coords,
            layer,
            offset,
            ..
        } => {
            let mut v = coords[..2].to_vec();
            v.extend([layer, offset].into_iter().flatten());
            v
        }
        Op::Shfl {
            src, index, mask, ..
        } => {
            let mut v = vec![src];
            v.extend(operand_reg(index));
            v.extend(operand_reg(mask));
            v
        }
        Op::Fswzadd { a, b, .. } => vec![a, b],
        Op::Suld {
            coords,
            handle_reg,
            dim,
            ..
        } => {
            let mut v = surface_coord_regs(coords, dim);
            v.extend(handle_reg);
            v
        }
        Op::Sust {
            src,
            coords,
            handle_reg,
            dim,
            data,
            ..
        } => {
            let mut v = surface_coord_regs(coords, dim);
            v.extend((0..surface_source_words(data) as u8).map(|i| src.wrapping_add(i)));
            v.extend(handle_reg);
            v
        }
        _ => Vec::new(),
    };
    out.retain(|&r| r != RZ);
    out
}

fn surface_coord_regs(coords: u8, dim: SurfaceDim) -> Vec<u8> {
    let count = match dim {
        SurfaceDim::D1 | SurfaceDim::Buffer1d => 1,
        SurfaceDim::Array1d | SurfaceDim::D2 => 2,
        SurfaceDim::Array2d | SurfaceDim::D3 => 3,
    };
    (0..count).map(|i| coords.wrapping_add(i)).collect()
}

/// Source registers of a surface store: four for formatted, one per 32 bits for raw.
pub(super) fn surface_source_words(data: SurfaceData) -> usize {
    match data {
        SurfaceData::Formatted(_) => 4,
        SurfaceData::Raw(size) => size.words(),
    }
}
