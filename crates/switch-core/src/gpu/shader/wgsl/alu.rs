//! Translation of everything that is not control flow or texturing.

use super::emitter::Emitter;
use super::layout::generic_slot;
use super::Unsupported;
use crate::gpu::shader::isa::{MemSize, Op, Operand, ShflMode, TexDim};
use crate::gpu::texture::TextureSlot;

impl Emitter<'_> {
    /// Everything that is not control flow.
    pub(super) fn emit_alu(&mut self, at: usize, op: Op) -> Result<(), Unsupported> {
        match op {
            // ---- attribute space ----
            Op::Ld {
                dst,
                offset,
                idx,
                size,
            } => {
                self.loads.extend(generic_slot(offset));
                let base = self.attr_base(offset, idx);
                for i in 0..size.regs() {
                    let word = i as u32 * 4;
                    self.set_f(dst.wrapping_add(i), &format!("attrIn({base} + {word}u)"));
                }
            }
            Op::St {
                offset,
                idx,
                src,
                size,
            } => {
                self.stores.extend(generic_slot(offset));
                let base = self.attr_base(offset, idx);
                for i in 0..size.regs() {
                    let word = i as u32 * 4;
                    let value = self.f(src.wrapping_add(i));
                    self.line(&format!("attrOut({base} + {word}u, {value});"));
                }
            }
            Op::Ipa {
                dst,
                offset,
                mul,
                perspective,
                sat,
                centroid,
            } => {
                self.loads.extend(generic_slot(offset));
                if centroid {
                    self.centroid_loads.extend(generic_slot(offset));
                }
                let mut value = format!("attrIn({offset}u)");
                if perspective {
                    if let Some(mul) = mul {
                        let factor = self.f(mul);
                        value = format!("({value} * {factor})");
                    }
                }
                let value = self.saturate(sat, value);
                self.set_f(dst, &value);
            }

            Op::Rro { .. } | Op::Fadd { .. } | Op::Fmul { .. } | Op::Ffma { .. } | Op::Fmnmx { .. } | Op::Mufu { .. } | Op::Hadd2 { .. } | Op::Hmul2 { .. } | Op::Hfma2 { .. } | Op::Hset2 { .. } | Op::Hsetp2 { .. } | Op::Fsetp { .. } | Op::Fset { .. } => self.emit_float(op)?,

            Op::Iadd { .. } | Op::Iadd3 { .. } | Op::Iscadd { .. } | Op::Vmnmx { .. } | Op::Imnmx { .. } | Op::Imul { .. } | Op::Xmad { .. } | Op::Isetp { .. } | Op::Iset { .. } | Op::Icmp { .. } | Op::Bfi { .. } | Op::R2p { .. } | Op::Lop { .. } | Op::Lop3 { .. } | Op::Shl { .. } | Op::Shr { .. } | Op::Shf { .. } | Op::Bfe { .. } | Op::Popc { .. } | Op::Flo { .. } | Op::Sel { .. } => self.emit_integer(op)?,

            Op::I2f { .. } | Op::F2i { .. } | Op::F2f { .. } | Op::I2i { .. } | Op::Mov { .. } | Op::Mov32i { .. } | Op::S2r { .. } | Op::Psetp { .. } | Op::Csetp { .. } => self.emit_conversion(at, op)?,

            // ---- memory ----
            Op::Ldc {
                dst,
                bank,
                offset,
                idx,
                size,
            } => {
                self.banks.insert(bank);
                let index = self.r(idx);
                let base = self.bind(&format!("{}u + {index}", offset as u32));
                for i in 0..size.regs() {
                    let word = i as u32 * 4;
                    self.set_r(
                        dst.wrapping_add(i),
                        &format!("cbRead({bank}u, ({base} + {word}u) & 0xffffu)"),
                    );
                }
            }

            // ---- texture ----
            // A `texs` array keeps its layer in the third coordinate register.
            Op::Texs {
                coords,
                dref,
                handle,
                dim,
                ..
            } => {
                let layer = (dim == TexDim::T2dArray).then_some(coords[2]);
                let slot = TextureSlot::Bound(handle);
                self.sample_texture(at, slot, dim, dref, coords, layer)?;
            }
            // `tex` keeps the layer before the coordinates; `.LL`/`.LB` sample the one level.
            Op::Tex {
                coords,
                layer,
                dref,
                offset,
                handle,
                handle_reg,
                dim,
                ..
            } => {
                let slot = match handle_reg {
                    None => TextureSlot::Bound(handle),
                    Some(reg) => self
                        .bindless_slot(at, reg)
                        .ok_or(Unsupported::UntracedHandle { at })?,
                };
                match offset {
                    None => self.sample_texture(at, slot, dim, dref, coords, layer)?,
                    Some(reg) => {
                        self.sample_offset(at, slot, dim, dref, coords, layer, reg, op)?;
                    }
                }
            }
            Op::Txq { lod, handle, .. } => {
                self.query_texture(at, TextureSlot::Bound(handle), lod);
            }
            // WGSL needs a constant gather offset.
            Op::Tld4 {
                coords,
                layer,
                offset: None,
                handle,
                dim,
                component,
                ..
            } => self.gather_texture(
                at,
                TextureSlot::Bound(handle),
                dim,
                coords,
                layer,
                component,
            )?,

            // `shfl` maps onto `quadSwapX`/`Y`/`Diagonal`, mirroring `interp::shuffle_source`.
            Op::Shfl {
                dst,
                pred,
                src,
                index,
                mask,
                mode,
            } => {
                self.quad.get_or_insert(at);
                self.quad_swap.get_or_insert(at);
                let value = self.r(src);
                let here = self.bind(&value);
                let x = self.bind(&format!("quadSwapX({here})"));
                let y = self.bind(&format!("quadSwapY({here})"));
                let d = self.bind(&format!("quadSwapDiagonal({here})"));
                let lane = self.bind("i32(quadLane())");
                let index = self.operand(index);
                let index = self.bind(&format!("i32({index})"));
                let mask = self.operand(mask);
                let mask = self.bind(&format!("i32({mask})"));
                let clamp = self.bind(&format!("({mask} & 31)"));
                let segment = self.bind(&format!("(({mask} >> 8) & 31)"));
                let floor = self.bind(&format!("({lane} & {segment})"));
                let ceiling = self.bind(&format!("({floor} | ({clamp} & ~{segment}))"));
                let from = match mode {
                    ShflMode::Idx => format!("(({index} & ~{segment}) | {floor})"),
                    ShflMode::Up => format!("({lane} - {index})"),
                    ShflMode::Down => format!("({lane} + {index})"),
                    ShflMode::Bfly => format!("({lane} ^ {index})"),
                };
                let from = self.bind(&from);
                // `up` is the one mode whose bound holds from below.
                let within = match mode {
                    ShflMode::Up => format!("({from} >= {ceiling})"),
                    _ => format!("({from} <= {ceiling})"),
                };
                let ok = self.bind(&format!("({within} && {from} >= 0)"));
                let sel = self.bind(&format!("u32(({from} ^ {lane}) & 3)"));
                let peer = format!(
                    "select(select(select({here}, {x}, {sel} == 1u), {y}, {sel} == 2u), {d}, {sel} == 3u)"
                );
                // Out-of-quad lanes keep their own value.
                let reachable = format!("({ok} && {from} < 4)");
                self.set_r(dst, &format!("select({here}, {peer}, {reachable})"));
                self.set_p(pred, &ok);
            }

            // `fswzadd` needs only this lane's index, not another lane's value.
            Op::Fswzadd {
                dst,
                a,
                b,
                swizzle,
                ftz,
            } => {
                self.quad.get_or_insert(at);
                let x = self.r(a);
                let x = self.flush(ftz, format!("bitcast<f32>({x})"));
                let x = self.bind(&x);
                let y = self.r(b);
                let y = self.flush(ftz, format!("bitcast<f32>({y})"));
                let y = self.bind(&y);
                let code = self.bind(&format!(
                    "((({swizzle}u) >> ((quadLane() & 3u) * 2u)) & 3u)"
                ));
                // `FSWZ_SIGNS` in `super::interp`, arm for arm.
                let ka = self.bind(&format!(
                    "select(select(-1.0, 1.0, {code} == 1u), 0.0, {code} == 3u)"
                ));
                let kb = self.bind(&format!("select(-1.0, 1.0, {code} == 2u)"));
                self.set_f(dst, &format!("{ka} * {x} + {kb} * {y}"));
            }

            Op::Nop | Op::Inert => {}

            // `ldg` from a constant bank descriptor binds a buffer; other global
            // and shared memory, and barriers, are unsupported.
            Op::Ldg {
                dst,
                addr,
                offset,
                size,
            } => {
                let Some((bank, at, index)) = self.global_base(at, addr) else {
                    return Err(Unsupported::Op { at, op });
                };
                let slot = match self.globals.iter().position(|g| *g == (bank, at)) {
                    Some(slot) => slot,
                    None => {
                        self.globals.push((bank, at));
                        self.globals.len() - 1
                    }
                };
                let index = self.r(index);
                let base = self.bind(&format!("({index} + {offset}u)"));
                for word in 0..size.regs() {
                    let byte = u32::from(word) * 4;
                    self.set_r(
                        dst.wrapping_add(word),
                        &format!("gRead({slot}u, {base} + {byte}u)"),
                    );
                }
            }

            Op::Ldl {
                dst,
                addr,
                offset,
                size,
            } => {
                self.helpers.insert("local");
                let base = self.local_address(addr, offset);
                let value = |word: u32| format!("localWord({base} + {}u)", word * 4);
                match size {
                    MemSize::U8 => self.set_r(dst, &format!("localByte({base})")),
                    MemSize::S8 => self.set_r(
                        dst,
                        &format!(
                            "bitcast<u32>(extractBits(bitcast<i32>(localByte({base})), 0u, 8u))"
                        ),
                    ),
                    MemSize::U16 | MemSize::S16 => {
                        let half = format!("(localByte({base}) | (localByte({base} + 1u) << 8u))");
                        let half = if size == MemSize::S16 {
                            format!("bitcast<u32>(extractBits(bitcast<i32>({half}), 0u, 16u))")
                        } else {
                            half
                        };
                        self.set_r(dst, &half);
                    }
                    _ => {
                        for word in 0..u32::from(size.regs()) {
                            self.set_r(dst.wrapping_add(word as u8), &value(word));
                        }
                    }
                }
            }
            Op::Stl {
                addr,
                offset,
                src,
                size,
            } => {
                self.helpers.insert("local");
                let base = self.local_address(addr, offset);
                let len = size.bytes();
                let words: Vec<String> = (0..size.regs())
                    .map(|i| self.r(src.wrapping_add(i)))
                    .collect();
                // Out of range drops the whole store, as the interpreter does.
                self.line(&format!("if ({base} + {len}u <= 1024u) {{"));
                self.indent += 1;
                for byte in 0..len {
                    let word = &words[(byte / 4) as usize];
                    let shift = (byte % 4) * 8;
                    self.line(&format!(
                        "setLocalByte({base} + {byte}u, {word} >> {shift}u);"
                    ));
                }
                self.indent -= 1;
                self.line("}");
            }
            Op::Stg { .. }
            | Op::Lds { .. }
            | Op::Sts { .. }
            | Op::Atom { .. }
            | Op::Bar { .. }
            // A ballot needs the warp, which is only a quad here.
            | Op::Vote { .. }
            | Op::Suld { .. }
            | Op::Sust { .. }
            | Op::Unimplemented { .. } => return Err(Unsupported::Op { at, op }),

            // Handled by `emit_terminator` and `emit_instruction`.
            Op::Bra { .. }
            | Op::Brx { .. }
            | Op::Ssy { .. }
            | Op::Pbk { .. }
            | Op::Pcnt { .. }
            | Op::Sync
            | Op::Brk
            | Op::Cont
            | Op::Exit
            | Op::Kil => unreachable!("control flow is emitted by emit_instruction"),
            // `tex.aoffi` with a non-constant offset is the rasterizer's.
            Op::Tld4 { .. } => return Err(Unsupported::Op { at, op }),
        }
        Ok(())
    }

    /// `a[offset + Rn]`'s byte address, wrapping at 16 bits.
    fn attr_base(&mut self, offset: u16, idx: u8) -> String {
        let index = self.r(idx);
        self.bind(&format!("({offset}u + ({index} & 0xffffu)) & 0xffffu"))
    }

    /// One 16-bit half of a register, as `xmad` reads it.
    pub(super) fn half(&mut self, value: &str, high: bool, signed: bool) -> String {
        let half = if high {
            format!("(({value}) >> 16u)")
        } else {
            format!("(({value}) & 0xffffu)")
        };
        if signed {
            self.need("sext");
            format!("sext({half}, 2u)")
        } else {
            half
        }
    }

    /// A shift instruction's count, masked when the encoding says to wrap.
    pub(super) fn shift_count(&mut self, operand: Operand, wrap: bool) -> String {
        let count = self.operand(operand);
        if wrap {
            format!("(({count}) & 31u)")
        } else {
            count
        }
    }

    /// A conversion's source byte lane, narrowed and extended back to 32 bits.
    pub(super) fn narrow(&mut self, raw: &str, sel: u8, bytes: u8, signed: bool) -> String {
        let shift = u32::from(sel) * 8;
        let shifted = if shift == 0 {
            raw.to_string()
        } else {
            format!("(({raw}) >> {shift}u)")
        };
        if signed {
            self.need("sext");
            format!("sext({shifted}, {bytes}u)")
        } else {
            self.need("truncw");
            format!("truncw({shifted}, {bytes}u)")
        }
    }
}
