//! Translation of everything that is not control flow or texturing.

use super::emitter::Emitter;
use super::layout::{atom_code, generic_slot};
use super::texture::{Fetch, Gather};
use super::Unsupported;
use crate::gpu::shader::isa::{
    AtomOp, AtomSpace, MemSize, Op, Operand, ShflMode, SurfaceData, SurfaceDim, TexDim, VoteMode,
    RZ,
};
use crate::gpu::shader::varying_slot;
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
                self.stores.extend(varying_slot(offset));
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
                self.loads.extend(varying_slot(offset));
                if centroid {
                    self.centroid_loads.extend(varying_slot(offset));
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

            Op::Fcmp { .. }
            | Op::Rro { .. }
            | Op::Fadd { .. }
            | Op::Fmul { .. }
            | Op::Ffma { .. }
            | Op::Fmnmx { .. }
            | Op::Mufu { .. }
            | Op::Hadd2 { .. }
            | Op::Hmul2 { .. }
            | Op::Hfma2 { .. }
            | Op::Hset2 { .. }
            | Op::Hsetp2 { .. }
            | Op::Fsetp { .. }
            | Op::Fset { .. } => self.emit_float(op)?,

            Op::Iadd { .. }
            | Op::Iadd3 { .. }
            | Op::Iscadd { .. }
            | Op::Lea { .. }
            | Op::Vmnmx { .. }
            | Op::Vmad { .. }
            | Op::Vsetp { .. }
            | Op::Imnmx { .. }
            | Op::Imul { .. }
            | Op::Xmad { .. }
            | Op::Isetp { .. }
            | Op::Iset { .. }
            | Op::Icmp { .. }
            | Op::Bfi { .. }
            | Op::R2p { .. }
            | Op::P2r { .. }
            | Op::Lop { .. }
            | Op::Lop3 { .. }
            | Op::Shl { .. }
            | Op::Shr { .. }
            | Op::Shf { .. }
            | Op::Bfe { .. }
            | Op::Popc { .. }
            | Op::Flo { .. }
            | Op::Sel { .. } => self.emit_integer(op)?,

            Op::I2f { .. }
            | Op::I2d { .. }
            | Op::D2f { .. }
            | Op::F2i { .. }
            | Op::F2f { .. }
            | Op::I2i { .. }
            | Op::Mov { .. }
            | Op::Mov32i { .. }
            | Op::S2r { .. }
            | Op::Psetp { .. }
            | Op::Pset { .. }
            | Op::Csetp { .. }
            | Op::Cset { .. } => self.emit_conversion(at, op)?,

            // ---- memory ----
            Op::Ldc {
                dst,
                bank,
                offset,
                idx,
                size,
            } => {
                self.note_bank(
                    bank,
                    (idx == RZ).then(|| offset as u32 + size.regs() as u32 * 4),
                );
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
            Op::Tlds {
                coords,
                layer,
                lod,
                offset,
                handle,
                dim,
                ..
            } => {
                let level = lod.filter(|&reg| reg != RZ);
                // Each level binds as a 2D texture of its own.
                if level.is_some() && dim != TexDim::T2d {
                    return Err(Unsupported::Op { at, op });
                }
                let fetch = Fetch {
                    slot: TextureSlot::Bound(handle),
                    dim,
                    coords,
                    layer,
                    offset,
                    level,
                };
                self.fetch_texture(at, fetch)?;
            }
            Op::Tld {
                coords,
                layer,
                lod,
                offset,
                handle,
                handle_reg,
                dim,
                sparse,
                ..
            } => {
                let level = lod.filter(|&reg| reg != RZ);
                // Each level binds as a 2D texture of its own, and `Fetch` offsets two axes.
                if (level.is_some() && dim != TexDim::T2d)
                    || (offset.is_some() && dim == TexDim::T3d)
                {
                    return Err(Unsupported::Op { at, op });
                }
                let slot = match handle_reg {
                    None => TextureSlot::Bound(handle),
                    Some(reg) => self
                        .bindless_slot(at, reg)
                        .ok_or(Unsupported::UntracedHandle { at })?,
                };
                let fetch = Fetch {
                    slot,
                    dim,
                    coords,
                    layer,
                    offset,
                    level,
                };
                self.fetch_texture(at, fetch)?;
                self.set_p(sparse, "false");
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
            Op::Tmml { dst } => self.set_r(dst, "0u"),
            Op::Txq {
                lod,
                handle,
                handle_reg,
                ..
            } => {
                let slot = match handle_reg {
                    None => TextureSlot::Bound(handle),
                    Some(reg) => self
                        .bindless_slot(at, reg)
                        .ok_or(Unsupported::UntracedHandle { at })?,
                };
                self.query_texture(at, slot, lod);
            }
            Op::Tld4 {
                coords,
                layer,
                offset,
                ptp,
                dref,
                handle,
                handle_reg,
                sparse,
                dim,
                component,
                ..
            } => {
                let slot = match handle_reg {
                    None => TextureSlot::Bound(handle),
                    Some(reg) => self
                        .bindless_slot(at, reg)
                        .ok_or(Unsupported::UntracedHandle { at })?,
                };
                self.set_p(sparse, "false");
                self.gather_texture(
                    at,
                    Gather {
                        slot,
                        dim,
                        coords,
                        layer,
                        offset,
                        ptp,
                        dref,
                        component,
                    },
                )?;
            }

            Op::Tld4s {
                coords,
                offset,
                dref,
                handle,
                component,
                ..
            } => self.gather_texture(
                at,
                Gather {
                    slot: TextureSlot::Bound(handle),
                    dim: TexDim::T2d,
                    coords: [coords[0], coords[1], coords[1]],
                    layer: None,
                    offset,
                    ptp: None,
                    dref,
                    component,
                },
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
                let value = self.r(src);
                let here = self.bind(&value);
                let lane = if self.warp_caps {
                    self.warp.get_or_insert(at);
                    self.bind("i32(sr_lane)")
                } else {
                    self.quad.get_or_insert(at);
                    self.quad_swap.get_or_insert(at);
                    self.bind("i32(quadLane())")
                };
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
                if self.warp_caps {
                    let peer = self.bind(&format!(
                        "subgroupShuffle({here}, u32(clamp({from}, 0, 31)))"
                    ));
                    self.set_r(dst, &format!("select({here}, {peer}, {ok})"));
                    self.set_p(pred, &ok);
                    return Ok(());
                }
                let x = self.bind(&format!("quadSwapX({here})"));
                let y = self.bind(&format!("quadSwapY({here})"));
                let d = self.bind(&format!("quadSwapDiagonal({here})"));
                let sel = self.bind(&format!("u32(({from} ^ {lane}) & 3)"));
                let peer = format!(
                    "select(select(select({here}, {x}, {sel} == 1u), {y}, {sel} == 2u), {d}, {sel} == 3u)"
                );
                // Out-of-quad lanes keep their own value.
                let reachable = format!("({ok} && {from} < 4)");
                self.set_r(dst, &format!("select({here}, {peer}, {reachable})"));
                self.set_p(pred, &ok);
            }

            // `fswzadd` needs only this lane's index, not another lane's value: in a warp, its place in its quad.
            Op::Fswzadd {
                dst,
                a,
                b,
                swizzle,
                ftz,
            } => {
                let lane = if self.warp_caps {
                    self.warp.get_or_insert(at);
                    "sr_lane"
                } else {
                    self.quad.get_or_insert(at);
                    "quadLane()"
                };
                let x = self.r(a);
                let x = self.flush(ftz, format!("bitcast<f32>({x})"));
                let x = self.bind(&x);
                let y = self.r(b);
                let y = self.flush(ftz, format!("bitcast<f32>({y})"));
                let y = self.bind(&y);
                let code = self.bind(&format!("((({swizzle}u) >> (({lane} & 3u) * 2u)) & 3u)"));
                // `FSWZ_SIGNS` in `super::interp`, arm for arm.
                let ka = self.bind(&format!(
                    "select(select(-1.0, 1.0, {code} == 1u), 0.0, {code} == 3u)"
                ));
                let kb = self.bind(&format!("select(-1.0, 1.0, {code} == 2u)"));
                self.set_f(dst, &format!("{ka} * {x} + {kb} * {y}"));
            }

            // The warp is the quad, as in the rasterizer: lane `n` is bit `n` of the ballot.
            Op::Vote {
                dst,
                pred,
                src,
                mode,
            } if self.warp_caps => {
                self.warp.get_or_insert(at);
                let holds = self.holds(src);
                // Lanes that exited are no longer in the subgroup, as they are out of the warp.
                let ballot = self.bind(&format!("subgroupBallot({holds}).x"));
                let active = self.bind("subgroupBallot(true).x");
                let verdict = match mode {
                    VoteMode::All => format!("{ballot} == {active}"),
                    VoteMode::Any => format!("{ballot} != 0u"),
                    VoteMode::Eq => format!("{ballot} == 0u || {ballot} == {active}"),
                };
                self.set_r(dst, &ballot);
                self.set_p(pred, &format!("({verdict})"));
            }
            Op::Vote {
                dst,
                pred,
                src,
                mode,
            } => {
                self.quad.get_or_insert(at);
                self.quad_swap.get_or_insert(at);
                let holds = self.holds(src);
                let here = self.bind(&format!("select(0u, 1u, {holds})"));
                let x = self.bind(&format!("quadSwapX({here})"));
                let y = self.bind(&format!("quadSwapY({here})"));
                let d = self.bind(&format!("quadSwapDiagonal({here})"));
                let lane = self.bind("(quadLane() & 3u)");
                let ballot = self.bind(&format!(
                    "(({here} << {lane}) | ({x} << ({lane} ^ 1u)) | ({y} << ({lane} ^ 2u)) | ({d} << ({lane} ^ 3u)))"
                ));
                let verdict = match mode {
                    VoteMode::All => format!("{ballot} == 15u"),
                    VoteMode::Any => format!("{ballot} != 0u"),
                    VoteMode::Eq => format!("{ballot} == 0u || {ballot} == 15u"),
                };
                self.set_r(dst, &ballot);
                self.set_p(pred, &format!("({verdict})"));
            }

            Op::Nop | Op::Inert => {}

            // Global memory is reachable only through a constant bank descriptor, which
            // binds a buffer; shared memory and barriers have no compute stage to run in.
            Op::Ldg {
                dst,
                addr,
                offset,
                size,
            } => {
                let Some((bank, descriptor)) = self.global_base(at, addr) else {
                    return Err(Unsupported::Op { at, op });
                };
                let slot = self.global_slot(bank, descriptor, false);
                let base = self.global_offset(bank, descriptor, addr, offset);
                match size {
                    MemSize::U8 | MemSize::S8 | MemSize::U16 | MemSize::S16 => {
                        let mut bytes = (0..size.bytes()).map(|byte| {
                            let place = self.bind(&format!("({base} + {byte}u)"));
                            let shift = self.bind(&format!("(({place} & 3u) * 8u)"));
                            format!("((gRead({slot}u, {place}) >> {shift}) & 0xffu)")
                        });
                        let low = bytes.next().expect("a byte");
                        let raw = match bytes.next() {
                            Some(high) => format!("({low} | ({high} << 8u))"),
                            None => low,
                        };
                        let raw = if matches!(size, MemSize::S8 | MemSize::S16) {
                            self.need("sext");
                            format!("sext({raw}, {}u)", size.bytes())
                        } else {
                            raw
                        };
                        self.set_r(dst, &raw);
                    }
                    _ => {
                        for word in 0..size.regs() {
                            let byte = u32::from(word) * 4;
                            self.set_r(
                                dst.wrapping_add(word),
                                &format!("gRead({slot}u, {base} + {byte}u)"),
                            );
                        }
                    }
                }
            }

            // Stores past the buffer's end change nothing, where the interpreter faults.
            Op::Stg {
                addr,
                offset,
                src,
                size,
            } => {
                let Some((bank, descriptor)) = self.global_base(at, addr) else {
                    return Err(Unsupported::Op { at, op });
                };
                let slot = self.global_slot(bank, descriptor, true);
                let base = self.global_offset(bank, descriptor, addr, offset);
                let words: Vec<String> = (0..size.regs())
                    .map(|i| self.r(src.wrapping_add(i)))
                    .collect();
                if size.bytes() < 4 {
                    for byte in 0..size.bytes() {
                        let place = self.bind(&format!("({base} + {byte}u)"));
                        let value = format!("(({}) >> {}u) & 0xffu", words[0], byte * 8);
                        self.line(&format!("gWrite({slot}u, {place}, {value}, 0xffu);"));
                    }
                } else {
                    for (word, value) in words.iter().enumerate() {
                        self.line(&format!(
                            "gWrite({slot}u, {base} + {}u, {value}, 0xffffffffu);",
                            word * 4
                        ));
                    }
                }
            }

            // A 32-bit atomic on a descriptor's buffer; the old value lands in `dst`.
            Op::Atom {
                dst,
                addr,
                offset,
                src,
                op: atom,
                ty,
                space: AtomSpace::Global,
            } => {
                let (Some(code), Some((bank, descriptor))) =
                    (atom_code(atom, ty), self.global_base(at, addr))
                else {
                    return Err(Unsupported::Op { at, op });
                };
                let slot = self.global_slot(bank, descriptor, true);
                let base = self.global_offset(bank, descriptor, addr, offset);
                let value = self.r(src);
                let stored = if atom == AtomOp::Cas {
                    self.r(src.wrapping_add(1))
                } else {
                    "0u".to_string()
                };
                let call = format!("gAtom({code}u, {slot}u, {base}, {value}, {stored})");
                if dst == RZ {
                    self.line(&format!("_ = {call};"));
                } else {
                    self.set_r(dst, &call);
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
            // A store outside the image is dropped, as the interpreter drops it.
            Op::Sust {
                src,
                coords,
                handle,
                handle_reg,
                dim: dim @ (SurfaceDim::D2 | SurfaceDim::Array2d),
                data: SurfaceData::Formatted([true, true, true, true]),
            } => {
                let slot = match handle_reg {
                    Some(reg) => self
                        .bindless_slot(at, reg)
                        .ok_or(Unsupported::UntracedHandle { at })?,
                    None => TextureSlot::Bound(handle),
                };
                self.bind_image(at, slot)?;
                let (x, y) = (self.r(coords), self.r(coords.wrapping_add(1)));
                let layer = match dim {
                    SurfaceDim::Array2d => {
                        format!("({} & 0xffffu)", self.r(coords.wrapping_add(2)))
                    }
                    _ => "0u".to_string(),
                };
                let value: Vec<String> = (0..4).map(|i| self.r(src.wrapping_add(i))).collect();
                self.line(&format!(
                    "imgStore({}u, {x}, {y}, {layer}, vec4<u32>({}));",
                    slot.key(),
                    value.join(", ")
                ));
            }
            // A reduction binds its image as words; one outside the image changes nothing.
            Op::Sured {
                src,
                coords,
                handle,
                handle_reg,
                dim: dim @ (SurfaceDim::D2 | SurfaceDim::Array2d),
                op: reduce,
                ty,
            } => {
                let code = atom_code(reduce, ty).ok_or(Unsupported::Op { at, op })?;
                let slot = match handle_reg {
                    Some(reg) => self
                        .bindless_slot(at, reg)
                        .ok_or(Unsupported::UntracedHandle { at })?,
                    None => TextureSlot::Bound(handle),
                };
                self.bind_image(at, slot)?;
                if !self.reduced_images.contains(&slot) {
                    self.reduced_images.push(slot);
                }
                let (x, y) = (self.r(coords), self.r(coords.wrapping_add(1)));
                let layer = match dim {
                    SurfaceDim::Array2d => {
                        format!("({} & 0xffffu)", self.r(coords.wrapping_add(2)))
                    }
                    _ => "0u".to_string(),
                };
                let value = self.r(src);
                self.line(&format!(
                    "imgReduce({code}u, {}u, {x}, {y}, {layer}, {value});",
                    slot.key()
                ));
            }
            Op::Lds { .. }
            | Op::Sts { .. }
            | Op::Atom { .. }
            | Op::Bar { .. }
            | Op::Suld { .. }
            | Op::Sust { .. }
            | Op::Sured { .. }
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
        }
        Ok(())
    }

    /// A global access's byte offset into its descriptor's buffer.
    ///
    /// The address register still holds descriptor plus offset, whatever happened to
    /// the register the offset was added from.
    fn global_offset(&mut self, bank: u8, descriptor: u16, addr: u8, offset: i32) -> String {
        let address = self.r(addr);
        let start = self.operand(Operand::Const {
            bank,
            offset: descriptor,
        });
        self.bind(&format!("({address} - {start} + {}u)", offset as u32))
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
