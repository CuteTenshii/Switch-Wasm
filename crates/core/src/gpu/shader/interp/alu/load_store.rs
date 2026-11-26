//! Attribute, constant and memory accesses.

use super::*;

impl Invocation {
    #[inline(always)]
    pub(super) fn run_attr(&mut self, op: Op) {
        match op {
            Op::Ld {
                dst,
                offset,
                idx,
                size,
            } => {
                let base = offset.wrapping_add(self.attr_index(idx));
                for i in 0..size.regs() {
                    let v = self.attr_in.get(base + i as u16 * 4);
                    self.set_reg_f32(dst.wrapping_add(i), v);
                }
            }
            Op::St {
                offset,
                idx,
                src,
                size,
            } => {
                let base = offset.wrapping_add(self.attr_index(idx));
                for i in 0..size.regs() {
                    let v = self.reg_f32(src.wrapping_add(i));
                    self.attr_out.set(base + i as u16 * 4, v);
                }
            }
            // `centroid` is exact as-is: shading is per covered sample at its centre.
            Op::Ipa {
                dst,
                offset,
                mul,
                perspective,
                sat,
                centroid: _,
            } => {
                let mut v = self.attr_in.get(offset);
                if perspective {
                    if let Some(m) = mul {
                        v *= self.reg_f32(m);
                    }
                }
                if sat {
                    v = v.clamp(0.0, 1.0);
                }
                self.set_reg_f32(dst, v);
            }
            _ => unreachable!("routed by run_alu"),
        }
    }

    #[inline(always)]
    pub(super) fn run_memory(&mut self, op: Op, env: &Env) -> ShaderResult<()> {
        match op {
            Op::Ldc {
                dst,
                bank,
                offset,
                idx,
                size,
            } => {
                let base = offset.wrapping_add(self.reg(idx) as i32);
                for i in 0..size.regs() {
                    let at = base.wrapping_add(i as i32 * 4);
                    let v = env.consts.read_const(bank, at as u16)?;
                    self.set_reg(dst.wrapping_add(i), v);
                }
            }
            Op::Ldg {
                dst,
                addr,
                offset,
                size,
            } => {
                let mem = env
                    .memory
                    .ok_or_else(|| Error::Gpu("shader: ldg with no global memory bound".into()))?;
                let base = (self.reg64(addr) as i64).wrapping_add(offset as i64) as u64;
                if let Some(raw) = narrow_load(size, |i| mem.read_u8(base + i as u64))? {
                    self.set_reg(dst, raw);
                } else {
                    for i in 0..size.regs() {
                        let v = mem.read_u32(base.wrapping_add(u64::from(i) * 4))?;
                        self.set_reg(dst.wrapping_add(i), v);
                    }
                }
            }
            Op::Stg {
                addr,
                offset,
                src,
                size,
            } => {
                let mem = env
                    .memory
                    .ok_or_else(|| Error::Gpu("shader: stg with no global memory bound".into()))?;
                let base = (self.reg64(addr) as i64).wrapping_add(offset as i64) as u64;
                let (bytes, len) = self.store_value(src, size);
                if len < 4 {
                    for (i, byte) in bytes[..len].iter().enumerate() {
                        mem.write_u8(base + i as u64, *byte)?;
                    }
                } else {
                    for i in 0..size.regs() {
                        let v = self.reg(src.wrapping_add(i));
                        mem.write_u32(base.wrapping_add(u64::from(i) * 4), v)?;
                    }
                }
            }
            Op::Ldl {
                dst,
                addr,
                offset,
                size,
            } => {
                let base = (self.reg(addr) as i64).wrapping_add(offset as i64) as usize;
                let values = read_scratch(&self.local, base, size);
                for i in 0..size.regs() {
                    self.set_reg(dst.wrapping_add(i), values[i as usize]);
                }
            }
            Op::Stl {
                addr,
                offset,
                src,
                size,
            } => {
                let base = (self.reg(addr) as i64).wrapping_add(offset as i64) as usize;
                let (bytes, len) = self.store_value(src, size);
                let cap = self.local_bytes;
                write_scratch(&mut self.local, cap, base, &bytes[..len]);
            }
            Op::Lds {
                dst,
                addr,
                offset,
                size,
            } => {
                let shared = env
                    .shared
                    .ok_or_else(|| Error::Gpu("shader: lds with no shared memory bound".into()))?;
                let base = (self.reg(addr) as i64).wrapping_add(offset as i64) as usize;
                let values = read_scratch(&shared.borrow(), base, size);
                for i in 0..size.regs() {
                    self.set_reg(dst.wrapping_add(i), values[i as usize]);
                }
            }
            Op::Sts {
                addr,
                offset,
                src,
                size,
            } => {
                let shared = env
                    .shared
                    .ok_or_else(|| Error::Gpu("shader: sts with no shared memory bound".into()))?;
                let base = (self.reg(addr) as i64).wrapping_add(offset as i64) as usize;
                let (bytes, len) = self.store_value(src, size);
                let mut block = shared.borrow_mut();
                let cap = block.len();
                write_scratch(&mut block, cap, base, &bytes[..len]);
            }
            Op::Atom {
                dst,
                addr,
                offset,
                src,
                op,
                ty,
                space,
            } => {
                self.run_atom(dst, addr, offset, src, op, ty, space, env)?;
            }
            _ => unreachable!("routed by run_alu"),
        }
        Ok(())
    }
}
