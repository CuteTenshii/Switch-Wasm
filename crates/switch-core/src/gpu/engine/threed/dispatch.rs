//! Method writes: macros, semaphores, constant buffer loads, binds, and draws.

use super::{
    DrawCall, Engine3D, BIND, BIND_CONSTBUF_OFFSET, BIND_LAST, BIND_SLOTS, BIND_STRIDE,
    CLEAR_BUFFERS, COLOR_MASK, COLOR_MASK_ALL, COLOR_TARGETS, CONSTBUF_BANKS,
    CONSTBUF_SELECTOR_ADDR, CONSTBUF_SELECTOR_SIZE, DRAW_ARRAYS_COUNT, DRAW_ELEMENTS_COUNT,
    FIRMWARE_CALL, FIRMWARE_CALL_LAST, INLINE_FIRST, INLINE_LAST, LOAD_CONSTBUF_DATA,
    LOAD_CONSTBUF_DATA_LAST, LOAD_CONSTBUF_OFFSET, MME_FIRMWARE_ARGS, MME_INSTRUCTION_RAM_LOAD,
    MME_INSTRUCTION_RAM_POINTER, MME_START_ADDRESS_RAM_LOAD, MME_START_ADDRESS_RAM_POINTER,
    REPORT_SEMAPHORE, REPORT_SEMAPHORE_OFFSET, REPORT_SEMAPHORE_PAYLOAD, SYNCPT_ACTION,
    VERTEX_BEGIN_GL, VERTEX_BEGIN_INSTANCE_NEXT, VERTEX_END_GL,
};
use crate::gpu::engine::{field, Registers, REGISTER_COUNT};
use crate::gpu::exec::ExecCtx;
use crate::gpu::macro_engine::{MacroEngine, MacroHost, MacroWrite, MACRO_METHODS_START};
use crate::gpu::renderer::Software;
use crate::{Error, Result};

impl Engine3D {
    pub fn new() -> Engine3D {
        let mut regs = Registers::new();
        for target in 0..COLOR_TARGETS {
            regs.set(COLOR_MASK + target, COLOR_MASK_ALL);
        }
        Engine3D {
            regs,
            renderer: Box::new(Software),
            instance_id: 0,
            traced_regs: None,
            macros: MacroEngine::new(),
            inline: crate::gpu::engine::inline::EngineInline::new(),
            last_draw: DrawCall::default(),
            depth_fill: std::cell::RefCell::new(None),
            constbuf_cursor: 0,
            bound_constbufs: [[None; CONSTBUF_BANKS]; BIND_SLOTS],
            activity: Default::default(),
        }
    }

    /// Handle one method write; `last_call` ends a method group and runs a pending macro.
    pub fn write(
        &mut self,
        method: u32,
        arg: u32,
        last_call: bool,
        ctx: &mut ExecCtx,
    ) -> Result<()> {
        if method >= MACRO_METHODS_START {
            return self.write_macro(method, arg, last_call, ctx);
        }
        self.regs.set(method, arg);
        match method {
            MME_INSTRUCTION_RAM_POINTER => self.macros.instruction_ram_pointer = arg,
            MME_INSTRUCTION_RAM_LOAD => self.macros.push_instruction(arg),
            MME_START_ADDRESS_RAM_POINTER => self.macros.start_address_pointer = arg,
            MME_START_ADDRESS_RAM_LOAD => self.macros.push_start_address(arg),
            SYNCPT_ACTION => self.syncpt_action(arg, ctx)?,
            CLEAR_BUFFERS => self.clear_buffers(arg, ctx)?,
            REPORT_SEMAPHORE => self.report_semaphore(arg, ctx)?,
            FIRMWARE_CALL..=FIRMWARE_CALL_LAST => self.firmware_call(arg, ctx)?,
            LOAD_CONSTBUF_OFFSET => self.constbuf_cursor = field(arg, 0, 15),
            LOAD_CONSTBUF_DATA..=LOAD_CONSTBUF_DATA_LAST => self.load_constbuf(arg, ctx)?,
            BIND..=BIND_LAST => self.bind(method, arg),
            DRAW_ARRAYS_COUNT => self.draw_arrays(arg, ctx)?,
            DRAW_ELEMENTS_COUNT => self.draw_elements(arg, ctx)?,
            VERTEX_BEGIN_GL => {
                self.last_draw.primitive = field(arg, 0, 15);
                self.instance_id = if arg & VERTEX_BEGIN_INSTANCE_NEXT != 0 {
                    self.instance_id.wrapping_add(1)
                } else {
                    0
                };
            }
            VERTEX_END_GL => {}
            // Shared with KEPLER_INLINE_TO_MEMORY_B; macro writes arrive here, not via the channel.
            INLINE_FIRST..=INLINE_LAST => self.inline.write(method, arg, ctx)?,
            _ => {
                ctx.stats.inert_methods += 1;
                if ctx.trace {
                    crate::traceln!("[gpu] inert method={method:#x} arg={arg:#010x}");
                }
            }
        }
        Ok(())
    }

    fn write_macro(
        &mut self,
        method: u32,
        arg: u32,
        last_call: bool,
        ctx: &mut ExecCtx,
    ) -> Result<()> {
        let offset = method - MACRO_METHODS_START;
        let slot = offset >> 1;
        if offset & 1 == 0 {
            self.macros.start(slot, arg);
        } else {
            self.macros.push_argument(arg);
        }
        if last_call {
            // Taken out so macro writes can go through `self.write` and be read back mid-macro.
            let mut macros = std::mem::take(&mut self.macros);
            struct Host<'e, 'c, 'x> {
                engine: &'e mut Engine3D,
                ctx: &'c mut ExecCtx<'x>,
            }
            impl<'e, 'c, 'x> MacroHost for Host<'e, 'c, 'x> {
                fn read_method(&self, method: u32) -> u32 {
                    self.engine.regs.get(method)
                }
                fn write_method(&mut self, write: MacroWrite) -> Result<()> {
                    if self.ctx.trace {
                        crate::traceln!(
                            "[gpu] mme method={:#05x} arg={:#010x}",
                            write.method,
                            write.arg
                        );
                    }
                    self.engine.write(write.method, write.arg, true, self.ctx)
                }
            }
            let mut host = Host { engine: self, ctx };
            let result = macros.run(&mut host);
            self.macros = macros;
            ctx.stats.macros += 1;
            result?;
        }
        Ok(())
    }

    fn syncpt_action(&mut self, arg: u32, ctx: &mut ExecCtx) -> Result<()> {
        let id = field(arg, 0, 11);
        if field(arg, 20, 20) != 0 {
            ctx.host1x.increment(id)?;
        }
        Ok(())
    }

    /// `SetReportSemaphore`: the 3D class's fence release.
    fn report_semaphore(&mut self, arg: u32, ctx: &mut ExecCtx) -> Result<()> {
        const OPERATION_RELEASE: u32 = 0;
        const STRUCTURE_ONE_WORD: u32 = 1;
        let operation = field(arg, 0, 1);
        if operation != OPERATION_RELEASE {
            // Acquire/counter/trap: work is already retired.
            return Ok(());
        }
        let addr = self.regs.iova(REPORT_SEMAPHORE_OFFSET);
        let payload = self.regs.get(REPORT_SEMAPHORE_PAYLOAD);
        if field(arg, 28, 28) == STRUCTURE_ONE_WORD {
            ctx.write_u32(addr, payload)?;
        } else {
            ctx.write_u64(addr, payload as u64)?;
            ctx.write_u64(addr + 8, ctx.stats.submissions)?;
        }
        Ok(())
    }

    /// `FirmwareCall[n]`: PGRAPH is not modelled, so just report completion.
    fn firmware_call(&mut self, _arg: u32, _ctx: &mut ExecCtx) -> Result<()> {
        self.regs.set(MME_FIRMWARE_ARGS, 1);
        Ok(())
    }

    /// `LoadConstbufData`: stream data into the selected constant buffer.
    fn load_constbuf(&mut self, arg: u32, ctx: &mut ExecCtx) -> Result<()> {
        let addr = self.regs.iova(CONSTBUF_SELECTOR_ADDR);
        let size = self.regs.field(CONSTBUF_SELECTOR_SIZE, 0, 16);
        if self.constbuf_cursor + 4 > size {
            return Err(Error::Gpu(format!(
                "3d: constant-buffer upload at {:#x} exceeds its {:#x}-byte size",
                self.constbuf_cursor, size
            )));
        }
        ctx.write_u32(addr + self.constbuf_cursor as u64, arg)?;
        self.constbuf_cursor += 4;
        Ok(())
    }

    /// `Bind[slot]`: snapshot or forget the selected constant buffer for a bank.
    fn bind(&mut self, method: u32, arg: u32) {
        let offset = method - BIND;
        let slot = offset / BIND_STRIDE;
        if offset % BIND_STRIDE != BIND_CONSTBUF_OFFSET {
            return;
        }
        let valid = field(arg, 0, 0) != 0;
        let index = field(arg, 4, 8);
        let Some(entry) = self
            .bound_constbufs
            .get_mut(slot as usize)
            .and_then(|banks| banks.get_mut(index as usize))
        else {
            return;
        };
        *entry = valid.then(|| {
            let addr = self.regs.iova(CONSTBUF_SELECTOR_ADDR);
            let size = self.regs.field(CONSTBUF_SELECTOR_SIZE, 0, 16);
            (addr, size)
        });
    }

    fn draw_arrays(&mut self, count: u32, ctx: &mut ExecCtx) -> Result<()> {
        self.last_draw = DrawCall {
            primitive: self.regs.field(VERTEX_BEGIN_GL, 0, 15),
            first: self.regs.get(0x35D),
            count,
            indexed: false,
            index_format: 0,
        };
        ctx.stats.draws += 1;
        self.trace_reg_diff();
        self.rasterize_or_log(ctx);
        Ok(())
    }

    fn draw_elements(&mut self, count: u32, ctx: &mut ExecCtx) -> Result<()> {
        self.last_draw = DrawCall {
            primitive: self.regs.field(VERTEX_BEGIN_GL, 0, 15),
            first: self.regs.get(0x5F7),
            count,
            indexed: true,
            index_format: self.regs.get(0x5F6),
        };
        ctx.stats.draws += 1;
        self.trace_reg_diff();
        self.rasterize_or_log(ctx);
        Ok(())
    }

    /// `TRACE_REGS=1`: log which registers changed since the previous draw.
    fn trace_reg_diff(&mut self) {
        if !crate::trace::enabled(crate::trace::Trace::Regs) {
            return;
        }
        let now: Vec<u32> = (0..REGISTER_COUNT as u32)
            .map(|m| self.regs.get(m))
            .collect();
        if let Some(prev) = &self.traced_regs {
            let diff: Vec<String> = now
                .iter()
                .enumerate()
                .filter(|(i, v)| prev[*i] != **v)
                .map(|(i, v)| format!("{i:#x}={v:#010x}"))
                .collect();
            crate::traceln!(
                "[regs] begin={:#010x} {}",
                self.regs.get(VERTEX_BEGIN_GL),
                diff.join(" ")
            );
        }
        self.traced_regs = Some(now);
    }

    fn rasterize_or_log(&mut self, ctx: &mut ExecCtx) {
        if ctx.trace && ctx.stats.draws == 1 {
            self.dump_vertex_input();
        }
        let trace_draw = ctx.trace || crate::trace::enabled(crate::trace::Trace::Draw);
        if trace_draw {
            let rt = self.render_target(self.render_target_slot(0));
            let cull = self.cull_state();
            crate::traceln!(
                "[gpu] draw {} prim={:#x} first={} count={} indexed={} cull={} -> rt0 {}",
                ctx.stats.draws,
                self.last_draw.primitive,
                self.last_draw.first,
                self.last_draw.count,
                self.last_draw.indexed,
                if cull.enabled {
                    format!(
                        "{}{}{}",
                        if cull.front_ccw { "ccw" } else { "cw" },
                        if cull.cull_front { "-front" } else { "" },
                        if cull.cull_back { "-back" } else { "" },
                    )
                } else {
                    "off".to_owned()
                },
                match rt {
                    Ok(Some(rt)) => format!(
                        "{:#x} {}x{} fmt={:#x} cpu {}",
                        rt.addr,
                        rt.width,
                        rt.height,
                        rt.format.raw,
                        match ctx.span(rt.addr, 4) {
                            Some(cpu) => format!("{cpu:#x}"),
                            None => "unmapped".to_owned(),
                        }
                    ),
                    other => format!("{other:x?}"),
                }
            );
        }
        let target = match self.render_target(self.render_target_slot(0)) {
            Ok(Some(rt)) => Some((
                rt.addr,
                ctx.span(rt.addr, 4),
                rt.width,
                rt.height,
                rt.format.raw,
            )),
            _ => None,
        };
        let result = self.with_renderer(ctx, |renderer, engine, ctx| renderer.draw(engine, ctx));
        let vertices = u64::from(self.last_draw.count);
        match target {
            Some((addr, cpu, width, height, format)) => self.activity.note(
                crate::gpu::activity::Kind::Draw,
                addr,
                0,
                vertices,
                result.is_err(),
                || crate::gpu::activity::surface_text(addr, cpu, width, height, format),
            ),
            None => self.activity.note(
                crate::gpu::activity::Kind::Draw,
                0,
                0,
                vertices,
                result.is_err(),
                || "no colour target bound".to_owned(),
            ),
        }
        if let Err(e) = result {
            ctx.stats.draws_skipped += 1;
            self.activity
                .refuse(crate::gpu::activity::Kind::Draw, e.to_string());
            if trace_draw {
                let va = self.vertex_array(0);
                crate::traceln!(
                    "[gpu] raster: {e} [vtx0 start={:#x} stride={} count={}]",
                    va.start,
                    va.stride,
                    self.last_draw.count
                );
            }
        }
    }

    /// Trace the first draw's bound streams and attributes.
    fn dump_vertex_input(&self) {
        for i in 0..16 {
            let a = self.vertex_attrib(i);
            if a.size == 0 && !a.is_fixed {
                continue;
            }
            crate::traceln!(
                "[gpu] attrib{i} buf={} fixed={} off={:#x} size={:#x} ty={}",
                a.buffer_id,
                a.is_fixed,
                a.offset,
                a.size,
                a.ty
            );
        }
        for i in 0..8 {
            let v = self.vertex_array(i);
            crate::traceln!(
                "[gpu] stream{i} en={} stride={} start={:#x} limit={:#x}",
                v.enabled,
                v.stride,
                v.start,
                v.limit
            );
        }
    }
}
