//! Basic blocks, branches and the final function text.

use super::emitter::Emitter;
use super::helpers::HELPERS;
use super::{is_terminator, Unsupported, RECONVERGENCE_DEPTH};
use crate::gpu::shader::compiled::NO_TARGET;
use crate::gpu::shader::isa::{Op, Pred};

impl Emitter<'_> {
    /// One `case` per basic block, in order.
    pub(super) fn emit_blocks(&mut self, leaders: &[usize]) -> Result<(), Unsupported> {
        for (n, &start) in leaders.iter().enumerate() {
            let end = leaders.get(n + 1).copied().unwrap_or(self.program.len());
            self.block = start;
            self.indent = 3;
            self.line(&format!("case {start}u: {{"));
            self.indent = 4;
            for at in start..end {
                self.emit_instruction(at, end)?;
            }
            if !is_terminator(self.program.op(end - 1)) {
                self.line(&format!("pc = {end}u;"));
            }
            self.indent = 3;
            self.line("}");
        }
        Ok(())
    }

    fn emit_instruction(&mut self, at: usize, fallthrough: usize) -> Result<(), Unsupported> {
        let op = self.program.op(at);
        let guard = self.program.pred(at);
        // A push falls through; its `PT` guard bits hold the target.
        if matches!(op, Op::Ssy { .. } | Op::Pbk { .. } | Op::Pcnt { .. }) {
            let target = self.program.target(at);
            if target == NO_TARGET {
                return Err(Unsupported::UndecodedTarget { at });
            }
            self.uses_stack = true;
            self.line(&format!("stack[sp] = {target}u;"));
            self.line("sp = sp + 1;");
            return Ok(());
        }
        if is_terminator(op) {
            return self.emit_terminator(at, op, guard, fallthrough);
        }
        if guard.is_always() {
            return self.emit_alu(at, op);
        }
        let cond = self.holds(guard);
        self.line(&format!("if ({cond}) {{"));
        self.indent += 1;
        let result = self.emit_alu(at, op);
        self.indent -= 1;
        self.line("}");
        result
    }

    /// A guarded terminator must say where control goes when the guard fails.
    fn emit_terminator(
        &mut self,
        at: usize,
        op: Op,
        guard: Pred,
        fallthrough: usize,
    ) -> Result<(), Unsupported> {
        if guard.is_always() {
            return self.emit_jump(at, op);
        }
        let cond = self.holds(guard);
        self.line(&format!("if ({cond}) {{"));
        self.indent += 1;
        let result = self.emit_jump(at, op);
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line(&format!("pc = {fallthrough}u;"));
        self.indent -= 1;
        self.line("}");
        result
    }

    fn emit_jump(&mut self, at: usize, op: Op) -> Result<(), Unsupported> {
        match op {
            Op::Bra { .. } => {
                let target = self.program.target(at);
                if target == NO_TARGET {
                    return Err(Unsupported::UndecodedTarget { at });
                }
                self.line(&format!("pc = {target}u;"));
            }
            Op::Exit => self.line("return false;"),
            Op::Kil => self.line("return true;"),
            Op::Sync | Op::Brk | Op::Cont => {
                self.uses_stack = true;
                self.line("sp = sp - 1;");
                self.line("pc = stack[sp];");
            }
            Op::Brx { base, reg } => {
                let targets: Vec<u32> = match self.program.indirect_targets(at) {
                    Some(targets) => targets.to_vec(),
                    None => return Err(Unsupported::IndirectBranch { at }),
                };
                let selector = self.r(reg);
                let raw = self.bind(&format!("{base}u + {selector}"));
                // A target on a `sched` word means the block's first instruction.
                let slot = self.bind(&format!("select({raw}, {raw} + 8u, ({raw} & 31u) == 0u)"));
                self.line(&format!("switch ({slot}) {{"));
                self.indent += 1;
                for target in targets {
                    let offset = self.program.offset(target as usize);
                    self.line(&format!("case {offset}u: {{ pc = {target}u; }}"));
                }
                // No way to raise the interpreter's error, so end the invocation.
                self.line("default: { return false; }");
                self.indent -= 1;
                self.line("}");
            }
            _ => unreachable!("emit_jump called with {op:?}"),
        }
        Ok(())
    }

    /// A local-memory byte address; `0xffffffffu` marks offsets past 2^31.
    pub(super) fn local_address(&mut self, addr: u8, offset: i32) -> String {
        let reg = self.r(addr);
        self.bind(&format!(
            "select(0xffffffffu, bitcast<u32>(bitcast<i32>({reg}) + ({offset})), {reg} < 0x80000000u)"
        ))
    }

    pub(super) fn finish(&self, leaders: &[usize]) -> String {
        let mut out = format!(
            "// {} instructions in {} blocks\n\n",
            self.program.len(),
            leaders.len()
        );
        for (name, source) in HELPERS {
            if self.helpers.contains(name) {
                out.push_str(source);
                out.push_str("\n\n");
            }
        }
        // Only registers are visible to the caller.
        for reg in &self.regs {
            out.push_str(&format!("var<private> r{reg}: u32 = 0u;\n"));
        }
        if !self.regs.is_empty() {
            out.push('\n');
        }
        out.push_str("fn run() -> bool {\n");
        for pred in &self.preds {
            out.push_str(&format!("  var p{pred}: bool = false;\n"));
        }
        if self.uses_carry {
            out.push_str("  var carry: bool = false;\n");
        }
        if self.uses_flags {
            out.push_str(
                "  var ccZ: bool = false;\n  var ccS: bool = false;\n  var ccO: bool = false;\n",
            );
        }
        if self.uses_stack {
            out.push_str(&format!(
                "  var stack: array<u32, {RECONVERGENCE_DEPTH}>;\n"
            ));
            out.push_str("  var sp: i32 = 0;\n");
        }
        out.push_str("  var pc: u32 = 0u;\n");
        out.push_str("  loop {\n");
        out.push_str("    switch (pc) {\n");
        out.push_str(&self.body);
        // Required by WGSL; the decoder rejects programs with no `exit`.
        out.push_str("      default: { return false; }\n");
        out.push_str("    }\n");
        out.push_str("  }\n");
        // Unreachable, but WGSL requires a final return.
        out.push_str("  return false;\n");
        out.push_str("}\n");
        out
    }
}
