//! The Macro Method Expander (MME) — the small processor in front of the
//! Maxwell 3D class.
//!
//! Methods `0xE00` and up are not registers: they are 128 macro slots. The
//! driver uploads a program into the MME instruction RAM
//! (`LoadMmeInstructionRam`) and binds slot entry points
//! (`LoadMmeStartAddressRam`). Writing to slot `n`'s even method starts macro
//! `n` with that value as the first argument; the odd method pushes further
//! arguments. The macro runs when the pushbuffer's method group ends, and it
//! emits ordinary method writes back into the class.
//!
//! deko3d compiles its draw calls into macros, so nothing draws without this.
//!
//! The ISA is 32 bits per instruction with 8 GPRs (R0 reads as zero):
//!
//! ```text
//!  bits 0..2    operation
//!  bits 4..6    assignment (what to do with the result)
//!  bits 8..10   destination register
//!  bits 11..13  source register A
//!  bits 14..16  source register B
//!  bits 17..21  ALU sub-operation / bitfield source bit
//!  bits 22..26  bitfield size
//!  bits 27..31  bitfield destination bit
//!  bits 14..31  sign-extended immediate (18 bits)
//!  bit 7        exit after the next instruction
//! ```
//!
//! A branch (operation 7) and the exit modifier both have a one-instruction
//! delay slot: they take effect only after the following instruction has run.
//! Branch bit 4 selects != 0 vs == 0 and bit 5 annuls the delay slot when the
//! branch is taken; combining the exit modifier with a branch exits only when
//! the branch is *not* taken.

use crate::{Error, Result};

/// Size of the MME instruction RAM in 32-bit words (Maxwell).
pub const INSTRUCTION_RAM_SIZE: usize = 0x1000;
/// Number of macro entry-point slots.
pub const START_ADDRESS_RAM_SIZE: usize = 0x100;
/// First method address that selects a macro instead of a class register.
pub const MACRO_METHODS_START: u32 = 0xE00;
/// Instruction budget per macro, so a corrupt program cannot hang the core.
const MAX_STEPS: u32 = 100_000;

/// What the macro program wants done with an instruction's result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Assignment {
    IgnoreAndFetch,
    Move,
    MoveAndSetMethod,
    FetchAndSend,
    MoveAndSend,
    FetchAndSetMethod,
    MoveAndSetMethodThenFetchAndSend,
    MoveAndSetMethodThenSendHigh,
}

impl Assignment {
    fn from_bits(bits: u32) -> Assignment {
        match bits & 7 {
            0 => Assignment::IgnoreAndFetch,
            1 => Assignment::Move,
            2 => Assignment::MoveAndSetMethod,
            3 => Assignment::FetchAndSend,
            4 => Assignment::MoveAndSend,
            5 => Assignment::FetchAndSetMethod,
            6 => Assignment::MoveAndSetMethodThenFetchAndSend,
            _ => Assignment::MoveAndSetMethodThenSendHigh,
        }
    }
}

/// A method write the macro emitted, for the caller to feed back into the
/// class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MacroWrite {
    pub method: u32,
    pub arg: u32,
}

/// The class-side interface a running macro needs: read a class register and
/// apply an emitted method write. Both are called in program order as the
/// macro runs, so a `read_method` sees the effect of every write the macro has
/// already emitted (and its side effects, such as a firmware call completing).
pub trait MacroHost {
    fn read_method(&self, method: u32) -> u32;
    fn write_method(&mut self, write: MacroWrite) -> Result<()>;
}

/// The MME's storage: instruction RAM plus the per-slot entry points. Lives
/// for the life of the channel, like the hardware's.
#[derive(Debug)]
pub struct MacroEngine {
    instruction_ram: Vec<u32>,
    start_addresses: Vec<u32>,
    /// Write cursor for `LoadMmeInstructionRam`.
    pub instruction_ram_pointer: u32,
    /// Write cursor for `LoadMmeStartAddressRam`.
    pub start_address_pointer: u32,
    /// Arguments pushed since the current macro was started.
    args: Vec<u32>,
    /// Macro slot the pending arguments belong to; `None` when idle.
    pending: Option<u32>,
}

impl Default for MacroEngine {
    fn default() -> Self {
        MacroEngine::new()
    }
}

impl MacroEngine {
    pub fn new() -> MacroEngine {
        MacroEngine {
            instruction_ram: vec![0; INSTRUCTION_RAM_SIZE],
            start_addresses: vec![0; START_ADDRESS_RAM_SIZE],
            instruction_ram_pointer: 0,
            start_address_pointer: 0,
            args: Vec::new(),
            pending: None,
        }
    }

    /// `LoadMmeInstructionRam`: append one word at the current pointer.
    pub fn push_instruction(&mut self, word: u32) {
        let idx = self.instruction_ram_pointer as usize;
        if idx < self.instruction_ram.len() {
            self.instruction_ram[idx] = word;
        }
        self.instruction_ram_pointer = self.instruction_ram_pointer.wrapping_add(1);
    }

    /// `LoadMmeStartAddressRam`: bind the next macro slot's entry point.
    pub fn push_start_address(&mut self, address: u32) {
        let idx = self.start_address_pointer as usize;
        if idx < self.start_addresses.len() {
            self.start_addresses[idx] = address;
        }
        self.start_address_pointer = self.start_address_pointer.wrapping_add(1);
    }

    /// Begin collecting arguments for macro `slot` (a write to its even
    /// method). Any half-collected macro is discarded, as the hardware does.
    pub fn start(&mut self, slot: u32, first_arg: u32) {
        self.args.clear();
        self.args.push(first_arg);
        self.pending = Some(slot);
    }

    /// A write to a macro's odd method: one more argument.
    pub fn push_argument(&mut self, arg: u32) {
        self.args.push(arg);
    }

    pub fn pending_slot(&self) -> Option<u32> {
        self.pending
    }

    /// Run the pending macro against `host`, which supplies the class register
    /// file for `read` and receives each emitted method write as it happens —
    /// so a later `read` sees the effect of earlier writes (e.g.
    /// `WriteHardwareReg`'s firmware-call completion poll).
    pub fn run<H: MacroHost + ?Sized>(&mut self, host: &mut H) -> Result<()> {
        let slot = match self.pending.take() {
            Some(slot) => slot,
            None => return Ok(()),
        };
        let entry = *self
            .start_addresses
            .get(slot as usize)
            .ok_or_else(|| Error::Gpu(format!("mme: macro slot {} out of range", slot)))?;
        let args = std::mem::take(&mut self.args);
        let mut state = MacroState {
            gprs: [0; 8],
            pc: entry,
            method_address: 0,
            method_increment: 0,
            args: &args,
            arg_index: 0,
            carry: false,
        };
        // The first argument arrives in R1 rather than through a fetch.
        state.gprs[1] = state.fetch_arg();
        let mut steps = 0u32;
        // A branch and the exit modifier both have a one-instruction delay slot:
        // they take effect only after the following instruction has run. A taken
        // branch overrides a concurrent exit, so an exit in a branch's delay
        // slot only fires when control falls through instead of jumping.
        let mut pending_jump: Option<u32> = None;
        let mut pending_exit = false;
        let mut trace: Vec<(u32, u32)> = Vec::new();
        loop {
            if steps >= MAX_STEPS {
                let mut detail = format!(
                    "mme: macro {} ran for {} instructions without exiting (entry {:#x})\n",
                    slot, MAX_STEPS, entry
                );
                for (pc, word) in trace.iter().rev().take(40).rev() {
                    detail.push_str(&format!(
                        "  {:#06x}: {:#010x}  {}\n",
                        pc,
                        word,
                        disasm(*word)
                    ));
                }
                return Err(Error::Gpu(detail));
            }
            steps += 1;
            let word = *self
                .instruction_ram
                .get(state.pc as usize)
                .ok_or_else(|| Error::Gpu(format!("mme: pc {:#x} out of range", state.pc)))?;
            if trace.len() < 2000 {
                trace.push((state.pc, word));
            }
            state.pc = state.pc.wrapping_add(1);
            let flow = state.execute(word, host)?;

            // The delay slot has just run; resolve the previous instruction's
            // branch/exit. A jump discards an exit scheduled by the delay slot.
            let was_delay_slot = pending_jump.is_some();
            if let Some(target) = pending_jump.take() {
                state.pc = target;
            } else if pending_exit {
                break;
            }

            // Schedule this instruction's control flow for the next slot.
            if flow.is_branch && flow.taken {
                if flow.annul {
                    state.pc = flow.target;
                } else {
                    pending_jump = Some(flow.target);
                }
            }
            if flow.exit && !(flow.is_branch && flow.taken) && !was_delay_slot {
                pending_exit = true;
            }
        }
        Ok(())
    }
}

/// The flow-control intent of a single executed instruction, resolved by the
/// caller once the instruction's delay slot has run.
struct Flow {
    is_branch: bool,
    taken: bool,
    target: u32,
    annul: bool,
    exit: bool,
}

struct MacroState<'a> {
    gprs: [u32; 8],
    pc: u32,
    method_address: u32,
    method_increment: u32,
    args: &'a [u32],
    arg_index: usize,
    carry: bool,
}

impl MacroState<'_> {
    fn fetch_arg(&mut self) -> u32 {
        let v = self.args.get(self.arg_index).copied().unwrap_or(0);
        self.arg_index += 1;
        v
    }

    fn gpr(&self, index: u32) -> u32 {
        if index == 0 {
            0
        } else {
            self.gprs[(index & 7) as usize]
        }
    }

    fn set_gpr(&mut self, index: u32, value: u32) {
        if index != 0 {
            self.gprs[(index & 7) as usize] = value;
        }
    }

    fn set_method(&mut self, value: u32) {
        self.method_address = value & 0xFFF;
        self.method_increment = (value >> 12) & 0x3F;
    }

    fn send<H: MacroHost + ?Sized>(&mut self, value: u32, host: &mut H) -> Result<()> {
        host.write_method(MacroWrite {
            method: self.method_address,
            arg: value,
        })?;
        self.method_address = self.method_address.wrapping_add(self.method_increment) & 0xFFF;
        Ok(())
    }

    /// Execute one instruction, returning its flow-control intent. The caller
    /// applies branches and the exit modifier after the delay slot has run.
    fn execute<H: MacroHost + ?Sized>(&mut self, word: u32, host: &mut H) -> Result<Flow> {
        let exit = (word >> 7) & 1 != 0;
        if word & 7 == 7 {
            // Branch: bit 4 selects != 0 vs == 0, bit 5 annuls the delay slot.
            let on_not_zero = (word >> 4) & 1 != 0;
            let annul = (word >> 5) & 1 != 0;
            let value = self.gpr((word >> 11) & 7);
            let taken = if on_not_zero { value != 0 } else { value == 0 };
            // The immediate is relative to the branch instruction, and the
            // pc has already advanced past it.
            let target = self.pc.wrapping_sub(1).wrapping_add(imm(word) as u32);
            return Ok(Flow {
                is_branch: true,
                taken,
                target,
                annul,
                exit,
            });
        }
        let result = self.alu(word, host)?;
        let dst = (word >> 8) & 7;
        match Assignment::from_bits(word >> 4) {
            Assignment::IgnoreAndFetch => {
                let v = self.fetch_arg();
                self.set_gpr(dst, v);
            }
            Assignment::Move => self.set_gpr(dst, result),
            Assignment::MoveAndSetMethod => {
                self.set_gpr(dst, result);
                self.set_method(result);
            }
            Assignment::FetchAndSend => {
                let v = self.fetch_arg();
                self.set_gpr(dst, v);
                self.send(result, host)?;
            }
            Assignment::MoveAndSend => {
                self.set_gpr(dst, result);
                self.send(result, host)?;
            }
            Assignment::FetchAndSetMethod => {
                let v = self.fetch_arg();
                self.set_gpr(dst, v);
                self.set_method(result);
            }
            Assignment::MoveAndSetMethodThenFetchAndSend => {
                self.set_gpr(dst, result);
                self.set_method(result);
                let v = self.fetch_arg();
                self.send(v, host)?;
            }
            Assignment::MoveAndSetMethodThenSendHigh => {
                self.set_gpr(dst, result);
                self.set_method(result);
                self.send((result >> 12) & 0x3F, host)?;
            }
        }
        Ok(Flow {
            is_branch: false,
            taken: false,
            target: 0,
            annul: false,
            exit,
        })
    }

    fn alu<H: MacroHost + ?Sized>(&mut self, word: u32, host: &mut H) -> Result<u32> {
        let a = self.gpr((word >> 11) & 7);
        let b = self.gpr((word >> 14) & 7);
        match word & 7 {
            0 => {
                let sub_op = (word >> 17) & 0x1F;
                Ok(match sub_op {
                    0 => {
                        let (v, c) = a.overflowing_add(b);
                        self.carry = c;
                        v
                    }
                    1 => {
                        let carry = self.carry as u32;
                        let (v0, c0) = a.overflowing_add(b);
                        let (v1, c1) = v0.overflowing_add(carry);
                        self.carry = c0 || c1;
                        v1
                    }
                    2 => {
                        let (v, c) = a.overflowing_sub(b);
                        self.carry = !c;
                        v
                    }
                    3 => {
                        let borrow = !self.carry as u32;
                        let (v0, c0) = a.overflowing_sub(b);
                        let (v1, c1) = v0.overflowing_sub(borrow);
                        self.carry = !(c0 || c1);
                        v1
                    }
                    8 => a ^ b,
                    9 => a | b,
                    10 => a & b,
                    11 => a & !b,
                    12 => !(a & b),
                    _ => {
                        return Err(Error::Gpu(format!(
                            "mme: unimplemented ALU sub-operation {:#x} (word {:#010x})",
                            sub_op, word
                        )))
                    }
                })
            }
            1 => Ok(a.wrapping_add(imm(word) as u32)),
            2..=4 => {
                let src_bit = (word >> 17) & 0x1F;
                let size = (word >> 22) & 0x1F;
                let dst_bit = (word >> 27) & 0x1F;
                let mask = if size >= 32 {
                    u32::MAX
                } else {
                    (1u32 << size) - 1
                };
                Ok(match word & 7 {
                    // Bitfield replace: splice B's field into A.
                    2 => {
                        let field = (b >> src_bit) & mask;
                        (a & !(mask << dst_bit)) | (field << dst_bit)
                    }
                    // Extract with the shift amount taken from A, then shift
                    // into place by the immediate destination bit.
                    3 => ((b >> (a & 31)) & mask) << dst_bit,
                    // Extract at a fixed bit, then shift by A.
                    _ => ((b >> src_bit) & mask) << (a & 31),
                })
            }
            5 => Ok(host.read_method(a.wrapping_add(imm(word) as u32) & 0xFFF)),
            other => Err(Error::Gpu(format!(
                "mme: unimplemented operation {} (word {:#010x})",
                other, word
            ))),
        }
    }
}

/// The sign-extended 18-bit immediate held in bits 14..31.
fn imm(word: u32) -> i32 {
    (word as i32) >> 14
}

/// Human-readable decode of one MME instruction, for debugging dumps.
pub fn disasm(word: u32) -> String {
    let op = word & 7;
    let assign = (word >> 4) & 7;
    let exit = if word & 0x80 != 0 { " exit" } else { "" };
    let dst = (word >> 8) & 7;
    let a = (word >> 11) & 7;
    let b = (word >> 14) & 7;
    match op {
        7 => {
            let cond = if word & 0x10 != 0 { "bnz" } else { "bz" };
            let annul = if word & 0x20 != 0 { " annul" } else { "" };
            format!("{cond} r{a} -> {}{annul}{exit}", imm(word))
        }
        5 => format!("read r{dst} = [r{a} + {}]{exit}", imm(word)),
        1 => format!("addi r{dst} = r{a} + {} (assign {assign}){exit}", imm(word)),
        2..=4 => {
            let src_bit = (word >> 17) & 0x1F;
            let size = (word >> 22) & 0x1F;
            let dst_bit = (word >> 27) & 0x1F;
            match op {
                2 => {
                    format!("insrt r{dst} = r{a}[{dst_bit}:{size}] <- r{b} (assign {assign}){exit}")
                }
                3 => format!(
                    "extr r{dst} = r{b}[r{a}:{}] << {} (assign {assign}){exit}",
                    size, dst_bit
                ),
                _ => format!(
                    "extr r{dst} = r{b}[{}:{}] << r{a} (assign {assign}){exit}",
                    src_bit, size
                ),
            }
        }
        0 => {
            let sub = (word >> 17) & 0x1F;
            format!("alu r{dst} = r{a} op{sub} r{b} (assign {assign}){exit}")
        }
        other => format!("op{other} (assign {assign}){exit}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Assemble one "move immediate into a register" instruction:
    /// `op = AddImmediate(1)`, A = R0 (zero), so the result is the immediate.
    fn add_imm(dst: u32, src: u32, value: i32, assignment: u32, exit: bool) -> u32 {
        1 | (assignment << 4)
            | ((exit as u32) << 7)
            | (dst << 8)
            | (src << 11)
            | ((value as u32) << 14)
    }

    fn load(engine: &mut MacroEngine, slot: u32, code: &[u32]) {
        engine.start_address_pointer = slot;
        engine.push_start_address(engine.instruction_ram_pointer);
        for &word in code {
            engine.push_instruction(word);
        }
    }

    /// Run a macro and collect its emitted writes.
    fn run_collect(engine: &mut MacroEngine, read: impl Fn(u32) -> u32) -> Result<Vec<MacroWrite>> {
        struct Host<'a, F> {
            read: F,
            writes: &'a mut Vec<MacroWrite>,
        }
        impl<F: Fn(u32) -> u32> MacroHost for Host<'_, F> {
            fn read_method(&self, method: u32) -> u32 {
                (self.read)(method)
            }
            fn write_method(&mut self, write: MacroWrite) -> Result<()> {
                self.writes.push(write);
                Ok(())
            }
        }
        let mut writes = Vec::new();
        let mut host = Host {
            read,
            writes: &mut writes,
        };
        engine.run(&mut host)?;
        Ok(writes)
    }

    #[test]
    fn macro_sets_a_method_and_sends_one_value() {
        let mut engine = MacroEngine::new();
        // R1 holds the first argument (the method address); set it as the
        // method and send the fetched second argument, then exit.
        load(
            &mut engine,
            0,
            &[
                add_imm(2, 1, 0, 6, false), // move R1 -> method, fetch+send arg
                add_imm(0, 0, 0, 1, true),  // no-op carrying the exit bit
                add_imm(0, 0, 0, 1, false),
            ],
        );
        engine.start(0, 0x360); // method 0x360, increment 0
        engine.push_argument(0x1234);
        let writes = run_collect(&mut engine, |_| 0).unwrap();
        assert_eq!(
            writes,
            vec![MacroWrite {
                method: 0x360,
                arg: 0x1234
            }]
        );
    }

    #[test]
    fn method_increment_advances_between_sends() {
        let mut engine = MacroEngine::new();
        load(
            &mut engine,
            1,
            &[
                add_imm(2, 1, 0, 6, false), // set method from R1, send arg 2
                add_imm(3, 0, 0, 6, false), // R0 = 0 -> method 0? no: keeps sending
                add_imm(0, 0, 0, 1, true),
                add_imm(0, 0, 0, 1, false),
            ],
        );
        // Method 0x100 with an increment of 1 in bits 12..17.
        engine.start(1, 0x100 | (1 << 12));
        engine.push_argument(0xAA);
        engine.push_argument(0xBB);
        let writes = run_collect(&mut engine, |_| 0).unwrap();
        assert_eq!(
            writes[0],
            MacroWrite {
                method: 0x100,
                arg: 0xAA
            }
        );
    }

    #[test]
    fn read_opcode_sees_the_class_register_file() {
        let mut engine = MacroEngine::new();
        load(
            &mut engine,
            0,
            &[
                // R2 = read(R1 + 0)
                5 | (1 << 4) | (2 << 8) | (1 << 11),
                // set method from R1, send R2
                2 | (2 << 4) | (0 << 8) | (1 << 11),
                4 | 0, // filler, replaced below
                add_imm(0, 0, 0, 1, true),
                add_imm(0, 0, 0, 1, false),
            ],
        );
        engine.start(0, 0x200);
        let writes =
            run_collect(&mut engine, |method| if method == 0x200 { 0x99 } else { 0 }).unwrap();
        // The macro only sets the method here; what matters is that `read`
        // resolved without faulting and the program exited.
        assert!(writes.is_empty());
    }

    #[test]
    fn runaway_macro_is_caught() {
        let mut engine = MacroEngine::new();
        // Unconditional-ish backward branch on R0 == 0, never exits.
        load(
            &mut engine,
            0,
            &[7 | (0 << 4) | (0 << 11) | ((0i32 as u32) << 14)],
        );
        engine.start(0, 0);
        assert!(run_collect(&mut engine, |_| 0).is_err());
    }

    /// deko3d's `FillRegisters` shape: a counting loop whose exit lives in the
    /// branch's delay slot, so the branch must cancel the exit on the way back
    /// around. This used to spin for `MAX_STEPS` instructions.
    #[test]
    fn branch_delay_slot_cancels_exit() {
        fn branch(src: u32, on_not_zero: bool, annul: bool, imm: i32) -> u32 {
            7 | ((on_not_zero as u32) << 4)
                | ((annul as u32) << 5)
                | (src << 11)
                | ((imm as u32) << 14)
        }
        let mut engine = MacroEngine::new();
        load(
            &mut engine,
            0,
            &[
                add_imm(2, 1, 0, 5, false),  // method addr = R1, fetch count -> R2
                add_imm(3, 0, 0, 0, false),  // fetch value -> R3
                add_imm(2, 2, -1, 1, false), // dec R2
                branch(2, true, false, -1),  // bnz R2 loop
                add_imm(0, 3, 0, 4, true),   // *send R3 (exit)
                add_imm(0, 0, 0, 1, false),  // nop (delay slot of the exit)
            ],
        );
        engine.start(0, 0x100);
        engine.push_argument(3); // three writes
        engine.push_argument(0xAA);
        let writes = run_collect(&mut engine, |_| 0).unwrap();
        assert_eq!(
            writes,
            vec![
                MacroWrite {
                    method: 0x100,
                    arg: 0xAA
                };
                3
            ]
        );
    }

    #[test]
    fn branch_with_exit_exits_only_when_not_taken() {
        fn branch(src: u32, on_not_zero: bool, annul: bool, imm: i32, exit: bool) -> u32 {
            7 | ((on_not_zero as u32) << 4)
                | ((annul as u32) << 5)
                | ((exit as u32) << 7)
                | (src << 11)
                | ((imm as u32) << 14)
        }
        let mut engine = MacroEngine::new();
        // deko3d's `CommonClearLoop` shape: send, then branch-with-exit back to
        // the top while a counter is non-zero; the exit fires when it hits zero.
        load(
            &mut engine,
            0,
            &[
                add_imm(2, 1, 0, 5, false),       // method addr = R1, fetch count -> R2
                add_imm(2, 2, -1, 1, false),      // dec R2
                add_imm(0, 1, 0, 4, false),       // send R1
                branch(2, true, false, -2, true), // *bnz R2 loop
                add_imm(1, 1, 1, 1, false),       // delay slot: R1 += 1
            ],
        );
        engine.start(0, 0x200);
        engine.push_argument(2); // two sends
        let writes = run_collect(&mut engine, |_| 0).unwrap();
        assert_eq!(
            writes,
            vec![
                MacroWrite {
                    method: 0x200,
                    arg: 0x200
                },
                MacroWrite {
                    method: 0x200,
                    arg: 0x201
                }
            ]
        );
    }
}
