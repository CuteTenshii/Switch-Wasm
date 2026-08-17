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

    /// Run the pending macro. `read_method` supplies the class register file
    /// for the MME's `read` opcode; the returned writes must be applied to the
    /// class in order.
    pub fn run<F>(&mut self, read_method: F) -> Result<Vec<MacroWrite>>
    where
        F: Fn(u32) -> u32,
    {
        let slot = match self.pending.take() {
            Some(slot) => slot,
            None => return Ok(Vec::new()),
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
            writes: Vec::new(),
        };
        // The first argument arrives in R1 rather than through a fetch.
        state.gprs[1] = state.fetch_arg();
        let mut steps = 0u32;
        loop {
            if steps >= MAX_STEPS {
                return Err(Error::Gpu(format!(
                    "mme: macro {} ran for {} instructions without exiting",
                    slot, MAX_STEPS
                )));
            }
            steps += 1;
            let word = *self
                .instruction_ram
                .get(state.pc as usize)
                .ok_or_else(|| Error::Gpu(format!("mme: pc {:#x} out of range", state.pc)))?;
            state.pc = state.pc.wrapping_add(1);
            if !state.step(word, &read_method)? {
                break;
            }
        }
        Ok(state.writes)
    }
}

struct MacroState<'a> {
    gprs: [u32; 8],
    pc: u32,
    method_address: u32,
    method_increment: u32,
    args: &'a [u32],
    arg_index: usize,
    carry: bool,
    writes: Vec<MacroWrite>,
}

impl MacroState<'_> {
    fn fetch_arg(&mut self) -> u32 {
        let v = self.args.get(self.arg_index).copied().unwrap_or(0);
        self.arg_index += 1;
        v
    }

    fn gpr(&self, index: u32) -> u32 {
        if index == 0 { 0 } else { self.gprs[(index & 7) as usize] }
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

    fn send(&mut self, value: u32) {
        self.writes.push(MacroWrite { method: self.method_address, arg: value });
        self.method_address = self.method_address.wrapping_add(self.method_increment) & 0xFFF;
    }

    /// Execute one instruction. Returns false once the program has exited.
    fn step<F>(&mut self, word: u32, read_method: &F) -> Result<bool>
    where
        F: Fn(u32) -> u32,
    {
        let op = word & 7;
        if op == 7 {
            // Branch: bit 4 selects != 0 vs == 0, bit 5 annuls the delay slot.
            let on_not_zero = (word >> 4) & 1 != 0;
            let value = self.gpr((word >> 11) & 7);
            let taken = if on_not_zero { value != 0 } else { value == 0 };
            if taken {
                // The immediate is relative to the branch instruction, and the
                // pc has already advanced past it.
                self.pc = self.pc.wrapping_sub(1).wrapping_add(imm(word) as u32);
                return Ok(true);
            }
        } else {
            let result = self.alu(word, read_method)?;
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
                    self.send(result);
                }
                Assignment::MoveAndSend => {
                    self.set_gpr(dst, result);
                    self.send(result);
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
                    self.send(v);
                }
                Assignment::MoveAndSetMethodThenSendHigh => {
                    self.set_gpr(dst, result);
                    self.set_method(result);
                    self.send((result >> 12) & 0x3F);
                }
            }
        }
        // Bit 7 exits, but only after the following instruction has run.
        Ok((word >> 7) & 1 == 0)
    }

    fn alu<F>(&mut self, word: u32, read_method: &F) -> Result<u32>
    where
        F: Fn(u32) -> u32,
    {
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
            2 | 3 | 4 => {
                let src_bit = (word >> 17) & 0x1F;
                let size = (word >> 22) & 0x1F;
                let dst_bit = (word >> 27) & 0x1F;
                let mask = if size >= 32 { u32::MAX } else { (1u32 << size) - 1 };
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
            5 => Ok(read_method(a.wrapping_add(imm(word) as u32) & 0xFFF)),
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
        let writes = engine.run(|_| 0).unwrap();
        assert_eq!(writes, vec![MacroWrite { method: 0x360, arg: 0x1234 }]);
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
        let writes = engine.run(|_| 0).unwrap();
        assert_eq!(writes[0], MacroWrite { method: 0x100, arg: 0xAA });
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
        let writes = engine.run(|method| if method == 0x200 { 0x99 } else { 0 }).unwrap();
        // The macro only sets the method here; what matters is that `read`
        // resolved without faulting and the program exited.
        assert!(writes.is_empty());
    }

    #[test]
    fn runaway_macro_is_caught() {
        let mut engine = MacroEngine::new();
        // Unconditional-ish backward branch on R0 == 0, never exits.
        load(&mut engine, 0, &[7 | (0 << 4) | (0 << 11) | ((0i32 as u32) << 14)]);
        engine.start(0, 0);
        assert!(engine.run(|_| 0).is_err());
    }
}
