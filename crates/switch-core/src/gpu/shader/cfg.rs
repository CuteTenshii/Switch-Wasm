//! Control flow recovered from a lowered program.
//!
//! Maxwell does not have structured control flow. It has a *reconvergence
//! stack*: `ssy` pushes an address and `sync` pops one and jumps there, and
//! the same for `pbk`/`brk` and `pcnt`/`cont`. The interpreter can run that
//! directly — it just keeps the stack — but a translator to any shading
//! language cannot, because WGSL and GLSL both want `if`/`else`/`loop` and
//! have nothing that pops a jump target off a runtime stack.
//!
//! Reconstructing structure from a stack machine is the hard part of
//! translating these shaders, and how hard depends entirely on a question
//! that can be answered by looking: **do the pushes and pops pair up
//! statically?** If every `sync` in a program can only ever pop the address
//! one particular `ssy` pushed, then each pair is an `if` (or a loop, for
//! `pbk`/`pcnt`) and the translation is mechanical. If a `sync` can be
//! reached with two different stacks, it cannot be, and the arms have to be
//! executed under per-lane masks instead — a far bigger and slower thing.
//!
//! So this does not try to build structured output yet. It answers the
//! question, over whatever shaders are actually put through it. See
//! [`Cfg::pairing`].

use super::compiled::{Compiled, NO_TARGET};
use super::isa::Op;
use std::collections::HashMap;

/// Which reconvergence stack an instruction uses. The hardware has one stack
/// with three kinds of entry on it, and a `sync` pops whatever is on top
/// regardless of what pushed it — but a program in which those interleave is
/// one no compiler emits, and [`Cfg::pairing`] reports it rather than
/// pretending otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reconverge {
    /// `ssy` / `sync` — the two arms of a branch rejoining.
    Sync,
    /// `pbk` / `brk` — leaving a loop.
    Break,
    /// `pcnt` / `cont` — the next iteration of one.
    Continue,
}

impl Reconverge {
    fn of_push(op: Op) -> Option<Reconverge> {
        match op {
            Op::Ssy { .. } => Some(Reconverge::Sync),
            Op::Pbk { .. } => Some(Reconverge::Break),
            Op::Pcnt { .. } => Some(Reconverge::Continue),
            _ => None,
        }
    }

    fn of_pop(op: Op) -> Option<Reconverge> {
        match op {
            Op::Sync => Some(Reconverge::Sync),
            Op::Brk => Some(Reconverge::Break),
            Op::Cont => Some(Reconverge::Continue),
            _ => None,
        }
    }
}

/// What a walk found out about a program's reconvergence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pairing {
    /// Every pop has exactly one push that could have put its target there,
    /// and every instruction is reached with one stack whatever path leads to
    /// it. This is the shape a `switch`, an `if`/`else` and a `for` lower to,
    /// and the shape that translates to structured control flow directly.
    Static,
    /// Some instruction is reachable with two different reconvergence stacks,
    /// so what a pop jumps to depends on the path taken to it. Named because
    /// it is the case a translator cannot handle by nesting blocks.
    PathDependent { at: usize },
    /// A pop of a kind that nothing on the stack pushed, or a pop with the
    /// stack empty. Either the program is malformed or the walk cannot follow
    /// it — a `brx` whose targets are unknown, most likely.
    Unbalanced { at: usize },
    /// The walk gave up, and why. It says so rather than reporting a
    /// conclusion it did not reach.
    Unknown(Give),
}

/// Why a walk stopped without an answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Give {
    /// A `brx` whose jump table the decoder could not read, so its arms are
    /// not known and the walk cannot follow them.
    IndirectBranch { at: usize },
    /// A branch or reconvergence push whose target was never decoded.
    UndecodedTarget { at: usize },
    /// More states than any real shader has. Either the program loops in a
    /// way this walk does not collapse, or it is not a shader.
    TooManyStates,
}

/// The reconvergence stack as the walk models it: the targets that are live,
/// innermost last.
type Stack = Vec<(Reconverge, usize)>;

/// A program's control flow.
pub struct Cfg<'a> {
    program: &'a Compiled,
    /// The stack each instruction is reached with, if the walk reached it.
    stacks: HashMap<usize, Stack>,
    pairing: Pairing,
}

/// Enough to stop a pathological program spinning the walk, generously above
/// any real shader: the largest this has seen is 1336 instructions.
const MAX_VISITS: usize = 1 << 16;

impl<'a> Cfg<'a> {
    /// Walk `program` from its entry, tracking the reconvergence stack.
    pub fn new(program: &'a Compiled) -> Cfg<'a> {
        let mut cfg = Cfg {
            program,
            stacks: HashMap::new(),
            pairing: Pairing::Static,
        };
        cfg.walk();
        cfg
    }

    /// What the walk concluded about this program's reconvergence.
    pub fn pairing(&self) -> &Pairing {
        &self.pairing
    }

    /// The instructions the walk reached. An instruction absent from this is
    /// dead code, or behind a branch the walk could not follow.
    pub fn reachable(&self) -> usize {
        self.stacks.len()
    }

    /// The deepest reconvergence stack the walk saw — how far an `if` inside
    /// an `if` inside a loop nests, which is what a structured translation
    /// would have to reproduce.
    pub fn max_depth(&self) -> usize {
        self.stacks.values().map(|s| s.len()).max().unwrap_or(0)
    }

    /// One line describing what the walk found, with byte offsets rather than
    /// instruction indices — those are the addresses a shader dump shows.
    pub fn describe(&self) -> String {
        let at = |i: usize| format!("{:#x}", self.program.offset(i));
        let verdict = match &self.pairing {
            Pairing::Static => "static".to_string(),
            Pairing::PathDependent { at: i } => format!("path-dependent at {}", at(*i)),
            Pairing::Unbalanced { at: i } => format!("unbalanced pop at {}", at(*i)),
            Pairing::Unknown(Give::IndirectBranch { at: i }) => {
                format!("brx with no known targets at {}", at(*i))
            }
            Pairing::Unknown(Give::UndecodedTarget { at: i }) => {
                format!("branch to undecoded target at {}", at(*i))
            }
            Pairing::Unknown(Give::TooManyStates) => "too many states".to_string(),
        };
        format!(
            "{} insns, {} reached, depth {}, {verdict}",
            self.program.len(),
            self.reachable(),
            self.max_depth()
        )
    }

    fn walk(&mut self) {
        let mut queue: Vec<(usize, Stack)> = vec![(0, Vec::new())];
        let mut visits = 0usize;
        while let Some((at, stack)) = queue.pop() {
            visits += 1;
            if visits > MAX_VISITS {
                self.pairing = Pairing::Unknown(Give::TooManyStates);
                return;
            }
            if at >= self.program.len() {
                continue;
            }
            // Reaching an instruction a second time is fine, and normal — a
            // loop body does it — but only with the same stack. With a
            // different one, what its pops jump to depends on the path.
            if let Some(seen) = self.stacks.get(&at) {
                if *seen != stack && self.pairing == Pairing::Static {
                    self.pairing = Pairing::PathDependent { at };
                }
                continue;
            }
            self.stacks.insert(at, stack.clone());

            let op = self.program.op(at);
            let predicated = !self.program.pred(at).is_always();

            // A push, then whatever the instruction does with control flow.
            if let Some(kind) = Reconverge::of_push(op) {
                let target = self.program.target(at);
                if target == NO_TARGET {
                    self.pairing = Pairing::Unknown(Give::UndecodedTarget { at });
                    return;
                }
                let mut pushed = stack.clone();
                pushed.push((kind, target as usize));
                queue.push((at + 1, pushed));
                continue;
            }

            if let Some(kind) = Reconverge::of_pop(op) {
                match stack.last() {
                    Some(&(top, target)) if top == kind => {
                        let mut popped = stack.clone();
                        popped.pop();
                        queue.push((target, popped));
                    }
                    _ => {
                        if matches!(self.pairing, Pairing::Static) {
                            self.pairing = Pairing::Unbalanced { at };
                        }
                    }
                }
                // A predicated pop can also fall through.
                if predicated {
                    queue.push((at + 1, stack));
                }
                continue;
            }

            match op {
                Op::Exit | Op::Kil => {
                    if predicated {
                        queue.push((at + 1, stack));
                    }
                }
                Op::Bra { .. } => {
                    let target = self.program.target(at);
                    if target == NO_TARGET {
                        self.pairing = Pairing::Unknown(Give::UndecodedTarget { at });
                        return;
                    }
                    queue.push((target as usize, stack.clone()));
                    if predicated {
                        queue.push((at + 1, stack));
                    }
                }
                // The target is a register value, but the decoder read the
                // jump table to find the arms in the first place, so the walk
                // can follow every one of them.
                Op::Brx { .. } => match self.program.indirect_targets(at) {
                    Some(targets) => {
                        for &target in targets {
                            queue.push((target as usize, stack.clone()));
                        }
                    }
                    None => {
                        self.pairing = Pairing::Unknown(Give::IndirectBranch { at });
                        return;
                    }
                },
                _ => queue.push((at + 1, stack)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::shader::isa::{Instruction, Pred};
    use crate::gpu::shader::{next_slot, Program, ENTRY_OFFSET};

    /// A program at the byte offsets a real 32-byte-block layout would put it
    /// at, so branch targets resolve the way they do in a decoded shader.
    fn program(entries: &[(Op, Pred)]) -> Compiled {
        let mut p = Program::default();
        let mut offset = ENTRY_OFFSET;
        for &(op, pred) in entries {
            p.insns.push(Instruction { pred, op });
            p.offsets.push(offset);
            offset = next_slot(offset);
        }
        Compiled::new(&p)
    }

    /// The byte offset instruction `index` lands at.
    fn at(index: usize) -> u32 {
        let mut offset = ENTRY_OFFSET;
        for _ in 0..index {
            offset = next_slot(offset);
        }
        offset
    }

    const ALWAYS: Pred = Pred::ALWAYS;
    /// `@p0` — the guard a two-armed branch is built out of.
    const IF_P0: Pred = Pred {
        reg: 0,
        negate: false,
    };

    #[test]
    fn a_two_armed_branch_pairs_statically() {
        // The shape an `if`/`else` lowers to: push the join, branch to the
        // else arm, both arms `sync` back to it.
        let p = program(&[
            (Op::Ssy { target: at(5) }, ALWAYS),
            (Op::Bra { target: at(3) }, IF_P0),
            (Op::Sync, ALWAYS),
            (Op::Nop, ALWAYS),
            (Op::Sync, ALWAYS),
            (Op::Exit, ALWAYS),
        ]);
        let cfg = Cfg::new(&p);
        assert_eq!(cfg.pairing(), &Pairing::Static);
        assert_eq!(cfg.reachable(), 6, "every instruction is on some path");
        assert_eq!(cfg.max_depth(), 1, "one join point live at a time");
    }

    #[test]
    fn nesting_deepens_the_stack() {
        let p = program(&[
            (Op::Pbk { target: at(5) }, ALWAYS),
            (Op::Ssy { target: at(4) }, ALWAYS),
            (Op::Nop, ALWAYS),
            (Op::Sync, ALWAYS),
            (Op::Brk, ALWAYS),
            (Op::Exit, ALWAYS),
        ]);
        let cfg = Cfg::new(&p);
        assert_eq!(cfg.pairing(), &Pairing::Static);
        assert_eq!(cfg.max_depth(), 2, "an if inside a loop");
    }

    #[test]
    fn reaching_a_pop_with_two_different_stacks_is_path_dependent() {
        // This is the case a translator cannot nest into blocks: instruction 4
        // is reached both inside the `ssy` region and outside it, so what its
        // `sync` jumps to depends on how control got there.
        let p = program(&[
            (Op::Bra { target: at(3) }, IF_P0),
            (Op::Ssy { target: at(6) }, ALWAYS),
            (Op::Bra { target: at(4) }, ALWAYS),
            (Op::Nop, ALWAYS),
            (Op::Sync, ALWAYS),
            (Op::Nop, ALWAYS),
            (Op::Exit, ALWAYS),
        ]);
        assert!(
            matches!(Cfg::new(&p).pairing(), Pairing::PathDependent { .. }),
            "got {:?}",
            Cfg::new(&p).pairing()
        );
    }

    #[test]
    fn a_pop_with_nothing_pushed_is_unbalanced() {
        let p = program(&[(Op::Sync, ALWAYS), (Op::Exit, ALWAYS)]);
        assert_eq!(Cfg::new(&p).pairing(), &Pairing::Unbalanced { at: 0 });
    }

    #[test]
    fn a_pop_of_the_wrong_kind_is_unbalanced() {
        // `ssy` pushes a join and `brk` wants a loop exit. Real code never
        // does this; saying so is how the walk stays honest about what it can
        // conclude.
        let p = program(&[
            (Op::Ssy { target: at(3) }, ALWAYS),
            (Op::Brk, ALWAYS),
            (Op::Nop, ALWAYS),
            (Op::Exit, ALWAYS),
        ]);
        assert_eq!(Cfg::new(&p).pairing(), &Pairing::Unbalanced { at: 1 });
    }

    #[test]
    fn a_brx_with_no_known_targets_stops_the_walk() {
        // The walk reports what stopped it rather than a verdict it did not
        // reach. Three of the Home Menu's fragment shaders used to end here,
        // until the jump-table walk learned to follow a selector back to a
        // clamp the scheduler had hoisted out of its window.
        let p = program(&[(Op::Brx { base: 0, reg: 1 }, ALWAYS), (Op::Exit, ALWAYS)]);
        assert_eq!(
            Cfg::new(&p).pairing(),
            &Pairing::Unknown(Give::IndirectBranch { at: 0 })
        );
    }

    #[test]
    fn a_loop_body_reached_twice_with_the_same_stack_is_still_static() {
        // Revisiting an instruction is normal; it is only revisiting it with a
        // *different* stack that defeats a structured translation.
        let p = program(&[
            (Op::Nop, ALWAYS),
            (Op::Bra { target: at(0) }, IF_P0),
            (Op::Exit, ALWAYS),
        ]);
        let cfg = Cfg::new(&p);
        assert_eq!(cfg.pairing(), &Pairing::Static);
        assert_eq!(cfg.reachable(), 3);
    }
}
