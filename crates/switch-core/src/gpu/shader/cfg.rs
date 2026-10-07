//! Control flow recovered from a lowered program.
//!
//! Checks whether Maxwell's reconvergence stack pushes and pops pair up
//! statically, which decides if a shader can be structured. See [`Cfg::pairing`].

use super::compiled::{Compiled, NO_TARGET};
use super::isa::Op;
use std::collections::HashMap;

/// Which reconvergence stack an instruction uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reconverge {
    /// `ssy` / `sync`, the two arms of a branch rejoining.
    Sync,
    /// `pbk` / `brk`, leaving a loop.
    Break,
    /// `pcnt` / `cont`, the next iteration of one.
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pairing {
    /// Every pop has one possible push and every instruction one stack.
    Static,
    /// Some instruction is reachable with two different reconvergence stacks.
    PathDependent { at: usize },
    /// A pop with no matching push, or with the stack empty.
    Unbalanced { at: usize },
    /// The walk gave up, and why.
    Unknown(Give),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Give {
    /// A `brx` whose jump table the decoder could not read.
    IndirectBranch { at: usize },
    /// A branch or reconvergence push whose target was never decoded.
    UndecodedTarget { at: usize },
    /// More states than any real shader has.
    TooManyStates,
}

/// The live reconvergence targets, innermost last.
type Stack = Vec<(Reconverge, usize)>;

pub struct Cfg<'a> {
    program: &'a Compiled,
    /// The stack each instruction is reached with, if the walk reached it.
    stacks: HashMap<usize, Stack>,
    pairing: Pairing,
}

/// Walk state limit, well above any real shader.
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

    pub fn pairing(&self) -> &Pairing {
        &self.pairing
    }

    /// Instructions the walk reached; anything else is dead or unfollowable.
    pub fn reachable(&self) -> usize {
        self.stacks.len()
    }

    /// The deepest reconvergence stack the walk saw.
    pub fn max_depth(&self) -> usize {
        self.stacks.values().map(|s| s.len()).max().unwrap_or(0)
    }

    /// One-line summary, with byte offsets as a shader dump shows them.
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
            // Revisits must arrive with the same stack.
            if let Some(seen) = self.stacks.get(&at) {
                if *seen != stack && self.pairing == Pairing::Static {
                    self.pairing = Pairing::PathDependent { at };
                }
                continue;
            }
            self.stacks.insert(at, stack.clone());

            let op = self.program.op(at);
            let predicated = !self.program.pred(at).is_always();

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
                // `brx` arms come from the decoded jump table.
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

    /// A program laid out at real 32-byte-block byte offsets.
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

    fn at(index: usize) -> u32 {
        let mut offset = ENTRY_OFFSET;
        for _ in 0..index {
            offset = next_slot(offset);
        }
        offset
    }

    const ALWAYS: Pred = Pred::ALWAYS;
    /// `@p0`: the guard a two-armed branch is built out of.
    const IF_P0: Pred = Pred {
        reg: 0,
        negate: false,
    };

    #[test]
    fn a_two_armed_branch_pairs_statically() {
        // An `if`/`else`: push the join, branch to else, both arms `sync`.
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
        // Instruction 4 is reached both inside and outside the `ssy` region.
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
        // `ssy` pushes a join but `brk` wants a loop exit.
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
        let p = program(&[(Op::Brx { base: 0, reg: 1 }, ALWAYS), (Op::Exit, ALWAYS)]);
        assert_eq!(
            Cfg::new(&p).pairing(),
            &Pairing::Unknown(Give::IndirectBranch { at: 0 })
        );
    }

    #[test]
    fn a_loop_body_reached_twice_with_the_same_stack_is_still_static() {
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
