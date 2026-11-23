use super::*;
use crate::cpu::jit::ir::Block;

fn block_of(ops: Vec<Op>) -> Block {
    let words = vec![0u32; ops.len()];
    Block::new(0x1000, ops, words, Vec::new(), None, vec![1])
}

const LAYOUT: Layout = Layout {
    regs: 0,
    nzcv: 2048,
    fpsr: 2092,
    fpcr: 2180,
    fpscr_nzcv: 2184,
    pc: 2084,
    fuel: 2088,
    memory: 2096,
    blocks: 2052,
    block_pages: 0,
    block_watched: 2048,
    read_watch_lo: 2056,
    read_watch_hi: 2064,
    watch_lo: 2072,
    watch_hi: 2080,
    pair_lock: 2176,
    vregs: 4096,
    exclusive: 2112,
    tpidr: 2144,
    tpidr_rw: 2152,
    core: 2160,
    clock_at: 2164,
    origin: 2168,
};

#[test]
fn an_unwritable_op_refuses_the_whole_block() {
    let good = block_of(vec![Op::MovConst { rd: 0, val: 7 }]);
    assert!(emit_block(&good, LAYOUT, true).is_ok());

    let ops = vec![
        Op::MovConst { rd: 0, val: 7 },
        Op::Interpret { insn: 0xD503201F },
    ];
    let words = vec![0, 0xD503201F];
    let bad = Block::new(0x1000, ops, words, Vec::new(), None, vec![1]);
    assert_eq!(emit_block(&bad, LAYOUT, true), Err(Refused::Op(0xD503201F)));
}

/// Walks guest state from the `Cpu`'s address by [`Layout::of_cpu`]
/// offsets alone, as emitted code does, to an address above 4 GiB.
#[test]
fn emitted_code_finds_guest_state_through_the_layout() {
    const SLOT: u8 = 5;
    const VALUE: u64 = 0x0123_4567_89AB_CDEF;
    const ADDR: u64 = 0x20_0800_1234;
    const BYTE: u8 = 0xA7;

    let mut cpu = Cpu::new();
    cpu.set_reg_at(SLOT, VALUE);
    cpu.nzcv = 0xC000_0000;
    cpu.exclusive.arm(ADDR, [VALUE, 0]);
    cpu.mem.write_u8(ADDR, BYTE).expect("the page is writable");
    cpu.mem.watch_reads(0x1000, 0x40);
    cpu.mem.watch_writes(0x2000, 0x80);
    cpu.mem.mark_readonly(0x3000, 0x4000);
    cpu.mem.mark_code_page(ADDR);

    let layout = Layout::of_cpu();
    let base = &cpu as *const Cpu as usize;
    // SAFETY: every read is at an in-bounds offset from a live `Cpu`, of
    // the type the field there holds.
    unsafe {
        let memory = *((base + layout.memory as usize) as *const usize);
        let u64_at = |off: u32| *((memory + off as usize) as *const u64);

        assert_eq!(
            *((base + layout.regs as usize + 8 * SLOT as usize) as *const u64),
            VALUE,
            "the register file is not where an emitted block reads it"
        );
        assert_eq!(
            *((base + layout.nzcv as usize) as *const u32),
            0xC000_0000,
            "NZCV moved"
        );

        let monitor = (base + layout.exclusive as usize) as *const u8;
        assert_eq!(
            (
                *(monitor.add(MONITOR_ADDR as usize) as *const u64),
                *(monitor.add(MONITOR_ARMED as usize) as *const u32),
                *(monitor.add(MONITOR_VALUE as usize) as *const u64),
            ),
            (ADDR, 1, VALUE),
            "the exclusive monitor moved"
        );

        assert_eq!(
            (u64_at(layout.read_watch_lo), u64_at(layout.read_watch_hi)),
            (0x1000, 0x1040),
            "the read watchpoint moved"
        );
        assert_eq!(
            (u64_at(layout.watch_lo), u64_at(layout.watch_hi)),
            (0x2000, 0x2080),
            "the write watchpoint moved"
        );

        // The walk: the table, the address's block, then its page's entry and bit.
        let table = *((memory + layout.blocks as usize) as *const *const *const u8);
        let block_of = |addr: u64| {
            *table.add(((addr >> (PAGE_BITS + BLOCK_BITS)) as usize) & (BLOCK_COUNT - 1))
        };
        let in_block = |addr: u64| ((addr >> PAGE_BITS) as usize) & (BLOCK_PAGES - 1);
        let watched = |addr: u64| {
            let bits = block_of(addr).add(layout.block_watched as usize) as *const u64;
            *bits.add(in_block(addr) >> 6) & (1u64 << (in_block(addr) & 63)) != 0
        };
        let entry = |addr: u64| {
            let pages = block_of(addr).add(layout.block_pages as usize) as *const *const u8;
            *pages.add(in_block(addr))
        };
        assert!(
            watched(ADDR),
            "the marked page's bit is not where an emitted store looks"
        );
        assert!(
            watched(0x3000),
            "a write-protected page is not set where an emitted store looks"
        );
        assert!(!entry(ADDR).is_null(), "a written page has no storage");
        assert_eq!(
            *entry(ADDR).add((ADDR as usize) & (PAGE_SIZE - 1)),
            BYTE,
            "the page-table walk does not reach the byte that was written"
        );
        const NOWHERE: u64 = 0x40_F000_0000;
        assert!(
            !block_of(NOWHERE).is_null(),
            "an untouched block's entry is null, which the walk does not test for"
        );
        assert!(
            entry(NOWHERE).is_null(),
            "an unmapped page is not a null entry"
        );
        assert!(
            !watched(NOWHERE),
            "an untouched block has a watched bit set"
        );
    }
}

/// The same for what a jump between regions touches.
#[test]
fn a_jump_finds_the_chain_table_through_its_layout() {
    let mut cpu = Cpu::new();
    cpu.jit.chain.fuel = -7;
    cpu.jit.chain.hops = 11;
    cpu.jit.chain.last = 0x0800_0010;
    cpu.jit.chain.slots[5] = [0x0800_0014, 42];

    let at = ChainLayout::OF_CPU;
    let fuel = Layout::of_cpu().fuel;
    let base = &cpu as *const Cpu as usize;
    // SAFETY: as above, in-bounds offsets from a live `Cpu` of the right types.
    unsafe {
        let u32_at = |off: u32| *((base + off as usize) as *const u32);
        assert_eq!(u32_at(fuel) as i32, -7, "the fuel moved");
        assert_eq!(u32_at(at.hops), 11, "the jump count moved");
        assert_eq!(u32_at(at.last), 0x0800_0010, "the last target moved");
        let slots = *((base + at.slots as usize) as *const *const u32);
        assert_eq!(
            (*slots.add(10), *slots.add(11)),
            (0x0800_0014, 42),
            "a chain slot is not two words at eight bytes a slot"
        );
    }
}

#[test]
fn a_followed_branch_is_emitted() {
    use crate::cpu::jit::ir::{Branch, Exit};

    let ops = vec![Op::Nop, Op::MovConst { rd: 0, val: 7 }];
    let exits = vec![Branch::new(0, Exit::Jump { target: 0x2000 })];
    let b = Block::new(0x1000, ops, vec![0, 0], exits, None, vec![1]);
    assert!(emit_block(&b, LAYOUT, true).is_ok());
}

#[test]
fn an_exit_past_the_body_refuses_rather_than_being_dropped() {
    use crate::cpu::jit::ir::{Branch, Exit};

    let ops = vec![Op::MovConst { rd: 0, val: 7 }];
    let exits = vec![Branch::new(4, Exit::Cond { cond: 0, target: 8 })];
    let b = Block::new(0x1000, ops, vec![0], exits, None, vec![1]);
    assert_eq!(emit_block(&b, LAYOUT, true), Err(Refused::ControlFlow));
}

#[test]
fn every_conditional_branch_is_written() {
    use crate::cpu::jit::ir::{Branch, Exit};

    let branches = [
        Exit::Cond {
            cond: 0,
            target: 0x2000,
        },
        Exit::Cbz {
            rt: 3,
            sf: true,
            nz: false,
            target: 0x2000,
        },
        Exit::Tbz {
            rt: 3,
            bit: 40,
            nz: true,
            target: 0x2000,
        },
        Exit::CmpImm {
            rn: 3,
            imm: 0x20,
            carry: 1,
            sf: false,
            cond: 11,
            target: 0x2000,
        },
        Exit::CmpReg {
            rn: 3,
            rm: 4,
            carry: 1,
            sf: true,
            cond: 11,
            target: 0x2000,
        },
    ];
    for exit in branches {
        // Ops before and after the branch.
        let ops = vec![Op::MovConst { rd: 0, val: 7 }, Op::Nop, Op::Nop];
        let exits = vec![Branch::new(1, exit)];
        let b = Block::new(0x1000, ops, vec![0; 3], exits, None, vec![1]);
        assert!(
            emit_block(&b, LAYOUT, true).is_ok(),
            "{exit:?} was not written out"
        );
    }
}

/// A fused compare retires two instructions with the branch.
#[test]
fn a_fused_compare_retires_the_pair() {
    use crate::cpu::jit::ir::{Branch, Exit};

    let pair = Branch::new(
        0,
        Exit::CmpImm {
            rn: 3,
            imm: 1,
            carry: 1,
            sf: true,
            cond: 0,
            target: 0x2000,
        },
    );
    assert_eq!(pair.span, 2);
    let b = Block::new(
        0x1000,
        vec![Op::Nop, Op::Nop],
        vec![0; 2],
        vec![pair],
        None,
        vec![1],
    );
    let bytes = emit_block(&b, LAYOUT, true).expect("a fused compare is written");
    // `2 | LEFT` as the signed LEB128 `i32.const` operand.
    let mut want = Vec::new();
    super::super::wasm::sleb(&mut want, i64::from((2u32 | LEFT) as i32));
    assert!(
        bytes.windows(want.len()).any(|w| w == want),
        "the leave does not report the pair"
    );
}
