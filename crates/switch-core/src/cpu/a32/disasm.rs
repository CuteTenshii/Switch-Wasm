//! A compact A32 disassembler, for reading a fault trace of 32-bit code.
//!
//! Not a full one — the shifter's syntax stops at naming the shift — but
//! enough to follow a crash back to the call that caused it, which is what a
//! trace is for.

use super::shift::expand_imm_c;

const COND: [&str; 16] = [
    "eq", "ne", "cs", "cc", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le", "", "",
];
const OPS: [&str; 16] = [
    "and", "eor", "sub", "rsb", "add", "adc", "sbc", "rsc", "tst", "teq", "cmp", "cmn", "orr",
    "mov", "bic", "mvn",
];
const SHIFTS: [&str; 4] = ["lsl", "lsr", "asr", "ror"];

/// `{r0, r4-r6, lr}` from a block transfer's 16-bit list, which is far easier
/// to read against a prologue than the raw mask.
fn register_list(list: u32) -> String {
    let name = |r: u32| match r {
        13 => "sp".to_string(),
        14 => "lr".to_string(),
        15 => "pc".to_string(),
        _ => format!("r{r}"),
    };
    let mut parts: Vec<String> = Vec::new();
    let mut reg = 0;
    while reg < 16 {
        if list & (1 << reg) == 0 {
            reg += 1;
            continue;
        }
        let start = reg;
        while reg < 16 && list & (1 << reg) != 0 {
            reg += 1;
        }
        let end = reg - 1;
        parts.push(match end - start {
            0 => name(start),
            1 => format!("{}, {}", name(start), name(end)),
            _ => format!("{}-{}", name(start), name(end)),
        });
    }
    format!("{{{}}}", parts.join(", "))
}

/// The second operand of a data-processing instruction.
fn operand2(insn: u32) -> String {
    if (insn >> 25) & 1 != 0 {
        let (imm, _) = expand_imm_c(insn & 0xFFF, false);
        return format!("#{imm:#x}");
    }
    let rm = insn & 0xF;
    let ty = ((insn >> 5) & 0b11) as usize;
    if (insn >> 4) & 1 != 0 {
        return format!("r{rm}, {} r{}", SHIFTS[ty], (insn >> 8) & 0xF);
    }
    let amount = (insn >> 7) & 0x1F;
    match (ty, amount) {
        (0, 0) => format!("r{rm}"),
        (3, 0) => format!("r{rm}, rrx"),
        (1 | 2, 0) => format!("r{rm}, {} #32", SHIFTS[ty]),
        _ => format!("r{rm}, {} #{amount}", SHIFTS[ty]),
    }
}

/// Name an A32 encoding: the mnemonic, its condition and its operands.
pub fn disassemble_a32(insn: u32) -> String {
    let cond = COND[((insn >> 28) & 0xF) as usize];
    let rn = (insn >> 16) & 0xF;
    let rd = (insn >> 12) & 0xF;
    let rm = insn & 0xF;

    if (insn >> 28) & 0xF == 0xF {
        return match insn {
            _ if (insn & 0xFFFF_FFF0) == 0xF57F_F040 => "dsb".into(),
            _ if (insn & 0xFFFF_FFF0) == 0xF57F_F050 => "dmb".into(),
            _ if (insn & 0xFFFF_FFF0) == 0xF57F_F060 => "isb".into(),
            _ if (insn & 0xFD30_F000) == 0xF510_F000 => "pld".into(),
            _ if (insn & 0xFE00_0000) == 0xFA00_0000 => "blx (thumb)".into(),
            _ => format!(".word {insn:#010x}"),
        };
    }

    match (insn >> 25) & 0x7 {
        0b101 => {
            let imm = ((insn & 0x00FF_FFFF) << 8) as i32 >> 6;
            let kind = if (insn >> 24) & 1 != 0 { "bl" } else { "b" };
            // Relative to the instruction, which is what the trace's own
            // addresses let a reader add up.
            format!("{kind}{cond} pc{}{:#x}", if imm < 0 { "-" } else { "+" }, imm.unsigned_abs() + 8)
        }
        0b100 => {
            let dir = match ((insn >> 24) & 1, (insn >> 23) & 1) {
                (0, 1) => "ia",
                (1, 1) => "ib",
                (0, 0) => "da",
                _ => "db",
            };
            let kind = if (insn >> 20) & 1 != 0 { "ldm" } else { "stm" };
            let bang = if (insn >> 21) & 1 != 0 { "!" } else { "" };
            let base = if rn == 13 { "sp".into() } else { format!("r{rn}") };
            format!("{kind}{dir}{cond} {base}{bang}, {}", register_list(insn & 0xFFFF))
        }
        0b010 | 0b011 => {
            if (insn >> 25) & 1 != 0 && insn & 0x10 != 0 {
                return format!("media{cond} {insn:#010x}");
            }
            let kind = if (insn >> 20) & 1 != 0 { "ldr" } else { "str" };
            let byte = if (insn >> 22) & 1 != 0 { "b" } else { "" };
            let sign = if (insn >> 23) & 1 != 0 { "" } else { "-" };
            let offset = if (insn >> 25) & 1 != 0 {
                format!(", {sign}{}", operand2(insn & !(1 << 25)))
            } else if insn & 0xFFF != 0 {
                format!(", #{sign}{:#x}", insn & 0xFFF)
            } else {
                String::new()
            };
            let base = if rn == 13 { "sp".into() } else { format!("r{rn}") };
            if (insn >> 24) & 1 != 0 {
                let bang = if (insn >> 21) & 1 != 0 { "!" } else { "" };
                format!("{kind}{byte}{cond} r{rd}, [{base}{offset}]{bang}")
            } else {
                format!("{kind}{byte}{cond} r{rd}, [{base}]{offset}")
            }
        }
        0b110 | 0b111 => {
            if (insn >> 24) & 1 != 0 && (insn >> 25) & 0x7 == 0b111 {
                return format!("svc{cond} #{:#x}", insn & 0x00FF_FFFF);
            }
            let coproc = (insn >> 8) & 0xF;
            if coproc == 10 || coproc == 11 {
                return super::vfp_mnemonic(insn, cond);
            }
            let dir = if (insn >> 20) & 1 != 0 { "mrc" } else { "mcr" };
            format!(
                "{dir}{cond} p{coproc}, {}, r{rd}, c{rn}, c{rm}, {}",
                (insn >> 21) & 0x7,
                (insn >> 5) & 0x7
            )
        }
        _ => {
            if (insn & 0x0FFF_FFF0) == 0x012F_FF10 {
                return format!("bx{cond} r{rm}");
            }
            if (insn & 0x0FFF_FFF0) == 0x012F_FF30 {
                return format!("blx{cond} r{rm}");
            }
            if (insn & 0x0FF0_00F0) == 0x0160_0010 {
                return format!("clz{cond} r{rd}, r{rm}");
            }
            if (insn & 0x0F90_0090) == 0x0000_0090 {
                let kind = if (insn >> 21) & 1 != 0 { "mla" } else { "mul" };
                return format!("{kind}{cond} r{rn}, r{rm}, r{}", (insn >> 8) & 0xF);
            }
            if (insn & 0x0F80_0090) == 0x0080_0090 {
                return format!(
                    "{}{cond} r{rd}, r{rn}, r{rm}",
                    ["umull", "umlal", "smull", "smlal"][((insn >> 21) & 0b11) as usize]
                );
            }
            // MOVW/MOVT, which the ordinary opcode table has no room for.
            if (insn & 0x0FB0_0000) == 0x0300_0000 {
                let wide = if (insn >> 22) & 1 != 0 { "movt" } else { "movw" };
                return format!("{wide}{cond} r{rd}, #{:#x}", ((insn >> 4) & 0xF000) | (insn & 0xFFF));
            }
            let op = (insn >> 21) & 0xF;
            let name = OPS[op as usize];
            let s = if (insn >> 20) & 1 != 0 && !(0x8..=0xB).contains(&op) {
                "s"
            } else {
                ""
            };
            let operand = operand2(insn);
            match op {
                // The comparisons write no register.
                0x8..=0xB => format!("{name}{cond} r{rn}, {operand}"),
                // MOV and MVN have no first operand.
                0xD | 0xF => format!("{name}{s}{cond} r{rd}, {operand}"),
                _ => format!("{name}{s}{cond} r{rd}, r{rn}, {operand}"),
            }
        }
    }
}
