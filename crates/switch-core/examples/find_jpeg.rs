// Temporary debug tool: scan the sdk module .text for adrp+add references to a
// page (default 0xe06e000, where the deferred-init result slot 0xe06e2ac and
// once-guard 0xe06e2a8 live) and print each site's offset + the following
// instruction, to find who reads vs writes that slot.
use std::env;
use std::fs;
use switch_core::cpu::Cpu;
use switch_core::nsp::Pfs0;

fn main() {
    let args: Vec<String> = env::args().collect();
    let nsp_path = &args[1];
    let prod_path = &args[2];
    let page = u32::from_str_radix(args.get(3).unwrap_or(&"0xe06e000".into()).trim_start_matches("0x"), 16).unwrap();

    let nsp_data = fs::read(nsp_path).expect("read nsp");
    let pfs0 = Pfs0::parse(&nsp_data).expect("parse nsp");
    let prod_text = fs::read_to_string(prod_path).expect("read prod.keys");
    let mut keys = switch_core::keys::keyset_from_prod(&switch_core::keys::parse_keys_file(&prod_text));

    let mut program: Option<&switch_core::nsp::Pfs0File> = None;
    for f in &pfs0.files {
        if !f.name.to_ascii_lowercase().ends_with(".nca") { continue; }
        let s = f.offset as usize;
        let e = s + f.size as usize;
        if e > nsp_data.len() { continue; }
        if let Ok(nca) = switch_core::nca::Nca::parse_with_keys(&nsp_data[s..e], Some(&keys)) {
            if nca.content_type == switch_core::nca::ContentType::Program { program = Some(f); }
        }
    }
    let f = program.expect("no program nca");
    let raw = &nsp_data[f.offset as usize..(f.offset + f.size) as usize];
    let nca = switch_core::nca::Nca::parse_with_keys(raw, Some(&keys)).expect("parse program nca");
    if nca.has_rights_id() && keys.title_key(&nca.rights_id).is_none() {
        if let Ok(tk) = switch_core::ticket::find_and_decrypt_title_key(&nca.rights_id, &pfs0.files, &nsp_data, &keys) {
            keys.title_keys.push((nca.rights_id, tk));
        }
    }
    let exefs = nca.decrypt_pfs0_section(raw, &keys, nca.exefs_section_index().expect("exefs")).expect("exefs");
    let exefs_pfs0 = Pfs0::parse(&exefs).expect("pfs0");
    let sdk = exefs_pfs0.find("sdk").expect("sdk");
    let sdk_bytes = &exefs[sdk.offset as usize..(sdk.offset + sdk.size) as usize];

    let mut cpu = Cpu::new();
    cpu.bootstrap();
    let base = 0x0ce15000u32;
    let _ = switch_core::nso::load_nso(&mut cpu.mem, sdk_bytes, base).expect("load sdk");
    let tsize = u32::from_le_bytes(sdk_bytes[0x18..0x1c].try_into().unwrap());
    let tmem = u32::from_le_bytes(sdk_bytes[0x14..0x18].try_into().unwrap());
    let text_start = base.wrapping_add(tmem);
    let text_end = text_start.wrapping_add(tsize);

    let mut pc = text_start;
    while pc + 4 <= text_end {
        let insn = cpu.mem.fetch(pc).unwrap_or(0);
        if ((insn >> 24) & 0x1F) == 0b10000 && (insn >> 31) & 1 == 1 {
            let rd = insn & 0x1F;
            let immhi = (insn >> 5) & 0x7_FFFF;
            let immlo = (insn >> 29) & 0b11;
            let imm = ((immhi << 2) | immlo) as i32;
            let imm = (imm << 20) >> 20;
            let tgt_page = (pc & !0xFFF).wrapping_add((imm as u32) << 12) & 0xFFFF_F000;
            if tgt_page == page {
                let n = cpu.mem.fetch(pc + 4).unwrap_or(0);
                let mut off = None;
                // add xD, xD, #imm : sf 00 10001 0 imm12 Rn Rd
                if ((n >> 24) & 0x1F) == 0b10001 && (n >> 21) & 1 == 0 {
                    let rn = (n >> 5) & 0x1F;
                    let rd2 = n & 0x1F;
                    if rn == rd && rd2 == rd { off = Some((n >> 10) & 0xFFF); }
                }
                // print the following instruction's opcode class
                let after = cpu.mem.fetch(pc + 8).unwrap_or(0);
                let class = if (after & 0xFFC0_0000) == 0xF900_0000 || (after & 0xFFC0_0000) == 0xF940_0000 {
                    "ldr"
                } else if (after & 0xFFC0_0000) == 0xB900_0000 || (after & 0xFFC0_0000) == 0xB940_0000 {
                    "str/ldr w"
                } else {
                    "-"
                };
                println!("{pc:#010x} rel={:#x} adrp x{rd} off={:?} next_class={class}", pc.wrapping_sub(base), off);
            }
        }
        pc += 4;
    }
}
