//! Ad-hoc diagnostic for debugging real-world NCA decryption against a real
//! `prod.keys`/`title.keys` — prints the fields `Nca::parse_with_keys`
//! derives (key index, generation, FS header layout) without needing the
//! actual section key to be present, so a "missing key" case still reports
//! everything else for sanity-checking.
//!
//! Usage: cargo run -p switch-core --example diag_nca -- <path.nca> <prod.keys> [title.keys]
mod common;

use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: diag_nca <path.nca> <prod.keys> [title.keys]");
        std::process::exit(1);
    }
    let nca_path = &args[1];
    let prod_path = &args[2];
    let title_path = args.get(3);

    let raw = fs::read(nca_path).expect("read nca");
    println!(
        "file size: {} bytes ({:.1} KiB)",
        raw.len(),
        raw.len() as f64 / 1024.0
    );

    let keys = common::keys(prod_path, title_path);
    println!(
        "header_key loaded: {}",
        keys.effective_header_key().is_some()
    );

    let nca = match switch_core::nca::Nca::parse_with_keys(&raw, Some(&keys)) {
        Ok(n) => n,
        Err(e) => {
            println!("parse_with_keys FAILED: {}", e);
            return;
        }
    };

    println!(
        "content_type: {:?} (raw {})",
        nca.content_type, nca.content_type_raw
    );
    println!("title_id: {:016x}", nca.title_id);
    println!("program_id: {:016x}", nca.program_id);
    println!("sdk_version: {:08x}", nca.sdk_version);
    println!(
        "crypto_type (existing field, offset 0x21C): {}",
        nca.crypto_type
    );
    println!("key_index (offset 0x207): {}", nca.key_index);
    println!(
        "key_generation_old/new (0x206/0x220): {} / {}",
        nca.key_generation_old, nca.key_generation_new
    );
    println!("rights_id: {}", hex(&nca.rights_id));
    println!("has_rights_id: {}", nca.has_rights_id());
    println!(
        "encrypted_key_area (0x300..0x340): {}",
        hex(&nca.encrypted_key_area)
    );
    println!("exefs_section_index: {:?}", nca.exefs_section_index());

    for (i, sec) in nca.sections.iter().enumerate() {
        println!(
            "section[{}]: media_offset={:#x} media_size={:#x} partition_index={}",
            i, sec.media_offset, sec.media_size, sec.partition_index
        );
        match &nca.fs_headers[i] {
            None => println!("  fs_header: not decrypted (need full header + header_key)"),
            Some(fs) => {
                println!(
                    "  fs_header: version={} partition_type={} fs_type={} encryption_type={} romfs_data_offset={:#x}",
                    fs.version, fs.partition_type, fs.fs_type, fs.encryption_type, fs.romfs_data_offset
                );
                println!(
                    "  hash_table=[{:#x}, {:#x}) data=[{:#x}, {:#x}) master_hash={}",
                    fs.hash_table_offset,
                    fs.hash_table_offset + fs.hash_table_size,
                    fs.data_offset,
                    fs.data_offset + fs.data_size,
                    hex(&fs.master_hash)
                );
                println!(
                    "  generation={:#x} secure_value={:#x}",
                    fs.generation, fs.secure_value
                );
            }
        }
    }

    match nca.section_key(&keys) {
        Ok(k) => println!("section_key resolved: {}", hex(&k)),
        Err(e) => println!("section_key FAILED: {}", e),
    }

    if let Some(idx) = nca.exefs_section_index() {
        match nca.decrypt_pfs0_section(&raw, &keys, idx) {
            Ok(pfs0) => {
                println!("decrypt_pfs0_section OK: {} bytes", pfs0.len());
                match switch_core::nsp::Pfs0::parse(&pfs0) {
                    Ok(p) => {
                        for f in &p.files {
                            println!("  exefs file: {} ({} bytes)", f.name, f.size);
                        }
                    }
                    Err(e) => println!("  Pfs0::parse failed: {}", e),
                }
            }
            Err(e) => println!("decrypt_pfs0_section FAILED: {}", e),
        }
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}
