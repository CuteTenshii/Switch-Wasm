//! What a title's RomFS holds, and which file a byte offset falls in:
//! `romfs_ls <path.nsp> <prod.keys> <title.keys> [offset,...]`.
//!
//! `TRACE_IPC`'s `[storage] read offset=…` lines name a byte range and nothing
//! else, so a trace of a title loading its assets says how much it read and
//! never *what*. That is the difference between "loading stopped" and "loading
//! stopped after the last scene descriptor it will ever ask for", and only the
//! second one says where to look next.
//!
//! Nothing is decrypted up front. A retail RomFS is the whole game, but its
//! metadata tables are a few hundred kilobytes at the end of the image, so
//! reading *those* out of the streamed section is enough to name every file in
//! it — which is why this can list a 7 GiB image in under a second.
//!
//! Offsets are given in the same coordinates the trace prints: from the start
//! of the RomFS image, past the NCA's IVFC hash levels.
mod common;

use switch_core::source::ByteSource;

const USAGE: &str = "romfs_ls <path.nsp> <prod.keys> <title.keys> [offset,...]";

/// The RomFS header's declared size. The format carries no magic number, so
/// this doubles as the check that the section decrypted to a RomFS at all.
const HEADER_SIZE: u64 = 0x50;
/// End-of-chain marker for the links between metadata entries.
const INVALID_OFFSET: u32 = 0xFFFF_FFFF;
/// Fixed part of a directory entry, before its name.
const DIR_ENTRY_SIZE: usize = 0x18;
/// Fixed part of a file entry, before its name.
const FILE_ENTRY_SIZE: usize = 0x20;

fn read_u32(data: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(data[at..at + 4].try_into().unwrap_or([0; 4]))
}

fn read_u64(data: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(data[at..at + 8].try_into().unwrap_or([0; 8]))
}

/// One file, in the coordinates the `[storage]` trace speaks: `start` is an
/// offset into the RomFS image, not into the file data region.
struct Entry {
    path: String,
    start: u64,
    size: u64,
}

/// Walk the directory and file metadata tables and collect every file.
///
/// The chains are offsets into the tables and a malformed image can point one
/// back at itself, so the walk is bounded the way [`switch_core::romfs`]
/// bounds its own: every entry is at least its fixed part long, so the table
/// sizes are an upper bound on how many there can be.
fn walk(dirs: &[u8], files: &[u8], data_offset: u64) -> Vec<Entry> {
    let mut budget = 2 * (dirs.len() / DIR_ENTRY_SIZE) + files.len() / FILE_ENTRY_SIZE + 2;
    let mut out = Vec::new();
    let mut pending = vec![(0u32, String::new())];
    while let Some((dir_offset, prefix)) = pending.pop() {
        let Some(dir) = entry(dirs, dir_offset, DIR_ENTRY_SIZE) else {
            continue;
        };
        let mut file_offset = read_u32(dir, 0x0C);
        while file_offset != INVALID_OFFSET && budget > 0 {
            budget -= 1;
            let Some(file) = entry(files, file_offset, FILE_ENTRY_SIZE) else {
                break;
            };
            out.push(Entry {
                path: format!("{prefix}/{}", name(file, FILE_ENTRY_SIZE)),
                start: data_offset + read_u64(file, 0x08),
                size: read_u64(file, 0x10),
            });
            file_offset = read_u32(file, 0x04);
        }
        let mut child_offset = read_u32(dir, 0x08);
        while child_offset != INVALID_OFFSET && budget > 0 {
            budget -= 1;
            let Some(child) = entry(dirs, child_offset, DIR_ENTRY_SIZE) else {
                break;
            };
            pending.push((
                child_offset,
                format!("{prefix}/{}", name(child, DIR_ENTRY_SIZE)),
            ));
            child_offset = read_u32(child, 0x04);
        }
    }
    out
}

/// The bytes of one metadata entry: its fixed part plus the name its own
/// length field sizes.
fn entry(table: &[u8], offset: u32, fixed_size: usize) -> Option<&[u8]> {
    let start = offset as usize;
    let fixed_end = start.checked_add(fixed_size)?;
    if fixed_end > table.len() {
        return None;
    }
    let end = fixed_end.checked_add(read_u32(table, fixed_end - 4) as usize)?;
    table.get(start..end)
}

fn name(entry: &[u8], fixed_size: usize) -> String {
    String::from_utf8_lossy(&entry[fixed_size..]).into_owned()
}

fn main() {
    let container = common::arg(1, USAGE);
    let prod = common::arg(2, USAGE);
    // Argument 3 is `title.keys` unless it is the offset list — a container
    // whose keys are all in `prod.keys` should not have to name a file that
    // does not exist just to reach the fourth argument.
    let third = common::opt_arg(3);
    let is_offsets =
        |s: &String| s.starts_with("0x") || s.starts_with(|c: char| c.is_ascii_digit());
    let title = third.clone().filter(|s| !is_offsets(s));
    let wanted: Vec<u64> = third
        .filter(is_offsets)
        .or_else(|| common::opt_arg(4))
        .map(|list| {
            list.split(',')
                .filter_map(|v| u64::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok())
                .collect()
        })
        .unwrap_or_default();

    let title = common::Title::open_nsp(&container, &prod, title.as_ref());
    let romfs = match title.romfs_source() {
        Some(Ok(romfs)) => romfs,
        Some(Err(e)) => common::usage(&format!("this title's RomFS could not be opened: {e}")),
        None => common::usage("this NCA has no RomFS section"),
    };

    let header = romfs
        .read_vec(0, HEADER_SIZE)
        .expect("read the RomFS header");
    if read_u64(&header, 0) != HEADER_SIZE {
        common::usage("that section does not start with a RomFS header");
    }
    let dirs = romfs
        .read_vec(read_u64(&header, 0x18), read_u64(&header, 0x20))
        .expect("read the directory metadata table");
    let files = romfs
        .read_vec(read_u64(&header, 0x38), read_u64(&header, 0x40))
        .expect("read the file metadata table");
    let data_offset = read_u64(&header, 0x48);
    println!(
        "RomFS: {:#x} bytes, {} directory table + {} file table, data at {data_offset:#x}",
        romfs.len(),
        dirs.len(),
        files.len()
    );

    let mut entries = walk(&dirs, &files, data_offset);
    entries.sort_by_key(|e| e.start);
    if wanted.is_empty() {
        for e in &entries {
            println!("{:#014x} +{:<10x} {}", e.start, e.size, e.path);
        }
        println!("{} files", entries.len());
        return;
    }
    // A read is reported against the file whose extent covers it, and a read
    // that covers none is reported as such: the offsets worth asking about
    // come out of a trace, and one that lands in no file at all is a finding
    // rather than a typo.
    for at in wanted {
        match entries
            .iter()
            .find(|e| at >= e.start && at < e.start + e.size)
        {
            Some(e) => println!(
                "{at:#014x} -> {} +{:#x} (file at {:#x}, {:#x} bytes)",
                e.path,
                at - e.start,
                e.start,
                e.size
            ),
            None => println!("{at:#014x} -> in no file's extent"),
        }
    }
}
