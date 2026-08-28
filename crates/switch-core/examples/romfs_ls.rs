//! What a title's RomFS holds, and which file a byte offset falls in:
//! `romfs_ls <container> <prod.keys> [title.keys] [offset,...]`.
//!
//! `TRACE_IPC`'s `[storage] read offset=…` lines name a byte range and nothing
//! else, so a trace of a title loading its assets says how much it read and
//! never *what*. That is the difference between "loading stopped" and "loading
//! stopped after the last scene descriptor it will ever ask for", and only the
//! second one says where to look next.
//!
//! Nothing is decrypted up front: the metadata tables are a few hundred
//! kilobytes at the end of the image, and reading those is enough to name
//! every file in it (see [`common::romfs`]).
//!
//! Offsets are given in the same coordinates the trace prints: from the start
//! of the RomFS image, past the NCA's IVFC hash levels.
mod common;

const USAGE: &str = "romfs_ls <container> <prod.keys> [title.keys] [offset,...]";

fn main() {
    let args = common::container_args(USAGE);
    let wanted: Vec<u64> = args
        .rest(0)
        .map(|list| {
            list.split(',')
                .filter_map(|v| u64::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok())
                .collect()
        })
        .unwrap_or_default();

    let title = args.open();
    let (_source, image) = title.romfs(USAGE);
    println!(
        "RomFS: {:#x} bytes, {} directory table + {} file table, data at {:#x}",
        image.len, image.dir_table_size, image.file_table_size, image.data_offset
    );

    if wanted.is_empty() {
        for e in &image.files {
            println!("{:#014x} +{:<10x} {}", e.start, e.size, e.path);
        }
        println!("{} files", image.files.len());
        return;
    }
    // A read is reported against the file whose extent covers it, and a read
    // that covers none is reported as such: the offsets worth asking about
    // come out of a trace, and one that lands in no file at all is a finding
    // rather than a typo.
    for at in wanted {
        match image.file_at(at) {
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
