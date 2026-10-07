//! List a title's RomFS and map byte offsets to files:
//! `romfs_ls <container> <prod.keys> [title.keys] [offset,...]`.
//!
//! Offsets are relative to the RomFS image, as `TRACE_IPC` prints them.
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
