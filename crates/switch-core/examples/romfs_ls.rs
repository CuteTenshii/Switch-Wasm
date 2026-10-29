//! List a title's RomFS and map byte offsets to files:
//! `romfs_ls <container> <prod.keys> [title.keys] [offset,...]`.
//!
//! Offsets are relative to the RomFS image, as `TRACE_IPC` prints them.
mod common;

use switch_core::source::ByteSource;

const USAGE: &str = "romfs_ls <container> <prod.keys> [title.keys] [offset,...]";

fn main() {
    let args = common::container_args(USAGE);
    let wanted: Vec<u64> = args
        .rest(0)
        .map(|list| list.split(',').filter_map(common::parse_hex).collect())
        .unwrap_or_default();

    let title = args.open();
    let (source, index) = title.romfs();
    println!("RomFS: {:#x} bytes", source.len());

    if wanted.is_empty() {
        let mut count = 0;
        for (start, end, path) in index.files() {
            println!("{start:#014x} +{:<10x} {path}", end - start);
            count += 1;
        }
        println!("{count} files");
        return;
    }
    for at in wanted {
        match index
            .files()
            .find(|&(start, end, _)| at >= start && at < end)
        {
            Some((start, end, path)) => println!(
                "{at:#014x} -> {path} +{:#x} (file at {start:#x}, {:#x} bytes)",
                at - start,
                end - start
            ),
            None => println!("{at:#014x} -> in no file's extent"),
        }
    }
}
