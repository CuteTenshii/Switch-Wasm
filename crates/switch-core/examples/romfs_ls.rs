//! List a title's RomFS and map byte offsets to files:
//! `romfs_ls <container> --keys <prod.keys> [--offsets <offset,...>]`.
mod common;

use clap::Parser;
use switch_core::source::ByteSource;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    target: common::Target,
    /// Hexadecimal offsets to map to files, relative to the RomFS image as the `ipc` trace prints
    /// them.
    #[arg(long, value_delimiter = ',', value_parser = common::hex64, value_name = "OFFSET")]
    offsets: Vec<u64>,
}

fn main() {
    let cli = Cli::parse();
    let wanted = cli.offsets;

    let title = cli.target.open();
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
