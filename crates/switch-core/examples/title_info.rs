//! Print a title's control data — icon, name, publisher and the rest of its
//! NACP — from an `.nsp` or a standalone Control `.nca`. The CLI equivalent of
//! the browser's title card, useful for checking a container without one.
//!
//! Usage: cargo run -p switch-core --example title_info -- <path.nsp|path.nca> <prod.keys> [title.keys] [icon_out.jpg]
mod common;

use switch_core::source::{ByteSource, FileSource};

const USAGE: &str = "title_info <path.nsp|path.nca> <prod.keys> [title.keys] [icon_out.jpg]";

fn main() {
    let args = common::container_args(USAGE);
    let mut keys = args.keys();
    if let Ok(src) = FileSource::open(&args.container) {
        println!("{}: {} bytes", args.container, src.len());
    }

    let control = match common::open_control(&args.container, &mut keys) {
        Ok(control) => control,
        Err(e) => {
            println!("control data unavailable: {e}");
            std::process::exit(1);
        }
    };
    let nacp = &control.nacp;

    println!("title id:       {:016x}", control.title_id);
    println!("name:           {}", control.name);
    println!("publisher:      {}", control.publisher);
    println!("version:        {}", nacp.display_version);
    println!("language shown: {}", control.language);
    println!(
        "localized:      {}",
        nacp.titles
            .iter()
            .map(|t| t.language)
            .collect::<Vec<_>>()
            .join(", ")
    );
    if !nacp.ratings.is_empty() {
        println!(
            "age rating:     {}",
            nacp.ratings
                .iter()
                .map(|r| format!("{} {}", r.organisation, r.age))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    println!("demo:           {}", nacp.is_demo);
    println!("user account:   {}", nacp.startup_user_account.name());
    println!("screenshots:    {}", nacp.screenshot.name());
    println!("video capture:  {}", nacp.video_capture.name());
    println!(
        "save data:      user {} (+{} journal), device {} (+{} journal), bcat {}",
        nacp.user_account_save_data_size,
        nacp.user_account_save_data_journal_size,
        nacp.device_save_data_size,
        nacp.device_save_data_journal_size,
        nacp.bcat_delivery_cache_storage_size
    );
    if nacp.add_on_content_base_id != 0 {
        println!("dlc base id:    {:016x}", nacp.add_on_content_base_id);
    }
    if nacp.save_data_owner_id != 0 {
        println!("save owner id:  {:016x}", nacp.save_data_owner_id);
    }
    if !nacp.application_error_code_category.is_empty() {
        println!("error codes:    {}", nacp.application_error_code_category);
    }
    if !nacp.isbn.is_empty() {
        println!("isbn:           {}", nacp.isbn);
    }
    println!(
        "icon:           {} bytes, {}",
        control.icon.len(),
        control.icon_mime()
    );

    if let Some(path) = args.rest(0) {
        if control.icon.is_empty() {
            println!("no icon to write");
        } else {
            std::fs::write(path, &control.icon).expect("write icon");
            println!("wrote {path}");
        }
    }
}
