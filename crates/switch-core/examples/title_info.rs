//! Print a title's control data — icon, name, publisher and the rest of its
//! NACP — from an `.nsp` or a standalone Control `.nca`. The CLI equivalent of
//! the browser's title card, useful for checking a container without one.
//!
//! Usage: cargo run -p switch-core --example title_info -- <path.nsp|path.nca> <prod.keys> [title.keys] [icon_out.jpg]
mod common;


use std::cell::RefCell;
use std::env;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use switch_core::control::Control;
use switch_core::nsp::Pfs0;
use switch_core::source::ByteSource;
use switch_core::Error;

/// A container read straight off disk, so a multi-gigabyte `.nsp` costs a few
/// seeks rather than its own size in memory — the native counterpart of the
/// browser's `host_read`.
#[derive(Debug)]
struct FileSource {
    file: RefCell<File>,
    len: u64,
}

impl FileSource {
    fn open(path: &str) -> std::io::Result<FileSource> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        Ok(FileSource { file: RefCell::new(file), len })
    }
}

impl ByteSource for FileSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<usize, Error> {
        if offset >= self.len {
            return Ok(0);
        }
        let mut file = self.file.borrow_mut();
        file.seek(SeekFrom::Start(offset)).map_err(|e| Error::Io(e.to_string()))?;
        let want = ((out.len() as u64).min(self.len - offset)) as usize;
        file.read_exact(&mut out[..want]).map_err(|e| Error::Io(e.to_string()))?;
        Ok(want)
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: title_info <path.nsp|path.nca> <prod.keys> [title.keys] [icon_out.jpg]");
        std::process::exit(1);
    }
    let container_path = &args[1];
    let prod_path = &args[2];
    let title_path = args.get(3).filter(|s| s.ends_with(".keys"));
    let icon_path = args.iter().skip(3).find(|s| !s.ends_with(".keys"));

    let mut keys = common::keys(prod_path, title_path);

    let src = FileSource::open(container_path).expect("open container");
    println!("{}: {} bytes", container_path, src.len());

    // A standalone `.nca` is its own container; anything else is a PFS0 whose
    // Control NCA has to be found first, and whose bundled ticket may be the
    // only place that NCA's title key exists.
    let control = if container_path.to_ascii_lowercase().ends_with(".nca") {
        Control::from_source(&src, &keys)
    } else {
        let pfs0 = Pfs0::read_from(&src).expect("parse container");
        println!("{} file(s) in the container", pfs0.files.len());
        let (index, nca) = switch_core::control::find_control_nca(&pfs0.files, &src, &keys)
            .expect("no Control NCA in this container (is prod.keys right?)");
        println!("Control NCA: {}", pfs0.files[index].name);
        if nca.has_rights_id() && keys.resolved_title_key(&nca.rights_id).is_none() {
            match switch_core::ticket::find_and_decrypt_title_key_from(
                &nca.rights_id,
                &pfs0.files,
                &src,
                &keys,
            ) {
                Ok(title_key) => keys.add_resolved_title_key(nca.rights_id, title_key),
                Err(e) => println!("ticket resolution failed: {}", e),
            }
        }
        pfs0.file_source(&src, index)
            .and_then(|window| Control::from_source(window, &keys))
    };

    let control = match control {
        Ok(control) => control,
        Err(e) => {
            println!("control data unavailable: {}", e);
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
        nacp.titles.iter().map(|t| t.language).collect::<Vec<_>>().join(", ")
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
    println!("icon:           {} bytes, {}", control.icon.len(), control.icon_mime());

    if let Some(path) = icon_path {
        if control.icon.is_empty() {
            println!("no icon to write");
        } else {
            std::fs::write(path, &control.icon).expect("write icon");
            println!("wrote {}", path);
        }
    }
}
