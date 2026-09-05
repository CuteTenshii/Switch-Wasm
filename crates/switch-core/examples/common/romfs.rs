//! The RomFS metadata tables: every file in an image, and where each one is.
//!
//! The tables are a few hundred kilobytes at the end of an image that runs to
//! gigabytes, so reading *those* out of the streamed section names every file
//! without decrypting the rest, which is what lets a tool list a 7 GiB RomFS
//! in under a second.
//!
//! Two tools want it and neither should own it: `romfs_ls` turns a
//! `[storage] read offset=…` line into the file it was for, and
//! `romfs_selftest` needs the real ranges a guest asks for, because a bug in
//! the layers underneath shows up at a file's boundaries long before it shows
//! up at a random offset.
//!
//! Offsets are in the coordinates the trace prints: from the start of the
//! RomFS image, past the NCA's IVFC hash levels.

use switch_core::source::ByteSource;

/// The RomFS header's declared size. The format carries no magic number, so
/// this doubles as the check that the section decrypted to a RomFS at all.
pub const HEADER_SIZE: u64 = 0x50;
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
pub struct Entry {
    pub path: String,
    pub start: u64,
    pub size: u64,
}

/// An image's metadata: its geometry and every file in it, ordered by offset.
pub struct Image {
    /// The decompressed image's length.
    pub len: u64,
    /// Where the file data region begins.
    pub data_offset: u64,
    /// The two metadata tables' sizes, which is what says whether an image
    /// with no files is empty or unreadable.
    pub dir_table_size: u64,
    pub file_table_size: u64,
    pub files: Vec<Entry>,
}

/// Read the tables and walk them.
///
/// Returns the reason rather than panicking: "that section does not start
/// with a RomFS header" is the first thing a wrong key looks like, and a
/// caller wants to say so in its own words.
pub fn read(source: &dyn ByteSource) -> Result<Image, String> {
    let header = source
        .read_vec(0, HEADER_SIZE)
        .map_err(|e| format!("the RomFS header could not be read: {e}"))?;
    if read_u64(&header, 0) != HEADER_SIZE {
        return Err("that section does not start with a RomFS header".into());
    }
    let dirs = source
        .read_vec(read_u64(&header, 0x18), read_u64(&header, 0x20))
        .map_err(|e| format!("the directory metadata table could not be read: {e}"))?;
    let files = source
        .read_vec(read_u64(&header, 0x38), read_u64(&header, 0x40))
        .map_err(|e| format!("the file metadata table could not be read: {e}"))?;
    let data_offset = read_u64(&header, 0x48);
    let mut entries = walk(&dirs, &files, data_offset);
    entries.sort_by_key(|e| e.start);
    Ok(Image {
        len: source.len(),
        data_offset,
        dir_table_size: dirs.len() as u64,
        file_table_size: files.len() as u64,
        files: entries,
    })
}

impl Image {
    /// The file an image offset falls in, if it falls in one at all.
    pub fn file_at(&self, at: u64) -> Option<&Entry> {
        self.files
            .iter()
            .find(|e| at >= e.start && at < e.start + e.size)
    }
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
