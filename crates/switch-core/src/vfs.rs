//! The emulated SD card.
//!
//! A path-addressed in-memory filesystem behind the `fsp-srv` service, so the
//! guest's `opendir`/`readdir`/`open`/`read` see a real, consistent tree
//! instead of a fixed reply. Hosts populate it — the browser frontend adds the
//! files the user drops in, and tests build small trees inline.

use std::collections::BTreeMap;

/// `FsDirEntryType`.
pub const ENTRY_TYPE_DIR: u8 = 0;
pub const ENTRY_TYPE_FILE: u8 = 1;

/// One entry as `fsDirRead` reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirEntry {
    pub name: String,
    pub kind: u8,
    pub size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Node {
    Dir,
    File(Vec<u8>),
}

#[derive(Debug, Default)]
pub struct Vfs {
    /// Normalized absolute paths (`/`, `/switch`, `/switch/app.nro`) to nodes.
    nodes: BTreeMap<String, Node>,
}

impl Vfs {
    /// An SD card with just the root and the `/switch` directory homebrew
    /// menus expect to exist.
    pub fn new() -> Vfs {
        let mut vfs = Vfs { nodes: BTreeMap::new() };
        vfs.nodes.insert("/".to_owned(), Node::Dir);
        vfs.create_dir("/switch");
        vfs
    }

    /// Normalize a guest path: strip any `device:` prefix and trailing
    /// slashes, and guarantee a leading slash.
    pub fn normalize(path: &str) -> String {
        let without_device = match path.split_once(":/") {
            Some((_, rest)) => rest,
            None => path,
        };
        let trimmed = without_device.trim_matches('/');
        if trimmed.is_empty() { "/".to_owned() } else { format!("/{}", trimmed) }
    }

    fn parent_of(path: &str) -> Option<String> {
        let path = path.trim_end_matches('/');
        match path.rfind('/') {
            Some(0) => Some("/".to_owned()),
            Some(index) => Some(path[..index].to_owned()),
            None => None,
        }
    }

    /// Create a directory and every missing parent.
    pub fn create_dir(&mut self, path: &str) {
        let path = Self::normalize(path);
        let mut current = String::from("/");
        self.nodes.insert(current.clone(), Node::Dir);
        for part in path.split('/').filter(|p| !p.is_empty()) {
            if current == "/" {
                current = format!("/{}", part);
            } else {
                current = format!("{}/{}", current, part);
            }
            self.nodes.entry(current.clone()).or_insert(Node::Dir);
        }
    }

    /// Add (or replace) a file, creating its parent directories.
    pub fn write_file(&mut self, path: &str, data: Vec<u8>) {
        let path = Self::normalize(path);
        if let Some(parent) = Self::parent_of(&path) {
            self.create_dir(&parent);
        }
        self.nodes.insert(path, Node::File(data));
    }

    pub fn remove(&mut self, path: &str) -> bool {
        self.nodes.remove(&Self::normalize(path)).is_some()
    }

    /// `FsDirEntryType` of a path, or `None` when it does not exist.
    pub fn entry_type(&self, path: &str) -> Option<u8> {
        match self.nodes.get(&Self::normalize(path))? {
            Node::Dir => Some(ENTRY_TYPE_DIR),
            Node::File(_) => Some(ENTRY_TYPE_FILE),
        }
    }

    pub fn file(&self, path: &str) -> Option<&[u8]> {
        match self.nodes.get(&Self::normalize(path))? {
            Node::File(data) => Some(data),
            Node::Dir => None,
        }
    }

    pub fn size(&self, path: &str) -> Option<u64> {
        self.file(path).map(|data| data.len() as u64)
    }

    /// Immediate children of a directory, or `None` when it is not one.
    pub fn read_dir(&self, path: &str) -> Option<Vec<DirEntry>> {
        let dir = Self::normalize(path);
        if !matches!(self.nodes.get(&dir), Some(Node::Dir)) {
            return None;
        }
        let prefix = if dir == "/" { String::from("/") } else { format!("{}/", dir) };
        let mut out = Vec::new();
        for (candidate, node) in &self.nodes {
            if candidate == &dir || !candidate.starts_with(&prefix) {
                continue;
            }
            let name = &candidate[prefix.len()..];
            if name.is_empty() || name.contains('/') {
                continue; // a deeper descendant, not an immediate child
            }
            out.push(DirEntry {
                name: name.to_owned(),
                kind: match node {
                    Node::Dir => ENTRY_TYPE_DIR,
                    Node::File(_) => ENTRY_TYPE_FILE,
                },
                size: match node {
                    Node::Dir => 0,
                    Node::File(data) => data.len() as u64,
                },
            });
        }
        Some(out)
    }

    /// Read up to `buf.len()` bytes of a file starting at `offset`. Returns
    /// how many bytes were read, or `None` when the path is not a file.
    pub fn read(&self, path: &str, offset: u64, buf: &mut [u8]) -> Option<usize> {
        let data = self.file(path)?;
        let start = (offset as usize).min(data.len());
        let n = buf.len().min(data.len() - start);
        buf[..n].copy_from_slice(&data[start..start + n]);
        Some(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_strips_device_and_slashes() {
        assert_eq!(Vfs::normalize("sdmc:/switch/"), "/switch");
        assert_eq!(Vfs::normalize("/switch/app.nro"), "/switch/app.nro");
        assert_eq!(Vfs::normalize("sdmc:/"), "/");
        assert_eq!(Vfs::normalize(""), "/");
        assert_eq!(Vfs::normalize("switch"), "/switch");
    }

    #[test]
    fn new_card_has_a_root_and_a_switch_directory() {
        let vfs = Vfs::new();
        assert_eq!(vfs.entry_type("/"), Some(ENTRY_TYPE_DIR));
        assert_eq!(vfs.entry_type("sdmc:/switch"), Some(ENTRY_TYPE_DIR));
        assert_eq!(vfs.entry_type("/nope"), None);
    }

    #[test]
    fn read_dir_lists_only_immediate_children() {
        let mut vfs = Vfs::new();
        vfs.write_file("/switch/app.nro", vec![1, 2, 3]);
        vfs.write_file("/switch/tools/deep.nro", vec![0; 10]);
        vfs.write_file("/top.txt", b"hi".to_vec());

        let mut root = vfs.read_dir("/").unwrap();
        root.sort_by(|a, b| a.name.cmp(&b.name));
        assert_eq!(
            root,
            vec![
                DirEntry { name: "switch".into(), kind: ENTRY_TYPE_DIR, size: 0 },
                DirEntry { name: "top.txt".into(), kind: ENTRY_TYPE_FILE, size: 2 },
            ]
        );

        let mut switch = vfs.read_dir("/switch").unwrap();
        switch.sort_by(|a, b| a.name.cmp(&b.name));
        assert_eq!(switch[0], DirEntry { name: "app.nro".into(), kind: ENTRY_TYPE_FILE, size: 3 });
        assert_eq!(switch[1], DirEntry { name: "tools".into(), kind: ENTRY_TYPE_DIR, size: 0 });
    }

    #[test]
    fn read_dir_on_a_file_or_missing_path_is_none() {
        let mut vfs = Vfs::new();
        vfs.write_file("/a.txt", b"x".to_vec());
        assert!(vfs.read_dir("/a.txt").is_none());
        assert!(vfs.read_dir("/missing").is_none());
    }

    #[test]
    fn reading_a_file_honours_offset_and_length() {
        let mut vfs = Vfs::new();
        vfs.write_file("/data.bin", (0..16u8).collect());
        let mut buf = [0u8; 4];
        assert_eq!(vfs.read("/data.bin", 4, &mut buf), Some(4));
        assert_eq!(buf, [4, 5, 6, 7]);
        // A read past the end returns a short count rather than failing.
        assert_eq!(vfs.read("/data.bin", 14, &mut buf), Some(2));
        assert_eq!(vfs.read("/data.bin", 99, &mut buf), Some(0));
        assert_eq!(vfs.read("/missing", 0, &mut buf), None);
        assert_eq!(vfs.size("/data.bin"), Some(16));
    }

    #[test]
    fn writing_a_file_creates_its_parents() {
        let mut vfs = Vfs::new();
        vfs.write_file("/a/b/c.txt", b"z".to_vec());
        assert_eq!(vfs.entry_type("/a"), Some(ENTRY_TYPE_DIR));
        assert_eq!(vfs.entry_type("/a/b"), Some(ENTRY_TYPE_DIR));
        assert_eq!(vfs.entry_type("/a/b/c.txt"), Some(ENTRY_TYPE_FILE));
    }

    #[test]
    fn remove_deletes_an_entry() {
        let mut vfs = Vfs::new();
        vfs.write_file("/a.txt", b"x".to_vec());
        assert!(vfs.remove("/a.txt"));
        assert!(!vfs.remove("/a.txt"));
        assert_eq!(vfs.entry_type("/a.txt"), None);
    }
}
