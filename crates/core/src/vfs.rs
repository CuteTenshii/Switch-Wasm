//! The emulated SD card and save data: a path-addressed filesystem behind `fsp-srv`, holding
//! each file in memory or in a [`HostFile`] the host keeps.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::source::ByteSource;

/// `FsDirEntryType`.
pub const ENTRY_TYPE_DIR: u8 = 0;
pub const ENTRY_TYPE_FILE: u8 = 1;

/// A file the host keeps and reads and writes in place, so its bytes need not live here.
/// Errors come back to the guest as a failed request.
pub trait HostFile: ByteSource {
    /// Write all of `data` at `offset`, growing the file and zero-filling any gap.
    fn write_at(&self, offset: u64, data: &[u8]) -> Result<(), crate::Error>;
    /// Truncate or zero-extend to `len`.
    fn set_len(&self, len: u64) -> Result<(), crate::Error>;
}

/// One entry as `fsDirRead` reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirEntry {
    pub name: String,
    pub kind: u8,
    pub size: u64,
}

#[derive(Debug, Clone)]
enum Node {
    Dir,
    File(Vec<u8>),
    /// Read-only, read from its source on demand.
    Mounted(Rc<dyn ByteSource>),
    /// Read and written through the host.
    Host(Rc<dyn HostFile>),
}

impl Node {
    fn kind(&self) -> u8 {
        match self {
            Node::Dir => ENTRY_TYPE_DIR,
            Node::File(_) | Node::Mounted(_) | Node::Host(_) => ENTRY_TYPE_FILE,
        }
    }

    fn size(&self) -> u64 {
        match self {
            Node::Dir => 0,
            Node::File(data) => data.len() as u64,
            Node::Mounted(source) => source.len(),
            Node::Host(file) => file.len(),
        }
    }
}

#[derive(Debug, Default)]
pub struct Vfs {
    /// Normalized absolute paths to nodes.
    nodes: BTreeMap<String, Node>,
    /// Paths the guest changed since [`Vfs::take_changes`]; host writes are not recorded.
    changed: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Change {
    pub path: String,
    /// `ENTRY_TYPE_FILE`, `ENTRY_TYPE_DIR`, or `None` when deleted.
    pub kind: Option<u8>,
    pub size: u64,
}

impl Vfs {
    /// Just the root, as save data starts.
    pub fn empty() -> Vfs {
        let mut vfs = Vfs {
            nodes: BTreeMap::new(),
            changed: BTreeSet::new(),
        };
        vfs.nodes.insert("/".to_owned(), Node::Dir);
        vfs
    }

    /// The root and the `/switch` directory homebrew menus expect.
    pub fn new() -> Vfs {
        let mut vfs = Vfs {
            nodes: BTreeMap::new(),
            changed: BTreeSet::new(),
        };
        vfs.nodes.insert("/".to_owned(), Node::Dir);
        vfs.create_dir("/switch");
        vfs
    }

    /// Normalize a guest path: strip a `device:` prefix, resolve `.` and `..`,
    /// collapse slashes, leading slash and no trailing one.
    pub fn normalize(path: &str) -> String {
        let without_device = match path.split_once(":/") {
            Some((_, rest)) => rest,
            None => path,
        };
        let mut parts: Vec<&str> = Vec::new();
        for part in without_device.split('/') {
            match part {
                // Exact match: `.get` is an ordinary name.
                "" | "." => {}
                // `..` stops at the root.
                ".." => {
                    parts.pop();
                }
                name => parts.push(name),
            }
        }
        if parts.is_empty() {
            "/".to_owned()
        } else {
            format!("/{}", parts.join("/"))
        }
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

    /// Add or replace a file as the host, creating its parents; not recorded as a change.
    pub fn write_file(&mut self, path: &str, data: Vec<u8>) {
        let path = Self::normalize(path);
        if let Some(parent) = Self::parent_of(&path) {
            self.create_dir(&parent);
        }
        self.nodes.insert(path, Node::File(data));
    }

    /// The guest's `CreateFile`: fails if anything is at `path`, never truncates.
    pub fn create_file(&mut self, path: &str, size: u64) -> bool {
        let path = Self::normalize(path);
        if self.nodes.contains_key(&path) {
            return false;
        }
        if let Some(parent) = Self::parent_of(&path) {
            self.guest_create_dir(&parent);
        }
        self.nodes
            .insert(path.clone(), Node::File(vec![0; size as usize]));
        self.changed.insert(path);
        true
    }

    /// [`Vfs::write_file`] recorded as a change, for services persisting state in a save.
    pub fn guest_write_file(&mut self, path: &str, data: Vec<u8>) {
        let path = Self::normalize(path);
        if let Some(parent) = Self::parent_of(&path) {
            self.guest_create_dir(&parent);
        }
        self.nodes.insert(path.clone(), Node::File(data));
        self.changed.insert(path);
    }

    /// [`Vfs::create_dir`], recording every directory it made.
    pub fn guest_create_dir(&mut self, path: &str) {
        let path = Self::normalize(path);
        let mut current = String::from("/");
        for part in path.split('/').filter(|p| !p.is_empty()) {
            if current == "/" {
                current = format!("/{}", part);
            } else {
                current = format!("{}/{}", current, part);
            }
            if !self.nodes.contains_key(&current) {
                self.changed.insert(current.clone());
            }
        }
        self.create_dir(&path);
    }

    /// Drain the guest's changed paths, each with its current state.
    pub fn take_changes(&mut self) -> Vec<Change> {
        std::mem::take(&mut self.changed)
            .into_iter()
            .map(|path| {
                let kind = self.entry_type(&path);
                let size = if kind == Some(ENTRY_TYPE_FILE) {
                    self.size(&path).unwrap_or(0)
                } else {
                    0
                };
                Change { path, kind, size }
            })
            .collect()
    }

    pub fn pending_changes(&self) -> usize {
        self.changed.len()
    }

    /// Write at `offset`, growing and zero-filling. `None` when not a writable file, or when
    /// the host could not write it.
    pub fn write(&mut self, path: &str, offset: u64, data: &[u8]) -> Option<usize> {
        let contents = match self.nodes.get_mut(&Self::normalize(path)) {
            Some(Node::File(contents)) => contents,
            Some(Node::Host(file)) => {
                file.write_at(offset, data).ok()?;
                self.changed.insert(Self::normalize(path));
                return Some(data.len());
            }
            _ => return None,
        };
        let start = offset as usize;
        let end = start.checked_add(data.len())?;
        if end > contents.len() {
            contents.resize(end, 0);
        }
        contents[start..end].copy_from_slice(data);
        self.changed.insert(Self::normalize(path));
        Some(data.len())
    }

    /// Resize a file, zero-filling growth. Returns whether it was a writable file the
    /// resize took.
    pub fn set_size(&mut self, path: &str, size: u64) -> bool {
        let resized = match self.nodes.get_mut(&Self::normalize(path)) {
            Some(Node::File(contents)) => {
                contents.resize(size as usize, 0);
                true
            }
            Some(Node::Host(file)) => file.set_len(size).is_ok(),
            _ => false,
        };
        if resized {
            self.changed.insert(Self::normalize(path));
        }
        resized
    }

    /// Remove an entry, and a directory's contents with it, each recorded as a change.
    pub fn remove(&mut self, path: &str) -> bool {
        let path = Self::normalize(path);
        if self.nodes.remove(&path).is_none() {
            return false;
        }
        if path != "/" {
            let prefix = format!("{path}/");
            let under: Vec<String> = self
                .nodes
                .range(prefix.clone()..)
                .take_while(|(candidate, _)| candidate.starts_with(&prefix))
                .map(|(candidate, _)| candidate.clone())
                .collect();
            for candidate in under {
                self.nodes.remove(&candidate);
                self.changed.insert(candidate);
            }
        }
        self.changed.insert(path);
        true
    }

    /// `FsDirEntryType` of a path, or `None` when it does not exist.
    pub fn entry_type(&self, path: &str) -> Option<u8> {
        Some(self.nodes.get(&Self::normalize(path))?.kind())
    }

    /// The bytes of a file held in memory; `None` for one the host or a source keeps.
    pub fn file(&self, path: &str) -> Option<&[u8]> {
        match self.nodes.get(&Self::normalize(path))? {
            Node::File(data) => Some(data),
            Node::Dir | Node::Mounted(_) | Node::Host(_) => None,
        }
    }

    pub fn size(&self, path: &str) -> Option<u64> {
        match self.nodes.get(&Self::normalize(path))? {
            Node::Dir => None,
            node => Some(node.size()),
        }
    }

    /// Every path but the root, with its kind and size, in path order.
    pub fn entries(&self) -> Vec<(String, u8, u64)> {
        self.nodes
            .iter()
            .filter(|(path, _)| path.as_str() != "/")
            .map(|(path, node)| (path.clone(), node.kind(), node.size()))
            .collect()
    }

    /// Total size of every file.
    pub fn bytes(&self) -> u64 {
        self.nodes.values().map(Node::size).sum()
    }

    /// Put a read-only file backed by `source` at `path`, with its parents. Not a guest change.
    pub fn mount_source(&mut self, path: &str, source: Rc<dyn ByteSource>) {
        let path = Self::normalize(path);
        if let Some(parent) = Self::parent_of(&path) {
            self.create_dir(&parent);
        }
        self.nodes.insert(path, Node::Mounted(source));
    }

    /// Put a file the host keeps at `path`, with its parents, replacing whatever was there.
    /// Not a guest change: the host already has its contents.
    pub fn attach_host(&mut self, path: &str, file: Rc<dyn HostFile>) {
        let path = Self::normalize(path);
        if let Some(parent) = Self::parent_of(&path) {
            self.create_dir(&parent);
        }
        self.nodes.insert(path, Node::Host(file));
    }

    /// Immediate children of a directory, or `None` when it is not one.
    pub fn read_dir(&self, path: &str) -> Option<Vec<DirEntry>> {
        let dir = Self::normalize(path);
        if !matches!(self.nodes.get(&dir), Some(Node::Dir)) {
            return None;
        }
        let prefix = if dir == "/" {
            String::from("/")
        } else {
            format!("{}/", dir)
        };
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
                kind: node.kind(),
                size: node.size(),
            });
        }
        Some(out)
    }

    /// Read from `offset`. `None` when not a file.
    pub fn read(&self, path: &str, offset: u64, buf: &mut [u8]) -> Option<usize> {
        match self.nodes.get(&Self::normalize(path))? {
            Node::Mounted(source) => return Some(source.read_at(offset, buf).unwrap_or(0)),
            Node::Host(file) => return Some(file.read_at(offset, buf).unwrap_or(0)),
            Node::Dir | Node::File(_) => {}
        }
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
    fn a_mounted_source_lists_and_reads_but_refuses_writes() {
        let mut vfs = Vfs::empty();
        vfs.mount_source(
            "/Contents/registered/a.nca",
            Rc::new(crate::source::MemSource(b"abcdef".to_vec())),
        );
        assert_eq!(vfs.entry_type("/Contents/registered"), Some(ENTRY_TYPE_DIR));
        assert_eq!(vfs.size("/Contents/registered/a.nca"), Some(6));
        let listed = vfs.read_dir("/Contents/registered").unwrap();
        assert_eq!((listed[0].name.as_str(), listed[0].size), ("a.nca", 6));
        let mut buf = [0u8; 4];
        assert_eq!(vfs.read("/Contents/registered/a.nca", 3, &mut buf), Some(3));
        assert_eq!(&buf[..3], b"def");
        assert_eq!(vfs.write("/Contents/registered/a.nca", 0, b"x"), None);
        assert!(vfs.pending_changes() == 0);
    }

    #[test]
    fn normalize_strips_the_device_and_resolves_dot_components() {
        for (path, want) in [
            ("", "/"),
            ("sdmc:/", "/"),
            ("switch", "/switch"),
            ("sdmc:/switch/", "/switch"),
            ("/switch/app.nro", "/switch/app.nro"),
            ("sdmc:/switch/.", "/switch"),
            ("sdmc:/switch/./.get/packages", "/switch/.get/packages"),
            ("/switch//a///b/", "/switch/a/b"),
            ("/switch/a/../b", "/switch/b"),
            ("/.get", "/.get"),
            ("/...", "/..."),
            ("/../../etc", "/etc"),
            ("/switch/../..", "/"),
        ] {
            assert_eq!(Vfs::normalize(path), want, "{path:?}");
        }
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
                DirEntry {
                    name: "switch".into(),
                    kind: ENTRY_TYPE_DIR,
                    size: 0
                },
                DirEntry {
                    name: "top.txt".into(),
                    kind: ENTRY_TYPE_FILE,
                    size: 2
                },
            ]
        );

        let mut switch = vfs.read_dir("/switch").unwrap();
        switch.sort_by(|a, b| a.name.cmp(&b.name));
        assert_eq!(
            switch[0],
            DirEntry {
                name: "app.nro".into(),
                kind: ENTRY_TYPE_FILE,
                size: 3
            }
        );
        assert_eq!(
            switch[1],
            DirEntry {
                name: "tools".into(),
                kind: ENTRY_TYPE_DIR,
                size: 0
            }
        );
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
    fn create_file_refuses_to_truncate_what_is_already_there() {
        // `fsdev` reopens a file by calling `CreateFile` and expecting failure.
        let mut vfs = Vfs::new();
        assert!(vfs.create_file("/switch/cfg.json", 0));
        assert_eq!(vfs.size("/switch/cfg.json"), Some(0));
        vfs.write("/switch/cfg.json", 0, b"{}").unwrap();

        assert!(!vfs.create_file("/switch/cfg.json", 0), "already exists");
        assert_eq!(vfs.file("/switch/cfg.json"), Some(&b"{}"[..]));
        assert!(!vfs.create_file("/switch", 0));

        assert!(vfs.create_file("/switch/blank.bin", 4));
        assert_eq!(vfs.file("/switch/blank.bin"), Some(&[0u8; 4][..]));
    }

    #[test]
    fn writing_past_the_end_grows_the_file() {
        let mut vfs = Vfs::new();
        vfs.write_file("/data.bin", vec![1, 2, 3, 4]);
        assert_eq!(vfs.write("/data.bin", 1, &[9, 9]), Some(2));
        assert_eq!(vfs.file("/data.bin"), Some(&[1, 9, 9, 4][..]));
        assert_eq!(vfs.write("/data.bin", 4, &[5, 6]), Some(2));
        assert_eq!(vfs.file("/data.bin"), Some(&[1, 9, 9, 4, 5, 6][..]));
        // left holding whatever the allocation had.
        assert_eq!(vfs.write("/data.bin", 8, &[7]), Some(1));
        assert_eq!(
            vfs.file("/data.bin"),
            Some(&[1, 9, 9, 4, 5, 6, 0, 0, 7][..])
        );
        assert_eq!(vfs.write("/switch", 0, &[1]), None);
        assert_eq!(vfs.write("/nope", 0, &[1]), None);
    }

    #[test]
    fn set_size_truncates_and_extends() {
        let mut vfs = Vfs::new();
        vfs.write_file("/data.bin", (0..8u8).collect());
        assert!(vfs.set_size("/data.bin", 3));
        assert_eq!(vfs.file("/data.bin"), Some(&[0, 1, 2][..]));
        assert!(vfs.set_size("/data.bin", 5));
        assert_eq!(vfs.file("/data.bin"), Some(&[0, 1, 2, 0, 0][..]));
        assert!(!vfs.set_size("/switch", 0), "not a file");
        assert!(!vfs.set_size("/nope", 0));
    }

    #[test]
    fn only_guest_writes_are_reported_as_changes() {
        let mut vfs = Vfs::new();
        // Host restores are not changes.
        vfs.write_file("/switch/restored.txt", b"x".to_vec());
        vfs.create_dir("/switch/olddir");
        assert_eq!(vfs.pending_changes(), 0);

        assert!(vfs.create_file("/switch/cfg.json", 0));
        vfs.write("/switch/cfg.json", 0, b"{}").unwrap();
        vfs.guest_create_dir("/switch/app/data");
        vfs.set_size("/switch/restored.txt", 4);
        vfs.remove("/switch/olddir");

        let changes = vfs.take_changes();
        let by_path: Vec<(&str, Option<u8>, u64)> = changes
            .iter()
            .map(|c| (c.path.as_str(), c.kind, c.size))
            .collect();
        assert_eq!(
            by_path,
            vec![
                ("/switch/app", Some(ENTRY_TYPE_DIR), 0),
                ("/switch/app/data", Some(ENTRY_TYPE_DIR), 0),
                ("/switch/cfg.json", Some(ENTRY_TYPE_FILE), 2),
                ("/switch/olddir", None, 0),
                ("/switch/restored.txt", Some(ENTRY_TYPE_FILE), 4),
            ]
        );

        assert_eq!(vfs.pending_changes(), 0);
        assert!(vfs.take_changes().is_empty());

        // A path written twice is reported once, in its final state.
        vfs.write("/switch/cfg.json", 0, b"ab").unwrap();
        vfs.write("/switch/cfg.json", 2, b"cd").unwrap();
        let changes = vfs.take_changes();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].size, 4);

        // Created then deleted is reported as deleted.
        vfs.create_file("/switch/tmp", 0);
        vfs.remove("/switch/tmp");
        let changes = vfs.take_changes();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].path, "/switch/tmp");
        assert_eq!(changes[0].kind, None);
    }

    /// A host file kept in memory; `broken` refuses every write.
    #[derive(Debug, Default)]
    struct MemHost {
        data: std::cell::RefCell<Vec<u8>>,
        broken: bool,
    }

    impl ByteSource for MemHost {
        fn len(&self) -> u64 {
            self.data.borrow().len() as u64
        }

        fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<usize, crate::Error> {
            let data = self.data.borrow();
            let start = (offset as usize).min(data.len());
            let n = out.len().min(data.len() - start);
            out[..n].copy_from_slice(&data[start..start + n]);
            Ok(n)
        }
    }

    impl HostFile for MemHost {
        fn write_at(&self, offset: u64, data: &[u8]) -> Result<(), crate::Error> {
            if self.broken {
                return Err(crate::Error::Io("refused".into()));
            }
            let mut contents = self.data.borrow_mut();
            let end = offset as usize + data.len();
            if end > contents.len() {
                contents.resize(end, 0);
            }
            contents[offset as usize..end].copy_from_slice(data);
            Ok(())
        }

        fn set_len(&self, len: u64) -> Result<(), crate::Error> {
            if self.broken {
                return Err(crate::Error::Io("refused".into()));
            }
            self.data.borrow_mut().resize(len as usize, 0);
            Ok(())
        }
    }

    #[test]
    fn a_host_file_is_read_and_written_in_place_and_attaching_it_is_no_change() {
        let mut vfs = Vfs::new();
        let host = Rc::new(MemHost {
            data: std::cell::RefCell::new(b"abcdef".to_vec()),
            broken: false,
        });
        vfs.attach_host("/save/data.bin", host.clone());
        assert_eq!(vfs.pending_changes(), 0);
        assert_eq!(vfs.entry_type("/save"), Some(ENTRY_TYPE_DIR));
        assert_eq!(vfs.entry_type("/save/data.bin"), Some(ENTRY_TYPE_FILE));
        assert_eq!(vfs.size("/save/data.bin"), Some(6));
        assert_eq!(vfs.file("/save/data.bin"), None, "not held in memory");

        let mut buf = [0u8; 4];
        assert_eq!(vfs.read("/save/data.bin", 4, &mut buf), Some(2));
        assert_eq!(&buf[..2], b"ef");

        assert_eq!(vfs.write("/save/data.bin", 8, b"xy"), Some(2));
        assert_eq!(&*host.data.borrow(), b"abcdef\0\0xy");
        assert!(vfs.set_size("/save/data.bin", 3));
        assert_eq!(&*host.data.borrow(), b"abc");

        let listed = vfs.read_dir("/save").unwrap();
        assert_eq!((listed[0].name.as_str(), listed[0].size), ("data.bin", 3));
        let changes = vfs.take_changes();
        assert_eq!(changes.len(), 1);
        assert_eq!(
            (changes[0].path.as_str(), changes[0].kind, changes[0].size),
            ("/save/data.bin", Some(ENTRY_TYPE_FILE), 3)
        );
    }

    #[test]
    fn a_host_file_that_refuses_a_write_reports_it_and_records_nothing() {
        let mut vfs = Vfs::new();
        vfs.attach_host(
            "/a.bin",
            Rc::new(MemHost {
                data: std::cell::RefCell::new(vec![1, 2]),
                broken: true,
            }),
        );
        assert_eq!(vfs.write("/a.bin", 0, b"z"), None);
        assert!(!vfs.set_size("/a.bin", 0));
        assert_eq!(vfs.pending_changes(), 0);
        assert_eq!(vfs.size("/a.bin"), Some(2));
    }

    #[test]
    fn attaching_replaces_a_file_held_in_memory() {
        let mut vfs = Vfs::new();
        assert!(vfs.create_file("/n.txt", 0));
        vfs.write("/n.txt", 0, b"hello").unwrap();
        vfs.take_changes();
        vfs.attach_host("/n.txt", Rc::new(MemHost::default()));
        assert_eq!(vfs.size("/n.txt"), Some(0));
        assert_eq!(vfs.pending_changes(), 0);
    }

    #[test]
    fn entries_and_bytes_cover_every_kind_of_file() {
        let mut vfs = Vfs::empty();
        vfs.write_file("/a/mem.bin", vec![0; 5]);
        vfs.attach_host(
            "/a/host.bin",
            Rc::new(MemHost {
                data: std::cell::RefCell::new(vec![0; 7]),
                broken: false,
            }),
        );
        vfs.mount_source("/b/src.bin", Rc::new(crate::source::MemSource(vec![0; 3])));
        assert_eq!(
            vfs.entries(),
            vec![
                ("/a".to_owned(), ENTRY_TYPE_DIR, 0),
                ("/a/host.bin".to_owned(), ENTRY_TYPE_FILE, 7),
                ("/a/mem.bin".to_owned(), ENTRY_TYPE_FILE, 5),
                ("/b".to_owned(), ENTRY_TYPE_DIR, 0),
                ("/b/src.bin".to_owned(), ENTRY_TYPE_FILE, 3),
            ]
        );
        assert_eq!(vfs.bytes(), 15);
    }

    #[test]
    fn removing_a_directory_removes_what_is_under_it_and_nothing_beside_it() {
        let mut vfs = Vfs::new();
        vfs.write_file("/game/save/a.bin", vec![1]);
        vfs.write_file("/game/b.bin", vec![2]);
        vfs.write_file("/game2/c.bin", vec![3]);
        vfs.write_file("/game.txt", vec![4]);
        assert!(vfs.remove("/game"));
        assert_eq!(vfs.entry_type("/game/save/a.bin"), None);
        assert_eq!(vfs.entry_type("/game/save"), None);
        assert_eq!(vfs.entry_type("/game/b.bin"), None);
        assert_eq!(vfs.entry_type("/game2/c.bin"), Some(ENTRY_TYPE_FILE));
        assert_eq!(vfs.entry_type("/game.txt"), Some(ENTRY_TYPE_FILE));
        let removed: Vec<(String, Option<u8>)> = vfs
            .take_changes()
            .into_iter()
            .map(|c| (c.path, c.kind))
            .collect();
        assert_eq!(
            removed,
            [
                ("/game".to_owned(), None),
                ("/game/b.bin".to_owned(), None),
                ("/game/save".to_owned(), None),
                ("/game/save/a.bin".to_owned(), None),
            ]
        );
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
