//! The emulated SD card.
//!
//! A path-addressed in-memory filesystem behind the `fsp-srv` service, so the
//! guest's `opendir`/`readdir`/`open`/`read` see a real, consistent tree
//! instead of a fixed reply. Hosts populate it — the browser frontend adds the
//! files the user drops in, and tests build small trees inline.

use std::collections::{BTreeMap, BTreeSet};

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
    /// Paths the **guest** has created, written, resized or deleted since the
    /// host last drained them ([`Vfs::take_changes`]).
    ///
    /// This is what lets a host persist the card without writing all of it
    /// back on every tick: the emulated SD card lives in memory, and the only
    /// thing a store outside the session needs to hear about is what changed.
    /// Host-side edits ([`Vfs::write_file`]) deliberately do **not** land here
    /// — that is the host loading the card *from* its store, and marking those
    /// would write every file straight back where it came from.
    changed: BTreeSet<String>,
}

/// What happened to a path, as [`Vfs::take_changes`] reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Change {
    pub path: String,
    /// `ENTRY_TYPE_FILE`, `ENTRY_TYPE_DIR`, or `None` when it was deleted.
    pub kind: Option<u8>,
    pub size: u64,
}

impl Vfs {
    /// An SD card with just the root and the `/switch` directory homebrew
    /// menus expect to exist.
    /// A filesystem with nothing in it but its root. Save data starts this
    /// way: a console creates the save empty and the title lays out whatever
    /// it wants inside, so seeding it with directories nobody asked for would
    /// invent structure the guest then has to work around.
    pub fn empty() -> Vfs {
        let mut vfs = Vfs {
            nodes: BTreeMap::new(),
            changed: BTreeSet::new(),
        };
        vfs.nodes.insert("/".to_owned(), Node::Dir);
        vfs
    }

    pub fn new() -> Vfs {
        let mut vfs = Vfs {
            nodes: BTreeMap::new(),
            changed: BTreeSet::new(),
        };
        vfs.nodes.insert("/".to_owned(), Node::Dir);
        vfs.create_dir("/switch");
        vfs
    }

    /// Normalize a guest path: strip any `device:` prefix, resolve the `.`
    /// and `..` components, collapse repeated slashes, and guarantee a
    /// leading slash and no trailing one.
    ///
    /// Resolving those components is not cosmetic. A guest builds paths by
    /// joining, so `.` and `..` arrive in them constantly, and treating them
    /// as ordinary names creates directories called `.` — `hb-appstore` left
    /// a tree of `/switch/.`, `/switch/./.get`, `/switch/./.get/packages`
    /// behind it, none of which it could find again by any other spelling.
    pub fn normalize(path: &str) -> String {
        let without_device = match path.split_once(":/") {
            Some((_, rest)) => rest,
            None => path,
        };
        let mut parts: Vec<&str> = Vec::new();
        for part in without_device.split('/') {
            match part {
                // An empty component is a leading, trailing or doubled slash,
                // and `.` is the directory you are already in. Neither names
                // anything. Note this is an exact match: `.get` is a perfectly
                // ordinary name that happens to start with a dot.
                "" | "." => {}
                // `..` goes up, and stops at the root rather than above it —
                // a guest does not get out of its own filesystem by asking.
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

    /// Add (or replace) a file, creating its parent directories.
    pub fn write_file(&mut self, path: &str, data: Vec<u8>) {
        let path = Self::normalize(path);
        if let Some(parent) = Self::parent_of(&path) {
            self.create_dir(&parent);
        }
        self.nodes.insert(path, Node::File(data));
    }

    /// Create an empty file of `size` bytes, failing if anything is already
    /// at `path`.
    ///
    /// Unlike [`Vfs::write_file`] — which is the *host* putting a file on the
    /// card and may replace what is there — this is the guest's `CreateFile`,
    /// and it must not truncate: `fsdev` opens an existing file by calling
    /// `CreateFile`, expecting "already exists", and then opening it. Losing
    /// that distinction silently emptied a file every time the guest reopened
    /// it.
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

    /// [`Vfs::write_file`], recorded as a change so a host that persists this
    /// storage writes the new contents out.
    ///
    /// This is the path for a *service* that keeps its state in a save —
    /// `set:sys` and its system settings — rather than the host staging a
    /// file the guest is about to read. The distinction is which side the
    /// write has to travel: [`Vfs::write_file`] is a value coming back from
    /// the store and must not be queued straight back into it.
    pub fn guest_write_file(&mut self, path: &str, data: Vec<u8>) {
        let path = Self::normalize(path);
        if let Some(parent) = Self::parent_of(&path) {
            self.guest_create_dir(&parent);
        }
        self.nodes.insert(path.clone(), Node::File(data));
        self.changed.insert(path);
    }

    /// [`Vfs::create_dir`], recording every directory it had to make so a host
    /// can persist an empty one the guest created.
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

    /// Drain the paths the guest has changed since this was last called.
    ///
    /// Each is reported with what is at it *now*, so a host can store the file
    /// or delete it from its store without a second lookup. A path that was
    /// created and deleted between two drains is reported once, as deleted.
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

    /// How many paths are waiting to be drained, without draining them.
    pub fn pending_changes(&self) -> usize {
        self.changed.len()
    }

    /// Write `data` at `offset`, growing the file (zero-filling any gap) when
    /// it runs past the end. Returns how many bytes were written, or `None`
    /// when the path is not a file.
    pub fn write(&mut self, path: &str, offset: u64, data: &[u8]) -> Option<usize> {
        let Some(Node::File(contents)) = self.nodes.get_mut(&Self::normalize(path)) else {
            return None;
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

    /// Resize a file, zero-filling any growth. Returns whether it was one.
    pub fn set_size(&mut self, path: &str, size: u64) -> bool {
        match self.nodes.get_mut(&Self::normalize(path)) {
            Some(Node::File(contents)) => {
                contents.resize(size as usize, 0);
                self.changed.insert(Self::normalize(path));
                true
            }
            _ => false,
        }
    }

    pub fn remove(&mut self, path: &str) -> bool {
        let path = Self::normalize(path);
        if self.nodes.remove(&path).is_none() {
            return false;
        }
        self.changed.insert(path);
        true
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

    #[test]
    fn normalize_resolves_dot_components() {
        // Guests build paths by joining, so `.` and `..` turn up in them all
        // the time. `hb-appstore` asked for "sdmc:/switch/./.get/packages" and
        // got a directory literally called "." with ".get" inside it, which
        // it could not find again by any other spelling.
        assert_eq!(Vfs::normalize("sdmc:/switch/."), "/switch");
        assert_eq!(
            Vfs::normalize("sdmc:/switch/./.get/packages"),
            "/switch/.get/packages"
        );
        assert_eq!(Vfs::normalize("/switch//a///b/"), "/switch/a/b");
        assert_eq!(Vfs::normalize("/switch/a/../b"), "/switch/b");
        // A leading dot is part of a name, not a component of its own.
        assert_eq!(Vfs::normalize("/.get"), "/.get");
        assert_eq!(Vfs::normalize("/..."), "/...");
        // And `..` stops at the root rather than climbing past it.
        assert_eq!(Vfs::normalize("/../../etc"), "/etc");
        assert_eq!(Vfs::normalize("sdmc:/"), "/");
        assert_eq!(Vfs::normalize("/switch/../.."), "/");
    }
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
    fn create_file_refuses_to_truncate_what_is_already_there() {
        // `fsdev` opens an existing file by calling `CreateFile` first and
        // expecting it to fail, so a create that quietly emptied the file
        // lost the contents on every reopen.
        let mut vfs = Vfs::new();
        assert!(vfs.create_file("/switch/cfg.json", 0));
        assert_eq!(vfs.size("/switch/cfg.json"), Some(0));
        vfs.write("/switch/cfg.json", 0, b"{}").unwrap();

        assert!(!vfs.create_file("/switch/cfg.json", 0), "already exists");
        assert_eq!(vfs.file("/switch/cfg.json"), Some(&b"{}"[..]));
        // A directory in the way counts as "already there" too.
        assert!(!vfs.create_file("/switch", 0));

        // The size argument is the file's initial length, zero-filled.
        assert!(vfs.create_file("/switch/blank.bin", 4));
        assert_eq!(vfs.file("/switch/blank.bin"), Some(&[0u8; 4][..]));
    }

    #[test]
    fn writing_past_the_end_grows_the_file() {
        let mut vfs = Vfs::new();
        vfs.write_file("/data.bin", vec![1, 2, 3, 4]);
        // Overwrite in place.
        assert_eq!(vfs.write("/data.bin", 1, &[9, 9]), Some(2));
        assert_eq!(vfs.file("/data.bin"), Some(&[1, 9, 9, 4][..]));
        // Past the end: the file grows to fit.
        assert_eq!(vfs.write("/data.bin", 4, &[5, 6]), Some(2));
        assert_eq!(vfs.file("/data.bin"), Some(&[1, 9, 9, 4, 5, 6][..]));
        // A gap between the end and the write is zero-filled rather than
        // left holding whatever the allocation had.
        assert_eq!(vfs.write("/data.bin", 8, &[7]), Some(1));
        assert_eq!(
            vfs.file("/data.bin"),
            Some(&[1, 9, 9, 4, 5, 6, 0, 0, 7][..])
        );
        // Neither a directory nor a missing path is writable.
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
        // The host restoring the card is not a change: reporting it would
        // write every file straight back to the store it just came from.
        vfs.write_file("/switch/restored.txt", b"x".to_vec());
        vfs.create_dir("/switch/olddir");
        assert_eq!(vfs.pending_changes(), 0);

        // Everything the guest does is.
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

        // Draining clears them, so a host only ever writes back what is new.
        assert_eq!(vfs.pending_changes(), 0);
        assert!(vfs.take_changes().is_empty());

        // A path written twice between drains is reported once, with the
        // state it ended up in.
        vfs.write("/switch/cfg.json", 0, b"ab").unwrap();
        vfs.write("/switch/cfg.json", 2, b"cd").unwrap();
        let changes = vfs.take_changes();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].size, 4);

        // ...including one created and then deleted, which is reported as the
        // deletion it ended as.
        vfs.create_file("/switch/tmp", 0);
        vfs.remove("/switch/tmp");
        let changes = vfs.take_changes();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].path, "/switch/tmp");
        assert_eq!(changes[0].kind, None);
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
