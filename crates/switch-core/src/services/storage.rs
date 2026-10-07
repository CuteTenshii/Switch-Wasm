//! RomFS, data archives and save data.

use crate::cpu::*;

/// A save's id plus owning user; the zero uid marks shared (system and device) saves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SaveKey {
    pub id: u64,
    pub user: [u8; 16],
}

impl SaveKey {
    pub const fn shared(id: u64) -> SaveKey {
        SaveKey { id, user: [0; 16] }
    }

    /// From the uid's two little-endian halves as the host passes them.
    pub fn from_halves(id: u64, user_lo: u64, user_hi: u64) -> SaveKey {
        let mut user = [0u8; 16];
        user[..8].copy_from_slice(&user_lo.to_le_bytes());
        user[8..].copy_from_slice(&user_hi.to_le_bytes());
        SaveKey { id, user }
    }
}

impl std::fmt::Display for SaveKey {
    /// `0100000000001000` for a shared save, `id@uid` (32 hex digits, memory order) for a user's.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:016x}", self.id)?;
        if self.user != [0; 16] {
            f.write_str("@")?;
            for byte in self.user {
                write!(f, "{byte:02x}")?;
            }
        }
        Ok(())
    }
}

impl Cpu {
    /// Set the decrypted RomFS that `OpenDataStorageByCurrentProcess` serves, for small images.
    /// See [`Cpu::set_romfs_source`].
    pub fn set_romfs(&mut self, data: Vec<u8>) {
        self.romfs = Some(Box::new(crate::source::MemSource(data)));
        self.romfs_indexes.remove(&None);
    }

    /// Register a system data archive for `OpenDataStorageByDataId`.
    pub fn add_data_archive(&mut self, data_id: u64, src: Box<dyn crate::source::ByteSource>) {
        self.data_archives.insert(data_id, src);
        self.romfs_indexes.remove(&Some(data_id));
    }

    /// Base id for this title's DLC: the NACP's, or the base program id (low 13 bits
    /// masked) plus 0x1000.
    pub fn add_on_content_base_id(&self) -> u64 {
        match self.add_on_content_base_id {
            0 => (self.program_id & !0x1FFF) + 0x1000,
            declared => declared,
        }
    }

    /// Set the DLC base id from the title's NACP.
    pub fn set_add_on_content_base_id(&mut self, base: u64) {
        self.add_on_content_base_id = base;
    }

    /// Register add-on content under its own id and return its index, or `None` when
    /// it belongs to another title.
    pub fn add_add_on_content(
        &mut self,
        content_id: u64,
        src: Box<dyn crate::source::ByteSource>,
    ) -> Option<u32> {
        let index = content_id.checked_sub(self.add_on_content_base_id())?;
        if index > 0x7FF {
            return None;
        }
        self.data_archives.insert(content_id, src);
        self.romfs_indexes.remove(&Some(content_id));
        self.add_on_content.insert(index as u32);
        // Tell a running title to re-read the list.
        if let Some(event) = self.aoc_list_changed_event {
            self.signal_event(event);
        }
        Some(index as u32)
    }

    pub fn has_data_archive(&self, data_id: u64) -> bool {
        self.data_archives.contains_key(&data_id)
    }

    pub fn add_on_content(&self) -> Vec<u32> {
        self.add_on_content.iter().copied().collect()
    }

    /// The save `key` names, created on first open as on a console.
    pub fn save_data_mut(&mut self, key: SaveKey) -> &mut crate::vfs::Vfs {
        self.saves.entry(key).or_insert_with(crate::vfs::Vfs::empty)
    }

    pub fn save_data(&self, key: SaveKey) -> Option<&crate::vfs::Vfs> {
        self.saves.get(&key)
    }

    /// Every opened save, for a host that persists them.
    pub fn save_keys(&self) -> Vec<SaveKey> {
        self.saves.keys().copied().collect()
    }

    pub(crate) fn vfs_for(&mut self, mount: Option<SaveKey>) -> &mut crate::vfs::Vfs {
        match mount {
            Some(key) => self.saves.entry(key).or_insert_with(crate::vfs::Vfs::empty),
            None => &mut self.fs,
        }
    }

    pub(crate) fn mount_of(&self, key: u64) -> Option<SaveKey> {
        self.fs_mount.get(&key).copied()
    }

    pub(crate) fn set_mount(&mut self, key: u64, mount: Option<SaveKey>) {
        match mount {
            Some(id) => {
                self.fs_mount.insert(key, id);
            }
            None => {
                self.fs_mount.remove(&key);
            }
        }
    }

    /// Same, backed by a decrypt-on-demand [`ByteSource`](crate::source::ByteSource).
    pub fn set_romfs_source(&mut self, src: Box<dyn crate::source::ByteSource>) {
        self.romfs = Some(src);
        self.romfs_indexes.remove(&None);
    }
}
