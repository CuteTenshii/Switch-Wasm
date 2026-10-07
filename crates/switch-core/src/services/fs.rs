//! `fsp-srv` and its objects: `IFileSystem`, `IFile`, `IDirectory`, `IStorage`
//! and the save-data interfaces. Storage reads stream from a [`crate::source::ByteSource`].

use crate::cpu::{Cpu, SaveKey};
use crate::trace::{Level, Trace};
use crate::Result;
use std::collections::VecDeque;

/// Emulated SD card size and free space; `ns` reports both and callers subtract them.
pub(crate) const SD_TOTAL_SPACE: u64 = 32 << 30;

pub(crate) const SD_FREE_SPACE: u64 = 16 << 30;

/// `Sdr104` / `Hs400`; 0 (`Identification`) would read as a device fault.
const SD_CARD_SPEED_MODE: i64 = 6;

const MMC_SPEED_MODE: i64 = 4;

/// eMMC user area mirrors the SD card; boot partitions are the X1's 4 MiB.
const MMC_USER_AREA_SIZE: i64 = SD_TOTAL_SPACE as i64;

const MMC_BOOT_PARTITION_SIZE: i64 = 4 << 20;

const SD_CARD_DETECTION: &str = "fsp-srv-sd-detection";

const GAME_CARD_DETECTION: &str = "fsp-srv-gamecard-detection";

/// Save and journal sizes reported before the title's NACP is read. Generous on
/// purpose: nothing enforces a quota, and under-reporting stops a title saving.
pub(crate) const DEFAULT_SAVE_DATA_SIZE: i64 = 0x400_0000;

pub(crate) const DEFAULT_SAVE_DATA_JOURNAL_SIZE: i64 = 0x100_0000;

pub(crate) const DEFAULT_CACHE_STORAGE_INDEX_MAX: i32 = 1;

/// The save quotas the running title's NACP declares. A real NACP's 0 is passed through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaveDataQuota {
    pub size: i64,
    pub journal_size: i64,
    /// Extension ceiling: a separate NACP field, commonly 0.
    pub size_max: i64,
    pub journal_size_max: i64,
    /// The same pair for the console-wide device save.
    pub device_size_max: i64,
    pub device_journal_size_max: i64,
    /// Cache storage ceiling (data plus journal) and how many the title may address.
    pub cache_storage_size_max: i64,
    pub cache_storage_index_max: i32,
}

impl Default for SaveDataQuota {
    fn default() -> SaveDataQuota {
        SaveDataQuota {
            size: DEFAULT_SAVE_DATA_SIZE,
            journal_size: DEFAULT_SAVE_DATA_JOURNAL_SIZE,
            size_max: DEFAULT_SAVE_DATA_SIZE,
            journal_size_max: DEFAULT_SAVE_DATA_JOURNAL_SIZE,
            device_size_max: DEFAULT_SAVE_DATA_SIZE,
            device_journal_size_max: DEFAULT_SAVE_DATA_JOURNAL_SIZE,
            cache_storage_size_max: DEFAULT_SAVE_DATA_SIZE,
            cache_storage_index_max: DEFAULT_CACHE_STORAGE_INDEX_MAX,
        }
    }
}

impl From<&crate::control::Nacp> for SaveDataQuota {
    fn from(nacp: &crate::control::Nacp) -> SaveDataQuota {
        SaveDataQuota {
            size: nacp.user_account_save_data_size,
            journal_size: nacp.user_account_save_data_journal_size,
            size_max: nacp.user_account_save_data_size_max,
            journal_size_max: nacp.user_account_save_data_journal_size_max,
            device_size_max: nacp.device_save_data_size_max,
            device_journal_size_max: nacp.device_save_data_journal_size_max,
            cache_storage_size_max: nacp.cache_storage_data_and_journal_size_max,
            cache_storage_index_max: i32::from(nacp.cache_storage_index_max),
        }
    }
}

/// `sizeof(FsSaveDataInfo)`.
pub(crate) const SAVE_DATA_INFO_SIZE: usize = 0x60;

/// `FsSaveDataSpaceId`.
const SPACE_SYSTEM: u8 = 0;

const SPACE_USER: u8 = 1;

/// `FsSaveDataType`.
const SAVE_TYPE_SYSTEM: u8 = 0;

const SAVE_TYPE_ACCOUNT: u8 = 1;

const SAVE_TYPE_DEVICE: u8 = 3;

/// System save ids have the top bit set; application ids never do.
fn save_data_type(key: SaveKey) -> u8 {
    if key.user != [0; 16] {
        SAVE_TYPE_ACCOUNT
    } else if key.id >> 63 == 1 {
        SAVE_TYPE_SYSTEM
    } else {
        SAVE_TYPE_DEVICE
    }
}

/// The criteria of a `FsSaveDataFilter` that are switched on.
#[derive(Clone, Copy)]
struct SaveDataFilter {
    application_id: Option<u64>,
    save_type: Option<u8>,
    user: Option<[u8; 16]>,
    system_save_id: Option<u64>,
    index: Option<u16>,
}

impl SaveDataFilter {
    fn admits(&self, key: SaveKey, kind: u8) -> bool {
        let (application, system) = match kind {
            SAVE_TYPE_SYSTEM => (0, key.id),
            _ => (key.id, 0),
        };
        self.application_id.is_none_or(|id| id == application)
            && self.save_type.is_none_or(|t| t == kind)
            && self.user.is_none_or(|u| u == key.user)
            && self.system_save_id.is_none_or(|id| id == system)
            && self.index.is_none_or(|i| i == 0)
    }
}

/// `fs`'s "path not found" (2002-0001).
const PATH_NOT_FOUND: u32 = 2 | (1 << 9);

/// A `Result` in Horizon's `2002-0001` form.
fn result_text(result: u32) -> String {
    if result == 0 {
        return "ok".to_owned();
    }
    let module = result & 0x1FF;
    let description = (result >> 9) & 0x1FFF;
    format!("{:04}-{description:04} ({result:#x})", 2000 + module)
}

fn fsp_srv_command(cmd: u32) -> Option<&'static str> {
    Some(match cmd {
        0 => "OpenFileSystem",
        1 => "SetCurrentProcess",
        2 => "OpenDataFileSystemByCurrentProcess",
        7 => "OpenFileSystemWithPatch",
        8 => "OpenFileSystemWithId",
        9 => "OpenDataFileSystemByApplicationId",
        11 => "OpenBisFileSystem",
        12 => "OpenBisStorage",
        17 => "OpenHostFileSystem",
        18 => "OpenSdCardFileSystem",
        22 => "CreateSaveDataFileSystem",
        23 => "CreateSaveDataFileSystemBySystemSaveDataId",
        30 => "OpenGameCardStorage",
        31 => "OpenGameCardFileSystem",
        51 => "OpenSaveDataFileSystem",
        52 => "OpenSaveDataFileSystemBySystemSaveDataId",
        53 => "OpenReadOnlySaveDataFileSystem",
        60 => "OpenSaveDataInfoReader",
        61 => "OpenSaveDataInfoReaderBySaveDataSpaceId",
        62 => "OpenSaveDataInfoReaderOnlyCacheStorage",
        68 => "OpenSaveDataInfoReaderWithFilter",
        200 => "OpenDataStorageByCurrentProcess",
        202 => "OpenDataStorageByDataId",
        203 => "OpenPatchDataStorageByCurrentProcess",
        400 => "OpenDeviceOperator",
        500 => "OpenSdCardDetectionEventNotifier",
        501 => "OpenGameCardDetectionEventNotifier",
        1003 => "DisableAutoSaveDataCreation",
        1004 => "SetGlobalAccessLogMode",
        1005 => "GetGlobalAccessLogMode",
        1006 => "OutputAccessLogToSdCard",
        1014 => "OutputMultiProgramTagAccessLog",
        1015 => "FlushAccessLogOnSdCard",
        1016 => "OutputApplicationInfoAccessLog",
        _ => return None,
    })
}

fn file_system_command(cmd: u32) -> Option<&'static str> {
    Some(match cmd {
        0 => "CreateFile",
        1 => "DeleteFile",
        2 => "CreateDirectory",
        3 => "DeleteDirectory",
        4 => "DeleteDirectoryRecursively",
        5 => "RenameFile",
        6 => "RenameDirectory",
        7 => "GetEntryType",
        8 => "OpenFile",
        9 => "OpenDirectory",
        10 => "Commit",
        11 => "GetFreeSpaceSize",
        12 => "GetTotalSpaceSize",
        13 => "CleanDirectoryRecursively",
        14 => "GetFileTimeStampRaw",
        15 => "QueryEntry",
        _ => return None,
    })
}

/// `IFile`'s commands, which `IStorage` shares.
fn file_command(cmd: u32) -> Option<&'static str> {
    Some(match cmd {
        0 => "Read",
        1 => "Write",
        2 => "Flush",
        3 => "SetSize",
        4 => "GetSize",
        5 => "OperateRange",
        _ => return None,
    })
}

fn directory_command(cmd: u32) -> Option<&'static str> {
    Some(match cmd {
        0 => "Read",
        1 => "GetEntryCount",
        _ => return None,
    })
}

/// Path operations [`FsActivity`] journals before it only counts them.
const JOURNAL_CAP: usize = 256;

/// Distinct files tallied per reading; the rest go under [`OTHER_FILES`].
const FILES_CAP: usize = 256;

pub const OTHER_FILES: &str = "(other files)";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FileIo {
    pub reads: u64,
    pub read_bytes: u64,
    pub writes: u64,
    pub write_bytes: u64,
}

/// Guest filesystem activity, taken (not read) by the host: a journal of path
/// operations and per-file read/write tallies.
#[derive(Debug, Default, Clone)]
pub struct FsActivity {
    /// Failed requests of any kind since boot.
    pub failures: u64,
    journal: Vec<String>,
    dropped: u64,
    files: std::collections::BTreeMap<String, FileIo>,
}

impl FsActivity {
    pub fn record(&mut self, line: String) {
        if self.journal.len() < JOURNAL_CAP {
            self.journal.push(line);
        } else {
            self.dropped += 1;
        }
    }

    pub fn take_journal(&mut self) -> (Vec<String>, u64) {
        (
            std::mem::take(&mut self.journal),
            std::mem::take(&mut self.dropped),
        )
    }

    pub fn read(&mut self, file: &str, bytes: u64) {
        let io = self.file(file);
        io.reads += 1;
        io.read_bytes += bytes;
    }

    pub fn wrote(&mut self, file: &str, bytes: u64) {
        let io = self.file(file);
        io.writes += 1;
        io.write_bytes += bytes;
    }

    pub fn take_files(&mut self) -> std::collections::BTreeMap<String, FileIo> {
        std::mem::take(&mut self.files)
    }

    fn file(&mut self, file: &str) -> &mut FileIo {
        // Look up by `&str` first so an already-tallied file allocates nothing.
        let name = if self.files.contains_key(file) || self.files.len() < FILES_CAP {
            file
        } else {
            OTHER_FILES
        };
        if !self.files.contains_key(name) {
            self.files.insert(name.to_owned(), FileIo::default());
        }
        self.files.get_mut(name).expect("inserted above")
    }
}

fn file_text(mount: Option<SaveKey>, path: &str) -> String {
    format!("{}:{path}", mount_text(mount))
}

fn storage_text(archive: Option<u64>) -> String {
    match archive {
        Some(id) => format!("data archive {id:016x}"),
        None => "romfs".to_owned(),
    }
}

fn mount_text(mount: Option<SaveKey>) -> String {
    match mount {
        Some(key) => format!("save {key}"),
        None => "sdmc".to_owned(),
    }
}

impl Cpu {
    /// Run one filesystem request, count a failure, and log it under `TRACE_FS` and
    /// to the journal. `subject` is only evaluated when the line is emitted.
    fn fs_traced(
        &mut self,
        interface: &str,
        cmd_id: Option<u32>,
        name: fn(u32) -> Option<&'static str>,
        journal: bool,
        subject: impl FnOnce(&Self) -> String,
        handle: impl FnOnce(&mut Self) -> Result<()>,
    ) -> Result<()> {
        let tracing = crate::trace::enabled(Trace::Fs);
        let subject = if tracing || journal {
            subject(self)
        } else {
            String::new()
        };
        self.last_ipc_result = None;
        let handled = handle(self);
        let failed = handled.is_err() || self.last_ipc_result.is_some_and(|r| r != 0);
        if failed {
            self.fs_activity.failures += 1;
        }
        if !tracing && !journal {
            return handled;
        }
        let command = match cmd_id {
            Some(cmd) => name(cmd).map_or_else(|| format!("cmd {cmd}"), str::to_owned),
            None => "<no command>".to_owned(),
        };
        let outcome = match (&handled, self.last_ipc_result) {
            (Err(e), _) => format!("failed: {e}"),
            (Ok(()), Some(result)) => result_text(result),
            (Ok(()), None) => "no reply".to_owned(),
        };
        let line = format!("{interface} {command}{subject} -> {outcome}");
        if tracing {
            crate::traceln!("[fs] {line}");
        }
        if journal {
            self.fs_activity.record(line);
        }
        handled
    }

    pub(crate) fn fsp_srv_request(
        &mut self,
        tls: u32,
        cmd_id: Option<u32>,
        handle: u64,
    ) -> Result<()> {
        // Control requests are session plumbing whose ids overlap `fsp-srv`'s.
        if self.ipc_is_control_request(tls) {
            return self.fsp_srv_dispatch(tls, cmd_id, handle);
        }
        // Access-log commands (1004+) would bury the journal.
        let journal = cmd_id.is_some_and(|cmd| cmd < 1000);
        self.fs_traced(
            "fsp-srv",
            cmd_id,
            fsp_srv_command,
            journal,
            |_| String::new(),
            |cpu| cpu.fsp_srv_dispatch(tls, cmd_id, handle),
        )
    }

    fn fsp_srv_dispatch(&mut self, tls: u32, cmd_id: Option<u32>, handle: u64) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "fsp-srv");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
            };
        }
        match cmd_id {
            // 0 = ConvertToDomain.
            Some(0) => {
                let obj = self.alloc_domain_object();
                self.record_domain_object(handle, obj, "fsp-srv");
                self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
            }
            Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // 18 = OpenSdCardFileSystem, 11 = OpenBisFileSystem.
            Some(18) | Some(11) => {
                self.reply_with_interface(tls, handle, "fsp-srv-fs")?;
                Ok(())
            }
            // 200 = OpenDataStorageByCurrentProcess: the title's RomFS as a raw `IStorage`.
            Some(200) => {
                if self.romfs.is_none() {
                    // No RomFS this session.
                    return self.write_ipc_response(tls, PATH_NOT_FOUND, &[], &[], &[]);
                }
                crate::trace!(
                    Trace::Fs,
                    "[fs] romfs -> {:#x} bytes",
                    self.storage_source(None).map_or(0, |s| s.len())
                );
                self.reply_with_interface(tls, handle, "fsp-srv-storage")?;
                Ok(())
            }
            // 202 = OpenDataStorageByDataId: a registered data archive
            // ([`Cpu::add_data_archive`]), or not found.
            Some(202) => {
                let data = self.ipc_request_data(tls);
                let data_id = self.mem.read_u64(data.wrapping_add(8))?;
                if !self.data_archives.contains_key(&data_id) {
                    self.diagnostic(
                        Level::Warn,
                        &format!(
                            "[fs] no system data archive registered for data id {data_id:016x}"
                        ),
                    );
                    return self.write_ipc_response(tls, PATH_NOT_FOUND, &[], &[], &[]);
                }
                let key = self.reply_with_interface(tls, handle, "fsp-srv-storage")?;
                self.fs_storage_archive.insert(key, data_id);
                if crate::trace::enabled(Trace::Fs) {
                    let size = self.storage_source(Some(data_id)).map_or(0, |s| s.len());
                    crate::traceln!("[fs] data archive {data_id:016x} -> {size:#x} bytes");
                }
                Ok(())
            }
            // 203 = OpenPatchDataStorageByCurrentProcess. There is no update NCA, and
            // `QueryMountRomCacheSize` only treats `TargetNotFound` as "no patch".
            Some(203) => {
                const TARGET_NOT_FOUND: u32 = 2 | (1002 << 9);
                self.write_ipc_response(tls, TARGET_NOT_FOUND, &[], &[], &[])
            }
            // 22/23 = Create, 51/52/53 = Open save data: a filesystem over the NAND save.
            Some(22) | Some(23) | Some(51) | Some(52) | Some(53) => {
                let id = self.save_data_key(tls);
                crate::trace!(Trace::Fs, "[fs] save data {id}");
                self.save_data_mut(id);
                if matches!(cmd_id, Some(22) | Some(23)) {
                    return self.write_ipc_response(tls, 0, &[], &[], &[]);
                }
                let key = self.reply_with_interface(tls, handle, "fsp-srv-fs")?;
                self.set_mount(key, Some(id));
                Ok(())
            }
            // 60 = all spaces, 61 = one space, 62 = cache storage (none), 68 = filtered.
            Some(60) | Some(61) | Some(62) | Some(68) => {
                let data = self.ipc_request_data(tls);
                let space = self.mem.read_u8(data).unwrap_or(0);
                let filter = match cmd_id {
                    Some(68) => Some(self.save_data_filter(data.wrapping_add(8))),
                    _ => None,
                };
                let infos = match cmd_id {
                    Some(62) => VecDeque::new(),
                    Some(60) => self.save_data_infos(None, None),
                    _ => self.save_data_infos(Some(space), filter),
                };
                let key = self.reply_with_interface(tls, handle, "fsp-srv-save-info-reader")?;
                self.fs_save_infos.insert(key, infos);
                Ok(())
            }
            // 400 = OpenDeviceOperator.
            Some(400) => {
                self.reply_with_interface(tls, handle, "fsp-srv-device-operator")?;
                Ok(())
            }
            // 500/501 = Open{SdCard,GameCard}DetectionEventNotifier.
            Some(500) | Some(501) => {
                let name = match cmd_id {
                    Some(500) => SD_CARD_DETECTION,
                    _ => GAME_CARD_DETECTION,
                };
                self.reply_with_interface(tls, handle, name)?;
                Ok(())
            }
            // 1003 = DisableAutoSaveDataCreation: accepted, not honoured.
            Some(1003) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // 1004 = SetGlobalAccessLogMode, 1005 = GetGlobalAccessLogMode.
            Some(1004) => {
                self.fs_access_log_mode = self.mem.read_u32(self.ipc_request_data(tls))?;
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(1005) => {
                let mode = self.fs_access_log_mode;
                self.write_ipc_response(tls, 0, &[], &mode.to_le_bytes(), &[])
            }
            // Access-log writes, dropped.
            Some(1006) | Some(1014) | Some(1015) | Some(1016) => {
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // `Open*` commands this console can't serve: a bare success would hand out a
            // null object, so answer "not found".
            Some(2) | Some(7) | Some(8) | Some(9) | Some(12) | Some(17) | Some(30) | Some(31) => {
                self.warn_no_implementation("fsp-srv", cmd_id);
                self.report_refused_open(tls, cmd_id);
                self.write_ipc_response(tls, PATH_NOT_FOUND, &[], &[], &[])
            }
            // Fabricated success, with a warning since the caller may read zeroes.
            _ => {
                self.warn_no_implementation("fsp-srv", cmd_id);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
        }
    }

    /// Log which content a refused `Open*` asked for, assuming `nn::fs`'s layout
    /// (proxy type, program id), with the raw words alongside.
    fn report_refused_open(&mut self, tls: u32, cmd_id: Option<u32>) {
        let data = self.ipc_request_data(tls);
        let first = self.mem.read_u64(data).unwrap_or(0);
        let second = self.mem.read_u64(data.wrapping_add(8)).unwrap_or(0);
        let kind = match first as u8 {
            0 => "Code",
            1 => "Rom",
            2 => "Logo",
            3 => "Control",
            4 => "Manual",
            5 => "Meta",
            6 => "Data",
            7 => "Package",
            8 => "RegisteredUpdate",
            _ => "?",
        };
        let cmd = cmd_id.unwrap_or(0);
        self.diagnostic(
            Level::Warn,
            &format!(
                "[fs] refused fsp-srv {cmd}: raw {first:#018x} {second:#018x} \
                 (reads as type {kind}, program {second:016x})"
            ),
        );
    }

    /// `ISaveDataInfoReader`: cmd 0 = `ReadSaveDataInfo`. Reporting zero entries
    /// ends the caller's scan.
    pub(crate) fn fs_save_data_info_reader_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        match cmd_id {
            Some(0) => {
                let key = self.ipc_object_key(tls, handle);
                let (addr, len) = self.ipc_output_buffer(tls, 0).unwrap_or((0, 0));
                let room = len as usize / SAVE_DATA_INFO_SIZE;
                let infos = self.fs_save_infos.entry(key).or_default();
                let batch: Vec<_> = infos.drain(..room.min(infos.len())).collect();
                for (i, info) in batch.iter().enumerate() {
                    self.mem
                        .write_bytes(addr.wrapping_add((i * SAVE_DATA_INFO_SIZE) as u32), info)?;
                }
                let count = batch.len() as i64;
                self.write_ipc_response(tls, 0, &[], &count.to_le_bytes(), &[])
            }
            _ => self.unimplemented_command(tls, "fsp-srv-save-info-reader", cmd_id),
        }
    }

    /// The `FsSaveDataInfo` of every save in `space` (all with `None`) that `filter` admits.
    fn save_data_infos(
        &self,
        space: Option<u8>,
        filter: Option<SaveDataFilter>,
    ) -> VecDeque<[u8; SAVE_DATA_INFO_SIZE]> {
        let mut keys: Vec<SaveKey> = self.saves.keys().copied().collect();
        keys.sort();
        keys.iter()
            .enumerate()
            .filter_map(|(index, &key)| {
                let kind = save_data_type(key);
                let system = kind == SAVE_TYPE_SYSTEM;
                let key_space = if system { SPACE_SYSTEM } else { SPACE_USER };
                if space.is_some_and(|s| s != key_space)
                    || filter.is_some_and(|f| !f.admits(key, kind))
                {
                    return None;
                }
                // Application saves have no id of their own here; their position stands in.
                let save_data_id = if system { key.id } else { index as u64 + 1 };
                let mut info = [0u8; SAVE_DATA_INFO_SIZE];
                info[..8].copy_from_slice(&save_data_id.to_le_bytes());
                info[8] = key_space;
                info[9] = kind;
                info[0x10..0x20].copy_from_slice(&key.user);
                let id_at = if system { 0x20 } else { 0x28 };
                info[id_at..id_at + 8].copy_from_slice(&key.id.to_le_bytes());
                Some(info)
            })
            .collect()
    }

    /// A `FsSaveDataFilter` at `addr`.
    fn save_data_filter(&self, addr: u32) -> SaveDataFilter {
        let byte = |offset: u32| self.mem.read_u8(addr.wrapping_add(offset)).unwrap_or(0);
        let word = |offset: u32| self.mem.read_u64(addr.wrapping_add(offset)).unwrap_or(0);
        let attr = 8;
        let mut user = [0u8; 16];
        for (i, b) in user.iter_mut().enumerate() {
            *b = byte(attr + 8 + i as u32);
        }
        let index = u16::from(byte(attr + 0x22)) | u16::from(byte(attr + 0x23)) << 8;
        SaveDataFilter {
            application_id: (byte(0) != 0).then(|| word(attr)),
            save_type: (byte(1) != 0).then(|| byte(attr + 0x20)),
            user: (byte(2) != 0).then_some(user),
            system_save_id: (byte(3) != 0).then(|| word(attr + 0x18)),
            index: (byte(4) != 0).then_some(index),
        }
    }

    /// `IDeviceOperator`: SD card, eMMC and game card queries.
    pub(crate) fn fs_device_operator_request(
        &mut self,
        tls: u32,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        const CID_SIZE: usize = 0x10;
        const EXTENDED_CSD_SIZE: usize = 0x200;
        match cmd_id {
            // 0 = IsSdCardInserted, 200 = IsGameCardInserted.
            Some(0) => self.write_ipc_response(tls, 0, &[], &[1u8], &[]),
            Some(200) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
            // 1 = GetSdCardSpeedMode, 101 = GetMmcSpeedMode.
            Some(1) => self.write_ipc_response(tls, 0, &[], &SD_CARD_SPEED_MODE.to_le_bytes(), &[]),
            Some(101) => self.write_ipc_response(tls, 0, &[], &MMC_SPEED_MODE.to_le_bytes(), &[]),
            // 2 = GetSdCardCid, 100 = GetMmcCid: zeroes, written to the full width.
            Some(2) | Some(100) => {
                let requested = self.mem.read_u64(self.ipc_request_data(tls)).unwrap_or(0);
                self.write_out_buffer(tls, &[0u8; CID_SIZE], requested)?;
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // 3 = GetSdCardUserAreaSize, 4 = GetSdCardProtectedAreaSize.
            Some(3) => {
                let size = SD_TOTAL_SPACE as i64;
                self.write_ipc_response(tls, 0, &[], &size.to_le_bytes(), &[])
            }
            Some(4) => self.write_ipc_response(tls, 0, &[], &0i64.to_le_bytes(), &[]),
            // 5 = GetAndClearSdCardErrorInfo, 113 = GetAndClearMmcErrorInfo.
            Some(5) | Some(113) => self.write_ipc_response(tls, 0, &[], &[0u8; 0x18], &[]),
            // 111 = GetMmcPartitionSize: 0 is user data, 1 and 2 are boot partitions.
            Some(111) => {
                let size = match self.mem.read_u32(self.ipc_request_data(tls)).unwrap_or(0) {
                    0 => MMC_USER_AREA_SIZE,
                    _ => MMC_BOOT_PARTITION_SIZE,
                };
                self.write_ipc_response(tls, 0, &[], &size.to_le_bytes(), &[])
            }
            // 112 = GetMmcPatrolCount.
            Some(112) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            // 114 = GetMmcExtendedCsd.
            Some(114) => {
                let requested = self.mem.read_u64(self.ipc_request_data(tls)).unwrap_or(0);
                self.write_out_buffer(tls, &[0u8; EXTENDED_CSD_SIZE], requested)?;
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // 115/116 = Suspend/ResumeMmcPatrol, 400/401 = Suspend/ResumeSdmmcControl.
            Some(115) | Some(116) | Some(400) | Some(401) => {
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // 300 = SetSpeedEmulationMode, 301 = GetSpeedEmulationMode, round-tripped.
            Some(300) => {
                self.fs_speed_emulation_mode =
                    self.mem.read_u32(self.ipc_request_data(tls)).unwrap_or(0);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(301) => {
                let mode = self.fs_speed_emulation_mode;
                self.write_ipc_response(tls, 0, &[], &mode.to_le_bytes(), &[])
            }
            // Game-card, erase and direct-write commands; no card is ever inserted.
            _ => self.unimplemented_command(tls, "fsp-srv-device-operator", cmd_id),
        }
    }

    /// `IEventNotifier`: cmd 0 = GetEventHandle. Card slots never change, so the event
    /// never fires. One event per slot, shared by every caller.
    pub(crate) fn fs_detection_notifier_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        let game_card = self.ipc_interface(tls, handle, SD_CARD_DETECTION) == GAME_CARD_DETECTION;
        let (slot, name) = if game_card {
            (501u32, GAME_CARD_DETECTION)
        } else {
            (500u32, SD_CARD_DETECTION)
        };
        match cmd_id {
            // A copy handle, as `eventLoadRemote` expects.
            Some(0) => {
                let event = match self.fs_detection_events.get(&slot) {
                    Some(&event) => event,
                    None => {
                        let event = self.alloc_event(name, false);
                        self.fs_detection_events.insert(slot, event);
                        event
                    }
                };
                self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
            }
            _ => self.unimplemented_command(tls, name, cmd_id),
        }
    }

    /// Fill an out buffer, clamped to its size and the requested width.
    fn write_out_buffer(&mut self, tls: u32, bytes: &[u8], requested: u64) -> Result<()> {
        let Some((addr, len)) = self.ipc_output_buffer(tls, 0) else {
            return Ok(());
        };
        let take = requested.min(bytes.len() as u64).min(u64::from(len)) as usize;
        self.mem.write_bytes(addr, &bytes[..take])
    }

    /// Which save a request's `SaveDataAttribute` names: system saves by system save
    /// id, applications by title id (none means the running title), per user by uid.
    fn save_data_key(&mut self, tls: u32) -> SaveKey {
        const ATTRIBUTE: u32 = 8;
        const USER_ID: u32 = 0x8;
        const SYSTEM_SAVE_DATA_ID: u32 = 0x18;
        let attribute = self.ipc_request_data(tls).wrapping_add(ATTRIBUTE);
        let application_id = self.mem.read_u64(attribute).unwrap_or(0);
        let mut user = [0u8; 16];
        for (index, byte) in user.iter_mut().enumerate() {
            *byte = self
                .mem
                .read_u8(attribute.wrapping_add(USER_ID + index as u32))
                .unwrap_or(0);
        }
        let system_save_id = self
            .mem
            .read_u64(attribute.wrapping_add(SYSTEM_SAVE_DATA_ID))
            .unwrap_or(0);
        let id = match (system_save_id, application_id) {
            (0, 0) => self.program_id(),
            (0, application) => application,
            (system, _) => system,
        };
        SaveKey { id, user }
    }

    /// `IStorage` over the process RomFS or a data archive.
    /// Cmd 0 = Read(u64 offset, u64 size), cmd 4 = GetSize.
    pub(crate) fn fs_storage_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        self.fs_traced(
            "storage",
            cmd_id,
            file_command,
            false,
            |cpu| {
                let key = cpu.ipc_object_key(tls, handle);
                format!(
                    " {}",
                    storage_text(cpu.fs_storage_archive.get(&key).copied())
                )
            },
            |cpu| cpu.fs_storage_dispatch(tls, handle, cmd_id),
        )
    }

    fn fs_storage_dispatch(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        // The process's RomFS (200) or a system data archive (202).
        let archive = self
            .fs_storage_archive
            .get(&self.ipc_object_key(tls, handle))
            .copied();
        let size = self.storage_source(archive).map_or(0, |s| s.len());
        match cmd_id {
            Some(0) => {
                let data = self.ipc_request_data(tls);
                // `IStorage::Read(s64 offset, u64 size)`: no option word, unlike `IFile`.
                let offset = self.mem.read_u64(data)?;
                let requested = self.mem.read_u64(data.wrapping_add(8))?;
                let trace_storage = crate::trace::enabled(Trace::Fs);
                if trace_storage {
                    crate::traceln!(
                        "[storage] read offset={offset:#x} size={requested:#x} of {size:#x}"
                    );
                }
                // All or nothing: real `fs` refuses a range past the end.
                const OUT_OF_RANGE: u32 = 2 | (3005 << 9);
                if offset > size || requested > size - offset {
                    return self.write_ipc_response(tls, OUT_OF_RANGE, &[], &[], &[]);
                }
                let start = offset;
                let end = start + requested;
                if let Some(addr) = self.ipc_output_buffer_addr(tls, 0) {
                    // Copy through a fixed staging buffer; the RomFS is never staged whole.
                    const CHUNK: u64 = 64 * 1024;
                    let mut buf = vec![0u8; (end - start).min(CHUNK) as usize];
                    let mut pos = start;
                    let mut written = 0u32;
                    while pos < end {
                        let take = ((end - pos).min(CHUNK)) as usize;
                        let got = match self.storage_source(archive) {
                            Some(src) => src.read_at(pos, &mut buf[..take])?,
                            None => 0,
                        };
                        if got == 0 {
                            break;
                        }
                        self.mem
                            .write_bytes(addr.wrapping_add(written), &buf[..got])?;
                        written += got as u32;
                        pos += got as u64;
                    }
                    self.tally_storage_read(archive, start, u64::from(written));
                } else {
                    self.tally_storage_read(archive, start, 0);
                }
                if trace_storage {
                    let head: Vec<u8> = match self.ipc_output_buffer_addr(tls, 0) {
                        Some(addr) => (0..16)
                            .map(|i| self.mem.read_u8(addr.wrapping_add(i)).unwrap_or(0))
                            .collect(),
                        None => Vec::new(),
                    };
                    crate::traceln!("[storage]   -> {head:02x?}");
                }
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // GetSize -> u64
            Some(4) => self.write_ipc_response(tls, 0, &[], &size.to_le_bytes(), &[]),
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// Charge a storage read to the RomFS files it overlaps; bytes outside every
    /// file are the header and tables.
    fn tally_storage_read(&mut self, archive: Option<u64>, offset: u64, len: u64) {
        if !self.romfs_indexes.contains_key(&archive) {
            let index = self
                .storage_source(archive)
                .and_then(|src| crate::romfs::RomFsIndex::read(src).ok());
            self.romfs_indexes.insert(archive, index);
        }
        let storage = storage_text(archive);
        let activity = &mut self.fs_activity;
        let Some(index) = self.romfs_indexes.get(&archive).and_then(Option::as_ref) else {
            activity.read(&storage, len);
            return;
        };
        let mut named = 0;
        for (path, bytes) in index.files_in(offset, len) {
            activity.read(&format!("{storage}:{path}"), bytes);
            named += bytes;
        }
        if named < len || len == 0 {
            activity.read(&format!("{storage} (tables)"), len - named);
        }
    }

    fn storage_source(&self, archive: Option<u64>) -> Option<&dyn crate::source::ByteSource> {
        match archive {
            Some(id) => self.data_archives.get(&id).map(|b| b.as_ref()),
            None => self.romfs.as_deref(),
        }
    }

    /// `IFileSystem` over the SD card or a save in [`crate::vfs`].
    pub(crate) fn fs_request(&mut self, tls: u32, cmd_id: Option<u32>, handle: u64) -> Result<()> {
        self.fs_traced(
            "fs",
            cmd_id,
            file_system_command,
            true,
            |cpu| {
                let path = cpu.ipc_request_path(tls);
                let mount = cpu.mount_of(cpu.ipc_object_key(tls, handle));
                format!(" {path:?} on {}", mount_text(mount))
            },
            |cpu| cpu.fs_dispatch(tls, cmd_id, handle),
        )
    }

    fn fs_dispatch(&mut self, tls: u32, cmd_id: Option<u32>, handle: u64) -> Result<()> {
        /// Horizon `fs` result: path already exists.
        const PATH_ALREADY_EXISTS: u32 = 2 | (2 << 9);
        let path = self.ipc_request_path(tls);
        // The SD card and saves share this interface; the object picks the storage.
        let mount = self.mount_of(self.ipc_object_key(tls, handle));
        match cmd_id {
            // CreateFile / CreateDirectory. An existing file is an error, not a truncation:
            // `fsdev` relies on "already exists".
            Some(0) => {
                let data = self.ipc_request_data(tls);
                let size = self.mem.read_u64(data.wrapping_add(8)).unwrap_or(0);
                if self.vfs_for(mount).create_file(&path, size) {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                } else {
                    self.write_ipc_response(tls, PATH_ALREADY_EXISTS, &[], &[], &[])
                }
            }
            Some(2) => {
                self.vfs_for(mount).guest_create_dir(&path);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // DeleteFile / DeleteDirectory / DeleteDirectoryRecursively
            Some(1) | Some(3) | Some(4) => {
                if self.vfs_for(mount).remove(&path) {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                } else {
                    self.write_ipc_response(tls, PATH_NOT_FOUND, &[], &[], &[])
                }
            }
            // GetEntryType
            Some(7) => match self.vfs_for(mount).entry_type(&path) {
                Some(kind) => {
                    self.write_ipc_response(tls, 0, &[], &(kind as u32).to_le_bytes(), &[])
                }
                None => self.write_ipc_response(tls, PATH_NOT_FOUND, &[], &[], &[]),
            },
            // OpenFile(u32 mode) -> IFile
            Some(8) => {
                if self.vfs_for(mount).entry_type(&path) != Some(crate::vfs::ENTRY_TYPE_FILE) {
                    return self.write_ipc_response(tls, PATH_NOT_FOUND, &[], &[], &[]);
                }
                let key = self.reply_with_interface(tls, handle, "fsp-srv-fs-file")?;
                self.fs_files.insert(key, path);
                self.set_mount(key, mount);
                Ok(())
            }
            // OpenDirectory(u32 mode) -> IDirectory
            Some(9) => match self.vfs_for(mount).read_dir(&path) {
                Some(entries) => {
                    let key = self.reply_with_interface(tls, handle, "fsp-srv-fs-dir")?;
                    self.fs_dirs.insert(key, entries);
                    Ok(())
                }
                None => self.write_ipc_response(tls, PATH_NOT_FOUND, &[], &[], &[]),
            },
            // GetFreeSpaceSize / GetTotalSpaceSize
            Some(11) | Some(12) => {
                let bytes = 32u64 << 30;
                self.write_ipc_response(tls, 0, &[], &bytes.to_le_bytes(), &[])
            }
            // GetFileTimeStampRaw: times are not recorded, so `is_valid` is 0.
            Some(14) => {
                if self.vfs_for(mount).entry_type(&path).is_none() {
                    return self.write_ipc_response(tls, PATH_NOT_FOUND, &[], &[], &[]);
                }
                self.write_ipc_response(tls, 0, &[], &[0u8; 0x20], &[])
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// `IDirectory`: cmd 0 = Read, cmd 1 = GetEntryCount.
    pub(crate) fn fs_dir_request(&mut self, tls: u32, cmd_id: Option<u32>, key: u64) -> Result<()> {
        self.fs_traced(
            "dir",
            cmd_id,
            directory_command,
            false,
            |cpu| {
                let entries = cpu.fs_dirs.get(&key).map_or(0, Vec::len);
                format!(" ({entries} entries left)")
            },
            |cpu| cpu.fs_dir_dispatch(tls, cmd_id, key),
        )
    }

    fn fs_dir_dispatch(&mut self, tls: u32, cmd_id: Option<u32>, key: u64) -> Result<()> {
        /// `sizeof(FsDirectoryEntry)`.
        const ENTRY_SIZE: u32 = 0x310;
        match cmd_id {
            Some(0) => {
                let entries = self.fs_dirs.remove(&key).unwrap_or_default();
                if let Some(buf) = self.ipc_output_buffer_addr(tls, 0) {
                    for (i, entry) in entries.iter().enumerate() {
                        let base = buf.wrapping_add(i as u32 * ENTRY_SIZE);
                        let name = entry.name.as_bytes();
                        for j in 0..0x301u32 {
                            let byte = name.get(j as usize).copied().unwrap_or(0);
                            self.mem.write_u8(base.wrapping_add(j), byte)?;
                        }
                        self.mem.write_u8(base.wrapping_add(0x304), entry.kind)?;
                        self.mem.write_u64(base.wrapping_add(0x308), entry.size)?;
                    }
                }
                let count = entries.len() as u64;
                self.write_ipc_response(tls, 0, &[], &count.to_le_bytes(), &[])
            }
            Some(1) => {
                let count = self.fs_dirs.get(&key).map(|v| v.len() as u64).unwrap_or(0);
                self.write_ipc_response(tls, 0, &[], &count.to_le_bytes(), &[])
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// `IFile`: cmd 0 = Read, 1 = Write, 2 = Flush, 3 = SetSize, 4 = GetSize.
    pub(crate) fn fs_file_request(
        &mut self,
        tls: u32,
        cmd_id: Option<u32>,
        key: u64,
    ) -> Result<()> {
        self.fs_traced(
            "file",
            cmd_id,
            file_command,
            false,
            |cpu| {
                let path = cpu.fs_files.get(&key).map_or("", String::as_str);
                format!(" {path:?} on {}", mount_text(cpu.mount_of(key)))
            },
            |cpu| cpu.fs_file_dispatch(tls, cmd_id, key),
        )
    }

    fn fs_file_dispatch(&mut self, tls: u32, cmd_id: Option<u32>, key: u64) -> Result<()> {
        let path = self.fs_files.get(&key).cloned().unwrap_or_default();
        let mount = self.mount_of(key);
        match cmd_id {
            // Read(u32 option, u64 offset, u64 size) -> u64 bytes_read
            Some(0) => {
                let data = self.ipc_request_data(tls);
                let offset = self.mem.read_u64(data.wrapping_add(8))?;
                let requested = self.mem.read_u64(data.wrapping_add(0x10))? as usize;
                let mut buf = vec![0u8; requested.min(1 << 24)];
                let read = self
                    .vfs_for(mount)
                    .read(&path, offset, &mut buf)
                    .unwrap_or(0);
                if crate::trace::enabled(Trace::Fs) {
                    crate::traceln!(
                        "[fs-file] read path={:?} offset={:#x} size={:#x} -> {:#x} buf={:?}",
                        path,
                        offset,
                        requested,
                        read,
                        self.ipc_output_buffer_addr(tls, 0)
                    );
                }
                if let Some(addr) = self.ipc_output_buffer_addr(tls, 0) {
                    self.mem.write_bytes(addr, &buf[..read])?;
                }
                self.fs_activity.read(&file_text(mount, &path), read as u64);
                self.write_ipc_response(tls, 0, &[], &(read as u64).to_le_bytes(), &[])
            }
            // Write(u32 option, s64 offset, u64 size); the Flush option bit is a no-op.
            Some(1) => {
                let data = self.ipc_request_data(tls);
                let offset = self.mem.read_u64(data.wrapping_add(8))?;
                let requested = self.mem.read_u64(data.wrapping_add(0x10))?;
                let bytes = match self.ipc_send_buffer(tls, 0) {
                    Some((addr, len)) => self.read_bytes(addr, (len as u64).min(requested) as u32),
                    None => Vec::new(),
                };
                if crate::trace::enabled(Trace::Fs) {
                    crate::traceln!(
                        "[fs-file] write path={:?} offset={:#x} size={:#x} -> {:#x}",
                        path,
                        offset,
                        requested,
                        bytes.len()
                    );
                }
                self.fs_activity
                    .wrote(&file_text(mount, &path), bytes.len() as u64);
                match self.vfs_for(mount).write(&path, offset, &bytes) {
                    Some(_) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                    None => self.write_ipc_response(tls, PATH_NOT_FOUND, &[], &[], &[]),
                }
            }
            // Flush
            Some(2) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // SetSize(s64 size), how `fsdev` implements `O_TRUNC`.
            Some(3) => {
                let size = self.mem.read_u64(self.ipc_request_data(tls))?;
                if self.vfs_for(mount).set_size(&path, size) {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                } else {
                    self.write_ipc_response(tls, PATH_NOT_FOUND, &[], &[], &[])
                }
            }
            // GetSize -> u64
            Some(4) => {
                let size = self.vfs_for(mount).size(&path).unwrap_or(0);
                self.write_ipc_response(tls, 0, &[], &size.to_le_bytes(), &[])
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// Reply with a save-data (size, journal) pair of s64s.
    pub(crate) fn write_save_data_pair(
        &mut self,
        tls: u32,
        size: i64,
        journal_size: i64,
    ) -> Result<()> {
        let mut sizes = Vec::with_capacity(16);
        sizes.extend_from_slice(&size.to_le_bytes());
        sizes.extend_from_slice(&journal_size.to_le_bytes());
        self.write_ipc_response(tls, 0, &[], &sizes, &[])
    }
}

#[cfg(test)]
mod tests {
    use crate::cpu::Cpu;
    use crate::kernel::ipc::testing::*;

    /// An `fsp-srv` command that hands back an object must not answer a bare success.
    #[test]
    fn an_unimplemented_open_reports_a_failure_rather_than_a_null_object() {
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        for cmd in [2, 7, 8, 9, 12, 17, 30, 31] {
            write_request(&mut cpu, cmd, &[]);
            cpu.fsp_srv_request(TLS, Some(cmd), 1).unwrap();
            let result = cpu.mem.read_u32(TLS + 0x18).unwrap();
            assert_ne!(result, 0, "fsp-srv cmd {cmd} answered a bare success");
        }
    }

    /// Commands with nothing to hand back keep their fabricated success.
    #[test]
    fn a_setter_shaped_command_still_succeeds() {
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        write_request(&mut cpu, 1003, &[]);
        cpu.fsp_srv_request(TLS, Some(1003), 1).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0);
    }

    #[test]
    fn a_file_written_through_ifile_reads_back() {
        let key = Cpu::object_key(9, 1);
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        cpu.mem.map_zero(0x3000, 0x100).unwrap();
        assert!(cpu.fs.create_file("/switch/cfg.json", 0));
        cpu.fs_files.insert(key, "/switch/cfg.json".to_owned());

        // fsFileWrite { u32 option, u32 pad, s64 offset, u64 size }
        for (i, &byte) in br#"{"v":5}"#.iter().enumerate() {
            cpu.mem.write_u8(0x3000 + i as u32, byte).unwrap();
        }
        let mut payload = [0u8; 0x18];
        payload[8..16].copy_from_slice(&0u64.to_le_bytes());
        payload[16..24].copy_from_slice(&7u64.to_le_bytes());
        write_map_buffer_request(&mut cpu, 1, &payload, 0x3000, 7, true);
        cpu.fs_file_request(TLS, Some(1), key).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.fs.file("/switch/cfg.json"), Some(&br#"{"v":5}"#[..]));
        let files = cpu.fs_activity.take_files();
        assert_eq!(
            files
                .get("sdmc:/switch/cfg.json")
                .map(|io| (io.writes, io.write_bytes)),
            Some((1, 7)),
            "the write is tallied under the file it went to: {files:?}"
        );

        write_request(&mut cpu, 4, &[]);
        cpu.fs_file_request(TLS, Some(4), key).unwrap();
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), 7);

        for (i, &byte) in b"!!".iter().enumerate() {
            cpu.mem.write_u8(0x3000 + i as u32, byte).unwrap();
        }
        let mut payload = [0u8; 0x18];
        payload[8..16].copy_from_slice(&7u64.to_le_bytes());
        payload[16..24].copy_from_slice(&2u64.to_le_bytes());
        write_map_buffer_request(&mut cpu, 1, &payload, 0x3000, 2, true);
        cpu.fs_file_request(TLS, Some(1), key).unwrap();
        assert_eq!(cpu.fs.size("/switch/cfg.json"), Some(9));

        write_request(&mut cpu, 3, &3u64.to_le_bytes());
        cpu.fs_file_request(TLS, Some(3), key).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.fs.file("/switch/cfg.json"), Some(&br#"{"v"#[..]));

        // A handle whose file is gone reports an error.
        cpu.fs.remove("/switch/cfg.json");
        write_request(&mut cpu, 3, &0u64.to_le_bytes());
        cpu.fs_file_request(TLS, Some(3), key).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x18).unwrap(),
            2 | (1 << 9),
            "path not found"
        );
    }

    #[test]
    fn create_file_on_one_that_exists_reports_it_rather_than_emptying_it() {
        const PATH_ALREADY_EXISTS: u32 = 2 | (2 << 9);
        // CreateFile(option, size)
        let mut payload = [0u8; 0x10];
        payload[8..16].copy_from_slice(&4u64.to_le_bytes());
        let mut cpu = request_with_path(0, "sdmc:/switch/cfg.json", &payload);
        cpu.record_handle(9, "fsp-srv");
        cpu.fs_request(TLS, Some(0), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "created");
        assert_eq!(cpu.fs.file("/switch/cfg.json"), Some(&[0u8; 4][..]));

        // Creating it again fails and leaves the contents alone.
        cpu.fs.write("/switch/cfg.json", 0, b"{}!!").unwrap();
        write_path_request(&mut cpu, 0, "sdmc:/switch/cfg.json", &payload);
        cpu.fs_request(TLS, Some(0), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), PATH_ALREADY_EXISTS);
        assert_eq!(cpu.fs.file("/switch/cfg.json"), Some(&b"{}!!"[..]));
    }

    #[test]
    fn the_access_log_mode_a_title_sets_is_the_one_it_reads_back() {
        // GetGlobalAccessLogMode defaults to off.
        let mut cpu = request(false, 1005, &[]);
        cpu.record_handle(9, "fsp-srv");
        cpu.fsp_srv_request(TLS, Some(1005), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "off until set");

        // SetGlobalAccessLogMode(2) must read back.
        write_request(&mut cpu, 1004, &2u32.to_le_bytes());
        cpu.fsp_srv_request(TLS, Some(1004), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");

        write_request(&mut cpu, 1005, &[]);
        cpu.fsp_srv_request(TLS, Some(1005), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x20).unwrap(),
            2,
            "the mode that was set"
        );

        write_request(&mut cpu, 1016, &[0u8; 0x10]);
        cpu.fsp_srv_request(TLS, Some(1016), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
    }

    #[test]
    fn a_console_with_no_save_data_ends_the_scan_on_the_first_read() {
        // OpenSaveDataInfoReaderBySaveDataSpaceId hands back an out-object.
        let mut cpu = request(false, 61, &[1u8]);
        cpu.record_handle(9, "fsp-srv");
        cpu.fsp_srv_request(TLS, Some(61), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        let reader = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;
        assert_eq!(cpu.service_name(reader), Some("fsp-srv-save-info-reader"));

        // Zero entries ends the caller's scan.
        write_request(&mut cpu, 0, &[]);
        cpu.fs_save_data_info_reader_request(TLS, reader, Some(0))
            .unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), 0, "entries");

        // Mounting a save creates it on first open.
        write_request(&mut cpu, 52, &[0u8; 0x40]);
        cpu.fsp_srv_request(TLS, Some(52), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        let saves = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;
        assert_eq!(cpu.service_name(saves), Some("fsp-srv-fs"));
    }

    #[test]
    fn each_user_has_their_own_save_of_a_title() {
        // Same title, two users and no user: three saves.
        const TITLE: u64 = 0x0100_0000_0000_1000;
        let mut cpu = request(false, 51, &[]);
        cpu.record_handle(9, "fsp-srv");
        for user in [*b"ann-uid-00000001", *b"ben-uid-00000002", [0; 16]] {
            let mut attribute = [0u8; 0x48];
            attribute[8..0x10].copy_from_slice(&TITLE.to_le_bytes());
            attribute[0x10..0x20].copy_from_slice(&user);
            write_request(&mut cpu, 51, &attribute);
            cpu.fsp_srv_request(TLS, Some(51), 9).unwrap();
            assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "opening the save");
        }
        let mut keys = cpu.save_keys();
        keys.sort();
        assert_eq!(
            keys,
            vec![
                crate::cpu::SaveKey::shared(TITLE),
                crate::cpu::SaveKey {
                    id: TITLE,
                    user: *b"ann-uid-00000001"
                },
                crate::cpu::SaveKey {
                    id: TITLE,
                    user: *b"ben-uid-00000002"
                },
            ]
        );
    }

    #[test]
    fn a_save_is_a_different_storage_from_the_sd_card() {
        // Saves and the SD card must not be confused.
        const SAVE_ID: u64 = 0x0100_0000_0000_1000;
        /// System save id offset within the space id plus `SaveDataAttribute`.
        const SYSTEM_SAVE_ID_AT: usize = 8 + 0x18;
        let mut cpu = request(false, 52, &[]);
        cpu.record_handle(9, "fsp-srv");

        let mut attribute = [0u8; 0x48];
        attribute[SYSTEM_SAVE_ID_AT..SYSTEM_SAVE_ID_AT + 8].copy_from_slice(&SAVE_ID.to_le_bytes());
        write_request(&mut cpu, 52, &attribute);
        cpu.fsp_srv_request(TLS, Some(52), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "opening the save");
        let saves = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;
        assert_eq!(cpu.service_name(saves), Some("fsp-srv-fs"));

        write_path_request(&mut cpu, 2, "/settings", &[]);
        cpu.fs_request(TLS, Some(2), saves).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x18).unwrap(),
            0,
            "creating a directory"
        );
        assert_eq!(
            cpu.save_data(crate::cpu::SaveKey::shared(SAVE_ID))
                .and_then(|save| save.entry_type("/settings")),
            Some(crate::vfs::ENTRY_TYPE_DIR),
            "the directory should be in the save"
        );
        assert_eq!(
            cpu.fs.entry_type("/settings"),
            None,
            "and must not have landed on the SD card"
        );

        // Reopening the same id finds it again.
        write_request(&mut cpu, 52, &attribute);
        cpu.fsp_srv_request(TLS, Some(52), 9).unwrap();
        let reopened = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;
        write_path_request(&mut cpu, 7, "/settings", &[]);
        cpu.fs_request(TLS, Some(7), reopened).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "GetEntryType");
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x20).unwrap(),
            u32::from(crate::vfs::ENTRY_TYPE_DIR)
        );
    }

    /// Open a save scan with `cmd` and `payload`, then read it one entry at a time.
    fn scan_saves(cpu: &mut Cpu, cmd: u32, payload: &[u8]) -> Vec<[u8; 0x60]> {
        write_request(cpu, cmd, payload);
        cpu.fsp_srv_request(TLS, Some(cmd), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "opening the scan");
        let reader = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;
        let mut infos = Vec::new();
        loop {
            write_map_buffer_request(cpu, 0, &[], 0x3000, 0x60, false);
            cpu.fs_save_data_info_reader_request(TLS, reader, Some(0))
                .unwrap();
            match cpu.mem.read_u64(TLS + 0x20).unwrap() {
                0 => return infos,
                1 => {
                    let mut info = [0u8; 0x60];
                    for (i, byte) in info.iter_mut().enumerate() {
                        *byte = cpu.mem.read_u8(0x3000 + i as u32).unwrap();
                    }
                    infos.push(info);
                }
                n => panic!("{n} entries in a buffer that holds one"),
            }
        }
    }

    fn seeded_saves() -> Cpu {
        let mut cpu = request(false, 0, &[]);
        cpu.mem.map_zero(0x3000, 0x100).unwrap();
        cpu.record_handle(9, "fsp-srv");
        for key in [
            crate::cpu::SaveKey {
                id: 0x0100_0000_0000_1000,
                user: *b"ann-uid-00000001",
            },
            crate::cpu::SaveKey {
                id: 0x0100_0000_0000_2000,
                user: *b"ben-uid-00000002",
            },
            crate::cpu::SaveKey::shared(0x0100_0000_0000_1000),
            crate::cpu::SaveKey::shared(0x8000_0000_0000_0010),
        ] {
            cpu.save_data_mut(key);
        }
        cpu
    }

    #[test]
    fn the_save_scan_lists_each_space_s_saves() {
        let mut cpu = seeded_saves();
        let user = scan_saves(&mut cpu, 61, &[1]);
        let listed: Vec<_> = user
            .iter()
            .map(|info| {
                let application = u64::from_le_bytes(info[0x28..0x30].try_into().unwrap());
                (info[8], info[9], application, info[0x10..0x20].to_vec())
            })
            .collect();
        assert_eq!(
            listed,
            vec![
                (1, 3, 0x0100_0000_0000_1000, vec![0; 16]),
                (1, 1, 0x0100_0000_0000_1000, b"ann-uid-00000001".to_vec()),
                (1, 1, 0x0100_0000_0000_2000, b"ben-uid-00000002".to_vec()),
            ],
            "(space, type, application, uid)"
        );

        let system = scan_saves(&mut cpu, 61, &[0]);
        assert_eq!(system.len(), 1);
        assert_eq!(
            (system[0][8], system[0][9]),
            (0, 0),
            "system space and type"
        );
        assert_eq!(
            u64::from_le_bytes(system[0][0x20..0x28].try_into().unwrap()),
            0x8000_0000_0000_0010,
            "system save id"
        );

        assert_eq!(scan_saves(&mut cpu, 60, &[]).len(), 4, "every space");
        assert!(
            scan_saves(&mut cpu, 62, &[1]).is_empty(),
            "no cache storage"
        );
    }

    #[test]
    fn the_filtered_save_scan_keeps_only_matching_saves() {
        let mut cpu = seeded_saves();
        // u8 space, pad, then FsSaveDataFilter: flags, rank, pad, FsSaveDataAttribute.
        let mut payload = [0u8; 0x50];
        payload[0] = 1;
        payload[8 + 2] = 1; // filter_by_user_id
        payload[0x10 + 8..0x10 + 0x18].copy_from_slice(b"ben-uid-00000002");
        let infos = scan_saves(&mut cpu, 68, &payload);
        assert_eq!(infos.len(), 1);
        assert_eq!(&infos[0][0x10..0x20], b"ben-uid-00000002");

        payload[8 + 2] = 0;
        payload[8 + 1] = 1; // filter_by_save_data_type
        payload[0x10 + 0x20] = 3; // Device
        let infos = scan_saves(&mut cpu, 68, &payload);
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0][9], 3);
    }

    #[test]
    fn the_filtered_save_scan_is_a_reader_too() {
        // 68 = OpenSaveDataInfoReaderWithFilter hands out the same reader.
        let mut payload = [0u8; 0x50];
        payload[0] = 1; // FsSaveDataSpaceId::User
        let mut cpu = request(false, 68, &payload);
        cpu.record_handle(9, "fsp-srv");
        cpu.fsp_srv_request(TLS, Some(68), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        let reader = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;
        assert_ne!(reader, 0, "reader session");
        assert_eq!(cpu.service_name(reader), Some("fsp-srv-save-info-reader"));

        write_request(&mut cpu, 0, &[]);
        cpu.fs_save_data_info_reader_request(TLS, reader, Some(0))
            .unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), 0, "entries");
    }

    #[test]
    fn the_device_operator_reports_the_card_this_console_has_and_the_one_it_has_not() {
        // OpenDeviceOperator hands back an object.
        let mut cpu = request(false, 400, &[]);
        cpu.record_handle(9, "fsp-srv");
        cpu.fsp_srv_request(TLS, Some(400), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        let operator = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;
        assert_ne!(operator, 0, "operator session");
        assert_eq!(cpu.service_name(operator), Some("fsp-srv-device-operator"));

        write_request(&mut cpu, 0, &[]);
        cpu.fs_device_operator_request(TLS, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 1, "sd card inserted");

        write_request(&mut cpu, 200, &[]);
        cpu.fs_device_operator_request(TLS, Some(200)).unwrap();
        assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 0, "no game card");

        write_request(&mut cpu, 3, &[]);
        cpu.fs_device_operator_request(TLS, Some(3)).unwrap();
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), super::SD_TOTAL_SPACE);

        // GetGameCardHandle is refused.
        const UNKNOWN_COMMAND_ID: u32 = 10 | (221 << 9);
        write_request(&mut cpu, 202, &[]);
        cpu.fs_device_operator_request(TLS, Some(202)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), UNKNOWN_COMMAND_ID);
    }

    #[test]
    fn a_device_operator_register_is_written_over_whatever_the_buffer_held() {
        // GetSdCardCid writes the full 0x10 bytes.
        const BUFFER: u32 = 0x4000;
        let mut cpu = request_with_recv_buffer(2, &0x10u64.to_le_bytes(), BUFFER, 0x10);
        cpu.mem.map_zero(BUFFER, 0x100).unwrap();
        for offset in 0..0x20 {
            cpu.mem.write_u8(BUFFER + offset, 0xAA).unwrap();
        }
        cpu.fs_device_operator_request(TLS, Some(2)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(
            cpu.read_bytes(BUFFER, 0x10),
            vec![0u8; 0x10],
            "no card, so no serial"
        );
        assert_eq!(
            cpu.read_bytes(BUFFER + 0x10, 0x10),
            vec![0xAA; 0x10],
            "past the buffer"
        );
    }

    #[test]
    fn the_speed_emulation_mode_a_caller_sets_is_the_one_it_reads_back() {
        let mut cpu = request(false, 301, &[]);
        cpu.fs_device_operator_request(TLS, Some(301)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "none until set");

        write_request(&mut cpu, 300, &2u32.to_le_bytes());
        cpu.fs_device_operator_request(TLS, Some(300)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");

        write_request(&mut cpu, 301, &[]);
        cpu.fs_device_operator_request(TLS, Some(301)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x20).unwrap(),
            2,
            "SpeedEmulationMode::Slower"
        );
    }

    #[test]
    fn each_card_slot_has_its_own_detection_event_and_neither_ever_fires() {
        // OpenSdCardDetectionEventNotifier hands back an object.
        let mut cpu = request(false, 500, &[]);
        cpu.record_handle(9, "fsp-srv");
        cpu.fsp_srv_request(TLS, Some(500), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        let sd = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;
        assert_eq!(cpu.service_name(sd), Some("fsp-srv-sd-detection"));

        write_request(&mut cpu, 501, &[]);
        cpu.fsp_srv_request(TLS, Some(501), 9).unwrap();
        let game_card = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;
        assert_eq!(
            cpu.service_name(game_card),
            Some("fsp-srv-gamecard-detection")
        );

        // One event per slot.
        write_request(&mut cpu, 0, &[]);
        cpu.fs_detection_notifier_request(TLS, sd, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        let sd_event = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;
        assert_ne!(sd_event, 0, "sd detection event");

        write_request(&mut cpu, 0, &[]);
        cpu.fs_detection_notifier_request(TLS, game_card, Some(0))
            .unwrap();
        let game_card_event = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;
        assert_ne!(game_card_event, sd_event, "one event per slot");

        assert_eq!(cpu.event_signaled(sd_event), Some(false));
        assert_eq!(cpu.event_signaled(game_card_event), Some(false));

        // Asking twice returns the same event.
        write_request(&mut cpu, 0, &[]);
        cpu.fs_detection_notifier_request(TLS, sd, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64, sd_event);
    }

    /// Path operations are journalled whether or not `TRACE_FS` is on.
    #[test]
    fn a_path_operation_is_journalled_with_its_result() {
        let mut cpu = request_with_path(7, "sdmc:/missing.bin", &[]);
        cpu.record_handle(9, "fsp-srv");
        cpu.fs_request(TLS, Some(7), 9).unwrap();
        let (journal, dropped) = cpu.fs_activity.take_journal();
        assert_eq!(dropped, 0);
        assert_eq!(journal.len(), 1, "{journal:?}");
        assert!(
            journal[0].starts_with("fs GetEntryType \"/missing.bin\" on sdmc ->"),
            "{journal:?}"
        );
        assert!(journal[0].ends_with("-> 2002-0001 (0x202)"), "{journal:?}");
        assert_eq!(cpu.fs_activity.failures, 1);
        assert!(
            cpu.fs_activity.take_journal().0.is_empty(),
            "taken, not read"
        );
    }

    #[test]
    fn a_traced_result_reads_as_horizon_prints_it() {
        assert_eq!(super::result_text(0), "ok");
        assert_eq!(
            super::result_text(super::PATH_NOT_FOUND),
            "2002-0001 (0x202)"
        );
        assert_eq!(super::result_text(2 | (3005 << 9)), "2002-3005 (0x177a02)");
    }

    /// `IStorage::Read` refuses a range past the end.
    #[test]
    fn a_storage_read_past_the_end_is_refused_rather_than_clamped() {
        const BUFFER: u32 = 0x4000;
        const OUT_OF_RANGE: u32 = 2 | (3005 << 9);
        let romfs: Vec<u8> = (0..=0xFFu8).collect();

        let read = |offset: u64, size: u64| {
            let mut payload = [0u8; 0x10];
            payload[..8].copy_from_slice(&offset.to_le_bytes());
            payload[8..].copy_from_slice(&size.to_le_bytes());
            payload
        };

        let mut cpu = request_with_recv_buffer(0, &read(0x10, 0x20), BUFFER, 0x40);
        cpu.mem.map_zero(BUFFER, 0x100).unwrap();
        cpu.set_romfs(romfs.clone());
        cpu.fs_storage_request(TLS, 1, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.read_bytes(BUFFER, 0x20), romfs[0x10..0x30]);

        write_map_buffer_request(&mut cpu, 0, &read(0, 0x100), BUFFER, 0x100, false);
        cpu.fs_storage_request(TLS, 1, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.read_bytes(BUFFER, 0x100), romfs);

        // One byte past is refused, leaving the buffer untouched.
        for offset in 0..0x100 {
            cpu.mem.write_u8(BUFFER + offset, 0xAA).unwrap();
        }
        write_map_buffer_request(&mut cpu, 0, &read(0xF0, 0x11), BUFFER, 0x40, false);
        cpu.fs_storage_request(TLS, 1, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), OUT_OF_RANGE);
        assert_eq!(cpu.read_bytes(BUFFER, 0x20), vec![0xAA; 0x20]);

        write_map_buffer_request(&mut cpu, 0, &read(0x100, 1), BUFFER, 0x40, false);
        cpu.fs_storage_request(TLS, 1, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), OUT_OF_RANGE);

        write_request(&mut cpu, 4, &[]);
        cpu.fs_storage_request(TLS, 1, Some(4)).unwrap();
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), 0x100);

        // Only served reads are tallied.
        let files = cpu.fs_activity.take_files();
        assert_eq!(
            files.get("romfs"),
            Some(&super::FileIo {
                reads: 2,
                read_bytes: 0x120,
                writes: 0,
                write_bytes: 0
            }),
            "{files:?}"
        );
    }
}
