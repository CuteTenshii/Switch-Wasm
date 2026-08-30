//! `fsp-srv` and the objects it hands out: `IFileSystem`, `IFile`,
//! `IDirectory`, `IStorage`, and the save-data interfaces beside them.
//!
//! The filesystems themselves are [`crate::vfs`]; the RomFS a title reads
//! through `IStorage` is never staged in memory (see AGENTS.md), so a read
//! here copies through a staging buffer out of a [`crate::source::ByteSource`].

use super::Cpu;
use crate::Result;

/// The size of the emulated SD card, and how much of it is free. `ns` reports
/// these from two interfaces and a caller does arithmetic between them, so
/// they are one pair of numbers rather than two.
pub(super) const SD_TOTAL_SPACE: u64 = 32 << 30;

pub(super) const SD_FREE_SPACE: u64 = 16 << 30;

/// `nn::fs::SdCardSpeedMode::Sdr104` and `nn::fs::MmcSpeedMode::Hs400`: the
/// fastest mode each bus negotiates. Nothing here is on a bus, but 0 in either
/// enum is `Identification` — a device that never finished initialising, which
/// is a fault rather than a missing measurement.
const SD_CARD_SPEED_MODE: i64 = 6;

const MMC_SPEED_MODE: i64 = 4;

/// The emulated eMMC: its user area is the same storage the SD card reports,
/// because both are the same host memory, and the two boot partitions beside
/// it are the 4 MiB a Tegra X1's eMMC carries.
const MMC_USER_AREA_SIZE: i64 = SD_TOTAL_SPACE as i64;

const MMC_BOOT_PARTITION_SIZE: i64 = 4 << 20;

/// The two `IEventNotifier`s `fsp-srv` hands out, named apart because the
/// event behind each one is a different slot's — and one handler serves both.
const SD_CARD_DETECTION: &str = "fsp-srv-sd-detection";

const GAME_CARD_DETECTION: &str = "fsp-srv-gamecard-detection";

/// What the save-data commands report before anything has read the running
/// title's NACP: 64 MiB of save data and 16 MiB of journal.
///
/// The real figures are per-title and only the Control NCA knows them —
/// Tomodachi Life declares 54 MiB and 10 MiB — so this is a fallback, and it
/// is deliberately generous. Reporting *more* than a title needs costs
/// nothing, since nothing here enforces a quota; reporting less is the answer
/// that hurts, because a title that reads a quota its save does not fit into
/// is a title that does not write one.
pub(crate) const DEFAULT_SAVE_DATA_SIZE: i64 = 0x400_0000;

pub(crate) const DEFAULT_SAVE_DATA_JOURNAL_SIZE: i64 = 0x100_0000;

/// How many cache storages a title may address when its NACP has not said.
/// One is enough for a title that asks without declaring: cache storage is
/// scratch space, and a title that wanted several would have declared them.
pub(crate) const DEFAULT_CACHE_STORAGE_INDEX_MAX: i32 = 1;

/// What the running title is allowed to store, as its own NACP declares it.
///
/// Every figure here is reported by one `IApplicationFunctions` command and
/// changes nothing else: the emulated NAND has no quota and grows with
/// whatever a title writes. They matter because a title reads them *before*
/// it writes — to decide whether its save fits, whether it may grow one, and
/// how many cache storages it may create — and acts on the answer.
///
/// A zero that came from a real NACP is passed through rather than corrected.
/// A title that declares no ceiling is one that never extends its save, and 0
/// is what says so; inventing headroom for it would be answering a question it
/// did not ask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaveDataQuota {
    /// The size the user-account save is created at, and its journal.
    pub size: i64,
    pub journal_size: i64,
    /// How far either may be extended, which is a separate NACP field and
    /// commonly 0 even when the size above is not.
    pub size_max: i64,
    pub journal_size_max: i64,
    /// The same pair for the console-wide save, which is not per-profile.
    pub device_size_max: i64,
    pub device_journal_size_max: i64,
    /// Cache storage: the ceiling on one storage's data and journal together,
    /// and how many of them the title may address.
    pub cache_storage_size_max: i64,
    pub cache_storage_index_max: i32,
}

impl Default for SaveDataQuota {
    fn default() -> SaveDataQuota {
        SaveDataQuota {
            size: DEFAULT_SAVE_DATA_SIZE,
            journal_size: DEFAULT_SAVE_DATA_JOURNAL_SIZE,
            // Room to grow, rather than the 0 that would tell a title its save
            // is already at its ceiling. Nothing here is measuring.
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

impl Cpu {
    pub(super) fn fsp_srv_request(
        &mut self,
        tls: u32,
        cmd_id: Option<u32>,
        handle: u64,
    ) -> Result<()> {
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
            // 0 = ConvertToDomain: hand back a fresh domain object id so the
            // session becomes a domain (libnx's serviceConvertToDomain reads it
            // from the out data). All later fsp-srv requests then carry the
            // object id in the CmifDomainInHeader.
            Some(0) => {
                let obj = self.alloc_domain_object();
                self.record_domain_object(handle, obj, "fsp-srv");
                self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
            }
            Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // 18 = fsOpenSdCardFileSystem, 11 = fsOpenBisFileSystem: both hand
            // out an FsFileSystem session as a domain out-object.
            Some(18) | Some(11) => {
                self.reply_with_interface(tls, handle, "fsp-srv-fs")?;
                Ok(())
            }
            // 200 = OpenDataStorageByCurrentProcess: hands back the calling
            // title's own RomFS as a raw `IStorage` (offset/size reads only —
            // no paths). libnx's `romfsMount`/`nn::fs::MountRom` parse the
            // RomFS header and directory/file tables entirely in guest code
            // against this; the emulator only has to serve byte ranges.
            Some(200) => {
                if self.romfs.is_none() {
                    // No NCA was decrypted this session (homebrew, or a
                    // title with no RomFS section) — report "not found"
                    // rather than handing out a storage backed by nothing.
                    const PATH_NOT_FOUND: u32 = 2 | (1 << 9);
                    return self.write_ipc_response(tls, PATH_NOT_FOUND, &[], &[], &[]);
                }
                self.reply_with_interface(tls, handle, "fsp-srv-storage")?;
                Ok(())
            }
            // 202 = OpenDataStorageByDataId(u8 storage_id, u64 data_id):
            // content that is *not* the calling title's own — an applet's
            // shared assets, the system's Mii and amiibo resources. On a real
            // console each is a separate Data NCA on the NAND, mounted by
            // data id; here the host registers whichever it was given
            // ([`Cpu::add_data_archive`]).
            //
            // A data id nobody registered is reported missing. Handing back an
            // empty storage instead would be answered as a zero-byte archive,
            // which is what the caller then blames — `cabinet` reported
            // `2002-3005` against its own resource load rather than against
            // the archive not being there.
            Some(202) => {
                let data = self.ipc_request_data(tls);
                let data_id = self.mem.read_u64(data.wrapping_add(8))?;
                if !self.data_archives.contains_key(&data_id) {
                    self.diagnostic(&format!(
                        "[fs] no system data archive registered for data id {data_id:016x}"
                    ));
                    const PATH_NOT_FOUND: u32 = 2 | (1 << 9);
                    return self.write_ipc_response(tls, PATH_NOT_FOUND, &[], &[], &[]);
                }
                let key = self.reply_with_interface(tls, handle, "fsp-srv-storage")?;
                self.fs_storage_archive.insert(key, data_id);
                if crate::env_flag!("TRACE_IPC") {
                    let size = self.storage_source(Some(data_id)).map_or(0, |s| s.len());
                    eprintln!("[fs] data archive {data_id:016x} -> {size:#x} bytes");
                }
                Ok(())
            }
            // 203 = OpenPatchDataStorageByCurrentProcess: the RomFS of the
            // title's *update*, which is a second NCA this emulator does not
            // have — it boots the base Program NCA and nothing beside it.
            // Saying so is the whole implementation, but it has to be said in
            // the one shape the caller recognises. `nn::fs::QueryMountRomCacheSize`
            // opens the base storage (200) and then this one, and treats only
            // fs's 2002-1001 and 2002-1002 (`TargetNotFound`) as "there is no
            // patch, use the base alone"; every other Result — a success most
            // of all — it acts on.
            //
            // Which is how the catch-all below stopped Just Dance 2017 dead.
            // A bare success carries no out-object, so `nnSdk`'s
            // `SharedPointer<IStorage>` stayed null; the SDK wrapped the null
            // in a `StorageServiceObjectAdapter` regardless, and the first
            // `Read` through it loaded a vtable from address 0 and branched
            // there. The fault it produced named `pc=0` and nothing else,
            // 190 million instructions after the request that caused it.
            Some(203) => {
                const TARGET_NOT_FOUND: u32 = 2 | (1002 << 9);
                self.write_ipc_response(tls, TARGET_NOT_FOUND, &[], &[], &[])
            }
            // 51 = OpenSaveDataFileSystem, 52 = ...BySystemSaveDataId,
            // 53 = OpenReadOnlySaveDataFileSystem; 22 and 23 are the two
            // Create forms. All of them address the NAND, which this console
            // now has: an `IFileSystem` over the save filed under the id the
            // request names.
            //
            // Every one of these used to report "not found", which is a thing
            // callers act on rather than shrug at — a title that cannot open
            // its save has nowhere to put anything, and the system applets
            // open theirs before they will do very much at all.
            Some(22) | Some(23) | Some(51) | Some(52) | Some(53) => {
                let id = self.save_data_id(tls);
                self.save_data_mut(id);
                // Create answers with a bare Result; Open hands back the
                // filesystem.
                if matches!(cmd_id, Some(22) | Some(23)) {
                    return self.write_ipc_response(tls, 0, &[], &[], &[]);
                }
                let key = self.reply_with_interface(tls, handle, "fsp-srv-fs")?;
                self.set_mount(key, Some(id));
                Ok(())
            }
            // 60 = OpenSaveDataInfoReader, 61 = ...BySaveDataSpaceId,
            // 62 = ...OnlyCacheStorage, 68 = ...WithFilter: the enumerator a
            // save manager walks to find what is on the console. All four hand
            // out the same `ISaveDataInfoReader`; the filter 68 takes (a save
            // type, a user id, a title id — which of them apply is a mask) only
            // narrows what it would report, and nothing is reported either way.
            //
            // 68 is the 6.0.0+ form, and it is the one a current JKSV opens.
            // It used to fall through to the catch-all below, which answers
            // with success and *no* out-object: `libnx` then read its reader
            // session handle out of a reply that had no handle in it and sent
            // `ReadSaveDataInfo` to handle 0 — the "<untracked session> cmd=0"
            // that the generic object-id reply answered with an object id the
            // caller read back as an entry count of several billion.
            Some(60) | Some(61) | Some(62) | Some(68) => {
                self.reply_with_interface(tls, handle, "fsp-srv-save-info-reader")?;
                Ok(())
            }
            // 400 = OpenDeviceOperator: the interface that answers for the
            // storage *devices* rather than the filesystems on them — whether
            // a card is in either slot, how big it is, and what the controller
            // has logged. It hands back an object, which is what the catch-all
            // below could not do: a caller reads one as a move handle, parses
            // the missing handle as 0, and makes its first call through a null
            // `SharedPointer` while the reply still says success.
            Some(400) => {
                self.reply_with_interface(tls, handle, "fsp-srv-device-operator")?;
                Ok(())
            }
            // 500 = OpenSdCardDetectionEventNotifier, 501 = ...GameCard...:
            // the other half of what the device operator above answers, for a
            // caller that would rather be told when a slot changes than ask.
            // Both hand back an `IEventNotifier`, and both used to be a bare
            // success — the same null out-object as 400, one command apart.
            Some(500) | Some(501) => {
                let name = match cmd_id {
                    Some(500) => SD_CARD_DETECTION,
                    _ => GAME_CARD_DETECTION,
                };
                self.reply_with_interface(tls, handle, name)?;
                Ok(())
            }
            // 1003 = DisableAutoSaveDataCreation. `ns` sends this once at
            // boot so that `fs` stops conjuring a save the moment a title
            // opens one, leaving the creation to `ns`'s own explicit call —
            // which is how a title that has never been launched is told its
            // save does not exist yet rather than handed an empty one.
            //
            // It is accepted and **not** honoured. Saves here are created on
            // open ([`Cpu::save_data_mut`]) and there is no installer to have
            // created them beforehand, so a console that obeyed this flag
            // would have no save for any title including the Home Menu's own
            // — the exact failure the flag exists to produce, against a NAND
            // that was never populated. It is answered rather than left to
            // the catch-all so that this is written down somewhere.
            Some(1003) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // 1004 = SetGlobalAccessLogMode(u32), 1005 = GetGlobalAccessLogMode.
            // `fs`'s access log is a development feature, and its mode is a
            // per-process setting the server stores and hands straight back.
            // Zero is "off", which is how a retail console boots and what this
            // reports until a title says otherwise.
            //
            // Neither command hands back an object, so the catch-all below was
            // survivable here: 1005's out word sits inside the section the
            // reply zeroes, so it read as "off" rather than as stale TLS. What
            // the catch-all could not do is *agree* with 1004 — the mode a
            // title had just set came back as zero, so a caller that reads its
            // own setting back to decide whether to keep building log strings
            // was told its request had been ignored, by a reply that claimed
            // to have honoured it.
            Some(1004) => {
                self.fs_access_log_mode = self.mem.read_u32(self.ipc_request_data(tls))?;
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(1005) => {
                let mode = self.fs_access_log_mode;
                self.write_ipc_response(tls, 0, &[], &mode.to_le_bytes(), &[])
            }
            // 1006 = OutputAccessLogToSdCard, 1014 = OutputMultiProgramTagAccessLog,
            // 1015 = FlushAccessLogOnSdCard, 1016 = OutputApplicationInfoAccessLog:
            // the writing end of that same log. Every one of them takes text or
            // a tag and answers with nothing but a `Result`, so accepting the
            // write and dropping it is the whole implementation — there is no
            // access log on this console for them to reach, and a caller only
            // sends them once 1005 has reported a mode that is not "off".
            Some(1006) | Some(1014) | Some(1015) | Some(1016) => {
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // Still a fabricated success — homebrew that only checks the
            // Result depends on it — but no longer a silent one. Every
            // command that reaches here is one whose out-object, out-handle
            // or out-value the caller is about to read as zero, and the line
            // this prints is the only warning it will get before that zero
            // surfaces somewhere else entirely.
            _ => {
                self.warn_no_implementation("fsp-srv", cmd_id);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
        }
    }

    /// `ISaveDataInfoReader`: cmd 0 = `ReadSaveDataInfo`, which fills an
    /// output buffer with `FsSaveDataInfo` entries and reports how many it
    /// wrote. A caller reads until it reports **zero**, which is the whole
    /// termination condition — there is no separate "end" signal.
    ///
    /// This console has no save data, so the first read is already the last
    /// one. Saying so is the entire implementation, and saying it *wrongly* is
    /// unusually expensive: the reader used to be a fabricated object id, and
    /// a fabricated success made every read look like it had returned more
    /// entries, so Checkpoint enumerated saves forever — 1434 rounds of
    /// mounting and scanning a save named after an all-zero title id, with no
    /// end in sight.
    pub(super) fn fs_save_data_info_reader_request(
        &mut self,
        tls: u32,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        match cmd_id {
            Some(0) => self.write_ipc_response(tls, 0, &[], &0i64.to_le_bytes(), &[]),
            _ => self.unimplemented_command(tls, "fsp-srv-save-info-reader", cmd_id),
        }
    }

    /// `IDeviceOperator`: what a console's two storage devices — the SD card
    /// and the internal eMMC — report about themselves, and whether a game
    /// card is in the slot.
    ///
    /// Most of it is diagnostic, collected into a crash report rather than
    /// acted on. The presence bools are the exception, and they are the two
    /// answers this console is certain of: there is an SD card
    /// ([`crate::vfs`]), and there is nothing that could hold a game card.
    pub(super) fn fs_device_operator_request(
        &mut self,
        tls: u32,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        /// A CID is 0x10 bytes and an eMMC's extended CSD 0x200, whatever
        /// width the caller sized its buffer at.
        const CID_SIZE: usize = 0x10;
        const EXTENDED_CSD_SIZE: usize = 0x200;
        match cmd_id {
            // 0 = IsSdCardInserted, 200 = IsGameCardInserted.
            Some(0) => self.write_ipc_response(tls, 0, &[], &[1u8], &[]),
            Some(200) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
            // 1 = GetSdCardSpeedMode, 101 = GetMmcSpeedMode -> s64.
            Some(1) => self.write_ipc_response(tls, 0, &[], &SD_CARD_SPEED_MODE.to_le_bytes(), &[]),
            Some(101) => self.write_ipc_response(tls, 0, &[], &MMC_SPEED_MODE.to_le_bytes(), &[]),
            // 2 = GetSdCardCid, 100 = GetMmcCid: a card identification
            // register into an out buffer, sized by an input s64. No physical
            // card stands behind either, so both are zero — still worth
            // writing, because a caller reads the full width back whether the
            // server filled it or not.
            Some(2) | Some(100) => {
                let requested = self.mem.read_u64(self.ipc_request_data(tls)).unwrap_or(0);
                self.write_out_buffer(tls, &[0u8; CID_SIZE], requested)?;
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // 3 = GetSdCardUserAreaSize, 4 = GetSdCardProtectedAreaSize -> s64.
            // The user area is the card `fsp-srv` and `ns` already report; the
            // protected area is the CPRM reserve, which this card has none of.
            Some(3) => {
                let size = SD_TOTAL_SPACE as i64;
                self.write_ipc_response(tls, 0, &[], &size.to_le_bytes(), &[])
            }
            Some(4) => self.write_ipc_response(tls, 0, &[], &0i64.to_le_bytes(), &[]),
            // 5 = GetAndClearSdCardErrorInfo, 113 = GetAndClearMmcErrorInfo ->
            // (StorageErrorInfo, s64 log size). Four failure counters and the
            // length of a log, all zero: nothing has failed, and there is no
            // controller that would have recorded it if it had.
            Some(5) | Some(113) => self.write_ipc_response(tls, 0, &[], &[0u8; 0x18], &[]),
            // 111 = GetMmcPartitionSize(MmcPartition) -> s64. 0 is the user
            // data this console's saves live on; 1 and 2 are the boot
            // partitions.
            Some(111) => {
                let size = match self.mem.read_u32(self.ipc_request_data(tls)).unwrap_or(0) {
                    0 => MMC_USER_AREA_SIZE,
                    _ => MMC_BOOT_PARTITION_SIZE,
                };
                self.write_ipc_response(tls, 0, &[], &size.to_le_bytes(), &[])
            }
            // 112 = GetMmcPatrolCount -> u32: how many times the background
            // scrub has swept the NAND. There is no scrub, so it never has.
            Some(112) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            // 114 = GetMmcExtendedCsd: the eMMC's 0x200-byte configuration
            // register. Zero reads as "not defined" in every field of it,
            // including the three bytes a NAND-health check looks at.
            Some(114) => {
                let requested = self.mem.read_u64(self.ipc_request_data(tls)).unwrap_or(0);
                self.write_out_buffer(tls, &[0u8; EXTENDED_CSD_SIZE], requested)?;
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // 115/116 = Suspend/ResumeMmcPatrol, 400/401 = Suspend/Resume-
            // SdmmcControl: stop and start machinery this console does not
            // run. Accepting them promises nothing, which is what separates
            // them from the commands refused below.
            Some(115) | Some(116) | Some(400) | Some(401) => {
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // 300 = SetSpeedEmulationMode, 301 = GetSpeedEmulationMode. No
            // read here is slowed to match, but the mode round-trips: a caller
            // whose setting reads back as `None` was told it had been refused.
            Some(300) => {
                self.fs_speed_emulation_mode =
                    self.mem.read_u32(self.ipc_request_data(tls)).unwrap_or(0);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(301) => {
                let mode = self.fs_speed_emulation_mode;
                self.write_ipc_response(tls, 0, &[], &mode.to_le_bytes(), &[])
            }
            // The rest of the game-card commands, and the erase and
            // direct-write ones beside them. A caller reaches those only once
            // 200 has said a card is there, which it never does — so they are
            // refused rather than answered with a fabricated card.
            _ => self.unimplemented_command(tls, "fsp-srv-device-operator", cmd_id),
        }
    }

    /// `IEventNotifier`: cmd 0 = GetEventHandle, the event `fs` signals when a
    /// card arrives in a slot or leaves one.
    ///
    /// It goes out **dark** and stays dark. A waiter here is waiting for a
    /// *change* — the SD card is already mounted at boot and never leaves, and
    /// there is no game card slot to change at all — so this is an event that
    /// genuinely never fires, rather than one that should have. The same
    /// reasoning `ns` applies to its own media events.
    ///
    /// One event per slot, handed back to every caller that asks: a poller
    /// given a fresh handle per call leaks one per call.
    pub(super) fn fs_detection_notifier_request(
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
            // The handle is a **copy** handle: `fs` keeps the event and the
            // caller gets a duplicate, which is what `eventLoadRemote` expects.
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

    /// Fill a command's out buffer with `bytes`, clamped to both the buffer
    /// the caller offered and the width it asked for.
    fn write_out_buffer(&mut self, tls: u32, bytes: &[u8], requested: u64) -> Result<()> {
        let Some((addr, len)) = self.ipc_output_buffer(tls, 0) else {
            return Ok(());
        };
        let take = requested.min(bytes.len() as u64).min(u64::from(len)) as usize;
        for (index, &byte) in bytes[..take].iter().enumerate() {
            self.mem.write_u8(addr.wrapping_add(index as u32), byte)?;
        }
        Ok(())
    }

    /// Which save an `fsp-srv` save-data request names.
    ///
    /// The request carries a `SaveDataAttribute`: a title id, a user id, and a
    /// system save id, of which the caller fills in whichever applies. The
    /// system's own saves are named by system save id; an application's by its
    /// title id. A request with neither is the running title asking for its
    /// own save, which is how `nn::fs::MountSaveData` spells it.
    fn save_data_id(&mut self, tls: u32) -> u64 {
        /// The attribute follows a `u8` space id, padded out to eight bytes.
        const ATTRIBUTE: u32 = 8;
        const SYSTEM_SAVE_DATA_ID: u32 = 0x18;
        let attribute = self.ipc_request_data(tls).wrapping_add(ATTRIBUTE);
        let application_id = self.mem.read_u64(attribute).unwrap_or(0);
        let system_save_id = self
            .mem
            .read_u64(attribute.wrapping_add(SYSTEM_SAVE_DATA_ID))
            .unwrap_or(0);
        match (system_save_id, application_id) {
            (0, 0) => self.program_id(),
            (0, application) => application,
            (system, _) => system,
        }
    }

    /// `IStorage`, backed by the current process's decrypted RomFS
    /// ([`Cpu::set_romfs`]). Cmd 0 = Read(u64 offset, u64 size), cmd 4 =
    /// GetSize — the same shape as `IFile`, but offset-addressed rather than
    /// path-addressed since there's exactly one of these per process.
    pub(super) fn fs_storage_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        // Which content this particular storage was opened on: the process's
        // own RomFS (command 200), or a system data archive (command 202).
        let archive = self
            .fs_storage_archive
            .get(&self.ipc_object_key(tls, handle))
            .copied();
        let size = self.storage_source(archive).map_or(0, |s| s.len());
        match cmd_id {
            // Read(u64 offset, u64 size) -> bytes into the recv buffer.
            Some(0) => {
                let data = self.ipc_request_data(tls);
                // `IStorage::Read(s64 offset, u64 size)`. Note the layout is
                // **not** `IFile::Read`'s: a file read leads with a `u32
                // option` and pads to 8, putting its offset at +8 and its size
                // at +0x10. This used to read those two fields, so every
                // storage read came back as "0 bytes at offset 0x50" — the
                // guest mounted its RomFS, parsed an empty header, and
                // `nn::fs::OpenDirectory("rom:/Data")` found nothing.
                let offset = self.mem.read_u64(data)?;
                let requested = self.mem.read_u64(data.wrapping_add(8))?;
                let trace_storage = crate::env_flag!("TRACE_IPC");
                if trace_storage {
                    eprintln!("[storage] read offset={offset:#x} size={requested:#x} of {size:#x}");
                }
                // A storage read is all or nothing: real `fs` checks the range
                // against the storage's size and refuses one that runs past
                // it. Clamping instead reports success over a buffer the
                // guest's own bytes are still in, and it acts on them — which
                // is how a RomFS layout this emulator did not implement
                // surfaced as `MountRom` calling the image corrupt, a hundred
                // million instructions from the section that caused it.
                const OUT_OF_RANGE: u32 = 2 | (3005 << 9);
                if offset > size || requested > size - offset {
                    return self.write_ipc_response(tls, OUT_OF_RANGE, &[], &[], &[]);
                }
                let start = offset;
                let end = start + requested;
                if let Some(addr) = self.ipc_output_buffer_addr(tls, 0) {
                    // The RomFS is not a buffer to slice: it is decrypted out
                    // of the container a range at a time (a retail one is
                    // gigabytes), so the copy goes through a fixed staging
                    // buffer no matter how much the guest asked for.
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
                        for (i, &byte) in buf[..got].iter().enumerate() {
                            self.mem
                                .write_u8(addr.wrapping_add(written + i as u32), byte)?;
                        }
                        written += got as u32;
                        pos += got as u64;
                    }
                }
                if trace_storage {
                    // What actually landed in the guest's buffer. A read that
                    // reports a size and delivers zeroes is indistinguishable
                    // from a successful one until you look.
                    let head: Vec<u8> = match self.ipc_output_buffer_addr(tls, 0) {
                        Some(addr) => (0..16)
                            .map(|i| self.mem.read_u8(addr.wrapping_add(i)).unwrap_or(0))
                            .collect(),
                        None => Vec::new(),
                    };
                    eprintln!("[storage]   -> {head:02x?}");
                }
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // GetSize -> u64
            Some(4) => self.write_ipc_response(tls, 0, &[], &size.to_le_bytes(), &[]),
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// The content an open `IStorage` is serving: a registered system data
    /// archive, or the process's own RomFS when it is not one.
    fn storage_source(&self, archive: Option<u64>) -> Option<&dyn crate::source::ByteSource> {
        match archive {
            Some(id) => self.data_archives.get(&id).map(|b| b.as_ref()),
            None => self.romfs.as_deref(),
        }
    }

    /// `IFileSystem`, backed by the emulated SD card in [`crate::vfs`].
    ///
    /// Paths arrive in the request's first static buffer, so every command
    /// resolves against the real tree: a missing path reports
    /// `FsError_PathNotFound` rather than pretending to succeed, which is what
    /// stops a menu from recursing forever into directories that do not exist.
    pub(super) fn fs_request(&mut self, tls: u32, cmd_id: Option<u32>, handle: u64) -> Result<()> {
        /// Horizon `fs` results: module 2, descriptions 1 (path not found) and
        /// 2 (path already exists).
        const PATH_NOT_FOUND: u32 = 2 | (1 << 9);
        const PATH_ALREADY_EXISTS: u32 = 2 | (2 << 9);
        let path = self.ipc_request_path(tls);
        // Which storage this `IFileSystem` addresses. The SD card and a save
        // are the same interface and the same paths; only the object they were
        // opened through tells them apart.
        let mount = self.mount_of(self.ipc_object_key(tls, handle));
        if crate::env_flag!("TRACE_IPC") {
            eprintln!(
                "[fs] pc={:#x} cmd={:?} path={:?} mount={mount:x?}",
                self.pc, cmd_id, path
            );
        }
        match cmd_id {
            // CreateFile(u32 option, s64 size) / CreateDirectory.
            //
            // Creating a file that already exists is an **error**, not a
            // truncation: `fsdev` opens a file for writing by calling this,
            // expecting "already exists", and then opening it. Answering with
            // a fresh empty file instead emptied the file on every reopen —
            // which is what made a config written one moment read back as
            // zero bytes the next.
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
            // GetEntryType -> FsDirEntryType
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
            // GetFreeSpaceSize / GetTotalSpaceSize: report a 32 GiB card.
            Some(11) | Some(12) => {
                let bytes = 32u64 << 30;
                self.write_ipc_response(tls, 0, &[], &bytes.to_le_bytes(), &[])
            }
            // GetFileTimeStampRaw -> FsTimeStampRaw { created, modified,
            // accessed, is_valid }. Nothing here records a file's times, and
            // `is_valid` is the field that says so — the point of answering
            // rather than falling through to the bare success below, which
            // left the caller reading three timestamps off its own stack.
            Some(14) => {
                if self.vfs_for(mount).entry_type(&path).is_none() {
                    return self.write_ipc_response(tls, PATH_NOT_FOUND, &[], &[], &[]);
                }
                self.write_ipc_response(tls, 0, &[], &[0u8; 0x20], &[])
            }
            // Commit and the remaining bookkeeping commands.
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// `IDirectory`: cmd 0 = `fsDirRead` (fill the out buffer with
    /// `FsDirectoryEntry` structs), cmd 1 = `fsDirGetEntryCount`.
    pub(super) fn fs_dir_request(&mut self, tls: u32, cmd_id: Option<u32>, key: u64) -> Result<()> {
        /// `sizeof(FsDirectoryEntry)`: a 0x301-byte name, padding, the entry
        /// type, more padding, then the 8-aligned size.
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

    /// `IFile`: cmd 0 = Read, cmd 1 = Write, cmd 2 = Flush, cmd 3 = SetSize,
    /// cmd 4 = GetSize.
    ///
    /// The writing half used to fall through to the catch-all below, so a
    /// guest was told its write succeeded and then read the file back empty.
    /// That is worse than an error: an application that cannot write its
    /// settings can say so, but one told it *did* write them has no reason to
    /// doubt the zero bytes it reads next. Checkpoint wrote its `config.json`,
    /// re-opened it, found nothing, and gave up before its first frame.
    pub(super) fn fs_file_request(
        &mut self,
        tls: u32,
        cmd_id: Option<u32>,
        key: u64,
    ) -> Result<()> {
        const PATH_NOT_FOUND: u32 = 2 | (1 << 9);
        let path = self.fs_files.get(&key).cloned().unwrap_or_default();
        // The storage the file was opened on, inherited from its filesystem.
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
                if crate::env_flag!("TRACE_IPC") {
                    eprintln!(
                        "[fs-file] read path={:?} offset={:#x} size={:#x} -> {:#x} buf={:?}",
                        path,
                        offset,
                        requested,
                        read,
                        self.ipc_output_buffer_addr(tls, 0)
                    );
                }
                if let Some(addr) = self.ipc_output_buffer_addr(tls, 0) {
                    for (i, &byte) in buf[..read].iter().enumerate() {
                        self.mem.write_u8(addr.wrapping_add(i as u32), byte)?;
                    }
                }
                self.write_ipc_response(tls, 0, &[], &(read as u64).to_le_bytes(), &[])
            }
            // Write(u32 option, s64 offset, u64 size) with the bytes in a
            // send buffer. `option`'s bit 0 is Flush, which costs nothing
            // here — the write has already reached the only copy there is.
            Some(1) => {
                let data = self.ipc_request_data(tls);
                let offset = self.mem.read_u64(data.wrapping_add(8))?;
                let requested = self.mem.read_u64(data.wrapping_add(0x10))?;
                let bytes = match self.ipc_send_buffer(tls, 0) {
                    Some((addr, len)) => self.read_bytes(addr, (len as u64).min(requested) as u32),
                    None => Vec::new(),
                };
                if crate::env_flag!("TRACE_IPC") {
                    eprintln!(
                        "[fs-file] write path={:?} offset={:#x} size={:#x} -> {:#x}",
                        path,
                        offset,
                        requested,
                        bytes.len()
                    );
                }
                match self.vfs_for(mount).write(&path, offset, &bytes) {
                    Some(_) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                    None => self.write_ipc_response(tls, PATH_NOT_FOUND, &[], &[], &[]),
                }
            }
            // Flush: there is no write-behind cache to flush.
            Some(2) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // SetSize(s64 size) — how `fsdev` truncates a file it opened with
            // `O_TRUNC`, so it has to actually shorten it.
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

    /// Reply with the two s64s every `am` save-data size query reports: a
    /// size and the journal beside it. Three commands differ only in which
    /// pair they name, and a reply that got the *width* wrong would be read as
    /// a size by all three.
    pub(super) fn write_save_data_pair(
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
    use crate::cpu::ipc::testing::*;
    use crate::cpu::Cpu;

    #[test]
    fn a_file_written_through_ifile_reads_back() {
        // `IFile`'s writing half used to fall through to the catch-all
        // "success, no data" reply, so the bytes went nowhere and the guest
        // had no way to know: Checkpoint wrote its config, re-opened it, read
        // zero bytes, and quit before drawing anything.
        let key = Cpu::object_key(9, 1);
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        cpu.mem.map_zero(0x3000, 0x100).unwrap();
        assert!(cpu.fs.create_file("/switch/cfg.json", 0));
        cpu.fs_files.insert(key, "/switch/cfg.json".to_owned());

        // fsFileWrite { u32 option, u32 pad, s64 offset, u64 size } with the
        // bytes in a send buffer.
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

        // GetSize agrees with what was written, rather than the zero the file
        // was created with.
        write_request(&mut cpu, 4, &[]);
        cpu.fs_file_request(TLS, Some(4), key).unwrap();
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), 7);

        // A write past the end grows the file.
        for (i, &byte) in b"!!".iter().enumerate() {
            cpu.mem.write_u8(0x3000 + i as u32, byte).unwrap();
        }
        let mut payload = [0u8; 0x18];
        payload[8..16].copy_from_slice(&7u64.to_le_bytes());
        payload[16..24].copy_from_slice(&2u64.to_le_bytes());
        write_map_buffer_request(&mut cpu, 1, &payload, 0x3000, 2, true);
        cpu.fs_file_request(TLS, Some(1), key).unwrap();
        assert_eq!(cpu.fs.size("/switch/cfg.json"), Some(9));

        // SetSize truncates — how `fsdev` honours `O_TRUNC`.
        write_request(&mut cpu, 3, &3u64.to_le_bytes());
        cpu.fs_file_request(TLS, Some(3), key).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.fs.file("/switch/cfg.json"), Some(&br#"{"v"#[..]));

        // A handle whose file is gone reports it rather than reporting
        // success — the distinction the catch-all could not make.
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
        // CreateFile(option, size): the size is the initial length.
        let mut payload = [0u8; 0x10];
        payload[8..16].copy_from_slice(&4u64.to_le_bytes());
        let mut cpu = request_with_path(0, "sdmc:/switch/cfg.json", &payload);
        cpu.record_handle(9, "fsp-srv");
        cpu.fs_request(TLS, Some(0), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "created");
        assert_eq!(cpu.fs.file("/switch/cfg.json"), Some(&[0u8; 4][..]));

        // Creating it again fails and leaves the contents alone — `fsdev`
        // opens an existing file this way, and truncating here is what made a
        // config read back empty right after it was written.
        cpu.fs.write("/switch/cfg.json", 0, b"{}!!").unwrap();
        write_path_request(&mut cpu, 0, "sdmc:/switch/cfg.json", &payload);
        cpu.fs_request(TLS, Some(0), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), PATH_ALREADY_EXISTS);
        assert_eq!(cpu.fs.file("/switch/cfg.json"), Some(&b"{}!!"[..]));
    }

    #[test]
    fn the_access_log_mode_a_title_sets_is_the_one_it_reads_back() {
        // GetGlobalAccessLogMode, before anything has set one: a retail
        // console boots with the access log off, and so does this.
        let mut cpu = request(false, 1005, &[]);
        cpu.record_handle(9, "fsp-srv");
        cpu.fsp_srv_request(TLS, Some(1005), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "off until set");

        // SetGlobalAccessLogMode(2): log to the SD card. The catch-all
        // answered this with a success that changed nothing, which is the one
        // reply a caller cannot tell from a server that agreed.
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

        // OutputApplicationInfoAccessLog, the writing end: a `Result` and
        // nothing else, which is what makes accepting and dropping it honest.
        write_request(&mut cpu, 1016, &[0u8; 0x10]);
        cpu.fsp_srv_request(TLS, Some(1016), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
    }

    #[test]
    fn a_console_with_no_save_data_ends_the_scan_on_the_first_read() {
        // OpenSaveDataInfoReaderBySaveDataSpaceId hands back an `IFileSystem`-
        // style out-object. The catch-all reply answered with success and no
        // object at all, so the caller read an object id out of its own reply
        // buffer and called `ReadSaveDataInfo` on whatever that named.
        let mut cpu = request(false, 61, &[1u8]);
        cpu.record_handle(9, "fsp-srv");
        cpu.fsp_srv_request(TLS, Some(61), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        let reader = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;
        assert_eq!(cpu.service_name(reader), Some("fsp-srv-save-info-reader"));

        // Zero entries is the whole termination condition — a reader stops
        // when a read reports nothing, and there is no other end signal. A
        // fabricated success is an endless scan: Checkpoint ran 1434 rounds of
        // it before this existed.
        write_request(&mut cpu, 0, &[]);
        cpu.fs_save_data_info_reader_request(TLS, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), 0, "entries");

        // Mounting one, on the other hand, hands back a filesystem over it.
        // This console has a NAND now, and a save that has never been written
        // is created on first open the way a console formats one on first use
        // -- which is a different thing from a save that does not exist, and
        // the reason the scan above still reports nothing to enumerate.
        write_request(&mut cpu, 52, &[0u8; 0x40]);
        cpu.fsp_srv_request(TLS, Some(52), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        let saves = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;
        assert_eq!(cpu.service_name(saves), Some("fsp-srv-fs"));
    }

    #[test]
    fn a_save_is_a_different_storage_from_the_sd_card() {
        // Save data and the SD card are the same interface over the same
        // paths; only the object a request arrives on says which is meant.
        // Confusing them would put a title's save on the card, where the next
        // title to mount the card would find it -- and a console keeps them on
        // different media for exactly that reason.
        const SAVE_ID: u64 = 0x0100_0000_0000_1000;
        /// The `SaveDataAttribute` follows a `u8` space id padded to eight,
        /// and carries the system save id 0x18 bytes into itself.
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

        // A directory created through that filesystem lands in the save.
        write_path_request(&mut cpu, 2, "/settings", &[]);
        cpu.fs_request(TLS, Some(2), saves).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x18).unwrap(),
            0,
            "creating a directory"
        );
        assert_eq!(
            cpu.save_data(SAVE_ID)
                .and_then(|save| save.entry_type("/settings")),
            Some(crate::vfs::ENTRY_TYPE_DIR),
            "the directory should be in the save"
        );
        assert_eq!(
            cpu.fs.entry_type("/settings"),
            None,
            "and must not have landed on the SD card"
        );

        // Reopening the same id finds it again: a save outlives the handle it
        // was opened through, which is the whole point of it.
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

    #[test]
    fn the_filtered_save_scan_is_a_reader_too() {
        // 68 = OpenSaveDataInfoReaderWithFilter, the 6.0.0+ form a current
        // JKSV opens once per save type. It hands out the same reader as the
        // unfiltered 60/61/62; falling through to the catch-all instead gave
        // the caller a success with no out-object, and `libnx` then sent
        // `ReadSaveDataInfo` to session handle 0.
        let mut payload = [0u8; 0x48];
        payload[0] = 1; // FsSaveDataSpaceId::User
        let mut cpu = request(false, 68, &payload);
        cpu.record_handle(9, "fsp-srv");
        cpu.fsp_srv_request(TLS, Some(68), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        let reader = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;
        assert_ne!(reader, 0, "reader session");
        assert_eq!(cpu.service_name(reader), Some("fsp-srv-save-info-reader"));

        // A filter can only narrow a list that is already empty.
        write_request(&mut cpu, 0, &[]);
        cpu.fs_save_data_info_reader_request(TLS, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), 0, "entries");
    }

    #[test]
    fn the_device_operator_reports_the_card_this_console_has_and_the_one_it_has_not() {
        // OpenDeviceOperator hands back an object, and the catch-all answered
        // it with a bare success — `nnSdk` reads an out-object on a plain
        // session as a move handle, parses the missing one as 0, and still
        // reports success, so the first call lands on a null proxy.
        let mut cpu = request(false, 400, &[]);
        cpu.record_handle(9, "fsp-srv");
        cpu.fsp_srv_request(TLS, Some(400), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        let operator = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;
        assert_ne!(operator, 0, "operator session");
        assert_eq!(cpu.service_name(operator), Some("fsp-srv-device-operator"));

        // The two presence bools, which are the only answers here that
        // anything decides on.
        write_request(&mut cpu, 0, &[]);
        cpu.fs_device_operator_request(TLS, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 1, "sd card inserted");

        write_request(&mut cpu, 200, &[]);
        cpu.fs_device_operator_request(TLS, Some(200)).unwrap();
        assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 0, "no game card");

        // GetSdCardUserAreaSize reports the same card `fsp-srv` and `ns` do.
        write_request(&mut cpu, 3, &[]);
        cpu.fs_device_operator_request(TLS, Some(3)).unwrap();
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), super::SD_TOTAL_SPACE);

        // GetGameCardHandle is refused rather than answered with a handle to a
        // card that is not in the slot the command before just denied.
        const UNKNOWN_COMMAND_ID: u32 = 10 | (221 << 9);
        write_request(&mut cpu, 202, &[]);
        cpu.fs_device_operator_request(TLS, Some(202)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), UNKNOWN_COMMAND_ID);
    }

    #[test]
    fn a_device_operator_register_is_written_over_whatever_the_buffer_held() {
        // GetSdCardCid reports nothing but a Result, so a caller reads the
        // whole 0x10 back whether the server filled it or not — a success that
        // writes nothing hands it its own stack as a card serial.
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
        // Nothing here is slowed to match, but a mode that reads back as
        // `None` is a request the caller was told had been refused.
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
        // OpenSdCardDetectionEventNotifier: the same null out-object the
        // device operator was, one command along. The notifier is an object,
        // and the event only reaches the caller through it.
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

        // GetEventHandle on each. A card arriving in one slot is not a card
        // arriving in the other, so one shared event would wake both waiters.
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

        // Neither is ever signalled: a waiter is waiting for a slot to
        // *change*, and nothing here can change one.
        assert_eq!(cpu.event_signaled(sd_event), Some(false));
        assert_eq!(cpu.event_signaled(game_card_event), Some(false));

        // Asking twice hands back the event already being waited on rather
        // than a second one — a poller would otherwise leak a handle per call.
        write_request(&mut cpu, 0, &[]);
        cpu.fs_detection_notifier_request(TLS, sd, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64, sd_event);
    }

    /// `IStorage::Read` is not a short read: real `fs` refuses a range that
    /// runs past the storage.
    #[test]
    fn a_storage_read_past_the_end_is_refused_rather_than_clamped() {
        const BUFFER: u32 = 0x4000;
        const OUT_OF_RANGE: u32 = 2 | (3005 << 9);
        let romfs: Vec<u8> = (0..=0xFFu8).collect();

        // `IStorage::Read(s64 offset, u64 size)` — no leading option word.
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

        // The whole storage exactly, which is in range and must stay so.
        write_map_buffer_request(&mut cpu, 0, &read(0, 0x100), BUFFER, 0x100, false);
        cpu.fs_storage_request(TLS, 1, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.read_bytes(BUFFER, 0x100), romfs);

        // One byte past it is not. The buffer keeps whatever the caller left
        // there — the point of refusing is that it never reads it as data.
        for offset in 0..0x100 {
            cpu.mem.write_u8(BUFFER + offset, 0xAA).unwrap();
        }
        write_map_buffer_request(&mut cpu, 0, &read(0xF0, 0x11), BUFFER, 0x40, false);
        cpu.fs_storage_request(TLS, 1, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), OUT_OF_RANGE);
        assert_eq!(cpu.read_bytes(BUFFER, 0x20), vec![0xAA; 0x20]);

        // And so is a read that starts past the end, however small.
        write_map_buffer_request(&mut cpu, 0, &read(0x100, 1), BUFFER, 0x40, false);
        cpu.fs_storage_request(TLS, 1, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), OUT_OF_RANGE);

        // GetSize is what a caller sizes those reads against.
        write_request(&mut cpu, 4, &[]);
        cpu.fs_storage_request(TLS, 1, Some(4)).unwrap();
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), 0x100);
    }
}
