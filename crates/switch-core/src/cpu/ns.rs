//! `ns`: the title manager — what is installed, how much room it takes, and
//! the DLC (`aoc`), play statistics (`pdm`) and album (`caps`) interfaces
//! alongside it.
//!
//! Most of `ns` is a creator that hands out sub-interfaces, and a fabricated
//! object id is not callable — so an unimplemented getter here does not fail
//! one command, it ends the whole chain at its first.

use super::fs::{SD_FREE_SPACE, SD_TOTAL_SPACE};
use super::Cpu;
use crate::Result;

impl Cpu {
    /// `ns:am2` (`IServiceGetterInterface`) and the interfaces it hands out:
    /// the console's record of which applications are **installed**.
    ///
    /// Nothing is installed here. There is no NAND to install a title to, no
    /// content manager to install one with, and no application record database
    /// to have recorded it — so `ListApplicationRecord` is an empty list, and
    /// that is the truthful answer rather than a gap. A save manager asks `ns`
    /// what a title id is *called* so it can label the save it found; with no
    /// records to label, it has nothing to ask about.
    ///
    /// The generic fallback answered the getters below with a fabricated
    /// object id, so a caller that asked for `IApplicationManagerInterface`
    /// got an id that was not one and then called `ListApplicationRecord` on
    /// it — which the fallback also answered with a fresh object id, leaving
    /// the caller to read its record count out of that.
    ///
    /// Before 3.0.0 `ns:am` *was* the application manager; from 3.0.0 the
    /// service is a getter and the manager is one of eleven interfaces it
    /// hands out (7988..=7999, with 7990 unassigned). Both routes land on the
    /// same interfaces here.
    pub(super) fn ns_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        const QUERY_POINTER_BUFFER_SIZE: u32 = 3;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let name = self.service_name(handle).unwrap_or("ns:am2").to_string();
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, &name);
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                // `libnx` records this on the session the moment `sm` hands it
                // over, before any `ns` command is sent — it is what the
                // fabricated-object-id reply was corrupting for `ns:am2`.
                Some(QUERY_POINTER_BUFFER_SIZE) => {
                    self.write_ipc_response(tls, 0, &[], &0x1000u16.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, "ns:control", cmd_id),
            };
        }
        let object_id = self.ipc_domain_object_id(tls);
        let iface = if self.ipc_is_domain_request(tls) {
            self.domain_interface(handle, object_id).unwrap_or("ns:am2").to_string()
        } else {
            match self.service_name(handle) {
                Some(name) => name.to_string(),
                None => "ns:am2".to_string(),
            }
        };
        match iface.as_str() {
            // The getter services. Every one of them is the same
            // `IServiceGetterInterface`; which system service you opened
            // decides what you are *allowed* to ask for, not what the
            // interface is, and nothing here enforces privilege.
            // `ns:su` is not one more getter service: `ISystemUpdateInterface`
            // has its own small command set, and it is opened at boot by the
            // Home Menu — which is the only process that has anywhere to show
            // "an update is available".
            "ns:su" => match cmd_id {
                // GetBackgroundNetworkUpdateState -> u8. Nothing downloads
                // here, so no update is staged.
                Some(0) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                // NotifyExFatDriverRequired / NotifyBackgroundNetworkUpdate /
                // NotifySystemUpdateForContentDelivery / PrepareShutdown:
                // announcements to an update pipeline that is not here.
                Some(2) | Some(5) | Some(10) | Some(11) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetSystemUpdateNotificationEventForContentDelivery: the
                // event that fires when an update becomes available. Handed
                // out and never signalled, for the same reason.
                Some(9) => {
                    let h = self.alloc_event("ns:system-update", true);
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "ns:am2" | "ns:ec" | "ns:rid" | "ns:rt" | "ns:web" | "ns:ro"
            | "ns:vm" | "ns:dev" => match cmd_id {
                // The ids are `libnx`'s (`nsGet*Interface` in `ns.c`), which
                // are Nintendo's own. Everything from 7989 up used to be
                // numbered one too low here, so a caller got the *next*
                // interface along: `nsInitialize` asks for 7996 and was handed
                // an `ns:account-proxy` to call `ListApplicationRecord` (cmd 0)
                // on, which is not a command that interface has. Note the gap
                // at 7990 — it is genuinely not assigned.
                Some(7988) => self.ns_reply_with_interface(tls, handle, "ns:dynamic-rights"),
                Some(7989) => self.ns_reply_with_interface(tls, handle, "ns:read-only-control"),
                Some(7991) => self.ns_reply_with_interface(tls, handle, "ns:read-only-record"),
                Some(7992) => self.ns_reply_with_interface(tls, handle, "ns:ecommerce"),
                Some(7993) => self.ns_reply_with_interface(tls, handle, "ns:app-version"),
                Some(7994) => self.ns_reply_with_interface(tls, handle, "ns:factory-reset"),
                Some(7995) => self.ns_reply_with_interface(tls, handle, "ns:account-proxy"),
                Some(7996) => self.ns_reply_with_interface(tls, handle, "ns:app-manager"),
                Some(7997) => self.ns_reply_with_interface(tls, handle, "ns:download-task"),
                Some(7998) => self.ns_reply_with_interface(tls, handle, "ns:content-management"),
                Some(7999) => self.ns_reply_with_interface(tls, handle, "ns:document"),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // `ns:am` is the pre-3.0.0 service, where the application manager
            // is reached directly rather than through a getter.
            "ns:am" | "ns:app-manager" => self.ns_application_manager_request(tls, &iface, cmd_id),
            // `IContentManagementInterface`: what is on each storage, and
            // whether the card holding it is there at all.
            "ns:content-management" => match cmd_id {
                // CheckSdCardMountStatus. There is an emulated SD card and it
                // is always mounted, so this succeeds; a *refusal* is how the
                // Home Menu is told the card it was using has gone, which is
                // not a state anything here can be in.
                Some(43) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetTotalSpaceSize / GetFreeSpaceSize(StorageId) -> u64. The
                // same 32 GiB card `fsp-srv` reports, half of it used, so that
                // a caller doing the arithmetic between the two answers gets a
                // number that is neither full nor impossible.
                Some(47) => {
                    self.write_ipc_response(tls, 0, &[], &SD_TOTAL_SPACE.to_le_bytes(), &[])
                }
                Some(48) => {
                    self.write_ipc_response(tls, 0, &[], &SD_FREE_SPACE.to_le_bytes(), &[])
                }
                // CountApplicationContentMeta(u64 application_id) -> u32:
                // nothing is installed, so nothing has content meta.
                Some(600) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // `IDownloadTaskInterface`: the background download queue. There
            // is no network and nothing to download, so the only commands that
            // mean anything are the two that turn auto-commit on and off.
            "ns:download-task" => match cmd_id {
                // EnableAutoCommit / DisableAutoCommit.
                Some(707) | Some(708) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // `IReadOnlyApplicationRecordInterface`: the record half of the
            // manager, for callers that only want to know what is installed.
            "ns:read-only-record" => match cmd_id {
                // HasApplicationRecord(u64 application_id) -> bool. Nothing is
                // installed, so nothing has a record.
                Some(0) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }

    /// Hand out one of `ns`'s sub-interfaces. The getters all have the same
    /// shape — no input, one out-interface — so they share this.
    fn ns_reply_with_interface(&mut self, tls: u32, handle: u64, name: &str) -> Result<()> {
        self.reply_with_interface(tls, handle, name)?;
        Ok(())
    }

    /// `aoc:u` — "nn::aocsrv::detail::IAddOnContentManager", the add-on
    /// content a title has been given. With nothing registered every answer
    /// here is the one a retail console gives a title whose DLC nobody has
    /// bought: the list is empty. What the host adds through
    /// [`Cpu::add_add_on_content`] shows up in it.
    ///
    /// The service had no implementation at all, so every command reached the
    /// generic fabricated-object reply — which hands a *void* command an
    /// object id, a sub-session and an event it never asked for, and hands
    /// `CountAddOnContent` an object id the caller then reads as a **count**.
    /// A title that believes it owns two pieces of DLC goes looking for two
    /// content archives that do not exist.
    pub(super) fn aoc_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        const QUERY_POINTER_BUFFER_SIZE: u32 = 3;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "aoc:u");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                Some(QUERY_POINTER_BUFFER_SIZE) => {
                    self.write_ipc_response(tls, 0, &[], &0x1000u16.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, "aoc:control", cmd_id),
            };
        }
        match cmd_id {
            // CountAddOnContent -> u32.
            Some(2) => {
                let count = self.add_on_content().len() as u32;
                self.write_ipc_response(tls, 0, &[], &count.to_le_bytes(), &[])
            }
            // ListAddOnContent(u32 offset, u32 count) -> u32 written, with the
            // indices themselves going into an out buffer. What a title does
            // with them is ask `fsp-srv` for base id + index, so an index that
            // is listed and then not mountable is worse than one never listed:
            // both halves come from the same registration.
            Some(3) => {
                let args = self.ipc_request_data(tls);
                let offset = self.mem.read_u32(args)? as usize;
                let count = self.mem.read_u32(args.wrapping_add(4))? as usize;
                let all = self.add_on_content();
                let listed = all.get(offset..).unwrap_or(&[]);
                let (addr, size) = self.ipc_output_buffer(tls, 0).unwrap_or((0, 0));
                let room = if addr == 0 { 0 } else { size as usize / 4 };
                let written = listed.len().min(count).min(room);
                for (i, index) in listed[..written].iter().enumerate() {
                    self.mem
                        .write_u32(addr.wrapping_add(4 * i as u32), *index)?;
                }
                self.write_ipc_response(tls, 0, &[], &(written as u32).to_le_bytes(), &[])
            }
            // GetAddOnContentBaseId -> u64, the number every add-on content
            // index is built against. Answering 0 would have a title asking
            // for content ids that belong to no title at all.
            Some(5) => {
                let base = self.add_on_content_base_id();
                self.write_ipc_response(tls, 0, &[], &base.to_le_bytes(), &[])
            }
            // PrepareAddOnContent(s32 index): make one entry of the list ready
            // to mount. Everything registered here already is — the host
            // handed over the container before the title started — so this is
            // the acknowledgement and nothing else, which is what Eden's
            // `IAddOnContentManager::PrepareAddOnContent` does too.
            Some(7) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // GetAddOnContentListChangedEvent, and the same event fetched on
            // another process's behalf (…WithProcessId, which differs only by
            // taking a pid). It fires when DLC is installed or removed while
            // the title runs; nothing here installs anything, so it is handed
            // out and never signalled.
            Some(8) | Some(10) => {
                let event = match self.aoc_list_changed_event {
                    Some(event) => event,
                    None => {
                        let event = self.alloc_event("aoc:list-changed", true);
                        self.aoc_list_changed_event = Some(event);
                        event
                    }
                };
                self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
            }
            // GetAddOnContentLostErrorCode -> the `nn::err::ErrorCode` a title
            // displays when the DLC it was using has gone. Nothing here can
            // take content away mid-run, so there is no code to show.
            Some(9) => self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[]),
            // NotifyMountAddOnContent / NotifyUnmountAddOnContent: a title
            // telling `aocsrv` it is holding content open, so the system does
            // not delete it underneath. Nothing is being held either way.
            Some(11) | Some(12) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // IsAddOnContentMountedForDebug -> bool.
            Some(13) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
            // CheckAddOnContentMountStatus: a title asking whether what it
            // mounted is still there. There is no out value — the **Result**
            // is the whole answer, and a failure is how a title is told its
            // DLC has been removed since it mounted it. Nothing here removes
            // content once it is registered, so this succeeds.
            Some(50) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            _ => self.unimplemented_command(tls, "aoc:u", cmd_id),
        }
    }

    /// `caps:a` — `nn::capsrv::detail::IAlbumAccessorService`, the album of
    /// screenshots and clips a console keeps on its NAND and its SD card.
    ///
    /// This console has neither, so every answer here describes an album that
    /// is **mounted and empty** — which is what a freshly initialised console
    /// has, and is a state the Album applet knows how to show. Reporting it
    /// unmounted instead is a different thing entirely: it is the card-removed
    /// error, and the applet puts a message on the screen rather than a
    /// gallery.
    ///
    /// Nothing implemented this at all, and the Album applet is precisely the
    /// title that asks: it polls `IsAlbumMounted` and `GetAutoSavingStorage`
    /// once a frame, and the fallback answered a **bool** with a fabricated
    /// object id, which is not 0 or 1 but a large number read one byte at a
    /// time.
    pub(super) fn caps_album_accessor_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(0) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "caps:a");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                _ => self.write_ipc_response(tls, 0, &[], &0x1000u16.to_le_bytes(), &[]),
            };
        }
        match cmd_id {
            // GetAlbumFileCount(AlbumStorage) -> u64, and GetAlbumFileCountEx0,
            // which takes the same storage plus a flags byte. No files.
            Some(0) | Some(100) => {
                self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[])
            }
            // GetAlbumFileList / …Ex0 -> u64 entries written, with the
            // `AlbumEntry` array itself going into a map-alias out buffer.
            // None are written, and the count says so — so the buffer is left
            // exactly as the caller left it and nothing walks it.
            Some(1) | Some(101) => {
                self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[])
            }
            // DeleteAlbumFile(AlbumFileId). There is no file any list here
            // could have named, so nothing can reach this with a real id.
            Some(3) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // IsAlbumMounted(AlbumStorage) -> bool.
            Some(5) => self.write_ipc_response(tls, 0, &[], &[1u8], &[]),
            // GetAlbumMountResult(AlbumStorage) -> Result. Mounted, so the
            // result *is* the success this reply already carries.
            Some(16) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // Command 18 has no name on switchbrew and none in `nnSdk`'s
            // symbols either; Eden calls it `Unknown18` and answers a written
            // length of zero into a caller-supplied buffer. The Album applet
            // issues it once, before anything else, with a 0x40-byte buffer.
            Some(18) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            // GetAutoSavingStorage -> bool: whether new captures are written
            // to the SD card rather than the NAND. There is no SD card.
            Some(401) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
            _ => self.unimplemented_command(tls, "caps:a", cmd_id),
        }
    }

    /// `IApplicationManagerInterface`, the interface a title actually asks
    /// about installed applications through.
    fn ns_application_manager_request(
        &mut self,
        tls: u32,
        iface: &str,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        match cmd_id {
            // ListApplicationRecord(s32 entry_offset) -> (s32 count, out
            // buffer of ApplicationRecord). Zero records, whatever the offset:
            // see the type comment for why that is the answer and not a stub.
            //
            // The count has to be written even though it is zero — a caller
            // that gets a success with no out-data reads its record count off
            // its own stack, which is how "no titles installed" turns into
            // several billion of them.
            Some(0) => self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[]),
            // GenerateApplicationRecordCount -> u64. Same answer as the list
            // above, in the shape the counting call expects.
            Some(1) => self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[]),
            // GetApplicationRecordUpdateSystemEvent -> event. The Home Menu
            // waits on this before it reads the title list, so it goes out
            // **signalled** — hardware hands out a record set that is already
            // current, and a dark event here is a Home Menu that never asks
            // what is installed. One event per process: a caller that asks
            // twice has to be given the one it is already waiting on.
            Some(2) => {
                let h = match self.application_record_event {
                    Some(h) => h,
                    None => {
                        let h = self.alloc_event("ns:record-update", false);
                        self.application_record_event = Some(h);
                        h
                    }
                };
                self.signal_event(h);
                self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
            }
            // CheckSdCardMountStatus, and the total/free space of a storage
            // id. `IContentManagementInterface` answers these too and the
            // manager delegates to it on hardware, so the answers are the same
            // ones — see the `ns:content-management` arm.
            Some(43) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            Some(47) => self.write_ipc_response(tls, 0, &[], &SD_TOTAL_SPACE.to_le_bytes(), &[]),
            Some(48) => self.write_ipc_response(tls, 0, &[], &SD_FREE_SPACE.to_le_bytes(), &[]),
            // GetStorageSize(u8 storage_id) -> (s64 total, s64 free): the two
            // above in one call.
            Some(71) => {
                let mut out = [0u8; 16];
                out[..8].copy_from_slice(&SD_TOTAL_SPACE.to_le_bytes());
                out[8..].copy_from_slice(&SD_FREE_SPACE.to_le_bytes());
                self.write_ipc_response(tls, 0, &[], &out, &[])
            }
            // ResumeAll: resume the download tasks. There are none.
            Some(70) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // The media events: SD card mount status, SD card removed, game
            // card attached, game card update detected, game card mount
            // failed. Nothing here can change any of that, so they are handed
            // out dark — but the same object every time, because the Home Menu
            // keeps one waiter per event for as long as it runs.
            Some(cmd @ (44 | 45 | 49 | 52 | 505)) => {
                let h = match self.ns_manager_events.get(&cmd) {
                    Some(&h) => h,
                    None => {
                        let name = match cmd {
                            44 => "ns:sd-mount-status",
                            45 => "ns:gamecard-attach",
                            49 => "ns:sd-removed",
                            52 => "ns:gamecard-update",
                            _ => "ns:gamecard-mount-failure",
                        };
                        let h = self.alloc_event(name, false);
                        self.ns_manager_events.insert(cmd, h);
                        h
                    }
                };
                self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
            }
            _ => self.unimplemented_command(tls, iface, cmd_id),
        }
    }

    /// `pdm:qry` (`IQueryService`): the play-history database — what has been
    /// played, for how long, and when.
    ///
    /// **Nothing has ever been played on this console.** There is no
    /// `pdm:ntfy` recording launches, nothing persists across a page reload,
    /// and no title has run here more than once. So every query answers with
    /// an empty result rather than a fabricated history: no play events, no
    /// account events, an empty available range, and zeroed statistics.
    ///
    /// An empty result is a state a real console has too — a factory-fresh one
    /// — which is what makes it a truthful answer rather than a placeholder.
    pub(super) fn pdm_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[]);
        }
        match cmd_id {
            // The list queries — QueryAppletEvent, QueryPlayEvent,
            // QueryAccountEvent, QueryAccountPlayEvent,
            // QueryRecentlyPlayedApplication — each fill an output array and
            // report how many entries they wrote. None.
            Some(0) | Some(5) | Some(7) | Some(8) | Some(11) => {
                self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[])
            }
            // QueryPlayStatisticsByApplicationId /
            // ...AndUserAccountId -> PdmPlayStatistics: an application that
            // has been launched zero times, for zero minutes.
            Some(2) | Some(3) => self.write_ipc_response(tls, 0, &[], &[0u8; 0x28], &[]),
            // GetAvailablePlayEventRange / GetAvailableAccountPlayEventRange
            // -> { s32 total, s32 start, s32 end }: an empty range.
            Some(6) | Some(9) => self.write_ipc_response(tls, 0, &[], &[0u8; 12], &[]),
            _ => self.unimplemented_command(tls, "pdm:qry", cmd_id),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::cpu::ipc::testing::*;
    use crate::cpu::Cpu;

    #[test]
    fn pdm_reports_a_console_nothing_has_been_played_on() {
        // QueryPlayEvent: no entries, because nothing here records any.
        let mut cpu = request(false, 5, &[]);
        cpu.pdm_request(TLS, Some(5)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0);

        // And the statistics for any application are a title launched zero
        // times — not a fabricated playtime.
        let mut cpu = request(false, 2, &[0u8; 8]);
        cpu.pdm_request(TLS, Some(2)).unwrap();
        assert_eq!(cpu.read_bytes(TLS + 0x20, 0x20), vec![0u8; 0x20]);
    }

    #[test]
    fn ns_answers_the_pointer_buffer_size_asked_before_any_command() {
        // `libnx` records the pointer buffer size on the session as part of
        // opening it, so this is the *first* thing `ns:am2` is ever asked and
        // the only thing a caller that never lists a title asks at all. The
        // generic fallback answered it with a fabricated object id, so the
        // size came back as whatever that id happened to be.
        let mut cpu = control_request(3);
        cpu.register_service_handle(9, "ns:am2");
        cpu.ns_request(TLS, 9, Some(3)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u16(TLS + 0x20).unwrap(), 0x1000);
    }

    #[test]
    fn ns_reports_the_sd_card_as_mounted() {
        // `IContentManagementInterface::CheckSdCardMountStatus` answers with a
        // bare Result, so *refusing* it is how the caller is told the card it
        // was using has gone. Nothing here can be in that state -- the
        // emulated card is always there -- and the Home Menu asks before it
        // has anything to show.
        let mut cpu = request(false, 43, &[]);
        cpu.register_service_handle(9, "ns:content-management");
        cpu.ns_request(TLS, 9, Some(43)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "the card reads as missing");

        // And the two space queries have to be answerable in either order:
        // free must be no larger than total, or the arithmetic between them
        // underflows into a card with negative space used.
        let mut cpu = request(false, 47, &[]);
        cpu.register_service_handle(9, "ns:content-management");
        cpu.ns_request(TLS, 9, Some(47)).unwrap();
        let total = cpu.mem.read_u64(TLS + 0x20).unwrap();
        let mut cpu = request(false, 48, &[]);
        cpu.register_service_handle(9, "ns:content-management");
        cpu.ns_request(TLS, 9, Some(48)).unwrap();
        let free = cpu.mem.read_u64(TLS + 0x20).unwrap();
        assert!(free > 0 && free <= total, "free {free} of total {total}");
    }

    #[test]
    fn ns_hands_out_the_interface_each_getter_names() {
        // From 3.0.0 `ns:am2` is a getter: every interface behind it is
        // reached by asking for it by command id, and answering the wrong one
        // (or a fabricated object) hands the caller an interface whose
        // commands mean something else entirely. That is not hypothetical —
        // this table used to be shifted one id down from 7989 up, so
        // `nsInitialize`'s 7996 came back as the account proxy and JKSV's
        // `ListApplicationRecord` landed on an interface without that command.
        for (command, expected) in [
            (7988u32, "ns:dynamic-rights"),
            (7989, "ns:read-only-control"),
            (7991, "ns:read-only-record"),
            (7992, "ns:ecommerce"),
            (7993, "ns:app-version"),
            (7994, "ns:factory-reset"),
            (7995, "ns:account-proxy"),
            (7996, "ns:app-manager"),
            (7997, "ns:download-task"),
            (7998, "ns:content-management"),
            (7999, "ns:document"),
        ] {
            let mut cpu = request(false, command, &[]);
            cpu.register_service_handle(9, "ns:am2");
            cpu.ns_request(TLS, 9, Some(command)).unwrap();
            assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "{command}");
            let session = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;
            assert_ne!(session, 0, "{command}");
            assert_eq!(cpu.service_name(session), Some(expected), "{command}");
        }
    }

    #[test]
    fn ns_reports_a_console_with_nothing_installed() {
        // There is no NAND to install a title to and no record database to
        // have recorded one, so the list is empty — and the count has to be
        // written even though it is zero, or the caller reads its record
        // count off its own stack.
        let mut cpu = request(false, 7996, &[]);
        cpu.register_service_handle(9, "ns:am2");
        cpu.ns_request(TLS, 9, Some(7996)).unwrap();
        let manager = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;

        write_request(&mut cpu, 0, &0i32.to_le_bytes()); // ListApplicationRecord
        cpu.ns_request(TLS, manager, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "record count");

        // And no application id has a record, including the one this process
        // is running as.
        let mut cpu = request(false, 7991, &[]);
        cpu.register_service_handle(9, "ns:am2");
        cpu.ns_request(TLS, 9, Some(7991)).unwrap();
        let records = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;

        write_request(&mut cpu, 0, &crate::cpu::ipc::DEFAULT_PROGRAM_ID.to_le_bytes());
        cpu.ns_request(TLS, records, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 0, "has record");
    }

    #[test]
    fn ns_reports_a_command_it_does_not_implement_rather_than_succeeding() {
        // The whole point of naming the service instead of leaving it to the
        // fallback: a command with nothing behind it has to fail, so the
        // caller fails at the command that is genuinely missing and the log
        // names the one to implement next.
        let mut cpu = request(false, 400, &[]);
        cpu.register_service_handle(9, "ns:am2");
        cpu.ns_request(TLS, 9, Some(400)).unwrap();
        const UNKNOWN_COMMAND_ID: u32 = 10 | (221 << 9);
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), UNKNOWN_COMMAND_ID);
    }

    /// Drive one `aoc:u` command on a session opened under that service.
    fn aoc(cpu: &mut Cpu, command_id: u32) {
        cpu.register_service_handle(9, "aoc:u");
        cpu.aoc_request(TLS, 9, Some(command_id)).unwrap();
    }

    #[test]
    fn aoc_reports_a_title_nobody_has_bought_add_on_content_for() {
        const SFCO: u32 = 0x4F43_4653;
        const PROGRAM_ID: u64 = 0x0100_4890_117B_2000;

        // CountAddOnContent. This is the answer the generic fabricated-object
        // reply was getting wrong in the most expensive way: it hands back an
        // object id, and an object id read as a count is a title looking for
        // content archives that were never installed.
        let mut cpu = request(false, 2, &[]);
        aoc(&mut cpu, 2);
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "Result");
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "count");

        // GetAddOnContentBaseId -> the program id plus 0x1000, the base every
        // add-on content index is built against. Zero would have a title
        // asking for content ids belonging to no title at all.
        let mut cpu = request(false, 5, &[]);
        cpu.set_program_id(PROGRAM_ID);
        aoc(&mut cpu, 5);
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), PROGRAM_ID + 0x1000);

        // CheckAddOnContentMountStatus: no out value at all -- the Result is
        // the whole answer, and a failure is how a title is told the DLC it
        // mounted has gone. Nothing was ever mounted, so nothing can go.
        let mut cpu = request(false, 50, &[]);
        aoc(&mut cpu, 50);
        assert_eq!(cpu.mem.read_u32(TLS + 0x10).unwrap(), SFCO);
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "Result");
    }

    #[test]
    fn registered_add_on_content_is_counted_listed_and_mountable() {
        const PROGRAM_ID: u64 = 0x0100_BEE0_17FC_0000;
        const BUFFER: u32 = 0x4000;
        // Just Dance 2023's two DLC containers: base + 0x1000 + index, so
        // `...1001` and `...1004` are indices 1 and 4.
        let content = [PROGRAM_ID + 0x1001, PROGRAM_ID + 0x1004];

        let register = |cpu: &mut Cpu| {
            cpu.set_program_id(PROGRAM_ID);
            for id in content {
                let src = crate::source::MemSource(vec![0u8; 0x20]);
                assert_eq!(cpu.add_add_on_content(id, Box::new(src)), Some((id - PROGRAM_ID - 0x1000) as u32));
            }
        };

        // CountAddOnContent reports what was registered, not an object id.
        let mut cpu = request(false, 2, &[]);
        register(&mut cpu);
        aoc(&mut cpu, 2);
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "Result");
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 2, "count");

        // ListAddOnContent writes the indices themselves into the caller's
        // buffer: what a title does with one is ask for base id + index, so an
        // index listed but not mountable is worse than one never listed.
        let mut list_args = [0u8; 8];
        list_args[4..].copy_from_slice(&8u32.to_le_bytes()); // offset 0, room for 8
        let mut cpu = request_with_recv_buffer(3, &list_args, BUFFER, 0x20);
        cpu.mem.map_zero(BUFFER, 0x100).unwrap();
        register(&mut cpu);
        cpu.register_service_handle(9, "aoc:u");
        cpu.aoc_request(TLS, 9, Some(3)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 2, "written");
        assert_eq!(cpu.mem.read_u32(BUFFER).unwrap(), 1);
        assert_eq!(cpu.mem.read_u32(BUFFER + 4).unwrap(), 4);

        // An offset past the end is not an error, it is the end of the list —
        // a title paging through one asks once more than there is content.
        let mut args = [0u8; 8];
        args[..4].copy_from_slice(&2u32.to_le_bytes());
        args[4..].copy_from_slice(&8u32.to_le_bytes());
        let mut cpu = request_with_recv_buffer(3, &args, BUFFER, 0x20);
        cpu.mem.map_zero(BUFFER, 0x100).unwrap();
        register(&mut cpu);
        cpu.register_service_handle(9, "aoc:u");
        cpu.aoc_request(TLS, 9, Some(3)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "Result");
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "written");

        // And the content is mountable by the id those indices are built
        // against: `fsp-srv`'s OpenDataStorageByDataId is the other half.
        let mut cpu = Cpu::new();
        register(&mut cpu);
        assert!(cpu.has_data_archive(content[0]));
    }

    #[test]
    fn add_on_content_belonging_to_another_title_is_refused() {
        // A DLC's id is its base title's plus an index below 0x800. Anything
        // else cannot be numbered against this title, and registering it would
        // list an index whose content id nothing will ever ask for.
        let mut cpu = Cpu::new();
        cpu.set_program_id(0x0100_BEE0_17FC_0000);
        let src = || Box::new(crate::source::MemSource(vec![0u8; 0x10]));
        assert_eq!(cpu.add_add_on_content(0x0100_0000_0000_1001, src()), None);
        assert_eq!(cpu.add_add_on_content(0x0100_BEE0_17FC_1801, src()), None);
        assert!(cpu.add_on_content().is_empty());
    }

    #[test]
    fn the_add_on_content_list_never_changes_so_its_event_never_fires() {
        // GetAddOnContentListChangedEvent, then the same event fetched through
        // the ...WithProcessId form a system title uses. They are one event on
        // hardware, and handing out two would leave a caller waiting on
        // whichever it asked for second.
        let mut cpu = request(false, 8, &[]);
        aoc(&mut cpu, 8);
        let event = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
        assert_ne!(event, 0, "GetAddOnContentListChangedEvent handed back no handle");
        assert_eq!(cpu.event_name(event), Some("aoc:list-changed"));
        assert_eq!(cpu.event_signaled(event), Some(false));

        marshal(&mut cpu, false, 10, &[]);
        cpu.aoc_request(TLS, 9, Some(10)).unwrap();
        assert_eq!(u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap()), event);
    }
}
