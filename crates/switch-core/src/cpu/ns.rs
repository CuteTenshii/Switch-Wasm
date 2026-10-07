//! `ns`: the title manager (installed titles, storage space), plus the DLC
//! (`aoc`), play statistics (`pdm`), play report (`prepo`) and album (`caps`)
//! services.

use super::fs::{SD_FREE_SPACE, SD_TOTAL_SPACE};
use super::Cpu;
use crate::Result;

impl Cpu {
    /// `ns:am2` (`IServiceGetterInterface`) and the interfaces it hands out.
    /// Nothing is installed, so record lists are empty. Before 3.0.0 `ns:am`
    /// was the application manager itself; both routes land here.
    pub(super) fn ns_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let name = self.service_name(handle).unwrap_or("ns:am2").to_string();
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, &name);
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, "ns:control", cmd_id),
            };
        }
        let object_id = self.ipc_domain_object_id(tls);
        let iface = if self.ipc_is_domain_request(tls) {
            self.domain_interface(handle, object_id)
                .unwrap_or("ns:am2")
                .to_string()
        } else {
            match self.service_name(handle) {
                Some(name) => name.to_string(),
                None => "ns:am2".to_string(),
            }
        };
        match iface.as_str() {
            // The getter services all share `IServiceGetterInterface`; privilege
            // is not enforced. `ns:su` is `ISystemUpdateInterface`.
            "ns:su" => match cmd_id {
                // GetBackgroundNetworkUpdateState -> u8: nothing staged.
                Some(0) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                // NotifyExFatDriverRequired / NotifyBackgroundNetworkUpdate /
                // NotifySystemUpdateForContentDelivery / PrepareShutdown.
                Some(2) | Some(5) | Some(10) | Some(11) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetSystemUpdateNotificationEventForContentDelivery: never signalled.
                Some(9) => {
                    let h = self.alloc_event("ns:system-update", true);
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // `IVulnerabilityManagerInterface::NeedsUpdateVulnerability` -> false.
            // The web applet aborts (2010-0221) without an answer.
            "ns:vm" if cmd_id == Some(1200) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
            "ns:am2" | "ns:ec" | "ns:rid" | "ns:rt" | "ns:web" | "ns:ro" | "ns:vm" | "ns:dev" => {
                match cmd_id {
                    // Ids from libnx's `nsGet*Interface`; 7990 is unassigned.
                    Some(7988) => self.ns_reply_with_interface(tls, handle, "ns:dynamic-rights"),
                    Some(7989) => self.ns_reply_with_interface(tls, handle, "ns:read-only-control"),
                    Some(7991) => self.ns_reply_with_interface(tls, handle, "ns:read-only-record"),
                    Some(7992) => self.ns_reply_with_interface(tls, handle, "ns:ecommerce"),
                    Some(7993) => self.ns_reply_with_interface(tls, handle, "ns:app-version"),
                    Some(7994) => self.ns_reply_with_interface(tls, handle, "ns:factory-reset"),
                    Some(7995) => self.ns_reply_with_interface(tls, handle, "ns:account-proxy"),
                    Some(7996) => self.ns_reply_with_interface(tls, handle, "ns:app-manager"),
                    Some(7997) => self.ns_reply_with_interface(tls, handle, "ns:download-task"),
                    Some(7998) => {
                        self.ns_reply_with_interface(tls, handle, "ns:content-management")
                    }
                    Some(7999) => self.ns_reply_with_interface(tls, handle, "ns:document"),
                    _ => self.unimplemented_command(tls, &iface, cmd_id),
                }
            }
            // `ns:am` is the pre-3.0.0 service, with no getter in between.
            "ns:am" | "ns:app-manager" => self.ns_application_manager_request(tls, &iface, cmd_id),
            // `IContentManagementInterface`.
            "ns:content-management" => match cmd_id {
                // CheckSdCardMountStatus: the emulated card is always mounted.
                Some(43) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetTotalSpaceSize / GetFreeSpaceSize(StorageId) -> u64. The
                // same 32 GiB card `fsp-srv` reports, half used.
                Some(47) => {
                    self.write_ipc_response(tls, 0, &[], &SD_TOTAL_SPACE.to_le_bytes(), &[])
                }
                Some(48) => self.write_ipc_response(tls, 0, &[], &SD_FREE_SPACE.to_le_bytes(), &[]),
                // CountApplicationContentMeta(u64 application_id) -> 0.
                Some(600) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // `IDownloadTaskInterface`: nothing to download.
            "ns:download-task" => match cmd_id {
                // EnableAutoCommit / DisableAutoCommit.
                Some(707) | Some(708) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // `IDynamicRightsInterface`: licence-sharing checks before launch.
            "ns:dynamic-rights" => match cmd_id {
                // HasAccountRestrictedRightsInRunningApplications -> false.
                Some(26) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // `IReadOnlyApplicationRecordInterface`.
            "ns:read-only-record" => match cmd_id {
                // HasApplicationRecord(u64 application_id) -> false.
                Some(0) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                // ListApplicationRecord(s32 entry_offset): none, count written.
                Some(3) => self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }

    /// Hand out one of `ns`'s sub-interfaces (no input, one out-interface).
    fn ns_reply_with_interface(&mut self, tls: u32, handle: u64, name: &str) -> Result<()> {
        self.reply_with_interface(tls, handle, name)?;
        Ok(())
    }

    /// `aoc:u` (`IAddOnContentManager`): the add-on content registered through
    /// [`Cpu::add_add_on_content`], empty by default.
    pub(super) fn aoc_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "aoc:u");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
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
            // ListAddOnContent(u32 offset, u32 count) -> u32 written, indices in
            // an out buffer.
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
            // GetAddOnContentBaseId -> u64.
            Some(5) => {
                let base = self.add_on_content_base_id();
                self.write_ipc_response(tls, 0, &[], &base.to_le_bytes(), &[])
            }
            // PrepareAddOnContent(s32 index): registered content is already ready.
            Some(7) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // GetAddOnContentListChangedEvent (and …WithProcessId): never signalled.
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
            // GetAddOnContentLostErrorCode: nothing can be lost mid-run.
            Some(9) => self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[]),
            // NotifyMountAddOnContent / NotifyUnmountAddOnContent.
            Some(11) | Some(12) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // IsAddOnContentMountedForDebug -> bool.
            Some(13) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
            // CheckAddOnContentMountStatus: the Result is the answer; content
            // is never removed.
            Some(50) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            _ => self.unimplemented_command(tls, "aoc:u", cmd_id),
        }
    }

    /// `caps:a` (`IAlbumAccessorService`): an album that is mounted and empty.
    /// Reporting it unmounted would be the card-removed error.
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
            // GetAlbumFileCount / …Ex0 (AlbumStorage[, flags]) -> 0.
            Some(0) | Some(100) => self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[]),
            // GetAlbumFileList / …Ex0 -> 0 entries written.
            Some(1) | Some(101) => self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[]),
            // DeleteAlbumFile(AlbumFileId): no list names a file.
            Some(3) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // IsAlbumMounted(AlbumStorage) -> bool.
            Some(5) => self.write_ipc_response(tls, 0, &[], &[1u8], &[]),
            // GetAlbumMountResult(AlbumStorage) -> success.
            Some(16) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // Unnamed (Eden's `Unknown18`): a written length of zero. The Album
            // applet issues it first, with a 0x40-byte buffer.
            Some(18) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            // GetAutoSavingStorage -> bool: false, no SD card.
            Some(401) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
            // GetAlbumAccessResultForDebug -> success, in the result and the data.
            Some(50011) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            _ => self.unimplemented_command(tls, "caps:a", cmd_id),
        }
    }

    /// `IApplicationManagerInterface`.
    fn ns_application_manager_request(
        &mut self,
        tls: u32,
        iface: &str,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        match cmd_id {
            // ListApplicationRecord(s32 entry_offset) -> (s32 count, records).
            // The zero count must be written.
            Some(0) => self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[]),
            // GenerateApplicationRecordCount -> u64.
            Some(1) => self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[]),
            // GetApplicationRecordUpdateSystemEvent: signalled, one per process.
            // The Home Menu waits on it before reading the title list.
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
            // CheckSdCardMountStatus and storage space, as `ns:content-management`.
            Some(43) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            Some(47) => self.write_ipc_response(tls, 0, &[], &SD_TOTAL_SPACE.to_le_bytes(), &[]),
            Some(48) => self.write_ipc_response(tls, 0, &[], &SD_FREE_SPACE.to_le_bytes(), &[]),
            // GetStorageSize(u8 storage_id) -> (s64 total, s64 free).
            Some(71) => {
                let mut out = [0u8; 16];
                out[..8].copy_from_slice(&SD_TOTAL_SPACE.to_le_bytes());
                out[8..].copy_from_slice(&SD_FREE_SPACE.to_le_bytes());
                self.write_ipc_response(tls, 0, &[], &out, &[])
            }
            // ResumeAll: no download tasks.
            Some(70) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // The media events: never signalled, the same object every time.
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
            // 20.0.0+, unnamed: event getters, signalled (as in Eden), one
            // object per command.
            Some(cmd @ (4022 | 4088)) => {
                let h = match self.ns_manager_events.get(&cmd) {
                    Some(&h) => h,
                    None => {
                        let name = if cmd == 4022 {
                            "ns:app-manager-4022"
                        } else {
                            "ns:app-manager-4088"
                        };
                        let h = self.alloc_event(name, false);
                        self.signal_event(h);
                        self.ns_manager_events.insert(cmd, h);
                        h
                    }
                };
                self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
            }
            // 20.0.0+, unnamed: one u64 out, 0 (as in Eden).
            Some(4023) => self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[]),
            _ => self.unimplemented_command(tls, iface, cmd_id),
        }
    }

    /// `prepo:u` and its privileged aliases (`IPrepoService`): play reports
    /// are accepted and dropped; the queue is always empty.
    pub(super) fn prepo_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            // Reports travel in map-alias buffers.
            return self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[]);
        }
        match cmd_id {
            // SaveReport / SaveReportWithUser (all four revisions) and the
            // SaveSystemReport pair.
            Some(10100..=10107) | Some(20100..=20103) => {
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // RequestImmediateTransmission: nothing queued.
            Some(10200) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // GetTransmissionStatus -> s32: 0, idle.
            Some(10300) => self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[]),
            _ => self.unimplemented_command(tls, "prepo:u", cmd_id),
        }
    }

    /// `pdm:qry` (`IQueryService`): an empty play history, as on a
    /// factory-fresh console.
    pub(super) fn pdm_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[]);
        }
        match cmd_id {
            // QueryAppletEvent, QueryPlayEvent, QueryAccountEvent,
            // QueryAccountPlayEvent, QueryRecentlyPlayedApplication: none.
            Some(0) | Some(5) | Some(7) | Some(8) | Some(11) => {
                self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[])
            }
            // QueryPlayStatisticsByApplicationId / ...AndUserAccountId: zeroed.
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
        let mut cpu = request(false, 5, &[]);
        cpu.pdm_request(TLS, Some(5)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0);

        let mut cpu = request(false, 2, &[0u8; 8]);
        cpu.pdm_request(TLS, Some(2)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.read_bytes(TLS + 0x20, 0x20), vec![0u8; 0x20]);
    }

    #[test]
    fn ns_reports_the_sd_card_as_mounted() {
        let mut cpu = request(false, 43, &[]);
        cpu.register_service_handle(9, "ns:content-management");
        cpu.ns_request(TLS, 9, Some(43)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x18).unwrap(),
            0,
            "the card reads as missing"
        );

        // Free must not exceed total.
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
    fn prepo_accepts_a_report_that_goes_nowhere() {
        for cmd in [10100u32, 10107, 10200, 20102] {
            let mut cpu = request(false, cmd, &[]);
            cpu.register_service_handle(9, "prepo:u");
            cpu.prepo_request(TLS, Some(cmd)).unwrap();
            assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "prepo {cmd}");
        }
        let mut cpu = request(false, 10300, &[]);
        cpu.register_service_handle(9, "prepo:u");
        cpu.prepo_request(TLS, Some(10300)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "status");
    }

    #[test]
    fn ns_reports_no_account_restricted_by_dynamic_rights() {
        let mut cpu = request(false, 26, &[]);
        cpu.register_service_handle(9, "ns:dynamic-rights");
        cpu.ns_request(TLS, 9, Some(26)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 0, "restricted");
    }

    #[test]
    fn ns_reports_a_console_with_nothing_installed() {
        // The count is written even when zero.
        let mut cpu = request(false, 7996, &[]);
        cpu.register_service_handle(9, "ns:am2");
        cpu.ns_request(TLS, 9, Some(7996)).unwrap();
        let manager = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;

        write_request(&mut cpu, 0, &0i32.to_le_bytes()); // ListApplicationRecord
        cpu.ns_request(TLS, manager, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "record count");

        // Not even the running title has a record.
        let mut cpu = request(false, 7991, &[]);
        cpu.register_service_handle(9, "ns:am2");
        cpu.ns_request(TLS, 9, Some(7991)).unwrap();
        let records = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;

        write_request(
            &mut cpu,
            0,
            &crate::cpu::ipc::DEFAULT_PROGRAM_ID.to_le_bytes(),
        );
        cpu.ns_request(TLS, records, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 0, "has record");

        cpu.mem.write_u32(TLS + 0x20, 0xFFFF_FFFF).unwrap();
        write_request(&mut cpu, 3, &0i32.to_le_bytes());
        cpu.ns_request(TLS, records, Some(3)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "record count");
    }

    #[test]
    fn ns_reports_a_command_it_does_not_implement_rather_than_succeeding() {
        // Unimplemented commands must fail, not be faked.
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

        let mut cpu = request(false, 2, &[]);
        aoc(&mut cpu, 2);
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "Result");
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "count");

        // GetAddOnContentBaseId -> program id + 0x1000.
        let mut cpu = request(false, 5, &[]);
        cpu.set_program_id(PROGRAM_ID);
        aoc(&mut cpu, 5);
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), PROGRAM_ID + 0x1000);

        let mut cpu = request(false, 50, &[]);
        aoc(&mut cpu, 50);
        assert_eq!(cpu.mem.read_u32(TLS + 0x10).unwrap(), SFCO);
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "Result");
    }

    #[test]
    fn registered_add_on_content_is_counted_listed_and_mountable() {
        const PROGRAM_ID: u64 = 0x0100_BEE0_17FC_0000;
        const BUFFER: u32 = 0x4000;
        // Just Dance 2023's two DLC containers: indices 1 and 4.
        let content = [PROGRAM_ID + 0x1001, PROGRAM_ID + 0x1004];

        let register = |cpu: &mut Cpu| {
            cpu.set_program_id(PROGRAM_ID);
            for id in content {
                let src = crate::source::MemSource(vec![0u8; 0x20]);
                assert_eq!(
                    cpu.add_add_on_content(id, Box::new(src)),
                    Some((id - PROGRAM_ID - 0x1000) as u32)
                );
            }
        };

        let mut cpu = request(false, 2, &[]);
        register(&mut cpu);
        aoc(&mut cpu, 2);
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "Result");
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 2, "count");

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

        // An offset past the end is the end of the list, not an error.
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

        // `fsp-srv`'s OpenDataStorageByDataId mounts them by base id + index.
        let mut cpu = Cpu::new();
        register(&mut cpu);
        assert!(cpu.has_data_archive(content[0]));
    }

    #[test]
    fn add_on_content_belonging_to_another_title_is_refused() {
        // A DLC's id is its base title's plus an index below 0x800.
        let mut cpu = Cpu::new();
        cpu.set_program_id(0x0100_BEE0_17FC_0000);
        let src = || Box::new(crate::source::MemSource(vec![0u8; 0x10]));
        assert_eq!(cpu.add_add_on_content(0x0100_0000_0000_1001, src()), None);
        assert_eq!(cpu.add_add_on_content(0x0100_BEE0_17FC_1801, src()), None);
        assert!(cpu.add_on_content().is_empty());
    }

    #[test]
    fn the_add_on_content_list_never_changes_so_its_event_never_fires() {
        // The ...WithProcessId form must hand out the same event.
        let mut cpu = request(false, 8, &[]);
        aoc(&mut cpu, 8);
        let event = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
        assert_ne!(
            event, 0,
            "GetAddOnContentListChangedEvent handed back no handle"
        );
        assert_eq!(cpu.event_name(event), Some("aoc:list-changed"));
        assert_eq!(cpu.event_signaled(event), Some(false));

        marshal(&mut cpu, false, 10, &[]);
        cpu.aoc_request(TLS, 9, Some(10)).unwrap();
        assert_eq!(u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap()), event);
    }
}
