//! `SendSyncRequest`: the IPC front door, and the routing of a request to its service's stub.

use super::RESULT_OK;
use crate::cpu::Cpu;
use crate::Result;
use std::ops::ControlFlow;

impl Cpu {
    /// SendSyncRequest variants: a named service's stub, else the generic reply.
    pub(super) fn svc_send_sync_request(&mut self) -> Result<()> {
        let tls = self.tpidr;
        let session = self.read_zr(0);
        // A Close request (type 2) has no command id; it tears down the session.
        if self.ipc_message_type(tls) == 2 {
            self.close_session(session);
            self.write_ipc_response(tls, 0, &[], &[], &[])?;
            self.write_zr(0, RESULT_OK);
            return Ok(());
        }
        // A clone reaches its owner's interface and domain objects.
        let handle = self
            .session_owners
            .get(&session)
            .copied()
            .unwrap_or(session);
        let cmd_id = self.ipc_command_id(tls);
        let svc_name = self.service_name(handle).map(|s| s.to_string());
        if crate::trace::enabled(crate::trace::Trace::Ipc) {
            let obj = self.ipc_domain_object_id(tls);
            let iface = self.domain_interface(handle, obj).map(|s| s.to_string());
            let words: Vec<String> = (0..8)
                .map(|i| format!("{:08x}", self.mem.read_u32(tls + i * 4).unwrap_or(0)))
                .collect();
            crate::traceln!(
                "[ipc] pc={:#x} h={} svc={:?} obj={} iface={:?} type={} cmd={:?} {}",
                self.pc,
                handle,
                svc_name,
                obj,
                iface,
                self.ipc_message_type(tls),
                cmd_id,
                words.join(" ")
            );
        }
        // CloneCurrentObject (control 2) and its Ex form (4) reply with a new session
        // handle as a move handle, sharing this session's interface and domain: objects
        // opened through either are reachable through both.
        if self.ipc_is_control_request(tls) && matches!(cmd_id, Some(2) | Some(4)) {
            if let Some(name) = svc_name.clone() {
                let clone = self.alloc_handle();
                self.record_handle(clone, &name);
                self.session_owners.insert(clone, handle);
                self.write_ipc_response(tls, 0, &[clone], &[], &[])?;
                self.write_zr(0, RESULT_OK);
                return Ok(());
            }
        }
        // QueryPointerBufferSize (control 3): `nnSdk` checks pointer args against it.
        if self.ipc_is_control_request(tls) && cmd_id == Some(3) {
            let size = crate::kernel::ipc::POINTER_BUFFER_SIZE.to_le_bytes();
            self.write_ipc_response(tls, 0, &[], &size, &[])?;
            self.write_zr(0, RESULT_OK);
            return Ok(());
        }
        // A domain Close has no command id; handle it before any service sees it.
        if self.ipc_is_domain_close(tls) {
            let object_id = self.ipc_domain_object_id(tls);
            self.close_domain_object(tls, handle, object_id)?;
            self.write_zr(0, RESULT_OK);
            return Ok(());
        }
        let flow = match svc_name {
            Some(name) => self.system_service_request(tls, handle, cmd_id, &name)?,
            None => self.untracked_session_request(tls, handle, cmd_id)?,
        };
        if flow.is_break() {
            return Ok(());
        }
        // A service that could not answer wrote no reply, so the request is still in the
        // buffer: rewind onto the `svc` and run the call again after the wait.
        if let Some(until) = std::mem::take(&mut self.pending_retry) {
            self.pc = self.pc.wrapping_sub(4);
            self.sleep_until(until);
            return Ok(());
        }
        self.write_zr(0, RESULT_OK);
        // Yield after X0 is written: `yield_thread` swaps register files.
        if std::mem::take(&mut self.pending_yield) {
            self.yield_thread();
        }
        // Same for a service that asked to sleep until a deadline (`vi`'s present).
        if let Some(until) = std::mem::take(&mut self.pending_sleep) {
            self.sleep_until(until);
        }
        Ok(())
    }

    /// A named session's request: system services here, the rest in the next router.
    fn system_service_request(
        &mut self,
        tls: u64,
        handle: u64,
        cmd_id: Option<u32>,
        name: &str,
    ) -> Result<ControlFlow<()>> {
        match name {
            "sm:" | "sm" => self.sm_request(tls, cmd_id, handle)?,
            "fsp-srv" | "fsp-srv:" => {
                // Domain sub-objects on fsp-srv, routed by recorded interface.
                let object_id = self.ipc_domain_object_id(tls);
                match self.domain_interface(handle, object_id) {
                    Some("fsp-srv-fs") => self.fs_request(tls, cmd_id, handle)?,
                    Some("fsp-srv-fs-dir") => {
                        self.fs_dir_request(tls, cmd_id, Self::object_key(handle, object_id))?
                    }
                    Some("fsp-srv-fs-file") => {
                        self.fs_file_request(tls, cmd_id, Self::object_key(handle, object_id))?
                    }
                    Some("fsp-srv-storage") => self.fs_storage_request(tls, handle, cmd_id)?,
                    Some("fsp-srv-save-info-reader") => {
                        self.fs_save_data_info_reader_request(tls, handle, cmd_id)?
                    }
                    Some("fsp-srv-device-operator") => {
                        self.fs_device_operator_request(tls, cmd_id)?
                    }
                    Some("fsp-srv-sd-detection") | Some("fsp-srv-gamecard-detection") => {
                        self.fs_detection_notifier_request(tls, handle, cmd_id)?
                    }
                    _ => self.fsp_srv_request(tls, cmd_id, handle)?,
                }
            }
            // The same interfaces over their own session handle (non-domain callers).
            "fsp-srv-fs" => self.fs_request(tls, cmd_id, handle)?,
            "fsp-srv-fs-dir" => self.fs_dir_request(tls, cmd_id, Self::object_key(handle, 0))?,
            "fsp-srv-fs-file" => self.fs_file_request(tls, cmd_id, Self::object_key(handle, 0))?,
            "fsp-srv-storage" => self.fs_storage_request(tls, handle, cmd_id)?,
            "fsp-srv-save-info-reader" => {
                self.fs_save_data_info_reader_request(tls, handle, cmd_id)?
            }
            "fsp-srv-device-operator" => self.fs_device_operator_request(tls, cmd_id)?,
            "fsp-srv-sd-detection" | "fsp-srv-gamecard-detection" => {
                self.fs_detection_notifier_request(tls, handle, cmd_id)?
            }
            "vi:m" | "vi:m:" => self.vi_request(tls, handle, cmd_id)?,
            // fatal:u, the guest giving up.
            "fatal:u" | "fatal:p" => self.fatal_request(tls, cmd_id)?,
            "set" => self.set_request(tls, handle, cmd_id)?,
            "set:sys" => self.set_sys_request(tls, cmd_id)?,
            "nvdrv" | "nvdrv:" | "nvdrv:a" | "nvdrv:a:" | "nvdrv:s" | "nvdrv:t" => {
                self.nvdrv_request(tls, cmd_id, handle)?
            }
            "pl:u" | "pl:s" => self.pl_request(tls, cmd_id)?,
            // caps:a, the screenshot album.
            "caps:a" => self.caps_album_accessor_request(tls, handle, cmd_id)?,
            // time:* sub-interfaces as domain out-objects.
            "time:s" | "time:u" | "time:a" | "time:r" => {
                let object_id = self.ipc_domain_object_id(tls);
                match self.domain_interface(handle, object_id) {
                    Some("time:system-clock") => self.time_system_clock_request(tls, cmd_id)?,
                    Some("time:steady-clock") => self.time_steady_clock_request(tls, cmd_id)?,
                    Some("time:timezone") => self.time_timezone_request(tls, cmd_id)?,
                    _ => self.time_request(tls, cmd_id, handle)?,
                }
            }
            // The same over their own session handle.
            "time:system-clock" => self.time_system_clock_request(tls, cmd_id)?,
            "time:steady-clock" => self.time_steady_clock_request(tls, cmd_id)?,
            "time:timezone" => self.time_timezone_request(tls, cmd_id)?,
            // psm (power state management).
            "psm" => {
                let object_id = self.ipc_domain_object_id(tls);
                match self.domain_interface(handle, object_id) {
                    Some("psm-session") => self.psm_session_request(tls, cmd_id)?,
                    _ => self.psm_request(tls, cmd_id, handle)?,
                }
            }
            "psm-session" => self.psm_session_request(tls, cmd_id)?,
            _ => return self.application_service_request(tls, handle, cmd_id, name),
        }
        Ok(ControlFlow::Continue(()))
    }

    /// Continues [`Self::system_service_request`] with `appletOE` through `pm`.
    fn application_service_request(
        &mut self,
        tls: u64,
        handle: u64,
        cmd_id: Option<u32>,
        name: &str,
    ) -> Result<ControlFlow<()>> {
        match name {
            // appletOE / appletAE and the `am` sub-interfaces over their own handles.
            "appletOE"
            | "appletAE"
            | "am:proxy-service"
            | "am:application-proxy"
            | "am:common-state-getter"
            | "am:self-controller"
            | "am:window-controller"
            | "am:audio-controller"
            | "am:display-controller"
            | "am:library-applet-creator"
            | "am:application-functions"
            | "am:library-applet-proxy"
            | "am:system-applet-proxy"
            | "am:library-applet-self-accessor"
            | "am:applet-common-functions"
            | "am:system-process-common-functions"
            | "am:application-observer"
            | "am:applet-alternative-functions"
            | "am:process-winding-controller"
            | "am:home-menu-functions"
            | "am:global-state-controller"
            | "am:cradle-firmware-updater"
            | "am:application-creator"
            | "am:lock-accessor"
            | "am:storage"
            | "am:storage-accessor"
            | "am:debug-functions" => self.applet_request(tls, handle, cmd_id)?,
            // nifm at all three privilege levels.
            "nifm:u" | "nifm:s" | "nifm:a" => {
                let object_id = self.ipc_domain_object_id(tls);
                match self.domain_interface(handle, object_id) {
                    Some("nifm:general-service") => {
                        self.nifm_general_service_request(tls, handle, cmd_id)?
                    }
                    Some("nifm:request") => {
                        self.nifm_request_object_request(tls, handle, cmd_id)?
                    }
                    _ => self.nifm_request(tls, cmd_id, handle)?,
                }
            }
            // The same two over their own session handles.
            "nifm:general-service" => self.nifm_general_service_request(tls, handle, cmd_id)?,
            "nifm:request" => self.nifm_request_object_request(tls, handle, cmd_id)?,
            // ssl and its contexts.
            "ssl" | "ssl:s" | "ssl:service" | "ssl:context" => {
                self.ssl_request(tls, handle, cmd_id)?
            }
            // hid, IAppletResource and IActiveVibrationDeviceList.
            "hid"
            | "hid:dbg"
            | "hid:sys"
            | "hid:server"
            | "hid:applet-resource"
            | "hid:vibration-devices" => self.hid_request(tls, handle, cmd_id)?,
            // lm, the log manager.
            "lm" | "lm:service" | "lm:logger" => self.lm_request(tls, handle, cmd_id)?,
            // acc, the user accounts.
            "acc:u0" | "acc:u1" | "acc:su" | "acc:profile" | "acc:profile-editor"
            | "acc:manager" | "acc:administrator" | "acc:async-context" | "acc:notifier" => {
                self.acc_request(tls, handle, cmd_id)?
            }
            // ns, installed content.
            "ns:am"
            | "ns:am2"
            | "ns:ec"
            | "ns:rid"
            | "ns:rt"
            | "ns:web"
            | "ns:ro"
            | "ns:su"
            | "ns:system-update-control"
            | "ns:vm"
            | "ns:dev"
            | "ns:app-manager"
            | "ns:read-only-record"
            | "ns:read-only-control"
            | "ns:content-management"
            | "ns:download-task"
            | "ns:account-proxy"
            | "ns:app-version"
            | "ns:factory-reset"
            | "ns:ecommerce"
            | "ns:dynamic-rights"
            | "ns:document"
            | "ns:request-server-stopper"
            | "ns:async-latest-update"
            | "ns:async-unnamed" => self.ns_request(tls, handle, cmd_id)?,
            // aoc, add-on content.
            "aoc:u" | "aoc:purchase-events" => self.aoc_request(tls, handle, cmd_id)?,
            // ldr:ro, used by `nn::ro`.
            "ldr:ro" => self.ldr_ro_request(tls, handle, cmd_id)?,
            // csrng and `spl:`.
            "csrng" => self.csrng_request(tls, cmd_id)?,
            "spl:" | "spl:mig" | "spl:fs" | "spl:ssl" | "spl:es" | "spl:manu" => {
                self.spl_request(tls, cmd_id)?
            }
            "es" => self.es_request(tls, cmd_id)?,
            // pdm, play history.
            "pdm:qry" | "pdm:ntfy" | "pdm:info" => self.pdm_request(tls, cmd_id)?,
            // prepo, play reports.
            "prepo:u" | "prepo:a" | "prepo:a2" | "prepo:m" | "prepo:s" => {
                self.prepo_request(tls, cmd_id)?
            }
            // pm, the process manager.
            "pm:shell" | "pm:dmnt" | "pm:info" | "pm:bm" => self.pm_request(tls, handle, cmd_id)?,
            _ => return self.other_service_request(tls, handle, cmd_id, name),
        }
        Ok(ControlFlow::Continue(()))
    }

    /// Continues [`Self::application_service_request`] with `pcv` through `erpt`.
    fn other_service_request(
        &mut self,
        tls: u64,
        handle: u64,
        cmd_id: Option<u32>,
        name: &str,
    ) -> Result<ControlFlow<()>> {
        match name {
            // pcv and clkrst, the clock manager.
            "pcv" | "clkrst" | "clkrst:i" | "clkrst:session-0" | "clkrst:session-1"
            | "clkrst:session-2" | "clkrst:session-3" => self.pcv_request(tls, handle, cmd_id)?,
            // mm:u, multimedia clocks.
            "mm:u" => self.mm_request(tls, cmd_id)?,
            // ts, temperature sensors.
            "ts" | "ts:u" | "ts:s" | "ts:session-internal" | "ts:session-external" => {
                self.ts_request(tls, handle, cmd_id)?
            }
            // sfdnsres, the DNS resolver.
            "sfdnsres" => self.sfdnsres_request(tls, cmd_id)?,
            // nsd, the environment's host names.
            "nsd:a" | "nsd:u" => self.nsd_request(tls, handle, cmd_id)?,
            // bsd, sockets.
            "bsd:u" | "bsd:s" => self.bsd_request(tls, handle, cmd_id)?,
            // apm, clock profiles.
            "apm" | "apm:p" | "apm:am" | "apm:sys" | "apm:session" => {
                self.apm_request(tls, handle, cmd_id)?
            }
            // pctl and IParentalControlService.
            "pctl" | "pctl:s" | "pctl:a" | "pctl:r" | "pctl:factory" | "pctl:service" => {
                self.pctl_request(tls, handle, cmd_id)?
            }
            // audout, PCM output.
            "audout:u" | "audout:a" | "audout:d" => self.audout_request(tls, cmd_id)?,
            "audout:iaudioout" => self.audio_out_request(tls, cmd_id, handle)?,
            // psc, power-state notifications.
            "psc:m" | "psc:service" | "psc:module" => self.psc_request(tls, handle, cmd_id)?,
            // gpio and its pad sessions.
            "gpio" | "gpio:pad" => self.gpio_request(tls, handle, cmd_id)?,
            // Mii database.
            "mii:e" | "mii:u" | "mii:s" => self.mii_request(tls, handle, cmd_id)?,
            "mii:database" | "mii:static" => self.mii_request(tls, handle, cmd_id)?,
            "miiimg" => self.miiimg_request(tls, cmd_id)?,
            // hwopus, the Opus decoder.
            "hwopus" | "hwopus:decoder" => self.hwopus_request(tls, handle, cmd_id)?,
            "audren:u" => self.audren_request(tls, handle, cmd_id)?,
            "audren:iaudiorenderer" => self.audren_renderer_request(tls, cmd_id, handle)?,
            "audren:iaudiodevice" => self.audio_device_request(tls, cmd_id, handle)?,
            // lbl, the backlight.
            "lbl" => self.lbl_request(tls, handle, cmd_id)?,
            // audctl, system audio settings.
            "audctl" => self.audctl_request(tls, handle, cmd_id)?,
            // nfc:sys; no reader attached.
            "nfc:sys" | "nfc:system" => self.nfc_request(tls, handle, cmd_id)?,
            // nfp:sys and nfp:user, amiibo; no reader attached.
            "nfp:sys" | "nfp:system" | "nfp:user" | "nfp:iuser" => {
                self.nfp_request(tls, handle, cmd_id)?
            }
            // btm:sys, the Bluetooth manager.
            "btm:sys" | "btm:core" => self.btm_request(tls, handle, cmd_id)?,
            // ngc, the profanity filter.
            "ngc:u" | "ngct:u" | "ngct:s" => self.ngc_request(tls, handle, cmd_id)?,
            "npns:s" | "npns:u" => self.npns_request(tls, handle, cmd_id)?,
            // ldn:m and lp2p:m, local wireless monitors; ldn:u, local wireless itself.
            "ldn:m" | "ldn:monitor" => self.ldn_monitor_request(tls, handle, cmd_id)?,
            "ldn:u" | "ldn:user" | "ldn:client-monitor" => {
                self.ldn_user_request(tls, handle, cmd_id)?
            }
            "lp2p:m" | "lp2p:monitor" => self.lp2p_monitor_request(tls, handle, cmd_id)?,
            // ovln, the overlay message queue.
            "ovln:snd" | "ovln:rcv" | "ovln:sender" | "ovln:receiver" => {
                self.ovln_request(tls, handle, cmd_id)?
            }
            // olsc, save-data cloud backup.
            "olsc:s"
            | "olsc:system-service"
            | "olsc:transfer-task-list"
            | "olsc:remote-storage"
            | "olsc:daemon"
            | "olsc:transfer-end-holder"
            | "olsc:transfer-start-holder"
            | "olsc:error-holder"
            | "olsc:stopper" => self.olsc_request(tls, handle, cmd_id)?,
            // friend and its IServiceCreator interfaces.
            "friend:u"
            | "friend:v"
            | "friend:m"
            | "friend:s"
            | "friend:a"
            | "friend:service"
            | "friend:notification"
            | "friend:daemon-suspend-session" => self.friend_request(tls, handle, cmd_id)?,
            // news and its article store.
            "news:a"
            | "news:c"
            | "news:m"
            | "news:p"
            | "news:v"
            | "news:service"
            | "news:arrival-event"
            | "news:overwrite-event"
            | "news:data"
            | "news:database" => self.news_request(tls, handle, cmd_id)?,
            // nim:eca, the eShop's web requests.
            "nim:eca" | "nim:shop-server" | "nim:shop-accessor" | "nim:shop-async" => {
                self.nim_eca_request(tls, handle, cmd_id)?
            }
            // bcat, background delivery.
            "bcat:a" | "bcat:m" | "bcat:u" | "bcat:s" | "bcat:service" | "bcat:storage"
            | "bcat:progress" | "bcat:notifier" | "bcat:suspension" | "bcat:file"
            | "bcat:directory" => self.bcat_request(tls, handle, cmd_id)?,
            // notif, alarms and notifications.
            "notif:a" | "notif:s" | "notif:event-accessor" => {
                self.notif_request(tls, handle, cmd_id)?
            }
            // ntc, the network clock's synchronisation.
            "ntc" | "ntc:ensure" => self.ntc_request(tls, handle, cmd_id)?,
            // usb:ds and usb:hs, with nothing plugged in.
            "usb:ds" | "usb:ds-service" | "usb:ds-interface" | "usb:ds-endpoint" => {
                self.usb_ds_request(tls, handle, cmd_id)?
            }
            "usb:hs" => self.usb_hs_request(tls, handle, cmd_id)?,
            // erpt, error reports.
            "erpt:c" => self.erpt_context_request(tls, handle, cmd_id)?,
            "erpt:r" | "erpt:report" | "erpt:manager" | "erpt:attachment" => {
                self.erpt_session_request(tls, handle, cmd_id)?
            }
            _ => return self.unstubbed_service_request(tls, handle, cmd_id, name),
        }
        Ok(ControlFlow::Continue(()))
    }

    /// Known service without a stub: reply with a fabricated sub-session
    /// (`reply_with_fabricated_object`).
    fn unstubbed_service_request(
        &mut self,
        tls: u64,
        handle: u64,
        cmd_id: Option<u32>,
        name: &str,
    ) -> Result<ControlFlow<()>> {
        // Control commands first: a fabricated pointer buffer size would
        // make callers use pointer buffers, which this IPC layer doesn't read.
        if self.ipc_is_control_request(tls) {
            match cmd_id {
                Some(0) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, name);
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])?;
                }
                _ => {
                    self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[])?;
                }
            }
            self.write_zr(0, RESULT_OK);
            return Ok(ControlFlow::Break(()));
        }
        self.warn_no_implementation(name, cmd_id);
        self.reply_with_fabricated_object(tls, handle, name, cmd_id)?;
        Ok(ControlFlow::Continue(()))
    }

    /// Unrecognized session: try the vi stub, else the generic object-id reply.
    fn untracked_session_request(
        &mut self,
        tls: u64,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<ControlFlow<()>> {
        if let Some(cmd) = cmd_id {
            // vi: GetIApplicationDisplayService (2) and display/session (100+).
            if cmd == 2 || cmd >= 100 {
                self.vi_request(tls, handle, cmd_id)?;
                return Ok(ControlFlow::Break(()));
            }
        }
        let start = self.ipc_reply_start(tls);
        let is_domain = self
            .mem
            .read_u32(tls.wrapping_add(u64::from(start + 0x10)))
            .unwrap_or(0)
            == 0x4943_4653;
        // A fresh object id for a caller that expects an out-object.
        self.warn_no_implementation("<untracked session>", cmd_id);
        let data = {
            let obj = self.next_object_id;
            self.next_object_id = obj.wrapping_add(1);
            obj
        };
        if is_domain {
            for i in 0..4u32 {
                let _ = self
                    .mem
                    .write_u32(tls.wrapping_add(u64::from(start + i * 4)), 0);
            }
            let _ = self
                .mem
                .write_u32(tls.wrapping_add(u64::from(start + 0x10)), 0x4F43_4653);
            let _ = self
                .mem
                .write_u32(tls.wrapping_add(u64::from(start + 0x14)), 0);
            let _ = self
                .mem
                .write_u32(tls.wrapping_add(u64::from(start + 0x18)), 0);
            let _ = self
                .mem
                .write_u32(tls.wrapping_add(u64::from(start + 0x1C)), 0);
            let _ = self
                .mem
                .write_u32(tls.wrapping_add(u64::from(start + 0x20)), data);
            let _ = self
                .mem
                .write_u32(tls.wrapping_add(u64::from(start + 0x24)), 0);
            let _ = self
                .mem
                .write_u32(tls.wrapping_add(u64::from(start + 0x28)), data);
        } else {
            let _ = self
                .mem
                .write_u32(tls.wrapping_add(u64::from(start)), 0x4F43_4653);
            let _ = self
                .mem
                .write_u32(tls.wrapping_add(u64::from(start + 0x04)), 0);
            let _ = self
                .mem
                .write_u32(tls.wrapping_add(u64::from(start + 0x08)), 0);
            let _ = self
                .mem
                .write_u32(tls.wrapping_add(u64::from(start + 0x0C)), 0);
            let _ = self
                .mem
                .write_u32(tls.wrapping_add(u64::from(start + 0x10)), data);
        }
        Ok(ControlFlow::Continue(()))
    }
}
