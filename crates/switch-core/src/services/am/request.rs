//! The `appletOE`/`appletAE` command dispatch.

use super::*;

impl Cpu {
    /// `appletOE`/`appletAE` and the sub-interfaces they hand out.
    pub(crate) fn applet_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "am:proxy-service");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, "am:control", cmd_id),
            };
        }
        // Which `am` sub-interface this request is for: by domain object id, or by session handle.
        let object_id = self.ipc_domain_object_id(tls);
        let iface = if self.ipc_is_domain_request(tls) {
            self.domain_interface(handle, object_id)
                .unwrap_or("am:unknown")
                .to_string()
        } else {
            match self.service_name(handle) {
                Some("appletOE") | Some("appletAE") | None => "am:proxy-service".to_string(),
                Some(name) => name.to_string(),
            }
        };
        match iface.as_str() {
            "am:proxy-service" => match cmd_id {
                Some(0) => {
                    self.set_applet_is_application(true);
                    self.reply_with_interface(tls, handle, "am:application-proxy")?;
                    Ok(())
                }
                // OpenLibraryAppletProxy, and OpenLibraryAppletProxyOld.
                Some(200) | Some(201) => {
                    self.set_applet_is_application(false);
                    self.reply_with_interface(tls, handle, "am:library-applet-proxy")?;
                    Ok(())
                }
                // OpenSystemAppletProxy (and Ex at 110), opened by the Home Menu.
                Some(100) | Some(110) => {
                    self.set_applet_is_application(false);
                    self.reply_with_interface(tls, handle, "am:system-applet-proxy")?;
                    Ok(())
                }
                // OpenSystemApplicationProxy.
                Some(350) => {
                    self.set_applet_is_application(true);
                    self.reply_with_interface(tls, handle, "am:application-proxy")?;
                    Ok(())
                }
                // OpenOverlayAppletProxy.
                Some(300) => {
                    self.set_applet_is_application(false);
                    self.reply_with_interface(tls, handle, "am:library-applet-proxy")?;
                    Ok(())
                }
                // GetSystemProcessCommonFunctions / GetAppletAlternativeFunctions: no proxy, so the applet kind is unchanged.
                Some(450) => {
                    self.reply_with_interface(tls, handle, "am:system-process-common-functions")?;
                    Ok(())
                }
                Some(460) => {
                    self.reply_with_interface(tls, handle, "am:applet-alternative-functions")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ISystemAppletProxy's Get* accessors.
            "am:system-applet-proxy" => {
                let sub = match cmd_id {
                    Some(0) => Some("am:common-state-getter"),
                    Some(1) => Some("am:self-controller"),
                    Some(2) => Some("am:window-controller"),
                    Some(3) => Some("am:audio-controller"),
                    Some(4) => Some("am:display-controller"),
                    Some(10) => Some("am:process-winding-controller"),
                    Some(11) => Some("am:library-applet-creator"),
                    Some(20) => Some("am:home-menu-functions"),
                    Some(21) => Some("am:global-state-controller"),
                    Some(22) => Some("am:application-creator"),
                    // GetAppletCommonFunctions (10.0.0+).
                    Some(23) => Some("am:applet-common-functions"),
                    Some(1000) => Some("am:debug-functions"),
                    _ => None,
                };
                match sub {
                    Some(name) => {
                        self.reply_with_interface(tls, handle, name)?;
                        Ok(())
                    }
                    None => self.unimplemented_command(tls, &iface, cmd_id),
                }
            }
            // ILibraryAppletProxy's Get* accessors.
            "am:library-applet-proxy" => {
                let sub = match cmd_id {
                    Some(0) => Some("am:common-state-getter"),
                    Some(1) => Some("am:self-controller"),
                    Some(2) => Some("am:window-controller"),
                    Some(3) => Some("am:audio-controller"),
                    Some(4) => Some("am:display-controller"),
                    Some(10) => Some("am:process-winding-controller"),
                    Some(11) => Some("am:library-applet-creator"),
                    Some(20) => Some("am:library-applet-self-accessor"),
                    Some(21) => Some("am:applet-common-functions"),
                    Some(22) => Some("am:home-menu-functions"),
                    Some(23) => Some("am:global-state-controller"),
                    Some(1000) => Some("am:debug-functions"),
                    _ => None,
                };
                match sub {
                    Some(name) => {
                        self.reply_with_interface(tls, handle, name)?;
                        Ok(())
                    }
                    None => self.unimplemented_command(tls, &iface, cmd_id),
                }
            }
            // IApplicationProxy's Get* accessors.
            "am:application-proxy" => {
                let sub = match cmd_id {
                    Some(0) => Some("am:common-state-getter"),
                    Some(1) => Some("am:self-controller"),
                    Some(2) => Some("am:window-controller"),
                    Some(3) => Some("am:audio-controller"),
                    Some(4) => Some("am:display-controller"),
                    Some(11) => Some("am:library-applet-creator"),
                    Some(20) => Some("am:application-functions"),
                    Some(1000) => Some("am:debug-functions"),
                    _ => None,
                };
                match sub {
                    Some(name) => {
                        self.reply_with_interface(tls, handle, name)?;
                        Ok(())
                    }
                    None => self.unimplemented_command(tls, &iface, cmd_id),
                }
            }
            // ICommonStateGetter.
            "am:common-state-getter" => match cmd_id {
                // GetSettingsPlatformRegion: 1 is Global.
                Some(300) => self.write_ipc_response(tls, 0, &[], &1u8.to_le_bytes(), &[]),
                // GetOperationModeSystemInfo.
                Some(200) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
                // GetHomeButtonReaderLockAccessor / GetReaderLockAccessorEx / GetWriterLockAccessorEx.
                Some(30) | Some(31) | Some(32) => {
                    self.reply_with_interface(tls, handle, "am:lock-accessor")?;
                    Ok(())
                }
                // GetEventHandle: signalled while a message is queued.
                Some(0) => {
                    let h = match self.applet_event {
                        Some(h) => h,
                        None => {
                            let h = self.alloc_event("am:applet-message", true);
                            self.applet_event = Some(h);
                            h
                        }
                    };
                    if self.has_applet_message() {
                        self.signal_event(h);
                    }
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                // ReceiveMessage.
                Some(1) => {
                    const NO_MESSAGES: u32 = 128 | (3 << 9); // am, "no message"
                    match self.next_applet_message() {
                        Some(message) => {
                            self.write_ipc_response(tls, 0, &[], &message.to_le_bytes(), &[])
                        }
                        None => self.write_ipc_response(tls, NO_MESSAGES, &[], &[], &[]),
                    }
                }
                // GetOperationMode: Handheld is 0, Console is 1.
                Some(5) => {
                    let mode = self.operation_mode() as u32;
                    self.write_ipc_response(tls, 0, &[], &mode.to_le_bytes(), &[])
                }
                // GetPerformanceMode.
                Some(6) => {
                    let mode = self.operation_mode().performance_mode();
                    self.write_ipc_response(tls, 0, &[], &mode.to_le_bytes(), &[])
                }
                Some(9) => self.write_ipc_response(tls, 0, &[], &1u32.to_le_bytes(), &[]), // GetCurrentFocusState: InFocus
                // GetBootMode: Normal.
                Some(8) => self.write_ipc_response(tls, 0, &[], &0u8.to_le_bytes(), &[]),
                // GetAcquiredSleepLockEvent: never signalled.
                Some(13) => {
                    let h = match self.sleep_lock_event {
                        Some(h) => h,
                        None => {
                            let h = self.alloc_event("am:sleep-lock", false);
                            self.sleep_lock_event = Some(h);
                            h
                        }
                    };
                    if self.sleep_lock_acquired {
                        self.signal_event(h);
                    }
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                // GetDefaultDisplayResolutionChangeEvent: fired on dock or undock.
                Some(61) => {
                    let h = match self.display_resolution_event {
                        Some(h) => h,
                        None => {
                            let h = self.alloc_event("am:display-resolution-changed", true);
                            self.display_resolution_event = Some(h);
                            h
                        }
                    };
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                // GetDefaultDisplayResolution.
                Some(60) => {
                    let (width, height) = self.operation_mode().display_size();
                    let mut raw = Vec::with_capacity(8);
                    raw.extend_from_slice(&width.to_le_bytes());
                    raw.extend_from_slice(&height.to_le_bytes());
                    self.write_ipc_response(tls, 0, &[], &raw, &[])
                }
                // RequestToAcquireSleepLock: granted at once.
                Some(10) => {
                    self.sleep_lock_acquired = true;
                    if let Some(h) = self.sleep_lock_event {
                        self.signal_event(h);
                    }
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // ReleaseSleepLock / ReleaseSleepLockTransiently.
                Some(11) | Some(12) => {
                    self.sleep_lock_acquired = false;
                    if let Some(h) = self.sleep_lock_event {
                        self.clear_event(h);
                    }
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // SetCpuBoostMode: no clock governor to move.
                Some(66) => {
                    self.warn_stub(&iface, cmd_id, "accepted; there is no clock to move");
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // SetRequestExitToLibraryAppletAtExecuteNextProgramEnabled.
                Some(900) => {
                    self.warn_stub(&iface, cmd_id, "the exit-request latch is not recorded");
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "am:application-functions" => match cmd_id {
                // PopLaunchParameter(u32 kind) -> IStorage, handed over once.
                Some(1) => {
                    const LAUNCH_PARAMETER_NOT_FOUND: u32 = 128 | (2 << 9);
                    let kind = self.mem.read_u32(self.ipc_request_data(tls))?;
                    match self.am_launch_parameters.remove(&kind) {
                        Some(data) => {
                            let key = self.reply_with_interface(tls, handle, "am:storage")?;
                            self.am_storages.insert(key, data);
                            Ok(())
                        }
                        None => {
                            self.write_ipc_response(tls, LAUNCH_PARAMETER_NOT_FOUND, &[], &[], &[])
                        }
                    }
                }
                // EnsureSaveData.
                Some(20) => {
                    self.warn_stub(&iface, cmd_id, "0 bytes ensured; no save was created");
                    self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[])
                }
                // ExtendSaveData(u8 type, u128 uid, s64 size, s64 journal): granted and remembered.
                Some(25) => {
                    let data = self.ipc_request_data(tls);
                    self.save_data_quota.size = self.mem.read_u64(data.wrapping_add(0x18))? as i64;
                    self.save_data_quota.journal_size =
                        self.mem.read_u64(data.wrapping_add(0x20))? as i64;
                    self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[])
                }
                // GetSaveDataSize(u8 type, u128 uid) -> two s64s.
                Some(26) => {
                    let quota = self.save_data_quota;
                    self.write_save_data_pair(tls, quota.size, quota.journal_size)
                }
                // GetSaveDataSizeMax / GetDeviceSaveDataSizeMax.
                Some(28) => {
                    let quota = self.save_data_quota;
                    self.write_save_data_pair(tls, quota.size_max, quota.journal_size_max)
                }
                Some(35) => {
                    let quota = self.save_data_quota;
                    self.write_save_data_pair(
                        tls,
                        quota.device_size_max,
                        quota.device_journal_size_max,
                    )
                }
                // GetCacheStorageMax -> s32, then s64 at +8.
                Some(29) => {
                    let quota = self.save_data_quota;
                    let mut out = Vec::with_capacity(16);
                    out.extend_from_slice(&quota.cache_storage_index_max.to_le_bytes());
                    out.extend_from_slice(&[0u8; 4]);
                    out.extend_from_slice(&quota.cache_storage_size_max.to_le_bytes());
                    self.write_ipc_response(tls, 0, &[], &out, &[])
                }
                // CreateCacheStorage(u16 index, s64 size, s64 journal).
                Some(27) => {
                    let mut out = Vec::with_capacity(16);
                    out.extend_from_slice(&1u32.to_le_bytes());
                    out.extend_from_slice(&[0u8; 4]);
                    out.extend_from_slice(&0u64.to_le_bytes());
                    self.write_ipc_response(tls, 0, &[], &out, &[])
                }
                // GetDesiredLanguage -> `nn::settings::LanguageCode`.
                Some(21) => {
                    let code = self.system_settings().language_code;
                    self.write_ipc_response(tls, 0, &[], &code.to_le_bytes(), &[])
                }
                // GetDisplayVersion -> a 16-byte version string.
                Some(23) => {
                    let mut version = [0u8; 16];
                    version[..5].copy_from_slice(b"1.0.0");
                    self.write_ipc_response(tls, 0, &[], &version, &[])
                }
                // BeginBlockingHomeButton{ShortAndLongPressed,} and End.
                Some(30) | Some(31) | Some(32) | Some(33) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // NotifyRunning.
                Some(40) => self.write_ipc_response(tls, 0, &[], &1u8.to_le_bytes(), &[]),
                // GetPseudoDeviceId -> 16 bytes.
                Some(50) => {
                    self.warn_stub(&iface, cmd_id, "an all-zero device id");
                    self.write_ipc_response(tls, 0, &[], &[0u8; 16], &[])
                }
                // GetGpuErrorDetectedSystemEvent: never signalled.
                Some(130) => {
                    self.warn_stub(&iface, cmd_id, "an event nothing here ever signals");
                    let h = self.alloc_event("am:gpu-error", true);
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                // SetTerminateResult(Result).
                Some(22) => {
                    let result = self.mem.read_u32(self.ipc_request_data(tls))?;
                    self.am_terminate_result = result;
                    if result != 0 {
                        let (module, description) = (result & 0x1ff, result >> 9);
                        self.diagnostic(
                            Level::Warn,
                            &format!(
                                "[am] the title set a terminate result of {result:#x} \
                                 ({module}-{description:04})"
                            ),
                        );
                    }
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetLastApplicationExitReason -> u32.
                Some(200) => {
                    let reason = self.am_terminate_result;
                    self.write_ipc_response(tls, 0, &[], &reason.to_le_bytes(), &[])
                }
                // InitializeGamePlayRecording / SetGamePlayRecordingState / SetDelayTimeToAbortOnGpuError.
                Some(66) | Some(67) | Some(131) => {
                    self.warn_stub(&iface, cmd_id, "accepted and not recorded");
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // Copyright frame buffer, image and visibility.
                Some(100) | Some(101) | Some(102) => {
                    self.warn_stub(&iface, cmd_id, "accepted, and no capture draws it");
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // 210 (20.0.0+, unnamed): one out event, never signalled.
                Some(210) => {
                    self.warn_stub(&iface, cmd_id, "an event nothing here ever signals");
                    let h = match self.application_functions_210_event {
                        Some(h) => h,
                        None => {
                            let h = self.alloc_event("am:application-functions-210", true);
                            self.application_functions_210_event = Some(h);
                            h
                        }
                    };
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ISelfController.
            "am:self-controller" => match cmd_id {
                // Setters and notifiers whose whole reply is a Result.
                Some(0..=4) | Some(10..=16) | Some(19) | Some(51) | Some(60) | Some(64)
                | Some(65) | Some(72) | Some(100) | Some(110) | Some(130) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // Set/GetIdleTimeDetectionExtension, SetAutoSleepDisabled / IsAutoSleepDisabled.
                Some(62) => {
                    let data = self.ipc_request_data(tls);
                    self.idle_time_detection_extension = self.mem.read_u32(data).unwrap_or(0);
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                Some(63) => {
                    let extension = self.idle_time_detection_extension;
                    self.write_ipc_response(tls, 0, &[], &extension.to_le_bytes(), &[])
                }
                Some(68) => {
                    let data = self.ipc_request_data(tls);
                    self.auto_sleep_disabled = self.mem.read_u8(data).unwrap_or(0) != 0;
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                Some(69) => {
                    let disabled = u8::from(self.auto_sleep_disabled);
                    self.write_ipc_response(tls, 0, &[], &[disabled], &[])
                }
                // SetHandlesRequestToDisplay: queues `RequestToDisplay`.
                Some(50) => {
                    let data = self.ipc_request_data(tls);
                    if self.mem.read_u8(data).unwrap_or(0) != 0 {
                        self.queue_applet_message(super::AppletMessage::RequestToDisplay);
                    }
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetLibraryAppletLaunchableEvent: signalled.
                Some(9) => {
                    let h = self.kept_event("am:library-applet-launchable", handle);
                    self.signal_event(h);
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                // GetAccumulatedSuspendedTickChangedEvent.
                Some(91) => {
                    self.warn_stub(&iface, cmd_id, "an event nothing here ever signals");
                    let h = self.kept_event("am:accumulated-suspended-tick-changed", handle);
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                // Unknown230(u32) -> u16.
                Some(230) => {
                    self.warn_stub(&iface, cmd_id, "an unknown command, answered 0");
                    self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[])
                }
                // GetAccumulatedSuspendedTickValue.
                Some(90) => self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[]),
                // IsSystemBufferSharingEnabled: false.
                Some(41) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetSystemSharedBufferHandle -> buffer id;
                // GetSystemSharedLayerHandle -> buffer id + layer id.
                Some(43) => self.write_ipc_response(tls, 0, &[], &1u64.to_le_bytes(), &[]),
                Some(42) => {
                    let mut raw = Vec::with_capacity(16);
                    raw.extend_from_slice(&1u64.to_le_bytes());
                    raw.extend_from_slice(&1u64.to_le_bytes());
                    self.write_ipc_response(tls, 0, &[], &raw, &[])
                }
                // CreateManagedDisplayLayer: `vi` models one layer, id 1.
                Some(40) => self.write_ipc_response(tls, 0, &[], &1u64.to_le_bytes(), &[]),
                // CreateManagedDisplaySeparableLayer.
                Some(44) => {
                    // The recording layer is 0 so the caller does not open layer 1 twice.
                    let mut raw = Vec::with_capacity(16);
                    raw.extend_from_slice(&1u64.to_le_bytes());
                    raw.extend_from_slice(&0u64.to_le_bytes());
                    self.write_ipc_response(tls, 0, &[], &raw, &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IWindowController and IDisplayController.
            "am:display-controller" => match cmd_id {
                // Acquire*CaptureSharedBuffer: a real, never-written slot past the two framebuffers.
                Some(22) | Some(24) | Some(26) => {
                    let mut raw = Vec::with_capacity(8);
                    raw.extend_from_slice(&[1u8, 0, 0, 0]); // was_written = true
                    raw.extend_from_slice(
                        &(super::SHARED_BUFFER_USABLE_SLOTS as i32).to_le_bytes(),
                    );
                    self.write_ipc_response(tls, 0, &[], &raw, &[])
                }
                // Capture releases and clears.
                Some(8) | Some(20) | Some(21) | Some(23) | Some(25) | Some(27) | Some(28) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // Update{LastForeground,CallerApplet}CaptureImage.
                Some(1) | Some(4) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // Get*CaptureImageEx: a black 1280x720 RGBA8888 image into the out buffer.
                Some(5) | Some(6) | Some(7) => {
                    if let Some((addr, size)) = self.ipc_output_buffer(tls, 0) {
                        let page = crate::mem::PAGE_SIZE as u32;
                        let end = addr.saturating_add(size);
                        let mut at = addr;
                        while at < end {
                            let run = (page - at % page).min(end - at);
                            if self.mem.fill_le(at, 1, 0, run).is_err() {
                                break;
                            }
                            at += run;
                        }
                    }
                    self.write_ipc_response(tls, 0, &[], &1u8.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "am:window-controller" => match cmd_id {
                // GetAppletResourceUserId / GetAppletResourceUserIdOfCallerApplet.
                Some(1) | Some(2) => self.write_ipc_response(tls, 0, &[], &1u64.to_le_bytes(), &[]),
                // AcquireForegroundRights / ReleaseForegroundRights / RejectToChangeIntoBackground.
                Some(10) | Some(11) | Some(12) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IAudioController.
            "am:audio-controller" => match cmd_id {
                // SetExpectedMasterVolume / ChangeMainAppletMasterVolume /
                // SetTransparentVolumeRate.
                Some(0) | Some(3) | Some(4) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // Get{Main,Library}AppletExpectedMasterVolume -> an f32.
                Some(1) | Some(2) => {
                    self.write_ipc_response(tls, 0, &[], &1.0f32.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IAppletCommonFunctions.
            "am:applet-common-functions" => match cmd_id {
                // Set/GetHomeButtonDoubleClickEnabled.
                Some(50) => {
                    let data = self.ipc_request_data(tls);
                    self.home_button_double_click_enabled =
                        self.mem.read_u8(data).unwrap_or(0) != 0;
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                Some(51) => {
                    let enabled = u8::from(self.home_button_double_click_enabled);
                    self.write_ipc_response(tls, 0, &[], &[enabled], &[])
                }
                // SetCpuBoostRequestPriority.
                Some(70) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // 20.0.0+, unnamed: one u16 out (per Eden).
                Some(350) => self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ISystemProcessCommonFunctions: hands back an IApplicationObserver.
            "am:system-process-common-functions" => match cmd_id {
                Some(1) => {
                    self.reply_with_interface(tls, handle, "am:application-observer")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IHomeMenuFunctions.
            "am:home-menu-functions" => match cmd_id {
                // RequestToGetForeground / LockForeground / UnlockForeground.
                Some(10) | Some(11) | Some(12) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // PopFromGeneralChannel: always empty.
                Some(20) => self.write_ipc_response(tls, AM_NO_DATA_IN_CHANNEL, &[], &[], &[]),
                // GetPopFromGeneralChannelEvent: never signalled.
                Some(21) => {
                    let h = match self.general_channel_event {
                        Some(h) => h,
                        None => {
                            let h = self.alloc_event("am:general-channel", true);
                            self.general_channel_event = Some(h);
                            h
                        }
                    };
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                // GetHomeButtonWriterLockAccessor / GetWriterLockAccessorEx.
                Some(30) | Some(31) => {
                    self.reply_with_interface(tls, handle, "am:lock-accessor")?;
                    Ok(())
                }
                // IsSleepEnabled / IsRebootEnabled.
                Some(40) | Some(41) => self.write_ipc_response(tls, 0, &[], &[1u8], &[]),
                // IsForceTerminateApplicationDisabledForDebug -> bool.
                Some(110) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                // SetLastApplicationExitReason.
                Some(1000) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ILockAccessor.
            "am:lock-accessor" => match cmd_id {
                // TryLock(bool return_handle) -> (bool locked, event).
                Some(1) => {
                    let want_handle =
                        self.mem.read_u8(self.ipc_request_data(tls)).unwrap_or(0) != 0;
                    let h = self.am_lock_accessor_event();
                    if want_handle {
                        self.write_ipc_reply(tls, 0, &[h], &[], &[1u8], &[])
                    } else {
                        self.write_ipc_response(tls, 0, &[], &[1u8], &[])
                    }
                }
                // Unlock.
                Some(2) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetEvent -> the event that says the lock is free.
                Some(3) => {
                    let h = self.am_lock_accessor_event();
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                // IsLocked -> bool.
                Some(4) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IGlobalStateController. Sleep, shutdown and reboot (0-4) are not implemented.
            "am:global-state-controller" => match cmd_id {
                // IsAutoPowerDownRequested.
                Some(9) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                // Idle policy, CEC, HOME long press and display resolution settings.
                Some(10) | Some(11) | Some(12) | Some(13) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // ShouldSleepOnBoot.
                Some(14) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                // GetHdcpAuthenticationFailedEvent.
                Some(15) => {
                    let h = self.alloc_event("am:hdcp-failed", true);
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IProcessWindingController.
            "am:process-winding-controller" => match cmd_id {
                // GetLaunchReason: all zero is a normal start.
                Some(0) => self.write_ipc_response(tls, 0, &[], &[0u8; 4], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ILibraryAppletSelfAccessor.
            "am:library-applet-self-accessor" => match cmd_id {
                // GetLibraryAppletInfo: AllForeground.
                Some(11) => {
                    let mut info = [0u8; 8];
                    info[..4].copy_from_slice(&applet_id_for(self.program_id()).to_le_bytes());
                    self.write_ipc_response(tls, 0, &[], &info, &[])
                }
                // ShouldSetGpuTimeSliceManually.
                Some(150) => self.write_ipc_response(tls, 0, &[], &0u8.to_le_bytes(), &[]),
                // GetMainAppletIdentityInfo / GetCallerAppletIdentityInfo: the home menu.
                Some(12) | Some(14) => {
                    let info = home_menu_identity();
                    self.write_ipc_response(tls, 0, &[], &info, &[])
                }
                // CanUseApplicationCore.
                Some(13) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                // GetMainAppletApplicationDesiredLanguage.
                Some(60) => {
                    let code = self.system_settings().language_code;
                    self.write_ipc_response(tls, 0, &[], &code.to_le_bytes(), &[])
                }
                // GetCallerAppletIdentityInfoStack: the home menu as the single entry.
                Some(17) => {
                    let info = home_menu_identity();
                    let (addr, size) = self.ipc_output_buffer(tls, 0).unwrap_or((0, 0));
                    let room = if addr == 0 {
                        0
                    } else {
                        size as usize / info.len()
                    };
                    let count = room.min(1);
                    if count == 1 {
                        for (index, &byte) in info.iter().enumerate() {
                            self.mem.write_u8(addr.wrapping_add(index as u32), byte)?;
                        }
                    }
                    self.write_ipc_response(tls, 0, &[], &(count as i32).to_le_bytes(), &[])
                }
                // GetDesirableKeyboardLayout: the console's layout.
                Some(19) => {
                    let layout = self.system_settings().keyboard_layout;
                    self.write_ipc_response(tls, 0, &[], &layout.to_le_bytes(), &[])
                }
                // PushOutData(IStorage): kept for [`Cpu::library_applet_results`].
                Some(1) => {
                    let data = self
                        .ipc_input_object_key(tls, handle, 0)
                        .and_then(|key| self.am_storages.get(&key))
                        .cloned();
                    let summary = match &data {
                        Some(data) => applet_result_summary(self.program_id(), data),
                        None => "a storage this session never handed out".to_owned(),
                    };
                    self.diagnostic(
                        Level::Info,
                        &format!(
                            "[am] the {} applet finished: {summary}",
                            applet_name(applet_id_for(self.program_id()))
                        ),
                    );
                    self.am_out_data.push(data.unwrap_or_default());
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // PushInteractiveOutData(IStorage): kept for the host.
                Some(3) => {
                    let data = self
                        .ipc_input_object_key(tls, handle, 0)
                        .and_then(|key| self.am_storages.get(&key))
                        .cloned()
                        .unwrap_or_default();
                    if self
                        .unimplemented_ipc
                        .insert(("am:applet-interactive-output".to_string(), cmd_id))
                    {
                        self.diagnostic(
                            Level::Warn,
                            &format!(
                                "[am] the applet is talking to its caller ({} bytes); nothing \
                                 here launched it, so only the host can answer",
                                data.len()
                            ),
                        );
                    }
                    self.am_interactive_out.push(data);
                    if self.am_interactive_out.len() > MAX_INTERACTIVE_MESSAGES {
                        self.am_interactive_out.remove(0);
                    }
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetPopInDataEvent / GetPopInteractiveInDataEvent.
                Some(5) | Some(6) => {
                    let queue = if cmd_id == Some(5) {
                        AppletQueue::InData
                    } else {
                        AppletQueue::InteractiveInData
                    };
                    let event = self.applet_queue_event(queue);
                    self.refresh_applet_pop_events();
                    self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                }
                // ExitProcessAndReturn.
                Some(10) => {
                    self.halted = true;
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // An unnamed init-time setter taking 16 bytes.
                Some(160) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // PopInData -> IStorage.
                Some(0) => self.pop_applet_storage(tls, handle, AppletQueue::InData),
                // PopInteractiveInData -> IStorage.
                Some(2) => self.pop_applet_storage(tls, handle, AppletQueue::InteractiveInData),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ILibraryAppletCreator.
            "am:library-applet-creator" => match cmd_id {
                // CreateLibraryApplet / CreateLibraryAppletEx (adds a thread id).
                Some(0) | Some(3) => {
                    let at = self.ipc_request_data(tls);
                    let id = self.mem.read_u32(at)?;
                    let mode = self.mem.read_u32(at.wrapping_add(4))?;
                    self.diagnostic(
                        Level::Warn,
                        &format!(
                            "[am] CreateLibraryApplet: {} (mode {mode}) — nothing here runs it, \
                         so it will report itself cancelled",
                            applet_name(id)
                        ),
                    );
                    let key =
                        self.reply_with_interface(tls, handle, "am:library-applet-accessor")?;
                    self.am_applets.insert(key, LibraryApplet::new(id, mode));
                    Ok(())
                }
                // TerminateAllLibraryApplets / AreAnyLibraryAppletsLeft.
                Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                Some(2) => self.write_ipc_response(tls, 0, &[], &0u8.to_le_bytes(), &[]),
                // CreateStorage(s64 size) -> IStorage.
                Some(10) => {
                    const MAX_STORAGE: u64 = 64 * 1024 * 1024;
                    /// `KERNELRESULT(OutOfMemory)`.
                    const OUT_OF_MEMORY: u32 = 1 | (104 << 9);
                    let size = self.mem.read_u64(self.ipc_request_data(tls))?;
                    if size > MAX_STORAGE {
                        return self.write_ipc_response(tls, OUT_OF_MEMORY, &[], &[], &[]);
                    }
                    let key = self.reply_with_interface(tls, handle, "am:storage")?;
                    self.am_storages.insert(key, vec![0u8; size as usize]);
                    Ok(())
                }
                // CreateTransferMemoryStorage / CreateHandleStorage: no backing memory to read.
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ILibraryAppletAccessor: the applet finishes, cancelled, as soon as it starts.
            "am:library-applet-accessor" => {
                let key = self.ipc_object_key(tls, handle);
                match cmd_id {
                    // GetAppletStateChangedEvent.
                    Some(0) => {
                        let event = self.library_applet_event(key, STATE_CHANGED_EVENT);
                        if self.library_applet_finished(key) {
                            self.signal_event(event);
                        }
                        self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                    }
                    // IsCompleted.
                    Some(1) => {
                        let done = u8::from(self.library_applet_finished(key));
                        self.write_ipc_response(tls, 0, &[], &done.to_le_bytes(), &[])
                    }
                    // Start / RequestExit / Terminate.
                    Some(10) | Some(20) | Some(25) => {
                        if let Some(applet) = self.am_applets.get_mut(&key) {
                            applet.finish();
                        }
                        self.signal_library_applet_event(key, STATE_CHANGED_EVENT);
                        self.write_ipc_response(tls, 0, &[], &[], &[])
                    }
                    // GetResult: cancelled.
                    Some(30) => {
                        /// `am` description 22, `LibAppletExitReason_Canceled`.
                        const CANCELLED: u32 = 128 | (22 << 9);
                        self.write_ipc_response(tls, CANCELLED, &[], &[], &[])
                    }
                    // PushInData / PushExtraStorage / PushInteractiveInData: dropped.
                    Some(100) | Some(102) | Some(103) => {
                        self.write_ipc_response(tls, 0, &[], &[], &[])
                    }
                    // PopOutData / PopInteractiveOutData: nothing produced.
                    Some(101) | Some(104) => {
                        const NO_DATA: u32 = 128 | (3 << 9);
                        self.write_ipc_response(tls, NO_DATA, &[], &[], &[])
                    }
                    // GetPopOutDataEvent / GetPopInteractiveOutDataEvent: never signalled.
                    Some(105) => {
                        let event = self.library_applet_event(key, POP_OUT_DATA_EVENT);
                        self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                    }
                    Some(106) => {
                        let event = self.library_applet_event(key, POP_INTERACTIVE_OUT_DATA_EVENT);
                        self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                    }
                    // NeedsToExitProcess.
                    Some(110) => self.write_ipc_response(tls, 0, &[], &0u8.to_le_bytes(), &[]),
                    // GetLibraryAppletInfo.
                    Some(120) => {
                        let mut info = [0u8; 8];
                        if let Some(applet) = self.am_applets.get(&key) {
                            info[..4].copy_from_slice(&applet.id.to_le_bytes());
                            info[4..].copy_from_slice(&applet.mode.to_le_bytes());
                        }
                        self.write_ipc_response(tls, 0, &[], &info, &[])
                    }
                    // RequestForAppletToGetForeground.
                    Some(150) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                    // GetIndirectLayerConsumerHandle.
                    _ => self.unimplemented_command(tls, &iface, cmd_id),
                }
            }
            // `am`'s IStorage, distinct from `fsp-srv`'s.
            "am:storage" => match cmd_id {
                // Open -> IStorageAccessor.
                Some(0) => {
                    let storage = self.ipc_object_key(tls, handle);
                    let accessor = self.reply_with_interface(tls, handle, "am:storage-accessor")?;
                    self.am_storage_of.insert(accessor, storage);
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "am:storage-accessor" => {
                let storage = self
                    .am_storage_of
                    .get(&self.ipc_object_key(tls, handle))
                    .copied()
                    .unwrap_or(0);
                match cmd_id {
                    // GetSize -> s64.
                    Some(0) => {
                        let size = self.am_storages.get(&storage).map_or(0, |d| d.len()) as u64;
                        self.write_ipc_response(tls, 0, &[], &size.to_le_bytes(), &[])
                    }
                    // Write(s64 offset, buffer<in>) / Read(s64 offset, buffer<out>).
                    Some(10) => {
                        let offset = self.mem.read_u64(self.ipc_request_data(tls))? as usize;
                        let Some((addr, len)) = self.ipc_input_buffer(tls, 0) else {
                            return self.write_ipc_response(tls, 0, &[], &[], &[]);
                        };
                        let mut bytes = Vec::with_capacity(len as usize);
                        for i in 0..len {
                            bytes.push(self.mem.read_u8(addr.wrapping_add(i))?);
                        }
                        let data = self.am_storages.entry(storage).or_default();
                        if data.len() < offset + bytes.len() {
                            data.resize(offset + bytes.len(), 0);
                        }
                        data[offset..offset + bytes.len()].copy_from_slice(&bytes);
                        self.write_ipc_response(tls, 0, &[], &[], &[])
                    }
                    Some(11) => {
                        let offset = self.mem.read_u64(self.ipc_request_data(tls))? as usize;
                        let data = self.am_storages.get(&storage).cloned().unwrap_or_default();
                        if let Some((addr, len)) = self.ipc_output_buffer(tls, 0) {
                            let end = data.len().min(offset.saturating_add(len as usize));
                            let chunk = if offset < end {
                                &data[offset..end]
                            } else {
                                &[][..]
                            };
                            for (i, &b) in chunk.iter().enumerate() {
                                self.mem.write_u8(addr.wrapping_add(i as u32), b)?;
                            }
                        }
                        self.write_ipc_response(tls, 0, &[], &[], &[])
                    }
                    _ => self.unimplemented_command(tls, &iface, cmd_id),
                }
            }
            // IDisplayController, IDebugFunctions, and unnamed sessions.
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }
}
