//! Online services (`friend`, `news`, `bcat`, `olsc`, `ovln`, `ldn`, `lp2p`),
//! answering as a console that has never been online. Their events never signal.

use super::Cpu;
use crate::Result;

impl Cpu {
    /// `ldn:m` and its `IMonitorService`: local wireless, idle with no network.
    pub(super) fn ldn_monitor_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        /// `nn::ldn::State::None`.
        const LDN_STATE_NONE: u32 = 0;
        /// `SecurityParameter` and `NetworkConfig`, both returned inline.
        const SECURITY_PARAMETER_SIZE: usize = 0x20;
        const NETWORK_CONFIG_SIZE: usize = 0x20;
        if self.ipc_answer_control(tls, handle, "ldn:m", cmd_id)? {
            return Ok(());
        }
        let iface = self.ipc_interface(tls, handle, "ldn:m");
        match iface.as_str() {
            "ldn:monitor" => match cmd_id {
                // GetState -> nn::ldn::State.
                Some(0) => self.write_ipc_response(tls, 0, &[], &LDN_STATE_NONE.to_le_bytes(), &[]),
                // GetNetworkInfo: a zeroed 0x480-byte NetworkInfo.
                Some(1) => {
                    self.zero_output_buffer(tls, 0);
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetIpv4Address -> address and subnet mask.
                Some(2) => self.write_ipc_response(tls, 0, &[], &[0u8; 8], &[]),
                // GetDisconnectReason -> s16.
                Some(3) => self.write_ipc_response(tls, 0, &[], &0i16.to_le_bytes(), &[]),
                // GetSecurityParameter / GetNetworkConfig.
                Some(4) => {
                    self.write_ipc_response(tls, 0, &[], &[0u8; SECURITY_PARAMETER_SIZE], &[])
                }
                Some(5) => self.write_ipc_response(tls, 0, &[], &[0u8; NETWORK_CONFIG_SIZE], &[]),
                // Initialize / Finalize; official software aborts if either fails.
                Some(100) | Some(101) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IMonitorServiceCreator: CreateMonitorService.
            _ => match cmd_id {
                Some(0) => {
                    self.reply_with_interface(tls, handle, "ldn:monitor")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
        }
    }

    /// `lp2p:m` and its `ISfMonitorService`: no group, no link.
    pub(super) fn lp2p_monitor_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        if self.ipc_answer_control(tls, handle, "lp2p:m", cmd_id)? {
            return Ok(());
        }
        let iface = self.ipc_interface(tls, handle, "lp2p:m");
        match iface.as_str() {
            "lp2p:monitor" => match cmd_id {
                // Initialize.
                Some(0) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetGroupInfo: an empty 0x200-byte GroupInfo rather than the real refusal.
                Some(288) => {
                    self.zero_output_buffer(tls, 0);
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetLinkLevel -> u32.
                Some(320) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ISfMonitorServiceCreator: CreateMonitorService.
            _ => match cmd_id {
                Some(0) => {
                    self.reply_with_interface(tls, handle, "lp2p:monitor")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
        }
    }

    /// `ovln:snd` / `ovln:rcv`: the overlay message queue, dropping sends and always empty.
    pub(super) fn ovln_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        let root = if self.service_name(handle) == Some("ovln:rcv") {
            "ovln:rcv"
        } else {
            "ovln:snd"
        };
        if self.ipc_answer_control(tls, handle, root, cmd_id)? {
            return Ok(());
        }
        let iface = self.ipc_interface(tls, handle, root);
        match iface.as_str() {
            "ovln:sender" => match cmd_id {
                // Send(RawMessage, SendOption).
                Some(0) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetUnreceivedMessageCount -> u32.
                Some(1) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "ovln:receiver" => match cmd_id {
                // AddSource / RemoveSource(SourceName).
                Some(0) | Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetReceiveEventHandle.
                Some(2) => {
                    let event = self.kept_event("ovln:receive", handle);
                    self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                }
                // Receive / ReceiveWithTick: only sent after the event fires.
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // OpenSender(SourceName, QueueAttribute) / OpenReceiver, both command 0.
            _ => match cmd_id {
                Some(0) => {
                    let name = if root == "ovln:rcv" {
                        "ovln:receiver"
                    } else {
                        "ovln:sender"
                    };
                    self.reply_with_interface(tls, handle, name)?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
        }
    }

    /// `olsc:s`: cloud save backup, with nothing backed up.
    pub(super) fn olsc_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        if self.ipc_answer_control(tls, handle, "olsc:s", cmd_id)? {
            return Ok(());
        }
        let iface = self.ipc_interface(tls, handle, "olsc:s");
        match iface.as_str() {
            "olsc:transfer-task-list" => match cmd_id {
                // GetTransferTaskCount* and ListTransferTaskInfo*: zero.
                Some(0) | Some(2) | Some(16) | Some(18) => {
                    self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[])
                }
                // Delete / RaiseTransferTaskPriority / SuspendTransferTask, both forms.
                Some(3) | Some(4) | Some(10) | Some(19) | Some(20) | Some(23) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // Get*EventNativeHandleHolder: two distinct holders, two events.
                Some(5) => {
                    self.reply_with_interface(tls, handle, "olsc:transfer-end-holder")?;
                    Ok(())
                }
                Some(9) => {
                    self.reply_with_interface(tls, handle, "olsc:transfer-start-holder")?;
                    Ok(())
                }
                // StopNextTransferTaskExecution -> IStopperObject.
                Some(8) => {
                    self.reply_with_interface(tls, handle, "olsc:stopper")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // INativeHandleHolder: GetNativeHandle, the holder picks the event.
            "olsc:transfer-end-holder" | "olsc:transfer-start-holder" | "olsc:error-holder" => {
                match cmd_id {
                    Some(0) => {
                        let purpose = match iface.as_str() {
                            "olsc:transfer-end-holder" => "olsc:transfer-end",
                            "olsc:transfer-start-holder" => "olsc:transfer-start",
                            _ => "olsc:transfer-error",
                        };
                        let event = self.kept_event(purpose, self.ipc_object_key(tls, handle));
                        self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                    }
                    _ => self.unimplemented_command(tls, &iface, cmd_id),
                }
            }
            "olsc:remote-storage" => match cmd_id {
                // GetCount / ListDataInfo.
                Some(3) | Some(17) => {
                    self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[])
                }
                // ClearDataInfoCache / DeleteDataInfoCache.
                Some(6) | Some(9) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetDataInfoCacheUpdateNativeHandleHolder.
                Some(19) => {
                    self.reply_with_interface(tls, handle, "olsc:error-holder")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "olsc:daemon" => match cmd_id {
                // GetApplicationAutoTransferSetting / GetGlobalAutoUpload/DownloadSetting.
                Some(0) | Some(2) | Some(5) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                // The matching setters, plus RunTransferTaskAutonomyRegistration.
                Some(1) | Some(3) | Some(4) | Some(6) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // StopAutonomyTaskExecution -> an IStopperObject.
                Some(11) => {
                    self.reply_with_interface(tls, handle, "olsc:stopper")?;
                    Ok(())
                }
                // GetAutonomyTaskStatus -> u32. Nothing is running.
                Some(12) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IOlscServiceForSystemService: the session itself before 17.0.0, else via the getter.
            _ => match cmd_id {
                // GetTransferTaskListController / GetRemoteStorageController /
                // GetDaemonController.
                Some(0) => {
                    self.reply_with_interface(tls, handle, "olsc:transfer-task-list")?;
                    Ok(())
                }
                Some(1) => {
                    self.reply_with_interface(tls, handle, "olsc:remote-storage")?;
                    Ok(())
                }
                Some(2) => {
                    self.reply_with_interface(tls, handle, "olsc:daemon")?;
                    Ok(())
                }
                // Delete*Property / InvalidateMountCache and the 900-block deletes.
                Some(10..=13) | Some(900) | Some(902..=908) | Some(910..=912) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // ListTransferTaskErrorInfo / GetTransferTaskErrorInfoCount.
                Some(100) | Some(101) => {
                    self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[])
                }
                // RemoveTransferTaskErrorInfo.
                Some(102) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetTransferTaskErrorInfoUpdateNativeHandleHolder.
                Some(104) => {
                    self.reply_with_interface(tls, handle, "olsc:error-holder")?;
                    Ok(())
                }
                // GetDataTransferPolicy(u64 application_id) -> two u8s.
                Some(200) => self.write_ipc_response(tls, 0, &[], &[0u8, 0u8], &[]),
                Some(201) | Some(204) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetUserSaveDataProperty(Uid, u64) and its setter.
                Some(300) => self.write_ipc_response(tls, 0, &[], &[0u8; 0x10], &[]),
                Some(301) | Some(400) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetOlscServiceForSystemService (17.0.0+).
                Some(10000) => {
                    self.reply_with_interface(tls, handle, "olsc:system-service")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
        }
    }

    /// `friend:u` and aliases: no friends, requests, blocks, or presence.
    pub(super) fn friend_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        /// Module 121, description 15: notification queue empty.
        const NO_NOTIFICATIONS: u32 = 121 | (15 << 9);
        if self.ipc_answer_control(tls, handle, "friend:u", cmd_id)? {
            return Ok(());
        }
        let iface = self.ipc_interface(tls, handle, "friend:u");
        match iface.as_str() {
            "friend:service" => match cmd_id {
                // GetCompletionEvent.
                Some(0) => {
                    let event = self.kept_event("friend:completion", handle);
                    self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                }
                // Cancel.
                Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // List commands: report zero written and leave the buffer alone.
                Some(10100) | Some(10101) | Some(10400) | Some(10500) | Some(10501)
                | Some(20105) | Some(20108) | Some(20201) | Some(20202) | Some(20300)
                | Some(20400) | Some(20402) | Some(20500) | Some(20502) | Some(20700)
                | Some(20702) | Some(22000) | Some(22002) => {
                    self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[])
                }
                // Count commands.
                Some(20100) | Some(20101) | Some(20200) | Some(22010) => {
                    self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[])
                }
                // Check/EnsureFriendListAvailable and the blocked-user pair.
                Some(10120) | Some(10121) | Some(10420) | Some(10421) => {
                    self.write_ipc_response(tls, 0, &[], &[1u8], &[])
                }
                // Online play session, presence, sync and cache commands.
                Some(10600) | Some(10601) | Some(10610) | Some(20103) | Some(20104)
                | Some(20401) | Some(20801) | Some(20900) | Some(40100) | Some(40400)
                | Some(49900) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetPlayHistoryStatistics(Uid) -> 0x10 bytes.
                Some(20701) => self.write_ipc_response(tls, 0, &[], &[0u8; 0x10], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "friend:notification" => match cmd_id {
                // GetEvent.
                Some(0) => {
                    let event = self.kept_event("friend:notification", handle);
                    self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                }
                // Clear.
                Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // Pop: refused, since a zeroed notification would read as a real event.
                Some(2) => self.write_ipc_response(tls, NO_NOTIFICATIONS, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IDaemonSuspendSessionService has no commands.
            "friend:daemon-suspend-session" => self.unimplemented_command(tls, &iface, cmd_id),
            // IServiceCreator.
            _ => match cmd_id {
                // CreateFriendService.
                Some(0) => {
                    self.reply_with_interface(tls, handle, "friend:service")?;
                    Ok(())
                }
                // CreateNotificationService(Uid).
                Some(1) => {
                    self.reply_with_interface(tls, handle, "friend:notification")?;
                    Ok(())
                }
                // CreateDaemonSuspendSessionService.
                Some(2) => {
                    self.reply_with_interface(tls, handle, "friend:daemon-suspend-session")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
        }
    }

    /// `news:*` (five permission levels, not modelled) and its objects: an empty database.
    pub(super) fn news_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        let root = self.service_name(handle).unwrap_or("news:p");
        let root: &'static str = match root {
            "news:a" => "news:a",
            "news:c" => "news:c",
            "news:m" => "news:m",
            "news:v" => "news:v",
            _ => "news:p",
        };
        if self.ipc_answer_control(tls, handle, root, cmd_id)? {
            return Ok(());
        }
        let iface = self.ipc_interface(tls, handle, root);
        match iface.as_str() {
            // INewlyArrivedEventHolder / IOverwriteEventHolder: Get, each its own event.
            "news:arrival-event" | "news:overwrite-event" => match cmd_id {
                Some(0) => {
                    let purpose = if iface == "news:arrival-event" {
                        "news:arrival"
                    } else {
                        "news:overwrite"
                    };
                    let key = self.ipc_object_key(tls, handle);
                    let event = self.kept_event(purpose, key);
                    self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "news:database" => match cmd_id {
                // GetListV1 / GetList / Count / CountWithKey.
                Some(0) | Some(1) | Some(2) | Some(1000) => {
                    self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[])
                }
                // UpdateIntegerValue / UpdateIntegerValueWithAddition / UpdateStringValue.
                Some(3) | Some(4) | Some(5) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // INewsDataService: no article to open, so refused.
            "news:data" => self.unimplemented_command(tls, &iface, cmd_id),
            "news:service" => match cmd_id {
                // PostLocalNews(msgpack buffer): dropped.
                Some(10100) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // SetPassphrase(u64, buffer).
                Some(20100) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetSubscriptionStatus and its setters.
                Some(30100) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
                Some(40100) | Some(40101) | Some(40201) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetTopicList.
                Some(30101) => self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[]),
                // 30110 -> news savedata usage and total size.
                Some(30110) => self.write_ipc_response(tls, 0, &[], &[0u8; 0x10], &[]),
                // IsSystemUpdateRequired -> bool.
                Some(30200) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                // 30210 -> `news!db_version`.
                Some(30210) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
                // RequestImmediateReception / ClearStorage.
                Some(30300) | Some(40200) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // [1.0.0] forms of the getters below.
                Some(30900) => {
                    self.reply_with_interface(tls, handle, "news:arrival-event")?;
                    Ok(())
                }
                Some(30901) => {
                    self.reply_with_interface(tls, handle, "news:data")?;
                    Ok(())
                }
                Some(30902) => {
                    self.reply_with_interface(tls, handle, "news:database")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IServiceCreator.
            _ => match cmd_id {
                Some(0) => {
                    self.reply_with_interface(tls, handle, "news:service")?;
                    Ok(())
                }
                Some(1) => {
                    self.reply_with_interface(tls, handle, "news:arrival-event")?;
                    Ok(())
                }
                Some(2) => {
                    self.reply_with_interface(tls, handle, "news:data")?;
                    Ok(())
                }
                Some(3) => {
                    self.reply_with_interface(tls, handle, "news:database")?;
                    Ok(())
                }
                Some(4) => {
                    self.reply_with_interface(tls, handle, "news:overwrite-event")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
        }
    }

    /// `bcat:*`: delivery cache, always empty with nothing to sync.
    pub(super) fn bcat_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        let root = self.service_name(handle).unwrap_or("bcat:u");
        let root: &'static str = match root {
            "bcat:a" => "bcat:a",
            "bcat:m" => "bcat:m",
            "bcat:s" => "bcat:s",
            _ => "bcat:u",
        };
        if self.ipc_answer_control(tls, handle, root, cmd_id)? {
            return Ok(());
        }
        let iface = self.ipc_interface(tls, handle, root);
        match iface.as_str() {
            "bcat:service" => match cmd_id {
                // RequestSyncDeliveryCache variants -> a finished progress object.
                Some(10100) | Some(10101) | Some(20100) | Some(20101) => {
                    self.reply_with_interface(tls, handle, "bcat:progress")?;
                    Ok(())
                }
                // Delivery task queueing commands.
                Some(10200) | Some(20400) | Some(20401) | Some(20410) | Some(30100)
                | Some(30200) | Some(30201) | Some(30202) | Some(30203) | Some(30210)
                | Some(30300) | Some(90201) | Some(90202) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetDeliveryCacheStorageUpdateNotifier(u64) -> INotifierService.
                Some(20300) => {
                    self.reply_with_interface(tls, handle, "bcat:notifier")?;
                    Ok(())
                }
                // RequestSuspendDeliveryTask(u64) -> IDeliveryTaskSuspensionService.
                Some(20301) => {
                    self.reply_with_interface(tls, handle, "bcat:suspension")?;
                    Ok(())
                }
                // GetDeliveryTaskList / ...ForSystem / GetDeliveryList / GetPushNotificationLog.
                Some(90100) | Some(90101) | Some(90200) | Some(90300) => {
                    self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[])
                }
                // GetDeliveryCacheStorageUsage -> two u64s.
                Some(90301) => self.write_ipc_response(tls, 0, &[], &[0u8; 0x10], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "bcat:storage" => match cmd_id {
                // CreateFileService / CreateDirectoryService.
                Some(0) => {
                    self.reply_with_interface(tls, handle, "bcat:file")?;
                    Ok(())
                }
                Some(1) => {
                    self.reply_with_interface(tls, handle, "bcat:directory")?;
                    Ok(())
                }
                // EnumerateDeliveryCacheDirectory.
                Some(10) => self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // Progress, notifier and suspension objects hand out one event; GetImpl is refused.
            "bcat:progress" | "bcat:notifier" | "bcat:suspension" => match cmd_id {
                Some(0) => {
                    let purpose = match iface.as_str() {
                        "bcat:progress" => "bcat:progress",
                        "bcat:notifier" => "bcat:notifier",
                        _ => "bcat:suspension",
                    };
                    let key = self.ipc_object_key(tls, handle);
                    let event = self.kept_event(purpose, key);
                    self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // File and directory services: unreachable through the empty listing.
            "bcat:file" | "bcat:directory" => self.unimplemented_command(tls, &iface, cmd_id),
            // IServiceCreator.
            _ => match cmd_id {
                // CreateBcatService(u64 process_id).
                Some(0) => {
                    self.reply_with_interface(tls, handle, "bcat:service")?;
                    Ok(())
                }
                // CreateDeliveryCacheStorageService, by process or application id.
                Some(1) | Some(2) => {
                    self.reply_with_interface(tls, handle, "bcat:storage")?;
                    Ok(())
                }
                // CreateDeliveryCacheProgressService, both forms (removed after 2.3.0).
                Some(3) | Some(4) => {
                    self.reply_with_interface(tls, handle, "bcat:progress")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
        }
    }
}
