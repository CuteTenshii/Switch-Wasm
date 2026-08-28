//! The online services, and the empty console they all describe.
//!
//! `friend`, `news`, `bcat`, `olsc` (cloud saves), `ovln`, `ldn` and
//! `lp2p`. None of them can reach anything, and every one of them answers as
//! though the console simply has nothing yet: no friends, no news, no
//! downloaded content, no local network, no paired console.
//!
//! **The answer is an empty console, not a broken one.** Every state described
//! here is one a real console reaches, so callers already have a path for it.
//! A *failure* puts them on the path built for hardware that broke. None of
//! the events handed out here ever signal.

use super::Cpu;
use crate::Result;

impl Cpu {
    /// `ldn:m` — "nn::ldn::detail::IMonitorServiceCreator", and the
    /// `IMonitorService` it hands out: the read-only view of local wireless
    /// that the Home Menu polls to decide whether to draw the local-play
    /// icon.
    ///
    /// There is no local wireless here — no radio, and nothing in the browser
    /// that could carry an ad-hoc network — so the monitor reports the state
    /// a console with the radio idle reports: `None`, no network, no address.
    /// That is a state the caller already handles, which is why it is the
    /// right answer rather than a failure.
    ///
    /// `Initialize` is the command that made this visible: official software
    /// **aborts** if it fails, and it is sent immediately after the object is
    /// created — so the fabricated object id the fallback returned was one
    /// command away from taking the caller down.
    pub(super) fn ldn_monitor_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        /// `nn::ldn::State::None` — the radio is up and doing nothing.
        const LDN_STATE_NONE: u32 = 0;
        /// `nn::ldn::SecurityParameter` and `nn::ldn::NetworkConfig`, both
        /// 0x20 bytes and both returned in the reply rather than a buffer.
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
                // GetNetworkInfo: a 0x480-byte NetworkInfo into an output
                // buffer. There is no network, and an untouched buffer would
                // be read as one.
                Some(1) => {
                    self.zero_output_buffer(tls, 0);
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetIpv4Address -> address and subnet mask. Both are
                // assigned when a network is joined, and none was.
                Some(2) => self.write_ipc_response(tls, 0, &[], &[0u8; 8], &[]),
                // GetDisconnectReason -> s16. The real service returns 0 here
                // unconditionally.
                Some(3) => self.write_ipc_response(tls, 0, &[], &0i16.to_le_bytes(), &[]),
                // GetSecurityParameter / GetNetworkConfig -> the credentials
                // and the shape of the network that was joined.
                Some(4) => {
                    self.write_ipc_response(tls, 0, &[], &[0u8; SECURITY_PARAMETER_SIZE], &[])
                }
                Some(5) => self.write_ipc_response(tls, 0, &[], &[0u8; NETWORK_CONFIG_SIZE], &[]),
                // Initialize / Finalize. Both return 0 on a real console
                // whatever the radio is doing, and official software aborts
                // if either does not.
                Some(100) | Some(101) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IMonitorServiceCreator: CreateMonitorService. The caller closes
            // the creator immediately afterwards and keeps the monitor.
            _ => match cmd_id {
                Some(0) => {
                    self.reply_with_interface(tls, handle, "ldn:monitor")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
        }
    }

    /// `lp2p:m` — "nn::lp2p::monitor::detail::ISfMonitorServiceCreator", and
    /// the `ISfMonitorService` it hands out.
    ///
    /// `lp2p` is the Wi-Fi-direct transport under local play that replaced
    /// `ldn`'s for newer titles; the monitor is the same read-only view, and
    /// gets the same answer for the same reason. The role is zero — this
    /// console is neither a group owner nor a member of one — so the group
    /// info is empty and the link level is nothing.
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
                // Initialize. The real service returns 0 and does nothing.
                Some(0) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetGroupInfo: a 0x200-byte GroupInfo into an output buffer.
                // With no group the real service refuses; answering with an
                // empty group instead keeps the caller on the path it takes
                // when there is nobody to play with, rather than on an error
                // path built for a radio that failed.
                Some(288) => {
                    self.zero_output_buffer(tls, 0);
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetLinkLevel -> u32. No link, no level.
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

    /// `ovln:snd` / `ovln:rcv` — "nn::ovln::ISenderService" and
    /// "nn::ovln::IReceiverService", the one-way message queue the overlay
    /// applet listens on.
    ///
    /// This is how a system module tells the overlay to draw something: a
    /// controller disconnecting, a screenshot being taken, a notification
    /// arriving. Each side opens an object first — `OpenSender` with the
    /// source name it is sending as, `OpenReceiver` for the overlay — and
    /// every message after that goes through *that* object, which is why the
    /// fabricated object id the fallback handed back was the end of the line
    /// rather than the start of one.
    ///
    /// The queue itself is not modelled: nothing here draws an overlay, so a
    /// message sent into it has nowhere to arrive. Sends are accepted and
    /// dropped, and the receiver's queue is permanently empty — which is
    /// exactly what a receiver sees on a console where nothing has happened.
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
                // GetUnreceivedMessageCount -> u32. Nothing was sent that
                // anything could still be waiting to receive.
                Some(1) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "ovln:receiver" => match cmd_id {
                // AddSource / RemoveSource(SourceName): which senders this
                // receiver listens to. None of them ever send.
                Some(0) | Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetReceiveEventHandle -> the event a message arriving
                // signals.
                Some(2) => {
                    let event = self.kept_event("ovln:receive", handle);
                    self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                }
                // Receive / ReceiveWithTick -> a RawMessage, and for the
                // second form the tick it arrived on. A caller only sends
                // these after the event above has fired, which it has not.
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ISenderService::OpenSender(SourceName, QueueAttribute) and
            // IReceiverService::OpenReceiver, both command 0.
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

    /// `olsc:s` — the save-data cloud backup service, as a system title
    /// reaches it.
    ///
    /// Five interfaces deep, and the Home Menu walks all of it on the way to
    /// its user page: `GetOlscServiceForSystemService` (17.0.0 moved the real
    /// interface behind this getter; before that the session *was* the
    /// interface, which is why both names dispatch here),
    /// `GetTransferTaskListController`, then the two
    /// `INativeHandleHolder`s that hold the events a transfer starting and
    /// finishing would signal, then `GetNativeHandle` on each to get the
    /// events themselves. Every step of that chain hands back an object, and
    /// the fallback's fabricated object id is not one — so the menu was
    /// waiting on handle 0 four objects before it ever got to a save.
    ///
    /// Nothing is backed up. There is no account linked to a Nintendo
    /// Account, no network under it, and no transfer queue — so the task
    /// list is empty, the error list is empty, and the events never fire.
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
                // GetTransferTaskCountForOcean / GetTransferTaskCount -> the
                // number of queued transfers, and ListTransferTaskInfo* ->
                // how many entries were written into the caller's buffer.
                // Nothing is queued, so every one of them is zero.
                Some(0) | Some(2) | Some(16) | Some(18) => {
                    self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[])
                }
                // DeleteTransferTask / RaiseTransferTaskPriority /
                // SuspendTransferTask, in both their Ocean and 10.1.0+ forms:
                // each names a task out of the empty list above.
                Some(3) | Some(4) | Some(10) | Some(19) | Some(20) | Some(23) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetTransferTaskEndEventNativeHandleHolder /
                // GetTransferTaskStartEventNativeHandleHolder. Two holders,
                // two different events — the caller waits on both and acts on
                // whichever fires, so they must not be the same object.
                Some(5) => {
                    self.reply_with_interface(tls, handle, "olsc:transfer-end-holder")?;
                    Ok(())
                }
                Some(9) => {
                    self.reply_with_interface(tls, handle, "olsc:transfer-start-holder")?;
                    Ok(())
                }
                // StopNextTransferTaskExecution -> an IStopperObject that
                // holds the queue stopped for as long as the caller keeps it.
                Some(8) => {
                    self.reply_with_interface(tls, handle, "olsc:stopper")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // INativeHandleHolder: GetNativeHandle, the one command it has.
            // Which holder this is decides which event comes back.
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
                // GetCount -> how many saves the cloud holds for this
                // console, and ListDataInfo -> how many it wrote out.
                Some(3) | Some(17) => {
                    self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[])
                }
                // ClearDataInfoCache / DeleteDataInfoCache: the cache of what
                // the cloud holds, which holds nothing.
                Some(6) | Some(9) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetDataInfoCacheUpdateNativeHandleHolder.
                Some(19) => {
                    self.reply_with_interface(tls, handle, "olsc:error-holder")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "olsc:daemon" => match cmd_id {
                // GetApplicationAutoTransferSetting(u64) -> bool, and
                // GetGlobalAutoUploadSetting / GetGlobalAutoDownloadSetting.
                // Automatic backup is off, and cannot be otherwise without an
                // account behind it.
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
            // IOlscServiceForSystemService, reached either as the session
            // itself (before 17.0.0) or through the getter below.
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
                // PrepareDeleteUserProperty / DeleteUserSaveDataProperty /
                // InvalidateMountCache / DeleteDeviceSaveDataProperty, and the
                // 900-block of "delete everything of this kind" commands.
                // There is nothing filed to delete, and each answers with a
                // bare Result.
                Some(10..=13) | Some(900) | Some(902..=908) | Some(910..=912) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // ListTransferTaskErrorInfo(u32 offset, buffer) -> how many
                // entries were written, and GetTransferTaskErrorInfoCount.
                Some(100) | Some(101) => {
                    self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[])
                }
                // RemoveTransferTaskErrorInfo, in both its forms.
                Some(102) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetTransferTaskErrorInfoUpdateNativeHandleHolder.
                Some(104) => {
                    self.reply_with_interface(tls, handle, "olsc:error-holder")?;
                    Ok(())
                }
                // GetDataTransferPolicy(u64 application_id) -> two u8s: what
                // this title is allowed to upload, and over what connection.
                // Neither, with no cloud behind it.
                Some(200) => self.write_ipc_response(tls, 0, &[], &[0u8, 0u8], &[]),
                // DeleteDataTransferPolicyCache / ClearDataTransferPolicyCache.
                Some(201) | Some(204) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetUserSaveDataProperty(Uid, u64) -> UserSaveDataProperty,
                // and its setter. The property is what the cloud knows about
                // one user's save for one title, which is nothing.
                Some(300) => self.write_ipc_response(tls, 0, &[], &[0u8; 0x10], &[]),
                Some(301) | Some(400) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetOlscServiceForSystemService: 17.0.0 turned the session
                // into a getter for the interface it used to be.
                Some(10000) => {
                    self.reply_with_interface(tls, handle, "olsc:system-service")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
        }
    }

    /// `friend:u` and its higher-privilege aliases —
    /// "nn::friends::detail::ipc::IServiceCreator", and the three interfaces
    /// it hands out.
    ///
    /// Every list here is empty and every count is zero, because the one
    /// account on this console ([`ACCOUNT_UID`]) is not linked to a Nintendo
    /// Account and there is no network behind it: no friends, no friend
    /// requests, no blocked users, no presence to publish. That is a real
    /// state of a real console — one that has never been online — and it is
    /// the state every other service here already describes.
    ///
    /// `CreateFriendService` is command 0 and the start of all of it, so the
    /// fallback's fabricated object id meant the *whole* interface was
    /// unreachable rather than merely unimplemented.
    pub(super) fn friend_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        /// `friends` (module 121) description 15: the notification queue is
        /// empty. `Pop` reports this rather than handing back a zeroed
        /// notification, which a caller would act on as a real event.
        const NO_NOTIFICATIONS: u32 = 121 | (15 << 9);
        if self.ipc_answer_control(tls, handle, "friend:u", cmd_id)? {
            return Ok(());
        }
        let iface = self.ipc_interface(tls, handle, "friend:u");
        match iface.as_str() {
            "friend:service" => match cmd_id {
                // GetCompletionEvent: the event every asynchronous command on
                // this interface signals when it finishes. Nothing here runs
                // asynchronously, so nothing finishes.
                Some(0) => {
                    let event = self.kept_event("friend:completion", handle);
                    self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                }
                // Cancel: there is no request in flight to cancel.
                Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // The list commands, in every generation: friend ids, friends,
                // blocked users, profiles, friend requests, friend candidates,
                // play history, received invitations. Each writes its entries
                // into a buffer and reports how many it wrote, and a caller
                // reads *that* number rather than the buffer's size — so zero
                // is the whole answer and the buffer is left alone.
                Some(10100) | Some(10101) | Some(10400) | Some(10500) | Some(10501)
                | Some(20105) | Some(20108) | Some(20201) | Some(20202) | Some(20300)
                | Some(20400) | Some(20402) | Some(20500) | Some(20502) | Some(20700)
                | Some(20702) | Some(22000) | Some(22002) => {
                    self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[])
                }
                // The count commands: friends, newly-added friends, received
                // friend requests, cached invitations.
                Some(20100) | Some(20101) | Some(20200) | Some(22010) => {
                    self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[])
                }
                // CheckFriendListAvailability / EnsureFriendListAvailable and
                // the blocked-user pair beside them: a caller asks whether the
                // cached list is usable before it reads it. An empty list is a
                // usable list.
                Some(10120) | Some(10121) | Some(10420) | Some(10421) => {
                    self.write_ipc_response(tls, 0, &[], &[1u8], &[])
                }
                // DeclareOpenOnlinePlaySession / DeclareCloseOnlinePlaySession
                // / UpdateUserPresence, and the sync and cache-clearing
                // commands. All of them publish or refresh something over a
                // network that is not there, and all answer with a Result.
                Some(10600) | Some(10601) | Some(10610) | Some(20103) | Some(20104)
                | Some(20401) | Some(20801) | Some(20900) | Some(40100) | Some(40400)
                | Some(49900) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetPlayHistoryStatistics(Uid) -> a 0x10-byte summary of how
                // much has been played with friends. Nothing has.
                Some(20701) => self.write_ipc_response(tls, 0, &[], &[0u8; 0x10], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "friend:notification" => match cmd_id {
                // GetEvent -> the event a notification arriving signals.
                Some(0) => {
                    let event = self.kept_event("friend:notification", handle);
                    self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                }
                // Clear: nothing queued to clear.
                Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // Pop -> the next notification. Refusing is the answer: a
                // caller only sends this after the event above has fired, and
                // a zeroed notification handed back instead would be read as a
                // friend coming online.
                Some(2) => self.write_ipc_response(tls, NO_NOTIFICATIONS, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IDaemonSuspendSessionService has no commands at all: holding
            // the object open is its entire purpose — it keeps the friend
            // daemon from running while the caller has it.
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

    /// `news:p` and its four siblings —
    /// "nn::news::detail::ipc::IServiceCreator", and the objects it hands
    /// out. This is the News channel: the articles that arrive over BCAT and
    /// surface as the Home Menu's News row.
    ///
    /// The five service names are the same interface at five permission
    /// levels (`news:a` may do everything, `news:p` may only post, and so
    /// on). Permissions are not modelled — there is no second process here to
    /// keep out — so all five dispatch to the same commands.
    ///
    /// Nothing has arrived and nothing can: there is no CDN behind this and
    /// no news savedata to have cached one. The database is empty rather than
    /// absent, which is what a console reports before its first sync.
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
            // INewlyArrivedEventHolder / IOverwriteEventHolder: one command
            // each, `Get`, and it returns the event. They are separate
            // objects holding separate events — one fires when an article
            // arrives, the other when an article already held is replaced —
            // and a caller waits on both.
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
                // GetListV1 / GetList: rows into a buffer, and how many were
                // written. Count / CountWithKey: how many rows match.
                Some(0) | Some(1) | Some(2) | Some(1000) => {
                    self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[])
                }
                // UpdateIntegerValue / UpdateIntegerValueWithAddition /
                // UpdateStringValue: each names a row of the empty table
                // above — marking an article read, counting a view.
                Some(3) | Some(4) | Some(5) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // INewsDataService reads an article's msgpack out of the news
            // savedata by name. There is no savedata and no article, so
            // `Open` has nothing to open — and a fabricated success there
            // would hand the caller an empty file to parse as an article.
            // Refusing puts the failure on the command that genuinely cannot
            // be answered.
            "news:data" => self.unimplemented_command(tls, &iface, cmd_id),
            "news:service" => match cmd_id {
                // PostLocalNews(msgpack buffer): a title posting an article
                // of its own. Accepted and dropped — there is no database to
                // put it in, and the arrival event nothing is waiting on is
                // what would announce it.
                Some(10100) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // SetPassphrase(u64, buffer): the key this title's BCAT
                // content is encrypted with.
                Some(20100) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetSubscriptionStatus -> u32, and SetSubscriptionStatus /
                // RequestAutoSubscription / ClearSubscriptionStatusAll beside
                // it. Nothing is subscribed to anything.
                Some(30100) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
                Some(40100) | Some(40101) | Some(40201) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetTopicList -> entries into a buffer, and how many.
                Some(30101) => self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[]),
                // 30110 -> the news savedata's usage and its total size, as
                // two u64s. The savedata is not mounted, so neither is used
                // nor available.
                Some(30110) => self.write_ipc_response(tls, 0, &[], &[0u8; 0x10], &[]),
                // IsSystemUpdateRequired -> bool: whether the news module
                // wants a firmware newer than this one. It does not.
                Some(30200) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                // 30210 -> the database version out of the `news!db_version`
                // system setting. An empty database is at version zero.
                Some(30210) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
                // RequestImmediateReception / ClearStorage: fetch now, and
                // throw away what was fetched. Nothing to do either way.
                Some(30300) | Some(40200) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // The [1.0.0] forms of the three getters below, from when
                // this interface *was* the service.
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

    /// `bcat:u` and its three siblings — "nn::bcat::ipc::IServiceCreator",
    /// and the delivery-cache objects it hands out.
    ///
    /// BCAT is the background download that brings a title its event data and
    /// the system its news. There is no network here, so every sync has
    /// nothing to fetch and every cache is empty — which is the state a title
    /// handles as "no new content", not as an error.
    ///
    /// The delivery cache is a filesystem in miniature (a storage, then a
    /// directory or a file within it), and each level is a separate object.
    /// The fallback answered `CreateBcatService` with an object id that is
    /// not an object, so nothing past the first command was reachable.
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
                // RequestSyncDeliveryCache, with and without an application
                // id or a directory name: each starts a download and hands
                // back the progress object that reports on it. The download
                // is over before it starts.
                Some(10100) | Some(10101) | Some(20100) | Some(20101) => {
                    self.reply_with_interface(tls, handle, "bcat:progress")?;
                    Ok(())
                }
                // CancelSyncDeliveryCacheRequest, the delivery-task
                // registration commands, SetPassphrase, and
                // RegisterSystemApplicationDeliveryTasks: all of them queue
                // or unqueue background work, and all answer with a Result.
                Some(10200) | Some(20400) | Some(20401) | Some(20410) | Some(30100)
                | Some(30200) | Some(30201) | Some(30202) | Some(30203) | Some(30210)
                | Some(30300) | Some(90201) | Some(90202) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetDeliveryCacheStorageUpdateNotifier(u64) -> an
                // INotifierService holding the event that fires when the
                // cache changes underneath the title.
                Some(20300) => {
                    self.reply_with_interface(tls, handle, "bcat:notifier")?;
                    Ok(())
                }
                // RequestSuspendDeliveryTask(u64) -> an
                // IDeliveryTaskSuspensionService: the task stays suspended
                // for as long as the caller holds the object.
                Some(20301) => {
                    self.reply_with_interface(tls, handle, "bcat:suspension")?;
                    Ok(())
                }
                // GetDeliveryTaskList / ...ForSystem / GetDeliveryList /
                // GetPushNotificationLog: entries into a buffer, and how many
                // were written.
                Some(90100) | Some(90101) | Some(90200) | Some(90300) => {
                    self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[])
                }
                // GetDeliveryCacheStorageUsage -> how much of the cache is in
                // use, as two u64s. None of it.
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
                // EnumerateDeliveryCacheDirectory -> directory names into a
                // buffer, and how many. The cache has no directories.
                Some(10) => self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IDeliveryCacheProgressService, INotifierService and
            // IDeliveryTaskSuspensionService each hand out one event.
            // `IDeliveryCacheProgressService::GetImpl` also reports the
            // finished state of the sync, which is a struct a caller reads
            // rather than an event it waits on — and one this has no honest
            // shape for, so it is refused rather than invented.
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
            // IDeliveryCacheFileService / IDeliveryCacheDirectoryService:
            // `Open` names an entry of the empty listing above, so a caller
            // can only reach these with a name the cache never reported.
            // Read and GetSize would then be answering about a file that was
            // never opened.
            "bcat:file" | "bcat:directory" => self.unimplemented_command(tls, &iface, cmd_id),
            // IServiceCreator.
            _ => match cmd_id {
                // CreateBcatService(u64 process_id).
                Some(0) => {
                    self.reply_with_interface(tls, handle, "bcat:service")?;
                    Ok(())
                }
                // CreateDeliveryCacheStorageService, by process id or by
                // application id.
                Some(1) | Some(2) => {
                    self.reply_with_interface(tls, handle, "bcat:storage")?;
                    Ok(())
                }
                // CreateDeliveryCacheProgressService, in the same two forms.
                // Both were removed after 2.3.0.
                Some(3) | Some(4) => {
                    self.reply_with_interface(tls, handle, "bcat:progress")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
        }
    }
}
