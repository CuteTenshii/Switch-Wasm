//! `erpt`: the error-report journal.
//!
//! Every process on a real console files context records here as it runs, and
//! the crash reporter turns them into reports. Both halves are real: a
//! submitted context is journalled, a report is filed from the journal, and
//! `erpt:r` reads back exactly what was filed — because a caller that files a
//! report and cannot then find it concludes the journal is broken.

use super::Cpu;
use crate::Result;

/// One category's worth of context `erpt` is holding, as the caller submitted
/// it.
///
/// The journal keeps at most one record per category: a module submitting
/// `ThermalInfo` every few seconds is *replacing* the record that is there,
/// not appending to a log. `fields` is the array data the entry's fields index
/// into, which is stored alongside because a field naming a string is useless
/// without it. Neither is interpreted — `erpt` collects context, it does not
/// read it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ErrorContext {
    category: u32,
    entry: Vec<u8>,
    fields: Vec<u8>,
}

/// One error report filed through `erpt:c`.
///
/// The body is the journal as it stood the moment the report was created,
/// which is what a real `erpt` writes out — as msgpack, where this keeps the
/// raw `ContextEntry` records, since nothing on either side of it here parses
/// one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ErrorReport {
    id: [u8; ERPT_ID_SIZE],
    report_type: u32,
    flags: u32,
    meta: Vec<u8>,
    body: Vec<u8>,
    timestamp: i64,
    attachments: Vec<[u8; ERPT_ID_SIZE]>,
}

/// One attachment, submitted ahead of the report that will claim it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ErrorReportAttachment {
    id: [u8; ERPT_ID_SIZE],
    owner: [u8; ERPT_ID_SIZE],
    flags: u32,
    name: String,
    data: Vec<u8>,
}

/// What one `IReport` or `IAttachment` object has open, and how far through it
/// the caller has read.
///
/// `Read` takes no offset — a caller drains a report by calling it until it
/// answers zero — so the cursor belongs to the object, not to the request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ErrorReportReader {
    id: [u8; ERPT_ID_SIZE],
    offset: usize,
}

/// `nn::erpt::ContextEntry`: a version, a field count, the category, twenty
/// 0x10-byte fields, and the array buffer they index.
const ERPT_CONTEXT_ENTRY_SIZE: usize = 0x160;

const ERPT_CONTEXT_CATEGORY: usize = 0x8;

const ERPT_FIELD_ENTRY_SIZE: usize = 0x10;

/// `nn::erpt::ReportId` and `AttachmentId`: twenty bytes holding a sixteen-byte
/// UUID. Only those sixteen are ever compared, which is why the tail stays
/// zero rather than being filled with more random.
const ERPT_ID_SIZE: usize = 0x14;

pub(super) const ERPT_UUID_SIZE: usize = 0x10;

/// `nn::erpt::ReportMetaData`, the opaque blob whoever files a report attaches
/// to it.
const ERPT_META_SIZE: usize = 0x20;

/// `nn::erpt::ReportInfo` and the `ReportList` of fifty that `GetReportList`
/// fills: a count, four bytes of padding, then the array.
const ERPT_REPORT_INFO_SIZE: usize = 0x70;

const ERPT_REPORT_COUNT_MAX: usize = 50;

/// `nn::erpt::AttachmentInfo` and the `AttachmentList` of five, laid out the
/// same way.
const ERPT_ATTACHMENT_INFO_SIZE: usize = 0x58;

const ERPT_ATTACHMENTS_PER_REPORT: usize = 5;

const ERPT_ATTACHMENT_NAME_MAX: usize = 0x20;

/// The caps a real `erpt` enforces on what a caller may hand over:
/// `AttachmentSizeMax`, `ArrayBufferSizeMax`, and the size `GetReportSizeMax`
/// reports. `ERPT_CONTEXT_ENTRIES_MAX` is this implementation's own — context
/// is submitted a category at a time, and the cap only stops a nonsense buffer
/// size asking for an unbounded read.
const ERPT_ATTACHMENT_SIZE_MAX: u32 = 512 * 1024;

const ERPT_ARRAY_BUFFER_MAX: u32 = 96 * 1024;

const ERPT_REPORT_SIZE_MAX: u32 = 0x3FF4F;

const ERPT_CONTEXT_ENTRIES_MAX: u32 = 64;

/// `nn::erpt::MultipleCategoryContextEntry`: a version and a count, then
/// parallel arrays of sixteen category ids, field counts and array-buffer
/// counts, and four fields per category behind them.
const ERPT_MULTI_CATEGORY_MAX: usize = 0x10;

const ERPT_MULTI_CATEGORIES: usize = 0x8;

/// `erpt` (module 147) description 8: nothing is filed under the id the caller
/// asked for. Reporting success instead would leave a caller reading a report
/// that does not exist.
const ERPT_NOT_FOUND: u32 = 147 | (8 << 9);

/// `nn::erpt::CategoryId`, so a filed report says what it is about rather than
/// listing numbers. A category this table predates prints as its number, which
/// is still the answer.
const ERPT_CATEGORIES: [&str; 157] = [
    "Test", "ErrorInfo", "ConnectionStatusInfo", "NetworkInfo", "NXMacAddressInfo",
    "StealthNetworkInfo", "LimitHighCapacityInfo", "NATTypeInfo", "WirelessAPMacAddressInfo",
    "GlobalIPAddressInfo", "EnableWirelessInterfaceInfo", "EnableWifiInfo",
    "EnableBluetoothInfo", "EnableNFCInfo", "NintendoZoneSSIDListVersionInfo",
    "LANAdapterMacAddressInfo", "ApplicationInfo", "OccurrenceInfo", "ProductModelInfo",
    "CurrentLanguageInfo", "UseNetworkTimeProtocolInfo", "TimeZoneInfo",
    "ControllerFirmwareInfo", "VideoOutputInfo", "NANDFreeSpaceInfo", "SDCardFreeSpaceInfo",
    "ScreenBrightnessInfo", "AudioFormatInfo", "MuteOnHeadsetUnpluggedInfo",
    "NumUserRegisteredInfo", "DataDeletionInfo", "ControllerVibrationInfo", "LockScreenInfo",
    "InternalBatteryLotNumberInfo", "LeftControllerSerialNumberInfo",
    "RightControllerSerialNumberInfo", "NotificationInfo", "TVInfo", "SleepInfo",
    "ConnectionInfo", "NetworkErrorInfo", "FileAccessPathInfo", "GameCardCIDInfo",
    "NANDCIDInfoDeprecated", "MicroSDCIDInfoDeprecated", "NANDSpeedModeInfo",
    "MicroSDSpeedModeInfo", "GameCardSpeedModeInfo", "UserAccountInternalIDInfo",
    "NetworkServiceAccountInternalIDInfo", "NintendoAccountInternalIDInfo", "USB3AvailableInfo",
    "CallStackInfo", "SystemStartupLogInfo", "RegionSettingInfo", "NintendoZoneConnectedInfo",
    "ForceSleepInfo", "ChargerInfo", "RadioStrengthInfo", "ErrorInfoAuto", "AccessPointInfo",
    "ErrorInfoDefaults", "SystemPowerStateInfo", "PerformanceInfo", "ThrottlingInfo",
    "GameCardErrorInfo", "EdidInfo", "ThermalInfo", "CradleFirmwareInfo",
    "RunningApplicationInfo", "RunningAppletInfo", "FocusedAppletHistoryInfo", "CompositorInfo",
    "BatteryChargeInfo", "NANDExtendedCsdDeprecated", "NANDPatrolInfo", "NANDErrorInfo",
    "NANDDriverLog", "SdCardSizeSpec", "SdCardErrorInfo", "", "FsProxyErrorInfo",
    "SystemAppletSceneInfo", "VideoInfo", "GpuErrorInfo", "PowerClockInfo", "AdspErrorInfo",
    "NvDispDeviceInfo", "NvDispDcWindowInfo", "NvDispDpModeInfo", "NvDispDpLinkSpec",
    "NvDispDpLinkStatus", "NvDispDpHdcpInfo", "NvDispDpAuxCecInfo", "NvDispDcInfo",
    "NvDispDsiInfo", "NvDispErrIDInfo", "SdCardMountInfo", "RetailInteractiveDisplayInfo",
    "CompositorStateInfo", "CompositorLayerInfo", "CompositorDisplayInfo", "CompositorHWCInfo",
    "MonitorCapability", "ErrorReportSharePermissionInfo", "MultimediaInfo",
    "ConnectedControllerInfo", "FsMemoryInfo", "UserClockContextInfo",
    "NetworkClockContextInfo", "AcpGeneralSettingsInfo", "AcpPlayLogSettingsInfo",
    "AcpAocSettingsInfo", "AcpBcatSettingsInfo", "AcpStorageSettingsInfo",
    "AcpRatingSettingsInfo", "MonitorSettings", "RebootlessSystemUpdateVersionInfo",
    "NifmConnectionTestInfo", "PcieLoggedStateInfo", "NetworkSecurityCertificateInfo",
    "AcpNeighborDetectionInfo", "GpuCrashInfo", "UsbStateInfo", "NvHostErrInfo",
    "RunningUlaInfo", "InternalPanelInfo", "ResourceLimitInfo",
    "ResourceLimitPeakInfoDeprecated", "TouchScreenInfo", "AcpUserAccountSettingsInfo",
    "AudioDeviceInfo", "AbnormalWakeInfo", "ServiceProfileInfo", "BluetoothAudioInfoDeprecated",
    "BluetoothPairingCountInfo", "FsProxyErrorInfo2", "BuiltInWirelessOUIInfo",
    "WirelessAPOUIInfo", "EthernetAdapterOUIInfo", "NANDTypeInfoDeprecated", "MicroSDTypeInfo",
    "AttachmentFileInfo", "WlanInfo", "HalfAwakeStateInfo", "PctlSettingInfo",
    "GameCardLogInfo", "WlanIoctlErrorInfo", "SdCardActivationInfo",
    "GameCardDetailedErrorInfo", "NetworkInfo2", "SystemSettingInfo", "MigrationStateInfo",
    "WinVdInfo", "PscTransitionStateInfo", "FsProxyErrorInfo3", "BluetoothErrorInfo",
];

impl Cpu {
    /// `erpt:c` — "nn::erpt::sf::IContext", the error-report collector.
    ///
    /// A console keeps a running journal of *context*: one record per category
    /// — `ErrorInfo`, `ApplicationInfo`, `ThermalInfo`, `GpuCrashInfo` — that
    /// whichever module owns it keeps current by resubmitting. When something
    /// goes wrong, whoever noticed calls one of the `CreateReport` commands and
    /// the journal as it stands at that instant is written out as a report, for
    /// the error-report transfer to upload later.
    ///
    /// That makes this the second account a guest ever gives of why it is
    /// unhappy — [`Cpu::fatal_request`] is the first — and a far more detailed
    /// one, which is why a report being filed is a diagnostic. The generic
    /// fallback answered `SubmitContext` with a fabricated object id and threw
    /// every one of those records away.
    ///
    /// Nothing is uploaded and nothing reaches the SYSTEM partition: the
    /// journal lives as long as the session does, which is what
    /// [`Cpu::erpt_manager_request`]'s statistics describe.
    pub(super) fn erpt_context_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        if self.ipc_answer_control(tls, handle, "erpt:c", cmd_id)? {
            return Ok(());
        }
        match cmd_id {
            // SubmitContext(ContextEntry[], FieldList).
            Some(0) => {
                self.erpt_submit_context(tls, 0, 1);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // CreateReportV0(ReportType, ContextEntry[], FieldList,
            // ReportMetaData), and the 11.0.0 and 17.0.0 revisions of it that
            // add option words after the type. All three file the journal
            // together with the context the caller brought along.
            Some(1) | Some(11) | Some(12) => {
                let categories = self.erpt_submit_context(tls, 0, 1);
                let report_type = self.ipc_arg_u32(tls, 0);
                let meta = match self.ipc_input_buffer(tls, 2) {
                    Some((addr, size)) => self.read_bytes(addr, size.min(ERPT_META_SIZE as u32)),
                    None => Vec::new(),
                };
                self.erpt_create_report(report_type, meta, &categories, &[]);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // CreateReportWithAttachments(ReportType, ContextEntry[],
            // FieldList, AttachmentId[]): the same report, claiming the
            // attachments `SubmitAttachment` filed ahead of it.
            Some(10) => {
                let categories = self.erpt_submit_context(tls, 0, 1);
                let report_type = self.ipc_arg_u32(tls, 0);
                let ids = match self.ipc_input_buffer(tls, 2) {
                    Some((addr, size)) => {
                        let max = (ERPT_ATTACHMENTS_PER_REPORT * ERPT_ID_SIZE) as u32;
                        self.read_bytes(addr, size.min(max))
                    }
                    None => Vec::new(),
                };
                let ids: Vec<[u8; ERPT_ID_SIZE]> = ids
                    .chunks_exact(ERPT_ID_SIZE)
                    .map(|id| {
                        let mut out = [0u8; ERPT_ID_SIZE];
                        out.copy_from_slice(id);
                        out
                    })
                    .collect();
                self.erpt_create_report(report_type, Vec::new(), &categories, &ids);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // SubmitAttachment(name, data) -> the AttachmentId the report that
            // will own it names. The id is the server's to assign, so a bare
            // success left the caller holding whatever its out-parameter was
            // initialized to and naming that in `CreateReportWithAttachments`.
            Some(9) => {
                let name = match self.ipc_input_buffer(tls, 0) {
                    Some((addr, size)) => {
                        self.read_string(addr, size.min(ERPT_ATTACHMENT_NAME_MAX as u32))
                    }
                    None => String::new(),
                };
                let data = match self.ipc_input_buffer(tls, 1) {
                    Some((addr, size)) => self.read_bytes(addr, size.min(ERPT_ATTACHMENT_SIZE_MAX)),
                    None => Vec::new(),
                };
                let id = self.erpt_new_id();
                self.erpt_attachments.push(ErrorReportAttachment {
                    id,
                    owner: [0u8; ERPT_ID_SIZE],
                    flags: 0,
                    name,
                    data,
                });
                self.write_ipc_response(tls, 0, &[], &id, &[])
            }
            // SubmitMultipleCategoryContext(MultipleCategoryContextEntry,
            // FieldList): several categories in one call, their ids in an
            // array of sixteen at the head of the struct. It arrives as a
            // buffer rather than in the payload, being far too large for one.
            Some(6) => {
                self.erpt_submit_multiple_context(tls);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // SetInitialLaunchSettingsCompletionTime(SteadyClockTimePoint) and
            // ClearInitialLaunchSettingsCompletionTime: when the console
            // finished its first-boot setup, which is context for a report
            // rather than something anything here reads back.
            Some(2) | Some(3) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // Four commands that were renumbered at 21.0.0: through 20.5.0
            // these were UpdatePowerOnTime, UpdateAwakeTime,
            // UpdateApplicationLaunchTime and ClearApplicationLaunchTime, and
            // afterwards CreateReportWithAdditionalContext,
            // SubmitMultipleContext and the running-application registration.
            // Every one of them answers with a bare `Result` and nothing else,
            // so the reply is the same either way; what differs is only
            // whether a report is filed, and filing one nobody asked for would
            // put an empty report in the journal.
            Some(4) | Some(5) | Some(7) | Some(8) => {
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // RegisterRunningApplet / UnregisterRunningApplet /
            // UpdateAppletSuspendedDuration, and the forced-shutdown detector
            // a clean shutdown invalidates on its way out. Nothing here loses
            // power without warning.
            Some(20) | Some(21) | Some(22) | Some(30) => {
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            _ => self.unimplemented_command(tls, "erpt:c", cmd_id),
        }
    }

    /// Fold the `ContextEntry` array in input buffer `entries`, and the field
    /// data in input buffer `fields` that they index, into the journal.
    ///
    /// Returns the categories the submission carried, in the order they
    /// arrived, for the report a `CreateReport` in the same request will file.
    fn erpt_submit_context(&mut self, tls: u32, entries: u32, fields: u32) -> Vec<u32> {
        let array = match self.ipc_input_buffer(tls, entries) {
            Some((addr, size)) => {
                let max = ERPT_CONTEXT_ENTRIES_MAX * ERPT_CONTEXT_ENTRY_SIZE as u32;
                self.read_bytes(addr, size.min(max))
            }
            None => return Vec::new(),
        };
        let data = match self.ipc_input_buffer(tls, fields) {
            Some((addr, size)) => self.read_bytes(addr, size.min(ERPT_ARRAY_BUFFER_MAX)),
            None => Vec::new(),
        };
        let mut categories = Vec::new();
        for entry in array.chunks_exact(ERPT_CONTEXT_ENTRY_SIZE) {
            let category = u32::from_le_bytes(
                entry[ERPT_CONTEXT_CATEGORY..ERPT_CONTEXT_CATEGORY + 4].try_into().unwrap(),
            );
            categories.push(category);
            self.erpt_record_context(ErrorContext {
                category,
                entry: entry.to_vec(),
                fields: data.clone(),
            });
        }
        if crate::env_flag!("TRACE_ERPT") {
            for &category in &categories {
                eprintln!("[erpt] context {}", Self::erpt_category_name(category));
            }
        }
        categories
    }

    /// The same, for `SubmitMultipleCategoryContext`'s single struct: a
    /// version, a category count, then parallel arrays of sixteen category
    /// ids, field counts and array-buffer counts, and sixty-four fields four
    /// to a category.
    ///
    /// Each category is unpacked into a `ContextEntry` of its own, so the
    /// journal holds one shape of record whichever command filled it and a
    /// report is a plain array of them. Storing the struct itself under every
    /// category it names would put sixteen copies of it in the next report.
    fn erpt_submit_multiple_context(&mut self, tls: u32) {
        let Some((addr, size)) = self.ipc_input_buffer(tls, 0) else {
            return;
        };
        let fields_at = ERPT_MULTI_CATEGORIES + 3 * 4 * ERPT_MULTI_CATEGORY_MAX;
        let struct_size = (fields_at + 4 * ERPT_MULTI_CATEGORY_MAX * ERPT_FIELD_ENTRY_SIZE) as u32;
        let multiple = self.read_bytes(addr, size.min(struct_size));
        if multiple.len() < struct_size as usize {
            return;
        }
        let data = match self.ipc_input_buffer(tls, 1) {
            Some((addr, size)) => self.read_bytes(addr, size.min(ERPT_ARRAY_BUFFER_MAX)),
            None => Vec::new(),
        };
        let version = &multiple[..4];
        let count = u32::from_le_bytes(multiple[4..8].try_into().unwrap()) as usize;
        let per_category = 4 * ERPT_FIELD_ENTRY_SIZE;
        for index in 0..count.min(ERPT_MULTI_CATEGORY_MAX) {
            let category_at = ERPT_MULTI_CATEGORIES + index * 4;
            let count_at = ERPT_MULTI_CATEGORIES + 4 * ERPT_MULTI_CATEGORY_MAX + index * 4;
            let fields = fields_at + index * per_category;
            let id = &multiple[category_at..category_at + 4];
            let mut entry = vec![0u8; ERPT_CONTEXT_ENTRY_SIZE];
            entry[..4].copy_from_slice(version);
            entry[4..8].copy_from_slice(&multiple[count_at..count_at + 4]);
            entry[8..12].copy_from_slice(id);
            entry[0xC..0xC + per_category]
                .copy_from_slice(&multiple[fields..fields + per_category]);
            let category = u32::from_le_bytes(id.try_into().unwrap());
            self.erpt_record_context(ErrorContext { category, entry, fields: data.clone() });
            if crate::env_flag!("TRACE_ERPT") {
                eprintln!("[erpt] context {}", Self::erpt_category_name(category));
            }
        }
    }

    /// File one context record, replacing the one the journal already holds
    /// for that category.
    fn erpt_record_context(&mut self, record: ErrorContext) {
        match self.erpt_contexts.iter_mut().find(|held| held.category == record.category) {
            Some(held) => *held = record,
            None => self.erpt_contexts.push(record),
        }
    }

    /// Write the journal out as a report, and say so.
    ///
    /// `categories` are the ones the caller submitted along with the report,
    /// which is what the report is *about* — the rest of the journal is the
    /// state the console happened to be in. The oldest report goes when the
    /// journal is full, the way a console's does.
    fn erpt_create_report(
        &mut self,
        report_type: u32,
        meta: Vec<u8>,
        categories: &[u32],
        attachments: &[[u8; ERPT_ID_SIZE]],
    ) {
        let id = self.erpt_new_id();
        let mut body = Vec::new();
        for context in &self.erpt_contexts {
            body.extend_from_slice(&context.entry);
            body.extend_from_slice(&context.fields);
        }
        for attachment in &mut self.erpt_attachments {
            let named =
                attachments.iter().any(|wanted| Self::erpt_same_id(wanted, &attachment.id));
            if named {
                attachment.owner = id;
            }
        }
        let about = match categories.is_empty() {
            true => "no context of its own".to_owned(),
            false => categories
                .iter()
                .map(|&category| Self::erpt_category_name(category))
                .collect::<Vec<_>>()
                .join(", "),
        };
        self.diagnostic(&format!(
            "[erpt] {} report {} filed: {about} ({} categories journalled, {} bytes)",
            Self::erpt_report_type_name(report_type),
            Self::erpt_id_text(&id),
            self.erpt_contexts.len(),
            body.len()
        ));
        if self.erpt_reports.len() >= ERPT_REPORT_COUNT_MAX {
            self.erpt_reports.remove(0);
        }
        let timestamp = self.unix_time();
        self.erpt_reports.push(ErrorReport {
            id,
            report_type,
            flags: 0,
            meta,
            body,
            timestamp,
            attachments: attachments.to_vec(),
        });
        self.erpt_signal_report_created();
    }

    /// Fire the event `IManager::GetEvent` handed out. A report really was
    /// just filed, so unlike every other event these services hand out, this
    /// one describes something that happens.
    fn erpt_signal_report_created(&mut self) {
        let events: Vec<u64> = self
            .service_events
            .iter()
            .filter(|(&(purpose, _), _)| purpose == "erpt:report-created")
            .map(|(_, &event)| event)
            .collect();
        for event in events {
            self.signal_event(event);
        }
    }

    /// A fresh `ReportId` or `AttachmentId`: a version-4 UUID in the first
    /// sixteen bytes of twenty. The last four stay zero because that is what
    /// they are on a console — the id is a `util::Uuid` in a twenty-byte
    /// field, and every comparison `erpt` makes is over the sixteen.
    fn erpt_new_id(&mut self) -> [u8; ERPT_ID_SIZE] {
        let mut id = [0u8; ERPT_ID_SIZE];
        id[..8].copy_from_slice(&self.next_random_u64().to_le_bytes());
        id[8..ERPT_UUID_SIZE].copy_from_slice(&self.next_random_u64().to_le_bytes());
        id[6] = (id[6] & 0x0f) | 0x40;
        id[8] = (id[8] & 0x3f) | 0x80;
        id
    }

    /// Whether two ids name the same report or attachment: the sixteen bytes
    /// of UUID, not the twenty of the field holding it, which is what a real
    /// `erpt` compares and all a caller has to have filled in.
    fn erpt_same_id(a: &[u8], b: &[u8]) -> bool {
        a.len() >= ERPT_UUID_SIZE
            && b.len() >= ERPT_UUID_SIZE
            && a[..ERPT_UUID_SIZE] == b[..ERPT_UUID_SIZE]
    }

    /// A report id in the grouped form a UUID is written in, for diagnostics.
    fn erpt_id_text(id: &[u8]) -> String {
        let hex: String = id.iter().take(ERPT_UUID_SIZE).map(|b| format!("{b:02x}")).collect();
        format!("{}-{}-{}-{}-{}", &hex[..8], &hex[8..12], &hex[12..16], &hex[16..20], &hex[20..])
    }

    /// `nn::erpt::ReportType`. `Any` is a filter rather than a kind, and only
    /// `GetReportList` is ever handed one.
    fn erpt_report_type_name(report_type: u32) -> &'static str {
        match report_type {
            0 => "visible",
            1 => "invisible",
            2 => "any",
            _ => "unknown",
        }
    }

    /// The name of an `nn::erpt::CategoryId`.
    fn erpt_category_name(category: u32) -> String {
        let name = match category {
            1000 => "TestNx",
            1001 => "NANDTypeInfo",
            1002 => "NANDExtendedCsd",
            1003 => "BluetoothAudioInfo",
            id => ERPT_CATEGORIES.get(id as usize).copied().unwrap_or(""),
        };
        if name.is_empty() { format!("category {category}") } else { name.to_owned() }
    }

    /// `erpt:r` — "nn::erpt::sf::ISession", and the three interfaces it opens
    /// onto the journal `erpt:c` fills.
    ///
    /// All three of its commands are object getters, so the generic fallback's
    /// fabricated object id ended every one of them at its first call. Nothing
    /// but the error-report transfer and the settings screen behind it opens
    /// this at all.
    pub(super) fn erpt_session_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        if self.ipc_answer_control(tls, handle, "erpt:r", cmd_id)? {
            return Ok(());
        }
        let iface = self.ipc_interface(tls, handle, "erpt:r");
        match iface.as_str() {
            "erpt:report" => self.erpt_reader_request(tls, handle, cmd_id, false),
            "erpt:attachment" => self.erpt_reader_request(tls, handle, cmd_id, true),
            "erpt:manager" => self.erpt_manager_request(tls, handle, cmd_id),
            _ => match cmd_id {
                // OpenReport / OpenManager / OpenAttachment.
                Some(0) => self.reply_with_interface(tls, handle, "erpt:report").map(|_| ()),
                Some(1) => self.reply_with_interface(tls, handle, "erpt:manager").map(|_| ()),
                Some(2) => self.reply_with_interface(tls, handle, "erpt:attachment").map(|_| ()),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
        }
    }

    /// `IReport` and `IAttachment`. The two interfaces are the same six
    /// commands over two different journals, and the object holds the cursor
    /// its `Read` advances.
    fn erpt_reader_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
        attachment: bool,
    ) -> Result<()> {
        let iface = if attachment { "erpt:attachment" } else { "erpt:report" };
        let key = self.ipc_object_key(tls, handle);
        match cmd_id {
            // Open(ReportId / AttachmentId).
            Some(0) => {
                let id = self.read_bytes(self.ipc_request_data(tls), ERPT_ID_SIZE as u32);
                if self.erpt_body(&id, attachment).is_none() {
                    return self.write_ipc_response(tls, ERPT_NOT_FOUND, &[], &[], &[]);
                }
                let mut stored = [0u8; ERPT_ID_SIZE];
                let len = id.len().min(ERPT_ID_SIZE);
                stored[..len].copy_from_slice(&id[..len]);
                self.erpt_readers.insert(key, ErrorReportReader { id: stored, offset: 0 });
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // Read -> as much as is left into the output buffer, and how many
            // bytes that was. A caller reads until this answers zero.
            Some(1) => {
                let Some(reader) = self.erpt_readers.get(&key).cloned() else {
                    return self.write_ipc_response(tls, ERPT_NOT_FOUND, &[], &[], &[]);
                };
                let Some(body) = self.erpt_body(&reader.id, attachment) else {
                    return self.write_ipc_response(tls, ERPT_NOT_FOUND, &[], &[], &[]);
                };
                let rest = body[reader.offset.min(body.len())..].to_vec();
                let written = self.write_output_buffer(tls, 0, &rest);
                if let Some(reader) = self.erpt_readers.get_mut(&key) {
                    reader.offset += written as usize;
                }
                self.write_ipc_response(tls, 0, &[], &written.to_le_bytes(), &[])
            }
            // SetFlags / GetFlags. The one flag that matters is Transmitted,
            // which the transfer sets once it has uploaded a report; it is
            // stored and read back rather than acted on, since nothing here
            // uploads anything.
            Some(2) => {
                let flags = self.ipc_arg_u32(tls, 0);
                let Some(reader) = self.erpt_readers.get(&key).cloned() else {
                    return self.write_ipc_response(tls, ERPT_NOT_FOUND, &[], &[], &[]);
                };
                if !self.erpt_set_flags(&reader.id, attachment, flags) {
                    return self.write_ipc_response(tls, ERPT_NOT_FOUND, &[], &[], &[]);
                }
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(3) => {
                let Some(reader) = self.erpt_readers.get(&key).cloned() else {
                    return self.write_ipc_response(tls, ERPT_NOT_FOUND, &[], &[], &[]);
                };
                let Some(flags) = self.erpt_flags(&reader.id, attachment) else {
                    return self.write_ipc_response(tls, ERPT_NOT_FOUND, &[], &[], &[]);
                };
                self.write_ipc_response(tls, 0, &[], &flags.to_le_bytes(), &[])
            }
            // Close: the object stays alive and may be opened again.
            Some(4) => {
                self.erpt_readers.remove(&key);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // GetSize -> s64. A caller sizes the buffer it passes `Read` from
            // this, so answering zero for an open report is what makes the
            // whole read loop skip.
            Some(5) => {
                let Some(reader) = self.erpt_readers.get(&key).cloned() else {
                    return self.write_ipc_response(tls, ERPT_NOT_FOUND, &[], &[], &[]);
                };
                let Some(body) = self.erpt_body(&reader.id, attachment) else {
                    return self.write_ipc_response(tls, ERPT_NOT_FOUND, &[], &[], &[]);
                };
                let size = body.len() as i64;
                self.write_ipc_response(tls, 0, &[], &size.to_le_bytes(), &[])
            }
            _ => self.unimplemented_command(tls, iface, cmd_id),
        }
    }

    /// The bytes behind one report or attachment id, or `None` if nothing is
    /// filed under it.
    fn erpt_body(&self, id: &[u8], attachment: bool) -> Option<Vec<u8>> {
        if attachment {
            let held = self.erpt_attachments.iter().find(|a| Self::erpt_same_id(&a.id, id))?;
            Some(held.data.clone())
        } else {
            let held = self.erpt_reports.iter().find(|r| Self::erpt_same_id(&r.id, id))?;
            Some(held.body.clone())
        }
    }

    fn erpt_flags(&self, id: &[u8], attachment: bool) -> Option<u32> {
        if attachment {
            self.erpt_attachments.iter().find(|a| Self::erpt_same_id(&a.id, id)).map(|a| a.flags)
        } else {
            self.erpt_reports.iter().find(|r| Self::erpt_same_id(&r.id, id)).map(|r| r.flags)
        }
    }

    /// Store the flags a caller set, reporting whether anything was there to
    /// set them on.
    fn erpt_set_flags(&mut self, id: &[u8], attachment: bool, flags: u32) -> bool {
        let held = if attachment {
            self.erpt_attachments
                .iter_mut()
                .find(|a| Self::erpt_same_id(&a.id, id))
                .map(|a| &mut a.flags)
        } else {
            self.erpt_reports
                .iter_mut()
                .find(|r| Self::erpt_same_id(&r.id, id))
                .map(|r| &mut r.flags)
        };
        match held {
            Some(stored) => {
                *stored = flags;
                true
            }
            None => false,
        }
    }

    /// `IManager` — the journal as a whole: what is in it, what it costs, and
    /// an event for a report arriving in it.
    fn erpt_manager_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        match cmd_id {
            // GetReportList(out ReportList, ReportType filter). The count is
            // the buffer's own first word rather than an out-parameter, and
            // the whole structure is written so a caller that reads past the
            // count reads zeros rather than its own stack.
            Some(0) => {
                let filter = self.ipc_arg_u32(tls, 0);
                let mut list = vec![0u8; 8 + ERPT_REPORT_COUNT_MAX * ERPT_REPORT_INFO_SIZE];
                let mut count = 0usize;
                for report in &self.erpt_reports {
                    if filter != 2 && report.report_type != filter {
                        continue;
                    }
                    let at = 8 + count * ERPT_REPORT_INFO_SIZE;
                    let info = &mut list[at..at + ERPT_REPORT_INFO_SIZE];
                    info[0x00..0x04].copy_from_slice(&report.report_type.to_le_bytes());
                    info[0x04..0x18].copy_from_slice(&report.id);
                    let meta = report.meta.len().min(ERPT_META_SIZE);
                    info[0x18..0x18 + meta].copy_from_slice(&report.meta[..meta]);
                    info[0x3C..0x40].copy_from_slice(&report.flags.to_le_bytes());
                    info[0x40..0x48].copy_from_slice(&report.timestamp.to_le_bytes());
                    info[0x48..0x50].copy_from_slice(&report.timestamp.to_le_bytes());
                    info[0x50..0x58].copy_from_slice(&(report.body.len() as i64).to_le_bytes());
                    count += 1;
                }
                list[..4].copy_from_slice(&(count as u32).to_le_bytes());
                self.write_output_buffer(tls, 0, &list);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // GetEvent -> the event a newly filed report signals, through the
            // copy list: the server keeps its own.
            Some(1) => {
                let event = self.kept_event("erpt:report-created", handle);
                self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
            }
            // CleanupReports: everything the journal holds goes, attachments
            // with it — an attachment outliving its report is what the real
            // cleanup exists to prevent.
            Some(2) => {
                self.erpt_reports.clear();
                self.erpt_attachments.clear();
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // DeleteReport(ReportId), and the attachments it owns.
            Some(3) => {
                let id = self.read_bytes(self.ipc_request_data(tls), ERPT_ID_SIZE as u32);
                self.erpt_reports.retain(|report| !Self::erpt_same_id(&report.id, &id));
                self.erpt_attachments.retain(|held| !Self::erpt_same_id(&held.owner, &id));
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // GetStorageUsageStatistics -> what the journal is costing. All of
            // it is derived from what is actually held, and nothing has ever
            // been transmitted, so every report in it is still waiting.
            Some(4) => {
                let mut stats = [0u8; 0x38];
                let journal = self.erpt_journal_id();
                stats[..ERPT_UUID_SIZE].copy_from_slice(&journal);
                let used: usize = self.erpt_reports.iter().map(|r| r.body.len()).sum::<usize>()
                    + self.erpt_attachments.iter().map(|a| a.data.len()).sum::<usize>();
                let largest =
                    self.erpt_reports.iter().map(|r| r.body.len()).max().unwrap_or(0) as i64;
                stats[0x10..0x14].copy_from_slice(&(used as u32).to_le_bytes());
                stats[0x18..0x20].copy_from_slice(&largest.to_le_bytes());
                for kind in 0..2u32 {
                    let count = self
                        .erpt_reports
                        .iter()
                        .filter(|report| report.report_type == kind)
                        .count() as u32;
                    let at = 0x20 + kind as usize * 4;
                    stats[at..at + 4].copy_from_slice(&count.to_le_bytes());
                    let at = 0x30 + kind as usize * 4;
                    stats[at..at + 4].copy_from_slice(&count.to_le_bytes());
                }
                self.write_ipc_response(tls, 0, &[], &stats, &[])
            }
            // GetAttachmentList(out AttachmentList, ReportId) -> how many were
            // written. Command 5 through 19.0.1 and 6 from 20.0.0; the two are
            // the same command, so both ids answer it.
            Some(5) | Some(6) => {
                let id = self.read_bytes(self.ipc_request_data(tls), ERPT_ID_SIZE as u32);
                let mut list =
                    vec![0u8; 8 + ERPT_ATTACHMENTS_PER_REPORT * ERPT_ATTACHMENT_INFO_SIZE];
                let mut count = 0usize;
                for held in &self.erpt_attachments {
                    if !Self::erpt_same_id(&held.owner, &id) || count >= ERPT_ATTACHMENTS_PER_REPORT
                    {
                        continue;
                    }
                    let at = 8 + count * ERPT_ATTACHMENT_INFO_SIZE;
                    let info = &mut list[at..at + ERPT_ATTACHMENT_INFO_SIZE];
                    info[0x00..0x14].copy_from_slice(&held.owner);
                    info[0x14..0x28].copy_from_slice(&held.id);
                    info[0x28..0x2C].copy_from_slice(&held.flags.to_le_bytes());
                    info[0x30..0x38].copy_from_slice(&(held.data.len() as i64).to_le_bytes());
                    let name = held.name.as_bytes();
                    let name = &name[..name.len().min(ERPT_ATTACHMENT_NAME_MAX - 1)];
                    info[0x38..0x38 + name.len()].copy_from_slice(name);
                    count += 1;
                }
                list[..4].copy_from_slice(&(count as u32).to_le_bytes());
                self.write_output_buffer(tls, 0, &list);
                self.write_ipc_response(tls, 0, &[], &(count as u32).to_le_bytes(), &[])
            }
            // PopNotifiableErrorCodes -> the error codes the Home Menu should
            // put in front of the user. The struct leads with its own count,
            // so zeroing the buffer is the empty list.
            Some(7) => {
                self.zero_output_buffer(tls, 0);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // GetReportSizeMax -> the size a caller allocates its read buffer
            // from, which is a property of the format rather than of anything
            // held.
            Some(10) => {
                self.write_ipc_response(tls, 0, &[], &ERPT_REPORT_SIZE_MAX.to_le_bytes(), &[])
            }
            _ => self.unimplemented_command(tls, "erpt:manager", cmd_id),
        }
    }

    /// The journal's own id, made on the first ask. It identifies this run of
    /// the journal to whoever is reading reports out of it, so it has to stay
    /// the same for as long as the journal does.
    fn erpt_journal_id(&mut self) -> [u8; ERPT_UUID_SIZE] {
        if let Some(id) = self.erpt_journal_id {
            return id;
        }
        let full = self.erpt_new_id();
        let mut id = [0u8; ERPT_UUID_SIZE];
        id.copy_from_slice(&full[..ERPT_UUID_SIZE]);
        self.erpt_journal_id = Some(id);
        id
    }
}

#[cfg(test)]
mod tests {
    use crate::cpu::ipc::testing::*;
    use crate::cpu::Cpu;

    /// One `nn::erpt::ContextEntry`. Only the category is read out of it; the
    /// twenty fields behind it are the caller's, and `marker` stands in for
    /// them so a replaced record can be told from the one it replaced.
    fn erpt_context_entry(category: u32, marker: u8) -> Vec<u8> {
        let mut entry = vec![0u8; super::ERPT_CONTEXT_ENTRY_SIZE];
        entry[super::ERPT_CONTEXT_CATEGORY..super::ERPT_CONTEXT_CATEGORY + 4]
            .copy_from_slice(&category.to_le_bytes());
        entry[0x10] = marker;
        entry
    }

    /// Submit one category of context on an `erpt:c` session, from a context
    /// buffer the caller has already mapped.
    fn erpt_submit(cpu: &mut Cpu, buffer: u32, category: u32, marker: u8) {
        let entry = erpt_context_entry(category, marker);
        write_send_buffer_request(cpu, 0, &[], &[(buffer, entry.len() as u32), (0, 0)]);
        for (index, &byte) in entry.iter().enumerate() {
            cpu.mem.write_u8(buffer + index as u32, byte).unwrap();
        }
        cpu.erpt_context_request(TLS, 9, Some(0)).unwrap();
    }

    #[test]
    fn erpt_journals_one_context_record_per_category() {
        // The journal is a picture of *now*, not a log: a module resubmitting
        // its own category is replacing the record that is already there.
        // SubmitContext is `erpt:c`'s command 0, so the generic fallback
        // answered it with a fabricated object id and dropped every record a
        // report would have been built out of.
        const CONTEXT: u32 = 0x4000;
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        cpu.mem.map_zero(CONTEXT, 0x1000).unwrap();
        cpu.register_service_handle(9, "erpt:c");

        erpt_submit(&mut cpu, CONTEXT, 1, 0xA1);
        erpt_submit(&mut cpu, CONTEXT, 67, 0xB2);
        erpt_submit(&mut cpu, CONTEXT, 1, 0xC3);

        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "a plain success");
        let categories: Vec<u32> = cpu.erpt_contexts.iter().map(|held| held.category).collect();
        assert_eq!(categories, vec![1, 67], "ErrorInfo and ThermalInfo, once each");
        assert_eq!(cpu.erpt_contexts[0].entry[0x10], 0xC3, "the newest ErrorInfo wins");
    }

    #[test]
    fn erpt_files_the_journal_as_a_report_and_reads_it_back() {
        const CONTEXT: u32 = 0x4000;
        const META: u32 = 0x6000;
        const OUT: u32 = 0x8000;
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        cpu.mem.map_zero(CONTEXT, 0x1000).unwrap();
        cpu.mem.map_zero(META, 0x1000).unwrap();
        cpu.mem.map_zero(OUT, 0x2000).unwrap();
        cpu.register_service_handle(9, "erpt:c");
        erpt_submit(&mut cpu, CONTEXT, 16, 0x11);

        // CreateReportV0(ReportType_Visible, ContextEntry[], FieldList,
        // ReportMetaData): the context it brings joins the journal, and the
        // whole journal is what gets written out.
        let entry = erpt_context_entry(1, 0x22);
        for (index, &byte) in entry.iter().enumerate() {
            cpu.mem.write_u8(CONTEXT + index as u32, byte).unwrap();
        }
        for index in 0..super::ERPT_META_SIZE as u32 {
            cpu.mem.write_u8(META + index, 0x7E).unwrap();
        }
        write_send_buffer_request(
            &mut cpu,
            1,
            &0u32.to_le_bytes(),
            &[(CONTEXT, entry.len() as u32), (0, 0), (META, super::ERPT_META_SIZE as u32)],
        );
        cpu.erpt_context_request(TLS, 9, Some(1)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "CreateReportV0");
        assert_eq!(cpu.erpt_reports.len(), 1);

        let report = cpu.erpt_reports[0].clone();
        assert_eq!(report.report_type, 0, "Visible");
        assert_eq!(report.meta, vec![0x7Eu8; super::ERPT_META_SIZE]);
        assert_eq!(
            report.body.len(),
            2 * super::ERPT_CONTEXT_ENTRY_SIZE,
            "ApplicationInfo and ErrorInfo"
        );

        // erpt:r opens an IReport onto the same journal.
        cpu.register_service_handle(10, "erpt:r");
        marshal(&mut cpu, false, 0, &[]);
        cpu.erpt_session_request(TLS, 10, Some(0)).unwrap();
        let object = u64::from(cpu.mem.read_u32(TLS + 0x0C).unwrap());
        assert_ne!(object, 0, "OpenReport handed back no session");

        marshal(&mut cpu, false, 0, &report.id);
        cpu.erpt_session_request(TLS, object, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "Open");

        marshal(&mut cpu, false, 5, &[]);
        cpu.erpt_session_request(TLS, object, Some(5)).unwrap();
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), report.body.len() as u64, "GetSize");

        let size = report.body.len() as u32;
        write_map_buffer_request(&mut cpu, 1, &[], OUT, size, false);
        cpu.erpt_session_request(TLS, object, Some(1)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), size, "Read");
        assert_eq!(cpu.read_bytes(OUT, size), report.body);

        // Read takes no offset, so the cursor is the object's: a caller drains
        // a report by calling it until it answers zero.
        write_map_buffer_request(&mut cpu, 1, &[], OUT, size, false);
        cpu.erpt_session_request(TLS, object, Some(1)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "already drained");

        // IManager::GetReportList puts its count in the buffer's own header
        // rather than in the reply, and ReportType_Any (2) matches both kinds.
        marshal(&mut cpu, false, 1, &[]);
        cpu.erpt_session_request(TLS, 10, Some(1)).unwrap();
        let manager = u64::from(cpu.mem.read_u32(TLS + 0x0C).unwrap());
        write_map_buffer_request(&mut cpu, 0, &2u32.to_le_bytes(), OUT, 0x1600, false);
        cpu.erpt_session_request(TLS, manager, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u32(OUT).unwrap(), 1, "one report listed");
        assert_eq!(cpu.read_bytes(OUT + 0x0C, super::ERPT_ID_SIZE as u32), report.id.to_vec());
        assert_eq!(
            cpu.mem.read_u64(OUT + 0x58).unwrap(),
            report.body.len() as u64,
            "the size the transfer would read"
        );
    }

    #[test]
    fn erpt_will_not_open_a_report_it_does_not_hold() {
        // Answering success would hand the caller an IReport onto nothing:
        // GetSize says zero and Read says zero, so the transfer concludes the
        // report it just listed is empty rather than that it asked wrongly.
        let mut cpu = request(false, 0, &[0xAAu8; super::ERPT_ID_SIZE]);
        cpu.register_service_handle(9, "erpt:report");
        cpu.erpt_session_request(TLS, 9, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), super::ERPT_NOT_FOUND);
    }

    #[test]
    fn an_attachment_is_claimed_by_the_report_that_names_it() {
        const NAME: u32 = 0x4000;
        const DATA: u32 = 0x5000;
        const IDS: u32 = 0x6000;
        const OUT: u32 = 0x8000;
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        cpu.mem.map_zero(NAME, 0x1000).unwrap();
        cpu.mem.map_zero(DATA, 0x1000).unwrap();
        cpu.mem.map_zero(IDS, 0x1000).unwrap();
        cpu.mem.map_zero(OUT, 0x1000).unwrap();
        cpu.register_service_handle(9, "erpt:c");

        // SubmitAttachment(name, data) -> an AttachmentId. The id is the
        // server's to assign, so a bare success left the caller naming
        // whatever its out-parameter happened to hold.
        for (index, &byte) in b"log.bin\0".iter().enumerate() {
            cpu.mem.write_u8(NAME + index as u32, byte).unwrap();
        }
        cpu.mem.write_u32(DATA, 0xDEAD_BEEF).unwrap();
        write_send_buffer_request(&mut cpu, 9, &[], &[(NAME, 8), (DATA, 4)]);
        cpu.erpt_context_request(TLS, 9, Some(9)).unwrap();
        let id = cpu.read_bytes(TLS + 0x20, super::ERPT_ID_SIZE as u32);
        assert_ne!(id, vec![0u8; super::ERPT_ID_SIZE], "no AttachmentId handed back");
        assert_eq!(cpu.erpt_attachments[0].name, "log.bin");
        assert_eq!(cpu.erpt_attachments[0].data, 0xDEAD_BEEFu32.to_le_bytes().to_vec());

        // CreateReportWithAttachments(ReportType, ..., AttachmentId[]).
        for (index, &byte) in id.iter().enumerate() {
            cpu.mem.write_u8(IDS + index as u32, byte).unwrap();
        }
        write_send_buffer_request(
            &mut cpu,
            10,
            &1u32.to_le_bytes(),
            &[(0, 0), (0, 0), (IDS, super::ERPT_ID_SIZE as u32)],
        );
        cpu.erpt_context_request(TLS, 9, Some(10)).unwrap();
        let report = cpu.erpt_reports[0].clone();
        assert_eq!(report.report_type, 1, "Invisible");
        assert_eq!(cpu.erpt_attachments[0].owner.to_vec(), report.id.to_vec());

        // IManager::GetAttachmentList(out, ReportId) finds it by its owner.
        cpu.register_service_handle(10, "erpt:manager");
        write_map_buffer_request(&mut cpu, 6, &report.id, OUT, 0x200, false);
        cpu.erpt_session_request(TLS, 10, Some(6)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 1, "one attachment");
        assert_eq!(cpu.mem.read_u32(OUT).unwrap(), 1);
        assert_eq!(cpu.read_bytes(OUT + 8 + 0x14, super::ERPT_ID_SIZE as u32), id);
    }

    #[test]
    fn the_manager_event_fires_because_a_report_really_is_filed() {
        // Every other event these services hand out describes something that
        // does not happen on this console. This one does: `erpt` signals it
        // when a report lands in the journal, and the transfer waits on it.
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        cpu.register_service_handle(9, "erpt:c");
        cpu.register_service_handle(10, "erpt:manager");

        marshal(&mut cpu, false, 1, &[]);
        cpu.erpt_session_request(TLS, 10, Some(1)).unwrap();
        let event = u64::from(cpu.mem.read_u32(TLS + 0x0C).unwrap());
        assert_ne!(event, 0, "GetEvent handed back no copy handle");
        assert_eq!(cpu.event_signaled(event), Some(false));

        write_send_buffer_request(&mut cpu, 1, &0u32.to_le_bytes(), &[(0, 0), (0, 0), (0, 0)]);
        cpu.erpt_context_request(TLS, 9, Some(1)).unwrap();
        assert_eq!(cpu.event_signaled(event), Some(true));
    }
}
