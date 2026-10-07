//! AArch64 (A64) interpreter core: decode and execute, with instruction groups,
//! A32 in `a32`, and the block JIT in `jit`.

use crate::mem::Memory;
use crate::trace::Level;
use crate::IdMap;
use crate::{Error, Result};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::ops::{Deref, DerefMut, Index, IndexMut};

mod a32;
mod alu;
mod bits;
mod crypto;
mod debug;
mod exec;
mod fp;
mod jit;
mod loadstore;
mod regs;
mod simd;
mod system;

use crate::kernel::*;
use crate::services::*;

pub(crate) use crate::kernel::events::*;
pub use crate::kernel::layout::*;
pub use crate::kernel::sched::*;
pub use crate::kernel::sync::ArbiterWait;
pub(crate) use crate::kernel::sync::*;
pub use crate::kernel::thread::{ThreadContext, ThreadState};
pub use crate::services::audio::*;
pub use crate::services::fonts::*;
pub use crate::services::input::*;
pub use regs::*;

pub use crate::kernel::ipc::POINTER_BUFFER_SIZE;
pub use crate::kernel::ipc::{GapKind, ServiceGap};
pub use crate::kernel::thread_report::ThreadReport;
pub use crate::services::fs::{FsActivity, SaveDataQuota};
pub use crate::services::storage::SaveKey;
pub use a32::ExecMode;
pub use jit::{
    defers, emits, set_jit_host, translates, Entry, JitHost, JitStats, Layout, Refused, HOT, LEFT,
};

pub use crate::services::acc::{UserAccount, UsersRefused, MAX_USERS, NICKNAME_LEN};
pub(crate) use crate::services::acc::{DEFAULT_NICKNAME, DEFAULT_USER_UID};
pub use crate::services::am::KeyboardRequest;
pub(crate) use bits::decode_bit_mask;
use bits::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunReport {
    pub steps: u64,
    /// True if the machine halted rather than exhausting the step budget.
    pub halted: bool,
}

#[derive(Debug)]
pub struct Cpu {
    pub mem: Memory,
    /// X0..=X30 and the three meanings of register 31; see [`REG_SLOTS`].
    pub(crate) regs: [u64; REG_FILE],
    pub(crate) pc: u32,
    /// NZCV as PSTATE packs it (N=31, Z=30, C=29, V=28); shared with AArch32's CPSR.
    pub(crate) nzcv: u32,
    pub(crate) mode: ExecMode,
    /// AArch32 CPSR.Q, sticky until an APSR write.
    pub(crate) cpsr_q: bool,
    /// CPSR.GE, written by parallel adds and read by `SEL`.
    pub(crate) cpsr_ge: u8,
    /// AArch32 FPSCR N/Z/C/V, separate from [`Cpu::nzcv`] (moved by `VMRS APSR_nzcv`).
    pub(crate) fpscr_nzcv: u32,
    /// Q0..=Q31.
    pub(crate) vregs: VRegs,
    /// Per-thread FPCR: rounding mode, flush-to-zero, default NaN.
    pub(crate) fpcr: u32,
    /// FPSR cumulative exception flags, sticky until written.
    pub(crate) fpsr: u32,
    /// Console output from the UART syscall mode.
    pub out: Vec<u8>,
    /// Debug trace: per-instruction disassembly (when enabled) and fault context.
    pub trace: Vec<u8>,
    pub trace_enabled: bool,
    /// Trace buffer cap; the oldest text is dropped first.
    trace_cap: usize,
    /// Text was dropped and the host has not been told; see [`Cpu::note_dropped_trace`].
    trace_dropped: bool,
    pub halted: bool,
    /// The last `fatal:u` report, kept to classify a later `ExitProcess` as a crash.
    pub(crate) guest_fatal: Option<String>,
    /// Clock in 1.02 GHz cycles (`svcGetSystemTick`). One instruction is one cycle, but
    /// [`Cpu::reschedule`] also idles it forward, so it is not an instruction count.
    pub cycles: u64,
    /// Instructions actually retired; idling does not advance it.
    pub steps: u64,
    /// Ring buffer of the last `RECENT_LEN` straight-line runs `(first pc, count)`, dumped on fault.
    pub(crate) recent: [(u32, u32); RECENT_LEN],
    pub(crate) recent_len: usize,
    /// TPIDRRO_EL0: kernel-set TLS base, where the IPC buffer lives.
    pub(crate) tpidr: u64,
    /// TPIDR_EL0: guest-writable, separate from `tpidr` (the SDK writes it).
    pub(crate) tpidr_rw: u64,
    /// Next domain IPC out-object id.
    pub(crate) next_object_id: u32,
    /// Fake handles from `ConnectToNamedPort`/`GetService` to their service name.
    pub(crate) service_handles: IdMap<u64, String>,
    /// Next fake handle; 0 is invalid.
    pub(crate) next_handle: u32,
    pub(crate) next_domain_object_id: u32,
    /// (session handle, domain object id) to interface name.
    pub(crate) domain_objects: HashMap<(u64, u32), String>,
    /// Non-domain vi session handles to their sub-interface.
    pub(crate) vi_ifaces: IdMap<u64, String>,
    /// AM's message queue for the running applet; each state change is queued once.
    pub(crate) applet_messages: VecDeque<u32>,
    /// Shared `GetEventHandle` event, signalled when a message is queued.
    pub(crate) applet_event: Option<u64>,
    /// Whether the startup focus message has been handed out.
    pub(crate) applet_focus_announced: bool,
    /// The applet's sleep-lock event and whether the lock is held.
    pub(crate) sleep_lock_event: Option<u64>,
    pub(crate) sleep_lock_acquired: bool,
    /// Events for `IApplicationFunctions` 210 and `aoc` list changes.
    pub(crate) application_functions_210_event: Option<u64>,
    pub(crate) aoc_list_changed_event: Option<u64>,
    /// `GetDefaultDisplayResolutionChangeEvent`, fired on dock changes.
    pub(crate) display_resolution_event: Option<u64>,
    /// `GetApplicationRecordUpdateSystemEvent`; starts signalled, as the Home Menu waits on it.
    pub(crate) application_record_event: Option<u64>,
    /// `IApplicationManagerInterface`'s SD and game card events, by command id; never fired.
    pub(crate) ns_manager_events: BTreeMap<u32, u64>,
    /// `GetPopFromGeneralChannelEvent`; never fired.
    pub(crate) general_channel_event: Option<u64>,
    /// The `ILockAccessor` event: always signalled. See `Cpu::am_lock_accessor_event`.
    pub(crate) lock_accessor_event: Option<u64>,
    /// `IHOSBinderDriver::GetNativeHandle`'s event, always signalled.
    pub(crate) binder_event: Option<u64>,
    /// The title's NACP save quota, reported by `IApplicationFunctions`; set through
    /// [`Cpu::set_save_data_quota`] and not enforced.
    pub(crate) save_data_quota: fs::SaveDataQuota,
    /// Chosen from the NPDM system resource size and program id.
    pub(crate) memory_layout: MemoryLayout,
    /// Kept because the layout also depends on the program id, which may arrive later.
    pub(crate) system_resource_size: u32,
    /// The system shared buffer's nvmap `(handle, id)` and the next slot to acquire.
    pub(crate) shared_buffer: Option<(u32, u32)>,
    pub(crate) shared_buffer_slot: u32,
    /// See [`Cpu::set_operation_mode`].
    pub(crate) operation_mode: OperationMode,
    /// Application proxy (told `FocusStateChanged`) vs applet (told `ChangeIntoForeground`).
    pub(crate) applet_is_application: bool,
    /// `ISelfController` auto-sleep settings, stored so getters read them back.
    pub(crate) idle_time_detection_extension: u32,
    pub(crate) auto_sleep_disabled: bool,
    /// Stored so the getter reads it back.
    pub(crate) home_button_double_click_enabled: bool,
    /// Last `SetTerminateResult`, read back by `GetLastApplicationExitReason`.
    pub(crate) am_terminate_result: u32,
    /// Count of `BuildRandom` Miis; picks the face and stamps the create id.
    pub(crate) mii_random_sequence: u32,
    /// Unimplemented `(interface, command)` pairs already warned about.
    pub(crate) unimplemented_ipc: HashSet<(String, Option<u32>)>,
    /// Stubbed `(interface, command)` pairs already warned about; see [`Cpu::warn_stub`].
    pub(crate) stubbed_ipc: HashSet<(String, Option<u32>)>,
    /// Calls to service gaps since the host last asked; see [`Cpu::take_service_gaps`].
    pub(crate) gap_calls: BTreeMap<(GapKind, String, Option<u32>), u64>,
    /// Failed nvdrv ioctls since the host last asked; see [`Cpu::take_nv_errors`].
    nv_errors: BTreeMap<(String, u32, u32), u64>,
    /// Reused objects for [`Cpu::reply_with_fabricated_object`], by `(session, command)`:
    /// domain object id, sub-session handle, event.
    pub(crate) fabricated_objects: HashMap<(u64, u32), (u32, u64, u64)>,
    /// NROs mapped by `ldr:ro`, by mapped address; see [`Cpu::ldr_ro_request`].
    pub(crate) ro_modules: BTreeMap<u32, ldr::RoModule>,
    /// Registered NRRs, by address. Signatures are not checked.
    pub(crate) ro_registrations: BTreeMap<u32, u32>,
    /// Handles modelled as kernel events. Other handles are treated as always signalled
    /// by `WaitSynchronization`.
    pub(crate) events: IdMap<u64, Event>,
    /// The vsync event, fired on present and at each display refresh.
    pub(crate) vsync_event: Option<u64>,
    pub(crate) last_vsync_frame: u64,
    /// For the refresh that fires without a present.
    pub(crate) last_vsync_cycles: u64,
    /// Open SD directory handles to the entries not yet yielded.
    pub(crate) fs_dirs: IdMap<u64, Vec<crate::vfs::DirEntry>>,
    /// Open `IFile` objects: domain object id to path.
    pub(crate) fs_files: IdMap<u64, String>,
    /// `am` `IStorage` contents, by object.
    pub(crate) am_storages: IdMap<u64, Vec<u8>>,
    /// System data archives by data id, as sources.
    pub(crate) data_archives: IdMap<u64, Box<dyn crate::source::ByteSource>>,
    /// DLC indices `aoc:u` reports; the content is in `data_archives`.
    pub(crate) add_on_content: std::collections::BTreeSet<u32>,
    /// Base DLC id from the NACP; zero means derive it from the program id.
    pub(crate) add_on_content_base_id: u64,
    /// Save data, by save key.
    pub(crate) saves: HashMap<SaveKey, crate::vfs::Vfs>,
    /// The save an `fsp-srv` object addresses; absent for the SD card.
    pub(crate) fs_mount: IdMap<u64, SaveKey>,
    /// The data archive an open `IStorage` serves; absent for the process's own RomFS.
    pub(crate) fs_storage_archive: IdMap<u64, u64>,
    /// `SetGlobalAccessLogMode`'s value; round-trips because `nnSdk` reads it at startup.
    pub(crate) fs_access_log_mode: u32,
    /// `SetSpeedEmulationMode`'s value; round-trips, no effect.
    pub(crate) fs_speed_emulation_mode: u32,
    /// Result of the most recent IPC reply, for `TRACE_FS`.
    pub(crate) last_ipc_result: Option<u32>,
    pub fs_activity: FsActivity,
    /// RomFS file index per storage (`None` key for the process's own RomFS);
    /// `None` value for unreadable tables.
    pub(crate) romfs_indexes: BTreeMap<Option<u64>, Option<crate::romfs::RomFsIndex>>,
    /// Each card slot's `IEventNotifier` event, by opening command.
    pub(crate) fs_detection_events: BTreeMap<u32, u64>,
    /// Storages queued for `PopInData`.
    pub(crate) am_in_data: VecDeque<Vec<u8>>,
    /// What a directly run library applet pushed through `PushOutData`, for the host.
    pub(crate) am_out_data: Vec<Vec<u8>>,
    /// Storages queued for `PopInteractiveInData`, filled by [`Cpu::push_applet_interactive_in_data`].
    pub(crate) am_interactive_in: VecDeque<Vec<u8>>,
    /// The applet's interactive output, capped.
    pub(crate) am_interactive_out: Vec<Vec<u8>>,
    /// `GetPopInDataEvent`/`GetPopInteractiveInDataEvent` events, by [`am::AppletQueue`] slot.
    pub(crate) am_pop_events: [Option<u64>; 2],
    /// Launch parameters by `LaunchParameterKind`, each delivered once.
    pub(crate) am_launch_parameters: IdMap<u32, Vec<u8>>,
    /// The storage an `IStorageAccessor` addresses.
    pub(crate) am_storage_of: IdMap<u64, u64>,
    /// Library applets created through `ILibraryAppletCreator`, by accessor object.
    pub(crate) am_applets: IdMap<u64, am::LibraryApplet>,
    /// The software keyboard waiting on the host for text.
    pub(crate) am_keyboard: Option<am::PendingKeyboard>,
    /// The process's own RomFS (`OpenDataStorageByCurrentProcess`), read by range.
    /// `None` for homebrew, which reads RomFS from the SD card.
    pub(crate) romfs: Option<Box<dyn crate::source::ByteSource>>,
    /// Guest address of hid shared memory; 0 until mapped.
    pub(crate) hid_shmem_addr: u32,
    /// Handle used to recognise hid's shared memory in `svcMapSharedMemory`.
    pub(crate) hid_shmem_handle: Option<u64>,
    /// Supported npad styles and joy-con hold type, read back by their getters.
    pub(crate) npad_style_set: u32,
    /// `AcquireNpadStyleSetUpdateEventHandle`'s auto-clearing event.
    pub(crate) npad_style_update_event: Option<u64>,
    pub(crate) npad_joy_hold_type: u64,
    /// Rumble amplitudes (low band, high band).
    pub(crate) vibration: (f32, f32),
    /// `ssl` state: interface revision, context count, and per-context options.
    pub(crate) ssl_interface_version: u32,
    pub(crate) ssl_contexts: u32,
    pub(crate) ssl_options: HashMap<(u64, u32), u32>,
    /// Built-in CA certificates, loaded on first use; empty if the store is missing.
    pub(crate) ssl_certificates: Option<Vec<net::SslCertificate>>,
    /// Next imported PKI id; 0 means "nothing imported".
    pub(crate) ssl_next_pki_id: u64,
    /// Service events by (purpose, object), so repeat requests get the same handle.
    /// See [`Cpu::kept_event`].
    pub(crate) service_events: HashMap<(&'static str, u64), u64>,
    /// `lbl` backlight settings.
    pub(crate) backlight: settings::Backlight,
    /// `set:sys` settings, read from save data on first use; see [`Cpu::system_settings`].
    pub(crate) system_settings: Option<settings::SystemSettings>,
    /// Settings items requested but missing, reported once each.
    pub(crate) missing_settings_items: HashSet<String>,
    /// `audctl`'s system-wide audio settings.
    pub(crate) audio_control: audout::AudioControl,
    /// `nfc:sys` initialized flag; see [`Cpu::nfc_request`].
    pub(crate) nfc_initialized: bool,
    /// `btm:sys`: whether controller pairing is running.
    pub(crate) bt_gamepad_pairing: bool,
    /// `notif` alarms and the next alarm id.
    pub(crate) notif_alarms: Vec<settings::AlarmSetting>,
    pub(crate) notif_next_alarm_id: u16,
    /// `erpt` journal state, kept only for the session.
    pub(crate) erpt_contexts: Vec<erpt::ErrorContext>,
    pub(crate) erpt_reports: Vec<erpt::ErrorReport>,
    pub(crate) erpt_attachments: Vec<erpt::ErrorReportAttachment>,
    pub(crate) erpt_readers: IdMap<u64, erpt::ErrorReportReader>,
    /// The journal id, created on first request.
    pub(crate) erpt_journal_id: Option<[u8; erpt::ERPT_UUID_SIZE]>,
    /// Sampling number for hid npad LIFO entries.
    pub(crate) sample_counter: u64,
    /// Last pad and contacts, republished by [`Cpu::hid_tick`].
    pub(crate) last_gamepad: (u64, i32, i32, i32, i32),
    pub(crate) last_touches: Vec<TouchPoint>,
    pub(crate) last_hid_cycles: u64,
    /// Touch LIFO sampling number, separate from npad's.
    pub(crate) touch_sample_counter: u64,
    /// Touch slots filled at the last publish, so stale ones are cleared.
    pub(crate) touch_published: usize,
    /// Contacts down at the last publish; see [`Cpu::set_touch_state`].
    pub(crate) touch_down: Vec<TouchPoint>,
    /// The TrueType font `pl:u` serves for every shared font type; empty means no text.
    pub(crate) shared_font: Vec<u8>,
    /// pl's shared memory image, built by [`Cpu::build_shared_fonts`].
    pub(crate) pl_shmem_image: Vec<u8>,
    /// Each font's place in [`Cpu::pl_shmem_image`], in `PlSharedFontType` order.
    pub(crate) shared_font_regions: Vec<FontRegion>,
    /// Guest address of pl's shared memory; 0 until mapped.
    pub(crate) pl_shmem_addr: u32,
    /// Per-`IAudioRenderer` state from `OpenAudioRenderer`, used to size update replies.
    pub(crate) audren_renderers: IdMap<u64, audren::AudioRenderer>,
    /// Open `IAudioOut`s, by session handle.
    pub(crate) audio_outs: IdMap<u64, audout::AudioOut>,
    /// Open `IHardwareOpusDecoder`s; the guest work buffer is unused.
    pub(crate) opus_decoders: IdMap<u64, hwopus::HwOpus>,
    /// Bounded queue of interleaved 16-bit PCM not yet taken by the host.
    pub(crate) audio_pcm: VecDeque<i16>,
    /// Samples produced, taken and dropped; see [`Cpu::audio_activity`].
    pub(crate) audio_produced: u64,
    pub(crate) audio_taken: u64,
    pub(crate) audio_dropped: u64,
    /// Rate and channel count of `audio_pcm`; `(0, 0)` until a device opens.
    pub(crate) audio_format: (u32, u32),
    /// POSIX seconds for `time:u`/`time:s`; the epoch until [`Cpu::set_unix_time`].
    pub(crate) unix_time: i64,
    /// User accounts in `acc` order; never empty.
    pub(crate) users: Vec<UserAccount>,
    /// Index of the playing user in `users`.
    pub(crate) current_user: usize,
    /// The user each `IProfile`/`IProfileEditor` object was opened for.
    pub(crate) acc_profiles: IdMap<u64, [u8; 16]>,
    /// See [`Cpu::take_profile_edits`].
    pub(crate) profiles_edited: bool,
    /// Program id for `pm:info`; defaults to the Album applet's, as hbmenu homebrew runs as.
    pub(crate) program_id: u64,
    /// Clock rate last set per module; default in `CLOCK_RATES_HZ`.
    pub(crate) clock_rates: IdMap<u32, u32>,
    /// `mm:u` requests by id: (module, floor).
    pub(crate) mm_requests: IdMap<u32, (u32, u32)>,
    /// `csrng` state, seeded lazily from the clock; zero means unseeded.
    pub(crate) rng_state: u64,
    /// Open `bsd` sockets and their options.
    pub(crate) bsd_sockets: HashMap<i32, net::BsdSocket>,
    pub(crate) bsd_socket_options: HashMap<(i32, u32, u32), u32>,
    /// Next descriptor; starts at 3, past the standard streams.
    pub(crate) next_bsd_fd: i32,
    /// Next ephemeral port, from the bottom of IANA's range.
    pub(crate) next_bsd_port: u16,
    /// `ApmPerformanceConfiguration` for Normal and Boost.
    pub(crate) apm_configuration: [u32; 2],
    /// Battery level for `psm`, 0-100; full until [`Cpu::set_battery`].
    pub(crate) battery_percent: u8,
    pub(crate) battery_charging: bool,
    /// The emulated SD card.
    pub fs: crate::vfs::Vfs,
    pub nv: crate::gpu::nvdrv::NvDrv,
    /// The window buffer queue frames are presented through.
    pub display: crate::display::BufferQueue,
    /// Guest threads; index 0 is main. The running thread's slot is stale while it runs.
    pub(crate) threads: Vec<ThreadContext>,
    pub(crate) current_thread: usize,
    /// Address of the outstanding exclusive load, or `None`. Cleared on context switch.
    pub(crate) exclusive: Option<u32>,
    /// Instructions since the running thread was scheduled, against [`TIME_SLICE`].
    pub(crate) slice_used: u64,
    /// Next cycle at which [`Cpu::sweep_timed_waits`] checks deadlines.
    pub(crate) next_expiry: u64,
    jit: jit::Jit,
    /// Whether [`Cpu::run`] uses the JIT. `SWITCH_NO_JIT` disables it on the host;
    /// see [`Cpu::set_jit_enabled`].
    jit_enabled: bool,
    /// Set by a service call that would block; acted on after the reply is written to the caller's X0.
    pub(crate) pending_yield: bool,
    /// A park deadline applied at the same point as [`Cpu::pending_yield`].
    pub(crate) pending_sleep: Option<u64>,
    /// See [`Cpu::pace_present`].
    pub(crate) last_present_cycles: u64,
    /// A presented frame awaiting the GPU backend; see [`Cpu::complete_pending_present`].
    pub(crate) pending_present: Option<crate::gpu::DisplayBuffer>,
    /// `steps` when the running thread was last scheduled.
    pub(crate) switched_in_at: u64,
    /// Thread lifecycle events since the host last asked; see [`ThreadReport`].
    pub(crate) thread_log: Vec<String>,
    pub(crate) thread_log_dropped: u64,
    /// Every loaded module's `(start, end, name)`.
    pub(crate) module_names: Vec<(u32, u32, String)>,
    /// See [`Cpu::set_main_thread_priority`].
    pub(crate) main_thread_priority: u8,
    /// See [`Cpu::set_main_thread_core`].
    pub(crate) main_thread_core: u8,
    /// See [`Cpu::set_process_core_mask`].
    pub(crate) process_core_mask: u64,
    /// Next thread id; the main thread is 1.
    pub(crate) next_thread_id: u64,
}

pub const RECENT_LEN: usize = 64;

impl Default for Cpu {
    fn default() -> Self {
        Cpu::new()
    }
}

impl Cpu {
    pub fn new() -> Cpu {
        let mut cpu = Cpu {
            mode: ExecMode::A64,
            cpsr_q: false,
            cpsr_ge: 0,
            fpscr_nzcv: 0,
            mem: Memory::new(),
            regs: [0; REG_FILE],
            pc: 0,
            nzcv: 0,
            vregs: VRegs::default(),
            fpcr: 0,
            fpsr: 0,
            out: Vec::new(),
            trace: Vec::new(),
            trace_enabled: false,
            trace_cap: 512 * 1024,
            trace_dropped: false,
            halted: false,
            guest_fatal: None,
            cycles: 0,
            steps: 0,
            recent: [(0, 0); RECENT_LEN],
            recent_len: 0,
            tpidr: 0,
            tpidr_rw: 0,
            next_object_id: 1,
            service_handles: IdMap::default(),
            next_handle: 0x1000,
            next_domain_object_id: 1,
            domain_objects: HashMap::new(),
            vi_ifaces: IdMap::default(),
            applet_messages: VecDeque::new(),
            sleep_lock_event: None,
            sleep_lock_acquired: false,
            application_functions_210_event: None,
            application_record_event: None,
            ns_manager_events: BTreeMap::new(),
            general_channel_event: None,
            lock_accessor_event: None,
            binder_event: None,
            aoc_list_changed_event: None,
            display_resolution_event: None,
            save_data_quota: fs::SaveDataQuota::default(),
            memory_layout: MemoryLayout::PLAIN,
            system_resource_size: 0,
            shared_buffer: None,
            shared_buffer_slot: 0,
            applet_focus_announced: false,
            applet_is_application: true,
            idle_time_detection_extension: 0,
            auto_sleep_disabled: false,
            am_terminate_result: 0,
            mii_random_sequence: 0,
            home_button_double_click_enabled: false,
            operation_mode: OperationMode::default(),
            applet_event: None,
            unimplemented_ipc: HashSet::new(),
            stubbed_ipc: HashSet::new(),
            gap_calls: BTreeMap::new(),
            nv_errors: BTreeMap::new(),
            fabricated_objects: HashMap::new(),
            ro_modules: BTreeMap::new(),
            ro_registrations: BTreeMap::new(),
            events: IdMap::default(),
            vsync_event: None,
            last_vsync_frame: 0,
            last_vsync_cycles: 0,
            fs_dirs: IdMap::default(),
            fs_files: IdMap::default(),
            data_archives: IdMap::default(),
            add_on_content: std::collections::BTreeSet::new(),
            add_on_content_base_id: 0,
            saves: HashMap::new(),
            fs_mount: IdMap::default(),
            fs_storage_archive: IdMap::default(),
            fs_access_log_mode: 0,
            fs_speed_emulation_mode: 0,
            last_ipc_result: None,
            fs_activity: FsActivity::default(),
            romfs_indexes: BTreeMap::new(),
            fs_detection_events: BTreeMap::new(),
            am_in_data: VecDeque::new(),
            am_out_data: Vec::new(),
            am_interactive_in: VecDeque::new(),
            am_interactive_out: Vec::new(),
            am_pop_events: [None; 2],
            am_launch_parameters: IdMap::default(),
            am_storages: IdMap::default(),
            am_storage_of: IdMap::default(),
            am_applets: IdMap::default(),
            am_keyboard: None,
            romfs: None,
            touch_sample_counter: 0,
            touch_published: 0,
            touch_down: Vec::new(),
            hid_shmem_addr: 0,
            hid_shmem_handle: None,
            npad_style_set: 0,
            npad_style_update_event: None,
            npad_joy_hold_type: 0,
            vibration: (0.0, 0.0),
            ssl_interface_version: 0,
            ssl_contexts: 0,
            ssl_options: HashMap::new(),
            ssl_certificates: None,
            ssl_next_pki_id: 1,
            service_events: HashMap::new(),
            backlight: settings::Backlight::default(),
            system_settings: None,
            missing_settings_items: HashSet::new(),
            audio_control: audout::AudioControl::default(),
            nfc_initialized: false,
            bt_gamepad_pairing: false,
            notif_alarms: Vec::new(),
            // Starts at 1 so a zero-initialized id never names a real alarm.
            notif_next_alarm_id: 1,
            erpt_contexts: Vec::new(),
            erpt_reports: Vec::new(),
            erpt_attachments: Vec::new(),
            erpt_readers: IdMap::default(),
            erpt_journal_id: None,
            sample_counter: 0,
            last_gamepad: (0, 0, 0, 0, 0),
            last_touches: Vec::new(),
            last_hid_cycles: 0,
            shared_font: Vec::new(),
            pl_shmem_image: Vec::new(),
            shared_font_regions: Vec::new(),
            pl_shmem_addr: 0,
            audren_renderers: IdMap::default(),
            audio_outs: IdMap::default(),
            opus_decoders: IdMap::default(),
            audio_pcm: VecDeque::new(),
            audio_produced: 0,
            audio_taken: 0,
            audio_dropped: 0,
            audio_format: (0, 0),
            unix_time: 0,
            users: vec![UserAccount::new(DEFAULT_USER_UID, DEFAULT_NICKNAME, None)],
            current_user: 0,
            acc_profiles: IdMap::default(),
            profiles_edited: false,
            apm_configuration: power::APM_DEFAULT_CONFIGURATION,
            program_id: ipc::DEFAULT_PROGRAM_ID,
            clock_rates: IdMap::default(),
            mm_requests: IdMap::default(),
            rng_state: 0,
            bsd_sockets: HashMap::new(),
            bsd_socket_options: HashMap::new(),
            next_bsd_fd: 3,
            next_bsd_port: net::BSD_FIRST_EPHEMERAL_PORT,
            battery_percent: 100,
            battery_charging: true,
            fs: crate::vfs::Vfs::new(),
            nv: crate::gpu::nvdrv::NvDrv::new(),
            display: crate::display::BufferQueue::new(),
            threads: Vec::new(),
            current_thread: 0,
            exclusive: None,
            slice_used: 0,
            next_expiry: 0,
            jit: jit::Jit::default(),
            jit_enabled: !crate::env_flag!("SWITCH_NO_JIT"),
            pending_yield: false,
            pending_sleep: None,
            last_present_cycles: 0,
            pending_present: None,
            switched_in_at: 0,
            thread_log: Vec::new(),
            thread_log_dropped: 0,
            module_names: Vec::new(),
            main_thread_priority: DEFAULT_THREAD_PRIORITY,
            main_thread_core: 0,
            process_core_mask: crate::npdm::APPLICATION_CORE_MASK,
            next_thread_id: MAIN_THREAD_ID + 1,
        };
        // Pre-map the fixed framebuffer and input regions.
        let _ = cpu.mem.map_zero(
            crate::FB_BASE,
            (crate::FB_WIDTH * crate::FB_HEIGHT * 4) as usize,
        );
        let _ = cpu.mem.map_zero(crate::INPUT_ADDR, 4096);
        cpu
    }
}
