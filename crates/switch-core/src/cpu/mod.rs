//! AArch64 (A64) interpreter core: decode and execute, with instruction groups,
//! services (reached through `svc`), A32 in `a32`, and the block JIT in `jit`.

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
mod fp;
mod jit;
mod loadstore;
mod simd;
mod svc;
mod system;

// Horizon's services: `ipc` marshalling plus one module per domain, dispatched from `svc.rs`.
mod acc;
mod am;
mod audout;
mod audren;
mod erpt;
mod fs;
mod hid;
mod hwopus;
mod ipc;
mod ldr;
mod log;
mod mii;
mod net;
mod ns;
mod nv;
mod online;
mod pl;
mod power;
mod settings;
mod thread_report;
mod time;
mod vi;

mod audio;
mod boot;
mod debug;
mod events;
mod exec;
mod fonts;
mod input;
mod layout;
mod machine;
mod regs;
mod sched;
mod storage;
mod sync;
mod thread;

pub use audio::*;
use events::*;
pub use fonts::*;
pub use input::*;
pub use layout::*;
pub use regs::*;
pub use sched::*;
use sync::*;

pub use a32::ExecMode;
pub use fs::{FsActivity, SaveDataQuota};
pub use ipc::POINTER_BUFFER_SIZE;
pub use jit::{
    defers, emits, set_jit_host, translates, Entry, JitHost, JitStats, Layout, Refused, HOT, LEFT,
};
pub use thread_report::ThreadReport;

pub use acc::{UserAccount, UsersRefused, MAX_USERS, NICKNAME_LEN};
use acc::{DEFAULT_NICKNAME, DEFAULT_USER_UID};
pub(crate) use bits::decode_bit_mask;
use bits::*;

/// A save's id plus owning user; the zero uid marks shared (system and device) saves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SaveKey {
    pub id: u64,
    pub user: [u8; 16],
}

impl SaveKey {
    pub const fn shared(id: u64) -> SaveKey {
        SaveKey { id, user: [0; 16] }
    }

    /// From the uid's two little-endian halves as the host passes them.
    pub fn from_halves(id: u64, user_lo: u64, user_hi: u64) -> SaveKey {
        let mut user = [0u8; 16];
        user[..8].copy_from_slice(&user_lo.to_le_bytes());
        user[8..].copy_from_slice(&user_hi.to_le_bytes());
        SaveKey { id, user }
    }
}

impl std::fmt::Display for SaveKey {
    /// `0100000000001000` for a shared save, `id@uid` (32 hex digits, memory order) for a user's.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:016x}", self.id)?;
        if self.user != [0; 16] {
            f.write_str("@")?;
            for byte in self.user {
                write!(f, "{byte:02x}")?;
            }
        }
        Ok(())
    }
}

/// How a guest request went unanswered, for [`Cpu::take_service_gaps`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GapKind {
    /// Unknown command id, refused with an error.
    Refused,
    /// Unimplemented service, answered with a fabricated success.
    Missing,
    /// Answered, with nothing behind the answer.
    Stub,
    /// `nvdrv` ioctl with no handler.
    Ioctl,
}

impl GapKind {
    pub const fn name(self) -> &'static str {
        match self {
            GapKind::Refused => "refused",
            GapKind::Missing => "missing",
            GapKind::Stub => "stub",
            GapKind::Ioctl => "ioctl",
        }
    }
}

/// One unanswered request and its count since the last [`Cpu::take_service_gaps`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceGap {
    pub kind: GapKind,
    /// Interface, service or device node.
    pub name: String,
    pub command: Option<u32>,
    pub calls: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunReport {
    pub steps: u64,
    /// True if the machine halted rather than exhausting the step budget.
    pub halted: bool,
}

/// A guest thread's state. Threads only switch at blocking syscalls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadState {
    /// Created but not yet started with `svcStartThread`.
    Created,
    Runnable,
    /// Returned from its entry point or called `svcExitThread`.
    Finished,
    /// Blocked in `svcArbitrateLock` on the mutex word at this address.
    WaitMutex(u32),
    /// Blocked in `svcWaitProcessWideKeyAtomic`; re-acquires `mutex` when woken.
    /// `deadline` is the expiry cycle for timed waits.
    WaitKey {
        key: u32,
        mutex: u32,
        deadline: Option<u64>,
    },
    /// Blocked in `svcWaitForAddress` until signalled or `deadline` passes.
    WaitAddress {
        addr: u32,
        deadline: Option<u64>,
    },
    /// Asleep until `deadline` with the PC on the `svc`, which is reissued on wake.
    Sleeping {
        deadline: u64,
    },
    /// Blocked in `svcWaitSynchronization` with the PC on the `svc`; reissued when
    /// [`Cpu::signal_event`] wakes it or at `deadline` (the display tick).
    WaitEvent {
        deadline: u64,
    },
}

/// How an `svcWaitForAddress` resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArbiterWait {
    /// The predicate held and the caller is now blocked.
    Blocked,
    /// The word did not hold the expected value; nothing to wait for.
    Mismatch,
    /// The predicate held but the timeout was zero.
    TimedOut,
}

/// A kernel event a service handed the guest a handle to.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Event {
    /// Diagnostics only.
    name: &'static str,
    /// Events start unsignalled.
    signaled: bool,
    /// Whether a successful wait consumes the signal (auto-clear).
    auto_clear: bool,
}

/// One style the pad can present in hid's shared memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NpadPresentation {
    /// `HidNpadStyleTag` bit.
    style: u32,
    /// `HidDeviceTypeBits`.
    device_type: u32,
    /// The per-style LIFO.
    lifo: u32,
    /// `HidNpadAttribute`.
    attributes: u32,
    /// `HidNpadJoyAssignmentMode`.
    joy_assignment: u32,
}

/// SIMD register file; indices come from 5-bit fields, so accesses are in range.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct VRegs([u128; 32]);

impl Deref for VRegs {
    type Target = [u128; 32];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for VRegs {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Index<usize> for VRegs {
    type Output = u128;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < 32);
        // SAFETY: all architectural register fields are five bits and the
        // public u8 accessors mask them before indexing.
        unsafe { self.0.get_unchecked(index) }
    }
}

impl IndexMut<usize> for VRegs {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        debug_assert!(index < 32);
        // SAFETY: see `Index` above.
        unsafe { self.0.get_unchecked_mut(index) }
    }
}

#[derive(Debug, Clone)]
pub struct ThreadContext {
    pub handle: u64,
    /// Kernel thread id (`svcGetThreadId`), distinct from the handle.
    id: u64,
    pub state: ThreadState,
    /// Suspended by `svcSetThreadActivity`; independent of `state`.
    paused: bool,
    /// Saved register file, SP included; see [`REG_SLOTS`].
    regs: [u64; REG_FILE],
    pc: u32,
    nzcv: u32,
    mode: ExecMode,
    cpsr_q: bool,
    cpsr_ge: u8,
    fpscr_nzcv: u32,
    vregs: VRegs,
    fpcr: u32,
    fpsr: u32,
    tpidr: u64,
    tpidr_rw: u64,
    /// 0 (most urgent) to 63; see [`Cpu::pick_next`].
    priority: u8,
    /// Current core, ideal core (-1 for none) and affinity mask. Not scheduled on;
    /// `core` is what `GetCurrentProcessorNumber` answers.
    core: u8,
    ideal_core: i32,
    affinity: u64,
    /// Decisions passed over while runnable; see [`STARVE_DECISIONS`].
    passed_over: u32,
    /// Entry point and argument (an `nn::os` thread's `ThreadType`), for the thread report.
    entry: u32,
    arg: u64,
    /// Instructions retired and times scheduled since the last [`ThreadReport`].
    ran: u64,
    switches: u64,
    /// Clock when it last did real work; see [`ThreadReport::idle_ms`].
    busy_at: u64,
}

#[derive(Debug)]
pub struct Cpu {
    pub mem: Memory,
    /// X0..=X30 and the three meanings of register 31; see [`REG_SLOTS`].
    regs: [u64; REG_FILE],
    pc: u32,
    /// NZCV as PSTATE packs it (N=31, Z=30, C=29, V=28); shared with AArch32's CPSR.
    nzcv: u32,
    mode: ExecMode,
    /// AArch32 CPSR.Q, sticky until an APSR write.
    cpsr_q: bool,
    /// CPSR.GE, written by parallel adds and read by `SEL`.
    cpsr_ge: u8,
    /// AArch32 FPSCR N/Z/C/V, separate from [`Cpu::nzcv`] (moved by `VMRS APSR_nzcv`).
    pub(super) fpscr_nzcv: u32,
    /// Q0..=Q31.
    vregs: VRegs,
    /// Per-thread FPCR: rounding mode, flush-to-zero, default NaN.
    fpcr: u32,
    /// FPSR cumulative exception flags, sticky until written.
    fpsr: u32,
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
    guest_fatal: Option<String>,
    /// Clock in 1.02 GHz cycles (`svcGetSystemTick`). One instruction is one cycle, but
    /// [`Cpu::reschedule`] also idles it forward, so it is not an instruction count.
    pub cycles: u64,
    /// Instructions actually retired; idling does not advance it.
    pub steps: u64,
    /// Ring buffer of the last `RECENT_LEN` straight-line runs `(first pc, count)`, dumped on fault.
    recent: [(u32, u32); RECENT_LEN],
    recent_len: usize,
    /// TPIDRRO_EL0: kernel-set TLS base, where the IPC buffer lives.
    tpidr: u64,
    /// TPIDR_EL0: guest-writable, separate from `tpidr` (the SDK writes it).
    tpidr_rw: u64,
    /// Next domain IPC out-object id.
    next_object_id: u32,
    /// Fake handles from `ConnectToNamedPort`/`GetService` to their service name.
    service_handles: IdMap<u64, String>,
    /// Next fake handle; 0 is invalid.
    next_handle: u32,
    next_domain_object_id: u32,
    /// (session handle, domain object id) to interface name.
    domain_objects: HashMap<(u64, u32), String>,
    /// Non-domain vi session handles to their sub-interface.
    vi_ifaces: IdMap<u64, String>,
    /// AM's message queue for the running applet; each state change is queued once.
    applet_messages: VecDeque<u32>,
    /// Shared `GetEventHandle` event, signalled when a message is queued.
    applet_event: Option<u64>,
    /// Whether the startup focus message has been handed out.
    applet_focus_announced: bool,
    /// The applet's sleep-lock event and whether the lock is held.
    sleep_lock_event: Option<u64>,
    sleep_lock_acquired: bool,
    /// Events for `IApplicationFunctions` 210 and `aoc` list changes.
    application_functions_210_event: Option<u64>,
    aoc_list_changed_event: Option<u64>,
    /// `GetDefaultDisplayResolutionChangeEvent`, fired on dock changes.
    display_resolution_event: Option<u64>,
    /// `GetApplicationRecordUpdateSystemEvent`; starts signalled, as the Home Menu waits on it.
    application_record_event: Option<u64>,
    /// `IApplicationManagerInterface`'s SD and game card events, by command id; never fired.
    ns_manager_events: BTreeMap<u32, u64>,
    /// `GetPopFromGeneralChannelEvent`; never fired.
    general_channel_event: Option<u64>,
    /// The `ILockAccessor` event: always signalled. See `Cpu::am_lock_accessor_event`.
    lock_accessor_event: Option<u64>,
    /// `IHOSBinderDriver::GetNativeHandle`'s event, always signalled.
    binder_event: Option<u64>,
    /// The title's NACP save quota, reported by `IApplicationFunctions`; set through
    /// [`Cpu::set_save_data_quota`] and not enforced.
    save_data_quota: fs::SaveDataQuota,
    /// Chosen from the NPDM system resource size and program id.
    memory_layout: MemoryLayout,
    /// Kept because the layout also depends on the program id, which may arrive later.
    system_resource_size: u32,
    /// The system shared buffer's nvmap `(handle, id)` and the next slot to acquire.
    shared_buffer: Option<(u32, u32)>,
    shared_buffer_slot: u32,
    /// See [`Cpu::set_operation_mode`].
    operation_mode: OperationMode,
    /// Application proxy (told `FocusStateChanged`) vs applet (told `ChangeIntoForeground`).
    applet_is_application: bool,
    /// `ISelfController` auto-sleep settings, stored so getters read them back.
    idle_time_detection_extension: u32,
    auto_sleep_disabled: bool,
    /// Stored so the getter reads it back.
    home_button_double_click_enabled: bool,
    /// Last `SetTerminateResult`, read back by `GetLastApplicationExitReason`.
    am_terminate_result: u32,
    /// Count of `BuildRandom` Miis; picks the face and stamps the create id.
    mii_random_sequence: u32,
    /// Unimplemented `(interface, command)` pairs already warned about.
    unimplemented_ipc: HashSet<(String, Option<u32>)>,
    /// Stubbed `(interface, command)` pairs already warned about; see [`Cpu::warn_stub`].
    stubbed_ipc: HashSet<(String, Option<u32>)>,
    /// Calls to service gaps since the host last asked; see [`Cpu::take_service_gaps`].
    gap_calls: BTreeMap<(GapKind, String, Option<u32>), u64>,
    /// Failed nvdrv ioctls since the host last asked; see [`Cpu::take_nv_errors`].
    nv_errors: BTreeMap<(String, u32, u32), u64>,
    /// Reused objects for [`Cpu::reply_with_fabricated_object`], by `(session, command)`:
    /// domain object id, sub-session handle, event.
    fabricated_objects: HashMap<(u64, u32), (u32, u64, u64)>,
    /// NROs mapped by `ldr:ro`, by mapped address; see [`Cpu::ldr_ro_request`].
    ro_modules: BTreeMap<u32, ldr::RoModule>,
    /// Registered NRRs, by address. Signatures are not checked.
    ro_registrations: BTreeMap<u32, u32>,
    /// Handles modelled as kernel events. Other handles are treated as always signalled
    /// by `WaitSynchronization`.
    events: IdMap<u64, Event>,
    /// The vsync event, fired on present and at each display refresh.
    vsync_event: Option<u64>,
    last_vsync_frame: u64,
    /// For the refresh that fires without a present.
    last_vsync_cycles: u64,
    /// Open SD directory handles to the entries not yet yielded.
    fs_dirs: IdMap<u64, Vec<crate::vfs::DirEntry>>,
    /// Open `IFile` objects: domain object id to path.
    fs_files: IdMap<u64, String>,
    /// `am` `IStorage` contents, by object.
    am_storages: IdMap<u64, Vec<u8>>,
    /// System data archives by data id, as sources.
    data_archives: IdMap<u64, Box<dyn crate::source::ByteSource>>,
    /// DLC indices `aoc:u` reports; the content is in `data_archives`.
    add_on_content: std::collections::BTreeSet<u32>,
    /// Base DLC id from the NACP; zero means derive it from the program id.
    add_on_content_base_id: u64,
    /// Save data, by save key.
    saves: HashMap<SaveKey, crate::vfs::Vfs>,
    /// The save an `fsp-srv` object addresses; absent for the SD card.
    fs_mount: IdMap<u64, SaveKey>,
    /// The data archive an open `IStorage` serves; absent for the process's own RomFS.
    fs_storage_archive: IdMap<u64, u64>,
    /// `SetGlobalAccessLogMode`'s value; round-trips because `nnSdk` reads it at startup.
    fs_access_log_mode: u32,
    /// `SetSpeedEmulationMode`'s value; round-trips, no effect.
    fs_speed_emulation_mode: u32,
    /// Result of the most recent IPC reply, for `TRACE_FS`.
    last_ipc_result: Option<u32>,
    pub fs_activity: FsActivity,
    /// RomFS file index per storage (`None` key for the process's own RomFS);
    /// `None` value for unreadable tables.
    romfs_indexes: BTreeMap<Option<u64>, Option<crate::romfs::RomFsIndex>>,
    /// Each card slot's `IEventNotifier` event, by opening command.
    fs_detection_events: BTreeMap<u32, u64>,
    /// Storages queued for `PopInData`.
    am_in_data: VecDeque<Vec<u8>>,
    /// What a directly run library applet pushed through `PushOutData`, for the host.
    am_out_data: Vec<Vec<u8>>,
    /// Storages queued for `PopInteractiveInData`, filled by [`Cpu::push_applet_interactive_in_data`].
    am_interactive_in: VecDeque<Vec<u8>>,
    /// The applet's interactive output, capped.
    am_interactive_out: Vec<Vec<u8>>,
    /// `GetPopInDataEvent`/`GetPopInteractiveInDataEvent` events, by [`am::AppletQueue`] slot.
    am_pop_events: [Option<u64>; 2],
    /// Launch parameters by `LaunchParameterKind`, each delivered once.
    am_launch_parameters: IdMap<u32, Vec<u8>>,
    /// The storage an `IStorageAccessor` addresses.
    am_storage_of: IdMap<u64, u64>,
    /// Library applets created through `ILibraryAppletCreator`, by accessor object.
    am_applets: IdMap<u64, am::LibraryApplet>,
    /// The process's own RomFS (`OpenDataStorageByCurrentProcess`), read by range.
    /// `None` for homebrew, which reads RomFS from the SD card.
    romfs: Option<Box<dyn crate::source::ByteSource>>,
    /// Guest address of hid shared memory; 0 until mapped.
    hid_shmem_addr: u32,
    /// Handle used to recognise hid's shared memory in `svcMapSharedMemory`.
    hid_shmem_handle: Option<u64>,
    /// Supported npad styles and joy-con hold type, read back by their getters.
    npad_style_set: u32,
    /// `AcquireNpadStyleSetUpdateEventHandle`'s auto-clearing event.
    npad_style_update_event: Option<u64>,
    npad_joy_hold_type: u64,
    /// Rumble amplitudes (low band, high band).
    vibration: (f32, f32),
    /// `ssl` state: interface revision, context count, and per-context options.
    ssl_interface_version: u32,
    ssl_contexts: u32,
    ssl_options: HashMap<(u64, u32), u32>,
    /// Built-in CA certificates, loaded on first use; empty if the store is missing.
    ssl_certificates: Option<Vec<net::SslCertificate>>,
    /// Next imported PKI id; 0 means "nothing imported".
    ssl_next_pki_id: u64,
    /// Service events by (purpose, object), so repeat requests get the same handle.
    /// See [`Cpu::kept_event`].
    service_events: HashMap<(&'static str, u64), u64>,
    /// `lbl` backlight settings.
    backlight: settings::Backlight,
    /// `set:sys` settings, read from save data on first use; see [`Cpu::system_settings`].
    system_settings: Option<settings::SystemSettings>,
    /// Settings items requested but missing, reported once each.
    missing_settings_items: HashSet<String>,
    /// `audctl`'s system-wide audio settings.
    audio_control: audout::AudioControl,
    /// `nfc:sys` initialized flag; see [`Cpu::nfc_request`].
    nfc_initialized: bool,
    /// `btm:sys`: whether controller pairing is running.
    bt_gamepad_pairing: bool,
    /// `notif` alarms and the next alarm id.
    notif_alarms: Vec<settings::AlarmSetting>,
    notif_next_alarm_id: u16,
    /// `erpt` journal state, kept only for the session.
    erpt_contexts: Vec<erpt::ErrorContext>,
    erpt_reports: Vec<erpt::ErrorReport>,
    erpt_attachments: Vec<erpt::ErrorReportAttachment>,
    erpt_readers: IdMap<u64, erpt::ErrorReportReader>,
    /// The journal id, created on first request.
    erpt_journal_id: Option<[u8; erpt::ERPT_UUID_SIZE]>,
    /// Sampling number for hid npad LIFO entries.
    sample_counter: u64,
    /// Last pad and contacts, republished by [`Cpu::hid_tick`].
    last_gamepad: (u64, i32, i32, i32, i32),
    last_touches: Vec<TouchPoint>,
    last_hid_cycles: u64,
    /// Touch LIFO sampling number, separate from npad's.
    touch_sample_counter: u64,
    /// Touch slots filled at the last publish, so stale ones are cleared.
    touch_published: usize,
    /// Contacts down at the last publish; see [`Cpu::set_touch_state`].
    touch_down: Vec<TouchPoint>,
    /// The TrueType font `pl:u` serves for every shared font type; empty means no text.
    shared_font: Vec<u8>,
    /// pl's shared memory image, built by [`Cpu::build_shared_fonts`].
    pl_shmem_image: Vec<u8>,
    /// Each font's place in [`Cpu::pl_shmem_image`], in `PlSharedFontType` order.
    shared_font_regions: Vec<FontRegion>,
    /// Guest address of pl's shared memory; 0 until mapped.
    pl_shmem_addr: u32,
    /// Per-`IAudioRenderer` state from `OpenAudioRenderer`, used to size update replies.
    audren_renderers: IdMap<u64, audren::AudioRenderer>,
    /// Open `IAudioOut`s, by session handle.
    audio_outs: IdMap<u64, audout::AudioOut>,
    /// Open `IHardwareOpusDecoder`s; the guest work buffer is unused.
    opus_decoders: IdMap<u64, hwopus::HwOpus>,
    /// Bounded queue of interleaved 16-bit PCM not yet taken by the host.
    audio_pcm: VecDeque<i16>,
    /// Samples produced, taken and dropped; see [`Cpu::audio_activity`].
    audio_produced: u64,
    audio_taken: u64,
    audio_dropped: u64,
    /// Rate and channel count of `audio_pcm`; `(0, 0)` until a device opens.
    audio_format: (u32, u32),
    /// POSIX seconds for `time:u`/`time:s`; the epoch until [`Cpu::set_unix_time`].
    unix_time: i64,
    /// User accounts in `acc` order; never empty.
    users: Vec<UserAccount>,
    /// Index of the playing user in `users`.
    current_user: usize,
    /// The user each `IProfile`/`IProfileEditor` object was opened for.
    acc_profiles: IdMap<u64, [u8; 16]>,
    /// See [`Cpu::take_profile_edits`].
    profiles_edited: bool,
    /// Program id for `pm:info`; defaults to the Album applet's, as hbmenu homebrew runs as.
    program_id: u64,
    /// Clock rate last set per module; default in `CLOCK_RATES_HZ`.
    clock_rates: IdMap<u32, u32>,
    /// `mm:u` requests by id: (module, floor).
    mm_requests: IdMap<u32, (u32, u32)>,
    /// `csrng` state, seeded lazily from the clock; zero means unseeded.
    rng_state: u64,
    /// Open `bsd` sockets and their options.
    bsd_sockets: HashMap<i32, net::BsdSocket>,
    bsd_socket_options: HashMap<(i32, u32, u32), u32>,
    /// Next descriptor; starts at 3, past the standard streams.
    next_bsd_fd: i32,
    /// Next ephemeral port, from the bottom of IANA's range.
    next_bsd_port: u16,
    /// `ApmPerformanceConfiguration` for Normal and Boost.
    apm_configuration: [u32; 2],
    /// Battery level for `psm`, 0-100; full until [`Cpu::set_battery`].
    battery_percent: u8,
    battery_charging: bool,
    /// The emulated SD card.
    pub fs: crate::vfs::Vfs,
    pub nv: crate::gpu::nvdrv::NvDrv,
    /// The window buffer queue frames are presented through.
    pub display: crate::display::BufferQueue,
    /// Guest threads; index 0 is main. The running thread's slot is stale while it runs.
    threads: Vec<ThreadContext>,
    current_thread: usize,
    /// Address of the outstanding exclusive load, or `None`. Cleared on context switch.
    pub(crate) exclusive: Option<u32>,
    /// Instructions since the running thread was scheduled, against [`TIME_SLICE`].
    slice_used: u64,
    /// Next cycle at which [`Cpu::sweep_timed_waits`] checks deadlines.
    next_expiry: u64,
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
    switched_in_at: u64,
    /// Thread lifecycle events since the host last asked; see [`ThreadReport`].
    thread_log: Vec<String>,
    thread_log_dropped: u64,
    /// Every loaded module's `(start, end, name)`.
    module_names: Vec<(u32, u32, String)>,
    /// See [`Cpu::set_main_thread_priority`].
    main_thread_priority: u8,
    /// See [`Cpu::set_main_thread_core`].
    main_thread_core: u8,
    /// See [`Cpu::set_process_core_mask`].
    process_core_mask: u64,
    /// Next thread id; the main thread is 1.
    next_thread_id: u64,
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

#[cfg(test)]
mod tests {
    use super::{Cpu, CURRENT_THREAD_PSEUDO_HANDLE, DEFAULT_THREAD_PRIORITY, MAIN_THREAD_HANDLE};

    #[test]
    fn the_idle_moves_the_clock_and_leaves_the_step_count_alone() {
        // Idling advances the clock but not the instruction count.
        let mut cpu = Cpu::new();
        cpu.bootstrap();
        let (clock, steps) = (cpu.cycles, cpu.steps);

        cpu.sleep_until(clock + 1_000_000);

        assert_eq!(
            cpu.cycles,
            clock + 1_000_000,
            "the clock idled to the deadline"
        );
        assert_eq!(cpu.steps, steps, "the idle executed nothing");
    }

    /// Run `rounds` yielding decisions and count how often each thread got the CPU.
    fn shares(cpu: &mut Cpu, rounds: usize) -> Vec<usize> {
        let mut held = vec![0; cpu.threads.len()];
        for _ in 0..rounds {
            cpu.yield_thread();
            held[cpu.current_thread] += 1;
        }
        held
    }

    /// The most urgent thread gets most of the CPU; others still run within `STARVE_DECISIONS`.
    #[test]
    fn the_most_urgent_thread_runs_most_and_starves_nobody() {
        let mut cpu = Cpu::new();
        let urgent = cpu.create_thread(0x0800_0000, 0, 0x1000_0000, 30, 0);
        let idle = cpu.create_thread(0x0800_0000, 0, 0x1100_0000, 50, 0);
        assert!(cpu.start_thread(urgent) && cpu.start_thread(idle));

        let held = shares(&mut cpu, 900);
        assert!(
            held[1] > held[0] * 4,
            "the urgent thread dominates: {held:?}"
        );
        assert!(
            held[1] > held[2] * 4,
            "the urgent thread dominates: {held:?}"
        );
        assert!(held[0] > 0 && held[2] > 0, "nobody is starved: {held:?}");
    }

    /// Equal priorities take turns.
    #[test]
    fn threads_of_one_priority_take_turns() {
        let mut cpu = Cpu::new();
        let a = cpu.create_thread(0x0800_0000, 0, 0x1000_0000, DEFAULT_THREAD_PRIORITY, 0);
        let b = cpu.create_thread(0x0800_0000, 0, 0x1100_0000, DEFAULT_THREAD_PRIORITY, 0);
        assert!(cpu.start_thread(a) && cpu.start_thread(b));
        assert_eq!(shares(&mut cpu, 300), vec![100, 100, 100]);
    }

    /// `svcSetThreadPriority` through the pseudo handle affects the next decision.
    #[test]
    fn a_priority_set_through_the_pseudo_handle_is_the_one_scheduled_on() {
        let mut cpu = Cpu::new();
        let worker = cpu.create_thread(0x0800_0000, 0, 0x1000_0000, DEFAULT_THREAD_PRIORITY, 0);
        assert!(cpu.start_thread(worker));
        assert_eq!(cpu.thread_priority(CURRENT_THREAD_PSEUDO_HANDLE), Some(44));
        assert_eq!(cpu.thread_priority(worker), Some(44));
        assert_eq!(cpu.thread_priority(0xdead), None, "not a thread");

        assert!(cpu.set_thread_priority(CURRENT_THREAD_PSEUDO_HANDLE, 10));
        assert_eq!(cpu.thread_priority(MAIN_THREAD_HANDLE), Some(10));
        cpu.yield_thread();
        assert_eq!(
            cpu.current_thread, 0,
            "the more urgent main thread keeps running"
        );

        cpu.set_main_thread_priority(20);
        assert_eq!(cpu.thread_priority(MAIN_THREAD_HANDLE), Some(20));
    }
}
