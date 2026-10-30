//! AArch64 (A64) interpreter core: decode and execute, with instruction groups,
//! A32 in `a32`, and the block JIT in `jit`.

use crate::kernel::cores::Handback;
use crate::mem::AddressSpace;
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

pub(crate) use crate::kernel::process::{Ending, Process, ROOT_PID};
pub use crate::kernel::program::{Installed, Program};
use crate::services::*;

pub use crate::kernel::cores::{serve, CoreSlot};
pub(crate) use crate::kernel::events::*;
pub use crate::kernel::layout::*;
pub use crate::kernel::memory_use::{Region, TitleMemory};
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
    defers, emits, set_jit_host, simd_probe, tail_call_probe, translates, Entry, JitHost, JitStats,
    Layout, Refused, HOT, LEFT,
};

pub use crate::services::acc::{UserAccount, UsersRefused, MAX_USERS, NICKNAME_LEN};
pub(crate) use crate::services::acc::{DEFAULT_NICKNAME, DEFAULT_USER_UID};
pub use crate::services::am::{ErrorDialog, KeyboardRequest};
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
    /// The address space of the process this core runs.
    pub mem: AddressSpace,
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
    /// Clock in 1.02 GHz cycles (`svcGetSystemTick`). One instruction is one cycle, but
    /// [`Cpu::reschedule`] also idles it forward, so it is not an instruction count.
    pub cycles: u64,
    /// Instructions actually retired; idling does not advance it.
    pub steps: u64,
    /// Where real time has reached on the clock, once a host paces it.
    /// See [`Cpu::pace_to_real_time`].
    pub(crate) real_cycles: Option<u64>,
    /// Ring buffer of the last `RECENT_LEN` straight-line runs `(first pc, count)`, dumped on fault.
    pub(crate) recent: [(u32, u32); RECENT_LEN],
    pub(crate) recent_len: usize,
    /// TPIDRRO_EL0: kernel-set TLS base, where the IPC buffer lives.
    pub(crate) tpidr: u64,
    /// TPIDR_EL0: guest-writable, separate from `tpidr` (the SDK writes it).
    pub(crate) tpidr_rw: u64,
    pub(crate) current_thread: usize,
    /// The outstanding exclusive load, or `None`. Cleared on context switch.
    pub(crate) exclusive: Monitor,
    /// Instructions since the running thread was scheduled, against [`TIME_SLICE`].
    pub(crate) slice_used: u64,
    /// Next cycle at which [`Cpu::sweep_timed_waits`] checks deadlines.
    pub(crate) next_expiry: u64,
    jit: jit::Jit,
    /// Translated code of the other processes this core has run, by pid.
    parked_jits: BTreeMap<u64, jit::Jit>,
    /// Whether [`Cpu::run`] uses the JIT; see [`Cpu::set_jit_enabled`].
    jit_enabled: bool,
    /// Whether a blocking wait on vsync parks until the tick; see [`Cpu::set_vsync_throttle`].
    pub(crate) vsync_throttle: bool,
    /// Set by a service call that would block; acted on after the reply is written to the caller's X0.
    pub(crate) pending_yield: bool,
    /// A park deadline applied at the same point as [`Cpu::pending_yield`].
    pub(crate) pending_sleep: Option<u64>,
    /// `steps` when the running thread was last scheduled.
    pub(crate) switched_in_at: u64,
    /// The process this core runs.
    pub(crate) pid: u64,
    /// Which of the [`crate::mem::CORES`] this is; 0 runs the kernel.
    pub(crate) core: usize,
    /// On another core, why the lent thread stops; see [`Cpu::run_lent`].
    pub(crate) handback: Option<Handback>,
    /// Set by core 0 to have this core hand its thread back.
    pub(crate) recall: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Core 0's clock as it last published it, which the other cores read.
    pub(crate) clock: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// `clock`'s counter, where emitted code reads it; set with it by [`Cpu::share_clock`].
    pub(crate) clock_at: *const std::sync::atomic::AtomicU64,
    /// How the running process ended, acted on once the `svc` that ended it replied.
    pub(crate) process_ending: Option<crate::kernel::process::Ending>,
    /// Kernel and service state: one per console, shared by its cores.
    pub(crate) machine: Box<Machine>,
}

/// What an exclusive load armed: where, and what it read there (a pair's
/// second doubleword in `value[1]`), which the store compares against. Emitted code
/// reads and writes the fields in place, so the layout is fixed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct Monitor {
    pub(crate) addr: u32,
    /// Nonzero while the monitor is armed.
    pub(crate) armed: u32,
    pub(crate) value: [u64; 2],
}

impl Monitor {
    pub(crate) fn arm(&mut self, addr: u32, value: [u64; 2]) {
        *self = Monitor {
            addr,
            armed: 1,
            value,
        };
    }

    pub(crate) fn clear(&mut self) {
        self.armed = 0;
    }

    /// Disarm, answering what it read if it was armed at `addr`.
    pub(crate) fn take_at(&mut self, addr: u32) -> Option<[u64; 2]> {
        let held = self.armed != 0 && self.addr == addr;
        self.armed = 0;
        held.then_some(self.value)
    }
}

/// What the console holds beyond a core: kernel objects, services and devices.
/// [`Cpu`] reaches it through `Deref`.
#[derive(Debug)]
pub struct Machine {
    /// Next domain IPC out-object id.
    pub(crate) next_object_id: u32,
    /// Fake handles from `ConnectToNamedPort`/`GetService` to their service name.
    pub(crate) service_handles: IdMap<u64, String>,
    /// Sessions made by `CloneCurrentObject`, by the session whose domain and
    /// objects they share.
    pub(crate) session_owners: IdMap<u64, u64>,
    /// Owners closed while clones of them were still open.
    pub(crate) closed_owners: std::collections::BTreeSet<u64>,
    /// Next fake handle; 0 is invalid.
    pub(crate) next_handle: u32,
    pub(crate) next_domain_object_id: u32,
    /// (session handle, domain object id) to interface name.
    pub(crate) domain_objects: HashMap<(u64, u32), String>,
    /// Non-domain vi session handles to their sub-interface.
    pub(crate) vi_ifaces: IdMap<u64, String>,
    /// `GetApplicationRecordUpdateSystemEvent`; starts signalled, as the Home Menu waits on it.
    pub(crate) application_record_event: Option<u64>,
    /// `IApplicationManagerInterface`'s SD and game card events, by command id; never fired.
    pub(crate) ns_manager_events: BTreeMap<u32, u64>,
    /// `NsApplicationControlData` (NACP, then icon) by application id.
    pub(crate) application_controls: BTreeMap<u64, Vec<u8>>,
    /// `IHOSBinderDriver::GetNativeHandle`'s event, always signalled.
    pub(crate) binder_event: Option<u64>,
    /// The system shared buffer's nvmap `(handle, id)` and the next slot to acquire.
    pub(crate) shared_buffer: Option<(u32, u32)>,
    pub(crate) shared_buffer_slot: u32,
    /// The pages behind the usable slots, mapped at `SHARED_BUFFER_ADDR` in each process that asks.
    pub(crate) shared_buffer_pages: Option<std::sync::Arc<crate::mem::SharedPages>>,
    /// Transfer memory objects by handle: the creator's pages, shared.
    pub(crate) transfer_memories: IdMap<u64, std::sync::Arc<crate::mem::SharedPages>>,
    /// See [`Cpu::set_operation_mode`].
    pub(crate) operation_mode: OperationMode,
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
    /// Handles modelled as kernel events. Other handles are treated as always signalled
    /// by `WaitSynchronization`.
    pub(crate) events: IdMap<u64, Event>,
    /// The vsync event, fired on present and at each display refresh.
    pub(crate) vsync_event: Option<u64>,
    pub(crate) last_vsync_frame: u64,
    /// For the refresh that fires without a present.
    pub(crate) last_vsync_cycles: u64,
    /// System data archives by data id, as sources.
    pub(crate) data_archives: IdMap<u64, Box<dyn crate::source::ByteSource>>,
    /// Save data, by save key.
    pub(crate) saves: HashMap<SaveKey, crate::vfs::Vfs>,
    /// `SetGlobalAccessLogMode`'s value; round-trips because `nnSdk` reads it at startup.
    pub(crate) fs_access_log_mode: u32,
    /// `SetSpeedEmulationMode`'s value; round-trips, no effect.
    pub(crate) fs_speed_emulation_mode: u32,
    /// Result of the most recent IPC reply, for `TRACE_FS`.
    pub(crate) last_ipc_result: Option<u32>,
    pub fs_activity: FsActivity,
    /// Each card slot's `IEventNotifier` event, by opening command.
    pub(crate) fs_detection_events: BTreeMap<u32, u64>,
    /// Programs on the NAND the kernel can start, by program id.
    pub(crate) installed: BTreeMap<u64, crate::kernel::program::Installed>,
    /// Error applets started and not yet shown to the player.
    pub(crate) error_dialogs: VecDeque<am::ErrorDialog>,
    /// Supported npad styles and joy-con hold type, read back by their getters.
    pub(crate) npad_style_set: u32,
    /// `AcquireNpadStyleSetUpdateEventHandle`'s auto-clearing event.
    pub(crate) npad_style_update_event: Option<u64>,
    pub(crate) npad_joy_hold_type: u64,
    /// `EnableSixAxisSensorUnalteredPassthrough`, read back by its getter.
    pub(crate) six_axis_passthrough: bool,
    /// Rumble amplitudes (low band, high band).
    pub(crate) vibration: (f32, f32),
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
    /// `usb:ds` string descriptor slots in use, one bit per index.
    pub(crate) usb_ds_strings: u64,
    /// `notif` alarms and the next alarm id.
    pub(crate) notif_alarms: Vec<settings::AlarmSetting>,
    pub(crate) notif_next_alarm_id: u16,
    /// `erpt` journal state, kept only for the session.
    pub(crate) erpt_contexts: Vec<erpt::ErrorContext>,
    pub(crate) erpt_reports: Vec<erpt::ErrorReport>,
    pub(crate) erpt_attachments: Vec<erpt::ErrorReportAttachment>,
    /// The journal id, created on first request.
    pub(crate) erpt_journal_id: Option<[u8; erpt::ERPT_UUID_SIZE]>,
    /// hid's shared memory, written here and mapped by each process that asks for it.
    pub(crate) hid_shmem: std::sync::Arc<crate::mem::SharedPages>,
    /// Whether any process has mapped it, so there is someone to publish to.
    pub(crate) hid_mapped: bool,
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
    /// See [`Cpu::set_utc_offset`].
    pub(crate) utc_offset: i32,
    /// `time`'s shared memory, mapped by each process that asks for it.
    pub(crate) time_shmem: std::sync::Arc<crate::mem::SharedPages>,
    /// User accounts in `acc` order; never empty.
    pub(crate) users: Vec<UserAccount>,
    /// Index of the playing user in `users`.
    pub(crate) current_user: usize,
    /// See [`Cpu::take_profile_edits`].
    pub(crate) profiles_edited: bool,
    /// Clock rate last set per module; default in `CLOCK_RATES_HZ`.
    pub(crate) clock_rates: IdMap<u32, u32>,
    /// `mm:u` requests by id: (module, floor).
    pub(crate) mm_requests: IdMap<u32, (u32, u32)>,
    /// `csrng` state, seeded lazily from the clock; zero means unseeded.
    pub(crate) rng_state: u64,
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
    /// See [`Cpu::pace_present`].
    pub(crate) last_present_cycles: u64,
    /// Presented frames awaiting the GPU backend, with the process presenting each and the
    /// readback generation it waits on; see [`Cpu::complete_pending_present`].
    pub(crate) pending_present: std::collections::VecDeque<(u64, crate::gpu::DisplayBuffer, u64)>,
    /// Thread lifecycle events since the host last asked; see [`ThreadReport`].
    pub(crate) thread_log: Vec<String>,
    pub(crate) thread_log_dropped: u64,
    /// Running processes by pid; see [`Cpu::process`].
    pub(crate) processes: BTreeMap<u64, Process>,
    /// The running title's pid; its ending halts the console.
    pub(crate) root: u64,
    /// The other cores core 0 lends threads to; see [`Cpu::add_core`].
    pub(crate) cores: Vec<std::sync::Arc<crate::kernel::cores::CoreSlot>>,
    /// Instructions those cores retired; see [`Cpu::other_core_steps`].
    pub(crate) lent_steps: u64,
}

pub const RECENT_LEN: usize = 64;

impl Cpu {
    /// The process this core runs.
    pub(crate) fn process(&self) -> &Process {
        &self.machine.processes[&self.pid]
    }

    pub(crate) fn process_mut(&mut self) -> &mut Process {
        self.machine.process_mut(self.pid)
    }

    /// Point this core at process `pid`: its address space and its translated code.
    pub(crate) fn enter_process(&mut self, pid: u64) {
        if pid == self.pid {
            return;
        }
        self.switch_translations(pid);
        self.mem = self.machine.processes[&pid].mem.clone();
        self.pid = pid;
    }

    /// Park the running process's translations and take out `pid`'s.
    pub(crate) fn switch_translations(&mut self, pid: u64) {
        let jit = self.parked_jits.remove(&pid).unwrap_or_default();
        let left = std::mem::replace(&mut self.jit, jit);
        self.parked_jits.insert(self.pid, left);
    }

    /// Drop the code this core translated for a process that ended.
    pub(crate) fn forget_translations(&mut self, pid: u64) {
        self.parked_jits.remove(&pid);
    }

    /// Run kernel bookkeeping in process `pid`'s address space, then come back.
    pub(crate) fn in_process<R>(&mut self, pid: u64, f: impl FnOnce(&mut Cpu) -> R) -> R {
        if pid == self.pid {
            return f(self);
        }
        let (pid_was, mem_was) = (self.pid, self.mem.clone());
        self.mem = self.machine.processes[&pid].mem.clone();
        self.pid = pid;
        let result = f(self);
        self.pid = pid_was;
        self.mem = mem_was;
        result
    }
}

impl Machine {
    /// Process `pid`, borrowing only the machine so a core's registers stay usable.
    pub(crate) fn process_mut(&mut self, pid: u64) -> &mut Process {
        self.processes
            .get_mut(&pid)
            .expect("the running process exists")
    }
}

impl Deref for Cpu {
    type Target = Machine;

    fn deref(&self) -> &Machine {
        &self.machine
    }
}

impl DerefMut for Cpu {
    fn deref_mut(&mut self) -> &mut Machine {
        &mut self.machine
    }
}

impl Default for Cpu {
    fn default() -> Self {
        Cpu::new()
    }
}

impl Cpu {
    pub fn new() -> Cpu {
        let mem = AddressSpace::new();
        let clock = std::sync::Arc::default();
        let cpu = Cpu {
            mode: ExecMode::A64,
            cpsr_q: false,
            cpsr_ge: 0,
            fpscr_nzcv: 0,
            mem: mem.clone(),
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
            cycles: 0,
            steps: 0,
            recent: [(0, 0); RECENT_LEN],
            recent_len: 0,
            tpidr: 0,
            tpidr_rw: 0,
            current_thread: 0,
            exclusive: Monitor::default(),
            slice_used: 0,
            next_expiry: 0,
            jit: jit::Jit::default(),
            parked_jits: BTreeMap::new(),
            jit_enabled: true,
            vsync_throttle: true,
            pending_yield: false,
            pending_sleep: None,
            switched_in_at: 0,
            pid: ROOT_PID,
            core: 0,
            handback: None,
            real_cycles: None,
            recall: Default::default(),
            clock_at: std::sync::Arc::as_ptr(&clock),
            clock,
            process_ending: None,
            machine: Box::new(Machine {
                next_object_id: 1,
                service_handles: IdMap::default(),
                session_owners: IdMap::default(),
                closed_owners: std::collections::BTreeSet::new(),
                next_handle: 0x1000,
                next_domain_object_id: 1,
                domain_objects: HashMap::new(),
                vi_ifaces: IdMap::default(),
                application_record_event: None,
                ns_manager_events: BTreeMap::new(),
                application_controls: BTreeMap::new(),
                binder_event: None,
                shared_buffer: None,
                shared_buffer_slot: 0,
                shared_buffer_pages: None,
                transfer_memories: IdMap::default(),
                am_terminate_result: 0,
                mii_random_sequence: 0,
                operation_mode: OperationMode::default(),
                unimplemented_ipc: HashSet::new(),
                stubbed_ipc: HashSet::new(),
                gap_calls: BTreeMap::new(),
                nv_errors: BTreeMap::new(),
                fabricated_objects: HashMap::new(),
                events: IdMap::default(),
                vsync_event: None,
                last_vsync_frame: 0,
                last_vsync_cycles: 0,
                data_archives: IdMap::default(),
                saves: HashMap::new(),
                fs_access_log_mode: 0,
                fs_speed_emulation_mode: 0,
                last_ipc_result: None,
                fs_activity: FsActivity::default(),
                fs_detection_events: BTreeMap::new(),
                installed: BTreeMap::new(),
                error_dialogs: VecDeque::new(),
                touch_sample_counter: 0,
                touch_published: 0,
                touch_down: Vec::new(),
                npad_style_set: 0,
                npad_style_update_event: None,
                npad_joy_hold_type: 0,
                six_axis_passthrough: false,
                vibration: (0.0, 0.0),
                ssl_certificates: None,
                ssl_next_pki_id: 1,
                service_events: HashMap::new(),
                backlight: settings::Backlight::default(),
                system_settings: None,
                missing_settings_items: HashSet::new(),
                audio_control: audout::AudioControl::default(),
                nfc_initialized: false,
                bt_gamepad_pairing: false,
                usb_ds_strings: 0,
                notif_alarms: Vec::new(),
                // Starts at 1 so a zero-initialized id never names a real alarm.
                notif_next_alarm_id: 1,
                erpt_contexts: Vec::new(),
                erpt_reports: Vec::new(),
                erpt_attachments: Vec::new(),
                erpt_journal_id: None,
                hid_shmem: crate::mem::SharedPages::new(HID_SHMEM_SIZE as usize),
                hid_mapped: false,
                sample_counter: 0,
                last_gamepad: (0, 0, 0, 0, 0),
                last_touches: Vec::new(),
                last_hid_cycles: 0,
                shared_font: Vec::new(),
                pl_shmem_image: Vec::new(),
                shared_font_regions: Vec::new(),
                audio_pcm: VecDeque::new(),
                audio_produced: 0,
                audio_taken: 0,
                audio_dropped: 0,
                audio_format: (0, 0),
                unix_time: 0,
                utc_offset: 0,
                time_shmem: crate::mem::SharedPages::new(crate::services::time::SHARED_MEMORY_SIZE),
                users: vec![UserAccount::new(DEFAULT_USER_UID, DEFAULT_NICKNAME, None)],
                current_user: 0,
                profiles_edited: false,
                apm_configuration: power::APM_DEFAULT_CONFIGURATION,
                clock_rates: IdMap::default(),
                mm_requests: IdMap::default(),
                rng_state: 0,
                next_bsd_port: net::BSD_FIRST_EPHEMERAL_PORT,
                battery_percent: 100,
                battery_charging: true,
                fs: crate::vfs::Vfs::new(),
                nv: crate::gpu::nvdrv::NvDrv::new(),
                display: crate::display::BufferQueue::new(),
                last_present_cycles: 0,
                pending_present: std::collections::VecDeque::new(),
                thread_log: Vec::new(),
                thread_log_dropped: 0,

                processes: BTreeMap::from([(ROOT_PID, Process::new(mem))]),
                root: ROOT_PID,
                cores: Vec::new(),
                lent_steps: 0,
            }),
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
