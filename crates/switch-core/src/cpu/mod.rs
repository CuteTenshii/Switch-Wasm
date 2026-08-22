//! AArch64 (A64) interpreter core.
//!
//! Implements a from-scratch decode + execute loop for the A64 instruction
//! set covering the integer core that compiled Switch homebrew actually uses:
//! integer ALU, shifts, bitfield ops, multiplies/divides, conditional selects
//! and compares, loads/stores (immediate, register-offset, literal, paired,
//! exclusive), PC-relative addressing, and the branch/subroutine family.
//!
//! System instructions (MRS/MSR/barriers/hints) are handled minimally, and
//! `SVC` drives a small, explicit syscall ABI used by the bundled demo
//! payload. Floating point, SIMD and the Horizon OS are out of scope for
//! Phase 1 and raise [`Error::Cpu`] if encountered.
//!
//! Encoding references are taken from the ARMv8 architecture and cross-checked
//! against QEMU's `target/arm/tcg/a64.decode`.

use crate::mem::Memory;
use crate::{Error, Result};
use std::collections::{HashMap, HashSet, VecDeque};

mod alu;
mod bits;
mod fp;
mod ipc;
mod loadstore;
mod simd;
mod svc;
mod system;

pub(crate) use bits::decode_bit_mask;
use bits::*;
use ipc::{DEFAULT_NICKNAME, NICKNAME_LEN};


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunReport {
    /// Number of instructions executed this run.
    pub steps: u64,
    /// True if the machine reached a halt trap rather than exhausting the
    /// step budget.
    pub halted: bool,
}

/// Host-provided stack for [`Cpu::bootstrap`]: 1 MiB full-descending, top at
/// `STACK_TOP`. Clear of the NRO image (0x08000000+), the ASLR region homebrew's
/// own allocators search (`AslrRegionAddress`/`Size`: [0x08000000, 0x27000000)
/// — see `svcGetInfo` types 12/13) and the real heap `svcSetHeapSize` hands
/// out (0x30000000).
///
/// Used to sit at 0x10000000, inside that ASLR region — fine for hbmenu's own
/// small deko3d memblocks, but Mesa/Nouveau's GPU buffer-object pool (JKSV
/// pulls in a full `nvc0` Gallium driver) keeps growing as more
/// textures/icons get created and doesn't re-verify each new allocation
/// against `svcQueryMemory`, so a big enough one blew straight through the
/// stack's mapped pages, `memset`-zeroing saved return addresses on it.
/// Past the ASLR region and before the heap is address space nothing else
/// claims.
pub const STACK_SIZE: u64 = 0x0010_0000;
pub const STACK_TOP: u64 = 0x2810_0000;

/// Address of the return-address trampoline for direct-entered homebrew.
pub const SELF_RETURN_TRAMPOLINE: u32 = 0x1F00_0000;

/// Where a guest thread's entry point returns to: a stub that calls
/// `svcExitThread` (svc 0x0A), the way libnx's thread entry does.
pub const THREAD_EXIT_TRAMPOLINE: u32 = 0x1F00_0100;

/// The handle the main thread is known by (the environment block advertises the
/// same value as `EntryType_MainThreadHandle`).
pub const MAIN_THREAD_HANDLE: u64 = 1;

/// Base of the per-thread TLS blocks handed to threads the guest creates. The
/// main thread keeps `0x1FE0_0000` (see [`Cpu::bootstrap`]); children get a
/// page each above it, clear of the heap and the stack.
pub const THREAD_TLS_BASE: u32 = 0x1FE1_0000;
/// Distance between two threads' TLS blocks. Horizon's are 0x200 bytes; a page
/// each keeps the newlib reentrancy struct that follows out of the way too.
pub const THREAD_TLS_STRIDE: u32 = 0x1000;

/// The system shared buffer: the surface the Home Menu and the system's own
/// applets actually draw into. It is not a layer of their own — AM hands out
/// one buffer the whole system shares, an applet asks `vi` for a slot in it,
/// renders there and presents the slot back.
///
/// The geometry is the console's, and it is what the applets expect to be
/// told: seven slots, each a 1280x768 block-linear RGBA8888 image of which
/// the top 1280x720 is the visible frame. The extra 48 rows are the
/// block-linear padding, not picture.
pub const SHARED_BUFFER_ADDR: u32 = 0x7E00_0000;
pub const SHARED_BUFFER_SLOTS: u32 = 7;
pub const SHARED_BUFFER_WIDTH: u32 = 1280;
pub const SHARED_BUFFER_HEIGHT: u32 = 720;
/// Rows actually allocated per slot, rounded up for the block-linear layout.
pub const SHARED_BUFFER_SLOT_ROWS: u32 = 768;
pub const SHARED_BUFFER_STRIDE: u32 = SHARED_BUFFER_WIDTH * 4;
pub const SHARED_BUFFER_SLOT_SIZE: u32 = SHARED_BUFFER_STRIDE * SHARED_BUFFER_SLOT_ROWS;
pub const SHARED_BUFFER_SIZE: u32 = SHARED_BUFFER_SLOT_SIZE * SHARED_BUFFER_SLOTS;
/// Only the first two slots are ever handed out, which is what the console
/// reports too — `AcquireSharedFrameBuffer` answers `{0, 1, -1, -1}`.
pub const SHARED_BUFFER_USABLE_SLOTS: u32 = 2;

/// The AM messages this emulator queues for the running applet. Horizon has
/// many more; these are the two an applet's own boot turns on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AppletMessage {
    /// The applet's focus changed. AM queues one at startup and then nothing
    /// until the state really changes.
    FocusStateChanged = 15,
    /// AM asking an applet that took charge of its own display
    /// (`SetHandlesRequestToDisplay`) to show itself.
    RequestToDisplay = 41,
    /// The same transition told to an *applet* rather than an application.
    ChangeIntoForeground = 1,
}

/// What a guest thread is doing. Threads only switch at the blocking
/// syscalls, so a critical section that does not block is effectively atomic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadState {
    /// Created but not yet started with `svcStartThread`.
    Created,
    /// Eligible to run.
    Runnable,
    /// Returned from its entry point or called `svcExitThread`.
    Finished,
    /// Blocked in `svcArbitrateLock` on the mutex word at this address.
    WaitMutex(u32),
    /// Blocked in `svcWaitProcessWideKeyAtomic` on a condition variable, to
    /// re-acquire `mutex` when woken.
    /// Blocked on a condition variable. `deadline` is the cycle count the
    /// wait expires at, for the timed form — `None` is a wait with no timeout.
    WaitKey { key: u32, mutex: u32, deadline: Option<u64> },
}

/// A kernel event a service handed the guest a handle to.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Event {
    /// What the event is for. Diagnostics only, but a wait trace is unreadable
    /// without it.
    name: &'static str,
    /// Whether it is currently signalled. Events start **unsignalled**: an
    /// event nothing has fired is the ordinary case, and reporting the
    /// opposite is what made `nn::os::TryWaitSystemEvent` tell
    /// `nn::oe::GpuErrorHandler` that the GPU had faulted.
    signaled: bool,
    /// Whether a successful wait consumes the signal. Horizon calls this an
    /// auto-clear event; the alternative stays signalled until the guest
    /// clears it by hand.
    auto_clear: bool,
}

/// Bit Horizon's mutex words set to mean "someone is blocked on me, so the
/// unlock has to go through `svcArbitrateUnlock`".
const MUTEX_HAS_LISTENERS: u32 = 0x4000_0000;

/// The address-space range `svcGetInfo` reports as the stack region: where
/// libnx mirrors the stacks of the threads the guest creates. It sits inside the
/// ASLR region and clear of the image, the main stack and the heap.
pub const GUEST_STACK_REGION_ADDR: u32 = 0x1800_0000;
pub const GUEST_STACK_REGION_SIZE: u32 = 0x0800_0000;

/// The alias region: where `svcMapPhysicalMemory` backs pages, and how a
/// 39-bit-address-space application grows its heap. It has to live inside the
/// soft-mapped low 2 GiB `Cpu::bootstrap` sets up, because this emulator
/// addresses guest memory with a `u32`. Horizon's own alias region starts at
/// 0x10_0000_0000, and reporting *that* through `svcGetInfo` had `nnSdk`
/// asking to map memory at an address the emulator cannot represent at all —
/// which `svcMapPhysicalMemory` would silently truncate to 0.
pub const GUEST_ALIAS_REGION_ADDR: u32 = 0x4000_0000;
pub const GUEST_ALIAS_REGION_SIZE: u32 = 0x4000_0000;

/// The heap region `svcSetHeapSize` grows, for a caller that uses it (`libnx`
/// does; a retail `nnSdk` title uses the alias region above instead). It ends
/// where [`crate::FB_BASE`] begins.
pub const GUEST_HEAP_REGION_ADDR: u32 = 0x3000_0000;
pub const GUEST_HEAP_REGION_SIZE: u32 = 0x0F00_0000;

/// Size of hid's shared memory, the value libnx passes to `svcMapSharedMemory`.
/// Used to tell that mapping apart from any other shared memory the guest maps.
pub const HID_SHMEM_SIZE: u32 = 0x4_0000;

/// Size of `pl:u`'s shared memory, the region the system fonts live in
/// (`SHAREDMEMFONT_SIZE` in libnx). Recognised the same way as hid's.
pub const PL_SHMEM_SIZE: u32 = 0x110_0000;

/// Offsets into libnx's `HidSharedMemory` (`switch/services/hid.h`): the npad
/// section, the per-controller stride, and the fields of `HidNpadInternalState`
/// that `padUpdate` reads.
mod hid_shmem {
    /// `offsetof(HidSharedMemory, npad)`.
    pub const NPAD: u32 = 0x9A00;
    /// `sizeof(HidNpadSharedMemoryEntry)`; `internal_state` sits at its start.
    pub const ENTRY_SIZE: u32 = 0x5000;
    /// Slot `HidNpadIdType_Handheld` reads (players 1-8 are slots 0-7).
    pub const HANDHELD_SLOT: u32 = 8;

    pub const STYLE_SET: u32 = 0x00;
    pub const JOY_ASSIGNMENT_MODE: u32 = 0x04;
    pub const FULL_KEY_LIFO: u32 = 0x28;
    pub const HANDHELD_LIFO: u32 = 0x378;
    pub const DEVICE_TYPE: u32 = 0x4188;

    /// `HidNpadCommonLifo`: a 0x20-byte header then 17 storage entries. The
    /// header's fields are unused/buffer_count/tail/count; a reader takes
    /// `count` entries ending at `tail`.
    pub const LIFO_BUFFER_COUNT: u32 = 0x08;
    pub const LIFO_TAIL: u32 = 0x10;
    pub const LIFO_COUNT: u32 = 0x18;
    pub const LIFO_STORAGE: u32 = 0x20;
    pub const LIFO_CAPACITY: u64 = 17;

    /// `HidNpadCommonStateAtomicStorage`: a sampling number the reader uses to
    /// detect a torn read, then the `HidNpadCommonState` itself.
    pub const STORAGE_SAMPLING_NUMBER: u32 = 0x00;
    pub const STATE_SAMPLING_NUMBER: u32 = 0x08;
    pub const STATE_BUTTONS: u32 = 0x10;
    pub const STATE_STICK_L: u32 = 0x18;
    pub const STATE_STICK_R: u32 = 0x20;
    pub const STATE_ATTRIBUTES: u32 = 0x28;

    pub const STYLE_FULL_KEY: u32 = 1 << 0;
    pub const STYLE_HANDHELD: u32 = 1 << 1;

    pub const DEVICE_FULL_KEY: u32 = 1 << 0;
    pub const DEVICE_HANDHELD: u32 = (1 << 2) | (1 << 3); // HandheldLeft|Right

    pub const ATTR_CONNECTED: u32 = 1 << 0;
    pub const ATTR_WIRED: u32 = 1 << 1;
    pub const ATTR_LEFT_CONNECTED: u32 = 1 << 2;
    pub const ATTR_LEFT_WIRED: u32 = 1 << 3;
    pub const ATTR_RIGHT_CONNECTED: u32 = 1 << 4;
    pub const ATTR_RIGHT_WIRED: u32 = 1 << 5;

    /// `offsetof(HidSharedMemory, touch_screen)` - straight after the debug
    /// pad's 0x400. Its `HidTouchScreenLifo` sits at the start of the region,
    /// with the same 0x20-byte header the npad LIFOs use.
    pub const TOUCH_SCREEN: u32 = 0x400;
    /// The storage entry is `{u64 sampling_number, HidTouchScreenState}`, so
    /// the state itself begins one `u64` into it.
    pub const TOUCH_STATE: u32 = 0x08;

    /// Fields of `HidTouchScreenState`: its own sampling number, how many of
    /// the sixteen touch slots are live, then the slots.
    pub const TOUCH_SAMPLING_NUMBER: u32 = 0x00;
    pub const TOUCH_COUNT: u32 = 0x08;
    pub const TOUCH_TOUCHES: u32 = 0x10;

    /// `sizeof(HidTouchState)` and the fields of one.
    pub const TOUCH_SIZE: u32 = 0x28;
    pub const TOUCH_DELTA_TIME: u32 = 0x00;
    pub const TOUCH_ATTRIBUTES: u32 = 0x08;
    pub const TOUCH_FINGER_ID: u32 = 0x0C;
    pub const TOUCH_X: u32 = 0x10;
    pub const TOUCH_Y: u32 = 0x14;
    pub const TOUCH_DIAMETER_X: u32 = 0x18;
    pub const TOUCH_DIAMETER_Y: u32 = 0x1C;
    pub const TOUCH_ROTATION_ANGLE: u32 = 0x20;
}

/// Deflection past which hid reports the `HidNpadButton_StickL*`/`StickR*`
/// pseudo-buttons, which is what `HidNpadButton_AnyLeft` and friends look at.
const HID_STICK_THRESHOLD: i32 = 0x4000;

/// The console's touchscreen digitizer resolution. This is *not* the resolution
/// the guest is presenting at - hid reports touches in this space whatever the
/// title renders in, so the frontend scales its canvas onto it rather than the
/// other way round.
pub const TOUCH_SCREEN_WIDTH: u32 = 1280;
pub const TOUCH_SCREEN_HEIGHT: u32 = 720;

/// `HidTouchScreenState.touches` is a fixed sixteen slots.
pub const TOUCH_MAX: usize = 16;

/// Contact size reported for every touch. Real hid measures the contact patch;
/// nothing that runs here does more than check it is non-zero, and a mouse or a
/// trackpad has no width to report anyway.
const TOUCH_DIAMETER: u32 = 10;

/// One finger on the touchscreen, in [`TOUCH_SCREEN_WIDTH`] x
/// [`TOUCH_SCREEN_HEIGHT`] coordinates.
///
/// `finger_id` identifies a contact for as long as it stays down, so a title
/// tracking a drag can follow it; the frontend keeps one id per pointer for the
/// life of that pointer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TouchPoint {
    pub finger_id: u32,
    pub x: u32,
    pub y: u32,
}

/// The counts an `OpenAudioRenderer` call fixes for the lifetime of its
/// `IAudioRenderer` session — how big every later `RequestUpdateAudioRenderer`
/// reply has to be.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AudrenParams {
    pub revision: u32,
    pub voice_count: u32,
    pub sink_count: u32,
    pub effect_count: u32,
}


/// One open `IAudioOut` session: what it was opened with, and the buffer
/// bookkeeping its client polls.
///
/// A real device releases a buffer once its samples have been clocked out to
/// the DAC. There is no DAC here — the samples are copied into
/// [`Cpu::audio_pcm`] for the host to play — so a buffer is released as soon
/// as it has been copied. The guest sees a device that always keeps up, never
/// one that succeeded at something it did not do.
#[derive(Debug, Clone)]
pub(crate) struct AudioOut {
    /// Sample rate and channel count the device was opened with.
    pub sample_rate: u32,
    pub channel_count: u32,
    /// Whether `StartAudioOut` has been called and `StopAudioOut` has not.
    pub started: bool,
    /// The volume the guest set, 0.0..=1.0. Applied when samples are taken.
    pub volume: f32,
    /// Signalled every time a buffer is released — what
    /// `audoutWaitPlayFinish` blocks on.
    pub event: u64,
    /// Tags of buffers already consumed, waiting for the guest to collect
    /// them with `GetReleasedAudioOutBuffer`.
    pub released: VecDeque<u64>,
    /// Frames handed over since the device was opened, which is what
    /// `GetAudioOutPlayedSampleCount` reports.
    pub played_frames: u64,
}

/// One open `bsd:u` socket.
///
/// A socket here can be created, configured, bound and listened on — every
/// local operation a real one supports — and can never carry a byte, because
/// there is no network behind this service. See [`Cpu::bsd_request`].
#[derive(Debug, Clone)]
pub(crate) struct BsdSocket {
    /// The address family and socket type it was created with. The family is
    /// carried for `DuplicateSocket`; the type decides which "went nowhere"
    /// errno the data path reports.
    pub domain: u32,
    pub kind: u32,
    /// The raw `sockaddr` bytes `bind` was given, which `GetSockName` reports
    /// back.
    pub bound: Vec<u8>,
    /// The flags word `fcntl(F_SETFL)` set, stored verbatim so `F_GETFL` hands
    /// back exactly what the guest wrote.
    pub flags: u32,
    /// Whether `listen` was called — an `accept` on a socket that never
    /// listened is a different error from one nobody has connected to.
    pub listening: bool,
}

#[derive(Debug, Clone)]
pub struct ThreadContext {
    pub handle: u64,
    pub state: ThreadState,
    /// Suspended by `svcSetThreadActivity`. Kept apart from `state` because
    /// suspension does not replace what the thread was doing — a paused thread
    /// blocked on a mutex is still blocked on it when it resumes; it is only
    /// taken out of the scheduler's rotation meanwhile.
    paused: bool,
    regs: [u64; 31],
    sp: u64,
    pc: u32,
    nzcv: u32,
    vregs: [u128; 32],
    tpidr: u64,
    tpidr_rw: u64,
}

#[derive(Debug)]
pub struct Cpu {
    pub mem: Memory,
    /// X0..=X30 (X31 is the stack pointer).
    regs: [u64; 31],
    /// The stack pointer register (X31).
    sp: u64,
    pc: u32,
    /// NZCV, packed as ARM PSTATE does: N=31, Z=30, C=29, V=28.
    nzcv: u32,
    /// SIMD vector registers Q0..=Q31 (128-bit). Only the handful of
    /// instructions libnx's `memset`/`memcpy` rely on are implemented;
    /// full NEON is out of scope for Phase 1.
    vregs: [u128; 32],
    /// Console output accumulated by the UART syscall mode.
    pub out: Vec<u8>,
    /// Debug trace: per-instruction disassembly (when enabled) plus fault
    /// context with a register snapshot.
    pub trace: Vec<u8>,
    /// When true, each executed instruction is appended to `trace`.
    pub trace_enabled: bool,
    /// Safety cap on the trace buffer to avoid unbounded growth.
    trace_cap: usize,
    pub halted: bool,
    /// Instructions executed in total.
    pub cycles: u64,
    /// Ring buffer of the most recent `RECENT_LEN` `(pc, insn)` pairs, dumped
    /// on fault so the path into a crash is visible without full tracing.
    recent: [(u32, u32); RECENT_LEN],
    /// Total instructions recorded into [`Cpu::recent`].
    recent_len: usize,
    /// Base of the kernel-fixed Thread Local Region (TPIDRRO_EL0): where the
    /// IPC message buffer lives and where `create_thread` points each
    /// thread's own TLS block. Real hardware makes this read-only at EL0 —
    /// only the kernel sets it — unlike [`Cpu::tpidr_rw`] below.
    tpidr: u64,
    /// TPIDR_EL0: freely readable *and writable* by guest code, unlike
    /// `tpidr` above. Nintendo's SDK uses it for its own per-thread pointer
    /// bookkeeping, entirely separate from the kernel-provided TLS region.
    /// Real hardware backs these with two distinct registers; aliasing them
    /// to the same storage let a guest `msr tpidr_el0, x0` silently stomp
    /// the kernel-fixed TLS pointer IPC dispatch and thread setup depend on
    /// — confirmed by tracing a real title's `nnSdk` init, which does
    /// exactly that write and then found its own IPC/TLS-relative reads
    /// pointing at low, unmapped-feeling addresses afterward.
    tpidr_rw: u64,
    /// Monotonic id handed out for domain IPC out-objects, so each synthesized
    /// subservice gets a distinct non-zero object id.
    next_object_id: u32,
    /// Maps fake handles returned by `ConnectToNamedPort` or sm:`GetService`
    /// to the named port / service they represent, so `SendSyncRequest` can
    /// dispatch IPC commands to the right stub.
    service_handles: HashMap<u64, String>,
    /// Monotonic fake-handle allocator. 0 is invalid, so we start above the
    /// earlier hard-coded `FAKE_HANDLE`.
    next_handle: u32,
    /// Monotonic domain object id allocator for services that use IPC domains
    /// (e.g. `vi:m`).
    next_domain_object_id: u32,
    /// Maps (session handle, domain object id) to the interface name used for
    /// dispatching domain commands.
    domain_objects: HashMap<(u64, u32), String>,
    /// Maps non-domain vi session handles to their sub-interface (vi:iads,
    /// vi:ihosbd, ...) so the display stub can dispatch binder vs. display
    /// commands on the right session.
    vi_ifaces: HashMap<u64, String>,
    /// AM's message queue for the running applet, and the event that says it
    /// is not empty.
    ///
    /// Real AM enqueues each state change once and then reports "no message"
    /// until the next one; answering every poll with a fresh message made
    /// `appletMainLoop` re-process a focus change on every call. The queue is
    /// what lets there be more than one such message -- an applet that took
    /// responsibility for its own display waits for a second one before it
    /// draws anything at all.
    applet_messages: VecDeque<u32>,
    /// The handle handed out by `ICommonStateGetter::GetEventHandle`, kept so
    /// every caller gets the same event and queueing a message can signal the
    /// one that is actually being waited on.
    applet_event: Option<u64>,
    /// Whether the startup focus transition has been handed out yet. AM has it
    /// waiting before the process's first poll, and reports "no message" ever
    /// after unless the state really changes.
    applet_focus_announced: bool,
    /// The applet's sleep-lock event, and whether the lock is held. There is
    /// one of each per applet — handing out a fresh event per call would
    /// signal an object nobody is waiting on.
    sleep_lock_event: Option<u64>,
    sleep_lock_acquired: bool,
    /// The system shared buffer's nvmap `(handle, id)` once an applet has
    /// asked for it, and the slot the next acquire hands out.
    shared_buffer: Option<(u32, u32)>,
    shared_buffer_slot: u32,
    /// Whether the process opened an *application* proxy. It decides which
    /// message that transition is: an application is told `FocusStateChanged`,
    /// while an applet — every one of the system's own, the Home Menu included
    /// — is told `ChangeIntoForeground`. Sending an applet the application's
    /// message is sending it one it does not act on.
    applet_is_application: bool,
    /// Every `(interface, command)` pair already reported as having no
    /// implementation behind it, so the warning naming it prints once instead
    /// of once per call (`appletMainLoop` polls `am` every frame).
    unimplemented_ipc: HashSet<(String, Option<u32>)>,
    /// What [`Cpu::reply_with_fabricated_object`] hands back for a command
    /// nothing implements, keyed by `(session handle, command id)`: the domain
    /// object id, the plain sub-session handle, and the event — one for each
    /// shape of out parameter a caller cannot invent for itself. Allocated
    /// once and reused, so a guest polling such a command is not handed a
    /// fresh handle on every call.
    fabricated_objects: HashMap<(u64, u32), (u32, u64, u64)>,
    /// Handles that name a kernel **event**, and whether each has been
    /// signalled yet. A handle that is not in here is not modelled as an
    /// event, and [`Cpu::horizon_syscall`]'s `WaitSynchronization` keeps
    /// treating it as immediately signalled — thread handles, and every
    /// service handle a guest happens to wait on.
    events: HashMap<u64, Event>,
    /// The display's vsync event, once `vi` has handed it out. Signalled every
    /// time the guest presents a frame, which is the only periodic tick this
    /// emulator has: a render loop that waits on it has to be woken by
    /// something, and there is no real clock behind the display.
    vsync_event: Option<u64>,
    /// Frame count the vsync event was last fired for.
    last_vsync_frame: u64,
    /// Cycle count the vsync event was last fired at, for the refresh that
    /// happens whether or not the guest presented anything.
    last_vsync_cycles: u64,
    /// Synthetic SD-card directory state for the fsp-srv stub: maps an open
    /// directory handle to the entries it has not yielded yet
    /// (name bytes, entry type, file size). Lets `fsFsOpenDirectory` /
    /// `fsDirRead` hand NX-Shell's `FS::GetDirList` a real listing.
    fs_dirs: HashMap<u64, Vec<crate::vfs::DirEntry>>,
    /// Open `IFile` objects: domain object id to the path it was opened on.
    fs_files: HashMap<u64, String>,
    /// `am` `IStorage` contents, keyed by object. A library applet is handed
    /// its launch arguments as one of these and returns its result in
    /// another, so the bytes have to outlive the request that made it.
    am_storages: HashMap<u64, Vec<u8>>,
    /// The console's system data archives, by data id: the read-only content
    /// a title mounts that is not its own — an applet's shared assets, the
    /// system's Mii and amiibo resources. Each is another NCA's RomFS, so
    /// they are sources rather than buffers, exactly like the running title's.
    data_archives: HashMap<u64, Box<dyn crate::source::ByteSource>>,
    /// Save data, by the id it was opened under. A console keeps these on its
    /// NAND -- one per application for its own save, one per system save id
    /// for the system's -- and they are the only writable storage a title has
    /// that is not the SD card.
    saves: HashMap<u64, crate::vfs::Vfs>,
    /// Which storage an `fsp-srv` object addresses: a save id, or absent for
    /// the SD card. Files and directories inherit it from the filesystem they
    /// were opened through, so a path means nothing without it.
    fs_mount: HashMap<u64, u64>,
    /// Which data archive an open `IStorage` is serving. Absent means the
    /// storage is the process's own RomFS.
    fs_storage_archive: HashMap<u64, u64>,
    /// Storages queued for `ILibraryAppletSelfAccessor::PopInData` — what the
    /// applet's caller would have pushed before starting it.
    am_in_data: VecDeque<Vec<u8>>,
    /// Which storage an `IStorageAccessor` reads and writes. The accessor is
    /// a separate object from the storage it was opened on, and both ends
    /// have to see the same bytes.
    am_storage_of: HashMap<u64, u64>,
    /// The current process's own RomFS — what
    /// `OpenDataStorageByCurrentProcess` hands back as an `IStorage`. `None`
    /// until the loader calls [`Cpu::set_romfs`] or
    /// [`Cpu::set_romfs_source`] (homebrew has no NCA and never sets this; it
    /// reads its RomFS off the SD card by path instead, through the regular
    /// `IFileSystem`/`IFile` path).
    ///
    /// A source rather than a buffer: a retail title's RomFS is the bulk of a
    /// container that does not fit in memory, so it stays where it is and is
    /// decrypted range by range as the guest reads it.
    romfs: Option<Box<dyn crate::source::ByteSource>>,
    /// Address the guest mapped its hid shared memory to (via `MapSharedMemory`
    /// on the handle hid's IPC returned). The host writes gamepad state into
    /// the libnx `HidSharedMemory` layout there so `padUpdate` sees it; 0 means
    /// hid hasn't been initialized yet.
    hid_shmem_addr: u32,
    /// The handle `hid`'s `IAppletResource::GetSharedMemoryHandle` handed out,
    /// so `svcMapSharedMemory` can recognise the region by **handle** rather
    /// than by guessing from its size.
    hid_shmem_handle: Option<u64>,
    /// The controller styles the guest said it supports
    /// (`SetSupportedNpadStyleSet`), and how it wants joy-cons held
    /// (`SetNpadJoyHoldType`). Both are read back by their `Get*` pairs, and a
    /// caller that reads back something it did not set decides the controller
    /// it wanted is not there.
    npad_style_set: u32,
    npad_joy_hold_type: u64,
    /// The amplitudes of the two rumble bands the guest last asked for, low
    /// then high. Switch rumble is two linear resonant actuators driven
    /// independently; the browser's Gamepad API exposes the same shape as
    /// `dual-rumble`'s strong and weak magnitudes.
    vibration: (f32, f32),
    /// `ssl` state: the interface revision the guest declared, how many TLS
    /// contexts it holds, and each context's options keyed by
    /// `(object_key, option)`. Options are read back, so they are stored
    /// rather than acknowledged and forgotten.
    ssl_interface_version: u32,
    ssl_contexts: u32,
    ssl_options: HashMap<(u64, u32), u32>,
    /// Monotonic sampling number for the hid shared-memory LIFO entries.
    sample_counter: u64,
    /// The touchscreen LIFO's own sampling number. Separate from the npad one
    /// because a reader compares it against the last value *it* saw from this
    /// LIFO, and pad and touch are published on independent schedules.
    touch_sample_counter: u64,
    /// How many touch slots the last publish filled, so the ones a shrinking
    /// contact count leaves behind can be cleared instead of lingering.
    touch_published: usize,
    /// The font `pl:u` serves as every shared font type, as a TrueType/OpenType
    /// file. Homebrew reads it out of pl's shared memory and hands it to
    /// FreeType, so an empty vector means no text renders at all.
    shared_font: Vec<u8>,
    /// Address the guest mapped pl's shared memory to, where the font was
    /// written; 0 until the guest calls `plInitialize`.
    pl_shmem_addr: u32,
    /// Per-`IAudioRenderer` session state (voice/sink/effect counts, revision)
    /// from its `OpenAudioRenderer` call, kept so `RequestUpdateAudioRenderer`
    /// can size its reply the same way the guest sized the buffer it passed
    /// in — `audrvUpdate` rejects a reply whose `mempools_sz`/`voices_sz`
    /// fields don't match what it computed from those same counts.
    audren_renderers: HashMap<u64, AudrenParams>,
    /// Every open `IAudioOut`, by session handle.
    audio_outs: HashMap<u64, AudioOut>,
    /// Interleaved 16-bit PCM the guest has handed to `audout` and the host
    /// has not played yet. Bounded: a host that never drains it (a headless
    /// test, a paused tab) must not be able to grow it without limit.
    audio_pcm: VecDeque<i16>,
    /// The rate and channel count the samples in `audio_pcm` are in, from the
    /// most recently opened device. `(0, 0)` until one is opened.
    audio_format: (u32, u32),
    /// The wall-clock time `time:u`/`time:s` reports, as POSIX seconds (UTC).
    /// `wasm32-unknown-unknown` has no OS clock, so this stays at the Unix
    /// epoch until the host calls [`Cpu::set_unix_time`].
    unix_time: i64,
    /// The nickname `acc` reports for the console's one user account, and
    /// the only part of that profile a guest can change: `IProfileEditor::
    /// Store` writes it back here and `IProfile::GetBase` reads it out again,
    /// so the pair agrees the way a real profile edit would.
    account_nickname: String,
    /// When that profile was last edited, as POSIX seconds — 0 until the guest
    /// stores one through `IProfileEditor`, which is what a profile nobody has
    /// touched reports.
    account_edited_at: i64,
    /// The program (title) id `pm:info` reports for this process. Defaults to
    /// the Album applet's, which is what homebrew launched from hbmenu runs
    /// as on real hardware; a loader that knows the real title id sets it with
    /// [`Cpu::set_program_id`].
    program_id: u64,
    /// The clock rate each module was last *set* to, by module index. A module
    /// with no entry runs at its default in `CLOCK_RATES_HZ`.
    clock_rates: HashMap<u32, u32>,
    /// State for the pseudo-random generator behind `csrng`, seeded lazily
    /// from the emulated clock. Zero means "not seeded yet".
    rng_state: u64,
    /// Every open `bsd` socket, by descriptor, and the socket options set on
    /// them keyed by `(descriptor, level, option)` — options are read back, so
    /// they are stored rather than acknowledged and forgotten.
    bsd_sockets: HashMap<i32, BsdSocket>,
    bsd_socket_options: HashMap<(i32, u32, u32), u32>,
    /// Monotonic descriptor allocator. Starts at 3, past the standard streams
    /// a guest's C library already holds.
    next_bsd_fd: i32,
    /// The `ApmPerformanceConfiguration` set for each performance mode
    /// (Normal, then Boost). Read back by `GetPerformanceConfiguration`, so
    /// they are stored rather than acknowledged and forgotten.
    apm_configuration: [u32; 2],
    /// The battery level `psm` reports, 0-100. There is no host battery API
    /// reachable from `wasm32-unknown-unknown` either, so this defaults to a
    /// full, charging battery until [`Cpu::set_battery`] says otherwise.
    battery_percent: u8,
    /// Whether `psm` reports a charger connected.
    battery_charging: bool,
    /// The emulated SD card `fsp-srv` serves.
    pub fs: crate::vfs::Vfs,
    /// The nvdrv driver and the GPU behind it.
    pub nv: crate::gpu::nvdrv::NvDrv,
    /// The app's window buffer queue: where rendered frames are handed to the
    /// display.
    pub display: crate::display::BufferQueue,
    /// Log every nvdrv IPC request to stderr (`TRACE_NV`).
    pub trace_nv: bool,
    /// Guest threads. Index 0 is the main thread; entries are appended by
    /// `svcCreateThread`. The running thread's registers are the `Cpu` fields,
    /// so its slot here is only up to date while another thread runs.
    threads: Vec<ThreadContext>,
    /// Which entry of `threads` is running.
    current_thread: usize,
    /// The address of an outstanding exclusive load (`LDXR`/`LDXP`), or
    /// `None` when the local monitor is clear.
    ///
    /// A `STXR` succeeds only against a monitor its own `LDXR` set, and a
    /// context switch clears it — which is what a real core does, and what
    /// makes an interrupted read-modify-write fail and be retried instead of
    /// silently losing the other thread's update.
    pub(crate) exclusive: Option<u32>,
    /// Instructions the running thread has executed since the scheduler last
    /// took the CPU away from it, against [`TIME_SLICE`].
    slice_used: u64,
    /// Set by a service call that answered "nothing is ready yet" and would
    /// have blocked on hardware. The reschedule cannot happen inside the
    /// handler — switching threads swaps the register file, and the syscall
    /// still has to write its result into the *caller's* X0 — so
    /// `svcSendSyncRequest` acts on this once the reply is in place.
    pub(crate) pending_yield: bool,
}

/// How many recently-executed instructions the fault trace shows.
pub const RECENT_LEN: usize = 64;

/// Instructions between display refreshes: the panel's 60 Hz against the
/// 1.02 GHz CPU one emulated instruction stands for.
///
/// A display refreshes whether or not anything drew, and until this was here
/// the only thing that fired the vsync event was the guest's own present —
/// which is a circle a title never gets into, because it waits for vsync
/// before it renders the frame that would have fired it. A present still
/// fires it too, so a guest that draws faster than the panel is not held to
/// this.
pub const VSYNC_PERIOD_CYCLES: u64 = 1_020_000_000 / 60;

/// How many instructions a thread runs before the scheduler takes the CPU
/// away from it.
///
/// Without this the only reschedule points were the blocking syscalls, so a
/// thread that runs a long stretch of arithmetic between two of them kept the
/// CPU for all of it. That is not a fairness nicety: an applet's audio thread
/// renders a whole buffer of samples per `AppendAudioOutBuffer`, and measured
/// at **99.9% of every instruction executed** — the Mii editor's own main loop
/// got the other 0.1%, which is why three system applets could boot, open a
/// layer, play their music and never reach a frame.
///
/// The number is a compromise against the cost of a switch, which copies the
/// whole register file including the 32 vector registers. Horizon's own tick
/// is 1 ms, and at the 1 µs-per-instruction scale `GetSystemTick` reports that
/// would be 1000 instructions — far more switching than the saving is worth
/// here, where a guest instruction is hundreds of host ones.
const TIME_SLICE: u64 = 20_000;

impl Default for Cpu {
    fn default() -> Self {
        Cpu::new()
    }
}

impl Cpu {
    pub fn new() -> Cpu {
        let mut cpu = Cpu {
            mem: Memory::new(),
            regs: [0; 31],
            sp: 0,
            pc: 0,
            nzcv: 0,
            vregs: [0; 32],
            out: Vec::new(),
            trace: Vec::new(),
            trace_enabled: false,
            trace_cap: 512 * 1024,
            halted: false,
            cycles: 0,
            recent: [(0, 0); RECENT_LEN],
            recent_len: 0,
            tpidr: 0,
            tpidr_rw: 0,
            next_object_id: 1,
            service_handles: HashMap::new(),
            next_handle: 0x1000,
            next_domain_object_id: 1,
            domain_objects: HashMap::new(),
            vi_ifaces: HashMap::new(),
            applet_messages: VecDeque::new(),
            sleep_lock_event: None,
            sleep_lock_acquired: false,
            shared_buffer: None,
            shared_buffer_slot: 0,
            applet_focus_announced: false,
            applet_is_application: true,
            applet_event: None,
            unimplemented_ipc: HashSet::new(),
            fabricated_objects: HashMap::new(),
            events: HashMap::new(),
            vsync_event: None,
            last_vsync_frame: 0,
            last_vsync_cycles: 0,
            fs_dirs: HashMap::new(),
            fs_files: HashMap::new(),
            data_archives: HashMap::new(),
            saves: HashMap::new(),
            fs_mount: HashMap::new(),
            fs_storage_archive: HashMap::new(),
            am_in_data: VecDeque::new(),
            am_storages: HashMap::new(),
            am_storage_of: HashMap::new(),
            romfs: None,
            touch_sample_counter: 0,
            touch_published: 0,
            hid_shmem_addr: 0,
            hid_shmem_handle: None,
            npad_style_set: 0,
            npad_joy_hold_type: 0,
            vibration: (0.0, 0.0),
            ssl_interface_version: 0,
            ssl_contexts: 0,
            ssl_options: HashMap::new(),
            sample_counter: 0,
            shared_font: Vec::new(),
            pl_shmem_addr: 0,
            audren_renderers: HashMap::new(),
            audio_outs: HashMap::new(),
            audio_pcm: VecDeque::new(),
            audio_format: (0, 0),
            unix_time: 0,
            account_nickname: String::from(DEFAULT_NICKNAME),
            account_edited_at: 0,
            apm_configuration: ipc::APM_DEFAULT_CONFIGURATION,
            program_id: ipc::DEFAULT_PROGRAM_ID,
            clock_rates: HashMap::new(),
            rng_state: 0,
            bsd_sockets: HashMap::new(),
            bsd_socket_options: HashMap::new(),
            next_bsd_fd: 3,
            battery_percent: 100,
            battery_charging: true,
            fs: crate::vfs::Vfs::new(),
            nv: crate::gpu::nvdrv::NvDrv::new(),
            display: crate::display::BufferQueue::new(),
            trace_nv: std::env::var("TRACE_NV").is_ok(),
            threads: Vec::new(),
            current_thread: 0,
            exclusive: None,
            slice_used: 0,
            pending_yield: false,
        };
        cpu.nv.gpu.trace = std::env::var("TRACE_GPU").is_ok();
        // The framebuffer and input registers are fixed hardware-mapped
        // regions: pre-map them so reads never fault and programs (or the
        // host) can touch them before writing.
        let _ = cpu
            .mem
            .map_zero(crate::FB_BASE, (crate::FB_WIDTH * crate::FB_HEIGHT * 4) as usize);
        let _ = cpu.mem.map_zero(crate::INPUT_ADDR, 4096);
        cpu
    }

    /// Map a host-provided runtime environment and point SP at a stack, the
    /// way the real loader does before jumping to a program's entry point.
    ///
    /// Without this, libnx-style crt0 writes to low memory (applet/env
    /// metadata, null-relative globals) fault on the unmapped zeropage and
    /// there is no stack to push to. The demo never touches the stack, so the
    /// unit tests keep SP at 0; only hosts that want to boot real homebrew
    /// should call this.
    pub fn bootstrap(&mut self) {
        // Present the low 2 GiB address space (everything below the old
        // 2 GiB NRO base) as lazily mapped: reads return zeros, writes
        // allocate a page on first touch, so nothing is reserved up front.
        // This lets libnx-style code read heap/init globals without faulting
        // even when a baked-in pointer is stale.
        self.mem.soft_map_zero(0, 0x8000_0000);
        // 1 MiB full-descending stack; SP starts at the top.
        let _ = self.mem.map_zero((STACK_TOP - STACK_SIZE) as u32, STACK_SIZE as usize);
        self.sp = STACK_TOP;
        // libnx reads TPIDR_EL0 expecting the loader (HBL/kernel) to have set
        // the thread-local-storage base. Point it at a writable region clear of
        // both the heap (`svcSetHeapSize` hands out 0x30000000) and the stack
        // (`STACK_TOP`, now 0x28100000) — if TPIDR overlaps either, the app's
        // IPC code writes its CMIF request over the heap's first chunk header
        // (and malloc stomps the TLS), corrupting the allocator.
        //
        // 0x0FF00000 used to work here, sitting just under where the stack
        // used to be. But a big enough Mesa/Nouveau GPU-buffer allocation —
        // nouveau reserves its own address range by scanning for free space
        // with `svcQueryMemory` rather than going through the regular heap,
        // and its search isn't guaranteed to stop at a single mapped page in
        // the middle of an otherwise-huge free run — grew past it and
        // `memset()`-zeroed straight over the `ThreadVars` magic, so the next
        // `malloc()` on that thread failed `__syscall_getreent`'s `BadReent`
        // check and the app aborted. Up here, past the stack and well clear of
        // everything the guest's own allocators have been observed to reach,
        // is safe.
        self.tpidr = 0x1FE0_0000;
        // A return-address trampoline: the loader enters homebrew's `main`
        // directly, so LR is 0 and any early return would branch to NULL.
        // Point LR at a stub that calls ExitProcess (svc 0x07), so main's
        // return surfaces as a clean exit code instead of a NULL jump.
        let _ = self.mem.map_zero(SELF_RETURN_TRAMPOLINE, 0x10);
        self.mem.write_u32(SELF_RETURN_TRAMPOLINE, 0xD400_00E1).ok(); // svc #7
        self.mem.write_u32(SELF_RETURN_TRAMPOLINE + 4, 0x1400_0000).ok(); // b .
        // The same for a thread's entry point: returning from it is
        // `svcExitThread` (svc 0x0A), not a process exit.
        let _ = self.mem.map_zero(THREAD_EXIT_TRAMPOLINE, 0x10);
        self.mem.write_u32(THREAD_EXIT_TRAMPOLINE, 0xD400_0141).ok(); // svc #0xa
        self.mem.write_u32(THREAD_EXIT_TRAMPOLINE + 4, 0x1400_0000).ok(); // b .
    }

    // ---- guest threads ----
    //
    // Cooperative: a thread runs until it makes a blocking syscall (sleep, wait,
    // lock, condvar) or exits, and only then does another get the CPU. Real
    // Horizon preempts, but every libnx synchronization primitive re-checks its
    // predicate in a loop, so co-operative switching makes the same handshakes
    // complete — which is all a stub scheduler needs to let `thrd_create`'s
    // "has the child started?" wait finish.

    /// The main thread's slot, created on demand so a single-threaded program
    /// costs nothing.
    fn ensure_main_thread(&mut self) {
        if self.threads.is_empty() {
            self.threads.push(ThreadContext {
                handle: MAIN_THREAD_HANDLE,
                state: ThreadState::Runnable,
                paused: false,
                regs: [0; 31],
                sp: 0,
                pc: 0,
                nzcv: 0,
                vregs: [0; 32],
                tpidr: self.tpidr,
                tpidr_rw: self.tpidr_rw,
            });
            self.current_thread = 0;
        }
    }

    /// Create a thread the way `svcCreateThread` does: its own TLS block (with
    /// the libnx `ThreadVars` the guest reads through TPIDRRO_EL0), the given
    /// stack and entry point, and the argument in x0. Returns its handle.
    pub(super) fn create_thread(&mut self, entry: u32, arg: u64, stack_top: u64) -> u64 {
        self.ensure_main_thread();
        let handle = self.alloc_handle();
        let index = self.threads.len() as u32;
        let tls = THREAD_TLS_BASE + index * THREAD_TLS_STRIDE;
        let _ = self.mem.map_zero(tls, THREAD_TLS_STRIDE as usize);
        // ThreadVars at TLS+0x1E0, same layout the main thread gets in
        // `boot_homebrew`: magic, handle, thread pointer, reent, tls_tp.
        const TV_MAGIC: u32 = 0x2154_5624; // "!TV$"
        let reent = tls + 0x400;
        let _ = self.mem.write_u32(tls + 0x1E0, TV_MAGIC);
        let _ = self.mem.write_u32(tls + 0x1E4, handle as u32);
        let _ = self.mem.write_u32(tls + 0x1E8, 0);
        let _ = self.mem.write_u32(tls + 0x1F0, reent);
        let _ = self.mem.write_u32(tls + 0x1F8, tls);

        let mut regs = [0u64; 31];
        regs[0] = arg;
        regs[30] = THREAD_EXIT_TRAMPOLINE as u64;
        self.threads.push(ThreadContext {
            handle,
            state: ThreadState::Created,
            paused: false,
            regs,
            sp: stack_top,
            pc: entry,
            nzcv: 0,
            vregs: [0; 32],
            tpidr: u64::from(tls),
            tpidr_rw: 0,
        });
        handle
    }

    /// Mark a created thread runnable (`svcStartThread`).
    pub(super) fn start_thread(&mut self, handle: u64) -> bool {
        for thread in &mut self.threads {
            if thread.handle == handle && thread.state == ThreadState::Created {
                thread.state = ThreadState::Runnable;
                return true;
            }
        }
        false
    }

    /// `svcSetThreadActivity`: take a thread out of the scheduler's rotation,
    /// or put it back. `Ok(())` on a real change; `Err(())` when the thread is
    /// already in the requested state, which Horizon reports rather than
    /// treating as a no-op.
    pub(super) fn set_thread_paused(&mut self, handle: u64, paused: bool) -> Option<bool> {
        let thread = self.threads.iter_mut().find(|t| t.handle == handle)?;
        if thread.paused == paused {
            return Some(false);
        }
        thread.paused = paused;
        Some(true)
    }

    /// Fill the 0x320-byte `ThreadContext` `svcGetThreadContext3` hands back:
    /// x0..x28, fp, lr, sp, pc, pstate, the vector registers, fpcr/fpsr and
    /// the thread pointer. IL2CPP's garbage collector suspends every thread
    /// and reads this to find the roots living in their registers, so the
    /// register file has to be the real one — the running thread's live, a
    /// switched-out thread's as saved when it last gave up the CPU.
    pub(super) fn write_thread_context(&mut self, out: u32, handle: u64) -> bool {
        self.ensure_main_thread();
        let Some(index) = self.threads.iter().position(|t| t.handle == handle) else {
            return false;
        };
        let (regs, sp, pc, nzcv, vregs, tpidr) = if index == self.current_thread {
            (self.regs, self.sp, self.pc, self.nzcv, self.vregs, self.tpidr)
        } else {
            let t = &self.threads[index];
            (t.regs, t.sp, t.pc, t.nzcv, t.vregs, t.tpidr)
        };
        let put64 = |cpu: &mut Self, off: u32, v: u64| {
            let _ = cpu.mem.write_u64(out.wrapping_add(off), v);
        };
        for (i, &r) in regs.iter().take(29).enumerate() {
            put64(self, i as u32 * 8, r);
        }
        put64(self, 0xE8, regs[29]); // fp
        put64(self, 0xF0, regs[30]); // lr
        put64(self, 0xF8, sp);
        put64(self, 0x100, u64::from(pc));
        let _ = self.mem.write_u32(out.wrapping_add(0x108), nzcv);
        let _ = self.mem.write_u32(out.wrapping_add(0x10C), 0);
        for (i, &v) in vregs.iter().enumerate() {
            let at = 0x110 + i as u32 * 16;
            put64(self, at, v as u64);
            put64(self, at + 8, (v >> 64) as u64);
        }
        // No FPCR/FPSR is modelled: rounding mode and the accrued exception
        // flags are both at their reset value.
        let _ = self.mem.write_u32(out.wrapping_add(0x310), 0);
        let _ = self.mem.write_u32(out.wrapping_add(0x314), 0);
        put64(self, 0x318, tpidr);
        true
    }

    /// Whether any thread other than the running one could run.
    pub(super) fn has_other_runnable(&self) -> bool {
        self.threads
            .iter()
            .enumerate()
            .any(|(i, t)| i != self.current_thread && t.state == ThreadState::Runnable && !t.paused)
    }

    /// End the running thread (`svcExitThread`, or a return through the exit
    /// trampoline) and switch away. The process only ends when the main thread
    /// exits, matching Horizon.
    pub(super) fn exit_thread(&mut self) {
        self.ensure_main_thread();
        if self.current_thread == 0 {
            self.halted = true;
            return;
        }
        self.threads[self.current_thread].state = ThreadState::Finished;
        if !self.switch_to_next_runnable() {
            // Nothing else can run: fall back to the main thread, which is
            // presumably waiting on this one.
            self.threads[0].state = ThreadState::Runnable;
            self.switch_to_next_runnable();
        }
    }

    /// Give up the CPU at a blocking syscall. Does nothing when this is the
    /// only runnable thread, so single-threaded programs behave exactly as
    /// before.
    pub(super) fn yield_thread(&mut self) {
        if self.threads.len() < 2 || !self.has_other_runnable() {
            return;
        }
        self.switch_to_next_runnable();
    }

    // ---- mutexes and condition variables ----
    //
    // Horizon keeps the lock word in guest memory and only asks the kernel to
    // arbitrate when a thread has to block: the word holds the owning thread's
    // handle, plus MUTEX_HAS_LISTENERS when someone is queued. libnx re-reads
    // that word after every arbitration, so ownership has to actually move —
    // returning success from the stubs left hbmenu's worker spinning on a lock
    // its main thread held.

    /// `svcArbitrateLock(owner, mutex_addr, self)`: block until the owner
    /// releases, unless the word has already changed under us.
    pub(super) fn arbitrate_lock(&mut self, owner: u32, addr: u32, _self_handle: u32) {
        self.ensure_main_thread();
        let word = self.mem.read_u32(addr).unwrap_or(0);
        if word & !MUTEX_HAS_LISTENERS != owner || owner == 0 {
            return; // stale request; the guest re-reads the word and retries
        }
        self.threads[self.current_thread].state = ThreadState::WaitMutex(addr);
        self.reschedule();
    }

    /// `svcArbitrateUnlock(mutex_addr)`: hand the mutex to a waiter, or clear
    /// it when there is none.
    pub(super) fn arbitrate_unlock(&mut self, addr: u32) {
        self.ensure_main_thread();
        let waiters: Vec<usize> = (0..self.threads.len())
            .filter(|&i| self.threads[i].state == ThreadState::WaitMutex(addr))
            .collect();
        match waiters.first() {
            Some(&next) => {
                let mut handle = self.threads[next].handle as u32;
                if waiters.len() > 1 {
                    handle |= MUTEX_HAS_LISTENERS;
                }
                let _ = self.mem.write_u32(addr, handle);
                self.threads[next].state = ThreadState::Runnable;
            }
            None => {
                let _ = self.mem.write_u32(addr, 0);
            }
        }
    }

    /// `svcWaitProcessWideKeyAtomic(mutex_addr, key, self, timeout)`: release
    /// the mutex and block on the condition variable.
    pub(super) fn wait_process_wide_key(
        &mut self,
        mutex: u32,
        key: u32,
        _self_handle: u32,
        timeout: i64,
    ) {
        self.ensure_main_thread();
        self.arbitrate_unlock(mutex);
        // A negative timeout waits forever; a positive one is nanoseconds, and
        // has to expire. A thread that asked to be woken in 100ms and never was
        // is a thread that does its work on a timer and never does it again.
        let deadline = (timeout > 0).then(|| {
            let cycles = (timeout as u128) * u128::from(crate::cpu::ipc::CLOCK_RATES_HZ[0]) / 1_000_000_000;
            self.cycles.wrapping_add(cycles as u64)
        });
        self.threads[self.current_thread].state = ThreadState::WaitKey { key, mutex, deadline };
        self.reschedule();
    }

    /// Wake every timed condition-variable wait whose deadline has passed.
    /// Horizon reports the timeout to the waiter, and `nn::os` answers one by
    /// re-checking its predicate, so waking is the whole of it.
    pub(super) fn expire_timed_waits(&mut self) {
        let now = self.cycles;
        for thread in &mut self.threads {
            if let ThreadState::WaitKey { deadline: Some(at), .. } = thread.state {
                if now >= at {
                    thread.state = ThreadState::Runnable;
                }
            }
        }
    }

    /// `svcSignalProcessWideKey(key, count)`: wake up to `count` waiters
    /// (`count` < 0 wakes all of them). A woken thread holds the mutex again,
    /// or queues for it if someone else took it meanwhile.
    pub(super) fn signal_process_wide_key(&mut self, key: u32, count: i32) {
        self.ensure_main_thread();
        let mut woken = 0;
        for i in 0..self.threads.len() {
            if count >= 0 && woken >= count {
                break;
            }
            if let ThreadState::WaitKey { key: waiting, mutex, .. } = self.threads[i].state {
                if waiting != key {
                    continue;
                }
                let handle = self.threads[i].handle as u32;
                let owner = self.mem.read_u32(mutex).unwrap_or(0);
                if owner == 0 {
                    let _ = self.mem.write_u32(mutex, handle);
                    self.threads[i].state = ThreadState::Runnable;
                } else {
                    // Someone holds it: queue up, and mark the word so the
                    // owner arbitrates its unlock instead of just clearing it.
                    let _ = self.mem.write_u32(mutex, owner | MUTEX_HAS_LISTENERS);
                    self.threads[i].state = ThreadState::WaitMutex(mutex);
                }
                woken += 1;
            }
        }
    }

    /// Switch away from the running thread after it blocked. If nothing can
    /// run, everything blocked is woken: guests re-check their predicates in a
    /// loop, so a spurious wake degrades to the old spin rather than a hang.
    fn reschedule(&mut self) {
        if self.switch_to_next_runnable() {
            return;
        }
        for thread in &mut self.threads {
            if matches!(
                thread.state,
                ThreadState::WaitMutex(_) | ThreadState::WaitKey { .. }
            ) {
                thread.state = ThreadState::Runnable;
            }
        }
        self.switch_to_next_runnable();
    }

    /// Round-robin to the next runnable thread. Returns false if there is none
    /// (in which case the running thread keeps going).
    fn switch_to_next_runnable(&mut self) -> bool {
        let count = self.threads.len();
        let start = self.current_thread;
        for step in 1..=count {
            let candidate = (start + step) % count;
            if candidate == start {
                continue;
            }
            if self.threads[candidate].state == ThreadState::Runnable
                && !self.threads[candidate].paused
            {
                self.save_context(start);
                self.load_context(candidate);
                return true;
            }
        }
        false
    }

    fn save_context(&mut self, index: usize) {
        let thread = &mut self.threads[index];
        thread.regs = self.regs;
        thread.sp = self.sp;
        thread.pc = self.pc;
        thread.nzcv = self.nzcv;
        thread.vregs = self.vregs;
        thread.tpidr = self.tpidr;
        thread.tpidr_rw = self.tpidr_rw;
    }

    fn load_context(&mut self, index: usize) {
        self.slice_used = 0;
        // Taking the CPU away from a thread clears the local monitor, so an
        // exclusive pair the switch landed inside fails and is retried rather
        // than completing across the other thread's writes.
        self.exclusive = None;
        let thread = self.threads[index].clone();
        self.regs = thread.regs;
        self.sp = thread.sp;
        self.pc = thread.pc;
        self.nzcv = thread.nzcv;
        self.vregs = thread.vregs;
        self.tpidr = thread.tpidr;
        self.tpidr_rw = thread.tpidr_rw;
        self.current_thread = index;
    }

    /// Handle of the thread that is running, as the guest knows it.
    pub fn current_thread_handle(&self) -> u64 {
        self.threads.get(self.current_thread).map_or(MAIN_THREAD_HANDLE, |t| t.handle)
    }

    /// How many threads the guest has created (including the main thread).
    pub fn thread_count(&self) -> usize {
        self.threads.len().max(1)
    }

    /// Boot a homebrew NRO the way HBL does: load the image, let the crt0's
    /// relocation pass run up to the point it calls `main`, run the `.init_array`
    /// (C++ static constructors) and set up the main thread's `ThreadVars`
    /// (newlib reentrancy) that the skipped `__libnx_init` would normally
    /// provide, then leave the CPU ready to enter `main`.
    ///
    /// The libnx "HOME BREW" crt0 runs the relocation pass itself and then
    /// jumps to main; when it omits the `__libnx_init` step, every std::string
    /// global is left empty and NX-Shell's `FS::GetDirList` resolves its SD
    /// path to "" and exits. Running the constructors fixes that.
    pub fn boot_homebrew(&mut self, data: &[u8]) -> Result<crate::nro::LoadedNro> {
        self.mem.clear_readonly();
        let loaded = crate::nro::load_nro(&mut self.mem, data)?;
        // Present the NRO on the SD card at the path the environment block
        // advertises as argv[0]: libnx's `romfsMountSelf` re-opens the running
        // NRO through the filesystem to read the RomFS appended to it, which
        // is where homebrew keeps its assets.
        self.fs.write_file(crate::nro::HOMEBREW_NRO_PATH, data.to_vec());
        self.out.clear();
        self.trace.clear();
        self.halted = false;
        self.trace_enabled = false;
        for i in 0..=30u8 {
            self.set_reg(i, 0);
        }
        self.set_reg(0, loaded.env_addr as u64);
        self.set_reg(1, if loaded.env_addr != 0 { u64::MAX } else { 1 });
        self.set_reg(30, SELF_RETURN_TRAMPOLINE as u64);

        let init = crate::nro::init_array_entries(data);
        if !init.is_empty() && loaded.env_addr != 0 {
            // The crt0 calls main with `bl` at entry+0xc0 (libnx switch_crt0
            // layout); by then BSS is zeroed and RELR relocations are applied.
            let main_call = loaded.entry.wrapping_add(0xc0);
            let main_insn = self.mem.fetch(main_call).ok();
            let is_bl = matches!(main_insn, Some(i) if (i & 0xFC00_0000) == 0x9400_0000);
            if is_bl {
                self.set_pc(loaded.entry);
                for _ in 0..5_000_000u64 {
                    if self.halted || self.get_pc() == main_call {
                        break;
                    }
                    self.step()?;
                }
                // ThreadVars at TLS+0x1E0: magic, handle, thread_ptr, _REENT,
                // tls_tp. The _REENT can be zeroed; newlib's malloc lazily
                // initializes it.
                const TV_MAGIC: u32 = 0x2154_5624; // "!TV$"
                const REENT_ADDR: u32 = 0x1FF1_0000;
                let tls = self.tls_base();
                let _ = self.mem.map_zero(REENT_ADDR, 0x400);
                let _ = self.mem.write_u32(tls + 0x1E0, TV_MAGIC);
                let _ = self.mem.write_u32(tls + 0x1E4, 0x100);
                let _ = self.mem.write_u32(tls + 0x1E8, 0);
                let _ = self.mem.write_u32(tls + 0x1F0, REENT_ADDR);
                let _ = self.mem.write_u32(tls + 0x1F8, tls);
                // Run the constructors; each returns via x30.
                const SENTINEL: u32 = 0x1FF0_0000;
                for &entry in &init {
                    if self.halted {
                        break;
                    }
                    for i in 0..=29u8 {
                        self.set_reg(i, 0);
                    }
                    self.set_reg(30, SENTINEL as u64);
                    self.set_pc(entry);
                    for _ in 0..20_000_000u64 {
                        if self.halted || self.get_pc() == SENTINEL {
                            break;
                        }
                        self.step()?;
                    }
                }
                // The constructors clobber the entry registers; restore them
                // and resume at the crt0's call so it is entered with the normal
                // calling convention (x30 = the crt0's return path).
                //
                // That call is libnx's `__libnx_init(ctx, main_thread,
                // saved_lr)`, and `saved_lr` is the loader's return address:
                // `envSetup` keeps it as the exit function pointer, and
                // `__nx_exit` branches straight to it. Leaving x2 at 0 made
                // every clean exit jump to NULL — NX-Shell looked like it
                // crashed when it was only returning from main.
                for i in 0..=30u8 {
                    self.set_reg(i, 0);
                }
                self.set_reg(0, loaded.env_addr as u64);
                self.set_reg(1, if loaded.env_addr != 0 { u64::MAX } else { 1 });
                self.set_reg(2, SELF_RETURN_TRAMPOLINE as u64);
                self.set_reg(30, SELF_RETURN_TRAMPOLINE as u64);
                self.set_pc(main_call);
                return Ok(loaded);
            }
        }
        self.set_pc(loaded.entry);
        Ok(loaded)
    }

    /// Boot a retail title's full module set (`rtld`, `main`, `subsdk*`,
    /// `sdk`) the way Nintendo's process creation does: load every module
    /// into one shared address space, back to back, and hand off to
    /// `rtld`'s entry point — *not* `main`'s.
    ///
    /// `rtld` is Nintendo's own runtime linker; its job is to process every
    /// other module's relocations (base-relative fixups, and resolving
    /// cross-module calls — e.g. `main` importing something `sdk` exports)
    /// before jumping into `main`'s own crt0. Jumping straight to `main`
    /// (this emulator's first attempt) leaves its GOT full of unrelocated
    /// placeholder addresses: confirmed against a real title, whose `main`
    /// crt0 runs cleanly right up to its first PLT-style indirect call,
    /// which lands on exactly such a placeholder.
    ///
    /// `modules` must be in Nintendo's required load order: `rtld`, `main`,
    /// `subsdk0..subsdk9`, `sdk` — whichever of those a title actually has.
    /// Actually running a retail title past `rtld`'s own work needs the
    /// Horizon service surface a full SDK program expects, which this
    /// emulator does not have yet; this gets it as far as that surface, the
    /// same "boot as far as it goes" spirit as `boot_homebrew`.
    pub fn boot_retail_program(&mut self, modules: &[(&str, &[u8])]) -> Result<Vec<crate::nso::LoadedNso>> {
        self.out.clear();
        self.trace.clear();
        self.halted = false;
        self.trace_enabled = false;
        self.mem.clear_readonly();
        for i in 0..=30u8 {
            self.set_reg(i, 0);
        }
        // Horizon's process entry ABI, which `rtld` reads literally at its
        // first two instructions (`cmp x0, #0` / `mov w19, w1`): X0 is the
        // launch argument — 0 for a normal process launch, non-zero only for
        // the homebrew loader's config block — and **X1 is the main thread's
        // handle**. `nnSdk` stores that handle in the main
        // `nn::os::ThreadType` (+0x1B0) and every `SdkMutex` compares its
        // lock word against it; leaving X1 at 0 makes an *unlocked* mutex
        // (lock word 0) compare equal to "owned by the current thread", so
        // `nn::os::SdkMutexType::Lock` fires its recursive-lock assertion and
        // `nn::oe::Initialize` aborts before the SDK ever reaches a service.
        self.set_reg(1, MAIN_THREAD_HANDLE);
        self.set_reg(30, SELF_RETURN_TRAMPOLINE as u64);

        // Real inter-module gaps are whatever the kernel's ASLR/layout
        // picked; page-aligned and back-to-back is a reasonable stand-in —
        // each module is fully self-contained PC-relative code, so the only
        // thing that matters is that nothing overlaps.
        const MODULE_ALIGN: u32 = 0x1000;
        let mut base = crate::nso::NSO_BASE;
        let mut loaded = Vec::with_capacity(modules.len());
        for (name, data) in modules {
            let module = crate::nso::load_nso(&mut self.mem, data, base).map_err(|e| {
                Error::Cpu(format!("loading module {:?} at {:#x}: {}", name, base, e))
            })?;
            let image_end = module
                .data
                .mem_addr
                .wrapping_add(module.data.file_size)
                .wrapping_add(module.bss_size);
            // Where each module actually landed. `rtld` does not take the
            // layout on trust — it finds modules itself, scanning with
            // `svcQueryMemory` for R-X regions carrying `MOD0` — so the base
            // it relocates against is one it worked out, and a fault
            // afterwards is unreadable without knowing what it was supposed
            // to have found.
            self.diagnostic(&format!(
                "[loader] {} at {:#010x}: text {:#010x}..{:#010x}, rodata {:#010x}..{:#010x}, data {:#010x}..{:#010x}, bss {:#010x}..{:#010x}",
                name,
                module.base,
                module.text.mem_addr,
                module.text.mem_addr.wrapping_add(module.text.file_size),
                module.ro.mem_addr,
                module.ro.mem_addr.wrapping_add(module.ro.file_size),
                module.data.mem_addr,
                module.data.mem_addr.wrapping_add(module.data.file_size),
                module.data.mem_addr.wrapping_add(module.data.file_size),
                image_end,
            ));
            base = image_end.wrapping_add(MODULE_ALIGN - 1) & !(MODULE_ALIGN - 1);
            loaded.push(module);
        }
        let entry = loaded
            .first()
            .ok_or_else(|| Error::Cpu("no modules to boot".into()))?
            .entry;
        self.set_pc(entry);
        self.seed_applet_launch_arguments();
        Ok(loaded)
    }

    /// Queue what a library applet's caller would have pushed before starting
    /// it, so `PopInData` has something to hand over.
    ///
    /// Every caller pushes `LibAppletCommonArguments` first — the 0x20-byte
    /// block naming the interface version the two sides agreed on and the
    /// theme to draw in. Running a library applet directly, as this emulator
    /// does, there is nobody to push it, and an applet that cannot read its
    /// own arguments aborts before it draws anything.
    ///
    /// Whatever the applet pops *after* that is its own launch struct, which
    /// only a real caller could fill in; nothing is queued for it.
    fn seed_applet_launch_arguments(&mut self) {
        self.am_in_data.clear();
        if !crate::cpu::ipc::is_library_applet(self.program_id) {
            return;
        }
        const COMMON_ARGS_VERSION: u32 = 1;
        const COMMON_ARGS_SIZE: u32 = 0x20;
        let mut args = Vec::with_capacity(COMMON_ARGS_SIZE as usize);
        args.extend_from_slice(&COMMON_ARGS_VERSION.to_le_bytes());
        args.extend_from_slice(&COMMON_ARGS_SIZE.to_le_bytes());
        // LaVersion: the applet-interface revision the caller speaks. Each
        // applet numbers its own, and a caller that claims one the applet
        // does not know is refused, so this is the applet's own.
        args.extend_from_slice(&crate::cpu::ipc::applet_interface_version(self.program_id).to_le_bytes());
        // ExpectedThemeColor: 0 is the basic white theme.
        args.extend_from_slice(&0u32.to_le_bytes());
        // PlayStartupSound, then padding out to the tick field.
        args.resize(0x18, 0);
        // The tick the caller started the applet at. Nothing here measures
        // elapsed time against it.
        args.extend_from_slice(&0u64.to_le_bytes());
        self.am_in_data.push_back(args);
        // Then the applet's own launch struct — see
        // [`crate::cpu::ipc::applet_launch_argument`]. Refusing the pop
        // instead is what a real applet treats as a launch it cannot honour,
        // and it aborts.
        let argument = crate::cpu::ipc::applet_launch_argument(self.program_id);
        self.am_in_data.push_back(argument);
    }

    /// Set the decrypted RomFS bytes `OpenDataStorageByCurrentProcess`
    /// serves. The caller (the NCA-decryption loader) supplies these — `Cpu`
    /// has no key material and doesn't know how to get from an NCA to a
    /// RomFS image itself.
    ///
    /// Only for a RomFS small enough to hold: see [`Cpu::set_romfs_source`]
    /// for the form a real title's uses.
    pub fn set_romfs(&mut self, data: Vec<u8>) {
        self.romfs = Some(Box::new(crate::source::MemSource(data)));
    }

    /// Register a system data archive under its data id, for
    /// `OpenDataStorageByDataId` to serve.
    ///
    /// These live on a real console's NAND as separate Data NCAs, one per
    /// data id; an applet mounts one to get at assets it ships apart from its
    /// own RomFS. Nothing here has a NAND, so the host registers whichever it
    /// has and a request for any other is reported missing rather than
    /// answered with an empty archive.
    pub fn add_data_archive(&mut self, data_id: u64, src: Box<dyn crate::source::ByteSource>) {
        self.data_archives.insert(data_id, src);
    }

    /// The save data filed under `id`, creating it if this is the first time
    /// anything has asked. A console formats a save on first open too.
    pub fn save_data_mut(&mut self, id: u64) -> &mut crate::vfs::Vfs {
        self.saves.entry(id).or_insert_with(crate::vfs::Vfs::empty)
    }

    /// The save data filed under `id`, if it exists.
    pub fn save_data(&self, id: u64) -> Option<&crate::vfs::Vfs> {
        self.saves.get(&id)
    }

    /// Every save id that has been opened, for a host that persists them.
    pub fn save_ids(&self) -> Vec<u64> {
        self.saves.keys().copied().collect()
    }

    /// The storage an `fsp-srv` object addresses.
    pub(super) fn vfs_for(&mut self, mount: Option<u64>) -> &mut crate::vfs::Vfs {
        match mount {
            Some(id) => self.saves.entry(id).or_insert_with(crate::vfs::Vfs::empty),
            None => &mut self.fs,
        }
    }

    /// Which storage the `fsp-srv` object under `key` addresses.
    pub(super) fn mount_of(&self, key: u64) -> Option<u64> {
        self.fs_mount.get(&key).copied()
    }

    /// Record that the object under `key` addresses `mount`.
    pub(super) fn set_mount(&mut self, key: u64, mount: Option<u64>) {
        match mount {
            Some(id) => {
                self.fs_mount.insert(key, id);
            }
            None => {
                self.fs_mount.remove(&key);
            }
        }
    }

    /// Same, backed by a [`ByteSource`](crate::source::ByteSource) that
    /// decrypts on demand — [`crate::nca::Nca::romfs_source`] over the
    /// container the title was launched from.
    pub fn set_romfs_source(&mut self, src: Box<dyn crate::source::ByteSource>) {
        self.romfs = Some(src);
    }

    // ---- register access ----

    #[inline]
    pub fn get_pc(&self) -> u32 {
        self.pc
    }

    #[inline]
    pub fn sp(&self) -> u64 {
        self.sp
    }

    pub fn set_pc(&mut self, pc: u32) {
        self.pc = pc;
    }

    /// Allocate a handle and record it as an event. Callers are the services
    /// that hand events out (`am`'s applet-message and GPU-error events,
    /// `vi`'s display vsync, `nvdrv`'s QueryEvent), and the handle has to
    /// reach the guest as a **copy** handle — see [`Cpu::write_ipc_reply`].
    pub(crate) fn alloc_event(&mut self, name: &'static str, auto_clear: bool) -> u64 {
        let handle = self.alloc_handle();
        self.events.insert(handle, Event { name, signaled: false, auto_clear });
        if std::env::var("TRACE_WAIT").is_ok() {
            eprintln!("[event] {name} = {handle:#x} auto_clear={auto_clear}");
        }
        handle
    }

    /// What an event handle is for, for diagnostics.
    pub(crate) fn event_name(&self, handle: u64) -> Option<&'static str> {
        self.events.get(&handle).map(|event| event.name)
    }

    /// Queue an applet message and wake whatever is polling for one.
    pub(super) fn queue_applet_message(&mut self, message: AppletMessage) {
        self.applet_messages.push_back(message as u32);
        if let Some(handle) = self.applet_event {
            self.signal_event(handle);
        }
    }

    /// Note which kind of proxy the process opened, so its focus transition is
    /// the one its own applet framework is waiting for.
    pub(super) fn set_applet_is_application(&mut self, is_application: bool) {
        self.applet_is_application = is_application;
    }

    /// The next AM message for the running applet, or `None` when the queue is
    /// empty. The startup focus transition comes first and exactly once.
    pub(super) fn next_applet_message(&mut self) -> Option<u32> {
        if !self.applet_focus_announced {
            self.applet_focus_announced = true;
            return Some(if self.applet_is_application {
                AppletMessage::FocusStateChanged as u32
            } else {
                AppletMessage::ChangeIntoForeground as u32
            });
        }
        self.applet_messages.pop_front()
    }

    /// Whether AM has a message waiting, which is what the applet event says.
    pub(super) fn has_applet_message(&self) -> bool {
        !self.applet_focus_announced || !self.applet_messages.is_empty()
    }

    /// Fire an event.
    pub fn signal_event(&mut self, handle: u64) {
        if let Some(event) = self.events.get_mut(&handle) {
            event.signaled = true;
        }
    }

    /// Whether `handle` names an event that has fired. `None` means the handle
    /// is not modelled as an event at all.
    pub(crate) fn event_signaled(&self, handle: u64) -> Option<bool> {
        self.events.get(&handle).map(|event| event.signaled)
    }

    /// Consume an auto-clear event's signal after a wait has reported it.
    pub(crate) fn consume_event(&mut self, handle: u64) {
        if let Some(event) = self.events.get_mut(&handle) {
            if event.auto_clear {
                event.signaled = false;
            }
        }
    }

    /// Clear an event's signal.
    pub(crate) fn clear_event(&mut self, handle: u64) {
        if let Some(event) = self.events.get_mut(&handle) {
            event.signaled = false;
        }
    }

    /// `svcResetSignal`: clear a signalled event, reporting whether it *was*
    /// signalled. An event this emulator does not model counts as signalled,
    /// which is the same answer a wait on one gets.
    pub(crate) fn reset_signal(&mut self, handle: u64) -> bool {
        match self.events.get_mut(&handle) {
            Some(event) => std::mem::replace(&mut event.signaled, false),
            None => true,
        }
    }

    /// Debug/test counterpart to [`Cpu::service_handles_snapshot`]: bind a
    /// handle to a service name without going through `sm`'s GetService, so a
    /// test can drive one service's IPC surface directly.
    pub fn register_service_handle(&mut self, handle: u64, name: &str) {
        self.record_handle(handle, name);
    }

    /// Debug: what interface a domain object id on `handle` names, or `None`
    /// once it has been closed. The counterpart to
    /// [`Cpu::service_handles_snapshot`] for the objects living on a session
    /// rather than the sessions themselves.
    pub fn domain_interface_name(&self, handle: u64, object_id: u32) -> Option<String> {
        self.domain_interface(handle, object_id).map(|s| s.to_owned())
    }

    /// Debug: dump the fake-handle -> service-name map.
    pub fn service_handles_snapshot(&self) -> Vec<(u64, String)> {
        let mut v: Vec<(u64, String)> = self
            .service_handles
            .iter()
            .map(|(&h, s)| (h, s.clone()))
            .collect();
        v.sort();
        v
    }

    /// Read X0..=X30 (X31 reads as zero / is the stack pointer).
    #[inline(always)]
    pub fn read_x(&self, idx: u8) -> u64 {
        match idx {
            0..=30 => self.regs[idx as usize],
            31 => self.sp,
            _ => 0,
        }
    }

    pub fn read_reg(&self, idx: u8) -> u64 {
        self.read_x(idx)
    }

    /// Read the 128-bit SIMD&FP register Qn.
    pub fn read_vreg(&self, idx: u8) -> u128 {
        self.vregs[idx as usize]
    }

    /// Base of the libnx TLS (thread-local storage) region.
    pub fn tls_base(&self) -> u32 {
        self.tpidr as u32
    }

    /// Write the 128-bit SIMD&FP register Qn.
    pub fn set_vreg(&mut self, idx: u8, val: u128) {
        self.vregs[idx as usize] = val;
    }

    /// Read X0..=X30 where X31 is ZR (always zero).
    #[inline(always)]
    fn read_zr(&self, idx: u8) -> u64 {
        if idx == 31 { 0 } else { self.regs[idx as usize] }
    }

    #[inline(always)]
    fn write_zr(&mut self, idx: u8, val: u64) {
        if idx != 31 {
            self.regs[idx as usize] = val;
        }
    }

    /// Write X0..=X30 where X31 is SP.
    #[inline]
    fn write_x(&mut self, idx: u8, val: u64) {
        match idx {
            0..=30 => self.regs[idx as usize] = val,
            31 => self.sp = val,
            _ => {}
        }
    }

    pub fn set_reg(&mut self, idx: u8, val: u64) {
        self.write_zr(idx, val);
    }

    pub fn read_u32_reg(&self, idx: u8) -> u32 {
        self.read_zr(idx) as u32
    }

    pub fn set_pc_and_sp(&mut self, pc: u32, sp: u64) {
        self.pc = pc;
        self.sp = sp;
    }

    /// Write the host gamepad state so the guest can see it. The button
    /// bitmask goes to the memory-mapped [`crate::INPUT_ADDR`] (simple polling
    /// mechanism); when libnx has mapped its hid shared memory, the same state
    /// is mirrored into the player-1 `HidNpadInternalState` layout that
    /// `padUpdate` reads, so real homebrew (padInitialize/padUpdate) works too.
    ///
    /// `buttons` is a bitfield of `HidNpadButton` (A=1<<0, B=1<<1, X=1<<2,
    /// Y=1<<3, StickL=1<<4, StickR=1<<5, L=1<<6, R=1<<7, ZL=1<<8, ZR=1<<9,
    /// Plus=1<<10, Minus=1<<11, DpadLeft=1<<12, DpadUp=1<<13, DpadRight=1<<14,
    /// DpadDown=1<<15). Sticks are signed -32768..32767, positive being right
    /// and *up* as Horizon reports them. The stick pseudo-buttons
    /// (`StickLLeft`..`StickRDown`, bits 16-23) are derived here, so a caller
    /// only has to pass the analog values.
    pub fn set_gamepad_state(&mut self, buttons: u64, stick_lx: i32, stick_ly: i32, stick_rx: i32, stick_ry: i32) {
        let buttons = buttons | Self::stick_pseudo_buttons(stick_lx, stick_ly, stick_rx, stick_ry);

        // Simple host→guest register: a u64 mask, then two analog sticks.
        let _ = self.mem.write_u64(crate::INPUT_ADDR, buttons);
        let _ = self.mem.write_u32(crate::INPUT_ADDR + 8, stick_lx as u32);
        let _ = self.mem.write_u32(crate::INPUT_ADDR + 12, stick_ly as u32);
        let _ = self.mem.write_u32(crate::INPUT_ADDR + 16, stick_rx as u32);
        let _ = self.mem.write_u32(crate::INPUT_ADDR + 20, stick_ry as u32);

        if self.hid_shmem_addr == 0 {
            return;
        }
        self.write_hid_gamepad_state(buttons, stick_lx, stick_ly, stick_rx, stick_ry);
    }

    /// The `HidNpadButton_StickL*`/`StickR*` bits hid sets from stick
    /// deflection. Homebrew navigates menus with `HidNpadButton_AnyUp` and
    /// friends, which are the d-pad bit OR'd with these.
    fn stick_pseudo_buttons(lx: i32, ly: i32, rx: i32, ry: i32) -> u64 {
        let mut mask = 0u64;
        for (i, (x, y)) in [(lx, ly), (rx, ry)].iter().enumerate() {
            let base = 16 + 4 * i as u64; // StickLLeft, then StickRLeft
            if *x < -HID_STICK_THRESHOLD {
                mask |= 1 << base;
            }
            if *y > HID_STICK_THRESHOLD {
                mask |= 1 << (base + 1);
            }
            if *x > HID_STICK_THRESHOLD {
                mask |= 1 << (base + 2);
            }
            if *y < -HID_STICK_THRESHOLD {
                mask |= 1 << (base + 3);
            }
        }
        mask
    }

    /// Mirror the gamepad state into libnx's `HidSharedMemory`. The host pad is
    /// published both as player 1 holding a Pro Controller and as the handheld
    /// controller, because homebrew polls whichever of the two it was built to
    /// expect; `padUpdate` merges the slots it was asked for, so a program that
    /// reads both still sees one pad's worth of input.
    fn write_hid_gamepad_state(&mut self, buttons: u64, lx: i32, ly: i32, rx: i32, ry: i32) {
        use hid_shmem as h;
        self.sample_counter = self.sample_counter.wrapping_add(1);
        let sample = self.sample_counter;
        self.write_npad_slot(
            0,
            h::STYLE_FULL_KEY,
            h::DEVICE_FULL_KEY,
            h::FULL_KEY_LIFO,
            h::ATTR_CONNECTED | h::ATTR_WIRED,
            sample,
            buttons,
            (lx, ly, rx, ry),
        );
        self.write_npad_slot(
            h::HANDHELD_SLOT,
            h::STYLE_HANDHELD,
            h::DEVICE_HANDHELD,
            h::HANDHELD_LIFO,
            h::ATTR_CONNECTED
                | h::ATTR_LEFT_CONNECTED
                | h::ATTR_LEFT_WIRED
                | h::ATTR_RIGHT_CONNECTED
                | h::ATTR_RIGHT_WIRED,
            sample,
            buttons,
            (lx, ly, rx, ry),
        );
    }

    /// Publish one `HidNpadInternalState`: the controller's style and device
    /// type, then a single-entry LIFO holding the current button/stick state.
    /// A reader takes `count` entries ending at `tail`, so one entry at index 0
    /// is all `hidGetNpadStates*` needs.
    #[allow(clippy::too_many_arguments)]
    fn write_npad_slot(
        &mut self,
        slot: u32,
        style: u32,
        device_type: u32,
        lifo_off: u32,
        attributes: u32,
        sample: u64,
        buttons: u64,
        sticks: (i32, i32, i32, i32),
    ) {
        use hid_shmem as h;
        let (lx, ly, rx, ry) = sticks;
        let base = self
            .hid_shmem_addr
            .wrapping_add(h::NPAD)
            .wrapping_add(slot.wrapping_mul(h::ENTRY_SIZE));
        let _ = self.mem.write_u32(base + h::STYLE_SET, style);
        let _ = self.mem.write_u32(base + h::JOY_ASSIGNMENT_MODE, 0); // Dual
        let _ = self.mem.write_u32(base + h::DEVICE_TYPE, device_type);

        let lifo = base.wrapping_add(lifo_off);
        let _ = self.mem.write_u64(lifo + h::LIFO_BUFFER_COUNT, h::LIFO_CAPACITY);
        let _ = self.mem.write_u64(lifo + h::LIFO_TAIL, 0);
        let _ = self.mem.write_u64(lifo + h::LIFO_COUNT, 1);

        let entry = lifo.wrapping_add(h::LIFO_STORAGE);
        let _ = self.mem.write_u64(entry + h::STORAGE_SAMPLING_NUMBER, sample);
        let _ = self.mem.write_u64(entry + h::STATE_SAMPLING_NUMBER, sample);
        let _ = self.mem.write_u64(entry + h::STATE_BUTTONS, buttons);
        let _ = self.mem.write_u32(entry + h::STATE_STICK_L, lx as u32);
        let _ = self.mem.write_u32(entry + h::STATE_STICK_L + 4, ly as u32);
        let _ = self.mem.write_u32(entry + h::STATE_STICK_R, rx as u32);
        let _ = self.mem.write_u32(entry + h::STATE_STICK_R + 4, ry as u32);
        let _ = self.mem.write_u32(entry + h::STATE_ATTRIBUTES, attributes);
    }

    /// Publish the host's touchscreen contacts where `hidGetTouchScreenStates`
    /// reads them. Touch is a handheld-only input on real hardware and this
    /// console always reports `AppletOperationMode_Handheld`, so there is no
    /// docked case to suppress.
    ///
    /// An empty slice is how a lift is reported: the state is still published,
    /// with a contact count of zero. Nothing is remembered between calls, so a
    /// caller has to keep sending while a finger is down - which is also what
    /// makes a touch that began before the guest mapped hid's shared memory
    /// show up as soon as it has.
    pub fn set_touch_state(&mut self, touches: &[TouchPoint]) {
        if self.hid_shmem_addr == 0 {
            return;
        }
        use hid_shmem as h;
        self.touch_sample_counter = self.touch_sample_counter.wrapping_add(1);
        let sample = self.touch_sample_counter;

        let lifo = self.hid_shmem_addr.wrapping_add(h::TOUCH_SCREEN);
        let _ = self.mem.write_u64(lifo + h::LIFO_BUFFER_COUNT, h::LIFO_CAPACITY);
        let _ = self.mem.write_u64(lifo + h::LIFO_TAIL, 0);
        let _ = self.mem.write_u64(lifo + h::LIFO_COUNT, 1);

        let storage = lifo.wrapping_add(h::LIFO_STORAGE);
        let _ = self.mem.write_u64(storage + h::STORAGE_SAMPLING_NUMBER, sample);
        let state = storage.wrapping_add(h::TOUCH_STATE);
        let _ = self.mem.write_u64(state + h::TOUCH_SAMPLING_NUMBER, sample);

        let count = touches.len().min(TOUCH_MAX);
        let _ = self.mem.write_u32(state + h::TOUCH_COUNT, count as u32);
        let slot = |i: usize| state + h::TOUCH_TOUCHES + i as u32 * h::TOUCH_SIZE;
        for (i, touch) in touches[..count].iter().enumerate() {
            let e = slot(i);
            // delta_time is how long this contact has been down. Nothing here
            // measures it, and a title that wants a duration times its own
            // frames; reporting a made-up figure would be worse than zero.
            let _ = self.mem.write_u64(e + h::TOUCH_DELTA_TIME, 0);
            let _ = self.mem.write_u32(e + h::TOUCH_ATTRIBUTES, 0);
            let _ = self.mem.write_u32(e + h::TOUCH_FINGER_ID, touch.finger_id);
            let _ = self
                .mem
                .write_u32(e + h::TOUCH_X, touch.x.min(TOUCH_SCREEN_WIDTH - 1));
            let _ = self
                .mem
                .write_u32(e + h::TOUCH_Y, touch.y.min(TOUCH_SCREEN_HEIGHT - 1));
            let _ = self.mem.write_u32(e + h::TOUCH_DIAMETER_X, TOUCH_DIAMETER);
            let _ = self.mem.write_u32(e + h::TOUCH_DIAMETER_Y, TOUCH_DIAMETER);
            let _ = self.mem.write_u32(e + h::TOUCH_ROTATION_ANGLE, 0);
        }
        // A reader that trusts the contact count never looks past it, but one
        // that scans the array would find the fingers a previous, larger sample
        // left there.
        for i in count..self.touch_published {
            let e = slot(i);
            for off in (0..h::TOUCH_SIZE).step_by(4) {
                let _ = self.mem.write_u32(e + off, 0);
            }
        }
        self.touch_published = count;
    }

    /// What the guest last asked the rumble motors to do, as `(low, high)`
    /// amplitudes in 0.0..=1.0. The frontend maps these onto the Gamepad API's
    /// `dual-rumble` strong and weak magnitudes.
    pub fn vibration(&self) -> (f32, f32) {
        self.vibration
    }

    pub(crate) fn set_vibration(&mut self, low: f32, high: f32) {
        let clamp = |v: f32| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
        self.vibration = (clamp(low), clamp(high));
    }

    pub fn hid_shmem_addr(&self) -> u32 {
        self.hid_shmem_addr
    }

    /// The rate and channel count of the samples [`Cpu::take_audio`] returns,
    /// as `(sample_rate, channels)`. `(0, 0)` before the guest has opened an
    /// audio device — there is nothing to play, and no format to play it in.
    pub fn audio_format(&self) -> (u32, u32) {
        self.audio_format
    }

    /// Move up to `out.len()` interleaved samples of queued PCM into `out`,
    /// returning how many were written. What is taken is gone: this is a
    /// hand-off to the host's audio device, not a peek.
    pub fn take_audio(&mut self, out: &mut [i16]) -> usize {
        let n = out.len().min(self.audio_pcm.len());
        for slot in out.iter_mut().take(n) {
            *slot = self.audio_pcm.pop_front().unwrap_or(0);
        }
        n
    }

    /// Queue interleaved PCM for the host, dropping the oldest samples once
    /// the backlog passes [`Cpu::AUDIO_QUEUE_LIMIT`]. Dropping the oldest is
    /// what a real device effectively does when nothing consumes its output:
    /// the guest keeps running, and only the audio that could never have been
    /// heard is lost.
    pub(crate) fn queue_audio(&mut self, samples: impl Iterator<Item = i16>) {
        self.audio_pcm.extend(samples);
        let over = self.audio_pcm.len().saturating_sub(Self::AUDIO_QUEUE_LIMIT);
        self.audio_pcm.drain(..over);
    }

    /// Roughly a second of 48 kHz stereo. Past this the host is not keeping
    /// up and the backlog is only latency.
    pub(crate) const AUDIO_QUEUE_LIMIT: usize = 48_000 * 2;

    /// Provide the font `pl:u` hands out as every shared font type, as the
    /// contents of a TrueType/OpenType file. Homebrew that draws text (hbmenu,
    /// anything using `plGetSharedFont`) feeds these bytes to FreeType, so
    /// without one nothing but pre-rendered bitmaps appears on screen.
    pub fn set_shared_font(&mut self, font: Vec<u8>) {
        self.shared_font = font;
        // A guest that already mapped the shared memory keeps the pointer it was
        // given, so refill the region in place.
        if self.pl_shmem_addr != 0 {
            self.write_shared_font(self.pl_shmem_addr);
        }
    }

    /// How many bytes of font data `pl:u` is serving.
    pub fn shared_font_len(&self) -> usize {
        self.shared_font.len()
    }

    /// Copy the shared font into pl's shared memory at `addr`.
    pub(super) fn write_shared_font(&mut self, addr: u32) {
        let font = std::mem::take(&mut self.shared_font);
        let _ = self.mem.map(addr, &font);
        self.shared_font = font;
    }

    /// Set the wall-clock time `time:u`/`time:s` reports, as POSIX seconds
    /// (UTC). There is no OS clock under `wasm32-unknown-unknown`, so without
    /// a host pushing this (from `Date.now()`), every clock reads the Unix
    /// epoch.
    pub fn set_unix_time(&mut self, seconds: i64) {
        self.unix_time = seconds;
    }

    /// Current value of the emulated RTC, as set by [`Cpu::set_unix_time`].
    pub fn unix_time(&self) -> i64 {
        self.unix_time
    }

    /// Set the battery level `psm` reports. There is no host battery API
    /// reachable from `wasm32-unknown-unknown`, so without a host pushing
    /// this (from the browser's Battery Status API, where available), `psm`
    /// reports a full, charging battery.
    pub fn set_battery(&mut self, percent: u8, charging: bool) {
        self.battery_percent = percent.min(100);
        self.battery_charging = charging;
    }

    /// Current battery reading, as set by [`Cpu::set_battery`].
    pub fn battery(&self) -> (u8, bool) {
        (self.battery_percent, self.battery_charging)
    }

    /// Set the nickname `acc` reports for the console's one user account.
    ///
    /// `nn::account::Nickname` is a fixed 0x20-byte NUL-terminated field, so
    /// anything longer is cut to the 0x1F bytes that fit — on a char
    /// boundary, since a nickname split mid-codepoint would reach the guest
    /// as mojibake rather than as a shorter name.
    pub fn set_user_nickname(&mut self, nickname: &str) {
        let mut end = nickname.len().min(NICKNAME_LEN - 1);
        while end > 0 && !nickname.is_char_boundary(end) {
            end -= 1;
        }
        self.account_nickname = nickname[..end].to_owned();
    }

    /// The nickname `acc` reports, as set by [`Cpu::set_user_nickname`] or by
    /// the guest's own `IProfileEditor::Store`.
    pub fn user_nickname(&self) -> &str {
        &self.account_nickname
    }

    /// Set the program (title) id `pm:info` reports for the running process.
    /// A loader that decrypted an NCA knows it; homebrew has none, and keeps
    /// the Album applet's id it would run under on real hardware.
    pub fn set_program_id(&mut self, program_id: u64) {
        self.program_id = program_id;
    }

    /// The program id `pm` reports, as set by [`Cpu::set_program_id`].
    pub fn program_id(&self) -> u64 {
        self.program_id
    }

    /// A pseudo-random 64-bit value, for `csrng`.
    ///
    /// splitmix64 over a state seeded from the emulated clock. This is **not**
    /// a CSPRNG and nothing that comes out of it should be used as a key:
    /// `wasm32-unknown-unknown` has no OS entropy to draw on, and the security
    /// processor whose hardware RNG really answers `csrng` is not modelled.
    /// What it does guarantee is that a caller asking for random bytes gets
    /// bytes that differ from each other and from the last call, which the
    /// generic reply — leaving the caller's buffer untouched — did not.
    pub(crate) fn next_random_u64(&mut self) -> u64 {
        if self.rng_state == 0 {
            self.rng_state = (self.unix_time as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ 0xA076_1D64_78BD_642F;
        }
        self.rng_state = self.rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    pub fn nzcv(&self) -> u32 {
        self.nzcv
    }

    #[inline(always)]
    fn condition_holds(&self, cond: u8) -> bool {
        let n = (self.nzcv >> 31) & 1;
        let z = (self.nzcv >> 30) & 1;
        let c = (self.nzcv >> 29) & 1;
        let v = (self.nzcv >> 28) & 1;
        match cond & 0xF {
            0x0 => z == 1,                 // EQ
            0x1 => z == 0,                 // NE
            0x2 => c == 1,                 // CS
            0x3 => c == 0,                 // CC
            0x4 => n == 1,                 // MI
            0x5 => n == 0,                 // PL
            0x6 => v == 1,                 // VS
            0x7 => v == 0,                 // VC
            0x8 => c == 1 && z == 0,       // HI
            0x9 => c == 0 || z == 1,       // LS
            0xA => n == v,                 // GE
            0xB => n != v,                 // LT
            0xC => z == 0 && n == v,       // GT
            0xD => z == 1 || n != v,       // LE
            _ => true,                     // AL / NV
        }
    }

    #[inline(always)]
    fn mask(sf: bool) -> u64 {
        if sf { u64::MAX } else { u32::MAX as u64 }
    }

    /// Compute `a + b + carry_in`, returning (result, carry-out, overflow).
    /// Operands are masked to the operation size first: callers pass `b` as
    /// the already-inverted subtrahend for SUB, whose 64-bit `!` would
    /// otherwise pollute the 32-bit carry/overflow computation.
    #[inline]
    fn add_carry_overflow(a: u64, b: u64, carry_in: u64, sf: bool) -> (u64, u32, u32) {
        let size = if sf { 64 } else { 32 };
        let mask = Self::mask(sf);
        let a = a & mask;
        let b = b & mask;
        let base = 1u128 << size;
        let sum = (a as u128) + (b as u128) + (carry_in as u128);
        let result = (sum & (base - 1)) as u64;
        let carry = ((sum >> size) & 1) as u32;
        let sign = 1u64 << (size - 1);
        let overflow = (((a & b & !result) | (!a & !b & result)) & sign != 0) as u32;
        (result, carry, overflow)
    }

    fn set_nzcv_from_alu(&mut self, result: u64, sf: bool, carry: u32, overflow: u32) {
        let n = ((result >> (if sf { 63 } else { 31 })) & 1) as u32;
        let z = (result == 0) as u32;
        self.nzcv = (n << 31) | (z << 30) | (carry << 29) | (overflow << 28);
    }

    fn set_nzcv_from_compare(&mut self, a: u64, b: u64, sub: bool, carry_in: u64, sf: bool) {
        let (result, carry, overflow) = if sub {
            Self::add_carry_overflow(a, !b, carry_in, sf)
        } else {
            Self::add_carry_overflow(a, b, carry_in, sf)
        };
        self.set_nzcv_from_alu(result, sf, carry, overflow);
    }

    /// The ADD/SUB core. `sp_form` says whether register 31 names SP rather
    /// than XZR, which differs by encoding: the immediate and extended-register
    /// forms use SP, the shifted-register form uses XZR. Getting that wrong
    /// turns `neg x1, x0` (`sub x1, xzr, x0`) into a read of the stack
    /// pointer — which is exactly how `aligned_alloc` computes its rounded
    /// size, so it silently corrupts every aligned allocation.
    #[inline(always)]
    fn add_sub(
        &mut self,
        rd: u8,
        rn: u8,
        rhs: u64,
        set_flags: bool,
        sub: bool,
        sf: bool,
        sp_form: bool,
    ) {
        let a = if sp_form { self.read_x(rn) } else { self.read_zr(rn) } & Self::mask(sf);
        let (result, carry, overflow) = if sub {
            Self::add_carry_overflow(a, !rhs, 1, sf)
        } else {
            Self::add_carry_overflow(a, rhs, 0, sf)
        };
        if set_flags {
            self.set_nzcv_from_alu(result, sf, carry, overflow);
        }
        // Rd=31 is SP only for the plain ADD/SUB immediate and extended forms.
        // For the flag-setting forms and the shifted-register form it is XZR,
        // so the result is discarded.
        if set_flags || !sp_form {
            self.write_zr(rd, result);
        } else {
            self.write_x(rd, result);
        }
    }

    // ---- main execution ----

    /// Execute a single instruction. Returns `Ok(())` on success.
    pub fn step(&mut self) -> Result<()> {
        self.step_inner()
    }

    /// The body of [`Cpu::step`], inlined into both the single-step entry point
    /// and [`Cpu::run`]'s loop so a run does not pay a call per instruction.
    #[inline(always)]
    fn step_inner(&mut self) -> Result<()> {
        if self.halted {
            return Err(Error::Cpu("attempted to step a halted CPU".into()));
        }
        // Horizon preempts, and until this was here the scheduler only moved
        // when a thread blocked. Between instructions is a safe place to
        // switch — the whole architectural state is in the context — and
        // `yield_thread` is a no-op when nothing else can run, so a
        // single-threaded guest pays one counter increment for it.
        self.slice_used += 1;
        if self.slice_used >= TIME_SLICE {
            self.slice_used = 0;
            // A timed wait expires on the same tick the scheduler runs on;
            // the slice is far finer than any timeout a guest asks for.
            self.expire_timed_waits();
            self.yield_thread();
        }
        let pc = self.pc;
        let insn = match self.mem.fetch(pc) {
            Ok(i) => i,
            Err(e) => {
                self.record_fault(&e, pc, 0);
                return Err(e);
            }
        };
        let next_pc = pc.wrapping_add(4);
        self.recent[self.recent_len % RECENT_LEN] = (pc, insn);
        self.recent_len = self.recent_len.saturating_add(1);
        let result = self.execute(insn, next_pc);
        if self.trace_enabled {
            self.trace_line(&format!("{:08x}: {:08x}  {}\n", pc, insn, crate::disasm::disassemble(insn)));
        }
        if let Err(e) = &result {
            self.record_fault(e, pc, insn);
        }
        self.cycles += 1;
        result
    }

    fn record_fault(&mut self, e: &Error, pc: u32, insn: u32) {
        self.trace_line(&format!(
            "\n=== FAULT ===\n{}\n  at pc={:#010x} insn={:#010x}  {}\n",
            e,
            pc,
            insn,
            if insn == 0 {
                String::new()
            } else {
                crate::disasm::disassemble(insn)
            }
        ));
        self.trace_regs(pc);
        // Show the run-up to the fault so the crash path is readable without
        // full tracing enabled.
        let n = self.recent_len.min(RECENT_LEN);
        if n > 0 {
            let start = self.recent_len.wrapping_sub(n) % RECENT_LEN;
            self.trace_line(&format!("--- last {} instructions ---\n", n));
            for i in 0..n {
                let (ipc, iinsn) = self.recent[(start + i) % RECENT_LEN];
                self.trace_line(&format!(
                    "{:08x}: {:08x}  {}\n",
                    ipc,
                    iinsn,
                    crate::disasm::disassemble(iinsn)
                ));
            }
        }
    }

    /// One line per guest thread: which one is running, what each is blocked
    /// on, and where it stopped. The counterpart to [`Cpu::backtrace`] for
    /// hangs that are about *scheduling* rather than about one call stack —
    /// a thread spinning without ever reaching a blocking syscall looks
    /// identical to a busy program until you can see that every other thread
    /// is Runnable and none of them has moved.
    /// Index of the thread the core is currently running, for host-side
    /// sampling profilers.
    pub fn current_thread_index(&self) -> usize {
        self.current_thread
    }

    /// Make every blocked thread runnable, and report how many that was. A
    /// debugging lever only: guests re-check their predicates in a loop, so a
    /// spurious wake degrades to a spin rather than a hang, and this answers
    /// "is this process idle because a worker it parked was never woken".
    pub fn wake_all_blocked(&mut self) -> usize {
        let mut woken = 0;
        for thread in &mut self.threads {
            if matches!(thread.state, ThreadState::WaitMutex(_) | ThreadState::WaitKey { .. }) {
                thread.state = ThreadState::Runnable;
                woken += 1;
            }
        }
        woken
    }

    /// Make every thread the guest created but never started runnable, and
    /// report how many that was. A debugging lever only: it answers "is this
    /// process idle because a thread it made never ran" without having to find
    /// the code that would have started it.
    pub fn start_created_threads(&mut self) -> usize {
        let mut started = 0;
        for thread in &mut self.threads {
            if thread.state == ThreadState::Created {
                thread.state = ThreadState::Runnable;
                started += 1;
            }
        }
        started
    }

    pub fn thread_dump(&self) -> String {
        let mut out = String::new();
        for (index, thread) in self.threads.iter().enumerate() {
            let running = index == self.current_thread;
            out.push_str(&format!(
                "  [{index}]{} handle={:#x} state={:?} paused={} pc={:#x}\n",
                if running { "*" } else { " " },
                thread.handle,
                thread.state,
                thread.paused,
                if running { self.pc } else { thread.pc },
            ));
        }
        out
    }

    /// Record a diagnostic the user needs to see wherever the emulator is
    /// running. On the host that is stderr; in the browser there is no stderr
    /// at all — `wasm32-unknown-unknown` has no WASI, so an `eprintln!` there
    /// goes nowhere and `std::env::var` always fails, which is why the
    /// `TRACE_*`-gated traces are CLI-only. The trace buffer is the channel
    /// the page actually drains (`switch_drain_trace`), so anything that must
    /// reach a browser user goes through here as well, and is recorded whether
    /// or not per-instruction tracing is on — the same as fault context.
    pub fn diagnostic(&mut self, line: &str) {
        eprintln!("{line}");
        self.trace_line(&format!("{line}\n"));
    }

    fn trace_line(&mut self, line: &str) {
        if self.trace.len() >= self.trace_cap {
            if !self.trace.ends_with(b"\n[TRACE TRUNCATED]\n") {
                self.trace
                    .extend_from_slice(b"\n[TRACE TRUNCATED]\n");
            }
            return;
        }
        self.trace.extend_from_slice(line.as_bytes());
    }

    fn trace_regs(&mut self, pc: u32) {
        let dump = self.reg_dump();
        self.trace_line(&dump);
        let _ = pc;
    }

    /// Walk the guest's frame-pointer chain and return the return addresses,
    /// innermost first. devkitA64 keeps X29 as a frame pointer (`stp x29, x30,
    /// [sp]; mov x29, sp`), so each frame stores `{saved fp, saved lr}` at the
    /// frame base. Stops as soon as the chain leaves mapped memory or fails to
    /// move forward, so a corrupt stack cannot loop.
    pub fn backtrace(&self, depth: usize) -> Vec<u32> {
        let mut out = Vec::with_capacity(depth + 1);
        out.push(self.regs[30] as u32);
        let mut fp = self.regs[29] as u32;
        for _ in 0..depth {
            let (next_fp, lr) = match (self.mem.read_u64(fp), self.mem.read_u64(fp + 8)) {
                (Ok(next_fp), Ok(lr)) => (next_fp as u32, lr as u32),
                _ => break,
            };
            if lr == 0 || next_fp <= fp {
                break;
            }
            out.push(lr);
            fp = next_fp;
        }
        out
    }

    /// Format a full register snapshot for debugging.
    /// One general-purpose register, for host-side debuggers that need to
    /// read an argument out of a running call rather than a whole dump.
    pub fn reg(&self, i: usize) -> u64 {
        self.regs[i]
    }

    pub fn reg_dump(&self) -> String {
        use std::fmt::Write;
        let mut s = String::with_capacity(1024);
        let n = (self.nzcv >> 31) & 1;
        let z = (self.nzcv >> 30) & 1;
        let c = (self.nzcv >> 29) & 1;
        let v = (self.nzcv >> 28) & 1;
        let _ = writeln!(
            s,
            "pc={:#010x}  sp={:#018x}  nzcv=N:{n} Z:{z} C:{c} V:{v}",
            self.pc, self.sp
        );
        for i in 0..31 {
            let _ = write!(s, "x{:<2}={:#018x}  ", i, self.regs[i]);
            if i % 4 == 3 {
                let _ = writeln!(s);
            }
        }
        let _ = writeln!(s);
        s
    }

    /// Run up to `max_steps` instructions, stopping early on halt or error.
    pub fn run(&mut self, max_steps: u64) -> Result<RunReport> {
        let mut steps = 0u64;
        while steps < max_steps && !self.halted {
            self.step_inner()?;
            steps += 1;
        }
        Ok(RunReport {
            steps,
            halted: self.halted,
        })
    }

    #[inline]
    fn b_imm(&mut self, next_pc: &mut u32, imm: i64) {
        *next_pc = (self.pc as i64).wrapping_add(imm) as u32;
    }

    /// Route an instruction by its top-level encoding group — bits 28:25 of
    /// every A64 instruction, the same classification the architecture manual's
    /// first decode table uses — and only then run that group's decoder.
    ///
    /// [`Cpu::execute_chain`] tries every group in turn, which means an `add`
    /// used to walk the whole load/store, SIMD and floating-point decode before
    /// anything recognised it: ~40ns per instruction, three quarters of the time
    /// the interpreter spent on integer code. Anything a group's decoder does not
    /// claim still falls through to the full chain, so this only changes which
    /// decoder gets first look, never what is decodable.
    fn execute(&mut self, insn: u32, next_pc: u32) -> Result<()> {
        let mut pc = next_pc;
        match (insn >> 25) & 0xF {
            // Data processing -- immediate, PC-relative addressing included.
            0x8 | 0x9 => {
                if self.try_pc_relative(insn) || self.try_data_proc_imm(insn, &mut pc)? {
                    self.pc = pc;
                    return Ok(());
                }
            }
            // Data processing -- register.
            0x5 | 0xD => {
                if self.try_data_proc_reg(insn, &mut pc)? {
                    self.pc = pc;
                    return Ok(());
                }
            }
            // Loads and stores, the literal (PC-relative) forms included.
            0x4 | 0x6 | 0xC | 0xE => {
                if self.try_load_literal(insn)? || self.try_load_store(insn, &mut pc)? {
                    self.pc = pc;
                    return Ok(());
                }
            }
            // Data processing -- SIMD and floating point. Scalar floating point
            // has its own top bytes (0x1E/0x1F for the data-processing and
            // 3-source forms, 0x9E/0x9F for the 64-bit register moves and
            // conversions); everything else in the group is Advanced SIMD. Both
            // decoders still get a look, but asking the right one first saves
            // walking the whole of the other's guard chain.
            0x7 | 0xF => {
                let scalar_fp = matches!((insn >> 24) & 0xFF, 0x1E | 0x1F | 0x9E | 0x9F);
                let claimed = if scalar_fp {
                    self.try_fp(insn)? || self.try_simd(insn)?
                } else {
                    self.try_simd(insn)? || self.try_fp(insn)?
                };
                if claimed {
                    self.pc = pc;
                    return Ok(());
                }
            }
            // Branches, exception generation and system instructions.
            0xA | 0xB => {
                if self.try_branch_or_system(insn, next_pc)? {
                    return Ok(());
                }
            }
            // The reserved and SVE groups, left to the chain.
            _ => {}
        }
        self.execute_chain(insn, next_pc)
    }

    /// ADR/ADRP. Fixed bits[28:24] == 10000; bits[30:29] are immlo (not zero in
    /// general, so an older check that required them to be 0 silently dropped
    /// real ADRP instructions).
    fn try_pc_relative(&mut self, insn: u32) -> bool {
        if ((insn >> 24) & 0x1F) != 0b10000 {
            return false;
        }
        let rd = (insn & 0x1F) as u8;
        let immhi = ((insn >> 5) & 0x7_FFFF) as u64;
        let immlo = ((insn >> 29) & 0b11) as u64;
        let imm = sext_u64((immhi << 2) | immlo, 21);
        let page = (insn >> 31) & 1 == 1;
        let target = if page {
            ((self.pc & !0xFFF) as u64).wrapping_add(imm.wrapping_shl(12))
        } else {
            (self.pc as u64).wrapping_add(imm)
        };
        self.write_zr(rd, target);
        true
    }

    /// `LDR Xt, label` and friends: the literal (PC-relative) load forms.
    fn try_load_literal(&mut self, insn: u32) -> Result<bool> {
        if ((insn >> 27) & 0b111) != 0b011 || ((insn >> 26) & 1) != 0 || ((insn >> 24) & 0b11) != 0b00
        {
            return Ok(false);
        }
        let rt = (insn & 0x1F) as u8;
        let imm = sext_u64((insn >> 5) & 0x7_FFFF, 19) << 2;
        let addr = (self.pc as i64).wrapping_add(imm as i64) as u32;
        match (insn >> 30) & 0b11 {
            0b00 => {
                let val = self.mem.read_u32(addr)? as u64;
                self.write_zr(rt, val & u64::from(u32::MAX));
            }
            0b01 => {
                let val = self.mem.read_u64(addr)?;
                self.write_zr(rt, val);
            }
            0b10 => {
                let val = self.mem.read_u32(addr)? as u64;
                self.write_zr(rt, sext_u64(val, 32));
            }
            // PRFM: a prefetch hint, so nothing to do.
            _ => {}
        }
        Ok(true)
    }

    /// Branches, exception generation and system instructions — the A64 group
    /// with top-level bits 28:25 = 101x — dispatched on the top byte and ordered
    /// by how often real code runs them. `b.cond` is the single most executed
    /// instruction in hbmenu's render loop (12% of a frame), so it is first.
    ///
    /// Returns whether the instruction was handled; a handler sets `self.pc`
    /// itself, since that is the whole point of the group.
    fn try_branch_or_system(&mut self, insn: u32, mut next_pc: u32) -> Result<bool> {
        match (insn >> 24) & 0xFF {
            // B.cond
            0x54 => {
                let imm = sext_u64((insn >> 5) & 0x7_FFFF, 19) << 2;
                let cond = (insn & 0xF) as u8;
                if self.condition_holds(cond) {
                    self.b_imm(&mut next_pc, imm as i64);
                }
                self.pc = next_pc;
                Ok(true)
            }
            // B #imm
            0x14..=0x17 => {
                let imm = sext_u64((insn & 0x3FF_FFFF) as u64, 26) << 2;
                self.b_imm(&mut next_pc, imm as i64);
                self.pc = next_pc;
                Ok(true)
            }
            // TBZ / TBNZ
            0x36 | 0x37 | 0xB6 | 0xB7 => {
                let rt = (insn & 0x1F) as u8;
                let nz = ((insn >> 24) & 1) == 1;
                let bit = ((insn >> 31) & 1) << 5 | ((insn >> 19) & 0x1F);
                let imm = sext_u64((insn >> 5) & 0x3FFF, 14) << 2;
                let bit_val = (self.read_zr(rt) >> bit) & 1 == 1;
                if bit_val == nz {
                    self.b_imm(&mut next_pc, imm as i64);
                }
                self.pc = next_pc;
                Ok(true)
            }
            // CBZ / CBNZ
            0x34 | 0x35 | 0xB4 | 0xB5 => {
                let rt = (insn & 0x1F) as u8;
                let nz = ((insn >> 24) & 1) == 1;
                let imm = sext_u64((insn >> 5) & 0x7_FFFF, 19) << 2;
                let val = self.read_zr(rt);
                let is_zero = if (insn >> 31) & 1 == 1 {
                    val == 0
                } else {
                    (val as u32) == 0
                };
                if is_zero == !nz {
                    self.b_imm(&mut next_pc, imm as i64);
                }
                self.pc = next_pc;
                Ok(true)
            }
            // BL #imm
            0x94..=0x97 => {
                let imm = sext_u64((insn & 0x3FF_FFFF) as u64, 26) << 2;
                self.write_zr(30, next_pc as u64);
                self.b_imm(&mut next_pc, imm as i64);
                self.pc = next_pc;
                Ok(true)
            }
            // BR / BLR / RET
            0xD6 | 0xD7 => {
                let opc = (insn >> 21) & 0xF;
                let op2 = (insn >> 16) & 0x1F;
                let op3 = (insn >> 10) & 0x3F;
                if op2 != 0x1F || op3 != 0 {
                    return Ok(false);
                }
                let rn = ((insn >> 5) & 0x1F) as u8;
                match opc {
                    0b0000 => {
                        // BR
                        self.pc = self.read_zr(rn) as u32;
                        Ok(true)
                    }
                    0b0001 => {
                        // BLR: read the target *before* linking, because the
                        // link register can be the target — `blr x30` is a
                        // return-and-relink, and writing x30 first made it jump
                        // to itself+4. hbmenu's NEON JPEG decoder ends its IDCT
                        // that way, so its icon decode never returned.
                        let target = self.read_zr(rn) as u32;
                        self.write_zr(30, next_pc as u64);
                        self.pc = target;
                        Ok(true)
                    }
                    0b0010 => {
                        // RET. A `ret` that returns to address 0 is a homebrew
                        // exit path whose return-address convention the boot
                        // model doesn't provide (the crt0 stashes the loader's
                        // LR in x27, but the atexit table runner returns via
                        // x30 = 0). Redirect to the exit trampoline so it
                        // surfaces as a clean ExitProcess instead of a NULL
                        // fetch.
                        let tgt = self.read_zr(rn) as u32;
                        self.pc = if tgt == 0 { SELF_RETURN_TRAMPOLINE } else { tgt };
                        Ok(true)
                    }
                    _ => Err(Error::Cpu(format!(
                        "unimplemented branch-register opc {:#b} at {:#x}",
                        opc, self.pc
                    ))),
                }
            }
            // Exception generation: SVC/HVC/SMC and BRK.
            0xD4 => match (insn >> 21) & 0b111 {
                0b000 => {
                    if (insn & 0x1F) == 0b00001 {
                        let imm = ((insn >> 5) & 0xFFFF) as u16;
                        // Retire the SVC before dispatching it: a syscall that
                        // switches threads installs the incoming thread's PC,
                        // and the outgoing one has to resume after its own SVC
                        // (which is what the real ELR holds).
                        self.pc = next_pc;
                        self.syscall(imm)?;
                        Ok(true)
                    } else {
                        Err(Error::Cpu(format!("unimplemented HVC/SMC at {:#x}", self.pc)))
                    }
                }
                0b001 => {
                    let imm = ((insn >> 5) & 0xFFFF) as u16;
                    Err(Error::Cpu(format!("BRK #{} at {:#x}", imm, self.pc)))
                }
                _ => Err(Error::Cpu(format!(
                    "unimplemented exception instruction at {:#x}",
                    self.pc
                ))),
            },
            // MSR/MRS, barriers and hints.
            0xD5 => {
                if ((insn >> 22) & 0x3FF) != 0b1101010100 {
                    return Ok(false);
                }
                self.system(insn, next_pc)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// The original whole-encoding-space chain, kept as the fallback for
    /// anything the group decoders above do not claim. Deliberately not inlined:
    /// it is large and rarely reached, and inlining it into [`Cpu::execute`] made
    /// the hot dispatcher too big to stay in cache.
    #[cold]
    #[inline(never)]
    fn execute_chain(&mut self, insn: u32, mut next_pc: u32) -> Result<()> {
        // ---------------- branches, exceptions, system ----------------
        if self.try_branch_or_system(insn, next_pc)? {
            return Ok(());
        }

        // ---------------- load literal ----------------
        if self.try_load_literal(insn)? {
            self.pc = next_pc;
            return Ok(());
        }

        // ---------------- loads & stores ----------------
        if self.try_load_store(insn, &mut next_pc)? {
            self.pc = next_pc;
            return Ok(());
        }

        // ---------------- minimal SIMD (vector registers) ----------------
        if self.try_simd(insn)? {
            self.pc = next_pc;
            return Ok(());
        }

        // ---------------- scalar floating point ----------------
        if self.try_fp(insn)? {
            self.pc = next_pc;
            return Ok(());
        }

        // ---------------- PC-relative addressing ----------------
        if self.try_pc_relative(insn) {
            self.pc = next_pc;
            return Ok(());
        }

        // ---------------- data processing: immediate ----------------
        if self.try_data_proc_imm(insn, &mut next_pc)? {
            self.pc = next_pc;
            return Ok(());
        }

        // ---------------- data processing: register ----------------
        if self.try_data_proc_reg(insn, &mut next_pc)? {
            self.pc = next_pc;
            return Ok(());
        }

        Err(Error::Cpu(format!(
            "unimplemented instruction 0x{:08x} at pc={:#x}",
            insn, self.pc
        )))
    }
}
