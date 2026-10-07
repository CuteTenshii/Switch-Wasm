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

/// What [`Cpu::audio_activity`] reports; sample counts are interleaved samples since session start.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AudioActivity {
    /// 0 before anything has played.
    pub sample_rate: u32,
    pub channels: u32,
    pub produced: u64,
    pub taken: u64,
    pub dropped: u64,
    pub backlog: u64,
    pub outputs: Vec<AudioOutActivity>,
    pub renderers: Vec<AudioRendererActivity>,
}

/// One open `audout` device; counts run from when it was opened.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioOutActivity {
    pub handle: u64,
    pub sample_rate: u32,
    pub channels: u32,
    pub started: bool,
    pub volume: f32,
    pub appended_buffers: u64,
    pub appended_frames: u64,
    pub released_buffers: u64,
    /// Appended and not yet handed back.
    pub pending_buffers: u64,
    /// Frames appended while stopped, which never play.
    pub discarded_frames: u64,
    /// Buffers whose descriptor pointed outside itself; see `audio_out_append`.
    pub unplayable_buffers: u64,
}

/// One open audio renderer; counts run from when it was opened.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioRendererActivity {
    pub handle: u64,
    pub sample_rate: u32,
    pub started: bool,
    pub updates: u64,
    pub rendered_frames: u64,
    pub voices: u32,
    pub voices_playing: u32,
    /// 0 when no playable sink is configured.
    pub sink_channels: u32,
}

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

/// Host-provided stack for [`Cpu::bootstrap`]: 1 MiB full-descending, top at `STACK_TOP`,
/// between the ASLR region and the heap.
pub const STACK_SIZE: u64 = 0x0010_0000;
pub const STACK_TOP: u64 = 0x2810_0000;

/// The ASLR region `svcGetInfo` reports; the stack region is carved from it.
pub const GUEST_ASLR_REGION_ADDR: u32 = 0x0800_0000;
pub const GUEST_ASLR_REGION_SIZE: u32 = 0x1F00_0000;

/// Return-address trampoline for direct-entered homebrew, just past the ASLR region.
pub const SELF_RETURN_TRAMPOLINE: u32 = GUEST_ASLR_REGION_ADDR + GUEST_ASLR_REGION_SIZE;

/// Thread entry return stub that calls `svcExitThread` (svc 0x0A).
pub const THREAD_EXIT_TRAMPOLINE: u32 = SELF_RETURN_TRAMPOLINE + 0x100;

/// Also advertised as `EntryType_MainThreadHandle`.
pub const MAIN_THREAD_HANDLE: u64 = 1;

/// Horizon's `CUR_THREAD` pseudo-handle.
pub const CURRENT_THREAD_PSEUDO_HANDLE: u64 = 0xFFFF_8000;

/// Placed in `tpidr` by `Cpu::bootstrap`.
pub const MAIN_THREAD_TLS_BASE: u32 = SELF_RETURN_TRAMPOLINE + 0x10_0000;

/// Per-thread TLS blocks for guest-created threads, a page each up to the main stack.
pub const THREAD_TLS_BASE: u32 = MAIN_THREAD_TLS_BASE + 0x1_0000;
/// A page per thread (Horizon uses 0x200), leaving room for newlib's reent struct.
pub const THREAD_TLS_STRIDE: u32 = 0x1000;

/// The system shared buffer the Home Menu and system applets draw into: seven
/// block-linear RGBA8888 slots handed out by AM and presented through `vi`.
pub const SHARED_BUFFER_ADDR: u32 = 0xFA00_0000;
pub const SHARED_BUFFER_SLOTS: u32 = 7;
/// The pool is laid out at the shared layer's size (720p), not the display's; it must not move with the dock.
pub const SHARED_BUFFER_GEOMETRY: OperationMode = OperationMode::Handheld;
/// Reserved address space: the larger geometry, as headroom.
pub const SHARED_BUFFER_RESERVED_SIZE: u32 = OperationMode::Docked.shared_buffer_size();
/// Matches `AcquireSharedFrameBuffer` answering `{0, 1, -1, -1}`.
pub const SHARED_BUFFER_USABLE_SLOTS: u32 = 2;

/// The AM messages queued for the running applet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AppletMessage {
    /// Queued once at startup, then only on a real change.
    FocusStateChanged = 15,
    /// For an applet that called `SetHandlesRequestToDisplay`.
    RequestToDisplay = 41,
    /// The same transition told to an applet rather than an application.
    ChangeIntoForeground = 1,
    /// Docked or undocked; titles re-read `GetOperationMode` on this.
    OperationModeChanged = 30,
    /// Sent alongside `OperationModeChanged`.
    PerformanceModeChanged = 31,
}

/// Horizon's `AppletOperationMode`: drives the reported resolution, performance
/// mode, GPU clock and touchscreen, which must all agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OperationMode {
    /// 720p handheld screen, with a touchscreen.
    #[default]
    Handheld = 0,
    /// 1080p dock. `AppletOperationMode_Console`.
    Docked = 1,
}

impl OperationMode {
    /// The display `vi` composes for this mode, as (width, height).
    pub const fn display_size(self) -> (u32, u32) {
        match self {
            OperationMode::Handheld => (1280, 720),
            OperationMode::Docked => (1920, 1080),
        }
    }

    /// `ApmPerformanceMode`: Normal handheld, Boost docked.
    pub const fn performance_mode(self) -> u32 {
        self as u32
    }

    /// GPU clock in Hz; only the GPU clock changes with the dock.
    pub const fn gpu_clock_hz(self) -> u32 {
        match self {
            OperationMode::Handheld => 384_000_000,
            OperationMode::Docked => 768_000_000,
        }
    }

    pub const fn shared_buffer_stride(self) -> u32 {
        self.display_size().0 * 4
    }

    /// Display height rounded up to a 128-row block-linear block (720 to 768, 1080 to 1152).
    pub const fn shared_buffer_rows(self) -> u32 {
        const BLOCK_ROWS: u32 = 128;
        self.display_size().1.div_ceil(BLOCK_ROWS) * BLOCK_ROWS
    }

    pub const fn shared_buffer_slot_size(self) -> u32 {
        self.shared_buffer_stride() * self.shared_buffer_rows()
    }

    pub const fn shared_buffer_size(self) -> u32 {
        self.shared_buffer_slot_size() * SHARED_BUFFER_SLOTS
    }
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

/// Mutex word bit meaning unlock must go through `svcArbitrateUnlock`.
const MUTEX_HAS_LISTENERS: u32 = 0x4000_0000;

/// Written into a condvar's word while a thread waits; `nn::os` skips the signal syscall when it is zero.
const CONDVAR_HAS_WAITERS: u32 = 1;

/// The stack region `svcGetInfo` reports, where `nn::os` places thread stacks at
/// random; it must be entirely free and large.
pub const GUEST_STACK_REGION_ADDR: u32 = 0x1800_0000;
pub const GUEST_STACK_REGION_SIZE: u32 = SELF_RETURN_TRAMPOLINE - GUEST_STACK_REGION_ADDR;

/// End of the guest address space: below is soft-mapped by [`Cpu::bootstrap`],
/// above faults (hbmenu probes the top to size the space).
pub const GUEST_SPACE_END: u32 = 0xFF00_0000;

/// Heap region (`svcSetHeapSize`) and alias region (`svcMapPhysicalMemory`). A process
/// uses only one route, so this layout gives the heap nearly everything.
pub const GUEST_HEAP_REGION_ADDR: u32 = 0x3000_0000;
pub const GUEST_HEAP_REGION_SIZE: u32 = 0xC800_0000;
pub const GUEST_ALIAS_REGION_ADDR: u32 =
    GUEST_HEAP_REGION_ADDR.wrapping_add(GUEST_HEAP_REGION_SIZE);
pub const GUEST_ALIAS_REGION_SIZE: u32 = 0x0200_0000;

/// Where `ldr:ro` maps run-time loaded modules: between [`STACK_TOP`] and the heap,
/// clear of every region the guest's allocators use.
pub const RO_MODULE_REGION_ADDR: u32 = 0x2900_0000;
pub const RO_MODULE_REGION_SIZE: u32 = GUEST_HEAP_REGION_ADDR.wrapping_sub(RO_MODULE_REGION_ADDR);

/// `svcGetInfo`'s total memory, which `nn::init` asks for as its heap: one region's worth.
pub const GUEST_TOTAL_MEMORY_SIZE: u32 = GUEST_HEAP_REGION_SIZE;

/// The arena `VammManagerImplByHorizon` claims at the alias region base; an SDK constant.
pub const VAMM_ARENA_SIZE: u32 = 0x3FE0_0000;

/// Layout for titles using virtual address memory: nearly everything is alias region.
/// The heap region is unused here, so it is small.
pub const VAMM_HEAP_REGION_SIZE: u32 = 0x0800_0000;
pub const VAMM_ALIAS_REGION_ADDR: u32 = GUEST_HEAP_REGION_ADDR.wrapping_add(VAMM_HEAP_REGION_SIZE);
/// Up to the system shared buffer.
pub const VAMM_ALIAS_REGION_SIZE: u32 = SHARED_BUFFER_ADDR.wrapping_sub(VAMM_ALIAS_REGION_ADDR);
/// `TotalMemorySize`, and the size of the alias-region reservation `nn::init` makes.
pub const VAMM_TOTAL_MEMORY_SIZE: u32 = 0x3800_0000;
/// `SystemResourceSizeTotal` for this layout.
pub const VAMM_SYSTEM_RESOURCE_SIZE: u32 = 0x0100_0000;

/// Layout for firmware library applets, which use both a Vamm arena and `svcSetHeapSize`.
pub const APPLET_HEAP_REGION_SIZE: u32 = 0x2000_0000;
pub const APPLET_ALIAS_REGION_ADDR: u32 =
    GUEST_HEAP_REGION_ADDR.wrapping_add(APPLET_HEAP_REGION_SIZE);
pub const APPLET_ALIAS_REGION_SIZE: u32 = SHARED_BUFFER_ADDR.wrapping_sub(APPLET_ALIAS_REGION_ADDR);

/// The address space a process is given, selected by its NPDM system resource size:
/// `nnSdk` uses the Vamm manager exactly when `SystemResourceSizeTotal` is nonzero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryLayout {
    pub heap_addr: u32,
    pub heap_size: u32,
    pub alias_addr: u32,
    pub alias_size: u32,
    /// `TotalMemorySize`.
    pub total_memory: u32,
    /// `SystemResourceSizeTotal`; zero keeps `nnSdk` off the Vamm manager.
    pub system_resource: u32,
}

impl MemoryLayout {
    /// No system resource: `libnx` homebrew and `nnSdk` titles declaring zero.
    pub const PLAIN: MemoryLayout = MemoryLayout {
        heap_addr: GUEST_HEAP_REGION_ADDR,
        heap_size: GUEST_HEAP_REGION_SIZE,
        alias_addr: GUEST_ALIAS_REGION_ADDR,
        alias_size: GUEST_ALIAS_REGION_SIZE,
        total_memory: GUEST_TOTAL_MEMORY_SIZE,
        system_resource: 0,
    };

    /// Titles declaring a system resource.
    pub const VIRTUAL_ADDRESS: MemoryLayout = MemoryLayout {
        heap_addr: GUEST_HEAP_REGION_ADDR,
        heap_size: VAMM_HEAP_REGION_SIZE,
        alias_addr: VAMM_ALIAS_REGION_ADDR,
        alias_size: VAMM_ALIAS_REGION_SIZE,
        total_memory: VAMM_TOTAL_MEMORY_SIZE,
        system_resource: VAMM_SYSTEM_RESOURCE_SIZE,
    };

    /// Firmware library applets; see [`APPLET_HEAP_REGION_SIZE`].
    pub const APPLET: MemoryLayout = MemoryLayout {
        heap_addr: GUEST_HEAP_REGION_ADDR,
        heap_size: APPLET_HEAP_REGION_SIZE,
        alias_addr: APPLET_ALIAS_REGION_ADDR,
        alias_size: APPLET_ALIAS_REGION_SIZE,
        total_memory: APPLET_HEAP_REGION_SIZE,
        system_resource: VAMM_SYSTEM_RESOURCE_SIZE,
    };

    /// Zero (or no readable manifest) selects the plain layout.
    pub fn for_system_resource(size: u32) -> MemoryLayout {
        if size == 0 {
            MemoryLayout::PLAIN
        } else {
            MemoryLayout::VIRTUAL_ADDRESS
        }
    }

    /// [`Self::for_system_resource`], except library applets get [`Self::APPLET`].
    pub fn for_program(program_id: u64, size: u32) -> MemoryLayout {
        if size != 0 && crate::cpu::am::is_library_applet(program_id) {
            return MemoryLayout::APPLET;
        }
        MemoryLayout::for_system_resource(size)
    }
}

/// hid's shared memory size, used to recognise that mapping.
pub const HID_SHMEM_SIZE: u32 = 0x4_0000;

/// `pl:u`'s shared memory size (`SHAREDMEMFONT_SIZE`), recognised the same way.
pub const PL_SHMEM_SIZE: u32 = 0x110_0000;

/// One shared font in pl's shared memory; `offset` points past the 8-byte header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FontRegion {
    pub offset: u32,
    pub size: u32,
}

/// Shared fonts in `PlSharedFontType` order (archive id, file name), plus
/// `nintendo_ext2_003` seventh, matching Eden's `SHARED_FONTS`.
const SHARED_FONTS: [(u64, &str); 7] = [
    (0x0100_0000_0000_0811, "/nintendo_udsg-r_std_003.bfttf"),
    (
        0x0100_0000_0000_0814,
        "/nintendo_udsg-r_org_zh-cn_003.bfttf",
    ),
    (
        0x0100_0000_0000_0814,
        "/nintendo_udsg-r_ext_zh-cn_003.bfttf",
    ),
    (0x0100_0000_0000_0813, "/nintendo_udjxh-db_zh-tw_003.bfttf"),
    (0x0100_0000_0000_0812, "/nintendo_udsg-r_ko_003.bfttf"),
    (0x0100_0000_0000_0810, "/nintendo_ext_003.bfttf"),
    (0x0100_0000_0000_0810, "/nintendo_ext2_003.bfttf"),
];

/// A `.bfttf`'s first four bytes; the xor key is derived from them.
const BFTTF_MAGIC: [u8; 4] = [0x36, 0xf8, 0x1a, 0x1e];
const BFTTF_KEY: [u8; 4] = [0x49, 0x62, 0x18, 0x06];

const BFTTF_HEADER: usize = 8;

/// Decode a `.bfttf` into header plus TrueType file. The size field stays byte-reversed, as on a console.
pub fn decode_bfttf(file: &[u8]) -> Option<Vec<u8>> {
    let len = file.len() / 4 * 4;
    if len < BFTTF_HEADER || file[..4] != BFTTF_MAGIC {
        return None;
    }
    let mut out: Vec<u8> = file[..len]
        .iter()
        .zip(BFTTF_KEY.iter().cycle())
        .map(|(b, k)| b ^ k)
        .collect();
    out[4..8].copy_from_slice(&[file[7], file[6], file[5], file[4]]);
    Some(out)
}

/// Wrap a TrueType file as a `.bfttf`, padding (not trimming) to whole words.
pub fn encode_bfttf(ttf: &[u8]) -> Vec<u8> {
    let len = ttf.len().next_multiple_of(4);
    let mut out = Vec::with_capacity(len + BFTTF_HEADER);
    out.extend_from_slice(&[0x7f, 0x9a, 0x02, 0x18]);
    out.extend_from_slice(&(len as u32).to_be_bytes());
    out.extend_from_slice(ttf);
    out.resize(len + BFTTF_HEADER, 0);
    for (i, b) in out.iter_mut().enumerate() {
        *b ^= BFTTF_KEY[i % 4];
    }
    out
}

/// Offsets into libnx's `HidSharedMemory` (`switch/services/hid.h`).
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
    /// The remaining per-style LIFOs at a 0x350 stride: Joy-Con pair, left, right, system.
    pub const JOY_DUAL_LIFO: u32 = 0x6C8;
    pub const JOY_LEFT_LIFO: u32 = 0xA18;
    pub const JOY_RIGHT_LIFO: u32 = 0xD68;
    pub const SYSTEM_EXT_LIFO: u32 = 0x1408;
    pub const DEVICE_TYPE: u32 = 0x4188;
    /// `HidNpadSystemProperties`, then the three `HidPowerInfo` battery levels.
    pub const SYSTEM_PROPERTIES: u32 = 0x4190;
    pub const BATTERY_LEVEL: u32 = 0x4198;
    /// Power infos per npad: the whole pad, then its left and right halves.
    pub const POWER_INFO_COUNT: u32 = 3;

    /// `HidNpadCommonLifo`: a 0x20-byte header (unused/buffer_count/tail/count) and 17 entries.
    pub const LIFO_BUFFER_COUNT: u32 = 0x08;
    pub const LIFO_TAIL: u32 = 0x10;
    pub const LIFO_COUNT: u32 = 0x18;
    pub const LIFO_STORAGE: u32 = 0x20;
    pub const LIFO_CAPACITY: u64 = 17;

    /// `HidNpadCommonStateAtomicStorage`: sampling number (doubled; bit 0 is the seqlock flag),
    /// then the `HidNpadCommonState`.
    pub const STORAGE_SAMPLING_NUMBER: u32 = 0x00;
    pub const STATE_SAMPLING_NUMBER: u32 = 0x08;
    pub const STATE_BUTTONS: u32 = 0x10;
    pub const STATE_STICK_L: u32 = 0x18;
    pub const STATE_STICK_R: u32 = 0x20;
    pub const STATE_ATTRIBUTES: u32 = 0x28;

    /// `HidNpadStyleTag` bits.
    pub const STYLE_FULL_KEY: u32 = 1 << 0;
    pub const STYLE_HANDHELD: u32 = 1 << 1;
    pub const STYLE_JOY_DUAL: u32 = 1 << 2;
    pub const STYLE_JOY_LEFT: u32 = 1 << 3;
    pub const STYLE_JOY_RIGHT: u32 = 1 << 4;
    pub const STYLE_SYSTEM_EXT: u32 = 1 << 29;

    pub const DEVICE_FULL_KEY: u32 = 1 << 0;
    pub const DEVICE_HANDHELD: u32 = (1 << 2) | (1 << 3); // HandheldLeft|Right
    pub const DEVICE_JOY_LEFT: u32 = 1 << 4;
    pub const DEVICE_JOY_RIGHT: u32 = 1 << 5;

    /// `HidNpadJoyAssignmentMode`.
    pub const JOY_ASSIGNMENT_DUAL: u32 = 0;
    pub const JOY_ASSIGNMENT_SINGLE: u32 = 1;

    /// `HidPowerInfo::battery_level`, 0 to 4.
    pub const BATTERY_FULL: u32 = 4;

    /// `PowerInfo{0,1,2}PowerConnected` bits of `system_properties`; `Charging` stays clear.
    pub const SYSTEM_PROP_POWER_CONNECTED: u32 = (1 << 3) | (1 << 4) | (1 << 5);
    /// Button capabilities: ABXY, plus/minus, d-pad.
    pub const SYSTEM_PROP_FULL_BUTTONS: u32 = (1 << 11) | (1 << 13) | (1 << 14) | (1 << 15);

    pub const ATTR_CONNECTED: u32 = 1 << 0;
    pub const ATTR_WIRED: u32 = 1 << 1;
    pub const ATTR_LEFT_CONNECTED: u32 = 1 << 2;
    pub const ATTR_LEFT_WIRED: u32 = 1 << 3;
    pub const ATTR_RIGHT_CONNECTED: u32 = 1 << 4;
    pub const ATTR_RIGHT_WIRED: u32 = 1 << 5;

    /// `offsetof(HidSharedMemory, npad_condition)`. `nn::hid::GetNpadJoyHoldType` reads it
    /// directly and aborts (`2202-0710`) unless `is_valid` is set.
    pub const NPAD_CONDITION: u32 = 0x3E200;
    /// Its four words: reserved, initialized flag, hold type, valid flag.
    pub const NPAD_CONDITION_INITIALIZED: u32 = 0x04;
    pub const NPAD_CONDITION_HOLD_TYPE: u32 = 0x08;
    pub const NPAD_CONDITION_VALID: u32 = 0x0C;

    /// `offsetof(HidSharedMemory, touch_screen)`; its LIFO uses the npad LIFO header.
    pub const TOUCH_SCREEN: u32 = 0x400;
    /// The state begins one `u64` into each storage entry.
    pub const TOUCH_STATE: u32 = 0x08;

    /// `HidTouchScreenState` fields: sampling number, live count, then the slots.
    pub const TOUCH_SAMPLING_NUMBER: u32 = 0x00;
    pub const TOUCH_COUNT: u32 = 0x08;
    pub const TOUCH_TOUCHES: u32 = 0x10;

    /// `sizeof(HidTouchState)` and the fields of one.
    pub const TOUCH_SIZE: u32 = 0x28;
    pub const TOUCH_DELTA_TIME: u32 = 0x00;
    pub const TOUCH_ATTRIBUTES: u32 = 0x08;
    /// `nn::hid::TouchAttribute` start and end bits.
    pub const TOUCH_ATTR_START: u32 = 1 << 0;
    pub const TOUCH_ATTR_END: u32 = 1 << 1;
    pub const TOUCH_FINGER_ID: u32 = 0x0C;
    pub const TOUCH_X: u32 = 0x10;
    pub const TOUCH_Y: u32 = 0x14;
    pub const TOUCH_DIAMETER_X: u32 = 0x18;
    pub const TOUCH_DIAMETER_Y: u32 = 0x1C;
    pub const TOUCH_ROTATION_ANGLE: u32 = 0x20;
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

/// Styles the pad can be published as, best first. A style the title does not
/// support is no pad at all to `nn::hid`. `SystemExt` is published alongside, not here.
const NPAD_PRESENTATIONS: [NpadPresentation; 4] = [
    NpadPresentation {
        style: hid_shmem::STYLE_FULL_KEY,
        device_type: hid_shmem::DEVICE_FULL_KEY,
        lifo: hid_shmem::FULL_KEY_LIFO,
        attributes: hid_shmem::ATTR_CONNECTED | hid_shmem::ATTR_WIRED,
        joy_assignment: hid_shmem::JOY_ASSIGNMENT_DUAL,
    },
    NpadPresentation {
        style: hid_shmem::STYLE_JOY_DUAL,
        device_type: hid_shmem::DEVICE_JOY_LEFT | hid_shmem::DEVICE_JOY_RIGHT,
        lifo: hid_shmem::JOY_DUAL_LIFO,
        attributes: hid_shmem::ATTR_CONNECTED
            | hid_shmem::ATTR_WIRED
            | hid_shmem::ATTR_LEFT_CONNECTED
            | hid_shmem::ATTR_LEFT_WIRED
            | hid_shmem::ATTR_RIGHT_CONNECTED
            | hid_shmem::ATTR_RIGHT_WIRED,
        joy_assignment: hid_shmem::JOY_ASSIGNMENT_DUAL,
    },
    NpadPresentation {
        style: hid_shmem::STYLE_JOY_LEFT,
        device_type: hid_shmem::DEVICE_JOY_LEFT,
        lifo: hid_shmem::JOY_LEFT_LIFO,
        attributes: hid_shmem::ATTR_CONNECTED
            | hid_shmem::ATTR_WIRED
            | hid_shmem::ATTR_LEFT_CONNECTED
            | hid_shmem::ATTR_LEFT_WIRED,
        joy_assignment: hid_shmem::JOY_ASSIGNMENT_SINGLE,
    },
    NpadPresentation {
        style: hid_shmem::STYLE_JOY_RIGHT,
        device_type: hid_shmem::DEVICE_JOY_RIGHT,
        lifo: hid_shmem::JOY_RIGHT_LIFO,
        attributes: hid_shmem::ATTR_CONNECTED
            | hid_shmem::ATTR_WIRED
            | hid_shmem::ATTR_RIGHT_CONNECTED
            | hid_shmem::ATTR_RIGHT_WIRED,
        joy_assignment: hid_shmem::JOY_ASSIGNMENT_SINGLE,
    },
];

/// The handheld pad: its own npad id, not one of player 1's styles.
const NPAD_HANDHELD: NpadPresentation = NpadPresentation {
    style: hid_shmem::STYLE_HANDHELD,
    device_type: hid_shmem::DEVICE_HANDHELD,
    lifo: hid_shmem::HANDHELD_LIFO,
    attributes: hid_shmem::ATTR_CONNECTED
        | hid_shmem::ATTR_LEFT_CONNECTED
        | hid_shmem::ATTR_LEFT_WIRED
        | hid_shmem::ATTR_RIGHT_CONNECTED
        | hid_shmem::ATTR_RIGHT_WIRED,
    joy_assignment: hid_shmem::JOY_ASSIGNMENT_DUAL,
};

/// Every style the pad can be published in, as reported to the controller applet and `hid:sys`.
fn supported_npad_style_set() -> u32 {
    NPAD_PRESENTATIONS.iter().fold(
        NPAD_HANDHELD.style | hid_shmem::STYLE_SYSTEM_EXT,
        |set, pad| set | pad.style,
    )
}

/// Player 1's presentation for the title's style set; Pro Controller when none match.
fn npad_presentation_for(style_set: u32) -> NpadPresentation {
    NPAD_PRESENTATIONS
        .into_iter()
        .find(|presentation| style_set & presentation.style != 0)
        .unwrap_or(NPAD_PRESENTATIONS[0])
}

/// Deflection past which the stick pseudo-buttons are reported.
const HID_STICK_THRESHOLD: i32 = 0x4000;

/// The touchscreen digitizer resolution; touches use this space whatever the guest renders at.
pub const TOUCH_SCREEN_WIDTH: u32 = 1280;
pub const TOUCH_SCREEN_HEIGHT: u32 = 720;

pub const TOUCH_MAX: usize = 16;

/// Contact size reported for every touch; only checked to be non-zero.
const TOUCH_DIAMETER: u32 = 10;

/// One finger in digitizer coordinates; `finger_id` is stable while it stays down.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TouchPoint {
    pub finger_id: u32,
    pub x: u32,
    pub y: u32,
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

/// Meaningful register slots: X0..=X30, then the zero register's discard slot and SP,
/// so register 31's meaning is an index chosen at decode time.
const REG_SLOTS: usize = 34;

/// Slots allocated: the full `u8` range, so indexing by a slot byte needs no bounds check.
const REG_FILE: usize = 256;
const _: () = assert!(REG_FILE == u8::MAX as usize + 1 && REG_FILE >= REG_SLOTS);

/// A slot as an index into the register file.
#[inline(always)]
fn reg_slot(slot: u8) -> usize {
    debug_assert!(
        (slot as usize) < REG_SLOTS,
        "register slot {slot} is not one"
    );
    slot as usize
}
/// Reads of `XZR`; never written.
const ZR_SLOT: usize = 31;
/// The read path indexes with the encoding's 5-bit field, so this must be 31.
const _: () = assert!(ZR_SLOT == 31);
/// Writes to `XZR` land here.
const ZR_DISCARD: usize = 32;

/// Exposed for difftest harnesses, which must skip it: its contents are not guest state.
pub const DISCARD_SLOT: usize = ZR_DISCARD;
const SP_SLOT: usize = 33;

/// Per condition code, a 16-bit mask with bit `nzcv` set when the code holds.
/// Branchless evaluation of `B.cond`.
const CONDITION_MASKS: [u16; 16] = condition_masks();

const fn condition_masks() -> [u16; 16] {
    let mut table = [0u16; 16];
    let mut cond = 0usize;
    while cond < 16 {
        let mut nzcv = 0usize;
        while nzcv < 16 {
            let n = (nzcv >> 3) & 1;
            let z = (nzcv >> 2) & 1;
            let c = (nzcv >> 1) & 1;
            let v = nzcv & 1;
            let holds = match cond {
                0x0 => z == 1,           // EQ
                0x1 => z == 0,           // NE
                0x2 => c == 1,           // CS
                0x3 => c == 0,           // CC
                0x4 => n == 1,           // MI
                0x5 => n == 0,           // PL
                0x6 => v == 1,           // VS
                0x7 => v == 0,           // VC
                0x8 => c == 1 && z == 0, // HI
                0x9 => c == 0 || z == 1, // LS
                0xA => n == v,           // GE
                0xB => n != v,           // LT
                0xC => z == 0 && n == v, // GT
                0xD => z == 1 || n != v, // LE
                _ => true,               // AL / NV
            };
            if holds {
                table[cond] |= 1 << nzcv;
            }
            nzcv += 1;
        }
        cond += 1;
    }
    table
}

/// Instructions per 60 Hz display refresh at 1.02 GHz; vsync fires even without a present.
pub const VSYNC_PERIOD_CYCLES: u64 = 1_020_000_000 / 60;

/// Instructions per 200 Hz `hid` sample; LIFOs advance even with no input.
pub const HID_SAMPLE_PERIOD_CYCLES: u64 = 1_020_000_000 / 200;

/// Instructions a thread runs before preemption.
const TIME_SLICE: u64 = 20_000;

/// Default thread priority (also what most retail manifests declare); 0 is most urgent, 63 least.
pub const DEFAULT_THREAD_PRIORITY: u8 = 44;

const MAIN_THREAD_ID: u64 = 1;

const LOWEST_PRIORITY: u8 = 63;
/// The console has four cores, 0 to 3.
const LAST_CORE: u8 = 3;

/// Decisions a runnable thread may be passed over before it runs regardless, since
/// all threads share one host core.
const STARVE_DECISIONS: u32 = 8;

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

    /// Map a runtime environment and point SP at a stack, as the loader does before
    /// jumping to a program's entry point. Only hosts booting real homebrew call this.
    pub fn bootstrap(&mut self) {
        // The whole guest space is lazily mapped: reads see zeros, writes allocate on first touch.
        self.mem.soft_map_zero(0, GUEST_SPACE_END);
        let _ = self
            .mem
            .map_zero((STACK_TOP - STACK_SIZE) as u32, STACK_SIZE as usize);
        self.regs[SP_SLOT] = STACK_TOP;
        // TLS base, clear of the heap, the stack and the GPU driver's own allocations.
        self.tpidr = u64::from(MAIN_THREAD_TLS_BASE);
        // LR points at a stub that calls ExitProcess (svc 0x07), so returning from main exits cleanly.
        let _ = self.mem.map_zero(SELF_RETURN_TRAMPOLINE, 0x10);
        self.mem.write_u32(SELF_RETURN_TRAMPOLINE, 0xD400_00E1).ok(); // svc #7
        self.mem
            .write_u32(SELF_RETURN_TRAMPOLINE + 4, 0x1400_0000)
            .ok(); // b .
                   // Returning from a thread entry point is `svcExitThread` (svc 0x0A).
        let _ = self.mem.map_zero(THREAD_EXIT_TRAMPOLINE, 0x10);
        self.mem.write_u32(THREAD_EXIT_TRAMPOLINE, 0xD400_0141).ok(); // svc #0xa
        self.mem
            .write_u32(THREAD_EXIT_TRAMPOLINE + 4, 0x1400_0000)
            .ok(); // b .
    }

    // ---- guest threads ----

    /// Created on demand so a single-threaded program costs nothing.
    fn ensure_main_thread(&mut self) {
        if self.threads.is_empty() {
            self.threads.push(ThreadContext {
                handle: MAIN_THREAD_HANDLE,
                id: MAIN_THREAD_ID,
                state: ThreadState::Runnable,
                paused: false,
                regs: [0; REG_FILE],
                pc: 0,
                nzcv: 0,
                mode: self.mode,
                cpsr_q: false,
                cpsr_ge: 0,
                fpscr_nzcv: 0,
                vregs: VRegs::default(),
                fpcr: self.fpcr,
                fpsr: 0,
                tpidr: self.tpidr,
                tpidr_rw: self.tpidr_rw,
                priority: self.main_thread_priority,
                core: self.main_thread_core,
                ideal_core: i32::from(self.main_thread_core),
                affinity: 1 << self.main_thread_core,
                passed_over: 0,
                entry: 0,
                arg: 0,
                ran: 0,
                switches: 0,
                busy_at: self.cycles,
            });
            self.current_thread = 0;
        }
    }

    /// Create a thread as `svcCreateThread` does (TLS with libnx `ThreadVars`, stack,
    /// entry, argument in x0). Returns its handle.
    pub(super) fn create_thread(
        &mut self,
        entry: u32,
        arg: u64,
        stack_top: u64,
        priority: u8,
        core: u8,
    ) -> u64 {
        self.ensure_main_thread();
        let handle = self.alloc_handle();
        let index = self.threads.len() as u32;
        let tls = THREAD_TLS_BASE + index * THREAD_TLS_STRIDE;
        let _ = self.mem.map_zero(tls, THREAD_TLS_STRIDE as usize);
        // ThreadVars at TLS+0x1E0: magic, handle, thread pointer, reent, tls_tp.
        const TV_MAGIC: u32 = 0x2154_5624; // "!TV$"
        let reent = tls + 0x400;
        let _ = self.mem.write_u32(tls + 0x1E0, TV_MAGIC);
        let _ = self.mem.write_u32(tls + 0x1E4, handle as u32);
        let _ = self.mem.write_u32(tls + 0x1E8, 0);
        let _ = self.mem.write_u32(tls + 0x1F0, reent);
        let _ = self.mem.write_u32(tls + 0x1F8, tls);

        let mut regs = [0u64; REG_FILE];
        regs[0] = arg;
        // Inherit the creator's execution state; LR and SP slots differ per mode.
        if self.mode == ExecMode::A32 {
            regs[14] = THREAD_EXIT_TRAMPOLINE as u64;
            regs[13] = stack_top;
        } else {
            regs[30] = THREAD_EXIT_TRAMPOLINE as u64;
            regs[SP_SLOT] = stack_top;
        }
        let id = self.next_thread_id;
        self.next_thread_id += 1;
        self.threads.push(ThreadContext {
            handle,
            id,
            state: ThreadState::Created,
            paused: false,
            regs,
            pc: entry,
            nzcv: 0,
            mode: self.mode,
            cpsr_q: false,
            cpsr_ge: 0,
            fpscr_nzcv: 0,
            vregs: VRegs::default(),
            fpcr: 0,
            fpsr: 0,
            tpidr: u64::from(tls),
            tpidr_rw: 0,
            priority: priority.min(LOWEST_PRIORITY),
            core,
            ideal_core: i32::from(core),
            affinity: 1 << core,
            passed_over: 0,
            entry,
            arg,
            ran: 0,
            switches: 0,
            busy_at: self.cycles,
        });
        let line = format!(
            "{} created by {}: arg {arg:#x}, stack top {stack_top:#x}, priority {priority}, \
             core {core}",
            self.thread_label(handle),
            self.thread_label(self.current_thread_handle())
        );
        self.log_thread(line);
        handle
    }

    /// Mark a created thread runnable (`svcStartThread`).
    pub(super) fn start_thread(&mut self, handle: u64) -> bool {
        for thread in &mut self.threads {
            if thread.handle == handle && thread.state == ThreadState::Created {
                thread.state = ThreadState::Runnable;
                let line = format!("{} started", self.thread_label(handle));
                self.log_thread(line);
                return true;
            }
        }
        let line = format!(
            "{} asked to start, but it is not a thread waiting to be started",
            self.thread_label(handle)
        );
        self.log_thread(line);
        false
    }

    /// `svcSetThreadActivity`. `Err(())` when already in the requested state, as Horizon reports.
    pub(super) fn set_thread_paused(&mut self, handle: u64, paused: bool) -> Option<bool> {
        let thread = self.threads.iter_mut().find(|t| t.handle == handle)?;
        if thread.paused == paused {
            return Some(false);
        }
        thread.paused = paused;
        let line = format!(
            "{} {} by {}",
            self.thread_label(handle),
            if paused { "paused" } else { "resumed" },
            self.thread_label(self.current_thread_handle())
        );
        self.log_thread(line);
        Some(true)
    }

    /// Fill the 0x320-byte `ThreadContext` for `svcGetThreadContext3` from the live or saved registers.
    pub(super) fn write_thread_context(&mut self, out: u32, handle: u64) -> bool {
        self.ensure_main_thread();
        let Some(index) = self.threads.iter().position(|t| t.handle == handle) else {
            return false;
        };
        let (regs, pc, nzcv, vregs, fpcr, fpsr, tpidr) = if index == self.current_thread {
            (
                self.regs, self.pc, self.nzcv, self.vregs, self.fpcr, self.fpsr, self.tpidr,
            )
        } else {
            let t = &self.threads[index];
            (t.regs, t.pc, t.nzcv, t.vregs, t.fpcr, t.fpsr, t.tpidr)
        };
        let sp = regs[SP_SLOT];
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
        let _ = self.mem.write_u32(out.wrapping_add(0x310), fpcr);
        let _ = self.mem.write_u32(out.wrapping_add(0x314), fpsr);
        put64(self, 0x318, tpidr);
        true
    }

    pub(super) fn has_other_runnable(&self) -> bool {
        self.threads
            .iter()
            .enumerate()
            .any(|(i, t)| i != self.current_thread && t.state == ThreadState::Runnable && !t.paused)
    }

    /// End the running thread and switch away; the process ends only when the main thread exits.
    pub(super) fn exit_thread(&mut self) {
        self.ensure_main_thread();
        let line = format!(
            "{} exited at {}",
            self.thread_label(self.current_thread_handle()),
            self.locate(self.pc)
        );
        self.log_thread(line);
        if self.current_thread == 0 {
            self.halted = true;
            return;
        }
        self.threads[self.current_thread].state = ThreadState::Finished;
        // Joiners are parked on this thread's handle, now signalled.
        self.wake_event_waiters();
        if !self.switch_to_next_runnable() {
            // Nothing else can run: fall back to the main thread.
            self.threads[0].state = ThreadState::Runnable;
            self.switch_to_next_runnable();
        }
    }

    /// Give the CPU to [`Cpu::pick_next`]'s choice, unless the running thread should keep it.
    pub(super) fn yield_thread(&mut self) {
        if self.threads.len() < 2 || !self.has_other_runnable() {
            return;
        }
        let current = self.current_thread;
        let still_runnable =
            self.threads[current].state == ThreadState::Runnable && !self.threads[current].paused;
        if still_runnable {
            if let Some(next) = self.pick_next() {
                let other = &self.threads[next];
                if other.passed_over < STARVE_DECISIONS
                    && self.threads[current].priority < other.priority
                {
                    self.pass_over_all_but(current);
                    return;
                }
            }
        }
        self.switch_to_next_runnable();
    }

    /// Next thread to run: a starved one first, then by priority, round-robin among equals.
    fn pick_next(&self) -> Option<usize> {
        let count = self.threads.len();
        let start = self.current_thread;
        let mut best: Option<usize> = None;
        for step in 1..count {
            let candidate = (start + step) % count;
            let thread = &self.threads[candidate];
            if thread.state != ThreadState::Runnable || thread.paused {
                continue;
            }
            if thread.passed_over >= STARVE_DECISIONS {
                return Some(candidate);
            }
            if best.is_none_or(|b| thread.priority < self.threads[b].priority) {
                best = Some(candidate);
            }
        }
        best
    }

    /// Count a passed-over decision against every runnable thread except `chosen`.
    fn pass_over_all_but(&mut self, chosen: usize) {
        for (index, thread) in self.threads.iter_mut().enumerate() {
            if index == chosen {
                thread.passed_over = 0;
            } else if thread.state == ThreadState::Runnable && !thread.paused {
                thread.passed_over = thread.passed_over.saturating_add(1);
            }
        }
    }

    /// `svcGetThreadPriority`; `None` for a non-thread handle.
    pub(super) fn thread_priority(&self, handle: u64) -> Option<u8> {
        let handle = self.resolve_thread_handle(handle);
        match self.threads.iter().find(|t| t.handle == handle) {
            Some(thread) => Some(thread.priority),
            // No main thread slot yet; use the manifest priority.
            None if handle == MAIN_THREAD_HANDLE => Some(self.main_thread_priority),
            None => None,
        }
    }

    /// `svcSetThreadPriority`; `false` for a non-thread handle.
    pub(super) fn set_thread_priority(&mut self, handle: u64, priority: u8) -> bool {
        let handle = self.resolve_thread_handle(handle);
        self.ensure_main_thread();
        match self.threads.iter_mut().find(|t| t.handle == handle) {
            Some(thread) => {
                thread.priority = priority;
                true
            }
            None => false,
        }
    }

    /// Resolve `CURRENT_THREAD` to the running thread's handle.
    fn resolve_thread_handle(&self, handle: u64) -> u64 {
        if handle == CURRENT_THREAD_PSEUDO_HANDLE {
            self.current_thread_handle()
        } else {
            handle
        }
    }

    /// The main thread's priority from `main.npdm`.
    pub fn set_main_thread_priority(&mut self, priority: u8) {
        self.main_thread_priority = priority.min(LOWEST_PRIORITY);
        if let Some(main) = self
            .threads
            .iter_mut()
            .find(|t| t.handle == MAIN_THREAD_HANDLE)
        {
            main.priority = self.main_thread_priority;
        }
    }

    /// The process core mask from `main.npdm`, for `svcGetInfo` and thread core checks.
    pub fn set_process_core_mask(&mut self, mask: u64) {
        self.process_core_mask = mask;
    }

    /// The main thread's core from `main.npdm`, also the process's default core.
    pub fn set_main_thread_core(&mut self, core: u8) {
        self.main_thread_core = core.min(LAST_CORE);
        if let Some(main) = self
            .threads
            .iter_mut()
            .find(|t| t.handle == MAIN_THREAD_HANDLE)
        {
            main.core = self.main_thread_core;
            main.ideal_core = i32::from(self.main_thread_core);
            main.affinity = 1 << self.main_thread_core;
        }
    }

    pub(super) fn thread_id(&mut self, handle: u64) -> Option<u64> {
        let handle = self.resolve_thread_handle(handle);
        self.ensure_main_thread();
        self.threads
            .iter()
            .find(|t| t.handle == handle)
            .map(|t| t.id)
    }

    pub(super) fn current_core(&self) -> u8 {
        self.threads
            .get(self.current_thread)
            .map_or(self.main_thread_core, |t| t.core)
    }

    /// A thread's ideal core (-1 for none) and affinity mask.
    pub(super) fn thread_core_mask(&mut self, handle: u64) -> Option<(i32, u64)> {
        let handle = self.resolve_thread_handle(handle);
        self.ensure_main_thread();
        self.threads
            .iter()
            .find(|t| t.handle == handle)
            .map(|t| (t.ideal_core, t.affinity))
    }

    /// Set a thread's ideal core and affinity mask and migrate it as the kernel does.
    /// `false` for an unknown handle.
    pub(super) fn set_thread_core_mask(
        &mut self,
        handle: u64,
        ideal_core: i32,
        affinity: u64,
    ) -> bool {
        let handle = self.resolve_thread_handle(handle);
        self.ensure_main_thread();
        match self.threads.iter_mut().find(|t| t.handle == handle) {
            Some(thread) => {
                thread.ideal_core = ideal_core;
                thread.affinity = affinity;
                if ideal_core >= 0 {
                    thread.core = ideal_core as u8;
                } else if affinity & (1 << thread.core) == 0 {
                    thread.core = affinity.trailing_zeros() as u8;
                }
                true
            }
            None => false,
        }
    }

    // ---- mutexes and condition variables ----
    //
    // The lock word holds the owner's handle plus MUTEX_HAS_LISTENERS when contended;
    // libnx re-reads it, so ownership must really move.

    /// `svcArbitrateLock`: block until the owner releases, unless the word already changed.
    pub(super) fn arbitrate_lock(&mut self, owner: u32, addr: u32, _self_handle: u32) {
        self.ensure_main_thread();
        let word = self.mem.read_u32(addr).unwrap_or(0);
        if word & !MUTEX_HAS_LISTENERS != owner || owner == 0 {
            return; // stale request; the guest re-reads the word and retries
        }
        self.threads[self.current_thread].state = ThreadState::WaitMutex(addr);
        self.reschedule();
    }

    /// `svcArbitrateUnlock`: hand the mutex to a waiter, or clear it.
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

    /// `svcWaitProcessWideKeyAtomic`: release the mutex and block on the condvar,
    /// marking the condvar word so `nn::os` signals it.
    pub(super) fn wait_process_wide_key(
        &mut self,
        mutex: u32,
        key: u32,
        _self_handle: u32,
        timeout: i64,
    ) {
        self.ensure_main_thread();
        let _ = self.mem.write_u32(key, CONDVAR_HAS_WAITERS);
        self.arbitrate_unlock(mutex);
        let deadline = self.wait_deadline(timeout);
        self.threads[self.current_thread].state = ThreadState::WaitKey {
            key,
            mutex,
            deadline,
        };
        self.reschedule();
    }

    /// Wake expired timed waits, at most once per [`TIME_SLICE`] cycles of the clock.
    #[inline(always)]
    pub(super) fn sweep_timed_waits(&mut self) {
        if self.cycles < self.next_expiry {
            return;
        }
        self.next_expiry = self.cycles.wrapping_add(TIME_SLICE);
        self.expire_timed_waits();
    }

    /// Wake every timed wait whose deadline has passed; `nn::os` rechecks its predicate.
    pub(super) fn expire_timed_waits(&mut self) {
        let now = self.cycles;
        for index in 0..self.threads.len() {
            let state = self.threads[index].state;
            let deadline = match state {
                ThreadState::WaitKey { deadline, .. }
                | ThreadState::WaitAddress { deadline, .. } => deadline,
                ThreadState::Sleeping { deadline } | ThreadState::WaitEvent { deadline } => {
                    Some(deadline)
                }
                _ => None,
            };
            if !deadline.is_some_and(|at| now >= at) {
                continue;
            }
            match state {
                ThreadState::WaitKey { mutex, .. } => self.wake_condvar_waiter(index, mutex),
                _ => self.threads[index].state = ThreadState::Runnable,
            }
        }
    }

    /// Dequeue a condvar waiter holding (or queued for) its mutex, as the kernel does on every wake.
    fn wake_condvar_waiter(&mut self, index: usize, mutex: u32) {
        let handle = self.threads[index].handle as u32;
        let owner = self.mem.read_u32(mutex).unwrap_or(0);
        if owner == 0 {
            let _ = self.mem.write_u32(mutex, handle);
            self.threads[index].state = ThreadState::Runnable;
        } else {
            // Contended: queue up and mark the word so the owner arbitrates its unlock.
            let _ = self.mem.write_u32(mutex, owner | MUTEX_HAS_LISTENERS);
            self.threads[index].state = ThreadState::WaitMutex(mutex);
        }
    }

    /// `svcSignalProcessWideKey`: wake up to `count` waiters (all if negative).
    pub(super) fn signal_process_wide_key(&mut self, key: u32, count: i32) {
        self.ensure_main_thread();
        let mut woken = 0;
        for i in 0..self.threads.len() {
            if count >= 0 && woken >= count {
                break;
            }
            if let ThreadState::WaitKey {
                key: waiting,
                mutex,
                ..
            } = self.threads[i].state
            {
                if waiting != key {
                    continue;
                }
                self.wake_condvar_waiter(i, mutex);
                woken += 1;
            }
        }
        // Clear the word once the queue is empty.
        let queued = self.threads.iter().any(
            |t| matches!(t.state, ThreadState::WaitKey { key: waiting, .. } if waiting == key),
        );
        if !queued {
            let _ = self.mem.write_u32(key, 0);
        }
    }

    /// Deadline for a `timeout` in nanoseconds; negative waits forever.
    fn wait_deadline(&self, timeout: i64) -> Option<u64> {
        (timeout > 0).then(|| {
            let cycles = (timeout as u128) * u128::from(crate::cpu::power::CLOCK_RATES_HZ[0])
                / 1_000_000_000;
            self.cycles.wrapping_add(cycles as u64)
        })
    }

    // ---- the address arbiter ----
    //
    // The arbiter word carries no ownership; the kernel only compares it with the caller's value.

    /// `svcWaitForAddress`'s decision, separate from [`Cpu::block_on_address`] so X0 is
    /// written before switching threads.
    pub(super) fn arbitrate_address(
        &mut self,
        addr: u32,
        arb_type: u32,
        value: i32,
        timeout: i64,
    ) -> ArbiterWait {
        self.ensure_main_thread();
        let Ok(current) = self.mem.read_u32(addr).map(|w| w as i32) else {
            return ArbiterWait::Mismatch;
        };
        let holds = match arb_type {
            // WaitIfLessThan, and its atomic-decrement variant.
            0 | 1 => current < value,
            // WaitIfEqual.
            2 => current == value,
            _ => return ArbiterWait::Mismatch,
        };
        if !holds {
            return ArbiterWait::Mismatch;
        }
        if arb_type == 1 {
            let _ = self.mem.write_u32(addr, current.wrapping_sub(1) as u32);
        }
        // A zero timeout is a poll.
        if timeout == 0 {
            return ArbiterWait::TimedOut;
        }
        ArbiterWait::Blocked
    }

    /// Park on the arbiter word at `addr`, after [`Cpu::arbitrate_address`] decided to wait.
    pub(super) fn block_on_address(&mut self, addr: u32, timeout: i64) {
        let deadline = self.wait_deadline(timeout);
        self.threads[self.current_thread].state = ThreadState::WaitAddress { addr, deadline };
        self.reschedule();
    }

    /// `svcSignalToAddress`: wake up to `count` waiters (all if negative) after the
    /// signal type's compare-and-modify. Reports whether the word held `value`.
    pub(super) fn signal_to_address(
        &mut self,
        addr: u32,
        signal_type: u32,
        value: i32,
        count: i32,
    ) -> bool {
        self.ensure_main_thread();
        let waiting = self
            .threads
            .iter()
            .filter(|t| matches!(t.state, ThreadState::WaitAddress { addr: a, .. } if a == addr))
            .count() as i32;
        if signal_type != 0 {
            let Ok(current) = self.mem.read_u32(addr).map(|w| w as i32) else {
                return false;
            };
            if current != value {
                return false;
            }
            let updated = match signal_type {
                // SignalAndIncrementIfEqual.
                1 => value.wrapping_add(1),
                // SignalAndModifyByWaitingCountIfEqual: the new word tells a semaphore whether waiters remain.
                _ => match (count > 0).then_some(waiting.cmp(&count)) {
                    Some(std::cmp::Ordering::Greater) => value.wrapping_sub(1),
                    Some(std::cmp::Ordering::Equal) => value,
                    Some(std::cmp::Ordering::Less) => value.wrapping_add(1),
                    None if waiting > 0 => value.wrapping_sub(1),
                    None => value.wrapping_add(1),
                },
            };
            let _ = self.mem.write_u32(addr, updated as u32);
        }
        let mut woken = 0;
        for i in 0..self.threads.len() {
            if count >= 0 && woken >= count {
                break;
            }
            if matches!(self.threads[i].state, ThreadState::WaitAddress { addr: a, .. } if a == addr)
            {
                self.threads[i].state = ThreadState::Runnable;
                woken += 1;
            }
        }
        true
    }

    /// The soonest timed-wait deadline: the furthest the clock may idle forward.
    pub(super) fn earliest_deadline(&self) -> Option<u64> {
        self.threads
            .iter()
            .filter(|t| !t.paused)
            .filter_map(|t| match t.state {
                ThreadState::WaitKey { deadline, .. }
                | ThreadState::WaitAddress { deadline, .. } => deadline,
                ThreadState::Sleeping { deadline } | ThreadState::WaitEvent { deadline } => {
                    Some(deadline)
                }
                _ => None,
            })
            .min()
    }

    /// Park the presenting thread until the next 60 Hz refresh, as a swapchain would.
    /// Takes effect after the reply is written; see [`Cpu::pending_sleep`].
    pub(super) fn pace_present(&mut self) {
        let tick = self.last_present_cycles.wrapping_add(VSYNC_PERIOD_CYCLES);
        if self.cycles < tick {
            self.pending_sleep = Some(tick);
        }
        // A title slower than the panel is not dragged backwards.
        self.last_present_cycles = self.cycles.max(tick);
    }

    /// Present a frame whose surface was still on the device, once it has come back.
    fn complete_pending_present(&mut self) {
        let Some(buffer) = self.pending_present else {
            return;
        };
        match self.nv.gpu.flush_renderers(&mut self.mem) {
            Ok(crate::gpu::renderer::Flush::Pending) => {}
            Ok(crate::gpu::renderer::Flush::Done) => {
                self.pending_present = None;
                if let Err(e) = self.nv.gpu.present(&self.mem, &buffer) {
                    self.diagnostic(
                        Level::Error,
                        &format!("[vi] the deferred present failed: {e}"),
                    );
                }
            }
            Err(e) => {
                // Drop the frame rather than stall the display.
                self.pending_present = None;
                self.diagnostic(
                    Level::Error,
                    &format!("[vi] the deferred readback failed: {e}"),
                );
            }
        }
    }

    pub(super) fn sleep_until(&mut self, deadline: u64) {
        self.ensure_main_thread();
        self.threads[self.current_thread].state = ThreadState::Sleeping { deadline };
        self.reschedule();
    }

    /// The next display refresh, a deadline that always arrives.
    pub(super) fn next_display_tick(&self) -> u64 {
        self.last_vsync_cycles.wrapping_add(VSYNC_PERIOD_CYCLES)
    }

    /// Park on awaited events until one fires or `deadline` passes; the `svc` is reissued.
    pub(super) fn park_on_events(&mut self, deadline: u64) {
        self.ensure_main_thread();
        self.threads[self.current_thread].state = ThreadState::WaitEvent { deadline };
        self.reschedule();
    }

    /// Switch away after blocking. If nothing can run, idle or wake everything.
    fn reschedule(&mut self) {
        if self.switch_to_next_runnable() {
            return;
        }
        // Idle the clock to the earliest deadline of any kind, as the console idles.
        if let Some(deadline) = self.earliest_deadline() {
            if deadline > self.cycles {
                self.cycles = deadline;
            }
            self.expire_timed_waits();
            if self.switch_to_next_runnable() {
                return;
            }
        }
        // No deadline either: wake everything rather than hang.
        for index in 0..self.threads.len() {
            match self.threads[index].state {
                ThreadState::WaitKey { mutex, .. } => self.wake_condvar_waiter(index, mutex),
                ThreadState::WaitMutex(_)
                | ThreadState::WaitAddress { .. }
                | ThreadState::WaitEvent { .. } => {
                    self.threads[index].state = ThreadState::Runnable;
                }
                _ => {}
            }
        }
        self.switch_to_next_runnable();
    }

    /// Account one retired instruction: a cycle and a step. Both engines call this.
    #[inline(always)]
    pub(super) fn retire(&mut self) {
        self.cycles += 1;
        self.steps += 1;
    }

    /// Record a run of `count` instructions from `start` in the fault trail.
    #[inline(always)]
    pub(super) fn record_run(&mut self, start: u32, count: u32) {
        // A single step that continues the last run extends it.
        if count == 1 && self.recent_len != 0 {
            let index = (self.recent_len - 1) & (RECENT_LEN - 1);
            let (last_start, last_count) = self.recent[index];
            if last_count < RECENT_LEN as u32 && last_start.wrapping_add(last_count * 4) == start {
                self.recent[index].1 = last_count + 1;
                return;
            }
        }
        self.push_run(start, count);
    }

    /// [`Cpu::record_run`] without the single-step merge, which only pays off
    /// for the interpreter's steps.
    #[inline(always)]
    pub(super) fn push_run(&mut self, start: u32, count: u32) {
        self.recent[self.recent_len % RECENT_LEN] = (start, count);
        self.recent_len = self.recent_len.wrapping_add(1);
    }

    /// Switch to the thread [`Cpu::pick_next`] chooses. Returns false if there
    /// is none (in which case the running thread keeps going).
    fn switch_to_next_runnable(&mut self) -> bool {
        let start = self.current_thread;
        let Some(candidate) = self.pick_next() else {
            return false;
        };
        self.pass_over_all_but(candidate);
        self.account_slice(start);
        self.threads[candidate].switches += 1;
        self.save_context(start);
        self.load_context(candidate);
        true
    }

    fn save_context(&mut self, index: usize) {
        let thread = &mut self.threads[index];
        thread.regs = self.regs;
        thread.pc = self.pc;
        thread.nzcv = self.nzcv;
        thread.mode = self.mode;
        thread.cpsr_q = self.cpsr_q;
        thread.cpsr_ge = self.cpsr_ge;
        thread.fpscr_nzcv = self.fpscr_nzcv;
        thread.vregs = self.vregs;
        thread.fpcr = self.fpcr;
        thread.fpsr = self.fpsr;
        thread.tpidr = self.tpidr;
        thread.tpidr_rw = self.tpidr_rw;
    }

    fn load_context(&mut self, index: usize) {
        self.slice_used = 0;
        // A switch clears the local monitor.
        self.exclusive = None;
        let thread = self.threads[index].clone();
        self.regs = thread.regs;
        self.pc = thread.pc;
        self.nzcv = thread.nzcv;
        self.mode = thread.mode;
        self.cpsr_q = thread.cpsr_q;
        self.cpsr_ge = thread.cpsr_ge;
        self.fpscr_nzcv = thread.fpscr_nzcv;
        self.vregs = thread.vregs;
        self.fpcr = thread.fpcr;
        self.fpsr = thread.fpsr;
        self.tpidr = thread.tpidr;
        self.tpidr_rw = thread.tpidr_rw;
        self.current_thread = index;
    }

    pub fn current_thread_handle(&self) -> u64 {
        self.threads
            .get(self.current_thread)
            .map_or(MAIN_THREAD_HANDLE, |t| t.handle)
    }

    /// Threads created, including the main thread.
    pub fn thread_count(&self) -> usize {
        self.threads.len().max(1)
    }

    /// Boot a homebrew NRO as HBL does: run the crt0 up to `main`, then the `.init_array`
    /// and main `ThreadVars` setup the skipped `__libnx_init` would provide.
    pub fn boot_homebrew(&mut self, data: &[u8]) -> Result<crate::nro::LoadedNro> {
        self.mem.clear_modules();
        self.module_names.clear();
        let loaded = crate::nro::load_nro(&mut self.mem, data)?;
        let end = loaded
            .data
            .mem_addr
            .wrapping_add(loaded.data.file_size)
            .wrapping_add(loaded.bss_size);
        self.record_module_name(loaded.base, end, "homebrew");
        // Expose the NRO at argv[0] on the SD card for `romfsMountSelf`.
        self.fs
            .write_file(crate::nro::HOMEBREW_NRO_PATH, data.to_vec());
        self.out.clear();
        self.trace.clear();
        self.halted = false;
        self.guest_fatal = None;
        self.trace_enabled = false;
        for i in 0..=30u8 {
            self.set_reg(i, 0);
        }
        self.set_reg(0, loaded.env_addr as u64);
        self.set_reg(1, if loaded.env_addr != 0 { u64::MAX } else { 1 });
        self.set_reg(30, SELF_RETURN_TRAMPOLINE as u64);

        let init = crate::nro::init_array_entries(data);
        if !init.is_empty() && loaded.env_addr != 0 {
            // The crt0 calls main at entry+0xc0; BSS is zeroed and relocations applied by then.
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
                // ThreadVars at TLS+0x1E0: magic, handle, thread_ptr, _REENT (zeroed; lazily set up), tls_tp.
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
                // Restore the entry registers and resume at the crt0's call; x2 is the loader's
                // return address, which `__nx_exit` jumps to.
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

    /// Boot a retail title's modules (`rtld`, `main`, `subsdk*`, `sdk`, in that order)
    /// back to back in one address space and enter `rtld`, which relocates the rest.
    pub fn boot_retail_program(
        &mut self,
        modules: &[(&str, &[u8])],
    ) -> Result<Vec<crate::nso::LoadedNso>> {
        self.out.clear();
        self.trace.clear();
        self.halted = false;
        self.guest_fatal = None;
        self.trace_enabled = false;
        self.mem.clear_modules();
        self.module_names.clear();
        for i in 0..=30u8 {
            self.set_reg(i, 0);
        }
        // Horizon's entry ABI: X0 is 0 for a normal launch, X1 the main thread handle
        // (`nnSdk` compares `SdkMutex` lock words against it).
        self.set_reg(1, MAIN_THREAD_HANDLE);
        if self.mode == ExecMode::A32 {
            // A32 keeps SP in r13; restore it after clearing the registers.
            self.regs[13] = self.regs[SP_SLOT];
            self.regs[14] = SELF_RETURN_TRAMPOLINE as u64;
        } else {
            self.set_reg(30, SELF_RETURN_TRAMPOLINE as u64);
        }

        const MODULE_ALIGN: u32 = 0x1000;
        let mut base = crate::nso::NSO_BASE;
        let mut loaded = Vec::with_capacity(modules.len());
        // Name the title, so fault addresses are read against the right binary.
        self.diagnostic(
            Level::Info,
            &format!("[loader] program {:#018x}", self.program_id),
        );
        for (name, data) in modules {
            let module = crate::nso::load_nso(&mut self.mem, data, base).map_err(|e| {
                Error::Cpu(format!("loading module {:?} at {:#x}: {}", name, base, e))
            })?;
            let image_end = module
                .data
                .mem_addr
                .wrapping_add(module.data.file_size)
                .wrapping_add(module.bss_size);
            // Where each module landed; `rtld` finds them itself via `svcQueryMemory`.
            self.diagnostic(Level::Info, &format!(
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
            self.record_module_name(module.base, image_end, name);
            base = image_end.wrapping_add(MODULE_ALIGN - 1) & !(MODULE_ALIGN - 1);
            loaded.push(module);
        }
        let entry = loaded
            .first()
            .ok_or_else(|| Error::Cpu("no modules to boot".into()))?
            .entry;
        self.set_pc(entry);
        self.seed_applet_launch_arguments();
        self.seed_launch_parameters();
        Ok(loaded)
    }

    /// Seed `am`'s launch parameters as a console's launcher would, including the
    /// `PreselectedUser` that `nn::account::OpenPreselectedUser` requires.
    fn seed_launch_parameters(&mut self) {
        self.am_launch_parameters.clear();
        if crate::cpu::am::is_library_applet(self.program_id) {
            return;
        }
        self.am_launch_parameters.insert(
            crate::cpu::am::LAUNCH_PARAMETER_PRESELECTED_USER,
            crate::cpu::am::preselected_user_parameter(self.current_user().uid),
        );
    }

    /// Queue what a library applet's caller would push: `LibAppletCommonArguments`, then
    /// the applet's own launch structs (see [`crate::cpu::am::applet_launch_storages`]).
    fn seed_applet_launch_arguments(&mut self) {
        self.am_in_data.clear();
        self.am_out_data.clear();
        self.am_interactive_in.clear();
        self.am_interactive_out.clear();
        if !crate::cpu::am::is_library_applet(self.program_id) {
            return;
        }
        const COMMON_ARGS_VERSION: u32 = 1;
        const COMMON_ARGS_SIZE: u32 = 0x20;
        let mut args = Vec::with_capacity(COMMON_ARGS_SIZE as usize);
        args.extend_from_slice(&COMMON_ARGS_VERSION.to_le_bytes());
        args.extend_from_slice(&COMMON_ARGS_SIZE.to_le_bytes());
        // LaVersion: the applet's own interface revision.
        args.extend_from_slice(
            &crate::cpu::am::applet_interface_version(self.program_id).to_le_bytes(),
        );
        // ExpectedThemeColor: 0 is the basic white theme.
        args.extend_from_slice(&0u32.to_le_bytes());
        // PlayStartupSound, then padding out to the tick field.
        args.resize(0x18, 0);
        // The tick the caller started the applet at.
        args.extend_from_slice(&0u64.to_le_bytes());
        self.am_in_data.push_back(args);
        // Then the applet's own launch structs.
        let user = self.current_user().uid;
        for storage in crate::cpu::am::applet_launch_storages(self.program_id, user) {
            self.am_in_data.push_back(storage);
        }
    }

    /// Set the decrypted RomFS that `OpenDataStorageByCurrentProcess` serves, for small images.
    /// See [`Cpu::set_romfs_source`].
    pub fn set_romfs(&mut self, data: Vec<u8>) {
        self.romfs = Some(Box::new(crate::source::MemSource(data)));
        self.romfs_indexes.remove(&None);
    }

    /// Register a system data archive for `OpenDataStorageByDataId`.
    pub fn add_data_archive(&mut self, data_id: u64, src: Box<dyn crate::source::ByteSource>) {
        self.data_archives.insert(data_id, src);
        self.romfs_indexes.remove(&Some(data_id));
    }

    /// Base id for this title's DLC: the NACP's, or the base program id (low 13 bits
    /// masked) plus 0x1000.
    pub fn add_on_content_base_id(&self) -> u64 {
        match self.add_on_content_base_id {
            0 => (self.program_id & !0x1FFF) + 0x1000,
            declared => declared,
        }
    }

    /// Set the DLC base id from the title's NACP.
    pub fn set_add_on_content_base_id(&mut self, base: u64) {
        self.add_on_content_base_id = base;
    }

    /// Register add-on content under its own id and return its index, or `None` when
    /// it belongs to another title.
    pub fn add_add_on_content(
        &mut self,
        content_id: u64,
        src: Box<dyn crate::source::ByteSource>,
    ) -> Option<u32> {
        let index = content_id.checked_sub(self.add_on_content_base_id())?;
        if index > 0x7FF {
            return None;
        }
        self.data_archives.insert(content_id, src);
        self.romfs_indexes.remove(&Some(content_id));
        self.add_on_content.insert(index as u32);
        // Tell a running title to re-read the list.
        if let Some(event) = self.aoc_list_changed_event {
            self.signal_event(event);
        }
        Some(index as u32)
    }

    pub fn has_data_archive(&self, data_id: u64) -> bool {
        self.data_archives.contains_key(&data_id)
    }

    pub fn add_on_content(&self) -> Vec<u32> {
        self.add_on_content.iter().copied().collect()
    }

    /// The save `key` names, created on first open as on a console.
    pub fn save_data_mut(&mut self, key: SaveKey) -> &mut crate::vfs::Vfs {
        self.saves.entry(key).or_insert_with(crate::vfs::Vfs::empty)
    }

    pub fn save_data(&self, key: SaveKey) -> Option<&crate::vfs::Vfs> {
        self.saves.get(&key)
    }

    /// Every opened save, for a host that persists them.
    pub fn save_keys(&self) -> Vec<SaveKey> {
        self.saves.keys().copied().collect()
    }

    pub(super) fn vfs_for(&mut self, mount: Option<SaveKey>) -> &mut crate::vfs::Vfs {
        match mount {
            Some(key) => self.saves.entry(key).or_insert_with(crate::vfs::Vfs::empty),
            None => &mut self.fs,
        }
    }

    pub(super) fn mount_of(&self, key: u64) -> Option<SaveKey> {
        self.fs_mount.get(&key).copied()
    }

    pub(super) fn set_mount(&mut self, key: u64, mount: Option<SaveKey>) {
        match mount {
            Some(id) => {
                self.fs_mount.insert(key, id);
            }
            None => {
                self.fs_mount.remove(&key);
            }
        }
    }

    /// Same, backed by a decrypt-on-demand [`ByteSource`](crate::source::ByteSource).
    pub fn set_romfs_source(&mut self, src: Box<dyn crate::source::ByteSource>) {
        self.romfs = Some(src);
        self.romfs_indexes.remove(&None);
    }

    // ---- register access ----

    #[inline]
    pub fn get_pc(&self) -> u32 {
        self.pc
    }

    /// SP from [`SP_SLOT`] in A64 or `r13` in AArch32.
    #[inline]
    pub fn sp(&self) -> u64 {
        match self.mode {
            ExecMode::A64 => self.regs[SP_SLOT],
            ExecMode::A32 => self.regs[13],
        }
    }

    pub fn set_pc(&mut self, pc: u32) {
        self.pc = pc;
    }

    /// Allocate an event handle; it must reach the guest as a copy handle.
    pub(crate) fn alloc_event(&mut self, name: &'static str, auto_clear: bool) -> u64 {
        let handle = self.alloc_handle();
        self.events.insert(
            handle,
            Event {
                name,
                signaled: false,
                auto_clear,
            },
        );
        if crate::trace::enabled(crate::trace::Trace::Wait) {
            crate::traceln!("[event] {name} = {handle:#x} auto_clear={auto_clear}");
        }
        handle
    }

    pub(crate) fn event_name(&self, handle: u64) -> Option<&'static str> {
        self.events.get(&handle).map(|event| event.name)
    }

    /// Queue an applet message and wake whatever polls for one.
    pub(super) fn queue_applet_message(&mut self, message: AppletMessage) {
        self.applet_messages.push_back(message as u32);
        if let Some(handle) = self.applet_event {
            self.signal_event(handle);
        }
    }

    pub fn operation_mode(&self) -> OperationMode {
        self.operation_mode
    }

    /// Dock or undock while running: queues `OperationModeChanged` and
    /// `PerformanceModeChanged` on a real change.
    pub fn set_operation_mode(&mut self, mode: OperationMode) {
        if self.operation_mode == mode {
            return;
        }
        self.operation_mode = mode;
        // Undocking lifts any touch; republish the sample either way.
        self.set_touch_state(&[]);
        // Default buffer queue geometry; sizes a guest already dequeued are kept.
        let (width, height) = mode.display_size();
        self.display.set_default_size(width, height);
        self.queue_applet_message(AppletMessage::OperationModeChanged);
        self.queue_applet_message(AppletMessage::PerformanceModeChanged);
        if let Some(event) = self.display_resolution_event {
            self.signal_event(event);
        }
    }

    /// Record the proxy kind, which selects the focus message.
    pub(super) fn set_applet_is_application(&mut self, is_application: bool) {
        self.applet_is_application = is_application;
    }

    /// Next AM message; the startup focus transition comes first, once.
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

    pub(super) fn has_applet_message(&self) -> bool {
        !self.applet_focus_announced || !self.applet_messages.is_empty()
    }

    /// Fire an event and wake every parked waiter; each rechecks its own handles.
    pub fn signal_event(&mut self, handle: u64) {
        let Some(event) = self.events.get_mut(&handle) else {
            return;
        };
        // Only a transition wakes waiters.
        if event.signaled {
            return;
        }
        event.signaled = true;
        self.wake_event_waiters();
    }

    /// Wake every thread parked in `svcWaitSynchronization`; each reissues its wait.
    fn wake_event_waiters(&mut self) {
        for thread in &mut self.threads {
            if matches!(thread.state, ThreadState::WaitEvent { .. }) {
                thread.state = ThreadState::Runnable;
            }
        }
    }

    /// Whether `handle` is a fired event or an exited thread; `None` if neither.
    pub(super) fn waitable_signaled(&self, handle: u64) -> Option<bool> {
        if let Some(signaled) = self.event_signaled(handle) {
            return Some(signaled);
        }
        self.threads
            .iter()
            .find(|thread| thread.handle == handle)
            .map(|thread| thread.state == ThreadState::Finished)
    }

    /// Whether `handle` names a fired event; `None` if it is not an event.
    pub fn event_signaled(&self, handle: u64) -> Option<bool> {
        self.events.get(&handle).map(|event| event.signaled)
    }

    /// Consume an auto-clear event's signal after a wait reported it.
    pub(crate) fn consume_event(&mut self, handle: u64) {
        if let Some(event) = self.events.get_mut(&handle) {
            if event.auto_clear {
                event.signaled = false;
            }
        }
    }

    pub(crate) fn clear_event(&mut self, handle: u64) {
        if let Some(event) = self.events.get_mut(&handle) {
            event.signaled = false;
        }
    }

    /// `svcResetSignal`: returns whether the event was signalled; unmodelled handles count as signalled.
    pub(crate) fn reset_signal(&mut self, handle: u64) -> bool {
        match self.events.get_mut(&handle) {
            Some(event) => std::mem::replace(&mut event.signaled, false),
            None => true,
        }
    }

    /// Bind a handle to a service name directly, for tests.
    pub fn register_service_handle(&mut self, handle: u64, name: &str) {
        self.record_handle(handle, name);
    }

    /// The interface a domain object id on `handle` names, or `None` once closed.
    pub fn domain_interface_name(&self, handle: u64, object_id: u32) -> Option<String> {
        self.domain_interface(handle, object_id)
            .map(|s| s.to_owned())
    }

    /// Debug: dump the fake-handle to service-name map.
    pub fn service_handles_snapshot(&self) -> Vec<(u64, String)> {
        let mut v: Vec<(u64, String)> = self
            .service_handles
            .iter()
            .map(|(&h, s)| (h, s.clone()))
            .collect();
        v.sort();
        v
    }

    /// Read a register where 31 is SP.
    #[inline(always)]
    pub fn read_x(&self, idx: u8) -> u64 {
        self.regs[reg_slot(Self::x_slot(idx))]
    }

    /// The slot a register number names when 31 means SP.
    #[inline(always)]
    pub(super) fn x_slot(idx: u8) -> u8 {
        let idx = idx & 0x1F;
        idx + (SP_SLOT as u8 - 31) * u8::from(idx == 31)
    }

    /// The slot a register number names when 31 means `XZR` and is written.
    #[inline(always)]
    pub(super) fn zr_write_slot(idx: u8) -> u8 {
        let idx = idx & 0x1F;
        idx + (ZR_DISCARD as u8 - 31) * u8::from(idx == 31)
    }

    pub fn read_reg(&self, idx: u8) -> u64 {
        self.read_x(idx)
    }

    pub fn read_vreg(&self, idx: u8) -> u128 {
        self.vregs.0[idx as usize]
    }

    pub fn tls_base(&self) -> u32 {
        self.tpidr as u32
    }

    pub fn set_vreg(&mut self, idx: u8, val: u128) {
        self.vregs.0[idx as usize] = val;
    }

    /// Read a register where 31 is `XZR`.
    #[inline(always)]
    fn read_zr(&self, idx: u8) -> u64 {
        self.regs[(idx & 0x1F) as usize]
    }

    /// Write a register where 31 is `XZR`.
    #[inline(always)]
    fn write_zr(&mut self, idx: u8, val: u64) {
        self.regs[reg_slot(Self::zr_write_slot(idx))] = val;
    }

    /// Write a register where 31 is SP.
    #[inline(always)]
    fn write_x(&mut self, idx: u8, val: u64) {
        self.regs[reg_slot(Self::x_slot(idx))] = val;
    }

    /// Read the register file by slot; see [`REG_SLOTS`].
    #[inline(always)]
    pub(super) fn reg_at(&self, slot: u8) -> u64 {
        self.regs[reg_slot(slot)]
    }

    #[inline(always)]
    pub(super) fn set_reg_at(&mut self, slot: u8, val: u64) {
        self.regs[reg_slot(slot)] = val;
    }

    pub fn set_reg(&mut self, idx: u8, val: u64) {
        self.write_zr(idx, val);
    }

    pub fn read_u32_reg(&self, idx: u8) -> u32 {
        self.read_zr(idx) as u32
    }

    pub fn set_pc_and_sp(&mut self, pc: u32, sp: u64) {
        self.pc = pc;
        match self.mode {
            ExecMode::A64 => self.regs[SP_SLOT] = sp,
            ExecMode::A32 => self.regs[13] = sp,
        }
    }

    /// Publish host gamepad state to [`crate::INPUT_ADDR`] and, once mapped, hid shared memory.
    /// `buttons` is a `HidNpadButton` mask; sticks are -32768..32767 with up positive.
    /// Stick pseudo-buttons are derived here.
    pub fn set_gamepad_state(
        &mut self,
        buttons: u64,
        stick_lx: i32,
        stick_ly: i32,
        stick_rx: i32,
        stick_ry: i32,
    ) {
        self.last_gamepad = (buttons, stick_lx, stick_ly, stick_rx, stick_ry);
        let buttons = buttons | Self::stick_pseudo_buttons(stick_lx, stick_ly, stick_rx, stick_ry);

        // Host-to-guest register: a u64 mask, then two analog sticks.
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

    /// Publish a fresh `hid` sample when the 200 Hz timer comes round.
    pub(super) fn hid_tick(&mut self) {
        if self.hid_shmem_addr == 0
            || self.cycles.wrapping_sub(self.last_hid_cycles) < HID_SAMPLE_PERIOD_CYCLES
        {
            return;
        }
        self.last_hid_cycles = self.cycles;
        let (buttons, lx, ly, rx, ry) = self.last_gamepad;
        self.set_gamepad_state(buttons, lx, ly, rx, ry);
        let touches = std::mem::take(&mut self.last_touches);
        self.set_touch_state(&touches);
    }

    /// `HidNpadButton_StickL*`/`StickR*` bits derived from stick deflection.
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

    /// Mirror the pad into `HidSharedMemory` as both player 1 and handheld, in the
    /// styles the title requested (see [`NPAD_PRESENTATIONS`]).
    fn write_hid_gamepad_state(&mut self, buttons: u64, lx: i32, ly: i32, rx: i32, ry: i32) {
        use hid_shmem as h;
        self.sample_counter = self.sample_counter.wrapping_add(1);
        let sample = self.sample_counter;
        let supported = self.npad_style_set;
        // Published on mapping, before the guest reads a pad.
        self.write_npad_condition();
        self.write_npad_slot(
            0,
            npad_presentation_for(supported),
            sample,
            buttons,
            (lx, ly, rx, ry),
        );
        // The handheld slot is always published.
        self.write_npad_slot(
            h::HANDHELD_SLOT,
            NPAD_HANDHELD,
            sample,
            buttons,
            (lx, ly, rx, ry),
        );
    }

    /// Publish `nn::hid::NpadCondition`, including the stored joy-con hold type.
    pub(super) fn write_npad_condition(&mut self) {
        use hid_shmem as h;
        if self.hid_shmem_addr == 0 {
            return;
        }
        let at = self.hid_shmem_addr.wrapping_add(h::NPAD_CONDITION);
        let _ = self.mem.write_u32(at + h::NPAD_CONDITION_INITIALIZED, 1);
        let _ = self.mem.write_u32(
            at + h::NPAD_CONDITION_HOLD_TYPE,
            self.npad_joy_hold_type as u32,
        );
        let _ = self.mem.write_u32(at + h::NPAD_CONDITION_VALID, 1);
    }

    fn write_npad_slot(
        &mut self,
        slot: u32,
        presentation: NpadPresentation,
        sample: u64,
        buttons: u64,
        sticks: (i32, i32, i32, i32),
    ) {
        use hid_shmem as h;
        let base = self
            .hid_shmem_addr
            .wrapping_add(h::NPAD)
            .wrapping_add(slot.wrapping_mul(h::ENTRY_SIZE));
        let _ = self.mem.write_u32(base + h::STYLE_SET, presentation.style);
        let _ = self
            .mem
            .write_u32(base + h::JOY_ASSIGNMENT_MODE, presentation.joy_assignment);
        let _ = self
            .mem
            .write_u32(base + h::DEVICE_TYPE, presentation.device_type);

        // Report external power and a full battery for the pad and each half; zero reads as flat.
        let _ = self.mem.write_u32(
            base + h::SYSTEM_PROPERTIES,
            h::SYSTEM_PROP_POWER_CONNECTED | h::SYSTEM_PROP_FULL_BUTTONS,
        );
        for info in 0..h::POWER_INFO_COUNT {
            let _ = self
                .mem
                .write_u32(base + h::BATTERY_LEVEL + info * 4, h::BATTERY_FULL);
        }

        self.write_npad_lifo(
            base.wrapping_add(presentation.lifo),
            sample,
            buttons,
            sticks,
            presentation.attributes,
        );
        // SystemExt: a second copy every pad carries; the Home Menu reads only this LIFO.
        self.write_npad_lifo(
            base.wrapping_add(h::SYSTEM_EXT_LIFO),
            sample,
            buttons,
            sticks,
            presentation.attributes,
        );
    }

    /// Publish one state into a `HidNpadCommonLifo`. The sampling number is doubled
    /// because bit 0 is the seqlock's "being written" flag.
    fn write_npad_lifo(
        &mut self,
        lifo: u32,
        sample: u64,
        buttons: u64,
        sticks: (i32, i32, i32, i32),
        attributes: u32,
    ) {
        use hid_shmem as h;
        let (lx, ly, rx, ry) = sticks;
        let _ = self
            .mem
            .write_u64(lifo + h::LIFO_BUFFER_COUNT, h::LIFO_CAPACITY);
        let _ = self.mem.write_u64(lifo + h::LIFO_TAIL, 0);
        let _ = self.mem.write_u64(lifo + h::LIFO_COUNT, 1);

        let entry = lifo.wrapping_add(h::LIFO_STORAGE);
        let _ = self
            .mem
            .write_u64(entry + h::STORAGE_SAMPLING_NUMBER, sample << 1);
        let _ = self.mem.write_u64(entry + h::STATE_SAMPLING_NUMBER, sample);
        let _ = self.mem.write_u64(entry + h::STATE_BUTTONS, buttons);
        let _ = self.mem.write_u32(entry + h::STATE_STICK_L, lx as u32);
        let _ = self.mem.write_u32(entry + h::STATE_STICK_L + 4, ly as u32);
        let _ = self.mem.write_u32(entry + h::STATE_STICK_R, rx as u32);
        let _ = self.mem.write_u32(entry + h::STATE_STICK_R + 4, ry as u32);
        let _ = self.mem.write_u32(entry + h::STATE_ATTRIBUTES, attributes);
    }

    /// Publish touchscreen contacts for `hidGetTouchScreenStates`. New ids get
    /// `start_touch`; lifted ids are published once more with `end_touch`. Docked, no
    /// contacts are reported, but the sample still advances.
    pub fn set_touch_state(&mut self, touches: &[TouchPoint]) {
        self.last_touches = touches.to_vec();
        if self.hid_shmem_addr == 0 {
            return;
        }
        use hid_shmem as h;
        self.touch_sample_counter = self.touch_sample_counter.wrapping_add(1);
        let sample = self.touch_sample_counter;

        let lifo = self.hid_shmem_addr.wrapping_add(h::TOUCH_SCREEN);
        let _ = self
            .mem
            .write_u64(lifo + h::LIFO_BUFFER_COUNT, h::LIFO_CAPACITY);
        let _ = self.mem.write_u64(lifo + h::LIFO_TAIL, 0);
        let _ = self.mem.write_u64(lifo + h::LIFO_COUNT, 1);

        let storage = lifo.wrapping_add(h::LIFO_STORAGE);
        let _ = self
            .mem
            .write_u64(storage + h::STORAGE_SAMPLING_NUMBER, sample << 1);
        let state = storage.wrapping_add(h::TOUCH_STATE);
        let _ = self.mem.write_u64(state + h::TOUCH_SAMPLING_NUMBER, sample);

        // Docked: every contact is gone.
        let down: &[TouchPoint] = match self.operation_mode {
            OperationMode::Handheld => touches,
            OperationMode::Docked => &[],
        };
        // Contacts down now, then those lifted since the last sample.
        let mut published: Vec<(TouchPoint, u32)> = Vec::with_capacity(TOUCH_MAX);
        for touch in down.iter().take(TOUCH_MAX) {
            let held = self
                .touch_down
                .iter()
                .any(|prev| prev.finger_id == touch.finger_id);
            let attributes = if held { 0 } else { h::TOUCH_ATTR_START };
            published.push((*touch, attributes));
        }
        for prev in &self.touch_down {
            if published.len() == TOUCH_MAX {
                break;
            }
            if !down.iter().any(|touch| touch.finger_id == prev.finger_id) {
                published.push((*prev, h::TOUCH_ATTR_END));
            }
        }
        self.touch_down = down.iter().take(TOUCH_MAX).copied().collect();

        let count = published.len();
        let _ = self.mem.write_u32(state + h::TOUCH_COUNT, count as u32);
        let slot = |i: usize| state + h::TOUCH_TOUCHES + i as u32 * h::TOUCH_SIZE;
        for (i, (touch, attributes)) in published.iter().enumerate() {
            let e = slot(i);
            // delta_time is not measured.
            let _ = self.mem.write_u64(e + h::TOUCH_DELTA_TIME, 0);
            let _ = self.mem.write_u32(e + h::TOUCH_ATTRIBUTES, *attributes);
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
        // Clear slots a previous larger sample left.
        for i in count..self.touch_published {
            let e = slot(i);
            for off in (0..h::TOUCH_SIZE).step_by(4) {
                let _ = self.mem.write_u32(e + off, 0);
            }
        }
        self.touch_published = count;
    }

    /// Rumble `(low, high)` amplitudes in 0.0..=1.0.
    pub fn vibration(&self) -> (f32, f32) {
        self.vibration
    }

    pub(crate) fn set_vibration(&mut self, low: f32, high: f32) {
        let clamp = |v: f32| {
            if v.is_finite() {
                v.clamp(0.0, 1.0)
            } else {
                0.0
            }
        };
        self.vibration = (clamp(low), clamp(high));
    }

    pub fn hid_shmem_addr(&self) -> u32 {
        self.hid_shmem_addr
    }

    /// `(sample_rate, channels)` of [`Cpu::take_audio`]'s samples; `(0, 0)` before a device opens.
    pub fn audio_format(&self) -> (u32, u32) {
        self.audio_format
    }

    /// Move up to `out.len()` queued interleaved samples into `out`; returns the count.
    pub fn take_audio(&mut self, out: &mut [i16]) -> usize {
        let n = out.len().min(self.audio_pcm.len());
        for slot in out.iter_mut().take(n) {
            *slot = self.audio_pcm.pop_front().unwrap_or(0);
        }
        self.audio_taken += n as u64;
        n
    }

    /// Every open audio device and renderer, and the samples queued for the host.
    pub fn audio_activity(&self) -> AudioActivity {
        let mut outputs: Vec<AudioOutActivity> = self
            .audio_outs
            .iter()
            .map(|(&handle, device)| AudioOutActivity {
                handle,
                sample_rate: device.sample_rate,
                channels: device.channel_count,
                started: device.started,
                volume: device.volume,
                appended_buffers: device.appended_buffers,
                appended_frames: device.appended_frames,
                released_buffers: device.released_buffers,
                pending_buffers: device.queued.len() as u64,
                discarded_frames: device.discarded_frames,
                unplayable_buffers: device.unplayable_buffers,
            })
            .collect();
        outputs.sort_by_key(|device| device.handle);
        let mut renderers: Vec<AudioRendererActivity> = self
            .audren_renderers
            .iter()
            .map(|(&handle, renderer)| AudioRendererActivity {
                handle,
                sample_rate: renderer.sample_rate,
                started: renderer.started,
                updates: renderer.updates,
                rendered_frames: renderer.elapsed_frames,
                voices: renderer.voices.len() as u32,
                voices_playing: renderer
                    .voices
                    .iter()
                    .filter(|voice| voice.in_use && voice.playing && voice.remaining > 0)
                    .count() as u32,
                sink_channels: renderer.sink.as_ref().map_or(0, |sink| sink.channels),
            })
            .collect();
        renderers.sort_by_key(|renderer| renderer.handle);
        AudioActivity {
            sample_rate: self.audio_format.0,
            channels: self.audio_format.1,
            produced: self.audio_produced,
            taken: self.audio_taken,
            dropped: self.audio_dropped,
            backlog: self.audio_pcm.len() as u64,
            outputs,
            renderers,
        }
    }

    /// Queue interleaved PCM, dropping the oldest past [`Cpu::AUDIO_QUEUE_LIMIT`].
    pub(crate) fn queue_audio(&mut self, samples: impl Iterator<Item = i16>) {
        let before = self.audio_pcm.len();
        self.audio_pcm.extend(samples);
        self.audio_produced += (self.audio_pcm.len() - before) as u64;
        let over = self.audio_pcm.len().saturating_sub(Self::AUDIO_QUEUE_LIMIT);
        self.audio_pcm.drain(..over);
        self.audio_dropped += over as u64;
    }

    /// About a second of 48 kHz stereo.
    pub(crate) const AUDIO_QUEUE_LIMIT: usize = 48_000 * 2;

    /// Set the font `pl:u` serves for every shared font type (TrueType/OpenType).
    pub fn set_shared_font(&mut self, font: Vec<u8>) {
        self.shared_font = font;
        self.pl_shmem_image.clear();
        self.shared_font_regions.clear();
        // Refill in place if the guest already mapped the region.
        if self.pl_shmem_addr != 0 {
            self.write_shared_font(self.pl_shmem_addr);
        }
    }

    pub fn shared_font_len(&self) -> usize {
        self.shared_font.len()
    }

    /// Assemble pl's shared memory lazily: firmware `.bfttf` fonts, or the host font
    /// wrapped the same way in every slot.
    pub(super) fn build_shared_fonts(&mut self) {
        if !self.shared_font_regions.is_empty() {
            return;
        }
        // Read each archive at most once: two hold two fonts.
        let mut archives: IdMap<u64, Vec<u8>> = IdMap::default();
        for (id, _) in SHARED_FONTS {
            if archives.contains_key(&id) {
                continue;
            }
            let Some(src) = self.data_archives.get(&id) else {
                continue;
            };
            let mut image = vec![0u8; src.len() as usize];
            if src.read_at(0, &mut image).is_err() {
                continue;
            }
            archives.insert(id, image);
        }

        for (id, name) in SHARED_FONTS {
            let font = archives
                .get(&id)
                .and_then(|image| crate::romfs::RomFs::parse(image).ok()?.read_path(name))
                .and_then(decode_bfttf);
            let Some(font) = font else { continue };
            self.push_shared_font(&font);
        }

        if self.shared_font_regions.is_empty() && !self.shared_font.is_empty() {
            // No firmware fonts: the host font stands in for every type.
            let font = decode_bfttf(&encode_bfttf(&self.shared_font.clone()));
            if let Some(font) = font {
                for _ in 0..SHARED_FONTS.len() {
                    self.push_shared_font(&font);
                }
            }
        }
        if crate::trace::enabled(crate::trace::Trace::Font) {
            crate::traceln!("[pl] {} bytes of shared font", self.pl_shmem_image.len());
            for (i, region) in self.shared_font_regions.iter().enumerate() {
                crate::traceln!(
                    "[pl]  type {i}: offset={:#x} size={:#x}",
                    region.offset,
                    region.size
                );
            }
        }
    }

    /// Append a decoded font; one that does not fit is dropped, not truncated.
    fn push_shared_font(&mut self, font: &[u8]) {
        let offset = self.pl_shmem_image.len();
        if offset + font.len() > PL_SHMEM_SIZE as usize {
            return;
        }
        self.pl_shmem_image.extend_from_slice(font);
        self.shared_font_regions.push(FontRegion {
            offset: (offset + BFTTF_HEADER) as u32,
            size: (font.len() - BFTTF_HEADER) as u32,
        });
    }

    #[cfg(test)]
    pub(super) fn shared_font_image(&mut self) -> &[u8] {
        self.build_shared_fonts();
        &self.pl_shmem_image
    }

    pub(super) fn shared_font_regions(&mut self) -> &[FontRegion] {
        self.build_shared_fonts();
        &self.shared_font_regions
    }

    /// Copy the shared fonts into pl's shared memory at `addr`.
    pub(super) fn write_shared_font(&mut self, addr: u32) {
        self.build_shared_fonts();
        let image = std::mem::take(&mut self.pl_shmem_image);
        let _ = self.mem.map(addr, &image);
        self.pl_shmem_image = image;
    }

    /// Set the POSIX time (UTC) `time:u`/`time:s` report; the epoch until the host sets it.
    pub fn set_unix_time(&mut self, seconds: i64) {
        self.unix_time = seconds;
    }

    pub fn unix_time(&self) -> i64 {
        self.unix_time
    }

    /// Set the battery `psm` reports; full and charging until the host sets it.
    pub fn set_battery(&mut self, percent: u8, charging: bool) {
        self.battery_percent = percent.min(100);
        self.battery_charging = charging;
    }

    pub fn battery(&self) -> (u8, bool) {
        (self.battery_percent, self.battery_charging)
    }

    /// Set the NACP save quota, passed through as declared.
    pub fn set_save_data_quota(&mut self, quota: fs::SaveDataQuota) {
        self.save_data_quota = quota;
    }

    pub fn save_data_quota(&self) -> fs::SaveDataQuota {
        self.save_data_quota
    }

    /// Set the NPDM `system_resource_size`, which selects the [`MemoryLayout`].
    /// Call before [`Cpu::boot_retail_program`].
    pub fn set_system_resource_size(&mut self, size: u32) {
        self.system_resource_size = size;
        self.refresh_memory_layout();
    }

    /// Re-choose the layout from the program id and system resource size, set in either order.
    fn refresh_memory_layout(&mut self) {
        self.memory_layout = MemoryLayout::for_program(self.program_id, self.system_resource_size);
    }

    pub fn memory_layout(&self) -> MemoryLayout {
        self.memory_layout
    }

    pub fn set_program_id(&mut self, program_id: u64) {
        self.program_id = program_id;
        self.refresh_memory_layout();
    }

    pub fn program_id(&self) -> u64 {
        self.program_id
    }

    /// What a library applet pushed back before exiting, oldest first.
    pub fn library_applet_results(&self) -> &[Vec<u8>] {
        &self.am_out_data
    }

    /// What a library applet pushed through `PushInteractiveOutData`, oldest first.
    pub fn library_applet_interactive_messages(&self) -> &[Vec<u8>] {
        &self.am_interactive_out
    }

    /// Answer the applet through `PopInteractiveInData`, firing its event.
    pub fn push_applet_interactive_in_data(&mut self, data: Vec<u8>) {
        self.am_interactive_in.push_back(data);
        self.refresh_applet_pop_events();
    }

    /// A pseudo-random u64 for `csrng`: splitmix64 seeded from the clock. Not a CSPRNG.
    pub(crate) fn next_random_u64(&mut self) -> u64 {
        if self.rng_state == 0 {
            self.rng_state =
                (self.unix_time as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA076_1D64_78BD_642F;
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
        (CONDITION_MASKS[(cond & 0xF) as usize] >> (self.nzcv >> 28)) & 1 == 1
    }

    #[inline(always)]
    fn mask(sf: bool) -> u64 {
        if sf {
            u64::MAX
        } else {
            u32::MAX as u64
        }
    }

    /// `a + b + carry_in` as (result, carry-out, overflow), with operands masked to the operation size.
    #[inline(always)]
    fn add_carry_overflow(a: u64, b: u64, carry_in: u64, sf: bool) -> (u64, u32, u32) {
        let mask = Self::mask(sf);
        let a = a & mask;
        let b = b & mask;
        // Two exclusive `overflowing_add`s form the carry chain without u128 or a width branch.
        let (sum, c1) = a.overflowing_add(b);
        let (sum, c2) = sum.overflowing_add(carry_in);
        let carry = if sf {
            u32::from(c1 | c2)
        } else {
            ((sum >> 32) & 1) as u32
        };
        let result = sum & mask;
        // Both operands the same sign and the result a different one.
        let sign = 1u64 << (if sf { 63 } else { 31 });
        let overflow = u32::from((!(a ^ b) & (a ^ result) & sign) != 0);
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

    /// The ADD/SUB core. `sp_form`: register 31 is SP (immediate and extended forms)
    /// rather than XZR (shifted-register form).
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
        // Rd=31 is SP only for the non-flag-setting immediate and extended forms.
        let rd = if set_flags || !sp_form {
            Self::zr_write_slot(rd)
        } else {
            Self::x_slot(rd)
        };
        let rn = if sp_form { Self::x_slot(rn) } else { rn & 0x1F };
        // Subtraction is addition of the inverted operand with a carry in.
        self.add_sub_pre(
            rd,
            rn,
            if sub { !rhs } else { rhs },
            u8::from(sub),
            set_flags,
            sf,
        );
    }

    /// ADD/SUB with direction folded into `rhs`/`carry` and register 31 resolved; shared
    /// by [`Cpu::add_sub`] and the block translator.
    #[inline(always)]
    pub(super) fn add_sub_pre(
        &mut self,
        rd: u8,
        rn: u8,
        rhs: u64,
        carry: u8,
        set_flags: bool,
        sf: bool,
    ) {
        let a = self.reg_at(rn) & Self::mask(sf);
        let (result, c, v) = Self::add_carry_overflow(a, rhs, u64::from(carry), sf);
        if set_flags {
            self.set_nzcv_from_alu(result, sf, c, v);
        }
        self.set_reg_at(rd, result);
    }

    // ---- main execution ----

    /// Execute a single instruction.
    pub fn step(&mut self) -> Result<()> {
        self.step_inner()
    }

    /// Body of [`Cpu::step`], inlined into [`Cpu::run`]'s loop.
    #[inline(always)]
    fn step_inner(&mut self) -> Result<()> {
        if self.halted {
            return Err(Error::Cpu("attempted to step a halted CPU".into()));
        }
        // Preemption point; `yield_thread` is a no-op when nothing else can run.
        self.slice_used += 1;
        if self.slice_used >= TIME_SLICE {
            self.slice_used = 0;
            self.yield_thread();
        }
        self.sweep_timed_waits();
        let pc = self.pc;
        let insn = match self.mem.fetch(pc) {
            Ok(i) => i,
            Err(e) => {
                self.record_fault(&e, pc, 0);
                return Err(e);
            }
        };
        let next_pc = pc.wrapping_add(4);
        self.record_run(pc, 1);
        let result = match self.mode {
            ExecMode::A64 => self.execute(insn, next_pc),
            ExecMode::A32 => self.execute_a32(insn),
        };
        if self.trace_enabled {
            self.trace_line(&format!(
                "{:08x}: {:08x}  {}\n",
                pc,
                insn,
                self.disassemble_for_mode(insn)
            ));
        }
        if let Err(e) = &result {
            self.record_fault(e, pc, insn);
        }
        self.retire();
        result
    }

    fn record_fault(&mut self, e: &Error, pc: u32, insn: u32) {
        // Traces from parts without a `Cpu` belong before the fault.
        self.absorb_traces();
        // Unmarked, so the separator does not grade the previous line.
        self.trace_line("\n");
        // The dump and trail below inherit the fault's level.
        self.trace_marked(
            Level::Error,
            &format!(
                "=== FAULT ===\n{}\n  at pc={:#010x} insn={:#010x}  {}",
                e,
                pc,
                insn,
                if insn == 0 {
                    String::new()
                } else {
                    self.disassemble_for_mode(insn)
                }
            ),
        );
        self.trace_regs(pc);
        self.trace_trail();
    }

    /// Trace the run-up to the current PC, expanding the recorded runs.
    pub(super) fn trace_trail(&mut self) {
        let trail = self.trail_text();
        if !trail.is_empty() {
            self.trace_line(&trail);
        }
    }

    /// The trail as text: a heading and one disassembled instruction per line.
    pub(super) fn trail_text(&self) -> String {
        let runs = self.recent_len.min(RECENT_LEN);
        if runs == 0 {
            return String::new();
        }
        let first = self.recent_len.wrapping_sub(runs) % RECENT_LEN;
        let mut trail: Vec<(u32, u32)> = Vec::new();
        for i in 0..runs {
            let (start, count) = self.recent[(first + i) % RECENT_LEN];
            for step in 0..count {
                let at = start.wrapping_add(4 * step);
                let word = self.mem.fetch(at).unwrap_or(0);
                trail.push((at, word));
            }
        }
        let shown = trail.len().min(RECENT_LEN);
        let mut text = format!("--- last {shown} instructions ---\n");
        for &(ipc, iinsn) in &trail[trail.len() - shown..] {
            text.push_str(&format!(
                "{:08x}: {:08x}  {}\n",
                ipc,
                iinsn,
                self.disassemble_for_mode(iinsn)
            ));
        }
        text
    }

    /// Index of the running thread, for host-side sampling profilers.
    pub fn current_thread_index(&self) -> usize {
        self.current_thread
    }

    /// Debugging lever: make every blocked thread runnable; returns the count.
    pub fn wake_all_blocked(&mut self) -> usize {
        let mut woken = 0;
        for index in 0..self.threads.len() {
            match self.threads[index].state {
                ThreadState::WaitKey { mutex, .. } => self.wake_condvar_waiter(index, mutex),
                ThreadState::WaitMutex(_) | ThreadState::WaitAddress { .. } => {
                    self.threads[index].state = ThreadState::Runnable;
                }
                _ => continue,
            }
            woken += 1;
        }
        woken
    }

    /// Every `(interface, command)` reported as unimplemented, sorted.
    pub fn unimplemented_ipc(&self) -> Vec<(String, Option<u32>)> {
        let mut all: Vec<(String, Option<u32>)> = self.unimplemented_ipc.iter().cloned().collect();
        all.sort();
        all
    }

    /// Every `(interface, command)` answered by a stub; see [`Cpu::warn_stub`].
    pub fn stubbed_ipc(&self) -> Vec<(String, Option<u32>)> {
        let mut all: Vec<(String, Option<u32>)> = self.stubbed_ipc.iter().cloned().collect();
        all.sort();
        all
    }

    /// Count one failed nvdrv ioctl towards the next [`Cpu::take_nv_errors`].
    pub(super) fn count_nv_error(&mut self, node: &str, request: u32, error: u32) {
        /// Distinct failures held between readings.
        const CAP: usize = 64;
        if let Some(calls) = self
            .nv_errors
            .iter_mut()
            .find(|((n, r, e), _)| n == node && *r == request && *e == error)
            .map(|(_, calls)| calls)
        {
            *calls += 1;
        } else if self.nv_errors.len() < CAP {
            self.nv_errors.insert((node.to_owned(), request, error), 1);
        }
    }

    /// nvdrv ioctl failures since the last call: node, request, error, count.
    pub fn take_nv_errors(&mut self) -> Vec<(String, u32, u32, u64)> {
        std::mem::take(&mut self.nv_errors)
            .into_iter()
            .map(|((node, request, error), calls)| (node, request, error, calls))
            .collect()
    }

    /// Requested `HidNpadStyleTag` bits (0 before any) and the presented style.
    pub fn npad_styles(&self) -> (u32, u32) {
        (
            self.npad_style_set,
            npad_presentation_for(self.npad_style_set).style,
        )
    }

    /// Debugging lever: start every created-but-unstarted thread; returns the count.
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
        // Report the implicit main thread for a program that never created one.
        if self.threads.is_empty() {
            out.push_str(&format!(
                "  [0]* handle={MAIN_THREAD_HANDLE:#x} state=Runnable paused=false pc={:#x}\n  \
                 (the main thread, which has no slot of its own until the guest creates a \
                 second)\n",
                self.pc
            ));
            return out;
        }
        for (index, thread) in self.threads.iter().enumerate() {
            let running = index == self.current_thread;
            out.push_str(&format!(
                "  [{index}]{} handle={:#x} priority={} state={:?} paused={} pc={:#x}\n",
                if running { "*" } else { " " },
                thread.handle,
                thread.priority,
                thread.state,
                thread.paused,
                if running { self.pc } else { thread.pc },
            ));
        }
        out
    }

    /// Record a user-facing diagnostic: to stderr on the host and to the trace buffer
    /// the browser drains, with a level the page colours by.
    pub fn diagnostic(&mut self, level: Level, line: &str) {
        // Not `traceln!`: that would feed the pending sink a second time.
        #[cfg(not(target_arch = "wasm32"))]
        eprintln!("{line}");
        self.absorb_traces();
        self.trace_marked(level, line);
    }

    /// Fold in traces from parts without a `Cpu` (rasterizer, shader translator, texture
    /// decoder); called before diagnostics, faults and drains to keep ordering.
    pub fn absorb_traces(&mut self) {
        let pending = crate::trace::take_pending();
        if pending.is_empty() {
            return;
        }
        self.note_dropped_trace();
        self.trace.extend_from_slice(&pending);
        self.trim_trace();
    }

    /// Append an unmarked line, a continuation of the previous one.
    fn trace_line(&mut self, line: &str) {
        self.note_dropped_trace();
        self.trace.extend_from_slice(line.as_bytes());
        self.trim_trace();
    }

    /// Append a line at `level`, inherited by following unmarked lines.
    fn trace_marked(&mut self, level: Level, line: &str) {
        self.note_dropped_trace();
        self.trace.push(level.marker());
        self.trace.extend_from_slice(line.as_bytes());
        if !line.ends_with('\n') {
            self.trace.push(b'\n');
        }
        self.trim_trace();
    }

    /// Note lost text once per loss, where it was lost.
    fn note_dropped_trace(&mut self) {
        if !self.trace_dropped {
            return;
        }
        self.trace_dropped = false;
        self.trace.push(Level::Warn.marker());
        self.trace
            .extend_from_slice(b"[trace] the buffer filled; older lines above were dropped\n");
    }

    /// Drop the oldest quarter of the trace at a line boundary to get under the cap.
    fn trim_trace(&mut self) {
        if self.trace.len() <= self.trace_cap {
            return;
        }
        let least = self.trace.len() - self.trace_cap;
        let want = (least + self.trace_cap / 4).min(self.trace.len());
        let cut = match self.trace[want..].iter().position(|&b| b == b'\n') {
            Some(at) => want + at + 1,
            // A single line longer than the buffer: drop it all.
            None => self.trace.len(),
        };
        self.trace.drain(..cut);
        self.trace_dropped = true;
    }

    fn trace_regs(&mut self, pc: u32) {
        let dump = self.reg_dump();
        self.trace_line(&dump);
        let _ = pc;
    }

    /// Walk the frame-pointer chain and return return addresses, innermost first.
    /// Frames must lie above SP and addresses must follow a call, since not all code keeps x29.
    pub fn backtrace(&self, depth: usize) -> Vec<u32> {
        self.walk_frames(&self.regs, self.mode, depth)
    }

    /// [`Cpu::backtrace`] over any register file, live or saved.
    pub(super) fn walk_frames(
        &self,
        regs: &[u64; REG_FILE],
        mode: ExecMode,
        depth: usize,
    ) -> Vec<u32> {
        // x29/x30 with 16-byte frames in A64, r11/r14 with 8-byte frames in AArch32.
        let (mut fp, lr, sp, width) = match mode {
            ExecMode::A64 => (regs[29] as u32, regs[30] as u32, regs[SP_SLOT] as u32, 8),
            ExecMode::A32 => (regs[11] as u32, regs[14] as u32, regs[13] as u32, 4),
        };
        let mut out = Vec::with_capacity(depth + 1);
        if self.is_return_address(lr, mode) {
            out.push(lr);
        }
        // The first frame is at or above SP; `next_fp <= fp` keeps later ones above it.
        if fp < sp {
            return out;
        }
        for _ in 0..depth {
            if !fp.is_multiple_of(width) {
                break;
            }
            let read = |at: u32| match mode {
                ExecMode::A64 => self.mem.read_u64(at).map(|v| v as u32),
                ExecMode::A32 => self.mem.read_u32(at),
            };
            let (next_fp, lr) = match (read(fp), read(fp.wrapping_add(width))) {
                (Ok(next_fp), Ok(lr)) => (next_fp, lr),
                _ => break,
            };
            if next_fp <= fp || !self.is_return_address(lr, mode) {
                break;
            }
            out.push(lr);
            fp = next_fp;
        }
        out
    }

    /// Whether `addr` follows a call instruction or is a thread/host return stub.
    fn is_return_address(&self, addr: u32, mode: ExecMode) -> bool {
        if addr == THREAD_EXIT_TRAMPOLINE || addr == SELF_RETURN_TRAMPOLINE {
            return true;
        }
        if addr < 4 || !addr.is_multiple_of(4) {
            return false;
        }
        let Ok(call) = self.mem.read_u32(addr - 4) else {
            return false;
        };
        match mode {
            // BL, then BLR.
            ExecMode::A64 => call & 0xFC00_0000 == 0x9400_0000 || call & 0xFFFF_FC1F == 0xD63F_0000,
            // BL (cond != 1111), BLX to an immediate, then BLX to a register.
            ExecMode::A32 => {
                (call & 0x0F00_0000 == 0x0B00_0000 && call >> 28 != 0xF)
                    || call & 0xFE00_0000 == 0xFA00_0000
                    || call & 0x0FFF_FFF0 == 0x012F_FF30
            }
        }
    }

    /// One general-purpose register, for host-side debuggers.
    pub fn reg(&self, i: usize) -> u64 {
        self.regs[i]
    }

    /// A register snapshot named for the current state (A64 or AArch32).
    pub fn reg_dump(&self) -> String {
        use std::fmt::Write;
        let mut s = String::with_capacity(1024);
        let n = (self.nzcv >> 31) & 1;
        let z = (self.nzcv >> 30) & 1;
        let c = (self.nzcv >> 29) & 1;
        let v = (self.nzcv >> 28) & 1;
        match self.mode {
            ExecMode::A64 => {
                let _ = writeln!(
                    s,
                    "pc={:#010x}  sp={:#018x}  nzcv=N:{n} Z:{z} C:{c} V:{v}",
                    self.pc, self.regs[SP_SLOT]
                );
                for i in 0..31 {
                    let _ = write!(s, "x{:<2}={:#018x}  ", i, self.regs[i]);
                    if i % 4 == 3 {
                        let _ = writeln!(s);
                    }
                }
            }
            ExecMode::A32 => {
                let _ = writeln!(
                    s,
                    "pc={:#010x}  nzcv=N:{n} Z:{z} C:{c} V:{v} Q:{} ge={:#06b}",
                    self.pc,
                    u8::from(self.cpsr_q),
                    self.cpsr_ge
                );
                for i in 0..15 {
                    let name = match i {
                        13 => "sp".to_string(),
                        14 => "lr".to_string(),
                        _ => format!("r{i}"),
                    };
                    let _ = write!(s, "{name:<3}={:#010x}  ", self.regs[i] as u32);
                    if i % 4 == 3 {
                        let _ = writeln!(s);
                    }
                }
                let _ = write!(s, "pc ={:#010x}  ", self.pc);
            }
        }
        let _ = writeln!(s);
        s
    }

    /// Run up to `max_steps` instructions, stopping early on halt or error. Uses the JIT
    /// when enabled, except with full tracing, which needs the interpreter.
    pub fn run(&mut self, max_steps: u64) -> Result<RunReport> {
        self.complete_pending_present();
        if self.jit_enabled && !self.trace_enabled && self.mode == ExecMode::A64 {
            return self.run_jit(max_steps);
        }
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

    /// Route an instruction by its top-level group (bits 28:25) to that group's decoder
    /// first, falling back to [`Cpu::execute_chain`].
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
            // SIMD and FP: scalar FP top bytes are 0x1E/0x1F and 0x9E/0x9F; ask that decoder first.
            0x7 | 0xF => {
                let scalar_fp = matches!((insn >> 24) & 0xFF, 0x1E | 0x1F | 0x9E | 0x9F);
                #[allow(clippy::if_same_then_else)] // the order is the point
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
            #[allow(clippy::collapsible_match)] // no fallible call in a guard
            0xA | 0xB => {
                if self.try_branch_or_system(insn, next_pc)? {
                    return Ok(());
                }
            }
            // Reserved and SVE groups, left to the chain.
            _ => {}
        }
        self.execute_chain(insn, next_pc)
    }

    /// ADR/ADRP: bits[28:24] == 10000; bits[30:29] are immlo.
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

    /// `LDR Xt, label` and friends.
    fn try_load_literal(&mut self, insn: u32) -> Result<bool> {
        if ((insn >> 27) & 0b111) != 0b011
            || ((insn >> 26) & 1) != 0
            || ((insn >> 24) & 0b11) != 0b00
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
            // PRFM: a prefetch hint.
            _ => {}
        }
        Ok(true)
    }

    /// Branches, exceptions and system (bits 28:25 = 101x), dispatched on the top byte
    /// in order of frequency. Returns whether handled; handlers set `self.pc`.
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
                        // BLR: read the target before linking, since it may be x30.
                        let target = self.read_zr(rn) as u32;
                        self.write_zr(30, next_pc as u64);
                        self.pc = target;
                        Ok(true)
                    }
                    0b0010 => {
                        // RET to 0 is a homebrew exit path; redirect to the exit trampoline.
                        let tgt = self.read_zr(rn) as u32;
                        self.pc = if tgt == 0 {
                            SELF_RETURN_TRAMPOLINE
                        } else {
                            tgt
                        };
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
                        // Retire the SVC first: a thread switch installs the incoming thread's PC.
                        self.pc = next_pc;
                        self.syscall(imm)?;
                        Ok(true)
                    } else {
                        Err(Error::Cpu(format!(
                            "unimplemented HVC/SMC at {:#x}",
                            self.pc
                        )))
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

    /// Fallback over the whole encoding space, kept out of line to keep `execute` small.
    /// Order: branch/system, load literal, load/store, SIMD, scalar FP, PC-relative,
    /// DP immediate, DP register.
    #[cold]
    #[inline(never)]
    fn execute_chain(&mut self, insn: u32, mut next_pc: u32) -> Result<()> {
        if self.try_branch_or_system(insn, next_pc)? {
            return Ok(());
        }

        if self.try_load_literal(insn)? {
            self.pc = next_pc;
            return Ok(());
        }

        if self.try_load_store(insn, &mut next_pc)? {
            self.pc = next_pc;
            return Ok(());
        }

        if self.try_simd(insn)? {
            self.pc = next_pc;
            return Ok(());
        }

        if self.try_fp(insn)? {
            self.pc = next_pc;
            return Ok(());
        }

        if self.try_pc_relative(insn) {
            self.pc = next_pc;
            return Ok(());
        }

        if self.try_data_proc_imm(insn, &mut next_pc)? {
            self.pc = next_pc;
            return Ok(());
        }

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
