//! Guest address space layout and operation modes.

use crate::mem::ADDRESS_SPACE_SIZE;

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

/// The stack region `svcGetInfo` reports, where `nn::os` places thread stacks at
/// random; it must be entirely free and large.
pub const GUEST_STACK_REGION_ADDR: u32 = 0x1800_0000;
pub const GUEST_STACK_REGION_SIZE: u32 = SELF_RETURN_TRAMPOLINE - GUEST_STACK_REGION_ADDR;

/// Start of the guest address space: below it faults, as on Horizon, so a NULL
/// dereference aborts where it happens instead of reading zeros.
pub const GUEST_SPACE_START: u32 = 0x0020_0000;

/// End of a 32-bit process's address space: below is soft-mapped by `Cpu::map_runtime`,
/// above faults (hbmenu probes the top to size the space). A 64-bit process's runs to
/// [`ADDRESS_SPACE_SIZE`].
pub const GUEST_SPACE_END: u32 = 0xFF00_0000;

/// Where `ldr:ro` maps run-time loaded modules: between [`STACK_TOP`] and the low heap region,
/// clear of every region the guest's allocators use.
pub const RO_MODULE_REGION_ADDR: u32 = 0x2900_0000;
pub const RO_MODULE_REGION_SIZE: u32 = LOW_HEAP_REGION_ADDR - RO_MODULE_REGION_ADDR;

/// A 32-bit process's heap region (`svcSetHeapSize`) and alias region (`svcMapPhysicalMemory`),
/// sharing what is left of 4 GiB. `nnSdk` takes one route per process, so the layout gives
/// that route nearly everything.
pub const LOW_HEAP_REGION_ADDR: u32 = 0x3000_0000;
pub const LOW_HEAP_REGION_SIZE: u32 = 0xC800_0000;
pub const LOW_ALIAS_REGION_ADDR: u32 = LOW_HEAP_REGION_ADDR + LOW_HEAP_REGION_SIZE;
pub const LOW_ALIAS_REGION_SIZE: u32 = SHARED_BUFFER_ADDR - LOW_ALIAS_REGION_ADDR;
/// The same for a 32-bit process on the alias route, which declares a system resource.
pub const LOW_VAMM_HEAP_REGION_SIZE: u32 = 0x0800_0000;
pub const LOW_VAMM_ALIAS_REGION_ADDR: u32 = LOW_HEAP_REGION_ADDR + LOW_VAMM_HEAP_REGION_SIZE;
pub const LOW_VAMM_ALIAS_REGION_SIZE: u32 = SHARED_BUFFER_ADDR - LOW_VAMM_ALIAS_REGION_ADDR;
/// Its `TotalMemorySize`, which `nn::init` reserves in the alias region beside the Vamm arena.
pub const LOW_VAMM_TOTAL_MEMORY_SIZE: u64 = 0x3800_0000;

/// A 64-bit process's heap and alias regions, above the 4 GiB the emulator's own furniture sits
/// in, at a console's sizes. Both routes fit at once, so a title that uses both (Tears of the
/// Kingdom), or reserves tens of GiB of address space (Paper Mario: The Origami King, Pokémon
/// Legends: Z-A), gets what it would on a console.
pub const HEAP_REGION_ADDR: u64 = 0x10_0000_0000;
pub const HEAP_REGION_SIZE: u64 = 0x1_8000_0000;
pub const ALIAS_REGION_ADDR: u64 = 0x20_0000_0000;
pub const ALIAS_REGION_SIZE: u64 = 0x10_0000_0000;
const _: () = assert!(ALIAS_REGION_ADDR + ALIAS_REGION_SIZE <= ADDRESS_SPACE_SIZE);

/// `svcGetInfo`'s total memory for an application, which `nn::init` sizes its heap from: about
/// the 3.2 GiB a console gives. Titles size pools from hardcoded constants (Just Dance 2019's
/// 699 MiB graphics pool), so less than a console's fails their allocations.
pub const GUEST_TOTAL_MEMORY_SIZE: u64 = LOW_HEAP_REGION_SIZE as u64;
/// The same for firmware library applets.
pub const APPLET_TOTAL_MEMORY_SIZE: u64 = 0x2000_0000;
const _: () = assert!(GUEST_TOTAL_MEMORY_SIZE <= crate::mem::MAX_MAPPED_BYTES);

/// The address space a process is given: its regions, from whether it is 64-bit and its NPDM
/// system resource size, and the memory figures `svcGetInfo` reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryLayout {
    pub heap_addr: u64,
    pub heap_size: u64,
    pub alias_addr: u64,
    pub alias_size: u64,
    /// `TotalMemorySize`.
    pub total_memory: u64,
    /// `SystemResourceSizeTotal`, as the NPDM declares it; nonzero puts `nnSdk` on the
    /// Vamm manager.
    pub system_resource: u64,
    /// The end of the address space: where the soft region and `svcQueryMemory`'s walks stop.
    pub space_end: u64,
}

impl MemoryLayout {
    /// A 64-bit process's, before its total memory and system resource are known.
    pub const WIDE: MemoryLayout = MemoryLayout {
        heap_addr: HEAP_REGION_ADDR,
        heap_size: HEAP_REGION_SIZE,
        alias_addr: ALIAS_REGION_ADDR,
        alias_size: ALIAS_REGION_SIZE,
        total_memory: GUEST_TOTAL_MEMORY_SIZE,
        system_resource: 0,
        space_end: ADDRESS_SPACE_SIZE,
    };

    /// A 32-bit process's on the heap route.
    pub const LOW: MemoryLayout = MemoryLayout {
        heap_addr: LOW_HEAP_REGION_ADDR as u64,
        heap_size: LOW_HEAP_REGION_SIZE as u64,
        alias_addr: LOW_ALIAS_REGION_ADDR as u64,
        alias_size: LOW_ALIAS_REGION_SIZE as u64,
        total_memory: GUEST_TOTAL_MEMORY_SIZE,
        system_resource: 0,
        space_end: GUEST_SPACE_END as u64,
    };

    /// A 32-bit process's on the alias route.
    pub const LOW_VAMM: MemoryLayout = MemoryLayout {
        heap_addr: LOW_HEAP_REGION_ADDR as u64,
        heap_size: LOW_VAMM_HEAP_REGION_SIZE as u64,
        alias_addr: LOW_VAMM_ALIAS_REGION_ADDR as u64,
        alias_size: LOW_VAMM_ALIAS_REGION_SIZE as u64,
        total_memory: LOW_VAMM_TOTAL_MEMORY_SIZE,
        system_resource: 0,
        space_end: GUEST_SPACE_END as u64,
    };

    /// The layout of a process: 64-bit or not, which program, and the NPDM's system resource.
    pub fn for_process(is_64_bit: bool, program_id: u64, system_resource: u32) -> MemoryLayout {
        let base = match (is_64_bit, system_resource) {
            (true, _) => MemoryLayout::WIDE,
            (false, 0) => MemoryLayout::LOW,
            (false, _) => MemoryLayout::LOW_VAMM,
        };
        let total_memory = if is_64_bit && crate::services::am::is_library_applet(program_id) {
            APPLET_TOTAL_MEMORY_SIZE
        } else {
            base.total_memory
        };
        MemoryLayout {
            total_memory,
            system_resource: u64::from(system_resource),
            ..base
        }
    }
}

/// hid's shared memory size, used to recognise that mapping.
pub const HID_SHMEM_SIZE: u32 = 0x4_0000;

/// `pl:u`'s shared memory size (`SHAREDMEMFONT_SIZE`), recognised the same way.
pub const PL_SHMEM_SIZE: u32 = 0x110_0000;
