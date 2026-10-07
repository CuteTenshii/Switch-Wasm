//! Guest address space layout and operation modes.

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
