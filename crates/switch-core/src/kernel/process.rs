//! A guest process: its threads, modules, layout and the service objects it opened.

use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::cpu::{MemoryLayout, SaveKey, ThreadContext, DEFAULT_THREAD_PRIORITY, MAIN_THREAD_ID};
use crate::services::*;
use crate::IdMap;

/// The pid of the process a session boots.
pub(crate) const ROOT_PID: u64 = 1;

#[derive(Debug)]
pub(crate) struct Process {
    /// The last `fatal:u` report, kept to classify a later `ExitProcess` as a crash.
    pub(crate) guest_fatal: Option<String>,
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
    /// `GetPopFromGeneralChannelEvent`; never fired.
    pub(crate) general_channel_event: Option<u64>,
    /// The `ILockAccessor` event: always signalled. See `Cpu::am_lock_accessor_event`.
    pub(crate) lock_accessor_event: Option<u64>,
    /// The title's NACP save quota, reported by `IApplicationFunctions`; set through
    /// [`Cpu::set_save_data_quota`] and not enforced.
    pub(crate) save_data_quota: fs::SaveDataQuota,
    /// Chosen from the NPDM system resource size and program id.
    pub(crate) memory_layout: MemoryLayout,
    /// Kept because the layout also depends on the program id, which may arrive later.
    pub(crate) system_resource_size: u32,
    /// Application proxy (told `FocusStateChanged`) vs applet (told `ChangeIntoForeground`).
    pub(crate) applet_is_application: bool,
    /// `ISelfController` auto-sleep settings, stored so getters read them back.
    pub(crate) idle_time_detection_extension: u32,
    pub(crate) auto_sleep_disabled: bool,
    /// Stored so the getter reads it back.
    pub(crate) home_button_double_click_enabled: bool,
    /// NROs mapped by `ldr:ro`, by mapped address; see [`Cpu::ldr_ro_request`].
    pub(crate) ro_modules: BTreeMap<u32, ldr::RoModule>,
    /// Registered NRRs, by address. Signatures are not checked.
    pub(crate) ro_registrations: BTreeMap<u32, u32>,
    /// Open SD directory handles to the entries not yet yielded.
    pub(crate) fs_dirs: IdMap<u64, Vec<crate::vfs::DirEntry>>,
    /// Open `IFile` objects: domain object id to path.
    pub(crate) fs_files: IdMap<u64, String>,
    /// `am` `IStorage` contents, by object.
    pub(crate) am_storages: IdMap<u64, Vec<u8>>,
    /// DLC indices `aoc:u` reports; the content is in `data_archives`.
    pub(crate) add_on_content: std::collections::BTreeSet<u32>,
    /// Base DLC id from the NACP; zero means derive it from the program id.
    pub(crate) add_on_content_base_id: u64,
    /// The save an `fsp-srv` object addresses; absent for the SD card.
    pub(crate) fs_mount: IdMap<u64, SaveKey>,
    /// The data archive an open `IStorage` serves; absent for the process's own RomFS.
    pub(crate) fs_storage_archive: IdMap<u64, u64>,
    /// `FsSaveDataInfo` records an open save scan has yet to yield.
    pub(crate) fs_save_infos: IdMap<u64, VecDeque<[u8; crate::services::fs::SAVE_DATA_INFO_SIZE]>>,
    /// RomFS file index per storage (`None` key for the process's own RomFS);
    /// `None` value for unreadable tables.
    pub(crate) romfs_indexes: BTreeMap<Option<u64>, Option<crate::romfs::RomFsIndex>>,
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
    /// `ssl` state: interface revision, context count, and per-context options.
    pub(crate) ssl_interface_version: u32,
    pub(crate) ssl_contexts: u32,
    pub(crate) ssl_options: HashMap<(u64, u32), u32>,
    pub(crate) erpt_readers: IdMap<u64, erpt::ErrorReportReader>,
    /// Guest address of pl's shared memory; 0 until mapped.
    pub(crate) pl_shmem_addr: u32,
    /// Per-`IAudioRenderer` state from `OpenAudioRenderer`, used to size update replies.
    pub(crate) audren_renderers: IdMap<u64, audren::AudioRenderer>,
    /// Open `IAudioOut`s, by session handle.
    pub(crate) audio_outs: IdMap<u64, audout::AudioOut>,
    /// Open `IHardwareOpusDecoder`s; the guest work buffer is unused.
    pub(crate) opus_decoders: IdMap<u64, hwopus::HwOpus>,
    /// The user each `IProfile`/`IProfileEditor` object was opened for.
    pub(crate) acc_profiles: IdMap<u64, [u8; 16]>,
    /// Program id for `pm:info`; defaults to the Album applet's, as hbmenu homebrew runs as.
    pub(crate) program_id: u64,
    /// Open `bsd` sockets and their options.
    pub(crate) bsd_sockets: HashMap<i32, net::BsdSocket>,
    pub(crate) bsd_socket_options: HashMap<(i32, u32, u32), u32>,
    /// Next descriptor; starts at 3, past the standard streams.
    pub(crate) next_bsd_fd: i32,
    /// Guest threads; index 0 is main. The running thread's slot is stale while it runs.
    pub(crate) threads: Vec<ThreadContext>,
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

impl Process {
    pub(crate) fn new() -> Self {
        Self {
            guest_fatal: None,
            applet_messages: VecDeque::new(),
            sleep_lock_event: None,
            sleep_lock_acquired: false,
            application_functions_210_event: None,
            general_channel_event: None,
            lock_accessor_event: None,
            aoc_list_changed_event: None,
            display_resolution_event: None,
            save_data_quota: fs::SaveDataQuota::default(),
            memory_layout: MemoryLayout::PLAIN,
            system_resource_size: 0,
            applet_focus_announced: false,
            applet_is_application: true,
            idle_time_detection_extension: 0,
            auto_sleep_disabled: false,
            home_button_double_click_enabled: false,
            applet_event: None,
            ro_modules: BTreeMap::new(),
            ro_registrations: BTreeMap::new(),
            fs_dirs: IdMap::default(),
            fs_files: IdMap::default(),
            add_on_content: std::collections::BTreeSet::new(),
            add_on_content_base_id: 0,
            fs_mount: IdMap::default(),
            fs_storage_archive: IdMap::default(),
            fs_save_infos: IdMap::default(),
            romfs_indexes: BTreeMap::new(),
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
            hid_shmem_addr: 0,
            hid_shmem_handle: None,
            ssl_interface_version: 0,
            ssl_contexts: 0,
            ssl_options: HashMap::new(),
            erpt_readers: IdMap::default(),
            pl_shmem_addr: 0,
            audren_renderers: IdMap::default(),
            audio_outs: IdMap::default(),
            opus_decoders: IdMap::default(),
            acc_profiles: IdMap::default(),
            program_id: crate::kernel::ipc::DEFAULT_PROGRAM_ID,
            bsd_sockets: HashMap::new(),
            bsd_socket_options: HashMap::new(),
            next_bsd_fd: 3,
            threads: Vec::new(),
            module_names: Vec::new(),
            main_thread_priority: DEFAULT_THREAD_PRIORITY,
            main_thread_core: 0,
            process_core_mask: crate::npdm::APPLICATION_CORE_MASK,
            next_thread_id: MAIN_THREAD_ID + 1,
        }
    }
}
