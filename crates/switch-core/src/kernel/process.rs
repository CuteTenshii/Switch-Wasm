//! A guest process: its threads, modules, layout and the service objects it opened.

use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::cpu::*;
use crate::mem::AddressSpace;
use crate::services::*;
use crate::trace::Level;
use crate::IdMap;

/// How a process ended.
#[derive(Debug)]
pub(crate) enum Ending {
    Exited,
    Faulted(String),
    /// Ended by the process that started it.
    Terminated,
}

/// The pid of the process a session boots.
pub(crate) const ROOT_PID: u64 = 1;

#[derive(Debug)]
pub(crate) struct Process {
    pub(crate) mem: AddressSpace,
    /// For a library applet: the process that started it and the accessor it answers through.
    pub(crate) caller: Option<(u64, u64)>,
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
    pub(crate) fn new(mem: AddressSpace) -> Self {
        Self {
            mem,
            caller: None,
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

impl Cpu {
    /// End the running process. The booted title ending halts the console; any other
    /// process ends once the `svc` that ended it has replied.
    pub(crate) fn exit_process(&mut self, ending: Ending) {
        if self.pid == ROOT_PID {
            self.halted = true;
        } else {
            self.process_ending = Some(ending);
        }
    }

    /// Tear down process `pid` and finish the accessor its caller waits on.
    pub(crate) fn end_process(&mut self, pid: u64, ending: Ending) {
        let Some(process) = self.processes.get(&pid) else {
            return;
        };
        let (program, caller) = (process.program_id, process.caller);
        let (level, why) = match &ending {
            Ending::Exited => (Level::Info, "exited".to_owned()),
            Ending::Faulted(why) => (
                Level::Error,
                format!("faulted, so it ends cancelled: {why}"),
            ),
            Ending::Terminated => (Level::Info, "was terminated by its caller".to_owned()),
        };
        self.diagnostic(
            level,
            &format!("[kernel] process {pid} ({program:016x}) {why}"),
        );
        for thread in &mut self.machine.process_mut(pid).threads {
            thread.state = ThreadState::Finished;
        }
        if let Some((caller, accessor)) = caller {
            let exited = matches!(ending, Ending::Exited);
            self.in_process(caller, |cpu| cpu.finish_library_applet(accessor, exited));
        }
        if self.pid == pid {
            self.reschedule();
            if self.pid == pid {
                self.diagnostic(
                    Level::Error,
                    &format!("[kernel] nothing can run after process {pid} ended"),
                );
                self.halted = true;
                return;
            }
        }
        self.machine.processes.remove(&pid);
        self.forget_translations(pid);
        self.machine.nv.close_process(pid);
        self.machine.nv.gpu.remove_layer(pid);
        if self.pending_present.is_some_and(|(owner, _)| owner == pid) {
            self.pending_present = None;
        }
    }

    pub(crate) fn pids(&self) -> Vec<u64> {
        self.processes.keys().copied().collect()
    }

    /// A process with an empty address space; nothing runs in it until its main thread starts.
    pub(crate) fn create_process(&mut self) -> u64 {
        let pid = self
            .processes
            .keys()
            .next_back()
            .map_or(ROOT_PID, |last| last + 1);
        self.machine
            .processes
            .insert(pid, Process::new(AddressSpace::new()));
        self.in_process(pid, Cpu::map_runtime);
        pid
    }

    /// Start process `pid`'s main thread at `entry` with `arg` in x0 and its handle in x1,
    /// as the loader does.
    pub(crate) fn start_main_thread(&mut self, pid: u64, entry: u32, arg: u64) {
        let cycles = self.cycles;
        let process = self.machine.process_mut(pid);
        let mut regs = [0; REG_FILE];
        regs[0] = arg;
        regs[1] = MAIN_THREAD_HANDLE;
        regs[30] = u64::from(SELF_RETURN_TRAMPOLINE);
        regs[SP_SLOT] = STACK_TOP;
        let (priority, core) = (process.main_thread_priority, process.main_thread_core);
        process.threads.push(ThreadContext {
            handle: MAIN_THREAD_HANDLE,
            id: MAIN_THREAD_ID,
            state: ThreadState::Runnable,
            paused: false,
            regs,
            pc: entry,
            nzcv: 0,
            mode: ExecMode::A64,
            cpsr_q: false,
            cpsr_ge: 0,
            fpscr_nzcv: 0,
            vregs: VRegs::default(),
            fpcr: 0,
            fpsr: 0,
            tpidr: u64::from(MAIN_THREAD_TLS_BASE),
            tpidr_rw: 0,
            priority,
            core,
            ideal_core: i32::from(core),
            affinity: 1 << core,
            passed_over: 0,
            entry,
            arg,
            ran: 0,
            switches: 0,
            busy_at: cycles,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CODE: u32 = 0x0800_0000;
    const WORD: u32 = 0x0900_0000;

    /// `movz w1, #value; str w1, [x0]; b .-8`
    fn store_forever(cpu: &mut Cpu, value: u32) {
        for (i, insn) in [0x5280_0001 | (value << 5), 0xB900_0001, 0x17FF_FFFE]
            .into_iter()
            .enumerate()
        {
            cpu.mem.write_u32(CODE + 4 * i as u32, insn).unwrap();
        }
    }

    fn two_processes_storing(jit: bool) {
        let mut cpu = Cpu::new();
        cpu.set_jit_enabled(jit);
        cpu.bootstrap();
        store_forever(&mut cpu, 1);
        cpu.pc = CODE;
        cpu.regs[0] = u64::from(WORD);
        let other = cpu.create_process();
        cpu.in_process(other, |cpu| store_forever(cpu, 2));
        cpu.start_main_thread(other, CODE, u64::from(WORD));

        cpu.run(4 * TIME_SLICE).unwrap();

        let word = |pid: u64| cpu.processes[&pid].mem.read_u32(WORD).unwrap();
        assert_eq!(
            word(ROOT_PID),
            1,
            "the first process did not run, or saw the other's store"
        );
        assert_eq!(
            word(other),
            2,
            "the second process did not run, or saw the other's store"
        );
    }

    #[test]
    fn two_processes_take_turns_each_in_its_own_memory() {
        two_processes_storing(false);
    }

    #[test]
    fn translated_code_stays_with_its_process() {
        two_processes_storing(true);
    }
}
