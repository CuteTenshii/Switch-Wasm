//! The Horizon supervisor calls (`SVC`) libnx homebrew issues at runtime.

use super::{
    Cpu, SyscallMode, GUEST_STACK_REGION_ADDR, GUEST_STACK_REGION_SIZE, HID_SHMEM_SIZE,
    PL_SHMEM_SIZE,
};
use crate::{Error, Result};
use std::fmt::Write;

impl Cpu {
    pub(super) fn syscall(&mut self, imm: u16) -> Result<()> {
        match self.syscall_mode {
            SyscallMode::None => {                if imm == 0 {
                    self.halted = true;
                    Ok(())
                } else {
                    Err(Error::Cpu(format!("unimplemented syscall #{}", imm)))
                }
            }
            SyscallMode::Horizon => self.horizon_syscall(imm),
        }
    }

    /// Permissive stubs for the Horizon syscall numbers libnx homebrew hits
    /// during startup and normal single-threaded operation. The syscall
    /// numbers follow the real Switch ABI as emitted by libnx
    /// (`nx/source/kernel/svc.s`). There are no real services or threads, so
    /// service/IPC calls return success with a fake handle and waits complete
    /// immediately; this lets the app's `main()` run as far as it can before
    /// it needs real hardware.
    ///
    /// Results follow the real ABI: X0 carries the Result (success is 0,
    /// errors have bit 31 set), out-handles come back in X1, and
    /// value-returning syscalls put their result in X1 so the libnx wrapper
    /// (`str x0; svc; ldr x2; str x1, [x2]`) stores it into the caller's out
    /// pointer.
    pub(super) fn horizon_syscall(&mut self, imm: u16) -> Result<()> {
        const RESULT_OK: u64 = 0;
        // Non-zero handle handed out by handle-returning syscalls (libnx
        // stores X1 into the caller's output pointer).
        const FAKE_HANDLE: u64 = 0x1000;
        // KERNELRESULT(InvalidMemoryRange), as libnx spells it.
        const RESULT_INVALID_MEMORY_RANGE: u64 = 0x8000_DC01;
        match imm {
            0x01 => {
                // SetHeapSize: report a heap at a soft-mapped address.
                self.write_zr(0, RESULT_OK);
                self.write_zr(1, 0x3000_0000);
                Ok(())
            }
            0x02 | 0x03 | 0x14 => {
                // SetMemoryPermission / SetMemoryAttribute / UnmapSharedMemory
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x04 => {
                // MapMemory(dst, src, size): libnx maps a thread's stack into
                // the stack region and from then on uses only that mirror, so
                // back the destination for real. It has to become mapped memory
                // and not just a promise: `virtmemFindStack` picks the next
                // thread's mirror by looking for an unmapped range, so while
                // this was a no-op every thread was handed the same address and
                // they all shared one stack, corrupting each other's frames.
                let dst = self.read_zr(0) as u32;
                let src = self.read_zr(1) as u32;
                let size = self.read_zr(2) as usize;
                if dst == 0 || size == 0 {
                    self.write_zr(0, RESULT_INVALID_MEMORY_RANGE);
                    return Ok(());
                }
                self.mem.copy_range(dst, src, size)?;
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x13 => {
                // MapSharedMemory(handle, addr, size, perm): back it with a
                // real zeroed buffer. Two of these regions have host-provided
                // contents and are recognised by their size. For hid's,
                // remember where it landed and immediately publish a connected
                // controller, otherwise a program that polls before the host
                // sends any input decides no pad exists; pl's gets filled with
                // the shared font the guest is about to read.
                let addr = self.read_zr(1) as u32;
                let size = self.read_zr(2) as u32;
                self.mem.map_zero(addr, size as usize)?;
                if size == HID_SHMEM_SIZE {
                    self.hid_shmem_addr = addr;
                    self.set_gamepad_state(0, 0, 0, 0, 0);
                } else if size == PL_SHMEM_SIZE {
                    self.pl_shmem_addr = addr;
                    self.write_shared_font(addr);
                }
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x05 => {
                // UnmapMemory(dst, src, size), the counterpart of 0x04. hbmenu
                // detects the process address space by unmapping the very top of
                // the 64-bit range and reading the failure code: an out-of-range
                // unmap returns a kernel error whose low bits are 0xd401 (39-bit
                // AArch64) or 0xdc01 (36-bit). Report 39-bit; a real unmap gives
                // the destination's contents back to the source range and frees
                // it, so the address space can be reused.
                let dst = self.read_zr(0);
                if (dst >> 48) == 0xFFFF {
                    self.write_zr(0, 0x8000_D401);
                    return Ok(());
                }
                let src = self.read_zr(1) as u32;
                let size = self.read_zr(2) as usize;
                if dst != 0 && src != 0 && size != 0 {
                    self.mem.copy_range(src, dst as u32, size)?;
                    self.mem.unmap(dst as u32, size);
                }
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
             0x06 => {
                 // QueryMemory(info, pageInfo, addr): report the contiguous
                 // run of pages in the same state (allocated vs untouched) as
                 // the queried page. Real pages (image, stack, heap, anything
                 // the app has written to) come back as RWX; untouched
                 // soft-mapped pages come back as unmapped so libnx virtmem
                 // address-space walks and reservations see free space. The
                 // old stub reported the whole low 2 GiB as one RWX region,
                 // which made deko3d's AS reservation fail.
                 //
                 // A module's `.text` reports the real R-X permission code
                 // (5) instead of blanket RWX: retail `rtld` discovers the
                 // other loaded modules (`main`/`subsdk*`/`sdk`) by walking
                 // `QueryMemory` across the address space and filtering for
                 // exactly `type == CodeStatic (3) && perm == R-X (5)` before
                 // checking each hit for a `MOD0` signature — reporting RWX
                 // there makes every module invisible to that scan, so it
                 // can never resolve a symbol from another module.
                 let out = self.read_zr(0) as u32;
                 let addr = self.read_zr(2) as u32;
                 let region = |a: u32| (self.mem.page_mapped(a & !0xFFF), self.mem.is_readonly(a & !0xFFF));
                 let mut base = addr & !0xFFF;
                 while base >= 0x1000 && region(base - 0x1000) == region(base) {
                     base -= 0x1000;
                 }
                 let (mapped, text) = region(base);
                 let mut end = base + 0x1000;
                 while end < 0x8000_0000 && region(end) == (mapped, text) {
                     end += 0x1000;
                 }
                 let mut info = Vec::with_capacity(40);
                 info.extend_from_slice(&(base as u64).to_le_bytes());
                 info.extend_from_slice(&((end - base) as u64).to_le_bytes());
                 for v in [
                     if mapped { 3u32 } else { 0 }, // type (CodeStatic / Unmapped)
                     0,
                     if text { 5u32 } else if mapped { 0b111 } else { 0 }, // perm (R-X / RWX / none)
                     0,
                     0,
                     0,
                 ] {
                     info.extend_from_slice(&v.to_le_bytes());
                 }
                 for (i, &b) in info.iter().enumerate() {
                     self.mem.write_u8(out.wrapping_add(i as u32), b)?;
                 }
                 self.write_zr(0, RESULT_OK);
                 self.write_zr(1, if mapped { 0x1000 } else { 0 }); // page info
                 Ok(())
             }
            0x07 => {
                // ExitProcess
                self.halted = true;
                Ok(())
            }
            0x0A => {
                // ExitThread: only the main thread ending stops the process.
                self.exit_thread();
                Ok(())
            }
            0x08 => {
                // CreateThread(entry = X1, arg = X2, stack_top = X3,
                // priority = W4, core = W5) -> handle in X1. The thread gets its
                // own TLS block and starts suspended.
                let entry = self.read_zr(1) as u32;
                let arg = self.read_zr(2);
                let stack_top = self.read_zr(3);
                let handle = self.create_thread(entry, arg, stack_top);
                self.write_zr(0, RESULT_OK);
                self.write_zr(1, handle);
                Ok(())
            }
            0x09 => {
                // StartThread: make it runnable. It gets the CPU at the next
                // point this thread blocks.
                let handle = self.read_zr(0);
                self.start_thread(handle);
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x0B => {
                // SleepThread: give another thread the CPU.
                self.write_zr(0, RESULT_OK);
                self.yield_thread();
                Ok(())
            }
            0x1A => {
                // ArbitrateLock(owner = W0, mutex = X1, self = W2)
                let owner = self.read_zr(0) as u32;
                let mutex = self.read_zr(1) as u32;
                let requester = self.read_zr(2) as u32;
                self.write_zr(0, RESULT_OK);
                self.arbitrate_lock(owner, mutex, requester);
                Ok(())
            }
            0x1B => {
                // ArbitrateUnlock(mutex = X0): hand the lock to a waiter.
                let mutex = self.read_zr(0) as u32;
                self.write_zr(0, RESULT_OK);
                self.arbitrate_unlock(mutex);
                Ok(())
            }
            0x1C => {
                // WaitProcessWideKeyAtomic(mutex = X0, key = X1, self = W2,
                // timeout = X3): release the mutex and block on the condvar.
                let mutex = self.read_zr(0) as u32;
                let key = self.read_zr(1) as u32;
                let requester = self.read_zr(2) as u32;
                self.write_zr(0, RESULT_OK);
                self.wait_process_wide_key(mutex, key, requester);
                Ok(())
            }
            0x1D => {
                // SignalProcessWideKey(key = X0, count = W1)
                let key = self.read_zr(0) as u32;
                let count = self.read_zr(1) as u32 as i32;
                self.write_zr(0, RESULT_OK);
                self.signal_process_wide_key(key, count);
                Ok(())
            }
            0x0C | 0x0D | 0x0E | 0x0F | 0x16 | 0x17 | 0x19 | 0x28 => {
                // get/set thread priority + core mask / CloseHandle /
                // ResetSignal / CancelSynchronization / ReturnFromException
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x10 => {
                // GetCurrentProcessorNumber
                self.write_zr(0, 0);
                Ok(())
            }
            0x11 | 0x12 => {
                // SignalEvent / ClearEvent
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x15 => {
                // CreateTransferMemory
                self.write_zr(0, RESULT_OK);
                self.write_zr(1, FAKE_HANDLE);
                Ok(())
            }
            0x18 => {
                // WaitSynchronization: report the wait as immediately satisfied.
                // X1 is the "number of signaled handles" that the libnx wrapper
                // stores to the caller's out pointer; deko3d indexes its waiter
                // array by it, so garbage here makes the fence wait retry forever.
                self.write_zr(0, RESULT_OK);
                self.write_zr(1, 1);
                // Waiting is where a thread that spins on another thread's
                // progress gives way to it.
                self.yield_thread();
                Ok(())
            }
            0x1E => {
                // GetSystemTick (ns scale, arbitrary)
                self.write_zr(0, self.cycles * 1000);
                Ok(())
            }
            0x1F => {
                // ConnectToNamedPort: read the port name so later IPC can be
                // dispatched to the right stub service.
                let name_ptr = self.read_zr(1) as u32;
                let name = if name_ptr != 0 {
                    self.read_port_name(name_ptr)
                } else {
                    String::new()
                };
                let handle = self.alloc_handle();
                self.record_handle(handle, &name);
                self.write_zr(0, RESULT_OK);
                self.write_zr(1, handle);
                Ok(())
            }
            0x20 | 0x21 | 0x22 | 0x23 => {
                // SendSyncRequest[Light|WithUserBuffer] / async variant.
                // If we recognize the target handle as a named service, dispatch
                // to a small stub implementation. Otherwise fall back to the
                // libnx applet-style generic reply so hbmenu/applet init still
                // progresses.
                let tls = self.tpidr as u32;
                let handle = self.read_zr(0);
                let cmd_id = self.ipc_command_id(tls);
                let svc_name = self.service_name(handle).map(|s| s.to_string());
                if std::env::var("TRACE_IPC").is_ok() {
                    let obj = self.ipc_domain_object_id(tls);
                    let iface = self.domain_interface(handle, obj).map(|s| s.to_string());
                    eprintln!(
                        "[ipc] pc={:#x} h={} svc={:?} obj={} iface={:?} domain={} type={} cmd={:?}",
                        self.pc,
                        handle,
                        svc_name,
                        obj,
                        iface,
                        self.ipc_is_domain_request(tls),
                        self.ipc_message_type(tls),
                        cmd_id
                    );
                }
                // A Close request (message type 2) carries no command id at
                // all: it tears the session down. Dispatching it on whatever
                // command id is still sitting in the TLS buffer runs a real
                // command instead — closing an `fsp-srv` session was landing on
                // `CreateFile` and adding an empty file to the SD card.
                if self.ipc_message_type(tls) == 2 {
                    self.forget_handle(handle);
                    self.write_ipc_response(tls, 0, &[], &[], &[])?;
                    self.write_zr(0, RESULT_OK);
                    return Ok(());
                }
                if let Some(name) = svc_name {
                    let name = name;
                    match name.as_str() {
                        "sm:" | "sm" => self.sm_request(tls, cmd_id, handle)?,
                        "fsp-srv" | "fsp-srv:" => {
                            // libnx converts fsp-srv to a domain, so the
                            // fs/dir/file sub-sessions come in as object ids on
                            // the same session handle. Route on the recorded
                            // object interface; unknown objects hit the root
                            // stub.
                            let object_id = self.ipc_domain_object_id(tls);
                            match self.domain_interface(handle, object_id) {
                                Some("fsp-srv-fs") => self.fs_request(tls, cmd_id, handle)?,
                                Some("fsp-srv-fs-dir") => {
                                    self.fs_dir_request(tls, cmd_id, Self::object_key(handle, object_id))?
                                }
                                Some("fsp-srv-fs-file") => {
                                    self.fs_file_request(tls, cmd_id, Self::object_key(handle, object_id))?
                                }
                                Some("fsp-srv-storage") => {
                                    self.fs_storage_request(tls, cmd_id)?
                                }
                                _ => self.fsp_srv_request(tls, cmd_id, handle)?,
                            }
                        }
                        // The same interfaces reached over their own session
                        // handle, which is how a caller that never converts to
                        // a domain (libtransistor) uses them.
                        "fsp-srv-fs" => self.fs_request(tls, cmd_id, handle)?,
                        "fsp-srv-fs-dir" => {
                            self.fs_dir_request(tls, cmd_id, Self::object_key(handle, 0))?
                        }
                        "fsp-srv-fs-file" => {
                            self.fs_file_request(tls, cmd_id, Self::object_key(handle, 0))?
                        }
                        "fsp-srv-storage" => self.fs_storage_request(tls, cmd_id)?,
                        "vi:m" | "vi:m:" => self.vi_request(tls, handle, cmd_id)?,
                        "set" => self.set_request(tls, cmd_id)?,
                        "set:sys" => self.set_sys_request(tls, cmd_id)?,
                        "nvdrv" | "nvdrv:" | "nvdrv:a" | "nvdrv:a:" | "nvdrv:s" | "nvdrv:t" => {
                            self.nvdrv_request(tls, cmd_id, handle)?
                        }
                        // pl:u, the shared-font service.
                        "pl:u" | "pl:s" => self.pl_request(tls, cmd_id)?,
                        // time:*, converted to a domain by libnx the same way
                        // fsp-srv is; the system clock / steady clock /
                        // timezone sub-interfaces come back as out-objects on
                        // this same session handle.
                        "time:s" | "time:u" | "time:a" | "time:r" => {
                            let object_id = self.ipc_domain_object_id(tls);
                            match self.domain_interface(handle, object_id) {
                                Some("time:system-clock") => {
                                    self.time_system_clock_request(tls, cmd_id)?
                                }
                                Some("time:steady-clock") => {
                                    self.time_steady_clock_request(tls, cmd_id)?
                                }
                                Some("time:timezone") => self.time_timezone_request(tls, cmd_id)?,
                                _ => self.time_request(tls, cmd_id, handle)?,
                            }
                        }
                        // The same sub-interfaces reached over their own
                        // session handle (the libtransistor case, as with
                        // fsp-srv-fs above).
                        "time:system-clock" => self.time_system_clock_request(tls, cmd_id)?,
                        "time:steady-clock" => self.time_steady_clock_request(tls, cmd_id)?,
                        "time:timezone" => self.time_timezone_request(tls, cmd_id)?,
                        // psm (power state management): the battery. Its
                        // IPsmSession sub-interface follows the same
                        // domain-or-own-handle split as time's above.
                        "psm" => {
                            let object_id = self.ipc_domain_object_id(tls);
                            match self.domain_interface(handle, object_id) {
                                Some("psm-session") => self.psm_session_request(tls, cmd_id)?,
                                _ => self.psm_request(tls, cmd_id, handle)?,
                            }
                        }
                        "psm-session" => self.psm_session_request(tls, cmd_id)?,
                        // appletOE (application applet) / appletAE (everything
                        // else). Both are the same IApplicationProxyService/
                        // IApplicationProxy/ICommonStateGetter chain.
                        "appletOE" | "appletAE" => self.applet_request(tls, handle, cmd_id)?,
                        "nifm:u" => {
                            let object_id = self.ipc_domain_object_id(tls);
                            match self.domain_interface(handle, object_id) {
                                Some("nifm:general-service") => {
                                    self.nifm_general_service_request(tls, handle, cmd_id)?
                                }
                                Some("nifm:request") => self.write_ipc_response(tls, 0, &[], &[], &[])?,
                                _ => self.nifm_request(tls, cmd_id, handle)?,
                            }
                        }
                        "audren:u" => self.audren_request(tls, cmd_id)?,
                        "audren:iaudiorenderer" => self.audren_renderer_request(tls, cmd_id, handle)?,
                         name => {
                             // Known service, no dedicated stub. The applet
                             // services get the state values their init polls
                             // for (ReceiveMessage → the applet message,
                             // Get*Mode/GetCurrentFocusState → the state); every
                             // other service must NOT get those numbers back —
                             // answering `pl:u`'s GetLoadState with the applet
                             // message left NX-Shell polling it 190k times.
                             let applet = name.starts_with("applet");
                             let data = match cmd_id {
                                 Some(1) if applet => 15, // ReceiveMessage → FocusStateChanged
                                 Some(5) if applet => 1,  // GetOperationMode → Handheld
                                 Some(6) if applet => 0,  // GetPerformanceMode → Normal
                                 Some(9) if applet => 1,  // GetCurrentFocusState → InFocus
                                 _ => {
                                     let obj = self.next_object_id;
                                     self.next_object_id = obj.wrapping_add(1);
                                     obj
                                 }
                             };
                             self.write_ipc_response(tls, 0, &[], &data.to_le_bytes(), &[])?
                         }
                    }
                } else {
                    // Unrecognized session handle. The display service's session
                    // handles come from generic object-id replies and aren't in
                    // service_handles, so try the vi stub first; fall back to the
                    // old applet-style generic reply if the request isn't a vi
                    // command (e.g. hid/time sessions).
                    if let Some(cmd) = cmd_id {
                        // vi commands: GetIApplicationDisplayService (2) and the
                        // display/session commands (100+). The generic reply already
                        // answers ConvertToDomain (0) with a valid object id, and the
                        // small applet state commands (1 = ReceiveMessage, 5/6/9)
                        // must keep the generic reply.
                        if cmd == 2 || cmd >= 100 {
                            return self.vi_request(tls, handle, cmd_id);
                        }
                    }
                    let start = self.ipc_reply_start(tls);
                    let is_domain = self
                        .mem
                        .read_u32(tls.wrapping_add(start + 0x10))
                        .unwrap_or(0)
                        == 0x4943_4653;
                    let data = match cmd_id {
                        Some(1) => 15,
                        Some(5) => 1,
                        Some(6) => 0,
                        Some(9) => 1,
                        _ => {
                            let obj = self.next_object_id;
                            self.next_object_id = obj.wrapping_add(1);
                            obj
                        }
                    };
                    if is_domain {
                        for i in 0..4u32 {
                            let _ = self.mem.write_u32(tls.wrapping_add(start + i * 4), 0);
                        }
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x10), 0x4F43_4653);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x14), 0);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x18), 0);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x1C), 0);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x20), data);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x24), 0);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x28), data);
                    } else {
                        let _ = self.mem.write_u32(tls.wrapping_add(start), 0x4F43_4653);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x04), 0);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x08), 0);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x0C), 0);
                        let _ = self.mem.write_u32(tls.wrapping_add(start + 0x10), data);
                    }
                }
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x24 => {
                // GetProcessId(out_process_id, process_handle): Result in
                // X0, id in X1 — the caller's wrapper stores X1 through the
                // out pointer. Confirmed wrong by tracing a real title's
                // `sdk` init through Binary Ninja: it treats any non-zero X0
                // as failure, and this used to hand back X0=1 (looking like
                // a "successful" id 1, but actually read as an error code)
                // with X1 left stale, sending real code down an error path
                // that ultimately aborted.
                self.write_zr(0, RESULT_OK);
                self.write_zr(1, 1);
                Ok(())
            }
            0x25 => {
                // GetThreadId(out_thread_id, handle): same shape and same
                // fix as GetProcessId above.
                self.write_zr(0, RESULT_OK);
                self.write_zr(1, 1);
                Ok(())
            }
            0x26 => {
                // Break(reason, arg, size): fatal debugger trap. Nintendo's
                // own abort path (nn::diag::detail::AbortImpl) reaches this
                // with real diagnostic info attached — a reason code and an
                // arg/size pair the caller chose to hand the debugger — and
                // a plain "[svcBreak]" marker was throwing all of it away.
                // Decode the reason, dereference the arg pointer when its
                // size is a plain integer width, and include a
                // frame-pointer backtrace: together, exactly what identified
                // the real cause the last few times this fired (an
                // unresolved symbol, a missing InfoType) without needing an
                // ad hoc debug build to see it.
                let reason = self.read_zr(0);
                let arg = self.read_zr(1);
                let size = self.read_zr(2);
                let reason_name = match reason & 0xFF {
                    0 => "Panic",
                    1 => "Assert",
                    2 => "User",
                    3 => "PreLoadDll",
                    4 => "PostLoadDll",
                    5 => "PreUnloadDll",
                    6 => "PostUnloadDll",
                    7 => "CppException",
                    _ => "Unknown",
                };
                let mut msg = format!(
                    "[svcBreak] reason={reason_name} ({reason:#x}) arg={arg:#x} size={size:#x}"
                );
                if let Some(value) = match size {
                    1 => self.mem.read_u8(arg as u32).ok().map(u64::from),
                    2 => self.mem.read_u16(arg as u32).ok().map(u64::from),
                    4 => self.mem.read_u32(arg as u32).ok().map(u64::from),
                    8 => self.mem.read_u64(arg as u32).ok(),
                    _ => None,
                } {
                    let _ = write!(msg, " value={value:#x}");
                }
                msg.push('\n');
                for addr in self.backtrace(16) {
                    let _ = write!(msg, "  {addr:#010x}\n");
                }
                self.out.extend_from_slice(msg.as_bytes());
                self.halted = true;
                Ok(())
            }
            0x27 => {
                // OutputDebugString(ptr, size) — log to the console.
                let ptr = self.read_zr(0) as u32;
                let len = (self.read_zr(1) as i64).clamp(0, 4096) as u32;
                if ptr != 0 && len > 0 {
                    for i in 0..len {
                        match self.mem.read_u8(ptr.wrapping_add(i)) {
                            Ok(b) => self.out.push(b),
                            Err(_) => break,
                        }
                    }
                }
                Ok(())
            }
             0x29 => {
                 // GetInfo(out, infoType, handle, infoSubValue): report the
                 // value in X1 (the libnx wrapper stores it to the out
                 // pointer). The InfoType numbering here matches the libnx
                 // build hbmenu is compiled against: 2/3 Alias, 4/5 Heap,
                 // 6/7 Total/Used memory, 11 RandomEntropy, 12/13 Aslr, 14/15
                 // Stack.
                 let info_type = self.read_zr(1);
                 let value = match info_type {
                     2 => 0x0000_0010_0000_0000, // AliasRegionAddress
                     3 => 0x0000_0000_2000_0000, // AliasRegionSize
                     4 => 0x0000_0002_0000_0000, // HeapRegionAddress
                     5 => 0x0000_0000_2000_0000, // HeapRegionSize
                     6 => 0x1E00_0000, // TotalMemorySize
                     7 => 0,         // UsedMemorySize
                     8 => 0,         // DebuggerAttached
                     9 => 0,         // ResourceLimit
                     // RandomEntropy: 4 words (infoSubValue 0..3) of kernel-
                     // supplied randomness, real hardware's seed for stack
                     // canaries/ASLR cookies. Real `sdk` startup (confirmed by
                     // tracing "A Short Hike"'s actual `rtld`+`sdk` boot)
                     // fetches two of these words and aborts
                     // (`svcBreak`/Panic) if what comes back looks unusable —
                     // an all-zero entropy pool reads as "broken RNG", not
                     // "no RNG", to security-conscious SDK init. There's no
                     // real entropy source to draw on here, so this returns
                     // *some* non-zero, per-subvalue-varying bits (SplitMix64
                     // keyed by the subvalue) rather than a cryptographically
                     // meaningful seed — it only needs to satisfy that "not
                     // obviously broken" check, not actually secure anything.
                     11 => {
                         let sub = self.read_zr(3).wrapping_add(1);
                         let mut z = sub.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                         z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                         z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                         z ^ (z >> 31)
                     }
                     12 => 0x0800_0000, // AslrRegionAddress
                     13 => 0x1F00_0000, // AslrRegionSize
                     // Where thread stacks get mirrored. It has to be clear of
                     // the main stack (`STACK_TOP`) and big enough for several
                     // stacks plus the guard pages libnx leaves around them:
                     // when a lookup finds no free range it hands back a null
                     // mirror address, and every thread ends up on one stack.
                     14 => u64::from(GUEST_STACK_REGION_ADDR),
                     15 => u64::from(GUEST_STACK_REGION_SIZE),
                     20 => 0,        // UserExceptionContextAddress
                     28 => 0,        // AliasRegionExtraSize
                     _ => 0,
                 };
                 self.write_zr(1, value);
                 self.write_zr(0, RESULT_OK);
                 Ok(())
             }
            0x6F => {
                // GetSystemInfo(out, handle, infoType): value in X1, as above.
                let info_type = self.read_zr(2);
                let value = match info_type {
                    2 => 0x1000_0000, // TotalMemorySize
                    3 => 0,           // UsedMemorySize
                    _ => 0,
                };
                self.write_zr(1, value);
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            _ => Err(Error::Cpu(format!("unimplemented Horizon syscall #{:#x}", imm))),
        }
    }
}
