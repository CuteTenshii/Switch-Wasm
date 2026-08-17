//! The Horizon supervisor calls (`SVC`) libnx homebrew issues at runtime.

use super::{Cpu, SyscallMode};
use crate::{Error, Result};

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
        match imm {
            0x01 => {
                // SetHeapSize: report a heap at a soft-mapped address.
                self.write_zr(0, RESULT_OK);
                self.write_zr(1, 0x3000_0000);
                Ok(())
            }
            0x02 | 0x03 | 0x04 | 0x14 => {
                // SetMemoryPermission / SetMemoryAttribute / MapMemory /
                // UnmapSharedMemory
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x13 => {
                // MapSharedMemory(handle, addr, size, perm): libnx maps the
                // hid service's shared memory this way; back it with a real
                // zeroed buffer and remember where so the host can write
                // gamepad state into the HidSharedMemory layout.
                let addr = self.read_zr(1) as u32;
                let size = self.read_zr(2) as u32;
                self.mem.map_zero(addr, size as usize)?;
                self.hid_shmem_addr = addr;
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x05 => {
                // UnmapMemory(addr, size). hbmenu detects the process address
                // space by unmapping the very top of the 64-bit range and
                // reading the failure code: an out-of-range unmap returns a
                // kernel error whose low bits are 0xd401 (39-bit AArch64) or
                // 0xdc01 (36-bit). Report 39-bit; anything in-range is a no-op
                // success.
                let addr = self.read_zr(0);
                if (addr >> 48) == 0xFFFF {
                    self.write_zr(0, 0x8000_D401);
                } else {
                    self.write_zr(0, RESULT_OK);
                }
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
                 let out = self.read_zr(0) as u32;
                 let addr = self.read_zr(2) as u32;
                 let state = |a: u32| self.mem.page_mapped(a & !0xFFF);
                 let mut base = addr & !0xFFF;
                 while base >= 0x1000 && state(base - 0x1000) == state(base) {
                     base -= 0x1000;
                 }
                 let mapped = state(base);
                 let mut end = base + 0x1000;
                 while end < 0x8000_0000 && state(end) == mapped {
                     end += 0x1000;
                 }
                 let mut info = Vec::with_capacity(40);
                 info.extend_from_slice(&(base as u64).to_le_bytes());
                 info.extend_from_slice(&((end - base) as u64).to_le_bytes());
                 for v in [
                     if mapped { 3u32 } else { 0 }, // type (CodeStatic / Unmapped)
                     0,
                     if mapped { 0b111 } else { 0 }, // perm (RWX / none)
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
                    eprintln!(
                        "[ipc] pc={:#x} h={} svc={:?} cmd={:?}",
                        self.pc, handle, svc_name, cmd_id
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
                        "sm:" | "sm" => self.stub_sm(tls, cmd_id, handle)?,
                        "fsp-srv" | "fsp-srv:" => {
                            // libnx converts fsp-srv to a domain, so the
                            // fs/dir/file sub-sessions come in as object ids on
                            // the same session handle. Route on the recorded
                            // object interface; unknown objects hit the root
                            // stub.
                            let object_id = self.ipc_domain_object_id(tls);
                            match self.domain_interface(handle, object_id) {
                                Some("fsp-srv-fs") => self.stub_fs(tls, cmd_id, handle)?,
                                Some("fsp-srv-fs-dir") => {
                                    self.stub_fs_dir(tls, cmd_id, Self::object_key(handle, object_id))?
                                }
                                Some("fsp-srv-fs-file") => {
                                    self.stub_fs_file(tls, cmd_id, Self::object_key(handle, object_id))?
                                }
                                _ => self.stub_fsp_srv(tls, cmd_id, handle)?,
                            }
                        }
                        // The same interfaces reached over their own session
                        // handle, which is how a caller that never converts to
                        // a domain (libtransistor) uses them.
                        "fsp-srv-fs" => self.stub_fs(tls, cmd_id, handle)?,
                        "fsp-srv-fs-dir" => {
                            self.stub_fs_dir(tls, cmd_id, Self::object_key(handle, 0))?
                        }
                        "fsp-srv-fs-file" => {
                            self.stub_fs_file(tls, cmd_id, Self::object_key(handle, 0))?
                        }
                        "vi:m" | "vi:m:" => self.stub_vi(tls, handle, cmd_id)?,
                        "set" => self.stub_set(tls, cmd_id)?,
                        "nvdrv" | "nvdrv:" | "nvdrv:a" | "nvdrv:a:" | "nvdrv:s" | "nvdrv:t" => {
                            self.nvdrv_request(tls, cmd_id, handle)?
                        }
                        // pl:u, the shared-font service.
                        "pl:u" | "pl:s" => self.stub_pl(tls, cmd_id)?,
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
                            return self.stub_vi(tls, handle, cmd_id);
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
                // GetProcessId
                self.write_zr(0, 1);
                Ok(())
            }
            0x25 => {
                // GetThreadId
                self.write_zr(0, 1);
                Ok(())
            }
            0x26 => {
                // Break: fatal debugger trap — surface and stop.
                self.out.extend_from_slice(b"[svcBreak]\n");
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
                 // 6/7 Total/Used memory, 12/13 Aslr, 14/15 Stack.
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
                     12 => 0x0800_0000, // AslrRegionAddress
                     13 => 0x1F00_0000, // AslrRegionSize
                     14 => 0x1000_0000, // StackRegionAddress
                     15 => 0x0010_0000, // StackRegionSize
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
