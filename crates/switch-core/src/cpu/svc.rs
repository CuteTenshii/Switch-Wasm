//! The Horizon supervisor calls (`SVC`) libnx homebrew issues at runtime.

use super::{
    Cpu, GUEST_ALIAS_REGION_ADDR, GUEST_ALIAS_REGION_SIZE, GUEST_HEAP_REGION_ADDR,
    GUEST_HEAP_REGION_SIZE, GUEST_STACK_REGION_ADDR, GUEST_STACK_REGION_SIZE, HID_SHMEM_SIZE,
    PL_SHMEM_SIZE,
};
use super::ipc::CLOCK_RATES_HZ;
use crate::{Error, Result};
use std::fmt::Write;

impl Cpu {
    /// Every SVC a guest issues, except `svc #0`.
    ///
    /// Horizon numbers its syscalls from 1, so 0 is free, and this reserves it
    /// as a host halt trap: a hand-assembled test program ends with `svc #0`,
    /// and so does the trampoline a program returns to.
    pub(super) fn syscall(&mut self, imm: u16) -> Result<()> {
        if imm == 0 {
            self.halted = true;
            return Ok(());
        }
        self.horizon_syscall(imm)
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
        // Kernel (module 1) description 114: a handle that names nothing.
        const RESULT_INVALID_HANDLE: u64 = 1 | (114 << 9);
        // What `svcGetInfo` reports as the process's memory pool, and the
        // slice of it the kernel reserves for its own per-process bookkeeping
        // (see InfoType 16 below for why that one is zero).
        const TOTAL_MEMORY_SIZE: u64 = 0x1E00_0000;
        const SYSTEM_RESOURCE_SIZE: u64 = 0;
        // The counterpart of `TRACE_IPC` for everything that is not a service
        // request. `svcSendSyncRequest` (0x21) is excluded because `TRACE_IPC`
        // already decodes it, and the two hot ones a running guest issues
        // thousands of times a frame — `svcWaitSynchronization` (0x18) and
        // `svcSleepThread` (0x0b) — would bury everything else.
        if !matches!(imm, 0x21 | 0x18 | 0x0b) && std::env::var("TRACE_SVC").is_ok() {
            eprintln!(
                "[svc] pc={:#x} #{:#04x} x0={:#x} x1={:#x} x2={:#x} x3={:#x}",
                self.pc,
                imm,
                self.read_zr(0),
                self.read_zr(1),
                self.read_zr(2),
                self.read_zr(3)
            );
        }
        match imm {
            0x01 => {
                // SetHeapSize: report a heap at a soft-mapped address, the
                // same one `svcGetInfo`'s HeapRegionAddress names.
                self.write_zr(0, RESULT_OK);
                self.write_zr(1, u64::from(GUEST_HEAP_REGION_ADDR));
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
                if std::env::var("TRACE_MAP").is_ok() {
                    eprintln!("[map] MapMemory dst={dst:#x} src={src:#x} size={size:#x}");
                }
                self.mem.copy_range(dst, src, size)?;
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x2C => {
                // MapPhysicalMemory(address, size): grow the process's heap by
                // backing `[address, address + size)` with physical pages. An
                // application built for the 39-bit address space grows its heap
                // this way rather than through `svcSetHeapSize` — it picks the
                // address itself out of its ASLR region — which is why a retail
                // title never issues syscall 0x01 at all.
                //
                // The pages are left to materialise on first write. `bootstrap`
                // soft-maps the whole low 2 GiB (reads see zeros, a write
                // allocates), so that *is* demand paging, and it is the only
                // workable answer here: `nn::init` asks for everything
                // `svcGetInfo` says is free, which is far more than the
                // emulator's RAM cap, and a title that actually touched all of
                // it could not run on this host either way.
                let addr = self.read_zr(0);
                let size = self.read_zr(1);
                let fits = addr.checked_add(size).is_some_and(|end| end <= u64::from(u32::MAX) + 1);
                if size == 0 || (addr & 0xFFF) != 0 || (size & 0xFFF) != 0 || !fits {
                    self.write_zr(0, RESULT_INVALID_MEMORY_RANGE);
                    return Ok(());
                }
                self.write_zr(0, RESULT_OK);
                Ok(())
            }
            0x2D => {
                // UnmapPhysicalMemory(address, size): the counterpart, and the
                // one direction that has to do real work — the pages go back so
                // the RAM cap sees them freed.
                let addr = self.read_zr(0) as u32;
                let size = self.read_zr(1) as usize;
                self.mem.unmap(addr, size);
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
                // Prefer the handle `hid` actually handed out; the size
                // match stays as a fallback for a caller that never asked for
                // one, which is how this worked before `hid` existed at all.
                if Some(self.read_zr(0)) == self.hid_shmem_handle || size == HID_SHMEM_SIZE {
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
            0x32 => {
                // SetThreadActivity(handle, activity): 0 = Runnable,
                // 1 = Paused. This is `nn::os::SuspendThread`/`ResumeThread`.
                // Horizon refuses to suspend the calling thread, and reports a
                // thread that is already in the requested state rather than
                // treating the call as a no-op.
                const RESULT_BUSY: u64 = 1 | (122 << 9);
                const RESULT_INVALID_STATE: u64 = 1 | (125 << 9);
                let handle = self.read_zr(0);
                let paused = self.read_zr(1) != 0;
                if self.current_thread_handle() == handle {
                    self.write_zr(0, RESULT_BUSY);
                    return Ok(());
                }
                let result = match self.set_thread_paused(handle, paused) {
                    Some(true) => RESULT_OK,
                    Some(false) => RESULT_INVALID_STATE,
                    None => RESULT_INVALID_HANDLE,
                };
                self.write_zr(0, result);
                Ok(())
            }
            0x33 => {
                // GetThreadContext3(out = X0, handle = X1): the suspended
                // thread's whole register file. IL2CPP's collector pairs this
                // with SetThreadActivity to scan the roots living in
                // registers, so it has to be the thread's real state.
                let out = self.read_zr(0) as u32;
                let handle = self.read_zr(1);
                let ok = self.write_thread_context(out, handle);
                self.write_zr(0, if ok { RESULT_OK } else { RESULT_INVALID_HANDLE });
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
                if std::env::var("TRACE_WAIT").is_ok() {
                    eprintln!(
                        "[wait] lock mutex={mutex:#x} owner={owner:#x} self={requester:#x} thread={:#x}",
                        self.current_thread_handle()
                    );
                }
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
                // Read for the trace below, and not acted on: every
                // condition-variable wait this has been seen to make is
                // untimed. A finite one would have to expire against a clock,
                // and the only clock here is the guest's own presents.
                let timeout = self.read_zr(3) as i64;
                if std::env::var("TRACE_WAIT").is_ok() {
                    eprintln!(
                        "[wait] condvar key={key:#x} mutex={mutex:#x} timeout={timeout} thread={:#x}",
                        self.current_thread_handle()
                    );
                }
                self.write_zr(0, RESULT_OK);
                self.wait_process_wide_key(mutex, key, requester);
                Ok(())
            }
            0x1D => {
                // SignalProcessWideKey(key = X0, count = W1)
                let key = self.read_zr(0) as u32;
                let count = self.read_zr(1) as u32 as i32;
                if std::env::var("TRACE_WAIT").is_ok() {
                    eprintln!(
                        "[wait] signal key={key:#x} count={count} thread={:#x}",
                        self.current_thread_handle()
                    );
                }
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
                // WaitSynchronization(out_index, handles, num_handles, timeout):
                // report the wait as immediately satisfied. X1 is the *index*
                // of the handle that signaled, which the libnx wrapper stores
                // to the caller's out pointer — callers index their own waiter
                // array by it, so garbage here sends them to the wrong object.
                // With every object pretended signaled, the first one is the
                // one that signaled: 0. It used to answer 1 unconditionally,
                // which is out of range for the single-handle waits `nnSdk`'s
                // system worker does (`nn::os::detail::MultiWaitImpl::WaitAny`)
                // — it then read a `MultiWaitHolderType` past the end of its
                // list and called its null handler pointer.
                // KERNELRESULT(TimedOut), as libnx spells it.
                const RESULT_TIMED_OUT: u64 = 0xEA01;
                let handles_ptr = self.read_zr(1) as u32;
                let count = (self.read_zr(2) as u32).min(0x40);
                let timeout = self.read_zr(3) as i64;
                let handles: Vec<u64> = (0..count)
                    .map(|i| {
                        u64::from(self.mem.read_u32(handles_ptr.wrapping_add(i * 4)).unwrap_or(0))
                    })
                    .collect();
                if std::env::var("TRACE_WAIT").is_ok() {
                    let named: Vec<String> = handles
                        .iter()
                        .map(|&h| match (self.event_name(h), self.event_signaled(h)) {
                            (Some(name), Some(true)) => format!("{h:#x} {name} signalled"),
                            (Some(name), _) => format!("{h:#x} {name}"),
                            _ => format!("{h:#x} (not an event)"),
                        })
                        .collect();
                    eprintln!("[wait] pc={:#x} timeout={timeout} {named:?}", self.pc);
                }
                // A presented frame is the only periodic tick this emulator
                // has, so it is what drives vsync: the guest's own present is
                // what advances the display, and a render loop waiting on
                // vsync is woken by it rather than by a clock.
                let refreshed = self.cycles.wrapping_sub(self.last_vsync_cycles)
                    >= super::VSYNC_PERIOD_CYCLES;
                if self.nv.gpu.frames != self.last_vsync_frame || refreshed {
                    self.last_vsync_frame = self.nv.gpu.frames;
                    self.last_vsync_cycles = self.cycles;
                    if let Some(vsync) = self.vsync_event {
                        self.signal_event(vsync);
                    }
                }
                // The first handle that is ready. A handle this emulator does
                // not model as an event still counts as ready, which is what
                // keeps thread handles and every unmodelled service handle
                // behaving as they always have.
                let ready = handles
                    .iter()
                    .position(|&h| self.event_signaled(h) != Some(false));
                if let Some(index) = ready {
                    self.consume_event(handles[index]);
                    self.write_zr(0, RESULT_OK);
                    self.write_zr(1, index as u64);
                    self.yield_thread();
                    return Ok(());
                }
                // Every handle names an event, and none of them has fired, so
                // the wait times out. That is the honest answer for a poll
                // (`nn::os::TryWaitSystemEvent`, and libnx's `waitSingle` with
                // no timeout, both issue one), and reporting the wait
                // *satisfied* instead is what told `nn::oe::GpuErrorHandler`
                // that the GPU had faulted, one callback into the SDK's system
                // worker.
                //
                // A blocking wait gets the same answer rather than blocking
                // until something fires, because nothing here can wake it:
                // there is no clock behind the events, only the guest's own
                // presents. `nn::os::detail::MultiWaitImplByHorizon::
                // WaitSynchronizationN` accepts exactly Success, Timeout and
                // Cancelled and asserts on anything else, and answers a
                // timeout by looping — so this degrades to the spin the old
                // always-signalled behaviour already had, without lying about
                // what fired.
                //
                // Note the result is written *before* yielding: `yield_thread`
                // switches register context, so writing X0/X1 after it lands
                // them in the next thread's registers and leaves this one
                // resuming on garbage.
                // A blocking wait really blocks, but only while some other
                // thread can make progress. `nnSdk`'s system worker waits
                // forever on events nothing here fires, and
                // `nn::os::detail::MultiWaitImpl::WaitAny` answers a timeout
                // by returning a **null holder** that
                // `nn::os::RegisterSystemWorkerHandler` then calls without
                // checking — so telling that thread "timed out" jumps to 0,
                // while letting it sleep is both correct and what a real
                // console does.
                //
                // Guarding on another thread being runnable keeps the last
                // runnable thread out of `WaitEvent`, so the process can never
                // block itself entirely.
                if timeout == 0 {
                    self.write_zr(0, RESULT_TIMED_OUT);
                    return Ok(());
                }
                // A wait on **no handles at all** is the one case where
                // neither answer is survivable. `nn::os::detail::
                // MultiWaitImpl::WaitAny` turns whatever comes back into a
                // holder from its own list, and an empty list has none: told
                // "handle 0 fired" it takes index 0 of nothing, told "timed
                // out" it returns the same null, and either way
                // `RegisterSystemWorkerHandler` calls it without checking and
                // the thread jumps to 0.
                //
                // Nothing can ever satisfy such a wait, so the honest thing is
                // not to answer it: rewind onto the `svc` and hand the CPU to
                // somebody who can make progress. The SVC path retires the
                // instruction before dispatching, which is why the PC has to
                // go back a word.
                //
                // "A Short Hike" faults at `pc=0` one instruction after this
                // wait. It always did — the thread that makes it simply never
                // got scheduled until threads started being preempted.
                if handles.is_empty() && self.has_other_runnable() {
                    self.pc = self.pc.wrapping_sub(4);
                    self.yield_thread();
                    return Ok(());
                }
                // A blocking wait on the **vsync** event is the one wait this
                // emulator can honour by actually waiting: the display tick is
                // generated from `cycles` a few lines up, so it is certain to
                // fire, and no other event here has that property. Rewinding
                // onto the `svc` throttles the guest's render loop to the
                // refresh rate.
                //
                // Answering it immediately instead is what kept the Home Menu
                // off the screen. Its frame loop ran at tens of kHz -- 58,547
                // laps of a four-command `pctl` poll in two seconds of console
                // time -- and took 92% of every instruction the process
                // executed, so the threads that prepare the frame it would
                // draw never got the CPU.
                if timeout != 0 && self.vsync_event.is_some_and(|v| handles.contains(&v)) {
                    if !self.has_other_runnable() {
                        // Nothing else can run, so there is no work to overlap
                        // the wait with: idle straight to the next tick instead
                        // of stepping seventeen million instructions that do
                        // nothing. This is the console's own idle, and without
                        // it the throttle costs more than it saves.
                        self.cycles =
                            self.last_vsync_cycles.wrapping_add(super::VSYNC_PERIOD_CYCLES);
                    }
                    self.pc = self.pc.wrapping_sub(4);
                    self.yield_thread();
                    return Ok(());
                }
                self.write_zr(0, RESULT_OK);
                self.write_zr(1, 0);
                self.yield_thread();
                Ok(())
            }
            0x1E => {
                // GetSystemTick: the 19.2 MHz counter every `nn::os` timing
                // API is built on, and the only clock a guest has.
                //
                // The scale is not arbitrary, which is what it used to be
                // (`cycles * 1000`). One emulated instruction stands for one
                // cycle of the 1.02 GHz CPU `apm` reports, so a tick is worth
                // 1_020_000_000 / 19_200_000 of them — about 53. Counting
                // 1000 ticks per instruction instead ran the guest's clock
                // **53,000x fast**: a frame of a hundred thousand
                // instructions read back as five seconds of wall time, and
                // anything that measures its own progress against the tick
                // was being told it had missed every deadline it had.
                const TICK_HZ: u128 = 19_200_000;
                let ticks = u128::from(self.cycles) * TICK_HZ / u128::from(CLOCK_RATES_HZ[0]);
                self.write_zr(0, ticks as u64);
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
                // generic reply below, which answers with a fresh object id.
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
                    // The raw message, for when the fields above do not add
                    // up — the bytes are the only ground truth left.
                    let words: Vec<String> = (0..8)
                        .map(|i| {
                            format!("{:08x}", self.mem.read_u32(tls + i * 4).unwrap_or(0))
                        })
                        .collect();
                    eprintln!("[ipc]   svc={:#x} tls={:#x} {}", imm, tls, words.join(" "));
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
                // CloneCurrentObject (control command 2) and its Ex form (4)
                // duplicate a session: the reply carries a **new session
                // handle as a move handle**, and the clone reaches the same
                // interface, holding the same domain objects, as the original.
                //
                // Every service's control path answered this with a bare
                // success and no handle at all. `nnSdk` clones `fsp-srv`
                // before it mounts anything, so it was left talking to handle
                // 0 and `nn::fs::MountRom("rom", ...)` failed without ever
                // issuing a filesystem command — which surfaced much later as
                // `nn::fs::OpenDirectory("rom:/Data")` reporting that no such
                // mount name was registered.
                if self.ipc_is_control_request(tls) && matches!(cmd_id, Some(2) | Some(4)) {
                    if let Some(name) = svc_name.clone() {
                        let clone = self.alloc_handle();
                        self.record_handle(clone, &name);
                        let objects: Vec<(u32, String)> = self
                            .domain_objects
                            .iter()
                            .filter(|((h, _), _)| *h == handle)
                            .map(|((_, obj), iface)| (*obj, iface.clone()))
                            .collect();
                        for (obj, iface) in objects {
                            self.record_domain_object(clone, obj, &iface);
                        }
                        self.write_ipc_response(tls, 0, &[clone], &[], &[])?;
                        self.write_zr(0, RESULT_OK);
                        return Ok(());
                    }
                }
                // Closing a domain object is a request *shape*, not a
                // command: `CmifDomainRequestType_Close` sits where
                // SendMessage's type byte would, and there is no command id
                // behind it at all. Dispatching one to a service reads
                // whatever follows as command 0 — so the Home Menu's
                // `IStorage` close ran as a **Read**, with the reply's own
                // "SFCO" magic for an offset, and the object stayed open.
                //
                // Thirteen services checked for this themselves and the rest
                // did not, which is the wrong shape for a rule that holds for
                // every domain session there is. It belongs here, before any
                // of them sees the request.
                if self.ipc_is_domain_close(tls) {
                    let object_id = self.ipc_domain_object_id(tls);
                    self.close_domain_object(tls, handle, object_id)?;
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
                                    self.fs_storage_request(tls, handle, cmd_id)?
                                }
                                Some("fsp-srv-save-info-reader") => {
                                    self.fs_save_data_info_reader_request(tls, cmd_id)?
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
                        "fsp-srv-storage" => self.fs_storage_request(tls, handle, cmd_id)?,
                        "fsp-srv-save-info-reader" => {
                            self.fs_save_data_info_reader_request(tls, cmd_id)?
                        }
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
                        // The same `am` sub-interfaces reached over their own
                        // session handle, which is how a caller that never
                        // converts the root session to a domain (`nnSdk`) uses
                        // them — the fsp-srv-fs / time:system-clock split
                        // above, again.
                        "appletOE" | "appletAE"
                        | "am:proxy-service"
                        | "am:application-proxy"
                        | "am:common-state-getter"
                        | "am:self-controller"
                        | "am:window-controller"
                        | "am:audio-controller"
                        | "am:display-controller"
                        | "am:library-applet-creator"
                        | "am:application-functions"
                        | "am:library-applet-proxy"
                        | "am:system-applet-proxy"
                        | "am:library-applet-self-accessor"
                        | "am:applet-common-functions"
                        | "am:process-winding-controller"
                        | "am:home-menu-functions"
                        | "am:global-state-controller"
                        | "am:application-creator"
                        | "am:lock-accessor"
                        | "am:storage"
                        | "am:storage-accessor"
                        | "am:debug-functions" => self.applet_request(tls, handle, cmd_id)?,
                        // nifm at all three privilege levels: `nifm:u` for an
                        // application, `nifm:s` for a system title, `nifm:a`
                        // for the administrator. The same interface behind
                        // each — and only the first was routed here, so a
                        // system title's network calls all went to the generic
                        // fallback.
                        "nifm:u" | "nifm:s" | "nifm:a" => {
                            let object_id = self.ipc_domain_object_id(tls);
                            match self.domain_interface(handle, object_id) {
                                Some("nifm:general-service") => {
                                    self.nifm_general_service_request(tls, handle, cmd_id)?
                                }
                                Some("nifm:request") => {
                                    self.nifm_request_object_request(tls, cmd_id)?
                                }
                                _ => self.nifm_request(tls, cmd_id, handle)?,
                            }
                        }
                        // The same two over their own session handles, which
                        // is how a caller that never converts to a domain
                        // reaches them.
                        "nifm:general-service" => {
                            self.nifm_general_service_request(tls, handle, cmd_id)?
                        }
                        "nifm:request" => self.nifm_request_object_request(tls, cmd_id)?,
                        // ssl, the system TLS stack, and the contexts it
                        // hands out over either route.
                        "ssl" | "ssl:service" | "ssl:context" => {
                            self.ssl_request(tls, handle, cmd_id)?
                        }
                        // hid, and the IAppletResource it hands the input
                        // shared memory over through.
                        "hid" | "hid:dbg" | "hid:sys" | "hid:server"
                        | "hid:applet-resource" => {
                            self.hid_request(tls, handle, cmd_id)?
                        }
                        // lm, the log manager: a title's own diagnostic
                        // output, and its ILogger over either route.
                        "lm" | "lm:service" | "lm:logger" => {
                            self.lm_request(tls, handle, cmd_id)?
                        }
                        // acc, the user accounts: `acc:u0` for an
                        // application, `acc:u1`/`acc:su` for the system side,
                        // plus the profile / manager / async-context objects
                        // they hand out over either route.
                        "acc:u0" | "acc:u1" | "acc:su" | "acc:profile"
                        | "acc:profile-editor" | "acc:manager"
                        | "acc:async-context" | "acc:notifier" => {
                            self.acc_request(tls, handle, cmd_id)?
                        }
                        // ns, the record of what is installed: the getter
                        // services either side of 3.0.0, plus the interfaces
                        // they hand out over either route.
                        "ns:am" | "ns:am2" | "ns:ec" | "ns:rid" | "ns:rt" | "ns:web"
                        | "ns:ro" | "ns:su" | "ns:vm" | "ns:dev" | "ns:app-manager"
                        | "ns:read-only-record" | "ns:read-only-control"
                        | "ns:content-management" | "ns:download-task"
                        | "ns:account-proxy" | "ns:app-version" | "ns:factory-reset"
                        | "ns:ecommerce" | "ns:dynamic-rights" | "ns:document" => {
                            self.ns_request(tls, handle, cmd_id)?
                        }
                        // csrng, the random number generator, and `spl:`,
                        // the security processor it really lives behind.
                        "csrng" => self.csrng_request(tls, cmd_id)?,
                        "spl:" | "spl:mig" | "spl:fs" | "spl:ssl" | "spl:es"
                        | "spl:manu" => self.spl_request(tls, cmd_id)?,
                        // pdm, the play-history database.
                        "pdm:qry" | "pdm:ntfy" | "pdm:info" => {
                            self.pdm_request(tls, cmd_id)?
                        }
                        // pm, the process manager, whose four services are
                        // four different interfaces.
                        "pm:shell" | "pm:dmnt" | "pm:info" | "pm:bm" => {
                            self.pm_request(tls, handle, cmd_id)?
                        }
                        // pcv and clkrst, the clock manager either side of
                        // 8.0.0, plus the per-module sessions clkrst hands out.
                        "pcv" | "clkrst" | "clkrst:i" | "clkrst:session-0"
                        | "clkrst:session-1" | "clkrst:session-2"
                        | "clkrst:session-3" => {
                            self.pcv_request(tls, handle, cmd_id)?
                        }
                        // ts, the temperature sensors, and the ISession
                        // later firmware moved the measurement onto.
                        "ts" | "ts:u" | "ts:s" | "ts:session-internal"
                        | "ts:session-external" => {
                            self.ts_request(tls, handle, cmd_id)?
                        }
                        // sfdnsres, the DNS resolver: the other half of the
                        // socket stack, opened alongside `bsd:u`.
                        "sfdnsres" => self.sfdnsres_request(tls, cmd_id)?,
                        // bsd, the socket service. `bsd:s` is the same
                        // interface at higher privilege.
                        "bsd:u" | "bsd:s" => self.bsd_request(tls, handle, cmd_id)?,
                        // apm, the clock profiles: the manager, the
                        // privileged system manager, and the ISession the
                        // manager hands out.
                        "apm" | "apm:p" | "apm:am" | "apm:sys" | "apm:session" => {
                            self.apm_request(tls, handle, cmd_id)?
                        }
                        // pctl and its aliases, plus the
                        // IParentalControlService reached over its own session
                        // handle (the non-domain route `nnSdk` takes).
                        "pctl" | "pctl:s" | "pctl:a" | "pctl:r" | "pctl:factory"
                        | "pctl:service" => self.pctl_request(tls, handle, cmd_id)?,
                        // audout, the plain PCM output device. `audout:a`
                        // and `audout:d` are the same interface at higher
                        // privilege; nothing here distinguishes them.
                        "audout:u" | "audout:a" | "audout:d" => {
                            self.audout_request(tls, cmd_id)?
                        }
                        "audout:iaudioout" => self.audio_out_request(tls, cmd_id, handle)?,
                        // psc, the power-state manager, and the IPmModule a
                        // module registers with it to be told about a change.
                        "psc:m" | "psc:service" | "psc:module" => {
                            self.psc_request(tls, handle, cmd_id)?
                        }
                        // gpio, the discrete wires into the SoC, and the
                        // IPadSession each one is read through.
                        "gpio" | "gpio:pad" => self.gpio_request(tls, handle, cmd_id)?,
                        // The Mii database, and the separate database of
                        // rendered Mii images. `mii:e` is the editor's
                        // read-write view of the same database `mii:u` reads.
                        "mii:e" | "mii:u" | "mii:s" => self.mii_request(tls, handle, cmd_id)?,
                        "mii:database" | "mii:static" => self.mii_request(tls, handle, cmd_id)?,
                        "miiimg" => self.miiimg_request(tls, cmd_id)?,
                        "audren:u" => self.audren_request(tls, handle, cmd_id)?,
                        "audren:iaudiorenderer" => self.audren_renderer_request(tls, cmd_id, handle)?,
                        "audren:iaudiodevice" => self.audio_device_request(tls, cmd_id, handle)?,
                         name => {
                             // Known service, no dedicated stub: answer with a
                             // sub-session and an object id, so a caller that
                             // expects an out-object gets one it can call
                             // rather than a null it cannot — see
                             // `reply_with_fabricated_object`.
                             //
                             // This used to special-case any service whose name
                             // starts with "applet", handing back the values
                             // ICommonStateGetter's pollers expect (command 1 →
                             // FocusStateChanged, 5 → Handheld, 6 → Normal, 9 →
                             // InFocus) for *whatever* command carried those
                             // ids. `appletOE`/`appletAE` have had a real
                             // dispatch of their own for a while now, so the
                             // guess only ever applied to some other applet
                             // service that would have been answered wrong —
                             // the same way `pl:u`'s GetLoadState once got the
                             // applet message back and left NX-Shell polling it
                             // 190k times.
                             let name = name.to_string();
                             // The control commands first. They are not this
                             // service's commands at all -- every session has
                             // them, whatever is behind it -- and a fabricated
                             // object id is a specific kind of wrong answer to
                             // each: as a *pointer buffer size* it is a large
                             // number, which is how a caller decides to marshal
                             // its buffers as pointer buffers, the one form
                             // this IPC layer does not read. Every service
                             // without a dedicated stub was telling `nifm`,
                             // `friend`, `olsc`, `prepo`, `btm` and the rest to
                             // send their data somewhere nothing looks.
                             if self.ipc_is_control_request(tls) {
                                 match cmd_id {
                                     Some(0) => {
                                         let obj = self.alloc_domain_object();
                                         self.record_domain_object(handle, obj, &name);
                                         self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])?;
                                     }
                                     _ => {
                                         self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[])?;
                                     }
                                 }
                                 self.write_zr(0, RESULT_OK);
                                 return Ok(());
                             }
                             self.warn_no_implementation(&name, cmd_id);
                             self.reply_with_fabricated_object(tls, handle, &name, cmd_id)?
                         }
                    }
                } else {
                    // Unrecognized session handle. The display service's session
                    // handles come from generic object-id replies and aren't in
                    // service_handles, so try the vi stub first; fall back to the
                    // generic object-id reply if the request isn't a vi command
                    // (e.g. hid/time sessions).
                    if let Some(cmd) = cmd_id {
                        // vi commands: GetIApplicationDisplayService (2) and the
                        // display/session commands (100+). The generic reply already
                        // answers ConvertToDomain (0) with a valid object id.
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
                    // A fresh object id, for a caller that expects an
                    // out-object. The applet-state guesses that used to live
                    // here (command 1 → FocusStateChanged, 5 → Handheld, 6 →
                    // Normal, 9 → InFocus) applied to *every* untracked
                    // session, not just an applet one, so `vi`'s and `hid`'s
                    // sessions were being answered with applet state whenever
                    // their command ids happened to collide.
                    self.warn_no_implementation("<untracked session>", cmd_id);
                    let data = {
                        let obj = self.next_object_id;
                        self.next_object_id = obj.wrapping_add(1);
                        obj
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
                // A service that answered "nothing yet" asked to be
                // descheduled. Do it here, after X0 is written: `yield_thread`
                // swaps the register file, so anything written past it would
                // land on whichever thread runs next.
                if std::mem::take(&mut self.pending_yield) {
                    self.yield_thread();
                }
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
                 // build hbmenu is compiled against: 0/1 Core/Priority mask,
                 // 2/3 Alias, 4/5 Heap, 6/7 Total/Used memory,
                 // 11 RandomEntropy, 12/13 Aslr, 14/15 Stack.
                 let info_type = self.read_zr(1);
                 let value = match info_type {
                     // CoreMask / PriorityMask describe what the process is
                     // allowed to schedule on, and they come from the NPDM's
                     // `ThreadInfo` kernel capability. "A Short Hike"'s
                     // `main.npdm` carries the ordinary application values —
                     // cores 0..2 and priorities 28..59 — which is what every
                     // retail application gets. Reporting 0 (the old `_ => 0`
                     // default) makes `nn::os::GetThreadAvailableCoreMask`
                     // hand `nn::os::RegisterSystemWorkerHandler` an empty
                     // mask, whose "highest set bit" scan then asserts.
                     0 => 0b0000_0111, // CoreMask: cores 0, 1, 2
                     1 => 0x0FFF_FFFF_F000_0000, // PriorityMask: 28..=59
                     // Alias/Heap region. Real Horizon puts these far above
                     // the 32-bit range (alias at 0x10_0000_0000, heap at
                     // 0x2_0000_0000) and this used to report those figures
                     // literally — but the emulator addresses guest memory
                     // with a `u32`, so `nnSdk` took the alias address at its
                     // word and asked `svcMapPhysicalMemory` to back
                     // 0x10_0000_0000, which is not a representable address
                     // here. See the region constants for the layout.
                     2 => u64::from(GUEST_ALIAS_REGION_ADDR), // AliasRegionAddress
                     3 => u64::from(GUEST_ALIAS_REGION_SIZE), // AliasRegionSize
                     4 => u64::from(GUEST_HEAP_REGION_ADDR),  // HeapRegionAddress
                     5 => u64::from(GUEST_HEAP_REGION_SIZE),  // HeapRegionSize
                     6 => TOTAL_MEMORY_SIZE, // TotalMemorySize
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
                     // The system resource is the slice of the process's
                     // memory the kernel keeps for its own per-process
                     // bookkeeping (page tables, handle tables), carved out of
                     // the application pool and declared in the NPDM.
                     //
                     // This is **deliberately 0 rather than the 16 MiB "A
                     // Short Hike"'s `main.npdm` asks for**, and it is the one
                     // figure here that does not follow the title's own
                     // manifest. `nnSdk` treats a non-zero answer as "this
                     // process has virtual address memory", and
                     // `nn::os::detail::VammManager::InitializeIfEnabled`
                     // switches the whole heap onto a manager that reserves
                     // address space out of the alias region and backs it a
                     // page at a time — kernel machinery this emulator does
                     // not have. `nn::os::AllocateAddressRegion` then fails
                     // (os result 3-12) and `nn::mem::StandardAllocator`
                     // aborts. Reporting 0 says what is actually true here —
                     // nothing is reserved for the kernel — and puts `nnSdk`
                     // on its plain heap path, which works.
                     16 => SYSTEM_RESOURCE_SIZE, // SystemResourceSizeTotal
                     17 => 0,                    // SystemResourceSizeUsed
                     // Total/UsedNonSystemMemorySize: the same figures as
                     // 6/7 with the system resource taken out, and what
                     // `nnSdk` actually sizes the application heap from —
                     // `nn::init`'s startup asks for
                     // `TotalNonSystem - UsedNonSystem` and hands the result
                     // straight to `nn::mem::StandardAllocator::Initialize`.
                     // Falling into the `_ => 0` default made that
                     // subtraction 0, and the allocator asserts on any span
                     // below its 16 KiB minimum — which is where the retail
                     // boot stopped once `nn::oe::Initialize` was working.
                     21 => TOTAL_MEMORY_SIZE - SYSTEM_RESOURCE_SIZE,
                     22 => 0,
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
                 if std::env::var("TRACE_SVC").is_ok() {
                     eprintln!("[svc]   -> GetInfo({info_type}) = {value:#x}");
                 }
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
