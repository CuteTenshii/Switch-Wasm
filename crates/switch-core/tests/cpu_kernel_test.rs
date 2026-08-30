//! The kernel surface: the syscalls, the scheduler, the guest's threads and
//! the address space they run in.

mod cpu;

use cpu::*;

#[test]
fn bootstrap_provides_stack_and_low_memory() {
    let mut cpu = Cpu::new();
    assert_eq!(cpu.sp(), 0);
    cpu.bootstrap();

    // SP points at the top of the mapped stack. Taken from the constant
    // rather than written out, because where the stack lives has moved once
    // already -- it sits above the heap and the guest's own stack region now,
    // and a literal here just goes stale.
    assert_eq!(cpu.sp(), switch_core::cpu::STACK_TOP);
    cpu.mem
        .write_u64((cpu.sp() - 8) as u32, 0x1234_5678)
        .unwrap();
    assert_eq!(
        cpu.mem.read_u64((cpu.sp() - 8) as u32).unwrap(),
        0x1234_5678
    );

    // Reads from untouched low memory return zero instead of faulting — the
    // exact `ldr x0, [x0]` at 0x244498 a real libnx binary hit.
    assert_eq!(cpu.mem.read_u32(0x244498).unwrap(), 0);
    // Writes allocate a private page on first touch.
    cpu.mem.write_u32(0xb00, 0xDEAD_BEEF).unwrap();
    assert_eq!(cpu.mem.read_u32(0xb00).unwrap(), 0xDEAD_BEEF);
    // The soft region ends where the guest's address space does; reads beyond
    // it still fault, so a pointer that walks off the top of the last region
    // is caught rather than answered with more zeros.
    assert!(cpu
        .mem
        .read_u32(switch_core::cpu::GUEST_SPACE_END + 0xDEAD)
        .is_err());
}

#[test]
fn horizon_syscall_stubs() {
    // OutputDebugString(0x3000, 5) logs the string to the console.
    let mut cpu = cpu_at(0x1000);
    cpu.mem.map(0x3000, b"hello").unwrap();
    cpu.set_reg(0, 0x3000);
    cpu.set_reg(1, 5);
    cpu.mem.map(0x1000, &svc(0x27).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.out, b"hello");

    // A null pointer / bogus length is tolerated (no fault).
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0);
    cpu.set_reg(1, 0xFFFFFFFFFFFFFFDCu64);
    cpu.mem.map(0x1000, &svc(0x27).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();

    // ExitProcess halts the machine.
    let mut cpu = cpu_at(0x1000);
    cpu.mem.map(0x1000, &svc(0x07).to_le_bytes()).unwrap();
    let report = cpu.run(1).unwrap();
    assert!(report.halted);

    // GetSystemTick counts the 19.2 MHz tick every `nn::os` timing API is
    // built on, against the 1.02 GHz CPU `apm` reports -- so one emulated
    // instruction is a *fraction* of a tick, about 1/53. It used to answer
    // `cycles * 1000`, running the guest's clock 53,000x fast: a frame of a
    // hundred thousand instructions read back as five seconds of wall time.
    let mut cpu = cpu_at(0x1000);
    let mut bytes = Vec::new();
    for _ in 0..5300 {
        bytes.extend_from_slice(&nop().to_le_bytes());
    }
    bytes.extend_from_slice(&svc(0x1E).to_le_bytes());
    cpu.mem.map_zero(0x1000, bytes.len() + 0x10).unwrap();
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.run(5301).unwrap();
    assert_eq!(cpu.read_x(0), 5300 * 19_200_000 / 1_020_000_000);

    // ConnectToNamedPort succeeds with a fake handle returned in X1.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x3000); // name pointer (ignored by the stub)
    cpu.set_reg(1, 4);
    cpu.mem.map(0x1000, &svc(0x1F).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);
    assert_eq!(cpu.read_x(1), 0x1000);

    // SendSyncRequest is a no-op success so service init proceeds.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x1000); // session handle
    cpu.set_reg(1, 0x3000); // ipc buffer pointer
    cpu.set_reg(2, 0x40);
    cpu.mem.map(0x1000, &svc(0x21).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);
}

#[test]
fn horizon_query_memory_and_get_info() {
    // QueryMemory writes a MemoryInfo struct to the out pointer and returns
    // the page info in X1. It reports the contiguous run of pages in the same
    // state as the queried address.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x3000); // MemoryInfo out
    cpu.set_reg(1, 0x4000); // PageInfo out
    cpu.set_reg(2, 0x0800_1000); // queried address
    cpu.mem.map(0x1000, &svc(0x06).to_le_bytes()).unwrap();
    cpu.mem.map_zero(0x0800_0000, 0x1_0000).unwrap(); // mapped 64 KiB run
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);
    assert_eq!(cpu.read_x(1), 0x1000); // mapped page info
    assert_eq!(cpu.mem.read_u64(0x3000).unwrap(), 0x0800_0000); // run base
    assert_eq!(cpu.mem.read_u64(0x3008).unwrap(), 0x1_0000); // run size
    assert_eq!(cpu.mem.read_u32(0x3010).unwrap(), 3); // type
    assert_eq!(cpu.mem.read_u32(0x3018).unwrap(), 0b011); // perm (RW-)

    // An untouched soft-mapped page reports as unmapped (type 0, no perm),
    // which is what lets libnx virtmem find free address space.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x3000);
    cpu.set_reg(1, 0x3040);
    cpu.set_reg(2, 0x1234000);
    cpu.mem.map_zero(0x3000, 0x60).unwrap();
    cpu.mem.map(0x1000, &svc(0x06).to_le_bytes()).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(1).unwrap();
    assert_eq!(cpu.mem.read_u32(0x3010).unwrap(), 0); // type (unmapped)
    assert_eq!(cpu.mem.read_u32(0x3018).unwrap(), 0); // perm

    // GetInfo returns the requested value in X1 (the libnx wrapper stores it
    // to the out pointer). InfoType 4 = HeapRegionAddress. Every region this
    // reports has to be an address the emulator can actually represent: guest
    // memory is addressed with a `u32`, and reporting Horizon's real
    // out-of-range region bases had `nnSdk` asking `svcMapPhysicalMemory` to
    // back 0x10_0000_0000.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, 4); // infoType
    cpu.set_reg(2, 0xffff_8001); // CUR_PROCESS_HANDLE
    cpu.mem.map(0x1000, &svc(0x29).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);
    assert_eq!(
        cpu.read_x(1),
        u64::from(switch_core::cpu::GUEST_HEAP_REGION_ADDR)
    );
    assert!(cpu.read_x(1) <= u64::from(u32::MAX));

    // InfoType 21/22 = Total/UsedNonSystemMemorySize, which is what `nnSdk`
    // sizes the application heap from — it hands the difference straight to
    // `nn::mem::StandardAllocator::Initialize`, which asserts on a span under
    // 16 KiB. Answering 0 (the old `_ => 0` default) made that difference 0.
    let total = u64::from(switch_core::cpu::GUEST_TOTAL_MEMORY_SIZE);
    for (info_type, expected) in [(21u64, total), (22, 0)] {
        let mut cpu = cpu_at(0x1000);
        cpu.set_reg(1, info_type);
        cpu.set_reg(2, 0xffff_8001);
        cpu.mem.map(0x1000, &svc(0x29).to_le_bytes()).unwrap();
        cpu.run(1).unwrap();
        assert_eq!(cpu.read_x(0), 0);
        assert_eq!(cpu.read_x(1), expected);
    }

    // InfoType 16 = SystemResourceSizeTotal, and this query is the whole of
    // what switches `nnSdk` onto its virtual address memory manager —
    // `IsVirtualAddressMemoryEnabled` is it succeeding and returning non-zero,
    // nothing else. A process whose NPDM declared nothing must read 0 here, or
    // it is put on a manager it never asked for and charged the address space
    // that costs.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, 16);
    cpu.set_reg(2, 0xffff_8001);
    cpu.mem.map(0x1000, &svc(0x29).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);
    assert_eq!(cpu.read_x(1), 0);

    // InfoType 6 = TotalMemorySize.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, 6);
    cpu.set_reg(2, 0xffff_8001);
    cpu.mem.map(0x1000, &svc(0x29).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);
    assert_eq!(cpu.read_x(1), total);

    // A title whose NPDM declares a system resource is told a different
    // address space entirely — the alias region has to carry the SDK's arena
    // as well as the heap, so it grows and the total shrinks to pay for it.
    // Just Dance 2023 declares 16 MiB and Just Dance 2019 declares 0, and
    // handing either one the other's figures breaks it: the first aborts in
    // AllocateAddressRegion, the second in its own allocator 378M steps in.
    use switch_core::cpu::{
        VAMM_ALIAS_REGION_ADDR, VAMM_ALIAS_REGION_SIZE, VAMM_SYSTEM_RESOURCE_SIZE,
        VAMM_TOTAL_MEMORY_SIZE,
    };
    for (info_type, expected) in [
        (6u64, u64::from(VAMM_TOTAL_MEMORY_SIZE)),
        (16, u64::from(VAMM_SYSTEM_RESOURCE_SIZE)),
        (
            21,
            u64::from(VAMM_TOTAL_MEMORY_SIZE - VAMM_SYSTEM_RESOURCE_SIZE),
        ),
        (2, u64::from(VAMM_ALIAS_REGION_ADDR)),
        (3, u64::from(VAMM_ALIAS_REGION_SIZE)),
    ] {
        let mut cpu = cpu_at(0x1000);
        cpu.set_system_resource_size(VAMM_SYSTEM_RESOURCE_SIZE);
        cpu.set_reg(1, info_type);
        cpu.set_reg(2, 0xffff_8001);
        cpu.mem.map(0x1000, &svc(0x29).to_le_bytes()).unwrap();
        cpu.run(1).unwrap();
        assert_eq!(cpu.read_x(0), 0);
        assert_eq!(
            cpu.read_x(1),
            expected,
            "InfoType {info_type} under the VAMM layout"
        );
    }

    // InfoType 12 = AslrRegionAddress.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, 12);
    cpu.set_reg(2, 0xffff_8001);
    cpu.mem.map(0x1000, &svc(0x29).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(1), 0x0800_0000);
}

#[test]
fn horizon_map_physical_memory() {
    use switch_core::cpu::GUEST_ALIAS_REGION_ADDR;
    // MapPhysicalMemory(address, size) is how an application built for the
    // 39-bit address space grows its heap — it picks the address itself out of
    // the alias region rather than calling svcSetHeapSize, which is why a
    // retail title never issues syscall 0x01 at all.
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.mem.map(0x1000, &svc(0x2c).to_le_bytes()).unwrap();
    cpu.set_reg(0, u64::from(GUEST_ALIAS_REGION_ADDR));
    cpu.set_reg(1, 0x10_0000);
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);
    // The pages are demand-allocated, so the range reads as zeros and costs
    // nothing until it is written to.
    assert_eq!(cpu.mem.read_u32(GUEST_ALIAS_REGION_ADDR).unwrap(), 0);

    // An unaligned or empty range is rejected, and so is one the emulator
    // cannot address: guest memory is indexed with a `u32`, and silently
    // truncating Horizon's real alias base (0x10_0000_0000) to 0 would map the
    // heap over the null page.
    for (addr, size) in [
        (u64::from(GUEST_ALIAS_REGION_ADDR), 0u64),
        (u64::from(GUEST_ALIAS_REGION_ADDR) + 1, 0x1000),
        (u64::from(GUEST_ALIAS_REGION_ADDR), 0x800),
        (0x10_0000_0000, 0x1000),
    ] {
        let mut cpu = cpu_at(0x1000);
        cpu.bootstrap();
        cpu.set_pc(0x1000);
        cpu.mem.map(0x1000, &svc(0x2c).to_le_bytes()).unwrap();
        cpu.set_reg(0, addr);
        cpu.set_reg(1, size);
        cpu.run(1).unwrap();
        assert_ne!(cpu.read_x(0), 0, "{addr:#x}+{size:#x} should be rejected");
    }
}

#[test]
fn map_memory_backs_the_destination_and_unmap_frees_it() {
    // svcMapMemory(dst, src, size) aliases a range; libnx uses it to mirror a
    // thread's stack and then finds the *next* thread's mirror by looking for an
    // unmapped range, so the destination has to read back as mapped memory
    // holding the source's bytes. svcUnmapMemory hands them back and frees it.
    const SRC: u32 = 0x3000_0000;
    const DST: u32 = 0x1800_0000;
    let mut cpu = cpu_at(0x1000);
    cpu.mem.map(SRC, &0xDEAD_BEEFu32.to_le_bytes()).unwrap();
    cpu.mem.map(0x1000, &svc(0x04).to_le_bytes()).unwrap();
    cpu.mem.map(0x1004, &svc(0x05).to_le_bytes()).unwrap();

    cpu.set_reg(0, DST as u64);
    cpu.set_reg(1, SRC as u64);
    cpu.set_reg(2, 0x2000);
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);
    assert!(cpu.mem.page_mapped(DST));
    assert!(cpu.mem.page_mapped(DST + 0x1000));
    assert_eq!(cpu.mem.read_u32(DST).unwrap(), 0xDEAD_BEEF);

    cpu.mem.write_u32(DST, 0x1234_5678).unwrap();
    cpu.set_reg(0, DST as u64);
    cpu.set_reg(1, SRC as u64);
    cpu.set_reg(2, 0x2000);
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);
    assert_eq!(cpu.mem.read_u32(SRC).unwrap(), 0x1234_5678);
    assert!(!cpu.mem.page_mapped(DST));
}

#[test]
fn query_memory_writes_40_byte_memoryinfo() {
    // svc 0x06 (QueryMemory) writes a 40-byte MemoryInfo
    // {base(u64), size(u64), type/attr/perm/device/ipc/padding(u32 each)} —
    // NOT 8 x u64. The old stub wrote 64 bytes, overflowing the struct by 24
    // bytes; when the app's info pointer sat near the top of its stack this
    // clobbered main's saved LR and made NX-Shell's main "return" to 0.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x3000); // info out pointer
    cpu.set_reg(1, 0x3040); // page info out pointer
    cpu.set_reg(2, 0x1234000); // address
    cpu.mem.map_zero(0x1234000, 0x1000).unwrap();
    cpu.mem.map_zero(0x3000, 0x60).unwrap();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&svc(0x06).to_le_bytes());
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(1).unwrap();
    // Only the first 40 bytes are written; byte 40+ must stay untouched (0).
    assert_eq!(cpu.mem.read_u64(0x3000).unwrap(), 0x1234000);
    assert_eq!(cpu.mem.read_u64(0x3008).unwrap(), 0x1000);
    assert_eq!(cpu.mem.read_u32(0x3010).unwrap(), 3); // type (mapped)
    assert_eq!(cpu.mem.read_u32(0x3014).unwrap(), 0); // attr
    assert_eq!(cpu.mem.read_u32(0x3018).unwrap(), 0b011); // perm (RW-)
    assert_eq!(cpu.mem.read_u32(0x301c).unwrap(), 0); // device_refcount
    assert_eq!(cpu.mem.read_u32(0x3020).unwrap(), 0); // ipc_refcount
    assert_eq!(cpu.mem.read_u32(0x3024).unwrap(), 0); // padding

    // An untouched soft-mapped page reports as unmapped (type 0, no perm),
    // which is what lets libnx virtmem find free address space.
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(0, 0x3000);
    cpu.set_reg(1, 0x3040);
    cpu.set_reg(2, 0x1234000);
    cpu.mem.map_zero(0x3000, 0x60).unwrap();
    cpu.mem.map(0x1000, &svc(0x06).to_le_bytes()).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(1).unwrap();
    // 0x1234000 is inside the soft-mapped range but never written -> unmapped.
    assert_eq!(cpu.mem.read_u32(0x3010).unwrap(), 0); // type (unmapped)
    assert_eq!(cpu.mem.read_u32(0x3018).unwrap(), 0); // perm
                                                      // The old bug wrote 24 more bytes here; 0x3028+ must be untouched zeros.
    assert_eq!(cpu.mem.read_u64(0x3028).unwrap(), 0);
    assert_eq!(cpu.mem.read_u64(0x3040).unwrap(), 0); // pageinfo written via x1? no, x1 holds it
    assert_eq!(cpu.read_x(1), 0); // unmapped soft page -> page info 0
}

#[test]
fn query_memory_gives_the_execute_bit_only_to_module_text() {
    // Retail `rtld` finds the other loaded modules by walking QueryMemory and
    // keeping every CodeStatic region that is executable, then reading the
    // candidate's word at +4 as the offset to its `MOD0` signature. While
    // every mapped page reported RWX, the first writable region also looked
    // executable — and `rtld`'s own `.rodata` opens with a note whose second
    // word is 0x1c, exactly where `MOD0` sits from there. `rtld` accepted it
    // as a module, relocated itself a second time against a base 0x3000 past
    // its real one, and ran off the end of the address space.
    //
    // So: `.text` reports R-X, the pages after it report RW-, and the two are
    // separate regions.
    let text = 0x0800_0000u32;
    let rodata = 0x0800_3000u32;
    let mut cpu = cpu_at(0x1000);
    cpu.mem.map(0x1000, &svc(0x06).to_le_bytes()).unwrap();
    cpu.mem.map_zero(text, 0x6000).unwrap();
    cpu.mem.mark_readonly(text, rodata);

    cpu.set_reg(0, 0x3000); // MemoryInfo out
    cpu.set_reg(1, 0x4000); // PageInfo out
    cpu.set_reg(2, (text + 0x1000) as u64);
    cpu.run(1).unwrap();
    assert_eq!(cpu.mem.read_u64(0x3000).unwrap(), text as u64);
    assert_eq!(cpu.mem.read_u64(0x3008).unwrap(), (rodata - text) as u64);
    assert_eq!(cpu.mem.read_u32(0x3010).unwrap(), 3); // type (CodeStatic)
    assert_eq!(cpu.mem.read_u32(0x3018).unwrap(), 0b101); // perm (R-X)

    let mut cpu = cpu_at(0x1000);
    cpu.mem.map(0x1000, &svc(0x06).to_le_bytes()).unwrap();
    cpu.mem.map_zero(text, 0x6000).unwrap();
    cpu.mem.mark_readonly(text, rodata);
    cpu.set_reg(0, 0x3000);
    cpu.set_reg(1, 0x4000);
    cpu.set_reg(2, rodata as u64);
    cpu.run(1).unwrap();
    assert_eq!(cpu.mem.read_u64(0x3000).unwrap(), rodata as u64);
    assert_eq!(cpu.mem.read_u32(0x3018).unwrap() & 0b100, 0); // not executable
}

#[test]
fn guest_threads_run_and_hand_over_at_blocking_syscalls() {
    // A guest program that creates a thread, starts it, and waits for it to set
    // a flag. Thread creation used to hand out a fake handle and never run
    // anything, so the wait spun forever — which is where hbmenu stopped.
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.mem.map_zero(0x4000, 0x2000).unwrap(); // the child's stack
    cpu.mem.map_zero(0x6000, 0x1000).unwrap(); // the flag and the arg it saw

    // main: svcCreateThread(entry = 0x2000, arg = 0x1234, stack_top = 0x5000),
    // svcStartThread, then sleep until the flag is set and exit.
    let main = [
        0xd284_0001u32, // mov x1, #0x2000  (entry)
        0xd282_4682,    // mov x2, #0x1234  (arg)
        0xd28a_0003,    // mov x3, #0x5000  (stack top)
        0x5280_0764,    // mov w4, #0x3b    (priority)
        0x1280_0005,    // mov w5, #-1      (core)
        0xd400_0101,    // svc #8           (CreateThread → handle in x1)
        0xaa01_03e0,    // mov x0, x1
        0xd400_0121,    // svc #9           (StartThread)
        0xd400_0161,    // svc #0xb         (SleepThread → yields)
        0xd28c_0009,    // mov x9, #0x6000
        0xb940_0122,    // ldr w2, [x9]
        0x34ff_ffa2,    // cbz w2, -12      (back to the sleep)
        0xd400_00e1,    // svc #7           (ExitProcess)
    ];
    // child: record the argument it was passed, set the flag, exit.
    let child = [
        0xd28c_0009u32, // mov x9, #0x6000
        0xb900_0520,    // str w0, [x9, #4]
        0x5280_0aa1,    // mov w1, #0x55
        0xb900_0121,    // str w1, [x9]
        0xd400_0141,    // svc #0xa         (ExitThread)
    ];
    let bytes = |code: &[u32]| -> Vec<u8> { code.iter().flat_map(|i| i.to_le_bytes()).collect() };
    cpu.mem.map_zero(0x1000, 0x100).unwrap();
    cpu.mem.map(0x1000, &bytes(&main)).unwrap();
    cpu.mem.map_zero(0x2000, 0x100).unwrap();
    cpu.mem.map(0x2000, &bytes(&child)).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(10_000).unwrap();

    assert!(
        cpu.halted,
        "main should reach ExitProcess once the child ran"
    );
    assert_eq!(
        cpu.mem.read_u32(0x6000).unwrap(),
        0x55,
        "the child set the flag"
    );
    assert_eq!(
        cpu.mem.read_u32(0x6004).unwrap(),
        0x1234,
        "with its argument in x0"
    );
    assert_eq!(cpu.thread_count(), 2);
}

#[test]
fn the_address_arbiter_compares_before_it_waits() {
    // `svcWaitForAddress`/`svcSignalToAddress`, the pair `nn::os` builds its
    // semaphores, barriers and newer condition variables out of. Neither one
    // waits or wakes unconditionally: each first compares the word in guest
    // memory against the value the caller passed, and reports InvalidState
    // when it does not match — which the caller reads as "already happened".
    const RESULT_INVALID_STATE: u64 = 1 | (125 << 9);
    const RESULT_TIMED_OUT: u64 = 0xEA01;

    // DecrementAndWaitIfLessThan with a zero timeout. The predicate holds
    // (0 < 1) so the decrement happens — that is how a semaphore's waiter
    // claims its place in the queue — but a zero timeout asked whether it
    // *would* block, not to block.
    let mut cpu = cpu_at(0x1000);
    cpu.mem.map(0x1000, &svc(0x34).to_le_bytes()).unwrap();
    cpu.mem.map_zero(0x6000, 0x1000).unwrap();
    cpu.set_reg(0, 0x6000);
    cpu.set_reg(1, 1); // DecrementAndWaitIfLessThan
    cpu.set_reg(2, 1); // value
    cpu.set_reg(3, 0); // timeout
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), RESULT_TIMED_OUT);
    assert_eq!(cpu.mem.read_u32(0x6000).unwrap(), (-1i32) as u32);

    // WaitIfEqual against a word holding something else: no wait, and the
    // word is left alone.
    let mut cpu = cpu_at(0x1000);
    cpu.mem.map(0x1000, &svc(0x34).to_le_bytes()).unwrap();
    cpu.mem.map_zero(0x6000, 0x1000).unwrap();
    cpu.mem.write_u32(0x6000, 7).unwrap();
    cpu.set_reg(0, 0x6000);
    cpu.set_reg(1, 2); // WaitIfEqual
    cpu.set_reg(2, 5);
    cpu.set_reg(3, u64::MAX); // wait forever
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), RESULT_INVALID_STATE);
    assert_eq!(cpu.mem.read_u32(0x6000).unwrap(), 7);

    // SignalAndIncrementIfEqual moves the word on when it matches...
    let mut cpu = cpu_at(0x1000);
    cpu.mem.map(0x1000, &svc(0x35).to_le_bytes()).unwrap();
    cpu.mem.map_zero(0x6000, 0x1000).unwrap();
    cpu.mem.write_u32(0x6000, 7).unwrap();
    cpu.set_reg(0, 0x6000);
    cpu.set_reg(1, 1); // SignalAndIncrementIfEqual
    cpu.set_reg(2, 7);
    cpu.set_reg(3, 1); // count
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);
    assert_eq!(cpu.mem.read_u32(0x6000).unwrap(), 8);

    // ...and refuses when it does not, without touching it.
    let mut cpu = cpu_at(0x1000);
    cpu.mem.map(0x1000, &svc(0x35).to_le_bytes()).unwrap();
    cpu.mem.map_zero(0x6000, 0x1000).unwrap();
    cpu.mem.write_u32(0x6000, 8).unwrap();
    cpu.set_reg(0, 0x6000);
    cpu.set_reg(1, 1);
    cpu.set_reg(2, 99);
    cpu.set_reg(3, 1);
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), RESULT_INVALID_STATE);
    assert_eq!(cpu.mem.read_u32(0x6000).unwrap(), 8);
}

#[test]
fn blocking_on_the_arbiter_leaves_the_next_thread_its_registers() {
    // A thread that blocks in `svcWaitForAddress` hands the CPU over inside
    // the syscall, so the syscall's *own* result has to be in X0 before that
    // happens — after it, X0 belongs to whoever took over.
    //
    // Writing it afterwards zeroed the incoming thread's X0. For Tomodachi
    // Life that thread was one `nn::os` had just started, and X0 was the
    // `ThreadType` its entry stub installs at TLS+0x1F8 — so the thread ran
    // with a null current-thread pointer, read its own handle as 0, and every
    // unlocked mutex it took (lock word 0) compared equal to one it already
    // held. `pthread_mutex_lock` aborts on that.
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.mem.map_zero(0x4000, 0x2000).unwrap(); // the child's stack
    cpu.mem.map_zero(0x6000, 0x1000).unwrap();

    // main: arm the arbiter word, start the child, then wait on it.
    let main = [
        0xd28c_0009u32, // mov x9, #0x6000
        0x5280_0021,    // mov w1, #1
        0xb900_0121,    // str w1, [x9]     (the word the child will signal)
        0xd284_0001,    // mov x1, #0x2000  (entry)
        0xd282_4682,    // mov x2, #0x1234  (arg — what must survive)
        0xd28a_0003,    // mov x3, #0x5000  (stack top)
        0x5280_0764,    // mov w4, #0x3b    (priority)
        0x1280_0005,    // mov w5, #-1      (core)
        0xd400_0101,    // svc #8           (CreateThread → handle in x1)
        0xaa01_03e0,    // mov x0, x1
        0xd400_0121,    // svc #9           (StartThread)
        0xd28c_0000,    // mov x0, #0x6000
        0x5280_0041,    // mov w1, #2       (WaitIfEqual)
        0x5280_0022,    // mov w2, #1       (value)
        0x9280_0003,    // mov x3, #-1      (wait forever)
        0xd400_0681,    // svc #0x34        (WaitForAddress → blocks)
        0xd28c_0009,    // mov x9, #0x6000
        0x5280_0aa1,    // mov w1, #0x55
        0xb900_0d21,    // str w1, [x9, #12] (main got the CPU back)
        0xd400_00e1,    // svc #7           (ExitProcess)
    ];
    // child: record the argument it was handed, then release main.
    let child = [
        0xd28c_0009u32, // mov x9, #0x6000
        0xb900_0520,    // str w0, [x9, #4]  (the ThreadType stand-in)
        0x5280_0041,    // mov w1, #2
        0xb900_0121,    // str w1, [x9]      (so main's predicate stops holding)
        0xd28c_0000,    // mov x0, #0x6000
        0x5280_0021,    // mov w1, #1        (SignalAndIncrementIfEqual)
        0x5280_0042,    // mov w2, #2        (the value it must still hold)
        0x5280_0023,    // mov w3, #1        (wake one)
        0xd400_06a1,    // svc #0x35         (SignalToAddress)
        0xd400_0141,    // svc #0xa          (ExitThread)
    ];
    let bytes = |code: &[u32]| -> Vec<u8> { code.iter().flat_map(|i| i.to_le_bytes()).collect() };
    cpu.mem.map_zero(0x1000, 0x100).unwrap();
    cpu.mem.map(0x1000, &bytes(&main)).unwrap();
    cpu.mem.map_zero(0x2000, 0x100).unwrap();
    cpu.mem.map(0x2000, &bytes(&child)).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(10_000).unwrap();

    assert_eq!(
        cpu.mem.read_u32(0x6004).unwrap(),
        0x1234,
        "the thread that took the CPU kept the argument in its x0"
    );
    assert_eq!(
        cpu.mem.read_u32(0x6000).unwrap(),
        3,
        "the signal's compare-and-increment ran"
    );
    assert_eq!(cpu.mem.read_u32(0x600c).unwrap(), 0x55, "main was woken");
    assert!(cpu.halted, "main reached ExitProcess");
}

#[test]
fn a_wait_on_no_handles_is_not_answered() {
    // `nn::os::detail::MultiWaitImpl::WaitAny` turns whatever
    // svcWaitSynchronization returns into a holder from its own list. An empty
    // list has none, so *either* answer is fatal: told "handle 0 fired" it
    // takes index 0 of nothing, told "timed out" it returns the same null, and
    // `RegisterSystemWorkerHandler` calls it without checking. "A Short Hike"
    // faults at pc=0 one instruction later.
    //
    // Nothing can ever satisfy a wait on nothing, so it is not answered at
    // all: the thread parks on the syscall and the CPU goes to somebody who
    // can make progress.
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.mem.map_zero(0x4000, 0x2000).unwrap(); // the child's stack
    cpu.mem.map_zero(0x6000, 0x1000).unwrap();

    let main = [
        0xd284_0001u32, // mov x1, #0x2000  (entry)
        0xaa1f_03e2,    // mov x2, xzr      (arg)
        0xd28a_0003,    // mov x3, #0x5000  (stack top)
        0x5280_0764,    // mov w4, #0x3b
        0x1280_0005,    // mov w5, #-1
        0xd400_0101,    // svc #8           (CreateThread)
        0xaa01_03e0,    // mov x0, x1
        0xd400_0121,    // svc #9           (StartThread)
        0xd400_0161,    // svc #0xb         (SleepThread -> hands over)
        0xd28c_0009,    // mov x9, #0x6000
        0x5280_0aa1,    // mov w1, #0x55
        0xb900_0121,    // str w1, [x9]
        0xd400_00e1,    // svc #7           (ExitProcess)
    ];
    // The child waits on an empty handle set, forever, and must never get past
    // it to record that it did.
    let child = [
        0xd28c_0001u32, // mov x1, #0x6000  (handles pointer, unread)
        0xaa1f_03e2,    // mov x2, xzr      (**no handles**)
        0x9280_0003,    // mov x3, #-1      (no timeout)
        0xd400_0301,    // svc #0x18        (WaitSynchronization)
        0xd28c_0009,    // mov x9, #0x6000
        0x5285_0ba1,    // mov w1, #0x285d
        0xb900_0521,    // str w1, [x9, #4]
        0xd400_0141,    // svc #0xa         (ExitThread)
    ];
    let bytes = |code: &[u32]| -> Vec<u8> { code.iter().flat_map(|i| i.to_le_bytes()).collect() };
    cpu.mem.map_zero(0x1000, 0x100).unwrap();
    cpu.mem.map(0x1000, &bytes(&main)).unwrap();
    cpu.mem.map_zero(0x2000, 0x100).unwrap();
    cpu.mem.map(0x2000, &bytes(&child)).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(500_000).unwrap();

    assert!(cpu.halted, "main never got the CPU back");
    assert_eq!(cpu.mem.read_u32(0x6000).unwrap(), 0x55);
    assert_eq!(
        cpu.mem.read_u32(0x6004).unwrap(),
        0,
        "the wait on nothing was answered, and the thread ran on past it"
    );
}

#[test]
fn a_blocking_wait_parks_rather_than_re_asking() {
    // A wait that cannot be satisfied yet used to rewind onto the `svc` and
    // hand the CPU on, so the thread re-asked on every scheduler slice. Only a
    // signal can change the answer, so each of those laps learned nothing --
    // and a display period is seventeen million cycles of them. The Home Menu
    // spent 131 of every 170 million steps getting to its tenth frame, and
    // Just Dance 2023 sat two threads in waits nothing ever satisfies and gave
    // them 70% of every instruction it retired.
    const VI: u64 = 0xB500;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(VI, "vi:m");
    let tls = cpu.tls_base();

    ipc_request_plain(&mut cpu, VI, 2, &[]);
    let display = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    ipc_request_plain(&mut cpu, display, 5202, &[]);
    let vsync = cpu.mem.read_u32(tls + 0x0c).unwrap();
    assert_ne!(vsync, 0);

    // One thread, waiting on the display for as long as that takes.
    cpu.mem.map_zero(0x6000, 0x1000).unwrap();
    cpu.mem.write_u32(0x6000, vsync).unwrap();
    let code = [
        0xd28c_0001u32, // mov x1, #0x6000  (the handle list)
        0xd280_0022,    // mov x2, #1       (one handle)
        0x9280_0003,    // mov x3, #-1      (no timeout)
        0xd400_0301,    // svc #0x18        (WaitSynchronization)
        0xd28c_0009,    // mov x9, #0x6000
        0x5280_0aa1,    // mov w1, #0x55
        0xb900_0521,    // str w1, [x9, #4]
        0xd400_00e1,    // svc #7           (ExitProcess)
    ];
    let bytes: Vec<u8> = code.iter().flat_map(|i| i.to_le_bytes()).collect();
    cpu.mem.map_zero(0x2000, 0x100).unwrap();
    cpu.mem.map(0x2000, &bytes).unwrap();
    cpu.set_pc(0x2000);
    let before = cpu.steps;
    cpu.run(1_000_000).unwrap();

    assert_eq!(
        cpu.mem.read_u32(0x6004).unwrap(),
        0x55,
        "the wait never ended"
    );
    assert!(
        cpu.cycles >= switch_core::cpu::VSYNC_PERIOD_CYCLES,
        "the wait ended before the display could have refreshed, so it ended for \
         the wrong reason"
    );
    // The clock covered a whole refresh. The CPU did not have to execute one.
    let retired = cpu.steps - before;
    assert!(
        retired < 1000,
        "the wait re-asked its way through the period: {retired} steps"
    );
}

#[test]
fn a_thread_that_never_blocks_is_still_taken_off_the_cpu() {
    // Threads used to hand over only at a blocking syscall, so a thread that
    // runs a long stretch of arithmetic between two of them kept the CPU for
    // all of it. That is not a fairness nicety: a system applet's audio thread
    // renders a whole buffer per AppendAudioOutBuffer, and took **99.9% of
    // every instruction executed** -- the Mii editor's own main loop got the
    // other 0.1%, which is why three applets could boot, open a layer, play
    // their music and never reach a frame.
    //
    // Here the child never makes a syscall at all, which is the same problem
    // with the dial turned all the way up.
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.mem.map_zero(0x4000, 0x2000).unwrap(); // the child's stack
    cpu.mem.map_zero(0x6000, 0x1000).unwrap(); // the flag main sets at the end

    let main = [
        0xd284_0001u32, // mov x1, #0x2000  (entry)
        0xaa1f_03e2,    // mov x2, xzr      (arg)
        0xd28a_0003,    // mov x3, #0x5000  (stack top)
        0x5280_0764,    // mov w4, #0x3b    (priority)
        0x1280_0005,    // mov w5, #-1      (core)
        0xd400_0101,    // svc #8           (CreateThread -> handle in x1)
        0xaa01_03e0,    // mov x0, x1
        0xd400_0121,    // svc #9           (StartThread)
        0xd400_0161,    // svc #0xb         (SleepThread -> hands over)
        0xd28c_0009,    // mov x9, #0x6000
        0x5280_0aa1,    // mov w1, #0x55
        0xb900_0121,    // str w1, [x9]
        0xd400_00e1,    // svc #7           (ExitProcess)
    ];
    // The child spins forever and never asks the kernel for anything.
    let child = [0x1400_0000u32]; // b .
    let bytes = |code: &[u32]| -> Vec<u8> { code.iter().flat_map(|i| i.to_le_bytes()).collect() };
    cpu.mem.map_zero(0x1000, 0x100).unwrap();
    cpu.mem.map(0x1000, &bytes(&main)).unwrap();
    cpu.mem.map_zero(0x2000, 0x100).unwrap();
    cpu.mem.map(0x2000, &bytes(&child)).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(200_000).unwrap();

    assert!(cpu.halted, "the spinning child never gave the CPU back");
    assert_eq!(cpu.mem.read_u32(0x6000).unwrap(), 0x55);
}

#[test]
fn a_timed_wait_expires_while_the_other_threads_hand_the_cpu_round() {
    // Timed waits used to be swept on the preemption tick, and the counter
    // behind that tick is reset by every context switch. So a process with two
    // threads that hand the CPU over more often than once every `TIME_SLICE`
    // instructions never reached the sweep at all, and *nothing that slept
    // ever woke up*.
    //
    // That is the ordinary shape of a stalled applet, not a corner case: three
    // of the Album applet's threads sat on an `svcWaitSynchronization` this
    // emulator cannot satisfy, yielding after a handful of instructions each,
    // and its main thread's 10 ms sleep was still outstanding 480 million
    // instructions later, with the whole process frozen around it.
    //
    // Here two children do nothing but yield, and the parent parks on the
    // address arbiter for 50 us. Nothing will ever signal that address, so the
    // only thing that can start the parent again is the sweep.
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.mem.map_zero(0x4000, 0x4000).unwrap(); // the two children's stacks
    cpu.mem.map_zero(0x9000, 0x1000).unwrap(); // the flag the parent sets

    let main = [
        0xd284_0001u32, // mov x1, #0x2000  (entry)
        0xaa1f_03e2,    // mov x2, xzr      (arg)
        0xd28c_0003,    // mov x3, #0x6000  (stack top)
        0x5280_0764,    // mov w4, #0x3b    (priority)
        0x1280_0005,    // mov w5, #-1      (core)
        0xd400_0101,    // svc #8           (CreateThread -> handle in x1)
        0xaa01_03e0,    // mov x0, x1
        0xd400_0121,    // svc #9           (StartThread)
        0xd284_0001,    // mov x1, #0x2000
        0xaa1f_03e2,    // mov x2, xzr
        0xd290_0003,    // mov x3, #0x8000  (the second child's stack top)
        0x5280_0764,    // mov w4, #0x3b
        0x1280_0005,    // mov w5, #-1
        0xd400_0101,    // svc #8
        0xaa01_03e0,    // mov x0, x1
        0xd400_0121,    // svc #9
        0xd292_0080,    // mov x0, #0x9004  (a word that is zero)
        0x5280_0041,    // mov w1, #2       (WaitIfEqual)
        0x2a1f_03e2,    // mov w2, wzr      (and it is)
        0xd298_6a03,    // movz x3, #50000  (nanoseconds)
        0xd400_0681,    // svc #0x34        (WaitForAddress -> parks until then)
        0xd292_0009,    // mov x9, #0x9000
        0x5280_0aa1,    // mov w1, #0x55
        0xb900_0121,    // str w1, [x9]
        0xd400_00e1,    // svc #7           (ExitProcess)
    ];
    // A yield is `SleepThread(0)`: Horizon spends the non-positive timeouts on
    // yield modes rather than durations, so this hands the CPU on and comes
    // straight back for it.
    let child = [
        0xaa1f_03e0u32, // mov x0, xzr
        0xd400_0161,    // svc #0xb
        0x17ff_fffe,    // b .-8
    ];
    let bytes = |code: &[u32]| -> Vec<u8> { code.iter().flat_map(|i| i.to_le_bytes()).collect() };
    cpu.mem.map_zero(0x1000, 0x100).unwrap();
    cpu.mem.map(0x1000, &bytes(&main)).unwrap();
    cpu.mem.map_zero(0x2000, 0x100).unwrap();
    cpu.mem.map(0x2000, &bytes(&child)).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(2_000_000).unwrap();

    assert!(cpu.halted, "the parked parent never woke");
    assert_eq!(cpu.mem.read_u32(0x9000).unwrap(), 0x55);
    // And it woke *because the time came*, not because something gave up on
    // it: 50 us is 51,000 cycles of the 1.02 GHz clock.
    assert!(
        cpu.cycles >= 51_000,
        "woke after only {} cycles",
        cpu.cycles
    );
}

#[test]
fn a_thread_polling_an_idle_socket_does_not_starve_the_others() {
    // NXpotify's Zeroconf listener is `if (poll(&pfd, 1, 200) <= 0) continue;`
    // around an idle socket. Nothing here will ever be ready, so the answer is
    // always zero — but a poll that was given a timeout is a *wait*, and
    // returning it instantly left that thread looping with no blocking syscall
    // in it. Threads only hand over at those, so the loop owned the CPU
    // forever and the main thread never drew another frame.
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.register_service_handle(0x30, "bsd:u");
    cpu.mem.map_zero(0x4000, 0x2000).unwrap(); // the child's stack
    cpu.mem.map_zero(0x6000, 0x1000).unwrap(); // the flag main sets

    // main: start the poller, sleep once to hand it the CPU, and then — only
    // if it hands the CPU back — set the flag and exit.
    let main = [
        0xd284_0001u32, // mov x1, #0x2000  (entry)
        0xd280_0002,    // mov x2, #0       (arg)
        0xd28a_0003,    // mov x3, #0x5000  (stack top)
        0x5280_0764,    // mov w4, #0x3b    (priority)
        0x1280_0005,    // mov w5, #-1      (core)
        0xd400_0101,    // svc #8           (CreateThread → handle in x1)
        0xaa01_03e0,    // mov x0, x1
        0xd400_0121,    // svc #9           (StartThread)
        0xd400_0161,    // svc #0xb         (SleepThread → over to the poller)
        0xd28c_0009,    // mov x9, #0x6000
        0x5280_0aa1,    // mov w1, #0x55
        0xb900_0121,    // str w1, [x9]
        0xd400_00e1,    // svc #7           (ExitProcess)
    ];
    // The poller: build a `Poll(nfds = 1, timeout = 200)` CMIF request in its
    // own TLS block and send it, forever. Threads get their TLS at
    // THREAD_TLS_BASE + index * stride, and this is thread 1.
    let child = [
        0xd282_0009u32,     // mov x9, #0x1000
        movk_x9_tls_high(), // (= THREAD_TLS_BASE + stride)
        0x5280_0081,        // mov w1, #4                  (message type: Request)
        0xb900_0121,        // str w1, [x9]
        0x5280_0101,        // mov w1, #8                  (data words)
        0xb900_0521,        // str w1, [x9, #4]
        0x5288_ca61,        // mov w1, #0x4653             ("SFCI")
        0x72a9_2861,        // movk w1, #0x4943, lsl #16
        0xb900_1121,        // str w1, [x9, #0x10]
        0x5280_00c1,        // mov w1, #6                  (command: Poll)
        0xb900_1921,        // str w1, [x9, #0x18]
        0x5280_0021,        // mov w1, #1                  (nfds)
        0xb900_2121,        // str w1, [x9, #0x20]
        0x5280_1901,        // mov w1, #200                (timeout, ms)
        0xb900_2521,        // str w1, [x9, #0x24]
        0xd280_0600,        // mov x0, #0x30               (the bsd:u handle)
        0xd400_0421,        // svc #0x21                   (SendSyncRequest)
        0x17ff_ffef,        // b -0x44                     (round again)
    ];
    let bytes = |code: &[u32]| -> Vec<u8> { code.iter().flat_map(|i| i.to_le_bytes()).collect() };
    cpu.mem.map_zero(0x1000, 0x100).unwrap();
    cpu.mem.map(0x1000, &bytes(&main)).unwrap();
    cpu.mem.map_zero(0x2000, 0x100).unwrap();
    cpu.mem.map(0x2000, &bytes(&child)).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(10_000).unwrap();

    assert!(
        cpu.halted,
        "main never got the CPU back from the polling thread"
    );
    assert_eq!(cpu.mem.read_u32(0x6000).unwrap(), 0x55);
}

#[test]
fn an_audio_thread_appending_buffers_does_not_starve_the_others() {
    // `audout` releases every buffer the moment it is appended -- a device
    // that never falls behind -- so the guest's mixer never has to wait on the
    // buffer event it registered. Threads here only hand over at a blocking
    // syscall, and a mixer with nothing to wait for never reaches one: in "A
    // Short Hike" the FMOD mixer thread took the CPU and kept it, and the main
    // thread sat `Runnable` and unscheduled for a billion instructions while
    // the game drew nothing. Appending is a round trip into the audio process
    // on hardware, so it is a place the caller gives the CPU up.
    const HANDLE_SLOT: u32 = 0x6100;
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.register_service_handle(0x30, "audout:u");
    cpu.mem.map_zero(0x4000, 0x2000).unwrap(); // the child's stack
    cpu.mem.map_zero(0x6000, 0x1000).unwrap(); // the flag main sets, and the handle

    // OpenAudioOut(48 kHz, stereo) -> an IAudioOut as a move handle, which is
    // the session the mixer below appends to.
    let mut args = Vec::new();
    args.extend_from_slice(&48_000u32.to_le_bytes());
    args.extend_from_slice(&2u32.to_le_bytes());
    args.extend_from_slice(&0u64.to_le_bytes()); // aruid
    ipc_request_plain(&mut cpu, 0x30, 1, &args);
    let device = u64::from(cpu.mem.read_u32(cpu.tls_base() + 0x0c).unwrap());
    assert_ne!(device, 0, "no IAudioOut came back");
    cpu.mem.write_u64(HANDLE_SLOT, device).unwrap();

    // main: start the mixer, sleep once to hand it the CPU, and then -- only
    // if it hands the CPU back -- set the flag and exit.
    let main = [
        0xd284_0001u32, // mov x1, #0x2000  (entry)
        0xd280_0002,    // mov x2, #0       (arg)
        0xd28a_0003,    // mov x3, #0x5000  (stack top)
        0x5280_0764,    // mov w4, #0x3b    (priority)
        0x1280_0005,    // mov w5, #-1      (core)
        0xd400_0101,    // svc #8           (CreateThread -> handle in x1)
        0xaa01_03e0,    // mov x0, x1
        0xd400_0121,    // svc #9           (StartThread)
        0xd400_0161,    // svc #0xb         (SleepThread -> over to the mixer)
        0xd28c_0009,    // mov x9, #0x6000
        0x5280_0aa1,    // mov w1, #0x55
        0xb900_0121,    // str w1, [x9]
        0xd400_00e1,    // svc #7           (ExitProcess)
    ];
    // The mixer: build an `AppendAudioOutBuffer` CMIF request in its own TLS
    // block and send it, forever -- no wait, no sleep, nothing that blocks.
    // Threads get their TLS at THREAD_TLS_BASE + index * stride; this is
    // thread 1. The request carries no buffer descriptor, so no samples move;
    // what is under test is who holds the CPU afterwards.
    let child = [
        0xd282_0009u32,     // mov x9, #0x1000
        movk_x9_tls_high(), // (= THREAD_TLS_BASE + stride)
        0x5280_0081,        // mov w1, #4                  (message type: Request)
        0xb900_0121,        // str w1, [x9]
        0x5280_0101,        // mov w1, #8                  (data words)
        0xb900_0521,        // str w1, [x9, #4]
        0x5288_ca61,        // mov w1, #0x4653             ("SFCI")
        0x72a9_2861,        // movk w1, #0x4943, lsl #16
        0xb900_1121,        // str w1, [x9, #0x10]
        0x5280_0061,        // mov w1, #3                  (AppendAudioOutBuffer)
        0xb900_1921,        // str w1, [x9, #0x18]
        0xd28c_200a,        // mov x10, #0x6100
        0xf940_0140,        // ldr x0, [x10]               (the IAudioOut handle)
        0xd400_0421,        // svc #0x21                   (SendSyncRequest)
        0x17ff_fff4,        // b -0x30                     (round again)
    ];
    let bytes = |code: &[u32]| -> Vec<u8> { code.iter().flat_map(|i| i.to_le_bytes()).collect() };
    cpu.mem.map_zero(0x1000, 0x100).unwrap();
    cpu.mem.map(0x1000, &bytes(&main)).unwrap();
    cpu.mem.map_zero(0x2000, 0x100).unwrap();
    cpu.mem.map(0x2000, &bytes(&child)).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(10_000).unwrap();

    assert!(
        cpu.halted,
        "main never got the CPU back from the mixing thread"
    );
    assert_eq!(cpu.mem.read_u32(0x6000).unwrap(), 0x55);
}

#[test]
fn set_thread_activity_takes_a_thread_out_of_the_rotation() {
    // `svcSetThreadActivity` is `nn::os::SuspendThread`/`ResumeThread`. A
    // suspended thread keeps whatever it was doing and simply stops being
    // scheduled; Horizon refuses to suspend the caller, and reports a thread
    // already in the requested state rather than treating the call as a no-op.
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.mem.map_zero(0x4000, 0x2000).unwrap(); // the child's stack
    cpu.mem.map_zero(0x6000, 0x1000).unwrap(); // the flag it sets

    // main: create and start a child, suspend it, yield a few times, then
    // record whether it ever ran, resume it, yield again, and exit.
    let main = [
        0xd284_0001u32, // mov x1, #0x2000  (entry)
        0xd280_0002,    // mov x2, #0        (arg)
        0xd28a_0003,    // mov x3, #0x5000   (stack top)
        0xd400_0101,    // svc #8            (CreateThread -> handle in x1)
        0xaa01_03ea,    // mov x10, x1       (keep the handle)
        0xaa01_03e0,    // mov x0, x1
        0xd400_0121,    // svc #9            (StartThread)
        0xaa0a_03e0,    // mov x0, x10
        0xd280_0021,    // mov x1, #1        (Paused)
        0xd400_0641,    // svc #0x32         (SetThreadActivity)
        0xd400_0161,    // svc #0xb          (SleepThread -> yields)
        0xd400_0161,    // svc #0xb
        0xd28c_0009,    // mov x9, #0x6000
        0xb940_0122,    // ldr w2, [x9]
        0xb900_0522,    // str w2, [x9, #4]  (what it saw while suspended)
        0xaa0a_03e0,    // mov x0, x10
        0xd280_0001,    // mov x1, #0        (Runnable)
        0xd400_0641,    // svc #0x32         (SetThreadActivity)
        0xd400_0161,    // svc #0xb
        0xd400_00e1,    // svc #7            (ExitProcess)
    ];
    let child = [
        0xd28c_0009u32, // mov x9, #0x6000
        0x5280_0aa1,    // mov w1, #0x55
        0xb900_0121,    // str w1, [x9]
        0xd400_0141,    // svc #0xa          (ExitThread)
    ];
    let bytes = |code: &[u32]| -> Vec<u8> { code.iter().flat_map(|i| i.to_le_bytes()).collect() };
    cpu.mem.map_zero(0x1000, 0x100).unwrap();
    cpu.mem.map(0x1000, &bytes(&main)).unwrap();
    cpu.mem.map_zero(0x2000, 0x100).unwrap();
    cpu.mem.map(0x2000, &bytes(&child)).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(10_000).unwrap();

    assert!(cpu.halted, "main should reach ExitProcess");
    assert_eq!(
        cpu.mem.read_u32(0x6004).unwrap(),
        0,
        "a suspended thread must not run"
    );
    assert_eq!(
        cpu.mem.read_u32(0x6000).unwrap(),
        0x55,
        "and must run once resumed"
    );
}

#[test]
fn arbitrate_lock_hands_the_mutex_to_a_waiter() {
    // Horizon keeps the lock word in guest memory: it holds the owner's handle,
    // plus bit30 when someone is queued. `svcArbitrateUnlock` has to move
    // ownership, or libnx's mutexLock re-reads the word and spins forever.
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.mem.map_zero(0x4000, 0x2000).unwrap();
    cpu.mem.map_zero(0x6000, 0x1000).unwrap();
    const MUTEX: u32 = 0x6100;

    // main: create + start a thread, then hold the mutex and unlock it. The
    // child blocks in ArbitrateLock and must come out owning the word.
    let main = [
        0xd284_0001u32, // mov x1, #0x2000
        0xd280_0002,    // mov x2, #0
        0xd28a_0003,    // mov x3, #0x5000
        0x5280_0764,    // mov w4, #0x3b
        0x1280_0005,    // mov w5, #-1
        0xd400_0101,    // svc #8
        0xaa01_03e0,    // mov x0, x1
        0xd400_0121,    // svc #9
        0xd400_0161,    // svc #0xb   (yield: the child blocks on the mutex)
        0xd28c_2009,    // mov x9, #0x6100
        0xaa0903e0u32,  // mov x0, x9  (ArbitrateUnlock takes the address in x0)
        0xd400_0361,    // svc #0x1b  (ArbitrateUnlock → hand it to the child)
        0xd400_0161,    // svc #0xb   (yield so the child can finish)
        0xd400_00e1,    // svc #7
    ];
    // child: ask the kernel to arbitrate a mutex main owns, then record the
    // word it ended up with.
    let child = [
        0xd28c_2009u32, // mov x9, #0x6100
        0xb940_0120,    // ldr w0, [x9]     (current owner)
        0xaa09_03e1,    // mov x1, x9       (the mutex address)
        0xd280_0022,    // mov x2, #1       (our handle, unused by the stub)
        0xd400_0341,    // svc #0x1a        (ArbitrateLock → blocks)
        0xb940_0122,    // ldr w2, [x9]
        0xd28c_0009,    // mov x9, #0x6000
        0xb900_0122,    // str w2, [x9]
        0xd400_0141,    // svc #0xa
    ];
    let bytes = |code: &[u32]| -> Vec<u8> { code.iter().flat_map(|i| i.to_le_bytes()).collect() };
    cpu.mem.map_zero(0x1000, 0x100).unwrap();
    cpu.mem.map(0x1000, &bytes(&main)).unwrap();
    cpu.mem.map_zero(0x2000, 0x100).unwrap();
    cpu.mem.map(0x2000, &bytes(&child)).unwrap();
    // main "owns" the mutex to begin with (handle 1 = the main thread).
    cpu.mem.write_u32(MUTEX, 1).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(10_000).unwrap();

    assert!(cpu.halted);
    // The child saw itself as the owner after the unlock handed it over.
    let observed = cpu.mem.read_u32(0x6000).unwrap();
    assert_ne!(observed, 0, "the child ran after the unlock");
    assert_ne!(observed, 1, "and the word no longer names the main thread");
}

#[test]
fn a_timed_out_condvar_wait_comes_back_holding_its_mutex() {
    // `svcWaitProcessWideKeyAtomic` releases the mutex on the way in and the
    // kernel re-acquires it on the way out — for a timeout exactly as for a
    // signal. Waking the waiter without doing that leaves it running outside a
    // lock it believes it holds, and `nn::os::UnlockMutex` checks: it compares
    // the word against its own thread tag and aborts on the mismatch. The Mii
    // editor's boot ended there, one millisecond after a 1 ms
    // `TimedWaitConditionVariable`.
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.mem.map_zero(0x4000, 0x2000).unwrap();
    cpu.mem.map_zero(0x6000, 0x1000).unwrap();

    // main: start the child, then spin long enough for the wait to expire —
    // timed waits are checked on the scheduler's slice boundary, so time only
    // passes while some other thread is running.
    let main = [
        0xd284_0001u32, // mov x1, #0x2000
        0xd280_0002,    // mov x2, #0
        0xd28a_0003,    // mov x3, #0x5000
        0x5280_0764,    // mov w4, #0x3b
        0x1280_0005,    // mov w5, #-1
        0xd400_0101,    // svc #8      (CreateThread -> x1 = the child's handle)
        0xd28c_0109,    // mov x9, #0x6008
        0xb900_0121,    // str w1, [x9]  (record it for the assertion)
        0xaa01_03e0,    // mov x0, x1
        0xd400_0121,    // svc #9      (StartThread)
        0xd293_880a,    // mov x10, #40000
        0xf100_054a,    // subs x10, x10, #1
        0xb5ff_ffea,    // cbnz x10, -4
        0xd400_00e1,    // svc #7
    ];
    // child: wait on a condition variable nobody ever signals, with a short
    // timeout. What the mutex word held before does not matter — the wait
    // releases it on the way in — so the only question the test asks is what
    // it holds on the way out.
    let child = [
        0xd28c_2009u32, // mov x9, #0x6100   (the mutex)
        0xaa09_03e0,    // mov x0, x9
        0xd28c_4001,    // mov x1, #0x6200   (the condition variable)
        0xd280_0042,    // mov x2, #2        (self tag; the stub reads the real one)
        0xd282_7103,    // mov x3, #5000     (nanoseconds)
        0xd400_0381,    // svc #0x1c         (WaitProcessWideKeyAtomic)
        0xb940_0122,    // ldr w2, [x9]
        0xd28c_0009,    // mov x9, #0x6000
        0xb900_0122,    // str w2, [x9]
        0xd400_0141,    // svc #0xa
    ];
    let bytes = |code: &[u32]| -> Vec<u8> { code.iter().flat_map(|i| i.to_le_bytes()).collect() };
    cpu.mem.map_zero(0x1000, 0x100).unwrap();
    cpu.mem.map(0x1000, &bytes(&main)).unwrap();
    cpu.mem.map_zero(0x2000, 0x100).unwrap();
    cpu.mem.map(0x2000, &bytes(&child)).unwrap();
    cpu.set_pc(0x1000);
    cpu.run(500_000).unwrap();

    let observed = cpu.mem.read_u32(0x6000).unwrap();
    let child_handle = cpu.mem.read_u32(0x6008).unwrap();
    assert_ne!(child_handle, 0, "the child was created");
    assert_ne!(observed, 0, "the wait came back to a mutex owned by nobody");
    assert_eq!(
        observed & !0x4000_0000,
        child_handle,
        "and it is the waiter's own"
    );
}

#[test]
// Every assertion here relates one constant to another, which is the point:
// the reason each bound exists is in the comment beside it, and a reader
// looking for the memory map finds the whole of it in one place.
#[allow(clippy::assertions_on_constants)]
fn the_guest_regions_are_disjoint_and_big_enough_for_what_they_promise() {
    // Every region below is carved out of one 4 GiB space, and the guest is
    // told where each one is and how big. Two that overlap are two subsystems
    // writing over each other, and it is not hypothetical: `svcSetHeapSize`
    // granted the 480 MiB `nn::init` asked for inside a 240 MiB heap region,
    // so a retail title's heap ran over the framebuffer and 224 MiB into the
    // alias region. Nothing had claimed those addresses yet, which is the only
    // reason it went unnoticed.
    use switch_core::cpu::{
        MemoryLayout, OperationMode, GUEST_ALIAS_REGION_ADDR, GUEST_ALIAS_REGION_SIZE,
        GUEST_HEAP_REGION_ADDR, GUEST_HEAP_REGION_SIZE, GUEST_SPACE_END, GUEST_STACK_REGION_ADDR,
        GUEST_STACK_REGION_SIZE, GUEST_TOTAL_MEMORY_SIZE, MAIN_THREAD_TLS_BASE,
        SELF_RETURN_TRAMPOLINE, SHARED_BUFFER_ADDR, SHARED_BUFFER_RESERVED_SIZE, STACK_TOP,
        THREAD_EXIT_TRAMPOLINE, THREAD_TLS_BASE, VAMM_ARENA_SIZE,
    };
    use switch_core::{FB_BASE, FB_HEIGHT, FB_WIDTH, INPUT_ADDR};

    assert!(GUEST_STACK_REGION_ADDR + GUEST_STACK_REGION_SIZE <= GUEST_HEAP_REGION_ADDR);
    // And clear of what the emulator keeps for itself. A guest picks the
    // address it maps a thread stack at out of this region and asks nobody:
    // whatever of ours is inside it gets overwritten sooner or later, and the
    // trampolines and the TLS blocks are the two things a thread cannot lose.
    for (what, addr) in [
        ("the self-return trampoline", SELF_RETURN_TRAMPOLINE),
        ("the thread-exit trampoline", THREAD_EXIT_TRAMPOLINE),
        ("the main thread's TLS", MAIN_THREAD_TLS_BASE),
        ("the child threads' TLS", THREAD_TLS_BASE),
    ] {
        assert!(
            !(GUEST_STACK_REGION_ADDR..GUEST_STACK_REGION_ADDR + GUEST_STACK_REGION_SIZE)
                .contains(&addr),
            "{what} ({addr:#x}) is inside the stack region the guest is told is free"
        );
    }
    assert!(
        STACK_TOP <= u64::from(GUEST_HEAP_REGION_ADDR),
        "the main stack is below the heap"
    );
    assert_eq!(
        GUEST_HEAP_REGION_ADDR + GUEST_HEAP_REGION_SIZE,
        GUEST_ALIAS_REGION_ADDR
    );
    assert!(GUEST_ALIAS_REGION_ADDR + GUEST_ALIAS_REGION_SIZE <= SHARED_BUFFER_ADDR);
    // The shared buffer is reserved for the *docked* geometry however the
    // console starts. The pool laid out in it is the shared layer's own and
    // does not follow the dock — the Home Menu stays 720p docked because
    // qlaunch lays out at 720p, not because of where the buffer ends — so
    // this is headroom for an applet that does honour the layout it is given,
    // and what it has to cover is whichever geometry is the larger.
    assert_eq!(
        SHARED_BUFFER_RESERVED_SIZE,
        OperationMode::Docked.shared_buffer_size(),
        "the reservation has to cover the larger of the two modes"
    );
    assert!(
        switch_core::cpu::SHARED_BUFFER_GEOMETRY.shared_buffer_size()
            <= SHARED_BUFFER_RESERVED_SIZE,
        "and the one actually laid out in it"
    );
    assert!(SHARED_BUFFER_ADDR + SHARED_BUFFER_RESERVED_SIZE <= FB_BASE);
    assert!(FB_BASE + FB_WIDTH * FB_HEIGHT * 4 <= INPUT_ADDR);
    assert!(INPUT_ADDR + 0x1000 <= GUEST_SPACE_END);

    // `nn::init` asks for the whole of what `svcGetInfo` calls total memory,
    // so the region it grows into may not be smaller than that figure. Which
    // region that is follows from the layout rather than from the title:
    // without virtual address memory it is `svcSetHeapSize` and the heap
    // region, every time, and the alias region is address space no title on
    // this layout ever asks for, so the rest is charged to the heap.
    assert!(GUEST_TOTAL_MEMORY_SIZE <= GUEST_HEAP_REGION_SIZE);

    // And the emulator has to be able to *back* what it advertises. A title
    // sizes its pools from `TotalMemorySize` and then touches them: with the
    // cap at 512 MiB against this figure, one reserved 1.5 GiB and faulted
    // part way through zeroing it, inside a `stp` loop that names no
    // allocation and no service. Whichever of the two moves, they move
    // together.
    assert!(
        u64::from(GUEST_TOTAL_MEMORY_SIZE) <= switch_core::mem::MAX_MAPPED_BYTES,
        "advertising {GUEST_TOTAL_MEMORY_SIZE:#x} of memory that cannot be backed"
    );
    // The alias region has to be a region a guest can read, and not a byte
    // more: `libnx` reads it at startup and never maps into it — hbmenu,
    // JKSV, Checkpoint, the appstore and NX-Shell issue `svcGetInfo` 2/3 and
    // zero `svcMapPhysicalMemory` — and `nnSdk` without virtual address
    // memory does not issue that syscall either. Persona 5 Royal's pools want
    // every byte of what the floor used to hold back.
    assert!(
        GUEST_ALIAS_REGION_SIZE >= 0x0100_0000,
        "the alias region is still a region"
    );

    // What actually binds the alias region is virtual address memory, not the
    // heap. `VammManager` claims `VAMM_ARENA_SIZE` at the region base before
    // the title reserves a byte of its own, and the heap `nn::init` asks for
    // — total minus the system resource — has to fit above it, as do the
    // reservations the title then makes for itself. Sizing the alias region
    // to the heap alone is what made `nn::os::AllocateAddressRegion` fail
    // with os result 3-12, and it fails as an abort inside the title rather
    // than as anything that names the layout, so it is worth an assert here.
    for layout in [MemoryLayout::PLAIN, MemoryLayout::VIRTUAL_ADDRESS] {
        assert_eq!(layout.heap_addr + layout.heap_size, layout.alias_addr);
        assert!(layout.alias_addr + layout.alias_size <= SHARED_BUFFER_ADDR);
        assert!(STACK_TOP <= u64::from(layout.heap_addr));
        assert!(layout.system_resource < layout.total_memory);
    }
    // The total has to fit the region the title actually grows into — and
    // *which* region that is, is the whole difference between the two
    // layouts. A plain title grows the heap region with `svcSetHeapSize`; one
    // on virtual address memory reserves out of the alias region and never
    // issues that syscall at all. Requiring the heap region of both is what
    // charged the VAMM layout 896 MiB of address space nothing ever grows
    // into, and took it from the region that does.
    assert!(MemoryLayout::PLAIN.total_memory <= MemoryLayout::PLAIN.heap_size);
    assert!(MemoryLayout::VIRTUAL_ADDRESS.total_memory <= MemoryLayout::VIRTUAL_ADDRESS.alias_size);
    assert_eq!(
        MemoryLayout::PLAIN.system_resource,
        0,
        "zero is what keeps a title off VAMM"
    );
    assert_ne!(MemoryLayout::VIRTUAL_ADDRESS.system_resource, 0);
    // A title's own manifest picks between them, and 0 has to mean the plain
    // heap: Just Dance 2019 declares 0 and is broken by the smaller total the
    // other layout reports, without ever touching the manager.
    assert_eq!(MemoryLayout::for_system_resource(0), MemoryLayout::PLAIN);
    assert_eq!(
        MemoryLayout::for_system_resource(0x0100_0000),
        MemoryLayout::VIRTUAL_ADDRESS
    );

    let vamm = MemoryLayout::VIRTUAL_ADDRESS;
    let heap_reservation = vamm.total_memory - vamm.system_resource;
    assert!(
        VAMM_ARENA_SIZE + heap_reservation < vamm.alias_size,
        "the alias region must hold the SDK's arena and a full heap reservation"
    );
    // Just Dance 2023 asks for five more regions after its heap, the largest
    // 0x207f000; running out on those aborts exactly as running out on the
    // heap does, so the headroom is part of the layout rather than slack.
    //
    // 274 MiB of it is measurably not enough. That is what this layout used
    // to leave, and the title's own block allocator spent all of it in
    // ~20 MiB segments, ran its last one up to 0xEFF0_0000, and was refused
    // the next 4.2 MiB. Nothing checked the null — the dlmalloc behind it
    // built a 4 MiB arena at address **0** and ran on it until a `Reallocate`
    // dereferenced a pointer no arena claimed. Nothing here can prove any
    // figure is *enough*; the floor is here so the region is not whittled
    // back to where a real title is known to fail.
    let headroom = vamm.alias_size - VAMM_ARENA_SIZE - heap_reservation;
    assert!(
        headroom >= 0x2000_0000,
        "leave a title room to reserve on its own account, got {headroom:#x}"
    );
}

#[test]
fn a_heap_bigger_than_its_region_is_refused() {
    // SetHeapSize used to say yes to any size at all and hand back the region
    // base regardless. A guest that is granted more than the region holds has
    // no way to find out, and writes past the end of it into whatever is next.
    use switch_core::cpu::{GUEST_HEAP_REGION_ADDR, GUEST_HEAP_REGION_SIZE};
    const OUT_OF_MEMORY: u64 = 1 | (104 << 9);

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.mem.map(0x1000, &svc(0x01).to_le_bytes()).unwrap();
    cpu.set_reg(1, u64::from(GUEST_HEAP_REGION_SIZE));
    cpu.run(1).unwrap();
    assert_eq!(
        cpu.read_x(0),
        0,
        "a heap that exactly fills its region is granted"
    );
    assert_eq!(cpu.read_x(1), u64::from(GUEST_HEAP_REGION_ADDR));

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.mem.map(0x1000, &svc(0x01).to_le_bytes()).unwrap();
    cpu.set_reg(1, u64::from(GUEST_HEAP_REGION_SIZE) + 0x1000);
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), OUT_OF_MEMORY);
}
