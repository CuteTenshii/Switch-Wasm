//! Kernel tests: syscalls, scheduler, threads and the address space.

mod cpu;

use cpu::*;

#[test]
fn bootstrap_provides_stack_and_low_memory() {
    let mut cpu = Cpu::new();
    assert_eq!(cpu.sp(), 0);
    cpu.bootstrap();

    // SP is the top of the mapped stack.
    assert_eq!(cpu.sp(), switch_core::cpu::STACK_TOP);
    cpu.mem
        .write_u64((cpu.sp() - 8) as u32, 0x1234_5678)
        .unwrap();
    assert_eq!(
        cpu.mem.read_u64((cpu.sp() - 8) as u32).unwrap(),
        0x1234_5678
    );

    // Untouched low memory reads as zero instead of faulting.
    assert_eq!(cpu.mem.read_u32(0x244498).unwrap(), 0);
    // Writes allocate a private page on first touch.
    cpu.mem.write_u32(0xb00, 0xDEAD_BEEF).unwrap();
    assert_eq!(cpu.mem.read_u32(0xb00).unwrap(), 0xDEAD_BEEF);
    // Reads past the end of the guest address space still fault.
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

    // GetSystemTick runs at 19.2 MHz against a 1.02 GHz CPU, about 1/53 tick per instruction.
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
    // QueryMemory writes MemoryInfo for the run of same-state pages; page info goes in X1.
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

    // GetInfo returns the value in X1. InfoType 4 = HeapRegionAddress; regions must fit in a `u32`.
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

    // InfoType 21/22 = Total/UsedNonSystemMemorySize; `nnSdk` sizes its heap from the difference.
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

    // InfoType 16 = SystemResourceSizeTotal. Non-zero puts `nnSdk` on VAMM, so default to 0.
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

    // A title declaring a system resource gets a larger alias region and a smaller total.
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
    // MapPhysicalMemory(address, size) grows the heap of a title on the 39-bit address space.
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.mem.map(0x1000, &svc(0x2c).to_le_bytes()).unwrap();
    cpu.set_reg(0, u64::from(GUEST_ALIAS_REGION_ADDR));
    cpu.set_reg(1, 0x10_0000);
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);

    // Unaligned, empty, or not `u32`-addressable ranges are rejected.
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
        assert_eq!(
            cpu.read_x(0),
            0x8000_DC01,
            "{addr:#x}+{size:#x} should be rejected as InvalidMemoryRange"
        );
    }
}

#[test]
fn map_memory_backs_the_destination_and_unmap_frees_it() {
    // MapMemory(dst, src, size) mirrors src at dst; UnmapMemory frees it.
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
    // QueryMemory writes a 40-byte MemoryInfo, not 8 x u64.
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

    // An untouched soft-mapped page reports as unmapped (type 0, no perm).
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
    assert_eq!(cpu.mem.read_u64(0x3028).unwrap(), 0);
    assert_eq!(cpu.mem.read_u64(0x3040).unwrap(), 0);
    assert_eq!(cpu.read_x(1), 0); // unmapped soft page -> page info 0
}

#[test]
fn query_memory_gives_the_execute_bit_only_to_module_text() {
    // `.text` reports R-X and the pages after it RW-, as separate regions.
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
    // A guest program that creates a thread, starts it, and waits for it to set a flag.
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
        0x1280_0025,    // mov w5, #-2      (core: the process default)
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
    // WaitForAddress/SignalToAddress report InvalidState when the guest word does not match.
    const RESULT_INVALID_STATE: u64 = 1 | (125 << 9);
    const RESULT_TIMED_OUT: u64 = 0xEA01;

    // DecrementAndWaitIfLessThan with a zero timeout decrements but does not block.
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

    // WaitIfEqual against a different value: no wait, word unchanged.
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
    // A blocking WaitForAddress must write its result to X0 before switching threads.
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
        0xd282_4682,    // mov x2, #0x1234  (arg, what must survive)
        0xd28a_0003,    // mov x3, #0x5000  (stack top)
        0x5280_0764,    // mov w4, #0x3b    (priority)
        0x1280_0025,    // mov w5, #-2      (core: the process default)
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
    // A wait on an empty handle set is never answered: the thread parks and others run.
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.mem.map_zero(0x4000, 0x2000).unwrap(); // the child's stack
    cpu.mem.map_zero(0x6000, 0x1000).unwrap();

    let main = [
        0xd284_0001u32, // mov x1, #0x2000  (entry)
        0xaa1f_03e2,    // mov x2, xzr      (arg)
        0xd28a_0003,    // mov x3, #0x5000  (stack top)
        0x5280_0764,    // mov w4, #0x3b
        0x1280_0025,    // mov w5, #-2      (core: the process default)
        0xd400_0101,    // svc #8           (CreateThread)
        0xaa01_03e0,    // mov x0, x1
        0xd400_0121,    // svc #9           (StartThread)
        0xd400_0161,    // svc #0xb         (SleepThread -> hands over)
        0xd28c_0009,    // mov x9, #0x6000
        0x5280_0aa1,    // mov w1, #0x55
        0xb900_0121,    // str w1, [x9]
        0xd400_00e1,    // svc #7           (ExitProcess)
    ];
    // The child waits on an empty handle set and must never get past it.
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
    // An unsatisfiable wait parks the thread instead of re-polling every slice.
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
    // A thread that never makes a syscall is still preempted.
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.mem.map_zero(0x4000, 0x2000).unwrap(); // the child's stack
    cpu.mem.map_zero(0x6000, 0x1000).unwrap(); // the flag main sets at the end

    let main = [
        0xd284_0001u32, // mov x1, #0x2000  (entry)
        0xaa1f_03e2,    // mov x2, xzr      (arg)
        0xd28a_0003,    // mov x3, #0x5000  (stack top)
        0x5280_0764,    // mov w4, #0x3b    (priority)
        0x1280_0025,    // mov w5, #-2      (core: the process default)
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
    // Timed waits expire even when other threads switch more often than every `TIME_SLICE`.
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.mem.map_zero(0x4000, 0x4000).unwrap(); // the two children's stacks
    cpu.mem.map_zero(0x9000, 0x1000).unwrap(); // the flag the parent sets

    let main = [
        0xd284_0001u32, // mov x1, #0x2000  (entry)
        0xaa1f_03e2,    // mov x2, xzr      (arg)
        0xd28c_0003,    // mov x3, #0x6000  (stack top)
        0x5280_0764,    // mov w4, #0x3b    (priority)
        0x1280_0025,    // mov w5, #-2      (core: the process default)
        0xd400_0101,    // svc #8           (CreateThread -> handle in x1)
        0xaa01_03e0,    // mov x0, x1
        0xd400_0121,    // svc #9           (StartThread)
        0xd284_0001,    // mov x1, #0x2000
        0xaa1f_03e2,    // mov x2, xzr
        0xd290_0003,    // mov x3, #0x8000  (the second child's stack top)
        0x5280_0764,    // mov w4, #0x3b
        0x1280_0025,    // mov w5, #-2      (core: the process default)
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
    // A yield is `SleepThread(0)`.
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
    // It woke because the deadline passed: 50 us is 51,000 cycles at 1.02 GHz.
    assert!(
        cpu.cycles >= 51_000,
        "woke after only {} cycles",
        cpu.cycles
    );
}

#[test]
fn a_thread_polling_an_idle_socket_does_not_starve_the_others() {
    // A poll with a timeout blocks, so the polling thread hands the CPU on.
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.register_service_handle(0x30, "bsd:u");
    cpu.mem.map_zero(0x4000, 0x2000).unwrap(); // the child's stack
    cpu.mem.map_zero(0x6000, 0x1000).unwrap(); // the flag main sets

    // main: start the poller, sleep once, then set the flag and exit if the CPU comes back.
    let main = [
        0xd284_0001u32, // mov x1, #0x2000  (entry)
        0xd280_0002,    // mov x2, #0       (arg)
        0xd28a_0003,    // mov x3, #0x5000  (stack top)
        0x5280_0764,    // mov w4, #0x3b    (priority)
        0x1280_0025,    // mov w5, #-2      (core: the process default)
        0xd400_0101,    // svc #8           (CreateThread → handle in x1)
        0xaa01_03e0,    // mov x0, x1
        0xd400_0121,    // svc #9           (StartThread)
        0xd400_0161,    // svc #0xb         (SleepThread → over to the poller)
        0xd28c_0009,    // mov x9, #0x6000
        0x5280_0aa1,    // mov w1, #0x55
        0xb900_0121,    // str w1, [x9]
        0xd400_00e1,    // svc #7           (ExitProcess)
    ];
    // The poller (thread 1) sends `Poll(nfds = 1, timeout = 200)` from its TLS forever.
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
    // AppendAudioOutBuffer yields the CPU, so a mixer that never waits cannot starve other threads.
    const HANDLE_SLOT: u32 = 0x6100;
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.register_service_handle(0x30, "audout:u");
    cpu.mem.map_zero(0x4000, 0x2000).unwrap(); // the child's stack
    cpu.mem.map_zero(0x6000, 0x1000).unwrap(); // the flag main sets, and the handle

    // OpenAudioOut(48 kHz, stereo) -> an IAudioOut move handle.
    let mut args = Vec::new();
    args.extend_from_slice(&48_000u32.to_le_bytes());
    args.extend_from_slice(&2u32.to_le_bytes());
    args.extend_from_slice(&0u64.to_le_bytes()); // aruid
    ipc_request_plain(&mut cpu, 0x30, 1, &args);
    let device = u64::from(cpu.mem.read_u32(cpu.tls_base() + 0x0c).unwrap());
    assert_ne!(device, 0, "no IAudioOut came back");
    cpu.mem.write_u64(HANDLE_SLOT, device).unwrap();

    // main: start the mixer, sleep once, then set the flag and exit if the CPU comes back.
    let main = [
        0xd284_0001u32, // mov x1, #0x2000  (entry)
        0xd280_0002,    // mov x2, #0       (arg)
        0xd28a_0003,    // mov x3, #0x5000  (stack top)
        0x5280_0764,    // mov w4, #0x3b    (priority)
        0x1280_0025,    // mov w5, #-2      (core: the process default)
        0xd400_0101,    // svc #8           (CreateThread -> handle in x1)
        0xaa01_03e0,    // mov x0, x1
        0xd400_0121,    // svc #9           (StartThread)
        0xd400_0161,    // svc #0xb         (SleepThread -> over to the mixer)
        0xd28c_0009,    // mov x9, #0x6000
        0x5280_0aa1,    // mov w1, #0x55
        0xb900_0121,    // str w1, [x9]
        0xd400_00e1,    // svc #7           (ExitProcess)
    ];
    // The mixer (thread 1) sends a descriptor-less `AppendAudioOutBuffer` from its TLS forever.
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
    // SetThreadActivity refuses the caller and reports a thread already in the requested state.
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
    // The lock word is the owner's handle plus bit30 when contended; ArbitrateUnlock hands it over.
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.mem.map_zero(0x4000, 0x2000).unwrap();
    cpu.mem.map_zero(0x6000, 0x1000).unwrap();
    const MUTEX: u32 = 0x6100;

    // main: start a thread, then unlock a held mutex the child is blocked on in ArbitrateLock.
    let main = [
        0xd284_0001u32, // mov x1, #0x2000
        0xd280_0002,    // mov x2, #0
        0xd28a_0003,    // mov x3, #0x5000
        0x5280_0584,    // mov w4, #0x2c  (main's priority: a lower one would never be yielded to)
        0x1280_0025,    // mov w5, #-2      (core: the process default)
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
    // child: arbitrate the mutex main owns, then record the word.
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
    assert_eq!(
        observed,
        cpu.read_x(1) as u32,
        "the word names the child, by the handle CreateThread returned"
    );
}

#[test]
fn a_timed_out_condvar_wait_comes_back_holding_its_mutex() {
    // WaitProcessWideKeyAtomic re-acquires the mutex on timeout as well as on signal.
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.mem.map_zero(0x4000, 0x2000).unwrap();
    cpu.mem.map_zero(0x6000, 0x1000).unwrap();

    // main: start the child, then spin until the wait expires at a slice boundary.
    let main = [
        0xd284_0001u32, // mov x1, #0x2000
        0xd280_0002,    // mov x2, #0
        0xd28a_0003,    // mov x3, #0x5000
        0x5280_0584,    // mov w4, #0x2c  (main's priority: a lower one would never be yielded to)
        0x1280_0025,    // mov w5, #-2      (core: the process default)
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
    // child: wait with a short timeout on a condition variable nobody signals.
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
#[allow(clippy::assertions_on_constants)]
fn the_guest_regions_are_disjoint_and_big_enough_for_what_they_promise() {
    // All regions share one 4 GiB space and must not overlap.
    use switch_core::cpu::{
        MemoryLayout, OperationMode, GUEST_ALIAS_REGION_ADDR, GUEST_ALIAS_REGION_SIZE,
        GUEST_ASLR_REGION_ADDR, GUEST_ASLR_REGION_SIZE, GUEST_HEAP_REGION_ADDR,
        GUEST_HEAP_REGION_SIZE, GUEST_SPACE_END, GUEST_STACK_REGION_ADDR, GUEST_STACK_REGION_SIZE,
        GUEST_TOTAL_MEMORY_SIZE, MAIN_THREAD_TLS_BASE, SELF_RETURN_TRAMPOLINE, SHARED_BUFFER_ADDR,
        SHARED_BUFFER_RESERVED_SIZE, STACK_SIZE, STACK_TOP, THREAD_EXIT_TRAMPOLINE,
        THREAD_TLS_BASE, THREAD_TLS_STRIDE, VAMM_ARENA_SIZE,
    };
    use switch_core::{FB_BASE, FB_HEIGHT, FB_WIDTH, INPUT_ADDR};

    assert!(GUEST_STACK_REGION_ADDR + GUEST_STACK_REGION_SIZE <= GUEST_HEAP_REGION_ADDR);
    // Inside the ASLR region, and large enough for randomly placed thread stacks.
    assert!(GUEST_STACK_REGION_ADDR >= GUEST_ASLR_REGION_ADDR);
    assert!(
        GUEST_STACK_REGION_ADDR + GUEST_STACK_REGION_SIZE
            <= GUEST_ASLR_REGION_ADDR + GUEST_ASLR_REGION_SIZE
    );
    assert!(GUEST_STACK_REGION_SIZE > 0x0800_0000);
    // TLS blocks sit between the stack region and the main stack; the gap is the thread limit.
    const THREADS: u32 = 1024;
    assert!(
        u64::from(THREAD_TLS_BASE + THREADS * THREAD_TLS_STRIDE) <= STACK_TOP - STACK_SIZE,
        "{THREADS} threads' TLS blocks run into the main stack"
    );
    // The guest maps thread stacks anywhere in the stack region, so keep emulator state out of it.
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
    // The shared buffer is reserved for the docked geometry regardless of the starting mode.
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

    // Without VAMM, `nn::init` grows the heap region to the full total memory.
    assert!(GUEST_TOTAL_MEMORY_SIZE <= GUEST_HEAP_REGION_SIZE);

    // The advertised total must be backable.
    assert!(
        u64::from(GUEST_TOTAL_MEMORY_SIZE) <= switch_core::mem::MAX_MAPPED_BYTES,
        "advertising {GUEST_TOTAL_MEMORY_SIZE:#x} of memory that cannot be backed"
    );
    // The alias region only needs to be readable; non-VAMM titles never map into it.
    assert!(
        GUEST_ALIAS_REGION_SIZE >= 0x0100_0000,
        "the alias region is still a region"
    );

    // On VAMM the alias region holds the SDK's arena plus the heap reservation.
    for layout in [MemoryLayout::PLAIN, MemoryLayout::VIRTUAL_ADDRESS] {
        assert_eq!(layout.heap_addr + layout.heap_size, layout.alias_addr);
        assert!(layout.alias_addr + layout.alias_size <= SHARED_BUFFER_ADDR);
        assert!(STACK_TOP <= u64::from(layout.heap_addr));
        assert!(layout.system_resource < layout.total_memory);
    }
    // Each layout's total must fit the region that layout grows into.
    assert!(MemoryLayout::PLAIN.total_memory <= MemoryLayout::PLAIN.heap_size);
    assert!(MemoryLayout::VIRTUAL_ADDRESS.total_memory <= MemoryLayout::VIRTUAL_ADDRESS.alias_size);
    assert_eq!(
        MemoryLayout::PLAIN.system_resource,
        0,
        "zero is what keeps a title off VAMM"
    );
    assert_ne!(MemoryLayout::VIRTUAL_ADDRESS.system_resource, 0);
    // A system resource of 0 selects the plain layout.
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
    // Headroom for a title's own reservations; 274 MiB was not enough for Just Dance 2023.
    let headroom = vamm.alias_size - VAMM_ARENA_SIZE - heap_reservation;
    assert!(
        headroom >= 0x2000_0000,
        "leave a title room to reserve on its own account, got {headroom:#x}"
    );
}

#[test]
fn a_heap_bigger_than_its_region_is_refused() {
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
