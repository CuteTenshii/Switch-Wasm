//! `ldr:ro`: loading and unloading relocatable modules.

mod cpu;

use cpu::*;

#[test]
fn ldr_ro_initialize_is_not_a_fabricated_object() {
    // RegisterProcessHandle (cmd 4), `nn::ro::Initialize`'s first call: a bare Result.
    let (mut cpu, handle) = ldr_ro_session();
    let tls = cpu.tls_base();

    ldr_ro_request(&mut cpu, handle, 4, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x10).unwrap(), 0x4F43_4653); // "SFCO"
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0);
    // Bit 31 of the second header word marks handles; there are none.
    assert_eq!(cpu.mem.read_u32(tls + 4).unwrap() >> 31, 0);
}

#[test]
fn ldr_ro_maps_a_module_where_nothing_else_lives() {
    // LoadModule must actually map the NRO at the returned address.
    use switch_core::cpu::{RO_MODULE_REGION_ADDR, RO_MODULE_REGION_SIZE};
    let (mut cpu, handle) = ldr_ro_session();
    let tls = cpu.tls_base();

    ldr_ro_request(&mut cpu, handle, 0, &[NRO_SOURCE, 0x3000, NRO_BSS, 0x1000]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0);
    let base = cpu.mem.read_u64(tls + 0x20).unwrap();
    assert!(
        (u64::from(RO_MODULE_REGION_ADDR)
            ..u64::from(RO_MODULE_REGION_ADDR) + u64::from(RO_MODULE_REGION_SIZE))
            .contains(&base),
        "a module must land in the region set aside for one, not at {base:#x}"
    );
    let base = base as u32;

    // The three segments in file order, then a zero-filled BSS.
    assert_eq!(cpu.mem.read_u32(base).unwrap(), 0x1400_0010);
    assert_eq!(cpu.mem.read_u8(base + 0x1000).unwrap(), 0xAA);
    assert_eq!(cpu.mem.read_u8(base + 0x2000).unwrap(), 0xBB);
    assert_eq!(cpu.mem.read_u8(base + 0x3000).unwrap(), 0);

    // `.text` is read-only; `.data` is writable for relocations.
    assert!(cpu.mem.write_u32(base, 0).is_err());
    assert!(cpu.mem.write_u32(base + 0x2000, 0).is_ok());
}

#[test]
fn ldr_ro_unload_frees_the_address_space_and_the_protection() {
    // Unloading removes the pages and the `.text` read-only marking.
    let (mut cpu, handle) = ldr_ro_session();
    let tls = cpu.tls_base();

    ldr_ro_request(&mut cpu, handle, 0, &[NRO_SOURCE, 0x3000, NRO_BSS, 0x1000]);
    let base = cpu.mem.read_u64(tls + 0x20).unwrap();
    ldr_ro_request(&mut cpu, handle, 1, &[base]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0);
    assert!(cpu.mem.write_u32(base as u32, 0).is_ok());

    // The address space is reused by the next load.
    ldr_ro_request(&mut cpu, handle, 0, &[NRO_SOURCE, 0x3000, NRO_BSS, 0x1000]);
    assert_eq!(cpu.mem.read_u64(tls + 0x20).unwrap(), base);

    // Unloading something never loaded is NotLoaded.
    const NOT_LOADED: u32 = 22 | (1028 << 9);
    ldr_ro_request(&mut cpu, handle, 1, &[0x2800_0000]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), NOT_LOADED);
}

#[test]
fn ldr_ro_two_modules_do_not_overlap() {
    // The second module goes behind the first, BSS included.
    let (mut cpu, handle) = ldr_ro_session();
    let tls = cpu.tls_base();

    ldr_ro_request(&mut cpu, handle, 0, &[NRO_SOURCE, 0x3000, NRO_BSS, 0x1000]);
    let first = cpu.mem.read_u64(tls + 0x20).unwrap();
    ldr_ro_request(&mut cpu, handle, 0, &[NRO_SOURCE, 0x3000, NRO_BSS, 0x1000]);
    let second = cpu.mem.read_u64(tls + 0x20).unwrap();
    assert_eq!(
        second,
        first + 0x4000,
        "image plus BSS, and no gap to waste"
    );
}

#[test]
fn ldr_ro_refuses_what_is_not_a_module() {
    // A bad NRO or an undersized BSS is refused.
    const INVALID_NRO: u32 = 22 | (4 << 9);
    const INVALID_ADDRESS: u32 = 22 | (1025 << 9);
    const INVALID_SIZE: u32 = 22 | (1026 << 9);
    let (mut cpu, handle) = ldr_ro_session();
    let tls = cpu.tls_base();

    ldr_ro_request(&mut cpu, handle, 0, &[0x1100_0000, 0x3000, NRO_BSS, 0x1000]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), INVALID_NRO);

    ldr_ro_request(
        &mut cpu,
        handle,
        0,
        &[NRO_SOURCE + 8, 0x3000, NRO_BSS, 0x1000],
    );
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), INVALID_ADDRESS);

    ldr_ro_request(&mut cpu, handle, 0, &[NRO_SOURCE, 0x3000, NRO_BSS, 0]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), INVALID_SIZE);
}

#[test]
fn ldr_ro_module_info_is_registered_before_it_is_unregistered() {
    // NRRs are tracked, so unregistering an unknown one is an error.
    const INVALID_NRR: u32 = 22 | (6 << 9);
    const NOT_REGISTERED: u32 = 22 | (1029 << 9);
    const NRR: u64 = 0x1020_0000;
    let (mut cpu, handle) = ldr_ro_session();
    let tls = cpu.tls_base();

    ldr_ro_request(&mut cpu, handle, 3, &[NRR]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), NOT_REGISTERED);

    // Nothing is at that address yet.
    ldr_ro_request(&mut cpu, handle, 2, &[NRR, 0x1000]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), INVALID_NRR);

    cpu.mem.map(NRR as u32, b"NRR0").unwrap();
    ldr_ro_request(&mut cpu, handle, 2, &[NRR, 0x1000]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0);
    ldr_ro_request(&mut cpu, handle, 3, &[NRR]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0);
}
