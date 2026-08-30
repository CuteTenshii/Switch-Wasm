//! The scaffolding the CPU test files share: a core to run code on, and the
//! A64 encoders they assemble with.
//!
//! Encodings were verified against QEMU's `a64.decode` where a doubt existed.
//!
//! Each test crate compiles the whole of this module and uses the piece it
//! needs, which is what `dead_code` is doing here.
#![allow(dead_code)]

pub use switch_core::cpu::Cpu;

pub fn cpu_at(pc: u32) -> Cpu {
    let mut cpu = Cpu::new();
    cpu.mem.map_zero(pc, 0x400).unwrap();
    cpu.set_pc(pc);
    cpu
}

/// Little helper for assembling a small program in memory then running it.
pub fn exec(code: &[u32], max: u64) -> Cpu {
    let mut cpu = cpu_at(0x1000);
    let mut bytes = Vec::with_capacity(code.len() * 4);
    for insn in code {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(0x1000, &bytes).unwrap();
    cpu.set_pc(0x1000);
    let steps = max.min(code.len() as u64);
    cpu.run(steps).unwrap();
    cpu
}

/// Map code, run exactly `code.len()` instructions, return the CPU.
pub fn run_program(mut cpu: Cpu, pc: u32, code: &[u32]) -> Cpu {
    let mut bytes = Vec::with_capacity(code.len() * 4);
    for insn in code {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    cpu.mem.map(pc, &bytes).unwrap();
    cpu.set_pc(pc);
    cpu.run(code.len() as u64).unwrap();
    cpu
}

// ---------------- instruction encodings (hand-assembled) ----------------

// ADD Xd, Xn, #imm  :  sf(1) op(0) S(0) 100010 sh imm12 Rn Rd
pub fn add_imm(rd: u32, rn: u32, imm: u32, sf: bool) -> u32 {
    let sf = if sf { 1u32 << 31 } else { 0 };
    sf | 0b100010 << 23 | ((imm & 0xFFF) << 10) | (rn << 5) | rd
}

// ADD Xd, Xn, Xm (shifted, LSL #0)
pub fn add_reg(rd: u32, rn: u32, rm: u32, sf: bool) -> u32 {
    let sf = if sf { 1u32 << 31 } else { 0 };
    sf | 0b01011 << 24 | (rm << 16) | (rn << 5) | rd
}

// MOVZ/MOVN/MOVK Xd, #imm16, LSL #(hw*16)
pub fn movz(rd: u32, imm16: u32, hw: u32, sf: bool) -> u32 {
    let sf = if sf { 1u32 << 31 } else { 0 };
    sf | 0b10 << 29 | 0b100101 << 23 | (hw << 21) | (imm16 << 5) | rd
}

pub fn movn(rd: u32, imm16: u32, hw: u32, sf: bool) -> u32 {
    let sf = if sf { 1u32 << 31 } else { 0 };
    sf | 0b00 << 29 | 0b100101 << 23 | (hw << 21) | (imm16 << 5) | rd
}

pub fn movk(rd: u32, imm16: u32, hw: u32, sf: bool) -> u32 {
    let sf = if sf { 1u32 << 31 } else { 0 };
    sf | 0b11 << 29 | 0b100101 << 23 | (hw << 21) | (imm16 << 5) | rd
}

// LDR Xt, [Xn, #imm]  (unsigned offset, 64-bit)
pub fn ldr64(rt: u32, rn: u32, imm: u32) -> u32 {
    0b11 << 30 | 0b111 << 27 | 0b01 << 24 | 0b01 << 22 | ((imm >> 3) & 0xFFF) << 10 | (rn << 5) | rt
}

pub fn ldr32(rt: u32, rn: u32, imm: u32) -> u32 {
    0b10 << 30 | 0b111 << 27 | 0b01 << 24 | 0b01 << 22 | ((imm >> 2) & 0xFFF) << 10 | (rn << 5) | rt
}

// STR Xt, [Xn, #imm]
pub fn str64(rt: u32, rn: u32, imm: u32) -> u32 {
    0b11 << 30 | 0b111 << 27 | 0b01 << 24 | 0b00 << 22 | ((imm >> 3) & 0xFFF) << 10 | (rn << 5) | rt
}

// LDUR Xt, [Xn, #imm]
pub fn ldur64(rt: u32, rn: u32, imm: i64) -> u32 {
    0b11 << 30
        | 0b111 << 27
        | 0b00 << 24
        | 0b01 << 22
        | ((imm as u32 & 0x1FF) << 12)
        | (rn << 5)
        | rt
}

// B #imm
pub fn b(imm: i32) -> u32 {
    0b000101 << 26 | ((imm >> 2) as u32 & 0x3FF_FFFF)
}

// BL #imm
pub fn bl(imm: i32) -> u32 {
    0b100101 << 26 | ((imm >> 2) as u32 & 0x3FF_FFFF)
}

// BR Xn
pub fn br(rn: u32) -> u32 {
    0xD61F0000 | (rn << 5)
}

// BLR Xn
pub fn blr(rn: u32) -> u32 {
    0xD63F0000 | (rn << 5)
}

// RET Xn
pub fn ret(rn: u32) -> u32 {
    0xD65F0000 | (rn << 5)
}

// B.cond #imm, cond
pub fn bcond(cond: u32, imm: i32) -> u32 {
    0b01010100 << 24 | ((imm >> 2) as u32 & 0x7_FFFF) << 5 | 0x10 | cond
}

// CBZ/CBNZ Xt, #imm
pub fn cbz(rt: u32, imm: i32, sf: bool, nz: bool) -> u32 {
    let sf = if sf { 1u32 << 31 } else { 0 };
    sf | 0b011010 << 25 | ((nz as u32) << 24) | ((imm >> 2) as u32 & 0x7_FFFF) << 5 | rt
}

// TBZ/TBNZ Xt, #bit, #imm
pub fn tbz(rt: u32, bit: u32, imm: i32, nz: bool) -> u32 {
    let sf = (bit >> 5) << 31;
    sf | 0b011011 << 25
        | ((nz as u32) << 24)
        | ((bit & 0x1F) << 19)
        | ((imm >> 2) as u32 & 0x3FFF) << 5
        | rt
}

// SVC #imm
pub fn svc(imm: u32) -> u32 {
    0xD4000000 | (imm << 5) | 1
}

// NOP
pub fn nop() -> u32 {
    0xD503201F
}

// MOV Xd, Xm  == ORR Xd, XZR, Xm
pub fn mov_reg(rd: u32, rm: u32, sf: bool) -> u32 {
    let sf = if sf { 1u32 << 31 } else { 0 };
    sf | 0b01 << 29 | 0b01010 << 24 | (rm << 16) | (31 << 5) | rd
}

// CMP Xn, Xm  == SUBS XZR, Xn, Xm
pub fn cmp_reg(rn: u32, rm: u32, sf: bool) -> u32 {
    let sf = if sf { 1u32 << 31 } else { 0 };
    sf | 0b11 << 29 | 0b01011 << 24 | (rm << 16) | (rn << 5) | 31
}

// ADR Xd, #imm
pub fn adr(rd: u32, imm: i32) -> u32 {
    let imm = imm as u32;
    0b10000 << 24 | (imm & 0x3) << 29 | ((imm >> 2) & 0x7_FFFF) << 5 | rd
}

// MRS Xt, NZCV
pub fn mrs_nzcv(rt: u32) -> u32 {
    0xD53B4200 | rt
}

// MSR XZR, NZCV (write flags from xzr = 0)
pub fn msr_nzcv() -> u32 {
    0xD51B4200 | 31
}

/// `movk x9, #(THREAD_TLS_BASE >> 16), lsl #16` — the second half of building
/// a guest thread's own TLS address, after `mov x9, #stride` supplies the low
/// word. Assembled from the constant rather than written out, because the two
/// starvation tests below reach into thread 1's TLS block by address and a
/// hand-written `movk` goes on pointing at wherever the block used to be.
pub fn movk_x9_tls_high() -> u32 {
    0xf2a0_0000 | ((switch_core::cpu::THREAD_TLS_BASE >> 16) << 5) | 9
}

// ---------------- Advanced SIMD (three-same / logical / permute) ----------------

// dup <Vd>.16B, <Wn>
pub fn dup16(rd: u32, rn: u32) -> u32 {
    0x4E01_0C00u32 | (rd & 0x1F) | ((rn & 0x1F) << 5)
}

// mov <Xd>, <Vn>.D[0]  (umov)
pub fn umov_d0(rd: u32, rn: u32) -> u32 {
    0x4E08_3C00u32 | rd | (rn << 5)
}

// SUB <Vd>.4S, <Vn>.4S, <Vm>.4S
pub fn sub4s(rd: u32, rn: u32, rm: u32) -> u32 {
    (1u32 << 30)
        | (1u32 << 29)
        | (0b1110 << 24)
        | (0b10 << 22)
        | (1 << 21)
        | (rm << 16)
        | (0b10000 << 11)
        | (1 << 10)
        | (rn << 5)
        | rd
}

// CMEQ <Vd>.16B, <Vn>.16B, <Vm>.16B
pub fn cmeq16(rd: u32, rn: u32, rm: u32) -> u32 {
    (1u32 << 30)
        | (1u32 << 29)
        | (0b1110 << 24)
        | (1 << 21)
        | (rm << 16)
        | (0b10001 << 11)
        | (1 << 10)
        | (rn << 5)
        | rd
}

// UHADD <Vd>.16B, <Vn>.16B, <Vm>.16B
pub fn uhadd16(rd: u32, rn: u32, rm: u32) -> u32 {
    // bit21 is what separates three-same from the copy/permute/table space —
    // every other helper here sets it, and this one used to leave it clear.
    // The decoder ignored bit21, so the malformed encoding still reached
    // UHADD; it is really an INS (element) opcode.
    (1u32 << 30)
        | (1u32 << 29)
        | (0b1110 << 24)
        | (1 << 21)
        | (rm << 16)
        | (1 << 10)
        | (rn << 5)
        | rd
}

// ADDP <Vd>.16B, <Vn>.16B, <Vm>.16B
pub fn addp16(rd: u32, rn: u32, rm: u32) -> u32 {
    (1u32 << 30)
        | (0b1110 << 24)
        | (1 << 21)
        | (rm << 16)
        | (0b10111 << 11)
        | (1 << 10)
        | (rn << 5)
        | rd
}

// ZIP1 <Vd>.16B, <Vn>.16B, <Vm>.16B
pub fn zip1_16(rd: u32, rn: u32, rm: u32) -> u32 {
    (1u32 << 30) | (0b1110 << 24) | (rm << 16) | (0b001110 << 10) | (rn << 5) | rd
}

// AdvSIMD table lookup: `0 Q 001110 00 0 Rm 0 len op 00 Rn Rd`. `op` picks
// TBX (1) over TBL (0); `len` is the table size in registers, minus one.
pub fn tbl_insn(q: u32, len: u32, op: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    (q << 30) | (0b001110 << 24) | (rm << 16) | (len << 13) | (op << 12) | (rn << 5) | rd
}

// AdvSIMD copy, element form: `0 1 1 01110000 imm5 0 imm4 1 Rn Rd`.
// `imm5` carries the element size and the destination lane, `imm4` the source
// lane (both shifted down by log2 of the element size).
pub fn ins_elem_b(rd: u32, dst_index: u32, rn: u32, src_index: u32) -> u32 {
    let imm5 = (dst_index << 1) | 1; // size = B → imm5<0> = 1
    0x6E00_0400u32 | (imm5 << 16) | (src_index << 11) | (rn << 5) | rd
}

// AdvSIMD across lanes: `0 Q U 01110 size 11000 opcode(5) 10 Rn Rd`.
pub fn across_lanes(q: u32, u: u32, size: u32, opcode: u32, rd: u32, rn: u32) -> u32 {
    (q << 30)
        | (u << 29)
        | (0b01110 << 24)
        | (size << 22)
        | (0b11000 << 17)
        | (opcode << 12)
        | (0b10 << 10)
        | (rn << 5)
        | rd
}

// ---------------- scalar floating point ----------------

// fmov <Vd>.D, <Xn>
pub fn fmov_dx(rd: u32, rn: u32) -> u32 {
    (1u32 << 31) | (0b0011110 << 24) | (0b01 << 22) | (0b100111 << 16) | (rn << 5) | rd
}

// fmov <Xd>, <Vn>.D
pub fn fmov_xd(rd: u32, rn: u32) -> u32 {
    (1u32 << 31) | (0b0011110 << 24) | (0b01 << 22) | (0b100110 << 16) | (rn << 5) | rd
}

// fadd <Dd>, <Dn>, <Dm>
pub fn fadd_d(rd: u32, rn: u32, rm: u32) -> u32 {
    (0b00011110 << 24) | (0b01 << 22) | (1 << 21) | (rm << 16) | (0b00101 << 11) | (rn << 5) | rd
}

// ---------------- Horizon IPC reply synthesis ----------------

/// A domain request carrying raw arguments after the CmifInHeader. The reply
/// overwrites the request in TLS, so the payload has to go in before the
/// request runs rather than by re-running it.
pub fn ipc_request_with_payload(
    cpu: &mut Cpu,
    handle: u64,
    object_id: u32,
    cmd: u32,
    payload: &[u8],
) {
    build_ipc_request(cpu, 4, Some(object_id), cmd);
    // No buffer descriptors, so the data area starts at 0x10: the domain
    // header, then the CmifInHeader at 0x20, then the arguments at 0x30.
    let tls = cpu.tls_base();
    for (i, &b) in payload.iter().enumerate() {
        cpu.mem.write_u8(tls + 0x30 + i as u32, b).unwrap();
    }
    run_ipc_request(cpu, handle);
}

/// Drive one IPC request at `handle` and return the CPU. The request is built
/// in the guest's own TLS buffer the way `libnx` marshals a CMIF message:
/// hipc header, an optional `CmifDomainInHeader`, then the `SFCI` in-header
/// carrying the command id.
pub fn ipc_request(cpu: &mut Cpu, handle: u64, msg_type: u32, object_id: Option<u32>, cmd: u32) {
    build_ipc_request(cpu, msg_type, object_id, cmd);
    run_ipc_request(cpu, handle);
}

/// Marshal a request into TLS without sending it.
pub fn build_ipc_request(cpu: &mut Cpu, msg_type: u32, object_id: Option<u32>, cmd: u32) {
    let tls = cpu.tls_base();
    for i in (0..0x100u32).step_by(4) {
        cpu.mem.write_u32(tls + i, 0).unwrap();
    }
    cpu.mem.write_u32(tls, msg_type).unwrap();
    cpu.mem.write_u32(tls + 4, 0x0c).unwrap();
    let cmif = match object_id {
        Some(obj) => {
            // CmifDomainRequestType_SendMessage, 0x10 bytes of data.
            cpu.mem.write_u32(tls + 0x10, 0x0010_0001).unwrap();
            cpu.mem.write_u32(tls + 0x14, obj).unwrap();
            tls + 0x20
        }
        None => tls + 0x10,
    };
    cpu.mem.write_u32(cmif, 0x4943_4653).unwrap(); // "SFCI"
    cpu.mem.write_u32(cmif + 8, cmd).unwrap();
}

/// Send whatever is marshalled in TLS to `handle`.
pub fn run_ipc_request(cpu: &mut Cpu, handle: u64) {
    cpu.set_reg(0, handle);
    let pc = cpu.get_pc();
    cpu.mem.map(pc, &svc(0x21).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    cpu.set_pc(pc);
}

/// A bootstrapped Horizon CPU with `appletOE` already bound to a handle and
/// converted to a domain, plus the object ids of the `IApplicationProxy` and
/// `ICommonStateGetter` opened through it.
pub fn applet_chain() -> (Cpu, u64, u32, u32) {
    const APPLET: u64 = 0x1000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(APPLET, "appletOE");
    let tls = cpu.tls_base();

    // Control::ConvertToDomain on the root session -> IApplicationProxyService.
    ipc_request(&mut cpu, APPLET, 5, None, 0);
    let proxy_service = cpu.mem.read_u32(tls + 0x20).unwrap();
    // IApplicationProxyService::OpenApplicationProxy -> IApplicationProxy.
    ipc_request(&mut cpu, APPLET, 4, Some(proxy_service), 0);
    let proxy = cpu.mem.read_u32(tls + 0x30).unwrap();
    // IApplicationProxy::GetCommonStateGetter.
    ipc_request(&mut cpu, APPLET, 4, Some(proxy), 0);
    let state_getter = cpu.mem.read_u32(tls + 0x30).unwrap();
    (cpu, APPLET, proxy, state_getter)
}

/// Build an IPC request carrying one map-alias send buffer, and run it.
pub fn ipc_request_with_buffer(
    cpu: &mut Cpu,
    handle: u64,
    object_id: u32,
    cmd: u32,
    buf: u32,
    len: u32,
    recv: bool,
    payload: &[u8],
) {
    let tls = cpu.tls_base();
    for i in (0..0x100u32).step_by(4) {
        cpu.mem.write_u32(tls + i, 0).unwrap();
    }
    // hdr1: type 4 (Request), one buffer — send buffers count in bits 23:20,
    // receive buffers in 27:24. Either way it is one 12-byte descriptor, so
    // the aligned data area lands at 0x20.
    cpu.mem
        .write_u32(tls, 4 | (1 << if recv { 24 } else { 20 }))
        .unwrap();
    cpu.mem.write_u32(tls + 4, 0x0c).unwrap();
    // HipcBufferDescriptor: size, address, then the high bits (all zero for a
    // 32-bit guest address).
    cpu.mem.write_u32(tls + 0x08, len).unwrap();
    cpu.mem.write_u32(tls + 0x0c, buf).unwrap();
    cpu.mem.write_u32(tls + 0x10, 0).unwrap();
    // One descriptor pushes the aligned data area out to 0x20.
    cpu.mem.write_u32(tls + 0x20, 0x0010_0001).unwrap();
    cpu.mem.write_u32(tls + 0x24, object_id).unwrap();
    cpu.mem.write_u32(tls + 0x30, 0x4943_4653).unwrap(); // "SFCI"
    cpu.mem.write_u32(tls + 0x38, cmd).unwrap();
    // The command's own arguments follow the 16-byte CmifInHeader.
    for (i, &b) in payload.iter().enumerate() {
        cpu.mem.write_u8(tls + 0x40 + i as u32, b).unwrap();
    }
    cpu.set_reg(0, handle);
    let pc = cpu.get_pc();
    cpu.mem.map(pc, &svc(0x21).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    cpu.set_pc(pc);
}

/// Write one `lm` LogPacket at `addr` and return its total length.
pub fn write_log_packet(
    cpu: &mut Cpu,
    addr: u32,
    flags: u8,
    severity: u8,
    tlvs: &[(u8, &[u8])],
) -> u32 {
    let mut payload = Vec::new();
    for &(key, data) in tlvs {
        payload.push(key);
        payload.push(data.len() as u8);
        payload.extend_from_slice(data);
    }
    for i in 0..0x18u32 {
        cpu.mem.write_u8(addr + i, 0).unwrap();
    }
    cpu.mem.write_u8(addr + 0x10, flags).unwrap();
    cpu.mem.write_u8(addr + 0x12, severity).unwrap();
    cpu.mem
        .write_u32(addr + 0x14, payload.len() as u32)
        .unwrap();
    for (i, &b) in payload.iter().enumerate() {
        cpu.mem.write_u8(addr + 0x18 + i as u32, b).unwrap();
    }
    0x18 + payload.len() as u32
}

/// Run one `svcWaitSynchronization` over `handles`, returning (result, index).
pub fn wait_sync(cpu: &mut Cpu, handles: &[u32], timeout: i64) -> (u64, u64) {
    const ARRAY: u32 = 0x7000;
    for (i, &h) in handles.iter().enumerate() {
        cpu.mem.write_u32(ARRAY + (i as u32) * 4, h).unwrap();
    }
    cpu.set_reg(1, u64::from(ARRAY));
    cpu.set_reg(2, handles.len() as u64);
    cpu.set_reg(3, timeout as u64);
    let pc = cpu.get_pc();
    cpu.mem.map(pc, &svc(0x18).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    cpu.set_pc(pc);
    (cpu.read_x(0), cpu.read_x(1))
}

/// Open `hid` and convert it to a domain: (cpu, session handle, IHidServer).
pub fn hid_server() -> (Cpu, u64, u32) {
    const HID: u64 = 0x9000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(HID, "hid");
    let tls = cpu.tls_base();
    // A control reply is not a domain reply: SFCO lands at 0x10 and the raw
    // data (here the new domain object id) at 0x20.
    ipc_request(&mut cpu, HID, 5, None, 0); // ConvertToDomain
    let server = cpu.mem.read_u32(tls + 0x20).unwrap();
    (cpu, HID, server)
}

/// 64 bytes of 0x00..0x3F at `addr`, for the structure load/store tests.
pub fn map_ramp(cpu: &mut Cpu, addr: u32, len: u32) {
    cpu.mem.map_zero(addr, len as usize).unwrap();
    for i in 0..len {
        cpu.mem.write_u8(addr + i, i as u8).unwrap();
    }
}

/// The 16 bytes at `addr` as the u128 a `ld1 {Vt.16b}` would produce.
pub fn mem_u128(cpu: &Cpu, addr: u32) -> u128 {
    u128::from_le_bytes(cpu.mem.dump(addr, 16).unwrap().try_into().unwrap())
}

/// Pack four 32-bit lanes, lane 0 in the low bits.
pub fn u32x4(lanes: [u32; 4]) -> u128 {
    lanes
        .iter()
        .rev()
        .fold(0u128, |acc, &l| (acc << 32) | u128::from(l))
}

pub fn f32x4(lanes: [f32; 4]) -> u128 {
    u32x4([
        lanes[0].to_bits(),
        lanes[1].to_bits(),
        lanes[2].to_bits(),
        lanes[3].to_bits(),
    ])
}

pub fn u64x2(lanes: [u64; 2]) -> u128 {
    u128::from(lanes[0]) | (u128::from(lanes[1]) << 64)
}

pub fn f64x2(lanes: [f64; 2]) -> u128 {
    u64x2([lanes[0].to_bits(), lanes[1].to_bits()])
}

pub fn lanes_f32(v: u128) -> [f32; 4] {
    [0, 1, 2, 3].map(|i| f32::from_bits((v >> (32 * i)) as u32))
}

pub fn lanes_u32(v: u128) -> [u32; 4] {
    [0, 1, 2, 3].map(|i| (v >> (32 * i)) as u32)
}

/// Run one instruction with the vector registers preloaded.
pub fn simd1(insn: u32, regs: &[(u8, u128)]) -> Cpu {
    let mut cpu = cpu_at(0x1000);
    for &(i, v) in regs {
        cpu.set_vreg(i, v);
    }
    cpu.mem.map(0x1000, &insn.to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    cpu
}

pub fn f32b(x: f32) -> u128 {
    u128::from(x.to_bits())
}

pub fn f64b(x: f64) -> u128 {
    u128::from(x.to_bits())
}

/// Marshal a non-domain request (an `SFCI` header straight after the hipc
/// header) with `payload` as its arguments, and send it. `nnSdk` keeps
/// `audout` as a plain session rather than converting it to a domain, so this
/// is the shape those commands actually arrive in.
pub fn ipc_request_plain(cpu: &mut Cpu, handle: u64, cmd: u32, payload: &[u8]) {
    build_ipc_request(cpu, 4, None, cmd);
    let tls = cpu.tls_base();
    for (i, &b) in payload.iter().enumerate() {
        cpu.mem.write_u8(tls + 0x20 + i as u32, b).unwrap();
    }
    run_ipc_request(cpu, handle);
}

/// The same, carrying one map-alias buffer in the direction `recv` asks for.
pub fn ipc_request_plain_with_buffer(
    cpu: &mut Cpu,
    handle: u64,
    cmd: u32,
    buf: u32,
    len: u32,
    recv: bool,
    payload: &[u8],
) {
    let tls = cpu.tls_base();
    for i in (0..0x100u32).step_by(4) {
        cpu.mem.write_u32(tls + i, 0).unwrap();
    }
    // Send buffers count in bits 23:20, receive buffers in 27:24.
    cpu.mem
        .write_u32(tls, 4 | (1 << if recv { 24 } else { 20 }))
        .unwrap();
    cpu.mem.write_u32(tls + 4, 0x0c).unwrap();
    cpu.mem.write_u32(tls + 0x08, len).unwrap();
    cpu.mem.write_u32(tls + 0x0c, buf).unwrap();
    cpu.mem.write_u32(tls + 0x10, 0).unwrap();
    // One descriptor pushes the aligned data area out to 0x20.
    cpu.mem.write_u32(tls + 0x20, 0x4943_4653).unwrap(); // "SFCI"
    cpu.mem.write_u32(tls + 0x28, cmd).unwrap();
    for (i, &b) in payload.iter().enumerate() {
        cpu.mem.write_u8(tls + 0x30 + i as u32, b).unwrap();
    }
    run_ipc_request(cpu, handle);
}

/// A request whose out buffer is marshalled the way `nnSdk`'s `...Auto`
/// commands marshal one: a real receive-static ("pointer") descriptor beside
/// the null map-alias descriptor the caller fills in for the form it did not
/// use. A server that reads only the map-alias one finds address 0.
pub fn ipc_request_auto_recv(
    cpu: &mut Cpu,
    handle: u64,
    cmd: u32,
    buf: u32,
    len: u32,
    payload: &[u8],
) {
    assert!(payload.len() <= 8, "the data words leave room for two");
    let tls = cpu.tls_base();
    for i in (0..0x100u32).step_by(4) {
        cpu.mem.write_u32(tls + i, 0).unwrap();
    }
    // One receive buffer, declared both ways: bits 27:24 count the map-alias
    // descriptors and bits 13:10 encode a single receive-static as 2.
    cpu.mem.write_u32(tls, 4 | (1 << 24)).unwrap();
    cpu.mem.write_u32(tls + 4, 9 | (2 << 10)).unwrap();
    // tls+8 is the map-alias descriptor, and it stays zeroed.
    cpu.mem.write_u32(tls + 0x20, 0x4943_4653).unwrap(); // "SFCI"
    cpu.mem.write_u32(tls + 0x28, cmd).unwrap();
    for (i, &b) in payload.iter().enumerate() {
        cpu.mem.write_u8(tls + 0x30 + i as u32, b).unwrap();
    }
    // The receive-static sits past the data words, as { address, size:16 }.
    cpu.mem.write_u32(tls + 0x38, buf).unwrap();
    cpu.mem.write_u32(tls + 0x3c, len << 16).unwrap();
    run_ipc_request(cpu, handle);
}

/// The section strides of a `RequestUpdateAudioRenderer` **input**, from
/// libnx's `audren.h`. They are not the reply's: an input entry and the output
/// entry describing the same object are different sizes.
pub const AUDREN_IN_HEADER: usize = 0x40;
pub const AUDREN_IN_BEHAVIOR: usize = 0x10;
pub const AUDREN_IN_CHANNEL: usize = 0x70;
pub const AUDREN_IN_VOICE: usize = 0x170;
pub const AUDREN_IN_MIX: usize = 0x930;
pub const AUDREN_IN_SINK: usize = 0x140;
pub const AUDREN_IN_PERF: usize = 0x10;

/// `PcmFormat_Int16` and `PcmFormat_Adpcm`.
pub const PCM_INT16: u8 = 2;
pub const PCM_ADPCM: u8 = 6;

/// One renderer frame in the emulated cycles that are this machine's only
/// clock: 5 ms of a 1.02 GHz CPU.
pub const AUDREN_FRAME_CYCLES: u64 = 1_020_000_000 / 200;

/// One `RequestUpdateAudioRenderer` input buffer, built the way `audrvUpdate`
/// builds it: a header declaring the size of every section, then the sections.
pub struct AudrenUpdate {
    pub data: Vec<u8>,
    pub channels_at: usize,
    pub voices_at: usize,
    pub mixes_at: usize,
    pub sinks_at: usize,
}

impl AudrenUpdate {
    pub fn new(voices: usize, mixes: usize, sinks: usize) -> Self {
        let channels_sz = voices * AUDREN_IN_CHANNEL;
        let voices_sz = voices * AUDREN_IN_VOICE;
        let mixes_sz = mixes * AUDREN_IN_MIX;
        let sinks_sz = sinks * AUDREN_IN_SINK;
        let channels_at = AUDREN_IN_HEADER + AUDREN_IN_BEHAVIOR;
        let voices_at = channels_at + channels_sz;
        let mixes_at = voices_at + voices_sz;
        let sinks_at = mixes_at + mixes_sz;
        let total = sinks_at + sinks_sz + AUDREN_IN_PERF;
        let mut update = AudrenUpdate {
            data: vec![0u8; total],
            channels_at,
            voices_at,
            mixes_at,
            sinks_at,
        };
        update.put(0x00, u32::from_le_bytes(*b"REV9"));
        update.put(0x04, AUDREN_IN_BEHAVIOR as u32);
        // No mempools: guest memory is the renderer's memory here, so a voice
        // plays out of a buffer whether or not a pool was attached over it.
        update.put(0x08, 0);
        update.put(0x0c, voices_sz as u32);
        update.put(0x10, channels_sz as u32);
        update.put(0x18, mixes_sz as u32);
        update.put(0x1c, sinks_sz as u32);
        update.put(0x20, AUDREN_IN_PERF as u32);
        update.put(0x3c, total as u32);
        update
    }

    pub fn put(&mut self, at: usize, value: u32) {
        self.data[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }

    pub fn put_f32(&mut self, at: usize, value: f32) {
        self.put(at, value.to_bits());
    }

    /// A started voice playing one wave buffer of `samples` samples.
    pub fn voice(
        &mut self,
        id: usize,
        format: u8,
        channels: u32,
        address: u32,
        size: u32,
        samples: u32,
    ) {
        let at = self.voices_at + id * AUDREN_IN_VOICE;
        self.put(at, id as u32);
        self.data[at + 0x08] = 1; // is_new
        self.data[at + 0x09] = 1; // is_used
        self.data[at + 0x0a] = 0; // AudioRendererVoicePlayState_Started
        self.data[at + 0x0b] = format;
        self.put(at + 0x0c, 48_000);
        self.put(at + 0x18, channels);
        self.put_f32(at + 0x1c, 1.0); // pitch
        self.put_f32(at + 0x20, 1.0); // volume
        self.put(at + 0x3c, 1); // one wave buffer,
        self.data[at + 0x40] = 0; // at the head of the ring
        self.put(at + 0x58, 0); // playing into the final mix
        let wavebuf = at + 0x60;
        self.put(wavebuf, address);
        self.put(wavebuf + 0x08, size);
        self.put(wavebuf + 0x10, 0); // start_sample_offset
        self.put(wavebuf + 0x14, samples); // end_sample_offset
        for channel in 0..channels as usize {
            self.put(at + 0x140 + channel * 4, channel as u32);
        }
    }

    /// Point a voice at its ADPCM coefficient table.
    pub fn extra_params(&mut self, id: usize, address: u32, size: u32) {
        let at = self.voices_at + id * AUDREN_IN_VOICE;
        self.put(at + 0x48, address);
        self.put(at + 0x50, size);
    }

    /// Send voice channel `channel` into mix buffer `dest` at `gain`.
    pub fn route(&mut self, channel: usize, dest: usize, gain: f32) {
        let at = self.channels_at + channel * AUDREN_IN_CHANNEL;
        self.put_f32(at + 4 + dest * 4, gain);
        self.data[at + 0x64] = 1; // is_used
    }

    /// The final mix, with `buffers` mix buffers and going nowhere but the sink.
    pub fn mix(&mut self, buffers: u32) {
        let at = self.mixes_at;
        self.put_f32(at, 1.0); // volume
        self.put(at + 0x08, buffers);
        self.data[at + 0x0c] = 1; // is_used
        self.put(at + 0x10, 0); // AUDREN_FINAL_MIX_ID
        self.put(at + 0x924, 0x7FFF_FFFF); // AUDREN_UNUSED_MIX_ID
    }

    /// A device sink reading one mix buffer into each output channel.
    pub fn sink(&mut self, inputs: &[u8]) {
        let at = self.sinks_at;
        self.data[at] = 1; // AudioRendererSinkType_Device
        self.data[at + 1] = 1; // is_used
                               // The union sits past the type, the node id and three reserved words,
                               // and a device sink's name fills the 0x100 bytes at the top of it.
        let sink = at + 0x20;
        self.put(sink + 0x100, inputs.len() as u32);
        for (i, &input) in inputs.iter().enumerate() {
            self.data[sink + 0x104 + i] = input;
        }
    }

    /// Write it where the guest would have and send it as `RequestUpdate-
    /// AudioRenderer`, which takes the input and the reply as map-alias
    /// buffers in that order.
    pub fn send(&self, cpu: &mut Cpu, renderer: u64, at: u32, out: u32, out_len: u32) {
        for (i, &b) in self.data.iter().enumerate() {
            cpu.mem.write_u8(at + i as u32, b).unwrap();
        }
        let send = (at, self.data.len() as u32);
        ipc_request_plain_with_both_buffers(cpu, renderer, 10, send, (out, out_len), &[]);
    }
}

/// `OpenAudioRenderer` at 48 kHz, 240 samples a frame, revision 9.
pub fn audren_open(cpu: &mut Cpu, manager: u64, voices: u32, sinks: u32, mix_buffers: u32) -> u64 {
    // `AudioRendererParameter`: rate, sample_count, mix_buffer_count,
    // submix_count, voice_count, sink_count, effect_count, … revision.
    let mut params = vec![0u8; 52];
    params[0..4].copy_from_slice(&48_000u32.to_le_bytes());
    params[4..8].copy_from_slice(&240u32.to_le_bytes());
    params[8..12].copy_from_slice(&mix_buffers.to_le_bytes());
    params[16..20].copy_from_slice(&voices.to_le_bytes());
    params[20..24].copy_from_slice(&sinks.to_le_bytes());
    params[48..52].copy_from_slice(b"REV9");
    ipc_request_plain(cpu, manager, 0, &params);
    u64::from(cpu.mem.read_u32(cpu.tls_base() + 0x0c).unwrap())
}

/// A renderer with one voice, one final mix of two buffers and a stereo device
/// sink — the smallest arrangement that actually plays — plus the manager and
/// renderer handles.
pub fn audren_stereo(cpu: &mut Cpu) -> u64 {
    const AUDREN: u64 = 0xB100;
    cpu.register_service_handle(AUDREN, "audren:u");
    let renderer = audren_open(cpu, AUDREN, 1, 1, 2);
    assert_ne!(renderer, 0, "no IAudioRenderer came back");
    renderer
}

/// Build an IPC request carrying one map-alias send buffer *and* one
/// map-alias receive buffer, the shape `IHOSBinderDriver::TransactParcel`
/// arrives in, and run it.
pub fn ipc_request_plain_with_both_buffers(
    cpu: &mut Cpu,
    handle: u64,
    cmd: u32,
    send: (u32, u32),
    recv: (u32, u32),
    payload: &[u8],
) {
    let tls = cpu.tls_base();
    for i in (0..0x100u32).step_by(4) {
        cpu.mem.write_u32(tls + i, 0).unwrap();
    }
    // Send buffers count in bits 23:20, receive buffers in 27:24.
    cpu.mem.write_u32(tls, 4 | (1 << 20) | (1 << 24)).unwrap();
    cpu.mem.write_u32(tls + 4, 0x0c).unwrap();
    cpu.mem.write_u32(tls + 0x08, send.1).unwrap();
    cpu.mem.write_u32(tls + 0x0c, send.0).unwrap();
    cpu.mem.write_u32(tls + 0x10, 0).unwrap();
    cpu.mem.write_u32(tls + 0x14, recv.1).unwrap();
    cpu.mem.write_u32(tls + 0x18, recv.0).unwrap();
    cpu.mem.write_u32(tls + 0x1c, 0).unwrap();
    // Two descriptors fill the words up to the aligned data area at 0x20.
    cpu.mem.write_u32(tls + 0x20, 0x4943_4653).unwrap(); // "SFCI"
    cpu.mem.write_u32(tls + 0x28, cmd).unwrap();
    for (i, &b) in payload.iter().enumerate() {
        cpu.mem.write_u8(tls + 0x30 + i as u32, b).unwrap();
    }
    run_ipc_request(cpu, handle);
}

/// One `IGraphicBufferProducer` request parcel: the interface token every
/// transaction starts with, followed by `body`.
pub fn binder_parcel(body: &[u8]) -> Vec<u8> {
    const NAME: &str = "android.gui.IGraphicBufferProducer";
    let mut payload = Vec::new();
    payload.extend_from_slice(&0x100u32.to_le_bytes()); // strict-mode policy
    payload.extend_from_slice(&(NAME.len() as u32).to_le_bytes());
    for c in NAME.chars().chain(std::iter::once('\0')) {
        payload.extend_from_slice(&(c as u16).to_le_bytes());
    }
    while payload.len() % 4 != 0 {
        payload.push(0);
    }
    payload.extend_from_slice(body);
    let mut parcel = Vec::new();
    parcel.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    parcel.extend_from_slice(&16u32.to_le_bytes()); // payload offset
    parcel.extend_from_slice(&0u32.to_le_bytes()); // no flattened objects
    parcel.extend_from_slice(&(16 + payload.len() as u32).to_le_bytes());
    parcel.extend_from_slice(&payload);
    parcel
}

/// `svcResetSignal(handle)`.
pub fn reset_signal(cpu: &mut Cpu, handle: u32) -> u64 {
    cpu.set_reg(0, u64::from(handle));
    let pc = cpu.get_pc();
    cpu.mem.map(pc, &svc(0x19).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    cpu.set_pc(pc);
    cpu.read_x(0)
}

// CRC32/CRC32C: sf 0 0 11010110 Rm 010 C sz Rn Rd. sf is set only for the
// doubleword form, whose data operand is an X register.
pub fn crc32(rd: u32, rn: u32, rm: u32, castagnoli: bool, sz: u32) -> u32 {
    let sf = if sz == 0b11 { 1u32 << 31 } else { 0 };
    let c = u32::from(castagnoli);
    sf | 0b11010110 << 21 | (rm << 16) | (0b010 << 13) | (c << 12) | (sz << 10) | (rn << 5) | rd
}

/// A minimal but well-formed NRO image: three 0x1000-byte segments and a
/// 0x1000-byte BSS, with the "NRO0" header at the offset a real NRO keeps it
/// (0x10, behind the entry branch and the `MOD0` pointer). Each segment is
/// filled with a distinct byte so the test can tell what landed where.
pub fn test_nro_image() -> Vec<u8> {
    const SEGMENT: u32 = 0x1000;
    let mut nro = vec![0u8; 3 * SEGMENT as usize];
    nro[0..4].copy_from_slice(&0x1400_0010u32.to_le_bytes()); // b entry
    nro[0x10..0x14].copy_from_slice(b"NRO0");
    for (offset, value) in [
        (0x14, 0),           // version
        (0x18, 3 * SEGMENT), // total size
        (0x1C, 0),           // flags
        (0x20, 0),           // .text offset
        (0x24, SEGMENT),     // .text size
        (0x28, SEGMENT),     // .rodata offset
        (0x2C, SEGMENT),     // .rodata size
        (0x30, 2 * SEGMENT), // .data offset
        (0x34, SEGMENT),     // .data size
        (0x38, SEGMENT),     // .bss size
    ] {
        nro[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    for (segment, fill) in [(1usize, 0xAAu8), (2, 0xBB)] {
        let start = segment * SEGMENT as usize;
        nro[start..start + SEGMENT as usize].fill(fill);
    }
    nro
}

/// A bootstrapped CPU with `ldr:ro` bound to a handle and the image above
/// already in guest memory at `NRO_SOURCE`, ready to be loaded.
pub fn ldr_ro_session() -> (Cpu, u64) {
    const LDR_RO: u64 = 0x2000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(LDR_RO, "ldr:ro");
    cpu.mem.map(0x1000_0000, &test_nro_image()).unwrap();
    (cpu, LDR_RO)
}

/// Where `ldr_ro_session` puts the caller's copy of the NRO, and the BSS
/// buffer it passes alongside it.
pub const NRO_SOURCE: u64 = 0x1000_0000;
pub const NRO_BSS: u64 = 0x1010_0000;

/// Marshal an `ldr:ro` request: the `u64` pid placeholder every command on the
/// interface opens with, then `args`.
pub fn ldr_ro_request(cpu: &mut Cpu, handle: u64, cmd: u32, args: &[u64]) {
    build_ipc_request(cpu, 4, None, cmd);
    let data = cpu.tls_base() + 0x20;
    cpu.mem.write_u64(data, 0).unwrap();
    for (i, &arg) in args.iter().enumerate() {
        cpu.mem.write_u64(data + 8 + 8 * i as u32, arg).unwrap();
    }
    run_ipc_request(cpu, handle);
}

// ---------------- cryptographic extension and half-precision ----------------

// AESE/AESD/AESMC/AESIMC: 0100 1110 00 10100 opcode(5) 10 Rn Rd
pub fn aes(opcode: u32, rd: u32, rn: u32) -> u32 {
    0x4E << 24 | 0b10100 << 17 | (opcode << 12) | 0b10 << 10 | (rn << 5) | rd
}

// Three-register SHA: 0101 1110 000 Rm 0 opcode(3) 00 Rn Rd
pub fn sha3(opcode: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    0x5E << 24 | (rm << 16) | (opcode << 12) | (rn << 5) | rd
}

// Two-register SHA: 0101 1110 00 10100 opcode(5) 10 Rn Rd
pub fn sha2(opcode: u32, rd: u32, rn: u32) -> u32 {
    0x5E << 24 | 0b10100 << 17 | (opcode << 12) | 0b10 << 10 | (rn << 5) | rd
}

// PMULL/PMULL2: 0 Q 0 01110 size 1 Rm 1110 00 Rn Rd
pub fn pmull(q: u32, size: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    (q << 30) | 0b01110 << 24 | (size << 22) | 1 << 21 | (rm << 16) | 0b1110 << 12 | (rn << 5) | rd
}

// Scalar FCVT: 0001 1110 ftype 1 0001 opc(2) 10000 Rn Rd
pub fn fcvt(ftype: u32, opc: u32, rd: u32, rn: u32) -> u32 {
    0x1E << 24
        | (ftype << 22)
        | 1 << 21
        | 0b0001 << 17
        | (opc << 15)
        | 0b10000 << 10
        | (rn << 5)
        | rd
}

// ---------------- variable (register) shifts ----------------

// Vector three-same: 0 Q U 01110 size 1 Rm opcode(5) 1 Rn Rd
pub fn simd_shift_reg(q: u32, u: u32, size: u32, op: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    (q << 30)
        | (u << 29)
        | 0b01110 << 24
        | (size << 22)
        | 1 << 21
        | (rm << 16)
        | (op << 11)
        | 1 << 10
        | (rn << 5)
        | rd
}

// Scalar three-same: 01 U 11110 size 1 Rm opcode(5) 1 Rn Rd
pub fn scalar_shift_reg(u: u32, size: u32, op: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    0b01 << 30
        | (u << 29)
        | 0b11110 << 24
        | (size << 22)
        | 1 << 21
        | (rm << 16)
        | (op << 11)
        | 1 << 10
        | (rn << 5)
        | rd
}

// ---------------- FPCR and FPSR ----------------

// `1101010100 L op0 op1 CRn CRm op2 Rt` — op0 is bits[20:19], so it does not
// fit in the 0xD53 prefix. `mrs x0, fpcr` is 0xD53B4400.
pub fn sysreg_move(read: bool, rt: u32, op0: u32, op1: u32, crn: u32, crm: u32, op2: u32) -> u32 {
    0b1101010100 << 22
        | u32::from(read) << 21
        | (op0 << 19)
        | (op1 << 16)
        | (crn << 12)
        | (crm << 8)
        | (op2 << 5)
        | rt
}

pub fn mrs(rt: u32, op1: u32, crn: u32, crm: u32, op2: u32) -> u32 {
    sysreg_move(true, rt, 3, op1, crn, crm, op2)
}

pub fn msr(rt: u32, op1: u32, crn: u32, crm: u32, op2: u32) -> u32 {
    sysreg_move(false, rt, 3, op1, crn, crm, op2)
}

pub fn fdiv_d(rd: u32, rn: u32, rm: u32) -> u32 {
    0x1E << 24 | 1 << 22 | 1 << 21 | (rm << 16) | 0b0001 << 12 | 0b10 << 10 | (rn << 5) | rd
}

/// Run one program through both engines and assert they agree with the value
/// the architecture calls for. The translator and the interpreter share these
/// helpers, so a decode bug in one is a decode bug in both — which is exactly
/// how the two below survived.
pub fn both_engines(setup: &[(u8, u64)], code: &[u32]) -> (Cpu, Cpu) {
    let mut out = Vec::new();
    for jit in [true, false] {
        let mut cpu = cpu_at(0x1000);
        cpu.set_jit_enabled(jit);
        for &(reg, val) in setup {
            cpu.set_reg(reg, val);
        }
        out.push(run_program(cpu, 0x1000, code));
    }
    let mut it = out.into_iter();
    (it.next().unwrap(), it.next().unwrap())
}

pub const OPUS_PACKET: [u8; 164] = [
    0xf8, 0x9f, 0xf7, 0xda, 0x9b, 0x32, 0x2b, 0xce, 0x91, 0xf2, 0x50, 0x86, 0xd0, 0xbe, 0x88, 0x91,
    0xe5, 0xfc, 0xff, 0xd1, 0xb8, 0x45, 0x4f, 0x82, 0x93, 0xbc, 0xa6, 0x61, 0x9e, 0x76, 0x03, 0x86,
    0x83, 0xf1, 0x65, 0x96, 0x94, 0xab, 0x3a, 0x3a, 0xaa, 0xb0, 0x12, 0x91, 0x97, 0xb5, 0x53, 0xd8,
    0x2f, 0x4d, 0xf4, 0x71, 0xc9, 0xdc, 0x90, 0xc5, 0x89, 0xdd, 0x76, 0xf2, 0xf0, 0x6d, 0xd1, 0x23,
    0x1a, 0xe6, 0x16, 0xcb, 0x37, 0x81, 0x53, 0xe9, 0x70, 0x84, 0x65, 0xc0, 0x8b, 0xb0, 0x29, 0x32,
    0x7b, 0xf2, 0x56, 0x71, 0x16, 0xc0, 0xc9, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x0d, 0xb2, 0x65, 0x69, 0x55, 0x44, 0x5f, 0x55, 0x62, 0xab, 0xe7, 0x8b, 0x74, 0x5b, 0x50, 0x1f,
    0x0f, 0xb4, 0x32, 0x07, 0xa3, 0xe8, 0x5d, 0x0a, 0xbe, 0x45, 0xd7, 0x13, 0xc8, 0xc5, 0x34, 0x08,
    0x98, 0x83, 0xc0, 0x29, 0x72, 0xf6, 0x33, 0xd6, 0xe2, 0x01, 0x31, 0x70, 0xfc, 0x4e, 0x5b, 0x77,
    0xf3, 0x98, 0x1c, 0x97, 0x17, 0xf6, 0xa4, 0xe7, 0x70, 0x96, 0x5b, 0x3e, 0x09, 0x53, 0x28, 0x6a,
    0xd9, 0xe3, 0xaa, 0x85,
];

pub const SSHL: u32 = 0b01000;
pub const SQSHL: u32 = 0b01001;
pub const SRSHL: u32 = 0b01010;
pub const SQRSHL: u32 = 0b01011;

pub const FPCR_REG: (u32, u32, u32, u32) = (3, 4, 4, 0);
pub const FPSR_REG: (u32, u32, u32, u32) = (3, 4, 4, 1);
