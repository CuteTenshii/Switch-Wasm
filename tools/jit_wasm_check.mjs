// Check that the browser build runs the blocks it emits, against the
// interpreter, on small guest loops:
//   node tools/jit_wasm_check.mjs [--wasm=<switch_wasm_bg.wasm>]
//
// Needs `make wasm` to have been run.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { commandLine, repoRoot } from './common.mjs';

const out = join(repoRoot, 'target/wasm32-unknown-unknown/debug');
const wasmPath = commandLine().flag('wasm') || join(out, 'switch_wasm_bg.wasm');

const api = await import(pathToFileURL(join(out, 'switch_wasm.js')).href);
await api.default({ module_or_path: readFileSync(wasmPath) });

const BASE = 0x04000000;
const DATA = BASE + 0x1000;

// Each A32 program is a loop through `bx lr` back to its own start, as the
// A64 ones are through `ret x30`. Encodings are `llvm-mc -triple=armv7-none-eabi`.
const A32_PROGRAMS = [
  {
    name: 'A32 data processing: flags, carries, shifts',
    a32: true,
    data: false,
    code: [
      0xe24fe008, // sub  lr, pc, #8      -> lr = BASE
      0xe2800001, // add  r0, r0, #1
      0xe0911000, // adds r1, r1, r0
      0xe2b22000, // adcs r2, r2, #0
      0xe0233181, // eor  r3, r3, r1, lsl #3
      0xe1a042e0, // mov  r4, r0, ror #5
      0xe0c55004, // sbc  r5, r5, r4
      0xe12fff1e, // bx   lr
    ],
    steps: 8001n,
    expect: { 0: 1000n },
  },
  {
    name: 'A32 loads and stores: write-back, halfwords, multiples, a skipped add',
    a32: true,
    data: true,
    code: [
      0xe24fe008, // sub   lr, pc, #8     -> lr = BASE
      0xe3011008, // movw  r1, #0x1008
      0xe3401400, // movt  r1, #0x0400    -> r1 = DATA + 8
      0xe2800001, // add   r0, r0, #1
      0xe4810004, // str   r0, [r1], #4
      0xe5a10004, // str   r0, [r1, #4]!
      0xe5912000, // ldr   r2, [r1]
      0xe15130b4, // ldrh  r3, [r1, #-4]
      0xe1d140d0, // ldrsb r4, [r1]
      0xe2505032, // subs  r5, r0, #50
      0x02866001, // addeq r6, r6, #1
      0xe9010005, // stmdb r1, {r0, r2}
      0xe8b10180, // ldmia r1!, {r7, r8}
      0xe12fff1e, // bx    lr
    ],
    steps: 14001n,
    expect: { 0: 1000n },
  },
  {
    name: 'A32 calls and returns, through conditional branches and links',
    a32: true,
    data: false,
    code: [
      0xe2800001, // add    r0, r0, #1
      0xeb000003, // bl     #0x18        -> f1
      0xeb000004, // bl     #0x20        -> f2
      0xe3500000, // cmp    r0, #0
      0x1afffffa, // bne    #-0x18       -> back to the add
      0xe320f000, // nop
      0xe2822001, // f1: add r2, r2, #1
      0xe12fff1e, //     bx  lr
      0xe0833000, // f2: add r3, r3, r0
      0xe0934002, //     adds r4, r3, r2
      0x03a05001, //     moveq r5, #1
      0x112fff1e, //     bxne lr
      0xe12fff1e, //     bx   lr
    ],
    steps: 20001n,
    expect: {},
  },
  {
    name: 'A32 exclusives, thread pointer reads, BLX register, LDR pc and POP {pc}',
    a32: true,
    data: true,
    code: [
      0xe24fe008, // sub   lr, pc, #8     -> lr = BASE
      0xe1a0800e, // mov   r8, lr
      0xe3011008, // movw  r1, #0x1008
      0xe3401400, // movt  r1, #0x0400    -> r1 = DATA + 8
      0xe281dc01, // add   sp, r1, #0x100
      0xe2800001, // add   r0, r0, #1
      0xe1912f9f, // ldrex r2, [r1]
      0xe1813f90, // strex r3, r0, [r1]
      0xee1d4f70, // mrc   p15, 0, r4, c13, c0, 3
      0xe28f900c, // add   r9, pc, #12    -> the push
      0xe5819004, // str   r9, [r1, #4]
      0xe28fc00c, // add   r12, pc, #12   -> the add below the pop
      0xe12fff3c, // blx   r12
      0xe591f004, // ldr   pc, [r1, #4]
      0xe92d0101, // push  {r0, r8}
      0xe8bd8020, // pop   {r5, pc}       -> BASE
      0xe2866001, // add   r6, r6, #1
      0xe12fff1e, // bx    lr
    ],
    steps: 17001n,
    expect: {},
  },
  {
    name: 'A32 loop whose back edge is a conditional branch',
    a32: true,
    data: false,
    code: [
      0xe24fe008, // sub  lr, pc, #8      -> lr = BASE
      0xe2800001, // add  r0, r0, #1
      0xe3500064, // cmp  r0, #100
      0x1afffffc, // bne  #-0x10          -> back to the add
      0xe2822001, // add  r2, r2, #1
      0xe12fff1e, // bx   lr
    ],
    steps: 4001n,
    expect: {},
  },
];

// Each program is one block the emitter can write whole; `adr` puts the loop's
// own address in x30.
const PROGRAMS = [
  {
    name: 'arithmetic',
    data: false,
    code: [
      0x1000001e, // adr  x30, #0        -> x30 = BASE
      0xd2824680, // movz x0,  #0x1234
      0xd503201f, // nop
      0xd65f03c0, // ret  x30
    ],
    // Expected state after the run, so a run that did nothing cannot pass.
    expect: { 0: 0x1234n },
  },
  {
    name: 'a store and the load that reads it back',
    data: true,
    code: [
      0x1000001e, // adr  x30, #0        -> x30 = BASE
      0xd2824680, // movz x0,  #0x1234
      0x10008001, // adr  x1,  #0x1000   -> x1 = DATA + 8
      0xf9000020, // str  x0,  [x1]
      0xf9400022, // ldr  x2,  [x1]
      0xd65f03c0, // ret  x30
    ],
    expect: { 0: 0x1234n, 2: 0x1234n },
  },
  {
    name: 'a barrier, and a store-release the load-acquire reads back',
    data: true,
    code: [
      0x1000001e, // adr  x30, #0        -> x30 = BASE
      0xd2824680, // movz x0,  #0x1234
      0x10008001, // adr  x1,  #0x1000   -> x1 = DATA + 8
      0xd5033bbf, // dmb  ish
      0xc89ffc20, // stlr x0,  [x1]
      0xc8dffc22, // ldar x2,  [x1]
      0xd65f03c0, // ret  x30
    ],
    expect: { 0: 0x1234n, 2: 0x1234n },
  },
  {
    name: 'a loop whose back edge is a conditional branch',
    data: false,
    code: [
      0x1000001e, // adr  x30, #0        -> x30 = BASE
      0x91000400, // add  x0, x0, #1
      0xf101901f, // cmp  x0, #0x64
      0x54ffffa1, // b.ne #-0xc          -> back to BASE
      0xd65f03c0, // ret  x30
    ],
    // x0 passes 100 and never matches again: 4001 steps end with x0 = 1000.
    expect: { 0: 1000n },
  },
  {
    name: 'a branch out of one block and into another',
    data: false,
    code: [
      0x1000001e, // adr  x30, #0        -> x30 = BASE
      0x91000400, // add  x0, x0, #1
      0x36000060, // tbz  w0, #0, #0xc   -> BASE+20 when x0 is even
      0x91000442, // add  x2, x2, #1     -> the not-taken path, x0 odd
      0xd65f03c0, // ret  x30
      0x91000463, // add  x3, x3, #1     -> the taken path, x0 even
      0xd65f03c0, // ret  x30
    ],
    // Either path is five instructions: 4001 steps is 800 trips plus one.
    expect: { 0: 800n, 2: 400n, 3: 400n },
  },
  {
    name: 'the clock, read in blocks that branch into each other',
    data: false,
    code: [
      0x1000001e, // adr  x30, #0        -> x30 = BASE
      0x91000400, // add  x0, x0, #1
      0xd53be046, // mrs  x6, cntvct_el0
      0x8b0600a5, // add  x5, x5, x6     -> a clock off by a few cycles changes the sum
      0x36000060, // tbz  w0, #0, #0xc   -> BASE+0x1c when x0 is even
      0xd53be027, // mrs  x7, cntpct_el0
      0xd65f03c0, // ret  x30
      0x91000463, // add  x3, x3, #1
      0xd53be047, // mrs  x7, cntvct_el0
      0xd65f03c0, // ret  x30
    ],
    // An odd and an even trip are fifteen instructions: 20001 steps is 1333 pairs and six more.
    steps: 20001n,
    expect: { 0: 2667n, 3: 1333n },
  },
  {
    name: 'calls and returns',
    data: false,
    code: [
      0x91000400, // add  x0, x0, #1
      0x94000004, // bl   #0x10          -> BASE+0x14
      0x100000a4, // adr  x4, #0x14      -> x4 = BASE+0x1c
      0xd63f0080, // blr  x4
      0x17fffffc, // b    #-0x10         -> back to BASE
      0x91000442, // add  x2, x2, #1
      0xd65f03c0, // ret
      0x91000463, // add  x3, x3, #1
      0xd65f03c0, // ret
    ],
    // Nine instructions a trip; enough trips for each block to be compiled.
    steps: 8001n,
    expect: { 0: 889n, 2: 889n, 3: 889n },
  },
];

function elf(program) {
  return program.a32 ? elf32(program) : elf64(program);
}

function elf64(program) {
  const code = new Uint8Array(program.code.length * 4);
  const cv = new DataView(code.buffer);
  program.code.forEach((insn, i) => cv.setUint32(i * 4, insn, true));

  const EHDR = 64;
  const PHDR = 56;
  const phnum = program.data ? 2 : 1;
  const body = EHDR + PHDR * phnum;
  const file = new Uint8Array(body + code.length);
  const v = new DataView(file.buffer);
  file.set([0x7f, 0x45, 0x4c, 0x46, 2, 1, 1, 0], 0); // \x7fELF, 64-bit, LE
  v.setUint16(16, 2, true); // ET_EXEC
  v.setUint16(18, 183, true); // EM_AARCH64
  v.setUint32(20, 1, true); // EV_CURRENT
  v.setBigUint64(24, BigInt(BASE), true); // e_entry
  v.setBigUint64(32, BigInt(EHDR), true); // e_phoff
  v.setUint16(52, EHDR, true); // e_ehsize
  v.setUint16(54, PHDR, true); // e_phentsize
  v.setUint16(56, phnum, true); // e_phnum

  const segment = (i, { offset, vaddr, filesz, memsz, flags }) => {
    const ph = EHDR + i * PHDR;
    v.setUint32(ph + 0, 1, true); // PT_LOAD
    v.setUint32(ph + 4, flags, true);
    v.setBigUint64(ph + 8, BigInt(offset), true);
    v.setBigUint64(ph + 16, BigInt(vaddr), true);
    v.setBigUint64(ph + 24, BigInt(vaddr), true);
    v.setBigUint64(ph + 32, BigInt(filesz), true);
    v.setBigUint64(ph + 40, BigInt(memsz), true);
    v.setBigUint64(ph + 48, 4096n, true);
  };
  segment(0, {
    offset: body,
    vaddr: BASE,
    filesz: code.length,
    memsz: code.length,
    flags: 5, // R+X
  });
  if (program.data) {
    // Zero-filled and writable: the store's target.
    segment(1, { offset: body, vaddr: DATA, filesz: 0, memsz: 0x1000, flags: 6 });
  }
  file.set(code, body);
  return file;
}

// A 32-bit little-endian ARM `ET_EXEC` with the same layout as `elf64`'s.
function elf32(program) {
  const code = new Uint8Array(program.code.length * 4);
  const cv = new DataView(code.buffer);
  program.code.forEach((insn, i) => cv.setUint32(i * 4, insn, true));

  const EHDR = 52;
  const PHDR = 32;
  const phnum = program.data ? 2 : 1;
  const body = EHDR + PHDR * phnum;
  const file = new Uint8Array(body + code.length);
  const v = new DataView(file.buffer);
  file.set([0x7f, 0x45, 0x4c, 0x46, 1, 1, 1, 0], 0); // \x7fELF, 32-bit, LE
  v.setUint16(16, 2, true); // ET_EXEC
  v.setUint16(18, 40, true); // EM_ARM
  v.setUint32(20, 1, true); // EV_CURRENT
  v.setUint32(24, BASE, true); // e_entry
  v.setUint32(28, EHDR, true); // e_phoff
  v.setUint16(40, EHDR, true); // e_ehsize
  v.setUint16(42, PHDR, true); // e_phentsize
  v.setUint16(44, phnum, true); // e_phnum

  const segment = (i, { offset, vaddr, filesz, memsz, flags }) => {
    const ph = EHDR + i * PHDR;
    v.setUint32(ph + 0, 1, true); // PT_LOAD
    v.setUint32(ph + 4, offset, true);
    v.setUint32(ph + 8, vaddr, true);
    v.setUint32(ph + 12, vaddr, true);
    v.setUint32(ph + 16, filesz, true);
    v.setUint32(ph + 20, memsz, true);
    v.setUint32(ph + 24, flags, true);
    v.setUint32(ph + 28, 4096, true);
  };
  segment(0, { offset: body, vaddr: BASE, filesz: code.length, memsz: code.length, flags: 5 });
  if (program.data) {
    segment(1, { offset: body, vaddr: DATA, filesz: 0, memsz: 0x1000, flags: 6 });
  }
  file.set(code, body);
  return file;
}

const STEPS = 4001n;

async function run(program, jit) {
  api.switch_init();
  api.switch_set_jit(jit);
  const blob = new Blob([elf(program)]);
  const entry = await api.switch_load_program(blob);
  if (entry !== BASE) {
    throw new Error(`the core loaded the program at ${entry}, not ${BASE}`);
  }
  const steps = api.switch_run(Number(program.steps ?? STEPS));
  const regs = [];
  for (let i = 0; i < (program.a32 ? 15 : 31); i++) regs.push(api.switch_get_reg(i));
  return {
    steps,
    regs,
    // Every register, the flags and the pc, as the core prints them.
    dump: api.switch_reg_dump(),
    retired: api.switch_get_steps(),
    jit: JSON.parse(api.switch_jit_stats_json()),
  };
}

let failed = false;
const check = (what, a, b) => {
  if (a === b) return;
  failed = true;
  console.log(`  FAIL ${what}: ${a} interpreted, ${b} translated`);
};

let totalChained = 0;
for (const program of [...PROGRAMS, ...A32_PROGRAMS]) {
  console.log(`${program.name}:`);
  const interpreted = await run(program, false);
  const translated = await run(program, true);

  check('steps run', interpreted.steps, translated.steps);
  check('steps retired', interpreted.retired, translated.retired);
  check('registers, flags and pc', interpreted.dump, translated.dump);
  for (let i = 0; i < interpreted.regs.length; i++) {
    check(`r${i}`, interpreted.regs[i], translated.regs[i]);
  }

  for (const [reg, want] of Object.entries(program.expect)) {
    if (interpreted.regs[reg] !== want) {
      failed = true;
      console.log(`  FAIL the interpreter left register ${reg} = ${interpreted.regs[reg]}, not ${want}`);
    }
  }

  const { emitted, enteredEmitted, executed, chained } = translated.jit;
  console.log(
    `  ${emitted} block(s) compiled, entered ${enteredEmitted} of ${executed} times,`
    + ` ${chained} by a jump from another`,
  );
  totalChained += chained;
  if (!emitted) {
    failed = true;
    console.log('  FAIL nothing was compiled, so the comparison proved nothing');
  } else if (!enteredEmitted) {
    failed = true;
    console.log('  FAIL a block was compiled and then never entered');
  }
}

if (!totalChained) {
  failed = true;
  console.log('\nFAIL no compiled block jumped into another');
}

console.log(failed ? '\nthe emitted path is wrong' : '\nthe emitted path runs');
process.exit(failed ? 1 : 0);
