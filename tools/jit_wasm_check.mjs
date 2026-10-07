// Check that the browser build runs the blocks it emits, against the
// interpreter, on small guest loops:
//   node tools/jit_wasm_check.mjs [switch_wasm_bg.wasm]
//
// Needs `make wasm` to have been run.
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const out = join(root, 'target/wasm32-unknown-unknown/debug');
const wasmPath = process.argv[2] || join(out, 'switch_wasm_bg.wasm');

// Node cannot resolve `@host/files`, so point it at a shim that refuses.
const shim =
  'data:text/javascript,'
  + encodeURIComponent('export const hostRead = () => 0;');
const glue = join(out, 'switch_wasm.jitcheck.mjs');
writeFileSync(
  glue,
  readFileSync(join(out, 'switch_wasm.js'), 'utf8').replace(
    '\'@host/files\'',
    JSON.stringify(shim),
  ),
);
const init = (await import(pathToFileURL(glue).href)).default;
const api = await init({ module_or_path: readFileSync(wasmPath) });

const BASE = 0x04000000;
const DATA = BASE + 0x1000;

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

function withBytes(bytes, body) {
  const ptr = api.switch_alloc(bytes.length) >>> 0;
  if (!ptr) throw new Error('the core refused the staging buffer');
  try {
    new Uint8Array(api.memory.buffer, ptr, bytes.length).set(bytes);
    return body(ptr, bytes.length);
  } finally {
    api.switch_free(ptr, bytes.length);
  }
}

function readJson(fill) {
  const cap = 4096;
  const ptr = api.switch_alloc(cap) >>> 0;
  try {
    const n = fill(ptr, cap);
    return JSON.parse(
      new TextDecoder().decode(new Uint8Array(api.memory.buffer, ptr, n)),
    );
  } finally {
    api.switch_free(ptr, cap);
  }
}

const STEPS = 4001n;

function run(program, jit) {
  const handle = api.switch_new();
  api.switch_set_jit(handle, jit ? 1 : 0);
  const entry = withBytes(elf(program), (ptr, len) =>
    api.switch_load_elf(handle, ptr, len),
  );
  if (entry !== BigInt(BASE)) {
    throw new Error(`the core loaded the program at ${entry}, not ${BASE}`);
  }
  const steps = api.switch_run(handle, program.steps ?? STEPS);
  const regs = [];
  for (let i = 0; i < 31; i++) regs.push(api.switch_get_reg(handle, i));
  const state = {
    steps,
    regs,
    pc: BigInt(api.switch_get_pc(handle) >>> 0),
    cycles: api.switch_get_cycles(handle),
    jit: readJson((ptr, cap) => api.switch_jit_stats_json(handle, ptr, cap)),
  };
  api.switch_free_session(handle);
  return state;
}

api.switch_init();

let failed = false;
const check = (what, a, b) => {
  if (a === b) return;
  failed = true;
  console.log(`  FAIL ${what}: ${a} interpreted, ${b} translated`);
};

let totalChained = 0;
for (const program of PROGRAMS) {
  console.log(`${program.name}:`);
  const interpreted = run(program, false);
  const translated = run(program, true);

  check('steps retired', interpreted.steps, translated.steps);
  check('cycles', interpreted.cycles, translated.cycles);
  check('pc', interpreted.pc, translated.pc);
  for (let i = 0; i < 31; i++) {
    check(`x${i}`, interpreted.regs[i], translated.regs[i]);
  }

  for (const [reg, want] of Object.entries(program.expect)) {
    if (interpreted.regs[reg] !== want) {
      failed = true;
      console.log(`  FAIL the interpreter left x${reg} = ${interpreted.regs[reg]}, not ${want}`);
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
