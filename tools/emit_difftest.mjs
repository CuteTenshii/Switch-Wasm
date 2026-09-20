// The other half of `examples/emit_difftest.rs` and `examples/emit_selftest.rs`:
// run each emitted block under V8 and check it left the guest state the
// interpreter left.
//
//   cargo run --profile quick --example emit_difftest -- <target>
//   node tools/emit_difftest.mjs [outdir]
//
// The Rust half cannot do this itself. `switch-core` has no dependencies and
// on the host there is no wasm engine, so the emitter's output can only be
// checked by something that can instantiate a module. That is the whole reason
// this is two programs: the interpreter is the reference, and it lives in the
// other one.
//
// A case is a module exporting `run(state) -> i32`, plus the register file and
// NZCV going in and coming out. Guest state goes into a bare
// `WebAssembly.Memory` at the offsets the manifest names, so nothing here has
// to know what a `Cpu` looks like.
//
// Guest *memory* is named the same way: a page table of four-byte entries, one
// per 4 KiB of the guest's address space, holding where in this memory a page
// lives and zero where it is not mapped. That is the shape `crate::mem::Memory`
// has, and an emitted access walks it directly. A manifest that names no
// window leaves every entry zero, which is a legitimate state: every access
// then hands its instruction back, and `run` reports how far it got.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const dir = process.argv[2] || 'target/emit-difftest';
// A check that has never been seen to fail says nothing. `INJECT=1` corrupts
// one expected register and one expected NZCV, and the run is supposed to
// report exactly those two: it proves the modules really are instantiated and
// the comparison really is reading what came back, rather than comparing two
// copies of the same thing.
const inject = process.env.INJECT === '1';
const manifest = readFileSync(join(dir, 'manifest.txt')).toString('utf8').trim().split('\n');

const HEADER_KEYS = new Set([
  'regs_at', 'nzcv_at', 'read_watch_lo_at', 'read_watch_hi_at', 'watch_lo_at',
  'watch_hi_at', 'readonly_lo_at', 'readonly_hi_at', 'watched_at', 'pages_at',
  'table_at', 'bitmap_at', 'slots', 'code_at', 'read_watch', 'watch',
  'readonly', 'watched_page', 'wasm_pages',
]);
const header = {};
// `(guest address, where in this memory it lives)` per mapped page, in guest
// order, which is the order `guest.bin` holds their contents in. They are
// deliberately not in that order in memory: a real page is its own allocation,
// and an access that runs off the end of one must not find its guest neighbour
// sitting behind it.
const pages = [];
let first = 0;
for (const line of manifest) {
  const parts = line.split(/\s+/);
  if (parts[0] === 'page') {
    pages.push([Number(parts[1]), Number(parts[2])]);
  } else if (HEADER_KEYS.has(parts[0])) {
    header[parts[0]] = parts.slice(1).map(Number);
  } else {
    break;
  }
  first++;
}

function need(key) {
  if (header[key] === undefined) {
    console.error(`manifest is missing ${key}`);
    process.exit(1);
  }
  return header[key][0];
}
const REGS = need('regs_at');
const NZCV = need('nzcv_at');
const SLOTS = need('slots');
const RW_LO_AT = need('read_watch_lo_at');
const RW_HI_AT = need('read_watch_hi_at');
const WATCH_LO_AT = need('watch_lo_at');
const WATCH_HI_AT = need('watch_hi_at');
const READONLY_LO_AT = need('readonly_lo_at');
const READONLY_HI_AT = need('readonly_hi_at');
const WATCHED_AT = need('watched_at');
const PAGES_AT = need('pages_at');
const TABLE_AT = need('table_at');
const PAGE_BYTES = 4096;

const memory = new WebAssembly.Memory({ initial: need('wasm_pages') });
const view = new DataView(memory.buffer);
const bytes = new Uint8Array(memory.buffer);

// Where the page table is, and the watchpoint bounds. Both are fields of the
// `Memory` an emitted block reads through, so the harness writes them where the
// manifest says that `Memory`'s would be.
view.setUint32(PAGES_AT, TABLE_AT, true);
// `(1, 0)` is how a disarmed watchpoint is written: nothing is below zero, so
// the overlap test can never hold. An empty set of protected ranges is
// `(0xffffffff, 0)`, which fails the same way from the other end.
const [readLo, readHi] = header.read_watch ?? [1, 0];
view.setUint32(RW_LO_AT, readLo, true);
view.setUint32(RW_HI_AT, readHi, true);
const [writeLo, writeHi] = header.watch ?? [1, 0];
view.setUint32(WATCH_LO_AT, writeLo, true);
view.setUint32(WATCH_HI_AT, writeHi, true);
const [roLo, roHi] = header.readonly ?? [0xffffffff, 0];
view.setUint32(READONLY_LO_AT, roLo, true);
view.setUint32(READONLY_HI_AT, roHi, true);

// The bitmap of pages whose contents something has cached, which a store tests
// before it can be written inline. Left null when the manifest watches no
// page: that is how a `Memory` with nothing watching it is written, and the
// emitted test has to cope with it.
if (header.watched_page) {
  const bitmap = need('bitmap_at');
  view.setUint32(WATCHED_AT, bitmap, true);
  for (const guest of header.watched_page) {
    const idx = guest >>> 12;
    const at = bitmap + (idx >>> 6) * 8;
    view.setBigUint64(at, view.getBigUint64(at, true) | (1n << BigInt(idx & 63)), true);
  }
}

// The mapped pages, if there are any: a page-table entry each, and their
// contents as the interpreter had them.
const codeAt = header.code_at ? header.code_at[0] : -1;
let pristine = null, codeWasm = -1;
if (pages.length) {
  pristine = new Uint8Array(readFileSync(join(dir, 'guest.bin')));
  if (pristine.length !== pages.length * PAGE_BYTES) {
    console.error(`guest.bin is ${pristine.length} bytes, manifest maps ${pages.length} pages`);
    process.exit(1);
  }
  for (const [guest, wasm] of pages) {
    view.setUint32(TABLE_AT + (guest >>> 12) * 4, wasm, true);
    if (guest === codeAt) codeWasm = wasm;
  }
}

// The code page's contents belong to the case rather than to the window every
// case shares, so the manifest carries them and they go back over it before
// each one.
let code = null;
// The window as a case starts from it, in guest order: what `resetGuest` puts
// back, and what an expected delta is applied to.
const start = pristine ? new Uint8Array(pristine.length) : null;
function resetGuest() {
  if (!pristine) return;
  start.set(pristine);
  if (code) start.set(code, codeAt - pages[0][0]);
  for (let i = 0; i < pages.length; i++) {
    bytes.set(start.subarray(i * PAGE_BYTES, (i + 1) * PAGE_BYTES), pages[i][1]);
  }
}

// Whether the window still holds `want`, which is the case's starting bytes
// with whatever the interpreter changed laid over them. Checked for every case
// that can reach guest memory at all, so a store that lands in the right place
// with the wrong bytes, or in the wrong place entirely, is caught rather than
// only a store that leaves a register wrong.
function windowDiffers(want) {
  for (let i = 0; i < pages.length; i++) {
    const at = pages[i][1];
    for (let j = 0; j < PAGE_BYTES; j++) {
      if (bytes[at + j] !== want[i * PAGE_BYTES + j]) {
        return `guest ${(pages[i][0] + j).toString(16)}: emitted ${bytes[at + j].toString(16)}, interpreted ${want[i * PAGE_BYTES + j].toString(16)}`;
      }
    }
  }
  return null;
}

let ran = 0, failed = 0, handedBack = 0;
const failures = [];

for (const line of manifest.slice(first)) {
  const f = line.trim().split(/\s+/);
  if (f[0] === 'code') {
    code = Uint8Array.from(f[1].match(/../g) ?? [], (b) => parseInt(b, 16));
    continue;
  }
  const [name, pc, ops, mode, nzcvBefore, nzcvAfter] = f;
  const bar = f.indexOf('|');
  const bang = f.indexOf('!');
  const before = f.slice(6, bar).map((h) => BigInt('0x' + h));
  const after = f.slice(bar + 1, bang).map((h) => BigInt('0x' + h));
  // `offset:hex` per run of bytes the block changed in the window.
  const delta = f.slice(bang + 1).map((run) => {
    const [at, hex] = run.split(':');
    return [Number(at), Uint8Array.from(hex.match(/../g), (b) => parseInt(b, 16))];
  });
  if (before.length !== SLOTS || after.length !== SLOTS) {
    console.error(`${name}: manifest says ${SLOTS} slots but carries ${before.length}/${after.length}`);
    process.exit(1);
  }

  resetGuest();
  for (let i = 0; i < SLOTS; i++) view.setBigUint64(REGS + i * 8, before[i], true);
  view.setUint32(NZCV, Number(BigInt(nzcvBefore)), true);

  let exports;
  try {
    const module = readFileSync(join(dir, `${name}.wasm`));
    exports = new WebAssembly.Instance(new WebAssembly.Module(module), { e: { m: memory } }).exports;
  } catch (e) {
    failures.push(`${name} at ${pc}: module would not compile -- ${e.message}`);
    failed++;
    continue;
  }

  const retired = exports.run(0);
  const bad = [];
  // An emitted block reports how much of itself it did, and the manifest says
  // which answers are allowed. `exact` is a block with no access in it, so all
  // of it. `maybe` is one whose single access the page table may or may not be
  // able to answer for, so all or none. `none` is one the interpreter's
  // watchpoint saw: reading it takes more than the page table, so the emitted
  // block has to have handed it back, and a block that went ahead anyway would
  // be leaving the watchpoint blind.
  const allowed = { exact: [Number(ops)], maybe: [Number(ops), 0], none: [0] }[mode];
  if (!allowed) {
    console.error(`${name}: manifest asks for an unknown mode ${mode}`);
    process.exit(1);
  }
  let want = after, wantNzcv = nzcvAfter;
  if (!allowed.includes(retired)) {
    bad.push(`retired ${retired}, expected ${allowed.join(' or ')}`);
    want = null;
  } else if (retired === 0 && Number(ops) !== 0) {
    want = before;
    wantNzcv = nzcvBefore;
    handedBack++;
  }

  if (want) {
    if (inject && ran === 0) want = want.map((v, i) => (i === 0 ? v ^ 1n : v));
    if (inject && ran === 1) wantNzcv = '0x' + ((Number(BigInt(wantNzcv)) ^ 0x40000000) >>> 0).toString(16);
    for (let i = 0; i < SLOTS; i++) {
      const got = view.getBigUint64(REGS + i * 8, true);
      if (got !== want[i]) {
        bad.push(`slot ${i}: emitted ${got.toString(16).padStart(16, '0')}, interpreted ${want[i].toString(16).padStart(16, '0')}`);
      }
    }
    const gotNzcv = view.getUint32(NZCV, true) >>> 0;
    if (gotNzcv !== (Number(BigInt(wantNzcv)) >>> 0)) {
      const bits = (v) => 'NZCV'.split('').map((c, i) => ((v >>> (31 - i)) & 1) ? c : '-').join('');
      bad.push(`nzcv: emitted ${bits(gotNzcv)}, interpreted ${bits(Number(BigInt(wantNzcv)) >>> 0)}`);
    }
    // Only for a block that can reach memory. An `exact` block has no access
    // in it, so the only stores its body contains are to the register file and
    // to NZCV, and both are compared above.
    if (pristine && mode !== 'exact') {
      if (want === after) for (const [at, run] of delta) start.set(run, at);
      const off = windowDiffers(start);
      if (off) bad.push(off);
    }
  }

  ran++;
  if (bad.length) {
    failed++;
    if (failures.length < 20) failures.push(`${name} at ${pc} (${ops} instructions):\n    ` + bad.join('\n    '));
  }
}

for (const f of failures) console.log(f);
if (failures.length && failed > failures.length) {
  console.log(`... and ${failed - failures.length} more`);
}
console.log(`\n${ran} blocks, ${ran - failed} agree, ${failed} differ`);
console.log(`${handedBack} handed their first instruction back to the interpreter`);
if (inject) {
  const ok = failed === 2;
  console.log(ok
    ? 'INJECT: both planted differences were reported, so the check is live'
    : `INJECT: expected exactly 2 failures, got ${failed} -- the check is not reading what it thinks`);
  process.exit(ok ? 0 : 1);
}
process.exit(failed ? 1 : 0);
