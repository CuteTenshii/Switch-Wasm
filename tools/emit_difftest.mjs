// The other half of `examples/emit_difftest.rs`: run each emitted block under
// V8 and check it left the guest state the interpreter left.
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
// NZCV going in and coming out. Guest state goes into a bare `WebAssembly.Memory`
// at the offsets the manifest names, so nothing here has to know what a `Cpu`
// looks like.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const dir = process.argv[2] || 'target/emit-difftest';
// A check that has never been seen to fail says nothing. `INJECT=1` corrupts
// one expected register and one expected NZCV, and the run is supposed to
// report exactly those two: it proves the modules really are instantiated and
// the comparison really is reading what came back, rather than comparing two
// copies of the same thing.
const inject = process.env.INJECT === '1';
const manifest = readFileSync(join(dir, 'manifest.txt'), 'utf8').trim().split('\n');

const header = {};
let first = 0;
for (const line of manifest) {
  const [k, v] = line.split(/\s+/);
  if (k === 'regs_at' || k === 'nzcv_at' || k === 'slots') { header[k] = Number(v); first++; }
  else break;
}
const REGS = header.regs_at, NZCV = header.nzcv_at, SLOTS = header.slots;
if (REGS === undefined || NZCV === undefined || SLOTS === undefined) {
  console.error('manifest is missing its header');
  process.exit(1);
}

// One page is plenty: the register file is 34 slots and NZCV sits just past it.
const memory = new WebAssembly.Memory({ initial: 1 });
const view = new DataView(memory.buffer);

let ran = 0, failed = 0;
const failures = [];

for (const line of manifest.slice(first)) {
  const f = line.trim().split(/\s+/);
  const [name, pc, ops, nzcvBefore] = f;
  let nzcvAfter = f[4];
  const bar = f.indexOf('|');
  const before = f.slice(5, bar).map((h) => BigInt('0x' + h));
  const after = f.slice(bar + 1).map((h) => BigInt('0x' + h));
  if (before.length !== SLOTS || after.length !== SLOTS) {
    console.error(`${name}: manifest says ${SLOTS} slots but carries ${before.length}/${after.length}`);
    process.exit(1);
  }

  for (let i = 0; i < SLOTS; i++) view.setBigUint64(REGS + i * 8, before[i], true);
  view.setUint32(NZCV, Number(BigInt(nzcvBefore)), true);

  if (inject && ran === 0) after[0] ^= 1n;
  if (inject && ran === 1) nzcvAfter = '0x' + ((Number(BigInt(nzcvAfter)) ^ 0x40000000) >>> 0).toString(16);

  let exports;
  try {
    const bytes = readFileSync(join(dir, `${name}.wasm`));
    exports = new WebAssembly.Instance(new WebAssembly.Module(bytes), { e: { m: memory } }).exports;
  } catch (e) {
    failures.push(`${name} at ${pc}: module would not compile -- ${e.message}`);
    failed++;
    continue;
  }

  const retired = exports.run(0);
  const bad = [];
  if (retired !== Number(ops)) bad.push(`retired ${retired}, expected ${ops}`);
  for (let i = 0; i < SLOTS; i++) {
    const got = view.getBigUint64(REGS + i * 8, true);
    if (got !== after[i]) {
      bad.push(`slot ${i}: emitted ${got.toString(16).padStart(16, '0')}, interpreted ${after[i].toString(16).padStart(16, '0')}`);
    }
  }
  const gotNzcv = view.getUint32(NZCV, true) >>> 0;
  const wantNzcv = Number(BigInt(nzcvAfter)) >>> 0;
  if (gotNzcv !== wantNzcv) {
    const bits = (v) => 'NZCV'.split('').map((c, i) => ((v >>> (31 - i)) & 1) ? c : '-').join('');
    bad.push(`nzcv: emitted ${bits(gotNzcv)}, interpreted ${bits(wantNzcv)}`);
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
if (inject) {
  const ok = failed === 2;
  console.log(ok
    ? 'INJECT: both planted differences were reported, so the check is live'
    : `INJECT: expected exactly 2 failures, got ${failed} -- the check is not reading what it thinks`);
  process.exit(ok ? 0 : 1);
}
process.exit(failed ? 1 : 0);
