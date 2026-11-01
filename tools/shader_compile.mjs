// Time how long the driver takes over WGSL shaders, slowest first:
//   node tools/shader_compile.mjs <file-or-dir>... [--replace=pattern::text]...
// Shaders come from `browser_boot.mjs --dump-wgsl=dir`. Each `--replace` is a regular expression
// applied to every shader, to time a change against the original before writing it in the emitter.
// A window opens, because only a headed browser gets the machine's GPU, and the watchdog is off so
// that a shader slow enough to be killed still gets a number. Literals are changed per run
// so the driver's on-disk cache does not answer instead.
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { basename, join } from 'node:path';
import { commandLine, fail, launchChromium } from './common.mjs';

const USAGE = 'usage: node tools/shader_compile.mjs <file-or-dir>... [--replace=pattern::text]...';
const { flags, positional: paths } = commandLine();
const replacements = flags('replace').map((replacement) => {
  const [pattern, text] = replacement.split('::');
  return [new RegExp(pattern, 'g'), text ?? ''];
});
if (!paths.length) fail(USAGE);
const files = paths.flatMap((path) => statSync(path).isDirectory()
  ? readdirSync(path).filter((name) => name.endsWith('.wgsl')).map((name) => join(path, name))
  : [path]);

const nonce = Date.now() % 1_000_000;
let salts = 0;
function salted(code) {
  salts += 1;
  let done = false;
  const out = code.replace(/\b(\d{5,9})u\b/, (_, n) => {
    done = true;
    return `${Number(n) + nonce + salts}u`;
  });
  return { code: out, salted: done };
}

// The vertex stage a fragment shader's inputs need, with its interpolation.
function vertexFor(code) {
  const struct = /struct FragmentInput \{([\s\S]*?)\n\}/.exec(code)?.[1] ?? '';
  const outputs = [...struct.matchAll(/@location\((\d+)\)[^\n]*?(@interpolate\([^)]*\))?\s*\w+:\s*(vec\d<\w+>|\w+),/g)]
    .map(([, location, interpolate, type], i) => `@location(${location}) ${interpolate ?? ''} v${i}: ${type},`);
  return `struct Out { @builtin(position) position: vec4<f32>, ${outputs.join(' ')} }
    @vertex fn vs_main() -> Out { var out: Out; return out; }`;
}

const NO_FRAGMENT = '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.0); }';

async function compile(page, code) {
  const isFragment = code.includes('@fragment');
  return page.evaluate(async ({ code, isFragment, vertex, noFragment }) => {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice({ requiredFeatures: [...adapter.features] });
    const given = device.createShaderModule({ code });
    const stage = device.createShaderModule({ code: isFragment ? vertex : noFragment });
    const targets = [{ format: 'rgba8unorm' }];
    const started = performance.now();
    try {
      await device.createRenderPipelineAsync({
        layout: 'auto',
        vertex: { module: isFragment ? stage : given },
        fragment: { module: isFragment ? given : stage, targets },
      });
      return { ms: Math.round(performance.now() - started) };
    } catch (error) {
      return { error: String(error).split('\n')[0] };
    } finally {
      device.destroy();
    }
  }, { code, isFragment, vertex: vertexFor(code), noFragment: NO_FRAGMENT });
}

const browser = await launchChromium({ args: ['--disable-gpu-watchdog'] });
try {
  const page = await browser.newPage();
  // WebGPU needs a secure context, which localhost is; nothing is served.
  await page.route('http://localhost/', (route) =>
    route.fulfill({ contentType: 'text/html', body: '<title>shader compile</title>' }));
  await page.goto('http://localhost/');

  const results = [];
  for (const file of files) {
    const source = readFileSync(file, 'utf8');
    const changed = replacements.reduce((code, [pattern, text]) => code.replace(pattern, text), source);
    const base = salted(source);
    const row = { name: basename(file), bytes: source.length, cached: !base.salted, ...await compile(page, base.code) };
    if (replacements.length) row.replaced = await compile(page, salted(changed).code);
    results.push(row);
    console.error(`${row.name}: ${row.ms ?? row.error}`);
  }

  results.sort((a, b) => (b.ms ?? -1) - (a.ms ?? -1));
  console.log('      ms  replaced     bytes  shader');
  for (const row of results) {
    const replaced = row.replaced ? String(row.replaced.ms ?? 'error') : '-';
    const note = row.error ?? (row.cached ? 'no literal to change: the driver may have cached it' : '');
    console.log(`${String(row.ms ?? '-').padStart(8)}  ${replaced.padStart(8)}  ${String(row.bytes).padStart(8)}  ${row.name} ${note}`);
  }
  const total = results.reduce((sum, row) => sum + (row.ms ?? 0), 0);
  console.log(`total ${total} ms over ${results.length} shaders`);
} finally {
  await browser.close();
}
