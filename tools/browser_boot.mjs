// Boot a container in the built site, in a real browser, and report what the
// page and its worker said:
//   node tools/browser_boot.mjs <container> [--keys=prod.keys] [--title-keys=title.keys]
//                               [--seconds=N] [--out=log.txt] [--jit-stats]
//
// Needs `make assets` first: this serves `dist` with Vite's preview server,
// on a port of its own, so nothing else has to be running. The browser is
// Playwright's (`bunx playwright install chromium`, once per machine) unless
// `CHROMIUM` names another Chromium to drive instead.
//
// What it is for is the browser, which no host test reaches: the worker, the
// page's run loop, the console the page keeps, and the browser's own errors.
// Every few seconds it samples the page's run state and how many lines its
// console holds, so a run that stops making progress is seen stopping rather
// than guessed at afterwards. At the end it presses "GPU stats" and "Thread
// dump" and saves the page's whole console with everything the page and the
// worker printed to the browser's.
//
// WebGPU is switched on, which headless Chromium leaves off. Without it the
// page renders everything on the software rasterizer and a run says nothing
// about the renderer a browser really uses. Headless, the adapter is usually
// SwiftShader's, a CPU implementation: right for whether draws reach the
// device or fall back, wrong for how fast they are.
//
// `--jit-stats` also presses "Block stats" at every sample. The counters are
// cumulative, so the difference between two samples is what the block
// translator and the wasm emitter did in that window: whether the code a
// title spends a given phase in runs compiled is a question only the browser
// build can answer, because nothing on a host compiles an emitted block.
//
// Each run gets a fresh browser profile, so nothing an earlier run stored
// (keys, the SD card, saves, the NAND) leaks into this one.
import { existsSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { preview } from 'vite';

const USAGE =
  'usage: node tools/browser_boot.mjs <container> [--keys=prod.keys] [--title-keys=title.keys]' +
  ' [--seconds=N] [--out=log.txt] [--jit-stats]';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const positional = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const flag = (name) => process.argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const container = positional[0];
if (!container) {
  console.error(USAGE);
  process.exit(1);
}
if (!existsSync(join(root, 'dist/index.html'))) {
  console.error('dist/ has not been built: run `make assets` first');
  process.exit(1);
}
const seconds = Number(flag('seconds') ?? 60);
const out = flag('out');
const jitStats = process.argv.includes('--jit-stats');
// How often the run state is sampled.
const SAMPLE_MS = 5000;

// Resolved against where the command was run, not against the repository: the
// preview server changes nothing, but a relative path is the caller's.
const keys = flag('keys') && resolve(flag('keys'));
const titleKeys = flag('title-keys') && resolve(flag('title-keys'));

const server = await preview({
  configFile: join(root, 'vite.config.ts'),
  logLevel: 'error',
  preview: { port: 0, strictPort: false, open: false },
});
const address = server.httpServer.address();
const url = `http://localhost:${address.port}/`;

const lines = [];
const note = (text) => lines.push(text);
const say = (text) => {
  note(text);
  console.log(text);
};

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM || undefined,
  args: ['--enable-unsafe-webgpu'],
});
try {
  const page = await browser.newPage();
  page.on('console', (m) => note(`page ${m.type()}: ${m.text()}`));
  page.on('pageerror', (e) => say(`page error: ${e.stack || e.message}`));
  // A crashed page answers nothing more, so the run stops sampling it and
  // saves what the page and the worker printed before it went.
  let crashed = false;
  page.on('crash', () => {
    crashed = true;
    say('the page crashed');
  });
  page.on('worker', (worker) => {
    note(`worker started: ${worker.url()}`);
    worker.on('console', (m) => note(`worker ${m.type()}: ${m.text()}`));
    worker.on('close', () => say('the worker closed'));
  });

  await page.goto(url);
  // The page's state reads "idle" before the core has loaded as well as
  // after, so it is the page's own announcement that is waited for.
  await page.waitForFunction(() =>
    [...(document.getElementById('console')?.children ?? [])]
      .some((row) => row.textContent.startsWith('core ready')));
  if (keys) await page.setInputFiles('#prod-keys', keys);
  if (titleKeys) await page.setInputFiles('#title-keys', titleKeys);
  await page.setInputFiles('#nro-file', resolve(container));

  const started = Date.now();
  let lastRows = -1;
  let lastLine = '';
  while (!crashed && Date.now() - started < seconds * 1000) {
    await page.waitForTimeout(SAMPLE_MS);
    if (crashed) break;
    let sample;
    try {
      sample = await page.evaluate(() => ({
        state: document.getElementById('state')?.textContent,
        rows: document.getElementById('console')?.childElementCount ?? 0,
        last: document.getElementById('console')?.lastElementChild?.textContent ?? '',
      }));
    } catch (error) {
      // A crash can land while the page is being asked, before the event
      // that says so has arrived.
      if (!page.isClosed() && !/crash/i.test(String(error))) throw error;
      crashed = true;
      break;
    }
    const at = Math.round((Date.now() - started) / 1000);
    // The console keeps a fixed number of rows, so once it is full the count
    // stops moving while lines still arrive: quiet is the last one unchanged.
    const unchanged = sample.rows === lastRows && sample.last === lastLine;
    const quiet = unchanged ? ', console quiet since the last sample' : '';
    say(`${at}s: ${sample.state}, ${sample.rows} console lines${quiet}; last: ${sample.last.slice(0, 160)}`);
    lastRows = sample.rows;
    lastLine = sample.last;
    if (jitStats) {
      await page.evaluate(() => document.getElementById('btn-jitstats').click());
    }
  }

  let pageConsole = '(the page crashed, and its console went with it)';
  if (!crashed) {
    // Pressed from script: the buttons live in the debug panel, which is not
    // open, and a click through the page's surface would wait for it to show.
    await page.evaluate(() => document.getElementById('btn-gpustats').click());
    await page.evaluate(() => document.getElementById('btn-threads').click());
    await page.waitForTimeout(3000);
    pageConsole = await page.evaluate(() =>
      [...document.getElementById('console').children]
        .map((row) => row.textContent + (row.dataset.repeat ? `  (x${row.dataset.repeat})` : ''))
        .join('\n'));
  }
  const report = `${lines.join('\n')}\n=== the page's console ===\n${pageConsole}\n`;
  if (out) {
    writeFileSync(out, report);
    console.log(`wrote ${out}`);
  } else {
    console.log(`=== the page's console ===\n${pageConsole}`);
  }
} finally {
  await browser.close();
  await server.close();
}
