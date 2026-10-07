// Boot a container in the built site in a real browser and save what the
// page and its worker said:
//   node tools/browser_boot.mjs <container> [--keys=prod.keys] [--title-keys=title.keys]
//                               [--seconds=N] [--out=log.txt] [--jit-stats]
// Serves `dist` (run `make assets` first) with WebGPU enabled and a fresh profile.
// `--jit-stats` samples the cumulative block stats every interval.
import { existsSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { preview } from 'vite';

const USAGE =
  'usage: node tools/browser_boot.mjs <container> [--keys=prod.keys] [--title-keys=title.keys]'
  + ' [--seconds=N] [--out=log.txt] [--jit-stats]';

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
const SAMPLE_MS = 5000;

// Relative to the caller's working directory.
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
  // "idle" also shows before the core loads, so wait for the announcement.
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
      // A crash can land before its event arrives.
      if (!page.isClosed() && !/crash/i.test(String(error))) throw error;
      crashed = true;
      break;
    }
    const at = Math.round((Date.now() - started) / 1000);
    // The console is capped, so quiet means the last line is unchanged.
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
    // Clicked from script because the debug panel is closed.
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
