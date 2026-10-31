// Boot a container in the built site in a real browser and save what the
// page and its worker said:
//   node tools/browser_boot.mjs <container> [--keys=prod.keys] [--title-keys=title.keys]
//                               [--firmware=dir] [--log-level=debug] [--debug-from=text]
//                               [--no-webgpu] [--seconds=N] [--out=log.txt]
//                               [--screenshot=frame.png] [--screenshot-every=N] [--jit-stats]
//                               [--press=key@seconds[:every],...]
// Serves `dist` (run `make assets` first) with WebGPU enabled and a fresh profile.
// `--firmware` installs a directory's NCAs to the NAND first, so library applets run.
// `--no-webgpu` hides WebGPU, so the software rasterizer draws every frame: the reference
// a device frame is compared against.
// `--debug-from` switches the console to debug once a line contains the text: debug from
// the start floods the log long before the moment worth reading.
// `HEADED=1` opens a window, whose browser gets the machine's GPU instead of a software adapter.
// `--jit-stats` records the cumulative block stats with every sample.
// `--press` taps keyboard keys, as `web/main/input.ts` maps them (z is A, x is B, the arrows
// are the d-pad), that long after the title starts and again every `every` seconds. A tap
// first closes any dialog, since keys belong to the page while one is open.
// `--screenshot-every` also saves the screen every N seconds, beside `--screenshot`'s file.
import { existsSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { preview } from 'vite';

const USAGE =
  'usage: node tools/browser_boot.mjs <container> [--keys=prod.keys] [--title-keys=title.keys]'
  + ' [--firmware=dir] [--log-level=debug] [--debug-from=text] [--no-webgpu] [--seconds=N] [--out=log.txt] [--screenshot=frame.png] [--screenshot-every=N] [--jit-stats] [--press=key@seconds[:every],...]';

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
const START_MS = 120_000;

// Relative to the caller's working directory.
const keys = flag('keys') && resolve(flag('keys'));
const titleKeys = flag('title-keys') && resolve(flag('title-keys'));
const firmware = flag('firmware') && resolve(flag('firmware'));
const screenshot = flag('screenshot') && resolve(flag('screenshot'));
const logLevel = flag('log-level');
const debugFrom = flag('debug-from');
const noWebGpu = process.argv.includes('--no-webgpu');
const screenshotEvery = Number(flag('screenshot-every') ?? 0);
const PRESS_HOLD_MS = 200;
const presses = (flag('press') ?? '').split(',').filter(Boolean).map((spec) => {
  const match = /^(.+)@(\d+(?:\.\d+)?)(?::(\d+(?:\.\d+)?))?$/.exec(spec);
  if (!match) {
    console.error(`--press=${spec}: expected key@seconds[:every]`);
    process.exit(1);
  }
  return { key: match[1], next: Number(match[2]), every: Number(match[3] ?? 0) };
});

const server = await preview({
  configFile: join(root, 'vite.config.ts'),
  logLevel: 'error',
  preview: { host: '127.0.0.1', port: 0, strictPort: false, open: false },
});
const address = server.httpServer.address();
const url = `http://127.0.0.1:${address.port}/`;

const lines = [];
const note = (text) => lines.push(text);
const say = (text) => {
  note(text);
  console.log(text);
};

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM || undefined,
  headless: !process.env.HEADED,
  args: noWebGpu ? [] : ['--enable-unsafe-webgpu'],
});
try {
  const page = await browser.newPage();
  // Chromium has no switch that takes WebGPU away, so the property goes before any
  // script runs: init scripts reach the page, and the worker's scripts are rewritten.
  if (noWebGpu) {
    await page.addInitScript(() => delete Navigator.prototype.gpu);
    await page.route(/\.js$/, async (route) => {
      const response = await route.fetch();
      const body = 'if (typeof WorkerNavigator !== \'undefined\') delete WorkerNavigator.prototype.gpu;\n'
        + await response.text();
      await route.fulfill({ response, body });
    });
  }
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
  // "idle" also shows before the core loads; the loading screen closes once it has.
  await page.waitForFunction(() =>
    document.getElementById('loading')?.classList.contains('hidden'));
  // The select sits in the closed console panel, so set it the way a change would.
  if (logLevel) {
    await page.evaluate((level) => {
      const select = document.getElementById('log-level');
      select.value = level;
      select.dispatchEvent(new Event('change'));
    }, logLevel);
  }
  // Watched in the page, so the switch lands with the line rather than at the next sample.
  if (debugFrom) {
    await page.evaluate((text) => {
      const consoleEl = document.getElementById('console');
      const observer = new MutationObserver(() => {
        if (!consoleEl.textContent.includes(text)) return;
        observer.disconnect();
        const select = document.getElementById('log-level');
        select.value = 'debug';
        select.dispatchEvent(new Event('change'));
      });
      observer.observe(consoleEl, { childList: true });
    }, debugFrom);
  }
  if (keys) await page.setInputFiles('#prod-keys', keys);
  if (titleKeys) await page.setInputFiles('#title-keys', titleKeys);
  if (firmware) {
    const ncas = readdirSync(firmware).filter((name) => /\.nca$/i.test(name));
    await page.setInputFiles('#firmware-ncas', ncas.map((name) => join(firmware, name)));
    // The page logs `Installed N title(s)` once every archive is registered.
    let progress = '';
    while (!(await page.locator('#console', { hasText: /Installed \d+ title/ }).count())) {
      await page.waitForTimeout(2000);
      const now = await page.locator('#firmware-state').innerText();
      if (now !== progress) say(`firmware: ${now}`);
      progress = now;
    }
    say(`installed the firmware in ${firmware} (${ncas.length} NCAs)`);
  }
  const before = await page.evaluate(() => document.getElementById('console').childElementCount);
  await page.setInputFiles('#nro-file', resolve(container));
  // Fail now, not after the whole run, when the title never starts.
  const booting = Date.now();
  let running = false;
  while (!crashed) {
    const { state, error } = await page.evaluate((from) => ({
      state: document.body.dataset.state,
      error: [...document.getElementById('console').children].slice(from)
        .some((row) => row.textContent.startsWith('e> ')),
    }), before);
    if (state === 'running') {
      running = true;
      say(`the title is running after ${Math.round((Date.now() - booting) / 1000)}s`);
      break;
    }
    if (error || state === 'fault' || state === 'halted' || Date.now() - booting > START_MS) {
      const tail = await page.evaluate(() =>
        [...document.getElementById('console').children].slice(-15)
          .map((row) => row.textContent).join('\n'));
      say(`the title did not start (state: ${state}); the console ends with:\n${tail}`);
      process.exitCode = 1;
      break;
    }
    await page.waitForTimeout(1000);
  }
  // The Translation section counts only while it is on screen.
  if (jitStats) {
    await page.locator('#btn-panel').click();
    await page.getByRole('tab', { name: 'Debug' }).click();
  }

  const started = Date.now();
  let stopped = false;
  const tap = async (key) => {
    await page.evaluate(() => document.querySelector('dialog[open] button')?.click());
    await page.keyboard.down(key);
    await page.waitForTimeout(PRESS_HOLD_MS);
    await page.keyboard.up(key);
    say(`${Math.round((Date.now() - started) / 1000)}s: tapped ${key}`);
  };
  const tapping = presses.map(async ({ key, next, every }) => {
    let at = next;
    while (running && !crashed && !stopped && at < seconds) {
      await page.waitForTimeout(started + at * 1000 - Date.now());
      if (crashed || stopped || Date.now() - started >= seconds * 1000) return;
      await tap(key);
      if (!every) return;
      at += every;
    }
  });
  const shooting = (async () => {
    if (!screenshot || !screenshotEvery) return;
    for (let at = screenshotEvery; running && !crashed && !stopped && at < seconds; at += screenshotEvery) {
      await page.waitForTimeout(started + at * 1000 - Date.now());
      if (crashed || stopped) return;
      const path = screenshot.replace(/(\.\w+)?$/, `-${at}s$1`);
      await page.locator('#screen').screenshot({ path });
    }
  })();
  let lastRows = -1;
  let lastLine = '';
  while (running && !crashed && Date.now() - started < seconds * 1000) {
    await page.waitForTimeout(SAMPLE_MS);
    if (crashed) break;
    let sample;
    try {
      sample = await page.evaluate(() => ({
        state: document.body.dataset.state,
        instructions: document.getElementById('instructions')?.textContent ?? '',
        fps: document.getElementById('fps')?.textContent ?? '',
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
    say(`${at}s: ${sample.state}, ${sample.instructions}, ${sample.fps}, ${sample.rows} console lines${quiet}; last: ${sample.last.slice(0, 160)}`);
    lastRows = sample.rows;
    lastLine = sample.last;
    // Nothing more will happen, and the log is what the run is for.
    if (sample.state === 'fault') {
      stopped = true;
      process.exitCode = 1;
      break;
    }
    if (jitStats) {
      const blocks = await page.locator('#jit-debug-data').innerText();
      say(`${at}s blocks: ${blocks.replace(/\n+/g, ' ').replace(/\t/g, ': ')}`);
    }
  }

  await Promise.allSettled([...tapping, shooting]);
  if (screenshot && !crashed) {
    await page.locator('#screen').screenshot({ path: screenshot });
    say(`saved the screen to ${screenshot}`);
  }

  let pageConsole = '(the page crashed, and its console went with it)';
  let rendering = '(not read)';
  if (!crashed) {
    // Save's download holds every line; the view keeps only the newest.
    const saved = page.waitForEvent('download', { timeout: 60_000 });
    await page.evaluate(() => document.getElementById('btn-save-console').click());
    pageConsole = readFileSync(await (await saved).path(), 'utf8');
    // The Rendering section refreshes only while it is on screen; a page busy logging
    // can be too slow to open it, which costs the section and not the report.
    try {
      if (!(await page.locator('body.panel-open').count())) await page.locator('#btn-panel').click();
      await page.getByRole('tab', { name: 'Debug' }).click();
      await page.getByRole('button', { name: 'Graphics' }).click();
      await page.waitForTimeout(3000);
      rendering = await page.locator('#render-debug-section').innerText();
    } catch (error) {
      rendering = `(not read: ${String(error.message).split('\n')[0]})`;
    }
  }
  const report = `${lines.join('\n')}\n=== rendering ===\n${rendering}\n`
    + `=== the page's console ===\n${pageConsole}\n`;
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
