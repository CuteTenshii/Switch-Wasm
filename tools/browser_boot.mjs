// Boot a container in the built site in a real browser and save what the
// page and its worker said.
// Serves `dist` (run `make assets` first) with WebGPU enabled and a fresh profile.
// `--firmware` installs a directory's NCAs to the NAND first, so library applets run.
// `--nand` launches the title of that name from the NAND tab, in place of a container.
// `--docked` starts the console docked.
// `--dump-wgsl` writes every shader the worker hands the browser into a directory.
// `--browser-arg` passes a switch to Chromium, and can be repeated.
// `--no-webgpu` hides WebGPU, so the software rasterizer draws every frame: the reference
// a device frame is compared against.
// `--debug-from` switches the console to debug once a line contains the text: debug from
// the start floods the log long before the moment worth reading.
// `--headed` opens a window, whose browser gets the machine's GPU instead of a software adapter.
// `--jit-stats` records the cumulative block stats with every sample.
// `--press` taps keyboard keys, as `web/main/input.ts` maps them (z is A, x is B, the arrows
// are the d-pad), that long after the title starts and again every `every` seconds. A tap
// first closes any dialog, since keys belong to the page while one is open.
// `--screenshot-every` also saves the screen every N seconds, beside `--screenshot`'s file.
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { preview } from 'vite';
import { commandLine, fail, launchChromium, repoRoot } from './common.mjs';

const USAGE = `usage: node tools/browser_boot.mjs <container> | --nand=name [--keys=prod.keys] [--title-keys=title.keys]
  [--firmware=dir] [--log-level=debug] [--debug-from=text] [--no-webgpu] [--headed] [--docked]
  [--dump-wgsl=dir] [--browser-arg=switch]
  [--seconds=N] [--out=log.txt] [--screenshot=frame.png] [--screenshot-every=N]
  [--jit-stats] [--press=key@seconds[:every],...]`;

const SAMPLE_MS = 5000;
const START_MS = 120_000;
const PRESS_HOLD_MS = 200;

const { flag, flags, has: hasFlag, path: pathFlag, positional } = commandLine();
const container = positional[0];
const nand = flag('nand');
if (!container && !nand) fail(USAGE);
if (!existsSync(join(repoRoot, 'dist/index.html'))) fail('dist/ has not been built: run `make assets` first');

const keys = pathFlag('keys');
const titleKeys = pathFlag('title-keys');
const firmware = pathFlag('firmware');
const screenshot = pathFlag('screenshot');
const out = flag('out');
const logLevel = flag('log-level');
const debugFrom = flag('debug-from');
const seconds = Number(flag('seconds') ?? 60);
const screenshotEvery = Number(flag('screenshot-every') ?? 0);
const jitStats = hasFlag('jit-stats');
const noWebGpu = hasFlag('no-webgpu');
const headed = hasFlag('headed');
const docked = hasFlag('docked');
const dumpWgsl = pathFlag('dump-wgsl');
const browserArgs = flags('browser-arg');
const presses = (flag('press') ?? '').split(',').filter(Boolean).map((spec) => {
  const match = /^(.+)@(\d+(?:\.\d+)?)(?::(\d+(?:\.\d+)?))?$/.exec(spec);
  if (!match) fail(`--press=${spec}: expected key@seconds[:every]`);
  return { key: match[1], next: Number(match[2]), every: Number(match[3] ?? 0) };
});

const server = await preview({
  configFile: join(repoRoot, 'vite.config.ts'),
  logLevel: 'error',
  preview: { host: '127.0.0.1', port: 0, strictPort: false, open: false },
});
const address = server.httpServer.address();
const url = `http://127.0.0.1:${address.port}/`;

const WGSL_MARK = '@@wgsl ';
let shadersDumped = 0;
const lines = [];
const note = (text) => lines.push(text);
const say = (text) => {
  lines.push(text);
  console.log(text);
};

const browser = await launchChromium({ headed, webGpu: !noWebGpu, args: browserArgs });
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
  if (dumpWgsl) {
    mkdirSync(dumpWgsl, { recursive: true });
    await page.route(/\.js$/, async (route) => {
      const response = await route.fetch();
      const body = `if (typeof GPUDevice !== 'undefined' && typeof WorkerGlobalScope !== 'undefined' && !self.dumpsWgsl) {
        self.dumpsWgsl = true;
        const create = GPUDevice.prototype.createShaderModule;
        GPUDevice.prototype.createShaderModule = function (descriptor) {
          console.log('${WGSL_MARK}' + descriptor.code);
          return create.call(this, descriptor);
        };
          }\n` + await response.text();
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
    worker.on('console', (m) => {
      const text = m.text();
      if (dumpWgsl && text.startsWith(WGSL_MARK)) {
        writeFileSync(join(dumpWgsl, `${shadersDumped++}.wgsl`), text.slice(WGSL_MARK.length));
      } else {
        note(`worker ${m.type()}: ${text}`);
      }
    });
    worker.on('close', () => say('the worker closed'));
  });

  await page.goto(url);
  // "idle" also shows before the core loads; the loading screen closes once it has.
  await page.waitForFunction(() =>
    document.getElementById('loading')?.classList.contains('hidden'));
  // The select sits in the closed console panel, so set it the way a change would.
  await page.evaluate(() => {
    window.setLogLevel = (level) => {
      const select = document.getElementById('log-level');
      select.value = level;
      select.dispatchEvent(new Event('change'));
    };
  });
  if (logLevel) await page.evaluate((level) => window.setLogLevel(level), logLevel);
  // Watched in the page, so the switch lands with the line rather than at the next sample.
  if (debugFrom) {
    await page.evaluate((text) => {
      const consoleEl = document.getElementById('console');
      const observer = new MutationObserver(() => {
        if (!consoleEl.textContent.includes(text)) return;
        observer.disconnect();
        window.setLogLevel('debug');
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
  if (docked) await page.locator('#btn-dock').evaluate((button) => button.click());
  if (nand) {
    await page.locator('#btn-panel').click();
    await page.locator('.tab[data-tab=nand]').click();
    await page.locator('#nand-titles').getByText(nand).click();
    await page.keyboard.press('Escape');
    await page.mouse.click(5, 5);
  } else {
    await page.setInputFiles('#nro-file', resolve(container));
  }
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
  const openDebugTab = async () => {
    if (!(await page.locator('body.panel-open').count())) await page.locator('#btn-panel').click();
    await page.getByRole('tab', { name: 'Debug' }).click();
  };
  // The Translation section counts only while it is on screen.
  if (jitStats) await openDebugTab();

  const started = Date.now();
  const elapsed = () => Math.round((Date.now() - started) / 1000);
  let stopped = false;
  const tap = async (key) => {
    await page.evaluate(() => document.querySelector('dialog[open] button')?.click());
    await page.keyboard.down(key);
    await page.waitForTimeout(PRESS_HOLD_MS);
    await page.keyboard.up(key);
    say(`${elapsed()}s: tapped ${key}`);
  };
  const repeat = async (first, every, action) => {
    for (let at = first; running && !crashed && !stopped && at < seconds; at += every) {
      await page.waitForTimeout(started + at * 1000 - Date.now());
      if (crashed || stopped || Date.now() - started >= seconds * 1000) return;
      await action(at);
      if (!every) return;
    }
  };
  const tapping = presses.map(({ key, next, every }) => repeat(next, every, () => tap(key)));
  const shooting = screenshot && screenshotEvery
    ? repeat(screenshotEvery, screenshotEvery, (at) =>
        page.locator('#screen').screenshot({ path: screenshot.replace(/(\.\w+)?$/, `-${at}s$1`) }))
    : null;
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
    const at = elapsed();
    // The console is capped, so quiet means the last line is unchanged.
    const unchanged = sample.rows === lastRows && sample.last === lastLine;
    const quiet = unchanged ? ', console quiet since the last sample' : '';
    say(`${at}s: ${sample.state}, ${sample.instructions}, ${sample.fps}, ${sample.rows} console lines${quiet}`
      + `; last: ${sample.last.slice(0, 160)}`);
    lastRows = sample.rows;
    lastLine = sample.last;
    // Nothing more will happen, and the log is what the run is for.
    if (sample.state === 'fault' || sample.state === 'halted') {
      stopped = true;
      if (sample.state === 'fault') process.exitCode = 1;
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
      await openDebugTab();
      await page.getByRole('button', { name: 'Graphics' }).click();
      await page.waitForTimeout(3000);
      rendering = await page.locator('#render-debug-section').innerText();
    } catch (error) {
      rendering = `(not read: ${String(error.message).split('\n')[0]})`;
    }
  }
  const consoleSection = `=== the page's console ===\n${pageConsole}`;
  if (out) {
    writeFileSync(out, `${lines.join('\n')}\n=== rendering ===\n${rendering}\n${consoleSection}\n`);
    console.log(`wrote ${out}`);
  } else {
    console.log(consoleSection);
  }
} finally {
  await browser.close();
  await server.close();
}
