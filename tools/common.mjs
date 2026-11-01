// What the tools in this directory share.
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const repoRoot = join(dirname(fileURLToPath(import.meta.url)), '..');

export function fail(message) {
  console.error(message);
  process.exit(1);
}

/** `--name=value` and `--name` flags and bare arguments, from this process's command line. */
export function commandLine() {
  const args = process.argv.slice(2);
  const prefix = (name) => `--${name}=`;
  const flags = (name) => args.filter((a) => a.startsWith(prefix(name))).map((a) => a.slice(prefix(name).length));
  const flag = (name) => flags(name)[0];
  return {
    args,
    flag,
    flags,
    has: (name) => args.includes(`--${name}`),
    // Relative to the caller's working directory.
    path: (name) => flag(name) && resolve(flag(name)),
    positional: args.filter((a) => !a.startsWith('--')),
  };
}

// Imported here so that tools with no browser do not need Playwright.
export async function launchChromium({ headed = true, webGpu = true, args = [] } = {}) {
  const { chromium } = await import('playwright');
  return chromium.launch({
    headless: !headed,
    args: [...(webGpu ? ['--enable-unsafe-webgpu'] : []), ...args],
  });
}
