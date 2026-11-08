import { test as base, expect, type Locator, type Page } from '@playwright/test';

// Fails any test during which the page threw or logged a console error.
export const test = base.extend<{ pageErrors: string[] }>({
  pageErrors: [async ({ page }, use) => {
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('console', (m) => {
      if (m.type() === 'error') errors.push(m.text());
    });
    await use(errors);
    const logged = await page.locator('#console .error').allTextContents();
    expect([...errors, ...logged]).toEqual([]);
  }, { auto: true }],
});

export { expect };

export type EmuState = 'idle' | 'running' | 'paused' | 'halted' | 'fault';

async function coreUp(page: Page): Promise<void> {
  await expect(page.locator('#loading')).toHaveClass(/hidden/, { timeout: 30_000 });
}

export async function openPage(page: Page): Promise<void> {
  await page.goto('./');
  await coreUp(page);
}

export async function reloadPage(page: Page): Promise<void> {
  await page.reload();
  await coreUp(page);
}

export async function expectState(page: Page, state: EmuState): Promise<void> {
  await expect(page.locator('body')).toHaveAttribute('data-state', state);
}

// Opens the side panel on `tab`, by its accessible name.
export async function openPanel(page: Page, tab: string): Promise<void> {
  if (!await page.locator('body').evaluate((body) => body.classList.contains('panel-open'))) {
    await page.locator('#btn-panel').click();
  }
  await page.getByRole('tab', { name: tab, exact: true }).click();
}

// The item of `list` whose `name` element reads exactly `text` (`hasText` is a substring match).
export function rowNamed(page: Page, list: string, name: string, text: string): Locator {
  return page.locator(list).filter({ has: page.locator(name).getByText(text, { exact: true }) });
}

// A static AArch64 ELF whose one segment, at 0x0800_0000, runs `code` from its entry.
export function aarch64Elf(code: number[]): Buffer {
  const vaddr = 0x0800_0000n;
  const codeAt = 0x78;
  const out = Buffer.alloc(codeAt + code.length * 4);
  out.set([0x7f, 0x45, 0x4c, 0x46, 2, 1, 1], 0);
  out.writeUInt16LE(2, 16); // ET_EXEC
  out.writeUInt16LE(183, 18); // EM_AARCH64
  out.writeUInt32LE(1, 20);
  out.writeBigUInt64LE(vaddr + BigInt(codeAt), 24);
  out.writeBigUInt64LE(0x40n, 32);
  out.writeUInt16LE(0x40, 52);
  out.writeUInt16LE(56, 54);
  out.writeUInt16LE(1, 56);
  out.writeUInt32LE(1, 64); // PT_LOAD
  out.writeUInt32LE(5, 68); // R+X
  out.writeBigUInt64LE(vaddr, 80);
  out.writeBigUInt64LE(vaddr, 88);
  out.writeBigUInt64LE(BigInt(out.length), 96);
  out.writeBigUInt64LE(BigInt(out.length), 104);
  out.writeBigUInt64LE(0x1000n, 112);
  code.forEach((word, i) => out.writeUInt32LE(word, codeAt + i * 4));
  return out;
}

export async function bootElf(page: Page, elf: Buffer, until?: EmuState): Promise<void> {
  await page.locator('#nro-file').setInputFiles({
    name: 'test.elf', mimeType: 'application/octet-stream', buffer: elf,
  });
  if (until) await expectState(page, until);
}

// Instruction words for the test programs.
export const ASM = {
  // b .
  spin: 0x14000000,
  // svc #0: the host's halt trap.
  halt: 0xd4000001,
  // svc #0x7 (ExitProcess)
  exitProcess: 0xd40000e1,
  // svc #0x29 (GetInfo)
  getInfo: 0xd4000521,
  // movz x1, #0xfff0, lsl #16; ldr x0, [x1]: a read above the guest address space.
  unmappedRead: [0xd2bffe01, 0xf9400020],
} as const;
