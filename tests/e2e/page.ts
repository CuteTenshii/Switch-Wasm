import { test as base, expect, type Page } from '@playwright/test';

/** Playwright's `test`, failing any test the page threw or logged an error
 *  during, at whatever point it happened. */
export const test = base.extend<{ pageErrors: string[] }>({
  pageErrors: [async ({ page }, use) => {
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('console', (m) => {
      if (m.type() === 'error') errors.push(m.text());
    });
    await use(errors);
    expect(errors).toEqual([]);
  }, { auto: true }],
});

export { expect };

/** Open the page and wait for the core to come up. */
export async function openPage(page: Page): Promise<void> {
  await page.goto('./');
  await expect(page.locator('#loading')).toHaveClass(/hidden/, { timeout: 30_000 });
}

/** A static AArch64 ELF whose one segment, at 0x0800_0000, runs `code` from
 *  its entry. */
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

/** Boot `elf` through the page's own file picker. */
export async function bootElf(page: Page, elf: Buffer): Promise<void> {
  await page.locator('#nro-file').setInputFiles({
    name: 'test.elf', mimeType: 'application/octet-stream', buffer: elf,
  });
}
