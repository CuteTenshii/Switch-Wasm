import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, openPage, openPanel, test } from './page';

const NEEDS_KEYS = 'Load prod.keys in the NAND tab to install firmware: each NCA is identified'
  + ' from its encrypted header before it is stored.';

// Each firmware NCA is identified from its encrypted header before it is stored, so without
// prod.keys the install buttons are unavailable and say why; loading them makes them available.
test('the firmware install buttons wait for prod.keys', async ({ page }) => {
  await openPage(page);
  await openPanel(page, 'NAND');
  const nand = page.locator('[data-panel="nand"]');
  const buttons = ['Install from folder…', 'Install from .zip…'].map((name) => nand.getByText(name));
  let pickers = 0;
  page.on('filechooser', () => pickers++);
  for (const button of buttons) {
    await expect(button).toHaveAttribute('aria-disabled', 'true');
    await expect(button).toHaveAttribute('title', NEEDS_KEYS);
    // Forced: Playwright itself will not click what says it is disabled.
    await button.click({ force: true });
  }
  await expect(page.locator('#firmware-status')).toHaveText(NEEDS_KEYS);
  expect(pickers).toBe(0);

  await page.locator('#prod-keys').setInputFiles({
    name: 'prod.keys',
    mimeType: 'text/plain',
    buffer: Buffer.from('header_key = ' + '00'.repeat(32) + '\n'),
  });
  for (const button of buttons) {
    await expect(button).toHaveAttribute('aria-disabled', 'false');
    await expect(button).toHaveAttribute('title', '');
  }
  await expect(page.locator('#firmware-status')).toHaveText('');
});

// A folder install takes only the NCAs directly in the folder picked, as Eden's does.
test('a firmware folder whose NCAs are in a sub-folder installs none', async ({ page }) => {
  await openPage(page);
  await openPanel(page, 'NAND');
  await page.locator('#prod-keys').setInputFiles({
    name: 'prod.keys',
    mimeType: 'text/plain',
    buffer: Buffer.from('header_key = ' + '00'.repeat(32) + '\n'),
  });
  const picked = join(mkdtempSync(join(tmpdir(), 'firmware-')), 'Firmware');
  mkdirSync(join(picked, '21.2.0'), { recursive: true });
  writeFileSync(join(picked, '21.2.0', 'a.nca'), 'not read');
  await page.locator('#firmware-dir').setInputFiles(picked);
  await expect(page.locator('#firmware-status')).toHaveText('Nothing in Firmware is an NCA.');
});
