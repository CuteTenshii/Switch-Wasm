import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, nspWithoutProgram, openPage, openPanel, test } from './page';

const NEEDS_KEYS = 'Load prod.keys in the NAND tab to install firmware: each NCA is identified'
  + ' from its encrypted header before it is stored.';

const FAKE_PROD_KEYS = {
  name: 'prod.keys',
  mimeType: 'text/plain',
  buffer: Buffer.from('header_key = ' + '00'.repeat(32) + '\n'),
};

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

  await page.locator('#prod-keys').setInputFiles(FAKE_PROD_KEYS);
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
  await page.locator('#prod-keys').setInputFiles(FAKE_PROD_KEYS);
  const picked = join(mkdtempSync(join(tmpdir(), 'firmware-')), 'Firmware');
  mkdirSync(join(picked, '21.2.0'), { recursive: true });
  writeFileSync(join(picked, '21.2.0', 'a.nca'), 'not read');
  await page.locator('#firmware-dir').setInputFiles(picked);
  await expect(page.locator('#firmware-status')).toHaveText('Nothing in Firmware is an NCA.');
});

test('a launch without prod.keys says so, opens the panel on the keys, and reruns once they load', async ({ page }) => {
  await openPage(page);
  await page.locator('#nro-file').setInputFiles({
    name: 'game.nsp', mimeType: 'application/octet-stream', buffer: nspWithoutProgram(),
  });
  const loading = page.locator('#loading');
  await expect(loading.locator('#loading-phase'))
    .toHaveText('This game is encrypted. Load the prod.keys file from your Switch to play it.');
  // The screen explains the failure, so the side panel stays shut.
  await expect(page.locator('body')).not.toHaveClass(/panel-open/);

  await loading.getByRole('button', { name: 'Load prod.keys' }).click();
  await expect(page.getByRole('tab', { name: 'NAND' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.locator('#keys-section > summary')).toBeFocused();

  await page.locator('#prod-keys').setInputFiles(FAKE_PROD_KEYS);
  // The rerun with keys finds no Program NCA either, and the core says so.
  await expect(loading.locator('#loading-phase')).toHaveText(/^Launch failed: no Program NCA/);
  await expect(loading.locator('#loading-action')).toBeHidden();
  await expect(page.locator('#console .info', { hasText: 'Launching game.nsp' })).toHaveCount(2);

  // The fixture fails on logged errors, and these were expected.
  await openPanel(page, 'Console');
  await page.locator('#btn-clear-console').click();
});

test('a dismissed launch failure is not rerun when prod.keys loads', async ({ page }) => {
  await openPage(page);
  await page.locator('#nro-file').setInputFiles({
    name: 'game.nsp', mimeType: 'application/octet-stream', buffer: nspWithoutProgram(),
  });
  const loading = page.locator('#loading');
  await loading.getByRole('button', { name: 'Dismiss' }).click();
  await expect(loading).toHaveClass(/hidden/);

  await openPanel(page, 'NAND');
  await page.locator('#prod-keys').setInputFiles(FAKE_PROD_KEYS);
  await expect(page.locator('#keys-section .section-note')).toHaveText('1 loaded');
  await expect(loading).toHaveClass(/hidden/);
  await expect(page.locator('#console .info', { hasText: 'Launching game.nsp' })).toHaveCount(1);

  await openPanel(page, 'Console');
  await page.locator('#btn-clear-console').click();
});
