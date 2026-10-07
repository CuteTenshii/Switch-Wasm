import type { Page } from '@playwright/test';
import { expect, openPage, test } from './page';

// The row whose name is exactly `name` (folders carry a trailing slash).
const row = (page: Page, name: string) =>
  page.locator('.files-row').filter({
    has: page.locator('.files-name').getByText(name, { exact: true }),
  });

async function openStorage(page: Page): Promise<void> {
  await page.locator('#btn-panel').click();
  await page.locator('.panel .tab[data-tab="files"]').click();
  await page.locator('#btn-files').click();
  await expect(page.locator('#files')).toBeVisible();
}

test.beforeEach(async ({ page }) => {
  await openPage(page);
  await openStorage(page);
});

test('an empty card, no saves and no system titles say so', async ({ page }) => {
  await expect(page.locator('.files-empty')).toContainText('Empty');
  await page.getByRole('tab', { name: 'Saves' }).click();
  await expect(page.locator('.files-empty')).toContainText('No save data yet');
  await page.getByRole('tab', { name: 'System titles' }).click();
  await expect(page.locator('.files-empty')).toContainText('Nothing installed');
  await expect(page.locator('#files-install-label')).toBeVisible();
});

test('folders and files can be made, renamed, downloaded and deleted, and survive a reload', async ({ page }) => {
  await page.locator('#btn-files-mkdir').click();
  await page.locator('.files-name-form input').fill('game');
  await page.locator('.files-name-form input').press('Enter');
  await row(page, 'game/').locator('.files-main').click();
  await expect(page.locator('.files-crumb.is-here')).toHaveText('game');

  await page.locator('#files-upload').setInputFiles({
    name: 'main.lua', mimeType: 'text/plain', buffer: Buffer.from('print("hi")\n'),
  });
  await expect(row(page, 'main.lua')).toContainText('12 B');

  await row(page, 'main.lua').getByRole('button', { name: 'Rename' }).click();
  await page.locator('.files-name-form input').fill('conf.lua');
  await page.locator('.files-name-form input').press('Enter');
  await expect(row(page, 'conf.lua')).toBeVisible();
  await expect(row(page, 'main.lua')).toHaveCount(0);

  const download = page.waitForEvent('download');
  await row(page, 'conf.lua').getByRole('button', { name: 'Download' }).click();
  expect((await download).suggestedFilename()).toBe('conf.lua');

  await page.reload();
  await expect(page.locator('#loading')).toHaveClass(/hidden/, { timeout: 30_000 });
  await openStorage(page);
  await row(page, 'game/').locator('.files-main').click();
  await expect(row(page, 'conf.lua')).toBeVisible();

  const zip = page.waitForEvent('download');
  await page.locator('#btn-files-export').click();
  expect((await zip).suggestedFilename()).toBe('game.zip');

  await page.locator('.files-crumb', { hasText: 'sdmc:' }).click();
  await row(page, 'game/').getByRole('button', { name: 'Delete' }).click();
  await row(page, 'game/').getByRole('button', { name: 'Delete' }).click();
  await expect(page.locator('.files-empty')).toContainText('Empty');
});

test('a name with a slash is refused', async ({ page }) => {
  await page.locator('#btn-files-mkdir').click();
  await page.locator('.files-name-form input').fill('a/b');
  await page.locator('.files-name-form input').press('Enter');
  await expect(page.locator('#files-status')).toHaveClass(/is-error/);
  await page.locator('.files-name-form input').press('Escape');
  await expect(page.locator('#files')).toBeVisible();
});

test('a text file opens in an editor that saves, and a binary one as hex', async ({ page }) => {
  await page.locator('#files-upload').setInputFiles([
    { name: 'notes.txt', mimeType: 'text/plain', buffer: Buffer.from('first line\n') },
    { name: 'blob.bin', mimeType: 'application/octet-stream', buffer: Buffer.from([0, 1, 2, 0xff]) },
  ]);
  await row(page, 'notes.txt').locator('.files-main').click();
  const editor = page.locator('.files-editor');
  await expect(editor).toHaveValue('first line\n');
  await editor.fill('second line\n');
  await page.locator('#files-view-tools').getByRole('button', { name: 'Save' }).click();
  await expect(page.locator('#files-status')).toContainText('Saved notes.txt');
  await page.locator('#files-view-tools').getByRole('button', { name: 'Close' }).click();

  await row(page, 'notes.txt').locator('.files-main').click();
  await expect(page.locator('.files-editor')).toHaveValue('second line\n');
  await page.locator('#files-view-tools').getByRole('button', { name: 'Close' }).click();

  await row(page, 'blob.bin').locator('.files-main').click();
  await expect(page.locator('.files-hex')).toContainText('00 01 02 ff');
});

test('closing with unsaved edits asks first', async ({ page }) => {
  await page.locator('#files-upload').setInputFiles({
    name: 'a.txt', mimeType: 'text/plain', buffer: Buffer.from('x'),
  });
  await row(page, 'a.txt').locator('.files-main').click();
  await page.locator('.files-editor').fill('changed');
  await page.keyboard.press('Escape');
  await expect(page.locator('#files')).toBeVisible();
  await expect(page.locator('#files-view-tools')).toContainText('Discard your edits?');
  await page.getByRole('button', { name: 'Keep editing' }).click();
  await expect(page.locator('.files-editor')).toHaveValue('changed');
});
