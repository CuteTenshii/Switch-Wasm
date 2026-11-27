import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, nspWithoutProgram, openPage, openPanel, reloadPage, test } from './page';

// A `control.nacp` with one American English title.
function nacp(name: string, publisher: string, version: string): Buffer {
  const data = Buffer.alloc(0x4000);
  data.write(name, 0);
  data.write(publisher, 0x200);
  data.write(version, 0x3060);
  return data;
}

// An NRO with no code and an asset section holding `icon` and `nacp`, as homebrew ships.
function nroWithAssets(icon: Buffer, control: Buffer): Buffer {
  const header = Buffer.alloc(0x50);
  header.write('NRO0', 0);
  header.writeUInt32LE(1, 4);
  header.writeUInt32LE(header.length, 8);
  const assets = Buffer.alloc(0x38);
  assets.write('ASET', 0);
  let offset = assets.length;
  [icon, control, Buffer.alloc(0)].forEach((part, i) => {
    assets.writeBigUInt64LE(BigInt(offset), 8 + i * 16);
    assets.writeBigUInt64LE(BigInt(part.length), 16 + i * 16);
    offset += part.length;
  });
  return Buffer.concat([header, assets, icon, control]);
}

const ICON = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0x6a, 0x70, 0x65, 0x67]);
const APP = nroWithAssets(ICON, nacp('Fixture App', 'Fixture Team', '1.2.3'));

test('a picked folder lists its games, a homebrew by its own name, and launches one', async ({ page }) => {
  const folder = join(mkdtempSync(join(tmpdir(), 'library-')), 'Games');
  mkdirSync(join(folder, 'Homebrew'), { recursive: true });
  mkdirSync(join(folder, '.Trash-1000'));
  writeFileSync(join(folder, 'Homebrew', 'app.nro'), APP);
  writeFileSync(join(folder, '.Trash-1000', 'old.nro'), APP);
  writeFileSync(join(folder, 'notes.txt'), 'not listed');
  writeFileSync(join(folder, 'game.nsp'), nspWithoutProgram());

  await openPage(page);
  await page.locator('#library-dir').setInputFiles(folder);
  const library = page.locator('#library');
  await expect(library.getByRole('heading')).toHaveText('Games');
  await expect(page.locator('#overlay h1')).toBeHidden();

  const rows = library.locator('.library-row');
  await expect(rows).toHaveCount(2);
  await expect(rows.nth(0).locator('.library-name')).toHaveText('Fixture App');
  await expect(rows.nth(0).locator('.library-detail')).toHaveText('Fixture Team · v1.2.3');
  await expect(rows.nth(0).locator('.library-format')).toHaveText('NRO');
  await expect(rows.nth(0).locator('img.library-icon')).toHaveCount(1);
  // Without prod.keys a container is known by its file name.
  await expect(rows.nth(1).locator('.library-name')).toHaveText('game');
  await expect(rows.nth(1).locator('.library-detail')).toHaveCount(0);

  const status = page.locator('#library-status');
  await expect(status).toContainText('Load prod.keys to see game names and icons');
  await status.getByRole('button', { name: 'Load prod.keys' }).click();
  await expect(page.getByRole('tab', { name: 'NAND' })).toHaveAttribute('aria-selected', 'true');

  await rows.nth(1).click();
  await expect(page.locator('#loading-phase'))
    .toHaveText('This game is encrypted. Load the prod.keys file from your Switch to play it.');
  await page.locator('#loading-dismiss').click();

  await library.getByRole('button', { name: 'Close' }).click();
  await expect(library).toBeHidden();
  await expect(page.locator('#overlay h1')).toBeVisible();

  // The fixture fails on logged errors, and the launch's was expected.
  await openPanel(page, 'Console');
  await page.locator('#btn-clear-console').click();
});

test('each icon keeps one URL while the rest of the folder is read', async ({ page }) => {
  const folder = join(mkdtempSync(join(tmpdir(), 'library-')), 'Games');
  mkdirSync(folder, { recursive: true });
  ['a', 'b', 'c'].forEach((name, i) => {
    const icon = Buffer.concat([ICON, Buffer.from([i])]);
    writeFileSync(join(folder, name + '.nro'), nroWithAssets(icon, nacp('App ' + name, 'Team', '1.0.0')));
  });

  await openPage(page);
  await page.evaluate(() => {
    const seen = new Set<string>();
    (window as unknown as { iconUrls: Set<string> }).iconUrls = seen;
    new MutationObserver(() => {
      for (const img of document.querySelectorAll<HTMLImageElement>('#library-list img.library-icon')) {
        seen.add(img.src);
      }
    }).observe(document.getElementById('library-list')!, { childList: true, subtree: true });
  });
  await page.locator('#library-dir').setInputFiles(folder);

  const icons = page.locator('#library .library-row img.library-icon');
  await expect(icons).toHaveCount(3);
  await expect(page.locator('#library-status')).not.toContainText('Reading game details');
  const urls = await page.evaluate(() => [...(window as unknown as { iconUrls: Set<string> }).iconUrls]);
  expect(urls).toHaveLength(3);

  const listed = await icons.nth(0).getAttribute('src');
  await page.locator('#library .library-row').nth(0).click();
  await expect(page.locator('#running-icon')).toHaveAttribute('src', listed!);
  await expect(icons.nth(0)).toHaveAttribute('src', listed!);

  // The fixture has no code, so it faults on its first instruction.
  await openPanel(page, 'Console');
  await page.locator('#btn-clear-console').click();
});

// The browser's private file system stands in for the folder picker: it hands back the same
// kind of handle, which is what the page keeps.
test('a folder picked through the browser comes back after a reload, until it is closed', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(window, 'showDirectoryPicker', {
      value: async () => (await navigator.storage.getDirectory()).getDirectoryHandle('Library', { create: true }),
    });
  });
  await openPage(page);
  await page.evaluate(async (bytes) => {
    const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle('Library', { create: true });
    const writable = await (await dir.getFileHandle('app.nro', { create: true })).createWritable();
    await writable.write(new Uint8Array(bytes));
    await writable.close();
  }, [...APP]);

  await page.getByRole('button', { name: 'Open a folder' }).click();
  const library = page.locator('#library');
  await expect(library.getByRole('heading')).toHaveText('Library');
  await expect(library.locator('.library-name')).toHaveText(['Fixture App']);

  await reloadPage(page);
  await expect(library.locator('.library-name')).toHaveText(['Fixture App']);

  await library.getByRole('button', { name: 'Close' }).click();
  await reloadPage(page);
  await expect(library).toBeHidden();
  await expect(page.getByRole('button', { name: 'Open a folder' })).toBeVisible();
});
