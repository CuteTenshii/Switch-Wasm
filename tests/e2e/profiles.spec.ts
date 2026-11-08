import type { Page } from '@playwright/test';
import { crc32, deflateSync } from 'node:zlib';
import { expect, openPage, reloadPage, rowNamed, test } from './page';

// A PNG whose column `x` is `rgb(x)`.
function png(width: number, height: number, rgb: (x: number) => [number, number, number]): Buffer {
  const chunk = (type: string, data: Buffer) => {
    const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
    const out = Buffer.alloc(body.length + 8);
    out.writeUInt32BE(data.length, 0);
    body.copy(out, 4);
    out.writeUInt32BE(crc32(body), body.length + 4);
    return out;
  };
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width, 0);
  header.writeUInt32BE(height, 4);
  header.set([8, 2, 0, 0, 0], 8);
  const line = Buffer.from([0, ...Array.from({ length: width }, (_, x) => rgb(x)).flat()]);
  const pixels = Buffer.concat(Array.from({ length: height }, () => line));
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', header),
    chunk('IDAT', deflateSync(pixels)),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

const row = (page: Page, name: string) => rowNamed(page, '.profile-row', '.profile-name', name);

async function addProfile(page: Page, name: string): Promise<void> {
  // Typed key by key, so game keyboard controls must not swallow the letters.
  await page.locator('#profile-add-name').pressSequentially(name);
  await page.locator('#profile-add-name').press('Enter');
  await expect(row(page, name)).toBeVisible();
}

async function openProfiles(page: Page): Promise<void> {
  await page.locator('#btn-profile').click();
  await expect(page.locator('#profiles')).toBeVisible();
}

async function uidOf(page: Page, name: string): Promise<string> {
  return (await row(page, name).locator('.profile-uid').textContent()) ?? '';
}

// Opens the UID form of `name` unless it is open, and submits `uid`.
async function submitUid(page: Page, name: string, uid: string): Promise<void> {
  const input = page.locator('.profile-uid-form input');
  if (!await input.isVisible()) await row(page, name).getByRole('button', { name: 'Edit UID' }).click();
  await input.fill(uid);
  await input.press('Enter');
}

// Every key in the NAND's saves store, after putting `seed` there.
async function saveKeys(page: Page, seed: string[] = []): Promise<string[]> {
  return page.evaluate(async (keys) => {
    const db = await new Promise<IDBDatabase>((resolve, reject) => {
      const req = indexedDB.open('switch-wasm-nand');
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error ?? new Error('open failed'));
    });
    const tx = db.transaction('saves', 'readwrite');
    const store = tx.objectStore('saves');
    for (const key of keys) store.put({ kind: 'file', data: new Uint8Array([1, 2, 3]) }, key);
    const all = store.getAllKeys();
    await new Promise((resolve) => {
      tx.oncomplete = resolve;
    });
    db.close();
    return (all.result as string[]).sort();
  }, seed);
}

test.beforeEach(async ({ page }) => {
  await openPage(page);
  await openProfiles(page);
});

test('a new console has one profile, playing', async ({ page }) => {
  await expect(page.locator('.profile-row')).toHaveCount(1);
  await expect(row(page, 'Player')).toContainText('Playing');
  await expect(page.locator('#btn-profile')).toContainText('Player');
});

test('profiles can be added, chosen and renamed, and survive a reload', async ({ page }) => {
  await addProfile(page, 'Jessie Axe');
  await row(page, 'Jessie Axe').getByRole('button', { name: 'Play as' }).click();
  await expect(row(page, 'Jessie Axe')).toContainText('Playing');
  await expect(page.locator('#btn-profile')).toContainText('Jessie Axe');

  await row(page, 'Player').getByRole('button', { name: 'Rename' }).click();
  await page.locator('.profile-rename input').fill('Sam');
  await page.locator('.profile-rename input').press('Enter');
  await expect(row(page, 'Sam')).toBeVisible();

  await reloadPage(page);
  await expect(page.locator('#btn-profile')).toContainText('Jessie Axe');
  await openProfiles(page);
  await expect(page.locator('.profile-name')).toHaveText(['Sam', 'Jessie Axe']);
});

test('deleting asks first, and the last profile cannot be deleted', async ({ page }) => {
  await expect(row(page, 'Player').getByRole('button', { name: 'Delete' })).toHaveCount(0);
  await addProfile(page, 'Quinn');
  await row(page, 'Quinn').getByRole('button', { name: 'Delete' }).click();
  await expect(page.locator('.profile-confirm')).toHaveText('Delete Quinn and their saves?');
  await page.getByRole('button', { name: 'Keep' }).click();
  await expect(row(page, 'Quinn')).toBeVisible();

  await row(page, 'Quinn').getByRole('button', { name: 'Delete' }).click();
  await page.locator('.profile-row.is-confirming').getByRole('button', { name: 'Delete' }).click();
  await expect(page.locator('.profile-row')).toHaveCount(1);
});

test('a bad UID is refused with the reason, and Escape leaves it unchanged', async ({ page }) => {
  const old = await uidOf(page, 'Player');
  expect(old).toMatch(/^[0-9a-f]{32}$/);
  const error = page.locator('.profile-error');
  await submitUid(page, 'Player', 'not hex');
  await expect(error).toHaveText('A UID is 32 hex digits, 0-9 and a-f.');
  await submitUid(page, 'Player', '0'.repeat(32));
  await expect(error).toContainText('can\'t be all zeros');

  await addProfile(page, 'Kim');
  await submitUid(page, 'Player', await uidOf(page, 'Kim'));
  await expect(error).toHaveText('Kim already has this UID.');

  await page.locator('.profile-uid-form input').press('Escape');
  await expect(page.locator('.profile-uid-form')).toHaveCount(0);
  expect(await uidOf(page, 'Player')).toBe(old);
});

test('a new UID takes the profile\'s saves along and survives a reload', async ({ page }) => {
  const old = await uidOf(page, 'Player');
  const fresh = '0123456789abcdef0123456789abcdef';
  await saveKeys(page, [`0100000000001000@${old}/save.bin`, '0100000000002000/shared.bin']);

  await submitUid(page, 'Player', fresh.toUpperCase());
  await expect(row(page, 'Player').locator('.profile-uid')).toHaveText(fresh);
  expect(await saveKeys(page)).toEqual([`0100000000001000@${fresh}/save.bin`, '0100000000002000/shared.bin']);

  await reloadPage(page);
  await openProfiles(page);
  await expect(row(page, 'Player').locator('.profile-uid')).toHaveText(fresh);
  await expect(row(page, 'Player')).toContainText('Playing');
});

test('a UID change is refused when both UIDs hold a save of the same title', async ({ page }) => {
  const old = await uidOf(page, 'Player');
  const other = 'fedcba9876543210fedcba9876543210';
  const seeded = [`0100000000001000@${old}/a.bin`, `0100000000001000@${other}/b.bin`].sort();
  await saveKeys(page, seeded);
  await submitUid(page, 'Player', other);
  await expect(page.locator('.profile-error')).toContainText('already has saves for title 0100000000001000');
  await expect(page.locator('.profile-uid-form')).toBeVisible();
  expect(await saveKeys(page)).toEqual(seeded);
});

test('a console holds at most eight profiles', async ({ page }) => {
  for (const name of ['B', 'C', 'D', 'E', 'F', 'G', 'H']) await addProfile(page, name);
  await expect(page.locator('.profile-row')).toHaveCount(8);
  await expect(page.locator('#profile-add-submit')).toBeDisabled();
  await expect(page.locator('#profile-add-note')).toBeVisible();
});

test('a picture is cropped to a square JPEG', async ({ page }) => {
  // Wider than tall, with blue bands only a crop off centre would keep.
  const picture = png(300, 200, (x) => (x < 50 || x >= 250 ? [40, 60, 200] : [200, 60, 40]));
  await row(page, 'Player').locator('input[type=file]').setInputFiles({
    name: 'red.png', mimeType: 'image/png', buffer: picture,
  });
  const img = row(page, 'Player').locator('img.profile-avatar');
  await expect(img).toBeVisible();
  const size = await img.evaluate((el: HTMLImageElement) => [el.naturalWidth, el.naturalHeight]);
  expect(size).toEqual([256, 256]);
  const { magic, edges } = await img.evaluate(async (el: HTMLImageElement) => {
    const bytes = new Uint8Array(await (await fetch(el.src)).arrayBuffer());
    const canvas = document.createElement('canvas');
    canvas.width = el.naturalWidth;
    canvas.height = el.naturalHeight;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(el, 0, 0);
    const at = (x: number) => [...ctx.getImageData(x, 128, 1, 1).data.slice(0, 3)];
    return { magic: [...bytes.slice(0, 3)], edges: [at(2), at(253)] };
  });
  expect(magic).toEqual([0xff, 0xd8, 0xff]);
  for (const [r, , b] of edges) {
    expect(r).toBeGreaterThan(150);
    expect(b).toBeLessThan(100);
  }
  await expect(row(page, 'Player').getByRole('button', { name: 'Remove picture' })).toBeVisible();
});
