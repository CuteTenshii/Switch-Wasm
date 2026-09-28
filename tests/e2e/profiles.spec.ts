import { expect, test, type Page } from '@playwright/test';
import { crc32, deflateSync } from 'node:zlib';
import { openPage } from './page';

/** A PNG of one colour, as an image file a user would pick. */
function solidPng(width: number, height: number, rgb: [number, number, number]): Buffer {
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
  const line = Buffer.from([0, ...Array.from({ length: width }, () => rgb).flat()]);
  const pixels = Buffer.concat(Array.from({ length: height }, () => line));
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', header),
    chunk('IDAT', deflateSync(pixels)),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

/** The row of the profile named exactly `name`: a string `hasText` is a
 *  case-insensitive substring match, and "E" is in "Player". */
const row = (page: Page, name: string) =>
  page.locator('.profile-row').filter({
    has: page.locator('.profile-name').getByText(name, { exact: true }),
  });

async function addProfile(page: Page, name: string): Promise<void> {
  // Typed key by key: the game's keyboard controls must not swallow letters
  // like A, S, E, Q, Z and X, or Enter.
  await page.locator('#profile-add-name').pressSequentially(name);
  await page.locator('#profile-add-name').press('Enter');
  await expect(row(page, name)).toBeVisible();
}

test.beforeEach(async ({ page }) => {
  await openPage(page);
  await page.locator('#btn-profile').click();
  await expect(page.locator('#profiles')).toBeVisible();
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

  await page.reload();
  await expect(page.locator('#loading')).toHaveClass(/hidden/, { timeout: 30_000 });
  await expect(page.locator('#btn-profile')).toContainText('Jessie Axe');
  await page.locator('#btn-profile').click();
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

test('a console holds at most eight profiles', async ({ page }) => {
  for (const name of ['B', 'C', 'D', 'E', 'F', 'G', 'H']) await addProfile(page, name);
  await expect(page.locator('.profile-row')).toHaveCount(8);
  await expect(page.locator('#profile-add-submit')).toBeDisabled();
  await expect(page.locator('#profile-add-note')).toBeVisible();
});

test('a picture is cropped to a square JPEG', async ({ page }) => {
  // Wider than tall, so the crop has something to do.
  const png = solidPng(300, 200, [200, 60, 40]);
  await row(page, 'Player').locator('input[type=file]').setInputFiles({
    name: 'red.png', mimeType: 'image/png', buffer: png,
  });
  const img = row(page, 'Player').locator('img.profile-avatar');
  await expect(img).toBeVisible();
  const size = await img.evaluate((el: HTMLImageElement) => [el.naturalWidth, el.naturalHeight]);
  expect(size).toEqual([256, 256]);
  await expect(row(page, 'Player').getByRole('button', { name: 'Remove picture' })).toBeVisible();
});
