import type { Locator, Page } from '@playwright/test';
import { aarch64Elf, ASM, bootElf, expect, openPage, openPanel, test } from './page';

const PHONE = { width: 320, height: 700 };

async function heights(buttons: Locator): Promise<number[]> {
  return buttons.evaluateAll((all) => all.map((button) => button.getBoundingClientRect().height));
}

async function overflows(target: Locator): Promise<boolean> {
  return target.evaluate((element) => element.scrollWidth > element.clientWidth);
}

async function pageOverflows(page: Page): Promise<boolean> {
  return page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
}

test.beforeEach(async ({ page }) => {
  await openPage(page);
});

test('a narrow bar moves its secondary controls into the More menu and back', async ({ page }) => {
  const more = page.locator('#btn-more');
  const menu = page.locator('#more-menu');
  const dock = page.locator('#btn-dock');
  await expect(more).toBeHidden();
  await expect(page.locator('.topbar-actions #btn-dock')).toHaveCount(1);

  await page.setViewportSize({ width: 412, height: 800 });
  await expect(page.locator('#more-menu #btn-dock')).toHaveCount(1);
  expect(await pageOverflows(page)).toBe(false);
  await more.click();
  await expect(menu).toBeVisible();
  await dock.click();
  await expect(dock).toHaveAttribute('aria-pressed', 'true');
  await expect(menu).toBeHidden();

  await page.setViewportSize({ width: 1200, height: 800 });
  await expect(page.locator('.topbar-actions #btn-dock')).toHaveCount(1);
  await expect(more).toBeHidden();
});

test('the debug panel fits a phone and its category buttons are tappable', async ({ page }) => {
  await openPanel(page, 'Debug');
  await page.setViewportSize(PHONE);
  for (const height of await heights(page.locator('.debug-group'))) expect(height).toBeGreaterThanOrEqual(44);
  expect(await overflows(page.locator('#panel'))).toBe(false);
});

test('the crash screen fits a phone and its actions are tappable', async ({ page }) => {
  await page.setViewportSize(PHONE);
  await bootElf(page, aarch64Elf([...ASM.unmappedRead]), 'fault');
  const crash = page.locator('#crash');
  for (const height of await heights(crash.getByRole('button'))) expect(height).toBeGreaterThanOrEqual(44);
  expect(await overflows(crash)).toBe(false);
  // The fixture fails on logged errors, and the fault report was expected.
  await crash.getByRole('button', { name: 'Open console' }).click();
  await page.locator('#btn-clear-console').click();
});

test('the idle screen fits a phone', async ({ page }) => {
  await page.setViewportSize(PHONE);
  await expect(page.locator('.overlay h1')).toBeVisible();
  expect(await pageOverflows(page)).toBe(false);
});
