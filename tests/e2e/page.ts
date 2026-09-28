import { expect, type Page } from '@playwright/test';

/** Open the page and wait for the core to come up, failing the test on any
 *  error the page throws or logs. */
export async function openPage(page: Page): Promise<void> {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });
  await page.goto('./');
  await expect(page.locator('#loading')).toHaveClass(/hidden/, { timeout: 30_000 });
  expect(errors).toEqual([]);
}
