import { expect, test } from '@playwright/test';
import { openPage } from './page';

test('the core starts and names its build', async ({ page }) => {
  await openPage(page);
  await expect(page.locator('#wasm-ver')).toHaveText(/^core \d+\.\d+\.\d+/);
  await expect(page.locator('#state')).toHaveText('idle');
});
