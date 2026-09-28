import { expect, test } from '@playwright/test';
import { openPage } from './page';

test('the core starts and names its build', async ({ page }) => {
  await openPage(page);
  await expect(page.locator('#wasm-ver')).toHaveText(/^core \d+\.\d+\.\d+/);
  await expect(page.locator('#state')).toHaveText('idle');
});

test('the display debugger starts empty and can reset its samples', async ({ page }) => {
  await openPage(page);
  await page.locator('#btn-panel').click();
  await page.getByRole('tab', { name: 'Debug' }).click();

  await expect(page.getByText('Tracing', { exact: true })).toBeVisible();
  await expect(page.getByText('Diagnostic channels', { exact: true })).toBeHidden();
  await page.getByRole('button', { name: 'System' }).click();
  await expect(page.getByText('Diagnostic channels', { exact: true })).toBeVisible();
  await expect(page.getByText('Tracing', { exact: true })).toBeHidden();
  await page.getByRole('button', { name: 'Graphics' }).click();

  const display = page.locator('#display-debug-section');
  await expect(display).toBeVisible();
  await expect(display.locator('#display-debug-badge')).toHaveText('no samples');
  await expect(display.locator('#display-debug-empty')).toHaveText(
    'Run a title to collect display samples.',
  );
  await expect(display.locator('#display-debug-data')).toBeHidden();

  await display.getByRole('button', { name: 'Reset samples' }).click();
  await expect(display.locator('#display-debug-badge')).toHaveText('no samples');

  await page.setViewportSize({ width: 320, height: 700 });
  const categoryHeight = await page.getByRole('button', { name: 'Graphics' })
    .evaluate((button) => button.getBoundingClientRect().height);
  expect(categoryHeight).toBeGreaterThanOrEqual(44);
  const panelOverflows = await page.locator('#panel').evaluate(
    (panel) => panel.scrollWidth > panel.clientWidth,
  );
  expect(panelOverflows).toBe(false);
});
