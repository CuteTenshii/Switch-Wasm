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
  await expect(page.getByText('Missing services', { exact: true })).toBeHidden();
  await page.getByRole('button', { name: 'System' }).click();
  await expect(page.getByText('Missing services', { exact: true })).toBeVisible();
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

test('the guest crash screen exposes recovery paths', async ({ page }) => {
  await openPage(page);

  const crash = page.locator('#crash');
  await expect(crash).toBeHidden();
  await crash.evaluate((element) => {
    element.querySelector('#crash-message')!.textContent = 'CPU: unmapped read at 0x1234';
    (element as HTMLElement).hidden = false;
  });

  await expect(crash.getByRole('heading')).toHaveText('Guest crashed');
  await expect(crash.locator('#crash-message')).toHaveText('CPU: unmapped read at 0x1234');
  await expect(crash.getByRole('button', { name: 'Save crash report' })).toBeVisible();

  await crash.getByRole('button', { name: 'Open console' }).click();
  await expect(page.locator('body')).toHaveClass(/panel-open/);
  await expect(page.getByRole('tab', { name: 'Console' })).toHaveAttribute('aria-selected', 'true');

  await page.setViewportSize({ width: 320, height: 700 });
  const actionHeights = await crash.getByRole('button').evaluateAll(
    (buttons) => buttons.map((button) => button.getBoundingClientRect().height),
  );
  expect(actionHeights.every((height) => height >= 44)).toBe(true);
  const overflows = await crash.evaluate((element) => element.scrollWidth > element.clientWidth);
  expect(overflows).toBe(false);
});
