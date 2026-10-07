import { aarch64Elf, bootElf, expect, openPage, test } from './page';

const SVC_EXIT_PROCESS = 0xd40000e1;

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

  await bootElf(page, aarch64Elf([SVC_EXIT_PROCESS]));
  await expect(page.locator('#state')).toHaveText('halted');
  await expect(display.locator('#display-debug-badge')).not.toHaveText('no samples');
  await expect(display.locator('#display-debug-data')).toBeVisible();

  await display.getByRole('button', { name: 'Reset samples' }).click();
  await expect(display.locator('#display-debug-badge')).toHaveText('no samples');
  await expect(display.locator('#display-debug-data')).toBeHidden();

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
  // movz x1, #0xfff0, lsl #16; ldr x0, [x1]: a read above the guest address space.
  await bootElf(page, aarch64Elf([0xd2bffe01, 0xf9400020]));

  await expect(page.locator('#state')).toHaveText('fault');
  await expect(crash).toBeVisible();
  await expect(crash.getByRole('heading')).toHaveText('Guest crashed');
  await expect(crash.locator('#crash-message')).toHaveText(
    'CPU: read from unmapped address 0xfff00000',
  );
  await expect(crash.getByRole('button', { name: 'Save crash report' })).toBeVisible();

  // The fixture fails on console errors, so clear the expected fault report.
  await expect(page.locator('#console .err', {
    hasText: 'Fault: CPU: read from unmapped address 0xfff00000',
  })).toHaveCount(1);

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
  await page.evaluate(() => document.getElementById('btn-clear-console')!.click());
});

test('trace channels record and show only at the debug level', async ({ page }) => {
  await openPage(page);
  // svc #0x29 (GetInfo), then svc #0: the host's halt trap.
  const program = aarch64Elf([0xd4000521, 0xd4000001]);
  const svcLine = page.locator('#console div', { hasText: '#0x29' });

  await expect(page.locator('#log-level')).toHaveValue('log');
  await bootElf(page, program);
  await expect(page.locator('#state')).toHaveText('halted');
  await page.locator('#btn-panel').click();
  await page.locator('#log-level').selectOption('debug');
  await expect(svcLine).toHaveCount(0);

  await bootElf(page, program);
  await expect(page.locator('#state')).toHaveText('halted');
  await expect(svcLine).toHaveCount(1);
  await page.locator('#log-level').selectOption('log');
  await expect(svcLine).toHaveCount(0);
});
