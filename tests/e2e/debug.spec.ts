import { aarch64Elf, ASM, bootElf, expect, expectState, openPage, openPanel, test } from './page';

test.beforeEach(async ({ page }) => {
  await openPage(page);
  await openPanel(page, 'Debug');
});

test('the display debugger starts empty, fills from a run and can reset its samples', async ({ page }) => {
  const display = page.locator('#display-debug-section');
  const reset = display.getByRole('button', { name: 'Reset samples' });
  await expect(page.getByText('Tracing', { exact: true })).toBeVisible();
  await expect(display).toBeHidden();
  await page.getByRole('button', { name: 'Graphics' }).click();
  await expect(display).toBeVisible();
  await expect(page.getByText('Tracing', { exact: true })).toBeHidden();
  await expect(display.locator('#display-debug-data')).toBeHidden();
  await expect(reset).toBeDisabled();

  await bootElf(page, aarch64Elf([ASM.exitProcess]), 'halted');
  await expect(display.locator('#display-debug-data')).toBeVisible();
  // It exited without presenting, so there is no frame to measure.
  await expect(display.locator('#display-frame-size')).toHaveText('-');

  await reset.click();
  await expect(display.locator('#display-debug-data')).toBeHidden();
  await expect(reset).toBeDisabled();
});

test('trace channels record and show only at the trace level', async ({ page }) => {
  const program = aarch64Elf([ASM.getInfo, ASM.halt]);
  const svcLine = page.locator('#console div', { hasText: '#0x29' });
  await openPanel(page, 'Console');
  await expect(page.locator('#log-level')).toHaveValue('log');

  await page.locator('#log-level').selectOption('debug');
  await bootElf(page, program, 'halted');
  await expect(svcLine).toHaveCount(0);

  await page.locator('#log-level').selectOption('trace');
  await bootElf(page, program, 'halted');
  await expect(svcLine).toHaveCount(1);
  await page.locator('#log-level').selectOption('log');
  await expect(svcLine).toHaveCount(0);
});

test('the memory section waits for a title, then shows where its RAM is', async ({ page }) => {
  await page.getByRole('button', { name: 'Memory' }).click();
  const ram = page.locator('#memory-debug-section');
  await expect(ram.locator('#memory-debug-data')).toBeHidden();
  // The heap and guest RAM are the core's, so they report before any title runs.
  await expect(page.locator('#heap-size')).toHaveText(/^\d+(\.\d+)? [KMG]iB of 4 GiB$/);
  await expect(page.locator('#ram')).toHaveText(/^\d+(\.\d+)? [KMG]?i?B \(\d+(\.\d+)? [KMG]iB\)$/);

  await bootElf(page, aarch64Elf([ASM.spin]), 'running');
  await expect(ram.locator('#memory-debug-data')).toBeVisible();
  await expect(ram.locator('#memory-backed')).not.toHaveText('0 B');
  // The program sits at 0x08000000, the start of the code region.
  const program = ram.locator('#memory-regions tr', { hasText: 'Program and its data' });
  await expect(program).toContainText('0x08000000');
  await expect(program.locator('td').first()).not.toHaveText('0 B');
  await expect(ram.locator('#memory-regions tr')).toHaveCount(7);

  await page.locator('#btn-stop').click();
  await expectState(page, 'idle');
  await expect(ram.locator('#memory-debug-data')).toBeHidden();
  await expect(page.locator('#heap-size')).not.toHaveText('-');
});

test('the translation section counts the blocks a program ran', async ({ page }) => {
  await bootElf(page, aarch64Elf([ASM.exitProcess]), 'halted');
  await expect(page.locator('#jit-translated')).toHaveText(/^[1-9][\d,.]*[MBT]?$/);
});

test('the threads section waits for a title, then ranks what it runs', async ({ page }) => {
  const threads = page.locator('#threads-debug-section');
  await expect(threads.locator('#threads-status')).toHaveText('Start a title to see its threads.');
  await expect(threads.locator('#threads-table')).toBeHidden();

  await bootElf(page, aarch64Elf([ASM.spin]), 'running');
  const main = threads.locator('#threads-rows tr').first();
  await expect(main.locator('th')).toContainText('0 main');
  await expect(main.locator('th')).toContainText('running, priority');
  await expect(main.locator('td').first()).toHaveText('100.0%');
  // The second reading covers a known span, so it has a rate.
  await expect(main.locator('td').nth(1)).toHaveText(/^[1-9][\d,.]*[MBT]?$/);
  await expect(page.locator('#threads-badge')).toHaveText(/instructions\/s$/);

  await page.locator('#btn-stop').click();
  await expectState(page, 'idle');
  await expect(threads.locator('#threads-status')).toHaveText('Start a title to see its threads.');
  await expect(threads.locator('#threads-table')).toBeHidden();
});
