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
  // The renderer's state shows in its own section, not the log.
  await expect(page.locator('#render-debug-badge')).toHaveText(/^(WebGPU|software|lost)$/);

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
