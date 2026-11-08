import { aarch64Elf, ASM, bootElf, expect, expectState, openPage, test } from './page';

test.beforeEach(async ({ page }) => {
  await openPage(page);
});

test('the idle screen\'s Open button boots what it picks', async ({ page }) => {
  const chooser = page.waitForEvent('filechooser');
  await page.locator('.overlay-actions label').click();
  await (await chooser).setFiles({
    name: 'test.elf', mimeType: 'application/octet-stream', buffer: aarch64Elf([ASM.halt]),
  });
  await expectState(page, 'halted');
});

test('the run controls show only while a program is loaded', async ({ page }) => {
  const controls = page.locator('#btn-run, #btn-step, #btn-stop');
  for (const control of await controls.all()) await expect(control).toBeHidden();

  await bootElf(page, aarch64Elf([ASM.spin]), 'running');
  for (const control of await controls.all()) await expect(control).toBeVisible();
  await expect(page.locator('#run-label')).toHaveText('Pause');
});

test('Stop ends a running program without a loading screen, and the next boot runs', async ({ page }) => {
  const stop = page.locator('#btn-stop');
  await bootElf(page, aarch64Elf([ASM.spin]), 'running');
  await stop.click();
  await expectState(page, 'idle');
  await expect(page.locator('#loading')).toBeHidden();
  await expect(stop).toBeHidden();

  await bootElf(page, aarch64Elf([ASM.halt]), 'halted');
  await expect(page.locator('#console .error')).toHaveCount(0);
});

test('a guest crash says why and offers the log, the console and Stop', async ({ page }) => {
  const crash = page.locator('#crash');
  await expect(crash).toBeHidden();
  await bootElf(page, aarch64Elf([...ASM.unmappedRead]), 'fault');

  await expect(crash.getByRole('heading')).toHaveText('Guest crashed');
  await expect(crash.locator('#crash-message')).toHaveText('CPU: read from unmapped address 0xfff00000');
  await expect(page.locator('#console .error', {
    hasText: 'Fault: CPU: read from unmapped address 0xfff00000',
  })).toHaveCount(1);

  const download = page.waitForEvent('download');
  await crash.getByRole('button', { name: 'Save log' }).click();
  expect((await download).suggestedFilename()).toMatch(/\.txt$/);

  await crash.getByRole('button', { name: 'Open console' }).click();
  await expect(page.getByRole('tab', { name: 'Console' })).toHaveAttribute('aria-selected', 'true');
  // The fixture fails on logged errors, and this one was expected.
  await page.locator('#btn-clear-console').click();

  await crash.getByRole('button', { name: 'Stop' }).click();
  await expectState(page, 'idle');
  await expect(crash).toBeHidden();
});

test.describe('on iOS', () => {
  test.use({ userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1' });

  test('file pickers take any file, since iOS greys out unknown extensions', async ({ page }) => {
    await expect(page.locator('input[type=file][accept^="."]')).toHaveCount(0);
  });
});
