import type { CDPSession, Page } from '@playwright/test';
import { aarch64Elf, bootElf, expect, openPage, test } from './page';

test.use({ viewport: { width: 412, height: 860 }, isMobile: true, hasTouch: true });

async function centre(page: Page, selector: string): Promise<{ x: number; y: number }> {
  const box = (await page.locator(selector).boundingBox())!;
  return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
}

// One finger, through CDP: Playwright's touchscreen only taps.
function finger(cdp: CDPSession) {
  const touch = (type: 'touchStart' | 'touchMove' | 'touchEnd', x = 0, y = 0) => cdp.send('Input.dispatchTouchEvent', {
    type, touchPoints: type === 'touchEnd' ? [] : [{ x, y }],
  });
  return {
    down: (p: { x: number; y: number }) => touch('touchStart', p.x, p.y),
    move: (p: { x: number; y: number }) => touch('touchMove', p.x, p.y),
    up: () => touch('touchEnd'),
  };
}

test('the on-screen controller shows while a program runs, and a finger slides between buttons', async ({ page }) => {
  await openPage(page);
  const pad = page.locator('#touchpad');
  await expect(pad).toBeHidden();
  await bootElf(page, aarch64Elf([0x14000000]));
  await expect(pad).toBeVisible();
  // The program never presents, so the first-frame wait has to be dismissed.
  await page.locator('#loading-dismiss').click();
  await expect(page.locator('#loading')).toBeHidden();
  const touch = finger(await page.context().newCDPSession(page));

  const a = page.locator('[data-button="A"]');
  const b = page.locator('[data-button="B"]');
  await touch.down(await centre(page, '[data-button="A"]'));
  await expect(a).toHaveClass(/is-down/);
  await expect(page.locator('#input-state')).toHaveText('touch controls');
  await touch.move(await centre(page, '[data-button="B"]'));
  await expect(b).toHaveClass(/is-down/);
  await expect(a).not.toHaveClass(/is-down/);
  await touch.up();
  await expect(page.locator('.tp-btn.is-down')).toHaveCount(0);

  const knob = page.locator('[data-stick="left"] .tp-knob');
  const stick = await centre(page, '[data-stick="left"]');
  await touch.down(stick);
  await touch.move({ x: stick.x + 200, y: stick.y });
  // Pushed to the base's edge: half its width less half the knob's.
  await expect(knob).toHaveAttribute('style', /translate\(28px, 0px\)/);
  await touch.up();
  await expect(knob).not.toHaveAttribute('style', /translate\(\d/);

  await page.locator('#btn-stop').click();
  await expect(pad).toBeHidden();
});

test('a landscape phone hides the bars while a program runs, and the chevron brings them back', async ({ page }) => {
  await page.setViewportSize({ width: 860, height: 412 });
  await openPage(page);
  const bar = page.locator('.topbar');
  const toggle = page.locator('#btn-chrome');
  await expect(bar).toBeVisible();
  await expect(toggle).toBeHidden();
  await bootElf(page, aarch64Elf([0x14000000]));
  await expect(bar).toBeHidden();
  await expect(page.locator('.statusbar')).toBeHidden();
  await page.locator('#loading-dismiss').click();
  await toggle.click();
  await expect(bar).toBeVisible();
  await expect(toggle).toHaveAttribute('aria-expanded', 'true');
  await toggle.click();
  await expect(bar).toBeHidden();
});
