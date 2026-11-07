import { createHash } from 'node:crypto';
import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import type { Page } from '@playwright/test';
import { expect, openPage, test } from './page';

// A relay's first frame: HELLO, version 1, 64 streams.
const HELLO = Buffer.from([0x80, 0, 0, 0, 0, 1, 0x40, 0]);

// A WebSocket server that greets like the relay and says nothing more. The page's socket lives
// in a worker, which Playwright cannot intercept, so it has to be a real one.
async function relayStub(): Promise<{ server: Server; url: string }> {
  const server = createServer();
  server.on('upgrade', (request, socket) => {
    const key = request.headers['sec-websocket-key'] as string;
    const accept = createHash('sha1')
      .update(`${key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11`)
      .digest('base64');
    socket.write(
      `HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${accept}\r\n\r\n`,
    );
    // An unmasked binary frame.
    socket.write(Buffer.concat([Buffer.from([0x82, HELLO.length]), HELLO]));
    socket.on('error', () => undefined);
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  return { server, url: `ws://127.0.0.1:${(server.address() as AddressInfo).port}` };
}

async function openTab(page: Page): Promise<void> {
  await openPage(page);
  await page.locator('#btn-panel').click();
  await page.locator('.tab[data-tab="network"]').click();
  await expect(page.locator('#net-badge')).toHaveText('off');
}

test('turning the network on with no relay running says so, and it can be retried', async ({ page }) => {
  await openTab(page);
  // A port nothing listens on.
  await page.locator('#net-url').fill('ws://127.0.0.1:1');
  await page.locator('#btn-net').click();
  await expect(page.locator('#net-badge')).toHaveText('no relay');
  await expect(page.locator('#net-note')).toContainText('ws://127.0.0.1:1');
  await expect(page.locator('#btn-net')).toHaveText('Turn on');
  await expect(page.locator('#net-url')).toBeEnabled();
});

test('an address that is not a WebSocket one is refused before anything connects', async ({ page }) => {
  await openTab(page);
  await page.locator('#net-url').fill('http://127.0.0.1:18787');
  await page.locator('#btn-net').click();
  await expect(page.locator('#net-note')).toContainText('ws://');
  await expect(page.locator('#net-badge')).toHaveText('off');
});

test('a relay that answers turns the network on, and turning it off puts it back', async ({ page }) => {
  const { server, url } = await relayStub();
  await openTab(page);
  await page.locator('#net-url').fill(url);
  await page.locator('#btn-net').click();
  await expect(page.locator('#net-badge')).toHaveText('on');
  await expect(page.locator('#btn-net')).toHaveText('Turn off');
  await expect(page.locator('#net-url')).toBeDisabled();
  await expect(page.locator('#net-empty')).toBeVisible();

  await page.locator('#btn-net').click();
  await expect(page.locator('#net-badge')).toHaveText('off');
  await expect(page.locator('#btn-net')).toHaveText('Turn on');
  server.closeAllConnections();
  server.close();
});

test('the relay address is remembered', async ({ page }) => {
  await openTab(page);
  await page.locator('#net-url').fill('ws://127.0.0.1:1');
  await page.locator('#btn-net').click();
  await expect(page.locator('#net-badge')).toHaveText('no relay');
  await page.reload();
  await openPage(page);
  await page.locator('#btn-panel').click();
  await page.locator('.tab[data-tab="network"]').click();
  await expect(page.locator('#net-url')).toHaveValue('ws://127.0.0.1:1');
});
