import { createHash } from 'node:crypto';
import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import type { Page } from '@playwright/test';
import { expect, openPage, openPanel, reloadPage, test } from './page';

// A relay's first frame: HELLO, version 3, 64 streams.
const HELLO = Buffer.from([0x80, 0, 0, 0, 0, 3, 0x40, 0]);

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

test.beforeEach(async ({ page }) => {
  await openPage(page);
  await openPanel(page, 'Network');
  await expect(page.locator('#net-badge')).toHaveText('off');
});

test('turning the network on with no relay running says so, and it can be retried', async ({ page }) => {
  // A port nothing listens on.
  await page.locator('#net-url').fill('ws://127.0.0.1:1');
  await page.locator('#btn-net').click();
  await expect(page.locator('#net-badge')).toHaveText('no relay');
  await expect(page.locator('#net-note')).toContainText('ws://127.0.0.1:1');
  await expect(page.locator('#btn-net')).toHaveText('Turn on');
  await expect(page.locator('#net-url')).toBeEnabled();
});

test('an address that is not a WebSocket one is refused before anything connects', async ({ page }) => {
  await page.locator('#net-url').fill('http://127.0.0.1:18787');
  await page.locator('#btn-net').click();
  await expect(page.locator('#net-note')).toContainText('ws://');
  await expect(page.locator('#net-badge')).toHaveText('off');
});

test('a relay that answers turns the network on, and turning it off puts it back', async ({ page }) => {
  const { server, url } = await relayStub();
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
  await page.locator('#net-url').fill('ws://127.0.0.1:1');
  await page.locator('#btn-net').click();
  await expect(page.locator('#net-badge')).toHaveText('no relay');
  await reloadPage(page);
  await openPanel(page, 'Network');
  await expect(page.locator('#net-url')).toHaveValue('ws://127.0.0.1:1');
});

interface RelaySettings {
  listening: number;
  port: number;
  allowPrivate: boolean;
  allowInbound: boolean;
  origins: string[];
  fixedOrigins: string[];
  locked: string[];
  saved: boolean;
}

type Change = Pick<RelaySettings, 'port' | 'allowPrivate' | 'allowInbound' | 'origins'>;
type Reply = { type: 'settings'; settings: RelaySettings } | { type: 'asking' } | { type: 'refused'; reason: number };
type Answer = (change: Change) => Reply[];

const SETTINGS: RelaySettings = {
  listening: 18787,
  port: 18787,
  allowPrivate: false,
  allowInbound: false,
  origins: ['https://saved.example'],
  fixedOrigins: ['http://localhost:8000', 'https://tenshii.moe'],
  locked: [],
  saved: true,
};

const DENIED = 0;

function texts(list: string[]): Buffer {
  return Buffer.concat([
    Buffer.from([list.length]),
    ...list.map((text) => Buffer.concat([Buffer.from([Buffer.byteLength(text)]), Buffer.from(text)])),
  ]);
}

// A relay-to-page frame, as docs/relay.md lays it out.
function encode(reply: Reply): Buffer {
  switch (reply.type) {
    case 'asking':
      return Buffer.from([0x8b, 0, 0, 0, 0]);
    case 'refused':
      return Buffer.from([0x8c, 0, 0, 0, 0, reply.reason]);
    case 'settings': {
      const s = reply.settings;
      const head = Buffer.alloc(10);
      head.writeUInt8(0x8a, 0);
      head.writeUInt16LE(s.listening, 5);
      head.writeUInt16LE(s.port, 7);
      const bits = [s.allowPrivate, s.allowInbound, s.saved, s.locked.includes('port'),
        s.locked.includes('allowPrivate'), s.locked.includes('allowInbound')];
      head.writeUInt8(bits.reduce((flags, on, bit) => (on ? flags | (1 << bit) : flags), 0), 9);
      return Buffer.concat([head, texts(s.origins), texts(s.fixedOrigins)]);
    }
  }
}

function decodeChange(frame: Buffer): Change {
  const origins: string[] = [];
  let at = 9;
  for (let count = frame[8]; count > 0; count--) {
    const len = frame[at];
    origins.push(frame.subarray(at + 1, at + 1 + len).toString());
    at += 1 + len;
  }
  return {
    port: frame.readUInt16LE(5),
    allowPrivate: (frame[7] & 1) !== 0,
    allowInbound: (frame[7] & 2) !== 0,
    origins,
  };
}

// Each whole frame in what a page sent; a page masks every frame.
function* clientFrames(data: Buffer): Generator<Buffer> {
  let at = 0;
  while (at + 2 <= data.length) {
    let len = data[at + 1] & 0x7f;
    let head = 2;
    if (len === 126) {
      len = data.readUInt16BE(at + 2);
      head = 4;
    }
    const mask = data.subarray(at + head, at + head + 4);
    const payload = Buffer.from(data.subarray(at + head + 4, at + head + 4 + len));
    for (let i = 0; i < payload.length; i++) payload[i] ^= mask[i % 4];
    if ((data[at] & 0x0f) === 2) yield payload;
    at += head + 4 + len;
  }
}

function serverFrame(payload: Buffer): Buffer {
  const head = payload.length < 126
    ? Buffer.from([0x82, payload.length])
    : Buffer.from([0x82, 126, payload.length >> 8, payload.length & 0xff]);
  return Buffer.concat([head, payload]);
}

// A relay that has settings: version 4, which answers `GET_SETTINGS` with `settings` and each
// `SET_SETTINGS` with what `answer` returns.
async function settingsRelay(settings: RelaySettings, answer: Answer): Promise<{ server: Server; url: string }> {
  const server = createServer();
  server.on('upgrade', (request, socket) => {
    const key = request.headers['sec-websocket-key'] as string;
    const accept = createHash('sha1')
      .update(`${key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11`)
      .digest('base64');
    socket.write(
      `HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${accept}\r\n\r\n`,
    );
    socket.write(serverFrame(Buffer.from([0x80, 0, 0, 0, 0, 4, 0x40, 0])));
    socket.on('data', (data: Buffer) => {
      for (const frame of clientFrames(data)) {
        if (frame[0] === 0x09) socket.write(serverFrame(encode({ type: 'settings', settings })));
        if (frame[0] === 0x0a) {
          for (const reply of answer(decodeChange(frame))) socket.write(serverFrame(encode(reply)));
        }
      }
    });
    socket.on('error', () => undefined);
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  return { server, url: `ws://127.0.0.1:${(server.address() as AddressInfo).port}` };
}

// Turns the network on against a relay with `settings`.
async function connectWithSettings(page: Page, settings: RelaySettings, answer: Answer = () => []): Promise<Server> {
  const { server, url } = await settingsRelay(settings, answer);
  await page.locator('#net-url').fill(url);
  await page.locator('#btn-net').click();
  await expect(page.locator('#net-badge')).toHaveText('on');
  return server;
}

function stop(server: Server): void {
  server.closeAllConnections();
  server.close();
}

test('the relay settings show by themselves once the relay is connected', async ({ page }) => {
  await expect(page.locator('#relay-note')).toContainText('turn on Internet');
  await expect(page.locator('#relay-form')).toBeHidden();
  const server = await connectWithSettings(page, SETTINGS);
  await expect(page.locator('#relay-form')).toBeVisible();
  await expect(page.locator('#relay-private')).not.toBeChecked();
  await expect(page.locator('#relay-port')).toHaveValue('18787');
  await expect(page.locator('#relay-origins li')).toHaveText([
    /http:\/\/localhost:8000\s*always/,
    /https:\/\/tenshii\.moe\s*always/,
    /https:\/\/saved\.example\s*Remove/,
  ]);
  await expect(page.locator('#relay-save')).toBeDisabled();

  await page.locator('#btn-net').click();
  await expect(page.locator('#relay-form')).toBeHidden();
  stop(server);
});

test('a change the relay asks about waits for the answer, then shows what was saved', async ({ page }) => {
  const sent: object[] = [];
  const server = await connectWithSettings(page, SETTINGS, (change) => {
    sent.push(change);
    return [{ type: 'asking' }, { type: 'settings', settings: { ...SETTINGS, ...change } }];
  });
  await page.locator('#relay-private').check();
  await page.locator('#relay-origin').fill('https://New.example/');
  await page.locator('#relay-origin').press('Enter');
  await page.getByRole('button', { name: 'Stop trusting https://saved.example' }).click();
  await page.locator('#relay-save').click();
  await expect(page.locator('#relay-note')).toHaveText('Saved.');
  expect(sent).toEqual([{ port: 18787, allowPrivate: true, allowInbound: false, origins: ['https://new.example'] }]);
  await expect(page.locator('#relay-private')).toBeChecked();
  await expect(page.locator('#relay-save')).toBeDisabled();
  stop(server);
});

test('a refused change says why and keeps the edits', async ({ page }) => {
  const server = await connectWithSettings(page, SETTINGS, () => [
    { type: 'asking' }, { type: 'refused', reason: DENIED },
  ]);
  await page.locator('#relay-inbound').check();
  await page.locator('#relay-save').click();
  await expect(page.locator('#relay-note')).toContainText('did not allow the change');
  await expect(page.locator('#relay-inbound')).toBeChecked();
  await expect(page.locator('#relay-save')).toBeEnabled();
  await page.locator('#relay-revert').click();
  await expect(page.locator('#relay-inbound')).not.toBeChecked();
  stop(server);
});

test('settings a flag sets cannot be changed, and a new port waits for a restart', async ({ page }) => {
  const server = await connectWithSettings(
    page,
    { ...SETTINGS, allowPrivate: true, locked: ['allowPrivate'] },
    (change) => [{ type: 'settings', settings: { ...SETTINGS, locked: ['allowPrivate'], ...change } }],
  );
  await expect(page.locator('#relay-private')).toBeDisabled();
  await expect(page.locator('#relay-private-lock')).toBeVisible();
  await page.locator('#relay-port').fill('19000');
  await expect(page.locator('#relay-port-note')).toContainText('then on 19000');
  await page.locator('#relay-save').click();
  await expect(page.locator('#relay-note')).toHaveText('Saved. Restart switch-relay to use the new port.');
  stop(server);
});

test('an address that is not a page is not trusted', async ({ page }) => {
  const server = await connectWithSettings(page, SETTINGS);
  await page.locator('#relay-origin').fill('https://example.org/some/path');
  await page.locator('#relay-add').click();
  await expect(page.locator('#relay-note')).toContainText('by its address alone');
  await expect(page.locator('#relay-origins li')).toHaveCount(3);
  stop(server);
});

test('a relay from before settings is called too old', async ({ page }) => {
  const { server, url } = await relayStub();
  await page.locator('#net-url').fill(url);
  await page.locator('#btn-net').click();
  await expect(page.locator('#relay-note')).toContainText('too old');
  await expect(page.locator('#relay-form')).toBeHidden();
  stop(server);
});
