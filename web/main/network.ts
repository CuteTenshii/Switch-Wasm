// The Network tab: the switch that gives the guest Internet through the relay
// (`crates/switch-relay`, run on this computer; see docs/relay.md), and the guest's
// connections. Off, the guest sees a network that never answers.

import { Power, PowerOff } from 'lucide';
import type { NetState } from '../shared/protocol';
import { fmtSize } from '../shared/format';
import { $ } from './dom';
import { icon } from './icons';
import { log } from './log';
import { call, onNetState, whenReady } from './rpc';

const DEFAULT_URL = 'ws://127.0.0.1:18787';
const URL_KEY = 'relay-url';
const ON_KEY = 'relay-on';

interface Connection {
  id: number;
  host: string;
  port: number;
  proto: 'tcp' | 'udp' | 'listen';
  state: string;
  sent: number;
  received: number;
}

const button = $<HTMLButtonElement>('btn-net');
const urlInput = $<HTMLInputElement>('net-url');
const badge = $('net-badge');
const note = $('net-note');
const table = $<HTMLElement>('net-table');
const link = $('net-link');
const rows = $('net-rows');
let state: NetState = 'off';

urlInput.value = localStorage.getItem(URL_KEY) ?? DEFAULT_URL;

const BADGES: Record<NetState, string> = {
  off: 'off',
  connecting: 'connecting',
  open: 'on',
  unreachable: 'no relay',
};

function on(): boolean {
  return state === 'connecting' || state === 'open';
}

function say(text: string): void {
  note.textContent = text;
  note.hidden = text === '';
}

function render(): void {
  badge.textContent = BADGES[state];
  badge.classList.toggle('on', state === 'open');
  button.replaceChildren(icon(on() ? PowerOff : Power), on() ? 'Turn off' : 'Turn on');
  button.setAttribute('aria-pressed', String(on()));
  urlInput.disabled = on();
}

/** A UDP socket or a listener has no single peer, only the port it asked for (0 is any). */
function describe(c: Connection): string {
  const port = c.port === 0 ? 'any port' : `port ${c.port}`;
  switch (c.proto) {
    case 'tcp': return `${c.host}:${c.port}`;
    case 'udp': return `UDP on ${port}`;
    case 'listen': return `Listening on ${port}`;
  }
}

function showConnections(list: Connection[]): void {
  $('net-count').textContent = String(list.length);
  $('net-empty').hidden = list.length > 0;
  table.hidden = list.length === 0;
  rows.replaceChildren(...list.map((c) => {
    const row = document.createElement('tr');
    // Names come from the guest, so they are text, never markup.
    for (const text of [
      describe(c), c.state, fmtSize(c.sent),
      fmtSize(c.received),
    ]) {
      const cell = document.createElement('td');
      cell.textContent = text;
      row.append(cell);
    }
    return row;
  }));
}

async function refresh(): Promise<void> {
  showConnections(JSON.parse(await call('netConnections')) as Connection[]);
  const status = JSON.parse(await call('netStatus')) as { socket: string; out: number; in: number };
  link.textContent = `Socket ${status.socket}, ${status.out} frames out, ${status.in} in.`;
}

onNetState((next) => {
  state = next;
  render();
  say(next === 'unreachable' ? `No relay answered at ${urlInput.value}. Start switch-relay, then turn this on.` : '');
  if (next === 'open') log('Online through the relay.');
  void refresh();
});

function connect(): void {
  const url = urlInput.value.trim();
  if (!/^wss?:\/\//.test(url)) {
    say('The relay address starts with ws:// or wss://.');
    return;
  }
  localStorage.setItem(URL_KEY, url);
  localStorage.setItem(ON_KEY, '1');
  say('');
  void call('connectNetwork', url).catch((e: Error) => say(`Relay: ${e.message}`));
}

button.addEventListener('click', () => {
  if (on()) {
    localStorage.removeItem(ON_KEY);
    void call('disconnectNetwork');
    return;
  }
  connect();
});

if (localStorage.getItem(ON_KEY) === '1') void whenReady().then(connect);

// Only while someone can see the table, and the guest can have connections.
setInterval(() => {
  const visible = table.closest('.tabpanel')?.classList.contains('is-active');
  if (visible && document.body.classList.contains('panel-open')) void refresh();
}, 1000);

render();
