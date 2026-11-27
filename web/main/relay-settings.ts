// The Network tab's Relay settings: switch-relay's settings, carried on the link to it
// (docs/relay.md, Settings). The worker asks for them as soon as the relay says hello.

import { $, el } from './dom';
import { onRelayLink } from './network';
import { call, onWorker } from './rpc';

interface Report {
  listening: number;
  port: number;
  allowPrivate: boolean;
  allowInbound: boolean;
  origins: string[];
  fixedOrigins: string[];
  locked: string[];
  saved: boolean;
}

interface Wanted {
  port: number;
  allowPrivate: boolean;
  allowInbound: boolean;
  origins: string[];
}

type Message =
  | ({ type: 'settings' } & Report)
  | { type: 'asking' }
  | { type: 'refused'; reason: string }
  | { type: 'old' };

const WAITING = 'Shown once the relay is connected: turn on Internet above.';

const REFUSALS: Record<string, string> = {
  denied: 'switch-relay did not allow the change, so nothing changed.',
  busy: 'switch-relay is still waiting for an answer about another change. Answer it, then save again.',
  locked: 'A flag switch-relay was started with sets that, so it cannot change from here.',
  invalid: 'switch-relay did not accept those settings.',
  unsaved: 'switch-relay could not write its settings file, so nothing changed. Its terminal says why.',
};

const note = $('relay-note');
const form = $<HTMLFormElement>('relay-form');
const privateBox = $<HTMLInputElement>('relay-private');
const inboundBox = $<HTMLInputElement>('relay-inbound');
const portInput = $<HTMLInputElement>('relay-port');
const portNote = $('relay-port-note');
const originList = $('relay-origins');
const originInput = $<HTMLInputElement>('relay-origin');
const addButton = $<HTMLButtonElement>('relay-add');
const saveButton = $<HTMLButtonElement>('relay-save');
const revertButton = $<HTMLButtonElement>('relay-revert');

// What the relay last said, and the origins as edited here since.
let report: Report | null = null;
let origins: string[] = [];
let busy = false;

function say(text: string, error = false): void {
  note.textContent = text;
  note.classList.toggle('is-error', error);
}

// `text` as a browser sends it in `Origin`, or null when it is not a page's address.
function asOrigin(text: string): string | null {
  try {
    const url = new URL(text.trim());
    if (url.protocol !== 'http:' && url.protocol !== 'https:') return null;
    return url.href === `${url.origin}/` ? url.origin : null;
  } catch {
    return null;
  }
}

function wanted(): Wanted {
  return {
    port: portInput.valueAsNumber,
    allowPrivate: privateBox.checked,
    allowInbound: inboundBox.checked,
    origins,
  };
}

function changed(): boolean {
  if (!report) return false;
  const now = wanted();
  return now.port !== report.port
    || now.allowPrivate !== report.allowPrivate
    || now.allowInbound !== report.allowInbound
    || now.origins.join('\n') !== report.origins.join('\n');
}

function portValid(): boolean {
  const port = portInput.valueAsNumber;
  return Number.isInteger(port) && port >= 1 && port <= 65535;
}

function sync(): void {
  const dirty = changed();
  saveButton.disabled = busy || !dirty || !portValid();
  revertButton.disabled = busy || !dirty;
  addButton.disabled = busy;
  form.inert = busy;
}

function originRow(origin: string, fixed: boolean): HTMLLIElement {
  const row = el('li', 'relay-origin');
  // Origins come from the relay's file, so they are text, never markup.
  row.append(el('span', 'relay-origin-name', origin));
  if (fixed) {
    row.append(el('span', 'relay-lock', 'always'));
    return row;
  }
  const remove = el('button', 'btn small', 'Remove');
  remove.type = 'button';
  remove.setAttribute('aria-label', `Stop trusting ${origin}`);
  remove.addEventListener('click', () => {
    origins = origins.filter((o) => o !== origin);
    renderOrigins();
    originInput.focus();
    sync();
  });
  row.append(remove);
  return row;
}

function renderOrigins(): void {
  if (!report) return;
  originList.replaceChildren(
    ...report.fixedOrigins.map((o) => originRow(o, true)),
    ...origins.map((o) => originRow(o, false)),
  );
}

function renderPortNote(): void {
  if (!report) return;
  const port = portValid() ? portInput.valueAsNumber : report.port;
  portNote.hidden = port === report.listening;
  portNote.textContent = `switch-relay listens on port ${report.listening} until it restarts, then on ${port}. The relay address above needs the new port then too.`;
}

function show(next: Report): void {
  report = next;
  origins = [...next.origins];
  const locked = new Set(next.locked);
  privateBox.checked = next.allowPrivate;
  privateBox.disabled = locked.has('allowPrivate');
  $('relay-private-lock').hidden = !privateBox.disabled;
  inboundBox.checked = next.allowInbound;
  inboundBox.disabled = locked.has('allowInbound');
  $('relay-inbound-lock').hidden = !inboundBox.disabled;
  portInput.value = String(next.port);
  portInput.disabled = locked.has('port');
  $('relay-port-lock').hidden = !portInput.disabled;
  renderOrigins();
  renderPortNote();
  form.hidden = false;
  sync();
}

function save(): void {
  if (!report || !portValid()) return;
  const { port, allowPrivate, allowInbound } = wanted();
  busy = true;
  sync();
  say('Saving…');
  call('setRelaySettings', port, allowPrivate, allowInbound, origins).catch((e: Error) => {
    busy = false;
    sync();
    say(`The change did not reach switch-relay: ${e.message}`, true);
  });
}

function received(message: Message): void {
  const saving = busy;
  switch (message.type) {
    case 'old':
      say('This switch-relay is too old to change its settings from here. Download the current one above.', true);
      return;
    case 'asking':
      say('Allow or deny the change in the window switch-relay opened on its computer.');
      return;
    case 'refused':
      busy = false;
      sync();
      say(REFUSALS[message.reason] ?? `switch-relay refused the change (${message.reason}).`, true);
      return;
    case 'settings':
      busy = false;
      show(message);
      if (!message.saved) {
        say(saving
          ? 'Changed until switch-relay exits. It has nowhere to save settings.'
          : 'switch-relay has nowhere to save settings, so changes last until it exits.');
      } else if (saving) {
        say(message.port === message.listening ? 'Saved.' : 'Saved. Restart switch-relay to use the new port.');
      } else {
        say('');
      }
  }
}

function addOrigin(): void {
  if (!report) return;
  const origin = asOrigin(originInput.value);
  if (!origin) {
    say('A page is trusted by its address alone, like https://example.org.', true);
    return;
  }
  if (report.fixedOrigins.includes(origin) || origins.includes(origin)) {
    say(`${origin} is already trusted.`);
  } else {
    origins = [...origins, origin];
    renderOrigins();
    say(`${origin} is trusted once you save.`);
  }
  originInput.value = '';
  sync();
}

onWorker('net_settings', ({ json }) => received(JSON.parse(json) as Message));
onRelayLink((state) => {
  if (state === 'open') {
    say('Reading settings…');
    return;
  }
  busy = false;
  report = null;
  form.hidden = true;
  say(WAITING);
});
addButton.addEventListener('click', addOrigin);
originInput.addEventListener('keydown', (event) => {
  if (event.key !== 'Enter') return;
  event.preventDefault();
  addOrigin();
});
form.addEventListener('submit', (event) => {
  event.preventDefault();
  save();
});
form.addEventListener('input', () => {
  renderPortNote();
  sync();
});
revertButton.addEventListener('click', () => {
  if (report) show(report);
  say('');
});
