// The guest's software keyboard, answered from a dialog while the guest waits.

import type { KeyboardRequest } from '../shared/protocol';
import { $ } from './dom';
import { log } from './log';
import { call, hasSession } from './rpc';

const dialogEl = $<HTMLDialogElement>('keyboard');
const formEl = $<HTMLFormElement>('keyboard-form');
const titleEl = $('keyboard-title');
const subEl = $('keyboard-sub');
const textEl = $<HTMLInputElement>('keyboard-text');
const countEl = $('keyboard-count');
const okEl = $<HTMLButtonElement>('btn-keyboard-ok');

let shown: KeyboardRequest | null = null;

function updateCount(): void {
  if (!shown) return;
  const { length } = textEl.value;
  okEl.disabled = length < shown.minLength;
  countEl.textContent = length < shown.minLength
    ? `At least ${shown.minLength} characters`
    : `${length}/${shown.maxLength}`;
}

function show(request: KeyboardRequest): void {
  shown = request;
  titleEl.textContent = request.header || 'The game asks for text';
  subEl.textContent = request.sub;
  subEl.hidden = !request.sub;
  textEl.value = '';
  textEl.placeholder = request.guide;
  textEl.maxLength = request.maxLength;
  textEl.type = request.password ? 'password' : 'text';
  okEl.textContent = request.ok || 'OK';
  updateCount();
  dialogEl.showModal();
}

async function answer(text: string | null): Promise<void> {
  shown = null;
  dialogEl.close();
  if (!(await call('keyboard_answer', text))) {
    log('The keyboard was no longer waiting; the text was not sent.', 'warn');
  }
}

// Open the dialog when the guest starts waiting for text.
export async function pollKeyboard(): Promise<void> {
  if (shown || !hasSession()) return;
  const request = await call('keyboard_request');
  if (request && !shown) show(request);
}

// For a session being replaced: its keyboard goes with it.
export function dropKeyboard(): void {
  shown = null;
  dialogEl.close();
}

textEl.addEventListener('input', updateCount);
formEl.addEventListener('submit', (e) => {
  e.preventDefault();
  if (shown && textEl.value.length >= shown.minLength) void answer(textEl.value);
});
$('btn-keyboard-cancel').addEventListener('click', () => void answer(null));
// Escape cancels, as B does on the console.
dialogEl.addEventListener('cancel', (e) => {
  e.preventDefault();
  if (shown) void answer(null);
});
