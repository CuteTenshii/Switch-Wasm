// Handheld/docked toggle (Horizon's `AppletOperationMode`), changeable while a
// title runs: the core queues the AM messages a real dock sends.

import { Gamepad2, Tv } from 'lucide';
import { $, el } from './dom';
import { icon } from './icons';
import { call, whenReady } from './rpc';

const DOCKED_KEY = 'docked';

const button = $<HTMLButtonElement>('btn-dock');
let docked = localStorage.getItem(DOCKED_KEY) === '1';

function render(): void {
  button.replaceChildren(icon(docked ? Tv : Gamepad2), el('span', null, docked ? 'Docked' : 'Handheld'));
  button.setAttribute('aria-pressed', docked ? 'true' : 'false');
}

button.addEventListener('click', () => {
  docked = !docked;
  render();
  localStorage.setItem(DOCKED_KEY, docked ? '1' : '0');
  void call('setOperationMode', docked);
});

render();
if (docked) void whenReady().then(() => call('setOperationMode', true));
