// Handheld/docked toggle (Horizon's `AppletOperationMode`), changeable while a
// title runs: the core queues the AM messages a real dock sends.

import { $ } from './dom';
import { log } from './log';
import { call } from './rpc';

const button = $<HTMLButtonElement>('btn-dock');
let docked = false;

function render(): void {
  button.textContent = docked ? 'Docked' : 'Handheld';
  button.setAttribute('aria-pressed', docked ? 'true' : 'false');
}

button.addEventListener('click', () => {
  docked = !docked;
  render();
  void call('set_operation_mode', docked ? 1 : 0);
  log(
    docked
      ? 'Docked - 1080p, boost clocks, no touchscreen.'
      : 'Handheld - 720p, normal clocks, touchscreen.',
  );
});

render();
