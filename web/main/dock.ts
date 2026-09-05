/* handheld / docked

   Horizon's `AppletOperationMode`, and the one switch behind everything that
   depends on it: the resolution `vi` reports, the performance mode `am` and
   `apm` report, the GPU clock `clkrst` reports, and whether the touchscreen
   exists at all.

   Changeable while a title runs, because that is what a dock is. The number
   on its own would change nothing: a title reads the operation mode once and
   lays out for that answer, so the emulator queues the two AM messages a real
   dock sends, and those are what send the title back to ask. */

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
  call('set_operation_mode', docked ? 1 : 0);
  log(
    docked
      ? 'Docked - 1080p, boost clocks, no touchscreen.'
      : 'Handheld - 720p, normal clocks, touchscreen.',
    'dim',
  );
});

render();
