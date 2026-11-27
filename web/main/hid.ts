// Motion controllers over WebHID. The worker opens and reads the device; only the page can ask
// the user to grant one.

import { $ } from './dom';
import { log } from './log';
import { call, onWorker, whenReady } from './rpc';

interface HidApi {
  requestDevice(options: { filters: { vendorId: number; productId: number }[] }): Promise<unknown[]>;
}

// Nintendo's Joy-Con (L), Joy-Con (R) and Pro Controller.
const FILTERS = [0x2006, 0x2007, 0x2009].map((productId) => ({ vendorId: 0x057e, productId }));

const hid = (navigator as Navigator & { hid?: HidApi }).hid;
const button = $<HTMLButtonElement>('btn-motion');
let device: string | null = null;
// Why the last attach failed, until one succeeds or the user tries again.
let failure: string | null = null;

function render(): void {
  button.setAttribute('aria-pressed', device ? 'true' : 'false');
  button.classList.toggle('failed', failure !== null);
  if (device) button.title = `Motion from ${device}. Click to disconnect.`;
  else if (failure) button.title = `Motion failed: ${failure}. Click to try again.`;
  else button.title = 'Motion controls: connect a Joy-Con or Pro Controller over WebHID';
}

function failed(err: unknown): void {
  failure = err instanceof Error ? err.message : String(err);
  log(`[input] motion: ${failure}`, 'warn');
  render();
}

async function toggle(hid: HidApi): Promise<void> {
  if (device) {
    await call('detachMotion');
    return;
  }
  failure = null;
  render();
  await hid.requestDevice({ filters: FILTERS });
  if (!(await call('attachMotion'))) log('[input] no motion controller was granted', 'warn');
}

if (hid) {
  button.hidden = false;
  button.addEventListener('click', () => {
    toggle(hid).catch(failed);
  });
  onWorker('motion', ({ device: name }) => {
    device = name;
    if (name) failure = null;
    render();
    if (name) log(`[input] motion from ${name}`);
  });
  // A device granted on an earlier visit needs no gesture; none granted is no failure.
  void whenReady().then(() => call('attachMotion')).catch(failed);
  render();
}
