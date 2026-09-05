/* host battery

   Feeds the Switch's psm (power management) service. Only Chromium exposes
   the Battery Status API (Firefox and Safari never shipped it, over privacy
   concerns), so elsewhere the emulated battery just stays at the wasm
   default (full, charging). Event-driven rather than polled: battery level
   changes far slower than the 16ms input tick, and the level/charging state
   is cached worker-side so a freshly created session (including after
   "reset") picks it up without this having to fire again. */

import { call } from './rpc';

interface BatteryManager extends EventTarget {
  level: number;
  charging: boolean;
}

type NavigatorWithBattery = Navigator & { getBattery?: () => Promise<BatteryManager> };

export function watchBattery(): void {
  const nav = navigator as NavigatorWithBattery;
  if (!nav.getBattery) return;
  nav.getBattery().then((battery) => {
    const push = () =>
      call('set_battery', Math.round(battery.level * 100), battery.charging ? 1 : 0);
    push();
    battery.addEventListener('levelchange', push);
    battery.addEventListener('chargingchange', push);
  });
}
