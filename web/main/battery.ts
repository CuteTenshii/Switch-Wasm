// Host battery state for the psm service (Chromium only, via the Battery Status API).

import { call, hasSession, whenReady } from './rpc';

interface BatteryManager extends EventTarget {
  level: number;
  charging: boolean;
}

type NavigatorWithBattery = Navigator & { getBattery?: () => Promise<BatteryManager> };

export function watchBattery(): void {
  const nav = navigator as NavigatorWithBattery;
  if (!nav.getBattery) return;
  void Promise.all([nav.getBattery(), whenReady()]).then(([battery]) => {
    const push = () => {
      if (!hasSession()) return;
      void call('setBattery', Math.round(battery.level * 100), battery.charging ? 1 : 0);
    };
    push();
    battery.addEventListener('levelchange', push);
    battery.addEventListener('chargingchange', push);
  });
}
