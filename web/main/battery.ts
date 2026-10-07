// Host battery state for the psm service (Chromium only, via the Battery Status API).

import { call } from './rpc';

interface BatteryManager extends EventTarget {
  level: number;
  charging: boolean;
}

type NavigatorWithBattery = Navigator & { getBattery?: () => Promise<BatteryManager> };

export function watchBattery(): void {
  const nav = navigator as NavigatorWithBattery;
  if (!nav.getBattery) return;
  void nav.getBattery().then((battery) => {
    const push = () => {
      void call('set_battery', Math.round(battery.level * 100), battery.charging ? 1 : 0);
    };
    push();
    battery.addEventListener('levelchange', push);
    battery.addEventListener('chargingchange', push);
  });
}
