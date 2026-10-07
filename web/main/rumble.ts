// Map guest rumble to the Gamepad API "dual-rumble" effect (Chromium only).

import { call } from './rpc';

// The actuator shape browsers expose; `reset` is Chromium's.
interface DualRumbleActuator {
  playEffect?(type: string, params: {
    duration: number;
    strongMagnitude: number;
    weakMagnitude: number;
  }): Promise<unknown>;
  reset?(): Promise<unknown>;
}

let lastRumble = -1;

export async function pullVibration(pad: Gamepad | undefined): Promise<void> {
  const actuator = pad?.vibrationActuator as DualRumbleActuator | undefined;
  if (!actuator?.playEffect) return;
  const packed = await call('vibration');
  if (packed === lastRumble) return; // re-issuing the same effect stutters it
  lastRumble = packed;
  const strong = (packed & 0xffff) / 1000;
  const weak = (packed >>> 16) / 1000;
  try {
    if (strong === 0 && weak === 0) {
      await actuator.reset?.();
    } else {
      // Outlive the poll interval so a held rumble stays continuous.
      await actuator.playEffect('dual-rumble', {
        duration: 120,
        strongMagnitude: strong,
        weakMagnitude: weak,
      });
    }
  } catch {
    // The actuator exists but refuses the effect.
  }
}
