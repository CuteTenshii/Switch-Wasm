/* ---------- rumble ----------

   Switch rumble drives two linear resonant actuators independently, and the
   Gamepad API's "dual-rumble" effect is the same shape: the guest's low band
   becomes strongMagnitude, its high band weakMagnitude. Only Chromium-family
   browsers implement vibrationActuator, so this is best-effort and silent
   where it is missing. */

import { call } from './rpc';

/** What the browsers that have one actually expose - `reset` is Chromium's,
 *  and the effect type is a string the spec has renamed more than once. */
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
  if (packed === lastRumble) return;   // re-issuing the same effect stutters it
  lastRumble = packed;
  const strong = (packed & 0xffff) / 1000;
  const weak = (packed >>> 16) / 1000;
  try {
    if (strong === 0 && weak === 0) {
      await actuator.reset?.();
    } else {
      // Outlive the poll interval so a held rumble is continuous rather than
      // a stutter, but stay short enough that it stops promptly when the
      // guest lets go.
      await actuator.playEffect('dual-rumble', {
        duration: 120,
        strongMagnitude: strong,
        weakMagnitude: weak,
      });
    }
  } catch {
    // A browser that advertises the actuator but refuses the effect.
  }
}
