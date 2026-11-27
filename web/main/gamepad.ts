// A browser gamepad read as the Switch controller it is: buttons as a `HidNpadButton` mask,
// sticks in Horizon's -32768..32767 with up positive.

import { BTN } from './buttons';
import type { HostPad } from '../shared/protocol';

// The W3C standard gamepad layout's buttons 0-15, in order.
const STANDARD_PAD = [
  BTN.B, BTN.A, BTN.Y, BTN.X,
  BTN.L, BTN.R, BTN.ZL, BTN.ZR,
  BTN.MINUS, BTN.PLUS, BTN.STICK_L, BTN.STICK_R,
  BTN.UP, BTN.DOWN, BTN.LEFT, BTN.RIGHT,
];

// Chrome turns a lone Joy-Con into a sideways mini pad: face buttons rotated, the rail
// button on Select, SL/SR on the shoulders. These give each standard index the button it
// really is, so the guest sees the Joy-Con a console would.
const LONE_JOY_CON_L = [
  BTN.LEFT, BTN.DOWN, BTN.UP, BTN.RIGHT,
  BTN.LEFT_SL, BTN.LEFT_SR, BTN.ZL, 0,
  BTN.L, BTN.MINUS, BTN.STICK_L,
];
const LONE_JOY_CON_R = [
  BTN.A, BTN.X, BTN.B, BTN.Y,
  BTN.RIGHT_SL, BTN.RIGHT_SR, 0, BTN.ZR,
  BTN.R, BTN.PLUS, BTN.STICK_R,
];

const DEADZONE = 0.15;

type Stick = [x: number, y: number];

export interface PadReading {
  hostPad: HostPad;
  buttons: number;
  // A stick the controller does not have is left out.
  left?: Stick;
  right?: Stick;
}

/** Which of Nintendo's controllers a standard-mapped pad is, by the product id in its name. */
export function hostPadOf(pad: Pick<Gamepad, 'id' | 'mapping'>): HostPad {
  if (pad.mapping !== 'standard') return 'full';
  const product = pad.id.match(/Vendor: 057e Product: (2006|2007)\b/i)?.[1];
  if (product === '2006') return 'joyLeft';
  if (product === '2007') return 'joyRight';
  return 'full';
}

export function readGamepad(pad: Pick<Gamepad, 'id' | 'mapping' | 'buttons' | 'axes'>): PadReading {
  const hostPad = hostPadOf(pad);
  const layout = hostPad === 'joyLeft' ? LONE_JOY_CON_L : hostPad === 'joyRight' ? LONE_JOY_CON_R : STANDARD_PAD;
  let buttons = 0;
  layout.forEach((bit, i) => {
    if (pad.buttons[i]?.pressed) buttons |= bit;
  });
  const axis = (i: number) => {
    const value = pad.axes[i] ?? 0;
    return Math.abs(value) > DEADZONE ? Math.round(value * 32767) : 0;
  };
  // Browsers point Y down. Chrome's sideways stick has the Joy-Con's up on X and its right
  // on Y, mirrored between the two Joy-Cons.
  switch (hostPad) {
    case 'joyLeft':
      return { hostPad, buttons, left: [-axis(1), -axis(0)] };
    case 'joyRight':
      return { hostPad, buttons, right: [axis(1), axis(0)] };
    case 'full':
      return { hostPad, buttons, left: [axis(0), -axis(1)], right: [axis(2), -axis(3)] };
  }
}
