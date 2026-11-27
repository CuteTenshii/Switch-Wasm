// HidNpadButton bits, in Horizon's order.
export const BTN = {
  A: 1 << 0, B: 1 << 1, X: 1 << 2, Y: 1 << 3,
  STICK_L: 1 << 4, STICK_R: 1 << 5,
  L: 1 << 6, R: 1 << 7, ZL: 1 << 8, ZR: 1 << 9,
  PLUS: 1 << 10, MINUS: 1 << 11,
  LEFT: 1 << 12, UP: 1 << 13, RIGHT: 1 << 14, DOWN: 1 << 15,
} as const;

export type Button = keyof typeof BTN;

export function isButton(name: string | undefined): name is Button {
  return name !== undefined && Object.hasOwn(BTN, name);
}
