import { describe, expect, test } from 'bun:test';
import { crc32 } from '../../web/shared/crc32';

const bytes = (text: string) => new TextEncoder().encode(text);

describe('crc32', () => {
  test('matches the CRC-32 check value', () => {
    expect(crc32(bytes('123456789'))).toBe(0xCBF43926);
  });

  test('is zero for no data', () => {
    expect(crc32(new Uint8Array(0))).toBe(0);
  });

  test('continues across pieces to the same result as the whole', () => {
    const whole = bytes('The quick brown fox jumps over the lazy dog');
    expect(crc32(whole.subarray(10), crc32(whole.subarray(0, 10)))).toBe(crc32(whole));
    expect(crc32(whole)).toBe(0x414FA339);
  });
});
