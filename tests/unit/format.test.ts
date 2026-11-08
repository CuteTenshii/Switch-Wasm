import { describe, expect, test } from 'bun:test';
import { formatBytes } from '../../web/main/format';
import { fmtCount, fmtSize } from '../../web/shared/format';

describe('fmtSize', () => {
  test.each([
    [0, '0 B'],
    [1023, '1023 B'],
    [1024, '1.0 KiB'],
    [1536, '1.5 KiB'],
    [1024 ** 2 - 1, '1024.0 KiB'],
    [1024 ** 2, '1.00 MiB'],
    [1024 ** 3, '1.00 GiB'],
    [5.5 * 1024 ** 3, '5.50 GiB'],
  ])('%d bytes is %s', (n, text) => {
    expect(fmtSize(n)).toBe(text);
  });
});

describe('formatBytes', () => {
  test.each([
    [512, '512 B'],
    [1024, '1 KiB'],
    [1536, '2 KiB'],
    [1024 ** 2, '1.0 MiB'],
    [3.25 * 1024 ** 3, '3.25 GiB'],
  ])('%d bytes is %s', (n, text) => {
    expect(formatBytes(n)).toBe(text);
  });
});

describe('fmtCount', () => {
  test.each([
    [0, '0'],
    [999_999, '999,999'],
    [1234.6, '1,235'],
    [1_000_000, '1.0M'],
    [12_345_678, '12.3M'],
    [1e9, '1.00B'],
    [2.5e12, '2.50T'],
  ])('%d is %s', (n, text) => {
    expect(fmtCount(n)).toBe(text);
  });
});
