import { describe, expect, test } from 'bun:test';
import { knownTitle } from '../../web/main/titles';

const icon = (...values: number[]) => new Uint8Array(values);

describe('knownTitle', () => {
  test('a game read twice keeps one icon URL', () => {
    const listed = knownTitle({ id: '0100aaaa00000000' }, icon(1, 2), 'image/jpeg');
    const launched = knownTitle({ id: '0100AAAA00000000' }, icon(1, 2), 'image/jpeg');
    expect(listed.iconUrl).not.toBeNull();
    expect(launched.iconUrl).toBe(listed.iconUrl);
  });

  test('a new icon for a title gets its own URL', () => {
    const first = knownTitle({ id: '0100cccc00000000' }, icon(4), 'image/jpeg');
    expect(knownTitle({ id: '0100cccc00000000' }, icon(5), 'image/jpeg').iconUrl).not.toBe(first.iconUrl);
  });

  test('homebrew is the same title from a peek and from its control data', () => {
    const peeked = knownTitle({ id: '', name: 'App' }, icon(6), 'image/png');
    const loaded = knownTitle({ id: '0000000000000000', name: 'App' }, icon(6), 'image/png');
    expect(loaded.iconUrl).toBe(peeked.iconUrl);
  });

  test('two homebrew apps with one icon keep apart', () => {
    const one = knownTitle({ id: '', name: 'One' }, icon(7), 'image/jpeg');
    const two = knownTitle({ id: '', name: 'Two' }, icon(7), 'image/jpeg');
    expect(two.iconUrl).not.toBe(one.iconUrl);
  });
});
