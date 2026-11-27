import { describe, expect, test } from 'bun:test';
import { knownTitle } from '../../web/main/titles';

const icon = (...values: number[]) => new Uint8Array(values);

describe('knownTitle', () => {
  test('a game read twice keeps one icon URL', () => {
    const listed = knownTitle({ title_id: '0100aaaa00000000', icon_mime: 'image/jpeg' }, icon(1, 2));
    const launched = knownTitle({ title_id: '0100AAAA00000000', icon_mime: 'image/jpeg' }, icon(1, 2));
    expect(listed.iconUrl).not.toBeNull();
    expect(launched.iconUrl).toBe(listed.iconUrl);
  });

  test('a new icon for a title gets its own URL', () => {
    const first = knownTitle({ title_id: '0100cccc00000000', icon_mime: 'image/jpeg' }, icon(4));
    expect(knownTitle({ title_id: '0100cccc00000000', icon_mime: 'image/jpeg' }, icon(5)).iconUrl).not.toBe(first.iconUrl);
  });

  test('homebrew is the same title from a peek and from its control data', () => {
    const peeked = knownTitle({ title_id: '', name: 'App', icon_mime: 'image/png' }, icon(6));
    const loaded = knownTitle({ title_id: '0000000000000000', name: 'App', icon_mime: 'image/png' }, icon(6));
    expect(loaded.iconUrl).toBe(peeked.iconUrl);
  });

  test('two homebrew apps with one icon keep apart', () => {
    const one = knownTitle({ title_id: '', name: 'One', icon_mime: 'image/jpeg' }, icon(7));
    const two = knownTitle({ title_id: '', name: 'Two', icon_mime: 'image/jpeg' }, icon(7));
    expect(two.iconUrl).not.toBe(one.iconUrl);
  });
});
