import { describe, expect, test } from 'bun:test';
import { parseSaveId, saveIdFor } from '../../web/shared/saves';

describe('parseSaveId', () => {
  test('a save with no owner is shared', () => {
    expect(parseSaveId('0100aaaa00000000')).toEqual({
      save: '0100aaaa00000000', uid: null, title: '0100aaaa00000000', cache: null,
    });
  });

  test('a profile\'s save names its owner', () => {
    const id = parseSaveId('0100aaaa00000000@00112233445566778899aabbccddeeff');
    expect(id.uid).toBe('00112233445566778899aabbccddeeff');
    expect(id.save).toBe('0100aaaa00000000');
    expect(id.cache).toBeNull();
  });

  test('cache storage names its title and index', () => {
    const id = parseSaveId('0100aaaa00000000.cache3');
    expect(id.title).toBe('0100aaaa00000000');
    expect(id.cache).toBe('3');
    expect(id.uid).toBeNull();
  });

  test('a save given to a profile reads back as owned by it', () => {
    const id = parseSaveId(saveIdFor('0100aaaa00000000', 'ffeeddccbbaa99887766554433221100'));
    expect(id.save).toBe('0100aaaa00000000');
    expect(id.uid).toBe('ffeeddccbbaa99887766554433221100');
  });
});
