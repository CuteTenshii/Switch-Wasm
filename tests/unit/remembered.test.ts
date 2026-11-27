import { afterEach, describe, expect, test } from 'bun:test';
import { remembered } from '../../web/main/remembered';

class MemoryStorage {
  values = new Map<string, string>();

  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value);
  }
}

function useStorage(storage: unknown): void {
  Object.assign(globalThis, { localStorage: storage });
}

afterEach(() => {
  Reflect.deleteProperty(globalThis, 'localStorage');
});

describe('remembered', () => {
  test('a written value is read back, by another reader of the same key too', () => {
    useStorage(new MemoryStorage());
    remembered('key', () => ({ n: 0 })).write({ n: 4 });
    expect(remembered('key', () => ({ n: 0 })).read()).toEqual({ n: 4 });
  });

  test('a key never written reads as the fallback', () => {
    useStorage(new MemoryStorage());
    expect(remembered('missing', () => [1]).read()).toEqual([1]);
  });

  test('a value that is not JSON reads as the fallback', () => {
    const storage = new MemoryStorage();
    storage.values.set('broken', '{');
    useStorage(storage);
    expect(remembered('broken', () => 'fallback').read()).toBe('fallback');
  });

  test('without a store, reads give the fallback and writes are dropped', () => {
    const memory = remembered('key', () => 7);
    memory.write(8);
    expect(memory.read()).toBe(7);
  });

  test('a store that refuses writes loses only the value', () => {
    useStorage({
      getItem: () => null,
      setItem: () => {
        throw new Error('quota');
      },
    });
    const memory = remembered('key', () => 'none');
    expect(() => memory.write('value')).not.toThrow();
    expect(memory.read()).toBe('none');
  });
});
