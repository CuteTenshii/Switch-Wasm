import { describe, expect, test } from 'bun:test';
import { bootRefusal, nandBusy, nandOperation, onNandBusy } from '../../web/main/nandbusy';

describe('nandOperation', () => {
  test('refuses a boot only while an operation runs', async () => {
    expect(bootRefusal('Minecraft')).toBeNull();
    let release = (): void => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    const running = nandOperation('Registering system data archives and applets', () => held);
    expect(nandBusy()).toBe('Registering system data archives and applets');
    expect(bootRefusal('Minecraft')).toBe(
      'Minecraft was not started: registering system data archives and applets is still'
      + ' running on the NAND. Start it again once that ends.',
    );
    release();
    await running;
    expect(nandBusy()).toBeNull();
    expect(bootRefusal('Minecraft')).toBeNull();
  });

  test('stays busy until every nested operation ends', async () => {
    await nandOperation('Installing firmware', async () => {
      await nandOperation('Registering system data archives and applets', () => {
        expect(nandBusy()).toBe('Installing firmware');
        return Promise.resolve();
      });
      expect(nandBusy()).toBe('Installing firmware');
    });
    expect(nandBusy()).toBeNull();
  });

  test('ends an operation that fails, and passes its error on', async () => {
    const failing = nandOperation('Erasing the NAND', () => Promise.reject(new Error('quota')));
    expect(failing).rejects.toThrow('quota');
    await failing.catch(() => {});
    expect(nandBusy()).toBeNull();
  });

  test('tells listeners when an operation starts and ends', async () => {
    const seen: (string | null)[] = [];
    onNandBusy(() => seen.push(nandBusy()));
    await nandOperation('Removing a title', () => Promise.resolve());
    expect(seen).toEqual(['Removing a title', null]);
  });
});
