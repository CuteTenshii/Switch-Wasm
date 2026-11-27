import { describe, expect, test } from 'bun:test';
import { gatherNcas } from '../../web/main/firmware';

const blob = (pieces: string[]): Promise<Blob> => Promise.resolve(new Blob(pieces));
const found = (paths: string[], folder: boolean) =>
  gatherNcas(paths.map((path) => ({ path, item: path })), blob, folder);

describe('gatherNcas', () => {
  test('a folder install takes its own NCAs and the pieces of a split one, not sub-folders', async () => {
    const dump = found([
      '21.2.0/a.nca',
      '21.2.0/b.nca/00',
      '21.2.0/b.nca/01',
      '21.2.0/notes.txt',
      '21.2.0/old/c.nca',
      '21.2.0/old/d.nca/00',
      '21.2.0/b.nca/extra/00',
    ], true);
    expect(dump.ncas.map((nca) => nca.name)).toEqual(['a.nca', 'b.nca']);
    expect(await (await dump.ncas[1].open()).text()).toBe('21.2.0/b.nca/0021.2.0/b.nca/01');
    expect(dump.ignored).toBe(1);
    expect(dump.failures).toEqual([]);
  });

  test('a folder whose NCAs are all in a sub-folder installs none', () => {
    const dump = found(['Firmware/21.2.0/a.nca', 'Firmware/21.2.0/b.nca'], true);
    expect(dump.ncas).toEqual([]);
    expect(dump.ignored).toBe(0);
  });

  test('a .zip is taken at any depth', () => {
    const dump = found(['a.nca', 'firmware/b.nca', 'firmware/split/c.nca/00'], false);
    expect(dump.ncas.map((nca) => nca.name)).toEqual(['a.nca', 'b.nca', 'c.nca']);
  });

  test('a split NCA missing a piece is a failure', () => {
    const dump = found(['21.2.0/b.nca/00', '21.2.0/b.nca/02'], true);
    expect(dump.ncas).toEqual([]);
    expect(dump.failures).toEqual(['b.nca (some of its pieces are missing)']);
  });
});
