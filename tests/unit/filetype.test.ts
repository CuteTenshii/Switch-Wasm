import { describe, expect, test } from 'bun:test';
import { classify, identify, readKeysFile } from '../../web/main/filetype';

// A file of `size` zero bytes with `magic` written at `at`.
function fileWith(name: string, at: number, magic: string, size = 0x400): File {
  const data = new Uint8Array(size);
  data.set(new TextEncoder().encode(magic), at);
  return new File([data], name);
}

describe('identify', () => {
  test.each([
    ['elf', 0, '\x7FELF'],
    ['nro', 0x10, 'NRO0'],
    ['nro', 0x80, 'NRO0'],
    ['pfs0', 0, 'PFS0'],
    ['xci', 0x100, 'HEAD'],
    ['nca', 0x200, 'NCA3'],
    ['nca', 0x200, 'NCA2'],
    ['nca', 0x200, 'NCA0'],
  ])('reads %s from its magic at %i', async (format, at, magic) => {
    expect(await identify(fileWith('x.bin', at, magic))).toBe(format as never);
  });

  test('ignores the name and anything else', async () => {
    expect(await identify(fileWith('game.nsp', 0, 'ZIP!'))).toBeNull();
    expect(await identify(new File([], 'empty.nro'))).toBeNull();
  });

  test('only scans the first 0x100 bytes for an NRO header', async () => {
    expect(await identify(fileWith('late.nro', 0x100, 'NRO0'))).toBeNull();
  });
});

describe('classify', () => {
  test('accepts a listed format', async () => {
    expect(await classify(fileWith('a.nsp', 0, 'PFS0'), ['pfs0', 'xci'])).toEqual({ ok: true, format: 'pfs0' });
  });

  test('names what it found, what it takes, and the hint', async () => {
    const verdict = await classify(fileWith('a.elf', 0, '\x7FELF'), ['pfs0', 'xci', 'nca'], 'Open it from the toolbar.');
    expect(verdict).toEqual({ ok: false, why: 'a.elf is an ELF; this takes NSP, XCI or NCA. Open it from the toolbar.' });
  });

  test('refuses an unknown file with the formats it expected', async () => {
    const verdict = await classify(fileWith('notes.txt', 0, 'hello'), ['nro']);
    expect(verdict).toEqual({ ok: false, why: 'notes.txt is not a format this reads - expected NRO.' });
  });

  test('trusts the name of an NCA whose header is still encrypted', async () => {
    expect(await classify(fileWith('Program.NCA', 0, 'junk'), ['nca'])).toEqual({ ok: true, format: 'nca' });
  });
});

describe('readKeysFile', () => {
  const keys = (text: string) => new File([text], 'prod.keys');

  test('takes name = hex lines', async () => {
    const text = 'header_key = 0123456789abcdef0123456789ABCDEF\nmaster_key_00=00112233445566778899aabbccddeeff\n';
    expect(await readKeysFile(keys(text))).toBe(text);
  });

  test('refuses empty, oversized and non-key files', async () => {
    expect(await readKeysFile(keys(''))).toBeNull();
    expect(await readKeysFile(keys('just some notes\n'))).toBeNull();
    expect(await readKeysFile(keys('short = 1234\n'))).toBeNull();
    expect(await readKeysFile(new File([new Uint8Array((1 << 20) + 1)], 'big.keys'))).toBeNull();
  });
});
