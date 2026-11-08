import { describe, expect, test } from 'bun:test';
import { deflateRawSync } from 'node:zlib';
import { crc32 } from '../../web/shared/crc32';
import { zipExtract, zipMembers } from '../../web/main/unzip';
import { zip } from '../../web/worker/zip';

const bytes = (text: string) => new TextEncoder().encode(text);

// The message `promise` rejects with; fails if it resolves.
async function rejection(promise: Promise<unknown>): Promise<string> {
  try {
    await promise;
  } catch (err) {
    return (err as Error).message;
  }
  throw new Error('expected a rejection');
}

async function text(blob: Blob): Promise<string> {
  return new TextDecoder().decode(await blob.arrayBuffer());
}

// One deflated member, as other tools write them; `crc` overrides the stored checksum.
function deflatedZip(name: string, data: Uint8Array, { crc = crc32(data), flags = 0 } = {}): Blob {
  const packed = deflateRawSync(data);
  const nameBytes = bytes(name);
  const local = new DataView(new ArrayBuffer(30));
  local.setUint32(0, 0x04034B50, true);
  local.setUint16(6, flags, true);
  local.setUint16(8, 8, true);
  local.setUint32(14, crc, true);
  local.setUint32(18, packed.length, true);
  local.setUint32(22, data.length, true);
  local.setUint16(26, nameBytes.length, true);
  const central = new DataView(new ArrayBuffer(46));
  central.setUint32(0, 0x02014B50, true);
  central.setUint16(8, flags, true);
  central.setUint16(10, 8, true);
  central.setUint32(16, crc, true);
  central.setUint32(20, packed.length, true);
  central.setUint32(24, data.length, true);
  central.setUint16(28, nameBytes.length, true);
  const centralAt = 30 + nameBytes.length + packed.length;
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054B50, true);
  end.setUint16(8, 1, true);
  end.setUint16(10, 1, true);
  end.setUint32(12, 46 + nameBytes.length, true);
  end.setUint32(16, centralAt, true);
  return new Blob([local.buffer, nameBytes, packed, central.buffer, nameBytes, end.buffer]);
}

describe('zip and unzip', () => {
  test('a written archive lists and extracts every entry as it went in', async () => {
    const archive = zip([
      { name: 'save/', data: new Uint8Array(0) },
      { name: 'save/main.lua', data: bytes('print("hi")\n') },
      { name: 'save/ünïcode.txt', data: bytes('names are UTF-8') },
      { name: 'empty.bin', data: new Uint8Array(0) },
    ]);
    const members = await zipMembers(archive);
    expect(members.map((m) => [m.path, m.size, m.method])).toEqual([
      ['save/', 0, 0],
      ['save/main.lua', 12, 0],
      ['save/ünïcode.txt', 15, 0],
      ['empty.bin', 0, 0],
    ]);
    expect(await text(await zipExtract(archive, members[1]))).toBe('print("hi")\n');
    expect(await text(await zipExtract(archive, members[2]))).toBe('names are UTF-8');
    expect((await zipExtract(archive, members[3])).size).toBe(0);
  });

  test('an archive with nothing in it has no members', async () => {
    expect(await zipMembers(zip([]))).toEqual([]);
  });

  test('a deflated member is inflated and checked', async () => {
    const data = bytes('compress me '.repeat(200));
    const archive = deflatedZip('big.txt', data);
    const [member] = await zipMembers(archive);
    expect(member.compressedSize).toBeLessThan(data.length);
    expect(await text(await zipExtract(archive, member))).toBe('compress me '.repeat(200));
  });

  test('a member whose contents fail the CRC is refused', async () => {
    const archive = deflatedZip('bad.txt', bytes('payload'), { crc: 0x12345678 });
    const [member] = await zipMembers(archive);
    expect(await rejection(zipExtract(archive, member))).toBe('its contents fail the CRC check');
  });

  test('an encrypted member is refused', async () => {
    const archive = deflatedZip('secret.txt', bytes('payload'), { flags: 1 });
    const [member] = await zipMembers(archive);
    expect(await rejection(zipExtract(archive, member))).toBe('it is encrypted');
  });

  test('something that is not a zip says so', async () => {
    expect(await rejection(zipMembers(new Blob([bytes('definitely not a zip archive')])))).toBe('it is not a zip archive');
  });
});
