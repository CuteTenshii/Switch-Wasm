// A store-only zip writer for exporting folders; no compression, no zip64.

import type { Bytes } from '../shared/protocol';

export interface ZipEntry {
  // Relative, `/`-separated; a directory ends with `/`.
  name: string;
  data: Bytes;
}

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(data: Bytes): number {
  let c = 0xFFFFFFFF;
  for (const byte of data) c = CRC_TABLE[(c ^ byte) & 0xFF] ^ (c >>> 8);
  return (c ^ 0xFFFFFFFF) >>> 0;
}

// Throws past 4 GiB or 65535 entries, which need zip64.
export function zip(entries: ZipEntry[]): Blob {
  if (entries.length > 0xFFFF) throw new Error('too many entries for a zip without zip64');
  const encoder = new TextEncoder();
  const parts: BlobPart[] = [];
  const central: Uint8Array<ArrayBuffer>[] = [];
  let offset = 0;
  for (const entry of entries) {
    const name = encoder.encode(entry.name);
    const crc = crc32(entry.data);
    const size = entry.data.length;
    if (offset + size > 0xFFFFFFFF) throw new Error('export is larger than 4 GiB');

    const local = new DataView(new ArrayBuffer(30));
    local.setUint32(0, 0x04034B50, true);
    local.setUint16(4, 20, true);
    // Bit 11: the name is UTF-8.
    local.setUint16(6, 0x0800, true);
    local.setUint32(14, crc, true);
    local.setUint32(18, size, true);
    local.setUint32(22, size, true);
    local.setUint16(26, name.length, true);
    parts.push(local.buffer, name, entry.data);

    const record = new DataView(new ArrayBuffer(46 + name.length));
    record.setUint32(0, 0x02014B50, true);
    record.setUint16(4, 20, true);
    record.setUint16(6, 20, true);
    record.setUint16(8, 0x0800, true);
    record.setUint32(16, crc, true);
    record.setUint32(20, size, true);
    record.setUint32(24, size, true);
    record.setUint16(28, name.length, true);
    // Directory attribute for `/`-terminated names.
    record.setUint32(38, entry.name.endsWith('/') ? 0x10 : 0, true);
    record.setUint32(42, offset, true);
    new Uint8Array(record.buffer).set(name, 46);
    central.push(new Uint8Array(record.buffer));

    offset += 30 + name.length + size;
  }
  const centralSize = central.reduce((n, r) => n + r.length, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054B50, true);
  end.setUint16(8, entries.length, true);
  end.setUint16(10, entries.length, true);
  end.setUint32(12, centralSize, true);
  end.setUint32(16, offset, true);
  return new Blob([...parts, ...central, end.buffer], { type: 'application/zip' });
}
