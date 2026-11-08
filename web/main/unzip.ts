// Reads a zip by range: the directory, then one member at a time, so the archive is never held whole.

import { crc32 } from '../shared/crc32';

export interface ZipMember {
  // `/`-separated as stored; a directory ends with `/`.
  path: string;
  flags: number;
  method: number;
  crc: number;
  compressedSize: number;
  size: number;
  localOffset: number;
}

const END_RECORD = 0x06054B50;
const ZIP64_LOCATOR = 0x07064B50;
const ZIP64_END_RECORD = 0x06064B50;
const CENTRAL_RECORD = 0x02014B50;
const LOCAL_HEADER = 0x04034B50;
const STORED = 0;
const DEFLATED = 8;

async function view(blob: Blob, start: number, end: number): Promise<DataView> {
  if (start < 0 || end > blob.size || start > end) throw new Error('the zip archive is truncated');
  return new DataView(await blob.slice(start, end).arrayBuffer());
}

function u64(data: DataView, at: number): number {
  const value = data.getBigUint64(at, true);
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('the zip archive has an impossible offset');
  return Number(value);
}

// The directory's entry count, size and offset; zip64 when the classic record overflows.
async function directory(zip: Blob): Promise<{ count: number; size: number; offset: number }> {
  // The end record is 22 bytes followed by a comment of up to 64 KiB.
  const tailStart = Math.max(0, zip.size - (22 + 0xFFFF));
  const tail = await view(zip, tailStart, zip.size);
  let at = -1;
  for (let i = tail.byteLength - 22; i >= 0; i--) {
    if (tail.getUint32(i, true) === END_RECORD) {
      at = i;
      break;
    }
  }
  if (at < 0) throw new Error('it is not a zip archive');
  const count = tail.getUint16(at + 10, true);
  const size = tail.getUint32(at + 12, true);
  const offset = tail.getUint32(at + 16, true);
  if (count !== 0xFFFF && size !== 0xFFFFFFFF && offset !== 0xFFFFFFFF) return { count, size, offset };

  const locator = await view(zip, tailStart + at - 20, tailStart + at);
  if (locator.getUint32(0, true) !== ZIP64_LOCATOR) throw new Error('its zip64 locator is missing');
  const recordAt = u64(locator, 8);
  const record = await view(zip, recordAt, recordAt + 56);
  if (record.getUint32(0, true) !== ZIP64_END_RECORD) throw new Error('its zip64 end record is missing');
  return { count: u64(record, 32), size: u64(record, 40), offset: u64(record, 48) };
}

export async function zipMembers(zip: Blob): Promise<ZipMember[]> {
  const { count, size, offset } = await directory(zip);
  const dir = await view(zip, offset, offset + size);
  const names = new TextDecoder();
  const members: ZipMember[] = [];
  let at = 0;
  for (let i = 0; i < count; i++) {
    if (at + 46 > dir.byteLength || dir.getUint32(at, true) !== CENTRAL_RECORD) {
      throw new Error('its central directory is damaged');
    }
    const nameLength = dir.getUint16(at + 28, true);
    const extraLength = dir.getUint16(at + 30, true);
    const commentLength = dir.getUint16(at + 32, true);
    const member: ZipMember = {
      path: names.decode(new Uint8Array(dir.buffer, dir.byteOffset + at + 46, nameLength)),
      flags: dir.getUint16(at + 8, true),
      method: dir.getUint16(at + 10, true),
      crc: dir.getUint32(at + 16, true),
      compressedSize: dir.getUint32(at + 20, true),
      size: dir.getUint32(at + 24, true),
      localOffset: dir.getUint32(at + 42, true),
    };
    // The zip64 extra holds, in this order, only the fields saturated above.
    let extra = at + 46 + nameLength;
    const extraEnd = extra + extraLength;
    while (extra + 4 <= extraEnd) {
      const id = dir.getUint16(extra, true);
      const length = dir.getUint16(extra + 2, true);
      if (id === 0x0001) {
        let field = extra + 4;
        if (member.size === 0xFFFFFFFF) {
          member.size = u64(dir, field);
          field += 8;
        }
        if (member.compressedSize === 0xFFFFFFFF) {
          member.compressedSize = u64(dir, field);
          field += 8;
        }
        if (member.localOffset === 0xFFFFFFFF) member.localOffset = u64(dir, field);
      }
      extra += 4 + length;
    }
    members.push(member);
    at = extraEnd + commentLength;
  }
  return members;
}

// The member's bytes: a slice of the archive when stored, inflated and checked when deflated.
export async function zipExtract(zip: Blob, member: ZipMember): Promise<Blob> {
  if (member.flags & 1) throw new Error('it is encrypted');
  const local = await view(zip, member.localOffset, member.localOffset + 30);
  if (local.getUint32(0, true) !== LOCAL_HEADER) throw new Error('its local header is damaged');
  const start = member.localOffset + 30 + local.getUint16(26, true) + local.getUint16(28, true);
  if (start + member.compressedSize > zip.size) throw new Error('the zip archive is truncated');
  const data = zip.slice(start, start + member.compressedSize);
  if (member.method === STORED) {
    if (member.compressedSize !== member.size) throw new Error('its stored size does not match');
    return data;
  }
  if (member.method !== DEFLATED) throw new Error(`it uses compression method ${member.method}`);
  let crc = 0;
  const check = new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      crc = crc32(chunk, crc);
      controller.enqueue(chunk);
    },
  });
  const inflated = await new Response(
    data.stream().pipeThrough(new DecompressionStream('deflate-raw')).pipeThrough(check),
  ).blob();
  if (inflated.size !== member.size || crc !== member.crc) throw new Error('its contents fail the CRC check');
  return inflated;
}
