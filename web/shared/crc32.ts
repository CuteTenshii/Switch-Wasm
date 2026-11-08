// CRC-32 as zip uses it.

const TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

// Pass the previous result as `crc` to continue over data that arrives in pieces.
export function crc32(data: Uint8Array, crc = 0): number {
  let c = (crc ^ 0xFFFFFFFF) >>> 0;
  for (const byte of data) c = TABLE[(c ^ byte) & 0xFF] ^ (c >>> 8);
  return (c ^ 0xFFFFFFFF) >>> 0;
}
