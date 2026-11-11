// Reads back device textures of a given size from the worker, for `browser_boot.mjs --dump-texture`.
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const TEXEL_BYTES = {
  'r8unorm': 1, 'rg8unorm': 2, 'rgba8unorm': 4, 'rgba8unorm-srgb': 4, 'bgra8unorm': 4, 'bgra8unorm-srgb': 4,
  'rgb10a2unorm': 4, 'r16float': 2, 'rg16float': 4, 'rgba16float': 8, 'r32float': 4, 'rg32float': 8, 'rgba32float': 16,
};
const KEPT = 8;

export const parseSpec = (text) => {
  const match = /^(\d+)x(\d+)(?:x(\d+))?$/.exec(text);
  if (!match) throw new Error(`--dump-texture=dir@${text}: expected WxH or WxHxLAYERS`);
  return { width: Number(match[1]), height: Number(match[2]), layers: match[3] ? Number(match[3]) : 0 };
};

/** Script run in the worker before the page's own: remembers matching textures, readable back. */
export const hookSource = (spec) => `
if (typeof GPUDevice !== 'undefined' && typeof WorkerGlobalScope !== 'undefined' && !self.tracksTextures) {
  self.tracksTextures = true;
  self.trackedTextures = [];
  const create = GPUDevice.prototype.createTexture;
  GPUDevice.prototype.createTexture = function (descriptor) {
    const size = Array.isArray(descriptor.size)
      ? { width: descriptor.size[0], height: descriptor.size[1] ?? 1, layers: descriptor.size[2] ?? 1 }
      : { width: descriptor.size.width, height: descriptor.size.height ?? 1, layers: descriptor.size.depthOrArrayLayers ?? 1 };
    const want = ${JSON.stringify(spec)};
    if (size.width !== want.width || size.height !== want.height || (want.layers && size.layers !== want.layers)) {
      return create.call(this, descriptor);
    }
    const texture = create.call(this, { ...descriptor, usage: descriptor.usage | GPUTextureUsage.COPY_SRC });
    self.trackedTextures.push({ texture, device: this, format: descriptor.format, ...size });
    if (self.trackedTextures.length > ${KEPT}) self.trackedTextures.shift();
    return texture;
  };
}
`;

/** Every tracked texture's layers, tightly packed, as base64. */
export const readTextures = (worker) =>
  worker.evaluate(async (texelBytes) => {
    const out = [];
    for (const { texture, device, format, width, height, layers } of self.trackedTextures ?? []) {
      const bytes = texelBytes[format];
      if (!bytes) {
        out.push({ format, width, height, layers, unsupported: true });
        continue;
      }
      const row = Math.ceil((width * bytes) / 256) * 256;
      const buffer = device.createBuffer({
        size: row * height * layers,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const encoder = device.createCommandEncoder();
      encoder.copyTextureToBuffer(
        { texture },
        { buffer, bytesPerRow: row, rowsPerImage: height },
        [width, height, layers],
      );
      device.queue.submit([encoder.finish()]);
      await buffer.mapAsync(GPUMapMode.READ);
      const padded = new Uint8Array(buffer.getMappedRange());
      const packed = new Uint8Array(width * bytes * height * layers);
      for (let y = 0; y < height * layers; y++) {
        packed.set(padded.subarray(y * row, y * row + width * bytes), y * width * bytes);
      }
      buffer.destroy();
      let text = '';
      for (let at = 0; at < packed.length; at += 0x8000) {
        text += String.fromCharCode(...packed.subarray(at, at + 0x8000));
      }
      out.push({ format, width, height, layers, data: btoa(text) });
    }
    return out;
  }, TEXEL_BYTES);

/** Raw bytes and, for 8-bit colour, a PPM per layer; one line of statistics per layer. */
export const writeTextures = (dir, textures, say) => {
  mkdirSync(dir, { recursive: true });
  textures.forEach((t, index) => {
    const name = `${index}-${t.format}-${t.width}x${t.height}x${t.layers}`;
    if (t.unsupported) {
      say(`texture ${name}: format not read back`);
      return;
    }
    const data = Buffer.from(t.data, 'base64');
    writeFileSync(join(dir, `${name}.bin`), data);
    const layerBytes = t.width * t.height * (data.length / (t.width * t.height * t.layers));
    for (let layer = 0; layer < t.layers; layer++) {
      const slice = data.subarray(layer * layerBytes, (layer + 1) * layerBytes);
      const sum = slice.reduce((total, byte) => total + byte, 0);
      say(`texture ${name} layer ${layer}: mean byte ${(sum / slice.length).toFixed(1)}, nonzero ${slice.filter((byte) => byte).length}/${slice.length}`);
      if (/^(rgba|bgra)8/.test(t.format)) {
        const bgra = t.format.startsWith('bgra');
        const rgb = Buffer.alloc(t.width * t.height * 3);
        for (let i = 0; i < t.width * t.height; i++) {
          rgb[i * 3] = slice[i * 4 + (bgra ? 2 : 0)];
          rgb[i * 3 + 1] = slice[i * 4 + 1];
          rgb[i * 3 + 2] = slice[i * 4 + (bgra ? 0 : 2)];
        }
        writeFileSync(join(dir, `${name}-layer${layer}.ppm`), Buffer.concat([Buffer.from(`P6\n${t.width} ${t.height}\n255\n`), rgb]));
      }
    }
  });
};
