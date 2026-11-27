// Counts the worker's undestroyed device textures and buffers by label, for
// `browser_boot.mjs --gpu-objects`. A browser frees a dropped one only once it is collected,
// so a label whose count keeps growing is being dropped without `destroy()`.

export const GPU_OBJECTS_MARK = '@@gpu-objects ';

const TEXEL_BYTES = {
  'r8unorm': 1, 'r8uint': 1, 'rg8unorm': 2, 'r16float': 2, 'r16uint': 2, 'depth16unorm': 2,
  'rgba8unorm': 4, 'rgba8unorm-srgb': 4, 'bgra8unorm': 4, 'bgra8unorm-srgb': 4, 'rgba8uint': 4,
  'rgb10a2unorm': 4, 'rg11b10ufloat': 4, 'rg16float': 4, 'r32float': 4, 'r32uint': 4, 'r32sint': 4,
  'depth32float': 4, 'depth24plus': 4, 'depth24plus-stencil8': 4, 'stencil8': 1,
  'rgba16float': 8, 'rgba16uint': 8, 'rg32float': 8, 'rg32uint': 8, 'depth32float-stencil8': 8,
  'rgba32float': 16, 'rgba32uint': 16,
  'bc1-rgba-unorm': 0.5, 'bc1-rgba-unorm-srgb': 0.5, 'bc4-r-unorm': 0.5, 'bc4-r-snorm': 0.5,
};

/** Script run in the worker before the page's own: logs the undestroyed objects every `everyMs`. */
export const hookSource = (everyMs) => `
if (typeof GPUDevice !== 'undefined' && typeof WorkerGlobalScope !== 'undefined' && !self.countsGpuObjects) {
  self.countsGpuObjects = true;
  const texelBytes = ${JSON.stringify(TEXEL_BYTES)};
  // Weak, so counting them does not keep the dropped ones from being collected.
  const live = new Set();
  const destroyed = new WeakSet();
  let made = 0;
  const track = (object, kind, label, bytes) => {
    made += bytes;
    live.add({ ref: new WeakRef(object), name: kind + ' ' + (label || '(no label)'), bytes });
    return object;
  };
  const createTexture = GPUDevice.prototype.createTexture;
  GPUDevice.prototype.createTexture = function (d) {
    const s = Array.isArray(d.size)
      ? { width: d.size[0], height: d.size[1] ?? 1, layers: d.size[2] ?? 1 }
      : { width: d.size.width, height: d.size.height ?? 1, layers: d.size.depthOrArrayLayers ?? 1 };
    // Compressed formats not listed count as one byte a texel; mip levels add a third at most.
    const texel = texelBytes[d.format] ?? 1;
    const levels = (d.mipLevelCount ?? 1) > 1 ? 4 / 3 : 1;
    const bytes = s.width * s.height * s.layers * texel * (d.sampleCount ?? 1) * levels;
    return track(createTexture.call(this, d), 'texture', d.label, bytes);
  };
  const createBuffer = GPUDevice.prototype.createBuffer;
  GPUDevice.prototype.createBuffer = function (d) {
    return track(createBuffer.call(this, d), 'buffer', d.label, d.size);
  };
  for (const type of [GPUTexture, GPUBuffer]) {
    const destroy = type.prototype.destroy;
    type.prototype.destroy = function () {
      destroyed.add(this);
      return destroy.call(this);
    };
  }
  setInterval(() => {
    const by = new Map();
    let collected = 0;
    for (const entry of live) {
      const object = entry.ref.deref();
      if (!object || destroyed.has(object)) {
        collected += !object;
        live.delete(entry);
        continue;
      }
      const [count, bytes] = by.get(entry.name) ?? [0, 0];
      by.set(entry.name, [count + 1, bytes + entry.bytes]);
    }
    const mib = (bytes) => (bytes / 1048576).toFixed(1);
    const rows = [...by].sort((a, b) => b[1][1] - a[1][1]);
    const total = rows.reduce((sum, [, [, bytes]]) => sum + bytes, 0);
    console.log('${GPU_OBJECTS_MARK}' + mib(total) + ' MiB undestroyed, ' + collected
      + ' collected undestroyed and ' + mib(made) + ' MiB made since the last count; '
      + rows.slice(0, 10).map(([name, [count, bytes]]) => name + ' x' + count + ' ' + mib(bytes) + ' MiB').join(', '));
    made = 0;
  }, ${everyMs});
}
`;
