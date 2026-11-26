// Check that the browser's WebCodecs decoder takes the VP9 streams `vp9/rebuild.rs` writes:
// decode every `x.rebuilt.ivf` in a directory and compare each frame with `x.rebuilt.yuv`.
//   cargo run --release -p switch-core --example vp9_rebuild_check -- <dir> --out <dir>
//   node tools/vp9_webcodecs_check.mjs <dir> [--headless]
import { readdirSync, readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join } from 'node:path';
import { commandLine, fail, launchChromium } from './common.mjs';

const CODEC = 'vp09.00.61.08';
const cli = commandLine();
const [dir] = cli.positional;
if (!dir) fail('usage: node tools/vp9_webcodecs_check.mjs <dir> [--headless]');
const clips = readdirSync(dir)
  .filter((name) => name.endsWith('.rebuilt.ivf'))
  .sort();
if (clips.length === 0) fail(`no .rebuilt.ivf in ${dir}; run the vp9_rebuild_check example with --out`);

const half = (n) => Math.ceil(n / 2);
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

/** The chunks of an IVF file, each marked key when its first frame is a key frame. */
function readIvf(path) {
  const ivf = readFileSync(path);
  const chunks = [];
  let at = 32;
  while (at + 12 <= ivf.length) {
    const size = ivf.readUInt32LE(at);
    at += 12;
    const data = ivf.subarray(at, at + size);
    at += size;
    // Neither a frame shown again (bit 3) nor an inter frame (bit 2).
    chunks.push({ key: (data[0] & 0x0c) === 0, data: [...data] });
  }
  return { width: ivf.readUInt16LE(12), height: ivf.readUInt16LE(14), chunks };
}

/** Runs in the page: decodes `chunks` and hashes each frame's visible planes as tight I420. */
async function decodeInPage({ chunks, codec }) {
  const frames = [];
  let error = null;
  const decoder = new VideoDecoder({
    output: (frame) => frames.push(frame),
    error: (e) => (error = String(e)),
  });
  decoder.configure({ codec, hardwareAcceleration: 'no-preference', optimizeForLatency: true });
  chunks.forEach((chunk, i) =>
    decoder.decode(
      new EncodedVideoChunk({ type: chunk.key ? 'key' : 'delta', timestamp: i, data: new Uint8Array(chunk.data) }),
    ),
  );
  try {
    await decoder.flush();
  } catch (e) {
    error ??= String(e);
  }
  const hashes = [];
  let format = null;
  for (const frame of frames) {
    format = frame.format;
    const { width, height } = frame.visibleRect;
    const [cw, ch] = [Math.ceil(width / 2), Math.ceil(height / 2)];
    const buffer = new Uint8Array(frame.allocationSize());
    const layout = await frame.copyTo(buffer);
    frame.close();
    const out = new Uint8Array(width * height + 2 * cw * ch);
    const row = (plane, y, length) => {
      const at = layout[plane].offset + y * layout[plane].stride;
      return buffer.subarray(at, at + length);
    };
    for (let y = 0; y < height; y++) out.set(row(0, y, width), y * width);
    const [u, v] = [width * height, width * height + cw * ch];
    for (let y = 0; y < ch; y++) {
      if (format === 'NV12') {
        const uv = row(1, y, 2 * cw);
        for (let x = 0; x < cw; x++) {
          out[u + y * cw + x] = uv[2 * x];
          out[v + y * cw + x] = uv[2 * x + 1];
        }
      } else {
        out.set(row(1, y, cw), u + y * cw);
        out.set(row(2, y, cw), v + y * cw);
      }
    }
    const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', out));
    hashes.push(Array.from(digest, (b) => b.toString(16).padStart(2, '0')).join(''));
  }
  return { hashes, error, format };
}

const browser = await launchChromium({ webGpu: false, headless: cli.has('headless') });
const page = await browser.newPage();
// WebCodecs is only exposed to secure contexts.
await page.route('https://check.invalid/', (route) => route.fulfill({ contentType: 'text/html', body: '<!doctype html>' }));
await page.goto('https://check.invalid/');

let failed = false;
for (const name of clips) {
  const { width, height, chunks } = readIvf(join(dir, name));
  const expected = readFileSync(join(dir, name.replace(/\.ivf$/, '.yuv')));
  const size = width * height + 2 * half(width) * half(height);
  const got = await page.evaluate(decodeInPage, { chunks, codec: CODEC });
  const frames = expected.length / size;
  let mismatch = -1;
  for (let i = 0; i < frames && mismatch < 0; i++) {
    if (got.hashes[i] !== sha256(expected.subarray(i * size, (i + 1) * size))) mismatch = i;
  }
  const ok = !got.error && got.hashes.length === frames && mismatch < 0;
  failed ||= !ok;
  const verdict = ok ? 'all match' : `frame ${mismatch} differs${got.error ? `, decoder error: ${got.error}` : ''}`;
  console.log(`${name}: ${got.hashes.length} of ${frames} frames as ${got.format}, ${verdict}`);
}
await browser.close();
process.exit(failed ? 1 : 0);
