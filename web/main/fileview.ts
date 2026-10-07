// Reading and editing one stored file: text in an editor, images shown, anything else as hex.

import type { Bytes } from '../shared/protocol';
import { el } from './dom';
import { fmtSize } from './format';

// Larger text opens read-only: a textarea that size is slow to edit.
const EDIT_LIMIT = 4 * 1024 * 1024;
const HEX_BYTES = 4096;

export interface View {
  name: string;
  data: Bytes;
  save(data: Bytes): Promise<void>;
  download(): void;
  close(): void;
}

function asText(data: Bytes): string | null {
  if (data.includes(0)) return null;
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(data);
  } catch {
    return null;
  }
}

function imageType(data: Bytes): string | null {
  const at = (bytes: number[], from = 0) => bytes.every((b, i) => data[from + i] === b);
  if (at([0x89, 0x50, 0x4E, 0x47])) return 'image/png';
  if (at([0xFF, 0xD8, 0xFF])) return 'image/jpeg';
  if (at([0x47, 0x49, 0x46, 0x38])) return 'image/gif';
  if (at([0x52, 0x49, 0x46, 0x46]) && at([0x57, 0x45, 0x42, 0x50], 8)) return 'image/webp';
  if (at([0x42, 0x4D])) return 'image/bmp';
  return null;
}

function hexDump(data: Bytes): string {
  const lines: string[] = [];
  for (let at = 0; at < Math.min(data.length, HEX_BYTES); at += 16) {
    const row = data.subarray(at, at + 16);
    const hex = [...row].map((b) => b.toString(16).padStart(2, '0')).join(' ');
    const text = [...row].map((b) => (b >= 0x20 && b < 0x7F ? String.fromCharCode(b) : '.')).join('');
    lines.push(at.toString(16).padStart(8, '0') + '  ' + hex.padEnd(47) + '  ' + text);
  }
  return lines.join('\n');
}

// Fills `host`; returns whether there are unsaved edits, for the dialog to ask before closing.
export function showView(host: HTMLElement, view: View, tools: HTMLElement): () => boolean {
  host.textContent = '';
  tools.textContent = '';
  const info = el('p', 'files-view-info muted tiny');
  host.appendChild(info);

  const button = (label: string, run: () => void, primary = false) => {
    const b = el('button', 'btn small' + (primary ? ' primary' : ''), label);
    b.type = 'button';
    b.addEventListener('click', run);
    tools.appendChild(b);
    return b;
  };

  const text = asText(view.data);
  const image = text === null ? imageType(view.data) : null;
  let dirty = false;

  if (text !== null && view.data.length <= EDIT_LIMIT) {
    info.textContent = 'Text, ' + fmtSize(view.data.length) + '. Edits are stored when you save.';
    const area = el('textarea', 'files-editor');
    area.value = text;
    area.spellcheck = false;
    area.setAttribute('aria-label', 'Contents of ' + view.name);
    const save = button('Save', () => {
      save.disabled = true;
      void view.save(new TextEncoder().encode(area.value)).then(() => {
        dirty = false;
        info.textContent = 'Saved, ' + fmtSize(new TextEncoder().encode(area.value).length) + '.';
      }, (err: unknown) => {
        save.disabled = false;
        info.textContent = 'Could not save: ' + String((err as Error).message || err);
      });
    }, true);
    save.disabled = true;
    area.addEventListener('input', () => {
      dirty = true;
      save.disabled = false;
    });
    host.appendChild(area);
    area.focus();
  } else if (text !== null) {
    info.textContent = 'Text, ' + fmtSize(view.data.length) + ': too large to edit here, shown read-only.';
    host.appendChild(el('pre', 'files-hex', text));
  } else if (image) {
    info.textContent = image.slice(6).toUpperCase() + ' image, ' + fmtSize(view.data.length) + '.';
    const url = URL.createObjectURL(new Blob([view.data], { type: image }));
    const img = el('img', 'files-image');
    img.alt = view.name;
    img.src = url;
    img.addEventListener('load', () => URL.revokeObjectURL(url), { once: true });
    host.appendChild(img);
  } else {
    info.textContent = 'Binary, ' + fmtSize(view.data.length)
      + (view.data.length > HEX_BYTES ? '. The first ' + fmtSize(HEX_BYTES) + ' as hex.' : ', as hex.');
    host.appendChild(el('pre', 'files-hex', hexDump(view.data)));
  }

  button('Download', () => view.download());
  button('Close', () => view.close());
  return () => dirty;
}
