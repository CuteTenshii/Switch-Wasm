// Identifies files handed to the page by their header, not their name.

export type FileFormat = 'nro' | 'elf' | 'pfs0' | 'xci' | 'nca';

export type Verdict<F extends FileFormat = FileFormat> =
  | { ok: true; format: F }
  | { ok: false; why: string };

const FORMAT_NAME: Record<FileFormat, string> = {
  nro: 'NRO',
  elf: 'ELF',
  pfs0: 'NSP',
  xci: 'XCI',
  nca: 'NCA',
};

// Covers the NCA magic at 0x200, the deepest of these.
const HEAD_LEN = 0x204;

function magicAt(data: Uint8Array, offset: number, magic: string): boolean {
  if (offset + magic.length > data.length) return false;
  for (let i = 0; i < magic.length; i++) {
    if (data[offset + i] !== magic.charCodeAt(i)) return false;
  }
  return true;
}

export async function identify(file: File): Promise<FileFormat | null> {
  const data = new Uint8Array(await file.slice(0, HEAD_LEN).arrayBuffer());
  if (magicAt(data, 0, '\x7FELF')) return 'elf';
  // Scan for NRO0 like `NroHeader::parse`: some builds prepend a boot stub.
  for (let at = 0; at + 4 <= Math.min(data.length, 0x100); at++) {
    if (magicAt(data, at, 'NRO0')) return 'nro';
  }
  if (magicAt(data, 0, 'PFS0')) return 'pfs0';
  if (magicAt(data, 0x100, 'HEAD')) return 'xci';
  if (['NCA3', 'NCA2', 'NCA0'].some((magic) => magicAt(data, 0x200, magic))) return 'nca';
  return null;
}

function nameList(formats: readonly FileFormat[]): string {
  const names = formats.map((f) => FORMAT_NAME[f]);
  return names.length < 2
    ? names.join('')
    : names.slice(0, -1).join(', ') + ' or ' + names[names.length - 1];
}

// Identify `file` and check it is in `accept`; `hint` says where a misplaced format goes.
export async function classify<F extends FileFormat>(
  file: File,
  accept: readonly F[],
  hint?: string,
): Promise<Verdict<F>> {
  let format: FileFormat | null;
  try {
    format = await identify(file);
  } catch (err) {
    return { ok: false, why: 'Could not read ' + file.name + ': ' + (err as Error).message };
  }
  // A CDN NCA header stays encrypted until prod.keys is loaded, so trust the name.
  if (!format && /\.nca$/i.test(file.name)) format = 'nca';
  if (!format) {
    return {
      ok: false,
      why: file.name + ' is not a format this reads - expected ' + nameList(accept) + '.',
    };
  }
  if (!(accept as readonly FileFormat[]).includes(format)) {
    const why = 'an ' + FORMAT_NAME[format] + '; this takes ' + nameList(accept);
    return { ok: false, why: file.name + ' is ' + why + '.' + (hint ? ' ' + hint : '') };
  }
  return { ok: true, format: format as F };
}

// A keys file is only `name = hex` lines.
const KEYS_LINE = /^[ \t]*[0-9A-Za-z_]+[ \t]*=[ \t]*[0-9A-Fa-f]{16,}[ \t]*$/m;
const KEYS_MAX_BYTES = 1 << 20;

export async function readKeysFile(file: File): Promise<string | null> {
  if (file.size === 0 || file.size > KEYS_MAX_BYTES) return null;
  const text = await file.text();
  return KEYS_LINE.test(text) ? text : null;
}
