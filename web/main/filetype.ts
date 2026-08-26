/* What a file the page was handed actually is.

   `accept` on an <input> is a filter the picker's "All files" option defeats,
   and a drop consults it not at all, so a name is not evidence: everything
   entering the page is identified by its header here instead. */

/** The formats anything on the page knows what to do with. */
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

// Recognised, but nothing here can open one - so it is turned away as what it
// is rather than as a PFS0 with bad magic.
const UNREADABLE: Partial<Record<FileFormat, string>> = {
  xci: 'a cartridge image, and only PFS0 containers (.nsp, .nsz) are read here',
};

// Far enough in for the NCA magic at 0x200, the deepest of these.
const HEAD_LEN = 0x204;

function magicAt(data: Uint8Array, offset: number, magic: string): boolean {
  if (offset + magic.length > data.length) return false;
  for (let i = 0; i < magic.length; i++) {
    if (data[offset + i] !== magic.charCodeAt(i)) return false;
  }
  return true;
}

/** The format a file's header declares, or null if nothing here recognises it. */
export async function identify(file: File): Promise<FileFormat | null> {
  const data = new Uint8Array(await file.slice(0, HEAD_LEN).arrayBuffer());
  if (magicAt(data, 0, '\x7FELF')) return 'elf';
  // `NroHeader::parse` scans for NRO0 rather than reading offset 0, because
  // some builds prepend a boot stub; match that or those files look foreign.
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

/** Identify `file` and check it is one of `accept`. `hint` is appended when a
 *  format this page does read was handed to the wrong place, to say where it
 *  does go. */
export async function classify<F extends FileFormat>(
  file: File,
  accept: readonly F[],
  hint?: string,
): Promise<Verdict<F>> {
  let format: FileFormat | null;
  try {
    format = await identify(file);
  } catch (err) {
    // A folder, or a file that has moved since the picker named it.
    return { ok: false, why: 'Could not read ' + file.name + ': ' + (err as Error).message };
  }
  // An NCA from the CDN keeps its header encrypted, so its magic stays
  // invisible until prod.keys decrypts it - the name is all there is to go on.
  if (!format && /\.nca$/i.test(file.name)) format = 'nca';
  if (!format) {
    return {
      ok: false,
      why: file.name + ' is not a format this reads - expected ' + nameList(accept) + '.',
    };
  }
  if (!(accept as readonly FileFormat[]).includes(format)) {
    const unreadable = UNREADABLE[format];
    const why = unreadable || 'an ' + FORMAT_NAME[format] + '; this takes ' + nameList(accept);
    return { ok: false, why: file.name + ' is ' + why + '.' + (!unreadable && hint ? ' ' + hint : '') };
  }
  return { ok: true, format: format as F };
}

// A keys file is `name = hex` lines and nothing else. Without this a binary
// picked by mistake is stored as if it were keys, and every decrypt after it
// fails without ever saying why.
const KEYS_LINE = /^[ \t]*[0-9A-Za-z_]+[ \t]*=[ \t]*[0-9A-Fa-f]{16,}[ \t]*$/m;
const KEYS_MAX_BYTES = 1 << 20;

/** The text of a keys file, or null if `file` is not one. */
export async function readKeysFile(file: File): Promise<string | null> {
  if (file.size === 0 || file.size > KEYS_MAX_BYTES) return null;
  const text = await file.text();
  return KEYS_LINE.test(text) ? text : null;
}
