// Finding the NCAs a firmware dump holds, in a folder or a .zip.

// What a folder or .zip holds: the NCAs to install, and what was left out.
export interface FirmwareFound {
  ncas: FirmwareNca[];
  // Files that are not NCAs.
  ignored: number;
  failures: string[];
}

// An NCA to install, read only when its turn comes.
export interface FirmwareNca {
  name: string;
  open: () => Promise<Blob>;
}

// `folder` takes only the NCAs directly in the folder each path starts with, and the pieces of a
// split one (`<id>.nca/00`); a .zip's are taken at any depth.
export function gatherNcas<T>(
  items: { path: string; item: T }[],
  open: (pieces: T[]) => Promise<Blob>,
  folder = false,
): FirmwareFound {
  const found = new Map<string, { name: string; pieces: { order: number; item: T }[] }>();
  let ignored = 0;
  for (const { path, item } of items) {
    const parts = path.split('/');
    const split = parts.length === 3 && /\.nca$/i.test(parts[1]);
    if (folder && parts.length > 2 && !split) continue;
    const base = parts[parts.length - 1];
    const parent = parts.length > 1 ? parts[parts.length - 2] : '';
    let key: string;
    let name: string;
    let order = 0;
    if (/\.nca$/i.test(base)) {
      key = path;
      name = base;
    } else if (/\.nca$/i.test(parent) && /^\d{2}$/.test(base)) {
      key = parts.slice(0, -1).join('/');
      name = parent;
      order = Number(base);
    } else {
      ignored++;
      continue;
    }
    const nca = found.get(key) ?? { name, pieces: [] };
    nca.pieces.push({ order, item });
    found.set(key, nca);
  }
  const ncas: FirmwareNca[] = [];
  const failures: string[] = [];
  for (const { name, pieces } of found.values()) {
    pieces.sort((a, b) => a.order - b.order);
    if (pieces.some((piece, i) => piece.order !== i)) {
      failures.push(name + ' (some of its pieces are missing)');
      continue;
    }
    ncas.push({ name, open: () => open(pieces.map((piece) => piece.item)) });
  }
  return { ncas, ignored, failures };
}
