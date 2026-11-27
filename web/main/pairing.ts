// Updates and add-on content, held for the title they belong to and applied when it launches.
// The files do not survive a reload; what each title was last launched with does, as a reminder.

import type { DlcEntry, UpdateInfo } from '../shared/protocol';
import { remembered } from './remembered';

export interface HeldUpdate {
  file: File;
  titleId: string;
  version: string;
}

export interface HeldDlc {
  file: File;
  titleId: string;
  indices: number[];
}

interface RememberedUpdate {
  name: string;
  version: string;
}

let heldUpdate: HeldUpdate | null = null;
let heldDlc: HeldDlc[] = [];

const updateMemory = remembered<Record<string, RememberedUpdate>>('switch-wasm:updates', () => ({}));
const dlcMemory = remembered<Record<string, string[]>>('switch-wasm:dlc', () => ({}));

/** Hold `file`, an update, for its title's next launch, in place of any other update. */
export function holdUpdate(file: File, { title: titleId, version }: UpdateInfo): HeldUpdate {
  heldUpdate = { file, titleId, version };
  const all = updateMemory.read();
  all[titleId] = { name: file.name, version };
  updateMemory.write(all);
  return heldUpdate;
}

/** Hold `file`'s add-on content `packs` for their title's next launch, replacing a file of that name. */
export function holdDlc(file: File, packs: DlcEntry[]): HeldDlc {
  const held = { file, titleId: packs[0]?.title_id ?? '', indices: packs.map((e) => e.index) };
  heldDlc = [...heldDlc.filter((h) => h.file.name !== file.name), held];
  const all = dlcMemory.read();
  all[held.titleId] = [...new Set([...all[held.titleId] ?? [], file.name])];
  dlcMemory.write(all);
  return held;
}

/** What a launch of `titleId` takes: the held update for it, and how many add-on packs. */
export function extrasFor(titleId: string): { update: HeldUpdate | null; packs: number } {
  return {
    update: heldUpdate?.titleId === titleId ? heldUpdate : null,
    packs: heldDlc.filter((h) => h.titleId === titleId).reduce((n, h) => n + h.indices.length, 0),
  };
}

export function describeDlc(packs: number): string {
  return packs === 1 ? '1 add-on content pack' : packs + ' add-on content packs';
}

export function describeUpdate(update: { version: string; name: string }): string {
  return update.version ? 'v' + update.version : update.name;
}

/** What is held, and for which title, as told under the open title (`''` for none). */
export function pairedNotes(openTitleId: string): string[] {
  const lines: string[] = [];
  if (heldUpdate) {
    const what = 'Update ' + describeUpdate({ version: heldUpdate.version, name: heldUpdate.file.name });
    lines.push(heldUpdate.titleId === openTitleId
      ? what + ' - applied when this title launches.'
      : what + ' held for ' + heldUpdate.titleId + ' - open that game to launch it patched.');
  } else {
    const known = openTitleId ? updateMemory.read()[openTitleId] : undefined;
    if (known) {
      lines.push('This title was last launched with update ' + describeUpdate(known)
        + ' (' + known.name + '). Drop that file again to launch it patched.');
    }
  }

  const mine = heldDlc.filter((held) => held.titleId === openTitleId);
  const { packs } = extrasFor(openTitleId);
  if (packs) {
    lines.push(describeDlc(packs) + ' - mounted when this title launches ('
      + mine.map((held) => held.file.name).join(', ') + ').');
  }
  for (const held of heldDlc.filter((h) => h.titleId !== openTitleId)) {
    lines.push(describeDlc(held.indices.length) + ' held for ' + held.titleId
      + ' - open that game to launch it with them.');
  }
  if (!packs && openTitleId) {
    const known = dlcMemory.read()[openTitleId] ?? [];
    const missing = known.filter((name) => !heldDlc.some((h) => h.file.name === name));
    if (missing.length) {
      lines.push('This title was last launched with add-on content ('
        + missing.join(', ') + '). Drop those files again to launch it with them.');
    }
  }
  return lines;
}
