// Each opened title's NACP and icon, so ns can describe it to other programs.

import type { Bytes } from '../shared/protocol';
import { idbApply, idbGetAll, NAND_CONTROLS, nandIdb } from './db';
import { log } from './log';
import { call } from './rpc';

interface StoredControl {
  nacp: Bytes;
  icon: Bytes;
}

// Store the control data the session has just read for `titleId`.
export async function controlRemember(titleId: string, icon: Bytes | null): Promise<void> {
  try {
    const nacp = await call('control_nacp');
    if (!nacp.length) return;
    const entry: StoredControl = { nacp, icon: icon ?? new Uint8Array(0) };
    await idbApply(await nandIdb(), NAND_CONTROLS, [[titleId, entry]]);
  } catch (err) {
    log('Title details: could not be stored (' + (err as Error).message + ')', 'err');
  }
}

export async function controlRestore(): Promise<void> {
  let entries: [string, StoredControl][];
  try {
    entries = await idbGetAll<StoredControl>(await nandIdb(), NAND_CONTROLS);
  } catch (err) {
    log('Title details: could not be read (' + (err as Error).message + ')', 'err');
    return;
  }
  for (const [id, control] of entries) {
    await call('add_application_control', id, control.nacp, control.icon);
  }
}
