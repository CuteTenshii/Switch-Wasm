// The titles seen this session, so a title read again (listed, then launched) shows one icon URL.

import type { Bytes } from '../shared/protocol';

/** A game or homebrew as the page shows it. */
export interface KnownTitle {
  // Empty for homebrew.
  id: string;
  name: string;
  publisher: string;
  // The NACP display version ("1.0.1"); may be empty.
  version: string;
  iconUrl: string | null;
}

// What the core says about a title, from a peek or its control data, in the core's own names.
interface TitleDetails {
  title_id?: string;
  name?: string;
  publisher?: string;
  version?: string;
  // Empty or absent where the icon is a JPEG, as the console stores them.
  icon_mime?: string;
}

interface TitleIcon {
  bytes: Bytes;
  type: string;
  url: string;
}

// URLs are kept for the session: a title's icon may still be on screen after it is read again.
const icons = new Map<string, TitleIcon>();

// Homebrew has no title id: a peek says "", its control data all zeros.
function titleId(details: TitleDetails): string {
  const id = details.title_id ?? '';
  return /^0*$/.test(id) ? '' : id.toLowerCase();
}

// Without a title id, homebrew is told apart by its NACP.
function titleKey(details: TitleDetails): string {
  return titleId(details) || [details.name ?? '', details.version ?? ''].join('\u0000');
}

function sameBytes(a: Bytes, b: Bytes): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function iconUrl(key: string, bytes: Bytes, type: string): string {
  const known = icons.get(key);
  if (known && known.type === type && sameBytes(known.bytes, bytes)) return known.url;
  const url = URL.createObjectURL(new Blob([bytes], { type }));
  icons.set(key, { bytes, type, url });
  return url;
}

/** The title `details` describe, with the icon URL it already has when the same icon was seen for it before. */
export function knownTitle(details: TitleDetails, icon: Bytes | null): KnownTitle {
  return {
    id: titleId(details),
    name: details.name ?? '',
    publisher: details.publisher ?? '',
    version: details.version ?? '',
    iconUrl: icon && icon.length ? iconUrl(titleKey(details), icon, details.icon_mime || 'image/jpeg') : null,
  };
}
