/* What is running, named on the top bar.

   Every launch path already knows which title it is starting - the loading
   screen is handed its name and icon - but that screen goes away on the first
   presented frame and used to take the page's only statement of what booted
   with it. This keeps it, on the bar and in the tab, for as long as the
   session holds that title. */

import { $ } from './dom';

/** What is loaded into the running session. A title out of a container has the
 *  name and icon its home menu would show; homebrew loaded as a bare
 *  executable has only the file it came from. */
export interface RunningTitle {
  name: string;
  icon: Blob | null;
  /** The version the NACP declares, or the update's where one was applied.
   *  Empty for homebrew and for the titles that set none. */
  version: string;
}

const rootEl = $('running');
const iconEl = $<HTMLImageElement>('running-icon');
const nameEl = $('running-name');
const versionEl = $('running-version');

const PAGE_TITLE = document.title;
// "switch-wasm" out of "switch-wasm - a Nintendo Switch emulator...": a tab
// naming the title has no room for the tagline as well, and the short form is
// already written once in index.html rather than twice.
const SHORT_TITLE = PAGE_TITLE.split(' - ')[0];

// This module's own URL for the icon, not the one `container.ts` made for the
// title card: that one is revoked the moment another container is opened, and
// what is running outlives the card it was launched from.
let iconUrl: string | null = null;

/** Say what the session is running, or `null` for a console with nothing in
 *  it. Every path that loads a program calls this, and so does every path that
 *  throws the session away. */
export function setRunning(title: RunningTitle | null): void {
  if (iconUrl) URL.revokeObjectURL(iconUrl);
  iconUrl = title?.icon ? URL.createObjectURL(title.icon) : null;
  rootEl.hidden = !title;
  // The bar truncates a long name, so the whole of it stays readable somewhere.
  rootEl.title = title
    ? title.name + (title.version ? ' - v' + title.version : '')
    : '';
  nameEl.textContent = title?.name || '';
  versionEl.textContent = title?.version ? 'v' + title.version : '';
  versionEl.hidden = !title?.version;
  iconEl.hidden = !iconUrl;
  // Marked on the element rather than inferred from the image, so the narrow
  // layout can drop the name only where there is an icon left to identify it.
  rootEl.classList.toggle('has-icon', Boolean(iconUrl));
  if (iconUrl) iconEl.src = iconUrl;
  else iconEl.removeAttribute('src');
  document.title = title ? title.name + ' - ' + SHORT_TITLE : PAGE_TITLE;
}

/** The running title's icon, for a screen that wants to show the same one.
 *  The URL belongs to this module, which revokes it when the next title
 *  replaces it - so read it again rather than keeping it. */
export function runningIconUrl(): string | null {
  return iconUrl;
}
