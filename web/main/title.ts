// The running title, shown on the top bar and in the tab.
import { $ } from './dom';
import type { KnownTitle } from './titles';

const rootEl = $('running');
const iconEl = $<HTMLImageElement>('running-icon');
const nameEl = $('running-name');
const versionEl = $('running-version');

const APP_NAME = 'switch-wasm';
// The page's own `<title>`, shown while nothing runs.
const PAGE_TITLE = document.title;

let iconUrl: string | null = null;

// `version` is the applied update when there is one. Pass `null` when the session is discarded.
export function setRunning(title: KnownTitle | null): void {
  iconUrl = title?.iconUrl ?? null;
  rootEl.hidden = !title;
  rootEl.title = title
    ? title.name + (title.version ? ' - v' + title.version : '')
    : '';
  nameEl.textContent = title?.name || '';
  versionEl.textContent = title?.version ? 'v' + title.version : '';
  versionEl.hidden = !title?.version;
  iconEl.hidden = !iconUrl;
  // Lets the narrow layout drop the name only when an icon remains.
  rootEl.classList.toggle('has-icon', Boolean(iconUrl));
  if (iconUrl) iconEl.src = iconUrl;
  else iconEl.removeAttribute('src');
  document.title = title
    ? [title.name, title.version ? 'v' + title.version : '', title.publisher, APP_NAME]
        .filter(Boolean)
        .join(' \u00b7 ')
    : PAGE_TITLE;
}

export function runningIconUrl(): string | null {
  return iconUrl;
}
