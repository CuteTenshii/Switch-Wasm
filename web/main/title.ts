// The running title, shown on the top bar and in the tab.
import { $ } from './dom';
import { holdUrl, releaseUrl } from './objecturl';

export interface RunningTitle {
  name: string;
  publisher: string;
  icon: Blob | null;
  // NACP version, or the applied update's; empty when none.
  version: string;
}

const rootEl = $('running');
const iconEl = $<HTMLImageElement>('running-icon');
const nameEl = $('running-name');
const versionEl = $('running-version');

const PAGE_TITLE = document.title;
const SHORT_TITLE = PAGE_TITLE.split(' - ')[0];

let icon: Blob | null = null;
let iconUrl: string | null = null;

// Pass `null` when the session is discarded.
export function setRunning(title: RunningTitle | null): void {
  if (icon) releaseUrl(icon);
  icon = title?.icon ?? null;
  iconUrl = icon ? holdUrl(icon) : null;
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
    ? [title.name, title.version ? 'v' + title.version : '', title.publisher, SHORT_TITLE]
        .filter(Boolean)
        .join(' \u00b7 ')
    : PAGE_TITLE;
}

// Revoked when the next title replaces it, so don't keep it.
export function runningIconUrl(): string | null {
  return iconUrl;
}
