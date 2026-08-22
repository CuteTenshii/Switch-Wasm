/** Element lookup and construction.
 *
 *  `$` throws instead of returning null: every id it is asked for is in
 *  index.html, so a miss is markup that no longer matches the code, and
 *  saying so once beats fifty null checks that would never fire. */
export function $<T extends HTMLElement = HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error('index.html has no #' + id);
  return node as T;
}

/** Create an element with a class and text, avoiding innerHTML entirely. */
export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string | null,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/** The file a change/drop event carries, or null. */
export function pickedFile(e: Event): File | null {
  const input = e.target as HTMLInputElement;
  return input.files?.[0] || null;
}
