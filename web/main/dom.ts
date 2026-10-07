// Element lookup and construction. `$` throws on a missing id.
export function $<T extends HTMLElement = HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error('index.html has no #' + id);
  return node as T;
}

// Create an element with a class and text, without innerHTML.
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

export function pickedFile(e: Event): File | null {
  const input = e.target as HTMLInputElement;
  return input.files?.[0] || null;
}
