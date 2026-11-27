// Small JSON values kept in `localStorage`. A blocked, full or missing store only loses them.

function storage(): Storage | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

export interface Remembered<T> {
  read(): T;
  write(value: T): void;
}

/** The JSON value under `key`, or `fallback` while there is none or it can't be read. */
export function remembered<T>(key: string, fallback: () => T): Remembered<T> {
  return {
    read() {
      try {
        const text = storage()?.getItem(key);
        return text ? JSON.parse(text) as T : fallback();
      } catch {
        return fallback();
      }
    },
    write(value) {
      try {
        storage()?.setItem(key, JSON.stringify(value));
      } catch {
        // Only the reminder is lost.
      }
    },
  };
}
