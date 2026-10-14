// One object URL per Blob, shared by everything showing it and revoked with its last user.

const held = new Map<Blob, { url: string; users: number }>();

export function holdUrl(blob: Blob): string {
  const entry = held.get(blob);
  if (entry) {
    entry.users++;
    return entry.url;
  }
  const url = URL.createObjectURL(blob);
  held.set(blob, { url, users: 1 });
  return url;
}

export function releaseUrl(blob: Blob): void {
  const entry = held.get(blob);
  if (!entry || --entry.users > 0) return;
  URL.revokeObjectURL(entry.url);
  held.delete(blob);
}
