export { fmtSize } from '../shared/format';

// Coarser than `fmtSize`, since it updates several times a second.
export function formatBytes(n: number): string {
  if (n >= 1024 * 1024 * 1024) return (n / (1024 * 1024 * 1024)).toFixed(2) + ' GiB';
  if (n >= 1024 * 1024) return (n / (1024 * 1024)).toFixed(1) + ' MiB';
  if (n >= 1024) return (n / 1024).toFixed(0) + ' KiB';
  return n + ' B';
}
