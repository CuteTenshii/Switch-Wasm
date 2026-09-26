export { fmtSize } from '../shared/format';

/** Live memory figures for the status bar, rounded harder: this one updates
 *  several times a second, and a digit that flickers is a digit nobody can
 *  read. */
export function formatBytes(n: number): string {
  if (n >= 1024 * 1024 * 1024) return (n / (1024 * 1024 * 1024)).toFixed(2) + ' GiB';
  if (n >= 1024 * 1024) return (n / (1024 * 1024)).toFixed(1) + ' MiB';
  if (n >= 1024) return (n / 1024).toFixed(0) + ' KiB';
  return n + ' B';
}
