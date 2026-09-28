/** File and container sizes, to two decimals: the numbers next to a name in
 *  a file list, where the exact figure is what someone is checking. Shared
 *  because the worker's console lines report sizes too. */
export function fmtSize(n: number): string {
  if (n >= 1 << 30) return (n / (1 << 30)).toFixed(2) + ' GiB';
  if (n >= 1 << 20) return (n / (1 << 20)).toFixed(2) + ' MiB';
  if (n >= 1 << 10) return (n / (1 << 10)).toFixed(1) + ' KiB';
  return n + ' B';
}

/** A large count the way a person reads one: 12.3M rather than 12345678.
 *  Million, billion and trillion rather than the SI prefixes, because next
 *  to sizes a G reads as bytes. Shared because the status bar and the
 *  worker's thread lines both count instructions. */
export function fmtCount(n: number): string {
  if (n >= 1e12) return (n / 1e12).toFixed(2) + 'T';
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  return Math.round(n).toLocaleString('en');
}
