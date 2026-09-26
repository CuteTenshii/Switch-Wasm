/** File and container sizes, to two decimals: the numbers next to a name in
 *  a file list, where the exact figure is what someone is checking. Shared
 *  because the worker's console lines report sizes too. */
export function fmtSize(n: number): string {
  if (n >= 1 << 30) return (n / (1 << 30)).toFixed(2) + ' GiB';
  if (n >= 1 << 20) return (n / (1 << 20)).toFixed(2) + ' MiB';
  if (n >= 1 << 10) return (n / (1 << 10)).toFixed(1) + ' KiB';
  return n + ' B';
}
