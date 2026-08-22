/* What the bundler adds to the module system.

   Declared by hand rather than by referencing vite/client, whose types assume a
   DOM: half of this frontend is a worker, and that half compiles against the
   WebWorker library instead. */

/** `?url` gives back the built asset's URL - hashed, and correct in both the
 *  dev server and a built site - instead of the literal path written here. */
declare module '*?url' {
  const url: string;
  export default url;
}

/** What Vite substitutes into the bundle. `DEV`/`PROD` are constants folded in
 *  at build time, so a branch on one of them is dead code the minifier drops
 *  rather than a runtime check. */
interface ImportMetaEnv {
  readonly MODE: string;
  readonly BASE_URL: string;
  readonly DEV: boolean;
  readonly PROD: boolean;
  readonly SSR: boolean;
  /** Anything the environment exports as `VITE_*`. Nothing reads one today. */
  readonly [key: string]: string | boolean | undefined;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
