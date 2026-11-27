// Stored save ids, as the core names them: `<save id>[@<user uid>]`, or `<title id>.cache<index>`
// for a title's cache storage.

export interface SaveId {
  // The id without its owner.
  save: string;
  // The owning profile's uid, or null for a save every profile shares.
  uid: string | null;
  // The title the save belongs to, and its cache storage index, or null for a save.
  title: string;
  cache: string | null;
}

const CACHE = '.cache';

export function parseSaveId(id: string): SaveId {
  const at = id.indexOf('@');
  const save = at < 0 ? id : id.slice(0, at);
  const dot = save.indexOf(CACHE);
  return {
    save,
    uid: at < 0 ? null : id.slice(at + 1),
    title: dot < 0 ? save : save.slice(0, dot),
    cache: dot < 0 ? null : save.slice(dot + CACHE.length),
  };
}

/** The stored id of `save` (an id without its owner) once `uid` owns it. */
export function saveIdFor(save: string, uid: string): string {
  return save + '@' + uid;
}
