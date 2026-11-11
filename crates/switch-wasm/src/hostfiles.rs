//! Host files the core reads by range: the open container and the system data archives.
//! A `Blob` is read in chunks with `FileReaderSync`, which only a worker has, and the chunks
//! are held in `SharedArrayBuffer`s outside linear memory, which the guest shares.

// Only the browser build reads; host builds compile it for the exports that register files.
#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]

use js_sys::{Array, Reflect, Uint8Array};
use switch_core::chunk_cache::ChunkIndex;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{Blob, File, FileReaderSync};

use crate::SyncCell;

const CHUNK: usize = 1 << 20;
const CACHE_CHUNKS: usize = 2048;

/// Reads out of one host file since the last [`take_io`].
#[derive(Default)]
pub(crate) struct Io {
    pub file: String,
    pub reads: u64,
    pub bytes: u64,
    pub chunk_misses: u64,
    pub disk_bytes: u64,
    pub failures: u64,
    pub last_failure: String,
}

struct HostFile {
    blob: Blob,
    io: Option<Io>,
}

struct Hosts {
    /// Slot 0 is the container being run, replaced by [`open_container`].
    files: Vec<Option<HostFile>>,
    index: ChunkIndex,
    /// One `SharedArrayBuffer` per cache slot, created when the slot is first used.
    slots: Vec<Option<Uint8Array>>,
    reader: Option<FileReaderSync>,
}

// SAFETY: only the system worker calls exports; see the `SESSION` comment.
static HOSTS: SyncCell<Option<Hosts>> = SyncCell::new(None);

fn hosts() -> &'static mut Hosts {
    // SAFETY: as above.
    unsafe { &mut *HOSTS.get() }.get_or_insert_with(|| Hosts {
        files: Vec::new(),
        index: ChunkIndex::new(CACHE_CHUNKS),
        slots: Vec::new(),
        reader: None,
    })
}

/// Make `blob` the container, returning its size.
pub(crate) fn open_container(blob: &Blob) -> u64 {
    let h = hosts();
    h.index.forget_file(0);
    if h.files.is_empty() {
        h.files.push(None);
    }
    h.files[0] = Some(HostFile {
        blob: blob.clone(),
        io: None,
    });
    blob.size() as u64
}

/// Register `blob` as the next host file, returning its index. Index 0 stays the container's.
pub(crate) fn add(blob: &Blob) -> u32 {
    let h = hosts();
    if h.files.is_empty() {
        h.files.push(None);
    }
    h.files.push(Some(HostFile {
        blob: blob.clone(),
        io: None,
    }));
    (h.files.len() - 1) as u32
}

impl Hosts {
    fn io(&mut self, file: u32) -> &mut Io {
        let entry = self.files[file as usize].as_mut().expect("file exists");
        entry.io.get_or_insert_with(Io::default)
    }

    fn reader(&mut self) -> Result<&FileReaderSync, JsValue> {
        if self.reader.is_none() {
            self.reader = Some(FileReaderSync::new()?);
        }
        Ok(self.reader.as_ref().expect("just set"))
    }

    fn read_blob(&mut self, file: u32, start: u64, end: u64) -> Result<Uint8Array, JsValue> {
        let blob = self.files[file as usize]
            .as_ref()
            .expect("file exists")
            .blob
            .clone();
        let part = blob.slice_with_f64_and_f64(start as f64, end as f64)?;
        let buffer = self.reader()?.read_as_array_buffer(&part)?;
        self.io(file).disk_bytes += end - start;
        Ok(Uint8Array::new(&buffer))
    }

    /// The cache slot's bytes, or `None` if the browser will not give another buffer.
    fn slot_view(&mut self, slot: usize) -> Option<&Uint8Array> {
        if self.slots.len() <= slot {
            self.slots.resize(slot + 1, None);
        }
        if self.slots[slot].is_none() {
            let ctor = Reflect::get(&js_sys::global(), &"SharedArrayBuffer".into()).ok()?;
            let ctor: &js_sys::Function = ctor.dyn_ref()?;
            let args = Array::of1(&(CHUNK as f64).into());
            let buffer = Reflect::construct(ctor, &args).ok()?;
            self.slots[slot] = Some(Uint8Array::new(&buffer));
        }
        self.slots[slot].as_ref()
    }

    /// The slot holding `chunk`, fetching it and what follows if a sequential reader is
    /// likely to want them. `None` when the cache cannot hold it.
    fn chunk(&mut self, file: u32, size: u64, chunk: u64) -> Result<Option<usize>, JsValue> {
        if let Some(slot) = self.index.get(file, chunk) {
            return Ok(Some(slot));
        }
        self.io(file).chunk_misses += 1;
        let slots = self.index.miss(file, chunk, size.div_ceil(CHUNK as u64));
        let start = chunk * CHUNK as u64;
        let end = (start + (slots.len() * CHUNK) as u64).min(size);
        let span = match self.read_blob(file, start, end) {
            Ok(span) => span,
            Err(e) => {
                for i in 0..slots.len() as u64 {
                    self.index.forget(file, chunk + i);
                }
                return Err(e);
            }
        };
        for (i, &slot) in slots.iter().enumerate() {
            let lo = (i * CHUNK) as u32;
            let len = (end - start - u64::from(lo)).min(CHUNK as u64) as u32;
            let from = Uint8Array::new_with_byte_offset_and_length(&span.buffer(), lo, len);
            match self.slot_view(slot) {
                Some(view) => view.set(&from, 0),
                None => {
                    for j in i..slots.len() {
                        self.index.retire(file, chunk + j as u64);
                    }
                    return Ok((i > 0).then_some(slots[0]));
                }
            }
        }
        Ok(Some(slots[0]))
    }

    fn read(
        &mut self,
        file: u32,
        size: u64,
        offset: u64,
        out: &mut [u8],
    ) -> Result<usize, JsValue> {
        if out.len() > CHUNK {
            let span = self.read_blob(file, offset, offset + out.len() as u64)?;
            span.copy_to(out);
            return Ok(out.len());
        }
        let mut done = 0;
        while done < out.len() {
            let at = offset + done as u64;
            let chunk = at / CHUNK as u64;
            let from = (at % CHUNK as u64) as usize;
            let Some(slot) = self.chunk(file, size, chunk)? else {
                let span = self.read_blob(file, at, offset + out.len() as u64)?;
                span.copy_to(&mut out[done..]);
                return Ok(out.len());
            };
            let held = (size - chunk * CHUNK as u64).min(CHUNK as u64) as usize;
            let take = (held - from).min(out.len() - done);
            let view = self.slots[slot].as_ref().expect("held slots are backed");
            view.subarray(from as u32, (from + take) as u32)
                .copy_to(&mut out[done..done + take]);
            done += take;
        }
        Ok(done)
    }
}

/// Fill `out` from `offset` of host file `file`, returning how many bytes were filled.
pub(crate) fn read(file: u32, offset: u64, out: &mut [u8]) -> usize {
    let h = hosts();
    let Some(Some(entry)) = h.files.get(file as usize) else {
        return 0;
    };
    let size = entry.blob.size() as u64;
    if offset >= size || out.is_empty() {
        return 0;
    }
    let want = out.len().min((size - offset) as usize);
    h.io(file).reads += 1;
    match h.read(file, size, offset, &mut out[..want]) {
        Ok(got) => {
            h.io(file).bytes += got as u64;
            got
        }
        Err(e) => {
            let io = h.io(file);
            io.failures += 1;
            io.last_failure = format!("read at {offset}: {}", describe(&e));
            0
        }
    }
}

fn describe(e: &JsValue) -> String {
    e.dyn_ref::<js_sys::Error>()
        .map_or_else(|| format!("{e:?}"), |e| String::from(e.message()))
}

/// What was read from each host file since the last call.
pub(crate) fn take_io() -> Vec<Io> {
    let mut taken = Vec::new();
    for (index, file) in hosts().files.iter_mut().enumerate() {
        let Some(file) = file else { continue };
        let Some(mut io) = file.io.take() else {
            continue;
        };
        io.file = file.blob.dyn_ref::<File>().map_or_else(
            || match index {
                0 => "the container".to_owned(),
                _ => format!("host file {index}"),
            },
            File::name,
        );
        taken.push(io);
    }
    taken
}
