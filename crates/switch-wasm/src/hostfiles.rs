//! Host files the core reads by range: the open container and the system data archives.
//! Chunks are held in `SharedArrayBuffer`s outside linear memory, which the guest shares, and
//! filled by the reader worker (`web/worker/reader.ts`) in the order they were asked for, so
//! a miss waits for its own chunk only and the read-ahead behind it overlaps emulation. Reads
//! the cache will not hold go through `FileReaderSync` here.

// Only the browser build reads; host builds compile it for the exports that register files.
#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]

use js_sys::{Array, Atomics, Int32Array, Object, Reflect, Uint8Array};
use switch_core::chunk_cache::ChunkIndex;
use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{Blob, File, FileReaderSync, Worker};

use crate::SyncCell;

const CHUNK: usize = 1 << 20;
const CACHE_CHUNKS: usize = 2048;
/// A read this small of a file with nothing cached is a header probe, read as asked: a whole
/// chunk would cost several times as much, and most files are probed once and never read again.
const PROBE_BYTES: usize = 64 << 10;
/// Probes each file gets before its reads fill the cache.
const PROBES: u8 = 2;
/// Longest a read waits for the reader worker before giving up on it.
const WAIT_MS: f64 = 30_000.0;

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
    /// Milliseconds the emulator waited on a read.
    pub stall_ms: f64,
}

struct HostFile {
    blob: Blob,
    io: Option<Io>,
    probes: u8,
}

/// The reader worker and the word per cache slot it answers in: the generation of the
/// request it filled the slot for, or its negation if the read failed.
struct Reader {
    worker: Worker,
    done: Int32Array,
}

struct Hosts {
    /// Slot 0 is the container being run, replaced by [`open_container`].
    files: Vec<Option<HostFile>>,
    index: ChunkIndex,
    /// One `SharedArrayBuffer` per cache slot, created when the slot is first used.
    slots: Vec<Option<Uint8Array>>,
    /// Per slot, the generation of the request still awaited, or 0 once it has arrived.
    awaited: Vec<i32>,
    generation: i32,
    reader: Option<Reader>,
    local: Option<FileReaderSync>,
}

// SAFETY: only the system worker calls exports; see the `SESSION` comment.
static HOSTS: SyncCell<Option<Hosts>> = SyncCell::new(None);

fn hosts() -> &'static mut Hosts {
    // SAFETY: as above.
    unsafe { &mut *HOSTS.get() }.get_or_insert_with(|| Hosts {
        files: Vec::new(),
        index: ChunkIndex::new(CACHE_CHUNKS),
        slots: Vec::new(),
        awaited: Vec::new(),
        generation: 0,
        reader: None,
        local: None,
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
        probes: 0,
    });
    h.send_file(0, blob);
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
        probes: 0,
    }));
    let index = (h.files.len() - 1) as u32;
    h.send_file(index, blob);
    index
}

/// Drop host file `index`, registered by [`add`].
pub(crate) fn forget(index: u32) {
    let h = hosts();
    h.index.forget_file(index);
    h.files[index as usize] = None;
}

impl Hosts {
    fn io(&mut self, file: u32) -> &mut Io {
        let entry = self.files[file as usize].as_mut().expect("file exists");
        entry.io.get_or_insert_with(Io::default)
    }

    fn local(&mut self) -> Result<&FileReaderSync, JsValue> {
        if self.local.is_none() {
            self.local = Some(FileReaderSync::new()?);
        }
        Ok(self.local.as_ref().expect("just set"))
    }

    fn read_blob(&mut self, file: u32, start: u64, end: u64) -> Result<Uint8Array, JsValue> {
        let blob = self.files[file as usize]
            .as_ref()
            .expect("file exists")
            .blob
            .clone();
        let part = blob.slice_with_f64_and_f64(start as f64, end as f64)?;
        let began = now();
        let buffer = self.local()?.read_as_array_buffer(&part)?;
        let io = self.io(file);
        io.stall_ms += now() - began;
        io.disk_bytes += end - start;
        Ok(Uint8Array::new(&buffer))
    }

    fn send(&self, kind: &str, fields: &[(&str, &JsValue)]) {
        let Some(reader) = &self.reader else { return };
        let message = Object::new();
        let _ = Reflect::set(&message, &"type".into(), &kind.into());
        for (key, value) in fields {
            let _ = Reflect::set(&message, &(*key).into(), value);
        }
        let _ = reader.worker.post_message(&message);
    }

    fn send_file(&self, index: u32, blob: &Blob) {
        self.send("file", &[("index", &index.into()), ("blob", blob)]);
    }

    /// The cache slot's bytes, or `None` if the browser will not give another buffer.
    fn slot_view(&mut self, slot: usize) -> Option<&Uint8Array> {
        if self.slots.len() <= slot {
            self.slots.resize(slot + 1, None);
            self.awaited.resize(slot + 1, 0);
        }
        if self.slots[slot].is_none() {
            let buffer = shared_buffer(CHUNK)?;
            self.send("slot", &[("slot", &slot.into()), ("buffer", &buffer)]);
            self.slots[slot] = Some(Uint8Array::new(&buffer));
        }
        self.slots[slot].as_ref()
    }

    /// Block until the reader has filled `slot` for `chunk`.
    fn arrived(&mut self, file: u32, chunk: u64, slot: usize) -> Result<(), JsValue> {
        let want = self.awaited[slot];
        if want == 0 {
            return Ok(());
        }
        let reader = self
            .reader
            .as_ref()
            .expect("a slot is awaited from a reader");
        let word = slot as u32;
        let began = now();
        let outcome = loop {
            let seen = Atomics::load(&reader.done, word)?;
            if seen == want || seen == -want {
                break seen;
            }
            if Atomics::wait_with_timeout(&reader.done, word, seen, WAIT_MS)? == "timed-out" {
                break 0;
            }
        };
        self.awaited[slot] = 0;
        self.io(file).stall_ms += now() - began;
        if outcome == want {
            return Ok(());
        }
        self.index.forget(file, chunk);
        Err(JsValue::from_str(if outcome == 0 {
            "the reader worker did not answer"
        } else {
            "the reader worker could not read it"
        }))
    }

    /// The slot holding `chunk` once it has arrived, asking the reader for it and what follows
    /// if a sequential reader is likely to want them. `None` when the cache cannot hold it.
    fn chunk(&mut self, file: u32, size: u64, chunk: u64) -> Result<Option<usize>, JsValue> {
        if let Some(slot) = self.index.get(file, chunk) {
            self.arrived(file, chunk, slot)?;
            return Ok(Some(slot));
        }
        if self.reader.is_none() {
            return Ok(None);
        }
        self.io(file).chunk_misses += 1;
        let slots = self.index.miss(file, chunk, size.div_ceil(CHUNK as u64));
        let backed = (0..slots.len())
            .find(|&i| self.slot_view(slots[i]).is_none())
            .unwrap_or(slots.len());
        for i in backed..slots.len() {
            self.index.retire(file, chunk + i as u64);
        }
        if backed == 0 {
            return Ok(None);
        }
        let (slot_ids, generations) = (Array::new(), Array::new());
        for &slot in &slots[..backed] {
            self.generation = self.generation % i32::MAX + 1;
            self.awaited[slot] = self.generation;
            slot_ids.push(&slot.into());
            generations.push(&self.generation.into());
        }
        let start = chunk * CHUNK as u64;
        self.io(file).disk_bytes += (backed as u64 * CHUNK as u64).min(size - start);
        self.send(
            "fetch",
            &[
                ("file", &file.into()),
                ("chunk", &(chunk as f64).into()),
                ("size", &(size as f64).into()),
                ("slots", &slot_ids),
                ("generations", &generations),
            ],
        );
        self.arrived(file, chunk, slots[0])?;
        Ok(Some(slots[0]))
    }

    /// Whether a small read at `chunk` is one of the file's first, with nothing there cached.
    fn is_probe(&mut self, file: u32, chunk: u64) -> bool {
        let probes = &mut self.files[file as usize]
            .as_mut()
            .expect("file exists")
            .probes;
        if *probes >= PROBES || self.index.get(file, chunk).is_some() {
            return false;
        }
        *probes += 1;
        true
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
        if out.len() <= PROBE_BYTES && self.is_probe(file, offset / CHUNK as u64) {
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

/// Milliseconds on the worker's clock.
fn now() -> f64 {
    js_sys::global()
        .unchecked_into::<web_sys::WorkerGlobalScope>()
        .performance()
        .map_or(0.0, |p| p.now())
}

fn shared_buffer(len: usize) -> Option<JsValue> {
    let ctor = Reflect::get(&js_sys::global(), &"SharedArrayBuffer".into()).ok()?;
    let ctor: &js_sys::Function = ctor.dyn_ref()?;
    let args = Array::of1(&(len as f64).into());
    Reflect::construct(ctor, &args).ok()
}

/// Hand the reader worker the cache: from here on, chunks are filled by it. Throws if the
/// browser has no `SharedArrayBuffer`.
#[wasm_bindgen]
pub fn switch_set_reader(worker: Worker) -> Result<(), String> {
    let done = shared_buffer(CACHE_CHUNKS * 4).ok_or("no SharedArrayBuffer")?;
    let h = hosts();
    h.reader = Some(Reader {
        worker,
        done: Int32Array::new(&done),
    });
    h.send("done", &[("buffer", &done)]);
    for (index, file) in h.files.iter().enumerate() {
        if let Some(file) = file {
            h.send_file(index as u32, &file.blob);
        }
    }
    Ok(())
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
