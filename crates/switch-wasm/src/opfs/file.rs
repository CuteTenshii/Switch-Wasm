//! An OPFS file as the core's [`HostFile`]. Opening a handle is a promise and guest file
//! access cannot wait for one, so a file starts closed, read from a snapshot, and its first
//! writes wait in memory until the next round opens it ([`OpfsFile::settle`]).

use std::cell::RefCell;
use std::rc::{Rc, Weak};

use js_sys::Uint8Array;
use switch_core::source::ByteSource;
use switch_core::vfs::HostFile;
use wasm_bindgen::JsValue;
use web_sys::{Blob, FileReaderSync, FileSystemFileHandle, FileSystemSyncAccessHandle};

enum State {
    Closed {
        snapshot: Blob,
        /// Bytes of the snapshot still current: a shrink cuts it, and growing again reads zeros.
        valid: u64,
        len: u64,
        /// Writes since the snapshot, applied over it in order.
        pending: Vec<(u64, Vec<u8>)>,
    },
    Open {
        access: FileSystemSyncAccessHandle,
        dirty: bool,
    },
}

#[derive(Debug)]
pub(crate) struct OpfsFile {
    handle: FileSystemFileHandle,
    state: RefCell<State>,
}

impl std::fmt::Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            State::Closed { len, pending, .. } => f
                .debug_struct("Closed")
                .field("len", len)
                .field("pending", &pending.len())
                .finish(),
            State::Open { dirty, .. } => f.debug_struct("Open").field("dirty", dirty).finish(),
        }
    }
}

thread_local! {
    /// Every file handed to the core, so a round can open the ones written to and flush the rest.
    static FILES: RefCell<Vec<Weak<OpfsFile>>> = const { RefCell::new(Vec::new()) };
    static READER: RefCell<Option<FileReaderSync>> = const { RefCell::new(None) };
}

fn io(err: JsValue) -> switch_core::Error {
    switch_core::Error::Io(format!("OPFS: {err:?}"))
}

/// Read `out.len()` bytes of `blob` from `at`; the caller keeps the range inside it.
fn read_snapshot(blob: &Blob, at: u64, out: &mut [u8]) -> Result<(), JsValue> {
    let part = blob.slice_with_f64_and_f64(at as f64, (at + out.len() as u64) as f64)?;
    let bytes = READER.with_borrow_mut(|reader| -> Result<js_sys::ArrayBuffer, JsValue> {
        if reader.is_none() {
            *reader = Some(FileReaderSync::new()?);
        }
        reader
            .as_ref()
            .expect("just set")
            .read_as_array_buffer(&part)
    })?;
    Uint8Array::new(&bytes).copy_to(out);
    Ok(())
}

impl OpfsFile {
    fn track(file: OpfsFile) -> Rc<OpfsFile> {
        let file = Rc::new(file);
        FILES.with_borrow_mut(|files| files.push(Rc::downgrade(&file)));
        file
    }

    /// A file read from `snapshot`, its contents as `handle` last showed them.
    pub(crate) fn closed(handle: FileSystemFileHandle, snapshot: Blob) -> Rc<OpfsFile> {
        let len = snapshot.size() as u64;
        Self::track(OpfsFile {
            handle,
            state: RefCell::new(State::Closed {
                snapshot,
                valid: len,
                len,
                pending: Vec::new(),
            }),
        })
    }

    /// A file already open through `access`.
    pub(crate) fn open(
        handle: FileSystemFileHandle,
        access: FileSystemSyncAccessHandle,
    ) -> Rc<OpfsFile> {
        Self::track(OpfsFile {
            handle,
            state: RefCell::new(State::Open {
                access,
                dirty: false,
            }),
        })
    }

    /// Every live file the core holds.
    pub(crate) fn all() -> Vec<Rc<OpfsFile>> {
        FILES.with_borrow_mut(|files| {
            files.retain(|file| file.strong_count() > 0);
            files.iter().filter_map(Weak::upgrade).collect()
        })
    }

    /// Whether writes wait in memory for a handle.
    pub(crate) fn waiting(&self) -> bool {
        matches!(&*self.state.borrow(), State::Closed { pending, valid, len, .. }
            if !pending.is_empty() || valid != len)
    }

    /// Open a file whose writes wait, and store them. One the core let go of meanwhile is
    /// closed again, so its entry can be removed.
    pub(crate) async fn settle(file: &Rc<OpfsFile>) -> Result<(), JsValue> {
        if !file.waiting() {
            return Ok(());
        }
        let access = super::open(&file.handle).await?;
        let mut state = file.state.borrow_mut();
        let State::Closed {
            valid,
            len,
            pending,
            ..
        } = &*state
        else {
            access.close();
            return Ok(());
        };
        let stored = (|| -> Result<(), JsValue> {
            access.truncate_with_f64(*valid as f64)?;
            access.truncate_with_f64(*len as f64)?;
            for (at, data) in pending {
                super::write_at(&access, *at, data)?;
            }
            access.flush()
        })();
        if let Err(err) = stored {
            access.close();
            return Err(err);
        }
        *state = State::Open {
            access,
            dirty: false,
        };
        drop(state);
        // Only this round holds it: the guest deleted it meanwhile.
        if Rc::strong_count(file) == 1 {
            file.close();
        }
        Ok(())
    }

    /// Flush what was written through the handle since the last flush.
    pub(crate) fn flush(&self) -> Result<(), JsValue> {
        if let State::Open { access, dirty } = &mut *self.state.borrow_mut() {
            if *dirty {
                access.flush()?;
                *dirty = false;
            }
        }
        Ok(())
    }

    fn close(&self) {
        if let State::Open { access, .. } = &*self.state.borrow() {
            access.flush().ok();
            access.close();
        }
    }
}

impl Drop for OpfsFile {
    fn drop(&mut self) {
        self.close();
    }
}

impl ByteSource for OpfsFile {
    fn len(&self) -> u64 {
        match &*self.state.borrow() {
            State::Closed { len, .. } => *len,
            State::Open { access, .. } => access.get_size().map_or(0, |size| size as u64),
        }
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<usize, switch_core::Error> {
        match &*self.state.borrow() {
            State::Open { access, .. } => super::read_at(access, offset, out).map_err(io),
            State::Closed {
                snapshot,
                valid,
                len,
                pending,
            } => {
                if offset >= *len {
                    return Ok(0);
                }
                let n = out.len().min((*len - offset) as usize);
                let out = &mut out[..n];
                let from_snapshot = valid.saturating_sub(offset).min(n as u64) as usize;
                read_snapshot(snapshot, offset, &mut out[..from_snapshot]).map_err(io)?;
                out[from_snapshot..].fill(0);
                let end = offset + n as u64;
                for (at, data) in pending {
                    let (start, stop) = ((*at).max(offset), (at + data.len() as u64).min(end));
                    if start < stop {
                        out[(start - offset) as usize..(stop - offset) as usize]
                            .copy_from_slice(&data[(start - at) as usize..(stop - at) as usize]);
                    }
                }
                Ok(n)
            }
        }
    }
}

impl HostFile for OpfsFile {
    fn write_at(&self, offset: u64, data: &[u8]) -> Result<(), switch_core::Error> {
        match &mut *self.state.borrow_mut() {
            State::Open { access, dirty } => {
                super::write_at(access, offset, data).map_err(io)?;
                *dirty = true;
            }
            State::Closed { len, pending, .. } => {
                *len = (*len).max(offset + data.len() as u64);
                pending.push((offset, data.to_vec()));
            }
        }
        Ok(())
    }

    fn set_len(&self, new_len: u64) -> Result<(), switch_core::Error> {
        match &mut *self.state.borrow_mut() {
            State::Open { access, dirty } => {
                access.truncate_with_f64(new_len as f64).map_err(io)?;
                *dirty = true;
            }
            State::Closed {
                valid,
                len,
                pending,
                ..
            } => {
                *valid = (*valid).min(new_len);
                *len = new_len;
                pending.retain_mut(|(at, data)| {
                    data.truncate(new_len.saturating_sub(*at) as usize);
                    !data.is_empty()
                });
            }
        }
        Ok(())
    }
}
