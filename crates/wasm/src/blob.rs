//! Reading a `File` or `Blob` from the worker.

use js_sys::Uint8Array;
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;
use web_sys::Blob;

/// Most of a `Blob` asked for at once, so no copy of the whole file sits in JavaScript.
const CHUNK: usize = 8 << 20;

/// All of `blob` in linear memory.
pub(crate) async fn read_vec(blob: &Blob) -> Result<Vec<u8>, JsValue> {
    let size = blob.size() as usize;
    let mut out = vec![0u8; size];
    let mut at = 0;
    while at < size {
        let end = (at + CHUNK).min(size);
        let part = blob.slice_with_f64_and_f64(at as f64, end as f64)?;
        let buffer = JsFuture::from(part.array_buffer()).await?;
        Uint8Array::new(&buffer).copy_to(&mut out[at..end]);
        at = end;
    }
    Ok(out)
}
