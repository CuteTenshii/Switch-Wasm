//! nvdec's H.264 and VP9 on the browser's decoder. It lives on the system worker, which runs
//! every SVC, so its pictures land between frames, when the worker's event loop runs.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use switch_core::gpu::multimedia::HostDecoder;
use switch_core::trace::{emit_at, Level};
use switch_core::vp9::Frame;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    CodecState, EncodedVideoChunk, EncodedVideoChunkInit, EncodedVideoChunkType,
    HardwareAcceleration, PlaneLayout, VideoDecoder, VideoDecoderConfig, VideoDecoderInit,
    VideoDecoderSupport, VideoFrame, VideoPixelFormat,
};

#[derive(Debug, Clone, Copy)]
pub struct Codec {
    name: &'static str,
    string: &'static str,
    acceleration: HardwareAcceleration,
}

/// High profile at level 5.1, past anything the console decodes.
pub const H264: Codec = Codec {
    name: "H.264",
    string: "avc1.640033",
    acceleration: HardwareAcceleration::PreferHardware,
};

/// Profile 0 at level 6.1, past anything the console decodes. Software (libvpx) where there is
/// no hardware decoder.
pub const VP9: Codec = Codec {
    name: "VP9",
    string: "vp09.00.61.08",
    acceleration: HardwareAcceleration::NoPreference,
};

thread_local! {
    /// Whether the browser decodes VP9, once [`probe_vp9`] has heard back.
    static VP9_SUPPORTED: Cell<bool> = const { Cell::new(false) };
}

fn config(codec: Codec) -> VideoDecoderConfig {
    let config = VideoDecoderConfig::new(codec.string);
    config.set_hardware_acceleration(codec.acceleration);
    config.set_optimize_for_latency(true);
    config
}

/// Asks the browser whether it decodes VP9, so a stream can go to it from its first key frame.
pub fn probe_vp9() {
    if !js_sys::Reflect::has(&js_sys::global(), &"VideoDecoder".into()).unwrap_or(false) {
        emit_at(
            Level::Info,
            "[video] no WebCodecs here; VP9 is decoded in software",
        );
        return;
    }
    let promise = VideoDecoder::is_config_supported(&config(VP9));
    wasm_bindgen_futures::spawn_local(async move {
        let supported = JsFuture::from(promise).await.is_ok_and(|support| {
            support
                .unchecked_into::<VideoDecoderSupport>()
                .get_supported()
                .unwrap_or(false)
        });
        if !supported {
            emit_at(
                Level::Info,
                "[video] the browser does not decode VP9; it is decoded in software",
            );
        }
        VP9_SUPPORTED.set(supported);
    });
}

/// A browser VP9 decoder for one stream, where the browser has one.
pub fn vp9_decoder() -> Option<Box<dyn HostDecoder>> {
    VP9_SUPPORTED
        .get()
        .then(|| Box::new(WebDecoder::new(VP9)) as Box<dyn HostDecoder>)
}

type Landed = Rc<RefCell<Vec<(u64, Option<Frame>)>>>;
/// The surface each chunk in flight decodes to, by its timestamp.
type InFlight = Rc<RefCell<HashMap<i32, u64>>>;

/// A WebCodecs decoder, opened at the first key frame and again after it fails.
#[derive(Debug)]
pub struct WebDecoder {
    codec: Codec,
    decoder: Option<VideoDecoder>,
    in_flight: InFlight,
    landed: Landed,
    next: i32,
    callbacks: Option<(Closure<dyn FnMut(JsValue)>, Closure<dyn FnMut(VideoFrame)>)>,
}

impl WebDecoder {
    pub fn new(codec: Codec) -> WebDecoder {
        WebDecoder {
            codec,
            decoder: None,
            in_flight: InFlight::default(),
            landed: Landed::default(),
            next: 0,
            callbacks: None,
        }
    }

    fn open(&mut self) -> Result<VideoDecoder, JsValue> {
        let (in_flight, landed) = (self.in_flight.clone(), self.landed.clone());
        let name = self.codec.name;
        let error = Closure::<dyn FnMut(JsValue)>::new(move |why: JsValue| {
            emit_at(
                Level::Warn,
                &format!("[video] the {name} decoder failed: {why:?}"),
            );
            let lost: Vec<u64> = in_flight.borrow_mut().drain().map(|(_, s)| s).collect();
            landed
                .borrow_mut()
                .extend(lost.into_iter().map(|s| (s, None)));
        });
        let (in_flight, landed) = (self.in_flight.clone(), self.landed.clone());
        let output = Closure::<dyn FnMut(VideoFrame)>::new(move |frame: VideoFrame| {
            let Some(surface) = in_flight.borrow_mut().remove(&(frame.timestamp() as i32)) else {
                frame.close();
                return;
            };
            let landed = landed.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let decoded = planes(&frame).await;
                frame.close();
                landed.borrow_mut().push((surface, decoded));
            });
        });
        let decoder = VideoDecoder::new(&VideoDecoderInit::new(
            error.as_ref().unchecked_ref(),
            output.as_ref().unchecked_ref(),
        ))?;
        decoder.configure(&config(self.codec))?;
        self.callbacks = Some((error, output));
        Ok(decoder)
    }

    fn send(&mut self, surface: u64, chunk: &[u8], key: bool) -> Result<(), JsValue> {
        let open = self
            .decoder
            .as_ref()
            .is_some_and(|d| d.state() == CodecState::Configured);
        if !open {
            if !key {
                return Err("a picture before any key frame".into());
            }
            self.decoder = Some(self.open()?);
        }
        let kind = if key {
            EncodedVideoChunkType::Key
        } else {
            EncodedVideoChunkType::Delta
        };
        let data = js_sys::Uint8Array::from(chunk);
        let chunk = EncodedVideoChunk::new(&EncodedVideoChunkInit::new_with_u8_array(
            &data, self.next, kind,
        ))?;
        self.in_flight.borrow_mut().insert(self.next, surface);
        self.next = self.next.wrapping_add(1);
        if let Some(decoder) = &self.decoder {
            decoder.decode(&chunk)?;
        }
        Ok(())
    }
}

impl HostDecoder for WebDecoder {
    fn decode(&mut self, surface: u64, chunk: &[u8], key: bool) {
        if let Err(why) = self.send(surface, chunk, key) {
            emit_at(
                Level::Warn,
                &format!(
                    "[video] {} picture for {surface:#x} not decoded: {why:?}",
                    self.codec.name
                ),
            );
            self.landed.borrow_mut().push((surface, None));
        }
    }

    fn take(&mut self) -> Vec<(u64, Option<Frame>)> {
        std::mem::take(&mut self.landed.borrow_mut())
    }
}

/// The frame's visible picture as three planes, from the NV12 a hardware decoder gives or the
/// I420 of a software one, each plane where the copy's layout puts it.
async fn planes(frame: &VideoFrame) -> Option<Frame> {
    let format = frame.format()?;
    let rect = frame.visible_rect()?;
    let (width, height) = (rect.width() as usize, rect.height() as usize);
    let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
    let buffer = js_sys::Uint8Array::new_with_length(frame.allocation_size().ok()?);
    let layout: js_sys::Array = JsFuture::from(frame.copy_to_with_buffer_source(&buffer))
        .await
        .ok()?
        .dyn_into()
        .ok()?;
    let bytes = buffer.to_vec();
    let plane = |i: u32| {
        let layout: PlaneLayout = layout.get(i).unchecked_into();
        (layout.get_offset() as usize, layout.get_stride() as usize)
    };
    let row = |(offset, stride): (usize, usize), y: usize, len: usize| -> Option<&[u8]> {
        bytes.get(offset + y * stride..)?.get(..len)
    };
    let mut out = Frame::new(width as u32, height as u32);
    let luma = plane(0);
    let target = &mut out.planes[0];
    for y in 0..height {
        target.data[y * target.stride..][..width].copy_from_slice(row(luma, y, width)?);
    }
    match format {
        VideoPixelFormat::Nv12 => {
            let chroma = plane(1);
            let [_, u, v] = &mut out.planes;
            for y in 0..ch {
                let source = row(chroma, y, 2 * cw)?;
                for x in 0..cw {
                    u.data[y * u.stride + x] = source[2 * x];
                    v.data[y * v.stride + x] = source[2 * x + 1];
                }
            }
        }
        VideoPixelFormat::I420 => {
            for i in 1..3 {
                let source = plane(i as u32);
                let target = &mut out.planes[i];
                for y in 0..ch {
                    target.data[y * target.stride..][..cw].copy_from_slice(row(source, y, cw)?);
                }
            }
        }
        _ => return None,
    }
    Some(out)
}
