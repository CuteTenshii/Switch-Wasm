//! nvdec's H.264 on the browser's decoder. It lives on the system worker, which runs every SVC,
//! so its pictures land between slices, when the worker's event loop runs.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use switch_core::gpu::multimedia::H264Decoder;
use switch_core::vp9::Frame;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    CodecState, EncodedVideoChunk, EncodedVideoChunkInit, EncodedVideoChunkType,
    HardwareAcceleration, VideoDecoder, VideoDecoderConfig, VideoDecoderInit, VideoFrame,
    VideoPixelFormat,
};

/// High profile at level 5.1, past anything the console decodes.
const CODEC: &str = "avc1.640033";

type Landed = Rc<RefCell<Vec<(u64, Option<Frame>)>>>;
/// The surface each chunk in flight decodes to, by its timestamp.
type InFlight = Rc<RefCell<HashMap<i32, u64>>>;

#[derive(Debug, Default)]
pub struct WebH264 {
    decoder: Option<VideoDecoder>,
    in_flight: InFlight,
    landed: Landed,
    next: i32,
    callbacks: Option<(Closure<dyn FnMut(JsValue)>, Closure<dyn FnMut(VideoFrame)>)>,
}

impl WebH264 {
    fn open(&mut self) -> Result<VideoDecoder, JsValue> {
        let (in_flight, landed) = (self.in_flight.clone(), self.landed.clone());
        let error = Closure::<dyn FnMut(JsValue)>::new(move |why: JsValue| {
            switch_core::traceln!("[video] the H.264 decoder failed: {why:?}");
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
        let config = VideoDecoderConfig::new(CODEC);
        config.set_hardware_acceleration(HardwareAcceleration::PreferHardware);
        config.set_optimize_for_latency(true);
        decoder.configure(&config)?;
        self.callbacks = Some((error, output));
        Ok(decoder)
    }

    fn send(&mut self, surface: u64, access_unit: &[u8], key: bool) -> Result<(), JsValue> {
        let open = self
            .decoder
            .as_ref()
            .is_some_and(|d| d.state() == CodecState::Configured);
        if !open {
            if !key {
                return Err("a picture before any IDR".into());
            }
            self.decoder = Some(self.open()?);
        }
        let kind = if key {
            EncodedVideoChunkType::Key
        } else {
            EncodedVideoChunkType::Delta
        };
        let data = js_sys::Uint8Array::from(access_unit);
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

impl H264Decoder for WebH264 {
    fn decode(&mut self, surface: u64, access_unit: &[u8], key: bool) {
        if let Err(why) = self.send(surface, access_unit, key) {
            switch_core::traceln!("[video] H.264 picture for {surface:#x} not decoded: {why:?}");
            self.landed.borrow_mut().push((surface, None));
        }
    }

    fn take(&mut self) -> Vec<(u64, Option<Frame>)> {
        std::mem::take(&mut self.landed.borrow_mut())
    }
}

/// The frame's three planes, from the NV12 a hardware decoder gives or the I420 of a software one.
async fn planes(frame: &VideoFrame) -> Option<Frame> {
    let format = frame.format()?;
    let (width, height) = (frame.coded_width() as usize, frame.coded_height() as usize);
    let size = frame.allocation_size().ok()? as usize;
    if size != width * height * 3 / 2 {
        return None;
    }
    let buffer = js_sys::Uint8Array::new_with_length(size as u32);
    JsFuture::from(frame.copy_to_with_buffer_source(&buffer))
        .await
        .ok()?;
    let bytes = buffer.to_vec();
    let (luma, chroma) = bytes.split_at(width * height);
    let mut out = Frame::new(width as u32, height as u32);
    let (cw, ch) = (width / 2, height / 2);
    for y in 0..height {
        let stride = out.planes[0].stride;
        out.planes[0].data[y * stride..][..width].copy_from_slice(&luma[y * width..][..width]);
    }
    for y in 0..ch {
        for x in 0..cw {
            let (u, v) = match format {
                VideoPixelFormat::Nv12 => {
                    (chroma[y * width + 2 * x], chroma[y * width + 2 * x + 1])
                }
                VideoPixelFormat::I420 => (chroma[y * cw + x], chroma[cw * ch + y * cw + x]),
                _ => return None,
            };
            let (su, sv) = (out.planes[1].stride, out.planes[2].stride);
            out.planes[1].data[y * su + x] = u;
            out.planes[2].data[y * sv + x] = v;
        }
    }
    Some(out)
}
