//! `IGraphicBufferProducer`, the buffer queue between an app and the compositor.
//! Android's interface plus `SET_PREALLOCATED_BUFFER`, with `NvGraphicBuffer` handles.

use crate::display::parcel::{ParcelReader, ParcelWriter};
use crate::gpu::{Crop, DisplayBuffer};

/// Transaction codes (`IGraphicBufferProducer.cpp`).
pub const REQUEST_BUFFER: u32 = 1;
pub const SET_BUFFER_COUNT: u32 = 2;
pub const DEQUEUE_BUFFER: u32 = 3;
pub const DETACH_BUFFER: u32 = 4;
pub const QUEUE_BUFFER: u32 = 7;
pub const CANCEL_BUFFER: u32 = 8;
pub const QUERY: u32 = 9;
pub const CONNECT: u32 = 10;
pub const DISCONNECT: u32 = 11;
pub const SET_PREALLOCATED_BUFFER: u32 = 14;

pub const MAX_SLOTS: usize = 64;

/// `NvMultiFence`: a count followed by four `{ id, value }` fences.
const MULTI_FENCE_SIZE: usize = 4 + 4 * 8;

/// `NvGraphicBuffer` fields start after ten header words, past its 12-byte `NativeHandle`.
const BLOB_INTS_OFFSET: usize = 40;
const NATIVE_HANDLE_SIZE: usize = 12;

/// Field offsets within `NvGraphicBuffer` (libnx `graphic_buffer.h`).
const GB_NVMAP_ID: usize = 0x10;
const GB_PLANES: usize = 0x40;
/// Field offsets within `NvSurface`.
const PLANE_WIDTH: usize = 0x00;
const PLANE_HEIGHT: usize = 0x04;
const PLANE_COLOR_FORMAT: usize = 0x08;
const PLANE_LAYOUT: usize = 0x10;
const PLANE_PITCH: usize = 0x14;
const PLANE_OFFSET: usize = 0x1C;
const PLANE_BLOCK_HEIGHT_LOG2: usize = 0x24;

/// Byte offsets inside the flattened `QueueBufferInput`.
const INPUT_CROP_OFFSET: usize = 12;
const INPUT_TRANSFORM_OFFSET: usize = 32;

/// `NATIVE_WINDOW_*` selectors for `QUERY`.
const QUERY_WIDTH: i32 = 0;
const QUERY_HEIGHT: i32 = 1;
const QUERY_FORMAT: i32 = 2;
const QUERY_MIN_UNDEQUEUED_BUFFERS: i32 = 3;
const QUERY_CONSUMER_RUNNING_BEHIND: i32 = 9;

/// Android status codes.
const STATUS_OK: i32 = 0;
const STATUS_NO_MEMORY: i32 = -12;
const STATUS_BAD_VALUE: i32 = -22;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum SlotState {
    #[default]
    Empty,
    Free,
    Dequeued,
}

#[derive(Debug, Clone, Default)]
struct Slot {
    state: SlotState,
    buffer: Option<DisplayBuffer>,
    /// Kept verbatim: `REQUEST_BUFFER` hands the same bytes back.
    blob: Option<Vec<u8>>,
}

/// What the caller must do after a transaction, beyond sending the reply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    None,
    Present(DisplayBuffer),
}

#[derive(Debug)]
pub struct BufferQueue {
    slots: [Slot; MAX_SLOTS],
    /// Geometry reported before any buffer is registered.
    pub width: u32,
    pub height: u32,
    pub connected: bool,
    pub queued: u64,
}

impl Default for BufferQueue {
    fn default() -> Self {
        BufferQueue::new()
    }
}

impl BufferQueue {
    pub fn new() -> BufferQueue {
        let (width, height) = crate::cpu::OperationMode::Handheld.display_size();
        BufferQueue {
            slots: std::array::from_fn(|_| Slot::default()),
            width,
            height,
            connected: false,
            queued: 0,
        }
    }

    /// Only the default; dequeued and queued buffers report their own size.
    pub fn set_default_size(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    pub fn transact(&mut self, code: u32, request: &[u8]) -> (Vec<u8>, Action) {
        let mut r = ParcelReader::new(request);
        r.skip_interface_token();
        let mut w = ParcelWriter::new();
        let mut action = Action::None;

        match code {
            REQUEST_BUFFER => {
                // `{ nonNull, [buffer], result }`; the app's `Surface` needs the buffer even though it preallocated it.
                let slot = r.read_i32();
                let blob = usize::try_from(slot)
                    .ok()
                    .and_then(|index| self.slots.get(index))
                    .and_then(|entry| entry.blob.as_deref());
                match blob {
                    Some(blob) => {
                        w.write_i32(1);
                        w.write_flattened(blob);
                        w.write_i32(STATUS_OK);
                    }
                    // Android reports an empty slot as a bad index.
                    None => {
                        w.write_i32(0);
                        w.write_i32(STATUS_BAD_VALUE);
                    }
                }
            }
            SET_BUFFER_COUNT | DETACH_BUFFER => {
                w.write_i32(STATUS_OK);
            }
            DEQUEUE_BUFFER => {
                let _async = r.read_i32();
                let width = r.read_u32();
                let height = r.read_u32();
                let _format = r.read_i32();
                let _usage = r.read_u32();
                if width != 0 && height != 0 {
                    self.width = width;
                    self.height = height;
                }
                match self.acquire_free_slot() {
                    Some(slot) => {
                        w.write_i32(slot as i32);
                        // No fence: the slot can be rendered into immediately.
                        w.write_i32(1);
                        w.write_flattened(&[0u8; MULTI_FENCE_SIZE]);
                        w.write_i32(STATUS_OK);
                    }
                    None => {
                        w.write_i32(-1);
                        w.write_i32(0);
                        w.write_i32(STATUS_NO_MEMORY);
                    }
                }
            }
            QUEUE_BUFFER => {
                let slot = r.read_i32();
                let input = r.read_flattened().unwrap_or_default();
                let word = |at: usize| read_u32(input, at) as i32;
                let transform = word(INPUT_TRANSFORM_OFFSET) as u32;
                let crop = Crop {
                    left: word(INPUT_CROP_OFFSET),
                    top: word(INPUT_CROP_OFFSET + 4),
                    right: word(INPUT_CROP_OFFSET + 8),
                    bottom: word(INPUT_CROP_OFFSET + 12),
                };
                action = match self.queue(slot, transform, crop) {
                    Some(buffer) => Action::Present(buffer),
                    None => Action::None,
                };
                self.write_buffer_output(&mut w);
                w.write_i32(if action == Action::None {
                    STATUS_BAD_VALUE
                } else {
                    STATUS_OK
                });
            }
            CANCEL_BUFFER => {
                let slot = r.read_i32();
                self.release(slot);
            }
            QUERY => {
                let what = r.read_i32();
                let value = match what {
                    QUERY_WIDTH => self.width as i32,
                    QUERY_HEIGHT => self.height as i32,
                    // PIXEL_FORMAT_RGBA_8888.
                    QUERY_FORMAT => 1,
                    QUERY_MIN_UNDEQUEUED_BUFFERS => 1,
                    QUERY_CONSUMER_RUNNING_BEHIND => 0,
                    _ => 0,
                };
                w.write_i32(value);
                w.write_i32(STATUS_OK);
            }
            CONNECT => {
                let _listener = r.read_i32();
                let _api = r.read_i32();
                let _producer_controlled_by_app = r.read_i32();
                self.connected = true;
                self.write_buffer_output(&mut w);
                w.write_i32(STATUS_OK);
            }
            DISCONNECT => {
                let _api = r.read_i32();
                self.connected = false;
                w.write_i32(STATUS_OK);
            }
            SET_PREALLOCATED_BUFFER => {
                let slot = r.read_i32();
                let has_input = r.read_i32();
                if has_input != 0 {
                    if let Some(blob) = r.read_flattened() {
                        self.set_preallocated(slot, blob);
                    }
                } else {
                    self.clear(slot);
                }
            }
            _ => {
                w.write_i32(STATUS_BAD_VALUE);
            }
        }
        (w.finish(), action)
    }

    /// `BqBufferOutput { width, height, transformHint, numPendingBuffers }`.
    fn write_buffer_output(&self, w: &mut ParcelWriter) {
        w.write_u32(self.width);
        w.write_u32(self.height);
        w.write_u32(0);
        w.write_u32(0);
    }

    fn acquire_free_slot(&mut self) -> Option<usize> {
        let index = self
            .slots
            .iter()
            .position(|s| s.state == SlotState::Free && s.buffer.is_some())?;
        self.slots[index].state = SlotState::Dequeued;
        Some(index)
    }

    /// Scan-out is immediate, so the slot goes straight back to free.
    fn queue(&mut self, slot: i32, transform: u32, crop: Crop) -> Option<DisplayBuffer> {
        let index = usize::try_from(slot).ok()?;
        let entry = self.slots.get_mut(index)?;
        let mut buffer = entry.buffer?;
        buffer.transform = transform;
        buffer.crop = crop;
        entry.state = SlotState::Free;
        self.queued += 1;
        Some(buffer)
    }

    fn release(&mut self, slot: i32) {
        if let Ok(index) = usize::try_from(slot) {
            if let Some(entry) = self.slots.get_mut(index) {
                if entry.buffer.is_some() {
                    entry.state = SlotState::Free;
                }
            }
        }
    }

    fn clear(&mut self, slot: i32) {
        if let Ok(index) = usize::try_from(slot) {
            if let Some(entry) = self.slots.get_mut(index) {
                *entry = Slot::default();
            }
        }
    }

    fn set_preallocated(&mut self, slot: i32, blob: &[u8]) {
        let index = match usize::try_from(slot) {
            Ok(index) if index < MAX_SLOTS => index,
            _ => return,
        };
        // Addresses the `NvGraphicBuffer` by its own offsets.
        let field = |offset: usize| -> u32 {
            read_u32(blob, BLOB_INTS_OFFSET + offset - NATIVE_HANDLE_SIZE)
        };
        let plane = |offset: usize| -> u32 { field(GB_PLANES + offset) };
        let color_format =
            (plane(PLANE_COLOR_FORMAT) as u64) | ((plane(PLANE_COLOR_FORMAT + 4) as u64) << 32);
        let buffer = DisplayBuffer {
            nvmap_id: field(GB_NVMAP_ID),
            offset: plane(PLANE_OFFSET),
            width: plane(PLANE_WIDTH),
            height: plane(PLANE_HEIGHT),
            pitch: plane(PLANE_PITCH),
            layout: plane(PLANE_LAYOUT),
            block_height_log2: plane(PLANE_BLOCK_HEIGHT_LOG2),
            color_format,
            // Set per queued frame, not per slot.
            transform: 0,
            crop: Crop::ALL,
        };
        if buffer.width != 0 && buffer.height != 0 {
            self.width = buffer.width;
            self.height = buffer.height;
        }
        self.slots[index] = Slot {
            state: SlotState::Free,
            buffer: Some(buffer),
            blob: Some(blob.to_vec()),
        };
    }
}

fn read_u32(data: &[u8], at: usize) -> u32 {
    let mut v = 0u32;
    for i in 0..4 {
        v |= (data.get(at + i).copied().unwrap_or(0) as u32) << (8 * i);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::NV_LAYOUT_BLOCK_LINEAR;

    fn token() -> Vec<u8> {
        let name = "android.gui.IGraphicBufferProducer";
        let mut w = ParcelWriter::new();
        w.write_u32(0x100);
        w.write_i32(name.len() as i32);
        let mut utf16 = Vec::new();
        for c in name.chars().chain(std::iter::once('\0')) {
            utf16.extend_from_slice(&(c as u16).to_le_bytes());
        }
        w.write_bytes(&utf16);
        w.finish()
    }

    /// Prefixes `body` with the interface token.
    fn request(body: &[u8]) -> Vec<u8> {
        let tok = token();
        let payload_len = read_u32(&tok, 0) as usize;
        let mut payload = tok[16..16 + payload_len].to_vec();
        payload.extend_from_slice(body);
        let mut out = Vec::new();
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&(16 + payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&payload);
        out
    }

    fn words(values: &[i32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn graphic_buffer_blob(nvmap_id: u32, width: u32, height: u32, offset: u32) -> Vec<u8> {
        let mut blob = vec![0u8; BLOB_INTS_OFFSET + 0x150 - NATIVE_HANDLE_SIZE];
        let mut put = |off: usize, value: u32| {
            let at = BLOB_INTS_OFFSET + off - NATIVE_HANDLE_SIZE;
            blob[at..at + 4].copy_from_slice(&value.to_le_bytes());
        };
        put(GB_NVMAP_ID, nvmap_id);
        put(GB_PLANES + PLANE_WIDTH, width);
        put(GB_PLANES + PLANE_HEIGHT, height);
        put(GB_PLANES + PLANE_COLOR_FORMAT, 0x0053_2120);
        put(GB_PLANES + PLANE_COLOR_FORMAT + 4, 0x01);
        put(GB_PLANES + PLANE_LAYOUT, NV_LAYOUT_BLOCK_LINEAR);
        put(GB_PLANES + PLANE_PITCH, width * 4);
        put(GB_PLANES + PLANE_OFFSET, offset);
        put(GB_PLANES + PLANE_BLOCK_HEIGHT_LOG2, 4);
        blob
    }

    fn preallocate(queue: &mut BufferQueue, slot: i32, nvmap_id: u32, offset: u32) {
        let blob = graphic_buffer_blob(nvmap_id, 1280, 720, offset);
        let mut body = ParcelWriter::new();
        body.write_i32(slot);
        body.write_i32(1);
        body.write_flattened(&blob);
        let raw = body.finish();
        let payload_len = read_u32(&raw, 0) as usize;
        let (_, action) = queue.transact(
            SET_PREALLOCATED_BUFFER,
            &request(&raw[16..16 + payload_len]),
        );
        assert_eq!(action, Action::None);
    }

    #[test]
    fn preallocated_buffer_is_decoded() {
        let mut q = BufferQueue::new();
        preallocate(&mut q, 0, 7, 0x1000);
        let slot = q.slots[0].buffer.expect("slot 0 registered");
        assert_eq!(slot.nvmap_id, 7);
        assert_eq!(slot.width, 1280);
        assert_eq!(slot.height, 720);
        assert_eq!(slot.offset, 0x1000);
        assert_eq!(slot.layout, NV_LAYOUT_BLOCK_LINEAR);
        assert_eq!(slot.block_height_log2, 4);
        assert_eq!(slot.color_format, 0x0100_5321_20);
    }

    #[test]
    fn request_buffer_hands_back_the_buffer_registered_in_the_slot() {
        // `REQUEST_BUFFER` is `{ nonNull, [flattened GraphicBuffer], result }`.
        let mut q = BufferQueue::new();
        preallocate(&mut q, 0, 7, 0x1000);
        let expected = graphic_buffer_blob(7, 1280, 720, 0x1000);

        let (reply, action) = q.transact(REQUEST_BUFFER, &request(&words(&[0])));
        assert_eq!(action, Action::None);
        let mut r = ParcelReader::new(&reply);
        assert_eq!(r.read_i32(), 1, "no buffer came back for a registered slot");
        assert_eq!(r.read_flattened(), Some(&expected[..]));
        assert_eq!(r.read_i32(), STATUS_OK);

        // An empty slot is a bad index, not an empty success.
        let (reply, _) = q.transact(REQUEST_BUFFER, &request(&words(&[5])));
        let mut r = ParcelReader::new(&reply);
        assert_eq!(r.read_i32(), 0);
        assert_eq!(r.read_i32(), STATUS_BAD_VALUE);
    }

    #[test]
    fn dequeue_then_queue_presents_the_buffer() {
        let mut q = BufferQueue::new();
        preallocate(&mut q, 0, 7, 0);
        preallocate(&mut q, 1, 7, 0x10_0000);

        let (reply, _) = q.transact(DEQUEUE_BUFFER, &request(&words(&[0, 1280, 720, 1, 0])));
        let mut r = ParcelReader::new(&reply);
        let slot = r.read_i32();
        assert_eq!(slot, 0);
        assert_eq!(r.read_i32(), 1); // has fence
        assert_eq!(r.read_flattened().map(|f| f.len()), Some(MULTI_FENCE_SIZE));
        assert_eq!(r.read_i32(), STATUS_OK);

        let (reply, action) = q.transact(QUEUE_BUFFER, &request(&words(&[slot, 0, 0])));
        match action {
            Action::Present(buffer) => assert_eq!(buffer.nvmap_id, 7),
            other => panic!("expected a present, got {:?}", other),
        }
        let mut r = ParcelReader::new(&reply);
        assert_eq!(r.read_u32(), 1280); // BqBufferOutput.width
        assert_eq!(r.read_u32(), 720);
        assert_eq!(r.read_u32(), 0);
        assert_eq!(r.read_u32(), 0);
        assert_eq!(r.read_i32(), STATUS_OK);
        assert_eq!(q.queued, 1);
    }

    #[test]
    fn dequeue_with_no_registered_buffers_fails_cleanly() {
        let mut q = BufferQueue::new();
        let (reply, _) = q.transact(DEQUEUE_BUFFER, &request(&words(&[0, 1280, 720, 1, 0])));
        let mut r = ParcelReader::new(&reply);
        assert_eq!(r.read_i32(), -1);
        assert_eq!(r.read_i32(), 0);
        assert_eq!(r.read_i32(), STATUS_NO_MEMORY);
    }

    #[test]
    fn a_queued_slot_is_dequeued_again_on_the_next_frame() {
        let mut q = BufferQueue::new();
        preallocate(&mut q, 0, 7, 0);
        preallocate(&mut q, 1, 7, 0x10_0000);
        let mut seen = Vec::new();
        for _ in 0..4 {
            let (reply, _) = q.transact(DEQUEUE_BUFFER, &request(&words(&[0, 1280, 720, 1, 0])));
            let slot = ParcelReader::new(&reply).read_i32();
            seen.push(slot);
            q.transact(QUEUE_BUFFER, &request(&words(&[slot, 0, 0])));
        }
        // Scan-out is immediate, so the display never holds slot 0.
        assert_eq!(seen, [0, 0, 0, 0]);
        assert_eq!(q.queued, 4);
    }

    #[test]
    fn connect_reports_the_window_geometry() {
        let mut q = BufferQueue::new();
        let (reply, _) = q.transact(CONNECT, &request(&words(&[0, 2, 0])));
        let mut r = ParcelReader::new(&reply);
        assert_eq!(r.read_u32(), 1280);
        assert_eq!(r.read_u32(), 720);
        assert_eq!(r.read_u32(), 0);
        assert_eq!(r.read_u32(), 0);
        assert_eq!(r.read_i32(), STATUS_OK);
        assert!(q.connected);
    }

    #[test]
    fn query_answers_the_native_window_selectors() {
        let mut q = BufferQueue::new();
        for (what, expected) in [(QUERY_WIDTH, 1280), (QUERY_HEIGHT, 720), (QUERY_FORMAT, 1)] {
            let (reply, _) = q.transact(QUERY, &request(&words(&[what])));
            let mut r = ParcelReader::new(&reply);
            assert_eq!(r.read_i32(), expected, "query {}", what);
            assert_eq!(r.read_i32(), STATUS_OK);
        }
    }

    #[test]
    fn cancel_returns_the_slot_to_the_free_pool() {
        let mut q = BufferQueue::new();
        preallocate(&mut q, 0, 7, 0);
        let (reply, _) = q.transact(DEQUEUE_BUFFER, &request(&words(&[0, 1280, 720, 1, 0])));
        let slot = ParcelReader::new(&reply).read_i32();
        q.transact(CANCEL_BUFFER, &request(&words(&[slot])));
        let (reply, _) = q.transact(DEQUEUE_BUFFER, &request(&words(&[0, 1280, 720, 1, 0])));
        assert_eq!(ParcelReader::new(&reply).read_i32(), slot);
    }
}
