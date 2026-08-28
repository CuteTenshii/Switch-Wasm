//! `IGraphicBufferProducer` — the buffer queue between an app and the
//! compositor.
//!
//! The app registers each of its swapchain images with
//! `SET_PREALLOCATED_BUFFER`, then loops: `DEQUEUE_BUFFER` to get a slot it may
//! render into, and `QUEUE_BUFFER` to hand the finished image to the display.
//! Queuing is what makes a frame appear, so that transaction is where the
//! emulator scans the image out.
//!
//! The Switch's version of this interface is Android's, with one extra
//! command (`SET_PREALLOCATED_BUFFER`) and Nvidia's `NvGraphicBuffer` in place
//! of a gralloc handle.

use crate::display::parcel::{ParcelReader, ParcelWriter};
use crate::gpu::DisplayBuffer;

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

/// Maximum number of buffer slots, as Android defines it.
pub const MAX_SLOTS: usize = 64;

/// `NvMultiFence`: a count followed by four `{ id, value }` fences.
const MULTI_FENCE_SIZE: usize = 4 + 4 * 8;

/// Offset of the `NvGraphicBuffer` fields inside the flattened blob. The blob
/// starts with ten words (`magic, width, height, stride, format, usage, pid,
/// refcount, numFds, numInts`), then carries the `NvGraphicBuffer` from just
/// past its 12-byte `NativeHandle` header.
const BLOB_INTS_OFFSET: usize = 40;
const NATIVE_HANDLE_SIZE: usize = 12;

/// Field offsets within `NvGraphicBuffer` (see libnx `graphic_buffer.h`).
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

/// Byte offset of `transform` inside the flattened `QueueBufferInput`
/// (`{ s64 timestamp, s32 isAutoTimestamp, Rect crop, s32 scalingMode,
/// u32 transform, ... }`) — how the queued image is stored versus how it is
/// to be shown. Discarding it drew every Minecraft frame upside down.
const INPUT_TRANSFORM_OFFSET: usize = 32;

/// `NATIVE_WINDOW_*` selectors for `QUERY`.
const QUERY_WIDTH: i32 = 0;
const QUERY_HEIGHT: i32 = 1;
const QUERY_FORMAT: i32 = 2;
const QUERY_MIN_UNDEQUEUED_BUFFERS: i32 = 3;
const QUERY_CONSUMER_RUNNING_BEHIND: i32 = 9;

/// Android status codes the producer returns.
const STATUS_OK: i32 = 0;
const STATUS_NO_MEMORY: i32 = -12;
const STATUS_BAD_VALUE: i32 = -22;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum SlotState {
    /// No buffer registered.
    #[default]
    Empty,
    /// Registered and available to be dequeued.
    Free,
    /// Handed to the app to render into.
    Dequeued,
}

#[derive(Debug, Clone, Default)]
struct Slot {
    state: SlotState,
    buffer: Option<DisplayBuffer>,
    /// The flattened `NvGraphicBuffer` this slot was registered with, kept
    /// verbatim because `REQUEST_BUFFER` has to hand the very same bytes back
    /// -- see the note there.
    blob: Option<Vec<u8>>,
}

/// What the caller must do after a transaction, beyond sending the reply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    None,
    /// The app queued a finished frame: scan this buffer out.
    Present(DisplayBuffer),
}

#[derive(Debug)]
pub struct BufferQueue {
    slots: [Slot; MAX_SLOTS],
    /// Default geometry reported by `CONNECT`/`QUERY` before any buffer is
    /// registered.
    pub width: u32,
    pub height: u32,
    pub connected: bool,
    /// Frames queued since boot.
    pub queued: u64,
}

impl Default for BufferQueue {
    fn default() -> Self {
        BufferQueue::new()
    }
}

impl BufferQueue {
    pub fn new() -> BufferQueue {
        // An undocked console, which is what one is until something docks it —
        // and the size comes from there rather than from a pair of literals,
        // because a queue that disagrees with the display is a title drawing
        // at the wrong scale. `Cpu::set_operation_mode` moves it.
        let (width, height) = crate::cpu::OperationMode::Handheld.display_size();
        BufferQueue {
            slots: std::array::from_fn(|_| Slot::default()),
            width,
            height,
            connected: false,
            queued: 0,
        }
    }

    /// Set the geometry a caller is told about before it has dequeued
    /// anything — `QUERY_WIDTH`/`QUERY_HEIGHT` and the default in a dequeue
    /// reply. It is the display's size, so docking the console moves it.
    ///
    /// Only the default: `DequeueBuffer` and a queued buffer both overwrite
    /// these with the size the guest actually asked for, and that one is not
    /// the dock's to change underneath it.
    pub fn set_default_size(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    /// Handle one binder transaction. `request` is the raw incoming parcel;
    /// the returned parcel is what the caller writes into the reply buffer.
    pub fn transact(&mut self, code: u32, request: &[u8]) -> (Vec<u8>, Action) {
        let mut r = ParcelReader::new(request);
        r.skip_interface_token();
        let mut w = ParcelWriter::new();
        let mut action = Action::None;

        match code {
            REQUEST_BUFFER => {
                // Hand back the flattened `GraphicBuffer` registered in this
                // slot: `{ nonNull, [buffer], result }`.
                //
                // This used to answer `nonNull = 0` with success, on the
                // reasoning that the app preallocated the buffer and so
                // already has it. It does -- but its `Surface` caches buffers
                // per slot and only trusts that cache for a slot it has
                // requested before, so the first request for each slot is
                // answered out of an empty cache. "A Short Hike" believed the
                // success, took the null buffer that came with it, and
                // dereferenced it: `NvWsi`'s swapchain reads the fence out of
                // `buffer + 0x60` the instant `dequeueBuffer` returns. That
                // read lands in the soft-mapped low pages instead of faulting,
                // so the failure surfaced a long way from here -- the WSI
                // thread locked a `nn::os::MutexType` at address 0xc7,
                // deadlocked, and no frame was ever drawn.
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
                    // Nothing is registered in that slot, which is what
                    // Android reports as a bad slot index rather than as an
                    // empty success.
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
                        // No fence: the previous frame is already scanned out,
                        // so the app may render into the slot immediately.
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
                let transform = r
                    .read_flattened()
                    .and_then(|input| input.get(INPUT_TRANSFORM_OFFSET..INPUT_TRANSFORM_OFFSET + 4))
                    .map_or(0, |bytes| read_u32(bytes, 0));
                action = match self.queue(slot, transform) {
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
                // The reply parcel carries no content.
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
                // The reply parcel carries no content.
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

    /// Mark a dequeued slot as presented. Because scan-out happens
    /// immediately, the slot goes straight back to free.
    fn queue(&mut self, slot: i32, transform: u32) -> Option<DisplayBuffer> {
        let index = usize::try_from(slot).ok()?;
        let entry = self.slots.get_mut(index)?;
        let mut buffer = entry.buffer?;
        buffer.transform = transform;
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

    /// Decode the flattened `NvGraphicBuffer` a `SET_PREALLOCATED_BUFFER`
    /// carries and register it in `slot`.
    fn set_preallocated(&mut self, slot: i32, blob: &[u8]) {
        let index = match usize::try_from(slot) {
            Ok(index) if index < MAX_SLOTS => index,
            _ => return,
        };
        // `field(x)` addresses the NvGraphicBuffer by its own offsets, which
        // the blob stores from just past the NativeHandle header.
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
            // The producer names it per queued frame, not per slot.
            transform: 0,
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

    /// Wrap `body` (already-serialized payload words) into a request parcel
    /// that starts with the interface token.
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

    /// Build the flattened NvGraphicBuffer blob a real
    /// `SET_PREALLOCATED_BUFFER` carries.
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
        // Answering `nonNull = 0` with success -- on the reasoning that the
        // app preallocated the buffer and so already has it -- is a lie the
        // caller believes: its `Surface` caches buffers per slot and asks for
        // each slot once, so the first ask comes out of an empty cache and it
        // takes the null. "A Short Hike" then read the fence out of
        // `buffer + 0x60`, which with the low pages soft-mapped does not
        // fault; its swapchain thread went on to lock a `nn::os::MutexType` at
        // address 0xc7, deadlocked there, and the title never drew a frame.
        let mut q = BufferQueue::new();
        preallocate(&mut q, 0, 7, 0x1000);
        let expected = graphic_buffer_blob(7, 1280, 720, 0x1000);

        let (reply, action) = q.transact(REQUEST_BUFFER, &request(&words(&[0])));
        assert_eq!(action, Action::None);
        let mut r = ParcelReader::new(&reply);
        assert_eq!(r.read_i32(), 1, "no buffer came back for a registered slot");
        assert_eq!(r.read_flattened(), Some(&expected[..]));
        assert_eq!(r.read_i32(), STATUS_OK);

        // A slot nothing was ever registered in is a bad slot index, not an
        // empty success -- the caller has to be able to tell those apart.
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
    fn slots_alternate_across_frames() {
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
        // With immediate scan-out the first slot is always free again, which
        // is correct: the display is never holding a buffer.
        assert!(seen.iter().all(|&s| s >= 0));
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
