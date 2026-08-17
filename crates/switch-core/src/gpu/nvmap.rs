//! nvmap: the GPU memory-object table behind `/dev/nvmap`.
//!
//! Unlike a discrete GPU, the Tegra nvmap driver does not own memory. The
//! guest allocates a CPU buffer itself (deko3d carves memblocks out of its
//! heap), calls `NVMAP_IOC_CREATE` to get a handle, then `NVMAP_IOC_ALLOC`
//! passing the buffer's CPU address. The handle is what gets mapped into a GPU
//! address space, so a handle is really just "this CPU range, with this memory
//! kind".

use crate::{Error, Result};
use std::collections::HashMap;

/// `NvMapHandleParam` selectors for `NVMAP_IOC_PARAM`.
pub const PARAM_SIZE: u32 = 1;
pub const PARAM_ALIGNMENT: u32 = 2;
pub const PARAM_BASE: u32 = 3;
pub const PARAM_HEAP: u32 = 4;
pub const PARAM_KIND: u32 = 5;
pub const PARAM_COMPR: u32 = 6;

/// The heap the Switch's nvmap reports for every allocation
/// (`NVMAP_HEAP_CARVEOUT_GENERIC`).
pub const HEAP_CARVEOUT_GENERIC: u32 = 0x0000_0001;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvMapHandle {
    pub handle: u32,
    pub id: u32,
    /// Size requested by `NVMAP_IOC_CREATE`.
    pub size: u32,
    /// CPU address the guest allocated and passed to `NVMAP_IOC_ALLOC`; 0
    /// until the handle is allocated.
    pub cpu_addr: u32,
    pub align: u32,
    pub heap_mask: u32,
    pub flags: u32,
    /// Block-linear memory kind (`NvKind`); 0 (`PITCH`) for linear buffers.
    pub kind: u8,
    pub allocated: bool,
    pub refcount: u32,
}

impl NvMapHandle {
    fn new(handle: u32, id: u32, size: u32) -> NvMapHandle {
        NvMapHandle {
            handle,
            id,
            size,
            cpu_addr: 0,
            align: 0,
            heap_mask: 0,
            flags: 0,
            kind: 0,
            allocated: false,
            refcount: 1,
        }
    }
}

#[derive(Debug, Default)]
pub struct NvMap {
    handles: HashMap<u32, NvMapHandle>,
    /// nvmap id -> handle, for `NVMAP_IOC_FROM_ID` (how a buffer crosses a
    /// process boundary; the graphics buffer queue passes ids, not handles).
    ids: HashMap<u32, u32>,
    next_handle: u32,
    next_id: u32,
}

impl NvMap {
    pub fn new() -> NvMap {
        NvMap {
            handles: HashMap::new(),
            ids: HashMap::new(),
            next_handle: 1,
            next_id: 1,
        }
    }

    /// `NVMAP_IOC_CREATE`: reserve a handle for a `size`-byte object. No
    /// memory is committed until [`NvMap::alloc`].
    pub fn create(&mut self, size: u32) -> u32 {
        let handle = self.next_handle;
        self.next_handle += 1;
        let id = self.next_id;
        self.next_id += 1;
        self.handles.insert(handle, NvMapHandle::new(handle, id, size));
        self.ids.insert(id, handle);
        handle
    }

    /// `NVMAP_IOC_ALLOC`: bind the guest-allocated buffer at `cpu_addr` to the
    /// handle and record its layout parameters.
    pub fn alloc(
        &mut self,
        handle: u32,
        heap_mask: u32,
        flags: u32,
        align: u32,
        kind: u8,
        cpu_addr: u32,
    ) -> Result<()> {
        let h = self
            .handles
            .get_mut(&handle)
            .ok_or_else(|| Error::Gpu(format!("nvmap: alloc of unknown handle {}", handle)))?;
        h.heap_mask = if heap_mask == 0 { HEAP_CARVEOUT_GENERIC } else { heap_mask };
        h.flags = flags;
        h.align = align.max(1);
        h.kind = kind;
        h.cpu_addr = cpu_addr;
        h.allocated = true;
        Ok(())
    }

    /// `NVMAP_IOC_FREE`: drop a reference; returns the handle as it was, so
    /// the caller can fill in the ioctl's out-fields.
    pub fn free(&mut self, handle: u32) -> Option<NvMapHandle> {
        let h = self.handles.get_mut(&handle)?;
        h.refcount = h.refcount.saturating_sub(1);
        let snapshot = *h;
        if snapshot.refcount == 0 {
            self.handles.remove(&handle);
            self.ids.remove(&snapshot.id);
        }
        Some(snapshot)
    }

    /// `NVMAP_IOC_FROM_ID`: take a reference to an existing object by id.
    pub fn from_id(&mut self, id: u32) -> Option<u32> {
        let handle = *self.ids.get(&id)?;
        if let Some(h) = self.handles.get_mut(&handle) {
            h.refcount += 1;
        }
        Some(handle)
    }

    pub fn get(&self, handle: u32) -> Option<&NvMapHandle> {
        self.handles.get(&handle)
    }

    pub fn by_id(&self, id: u32) -> Option<&NvMapHandle> {
        self.handles.get(self.ids.get(&id)?)
    }

    /// `NVMAP_IOC_PARAM`.
    pub fn param(&self, handle: u32, param: u32) -> Result<u32> {
        let h = self
            .handles
            .get(&handle)
            .ok_or_else(|| Error::Gpu(format!("nvmap: param on unknown handle {}", handle)))?;
        Ok(match param {
            PARAM_SIZE => h.size,
            PARAM_ALIGNMENT => h.align,
            PARAM_BASE => h.cpu_addr,
            PARAM_HEAP => h.heap_mask,
            PARAM_KIND => h.kind as u32,
            PARAM_COMPR => 0,
            _ => 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_alloc_param_roundtrip() {
        let mut nvmap = NvMap::new();
        let h = nvmap.create(0x2000);
        nvmap.alloc(h, 0, 0, 0x1000, 0xFE, 0x3000_0000).unwrap();
        assert_eq!(nvmap.param(h, PARAM_SIZE).unwrap(), 0x2000);
        assert_eq!(nvmap.param(h, PARAM_ALIGNMENT).unwrap(), 0x1000);
        assert_eq!(nvmap.param(h, PARAM_BASE).unwrap(), 0x3000_0000);
        assert_eq!(nvmap.param(h, PARAM_KIND).unwrap(), 0xFE);
        assert_eq!(nvmap.param(h, PARAM_HEAP).unwrap(), HEAP_CARVEOUT_GENERIC);
    }

    #[test]
    fn ids_are_distinct_and_resolve_back() {
        let mut nvmap = NvMap::new();
        let a = nvmap.create(0x1000);
        let b = nvmap.create(0x1000);
        let id_a = nvmap.get(a).unwrap().id;
        let id_b = nvmap.get(b).unwrap().id;
        assert_ne!(id_a, id_b);
        assert_eq!(nvmap.from_id(id_a), Some(a));
        assert_eq!(nvmap.from_id(id_b), Some(b));
    }

    #[test]
    fn free_drops_the_object_at_zero_refs() {
        let mut nvmap = NvMap::new();
        let h = nvmap.create(0x1000);
        let id = nvmap.get(h).unwrap().id;
        nvmap.from_id(id); // refcount 2
        assert!(nvmap.free(h).is_some());
        assert!(nvmap.get(h).is_some());
        assert!(nvmap.free(h).is_some());
        assert!(nvmap.get(h).is_none());
    }

    #[test]
    fn param_on_unknown_handle_errors() {
        let nvmap = NvMap::new();
        assert!(nvmap.param(7, PARAM_SIZE).is_err());
    }
}
