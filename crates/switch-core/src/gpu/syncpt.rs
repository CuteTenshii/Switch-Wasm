//! host1x syncpoints and the `/dev/nvhost-ctrl` event slots.
//!
//! Submissions complete inside the submitting ioctl, but the counters are
//! still tracked because guests read them directly.

use crate::{Error, Result};

pub const SYNCPT_COUNT: usize = 192;
pub const EVENT_COUNT: usize = 64;

/// Lower ids are reserved for VI/ISP/display engines.
const FIRST_ALLOCATABLE: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NvFence {
    pub id: u32,
    pub value: u32,
}

impl NvFence {
    /// The id nvhost uses for "no fence".
    pub const INVALID_ID: u32 = 0xFFFF_FFFF;

    pub fn invalid() -> NvFence {
        NvFence {
            id: NvFence::INVALID_ID,
            value: 0,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.id != NvFence::INVALID_ID
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Syncpoint {
    value: u32,
    /// Highest value any outstanding job will raise it to.
    max: u32,
    allocated: bool,
}

/// A registered `/dev/nvhost-ctrl` event slot (`EVENT_REGISTER`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EventSlot {
    pub registered: bool,
    pub fence: NvFence,
    pub signalled: bool,
}

#[derive(Debug)]
pub struct Host1x {
    points: [Syncpoint; SYNCPT_COUNT],
    pub events: [EventSlot; EVENT_COUNT],
}

impl Default for Host1x {
    fn default() -> Self {
        Host1x::new()
    }
}

impl Host1x {
    pub fn new() -> Host1x {
        Host1x {
            points: [Syncpoint::default(); SYNCPT_COUNT],
            events: [EventSlot::default(); EVENT_COUNT],
        }
    }

    fn slot(&self, id: u32) -> Result<&Syncpoint> {
        self.points
            .get(id as usize)
            .ok_or_else(|| Error::Gpu(format!("host1x: syncpoint {} out of range", id)))
    }

    fn slot_mut(&mut self, id: u32) -> Result<&mut Syncpoint> {
        self.points
            .get_mut(id as usize)
            .ok_or_else(|| Error::Gpu(format!("host1x: syncpoint {} out of range", id)))
    }

    pub fn allocate(&mut self) -> Result<u32> {
        for id in FIRST_ALLOCATABLE as usize..SYNCPT_COUNT {
            if !self.points[id].allocated {
                self.points[id].allocated = true;
                return Ok(id as u32);
            }
        }
        Err(Error::Gpu("host1x: no free syncpoints".into()))
    }

    pub fn release(&mut self, id: u32) {
        if let Some(p) = self.points.get_mut(id as usize) {
            p.allocated = false;
        }
    }

    pub fn read(&self, id: u32) -> Result<u32> {
        Ok(self.slot(id)?.value)
    }

    pub fn read_max(&self, id: u32) -> Result<u32> {
        Ok(self.slot(id)?.max)
    }

    /// Reserve `count` future increments and return the resulting threshold.
    pub fn incr_max(&mut self, id: u32, count: u32) -> Result<u32> {
        let p = self.slot_mut(id)?;
        p.max = p.max.wrapping_add(count);
        Ok(p.max)
    }

    pub fn increment(&mut self, id: u32) -> Result<u32> {
        let p = self.slot_mut(id)?;
        p.value = p.value.wrapping_add(1);
        if p.value.wrapping_sub(p.max) as i32 > 0 {
            p.max = p.value;
        }
        Ok(p.value)
    }

    /// Raise the counter to `value`; it never moves backwards.
    pub fn set(&mut self, id: u32, value: u32) -> Result<()> {
        let p = self.slot_mut(id)?;
        if value.wrapping_sub(p.value) as i32 > 0 {
            p.value = value;
        }
        if (p.max.wrapping_sub(p.value) as i32) < 0 {
            p.max = p.value;
        }
        Ok(())
    }

    /// Wrapping comparison, as the hardware does it.
    pub fn is_expired(&self, id: u32, threshold: u32) -> Result<bool> {
        let p = self.slot(id)?;
        Ok(p.value.wrapping_sub(threshold) as i32 >= 0)
    }

    pub fn register_event(&mut self, slot: u32) -> Result<()> {
        let e = self
            .events
            .get_mut(slot as usize)
            .ok_or_else(|| Error::Gpu(format!("host1x: event slot {} out of range", slot)))?;
        *e = EventSlot {
            registered: true,
            fence: NvFence::invalid(),
            signalled: false,
        };
        Ok(())
    }

    pub fn unregister_event(&mut self, slot: u32) -> Result<()> {
        let e = self
            .events
            .get_mut(slot as usize)
            .ok_or_else(|| Error::Gpu(format!("host1x: event slot {} out of range", slot)))?;
        *e = EventSlot::default();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_hands_out_distinct_points() {
        let mut h = Host1x::new();
        let a = h.allocate().unwrap();
        let b = h.allocate().unwrap();
        assert_ne!(a, b);
        h.release(a);
        assert_eq!(h.allocate().unwrap(), a);
    }

    #[test]
    fn incr_max_then_increment_expires_the_fence() {
        let mut h = Host1x::new();
        let id = h.allocate().unwrap();
        let threshold = h.incr_max(id, 1).unwrap();
        assert!(!h.is_expired(id, threshold).unwrap());
        h.increment(id).unwrap();
        assert!(h.is_expired(id, threshold).unwrap());
    }

    #[test]
    fn expiry_uses_wrapping_comparison() {
        let mut h = Host1x::new();
        let id = h.allocate().unwrap();
        h.set(id, 5).unwrap();
        assert!(h.is_expired(id, 0xFFFF_FFF0).unwrap());
        assert!(!h.is_expired(id, 6).unwrap());
    }

    #[test]
    fn out_of_range_syncpoint_errors() {
        let h = Host1x::new();
        assert!(h.read(SYNCPT_COUNT as u32).is_err());
    }
}
