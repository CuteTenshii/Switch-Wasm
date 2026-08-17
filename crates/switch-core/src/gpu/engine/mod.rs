//! The GM20B engine classes.
//!
//! A channel binds a class to each of its eight subchannels with a write to
//! method 0 (`SetObject`); every later method write on that subchannel lands
//! in that class's register file. A class's registers *are* its interface —
//! writing a register both stores a value and, for a handful of "trigger"
//! registers, starts work.

pub mod compute;
pub mod copy;
pub mod inline;
pub mod threed;
pub mod twod;

/// Class ids reported by `NVGPU_GPU_IOCTL_GET_CHARACTERISTICS`.
pub const CLASS_2D: u32 = 0x902D; // FERMI_TWOD_A
pub const CLASS_3D: u32 = 0xB197; // MAXWELL_B
pub const CLASS_COMPUTE: u32 = 0xB1C0; // MAXWELL_COMPUTE_B
pub const CLASS_INLINE: u32 = 0xA140; // KEPLER_INLINE_TO_MEMORY_B
pub const CLASS_COPY: u32 = 0xB0B5; // MAXWELL_DMA_COPY_A
pub const CLASS_GPFIFO: u32 = 0xB06F; // MAXWELL_CHANNEL_GPFIFO_A

/// Registers below this are real class state; at and above it, the 3D class
/// interprets a write as a macro invocation.
pub const REGISTER_COUNT: usize = 0xE00;

/// A class's register file. Methods are dword indices into it.
#[derive(Debug, Clone)]
pub struct Registers {
    words: Vec<u32>,
}

impl Default for Registers {
    fn default() -> Self {
        Registers::new()
    }
}

impl Registers {
    pub fn new() -> Registers {
        Registers { words: vec![0; REGISTER_COUNT] }
    }

    #[inline]
    pub fn get(&self, method: u32) -> u32 {
        self.words.get(method as usize).copied().unwrap_or(0)
    }

    #[inline]
    pub fn set(&mut self, method: u32, value: u32) {
        if let Some(slot) = self.words.get_mut(method as usize) {
            *slot = value;
        }
    }

    /// A 40-bit GPU address stored as a high/low register pair, in the order
    /// the hardware (and deko3d's `Iova` helper) uses: high word first.
    #[inline]
    pub fn iova(&self, method: u32) -> u64 {
        ((self.get(method) as u64) << 32) | self.get(method + 1) as u64
    }

    #[inline]
    pub fn float(&self, method: u32) -> f32 {
        f32::from_bits(self.get(method))
    }

    /// Extract `[lo, hi]` (inclusive) bits of a register.
    #[inline]
    pub fn field(&self, method: u32, lo: u32, hi: u32) -> u32 {
        let width = hi - lo + 1;
        let mask = if width >= 32 { u32::MAX } else { (1u32 << width) - 1 };
        (self.get(method) >> lo) & mask
    }

    #[inline]
    pub fn bit(&self, method: u32, index: u32) -> bool {
        self.get(method) >> index & 1 != 0
    }
}

/// Extract `[lo, hi]` (inclusive) bits of a raw method argument.
#[inline]
pub fn field(value: u32, lo: u32, hi: u32) -> u32 {
    let width = hi - lo + 1;
    let mask = if width >= 32 { u32::MAX } else { (1u32 << width) - 1 };
    (value >> lo) & mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iova_is_high_word_first() {
        let mut regs = Registers::new();
        regs.set(0x200, 0x0000_0012);
        regs.set(0x201, 0x3456_7890);
        assert_eq!(regs.iova(0x200), 0x12_3456_7890);
    }

    #[test]
    fn field_extraction() {
        let mut regs = Registers::new();
        regs.set(0x674, 0b1010_1111_0011);
        assert_eq!(regs.field(0x674, 0, 1), 0b11);
        assert_eq!(regs.field(0x674, 6, 9), 0b1011);
        assert!(regs.bit(0x674, 0));
        assert!(!regs.bit(0x674, 2));
    }

    #[test]
    fn out_of_range_methods_are_inert() {
        let mut regs = Registers::new();
        regs.set(0xFFFF, 1);
        assert_eq!(regs.get(0xFFFF), 0);
    }
}
