//! The shader-side view of textures: which constant-bank words name a sampled texture, and
//! where its swizzle takes each channel from.

/// nouveau's `TexCbIndex` (bank 15), a default for Mesa-captured test fixtures
/// only; deko3d uses 0.
pub const NOUVEAU_TEX_CB_INDEX: u8 = 15;

/// The byte offset of a `texs` handle. The immediate is a dword index
/// (nouveau emits `texBindBase / 4 + unit`).
pub fn handle_offset(immediate: u16) -> u16 {
    immediate.wrapping_mul(4)
}

/// The constant-bank word a sampling instruction reads its handle from; the
/// backend binds that word's texture at draw time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureSlot {
    /// A `tex`/`texs` immediate; see [`handle_offset`].
    Bound(u16),
    /// A bindless `tex.b` handle loaded from this word of any bank.
    Bindless { bank: u8, offset: u16 },
    /// A bindless handle ORed from two words, as NVN splits the image and sampler indices.
    BindlessPair {
        bank: u8,
        offset: u16,
        other_bank: u8,
        other_offset: u16,
    },
    /// Mip level `level` of the `Bound(immediate)` texture, for a `tlds` naming its level.
    Level { immediate: u16, level: u8 },
}

impl TextureSlot {
    /// The bank and byte offset of each word ORed into this slot's handle.
    pub fn handle_words(self, tex_cb_index: u8) -> ((u8, u16), Option<(u8, u16)>) {
        match self {
            TextureSlot::Bound(immediate) | TextureSlot::Level { immediate, .. } => {
                ((tex_cb_index, handle_offset(immediate)), None)
            }
            TextureSlot::Bindless { bank, offset } => ((bank, offset), None),
            TextureSlot::BindlessPair {
                bank,
                offset,
                other_bank,
                other_offset,
            } => ((bank, offset), Some((other_bank, other_offset))),
        }
    }

    /// A key for the WGSL sampling hooks: the top bit marks bindless slots, the next a pair,
    /// the third a level.
    /// A pair keys by its first word, so two pairs sharing it clash; the translator refuses those.
    pub fn key(self) -> u32 {
        match self {
            TextureSlot::Bound(immediate) => u32::from(immediate),
            // Bound immediates are 13 bits, so a level's key sits clear of them.
            TextureSlot::Level { immediate, level } => {
                1 << 29 | u32::from(level) << 16 | u32::from(immediate)
            }
            TextureSlot::Bindless { bank, offset } => {
                1 << 31 | u32::from(bank) << 16 | u32::from(offset)
            }
            TextureSlot::BindlessPair { bank, offset, .. } => {
                1 << 31 | 1 << 30 | u32::from(bank) << 16 | u32::from(offset)
            }
        }
    }
}

/// `TIC2`'s `X_SOURCE`..`W_SOURCE`: a stored channel or a constant per component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwizzleSource {
    Zero,
    R,
    G,
    B,
    A,
    One,
}
