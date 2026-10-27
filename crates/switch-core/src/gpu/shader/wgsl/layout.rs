//! Bindings, attribute slots and the stage layout read off a translation.

use super::Translation;
use crate::gpu::pipeline::{AttributeBase, Packed1010102};
use crate::gpu::shader::isa::TexDim;
use crate::gpu::texture::{SwizzleSource, TextureSlot};

/// Which pipeline stage a module is built for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Vertex,
    Fragment,
}

/// Module wiring as slot and bank numbers, mostly read off the program.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Layout {
    /// Generic slots a vertex shader loads, by `@location`, each four floats.
    pub attributes: Vec<usize>,
    /// Integer-format attribute slots, declared `vec4<i32>`/`vec4<u32>`; filled from the draw.
    pub integer_attributes: Vec<(usize, AttributeBase)>,
    /// Generic slots passed from vertex to fragment.
    pub varyings: Vec<usize>,
    /// Varyings read with `ipa.centroid`; both stages must agree.
    pub centroid_varyings: Vec<usize>,
    pub const_banks: Vec<u8>,
    pub textures: Vec<TextureBinding>,
    /// See [`Translation::texture_offsets`].
    pub texture_offsets: Vec<(i32, i32)>,
    pub globals: Vec<(u8, u16)>,
    /// Colour targets written, from `r0` in fours. Zero is a depth-only pass.
    pub targets: u32,
    /// This module's bind group; each stage needs its own.
    pub group: u32,
    /// Whether the vertex entry negates `position.y`: when the guest's viewport
    /// does *not* mirror y, since WebGPU's NDC-to-framebuffer already does.
    ///
    /// Setting this from `pipeline::Viewport::flip_y` directly flips twice.
    /// It looks almost right, because a full-screen quad is symmetric about
    /// the centre and so is most of a UI: the Home Menu came out 94.87%
    /// correct that way, with one off-centre band mirrored onto the other
    /// side of the screen.
    pub flip_y: bool,
    /// Whether to remap z from GL's `-w..w` onto WebGPU's `0..w`.
    pub depth_minus_one_to_one: bool,
    /// Attribute slots fetched as BGRA; filled from the draw.
    pub bgra_attributes: Vec<usize>,
    /// Attribute slots packed as 10-10-10-2; filled from the draw.
    pub packed_attributes: Vec<(usize, Packed1010102)>,
    /// Coverage for a backend rendering multisampled surfaces per texel.
    pub coverage: Option<Coverage>,
}

/// Per-sample coverage for an expanded multisample surface: sample mask and
/// alpha-to-coverage, from the draw's [`crate::gpu::surface::SampleGrid`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Coverage {
    pub samples_x: u32,
    pub samples_y: u32,
    /// The sample each texel of a pixel's tile holds, by `dy * samples_x + dx`.
    pub sample_of_slot: Vec<u32>,
    pub sample_mask: u32,
    pub alpha_to_coverage: bool,
    /// The draw is repeated once per sample, each keeping only the texels of the sample named
    /// by the `pass_sample` pipeline constant, with the geometry moved onto that sample.
    pub per_sample_pass: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureBinding {
    pub slot: TextureSlot,
    /// What the instruction samples it as.
    pub dim: TexDim,
    /// The descriptor's channel swizzle, filled from the TIC by the backend.
    pub swizzle: [SwizzleSource; 4],
    /// A shadow map, bound as `texture_depth_*` with a `sampler_comparison`.
    pub compare: bool,
    /// Set by the backend for an image taller than the device allows.
    pub fold: Option<Fold>,
}

/// A 2D image taller than the device allows, bound as array layers of overlapping
/// tiles. Tile `k` holds rows `k * (tile_rows - 1)` to `k * (tile_rows - 1) + tile_rows - 1`,
/// so the two rows a bilinear sample or a gather reads are always in one tile; rows past
/// the image repeat its last, as clamping to the edge would. A repeating image adds a
/// last tile that starts at its last row and carries on from its first, for the one pair
/// of rows no other tile holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fold {
    pub height: u32,
    pub tile_rows: u32,
    /// Every layer, the wrap tile included.
    pub layers: u32,
    pub wrap: bool,
}

impl Fold {
    /// How to fold an image `height` rows tall into tiles of at most `max_rows`, if it
    /// needs folding and `max_layers` layers hold it.
    pub fn of(height: u32, max_rows: u32, max_layers: u32, wrap: bool) -> Option<Fold> {
        if height <= max_rows || max_rows < 2 {
            return None;
        }
        let layers = (height - 1) / (max_rows - 1) + 1 + u32::from(wrap);
        (layers <= max_layers).then_some(Fold {
            height,
            tile_rows: max_rows,
            layers,
            wrap,
        })
    }

    /// The tiles that are not the wrap tile.
    pub fn tiles(&self) -> u32 {
        self.layers - u32::from(self.wrap)
    }

    /// The image row at row `row` of tile `layer`.
    pub fn image_row(&self, layer: u32, row: u32) -> u32 {
        if self.wrap && layer == self.tiles() {
            return (self.height - 1 + row) % self.height;
        }
        (layer * (self.tile_rows - 1) + row).min(self.height - 1)
    }
}

pub const IDENTITY_SWIZZLE: [SwizzleSource; 4] = [
    SwizzleSource::R,
    SwizzleSource::G,
    SwizzleSource::B,
    SwizzleSource::A,
];

pub(super) const ATTRIBUTE_WORDS: usize = 0x400 / 4;
/// `GENERIC_BASE + n * GENERIC_STRIDE + c * 4` addresses slot `n`, component `c`.
pub(super) const GENERIC_BASE: usize = 0x80;
pub(super) const GENERIC_STRIDE: usize = 0x10;
const GENERIC_SLOTS: usize = 32;
/// Clip position. Its `w` is also the fragment shader's `1/w` input.
pub(super) const POSITION: usize = 0x70;
/// `InstanceId` then `VertexId`.
pub(super) const INSTANCE_ID: usize = 0x2f8;
pub(super) const VERTEX_ID: usize = 0x2fc;

/// Bank `b` binds at `b`; texture `i` at `TEXTURE_BINDING + 2i`, sampler beside it.
pub(super) const TEXTURE_BINDING: u32 = 32;

pub const GLOBAL_BINDING: u32 = 96;

impl Layout {
    /// Read a program's interface off its translation.
    pub fn of(translated: &Translation, stage: Stage) -> Layout {
        let (attributes, varyings) = match stage {
            Stage::Vertex => (translated.loads.clone(), translated.stores.clone()),
            // Fragment `a[]` stores go nowhere.
            Stage::Fragment => (Vec::new(), translated.loads.clone()),
        };
        Layout {
            attributes,
            integer_attributes: Vec::new(),
            varyings,
            // Copied from the fragment stage by the backend.
            centroid_varyings: match stage {
                Stage::Vertex => Vec::new(),
                Stage::Fragment => translated.centroid_loads.clone(),
            },
            const_banks: translated.const_banks.clone(),
            textures: translated
                .textures
                .iter()
                .map(|&(slot, dim, compare)| TextureBinding {
                    slot,
                    dim,
                    compare,
                    swizzle: IDENTITY_SWIZZLE,
                    fold: None,
                })
                .collect(),
            texture_offsets: translated.texture_offsets.clone(),
            globals: translated.globals.clone(),
            targets: 1,
            group: 0,
            // Both from the draw's viewport.
            flip_y: false,
            depth_minus_one_to_one: false,
            bgra_attributes: Vec::new(),
            packed_attributes: Vec::new(),
            coverage: None,
        }
    }

    /// The sampling half of a varying's `@interpolate`: `", centroid"` or nothing.
    pub(super) fn sampling(&self, slot: usize) -> &'static str {
        if self.centroid_varyings.contains(&slot) {
            ", centroid"
        } else {
            ""
        }
    }

    /// How slot `slot` is packed, if as one 10-10-10-2 word.
    pub fn packing(&self, slot: usize) -> Option<Packed1010102> {
        self.packed_attributes
            .iter()
            .find(|&&(at, _)| at == slot)
            .map(|&(_, packing)| packing)
    }

    /// What slot `slot` arrives as; float unless the draw recorded an integer format.
    pub fn attribute_base(&self, slot: usize) -> AttributeBase {
        self.integer_attributes
            .iter()
            .find(|&&(at, _)| at == slot)
            .map_or(AttributeBase::Float, |&(_, base)| base)
    }
}

/// The four `a[]` words a 10-10-10-2 attribute unpacks to, as `raster::fetch_attribute` does.
pub(super) fn unpack_1010102(word: &str, packing: Packed1010102) -> [String; 4] {
    let fields = [(0u32, 10u32), (10, 10), (20, 10), (30, 2)];
    fields.map(|(offset, bits)| {
        let signed = format!("extractBits(bitcast<i32>({word}), {offset}u, {bits}u)");
        let unsigned = format!("extractBits({word}, {offset}u, {bits}u)");
        let largest = (1u32 << bits) - 1;
        match packing {
            Packed1010102::Snorm => {
                let positive = (1u32 << (bits - 1)) - 1;
                format!("max(f32({signed}) / {positive}.0, -1.0)")
            }
            Packed1010102::Unorm => format!("f32({unsigned}) / {largest}.0"),
            Packed1010102::Sint => format!("bitcast<f32>({signed})"),
            Packed1010102::Uint => format!("bitcast<f32>({unsigned})"),
        }
    })
}

pub(super) fn attribute_scalar(base: AttributeBase) -> &'static str {
    match base {
        AttributeBase::Float => "f32",
        AttributeBase::Sint => "i32",
        AttributeBase::Uint => "u32",
    }
}

pub(super) fn generic_slot(offset: u16) -> Option<usize> {
    let offset = usize::from(offset);
    if (GENERIC_BASE..GENERIC_BASE + GENERIC_SLOTS * GENERIC_STRIDE).contains(&offset) {
        Some((offset - GENERIC_BASE) / GENERIC_STRIDE)
    } else {
        None
    }
}
