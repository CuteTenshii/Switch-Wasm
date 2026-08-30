//! What a piece of Maxwell pipeline state is called on the device.
//!
//! Every function here maps one of `switch_core::gpu::pipeline`'s enums onto
//! wgpu's name for the same thing. The format entry points are the exception:
//! a format the guest names is not always one this device can hold, still
//! less one it can draw into, so those answer `Result`.

use switch_core::gpu::pipeline::{self as state, Format};
use switch_core::gpu::upload::{DepthKind, IndexFormat};
use switch_core::{Error, Result};

pub(crate) fn topology(topology: state::Topology) -> wgpu::PrimitiveTopology {
    match topology {
        state::Topology::PointList => wgpu::PrimitiveTopology::PointList,
        state::Topology::LineList => wgpu::PrimitiveTopology::LineList,
        state::Topology::LineStrip => wgpu::PrimitiveTopology::LineStrip,
        state::Topology::TriangleList => wgpu::PrimitiveTopology::TriangleList,
        state::Topology::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
    }
}

pub(crate) fn index_format(format: IndexFormat) -> wgpu::IndexFormat {
    match format {
        IndexFormat::Uint16 => wgpu::IndexFormat::Uint16,
        IndexFormat::Uint32 => wgpu::IndexFormat::Uint32,
    }
}

/// The guest's per-channel colour write enables, as WebGPU spells them.
pub(crate) fn write_mask(mask: [bool; 4]) -> wgpu::ColorWrites {
    let channels = [
        wgpu::ColorWrites::RED,
        wgpu::ColorWrites::GREEN,
        wgpu::ColorWrites::BLUE,
        wgpu::ColorWrites::ALPHA,
    ];
    let mut writes = wgpu::ColorWrites::empty();
    for (enabled, channel) in mask.into_iter().zip(channels) {
        if enabled {
            writes |= channel;
        }
    }
    writes
}

pub(crate) fn vertex_format(format: state::VertexFormat) -> wgpu::VertexFormat {
    match format {
        state::VertexFormat::Float32 => wgpu::VertexFormat::Float32,
        state::VertexFormat::Float32x2 => wgpu::VertexFormat::Float32x2,
        state::VertexFormat::Float32x3 => wgpu::VertexFormat::Float32x3,
        state::VertexFormat::Float32x4 => wgpu::VertexFormat::Float32x4,
        state::VertexFormat::Float16x2 => wgpu::VertexFormat::Float16x2,
        state::VertexFormat::Float16x4 => wgpu::VertexFormat::Float16x4,
        state::VertexFormat::Unorm16x2 => wgpu::VertexFormat::Unorm16x2,
        state::VertexFormat::Unorm16x4 => wgpu::VertexFormat::Unorm16x4,
        state::VertexFormat::Snorm16x2 => wgpu::VertexFormat::Snorm16x2,
        state::VertexFormat::Snorm16x4 => wgpu::VertexFormat::Snorm16x4,
        state::VertexFormat::Sint16x2 => wgpu::VertexFormat::Sint16x2,
        state::VertexFormat::Sint16x4 => wgpu::VertexFormat::Sint16x4,
        state::VertexFormat::Uint16x2 => wgpu::VertexFormat::Uint16x2,
        state::VertexFormat::Uint16x4 => wgpu::VertexFormat::Uint16x4,
        state::VertexFormat::Unorm8x4 => wgpu::VertexFormat::Unorm8x4,
        state::VertexFormat::Snorm8x4 => wgpu::VertexFormat::Snorm8x4,
        state::VertexFormat::Sint8x4 => wgpu::VertexFormat::Sint8x4,
        state::VertexFormat::Uint8x4 => wgpu::VertexFormat::Uint8x4,
    }
}

pub(crate) fn blend_factor(factor: state::BlendFactor) -> wgpu::BlendFactor {
    match factor {
        state::BlendFactor::Zero => wgpu::BlendFactor::Zero,
        state::BlendFactor::One => wgpu::BlendFactor::One,
        state::BlendFactor::Src => wgpu::BlendFactor::Src,
        state::BlendFactor::OneMinusSrc => wgpu::BlendFactor::OneMinusSrc,
        state::BlendFactor::SrcAlpha => wgpu::BlendFactor::SrcAlpha,
        state::BlendFactor::OneMinusSrcAlpha => wgpu::BlendFactor::OneMinusSrcAlpha,
        state::BlendFactor::Dst => wgpu::BlendFactor::Dst,
        state::BlendFactor::OneMinusDst => wgpu::BlendFactor::OneMinusDst,
        state::BlendFactor::DstAlpha => wgpu::BlendFactor::DstAlpha,
        state::BlendFactor::OneMinusDstAlpha => wgpu::BlendFactor::OneMinusDstAlpha,
        state::BlendFactor::SrcAlphaSaturated => wgpu::BlendFactor::SrcAlphaSaturated,
        state::BlendFactor::Constant => wgpu::BlendFactor::Constant,
        state::BlendFactor::OneMinusConstant => wgpu::BlendFactor::OneMinusConstant,
    }
}

pub(crate) fn blend_operation(operation: state::BlendOperation) -> wgpu::BlendOperation {
    match operation {
        state::BlendOperation::Add => wgpu::BlendOperation::Add,
        state::BlendOperation::Subtract => wgpu::BlendOperation::Subtract,
        state::BlendOperation::ReverseSubtract => wgpu::BlendOperation::ReverseSubtract,
        state::BlendOperation::Min => wgpu::BlendOperation::Min,
        state::BlendOperation::Max => wgpu::BlendOperation::Max,
    }
}

pub(crate) fn compare(compare: state::Compare) -> wgpu::CompareFunction {
    match compare {
        state::Compare::Never => wgpu::CompareFunction::Never,
        state::Compare::Less => wgpu::CompareFunction::Less,
        state::Compare::Equal => wgpu::CompareFunction::Equal,
        state::Compare::LessEqual => wgpu::CompareFunction::LessEqual,
        state::Compare::Greater => wgpu::CompareFunction::Greater,
        state::Compare::NotEqual => wgpu::CompareFunction::NotEqual,
        state::Compare::GreaterEqual => wgpu::CompareFunction::GreaterEqual,
        state::Compare::Always => wgpu::CompareFunction::Always,
    }
}

/// The depth format a device holds a guest surface in. See
/// [`switch_core::gpu::upload::DepthKind`] for why there are only two.
pub(crate) fn depth_texture_format(kind: DepthKind) -> wgpu::TextureFormat {
    match kind {
        DepthKind::Unorm16 => wgpu::TextureFormat::Depth16Unorm,
        DepthKind::Float32 => wgpu::TextureFormat::Depth32Float,
    }
}

pub(crate) fn blend(blend: state::Blend) -> wgpu::BlendState {
    let component = |c: state::BlendComponent| wgpu::BlendComponent {
        src_factor: blend_factor(c.src_factor),
        dst_factor: blend_factor(c.dst_factor),
        operation: blend_operation(c.operation),
    };
    wgpu::BlendState {
        color: component(blend.color),
        alpha: component(blend.alpha),
    }
}

/// The wgpu name for a format `switch_core::gpu::pipeline` resolved, checked
/// against what this device was actually given.
///
/// WebGPU gates the compressed families behind optional features, and creating
/// a texture in one the device does not have is not an error you can catch --
/// `createTexture` throws, and wgpu's web backend unwraps it. That is a panic
/// in the middle of a draw, which on wasm is a bare `unreachable` that takes
/// the whole core down: Just Dance 2019 reached its first BC1 texture and the
/// run loop stopped.
///
/// Refusing it here instead makes it what every other thing this backend
/// cannot express already is -- one draw on the software rasterizer.
pub(crate) fn device_texture_format(
    features: wgpu::Features,
    format: Format,
) -> Result<wgpu::TextureFormat> {
    let wanted = texture_format(format)?;
    let needs = wanted.required_features();
    if !features.contains(needs) {
        return Err(Error::Gpu(format!(
            "the device was not given {needs:?}, which {wanted:?} needs"
        )));
    }
    Ok(wanted)
}

/// How a sampled texture's texels have to be rewritten to reach the device
/// format [`sampled_texture_format`] chose for them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Widen {
    /// The device holds the guest's format itself; the bytes go as they are.
    None,
    /// Each `u16` becomes the `f32` sampling it would have produced.
    Unorm16,
    /// Each `i16` becomes the `f32` sampling it would have produced.
    Snorm16,
}

/// The device format for a texture that is only *sampled*, and what its
/// texels have to become on the way.
///
/// The normalized 16-bit formats are wgpu's `TEXTURE_FORMAT_16BIT_NORM`,
/// which is native-only: WebGPU has no spelling for them, so no browser will
/// ever offer one and A Short Hike's `R16` texture fell back on every device
/// this actually ships to — and one fallback latches the whole session onto
/// the rasterizer, so a format nothing here can hold cost every frame.
///
/// A 32-bit float sibling holds them exactly rather than approximately: an
/// `f32` has 24 bits of significand, so `v / 65535` is the same number the
/// hardware would have handed the shader, to the bit. What it costs is
/// twice the bytes on the device and `float32-filterable`, without which a
/// sampler could not filter the result — and where the device has neither
/// route this is the fallback it always was.
///
/// Only for sampled textures. A *render target* of the same format is
/// [`device_attachment_format`]'s, and stays an honest refusal: a float
/// target neither clamps nor blends the way a normalized one does, and the
/// readback that puts it back in guest memory copies device texels straight
/// into a guest surface that is still 16 bits wide.
pub(crate) fn sampled_texture_format(
    features: wgpu::Features,
    format: Format,
) -> Result<(wgpu::TextureFormat, Widen)> {
    match device_texture_format(features, format) {
        Ok(wanted) => Ok((wanted, Widen::None)),
        Err(refused) => {
            let widened = match format {
                Format::R16Unorm => Some((wgpu::TextureFormat::R32Float, Widen::Unorm16)),
                Format::R16Snorm => Some((wgpu::TextureFormat::R32Float, Widen::Snorm16)),
                Format::Rg16Unorm => Some((wgpu::TextureFormat::Rg32Float, Widen::Unorm16)),
                Format::Rg16Snorm => Some((wgpu::TextureFormat::Rg32Float, Widen::Snorm16)),
                Format::Rgba16Unorm => Some((wgpu::TextureFormat::Rgba32Float, Widen::Unorm16)),
                Format::Rgba16Snorm => Some((wgpu::TextureFormat::Rgba32Float, Widen::Snorm16)),
                _ => None,
            };
            match widened {
                Some(pair) if features.contains(wgpu::Features::FLOAT32_FILTERABLE) => Ok(pair),
                _ => Err(refused),
            }
        }
    }
}

/// The texels of a linear image whose 16-bit channels have to reach the
/// device as `f32`, which is [`Widen`]'s whole job.
///
/// Every two bytes become four, wherever they sit in the row: a row's
/// padding is as much a part of the layout as its texels, so widening the
/// row rather than the texels keeps a stride the caller can still describe
/// as twice what it was.
pub(crate) fn widen(bytes: &[u8], widen: Widen) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() * 2);
    for pair in bytes.chunks_exact(2) {
        let raw = u16::from_le_bytes([pair[0], pair[1]]);
        let value = match widen {
            // Unreachable: `Widen::None` is what "do not call this" is
            // spelled as, and every caller checks. Zero rather than a panic
            // in the middle of a draw.
            Widen::None => 0.0,
            Widen::Unorm16 => f32::from(raw) / 65535.0,
            // The most negative `i16` is one step past -1 and clamps to it,
            // which is what sampling a snorm does.
            Widen::Snorm16 => (f32::from(raw as i16) / 32767.0).max(-1.0),
        };
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// [`device_texture_format`], for a format that has to be a **colour
/// attachment** rather than only a sampled texture.
///
/// `required_features` does not cover this. `rg11b10ufloat` needs no feature
/// to be sampled and reports none, but rendering into it is gated behind
/// `RG11B10UFLOAT_RENDERABLE` — expressed in wgpu as the allowed *usages* the
/// format has given a device's features, not as a required feature. Asking
/// the usage question directly covers every format that is sampled more
/// widely than it is drawn into, rather than this one by name.
pub(crate) fn device_attachment_format(
    features: wgpu::Features,
    format: Format,
) -> Result<wgpu::TextureFormat> {
    let wanted = device_texture_format(features, format)?;
    let usages = wanted.guaranteed_format_features(features).allowed_usages;
    if !usages.contains(wgpu::TextureUsages::RENDER_ATTACHMENT) {
        return Err(Error::Gpu(format!(
            "this device cannot render into {wanted:?}"
        )));
    }
    Ok(wanted)
}

/// The wgpu name for a format `switch_core::gpu::pipeline` resolved.
pub(crate) fn texture_format(format: Format) -> Result<wgpu::TextureFormat> {
    use wgpu::TextureFormat as T;
    Ok(match format {
        Format::R8Unorm => T::R8Unorm,
        Format::R8Snorm => T::R8Snorm,
        Format::Rg8Unorm => T::Rg8Unorm,
        Format::Rg8Snorm => T::Rg8Snorm,
        Format::Rg11b10Ufloat => T::Rg11b10Ufloat,
        Format::Rgba8Unorm => T::Rgba8Unorm,
        Format::Rgba8Snorm => T::Rgba8Snorm,
        Format::Rgba8UnormSrgb => T::Rgba8UnormSrgb,
        Format::Bgra8Unorm => T::Bgra8Unorm,
        Format::Bgra8UnormSrgb => T::Bgra8UnormSrgb,
        Format::Rgb10a2Unorm => T::Rgb10a2Unorm,
        Format::R32Float => T::R32Float,
        Format::Rg32Float => T::Rg32Float,
        Format::R16Float => T::R16Float,
        Format::Rg16Float => T::Rg16Float,
        Format::R16Unorm => T::R16Unorm,
        Format::R16Snorm => T::R16Snorm,
        Format::Rg16Unorm => T::Rg16Unorm,
        Format::Rg16Snorm => T::Rg16Snorm,
        Format::Rgba16Unorm => T::Rgba16Unorm,
        Format::Rgba16Snorm => T::Rgba16Snorm,
        Format::Rgba16Float => T::Rgba16Float,
        Format::Rgba32Float => T::Rgba32Float,
        Format::Depth16Unorm => T::Depth16Unorm,
        Format::Depth24Plus => T::Depth24Plus,
        Format::Depth24PlusStencil8 => T::Depth24PlusStencil8,
        Format::Depth32Float => T::Depth32Float,
        Format::Depth32FloatStencil8 => T::Depth32FloatStencil8,
        Format::Bc1RgbaUnorm => T::Bc1RgbaUnorm,
        Format::Bc1RgbaUnormSrgb => T::Bc1RgbaUnormSrgb,
        Format::Bc2RgbaUnorm => T::Bc2RgbaUnorm,
        Format::Bc2RgbaUnormSrgb => T::Bc2RgbaUnormSrgb,
        Format::Bc3RgbaUnorm => T::Bc3RgbaUnorm,
        Format::Bc3RgbaUnormSrgb => T::Bc3RgbaUnormSrgb,
        Format::Bc4RUnorm => T::Bc4RUnorm,
        Format::Bc4RSnorm => T::Bc4RSnorm,
        Format::Bc5RgUnorm => T::Bc5RgUnorm,
        Format::Bc5RgSnorm => T::Bc5RgSnorm,
        Format::Bc6hRgbUfloat => T::Bc6hRgbUfloat,
        Format::Bc6hRgbFloat => T::Bc6hRgbFloat,
        Format::Bc7RgbaUnorm => T::Bc7RgbaUnorm,
        Format::Bc7RgbaUnormSrgb => T::Bc7RgbaUnormSrgb,
    })
}

#[cfg(test)]
mod tests {
    use super::{sampled_texture_format, widen, Widen};
    use switch_core::gpu::pipeline::Format;
    use switch_core::gpu::surface::ColorFormat;

    /// Every browser device is the middle case: WebGPU has no spelling for
    /// the normalized 16-bit formats, so wgpu's is native-only and an
    /// adapter on the web never reports it.
    #[test]
    fn a_sixteen_bit_norm_texture_is_widened_only_where_it_has_to_be() {
        let native = wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
        assert_eq!(
            sampled_texture_format(native, Format::R16Unorm).unwrap(),
            (wgpu::TextureFormat::R16Unorm, Widen::None),
            "a device that holds the format itself should be given it"
        );

        let web = wgpu::Features::FLOAT32_FILTERABLE;
        for (format, wanted, how) in [
            (
                Format::R16Unorm,
                wgpu::TextureFormat::R32Float,
                Widen::Unorm16,
            ),
            (
                Format::Rg16Snorm,
                wgpu::TextureFormat::Rg32Float,
                Widen::Snorm16,
            ),
            (
                Format::Rgba16Unorm,
                wgpu::TextureFormat::Rgba32Float,
                Widen::Unorm16,
            ),
        ] {
            assert_eq!(
                sampled_texture_format(web, format).unwrap(),
                (wanted, how),
                "{format:?} should widen where the device cannot hold it"
            );
        }

        // Without a filterable float there is no route, and the refusal is
        // the one this always answered.
        assert!(sampled_texture_format(wgpu::Features::empty(), Format::R16Unorm).is_err());
        // And nothing else is widened: a compressed family the adapter
        // lacks is still a draw for the rasterizer.
        assert!(sampled_texture_format(web, Format::Bc1RgbaUnorm).is_err());
    }

    /// The claim the widening rests on: an `f32` holds `v / 65535` exactly,
    /// so the number the shader samples is the one the rasterizer decodes
    /// rather than one near it.
    #[test]
    fn a_widened_channel_is_the_number_the_rasterizer_decodes() {
        // `0xEE` is R16Unorm and `0xEF` R16Snorm, as `pipeline`'s table
        // reads them.
        for (raw_format, how) in [(0xEE, Widen::Unorm16), (0xEF, Widen::Snorm16)] {
            let reference = ColorFormat::from_raw(raw_format).expect("a 16-bit red format");
            for stored in [
                0u16, 1, 0x0100, 0x1234, 0x7fff, 0x8000, 0x8001, 0xfffe, 0xffff,
            ] {
                let widened = widen(&stored.to_le_bytes(), how);
                let got = f32::from_le_bytes(widened.try_into().expect("one f32"));
                let want = reference.decode(u128::from(stored)).expect("a decode")[0];
                assert_eq!(got, want, "{how:?} of {stored:#06x}");
            }
        }
    }

    /// A row is widened whole, padding included, so the stride the caller
    /// describes stays twice the one it had.
    #[test]
    fn widening_doubles_every_byte_of_a_row() {
        let row = [0u8; 12];
        assert_eq!(widen(&row, Widen::Unorm16).len(), 24);
    }
}
