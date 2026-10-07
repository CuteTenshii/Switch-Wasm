//! Maxwell pipeline state mapped to wgpu names. Format lookups return `Result`
//! because not every guest format is available on the device.

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
        state::VertexFormat::Sint32 => wgpu::VertexFormat::Sint32,
        state::VertexFormat::Sint32x2 => wgpu::VertexFormat::Sint32x2,
        state::VertexFormat::Sint32x3 => wgpu::VertexFormat::Sint32x3,
        state::VertexFormat::Sint32x4 => wgpu::VertexFormat::Sint32x4,
        state::VertexFormat::Uint32 => wgpu::VertexFormat::Uint32,
        state::VertexFormat::Uint32x2 => wgpu::VertexFormat::Uint32x2,
        state::VertexFormat::Uint32x3 => wgpu::VertexFormat::Uint32x3,
        state::VertexFormat::Uint32x4 => wgpu::VertexFormat::Uint32x4,
        state::VertexFormat::Float16 => wgpu::VertexFormat::Float16,
        state::VertexFormat::Unorm16 => wgpu::VertexFormat::Unorm16,
        state::VertexFormat::Snorm16 => wgpu::VertexFormat::Snorm16,
        state::VertexFormat::Sint16 => wgpu::VertexFormat::Sint16,
        state::VertexFormat::Uint16 => wgpu::VertexFormat::Uint16,
        state::VertexFormat::Unorm8 => wgpu::VertexFormat::Unorm8,
        state::VertexFormat::Snorm8 => wgpu::VertexFormat::Snorm8,
        state::VertexFormat::Sint8 => wgpu::VertexFormat::Sint8,
        state::VertexFormat::Uint8 => wgpu::VertexFormat::Uint8,
        state::VertexFormat::Unorm8x2 => wgpu::VertexFormat::Unorm8x2,
        state::VertexFormat::Snorm8x2 => wgpu::VertexFormat::Snorm8x2,
        state::VertexFormat::Sint8x2 => wgpu::VertexFormat::Sint8x2,
        state::VertexFormat::Uint8x2 => wgpu::VertexFormat::Uint8x2,
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
        // WebGPU has no signed or integer 10-10-10-2 format; fetched as a word and unpacked in the entry point.
        state::VertexFormat::Packed1010102(_) => wgpu::VertexFormat::Uint32,
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

/// The device depth format for a guest surface (see [`switch_core::gpu::upload::DepthKind`]).
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

/// The wgpu format for a resolved guest format, refused if the device lacks the
/// feature (creating it would panic), so the draw falls back to the rasterizer.
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

/// How sampled texels are rewritten for the format [`sampled_texture_format`] chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Widen {
    None,
    Unorm16,
    Snorm16,
}

/// Device format for a sampled-only texture, widening 16-bit norm formats to
/// `f32` (exact) where the device lacks them, as every browser does.
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

/// Widen 16-bit channels to `f32`, row padding included, so the stride doubles.
pub(crate) fn widen(bytes: &[u8], widen: Widen) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() * 2);
    for pair in bytes.as_chunks::<2>().0 {
        let raw = u16::from_le_bytes(*pair);
        let value = match widen {
            // Never called with `None`; zero rather than a panic mid-draw.
            Widen::None => 0.0,
            Widen::Unorm16 => f32::from(raw) / 65535.0,
            // `i16::MIN` clamps to -1, as snorm sampling does.
            Widen::Snorm16 => (f32::from(raw as i16) / 32767.0).max(-1.0),
        };
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// [`device_texture_format`] for a colour attachment, checked by allowed usages
/// rather than required features (e.g. `rg11b10ufloat`).
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

    /// Browser devices never report the 16-bit norm formats.
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

        assert!(sampled_texture_format(wgpu::Features::empty(), Format::R16Unorm).is_err());
        assert!(sampled_texture_format(web, Format::Bc1RgbaUnorm).is_err());
    }

    /// An `f32` holds `v / 65535` exactly.
    #[test]
    fn a_widened_channel_is_the_number_the_rasterizer_decodes() {
        // `0xEE` is R16Unorm and `0xEF` R16Snorm.
        for (raw_format, how) in [(0xEE, Widen::Unorm16), (0xEF, Widen::Snorm16)] {
            let reference = ColorFormat::from_raw(raw_format).expect("a 16-bit red format");
            let stored = [
                0u16, 1, 0x0100, 0x1234, 0x7fff, 0x8000, 0x8001, 0xfffe, 0xffff,
            ];
            let row: Vec<u8> = stored.iter().flat_map(|s| s.to_le_bytes()).collect();
            let widened = widen(&row, how);
            assert_eq!(widened.len(), row.len() * 2, "{how:?}");
            for (value, chunk) in stored.iter().zip(widened.as_chunks::<4>().0) {
                let got = f32::from_le_bytes(*chunk);
                let want = reference.decode(u128::from(*value)).expect("a decode")[0];
                assert_eq!(got, want, "{how:?} of {value:#06x}");
            }
        }
    }
}
