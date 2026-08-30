//! What a piece of Maxwell pipeline state is called on the device.
//!
//! Every function here maps one of `switch_core::gpu::pipeline`'s enums onto
//! wgpu's name for the same thing. The format entry points are the exception:
//! a format the guest names is not always one this device can hold, still
//! less one it can draw into, so those answer `Result`.

use switch_core::gpu::pipeline::{self as state, Format};
use switch_core::gpu::upload::DepthKind;
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
    device: &wgpu::Device,
    format: Format,
) -> Result<wgpu::TextureFormat> {
    let wanted = texture_format(format)?;
    let needs = wanted.required_features();
    if !device.features().contains(needs) {
        return Err(Error::Gpu(format!(
            "the device was not given {needs:?}, which {wanted:?} needs"
        )));
    }
    Ok(wanted)
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
    device: &wgpu::Device,
    format: Format,
) -> Result<wgpu::TextureFormat> {
    let wanted = device_texture_format(device, format)?;
    let usages = wanted
        .guaranteed_format_features(device.features())
        .allowed_usages;
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
