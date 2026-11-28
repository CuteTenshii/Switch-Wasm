//! Vertex attribute types as a shader's inputs see them.

/// A vertex format's component base type, which picks the shader input type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttributeBase {
    Float,
    Sint,
    Uint,
    /// An integer the shader reads as the float of its value.
    Sscaled,
    Uscaled,
}

/// A 10-10-10-2 attribute (size `0x30`), red in the low bits. WebGPU only has
/// `unorm10-10-10-2`, so backends fetch a word and unpack it in the entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Packed1010102 {
    Snorm,
    Unorm,
    Sint,
    Uint,
}
