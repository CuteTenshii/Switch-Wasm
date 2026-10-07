//! Wrapping a translation into a complete module with its entry points.

use super::layout::{
    attribute_scalar, unpack_1010102, Coverage, Layout, Stage, ATTRIBUTE_WORDS, GENERIC_BASE,
    GENERIC_STRIDE, GLOBAL_BINDING, IDENTITY_SWIZZLE, INSTANCE_ID, POSITION, TEXTURE_BINDING,
    VERTEX_ID,
};
use super::{Translation, Unsupported};
use crate::gpu::pipeline::AttributeBase;
use crate::gpu::shader::isa::TexDim;
use crate::gpu::texture::SwizzleSource;

/// `quadSwapX`/`Y`/`Diagonal` from fine derivatives, for devices without quad
/// operations (every browser). Exact because 16-bit halves survive the f32 round trip.
const QUAD_SWAP: &str = "\
fn quadSwapHalves(low: f32, high: f32, d_low: f32, d_high: f32, sign: f32) -> u32 {
  let other_low = clamp(round(low + sign * d_low), 0.0, 65535.0);
  let other_high = clamp(round(high + sign * d_high), 0.0, 65535.0);
  return (u32(other_high) << 16u) | u32(other_low);
}

fn quadSwapX(v: u32) -> u32 {
  // `quad_lane`'s low bit is the column parity: the left lane of the pair
  // adds the difference, the right one subtracts it.
  let low = f32(v & 0xffffu);
  let high = f32(v >> 16u);
  let sign = select(1.0, -1.0, (quad_lane & 1u) == 1u);
  return quadSwapHalves(low, high, dpdxFine(low), dpdxFine(high), sign);
}

fn quadSwapY(v: u32) -> u32 {
  // And its second bit is the row parity, which `dpdyFine` runs down just as
  // `position.y` does.
  let low = f32(v & 0xffffu);
  let high = f32(v >> 16u);
  let sign = select(1.0, -1.0, (quad_lane & 2u) == 2u);
  return quadSwapHalves(low, high, dpdyFine(low), dpdyFine(high), sign);
}

fn quadSwapDiagonal(v: u32) -> u32 {
  // The lane across the quad is the one across the row from the one across
  // the column, and the first swap's answer is an exact word to take the
  // second derivative of.
  return quadSwapY(quadSwapX(v));
}";

/// Wrap a translation into a complete shader module: bindings, attribute
/// space and entry point. Varyings are `@interpolate(linear)` carrying value/w,
/// because Maxwell's `ipa` finishes the perspective divide itself.
pub fn module(
    translated: &Translation,
    stage: Stage,
    layout: &Layout,
) -> Result<String, Unsupported> {
    let mut out = String::new();
    // A quad outside the fragment stage is unsupported.
    if let Some(at) = translated.quad {
        if stage != Stage::Fragment {
            return Err(Unsupported::Quad { at });
        }
    }
    // Directives first, then the lane index.
    if translated.quad_swap.is_some() {
        if translated.subgroups {
            if translated.subgroup_enable {
                out.push_str("enable subgroups;\n\n");
            }
        } else {
            // Derivatives sit inside the dispatch loop, so silence uniformity analysis.
            out.push_str("diagnostic(off, derivative_uniformity);\n\n");
        }
    }
    if translated.quad.is_some() {
        out.push_str("var<private> quad_lane: u32;\n");
        out.push_str("fn quadLane() -> u32 { return quad_lane; }\n\n");
    }
    if translated.quad_swap.is_some() && !translated.subgroups {
        out.push_str(QUAD_SWAP);
        out.push_str("\n\n");
    }
    for bank in &layout.const_banks {
        out.push_str(&format!(
            "@group({}) @binding({bank}) var<storage, read> cb{bank}: array<u32>;\n",
            layout.group
        ));
    }
    for (index, texture) in layout.textures.iter().enumerate() {
        let binding = TEXTURE_BINDING + 2 * index as u32;
        out.push_str(&format!(
            "@group({}) @binding({binding}) var tex{index}: {};\n",
            layout.group,
            texture_type(texture.dim, texture.compare)?
        ));
        out.push_str(&format!(
            "@group({}) @binding({}) var smp{index}: {};\n",
            layout.group,
            binding + 1,
            if texture.compare {
                "sampler_comparison"
            } else {
                "sampler"
            }
        ));
    }
    for slot in 0..layout.globals.len() {
        out.push_str(&format!(
            "@group({}) @binding({}) var<storage, read> g{slot}: array<u32>;\n",
            layout.group,
            GLOBAL_BINDING + slot as u32
        ));
    }
    if !layout.const_banks.is_empty() || !layout.textures.is_empty() || !layout.globals.is_empty() {
        out.push('\n');
    }
    if !layout.globals.is_empty() {
        out.push_str("fn gRead(slot: u32, offset: u32) -> u32 {\n  switch (slot) {\n");
        for slot in 0..layout.globals.len() {
            out.push_str(&format!(
                "    case {slot}u: {{ return g{slot}[offset >> 2u]; }}\n"
            ));
        }
        // Out of range reads zero.
        out.push_str("    default: { return 0u; }\n  }\n}\n\n");
    }

    // `a[]` in and out are separate so they do not alias.
    out.push_str(&format!(
        "var<private> attr_in: array<f32, {ATTRIBUTE_WORDS}>;\n"
    ));
    out.push_str(&format!(
        "var<private> attr_out: array<f32, {ATTRIBUTE_WORDS}>;\n\n"
    ));
    let mask = ATTRIBUTE_WORDS - 1;
    out.push_str(&format!(
        "fn attrIn(offset: u32) -> f32 {{ return attr_in[(offset >> 2u) & {mask}u]; }}\n"
    ));
    out.push_str(&format!(
        "fn attrOut(offset: u32, value: f32) {{ attr_out[(offset >> 2u) & {mask}u] = value; }}\n\n"
    ));

    out.push_str("fn cbRead(bank: u32, offset: u32) -> u32 {\n  switch (bank) {\n");
    for bank in &layout.const_banks {
        out.push_str(&format!(
            "    case {bank}u: {{ return cb{bank}[offset >> 2u]; }}\n"
        ));
    }
    // An unbound bank reads zero.
    out.push_str("    default: { return 0u; }\n  }\n}\n\n");

    // `dim` is unused here but part of `HOST_INTERFACE`.
    out.push_str(
        "fn texSample(imm: u32, dim: u32, u: f32, v: f32, layer: u32, w: f32) -> vec4<f32> {\n\
         \x20 switch (imm) {\n",
    );
    for (index, texture) in layout.textures.iter().enumerate() {
        if texture.compare {
            continue;
        }
        let coords = match texture.dim {
            TexDim::T2d => "vec2<f32>(u, v), 0.0",
            TexDim::T2dArray => "vec2<f32>(u, v), layer, 0.0",
            TexDim::T3d | TexDim::TCube => "vec3<f32>(u, v, w), 0.0",
            TexDim::TCubeArray => "vec3<f32>(u, v, w), layer, 0.0",
            other => return Err(Unsupported::TextureDimension { dim: other }),
        };
        let key = texture.slot.key();
        let sample = format!("textureSampleLevel(tex{index}, smp{index}, {coords})");
        out.push_str(&format!(
            "    case {key}u: {{ {} }}\n",
            return_swizzled(&sample, texture.swizzle)
        ));
    }
    out.push_str("    default: { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }\n  }\n}\n\n");

    // Shadow samples: alpha one, the comparison in the rest, as Eden's `Extract`.
    out.push_str(
        "fn texSampleCompare(imm: u32, dim: u32, u: f32, v: f32, layer: u32, dref: f32)\
         \x20-> vec4<f32> {\n\
         \x20 switch (imm) {\n",
    );
    for (index, texture) in layout.textures.iter().enumerate() {
        if !texture.compare {
            continue;
        }
        let coords = match texture.dim {
            TexDim::T2d => "vec2<f32>(u, v)",
            TexDim::T2dArray => "vec2<f32>(u, v), layer",
            other => return Err(Unsupported::TextureDimension { dim: other }),
        };
        let key = texture.slot.key();
        out.push_str(&format!(
            "    case {key}u: {{\n      let c = textureSampleCompareLevel(tex{index}, \
             smp{index}, {coords}, dref);\n      return vec4<f32>(c, c, c, 1.0);\n    }}\n"
        ));
    }
    out.push_str("    default: { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }\n  }\n}\n\n");

    // One case per binding and offset: offsets must be constants.
    out.push_str(
        "fn texSampleOffset(imm: u32, offset: u32, u: f32, v: f32, layer: u32) -> vec4<f32> {\n  \
         switch (imm) {\n",
    );
    for (index, texture) in layout.textures.iter().enumerate() {
        let coords = match (texture.dim, texture.compare) {
            (TexDim::T2d, false) => "vec2<f32>(u, v), 0.0",
            (TexDim::T2dArray, false) => "vec2<f32>(u, v), layer, 0.0",
            _ => continue,
        };
        let key = texture.slot.key();
        out.push_str(&format!("    case {key}u: {{\n      switch (offset) {{\n"));
        for (n, (x, y)) in layout.texture_offsets.iter().enumerate() {
            let sample = format!(
                "textureSampleLevel(tex{index}, smp{index}, {coords}, vec2<i32>({x}, {y}))"
            );
            out.push_str(&format!(
                "        case {n}u: {{ {} }}\n",
                return_swizzled(&sample, texture.swizzle)
            ));
        }
        out.push_str("        default: { return vec4<f32>(0.0); }\n      }\n    }\n");
    }
    out.push_str("    default: { return vec4<f32>(0.0); }\n  }\n}\n\n");

    // `txq` sizes, as `interp`'s `run_txq` answers.
    out.push_str("fn texDims(imm: u32, lod: u32) -> vec4<u32> {\n  switch (imm) {\n");
    for (index, texture) in layout.textures.iter().enumerate() {
        let depth = match texture.dim {
            TexDim::T2dArray => format!("textureNumLayers(tex{index})"),
            TexDim::T3d => "s.z".to_string(),
            TexDim::TCube => "6u".to_string(),
            TexDim::TCubeArray => format!("textureNumLayers(tex{index}) * 6u"),
            _ => "1u".to_string(),
        };
        let key = texture.slot.key();
        out.push_str(&format!(
            "    case {key}u: {{\n      let s = textureDimensions(tex{index});\n      \
             return vec4<u32>(select(max(s.x >> lod, 1u), 1u, lod >= 32u), \
             select(max(s.y >> lod, 1u), 1u, lod >= 32u), {depth}, \
             textureNumLevels(tex{index}));\n    }}\n"
        ));
    }
    out.push_str("    default: { return vec4<u32>(1u, 1u, 1u, 1u); }\n  }\n}\n\n");

    // One call per gather channel, through the descriptor's swizzle.
    out.push_str(
        "fn texGather(imm: u32, component: u32, u: f32, v: f32, layer: u32) -> vec4<f32> {\n  \
         switch (imm) {\n",
    );
    for (index, texture) in layout.textures.iter().enumerate() {
        let coords = match (texture.dim, texture.compare) {
            (TexDim::T2d, false) => "vec2<f32>(u, v)",
            (TexDim::T2dArray, false) => "vec2<f32>(u, v), layer",
            _ => continue,
        };
        let key = texture.slot.key();
        out.push_str(&format!(
            "    case {key}u: {{\n      switch (component) {{\n"
        ));
        for (channel, source) in texture.swizzle.iter().enumerate() {
            let texels = match source {
                SwizzleSource::R => format!("textureGather(0, tex{index}, smp{index}, {coords})"),
                SwizzleSource::G => format!("textureGather(1, tex{index}, smp{index}, {coords})"),
                SwizzleSource::B => format!("textureGather(2, tex{index}, smp{index}, {coords})"),
                SwizzleSource::A => format!("textureGather(3, tex{index}, smp{index}, {coords})"),
                SwizzleSource::Zero => "vec4<f32>(0.0)".to_string(),
                SwizzleSource::One => "vec4<f32>(1.0)".to_string(),
            };
            out.push_str(&format!(
                "        case {channel}u: {{ return {texels}; }}\n"
            ));
        }
        out.push_str("        default: { return vec4<f32>(0.0); }\n      }\n    }\n");
    }
    out.push_str("    default: { return vec4<f32>(0.0); }\n  }\n}\n\n");

    out.push_str(&translated.source);
    out.push('\n');
    out.push_str(&match stage {
        Stage::Vertex => vertex_entry(layout),
        Stage::Fragment => fragment_entry(translated, layout),
    });
    Ok(out)
}

/// `return` a sample rearranged by a descriptor swizzle.
fn return_swizzled(sample: &str, swizzle: [SwizzleSource; 4]) -> String {
    if swizzle == IDENTITY_SWIZZLE {
        return format!("return {sample};");
    }
    let channels: Vec<&str> = swizzle
        .iter()
        .map(|source| match source {
            SwizzleSource::Zero => "0.0",
            SwizzleSource::R => "sampled.x",
            SwizzleSource::G => "sampled.y",
            SwizzleSource::B => "sampled.z",
            SwizzleSource::A => "sampled.w",
            SwizzleSource::One => "1.0",
        })
        .collect();
    format!(
        "let sampled = {sample}; return vec4<f32>({});",
        channels.join(", ")
    )
}

/// The WGSL type a `texs` of this dimensionality samples.
fn texture_type(dim: TexDim, compare: bool) -> Result<&'static str, Unsupported> {
    match (dim, compare) {
        (TexDim::T2d, false) => Ok("texture_2d<f32>"),
        (TexDim::T2dArray, false) => Ok("texture_2d_array<f32>"),
        (TexDim::T3d, false) => Ok("texture_3d<f32>"),
        (TexDim::TCube, false) => Ok("texture_cube<f32>"),
        (TexDim::TCubeArray, false) => Ok("texture_cube_array<f32>"),
        (TexDim::T2d, true) => Ok("texture_depth_2d"),
        (TexDim::T2dArray, true) => Ok("texture_depth_2d_array"),
        (dim, _) => Err(Unsupported::TextureDimension { dim }),
    }
}

fn generic_word(slot: usize, component: usize) -> usize {
    (GENERIC_BASE + slot * GENERIC_STRIDE) / 4 + component
}

fn gather(array: &str, base: usize) -> String {
    let words: Vec<String> = (0..4).map(|c| format!("{array}[{}u]", base + c)).collect();
    format!("vec4<f32>({})", words.join(", "))
}

fn vertex_entry(layout: &Layout) -> String {
    let mut out = String::new();
    if !layout.attributes.is_empty() {
        out.push_str("struct VertexInput {\n");
        for slot in &layout.attributes {
            if layout.packing(*slot).is_some() {
                out.push_str(&format!("  @location({slot}) attr{slot}: u32,\n"));
                continue;
            }
            out.push_str(&format!(
                "  @location({slot}) attr{slot}: vec4<{}>,\n",
                attribute_scalar(layout.attribute_base(*slot))
            ));
        }
        out.push_str("}\n\n");
    }
    out.push_str("struct VertexOutput {\n  @builtin(position) position: vec4<f32>,\n");
    for slot in &layout.varyings {
        out.push_str(&format!(
            "  @location({slot}) @interpolate(linear{}) vary{slot}: vec4<f32>,\n",
            layout.sampling(*slot)
        ));
    }
    out.push_str("}\n\n@vertex\nfn vs_main(\n");
    if !layout.attributes.is_empty() {
        out.push_str("  input: VertexInput,\n");
    }
    out.push_str(
        "  @builtin(vertex_index) vertex: u32,\n\
         \x20 @builtin(instance_index) instance: u32,\n\
         ) -> VertexOutput {\n",
    );
    out.push_str(&format!(
        "  attr_in[{}u] = bitcast<f32>(instance);\n  attr_in[{}u] = bitcast<f32>(vertex);\n",
        INSTANCE_ID / 4,
        VERTEX_ID / 4
    ));
    for slot in &layout.attributes {
        if let Some(packing) = layout.packing(*slot) {
            for (component, value) in unpack_1010102(&format!("input.attr{slot}"), packing)
                .iter()
                .enumerate()
            {
                out.push_str(&format!(
                    "  attr_in[{}u] = {value};\n",
                    generic_word(*slot, component)
                ));
            }
            continue;
        }
        // Integer attributes are carried as bits.
        let integer = layout.attribute_base(*slot) != AttributeBase::Float;
        // BGRA swaps the first and third components.
        let axes = if layout.bgra_attributes.contains(slot) {
            ["z", "y", "x", "w"]
        } else {
            ["x", "y", "z", "w"]
        };
        for (component, axis) in axes.iter().enumerate() {
            let read = format!("input.attr{slot}.{axis}");
            let read = if integer {
                format!("bitcast<f32>({read})")
            } else {
                read
            };
            out.push_str(&format!(
                "  attr_in[{}u] = {read};\n",
                generic_word(*slot, component)
            ));
        }
    }
    // An unwritten clip position is (0, 0, 0, 1).
    out.push_str(&format!("  attr_out[{}u] = 1.0;\n", POSITION / 4 + 3));
    out.push_str("  run();\n  var out: VertexOutput;\n");
    out.push_str(&format!(
        "  out.position = {};\n",
        gather("attr_out", POSITION / 4)
    ));
    if layout.depth_minus_one_to_one {
        out.push_str("  // Maxwell clips z from -w to w and WebGPU clips it from 0 to w;\n");
        out.push_str("  // left alone, the near half of the frustum is clipped away.\n");
        out.push_str("  out.position.z = (out.position.z + out.position.w) * 0.5;\n");
    }
    if layout.flip_y {
        out.push_str("  // The viewport transform mirrors y, and WebGPU has no\n");
        out.push_str("  // negative viewport height to mirror it with.\n");
        out.push_str("  out.position.y = -out.position.y;\n");
    }
    if !layout.varyings.is_empty() {
        // The fragment stage receives value/w.
        out.push_str("  let over_w = 1.0 / out.position.w;\n");
        for slot in &layout.varyings {
            out.push_str(&format!(
                "  out.vary{slot} = {} * over_w;\n",
                gather("attr_out", generic_word(*slot, 0))
            ));
        }
    }
    out.push_str("  return out;\n}\n");
    out
}

fn fragment_entry(translated: &Translation, layout: &Layout) -> String {
    let colour = |target: u32| -> String {
        let components: Vec<String> = (0..4)
            .map(|c| {
                let reg = (target * 4 + c) as u8;
                if translated.registers.contains(&reg) {
                    format!("bitcast<f32>(r{reg})")
                } else {
                    // An unwritten channel is zero.
                    "0.0".to_string()
                }
            })
            .collect();
        format!("vec4<f32>({})", components.join(", "))
    };

    let mut out =
        String::from("struct FragmentInput {\n  @builtin(position) position: vec4<f32>,\n");
    for slot in &layout.varyings {
        out.push_str(&format!(
            "  @location({slot}) @interpolate(linear{}) vary{slot}: vec4<f32>,\n",
            layout.sampling(*slot)
        ));
    }
    out.push_str("}\n\n");
    let targets = layout.targets;
    if targets > 1 {
        out.push_str("struct FragmentOutput {\n");
        for target in 0..targets {
            out.push_str(&format!(
                "  @location({target}) target{target}: vec4<f32>,\n"
            ));
        }
        out.push_str("}\n\n");
    }
    let returns = match targets {
        0 => String::new(),
        1 => " -> @location(0) vec4<f32>".to_string(),
        _ => " -> FragmentOutput".to_string(),
    };
    out.push_str(&format!(
        "@fragment\nfn fs_main(input: FragmentInput){returns} {{\n"
    ));
    // Coverage first: the sample mask applies before shading.
    if let Some(coverage) = &layout.coverage {
        out.push_str(&sample_index(coverage));
        if coverage.sample_mask != u32::MAX {
            out.push_str(&format!(
                "  if (({}u >> sample) & 1u) == 0u {{ discard; }}\n",
                coverage.sample_mask
            ));
        }
    }
    // Quad lane numbered as `raster::quad_pixel`: row parity above column parity.
    if translated.quad.is_some() {
        out.push_str(
            "  quad_lane = (u32(input.position.y) & 1u) * 2u + (u32(input.position.x) & 1u);\n",
        );
    }
    // Fragment `position.w` is `1/w`, as `a[0x7c]` holds.
    out.push_str(&format!(
        "  attr_in[{}u] = input.position.w;\n",
        POSITION / 4 + 3
    ));
    for slot in &layout.varyings {
        for (component, axis) in ["x", "y", "z", "w"].iter().enumerate() {
            out.push_str(&format!(
                "  attr_in[{}u] = input.vary{slot}.{axis};\n",
                generic_word(*slot, component)
            ));
        }
    }
    out.push_str("  if (run()) { discard; }\n");
    // Alpha-to-coverage applies after shading.
    let alpha_to_coverage = layout.coverage.as_ref().filter(|c| c.alpha_to_coverage);
    if targets > 1 {
        out.push_str("  var out: FragmentOutput;\n");
        for target in 0..targets {
            out.push_str(&format!("  out.target{target} = {};\n", colour(target)));
        }
        if let Some(coverage) = alpha_to_coverage {
            out.push_str(&alpha_coverage(coverage, "out.target0.w"));
        }
        out.push_str("  return out;\n");
    } else if let Some(coverage) = alpha_to_coverage {
        // Shaded even depth-only, for alpha-to-coverage.
        out.push_str(&format!("  let target0 = {};\n", colour(0)));
        out.push_str(&alpha_coverage(coverage, "target0.w"));
        if targets == 1 {
            out.push_str("  return target0;\n");
        }
    } else if targets == 1 {
        out.push_str(&format!("  return {};\n", colour(0)));
    }
    out.push_str("}\n");
    out
}

/// Which sample of its pixel this fragment is, by table lookup.
fn sample_index(coverage: &Coverage) -> String {
    let slots: Vec<String> = coverage
        .sample_of_slot
        .iter()
        .map(|s| format!("{s}u"))
        .collect();
    let count = slots.len();
    format!(
        "  var sample_of_slot = array<u32, {count}>({});\n\
         \x20 let tile = vec2<u32>(u32(input.position.x) % {}u, u32(input.position.y) % {}u);\n\
         \x20 let sample = sample_of_slot[tile.y * {}u + tile.x];\n",
        slots.join(", "),
        coverage.samples_x,
        coverage.samples_y,
        coverage.samples_x,
    )
}

/// Samples kept for `alpha`: a prefix of `round(alpha * count)`.
fn alpha_coverage(coverage: &Coverage, alpha: &str) -> String {
    let count = coverage.samples_x * coverage.samples_y;
    // `floor(x + 0.5)`: WGSL's `round` ties to even, Rust's away from zero.
    format!(
        "  if (sample >= u32(floor(clamp({alpha}, 0.0, 1.0) * {}.0 + 0.5))) {{ discard; }}\n",
        count
    )
}
