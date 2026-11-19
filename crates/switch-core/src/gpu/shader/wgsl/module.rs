//! Wrapping a translation into a complete module with its entry points.

use super::layout::{
    attribute_scalar, unpack_1010102, Coverage, Crop, Fold, Layout, Stage, StorageNumeric, Update,
    ATOMS, ATTRIBUTE_WORDS, CONSTANTS_BINDING, CONSTANT_BANKS, GENERIC_BASE, GENERIC_STRIDE,
    GLOBAL_BINDING, IDENTITY_SWIZZLE, IMAGE_BINDING, INSTANCE_ID, POSITION, TEXTURE_BINDING,
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

/// Compare-exchange attempts `gAtom` makes; an unbounded loop zeroed the buffer on an Apple M4.
const CAS_TRIES: u32 = 100_000;

/// WGSL that replaces `array[word]` with `next` (an expression over `was`) by compare-exchange,
/// leaving the word's previous value in `was`.
fn compare_exchange(array: &str, word: &str, next: &str, indent: &str) -> String {
    format!(
        "{indent}var was = atomicLoad(&{array}[{word}]);\n\
         {indent}for (var tries = 0u; tries < {CAS_TRIES}u; tries++) {{\n\
         {indent}  let seen = atomicCompareExchangeWeak(&{array}[{word}], was, {next});\n\
         {indent}  if (seen.exchanged) {{ break; }}\n\
         {indent}  was = seen.old_value;\n\
         {indent}}}\n"
    )
}

/// WGSL that stores the `mask` bits of `value` into `array[word]`, atomically.
fn merge(array: &str, word: &str, mask: &str, value: &str) -> String {
    let swap = compare_exchange(
        array,
        word,
        &format!("(was & ~{mask}) | ({value} & {mask})"),
        "          ",
    );
    format!(
        "        if ({mask} == 0xffffffffu) {{ atomicStore(&{array}[{word}], {value}); }} else {{\n{swap}        }}\n"
    )
}

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
    if stage == Stage::Vertex && layout.globals.iter().any(|g| g.written) {
        return Err(Unsupported::VertexWrite);
    }
    if let Some(at) = translated.warp.filter(|_| stage != Stage::Compute) {
        return Err(Unsupported::Quad { at });
    }
    if stage == Stage::Compute {
        // Lanes of a warp exist only in a fragment quad here.
        if let Some(at) = translated.quad.or(translated.quad_swap) {
            return Err(Unsupported::Quad { at });
        }
        if let Some(&sr) = translated
            .special
            .iter()
            .find(|sr| PACKED_SPECIAL_REGISTERS.contains(sr))
        {
            return Err(Unsupported::PackedSpecialRegister { sr });
        }
    }
    // Directives first, then the lane index.
    if translated.warp.is_some() && translated.subgroup_enable {
        out.push_str("enable subgroups;\n\n");
    }
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
    if !layout.const_banks.is_empty() {
        out.push_str(&format!(
            "@group({}) @binding({CONSTANTS_BINDING}) var<storage, read> cbs: array<u32>;\n",
            layout.group
        ));
    }
    for (index, texture) in layout.textures.iter().enumerate() {
        let smp = texture.sampler.unwrap_or(index as u32);
        let binding = TEXTURE_BINDING + 2 * index as u32;
        out.push_str(&format!(
            "@group({}) @binding({binding}) var tex{index}: {};\n",
            layout.group,
            match (texture.uint, texture.fold) {
                (true, _) => uint_texture_type(texture.dim, texture.compare)?,
                (false, Some(_)) => folded_texture_type(texture.dim, texture.compare)?,
                (false, None) => texture_type(texture.dim, texture.compare)?,
            }
        ));
        if texture.sampler.is_none() {
            out.push_str(&format!(
                "@group({}) @binding({}) var smp{smp}: {};\n",
                layout.group,
                binding + 1,
                if texture.compare {
                    "sampler_comparison"
                } else {
                    "sampler"
                }
            ));
        }
    }
    for (slot, global) in layout.globals.iter().enumerate() {
        let (access, word) = if global.written {
            ("read_write", "atomic<u32>")
        } else {
            ("read", "u32")
        };
        out.push_str(&format!(
            "@group({}) @binding({}) var<storage, {access}> g{slot}: array<{word}>;\n",
            layout.group,
            GLOBAL_BINDING + slot as u32
        ));
    }
    let mut stores = String::new();
    for (index, image) in layout.images.iter().enumerate() {
        let format = image.format.ok_or(Unsupported::ImageFormat)?;
        out.push_str(&format!(
            "@group({}) @binding({}) var img{index}: texture_storage_2d<{}, write>;\n",
            layout.group,
            IMAGE_BINDING + index as u32,
            format.name
        ));
        let texel = match format.numeric {
            StorageNumeric::Float => "bitcast<vec4<f32>>(value)",
            StorageNumeric::Uint => "value",
            StorageNumeric::Sint => "bitcast<vec4<i32>>(value)",
        };
        stores.push_str(&format!(
            "    case {}u: {{ if (all(vec2<u32>(x, y) < textureDimensions(img{index}))) {{ \
             textureStore(img{index}, vec2<u32>(x, y), {texel}); }} }}\n",
            image.slot.key()
        ));
    }
    if !layout.const_banks.is_empty()
        || !layout.textures.is_empty()
        || !layout.globals.is_empty()
        || !layout.images.is_empty()
    {
        out.push('\n');
    }
    out.push_str(&format!(
        "fn imgStore(key: u32, x: u32, y: u32, value: vec4<u32>) {{\n  switch (key) {{\n{stores}    \
         default: {{ }}\n  }}\n}}\n\n"
    ));
    if !layout.globals.is_empty() {
        out.push_str("fn gRead(slot: u32, offset: u32) -> u32 {\n  let word = offset >> 2u;\n  switch (slot) {\n");
        for (slot, global) in layout.globals.iter().enumerate() {
            let read = if global.written {
                format!("atomicLoad(&g{slot}[word])")
            } else {
                format!("g{slot}[word]")
            };
            out.push_str(&format!(
                "    case {slot}u: {{\n      if (word >= arrayLength(&g{slot})) {{ return 0u; }}\n      return {read};\n    }}\n"
            ));
        }
        // An unbound slot reads zero.
        out.push_str("    default: { return 0u; }\n  }\n}\n\n");
    }
    if layout.globals.iter().any(|g| g.written) {
        // `value` under `mask` replaces the bits at `offset`; a store off the word boundary
        // spills into the next word.
        out.push_str(
            "fn gWrite(slot: u32, offset: u32, value: u32, mask: u32) {\n  let word = offset >> 2u;\n  \
             let shift = (offset & 3u) * 8u;\n  let low_mask = mask << shift;\n  let low = value << shift;\n  \
             let high_mask = select(0u, mask >> (32u - shift), shift != 0u);\n  \
             let high = select(0u, value >> (32u - shift), shift != 0u);\n  switch (slot) {\n",
        );
        for (slot, _) in layout.globals.iter().enumerate().filter(|(_, g)| g.written) {
            let array = format!("g{slot}");
            let low = merge(&array, "word", "low_mask", "low");
            let high = merge(&array, "word + 1u", "high_mask", "high");
            out.push_str(&format!(
                "    case {slot}u: {{\n      if (word < arrayLength(&{array})) {{\n{low}      }}\n      \
                 if (high_mask != 0u && word + 1u < arrayLength(&{array})) {{\n{high}      }}\n    }}\n"
            ));
        }
        out.push_str("    default: { }\n  }\n}\n\n");

        out.push_str(
            "fn gAtom(op: u32, slot: u32, offset: u32, value: u32, stored: u32) -> u32 {\n  let word = offset >> 2u;\n  switch (slot) {\n",
        );
        for (slot, _) in layout.globals.iter().enumerate().filter(|(_, g)| g.written) {
            out.push_str(&format!(
                "    case {slot}u: {{\n      if (word < arrayLength(&g{slot})) {{\n        switch (op) {{\n"
            ));
            for (code, (_, _, update)) in ATOMS.iter().enumerate() {
                match update {
                    Update::Direct(function) => out.push_str(&format!(
                        "          case {code}u: {{ return {function}(&g{slot}[word], value); }}\n"
                    )),
                    Update::Loop(next) => out.push_str(&format!(
                        "          case {code}u: {{\n{}            return was;\n          }}\n",
                        compare_exchange(&format!("g{slot}"), "word", next, "            ")
                    )),
                }
            }
            out.push_str("          default: { }\n        }\n      }\n    }\n");
        }
        // Out of range, or no such slot, changes nothing and reads zero.
        out.push_str("    default: { }\n  }\n  return 0u;\n}\n\n");
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

    // An unbound bank, or a read past the end of one, is zero.
    if layout.const_banks.is_empty() {
        out.push_str("fn cbRead(bank: u32, offset: u32) -> u32 { return 0u; }\n\n");
    } else {
        out.push_str(&format!(
            "fn cbRead(bank: u32, offset: u32) -> u32 {{\n  \
             if (bank >= {CONSTANT_BANKS}u) {{ return 0u; }}\n  \
             let word = offset >> 2u;\n  \
             if (word >= cbs[bank * 2u + 1u]) {{ return 0u; }}\n  \
             return cbs[cbs[bank * 2u] + word];\n}}\n\n"
        ));
    }

    if layout.textures.iter().any(|t| t.crop.is_some()) {
        out.push_str(CROPPED);
    }

    // `dim` is unused here but part of `HOST_INTERFACE`.
    out.push_str(
        "fn texSample(imm: u32, dim: u32, u: f32, v: f32, layer: u32, w: f32) -> vec4<f32> {\n\
         \x20 switch (imm) {\n",
    );
    for (index, texture) in layout.textures.iter().enumerate() {
        let smp = texture.sampler.unwrap_or(index as u32);
        if texture.compare {
            continue;
        }
        if texture.uint {
            let (key, load) = (texture.slot.key(), uint_load(index, texture.dim, (0, 0))?);
            out.push_str(&format!(
                "    case {key}u: {{ {load} return bitcast<vec4<f32>>(texel); }}\n"
            ));
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
        let body = match (texture.fold, texture.crop) {
            (Some(fold), _) => format!(
                "{} {}",
                folded_coords(fold, 0),
                return_swizzled(
                    &format!("textureSampleLevel(tex{index}, smp{smp}, vec2<f32>(u, fv), fk, 0.0)"),
                    texture.swizzle
                )
            ),
            (None, Some(crop)) => return_swizzled(
                &format!(
                    "textureSampleLevel(tex{index}, smp{smp}, {}, 0.0)",
                    cropped(crop, (0, 0))
                ),
                texture.swizzle,
            ),
            (None, None) => return_swizzled(
                &format!("textureSampleLevel(tex{index}, smp{smp}, {coords})"),
                texture.swizzle,
            ),
        };
        out.push_str(&format!("    case {key}u: {{ {body} }}\n"));
    }
    out.push_str("    default: { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }\n  }\n}\n\n");

    // Shadow samples: alpha one, the comparison in the rest, as Eden's `Extract`.
    out.push_str(
        "fn texSampleCompare(imm: u32, dim: u32, u: f32, v: f32, layer: u32, dref: f32)\
         \x20-> vec4<f32> {\n\
         \x20 switch (imm) {\n",
    );
    for (index, texture) in layout.textures.iter().enumerate() {
        let smp = texture.sampler.unwrap_or(index as u32);
        if !texture.compare {
            continue;
        }
        let coords = match texture.dim {
            TexDim::T2d => "vec2<f32>(u, v)",
            TexDim::T2dArray => "vec2<f32>(u, v), layer",
            // A cube's depth image is only gathered.
            TexDim::TCube | TexDim::TCubeArray => continue,
            other => return Err(Unsupported::TextureDimension { dim: other }),
        };
        let key = texture.slot.key();
        out.push_str(&format!(
            "    case {key}u: {{\n      let c = textureSampleCompareLevel(tex{index}, \
             smp{smp}, {coords}, dref);\n      return vec4<f32>(c, c, c, 1.0);\n    }}\n"
        ));
    }
    out.push_str("    default: { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }\n  }\n}\n\n");

    // One case per binding and offset: offsets must be constants.
    out.push_str(
        "fn texSampleOffset(imm: u32, offset: u32, u: f32, v: f32, layer: u32) -> vec4<f32> {\n  \
         switch (imm) {\n",
    );
    for (index, texture) in layout.textures.iter().enumerate() {
        let smp = texture.sampler.unwrap_or(index as u32);
        let coords = match (texture.dim, texture.compare) {
            (TexDim::T2d, false) => "vec2<f32>(u, v), 0.0",
            (TexDim::T2dArray, false) => "vec2<f32>(u, v), layer, 0.0",
            _ => continue,
        };
        let key = texture.slot.key();
        out.push_str(&format!("    case {key}u: {{\n      switch (offset) {{\n"));
        for (n, &(x, y)) in layout.texture_offsets.iter().enumerate() {
            if texture.uint {
                let load = uint_load(index, texture.dim, (x, y))?;
                out.push_str(&format!(
                    "        case {n}u: {{ {load} return bitcast<vec4<f32>>(texel); }}\n"
                ));
                continue;
            }
            // A folded image takes the row offset before choosing its tile.
            let body = match (texture.fold, texture.crop) {
                (Some(fold), _) => format!(
                    "{} {}",
                    folded_coords(fold, y),
                    return_swizzled(
                        &format!(
                            "textureSampleLevel(tex{index}, smp{smp}, vec2<f32>(u, fv), fk, \
                             0.0, vec2<i32>({x}, 0))"
                        ),
                        texture.swizzle
                    )
                ),
                // The offset moves the coordinates before they are clamped, as the sampler does.
                (None, Some(crop)) => return_swizzled(
                    &format!(
                        "textureSampleLevel(tex{index}, smp{smp}, {}, 0.0)",
                        cropped(crop, (x, y))
                    ),
                    texture.swizzle,
                ),
                (None, None) => return_swizzled(
                    &format!(
                        "textureSampleLevel(tex{index}, smp{smp}, {coords}, vec2<i32>({x}, {y}))"
                    ),
                    texture.swizzle,
                ),
            };
            out.push_str(&format!("        case {n}u: {{ {body} }}\n"));
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
        // A folded image answers with the size it has, one layer and one level.
        let (height, depth) = match texture.fold {
            Some(fold) => (format!("{}u", fold.height), "1u".to_string()),
            None => ("s.y".to_string(), depth),
        };
        out.push_str(&format!(
            "    case {key}u: {{\n      let s = textureDimensions(tex{index});\n      \
             return vec4<u32>(select(max(s.x >> lod, 1u), 1u, lod >= 32u), \
             select(max({height} >> lod, 1u), 1u, lod >= 32u), {depth}, \
             textureNumLevels(tex{index}));\n    }}\n"
        ));
    }
    out.push_str("    default: { return vec4<u32>(1u, 1u, 1u, 1u); }\n  }\n}\n\n");

    // One texel in normalized coordinates, as `TextureSource::texel_step` answers.
    out.push_str("fn texStep(imm: u32) -> vec2<f32> {\n  switch (imm) {\n");
    for (index, texture) in layout.textures.iter().enumerate() {
        let height = match texture.fold {
            Some(fold) => format!("{}u", fold.height),
            None => format!("textureDimensions(tex{index}).y"),
        };
        out.push_str(&format!(
            "    case {}u: {{ return vec2<f32>(1.0 / f32(textureDimensions(tex{index}).x), \
             1.0 / f32({height})); }}\n",
            texture.slot.key()
        ));
    }
    out.push_str("    default: { return vec2<f32>(1.0, 1.0); }\n  }\n}\n\n");

    // One call per gather channel, through the descriptor's swizzle.
    out.push_str(
        "fn texGather(imm: u32, component: u32, u: f32, v: f32, w: f32, layer: u32) -> vec4<f32> {\n  \
         switch (imm) {\n",
    );
    for (index, texture) in layout.textures.iter().enumerate() {
        let smp = texture.sampler.unwrap_or(index as u32);
        let coords = match (texture.dim, texture.compare) {
            (TexDim::T2d, false) => "vec2<f32>(u, v)",
            (TexDim::T2dArray, false) => "vec2<f32>(u, v), layer",
            (TexDim::TCube, false) => "vec3<f32>(u, v, w)",
            (TexDim::TCubeArray, false) => "vec3<f32>(u, v, w), layer",
            _ => continue,
        };
        let key = texture.slot.key();
        out.push_str(&format!(
            "    case {key}u: {{\n      switch (component) {{\n"
        ));
        if texture.uint {
            // The raw words, whatever the descriptor says.
            for channel in 0..4 {
                out.push_str(&format!(
                    "        case {channel}u: {{ return bitcast<vec4<f32>>(textureGather({channel}, \
                     tex{index}, smp{smp}, {coords})); }}\n"
                ));
            }
            out.push_str("        default: { return vec4<f32>(0.0); }\n      }\n    }\n");
            continue;
        }
        let (prelude, coords) = match texture.fold {
            Some(fold) => (folded_coords(fold, 0), "vec2<f32>(u, fv), fk"),
            None => (String::new(), coords),
        };
        for (channel, source) in texture.swizzle.iter().enumerate() {
            let texels = match source {
                SwizzleSource::R => format!("textureGather(0, tex{index}, smp{smp}, {coords})"),
                SwizzleSource::G => format!("textureGather(1, tex{index}, smp{smp}, {coords})"),
                SwizzleSource::B => format!("textureGather(2, tex{index}, smp{smp}, {coords})"),
                SwizzleSource::A => format!("textureGather(3, tex{index}, smp{smp}, {coords})"),
                SwizzleSource::Zero => "vec4<f32>(0.0)".to_string(),
                SwizzleSource::One => "vec4<f32>(1.0)".to_string(),
            };
            out.push_str(&format!(
                "        case {channel}u: {{ {prelude} return {texels}; }}\n"
            ));
        }
        out.push_str("        default: { return vec4<f32>(0.0); }\n      }\n    }\n");
    }
    out.push_str("    default: { return vec4<f32>(0.0); }\n  }\n}\n\n");

    // `tlds`: the texel at integer coordinates, zero outside the image.
    out.push_str(
        "fn texFetch(imm: u32, x: u32, y: u32, z: u32) -> vec4<f32> {\n  switch (imm) {\n",
    );
    for (index, texture) in layout.textures.iter().enumerate() {
        if texture.compare || !matches!(texture.dim, TexDim::T2d | TexDim::T2dArray | TexDim::T3d) {
            continue;
        }
        let key = texture.slot.key();
        let (size, load) = match (texture.fold, texture.dim) {
            (Some(fold), _) => {
                let stride = fold.tile_rows - 1;
                let last = fold.tiles() - 1;
                (
                    format!(
                        "vec3<u32>(textureDimensions(tex{index}).x, {}u, 1u)",
                        fold.height
                    ),
                    format!(
                        "let fk = min(y / {stride}u, {last}u); let texel = \
                         textureLoad(tex{index}, vec2<u32>(x, y - fk * {stride}u), fk, 0);"
                    ),
                )
            }
            (None, TexDim::T2dArray) => (
                format!("vec3<u32>(textureDimensions(tex{index}), textureNumLayers(tex{index}))"),
                format!("let texel = textureLoad(tex{index}, vec2<u32>(x, y), z, 0);"),
            ),
            (None, TexDim::T3d) => (
                format!("textureDimensions(tex{index})"),
                format!("let texel = textureLoad(tex{index}, vec3<u32>(x, y, z), 0);"),
            ),
            (None, _) => (
                format!("vec3<u32>(textureDimensions(tex{index}), 1u)"),
                format!("let texel = textureLoad(tex{index}, vec2<u32>(x, y), 0);"),
            ),
        };
        let result = if texture.uint {
            format!("{load} return bitcast<vec4<f32>>(texel);")
        } else {
            format!("{load} {}", return_swizzled("texel", texture.swizzle))
        };
        out.push_str(&format!(
            "    case {key}u: {{\n      let size = {size};\n      \
             if (x >= size.x || y >= size.y || z >= size.z) {{ return vec4<f32>(0.0); }}\n      {result}\n    }}\n"
        ));
    }
    out.push_str("    default: { return vec4<f32>(0.0); }\n  }\n}\n\n");

    // A shadow gather: each texel's comparison, unswizzled like `texSampleCompare`.
    out.push_str(
        "fn texGatherCompare(imm: u32, u: f32, v: f32, w: f32, layer: u32, dref: f32) -> vec4<f32> {\n  \
         switch (imm) {\n",
    );
    for (index, texture) in layout.textures.iter().enumerate() {
        let smp = texture.sampler.unwrap_or(index as u32);
        let coords = match (texture.dim, texture.compare) {
            (TexDim::T2d, true) => "vec2<f32>(u, v)",
            (TexDim::T2dArray, true) => "vec2<f32>(u, v), layer",
            (TexDim::TCube, true) => "vec3<f32>(u, v, w)",
            (TexDim::TCubeArray, true) => "vec3<f32>(u, v, w), layer",
            _ => continue,
        };
        let key = texture.slot.key();
        out.push_str(&format!(
            "    case {key}u: {{ return textureGatherCompare(tex{index}, smp{smp}, {coords}, dref); }}\n"
        ));
    }
    out.push_str("    default: { return vec4<f32>(0.0); }\n  }\n}\n\n");

    out.push_str(&special_registers(stage, layout));

    out.push_str(&translated.source);
    out.push('\n');
    out.push_str(&match stage {
        Stage::Vertex => vertex_entry(layout),
        Stage::Fragment => fragment_entry(translated, layout),
        Stage::Compute => compute_entry(layout, translated.warp.is_some()),
    });
    Ok(out)
}

/// Coordinates into a [`Crop`]'s image, moved by `offset` texels, as coordinates into the texture
/// it is the corner of, clamped to the image's edge texel centres.
const CROPPED: &str = "\
fn cropped(u: f32, v: f32, size: vec2<f32>, full: vec2<f32>, offset: vec2<f32>) -> vec2<f32> {
  return clamp(vec2<f32>(u, v) * size + offset, vec2<f32>(0.5), size - vec2<f32>(0.5)) / full;
}

";

/// The `cropped` call for a binding's coordinates `u`, `v` and a constant texel offset.
fn cropped(crop: Crop, (x, y): (i32, i32)) -> String {
    format!(
        "cropped(u, v, vec2<f32>({}.0, {}.0), vec2<f32>({}.0, {}.0), vec2<f32>({x}.0, {y}.0))",
        crop.width, crop.height, crop.full_width, crop.full_height
    )
}

/// `SR_TID` and `SR_NTID` read whole, refused as the interpreter refuses them.
const PACKED_SPECIAL_REGISTERS: [u8; 2] = [0x20, 0x28];

/// `sreg`, what `s2r` reads. Only a compute invocation has an identity; the graphics stages
/// read zero, as they always have here.
fn special_registers(stage: Stage, layout: &Layout) -> String {
    let Some(shape) = layout.compute.filter(|_| stage == Stage::Compute) else {
        return "fn sreg(sr: u32) -> u32 { return 0u; }\n\n".to_string();
    };
    format!(
        "var<private> sr_lane: u32;\n\
         var<private> sr_tid: vec3<u32>;\n\
         var<private> sr_ctaid: vec3<u32>;\n\n\
         fn sreg(sr: u32) -> u32 {{\n  switch (sr) {{\n\
         \x20   case 0x00u: {{ return sr_lane; }}\n\
         \x20   case 0x21u: {{ return sr_tid.x; }}\n\
         \x20   case 0x22u: {{ return sr_tid.y; }}\n\
         \x20   case 0x23u: {{ return sr_tid.z; }}\n\
         \x20   case 0x25u: {{ return sr_ctaid.x; }}\n\
         \x20   case 0x26u: {{ return sr_ctaid.y; }}\n\
         \x20   case 0x27u: {{ return sr_ctaid.z; }}\n\
         \x20   case 0x32u: {{ return {}u; }}\n\
         \x20   case 0x36u: {{ return {}u; }}\n\
         \x20   default: {{ return 0u; }}\n  }}\n}}\n\n",
        shape.shared_size, shape.local_size
    )
}

/// One invocation per thread of a CTA, a workgroup per CTA. A program with warp instructions
/// takes its lane from the subgroup they run on: every invocation is active on entry, so the
/// count before it is its index there, whatever the workgroup's shape.
fn compute_entry(layout: &Layout, warp: bool) -> String {
    let [x, y, z] = layout
        .compute
        .expect("a compute module is built for a shape")
        .block;
    let lane = if warp {
        "subgroupExclusiveAdd(1u)"
    } else {
        "index % 32u"
    };
    format!(
        "@compute @workgroup_size({x}, {y}, {z})\n\
         fn cs_main(\n\
         \x20 @builtin(local_invocation_id) local: vec3<u32>,\n\
         \x20 @builtin(local_invocation_index) index: u32,\n\
         \x20 @builtin(workgroup_id) group: vec3<u32>,\n\
         ) {{\n\
         \x20 sr_lane = {lane};\n\
         \x20 sr_tid = local;\n\
         \x20 sr_ctaid = group;\n\
         \x20 _ = run();\n}}\n"
    )
}

/// Statements binding `texel`, the raw words of the integer image's texel under `(u, v)` moved
/// by `offset` texels: no filtering, the nearest texel, clamped to the edge.
fn uint_load(index: usize, dim: TexDim, offset: (i32, i32)) -> Result<String, Unsupported> {
    let layer = match dim {
        TexDim::T2d => "",
        TexDim::T2dArray => ", i32(layer)",
        dim => return Err(Unsupported::TextureDimension { dim }),
    };
    // Moved as `interp` moves it, by whole texels of normalized coordinates.
    let uv = match offset {
        (0, 0) => "vec2<f32>(u, v)".to_string(),
        (x, y) => format!(
            "vec2<f32>(u + f32({x}) * (1.0 / f32(size.x)), v + f32({y}) * (1.0 / f32(size.y)))"
        ),
    };
    Ok(format!(
        "let size = vec2<i32>(textureDimensions(tex{index})); \
         let at = clamp(vec2<i32>(floor({uv} * vec2<f32>(size))), vec2<i32>(0), size - vec2<i32>(1)); \
         let texel = textureLoad(tex{index}, at{layer}, 0);"
    ))
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

/// Statements binding `fk`, the layer a folded image is sampled in, and `fv`, the
/// normalized `v` within it, for `v` moved `dy` texels. See [`Fold`].
fn folded_coords(fold: Fold, dy: i32) -> String {
    let (height, stride, rows) = (fold.height, fold.tile_rows - 1, fold.tile_rows);
    let last = fold.tiles() - 1;
    if !fold.wrap {
        return format!(
            "let fy = v * {height}.0 - 0.5 + f32({dy}); \
             let fk = min(u32(max(floor(fy), 0.0)) / {stride}u, {last}u); \
             let fv = (fy - f32(fk * {stride}u) + 0.5) / {rows}.0;"
        );
    }
    // The row wraps into the image first; only the last row pairs with the first.
    format!(
        "let fw = v * {height}.0 - 0.5 + f32({dy}); \
         let fy = fw - {height}.0 * floor(fw / {height}.0); \
         let fr = min(u32(floor(fy)), {}u); \
         let fwrap = fr == {}u; \
         let fk = select(min(fr / {stride}u, {last}u), {}u, fwrap); \
         let fv = (fy - f32(select(fk * {stride}u, {}u, fwrap)) + 0.5) / {rows}.0;",
        height - 1,
        height - 1,
        fold.tiles(),
        height - 1
    )
}

/// The WGSL type of a folded image's binding: only plain 2D images fold.
fn folded_texture_type(dim: TexDim, compare: bool) -> Result<&'static str, Unsupported> {
    match (dim, compare) {
        (TexDim::T2d | TexDim::T2dArray, false) => Ok("texture_2d_array<f32>"),
        (dim, _) => Err(Unsupported::TextureDimension { dim }),
    }
}

/// The WGSL type of an integer image's binding: only plain 2D ones are read.
fn uint_texture_type(dim: TexDim, compare: bool) -> Result<&'static str, Unsupported> {
    match (dim, compare) {
        (TexDim::T2d, false) => Ok("texture_2d<u32>"),
        (TexDim::T2dArray, false) => Ok("texture_2d_array<u32>"),
        (dim, _) => Err(Unsupported::TextureDimension { dim }),
    }
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
        (TexDim::TCube, true) => Ok("texture_depth_cube"),
        (TexDim::TCubeArray, true) => Ok("texture_depth_cube_array"),
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
    for (location, slot) in layout.varyings.iter().enumerate() {
        out.push_str(&format!(
            "  @location({location}) @interpolate(linear{}) vary{slot}: vec4<f32>,\n",
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
        // Integer attributes are carried as bits, scaled ones as their value.
        let base = layout.attribute_base(*slot);
        // BGRA swaps the first and third components.
        let axes = if layout.bgra_attributes.contains(slot) {
            ["z", "y", "x", "w"]
        } else {
            ["x", "y", "z", "w"]
        };
        for (component, axis) in axes.iter().enumerate() {
            let read = format!("input.attr{slot}.{axis}");
            let read = match base {
                AttributeBase::Float => read,
                AttributeBase::Sint | AttributeBase::Uint => format!("bitcast<f32>({read})"),
                AttributeBase::Sscaled | AttributeBase::Uscaled => format!("f32({read})"),
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
    if let Some((width, height)) = layout.window_coordinates.filter(|&(w, h)| w > 0 && h > 0) {
        out.push_str(&format!(
            "  out.position.x = out.position.x * {:?} - out.position.w;\n",
            2.0 / width as f32
        ));
        out.push_str(&format!(
            "  out.position.y = out.position.y * {:?} - out.position.w;\n",
            2.0 / height as f32
        ));
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
        let max = layout
            .uint_targets
            .iter()
            .find(|(t, _)| *t == target)
            .map(|&(_, max)| max);
        let uint = max.is_some();
        let components: Vec<String> = (0..4)
            .map(|c| {
                let reg = (target * 4 + c) as u8;
                match (translated.registers.contains(&reg), max) {
                    (true, Some(u32::MAX)) => format!("r{reg}"),
                    (true, Some(max)) => format!("min(r{reg}, {max:#x}u)"),
                    (true, None) => format!("bitcast<f32>(r{reg})"),
                    // An unwritten channel is zero.
                    (false, Some(_)) => "0u".to_string(),
                    (false, None) => "0.0".to_string(),
                }
            })
            .collect();
        let vec = if uint { "vec4<u32>" } else { "vec4<f32>" };
        format!("{vec}({})", components.join(", "))
    };
    let output_type = |target: u32| {
        if layout.uint_targets.iter().any(|(t, _)| *t == target) {
            "vec4<u32>"
        } else {
            "vec4<f32>"
        }
    };

    // Applied after alpha-to-coverage, which reads the shader's own alpha.
    let stored = |target: u32, value: &str| -> String {
        if layout.unorm16_targets.contains(&target) {
            format!(
                "floor(clamp({value}, vec4<f32>(0.0), vec4<f32>(1.0)) * 65535.0 + 0.5) / 65535.0"
            )
        } else {
            value.to_string()
        }
    };

    let mut out = String::new();
    if layout.coverage.as_ref().is_some_and(|c| c.per_sample_pass) {
        out.push_str("override pass_sample: u32 = 0u;\n\n");
    }
    out.push_str("struct FragmentInput {\n  @builtin(position) position: vec4<f32>,\n");
    for (location, slot) in layout.varyings.iter().enumerate() {
        out.push_str(&format!(
            "  @location({location}) @interpolate(linear{}) vary{slot}: vec4<f32>,\n",
            layout.sampling(*slot)
        ));
    }
    out.push_str("}\n\n");
    let targets = layout.targets;
    if targets > 1 {
        out.push_str("struct FragmentOutput {\n");
        for target in 0..targets {
            out.push_str(&format!(
                "  @location({target}) target{target}: {},\n",
                output_type(target)
            ));
        }
        out.push_str("}\n\n");
    }
    let returns = match targets {
        0 => String::new(),
        1 => format!(" -> @location(0) {}", output_type(0)),
        _ => " -> FragmentOutput".to_string(),
    };
    out.push_str(&format!(
        "@fragment\nfn fs_main(input: FragmentInput){returns} {{\n"
    ));
    // Coverage first: the sample mask applies before shading.
    if let Some(coverage) = &layout.coverage {
        out.push_str(&sample_index(coverage));
        if coverage.per_sample_pass {
            out.push_str("  if (sample != pass_sample) { discard; }\n");
        }
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
        for &target in layout.unorm16_targets.iter().filter(|&&t| t < targets) {
            let value = format!("out.target{target}");
            out.push_str(&format!("  {value} = {};\n", stored(target, &value)));
        }
        out.push_str("  return out;\n");
    } else if let Some(coverage) = alpha_to_coverage {
        // Shaded even depth-only, for alpha-to-coverage.
        out.push_str(&format!("  let target0 = {};\n", colour(0)));
        out.push_str(&alpha_coverage(coverage, "target0.w"));
        if targets == 1 {
            out.push_str(&format!("  return {};\n", stored(0, "target0")));
        }
    } else if targets == 1 {
        out.push_str(&format!("  return {};\n", stored(0, &colour(0))));
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
