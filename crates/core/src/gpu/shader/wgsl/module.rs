//! Wrapping a translation into a complete module with its entry points.

use super::layout::{
    attribute_scalar, unpack_1010102, Coverage, Crop, Fold, Global, Layout, Stage, StorageNumeric,
    TextureBinding, Update, WordImage, WordPack, ATOMS, ATTRIBUTE_WORDS, CONSTANTS_BINDING,
    CONSTANT_BANKS, FRONT_FACING, GENERIC_BASE, GENERIC_STRIDE, GLOBAL_BINDING, IDENTITY_SWIZZLE,
    INSTANCE_ID, POSITION, TEXTURE_BINDING, VERTEX_ID,
};
use super::{Translation, Unsupported};
use crate::gpu::pipeline::AttributeBase;
use crate::gpu::shader::isa::TexDim;
use crate::gpu::texture::{SwizzleSource, TextureSlot};

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
    check_stage(translated, stage, layout)?;
    directives(&mut out, translated, stage);
    let (stores, reductions) = bindings(&mut out, layout)?;
    image_functions(&mut out, layout, &stores, &reductions);
    global_functions(&mut out, layout);
    attribute_and_constant_functions(&mut out, layout);
    tex_sample(&mut out, layout)?;
    tex_sample_compare(&mut out, layout)?;
    tex_sample_offset(&mut out, layout)?;
    tex_dims(&mut out, layout);
    tex_step(&mut out, layout);
    tex_gather(&mut out, layout);
    tex_fetch_level(&mut out, layout);
    tex_fetch(&mut out, layout);
    tex_gather_compare(&mut out, layout);

    out.push_str(&special_registers(
        stage,
        layout,
        in_quad(translated, stage),
    ));

    out.push_str(&translated.source);
    out.push('\n');
    out.push_str(&match stage {
        Stage::Vertex => vertex_entry(layout),
        Stage::Fragment => fragment_entry(translated, layout),
        Stage::Compute => compute_entry(layout, translated.warp.is_some()),
    });
    Ok(out)
}

/// What a stage cannot run, refused before anything is written.
fn check_stage(translated: &Translation, stage: Stage, layout: &Layout) -> Result<(), Unsupported> {
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
    Ok(())
}

/// Directives first, then the lane index.
fn directives(out: &mut String, translated: &Translation, stage: Stage) {
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
    if in_quad(translated, stage) {
        out.push_str("var<private> quad_lane: u32;\n");
        out.push_str("fn quadLane() -> u32 { return quad_lane; }\n\n");
    }
    if translated.quad_swap.is_some() && !translated.subgroups {
        out.push_str(QUAD_SWAP);
        out.push_str("\n\n");
    }
}

/// Every binding's declaration, returning the `imgStore` and `imgReduce` cases their images need.
fn bindings(out: &mut String, layout: &Layout) -> Result<(String, String), Unsupported> {
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
    let cases = image_bindings(out, layout)?;
    if !layout.const_banks.is_empty()
        || !layout.textures.is_empty()
        || !layout.globals.is_empty()
        || !layout.images.is_empty()
    {
        out.push('\n');
    }
    Ok(cases)
}

/// Each layer of an image is its own binding, so a store picks one by its layer; a layer
/// past the image's is out of range.
fn image_bindings(out: &mut String, layout: &Layout) -> Result<(String, String), Unsupported> {
    let mut stores = String::new();
    let mut reductions = String::new();
    for (index, image) in layout.images.iter().enumerate() {
        if let Some(words) = image.words {
            word_image_binding(out, layout, index, words, &mut stores, &mut reductions);
            continue;
        }
        if image.reduced {
            return Err(Unsupported::ImageFormat);
        }
        let format = image.format.ok_or(Unsupported::ImageFormat)?;
        let texel = match format.numeric {
            StorageNumeric::Float => "bitcast<vec4<f32>>(value)",
            StorageNumeric::Uint => "value",
            StorageNumeric::Sint => "bitcast<vec4<i32>>(value)",
        };
        let mut layers = String::new();
        for layer in 0..image.layers {
            out.push_str(&format!(
                "@group({}) @binding({}) var img{index}_{layer}: texture_storage_2d<{}, write>;\n",
                layout.group,
                layout.image_binding(index, layer),
                format.name
            ));
            layers.push_str(&format!(
                "        case {layer}u: {{ if (all(vec2<u32>(x, y) < textureDimensions(img{index}_{layer}))) \
                 {{ textureStore(img{index}_{layer}, vec2<u32>(x, y), {texel}); }} }}\n"
            ));
        }
        stores.push_str(&format!(
            "    case {}u: {{\n      switch (layer) {{\n{layers}        default: {{ }}\n      }}\n    }}\n",
            image.slot.key()
        ));
    }
    Ok((stores, reductions))
}

/// An image bound as a buffer of words: one array per layer, with its store and reduction cases.
fn word_image_binding(
    out: &mut String,
    layout: &Layout,
    index: usize,
    words: WordImage,
    stores: &mut String,
    reductions: &mut String,
) {
    let image = &layout.images[index];
    let WordImage {
        width,
        height,
        row_words,
        pack,
    } = words;
    let word = match pack {
        WordPack::Word => "value.x",
        WordPack::Rg11b10 => "packRg11b10(bitcast<vec4<f32>>(value))",
    };
    let (mut layers, mut reduced) = (String::new(), String::new());
    for layer in 0..image.layers {
        let array = format!("img{index}_{layer}");
        out.push_str(&format!(
            "@group({}) @binding({}) var<storage, read_write> {array}: array<atomic<u32>>;\n",
            layout.group,
            layout.image_binding(index, layer),
        ));
        let at = format!("y * {row_words}u + x");
        layers.push_str(&format!(
            "        case {layer}u: {{ if (x < {width}u && y < {height}u) {{ \
             atomicStore(&{array}[{at}], {word}); }} }}\n"
        ));
        reduced.push_str(&format!(
            "        case {layer}u: {{\n          if (x < {width}u && y < {height}u) {{\n            \
             let word = {at};\n            switch (op) {{\n"
        ));
        for (code, (_, _, update)) in ATOMS.iter().enumerate() {
            match update {
                Update::Direct(function) => reduced.push_str(&format!(
                    "              case {code}u: {{ {function}(&{array}[word], value); }}\n"
                )),
                Update::Loop(next) => reduced.push_str(&format!(
                    "              case {code}u: {{\n{}              }}\n",
                    compare_exchange(&array, "word", next, "                ")
                )),
            }
        }
        reduced.push_str("              default: { }\n            }\n          }\n        }\n");
    }
    let key = image.slot.key();
    stores.push_str(&format!(
        "    case {key}u: {{\n      switch (layer) {{\n{layers}        default: {{ }}\n      }}\n    }}\n"
    ));
    reductions.push_str(&format!(
        "    case {key}u: {{\n      switch (layer) {{\n{reduced}        default: {{ }}\n      }}\n    }}\n"
    ));
}

/// `imgReduce` and `imgStore`, with the packing a word image needs.
fn image_functions(out: &mut String, layout: &Layout, stores: &str, reductions: &str) {
    if layout
        .images
        .iter()
        .any(|image| image.words.is_some_and(|w| w.pack == WordPack::Rg11b10))
    {
        out.push_str(PACK_RG11B10);
    }
    // `sured`: no value comes back, and a compare-and-swap's `stored` is never asked for.
    out.push_str(&format!(
        "fn imgReduce(op: u32, key: u32, x: u32, y: u32, layer: u32, value: u32) {{\n  \
         let stored = 0u;\n  switch (key) {{\n{reductions}    default: {{ }}\n  }}\n}}\n\n"
    ));
    out.push_str(&format!(
        "fn imgStore(key: u32, x: u32, y: u32, layer: u32, value: vec4<u32>) {{\n  switch (key) {{\n{stores}    \
         default: {{ }}\n  }}\n}}\n\n"
    ));
}

/// `gRead`, and for written buffers `gWrite` and `gAtom`.
fn global_functions(out: &mut String, layout: &Layout) {
    if !layout.globals.is_empty() {
        out.push_str("fn gRead(slot: u32, offset: u32) -> u32 {\n  let word = offset >> 2u;\n  switch (slot) {\n");
        for (slot, global) in layout.globals.iter().enumerate() {
            let read = if global.written {
                format!("atomicLoad(&g{slot}[word])")
            } else {
                format!("g{slot}[word]")
            };
            out.push_str(&format!(
                "    case {slot}u: {{\n      {}if (word >= arrayLength(&g{slot})) {{ return 0u; }}\n      return {read};\n    }}\n",
                rebase(global)
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
        for (slot, global) in layout.globals.iter().enumerate().filter(|(_, g)| g.written) {
            let array = format!("g{slot}");
            let low = merge(&array, "word", "low_mask", "low");
            let high = merge(&array, "word + 1u", "high_mask", "high");
            out.push_str(&format!(
                "    case {slot}u: {{\n      {}if (word < arrayLength(&{array})) {{\n{low}      }}\n      \
                 if (high_mask != 0u && word + 1u < arrayLength(&{array})) {{\n{high}      }}\n    }}\n",
                rebase(global)
            ));
        }
        out.push_str("    default: { }\n  }\n}\n\n");

        out.push_str(
            "fn gAtom(op: u32, slot: u32, offset: u32, value: u32, stored: u32) -> u32 {\n  let word = offset >> 2u;\n  switch (slot) {\n",
        );
        for (slot, global) in layout.globals.iter().enumerate().filter(|(_, g)| g.written) {
            out.push_str(&format!(
                "    case {slot}u: {{\n      {}if (word < arrayLength(&g{slot})) {{\n        switch (op) {{\n",
                rebase(global)
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
}

/// Attribute space, `cbRead`, and `cropped` when a binding is cropped.
fn attribute_and_constant_functions(out: &mut String, layout: &Layout) {
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

    // An unbound bank, or a read past the end of one, is zero; a branch here doubled the
    // driver's compile of large vertex shaders (docs/gpu-backend.md).
    if layout.const_banks.is_empty() {
        out.push_str("fn cbRead(bank: u32, offset: u32) -> u32 { return 0u; }\n\n");
    } else {
        let last = CONSTANT_BANKS - 1;
        out.push_str(&format!(
            "fn cbRead(bank: u32, offset: u32) -> u32 {{\n  \
             let word = offset >> 2u;\n  \
             let slot = min(bank, {last}u) * 2u;\n  \
             let inside = (bank < {CONSTANT_BANKS}u) & (word < cbs[slot + 1u]);\n  \
             return select(0u, cbs[cbs[slot] + word], inside);\n}}\n\n"
        ));
    }

    if layout.textures.iter().any(|t| t.crop.is_some()) {
        out.push_str(CROPPED);
    }
}

/// `dim` is unused here but part of `HOST_INTERFACE`.
fn tex_sample(out: &mut String, layout: &Layout) -> Result<(), Unsupported> {
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
                "    case {key}u: {{ {load} return {}; }}\n",
                uint_swizzled(texture.swizzle)
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
    Ok(())
}

/// Shadow samples: alpha one, the comparison in the rest, as Eden's `Extract`.
fn tex_sample_compare(out: &mut String, layout: &Layout) -> Result<(), Unsupported> {
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
    Ok(())
}

/// One case per binding and offset: offsets must be constants.
fn tex_sample_offset(out: &mut String, layout: &Layout) -> Result<(), Unsupported> {
    out.push_str(
        "fn texSampleOffset(imm: u32, offset: u32, u: f32, v: f32, layer: u32, w: f32) \
         -> vec4<f32> {\n  \
         switch (imm) {\n",
    );
    for (index, texture) in layout.textures.iter().enumerate() {
        let smp = texture.sampler.unwrap_or(index as u32);
        let coords = match (texture.dim, texture.compare) {
            (TexDim::T2d, false) => "vec2<f32>(u, v), 0.0",
            (TexDim::T2dArray, false) => "vec2<f32>(u, v), layer, 0.0",
            (TexDim::T3d, false) => "vec3<f32>(u, v, w), 0.0",
            _ => continue,
        };
        let key = texture.slot.key();
        out.push_str(&format!("    case {key}u: {{\n      switch (offset) {{\n"));
        for (n, &(x, y, z)) in layout.texture_offsets.iter().enumerate() {
            if texture.uint {
                let load = uint_load(index, texture.dim, (x, y))?;
                out.push_str(&format!(
                    "        case {n}u: {{ {load} return {}; }}\n",
                    uint_swizzled(texture.swizzle)
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
                // Constant offsets count the device's texels, which are larger than the guest's.
                (None, None) if texture.shrink > 0 => {
                    let moved = format!("vec2<f32>(u + f32({x}) / d.x, v + f32({y}) / d.y)");
                    let coords = coords.replacen("vec2<f32>(u, v)", &moved, 1);
                    format!(
                        "let d = vec2<f32>({}); {}",
                        guest_size(index, texture),
                        return_swizzled(
                            &format!("textureSampleLevel(tex{index}, smp{smp}, {coords})"),
                            texture.swizzle
                        )
                    )
                }
                (None, None) => {
                    let texels = match texture.dim {
                        TexDim::T3d => format!("vec3<i32>({x}, {y}, {z})"),
                        _ => format!("vec2<i32>({x}, {y})"),
                    };
                    return_swizzled(
                        &format!("textureSampleLevel(tex{index}, smp{smp}, {coords}, {texels})"),
                        texture.swizzle,
                    )
                }
            };
            out.push_str(&format!("        case {n}u: {{ {body} }}\n"));
        }
        out.push_str("        default: { return vec4<f32>(0.0); }\n      }\n    }\n");
    }
    out.push_str("    default: { return vec4<f32>(0.0); }\n  }\n}\n\n");
    Ok(())
}

/// `txq` sizes, as `interp`'s `run_txq` answers.
fn tex_dims(out: &mut String, layout: &Layout) {
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
            "    case {key}u: {{\n      let s = {};\n      \
             return vec4<u32>(select(max(s.x >> lod, 1u), 1u, lod >= 32u), \
             select(max({height} >> lod, 1u), 1u, lod >= 32u), {depth}, \
             textureNumLevels(tex{index}));\n    }}\n",
            guest_size(index, texture)
        ));
    }
    out.push_str("    default: { return vec4<u32>(1u, 1u, 1u, 1u); }\n  }\n}\n\n");
}

/// One texel in normalized coordinates, as `TextureSource::texel_step` answers.
fn tex_step(out: &mut String, layout: &Layout) {
    out.push_str("fn texStep(imm: u32) -> vec2<f32> {\n  switch (imm) {\n");
    for (index, texture) in layout.textures.iter().enumerate() {
        let height = match texture.fold {
            Some(fold) => format!("{}u", fold.height),
            None => format!("{}.y", guest_size(index, texture)),
        };
        out.push_str(&format!(
            "    case {}u: {{ return vec2<f32>(1.0 / f32({}.x), 1.0 / f32({height})); }}\n",
            texture.slot.key(),
            guest_size(index, texture)
        ));
    }
    out.push_str("    default: { return vec2<f32>(1.0, 1.0); }\n  }\n}\n\n");
}

/// One call per gather channel, through the descriptor's swizzle.
fn tex_gather(out: &mut String, layout: &Layout) {
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
            // The raw words through the swizzle, a one being the integer.
            for (channel, source) in texture.swizzle.iter().enumerate() {
                let texels = match source {
                    SwizzleSource::Zero => "vec4<u32>(0u)".to_string(),
                    SwizzleSource::One => "vec4<u32>(1u)".to_string(),
                    picked => format!(
                        "textureGather({}, tex{index}, smp{smp}, {coords})",
                        swizzle_channel(*picked)
                    ),
                };
                out.push_str(&format!(
                    "        case {channel}u: {{ return bitcast<vec4<f32>>({texels}); }}\n"
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
}

/// A `tlds` naming its level: each level binds as a texture of its own; past the levels
/// bound, zero.
fn tex_fetch_level(out: &mut String, layout: &Layout) {
    out.push_str(
        "fn texFetchLevel(imm: u32, level: u32, x: u32, y: u32, z: u32) -> vec4<f32> {\n  \
         switch (imm) {\n",
    );
    for &(base, levels) in &layout.fetch_levels {
        let TextureSlot::Bound(immediate) = base else {
            continue;
        };
        out.push_str(&format!(
            "    case {}u: {{\n      switch (level) {{\n",
            base.key()
        ));
        out.push_str(&format!(
            "        case 0u: {{ return texFetch({}u, x, y, z); }}\n",
            base.key()
        ));
        for level in 1..levels {
            let slot = TextureSlot::Level {
                immediate,
                level: level as u8,
            };
            out.push_str(&format!(
                "        case {level}u: {{ return texFetch({}u, x, y, z); }}\n",
                slot.key()
            ));
        }
        out.push_str("        default: { return vec4<f32>(0.0); }\n      }\n    }\n");
    }
    out.push_str("    default: { return vec4<f32>(0.0); }\n  }\n}\n\n");
}

/// `tlds`: the texel at integer coordinates, zero outside the image.
fn tex_fetch(out: &mut String, layout: &Layout) {
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
            format!("{load} return {};", uint_swizzled(texture.swizzle))
        } else {
            format!("{load} {}", return_swizzled("texel", texture.swizzle))
        };
        // A smaller level holds each texel of the guest's in one of its own.
        let shrink = match texture.shrink {
            0 => String::new(),
            levels => format!("let x = x >> {levels}u; let y = y >> {levels}u;\n      "),
        };
        out.push_str(&format!(
            "    case {key}u: {{\n      {shrink}let size = {size};\n      \
             if (x >= size.x || y >= size.y || z >= size.z) {{ return vec4<f32>(0.0); }}\n      {result}\n    }}\n"
        ));
    }
    out.push_str("    default: { return vec4<f32>(0.0); }\n  }\n}\n\n");
}

/// A shadow gather: each texel's comparison, unswizzled like `texSampleCompare`.
fn tex_gather_compare(out: &mut String, layout: &Layout) {
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

/// Whether a fragment knows its place in its quad: it exchanges values with its neighbours, or
/// reads its lane as `raster` shades it, in a quad.
fn in_quad(translated: &Translation, stage: Stage) -> bool {
    translated.quad.is_some()
        || (stage == Stage::Fragment
            && translated
                .special
                .iter()
                .any(|&sr| crate::gpu::shader::reads_lane(sr)))
}

/// `SR_TID` and `SR_NTID` read whole, refused as the interpreter refuses them.
const PACKED_SPECIAL_REGISTERS: [u8; 2] = [0x20, 0x28];

/// `sreg`, what `s2r` reads. A compute invocation has an identity; in the graphics stages only a
/// fragment has a lane, its place in its quad, as its shuffles and votes have.
fn special_registers(stage: Stage, layout: &Layout, quad: bool) -> String {
    let Some(shape) = layout.compute.filter(|_| stage == Stage::Compute) else {
        let lane = if stage == Stage::Fragment && quad {
            "quadLane()"
        } else {
            "0u"
        };
        return format!(
            "fn sreg(sr: u32) -> u32 {{\n  let lane = {lane};\n  switch (sr) {{\n\
             \x20   case 0x00u: {{ return lane; }}\n\
             {LANE_MASKS}\
             \x20   default: {{ return 0u; }}\n  }}\n}}\n\n"
        );
    };
    format!(
        "var<private> sr_lane: u32;\n\
         var<private> sr_tid: vec3<u32>;\n\
         var<private> sr_ctaid: vec3<u32>;\n\n\
         fn sreg(sr: u32) -> u32 {{\n  let lane = sr_lane;\n  switch (sr) {{\n\
         \x20   case 0x00u: {{ return lane; }}\n\
         \x20   case 0x21u: {{ return sr_tid.x; }}\n\
         \x20   case 0x22u: {{ return sr_tid.y; }}\n\
         \x20   case 0x23u: {{ return sr_tid.z; }}\n\
         \x20   case 0x25u: {{ return sr_ctaid.x; }}\n\
         \x20   case 0x26u: {{ return sr_ctaid.y; }}\n\
         \x20   case 0x27u: {{ return sr_ctaid.z; }}\n\
         \x20   case 0x32u: {{ return {}u; }}\n\
         \x20   case 0x36u: {{ return {}u; }}\n\
         {LANE_MASKS}\
         \x20   default: {{ return 0u; }}\n  }}\n}}\n\n",
        shape.shared_size, shape.local_size
    )
}

/// `SR_EQMASK` through `SR_GEMASK` for `lane`, as `interp::SpecialRegs::read` gives them. The
/// at-or-below mask is built from two shifts, so lane 31 wraps to all ones.
const LANE_MASKS: &str = "\
\x20   case 0x38u: { return 1u << lane; }
    case 0x39u: { return (1u << lane) - 1u; }
    case 0x3Au: { return ((1u << lane) << 1u) - 1u; }
    case 0x3Bu: { return ~(((1u << lane) << 1u) - 1u); }
    case 0x3Cu: { return ~((1u << lane) - 1u); }
";

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

/// A global's word in the buffer bound for it, which starts `base` words before its address.
fn rebase(global: &Global) -> String {
    match global.base {
        0 => String::new(),
        base => format!("let word = word + {base}u;\n      "),
    }
}

/// The guest's size of a 2D image or layer, which a smaller mip level stands in for.
fn guest_size(index: usize, texture: &TextureBinding) -> String {
    match texture.shrink {
        0 => format!("textureDimensions(tex{index})"),
        levels => format!("(textureDimensions(tex{index}) << vec2<u32>({levels}u))"),
    }
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
/// An integer texel `texel` through `swizzle` as the float bits a shader reads; a one is
/// the integer, as the rasterizer's is.
fn uint_swizzled(swizzle: [SwizzleSource; 4]) -> String {
    let channels: Vec<String> = swizzle
        .iter()
        .map(|source| match source {
            SwizzleSource::Zero => "0u".to_string(),
            SwizzleSource::One => "1u".to_string(),
            picked => format!("texel[{}]", swizzle_channel(*picked)),
        })
        .collect();
    format!("bitcast<vec4<f32>>(vec4<u32>({}))", channels.join(", "))
}

/// The channel index a picking swizzle source reads.
fn swizzle_channel(source: SwizzleSource) -> u32 {
    match source {
        SwizzleSource::G => 1,
        SwizzleSource::B => 2,
        SwizzleSource::A => 3,
        _ => 0,
    }
}

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
        // Mapped as `raster` maps it; a headerless program's colours are `r0` onward in fours.
        let header = translated.header.filter(|h| h.writes_any_color());
        let components: Vec<String> = (0..4)
            .map(|c| {
                let reg = match header {
                    Some(header) => header.fragment_output_reg(target, c),
                    None => Some((target * 4 + c) as u8),
                };
                match (reg, max) {
                    // A component the header leaves out: colour zero, alpha opaque.
                    (None, Some(_)) => format!("{}u", u32::from(c == 3)),
                    (None, None) => format!("{:?}", f32::from(u8::from(c == 3))),
                    (Some(reg), _) if !translated.registers.contains(&reg) => match max {
                        // A register the program never writes is zero.
                        Some(_) => "0u".to_string(),
                        None => "0.0".to_string(),
                    },
                    (Some(reg), Some(u32::MAX)) => format!("r{reg}"),
                    (Some(reg), Some(max)) => format!("min(r{reg}, {max:#x}u)"),
                    (Some(reg), None) => format!("bitcast<f32>(r{reg})"),
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
    out.push_str(
        "struct FragmentInput {\n  @builtin(position) position: vec4<f32>,\n  \
         @builtin(front_facing) front: bool,\n",
    );
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
    if in_quad(translated, Stage::Fragment) {
        out.push_str(
            "  quad_lane = (u32(input.position.y) & 1u) * 2u + (u32(input.position.x) & 1u);\n",
        );
    }
    // `a[0x70]` and `a[0x74]` are the pixel's centre in window coordinates; an expanded
    // multisample attachment holds a pixel as a tile of texels.
    let (samples_x, samples_y) = layout
        .coverage
        .as_ref()
        .map_or((1, 1), |c| (c.samples_x, c.samples_y));
    // `a[0x3fc]`: all ones on a front face, as the rasterizer seeds it.
    out.push_str(&format!(
        "  attr_in[{}u] = bitcast<f32>(select(0u, 0xffffffffu, input.front));\n",
        FRONT_FACING / 4
    ));
    out.push_str(&format!(
        "  attr_in[{}u] = floor(input.position.x / {samples_x}.0) + 0.5;\n",
        POSITION / 4
    ));
    let y = format!("(floor(input.position.y / {samples_y}.0) + 0.5)");
    out.push_str(&format!(
        "  attr_in[{}u] = {};\n",
        POSITION / 4 + 1,
        match layout.window_height {
            Some(height) => format!("{height}.0 - {y}"),
            None => y,
        }
    ));
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

/// A `B10G11R11_FLOAT` texel's word as the rasterizer packs it: each channel through a half,
/// its mantissa rounded half up to 6 or 5 bits and held below infinity; negatives and NaN are 0.
const PACK_RG11B10: &str = "\
fn packSmallFloat(v: f32, mantissa_bits: u32) -> u32 {
  if (!(v > 0.0)) { return 0u; }
  let shift = 10u - mantissa_bits;
  let half = pack2x16float(vec2<f32>(v, 0.0)) & 0x7fffu;
  let largest = (1u << (5u + mantissa_bits)) - 1u;
  return min((half + (1u << (shift - 1u))) >> shift, largest);
}

fn packRg11b10(v: vec4<f32>) -> u32 {
  return packSmallFloat(v.x, 6u) | (packSmallFloat(v.y, 6u) << 11u) | (packSmallFloat(v.z, 5u) << 22u);
}

";
