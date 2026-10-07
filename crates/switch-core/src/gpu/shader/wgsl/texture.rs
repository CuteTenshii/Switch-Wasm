//! Texture binding, sampling, gathering and queries.

use super::emitter::Emitter;
use super::{tex_dim_code, Unsupported};
use crate::gpu::shader::isa::{Op, TexDim, TexsStore};
use crate::gpu::texture::TextureSlot;

const COMPONENT: [&str; 4] = ["x", "y", "z", "w"];

impl Emitter<'_> {
    /// Record that the program samples `slot` as `dim`, a shadow map or not.
    fn bind_texture(
        &mut self,
        at: usize,
        slot: TextureSlot,
        dim: TexDim,
        compare: bool,
    ) -> Result<(), Unsupported> {
        match self.textures.iter().find(|&&(seen, _, _)| seen == slot) {
            // A slot sampled both as colour and depth cannot be one binding.
            Some(&(_, _, was)) if was != compare => Err(Unsupported::DepthCompare { at }),
            Some(_) => Ok(()),
            None => {
                self.textures.push((slot, dim, compare));
                Ok(())
            }
        }
    }

    /// `tex.aoffi`: a sample offset by an immediate texel offset (signed nibbles, x low).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sample_offset(
        &mut self,
        at: usize,
        slot: TextureSlot,
        dim: TexDim,
        dref: Option<u8>,
        coords: [u8; 3],
        layer: Option<u8>,
        offset: u8,
        op: Op,
    ) -> Result<(), Unsupported> {
        let refuse = Unsupported::Op { at, op };
        if dref.is_some() || !matches!(dim, TexDim::T2d | TexDim::T2dArray) {
            return Err(refuse);
        }
        let packed = self.constant_in(at, offset).ok_or(refuse)?;
        let axis = |shift: u32| ((packed >> shift) as i32) << 28 >> 28;
        let texel = (axis(0), axis(4));
        let index = match self.texture_offsets.iter().position(|&seen| seen == texel) {
            Some(index) => index,
            None => {
                self.texture_offsets.push(texel);
                self.texture_offsets.len() - 1
            }
        };
        self.bind_texture(at, slot, dim, false)?;
        let u = self.f(coords[0]);
        let v = self.f(coords[1]);
        let layer = match layer {
            Some(reg) => {
                let reg = self.r(reg);
                format!("({reg} & 0xffffu)")
            }
            None => "0u".to_string(),
        };
        let color = self.bind(&format!(
            "texSampleOffset({}u, {index}u, {u}, {v}, {layer})",
            slot.key()
        ));
        for (reg, store, _) in self.program.texs_writes(at).to_vec() {
            if let TexsStore::Float(channel) = store {
                self.set_f(reg, &format!("{color}.{}", COMPONENT[channel]));
            }
        }
        Ok(())
    }

    /// `txq`: texture size as integers.
    pub(super) fn query_texture(&mut self, at: usize, slot: TextureSlot, lod: u8) {
        self.queried.push(slot);
        let lod = self.r(lod);
        let size = self.bind(&format!("texDims({}u, {lod})", slot.key()));
        for (reg, store, _) in self.program.texs_writes(at).to_vec() {
            if let TexsStore::Float(channel) = store {
                self.set_r(reg, &format!("{size}.{}", COMPONENT[channel]));
            }
        }
    }

    /// `tld4`: one channel of the four bilinear texels, as `textureGather`.
    pub(super) fn gather_texture(
        &mut self,
        at: usize,
        slot: TextureSlot,
        dim: TexDim,
        coords: [u8; 3],
        layer: Option<u8>,
        component: u8,
    ) -> Result<(), Unsupported> {
        self.bind_texture(at, slot, dim, false)?;
        let u = self.f(coords[0]);
        let v = self.f(coords[1]);
        let layer = match layer {
            Some(reg) => {
                let reg = self.r(reg);
                format!("({reg} & 0xffffu)")
            }
            None => "0u".to_string(),
        };
        let texels = self.bind(&format!(
            "texGather({}u, {component}u, {u}, {v}, {layer})",
            slot.key()
        ));
        for (reg, store, _) in self.program.texs_writes(at).to_vec() {
            if let TexsStore::Float(channel) = store {
                self.set_f(reg, &format!("{texels}.{}", COMPONENT[channel]));
            }
        }
        Ok(())
    }

    /// Sample and store the channels, for `texs` and `tex`.
    pub(super) fn sample_texture(
        &mut self,
        at: usize,
        slot: TextureSlot,
        dim: TexDim,
        dref: Option<u8>,
        coords: [u8; 3],
        layer: Option<u8>,
    ) -> Result<(), Unsupported> {
        let compare = dref.is_some();
        self.bind_texture(at, slot, dim, compare)?;
        let key = slot.key();
        let u = self.f(coords[0]);
        let v = match dim {
            TexDim::T1d => "0.0".to_string(),
            _ => self.f(coords[1]),
        };
        // The layer is an integer in the register's low half.
        let layer = match layer {
            Some(reg) => {
                let reg = self.r(reg);
                format!("({reg} & 0xffffu)")
            }
            None => "0u".to_string(),
        };
        // A 3D third coordinate is normalized; an array's is a layer.
        let w = match dim {
            TexDim::T3d | TexDim::TCube | TexDim::TCubeArray => self.f(coords[2]),
            _ => "0.0".to_string(),
        };
        let code = tex_dim_code(dim);
        let color = match dref {
            // A shadow sample fills every requested channel except alpha.
            Some(reg) => {
                let reference = self.f(reg);
                self.bind(&format!(
                    "texSampleCompare({key}u, {code}u, {u}, {v}, {layer}, {reference})"
                ))
            }
            None => self.bind(&format!(
                "texSample({key}u, {code}u, {u}, {v}, {layer}, {w})"
            )),
        };
        // Stored now rather than at first use like the interpreter; equivalent
        // unless the destination is overwritten before being read.
        let writes = self.program.texs_writes(at).to_vec();
        for (reg, store, _) in writes {
            match store {
                TexsStore::Float(channel) => {
                    self.set_f(reg, &format!("{color}.{}", COMPONENT[channel]));
                }
                // `.F16` packs two channels as halves.
                TexsStore::Halves(low, high) => {
                    let half = |c: Option<usize>| match c {
                        Some(channel) => format!("{color}.{}", COMPONENT[channel]),
                        None => "0.0".to_string(),
                    };
                    self.set_r(
                        reg,
                        &format!(
                            "pack2x16float(vec2<f32>({}, {}))",
                            half(Some(low)),
                            half(high)
                        ),
                    );
                }
            }
        }
        Ok(())
    }
}
