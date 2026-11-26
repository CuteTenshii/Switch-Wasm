//! The probability context a frame decodes with, the symbol counts it gathers, and the backward
//! adaptation that turns the two into the next frame's context.

use super::tables::*;
use super::types::*;

/// `[tx size][plane > 0][inter][band][context][node]`.
pub type CoefProbs = [[[[[[u8; 3]; 6]; 6]; 2]; 2]; 4];
pub type CoefCounts = [[[[[[u32; 4]; 6]; 6]; 2]; 2]; 4];
pub type EobBranchCounts = [[[[[u32; 6]; 6]; 2]; 2]; 4];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MvComponentProbs {
    pub sign: u8,
    pub classes: [u8; 10],
    pub class0: [u8; 1],
    pub bits: [u8; 10],
    pub class0_fp: [[u8; 3]; 2],
    pub fp: [u8; 3],
    pub class0_hp: u8,
    pub hp: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MvProbs {
    pub joints: [u8; 3],
    pub comps: [MvComponentProbs; 2],
}

impl MvProbs {
    fn from_raw(raw: &[u8; 69]) -> MvProbs {
        let component = |at: usize| {
            let r = &raw[at..at + 33];
            MvComponentProbs {
                sign: r[0],
                classes: r[1..11].try_into().unwrap(),
                class0: [r[11]],
                bits: r[12..22].try_into().unwrap(),
                class0_fp: [r[22..25].try_into().unwrap(), r[25..28].try_into().unwrap()],
                fp: r[28..31].try_into().unwrap(),
                class0_hp: r[31],
                hp: r[32],
            }
        };
        MvProbs {
            joints: raw[0..3].try_into().unwrap(),
            comps: [component(3), component(36)],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameContext {
    pub y_mode: [[u8; 9]; 4],
    pub uv_mode: [[u8; 9]; 10],
    pub partition: [[u8; 3]; 16],
    pub coef: CoefProbs,
    pub switchable_interp: [[u8; 2]; 4],
    pub inter_mode: [[u8; 3]; 7],
    pub intra_inter: [u8; 4],
    pub comp_inter: [u8; 5],
    pub single_ref: [[u8; 2]; 5],
    pub comp_ref: [u8; 5],
    pub tx8: [[u8; 1]; 2],
    pub tx16: [[u8; 2]; 2],
    pub tx32: [[u8; 3]; 2],
    pub skip: [u8; 3],
    pub mv: MvProbs,
}

impl FrameContext {
    pub fn defaults() -> FrameContext {
        let raw = &DEFAULT_TX_PROBS_RAW;
        FrameContext {
            y_mode: DEFAULT_IF_Y_PROBS,
            uv_mode: DEFAULT_IF_UV_PROBS,
            partition: DEFAULT_PARTITION_PROBS,
            coef: [
                DEFAULT_COEF_PROBS_4X4,
                DEFAULT_COEF_PROBS_8X8,
                DEFAULT_COEF_PROBS_16X16,
                DEFAULT_COEF_PROBS_32X32,
            ],
            switchable_interp: [[235, 162], [36, 255], [34, 3], [149, 144]],
            inter_mode: DEFAULT_INTER_MODE_PROBS,
            intra_inter: [9, 102, 187, 225],
            comp_inter: DEFAULT_COMP_INTER_PROBS,
            single_ref: DEFAULT_SINGLE_REF_PROBS,
            comp_ref: DEFAULT_COMP_REF_PROBS,
            tx32: [[raw[0], raw[1], raw[2]], [raw[3], raw[4], raw[5]]],
            tx16: [[raw[6], raw[7]], [raw[8], raw[9]]],
            tx8: [[raw[10]], [raw[11]]],
            skip: DEFAULT_SKIP_PROBS,
            mv: MvProbs::from_raw(&DEFAULT_NMV_CONTEXT_RAW),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MvComponentCounts {
    pub sign: [u32; 2],
    pub classes: [u32; 11],
    pub class0: [u32; 2],
    pub bits: [[u32; 2]; 10],
    pub class0_fp: [[u32; 4]; 2],
    pub fp: [u32; 4],
    pub class0_hp: [u32; 2],
    pub hp: [u32; 2],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Counts {
    pub y_mode: [[u32; 10]; 4],
    pub uv_mode: [[u32; 10]; 10],
    pub partition: [[u32; 4]; 16],
    pub coef: Box<CoefCounts>,
    pub eob_branch: Box<EobBranchCounts>,
    pub switchable_interp: [[u32; 3]; 4],
    pub inter_mode: [[u32; 4]; 7],
    pub intra_inter: [[u32; 2]; 4],
    pub comp_inter: [[u32; 2]; 5],
    pub single_ref: [[[u32; 2]; 2]; 5],
    pub comp_ref: [[u32; 2]; 5],
    pub tx8: [[u32; 2]; 2],
    pub tx16: [[u32; 3]; 2],
    pub tx32: [[u32; 4]; 2],
    pub skip: [[u32; 2]; 3],
    pub mv_joints: [u32; 4],
    pub mv_comps: [MvComponentCounts; 2],
}

impl Default for Counts {
    fn default() -> Counts {
        Counts {
            y_mode: Default::default(),
            uv_mode: Default::default(),
            partition: Default::default(),
            coef: Box::new([[[[[[0; 4]; 6]; 6]; 2]; 2]; 4]),
            eob_branch: Box::new([[[[[0; 6]; 6]; 2]; 2]; 4]),
            switchable_interp: Default::default(),
            inter_mode: Default::default(),
            intra_inter: Default::default(),
            comp_inter: Default::default(),
            single_ref: Default::default(),
            comp_ref: Default::default(),
            tx8: Default::default(),
            tx16: Default::default(),
            tx32: Default::default(),
            skip: Default::default(),
            mv_joints: Default::default(),
            mv_comps: Default::default(),
        }
    }
}

const COUNT_TO_UPDATE_FACTOR: [u32; 21] = [
    0, 6, 12, 19, 25, 32, 38, 44, 51, 57, 64, 70, 76, 83, 89, 96, 102, 108, 115, 121, 128,
];

fn get_prob(num: u32, den: u32) -> u8 {
    let p = ((u64::from(num) * 256 + u64::from(den >> 1)) / u64::from(den)) as i32;
    p.clamp(1, 255) as u8
}

fn weighted_prob(prob1: u8, prob2: u8, factor: u32) -> u8 {
    ((u32::from(prob1) * (256 - factor) + u32::from(prob2) * factor + 128) >> 8) as u8
}

fn merge_probs(pre: u8, ct: [u32; 2], count_sat: u32, max_update_factor: u32) -> u8 {
    let den = ct[0] + ct[1];
    let prob = if den == 0 { 128 } else { get_prob(ct[0], den) };
    let count = den.min(count_sat);
    weighted_prob(pre, prob, max_update_factor * count / count_sat)
}

fn mode_mv_merge(pre: u8, ct: [u32; 2]) -> u8 {
    let den = ct[0] + ct[1];
    if den == 0 {
        return pre;
    }
    let factor = COUNT_TO_UPDATE_FACTOR[den.min(20) as usize];
    weighted_prob(pre, get_prob(ct[0], den), factor)
}

fn tree_merge(tree: &[i8], at: usize, pre: &[u8], counts: &[u32], out: &mut [u8]) -> u32 {
    let side = |index: i8, out: &mut [u8]| {
        if index <= 0 {
            counts[(-index) as usize]
        } else {
            tree_merge(tree, index as usize, pre, counts, out)
        }
    };
    let left = side(tree[at], out);
    let right = side(tree[at + 1], out);
    out[at >> 1] = mode_mv_merge(pre[at >> 1], [left, right]);
    left + right
}

fn merge_tree(tree: &[i8], pre: &[u8], counts: &[u32], out: &mut [u8]) {
    tree_merge(tree, 0, pre, counts, out);
}

impl FrameContext {
    /// The coefficient probabilities of a decoded frame, merged from `pre` with its counts.
    pub fn adapt_coef(
        &mut self,
        pre: &FrameContext,
        counts: &Counts,
        intra: bool,
        last_was_key: bool,
    ) {
        let (count_sat, update_factor) = if intra {
            (24, 112)
        } else if last_was_key {
            (24, 128)
        } else {
            (24, 112)
        };
        for tx in 0..4 {
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..6 {
                        for l in 0..(if k == 0 { 3 } else { 6 }) {
                            let c = counts.coef[tx][i][j][k][l];
                            let eob = counts.eob_branch[tx][i][j][k][l];
                            let branch = [
                                [c[3], eob.wrapping_sub(c[3])],
                                [c[0], c[1] + c[2]],
                                [c[1], c[2]],
                            ];
                            for m in 0..3 {
                                self.coef[tx][i][j][k][l][m] = merge_probs(
                                    pre.coef[tx][i][j][k][l][m],
                                    branch[m],
                                    count_sat,
                                    update_factor,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Everything but the coefficients, for a frame that is not intra only.
    pub fn adapt_modes(
        &mut self,
        pre: &FrameContext,
        counts: &Counts,
        switchable: bool,
        tx_select: bool,
    ) {
        for i in 0..4 {
            self.intra_inter[i] = mode_mv_merge(pre.intra_inter[i], counts.intra_inter[i]);
        }
        for i in 0..5 {
            self.comp_inter[i] = mode_mv_merge(pre.comp_inter[i], counts.comp_inter[i]);
            self.comp_ref[i] = mode_mv_merge(pre.comp_ref[i], counts.comp_ref[i]);
            for j in 0..2 {
                self.single_ref[i][j] =
                    mode_mv_merge(pre.single_ref[i][j], counts.single_ref[i][j]);
            }
        }
        for i in 0..7 {
            merge_tree(
                &INTER_MODE_TREE,
                &pre.inter_mode[i],
                &counts.inter_mode[i],
                &mut self.inter_mode[i],
            );
        }
        for i in 0..4 {
            merge_tree(
                &INTRA_MODE_TREE,
                &pre.y_mode[i],
                &counts.y_mode[i],
                &mut self.y_mode[i],
            );
        }
        for i in 0..10 {
            merge_tree(
                &INTRA_MODE_TREE,
                &pre.uv_mode[i],
                &counts.uv_mode[i],
                &mut self.uv_mode[i],
            );
        }
        for i in 0..16 {
            merge_tree(
                &PARTITION_TREE,
                &pre.partition[i],
                &counts.partition[i],
                &mut self.partition[i],
            );
        }
        if switchable {
            for i in 0..4 {
                merge_tree(
                    &SWITCHABLE_INTERP_TREE,
                    &pre.switchable_interp[i],
                    &counts.switchable_interp[i],
                    &mut self.switchable_interp[i],
                );
            }
        }
        if tx_select {
            for i in 0..2 {
                let c8 = counts.tx8[i];
                self.tx8[i][0] = mode_mv_merge(pre.tx8[i][0], [c8[0], c8[1]]);
                let c16 = counts.tx16[i];
                let branch16 = [[c16[0], c16[1] + c16[2]], [c16[1], c16[2]]];
                for j in 0..2 {
                    self.tx16[i][j] = mode_mv_merge(pre.tx16[i][j], branch16[j]);
                }
                let c32 = counts.tx32[i];
                let branch32 = [
                    [c32[0], c32[1] + c32[2] + c32[3]],
                    [c32[1], c32[2] + c32[3]],
                    [c32[2], c32[3]],
                ];
                for j in 0..3 {
                    self.tx32[i][j] = mode_mv_merge(pre.tx32[i][j], branch32[j]);
                }
            }
        }
        for i in 0..3 {
            self.skip[i] = mode_mv_merge(pre.skip[i], counts.skip[i]);
        }
    }

    pub fn adapt_mv(&mut self, pre: &FrameContext, counts: &Counts, allow_hp: bool) {
        merge_tree(
            &MV_JOINT_TREE,
            &pre.mv.joints,
            &counts.mv_joints,
            &mut self.mv.joints,
        );
        for i in 0..2 {
            let (comp, pre_comp, c) =
                (&mut self.mv.comps[i], &pre.mv.comps[i], &counts.mv_comps[i]);
            comp.sign = mode_mv_merge(pre_comp.sign, c.sign);
            merge_tree(
                &MV_CLASS_TREE,
                &pre_comp.classes,
                &c.classes,
                &mut comp.classes,
            );
            merge_tree(
                &MV_CLASS0_TREE,
                &pre_comp.class0,
                &c.class0,
                &mut comp.class0,
            );
            for j in 0..10 {
                comp.bits[j] = mode_mv_merge(pre_comp.bits[j], c.bits[j]);
            }
            for j in 0..2 {
                merge_tree(
                    &MV_FP_TREE,
                    &pre_comp.class0_fp[j],
                    &c.class0_fp[j],
                    &mut comp.class0_fp[j],
                );
            }
            merge_tree(&MV_FP_TREE, &pre_comp.fp, &c.fp, &mut comp.fp);
            if allow_hp {
                comp.class0_hp = mode_mv_merge(pre_comp.class0_hp, c.class0_hp);
                comp.hp = mode_mv_merge(pre_comp.hp, c.hp);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_motion_vector_probabilities_read_back_in_libvpx_order() {
        let mv = FrameContext::defaults().mv;
        assert_eq!(mv.joints, [32, 64, 96]);
        assert_eq!(mv.comps[0].sign, 128);
        assert_eq!(
            mv.comps[0].classes,
            [224, 144, 192, 168, 192, 176, 192, 198, 198, 245]
        );
        assert_eq!(mv.comps[0].class0, [216]);
        assert_eq!(mv.comps[1].class0, [208]);
        assert_eq!(mv.comps[1].hp, 128);
    }

    #[test]
    fn default_transform_probabilities_are_split_by_size() {
        let fc = FrameContext::defaults();
        assert_eq!(fc.tx32, [[3, 136, 37], [5, 52, 13]]);
        assert_eq!(fc.tx16, [[20, 152], [15, 101]]);
        assert_eq!(fc.tx8, [[100], [66]]);
    }

    #[test]
    fn an_unseen_symbol_leaves_a_probability_alone_and_a_seen_one_moves_it() {
        assert_eq!(mode_mv_merge(77, [0, 0]), 77);
        assert!(mode_mv_merge(77, [20, 0]) > 77);
        assert!(mode_mv_merge(77, [0, 20]) < 77);
    }

    #[test]
    fn tree_merge_visits_every_node_once() {
        let pre = [128u8; 9];
        let mut out = [0u8; 9];
        let counts = [10, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        merge_tree(&INTRA_MODE_TREE, &pre, &counts, &mut out);
        assert!(out[0] > 128);
        assert_eq!(out[1], 128);
    }
}
