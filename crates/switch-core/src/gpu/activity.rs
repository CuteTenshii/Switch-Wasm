//! What the GPU did, by surface: the draws into each render target, the
//! clears of each, every copy and blit by source and destination, and which
//! surface each presented frame scanned out.
//!
//! Counts alone ("9 draws, 1 clear") cannot say whether the draws landed in
//! the surface that was then presented, which is the question a black frame
//! asks. Each engine keeps one of these, and [`crate::gpu::Gpu::take_activity`]
//! gathers them for the host to report.
//!
//! Keyed by numbers, not text: a title issues thousands of draws a frame, and
//! a label formatted per draw would cost more than the draw's bookkeeping.
//! The label is built once, when a surface is first seen.

use std::collections::BTreeMap;

/// How many distinct entries one tally holds between two readings. Past it,
/// new surfaces are summed under one entry rather than dropped.
const CAP: usize = 256;

/// What an entry counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    /// Draws into a colour target. `amount` is vertices (or indices).
    Draw,
    /// Clears of a colour or depth target.
    Clear,
    /// Copy-engine transfers. `amount` is bytes.
    Copy,
    /// Inline uploads into memory. `amount` is bytes.
    Upload,
    /// 2D-engine blits. `amount` is destination pixels.
    Blit,
    /// Frames scanned out of a surface.
    Present,
}

impl Kind {
    /// The name the host sees.
    pub const fn name(self) -> &'static str {
        match self {
            Kind::Draw => "draw",
            Kind::Clear => "clear",
            Kind::Copy => "copy",
            Kind::Upload => "upload",
            Kind::Blit => "blit",
            Kind::Present => "present",
        }
    }
}

/// One entry: what it is about, how many times, and how much.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Tally {
    pub label: String,
    pub count: u64,
    pub amount: u64,
    /// Of `count`, how many the backend refused: a hole in the frame.
    pub failed: u64,
}

/// The entries of one engine, or of the whole GPU once gathered.
#[derive(Debug, Default, Clone)]
pub struct GpuActivity {
    tallies: BTreeMap<(Kind, u64, u64), Tally>,
}

impl GpuActivity {
    /// Count one `kind` on the surfaces `a` and `b` (a target, or a copy's
    /// source and destination). `label` describes them, and is only called
    /// the first time the pair is seen.
    pub fn note(
        &mut self,
        kind: Kind,
        a: u64,
        b: u64,
        amount: u64,
        failed: bool,
        label: impl FnOnce() -> String,
    ) {
        let mut key = (kind, a, b);
        if !self.tallies.contains_key(&key) && self.tallies.len() >= CAP {
            key = (kind, u64::MAX, u64::MAX);
        }
        let tally = self.tallies.entry(key).or_insert_with(|| Tally {
            label: if key.1 == u64::MAX && key.2 == u64::MAX {
                "(other surfaces)".to_owned()
            } else {
                label()
            },
            ..Tally::default()
        });
        tally.count += 1;
        tally.amount += amount;
        tally.failed += u64::from(failed);
    }

    /// Move everything in `other` into this one, leaving `other` empty.
    pub fn absorb(&mut self, other: &mut GpuActivity) {
        for (key, tally) in std::mem::take(&mut other.tallies) {
            let into = self.tallies.entry(key).or_insert_with(|| Tally {
                label: tally.label.clone(),
                ..Tally::default()
            });
            into.count += tally.count;
            into.amount += tally.amount;
            into.failed += tally.failed;
        }
    }

    /// Every entry since the last call, in kind order.
    pub fn take(&mut self) -> Vec<(Kind, Tally)> {
        std::mem::take(&mut self.tallies)
            .into_iter()
            .map(|((kind, _, _), tally)| (kind, tally))
            .collect()
    }
}

/// A surface the way the entries name it: where the GPU sees it, where the
/// CPU does when that is known (which is what matches a render target to the
/// buffer `present` names), its size and format.
pub fn surface_text(gpu_va: u64, cpu: Option<u32>, width: u32, height: u32, format: u32) -> String {
    let cpu = match cpu {
        Some(cpu) => format!(" (cpu {cpu:#x})"),
        None => String::new(),
    };
    format!("{gpu_va:#x}{cpu} {width}x{height} fmt {format:#x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_surface_is_labelled_once_and_counted_every_time() {
        let mut activity = GpuActivity::default();
        let mut labelled = 0;
        for _ in 0..3 {
            activity.note(Kind::Draw, 0x1000, 0, 6, false, || {
                labelled += 1;
                "rt".to_owned()
            });
        }
        activity.note(Kind::Draw, 0x1000, 0, 3, true, || unreachable!());
        let taken = activity.take();
        assert_eq!(labelled, 1);
        assert_eq!(
            taken,
            vec![(
                Kind::Draw,
                Tally {
                    label: "rt".to_owned(),
                    count: 4,
                    amount: 21,
                    failed: 1
                }
            )]
        );
        assert!(activity.take().is_empty(), "taken, not read");
    }

    #[test]
    fn surfaces_past_the_cap_are_summed_rather_than_lost() {
        let mut activity = GpuActivity::default();
        for addr in 0..CAP as u64 + 10 {
            activity.note(Kind::Copy, addr, addr, 1, false, || format!("{addr}"));
        }
        let taken = activity.take();
        assert_eq!(taken.len(), CAP + 1);
        let other = taken.last().unwrap();
        assert_eq!(other.1.label, "(other surfaces)");
        assert_eq!(other.1.count, 10);
    }

    #[test]
    fn gathering_two_engines_adds_their_counts() {
        let mut a = GpuActivity::default();
        let mut b = GpuActivity::default();
        a.note(Kind::Clear, 1, 0, 0, false, || "x".to_owned());
        b.note(Kind::Clear, 1, 0, 0, false, || "x".to_owned());
        a.absorb(&mut b);
        assert_eq!(a.take()[0].1.count, 2);
        assert!(b.take().is_empty());
    }
}
