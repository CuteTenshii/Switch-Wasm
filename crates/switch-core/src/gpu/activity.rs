//! GPU work by surface (draws, clears, copies, blits, presents), so a black
//! frame can be traced to where the draws landed. Labels are built once per surface.

use std::collections::BTreeMap;

/// Distinct entries per reading; past it, new surfaces share one entry.
const CAP: usize = 256;

/// Distinct refusal reasons per reading; reasons can carry addresses.
const REFUSAL_CAP: usize = 32;

const OTHER_REASONS: &str = "(other reasons)";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    /// `amount` is vertices or indices.
    Draw,
    Clear,
    /// `amount` is bytes.
    Copy,
    /// `amount` is bytes.
    Upload,
    /// `amount` is destination pixels.
    Blit,
    Present,
    /// Only ever a refusal.
    Dispatch,
}

impl Kind {
    pub const fn name(self) -> &'static str {
        match self {
            Kind::Draw => "draw",
            Kind::Clear => "clear",
            Kind::Copy => "copy",
            Kind::Upload => "upload",
            Kind::Blit => "blit",
            Kind::Present => "present",
            Kind::Dispatch => "dispatch",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Tally {
    pub label: String,
    pub count: u64,
    pub amount: u64,
    /// Of `count`, how many the backend refused.
    pub failed: u64,
}

/// One engine's entries; refusals are kept by reason rather than by surface.
#[derive(Debug, Default, Clone)]
pub struct GpuActivity {
    tallies: BTreeMap<(Kind, u64, u64), Tally>,
    refusals: BTreeMap<(Kind, String), u64>,
}

impl GpuActivity {
    /// Count one `kind` on surfaces `a` and `b`; `label` is only called for a new pair.
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

    pub fn refuse(&mut self, kind: Kind, reason: String) {
        self.refuse_times(kind, reason, 1);
    }

    fn refuse_times(&mut self, kind: Kind, reason: String, times: u64) {
        let mut key = (kind, reason);
        if !self.refusals.contains_key(&key) && self.refusals.len() >= REFUSAL_CAP {
            key.1 = OTHER_REASONS.to_owned();
        }
        *self.refusals.entry(key).or_insert(0) += times;
    }

    /// Move everything from `other` into this one, leaving it empty.
    pub fn absorb(&mut self, other: &mut GpuActivity) {
        for ((kind, reason), times) in std::mem::take(&mut other.refusals) {
            self.refuse_times(kind, reason, times);
        }
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

    /// Every refusal since the last call, in kind order.
    pub fn take_refusals(&mut self) -> Vec<(Kind, String, u64)> {
        std::mem::take(&mut self.refusals)
            .into_iter()
            .map(|((kind, reason), times)| (kind, reason, times))
            .collect()
    }
}

/// A surface's GPU address, CPU address when known, size, and format.
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

    #[test]
    fn refusals_are_counted_by_reason_across_engines() {
        let mut a = GpuActivity::default();
        let mut b = GpuActivity::default();
        a.refuse(Kind::Draw, "no ldg b128".to_owned());
        b.refuse(Kind::Draw, "no ldg b128".to_owned());
        b.refuse(Kind::Dispatch, "bad qmd".to_owned());
        a.absorb(&mut b);
        assert_eq!(
            a.take_refusals(),
            vec![
                (Kind::Draw, "no ldg b128".to_owned(), 2),
                (Kind::Dispatch, "bad qmd".to_owned(), 1),
            ]
        );
        assert!(a.take_refusals().is_empty(), "taken, not read");
    }

    #[test]
    fn reasons_past_the_cap_are_summed_rather_than_lost() {
        let mut activity = GpuActivity::default();
        for at in 0..REFUSAL_CAP + 5 {
            activity.refuse(Kind::Draw, format!("refused at {at}"));
        }
        let taken = activity.take_refusals();
        assert_eq!(taken.len(), REFUSAL_CAP + 1);
        let other = taken.iter().find(|(_, reason, _)| reason == OTHER_REASONS);
        assert_eq!(other.map(|(_, _, times)| *times), Some(5));
    }
}
