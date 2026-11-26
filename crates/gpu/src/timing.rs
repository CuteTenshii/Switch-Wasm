//! How long the device takes over each pass, from timestamp queries. A debug option, off until
//! asked for: it adds a resolve and a readback to every submit, and per draw it undoes the
//! grouping of draws into passes.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use switch_core::gpu::renderer::DeviceTiming;
use switch_core::json;

use crate::readback::{MAP_FAILED, MAP_READY, MAP_WAITING};
use crate::Gpu;

/// WebGPU's limit on one query set, two a pass: the passes past it before a submit go untimed.
const QUERIES: u32 = 4096;
/// Readbacks the device has not answered yet; past this, passes go untimed rather than pile up.
/// Maps resolve only between run slices, and a slice can submit hundreds of times.
const IN_FLIGHT: usize = 1024;
/// Rows of the report, slowest first.
const ROWS: usize = 24;
/// Shaders a pass's row names before counting the rest.
const SHADERS_NAMED: usize = 4;

#[derive(Debug)]
pub(crate) struct DeviceTimes {
    per_draw: bool,
    queries: wgpu::QuerySet,
    resolved: wgpu::Buffer,
    /// The passes recorded since the last submit; pass `i` writes queries `2i` and `2i + 1`.
    passes: Vec<Pass>,
    in_flight: Vec<InFlight>,
    totals: HashMap<String, Total>,
    /// Passes left untimed because the query set or the readbacks were full, or a map failed.
    untimed: u64,
    since: web_time::Instant,
    /// Nanoseconds a timestamp tick.
    period: f64,
}

#[derive(Debug)]
struct Pass {
    what: String,
    /// Each draw's or launch's shaders, once.
    shaders: Vec<String>,
}

impl Pass {
    /// The row a pass is totalled under: what it is, and what it ran.
    fn row(&self) -> String {
        if self.shaders.is_empty() {
            return self.what.clone();
        }
        let named = self.shaders.iter().take(SHADERS_NAMED);
        let mut row = format!(
            "{}: {}",
            self.what,
            named.cloned().collect::<Vec<_>>().join(", ")
        );
        if self.shaders.len() > SHADERS_NAMED {
            row += &format!(" and {} more", self.shaders.len() - SHADERS_NAMED);
        }
        row
    }
}

/// A submit's timestamps on their way back.
#[derive(Debug)]
pub(crate) struct InFlight {
    staging: wgpu::Buffer,
    rows: Vec<String>,
    state: Arc<AtomicU8>,
}

/// A row's passes and their time on the device.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Total {
    pub(crate) passes: u64,
    pub(crate) ns: u64,
    pub(crate) max_ns: u64,
}

/// The size and formats of a pass's attachments, for its row.
pub(crate) fn attachments<'a>(textures: impl IntoIterator<Item = &'a wgpu::Texture>) -> String {
    let textures: Vec<&wgpu::Texture> = textures.into_iter().collect();
    let Some(first) = textures.first() else {
        return String::new();
    };
    let formats: Vec<String> = textures
        .iter()
        .map(|t| format!("{:?}", t.format()))
        .collect();
    format!("{}x{} {}", first.width(), first.height(), formats.join("+"))
}

impl Gpu {
    /// Start measuring by `timing`, from nothing, or stop. A device without timestamp
    /// queries reports that it has none.
    pub(crate) fn set_timing(&mut self, timing: DeviceTiming) {
        self.submit_batch();
        if let Some(old) = self.device_times.take() {
            old.resolved.destroy();
            for flight in old.in_flight {
                flight.staging.destroy();
            }
        }
        self.timing_asked = timing;
        if timing == DeviceTiming::Off
            || !self
                .device_features
                .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return;
        }
        self.device_times = Some(DeviceTimes {
            per_draw: timing == DeviceTiming::Draws,
            queries: self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("device time"),
                ty: wgpu::QueryType::Timestamp,
                count: QUERIES,
            }),
            resolved: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("device time"),
                size: u64::from(QUERIES) * 8,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            passes: Vec::new(),
            in_flight: Vec::new(),
            totals: HashMap::new(),
            untimed: 0,
            since: web_time::Instant::now(),
            period: f64::from(self.queue.get_timestamp_period()),
        });
    }

    /// Whether each draw gets a pass of its own, to be timed alone.
    pub(crate) fn draws_alone(&self) -> bool {
        self.device_times.as_ref().is_some_and(|t| t.per_draw)
    }

    /// A pass about to begin, and the index its timestamps go in, when timing is on and
    /// there is room.
    pub(crate) fn time_pass(&mut self, what: impl FnOnce() -> String) -> Option<u32> {
        let times = self.device_times.as_mut()?;
        if times.passes.len() as u32 >= QUERIES / 2 || times.in_flight.len() >= IN_FLIGHT {
            times.untimed += 1;
            return None;
        }
        times.passes.push(Pass {
            what: what(),
            shaders: Vec::new(),
        });
        Some(times.passes.len() as u32 - 1)
    }

    /// Name the shaders a draw or launch ran in a timed pass.
    pub(crate) fn note_shaders(&mut self, pass: Option<u32>, shaders: impl FnOnce() -> String) {
        let Some((times, pass)) = self.device_times.as_mut().zip(pass) else {
            return;
        };
        if let Some(pass) = times.passes.get_mut(pass as usize) {
            let shaders = shaders();
            if !pass.shaders.contains(&shaders) {
                pass.shaders.push(shaders);
            }
        }
    }

    pub(crate) fn render_timestamps(
        &self,
        pass: Option<u32>,
    ) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let times = self.device_times.as_ref()?;
        pass.map(|pass| wgpu::RenderPassTimestampWrites {
            query_set: &times.queries,
            beginning_of_pass_write_index: Some(2 * pass),
            end_of_pass_write_index: Some(2 * pass + 1),
        })
    }

    pub(crate) fn compute_timestamps(
        &self,
        pass: Option<u32>,
    ) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let times = self.device_times.as_ref()?;
        pass.map(|pass| wgpu::ComputePassTimestampWrites {
            query_set: &times.queries,
            beginning_of_pass_write_index: Some(2 * pass),
            end_of_pass_write_index: Some(2 * pass + 1),
        })
    }

    /// Copy the timestamps of the passes recorded so far out at the end of `encoder`, the last
    /// one of a submit. Hand what this returns to [`Gpu::map_times`] once submitted.
    pub(crate) fn resolve_times(&mut self, encoder: &mut wgpu::CommandEncoder) -> Option<InFlight> {
        let times = self.device_times.as_mut()?;
        if times.passes.is_empty() {
            return None;
        }
        let count = times.passes.len() as u32 * 2;
        let bytes = u64::from(count) * 8;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("device time"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.resolve_query_set(&times.queries, 0..count, &times.resolved, 0);
        encoder.copy_buffer_to_buffer(&times.resolved, 0, &staging, 0, bytes);
        let rows = times.passes.drain(..).map(|pass| pass.row()).collect();
        Some(InFlight {
            staging,
            rows,
            state: Arc::new(AtomicU8::new(MAP_WAITING)),
        })
    }

    /// Ask for the timestamps of a submit [`Gpu::resolve_times`] added to, and total the ones
    /// that have arrived.
    pub(crate) fn map_times(&mut self, flight: Option<InFlight>) {
        if let Some(flight) = flight {
            let sink = flight.state.clone();
            flight
                .staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    let done = if r.is_ok() { MAP_READY } else { MAP_FAILED };
                    sink.store(done, Ordering::Release);
                });
            if let Some(times) = self.device_times.as_mut() {
                times.in_flight.push(flight);
            }
        }
        self.total_times();
    }

    fn total_times(&mut self) {
        let Some(times) = self.device_times.as_mut() else {
            return;
        };
        if times.in_flight.is_empty() {
            return;
        }
        // Natively the callbacks run only from a poll; in a browser, from the event loop.
        let _ = self.device.poll(wgpu::PollType::Poll);
        let period = times.period;
        let mut kept = Vec::with_capacity(times.in_flight.len());
        for flight in times.in_flight.drain(..) {
            match flight.state.load(Ordering::Acquire) {
                MAP_WAITING => {
                    kept.push(flight);
                    continue;
                }
                MAP_READY => {
                    if let Ok(mapped) = flight.staging.slice(..).get_mapped_range() {
                        let ticks: Vec<u64> = mapped
                            .as_chunks::<8>()
                            .0
                            .iter()
                            .map(|b| u64::from_le_bytes(*b))
                            .collect();
                        for (row, &[begin, end]) in flight.rows.iter().zip(ticks.as_chunks::<2>().0)
                        {
                            // A pass whose encoder was never submitted wrote nothing.
                            if begin == 0 || end < begin {
                                times.untimed += 1;
                                continue;
                            }
                            let ns = ((end - begin) as f64 * period) as u64;
                            let total = times.totals.entry(row.clone()).or_default();
                            total.passes += 1;
                            total.ns += ns;
                            total.max_ns = total.max_ns.max(ns);
                        }
                    } else {
                        times.untimed += flight.rows.len() as u64;
                    }
                    flight.staging.unmap();
                }
                _ => times.untimed += flight.rows.len() as u64,
            }
            flight.staging.destroy();
        }
        times.in_flight = kept;
    }

    /// Every row with its totals, slowest first, and how many passes went untimed.
    pub(crate) fn device_time_rows(&self) -> (Vec<(&str, Total)>, u64) {
        let Some(times) = &self.device_times else {
            return (Vec::new(), 0);
        };
        let mut rows: Vec<(&str, Total)> = times
            .totals
            .iter()
            .map(|(what, total)| (what.as_str(), *total))
            .collect();
        rows.sort_by_key(|(_, total)| std::cmp::Reverse(total.ns));
        (rows, times.untimed)
    }

    /// The slowest rows and the totals, as the `deviceTime` object of the report.
    pub(crate) fn report_times(&self, out: &mut json::Object) {
        if self.timing_asked == DeviceTiming::Off {
            return;
        }
        let ms = |ns: u64| json::Decimals(ns as f64 / 1e6, 2);
        let (rows, untimed) = self.device_time_rows();
        out.object("deviceTime", |o| {
            let Some(times) = &self.device_times else {
                o.field("mode", "unsupported");
                return;
            };
            let mode = if times.per_draw { "draws" } else { "passes" };
            o.field("mode", mode)
                .field(
                    "seconds",
                    json::Decimals(times.since.elapsed().as_secs_f64(), 1),
                )
                .field("ms", ms(rows.iter().map(|(_, t)| t.ns).sum()))
                .field("passes", rows.iter().map(|(_, t)| t.passes).sum::<u64>())
                .field("untimed", untimed)
                .array("rows", |a| {
                    for (what, total) in rows.iter().take(ROWS) {
                        a.object(|r| {
                            r.field("what", *what)
                                .field("passes", total.passes)
                                .field("ms", ms(total.ns))
                                .field("maxMs", ms(total.max_ns));
                        });
                    }
                });
        });
    }

    /// Wait for the timestamps still on their way, natively; a browser cannot block.
    pub(crate) fn wait_for_times(&mut self) {
        let waiting = self
            .device_times
            .as_ref()
            .is_some_and(|t| !t.in_flight.is_empty());
        if waiting && !cfg!(target_arch = "wasm32") {
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            });
        }
        self.total_times();
    }

    /// The slowest rows, for the `Drop` summary of a native run; a browser's is the report.
    pub(crate) fn print_times(&self) {
        let Some(times) = &self.device_times else {
            return;
        };
        let (rows, untimed) = self.device_time_rows();
        let ms = |ns: u64| ns as f64 / 1e6;
        eprintln!(
            "[gpu] device time over {:.1} s: {:.1} ms in {} passes, {untimed} untimed",
            times.since.elapsed().as_secs_f64(),
            ms(rows.iter().map(|(_, t)| t.ns).sum()),
            rows.iter().map(|(_, t)| t.passes).sum::<u64>(),
        );
        for (what, total) in rows.iter().take(ROWS) {
            eprintln!(
                "[gpu]   {:9.2} ms  {:6}x  max {:7.2} ms  {what}",
                ms(total.ns),
                total.passes,
                ms(total.max_ns),
            );
        }
    }
}
