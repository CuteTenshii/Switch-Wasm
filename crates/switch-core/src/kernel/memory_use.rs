//! How much guest memory the title has backed, and where in its address space.

use crate::cpu::*;
use crate::mem::AddressSpace;

/// The parts of a title's address space, each one or two ranges of [`MemoryLayout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    /// The ASLR region below the stack region, where `main`, `sdk` and the NRO load.
    Code,
    /// Thread stacks: `svcGetInfo`'s stack region.
    Stacks,
    /// The exit trampolines, TLS blocks and the main thread's stack.
    Runtime,
    /// Modules `ldr:ro` maps at run time.
    Loaded,
    Heap,
    Alias,
}

impl Region {
    pub const ALL: [Region; 6] = [
        Region::Code,
        Region::Stacks,
        Region::Runtime,
        Region::Loaded,
        Region::Heap,
        Region::Alias,
    ];

    /// The name the page keys its labels by.
    pub fn key(self) -> &'static str {
        match self {
            Region::Code => "code",
            Region::Stacks => "stacks",
            Region::Runtime => "runtime",
            Region::Loaded => "loaded",
            Region::Heap => "heap",
            Region::Alias => "alias",
        }
    }

    /// The address range, as `start..end`.
    pub fn range(self, layout: &MemoryLayout) -> (u64, u64) {
        let span = |start: u32, size: u32| (u64::from(start), u64::from(start) + u64::from(size));
        match self {
            Region::Code => (
                u64::from(GUEST_ASLR_REGION_ADDR),
                u64::from(GUEST_STACK_REGION_ADDR),
            ),
            Region::Stacks => span(GUEST_STACK_REGION_ADDR, GUEST_STACK_REGION_SIZE),
            Region::Runtime => (
                u64::from(SELF_RETURN_TRAMPOLINE),
                u64::from(RO_MODULE_REGION_ADDR),
            ),
            Region::Loaded => span(RO_MODULE_REGION_ADDR, RO_MODULE_REGION_SIZE),
            Region::Heap => span(layout.heap_addr, layout.heap_size),
            Region::Alias => span(layout.alias_addr, layout.alias_size),
        }
    }
}

/// The title's backed guest memory, by region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TitleMemory {
    pub backed: u64,
    /// The ceiling on backed memory; past it, guest writes fault.
    pub cap: u64,
    pub regions: Vec<(Region, (u64, u64), u64)>,
    /// Backed outside every region: the shared display buffers and the homebrew environment.
    pub elsewhere: u64,
    /// Every process's, library applets included.
    pub all_processes: u64,
}

impl Cpu {
    /// The title's process, whatever is running now; the running one if the title has none.
    fn title_process(&self) -> (&AddressSpace, MemoryLayout) {
        match self.machine.processes.get(&self.machine.root) {
            Some(title) => (&title.mem, title.memory_layout),
            None => (&self.mem, self.process().memory_layout),
        }
    }

    /// The title's backed guest memory: [`Cpu::title_memory`]'s total, without the region scan.
    pub fn title_backed(&self) -> u64 {
        self.title_process().0.mapped_bytes()
    }

    pub fn title_memory(&self) -> TitleMemory {
        let (mem, layout) = self.title_process();
        let regions: Vec<_> = Region::ALL
            .iter()
            .map(|&region| {
                let (start, end) = region.range(&layout);
                (region, (start, end), mem.mapped_bytes_in(start, end))
            })
            .collect();
        let backed = mem.mapped_bytes();
        let in_regions: u64 = regions.iter().map(|&(_, _, bytes)| bytes).sum();
        TitleMemory {
            backed,
            cap: mem.max_mapped_bytes(),
            elsewhere: backed.saturating_sub(in_regions),
            regions,
            all_processes: self
                .machine
                .processes
                .values()
                .map(|process| process.mem.mapped_bytes())
                .sum(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_regions_of_every_layout_do_not_overlap() {
        for layout in [
            MemoryLayout::PLAIN,
            MemoryLayout::VIRTUAL_ADDRESS,
            MemoryLayout::APPLET,
        ] {
            let mut ranges: Vec<_> = Region::ALL.iter().map(|r| r.range(&layout)).collect();
            ranges.sort_unstable();
            for pair in ranges.windows(2) {
                assert!(pair[0].1 <= pair[1].0, "{layout:?}: {pair:x?} overlap");
            }
        }
    }

    #[test]
    fn backed_pages_are_counted_in_the_region_they_fall_in() {
        let mut cpu = Cpu::new();
        cpu.bootstrap();
        let page = crate::mem::PAGE_SIZE as u32;
        cpu.mem
            .write_u32(GUEST_STACK_REGION_ADDR + 0x10, 1)
            .unwrap();
        cpu.mem.write_u32(GUEST_HEAP_REGION_ADDR, 1).unwrap();
        cpu.mem.write_u32(GUEST_HEAP_REGION_ADDR + page, 1).unwrap();
        let report = cpu.title_memory();
        let bytes = |wanted| {
            report
                .regions
                .iter()
                .find(|&&(region, _, _)| region == wanted)
                .map(|&(_, _, bytes)| bytes)
                .unwrap()
        };
        assert_eq!(bytes(Region::Stacks), u64::from(page));
        assert_eq!(bytes(Region::Heap), 2 * u64::from(page));
        let in_regions: u64 = report.regions.iter().map(|&(_, _, b)| b).sum();
        assert_eq!(in_regions + report.elsewhere, report.backed);
        assert_eq!(report.all_processes, report.backed);
    }
}
