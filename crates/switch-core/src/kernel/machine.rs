//! Host-configured machine state.

use crate::cpu::*;
use crate::services::fs;

impl Cpu {
    /// Set the POSIX time (UTC) `time:u`/`time:s` report; the epoch until the host sets it.
    pub fn set_unix_time(&mut self, seconds: i64) {
        self.unix_time = seconds;
    }

    pub fn unix_time(&self) -> i64 {
        self.unix_time
    }

    /// Set the battery `psm` reports; full and charging until the host sets it.
    pub fn set_battery(&mut self, percent: u8, charging: bool) {
        self.battery_percent = percent.min(100);
        self.battery_charging = charging;
    }

    pub fn battery(&self) -> (u8, bool) {
        (self.battery_percent, self.battery_charging)
    }

    /// Set the NACP save quota, passed through as declared.
    pub fn set_save_data_quota(&mut self, quota: fs::SaveDataQuota) {
        self.save_data_quota = quota;
    }

    pub fn save_data_quota(&self) -> fs::SaveDataQuota {
        self.save_data_quota
    }

    /// Set the NPDM `system_resource_size`, which selects the [`MemoryLayout`].
    /// Call before [`Cpu::boot_retail_program`].
    pub fn set_system_resource_size(&mut self, size: u32) {
        self.system_resource_size = size;
        self.refresh_memory_layout();
    }

    /// Re-choose the layout from the program id and system resource size, set in either order.
    fn refresh_memory_layout(&mut self) {
        self.memory_layout = MemoryLayout::for_program(self.program_id, self.system_resource_size);
    }

    pub fn memory_layout(&self) -> MemoryLayout {
        self.memory_layout
    }

    pub fn set_program_id(&mut self, program_id: u64) {
        self.program_id = program_id;
        self.refresh_memory_layout();
    }

    pub fn program_id(&self) -> u64 {
        self.program_id
    }

    /// What a library applet pushed back before exiting, oldest first.
    pub fn library_applet_results(&self) -> &[Vec<u8>] {
        &self.am_out_data
    }

    /// What a library applet pushed through `PushInteractiveOutData`, oldest first.
    pub fn library_applet_interactive_messages(&self) -> &[Vec<u8>] {
        &self.am_interactive_out
    }

    /// Answer the applet through `PopInteractiveInData`, firing its event.
    pub fn push_applet_interactive_in_data(&mut self, data: Vec<u8>) {
        self.am_interactive_in.push_back(data);
        self.refresh_applet_pop_events();
    }

    /// A pseudo-random u64 for `csrng`: splitmix64 seeded from the clock. Not a CSPRNG.
    pub(crate) fn next_random_u64(&mut self) -> u64 {
        if self.rng_state == 0 {
            self.rng_state =
                (self.unix_time as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA076_1D64_78BD_642F;
        }
        self.rng_state = self.rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}
