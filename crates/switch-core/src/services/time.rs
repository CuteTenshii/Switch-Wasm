//! `time`: the console's clocks and calendar arithmetic, derived from the
//! host-set Unix time (`Cpu::set_unix_time`) and `cycles`.

use crate::cpu::Cpu;
use crate::Result;

/// `time`'s shared memory: a steady time point, the local and network clock contexts,
/// the automatic-correction flag and the continuous adjustment, each double-buffered
/// behind a counter whose low bit picks the current copy.
pub(crate) const SHARED_MEMORY_SIZE: usize = 0x1000;
const STEADY_TIME_POINT_AT: u32 = 0x00;
const LOCAL_CONTEXT_AT: u32 = 0x38;
const NETWORK_CONTEXT_AT: u32 = 0x80;
const AUTOMATIC_CORRECTION_AT: u32 = 0xC8;
const CONTINUOUS_ADJUSTMENT_AT: u32 = 0xD0;

/// The steady clock's source id; a context naming another one is from a past boot.
const CLOCK_SOURCE: [u8; 16] = *b"switch-wasm.time";

/// `SteadyClockTimePoint`: a steady-clock value and the clock's source id.
fn steady_time_point(value: i64) -> [u8; 0x18] {
    let mut raw = [0u8; 0x18];
    raw[..8].copy_from_slice(&value.to_le_bytes());
    raw[8..].copy_from_slice(&CLOCK_SOURCE);
    raw
}

/// A zone name for an offset that fits `CalendarAdditionalInfo`: `UTC`, `UTC+2`, `UTC-0930`.
fn zone_name(utc_offset: i32) -> String {
    let sign = if utc_offset < 0 { '-' } else { '+' };
    let (hours, minutes) = (utc_offset.abs() / 3600, utc_offset.abs() / 60 % 60);
    match (hours, minutes) {
        (0, 0) => "UTC".to_owned(),
        (_, 0) => format!("UTC{sign}{hours}"),
        _ => format!("UTC{sign}{hours:02}{minutes:02}"),
    }
}

/// Days since 1970-01-01 to (year, month, day), after Howard Hinnant's
/// `civil_from_days`.
pub(crate) fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097); // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if m <= 2 { y + 1 } else { y };
    (year, m as u32, d as u32)
}

/// Inverse of [`civil_from_days`].
pub(crate) fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let (m, d) = (m as i64, d as i64);
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400); // [0, 399]
    let mp = if m > 2 { m - 3 } else { m + 9 }; // [0, 11]
    let doy = (153 * mp + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468
}

impl Cpu {
    /// `ITimeServiceManager` (`time:s`/`time:u`/`time:a`/`time:r`).
    ///
    /// Its command ids overlap `ConvertToDomain`/`QueryPointerBufferSize`, so
    /// Control requests must be checked first.
    pub(crate) fn time_request(
        &mut self,
        tls: u32,
        cmd_id: Option<u32>,
        handle: u64,
    ) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "time");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
            };
        }
        const GET_STANDARD_USER_SYSTEM_CLOCK: u32 = 0;
        const GET_STANDARD_NETWORK_SYSTEM_CLOCK: u32 = 1;
        const GET_STANDARD_STEADY_CLOCK: u32 = 2;
        const GET_TIME_ZONE_SERVICE: u32 = 3;
        const GET_STANDARD_LOCAL_SYSTEM_CLOCK: u32 = 4;
        const GET_SHARED_MEMORY_NATIVE_HANDLE: u32 = 20;
        const GET_STANDARD_STEADY_CLOCK_RTC_VALUE: u32 = 51;
        const IS_STANDARD_USER_SYSTEM_CLOCK_AUTOMATIC_CORRECTION_ENABLED: u32 = 100;
        match cmd_id {
            // All three system clocks are the same clock.
            Some(GET_STANDARD_USER_SYSTEM_CLOCK)
            | Some(GET_STANDARD_NETWORK_SYSTEM_CLOCK)
            | Some(GET_STANDARD_LOCAL_SYSTEM_CLOCK) => {
                self.reply_with_interface(tls, handle, "time:system-clock")?;
                Ok(())
            }
            Some(GET_STANDARD_STEADY_CLOCK) => {
                self.reply_with_interface(tls, handle, "time:steady-clock")?;
                Ok(())
            }
            Some(GET_TIME_ZONE_SERVICE) => {
                self.reply_with_interface(tls, handle, "time:timezone")?;
                Ok(())
            }
            Some(GET_SHARED_MEMORY_NATIVE_HANDLE) => {
                let shmem = self.alloc_handle();
                self.process_mut().time_shmem_handle = Some(shmem);
                self.write_ipc_reply(tls, 0, &[shmem], &[], &[], &[])
            }
            Some(GET_STANDARD_STEADY_CLOCK_RTC_VALUE) => self.write_ipc_response(
                tls,
                0,
                &[],
                &(self.steady_clock_seconds() as u64).to_le_bytes(),
                &[],
            ),
            // Always "corrected": the host pushes wall-clock time.
            Some(IS_STANDARD_USER_SYSTEM_CLOCK_AUTOMATIC_CORRECTION_ENABLED) => {
                self.write_ipc_response(tls, 0, &[], &[1u8], &[])
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// `ISystemClock`: POSIX seconds from [`Cpu::set_unix_time`]; setters are
    /// accepted but have no effect.
    pub(crate) fn time_system_clock_request(
        &mut self,
        tls: u32,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        const GET_CURRENT_TIME: u32 = 0;
        const SET_CURRENT_TIME: u32 = 1;
        const GET_SYSTEM_CLOCK_CONTEXT: u32 = 2;
        const SET_SYSTEM_CLOCK_CONTEXT: u32 = 3;
        match cmd_id {
            Some(GET_CURRENT_TIME) => {
                let posix = self.unix_time();
                self.write_ipc_response(tls, 0, &[], &posix.to_le_bytes(), &[])
            }
            Some(SET_CURRENT_TIME) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            Some(GET_SYSTEM_CLOCK_CONTEXT) => {
                let context = self.system_clock_context();
                self.write_ipc_response(tls, 0, &[], &context, &[])
            }
            Some(SET_SYSTEM_CLOCK_CONTEXT) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    pub(crate) fn time_steady_clock_request(
        &mut self,
        tls: u32,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        const GET_CURRENT_TIME_POINT: u32 = 0;
        const GET_RTC_VALUE: u32 = 100;
        const IS_RTC_RESET_DETECTED: u32 = 101;
        const GET_SETUP_RESULT_VALUE: u32 = 102;
        match cmd_id {
            Some(GET_CURRENT_TIME_POINT) => {
                let point = steady_time_point(self.steady_clock_seconds());
                self.write_ipc_response(tls, 0, &[], &point, &[])
            }
            Some(GET_RTC_VALUE) => self.write_ipc_response(
                tls,
                0,
                &[],
                &(self.steady_clock_seconds() as u64).to_le_bytes(),
                &[],
            ),
            Some(IS_RTC_RESET_DETECTED) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
            Some(GET_SETUP_RESULT_VALUE) => {
                self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[])
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// Seconds since boot by the guest's own clock, the 19.2 MHz tick.
    fn steady_clock_seconds(&self) -> i64 {
        (self.system_tick() / 19_200_000) as i64
    }

    /// `SystemClockContext`: the offset that turns the steady clock into the host's time.
    fn system_clock_context(&self) -> [u8; 0x20] {
        let steady = self.steady_clock_seconds();
        let mut raw = [0u8; 0x20];
        raw[..8].copy_from_slice(&(self.unix_time - steady).to_le_bytes());
        raw[8..].copy_from_slice(&steady_time_point(steady));
        raw
    }

    /// Bring `time`'s shared memory up to the host's time, as each slice starts.
    pub(crate) fn publish_time(&mut self) {
        let block = &self.time_shmem;
        // Copy `value` into the slot after the current one, then make it current.
        let write = |at: u32, align: u32, value: &[u8]| {
            let counter = block.read_u32(at).unwrap_or(0).wrapping_add(1);
            let slot = at + align + (counter % 2) * value.len() as u32;
            let _ = block.write(slot, value);
            let _ = block.write_u32(at, counter);
        };
        // Boot is the steady clock's zero, in nanoseconds.
        write(STEADY_TIME_POINT_AT, 8, &steady_time_point(0));
        let context = self.system_clock_context();
        write(LOCAL_CONTEXT_AT, 8, &context);
        write(NETWORK_CONTEXT_AT, 8, &context);
        write(AUTOMATIC_CORRECTION_AT, 4, &[1]);
        // `ContinuousAdjustmentTimePoint`: no drift to correct, so only the source id.
        let mut adjustment = [0u8; 0x38];
        adjustment[0x28..].copy_from_slice(&CLOCK_SOURCE);
        write(CONTINUOUS_ADJUSTMENT_AT, 8, &adjustment);
    }

    /// `ITimeZoneService`: every conversion uses the host's current offset from UTC (no
    /// DST history), and the zone name is the one `set:sys` stores.
    pub(crate) fn time_timezone_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        const LOCATION_NAME: &[u8] = super::settings::DEVICE_TIME_ZONE;
        const GET_DEVICE_LOCATION_NAME: u32 = 0;
        const GET_TOTAL_LOCATION_NAME_COUNT: u32 = 2;
        const LOAD_LOCATION_NAME_LIST: u32 = 3;
        const LOAD_TIME_ZONE_RULE: u32 = 4;
        const TO_CALENDAR_TIME: u32 = 100;
        const TO_CALENDAR_TIME_WITH_MY_RULE: u32 = 101;
        const TO_POSIX_TIME: u32 = 201;
        const TO_POSIX_TIME_WITH_MY_RULE: u32 = 202;
        match cmd_id {
            Some(GET_DEVICE_LOCATION_NAME) => {
                let raw = self.system_settings().device_time_zone_location_name;
                self.write_ipc_response(tls, 0, &[], &raw, &[])
            }
            Some(GET_TOTAL_LOCATION_NAME_COUNT) => {
                self.write_ipc_response(tls, 0, &[], &1u32.to_le_bytes(), &[])
            }
            Some(LOAD_LOCATION_NAME_LIST) => {
                if let Some(&(addr, size)) = self.ipc_buffers(tls).1.first() {
                    if size >= LOCATION_NAME.len() as u32 {
                        for (i, &b) in LOCATION_NAME.iter().enumerate() {
                            self.mem.write_u8(addr.wrapping_add(i as u32), b)?;
                        }
                    }
                }
                self.write_ipc_response(tls, 0, &[], &1u32.to_le_bytes(), &[])
            }
            // The rule is never used, since conversions are always UTC.
            Some(LOAD_TIME_ZONE_RULE) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            Some(TO_CALENDAR_TIME) | Some(TO_CALENDAR_TIME_WITH_MY_RULE) => {
                let posix = self.mem.read_u64(self.ipc_request_data(tls)).unwrap_or(0) as i64;
                let calendar = Self::to_calendar_time(posix, self.utc_offset);
                self.write_ipc_response(tls, 0, &[], &calendar, &[])
            }
            // Always exactly one match: there is no DST.
            Some(TO_POSIX_TIME) | Some(TO_POSIX_TIME_WITH_MY_RULE) => {
                let data = self.ipc_request_data(tls);
                let posix = Self::from_calendar_time(
                    self.mem.read_u16(data).unwrap_or(1970),
                    self.mem.read_u8(data.wrapping_add(2)).unwrap_or(1),
                    self.mem.read_u8(data.wrapping_add(3)).unwrap_or(1),
                    self.mem.read_u8(data.wrapping_add(4)).unwrap_or(0),
                    self.mem.read_u8(data.wrapping_add(5)).unwrap_or(0),
                    self.mem.read_u8(data.wrapping_add(6)).unwrap_or(0),
                ) - i64::from(self.utc_offset);
                if let Some(&(addr, size)) = self.ipc_buffers(tls).1.first() {
                    if size >= 8 {
                        self.mem.write_u64(addr, posix as u64)?;
                    }
                }
                self.write_ipc_response(tls, 0, &[], &1u32.to_le_bytes(), &[])
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// `CalendarTime` (8 bytes) then `CalendarAdditionalInfo` (0x18 bytes) for
    /// a POSIX time, `utc_offset` seconds east of UTC.
    pub(crate) fn to_calendar_time(posix: i64, utc_offset: i32) -> [u8; 0x20] {
        let local = posix + i64::from(utc_offset);
        let days = local.div_euclid(86400);
        let secs_of_day = local.rem_euclid(86400);
        let (year, month, day) = civil_from_days(days);
        let day_of_week = (days + 4).rem_euclid(7); // 1970-01-01 was a Thursday
        let day_of_year = days - days_from_civil(year, 1, 1);

        let mut raw = [0u8; 0x20];
        raw[0..2].copy_from_slice(&(year.clamp(0, u16::MAX as i64) as u16).to_le_bytes());
        raw[2] = month as u8;
        raw[3] = day as u8;
        raw[4] = (secs_of_day / 3600) as u8;
        raw[5] = ((secs_of_day / 60) % 60) as u8;
        raw[6] = (secs_of_day % 60) as u8;
        raw[8..12].copy_from_slice(&(day_of_week as u32).to_le_bytes());
        raw[12..16].copy_from_slice(&(day_of_year as u32).to_le_bytes());
        // libnx builds its `TZ` from this name, and an empty one reads as UTC.
        let name = zone_name(utc_offset);
        raw[16..16 + name.len()].copy_from_slice(name.as_bytes());
        raw[28..32].copy_from_slice(&utc_offset.to_le_bytes());
        raw
    }

    /// The POSIX time of a calendar time in UTC.
    pub(crate) fn from_calendar_time(
        year: u16,
        month: u8,
        day: u8,
        hour: u8,
        minute: u8,
        second: u8,
    ) -> i64 {
        let days = days_from_civil(year as i64, month.max(1) as u32, day.max(1) as u32);
        days * 86400 + hour as i64 * 3600 + minute as i64 * 60 + second as i64
    }
}

#[cfg(test)]
mod tests {
    use crate::cpu::Cpu;
    use crate::kernel::ipc::testing::*;

    /// What a client reads: the current copy of a double-buffered value.
    fn read_current(cpu: &Cpu, at: u32, len: u32) -> Vec<u8> {
        let counter = cpu.mem.read_u32(at).unwrap();
        cpu.read_bytes(at + 8 + (counter % 2) * len, len)
    }

    #[test]
    fn the_shared_memory_clock_reads_the_host_time_as_it_moves() {
        const SHMEM: u32 = 0x3000_0000;
        let mut cpu = request(false, 20, &[]);
        cpu.register_service_handle(9, "time:u");
        cpu.time_request(TLS, Some(20), 9).unwrap();
        let handle = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
        assert_eq!(cpu.process().time_shmem_handle, Some(handle));
        let block = cpu.time_shmem.clone();
        cpu.mem.map_shared(SHMEM, &block).unwrap();

        // Offset plus the steady clock is the time; the steady clock moves with the tick.
        let now = |cpu: &Cpu| {
            let context = read_current(cpu, SHMEM + super::LOCAL_CONTEXT_AT, 0x20);
            let offset = i64::from_le_bytes(context[..8].try_into().unwrap());
            assert_eq!(
                &context[0x10..0x20],
                &super::CLOCK_SOURCE,
                "not this boot's clock"
            );
            offset + cpu.steady_clock_seconds()
        };
        cpu.set_unix_time(1_700_000_000);
        assert_eq!(now(&cpu), 1_700_000_000);
        cpu.cycles += 5 * 1_020_000_000;
        cpu.set_unix_time(1_700_000_007);
        assert_eq!(now(&cpu), 1_700_000_007);
        let steady = read_current(&cpu, SHMEM + super::STEADY_TIME_POINT_AT, 0x18);
        assert_eq!(&steady[8..], &super::CLOCK_SOURCE);
    }

    #[test]
    fn civil_days_round_trip_the_epoch_and_a_leap_day() {
        use crate::services::time::{civil_from_days, days_from_civil};
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        for &(y, m, d) in &[
            (1969, 12, 31), // just before the epoch
            (2024, 2, 29),  // a leap day
            (2001, 9, 9),   // the "1 billion seconds" date
            (1900, 1, 1),   // not a leap year despite ending in 00
            (2000, 2, 29),  // is a leap year (divisible by 400)
        ] {
            let days = days_from_civil(y, m, d);
            assert_eq!(
                civil_from_days(days),
                (y, m, d),
                "{y}-{m}-{d} (days={days})"
            );
        }
    }

    #[test]
    fn to_calendar_time_matches_the_epoch_and_a_known_date() {
        let epoch = Cpu::to_calendar_time(0, 0);
        assert_eq!(&epoch[0..2], &1970u16.to_le_bytes()[..]);
        assert_eq!(epoch[2], 1); // month
        assert_eq!(epoch[3], 1); // day
        assert_eq!(epoch[4], 0); // hour
        assert_eq!(epoch[5], 0); // minute
        assert_eq!(epoch[6], 0); // second
        assert_eq!(u32::from_le_bytes(epoch[8..12].try_into().unwrap()), 4); // Thursday
        assert_eq!(u32::from_le_bytes(epoch[12..16].try_into().unwrap()), 0);

        let east = Cpu::to_calendar_time(0, 2 * 3600);
        assert_eq!(east[4], 2, "the local hour is two hours east of UTC");
        assert_eq!(i32::from_le_bytes(east[28..32].try_into().unwrap()), 7200);
        assert_eq!(
            &east[16..21],
            b"UTC+2",
            "libnx reads an unnamed zone as UTC"
        );
        assert_eq!(super::zone_name(-(9 * 3600 + 30 * 60)), "UTC-0930");

        let billion = Cpu::to_calendar_time(1_000_000_000, 0);
        assert_eq!(&billion[0..2], &2001u16.to_le_bytes()[..]);
        assert_eq!(billion[2], 9);
        assert_eq!(billion[3], 9);
        assert_eq!(billion[4], 1);
        assert_eq!(billion[5], 46);
        assert_eq!(billion[6], 40);

        assert_eq!(
            Cpu::from_calendar_time(2001, 9, 9, 1, 46, 40),
            1_000_000_000
        );
    }

    #[test]
    fn system_clock_get_current_time_reports_the_host_supplied_value() {
        let mut cpu = request(false, 0, &[]);
        cpu.set_unix_time(1_700_000_000);
        cpu.time_system_clock_request(TLS, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap() as i64, 1_700_000_000);
    }

    #[test]
    fn timezone_service_converts_posix_time_to_calendar_time_over_ipc() {
        let mut cpu = request(false, 101, &1_000_000_000i64.to_le_bytes());
        cpu.time_timezone_request(TLS, Some(101)).unwrap();
        assert_eq!(cpu.mem.read_u16(TLS + 0x20).unwrap(), 2001);
        assert_eq!(cpu.mem.read_u8(TLS + 0x22).unwrap(), 9); // month
        assert_eq!(cpu.mem.read_u8(TLS + 0x23).unwrap(), 9); // day
    }
}
