//! `time`: the console's clocks and calendar arithmetic, derived from the
//! host-set Unix time (`Cpu::set_unix_time`) and `cycles`.

use super::Cpu;
use crate::Result;

/// Days since 1970-01-01 to (year, month, day), after Howard Hinnant's
/// `civil_from_days`.
pub(super) fn civil_from_days(z: i64) -> (i64, u32, u32) {
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
pub(super) fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
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
    pub(super) fn time_request(
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
    pub(super) fn time_system_clock_request(
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
            // -> SystemClockContext { s64 offset; SteadyClockTimePoint timestamp }
            Some(GET_SYSTEM_CLOCK_CONTEXT) => {
                let mut raw = [0u8; 0x20];
                raw[0x08..0x10].copy_from_slice(&self.steady_clock_seconds().to_le_bytes());
                self.write_ipc_response(tls, 0, &[], &raw, &[])
            }
            Some(SET_SYSTEM_CLOCK_CONTEXT) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    pub(super) fn time_steady_clock_request(
        &mut self,
        tls: u32,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        const GET_CURRENT_TIME_POINT: u32 = 0;
        const GET_RTC_VALUE: u32 = 100;
        const IS_RTC_RESET_DETECTED: u32 = 101;
        const GET_SETUP_RESULT_VALUE: u32 = 102;
        match cmd_id {
            // -> SteadyClockTimePoint { s64 value; u8 source_id[0x10] }
            Some(GET_CURRENT_TIME_POINT) => {
                let mut raw = [0u8; 0x18];
                raw[..8].copy_from_slice(&self.steady_clock_seconds().to_le_bytes());
                self.write_ipc_response(tls, 0, &[], &raw, &[])
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

    /// Retired cycles stand in for elapsed time; only monotonicity matters.
    fn steady_clock_seconds(&self) -> i64 {
        (self.cycles / 1_000_000) as i64
    }

    /// `ITimeZoneService`: every conversion is UTC, and the zone name is the
    /// one `set:sys` stores.
    pub(super) fn time_timezone_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
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
                self.write_ipc_response(tls, 0, &[], &Self::to_calendar_time(posix), &[])
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
                );
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
    /// a POSIX time, in UTC.
    pub(super) fn to_calendar_time(posix: i64) -> [u8; 0x20] {
        let days = posix.div_euclid(86400);
        let secs_of_day = posix.rem_euclid(86400);
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
        raw[16..19].copy_from_slice(b"UTC");
        raw
    }

    /// Inverse of [`Cpu::to_calendar_time`], in UTC.
    pub(super) fn from_calendar_time(
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
    use crate::cpu::ipc::testing::*;
    use crate::cpu::Cpu;

    #[test]
    fn civil_days_round_trip_the_epoch_and_a_leap_day() {
        use crate::cpu::time::{civil_from_days, days_from_civil};
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
        let epoch = Cpu::to_calendar_time(0);
        assert_eq!(&epoch[0..2], &1970u16.to_le_bytes()[..]);
        assert_eq!(epoch[2], 1); // month
        assert_eq!(epoch[3], 1); // day
        assert_eq!(epoch[4], 0); // hour
        assert_eq!(epoch[5], 0); // minute
        assert_eq!(epoch[6], 0); // second
        assert_eq!(u32::from_le_bytes(epoch[8..12].try_into().unwrap()), 4); // Thursday
        assert_eq!(u32::from_le_bytes(epoch[12..16].try_into().unwrap()), 0);

        let billion = Cpu::to_calendar_time(1_000_000_000);
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
