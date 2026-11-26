//! Notices a relay that stopped talking without closing the socket, as one does when its
//! machine sleeps or the network drops under it.

use switch_relay_proto::ALIVE_SINCE;

/// Longest the relay may send nothing. It sends `ALIVE` every 15 seconds.
const SILENCE_MS: f64 = 45_000.0;

/// A gap this long between checks means the worker was not running, not that the relay was
/// quiet: the frames sent meanwhile are still waiting to be delivered.
const STALL_MS: f64 = 5_000.0;

/// Times are milliseconds from any one clock.
#[derive(Debug, Default)]
pub(crate) struct Watchdog {
    /// Set by a relay that sends `ALIVE`; an older one may be silent for as long as it likes.
    armed: bool,
    heard: f64,
    checked: f64,
}

impl Watchdog {
    /// The relay said hello with `version`.
    pub fn hello(&mut self, version: u8, now: f64) {
        self.armed = version >= ALIVE_SINCE;
        self.heard = now;
        self.checked = now;
    }

    /// Once per slice, with whether any frame came since the last call: whether the relay has
    /// now been silent for too long.
    pub fn silent(&mut self, now: f64, heard: bool) -> bool {
        if !self.armed {
            return false;
        }
        let gap = now - self.checked;
        self.checked = now;
        if gap > STALL_MS {
            self.heard += gap;
        }
        if heard {
            self.heard = now;
        }
        now - self.heard > SILENCE_MS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Check every 16 ms from `start` to `until` with nothing heard: when the relay first
    /// counted as silent.
    fn first_silent(watchdog: &mut Watchdog, start: f64, until: f64) -> Option<f64> {
        let mut now = start;
        while now <= until {
            if watchdog.silent(now, false) {
                return Some(now);
            }
            now += 16.0;
        }
        None
    }

    fn armed() -> Watchdog {
        let mut watchdog = Watchdog::default();
        watchdog.hello(ALIVE_SINCE, 0.0);
        watchdog
    }

    #[test]
    fn a_relay_that_goes_quiet_is_silent_after_the_limit_and_not_before() {
        let mut watchdog = armed();
        let at = first_silent(&mut watchdog, 16.0, 60_000.0).expect("silent");
        assert!((SILENCE_MS..SILENCE_MS + 32.0).contains(&at), "{at}");
    }

    #[test]
    fn any_frame_starts_the_wait_again() {
        let mut watchdog = armed();
        assert_eq!(first_silent(&mut watchdog, 16.0, 40_000.0), None);
        assert!(!watchdog.silent(40_016.0, true));
        assert_eq!(first_silent(&mut watchdog, 40_032.0, 85_000.0), None);
        assert!(first_silent(&mut watchdog, 85_016.0, 90_000.0).is_some());
    }

    #[test]
    fn time_the_worker_did_not_run_is_not_silence() {
        let mut watchdog = armed();
        assert_eq!(first_silent(&mut watchdog, 16.0, 30_000.0), None);
        // A minute in a background tab, then slices again with the relay's frames not yet in.
        assert!(!watchdog.silent(90_000.0, false));
        assert_eq!(first_silent(&mut watchdog, 90_016.0, 104_000.0), None);
        assert!(first_silent(&mut watchdog, 104_016.0, 110_000.0).is_some());
    }

    #[test]
    fn a_relay_older_than_alive_is_never_silent() {
        let mut watchdog = Watchdog::default();
        watchdog.hello(ALIVE_SINCE - 1, 0.0);
        assert_eq!(first_silent(&mut watchdog, 16.0, 200_000.0), None);
    }

    #[test]
    fn nothing_is_silent_before_the_relay_says_hello() {
        let mut watchdog = Watchdog::default();
        assert_eq!(first_silent(&mut watchdog, 0.0, 200_000.0), None);
    }
}
