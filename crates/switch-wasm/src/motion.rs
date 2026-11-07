//! Motion from a Joy-Con or Pro Controller, read over WebHID beside the core.

// Only the wasm build reads the device; host builds compile the parsing for its tests.
#![cfg_attr(not(all(feature = "gpu", target_arch = "wasm32")), allow(dead_code))]

use switch_core::cpu::Cpu;

/// Nintendo's vendor id, and the product ids of the controllers with an IMU.
pub(crate) const VENDOR: u16 = 0x057e;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Left,
    Right,
    Pro,
}

impl Kind {
    fn from_product(id: u16) -> Option<Self> {
        match id {
            0x2006 => Some(Self::Left),
            0x2007 => Some(Self::Right),
            0x2009 => Some(Self::Pro),
            _ => None,
        }
    }

    /// Signs that bring each kind's sensor into one frame.
    fn signs(self) -> ([f32; 3], [f32; 3]) {
        match self {
            Self::Left => ([1.0, -1.0, -1.0], [-1.0, 1.0, 1.0]),
            Self::Right => ([-1.0, -1.0, 1.0], [1.0, 1.0, -1.0]),
            Self::Pro => ([1.0, -1.0, -1.0], [-1.0, 1.0, 1.0]),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Axis {
    offset: f32,
    scale: f32,
}

/// The sensor's offsets and scales, in the report's axis order.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Calibration {
    accel: [Axis; 3],
    gyro: [Axis; 3],
}

impl Default for Calibration {
    fn default() -> Self {
        let axis = |scale| Axis { offset: 0.0, scale };
        Self {
            accel: [axis(16384.0); 3],
            gyro: [axis(15335.0); 3],
        }
    }
}

impl Calibration {
    /// The 0x18 bytes at SPI `0x6020` or `0x8028`: accelerometer offsets and scales, then the
    /// gyroscope's, each three `i16`. An axis whose scale equals its offset keeps the default.
    fn from_spi(raw: &[u8]) -> Self {
        let mut cal = Self::default();
        if raw.len() < 0x18 {
            return cal;
        }
        let word = |i: usize| f32::from(i16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]));
        for (first, axes) in [(0, &mut cal.accel), (6, &mut cal.gyro)] {
            for (i, axis) in axes.iter_mut().enumerate() {
                let (offset, scale) = (word(first + i), word(first + 3 + i));
                if scale != offset {
                    *axis = Axis { offset, scale };
                }
            }
        }
        cal
    }
}

/// Where the IMU samples start in a `0x30` report (the report id excluded), and their size.
const IMU: usize = 12;
const IMU_SAMPLES: usize = 3;
const SAMPLE_BYTES: usize = 12;

/// Acceleration in G and angular velocity in revolutions per second, averaged over the three
/// samples of a report. The report's x and y axes are swapped.
fn sample(report: &[u8], kind: Kind, cal: &Calibration) -> Option<([f32; 3], [f32; 3])> {
    if report.len() < IMU + IMU_SAMPLES * SAMPLE_BYTES {
        return None;
    }
    let (accel_signs, gyro_signs) = kind.signs();
    let mut accel = [0.0; 3];
    let mut gyro = [0.0; 3];
    for s in 0..IMU_SAMPLES {
        let at = IMU + s * SAMPLE_BYTES;
        let raw = |i: usize| {
            f32::from(i16::from_le_bytes([
                report[at + i * 2],
                report[at + i * 2 + 1],
            ]))
        };
        for (i, axis) in [1, 0, 2].into_iter().enumerate() {
            let a = cal.accel[axis];
            let g = cal.gyro[axis];
            accel[i] += raw(axis) / (a.scale - a.offset) * 4.0 * accel_signs[i];
            gyro[i] +=
                (raw(3 + axis) - g.offset) * (936.0 / (g.scale - g.offset)) / 360.0 * gyro_signs[i];
        }
    }
    let n = IMU_SAMPLES as f32;
    Some((accel.map(|v| v / n), gyro.map(|v| v / n)))
}

/// Hand the core a reading, or the loss of the sensor.
pub(crate) fn publish(cpu: &mut Cpu, reading: Option<([f32; 3], [f32; 3])>) {
    let (accel, gyro) = reading.unwrap_or_default();
    cpu.set_motion_state(accel, gyro, reading.is_some());
}

#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
mod device;

#[cfg(test)]
mod tests {
    use super::*;

    fn report(samples: [[i16; 6]; 3]) -> Vec<u8> {
        let mut out = vec![0u8; IMU];
        for sample in samples {
            out.extend(sample.iter().flat_map(|v| v.to_le_bytes()));
        }
        out
    }

    #[test]
    fn a_report_at_rest_reads_one_g() {
        // A left Joy-Con lying flat reports gravity on its z axis.
        let at_rest = [0, 0, 4096, 0, 0, 0];
        let (accel, gyro) =
            sample(&report([at_rest; 3]), Kind::Left, &Calibration::default()).unwrap();
        assert_eq!(accel, [0.0, 0.0, -1.0]);
        assert_eq!(gyro, [0.0; 3]);
    }

    #[test]
    fn x_and_y_swap_and_each_kind_flips_its_own_axes() {
        let moving = [100, 200, 300, 400, 500, 600];
        let accel_of = |kind| {
            sample(&report([moving; 3]), kind, &Calibration::default())
                .unwrap()
                .0
        };
        let g = 4.0 / 16384.0;
        assert_eq!(accel_of(Kind::Left), [200.0 * g, -100.0 * g, -300.0 * g]);
        assert_eq!(accel_of(Kind::Right), [-200.0 * g, -100.0 * g, 300.0 * g]);
    }

    #[test]
    fn the_gyro_is_scaled_to_revolutions_per_second() {
        let spin = [0, 0, 0, 0, 0, 15335];
        let (_, gyro) = sample(&report([spin; 3]), Kind::Left, &Calibration::default()).unwrap();
        assert!((gyro[2] - 936.0 / 360.0).abs() < 1e-5, "{gyro:?}");
    }

    #[test]
    fn a_short_report_is_no_sample() {
        assert!(sample(&[0; 30], Kind::Pro, &Calibration::default()).is_none());
    }

    #[test]
    fn calibration_comes_from_the_spi_words_and_a_blank_axis_keeps_the_default() {
        let mut raw = [0u8; 0x18];
        raw[0..2].copy_from_slice(&(-8i16).to_le_bytes()); // accel x offset
        raw[6..8].copy_from_slice(&(16000i16).to_le_bytes()); // accel x scale
        let cal = Calibration::from_spi(&raw);
        assert_eq!(
            cal.accel[0],
            Axis {
                offset: -8.0,
                scale: 16000.0
            }
        );
        assert_eq!(cal.accel[1], Calibration::default().accel[1]);
        assert_eq!(cal.gyro, Calibration::default().gyro);
    }

    #[test]
    fn only_controllers_with_an_imu_are_recognised() {
        assert_eq!(Kind::from_product(0x2006), Some(Kind::Left));
        assert_eq!(Kind::from_product(0x2009), Some(Kind::Pro));
        assert_eq!(Kind::from_product(0x200e), None);
    }
}
