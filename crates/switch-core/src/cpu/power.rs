//! The power, clock and sensor services: `psm` (the battery), `apm`
//! (performance mode), `pcv`/`clkrst` (module clock rates), `ts` (the
//! temperature sensors), `psc` (power-state change notification) and `gpio`.
//!
//! [`CLOCK_RATES_HZ`] lives here and is load-bearing well beyond this module:
//! one emulated instruction stands for one cycle of its first entry, which is
//! the rate the display tick, the thread deadlines and both audio clocks are
//! counted in.

use super::Cpu;
use crate::Result;

/// The rate each clock module runs at, in Hz: CPU, GPU, memory, then every
/// module this does not model, which runs at nothing. These are an original
/// console's **handheld** rates, matching the operation mode `am` reports and
/// the Normal performance mode `apm` does — a docked console runs its GPU at
/// 768 MHz instead, and claiming that while presenting a 720p handheld
/// framebuffer would be two answers to the same question.
pub(super) const CLOCK_RATES_HZ: [u32; 4] = [1_020_000_000, 384_000_000, 1_600_000_000, 0];

/// The temperatures `ts` reports, in degrees Celsius: the SoC
/// (`TsLocation_Internal`) first, the PCB (`TsLocation_External`) second.
///
/// Fixed, and deliberately an idle console's: nothing this emulator runs makes
/// silicon warm, so an idle reading is the true state rather than a
/// placeholder for one that could not be taken.
const TS_TEMPERATURE_C: [i32; 2] = [40, 35];

/// The range `ts` says its sensors report over. Both readings above sit inside
/// it — a caller that scales a gauge by this range would otherwise draw the
/// needle off the end.
const TS_TEMPERATURE_RANGE_C: (i32, i32) = (0, 100);

/// `ApmPerformanceMode_Normal`: the handheld clock profile, and the mode
/// `am`'s `ICommonStateGetter::GetPerformanceMode` already reports. Boost (1)
/// is the docked one.
const APM_PERFORMANCE_MODE_NORMAL: u32 = 0;

/// The `ApmPerformanceConfiguration` each performance mode starts at, indexed
/// by mode.
///
/// Nothing here derives a clock from these — no CPU, GPU or memory frequency
/// in this emulator is settable — but they cannot be zero, which is
/// `ApmPerformanceConfiguration_Invalid`, and whatever a title sets has to
/// read back unchanged.
pub(super) const APM_DEFAULT_CONFIGURATION: [u32; 2] = [0x0001_0000, 0x0002_0000];

impl Cpu {
    /// `psm` (Power State Management): the battery. Its own commands share
    /// command ids with `ConvertToDomain`/`QueryPointerBufferSize` the same
    /// way `time_request` does, so the control path is checked first.
    pub(super) fn psm_request(&mut self, tls: u32, cmd_id: Option<u32>, handle: u64) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        const QUERY_POINTER_BUFFER_SIZE: u32 = 3;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "psm");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                Some(QUERY_POINTER_BUFFER_SIZE) => {
                    self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[])
                }
                _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
            };
        }
        const GET_BATTERY_CHARGE_PERCENTAGE: u32 = 0;
        const GET_CHARGER_TYPE: u32 = 1;
        const ENABLE_BATTERY_CHARGING: u32 = 2;
        const DISABLE_BATTERY_CHARGING: u32 = 3;
        const IS_BATTERY_CHARGING_ENABLED: u32 = 4;
        const OPEN_SESSION: u32 = 7;
        // ChargerType: 0 Unconnected, 1 EnoughPower, 2 LowPower, 3 NotSupported.
        // The Battery Status API (where the host exposes one) only reports a
        // charging bool, not the power level a real charger negotiates, so
        // charging maps to EnoughPower and not charging to Unconnected.
        const CHARGER_UNCONNECTED: u32 = 0;
        const CHARGER_ENOUGH_POWER: u32 = 1;
        match cmd_id {
            Some(GET_BATTERY_CHARGE_PERCENTAGE) => {
                let (percent, _) = self.battery();
                self.write_ipc_response(tls, 0, &[], &(percent as u32).to_le_bytes(), &[])
            }
            Some(GET_CHARGER_TYPE) => {
                let (_, charging) = self.battery();
                let charger = if charging { CHARGER_ENOUGH_POWER } else { CHARGER_UNCONNECTED };
                self.write_ipc_response(tls, 0, &[], &charger.to_le_bytes(), &[])
            }
            // EnableBatteryCharging/DisableBatteryCharging: accepted, but
            // charging state here mirrors the host battery, not a guest
            // setting — there is nothing to actually stop charging.
            Some(ENABLE_BATTERY_CHARGING) | Some(DISABLE_BATTERY_CHARGING) => {
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(IS_BATTERY_CHARGING_ENABLED) => {
                self.write_ipc_response(tls, 0, &[], &[1u8], &[])
            }
            Some(OPEN_SESSION) => {
                self.reply_with_interface(tls, handle, "psm-session")?;
                Ok(())
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// `IPsmSession`: the live charger/battery-state-change notifications a
    /// caller can subscribe to. There is no push channel from the host
    /// battery here — [`Cpu::set_battery`] is polled, the way
    /// `GetBatteryChargePercentage` already is — so the bound event is
    /// handed out but never signalled; a caller has to keep polling rather
    /// than wait on it.
    pub(super) fn psm_session_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        const BIND_STATE_CHANGE_EVENT: u32 = 0;
        const UNBIND_STATE_CHANGE_EVENT: u32 = 1;
        const SET_CHARGER_TYPE_CHANGE_EVENT_ENABLED: u32 = 2;
        const SET_POWER_SUPPLY_CHANGE_EVENT_ENABLED: u32 = 3;
        const SET_BATTERY_VOLTAGE_STATE_CHANGE_EVENT_ENABLED: u32 = 4;
        match cmd_id {
            Some(BIND_STATE_CHANGE_EVENT) => {
                let event = self.alloc_event("psm:state-change", true);
                self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
            }
            Some(UNBIND_STATE_CHANGE_EVENT)
            | Some(SET_CHARGER_TYPE_CHANGE_EVENT_ENABLED)
            | Some(SET_POWER_SUPPLY_CHANGE_EVENT_ENABLED)
            | Some(SET_BATTERY_VOLTAGE_STATE_CHANGE_EVENT_ENABLED) => {
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// `clkrst` (`IClkrstManager`) and `pcv`, the same clock-and-voltage
    /// manager either side of 8.0.0: what rate each hardware module runs at.
    ///
    /// Nothing here is clocked — the CPU is an interpreter and the GPU a
    /// software rasterizer — so these report the rates an idle console in
    /// handheld mode runs at, which is the mode `am` and `apm` already agree
    /// this console is in. A rate a guest *sets* is stored and read back:
    /// that pair has to agree even when neither value drives anything, since
    /// a caller that reads back a different rate concludes its request failed.
    ///
    /// The old `pcv` interface takes a small module enum and the newer
    /// `clkrst` a device code; both end up in [`Cpu::clkrst_module`], which is
    /// where the two numbering schemes are reconciled.
    pub(super) fn pcv_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[]);
        }
        let data = self.ipc_request_data(tls);
        let iface = self.service_name(handle).unwrap_or("pcv").to_string();
        if iface == "clkrst" {
            return match cmd_id {
                // IClkrstManager::OpenSession(u32 device_code, u32 unk) ->
                // IClkrstSession. Which module the session is for rides in
                // the interface name, the way `ts`'s sensor does.
                Some(0) => {
                    let module = self.clkrst_module(self.mem.read_u32(data).unwrap_or(0));
                    let name = Self::clkrst_session_name(module);
                    self.reply_with_interface(tls, handle, name)?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            };
        }
        if let Some(module) = iface.strip_prefix("clkrst:session-") {
            let module = module.parse::<u32>().unwrap_or(0);
            return match cmd_id {
                // IClkrstSession::SetClockRate(u32 hz) / GetClockRate -> hz.
                Some(7) => {
                    let rate = self.mem.read_u32(data).unwrap_or(0);
                    self.clock_rates.insert(module, rate);
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                Some(8) => {
                    let rate = self.clock_rate(module);
                    self.write_ipc_response(tls, 0, &[], &rate.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            };
        }
        // `pcv`, where the module is an argument rather than a session.
        match cmd_id {
            // SetClockRate(PcvModule, u32 hz) / GetClockRate(PcvModule) -> hz.
            Some(2) => {
                let module = self.clkrst_module(self.mem.read_u32(data).unwrap_or(0));
                let rate = self.mem.read_u32(data.wrapping_add(4)).unwrap_or(0);
                self.clock_rates.insert(module, rate);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(3) => {
                let module = self.clkrst_module(self.mem.read_u32(data).unwrap_or(0));
                let rate = self.clock_rate(module);
                self.write_ipc_response(tls, 0, &[], &rate.to_le_bytes(), &[])
            }
            // SetPowerEnabled / SetClockEnabled and their disables: there is
            // no rail to switch.
            Some(0) | Some(1) | Some(4) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }

    /// Reconcile `pcv`'s module enum and `clkrst`'s device codes into one
    /// index into [`CLOCK_RATES_HZ`].
    ///
    /// The two number the same hardware differently, and by an offset rather
    /// than a rename: a `clkrst` device code is `0x40000000 + module + 1`,
    /// where `module` is the `PcvModule` value `pcv` takes directly. NX-Fetch
    /// asks `clkrst` for `0x40000001`, `0x40000002` and `0x40000039` and
    /// labels the answers CPU, GPU and Memory — so those are `PcvModule`s 0
    /// (CpuBus), 1 (GPU) and 0x38 (EMC), and reading the code's low bits as
    /// the module directly is off by one.
    ///
    /// A module this does not model reports its own entry, which is 0 Hz —
    /// "not running" — rather than another module's rate.
    fn clkrst_module(&self, code: u32) -> u32 {
        const PCV_MODULE_CPU_BUS: u32 = 0;
        const PCV_MODULE_GPU: u32 = 1;
        const PCV_MODULE_EMC: u32 = 0x38;
        let module = if code >= 0x4000_0000 { (code & 0xFF).wrapping_sub(1) } else { code };
        match module {
            PCV_MODULE_CPU_BUS => 0,
            PCV_MODULE_GPU => 1,
            PCV_MODULE_EMC => 2,
            _ => 3,
        }
    }

    /// The rate a module runs at: whatever was last set for it, else the
    /// default for the mode the console is in.
    ///
    /// Only the GPU's default moves with the dock. The CPU and memory clocks
    /// are the same either way on an original console, and the CPU's is
    /// load-bearing beyond this — it is the rate one emulated instruction
    /// stands for, and what `GetSystemTick` and every timed wait are derived
    /// from — so it is not a figure the dock gets to change.
    fn clock_rate(&self, module: u32) -> u32 {
        const GPU: u32 = 1;
        match self.clock_rates.get(&module) {
            Some(&rate) => rate,
            None if module == GPU => self.operation_mode().gpu_clock_hz(),
            None => CLOCK_RATES_HZ[module as usize],
        }
    }

    /// The interface name a `clkrst` session for `module` is filed under.
    fn clkrst_session_name(module: u32) -> &'static str {
        match module {
            1 => "clkrst:session-1",
            2 => "clkrst:session-2",
            3 => "clkrst:session-3",
            _ => "clkrst:session-0",
        }
    }

    /// `ts` (`IMeasurementServer`): the console's thermometers.
    ///
    /// Real hardware has two — one on the SoC die (`TsLocation_Internal`) and
    /// one on the PCB beside it (`TsLocation_External`) — and system-info
    /// homebrew puts their readings on screen. There is no silicon here to be
    /// warm, so both report a **fixed idle temperature**, which is the true
    /// state of a console that is not dissipating anything rather than a
    /// number standing in for one that could not be read.
    ///
    /// The two commands that report the same measurement in different units
    /// have to agree — `GetTemperatureMilliC` is `GetTemperature` times a
    /// thousand — and the reading has to sit inside the range
    /// `GetTemperatureRange` reports, or a caller that scales a gauge by that
    /// range draws it off the end.
    pub(super) fn ts_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        const QUERY_POINTER_BUFFER_SIZE: u32 = 3;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "ts");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                Some(QUERY_POINTER_BUFFER_SIZE) => {
                    self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, "ts:control", cmd_id),
            };
        }
        let object_id = self.ipc_domain_object_id(tls);
        // A session reached over either route — its own handle, or an object
        // id on a domain — is a different interface from the server.
        let iface = if self.ipc_is_domain_request(tls) {
            self.domain_interface(handle, object_id).unwrap_or("ts").to_string()
        } else {
            self.service_name(handle).unwrap_or("ts").to_string()
        };
        if iface.starts_with("ts:session") {
            return self.ts_session_request(tls, &iface, cmd_id);
        }
        // The location is a single byte of the payload: 0 = Internal (the
        // SoC), 1 = External (the PCB). Anything else reads as Internal.
        let location = self.mem.read_u8(self.ipc_request_data(tls)).unwrap_or(0);
        let celsius = TS_TEMPERATURE_C[usize::from(location).min(TS_TEMPERATURE_C.len() - 1)];
        match cmd_id {
            // GetTemperatureRange(TsLocation) -> (s32 min, s32 max): the
            // range the sensor can report over, not today's weather.
            Some(0) => {
                let mut range = [0u8; 8];
                range[..4].copy_from_slice(&TS_TEMPERATURE_RANGE_C.0.to_le_bytes());
                range[4..].copy_from_slice(&TS_TEMPERATURE_RANGE_C.1.to_le_bytes());
                self.write_ipc_response(tls, 0, &[], &range, &[])
            }
            // GetTemperature(TsLocation) -> s32 degrees Celsius.
            Some(1) => self.write_ipc_response(tls, 0, &[], &celsius.to_le_bytes(), &[]),
            // SetMeasurementMode(TsLocation, TsMeasurementMode): how often the
            // sensor is sampled. Nothing here samples anything, and the
            // reading does not vary with the mode.
            Some(2) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // GetTemperatureMilliC(TsLocation) -> s32 millidegrees.
            Some(3) => {
                let milli = celsius * 1000;
                self.write_ipc_response(tls, 0, &[], &milli.to_le_bytes(), &[])
            }
            // OpenSession(u32 device_code) -> ISession, the per-device
            // interface later firmware moved the measurement onto.
            //
            // Which sensor the session is for rides in the interface name
            // rather than in a side table, and the two names route straight
            // back here.
            //
            // The device code's **high byte** is what separates them —
            // `0x41……` is the SoC and `0x43……` the PCB — not its low byte,
            // which varies between the codes a guest may use for the same
            // sensor: NX-Fetch asks for `0x41000002` and labels what comes
            // back "CPU", so reading the low byte made it print the PCB's
            // temperature under the SoC's name.
            Some(4) => {
                let device_code = self.mem.read_u32(self.ipc_request_data(tls)).unwrap_or(0);
                let name = match device_code >> 24 {
                    0x43 => "ts:session-external",
                    _ => "ts:session-internal",
                };
                self.reply_with_interface(tls, handle, name)?;
                Ok(())
            }
            _ => self.unimplemented_command(tls, "ts", cmd_id),
        }
    }

    /// `ISession`, the per-sensor interface `ts::OpenSession` hands out.
    ///
    /// Its `GetTemperature` is **command 4 and reports a `float`**, where the
    /// server's own command 4 is `OpenSession` and its temperature commands
    /// report integers. Sharing one dispatch between the two therefore
    /// answered a session's temperature request with another session object,
    /// and NX-Fetch drew whatever the first word of that reply happened to be
    /// as the console's temperature.
    pub(super) fn ts_session_request(
        &mut self,
        tls: u32,
        iface: &str,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        let celsius = match iface {
            "ts:session-external" => TS_TEMPERATURE_C[1],
            _ => TS_TEMPERATURE_C[0],
        };
        match cmd_id {
            // GetTemperature -> f32 degrees Celsius.
            Some(4) => {
                let reading = celsius as f32;
                self.write_ipc_response(tls, 0, &[], &reading.to_le_bytes(), &[])
            }
            _ => self.unimplemented_command(tls, iface, cmd_id),
        }
    }

    /// `apm` (`IManager`) and `apm:sys` (`ISystemManager`): performance
    /// management — which clock profile the console runs at.
    ///
    /// There is nothing to clock here. The CPU is an interpreter, the GPU is a
    /// software rasterizer, and neither runs faster because a title asked for
    /// the docked profile. What `apm` still has to do is be *consistent*: it
    /// reports the same performance mode `am`'s `ICommonStateGetter` does
    /// (Normal — this console is handheld), and a configuration it was told to
    /// set is the configuration it reports back. A title that sets a profile
    /// and reads back a different one concludes the request failed.
    ///
    /// `apm` is opened by more or less everything: `libnx`'s `apmInitialize`
    /// runs from `__appInit`, so JKSV asks for it before it draws anything.
    pub(super) fn apm_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        const QUERY_POINTER_BUFFER_SIZE: u32 = 3;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let name = self.service_name(handle).unwrap_or("apm").to_string();
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, &name);
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                Some(QUERY_POINTER_BUFFER_SIZE) => {
                    self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, "apm:control", cmd_id),
            };
        }
        let object_id = self.ipc_domain_object_id(tls);
        let iface = if self.ipc_is_domain_request(tls) {
            self.domain_interface(handle, object_id).unwrap_or("apm").to_string()
        } else {
            match self.service_name(handle) {
                Some(name) => name.to_string(),
                None => "apm".to_string(),
            }
        };
        let data = self.ipc_request_data(tls);
        match iface.as_str() {
            // IManager. `apm:p` and `apm:am` are the same interface at higher
            // privilege; nothing here distinguishes them.
            "apm" | "apm:p" | "apm:am" => match cmd_id {
                // OpenSession -> ISession.
                Some(0) => {
                    self.reply_with_interface(tls, handle, "apm:session")?;
                    Ok(())
                }
                // GetPerformanceMode -> ApmPerformanceMode. The same answer
                // `am`'s GetPerformanceMode gives, from the same switch: two
                // services disagreeing about whether the console is docked is
                // worse than either answer on its own.
                Some(1) => {
                    let mode = self.operation_mode().performance_mode();
                    self.write_ipc_response(tls, 0, &[], &mode.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ISession.
            "apm:session" => match cmd_id {
                // SetPerformanceConfiguration(ApmPerformanceMode,
                // ApmPerformanceConfiguration): remembered per mode, because
                // command 1 has to give it back.
                Some(0) => {
                    let mode = self.mem.read_u32(data).unwrap_or(0);
                    let configuration = self.mem.read_u32(data.wrapping_add(4)).unwrap_or(0);
                    if let Some(slot) = self.apm_configuration.get_mut(mode as usize) {
                        *slot = configuration;
                    }
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetPerformanceConfiguration(ApmPerformanceMode) ->
                // ApmPerformanceConfiguration.
                Some(1) => {
                    let mode = self.mem.read_u32(data).unwrap_or(0) as usize;
                    let configuration = self.apm_configuration(mode);
                    self.write_ipc_response(tls, 0, &[], &configuration.to_le_bytes(), &[])
                }
                // SetCpuOverclockEnabled(bool): there is no clock to raise.
                Some(2) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ISystemManager, the privileged side: the system, not a title,
            // decides the mode on real hardware.
            "apm:sys" => match cmd_id {
                // RequestPerformanceMode(ApmPerformanceMode): accepted, and
                // changes nothing — the same answer as a console that is
                // already in the mode it was asked for.
                Some(0) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // ClearLastThrottlingState / LoadAndApplySettings /
                // SetCpuBoostMode(u32): nothing throttles and nothing boosts.
                Some(4) | Some(5) | Some(6) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetCurrentPerformanceConfiguration ->
                // ApmPerformanceConfiguration, for the mode the console is
                // actually in.
                Some(7) => {
                    let configuration = self.apm_configuration(APM_PERFORMANCE_MODE_NORMAL as usize);
                    self.write_ipc_response(tls, 0, &[], &configuration.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }

    /// The `ApmPerformanceConfiguration` in force for a performance mode: what
    /// was last set for it, or the console's default.
    fn apm_configuration(&self, mode: usize) -> u32 {
        *self
            .apm_configuration
            .get(mode)
            .unwrap_or(&APM_DEFAULT_CONFIGURATION[APM_PERFORMANCE_MODE_NORMAL as usize])
    }

    /// `psc:m`: power-state change notifications.
    ///
    /// A process registers a module here and is told when the console is
    /// about to sleep, wake or shut down, so it can save what it is doing.
    /// This console does none of those things — there is no sleep, no
    /// shutdown and no battery to run down — so the module registers
    /// successfully and its event never fires.
    pub(super) fn psc_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(3) => self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[]),
                Some(0) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "psc:service");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
            };
        }
        let object_id = self.ipc_domain_object_id(tls);
        let iface = if self.ipc_is_domain_request(tls) {
            self.domain_interface(handle, object_id).unwrap_or("psc:service").to_string()
        } else {
            self.service_name(handle).unwrap_or("psc:service").to_string()
        };
        match iface.as_str() {
            // IPmService::GetPmModule -> IPmModule.
            "psc:m" | "psc:service" => match cmd_id {
                Some(0) => {
                    self.reply_with_interface(tls, handle, "psc:module")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "psc:module" => match cmd_id {
                // Initialize(u32 module_id, buffer<dependencies>) -> the
                // event the module waits on for a state change. Handed out
                // and never signalled: nothing here ever changes power state.
                Some(0) => {
                    let h = self.alloc_event("psc:module", true);
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                // GetRequest -> { PscPmState state, u32 flags }. Only read
                // after the event fires, which it does not; the honest answer
                // if it is asked anyway is the state the console is in.
                Some(1) => {
                    const PSC_PM_STATE_AWAKE: u32 = 0;
                    let mut raw = [0u8; 8];
                    raw[..4].copy_from_slice(&PSC_PM_STATE_AWAKE.to_le_bytes());
                    self.write_ipc_response(tls, 0, &[], &raw, &[])
                }
                // Acknowledge / Finalize / AcknowledgeEx: the module telling
                // the system it has finished reacting to a change that never
                // happened.
                Some(2) | Some(3) | Some(4) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }

    /// `gpio` (`IManager`), and the `IPadSession` it hands out for one pad.
    ///
    /// A GPIO pad is a single wire into the SoC, addressed by a device code.
    /// Nothing is wired to this console — no volume rocker, no SD card detect
    /// switch, no dock — so no pad is ever driven and no interrupt ever fires.
    ///
    /// An undriven pad reads **High**, and that polarity is load-bearing
    /// rather than cosmetic: the buttons are active-low, and boot2 enters
    /// maintenance mode when *both* volume pads read Low, so answering 0 here
    /// boots this console into maintenance mode every single time.
    pub(super) fn gpio_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        /// `GpioValue::High`, the level a pad with nothing pulling it down
        /// sits at — which is what an unpressed active-low button looks like.
        const HIGH: u32 = 1;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(0) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "gpio");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                Some(3) => self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[]),
                _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
            };
        }
        let object_id = self.ipc_domain_object_id(tls);
        let iface = if self.ipc_is_domain_request(tls) {
            self.domain_interface(handle, object_id).unwrap_or("gpio").to_string()
        } else {
            self.service_name(handle).unwrap_or("gpio").to_string()
        };
        match iface.as_str() {
            "gpio:pad" => match cmd_id {
                // SetDirection / SetInterruptMode / SetInterruptEnable /
                // ClearInterruptStatus / SetValue / UnbindInterrupt /
                // SetDebounceEnabled / SetDebounceTime / SetValueForSleepState:
                // accepted and dropped. There is no pad on the other end to
                // change, and nothing reads these back except the getters
                // below, which answer from the same nothing.
                Some(0) | Some(2) | Some(4) | Some(7) | Some(8) | Some(11) | Some(12)
                | Some(14) | Some(16) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetDirection (Input) / GetInterruptMode / GetInterruptEnable
                // / GetInterruptStatus (Inactive) / GetDebounceEnabled /
                // GetDebounceTime.
                Some(1) | Some(3) | Some(5) | Some(6) | Some(13) | Some(15) => {
                    self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[])
                }
                // GetValue / GetValueForSleepState -> GpioValue.
                Some(9) | Some(17) => self.write_ipc_response(tls, 0, &[], &HIGH.to_le_bytes(), &[]),
                // BindInterrupt -> the event the pad's interrupt signals.
                // Handed out and never signalled, because nothing drives it.
                Some(10) => {
                    let h = self.alloc_event("gpio:pad", false);
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IManager.
            _ => match cmd_id {
                // OpenSessionForDev / OpenSession / OpenSessionForTest /
                // OpenSession2 -> IPadSession. Which pad was asked for does not
                // matter; every pad here behaves the same way.
                Some(0) | Some(1) | Some(2) | Some(7) => {
                    self.reply_with_interface(tls, handle, "gpio:pad")?;
                    Ok(())
                }
                // IsWakeEventActive / IsWakeEventActive2 -> bool. Nothing here
                // wakes the console, because nothing here sleeps.
                Some(3) | Some(8) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                // GetWakeEventActiveFlagSet -> the set of pads that would.
                Some(4) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
                // The debug settings for that same never-taken wake path, plus
                // SetRetryValues. Firmware moved a `GetWakeEventActiveFlagSet2`
                // through this range, so these answer with a zeroed word as
                // well as a success — a setter's caller ignores it, and a
                // getter's reads the flag set as empty, which is the same
                // answer command 4 gives.
                Some(5) | Some(6) | Some(9) | Some(10) | Some(11) => {
                    self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::cpu::ipc::testing::*;

    #[test]
    fn psm_reports_the_host_supplied_battery_level() {
        let mut cpu = request(false, 0, &[]);
        cpu.set_battery(42, false);
        cpu.psm_request(TLS, Some(0), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 42);

        let mut cpu = request(false, 1, &[]);
        cpu.set_battery(42, false);
        cpu.psm_request(TLS, Some(1), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "not charging -> Unconnected");

        cpu.set_battery(100, true);
        cpu.psm_request(TLS, Some(1), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 1, "charging -> EnoughPower");
    }

    #[test]
    fn ts_open_session_picks_the_sensor_by_the_device_code() {
        // The high byte separates them: 0x41…… is the SoC and 0x43…… the PCB.
        // NX-Fetch asks for 0x41000002 and labels what comes back "CPU", so
        // reading the *low* byte handed it the PCB's temperature under the
        // SoC's name.
        for (device_code, expected) in
            [(0x4100_0002u32, "ts:session-internal"), (0x4300_0001, "ts:session-external")]
        {
            let mut cpu = request(false, 4, &device_code.to_le_bytes());
            cpu.register_service_handle(9, "ts");
            cpu.ts_request(TLS, 9, Some(4)).unwrap();
            let session = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;
            assert_eq!(cpu.service_name(session), Some(expected), "{device_code:#x}");
        }
    }

    #[test]
    fn ts_sessions_report_their_own_sensor_as_a_float() {
        // `ISession::GetTemperature` is command 4 — the same id the *server*
        // uses for OpenSession — and reports a float. Sharing one dispatch
        // between them answered a temperature request with a session object,
        // which is what NX-Fetch drew as "8 C".
        for (iface, expected) in [
            ("ts:session-internal", super::TS_TEMPERATURE_C[0]),
            ("ts:session-external", super::TS_TEMPERATURE_C[1]),
        ] {
            let mut cpu = request(false, 4, &[]);
            cpu.register_service_handle(9, iface);
            cpu.ts_request(TLS, 9, Some(4)).unwrap();
            let reading = f32::from_le_bytes(
                cpu.read_bytes(TLS + 0x20, 4).try_into().unwrap(),
            );
            assert_eq!(reading, expected as f32, "{iface}");
        }
    }

    #[test]
    fn ts_reports_the_same_temperature_in_both_units_and_inside_its_range() {
        // Internal (the SoC), then External (the PCB): two sensors, two
        // readings, and the pair of commands that report each one in degrees
        // and in millidegrees have to agree.
        for location in [0u8, 1] {
            let mut cpu = request(false, 1, &[location]);
            cpu.register_service_handle(9, "ts");
            cpu.ts_request(TLS, 9, Some(1)).unwrap();
            let celsius = cpu.mem.read_u32(TLS + 0x20).unwrap() as i32;

            write_request(&mut cpu, 3, &[location]);
            cpu.ts_request(TLS, 9, Some(3)).unwrap();
            let milli = cpu.mem.read_u32(TLS + 0x20).unwrap() as i32;
            assert_eq!(milli, celsius * 1000, "location {location}");

            // And the reading has to sit inside the range the same service
            // reports, or a caller scaling a gauge by it draws off the end.
            write_request(&mut cpu, 0, &[location]);
            cpu.ts_request(TLS, 9, Some(0)).unwrap();
            let low = cpu.mem.read_u32(TLS + 0x20).unwrap() as i32;
            let high = cpu.mem.read_u32(TLS + 0x24).unwrap() as i32;
            assert!(low <= celsius && celsius <= high, "{celsius} outside {low}..={high}");
        }
    }

    #[test]
    fn clkrst_reports_handheld_rates_for_the_modules_nx_fetch_asks_about() {
        // The device codes NX-Fetch sends, and the labels it puts on them.
        // A code is 0x40000000 + module + 1, so reading the low bits as the
        // module is off by one — which had the GPU's rate under "CPU".
        for (code, expected) in [
            (0x4000_0001u32, super::CLOCK_RATES_HZ[0]), // CpuBus
            (0x4000_0002, super::CLOCK_RATES_HZ[1]),    // GPU
            (0x4000_0039, super::CLOCK_RATES_HZ[2]),    // EMC
        ] {
            let mut cpu = request(false, 0, &code.to_le_bytes());
            cpu.register_service_handle(9, "clkrst");
            cpu.pcv_request(TLS, 9, Some(0)).unwrap();
            let session = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;

            write_request(&mut cpu, 8, &[]);
            cpu.pcv_request(TLS, session, Some(8)).unwrap();
            assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), expected, "{code:#x}");
        }
    }

    #[test]
    fn clkrst_gives_back_the_rate_it_was_set_to() {
        let mut cpu = request(false, 0, &0x4000_0002u32.to_le_bytes());
        cpu.register_service_handle(9, "clkrst");
        cpu.pcv_request(TLS, 9, Some(0)).unwrap();
        let session = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;

        write_request(&mut cpu, 7, &768_000_000u32.to_le_bytes());
        cpu.pcv_request(TLS, session, Some(7)).unwrap();
        write_request(&mut cpu, 8, &[]);
        cpu.pcv_request(TLS, session, Some(8)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 768_000_000);

        // The CPU's rate is its own, not the one just set for the GPU.
        let mut cpu2 = request(false, 3, &0u32.to_le_bytes());
        cpu2.register_service_handle(9, "pcv");
        cpu2.pcv_request(TLS, 9, Some(3)).unwrap();
        assert_eq!(cpu2.mem.read_u32(TLS + 0x20).unwrap(), super::CLOCK_RATES_HZ[0]);
    }

    #[test]
    fn apm_agrees_with_am_about_the_performance_mode() {
        // `IManager::GetPerformanceMode` and `ICommonStateGetter::
        // GetPerformanceMode` are two routes to the same fact, and a title
        // that gets two answers concludes the mode changed underneath it.
        let mut cpu = request(false, 1, &[]);
        cpu.register_service_handle(9, "apm");
        cpu.apm_request(TLS, 9, Some(1)).unwrap();
        let from_apm = cpu.mem.read_u32(TLS + 0x20).unwrap();

        let mut cpu = request(false, 6, &[]);
        cpu.register_service_handle(9, "am:common-state-getter");
        cpu.applet_request(TLS, 9, Some(6)).unwrap();
        assert_eq!(from_apm, cpu.mem.read_u32(TLS + 0x20).unwrap());
    }

    #[test]
    fn apm_gives_back_the_performance_configuration_it_was_given() {
        // OpenSession, then set a configuration for Boost and read it back.
        let mut cpu = request(false, 0, &[]);
        cpu.register_service_handle(9, "apm");
        cpu.apm_request(TLS, 9, Some(0)).unwrap();
        let session = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;
        assert_eq!(cpu.service_name(session), Some("apm:session"));

        let mut payload = [0u8; 8];
        payload[..4].copy_from_slice(&1u32.to_le_bytes()); // Boost
        payload[4..].copy_from_slice(&0x0002_0003u32.to_le_bytes());
        write_request(&mut cpu, 0, &payload);
        cpu.apm_request(TLS, session, Some(0)).unwrap();

        write_request(&mut cpu, 1, &1u32.to_le_bytes());
        cpu.apm_request(TLS, session, Some(1)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0x0002_0003);

        // Normal keeps its own, un-set, configuration: the two modes are
        // separate settings.
        write_request(&mut cpu, 1, &0u32.to_le_bytes());
        cpu.apm_request(TLS, session, Some(1)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), super::APM_DEFAULT_CONFIGURATION[0]);
        assert_ne!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "0 is Invalid");
    }
}
