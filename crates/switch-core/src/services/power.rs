//! Power, clock and sensor services: `psm`, `apm`, `pcv`/`clkrst`, `mm:u`,
//! `ts`, `psc` and `gpio`.
//!
//! One emulated instruction is one cycle of [`CLOCK_RATES_HZ`]`[0]`, the rate
//! the display tick, thread deadlines and audio clocks are counted in.

use crate::cpu::Cpu;
use crate::Result;

/// CPU, GPU, memory and unmodelled module rates in Hz, at handheld clocks.
pub(crate) const CLOCK_RATES_HZ: [u32; 4] = [1_020_000_000, 384_000_000, 1_600_000_000, 0];

/// Fixed idle `ts` readings in Celsius: SoC (`TsLocation_Internal`), then PCB.
const TS_TEMPERATURE_C: [i32; 2] = [40, 35];

const TS_TEMPERATURE_RANGE_C: (i32, i32) = (0, 100);

/// `ApmPerformanceMode_Normal`, the handheld mode `am` also reports.
const APM_PERFORMANCE_MODE_NORMAL: u32 = 0;

/// Default `ApmPerformanceConfiguration` per mode; nonzero because 0 is `Invalid`.
pub(crate) const APM_DEFAULT_CONFIGURATION: [u32; 2] = [0x0001_0000, 0x0002_0000];

impl Cpu {
    /// `psm`: the battery. Control commands share ids with its own, so they are checked first.
    pub(crate) fn psm_request(&mut self, tls: u32, cmd_id: Option<u32>, handle: u64) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "psm");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
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
        // ChargerType: the host only reports a charging bool, so EnoughPower or Unconnected.
        const CHARGER_UNCONNECTED: u32 = 0;
        const CHARGER_ENOUGH_POWER: u32 = 1;
        match cmd_id {
            Some(GET_BATTERY_CHARGE_PERCENTAGE) => {
                let (percent, _) = self.battery();
                self.write_ipc_response(tls, 0, &[], &(percent as u32).to_le_bytes(), &[])
            }
            Some(GET_CHARGER_TYPE) => {
                let (_, charging) = self.battery();
                let charger = if charging {
                    CHARGER_ENOUGH_POWER
                } else {
                    CHARGER_UNCONNECTED
                };
                self.write_ipc_response(tls, 0, &[], &charger.to_le_bytes(), &[])
            }
            // Charging mirrors the host battery, so these are accepted and ignored.
            Some(ENABLE_BATTERY_CHARGING) | Some(DISABLE_BATTERY_CHARGING) => {
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(IS_BATTERY_CHARGING_ENABLED) => self.write_ipc_response(tls, 0, &[], &[1u8], &[]),
            Some(OPEN_SESSION) => {
                self.reply_with_interface(tls, handle, "psm-session")?;
                Ok(())
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// `IPsmSession`: its event is never signalled; [`Cpu::set_battery`] is polled.
    pub(crate) fn psm_session_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
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

    /// `clkrst` and `pcv`, the same clock manager either side of 8.0.0.
    /// Rates a guest sets are stored and read back.
    pub(crate) fn pcv_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return self.write_ipc_response(tls, 0, &[], &[], &[]);
        }
        let data = self.ipc_request_data(tls);
        let iface = self.service_name(handle).unwrap_or("pcv").to_string();
        if iface == "clkrst" {
            return match cmd_id {
                // OpenSession(u32 device_code, u32 unk) -> IClkrstSession.
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
            // SetPowerEnabled / SetClockEnabled and their disables.
            Some(0) | Some(1) | Some(4) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }

    /// `mm:u`: multimedia clock requests. Older commands (below 4) address a
    /// request by module, newer ones by the id `Initialize` returns.
    pub(crate) fn mm_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return self.write_ipc_response(tls, 0, &[], &[], &[]);
        }
        let data = self.ipc_request_data(tls);
        let arg = |cpu: &Cpu, i: u32| cpu.mem.read_u32(data.wrapping_add(4 * i)).unwrap_or(0);
        match cmd_id {
            // InitializeOld(module, priority, clear mode).
            Some(0) => {
                let module = arg(self, 0);
                self.mm_requests.insert(module, (module, 0));
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // Initialize(module, priority, clear mode) -> request id.
            Some(4) => {
                let module = arg(self, 0);
                let id = (0..=u32::MAX)
                    .find(|id| !self.mm_requests.contains_key(id))
                    .unwrap_or(0);
                self.mm_requests.insert(id, (module, 0));
                self.write_ipc_response(tls, 0, &[], &id.to_le_bytes(), &[])
            }
            // FinalizeOld(module) / Finalize(request id).
            Some(1) | Some(5) => {
                self.mm_requests.remove(&arg(self, 0));
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // SetAndWaitOld(module, min, max) / SetAndWait(request id, min, max).
            Some(2) | Some(6) => {
                let (key, floor) = (arg(self, 0), arg(self, 1));
                let module = self
                    .mm_requests
                    .get(&key)
                    .map_or(key, |&(module, _)| module);
                self.mm_requests.insert(key, (module, floor));
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // GetOld(module) / Get(request id) -> the clock, in Hz.
            Some(3) | Some(7) => {
                let rate = self
                    .mm_requests
                    .get(&arg(self, 0))
                    .map_or(0, |&(_, floor)| floor);
                self.write_ipc_response(tls, 0, &[], &rate.to_le_bytes(), &[])
            }
            _ => self.unimplemented_command(tls, "mm:u", cmd_id),
        }
    }

    /// Map `pcv`'s module enum or a `clkrst` device code (`0x40000000 + module + 1`)
    /// to an index into [`CLOCK_RATES_HZ`]; unmodelled modules report 0 Hz.
    fn clkrst_module(&self, code: u32) -> u32 {
        const PCV_MODULE_CPU_BUS: u32 = 0;
        const PCV_MODULE_GPU: u32 = 1;
        const PCV_MODULE_EMC: u32 = 0x38;
        let module = if code >= 0x4000_0000 {
            (code & 0xFF).wrapping_sub(1)
        } else {
            code
        };
        match module {
            PCV_MODULE_CPU_BUS => 0,
            PCV_MODULE_GPU => 1,
            PCV_MODULE_EMC => 2,
            _ => 3,
        }
    }

    /// The last rate set for a module, else its default. Only the GPU's default
    /// follows the dock; the CPU rate defines emulated time.
    fn clock_rate(&self, module: u32) -> u32 {
        const GPU: u32 = 1;
        match self.clock_rates.get(&module) {
            Some(&rate) => rate,
            None if module == GPU => self.operation_mode().gpu_clock_hz(),
            None => CLOCK_RATES_HZ[module as usize],
        }
    }

    fn clkrst_session_name(module: u32) -> &'static str {
        match module {
            1 => "clkrst:session-1",
            2 => "clkrst:session-2",
            3 => "clkrst:session-3",
            _ => "clkrst:session-0",
        }
    }

    /// `ts` (`IMeasurementServer`): two fixed idle sensor readings.
    pub(crate) fn ts_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "ts");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, "ts:control", cmd_id),
            };
        }
        let object_id = self.ipc_domain_object_id(tls);
        // A session, by handle or domain object, is a different interface from the server.
        let iface = if self.ipc_is_domain_request(tls) {
            self.domain_interface(handle, object_id)
                .unwrap_or("ts")
                .to_string()
        } else {
            self.service_name(handle).unwrap_or("ts").to_string()
        };
        if iface.starts_with("ts:session") {
            return self.ts_session_request(tls, &iface, cmd_id);
        }
        // 0 = Internal (SoC), 1 = External (PCB); anything else reads as Internal.
        let location = self.mem.read_u8(self.ipc_request_data(tls)).unwrap_or(0);
        let celsius = TS_TEMPERATURE_C[usize::from(location).min(TS_TEMPERATURE_C.len() - 1)];
        match cmd_id {
            // GetTemperatureRange(TsLocation) -> (s32 min, s32 max).
            Some(0) => {
                let mut range = [0u8; 8];
                range[..4].copy_from_slice(&TS_TEMPERATURE_RANGE_C.0.to_le_bytes());
                range[4..].copy_from_slice(&TS_TEMPERATURE_RANGE_C.1.to_le_bytes());
                self.write_ipc_response(tls, 0, &[], &range, &[])
            }
            // GetTemperature(TsLocation) -> s32 degrees Celsius.
            Some(1) => self.write_ipc_response(tls, 0, &[], &celsius.to_le_bytes(), &[]),
            // SetMeasurementMode(TsLocation, TsMeasurementMode).
            Some(2) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // GetTemperatureMilliC(TsLocation) -> s32 millidegrees.
            Some(3) => {
                let milli = celsius * 1000;
                self.write_ipc_response(tls, 0, &[], &milli.to_le_bytes(), &[])
            }
            // OpenSession(u32 device_code) -> ISession. The code's high byte picks
            // the sensor: `0x41……` SoC, `0x43……` PCB.
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

    /// `ISession` from `ts::OpenSession`. Its command 4 is `GetTemperature` (f32),
    /// unlike the server's command 4.
    pub(crate) fn ts_session_request(
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

    /// `apm` and `apm:sys`. Reports the same performance mode as `am` and reads
    /// back configurations it was given.
    pub(crate) fn apm_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let name = self.service_name(handle).unwrap_or("apm").to_string();
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, &name);
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, "apm:control", cmd_id),
            };
        }
        let object_id = self.ipc_domain_object_id(tls);
        let iface = if self.ipc_is_domain_request(tls) {
            self.domain_interface(handle, object_id)
                .unwrap_or("apm")
                .to_string()
        } else {
            match self.service_name(handle) {
                Some(name) => name.to_string(),
                None => "apm".to_string(),
            }
        };
        let data = self.ipc_request_data(tls);
        match iface.as_str() {
            // IManager; `apm:p` and `apm:am` are the same interface.
            "apm" | "apm:p" | "apm:am" => match cmd_id {
                // OpenSession -> ISession.
                Some(0) => {
                    self.reply_with_interface(tls, handle, "apm:session")?;
                    Ok(())
                }
                // GetPerformanceMode: must match `am`'s.
                Some(1) => {
                    let mode = self.operation_mode().performance_mode();
                    self.write_ipc_response(tls, 0, &[], &mode.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ISession.
            "apm:session" => match cmd_id {
                // SetPerformanceConfiguration(ApmPerformanceMode, ApmPerformanceConfiguration).
                Some(0) => {
                    let mode = self.mem.read_u32(data).unwrap_or(0);
                    let configuration = self.mem.read_u32(data.wrapping_add(4)).unwrap_or(0);
                    if let Some(slot) = self.apm_configuration.get_mut(mode as usize) {
                        *slot = configuration;
                    }
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetPerformanceConfiguration(ApmPerformanceMode) -> ApmPerformanceConfiguration.
                Some(1) => {
                    let mode = self.mem.read_u32(data).unwrap_or(0) as usize;
                    let configuration = self.apm_configuration(mode);
                    self.write_ipc_response(tls, 0, &[], &configuration.to_le_bytes(), &[])
                }
                // SetCpuOverclockEnabled(bool).
                Some(2) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ISystemManager.
            "apm:sys" => match cmd_id {
                // RequestPerformanceMode(ApmPerformanceMode): accepted, changes nothing.
                Some(0) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // ClearLastThrottlingState / LoadAndApplySettings / SetCpuBoostMode(u32).
                Some(4) | Some(5) | Some(6) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetCurrentPerformanceConfiguration -> ApmPerformanceConfiguration.
                Some(7) => {
                    let configuration =
                        self.apm_configuration(APM_PERFORMANCE_MODE_NORMAL as usize);
                    self.write_ipc_response(tls, 0, &[], &configuration.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }

    /// The last configuration set for a mode, or the default.
    fn apm_configuration(&self, mode: usize) -> u32 {
        *self
            .apm_configuration
            .get(mode)
            .unwrap_or(&APM_DEFAULT_CONFIGURATION[APM_PERFORMANCE_MODE_NORMAL as usize])
    }

    /// `psc:m`: power-state change notifications. Nothing here changes power
    /// state, so module events never fire.
    pub(crate) fn psc_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
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
            self.domain_interface(handle, object_id)
                .unwrap_or("psc:service")
                .to_string()
        } else {
            self.service_name(handle)
                .unwrap_or("psc:service")
                .to_string()
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
                // Initialize(u32 module_id, buffer<dependencies>) -> event, never signalled.
                Some(0) => {
                    let h = self.alloc_event("psc:module", true);
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                // GetRequest -> { PscPmState state, u32 flags }.
                Some(1) => {
                    const PSC_PM_STATE_AWAKE: u32 = 0;
                    let mut raw = [0u8; 8];
                    raw[..4].copy_from_slice(&PSC_PM_STATE_AWAKE.to_le_bytes());
                    self.write_ipc_response(tls, 0, &[], &raw, &[])
                }
                // Acknowledge / Finalize / AcknowledgeEx.
                Some(2) | Some(3) | Some(4) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }

    /// `gpio` and its `IPadSession`. No pad is driven; every pad reads High,
    /// since boot2 enters maintenance mode if both active-low volume pads read Low.
    pub(crate) fn gpio_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        /// `GpioValue::High`, an unpressed active-low button.
        const HIGH: u32 = 1;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(0) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "gpio");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
            };
        }
        let object_id = self.ipc_domain_object_id(tls);
        let iface = if self.ipc_is_domain_request(tls) {
            self.domain_interface(handle, object_id)
                .unwrap_or("gpio")
                .to_string()
        } else {
            self.service_name(handle).unwrap_or("gpio").to_string()
        };
        match iface.as_str() {
            "gpio:pad" => match cmd_id {
                // SetDirection / SetInterruptMode / SetInterruptEnable /
                // ClearInterruptStatus / SetValue / UnbindInterrupt /
                // SetDebounceEnabled / SetDebounceTime / SetValueForSleepState.
                Some(0) | Some(2) | Some(4) | Some(7) | Some(8) | Some(11) | Some(12)
                | Some(14) | Some(16) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetDirection (Input) / GetInterruptMode / GetInterruptEnable
                // / GetInterruptStatus (Inactive) / GetDebounceEnabled /
                // GetDebounceTime.
                Some(1) | Some(3) | Some(5) | Some(6) | Some(13) | Some(15) => {
                    self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[])
                }
                // GetValue / GetValueForSleepState -> GpioValue.
                Some(9) | Some(17) => {
                    self.write_ipc_response(tls, 0, &[], &HIGH.to_le_bytes(), &[])
                }
                // BindInterrupt -> event, never signalled.
                Some(10) => {
                    let h = self.alloc_event("gpio:pad", false);
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IManager.
            _ => match cmd_id {
                // OpenSessionForDev / OpenSession / OpenSessionForTest /
                // OpenSession2 -> IPadSession; every pad behaves the same.
                Some(0) | Some(1) | Some(2) | Some(7) => {
                    self.reply_with_interface(tls, handle, "gpio:pad")?;
                    Ok(())
                }
                // IsWakeEventActive / IsWakeEventActive2 -> false.
                Some(3) | Some(8) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                // GetWakeEventActiveFlagSet -> empty.
                Some(4) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
                // Wake-path debug settings and SetRetryValues; a zeroed word serves
                // both setters and `GetWakeEventActiveFlagSet2`.
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
    use crate::kernel::ipc::testing::*;

    #[test]
    fn psm_reports_the_host_supplied_battery_level() {
        let mut cpu = request(false, 0, &[]);
        cpu.set_battery(42, false);
        cpu.psm_request(TLS, Some(0), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 42);

        let mut cpu = request(false, 1, &[]);
        cpu.set_battery(42, false);
        cpu.psm_request(TLS, Some(1), 9).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x20).unwrap(),
            0,
            "not charging -> Unconnected"
        );

        cpu.set_battery(100, true);
        cpu.psm_request(TLS, Some(1), 9).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x20).unwrap(),
            1,
            "charging -> EnoughPower"
        );
    }

    #[test]
    fn ts_open_session_picks_the_sensor_by_the_device_code() {
        // The high byte picks the sensor: 0x41…… SoC, 0x43…… PCB.
        for (device_code, expected) in [
            (0x4100_0002u32, "ts:session-internal"),
            (0x4300_0001, "ts:session-external"),
        ] {
            let mut cpu = request(false, 4, &device_code.to_le_bytes());
            cpu.register_service_handle(9, "ts");
            cpu.ts_request(TLS, 9, Some(4)).unwrap();
            let session = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;
            assert_eq!(
                cpu.service_name(session),
                Some(expected),
                "{device_code:#x}"
            );
        }
    }

    #[test]
    fn ts_sessions_report_their_own_sensor_as_a_float() {
        // `ISession::GetTemperature` is command 4, like the server's OpenSession, and returns f32.
        for (iface, expected) in [
            ("ts:session-internal", super::TS_TEMPERATURE_C[0]),
            ("ts:session-external", super::TS_TEMPERATURE_C[1]),
        ] {
            let mut cpu = request(false, 4, &[]);
            cpu.register_service_handle(9, iface);
            cpu.ts_request(TLS, 9, Some(4)).unwrap();
            let reading = f32::from_le_bytes(cpu.read_bytes(TLS + 0x20, 4).try_into().unwrap());
            assert_eq!(reading, expected as f32, "{iface}");
        }
    }

    #[test]
    fn ts_reports_the_same_temperature_in_both_units_and_inside_its_range() {
        for location in [0u8, 1] {
            let mut cpu = request(false, 1, &[location]);
            cpu.register_service_handle(9, "ts");
            cpu.ts_request(TLS, 9, Some(1)).unwrap();
            let celsius = cpu.mem.read_u32(TLS + 0x20).unwrap() as i32;

            write_request(&mut cpu, 3, &[location]);
            cpu.ts_request(TLS, 9, Some(3)).unwrap();
            let milli = cpu.mem.read_u32(TLS + 0x20).unwrap() as i32;
            assert_eq!(milli, celsius * 1000, "location {location}");

            // The reading must sit inside the reported range.
            write_request(&mut cpu, 0, &[location]);
            cpu.ts_request(TLS, 9, Some(0)).unwrap();
            let low = cpu.mem.read_u32(TLS + 0x20).unwrap() as i32;
            let high = cpu.mem.read_u32(TLS + 0x24).unwrap() as i32;
            assert!(
                low <= celsius && celsius <= high,
                "{celsius} outside {low}..={high}"
            );
        }
    }

    #[test]
    fn clkrst_reports_handheld_rates_for_the_modules_nx_fetch_asks_about() {
        // NX-Fetch's device codes: 0x40000000 + module + 1.
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
    fn mm_gives_back_the_clock_a_request_asked_for_until_it_is_finalized() {
        let words =
            |values: &[u32]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
        const MODULE: u32 = 5;
        let mut cpu = request(false, 4, &words(&[MODULE, 0, 0]));
        cpu.mm_request(TLS, Some(4)).unwrap();
        let id = cpu.mem.read_u32(TLS + 0x20).unwrap();

        write_request(&mut cpu, 6, &words(&[id, 600_000_000, u32::MAX]));
        cpu.mm_request(TLS, Some(6)).unwrap();
        write_request(&mut cpu, 7, &words(&[id]));
        cpu.mm_request(TLS, Some(7)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 600_000_000);

        write_request(&mut cpu, 5, &words(&[id]));
        cpu.mm_request(TLS, Some(5)).unwrap();
        write_request(&mut cpu, 7, &words(&[id]));
        cpu.mm_request(TLS, Some(7)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "finalized");
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
        assert_eq!(
            cpu2.mem.read_u32(TLS + 0x20).unwrap(),
            super::CLOCK_RATES_HZ[0]
        );
    }

    #[test]
    fn apm_agrees_with_am_about_the_performance_mode() {
        // Both routes to the performance mode must agree.
        use crate::cpu::OperationMode;
        for (mode, want) in [(OperationMode::Handheld, 0), (OperationMode::Docked, 1)] {
            let mut cpu = request(false, 1, &[]);
            cpu.set_operation_mode(mode);
            cpu.register_service_handle(9, "apm");
            cpu.apm_request(TLS, 9, Some(1)).unwrap();
            assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "apm result");
            assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), want, "apm {mode:?}");

            let mut cpu = request(false, 6, &[]);
            cpu.set_operation_mode(mode);
            cpu.register_service_handle(9, "am:common-state-getter");
            cpu.applet_request(TLS, 9, Some(6)).unwrap();
            assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "am result");
            assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), want, "am {mode:?}");
        }
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

        // Normal keeps its own configuration.
        write_request(&mut cpu, 1, &0u32.to_le_bytes());
        cpu.apm_request(TLS, session, Some(1)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x20).unwrap(),
            super::APM_DEFAULT_CONFIGURATION[0]
        );
        assert_ne!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "0 is Invalid");
    }
}
