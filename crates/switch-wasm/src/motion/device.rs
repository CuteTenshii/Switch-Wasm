//! The WebHID side: open the controller the page was granted, set up its IMU and stream it.

use std::cell::RefCell;

use js_sys::{DataView, Function, Promise, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::WorkerGlobalScope;

use super::{publish, sample, Calibration, Kind, VENDOR};
use crate::post::post;
use crate::session_if_live;

// web-sys has WebHID only behind `web_sys_unstable_apis`, which also reshapes bindings the
// rest of the crate uses, so the few members needed are declared here.
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = HID)]
    type Hid;
    #[wasm_bindgen(method, js_name = getDevices)]
    fn get_devices(this: &Hid) -> Promise;
    #[wasm_bindgen(method, setter, js_name = ondisconnect)]
    fn set_ondisconnect(this: &Hid, handler: Option<&Function>);

    #[wasm_bindgen(js_name = HIDDevice)]
    #[derive(Clone)]
    type HidDevice;
    #[wasm_bindgen(method, getter, js_name = vendorId)]
    fn vendor_id(this: &HidDevice) -> u16;
    #[wasm_bindgen(method, getter, js_name = productId)]
    fn product_id(this: &HidDevice) -> u16;
    #[wasm_bindgen(method, getter, js_name = productName)]
    fn product_name(this: &HidDevice) -> String;
    #[wasm_bindgen(method, getter)]
    fn opened(this: &HidDevice) -> bool;
    #[wasm_bindgen(method)]
    fn open(this: &HidDevice) -> Promise;
    #[wasm_bindgen(method)]
    fn close(this: &HidDevice) -> Promise;
    #[wasm_bindgen(method, setter, js_name = oninputreport)]
    fn set_oninputreport(this: &HidDevice, handler: Option<&Function>);
    #[wasm_bindgen(method, catch, js_name = sendReport)]
    fn send_report(this: &HidDevice, report_id: u8, data: &mut [u8]) -> Result<Promise, JsValue>;

    type HidInputReportEvent;
    #[wasm_bindgen(method, getter, js_name = reportId)]
    fn report_id(this: &HidInputReportEvent) -> u8;
    #[wasm_bindgen(method, getter)]
    fn data(this: &HidInputReportEvent) -> DataView;

    type HidConnectionEvent;
    #[wasm_bindgen(method, getter)]
    fn device(this: &HidConnectionEvent) -> HidDevice;
}

const REPORT_SUBCOMMAND: u8 = 0x01;
const REPORT_REPLY: u8 = 0x21;
const REPORT_FULL: u8 = 0x30;
const SUB_REPORT_MODE: u8 = 0x03;
const SUB_SPI_READ: u8 = 0x10;
const SUB_ENABLE_IMU: u8 = 0x40;
const SPI_FACTORY_IMU: u32 = 0x6020;
const SPI_USER_IMU_MAGIC: u32 = 0x8026;
const SPI_USER_IMU: u32 = 0x8028;
const NEUTRAL_RUMBLE: [u8; 8] = [0x00, 0x01, 0x40, 0x40, 0x00, 0x01, 0x40, 0x40];
const REPLY_TIMEOUT_MS: i32 = 1000;
/// Offsets into a reply's data, which excludes the report id.
const ACK: usize = 12;
const REPLY_SUBCOMMAND: usize = 13;
const SPI_DATA: usize = 19;

struct Active {
    device: HidDevice,
    kind: Kind,
    calibration: Calibration,
    _on_report: Closure<dyn FnMut(HidInputReportEvent)>,
    _on_disconnect: Closure<dyn FnMut(HidConnectionEvent)>,
}

struct Pending {
    subcommand: u8,
    resolve: Function,
}

thread_local! {
    static ACTIVE: RefCell<Option<Active>> = const { RefCell::new(None) };
    static PENDING: RefCell<Option<Pending>> = const { RefCell::new(None) };
    static COUNTER: RefCell<u8> = const { RefCell::new(0) };
}

fn describe(err: JsValue) -> String {
    err.as_string()
        .or_else(|| Reflect::get(&err, &"message".into()).ok()?.as_string())
        .unwrap_or_else(|| "unknown error".into())
}

fn hid() -> Result<Hid, String> {
    let scope: WorkerGlobalScope = js_sys::global().unchecked_into();
    Reflect::get(&scope.navigator(), &"hid".into())
        .ok()
        .and_then(|hid| hid.dyn_into().ok())
        .ok_or_else(|| "this browser has no WebHID".to_string())
}

fn on_report(event: HidInputReportEvent) {
    let data = Uint8Array::new(&event.data().buffer()).to_vec();
    match event.report_id() {
        REPORT_REPLY => {
            let reply = PENDING.with_borrow_mut(|pending| match pending {
                Some(p) if data.get(REPLY_SUBCOMMAND) == Some(&p.subcommand) => pending.take(),
                _ => None,
            });
            if let Some(Pending { resolve, .. }) = reply {
                resolve
                    .call1(&JsValue::NULL, &Uint8Array::from(&data[..]))
                    .ok();
            }
        }
        REPORT_FULL => {
            let reading = ACTIVE.with_borrow(|active| {
                let active = active.as_ref()?;
                sample(&data, active.kind, &active.calibration)
            });
            if let (Some(reading), Some(s)) = (reading, session_if_live()) {
                publish(&mut s.cpu, Some(reading));
            }
        }
        _ => {}
    }
}

/// Send a subcommand and wait for its reply data.
async fn subcommand(device: &HidDevice, subcommand: u8, args: &[u8]) -> Result<Vec<u8>, String> {
    let reply = Promise::new(&mut |resolve, _| {
        PENDING.set(Some(Pending {
            subcommand,
            resolve,
        }));
    });
    let timeout = Promise::new(&mut |resolve, _| {
        let scope: WorkerGlobalScope = js_sys::global().unchecked_into();
        scope
            .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, REPLY_TIMEOUT_MS)
            .ok();
    });
    let counter = COUNTER.with_borrow_mut(|c| {
        *c = (*c + 1) & 0x0f;
        *c
    });
    let mut packet = vec![counter];
    packet.extend(NEUTRAL_RUMBLE);
    packet.push(subcommand);
    packet.extend(args);
    let sent = device
        .send_report(REPORT_SUBCOMMAND, &mut packet)
        .map_err(describe)?;
    JsFuture::from(sent).await.map_err(describe)?;
    let winner = JsFuture::from(Promise::race(&js_sys::Array::of2(&reply, &timeout)))
        .await
        .map_err(describe)?;
    PENDING.set(None);
    let data = winner
        .dyn_into::<Uint8Array>()
        .map_err(|_| format!("no reply to subcommand {subcommand:#x}"))?
        .to_vec();
    if data.get(ACK).is_some_and(|ack| ack & 0x80 != 0) {
        Ok(data)
    } else {
        Err(format!("subcommand {subcommand:#x} refused"))
    }
}

async fn read_spi(device: &HidDevice, address: u32, length: u8) -> Result<Vec<u8>, String> {
    let [low, high, ..] = address.to_le_bytes();
    let reply = subcommand(device, SUB_SPI_READ, &[low, high, 0, 0, length]).await?;
    reply
        .get(SPI_DATA..SPI_DATA + usize::from(length))
        .map(<[u8]>::to_vec)
        .ok_or_else(|| "short SPI reply".to_string())
}

async fn read_calibration(device: &HidDevice) -> Result<Calibration, String> {
    let magic = read_spi(device, SPI_USER_IMU_MAGIC, 2).await?;
    let user = magic == [0xb2, 0xa1];
    let raw = read_spi(
        device,
        if user { SPI_USER_IMU } else { SPI_FACTORY_IMU },
        0x18,
    )
    .await?;
    Ok(Calibration::from_spi(&raw))
}

fn state(device: Option<&HidDevice>) {
    let name = device.map_or(JsValue::NULL, |d| d.product_name().into());
    post("motion", &[("device", name)]);
}

/// Open the first controller the page has been granted and stream its motion to the core.
/// Returns its name, or an empty string when there is none.
#[wasm_bindgen]
pub async fn switch_motion_attach() -> Result<String, String> {
    if ACTIVE.with_borrow(Option::is_some) {
        return Err("a motion controller is already attached".into());
    }
    let hid = hid()?;
    let devices = JsFuture::from(hid.get_devices()).await.map_err(describe)?;
    let Some((device, kind)) = js_sys::Array::from(&devices)
        .iter()
        .map(JsCast::unchecked_into::<HidDevice>)
        .find_map(|d| {
            let kind = Kind::from_product(d.product_id()).filter(|_| d.vendor_id() == VENDOR)?;
            Some((d, kind))
        })
    else {
        return Ok(String::new());
    };
    if !device.opened() {
        JsFuture::from(device.open()).await.map_err(describe)?;
    }
    let on_report = Closure::<dyn FnMut(HidInputReportEvent)>::new(on_report);
    device.set_oninputreport(Some(on_report.as_ref().unchecked_ref()));
    let on_disconnect =
        Closure::<dyn FnMut(HidConnectionEvent)>::new(move |event: HidConnectionEvent| {
            let lost =
                ACTIVE.with_borrow(|a| a.as_ref().is_some_and(|a| *a.device == *event.device()));
            if lost {
                wasm_bindgen_futures::spawn_local(switch_motion_detach());
            }
        });
    hid.set_ondisconnect(Some(on_disconnect.as_ref().unchecked_ref()));
    let setup = async {
        let calibration = read_calibration(&device).await.unwrap_or_default();
        subcommand(&device, SUB_ENABLE_IMU, &[1]).await?;
        subcommand(&device, SUB_REPORT_MODE, &[REPORT_FULL]).await?;
        Ok::<_, String>(calibration)
    };
    match setup.await {
        Ok(calibration) => {
            let name = device.product_name();
            state(Some(&device));
            ACTIVE.set(Some(Active {
                device,
                kind,
                calibration,
                _on_report: on_report,
                _on_disconnect: on_disconnect,
            }));
            Ok(name)
        }
        Err(err) => {
            device.set_oninputreport(None);
            hid.set_ondisconnect(None);
            JsFuture::from(device.close()).await.ok();
            Err(err)
        }
    }
}

/// Stop the stream and close the controller.
#[wasm_bindgen]
pub async fn switch_motion_detach() {
    let Some(active) = ACTIVE.take() else {
        return;
    };
    active.device.set_oninputreport(None);
    if let Ok(hid) = hid() {
        hid.set_ondisconnect(None);
    }
    if let Some(s) = session_if_live() {
        publish(&mut s.cpu, None);
    }
    state(None);
    JsFuture::from(active.device.close()).await.ok();
}
