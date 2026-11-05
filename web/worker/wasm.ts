// The wasm module and its session. The exports that carry bytes or text are the glue's own
// (`@core/switch_wasm.js`); what is here are the raw scalar ones.

export interface WasmExports {
  memory: WebAssembly.Memory;

  switch_init(): void;
  switch_add_core(): number;
  switch_end_title(): void;

  switch_open_nca(size: bigint): number;
  switch_clear_dlc(): void;
  switch_clear_update(): void;

  switch_load_control(): number;

  switch_load_nca(): bigint;
  switch_load_nca_from_nsp(index: number): bigint;
  switch_program_nca_index(): number;

  switch_set_trace(enabled: number): void;
  switch_set_jit(enabled: number): void;
  switch_set_trace_channels(on: number): void;
  switch_gpu_lost(): number;
  switch_log_activity(seconds: number): void;
  switch_log_next(): number;
  switch_set_input(
    buttons: bigint,
    stickLx: number, stickLy: number, stickRx: number, stickRy: number): void;
  switch_set_battery(percent: number, charging: number): void;
  switch_set_operation_mode(docked: number): void;
  switch_vibration(): number;

  switch_run(maxSteps: bigint): bigint;
  switch_pace(elapsed: bigint): bigint;
  switch_halted(): number;
  switch_log_regs(): void;
  switch_log_threads(): void;
  switch_get_reg(idx: number): bigint;
  switch_get_steps(): bigint;
  switch_guest_ram(): bigint;

  switch_fb_width(): number;
  switch_fb_height(): number;
  switch_frame_count(): number;
}

// The core's exports, set once at load. A live binding: importers see the assignment.
export let core = null as unknown as WasmExports;

export function setCore(exports: WasmExports): void {
  core = exports;
}

// IndexedDB commands can arrive while the module is still loading.
export function coreLoaded(): boolean {
  return (core as WasmExports | null) !== null;
}

// An empty answer is an empty report, so fall back instead of throwing.
export function parseOr<T>(text: string, fallback: T): T {
  try {
    return JSON.parse(text) as T;
  } catch {
    return fallback;
  }
}
