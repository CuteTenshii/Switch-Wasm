// The debug panel's Audio group: the title's mixers, the clock they keep time by, and the
// sound's way out through the emulator's ring and the browser's output.

import type {
  AudioOutReport, AudioReport, AudioRendererReport, AudioVoiceReport,
} from '../shared/protocol';
import { audioOutput, type AudioOutput } from './audio';
import { $, el } from './dom';
import { fmtCount } from '../shared/format';
import { call, hasSession } from './rpc';
import { titleBooted } from './session';
import { setNote } from './shell';

interface Reading {
  at: number;
  report: AudioReport;
  page: AudioOutput;
}

// The previous reading, so totals become rates over the second between them.
let last: Reading | null = null;

const mixerSection = $<HTMLDetailsElement>('audio-mixer-section');
const clockSection = $<HTMLDetailsElement>('audio-clock-section');
const outputSection = $<HTMLDetailsElement>('audio-output-section');

function layout(channels: number): string {
  if (channels === 1) return 'mono';
  if (channels === 2) return 'stereo';
  return `${channels} channels`;
}

function perSecond(now: number, before: number | undefined, seconds: number | null): string {
  if (before === undefined || !seconds) return '-';
  return `${fmtCount(Math.max(0, now - before) / seconds)}/s`;
}

function ms(seconds: number): string {
  return seconds >= 10 ? `${seconds.toFixed(1)} s` : `${Math.round(seconds * 1000)} ms`;
}

function showStatus(id: string, data: string, text: string): void {
  const status = $(id);
  status.textContent = text;
  status.hidden = !text;
  $(data).hidden = Boolean(text);
}

function voiceRow(v: AudioVoiceReport): HTMLTableRowElement {
  const name = el('th', null, `${v.id} ${v.playing ? 'playing' : 'stopped'}`);
  name.scope = 'row';
  name.append(el('span', 'thread-detail',
    `${v.format}, ${fmtCount(v.sampleRate)} Hz ${layout(v.channels)}, pitch ${v.pitch.toFixed(2)}, `
    + `volume ${v.volume.toFixed(2)}`));
  const tr = el('tr');
  tr.append(name, el('td', null, String(v.queuedBuffers)), el('td', null, fmtCount(v.underruns)));
  return tr;
}

function voicesTable(r: AudioRendererReport): HTMLElement {
  if (!r.voiceList.length) return el('p', 'muted tiny', 'No voice in use.');
  const head = el('tr');
  for (const name of ['Voice', 'Queued', 'Underruns']) {
    const th = el('th', null, name);
    th.scope = 'col';
    head.append(th);
  }
  const thead = el('thead');
  thead.append(head);
  const tbody = el('tbody');
  tbody.append(...r.voiceList.map(voiceRow));
  const table = el('table', 'debug-timings threads-table');
  table.append(el('caption', null, 'Voices in use; Queued counts wave buffers not yet played'), thead, tbody);
  return table;
}

function totals(rows: [string, string][]): HTMLDListElement {
  const dl = el('dl', 'debug-totals');
  for (const [term, value] of rows) {
    const row = el('div');
    row.append(el('dt', null, term), el('dd', null, value));
    dl.append(row);
  }
  return dl;
}

// Updates per rendered frame over the last second: 1.00 is a mixer woken for every frame.
function updatesPerFrame(r: AudioRendererReport, before: AudioRendererReport | undefined): number | null {
  if (!before) return null;
  const frames = r.renderedFrames - before.renderedFrames;
  return frames > 0 ? (r.updates - before.updates) / frames : null;
}

function rendererBlock(r: AudioRendererReport, before: AudioRendererReport | undefined, seconds: number | null): HTMLElement[] {
  const ratio = updatesPerFrame(r, before);
  const state = r.started ? '' : ', stopped';
  const sink = r.sinkChannels ? layout(r.sinkChannels) : 'no playable sink';
  const underrunRate = perSecond(r.underruns, before?.underruns, seconds);
  return [
    el('p', 'muted tiny debug-caption', `Renderer ${r.handle.toString(16)}, ${fmtCount(r.sampleRate)} Hz to ${sink}${state}`),
    totals([
      ['Voices playing', `${r.voicesPlaying} of ${r.voices}`],
      ['Frames rendered', perSecond(r.renderedFrames, before?.renderedFrames, seconds)],
      ['Frame wakes', perSecond(r.frameWakes, before?.frameWakes, seconds)],
      ['Updates from the title', perSecond(r.updates, before?.updates, seconds)],
      ['Updates per frame', ratio === null ? '-' : ratio.toFixed(2)],
      ['Wakes dropped, too far behind', fmtCount(r.frameWakesDropped)],
      ['Underruns', `${underrunRate}, ${fmtCount(r.underruns)} in all`],
    ]),
    voicesTable(r),
  ];
}

function outBlock(o: AudioOutReport, before: AudioOutReport | undefined, seconds: number | null): HTMLElement[] {
  const state = o.started ? '' : ', stopped';
  return [
    el('p', 'muted tiny debug-caption', `Audio out ${o.handle.toString(16)}, ${fmtCount(o.sampleRate)} Hz ${layout(o.channels)}${state}`),
    totals([
      ['Frames appended', perSecond(o.appendedFrames, before?.appendedFrames, seconds)],
      ['Buffers waiting', String(o.pendingBuffers)],
      ['Frames appended while stopped', fmtCount(o.discardedFrames)],
    ]),
  ];
}

function showMixer(report: AudioReport, before: AudioReport | undefined, seconds: number | null): void {
  const blocks = [
    ...report.renderers.flatMap((r) =>
      rendererBlock(r, before?.renderers.find((b) => b.handle === r.handle), seconds)),
    ...report.outputs.flatMap((o) =>
      outBlock(o, before?.outputs.find((b) => b.handle === o.handle), seconds)),
  ];
  showStatus('audio-mixer-status', 'audio-renderers',
    blocks.length ? '' : 'The title has not opened an audio renderer or output yet.');
  $('audio-renderers').replaceChildren(...blocks);
  const playing = report.renderers.find((r) => r.started);
  const ratio = playing
    ? updatesPerFrame(playing, before?.renderers.find((b) => b.handle === playing.handle))
    : null;
  setNote('audio-mixer-badge', ratio === null ? '' : `${ratio.toFixed(2)} updates per frame`);
}

function showClock(report: AudioReport, before: AudioReport | undefined, seconds: number | null): void {
  showStatus('audio-clock-status', 'audio-clock-data', '');
  const { hz, jumps, skipped, ahead } = report.clock;
  const skippedPerSecond = before && seconds && hz
    ? (skipped - before.clock.skipped) / hz / seconds
    : null;
  $('audio-clock-jumps').textContent = perSecond(jumps, before?.clock.jumps, seconds);
  $('audio-clock-skipped').textContent = skippedPerSecond === null
    ? '-'
    : `${ms(skippedPerSecond)} per second, ${ms(hz ? skipped / hz : 0)} in all`;
  $('audio-clock-ahead').textContent = hz ? ms(ahead / hz) : '-';
  setNote('audio-clock-badge', skippedPerSecond === null ? '' : skippedPerSecond > 0 ? 'falling behind' : 'keeping up');
}

// Seconds of sound per second of real time, from interleaved sample counts.
function soundRate(now: number, before: number | undefined, report: AudioReport, seconds: number | null): string {
  const { sampleRate, channels } = report.ring;
  if (before === undefined || !seconds || !sampleRate || !channels) return '-';
  return `${((now - before) / channels / sampleRate / seconds).toFixed(2)} s per second`;
}

function showOutput(reading: Reading, before: Reading | null, seconds: number | null): void {
  const { report, page } = reading;
  const { ring } = report;
  showStatus('audio-output-status', 'audio-output-data', '');
  const frames = (samples: number) => ring.sampleRate && ring.channels
    ? ms(samples / ring.channels / ring.sampleRate)
    : '-';
  $('audio-ring-format').textContent = ring.sampleRate
    ? `${fmtCount(ring.sampleRate)} Hz ${layout(ring.channels)}`
    : 'nothing played yet';
  $('audio-ring-produced').textContent = soundRate(ring.produced, before?.report.ring.produced, report, seconds);
  $('audio-ring-taken').textContent = soundRate(ring.taken, before?.report.ring.taken, report, seconds);
  $('audio-ring-backlog').textContent = frames(ring.backlog);
  $('audio-ring-dropped').textContent = `${frames(ring.dropped)} in all`;

  const missing = page.missing === 'unavailable'
    ? 'This browser could not open an audio output, so the sound is off.'
    : page.missing === 'waiting'
      ? 'The browser output opens with the first sound the title plays.'
      : '';
  showStatus('audio-page-status', 'audio-page-data', missing);
  if (missing) {
    setNote('audio-output-badge', '');
    return;
  }
  const latency = page.outputLatency
    ? `${ms(page.baseLatency)}, ${ms(page.outputLatency)} to the speakers`
    : ms(page.baseLatency);
  const playedRate = before && seconds ? (page.played - before.page.played) / seconds : null;
  $('audio-page-format').textContent = `${fmtCount(page.sampleRate)} Hz, ${page.state ?? 'closed'}`;
  $('audio-page-latency').textContent = latency;
  $('audio-page-played').textContent = playedRate === null ? '-' : `${playedRate.toFixed(2)} s per second`;
  $('audio-page-gaps').textContent = `${fmtCount(page.gaps)}, ${ms(page.silence)} silent in all`;
  $('audio-page-trimmed').textContent = `${ms(page.trimmed)} in all`;
  $('audio-page-queued').textContent = ms(page.queued);
  setNote('audio-output-badge', page.state === 'suspended' ? 'held back by the browser' : '');
}

function showAllStatus(text: string): void {
  showStatus('audio-mixer-status', 'audio-renderers', text);
  showStatus('audio-clock-status', 'audio-clock-data', text);
  showStatus('audio-output-status', 'audio-output-data', text);
  for (const badge of ['audio-mixer-badge', 'audio-clock-badge', 'audio-output-badge']) setNote(badge, '');
}

export async function updateAudioDebug(): Promise<void> {
  if (!titleBooted() || !hasSession()) {
    last = null;
    showAllStatus('Start a title to see its audio.');
    return;
  }
  // The worker answers between run slices, which a busy title stretches to seconds.
  if (!last) showAllStatus('Waiting for the first reading.');
  let report: AudioReport;
  try {
    report = await call('audioReport');
  } catch (err) {
    showAllStatus(`Couldn't read the audio: ${String(err)}`);
    return;
  }
  const reading: Reading = { at: performance.now(), report, page: audioOutput() };
  // Totals that went backwards belong to a title that has since been replaced.
  const before = last && report.ring.produced >= last.report.ring.produced ? last : null;
  const seconds = before ? (reading.at - before.at) / 1000 : null;
  showMixer(report, before?.report, seconds);
  showClock(report, before?.report, seconds);
  showOutput(reading, before, seconds);
  last = reading;
}

// Rendered, so its panel, tab, group and section are all open.
function onScreen(node: HTMLElement): boolean {
  return node.offsetParent !== null;
}

for (const section of [mixerSection, clockSection, outputSection]) {
  section.addEventListener('toggle', () => {
    if (section.open && onScreen(section)) void updateAudioDebug();
  });
}

setInterval(() => {
  if ([mixerSection, clockSection, outputSection].some(onScreen)) void updateAudioDebug();
}, 1000);
