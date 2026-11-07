"use strict";
// npu-sim live: binds GET /api/state (npuperf/live.py) to the page.
//   telemetry  npu-sim live snapshot (schema_version 1, src/main.cpp snapshot_json)
//   workload   shape and weight format resolved from the snapshot config
//   machine    peak rates, ridge point and analytic bound of the running design
//   slices     counter deltas over the last advances (strip chart)
//   timeline   stage transitions sampled inside each advance (pipeline chart)
// Each panel renders only from fields that are present; a missing field shows as a dash.

const $ = id => document.getElementById(id);
const SVG_NS = "http://www.w3.org/2000/svg";
const STRIP_SLOTS = 200;
const STRIP_MEAN = 8;
const GANTT_TILES = 6;
const TILE_LIMIT = 4096;
const TILE_MAP_HEIGHT = 168;
const STAGES = ["load", "compute", "store"];
const BOUND_NAMES = {compute: "compute", bandwidth: "DRAM bandwidth", latency: "DMA latency"};

let state = null;
let busy = false;
let profiles = {};
let stripHover = null;  // slice index under the pointer

// Formatting

const isNum = value => typeof value === "number" && Number.isFinite(value);
const int = value => isNum(value) ? Math.round(value).toLocaleString("en-US") : "—";
const fixed = (value, digits = 1) => isNum(value)
  ? value.toLocaleString("en-US", {minimumFractionDigits: digits, maximumFractionDigits: digits})
  : "—";
const percent = (value, digits = 1) => isNum(value) ? `${fixed(100 * value, digits)}%` : "—";

function sig(value) {
  if (!isNum(value)) return "—";
  const size = Math.abs(value);
  if (size >= 100) return int(value);
  if (size >= 10) return fixed(value, 1);
  if (size >= 1) return fixed(value, 2);
  return value.toPrecision(2);
}

function bytes(value) {
  if (!isNum(value)) return "—";
  if (value < 1024) return `${int(value)} B`;
  if (value < 1048576) return `${fixed(value / 1024, 1)} KiB`;
  return `${fixed(value / 1048576, 2)} MiB`;
}

function ratio(numerator, denominator) {
  return isNum(numerator) && isNum(denominator) && denominator > 0 ? numerator / denominator : null;
}

function text(id, value) {
  $(id).textContent = value;
}

// SVG helpers

function node(tag, attributes = {}, parent = null, content = null) {
  const item = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attributes)) item.setAttribute(name, value);
  if (content !== null) item.textContent = content;
  if (parent) parent.append(item);
  return item;
}

function canvas(id) {
  const svg = $(id);
  const width = Math.max(240, Math.round(svg.clientWidth || svg.parentElement.clientWidth || 480));
  const height = Math.round(svg.clientHeight || 200);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.replaceChildren();
  return {svg, width, height};
}

function empty(svg, width, height, label) {
  node("text", {x: width / 2, y: height / 2, "text-anchor": "middle", class: "empty"}, svg, label);
}

function tip(item, label) {
  node("title", {}, item, label);
  return item;
}

// Derived job metrics

function jobMetrics(t) {
  const traffic = isNum(t.read_bytes) && isNum(t.write_bytes) ? t.read_bytes + t.write_bytes : null;
  return {
    cycles: t.cycles,
    macRate: ratio(t.macs, t.cycles),
    dramBytes: traffic,
    dramRate: ratio(traffic, t.cycles),
    intensity: ratio(t.macs, traffic),
  };
}

function currentJob(value) {
  const t = value.telemetry || {};
  return t.completed ? value.jobs_completed : value.jobs_completed + 1;
}

function totalMacs(workload) {
  const {m, n, k, groups} = workload;
  return isNum(m) && isNum(n) && isNum(k) ? m * n * k * (isNum(groups) ? groups : 1) : null;
}

// Title bar

function renderMachine(value) {
  const workload = value.workload || {};
  const machine = value.machine || {};
  const parts = [];
  if (isNum(workload.m)) {
    const groups = isNum(workload.groups) && workload.groups > 1 ? ` g${workload.groups}` : "";
    parts.push(`gemm ${workload.m}×${workload.n}×${workload.k}${groups}`);
  }
  if (isNum(machine.tile_m)) {
    const depth = machine.tile_k || workload.k;
    parts.push(`tile ${machine.tile_m}×${machine.tile_n}×${depth ?? "k"}`);
  }
  if (isNum(machine.buffer_slots)) parts.push(`${machine.buffer_slots} slots`);
  if (isNum(machine.macs_per_cycle)) parts.push(`${machine.macs_per_cycle} MAC/cyc`);
  if (isNum(machine.bytes_per_cycle)) parts.push(`${sig(machine.bytes_per_cycle)} B/cyc`);
  if (isNum(machine.memory_latency) && machine.memory_model === "serial")
    parts.push(`${machine.memory_latency} cyc DMA latency`);
  else if (machine.memory_model) parts.push(`${machine.memory_model} memory`);
  if (isNum(machine.cores) && machine.cores > 1) parts.push(`${machine.cores} cores`);
  if (isNum(workload.weight_bits)) parts.push(`int${workload.weight_bits} ${workload.weight_encoding ?? ""}`.trim());
  const clock = isNum(workload.frequency_mhz) ? workload.frequency_mhz : machine.frequency_mhz;
  if (isNum(clock)) parts.push(`${int(clock)} MHz`);
  text("machine", parts.length ? parts.join(" · ") : "—");
}

function renderControls(value) {
  const live = ["running", "paused"].includes(value.state);
  const pill = $("run-state");
  pill.dataset.state = value.state;
  pill.textContent = value.state;
  if (live && value.profile in profiles) $("profile").value = value.profile;
  $("profile").disabled = busy || live;
  $("start").disabled = busy || live || !$("profile").value;
  $("pause").disabled = busy || !live;
  $("pause").textContent = value.state === "paused" ? "Resume" : "Pause";
  $("step").disabled = busy || value.state !== "paused";
  $("step-cycles").disabled = busy || value.state !== "paused";
  $("stop").disabled = busy || !["running", "paused", "error"].includes(value.state);
  $("reset").disabled = busy || !$("profile").value;
  const limit = isNum(value.max_step) ? value.max_step : Infinity;
  for (const option of $("step-cycles").options) option.disabled = Number(option.value) > limit;
}

function showMessage(textValue) {
  const line = $("message");
  line.textContent = textValue || "";
  line.hidden = !textValue;
}

// KPI strip

function renderKpis(value) {
  const t = value.telemetry || {};
  const workload = value.workload || {};
  const machine = value.machine || {};
  const analysis = machine.analysis || null;
  const job = jobMetrics(t);
  const clock = isNum(workload.frequency_mhz) ? workload.frequency_mhz : machine.frequency_mhz;

  text("kpi-cycles", int(job.cycles));
  text("kpi-latency", isNum(t.modeled_latency_ns)
    ? `${fixed(t.modeled_latency_ns / 1000, 2)} µs @ ${int(clock)} MHz` : "—");

  $("kpi-mac-rate").textContent = sig(job.macRate);
  text("kpi-mac-peak", isNum(machine.macs_per_cycle)
    ? `${percent(ratio(job.macRate, machine.macs_per_cycle))} of ${machine.macs_per_cycle} peak` : "—");

  text("kpi-dram-rate", sig(job.dramRate));
  text("kpi-dram-peak", isNum(machine.bytes_per_cycle)
    ? `${percent(ratio(job.dramRate, machine.bytes_per_cycle))} of ${sig(machine.bytes_per_cycle)} peak`
    : "—");

  $("kpi-intensity").replaceChildren(sig(job.intensity));
  if (isNum(job.intensity)) {
    const unit = document.createElement("small");
    unit.textContent = "MAC/B";
    $("kpi-intensity").append(unit);
  }
  if (isNum(machine.ridge)) {
    const side = isNum(job.intensity) ? (job.intensity < machine.ridge ? " · below" : " · above") : "";
    text("kpi-ridge", `ridge ${sig(machine.ridge)}${side}`);
  } else {
    text("kpi-ridge", "—");
  }

  const last = (value.recent_jobs || [])[0];
  if (analysis) {
    text("kpi-bound", BOUND_NAMES[analysis.bound] || analysis.bound);
    const attained = last ? ratio(analysis.resource_bound_cycles, last.telemetry.cycles) : null;
    text("kpi-bound-detail", `${int(analysis.resource_bound_cycles)} cyc` +
      (attained === null ? " analytic" : ` · ${percent(attained)} attained`));
  } else if (isNum(t.compute_busy_cycles) && isNum(t.dma_busy_cycles) && t.cycles > 0) {
    text("kpi-bound", t.compute_busy_cycles >= t.dma_busy_cycles ? "compute" : "DMA");
    text("kpi-bound-detail", `measured · MAC ${percent(t.compute_busy_cycles / t.cycles, 0)}` +
      ` DMA ${percent(t.dma_busy_cycles / t.cycles, 0)}`);
  } else {
    text("kpi-bound", "—");
    text("kpi-bound-detail", "—");
  }

  text("kpi-jobs", int(value.jobs_completed));
  text("kpi-jobs-detail", "vs scalar reference");
}

// Roofline (log-log): compute roof, peak DRAM roof, latency-inclusive DRAM roof, jobs.

function renderRoofline(value) {
  const {svg, width, height} = canvas("roofline");
  const machine = value.machine || {};
  const analysis = machine.analysis || null;
  const peak = machine.macs_per_cycle;
  const bandwidth = machine.bytes_per_cycle;
  if (!isNum(peak) || !isNum(bandwidth) || bandwidth <= 0) {
    empty(svg, width, height, "no machine description");
    text("roofline-meta", "");
    return;
  }
  const ridge = peak / bandwidth;
  const effective = analysis && isNum(analysis.effective_bytes_per_cycle)
    ? analysis.effective_bytes_per_cycle : null;
  const t = value.telemetry || {};
  const current = jobMetrics(t);
  const done = (value.recent_jobs || []).map(job => ({job: job.job, ...jobMetrics(job.telemetry)}));
  const boundPoint = analysis && isNum(analysis.resource_bound_cycles) && totalMacs(value.workload || {})
    ? {intensity: analysis.arithmetic_intensity,
       macRate: totalMacs(value.workload) / analysis.resource_bound_cycles}
    : null;

  const points = [...done, current, boundPoint].filter(p => p && isNum(p.intensity) && isNum(p.macRate)
                                                       && p.intensity > 0 && p.macRate > 0);
  const xs = [ridge, ...points.map(p => p.intensity)];
  let x0 = Math.floor(Math.log10(Math.min(...xs))) - 1;
  let x1 = Math.ceil(Math.log10(Math.max(...xs))) + 1;
  if (x1 - x0 < 3) x1 = x0 + 3;
  const y1 = Math.ceil(Math.log10(peak * 1.25));
  const lowest = Math.min(...points.map(p => p.macRate), peak / 100);
  const y0 = Math.max(y1 - 5, Math.floor(Math.log10(lowest)));

  const pad = {left: 46, right: 12, top: 8, bottom: 30};
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const sx = v => pad.left + (Math.log10(v) - x0) / (x1 - x0) * plotW;
  const sy = v => pad.top + (1 - (Math.log10(v) - y0) / (y1 - y0)) * plotH;
  const label = exponent => {
    const v = 10 ** exponent;
    return v >= 1 ? int(v) : String(Number(v.toPrecision(1)));
  };

  for (let e = x0; e <= x1; e++) {
    node("line", {x1: sx(10 ** e), x2: sx(10 ** e), y1: pad.top, y2: pad.top + plotH, class: "grid"}, svg);
    node("text", {x: sx(10 ** e), y: pad.top + plotH + 13, "text-anchor": "middle"}, svg, label(e));
  }
  for (let e = y0; e <= y1; e++) {
    node("line", {x1: pad.left, x2: pad.left + plotW, y1: sy(10 ** e), y2: sy(10 ** e), class: "grid"}, svg);
    node("text", {x: pad.left - 6, y: sy(10 ** e) + 3.5, "text-anchor": "end"}, svg, label(e));
  }
  node("rect", {x: pad.left, y: pad.top, width: plotW, height: plotH, fill: "none", class: "axis"}, svg);
  node("text", {x: pad.left + plotW, y: height - 3, "text-anchor": "end", class: "axis-title"}, svg,
       "arithmetic intensity, MAC/B");
  node("text", {x: 0, y: 0, transform: `translate(11 ${pad.top + plotH / 2}) rotate(-90)`,
                "text-anchor": "middle", class: "axis-title"}, svg, "MAC/cycle");

  const xMin = 10 ** x0, xMax = 10 ** x1, yMin = 10 ** y0;
  const roof = rate => {
    const start = Math.max(xMin, yMin / rate);
    const knee = peak / rate;
    const path = [`M${sx(start)},${sy(Math.min(peak, rate * start))}`];
    if (knee > start && knee < xMax) path.push(`L${sx(knee)},${sy(peak)}`);
    path.push(`L${sx(xMax)},${sy(Math.min(peak, rate * xMax))}`);
    return path.join("");
  };
  node("line", {x1: sx(ridge), x2: sx(ridge), y1: sy(peak), y2: pad.top + plotH, class: "ridge"}, svg);
  if (effective && effective < bandwidth)
    tip(node("path", {d: roof(effective), class: "roof-effective"}, svg),
        `latency-inclusive DRAM roof: ${sig(effective)} B/cycle`);
  tip(node("path", {d: roof(bandwidth), class: "roof"}, svg),
      `roofs: ${peak} MAC/cycle, ${sig(bandwidth)} B/cycle`);
  node("text", {x: pad.left + plotW - 4, y: sy(peak) - 5, "text-anchor": "end"}, svg,
       `${peak} MAC/cyc`);
  node("text", {x: sx(ridge) + 5, y: pad.top + plotH - 5}, svg, `ridge ${sig(ridge)}`);

  for (const point of done) {
    if (!points.includes(point)) continue;
    tip(node("circle", {cx: sx(point.intensity), cy: sy(point.macRate), r: 4, class: "point-done"}, svg),
        `job ${point.job}: ${sig(point.intensity)} MAC/B, ${sig(point.macRate)} MAC/cycle`);
  }
  if (boundPoint && points.includes(boundPoint)) {
    const x = sx(boundPoint.intensity), y = sy(boundPoint.macRate);
    tip(node("path", {d: `M${x},${y - 5.5}L${x + 5.5},${y}L${x},${y + 5.5}L${x - 5.5},${y}Z`,
                      class: "point-bound"}, svg),
        `analytic bound: ${sig(boundPoint.macRate)} MAC/cycle`);
  }
  if (points.includes(current))
    tip(node("circle", {cx: sx(current.intensity), cy: sy(current.macRate), r: 5, class: "point-current"}, svg),
        `current job: ${sig(current.intensity)} MAC/B, ${sig(current.macRate)} MAC/cycle`);

  $("legend-effective").hidden = !(effective && effective < bandwidth);
  $("legend-bound").hidden = !(boundPoint && points.includes(boundPoint));
  text("roofline-meta", `${peak} MAC/cyc · ${sig(bandwidth)} B/cyc` +
    (effective ? ` · ${sig(effective)} B/cyc with latency` : ""));
}

// Utilisation strip chart over the last STRIP_SLOTS slices.

const STRIP_SERIES = [
  {key: "compute_busy_fraction", className: "series-mac", legend: "strip-mac"},
  {key: "dma_read_busy_fraction", className: "series-read", legend: "strip-read"},
  {key: "dma_write_busy_fraction", className: "series-write", legend: "strip-write"},
];

function renderStrip(value) {
  const {svg, width, height} = canvas("strip");
  const slices = Array.isArray(value.slices) ? value.slices.slice(-STRIP_SLOTS) : [];
  const pad = {left: 40, right: 8, top: 8, bottom: 22};
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const step = plotW / STRIP_SLOTS;
  const offset = STRIP_SLOTS - slices.length;
  const sx = index => pad.left + (offset + index) * step;
  const sy = fraction => pad.top + (1 - Math.min(1, Math.max(0, fraction))) * plotH;

  for (const level of [0, 0.25, 0.5, 0.75, 1]) {
    node("line", {x1: pad.left, x2: pad.left + plotW, y1: sy(level), y2: sy(level),
                  class: level === 0 ? "axis" : "grid"}, svg);
    node("text", {x: pad.left - 6, y: sy(level) + 3.5, "text-anchor": "end"}, svg, `${100 * level}%`);
  }
  if (!slices.length) {
    empty(svg, width, height, "no slices yet");
    for (const series of STRIP_SERIES) text(series.legend, "—");
    text("strip-meta", `last ${STRIP_SLOTS} slices`);
    return;
  }
  slices.forEach((slice, index) => {
    if (index && slice.job !== slices[index - 1].job) {
      node("line", {x1: sx(index), x2: sx(index), y1: pad.top, y2: pad.top + plotH, class: "job-edge"}, svg);
      node("text", {x: sx(index) + 3, y: pad.top + 9}, svg, `job ${slice.job}`);
    }
  });
  for (const series of STRIP_SERIES) {
    const raw = [], mean = [];
    slices.forEach((slice, index) => {
      const fraction = slice[series.key];
      if (!isNum(fraction)) return;
      const y = sy(fraction);
      raw.push(`${raw.length ? "L" : "M"}${sx(index)},${y}`, `L${sx(index + 1)},${y}`);
      // Cycle-weighted mean over the last STRIP_MEAN slices: busy cycles over elapsed cycles.
      let busyCycles = 0, cycles = 0;
      for (const earlier of slices.slice(Math.max(0, index - STRIP_MEAN + 1), index + 1)) {
        if (!isNum(earlier[series.key]) || !isNum(earlier.cycles)) continue;
        busyCycles += earlier[series.key] * earlier.cycles;
        cycles += earlier.cycles;
      }
      if (cycles) mean.push(`${mean.length ? "L" : "M"}${sx(index + 0.5)},${sy(busyCycles / cycles)}`);
    });
    if (raw.length) node("path", {d: raw.join(""), class: `series raw ${series.className}`}, svg);
    if (mean.length) node("path", {d: mean.join(""), class: `series ${series.className}`}, svg);
  }
  const first = slices[0], last = slices[slices.length - 1];
  node("text", {x: pad.left, y: height - 5}, svg, `job ${first.job} @ ${int(first.end_cycle - first.cycles)}`);
  node("text", {x: pad.left + plotW, y: height - 5, "text-anchor": "end"}, svg,
       `job ${last.job} @ ${int(last.end_cycle)}`);

  const shown = stripHover !== null && stripHover < slices.length ? slices[stripHover] : last;
  if (shown !== last) {
    const index = slices.indexOf(shown);
    node("line", {x1: sx(index + 0.5), x2: sx(index + 0.5), y1: pad.top, y2: pad.top + plotH, class: "cursor"}, svg);
  }
  for (const series of STRIP_SERIES) text(series.legend, percent(shown[series.key]));
  text("strip-meta", `job ${shown.job} · cycles ${int(shown.end_cycle - shown.cycles)}–${int(shown.end_cycle)}` +
    ` · line: ${STRIP_MEAN}-slice mean`);

  svg.onmousemove = event => {
    const box = svg.getBoundingClientRect();
    const x = (event.clientX - box.left) * width / box.width;
    const index = Math.floor((x - pad.left) / step) - offset;
    const next = index >= 0 && index < slices.length ? index : null;
    if (next !== stripHover) {
      stripHover = next;
      renderStrip(state);
    }
  };
  svg.onmouseleave = () => {
    stripHover = null;
    renderStrip(state);
  };
}

// Pipeline timeline: one lane per stage, one segment per tile, from stage transitions.

function segments(transitions, now) {
  const open = {}, result = [];
  for (const transition of transitions) {
    for (const stage of STAGES) {
      const tile = transition[stage];
      const current = open[stage];
      if (current && current.tile === tile) continue;
      if (current) result.push({...current, end: transition.cycle});
      open[stage] = tile === null || tile === undefined ? null : {stage, tile, start: transition.cycle};
    }
  }
  for (const stage of STAGES) if (open[stage]) result.push({...open[stage], end: now, open: true});
  return result.filter(segment => isNum(segment.start) && isNum(segment.end) && segment.end > segment.start);
}

function renderGantt(value) {
  const {svg, width, height} = canvas("gantt");
  const t = value.telemetry || {};
  const timeline = value.timeline || {};
  const job = currentJob(value);
  const transitions = (Array.isArray(timeline.transitions) ? timeline.transitions : [])
    .filter(transition => transition.job === job);
  const all = isNum(t.cycles) ? segments(transitions, t.cycles) : [];
  if (!all.length) {
    empty(svg, width, height, transitions.length ? "pipeline idle" : "no stage samples");
    text("gantt-meta", "");
    return;
  }
  const recent = [...new Set(all.map(segment => segment.tile))].sort((a, b) => b - a).slice(0, GANTT_TILES);
  const oldest = Math.min(...recent);
  const visible = all.filter(segment => segment.tile >= oldest);
  const start = Math.min(...visible.map(segment => segment.start));
  const end = Math.max(t.cycles, start + 32);

  const pad = {left: 56, right: 8, top: 4, bottom: 20};
  const laneH = (height - pad.top - pad.bottom) / STAGES.length;
  const plotW = width - pad.left - pad.right;
  const sx = cycle => pad.left + (cycle - start) / (end - start) * plotW;

  STAGES.forEach((stage, lane) => {
    const y = pad.top + lane * laneH;
    node("rect", {x: pad.left, y: y + 2, width: plotW, height: laneH - 4, class: "lane"}, svg);
    node("text", {x: 0, y: y + laneH / 2 + 3.5}, svg, stage);
  });
  for (const segment of visible) {
    const lane = STAGES.indexOf(segment.stage);
    const x = sx(Math.max(start, segment.start));
    const w = Math.max(1, sx(segment.end) - x - 1);
    const y = pad.top + lane * laneH + 2;
    const block = node("rect", {x, y, width: w, height: laneH - 4,
                                class: `seg-${segment.stage}${segment.tile % 2 ? " seg-odd" : ""}`}, svg);
    tip(block, `tile ${segment.tile} ${segment.stage}: cycles ${int(segment.start)}–${int(segment.end)}` +
      ` (${int(segment.end - segment.start)})${segment.open ? ", in progress" : ""}`);
    if (w >= 22) node("text", {x: x + w / 2, y: y + laneH / 2 + 1.5, "text-anchor": "middle",
                               class: "seg-label"}, svg, String(segment.tile));
  }
  node("line", {x1: sx(t.cycles), x2: sx(t.cycles), y1: pad.top, y2: height - pad.bottom, class: "now"}, svg);
  const span = end - start;
  const tick = 10 ** Math.floor(Math.log10(span / 4));
  const unit = [1, 2, 5, 10].map(f => f * tick).find(u => span / u <= 6) || tick * 10;
  for (let c = Math.ceil(start / unit) * unit; c <= end; c += unit)
    node("text", {x: sx(c), y: height - 5, "text-anchor": "middle"}, svg, int(c));
  const stride = isNum(timeline.stride_cycles) ? ` · ${timeline.stride_cycles}-cycle samples` : "";
  text("gantt-meta", `job ${job} · tiles ${oldest}–${Math.max(...recent)} · cycles ${int(start)}–${int(t.cycles)}${stride}`);
}

// Output tile map, coloured by stage.

const TILE_STATES = ["pending", "resident", "loading", "computing", "storing", "stored"];

function renderTiles(value) {
  // The grid takes the space beside the legend; cells stay square.
  const svg = $("tiles");
  const width = Math.max(120, svg.parentElement.clientWidth - $("tiles-legend").offsetWidth - 20);
  const height = TILE_MAP_HEIGHT;
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.style.width = `${width}px`;
  svg.style.height = `${height}px`;
  const t = value.telemetry || {};
  const workload = value.workload || {};
  const machine = value.machine || {};
  const legend = $("tiles-legend");
  if (!isNum(t.tiles_total) || !isNum(t.tiles_completed) || t.tiles_total <= 0) {
    empty(svg, width, height, "no tile counters");
    legend.replaceChildren();
    text("tiles-meta", "");
    return;
  }
  const total = Math.min(t.tiles_total, TILE_LIMIT);
  const columns = isNum(workload.n) && isNum(machine.tile_n) && machine.tile_n > 0
    ? Math.ceil(workload.n / machine.tile_n) : Math.ceil(Math.sqrt(total));
  const rows = Math.ceil(total / columns);
  const cell = Math.max(2, Math.min(22, Math.floor(Math.min(width / columns, (height - 2) / rows))));
  const gap = cell >= 6 ? 1 : 0;
  const active = {};
  if (t.dma_busy && isNum(t.dma_tile_id)) active[t.dma_tile_id] = t.dma_read ? "loading" : "storing";
  if (t.compute_busy && isNum(t.compute_tile_id)) active[t.compute_tile_id] = "computing";
  const resident = t.tiles_completed + (isNum(t.inflight_tiles) ? t.inflight_tiles : 0);
  const counts = Object.fromEntries(TILE_STATES.map(name => [name, 0]));
  for (let id = 0; id < total; id++) {
    const status = id < t.tiles_completed ? "stored" : active[id] || (id < resident ? "resident" : "pending");
    counts[status]++;
    const x = (id % columns) * cell, y = Math.floor(id / columns) * cell;
    tip(node("rect", {x, y, width: cell - gap, height: cell - gap, rx: cell >= 8 ? 1 : 0,
                      class: `tile-${status}`}, svg), `tile ${id}: ${status}`);
  }
  svg.setAttribute("viewBox", `0 0 ${columns * cell} ${rows * cell}`);
  svg.style.width = `${columns * cell}px`;
  svg.style.height = `${rows * cell}px`;
  legend.replaceChildren(...TILE_STATES.map(name => {
    const item = document.createElement("li");
    const key = document.createElement("i");
    key.className = `key cell tile-${name}`;
    const count = document.createElement("b");
    count.textContent = int(counts[name]);
    item.append(key, `${name} `, count);
    return item;
  }));
  text("tiles-meta", `${int(t.tiles_completed)} / ${int(t.tiles_total)} stored · ${columns}×${rows}` +
    (t.tiles_total > TILE_LIMIT ? ` · first ${TILE_LIMIT}` : ""));
}

// Job table

function cellRow(values, classes = []) {
  const row = document.createElement("tr");
  values.forEach((content, index) => {
    const cell = document.createElement("td");
    cell.textContent = content;
    if (classes[index]) cell.className = classes[index];
    row.append(cell);
  });
  return row;
}

function renderJobs(value) {
  const machine = value.machine || {};
  const analysis = machine.analysis || null;
  const workload = value.workload || {};
  const clock = isNum(workload.frequency_mhz) ? workload.frequency_mhz : machine.frequency_mhz;
  const recent = value.recent_jobs || [];
  const body = $("jobs").tBodies[0];
  const foot = $("jobs").tFoot;
  if (!recent.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 9;
    cell.className = "empty";
    cell.textContent = "no completed jobs";
    row.append(cell);
    body.replaceChildren(row);
    foot.replaceChildren();
    text("jobs-meta", "");
    return;
  }
  body.replaceChildren(...recent.map(job => {
    const t = job.telemetry;
    const metrics = jobMetrics(t);
    const bound = analysis ? ratio(analysis.resource_bound_cycles, t.cycles) : null;
    return cellRow([
      String(job.job), int(t.cycles), fixed(ratio(t.cycles, clock), 2), sig(metrics.macRate),
      percent(ratio(metrics.macRate, machine.macs_per_cycle)), bound === null ? "—" : percent(bound),
      int(metrics.dramBytes), sig(metrics.dramRate), t.validated === true ? "bit-exact" : "—",
    ], ["", "", "", "", "", "", "", "", "pass"]);
  }));
  const cycles = value.completed_job_cycles, macs = value.completed_job_macs;
  foot.replaceChildren(cellRow([
    `all ${int(value.jobs_completed)}`, int(cycles), "", sig(ratio(macs, cycles)),
    percent(ratio(ratio(macs, cycles), machine.macs_per_cycle)), "", "", "", "",
  ]));
  text("jobs-meta", `last ${recent.length} · ${value.profile}`);
}

// Counters: known snapshot fields first, then any other numeric field and any nested
// object or array of objects (memory channels, per-core state) under its own name.

const COUNTERS = [
  ["macs", "MACs", int], ["executed_macs", "MACs executed", int], ["skipped_macs", "MACs skipped", int],
  ["compute_busy_cycles", "MAC busy cycles", int], ["dma_busy_cycles", "DMA busy cycles", int],
  ["read_bytes", "DRAM read", bytes], ["write_bytes", "DRAM write", bytes],
  ["weight_read_bytes", "weight read", bytes], ["weight_decode_cycles", "decode cycles", int],
  ["decoded_weight_values", "decoded values", int],
  ["encoded_weight_storage_bytes", "weights encoded", bytes],
  ["dense_weight_storage_bytes", "weights dense", bytes],
  ["loaded_panels", "panels loaded", int], ["panels_completed", "panels computed", int],
  ["panels_total", "panels total", int], ["tiles_completed", "tiles stored", int],
  ["tiles_total", "tiles total", int], ["inflight_tiles", "tiles in flight", int],
  ["read_queue_depth", "read queue", int], ["write_queue_depth", "write queue", int],
];
const NOT_COUNTERS = new Set([
  "schema_version", "type", "simulation", "config", "cycles", "completed", "validated", "numerics",
  "numerical_match", "compute_utilization", "memory_utilization", "dma_busy", "compute_busy",
  "dma_read", "dma_tile_id", "dma_block_id", "compute_tile_id", "compute_block_id",
  "modeled_latency_ns", "active_stages", "output_hash", "input_preview", "output_preview",
]);

function pair(label, content) {
  const wrap = document.createElement("div");
  const term = document.createElement("dt");
  const data = document.createElement("dd");
  term.textContent = label;
  data.textContent = content;
  wrap.append(term, data);
  return wrap;
}

function leaves(object, prefix = "", out = []) {
  for (const [key, item] of Object.entries(object)) {
    const name = prefix ? `${prefix}.${key}` : key;
    if (item && typeof item === "object" && !Array.isArray(item)) leaves(item, name, out);
    else if (isNum(item) || typeof item === "string" || typeof item === "boolean") out.push([name, item]);
  }
  return out;
}

const show = item => isNum(item) ? (Number.isInteger(item) ? int(item) : sig(item)) : String(item);

function renderCounters(value) {
  const t = value.telemetry || {};
  const known = new Set(COUNTERS.map(([key]) => key));
  const items = COUNTERS.filter(([key]) => isNum(t[key])).map(([key, label, format]) => pair(label, format(t[key])));
  for (const [key, item] of Object.entries(t))
    if (!known.has(key) && !NOT_COUNTERS.has(key) && isNum(item)) items.push(pair(key, show(item)));
  $("counters").replaceChildren(...items);

  const groups = [];
  for (const [key, item] of Object.entries(t)) {
    if (NOT_COUNTERS.has(key) || !item || typeof item !== "object") continue;
    const heading = document.createElement("h3");
    heading.textContent = key;
    if (Array.isArray(item) && item.length && item.every(entry => entry && typeof entry === "object")) {
      const rows = item.map(entry => Object.fromEntries(leaves(entry)));
      const columns = [...new Set(rows.flatMap(row => Object.keys(row)))];
      const table = document.createElement("table");
      table.className = "data";
      const head = document.createElement("tr");
      for (const column of ["#", ...columns]) {
        const cell = document.createElement("th");
        cell.textContent = column;
        head.append(cell);
      }
      table.createTHead().append(head);
      const body = table.createTBody();
      rows.forEach((row, index) => body.append(cellRow([String(index),
        ...columns.map(column => column in row ? show(row[column]) : "—")])));
      const scroll = document.createElement("div");
      scroll.className = "scroll";
      scroll.append(table);
      groups.push(heading, scroll);
    } else if (!Array.isArray(item)) {
      const list = document.createElement("dl");
      list.className = "counters";
      list.append(...leaves(item).map(([name, entry]) => pair(name, show(entry))));
      groups.push(heading, list);
    }
  }
  $("extra").replaceChildren(...groups);
}

function renderPreviews(value) {
  const t = value.telemetry || {};
  const last = (value.recent_jobs || [])[0];
  const output = last ? last.telemetry.output_preview : null;
  text("input-preview", Array.isArray(t.input_preview) ? t.input_preview.join(" ") : "—");
  text("output-preview", Array.isArray(output) && output.length ? output.join(" ") : "—");
}

// State

function render(value) {
  state = value;
  const t = value.telemetry || {};
  const schemaOk = t.schema_version === undefined || t.schema_version === 1;
  renderControls(value);
  if (schemaOk) {
    renderMachine(value);
    renderKpis(value);
    renderRoofline(value);
    renderStrip(value);
    renderGantt(value);
    renderTiles(value);
    renderCounters(value);
    renderPreviews(value);
  }
  renderJobs(value);
  if (!schemaOk) showMessage(`unsupported snapshot schema_version ${t.schema_version}; expected 1`);
  else showMessage(value.error || "");
  text("connection", `${value.profile} · ${int(value.slice_cycles)} cycles per tick`);
  text("refreshed", new Date().toLocaleTimeString("en-GB"));
}

async function control(action) {
  if (busy) return;
  busy = true;
  if (state) renderControls(state);
  let failure = "";
  try {
    const body = {action};
    if (action === "start" || action === "reset") body.profile = $("profile").value;
    if (action === "step") body.cycles = Number($("step-cycles").value);
    const response = await fetch("/api/control", {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || `control request failed (${response.status})`);
    busy = false;
    render(value);
  } catch (error) {
    failure = error.message;
  } finally {
    busy = false;
    if (state) renderControls(state);
    if (failure) showMessage(failure);
  }
}

async function loadProfiles() {
  try {
    const response = await fetch("/api/profiles", {cache: "no-store"});
    if (!response.ok) throw new Error(`profiles: HTTP ${response.status}`);
    profiles = await response.json();
  } catch (error) {
    showMessage(`cannot load profiles: ${error.message}`);
    return;
  }
  const select = $("profile");
  select.replaceChildren(...Object.entries(profiles).map(([name, info]) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = info.bound ? `${info.label} · ${info.bound}-bound` : info.label;
    option.title = info.description;
    return option;
  }));
  if (state && state.profile in profiles) select.value = state.profile;
  if (state) renderControls(state);
}

$("start").onclick = () => control("start");
$("pause").onclick = () => control(state && state.state === "paused" ? "resume" : "pause");
$("step").onclick = () => control("step");
$("stop").onclick = () => control("stop");
$("reset").onclick = () => control("reset");
window.addEventListener("resize", () => { if (state) render(state); });

async function poll() {
  try {
    const response = await fetch("/api/state", {cache: "no-store"});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const value = await response.json();
    if (!busy) render(value);
  } catch (error) {
    const pill = $("run-state");
    pill.dataset.state = "offline";
    pill.textContent = "offline";
    showMessage(`controller unreachable: ${error.message}`);
  }
  setTimeout(poll, document.hidden ? 1500 : 250);
}

loadProfiles().then(poll);
