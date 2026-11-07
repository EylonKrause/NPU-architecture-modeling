"use strict";
// Binds the controller state (npuperf/live.py) and the npu-sim live snapshot
// (schema_version 1, src/main.cpp snapshot_json) to the page.

const element = id => document.getElementById(id);
const format = value => typeof value === "number" ? value.toLocaleString("en-US") : "—";
let state = null;
let busy = false;
let previous = null;  // last telemetry, for per-slice utilisation

function bytes(value) {
  if (typeof value !== "number") return "—";
  if (value < 1024) return `${format(value)} B`;
  if (value < 1048576) return `${(value / 1024).toFixed(1)} KiB`;
  return `${(value / 1048576).toFixed(1)} MiB`;
}

function message(text, error = false) {
  element("message").textContent = text;
  element("message").classList.toggle("error", error);
}

// Config value by name: a flat key or a leaf of a nested config object.
function configValue(config, name) {
  if (!config || typeof config !== "object") return undefined;
  if (name in config && typeof config[name] !== "object") return config[name];
  for (const value of Object.values(config)) {
    const found = configValue(value, name);
    if (found !== undefined) return found;
  }
  return undefined;
}

// Busy fraction over the last slice; falls back to the whole job at a job boundary.
function sliceFraction(t, key) {
  if (previous && t.cycles > previous.cycles && t[key] >= previous[key])
    return (t[key] - previous[key]) / (t.cycles - previous.cycles);
  return t.cycles ? t[key] / t.cycles : 0;
}

function renderTelemetry(t) {
  const config = t.config || {};
  element("cycles").textContent = format(t.cycles);
  element("macs").textContent = format(t.macs);
  const traffic = typeof t.read_bytes === "number" && typeof t.write_bytes === "number";
  element("traffic").textContent = traffic ? bytes(t.read_bytes + t.write_bytes) : "—";
  element("traffic-detail").textContent = `${bytes(t.read_bytes)} read / ${bytes(t.write_bytes)} write`;
  const m = configValue(config, "m"), n = configValue(config, "n"), k = configValue(config, "k");
  const bits = configValue(config, "weight_bits"), encoding = configValue(config, "weight_encoding");
  element("workload-shape").textContent = m
    ? `${m} × ${n} × ${k} · INT${bits || 8} weights · ${encoding || "dense"}` : "—";
  element("compression-detail").textContent = encoding && encoding !== "dense"
    ? `${encoding} · ${bytes(t.encoded_weight_storage_bytes)} encoded / ` +
      `${bytes(t.dense_weight_storage_bytes)} dense weights · ` +
      `${format(t.weight_decode_cycles)} decode cycles · ${format(t.skipped_macs)} MACs skipped`
    : "Dense weights";
  element("target-time").textContent = typeof t.modeled_latency_ns === "number"
    ? `${format(Math.round(t.modeled_latency_ns))} ns at ${format(configValue(config, "frequency_mhz"))} MHz`
    : "—";
  for (const [name, key] of [["compute", "compute_busy_cycles"], ["dma", "dma_busy_cycles"]]) {
    const percent = Math.min(100, 100 * sliceFraction(t, key));
    element(`${name}-util`).textContent = t.cycles ? `${percent.toFixed(1)}%` : "—";
    element(`${name}-bar`).value = percent;
  }
  const active = Array.isArray(t.active_stages) ? t.active_stages : [];
  for (const name of ["load", "compute", "store"])
    element(`stage-${name}`).classList.toggle("active", active.includes(name));
  element("load-count").textContent = `${bytes(t.read_bytes)} transferred`;
  element("compute-count").textContent = `${format(t.macs)} MACs complete`;
  element("store-count").textContent = `${bytes(t.write_bytes)} transferred`;
  const container = element("tiles");
  container.replaceChildren();
  if (typeof t.tiles_total === "number" && typeof t.tiles_completed === "number") {
    const shown = Math.min(t.tiles_total, 256);
    for (let index = 0; index < shown; index++) {
      const tile = document.createElement("span");
      const done = index < t.tiles_completed;
      tile.className = "tile" + (done ? " done" : "");
      tile.title = `Output tile ${index}: ${done ? "stored" : "pending"}`;
      container.append(tile);
    }
    element("tiles-label").textContent =
      `${t.tiles_completed} / ${t.tiles_total} stored${t.tiles_total > 256 ? " · first 256 shown" : ""}`;
  } else {
    element("tiles-label").textContent = "—";
  }
  element("input-preview").textContent = Array.isArray(t.input_preview) ? t.input_preview.join("  ") : "—";
}

function renderHistory(value) {
  const recent = value.recent_jobs || [];
  const last = recent.length ? recent[0].telemetry : {};
  element("output-preview").textContent = Array.isArray(last.output_preview) && last.output_preview.length
    ? last.output_preview.join("  ") : "—";
  // A job enters the history only after its output matched the scalar reference.
  element("validation").textContent = recent.length ? "Bit-exact against the scalar reference" : "No completed jobs";
  element("validation").className = "badge" + (recent.length ? " valid" : "");
  if (!recent.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "No completed jobs.";
    element("history").replaceChildren(empty);
    return;
  }
  element("history").replaceChildren(...recent.map(job => {
    const row = document.createElement("div");
    row.className = "history-row";
    const number = document.createElement("span");
    number.textContent = `Job ${job.job}`;
    const detail = document.createElement("div");
    detail.textContent = `${format(job.telemetry.cycles)} cycles · ${format(job.telemetry.macs)} MACs`;
    const hash = document.createElement("code");
    hash.textContent = `output ${job.telemetry.output_hash ?? "—"}`;
    detail.append(hash);
    const verdict = document.createElement("strong");
    verdict.textContent = "BIT-EXACT";
    row.append(number, detail, verdict);
    return row;
  }));
}

function render(value) {
  state = value;
  const t = value.telemetry || {};
  if (["running", "paused"].includes(value.state)) element("profile").value = value.profile;
  element("status").textContent = value.state[0].toUpperCase() + value.state.slice(1);
  element("status-dot").className = value.state;
  const schemaOk = t.schema_version === undefined || t.schema_version === 1;
  if (schemaOk) renderTelemetry(t);
  renderHistory(value);
  element("jobs").textContent = format(value.jobs_completed);
  const slice = value.slice_cycles || 256;
  element("step").textContent = `Step ${format(slice)} cycles`;
  element("profile").disabled = busy || value.state === "running" || value.state === "paused";
  element("start").disabled = busy || value.state === "running" || value.state === "paused";
  element("pause").disabled = busy || !["running", "paused"].includes(value.state);
  element("pause").textContent = value.state === "paused" ? "Resume" : "Pause";
  element("step").disabled = busy || value.state !== "paused";
  element("stop").disabled = busy || !["running", "paused", "error"].includes(value.state);
  element("reset").disabled = busy;
  element("connection").textContent = "Local controller · refreshed " + new Date().toLocaleTimeString();
  if (!schemaOk) message(`Unsupported snapshot schema_version ${t.schema_version}; expected 1.`, true);
  else if (value.error) message(value.error, true);
  else if (value.state === "running") message(`Running · ${format(slice)} cycles per tick`);
  else if (value.state === "paused") message(`Paused at cycle ${format(t.cycles)}`);
  else if (value.state === "stopped") message("Stopped");
  else message("Idle");
  previous = typeof t.cycles === "number" ? t : null;
}

async function control(action) {
  if (busy) return;
  busy = true;
  if (state) render(state);
  let controlError = "";
  try {
    const body = {action};
    if (action === "start" || action === "reset") body.profile = element("profile").value;
    if (action === "step") body.cycles = (state && state.slice_cycles) || 256;
    const response = await fetch("/api/control", {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || "Control request failed.");
    render(value);
  } catch (error) {
    controlError = error.message;
  } finally {
    busy = false;
    if (state) render(state);
    if (controlError) message(controlError, true);
  }
}

async function loadProfiles() {
  try {
    const response = await fetch("/api/profiles", {cache: "no-store"});
    if (!response.ok) return;
    const profiles = await response.json();
    const select = element("profile");
    const chosen = select.value;
    select.replaceChildren(...Object.entries(profiles).map(([name, info]) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = info.bound ? `${info.label} (${info.bound}-bound)` : info.label;
      option.title = info.description;
      return option;
    }));
    if (chosen in profiles) select.value = chosen;
  } catch (error) {
    // The options in index.html stay in place.
  }
}

element("start").onclick = () => control("start");
element("pause").onclick = () => control(state?.state === "paused" ? "resume" : "pause");
element("step").onclick = () => control("step");
element("stop").onclick = () => control("stop");
element("reset").onclick = () => control("reset");

async function poll() {
  try {
    const response = await fetch("/api/state", {cache: "no-store"});
    if (!response.ok) throw new Error("Controller is unavailable.");
    const value = await response.json();
    if (!busy) render(value);
  } catch (error) {
    element("status").textContent = "Disconnected";
    element("status-dot").className = "error";
    message("Controller unreachable — is run.py still running?", true);
  }
  setTimeout(poll, document.hidden ? 1500 : 250);
}

loadProfiles();
poll();
