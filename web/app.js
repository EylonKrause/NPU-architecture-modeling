"use strict";
const element = id => document.getElementById(id);
const format = value => typeof value === "number" ? value.toLocaleString("en-US") : "—";
let state = null;
let busy = false;
function bytes(value) { if (typeof value !== "number") return "—"; if(value < 1024) return `${format(value)} B`; if(value < 1048576) return `${(value/1024).toFixed(1)} KiB`; return `${(value/1048576).toFixed(1)} MiB`; }
function message(text, error = false) {element("message").textContent=text; element("message").classList.toggle("error",error);}
function pick(object,...keys) {for(const key of keys) if(object[key]!==undefined) return object[key]; return undefined;}
function render(value) {
  state=value; const t=value.telemetry||{}; const stats=t.statistics||t;
  if(["running","paused"].includes(value.state)) element("profile").value=value.profile;
  const cycles=pick(t,"cycles","simulated_cycles")??stats.cycles;
  const macs=pick(stats,"macs","completed_macs");
  const read=stats.read_bytes, write=stats.write_bytes;
  element("status").textContent=value.state[0].toUpperCase()+value.state.slice(1);
  element("status-dot").className=value.state;
  element("cycles").textContent=format(cycles);
  element("macs").textContent=format(macs);
  element("traffic").textContent=typeof read==="number"&&typeof write==="number"?bytes(read+write):"—";
  element("traffic-detail").textContent=`${bytes(read)} read / ${bytes(write)} write`;
  element("jobs").textContent=format(value.jobs_completed);
  const config=t.config||t.configuration||{};
  element("workload-shape").textContent=config.m?`${config.m} × ${config.n} × ${config.k} · signed INT${config.weight_bits||8} weights · ${config.weight_encoding||"dense"}`:"INT8 matrix computation";
  element("compression-detail").textContent=(config.weight_encoding&&config.weight_encoding!=="dense")
    ?`${config.weight_encoding} · ${bytes(stats.encoded_weight_storage_bytes)} encoded / ${bytes(stats.dense_weight_storage_bytes)} packed-dense weights · ${format(stats.weight_decode_cycles)} decode cycles · ${format(stats.skipped_macs)} MACs skipped`
    :"Dense weights · no decompression overhead";
  const targetTime=t.target_time_ns??(typeof cycles==="number"&&config.frequency_mhz?cycles*1000/config.frequency_mhz:undefined);
  element("target-time").textContent=typeof targetTime==="number"?`${format(targetTime)} ns of target time`:"Current inference · simulated clock";
  const compute=stats.compute_busy_cycles||0, dma=stats.dma_busy_cycles||0;
  for(const [name,numerator] of [["compute",compute],["dma",dma]]) {
    const percent=cycles?Math.min(100,100*numerator/cycles):0;
    element(`${name}-util`).textContent=cycles?`${percent.toFixed(1)}%`:"—";
    element(`${name}-bar`).value=percent;
  }
  const active=t.active_stages||{};
  for(const name of ["load","compute","store"]) element(`stage-${name}`).classList.toggle("active",Array.isArray(active)?active.includes(name):Boolean(active[name]));
  element("load-count").textContent=bytes(read)+" transferred";
  element("compute-count").textContent=format(macs)+" MACs complete";
  element("store-count").textContent=bytes(write)+" transferred";
  const total=pick(t,"tiles_total","total_tiles");
  const complete=pick(t,"tiles_completed","completed_tiles","stored_tiles");
  const container=element("tiles"); container.replaceChildren();
  if(typeof total==="number"&&typeof complete==="number") {
    const shown=Math.min(total,256);
    for(let index=0;index<shown;index++) {const tile=document.createElement("span");tile.className="tile"+(index<complete?" done":"");tile.title=`Output tile ${index}: ${index<complete?"stored":"pending"}`;container.append(tile);}
    element("tiles-label").textContent=`${complete} / ${total} stored${total>256?" · first 256 shown":""}`;
  } else element("tiles-label").textContent="Waiting for tile counters";
  const recent=value.recent_jobs||[];
  const last=recent.length?recent[0].telemetry:{};
  element("input-preview").textContent=Array.isArray(t.input_preview)?t.input_preview.join("  "):"Waiting for an inference";
  element("output-preview").textContent=Array.isArray(last.output_preview)?last.output_preview.join("  "):"Output is shown only after full numerical validation";
  const validated=pick(last,"validated","validation_passed","correct");
  element("validation").textContent=validated===true?"Scalar reference matched":validated===false?"Validation failed":"Awaiting a complete inference";
  element("validation").className="badge"+(validated===true?" valid":validated===false?" invalid":"");
  if(recent.length) {
    element("history").replaceChildren(...recent.map(job=>{
      const row=document.createElement("div");row.className="history-row";
      const number=document.createElement("span");number.textContent=`Job ${job.job}`;
      const detail=document.createElement("div");detail.textContent=`${format(job.telemetry.cycles)} target cycles`;
      const hash=document.createElement("code");hash.textContent=`output ${job.telemetry.output_hash??job.telemetry.output_checksum??"signature unavailable"}`;detail.append(hash);
      const valid=document.createElement("strong");valid.textContent=pick(job.telemetry,"validated","validation_passed","correct")===true?"VERIFIED":"FAILED";
      row.append(number,detail,valid);return row;
    }));
  } else {const empty=document.createElement("p");empty.className="empty";empty.textContent="Completed inferences appear here with their real output signature and validation.";element("history").replaceChildren(empty);}
  element("profile").disabled=busy||value.state==="running"||value.state==="paused";
  element("start").disabled=busy||value.state==="running"||value.state==="paused";
  element("pause").disabled=busy||!["running","paused"].includes(value.state);
  element("pause").textContent=value.state==="paused"?"Resume":"Pause";
  element("step").disabled=busy||value.state!=="paused";
  element("stop").disabled=busy||!["running","paused","error"].includes(value.state);
  element("reset").disabled=busy;
  element("connection").textContent="Local controller · refreshed "+new Date().toLocaleTimeString();
  if(value.error) message(value.error,true);
  else if(value.state==="running") message("Continuous inference is executing. Each completed result is checked against the scalar reference before the next job.");
  else if(value.state==="paused") message("Simulation time is frozen. Step advances 256 target cycles; Resume continues execution.");
  else if(value.state==="stopped") message("Simulator stopped. Only the dashboard's own model process was closed.");
  else message("The simulator is idle. Start a workload to execute the C++/SystemC model.");
}
async function control(action) {
  if(busy) return; busy=true; if(state)render(state);
  let controlError="";
  try {
    const body={action}; if(action==="start"||action==="reset")body.profile=element("profile").value;
    if(action==="step")body.cycles=256;
    const response=await fetch("/api/control",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    const value=await response.json();if(!response.ok)throw new Error(value.error||"Control request failed.");
    render(value);
  } catch(error) {controlError=error.message;} finally {busy=false;if(state)render(state);if(controlError)message(controlError,true);}
}
element("start").onclick=()=>control("start");
element("pause").onclick=()=>control(state?.state==="paused"?"resume":"pause");
element("step").onclick=()=>control("step");
element("stop").onclick=()=>control("stop");
element("reset").onclick=()=>control("reset");
async function poll() {
  try {const response=await fetch("/api/state",{cache:"no-store"});if(!response.ok)throw new Error("Controller is unavailable.");const value=await response.json();if(!busy)render(value);}
  catch(error) {element("status").textContent="Disconnected";element("status-dot").className="error";message("Cannot reach the local controller. Start scripts/live_dashboard.py to reconnect.",true);}
  setTimeout(poll,document.hidden?1500:250);
}
poll();
