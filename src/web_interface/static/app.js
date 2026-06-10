import { getJSON, postJSON, watchJob } from "./api.js";
import { $, csvNumbers, escapeHTML, option, table } from "./dom.js";
import { drawComponents, drawNorms, drawTrajectory } from "./plots.js";

const state = {
  api: null,
  run: null,
  activeTool: "generation_summary",
  params: {
    trajectory_projection: { layer: null, interval: 4, method: "pca" },
    pca_components: { layer: null, n: 24 },
  },
  view3d: { pitch: -0.55, yaw: 0.72, zoom: 1 },
};

init();

async function init() {
  state.api = await getJSON("/api/state");
  hydrateFromURL();
  renderStaticControls();
  bindGlobalEvents();
  if (state.api.runs.length) {
    $("runSelect").value = currentRunFromURL() || state.api.runs[0].path;
    await loadRun();
  }
  renderTabs();
}

function hydrateFromURL() {
  const params = new URLSearchParams(window.location.search);
  state.activeTool = params.get("tool") || state.activeTool;
}

function currentRunFromURL() {
  return new URLSearchParams(window.location.search).get("run");
}

function renderStaticControls() {
  $("runSelect").innerHTML = state.api.runs.map(run => option(run.path, `${run.model_name} / ${run.name}`)).join("");
  $("modelSelect").innerHTML = state.api.models.map(model => option(model.id, model.label)).join("");
  $("datasetSelect").innerHTML = state.api.datasets.map(dataset => option(dataset.path, dataset.label)).join("");
  $("toolList").innerHTML = state.api.tools.map(tool => `
    <button class="ghostButton" data-testid="run-${tool.name}" data-tool="${tool.name}">
      <span>${escapeHTML(tool.label)}</span>
      <small>${escapeHTML(tool.description)}</small>
    </button>`).join("");
}

function bindGlobalEvents() {
  $("loadRun").addEventListener("click", loadRun);
  $("startGeneration").addEventListener("click", startGeneration);
  $("runAll").addEventListener("click", () => startAnalysis("all"));
  $("toolList").addEventListener("click", event => {
    const tool = event.target.closest("[data-tool]")?.dataset.tool;
    if (tool) startAnalysis(tool);
  });
  $("tabs").addEventListener("click", event => {
    const tool = event.target.closest("[data-tool]")?.dataset.tool;
    if (!tool) return;
    state.activeTool = tool;
    syncURL();
    renderTabs();
    renderActiveTool();
  });
  window.addEventListener("resize", () => renderActiveTool());
}

async function loadRun() {
  const runPath = $("runSelect").value;
  if (!runPath) return;
  state.run = await getJSON(`/api/run?run_path=${encodeURIComponent(runPath)}`);
  ensureValidLayers();
  renderRunMeta();
  syncURL();
  await renderActiveTool();
}

function ensureValidLayers() {
  const layers = (state.run?.layers || []).map(String);
  const fallback = lastLayer();
  for (const params of [state.params.trajectory_projection, state.params.pca_components]) {
    if (!params.layer || !layers.includes(String(params.layer))) params.layer = fallback;
  }
}

function renderRunMeta() {
  const layers = state.run.layers.length ? `layers ${state.run.layers.join(", ")}` : "no activations";
  $("runMeta").textContent = `${state.run.run_path} · ${state.run.generations} generations · ${layers}`;
}

async function startGeneration() {
  const selected = state.api.models.find(model => model.id === $("modelSelect").value) || {};
  const custom = $("customModel").value.trim();
  const body = {
    model_name: custom || selected.id,
    backend: selected.backend || "hf",
    dataset_path: $("datasetSelect").value,
    run_name: $("runName").value.trim() || "web_run",
    max_new_tokens: Number($("maxTokens").value || 160),
    seeds: csvNumbers($("seeds").value, Number),
    temperatures: csvNumbers($("temperatures").value, Number),
    layers: csvNumbers($("layers").value, Number),
  };
  await runJob("/api/generate", body);
  state.api = await getJSON("/api/state");
  renderStaticControls();
}

async function startAnalysis(tool) {
  if (!state.run) return;
  await runJob("/api/analyze", { run_path: state.run.run_path, tool, params: state.params });
  await loadRun();
}

async function runJob(url, body) {
  setJob({ label: "starting", status: "queued" });
  const job = await postJSON(url, body);
  await watchJob(job, setJob);
}

function setJob(job) {
  const label = `${job.label}: ${job.status}`;
  $("jobStatus").textContent = label;
  $("jobStatus").dataset.status = job.status;
}

function renderTabs() {
  $("tabs").innerHTML = state.api.tools.map(tool => `
    <button class="tab ${tool.name === state.activeTool ? "active" : ""}" data-testid="tab-${tool.name}" data-tool="${tool.name}">
      ${escapeHTML(tool.label)}
    </button>`).join("");
}

async function renderActiveTool() {
  if (!state.run) {
    $("panel").innerHTML = `<div class="empty">Select or generate a run.</div>`;
    return;
  }

  try {
    if (state.activeTool === "generation_summary") return renderSummary(await toolData("generation_summary"));
    if (state.activeTool === "activation_norms") return renderNorms(await toolData("activation_norms"));
    if (state.activeTool === "trajectory_projection") return renderTrajectory(await toolData("trajectory_projection", state.params.trajectory_projection));
    if (state.activeTool === "pca_components") return renderComponents(await toolData("pca_components", state.params.pca_components));
  } catch (error) {
    renderMissing(error);
  }
}

function renderSummary(data) {
  const rows = data.rows || [];
  $("panel").innerHTML = `
    <section class="viewHeader">
      <div><h2>Generation summary</h2><p>One row per generated answer.</p></div>
    </section>
    ${table(rows)}`;
}

function renderNorms(data) {
  $("panel").innerHTML = `
    <section class="viewHeader">
      <div><h2>Activation norms</h2><p>Layer-wise magnitude summary across generated tokens.</p></div>
    </section>
    <div class="plotCard"><canvas id="normCanvas"></canvas></div>`;
  drawNorms($("normCanvas"), data.rows || []);
}

function renderTrajectory(data) {
  $("panel").innerHTML = `
    <section class="viewHeader">
      <div><h2>3D trajectory projection</h2><p>Drag the plot to rotate. PCA/t-SNE produces three coordinates: x, y, z.</p></div>
      <button class="secondaryButton" id="runCurrentTool">Recompute</button>
    </section>
    ${trajectoryControls()}
    <div class="plotCard tall"><canvas id="trajectoryCanvas"></canvas></div>`;
  bindToolControls();
  drawTrajectory($("trajectoryCanvas"), data, state.view3d);
  bind3DCanvas($("trajectoryCanvas"), data);
}

function renderComponents(data) {
  $("panel").innerHTML = `
    <section class="viewHeader">
      <div><h2>PCA components</h2><p>Singular-value amplitude and variance concentration.</p></div>
      <button class="secondaryButton" id="runCurrentTool">Recompute</button>
    </section>
    ${componentControls()}
    <div class="plotCard"><canvas id="componentCanvas"></canvas></div>`;
  bindToolControls();
  drawComponents($("componentCanvas"), data);
}

function renderMissing(error) {
  $("panel").innerHTML = `
    <div class="empty">
      <strong>Missing or stale analysis output.</strong><br>
      ${escapeHTML(error.message)}
      <br><br>
      <button id="missingRunTool">Run this analysis</button>
    </div>`;
  $("missingRunTool").addEventListener("click", () => startAnalysis(state.activeTool));
}

function trajectoryControls() {
  const p = state.params.trajectory_projection;
  return `
    <div class="toolbar">
      <label>Layer<select id="layerControl">${layerOptions(p.layer)}</select></label>
      <label>Interval<input id="intervalControl" type="number" min="1" value="${escapeHTML(p.interval)}"></label>
      <label>Projection<select id="methodControl">
        ${option("pca", "PCA", p.method === "pca")}
        ${option("tsne", "t-SNE", p.method === "tsne")}
      </select></label>
      <label>Zoom<input id="zoomControl" type="range" min="70" max="150" value="${Math.round(state.view3d.zoom * 100)}"></label>
      <button class="secondaryButton" id="resetView">Reset view</button>
    </div>`;
}

function componentControls() {
  const p = state.params.pca_components;
  return `
    <div class="toolbar">
      <label>Layer<select id="layerControl">${layerOptions(p.layer)}</select></label>
      <label>Components<input id="componentCount" type="number" min="1" max="128" value="${escapeHTML(p.n)}"></label>
    </div>`;
}

function bindToolControls() {
  $("runCurrentTool")?.addEventListener("click", () => startAnalysis(state.activeTool));
  $("layerControl")?.addEventListener("change", event => {
    activeParams().layer = event.target.value;
    renderActiveTool();
  });
  $("intervalControl")?.addEventListener("change", event => {
    state.params.trajectory_projection.interval = Math.max(1, Number(event.target.value || 4));
    renderActiveTool();
  });
  $("methodControl")?.addEventListener("change", event => {
    state.params.trajectory_projection.method = event.target.value;
    renderActiveTool();
  });
  $("componentCount")?.addEventListener("change", event => {
    state.params.pca_components.n = Math.max(1, Number(event.target.value || 24));
    renderActiveTool();
  });
  $("zoomControl")?.addEventListener("input", event => {
    state.view3d.zoom = Number(event.target.value) / 100;
    renderActiveTool();
  });
  $("resetView")?.addEventListener("click", () => {
    state.view3d = { pitch: -0.55, yaw: 0.72, zoom: 1 };
    renderActiveTool();
  });
}

function bind3DCanvas(canvas, data) {
  let dragging = false;
  let last = null;
  canvas.addEventListener("pointerdown", event => {
    dragging = true;
    last = { x: event.clientX, y: event.clientY };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", event => {
    if (!dragging || !last) return;
    state.view3d.yaw += (event.clientX - last.x) * 0.012;
    state.view3d.pitch += (event.clientY - last.y) * 0.012;
    state.view3d.pitch = Math.max(-1.45, Math.min(1.45, state.view3d.pitch));
    last = { x: event.clientX, y: event.clientY };
    drawTrajectory(canvas, data, state.view3d);
  });
  canvas.addEventListener("pointerup", () => { dragging = false; last = null; });
  canvas.addEventListener("pointercancel", () => { dragging = false; last = null; });
}

function activeParams() {
  return state.activeTool === "pca_components" ? state.params.pca_components : state.params.trajectory_projection;
}

function layerOptions(selected) {
  const layers = state.run?.layers || [];
  return (layers.length ? layers : ["0"]).map(layer => option(layer, layer, String(layer) === String(selected || lastLayer()))).join("");
}

function lastLayer() {
  const layers = state.run?.layers || [];
  return layers.length ? String(layers[layers.length - 1]) : "0";
}

async function toolData(tool, params = {}) {
  const query = new URLSearchParams({ run_path: state.run.run_path, tool, ...params });
  return getJSON(`/api/tool-data?${query}`);
}

function syncURL() {
  if (!state.run) return;
  history.replaceState(null, "", `?run=${encodeURIComponent(state.run.run_path)}&tool=${encodeURIComponent(state.activeTool)}`);
}
