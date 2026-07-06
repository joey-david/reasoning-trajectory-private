import { createGenerationView } from "./generations.js";
import { createTrajectoryView } from "./trajectories.js";
import { createDiagnosticsView } from "./diagnostics.js";
import {
  $,
  debounce,
  escapeHtml,
  fetchJson,
  fetchJsonl,
  formatNumber,
  formatPercent,
  isFormTarget,
  median,
  outcome,
  questionText,
  setOptions,
} from "./ui.js";

const state = {
  manifest: null,
  run: null,
  rows: [],
  markers: null,
  hardQuestions: [],
  view: "overview",
};

const generationView = createGenerationView({
  getState: () => state,
  setQuery,
  openTrajectory,
});
const trajectoryView = createTrajectoryView({
  getState: () => state,
  setQuery,
  openGeneration,
});
const diagnosticsView = createDiagnosticsView({
  getState: () => state,
  setQuery,
});

init();

async function init() {
  bindShell();
  try {
    state.manifest = await fetchJson("data/runs.json");
    const runs = state.manifest.runs ?? [];
    if (!runs.length) throw new Error("No analyzed runs are listed in web/data/runs.json.");
    const route = routeState();
    const models = [...new Set(runs.map(run => run.model))].sort();
    const defaultRun = runs.find(run => run.trajectory_metrics) ?? runs[0];
    setOptions("model", models, null, route.model ?? defaultRun.model);
    fillRunOptions(route.run ?? (
      $("model").value === defaultRun.model ? defaultRun.run : null
    ));
    await loadRun(route);
  } catch (error) {
    showSetupError(error);
  }
}

function bindShell() {
  $("model").addEventListener("change", () => {
    fillRunOptions();
    loadRun({});
  });
  $("run").addEventListener("change", () => loadRun({}));
  for (const button of document.querySelectorAll("[data-view]")) {
    button.addEventListener("click", () => showView(button.dataset.view));
  }
  $("overview-search").addEventListener("input", debounce(renderQuestionTable));
  $("question-table").addEventListener("click", handleQuestionAction);
  for (const button of document.querySelectorAll("[data-toggle-panel]")) {
    button.addEventListener("click", () => togglePanel(button));
  }
  document.addEventListener("keydown", handleShortcut);
  window.addEventListener("popstate", () => window.location.reload());
  const narrow = window.matchMedia("(max-width: 620px)");
  if (narrow.matches) {
    setPanelCollapsed(document.querySelector(".generation-workspace .control-panel"), true);
    setPanelCollapsed(document.querySelector(".trajectory-workspace .control-panel"), true);
  }
}

function fillRunOptions(preferred = null) {
  const runs = state.manifest.runs.filter(run => run.model === $("model").value);
  setOptions("run", runs.map(run => run.run), null, preferred);
}

async function loadRun(route) {
  const run = state.manifest.runs.find(
    candidate => candidate.model === $("model").value && candidate.run === $("run").value,
  );
  if (!run) return;

  showLoading(`Loading ${run.model} / ${run.run}…`);
  state.run = run;
  try {
    const [rows, markers, hardQuestions] = await Promise.all([
      fetchJsonl(run.generations),
      run.step_markers ? fetchJson(run.step_markers).catch(() => null) : null,
      run.hard_questions ? fetchJsonl(run.hard_questions).catch(() => []) : [],
    ]);
    state.rows = normalizeRows(rows, run.generation_format);
    state.markers = markers;
    state.hardQuestions = hardQuestions;
    $("app-status").hidden = true;
    $("header-run-status").textContent = `${formatNumber(state.rows.length)} trajectories · ${formatNumber(Object.keys(run.samples ?? {}).length)} questions`;
    renderOverview();
    showView(["overview", "generations", "trajectories", "diagnostics"].includes(route.view) ? route.view : "overview");
    setQuery({ model: run.model, run: run.run, view: state.view });
  } catch (error) {
    showSetupError(error);
  }
}

function normalizeRows(rows, generationFormat) {
  if (generationFormat !== "causal_patching") return rows;
  return rows.map(row => ({
    ...row,
    sample_id: row.target?.sample_id ?? row.target_question,
    reasoning_length: row.generated_token_ids?.length ?? 0,
  }));
}

function showView(view, updateUrl = true) {
  state.view = view;
  for (const button of document.querySelectorAll("[data-view]")) {
    button.setAttribute("aria-selected", String(button.dataset.view === view));
  }
  for (const name of ["overview", "generations", "trajectories", "diagnostics"]) {
    $(`${name}-panel`).hidden = name !== view;
  }
  if (view === "generations") generationView.load(routeState());
  if (view === "trajectories") {
    trajectoryView.load(routeState());
    setTimeout(() => window.Plotly?.Plots.resize($("plot3d")), 0);
  }
  if (view === "diagnostics") diagnosticsView.load(routeState());
  if (updateUrl) setQuery({ ...queryCleanup(view), view });
}

function renderOverview() {
  const rows = state.rows;
  const correct = rows.filter(row => row.is_correct === true).length;
  const incorrect = rows.filter(row => row.is_correct === false).length;
  const unknown = rows.length - correct - incorrect;
  const lengths = rows.map(row => Number(row.reasoning_length)).filter(Number.isFinite);
  const samples = Object.keys(state.run.samples ?? {});
  const projectionCount = (state.run.interactive_plots?.length ?? 0)
    + (state.run.step_classification_plots?.length ?? 0);

  $("run-metrics").innerHTML = [
    ["Trajectories", formatNumber(rows.length), `${formatNumber(samples.length)} questions`],
    ["Correct", formatPercent(correct / Math.max(rows.length, 1), 1), `${formatNumber(incorrect)} incorrect · ${formatNumber(unknown)} unscored`],
    ["Median reasoning", formatNumber(median(lengths)), "generated tokens"],
    ["Projections", formatNumber(projectionCount), state.run.step_classification_plots?.length ? "token and step level" : "token level"],
  ].map(([label, value, note]) => `<article class="metric"><span>${label}</span><strong>${value}</strong><small>${note}</small></article>`).join("");

  const artifacts = [
    ["Generations", true],
    ["Step markers", Boolean(state.run.step_markers)],
    ["Token projections", Boolean(state.run.interactive_plots?.length)],
    ["Step clusters", Boolean(state.run.step_classification_plots?.length)],
    ["Trajectory diagnostics", Boolean(state.run.trajectory_metrics)],
    ["Hardness ranking", Boolean(state.run.hard_questions)],
  ];
  $("artifact-status").innerHTML = artifacts
    .filter(([, available]) => available)
    .map(([label]) => `<span class="artifact-chip available">${label}</span>`)
    .join("");
  renderQuestionTable();
}

function renderQuestionTable() {
  if (!state.run) return;
  const search = $("overview-search").value.trim().toLowerCase();
  const hardBySample = new Map(state.hardQuestions.map(item => [item.sample_id, item]));
  const grouped = new Map();
  for (const row of state.rows) {
    if (!grouped.has(row.sample_id)) grouped.set(row.sample_id, []);
    grouped.get(row.sample_id).push(row);
  }

  const summaries = [...grouped.entries()].map(([sampleId, rows]) => {
    const sample = state.run.samples[sampleId] ?? {};
    const correct = rows.filter(row => row.is_correct === true).length;
    const answers = new Set(rows.map(row => row.produced_answer).filter(Boolean));
    return {
      sampleId,
      text: questionText(sample.prompt),
      rows,
      correct,
      answers: answers.size,
      avgTokens: rows.reduce((sum, row) => sum + Number(row.reasoning_length ?? 0), 0) / rows.length,
      hardness: hardBySample.get(sampleId)?.hardness_score,
    };
  }).filter(item => !search || `${item.sampleId} ${item.text}`.toLowerCase().includes(search))
    .sort((a, b) => {
      const accuracyDifference = a.correct / a.rows.length - b.correct / b.rows.length;
      return accuracyDifference || Number(b.hardness ?? 0) - Number(a.hardness ?? 0) || b.avgTokens - a.avgTokens;
    });

  if (!summaries.length) {
    $("question-table").innerHTML = `<div class="empty-state">No questions match this filter.</div>`;
    return;
  }

  $("question-table").innerHTML = `<table class="question-table">
    <thead><tr><th>Question</th><th>Runs</th><th>Correct</th><th>Answers</th><th>Avg. tokens</th><th>Open</th></tr></thead>
    <tbody>${summaries.map(item => `<tr>
      <td class="question-cell"><strong>${escapeHtml(item.sampleId)}</strong><span>${escapeHtml(item.text)}</span></td>
      <td>${formatNumber(item.rows.length)}</td>
      <td><span class="status-pill ${item.correct === item.rows.length ? "correct" : item.correct ? "unknown" : "incorrect"}">${formatPercent(item.correct / item.rows.length)}</span></td>
      <td>${formatNumber(item.answers)}</td>
      <td>${formatNumber(item.avgTokens)}</td>
      <td><div class="table-actions">
        <button class="text-button" type="button" data-action="generations" data-sample-id="${escapeHtml(item.sampleId)}">Traces</button>
        <button class="text-button" type="button" data-action="trajectories" data-sample-id="${escapeHtml(item.sampleId)}">Latent</button>
      </div></td>
    </tr>`).join("")}</tbody>
  </table>`;
}

function handleQuestionAction(event) {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const sampleId = button.dataset.sampleId;
  if (button.dataset.action === "generations") {
    setQuery({ view: "generations", question: sampleId, seed: "" });
    showView("generations");
    return;
  }
  openTrajectory(sampleId, "");
}

function openTrajectory(sampleId, seed) {
  setQuery({ view: "trajectories", question: sampleId, seed });
  showView("trajectories");
}

function openGeneration(point) {
  setQuery({
    view: "generations",
    question: point.sample_id,
    seed: point.seed,
    token: point.token_idx,
    char_start: point.char_start,
    char_end: point.char_end,
  }, true);
  showView("generations");
}

function handleShortcut(event) {
  if (event.metaKey || event.ctrlKey || event.altKey || isFormTarget(event.target)) return;
  const view = { o: "overview", g: "generations", l: "trajectories", d: "diagnostics" }[event.key.toLowerCase()];
  if (view) {
    event.preventDefault();
    showView(view);
    return;
  }
  if (event.key === "/") {
    event.preventDefault();
    if (state.view === "overview") $("overview-search").focus();
    if (state.view === "generations") generationView.focusSearch();
  }
}

function togglePanel(button) {
  const panel = button.closest(".control-panel");
  setPanelCollapsed(panel, !panel.classList.contains("collapsed"));
}

function setPanelCollapsed(panel, collapsed) {
  if (!panel) return;
  panel.classList.toggle("collapsed", collapsed);
  const button = panel.querySelector("[data-toggle-panel]");
  button?.setAttribute("aria-expanded", String(!collapsed));
  if (button) button.textContent = collapsed ? "Show" : "Hide";
}

function routeState() {
  return Object.fromEntries(new URLSearchParams(window.location.search));
}

function setQuery(values, push = false) {
  const query = new URLSearchParams(window.location.search);
  for (const [key, value] of Object.entries(values)) {
    if (value === undefined) continue;
    if (value === null || value === "") query.delete(key);
    else query.set(key, value);
  }
  const method = push ? "pushState" : "replaceState";
  history[method](null, "", `${window.location.pathname}?${query.toString()}${window.location.hash}`);
}

function queryCleanup(view) {
  const generationKeys = [
    "search",
    "outcome",
    "marker",
    "sort",
    "entropy",
    "token",
    "char_start",
    "char_end",
  ];
  const trajectoryKeys = ["source", "selector", "cluster", "color", "limit", "start", "end"];
  const diagnosticKeys = ["diagnostic"];
  const keys = view === "overview"
    ? [...generationKeys, ...trajectoryKeys, ...diagnosticKeys, "question", "seed"]
    : view === "generations"
      ? [...trajectoryKeys, ...diagnosticKeys]
      : view === "trajectories"
        ? [...generationKeys, ...diagnosticKeys]
        : [...generationKeys, ...trajectoryKeys, "question", "seed"];
  return Object.fromEntries(keys.map(key => [key, null]));
}

function showLoading(message) {
  $("app-status").hidden = false;
  $("app-status").className = "app-status";
  $("app-status").innerHTML = `<span class="spinner" aria-hidden="true"></span>${escapeHtml(message)}`;
  for (const name of ["overview", "generations", "trajectories", "diagnostics"]) {
    $(`${name}-panel`).hidden = true;
  }
}

function showSetupError(error) {
  const fileHint = window.location.protocol === "file:"
    ? "The page was opened as a file, which blocks data loading."
    : "The run index or one of its artifacts could not be loaded.";
  $("app-status").hidden = false;
  $("app-status").className = "app-status error";
  $("app-status").innerHTML = `
    <strong>${escapeHtml(fileHint)}</strong>
    <span>${escapeHtml(error.message)}</span>
    <code>python3 -m http.server 8765</code>
    <span>Then open http://localhost:8765/web/index.html</span>`;
}
