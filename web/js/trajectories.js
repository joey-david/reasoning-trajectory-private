import {
  $,
  debounce,
  escapeHtml,
  formatNumber,
  setOptions,
} from "./ui.js";

const CLUSTER_COLORS = [
  "#2563b8",
  "#8a4aa5",
  "#c04d78",
  "#c3672d",
  "#9a7a12",
  "#20815f",
  "#187d8f",
  "#5a5db5",
  "#66736c",
  "#a53d4d",
];

export function createTrajectoryView({ getState, setQuery }) {
  const payloadCache = new Map();
  let payload = null;
  let activePlot = null;
  let loadSequence = 0;

  $("plot-source").addEventListener("change", () => loadSelectedPlot());
  $("plot-question").addEventListener("change", () => {
    updateSeedOptions();
    render();
    syncQuery();
  });
  for (const id of ["plot-seed", "plot-selector", "plot-cluster", "plot-color-mode"]) {
    $(id).addEventListener("change", () => {
      render();
      syncQuery();
    });
  }
  for (const id of ["plot-max-trajectories", "plot-token-start", "plot-token-end"]) {
    $(id).addEventListener("input", debounce(() => {
      updateRangeOutputs();
      render();
      syncQuery();
    }, 50));
  }
  $("trajectory-clear").addEventListener("click", clearFilters);
  $("reset-camera").addEventListener("click", resetCamera);
  $("copy-view-link").addEventListener("click", copyViewLink);

  function load(route) {
    const plots = allPlots();
    setOptions(
      "plot-source",
      plots.map((plot, index) => [index, plotLabel(plot)]),
      null,
      route.source,
    );
    const { rows } = getState();
    setOptions(
      "plot-question",
      [...new Set(rows.map(row => row.sample_id))].sort(),
      { value: "", label: "All questions" },
      route.question,
    );
    $("plot-max-trajectories").value = validRange(route.limit, 1, 50, 12);
    $("plot-token-start").value = validRange(route.start, 0, 100, 0);
    $("plot-token-end").value = validRange(route.end, 0, 100, 100);
    $("plot-color-mode").value = route.color === "cluster" ? "cluster" : "correctness";
    updateSeedOptions(route.seed);
    updateRangeOutputs();
    renderStaticPlots();
    loadSelectedPlot(route);
  }

  function allPlots() {
    const { run } = getState();
    return [
      ...(run.interactive_plots ?? []).map(plot => ({ ...plot, plot_type: "token" })),
      ...(run.step_classification_plots ?? []).map(plot => ({ ...plot, plot_type: "step" })),
    ];
  }

  async function loadSelectedPlot(route = {}) {
    const sequence = ++loadSequence;
    activePlot = allPlots()[Number($("plot-source").value || 0)] ?? null;
    payload = null;
    showLoading(true);
    $("plot-title").textContent = activePlot ? plotLabel(activePlot) : "Projection";
    if (!activePlot) {
      showLoading(false);
      showPlotMessage("No interactive projections are available. Run scripts/analyze.py for this run.");
      return;
    }

    try {
      if (!payloadCache.has(activePlot.path)) {
        const response = await fetch(activePlot.path);
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        payloadCache.set(activePlot.path, await response.json());
      }
      if (sequence !== loadSequence) return;
      payload = payloadCache.get(activePlot.path);
      updatePayloadOptions(route);
      showLoading(false);
      render();
      syncQuery();
    } catch (error) {
      if (sequence !== loadSequence) return;
      showLoading(false);
      showPlotMessage(`Could not load this projection: ${error.message}`);
    }
  }

  function updatePayloadOptions(route) {
    const selectors = [...new Set((payload.points ?? []).map(point => point.selector).filter(Boolean))];
    const clusters = [...new Set(
      (payload.points ?? []).map(point => point.cluster_id).filter(value => value !== undefined && value !== null),
    )].sort((a, b) => Number(a) - Number(b));

    const defaultSelector = route.selector ?? selectors[0] ?? "";
    setOptions("plot-selector", selectors, { value: "", label: "All sampling methods" }, defaultSelector);
    setOptions(
      "plot-cluster",
      clusters.map(value => [value, `Cluster ${value}`]),
      { value: "", label: "All clusters" },
      route.cluster,
    );
    $("plot-selector-control").hidden = selectors.length === 0;
    $("plot-cluster-control").hidden = clusters.length === 0;
    $("plot-color-mode").querySelector('option[value="cluster"]').disabled = clusters.length === 0;
    if (!clusters.length) $("plot-color-mode").value = "correctness";
  }

  function updateSeedOptions(preferred = $("plot-seed").value) {
    const { rows } = getState();
    const question = $("plot-question").value;
    const seeds = [...new Set(
      rows.filter(row => !question || row.sample_id === question).map(row => row.seed),
    )].sort((a, b) => Number(a) - Number(b));
    setOptions("plot-seed", seeds, { value: "", label: "All sub-runs" }, preferred);
  }

  function render() {
    if (!payload) return;
    if (!window.Plotly) {
      showPlotMessage("Plotly did not load. Check the network connection and reload.");
      return;
    }

    const { points, matchingCount } = filteredPoints();
    const trajectoryCount = new Set(points.map(trajectoryKey)).size;
    const pointCount = points.length < matchingCount
      ? `${formatNumber(points.length)} of ${formatNumber(matchingCount)} matching points`
      : `${formatNumber(points.length)} points`;
    const samplingNote = payload.sampled
      ? ` · globally sampled from ${formatNumber(payload.source_points)}`
      : "";
    $("plot-status").textContent = `${pointCount} · ${formatNumber(trajectoryCount)} trajectories${samplingNote}`;

    if (!points.length) {
      window.Plotly.purge("plot3d");
      showPlotMessage("No points match the current filters.");
      return;
    }

    const traces = tracesForPoints(points);
    window.Plotly.react("plot3d", traces, plotLayout(), {
      responsive: true,
      scrollZoom: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["select2d", "lasso2d"],
    }).then(() => {
      improveModebarAccessibility();
      bindPlotClick();
    }).catch(error => {
      showPlotMessage(`Plot rendering failed: ${error.message}`);
    });
  }

  function filteredPoints() {
    const question = $("plot-question").value;
    const seed = $("plot-seed").value;
    const selector = $("plot-selector").value;
    const cluster = $("plot-cluster").value;
    const start = Math.min(Number($("plot-token-start").value), Number($("plot-token-end").value)) / 100;
    const end = Math.max(Number($("plot-token-start").value), Number($("plot-token-end").value)) / 100;
    const maxTrajectories = Number($("plot-max-trajectories").value);
    const selectedTrajectories = new Set();
    const points = [];

    for (const point of payload.points ?? []) {
      if (question && point.sample_id !== question) continue;
      if (seed && String(point.seed) !== seed) continue;
      if (selector && point.selector !== selector) continue;
      if (cluster && String(point.cluster_id) !== cluster) continue;
      if (point.token_fraction < start || point.token_fraction > end) continue;
      const key = trajectoryKey(point);
      if (!selectedTrajectories.has(key)) {
        if (selectedTrajectories.size >= maxTrajectories) continue;
        selectedTrajectories.add(key);
      }
      points.push(point);
    }
    return {
      points: evenlyCapped(points, Number(payload.max_points)),
      matchingCount: points.length,
    };
  }

  function tracesForPoints(points) {
    const groups = new Map();
    for (const point of points) {
      const key = `${trajectoryKey(point)}::${point.selector ?? ""}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(point);
    }

    return [...groups.values()].flatMap(group => {
      group.sort((a, b) => Number(a.token_idx) - Number(b.token_idx));
      const correctness = group[0].is_correct;
      const lineColor = correctness === true
        ? "rgba(22,128,74,0.52)"
        : correctness === false
          ? "rgba(189,63,53,0.55)"
          : "rgba(107,114,128,0.48)";
      const markerColors = group.map(point => pointColor(point, correctness));
      const name = `${group[0].sample_id} · ${group[0].seed}${group[0].selector ? ` · ${group[0].selector}` : ""}`;
      const common = {
        type: "scatter3d",
        name,
        x: group.map(point => point.x),
        y: group.map(point => point.y),
        z: group.map(point => point.z),
        customdata: group,
        text: group.map(hoverText),
        hoverinfo: "text",
        showlegend: false,
      };
      return [
        {
          ...common,
          mode: "lines+markers",
          marker: { size: 4, color: markerColors, opacity: 0.9 },
          line: { width: 4, color: lineColor },
        },
        endpointTrace(`${name} · start`, group[0], "triangle", markerColors[0]),
        endpointTrace(`${name} · end`, group.at(-1), "square", markerColors.at(-1)),
      ];
    });
  }

  function plotLayout() {
    const component = payload.method?.toUpperCase() ?? "Projection";
    return {
      autosize: true,
      margin: { l: 0, r: 0, t: 0, b: 0 },
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      showlegend: false,
      hoverlabel: {
        bgcolor: "#17211c",
        bordercolor: "#17211c",
        font: { color: "#ffffff", size: 12 },
      },
      scene: {
        bgcolor: "#fbfcfb",
        dragmode: "orbit",
        aspectmode: "cube",
        xaxis: axisStyle(`${component} 1`),
        yaxis: axisStyle(`${component} 2`),
        zaxis: axisStyle(`${component} 3`),
      },
      uirevision: activePlot?.path ?? "projection",
    };
  }

  function bindPlotClick() {
    const plot = $("plot3d");
    plot.removeAllListeners?.("plotly_click");
    plot.on?.("plotly_click", event => {
      const point = event.points?.[0]?.customdata;
      if (point) renderInspector(point);
    });
  }

  function improveModebarAccessibility() {
    for (const button of $("plot3d").querySelectorAll(".modebar-btn[data-title]")) {
      button.setAttribute("aria-label", button.dataset.title);
      button.setAttribute("role", "button");
      button.tabIndex = 0;
      if (button.dataset.keyboardBound) continue;
      button.dataset.keyboardBound = "true";
      button.addEventListener("keydown", event => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        button.click();
      });
    }
  }

  function renderInspector(point) {
    const metrics = [
      ["Question", point.sample_id],
      ["Sub-run", point.seed],
      [point.step_idx !== undefined ? "Step" : "Token", point.step_idx ?? point.token_idx],
      ["Position", `${Math.round(Number(point.token_fraction ?? 0) * 100)}%`],
      ["Selector", point.selector],
      ["Cluster", point.cluster_id],
      ["Variance", formatMetric(point.variance)],
      ["Direction", formatMetric(point.direction_norm)],
      ["Nudge", formatMetric(point.nudge_norm)],
      ["Answer", point.produced_answer],
    ].filter(([, value]) => value !== undefined && value !== null && value !== "");
    $("point-inspector").innerHTML = `
      <div>
        <p class="eyebrow">Selected point</p>
        <h2 id="point-inspector-title">${point.step_idx !== undefined ? `Step ${escapeHtml(point.step_idx)}` : `Token ${escapeHtml(point.token_idx)}`}</h2>
      </div>
      <dl>
        ${metrics.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")}
        ${point.step_text ? `<div class="inspector-text">${escapeHtml(point.step_text)}</div>` : ""}
      </dl>`;
  }

  function clearFilters() {
    $("plot-question").value = "";
    updateSeedOptions("");
    $("plot-selector").value = $("plot-selector").options[1]?.value ?? "";
    $("plot-cluster").value = "";
    $("plot-color-mode").value = "correctness";
    $("plot-max-trajectories").value = 12;
    $("plot-token-start").value = 0;
    $("plot-token-end").value = 100;
    updateRangeOutputs();
    render();
    syncQuery();
  }

  function resetCamera() {
    if (!window.Plotly || !payload) return;
    window.Plotly.relayout("plot3d", { "scene.camera": null });
    $("plot-link-status").textContent = "Camera reset";
    setTimeout(() => { $("plot-link-status").textContent = ""; }, 1200);
  }

  async function copyViewLink() {
    syncQuery();
    try {
      await navigator.clipboard.writeText(window.location.href);
      $("plot-link-status").textContent = "Link copied";
    } catch {
      $("plot-link-status").textContent = "Copy unavailable";
    }
    setTimeout(() => { $("plot-link-status").textContent = ""; }, 1500);
  }

  function updateRangeOutputs() {
    $("plot-max-output").value = $("plot-max-trajectories").value;
    $("plot-start-output").value = `${$("plot-token-start").value}%`;
    $("plot-end-output").value = `${$("plot-token-end").value}%`;
  }

  function syncQuery() {
    setQuery({
      source: $("plot-source").value,
      question: $("plot-question").value,
      seed: $("plot-seed").value,
      selector: $("plot-selector").value,
      cluster: $("plot-cluster").value,
      color: $("plot-color-mode").value === "correctness" ? "" : $("plot-color-mode").value,
      limit: $("plot-max-trajectories").value === "12" ? "" : $("plot-max-trajectories").value,
      start: $("plot-token-start").value === "0" ? "" : $("plot-token-start").value,
      end: $("plot-token-end").value === "100" ? "" : $("plot-token-end").value,
    });
  }

  function showLoading(visible) {
    $("plot-loading").hidden = !visible;
  }

  function showPlotMessage(message) {
    $("plot3d").innerHTML = `<div class="empty-state">${escapeHtml(message)}</div>`;
    $("plot-status").textContent = "";
  }

  function renderStaticPlots() {
    const { run } = getState();
    $("static-plots").innerHTML = (run.plots ?? []).map(plot => `
      <figure>
        <img src="${escapeHtml(plot.path)}" alt="${escapeHtml(plot.method)} projection, layer ${escapeHtml(plot.layer)}">
        <figcaption>${escapeHtml(plot.method.toUpperCase())} · layer ${escapeHtml(plot.layer)} · ${formatNumber(plot.trajectories)} trajectories</figcaption>
      </figure>`).join("") || `<p class="muted">No static plots were generated for this run.</p>`;
  }

  return { load };
}

function trajectoryKey(point) {
  return `${point.sample_id}::${point.seed}`;
}

function evenlyCapped(items, maxItems) {
  if (!Number.isFinite(maxItems) || maxItems <= 0 || items.length <= maxItems) return items;
  if (maxItems === 1) return [items[0]];
  return Array.from(
    { length: maxItems },
    (_, index) => items[Math.floor(index * (items.length - 1) / (maxItems - 1))],
  );
}

function plotLabel(plot) {
  const level = plot.plot_type === "step" ? "Step averages" : "Token states";
  return `${level} · ${plot.method.toUpperCase()} · layer ${plot.layer}`;
}

function pointColor(point, correctness) {
  if ($("plot-color-mode").value === "cluster" && point.cluster_id !== undefined) {
    return CLUSTER_COLORS[Math.abs(Number(point.cluster_id)) % CLUSTER_COLORS.length];
  }
  const fraction = Math.max(0, Math.min(1, Number(point.token_fraction ?? 0)));
  if (correctness === true) return `hsl(148 58% ${68 - fraction * 34}%)`;
  if (correctness === false) return `hsl(5 68% ${70 - fraction * 32}%)`;
  return `hsl(210 10% ${68 - fraction * 30}%)`;
}

function endpointTrace(name, point, symbol, color) {
  if (symbol === "triangle") {
    return {
      type: "scatter3d",
      mode: "text",
      name,
      showlegend: false,
      x: [point.x],
      y: [point.y],
      z: [point.z],
      customdata: [point],
      text: ["▲"],
      textfont: { color, size: 18 },
      hovertext: [`${escapeHtml(name)}<br>${hoverText(point)}`],
      hoverinfo: "text",
    };
  }
  return {
    type: "scatter3d",
    mode: "markers",
    name,
    showlegend: false,
    x: [point.x],
    y: [point.y],
    z: [point.z],
    customdata: [point],
    text: [`${escapeHtml(name)}<br>${hoverText(point)}`],
    hoverinfo: "text",
    marker: {
      symbol,
      size: 8,
      color,
      line: { color: "#17211c", width: 1.5 },
    },
  };
}

function hoverText(point) {
  return [
    `<b>${escapeHtml(point.sample_id)}</b>`,
    `sub-run ${escapeHtml(point.seed)}`,
    point.selector ? escapeHtml(point.selector) : null,
    point.step_idx !== undefined ? `step ${escapeHtml(point.step_idx)}` : `token ${escapeHtml(point.token_idx)}`,
    point.cluster_id !== undefined ? `cluster ${escapeHtml(point.cluster_id)}` : null,
    `position ${Math.round(Number(point.token_fraction ?? 0) * 100)}%`,
    `answer ${escapeHtml(point.produced_answer ?? "—")}`,
    `correct ${escapeHtml(point.is_correct)}`,
    point.step_text ? escapeHtml(point.step_text).slice(0, 220) : null,
  ].filter(Boolean).join("<br>");
}

function axisStyle(title) {
  return {
    title: { text: title, font: { size: 11, color: "#5c6a62" } },
    color: "#5c6a62",
    gridcolor: "#dce3df",
    zerolinecolor: "#b9c5be",
    showbackground: true,
    backgroundcolor: "#fbfcfb",
    tickfont: { size: 9 },
  };
}

function validRange(value, min, max, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(min, Math.min(max, number)) : fallback;
}

function formatMetric(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(2) : value;
}
