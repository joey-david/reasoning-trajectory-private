import {
  $,
  debounce,
  escapeHtml,
  formatNumber,
  setOptions,
} from "./ui.js";

export function createLayerwiseView({ getState, setQuery }) {
  const payloadCache = new Map();
  let payload = null;
  let activePlot = null;
  let loadSequence = 0;
  let recreatePlotOnNextRender = false;

  $("layerwise-source").addEventListener("change", () => loadSelectedPlot());
  $("layerwise-question").addEventListener("change", () => {
    updateSeedOptions();
    render();
    syncQuery();
  });
  for (const id of ["layerwise-seed", "layerwise-color-mode"]) {
    $(id).addEventListener("change", () => {
      render();
      syncQuery();
    });
  }
  for (const id of ["layerwise-max-trajectories", "layerwise-token-start", "layerwise-token-end"]) {
    $(id).addEventListener("input", debounce(() => {
      updateRangeOutputs();
      render();
      syncQuery();
    }, 50));
  }
  for (const id of [
    "layerwise-show-lines",
    "layerwise-show-points",
    "layerwise-hover-highlight",
    "layerwise-line-width",
    "layerwise-point-size",
    "layerwise-opacity",
  ]) {
    $(id).addEventListener("input", debounce(() => {
      renderVisualChange();
    }, 50));
    $(id).addEventListener("change", () => {
      renderVisualChange();
    });
  }
  $("layerwise-clear").addEventListener("click", clearFilters);
  $("layerwise-reset-camera").addEventListener("click", resetCamera);
  $("layerwise-copy-view-link").addEventListener("click", copyViewLink);

  function load(route) {
    const plots = getState().run.layerwise_plots ?? [];
    setOptions(
      "layerwise-source",
      plots.map((plot, index) => [index, plotLabel(plot)]),
      null,
      route.lsource,
    );
    const { rows } = getState();
    setOptions(
      "layerwise-question",
      [...new Set(rows.map(row => row.sample_id))].sort(),
      { value: "", label: "All questions" },
      route.question,
    );
    $("layerwise-max-trajectories").value = validRange(route.llimit, 1, 30, 4);
    $("layerwise-token-start").value = validRange(route.lstart, 0, 100, 0);
    $("layerwise-token-end").value = validRange(route.lend, 0, 100, 100);
    $("layerwise-color-mode").value = ["token", "correctness"].includes(route.lcolor)
      ? route.lcolor
      : "value";
    $("layerwise-show-lines").checked = route.llines !== "0";
    $("layerwise-show-points").checked = route.lpoints !== "0";
    $("layerwise-hover-highlight").checked = route.lhover !== "0";
    $("layerwise-line-width").value = validRange(route.lline_width, 0.5, 12, 1.5);
    $("layerwise-point-size").value = validRange(route.lpoint_size, 1, 10, 3);
    $("layerwise-opacity").value = validRange(route.lopacity, 10, 100, 75);
    updateSeedOptions(route.seed);
    updateRangeOutputs();
    updateVisualOutputs();
    loadSelectedPlot(route);
  }

  async function loadSelectedPlot() {
    const sequence = ++loadSequence;
    activePlot = (getState().run.layerwise_plots ?? [])[Number($("layerwise-source").value || 0)] ?? null;
    payload = null;
    showLoading(true);
    $("layerwise-plot-title").textContent = activePlot ? plotLabel(activePlot) : "Layerwise plot";
    if (!activePlot) {
      showLoading(false);
      showPlotMessage("No layerwise plots are available. Enable layer_variations and rerun scripts/analysis/analyze.py for this run.");
      return;
    }

    try {
      if (!payloadCache.has(activePlot.path)) {
        const response = await fetch(activePlot.path, { cache: "no-store" });
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        payloadCache.set(activePlot.path, await response.json());
      }
      if (sequence !== loadSequence) return;
      payload = payloadCache.get(activePlot.path);
      showLoading(false);
      render();
      syncQuery();
    } catch (error) {
      if (sequence !== loadSequence) return;
      showLoading(false);
      showPlotMessage(`Could not load this layerwise plot: ${error.message}`);
    }
  }

  function updateSeedOptions(preferred = $("layerwise-seed").value) {
    const question = $("layerwise-question").value;
    const seeds = [...new Set(
      getState().rows
        .filter(row => !question || row.sample_id === question)
        .map(row => row.seed),
    )].sort((a, b) => Number(a) - Number(b));
    setOptions("layerwise-seed", seeds, { value: "", label: "All sub-runs" }, preferred);
  }

  function render() {
    if (!payload) return;
    if (!window.Plotly) {
      showPlotMessage("Plotly did not load. Check the network connection and reload.");
      return;
    }

    const { points, matchingCount } = filteredPoints();
    const traceCount = new Set(points.map(traceKey)).size;
    const pointCount = points.length < matchingCount
      ? `${formatNumber(points.length)} of ${formatNumber(matchingCount)} matching points`
      : `${formatNumber(points.length)} points`;
    $("layerwise-plot-status").textContent = `${pointCount} · ${formatNumber(traceCount)} traces`;

    if (!points.length) {
      window.Plotly.purge("layerwise3d");
      showPlotMessage("No points match the current filters.");
      return;
    }

    const traces = tracesForPoints(points);
    if (!traces.length) {
      window.Plotly.purge("layerwise3d");
      showPlotMessage("Enable lines or points to render the layerwise plot.");
      return;
    }

    const shouldRecreate = recreatePlotOnNextRender;
    recreatePlotOnNextRender = false;
    const plotMethod = shouldRecreate ? window.Plotly.newPlot : window.Plotly.react;
    if (shouldRecreate) window.Plotly.purge("layerwise3d");
    plotMethod("layerwise3d", traces, plotLayout(points), {
      responsive: true,
      scrollZoom: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["select2d", "lasso2d"],
    }).then(() => {
      improveModebarAccessibility();
      bindPlotInteractions();
    }).catch(error => {
      showPlotMessage(`Plot rendering failed: ${error.message}`);
    });
  }

  function filteredPoints() {
    const question = $("layerwise-question").value;
    const seed = $("layerwise-seed").value;
    const start = Math.min(Number($("layerwise-token-start").value), Number($("layerwise-token-end").value)) / 100;
    const end = Math.max(Number($("layerwise-token-start").value), Number($("layerwise-token-end").value)) / 100;
    const maxTraces = Number($("layerwise-max-trajectories").value);
    const selectedTraces = new Set();
    const points = [];
    for (const point of payload.points ?? []) {
      if (question && point.sample_id !== question) continue;
      if (seed && String(point.seed) !== seed) continue;
      if (point.token_fraction < start || point.token_fraction > end) continue;
      const key = traceKey(point);
      if (!selectedTraces.has(key)) {
        if (selectedTraces.size >= maxTraces) continue;
        selectedTraces.add(key);
      }
      points.push(point);
    }
    return { points, matchingCount: points.length };
  }

  function tracesForPoints(points) {
    const visuals = visualSettings();
    const groups = new Map();
    for (const point of points) {
      const key = `${traceKey(point)}::token${point.token_idx}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(point);
    }
    const valueScale = metricScale(points, "value");
    const traces = [];
    for (const group of groups.values()) {
      group.sort((a, b) => Number(a.layer_pair_index) - Number(b.layer_pair_index));
      const first = group[0];
      const name = `${first.sample_id} · ${first.seed} · token ${first.token_idx}`;
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
      if (visuals.showLines) {
        traces.push({
          ...common,
          mode: "lines",
          meta: { layerwiseTrace: "line" },
          line: {
            width: visuals.lineWidth,
            color: traceLineColor(first, group, valueScale, visuals.opacity),
          },
        });
      }
      if (visuals.showPoints) {
        traces.push({
          ...common,
          mode: "markers",
          meta: { layerwiseTrace: "point" },
          marker: {
            size: visuals.pointSize,
            color: group.map(point => pointColor(point, valueScale)),
            opacity: visuals.opacity,
          },
        });
      }
    }
    return traces;
  }

  function plotLayout(points) {
    const metricLabel = payload.metric_label ?? payload.metric ?? "metric";
    return {
      autosize: true,
      margin: { l: 0, r: 0, t: 0, b: 0 },
      paper_bgcolor: "#111417",
      plot_bgcolor: "#111417",
      showlegend: false,
      datarevision: visualStateKey(),
      hoverlabel: {
        bgcolor: "#20262b",
        bordercolor: "#3d474f",
        font: { color: "#f4f5f6", size: 12 },
        align: "left",
      },
      scene: {
        bgcolor: "#111417",
        dragmode: "orbit",
        aspectmode: "cube",
        xaxis: axisStyle("Adjacent layer pair", layerTickValues(points), layerTickText(points)),
        yaxis: axisStyle("Generated token"),
        zaxis: axisStyle(metricLabel),
      },
      uirevision: activePlot?.path ?? "layerwise",
    };
  }

  function bindPlotInteractions() {
    const plot = $("layerwise3d");
    let highlightedCurve = -1;
    plot.removeAllListeners?.("plotly_click");
    plot.removeAllListeners?.("plotly_hover");
    plot.removeAllListeners?.("plotly_unhover");
    plot.on?.("plotly_click", event => {
      const point = event.points?.[0]?.customdata;
      if (point) renderInspector(point);
    });
    plot.on?.("plotly_hover", event => {
      setPlotCursor(plot, "pointer");
      if (!$("layerwise-hover-highlight").checked) return;
      const curveNumber = event.points?.[0]?.curveNumber;
      const lineCurve = lineCurveFor(plot, curveNumber);
      if (lineCurve < 0 || lineCurve === highlightedCurve) return;
      highlightedCurve = lineCurve;
      restyleLine(plot, lineCurve, Math.max(Number($("layerwise-line-width").value) * 2.2, 3));
    });
    plot.on?.("plotly_unhover", () => {
      setPlotCursor(plot, "grab");
      if (highlightedCurve < 0) return;
      const lineCurve = highlightedCurve;
      highlightedCurve = -1;
      restyleLine(plot, lineCurve, Number($("layerwise-line-width").value));
    });
  }

  function renderInspector(point) {
    const metrics = [
      ["Question", point.sample_id],
      ["Sub-run", point.seed],
      ["Token", point.token_idx],
      ["Layer pair", point.layer_pair],
      ["Metric", payload.metric_label ?? payload.metric],
      ["Value", formatMetric(point.value)],
      ["Position", `${Math.round(Number(point.token_fraction ?? 0) * 100)}%`],
      ["Outcome", outcomeLabel(point)],
      ["Answer", point.produced_answer],
    ].filter(([, value]) => value !== undefined && value !== null && value !== "");
    $("layerwise-inspector").innerHTML = `
      <div>
        <p class="eyebrow">Selected point</p>
        <h2 id="layerwise-inspector-title">Token ${escapeHtml(point.token_idx)} · ${escapeHtml(point.layer_pair)}</h2>
      </div>
      <dl>${metrics.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")}</dl>`;
  }

  function renderVisualChange() {
    updateVisualOutputs();
    recreatePlotOnNextRender = true;
    render();
    syncQuery();
  }

  function clearFilters() {
    $("layerwise-question").value = "";
    updateSeedOptions("");
    $("layerwise-seed").value = "";
    $("layerwise-color-mode").value = "value";
    $("layerwise-show-lines").checked = true;
    $("layerwise-show-points").checked = true;
    $("layerwise-hover-highlight").checked = true;
    $("layerwise-line-width").value = 1.5;
    $("layerwise-point-size").value = 3;
    $("layerwise-opacity").value = 75;
    $("layerwise-max-trajectories").value = 4;
    $("layerwise-token-start").value = 0;
    $("layerwise-token-end").value = 100;
    updateRangeOutputs();
    updateVisualOutputs();
    render();
    syncQuery();
  }

  function resetCamera() {
    if (!window.Plotly || !payload) return;
    window.Plotly.relayout("layerwise3d", { "scene.camera": null });
    $("layerwise-link-status").textContent = "Camera reset";
    setTimeout(() => { $("layerwise-link-status").textContent = ""; }, 1200);
  }

  async function copyViewLink() {
    syncQuery();
    try {
      await navigator.clipboard.writeText(window.location.href);
      $("layerwise-link-status").textContent = "Link copied";
    } catch {
      $("layerwise-link-status").textContent = "Copy unavailable";
    }
    setTimeout(() => { $("layerwise-link-status").textContent = ""; }, 1500);
  }

  function updateRangeOutputs() {
    $("layerwise-max-output").textContent = $("layerwise-max-trajectories").value;
    $("layerwise-start-output").textContent = `${$("layerwise-token-start").value}%`;
    $("layerwise-end-output").textContent = `${$("layerwise-token-end").value}%`;
  }

  function updateVisualOutputs() {
    $("layerwise-line-width-output").textContent = $("layerwise-line-width").value;
    $("layerwise-point-size-output").textContent = $("layerwise-point-size").value;
    $("layerwise-opacity-output").textContent = `${$("layerwise-opacity").value}%`;
  }

  function syncQuery(push = false) {
    setQuery({
      lsource: $("layerwise-source").value,
      question: $("layerwise-question").value,
      seed: $("layerwise-seed").value,
      lcolor: $("layerwise-color-mode").value === "value" ? "" : $("layerwise-color-mode").value,
      llines: $("layerwise-show-lines").checked ? "" : "0",
      lpoints: $("layerwise-show-points").checked ? "" : "0",
      lhover: $("layerwise-hover-highlight").checked ? "" : "0",
      lline_width: $("layerwise-line-width").value === "1.5" ? "" : $("layerwise-line-width").value,
      lpoint_size: $("layerwise-point-size").value === "3" ? "" : $("layerwise-point-size").value,
      lopacity: $("layerwise-opacity").value === "75" ? "" : $("layerwise-opacity").value,
      llimit: $("layerwise-max-trajectories").value === "4" ? "" : $("layerwise-max-trajectories").value,
      lstart: $("layerwise-token-start").value === "0" ? "" : $("layerwise-token-start").value,
      lend: $("layerwise-token-end").value === "100" ? "" : $("layerwise-token-end").value,
    }, push);
  }

  function showLoading(visible) {
    $("layerwise-plot-loading").hidden = !visible;
  }

  function showPlotMessage(message) {
    $("layerwise3d").innerHTML = `<div class="empty-state">${escapeHtml(message)}</div>`;
    $("layerwise-plot-status").textContent = "";
  }

  return { load };
}

function traceKey(point) {
  return `${point.sample_id}::${point.seed}`;
}

function plotLabel(plot) {
  return `${plot.metric_label ?? plot.metric} · ${formatNumber(plot.traces)} traces · ${formatNumber(plot.points)} points`;
}

function visualSettings() {
  return {
    showLines: $("layerwise-show-lines").checked,
    showPoints: $("layerwise-show-points").checked,
    lineWidth: Number($("layerwise-line-width").value),
    pointSize: Number($("layerwise-point-size").value),
    opacity: Number($("layerwise-opacity").value) / 100,
  };
}

function traceLineColor(point, group, valueScale, opacity) {
  if ($("layerwise-color-mode").value === "value") {
    const meanHeight = group.reduce((sum, item) => sum + Number(item.value), 0) / group.length;
    return metricColor(valueScale(meanHeight), opacity);
  }
  if (point.is_correct === true) return `rgba(22,128,74,${opacity})`;
  if (point.is_correct === false) return `rgba(189,63,53,${opacity})`;
  return `rgba(107,114,128,${opacity})`;
}

function pointColor(point, valueScale) {
  if ($("layerwise-color-mode").value === "token") {
    const fraction = Math.max(0, Math.min(1, Number(point.token_fraction ?? 0)));
    return `hsl(${208 - fraction * 160} 82% ${68 - fraction * 18}%)`;
  }
  if ($("layerwise-color-mode").value === "correctness") {
    if (point.is_correct === true) return "#63c59d";
    if (point.is_correct === false) return "#e27c6f";
    return "#8f9aa2";
  }
  return metricColor(valueScale(Number(point.value)));
}

function metricScale(points, key) {
  const values = points.map(point => Number(point[key])).filter(Number.isFinite);
  if (!values.length) return () => 0.5;
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (max <= min) return () => 0.5;
  return value => Math.max(0, Math.min(1, (value - min) / (max - min)));
}

function metricColor(fraction, alpha = 1) {
  const stops = [
    [0, [59, 76, 192]],
    [0.5, [221, 220, 220]],
    [1, [180, 4, 38]],
  ];
  const value = Math.max(0, Math.min(1, Number(fraction)));
  const [lower, upper] = value <= 0.5 ? stops.slice(0, 2) : stops.slice(1);
  const local = (value - lower[0]) / (upper[0] - lower[0]);
  const channels = lower[1].map((channel, index) => Math.round(channel + (upper[1][index] - channel) * local));
  return `rgba(${channels.join(",")},${alpha})`;
}

function hoverText(point) {
  return [
    `<b>${escapeHoverText(point.sample_id)}</b>`,
    `seed ${escapeHoverText(point.seed)}`,
    `token ${escapeHoverText(point.token_idx)} (${Math.round(Number(point.token_fraction ?? 0) * 100)}%)`,
    `layers ${escapeHoverText(point.layer_pair)}`,
    `${escapeHoverText(point.value)}`,
    "<b>Click to inspect</b>",
  ].join("<br>");
}

function escapeHoverText(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function layerTickValues(points) {
  return [...new Set(points.map(point => Number(point.layer_pair_index)))].sort((a, b) => a - b);
}

function layerTickText(points) {
  const labels = new Map(points.map(point => [Number(point.layer_pair_index), point.layer_pair]));
  return layerTickValues(points).map(value => labels.get(value) ?? String(value));
}

function axisStyle(title, tickvals = null, ticktext = null) {
  const axis = {
    title: { text: title, font: { size: 11, color: "#8f9aa2" } },
    color: "#8f9aa2",
    gridcolor: "#252c31",
    zerolinecolor: "#343d44",
    showbackground: true,
    backgroundcolor: "#111417",
    tickfont: { size: 9 },
  };
  if (tickvals?.length) {
    axis.tickmode = "array";
    axis.tickvals = tickvals;
    axis.ticktext = ticktext;
  }
  return axis;
}

function improveModebarAccessibility() {
  for (const button of $("layerwise3d").querySelectorAll(".modebar-btn[data-title]")) {
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

function setPlotCursor(plot, cursor) {
  plot.style.cursor = cursor;
  for (const element of plot.querySelectorAll("canvas, .nsewdrag")) {
    element.style.cursor = cursor;
  }
}

function lineCurveFor(plot, curveNumber) {
  if (!Number.isInteger(curveNumber)) return -1;
  if (plot.data?.[curveNumber]?.meta?.layerwiseTrace === "line") return curveNumber;
  const name = plot.data?.[curveNumber]?.name;
  for (let index = curveNumber - 1; index >= 0; index -= 1) {
    const trace = plot.data?.[index];
    if (trace?.meta?.layerwiseTrace === "line" && trace.name === name) return index;
  }
  return -1;
}

function restyleLine(plot, curveNumber, width) {
  if (!window.Plotly || curveNumber < 0) return;
  try {
    Promise.resolve(window.Plotly.restyle(plot, { "line.width": width }, [curveNumber])).catch(() => {});
  } catch {
    // Hover should never take down the interactive plot if Plotly is mid-update.
  }
}

function visualStateKey() {
  return [
    $("layerwise-show-lines").checked ? "lines" : "nolines",
    $("layerwise-show-points").checked ? "points" : "nopoints",
    $("layerwise-hover-highlight").checked ? "hover" : "nohover",
    $("layerwise-color-mode").value,
    $("layerwise-line-width").value,
    $("layerwise-point-size").value,
    $("layerwise-opacity").value,
  ].join(":");
}

function outcomeLabel(point) {
  if (point.is_correct === true) return "Correct";
  if (point.is_correct === false) return "Incorrect";
  return "Unscored";
}

function validRange(value, min, max, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(min, Math.min(max, number)) : fallback;
}

function formatMetric(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(4) : value;
}
