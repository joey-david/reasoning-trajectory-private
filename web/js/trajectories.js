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

export function createTrajectoryView({ getState, setQuery, openGeneration }) {
  const payloadCache = new Map();
  let payload = null;
  let activePlot = null;
  let loadSequence = 0;
  let interactionVersion = 0;
  let recreatePlotOnNextRender = false;

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
  for (const id of [
    "plot-show-lines",
    "plot-show-points",
    "plot-show-endpoints",
    "plot-hover-highlight",
    "plot-line-width",
    "plot-point-size",
    "plot-start-correct-color",
    "plot-start-incorrect-color",
    "plot-start-unknown-color",
    "plot-end-correct-color",
    "plot-end-incorrect-color",
    "plot-end-unknown-color",
  ]) {
    $(id).addEventListener("input", debounce(() => {
      renderVisualChange();
    }, 50));
    $(id).addEventListener("change", () => {
      renderVisualChange();
    });
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
    $("plot-color-mode").value = ["activation_delta", "cluster"].includes(route.color)
      ? route.color
      : "correctness";
    $("plot-show-lines").checked = route.lines !== "0";
    $("plot-show-points").checked = route.points !== "0";
    $("plot-show-endpoints").checked = route.endpoints !== "0";
    $("plot-hover-highlight").checked = route.hover_highlight !== "0";
    $("plot-line-width").value = validRange(route.line_width, 0.5, 12, 1.5);
    $("plot-point-size").value = validRange(route.point_size, 1, 10, 3);
    setColorValue("plot-start-correct-color", route.start_correct, "#7dddb4");
    setColorValue("plot-start-incorrect-color", route.start_incorrect, "#f09186");
    setColorValue("plot-start-unknown-color", route.start_unknown, "#aeb6bd");
    setColorValue("plot-end-correct-color", route.end_correct, "#1c8f5b");
    setColorValue("plot-end-incorrect-color", route.end_incorrect, "#c4483c");
    setColorValue("plot-end-unknown-color", route.end_unknown, "#68737b");
    updateSeedOptions(route.seed);
    updateRangeOutputs();
    updateVisualOutputs();
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
      showPlotMessage("No interactive projections are available. Run scripts/analysis/analyze.py for this run.");
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
    $("plot-color-mode").querySelector('option[value="activation_delta"]').disabled = !hasMetric(
      "direction_norm",
      payload.points ?? [],
    );
    if (!clusters.length && $("plot-color-mode").value === "cluster") {
      $("plot-color-mode").value = "correctness";
    }
    if ($("plot-color-mode").value === "activation_delta" && !hasMetric("direction_norm", payload.points ?? [])) {
      $("plot-color-mode").value = "correctness";
    }
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
    const trust = Number(payload.diagnostics?.trustworthiness);
    const trustNote = Number.isFinite(trust)
      ? ` · projection trust ${(trust * 100).toFixed(1)}%`
      : "";
    const warning = payload.diagnostics?.warning ?? "";
    $("plot-status").textContent = `${pointCount} · ${formatNumber(trajectoryCount)} trajectories${samplingNote}${trustNote}`;
    $("plot-status").classList.toggle("warning", Boolean(warning));
    $("plot-status").title = warning;

    if (!points.length) {
      window.Plotly.purge("plot3d");
      showPlotMessage("No points match the current filters.");
      return;
    }

    const traces = tracesForPoints(points, trajectoryCount > 1);
    if (!traces.length) {
      window.Plotly.purge("plot3d");
      showPlotMessage("Enable lines, points, or endpoint markers to render the projection.");
      return;
    }
    const shouldRecreate = recreatePlotOnNextRender;
    recreatePlotOnNextRender = false;
    const plotMethod = shouldRecreate ? window.Plotly.newPlot : window.Plotly.react;
    if (shouldRecreate) {
      window.Plotly.purge("plot3d");
    }
    plotMethod("plot3d", traces, plotLayout(), {
      responsive: true,
      scrollZoom: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["select2d", "lasso2d"],
    }).then(() => {
      improveModebarAccessibility();
      bindPlotInteractions(trajectoryCount);
    }).catch(error => {
      showPlotMessage(`Plot rendering failed: ${error.message}`);
    });
  }

  function renderVisualChange() {
    updateVisualOutputs();
    recreatePlotOnNextRender = true;
    render();
    syncQuery();
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

  function tracesForPoints(points, multipleTrajectories) {
    const groups = new Map();
    const visuals = visualSettings();
    const rowsByTrajectory = new Map(
      getState().rows.map(row => [trajectoryKey(row), row]),
    );
    const pointHoverText = point => hoverText(
      point,
      transcriptSlice(point, rowsByTrajectory.get(trajectoryKey(point))),
      multipleTrajectories,
    );
    for (const point of points) {
      const key = `${trajectoryKey(point)}::${point.selector ?? ""}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(point);
    }

    const traces = [...groups.values()].flatMap(group => {
      group.sort((a, b) => Number(a.token_idx) - Number(b.token_idx));
      const correctness = group[0].is_correct;
      const lineColor = correctness === true
        ? "rgba(22,128,74,0.52)"
        : correctness === false
          ? "rgba(189,63,53,0.55)"
          : "rgba(107,114,128,0.48)";
      const colorScale = metricScale(points, "direction_norm");
      const markerColors = group.map(point => pointColor(point, correctness, colorScale));
      const name = `${group[0].sample_id} · ${group[0].seed}${group[0].selector ? ` · ${group[0].selector}` : ""}`;
      const common = {
        type: "scatter3d",
        name,
        x: group.map(point => point.x),
        y: group.map(point => point.y),
        z: group.map(point => point.z),
        customdata: group,
        text: group.map(pointHoverText),
        hoverinfo: "text",
        showlegend: false,
      };
      const groupTraces = [];
      if (visuals.showLines) {
        groupTraces.push({
          ...common,
          mode: "lines",
          meta: { trajectoryTrace: "line" },
          line: { width: visuals.lineWidth, color: lineColor },
        });
      }
      if (visuals.showPoints) {
        groupTraces.push({
          ...common,
          mode: "markers",
          meta: { trajectoryTrace: "point" },
          marker: { size: visuals.pointSize, color: markerColors, opacity: 0.9 },
        });
      }
      if (visuals.showEndpoints) {
        groupTraces.push(
          endpointTrace(
            `${name} · start`,
            group[0],
            "triangle",
            endpointColor("start", correctness),
            pointHoverText(group[0]),
          ),
          endpointTrace(
            `${name} · end`,
            group.at(-1),
            "square",
            endpointColor("end", correctness),
            pointHoverText(group.at(-1)),
          ),
        );
      }
      return groupTraces;
    });
    return traces;
  }

  function plotLayout() {
    const component = payload.method?.toUpperCase() ?? "Projection";
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
        xaxis: axisStyle(`${component} 1`),
        yaxis: axisStyle(`${component} 2`),
        zaxis: axisStyle(`${component} 3`),
      },
      uirevision: activePlot?.path ?? "projection",
    };
  }

  function bindPlotInteractions(trajectoryCount) {
    const plot = $("plot3d");
    const version = ++interactionVersion;
    const multipleTrajectories = trajectoryCount > 1;
    const hoverHighlight = $("plot-hover-highlight").checked;
    const hiddenTrace = -1;
    const baseStyles = new Map();
    if (multipleTrajectories) {
      for (let traceIndex = 0; traceIndex < plot.data.length; traceIndex += 1) {
        if (isMainTrajectoryTrace(plot.data[traceIndex])) {
          baseStyles.set(traceIndex, traceStyle(plot.data[traceIndex]));
        }
      }
    }
    let desiredTrace = hiddenTrace;
    let appliedTrace = hiddenTrace;
    let updateScheduled = false;
    let updateInFlight = false;
    let latestCamera = currentCamera(plot);

    const scheduleUpdate = () => {
      if (updateScheduled || updateInFlight || version !== interactionVersion) return;
      updateScheduled = true;
      requestAnimationFrame(() => {
        updateScheduled = false;
        applyLatestHighlight();
      });
    };

    const applyLatestHighlight = async () => {
      if (updateInFlight || appliedTrace === desiredTrace || version !== interactionVersion) return;
      const nextTrace = desiredTrace;
      updateInFlight = true;
      try {
        await transitionHighlight(
          plot,
          appliedTrace,
          nextTrace,
          baseStyles,
          latestCamera,
        );
        appliedTrace = nextTrace;
      } catch {
        desiredTrace = hiddenTrace;
        appliedTrace = hiddenTrace;
      } finally {
        updateInFlight = false;
        if (appliedTrace !== desiredTrace) scheduleUpdate();
      }
    };

    const clearHighlight = () => {
      setPlotCursor(plot, "grab");
      if (!multipleTrajectories || !hoverHighlight) return;
      desiredTrace = hiddenTrace;
      scheduleUpdate();
    };

    plot.removeAllListeners?.("plotly_click");
    plot.removeAllListeners?.("plotly_hover");
    plot.removeAllListeners?.("plotly_unhover");
    plot.removeAllListeners?.("plotly_relayout");
    if (plot._trajectoryMouseleave) {
      plot.removeEventListener("mouseleave", plot._trajectoryMouseleave);
    }
    plot._trajectoryMouseleave = clearHighlight;
    plot.addEventListener("mouseleave", clearHighlight);
    plot.on?.("plotly_relayout", update => {
      if (update["scene.camera"]) {
        latestCamera = copyCamera(update["scene.camera"]);
      } else if (Object.keys(update).some(key => key.startsWith("scene.camera."))) {
        latestCamera = currentCamera(plot);
      }
    });
    plot.on?.("plotly_hover", event => {
      setPlotCursor(plot, "pointer");
      if (!multipleTrajectories || !hoverHighlight) return;
      const curveNumber = event.points?.[0]?.curveNumber;
      if (!Number.isInteger(curveNumber)) return;
      const mainTrace = mainTraceForCurve(plot, curveNumber);
      if (mainTrace < 0) return;
      if (desiredTrace === mainTrace) return;
      desiredTrace = mainTrace;
      scheduleUpdate();
    });
    plot.on?.("plotly_unhover", clearHighlight);
    plot.on?.("plotly_click", event => {
      const clicked = event.points?.[0];
      const point = clicked?.customdata;
      if (!point) return;
      const trace = plot.data?.[clicked.curveNumber];
      if (trace?.meta?.trajectoryTrace === "line") {
        isolateTrajectory(point);
        return;
      }
      openGeneration(point);
    });
  }

  function isolateTrajectory(point) {
    $("plot-question").value = String(point.sample_id);
    updateSeedOptions(point.seed);
    $("plot-seed").value = String(point.seed);
    render();
    syncQuery(true);
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
      ["Activation change", formatMetric(point.direction_norm)],
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
    $("plot-show-lines").checked = true;
    $("plot-show-points").checked = true;
    $("plot-show-endpoints").checked = true;
    $("plot-hover-highlight").checked = true;
    $("plot-line-width").value = 1.5;
    $("plot-point-size").value = 3;
    $("plot-max-trajectories").value = 12;
    $("plot-token-start").value = 0;
    $("plot-token-end").value = 100;
    updateRangeOutputs();
    updateVisualOutputs();
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
    $("plot-max-output").textContent = $("plot-max-trajectories").value;
    $("plot-start-output").textContent = `${$("plot-token-start").value}%`;
    $("plot-end-output").textContent = `${$("plot-token-end").value}%`;
  }

  function updateVisualOutputs() {
    $("plot-line-width-output").textContent = $("plot-line-width").value;
    $("plot-point-size-output").textContent = $("plot-point-size").value;
  }

  function syncQuery(push = false) {
    setQuery({
      source: $("plot-source").value,
      question: $("plot-question").value,
      seed: $("plot-seed").value,
      selector: $("plot-selector").value,
      cluster: $("plot-cluster").value,
      color: $("plot-color-mode").value === "correctness" ? "" : $("plot-color-mode").value,
      lines: $("plot-show-lines").checked ? "" : "0",
      points: $("plot-show-points").checked ? "" : "0",
      endpoints: $("plot-show-endpoints").checked ? "" : "0",
      hover_highlight: $("plot-hover-highlight").checked ? "" : "0",
      line_width: $("plot-line-width").value === "1.5" ? "" : $("plot-line-width").value,
      point_size: $("plot-point-size").value === "3" ? "" : $("plot-point-size").value,
      start_correct: colorQueryValue("plot-start-correct-color", "#7dddb4"),
      start_incorrect: colorQueryValue("plot-start-incorrect-color", "#f09186"),
      start_unknown: colorQueryValue("plot-start-unknown-color", "#aeb6bd"),
      end_correct: colorQueryValue("plot-end-correct-color", "#1c8f5b"),
      end_incorrect: colorQueryValue("plot-end-incorrect-color", "#c4483c"),
      end_unknown: colorQueryValue("plot-end-unknown-color", "#68737b"),
      limit: $("plot-max-trajectories").value === "12" ? "" : $("plot-max-trajectories").value,
      start: $("plot-token-start").value === "0" ? "" : $("plot-token-start").value,
      end: $("plot-token-end").value === "100" ? "" : $("plot-token-end").value,
    }, push);
  }

  function showLoading(visible) {
    $("plot-loading").hidden = !visible;
  }

  function showPlotMessage(message) {
    $("plot3d").innerHTML = `<div class="empty-state">${escapeHtml(message)}</div>`;
    $("plot-status").textContent = "";
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

function pointColor(point, correctness, colorScale) {
  if ($("plot-color-mode").value === "cluster" && point.cluster_id !== undefined) {
    return CLUSTER_COLORS[Math.abs(Number(point.cluster_id)) % CLUSTER_COLORS.length];
  }
  if ($("plot-color-mode").value === "activation_delta" && colorScale) {
    return activationColor(colorScale(Number(point.direction_norm)));
  }
  const fraction = Math.max(0, Math.min(1, Number(point.token_fraction ?? 0)));
  if (correctness === true) return `hsl(148 58% ${68 - fraction * 34}%)`;
  if (correctness === false) return `hsl(5 68% ${70 - fraction * 32}%)`;
  return `hsl(210 10% ${68 - fraction * 30}%)`;
}

function hasMetric(key, points) {
  return points.some(point => Number.isFinite(Number(point[key])));
}

function metricScale(points, key) {
  if ($("plot-color-mode").value !== "activation_delta") return null;
  const values = points.map(point => Number(point[key])).filter(Number.isFinite);
  if (!values.length) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (max <= min) return () => 0.5;
  return value => Math.max(0, Math.min(1, (value - min) / (max - min)));
}

function activationColor(fraction) {
  const hue = 205 - fraction * 165;
  const lightness = 68 - fraction * 18;
  return `hsl(${hue} 86% ${lightness}%)`;
}

function visualSettings() {
  return {
    showLines: $("plot-show-lines").checked,
    showPoints: $("plot-show-points").checked,
    showEndpoints: $("plot-show-endpoints").checked,
    lineWidth: Number($("plot-line-width").value),
    pointSize: Number($("plot-point-size").value),
  };
}

function endpointColor(position, correctness) {
  const outcome = correctness === true ? "correct" : correctness === false ? "incorrect" : "unknown";
  return $(`plot-${position}-${outcome}-color`).value;
}

function setColorValue(id, value, fallback) {
  $(id).value = /^#[0-9a-f]{6}$/i.test(String(value ?? "")) ? value : fallback;
}

function colorQueryValue(id, fallback) {
  const value = $(id).value.toLowerCase();
  return value === fallback.toLowerCase() ? "" : value;
}

function visualStateKey() {
  return [
    $("plot-show-lines").checked ? "lines" : "nolines",
    $("plot-show-points").checked ? "points" : "nopoints",
    $("plot-show-endpoints").checked ? "endpoints" : "noendpoints",
    $("plot-hover-highlight").checked ? "hover" : "nohover",
    $("plot-color-mode").value,
    $("plot-line-width").value,
    $("plot-point-size").value,
    $("plot-start-correct-color").value,
    $("plot-start-incorrect-color").value,
    $("plot-start-unknown-color").value,
    $("plot-end-correct-color").value,
    $("plot-end-incorrect-color").value,
    $("plot-end-unknown-color").value,
  ].join(":");
}

function endpointTrace(name, point, symbol, color, hover) {
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
      meta: { trajectoryTrace: "point" },
      text: ["▲"],
      textfont: { color, size: 18 },
      hovertext: [`${escapeHtml(name)}<br>${hover}`],
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
    meta: { trajectoryTrace: "point" },
    text: [`${escapeHtml(name)}<br>${hover}`],
    hoverinfo: "text",
    marker: {
      symbol,
      size: 8,
      color,
      line: { color: "#111417", width: 1.5 },
    },
  };
}

function isMainTrajectoryTrace(trace) {
  return trace?.meta?.trajectoryTrace === "line";
}

function mainTraceForCurve(plot, curveNumber) {
  if (isMainTrajectoryTrace(plot.data[curveNumber])) return curveNumber;
  for (let index = curveNumber - 1; index >= 0; index -= 1) {
    if (isMainTrajectoryTrace(plot.data[index]) && plot.data[index].name === plot.data[curveNumber]?.name) return index;
  }
  return -1;
}

function hoverText(point, transcriptText, multipleTrajectories) {
  const content = point.step_idx !== undefined
    ? formatStepText(transcriptText)
    : formatTokenCharacters(transcriptText);
  return [
    `<b>${escapeHoverText(point.sample_id)}</b>`,
    `position ${Math.round(Number(point.token_fraction ?? 0) * 100)}%`,
    content,
    multipleTrajectories
      ? "<b>Click to isolate this run</b>"
      : "<b>Click to open transcript</b>",
  ].filter(Boolean).join("<br>");
}

function transcriptSlice(point, row) {
  const text = String(row?.produced_text ?? "");
  const start = Number(point.char_start);
  const end = Number(point.char_end);
  if (Number.isInteger(start) && Number.isInteger(end) && start >= 0 && end >= start && end <= text.length) {
    return text.slice(start, end);
  }
  return String(point.step_text ?? "");
}

function formatTokenCharacters(text) {
  if (text === "") return "⟨no visible characters⟩";
  return escapeHoverText(text)
    .replace(/ /g, "␠")
    .replace(/\t/g, "⇥")
    .replace(/\n/g, "↵");
}

function formatStepText(text) {
  return escapeHoverText(text).replace(/\n/g, "<br>");
}

function escapeHoverText(value) {
  return String(value ?? "")
    .replace(/&/g, "＆")
    .replace(/</g, "‹")
    .replace(/>/g, "›");
}

function setPlotCursor(plot, cursor) {
  plot.style.cursor = cursor;
  for (const element of plot.querySelectorAll("canvas, .nsewdrag")) {
    element.style.cursor = cursor;
  }
}

function traceStyle(trace) {
  return {
    lineColor: trace.line?.color,
    lineWidth: trace.line?.width,
    markerColor: trace.marker?.color,
    markerSize: trace.marker?.size,
    markerOpacity: trace.marker?.opacity,
  };
}

function transitionHighlight(
  plot,
  previousTrace,
  nextTrace,
  baseStyles,
  camera,
) {
  const traceIndices = [];
  const lineColors = [];
  const lineWidths = [];
  const markerColors = [];
  const markerSizes = [];
  const markerOpacities = [];
  if (previousTrace >= 0) {
    const style = baseStyles.get(previousTrace);
    traceIndices.push(previousTrace);
    lineColors.push(style.lineColor);
    lineWidths.push(style.lineWidth);
    markerColors.push(style.markerColor);
    markerSizes.push(style.markerSize);
    markerOpacities.push(style.markerOpacity);
  }
  if (nextTrace >= 0) {
    const style = baseStyles.get(nextTrace);
    const highlightLineWidth = Math.max(Number(style?.lineWidth ?? 1.5) * 2.5, Number(style?.lineWidth ?? 1.5) + 2);
    const highlightMarkerSize = Math.max(Number(style?.markerSize ?? 3) + 2.5, Number(style?.markerSize ?? 3) * 1.7);
    traceIndices.push(nextTrace);
    lineColors.push("#d99a00");
    lineWidths.push(highlightLineWidth);
    markerColors.push("#f2b21b");
    markerSizes.push(highlightMarkerSize);
    markerOpacities.push(1);
  }
  if (!traceIndices.length) return;
  return window.Plotly.update(
    plot,
    {
      "line.color": lineColors,
      "line.width": lineWidths,
      "marker.color": markerColors,
      "marker.size": markerSizes,
      "marker.opacity": markerOpacities,
    },
    camera ? { "scene.camera": camera } : {},
    traceIndices,
  );
}

function currentCamera(plot) {
  const camera = plot.layout?.scene?.camera ?? plot._fullLayout?.scene?.camera;
  return copyCamera(camera);
}

function copyCamera(camera) {
  if (!camera) return null;
  return {
    eye: { ...camera.eye },
    center: { ...camera.center },
    up: { ...camera.up },
    projection: { ...camera.projection },
  };
}

function axisStyle(title) {
  return {
    title: { text: title, font: { size: 11, color: "#8f9aa2" } },
    color: "#8f9aa2",
    gridcolor: "#252c31",
    zerolinecolor: "#343d44",
    showbackground: true,
    backgroundcolor: "#111417",
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
