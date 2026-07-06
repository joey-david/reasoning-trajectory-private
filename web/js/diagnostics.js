import { $, escapeHtml, fetchJson, formatNumber } from "./ui.js";

const TOOLS = {
  geometry: {
    title: "Path geometry",
    description: "How far, directly, and smoothly each token-level hidden-state path moves.",
  },
  alignment: {
    title: "Path alignment",
    description: "Whether two sampled solutions follow similar latent shapes after time alignment.",
  },
  compression: {
    title: "Compression",
    description: "How much original-space variance survives a shared low-dimensional PCA bottleneck.",
  },
  basins: {
    title: "Endpoint basins",
    description: "Clusters of final hidden states. Silhouette measures whether those groups are actually separated.",
  },
  failures: {
    title: "Failure divergence",
    description: "Where an incorrect path first departs persistently from its nearest correct path for the same problem.",
  },
};

export function createDiagnosticsView({ getState, setQuery }) {
  let report = null;
  let loadedUrl = null;
  let activeTool = "geometry";

  for (const button of document.querySelectorAll("[data-diagnostic]")) {
    button.addEventListener("click", () => {
      activeTool = button.dataset.diagnostic;
      render();
      setQuery({ diagnostic: activeTool });
    });
  }

  async function load(route) {
    const { run } = getState();
    activeTool = TOOLS[route.diagnostic] ? route.diagnostic : activeTool;
    if (!run.trajectory_metrics) {
      showEmpty("No trajectory diagnostics exist for this run. Re-run the analysis command to create them.");
      return;
    }
    if (loadedUrl !== run.trajectory_metrics) {
      const requestUrl = run.trajectory_metrics;
      loadedUrl = requestUrl;
      try {
        const nextReport = await fetchJson(requestUrl);
        if (getState().run.trajectory_metrics !== requestUrl) return;
        report = nextReport;
      } catch (error) {
        if (getState().run.trajectory_metrics !== requestUrl) return;
        showEmpty(`Could not load trajectory diagnostics: ${error.message}`);
        return;
      }
    }
    $("diagnostics-empty").hidden = true;
    $("diagnostics-workspace").hidden = false;
    render();
  }

  function render() {
    if (!report) return;
    for (const button of document.querySelectorAll("[data-diagnostic]")) {
      button.setAttribute("aria-selected", String(button.dataset.diagnostic === activeTool));
    }
    const tool = TOOLS[activeTool];
    $("diagnostic-tool-title").textContent = tool.title;
    $("diagnostic-tool-description").textContent = tool.description;
    $("diagnostic-layer").textContent = `layer ${report.layer}`;
    $("diagnostics-sampling").textContent =
      `${formatNumber(report.sampling.trajectories)} trajectories · ${formatNumber(report.sampling.pair_count)} pairs`;
    renderSummary();
    renderTool();
  }

  function renderSummary() {
    const summary = report.summary;
    const values = [
      ["Paths", formatNumber(summary.trajectories)],
      ["Net / path", formatNumber(summary.median_net_path_ratio, 3)],
      ["Curvature", formatNumber(summary.median_curvature, 3)],
      ["Median CKA", formatNumber(summary.median_cka, 3)],
    ];
    $("diagnostic-summary").innerHTML = values
      .map(([label, value]) => `<span><small>${label}</small><strong>${value}</strong></span>`)
      .join("");
  }

  function renderTool() {
    const renderers = { geometry, alignment, compression, basins, failures };
    renderers[activeTool]();
  }

  function geometry() {
    warning("");
    const rows = report.geometry;
    plot([{
      type: "scatter",
      mode: "markers",
      x: rows.map(row => row.net_path_ratio),
      y: rows.map(row => row.mean_curvature),
      text: rows.map(row => row.trajectory_id),
      customdata: rows.map(row => [row.path_length, row.effective_width, row.is_correct]),
      marker: {
        size: 8,
        color: rows.map(row => outcomeColor(row.is_correct)),
        opacity: 0.78,
        line: { width: 0 },
      },
      hovertemplate: "%{text}<br>net/path %{x:.3f}<br>curvature %{y:.3f}<br>path length %{customdata[0]:.1f}<br>effective width %{customdata[1]:.3f}<extra></extra>",
    }], "Net distance / path length", "Mean directional curvature");
    detail([
      ["Net / path", "1 is a straight route; values near 0 indicate substantial movement with little net displacement."],
      ["Curvature", "0 means consecutive changes point the same way; 1 is orthogonal; 2 reverses direction."],
      ["Effective width", "Fraction of transitions carrying the path movement. Near 1 means change is distributed, not a single spike."],
    ]);
  }

  function alignment() {
    warning("");
    const metrics = report.alignment.metrics;
    const names = ["cosine_path_similarity", "cka", "rsa"];
    plot([
      {
        type: "bar",
        name: "Same outcome",
        x: names.map(metricLabel),
        y: names.map(name => metrics[name].same_outcome),
        marker: { color: "#63c59d" },
      },
      {
        type: "bar",
        name: "Different outcome",
        x: names.map(metricLabel),
        y: names.map(name => metrics[name].different_outcome),
        marker: { color: "#e27c6f" },
      },
    ], "", "Median similarity", { barmode: "group" });
    detail([
      ["CKA", "Centered Kernel Alignment compares the relational structure of two paths; 1 means equivalent geometry."],
      ["RSA", "Representational Similarity Analysis correlates all within-path pairwise distances."],
      ["Cosine path", "Compares the direction of successive latent changes after aligning progress."],
    ]);
  }

  function compression() {
    warning("");
    const rows = report.compression;
    plot([
      {
        type: "scatter",
        mode: "lines+markers",
        name: "Variance retained",
        x: rows.map(row => row.dimensions),
        y: rows.map(row => row.explained_variance),
        line: { color: "#67a6e8", width: 2 },
      },
      {
        type: "scatter",
        mode: "lines+markers",
        name: "Reconstruction error",
        x: rows.map(row => row.dimensions),
        y: rows.map(row => row.normalized_reconstruction_error),
        line: { color: "#d6a75d", width: 2 },
      },
    ], "PCA dimensions", "Fraction");
    detail([
      ["Variance retained", "Fraction of activation variance represented by the bottleneck."],
      ["Reconstruction error", "Mean squared reconstruction error divided by original activation variance."],
    ]);
  }

  function basins() {
    const data = report.basins;
    const projection = data.projection ?? {};
    warning(projection.warning || "");
    plot([{
      type: "bar",
      x: data.sizes.map((_, index) => `Basin ${index}`),
      y: data.sizes,
      marker: { color: "#67a6e8" },
      hovertemplate: "%{x}<br>%{y} trajectories<extra></extra>",
    }], "", "Trajectories");
    detail([
      ["Silhouette", `${formatNumber(data.silhouette, 3)}. Near 1 means clean separation; near 0 means overlapping clusters.`],
      ["Branch entropy", `${formatNumber(data.entropy_bits, 3)} bits. Higher values mean trajectories are spread more evenly across endpoints.`],
      ["Projection trust", `${formatNumber(projection.trustworthiness, 3)} on a 0–1 neighborhood-preservation score for the 3D endpoint view.`],
    ]);
  }

  function failures() {
    const rows = report.failures ?? [];
    warning(rows.length ? "" : "This analysis needs both correct and incorrect trajectories.");
    plot([{
      type: "scatter",
      mode: "markers",
      x: rows.map(row => row.first_divergence_fraction),
      y: rows.map(row => row.endpoint_distance),
      text: rows.map(row => row.trajectory_id),
      customdata: rows.map(row => row.nearest_correct),
      marker: { size: 9, color: "#e27c6f", opacity: 0.8 },
      hovertemplate: "%{text}<br>divergence %{x:.2f}<br>endpoint distance %{y:.2f}<br>reference %{customdata}<extra></extra>",
    }], "Fraction of aligned path", "Endpoint distance");
    detail([
      ["Reference", "The closest correct endpoint for the same question when one is available."],
      ["First divergence", "First of two consecutive aligned points above the path's upper-quartile distance."],
      ["−1", "No sustained divergence point was identified under this conservative rule."],
    ]);
  }

  function plot(traces, xTitle, yTitle, extra = {}) {
    window.Plotly.react("diagnostic-plot", traces, {
      ...plotLayout(),
      ...extra,
      xaxis: { ...axis(), title: xTitle },
      yaxis: { ...axis(), title: yTitle },
    }, {
      displaylogo: false,
      responsive: true,
      scrollZoom: true,
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
    });
  }

  function detail(entries) {
    $("diagnostic-detail").innerHTML = entries
      .map(([term, explanation]) => `<div><strong>${escapeHtml(term)}</strong><p>${escapeHtml(explanation)}</p></div>`)
      .join("");
  }

  function warning(message) {
    $("diagnostic-warning").hidden = !message;
    $("diagnostic-warning").textContent = message;
  }

  function showEmpty(message) {
    report = null;
    $("diagnostics-workspace").hidden = true;
    $("diagnostics-empty").hidden = false;
    $("diagnostics-empty").textContent = message;
  }

  return { load };
}

function plotLayout() {
  return {
    paper_bgcolor: "#111417",
    plot_bgcolor: "#111417",
    font: { color: "#c7cdd2", family: "Inter, ui-sans-serif, system-ui" },
    margin: { l: 58, r: 18, t: 24, b: 50 },
    hoverlabel: { bgcolor: "#20262b", bordercolor: "#3d474f", font: { color: "#f4f5f6" } },
    legend: { orientation: "h", y: 1.12, x: 0 },
  };
}

function axis() {
  return {
    gridcolor: "#252c31",
    zerolinecolor: "#343d44",
    linecolor: "#343d44",
    tickfont: { color: "#9ca5ac" },
  };
}

function outcomeColor(value) {
  if (value === true) return "#63c59d";
  if (value === false) return "#e27c6f";
  return "#8f9aa2";
}

function metricLabel(name) {
  return { cosine_path_similarity: "Cosine path", cka: "CKA", rsa: "RSA" }[name] ?? name;
}
