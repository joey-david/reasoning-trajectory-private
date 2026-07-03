import { capturePoints, captureTokens, glossary, matrixData } from "./data.js";

init();

function init() {
  renderGlossary();
  renderMatrix();
  bindGlossary();
  bindNavigation();
  bindScrubber();
  setupCanvases();
}

function renderGlossary() {
  const grid = document.querySelector("#glossary-grid");
  grid.innerHTML = Object.entries(glossary)
    .sort((a, b) => a[1].title.localeCompare(b[1].title))
    .map(([, item]) => `
      <dl class="glossary-entry">
        <dt>${item.title}</dt>
        <dd>${item.body}</dd>
      </dl>
    `).join("");
}

function renderMatrix() {
  const target = document.querySelector("#objective-matrix");
  target.innerHTML = `<span class="heatmap-label"></span>`
    + matrixData.columns.map(column => `<span class="heatmap-label column">${column}</span>`).join("")
    + matrixData.rows.map((row, rowIndex) => {
      const label = `<span class="heatmap-label">${row}</span>`;
      const cells = matrixData.values[rowIndex].map((value, columnIndex) => {
        const color = utilityColor(value);
        return `<button class="heatmap-cell" type="button"
          style="background:${color}"
          data-row="${rowIndex}" data-column="${columnIndex}"
          aria-label="${row} on ${matrixData.columns[columnIndex]} objective: ${value.toFixed(3)}">
          ${value.toFixed(2)}
        </button>`;
      }).join("");
      return label + cells;
    }).join("");

  target.addEventListener("click", event => {
    const cell = event.target.closest(".heatmap-cell");
    if (!cell) return;
    target.querySelectorAll(".heatmap-cell").forEach(item => item.classList.toggle("active", item === cell));
    const row = Number(cell.dataset.row);
    const column = Number(cell.dataset.column);
    const value = matrixData.values[row][column];
    const same = row === column;
    document.querySelector("#matrix-readout").innerHTML = same
      ? `<strong>${matrixData.rows[row]} → ${matrixData.columns[column]}: ${value.toFixed(3)}.</strong> This is the objective that defined the oracle, so utility is 1 by construction.`
      : `<strong>${matrixData.rows[row]} → ${matrixData.columns[column]}: ${value.toFixed(3)}.</strong> ${crossObjectiveMeaning(value)}`;
  });
}

function utilityColor(value) {
  const bounded = Math.max(0, Math.min(1, value));
  const lightness = 96 - bounded * 52;
  return `hsl(151 38% ${lightness}%)`;
}

function crossObjectiveMeaning(value) {
  if (value < 0) return "It performs slightly worse than random expectation on this different objective.";
  if (value < 0.08) return "It captures almost none of the other objective’s available signal.";
  if (value < 0.25) return "There is limited overlap, but most objective-specific signal is missed.";
  return "This is the clearest cross-objective overlap, but it remains far below the target-specific optimum.";
}

function bindGlossary() {
  const drawer = document.querySelector("#glossary-drawer");
  const backdrop = document.querySelector("#drawer-backdrop");
  const button = document.querySelector("#glossary-button");
  const close = () => {
    drawer.hidden = true;
    backdrop.hidden = true;
    button.setAttribute("aria-expanded", "false");
  };
  const open = key => {
    const item = glossary[key] ?? {
      title: "Glossary",
      body: "Select a dotted term in the report to see its definition."
    };
    document.querySelector("#drawer-title").textContent = item.title;
    document.querySelector("#drawer-content").innerHTML = `<h3>${item.title}</h3><p>${item.body}</p>`;
    drawer.hidden = false;
    backdrop.hidden = false;
    button.setAttribute("aria-expanded", "true");
    document.querySelector("#glossary-close").focus();
  };

  document.addEventListener("click", event => {
    const term = event.target.closest("[data-term]");
    if (term) open(term.dataset.term);
  });
  button.addEventListener("click", () => open(null));
  document.querySelector("#glossary-close").addEventListener("click", close);
  drawer.querySelector('a[href="#glossary"]').addEventListener("click", close);
  backdrop.addEventListener("click", close);
  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && !drawer.hidden) close();
  });
}

function bindNavigation() {
  const links = [...document.querySelectorAll(".chapter-nav a")];
  const sections = [...document.querySelectorAll("[data-observe]")];
  const observer = new IntersectionObserver(entries => {
    const visible = entries
      .filter(entry => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    links.forEach(link => {
      link.setAttribute("aria-current", String(link.dataset.chapter === visible.target.dataset.observe));
    });
  }, { rootMargin: "-30% 0px -55% 0px", threshold: [0, 0.1, 0.4] });
  sections.forEach(section => observer.observe(section));
}

function bindScrubber() {
  const slider = document.querySelector("#token-scrubber");
  const render = () => {
    const index = Number(slider.value);
    document.querySelector("#token-output").textContent = `${index + 1} / ${captureTokens.length}`;
    document.querySelector("#active-token").textContent = captureTokens[index];
    const [x, y] = capturePoints[index];
    document.querySelector("#active-vector").textContent = `h${toSubscript(index + 1)} = [${(x - .5).toFixed(2)}, ${(y - .5).toFixed(2)}, …]`;
    drawCapture(index);
  };
  slider.addEventListener("input", render);
  render();
}

function toSubscript(number) {
  return String(number).replace(/\d/g, digit => "₀₁₂₃₄₅₆₇₈₉"[Number(digit)]);
}

function setupCanvases() {
  const redraw = debounce(() => {
    drawHero();
    drawCapture(Number(document.querySelector("#token-scrubber").value));
    drawWave();
    drawPatchChart();
    drawPredictionChart();
  }, 80);
  window.addEventListener("resize", redraw);
  redraw();
}

function canvasContext(selector) {
  const canvas = document.querySelector(selector);
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { canvas, context, width: rect.width, height: rect.height };
}

function drawHero() {
  const { context: ctx, width, height } = canvasContext("#hero-canvas");
  ctx.clearRect(0, 0, width, height);
  const paths = [
    { color: "#72b89a", offset: 0, alpha: .55 },
    { color: "#e0b95a", offset: 80, alpha: .34 },
    { color: "#6d93bd", offset: -65, alpha: .3 }
  ];
  for (const path of paths) {
    ctx.beginPath();
    for (let i = 0; i < 72; i++) {
      const t = i / 71;
      const x = width * (.34 + .7 * t);
      const y = height * (.52 + .18 * Math.sin(t * 9 + path.offset / 60))
        + path.offset + 46 * Math.sin(t * 23);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = path.color;
    ctx.globalAlpha = path.alpha;
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  for (let i = 0; i < 42; i++) {
    const t = i / 41;
    const x = width * (.34 + .7 * t);
    const y = height * (.52 + .18 * Math.sin(t * 9)) + 46 * Math.sin(t * 23);
    ctx.beginPath();
    ctx.arc(x, y, i % 7 === 0 ? 4 : 2, 0, Math.PI * 2);
    ctx.fillStyle = i % 7 === 0 ? "#e0b95a" : "#8fd0b3";
    ctx.fill();
  }
}

function drawCapture(activeIndex) {
  const { context: ctx, width, height } = canvasContext("#capture-canvas");
  ctx.clearRect(0, 0, width, height);
  const pad = { left: 44, right: 34, top: 38, bottom: 38 };
  drawGrid(ctx, width, height, pad, "#d3ddd7");
  const points = capturePoints.map(([x, y]) => [
    pad.left + x * (width - pad.left - pad.right),
    pad.top + y * (height - pad.top - pad.bottom)
  ]);
  ctx.beginPath();
  points.forEach(([x, y], index) => index ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
  ctx.strokeStyle = "#7d9589";
  ctx.lineWidth = 2;
  ctx.stroke();
  points.forEach(([x, y], index) => {
    ctx.beginPath();
    ctx.arc(x, y, index === activeIndex ? 7 : 3, 0, Math.PI * 2);
    ctx.fillStyle = index === activeIndex ? "#c4513f" : index < activeIndex ? "#16634c" : "#9aa8a0";
    ctx.fill();
  });
  const [x, y] = points[activeIndex];
  ctx.fillStyle = "#18211d";
  ctx.font = "700 12px Inter, sans-serif";
  ctx.fillText(captureTokens[activeIndex], Math.min(x + 11, width - 90), Math.max(y - 10, 18));
}

function drawGrid(ctx, width, height, pad, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  for (let index = 0; index < 5; index++) {
    const x = pad.left + index * (width - pad.left - pad.right) / 4;
    const y = pad.top + index * (height - pad.top - pad.bottom) / 4;
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, height - pad.bottom);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
  }
}

function drawWave() {
  const { context: ctx, width, height } = canvasContext("#wave-canvas");
  ctx.clearRect(0, 0, width, height);
  const pad = 42;
  const split = height * .5;
  ctx.fillStyle = "#f3ded8";
  ctx.fillRect(width * .25, 28, width * .5, height - 56);
  ctx.fillStyle = "#746f69";
  ctx.font = "700 10px Inter, sans-serif";
  ctx.fillText("EXPECTED: ONE SHARP EVENT", pad, 26);
  ctx.fillText("OBSERVED: DISTRIBUTED CHANGE", pad, split + 26);
  ctx.strokeStyle = "#d4d4ca";
  ctx.beginPath();
  ctx.moveTo(pad, split);
  ctx.lineTo(width - pad, split);
  ctx.stroke();
  drawSeries(ctx, width, pad, split * .45, t => .05 + .92 * Math.exp(-Math.pow((t - .5) / .055, 2)), "#c4513f");
  drawSeries(ctx, width, pad, split + (height - split) * .52, t =>
    .22 + .17 * Math.sin(t * 18) + .1 * Math.sin(t * 47) + .28 * Math.exp(-Math.pow((t - .52) / .25, 2)), "#16634c");
  ctx.fillStyle = "#8a5f56";
  ctx.font = "600 10px Inter, sans-serif";
  ctx.fillText("symbolic update interval", width * .25 + 8, height - 15);
}

function drawSeries(ctx, width, pad, baseline, valueAt, color) {
  ctx.beginPath();
  for (let i = 0; i < 120; i++) {
    const t = i / 119;
    const x = pad + t * (width - pad * 2);
    const y = baseline - valueAt(t) * 105;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.stroke();
}

function drawPatchChart() {
  const { context: ctx, width, height } = canvasContext("#patch-chart");
  ctx.clearRect(0, 0, width, height);
  const rows = [
    ["Full vs random", 6.9, -3.6, 16.9],
    ["Full vs mismatch", 3.3, -5.0, 12.5],
    ["Subspace vs random", 5.8, -6.7, 18.9],
    ["Subspace vs mismatch", 2.8, -7.2, 13.9]
  ];
  const pad = { left: 132, right: 28, top: 42, bottom: 48 };
  const min = -10;
  const max = 20;
  const scaleX = value => pad.left + (value - min) / (max - min) * (width - pad.left - pad.right);
  ctx.strokeStyle = "#d4d4ca";
  ctx.fillStyle = "#66716b";
  ctx.font = "10px Inter, sans-serif";
  [-10, 0, 10, 20].forEach(value => {
    const x = scaleX(value);
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, height - pad.bottom);
    ctx.stroke();
    ctx.fillText(`${value > 0 ? "+" : ""}${value}`, x - 8, height - 25);
  });
  ctx.fillText("accuracy difference (percentage points)", pad.left, height - 8);
  rows.forEach(([label, estimate, low, high], index) => {
    const y = pad.top + 42 + index * (height - pad.top - pad.bottom - 42) / rows.length;
    ctx.fillStyle = "#39443e";
    ctx.font = "600 11px Inter, sans-serif";
    ctx.fillText(label, 14, y + 4);
    ctx.strokeStyle = "#c4513f";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(scaleX(low), y);
    ctx.lineTo(scaleX(high), y);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(scaleX(estimate), y, 6, 0, Math.PI * 2);
    ctx.fillStyle = "#c4513f";
    ctx.fill();
  });
}

function drawPredictionChart() {
  const { context: ctx, width, height } = canvasContext("#prediction-chart");
  ctx.clearRect(0, 0, width, height);
  const pad = { left: 62, right: 32, top: 38, bottom: 58 };
  const series = [
    { label: "Sentence mean + variance", color: "#16634c", values: [.757, .740, .762] },
    { label: "Symbolic updates", color: "#c4513f", values: [.631, .652, .700] },
    { label: "Latent waves", color: "#3168a6", values: [.639, .616, .680] }
  ];
  const xAt = index => pad.left + index * (width - pad.left - pad.right) / 2;
  const yAt = value => height - pad.bottom - (value - .5) / .35 * (height - pad.top - pad.bottom);
  ctx.font = "10px Inter, sans-serif";
  [.5, .6, .7, .8].forEach(value => {
    const y = yAt(value);
    ctx.strokeStyle = "#d4d4ca";
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.fillStyle = "#69736d";
    ctx.fillText(value.toFixed(1), 24, y + 4);
  });
  ["25%", "50%", "75%"].forEach((label, index) => {
    ctx.fillStyle = "#69736d";
    ctx.fillText(label, xAt(index) - 10, height - 29);
  });
  ctx.fillText("trace consumed", width / 2 - 35, height - 9);
  series.forEach((item, seriesIndex) => {
    ctx.beginPath();
    item.values.forEach((value, index) => {
      const x = xAt(index);
      const y = yAt(value);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 3;
    ctx.stroke();
    item.values.forEach((value, index) => {
      ctx.beginPath();
      ctx.arc(xAt(index), yAt(value), 5, 0, Math.PI * 2);
      ctx.fillStyle = item.color;
      ctx.fill();
    });
    ctx.fillStyle = item.color;
    ctx.font = "700 11px Inter, sans-serif";
    ctx.fillText(item.label, pad.left + seriesIndex * 155, 18);
  });
}

function debounce(callback, delay) {
  let timeout;
  return (...args) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => callback(...args), delay);
  };
}
