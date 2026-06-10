export const palette = {
  ink: "#101114",
  muted: "#70747c",
  paper: "#fbfaf7",
  panel: "#ffffff",
  grid: "#ded8cc",
  blue: "#2f62ff",
  purple: "#6727d8",
  magenta: "#c73795",
  orange: "#f18f24",
  yellow: "#f7dc4f",
  green: "#00a854",
  red: "#e5484d",
};

export function drawTrajectory(canvas, data, view = {}) {
  const ctx = setup(canvas);
  const points = data.points || [];
  if (!points.length) return empty(ctx, "No trajectory points. Run analysis first.");

  const camera = {
    pitch: view.pitch ?? -0.55,
    yaw: view.yaw ?? 0.72,
    zoom: view.zoom ?? 1,
  };
  const basis = makeBasis(points, canvas, camera);
  drawGrid3D(ctx, basis);
  drawAxes3D(ctx, basis, data.coordinate_names || ["PC1", "PC2", "PC3"]);

  const projected = points.map(point => ({ ...point, ...basis.project(point) }));
  const groups = groupBy(projected, p => `${p.sample_id}:${p.seed}:${p.temperature}`);
  for (const group of groups.values()) drawTrajectoryPath(ctx, group);
  [...projected].sort((a, b) => a.depth - b.depth).forEach(point => marker(ctx, point));

  const title = `${String(data.method_actual || data.method_requested).toUpperCase()} 3D · layer ${data.layer} · interval ${data.interval}`;
  label(ctx, title, 24, 32, "title");
  if ((data.explained_variance_ratio || []).length) {
    const pct = data.explained_variance_ratio.slice(0, 3).map(v => `${(100 * v).toFixed(1)}%`).join(" / ");
    label(ctx, `variance explained: ${pct}`, 24, 56, "small");
  }
  if (data.warning) label(ctx, data.warning, 24, 80, "warn");
  legend(ctx, canvas);
}

export function drawComponents(canvas, data) {
  const ctx = setup(canvas);
  const values = data.components || [];
  if (!values.length) return empty(ctx, "No PCA components. Run analysis first.");

  const pad = scaledPad(canvas);
  const w = canvas.width - pad.left - pad.right;
  const h = canvas.height - pad.top - pad.bottom;
  const maxAmp = Math.max(...values.map(v => Number(v.amplitude) || 0), 1);
  drawFrame(ctx, canvas, pad);
  values.forEach((value, idx) => {
    const x = pad.left + idx * (w / values.length);
    const barW = Math.max(3, w / values.length - 5);
    const barH = h * ((Number(value.amplitude) || 0) / maxAmp);
    ctx.fillStyle = ramp(idx / Math.max(1, values.length - 1));
    ctx.fillRect(x, pad.top + h - barH, barW, barH);
  });
  label(ctx, `PCA component amplitudes · layer ${data.layer}`, 24, 32, "title");
  label(ctx, `${data.token_vectors_used ?? data.token_vectors ?? 0} / ${data.token_vectors_total ?? data.token_vectors ?? 0} token vectors`, 24, 56, "small");
}

export function drawNorms(canvas, rows) {
  const ctx = setup(canvas);
  const values = rows
    .map(row => ({ layer: Number(row.layer), y: Number(row.l2_norm) }))
    .filter(value => Number.isFinite(value.layer) && Number.isFinite(value.y))
    .sort((a, b) => a.layer - b.layer);
  if (!values.length) return empty(ctx, "No activation norm rows. Run analysis first.");

  const pad = scaledPad(canvas);
  const minLayer = Math.min(...values.map(v => v.layer));
  const maxLayer = Math.max(...values.map(v => v.layer));
  const maxY = Math.max(...values.map(v => v.y), 1);
  drawFrame(ctx, canvas, pad);
  ctx.beginPath();
  values.forEach((value, idx) => {
    const x = pad.left + ((value.layer - minLayer) / Math.max(1, maxLayer - minLayer)) * (canvas.width - pad.left - pad.right);
    const y = canvas.height - pad.bottom - (value.y / maxY) * (canvas.height - pad.top - pad.bottom);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = palette.blue;
  ctx.lineWidth = 2 * pixelRatio();
  ctx.stroke();
  values.forEach(value => {
    const x = pad.left + ((value.layer - minLayer) / Math.max(1, maxLayer - minLayer)) * (canvas.width - pad.left - pad.right);
    const y = canvas.height - pad.bottom - (value.y / maxY) * (canvas.height - pad.top - pad.bottom);
    dot(ctx, x, y, 4 * pixelRatio(), palette.orange);
  });
  label(ctx, "Mean activation L2 norm by layer", 24, 32, "title");
}

function setup(canvas) {
  const rect = canvas.getBoundingClientRect();
  const scale = pixelRatio();
  canvas.width = Math.max(640, Math.floor(rect.width * scale));
  canvas.height = Math.max(440, Math.floor(rect.height * scale));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = palette.panel;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  return ctx;
}

function makeBasis(points, canvas, camera) {
  const ranges = ["x", "y", "z"].map(axis => range(points.map(point => Number(point[axis]) || 0)));
  const scale = Math.min(canvas.width, canvas.height) * 0.34 * camera.zoom;
  const center = { x: canvas.width * 0.53, y: canvas.height * 0.54 };
  const rotate = ({ x, y, z }) => {
    const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw);
    const cx = Math.cos(camera.pitch), sx = Math.sin(camera.pitch);
    const rx = x * cy + z * sy;
    const rz = -x * sy + z * cy;
    const ry = y * cx - rz * sx;
    const dz = y * sx + rz * cx;
    return { rx, ry, dz };
  };
  return {
    project(point) {
      const normalized = {
        x: norm(Number(point.x) || 0, ranges[0]),
        y: norm(Number(point.y) || 0, ranges[1]),
        z: norm(Number(point.z) || 0, ranges[2]),
      };
      const rotated = rotate(normalized);
      return {
        sx: center.x + rotated.rx * scale,
        sy: center.y - rotated.ry * scale,
        depth: rotated.dz,
      };
    },
    projectRaw(raw) {
      const rotated = rotate(raw);
      return {
        sx: center.x + rotated.rx * scale,
        sy: center.y - rotated.ry * scale,
        depth: rotated.dz,
      };
    },
  };
}

function drawGrid3D(ctx, basis) {
  ctx.strokeStyle = palette.grid;
  ctx.lineWidth = 1 * pixelRatio();
  ctx.globalAlpha = 0.7;
  for (let i = -2; i <= 2; i++) {
    const a = basis.projectRaw({ x: -1, y: i / 2, z: 0 });
    const b = basis.projectRaw({ x: 1, y: i / 2, z: 0 });
    const c = basis.projectRaw({ x: i / 2, y: -1, z: 0 });
    const d = basis.projectRaw({ x: i / 2, y: 1, z: 0 });
    line(ctx, a, b);
    line(ctx, c, d);
  }
  ctx.globalAlpha = 1;
}

function drawAxes3D(ctx, basis, names) {
  const origin = basis.projectRaw({ x: 0, y: 0, z: 0 });
  const axes = [
    { end: basis.projectRaw({ x: 1.18, y: 0, z: 0 }), name: names[0] || "x" },
    { end: basis.projectRaw({ x: 0, y: 1.18, z: 0 }), name: names[1] || "y" },
    { end: basis.projectRaw({ x: 0, y: 0, z: 1.18 }), name: names[2] || "z" },
  ];
  ctx.strokeStyle = palette.ink;
  ctx.fillStyle = palette.ink;
  ctx.lineWidth = 1.4 * pixelRatio();
  for (const axis of axes) {
    arrow(ctx, origin, axis.end);
    label(ctx, axis.name, axis.end.sx / pixelRatio() + 8, axis.end.sy / pixelRatio(), "axis");
  }
}

function drawFrame(ctx, canvas, pad) {
  ctx.strokeStyle = palette.grid;
  ctx.lineWidth = 1 * pixelRatio();
  for (let i = 1; i < 8; i++) {
    const x = pad.left + i * ((canvas.width - pad.left - pad.right) / 8);
    const y = pad.top + i * ((canvas.height - pad.top - pad.bottom) / 8);
    line(ctx, { sx: x, sy: pad.top }, { sx: x, sy: canvas.height - pad.bottom });
    line(ctx, { sx: pad.left, sy: y }, { sx: canvas.width - pad.right, sy: y });
  }
  ctx.strokeStyle = palette.ink;
  ctx.lineWidth = 1.3 * pixelRatio();
  ctx.strokeRect(pad.left, pad.top, canvas.width - pad.left - pad.right, canvas.height - pad.top - pad.bottom);
}

function drawTrajectoryPath(ctx, group) {
  ctx.beginPath();
  group.forEach((point, idx) => {
    if (idx === 0) ctx.moveTo(point.sx, point.sy);
    else ctx.lineTo(point.sx, point.sy);
  });
  ctx.lineWidth = 1.7 * pixelRatio();
  ctx.strokeStyle = group[0].success ? palette.green : palette.red;
  ctx.globalAlpha = 0.7;
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function marker(ctx, point) {
  const depthSize = 1 + 0.18 * point.depth;
  const radius = Math.max(2.8, 4.2 * depthSize) * pixelRatio();
  if (point.is_start) triangle(ctx, point.sx, point.sy, radius * 1.5, palette.yellow);
  else if (point.is_end) diamond(ctx, point.sx, point.sy, radius * 1.55, palette.orange);
  else dot(ctx, point.sx, point.sy, radius, point.success ? palette.green : palette.red);
}

function dot(ctx, x, y, r, fill) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 1.1 * pixelRatio();
  ctx.stroke();
}

function triangle(ctx, x, y, r, fill) {
  ctx.beginPath();
  ctx.moveTo(x, y - r);
  ctx.lineTo(x + r, y + r);
  ctx.lineTo(x - r, y + r);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.strokeStyle = palette.ink;
  ctx.stroke();
}

function diamond(ctx, x, y, r, fill) {
  ctx.beginPath();
  ctx.moveTo(x, y - r);
  ctx.lineTo(x + r, y);
  ctx.lineTo(x, y + r);
  ctx.lineTo(x - r, y);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.strokeStyle = palette.ink;
  ctx.stroke();
}

function arrow(ctx, a, b) {
  line(ctx, a, b);
  const angle = Math.atan2(b.sy - a.sy, b.sx - a.sx);
  const size = 8 * pixelRatio();
  ctx.beginPath();
  ctx.moveTo(b.sx, b.sy);
  ctx.lineTo(b.sx - size * Math.cos(angle - 0.45), b.sy - size * Math.sin(angle - 0.45));
  ctx.lineTo(b.sx - size * Math.cos(angle + 0.45), b.sy - size * Math.sin(angle + 0.45));
  ctx.closePath();
  ctx.fillStyle = palette.ink;
  ctx.fill();
}

function line(ctx, a, b) {
  ctx.beginPath();
  ctx.moveTo(a.sx, a.sy);
  ctx.lineTo(b.sx, b.sy);
  ctx.stroke();
}

function label(ctx, text, x, y, kind = "body") {
  const scale = pixelRatio();
  const fonts = {
    title: `${14 * scale}px Inter, system-ui, sans-serif`,
    small: `${12 * scale}px Inter, system-ui, sans-serif`,
    warn: `${12 * scale}px Inter, system-ui, sans-serif`,
    axis: `${12 * scale}px ui-monospace, SFMono-Regular, Menlo, monospace`,
    body: `${13 * scale}px Inter, system-ui, sans-serif`,
  };
  ctx.font = fonts[kind] || fonts.body;
  ctx.fillStyle = kind === "warn" ? palette.orange : kind === "small" ? palette.muted : palette.ink;
  ctx.fillText(text, x * scale, y * scale);
}

function legend(ctx, canvas) {
  const scale = pixelRatio();
  const y = canvas.height - 28 * scale;
  const items = [
    [palette.green, "correct path"],
    [palette.red, "wrong path"],
    [palette.yellow, "start"],
    [palette.orange, "end"],
  ];
  let x = 24 * scale;
  for (const [color, text] of items) {
    dot(ctx, x, y - 4 * scale, 4 * scale, color);
    ctx.fillStyle = palette.muted;
    ctx.font = `${12 * scale}px Inter, system-ui, sans-serif`;
    ctx.fillText(text, x + 11 * scale, y);
    x += (ctx.measureText(text).width + 34 * scale);
  }
}

function empty(ctx, text) {
  label(ctx, text, 24, 42, "body");
}

function scaledPad(canvas) {
  const scale = pixelRatio();
  return { left: 64 * scale, right: 32 * scale, top: 72 * scale, bottom: 60 * scale };
}

function range(values) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  return { min, max: max === min ? min + 1 : max };
}

function norm(value, r) {
  return ((value - r.min) / (r.max - r.min) - 0.5) * 2;
}

function groupBy(items, keyFn) {
  const groups = new Map();
  for (const item of items) {
    const key = keyFn(item);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(item);
  }
  return groups;
}

function ramp(t) {
  const stops = [palette.purple, palette.magenta, palette.orange, palette.yellow];
  const idx = Math.min(stops.length - 1, Math.floor(t * stops.length));
  return stops[idx];
}

function pixelRatio() {
  return window.devicePixelRatio || 1;
}
