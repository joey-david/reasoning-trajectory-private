export const $ = id => document.getElementById(id);

export async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}: ${url}`);
  return response.json();
}

export async function fetchJsonl(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}: ${url}`);
  const text = await response.text();
  return text.split(/\n+/).filter(Boolean).map(line => JSON.parse(line));
}

export function setOptions(id, values, first = null, selected = null) {
  const select = $(id);
  select.replaceChildren();
  if (first) select.add(new Option(first.label, first.value));
  for (const item of values) {
    const [value, label] = Array.isArray(item) ? item : [item, item];
    select.add(new Option(String(label), String(value)));
  }
  if (selected !== null && [...select.options].some(option => option.value === String(selected))) {
    select.value = String(selected);
  }
}

export function escapeHtml(value) {
  return String(value ?? "").replace(
    /[&<>"']/g,
    char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char],
  );
}

export function formatNumber(value, digits = 0) {
  if (!Number.isFinite(Number(value))) return "—";
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(Number(value));
}

export function formatPercent(value, digits = 0) {
  if (!Number.isFinite(Number(value))) return "—";
  return new Intl.NumberFormat(undefined, {
    style: "percent",
    maximumFractionDigits: digits,
  }).format(Number(value));
}

export function median(values) {
  const sorted = values.filter(Number.isFinite).slice().sort((a, b) => a - b);
  if (!sorted.length) return NaN;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function questionText(prompt) {
  const raw = String(prompt ?? "");
  const tagged = raw.match(/<\|im_start\|>user\s*([\s\S]*?)<\|im_end\|>/);
  const userText = (tagged?.[1] ?? raw).trim();
  const blocks = userText.split(/\n{2,}/).map(block => block.trim()).filter(Boolean);
  return blocks.at(-1) ?? userText;
}

export function outcome(row) {
  if (row.is_correct === true) return "correct";
  if (row.is_correct === false) return "incorrect";
  return "unknown";
}

export function isFormTarget(target) {
  return target instanceof HTMLInputElement
    || target instanceof HTMLSelectElement
    || target instanceof HTMLTextAreaElement
    || target?.isContentEditable;
}

export function debounce(fn, delay = 120) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}
