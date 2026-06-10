export const $ = id => document.getElementById(id);

export function option(value, label, selected = false) {
  return `<option value="${escapeHTML(String(value))}" ${selected ? "selected" : ""}>${escapeHTML(String(label))}</option>`;
}

export function table(rows) {
  if (!rows.length) return `<div class="empty">No rows to display.</div>`;
  const cols = Object.keys(rows[0]);
  return `
    <div class="tableWrap">
      <table>
        <thead><tr>${cols.map(col => `<th>${escapeHTML(col)}</th>`).join("")}</tr></thead>
        <tbody>${rows.map(row => `<tr>${cols.map(col => cell(col, row[col])).join("")}</tr>`).join("")}</tbody>
      </table>
    </div>`;
}

export function cell(col, value) {
  const isSuccess = String(value) === "True" || String(value) === "true";
  const klass = col === "success" ? (isSuccess ? "success" : "failure") : "";
  return `<td class="${klass}">${escapeHTML(String(value ?? ""))}</td>`;
}

export function csvNumbers(text, cast = Number) {
  return text.split(",").map(x => x.trim()).filter(Boolean).map(cast);
}

export function escapeHTML(value) {
  return String(value).replace(/[&<>"']/g, ch => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[ch]));
}
