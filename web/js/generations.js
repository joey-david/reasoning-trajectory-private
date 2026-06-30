import {
  $,
  debounce,
  escapeHtml,
  formatNumber,
  outcome,
  questionText,
  setOptions,
} from "./ui.js";

const PAGE_SIZE = 12;

export function createGenerationView({ getState, setQuery, openTrajectory }) {
  let visibleCount = PAGE_SIZE;
  let filteredRows = [];
  let activationTarget = null;

  const rerender = () => {
    visibleCount = PAGE_SIZE;
    updateSeedOptions();
    render();
    syncQuery();
  };

  for (const id of [
    "generation-question",
    "generation-seed",
    "generation-outcome",
    "generation-marker",
    "generation-sort",
    "generation-entropy",
  ]) {
    $(id).addEventListener("change", rerender);
  }
  $("generation-search").addEventListener("input", debounce(rerender));
  $("generation-clear").addEventListener("click", clear);
  $("generation-more").addEventListener("click", () => {
    visibleCount += PAGE_SIZE;
    renderRows();
  });
  $("generation-list").addEventListener("click", handleListAction);

  function load(route) {
    const { rows, markers } = getState();
    const sampleIds = [...new Set(rows.map(row => row.sample_id))].sort();
    const markerNames = markers ? Object.keys(markers.selectors ?? {}) : [];
    setOptions("generation-question", sampleIds, { value: "", label: "All questions" }, route.question);
    setOptions("generation-marker", markerNames, { value: "", label: "No markers" }, route.marker);
    $("generation-search").value = route.search ?? "";
    $("generation-outcome").value = route.outcome ?? "";
    $("generation-sort").value = route.sort ?? "question";
    activationTarget = parseActivationTarget(route);

    const hasEntropy = rows.some(hasEntropyTimesteps);
    $("generation-entropy").disabled = !hasEntropy;
    $("generation-entropy").checked = hasEntropy && route.entropy === "1";
    $("entropy-status").textContent = hasEntropy
      ? "Color is normalized within each generation."
      : "This run did not store timestep entropy diagnostics.";
    $("entropy-legend").hidden = !hasEntropy || !$("generation-entropy").checked;

    updateSeedOptions(route.seed);
    visibleCount = PAGE_SIZE;
    render();
  }

  function clear() {
    $("generation-search").value = "";
    $("generation-question").value = "";
    $("generation-outcome").value = "";
    $("generation-marker").value = "";
    $("generation-sort").value = "question";
    $("generation-entropy").checked = false;
    rerender();
    $("generation-search").focus();
  }

  function updateSeedOptions(preferred = $("generation-seed").value) {
    const { rows } = getState();
    const question = $("generation-question").value;
    const seeds = [...new Set(
      rows.filter(row => !question || row.sample_id === question).map(row => row.seed),
    )].sort((a, b) => Number(a) - Number(b));
    setOptions("generation-seed", seeds, { value: "", label: "All sub-runs" }, preferred);
  }

  function render() {
    const { rows, run } = getState();
    const search = $("generation-search").value.trim().toLowerCase();
    const question = $("generation-question").value;
    const seed = $("generation-seed").value;
    const selectedOutcome = $("generation-outcome").value;

    filteredRows = rows.filter(row => {
      if (question && row.sample_id !== question) return false;
      if (seed && String(row.seed) !== seed) return false;
      if (selectedOutcome && outcome(row) !== selectedOutcome) return false;
      if (!search) return true;
      const sample = run.samples[row.sample_id] ?? {};
      return [
        row.sample_id,
        row.produced_answer,
        row.produced_text,
        questionText(sample.prompt),
      ].some(value => String(value ?? "").toLowerCase().includes(search));
    });

    const sort = $("generation-sort").value;
    filteredRows.sort((a, b) => {
      if (sort === "longest") return Number(b.reasoning_length ?? 0) - Number(a.reasoning_length ?? 0);
      if (sort === "shortest") return Number(a.reasoning_length ?? 0) - Number(b.reasoning_length ?? 0);
      return a.sample_id.localeCompare(b.sample_id) || Number(a.seed) - Number(b.seed);
    });

    $("entropy-legend").hidden = !$("generation-entropy").checked || $("generation-entropy").disabled;
    renderRows();
  }

  function renderRows() {
    const shown = filteredRows.slice(0, visibleCount);
    $("generation-count").textContent = `${formatNumber(filteredRows.length)} matching ${filteredRows.length === 1 ? "generation" : "generations"}`;
    $("generation-list").innerHTML = shown.map(rowHtml).join("");
    $("generation-empty").hidden = filteredRows.length > 0;
    $("generation-empty").textContent = filteredRows.length ? "" : "No generations match these filters.";
    $("generation-more").hidden = visibleCount >= filteredRows.length;
    if (!$("generation-more").hidden) {
      $("generation-more").textContent = `Show ${Math.min(PAGE_SIZE, filteredRows.length - visibleCount)} more`;
    }
    requestAnimationFrame(focusActivationTarget);
  }

  function rowHtml(row) {
    const { run } = getState();
    const sample = run.samples[row.sample_id] ?? {};
    const status = outcome(row);
    const markerHtml = markerStrip(row);
    return `<article class="generation-card" data-sample-id="${escapeHtml(row.sample_id)}" data-seed="${escapeHtml(row.seed)}">
      <header class="generation-card-header">
        <div class="generation-identity">
          <strong title="${escapeHtml(questionText(sample.prompt))}">${escapeHtml(row.sample_id)}</strong>
          <div class="generation-meta">
            <span class="status-pill ${status}">${status === "unknown" ? "not scored" : status}</span>
            <span>sub-run ${escapeHtml(row.seed)}</span>
            <span>answer <strong>${escapeHtml(row.produced_answer ?? "—")}</strong></span>
            <span>${formatNumber(row.reasoning_length)} reasoning tokens</span>
          </div>
        </div>
        <div class="generation-actions">
          <button class="text-button" type="button" data-action="toggle-output">Expand output</button>
          <button class="secondary-button" type="button" data-action="open-trajectory">View latent</button>
        </div>
      </header>
      ${markerHtml}
      <div class="generation-output">${formatOutput(row)}</div>
      <div class="generation-details">
        <details>
          <summary>Prompt</summary>
          <pre>${escapeHtml(sample.prompt ?? "")}</pre>
        </details>
        <details>
          <summary>Reference answer</summary>
          <pre>${escapeHtml(sample.gold_answer ?? "")}</pre>
        </details>
      </div>
    </article>`;
  }

  function markerStrip(row) {
    const { markers } = getState();
    const markerName = $("generation-marker").value;
    if (!markerName || !markers) return "";
    const record = markers.records?.find(
      item => item.sample_id === row.sample_id && String(item.seed) === String(row.seed),
    );
    const values = record?.selectors?.[markerName] ?? [];
    if (!values.length) return `<div class="marker-strip">No ${escapeHtml(markerName)} markers</div>`;
    const chips = values.slice(0, 16).map(value => `<span class="marker-chip">${escapeHtml(value)}</span>`).join("");
    const remainder = values.length > 16 ? `<span>+${values.length - 16} more</span>` : "";
    return `<div class="marker-strip"><strong>${values.length} markers</strong>${chips}${remainder}</div>`;
  }

  function formatOutput(row) {
    if (matchesActivationTarget(row)) {
      const text = row.produced_text ?? "";
      const { charStart, charEnd, tokenIdx } = activationTarget;
      const tokenText = charEnd > charStart
        ? escapeHtml(text.slice(charStart, charEnd))
        : `⟨token ${escapeHtml(tokenIdx)}⟩`;
      return `${escapeHtml(text.slice(0, charStart))}<mark class="activation-token-highlight" data-testid="activation-token-highlight" tabindex="-1" title="Latent activation at token ${escapeHtml(tokenIdx)}">${tokenText}</mark>${escapeHtml(text.slice(charEnd))}`;
    }
    if (!$("generation-entropy").checked || !hasEntropyTimesteps(row)) {
      return escapeHtml(row.produced_text ?? "");
    }
    const values = row.timesteps.map(entropyValue).filter(Number.isFinite);
    const low = Math.min(...values);
    const high = Math.max(...values);
    const span = Math.max(high - low, 1e-9);
    return row.timesteps.map(timestep => {
      const entropy = entropyValue(timestep);
      if (!Number.isFinite(entropy)) return escapeHtml(timestep.token_str ?? "");
      const level = (entropy - low) / span;
      const hue = 152 - level * 145;
      const lightness = 92 - level * 48;
      const color = level > 0.72 ? "white" : "inherit";
      return `<span class="token-entropy" title="Entropy ${entropy.toFixed(3)}" style="background:hsl(${hue} 74% ${lightness}%);color:${color}">${escapeHtml(timestep.token_str ?? "")}</span>`;
    }).join("");
  }

  function handleListAction(event) {
    const button = event.target.closest("button[data-action]");
    if (!button) return;
    const card = button.closest(".generation-card");
    if (button.dataset.action === "toggle-output") {
      const output = card.querySelector(".generation-output");
      output.classList.toggle("expanded");
      button.textContent = output.classList.contains("expanded") ? "Collapse output" : "Expand output";
      return;
    }
    openTrajectory(card.dataset.sampleId, card.dataset.seed);
  }

  function syncQuery() {
    setQuery({
      question: $("generation-question").value,
      seed: $("generation-seed").value,
      search: $("generation-search").value,
      outcome: $("generation-outcome").value,
      marker: $("generation-marker").value,
      sort: $("generation-sort").value === "question" ? "" : $("generation-sort").value,
      entropy: $("generation-entropy").checked ? "1" : "",
    });
  }

  function matchesActivationTarget(row) {
    if (!activationTarget) return false;
    const textLength = String(row.produced_text ?? "").length;
    return row.sample_id === activationTarget.sampleId
      && String(row.seed) === activationTarget.seed
      && activationTarget.charStart >= 0
      && activationTarget.charEnd >= activationTarget.charStart
      && activationTarget.charEnd <= textLength;
  }

  function focusActivationTarget() {
    const target = document.querySelector('[data-testid="activation-token-highlight"]');
    if (!target) return;
    target.scrollIntoView({ block: "center" });
    target.focus({ preventScroll: true });
  }

  return {
    load,
    focusSearch: () => $("generation-search").focus(),
  };
}

function parseActivationTarget(route) {
  if (route.token === undefined || route.char_start === undefined || route.char_end === undefined) {
    return null;
  }
  const tokenIdx = Number(route.token);
  const charStart = Number(route.char_start);
  const charEnd = Number(route.char_end);
  if (![tokenIdx, charStart, charEnd].every(Number.isInteger)) return null;
  return {
    sampleId: route.question ?? "",
    seed: String(route.seed ?? ""),
    tokenIdx,
    charStart,
    charEnd,
  };
}

function hasEntropyTimesteps(row) {
  return Array.isArray(row.timesteps) && row.timesteps.some(timestep => Number.isFinite(entropyValue(timestep)));
}

function entropyValue(timestep) {
  if (Array.isArray(timestep.entropy)) {
    const values = timestep.entropy.filter(Number.isFinite);
    return values.length ? Math.max(...values) : NaN;
  }
  return Number.isFinite(timestep.entropy) ? timestep.entropy : NaN;
}
