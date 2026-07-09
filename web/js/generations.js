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
    "generation-activation-delta",
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
    const hasActivationDelta = rows.some(hasActivationDeltaTimesteps);
    $("generation-entropy").disabled = !hasEntropy;
    $("generation-activation-delta").disabled = !hasActivationDelta;
    $("generation-entropy").checked = hasEntropy && route.entropy === "1" && route.activation_delta !== "1";
    $("generation-activation-delta").checked = hasActivationDelta && route.activation_delta === "1";
    $("entropy-status").textContent = statusText(hasEntropy, hasActivationDelta);
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
    $("generation-activation-delta").checked = false;
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
        row.patch_mode,
        row.condition,
        row.pair_id,
        questionText(sample.prompt),
      ].some(value => String(value ?? "").toLowerCase().includes(search));
    });

    const sort = $("generation-sort").value;
    filteredRows.sort((a, b) => {
      if (sort === "longest") return Number(b.reasoning_length ?? 0) - Number(a.reasoning_length ?? 0);
      if (sort === "shortest") return Number(a.reasoning_length ?? 0) - Number(b.reasoning_length ?? 0);
      return a.sample_id.localeCompare(b.sample_id) || Number(a.seed) - Number(b.seed);
    });

    if ($("generation-activation-delta").checked) $("generation-entropy").checked = false;
    $("entropy-legend").hidden = (!$("generation-entropy").checked && !$("generation-activation-delta").checked)
      || ($("generation-entropy").disabled && $("generation-activation-delta").disabled);
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
    const patchMetadata = row.patch_mode
      ? `<span>pair ${escapeHtml(row.pair_id)}</span><span>${escapeHtml(row.patch_mode)}</span><span>${escapeHtml(row.condition)}</span>`
      : "";
    const hasTrajectory = (run.interactive_plots?.length ?? 0)
      + (run.step_classification_plots?.length ?? 0) > 0;
    return `<article class="generation-card" data-sample-id="${escapeHtml(row.sample_id)}" data-seed="${escapeHtml(row.seed)}">
      <header class="generation-card-header">
        <div class="generation-identity">
          <strong title="${escapeHtml(questionText(sample.prompt))}">${escapeHtml(row.sample_id)}</strong>
          <div class="generation-meta">
            <span class="status-pill ${status}">${status === "unknown" ? "not scored" : status}</span>
            <span>sub-run ${escapeHtml(row.seed)}</span>
            ${patchMetadata}
            <span>answer <strong>${escapeHtml(row.produced_answer ?? "—")}</strong></span>
            <span>${formatNumber(row.reasoning_length)} reasoning tokens</span>
          </div>
        </div>
        <div class="generation-actions">
          <button class="text-button" type="button" data-action="toggle-output">Expand output</button>
          ${hasTrajectory ? `<button class="secondary-button" type="button" data-action="open-trajectory">View latent</button>` : ""}
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
    const chips = values.slice(0, 16).map(value => {
      const target = tokenTarget(row, Number(value));
      return `<button class="marker-chip-button" type="button" data-action="jump-marker" data-token="${escapeHtml(value)}" data-char-start="${target.charStart}" data-char-end="${target.charEnd}" title="Jump to token ${escapeHtml(value)}">${escapeHtml(value)}</button>`;
    }).join("");
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
    if ($("generation-activation-delta").checked && hasActivationDeltaTimesteps(row)) {
      return formatMetricTokens(
        row,
        activationDeltaValue,
        activationDeltaColor,
        value => `Activation change ${value.toFixed(2)}`,
      );
    }
    if (!$("generation-entropy").checked || !hasEntropyTimesteps(row)) {
      return escapeHtml(row.produced_text ?? "");
    }
    return formatMetricTokens(
      row,
      entropyValue,
      entropyColor,
      value => `Entropy ${value.toFixed(3)}`,
    );
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
    if (button.dataset.action === "jump-marker") {
      const tokenIdx = Number(button.dataset.token);
      activationTarget = {
        sampleId: card.dataset.sampleId,
        seed: String(card.dataset.seed),
        tokenIdx,
        charStart: Number(button.dataset.charStart),
        charEnd: Number(button.dataset.charEnd),
      };
      syncQuery();
      renderRows();
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
      activation_delta: $("generation-activation-delta").checked ? "1" : "",
      token: activationTarget?.tokenIdx ?? "",
      char_start: activationTarget?.charStart ?? "",
      char_end: activationTarget?.charEnd ?? "",
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
  if (route.token === undefined) {
    return null;
  }
  const tokenIdx = Number(route.token);
  const charStart = Number(route.char_start ?? 0);
  const charEnd = Number(route.char_end ?? charStart);
  if (![tokenIdx, charStart, charEnd].every(Number.isInteger)) return null;
  return {
    sampleId: route.question ?? "",
    seed: String(route.seed ?? ""),
    tokenIdx,
    charStart,
    charEnd,
  };
}

function tokenTarget(row, tokenIdx) {
  const text = String(row.produced_text ?? "");
  const fromTimesteps = tokenTargetFromTimesteps(row, tokenIdx, text.length);
  if (fromTimesteps) return fromTimesteps;
  const tokenCount = Number(row.reasoning_length ?? row.generated_token_ids?.length ?? row.token_count);
  const fraction = Number.isFinite(tokenCount) && tokenCount > 1
    ? Math.max(0, Math.min(1, tokenIdx / (tokenCount - 1)))
    : 0;
  const charStart = Math.max(0, Math.min(text.length, Math.floor(fraction * text.length)));
  const charEnd = Math.min(text.length, Math.max(charStart + 1, charStart + 12));
  return { charStart, charEnd };
}

function tokenTargetFromTimesteps(row, tokenIdx, textLength) {
  if (!Array.isArray(row.timesteps)) return null;
  const step = row.timesteps.find(item => Number(item.token_idx ?? item.token_index ?? item.index) === tokenIdx);
  if (!step) return null;
  const charStart = Number(step.char_start ?? step.start_char ?? step.start);
  const charEnd = Number(step.char_end ?? step.end_char ?? step.end);
  if (![charStart, charEnd].every(Number.isInteger) || charStart < 0 || charEnd < charStart || charEnd > textLength) {
    return null;
  }
  return { charStart, charEnd };
}

function hasEntropyTimesteps(row) {
  return Array.isArray(row.timesteps) && row.timesteps.some(timestep => Number.isFinite(entropyValue(timestep)));
}

function hasActivationDeltaTimesteps(row) {
  return Array.isArray(row.timesteps) && row.timesteps.some(timestep => Number.isFinite(activationDeltaValue(timestep)));
}

function entropyValue(timestep) {
  if (Array.isArray(timestep.entropy)) {
    const values = timestep.entropy.filter(Number.isFinite);
    return values.length ? Math.max(...values) : NaN;
  }
  return Number.isFinite(timestep.entropy) ? timestep.entropy : NaN;
}

function activationDeltaValue(timestep) {
  return Number.isFinite(timestep.activation_delta) ? timestep.activation_delta : NaN;
}

function formatMetricTokens(row, valueFn, colorFn, titleFn) {
  const timesteps = metricTimesteps(row, valueFn);
  const values = timesteps.map(item => item.value);
  const low = Math.min(...values);
  const high = Math.max(...values);
  const span = Math.max(high - low, 1e-9);
  const text = String(row.produced_text ?? "");
  if (text && timesteps.some(item => item.charStart !== null && item.charEnd !== null)) {
    return formatMetricIntervals(text, timesteps, low, span, colorFn, titleFn);
  }
  return timesteps.map(({ timestep, value }) => {
    const token = escapeHtml(timestep.token_str ?? "");
    const level = (value - low) / span;
    const { background, color } = colorFn(level);
    return `<span class="token-entropy" title="${escapeHtml(titleFn(value))}" style="background:${background};color:${color}">${token}</span>`;
  }).join("");
}

function metricTimesteps(row, valueFn) {
  return (row.timesteps ?? []).map((timestep, index) => {
    const value = valueFn(timestep);
    const tokenIdx = Number(timestep.token_idx ?? timestep.token_index ?? timestep.index ?? index);
    const target = Number.isFinite(tokenIdx)
      ? tokenTarget(row, tokenIdx)
      : { charStart: null, charEnd: null };
    return {
      timestep,
      value,
      tokenIdx,
      charStart: Number.isInteger(target.charStart) ? target.charStart : null,
      charEnd: Number.isInteger(target.charEnd) ? target.charEnd : null,
    };
  }).filter(item => Number.isFinite(item.value));
}

function formatMetricIntervals(text, timesteps, low, span, colorFn, titleFn) {
  const ordered = [...timesteps]
    .filter(item => item.charStart !== null && item.charEnd !== null)
    .sort((a, b) => a.charStart - b.charStart || a.charEnd - b.charEnd);
  let cursor = 0;
  const pieces = [];
  for (let index = 0; index < ordered.length; index += 1) {
    const item = ordered[index];
    const next = ordered[index + 1];
    const start = Math.max(cursor, Math.min(text.length, item.charStart));
    const fallbackEnd = Math.max(start + 1, item.charEnd);
    const intervalEnd = next
      ? Math.max(fallbackEnd, Math.floor((item.charEnd + next.charStart) / 2))
      : fallbackEnd;
    const end = Math.max(start, Math.min(text.length, intervalEnd));
    if (cursor < start) pieces.push(escapeHtml(text.slice(cursor, start)));
    const level = (item.value - low) / span;
    const { background, color } = colorFn(level);
    pieces.push(
      `<span class="token-entropy" title="${escapeHtml(titleFn(item.value))}" style="background:${background};color:${color}">${escapeHtml(text.slice(start, end))}</span>`,
    );
    cursor = end;
  }
  if (cursor < text.length) pieces.push(escapeHtml(text.slice(cursor)));
  return pieces.join("");
}

function entropyColor(level) {
  return {
    background: `hsl(${152 - level * 145} 74% ${92 - level * 48}%)`,
    color: level > 0.56 ? "white" : "#0d0f11",
  };
}

function activationDeltaColor(level) {
  return {
    background: `hsl(${205 - level * 165} 86% ${88 - level * 38}%)`,
    color: level > 0.6 ? "white" : "#0d0f11",
  };
}

function statusText(hasEntropy, hasActivationDelta) {
  if (hasEntropy && hasActivationDelta) {
    return "Color is normalized within each generation.";
  }
  if (hasActivationDelta) {
    return "Activation color is normalized within each generation.";
  }
  if (hasEntropy) {
    return "Entropy color is normalized within each generation.";
  }
  return "This run did not store timestep entropy or activation-change diagnostics.";
}
