const MODULUS = 8;
const FINAL_TABLE = [3, 6, 0, 4, 7, 2, 1, 5];

const elements = {
  start: document.querySelector("#start-state"),
  stepOne: document.querySelector("#step-one"),
  stepTwo: document.querySelector("#step-two"),
  stateZero: document.querySelector("#state-zero"),
  stateOne: document.querySelector("#state-one"),
  currentState: document.querySelector("#current-state"),
  finalAnswer: document.querySelector("#final-answer"),
  opOne: document.querySelector("#op-one"),
  opTwo: document.querySelector("#op-two"),
  program: document.querySelector("#program-text"),
};

function populateStateSelect(select, initial) {
  for (let state = 0; state < MODULUS; state += 1) {
    const option = document.createElement("option");
    option.value = String(state);
    option.textContent = String(state);
    option.selected = state === initial;
    select.append(option);
  }
}

function currentProgram() {
  const start = Number(elements.start.value);
  const stepOne = Number(elements.stepOne.value);
  const stepTwo = Number(elements.stepTwo.value);
  const stateOne = (start + stepOne) % MODULUS;
  const stateTwo = (stateOne + stepTwo) % MODULUS;
  return {
    start,
    stepOne,
    stepTwo,
    stateOne,
    stateTwo,
    answer: FINAL_TABLE[stateTwo],
  };
}

function renderProgram() {
  const program = currentProgram();
  elements.stateZero.textContent = program.start;
  elements.stateOne.textContent = program.stateOne;
  elements.currentState.textContent = program.stateTwo;
  elements.finalAnswer.textContent = program.answer;
  elements.opOne.textContent = `+${program.stepOne} mod 8`;
  elements.opTwo.textContent = `+${program.stepTwo} mod 8`;
  elements.program.textContent = [
    `Start state: ${program.start}`,
    `Step 1: add ${program.stepOne} modulo 8`,
    `Step 2: add ${program.stepTwo} modulo 8`,
    `FINAL: look up the state in [${FINAL_TABLE.join(", ")}]`,
  ].join("\n");
  renderCondition(activeCondition());
}

function flow(parts) {
  return parts.map((part) => {
    const span = document.createElement("span");
    span.className = part.type;
    span.textContent = part.text;
    return span;
  });
}

function activeCondition() {
  return document.querySelector('[role="tab"][aria-selected="true"]').dataset.condition;
}

function setFlow(target, parts) {
  target.replaceChildren(...flow(parts));
}

function renderCondition(condition) {
  const program = currentProgram();
  const outcome = condition === "outcome";
  const tabs = document.querySelectorAll('[role="tab"]');
  for (const tab of tabs) {
    tab.setAttribute("aria-selected", String(tab.dataset.condition === condition));
    tab.tabIndex = tab.dataset.condition === condition ? 0 : -1;
  }

  document.querySelector("#condition-kicker").textContent = outcome ? "Control condition" : "State-interface condition";
  document.querySelector("#condition-name").textContent = outcome ? "Outcome-only LoRA" : "Explicit-handoff LoRA";
  document.querySelector("#condition-description").textContent = outcome
    ? "The model sees the full history and FINAL rule, then learns only the answer. There is no intermediate state target."
    : "One call learns history → current state. A separate call learns true current state + FINAL → answer. The second call never sees the history.";

  const first = document.querySelector("#pass-one-flow");
  const second = document.querySelector("#pass-two-flow");
  if (outcome) {
    document.querySelector("#pass-one-label").textContent = "Compose";
    document.querySelector("#pass-two-label").textContent = "Compose again";
    setFlow(first, [
      { type: "token-block", text: "full history + FINAL" },
      { type: "flow-arrow", text: "→" },
      { type: "model-block", text: "Qwen + LoRA" },
      { type: "flow-arrow", text: "→" },
      { type: "target-token", text: String(program.answer) },
    ]);
    setFlow(second, [
      { type: "token-block", text: "full history + FINAL" },
      { type: "flow-arrow", text: "→" },
      { type: "model-block", text: "Qwen + LoRA" },
      { type: "flow-arrow", text: "→" },
      { type: "target-token", text: String(program.answer) },
    ]);
    document.querySelector("#pass-one-note").textContent = "The prompt includes every history step and the pointer table.";
    document.querySelector("#pass-two-note").textContent = "A duplicate pass matches the explicit condition's two calls and two target tokens.";
    document.querySelector("#loss-equation").textContent = "L = CE(answer)";
  } else {
    document.querySelector("#pass-one-label").textContent = "Synthesize state";
    document.querySelector("#pass-two-label").textContent = "Use gold state";
    setFlow(first, [
      { type: "token-block", text: "history only" },
      { type: "flow-arrow", text: "→" },
      { type: "model-block", text: "Qwen + LoRA" },
      { type: "flow-arrow", text: "→" },
      { type: "target-token", text: String(program.stateTwo) },
    ]);
    setFlow(second, [
      { type: "token-block", text: `state ${program.stateTwo} + FINAL` },
      { type: "flow-arrow", text: "→" },
      { type: "model-block", text: "same Qwen + LoRA" },
      { type: "flow-arrow", text: "→" },
      { type: "target-token", text: String(program.answer) },
    ]);
    document.querySelector("#pass-one-note").textContent = `The one-token target ${program.stateTwo} is the exact state after the history.`;
    document.querySelector("#pass-two-note").textContent = "Training uses the simulator's true state. The original history is absent from this prompt.";
    document.querySelector("#loss-equation").textContent = "L = CE(state) + CE(answer | state, FINAL)";
  }
}

function selectTab(tab) {
  renderCondition(tab.dataset.condition);
  const url = new URL(window.location.href);
  url.searchParams.set("condition", tab.dataset.condition);
  window.history.replaceState({}, "", url);
}

populateStateSelect(elements.start, 2);
populateStateSelect(elements.stepOne, 0);
populateStateSelect(elements.stepTwo, 6);

for (const select of [elements.start, elements.stepOne, elements.stepTwo]) {
  select.addEventListener("change", renderProgram);
}

const tabs = [...document.querySelectorAll('[role="tab"]')];
for (const [index, tab] of tabs.entries()) {
  tab.addEventListener("click", () => selectTab(tab));
  tab.addEventListener("keydown", (event) => {
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    event.preventDefault();
    const offset = event.key === 'ArrowRight' ? 1 : -1;
    const next = tabs[(index + offset + tabs.length) % tabs.length];
    selectTab(next);
    next.focus();
  });
}

const requestedCondition = new URLSearchParams(window.location.search).get("condition");
const initialTab = tabs.find((tab) => tab.dataset.condition === requestedCondition) || tabs[0];
selectTab(initialTab);
renderProgram();
