(() => {
  const GiB = 2 ** 30;
  const capacity = 184.31 * GiB;
  const { D, H, C, Z, F, experts, topK, layers, partition } = window.KimiK3Model;
  const keys = ["pp", "ep", "fsdp", "stage", "seq", "rows", "mbs", "ac", "unit", "grad", "reserve"];
  const controls = Object.fromEntries(keys.map((key) => [key, document.getElementById(`mem-${key}`)]));
  const defaults = { pp: 8, ep: 64, fsdp: "on", stage: 3, seq: 4096, rows: 1, mbs: 64, ac: "full", unit: "stage", grad: "deferred", reserve: 24 };
  const colors = {
    master: "master",
    momentum: "momentum",
    adam: "adam",
    gradient: "gradient",
    compute: "compute",
    activation: "activation",
    reserve: "reserve",
  };
  const bytes = (value) =>
    value >= 2 ** 40
      ? `${(value / 2 ** 40).toFixed(2)} TiB`
      : value >= GiB
        ? `${(value / GiB).toFixed(2)} GiB`
        : `${(value / 2 ** 20).toFixed(2)} MiB`;
  const count = (value) => (value >= 1e12 ? `${(value / 1e12).toFixed(3)}T` : value >= 1e9 ? `${(value / 1e9).toFixed(3)}B` : value.toLocaleString());
  const sum = (items, get) => items.reduce((total, item) => total + get(item), 0);
  function activationCandidates(layer, tokens) {
    const input = tokens * D * 2;
    const attention = layer.mla ? tokens * (2 * H * 192 + 3 * C + 1536 + 576) * 2 : tokens * (5 * C * 2 + C * 4 + H * 4);
    if (layer.index === 0) return input + attention + tokens * 3 * 33792 * 2;
    const routedTokens = tokens * topK;
    const routed = (2 * routedTokens * Z + 3 * routedTokens * F) * 2;
    const shared = tokens * 3 * (2 * F) * 2;
    const latent = tokens * Z * 2;
    const routing = tokens * experts * 4;
    return input + attention + routed + shared + latent + routing;
  }

  function model(config) {
    const dp = 512 / config.pp;
    const denseFsdp = config.fsdp === "on" ? dp : 1;
    const expertFsdp = config.fsdp === "on" ? dp / config.ep : 1;
    const stage = partition(config.pp, config.stage);
    const storedDense = stage.dense / denseFsdp;
    const storedExperts = stage.routed / (config.ep * expertFsdp);
    const stored = storedDense + storedExperts;
    const adam = stage.adam / denseFsdp;
    const computeParameters =
      config.unit === "stage"
        ? stage.dense + stage.routed / config.ep
        : Math.max(stage.front, stage.back, ...stage.layers.map((l) => l.dense + l.routed / config.ep));
    const live = Math.min(config.mbs, config.pp - config.stage);
    const tokens = config.rows * config.seq;
    const masterBytes = 4 * stored;
    const momentumBytes = 4 * (stored - adam);
    const adamBytes = 8 * adam;
    const reducedGradientBytes = 4 * stored;
    const gradientBytes = config.grad === "deferred" ? 4 * (stage.dense + stage.routed / config.ep) : reducedGradientBytes;
    const computeBytes = 2 * computeParameters;
    const activationBytes =
      config.ac === "full" ? stage.layers.length * tokens * D * 2 * live : sum(stage.layers, (l) => activationCandidates(l, tokens)) * live;
    const reserveBytes = config.reserve * GiB;
    const parts = [
      {
        key: "master",
        label: "FP32 master shards",
        value: masterBytes,
        info: "Resident parameter storage on this rank. Includes routed and non-routed weights.",
      },
      {
        key: "momentum",
        label: "FP32 Muon momentum",
        value: momentumBytes,
        info: "One momentum tensor for each Muon parameter; same storage layout as its gradient.",
      },
      { key: "adam", label: "FP32 AdamW moments", value: adamBytes, info: "Two moment tensors, only for the AdamW-assigned parameters." },
      {
        key: "gradient",
        label: config.grad === "deferred" ? "FP32 unreduced stage gradients" : "FP32 reduced gradient shards",
        value: gradientBytes,
        info:
          config.grad === "deferred"
            ? "All stage gradients before the delayed DP reduction. Dense gradients are not divided by FSDP here. Replaces the reduced gradient row for this phase."
            : "Optimistic reduced-state phase. Does not represent the deferred PP accumulation footprint.",
      },
      {
        key: "compute",
        label: "BF16 compute views",
        value: computeBytes,
        info:
          config.unit === "stage"
            ? "Whole stage gathered over FSDP; still PP-local and EP-local. A residency assumption, not an all-gather allocation trace."
            : "Largest single layer or front/back module group. An optimistic alternative, not the default PP cache policy.",
      },
      {
        key: "activation",
        label: config.ac === "full" ? "Checkpoint-input floor" : "Selected activation candidates",
        value: activationBytes,
        info:
          config.ac === "full"
            ? `${
                stage.layers.length
              } layer inputs x ${tokens.toLocaleString()} tokens x 7168 x 2 bytes x ${live} unfinished forwards. Omits residual-bank keepalives and recomputation workspaces.`
            : "Representative BF16 input, attention, dispatch/combine, gate/up, SiTU product, shared/latent tensors, and FP32 router scores. Not an exhaustive saved-tensor audit; backend aliasing and recompute can change the tally.",
      },
      {
        key: "reserve",
        label: "Chosen workspace allowance",
        value: reserveBytes,
        info: "User-selected budget for unmodeled memory. Not a measurement, calibration, or automatic OOM safeguard.",
      },
    ];
    return {
      config,
      dp,
      denseFsdp,
      expertFsdp,
      stage,
      stored,
      adam,
      live,
      tokens,
      stateTotal: masterBytes + momentumBytes + adamBytes + reducedGradientBytes,
      reducedGradientBytes,
      parts,
      budget: sum(parts, (p) => p.value),
      globalTokens: dp * config.mbs * tokens,
      globalRows: dp * config.mbs * config.rows,
    };
  }

  function readConfig() {
    const numeric = new Set(["pp", "ep", "stage", "seq", "rows", "mbs", "reserve"]);
    return Object.fromEntries(keys.map((key) => [key, numeric.has(key) ? Number(controls[key].value) : controls[key].value]));
  }

  function updateStages(stage) {
    const pp = Number(controls.pp.value);
    controls.stage.replaceChildren();
    for (let i = 0; i < pp; i++) {
      const option = document.createElement("option");
      option.value = i;
      option.textContent = `Rank ${i}`;
      controls.stage.appendChild(option);
    }
    controls.stage.value = Math.min(stage, pp - 1);
  }

  function validExpertDegree() {
    const dp = 512 / Number(controls.pp.value);
    const warning = document.getElementById("config-warning");
    const invalid = Number(controls.ep.value) > dp;
    for (const option of controls.ep.options) option.disabled = Number(option.value) > dp;
    warning.hidden = !invalid;
    if (invalid) {
      controls.ep.value = dp;
      warning.textContent = `This PP setting leaves ${dp} ranks per stage. EP was reduced to ${dp}; EP cannot exceed the ranks available in one pipeline stage.`;
    }
  }

  function renderLedger(result) {
    const tbody = document.querySelector("#memory-ledger tbody");
    tbody.replaceChildren();
    for (const part of result.parts) {
      const row = document.createElement("tr");
      row.dataset.category = part.key;
      const name = document.createElement("td");
      const swatch = document.createElement("i");
      swatch.className = "swatch";
      swatch.style.background = `var(--${colors[part.key]})`;
      name.append(swatch, ` ${part.label}`);
      const value = document.createElement("td");
      value.className = "bytes";
      value.dataset.bytes = part.value;
      value.textContent = bytes(part.value);
      const explanation = document.createElement("td");
      explanation.textContent = part.info;
      row.append(name, value, explanation);
      tbody.appendChild(row);
    }
    const chart = document.getElementById("budget-bar");
    const legend = document.getElementById("budget-legend");
    chart.replaceChildren();
    legend.replaceChildren();
    const axis = Math.max(capacity, result.budget);
    for (const part of result.parts) {
      const segment = document.createElement("div");
      segment.className = "budget-segment";
      segment.style.width = `${(100 * part.value) / axis}%`;
      segment.style.background = `var(--${colors[part.key]})`;
      segment.title = `${part.label}: ${bytes(part.value)}`;
      chart.appendChild(segment);
      const label = document.createElement("span");
      const swatch = document.createElement("i");
      swatch.className = "swatch";
      swatch.style.background = segment.style.background;
      label.append(swatch, `${part.label}: ${bytes(part.value)}`);
      legend.appendChild(label);
    }
    const marker = document.createElement("div");
    marker.className = "capacity-mark";
    marker.style.left = `${(100 * capacity) / axis}%`;
    marker.title = "184.31 GiB reported device capacity";
    chart.appendChild(marker);
    document.getElementById("budget-axis").textContent = `Linear scale: 0 to ${bytes(axis)}. Dashed marker = reported HBM.`;
  }

  function renderPipeline(result) {
    const container = document.getElementById("pipeline-lifetimes");
    container.replaceChildren();
    for (let rank = 0; rank < result.config.pp; rank++) {
      const live = Math.min(result.config.mbs, result.config.pp - rank);
      const row = document.createElement("div");
      row.className = `life-row${rank === result.config.stage ? " selected" : ""}`;
      const name = document.createElement("span");
      name.textContent = `Rank ${rank}`;
      const cells = document.createElement("div");
      cells.className = "life-cells";
      cells.setAttribute("aria-label", `${live} unfinished forwards`);
      for (let i = 0; i < live; i++) cells.appendChild(document.createElement("span"));
      const number = document.createElement("small");
      number.textContent = `${live} live forward${live === 1 ? "" : "s"}`;
      row.append(name, cells, number);
      container.appendChild(row);
    }
    document.getElementById("pipeline-caption").textContent =
      `Ordinary 1F1B / ${result.config.pp} physical stages / ${result.config.mbs} microbatches`;
  }

  function render() {
    const result = model(readConfig());
    const c = result.config;
    const stateTotal = document.getElementById("state-total");
    stateTotal.textContent = bytes(result.stateTotal);
    stateTotal.dataset.bytes = result.stateTotal;
    document.getElementById("state-detail").textContent = `Rank ${c.stage}: ${result.stage.layers.length} layers, ${count(
      result.stored
    )} ideally divided resident parameters. Includes FP32 reduced gradient shards, not full deferred gradients.`;
    const budgetTotal = document.getElementById("budget-total");
    budgetTotal.textContent = bytes(result.budget);
    budgetTotal.dataset.bytes = result.budget;
    budgetTotal.classList.toggle("over", result.budget > capacity);
    document.getElementById("budget-verdict").textContent =
      result.budget > capacity
        ? `The chosen planning assumptions exceed reported HBM by ${bytes(
            result.budget - capacity
          )}. Inspect the assumptions and working-set lifetimes; this is not a measured allocation trace.`
        : `${((100 * result.budget) / capacity).toFixed(
            1
          )}% of reported HBM under these assumptions. A subtotal below capacity is not proof that a run fits.`;
    document.getElementById("ownership-note").textContent = `512 GPUs / PP${c.pp} = ${result.dp} ranks per stage. Dense FSDP${result.denseFsdp}; EP${
      c.ep
    }; expert FSDP${result.expertFsdp}. T=${result.tokens.toLocaleString()}, peak unfinished forwards=${
      result.live
    }. Global batch=${result.globalRows.toLocaleString()} rows / ${result.globalTokens.toLocaleString()} tokens. Compute and all persistent state dtypes remain fixed.`;
    const map = document.getElementById("stage-map");
    map.replaceChildren();
    for (let rank = 0; rank < c.pp; rank++) {
      const own = partition(c.pp, rank);
      const button = document.createElement("button");
      button.type = "button";
      button.setAttribute("aria-pressed", String(rank === c.stage));
      button.textContent = `PP ${rank}`;
      const label = document.createElement("span");
      label.textContent = `L${own.layers[0].index + 1}-${own.layers[own.layers.length - 1].index + 1}`;
      button.appendChild(label);
      button.addEventListener("click", () => {
        controls.stage.value = rank;
        render();
      });
      map.appendChild(button);
    }
    renderLedger(result);
    renderPipeline(result);
  }

  function applyConfig(config) {
    controls.pp.value = config.pp;
    updateStages(config.stage);
    for (const key of keys) controls[key].value = config[key];
    validExpertDegree();
    render();
  }

  const deckSteps = [
    {
      title: "1 / No model partitioning",
      description:
        "512 replicas would each need to own the entire model. The FP32 update-state tally alone is about 30.35 TiB per GPU. More GPUs are not useful for capacity until something is actually partitioned.",
      config: { pp: 1, ep: 1, fsdp: "off", stage: 0, ac: "none" },
    },
    {
      title: "2 / Divide the depth with PP8",
      description:
        "Give each PP rank a contiguous slice of the model. The displayed interior rank owns 12 layers. Master weights, momentum, and working weights shrink with the local model slice, but the expert bank is still enormous.",
      config: { ep: 1, fsdp: "off", ac: "none" },
    },
    {
      title: "3 / Partition routed experts with EP64",
      description:
        "Keep only 14 of 896 routed experts per MoE layer on each rank. Non-routed weights remain replicated. Balanced EP still receives assignments from other input ranks: local expert activation volume does not simply divide by 64.",
      config: { fsdp: "off", ac: "none" },
    },
    {
      title: "4 / Shard resident state with FSDP",
      description:
        "Shard dense master weights and optimizer history over the 64 ranks of this pipeline stage. Expert FSDP is one because EP64 already uses all 64 ranks. During deferred PP accumulation, dense gradients can still be full-stage, not final reduced shards.",
      config: { ac: "none" },
    },
    {
      title: "5 / Checkpoint layer internals",
      description:
        "Keep boundary inputs and regenerate internal activations in backward. The checkpoint-input bar is only a floor; residual keepalives and the current recomputation's working tensors still exist. This substantially changes saved activation candidates, not resident master state.",
      config: {},
    },
    {
      title: "6 / Validate the actual runtime",
      description:
        "The structural formula is unchanged. Compile pointwise regions, bound buffer lifetimes, and initialize real communicators before memory pressure. A 512-GB200 4K reference completed 12 updates with a 146.18-GiB maximum logged tensor allocation; that measurement, not this subtotal, establishes feasibility.",
      config: {},
    },
  ];
  let deckIndex = 0;
  try {
    const hash = decodeURIComponent(location.hash.slice(1));
    if (hash.startsWith("d:fitdeck=")) {
      const parsed = JSON.parse(hash.slice("d:fitdeck=".length));
      if (Number.isInteger(parsed.i) && parsed.i >= 0 && parsed.i < deckSteps.length) deckIndex = parsed.i;
    }
  } catch {}

  function renderDeck() {
    const step = deckSteps[deckIndex];
    const result = model({ ...defaults, ...step.config });
    document.getElementById("deck-position").textContent = `Step ${deckIndex + 1} of ${deckSteps.length}`;
    document.getElementById("deck-title").textContent = step.title;
    document.getElementById("deck-description").textContent = step.description;
    document.getElementById("deck-prev").disabled = deckIndex === 0;
    document.getElementById("deck-next").disabled = deckIndex === deckSteps.length - 1;
    const bars = document.getElementById("deck-bars");
    bars.replaceChildren();
    for (const part of result.parts) {
      const row = document.createElement("div");
      row.className = "log-row";
      const name = document.createElement("span");
      name.textContent = part.label;
      const track = document.createElement("div");
      track.className = "log-track";
      const fill = document.createElement("div");
      fill.className = "log-fill";
      fill.style.width = `${part.value > 0 ? Math.min(100, Math.max(0, (100 * Math.log2(part.value / 2 ** 20)) / 26)) : 0}%`;
      fill.style.background = `var(--${colors[part.key]})`;
      const marker = document.createElement("div");
      marker.className = "capacity-mark";
      marker.style.left = `${(100 * Math.log2(capacity / 2 ** 20)) / 26}%`;
      marker.title = "184.31 GiB reported HBM";
      track.append(fill, marker);
      const value = document.createElement("span");
      value.className = "log-value";
      value.textContent = bytes(part.value);
      row.append(name, track, value);
      bars.appendChild(row);
    }
    document.getElementById("deck-total").textContent = `Update-state tally: ${bytes(result.stateTotal)}. Planning subtotal + reserve: ${bytes(
      result.budget
    )}.`;
    document.getElementById("deck-caveat").textContent =
      "Dashed lines mark reported device capacity. The activation candidates are approximate, the 24-GiB reserve is chosen, and the BF16-view bar assumes whole-stage residency. Only PP8 / FSDP64 / EP64 with the stated baseline policy is the measured reference. Non-default steps illustrate ownership, not runnable recommendations.";
  }

  for (const direction of ["prev", "next"])
    document.getElementById(`deck-${direction}`).addEventListener("click", () => {
      deckIndex += direction === "next" ? 1 : -1;
      renderDeck();
      if (location.protocol === "http:" || location.protocol === "https:")
        history.replaceState(null, "", `#d:fitdeck=${encodeURIComponent(JSON.stringify({ i: deckIndex }))}`);
    });
  document.getElementById("deck-apply").addEventListener("click", () => {
    applyConfig({ ...defaults, ...deckSteps[deckIndex].config });
    document.getElementById("lab").scrollIntoView({ behavior: "smooth", block: "start" });
  });
  for (const key of keys)
    controls[key].addEventListener("change", () => {
      if (key === "pp") {
        updateStages(Number(controls.stage.value));
        validExpertDegree();
      }
      render();
    });
  const themeButton = document.getElementById("theme-toggle");
  let savedTheme;
  try {
    savedTheme = localStorage.getItem("kimi-k3-study-theme");
  } catch {}
  document.documentElement.classList.toggle("dark", savedTheme === "dark" || (!savedTheme && matchMedia("(prefers-color-scheme: dark)").matches));
  const themeLabel = () => {
    themeButton.textContent = document.documentElement.classList.contains("dark") ? "Light" : "Dark";
  };
  themeButton.addEventListener("click", () => {
    document.documentElement.classList.toggle("dark");
    try {
      localStorage.setItem("kimi-k3-study-theme", document.documentElement.classList.contains("dark") ? "dark" : "light");
    } catch {}
    themeLabel();
  });
  themeLabel();
  applyConfig(defaults);
  renderDeck();
})();
