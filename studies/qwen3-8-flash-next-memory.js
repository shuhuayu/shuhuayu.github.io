(() => {
  const A = window.QwenFlashNextAccounting;
  const M = window.QwenFlashNextMemory;
  const $ = (id) => document.getElementById(id);
  // The context length is fixed at the model's 4K; every other default has a control.
  const keys = Object.keys(M.defaults).filter((key) => document.getElementById(`mem-${key}`));
  const numeric = new Set(keys.filter((key) => typeof M.defaults[key] === "number"));
  const controls = Object.fromEntries(keys.map((key) => [key, $(`mem-${key}`)]));
  const colors = {
    master: "master",
    momentum: "momentum",
    adam: "adam",
    gradient: "gradient",
    compute: "compute",
    activation: "activation",
    slot: "slot",
    scratch: "scratch",
    working: "working",
    reserve: "reserve",
  };
  const view = { deck: 0 };
  let current;

  const sum = A.sum;
  const esc = (value) => String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
  const count = (value) => Math.round(value).toLocaleString("en-US");
  const bytes = (value) => {
    if (value >= 2 ** 40) return `${(value / 2 ** 40).toFixed(2)} TiB`;
    if (value >= M.GiB) return `${(value / M.GiB).toFixed(2)} GiB`;
    if (value >= M.MiB) return `${(value / M.MiB).toFixed(1)} MiB`;
    if (value >= 1024) return `${(value / 1024).toFixed(1)} KiB`;
    return `${Math.round(value)} B`;
  };
  const layerRange = (chunk) => `L${chunk.layers[0].index + 1}-${chunk.layers.at(-1).index + 1}`;

  function readConfig() {
    return { ...M.defaults, ...Object.fromEntries(keys.map((key) => [key, numeric.has(key) ? Number(controls[key].value) : controls[key].value])) };
  }

  function setConfig(config) {
    controls.stage.innerHTML = Array.from({ length: config.pp }, (_, rank) => `<option value="${rank}">${rank}</option>`).join("");
    for (const key of keys) controls[key].value = String(config[key]);
  }

  function valid(config) {
    try {
      M.normalize(config);
      return true;
    } catch {
      return false;
    }
  }

  // Keep the selects inside the model's domain; disable options that cannot work.
  function constrain() {
    const notices = [];
    const pp = Number(controls.pp.value);
    const stage = Math.min(Number(controls.stage.value || 0), pp - 1);
    controls.stage.innerHTML = Array.from({ length: pp }, (_, rank) => `<option value="${rank}">${rank}</option>`).join("");
    controls.stage.value = String(stage);
    for (const key of ["ep", "vp", "mbs"]) {
      const options = [...controls[key].options];
      for (const option of options) option.disabled = !valid({ ...readConfig(), [key]: Number(option.value) });
      if (controls[key].selectedOptions[0]?.disabled) {
        const wanted = Number(controls[key].value);
        const enabled = options.filter((option) => !option.disabled).map((option) => Number(option.value));
        if (!enabled.length) continue;
        const below = enabled.filter((value) => value <= wanted);
        const choice = below.length ? below.at(-1) : enabled[0];
        notices.push(`${{ ep: "EP", vp: "Virtual stages", mbs: "Microbatches" }[key]} changed from ${wanted} to ${choice} to keep the layout valid.`);
        controls[key].value = String(choice);
      }
    }
    $("config-warning").hidden = !notices.length;
    $("config-warning").textContent = notices.join(" ");
  }

  function saveURL() {
    const hash = new URLSearchParams({ memory: JSON.stringify(readConfig()) }).toString();
    history.replaceState(null, "", `#${hash}`);
    return location.href;
  }

  function restoreURL() {
    const params = new URLSearchParams(location.hash.slice(1));
    if (!params.has("memory")) return;
    try {
      setConfig(M.normalize(JSON.parse(params.get("memory"))));
    } catch {}
  }

  function renderBudget() {
    const c = current.config;
    $("state-total").textContent = bytes(current.stage.stateTotal);
    $("state-detail").textContent = `${count(
      current.stage.storedTotal
    )} resident parameter elements on this GPU: master, reduced gradient and optimizer state, independent of the lifecycle phase.`;
    $("budget-total").textContent = bytes(current.budget);
    $("budget-total").classList.toggle("over-capacity", current.budget > M.capacity);
    $("budget-verdict").textContent =
      current.budget > M.capacity
        ? `Exceeds the 288-GiB physical HBM by ${bytes(current.budget - M.capacity)}, before driver and context reservations.`
        : `${bytes(M.capacity - current.budget)} below the physical HBM in this planning scenario; the driver and CUDA context take part of that.`;
    $("budget-title").textContent = {
      forward: "First forward: planning budget",
      backward: "Schedule steady-state planning budget",
      optimizer: "After-reduction / optimizer planning budget",
    }[c.phase];
    const axis = 1.08 * Math.max(M.capacity, current.budget);
    $("budget-axis").textContent = `Linear bytes; axis ${bytes(axis)}`;
    $("budget-bar").replaceChildren();
    $("budget-legend").replaceChildren();
    for (const part of current.parts) {
      const segment = document.createElement("div");
      segment.className = "budget-part";
      segment.style.width = `${(part.value / axis) * 100}%`;
      segment.style.background = `var(--${colors[part.key]})`;
      segment.title = `${part.label}: ${bytes(part.value)}`;
      $("budget-bar").appendChild(segment);
      const legend = document.createElement("span");
      legend.innerHTML = `<i style="background:var(--${colors[part.key]})"></i>${esc(part.label)} <b>${bytes(part.value)}</b>`;
      $("budget-legend").appendChild(legend);
    }
    const marker = document.createElement("div");
    marker.className = "budget-limit";
    marker.style.left = `${(M.capacity / axis) * 100}%`;
    marker.title = "288 GiB physical HBM";
    $("budget-bar").appendChild(marker);
    $("memory-ledger").tBodies[0].innerHTML = current.parts
      .map(
        (part) =>
          `<tr><td><i class="ledger-dot" style="background:var(--${colors[part.key]})"></i>${esc(part.label)}</td><td>${bytes(
            part.value
          )}</td><td>${esc(part.detail)}</td></tr>`
      )
      .join("");
    $("memory-ledger").tBodies[0].insertAdjacentHTML(
      "beforeend",
      `<tr><td><strong>Planning subtotal</strong></td><td><strong>${bytes(current.budget)}</strong></td><td>Sum of the rows above.</td></tr>`
    );
  }

  function renderLayout() {
    const c = current.config;
    const owned = current.stage.owned.map(
      (chunk) =>
        `chunk ${chunk.index} (${layerRange(chunk)}${chunk.first ? ", embedding" : ""}${chunk.last ? ", head" : ""}${
          chunk.table ? ", PLE table" : ""
        })`
    );
    $("layout-note").textContent = `PP rank ${c.stage} of ${c.pp} runs ${owned.join(" and ")}. ${current.dp} DP ranks per stage; EP${c.ep} gives ${
      A.E / c.ep
    } experts per layer per GPU and expert FSDP ${current.edp}. ${count(current.globalTokens)} tokens per optimizer step; nominal bubble ${(
      current.bubble * 100
    ).toFixed(2)}%.`;
    $("stage-map").innerHTML = current.stages
      .map((stage) => {
        const budget = M.calculate({ ...c, stage: stage.rank }).budget;
        return `<button type="button" data-rank="${stage.rank}" aria-pressed="${stage.rank === c.stage}">PP ${stage.rank}<span>${bytes(budget)}${
          budget > M.capacity ? " (over)" : ""
        }</span></button>`;
      })
      .join("");
    for (const button of $("stage-map").querySelectorAll("button"))
      button.addEventListener("click", () => {
        controls.stage.value = button.dataset.rank;
        update();
      });
  }

  function tensorRows(kind) {
    const T = current.config.tokens;
    const w = current.workspaces;
    const moe = current.moe;
    const saved = "Saved until backward";
    const transient = "Regenerated per layer";
    const rows = [["Four-stream layer input", `[${T}, 4 x 2560]`, "BF16", T * A.C * 2, saved]];
    if (kind === "gdn" || kind === "ple") {
      if (kind === "ple")
        rows.push(
          ["PLE hashed row IDs", `[${T}, 16]`, "INT64", w.pleIds, "Prepared before the layer runs"],
          ["Gathered PLE rows E", `[${T}, 16, 160]`, "BF16", w.pleOutput, "Lookup output"],
          [
            "PLE request rows (capacity)",
            `[${current.config.table === "world" ? current.config.world : current.dp} x ${count(w.pleCapacity)}, 160]`,
            "BF16",
            w.pleRows,
            "Communication buffer",
          ]
        );
      rows.push(
        ["GDN projected Q/K/V + z, a, b", `[${T}, 16480]`, "BF16", w.gdnProjection, transient],
        ["GDN chunk states", `[${w.gdnChunks}, 48, 128, 128]`, "FP32", w.gdnChunkStates, transient]
      );
    } else if (kind === "qsa") {
      rows.push(
        ["Indexer block scores (one chunk)", `[1024, ${T / 4}, 4]`, "FP32", w.indexerScores, transient],
        ["Selected key indices", `[${T}, ${w.J}]`, "INT64", w.selected, transient],
        ["Teacher gathered keys (one chunk)", `[512, 2056, 2, 256]`, "BF16", w.teacherKeys, transient],
        ["Teacher logits (one chunk)", `[2, 512, 12, 2056]`, "FP32", w.teacherLogits, transient],
        ["Indexer KL target", `[${T}, 512]`, "FP32", T * Math.min(512, T / 4) * 4, saved]
      );
    }
    if (kind === "head") {
      rows.length = 0;
      rows.push(
        ["Four-stream state into the mixer", `[${T}, 4 x 2560]`, "BF16", T * A.C * 2, "Saved boundary of the last layer"],
        ["Final hidden state", `[${T}, 2560]`, "BF16", T * A.D * 2, saved],
        ["Full logits (never built)", `[${T}, 248320]`, "BF16", w.fullLogits, "Avoided by the chunked loss"]
      );
    } else {
      rows.push(
        ["MoE input (mandatory slot part)", `[${T}, 2560]`, "BF16", moe.minimumLayer, "Dist-MoE slot"],
        ["Dispatched rows x_gathered", `[${moe.R}, 2560]`, "BF16", moe.R * A.D * 2, `Slot, factor ${current.config.slot}`],
        ["Expert h1 = [gate | up]", `[${moe.R}, 1280]`, "BF16", moe.R * 2 * A.F * 2, `Slot, factor ${current.config.slot}`],
        ["Staged expert output h3", `[${moe.R}, 2560]`, "BF16", moe.R * A.D * 2, `Slot, factor ${current.config.slot}`]
      );
    }
    return rows;
  }

  function renderTensors() {
    const kind = $("tensor-layer").value;
    $("tensor-table").tBodies[0].innerHTML = tensorRows(kind)
      .map(
        ([name, shape, dtype, size, life]) =>
          `<tr><td>${esc(name)}</td><td><code>${esc(shape)}</code></td><td>${dtype}</td><td>${bytes(size)}</td><td>${esc(life)}</td></tr>`
      )
      .join("");
    const c = current.config;
    const groups =
      kind === "head"
        ? A.backGroups
        : A.layerGroups(A.layers[kind === "qsa" ? 3 : kind === "ple" ? A.ple.layer : 0]).filter(
            (group) => kind === "ple" || !group.id.startsWith("ple")
          );
    $("param-table").tBodies[0].innerHTML = groups
      .map((group) => {
        const owners = c.table === "world" ? c.world : current.dp;
        const stored =
          group.role === "routed" ? group.params / c.ep / current.edp : group.role === "table" ? group.params / owners : group.params / current.dp;
        const viewBytes = group.role === "table" ? 0 : 2 * (group.role === "routed" ? group.params / c.ep : group.params);
        const opt = { muon: "DistMuon", adamw: "AdamW", adam: "Adam (no WD)" }[group.opt];
        return `<tr><td>${esc(group.label)}</td><td><code>${esc(group.shape)}</code></td><td>${count(group.params)}</td><td>${opt}</td><td>${bytes(
          stored * 4
        )}</td><td>${group.role === "table" ? "none" : bytes(viewBytes)}</td></tr>`;
      })
      .join("");
  }

  function values(rows) {
    return `<dl>${rows.map(([label, value]) => `<div><dt>${esc(label)}</dt><dd>${value}</dd></div>`).join("")}</dl>`;
  }

  function renderWorkspaces() {
    const w = current.workspaces;
    const moe = current.moe;
    const T = current.config.tokens;
    $("gdn-workspace").innerHTML = values([
      ["Decode state / layer", bytes(w.gdnInference)],
      [`Training: ${w.gdnChunks} chunk states`, bytes(w.gdnChunkStates)],
      ["Ratio", `${w.gdnChunks}x`],
    ]);
    $("qsa-workspace").innerHTML = values([
      ["Selected indices / layer", bytes(w.selected)],
      ["Teacher gather, one 512-query chunk", bytes(w.teacherKeys + w.teacherLogits)],
      ["Saved KL target / layer", bytes(T * Math.min(512, T / 4) * 4)],
    ]);
    $("moe-workspace").innerHTML = values([
      ["Minimum / layer", bytes(moe.minimumLayer)],
      ["Balanced full save / layer", bytes(moe.fullLayer)],
      [`One ${current.layersPerSlot}-layer slot at ${current.config.slot}`, bytes(moe.slotBytes)],
      [`Scratch, ${count(moe.rows)} rows`, bytes(moe.scratchBytes)],
    ]);
    $("loss-workspace").innerHTML = values([
      ["Full BF16 logits", bytes(w.fullLogits)],
      ["FP32 logits for the loss", bytes(w.fullLogitsFP32)],
      ["Saved final hidden", bytes(T * A.D * 2)],
    ]);
  }

  function renderSchedule() {
    const c = current.config;
    const width = 1000;
    const left = 70;
    const right = 20;
    const plot = width - left - right;
    const rowHeight = 92;
    const height = 30 + rowHeight * current.stages.length;
    let markup = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Forward and backward actions and live chunk forwards on each pipeline rank">`;
    const peak = Math.max(1, ...current.stages.map((stage) => stage.timeline.peak));
    current.stages.forEach((stage, row) => {
      const top = 20 + row * rowHeight;
      const N = stage.timeline.actions.length;
      const x = (step) => left + (plot * step) / N;
      stage.timeline.actions.forEach((action, index) => {
        markup += `<rect x="${x(index).toFixed(2)}" y="${top}" width="${(plot / N + 0.1).toFixed(2)}" height="14" class="action-${
          action.kind
        } local-${action.chunk % 2}"><title>PP ${stage.rank}: ${action.kind} chunk ${stage.owned[action.chunk].index}, microbatch ${
          action.mb + 1
        }</title></rect>`;
      });
      const base = top + 70;
      const y = (value) => base - (46 * value) / peak;
      const path = stage.timeline.live.map((live, index) => `${index ? "L" : "M"}${x(index).toFixed(2)},${y(live).toFixed(2)}`).join(" ");
      markup += `<path d="${path} L${x(N)},${base} L${left},${base} Z" class="schedule-area"/><path d="${path}" class="schedule-curve"/>`;
      markup += `<text x="4" y="${top + 11}" class="schedule-label">PP ${stage.rank}</text><text x="4" y="${
        y(stage.timeline.peak) + 4
      }" class="schedule-label">${stage.timeline.peak} live</text><line x1="${left}" x2="${
        width - right
      }" y1="${base}" y2="${base}" class="schedule-grid"/>`;
    });
    markup += "</svg>";
    $("schedule-chart").innerHTML = markup;
    $("schedule-axis").textContent = `${c.mbs} microbatches, PP${c.pp}, VPP${c.vp}`;
    $("schedule-legend").innerHTML = current.stages
      .map(
        (stage) =>
          `<span>PP ${stage.rank}: peak <b>${stage.timeline.peak}</b> unfinished chunk forwards, ${bytes(
            stage.savedPeak + stage.slotTotal
          )} saved inputs + slots</span>`
      )
      .join("");
    const base = current.config;
    $("schedule-compare").tBodies[0].innerHTML = [1, 2, 3, 4]
      .filter((vp) => valid({ ...base, vp, stage: 0 }))
      .map((vp) => {
        const r = M.calculate({ ...base, vp, stage: 0 });
        const last = r.stages.at(-1);
        return `<tr${vp === base.vp ? ' class="selected-row"' : ""}><td>${vp === 1 ? "1F1B" : `Interleaved, VPP${vp}`}</td><td>${
          r.layersPerSlot
        }</td><td>${r.stages[0].timeline.peak} / ${last.timeline.peak}</td><td>${bytes(r.stages[0].savedPeak)}</td><td>${bytes(
          r.stages[0].slotTotal
        )}</td><td>${(r.bubble * 100).toFixed(2)}%</td></tr>`;
      })
      .join("");
  }

  // The fitting walkthrough follows the anchor's most loaded GPU.
  function deckSteps() {
    const anchor = M.calculate();
    const s = anchor.stages[0];
    const bill = M.stateBill();
    const GiB = M.GiB;
    const even = bill.total / anchor.config.world;
    const p = (key) => anchor.parts.find((part) => part.key === key).value;
    return [
      {
        title: "The logical bill: 2.13 TiB of FP32 state",
        text: "Every trainable parameter has an FP32 master and gradient; Muon adds one momentum, Adam and AdamW add two moments. No GPU holds this, but it is what 16 GPUs must divide.",
        bars: [
          ["DistMuon state", bill.muon],
          ["PLE table Adam state", bill.adam],
          ["AdamW state", bill.adamw],
        ],
      },
      {
        title: "An even 16-way split would be 136 GiB",
        text: "If every byte divided evenly over 16 GPUs, persistent state would use under half of the HBM. The layout below never achieves this average.",
        bars: [["Average update state", even]],
      },
      {
        title: "PP2 halves the decoder, but the table stays on rank 0",
        text: "Rank 0 owns layers 1-12 and 25-36, the embedding, and all 320M table rows over its 8 DP ranks. Its update state is 95 GiB larger than rank 1's.",
        bars: [
          ["Table rows (Adam, 16 B)", s.stored.table * 16],
          ["Experts, 24 layers x 64 (Muon)", s.stored.routed * 12],
          ["Dense shards", s.stateTotal - s.stored.table * 16 - s.stored.routed * 12],
        ],
      },
      {
        title: "PP keeps dense gradients unsharded and every view gathered",
        text: "Inside one schedule, PipelineStage disables FSDP gradient sync and resharding, so dense gradients accumulate at full size and BF16 views of both local chunks stay live.",
        bars: [
          ["Persistent state, reduced gradients", s.stateTotal],
          ["Extra for deferred dense gradients", s.deferredGrad - s.reducedGrad],
          ["BF16 compute views", s.viewsStage],
        ],
      },
      {
        title: "Five unfinished chunk forwards",
        text: "Interleaved warmup leaves 5 chunk forwards open on rank 0. Each keeps 12 four-stream layer inputs and 3 KL targets, and Dist-MoE allocates one 12-layer slot per open forward.",
        bars: [
          ["Saved boundaries + KL targets", s.savedPeak],
          [`Dist-MoE slots (5 x ${(anchor.moe.slotBytes / GiB).toFixed(1)} GiB)`, s.slotTotal],
        ],
      },
      {
        title: "Scratch, one layer's workspace, and an allowance",
        text: "Dist-MoE reserves scratch for 4x balanced routing. One layer's recompute workspace and an 8-GiB allowance for NCCL, FSDP buffers and the allocator complete the planning budget.",
        bars: [
          ["Dist-MoE scratch", p("scratch")],
          ["Largest layer workspace", p("working")],
          ["Workspace allowance", p("reserve")],
        ],
      },
      {
        title: `Rank 0 plans ${(anchor.budget / GiB).toFixed(0)} GiB; rank 1 plans ${(M.calculate({ stage: 1 }).budget / GiB).toFixed(0)} GiB`,
        text: "The anchor's rank-0 budget exceeds the 288-GiB device before driver reservations, while rank 1 has room to spare. The imbalance comes from the table, plus the deeper warmup of the first pipeline rank.",
        bars: [
          ["Rank 0 planning budget", anchor.budget],
          ["Rank 1 planning budget", M.calculate({ stage: 1 }).budget],
          ["Physical HBM", M.capacity],
        ],
        total: `Rank 0 exceeds the physical HBM by ${bytes(anchor.budget - M.capacity)}.`,
      },
    ];
  }

  function renderDeck() {
    const steps = deckSteps();
    view.deck = Math.max(0, Math.min(view.deck, steps.length - 1));
    const step = steps[view.deck];
    $("deck-position").textContent = `Step ${view.deck + 1} of ${steps.length}`;
    $("deck-prev").disabled = view.deck === 0;
    $("deck-next").disabled = view.deck === steps.length - 1;
    $("deck-title").textContent = step.title;
    $("deck-description").textContent = step.text;
    // Logarithmic axis from 1 MiB to 64 TiB, matching the axis labels.
    const lo = Math.log2(M.MiB);
    const hi = Math.log2(64 * 2 ** 40);
    $("deck-bars").innerHTML = step.bars
      .map(([label, value], index) => {
        const width = Math.max(0, Math.min(100, ((Math.log2(Math.max(value, M.MiB)) - lo) / (hi - lo)) * 100));
        const color = ["master", "adam", "gradient", "compute"][index % 4];
        return `<div class="log-row"><span>${esc(
          label
        )}</span><div class="log-track"><div class="log-fill" style="width:${width}%;background:var(--${color})"></div></div><span class="log-value">${bytes(
          value
        )}</span></div>`;
      })
      .join("");
    $("deck-total").textContent = step.total ?? `Sum of the bars: ${bytes(sum(step.bars, ([, value]) => value))}`;
  }

  function renderLevers() {
    const levers = [
      ["Anchor recipe", {}, "Reference point."],
      ["Dist-MoE slot factor 0", { slot: 0 }, "Saves only MoE inputs; every expert intermediate is recomputed in backward."],
      ["Slot factor 0.25", { slot: 0.25 }, "Half the saved expert intermediates of the anchor."],
      ["8,192 tokens x 32 microbatches", { tokens: 8192, mbs: 32 }, "Same global batch; twice as many microbatches, each with half-height GEMMs."],
      ["1F1B instead of interleaved (VPP1)", { vp: 1 }, "Fewer open forwards but 24-layer slots, and twice the nominal bubble."],
      ["Reduced gradients during the schedule", { grad: "reduced" }, "A reduce-scatter per microbatch instead of one per schedule."],
      ["Table rows over all 16 GPUs", { table: "world" }, "Alternative design: rank-1 GPUs would serve rows for a layer they do not run."],
      ["Slot factor 0 + table over all 16 GPUs", { slot: 0, table: "world" }, "Both levers together."],
    ];
    $("lever-table").tBodies[0].innerHTML = levers
      .map(([label, change, trade]) => {
        const rank0 = M.calculate({ ...change, stage: 0 }).budget;
        const lastConfig = M.normalize(change);
        const last = M.calculate({ ...change, stage: lastConfig.pp - 1 }).budget;
        const flag = (value) => `${bytes(value)}${value > M.capacity ? " (over)" : ""}`;
        return `<tr><td>${esc(label)}</td><td>${flag(rank0)}</td><td>${flag(last)}</td><td>${esc(trade)}</td></tr>`;
      })
      .join("");
  }

  function update() {
    constrain();
    try {
      current = M.calculate(readConfig());
    } catch (error) {
      $("config-warning").hidden = false;
      $("config-warning").textContent = error.message;
      return;
    }
    renderLayout();
    renderBudget();
    renderTensors();
    renderWorkspaces();
    renderSchedule();
  }

  for (const control of Object.values(controls)) control.addEventListener("change", update);
  $("tensor-layer").addEventListener("change", renderTensors);
  $("memory-reset").addEventListener("click", () => {
    setConfig(M.defaults);
    update();
  });
  $("memory-share").addEventListener("click", async () => {
    const url = saveURL();
    try {
      await navigator.clipboard.writeText(url);
      $("share-status").textContent = "Link copied";
    } catch {
      $("share-status").textContent = "Link in the address bar";
    }
  });
  $("deck-prev").addEventListener("click", () => {
    view.deck--;
    renderDeck();
  });
  $("deck-next").addEventListener("click", () => {
    view.deck++;
    renderDeck();
  });
  $("deck-apply").addEventListener("click", () => {
    setConfig(M.defaults);
    update();
    $("lab").scrollIntoView();
  });

  const themeButton = $("theme-toggle");
  let savedTheme;
  try {
    savedTheme = localStorage.getItem("qwen-flash-next-study-theme");
  } catch {}
  document.documentElement.classList.toggle("dark", savedTheme === "dark" || (!savedTheme && matchMedia("(prefers-color-scheme: dark)").matches));
  const themeLabel = () => {
    themeButton.textContent = document.documentElement.classList.contains("dark") ? "Light" : "Dark";
  };
  themeButton.addEventListener("click", () => {
    document.documentElement.classList.toggle("dark");
    try {
      localStorage.setItem("qwen-flash-next-study-theme", document.documentElement.classList.contains("dark") ? "dark" : "light");
    } catch {}
    themeLabel();
  });
  themeLabel();

  setConfig(M.defaults);
  restoreURL();
  update();
  renderDeck();
  renderLevers();
})();
