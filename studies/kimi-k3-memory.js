(() => {
  const A = window.KimiK3Model;
  const M = window.KimiK3Memory;
  const keys = Object.keys(M.defaults);
  const numeric = new Set(keys.filter((key) => typeof M.defaults[key] === "number"));
  const controls = Object.fromEntries(keys.map((key) => [key, document.getElementById(`mem-${key}`)]));
  const colors = {
    master: "master",
    momentum: "momentum",
    adam: "adam",
    gradient: "gradient",
    compute: "compute",
    activation: "activation",
    working: "working",
    residual: "residual",
    reserve: "reserve",
  };
  const view = { layer: 36, node: "qkv", lens: "activation", marks: {}, deck: 0, batch: 14, step: null };
  const $ = (id) => document.getElementById(id);
  const sum = (items, get) => items.reduce((total, item, index) => total + get(item, index), 0);
  const count = (value) =>
    value >= 1e12
      ? `${(value / 1e12).toFixed(3)}T`
      : value >= 1e9
        ? `${(value / 1e9).toFixed(3)}B`
        : value >= 1e6
          ? `${(value / 1e6).toFixed(2)}M`
          : value.toLocaleString();
  const bytes = (value) =>
    value === 0
      ? "0 B"
      : value >= 2 ** 40
        ? `${(value / 2 ** 40).toFixed(2)} TiB`
        : value >= M.GiB
          ? `${(value / M.GiB).toFixed(2)} GiB`
          : value >= 2 ** 20
            ? `${(value / 2 ** 20).toFixed(2)} MiB`
            : value >= 1024
              ? `${(value / 1024).toFixed(2)} KiB`
              : `${value.toFixed(0)} B`;
  const esc = (value) => String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
  const layerRange = (chunk) => (chunk.layers.length ? `L${chunk.layers[0].index + 1}-${chunk.layers.at(-1).index + 1}` : "head only");
  let current;
  let catalog;
  let nodes;
  let graphEdges = [];

  function readConfig() {
    return Object.fromEntries(keys.map((key) => [key, numeric.has(key) ? Number(controls[key].value) : controls[key].value]));
  }

  function constrain() {
    const notices = [];
    const pp = Number(controls.pp.value);
    const world = Number(controls.world.value);
    const oldRank = Math.min(Number(controls.stage.value || M.defaults.stage), pp - 1);
    controls.stage.replaceChildren(...Array.from({ length: pp }, (_, rank) => new Option(`PP ${rank}`, rank)));
    controls.stage.value = oldRank;
    for (const key of ["cp", "ep", "vp", "active"]) {
      const options = [...controls[key].options];
      for (const option of options) {
        const value = Number(option.value);
        option.disabled =
          key === "cp"
            ? (world / pp) % value !== 0
            : key === "ep"
              ? (world / pp) % value !== 0 || A.experts % value !== 0
              : key === "vp"
                ? value * pp > A.layers.length
                : value > Number(controls.vp.value);
      }
      if (controls[key].selectedOptions[0]?.disabled) {
        const valid = options.filter((option) => !option.disabled && Number(option.value) <= Number(controls[key].value));
        controls[key].value = valid.at(-1)?.value || options.find((option) => !option.disabled).value;
        notices.push(`${key.toUpperCase()} adjusted to ${controls[key].value} to respect the mesh/chunk constraints.`);
      }
    }
    controls.sched.options[0].disabled = Number(controls.vp.value) > 1;
    if (controls.sched.options[0].disabled && controls.sched.value === "1f1b") {
      controls.sched.value = "interleaved";
      notices.push("VP > 1 requires an interleaved schedule; ordinary 1F1B has one local stage.");
    }
    controls.active.disabled = controls.unit.value !== "chunk";
    $("config-warning").hidden = notices.length === 0;
    $("config-warning").textContent = notices.join(" ");
  }

  function setConfig(config) {
    for (const key of keys) if (key !== "stage") controls[key].value = config[key];
    controls.stage.replaceChildren(...Array.from({ length: config.pp }, (_, rank) => new Option(`PP ${rank}`, rank)));
    controls.stage.value = config.stage;
    constrain();
  }

  function saveURL() {
    const data = {
      c: readConfig(),
      v: { layer: view.layer, node: view.node, lens: view.lens, marks: view.marks, deck: view.deck, batch: view.batch, step: view.step },
    };
    const hash = new URLSearchParams({ memory: JSON.stringify(data) }).toString();
    history.replaceState(null, "", `${location.pathname}${location.search}#${hash}`);
  }

  function restoreURL() {
    const params = new URLSearchParams(location.hash.slice(1));
    if (!params.has("memory")) {
      const old = params.get("d:fitdeck");
      if (old) {
        try {
          const index = JSON.parse(old).i;
          if (Number.isInteger(index) && index >= 0 && index < deckSteps.length) {
            view.deck = index;
            return true;
          }
        } catch {}
      }
      return false;
    }
    try {
      const data = JSON.parse(params.get("memory"));
      const config = M.normalize(data.c);
      for (const key of keys)
        if (key !== "stage" && ![...controls[key].options].some((option) => option.value === String(config[key])))
          throw new Error("Unknown control value");
      setConfig(config);
      const v = data.v || {};
      if (Number.isInteger(v.layer) && v.layer >= 0 && v.layer < A.layers.length) view.layer = v.layer;
      if ([...$("tensor-lens").options].some((option) => option.value === v.lens)) view.lens = v.lens;
      if (typeof v.node === "string" && /^[a-z_]+$/.test(v.node)) view.node = v.node;
      if (Number.isInteger(v.deck) && v.deck >= 0 && v.deck < deckSteps.length) view.deck = v.deck;
      if ([1, 14, 28, 56].includes(v.batch)) view.batch = v.batch;
      view.step = Number.isSafeInteger(v.step) && v.step >= 0 ? v.step : null;
      const allowed = new Set(
        [0, 1, 3].flatMap((index) =>
          M.tensorCatalog(A.layers[index], config)
            .tensors.filter((row) => !row.alias)
            .map((row) => row.id)
        )
      );
      view.marks = Object.fromEntries(Object.entries(v.marks || {}).filter(([id, enabled]) => allowed.has(id) && typeof enabled === "boolean"));
      return true;
    } catch {
      Object.assign(view, { layer: 36, node: "qkv", lens: "activation", marks: {}, deck: 0, step: null, batch: 14 });
      setConfig(M.defaults);
      $("config-warning").hidden = false;
      $("config-warning").textContent = "This analysis link is invalid or uses unsupported controls. Showing the baseline instead.";
      return true;
    }
  }

  function renderBudget() {
    const c = current.config;
    $("state-total").textContent = bytes(current.stateTotal);
    $("state-detail").textContent = `${count(
      current.stored
    )} resident parameter elements on this GPU. This is the complete post-reduction state tally, independent of the selected lifecycle scenario.`;
    $("budget-total").textContent = bytes(current.budget);
    $("budget-total").classList.toggle("over-capacity", current.budget > M.capacity);
    $("budget-verdict").textContent =
      current.budget > M.capacity
        ? `Exceeds reported HBM by ${bytes(current.budget - M.capacity)} before additional unmodeled effects.`
        : `${bytes(M.capacity - current.budget)} below reported HBM in this planning scenario; not a fit guarantee.`;
    $("budget-title").textContent = {
      forward: "Forward after zero_grad: planning scenario",
      backward: "Backward-accumulation planning scenario",
      optimizer: "After-reduction / optimizer planning scenario",
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
    marker.title = "184.31 GiB reported device capacity";
    $("budget-bar").appendChild(marker);
    $("memory-ledger").tBodies[0].innerHTML = current.parts
      .map(
        (part) =>
          `<tr><td><i class="ledger-dot" style="background:var(--${colors[part.key]})"></i>${esc(part.label)}</td><td>${bytes(
            part.value
          )}</td><td>${esc(part.detail)}</td></tr>`
      )
      .join("");
    $("ownership-note").textContent = `${c.world} GPUs / PP${c.pp} = ${current.stageRanks} ranks per physical pipeline stage = DP${current.dp} x CP${
      c.cp
    }. Dense FSDP${current.denseFsdp}; expert FSDP${current.expertFsdp}; ${A.experts / c.ep} EP-local experts. Local T = ${
      c.rows
    } x ${c.seq.toLocaleString()} / CP${
      c.cp
    } = ${current.tokens.toLocaleString()}. Global batch: ${current.globalRows.toLocaleString()} sequences, ${current.globalTokens.toLocaleString()} tokens/update.`;
    $("layout-note").textContent = `${
      c.vp === 1
        ? "VP1: reference ceil(93 / PP) contiguous split."
        : "Virtual chunks: automatic 95-effective-layer split (93 decoder layers + embedding/head weights of one), cyclic rank assignment."
    } Selected rank owns ${current.owned.map((chunk) => `C${chunk.index}: ${layerRange(chunk)}`).join("; ")}. ${
      c.vision === "off"
        ? "Vision parameters are excluded; no vision activation budget is included."
        : "Canonical parameter inventory includes vision at the front, assigned to the AdamW remainder here; image activations are not modeled."
    }`;
    $("stage-map").replaceChildren();
    for (const stage of current.stages) {
      const button = document.createElement("button");
      button.type = "button";
      button.setAttribute("aria-pressed", String(stage.rank === c.stage));
      button.innerHTML = `PP ${stage.rank}<span>${stage.chunks.map(layerRange).join(" / ")}</span>`;
      button.addEventListener("click", () => {
        controls.stage.value = stage.rank;
        view.layer = stage.chunks.flatMap((chunk) => chunk.layers)[0]?.index ?? 92;
        view.step = null;
        render();
      });
      $("stage-map").appendChild(button);
    }
  }

  function buildNodes(layer) {
    const result = [];
    const node = (id, title, groups = [id], params = [], note = "") => {
      const item = {
        id,
        title,
        rows: catalog.tensors.filter((row) => groups.includes(row.group) || (id === "input" && row.id === "stream")),
        params: M.parameterGroups(layer, current.config.ep).filter((group) => params.includes(group.id)),
        note,
      };
      result.push(item);
      return id;
    };
    const input = node(
      "input",
      "Boundary stream + norm inventory",
      ["input"],
      ["stream_norms"],
      "The norm inventory groups both stream RMSNorms and AttnRes projections/norms. Their weights are counted here once, not once per tensor view."
    );
    const norm = node("norm", layer.index === 0 ? "Attention RMSNorm" : "AttnRes mix + attention RMSNorm");
    let branches;
    if (layer.mla) {
      branches = [
        [node("qa", "Query compression + RMSNorm", ["qa"], ["qa", "qnorm"]), node("qb", "Query expansion", ["qb"], ["qb"])],
        [
          node("kva", "KV compression + RMSNorm", ["kva"], ["kva", "kvnorm"]),
          node(
            "kvb",
            "KV expansion + full K",
            ["kvb"],
            ["kvb"],
            "The V slice retains its K/V backing buffer. expand() aliases the shared key; cat() allocates full K."
          ),
        ],
        [node("gate", "Output-gate projection", ["gate"], ["output_gate"])],
      ];
    } else {
      branches = [
        [
          node("qkv", "Q / K / V projections", ["qkv"], ["q", "k", "v"]),
          node("conv", "Concatenate + short convolution", ["conv"], ["conv"]),
          node("qknorm", "Head views + Q/K L2Norm"),
        ],
        [
          node("forget", "Forget compression / expansion", ["forget"], ["forget_a", "forget_b"]),
          node("decay", "Bounded FP32 log-decay", ["decay"], ["decay"]),
        ],
        [node("beta", "FP32 write-strength sigmoid", ["beta"], ["beta"]), node("gate", "Output-gate projection", ["gate"], ["output_gate"])],
      ];
    }
    const tail = [
      node(
        "core",
        layer.mla ? "Causal varlen attention" : "Chunk KDA",
        ["core"],
        [],
        layer.mla
          ? "Q/K/V, output, and row statistics replace a materialized attention-probability matrix. Under head-sharded Ulysses, [T_local, H] exchanges into [T_global, H/CP] with the same element count."
          : "Training needs token tensors and kernel-specific chunk state. A fixed-size inference state is not a constant training-memory bound."
      ),
      node("gated", layer.mla ? "Sigmoid gate x attention output" : "Per-head RMSNorm + output gate", ["gated"], layer.mla ? [] : ["output_norm"]),
      node("output", "Attention output projection", ["output"], ["output_proj"]),
      node("ffn_norm", "AttnRes mixture + FFN RMSNorm"),
    ];
    let ffnBranches;
    let finish;
    if (layer.index === 0) {
      ffnBranches = [[node("dense", "Dense gate + up projections", ["dense"], ["dense_gate_up"]), node("dense_act", "Dense SiTU-GLU")]];
      finish = [node("finish", "Dense down + residual output", ["finish"], ["dense_down"])];
    } else {
      ffnBranches = [
        [
          node(
            "router",
            "Router + top-16 selection",
            ["router"],
            ["router"],
            "Router input is the full 7168-channel stream. Routing probabilities and indices are separate metadata, not trainable expert state."
          ),
        ],
        [
          node("latent", "Compress D -> Z", ["latent"], ["latent_down"]),
          node(
            "dispatch",
            "EP dispatch / received rows",
            ["dispatch"],
            [],
            "Balanced received assignment count is 16T, not 16T/EP. Increasing EP leaves this expected volume unchanged; imbalance can increase it."
          ),
          node("experts", "Routed gate/up GEMM", ["experts"], ["routed_gate_up"]),
          node("expert_act", "Routed SiTU-GLU"),
          node("expert_down", "Routed down GEMM", ["expert_down"], ["routed_down"]),
        ],
        [
          node("shared", "Two shared experts: gate/up", ["shared"], ["shared_gate_up"]),
          node("shared_act", "Shared SiTU-GLU"),
          node("shared_out", "Shared down GEMM", ["shared_out"], ["shared_down"]),
        ],
      ];
      finish = [
        node("combine", "EP combine + latent RMSNorm", ["combine"], ["latent_norm"]),
        node("expand", "Expand Z -> D", ["expand"], ["latent_up"]),
        node("finish", "Shared + routed + residual output"),
      ];
    }
    const edges = [[input, norm]];
    if (layer.mla) edges.push([norm, "qa"], ["qa", "qb"], [norm, "kva"], ["kva", "kvb"], [norm, "gate"], ["qb", "core"], ["kvb", "core"]);
    else
      edges.push(
        [norm, "qkv"],
        ["qkv", "conv"],
        ["conv", "qknorm"],
        [norm, "forget"],
        ["forget", "decay"],
        [norm, "beta"],
        [norm, "gate"],
        ["qknorm", "core"],
        ["decay", "core"],
        ["beta", "core"]
      );
    edges.push(["core", "gated"], ["gate", "gated"], ["gated", "output"], ["output", "ffn_norm"]);
    if (layer.index === 0) edges.push(["ffn_norm", "dense"], ["dense", "dense_act"], ["dense_act", "finish"]);
    else
      edges.push(
        ["ffn_norm", "router"],
        ["ffn_norm", "latent"],
        ["ffn_norm", "shared"],
        ["router", "dispatch"],
        ["router", "combine"],
        ["latent", "dispatch"],
        ["dispatch", "experts"],
        ["experts", "expert_act"],
        ["expert_act", "expert_down"],
        ["expert_down", "combine"],
        ["combine", "expand"],
        ["expand", "finish"],
        ["shared", "shared_act"],
        ["shared_act", "shared_out"],
        ["shared_out", "finish"]
      );
    return { result, input, norm, branches, tail, ffnBranches, finish, edges };
  }

  function nodeValue(node) {
    const tensorBytes = sum(node.rows, (row) => row.bytes);
    const savedBytes = sum(node.rows, (row) => (M.isSaved(row, current.config, view.marks) ? row.bytes : 0));
    const copies = node.params.map((group) => M.parameterCopies(group, current.config));
    if (view.lens === "activation") return tensorBytes;
    if (view.lens === "saved") return savedBytes;
    if (view.lens === "combined") return tensorBytes + sum(copies, (copy) => copy.master + copy.optimizer + copy.gradient + copy.compute);
    return sum(copies, (copy) => copy[view.lens]);
  }

  function renderAnatomy() {
    const layer = A.layers[view.layer];
    catalog = M.tensorCatalog(layer, current.config);
    const graph = buildNodes(layer);
    nodes = graph.result;
    graphEdges = graph.edges;
    if (!nodes.some((node) => node.id === view.node)) view.node = layer.mla ? "kvb" : "qkv";
    $("tensor-layer").replaceChildren(
      ...A.layers.map(
        (item) =>
          new Option(
            `L${item.index + 1}: ${item.mla ? "MLA" : "KDA"} + ${item.index === 0 ? "dense FFN" : "latent MoE"}${
              current.layers.includes(item) ? " (local rank)" : ""
            }`,
            item.index
          )
      )
    );
    $("tensor-layer").value = view.layer;
    $("tensor-lens").value = view.lens;
    $("tensor-shape-note").textContent =
      `T=${catalog.T.toLocaleString()} CP-local tokens; D=7168; H=96; K=128; C=H x K=12288; Z=3584; F=3072; E=896; topK=16. Received R=${catalog.R.toLocaleString()} (${
        current.config.imbalance
      }x balanced); ${catalog.localExperts} local experts average ${catalog.rowsPerExpert.toFixed(2)} routed rows each. ${
        current.layers.includes(layer)
          ? "This layer is on the selected physical PP rank."
          : "This representative layer is outside the selected rank; its table does not change that rank's ownership."
      }`;
    const total = sum(catalog.tensors, (row) => row.bytes);
    const saved = M.savedForLayer(layer, current.config, view.marks);
    const aliases = catalog.tensors.filter((row) => row.alias);
    $("tensor-summary").innerHTML = `<div><strong>${bytes(
      total
    )}</strong><span>sum of unique forward candidates; not a peak</span></div><div><strong>${bytes(
      saved
    )}</strong><span>selected saved candidates / layer / microbatch</span></div><div><strong>${aliases.length} views</strong><span>${bytes(
      sum(aliases, (row) => row.logicalBytes)
    )} logical view sizes; zero additional storage</span></div>`;
    $("anatomy-title").textContent = `L${view.layer + 1}: ${layer.mla ? "MLA" : "KDA"} -> ${layer.index === 0 ? "dense FFN" : "latent MoE"}`;
    $("anatomy-lens").textContent = $("tensor-lens").selectedOptions[0].textContent;
    const max = Math.max(1, ...nodes.map(nodeValue));
    const diagram = $("memory-anatomy");
    diagram.replaceChildren();
    const box = (id) => {
      const node = nodes.find((item) => item.id === id);
      const button = document.createElement("button");
      button.type = "button";
      button.className = "memory-node";
      button.classList.toggle(
        "has-saved",
        node.rows.some((row) => M.isSaved(row, current.config, view.marks))
      );
      button.setAttribute("aria-pressed", String(view.node === id));
      button.dataset.node = id;
      button.style.setProperty("--charge", `${(100 * nodeValue(node)) / max}%`);
      const main = node.rows.find((row) => !row.alias);
      button.innerHTML = `<strong>${esc(node.title)}</strong><code>${esc(
        main?.logical || "weights / local operation"
      )}</code><span class="node-bytes">${bytes(nodeValue(node))}</span>${
        node.rows.some((row) => row.alias) ? '<small class="alias-tag">includes zero-storage views</small>' : ""
      }`;
      button.addEventListener("click", () => {
        view.node = id;
        for (const item of diagram.querySelectorAll("[data-node]")) item.setAttribute("aria-pressed", String(item.dataset.node === id));
        renderInspector();
        saveURL();
      });
      return button;
    };
    const chain = (ids, container) => ids.forEach((id) => container.appendChild(box(id)));
    const branches = (items, titles) => {
      const split = document.createElement("div");
      split.className = `memory-branches ${items.length === 1 ? "one-branch" : ""}`;
      items.forEach((ids, index) => {
        const lane = document.createElement("div");
        lane.className = "memory-lane";
        const heading = document.createElement("div");
        heading.className = "lane-title";
        heading.textContent = titles[index];
        lane.appendChild(heading);
        chain(ids, lane);
        split.appendChild(lane);
      });
      diagram.appendChild(split);
    };
    chain([graph.input, graph.norm], diagram);
    branches(
      graph.branches,
      layer.mla ? ["Query path", "Key / value path", "Output gate"] : ["Q / K / V path", "Channel forgetting", "Write + output gates"]
    );
    chain(graph.tail, diagram);
    branches(
      graph.ffnBranches,
      layer.index === 0 ? ["First-layer dense FFN"] : ["Full-stream routing", "Compressed routed experts", "Shared experts"]
    );
    chain(graph.finish, diagram);
    drawConnections();
    renderTensorTables();
    renderInspector();
    renderWorkspaces();
  }

  function drawConnections() {
    const diagram = $("memory-anatomy");
    const bounds = diagram.getBoundingClientRect();
    if (!bounds.width) return;
    let svg = diagram.querySelector(".memory-wires");
    if (!svg) {
      svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.classList.add("memory-wires");
      svg.setAttribute("aria-hidden", "true");
      diagram.prepend(svg);
    }
    const padding = 8;
    svg.setAttribute("viewBox", `0 0 ${bounds.width + 2 * padding} ${bounds.height + 2 * padding}`);
    svg.style.width = `${bounds.width + 2 * padding}px`;
    svg.style.height = `${bounds.height + 2 * padding}px`;
    let markup =
      '<defs><marker id="memory-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z"/></marker></defs>';
    const rect = (id) => {
      const box = diagram.querySelector(`[data-node="${id}"]`).getBoundingClientRect();
      return {
        left: box.left - bounds.left + padding,
        right: box.right - bounds.left + padding,
        top: box.top - bounds.top + padding,
        bottom: box.bottom - bounds.top + padding,
        width: box.width,
        cx: box.left - bounds.left + padding + box.width / 2,
        cy: box.top - bounds.top + padding + box.height / 2,
      };
    };
    for (const [from, to] of graphEdges) {
      const a = rect(from);
      const b = rect(to);
      let path;
      let bypass = false;
      if (from === "gate" || from === "shared_out" || (from === "beta" && to === "core")) {
        const lane = bounds.width + padding + (from === "beta" ? 2 : 6);
        path = `M${a.right},${a.cy} H${lane} V${b.cy} H${b.right}`;
        bypass = true;
      } else if (from === "router") {
        const lane = to === "combine" ? padding - 6 : a.right + 6;
        path = `M${to === "combine" ? a.left : a.right},${a.cy} H${lane} V${b.cy} H${b.left}`;
        bypass = true;
      } else if (from === "norm" && to === "gate" && !A.layers[view.layer].mla) {
        const lane = bounds.width + padding + 6;
        path = `M${a.right},${a.cy} H${lane} V${b.cy} H${b.right}`;
        bypass = true;
      } else {
        const ax = a.width > b.width * 1.5 ? b.cx : a.cx;
        const bx = b.width > a.width * 1.5 ? a.cx : b.cx;
        const mid = (a.bottom + b.top) / 2;
        path = `M${ax},${a.bottom} V${mid} H${bx} V${b.top}`;
      }
      markup += `<path d="${path}" class="memory-wire ${bypass ? "bypass-wire" : ""}" marker-end="url(#memory-arrow)"><title>${esc(
        nodes.find((node) => node.id === from).title
      )} to ${esc(nodes.find((node) => node.id === to).title)}</title></path>`;
    }
    svg.innerHTML = markup;
  }

  function renderInspector() {
    const node = nodes.find((item) => item.id === view.node);
    const c = current.config;
    const copies = node.params.map((group) => M.parameterCopies(group, c));
    const tensorTotal = sum(node.rows, (row) => row.bytes);
    const savedTotal = sum(node.rows, (row) => (M.isSaved(row, c, view.marks) ? row.bytes : 0));
    $("tensor-inspector").innerHTML = `<span class="eyebrow">Selected computation</span><h3>${esc(node.title)}</h3><p>${esc(
      node.note ||
        "Inspect the unique tensor allocations and training-state copies below. Backward saves are explicit candidates, not an inferred autograd trace."
    )}</p><div class="inspector-totals"><div><b>${bytes(tensorTotal)}</b><small>forward candidates</small></div><div><b>${bytes(
      savedTotal
    )}</b><small>selected saved storage</small></div><div><b>${bytes(
      sum(copies, (copy) => copy.master)
    )}</b><small>FP32 master shards</small></div><div><b>${bytes(
      sum(copies, (copy) => copy.compute)
    )}</b><small>BF16 gathered weights</small></div></div><h4>Output tensors / views</h4>${
      node.rows.length
        ? `<ul class="inspector-tensors">${node.rows
            .map(
              (row) =>
                `<li><strong>${esc(row.label)}</strong><code>${esc(row.logical)} = [${row.shape
                  .map((size) => size.toLocaleString())
                  .join(", ")}]</code><span>${row.dtype} / ${bytes(row.bytes)} ${
                  row.alias
                    ? `<b class="alias-text">view of ${esc(row.alias)}</b>`
                    : M.isSaved(row, c, view.marks)
                      ? '<b class="saved-text">selected saved</b>'
                      : '<b class="recompute-text">transient / recomputed</b>'
                }</span>${row.note ? `<small>${esc(row.note)}</small>` : ""}</li>`
            )
            .join("")}</ul>`
        : "<p>No separate output allocation is assigned here.</p>"
    }<h4>Parameter groups</h4>${
      node.params.length
        ? `<ul class="inspector-weights">${node.params
            .map(
              (group, index) =>
                `<li><b>${esc(group.label)}</b><code>${esc(group.shape)}</code><small>${count(group.params)} global parameters; FSDP${
                  copies[index].degree
                }. Master ${bytes(copies[index].master)}; optimizer ${bytes(copies[index].optimizer)}; gradient ${bytes(
                  copies[index].gradient
                )}.</small></li>`
            )
            .join("")}</ul>`
        : "<p>No trainable weight matrix in this operation.</p>"
    }<p class="note">BF16 gathered view and FP32 resident shard are different copies. The per-layer table is below; the whole-rank budget is above.</p>`;
    for (const row of $("activation-ledger").querySelectorAll("tbody tr"))
      row.classList.toggle("selected-tensor-row", row.dataset.group && node.rows.some((item) => item.group === row.dataset.group));
  }

  function renderTensorTables() {
    const c = current.config;
    $("activation-ledger").tBodies[0].innerHTML = catalog.tensors
      .map(
        (row) =>
          `<tr data-group="${row.group}" class="${row.alias ? "alias-row" : ""}"><td>${
            row.alias
              ? '<span title="Count its backing storage instead">view</span>'
              : `<input type="checkbox" data-save-id="${row.id}" aria-label="Count ${esc(row.label)} as saved" ${
                  M.isSaved(row, c, view.marks) ? "checked" : ""
                } />`
          }</td><td><strong>${esc(row.label)}</strong><small>${esc(row.id)}</small></td><td><code>${esc(row.logical)}</code><small>[${row.shape
            .map((size) => size.toLocaleString())
            .join(", ")}]</small></td><td>${row.dtype}</td><td>${bytes(row.bytes)}</td><td>${
            row.alias ? `Aliases <code>${esc(row.alias)}</code>. ` : ""
          }${esc(row.note || "Named backward-save candidate; exact backend lifetime can differ.")}</td></tr>`
      )
      .join("");
    const groups = M.parameterGroups(A.layers[view.layer], c.ep);
    const copies = groups.map((group) => M.parameterCopies(group, c));
    $("parameter-ledger").tBodies[0].innerHTML = groups
      .map(
        (group, index) =>
          `<tr><td>${esc(group.label)}${group.routed ? '<small class="routed-label">EP-local expert slice</small>' : ""}</td><td><code>${esc(
            group.shape
          )}</code></td><td>${count(group.params)}</td>${["master", "optimizer", "gradient", "compute"]
            .map((key) => `<td>${bytes(copies[index][key])}</td>`)
            .join("")}</tr>`
      )
      .join("");
    $("parameter-ledger").tFoot.innerHTML = `<tr><th colspan="2">Layer totals</th><th>${count(sum(groups, (group) => group.params))}</th>${[
      "master",
      "optimizer",
      "gradient",
      "compute",
    ]
      .map((key) => `<th>${bytes(sum(copies, (copy) => copy[key]))}</th>`)
      .join("")}</tr>`;
  }

  function beginCustom() {
    if (current.config.ac !== "custom") {
      view.marks = Object.fromEntries(
        [0, 1, 3].flatMap((index) =>
          M.tensorCatalog(A.layers[index], current.config)
            .tensors.filter((row) => !row.alias)
            .map((row) => [row.id, M.isSaved(row, current.config, view.marks)])
        )
      );
      controls.ac.value = "custom";
    }
  }

  function renderWorkspaces() {
    $("workspace-batch").value = view.batch;
    const w = M.workspaces(A.layers[view.layer], current.config, view.batch);
    const rows = (entries) =>
      `<dl class="workspace-values">${entries.map(([label, value]) => `<div><dt>${label}</dt><dd>${value}</dd></div>`).join("")}</dl>`;
    $("kda-workspace").innerHTML = rows([
      ["One inference state [H, K, K] FP32", bytes(w.inferenceState)],
      [`${w.chunks} chunks: [chunks, H, K, K] FP32`, bytes(w.kdaChunkState)],
    ]);
    $("residual-workspace").innerHTML = rows([
      [`L${view.layer + 1} FFN mixture: N=${w.banks} sources`, "blocks + partial"],
      ["One BF16 [T, N, D] stacked copy", bytes(w.residualStack)],
      ["One full FP32 expansion, if materialized", bytes(w.residualFP32)],
    ]);
    $("situ-workspace").innerHTML = rows([
      [`One [${w.R.toLocaleString()}, 3072] FP32 cast`, bytes(w.situCast)],
      ["Gate + up casts only", bytes(w.situCasts)],
      ["Further pointwise temporaries", "not included"],
    ]);
    $("muon-workspace").innerHTML = rows([
      ["One BF16 iterate [3072, 3584]", bytes(w.matrix)],
      ["One BF16 Gram [3072, 3072]", bytes(w.gram)],
      [`${view.batch} x (two iterates + two Grams)`, bytes(w.nsCandidates)],
    ]);
    $("head-workspace").textContent =
      `The output head is another special case: a fully materialized BF16 [T, 163840] logits tensor at this local shape is ${bytes(
        w.logits
      )}. An FP32 copy is ${bytes(
        2 * w.logits
      )}. Fused/chunked cross-entropy can avoid those full allocations; they are not assumed by every loss backend or added to an interior rank's budget.`;
  }

  function renderPipeline() {
    const c = current.config;
    const max = Math.max(1, ...current.stages.map((stage) => stage.timeline.peak));
    $("pipeline-caption").textContent = `${c.sched === "1f1b" ? "Ordinary" : "Interleaved"} 1F1B / PP${c.pp} / VP${c.vp} / ${c.mbs} microbatches`;
    $("pipeline-lifetimes").innerHTML = current.stages
      .map(
        (stage) =>
          `<div class="live-rank-row ${stage.rank === c.stage ? "selected" : ""}"><b>PP ${
            stage.rank
          }</b><div class="live-rank-track"><span style="width:${(100 * stage.timeline.peak) / max}%"></span></div><span>${bytes(
            stage.timeline.peak
          )}<small>max ${stage.timeline.peakCount} unfinished chunk forwards</small></span></div>`
      )
      .join("");
    $("pipeline-formula").textContent =
      c.sched === "1f1b"
        ? "Ordinary 1F1B: peak unfinished forwards on rank r = min(M, PP - r)"
        : "Interleaved 1F1B: warmup F operations = min(M x VP, (VP - 1) x M_per_round + 2 x (PP - 1 - r)); steady state adds F, then B.";
    const timeline = current.stage.timeline;
    const peakCounts = timeline.snapshots[timeline.peakStep].counts;
    $("pipeline-live-note").textContent = `At the selected rank's weighted saved-storage peak: ${current.owned
      .map(
        (chunk, index) =>
          `C${chunk.index} (${chunk.layers.length} layers): ${peakCounts[index]} live microbatches x ${bytes(
            current.stage.saves[index].bytes
          )} saved candidates each`
      )
      .join("; ")}. Total ${bytes(timeline.peak)}. Counted incoming AttnRes bank births per chunk: ${current.stage.saves
      .map((save) => save.incomingBanks.length)
      .join(" / ")}. Different chunk maxima are not added independently: the curve evaluates all chunks at the same logical action.`;
    renderSchedule();
    $("policy-comparison").tBodies[0].innerHTML = [
      { sched: "1f1b", vp: 1, label: "Ordinary 1F1B / VP1" },
      { sched: "interleaved", vp: 1, label: "Interleaved 1F1B / VP1" },
      { sched: "interleaved", vp: 2, label: "Interleaved 1F1B / VP2" },
    ]
      .map((policy) => {
        const other = M.calculate({ ...c, sched: policy.sched, vp: policy.vp, active: 1, unit: "chunk", phase: "backward" }, view.marks);
        return `<tr><td>${policy.label}</td><td>${other.owned.map((chunk) => `C${chunk.index}: ${chunk.layers.length} layers`).join("; ")}</td><td>${
          other.stage.timeline.peakCount
        }</td><td>${bytes(other.parts.find((part) => part.key === "compute").value)}</td><td>${bytes(other.stage.timeline.peak)}</td><td>${bytes(
          other.stateTotal
        )}</td></tr>`;
      })
      .join("");
  }

  function renderSchedule() {
    const timeline = current.stage.timeline;
    const N = timeline.actions.length;
    if (view.step === null) view.step = timeline.peakStep;
    view.step = Math.min(view.step, N);
    $("life-step").max = N;
    $("life-step").value = view.step;
    $("schedule-caption").textContent = `PP ${current.config.stage}; ${N} local F/B actions; ${timeline.peakCount} maximum live chunk forwards`;
    const width = 1000;
    const left = 100;
    const right = 24;
    const plot = width - left - right;
    const x = (step) => left + (plot * step) / N;
    const lives = timeline.lifetimes.filter((life) => life.mb < Math.min(6, current.config.mbs)).sort((a, b) => a.chunk - b.chunk || a.mb - b.mb);
    const height = 270 + 25 * lives.length;
    const top = 100;
    const bottom = 190;
    const y = (value) => bottom - ((bottom - top) * value) / Math.max(1, timeline.peak);
    const path = timeline.snapshots.map((snapshot, index) => `${index ? "L" : "M"}${x(index).toFixed(2)},${y(snapshot.bytes).toFixed(2)}`).join(" ");
    let markup = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Logical forward/backward actions, live saved-storage curve, and activation lifetimes"><text x="${left}" y="20" class="schedule-label">F = forward; B = backward. Each equal-width cell is one local action, not one unit of time.</text>`;
    timeline.actions.forEach((action, index) => {
      markup += `<rect x="${x(index)}" y="35" width="${plot / N + 0.1}" height="22" class="action-${action.kind}"><title>${action.kind} / C${
        current.owned[action.chunk].index
      } / MB${action.mb + 1}; action ${index + 1}</title></rect>`;
    });
    markup += `<path d="${path} L${x(
      N
    )},${bottom} L${left},${bottom} Z" class="schedule-area"/><path d="${path}" class="schedule-curve"/><text x="4" y="${
      top + 5
    }" class="schedule-label">${bytes(timeline.peak)}</text><text x="4" y="${bottom}" class="schedule-label">0 saved</text>`;
    for (let tick = 0; tick <= 4; tick++) {
      const step = Math.round((N * tick) / 4);
      markup += `<line x1="${x(step)}" x2="${x(step)}" y1="${top}" y2="${height - 25}" class="schedule-grid"/><text x="${x(
        step
      )}" y="215" text-anchor="middle" class="schedule-label">${step}</text>`;
    }
    markup += `<text x="${left + plot / 2}" y="237" text-anchor="middle" class="schedule-label">Local logical action index</text>`;
    lives.forEach((life, index) => {
      const yy = 253 + index * 25;
      markup += `<text x="4" y="${yy + 13}" class="schedule-label">C${current.owned[life.chunk].index} / MB${life.mb + 1}</text><rect x="${x(
        life.start
      )}" y="${yy}" width="${x(life.end) - x(life.start)}" height="18" rx="3" class="lifetime-span chunk-${life.chunk % 4}" data-start="${
        life.start
      }" data-end="${life.end}" tabindex="0" role="button" aria-label="Inspect C${current.owned[life.chunk].index} MB${life.mb + 1} from F action ${
        life.start + 1
      } to B action ${life.end}"><title>Saved from forward ${life.start + 1} through backward ${life.end}</title></rect>`;
    });
    markup += `<line id="schedule-cursor" x1="${x(view.step)}" x2="${x(view.step)}" y1="30" y2="${height - 20}" class="schedule-cursor"/></svg>`;
    $("schedule-diagram").innerHTML = markup;
    for (const span of $("schedule-diagram").querySelectorAll(".lifetime-span")) {
      const inspect = () => {
        view.step = Number(span.dataset.start) + 1;
        $("life-step").value = view.step;
        renderSchedulePoint();
        saveURL();
      };
      span.addEventListener("click", inspect);
      span.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          inspect();
        }
      });
    }
    renderSchedulePoint();
  }

  function renderSchedulePoint() {
    const timeline = current.stage.timeline;
    const point = timeline.snapshots[view.step];
    const action = timeline.actions[view.step - 1];
    $("life-position").textContent = `${view.step} / ${timeline.actions.length}`;
    $("schedule-inspector").textContent = `${
      action
        ? `After ${action.kind === "F" ? "forward" : "backward"} C${current.owned[action.chunk].index} / MB${action.mb + 1}`
        : "Before first forward"
    }: ${point.counts.map((n, index) => `C${current.owned[index].index}: ${n} live`).join("; ")}. Selected saved-storage candidates: ${bytes(
      point.bytes
    )}.`;
    const x = 100 + (876 * view.step) / timeline.actions.length;
    $("schedule-cursor")?.setAttribute("x1", x);
    $("schedule-cursor")?.setAttribute("x2", x);
    for (const span of $("schedule-diagram").querySelectorAll(".lifetime-span"))
      span.classList.toggle("live-at-cursor", Number(span.dataset.start) < view.step && view.step < Number(span.dataset.end));
  }

  function render(writeURL = true) {
    current = M.calculate(readConfig(), view.marks);
    renderBudget();
    renderAnatomy();
    renderPipeline();
    if (writeURL) saveURL();
  }

  const deckSteps = [
    {
      title: "1 / No model partitioning",
      description:
        "512 replicas each own the full model. The FP32 update-state tally alone is about 30.35 TiB per GPU. More GPUs only add capacity when ownership is actually divided.",
      config: { pp: 1, ep: 1, fsdp: "off", stage: 0, ac: "none" },
    },
    {
      title: "2 / Divide the depth with PP8",
      description:
        "Give this interior PP rank 12 layers. Master weights, momentum, and gathered weights shrink with local depth, but every rank still holds all routed experts for its layers.",
      config: { ep: 1, fsdp: "off", ac: "none" },
    },
    {
      title: "3 / Partition routed experts with EP64",
      description:
        "Own 14 of 896 experts per MoE layer. Dense and shared weights are not divided by EP. Balanced received assignments remain 16T because dispatch also brings other input ranks' rows.",
      config: { fsdp: "off", ac: "none" },
    },
    {
      title: "4 / Shard resident history with FSDP",
      description:
        "Dense master/gradient/momentum shards span 64 ranks. EP64 leaves expert FSDP1. Deferred PP accumulation can retain full dense gradients until the schedule ends.",
      config: { ac: "none" },
    },
    {
      title: "5 / Checkpoint and bound residency",
      description:
        "Keep boundary inputs and residual banks, regenerate internals during backward, and budget one gathered chunk. One active recomputation and communication/optimizer workspaces still need memory.",
      config: {},
    },
    {
      title: "6 / Measure the real implementation",
      description:
        "The 4K 512-GB200 reference completed 12 updates with a 146.18-GiB maximum logged tensor allocation. Compilation, residual/communication lifetimes, and initialization order matter. The measured run, not this subtotal, establishes feasibility.",
      config: {},
    },
  ];

  function renderDeck() {
    const step = deckSteps[view.deck];
    const result = M.calculate({ ...M.defaults, ...step.config });
    $("deck-position").textContent = `Step ${view.deck + 1} of ${deckSteps.length}`;
    $("deck-title").textContent = step.title;
    $("deck-description").textContent = step.description;
    $("deck-prev").disabled = view.deck === 0;
    $("deck-next").disabled = view.deck === deckSteps.length - 1;
    $("deck-bars").innerHTML = result.parts
      .map(
        (part) =>
          `<div class="log-row"><span>${esc(part.label)}</span><div class="log-track"><div class="log-fill" style="width:${
            part.value > 0 ? Math.min(100, Math.max(0, (100 * Math.log2(part.value / 2 ** 20)) / 26)) : 0
          }%;background:var(--${colors[part.key]})"></div><div class="capacity-mark" style="left:${
            (100 * Math.log2(M.capacity / 2 ** 20)) / 26
          }%" title="184.31 GiB reported HBM"></div></div><span class="log-value">${bytes(part.value)}</span></div>`
      )
      .join("");
    $("deck-total").textContent = `Update-state tally: ${bytes(result.stateTotal)}. Planning subtotal + chosen reserve: ${bytes(result.budget)}.`;
    $("deck-caveat").textContent =
      "Logarithmic bars: add values, not lengths. The dashed capacity line is a reference, not a fit claim. Saved/recomputed tensor candidates and the 24-GiB workspace allowance are illustrative. Only the stated 512-GPU baseline is the fixed measured reference.";
  }

  for (const key of keys)
    controls[key].addEventListener("change", () => {
      if (key === "ac" && controls.ac.value === "custom") beginCustom();
      constrain();
      view.step = null;
      render();
    });
  $("tensor-layer").addEventListener("change", () => {
    view.layer = Number($("tensor-layer").value);
    render();
  });
  $("tensor-lens").addEventListener("change", () => {
    view.lens = $("tensor-lens").value;
    render();
  });
  $("activation-ledger").addEventListener("change", (event) => {
    const id = event.target.dataset.saveId;
    if (!id) return;
    beginCustom();
    view.marks[id] = event.target.checked;
    view.step = null;
    render();
    $("activation-ledger").querySelector(`[data-save-id="${id}"]`)?.focus({ preventScroll: true });
  });
  $("tensor-save-all").addEventListener("click", () => {
    controls.ac.value = "none";
    view.marks = {};
    view.step = null;
    render();
  });
  $("tensor-checkpoint").addEventListener("click", () => {
    controls.ac.value = "full";
    view.marks = {};
    view.step = null;
    render();
  });
  $("workspace-batch").addEventListener("change", () => {
    view.batch = Number($("workspace-batch").value);
    renderWorkspaces();
    saveURL();
  });
  $("life-step").addEventListener("input", () => {
    view.step = Number($("life-step").value);
    renderSchedulePoint();
    saveURL();
  });
  $("memory-reset").addEventListener("click", () => {
    Object.assign(view, { layer: 36, node: "qkv", lens: "activation", marks: {}, step: null, batch: 14 });
    setConfig(M.defaults);
    render();
  });
  $("memory-share").addEventListener("click", async () => {
    saveURL();
    try {
      await navigator.clipboard.writeText(location.href);
      $("share-status").textContent = "Analysis link copied.";
    } catch {
      $("share-status").textContent = "Your analysis is in the address bar; copy that URL to share it.";
    }
  });
  for (const direction of ["prev", "next"])
    $("deck-" + direction).addEventListener("click", () => {
      view.deck += direction === "next" ? 1 : -1;
      renderDeck();
      saveURL();
    });
  $("deck-apply").addEventListener("click", () => {
    view.marks = {};
    view.step = null;
    setConfig({ ...M.defaults, ...deckSteps[view.deck].config });
    render();
    $("lab").scrollIntoView({ behavior: "smooth", block: "start" });
  });
  window.addEventListener("hashchange", () => {
    if (restoreURL()) {
      render(false);
      renderDeck();
    }
  });
  const themeButton = $("theme-toggle");
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
  if (window.ResizeObserver) new ResizeObserver(drawConnections).observe($("memory-anatomy"));
  setConfig(M.defaults);
  restoreURL();
  render(false);
  renderDeck();
})();
