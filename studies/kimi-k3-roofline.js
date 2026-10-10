(() => {
  const A = window.KimiK3Model;
  const R = window.KimiK3Roofline;
  const keys = Object.keys(R.defaults);
  const controls = Object.fromEntries(keys.map((key) => [key, document.getElementById(`roof-${key}`)]));
  const get = (id) => document.getElementById(id);
  const number = (value, digits = 2) => value.toLocaleString("en-US", { maximumFractionDigits: digits });
  const bytes = (value) =>
    value >= 2 ** 30 ? `${number(value / 2 ** 30)} GiB` : value >= 2 ** 20 ? `${number(value / 2 ** 20)} MiB` : `${number(value)} B`;
  const seconds = (value) => `${number(value, 3)} s`;
  let selectedGemm = "Routed gate/up";
  let selectedStage = 3;
  let current;

  function readConfig() {
    return Object.fromEntries(keys.map((key) => [key, typeof R.defaults[key] === "number" ? Number(controls[key].value) : controls[key].value]));
  }

  function normalizeAxes() {
    const ranks = 512 / Number(controls.pp.value);
    const changes = [];
    for (const key of ["cp", "ep", "domain"]) {
      const control = controls[key];
      for (const option of control.options) option.disabled = Number(option.value) > ranks;
      if (Number(control.value) > ranks) {
        const before = control.value;
        control.value = [...control.options].filter((option) => !option.disabled).at(-1).value;
        changes.push(`${key.toUpperCase()} ${before} -> ${control.value}`);
      }
    }
    selectedStage = Math.min(selectedStage, Number(controls.pp.value) - 1);
    return changes;
  }

  function text(id, value) {
    get(id).textContent = value;
  }

  function makeRow(cells) {
    const row = document.createElement("tr");
    cells.forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.appendChild(cell);
    });
    return row;
  }

  function svgElement(name, attributes, content) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    if (content !== undefined) element.textContent = content;
    return element;
  }

  function renderRoofline(result) {
    const svg = get("roof-chart");
    svg.querySelectorAll("g, path, line, circle, text").forEach((element) => element.remove());
    const left = 76,
      right = 810,
      top = 34,
      bottom = 371;
    const x = (intensity) => left + (Math.log2(intensity) / 13) * (right - left);
    const y = (tf) => bottom - (Math.log10(tf / 0.1) / Math.log10(50000)) * (bottom - top);
    const rate = result.config.gemm;
    const ridge = rate / result.config.hbm;
    const low = result.config.hbm;
    const roofPoints = `${left},${y(low)} ${x(ridge)},${y(rate)} ${right},${y(rate)}`;
    svg.appendChild(svgElement("path", { class: "plot-area", d: `M${left},${bottom} L${roofPoints.replaceAll(" ", " L")} L${right},${bottom} Z` }));
    for (const tick of [1, 4, 16, 64, 256, 1024, 4096]) {
      svg.appendChild(svgElement("line", { class: "plot-grid", x1: x(tick), x2: x(tick), y1: top, y2: bottom }));
      svg.appendChild(svgElement("text", { x: x(tick), y: bottom + 23, "text-anchor": "middle" }, number(tick, 0)));
    }
    for (const tick of [0.1, 1, 10, 100, 1000]) {
      svg.appendChild(svgElement("line", { class: "plot-grid", x1: left, x2: right, y1: y(tick), y2: y(tick) }));
      svg.appendChild(svgElement("text", { x: left - 12, y: y(tick) + 4, "text-anchor": "end" }, number(tick, 1)));
    }
    svg.appendChild(svgElement("line", { class: "plot-ceiling", x1: left, x2: right, y1: y(2500), y2: y(2500) }));
    svg.appendChild(svgElement("text", { x: right, y: y(2500) - 10, "text-anchor": "end" }, "Nominal dense BF16: 2,500 TF/s"));
    svg.appendChild(svgElement("path", { class: "plot-roof", d: `M${roofPoints.replaceAll(" ", " L")}` }));
    svg.appendChild(
      svgElement(
        "text",
        { class: "plot-label", x: (left + right) / 2, y: bottom + 60, "text-anchor": "middle" },
        "HBM arithmetic intensity / FLOP per byte (log scale)"
      )
    );
    svg.appendChild(
      svgElement(
        "text",
        { class: "plot-label", transform: "translate(20 205) rotate(-90)", "text-anchor": "middle" },
        "Attainable TFLOP/s (log scale)"
      )
    );
    svg.appendChild(svgElement("text", { x: x(16), y: y(result.config.hbm * 16) - 14 }, `${result.config.hbm} TB/s HBM slope`));
    const buttons = get("roof-gemm-buttons");
    buttons.replaceChildren();
    for (const [index, gemm] of result.forwardGemmPoints.entries()) {
      const tf = Math.min(rate, result.config.hbm * gemm.intensity);
      const chosen = selectedGemm === gemm.name;
      if (chosen) svg.appendChild(svgElement("circle", { class: "plot-halo", cx: x(gemm.intensity), cy: y(tf), r: 12 }));
      const dot = svgElement("circle", { class: `plot-dot${chosen ? " selected" : ""}`, cx: x(gemm.intensity), cy: y(tf), r: chosen ? 6 : 4 });
      dot.appendChild(svgElement("title", {}, `${gemm.name}: ${number(gemm.intensity)} FLOP/B, ${number(tf)} TF/s roof, ${gemm.bound}-limited`));
      svg.appendChild(dot);
      if (chosen)
        svg.appendChild(svgElement("text", { class: "point-label", x: x(gemm.intensity), y: y(tf) + 26, "text-anchor": "middle" }, gemm.name));
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = `${index + 1}. ${gemm.name}`;
      button.setAttribute("aria-pressed", String(chosen));
      button.addEventListener("click", () => {
        selectedGemm = gemm.name;
        renderRoofline(current);
      });
      buttons.appendChild(button);
    }
    const chosen = result.forwardGemmPoints.find((gemm) => gemm.name === selectedGemm);
    const inspector = get("roof-inspector");
    inspector.replaceChildren();
    const heading = document.createElement("h3");
    heading.textContent = `${chosen.name}: [M,K] x [K,N], M=${number(chosen.m)}, K=${number(chosen.k, 0)}, N=${number(chosen.n, 0)}`;
    const stats = document.createElement("div");
    stats.className = "gemm-stats";
    for (const [value, label] of [
      [`${number(chosen.flops / 1e9)} GFLOP`, "One forward GEMM / one expert if grouped"],
      [bytes(chosen.bytes), "Modeled BF16 operand traffic"],
      [`${number(chosen.intensity)} FLOP/B`, `${chosen.bound}-limited at selected ceilings`],
      [`${number(chosen.seconds * 1e6)} us`, "Ideal one-GEMM service time / not measured"],
    ]) {
      const box = document.createElement("div");
      const valueElement = document.createElement("strong");
      valueElement.textContent = value;
      const labelElement = document.createElement("span");
      labelElement.textContent = label;
      box.append(valueElement, labelElement);
      stats.appendChild(box);
    }
    inspector.append(heading, stats);
    text("roof-ridge", `Ridge: ${number(ridge)} FLOP/B`);
  }

  function renderBars(id, rows, format) {
    const container = get(id);
    container.replaceChildren();
    const maximum = Math.max(...rows.map((row) => row.value));
    for (const item of rows) {
      const row = document.createElement("div");
      row.className = "cost-row";
      const label = document.createElement("span");
      label.textContent = item.label;
      const track = document.createElement("div");
      track.className = "cost-track";
      const fill = document.createElement("div");
      fill.className = "cost-fill";
      fill.style.width = `${100 * (item.value / maximum)}%`;
      fill.style.background = `var(--${item.color})`;
      fill.dataset.value = item.value;
      track.appendChild(fill);
      const value = document.createElement("span");
      value.className = "cost-value";
      value.textContent = format(item.value);
      row.append(label, track, value);
      container.appendChild(row);
    }
  }

  function renderFlops(result) {
    const kda = 69 * R.attentionWork(false, result.config).model;
    const mla = 24 * R.attentionWork(true, result.config).model;
    const flash = 24 * R.attentionWork(true, result.config).flashReplay;
    const checkpoint = result.issuedFlops / result.globalTokens - 6 * A.linearParams - kda - mla - flash;
    renderBars(
      "flop-bars",
      [
        { label: "Learned matrices: forward + backward", value: 6 * A.linearParams, color: "kda" },
        { label: "KDA core: algorithmic contractions", value: kda, color: "roof-blue" },
        { label: "MLA core: causal contractions", value: mla, color: "mla" },
        { label: "Full layer checkpoint replay", value: Math.max(0, checkpoint), color: "moe" },
        { label: "FlashAttention QK replay", value: flash, color: "gold" },
      ],
      (value) => `${number(value / 1e9)} GF/tok`
    );
    text(
      "flop-summary",
      `Causal algorithmic model work: ${number(result.causalModelFlops / result.globalTokens / 1e9, 3)} GFLOP/token. With modeled replay: ${number(
        result.issuedFlops / result.globalTokens / 1e9,
        3
      )} GFLOP/token. Conventional TorchTitan metric at this configured context: ${number(
        result.conventionPerToken / 1e9,
        3
      )} GFLOP/token; it does not follow the equal-document split control. Bars share a linear scale; small components are intentionally small.`
    );
  }

  function renderTraffic(result) {
    const stage = result.stages[selectedStage];
    const container = get("roof-stage-buttons");
    container.replaceChildren();
    for (const candidate of result.stages) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = `PP ${candidate.rank}`;
      button.setAttribute("aria-pressed", String(candidate.rank === selectedStage));
      const caption = document.createElement("span");
      caption.textContent = `L${candidate.owned.layers[0].index + 1}-${candidate.owned.layers.at(-1).index + 1}`;
      button.appendChild(caption);
      button.addEventListener("click", () => {
        selectedStage = candidate.rank;
        renderTraffic(current);
      });
      container.appendChild(button);
    }
    text(
      "roof-stage-caption",
      `PP${stage.rank}: ${stage.kda} KDA + ${stage.mla} MLA; ${stage.moeLayers} MoE layers. ${result.rounds.ag} weight AG rounds and ${
        result.rounds.rs
      } gradient RS rounds per layer/group. FP32 update-state tally: ${bytes(stage.stateBytes)}; not a peak memory estimate. Expert FSDP${
        result.expertFsdp
      }.`
    );
    const body = get("traffic-ledger").querySelector("tbody");
    body.replaceChildren();
    for (const item of stage.traffic) {
      const row = makeRow([
        item.label,
        bytes(item.nvl),
        seconds(item.nvl / (result.config.nvl * 1e9)),
        bytes(item.network),
        seconds(item.network / (result.config.network * 1e9)),
      ]);
      row.dataset.traffic = item.key;
      row.dataset.nvlBytes = item.nvl;
      row.dataset.networkBytes = item.network;
      body.appendChild(row);
    }
    text("muon-work", `${number(stage.nsFlops / 1e12)} TFLOP / update`);
    text(
      "muon-work-detail",
      `${stage.moeLayers} MoE layers x ${result.localExperts} experts x 3 logical matrices, with an ideal 1/${
        result.expertFsdp
      } expert-FSDP compute-ownership share. Expert Newton-Schulz contractions only; excludes dense matrices and redistribution. At the selected BF16 ceiling: ${seconds(
        stage.nsFlops / (result.config.gemm * 1e12)
      )}, not a measured timing.`
    );
  }

  function renderResources(result) {
    renderBars(
      "resource-bars",
      [
        { label: `Modeled compute / PP ${result.computeStage.rank}`, value: result.computeSeconds, color: "kda" },
        { label: `Shared NVLink budget / PP ${result.nvlStage.rank}`, value: result.nvlSeconds, color: "moe" },
        { label: `Shared NIC budget / PP ${result.networkStage.rank}`, value: result.networkSeconds, color: "mla" },
      ],
      seconds
    );
    text(
      "resource-summary",
      `Resource-load envelope = max(compute, shared NVLink, shared NIC) = ${seconds(
        result.resourceSeconds
      )}. Adding the nominal balanced-stage compute bubble (${seconds(result.bubbleSeconds)}) gives the ${seconds(
        result.sketchSeconds
      )} diagnostic sketch. This is not a simulated schedule or validated step-time prediction; different resources may bottleneck on different ranks, and real overlap is not free.`
    );
    get("bubble-useful").style.width = `${100 / (1 + result.bubbleRatio)}%`;
    get("bubble-idle").style.width = `${(100 * result.bubbleRatio) / (1 + result.bubbleRatio)}%`;
    text(
      "bubble-label",
      `${number(100 * result.bubbleRatio)}% overhead / ${number((100 * result.bubbleRatio) / (1 + result.bubbleRatio))}% interval share`
    );
    text(
      "bubble-caption",
      `${result.config.pp} stages x ${result.config.mbs} microbatches, one stage per rank. Scheduled compute: ${seconds(
        result.computeSeconds
      )}; nominal fill/drain: ${seconds(result.bubbleSeconds)}. Stage imbalance, pointwise work, CPU stalls and exposed communication are additional.`
    );
    const packets = get("pp-packets");
    packets.replaceChildren();
    const title = document.createElement("h3");
    title.textContent = "Outgoing PP packets: hidden + residual blocks";
    packets.appendChild(title);
    if (result.config.pp === 1) {
      const note = document.createElement("p");
      note.textContent = "PP1 has no pipeline send/receive edges.";
      packets.appendChild(note);
    }
    for (const stage of result.stages.slice(0, -1)) {
      const row = document.createElement("div");
      row.className = "packet-row";
      const label = document.createElement("span");
      label.textContent = `PP${stage.rank} -> ${stage.rank + 1}`;
      const blocks = document.createElement("div");
      blocks.className = "packet-blocks";
      blocks.setAttribute("aria-label", `One hidden stream and ${stage.after} residual blocks`);
      for (let i = 0; i <= stage.after; i++) blocks.appendChild(document.createElement("span"));
      const volume = document.createElement("span");
      volume.textContent = `${bytes((1 + stage.after) * result.tokens * A.D * 2)} / forward`;
      row.append(label, blocks, volume);
      packets.appendChild(row);
    }
    const legend = document.createElement("p");
    legend.className = "note";
    legend.textContent =
      "Green = one hidden stream; pink = residual blocks. Each block represents [T,7168] in BF16. Backward sends the corresponding gradient packet; the per-stage ledger includes both adjacent edges' outbound traffic.";
    packets.appendChild(legend);
  }

  function render() {
    const changes = normalizeAxes();
    const warning = get("roof-warning");
    if (!get("roof-controls").checkValidity()) {
      warning.hidden = false;
      warning.textContent = "Use positive rates within the displayed input ranges. Charts retain the last valid configuration.";
      return;
    }
    current = R.calculate(readConfig());
    const c = current.config;
    const notices = changes.map((change) => `Adjusted to fit the 512-GPU mesh: ${change}.`);
    if (c.cp > 1) notices.push("CP attention communication is not priced. The long-context time sketch is incomplete.");
    if (c.policy === "retain")
      notices.push(
        "Retained compute views and unreduced gradients increase live memory. This is an illustrative lifecycle, not proof that the recipe fits."
      );
    if (c.segments > 1)
      notices.push(`Each logical document is ${number(c.seq / c.segments, 0)} tokens, not a contiguous ${number(c.seq, 0)}-token context.`);
    warning.hidden = notices.length === 0;
    warning.textContent = notices.join(" ");
    text(
      "roof-shape",
      `512 = PP${c.pp} x CP${c.cp} x DP${current.dp}; dense FSDP${current.stageRanks}; EP${c.ep} x expert FSDP${current.expertFsdp}. T_local = ${
        c.rows
      } x ${number(c.seq, 0)} / ${c.cp} = ${number(current.tokens, 0)}. FP32 state / BF16 GEMMs.`
    );
    text("roof-tokens", `${number(current.globalTokens / 1e6)}M`);
    get("roof-tokens").dataset.tokens = current.globalTokens;
    text("roof-batch", `${number(current.globalRows, 0)} rows x ${number(c.seq, 0)} tokens. ${c.mbs} microbatches in one schedule.`);
    text("roof-expert-rows", number(current.expertRows));
    get("roof-expert-rows").dataset.rows = current.expertRows;
    text(
      "roof-expert-detail",
      `${current.localExperts} local experts; ${number(
        current.tokens * A.topK,
        0
      )} balanced routed assignments per input-rank equivalent. Expected rows, no token dropping.`
    );
    text("roof-sketch", seconds(current.sketchSeconds));
    text(
      "roof-sketch-detail",
      `Ideal rates + nominal 1F1B bubble. ${c.cp > 1 ? "CP communication omitted. " : ""}Optimizer and non-matmul service omitted; not measured.`
    );
    renderRoofline(current);
    renderFlops(current);
    renderTraffic(current);
    renderResources(current);
  }

  function apply(config) {
    for (const key of keys) controls[key].value = config[key];
    render();
  }

  const presets = {
    baseline: {},
    ep1: { ep: 1 },
    split: { domain: 8 },
    long: { seq: 65536, cp: 16 },
    retain: { policy: "retain" },
  };
  document.querySelectorAll("[data-preset]").forEach((button) =>
    button.addEventListener("click", () => {
      selectedStage = 3;
      apply({ ...R.defaults, ...presets[button.dataset.preset] });
    })
  );
  get("show-critical").addEventListener("click", () => {
    selectedStage = current.computeStage.rank;
    renderTraffic(current);
  });
  for (const key of keys) controls[key].addEventListener("change", render);

  const references = [
    { label: "2K / forced-balanced", seq: 2048, rows: 2, throughput: 372 },
    { label: "4K / forced-balanced", seq: 4096, rows: 1, throughput: 370 },
    { label: "4K / learned + blocking", seq: 4096, rows: 1, throughput: 297 },
  ];
  for (const reference of references) {
    const result = R.calculate({ seq: reference.seq, rows: reference.rows });
    const tf = (reference.throughput * result.conventionPerToken) / 1e12;
    const row = makeRow([
      reference.label,
      `${reference.rows} x ${number(reference.seq, 0)}`,
      reference.throughput,
      seconds(result.globalTokens / (512 * reference.throughput)),
      number(tf, 1),
      `${number((100 * tf) / 2500)}%`,
    ]);
    row.dataset.reference = reference.seq === 2048 ? "v1" : reference.throughput === 370 ? "v1-5" : "v1-6";
    get("measured-runs").querySelector("tbody").appendChild(row);
  }
  text("metadata-sol", `${number((R.calculate().metadataCall.nvl / (900 * 1e9)) * 1e3, 3)} ms`);

  function download(name, type, content) {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const link = document.createElement("a");
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  get("export-json").addEventListener("click", () =>
    download(
      "kimi-k3-roofline.json",
      "application/json",
      JSON.stringify(
        {
          description: "Optimistic accounting sketch, not profiler data or a launch configuration. CP communication and non-matmul work omitted.",
          precision: "FP32 master, gradient and optimizer state; BF16 model GEMMs and Muon Newton-Schulz working copy",
          config: current.config,
          structural: { parameters: A.totalParams, activeParameterConvention: A.activeParams, learnedActiveMatrices: A.linearParams },
          derived: {
            dp: current.dp,
            stageRanks: current.stageRanks,
            expertFsdp: current.expertFsdp,
            localTokens: current.tokens,
            globalTokens: current.globalTokens,
            expertRows: current.expertRows,
            causalModelFlops: current.causalModelFlops,
            modeledIssuedFlops: current.issuedFlops,
            conventionalFlopsPerToken: current.conventionPerToken,
            computeSeconds: current.computeSeconds,
            nvlSeconds: current.nvlSeconds,
            networkSeconds: current.networkSeconds,
            nominalBubbleSeconds: current.bubbleSeconds,
            sketchSeconds: current.sketchSeconds,
          },
          stages: current.stages.map((stage) => ({
            rank: stage.rank,
            layers: stage.owned.layers.map((layer) => layer.index),
            stateBytes: stage.stateBytes,
            computeSeconds: stage.computeSeconds,
            nvlSeconds: stage.nvlSeconds,
            networkSeconds: stage.networkSeconds,
            traffic: stage.traffic,
            expertNSFlops: stage.nsFlops,
          })),
        },
        null,
        2
      )
    )
  );
  get("export-csv").addEventListener("click", () => {
    const header = [
      "pp_rank",
      "pp_degree",
      "cp_degree",
      "ep_degree",
      "context",
      "rows_per_microbatch",
      "microbatches",
      "documents_per_row",
      "layer_checkpoint",
      "fsdp_policy",
      "stage_ranks_per_nvl",
      "payload_copy_policy",
      "pp_hop_resource",
      "local_tokens",
      "global_tokens",
      "gemm_tflops",
      "hbm_tb_s",
      "nvl_gb_s",
      "network_gb_s",
      "state_bytes",
      "compute_sketch_s",
      "nvl_service_s",
      "network_service_s",
      "pp_outbound_bytes",
      "expert_ns_flops",
    ];
    const c = current.config;
    const rows = current.stages.map((stage) => [
      stage.rank,
      c.pp,
      c.cp,
      c.ep,
      c.seq,
      c.rows,
      c.mbs,
      c.segments,
      c.ac,
      c.policy,
      c.domain,
      c.copies,
      c.ppLink,
      current.tokens,
      current.globalTokens,
      c.gemm,
      c.hbm,
      c.nvl,
      c.network,
      stage.stateBytes,
      stage.computeSeconds,
      stage.nvlSeconds,
      stage.networkSeconds,
      stage.ppBytes,
      stage.nsFlops,
    ]);
    download("kimi-k3-roofline-stages.csv", "text/csv", [header, ...rows].map((row) => row.join(",")).join("\n") + "\n");
  });
  get("print-page").addEventListener("click", () => window.print());
  const theme = get("theme-toggle");
  let saved;
  try {
    saved = localStorage.getItem("kimi-k3-study-theme");
  } catch {}
  document.documentElement.classList.toggle("dark", saved === "dark" || (!saved && matchMedia("(prefers-color-scheme: dark)").matches));
  const themeLabel = () => (theme.textContent = document.documentElement.classList.contains("dark") ? "Light" : "Dark");
  theme.addEventListener("click", () => {
    document.documentElement.classList.toggle("dark");
    try {
      localStorage.setItem("kimi-k3-study-theme", document.documentElement.classList.contains("dark") ? "dark" : "light");
    } catch {}
    themeLabel();
  });
  themeLabel();
  apply(R.defaults);
})();
