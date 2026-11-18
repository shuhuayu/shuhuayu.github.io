(() => {
  const A = window.QwenFlashNextAccounting;
  const R = window.QwenFlashNextRoofline;
  const $ = (id) => document.getElementById(id);
  const keys = Object.keys(R.defaults).filter((key) => document.getElementById(`roof-${key}`));
  const numeric = new Set(keys.filter((key) => typeof R.defaults[key] === "number"));
  const controls = Object.fromEntries(keys.map((key) => [key, $(`roof-${key}`)]));
  let current;

  const sum = A.sum;
  const esc = (value) => String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
  const count = (value) => Math.round(value).toLocaleString("en-US");
  const seconds = (value) => (value >= 1 ? `${value.toFixed(3)} s` : `${(value * 1e3).toFixed(1)} ms`);
  const bytes = (value) =>
    value >= 2 ** 30 ? `${(value / 2 ** 30).toFixed(1)} GiB` : value >= 2 ** 20 ? `${(value / 2 ** 20).toFixed(1)} MiB` : `${count(value)} B`;
  const mflop = (value) => `${(value / 1e6).toFixed(2)} MFLOP`;

  function readConfig() {
    return { ...R.defaults, ...Object.fromEntries(keys.map((key) => [key, numeric.has(key) ? Number(controls[key].value) : controls[key].value])) };
  }

  function setConfig(config) {
    for (const key of keys) controls[key].value = String(config[key]);
  }

  function valid(config) {
    try {
      R.normalize(config);
      return true;
    } catch {
      return false;
    }
  }

  function constrain() {
    const notices = [];
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
    $("roof-warning").hidden = !notices.length;
    $("roof-warning").textContent = notices.join(" ");
  }

  function renderResults() {
    const r = current;
    $("roof-tps").textContent = count(r.tokensPerSecondPerGPU);
    $("roof-tflops").textContent = count(r.modelTflopsPerGPU);
    $("roof-mfu").textContent = `model TFLOP/s/GPU; ${(r.mfu * 100).toFixed(1)}% of the 2,500-TFLOP/s nominal peak`;
    $("roof-step").textContent = seconds(r.stepSeconds);
    const slowest = r.stages.reduce((a, b) => (b.resourceSeconds > a.resourceSeconds ? b : a));
    const limiter = slowest.computeSeconds >= slowest.nvlSeconds ? "compute / HBM" : "NVLink";
    $("roof-limiter").textContent = `${count(r.globalTokens)} tokens per update. PP rank ${
      slowest.rank
    } sets the pace (${limiter}); the nominal bubble adds ${seconds(r.bubble * r.computeSeconds)}.`;
    $("roof-stage-table").tBodies[0].innerHTML = r.stages
      .map((stage) => {
        const range = stage.owned
          .map((chunk) => `L${chunk.layers[0].index + 1}-${chunk.layers.at(-1).index + 1}${chunk.last ? " + head" : ""}`)
          .join(", ");
        return `<tr><td>${stage.rank}</td><td>${range}</td><td>${seconds(stage.gemmSeconds)}</td><td>${seconds(
          stage.computeSeconds - stage.gemmSeconds
        )}</td><td>${seconds(stage.computeSeconds)}</td><td>${seconds(stage.nvlSeconds)}</td><td>${
          stage.computeSeconds >= stage.nvlSeconds ? "Compute / HBM" : "NVLink"
        }</td></tr>`;
      })
      .join("");
    $("hero-tps").textContent = count(R.calculate().tokensPerSecondPerGPU);
    $("hero-tflops").textContent = count(R.calculate().modelTflopsPerGPU);
  }

  // One representative layer of each kind, plus the head, on the slowest rank.
  function representativeGemms() {
    const T = current.config.tokens;
    const pick = (layer) => R.layerGemms(layer, T, current.config.ep).map((gemm) => ({ ...gemm, layer: layer.kind.toUpperCase() }));
    const gdn = pick(A.layers[0]).filter((gemm) => !gemm.name.startsWith("Residual write") || true);
    const qsa = pick(A.layers[3]).filter((gemm) => gemm.name.startsWith("QSA") || gemm.name.startsWith("Indexer"));
    const ple = pick(A.layers[A.ple.layer]).filter((gemm) => gemm.name.startsWith("PLE"));
    const head = R.headGemms(T).map((gemm) => ({ ...gemm, layer: "Head" }));
    return [...gdn, ...qsa, ...ple, ...head].map((gemm) => ({ ...gemm, phases: R.gemmPhases(gemm, current.config) }));
  }

  function renderGemms() {
    const c = current.config;
    const gemms = representativeGemms();
    $("roof-gemm-table").tBodies[0].innerHTML = gemms
      .map((gemm) => {
        const f = gemm.phases[0];
        const bound = f.flops / (c.gemm * 1e12) >= f.bytes / (c.hbm * 1e12) ? "Compute" : "HBM";
        return `<tr><td>${esc(gemm.name)}</td><td><code>[${count(gemm.m)}, ${count(gemm.k)}] x [${count(gemm.k)}, ${count(gemm.n)}]</code></td><td>${
          gemm.groups
        }</td><td>${f.intensity.toFixed(0)}</td><td>${bound}</td><td>${seconds(gemm.groups * sum(gemm.phases, (p) => p.seconds))}</td></tr>`;
      })
      .join("");
    // Log-log roofline: intensity 1..10^4 FLOP/B, throughput 1..10^4 TFLOP/s.
    const W = 900;
    const H = 420;
    const left = 70;
    const right = 20;
    const top = 20;
    const bottom = 50;
    const x = (i) => left + ((Math.log10(i) - 0) / 4) * (W - left - right);
    const y = (t) => H - bottom - ((Math.log10(t) - 0) / 4) * (H - top - bottom);
    const ridge = (c.gemm * 1e12) / (c.hbm * 1e12);
    let svg = "";
    for (let e = 0; e <= 4; e++) {
      svg += `<line x1="${x(10 ** e)}" x2="${x(10 ** e)}" y1="${top}" y2="${H - bottom}" class="schedule-grid"/><text x="${x(10 ** e)}" y="${
        H - bottom + 18
      }" text-anchor="middle">${count(10 ** e)}</text>`;
      svg += `<line x1="${left}" x2="${W - right}" y1="${y(10 ** e)}" y2="${y(10 ** e)}" class="schedule-grid"/><text x="${left - 8}" y="${
        y(10 ** e) + 4
      }" text-anchor="end">${count(10 ** e)}</text>`;
    }
    svg += `<polyline points="${x(1)},${y(c.hbm)} ${x(ridge)},${y(c.gemm)} ${x(10 ** 4)},${y(
      c.gemm
    )}" fill="none" stroke="var(--ink)" stroke-width="2"/>`;
    svg += `<text x="${x(ridge) + 6}" y="${y(c.gemm) - 8}">BF16 ${count(c.gemm)} TFLOP/s; ridge ${ridge.toFixed(0)} FLOP/B</text>`;
    svg += `<text x="${left + (W - left - right) / 2}" y="${H - 10}" text-anchor="middle">Arithmetic intensity, FLOP / byte (log)</text>`;
    svg += `<text transform="translate(16 ${top + (H - top - bottom) / 2}) rotate(-90)" text-anchor="middle">Attainable TFLOP/s (log)</text>`;
    const colors = { GDN: "var(--kda)", QSA: "var(--gradient)", Head: "var(--master)" };
    for (const gemm of gemms) {
      const f = gemm.phases[0];
      const attainable = Math.min(c.gemm, (f.intensity * c.hbm * 1e12) / 1e12);
      const color = gemm.routed ? "var(--moe)" : gemm.name.startsWith("PLE") ? "var(--gold)" : colors[gemm.layer] || "var(--muted)";
      svg += `<circle cx="${x(Math.max(1, f.intensity))}" cy="${y(Math.max(1, attainable))}" r="6" fill="${color}" fill-opacity="0.8"><title>${esc(
        gemm.name
      )}: ${f.intensity.toFixed(0)} FLOP/B, ${count(attainable)} TFLOP/s attainable</title></circle>`;
    }
    $("roof-chart").innerHTML = svg;
    $("roof-chart-caption").textContent = `${count(c.tokens)} tokens; ${count(current.expertRows)} rows per local expert`;
  }

  function renderFlops() {
    const w = current.perToken.work;
    const rows = [
      [
        "QSA attention (24 heads, QK 256 + V 256)",
        6 * 24 * 512 * Math.min(current.config.seq, 2051),
        3 * w.qsaForward + 2 * 24 * 256 * w.keys,
        `Causal mean of ${w.keys.toFixed(0)} keys read per query in ${count(
          current.config.seq
        )}-token documents, plus the QK replay of the backward.`,
      ],
      [
        "QSA indexer",
        6 * 4 * 128 * (current.config.seq / 4),
        2 * 4 * 128 * w.meanBlocks + 3 * 2 * 4 * 128 * Math.min(512, w.meanBlocks),
        "No-grad selection over all visible blocks, plus trained scores on the selected blocks.",
      ],
      [
        "Indexer KL teacher",
        0,
        w.teacher,
        "No-grad QK over every budget slot (padded to 2,056), kept instead of recomputed. Real work, not model FLOPs.",
      ],
      [
        "GDN state contractions (48 heads, 128 x 128)",
        w.gdnModel,
        w.gdnIssued,
        "TorchTitan's 6 x 3 x H x K x V; full recompute replays the forward third. Intra-chunk terms omitted.",
      ],
    ];
    $("core-table").tBodies[0].innerHTML = rows
      .map(
        ([name, model, issued, note]) =>
          `<tr><td>${esc(name)}</td><td>${model ? mflop(model) : "0"}</td><td>${mflop(issued)}</td><td>${esc(note)}</td></tr>`
      )
      .join("");
    $("qsa-note").innerHTML = `In ${count(current.config.seq)}-token documents, a query can see ${count(
      (current.config.seq + 1) / 2
    )} keys on average and QSA reads ${w.keys.toFixed(0)} of them: the sparse forward is ${mflop(w.qsaForward)} per token per layer against ${mflop(
      w.denseCounterfactual
    )} for dense causal attention, a ${(100 * (1 - w.qsaForward / w.denseCounterfactual)).toFixed(
      0
    )}% saving. The 2,048-key budget only pays off when documents are much longer than 4K: a query at position 262,144 reads 2,051 of 262,144 keys, under 1%.`;
    const p = current.perToken;
    $("token-flops").tBodies[0].innerHTML = [
      ["6 x activated matmul parameters", 6 * p.active.total],
      ["12 QSA layers (conventional)", 12 * w.qsaModel],
      ["36 GDN layers (conventional)", 36 * w.gdnModel],
      ["Conventional model FLOPs / token", p.conventional],
    ]
      .map(
        ([label, value], index, all) =>
          `<tr><td>${index === all.length - 1 ? `<strong>${label}</strong>` : label}</td><td>${(value / 1e9).toFixed(3)}</td></tr>`
      )
      .join("");
  }

  function renderTraffic() {
    const stages = current.stages;
    const notes = {
      ep: "Balanced routing; one copy per remote assignment; 4 exchanges + re-dispatch.",
      ple: "16 lookups / token, 7/8 remote, no deduplication; only on the PLE stage.",
      fsdp:
        current.config.pp > 1
          ? "One BF16 gather + one FP32 reduce-scatter of dense weights per update."
          : "Two gathers + one reduction per microbatch.",
      pp: "[T, 10240] BF16 forward + gradient per remote edge touching the rank.",
    };
    const cols = (key) => [0, 1].map((rank) => (stages[rank] ? bytes(stages[rank].traffic.find((item) => item.key === key).bytes) : "n/a"));
    $("traffic-table").tBodies[0].innerHTML =
      stages[0].traffic
        .map(
          (item) => `<tr><td>${esc(item.label)}</td><td>${cols(item.key)[0]}</td><td>${cols(item.key)[1]}</td><td>${esc(notes[item.key])}</td></tr>`
        )
        .join("") +
      `<tr><td><strong>NVLink service</strong></td><td><strong>${seconds(stages[0].nvlSeconds)}</strong></td><td><strong>${
        stages[1] ? seconds(stages[1].nvlSeconds) : "n/a"
      }</strong></td><td>Total bytes / one-way NVLink rate.</td></tr>`;
  }

  function renderPipeline() {
    const stages = current.stages;
    const max = Math.max(...stages.map((stage) => stage.computeSeconds));
    $("pipeline-bars").innerHTML = stages
      .map(
        (stage) =>
          `<div class="cost-row"><span>PP rank ${stage.rank}</span><div class="cost-track"><div class="cost-fill" style="width:${(
            (100 * stage.computeSeconds) /
            max
          ).toFixed(1)}%"></div></div><span class="cost-value">${seconds(stage.computeSeconds)}</span></div>`
      )
      .join("");
    $("pipeline-caption").textContent = `bubble ${(current.bubble * 100).toFixed(2)}%`;
    if (stages.length > 1) {
      const fast = Math.min(...stages.map((stage) => stage.computeSeconds));
      $("pipeline-text").textContent = `In the configured recipe, the slowest rank needs ${seconds(
        max
      )} of compute per update and the fastest ${seconds(fast)}: the fast rank idles ${((100 * (max - fast)) / max).toFixed(
        1
      )}% of the time on top of the nominal ${(current.bubble * 100).toFixed(
        2
      )}% bubble. The LM head alone is a [T, 2560] x [2560, 248320] GEMM, 62.5 TFLOP per 16K-token microbatch over its three phases.`;
    } else {
      $("pipeline-text").textContent = "With PP1 there is no pipeline bubble; every GPU runs all 48 layers and the head.";
    }
  }

  function renderMuon() {
    const stage = current.stages[0];
    const perMatrix = 8 * (4 * 640 * 640 * 2560 + 2 * 640 ** 3);
    $("muon-formula").textContent = `Per expert matrix  = 8 * (4 * 640^2 * 2560 + 2 * 640^3) = ${(perMatrix / 1e9).toFixed(2)} GFLOP
Matrices per GPU    = ${stage.layers.length} layers * ${A.E / current.config.ep} experts * 3 / expert FSDP ${current.edp} = ${count(
      stage.muonMatrices
    )}
Routed Muon work    = ${(stage.muonFlops / 1e12).toFixed(1)} TFLOP per update = ${seconds(stage.muonSeconds)} at ${count(current.config.gemm)} TFLOP/s
Share of the ideal step = ${((100 * stage.muonSeconds) / current.stepSeconds).toFixed(1)}%`;
  }

  function update() {
    constrain();
    try {
      current = R.calculate(readConfig());
    } catch (error) {
      $("roof-warning").hidden = false;
      $("roof-warning").textContent = error.message;
      return;
    }
    renderResults();
    renderGemms();
    renderFlops();
    renderTraffic();
    renderPipeline();
    renderMuon();
  }

  for (const control of Object.values(controls)) control.addEventListener(control.tagName === "INPUT" ? "input" : "change", update);
  $("roof-reset").addEventListener("click", () => {
    setConfig(R.defaults);
    update();
  });
  $("roof-share").addEventListener("click", async () => {
    history.replaceState(null, "", `#${new URLSearchParams({ roofline: JSON.stringify(readConfig()) })}`);
    try {
      await navigator.clipboard.writeText(location.href);
      $("roof-share-status").textContent = "Link copied";
    } catch {
      $("roof-share-status").textContent = "Link in the address bar";
    }
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

  setConfig(R.defaults);
  const params = new URLSearchParams(location.hash.slice(1));
  if (params.has("roofline"))
    try {
      setConfig(R.normalize(JSON.parse(params.get("roofline"))));
    } catch {}
  update();
})();
