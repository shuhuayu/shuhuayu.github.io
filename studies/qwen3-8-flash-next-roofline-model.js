((root, factory) => {
  const A = typeof module === "object" && module.exports ? require("./qwen3-8-flash-next-accounting.js") : root.QwenFlashNextAccounting;
  const api = factory(A);
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.QwenFlashNextRoofline = api;
})(typeof globalThis !== "undefined" ? globalThis : this, (A) => {
  "use strict";

  const { D, C, V, E, F, topK, sum } = A;
  // Practical sustained GB300 targets shared with the Kimi K3 roofline study;
  // the nominal peak is only the denominator of the conventional MFU metric.
  const referenceRates = Object.freeze({ gemm: 1800, hbm: 8, nvl: 900 });
  const nominalPeak = 2500;
  const defaults = Object.freeze({
    world: 16,
    pp: 2,
    vp: 2,
    ep: 8,
    tokens: 16384,
    seq: 4096,
    mbs: 16,
    ac: "full",
    slot: 0.5,
    ...referenceRates,
  });

  function normalize(input = {}) {
    const c = { ...defaults, ...input };
    for (const key of ["world", "pp", "vp", "ep", "tokens", "seq", "mbs"])
      if (!Number.isSafeInteger(c[key]) || c[key] < 1) throw new RangeError(`${key} must be a positive integer`);
    if (c.world % c.pp) throw new RangeError("GPU count must be divisible by PP");
    const dp = c.world / c.pp;
    if (dp % c.ep || E % c.ep) throw new RangeError("EP must divide both 512 experts and the DP ranks of one stage");
    if (A.L % (c.pp * c.vp)) throw new RangeError("PP x VPP must divide the 48 decoder layers");
    if (c.vp > 1 && (c.pp === 1 || c.mbs % c.pp)) throw new RangeError("Interleaving needs PP > 1 and microbatches divisible by PP");
    if (c.tokens % c.seq) throw new RangeError("Tokens per microbatch must be whole documents of the context length");
    if (!["full", "none"].includes(c.ac)) throw new RangeError("Unknown recompute policy");
    if (!Number.isFinite(c.slot) || c.slot < 0 || c.slot > 1) throw new RangeError("Slot factor must be in [0, 1]");
    for (const key of Object.keys(referenceRates))
      if (!Number.isFinite(c[key]) || c[key] <= 0) throw new RangeError(`${key} must be a positive rate`);
    return c;
  }

  // Forward GEMM shapes [M, K] x [K, N] of one layer on T local tokens.
  function layerGemms(layer, T, ep) {
    const gemms = [];
    const add = (name, m, n, k, extra = {}) => gemms.push({ name, m, n, k, groups: 1, ...extra });
    if (layer.kind === "gdn") {
      add("GDN Q/K/V projection", T, 10240, D);
      add("GDN z gate, a, b", T, 6240, D);
      add("GDN output projection", T, D, 6144);
    } else {
      add("QSA query projection", T, 6144, D);
      add("QSA output gate", T, 6144, D);
      add("QSA K + V", T, 1024, D);
      add("QSA output projection", T, D, 6144);
      add("Indexer Q/K projection", T, 640, D);
    }
    if (layer.ple) {
      add("PLE key projection", T, C, D);
      add("PLE value projection", T, D, D);
    }
    for (const mixer of ["attention", "MoE"]) {
      add(`Residual read down (${mixer})`, T, A.hcRank, C, { residual: true });
      add(`Residual read up (${mixer})`, T, C, A.hcRank, { residual: true });
      add(`Residual write (${mixer})`, T, A.S, C, { residual: true });
    }
    add("Router", T, E, D);
    add("Shared expert gate/up", T, 2 * F, D);
    add("Shared expert down", T, D, F);
    const rows = (ep * T * topK) / E;
    add("Routed expert gate/up", rows, 2 * F, D, { groups: E / ep, routed: "fc13" });
    add("Routed expert down", rows, D, F, { groups: E / ep, routed: "fc2" });
    return gemms;
  }

  function headGemms(T) {
    return [
      { name: "Final residual mixer down", m: T, n: A.hcRank, k: C, groups: 1, head: true },
      { name: "Final residual mixer up", m: T, n: C, k: A.hcRank, groups: 1, head: true },
      { name: "LM head", m: T, n: V, k: D, groups: 1, head: true },
    ];
  }

  function gemmCost(m, n, k, phase, c) {
    const flops = 2 * m * n * k;
    // One pass over BF16 operands; weight gradients are written in FP32.
    const bytes = 2 * (m * k + k * n) + (phase === "wgrad" ? 4 : 2) * m * n;
    return { flops, bytes, intensity: flops / bytes, seconds: Math.max(flops / (c.gemm * 1e12), bytes / (c.hbm * 1e12)) };
  }

  // Forward, input gradient, weight gradient, and any recompute replay.
  function gemmPhases(gemm, c) {
    const { m, n, k } = gemm;
    const phases = [
      { phase: "forward", ...gemmCost(m, n, k, "forward", c) },
      { phase: "dgrad", ...gemmCost(m, k, n, "dgrad", c) },
      { phase: "wgrad", ...gemmCost(k, n, m, "wgrad", c) },
    ];
    if (c.ac === "full" && !gemm.head) {
      // Dist-MoE recomputes only gate/up for rows its slot did not keep;
      // the routed down projection is never replayed.
      const share = gemm.routed === "fc13" ? 1 - c.slot : gemm.routed === "fc2" ? 0 : 1;
      if (share > 0) {
        const replay = gemmCost(m, n, k, "forward", c);
        phases.push({
          phase: "replay",
          flops: replay.flops * share,
          bytes: replay.bytes * share,
          intensity: replay.intensity,
          seconds: replay.seconds * share,
        });
      }
    }
    return phases;
  }

  // Mean causal keys read per query in documents of `seq` tokens: every
  // visible key until the 2,048-token budget, then 512 blocks plus the tail.
  function meanSelectedKeys(seq) {
    let total = 0;
    for (let p = 0; p < seq; p++) {
      const visible = p + 1;
      const blocks = Math.floor(visible / A.qsa.blockSize);
      total += Math.min(blocks, A.qsa.budget / A.qsa.blockSize) * A.qsa.blockSize + (visible % A.qsa.blockSize);
    }
    return total / seq;
  }

  function attentionWork(c) {
    const H = A.qsa.queryHeads;
    const Dh = A.qsa.headDim;
    const Hi = A.qsa.indexerHeads;
    const Di = A.qsa.indexerHeadDim;
    const budgetKeys = A.qsa.budget + A.qsa.blockSize - 1;
    const keys = meanSelectedKeys(c.seq);
    const meanBlocks =
      sum(
        Array.from({ length: c.seq }, (_, p) => Math.floor((p + 1) / A.qsa.blockSize)),
        (b) => b
      ) / c.seq;
    const selectedBlocks = Math.min(A.qsa.budget / A.qsa.blockSize, meanBlocks);
    const jPad = Math.ceil(budgetKeys / 8) * 8;
    const full = c.ac === "full";
    const qsaForward = 2 * H * (Dh + Dh) * keys;
    const selectForward = 2 * Hi * Di * meanBlocks;
    const scoreForward = 2 * Hi * Di * selectedBlocks;
    const teacher = 2 * H * Dh * jPad;
    const gdnModel = 6 * 3 * A.gdn.valueHeads * A.gdn.headDim * A.gdn.headDim;
    return {
      keys,
      meanBlocks,
      // TorchTitan's per-token convention, used by its TFLOPS and MFU logs.
      qsaModel: 6 * H * (Dh + Dh) * Math.min(c.seq, budgetKeys) + 6 * Hi * Di * (c.seq / A.qsa.blockSize),
      gdnModel,
      // Issued-work sketch: causal selected keys, the QK replay inside the
      // attention backward, no-grad selection and teacher, and layer replay.
      qsaIssued:
        3 * qsaForward + 2 * H * Dh * keys + selectForward + 3 * scoreForward + teacher + (full ? qsaForward + selectForward + scoreForward : 0),
      gdnIssued: gdnModel * (full ? 4 / 3 : 1),
      teacher,
      qsaForward,
      denseCounterfactual: 2 * H * (Dh + Dh) * ((c.seq + 1) / 2),
    };
  }

  // Conventional active parameters: embeddings and the lookup table excluded,
  // routed experts weighted by top-k / experts.
  function activeParams() {
    const embedding = V * D;
    const dense = A.totals.params - A.totals.routed - A.totals.table - embedding;
    const routed = (A.totals.routed * topK) / E;
    return { dense, routed, total: dense + routed };
  }

  function flopsPerToken(c) {
    const work = attentionWork(c);
    const active = activeParams();
    const conventional = 6 * active.total + 12 * work.qsaModel + 36 * work.gdnModel;
    return { active, work, conventional };
  }

  function collective(params, size, bytesPerElement) {
    return size > 1 ? bytesPerElement * params * (1 - 1 / size) : 0;
  }

  function calculate(input = {}) {
    const c = normalize(input);
    const T = c.tokens;
    const dp = c.world / c.pp;
    const edp = dp / c.ep;
    const chunks = A.partition(c.pp, c.vp);
    const full = c.ac === "full";
    const globalTokens = dp * c.mbs * T;
    const perToken = flopsPerToken(c);
    const work = perToken.work;
    const pleOwners = dp;
    const remote = 1 - 1 / c.ep;
    const stages = Array.from({ length: c.pp }, (_, rank) => {
      const owned = chunks.filter((chunk) => chunk.rank === rank);
      const layers = owned.flatMap((chunk) => chunk.layers);
      const gemms = [...layers.flatMap((layer) => layerGemms(layer, T, c.ep)), ...(owned.some((chunk) => chunk.last) ? headGemms(T) : [])].map(
        (gemm) => ({
          ...gemm,
          phases: gemmPhases(gemm, c),
        })
      );
      const qsa = layers.filter((layer) => layer.kind === "qsa").length;
      const gdn = layers.length - qsa;
      const gemmSeconds = c.mbs * sum(gemms, (g) => g.groups * sum(g.phases, (p) => p.seconds));
      const gemmModelFlops = c.mbs * sum(gemms, (g) => g.groups * 3 * g.phases[0].flops);
      const gemmIssuedFlops = c.mbs * sum(gemms, (g) => g.groups * sum(g.phases, (p) => p.flops));
      const coreIssuedFlops = c.mbs * T * (qsa * work.qsaIssued + gdn * work.gdnIssued);
      const coreModelFlops = c.mbs * T * (qsa * work.qsaModel + gdn * work.gdnModel);
      const computeSeconds = gemmSeconds + coreIssuedFlops / (c.gemm * 1e12);
      // Expert payloads: forward dispatch + combine, their two reverses, and the
      // re-dispatch of rows the Dist-MoE slot did not keep.
      const epExchanges = 4 + (full ? 1 - c.slot : 0);
      const ep = c.mbs * layers.length * epExchanges * T * topK * remote * D * 2;
      // PLE: IDs out and BF16 rows back, row gradients back, and a refetch under
      // full recompute; no deduplication is assumed.
      const pleLayers = layers.filter((layer) => layer.ple).length;
      const pleRemote = 1 - 1 / pleOwners;
      const ple = pleLayers * c.mbs * A.ple.heads * T * pleRemote * (8 + A.ple.rowDim * 2 + A.ple.rowDim * 2 + (full ? 8 + A.ple.rowDim * 2 : 0));
      // FSDP: PP keeps views and defers reduction for the whole schedule; without
      // PP each microbatch gathers for forward and backward and reduces once.
      const dense = sum(owned, (chunk) => chunk.dense);
      const routedLocal = sum(owned, (chunk) => chunk.routed) / c.ep;
      const rounds = c.pp > 1 ? { ag: 1, rs: 1 } : { ag: 2 * c.mbs, rs: c.mbs };
      const fsdp =
        rounds.ag * (collective(dense, dp, 2) + collective(routedLocal, edp, 2)) +
        rounds.rs * (collective(dense, dp, 4) + collective(routedLocal, edp, 4));
      // PP: the four-stream state forward and its gradient backward, per remote edge.
      const packet = T * C * 2;
      const edges = chunks.slice(0, -1).filter((chunk, index) => chunk.rank !== chunks[index + 1].rank);
      const pp = c.mbs * packet * edges.filter((chunk, index) => chunk.rank === rank || chunks[chunk.index + 1].rank === rank).length;
      const traffic = [
        { key: "ep", label: "Expert dispatch / combine", bytes: ep },
        { key: "ple", label: "PLE IDs, rows and row gradients", bytes: ple },
        { key: "fsdp", label: "FSDP gathers and reductions", bytes: fsdp },
        { key: "pp", label: "PP four-stream packets", bytes: pp },
      ];
      const nvlBytes = sum(traffic, (item) => item.bytes);
      const nvlSeconds = nvlBytes / (c.nvl * 1e9);
      // Newton-Schulz once per update on this GPU's Muon matrices.
      const nsIterations = 8;
      const nsFlops = (r, cols) => nsIterations * (4 * r * r * cols + 2 * r * r * r);
      const muonMatrices = (layers.length * (E / c.ep) * 3) / edp;
      const muonFlops = muonMatrices * nsFlops(F, D);
      return {
        rank,
        owned,
        layers,
        gemms,
        qsa,
        gdn,
        gemmSeconds,
        gemmModelFlops,
        gemmIssuedFlops,
        coreIssuedFlops,
        coreModelFlops,
        computeSeconds,
        traffic,
        nvlBytes,
        nvlSeconds,
        muonMatrices,
        muonFlops,
        muonSeconds: muonFlops / (c.gemm * 1e12),
        resourceSeconds: Math.max(computeSeconds, nvlSeconds),
      };
    });
    const computeSeconds = Math.max(...stages.map((stage) => stage.computeSeconds));
    const resourceSeconds = Math.max(...stages.map((stage) => stage.resourceSeconds));
    const bubble = c.pp === 1 ? 0 : (c.pp - 1) / (c.mbs * c.vp);
    const stepSeconds = resourceSeconds + bubble * computeSeconds;
    const tokensPerSecondPerGPU = globalTokens / (c.world * stepSeconds);
    // Bound with no overlap at all: every NVLink byte waits on (and delays) compute.
    const exposedStepSeconds = Math.max(...stages.map((stage) => stage.computeSeconds + stage.nvlSeconds)) + bubble * computeSeconds;
    const exposedTokensPerSecondPerGPU = globalTokens / (c.world * exposedStepSeconds);
    return {
      config: c,
      dp,
      edp,
      chunks,
      stages,
      perToken,
      globalTokens,
      bubble,
      computeSeconds,
      resourceSeconds,
      stepSeconds,
      computeTokensPerSecondPerGPU: globalTokens / (c.world * computeSeconds),
      tokensPerSecondPerGPU,
      modelTflopsPerGPU: (tokensPerSecondPerGPU * perToken.conventional) / 1e12,
      mfu: (tokensPerSecondPerGPU * perToken.conventional) / (nominalPeak * 1e12),
      exposedStepSeconds,
      exposedTokensPerSecondPerGPU,
      exposedModelTflopsPerGPU: (exposedTokensPerSecondPerGPU * perToken.conventional) / 1e12,
      expertRows: (c.ep * T * topK) / E,
    };
  }

  return Object.freeze({
    defaults,
    referenceRates,
    nominalPeak,
    normalize,
    layerGemms,
    headGemms,
    gemmCost,
    gemmPhases,
    meanSelectedKeys,
    attentionWork,
    activeParams,
    flopsPerToken,
    calculate,
  });
});
