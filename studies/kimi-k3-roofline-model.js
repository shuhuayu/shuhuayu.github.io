((root) => {
  const A = typeof module === "object" && module.exports ? require("./kimi-k3-accounting.js") : root.KimiK3Model;
  const { D, H, K, Z, F, experts, topK } = A;
  const sum = (items, get) => items.reduce((total, item) => total + get(item), 0);
  const referenceRates = Object.freeze({ gemm: 1800, fp8Gemm: 3600, hbm: 8, nvl: 900, network: 100 });
  const rateKeys = Object.keys(referenceRates);
  const hardwareProfiles = {
    gb200: {
      label: "GB200",
      computeBasis: "gb300-reference-proxy",
      note: "The compute preset reuses Edward's GB300 targets as an uncalibrated GB200 proxy. Replace them with practical GB200 rates before comparing the two GPUs.",
      rates: referenceRates,
    },
    gb300: {
      label: "GB300",
      computeBasis: "gb300-practical-target",
      note: "Edward's practical GB300 targets: BF16 is empirically informed; MXFP8 assumes twice BF16, not a directly measured kernel rate.",
      rates: referenceRates,
    },
  };
  const defaults = {
    hardware: "gb200",
    world: 512,
    pp: 8,
    vp: 1,
    cp: 1,
    ep: 64,
    seq: 4096,
    rows: 1,
    mbs: 64,
    segments: 1,
    ac: "full",
    policy: "reference",
    precision: "bf16",
    routingMap: "none",
    kdaChunk: 64,
    domain: 64,
    copies: "assignment",
    ppLink: "network",
    ...hardwareProfiles.gb200.rates,
  };

  function gemmCost(gemm, config) {
    const { m, n, k } = gemm;
    const mxfp8 = !!gemm.routed && config.precision === "mxfp8-experts";
    const rate = mxfp8 ? config.fp8Gemm : config.gemm;
    const operandBytes = mxfp8 ? 1 + 1 / 32 : 2;
    const outputBytes = gemm.phase === "wgrad" ? 4 : 2;
    const flops = 2 * m * n * k;
    const bytes = operandBytes * (m * k + k * n) + outputBytes * m * n;
    const compute = flops / (rate * 1e12);
    const memory = bytes / (config.hbm * 1e12);
    return {
      ...gemm,
      precision: mxfp8 ? "MXFP8" : "BF16",
      rate,
      flops,
      bytes,
      intensity: flops / bytes,
      seconds: Math.max(compute, memory),
      bound: memory > compute ? "HBM" : "Compute",
    };
  }

  function gemmPhases(gemm, config) {
    const { m, n, k } = gemm;
    const phases = [
      { ...gemm, phase: "forward" },
      { ...gemm, m, n: k, k: n, phase: "dgrad" },
      { ...gemm, m: k, n, k: m, phase: "wgrad" },
    ];
    if (config.ac === "full" && !gemm.head) phases.push({ ...gemm, phase: "recompute" });
    return phases.map((phase) => gemmCost(phase, config));
  }

  function attentionWork(mla, config) {
    const segment = config.seq / config.segments;
    // Count causal pairs, not the square of the flattened multi-row token count.
    // Edward Yang's K3 sheet adds the intra-chunk terms to the recurrent state work.
    const forward = mla ? H * (192 + 128) * (segment + 1) : H * (6 * K * K + 10 * config.kdaChunk * K);
    const model = 3 * forward;
    const flashReplay = mla ? H * 192 * (segment + 1) : 0;
    const checkpoint = config.ac === "full" ? forward : 0;
    return { forward, model, flashReplay, checkpoint, issued: model + flashReplay + checkpoint };
  }

  function remoteCopies(config) {
    const local = Math.min(config.domain, config.ep);
    if (config.copies === "assignment") return { ranks: topK * (1 - 1 / config.ep), domains: topK * (1 - local / config.ep) };
    let miss = 1;
    for (let i = 0; i < topK; i++) miss *= (experts - experts / config.ep - i) / (experts - i);
    const hit = 1 - miss;
    return { ranks: (config.ep - 1) * hit, domains: (config.ep - local) * hit };
  }

  function collectiveBytes(parameters, size, local, dtype) {
    // NIC accounting imports each remote shard once per NVLink domain, then shares it locally.
    return {
      nvl: dtype * parameters * (1 - 1 / size),
      network: (dtype * parameters * (1 - local / size)) / local,
    };
  }

  function ppPacketBlocks(stage) {
    return stage.layers.length ? Math.ceil((stage.layers[stage.layers.length - 1].index + 1) / A.attnResBlockSize) : 0;
  }

  function virtualChunks(config) {
    const { pp, vp } = config;
    if (!Number.isInteger(pp) || !Number.isInteger(vp) || pp < 1 || vp < 1 || pp * vp > A.layers.length)
      throw new RangeError("PP and VPP must be positive integers with enough decoder layers for their chunks.");
    if (vp === 1 && (pp - 1) * Math.ceil(A.layers.length / pp) >= A.layers.length)
      throw new RangeError("The contiguous PP split cannot create empty stages.");
    if (vp === 1) return Array.from({ length: pp }, (_, rank) => ({ index: rank, rank, local: 0, owned: A.partition(pp, rank) }));
    const count = pp * vp;
    const effective = A.layers.length + 2;
    let start = 0;
    return Array.from({ length: count }, (_, index) => {
      // Match the memory study's automatic loop split: embedding and head each occupy one layer slot.
      const size = Math.floor(effective / count) + Number(index < effective % count) - Number(index === 0) - Number(index === count - 1);
      if (size < 1) throw new RangeError("The estimated VPP split cannot create empty decoder chunks.");
      const layers = A.layers.slice(start, start + size);
      start += size;
      const front = index === 0 ? A.embedding + A.vision : 0;
      const back = index === count - 1 ? A.embedding + 3 * D : 0;
      return {
        index,
        rank: index % pp,
        local: Math.floor(index / pp),
        owned: {
          layers,
          dense: sum(layers, (layer) => layer.dense) + front + back,
          routed: sum(layers, (layer) => layer.routed),
          adam: sum(layers, (layer) => layer.adam) + front + back,
          front,
          back,
        },
      };
    });
  }

  function pipelineEdges(chunks, config, tokens) {
    return chunks.slice(0, -1).map((chunk, index) => {
      const next = chunks[index + 1];
      const blocks = ppPacketBlocks(chunk.owned);
      const forwardBytes = tokens * D * 2 * (1 + blocks);
      return {
        sourceChunk: chunk.index,
        destinationChunk: next.index,
        sourceRank: chunk.rank,
        destinationRank: next.rank,
        remote: chunk.rank !== next.rank,
        blocks,
        forwardBytes,
        bytesPerDirection: config.mbs * forwardBytes,
      };
    });
  }

  function calculate(input = {}) {
    const hardware = input.hardware === undefined ? defaults.hardware : input.hardware;
    if (!Object.keys(hardwareProfiles).includes(hardware)) throw new RangeError("Unknown GPU hardware profile.");
    const profile = hardwareProfiles[hardware];
    // Explicit rates override the selected profile; the device name never applies a hidden speed multiplier.
    const config = { ...defaults, ...profile.rates, ...input, hardware };
    for (const key of ["world", "pp", "vp", "cp", "ep", "seq", "rows", "mbs", "segments", "domain", "kdaChunk"])
      if (!Number.isInteger(config[key]) || config[key] <= 0) throw new RangeError(`${key} must be a positive integer.`);
    if (config.vp === 1 && (config.pp - 1) * Math.ceil(A.layers.length / config.pp) >= A.layers.length)
      throw new RangeError("The contiguous PP split cannot create empty stages.");
    for (const [key, values] of Object.entries({
      ac: ["full", "none"],
      policy: ["reference", "reshard", "retain"],
      precision: ["bf16", "mxfp8-experts"],
      routingMap: ["none", "hybridep"],
      copies: ["assignment", "rank"],
      ppLink: ["network", "nvlink"],
    }))
      if (!values.includes(config[key])) throw new RangeError(`Unknown ${key} policy.`);
    if (!Number.isInteger(config.world / (config.pp * config.cp))) throw new RangeError("PP x CP must divide the configured GPU count.");
    const stageRanks = config.world / config.pp;
    if (experts % config.ep || stageRanks % config.ep) throw new RangeError("EP must divide 896 experts and the ranks of one PP stage.");
    for (const key of ["gemm", "fp8Gemm", "hbm", "nvl", "network"])
      if (!Number.isFinite(config[key]) || config[key] <= 0) throw new RangeError(`${key} must be a positive finite rate.`);
    if (config.seq % config.segments || config.seq % config.cp) throw new RangeError("Context must divide evenly into segments and CP shards.");
    if (config.domain > stageRanks || stageRanks % config.domain)
      throw new RangeError("Stage-local ranks per NVLink domain must divide the stage group.");
    const full = config.ac === "full";
    const tokens = (config.rows * config.seq) / config.cp;
    const chunks = virtualChunks(config);
    const ppEdges = pipelineEdges(chunks, config, tokens);
    const dp = stageRanks / config.cp;
    const globalTokens = dp * config.mbs * config.rows * config.seq;
    const expertFsdp = stageRanks / config.ep;
    const rounds =
      config.policy === "retain"
        ? { ag: 1, rs: 1 }
        : config.policy === "reshard"
          ? { ag: 2 * config.mbs, rs: config.mbs }
          : { ag: config.mbs + 1, rs: 1 };
    const copies = remoteCopies(config);
    const epExchanges = full ? 6 : 4;
    const metadataRounds = config.routingMap === "hybridep" ? (full ? 2 : 1) : 0;
    const metadataInput = tokens * experts;
    const metadataCall = {
      nvl: (config.ep - 1) * metadataInput,
      network: (config.ep / Math.min(config.domain, config.ep) - 1) * metadataInput,
    };
    const stages = Array.from({ length: config.pp }, (_, rank) => {
      const ownedChunks = chunks.filter((chunk) => chunk.rank === rank);
      const owned =
        config.vp === 1
          ? ownedChunks[0].owned
          : {
              layers: ownedChunks.flatMap((chunk) => chunk.owned.layers),
              ...Object.fromEntries(["dense", "routed", "adam", "front", "back"].map((key) => [key, sum(ownedChunks, (chunk) => chunk.owned[key])])),
            };
      const allGemms = owned.layers.flatMap((layer) => A.gemmsForLayer(layer, tokens, config.ep));
      if (rank === config.pp - 1) allGemms.push({ name: "LM head", m: tokens, n: 163840, k: D, groups: 1, head: true });
      const gemms = allGemms.map((gemm) => {
        const phases = gemmPhases(gemm, config);
        return { ...gemmCost(gemm, config), phases, passes: phases.length };
      });
      const learnedModelFlops = config.mbs * sum(gemms, (gemm) => 3 * gemm.flops * gemm.groups);
      const learnedIssuedFlops = config.mbs * sum(gemms, (gemm) => gemm.passes * gemm.flops * gemm.groups);
      const kda = owned.layers.filter((layer) => !layer.mla).length;
      const mla = owned.layers.length - kda;
      const kdaWork = attentionWork(false, config);
      const mlaWork = attentionWork(true, config);
      const coreModelFlops = config.mbs * tokens * (kda * kdaWork.model + mla * mlaWork.model);
      const coreIssuedFlops = config.mbs * tokens * (kda * kdaWork.issued + mla * mlaWork.issued);
      const gemmSeconds = config.mbs * sum(gemms, (gemm) => gemm.groups * sum(gemm.phases, (phase) => phase.seconds));
      const computeSeconds = gemmSeconds + coreIssuedFlops / (config.gemm * 1e12);
      const moeLayers = owned.layers.filter((layer) => layer.index !== 0).length;
      const denseLocal = Math.min(config.domain, stageRanks);
      const expertLocal = Math.min(expertFsdp, Math.max(1, config.domain / config.ep));
      const denseAG = collectiveBytes(owned.dense, stageRanks, denseLocal, 2);
      const denseRS = collectiveBytes(owned.dense, stageRanks, denseLocal, 4);
      const expertAG = collectiveBytes(owned.routed / config.ep, expertFsdp, expertLocal, 2);
      const expertRS = collectiveBytes(owned.routed / config.ep, expertFsdp, expertLocal, 4);
      const calls = config.mbs * moeLayers;
      const payload = 2 * Z * tokens * epExchanges * calls;
      let before = null;
      let after = null;
      let ppBytes;
      if (config.vp === 1) {
        // Preserve the original single-stage accounting, including its operation order.
        before = rank === 0 ? 0 : ppPacketBlocks(A.partition(config.pp, rank - 1));
        after = rank === config.pp - 1 ? 0 : ppPacketBlocks(owned);
        const packetUnits = (rank > 0 ? 1 + before : 0) + (rank < config.pp - 1 ? 1 + after : 0);
        ppBytes = config.mbs * tokens * D * 2 * packetUnits;
      } else {
        // Charge forward at the source and a matching full-gradient packet at the destination; no residual-cache reuse.
        ppBytes = sum(
          ppEdges.filter((edge) => edge.remote && (edge.sourceRank === rank || edge.destinationRank === rank)),
          (edge) => edge.bytesPerDirection
        );
      }
      const traffic = [
        { key: "payload", label: "Expert token payload", nvl: payload * copies.ranks, network: payload * copies.domains },
        {
          key: "metadata",
          label: config.routingMap === "hybridep" ? "HybridEP routing map (optional)" : "Global routing map (not selected)",
          nvl: calls * metadataRounds * metadataCall.nvl,
          network: calls * metadataRounds * metadataCall.network,
        },
        {
          key: "fsdp",
          label: "FSDP weights + gradients",
          nvl: rounds.ag * (denseAG.nvl + expertAG.nvl) + rounds.rs * (denseRS.nvl + expertRS.nvl),
          network: rounds.ag * (denseAG.network + expertAG.network) + rounds.rs * (denseRS.network + expertRS.network),
        },
        {
          key: "pipeline",
          label: "PP hidden + AttnRes",
          nvl: config.ppLink === "nvlink" ? ppBytes : 0,
          network: config.ppLink === "network" ? ppBytes : 0,
        },
      ];
      const nvlSeconds = sum(traffic, (item) => item.nvl) / (config.nvl * 1e9);
      const networkSeconds = sum(traffic, (item) => item.network) / (config.network * 1e9);
      const stored = (owned.dense + owned.routed) / stageRanks;
      const adam = owned.adam / stageRanks;
      const stateBytes = 12 * (stored - adam) + 16 * adam;
      const small = Math.min(Z, F);
      const large = Math.max(Z, F);
      const nsPerMatrix = 5 * (4 * small * small * large + 2 * small ** 3);
      const nsMatrices = moeLayers * (experts / config.ep) * 3;
      // Update-work ownership is idealized; redistribution can rebalance it in a real optimizer.
      const nsFlops = (nsMatrices * nsPerMatrix) / expertFsdp;
      return {
        rank,
        owned,
        chunks: ownedChunks,
        gemms,
        kda,
        mla,
        moeLayers,
        learnedModelFlops,
        learnedIssuedFlops,
        coreModelFlops,
        coreIssuedFlops,
        gemmSeconds,
        computeSeconds,
        traffic,
        nvlSeconds,
        networkSeconds,
        stateBytes,
        before,
        after,
        ppBytes,
        nsPerMatrix,
        nsMatrices,
        nsFlops,
      };
    });
    const computeStage = stages.reduce((best, stage) => (stage.computeSeconds > best.computeSeconds ? stage : best));
    const nvlStage = stages.reduce((best, stage) => (stage.nvlSeconds > best.nvlSeconds ? stage : best));
    const networkStage = stages.reduce((best, stage) => (stage.networkSeconds > best.networkSeconds ? stage : best));
    const computeSeconds = computeStage.computeSeconds;
    const nvlSeconds = nvlStage.nvlSeconds;
    const networkSeconds = networkStage.networkSeconds;
    const resourceSeconds = Math.max(computeSeconds, nvlSeconds, networkSeconds);
    const bubbleRatio = config.vp === 1 ? (config.pp - 1) / config.mbs : (config.pp - 1) / (config.mbs * config.vp);
    const bubbleSeconds = computeSeconds * bubbleRatio;
    const sketchSeconds = resourceSeconds + bubbleSeconds;
    const causalModelFlops = stageRanks * sum(stages, (stage) => stage.learnedModelFlops + stage.coreModelFlops);
    const issuedFlops = stageRanks * sum(stages, (stage) => stage.learnedIssuedFlops + stage.coreIssuedFlops);
    const conventionPerToken = 6 * A.activeParams + 69 * 18 * H * K * K + 24 * 6 * H * (192 + 128) * config.seq;
    const tokensPerGPUUpdate = globalTokens / config.world;
    // Neither measured throughput nor backend-specific communication sets the compute ceiling.
    const computeTokensPerSecondPerGPU = tokensPerGPUUpdate / computeSeconds;
    const resourceTokensPerSecondPerGPU = tokensPerGPUUpdate / sketchSeconds;
    const forwardGemmPoints = [...A.gemmsForLayer(A.layers[1], tokens, config.ep), ...A.gemmsForLayer(A.layers[3], tokens, config.ep)]
      .filter((gemm) =>
        ["KDA Q projection", "KDA decay expansion", "Latent compression", "Router", "MLA Q expansion", "Routed gate/up", "Routed down"].includes(
          gemm.name
        )
      )
      .filter((gemm, index, list) => list.findIndex((other) => other.name === gemm.name) === index)
      .map((gemm) => gemmCost(gemm, config));
    return {
      config,
      pipeline: {
        estimated: config.vp > 1,
        schedule: config.vp === 1 ? "ordinary-1f1b" : "interleaved-1f1b",
        placement: config.vp === 1 ? "legacy-contiguous" : "automatic-loop",
        bubbleFormula: config.vp === 1 ? "(PP - 1) / M" : "(PP - 1) / (M * VPP)",
        payloadPolicy: "full-hidden-and-residual-banks",
        residualCacheReuse: false,
      },
      hardware: {
        profile: config.hardware,
        label: profile.label,
        computeBasis: profile.computeBasis,
        note: profile.note,
        source: "https://deepseek-v3.ezyang.com/studies/kimi-k3.html",
        referenceRates: profile.rates,
        overriddenRates: rateKeys.filter((key) => config[key] !== profile.rates[key]),
        networkBasis: "Editable per-GPU system budget; 100 GB/s is the GB300 reference assumption, not a measured allocation rate.",
      },
      dp,
      stageRanks,
      expertFsdp,
      tokens,
      globalTokens,
      tokensPerGPUUpdate,
      computeTokensPerSecondPerGPU,
      computeModelTFLOPsPerGPU: (computeTokensPerSecondPerGPU * conventionPerToken) / 1e12,
      resourceTokensPerSecondPerGPU,
      resourceModelTFLOPsPerGPU: (resourceTokensPerSecondPerGPU * conventionPerToken) / 1e12,
      globalRows: dp * config.mbs * config.rows,
      rounds,
      copies,
      epExchanges,
      metadataRounds,
      metadataInput,
      metadataCall,
      chunks,
      ppEdges,
      stages,
      computeStage,
      nvlStage,
      networkStage,
      computeSeconds,
      nvlSeconds,
      networkSeconds,
      resourceSeconds,
      bubbleRatio,
      bubbleSeconds,
      sketchSeconds,
      causalModelFlops,
      issuedFlops,
      conventionPerToken,
      forwardGemmPoints,
      expertRows: (config.ep * tokens * topK) / experts,
      localExperts: experts / config.ep,
    };
  }

  const api = { defaults, hardwareProfiles, rateKeys, calculate, virtualChunks, gemmCost, gemmPhases, attentionWork, collectiveBytes, remoteCopies };
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.KimiK3Roofline = api;
})(typeof window === "undefined" ? globalThis : window);
