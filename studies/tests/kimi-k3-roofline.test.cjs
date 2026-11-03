const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { createHash } = require("node:crypto");
const A = require("../kimi-k3-accounting.js");
const R = require("../kimi-k3-roofline-model.js");
const M = require("../kimi-k3-memory-model.js");
const legacyVPP1 = require("./fixtures/kimi-k3-roofline-vp1.json");
const sum = (items, get) => items.reduce((total, item, index) => total + get(item, index), 0);
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) <= 1e-12 * Math.max(1, Math.abs(expected)), `${actual} != ${expected}`);

test("VPP1 preserves the complete pre-VPP calculation across frozen representative recipes", () => {
  assert.equal(legacyVPP1.sourceCommit, "5903634fb9e3376dae7b4c6289430e2b1fcf80f2");
  for (const { input, sha256 } of legacyVPP1.cases) {
    for (const config of [input, { ...input, vp: 1 }]) {
      const result = JSON.parse(JSON.stringify(R.calculate(config)));
      // Remove only the added VPP metadata, not any existing calculation or parameter/traffic detail.
      delete result.config.vp;
      delete result.pipeline;
      delete result.chunks;
      delete result.ppEdges;
      for (const stage of [...result.stages, result.computeStage, result.nvlStage, result.networkStage]) delete stage.chunks;
      assert.equal(createHash("sha256").update(JSON.stringify(result)).digest("hex"), sha256, JSON.stringify(config));
    }
  }
});

test("estimated loop chunks agree with the existing memory model's placement and parameter ownership", () => {
  for (const pp of [1, 2, 4, 8, 16]) {
    for (const vp of [1, 2, 4]) {
      if (pp === 16 && vp === 4) continue;
      const config = M.normalize({ pp, vp, stage: 0, ep: Math.min(64, 512 / pp), sched: vp > 1 ? "interleaved" : "1f1b" });
      const expected = M.partitionChunks(config);
      const chunks = R.virtualChunks(config);
      assert.equal(chunks.length, pp * vp);
      assert.deepEqual(
        chunks.flatMap((chunk) => chunk.owned.layers.map((layer) => layer.index)),
        Array.from({ length: 93 }, (_, i) => i)
      );
      for (let index = 0; index < chunks.length; index++) {
        const chunk = chunks[index];
        assert.equal(chunk.index, expected[index].index);
        assert.equal(chunk.rank, index % pp);
        assert.equal(chunk.local, Math.floor(index / pp));
        assert.deepEqual(chunk.owned.layers, expected[index].layers);
        for (const key of ["dense", "routed", "adam", "front", "back"]) assert.equal(chunk.owned[key], expected[index][key]);
      }
    }
  }
});

test("VPP estimates change ownership and nominal bubbles without dividing work, batch, FSDP degree or resident state", () => {
  for (const precision of ["bf16", "mxfp8-experts"]) {
    const base = R.calculate({ pp: 4, mbs: 32, precision });
    for (const vp of [2, 4]) {
      const result = R.calculate({ pp: 4, mbs: 32, precision, vp });
      assert.equal(result.pipeline.estimated, true);
      assert.equal(result.pipeline.schedule, "interleaved-1f1b");
      assert.equal(result.pipeline.placement, "automatic-loop");
      assert.equal(result.globalTokens, base.globalTokens);
      assert.equal(result.tokens, base.tokens);
      assert.equal(result.stageRanks, base.stageRanks);
      assert.equal(result.expertFsdp, base.expertFsdp);
      assert.equal(result.conventionPerToken, base.conventionPerToken);
      close(result.causalModelFlops, base.causalModelFlops);
      close(result.issuedFlops, base.issuedFlops);
      close(
        sum(result.stages, (stage) => stage.computeSeconds),
        sum(base.stages, (stage) => stage.computeSeconds)
      );
      close(
        sum(result.stages, (stage) => stage.stateBytes),
        sum(base.stages, (stage) => stage.stateBytes)
      );
      assert.equal(
        sum(result.stages, (stage) => stage.owned.dense + stage.owned.routed),
        A.totalParams
      );
      assert.equal(result.bubbleRatio, (4 - 1) / (32 * vp));
      assert.equal(result.bubbleSeconds, result.computeSeconds * result.bubbleRatio);
      for (const stage of result.stages) assert.equal(stage.chunks.length, vp);
      assert.equal(result.resourceTokensPerSecondPerGPU, result.tokensPerGPUUpdate / (result.resourceSeconds + result.bubbleSeconds));
    }
  }
});

test("VPP PP traffic charges all logical edges, wraparound and both directions without residual-cache reuse", () => {
  const result = R.calculate({ pp: 4, mbs: 32, vp: 2 });
  assert.equal(result.ppEdges.length, 7);
  assert.equal(result.pipeline.residualCacheReuse, false);
  assert.deepEqual(
    result.ppEdges.map((edge) => edge.blocks),
    [1, 2, 3, 4, 5, 6, 7]
  );
  const wrap = result.ppEdges.find((edge) => edge.sourceRank === 3 && edge.destinationRank === 0);
  assert.equal(wrap.sourceChunk, 3);
  assert.equal(wrap.destinationChunk, 4);
  assert.equal(wrap.forwardBytes, 4096 * 7168 * 2 * 5);
  assert.equal(result.stages[0].ppBytes, 32 * 4096 * 7168 * 2 * (2 + 5 + 6));
  assert.equal(
    sum(result.stages, (stage) => stage.ppBytes),
    2 * sum(result.ppEdges, (edge) => edge.bytesPerDirection)
  );
  const local = R.calculate({ pp: 1, vp: 4 });
  assert.ok(local.ppEdges.every((edge) => !edge.remote));
  assert.equal(local.stages[0].ppBytes, 0);
  assert.equal(local.bubbleRatio, 0);
  close(local.computeSeconds, R.calculate({ pp: 1 }).computeSeconds);
});

test("estimated VPP rejects splits with empty decoder chunks", () => {
  assert.throws(() => R.calculate({ pp: 16, vp: 4, ep: 32, domain: 32 }), /empty decoder chunks/);
  assert.throws(() => R.virtualChunks({ pp: 32, vp: 1 }), /empty stages/);
});

test("shared accounting matches the architecture and FP32 memory studies", () => {
  assert.equal(A.totalParams, 2779931738208);
  assert.equal(A.activeParams, 104189515872);
  assert.equal(A.linearParams, 104174092288);
  assert.equal(A.muonParams, 2777072664576);
  assert.equal(A.adamParams, 2859073632);
  assert.equal(A.layers.filter((layer) => layer.mla).length, 24);
  assert.equal(A.layers.filter((layer) => !layer.mla).length, 69);
  for (const pp of [1, 2, 4, 8, 16]) {
    const stages = Array.from({ length: pp }, (_, rank) => A.partition(pp, rank));
    assert.equal(
      sum(stages, (stage) => stage.dense + stage.routed),
      A.totalParams
    );
    assert.equal(
      sum(stages, (stage) => stage.adam),
      A.adamParams
    );
    assert.equal(
      sum(stages, (stage) => stage.layers.length),
      93
    );
  }
  const x = R.calculate();
  close(x.stages[3].stateBytes / 2 ** 30, 63.23320531100035);
  close(x.stageRanks * sum(x.stages, (stage) => stage.stateBytes), 12 * A.muonParams + 16 * A.adamParams);
});

test("6ND learned-matrix work is conserved across PP, CP and EP", () => {
  for (const pp of [1, 2, 4, 8, 16]) {
    for (const cp of [1, 4, 16]) {
      for (const ep of [1, 8, 32]) {
        const x = R.calculate({ pp, cp, ep, domain: Math.min(64, 512 / pp) });
        close(x.stageRanks * sum(x.stages, (stage) => stage.learnedModelFlops), 6 * A.linearParams * x.globalTokens);
        assert.ok(x.computeSeconds > 0 && Number.isFinite(x.sketchSeconds));
      }
    }
  }
});

test("EP64 receives the input rows of all EP ranks, not just one input rank", () => {
  const ep64 = R.calculate();
  const ep1 = R.calculate({ ep: 1 });
  close(ep64.expertRows, 4681.142857142857);
  close(ep1.expertRows, 73.14285714285714);
  assert.equal(ep64.localExperts, 14);
  assert.equal(ep1.localExperts, 896);
  close(ep64.expertRows * ep64.localExperts, 4096 * 16);
  close(ep1.expertRows * ep1.localExperts, 4096 * 16);
  const routed = (x) => x.forwardGemmPoints.find((gemm) => gemm.name === "Routed gate/up");
  assert.equal(routed(ep64).bound, "Compute");
  assert.equal(routed(ep1).bound, "HBM");
  assert.ok(routed(ep64).intensity > routed(ep1).intensity);
});

test("the BF16 roofline uses FLOPs, operand bytes and dense hardware rates", () => {
  const x = R.calculate();
  const gemm = x.forwardGemmPoints.find((item) => item.name === "KDA Q projection");
  assert.equal(gemm.flops, 2 * 4096 * 12288 * 7168);
  assert.equal(gemm.bytes, 2 * (4096 * 7168 + 7168 * 12288 + 4096 * 12288));
  close(gemm.intensity, 2150.4);
  close(gemm.seconds, Math.max(gemm.flops / 1800e12, gemm.bytes / 8e12));
  assert.equal(x.config.gemm / x.config.hbm, 225);
});

test("two 2K rows preserve tokenwise work but change causal MLA pairs", () => {
  const two = R.calculate({ seq: 2048, rows: 2 });
  const one = R.calculate();
  assert.equal(two.tokens, one.tokens);
  assert.equal(two.globalTokens, one.globalTokens);
  assert.equal(two.stages[3].learnedModelFlops, one.stages[3].learnedModelFlops);
  close(R.attentionWork(true, one.config).model / R.attentionWork(true, two.config).model, 4097 / 2049);
  assert.equal(R.attentionWork(false, one.config).model, R.attentionWork(false, two.config).model);
});

test("causal MLA counts actual equal-document pairs and separate FlashAttention replay", () => {
  const x = R.calculate();
  const core = R.attentionWork(true, x.config);
  assert.equal(core.model, 3 * 96 * (192 + 128) * 4097);
  assert.equal(core.flashReplay, 96 * 192 * 4097);
  assert.equal(core.checkpoint, 96 * (192 + 128) * 4097);
  const packed = R.attentionWork(true, { ...x.config, segments: 4 });
  close(core.model / packed.model, 4097 / 1025);
  assert.equal(R.calculate({ segments: 4 }).conventionPerToken, x.conventionPerToken);
});

test("the fixed conventional FLOP metric reproduces the reported 4K TFLOP/s", () => {
  const x = R.calculate();
  assert.equal(x.conventionPerToken, 645209985600);
  close((370 * x.conventionPerToken) / 1e12, 238.727694672);
  close((100 * 370 * x.conventionPerToken) / 1e12 / 2500, 9.54910778688);
  assert.ok(x.causalModelFlops / x.globalTokens < x.conventionPerToken);
});

test("full checkpoint adds decoder forwards but not an extra output-head GEMM", () => {
  const full = R.calculate({ routingMap: "hybridep" });
  const none = R.calculate({ ac: "none", routingMap: "hybridep" });
  const extraMatrices = full.stageRanks * sum(full.stages, (stage, rank) => stage.learnedIssuedFlops - none.stages[rank].learnedIssuedFlops);
  close(extraMatrices, 2 * (A.linearParams - A.embedding) * full.globalTokens);
  assert.equal(full.epExchanges, 6);
  assert.equal(none.epExchanges, 4);
  assert.equal(full.metadataRounds, 2);
  assert.equal(none.metadataRounds, 1);
  const traffic = (x, key) => x.stages[3].traffic.find((item) => item.key === key).nvl;
  close(traffic(full, "payload") / traffic(none, "payload"), 1.5);
  close(traffic(full, "metadata") / traffic(none, "metadata"), 2);
});

test("64K CP16 keeps 4K local tokens, 16.78M global tokens and FSDP64", () => {
  const x = R.calculate({ seq: 65536, cp: 16 });
  assert.equal(x.tokens, 4096);
  assert.equal(x.dp, 4);
  assert.equal(x.stageRanks, 64);
  assert.equal(x.expertFsdp, 1);
  assert.equal(x.globalTokens, 16777216);
  assert.equal(x.globalRows, 256);
  assert.equal(x.stages[3].stateBytes, R.calculate().stages[3].stateBytes);
  assert.ok(x.stages[3].coreIssuedFlops > R.calculate().stages[3].coreIssuedFlops);
});

test("EP and dense FSDP inside one assumed NVLink domain have no scale-out collective budget", () => {
  const x = R.calculate({ routingMap: "hybridep" });
  for (const stage of x.stages) for (const item of stage.traffic) if (item.key !== "pipeline") assert.equal(item.network, 0);
  const split = R.calculate({ domain: 8, routingMap: "hybridep" });
  for (const key of ["payload", "metadata", "fsdp"]) assert.ok(split.stages[3].traffic.find((item) => item.key === key).network > 0);
  assert.equal(x.copies.ranks, 15.75);
  assert.equal(x.copies.domains, 0);
  assert.equal(split.copies.domains, 14);
});

test("optional rank deduplication is bounded by the assignment budget", () => {
  for (const ep of [1, 8, 16, 32, 64, 128]) {
    const c = { ...R.defaults, ep, domain: Math.min(8, ep) };
    const rank = R.remoteCopies({ ...c, copies: "rank" });
    const assignment = R.remoteCopies(c);
    assert.ok(rank.ranks <= assignment.ranks && rank.ranks >= 0);
    assert.ok(rank.domains <= assignment.domains && rank.domains >= 0);
  }
  assert.equal(R.remoteCopies({ ...R.defaults, ep: 1, copies: "rank" }).ranks, 0);
});

test("the routing-map collective uses one byte per Boolean, not top-k integers", () => {
  const x = R.calculate({ routingMap: "hybridep" });
  assert.equal(x.metadataInput, 3670016);
  assert.equal((x.metadataInput * x.config.ep) / 2 ** 20, 224);
  assert.equal(x.metadataCall.nvl / 2 ** 20, 220.5);
  const stage = x.stages[3];
  assert.equal(x.config.mbs * stage.moeLayers * x.metadataRounds, 1536);
  assert.equal(stage.traffic.find((item) => item.key === "metadata").nvl, 1536 * 220.5 * 2 ** 20);
});

test("FSDP round counts distinguish all four parameter and gradient lifecycles", () => {
  assert.deepEqual(R.calculate().rounds, { ag: 65, rs: 1 });
  assert.deepEqual(R.calculate({ policy: "reshard" }).rounds, { ag: 128, rs: 64 });
  assert.deepEqual(R.calculate({ policy: "retain" }).rounds, { ag: 1, rs: 1 });
  assert.deepEqual(R.calculate({ policy: "reshard-deferred" }).rounds, { ag: 128, rs: 1 });
  assert.equal(R.calculate().expertFsdp, 1);
  const bytes = R.collectiveBytes(100, 1, 1, 2);
  assert.deepEqual(bytes, { nvl: 0, network: 0 });
  const hierarchical = R.collectiveBytes(100, 64, 8, 2);
  assert.equal(hierarchical.nvl, (200 * 63) / 64);
  assert.equal(hierarchical.network, (200 * 7) / 8 / 8);
});

test("deferred reduction keeps two gathers per microbatch without multiplying rounds by VPP", () => {
  for (const mbs of [1, 8, 32, 64]) {
    for (const vp of [1, 2, 4]) {
      const result = R.calculate({ hardware: "gb300", world: 256, pp: 4, vp, mbs, policy: "reshard-deferred" });
      assert.deepEqual(result.rounds, { ag: 2 * mbs, rs: 1 });
      assert.equal(result.expertFsdp, 1);
      assert.equal(result.globalTokens, 64 * mbs * 4096);
    }
  }
});

test("reshard-deferred removes only repeated gradient reductions from the fully resharded traffic budget", () => {
  for (const world of [256, 512]) {
    for (const vp of [1, 2, 4]) {
      for (const domain of [8, 64]) {
        const input = { hardware: "gb300", world, pp: 4, vp, mbs: 32, domain, precision: "mxfp8-experts" };
        const result = R.calculate({ ...input, policy: "reshard-deferred" });
        const perMicrobatch = R.calculate({ ...input, policy: "reshard" });
        const reference = R.calculate({ ...input, policy: "reference" });
        assert.equal(result.globalTokens, perMicrobatch.globalTokens);
        assert.equal(result.issuedFlops, perMicrobatch.issuedFlops);
        assert.equal(result.computeSeconds, perMicrobatch.computeSeconds);
        for (const stage of result.stages) {
          const control = perMicrobatch.stages[stage.rank];
          const denseRS = R.collectiveBytes(stage.owned.dense, result.stageRanks, Math.min(domain, result.stageRanks), 4);
          const expertRS = R.collectiveBytes(
            stage.owned.routed / result.config.ep,
            result.expertFsdp,
            Math.min(result.expertFsdp, Math.max(1, domain / result.config.ep)),
            4
          );
          assert.equal(stage.stateBytes, control.stateBytes);
          assert.deepEqual(stage.gemms, control.gemms);
          for (const item of stage.traffic) {
            const original = control.traffic.find((entry) => entry.key === item.key);
            if (item.key !== "fsdp") {
              assert.deepEqual(item, original);
              continue;
            }
            const oldReference = reference.stages[stage.rank].traffic.find((entry) => entry.key === "fsdp");
            for (const link of ["nvl", "network"]) {
              close(original[link] - item[link], 31 * (denseRS[link] + expertRS[link]));
              close(item[link], (64 + 2) * ((denseRS[link] + expertRS[link]) / 2));
              assert.ok(item[link] <= original[link]);
              assert.ok(item[link] >= oldReference[link]);
            }
          }
        }
      }
    }
  }
});

test("PP sends hidden plus the full first-visit residual stack", () => {
  const x = R.calculate();
  assert.deepEqual(
    x.stages.slice(0, -1).map((stage) => stage.after),
    [1, 2, 3, 4, 5, 6, 7]
  );
  assert.equal(A.attnResBlockSize, 12);
  assert.equal(((1 + x.stages[6].after) * x.tokens * A.D * 2) / 2 ** 20, 448);
  assert.equal(x.stages[3].ppBytes, 64 * 4096 * 7168 * 2 * (1 + 3 + (1 + 4)));
  assert.equal(R.calculate({ pp: 1 }).stages[0].ppBytes, 0);
  const nvlink = R.calculate({ ppLink: "nvlink" });
  assert.equal(nvlink.stages[3].traffic.find((item) => item.key === "pipeline").network, 0);
  assert.equal(nvlink.stages[3].traffic.find((item) => item.key === "pipeline").nvl, x.stages[3].ppBytes);
  assert.equal(x.bubbleRatio, 7 / 64);
});

test("Muon NS estimate counts three logical matrices, not one fused storage tensor", () => {
  const stage = R.calculate().stages[3];
  assert.equal(stage.nsPerMatrix, 5 * (4 * 3072 ** 2 * 3584 + 2 * 3072 ** 3));
  assert.equal(stage.nsMatrices, 504);
  assert.equal(stage.nsFlops, 504 * stage.nsPerMatrix);
});

test("practical SOL defaults reproduce Edward's current Blackwell K3 rate inputs", () => {
  assert.equal(R.defaults.hardware, "gb200");
  assert.equal(R.defaults.gemm, 1800);
  assert.equal(R.defaults.fp8Gemm, 3600);
  assert.equal(R.defaults.hbm, 8);
  assert.equal(R.defaults.nvl, 900);
  assert.equal(R.defaults.network, 100);
});

test("GB200 and GB300 profiles distinguish the uncalibrated proxy from the practical GB300 targets", () => {
  assert.deepEqual(Object.keys(R.hardwareProfiles), ["gb200", "gb300"]);
  assert.deepEqual(R.rateKeys, ["gemm", "fp8Gemm", "hbm", "nvl", "network"]);
  const gb200 = R.calculate();
  const gb300 = R.calculate({ hardware: "gb300" });
  assert.deepEqual(R.calculate({ hardware: undefined }), gb200);
  assert.equal(gb200.hardware.profile, "gb200");
  assert.equal(gb200.hardware.computeBasis, "gb300-reference-proxy");
  assert.match(gb200.hardware.note, /uncalibrated GB200 proxy/);
  assert.equal(gb300.hardware.profile, "gb300");
  assert.equal(gb300.hardware.computeBasis, "gb300-practical-target");
  assert.match(gb300.hardware.note, /not a directly measured kernel rate/);
  for (const result of [gb200, gb300]) {
    assert.equal(result.config.world, 512);
    assert.deepEqual(result.hardware.overriddenRates, []);
    assert.match(result.hardware.networkBasis, /not a measured allocation rate/);
    assert.equal(result.hardware.source, "https://deepseek-v3.ezyang.com/studies/kimi-k3.html");
    for (const key of R.rateKeys) assert.equal(result.config[key], R.hardwareProfiles[result.config.hardware].rates[key]);
  }
});

test("identical reference rates give identical modeled times, without a hidden GPU-generation multiplier", () => {
  for (const input of [{}, { world: 256, pp: 4, mbs: 32, precision: "mxfp8-experts", domain: 8, network: 50 }]) {
    const gb200 = R.calculate({ ...input, hardware: "gb200" });
    const gb300 = R.calculate({ ...input, hardware: "gb300" });
    for (const key of ["globalTokens", "issuedFlops", "conventionPerToken", "computeSeconds", "nvlSeconds", "networkSeconds", "sketchSeconds"])
      assert.equal(gb200[key], gb300[key], key);
    assert.equal(gb200.resourceTokensPerSecondPerGPU, gb300.resourceTokensPerSecondPerGPU);
    assert.deepEqual(gb200.stages, gb300.stages);
  }
});

test("explicit hardware-rate overrides are used and identified without changing FLOP or token counts", () => {
  const base = R.calculate({ hardware: "gb200" });
  // Sensitivity-test inputs, not claimed GB200 measurements.
  const input = { hardware: "gb200", gemm: 1234.5, fp8Gemm: 2469, hbm: 6.2, nvl: 825, network: 50 };
  const custom = R.calculate(input);
  assert.equal(custom.globalTokens, base.globalTokens);
  assert.equal(custom.issuedFlops, base.issuedFlops);
  assert.equal(custom.conventionPerToken, base.conventionPerToken);
  assert.ok(custom.computeSeconds > base.computeSeconds);
  assert.ok(custom.resourceTokensPerSecondPerGPU < base.resourceTokensPerSecondPerGPU);
  assert.deepEqual(custom.hardware.overriddenRates, R.rateKeys);
  for (const key of R.rateKeys) assert.equal(custom.config[key], input[key]);
  assert.deepEqual(R.calculate(JSON.parse(JSON.stringify(custom.config))), custom);
});

test("KDA core includes the reference's recurrent and intra-chunk contractions", () => {
  const x = R.calculate();
  const core = R.attentionWork(false, x.config);
  assert.equal(x.config.kdaChunk, 64);
  assert.equal(core.forward, 17301504);
  assert.equal(core.model, 3 * 17301504);
  assert.equal(core.flashReplay, 0);
  assert.equal(core.checkpoint, 17301504);
  assert.equal(69 * core.model, 3581411328);
  const largerChunk = R.attentionWork(false, { ...x.config, kdaChunk: 128 });
  assert.equal(largerChunk.forward - core.forward, 96 * 10 * 64 * 128);
  // The published training-log convention remains unchanged by a runtime correction.
  assert.equal(R.calculate({ kdaChunk: 128 }).conventionPerToken, x.conventionPerToken);
});

test("compute service and recipe throughput are derived from hardware and configuration, not measurements", () => {
  const x = R.calculate();
  assert.equal(x.tokensPerGPUUpdate, 32768);
  close(x.computeTokensPerSecondPerGPU, x.globalTokens / (512 * x.computeSeconds));
  close(x.computeModelTFLOPsPerGPU, (x.computeTokensPerSecondPerGPU * x.conventionPerToken) / 1e12);
  close(x.resourceTokensPerSecondPerGPU, x.globalTokens / (512 * x.sketchSeconds));
  close(x.resourceModelTFLOPsPerGPU, (x.resourceTokensPerSecondPerGPU * x.conventionPerToken) / 1e12);
  assert.ok(x.computeTokensPerSecondPerGPU > x.resourceTokensPerSecondPerGPU);
  const source = readFileSync(join(__dirname, "..", "kimi-k3-roofline-model.js"), "utf8");
  assert.doesNotMatch(source, /throughput:\s*(?:370|372|297)|FakePG/);
});

test("backend metadata, topology and communication policy change recipe constraints, not compute work", () => {
  const x = R.calculate();
  assert.equal(x.metadataRounds, 0);
  assert.ok(x.stages.every((stage) => stage.traffic.find((item) => item.key === "metadata").nvl === 0));
  for (const input of [
    { routingMap: "hybridep" },
    { routingMap: "hybridep", domain: 8, network: 10 },
    { copies: "rank", policy: "retain", ppLink: "nvlink", nvl: 25 },
    { fp8Gemm: 1000 },
  ]) {
    const other = R.calculate(input);
    assert.equal(other.computeSeconds, x.computeSeconds);
    assert.equal(other.computeTokensPerSecondPerGPU, x.computeTokensPerSecondPerGPU);
  }
  assert.ok(R.calculate({ routingMap: "hybridep", domain: 8, network: 10 }).resourceTokensPerSecondPerGPU < x.resourceTokensPerSecondPerGPU);
  assert.ok(R.calculate({ gemm: 1200 }).computeTokensPerSecondPerGPU < x.computeTokensPerSecondPerGPU);
});

test("MXFP8 changes only routed GEMM pricing, never the model's FLOP or token counts", () => {
  const bf16 = R.calculate();
  const mx = R.calculate({ precision: "mxfp8-experts" });
  assert.equal(mx.globalTokens, bf16.globalTokens);
  assert.equal(mx.causalModelFlops, bf16.causalModelFlops);
  assert.equal(mx.issuedFlops, bf16.issuedFlops);
  assert.ok(mx.computeTokensPerSecondPerGPU > bf16.computeTokensPerSecondPerGPU);
  for (const gemm of mx.stages.flatMap((stage) => stage.gemms)) {
    assert.equal(gemm.precision, gemm.routed ? "MXFP8" : "BF16");
    assert.equal(gemm.rate, gemm.routed ? 3600 : 1800);
  }
  const routed = mx.forwardGemmPoints.find((gemm) => gemm.name === "Routed gate/up");
  close(routed.bytes, (1 + 1 / 32) * (routed.m * routed.k + routed.k * routed.n) + 2 * routed.m * routed.n);
  close(routed.seconds, Math.max(routed.flops / 3600e12, routed.bytes / 8e12));
});

test("forward, DGRAD and WGRAD have their own shapes and output-byte budgets", () => {
  const x = R.calculate({ precision: "mxfp8-experts" });
  const routed = x.stages[3].gemms.find((gemm) => gemm.name === "Routed gate/up");
  assert.deepEqual(
    routed.phases.map((phase) => phase.phase),
    ["forward", "dgrad", "wgrad", "recompute"]
  );
  const [forward, dgrad, wgrad, replay] = routed.phases;
  assert.deepEqual([dgrad.m, dgrad.n, dgrad.k], [forward.m, forward.k, forward.n]);
  assert.deepEqual([wgrad.m, wgrad.n, wgrad.k], [forward.k, forward.n, forward.m]);
  assert.equal(dgrad.flops, forward.flops);
  assert.equal(wgrad.flops, forward.flops);
  close(wgrad.bytes, (1 + 1 / 32) * (wgrad.m * wgrad.k + wgrad.k * wgrad.n) + 4 * wgrad.m * wgrad.n);
  assert.equal(replay.seconds, forward.seconds);
  const head = x.stages.at(-1).gemms.find((gemm) => gemm.head);
  assert.deepEqual(
    head.phases.map((phase) => phase.phase),
    ["forward", "dgrad", "wgrad"]
  );
  assert.equal(head.precision, "BF16");
});

test("every calculator input exists and recipe-specific results agree with the default", () => {
  const html = readFileSync(join(__dirname, "..", "kimi-k3-roofline.html"), "utf8");
  for (const key of Object.keys(R.defaults)) assert.ok(html.includes(`id="roof-${key}"`), key);
  const x = R.calculate();
  const throughput = html.match(/<td id="roof-recipe-throughput">([^<]+)<\/td>/)[1];
  const tflops = html.match(/<td id="roof-recipe-tflops">([^<]+)<\/td>/)[1];
  assert.equal(throughput, `${x.resourceTokensPerSecondPerGPU.toLocaleString("en-US", { maximumFractionDigits: 0 })} tokens/s/GPU`);
  assert.equal(tflops, `${x.resourceModelTFLOPsPerGPU.toLocaleString("en-US", { maximumFractionDigits: 1 })} model TFLOP/s/GPU`);
  const header = html.match(/<header>([\s\S]*?)<\/header>/)[1];
  assert.doesNotMatch(header, /headline-|tokens\/s|TFLOP|TF\/s/);
  assert.doesNotMatch(html, /id="headline-/);
  assert.match(html, /Default analyzed recipe: 512 GB200 GPUs/);
  assert.match(html, /value="gb200" selected/);
  assert.match(html, /value="gb300"/);
  assert.match(html, /id="roof-hardware-note"/);
  assert.match(html, /id="roof-vp"/);
  assert.match(html, /VPP1 keeps the existing TP1, ordinary-1F1B model unchanged/);
  assert.match(html, /VPP &gt; 1 is an estimate, not a schedule simulation/);
  assert.match(html, /Identical rate inputs produce identical modeled times/);
  assert.match(html, /Roofline for the configured recipe/);
  assert.match(html, /Historical measurements: comparison only, not the roofline/);
  assert.match(html, /https:\/\/deepseek-v3\.ezyang\.com\/studies\/kimi-k3\.html/);
});

test("configurable GPU counts conserve per-token work and derive the actual mesh and global batch", () => {
  for (const world of [128, 256, 512, 1024]) {
    const x = R.calculate({ world, pp: 4, ep: 32, mbs: 32, domain: Math.min(64, world / 4) });
    assert.equal(x.config.world, world);
    assert.equal(x.stageRanks, world / 4);
    assert.equal(x.dp, world / 4);
    assert.equal(x.expertFsdp, world / 4 / 32);
    assert.equal(x.globalTokens, x.dp * 32 * 4096);
    assert.equal(x.tokensPerGPUUpdate, x.globalTokens / world);
    close(
      x.causalModelFlops / x.globalTokens,
      R.calculate({ pp: 4, ep: 32, mbs: 32 }).causalModelFlops / R.calculate({ pp: 4, ep: 32, mbs: 32 }).globalTokens
    );
    close(x.resourceTokensPerSecondPerGPU, x.globalTokens / (world * x.sketchSeconds));
    close(x.resourceModelTFLOPsPerGPU, (x.resourceTokensPerSecondPerGPU * x.conventionPerToken) / 1e12);
  }
});

test("a communication bottleneck lowers the configured recipe's throughput and model TFLOP/s", () => {
  const base = R.calculate();
  const slowNetwork = R.calculate({ domain: 8, network: 5, routingMap: "hybridep" });
  assert.equal(base.computeSeconds, slowNetwork.computeSeconds);
  assert.ok(slowNetwork.networkSeconds > slowNetwork.computeSeconds);
  assert.equal(slowNetwork.resourceSeconds, slowNetwork.networkSeconds);
  assert.ok(slowNetwork.resourceTokensPerSecondPerGPU < base.resourceTokensPerSecondPerGPU);
  assert.ok(slowNetwork.resourceModelTFLOPsPerGPU < base.resourceModelTFLOPsPerGPU);
});

test("invalid user-facing accounting inputs fail explicitly", () => {
  for (const config of [
    { hardware: "unknown" },
    { hardware: "constructor" },
    { hardware: null },
    { vp: 0 },
    { vp: -1 },
    { vp: 1.5 },
    { vp: Infinity },
    { vp: 128 },
    { world: 0 },
    { world: -1 },
    { world: 255 },
    { world: 64 },
    { pp: 3 },
    { pp: 3.2 },
    { pp: 0 },
    { pp: 128 },
    { pp: 32, ep: 16, domain: 16 },
    { ep: 128 },
    { ep: 7 },
    { domain: 128 },
    { cp: 128 },
    { seq: 4097, cp: 4 },
    { segments: 3 },
    { gemm: 0 },
    { fp8Gemm: 0 },
    { precision: "fp4" },
    { routingMap: "unknown" },
    { kdaChunk: 0 },
    { hbm: Infinity },
    { rows: -1 },
    { policy: "unknown" },
    { ac: "partial" },
  ])
    assert.throws(() => R.calculate(config), RangeError, JSON.stringify(config));
});

test("all three studies expose reciprocal series links without a homepage entry", () => {
  const pages = ["kimi-k3.html", "kimi-k3-memory.html", "kimi-k3-roofline.html"];
  for (const page of pages) {
    const source = readFileSync(join(__dirname, "..", page), "utf8");
    const nav = source.match(/<nav class="series"[^>]*>([\s\S]*?)<\/nav>/)[1];
    for (const name of pages) assert.ok(nav.includes(`href="./${name}"`));
    assert.equal((nav.match(/aria-current="page"/g) || []).length, 1);
    assert.match(source, /name="robots" content="noindex, follow"/);
  }
});
