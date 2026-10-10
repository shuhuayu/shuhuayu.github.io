const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const A = require("../kimi-k3-accounting.js");
const R = require("../kimi-k3-roofline-model.js");
const sum = (items, get) => items.reduce((total, item, index) => total + get(item, index), 0);
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) <= 1e-12 * Math.max(1, Math.abs(expected)), `${actual} != ${expected}`);

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
  close(gemm.seconds, Math.max(gemm.flops / 2500e12, gemm.bytes / 8e12));
  assert.equal(x.config.gemm / x.config.hbm, 312.5);
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
  const full = R.calculate();
  const none = R.calculate({ ac: "none" });
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
  const x = R.calculate();
  for (const stage of x.stages) for (const item of stage.traffic) if (item.key !== "pipeline") assert.equal(item.network, 0);
  const split = R.calculate({ domain: 8 });
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
  const x = R.calculate();
  assert.equal(x.metadataInput, 3670016);
  assert.equal((x.metadataInput * x.config.ep) / 2 ** 20, 224);
  assert.equal(x.metadataCall.nvl / 2 ** 20, 220.5);
  const stage = x.stages[3];
  assert.equal(x.config.mbs * stage.moeLayers * x.metadataRounds, 1536);
  assert.equal(stage.traffic.find((item) => item.key === "metadata").nvl, 1536 * 220.5 * 2 ** 20);
});

test("FSDP round counts distinguish observed, fully resharded and retained views", () => {
  assert.deepEqual(R.calculate().rounds, { ag: 65, rs: 1 });
  assert.deepEqual(R.calculate({ policy: "reshard" }).rounds, { ag: 128, rs: 64 });
  assert.deepEqual(R.calculate({ policy: "retain" }).rounds, { ag: 1, rs: 1 });
  assert.equal(R.calculate().expertFsdp, 1);
  const bytes = R.collectiveBytes(100, 1, 1, 2);
  assert.deepEqual(bytes, { nvl: 0, network: 0 });
  const hierarchical = R.collectiveBytes(100, 64, 8, 2);
  assert.equal(hierarchical.nvl, (200 * 63) / 64);
  assert.equal(hierarchical.network, (200 * 7) / 8 / 8);
});

test("PP sends hidden plus the full first-visit residual stack", () => {
  const x = R.calculate();
  assert.deepEqual(
    x.stages.slice(0, -1).map((stage) => stage.after),
    [2, 4, 6, 8, 10, 12, 14]
  );
  assert.equal(((1 + x.stages[6].after) * x.tokens * A.D * 2) / 2 ** 20, 840);
  assert.equal(x.stages[3].ppBytes, 64 * 4096 * 7168 * 2 * (1 + 6 + (1 + 8)));
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

test("invalid user-facing accounting inputs fail explicitly", () => {
  for (const config of [
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
