const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { createHash } = require("node:crypto");
const A = require("../kimi-k3-accounting.js");
const M = require("../kimi-k3-memory-model.js");
const sum = (items, get) => items.reduce((total, item, index) => total + get(item, index), 0);
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) <= 1e-12 * Math.max(1, Math.abs(expected)), `${actual} != ${expected}`);

test("parameter groups reconstruct every layer's existing architecture and optimizer assignment", () => {
  for (const layer of A.layers) {
    for (const ep of [1, 32, 64, 128]) {
      const groups = M.parameterGroups(layer, ep);
      assert.equal(
        sum(groups, (group) => group.params),
        layer.dense + layer.routed
      );
      assert.equal(
        sum(groups, (group) => group.adam),
        layer.adam
      );
      assert.equal(
        sum(
          groups.filter((group) => group.routed),
          (group) => group.params
        ),
        layer.routed
      );
      assert.equal(
        sum(groups, (group) => group.epLocal),
        layer.dense + layer.routed / ep
      );
    }
  }
});

test("PP and virtual chunks conserve all 93 layers, parameters, and optimizer groups", () => {
  for (const pp of [1, 2, 4, 8, 16]) {
    for (const vp of [1, 2, 4]) {
      const c = M.normalize({ pp, vp, stage: 0, ep: Math.min(64, 512 / pp), sched: vp > 1 ? "interleaved" : "1f1b" });
      const chunks = M.partitionChunks(c);
      assert.equal(chunks.length, pp * vp);
      assert.deepEqual(
        chunks.flatMap((chunk) => chunk.layers.map((layer) => layer.index)),
        Array.from({ length: 93 }, (_, index) => index)
      );
      assert.equal(
        sum(chunks, (chunk) => chunk.dense + chunk.routed),
        A.totalParams
      );
      assert.equal(
        sum(chunks, (chunk) => chunk.adam),
        A.adamParams
      );
      for (let rank = 0; rank < pp; rank++) assert.equal(chunks.filter((chunk) => chunk.rank === rank).length, vp);
    }
  }
  const chunks = M.partitionChunks(M.normalize({ vp: 2, sched: "interleaved" }));
  assert.deepEqual(
    chunks.map((chunk) => chunk.layers.length),
    [5, ...Array(14).fill(6), 4]
  );
  assert.deepEqual(
    chunks.filter((chunk) => chunk.rank === 3).map((chunk) => [chunk.layers[0].index, chunk.layers.at(-1).index]),
    [
      [17, 22],
      [65, 70],
    ]
  );
});

test("the FP32 state tally matches the existing baseline and the model-wide 12P+16P formula", () => {
  const base = M.calculate();
  close(base.stateTotal / M.GiB, 63.23320531100035);
  const stageStates = Array.from({ length: 8 }, (_, stage) => M.calculate({ stage }).stateTotal);
  close(sum(stageStates, (size) => size) * 64, A.muonParams * 12 + A.adamParams * 16);
  for (const vp of [1, 2, 4]) {
    const states = Array.from({ length: 8 }, (_, stage) => M.calculate({ vp, stage, sched: vp > 1 ? "interleaved" : "1f1b" }).stateTotal);
    close(sum(states, (size) => size) * 64, A.muonParams * 12 + A.adamParams * 16);
  }
});

test("CP changes local tokens and DP batch, not the flattened FSDP storage degree", () => {
  const short = M.calculate();
  const long = M.calculate({ seq: 65536, cp: 16 });
  assert.equal(long.tokens, 4096);
  assert.equal(long.dp, 4);
  assert.equal(long.denseFsdp, 64);
  assert.equal(long.expertFsdp, 1);
  assert.equal(long.stateTotal, short.stateTotal);
  assert.equal(long.globalTokens, short.globalTokens);
  assert.equal(long.stage.timeline.peak, short.stage.timeline.peak);
  assert.equal(long.globalRows, 256);
});

test("EP shrinks compute expert weights but balanced routed activation volume is conserved", () => {
  const ep32 = M.calculate({ ep: 32 });
  const ep64 = M.calculate();
  assert.equal(ep32.stateTotal, ep64.stateTotal);
  assert.equal(ep32.expertFsdp, 2);
  assert.equal(ep64.expertFsdp, 1);
  assert.ok(ep32.parts.find((part) => part.key === "compute").value > ep64.parts.find((part) => part.key === "compute").value);
  for (const ep of [1, 32, 64]) {
    const catalog = M.tensorCatalog(A.layers[1], M.normalize({ ep }));
    assert.equal(catalog.R, 4096 * 16);
    close(catalog.rowsPerExpert * catalog.localExperts, 4096 * 16);
    assert.equal(catalog.tensors.find((row) => row.id === "expert_gate_up").bytes, 4096 * 16 * 6144 * 2);
  }
  const imbalanced = M.tensorCatalog(A.layers[1], M.normalize({ imbalance: 1.5 }));
  assert.equal(imbalanced.R, 1.5 * 4096 * 16);
});

test("view aliases contribute zero allocation bytes, while cat() allocates storage", () => {
  for (const index of [0, 1, 3, 92]) {
    const catalog = M.tensorCatalog(A.layers[index], M.defaults);
    assert.equal(new Set(catalog.tensors.map((row) => row.id)).size, catalog.tensors.length);
    for (const row of catalog.tensors.filter((item) => item.alias)) {
      assert.equal(row.bytes, 0);
      assert.ok(row.logicalBytes > 0);
      assert.ok(catalog.tensors.some((owner) => owner.id === row.alias && !owner.alias));
      assert.equal(M.isSaved(row, { ...M.defaults, ac: "custom" }, { [row.id]: true }), false);
    }
  }
  const mla = M.tensorCatalog(A.layers[3], M.defaults);
  assert.equal(mla.tensors.find((row) => row.id === "mla_expand").bytes, 0);
  assert.equal(mla.tensors.find((row) => row.id === "mla_k").bytes, 4096 * 96 * 192 * 2);
  assert.ok(!mla.tensors.some((row) => row.logical === "[T, T]"));
  const kda = M.tensorCatalog(A.layers[1], M.defaults);
  assert.equal(kda.tensors.find((row) => row.id === "kda_cat").bytes, 4096 * 3 * 12288 * 2);
  assert.equal(kda.tensors.find((row) => row.id === "kda_decay").dtype, "FP32");
});

test("full checkpoint inputs, custom saved candidates, and residual bank aliases are separate", () => {
  const stream = 4096 * 7168 * 2;
  assert.equal(M.savedForLayer(A.layers[1], M.defaults), stream);
  const none = M.normalize({ ac: "none" });
  assert.ok(M.savedForLayer(A.layers[1], none) > 40 * stream);
  const custom = M.normalize({ ac: "custom" });
  const catalog = M.tensorCatalog(A.layers[1], custom);
  const marks = Object.fromEntries(catalog.tensors.map((row) => [row.id, false]));
  marks.stream = true;
  marks.kda_decay = true;
  assert.equal(M.savedForLayer(A.layers[1], custom, marks), stream + 4096 * 96 * 128 * 4);
  const chunk = M.partitionChunks(M.defaults)[3];
  const saved = M.chunkActivations(chunk, M.defaults);
  assert.equal(saved.layerBytes, 12 * stream);
  assert.deepEqual(saved.incomingBanks, [0, 12, 24]);
  assert.equal(saved.bankBytes, 3 * stream);
  marks.stream = false;
  assert.deepEqual(M.chunkActivations(chunk, custom, marks).incomingBanks, [0, 12, 24, 36]);
});

test("ordinary 1F1B lifetimes implement the PP-rank live-forward bound", () => {
  for (const mbs of [1, 4, 8, 64]) {
    for (let rank = 0; rank < 8; rank++) {
      const c = M.normalize({ mbs });
      const timeline = M.schedule(c, rank, [7]);
      assert.equal(timeline.actions.length, 2 * mbs);
      assert.equal(timeline.lifetimes.length, mbs);
      assert.equal(timeline.peakCount, Math.min(mbs, 8 - rank));
      assert.equal(timeline.peak, 7 * Math.min(mbs, 8 - rank));
      assert.deepEqual(timeline.snapshots.at(-1).counts, [0]);
    }
  }
});

test("interleaved order has matching forward/backward pairs and weights simultaneous chunk lifetimes", () => {
  for (const pp of [1, 2, 4, 8]) {
    for (const vp of [1, 2, 4]) {
      for (const mbs of [8, 16, 64]) {
        const c = M.normalize({ pp, vp, mbs, stage: 0, sched: "interleaved" });
        for (let rank = 0; rank < pp; rank++) {
          const weights = Array.from({ length: vp }, (_, i) => i + 2);
          const timeline = M.schedule(c, rank, weights);
          assert.equal(timeline.actions.length, 2 * vp * mbs);
          assert.equal(timeline.lifetimes.length, vp * mbs);
          assert.deepEqual(timeline.snapshots.at(-1).counts, Array(vp).fill(0));
          assert.equal(timeline.peak, Math.max(...timeline.snapshots.map((snapshot) => sum(snapshot.counts, (n, i) => n * weights[i]))));
          for (const life of timeline.lifetimes) assert.ok(life.start < life.end);
        }
      }
    }
  }
  const ordinary = M.schedule(M.defaults, 3);
  const interleaved1 = M.schedule(M.normalize({ sched: "interleaved" }), 3);
  const interleaved2 = M.schedule(M.normalize({ vp: 2, sched: "interleaved" }), 3);
  assert.equal(ordinary.peakCount, 5);
  assert.equal(interleaved1.peakCount, 9);
  assert.equal(interleaved2.peakCount, 17);
});

test("interleaved action order matches independently generated PyTorch schedule fixtures", () => {
  // Generated with ScheduleInterleaved1F1B in PyTorch 2.15.0.dev20261005.
  const cases = [
    { pp: 8, vp: 1, mbs: 64, rank: 3, hash: "430df2eb0480ee50a2582089d7ebba30d03dd50b65e06f5a457b7df1eac09991" },
    { pp: 8, vp: 2, mbs: 64, rank: 3, hash: "32d6f3ea610782069d2639c4628f4e50fa288cc9e84977c4bf11acf50813cb97" },
    { pp: 4, vp: 2, mbs: 8, rank: 1, hash: "8eb077ca5bf9a5ab9c7ff9967146ca43598ed783035259cde4805dc47068679f" },
    { pp: 8, vp: 2, mbs: 8, rank: 7, hash: "418fbc9128022ccbac9925715de86c8ed208082f31e74a0594821c44015d1298" },
  ];
  for (const row of cases) {
    const config = M.normalize({ pp: row.pp, vp: row.vp, mbs: row.mbs, stage: row.rank, sched: "interleaved" });
    const actions = M.schedule(config, row.rank).actions.map((action) => [
      action.kind === "F" ? "_ComputationType.FORWARD" : "_ComputationType.FULL_BACKWARD",
      action.chunk,
      action.mb,
    ]);
    assert.equal(createHash("sha256").update(JSON.stringify(actions)).digest("hex"), row.hash);
  }
});

test("smaller virtual chunks reduce gathered residency without deleting the other chunks' state", () => {
  const vp2 = M.calculate({ vp: 2, sched: "interleaved" });
  const all = M.calculate({ vp: 2, sched: "interleaved", unit: "stage" });
  const both = M.calculate({ vp: 2, sched: "interleaved", active: 2 });
  const compute = (result) => result.parts.find((part) => part.key === "compute").value;
  assert.equal(vp2.owned.length, 2);
  assert.equal(vp2.layers.length, 12);
  assert.equal(vp2.stateTotal, all.stateTotal);
  assert.ok(compute(vp2) < compute(all));
  assert.equal(compute(both), compute(all));
});

test("gradient phases replace rather than double-count complete gradient histories", () => {
  const base = M.calculate();
  const sharded = M.calculate({ grad: "sharded" });
  const forward = M.calculate({ phase: "forward" });
  const optimizer = M.calculate({ phase: "optimizer" });
  const part = (result, key) => result.parts.find((item) => item.key === key).value;
  assert.equal(part(base, "gradient"), base.fullGrad);
  assert.equal(part(sharded, "gradient"), sharded.reducedGrad);
  assert.equal(part(forward, "gradient"), 0);
  assert.equal(part(optimizer, "gradient"), optimizer.reducedGrad);
  for (const key of ["compute", "activation", "working", "residual"]) assert.equal(part(optimizer, key), 0);
  assert.equal(forward.stateTotal, base.stateTotal);
  assert.equal(M.parameterCopies(M.parameterGroups(A.layers[1])[0], forward.config).gradient, 0);
  assert.equal(
    base.budget,
    sum(base.parts, (item) => item.value)
  );
});

test("temporary examples have explicit dtype and layout, and are not additional budget terms", () => {
  const w = M.workspaces(A.layers[36], M.defaults, 14);
  assert.equal(w.inferenceState, 6 * 2 ** 20);
  assert.equal(w.kdaChunkState, 384 * 2 ** 20);
  assert.equal(w.situCasts, 1.5 * M.GiB);
  assert.equal(w.matrix, 21 * 2 ** 20);
  assert.equal(w.gram, 18 * 2 ** 20);
  assert.equal(w.nsCandidates, 14 * (42 + 36) * 2 ** 20);
  assert.equal(w.logits, 1.25 * M.GiB);
  const base = M.calculate();
  const list = M.calculate({ residual: "list" });
  assert.equal(base.budget - list.budget, base.parts.find((part) => part.key === "residual").value);
  assert.equal(base.stage.timeline.peak, list.stage.timeline.peak);
});

test("removing vision changes only the front's declared parameter inventory", () => {
  const front = M.calculate({ stage: 0 });
  const text = M.calculate({ stage: 0, vision: "off" });
  close(front.stateTotal - text.stateTotal, (16 * A.vision) / 64);
  assert.equal(M.calculate({ vision: "off" }).stateTotal, M.calculate().stateTotal);
});

test("invalid mesh, schedule, dtype scenario, or chunk contracts fail explicitly", () => {
  for (const config of [
    { world: 500 },
    { ep: 128 },
    { pp: 16, cp: 64 },
    { seq: 4097, cp: 2 },
    { vp: 2 },
    { vp: 1, active: 2 },
    { pp: 32, vp: 4, stage: 0, ep: 16, sched: "interleaved" },
    { stage: 8 },
    { sched: "dualpipe" },
    { ac: "unknown" },
    { reserve: -1 },
    { imbalance: 0.5 },
    { sched: "interleaved", mbs: 17, pp: 8 },
  ])
    assert.throws(() => M.calculate(config));
});

test("the public page loads the model before the UI and identifies estimates and logical time", () => {
  const html = readFileSync(join(__dirname, "../kimi-k3-memory.html"), "utf8");
  const scripts = [...html.matchAll(/<script src="\.\/([^\"]+)"/g)].map((match) => match[1].split("?")[0]);
  assert.deepEqual(scripts, ["kimi-k3-accounting.js", "kimi-k3-memory-model.js", "kimi-k3-memory.js"]);
  assert.ok(html.includes("kimi-k3-memory.js?v=memory-2"));
  for (const key of Object.keys(M.defaults)) assert.ok(html.includes(`id="mem-${key}"`), `missing control: ${key}`);
  for (const id of ["tensors", "activation-ledger", "parameter-ledger", "workspaces", "schedule-diagram"]) assert.ok(html.includes(`id="${id}"`));
  assert.ok(html.includes("not elapsed time"));
  assert.ok(html.toLowerCase().includes("not a fit guarantee"));
  assert.ok(!/manifold:\/\/|internalfb\.com|TIER:|\/home\/shuhuay|mast\./.test(html));
});
