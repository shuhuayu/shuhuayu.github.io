const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const A = require("../qwen3-8-flash-next-accounting.js");
const M = require("../qwen3-8-flash-next-memory-model.js");
const checkpoint = require("../qwen3-8-flash-next-checkpoint.json");
const sum = A.sum;
const close = (actual, expected, tol = 1e-9) =>
  assert.ok(Math.abs(actual - expected) <= tol * Math.max(1, Math.abs(expected)), `${actual} != ${expected}`);
const gib = (value) => Number((value / M.GiB).toFixed(2));

test("trainable parameters equal the checkpoint audit minus vision and MTP", () => {
  const g = checkpoint.groups;
  assert.equal(A.totals.params, checkpoint.dtype_elements.BF16 - g.vision - g.mtp);
  assert.equal(A.totals.params, 176943899520);
  assert.equal(A.totals.routed, g.routed_experts);
  assert.equal(A.totals.table, g.lexical_table);
  assert.equal(A.totals.muon + A.totals.adamw + A.totals.adam, A.totals.params);
  assert.equal(A.totals.muon, 123003207680);
  assert.equal(A.totals.adamw, 2740446080);
  assert.equal(A.totals.adam, 51200245760);
  assert.equal(sum(A.ple.headSizes, (size) => size) <= A.ple.rows, true);
});

test("every PP x VPP split conserves layers and parameters", () => {
  for (const [pp, vp] of [
    [1, 1],
    [2, 1],
    [2, 2],
    [2, 3],
    [4, 2],
    [4, 4],
  ]) {
    const chunks = A.partition(pp, vp);
    assert.deepEqual(
      chunks.flatMap((chunk) => chunk.layers.map((layer) => layer.index)),
      A.layers.map((layer) => layer.index)
    );
    assert.equal(
      sum(chunks, (chunk) => chunk.dense + chunk.routed + chunk.table),
      A.totals.params
    );
    for (let rank = 0; rank < pp; rank++) assert.equal(chunks.filter((chunk) => chunk.rank === rank).length, vp);
  }
  assert.throws(() => A.partition(5, 1), RangeError);
});

test("per-GPU state over all ranks reproduces the logical 12P + 16P bill", () => {
  const bill = M.stateBill().total;
  assert.equal(bill, 2339089561600);
  for (const config of [{}, { vp: 1 }, { pp: 1, vp: 1, ep: 16 }, { world: 32, ep: 16 }, { pp: 4, vp: 2, ep: 4 }]) {
    const r = M.calculate(config);
    close(
      sum(r.stages, (stage) => stage.stateTotal * r.dp),
      bill
    );
    const world = M.calculate({ ...config, table: "world" });
    close(sum(world.stages, (stage) => (stage.stateTotal - stage.stored.table * 16) * world.dp) + A.totals.table * 16, bill);
  }
});

test("the anchor ledger matches the page's static numbers", () => {
  const r = M.calculate();
  const [rank0, rank1] = r.stages;
  assert.deepEqual(
    [gib(rank0.master), gib(rank0.momentum), gib(rank0.moments), gib(rank0.reducedGrad), gib(rank0.stateTotal)],
    [53.13, 28.65, 48.96, 53.13, 183.85]
  );
  assert.deepEqual(
    [gib(rank1.master), gib(rank1.momentum), gib(rank1.moments), gib(rank1.reducedGrad), gib(rank1.stateTotal)],
    [29.27, 28.63, 1.28, 29.27, 88.45]
  );
  assert.equal(gib(rank0.stored.table * 16), 95.37);
  assert.equal(gib(M.stateBill().total / 16), 136.15);
  const html = readFileSync(join(__dirname, "../qwen3-8-flash-next-memory.html"), "utf8");
  for (const text of ["183.85 GiB", "88.45 GiB", "95.37 GiB", "136.15 GiB", "2,339,089,561,600", "40,000,192", "33,180"])
    assert.ok(html.includes(text), text);
});

test("the interleaved schedule leaves 5 and 3 unfinished chunk forwards", () => {
  const r = M.calculate();
  assert.deepEqual(
    r.stages.map((stage) => stage.timeline.peak),
    [5, 3]
  );
  for (const stage of r.stages) {
    assert.equal(stage.timeline.actions.length, 2 * r.config.mbs * r.config.vp);
    assert.equal(stage.timeline.live.at(-1), 0);
  }
  assert.deepEqual(
    M.calculate({ vp: 1 }).stages.map((stage) => stage.timeline.peak),
    [2, 1]
  );
  close(r.bubble, 1 / 32);
  assert.equal(r.globalTokens, 2097152);
});

test("Dist-MoE slot and scratch follow the planner's BF16 formulas", () => {
  const r = M.calculate();
  const T = 16384;
  const R = 10 * T;
  assert.equal(r.moe.minimumLayer, T * 2560 * 2);
  assert.equal(r.moe.fullLayer, R * (2560 + 1280 + 2560) * 2);
  close(r.moe.slotBytes, 12 * (r.moe.minimumLayer + 0.5 * (r.moe.fullLayer - r.moe.minimumLayer)));
  assert.equal(r.moe.rows, 655360);
  assert.equal(r.moe.scratchBytes, 655360 * 16640);
  assert.equal(r.stage.slotTotal, 5 * r.moe.slotBytes);
  // The topology cap binds when the scratch factor exceeds what EP can route.
  assert.equal(M.calculate({ scratch: 8 }).moe.rows, Math.min(8 * R, T * 8 * 10));
});

test("the anchor's rank 0 exceeds the device and the levers move it", () => {
  const anchor = M.calculate().budget;
  assert.ok(anchor > M.capacity);
  assert.ok(M.calculate({ stage: 1 }).budget < M.capacity);
  assert.ok(M.calculate({ slot: 0 }).budget < M.capacity);
  assert.ok(M.calculate({ tokens: 8192, mbs: 32 }).budget < M.capacity);
  assert.ok(M.calculate({ table: "world" }).budget < M.capacity);
  assert.ok(M.calculate({ phase: "optimizer" }).budget < M.calculate({ phase: "backward" }).budget);
  assert.equal(M.calculate({ phase: "forward" }).parts.find((part) => part.key === "gradient").value, 0);
});

test("PLE request capacity follows the table layout, not the data", () => {
  assert.equal(A.pleRequestCapacity(16384, 8), 33180);
  assert.equal(A.pleRequestCapacity(16384, 16), 16796);
  assert.equal(A.pleRequestCapacity(4096, 8), 8604);
  assert.equal(M.workspaces(M.normalize()).pleRows, 8 * 33180 * 160 * 2);
});

test("invalid layouts are rejected", () => {
  assert.throws(() => M.normalize({ ep: 16 }), RangeError);
  assert.throws(() => M.normalize({ pp: 4, vp: 4, mbs: 6 }), RangeError);
  assert.throws(() => M.normalize({ vp: 2, pp: 1 }), RangeError);
  assert.throws(() => M.normalize({ tokens: 5000 }), RangeError);
  assert.throws(() => M.normalize({ stage: 2 }), RangeError);
});
