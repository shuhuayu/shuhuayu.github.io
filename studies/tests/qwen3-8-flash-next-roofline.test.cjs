const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const A = require("../qwen3-8-flash-next-accounting.js");
const R = require("../qwen3-8-flash-next-roofline-model.js");
const sum = A.sum;
const close = (actual, expected, tol = 1e-9) =>
  assert.ok(Math.abs(actual - expected) <= tol * Math.max(1, Math.abs(expected)), `${actual} != ${expected}`);

test("activated parameters follow the conventional definition", () => {
  const active = R.activeParams();
  assert.equal(active.routed, 2359296000);
  assert.equal(active.routed, 48 * 10 * 3 * 2560 * 640);
  assert.equal(active.dense, 4311999360);
  assert.equal(active.total, 6671295360);
  close(R.flopsPerToken(R.normalize()).conventional, 6 * 6671295360 + 12 * (6 * 24 * 512 * 2051 + 6 * 4 * 128 * 1024) + 36 * 18 * 48 * 128 * 128);
});

test("layer GEMMs reproduce every matmul parameter exactly once", () => {
  // K x N summed over layers (and all 512 experts) must equal every parameter
  // except the table and the vector-shaped ones: convolutions, norms, decay
  // parameters and the [1, D] shared-expert gate, which is not priced.
  const T = 16384;
  const fromGemms =
    sum(A.layers, (layer) => sum(R.layerGemms(layer, T, 8), (g) => (g.routed ? g.k * g.n * A.E : g.k * g.n))) + sum(R.headGemms(T), (g) => g.k * g.n);
  const gdnVectors = 2 * 2048 * 4 + 6144 * 4 + 48 + 48 + 128;
  const qsaVectors = 2 * 256 + 2 * 128;
  const perLayer = 2 * 10240 + 2560;
  const pleVectors = 3 * 10240 + 10240 * 4;
  const vectors = 36 * gdnVectors + 12 * qsaVectors + 48 * perLayer + pleVectors + 10240;
  assert.equal(fromGemms + vectors + A.totals.table + A.V * A.D, A.totals.params);
});

test("the anchor roofline matches the page's static numbers", () => {
  const r = R.calculate();
  assert.equal(Math.round(r.tokensPerSecondPerGPU), 30457);
  assert.equal(Math.round(r.modelTflopsPerGPU), 1291);
  assert.equal((r.perToken.conventional / 1e9).toFixed(2), "42.39");
  assert.equal(r.globalTokens, 2097152);
  assert.equal(r.expertRows, 2560);
  close(r.bubble, 1 / 32);
  const html = readFileSync(join(__dirname, "../qwen3-8-flash-next-roofline.html"), "utf8");
  for (const text of ["30,457", "1,291", "42.39", "6.67B", "2,359,296,000", "4,311,999,360", "6,671,295,360", "3.08 GiB"])
    assert.ok(html.includes(text), text);
});

test("the anchor is compute-bound and the head makes the last rank slowest", () => {
  const r = R.calculate();
  for (const stage of r.stages) assert.ok(stage.computeSeconds > stage.nvlSeconds);
  assert.ok(r.stages[1].computeSeconds > r.stages[0].computeSeconds);
  const ep = r.stages[0].traffic.find((item) => item.key === "ep").bytes;
  assert.equal(ep, 16 * 24 * 4.5 * 16384 * 10 * (7 / 8) * 2560 * 2);
  assert.ok(r.stages[0].traffic.find((item) => item.key === "ple").bytes < 0.01 * ep);
  assert.equal(r.stages[1].traffic.find((item) => item.key === "ple").bytes, 0);
});

test("recompute adds issued work but not model FLOPs", () => {
  const full = R.calculate();
  const none = R.calculate({ ac: "none" });
  assert.equal(full.perToken.conventional, none.perToken.conventional);
  assert.ok(full.stages[0].gemmIssuedFlops > none.stages[0].gemmIssuedFlops);
  assert.equal(full.stages[0].gemmModelFlops, none.stages[0].gemmModelFlops);
  assert.ok(none.tokensPerSecondPerGPU > full.tokensPerSecondPerGPU);
});

test("QSA reads 1,537 of the visible keys on average in 4K documents", () => {
  assert.equal(R.meanSelectedKeys(4096).toFixed(0), "1537");
  assert.equal(R.meanSelectedKeys(2048), (2048 + 1) / 2);
  const work = R.attentionWork(R.normalize());
  assert.ok(work.qsaForward < work.denseCounterfactual);
});

test("Muon work counts 4,608 expert matrices per GPU", () => {
  const stage = R.calculate().stages[0];
  assert.equal(stage.muonMatrices, 4608);
  close(stage.muonFlops, 4608 * 8 * (4 * 640 ** 2 * 2560 + 2 * 640 ** 3));
});

test("invalid layouts are rejected", () => {
  assert.throws(() => R.normalize({ ep: 16 }), RangeError);
  assert.throws(() => R.normalize({ vp: 2, pp: 1 }), RangeError);
  assert.throws(() => R.normalize({ gemm: 0 }), RangeError);
  assert.throws(() => R.normalize({ ac: "partial" }), RangeError);
});
