const test = require("node:test");
const assert = require("node:assert/strict");
const M = require("../deepseek-v4-model.js");

test("the Pro and Flash decoder maps exclude the extra SWA-only MTP block", () => {
  assert.deepEqual(M.counts("pro"), { CSA: 30, HCA: 31, SWA: 0 });
  assert.deepEqual(M.counts("flash"), { CSA: 21, HCA: 20, SWA: 2 });
  assert.equal(M.layers("pro").length, 61);
  assert.equal(M.layers("flash").length, 43);
  for (const id of ["pro", "flash"]) {
    assert.equal(M.layers(id).filter((item) => item.hash).length, 3);
    assert.equal(M.layers(id).at(-1).mode, "CSA");
    assert.equal(M.spec(id).mtpLayers, 1);
    assert.equal(M.spec(id).activeExperts, 6);
  }
});

test("one-million-token mixed payload follows shared KV, not separate K and V per query head", () => {
  const c = M.cache("pro", 1048576);
  assert.equal(c.mainEntry, 576);
  assert.equal(c.indexEntry, 64);
  assert.equal(c.csaEntries, 30 * 262144);
  assert.equal(c.hcaEntries, 31 * 8192);
  assert.equal(c.local, 61 * 128 * 576);
  assert.equal(c.total, c.csa + c.hca + c.index + c.local);
  assert.equal(c.baseline, 61 * 1048576 * 4096);
  assert.ok(c.fraction > 0.019 && c.fraction < 0.021);
  assert.equal(c.csaSlots, 1152);
  assert.equal(c.hcaSlots, 8320);
  assert.equal(M.cache("flash", 1048576).csaSlots, 640);
});

test("short / odd contexts count completed blocks and bounded local caches", () => {
  for (const id of ["pro", "flash"]) {
    for (const tokens of [1, 3, 4, 127, 128, 129, 65535]) {
      const c = M.cache(id, tokens);
      const count = M.counts(id);
      assert.equal(c.csaEntries, count.CSA * Math.floor(tokens / 4));
      assert.equal(c.hcaEntries, count.HCA * Math.floor(tokens / 128));
      assert.equal(c.local, M.spec(id).numLayers * Math.min(tokens, 128) * 576);
    }
    assert.equal(M.cache(id, 1).hca, 0);
    assert.equal(M.cache(id, 1).csa, 0);
  }
});

test("all-BF16 payload and request count change storage, not attention positions", () => {
  const single = M.cache("pro", 65536);
  const batch = M.cache("pro", 65536, "mixed", 4);
  const bf16 = M.cache("pro", 65536, "bf16");
  assert.equal(batch.total, 4 * single.total);
  assert.equal(batch.baseline, 4 * single.baseline);
  assert.equal(batch.csaSlots, single.csaSlots);
  assert.equal(bf16.mainEntry, 1024);
  assert.equal(bf16.indexEntry, 256);
  assert.ok(bf16.total > single.total);
  assert.throws(() => M.cache("pro", 0), RangeError);
  assert.throws(() => M.cache("pro", 1048577), RangeError);
  assert.throws(() => M.cache("pro", 8, "fp4-main"), RangeError);
});

test("parameter accounting treats every decoder FFN as MoE and counts shared MTP heads only once", () => {
  for (const id of ["pro", "flash"]) {
    const s = M.spec(id);
    const p = M.parameters(id);
    assert.equal(p.oneExpert, 3 * s.dim * s.expertDim);
    assert.equal(p.rows[0].value, s.numLayers * s.experts * p.oneExpert);
    assert.equal(
      p.total,
      p.rows.reduce((sum, row) => sum + row.value, 0)
    );
    assert.equal(p.withMtp, p.total + p.mtp);
    assert.ok(p.routedFraction > 0.97);
    assert.ok(p.mtp < p.total / 30);
    assert.ok(p.rows.every((row) => Number.isSafeInteger(row.value)));
  }
  assert.ok(M.parameters("pro").withMtp > 1.5e12 && M.parameters("pro").withMtp < 1.7e12);
  assert.ok(M.parameters("flash").withMtp > 280e9 && M.parameters("flash").withMtp < 300e9);
});

test("the V4 hybrid schedule is eight fast iterations followed by two stable iterations", () => {
  assert.equal(M.nsSchedule.length, 10);
  assert.deepEqual(M.nsSchedule.slice(0, 8), Array(8).fill(M.fast));
  assert.deepEqual(M.nsSchedule.slice(8), Array(2).fill(M.stable));
  assert.equal(M.polynomial(1, M.stable), 1);
  assert.ok(Math.abs(M.polynomial(1, M.fast) - 0.701) < 1e-12);
  const values = M.singularValues(0.4);
  assert.equal(values.length, 11);
  assert.ok(Math.abs(values.at(-1) - 1) < Math.abs(values[8] - 1));
  assert.deepEqual(M.singularValues(0), Array(11).fill(0));
});
