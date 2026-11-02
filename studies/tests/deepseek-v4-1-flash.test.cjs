const test = require("node:test");
const assert = require("node:assert/strict");
const M = require("../deepseek-v4-1-flash-model.js");

test("the released 40-layer architecture has four KV owners and eight index producers", () => {
  assert.deepEqual(
    M.layers.filter((layer) => layer.mode === "Full").map((layer) => layer.number),
    [3, 9, 15, 21]
  );
  assert.deepEqual(
    M.layers.filter((layer) => layer.mode === "Reindex").map((layer) => layer.number),
    [25, 29, 33, 37]
  );
  assert.equal(M.layers.filter((layer) => layer.mode === "Reuse").length, 30);
  assert.equal(M.layers.filter((layer) => layer.mode === "SWA").length, 2);
  assert.deepEqual(
    M.layers.filter((layer) => layer.hasEngram).map((layer) => layer.number),
    [2, 15]
  );
});

test("layer 30 shares layer 21 KV and layer 29 indices; all decoder consumers use the encoder-derived cache", () => {
  assert.equal(M.layer(29).kvSource, 20);
  assert.equal(M.layer(29).indexSource, 28);
  for (const layer of M.layers.slice(20)) assert.equal(layer.kvSource, 20);
  for (const layer of M.layers.filter((item) => item.ratio)) {
    assert.ok(layer.kvSource <= layer.indexSource && layer.indexSource <= layer.index);
    assert.equal(M.layer(layer.kvSource).ratio, layer.ratio);
  }
});

test("bounded replay saves only long-prompt prefill work, while decode and training retain 40 layers", () => {
  assert.equal(M.work(128).fraction, 1);
  assert.equal(M.work(65536).total, 20 * 65536 + 20 * 128);
  assert.equal(M.work(65536, "decode").total, 40);
  assert.equal(M.work(65536, "decode").fraction, 1);
  assert.equal(M.work(65536, "training").total, 40 * 65536);
  assert.equal(M.work(65536, "training").fraction, 1);
});

test("cache accounting reconstructs the model card's 890 global bytes per token including quantization scales", () => {
  const c = M.cache(1048576);
  assert.equal(c.bytesPerToken, 890);
  assert.equal(c.global, 890 * 1048576);
  assert.equal(c.mainScales, 80 * 2 ** 20);
  assert.equal(c.indexScales, 10 * 2 ** 20);
  assert.equal(c.local, 2.5 * 2 ** 20);
  assert.equal(c.total, c.mainPayload + c.mainScales + c.indexPayload + c.indexScales + c.local);
  assert.equal(c.unsharedEntries, 29 * 1048576);
});

test("short and odd contexts count completed encoder pairs and bounded local state", () => {
  for (const tokens of [1, 127, 128, 129, 65535]) {
    const c = M.cache(tokens);
    assert.equal(c.entries, 3 * Math.floor(tokens / 2) + tokens);
    assert.equal(c.local, 40 * Math.min(tokens, 128) * 512);
    assert.equal(
      c.producers.reduce((total, producer) => total + producer.entries, 0),
      c.entries
    );
  }
  assert.equal(M.cache(1).entries, 1);
  assert.equal(M.cache(1048576, 16).mainScales, 0);
  assert.equal(M.cache(1048576, 16).index, M.cache(1048576).index);
});

test("Engram separates all stored table parameters from the 48 rows accessed per text token", () => {
  const e = M.engram();
  assert.equal(e.tableParams, 196613849600);
  assert.equal(e.rowsPerModule, 24);
  assert.equal(e.rowsPerToken, 48);
  assert.equal(e.lookupPayload, 12 * 1024);
  assert.equal(e.moduleInput, 6144);
  assert.equal(e.projectionParamsPerModule, 157286400);
  assert.deepEqual(
    M.recentNgrams("I live in New York").map((group) => group.tokens.join(" ")),
    ["New York", "in New York", "live in New York"]
  );
  assert.deepEqual(M.recentNgrams("York")[0].tokens, ["<pad>", "York"]);
});

test("the worked search masks future positions and restricts Reindex to the Full layer's candidate blocks", () => {
  const full = M.attentionExample("Full");
  const reindex = M.attentionExample("Reindex");
  assert.equal(full.memoryEntries, 25);
  assert.equal(full.scoredPositions.length, 25);
  assert.deepEqual(full.selected, [2, 17, 22]);
  assert.deepEqual(reindex.selected, [1, 18, 23]);
  assert.equal(reindex.scoredPositions.length, 12);
  assert.deepEqual(reindex.candidatePositions, full.candidatePositions);
  assert.ok(!reindex.scoredPositions.includes(10));
  for (const example of [full, reindex]) {
    assert.ok(example.selected.every((position) => position <= example.currentPosition));
    assert.ok(example.positions.filter((position) => !position.visible).every((position) => position.score === null && !position.candidate));
  }
});

test("Reuse keeps the latest selection and shared KV while performing no new index search", () => {
  const reindex = M.attentionExample("Reindex");
  const reuse = M.attentionExample("Reuse");
  assert.deepEqual(reuse.selected, reindex.selected);
  assert.deepEqual(reuse.scoredPositions, []);
  assert.equal(reuse.kvOwner, 21);
  assert.equal(reuse.indexOwner, 29);
  assert.equal(reuse.memoryEntries, reindex.memoryEntries);
});

test("bounded SwiGLU caps only the gate's upper end, bounds both up ends, and uses ordinary SiLU", () => {
  const upper = M.boundedSwiGLU(12, -14);
  assert.equal(upper.gate, 10);
  assert.equal(upper.up, -10);
  assert.equal(upper.product, (10 / (1 + Math.exp(-10))) * -10);
  const lower = M.boundedSwiGLU(-15, 12);
  assert.equal(lower.gate, -15);
  assert.equal(lower.up, 10);
  const inside = M.boundedSwiGLU(2, 3);
  assert.equal(inside.product, (2 / (1 + Math.exp(-2))) * 3);
  assert.equal(M.boundedSwiGLU(0, 9).product, 0);
});
