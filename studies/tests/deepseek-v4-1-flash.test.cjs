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
