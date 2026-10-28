const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const model = require("../qwen3-8-flash-next-model.js");
const audit = JSON.parse(fs.readFileSync(path.join(__dirname, "../qwen3-8-flash-next-checkpoint.json"), "utf8"));

test("diagram dimensions and layer positions match the pinned released config", () => {
  const cfg = audit.config.text_config;
  assert.equal(model.config.revision, audit.revision);
  for (const [name, field] of Object.entries({
    layers: "num_hidden_layers",
    dim: "hidden_size",
    streams: "hc_count",
    mixingRank: "hc_lowrank",
    vocab: "vocab_size",
    experts: "num_experts",
    topK: "num_experts_per_tok",
    expertWidth: "moe_intermediate_size",
    tokenBudget: "indexer_budget",
    blockSize: "indexer_compress_ratio",
    indexerHeadDim: "indexer_head_dim",
    qsaQueryHeads: "num_attention_heads",
    qsaKVHeads: "num_key_value_heads",
    qsaHeadDim: "head_dim",
    nativeContext: "max_position_embeddings",
  })) {
    assert.equal(model.config[name], cfg[field], name);
  }
  for (let layer = 1; layer <= cfg.num_hidden_layers; layer++) {
    const spec = model.layerSpec(layer);
    assert.equal(spec.attention, cfg.layer_types[layer - 1] === "linear_attention" ? "gdn" : "qsa");
    assert.equal(spec.ple, cfg.ple_layer_ids.includes(layer));
  }
});

test("all parameter formulas match independently read safetensors headers", () => {
  for (const group of model.parameterGroups) assert.equal(group.total, audit.groups[group.id], group.id);
  assert.equal(model.parameterCount("stored").total, 179999981424);
  assert.equal(model.backboneParams, 177392830576);
  assert.equal(model.parameterCount("backbone").total, 125743653760);
  assert.equal(model.parameterCount("active").total, 6671295360);
});

test("audit inventory reconciles every weight and integer buffer with the checkpoint payload", () => {
  const tensors = Object.entries(audit.tensors);
  assert.equal(tensors.length, 1658);
  assert.equal(tensors.filter(([name]) => name.startsWith("mtp.")).length, 31);
  assert.equal(new Set(tensors.map(([, info]) => info.shard)).size, 131);
  let bytes = 0;
  let bf16Elements = 0;
  let integerElements = 0;
  for (const [, info] of tensors) {
    const elements = info.shape.reduce((product, dim) => product * dim, 1);
    assert.equal(elements, info.numel);
    if (info.dtype === "BF16") {
      bf16Elements += elements;
      bytes += 2 * elements;
    } else {
      assert.equal(info.dtype, "I64");
      integerElements += elements;
      bytes += 8 * elements;
    }
  }
  assert.equal(bf16Elements, 179999981424);
  assert.equal(integerElements, 35);
  assert.equal(bytes, 359999963128);
  assert.equal(bytes, audit.weight_bytes);
});

test("native-context cache agrees with hand-derived GiB and MiB values", () => {
  const sizes = model.cacheBytes(262144);
  assert.equal(sizes.qsa, 6 * 2 ** 30);
  assert.equal(sizes.indexer, 768 * 2 ** 20);
  assert.equal(sizes.gdn, 108 * 2 ** 20);
  assert.equal(sizes.total, 6.85546875 * 2 ** 30);
  assert.equal(sizes.baseline, 24 * 2 ** 30);
  const longer = model.cacheBytes(1000000);
  assert.equal(longer.gdn, sizes.gdn);
  assert.equal(longer.qsa / sizes.qsa, 1000000 / 262144);
  const pooled = model.cacheBytes(262144, 2, 4, "pooled");
  assert.equal(pooled.indexer, 192 * 2 ** 20);
  assert.equal(pooled.qsa, sizes.qsa);
  assert.equal(pooled.gdn, sizes.gdn);
});

test("block selection chooses complete top blocks and always includes the partial tail", () => {
  const result = model.selectBlocks(19, 8, [0.1, 0.9, 0.3, 0.8]);
  assert.deepEqual(result.blocks, [1, 3]);
  assert.deepEqual(result.selected, [4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18]);
  assert.deepEqual(result.tail, [16, 17, 18]);
  assert.deepEqual(model.selectBlocks(3, 2048, []).selected, [0, 1, 2]);
  assert.deepEqual(model.selectBlocks(4, 2048, [1]).selected, [0, 1, 2, 3]);
});

test("every demo prefix preserves causality, whole blocks, and the budget-plus-tail bound", () => {
  for (let visible = 1; visible <= 64; visible++) {
    for (const budget of [4, 8, 16, 2048]) {
      const selection = model.selectBlocks(
        visible,
        budget,
        Array.from({ length: 16 }, (_, block) => block + 1)
      );
      assert(selection.selected.every((token) => token < visible));
      assert.equal(new Set(selection.selected).size, selection.selected.length);
      assert.equal(selection.tail.length, visible % 4);
      assert(selection.selected.length <= budget + 3);
      for (const block of selection.blocks) assert.equal(selection.selected.filter((token) => Math.floor(token / 4) === block).length, 4);
      for (let token = visible - (visible % 4); token < visible; token++) assert(selection.selected.includes(token));
    }
  }
});
