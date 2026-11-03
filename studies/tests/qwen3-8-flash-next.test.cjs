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

test("the residual example starts with copies but produces distinct stream updates", () => {
  const first = model.residualExample("first");
  for (const stream of first.hidden) assert.deepEqual(stream, [1, 2]);
  assert.equal(new Set(first.writeGates).size, 4);
  assert.equal(new Set(first.output.map((stream) => JSON.stringify(stream))).size, 4);
  for (const bias of [-12, 0, 12]) {
    const example = model.residualExample("later", bias);
    for (let stream = 0; stream < 4; stream++) {
      assert(example.writeGates[stream] > 0 && example.writeGates[stream] < 2);
      for (let channel = 0; channel < 2; channel++) {
        assert(example.readGates[stream][channel] > 0 && example.readGates[stream][channel] < 1);
        assert(
          Math.abs(example.output[stream][channel] - example.hidden[stream][channel] - example.writeGates[stream] * example.update[channel]) < 1e-12
        );
      }
    }
  }
});

test("the delta-rule example can forget, preserve, and overwrite an addressed value", () => {
  const overwrite = model.deltaExample(0.9, 1);
  for (let channel = 0; channel < 2; channel++) assert(Math.abs(overwrite.output[channel] - overwrite.value[channel]) < 1e-12);
  assert.deepEqual(overwrite.state[1], [2.7, 3.6]);
  assert.deepEqual(model.deltaExample(1, 0).state, [
    [1, 2],
    [3, 4],
  ]);
  assert.deepEqual(model.deltaExample(0, 0).state, [
    [0, 0],
    [0, 0],
  ]);
  assert.deepEqual(model.deltaExample(0.9, 1, "other").output, [2.7, 3.6]);
});

test("the lookup addresses match the independently evaluated pinned Python hash reference", () => {
  assert.deepEqual(
    model
      .lexicalHashes([10, 20, 30])
      .at(-1)
      .map((entry) => entry.row),
    [
      9878115, 26555603, 54895210, 62571545, 80580723, 119917398, 128922427, 147596134, 168936175, 195223391, 219226064, 233524685, 246670267,
      279816194, 297531600, 306108296,
    ]
  );
  const rows = model.lexicalHeadSizes.reduce((sum, size) => sum + size, 0);
  assert.equal(Math.ceil(rows / 128) * 128, 320001536);
  assert.deepEqual(
    model
      .lexicalHashes([248319, 248318, 248317])
      .at(-1)
      .map((entry) => entry.row),
    [
      18204799, 30983358, 48501439, 78291015, 93570287, 105570872, 133797543, 158521149, 178582661, 198209494, 218766759, 226248959, 248124088,
      270195567, 290929800, 314927571,
    ]
  );
  for (const entries of model.lexicalHashes([248319, 248318, 248317, 248044])) {
    for (const entry of entries) {
      assert(entry.bucket >= 0 && entry.bucket < model.lexicalHeadSizes[entry.head]);
      assert(entry.row >= entry.offset && entry.row < entry.offset + model.lexicalHeadSizes[entry.head]);
    }
  }
});

test("hashes encode ordered token content while document and EOS boundaries reset history", () => {
  const baseline = model.lexicalHashes([10, 20, 30]).at(-1);
  assert.deepEqual(model.lexicalHashes([10, 20, 30], [100, 101, 102]).at(-1), baseline);
  const differentOlderToken = model.lexicalHashes([40, 20, 30]).at(-1);
  assert.deepEqual(differentOlderToken.slice(0, 8), baseline.slice(0, 8));
  assert.notDeepEqual(differentOlderToken.slice(8), baseline.slice(8));
  assert.notDeepEqual(model.lexicalHashes([10, 50, 30]).at(-1), baseline);
  const fresh = model.lexicalHashes([30])[0];
  assert.deepEqual(model.lexicalHashes([10, 20, 30], [0, 1, 0]).at(-1), fresh);
  assert.deepEqual(model.lexicalHashes([10, 248044, 30]).at(-1), fresh);
});

test("processed visual grids preserve temporal groups and two-by-two spatial merging", () => {
  assert.deepEqual(model.visualTokens(448, 2), { spatial: 28, temporal: 1, patches: 784, merged: 196 });
  assert.equal(model.visualTokens(448, 4).merged, 392);
  assert.equal(model.visualTokens(896, 8).patches, 12544);
  assert.equal(model.visualTokens(896, 8).merged, 3136);
});

test("SP changes carried activation sizes without shrinking the selected expert workload", () => {
  const sharded = model.trainingFootprint(4096, 2, 2, true);
  const replicated = model.trainingFootprint(4096, 2, 2, false);
  assert.equal(sharded.residual, 80 * 2 ** 20);
  assert.equal(sharded.localResidual, 40 * 2 ** 20);
  assert.equal(replicated.localResidual, sharded.residual);
  assert.equal(sharded.localResidual / sharded.localSublayer, 4);
  assert.equal(sharded.assignmentPayload, replicated.assignmentPayload);
  assert.equal(sharded.assignmentPayload, 200 * 2 ** 20);
  assert.equal(sharded.tokenMetadata, 64 * 1024);
  assert.equal(sharded.lexicalRead, 20 * 2 ** 20);
  assert.equal(sharded.lexicalWeights, audit.groups.lexical_table * 2);
  assert.equal(sharded.lexicalMoments, audit.groups.lexical_table * 8);
});
