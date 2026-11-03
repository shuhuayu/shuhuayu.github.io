(function (root, factory) {
  const model = factory();
  if (typeof module === "object" && module.exports) module.exports = model;
  else root.QwenFlashNextModel = model;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const config = Object.freeze({
    revision: "de4b8e4d43b917e7706784d8bb445c9af86a3540",
    layers: 48,
    dim: 2560,
    streams: 4,
    mixingRank: 320,
    vocab: 248320,
    experts: 512,
    topK: 10,
    expertWidth: 640,
    gdnLayers: 36,
    qsaLayers: 12,
    gdnKeyHeads: 16,
    gdnValueHeads: 48,
    gdnHeadDim: 128,
    qsaQueryHeads: 24,
    qsaKVHeads: 2,
    qsaHeadDim: 256,
    indexerQueryHeads: 4,
    indexerHeadDim: 128,
    blockSize: 4,
    tokenBudget: 2048,
    lexicalRows: 320001536,
    lexicalHeadDim: 160,
    lexicalHeads: 16,
    lexicalParts: 128,
    visionParams: 448931056,
    checkpointBF16Elements: 179999981424,
    checkpointIntegerElements: 35,
    nativeContext: 262144,
  });
  const { dim: D, streams: S, layers: L, expertWidth: F, experts: E } = config;
  const C = S * D;
  const gdnChannels = 2 * config.gdnKeyHeads * config.gdnHeadDim + config.gdnValueHeads * config.gdnHeadDim;
  const gdnValueChannels = config.gdnValueHeads * config.gdnHeadDim;
  const expertParams = 3 * D * F;

  function residualParams(write = true) {
    return C + 2 * C * config.mixingRank + (write ? C * S : 0);
  }

  const gdnLayerParams = D * gdnChannels + 2 * D * gdnValueChannels + 2 * D * config.gdnValueHeads + 4 * gdnChannels + 2 * config.gdnValueHeads + 128;
  const qsaLayerParams =
    D * config.qsaQueryHeads * config.qsaHeadDim * 3 +
    2 * D * config.qsaKVHeads * config.qsaHeadDim +
    2 * config.qsaHeadDim +
    D * 5 * config.indexerHeadDim +
    2 * config.indexerHeadDim;
  const backboneGroups = [
    { id: "routed_experts", name: "Routed expert bank", total: L * E * expertParams, active: L * config.topK * expertParams, color: "#7652b8" },
    { id: "lexical_table", name: "N-gram lookup table", total: config.lexicalRows * config.lexicalHeadDim, active: 0, color: "#a87816" },
    { id: "gdn", name: "36 Gated DeltaNet layers", total: config.gdnLayers * gdnLayerParams, color: "#0d887d" },
    { id: "gated_residual", name: "Gated residuals + final mixer", total: 2 * L * residualParams() + residualParams(false), color: "#b84d6d" },
    { id: "input_embedding", name: "Input embedding", total: config.vocab * D, active: 0, color: "#718096" },
    { id: "output_head", name: "LM output head", total: config.vocab * D, color: "#3974ad" },
    { id: "qsa", name: "12 QSA layers + indexers", total: config.qsaLayers * qsaLayerParams, color: "#d46b2c" },
    { id: "vision", name: "Vision encoder + merger", total: config.visionParams, active: 0, color: "#568f54" },
    { id: "shared_expert", name: "Shared experts + gates", total: L * (expertParams + D), color: "#9b72d0" },
    { id: "router", name: "MoE routers", total: L * D * E, color: "#ba68a0" },
    { id: "lexical_injection", name: "PLE projections + norms + convolution", total: D * C + D * D + 3 * C + 4 * C, color: "#b18b4b" },
  ].map((item) => Object.freeze({ ...item, active: item.active ?? item.total }));
  const backboneParams = backboneGroups.reduce((sum, group) => sum + group.total, 0);
  const parameterGroups = Object.freeze([
    ...backboneGroups,
    Object.freeze({ id: "mtp", name: "MTP unique weights", total: config.checkpointBF16Elements - backboneParams, active: 0, color: "#737b8c" }),
  ]);

  function parameterCount(mode = "stored") {
    if (!["stored", "backbone", "active"].includes(mode)) throw new RangeError("Unknown parameter accounting mode");
    const groups = parameterGroups.map((group) => ({
      ...group,
      value: mode === "active" ? group.active : mode === "backbone" && ["lexical_table", "vision", "mtp"].includes(group.id) ? 0 : group.total,
    }));
    return { groups, total: groups.reduce((sum, group) => sum + group.value, 0) };
  }

  function layerSpec(layer) {
    if (!Number.isInteger(layer) || layer < 1 || layer > L) throw new RangeError("Layer must be an integer from 1 to 48");
    return { layer, attention: layer % 4 === 0 ? "qsa" : "gdn", ple: layer === 2 };
  }

  function cacheBytes(tokens, elementBytes = 2, stateBytes = 4, indexerStorage = "reference") {
    if (!Number.isInteger(tokens) || tokens < 1) throw new RangeError("Context length must be a positive integer");
    if (![1, 2, 4].includes(elementBytes) || ![2, 4].includes(stateBytes)) throw new RangeError("Unsupported cache dtype");
    if (!["reference", "pooled"].includes(indexerStorage)) throw new RangeError("Unknown indexer storage convention");
    const qsa = tokens * config.qsaLayers * 2 * config.qsaKVHeads * config.qsaHeadDim * elementBytes;
    const indexerEntries = indexerStorage === "reference" ? tokens : Math.floor(tokens / config.blockSize) + (tokens % config.blockSize);
    const indexer = indexerEntries * config.qsaLayers * config.indexerHeadDim * elementBytes;
    const gdn = config.gdnLayers * config.gdnValueHeads * config.gdnHeadDim * config.gdnHeadDim * stateBytes;
    const baseline = tokens * L * 2 * config.qsaKVHeads * config.qsaHeadDim * elementBytes;
    return { qsa, indexer, gdn, total: qsa + indexer + gdn, baseline };
  }

  function selectBlocks(visibleTokens, budget, scores) {
    if (!Number.isInteger(visibleTokens) || visibleTokens < 1) throw new RangeError("Visible tokens must be a positive integer");
    if (!Number.isInteger(budget) || budget < config.blockSize || budget % config.blockSize !== 0)
      throw new RangeError("Budget must be a positive multiple of four");
    const completeBlocks = Math.floor(visibleTokens / config.blockSize);
    if (scores.length < completeBlocks || scores.slice(0, completeBlocks).some((score) => !Number.isFinite(score)))
      throw new RangeError("Each complete block needs a finite score");
    const ranked = Array.from({ length: completeBlocks }, (_, block) => block).sort((a, b) => scores[b] - scores[a] || a - b);
    const blocks = ranked.slice(0, budget / config.blockSize).sort((a, b) => a - b);
    const tail = Array.from({ length: visibleTokens % config.blockSize }, (_, i) => completeBlocks * config.blockSize + i);
    const selected = [...blocks.flatMap((block) => Array.from({ length: config.blockSize }, (_, i) => block * config.blockSize + i)), ...tail];
    return { blocks, tail, selected, completeBlocks };
  }

  const sigmoid = (value) => 1 / (1 + Math.exp(-value));
  const dot = (left, right) => left.reduce((sum, value, index) => sum + value * right[index], 0);
  const project = (weight, input) => weight.map((row) => dot(row, input));

  function residualExample(entry = "later", writeBias = 0) {
    if (!["first", "later"].includes(entry) || !Number.isFinite(writeBias)) throw new RangeError("Invalid residual example");
    const hidden =
      entry === "first"
        ? Array.from({ length: 4 }, () => [1, 2])
        : [
            [1, 2],
            [2, -1],
            [-1, 1],
            [0.5, 3],
          ];
    const normalized = hidden.map((stream) => {
      const rms = Math.sqrt(dot(stream, stream) / stream.length + 1e-6);
      return stream.map((value) => value / rms);
    });
    const down = [
      [0.5, 0, 0.25, 0, -0.5, 0, 0.25, 0],
      [0, 0.5, 0, -0.25, 0, 0.5, 0, 0.25],
    ];
    const up = [
      [2, 0.5],
      [0.5, -1],
      [-1, 0.5],
      [0, 2],
      [0.5, 1],
      [-1, -0.5],
      [1, -1],
      [0.5, 0.5],
    ];
    const write = [
      [1, 0, 0.2, 0, 0, 0.1, 0, 0],
      [0, 0.2, 2, 0, 0.1, 0, 0, 0],
      [0, 0, 0.1, 0, -1, 0.3, 0, 0],
      [0.1, 0, 0, 0.1, 0, 0, 0.5, 0],
    ];
    const flat = normalized.flat();
    const bottleneck = project(down, flat).map((value) => (value / 4) * sigmoid(value / 4));
    const gates = project(up, bottleneck).map(sigmoid);
    const readGates = Array.from({ length: 4 }, (_, stream) => gates.slice(stream * 2, stream * 2 + 2));
    const input = [0, 1].map((channel) => normalized.reduce((sum, stream, index) => sum + stream[channel] * readGates[index][channel], 0) / 4);
    const update = [0.8, -0.4];
    const writeLogits = project(write, flat).map((value) => value + writeBias);
    const writeGates = writeLogits.map((value) => 2 * sigmoid(value / 4));
    const output = hidden.map((stream, index) => stream.map((value, channel) => value + writeGates[index] * update[channel]));
    return { hidden, normalized, bottleneck, readGates, input, update, writeLogits, writeGates, output, down, up, write };
  }

  function deltaExample(decay = 0.9, beta = 0.5, query = "match") {
    if (decay < 0 || decay > 1 || beta < 0 || beta > 1 || !["match", "other", "mixed"].includes(query))
      throw new RangeError("Invalid delta-rule example");
    const previous = [
      [1, 2],
      [3, 4],
    ];
    const key = [1, 0];
    const value = [2, -1];
    const q = query === "match" ? [1, 0] : query === "other" ? [0, 1] : [Math.SQRT1_2, Math.SQRT1_2];
    const decayed = previous.map((row) => row.map((entry) => entry * decay));
    const prediction = [0, 1].map((column) => key.reduce((sum, entry, row) => sum + entry * decayed[row][column], 0));
    const error = value.map((entry, column) => entry - prediction[column]);
    const correction = key.map((entry) => error.map((difference) => beta * entry * difference));
    const state = decayed.map((row, index) => row.map((entry, column) => entry + correction[index][column]));
    const output = [0, 1].map((column) => q.reduce((sum, entry, row) => sum + entry * state[row][column], 0));
    return { previous, key, value, q, decay, beta, decayed, prediction, error, correction, state, output };
  }

  function primeHeadSizes() {
    const result = [];
    for (let candidate = 20000000; result.length < config.lexicalHeads; candidate++) {
      let prime = true;
      for (let divisor = 2; divisor * divisor <= candidate; divisor++) {
        if (candidate % divisor === 0) {
          prime = false;
          break;
        }
      }
      if (prime) result.push(candidate);
    }
    return Object.freeze(result);
  }

  const lexicalHeadSizes = primeHeadSizes();
  const lexicalHeadOffsets = Object.freeze(lexicalHeadSizes.map((_, head) => lexicalHeadSizes.slice(0, head).reduce((sum, size) => sum + size, 0)));
  const mask64 = (1n << 64n) - 1n;
  const halfBound = ((1n << 63n) - 1n) / BigInt(config.vocab) / 2n;
  const lexicalMultipliers = Object.freeze(
    Array.from({ length: 3 }, (_, position) => {
      // The reference advances once per position and once inside SplitMix64.
      let value = (1234n + 0x9e3779b97f4a7c15n * BigInt(position + 2)) & mask64;
      value = ((value ^ (value >> 30n)) * 0xbf58476d1ce4e5b9n) & mask64;
      value = ((value ^ (value >> 27n)) * 0x94d049bb133111ebn) & mask64;
      value ^= value >> 31n;
      return 2n * (value % halfBound) + 1n;
    })
  );

  function lexicalHashes(tokenIds, documentPositions = tokenIds.map((_, index) => index)) {
    if (
      !tokenIds.length ||
      tokenIds.length !== documentPositions.length ||
      tokenIds.some((id) => !Number.isInteger(id) || id < 0 || id >= config.vocab)
    )
      throw new RangeError("Expected token IDs and document positions of equal, nonzero length");
    let start = 0;
    return tokenIds.map((current, position) => {
      if (documentPositions[position] === 0 || (position > 0 && tokenIds[position - 1] === 248044)) start = position;
      const history = [0, 1, 2].map((shift) => (position - shift >= start ? tokenIds[position - shift] : 248044));
      const products = history.map((id, index) => BigInt.asIntN(64, BigInt(id) * lexicalMultipliers[index]));
      const pair = BigInt.asIntN(64, products[0] ^ products[1]);
      const triple = BigInt.asIntN(64, pair ^ products[2]);
      return lexicalHeadSizes.map((size, head) => {
        const mixed = head < 8 ? pair : triple;
        const bucket = Number(((mixed % BigInt(size)) + BigInt(size)) % BigInt(size));
        return { head, order: head < 8 ? 2 : 3, bucket, offset: lexicalHeadOffsets[head], row: bucket + lexicalHeadOffsets[head] };
      });
    });
  }

  function visualTokens(resolution = 448, frames = 2) {
    if (!Number.isInteger(resolution) || resolution < 32 || resolution % 32 || !Number.isInteger(frames) || frames < 2 || frames % 2)
      throw new RangeError("Processed resolution must be a multiple of 32; frames must be a positive multiple of two");
    const spatial = resolution / 16;
    const temporal = frames / 2;
    return { spatial, temporal, patches: spatial * spatial * temporal, merged: (spatial / 2) ** 2 * temporal };
  }

  function trainingFootprint(tokens = 4096, elementBytes = 2, tp = 2, sequenceParallel = true) {
    if (!Number.isInteger(tokens) || tokens < 1 || ![2, 4].includes(elementBytes) || ![1, 2, 4, 8].includes(tp) || tokens % tp)
      throw new RangeError("Invalid activation accounting inputs");
    const localTokens = sequenceParallel ? tokens / tp : tokens;
    return {
      tokens,
      localTokens,
      tp,
      sequenceParallel,
      residual: tokens * C * elementBytes,
      localResidual: localTokens * C * elementBytes,
      sublayer: tokens * D * elementBytes,
      localSublayer: localTokens * D * elementBytes,
      assignments: tokens * config.topK,
      assignmentPayload: tokens * config.topK * D * elementBytes,
      tokenMetadata: tokens * 2 * 8,
      lexicalRows: tokens * config.lexicalHeads,
      lexicalRead: tokens * config.lexicalHeads * config.lexicalHeadDim * 2,
      lexicalWeights: config.lexicalRows * config.lexicalHeadDim * 2,
      lexicalMoments: config.lexicalRows * config.lexicalHeadDim * 8,
    };
  }

  return Object.freeze({
    config,
    parameterGroups,
    backboneParams,
    gdnLayerParams,
    qsaLayerParams,
    residualParams,
    parameterCount,
    layerSpec,
    cacheBytes,
    selectBlocks,
    residualExample,
    deltaExample,
    lexicalHeadSizes,
    lexicalHeadOffsets,
    lexicalHashes,
    visualTokens,
    trainingFootprint,
  });
});
