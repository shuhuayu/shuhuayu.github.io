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
  });
});
