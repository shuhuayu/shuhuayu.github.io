((root) => {
  const spec = Object.freeze({
    dim: 5120,
    numLayers: 40,
    encoderLayers: 20,
    heads: 64,
    headDim: 512,
    ropeDim: 64,
    queryRank: 1280,
    outputGroups: 8,
    outputRank: 1024,
    indexHeads: 32,
    indexDim: 128,
    topK: 512,
    window: 128,
    experts: 384,
    activeExperts: 6,
    expertDim: 2304,
    hcCopies: 4,
    vocab: 129280,
    maxContext: 1048576,
    kvSources: Object.freeze([2, 8, 14, 20]),
    indexSources: Object.freeze([2, 8, 14, 20, 24, 28, 32, 36]),
    engramLayers: Object.freeze([1, 14]),
    engramRows: Object.freeze([384006168, 384016682]),
    engramHeads: 8,
    engramHeadDim: 256,
    candidateBlocks: 2048,
    candidateBlockSize: 8,
  });

  function layer(index) {
    if (!Number.isInteger(index) || index < 0 || index >= spec.numLayers) throw new RangeError("Layer index must be in [0, 40)");
    const ratio = index < 2 ? 0 : index < spec.encoderLayers ? 2 : 1;
    const kvSource = ratio ? spec.kvSources.filter((source) => source <= index).at(-1) : null;
    const indexSource = ratio ? spec.indexSources.filter((source) => source <= index).at(-1) : null;
    const mode = !ratio ? "SWA" : kvSource === index ? "Full" : indexSource === index ? "Reindex" : "Reuse";
    return Object.freeze({
      index,
      number: index + 1,
      half: index < spec.encoderLayers ? "Encoder" : "Decoder",
      ratio,
      mode,
      kvSource,
      indexSource,
      hasEngram: spec.engramLayers.includes(index),
    });
  }

  const layers = Object.freeze(Array.from({ length: spec.numLayers }, (_, index) => layer(index)));

  function validTokens(tokens) {
    if (!Number.isSafeInteger(tokens) || tokens < 1 || tokens > spec.maxContext) throw new RangeError("Context must be in [1, 1048576]");
  }

  function work(tokens, phase = "prefill") {
    validTokens(tokens);
    if (!["prefill", "decode", "training"].includes(phase)) throw new RangeError("Unknown execution phase");
    const encoderTokens = phase === "decode" ? 1 : tokens;
    const decoderTokens = phase === "prefill" ? Math.min(tokens, spec.window) : encoderTokens;
    const encoder = encoderTokens * spec.encoderLayers;
    const decoder = decoderTokens * (spec.numLayers - spec.encoderLayers);
    const baseline = encoderTokens * spec.numLayers;
    return { encoderTokens, decoderTokens, encoder, decoder, total: encoder + decoder, baseline, fraction: (encoder + decoder) / baseline };
  }

  function cache(tokens, mainBits = 4) {
    validTokens(tokens);
    if (![4, 16].includes(mainBits)) throw new RangeError("Main KV comparison must use FP4 or BF16");
    const producers = spec.kvSources.map((index) => ({ layer: index + 1, entries: Math.floor(tokens / layers[index].ratio) }));
    const entries = producers.reduce((total, producer) => total + producer.entries, 0);
    const unsharedEntries = layers.reduce((total, item) => total + (item.ratio ? Math.floor(tokens / item.ratio) : 0), 0);
    const mainEntryPayload = (spec.headDim * mainBits) / 8;
    const mainEntryScales = mainBits === 4 ? spec.headDim / 16 : 0;
    const indexEntryPayload = spec.indexDim / 2;
    const indexEntryScales = spec.indexDim / 32;
    const entryBytes = mainEntryPayload + mainEntryScales + indexEntryPayload + indexEntryScales;
    const mainPayload = entries * mainEntryPayload;
    const mainScales = entries * mainEntryScales;
    const indexPayload = entries * indexEntryPayload;
    const indexScales = entries * indexEntryScales;
    const local = spec.numLayers * Math.min(tokens, spec.window) * spec.headDim;
    const global = entries * entryBytes;
    return {
      producers,
      entries,
      unsharedEntries,
      mainPayload,
      mainScales,
      indexPayload,
      indexScales,
      main: mainPayload + mainScales,
      index: indexPayload + indexScales,
      local,
      global,
      total: global + local,
      unsharedGlobal: unsharedEntries * entryBytes,
      bytesPerToken: global / tokens,
    };
  }

  function engram() {
    const orders = [2, 3, 4];
    const rowsPerModule = orders.length * spec.engramHeads;
    const moduleInput = rowsPerModule * spec.engramHeadDim;
    const tableParams = spec.engramRows.reduce((total, rows) => total + rows * spec.engramHeadDim, 0);
    return {
      orders,
      rowsPerModule,
      rowsPerToken: rowsPerModule * spec.engramLayers.length,
      moduleInput,
      tableParams,
      fp8TableBytes: tableParams,
      lookupPayload: moduleInput * spec.engramLayers.length,
      projectionParamsPerModule: moduleInput * spec.dim * (spec.hcCopies + 1),
    };
  }

  function recentNgrams(text) {
    const tokens = text.trim().split(/\s+/).filter(Boolean);
    return [2, 3, 4].map((order) => {
      const recent = tokens.slice(-order);
      return { order, tokens: [...Array(Math.max(0, order - recent.length)).fill("<pad>"), ...recent] };
    });
  }

  function attentionExample(mode = "Reuse") {
    if (!["Full", "Reindex", "Reuse"].includes(mode)) throw new RangeError("Unknown CSA2 mode");
    // Reduced sizes and invented scores illustrate the decoder's inference search.
    const currentPosition = 25;
    const blockSize = 4;
    const numCandidateBlocks = 3;
    const topK = 3;
    const fullScores = [
      0.32, 0.94, 0.44, 0.21, 0.48, 0.36, 0.62, 0.19, 0.78, 0.82, 0.23, 0.14, 0.75, 0.27, 0.15, 0.3, 0.9, 0.67, 0.52, 0.4, 0.56, 0.98, 0.8, 0.2, 0.66,
      0.95, 0.84, 0.99, 0.72, 0.83, 0.6, 0.51,
    ];
    const reindexScores = [
      0.99, 0.11, 0.44, 0.2, 0.4, 0.5, 0.6, 0.3, 0.35, 0.999, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.28, 0.97, 0.6, 0.2, 0.1, 0.3, 0.95, 0.5, 0.77, 0.85,
      0.88, 0.999, 0.8, 0.6, 0.5, 0.4,
    ];
    const visible = Array.from({ length: currentPosition }, (_, index) => index);
    const rank = (positions, scores) => [...positions].sort((a, b) => scores[b] - scores[a] || a - b);
    const blocks = Array.from({ length: Math.ceil(fullScores.length / blockSize) }, (_, index) => {
      const positions = visible.filter((position) => Math.floor(position / blockSize) === index);
      return { index, positions, score: positions.length ? Math.max(...positions.map((position) => fullScores[position])) : -Infinity };
    });
    const candidateBlocks = [...blocks]
      .filter((block) => block.positions.length)
      .sort((a, b) => b.score - a.score || a.index - b.index)
      .slice(0, numCandidateBlocks)
      .map((block) => block.index);
    const candidates = visible.filter((position) => candidateBlocks.includes(Math.floor(position / blockSize)));
    const selected = rank(mode === "Full" ? visible : candidates, mode === "Full" ? fullScores : reindexScores).slice(0, topK);
    const scored = mode === "Full" ? visible : mode === "Reindex" ? candidates : [];
    return {
      currentPosition,
      blockSize,
      numCandidateBlocks,
      topK,
      kvOwner: 21,
      indexOwner: mode === "Full" ? 21 : 29,
      memoryEntries: currentPosition,
      selected: selected.map((position) => position + 1).sort((a, b) => a - b),
      scoredPositions: scored.map((position) => position + 1),
      candidatePositions: candidates.map((position) => position + 1),
      positions: fullScores.map((_, index) => ({
        position: index + 1,
        block: Math.floor(index / blockSize) + 1,
        visible: index < currentPosition,
        candidate: candidates.includes(index),
        selected: selected.includes(index),
        score: scored.includes(index) ? (mode === "Full" ? fullScores[index] : reindexScores[index]) : null,
      })),
    };
  }

  function boundedSwiGLU(gate, up) {
    const boundedGate = Math.min(gate, 10);
    const boundedUp = Math.max(-10, Math.min(up, 10));
    const siluGate = boundedGate / (1 + Math.exp(-boundedGate));
    return { gate: boundedGate, up: boundedUp, siluGate, product: siluGate * boundedUp };
  }

  const api = Object.freeze({ spec, layers, layer, work, cache, engram, recentNgrams, attentionExample, boundedSwiGLU });
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.DeepSeekV41 = api;
})(typeof globalThis === "object" ? globalThis : this);
