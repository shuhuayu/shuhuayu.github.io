((root) => {
  // Logical, unsharded dimensions from the official V4 release configs.
  // The final compression-ratio entry belongs to MTP, not the decoder stack.
  const shared = {
    headDim: 512,
    ropeDim: 64,
    outputRank: 1024,
    indexHeads: 64,
    indexDim: 128,
    window: 128,
    activeExperts: 6,
    sharedExperts: 1,
    hashLayers: 3,
    hcCopies: 4,
    sinkhornSteps: 20,
    vocab: 129280,
    maxContext: 1048576,
    mtpLayers: 1,
  };
  const models = Object.freeze({
    pro: Object.freeze({
      ...shared,
      name: "DeepSeek V4 Pro",
      dim: 7168,
      numLayers: 61,
      heads: 128,
      queryRank: 1536,
      outputGroups: 16,
      experts: 384,
      expertDim: 3072,
      topK: 1024,
      reportedTotal: "1.6T",
      reportedActive: "49B",
      ratios: Object.freeze([128, 128, ...Array.from({ length: 59 }, (_, i) => (i % 2 ? 128 : 4))]),
    }),
    flash: Object.freeze({
      ...shared,
      name: "DeepSeek V4 Flash",
      dim: 4096,
      numLayers: 43,
      heads: 64,
      queryRank: 1024,
      outputGroups: 8,
      experts: 256,
      expertDim: 2048,
      topK: 512,
      reportedTotal: "284B",
      reportedActive: "13B",
      ratios: Object.freeze([0, 0, ...Array.from({ length: 41 }, (_, i) => (i % 2 ? 128 : 4))]),
    }),
  });

  function spec(id) {
    if (!Object.hasOwn(models, id)) throw new RangeError("Choose pro or flash");
    return models[id];
  }

  function layers(id) {
    return spec(id).ratios.map((ratio, index) => ({
      index,
      number: index + 1,
      ratio,
      mode: ratio === 4 ? "CSA" : ratio === 128 ? "HCA" : "SWA",
      hash: index < spec(id).hashLayers,
    }));
  }

  function counts(id) {
    const list = layers(id);
    return Object.fromEntries(["CSA", "HCA", "SWA"].map((mode) => [mode, list.filter((item) => item.mode === mode).length]));
  }

  function cache(id, tokens, precision = "mixed", requests = 1) {
    const s = spec(id);
    if (!Number.isSafeInteger(tokens) || tokens < 1 || tokens > s.maxContext) throw new RangeError("Context must be in [1, 1048576]");
    if (!Number.isSafeInteger(requests) || requests < 1 || requests > 128) throw new RangeError("Requests must be in [1, 128]");
    if (!["mixed", "bf16"].includes(precision)) throw new RangeError("Choose mixed or bf16 cache payload");
    const c = counts(id);
    const mainEntry = precision === "mixed" ? s.headDim - s.ropeDim + 2 * s.ropeDim : 2 * s.headDim;
    const indexEntry = precision === "mixed" ? s.indexDim / 2 : 2 * s.indexDim;
    const csaEntries = c.CSA * Math.floor(tokens / 4);
    const hcaEntries = c.HCA * Math.floor(tokens / 128);
    const local = s.numLayers * Math.min(tokens, s.window) * mainEntry * requests;
    const csa = csaEntries * mainEntry * requests;
    const hca = hcaEntries * mainEntry * requests;
    const index = csaEntries * indexEntry * requests;
    const total = local + csa + hca + index;
    const baseline = s.numLayers * tokens * 2 * 8 * 128 * 2 * requests;
    return {
      tokens,
      mainEntry,
      indexEntry,
      csaEntries,
      hcaEntries,
      local,
      csa,
      hca,
      index,
      total,
      baseline,
      fraction: total / baseline,
      csaSlots: Math.min(s.topK, Math.floor(tokens / 4)) + Math.min(tokens, s.window),
      hcaSlots: Math.floor(tokens / 128) + Math.min(tokens, s.window),
    };
  }

  function parameters(id) {
    const s = spec(id);
    const c = counts(id);
    const { dim: d, heads: h, headDim: k, queryRank: q, expertDim: f, numLayers: l } = s;
    const oneExpert = 3 * d * f;
    const baseAttention = d * q + q * h * k + d * k + h * k * s.outputRank + s.outputGroups * s.outputRank * d;
    const hcControls = (2 + s.hcCopies) * s.hcCopies;
    const hcPerLayer = 2 * (hcControls * s.hcCopies * d + hcControls + 3);
    const baseNorms = q + k + h + 2 * d;
    const compressorMatrices = (4 * c.CSA + 2 * c.HCA) * d * k;
    const compressorSmall = c.CSA * (8 * k + k) + c.HCA * (128 * k + k);
    const indexMatrices = c.CSA * (q * s.indexHeads * s.indexDim + d * s.indexHeads + 4 * d * s.indexDim);
    const indexSmall = c.CSA * (8 * s.indexDim + s.indexDim);
    const headMix = s.hcCopies * s.hcCopies * d + s.hcCopies + 1;
    const rows = [
      { key: "routed", label: "Routed expert banks", value: l * s.experts * oneExpert, formula: `${l} × ${s.experts} × 3DF` },
      { key: "shared", label: "Always-on shared experts", value: l * s.sharedExperts * oneExpert, formula: `${l} × 1 × 3DF` },
      { key: "attention", label: "Query, local KV, grouped output", value: l * baseAttention, formula: "L × (Dq + qHK + DK + HKr + GrD)" },
      {
        key: "compressors",
        label: "Main KV compressors",
        value: compressorMatrices + compressorSmall,
        formula: "CSA: 4DK; HCA: 2DK; plus position biases / norms",
      },
      {
        key: "indexers",
        label: "CSA lightning indexers",
        value: indexMatrices + indexSmall,
        formula: "CSA × (qHᵢKᵢ + DHᵢ + 4DKᵢ), plus biases / norms",
      },
      {
        key: "routing",
        label: "MoE routing projections + bias",
        value: l * d * s.experts + (l - s.hashLayers) * s.experts,
        formula: "LDE + (L − 3)E",
      },
      { key: "mhc", label: "mHC transforms + final mixing", value: l * hcPerLayer + headMix, formula: "2L × (24 × 4D + 27), plus final 4 × 4D mix" },
      { key: "embedding", label: "Embedding + untied vocabulary head", value: 2 * d * s.vocab, formula: "2DV" },
      { key: "norms", label: "Other RMSNorm weights + attention sinks", value: l * baseNorms + d, formula: "L × (q + K + H + 2D) + D" },
    ];
    const total = rows.reduce((sum, row) => sum + row.value, 0);
    const mtpBlock = (s.experts + s.sharedExperts) * oneExpert + baseAttention + hcPerLayer + baseNorms + d * s.experts + s.experts;
    const mtp = s.mtpLayers * (mtpBlock + 2 * d * d + 3 * d + headMix);
    return { rows, total, mtp, withMtp: total + mtp, oneExpert, routedFraction: rows[0].value / total, baseAttention };
  }

  const fast = Object.freeze([3.4445, -4.775, 2.0315]);
  const stable = Object.freeze([2, -1.5, 0.5]);
  const nsSchedule = Object.freeze([...Array(8).fill(fast), ...Array(2).fill(stable)]);

  function polynomial(value, coefficients) {
    const [a, b, c] = coefficients;
    return a * value + b * value ** 3 + c * value ** 5;
  }

  function singularValues(initial, schedule = nsSchedule) {
    if (!Number.isFinite(initial) || initial < 0 || initial > 1) throw new RangeError("Initial normalized singular value must be in [0, 1]");
    return schedule.reduce((values, coefficients) => [...values, polynomial(values.at(-1), coefficients)], [initial]);
  }

  const api = Object.freeze({ models, spec, layers, counts, cache, parameters, fast, stable, nsSchedule, polynomial, singularValues });
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.DeepSeekV4 = api;
})(typeof globalThis === "object" ? globalThis : this);
