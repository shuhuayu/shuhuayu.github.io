((root, factory) => {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.QwenFlashNextAccounting = api;
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  "use strict";

  // Logical dimensions of the released text decoder (no vision encoder, no MTP).
  const D = 2560;
  const S = 4;
  const C = S * D;
  const V = 248320;
  const E = 512;
  const topK = 10;
  const F = 640;
  const L = 48;
  const hcRank = 320;
  const gdn = Object.freeze({ keyHeads: 16, valueHeads: 48, headDim: 128, conv: 4, chunk: 64 });
  const qsa = Object.freeze({ queryHeads: 24, kvHeads: 2, headDim: 256, indexerHeads: 4, indexerHeadDim: 128, blockSize: 4, budget: 2048 });
  // Hash-head row ranges: 8 bigram + 8 trigram heads, padded to a multiple of 128 rows.
  const pleHeadSizes = Object.freeze([
    20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081, 20000093, 20000107, 20000147, 20000153, 20000159,
    20000161, 20000171,
  ]);
  const ple = Object.freeze({ layer: 1, rows: 320001536, rowDim: 160, heads: 16, conv: 4, headSizes: pleHeadSizes });

  // TorchTitan's fixed request capacity: an owner receives at most
  // min(tokens, overlap) distinct IDs per head from one requester.
  function pleRequestCapacity(tokens, owners) {
    const rowsPerOwner = Math.floor(ple.rows / owners);
    const offsets = [0];
    for (const size of pleHeadSizes) offsets.push(offsets.at(-1) + size);
    let best = 0;
    for (let owner = 0; owner < owners; owner++) {
      const first = owner * rowsPerOwner;
      const last = first + rowsPerOwner;
      let total = 0;
      for (let h = 0; h < pleHeadSizes.length; h++)
        total += Math.min(tokens, Math.max(0, Math.min(offsets[h + 1], last) - Math.max(offsets[h], first)));
      best = Math.max(best, total);
    }
    return best;
  }
  const sum = (items, get) => items.reduce((total, item, index) => total + get(item, index), 0);

  const layers = Object.freeze(
    Array.from({ length: L }, (_, index) => Object.freeze({ index, kind: (index + 1) % 4 === 0 ? "qsa" : "gdn", ple: index === ple.layer }))
  );

  // One parameter group per torchtitan parameter (or per stacked family), with
  // the optimizer chosen by the recipe's first-match patterns.
  function group(id, label, shape, params, opt, role = "dense") {
    return Object.freeze({ id, label, shape, params, opt, role });
  }

  function layerGroups(layer) {
    const groups = [];
    const add = (...args) => groups.push(group(...args));
    const gdnQK = gdn.keyHeads * gdn.headDim;
    const gdnV = gdn.valueHeads * gdn.headDim;
    if (layer.kind === "gdn") {
      add("gdn_proj", "GDN Q/K/V projection", "[10240, D]", (2 * gdnQK + gdnV) * D, "muon");
      add("gdn_gates", "GDN z gate, a, b", "[6240, D]", (gdnV + 2 * gdn.valueHeads) * D, "adamw");
      add("gdn_conv", "Three depthwise convolutions", "[2048 | 2048 | 6144, 1, 4]", (2 * gdnQK + gdnV) * gdn.conv, "adamw");
      add("gdn_decay", "A_log, dt_bias, head norm", "[48], [48], [128]", 2 * gdn.valueHeads + gdn.headDim, "adamw");
      add("gdn_out", "GDN output projection", "[D, 6144]", D * gdnV, "muon");
    } else {
      const q = qsa.queryHeads * qsa.headDim;
      const kv = qsa.kvHeads * qsa.headDim;
      add("qsa_q", "QSA query projection", "[6144, D]", q * D, "muon");
      add("qsa_gate", "QSA output gate", "[6144, D]", q * D, "adamw");
      add("qsa_kv", "QSA K and V projections", "2 x [512, D]", 2 * kv * D, "muon");
      add("qsa_out", "QSA output projection", "[D, 6144]", D * q, "muon");
      add("indexer_proj", "Indexer Q/K projection", "[640, D]", (qsa.indexerHeads + 1) * qsa.indexerHeadDim * D, "muon");
      add("qsa_norms", "Q/K and indexer norms", "[256] x 2, [128] x 2", 2 * qsa.headDim + 2 * qsa.indexerHeadDim, "adamw");
    }
    add("router", "Router", "[512, D]", E * D, "adamw");
    add("shared", "Shared expert gate/up + down", "[2, 640, D] + [D, 640]", 3 * F * D, "muon");
    add("shared_gate", "Shared expert gate", "[1, D]", D, "adamw");
    add("routed", "Routed expert bank", "[512, 2, 640, D] + [512, D, 640]", E * 3 * F * D, "muon", "routed");
    add("residual", "Two gated residual mixers", "2 x ([C] + [320, C] + [C, 320] + [4, C])", 2 * (C + 2 * C * hcRank + S * C), "adamw");
    if (layer.ple) {
      add("ple_table", "PLE n-gram table", "[320001536, 160]", ple.rows * ple.rowDim, "adam", "table");
      add("ple_proj", "PLE key and value projections", "[C, D] + [D, D]", C * D + D * D, "muon");
      add("ple_small", "PLE norms + convolution", "3 x [C] + [C, 1, 4]", 3 * C + C * ple.conv, "adamw");
    }
    return Object.freeze(groups);
  }

  const frontGroups = Object.freeze([group("embedding", "Input embedding", "[V, D]", V * D, "adamw")]);
  const backGroups = Object.freeze([
    group("final_mixer", "Final four-stream mixer", "[C] + [320, C] + [C, 320]", C + 2 * C * hcRank, "adamw"),
    group("lm_head", "LM head", "[V, D]", V * D, "adamw"),
  ]);
  const allGroups = [...frontGroups, ...layers.flatMap(layerGroups), ...backGroups];
  const totals = Object.freeze({
    params: sum(allGroups, (g) => g.params),
    muon: sum(allGroups, (g) => (g.opt === "muon" ? g.params : 0)),
    adamw: sum(allGroups, (g) => (g.opt === "adamw" ? g.params : 0)),
    adam: sum(allGroups, (g) => (g.opt === "adam" ? g.params : 0)),
    routed: sum(allGroups, (g) => (g.role === "routed" ? g.params : 0)),
    table: sum(allGroups, (g) => (g.role === "table" ? g.params : 0)),
  });

  // Equal decoder-layer split; chunk i runs on physical rank i % PP (loop order).
  function partition(pp, vp = 1) {
    const count = pp * vp;
    if (!Number.isInteger(count) || count < 1 || L % count) throw new RangeError("PP x VPP must divide the 48 decoder layers");
    const size = L / count;
    return Array.from({ length: count }, (_, index) => {
      const owned = layers.slice(index * size, (index + 1) * size);
      const groups = [...(index === 0 ? frontGroups : []), ...owned.flatMap(layerGroups), ...(index === count - 1 ? backGroups : [])];
      return Object.freeze({
        index,
        rank: index % pp,
        local: Math.floor(index / pp),
        layers: owned,
        groups,
        first: index === 0,
        last: index === count - 1,
        dense: sum(groups, (g) => (g.role === "dense" ? g.params : 0)),
        routed: sum(groups, (g) => (g.role === "routed" ? g.params : 0)),
        table: sum(groups, (g) => (g.role === "table" ? g.params : 0)),
        muon: sum(groups, (g) => (g.opt === "muon" ? g.params : 0)),
        adamw: sum(groups, (g) => (g.opt === "adamw" ? g.params : 0)),
        adam: sum(groups, (g) => (g.opt === "adam" ? g.params : 0)),
        denseMuon: sum(groups, (g) => (g.role === "dense" && g.opt === "muon" ? g.params : 0)),
        moeLayers: owned.length,
        qsaLayers: owned.filter((layer) => layer.kind === "qsa").length,
        gdnLayers: owned.filter((layer) => layer.kind === "gdn").length,
      });
    });
  }

  return Object.freeze({
    D,
    S,
    C,
    V,
    E,
    topK,
    F,
    L,
    hcRank,
    gdn,
    qsa,
    ple,
    pleRequestCapacity,
    layers,
    layerGroups,
    frontGroups,
    backGroups,
    totals,
    partition,
    sum,
  });
});
