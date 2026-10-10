((root) => {
  const D = 7168;
  const H = 96;
  const K = 128;
  const C = H * K;
  const Z = 3584;
  const F = 3072;
  const experts = 896;
  const topK = 16;
  const embedding = D * 163840;
  const vision = 447358976;
  const sum = (items, get) => items.reduce((total, item) => total + get(item), 0);
  const layers = Array.from({ length: 93 }, (_, index) => {
    const mla = (index + 1) % 4 === 0 || index === 92;
    const attention = mla ? 232196096 : 443740384;
    const ffn = index === 0 ? 726663168 : 132120576 + 51383808 + 6422528;
    const norms = 2 * D + (index === 0 ? 2 : 4) * D;
    return {
      index,
      mla,
      dense: attention + ffn + norms,
      routed: index === 0 ? 0 : experts * 3 * Z * F,
      adam: (mla ? 2048 : 848096) + (index === 0 ? 0 : Z) + norms,
    };
  });

  function partition(pp, stage) {
    const chunk = Math.ceil(layers.length / pp);
    const owned = layers.slice(stage * chunk, Math.min((stage + 1) * chunk, layers.length));
    const front = stage === 0 ? embedding + vision : 0;
    const back = stage === pp - 1 ? embedding + 3 * D : 0;
    return {
      layers: owned,
      dense: sum(owned, (layer) => layer.dense) + front + back,
      routed: sum(owned, (layer) => layer.routed),
      adam: sum(owned, (layer) => layer.adam) + front + back,
      front,
      back,
    };
  }

  function gemmsForLayer(layer, tokens, ep) {
    const gemm = (name, n, k, options = {}) => ({ name, m: tokens, n, k, groups: 1, ...options });
    const attention = layer.mla
      ? [
          gemm("MLA Q compression", 1536, D),
          gemm("MLA Q expansion", H * 192, 1536),
          gemm("MLA KV compression", 576, D),
          gemm("MLA KV expansion", H * 256, 512),
          gemm("MLA output gate", C, D),
          gemm("MLA output projection", D, C),
        ]
      : [
          ...["Q", "K", "V"].map((name) => gemm(`KDA ${name} projection`, C, D)),
          gemm("KDA decay compression", K, D),
          gemm("KDA decay expansion", C, K),
          gemm("KDA beta projection", H, D),
          gemm("KDA output gate", C, D),
          gemm("KDA output projection", D, C),
        ];
    if (layer.index === 0) return [...attention, gemm("Dense gate/up", 2 * 33792, D), gemm("Dense down", D, 33792)];
    const routed = { m: (ep * tokens * topK) / experts, groups: experts / ep, routed: true };
    return [
      ...attention,
      gemm("Shared gate/up", 4 * F, D),
      gemm("Shared down", D, 2 * F),
      gemm("Latent compression", Z, D),
      gemm("Latent expansion", D, Z),
      gemm("Router", experts, D),
      gemm("Routed gate/up", 2 * F, Z, routed),
      gemm("Routed down", Z, F, routed),
    ];
  }

  const totalParams = sum(layers, (layer) => layer.dense + layer.routed) + 2 * embedding + vision + 3 * D;
  const adamParams = sum(layers, (layer) => layer.adam) + 2 * embedding + vision + 3 * D;
  const activeParams = sum(layers, (layer) => layer.dense + (layer.routed * topK) / experts) + embedding + 3 * D;
  const linearParams = sum(layers, (layer) => sum(gemmsForLayer(layer, 1, 1), (gemm) => gemm.m * gemm.n * gemm.k * gemm.groups)) + embedding;
  const model = {
    D,
    H,
    K,
    C,
    Z,
    F,
    experts,
    topK,
    embedding,
    vision,
    layers,
    partition,
    gemmsForLayer,
    totalParams,
    activeParams,
    linearParams,
    adamParams,
    muonParams: totalParams - adamParams,
  };
  if (typeof module === "object" && module.exports) module.exports = model;
  else root.KimiK3Model = model;
})(typeof window === "undefined" ? globalThis : window);
