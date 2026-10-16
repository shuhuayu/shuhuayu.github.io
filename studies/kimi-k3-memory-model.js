((root) => {
  const A = typeof module === "object" && module.exports ? require("./kimi-k3-accounting.js") : root.KimiK3Model;
  const GiB = 2 ** 30;
  const capacity = 184.31 * GiB;
  const sum = (items, get) => items.reduce((total, item, index) => total + get(item, index), 0);
  const elements = (shape) => shape.reduce((total, size) => total * size, 1);
  const dtypeBytes = { BF16: 2, FP32: 4, INT64: 8, BOOL: 1 };
  const defaults = Object.freeze({
    world: 512,
    pp: 8,
    cp: 1,
    ep: 64,
    fsdp: "on",
    stage: 3,
    seq: 4096,
    rows: 1,
    mbs: 64,
    vp: 1,
    sched: "1f1b",
    ac: "full",
    unit: "chunk",
    active: 1,
    grad: "deferred",
    phase: "backward",
    reserve: 24,
    vision: "on",
    residual: "stack",
    interval: A.attnResBlockSize,
    imbalance: 1,
  });

  function normalize(input = {}) {
    const c = { ...defaults, ...input };
    for (const key of ["world", "pp", "cp", "ep", "seq", "rows", "mbs", "vp", "active", "interval"])
      if (!Number.isSafeInteger(c[key]) || c[key] < 1) throw new RangeError(`${key} must be a positive integer`);
    if (!Number.isInteger(c.stage) || c.stage < 0 || c.stage >= c.pp) throw new RangeError("Pipeline rank must be in [0, PP)");
    if (c.world % (c.pp * c.cp) !== 0) throw new RangeError("GPU count must be divisible by PP x CP");
    if ((c.seq * c.rows) % c.cp !== 0) throw new RangeError("Packed token count must be divisible by CP");
    if (c.world / c.pp < c.ep || (c.world / c.pp) % c.ep || A.experts % c.ep)
      throw new RangeError("EP must divide both 896 experts and the ranks per pipeline stage");
    if (c.pp * c.vp > A.layers.length) throw new RangeError("There must be at least one decoder layer per virtual chunk on average");
    if (c.active > c.vp) throw new RangeError("Active gathered chunks cannot exceed virtual chunks per rank");
    for (const [key, choices] of Object.entries({
      fsdp: ["on", "off"],
      sched: ["1f1b", "interleaved"],
      ac: ["full", "none", "custom"],
      unit: ["stage", "chunk", "layer"],
      grad: ["deferred", "sharded"],
      phase: ["forward", "backward", "optimizer"],
      vision: ["on", "off"],
      residual: ["stack", "list"],
    }))
      if (!choices.includes(c[key])) throw new RangeError(`Unsupported ${key}: ${c[key]}`);
    if (c.sched === "1f1b" && c.vp !== 1) throw new RangeError("Ordinary 1F1B has one local stage; choose interleaved for VP > 1");
    const rounds = Math.max(1, Math.floor(c.mbs / c.pp));
    if (c.sched === "interleaved" && c.mbs % rounds !== 0)
      throw new RangeError("Interleaved microbatches must be divisible by max(1, floor(microbatches / PP))");
    if (!Number.isFinite(c.reserve) || c.reserve < 0 || !Number.isFinite(c.imbalance) || c.imbalance < 1)
      throw new RangeError("Workspace allowance must be nonnegative; received-token multiplier must be at least one");
    return c;
  }

  function makeChunk(layers, index, count, c) {
    const front = index === 0 ? A.embedding + (c.vision === "on" ? A.vision : 0) : 0;
    const back = index === count - 1 ? A.embedding + 3 * A.D : 0;
    return {
      index,
      rank: index % c.pp,
      local: Math.floor(index / c.pp),
      layers,
      front,
      back,
      dense: sum(layers, (layer) => layer.dense) + front + back,
      routed: sum(layers, (layer) => layer.routed),
      adam: sum(layers, (layer) => layer.adam) + front + back,
    };
  }

  function partitionChunks(c) {
    const count = c.pp * c.vp;
    let start = 0;
    return Array.from({ length: count }, (_, index) => {
      let size;
      if (c.vp === 1) size = Math.min(Math.ceil(A.layers.length / count), A.layers.length - start);
      else {
        // PyTorch's automatic split counts embedding and head as one layer each.
        const effective = A.layers.length + 2;
        size = Math.floor(effective / count) + Number(index < effective % count) - Number(index === 0) - Number(index === count - 1);
      }
      const owned = A.layers.slice(start, start + size);
      start += size;
      return makeChunk(owned, index, count, c);
    });
  }

  function parameterGroups(layer, ep = 64) {
    const { D, H, K, C, Z, F, experts } = A;
    const groups = [];
    const add = (id, label, shape, params, adam = 0, routed = false) =>
      groups.push({ id, label, shape, params, adam, routed, epLocal: routed ? params / ep : params });
    if (layer.mla) {
      add("qa", "MLA query compression", "[1536, D]", 1536 * D);
      add("qnorm", "Query RMSNorm", "[1536]", 1536, 1536);
      add("qb", "MLA query expansion", "[H x 192, 1536]", H * 192 * 1536);
      add("kva", "MLA KV compression", "[576, D]", 576 * D);
      add("kvnorm", "KV RMSNorm", "[512]", 512, 512);
      add("kvb", "MLA KV expansion", "[H x 256, 512]", H * 256 * 512);
      add("output_gate", "Attention output gate", "[C, D]", C * D);
      add("output_proj", "Attention output projection", "[D, C]", D * C);
    } else {
      for (const id of ["q", "k", "v"]) add(id, `KDA ${id.toUpperCase()} projection`, "[C, D]", C * D);
      add("conv", "Three depthwise convolutions", "3 x [C, 1, 4]", 3 * C * 4, 3 * C * 4);
      add("forget_a", "Forget compression", "[K, D]", K * D);
      add("forget_b", "Forget expansion", "[C, K]", C * K);
      add("decay", "Decay parameters", "A_log [H]; dt_bias [H, K]", H + H * K, H + H * K);
      add("beta", "Write-strength projection", "[H, D]", H * D, H * D);
      add("output_gate", "Attention output gate", "[C, D]", C * D);
      add("output_norm", "Shared per-head RMSNorm", "[K]", K, K);
      add("output_proj", "Attention output projection", "[D, C]", D * C);
    }
    if (layer.index === 0) {
      add("dense_gate_up", "Dense gate + up", "2 x [33792, D]", 2 * 33792 * D);
      add("dense_down", "Dense down", "[D, 33792]", D * 33792);
    } else {
      add("router", "Router (full stream, not latent)", "[896, D]", experts * D);
      add("latent_down", "Latent compression", "[Z, D]", Z * D);
      add("latent_up", "Latent expansion", "[D, Z]", D * Z);
      add("latent_norm", "Latent RMSNorm", "[Z]", Z, Z);
      add("shared_gate_up", "Two shared experts: gate + up", "2 x [2F, D]", 4 * F * D);
      add("shared_down", "Shared down", "[D, 2F]", D * 2 * F);
      add("routed_gate_up", "Routed fused gate/up storage", `[${experts / ep}, 2F, Z] on this EP rank`, experts * 2 * F * Z, 0, true);
      add("routed_down", "Routed down storage", `[${experts / ep}, Z, F] on this EP rank`, experts * Z * F, 0, true);
    }
    const norms = 2 * D + (layer.index === 0 ? 2 : 4) * D;
    add("stream_norms", "Stream norms + AttnRes projections/norms", `${layer.index === 0 ? 4 : 6} x [D]`, norms, norms);
    return groups;
  }

  function tensorCatalog(layer, c) {
    const { D, H, K, C, Z, F, experts, topK } = A;
    const T = (c.rows * c.seq) / c.cp;
    const R = Math.ceil(T * topK * c.imbalance);
    const tensors = [];
    const add = (id, label, shape, logical, options = {}) => {
      const row = { id, label, shape, logical, dtype: "BF16", save: true, group: "input", note: "", ...options };
      row.logicalBytes = elements(shape) * dtypeBytes[row.dtype];
      row.bytes = row.alias ? 0 : row.logicalBytes;
      tensors.push(row);
    };
    const alias = (id, label, shape, logical, owner, group, note) => add(id, label, shape, logical, { alias: owner, group, note, save: false });
    add("stream", "Layer boundary input", [T, D], "[T, D]", { note: "One checkpoint boundary candidate; references do not create a new copy." });
    add("attn_norm", "Normalized attention input", [T, D], "[T, D]", { group: "norm", note: "RMSNorm output is a projection input." });
    if (layer.mla) {
      add("mla_qa", "Query latent", [T, 1536], "[T, 1536]", { group: "qa" });
      add("mla_qnorm", "Normalized query latent", [T, 1536], "[T, 1536]", { group: "qa" });
      add("mla_q", "Expanded Q", [T, H, 192], "[T, H, 192]", { group: "qb" });
      add("mla_kva", "KV latent + shared key slice", [T, 576], "[T, 576]", { group: "kva" });
      alias(
        "mla_klatent",
        "512-channel latent view",
        [T, 512],
        "[T, 512]",
        "mla_kva",
        "kva",
        "A slice of the 576-channel projection, not another allocation."
      );
      alias("mla_shared", "Shared 64-channel key view", [T, 64], "[T, 64]", "mla_kva", "kva", "The headless key slice bypasses KV expansion.");
      add("mla_kvnorm", "Normalized KV latent", [T, 512], "[T, 512]", { group: "kva" });
      add("mla_kvb", "Expanded K-nope + V", [T, H, 256], "[T, H, 256]", { group: "kvb" });
      alias("mla_v", "V view", [T, H, 128], "[T, H, 128]", "mla_kvb", "kvb", "Retaining V can retain its entire K/V backing allocation.");
      alias(
        "mla_expand",
        "Shared key expanded over heads",
        [T, H, 64],
        "[T, H, 64]",
        "mla_kva",
        "kvb",
        "expand() is a zero-stride view; it has no independent storage."
      );
      add("mla_k", "Concatenated full K", [T, H, 192], "[T, H, 192]", { group: "kvb", note: "Unlike expand(), torch.cat creates a new buffer." });
      add("mla_o", "Varlen attention output", [T, H, K], "[T, H, K]", { group: "core" });
      add("mla_lse", "Attention log-sum-exp", [T, H], "[T, H]", {
        group: "core",
        dtype: "FP32",
        note: "Backend-dependent saved statistics; no [T, T] probability tensor.",
      });
      add("mla_rstd", "Query + KV RMS statistics", [2, T], "2 x [T]", { group: "qa", dtype: "FP32" });
    } else {
      for (const id of ["q", "k", "v"])
        add(`kda_${id}`, `${id.toUpperCase()} projection output`, [T, C], "[T, C]", {
          group: "qkv",
          save: false,
          note: "Forward allocation; concatenation backward need not save this projection output.",
        });
      add("kda_cat", "Concatenated short-conv input", [T, 3 * C], "[T, 3C]", {
        group: "conv",
        note: "torch.cat makes storage in addition to the three projection outputs while they overlap.",
      });
      add("kda_conv", "Convolved Q/K/V + SiLU", [T, 3 * C], "[T, 3C]", {
        group: "conv",
        note: "Fused-convolution save/recompute details are backend-specific.",
      });
      for (const id of ["q", "k", "v"])
        alias(
          `kda_${id}_view`,
          `${id.toUpperCase()} head view`,
          [T, H, K],
          "[T, H, K]",
          "kda_conv",
          "qknorm",
          "chunk() and head unflattening alias the convolved buffer."
        );
      for (const id of ["q", "k"]) add(`kda_${id}_norm`, `L2-normalized ${id.toUpperCase()}`, [T, H, K], "[T, H, K]", { group: "qknorm" });
      add("kda_fa", "Forget latent", [T, K], "[T, K]", { group: "forget" });
      add("kda_fg", "Raw forget gate", [T, H, K], "[T, H, K]", { group: "forget" });
      add("kda_decay", "Bounded log-decay", [T, H, K], "[T, H, K]", {
        group: "decay",
        dtype: "FP32",
        note: "Gate preparation outputs FP32 even when projections use BF16.",
      });
      add("kda_beta_raw", "Raw write strength", [T, H], "[T, H]", { group: "beta", save: false });
      add("kda_beta", "Sigmoid write strength", [T, H], "[T, H]", { group: "beta", dtype: "FP32" });
      add("kda_o", "Chunk KDA output", [T, H, K], "[T, H, K]", {
        group: "core",
        note: "Token output only; chunk-state workspace is analyzed separately below.",
      });
      add("kda_rstd", "Per-head output RMS statistics", [T, H], "[T, H]", { group: "gated", dtype: "FP32" });
    }
    add("output_gate", "Attention output-gate logits", [T, H, K], "[T, H, K]", { group: "gate" });
    add("gated_o", "Normalized/gated projection input", [T, C], "[T, C]", { group: "gated" });
    add("attn_out", "Attention projection output", [T, D], "[T, D]", { group: "output", save: false });
    add("ffn_stream", "Post-attention residual stream", [T, D], "[T, D]", { group: "ffn_norm" });
    add("ffn_norm", "Normalized FFN input", [T, D], "[T, D]", { group: "ffn_norm" });
    if (layer.index === 0) {
      add("dense_gate", "Dense gate projection", [T, 33792], "[T, 33792]", { group: "dense" });
      add("dense_up", "Dense up projection", [T, 33792], "[T, 33792]", {
        group: "dense",
        note: "Separate linears; equal shape does not imply aliasing.",
      });
      add("dense_product", "Dense SiTU product", [T, 33792], "[T, 33792]", { group: "dense_act" });
    } else {
      add("router_logits", "Router logits", [T, experts], "[T, E]", {
        group: "router",
        dtype: "FP32",
        note: "Router operates on D=7168, not the compressed expert width.",
      });
      add("router_scores", "Router sigmoid scores", [T, experts], "[T, E]", { group: "router", dtype: "FP32" });
      add("route_weights", "Selected routing weights", [T, topK], "[T, topK]", { group: "router", dtype: "FP32" });
      add("route_ids", "Selected expert indices", [T, topK], "[T, topK]", { group: "router", dtype: "INT64" });
      add("route_map", "Dense dispatch-map candidate", [T, experts], "[T, E]", {
        group: "router",
        dtype: "BOOL",
        save: false,
        note: "Optional backend metadata; not assumed saved by every dispatcher.",
      });
      add("latent", "Compressed expert input", [T, Z], "[T, Z]", { group: "latent" });
      add("dispatched", "Received routed assignments", [R, Z], "[R, Z]", {
        group: "dispatch",
        note: "Balanced R = T x 16, including assignments from other EP ranks. It is NOT T x 16 / EP.",
      });
      add("expert_gate_up", "Routed gate/up outputs", [R, 2 * F], "[R, 2F]", { group: "experts" });
      alias("expert_gate", "Gate half", [R, F], "[R, F]", "expert_gate_up", "experts", "A view of fused gate/up output; counted through the owner.");
      alias("expert_up", "Up half", [R, F], "[R, F]", "expert_gate_up", "experts", "A view of fused gate/up output; counted through the owner.");
      add("expert_product", "Routed SiTU product", [R, F], "[R, F]", { group: "expert_act" });
      add("expert_down", "Routed down outputs", [R, Z], "[R, Z]", { group: "expert_down" });
      add("combined", "Combined latent output", [T, Z], "[T, Z]", { group: "combine" });
      add("combined_norm", "Normalized combined latent", [T, Z], "[T, Z]", { group: "combine" });
      add("shared_gate_up", "Two shared experts: gate/up", [T, 4 * F], "[T, 4F]", { group: "shared" });
      add("shared_product", "Shared SiTU product", [T, 2 * F], "[T, 2F]", { group: "shared_act" });
      add("shared_out", "Shared output", [T, D], "[T, D]", { group: "shared_out", save: false });
      add("expanded", "Expanded latent output", [T, D], "[T, D]", { group: "expand", save: false });
    }
    add("layer_out", "Layer output", [T, D], "[T, D]", {
      group: "finish",
      save: false,
      note: "The next layer's boundary input may be this tensor; do not count both as separate storage.",
    });
    return { tensors, T, R, localExperts: experts / c.ep, rowsPerExpert: R / (experts / c.ep) };
  }

  function isSaved(row, c, marks = {}) {
    if (row.alias) return false;
    if (c.ac === "full") return row.id === "stream";
    if (c.ac === "custom" && Object.prototype.hasOwnProperty.call(marks, row.id)) return marks[row.id] === true;
    return row.save;
  }

  function savedForLayer(layer, c, marks = {}) {
    return sum(tensorCatalog(layer, c).tensors, (row) => (isSaved(row, c, marks) ? row.bytes : 0));
  }

  function chunkActivations(chunk, c, marks = {}) {
    const T = (c.rows * c.seq) / c.cp;
    const boundaries = new Set(chunk.layers.filter((layer) => isSaved(tensorCatalog(layer, c).tensors[0], c, marks)).map((layer) => layer.index));
    const last = chunk.layers.at(-1)?.index;
    const incomingBanks =
      last === undefined
        ? []
        : Array.from({ length: Math.floor(last / c.interval) + 1 }, (_, i) => i * c.interval).filter((birth) => !boundaries.has(birth));
    const layerBytes = sum(chunk.layers, (layer) => savedForLayer(layer, c, marks));
    // A residual bank born at an already-counted local boundary is a reference.
    const bankBytes = incomingBanks.length * T * A.D * 2;
    return { layerBytes, bankBytes, incomingBanks, bytes: layerBytes + bankBytes };
  }

  function schedule(c, rank, weights = Array(c.vp).fill(1)) {
    const actions = [];
    const forwardIds = Array(c.vp).fill(0);
    const backwardIds = Array(c.vp).fill(0);
    const append = (kind, chunk) => {
      const ids = kind === "F" ? forwardIds : backwardIds;
      actions.push({ kind, chunk, mb: ids[chunk]++ });
    };
    if (c.sched === "1f1b") {
      const warmup = Math.min(c.mbs, c.pp - rank);
      for (let i = 0; i < warmup; i++) append("F", 0);
      for (let i = warmup; i < c.mbs; i++) {
        append("B", 0);
        append("F", 0);
      }
      for (let i = 0; i < warmup; i++) append("B", 0);
    } else {
      // This is the logical F/B order of PyTorch ScheduleInterleaved1F1B.
      const rounds = Math.max(1, Math.floor(c.mbs / c.pp));
      const perRound = c.mbs / rounds;
      const total = c.mbs * c.vp;
      const warmup = Math.min(total, (c.vp - 1) * perRound + 2 * (c.pp - 1 - rank));
      for (let step = 0; step < total + warmup; step++) {
        if (step < total) append("F", Math.floor(step / perRound) % c.vp);
        if (step >= warmup) append("B", c.vp - 1 - (Math.floor((step - warmup) / perRound) % c.vp));
      }
    }
    const pending = Array.from({ length: c.vp }, () => new Map());
    const counts = Array(c.vp).fill(0);
    const lifetimes = [];
    const snapshots = [{ counts: [...counts], bytes: 0 }];
    let peak = 0;
    let peakStep = 0;
    let peakCount = 0;
    actions.forEach((action, index) => {
      const { kind, chunk, mb } = action;
      if (kind === "F") {
        pending[chunk].set(mb, index);
        counts[chunk]++;
      } else {
        if (!pending[chunk].has(mb)) throw new Error("Backward has no matching forward");
        lifetimes.push({ chunk, mb, start: pending[chunk].get(mb, index), end: index + 1 });
        pending[chunk].delete(mb);
        counts[chunk]--;
      }
      const bytes = sum(counts, (count, chunkIndex) => count * weights[chunkIndex]);
      snapshots.push({ counts: [...counts], bytes });
      if (bytes > peak) {
        peak = bytes;
        peakStep = index + 1;
      }
      peakCount = Math.max(
        peakCount,
        sum(counts, (count) => count)
      );
    });
    return { actions, snapshots, lifetimes, peak, peakStep, peakCount };
  }

  function parameterCopies(group, c) {
    const stageRanks = c.world / c.pp;
    const degree = c.fsdp === "off" ? 1 : group.routed ? stageRanks / c.ep : stageRanks;
    const localAdam = group.routed ? group.adam / c.ep : group.adam;
    const stored = group.epLocal / degree;
    const storedAdam = localAdam / degree;
    return {
      stored,
      degree,
      master: stored * 4,
      optimizer: (stored - storedAdam) * 4 + storedAdam * 8,
      gradient: c.phase === "forward" ? 0 : (c.grad === "deferred" && c.phase === "backward" ? group.epLocal : stored) * 4,
      compute: group.epLocal * 2,
    };
  }

  function calculate(input = {}, marks = {}) {
    const c = normalize(input);
    const chunks = partitionChunks(c);
    const stageRanks = c.world / c.pp;
    const dp = stageRanks / c.cp;
    const denseFsdp = c.fsdp === "on" ? stageRanks : 1;
    const expertFsdp = c.fsdp === "on" ? stageRanks / c.ep : 1;
    const owned = chunks.filter((chunk) => chunk.rank === c.stage);
    const layers = owned.flatMap((chunk) => chunk.layers);
    const dense = sum(owned, (chunk) => chunk.dense);
    const routed = sum(owned, (chunk) => chunk.routed);
    const adam = sum(owned, (chunk) => chunk.adam) / denseFsdp;
    const stored = dense / denseFsdp + routed / (c.ep * expertFsdp);
    const master = stored * 4;
    const momentum = (stored - adam) * 4;
    const adamState = adam * 8;
    const reducedGrad = stored * 4;
    const fullGrad = (dense + routed / c.ep) * 4;
    const gradient = c.phase === "forward" ? 0 : c.phase === "backward" && c.grad === "deferred" ? fullGrad : reducedGrad;
    const gatherSizes = owned.map((chunk) => 2 * (chunk.dense + chunk.routed / c.ep)).sort((a, b) => b - a);
    const layerViews = layers.map((layer) => 2 * (layer.dense + layer.routed / c.ep));
    for (const chunk of owned) layerViews.push(2 * chunk.front, 2 * chunk.back);
    const compute =
      c.phase === "optimizer"
        ? 0
        : c.unit === "stage"
          ? sum(gatherSizes, (size) => size)
          : c.unit === "chunk"
            ? sum(gatherSizes.slice(0, c.active), (size) => size)
            : Math.max(...layerViews);
    const stages = Array.from({ length: c.pp }, (_, rank) => {
      const rankChunks = chunks.filter((chunk) => chunk.rank === rank);
      const saves = rankChunks.map((chunk) => chunkActivations(chunk, c, marks));
      const timeline = schedule(
        c,
        rank,
        saves.map((save) => save.bytes)
      );
      return { rank, chunks: rankChunks, saves, timeline };
    });
    const stage = stages[c.stage];
    const activation = c.phase === "optimizer" ? 0 : stage.timeline.peak;
    const activeCandidates =
      c.phase === "optimizer" || c.ac !== "full"
        ? 0
        : Math.max(0, ...layers.map((layer) => sum(tensorCatalog(layer, c).tensors, (row) => (row.save && row.id !== "stream" ? row.bytes : 0))));
    const bankSlots = Math.max(0, ...layers.map((layer) => Math.floor(layer.index / c.interval) + 2));
    const residualStack = c.phase === "optimizer" || c.residual === "list" ? 0 : ((c.rows * c.seq) / c.cp) * bankSlots * A.D * 2;
    const reserve = c.reserve * GiB;
    const parts = [
      { key: "master", label: "FP32 resident / master", value: master, detail: "The FSDP storage shard is the master; no second master copy." },
      { key: "momentum", label: "FP32 Muon momentum", value: momentum, detail: "One persistent tensor per Muon parameter shard." },
      { key: "adam", label: "FP32 AdamW moments", value: adamState, detail: "Two moments for embeddings, norms, and the declared AdamW remainder." },
      {
        key: "gradient",
        label: c.phase === "backward" && c.grad === "deferred" ? "Deferred FP32 gradients" : "FP32 gradient shards",
        value: gradient,
        detail:
          c.phase === "forward"
            ? "Forward after set_to_none: gradients cleared, initialized optimizer state retained."
            : "One accumulated gradient history, not one copy per microbatch. Reduction overlap is extra.",
      },
      {
        key: "compute",
        label: "BF16 gathered views",
        value: compute,
        detail:
          c.unit === "stage"
            ? "All virtual chunks owned by this physical rank."
            : c.unit === "chunk"
              ? `The ${c.active} largest local chunk(s); residency assumption, not concurrent compute.`
              : "Largest single layer or front/back module; an optimistic residency alternative.",
      },
      {
        key: "activation",
        label: "Saved tensors + residual banks",
        value: activation,
        detail: `Weighted logical schedule peak. ${
          c.ac === "full" ? "Checkpoint boundary floor" : "Selected named candidates"
        }; aliases within a chunk are counted once.`,
      },
      {
        key: "working",
        label: "Active recompute candidates",
        value: activeCandidates,
        detail: "Full checkpoint mode regenerates one layer's internal save candidates. Their sum is an allowance, not a measured simultaneous peak.",
      },
      {
        key: "residual",
        label: "One AttnRes stack",
        value: residualStack,
        detail: "One materialized [T, N, D] stack beyond the residual references; FP32 eager expansion is not included.",
      },
      {
        key: "reserve",
        label: "Other workspace allowance",
        value: reserve,
        detail: "Chosen budget for KDA chunks, dispatch, FSDP, Muon, NCCL, graph pools, allocator effects, and uncatalogued tensors.",
      },
    ];
    return {
      config: c,
      stageRanks,
      dp,
      denseFsdp,
      expertFsdp,
      chunks,
      owned,
      layers,
      stage,
      stages,
      stored,
      fullGrad,
      reducedGrad,
      stateTotal: master + momentum + adamState + reducedGrad,
      budget: sum(parts, (part) => part.value),
      parts,
      tokens: (c.rows * c.seq) / c.cp,
      globalRows: dp * c.mbs * c.rows,
      globalTokens: dp * c.mbs * c.rows * c.seq,
    };
  }

  function workspaces(layer, c, matrixBatch = 14) {
    const T = (c.rows * c.seq) / c.cp;
    const R = Math.ceil(T * A.topK * c.imbalance);
    const banks = Math.floor(layer.index / c.interval) + 2;
    const chunks = Math.ceil(T / 64);
    const matrix = 2 * A.F * A.Z;
    const gram = 2 * A.F * A.F;
    return {
      T,
      R,
      banks,
      residualStack: T * banks * A.D * 2,
      residualFP32: T * banks * A.D * 4,
      inferenceState: 4 * A.H * A.K * A.K,
      kdaChunkState: 4 * chunks * A.H * A.K * A.K,
      chunks,
      situCast: R * A.F * 4,
      situCasts: R * A.F * 4 * 2,
      matrix,
      gram,
      matrixBatch,
      nsCandidates: matrixBatch * (2 * matrix + 2 * gram),
      logits: T * 163840 * 2,
    };
  }

  const api = {
    defaults,
    capacity,
    GiB,
    normalize,
    partitionChunks,
    parameterGroups,
    tensorCatalog,
    isSaved,
    savedForLayer,
    chunkActivations,
    schedule,
    parameterCopies,
    calculate,
    workspaces,
  };
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.KimiK3Memory = api;
})(typeof window === "undefined" ? globalThis : window);
