((root, factory) => {
  const A = typeof module === "object" && module.exports ? require("./qwen3-8-flash-next-accounting.js") : root.QwenFlashNextAccounting;
  const api = factory(A);
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.QwenFlashNextMemory = api;
})(typeof globalThis !== "undefined" ? globalThis : this, (A) => {
  "use strict";

  const GiB = 2 ** 30;
  const MiB = 2 ** 20;
  const { sum } = A;
  // GB300: eight 36-GiB HBM3e stacks. The runtime and driver keep part of it.
  const capacity = 288 * GiB;
  const align = (bytes) => Math.ceil(bytes / 128) * 128;

  const defaults = Object.freeze({
    world: 16,
    pp: 2,
    vp: 2,
    ep: 8,
    stage: 0,
    tokens: 16384,
    seq: 4096,
    mbs: 16,
    slot: 0.5,
    scratch: 4,
    phase: "backward",
    views: "stage",
    grad: "deferred",
    table: "stage",
    reserve: 8,
  });

  function normalize(input = {}) {
    const c = { ...defaults, ...input };
    for (const key of ["world", "pp", "vp", "ep", "tokens", "seq", "mbs"])
      if (!Number.isSafeInteger(c[key]) || c[key] < 1) throw new RangeError(`${key} must be a positive integer`);
    if (!Number.isInteger(c.stage) || c.stage < 0 || c.stage >= c.pp) throw new RangeError("Pipeline rank must be in [0, PP)");
    if (c.world % c.pp) throw new RangeError("GPU count must be divisible by PP");
    const dp = c.world / c.pp;
    if (dp % c.ep || A.E % c.ep) throw new RangeError("EP must divide both 512 experts and the DP ranks of one stage");
    if (A.L % (c.pp * c.vp)) throw new RangeError("PP x VPP must divide the 48 decoder layers");
    if (c.vp > 1 && (c.pp === 1 || c.mbs % c.pp)) throw new RangeError("Interleaving needs PP > 1 and microbatches divisible by PP");
    if (c.pp === 1 && c.vp !== 1) throw new RangeError("VPP needs PP > 1");
    if (c.tokens % c.seq) throw new RangeError("Tokens per microbatch must be whole documents of the context length");
    if (c.mbs < c.pp) throw new RangeError("Microbatches must be at least PP");
    for (const key of ["slot", "scratch", "reserve"]) if (!Number.isFinite(c[key]) || c[key] < 0) throw new RangeError(`${key} must be nonnegative`);
    if (c.scratch <= 0) throw new RangeError("Scratch capacity factor must be positive");
    if (!["forward", "backward", "optimizer"].includes(c.phase)) throw new RangeError("Unknown lifecycle phase");
    if (!["stage", "chunk", "layer"].includes(c.views)) throw new RangeError("Unknown BF16 view residency");
    if (!["deferred", "reduced"].includes(c.grad)) throw new RangeError("Unknown gradient lifecycle");
    if (!["stage", "world"].includes(c.table)) throw new RangeError("Unknown PLE table placement");
    return c;
  }

  // Logical F/B order of PyTorch's Schedule1F1B / ScheduleInterleaved1F1B.
  function schedule(c, rank) {
    const actions = [];
    const forwardIds = Array(c.vp).fill(0);
    const backwardIds = Array(c.vp).fill(0);
    const append = (kind, chunk) => {
      const ids = kind === "F" ? forwardIds : backwardIds;
      actions.push({ kind, chunk, mb: ids[chunk]++ });
    };
    if (c.vp === 1) {
      const warmup = Math.min(c.mbs, c.pp - rank);
      for (let i = 0; i < warmup; i++) append("F", 0);
      for (let i = warmup; i < c.mbs; i++) {
        append("B", 0);
        append("F", 0);
      }
      for (let i = 0; i < warmup; i++) append("B", 0);
    } else {
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
    const live = [0];
    let peak = 0;
    let peakStep = 0;
    actions.forEach((action, index) => {
      const { kind, chunk, mb } = action;
      if (kind === "F") {
        pending[chunk].set(mb, index);
        counts[chunk]++;
      } else {
        if (!pending[chunk].has(mb)) throw new Error("Backward has no matching forward");
        lifetimes.push({ chunk, mb, start: pending[chunk].get(mb), end: index + 1 });
        pending[chunk].delete(mb);
        counts[chunk]--;
      }
      const now = sum(counts, (count) => count);
      live.push(now);
      if (now > peak) {
        peak = now;
        peakStep = index + 1;
      }
    });
    return { actions, lifetimes, live, peak, peakStep };
  }

  // Dist-MoE BF16 planner formulas: a slot holds every MoE layer of one stage
  // for one microbatch; its optional part scales with the capacity factor.
  function distMoe(c, layersPerSlot) {
    const T = c.tokens;
    const R = T * A.topK;
    const minimumLayer = align(T * A.D * 2);
    const fullLayer = align(R * A.D * 2) + align(R * 2 * A.F * 2) + align(R * A.D * 2);
    const slotBytes = layersPerSlot * (minimumLayer + c.slot * (fullLayer - minimumLayer));
    const rows = Math.min(Math.round(R * c.scratch), T * c.ep * Math.min(A.topK, A.E / c.ep));
    // Recompute-mode backward scratch: x_gathered, h1, grad_h1, grad_h2, grad_h3.
    const scratchRow = A.D * 2 + 2 * A.F * 2 + 2 * A.F * 2 + A.F * 2 + A.D * 2;
    return { T, R, minimumLayer, fullLayer, slotBytes, rows, scratchRow, scratchBytes: rows * scratchRow };
  }

  // Tensors saved per chunk and microbatch under full layer recompute.
  function chunkSaves(chunk, c) {
    const T = c.tokens;
    const boundary = T * A.C * 2;
    const teacherBlocks = Math.min(A.qsa.budget / A.qsa.blockSize, T / A.qsa.blockSize);
    const teacher = T * teacherBlocks * 4;
    const finalHidden = chunk.last ? T * A.D * 2 : 0;
    return {
      boundary,
      boundaries: chunk.layers.length * boundary,
      teacher,
      teachers: chunk.qsaLayers * teacher,
      finalHidden,
      bytes: chunk.layers.length * boundary + chunk.qsaLayers * teacher + finalHidden,
    };
  }

  function rankState(c, chunks, rank) {
    const dp = c.world / c.pp;
    const edp = dp / c.ep;
    const owned = chunks.filter((chunk) => chunk.rank === rank);
    const dense = sum(owned, (chunk) => chunk.dense);
    const routed = sum(owned, (chunk) => chunk.routed);
    const tableParams = sum(owned, (chunk) => chunk.table);
    const tableOwners = c.table === "world" ? c.world : dp;
    // Under the "world" alternative every GPU holds an equal row range.
    const table = c.table === "world" ? A.totals.table / c.world : tableParams / tableOwners;
    const denseMuon = sum(owned, (chunk) => chunk.denseMuon);
    const routedLocal = routed / c.ep;
    const stored = { dense: dense / dp, routed: routedLocal / edp, table };
    const storedTotal = stored.dense + stored.routed + stored.table;
    const muon = denseMuon / dp + stored.routed;
    const adamLike = storedTotal - muon;
    const master = storedTotal * 4;
    const momentum = muon * 4;
    const moments = adamLike * 8;
    const reducedGrad = storedTotal * 4;
    const deferredGrad = (dense + routedLocal + table) * 4;
    const viewsStage = sum(owned, (chunk) => 2 * (chunk.dense + chunk.routed / c.ep));
    const viewsChunk = Math.max(0, ...owned.map((chunk) => 2 * (chunk.dense + chunk.routed / c.ep)));
    // Resharding after forward and backward: the running layer plus one prefetched
    // layer (or the embedding / head unit, whichever is larger).
    const unitViews = owned
      .flatMap((chunk) => [
        ...chunk.layers.map(
          (layer) => 2 * A.sum(A.layerGroups(layer), (g) => (g.role === "routed" ? g.params / c.ep : g.role === "dense" ? g.params : 0))
        ),
        ...(chunk.first ? [2 * A.sum(A.frontGroups, (g) => g.params)] : []),
        ...(chunk.last ? [2 * A.sum(A.backGroups, (g) => g.params)] : []),
      ])
      .sort((a, b) => b - a);
    const viewsLayer = (unitViews[0] || 0) + (unitViews[1] || 0);
    return {
      rank,
      owned,
      dp,
      edp,
      dense,
      routed,
      routedLocal,
      tableParams,
      stored,
      storedTotal,
      master,
      momentum,
      moments,
      reducedGrad,
      deferredGrad,
      viewsStage,
      viewsChunk,
      viewsLayer,
      stateTotal: master + momentum + moments + reducedGrad,
    };
  }

  // Transient tensors that exist while one layer runs (or the loss / optimizer).
  function workspaces(c) {
    const T = c.tokens;
    const chunks = Math.ceil(T / A.gdn.chunk);
    const J = A.qsa.budget + A.qsa.blockSize - 1;
    const Jpad = Math.ceil(J / 8) * 8;
    const teacherChunk = Math.min(512, T);
    const indexerChunk = Math.min(1024, T);
    const blocks = T / A.qsa.blockSize;
    const owners = c.table === "world" ? c.world : c.world / c.pp;
    const capacityPerOwner = A.pleRequestCapacity(T, owners);
    return {
      gdnInference: A.gdn.valueHeads * A.gdn.headDim * A.gdn.headDim * 4,
      gdnChunkStates: chunks * A.gdn.valueHeads * A.gdn.headDim * A.gdn.headDim * 4,
      gdnChunks: chunks,
      gdnProjection:
        T * (2 * A.gdn.keyHeads * A.gdn.headDim + A.gdn.valueHeads * A.gdn.headDim + A.gdn.valueHeads * A.gdn.headDim + 2 * A.gdn.valueHeads) * 2,
      selected: T * J * 8,
      J,
      teacherKeys: teacherChunk * Jpad * A.qsa.kvHeads * A.qsa.headDim * 2,
      teacherLogits: A.qsa.kvHeads * teacherChunk * (A.qsa.queryHeads / A.qsa.kvHeads) * Jpad * 4,
      indexerScores: indexerChunk * blocks * A.qsa.indexerHeads * 4,
      fullLogits: T * A.V * 2,
      fullLogitsFP32: T * A.V * 4,
      pleIds: T * A.ple.heads * 8,
      pleCapacity: capacityPerOwner,
      pleRequests: owners * capacityPerOwner * 8,
      pleRows: owners * capacityPerOwner * A.ple.rowDim * 2,
      pleOutput: T * A.ple.heads * A.ple.rowDim * 2,
      expertMatrix: A.F * A.D * 2,
      expertGram: A.F * A.F * 2,
    };
  }

  function calculate(input = {}) {
    const c = normalize(input);
    const chunks = A.partition(c.pp, c.vp);
    const dp = c.world / c.pp;
    const layersPerSlot = A.L / (c.pp * c.vp);
    const moe = distMoe(c, layersPerSlot);
    const w = workspaces(c);
    const stages = Array.from({ length: c.pp }, (_, rank) => {
      const state = rankState(c, chunks, rank);
      const saves = state.owned.map((chunk) => chunkSaves(chunk, c));
      const timeline = schedule(c, rank);
      // Saved boundaries follow the weighted schedule; Dist-MoE slots are a
      // fixed allocation of `peak` live microbatch-stage slots.
      const pending = Array.from({ length: c.vp }, () => 0);
      let savedPeak = 0;
      for (const action of timeline.actions) {
        pending[action.chunk] += action.kind === "F" ? 1 : -1;
        savedPeak = Math.max(
          savedPeak,
          sum(pending, (count, index) => count * saves[index].bytes)
        );
      }
      return { ...state, saves, timeline, savedPeak, slots: timeline.peak, slotTotal: timeline.peak * moe.slotBytes };
    });
    const stage = stages[c.stage];
    const active = c.phase !== "optimizer";
    const gradient = c.phase === "forward" ? 0 : c.phase === "backward" && c.grad === "deferred" ? stage.deferredGrad : stage.reducedGrad;
    const views = !active ? 0 : { stage: stage.viewsStage, chunk: stage.viewsChunk, layer: stage.viewsLayer }[c.views];
    // One layer's recompute is live at a time; take the largest named working set.
    const layerWorking = Math.max(w.gdnChunkStates + w.gdnProjection, w.selected + w.teacherKeys + w.teacherLogits);
    const parts = [
      {
        key: "master",
        label: "FP32 resident / master",
        value: stage.master,
        detail: "Dense FSDP shards, EP-local expert shards and owned PLE rows.",
      },
      { key: "momentum", label: "FP32 Muon momentum", value: stage.momentum, detail: "One tensor per DistMuon parameter shard." },
      {
        key: "adam",
        label: "FP32 Adam / AdamW moments",
        value: stage.moments,
        detail: "Two moments for the PLE table (Adam) and the AdamW remainder.",
      },
      {
        key: "gradient",
        label: c.phase === "backward" && c.grad === "deferred" ? "Deferred FP32 gradients" : "FP32 gradient shards",
        value: gradient,
        detail:
          c.phase === "forward"
            ? "First forward after zero_grad(set_to_none=True)."
            : c.phase === "backward" && c.grad === "deferred"
              ? "PP stages skip gradient sync until the last backward: dense gradients stay unsharded. Expert and PLE gradients are already local."
              : "Reduced shards, the size of the resident parameters.",
      },
      {
        key: "compute",
        label: "BF16 compute views",
        value: views,
        detail:
          c.views === "stage"
            ? "PP keeps every local chunk unsharded from its first forward to the schedule end. The PLE table has no view."
            : c.views === "layer"
              ? "Reshard after forward and backward: the running layer plus one prefetched layer."
              : "Only the largest local chunk at a time: an optimistic residency alternative.",
      },
      {
        key: "activation",
        label: "Saved boundaries + teacher targets",
        value: active ? stage.savedPeak : 0,
        detail: "Four-stream layer inputs and QSA KL targets of every unfinished chunk forward.",
      },
      {
        key: "slot",
        label: "Dist-MoE activation slots",
        value: active ? stage.slotTotal : 0,
        detail: `${stage.slots} live microbatch-stage slot(s) of ${layersPerSlot} MoE layers, capacity factor ${c.slot}.`,
      },
      {
        key: "scratch",
        label: "Dist-MoE shared scratch",
        value: active ? moe.scratchBytes : 0,
        detail: `${moe.rows.toLocaleString("en-US")} receive rows (${c.scratch}x balanced) of backward recompute scratch.`,
      },
      {
        key: "working",
        label: "One layer's recompute workspace",
        value: active ? layerWorking : 0,
        detail: "Largest named transient: GDN chunk states or QSA indices plus one teacher chunk.",
      },
      {
        key: "reserve",
        label: "Other workspace allowance",
        value: c.reserve * GiB,
        detail: "Chosen budget for NCCL, FSDP buffers, allocator effects and uncatalogued tensors.",
      },
    ];
    return {
      config: c,
      chunks,
      dp,
      edp: dp / c.ep,
      layersPerSlot,
      moe,
      workspaces: w,
      stages,
      stage,
      parts,
      budget: sum(parts, (part) => part.value),
      globalTokens: dp * c.mbs * c.tokens,
      bubble: c.pp === 1 ? 0 : (c.pp - 1) / (c.mbs * c.vp),
    };
  }

  // The logical bill before any parallelism, by optimizer family.
  function stateBill() {
    const t = A.totals;
    const muon = t.muon * 12;
    const adamw = t.adamw * 16;
    const adam = t.adam * 16;
    return { muon, adamw, adam, total: muon + adamw + adam };
  }

  return Object.freeze({ GiB, MiB, capacity, defaults, normalize, schedule, distMoe, chunkSaves, rankState, workspaces, calculate, stateBill });
});
