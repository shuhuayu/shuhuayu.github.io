(() => {
  const M = window.DeepSeekV41;
  const S = M.spec;
  const D = S.dim;
  const K = S.headDim;
  const contexts = [128, 1024, 4096, 16384, 65536, 262144, 1048576];
  const byId = (id) => document.getElementById(id);
  const integer = (value) => value.toLocaleString("en-US");
  const count = (value) => (value >= 1e9 ? `${(value / 1e9).toFixed(2)}B` : value >= 1e6 ? `${(value / 1e6).toFixed(2)}M` : integer(value));
  const bytes = (value) =>
    value >= 2 ** 30
      ? `${(value / 2 ** 30).toFixed(2)} GiB`
      : value >= 2 ** 20
        ? `${(value / 2 ** 20).toFixed(2)} MiB`
        : value >= 1024
          ? `${(value / 1024).toFixed(2)} KiB`
          : `${integer(value)} B`;
  const escape = (value) =>
    String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
  let selectedLayer = 29;
  let selectedOperation = "global_shared";
  let operationMap = new Map();

  const node = (id, title, shape, options = {}) => ({
    id,
    title,
    shape,
    kind: "op",
    params: 0,
    weights: "No learned matrix",
    code: "Attention.forward",
    precision: "Logical BF16 activations; reductions may use FP32",
    info: "",
    backward: "Inputs and kernel-specific intermediates, or recomputation under checkpointing.",
    ...options,
  });
  const projection = (id, title, shape, input, output, options = {}) =>
    node(id, title, shape, {
      kind: "gemm",
      params: input * output,
      weights: `[${output}, ${input}]`,
      ...options,
    });

  function graph(layer) {
    const stages = [];
    const inputSource = layer.number === 1 ? "token embeddings" : `layer ${layer.number - 1}'s output`;
    const current = `Layer ${layer.number}`;
    const ratio = layer.ratio;
    if (layer.hasEngram)
      stages.push({
        title: "Engram / before the block",
        color: "gold",
        branches: [
          {
            label: "Token-addressed memory",
            nodes: [
              node("engram_lookup", "2/3/4-gram hash lookups", "[T, 24, 256]", {
                kind: "shared",
                code: "Engram.embed / NgramHashState",
                weights: `${integer(S.engramRows[S.engramLayers.indexOf(layer.index)])} rows × 256 channels`,
                params: S.engramRows[S.engramLayers.indexOf(layer.index)] * S.engramHeadDim,
                precision: "FP8 table rows and their scales",
                info: "8 hash heads for each of 3 orders fetch 24 rows. Addresses depend only on token IDs and can be prefetched before this layer.",
                backward: "Lookup IDs and gradients to selected rows; distributed training returns row gradients to the table owners.",
              }),
              projection("engram_project", "Memory key / value projection", "keys [T, 4, 5120] + value [T, 5120]", 6144, 25600, {
                code: "Engram.wkv",
                info: "Flatten the fetched vectors. Produce one key for each residual stream and one value shared by all four streams.",
                backward: "Fetched embeddings, projection weights, and gradients from the keys and value.",
              }),
              node("engram_gate", "Context gate + residual addition", "[T, 4, 5120]", {
                code: "Engram.forward",
                params: 2 * S.hcCopies * D,
                weights: "q_weight and k_weight: 2 × [4, 5120]",
                info: "Normalized hidden/key compatibility passes through a signed square root and sigmoid. Each stream gets its own gate. Image positions receive no contribution.",
                backward: "Incoming hidden states, projected keys/value, and gate intermediates.",
              }),
            ],
          },
        ],
      });

    stages.push({
      title: "Single-Pass mHC / attention input",
      color: "residual",
      branches: [
        {
          label: `Incoming state: ${inputSource}`,
          nodes: [
            node("input_mix", "Apply carried input coefficients", "[T, 5120]", {
              code: "Block.forward / hc_pre",
              weights: layer.index === 0 ? "Initial uniform 1/4 coefficients" : "Input mix belongs to the previous sublayer",
              info:
                layer.index === 0
                  ? "The first block starts with a uniform average of the four embedding streams."
                  : "Collapse four residual streams using the previous sublayer's input-mixing coefficients. These are already available when this block starts.",
              backward:
                layer.index === 0
                  ? "Input embedding streams; the initial coefficients are constants."
                  : "The four input streams and incoming coefficients; gradients also flow into the preceding coefficient-producing sublayer.",
            }),
          ],
        },
        {
          label: "Predict coefficients in parallel",
          nodes: [
            projection("mhc_coeff", "mHC coefficient projection", "[T, 24]", 4 * D, 24, {
              code: "Block.hc_mixes / hc_attn_fn",
              precision: "FP32 coefficient projection and RMS arithmetic in the reference",
              info: "Flatten four streams to width 20,480 and predict 4 input, 4 output, and 16 residual-matrix coefficients. RMS statistics normalize the projection.",
              backward: "Input residual streams, projection weights, and normalization statistics.",
            }),
            node("sinkhorn", "Sigmoid + 20 Sinkhorn iterations", "A [T, 4], B [T, 4, 4], C [T, 4]", {
              code: "hc_split_sinkhorn",
              params: 27,
              weights: "24 base values + 3 scale values",
              precision: "FP32 coefficient arithmetic",
              info: "B and C combine the attention output below. The newly predicted A is carried forward to the FFN input mix.",
              backward: "Coefficient logits, scaling values, and Sinkhorn intermediates or their recomputation.",
            }),
          ],
        },
      ],
      junction: "Dense input uses the carried mix. Newly predicted B/C remain live for the residual merge.",
      tail: [
        node("attn_norm", "Attention RMSNorm", "[T, 5120]", {
          code: "Block.attn_norm",
          params: D,
          weights: "[5120] scale",
          info: "Normalize the mixed dense stream before the query, local KV, and any current-layer indexer projections.",
        }),
      ],
    });

    const query = [
      projection("q_latent", "Query compression", "[T, 1280]", D, S.queryRank, {
        code: "Attention.wq_a",
        info: "A low-rank query latent also supplies features for this layer's indexer, when the layer produces a selection.",
        backward:
          "Dense input and projection weights for the main query. An indexer objective may also use this latent, depending on the training gradient policy.",
      }),
      node("q_norm", "Query RMSNorm", "[T, 1280]", {
        code: "Attention.q_norm",
        params: S.queryRank,
        weights: "[1280] scale",
        info: "Normalize the query latent before expansion into main and optional indexer heads.",
      }),
      projection("q_expand", "Main query heads + RoPE", "[T, 64, 512]", S.queryRank, S.heads * K, {
        code: "Attention.wq_b",
        info: "Every layer makes 64 main query heads, each width 512. RoPE transforms the last 64 channels. Reuse layers still make fresh queries.",
        backward: "Normalized query latent and weights; attention backward consumes the expanded query.",
      }),
    ];
    const local = [
      projection("local_project", "Local KV projection", "[T, 512]", D, K, {
        code: "Attention.wkv",
        info: `Local KV is generated from ${current}'s incoming dense state, ultimately from ${inputSource}. It is specific to this layer.`,
        backward: "Current dense input and weights; local attention contributes gradients through this projection.",
      }),
      node("local_cache", "RMSNorm + RoPE + local cache", "[min(N, 128), 512]", {
        code: "Attention._window_kv",
        params: K,
        weights: "[512] RMSNorm scale",
        precision: "FP8 cache; quantized after RoPE",
        info: "Store one shared latent per position in this layer's 128-token local window. All 40 layers own a separate local cache.",
        backward:
          "In training, token activations and quantization-aware intermediates are needed; inference ring-buffer state is a separate concern.",
      }),
    ];
    const branches = [
      { label: "Main query / always new", nodes: query },
      { label: "Local KV / always new", nodes: local },
    ];
    if (ratio) {
      const global =
        layer.mode === "Full"
          ? [
              projection(
                "global_project",
                ratio === 2 ? "Global KV + pooling gates" : "Encoder-derived global KV",
                ratio === 2 ? "2 × [G, 512]" : "[G, 512]",
                D,
                ratio === 2 ? 2 * K : K,
                {
                  code: ratio === 2 ? "Compressor.wkv / wgate" : "Compressor.wkv / CED source",
                  weights: ratio === 2 ? "2 × [512, 5120]" : "[512, 5120]",
                  precision: ratio === 2 ? "FP32 projections and softmax pooling in the reference" : "BF16 projection in the reference",
                  info:
                    ratio === 2
                      ? "Project values and channel-wise pooling scores from the current encoder layer input. Each completed, non-overlapping pair produces one latent."
                      : "Project global KV from H20 across the context. This is the global store shared by all 20 decoder layers; ratio 1 needs no pooling gate.",
                  backward: "Source hidden states and projection weights. Gradients from every consuming layer accumulate into this source.",
                }
              ),
              node(
                "global_pool",
                ratio === 2 ? "Pair pooling + RMSNorm" : "Global latent RMSNorm",
                ratio === 2 ? "[completed pairs, 512]" : "[G, 512]",
                {
                  code: "Compressor.forward / norm",
                  params: K,
                  weights: "[512] RMSNorm scale",
                  info: "Preserve the unrotated normalized latent for the indexer K projection. Then apply RoPE and main-cache quantization. Incomplete encoder pairs wait for the next token.",
                  backward:
                    ratio === 2
                      ? "Projected values, pooling scores/softmax weights, and norm intermediates."
                      : "Source projection output and norm intermediates.",
                }
              ),
              node("global_cache", "RoPE + shared main KV cache", "[S, 512]", {
                kind: "shared",
                code: "Attention.compress_kv_cache",
                precision: "Deployment: E2M1 + E4M3 scales per 16 channels",
                info: `${current} owns this cache. The indexer consumes the pre-RoPE latent on a separate branch. The cache contains one latent per ${ratio} context token(s).`,
                backward:
                  "Training retains or recomputes source latents and handles gradients from all consumers. The minimal reference fake-quantizes dequantized buffers.",
              }),
            ]
          : [
              node("global_shared", `Global KV from layer ${layer.kvSource + 1}`, "[S, 512]", {
                kind: "shared",
                code: "SharedAttentionRuntime.compress_kv",
                precision: "FP4 main KV; dequantized for attention",
                info:
                  layer.half === "Decoder"
                    ? "All decoder layers read the same global KV derived from H20 and projected by L21. This layer still follows its sequential hidden-state path."
                    : "Read the global KV made by the Full layer of this six-layer encoder group.",
                backward:
                  "Gradients must return to the producer and sum contributions from the other consumers. Shared state must survive until all backward consumers finish.",
              }),
            ];
      branches.push({ label: layer.mode === "Full" ? "Global memory / produce" : "Global memory / reuse", nodes: global });
    }
    stages.push({ title: `${layer.half} attention / query and memory sources`, color: layer.half.toLowerCase(), branches });

    if (layer.mode === "Reuse")
      stages.push({
        title: "Sparse selection / reuse",
        color: "decoder",
        branches: [
          {
            label: "No new indexer query or scoring",
            nodes: [
              node("indices_shared", `Top-K from layer ${layer.indexSource + 1}`, "[T, ≤512]", {
                kind: "shared",
                code: "SharedAttentionRuntime.topk_idxs",
                precision: "INT32 position indices",
                info: "Reuse the producer's selected positions for each current token. Main attention computes new weights using this layer's own main query.",
                backward: "Indices are discrete. Their shared buffer must remain available for attention backward or be recomputed.",
              }),
            ],
          },
        ],
      });
    else if (ratio) {
      const indexKeys =
        layer.mode === "Full"
          ? [
              projection("index_key", "Indexer K from unrotated global latent", "new keys → cache [S, 128]", K, S.indexDim, {
                code: "Indexer.wk / k_norm",
                params: K * S.indexDim + S.indexDim,
                weights: "[128, 512] matrix + [128] norm",
                precision: "BF16 projection, then norm / RoPE / FP4 quantization",
                info: "Project the main compressor's normalized pre-RoPE latent. Indexer K no longer has a separate hidden-state compressor.",
                backward:
                  "Unrotated main latents and K weights for indexer training. Gradient flow into the backbone depends on the training objective and detach policy.",
              }),
            ]
          : [
              node("index_key_shared", "Indexer K from layer 21", "[S, 128]", {
                kind: "shared",
                code: "SharedAttentionRuntime.index_k",
                precision: "MXFP4; one UE8M0 scale per 32 channels",
                info: "Reuse the key cache that was made alongside the shared main KV. Produce a fresh indexer query below.",
                backward:
                  "Shared indexer keys and their producer for indexer training. Top-K itself is discrete; the objective determines gradient routing.",
              }),
            ];
      stages.push({
        title: `Sparse selection / ${layer.mode}`,
        color: "decoder",
        branches: [
          {
            label: "Fresh indexer query and weights",
            nodes: [
              projection("index_query", "Indexer query + RoPE", "[T, 32, 128]", S.queryRank, S.indexHeads * S.indexDim, {
                code: "Indexer.wq_b",
                precision: "FP4 query quantization after RoPE",
                info: "Expand this layer's query latent into 32 small heads of width 128. These select positions; the 64 main heads perform attention.",
                backward: "Query latent, indexer weights, and quantization-aware intermediates.",
              }),
              projection("index_weights", "Indexer head weights", "[T, 32]", D, S.indexHeads, {
                code: "Indexer.weights_proj",
                precision: "BF16 projection",
                info: "Weight the rectified per-head index scores before summing across the 32 heads.",
              }),
            ],
          },
          { label: "Global indexer key source", nodes: indexKeys },
        ],
        junction:
          layer.mode === "Reindex"
            ? "Restrict this query to the candidate pool selected at L21."
            : "Score all causally visible entries. Decoder L21 also produces the candidate pool.",
        tail: [
          node("index_select", layer.mode === "Reindex" ? "Candidate scoring + fresh Top-512" : "Full-range scoring + Top-512", "[T, ≤512]", {
            kind: "attention",
            code: "Indexer.forward / select_candidate_blocks",
            precision: "FP4 Q/K contraction; INT32 selected positions",
            info:
              layer.mode === "Reindex"
                ? "Rescore at most 16,384 candidate positions per query. The selected positions can differ from earlier layers even though KV is shared."
                : "The initial search scans visible global entries. At L21, top 2,048 eight-position blocks also create the shared candidate pool.",
            backward:
              "Discrete indices themselves do not carry gradients. Training the indexer requires its dedicated objective and score-related intermediates.",
          }),
        ],
      });
    }

    stages.push({
      title: "Read the selected memory / main attention",
      color: layer.half.toLowerCase(),
      branches: [
        {
          label: ratio ? "Own query + own local KV + selected shared global KV" : "Own query + own local KV",
          nodes: [
            node("attention", ratio ? "Gathered sparse attention + sink" : "Sliding-window attention + sink", "[T, 64, 512]", {
              kind: "attention",
              code: "sparse_attn",
              params: S.heads,
              weights: "[64] learned sink logits",
              precision: "Dequantized cached latents; backend-specific attention arithmetic",
              info:
                "Each head reads up to 128 local positions" +
                (ratio ? " plus 512 selected global positions" : "") +
                ". The same stored 512-channel latent supplies K and V. A per-head sink contributes to normalization.",
              backward:
                "Main Q, selected/local KV, output and normalization statistics. The gather/scatter backward sends gradients to all memory sources.",
            }),
            projection("output_group", "Grouped output compression", "[T, 8, 1024]", (S.heads * K) / S.outputGroups, S.outputGroups * S.outputRank, {
              code: "Attention.wo_a",
              weights: "[8, 1024, 4096]",
              precision: "BF16 weights in the reference",
              info: "Eight groups each combine 8 query-head outputs of width 512 into an intermediate of width 1024.",
            }),
            projection("output_project", "Output projection", "[T, 5120]", S.outputGroups * S.outputRank, D, {
              code: "Attention.wo_b",
              precision: "BF16 weights in the reference",
              info: "Concatenate the eight 1024-channel intermediates and project back to model width.",
            }),
            node("attention_merge", "mHC residual / output merge", "[T, 4, 5120]", {
              code: "hc_post",
              info: "Use the current B residual matrix and C output coefficients to combine the original four streams and the dense attention output.",
              backward: "Original residual, dense attention output, and B/C coefficients.",
            }),
          ],
        },
      ],
    });
    stages.push({
      title: "Feed-forward / DeepSeekMoE",
      color: "moe",
      branches: [
        {
          label: "Repeat the Single-Pass mHC cycle",
          nodes: [
            node("ffn_mix", "FFN mHC input mix + RMSNorm", "[T, 5120]", {
              kind: "gemm",
              code: "Block.forward / hc_ffn_fn / ffn_norm",
              params: 4 * D * 24 + 27 + D,
              weights: "[24, 20480] coefficient matrix + scales + [5120] norm",
              info: "Use A predicted by the attention sublayer for the FFN input mix. Predict new FFN coefficients; its A will feed the next block.",
              backward: "Residual streams, attention-produced A, FFN coefficient projection inputs, and norm intermediates.",
            }),
          ],
        },
      ],
      tail: [
        projection("router", "Router projection / sqrt(softplus)", "[T, 384]", D, S.experts, {
          code: "Gate.forward",
          precision: "FP32 routing scores",
          info: "Correction biases for image and text tokens affect which experts are selected. Unbiased scores determine normalized mixture weights, scaled by 1.5.",
          backward: "Router input and weights; selected routing weights supply gradients.",
        }),
      ],
    });
    stages.push({
      title: "Routed and shared expert paths",
      color: "moe",
      branches: [
        {
          label: "Select and dispatch",
          nodes: [
            node("top_experts", "Select 6 expert IDs + weights", "[T, 6]", {
              code: "Gate.forward / topk",
              precision: "Discrete expert IDs + FP32 weights",
              info: "Select six of 384 experts per token. Expert IDs are discrete; selected score values remain part of the differentiable mixture.",
            }),
            node("dispatch", "Expert dispatch", "[R, 5120] / R = 6T", {
              kind: "comm",
              code: "Distributed expert dispatch",
              info: "Group token-expert assignments for local expert GEMMs. Expert parallelism routes each assignment to its expert owner; a GPU receives its share of R.",
              backward:
                "Reverse dispatch/combine and route metadata. The minimal reference loops over local experts rather than providing the production dispatcher.",
            }),
          ],
        },
        {
          label: "Routed expert bank / 6 active per token",
          nodes: [
            projection("expert_up", "Routed gate + up GEMMs", "[R, 4608]", D, 2 * S.expertDim, {
              params: S.experts * 2 * D * S.expertDim,
              weights: "384 × 2 × [2304, 5120]",
              code: "MLP.w1 / w3",
              precision: "FP4 routed weights with block scales; logical activation path",
              info: "The parameter label counts the full bank. Each token executes only six experts, producing two 2304-channel branches per assignment.",
              backward: "Dispatched inputs, selected expert weights, and gate/up activations.",
            }),
            node("expert_act", "Clamped SwiGLU", "[R, 2304]", {
              code: "MLP.forward",
              info: "Clamp according to the model's limit of 10, apply SiLU to the gate, and multiply by the up branch.",
              backward: "Gate and up values and clamp/activation intermediates.",
            }),
            projection("expert_down", "Routed down GEMMs", "[R, 5120]", S.expertDim, D, {
              params: S.experts * S.expertDim * D,
              weights: "384 × [5120, 2304]",
              code: "MLP.w2",
              precision: "FP4 routed weights with block scales",
              info: "Return each routed assignment to model width before weighted combination.",
              backward: "SwiGLU products and selected down weights.",
            }),
          ],
        },
        {
          label: "One shared expert / always active",
          nodes: [
            node("shared_expert", "Shared SwiGLU expert", "[T, 5120]", {
              kind: "gemm",
              code: "MoE.shared_experts",
              params: 3 * D * S.expertDim,
              weights: "2 × [2304, 5120] + [5120, 2304]",
              info: "One gate/up/activation/down expert runs for every token and is added to the routed mixture.",
              backward: "Dense expert input, gate/up intermediates, SwiGLU output, and all three shared expert matrices.",
            }),
          ],
        },
      ],
      junction: "Combine the routed results using their selected weights, and add the shared expert.",
      tail: [
        node("combine", "Weighted expert combine", "[T, 5120]", {
          kind: "comm",
          code: "MoE.forward / expert combine",
          info: "Return routed outputs to their token positions, sum the six weighted contributions, and add the shared path.",
          backward: "Expert outputs, route metadata, and mixture weights; gradients flow into expert outputs and routing weights.",
        }),
        node("ffn_merge", "mHC merge / carry A to next block", "[T, 4, 5120] + next input mix [T, 4]", {
          code: "hc_post / Block.forward return",
          info: "Combine the dense FFN output with the four residual streams. Return the FFN's A coefficients for the following block.",
          backward: "Residual and FFN outputs, current B/C, and the carried A dependency into the next block.",
        }),
      ],
    });
    return stages;
  }

  function renderInspector() {
    const op = operationMap.get(selectedOperation);
    const inspector = byId("flow-inspector");
    inspector.style.setProperty("--accent", `var(--${op.color})`);
    inspector.innerHTML = `<div class="eyebrow">Layer ${selectedLayer + 1} / ${escape(op.kind)}</div>
      <h3>${escape(op.title)}</h3><p>${escape(op.info)}</p>
      <dl><dt>Logical output</dt><dd><code>${escape(op.shape)}</code></dd>
      <dt>Owned parameters</dt><dd>${op.params ? integer(op.params) + " / " + count(op.params) : "0 / no new parameter matrix here"}</dd>
      <dt>Weights / source</dt><dd><code>${escape(op.weights)}</code></dd>
      <dt>Precision</dt><dd>${escape(op.precision)}</dd>
      <dt>Reference path</dt><dd><code>${escape(op.code)}</code></dd>
      <dt>Backward dependencies</dt><dd>${escape(op.backward)}</dd></dl>
      <p class="note">Shared cache size and owned weight count are separate. Training needs saved intermediates or their recomputation.</p>
      <a class="note" href="#operation-${op.id}">Back to this operation &uarr;</a>`;
  }

  function renderGraph() {
    const layer = M.layer(selectedLayer);
    const stages = graph(layer);
    const tokens = Number(byId("flow-tokens").value);
    operationMap = new Map();
    for (const stage of stages)
      for (const op of [...stage.branches.flatMap((branch) => branch.nodes), ...(stage.tail || [])]) {
        op.color = stage.color;
        operationMap.set(op.id, op);
      }
    if (!operationMap.has(selectedOperation)) selectedOperation = "q_expand";
    const labels = byId("flow-lens").value;
    const button = (op) => {
      const label = labels === "params" ? (op.params ? `${count(op.params)} parameters` : "No owned matrix") : op.shape;
      return `<button type="button" class="flow-node ${op.kind}${op.id === selectedOperation ? " selected" : ""}" id="operation-${
        op.id
      }" data-operation="${op.id}" aria-pressed="${op.id === selectedOperation}">
        <strong>${escape(op.title)}</strong><small class="mono">${escape(label)}</small></button>`;
    };
    const chain = (ops) => ops.map(button).join('<div class="flow-arrow" aria-hidden="true">&darr;</div>');
    byId("flow-diagram").innerHTML = `<h3 class="flow-title">Layer ${layer.number} / ${layer.half} ${layer.mode}</h3>
      <p class="flow-caption">T = ${integer(tokens)} &middot; R = ${integer(6 * tokens)} &middot; ${
        layer.ratio ? "global ratio " + layer.ratio + ":1" : "local attention only"
      } &middot; release index ${layer.index}</p>
      ${stages
        .map(
          (stage) => `<div class="flow-stage" style="--accent: var(--${stage.color}); --accent-soft: var(--${stage.color}-soft)">
        <div class="flow-stage-title">${escape(stage.title)}</div>
        <div class="${stage.branches.length === 1 ? "flow-single" : "flow-branches" + (stage.branches.length === 2 ? " two" : "")}">
          ${stage.branches.map((branch) => `<div><div class="flow-branch-label">${escape(branch.label)}</div>${chain(branch.nodes)}</div>`).join("")}
        </div>${stage.junction ? `<div class="flow-junction">${escape(stage.junction)}</div>` : ""}
        ${stage.tail ? `<div class="flow-single">${chain(stage.tail)}</div>` : ""}
      </div>`
        )
        .join("")}`;
    renderInspector();
  }

  function selectLayer(index) {
    selectedLayer = index;
    const layer = M.layer(index);
    byId("flow-layer").value = String(index);
    for (const tile of byId("layer-map").querySelectorAll("[data-layer]")) {
      const tileIndex = Number(tile.dataset.layer);
      tile.classList.toggle("selected", tileIndex === index);
      tile.classList.toggle("kv-owner", layer.kvSource === tileIndex && tileIndex !== index);
      tile.classList.toggle("index-owner", layer.indexSource === tileIndex && tileIndex !== index);
      tile.setAttribute("aria-pressed", String(tileIndex === index));
    }
    byId("layer-readout").innerHTML = `<div><strong>Layer ${layer.number} / ${layer.mode}</strong><span>${layer.half} &middot; release index ${
      layer.index
    }${layer.hasEngram ? " &middot; Engram" : ""}</span></div>
      <div><strong>${layer.ratio ? "Global KV: layer " + (layer.kvSource + 1) : "No global KV"}</strong><span>${
        layer.half === "Decoder" ? "Derived from encoder output H20" : layer.ratio ? "Compression " + layer.ratio + ":1" : "Local window only"
      }</span></div>
      <div><strong>${
        layer.ratio ? "Top-K indices: layer " + (layer.indexSource + 1) : "No global indexer"
      }</strong><span>Own main query and local KV</span></div>`;
    renderGraph();
  }

  function renderLayers() {
    const groups = ["Encoder", "Decoder"];
    byId("layer-map").innerHTML = groups
      .map(
        (half) => `<div class="layer-group ${half.toLowerCase()}">
      <div class="layer-group-title"><strong>${half}</strong><span>${half === "Encoder" ? "Layers 1–20" : "Layers 21–40"}</span></div>
      <div class="layer-tiles">${M.layers
        .filter((layer) => layer.half === half)
        .map((layer) => {
          const mode = { Full: "F", Reindex: "I", Reuse: "R", SWA: "S" }[layer.mode];
          return `<button type="button" class="layer-tile mode-${layer.mode.toLowerCase()}" data-layer="${layer.index}" aria-pressed="false"
          aria-label="Layer ${layer.number}, ${layer.half}, ${layer.mode}${layer.hasEngram ? ", Engram" : ""}">
          <strong>${layer.number}</strong><small>${mode}</small>${
            layer.hasEngram ? '<span class="engram-mark" aria-hidden="true">E</span>' : ""
          }</button>`;
        })
        .join("")}</div></div>`
      )
      .join("");
    byId("flow-layer").innerHTML = groups
      .map(
        (half) =>
          `<optgroup label="${half}">${M.layers
            .filter((layer) => layer.half === half)
            .map((layer) => `<option value="${layer.index}">Layer ${layer.number}: ${layer.mode}${layer.hasEngram ? " + Engram" : ""}</option>`)
            .join("")}</optgroup>`
      )
      .join("");
    byId("layer-map").addEventListener("click", (event) => {
      const tile = event.target.closest("[data-layer]");
      if (tile) selectLayer(Number(tile.dataset.layer));
    });
    byId("flow-layer").addEventListener("change", (event) => selectLayer(Number(event.target.value)));
    for (const id of ["flow-tokens", "flow-lens"]) byId(id).addEventListener("change", renderGraph);
    byId("flow-diagram").addEventListener("click", (event) => {
      const button = event.target.closest("[data-operation]");
      if (!button) return;
      selectedOperation = button.dataset.operation;
      for (const item of byId("flow-diagram").querySelectorAll("[data-operation]")) {
        const selected = item.dataset.operation === selectedOperation;
        item.classList.toggle("selected", selected);
        item.setAttribute("aria-pressed", String(selected));
      }
      renderInspector();
      if (window.matchMedia("(max-width: 760px)").matches) byId("flow-inspector").scrollIntoView({ block: "start" });
    });
    selectLayer(selectedLayer);
  }

  function renderCed() {
    const tokens = contexts[Number(byId("ced-context").value)];
    const phase = byId("ced-phase").value;
    const w = M.work(tokens, phase);
    byId("ced-context-label").textContent = `${integer(tokens)} tokens`;
    byId("ced-context").setAttribute("aria-valuetext", `${integer(tokens)} tokens`);
    byId("ced-work-fraction").textContent = `${(100 * w.fraction).toFixed(1)}%`;
    byId("ced-encoder-label").textContent =
      phase === "decode" ? "Current generated token" : `All ${integer(tokens)} ${phase === "training" ? "training" : "prompt"} tokens`;
    byId("ced-decoder-label").textContent =
      phase === "decode"
        ? "Same generated token"
        : phase === "training"
          ? `All ${integer(tokens)} training tokens`
          : `Last ${integer(w.decoderTokens)} prompt tokens`;
    const notes = {
      prefill:
        "Compute H20 for the full prompt, project decoder global KV, then process the last 128 positions through the decoder. Short prompts use their full length.",
      decode:
        "Each new token runs all 40 layers. Global KV is extended at its producer layers; every layer updates its local cache. Attention can read the causally visible prefix.",
      training:
        "Teacher-forced next-token training needs decoder outputs at all prediction positions. Every training token runs both halves with causal masking.",
    };
    byId("ced-phase-note").textContent = notes[phase];
    byId("ced-memory-label").textContent =
      phase === "prefill"
        ? "Global KV from encoder outputs across the full prompt → decoder attention"
        : "Encoder-derived global KV for current and earlier positions → decoder attention";
    for (const [key, value] of [
      ["encoder", w.encoder],
      ["decoder", w.decoder],
      ["baseline", w.baseline],
    ]) {
      byId(`ced-${key}-fill`).style.width = `${(100 * value) / w.baseline}%`;
      byId(`ced-${key}-work`).textContent = count(value);
    }
    byId("ced-work-formula").textContent =
      phase === "decode"
        ? "20 × 1 + 20 × 1 = 40 token-layer executions"
        : phase === "training"
          ? `40 × ${integer(tokens)} = ${integer(w.total)} token-layer executions`
          : `20 × ${integer(tokens)} + 20 × ${integer(w.decoderTokens)} = ${integer(w.total)}`;
  }

  function renderCache() {
    const tokens = contexts[Number(byId("cache-context").value)];
    const bits = Number(byId("cache-bits").value);
    const c = M.cache(tokens, bits);
    byId("cache-context-label").textContent = `${integer(tokens)} tokens`;
    byId("cache-context").setAttribute("aria-valuetext", `${integer(tokens)} tokens`);
    byId("cache-total").textContent = bytes(c.total);
    for (const key of ["main", "index", "local"]) {
      byId(`cache-${key}`).textContent = bytes(c[key]);
      byId(`cache-${key}-fill`).style.width = `${(100 * c[key]) / c.total}%`;
    }
    byId("cache-formula").textContent = `${integer(c.entries)} entries × ${bits === 4 ? "(256 + 32 + 64 + 4)" : "(1024 + 0 + 64 + 4)"} B / ${integer(
      tokens
    )} tokens = ${c.bytesPerToken.toFixed(0)} global B/token`;
    byId("cache-scale-note").textContent =
      bits === 4
        ? `Main scales add ${bytes(c.mainScales)}; indexer scales add ${bytes(c.indexScales)}. Included above. Local values count FP8 payload only.`
        : "Hypothetical BF16 main cache; indexer K retains FP4 and its scales. Local values count FP8 payload only.";
    byId("cache-producers").innerHTML = c.producers
      .map(
        (producer) => `<div class="producer"><strong>L${producer.layer}</strong><span>${integer(producer.entries)}</span><small>entries</small></div>`
      )
      .join("");
    byId("cache-unshared").textContent = `Independent global KV at every CSA2 layer: ${bytes(c.unsharedGlobal)}. Shared global KV: ${bytes(
      c.global
    )}. Sharing reduces this comparison by ${(c.unsharedGlobal / c.global).toFixed(1)}×.`;
  }

  function renderEngram() {
    const groups = M.recentNgrams(byId("engram-text").value);
    byId("ngram-grid").innerHTML = groups
      .map(
        (group) => `<div class="ngram-group"><span class="label">${group.order}-gram / ending at current token</span>
      <strong class="ngram-phrase">${escape(group.tokens.join(" "))}</strong>
      <div class="hash-heads" role="group" aria-label="Eight hash-head table lookups">${Array.from(
        { length: 8 },
        (_, index) => `<span>Head ${index + 1}</span>`
      ).join("")}</div>
      <p class="note">8 vectors × 256 channels</p></div>`
      )
      .join("");
    const gate = Number(byId("engram-gate").value) / 100;
    byId("engram-gate-label").textContent = gate.toFixed(2);
    byId("engram-gate-result").textContent = `Updated stream = H + ${gate.toFixed(2)} × V`;
  }

  function initTheme() {
    let saved;
    try {
      saved = localStorage.getItem("architecture-study-theme") || localStorage.getItem("kimi-k3-study-theme");
    } catch {}
    const root = document.documentElement;
    if (saved === "dark" || (!saved && window.matchMedia("(prefers-color-scheme: dark)").matches)) root.classList.add("dark");
    const label = () => {
      const dark = root.classList.contains("dark");
      byId("theme-toggle").textContent = dark ? "Light" : "Dark";
      byId("theme-toggle").setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
    };
    label();
    byId("theme-toggle").addEventListener("click", () => {
      root.classList.toggle("dark");
      try {
        localStorage.setItem("architecture-study-theme", root.classList.contains("dark") ? "dark" : "light");
      } catch {}
      label();
    });
    byId("print-button").addEventListener("click", () => window.print());
  }

  initTheme();
  renderLayers();
  for (const id of ["ced-context", "ced-phase"]) byId(id).addEventListener("input", renderCed);
  for (const id of ["cache-context", "cache-bits"]) byId(id).addEventListener("input", renderCache);
  for (const id of ["engram-text", "engram-gate"]) byId(id).addEventListener("input", renderEngram);
  renderCed();
  renderCache();
  renderEngram();
})();
