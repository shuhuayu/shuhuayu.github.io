(() => {
  "use strict";
  const model = window.QwenFlashNextModel;
  const c = model.config;
  const D = c.dim,
    C = D * c.streams,
    F = c.expertWidth;
  const controls = Object.fromEntries(["layer", "lens", "storage", "tokens", "dtype"].map((key) => [key, document.getElementById(`kernel-${key}`)]));
  const diagram = document.getElementById("kernel-diagram");
  const inspector = document.getElementById("kernel-inspector");
  let selectedNode = "attn-norm";
  let nodes = new Map();

  const count = (value) =>
    value >= 1e9 ? `${(value / 1e9).toFixed(3)}B` : value >= 1e6 ? `${(value / 1e6).toFixed(2)}M` : value.toLocaleString("en-US");
  const bytes = (value) =>
    value >= 2 ** 30
      ? `${(value / 2 ** 30).toFixed(2)} GiB`
      : value >= 2 ** 20
        ? `${(value / 2 ** 20).toFixed(2)} MiB`
        : value >= 1024
          ? `${(value / 1024).toFixed(2)} KiB`
          : `${value} B`;
  const shape = (dims) => `[${dims.join(", ")}]`;
  const op = (id, title, dims, options = {}) => ({
    id,
    title,
    dims,
    kind: "op",
    params: 0,
    weight: "No learned weight matrix",
    save: true,
    ...options,
  });
  const gemm = (id, title, dims, input, output, options = {}) =>
    op(id, title, dims, { kind: "gemm", params: input * output, weight: shape([output, input]), ...options });
  const numericDim = (dim) =>
    dim === "T"
      ? Number(controls.tokens.value)
      : dim === "R"
        ? c.topK * Number(controls.tokens.value)
        : dim === "G"
          ? Math.floor(Number(controls.tokens.value) / c.blockSize)
          : dim === "B"
            ? Math.min(512, Math.floor(Number(controls.tokens.value) / c.blockSize))
            : dim;

  function residualRead(prefix, title) {
    return {
      title,
      accent: "residual",
      pre: [
        op(`${prefix}-norm`, "Grouped offset RMSNorm", ["T", C], {
          params: C,
          weight: `[${C}] offset scales, grouped into 4 streams`,
          info: "Normalize each 2560-wide stream separately in FP32 and apply (1 + scale). The read gate and write gate both use the normalized flattened streams.",
          backward: "Residual input, per-stream RMS statistics, and normalization scales.",
        }),
      ],
      branches: [
        {
          label: "Element-wise read gates",
          nodes: [
            gemm(`${prefix}-down`, "Read down / 4 + SiLU", ["T", 320], C, 320, {
              info: "Project all four normalized streams into a rank-320 bottleneck, divide by four, then apply SiLU.",
              backward: "Normalized streams, projection weights, and pre-SiLU values.",
            }),
            gemm(`${prefix}-up`, "Read up + sigmoid", ["T", C], 320, C, {
              info: "Produce one sigmoid gate for each stream and each channel.",
              backward: "Bottleneck activation, projection weights, and sigmoid gate values.",
            }),
          ],
        },
        {
          label: "Per-stream write gates",
          nodes: [
            gemm(`${prefix}-write`, "Write projection + 2 sigmoid", ["T", 4], C, 4, {
              info: "Predict four scalar gates as 2 * sigmoid(projection / 4). Keep these gates for the residual injection after the sublayer.",
              backward: "Normalized streams, write projection weights, and gate values.",
            }),
          ],
        },
      ],
      tail: [
        op(`${prefix}-read`, "Mean of gated residual streams", ["T", D], {
          info: "Multiply each normalized stream by its channel-wise read gates and average the four streams. The sublayer still operates at width 2560.",
          backward: "Read gates and normalized stream values; the average contributes a factor of 1/4.",
        }),
      ],
    };
  }

  function injection(prefix, title) {
    return {
      title,
      accent: "residual",
      tail: [
        op(`${prefix}-inject`, "Residual + scalar-gated update", ["T", C], {
          info: "For stream i, add write_gate[i] * sublayer_update to the original residual. The four streams persist into the next read.",
          backward: "Sublayer update and write gates. Addition passes an identity gradient to the incoming residual.",
        }),
      ],
    };
  }

  function lexicalStage() {
    return {
      title: "Lexical injection at layer 2",
      accent: "gold",
      pre: [
        op("ple-hash", "Bigram / trigram hash IDs", ["T", 16], {
          save: false,
          elementBytes: 8,
          info: "Compute sixteen integer hash addresses from original token IDs. Each hash head has its own prime-sized vocabulary. History resets at EOS and document boundaries.",
          backward: "Hashing has no gradient. The embedding backward uses the resulting integer lookup IDs.",
        }),
        op("ple-lookup", "16 lexical lookups + concatenate", ["T", D], {
          params: c.lexicalRows * c.lexicalHeadDim,
          weight: `[${c.lexicalRows}, ${c.lexicalHeadDim}] in 128 checkpoint row chunks`,
          info: "Gather sixteen rows of width 160, then concatenate into 2560 lexical features. The full 51.2B table is a lookup bank, not a dense matrix multiply.",
          backward: "Integer hash IDs select the rows receiving embedding gradients.",
        }),
      ],
      branches: [
        {
          label: "Lexical keys",
          nodes: [
            gemm("ple-key", "Key projection", ["T", C], D, C, { backward: "Lexical feature vectors and key projection weights." }),
            op("ple-key-norm", "Grouped key RMSNorm", ["T", C], {
              params: C,
              weight: `[${C}] grouped offset scales`,
              backward: "Projected keys and per-stream RMS statistics.",
            }),
          ],
        },
        {
          label: "Shared lexical value",
          nodes: [gemm("ple-value", "Value projection", ["T", D], D, D, { backward: "Lexical feature vectors and value projection weights." })],
        },
        {
          label: "Incoming residual streams",
          nodes: [
            op("ple-query-norm", "Grouped query RMSNorm", ["T", C], {
              params: C,
              weight: `[${C}] grouped offset scales`,
              info: "This branch reads the incoming residual, separately from the lexical lookup output.",
              backward: "Incoming residual streams and per-stream RMS statistics.",
            }),
          ],
        },
      ],
      tail: [
        op("ple-gate", "Key / query dot + signed sqrt + sigmoid", ["T", 4], {
          info: "Compute one similarity per stream, scale by sqrt(2560), apply signed sqrt with the reference epsilon, then sigmoid.",
          backward: "Normalized keys and queries, similarity signs/magnitudes, and sigmoid gates.",
        }),
        op("ple-gated-value", "Broadcast value x stream gates", ["T", C], { backward: "Shared value vector and the four stream gates." }),
        op("ple-conv-norm", "Grouped convolution RMSNorm", ["T", C], {
          params: C,
          weight: `[${C}] grouped offset scales`,
          backward: "Gated values and RMS statistics.",
        }),
        op("ple-conv", "Width-4 / dilation-3 causal conv + SiLU", ["T", C], {
          params: 4 * C,
          weight: `[${C}, 1, 4] depthwise kernel`,
          info: "Read current and earlier positions at offsets 0, 3, 6, and 9, respecting document boundaries. The convolution adds local lexical context.",
          backward: "Normalized gated values, convolution weights, and pre-SiLU output.",
        }),
        op("ple-add", "Residual + gated value + conv output", ["T", C], {
          backward: "Addition sends gradients to the residual, gated lexical value, and convolution output.",
        }),
      ],
    };
  }

  function gdnStage() {
    return {
      title: "Gated DeltaNet",
      accent: "gdn",
      branches: [
        {
          label: "Q / K / V",
          nodes: [
            gemm("gdn-qkv", "Q + K + V projection", ["T", 10240], D, 10240, {
              info: "Split into Q [T,16,128], K [T,16,128], and V [T,48,128].",
              backward: "Sublayer input and QKV projection weights.",
            }),
            op("gdn-conv", "Width-4 causal conv + SiLU", ["T", 10240], {
              params: 40960,
              weight: "[10240, 1, 4] depthwise kernel",
              backward: "QKV projection output, convolution kernel, and pre-SiLU values.",
            }),
            op("gdn-qk-norm", "Q / K L2 norm + head grouping", ["T", 48, 128], {
              info: "Normalize Q and K along their 128 channels. Each of the sixteen Q/K heads serves three value heads; the displayed shape is for one expanded Q or K tensor.",
              backward: "Unnormalized Q/K and inverse L2 norms. Head-group expansion sums repeated gradients back to sixteen heads.",
            }),
          ],
        },
        {
          label: "State update gates",
          nodes: [
            gemm("gdn-state-gates", "Beta / decay projections + transforms", ["T", 96], D, 96, {
              params: D * 96 + 96,
              weight: `2 x [48, ${D}], plus [48] A_log and [48] dt_bias`,
              elementBytes: 4,
              info: "Produce 48 beta scalars and 48 decay scalars. Beta uses sigmoid; the decay uses learned A_log and dt_bias with softplus. Decay/state arithmetic uses FP32.",
              backward: "Sublayer input, projection weights, beta/decay values, A_log, and dt_bias.",
            }),
          ],
        },
        {
          label: "Output gate",
          nodes: [
            gemm("gdn-output-gate", "Output gate projection", ["T", 6144], D, 6144, {
              backward: "Sublayer input and output-gate projection weights.",
            }),
          ],
        },
      ],
      tail: [
        op("gdn-recurrence", "Chunked gated delta-rule kernel", ["T", 48, 128], {
          kind: "attention",
          info: "Mix Q/K/V using recurrent matrices [48,128,128]. Training processes chunks; decode updates persistent state one token at a time.",
          backward: "Q/K/V, decay and beta gates, and backend-specific recurrent/chunk summaries; a fused backend can recompute parts.",
        }),
        op("gdn-output-norm", "Per-head RMSNorm x sigmoid gate", ["T", 48, 128], {
          params: 128,
          weight: "[128] shared per-head RMSNorm scale",
          info: "Normalize each 128-wide value head and multiply by sigmoid of the projected gate. The 128 normalization weights are shared across heads.",
          backward: "Ungated recurrent output, RMS statistics, scales, and sigmoid gate values.",
        }),
        gemm("gdn-output", "Output projection", ["T", D], 6144, D, { backward: "Normalized/gated output and output projection weights." }),
      ],
    };
  }

  function qsaStage() {
    return {
      title: "Qwen Sparse Attention",
      accent: "qsa",
      branches: [
        {
          label: "Learned block indexer",
          nodes: [
            gemm("qsa-index-proj", "Indexer Q / raw K projection", ["T", 640], D, 640, {
              info: "Four query heads and one shared key head, each with 128 channels.",
              backward: "Sublayer input and the small indexer projection weights.",
            }),
            op("qsa-index-pool", "Average raw keys in 4-token blocks", ["G", 128], {
              info: "Pool raw key vectors before normalization and positional encoding. G = floor(T/4) for a single document; packed documents each reset block boundaries.",
              backward: "Pool backward distributes the block gradient evenly to its four raw keys.",
            }),
            op("qsa-index-norm", "Indexer Q/K RMSNorm + partial RoPE", ["G", 128], {
              params: 256,
              weight: "2 x [128] offset scales",
              info: "Normalize four query heads and one pooled key head. Rotate 64 channels; the pooled key uses its block's first token position. The displayed shape describes pooled keys.",
              backward: "Indexer query/key values, RMS statistics, normalization scales, and rotary coordinates.",
            }),
            op("qsa-select", "Rectified head scores + top blocks", ["T", "B"], {
              elementBytes: 4,
              info: "Score all causally visible complete blocks, sum ReLU scores across the four query heads, and retain up to 512. B is min(512,G). The tail is included separately.",
              backward: "Top-k has no gradient through its discrete indices. Selected score values receive the indexer KL gradient.",
            }),
          ],
        },
        {
          label: "Main query + output gate",
          nodes: [
            gemm("qsa-query", "Q + output gate projection", ["T", 12288], D, 12288, {
              info: "Split into query [T,24,256] and output gate [T,24,256].",
              backward: "Sublayer input and query/gate projection weights.",
            }),
            op("qsa-query-norm", "Query RMSNorm + 64-channel RoPE", ["T", 24, 256], {
              params: 256,
              weight: "[256] offset RMSNorm scale",
              backward: "Projected Q, RMS statistics, normalization scale, and multimodal rotary coordinates.",
            }),
          ],
        },
        {
          label: "Main keys / values",
          nodes: [
            gemm("qsa-kv", "K / V projections", ["T", 2, 2, 256], D, 1024, {
              weight: "2 x [512, 2560]",
              info: "Two KV heads of width 256 for each of K and V. These full-resolution vectors remain in the growing history cache.",
              backward: "Sublayer input, K/V projection weights, and attention gradients.",
            }),
            op("qsa-key-norm", "Key RMSNorm + 64-channel RoPE", ["T", 2, 256], {
              params: 256,
              weight: "[256] offset RMSNorm scale",
              backward: "Projected K, RMS statistics, normalization scale, and rotary coordinates.",
            }),
          ],
        },
      ],
      tail: [
        op("qsa-sparse", "Selected-token GQA", ["T", 24, 256], {
          kind: "attention",
          info: "Expand selected block IDs to original token IDs and attend to up to 2048 + 3 tokens per query. GQA maps 24 query heads to two KV heads. A sparse kernel or chunked gather bounds temporary storage.",
          backward: "Main Q/K/V, selected indices, output, and backend normalization statistics; exact saved tensors depend on the kernel.",
        }),
        op("qsa-output-gate", "Attention output x sigmoid gate", ["T", 6144], {
          backward: "Ungated attention output and the query projection's sigmoid output gate.",
        }),
        gemm("qsa-output", "Output projection", ["T", D], 6144, D, { backward: "Gated attention output and output projection weights." }),
      ],
      auxiliary: [
        op("qsa-kl", "Detached attention teacher + indexer KL", [], {
          elementBytes: 4,
          info: "Sparse training forms a detached block teacher [T,B] by summing main-attention probabilities across heads, max-pooling within complete blocks, and normalizing over selected blocks. The KL objective produces a scalar auxiliary loss; the main attention output continues along its own path.",
          backward: "Selected indexer logits and target distribution. The detached teacher carries no gradient into main Q/K.",
        }),
      ],
    };
  }

  function moeStage() {
    return {
      title: "Routed + shared SwiGLU experts",
      accent: "moe",
      branches: [
        {
          label: "512 choose 10 routed experts",
          nodes: [
            gemm("moe-router", "Router logits", ["T", 512], D, 512, {
              info: "Logits are rounded to the activation dtype before FP32 softmax, preserving reference expert selection near ties.",
              backward: "Sublayer input, router weight, and gradients through selected probabilities.",
            }),
            op("moe-topk", "FP32 softmax + top 10 + normalize", ["T", 10], {
              info: "Normalize selected probabilities and cast the mixing weights to the activation dtype. Keep the selected IDs and routing metadata.",
              backward: "Router probabilities, selected expert IDs, normalization sums, and mixing weights.",
            }),
            op("moe-dispatch", "Dispatch tokens to expert owners", ["R", D], {
              kind: "comm",
              info: "R = 10T is the assignment count over the logical expert group. Permute and communicate activations according to expert ownership.",
              backward: "Routing/permutation metadata. Backward returns gradients to the original token owners.",
            }),
            gemm("moe-gate-up", "Grouped expert gate / up GEMMs", ["R", 1280], D, 1280, {
              params: 512 * D * 1280,
              weight: "[512, 1280, 2560] logical fused expert bank",
              info: "Stored weight count includes all 512 experts. Each token uses only its selected ten expert matrices.",
              backward: "Dispatched expert inputs and gate/up weights for the experts receiving assignments.",
            }),
            op("moe-swiglu", "SiLU(gate) x up", ["R", F], { backward: "Gate and up activations before the nonlinear product." }),
            gemm("moe-down", "Grouped expert down GEMM", ["R", D], F, D, {
              params: 512 * F * D,
              weight: "[512, 2560, 640] expert bank",
              backward: "SwiGLU products and selected down-projection weights.",
            }),
            op("moe-combine", "Weighted combine + return to owners", ["T", D], {
              kind: "comm",
              backward: "Expert outputs, mixing weights, and routing metadata; reduce multiple assignments back into each token.",
            }),
          ],
        },
        {
          label: "One shared expert",
          nodes: [
            gemm("shared-gate-up", "Shared gate / up projection", ["T", 1280], D, 1280, {
              backward: "Every token's sublayer input and shared gate/up weights.",
            }),
            op("shared-swiglu", "Shared SwiGLU product", ["T", F], { backward: "Shared gate and up activations." }),
            gemm("shared-down", "Shared down projection", ["T", D], F, D, { backward: "Shared SwiGLU products and down weights." }),
            gemm("shared-gate", "Shared output gate + sigmoid", ["T", 1], D, 1, {
              info: "This projection reads the original sublayer input. Its scalar sigmoid modulates the shared expert output.",
              backward: "Sublayer input, gate weights, sigmoid values, and shared expert output.",
            }),
          ],
        },
      ],
      tail: [
        op("moe-sum", "Routed output + gated shared output", ["T", D], {
          backward: "Shared output and its scalar gate; addition forwards gradients to both expert branches.",
        }),
      ],
    };
  }

  function nodeHTML(item) {
    nodes.set(item.id, item);
    const label = controls.lens.value === "params" ? (item.params ? `${count(item.params)} stored weights` : "No learned weights") : shape(item.dims);
    const saved = controls.storage.value === "autograd" && item.save;
    return `<button type="button" class="kernel-node ${item.kind}${saved ? " saved" : ""}" data-node="${item.id}" data-params="${
      item.params
    }" aria-pressed="${item.id === selectedNode}"><span class="title">${item.title}</span><span class="shape">${label}</span></button>`;
  }

  function stageHTML(stage) {
    const pre = stage.pre?.length ? `<div class="kernel-tail">${stage.pre.map(nodeHTML).join("")}</div>` : "";
    const branches = stage.branches?.length
      ? `<div class="kernel-branches">${stage.branches
          .map((branch) => `<div><span class="kernel-branch-label">${branch.label}</span>${branch.nodes.map(nodeHTML).join("")}</div>`)
          .join("")}</div>`
      : "";
    const tail = stage.tail?.length ? `<div class="kernel-tail">${stage.tail.map(nodeHTML).join("")}</div>` : "";
    const auxiliary = stage.auxiliary?.length
      ? `<div class="kernel-aux"><span class="kernel-branch-label">Training branch from attention probabilities and indexer scores</span>${stage.auxiliary
          .map(nodeHTML)
          .join("")}</div>`
      : "";
    return `<div class="kernel-stage" style="--accent: var(--${stage.accent})"><h4>${stage.title}</h4>${pre}${branches}${tail}${auxiliary}</div>`;
  }

  function selectNode(id) {
    selectedNode = id;
    const item = nodes.get(id);
    diagram.querySelectorAll("[data-node]").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.node === id)));
    const outputElements = item.dims.reduce((total, dim) => total * numericDim(dim), 1);
    const outputBytes = outputElements * (item.elementBytes ?? Number(controls.dtype.value));
    const checkpoint = controls.storage.value === "checkpoint";
    inspector.innerHTML = `<span class="label">Selected operation</span><h3>${item.title}</h3><p>${
      item.info ?? "This operation transforms the previous logical tensor as shown in the diagram."
    }</p><h4>Logical output</h4><p class="mono inspector-value">${shape(item.dims)}</p><p>${bytes(outputBytes)} at the selected dtype${
      item.elementBytes ? ` (${item.elementBytes}-byte elements for this operation)` : ""
    }.</p><h4>Learned weights</h4><p class="inspector-value">${count(item.params)}</p><p class="mono">${item.weight}</p><h4>Backward needs</h4><p>${
      item.backward
    }</p><h4>${checkpoint ? "Full layer checkpoint" : "Storage interpretation"}</h4><p>${
      checkpoint
        ? "Intermediates are regenerated during backward. The layer boundary input, original token IDs/document metadata, and weights must still be available."
        : "Dots indicate outputs used by later backward computations. Fusion and backend recomputation can change which tensors are retained."
    }</p><p class="note">Output bytes are a logical tensor size, not a peak-memory estimate. Attention blocks and grouped operations can process smaller chunks.</p>`;
  }

  function renderDiagram() {
    const layer = model.layerSpec(Number(controls.layer.value));
    nodes = new Map();
    const stages = [
      ...(layer.ple ? [lexicalStage()] : []),
      residualRead("attn", "Gated residual read for attention"),
      layer.attention === "gdn" ? gdnStage() : qsaStage(),
      injection("attn", "Write attention output into four streams"),
      residualRead("ffn", "Gated residual read for MoE"),
      moeStage(),
      injection("ffn", "Write MoE output into four streams"),
    ];
    const content = stages.map(stageHTML).join("");
    const layerWeights = [...nodes.values()].reduce((sum, item) => sum + item.params, 0);
    diagram.innerHTML = `<h3>Layer ${layer.layer}: ${layer.ple ? "PLE + " : ""}${
      layer.attention === "gdn" ? "GDN" : "QSA"
    } + MoE</h3><p class="kernel-caption">Input / output [T, 10240] &middot; sublayers [T, 2560] &middot; ${count(layerWeights)} stored weights${
      layer.ple ? ", including the lexical table" : ""
    }</p>${content}`;
    if (!nodes.has(selectedNode)) selectedNode = "attn-norm";
    diagram.querySelectorAll("[data-node]").forEach((button) => button.addEventListener("click", () => selectNode(button.dataset.node)));
    selectNode(selectedNode);
    document
      .querySelectorAll(".layer-tile")
      .forEach((button) => button.setAttribute("aria-pressed", String(Number(button.dataset.layer) === layer.layer)));
    document.getElementById("layer-readout").innerHTML = `<strong>Layer ${layer.layer}</strong><span>${layer.ple ? "Lexical injection, then " : ""}${
      layer.attention === "gdn" ? "GDN with a fixed recurrent matrix" : "QSA with a 512-block selection budget"
    }; 512 choose 10 routed experts + one shared expert. Attention and MoE each read and write four gated residual streams.</span>`;
    document.getElementById("kernel-memory-note").textContent = `At T = ${Number(controls.tokens.value).toLocaleString(
      "en-US"
    )}, one [T,10240] residual tensor is ${bytes(
      Number(controls.tokens.value) * C * Number(controls.dtype.value)
    )}. A width-2560 sublayer input is one quarter of that size. Expert assignment shapes count all 10T assignments before expert parallelism divides them among GPUs.`;
  }

  for (let layer = 1; layer <= c.layers; layer++) {
    const spec = model.layerSpec(layer);
    const option = document.createElement("option");
    option.value = String(layer);
    option.textContent = `Layer ${layer}: ${spec.ple ? "PLE + " : ""}${spec.attention.toUpperCase()} + MoE`;
    controls.layer.appendChild(option);
    const tile = document.createElement("button");
    tile.type = "button";
    tile.dataset.layer = String(layer);
    tile.className = `layer-tile ${spec.attention}${spec.ple ? " ple" : ""}`;
    tile.textContent = String(layer);
    tile.setAttribute("aria-label", option.textContent);
    tile.addEventListener("click", () => {
      controls.layer.value = String(layer);
      renderDiagram();
    });
    document.getElementById("layer-map").appendChild(tile);
  }
  controls.layer.value = "2";
  Object.values(controls).forEach((control) => control.addEventListener("change", renderDiagram));
  renderDiagram();

  const tabs = [...document.querySelectorAll("[data-tab]")];
  function activateTab(tab, focus = false) {
    tabs.forEach((candidate) => {
      candidate.setAttribute("aria-selected", String(candidate === tab));
      candidate.tabIndex = candidate === tab ? 0 : -1;
    });
    document.querySelectorAll("[data-panel]").forEach((panel) => {
      panel.hidden = panel.dataset.panel !== tab.dataset.tab;
    });
    if (focus) tab.focus();
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activateTab(tab));
    tab.addEventListener("keydown", (event) => {
      const next =
        event.key === "ArrowRight"
          ? (index + 1) % tabs.length
          : event.key === "ArrowLeft"
            ? (index + tabs.length - 1) % tabs.length
            : event.key === "Home"
              ? 0
              : event.key === "End"
                ? tabs.length - 1
                : null;
      if (next !== null) {
        event.preventDefault();
        activateTab(tabs[next], true);
      }
    });
  });
  activateTab(tabs[0]);

  function renderParameters(mode) {
    const result = model.parameterCount(mode);
    document.getElementById("param-total").textContent = count(result.total);
    document.getElementById("param-bar").replaceChildren();
    document.getElementById("param-details").replaceChildren();
    for (const group of result.groups.filter((item) => item.value > 0).sort((a, b) => b.value - a.value)) {
      const segment = document.createElement("div");
      segment.className = "param-segment";
      segment.style.flexGrow = String(group.value / result.total);
      segment.style.background = group.color;
      segment.title = `${group.name}: ${count(group.value)} (${((100 * group.value) / result.total).toFixed(2)}%)`;
      document.getElementById("param-bar").appendChild(segment);
      const row = document.createElement("div");
      row.className = "param-row";
      row.innerHTML = `<span class="swatch" style="background:${group.color}"></span><span>${group.name}</span><span class="value">${count(
        group.value
      )}</span>`;
      document.getElementById("param-details").appendChild(row);
    }
    const explanations = {
      stored:
        "<strong>Full checkpoint:</strong> 177.393B backbone + vision weights and 2.607B unique MTP weights = 180.000B. The lexical table contributes 51.200B and the routed expert bank 120.796B. BF16 storage is about 335.28 GiB.",
      backbone:
        "<strong>Text backbone:</strong> full weights minus lexical table, vision, and MTP = 125.744B. This scope retains lexical projections, four-stream residual mixers, and input/output tables; it reconstructs the card's 125B backbone label.",
      active:
        "<strong>Active text path:</strong> 120.796B routed weights x 10/512 + 4.312B always-used text weights = 6.671B. This structural convention includes the 635.7M LM head and small normalization weights; it excludes full lookup tables, vision, and the speculative draft. Removing the LM head gives 6.036B. Per-token input/lexical row reads are 2560 + 2560 elements.",
    };
    document.getElementById("param-derivation").innerHTML = explanations[mode];
    document
      .querySelectorAll("[data-param-mode]")
      .forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.paramMode === mode)));
  }
  document
    .querySelectorAll("[data-param-mode]")
    .forEach((button) => button.addEventListener("click", () => renderParameters(button.dataset.paramMode)));
  renderParameters("stored");

  const contexts = [8192, 32768, 65536, 262144, 1000000];
  const cacheControls = ["context-slider", "cache-dtype", "state-dtype", "indexer-storage"].map((id) => document.getElementById(id));
  function renderCache() {
    const tokens = contexts[Number(cacheControls[0].value)];
    const storage = cacheControls[3].value;
    const result = model.cacheBytes(tokens, Number(cacheControls[1].value), Number(cacheControls[2].value), storage);
    document.getElementById("context-label").textContent = `${tokens.toLocaleString("en-US")} tokens${
      tokens > c.nativeContext ? " / RoPE extension" : ""
    }`;
    const scale = Math.max(result.total, result.baseline);
    for (const [id, value] of Object.entries(result)) {
      document.getElementById(`${id}-fill`).style.width = `${(100 * value) / scale}%`;
      document.getElementById(`${id}-value`).textContent = bytes(value);
    }
    document.getElementById("cache-note").textContent =
      storage === "reference"
        ? "Reference mode retains one 128-wide raw indexer key per token in each QSA layer. At the native 256K context with BF16 caches and FP32 GDN matrices, the structural total is 6.86 GiB per sequence."
        : "Pooled mode estimates one 128-wide key per complete four-token block, plus raw keys for the partial tail. This is an optimized representation estimate; the pinned Transformers reference caches raw token keys.";
  }
  cacheControls.forEach((control) => control.addEventListener("input", renderCache));
  renderCache();

  function renderSelection() {
    const visible = Number(document.getElementById("selection-query").value);
    const budget = Number(document.getElementById("selection-budget").value);
    const scores = Array.from({ length: 16 }, (_, block) => 1.4 + Math.sin((block + 1) * 1.71 + visible * 0.11));
    const selection = model.selectBlocks(visible, budget, scores);
    const selected = new Set(selection.selected),
      tail = new Set(selection.tail);
    const grid = document.getElementById("selection-grid");
    grid.replaceChildren();
    for (let index = 0; index < 64; index++) {
      const cell = document.createElement("span");
      const status = index >= visible ? "future" : tail.has(index) ? "tail" : selected.has(index) ? "selected" : "unselected";
      cell.className = `token-cell ${status}`;
      cell.textContent = String(index + 1);
      cell.title = `Token ${index + 1}: ${status === "tail" ? "included partial tail" : status}`;
      grid.appendChild(cell);
    }
    document.getElementById("selection-query-label").textContent = `${visible} tokens`;
    document.getElementById("selection-readout").textContent = `${selection.completeBlocks} complete blocks visible; selected blocks ${
      selection.blocks.length ? selection.blocks.map((block) => block + 1).join(", ") : "none"
    }. ${selection.blocks.length * 4} block tokens + ${selection.tail.length} tail tokens = ${
      selection.selected.length
    } tokens read. Future tokens are excluded.`;
  }
  ["selection-query", "selection-budget"].forEach((id) => document.getElementById(id).addEventListener("input", renderSelection));
  renderSelection();

  const root = document.documentElement;
  const themeButton = document.getElementById("theme-toggle");
  let savedTheme;
  try {
    savedTheme = localStorage.getItem("qwen-flash-next-study-theme");
  } catch {}
  if (savedTheme === "dark" || (!savedTheme && window.matchMedia("(prefers-color-scheme: dark)").matches)) root.classList.add("dark");
  const updateTheme = () => {
    const dark = root.classList.contains("dark");
    themeButton.textContent = dark ? "Light" : "Dark";
    themeButton.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
  };
  updateTheme();
  themeButton.addEventListener("click", () => {
    root.classList.toggle("dark");
    try {
      localStorage.setItem("qwen-flash-next-study-theme", root.classList.contains("dark") ? "dark" : "light");
    } catch {}
    updateTheme();
  });
  document.getElementById("print-page").addEventListener("click", () => window.print());
})();
