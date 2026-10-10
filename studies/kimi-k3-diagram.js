(() => {
  const controls = Object.fromEntries(["layer", "lens", "storage", "tokens", "dtype"].map((key) => [key, document.getElementById(`kernel-${key}`)]));
  const diagram = document.getElementById("kernel-diagram");
  const inspector = document.getElementById("kernel-inspector");
  const memoryNote = document.getElementById("kernel-memory-note");
  const D = 7168;
  const H = 96;
  const K = 128;
  const Z = 3584;
  const F = 3072;
  const E = 896;
  const topK = 16;
  const C = H * K;
  let selected = "qkv";
  let nodes = new Map();

  const count = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(3)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n.toLocaleString());
  const shape = (dims) => `[${dims.join(", ")}]`;
  const bytes = (n) => (n >= 2 ** 30 ? `${(n / 2 ** 30).toFixed(2)} GiB` : `${(n / 2 ** 20).toFixed(2)} MiB`);
  const elements = (dims) =>
    dims.reduce((total, dim) => total * (dim === "T" ? Number(controls.tokens.value) : dim === "R" ? topK * Number(controls.tokens.value) : dim), 1);
  const node = (id, title, dims, options = {}) => ({ id, title, dims, kind: "op", params: 0, weights: "No weight matrix", save: true, ...options });
  const matmul = (id, title, dims, inputWidth, outputWidth, options = {}) =>
    node(id, title, dims, {
      kind: "gemm",
      params: inputWidth * outputWidth,
      weights: shape([outputWidth, inputWidth]),
      ...options,
    });

  function attentionNodes(isMla) {
    if (isMla) {
      return {
        branches: [
          {
            label: "Query",
            nodes: [
              matmul("qa", "Q compression", ["T", 1536], D, 1536, {
                code: "attention.wq_a",
                info: "Compress the model stream to the query latent. Backward computes both an input gradient and a weight gradient.",
                backward: "Input x and the projection weights. The output is needed by query RMSNorm.",
              }),
              node("qn", "Query RMSNorm", ["T", 1536], {
                params: 1536,
                weights: "[1536] scale",
                code: "attention.q_norm",
                backward: "Query latent and RMS statistics; normalization arithmetic may use FP32.",
              }),
              matmul("qb", "Q expansion", ["T", H, 192], 1536, H * 192, {
                code: "attention.wq_b",
                info: "Each of 96 query heads has 192 channels: 128 ordinary channels plus a 64-channel slice. K3 applies no rotary transform to the text path.",
                backward: "Normalized query latent and weights; Q is needed by attention backward.",
              }),
            ],
          },
          {
            label: "Key / value",
            nodes: [
              matmul("kva", "KV compression", ["T", 576], D, 576, {
                code: "attention.wkv_a",
                info: "Split the output into a 512-channel KV latent and a shared 64-channel key slice.",
                backward: "Input x and weights. The latent feeds RMSNorm; the shared key slice bypasses expansion.",
              }),
              node("kvn", "KV RMSNorm", ["T", 512], {
                params: 512,
                weights: "[512] scale",
                code: "attention.kv_norm",
                backward: "KV latent and RMS statistics.",
              }),
              matmul("kvb", "K / V expansion", ["T", H, 256], 512, H * 256, {
                code: "attention.wkv_b",
                info: "Split each head into 128 key channels and 128 value channels. Append the shared 64-channel key slice to form K [T, 96, 192].",
                backward: "Normalized KV latent and weights. Attention backward uses the reconstructed K and V.",
              }),
            ],
          },
          {
            label: "Output modulation",
            nodes: [
              matmul("gate", "Output gate", ["T", H, K], D, C, {
                code: "attention.gate",
                info: "A full projection of the original input supplies the sigmoid gate applied after attention.",
                backward: "Input x, gate logits or sigmoid values, and the ungated attention output.",
              }),
            ],
          },
        ],
        tail: [
          node("mix", "Causal variable-length attention", ["T", H, K], {
            kind: "attention",
            save: true,
            code: "attention.inner_attention",
            info: "Attend independently within each document. FlashAttention implementations compute tiled attention without storing a full T by T probability matrix.",
            backward: "Q, K, V, output, and per-row normalization statistics; exact tensors depend on the attention backend.",
            extra: "The 64-channel shared K slice is expanded over heads. This is layout and concatenation work, with no extra projection matrix.",
          }),
          node("gated", "Sigmoid gate x attention output", ["T", C], {
            code: "attention.forward: output gating",
            backward: "Ungated output and gate values; the product is the output-projection input.",
          }),
          matmul("out", "Output projection", ["T", D], C, D, {
            code: "attention.wo",
            save: false,
            backward: "Gated input and output-projection weights.",
          }),
        ],
        caption: "24 layers; six linear projections including the output gate.",
      };
    }
    return {
      branches: [
        {
          label: "Query / key / value",
          nodes: [
            matmul("qkv", "Q + K + V projections", ["T", 3 * C], D, 3 * C, {
              params: 3 * D * C,
              weights: `3 x ${shape([C, D])}`,
              code: "delta_attention.{q_proj,k_proj,v_proj}",
              info: "Three independent projections of the normalized input. Each output splits into 96 heads of width 128.",
              backward: "Input x and Q/K/V weights; projection outputs feed the short convolution.",
              extra: "This box groups three GEMMs. TorchTitan stores three weight tensors; horizontal fusion is a separate implementation choice.",
            }),
            node("conv", "Width-4 causal convolution + SiLU", ["T", 3 * C], {
              params: 3 * C * 4,
              weights: `3 x ${shape([C, 1, 4])}`,
              code: "delta_attention.inner_kda.short_convolution",
              info: "A depthwise convolution mixes the previous three tokens and the current token. Sequence metadata resets convolution history at document boundaries.",
              backward: "Convolution inputs, weights, and activation information.",
            }),
            node("qknorm", "Split heads + Q/K L2Norm", ["T", 3, H, K], {
              code: "delta_attention.inner_kda.kda_core",
              backward: "Q/K and normalization factors. V does not receive L2 normalization.",
            }),
          ],
        },
        {
          label: "Channel forgetting",
          nodes: [
            matmul("fa", "Forget compression", ["T", K], D, K, {
              code: "delta_attention.forget_a",
              backward: "Input x and weights; output feeds forget expansion.",
            }),
            matmul("fb", "Forget expansion", ["T", H, K], K, C, {
              code: "delta_attention.forget_b",
              backward: "128-channel latent and weights; logits feed the bounded decay transform.",
            }),
            node("decay", "Bounded log-decay", ["T", H, K], {
              params: H + H * K,
              weights: "A_log [96], dt_bias [96, 128]",
              fixedBytes: 4,
              code: "delta_attention.{A_log,dt_bias}",
              info: "Compute a bounded log-decay between -5 and 0 for every token, head, and key channel. The gate arithmetic uses FP32.",
              backward: "Gate logits, decay parameters, and transformed gate values.",
            }),
          ],
        },
        {
          label: "Write strength + output gate",
          nodes: [
            matmul("beta", "Write-strength projection", ["T", H], D, H, {
              code: "delta_attention.beta",
              info: "Sigmoid produces one delta-rule write strength per token and head.",
              backward: "Input x, weights, and sigmoid values.",
            }),
            matmul("gate", "Output-gate projection", ["T", H, K], D, C, {
              code: "delta_attention.output_gate",
              info: "This full-rank gate is applied after the KDA state update and per-head normalization.",
              backward: "Input x, output-gate weights, and sigmoid gate values.",
            }),
          ],
        },
      ],
      tail: [
        node("mix", "Chunk KDA", ["T", H, K], {
          kind: "attention",
          code: "delta_attention.inner_kda.kernel",
          info: "Forget part of the old state, correct it toward the new key/value pair, then read it with the query. Inference carries a fixed [96, 128, 128] state per document.",
          backward: "Normalized Q/K/V, gate and beta information, plus chunk state and kernel-specific intermediates.",
          extra: "A fixed inference state does not imply constant training activation memory. Training also needs token and chunk intermediates.",
        }),
        node("gated", "Per-head RMSNorm + sigmoid gate", ["T", C], {
          params: K,
          weights: "[128] shared per-head scale",
          code: "delta_attention.output_norm",
          backward: "KDA output, RMS statistics, gate values, and scale weights.",
        }),
        matmul("out", "Output projection", ["T", D], C, D, {
          code: "delta_attention.output_proj",
          save: false,
          backward: "Normalized, gated input and output-projection weights.",
        }),
      ],
      caption: "69 layers; eight linear projections plus depthwise convolution.",
    };
  }

  function ffnNodes(isDense) {
    if (isDense) {
      const hidden = 33792;
      return {
        branches: [
          {
            label: "Dense SiTU FFN",
            nodes: [
              matmul("densegu", "Gate + up projections", ["T", 2, hidden], D, 2 * hidden, {
                weights: `2 x ${shape([hidden, D])}`,
                code: "feed_forward.{w1,w3}",
                info: "The first decoder layer uses a much wider dense FFN. Every token executes both matrices.",
                backward: "Normalized FFN input and both matrices; gate/up outputs feed SiTU-GLU.",
              }),
              node("denseact", "SiTU-GLU", ["T", hidden], {
                code: "feed_forward.activation_fn",
                info: "Bound the gate branch with 4*tanh(g/4)*sigmoid(g), bound the up branch with 25*tanh(u/25), then multiply. The native activation computes intermediate math in FP32 and returns the input dtype.",
                backward: "Gate and up values. Compiling the pointwise region can fuse temporary arrays.",
              }),
              matmul("densedown", "Down projection", ["T", D], hidden, D, {
                code: "feed_forward.w2",
                save: false,
                backward: "SiTU product and weights.",
              }),
            ],
          },
        ],
        routed: false,
      };
    }
    return {
      branches: [
        {
          label: "Routing decisions",
          nodes: [
            matmul("router", "Router projection + sigmoid", ["T", E], D, E, {
              fixedBytes: 4,
              code: "moe.router.gate",
              info: "Routing reads the full 7168-channel stream. Quantile balancing adjusts selection; the original scores supply the normalized mixture weights.",
              backward: "Router input, scores, and selected routing weights. Expert indices are discrete.",
            }),
            node("topk", "Top-16 + score normalization", ["T", topK], {
              fixedBytes: 4,
              code: "moe.router",
              info: "Choose 16 of 896 routed experts. Selected IDs have shape [T, 16]; routing maps may also have shape [T, 896].",
              backward: "Selected scores/indices and normalization metadata.",
              extra:
                "The systems experiments also support forced balanced routing. It changes token placement, while preserving the expert shapes shown here.",
            }),
          ],
        },
        {
          label: "Latent routed experts",
          nodes: [
            matmul("latentdown", "Latent down projection", ["T", Z], D, Z, {
              code: "moe.routed_down",
              backward: "Normalized FFN input and projection weights.",
            }),
            node("dispatch", "Dispatch to selected experts", ["R", Z], {
              kind: "comm",
              code: "moe.routed_experts: dispatch",
              info: "Duplicate each token for its 16 assignments, reorder by expert, and send to the owning EP rank. R here describes the whole group; padding and load imbalance affect each rank.",
              backward: "Routing indices/counts and the reverse token mapping.",
            }),
            matmul("expertgu", "Expert gate + up", ["R", 2, F], Z, 2 * F, {
              params: E * 2 * Z * F,
              weights: `2 x ${shape([E, F, Z])}`,
              code: "routed_experts.{w1_EFD,w3_EFD}",
              info: "Each token assignment runs the selected expert's two projections. Native grouped GEMMs and Dist-MoE kernels implement this same logical operation.",
              backward: "Dispatched latent inputs and expert matrices; the gate/up outputs feed SiTU-GLU.",
              extra:
                "The count covers all 896 expert weights. A token evaluates 16 experts. BF16, MXFP8 training, and fake-quantized QAT are different execution policies for these matrices.",
            }),
            node("expertact", "Expert SiTU-GLU", ["R", F], {
              code: "routed_experts.activation_fn",
              info: "The bounded activation uses the same beta=4 and linear_beta=25 as the dense and shared FFNs.",
              backward: "Expert gate/up values. Full recomputation can regenerate them during backward.",
            }),
            matmul("expertdown", "Expert down projection", ["R", Z], F, Z, {
              params: E * F * Z,
              weights: shape([E, Z, F]),
              code: "routed_experts.w2_EDF",
              backward: "Expert SiTU activations and expert down weights.",
            }),
            node("combine", "Return + weighted scatter-add", ["T", Z], {
              kind: "comm",
              code: "moe.routed_experts: combine",
              info: "Return expert outputs, apply routing scores, and sum each token's 16 contributions. Score placement can vary between backends.",
              backward: "Expert outputs, routing weights, and the reverse mapping.",
            }),
            node("latentnorm", "Routed RMSNorm", ["T", Z], {
              params: Z,
              weights: "[3584] scale",
              code: "moe.routed_norm",
              backward: "Combined routed output and RMS statistics.",
            }),
            matmul("latentup", "Latent up projection", ["T", D], Z, D, {
              code: "moe.routed_up",
              save: false,
              backward: "Normalized routed output and up weights.",
            }),
          ],
        },
        {
          label: "Shared experts",
          nodes: [
            matmul("sharedgu", "Shared gate + up", ["T", 2, 2 * F], D, 4 * F, {
              weights: `2 x ${shape([2 * F, D])}`,
              code: "moe.shared_experts.{w1,w3}",
              info: "Two always-active shared experts are represented as one FFN of hidden width 6144. This branch retains the full model width.",
              backward: "FFN input and both shared weight matrices.",
            }),
            node("sharedact", "Shared SiTU-GLU", ["T", 2 * F], { code: "moe.shared_experts.activation_fn", backward: "Shared gate and up values." }),
            matmul("shareddown", "Shared down projection", ["T", D], 2 * F, D, {
              code: "moe.shared_experts.w2",
              save: false,
              backward: "Shared SiTU product and down weights.",
            }),
          ],
        },
      ],
      routed: true,
    };
  }

  function button(n) {
    nodes.set(n.id, n);
    const element = document.createElement("button");
    element.type = "button";
    element.className = `kernel-node ${n.kind}${selected === n.id ? " selected" : ""}`;
    element.dataset.node = n.id;
    element.setAttribute("aria-pressed", String(selected === n.id));
    const title = document.createElement("strong");
    title.textContent = n.title;
    const label = document.createElement("small");
    label.className = "kernel-shape";
    label.textContent = controls.lens.value === "params" ? (n.params ? `${count(n.params)} parameters` : "No weight parameters") : shape(n.dims);
    element.append(title, label);
    const checkpoint = controls.storage.value === "checkpoint";
    if (n.save) {
      const saved = document.createElement("span");
      saved.className = "kernel-saved";
      saved.textContent = n.keep ? "Retained checkpoint input" : checkpoint ? "Regenerated during backward" : "Backward needs output";
      element.appendChild(saved);
    }
    element.addEventListener("click", () => {
      selected = n.id;
      diagram.querySelectorAll("[data-node]").forEach((b) => {
        b.classList.toggle("selected", b.dataset.node === selected);
        b.setAttribute("aria-pressed", String(b.dataset.node === selected));
      });
      inspect(n);
    });
    return element;
  }

  function arrow() {
    const element = document.createElement("div");
    element.className = "kernel-arrow";
    element.setAttribute("aria-hidden", "true");
    element.textContent = "\u2193";
    return element;
  }

  function chain(parent, items) {
    items.forEach((n, i) => {
      if (i) parent.appendChild(arrow());
      parent.appendChild(button(n));
    });
  }

  function branches(parent, groups) {
    const wrapper = document.createElement("div");
    wrapper.className = `kernel-branches${groups.length === 2 ? " two" : ""}`;
    if (groups.length === 1) wrapper.style.gridTemplateColumns = "1fr";
    groups.forEach((group) => {
      const lane = document.createElement("div");
      const label = document.createElement("div");
      label.className = "kernel-branch-label";
      label.textContent = group.label;
      lane.appendChild(label);
      chain(lane, group.nodes);
      wrapper.appendChild(lane);
    });
    parent.appendChild(wrapper);
  }

  function inspect(n) {
    const checkpoint = controls.storage.value === "checkpoint";
    const memory = bytes(elements(n.dims) * (n.fixedBytes || Number(controls.dtype.value)));
    inspector.innerHTML = `<div class="eyebrow">Selected operation</div><h3></h3><p class="kernel-description"></p><dl><dt>Output shape</dt><dd class="inspect-shape"></dd><dt>Weight tensor(s)</dt><dd class="inspect-weight"></dd><dt>Parameters</dt><dd class="inspect-params"></dd><dt>Implementation</dt><dd class="inspect-code"></dd></dl><p class="kernel-memory"></p><p class="inspect-backward"></p><p class="kernel-fusion"></p>`;
    inspector.querySelector("h3").textContent = n.title;
    inspector.querySelector(".kernel-description").textContent = n.info || "Follow the arrows to the operations that consume this tensor.";
    inspector.querySelector(".inspect-shape").textContent = shape(n.dims);
    inspector.querySelector(".inspect-weight").textContent = n.weights;
    inspector.querySelector(".inspect-params").textContent = n.params ? `${n.params.toLocaleString()} (${count(n.params)})` : "0";
    inspector.querySelector(".inspect-code").textContent = n.code;
    inspector.querySelector(".kernel-memory").textContent = `${memory} of logical output values at T=${Number(
      controls.tokens.value
    ).toLocaleString()}. ${
      n.fixedBytes === 4 ? "This tensor uses FP32 in the diagram." : "Uses the selected activation dtype."
    } This excludes weights, gradients, other outputs, and kernel workspaces.`;
    inspector.querySelector(".inspect-backward").textContent = `Backward dependencies: ${n.backward || "Upstream gradient."} ${
      checkpoint && !n.keep ? "A full layer checkpoint regenerates internal activations on the backward pass." : ""
    }`;
    inspector.querySelector(".kernel-fusion").textContent =
      n.extra || "Shapes describe the full logical model. TP, CP, EP, and FSDP change each GPU's tensor and weight ownership.";
  }

  function render() {
    const isMla = controls.layer.value === "mla-moe";
    const isDense = controls.layer.value === "kda-dense";
    const attention = attentionNodes(isMla);
    const ffn = ffnNodes(isDense);
    nodes = new Map();
    diagram.replaceChildren();
    diagram.style.setProperty("--accent", `var(--${isMla ? "mla" : "kda"})`);
    diagram.style.setProperty("--accent-soft", `var(--${isMla ? "mla" : "kda"}-soft)`);
    const heading = document.createElement("h3");
    heading.textContent = controls.layer.selectedOptions[0].textContent;
    const caption = document.createElement("div");
    caption.className = "kernel-block-caption";
    caption.textContent = attention.caption;
    diagram.append(heading, caption);
    const input = document.createElement("div");
    input.className = "kernel-single";
    chain(input, [
      node("input", "Block input + residual bank", ["T", D], {
        keep: true,
        code: "KimiK3TransformerBlock.forward",
        info: "The layer receives the current partial block and a list of completed residual tensors. Each residual entry has shape [T, 7168].",
        backward: "The checkpoint boundary retains layer inputs. Residual-bank entries also participate in gradient propagation.",
      }),
      node("attnnorm", isDense ? "Attention RMSNorm" : "AttnRes mix + attention RMSNorm", ["T", D], {
        code: "attention_res_proj / attention_norm",
        info: "Attention Residuals computes a softmax over depth independently for each token, then forms a weighted mixture of residual entries. Layer 1 bypasses the attention-residual mix.",
        backward: "Residual entries, depth weights, and normalization information.",
        extra:
          "The residual-list representation avoids repeatedly copying a growing stack. A compiled reduction can consume individual entries directly.",
      }),
    ]);
    diagram.appendChild(input);
    const stage = document.createElement("div");
    stage.className = "kernel-stage";
    stage.innerHTML = `<div class="kernel-stage-title ${isMla ? "mla" : "kda"}">${isMla ? "Gated MLA" : "Kimi Delta Attention"}</div>`;
    branches(stage, attention.branches);
    const junction = document.createElement("div");
    junction.className = "kernel-junction";
    junction.textContent = isMla
      ? "Q + reconstructed K/V feed attention; the output gate joins afterward."
      : "Q/K/V + bounded decay + write strength feed KDA; the output gate joins afterward.";
    stage.appendChild(junction);
    const tail = document.createElement("div");
    tail.className = "kernel-single";
    chain(tail, attention.tail);
    tail.appendChild(arrow());
    tail.appendChild(
      button(
        node("ffnnorm", "Update partial residual + FFN AttnRes + RMSNorm", ["T", D], {
          code: "ffn_res_proj / ffn_norm",
          backward: "Attention output, residual entries, depth weights, and RMS statistics.",
        })
      )
    );
    stage.appendChild(tail);
    diagram.appendChild(stage);
    const channelStage = document.createElement("div");
    channelStage.className = "kernel-stage";
    channelStage.innerHTML = `<div class="kernel-stage-title moe">${isDense ? "Dense FFN" : "Stable LatentMoE"}</div>`;
    branches(channelStage, ffn.branches);
    const finish = document.createElement("div");
    finish.className = "kernel-single";
    finish.appendChild(arrow());
    finish.appendChild(
      button(
        node("finish", isDense ? "Update partial residual" : "Add routed + shared outputs; update partial residual", ["T", D], {
          save: false,
          code: "KimiK3TransformerBlock.forward: return",
          info: "The output continues as the current partial block. Completed blocks remain available to later Attention Residuals mixes.",
          backward: "Gradients flow through both the FFN result and the residual path.",
        })
      )
    );
    channelStage.appendChild(finish);
    diagram.appendChild(channelStage);
    const attnParams = attention.branches
      .flatMap((g) => g.nodes)
      .concat(attention.tail)
      .reduce((sum, n) => sum + n.params, 0);
    const ffnParams = ffn.branches.flatMap((g) => g.nodes).reduce((sum, n) => sum + n.params, 0);
    const tally = document.createElement("div");
    tally.className = "kernel-tally";
    tally.innerHTML = `<span>Attention parameters<strong>${count(attnParams)}</strong></span><span>${
      isDense ? "Dense FFN" : "LatentMoE"
    } parameters<strong>${count(ffnParams)}</strong></span>`;
    if (ffn.routed) tally.innerHTML += `<span>One routed expert<strong>${count(3 * Z * F)}</strong></span>`;
    diagram.appendChild(tally);
    if (!nodes.has(selected)) selected = isMla ? "qa" : "qkv";
    diagram.querySelectorAll("[data-node]").forEach((b) => {
      b.classList.toggle("selected", b.dataset.node === selected);
      b.setAttribute("aria-pressed", String(b.dataset.node === selected));
    });
    inspect(nodes.get(selected));
    memoryNote.textContent =
      controls.storage.value === "checkpoint"
        ? "Full layer checkpoint: retain boundary inputs and regenerate internal tensors during backward. The diagram illustrates this dependency pattern; real RegionAC policies can retain selected operations. CUDA graphs replay operations and do not themselves eliminate these dependencies."
        : "Without checkpointing: gold labels mark outputs consumed by backward. They are dependency candidates, not an additive peak-memory estimate. Kernels may save, alias, or recompute different tensors; overlapping lifetimes determine the measured peak.";
  }

  Object.values(controls).forEach((control) => control.addEventListener("change", render));
  render();
})();
