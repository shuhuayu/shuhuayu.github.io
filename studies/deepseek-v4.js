(() => {
  const M = window.DeepSeekV4;
  const $ = (id) => document.getElementById(id);
  const number = (value) => value.toLocaleString("en-US");
  const params = (value) =>
    value >= 1e12 ? `${(value / 1e12).toFixed(3)}T` : value >= 1e9 ? `${(value / 1e9).toFixed(2)}B` : `${(value / 1e6).toFixed(2)}M`;
  const bytes = (value) =>
    value >= 2 ** 30
      ? `${(value / 2 ** 30).toFixed(2)} GiB`
      : value >= 2 ** 20
        ? `${(value / 2 ** 20).toFixed(2)} MiB`
        : `${(value / 2 ** 10).toFixed(1)} KiB`;
  let model = "pro";
  let layerIndex = 2;
  let selectedNode = "attention";
  let operations = {};

  const root = document.documentElement;
  const themeKey = "kimi-k3-study-theme";
  let savedTheme = null;
  try {
    savedTheme = localStorage.getItem(themeKey);
  } catch {
    /* Storage may be disabled by browser privacy settings. */
  }
  root.classList.toggle("dark", savedTheme === "dark" || (!savedTheme && window.matchMedia("(prefers-color-scheme: dark)").matches));
  function themeLabel() {
    $("theme-toggle").textContent = root.classList.contains("dark") ? "Light" : "Dark";
  }
  $("theme-toggle").addEventListener("click", () => {
    root.classList.toggle("dark");
    try {
      localStorage.setItem(themeKey, root.classList.contains("dark") ? "dark" : "light");
    } catch {
      /* Keep the theme usable without persistent storage. */
    }
    themeLabel();
  });
  themeLabel();
  $("print-button").addEventListener("click", () => window.print());

  function renderModel() {
    const s = M.spec(model),
      c = M.counts(model);
    layerIndex = Math.min(layerIndex, s.numLayers - 1);
    document.querySelectorAll("[data-model]").forEach((button) => button.setAttribute("aria-pressed", button.dataset.model === model));
    $("model-readout").textContent = `${s.name} · hidden width ${number(s.dim)} · ${s.heads} query heads`;
    $("stat-total").textContent = s.reportedTotal;
    $("stat-active").textContent = s.reportedActive;
    $("stat-layers").textContent = s.numLayers;
    $("stat-attention").textContent = c.SWA ? `${c.CSA} + ${c.HCA} + ${c.SWA}` : `${c.CSA} + ${c.HCA}`;
    $("stat-attention-label").textContent = c.SWA ? "CSA + HCA + SWA-only" : "CSA + HCA layers";
    $("stat-experts").textContent = `6 / ${s.experts}`;
    $("embedding-shape").textContent = `T × ${number(s.dim)}`;
    $("stack-label").textContent = `${s.numLayers} blocks at width ${number(s.dim)}`;
    $("expert-ribbon").textContent = `${s.experts} routed experts, choose 6 + 1 shared · expert width ${number(s.expertDim)}`;
    $("csa-topk").textContent = `${number(s.topK)} in ${model === "pro" ? "Pro" : "Flash"}`;
    $("layer-map").innerHTML = M.layers(model)
      .map(
        (item) =>
          `<button type="button" class="layer-tile ${item.mode.toLowerCase()} ${item.hash ? "hash" : ""}" data-layer="${item.index}" aria-pressed="${
            item.index === layerIndex
          }" aria-label="Layer ${item.number}, ${item.mode}, ${item.hash ? "hash" : "score"}-routed MoE"><strong>${item.number}</strong><small>${
            item.mode
          }</small></button>`
      )
      .join("");
    $("kernel-layer").innerHTML = M.layers(model)
      .map((item) => `<option value="${item.index}">L${item.number} / ${item.mode}${item.hash ? " / hash MoE" : ""}</option>`)
      .join("");
    $("kernel-layer").value = layerIndex;
    renderLayer();
    renderParameters();
    renderCache();
  }

  function renderLayer() {
    const s = M.spec(model),
      item = M.layers(model)[layerIndex];
    $("kernel-layer").value = layerIndex;
    document.querySelectorAll("[data-layer]").forEach((button) => button.setAttribute("aria-pressed", Number(button.dataset.layer) === layerIndex));
    $("layer-readout").innerHTML = `<div><strong>Layer ${item.number} / ${item.mode}</strong><span>Release index ${item.index} · ${
      item.hash ? "hash" : "score"
    }-routed MoE</span></div><div><strong>${item.ratio ? `${item.ratio}:1 compressed history` : "Uncompressed local window only"}</strong><span>${
      item.mode === "CSA"
        ? `Lightning indexer selects up to ${number(s.topK)} entries`
        : item.mode === "HCA"
          ? "All completed compressed blocks, no indexer"
          : "First two Flash blocks have no global compressor"
    }</span></div><div><strong>4 streams × ${number(s.dim)} channels</strong><span>One D-wide input to attention and one to MoE</span></div>`;
    renderDiagram();
  }

  function operationData() {
    const s = M.spec(model),
      item = M.layers(model)[layerIndex],
      t = Number($("kernel-tokens").value);
    const d = s.dim,
      h = s.heads,
      k = s.headDim,
      q = s.queryRank,
      g = s.outputGroups,
      r = s.outputRank,
      f = s.expertDim;
    const td = `${number(t)} × ${number(d)}`,
      residual = `${number(t)} × 4 × ${number(d)}`,
      assignments = t * 6;
    const count = item.ratio ? Math.floor(t / item.ratio) : 0;
    const shapes = (...values) => values.map((value) => (typeof value === "number" ? number(value) : value)).join(" × ");
    const op = (title, shape, weight, n, description, backward, elements, kind = "gemm") => ({
      title,
      shape,
      weight,
      params: n,
      description,
      backward,
      elements,
      kind,
    });
    return {
      "hc-attn": op(
        "mHC pre + RMSNorm",
        td,
        shapes(24, 4 * d),
        24 * 4 * d,
        "Flatten the four residual streams, generate 24 mixing coefficients, apply sigmoid/Sinkhorn constraints, and contract to a single D-wide attention input.",
        "The expanded residual, normalization statistics, and learned mixing coefficients; a fused implementation can recompute small intermediates.",
        t * d
      ),
      "query-down": op(
        "Query down + RMSNorm",
        shapes(t, q),
        shapes(q, d),
        q * d,
        "The low-rank query latent is shared by main attention and the CSA indexer. It is normalized before the two up-projection branches.",
        "Input activation and normalized query latent; RMSNorm backward also needs its statistics.",
        t * q
      ),
      "query-up": op(
        "Query up + norm / RoPE",
        shapes(t, h, k),
        shapes(h * k, q),
        h * k * q,
        "Expand the latent to all query heads, normalize each head, and apply RoPE to the final 64 channels.",
        "Query latent plus head-normalization data. Attention backward consumes the final query tensor.",
        t * h * k
      ),
      "local-kv": op(
        "Local shared KV",
        shapes(t, k),
        shapes(k, d),
        k * d,
        "One 512-dimensional vector serves as both K and V for every query head. Normalize it, rotate the final 64 channels, and retain the latest 128 entries for decode.",
        "Layer input, normalization data, and KV consumed by attention. This is not two separate H-head caches.",
        t * k
      ),
      compressor: op(
        `${item.mode} KV compressor`,
        shapes(count, k),
        `2 projections: ${shapes(item.mode === "CSA" ? 2 * k : k, d)}`,
        (item.mode === "CSA" ? 4 : 2) * d * k,
        item.mode === "CSA"
          ? "KV and gate projections each have two halves. Per-channel softmax pools previous/current groups of four tokens, then RMSNorm and RoPE produce one completed entry per four tokens."
          : "One KV projection and one gate projection pool 128 non-overlapping tokens. RMSNorm and RoPE produce one completed entry per block.",
        "Pooling candidates, gate logits/softmax weights, position biases, and normalization statistics. Incomplete blocks require separate compressor state during decode.",
        count * k
      ),
      indexer: op(
        "Lightning indexer + top-K",
        `${shapes(t, Math.floor(t / 4))} scores → top-${Math.min(s.topK, Math.floor(t / 4))}`,
        `Q: ${shapes(64 * 128, q)}; gate: ${shapes(64, d)}; compressor: 2 × ${shapes(256, d)}`,
        q * 64 * 128 + d * 64 + 4 * d * 128,
        "A separate overlapping 4:1 compressor creates 128-dimensional index keys. Sixty-four FP4 query heads score the candidate pool; top-K positions select main KV. The full score tensor is a logical view, not a requirement to materialize it in a fused kernel.",
        "Top-K indices route main attention. Selection is discrete; it is not a differentiable dense attention weight over every candidate.",
        t * Math.floor(t / 4),
        "attention"
      ),
      attention: op(
        `${item.mode} shared-KV attention`,
        shapes(t, h, k),
        `${h} learned sink logits; no projection matrix`,
        h,
        `Combine a causal 128-token local window with ${
          item.mode === "CSA"
            ? `up to ${s.topK} selected compressed entries`
            : item.mode === "HCA"
              ? "every available 128:1 compressed entry"
              : "no compressed history"
        }. A learned sink changes the softmax denominator. Apply inverse RoPE to the last 64 output channels.`,
        "Queries, shared KV, output/statistics, and causal/selected positions. Memory-efficient attention avoids saving the full attention-probability matrix.",
        t * h * k,
        "attention"
      ),
      "output-group": op(
        "Grouped output Wₒᵃ",
        shapes(t, g, r),
        shapes(g, r, (h * k) / g),
        h * k * r,
        `${g} independent group projections map ${number(
          (h * k) / g
        )} channels each to rank ${r}. The physical stacked weight must still be interpreted as separate group matrices.`,
        "Attention outputs grouped by heads, and the grouped weights. The logical operation is batched GEMM.",
        t * g * r
      ),
      "output-final": op(
        "Final output Wₒᵇ",
        td,
        shapes(d, g * r),
        d * g * r,
        "Concatenate the group latents and project back to the D-wide sublayer output.",
        "Grouped output latents and output weight.",
        t * d
      ),
      "hc-post": op(
        "mHC attention update",
        residual,
        "Reuse precomputed post / residual coefficients",
        0,
        "Add the attention result to all four streams through C, while B mixes the previous four-stream residual.",
        "Original residual, sublayer output, and mixing coefficients. The following MoE receives a fresh mHC contraction.",
        t * 4 * d,
        "op"
      ),
      "hc-ffn": op(
        "mHC pre + FFN RMSNorm",
        td,
        shapes(24, 4 * d),
        24 * 4 * d,
        "A separate learned mHC transform contracts the post-attention residual to the input of MoE. Its weights are not shared with the attention mHC transform.",
        "Expanded residual, norm statistics and this sublayer's mixing coefficients.",
        t * d
      ),
      router: op(
        item.hash ? "Hash IDs + routing weights" : "Scores + top-6 routing",
        `${shapes(t, s.experts)} scores; ${shapes(t, 6)} IDs / weights`,
        shapes(s.experts, d),
        s.experts * d,
        item.hash
          ? "The first three layers look up six expert IDs from a fixed token-ID table. The learned projection still computes sqrt(softplus) scores used for those experts' normalized weights."
          : "Apply sqrt(softplus) to router logits. A balancing bias changes top-6 selection, while the original scores determine normalized expert weights.",
        "Selected IDs, routing weights, and input activations. Indices are discrete; gradients flow through gathered weights.",
        t * s.experts
      ),
      dispatch: op(
        "Expert dispatch",
        shapes(assignments, d),
        "Distributed permutation / transfer",
        0,
        `R = 6T = ${number(assignments)} assignments across the expert group. A participant receives its subset, not R copies on every GPU.`,
        "Expert IDs and the inverse permutation / communication schedule used for combine and backward.",
        assignments * d,
        "comm"
      ),
      "expert-up": op(
        "Routed W₁ / W₃",
        shapes(assignments, 2 * f),
        shapes(s.experts, 2 * f, d),
        2 * s.experts * f * d,
        "Independent expert gate/up GEMMs. The label counts all expert-bank weights; each token executes only its six selected experts.",
        "Dispatched input and gate/up activations; expert ownership determines each GPU's local slice.",
        assignments * 2 * f
      ),
      swiglu: op(
        "Clipped SwiGLU",
        shapes(assignments, f),
        "Elementwise SiLU × up",
        0,
        "Clamp the up branch to [-10, 10] and gate branch from above at 10, then multiply SiLU(gate) by up, as in the released reference.",
        "Gate/up preactivations and clipping masks, or sufficient inputs to recompute them.",
        assignments * f,
        "op"
      ),
      "expert-down": op(
        "Routed W₂",
        shapes(assignments, d),
        shapes(s.experts, d, f),
        s.experts * d * f,
        "Project each selected expert's intermediate back to D. Per-expert matrix boundaries remain distinct even in batched kernels.",
        "SwiGLU output, expert weights, and routing weights / assignment map.",
        assignments * d
      ),
      shared: op(
        "Always-on shared expert",
        td,
        `W₁ / W₂ / W₃: 3 × ${shapes(f, d)}`,
        3 * f * d,
        "A separate shared SwiGLU expert executes once per input token, independently of the routed top-6 selection.",
        "Input and shared expert intermediates; there is no expert dispatch for this branch.",
        t * d
      ),
      combine: op(
        "Weighted expert combine",
        td,
        "Inverse dispatch + shared output",
        0,
        "Restore token order, sum selected expert outputs with their routing weights, and add the shared expert output.",
        "Assignment map and routing weights, plus expert outputs when required for gradients to routing weights.",
        t * d,
        "comm"
      ),
      "hc-final": op(
        "mHC MoE update",
        residual,
        "Reuse FFN post / residual coefficients",
        0,
        "Reinject the MoE result into the four residual streams. This state is the input to the next decoder block.",
        "FFN input residual, output, and mHC coefficients.",
        t * 4 * d,
        "op"
      ),
    };
  }

  function renderDiagram() {
    const s = M.spec(model),
      item = M.layers(model)[layerIndex];
    operations = operationData();
    if ((selectedNode === "indexer" && item.mode !== "CSA") || (selectedNode === "compressor" && !item.ratio)) selectedNode = "attention";
    const lens = $("kernel-lens").value;
    const node = (id) => {
      const op = operations[id];
      return `<button type="button" class="kernel-node ${op.kind}" data-node="${id}" aria-pressed="${id === selectedNode}"><strong>${
        op.title
      }</strong><small class="mono">${
        lens === "params" ? (op.params ? `${params(op.params)} parameter elements` : "No matrix weights") : op.shape
      }</small></button>`;
    };
    const arrow = '<div class="kernel-arrow" aria-hidden="true">↓</div>';
    const branch = (ids) => ids.map(node).join(arrow);
    $("kernel-diagram").className = `kernel-diagram card ${item.mode.toLowerCase()}`;
    $("kernel-diagram").innerHTML = `<h3>${s.name} · Layer ${item.number}</h3><p class="note">${item.mode} + ${
      item.hash ? "hash-routed" : "score-routed"
    } MoE · output is four residual streams</p><div class="kernel-stage-title mhc">Contract the attention input</div>${node(
      "hc-attn"
    )}${arrow}<div class="kernel-branches"><div>${branch(["query-down", "query-up"])}</div><div>${node("local-kv")}</div><div>${
      item.ratio
        ? node("compressor") +
          (item.mode === "CSA" ? `<div class="kernel-junction">Separate index branch<br>Query latent + layer input</div>${node("indexer")}` : "")
        : '<div class="kernel-junction">SWA only<br>No compressor / indexer</div>'
    }</div></div><div class="kernel-junction">Queries + local${item.ratio ? " / compressed" : ""} KV feed one softmax${
      item.mode === "CSA" ? " · indexer supplies selected positions" : ""
    }</div>${branch([
      "attention",
      "output-group",
      "output-final",
      "hc-post",
    ])}<div class="kernel-stage-title moe">Contract, route, and compute expert outputs</div>${branch([
      "hc-ffn",
      "router",
    ])}${arrow}<div class="kernel-branches two moe"><div>${branch(["dispatch", "expert-up", "swiglu", "expert-down"])}</div><div>${node(
      "shared"
    )}<div class="kernel-junction">T tokens, not 6T assignments<br>Local always-on branch</div></div></div>${arrow}<div class="moe">${node(
      "combine"
    )}</div>${arrow}<div class="mhc">${node("hc-final")}</div>`;
    renderInspector();
  }

  function renderInspector() {
    const op = operations[selectedNode],
      checkpoint = $("kernel-storage").value === "checkpoint";
    const item = M.layers(model)[layerIndex];
    $("kernel-inspector").innerHTML = `<span class="label">Operation inspector</span><h3>${op.title}</h3><p>${
      op.description
    }</p><dl><dt>Logical output</dt><dd class="mono">${op.shape}</dd><dt>Weight view / operation</dt><dd class="mono">${
      op.weight
    }</dd><dt>Parameter elements</dt><dd>${
      op.params ? `${params(op.params)} (${number(op.params)})` : "No trainable matrix in this operation"
    }</dd><dt>BF16-equivalent output payload</dt><dd>${bytes(op.elements * 2)} · unsharded, illustrative</dd><dt>Backward dependencies</dt><dd>${
      op.backward
    }</dd><dt>Selected storage policy</dt><dd>${
      checkpoint
        ? "Full block checkpoint: retain the expanded block input, then replay internal operations for backward. Distributed dispatch/combine may also be replayed depending on the implementation."
        : "No checkpoint: autograd retains the tensors needed by backward. Fusion can save or recompute fewer intermediates than this conceptual graph."
    }</dd></dl><p class="note">Output payload is not a saved-activation total or a peak-memory measurement. Kernel dtypes, fusion, packing and sharding change actual storage.</p><p class="source-note"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py">Released model implementation</a> · ${
      item.mode
    } block</p>`;
    $("kernel-diagram")
      .querySelectorAll("[data-node]")
      .forEach((button) => button.setAttribute("aria-pressed", button.dataset.node === selectedNode));
  }

  function renderParameters() {
    const p = M.parameters(model);
    $("param-total").textContent = params(p.total);
    $("param-routed").textContent = `${(100 * p.routedFraction).toFixed(2)}%`;
    $("param-mtp").textContent = params(p.mtp);
    $("parameter-rows").innerHTML =
      p.rows
        .map(
          (row) =>
            `<tr><td>${row.label}</td><td class="mono">${row.formula}</td><td class="numeric" title="${number(row.value)}">${params(
              row.value
            )}</td><td class="numeric">${((100 * row.value) / p.total).toFixed(2)}%</td></tr>`
        )
        .join("") +
      `<tr class="param-total"><td>Decoder + embedding / head</td><td>MTP excluded</td><td class="numeric">${params(
        p.total
      )}</td><td class="numeric">100%</td></tr><tr><td>Additional MTP block</td><td>Shared embedding / vocabulary head not recounted</td><td class="numeric">${params(
        p.mtp
      )}</td><td class="numeric">—</td></tr>`;
  }

  function renderCache() {
    const tokens = 2 ** Number($("context-range").value);
    const requestInput = $("cache-requests");
    const requests = Math.min(128, Math.max(1, Math.round(Number(requestInput.value) || 1)));
    requestInput.value = requests;
    const c = M.cache(model, tokens, $("cache-precision").value, requests);
    $("context-value").textContent = `${number(tokens)} tokens`;
    $("context-range").setAttribute("aria-valuetext", `${number(tokens)} tokens`);
    for (const key of ["csa", "hca", "index", "local", "total"]) {
      $(`${key}-fill`).style.width = `${(100 * c[key]) / c.baseline}%`;
      $(`${key}-value`).textContent = bytes(c[key]);
    }
    $("baseline-value").textContent = bytes(c.baseline);
    $("cache-summary").textContent = `${(100 * c.fraction).toFixed(2)}% of GQA8 baseline · ${requests} request${
      requests === 1 ? "" : "s"
    } · payload only, excluding scales / workspace.`;
    $("attention-slots").textContent = `Last-query main-attention positions: up to ${number(c.csaSlots)} in CSA and ${number(
      c.hcaSlots
    )} in HCA (including the local window). This is per head, not cache size.`;
  }

  function renderNs() {
    const initial = 10 ** Number($("ns-initial").value),
      values = M.singularValues(initial);
    const x = (index) => 40 + index * 49,
      y = (value) => 213 - (value / 1.6) * 190;
    const points = (start, end) =>
      values
        .slice(start, end + 1)
        .map((value, i) => `${x(start + i)},${y(value)}`)
        .join(" ");
    $("ns-initial-value").textContent = initial < 0.001 ? initial.toExponential(2) : initial.toFixed(4);
    $("ns-initial").setAttribute("aria-valuetext", `Initial singular value ${initial.toPrecision(3)}`);
    $("ns-plot").innerHTML = `<title>Hybrid Newton-Schulz from singular value ${initial.toPrecision(3)}</title>${[0, 0.5, 1, 1.5]
      .map(
        (value) =>
          `<line x1="40" x2="530" y1="${y(value)}" y2="${y(value)}" stroke="var(--line)" ${
            value === 1 ? 'stroke-dasharray="4 4"' : ""
          }/><text x="5" y="${y(value) + 4}">${value}</text>`
      )
      .join("")}<polyline points="${points(0, 8)}" stroke="var(--csa)" fill="none" stroke-width="2.5"/><polyline points="${points(
      8,
      10
    )}" stroke="var(--gold)" fill="none" stroke-width="3"/>${values
      .map(
        (value, i) =>
          `<circle cx="${x(i)}" cy="${y(value)}" r="${i === 5 ? 5 : 3}" fill="${
            i === 5 ? "var(--swa)" : i > 8 ? "var(--gold)" : "var(--csa)"
          }"/><text x="${x(i)}" y="234" text-anchor="middle">${i}</text>`
      )
      .join("")}<text x="285" y="249" text-anchor="middle">NS iteration (inside one optimizer update)</text>`;
    $("ns-result").innerHTML = `<span style="color: var(--swa)">●</span> Fixed five-step Muon ends at s₅ = ${values[5].toFixed(
      4
    )}.<br><span style="color: var(--gold)">●</span> V4's 8+2 schedule ends at s₁₀ = ${values[10].toFixed(6)}.`;
  }

  document.querySelectorAll("[data-model]").forEach((button) =>
    button.addEventListener("click", () => {
      model = button.dataset.model;
      renderModel();
    })
  );
  $("layer-map").addEventListener("click", (event) => {
    const button = event.target.closest("[data-layer]");
    if (button) {
      layerIndex = Number(button.dataset.layer);
      renderLayer();
    }
  });
  $("kernel-layer").addEventListener("change", () => {
    layerIndex = Number($("kernel-layer").value);
    renderLayer();
  });
  $("kernel-diagram").addEventListener("click", (event) => {
    const button = event.target.closest("[data-node]");
    if (button) {
      selectedNode = button.dataset.node;
      renderInspector();
    }
  });
  for (const id of ["kernel-tokens", "kernel-lens"]) $(id).addEventListener("change", renderDiagram);
  $("kernel-storage").addEventListener("change", renderInspector);
  $("context-range").addEventListener("input", renderCache);
  $("cache-precision").addEventListener("change", renderCache);
  $("cache-requests").addEventListener("change", renderCache);
  $("ns-initial").addEventListener("input", renderNs);
  $("ns-schedule").innerHTML = M.nsSchedule
    .map((_, i) => `<div class="ns-step ${i >= 8 ? "stable" : ""}"><strong>${i + 1}</strong>${i >= 8 ? "stable" : "fast"}</div>`)
    .join("");
  renderModel();
  renderNs();
})();
