(() => {
  "use strict";
  const M = window.QwenFlashNextModel;
  const byId = (id) => document.getElementById(id);
  const integer = (value) => value.toLocaleString("en-US");
  const escape = (value) => String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const bytes = (value) =>
    value >= 2 ** 30
      ? `${(value / 2 ** 30).toFixed(2)} GiB`
      : value >= 2 ** 20
        ? `${(value / 2 ** 20).toFixed(2)} MiB`
        : value >= 1024
          ? `${(value / 1024).toFixed(2)} KiB`
          : `${value} B`;
  const fixed = (value) => (Math.abs(value) < 0.0005 ? 0 : value).toFixed(3);
  const vector = (values) => `[${values.map(fixed).join(", ")}]`;
  const code = (text) => `<pre><code>${escape(text)}</code></pre>`;
  const shapeCard = (title, shape, description) =>
    `<div class="shape-card"><strong>${title}</strong><code>${shape}</code><p>${description}</p></div>`;

  function matrix(title, values, logicalShape, rowNames, column = "d") {
    const rows = Array.isArray(values[0]) ? values : [values];
    const columns = rows[0].map((_, index) => `<th scope="col">${column}${index + 1}</th>`).join("");
    return `<div class="numeric-card"><strong>${title}</strong><code>${logicalShape}</code><table class="numeric-matrix" aria-label="${title}"><thead><tr><th scope="col">${
      rows.length === 4 ? "Stream" : "Row"
    }</th>${columns}</tr></thead><tbody>${rows
      .map(
        (row, index) =>
          `<tr${rows.length === 4 ? ` data-stream="${index}"` : ""}><th scope="row">${
            rowNames?.[index] ?? (rows.length === 4 ? `S${index + 1}` : rows.length === 1 ? "t" : `k${index + 1}`)
          }</th>${row.map((value) => `<td>${fixed(value)}</td>`).join("")}</tr>`
      )
      .join("")}</tbody></table></div>`;
  }

  function stepper(prefix, total, render) {
    let selected = 0;
    const buttons = [...document.querySelectorAll(`[data-${prefix}-step]`)];
    function refresh() {
      buttons.forEach((button) => button.setAttribute("aria-pressed", String(Number(button.getAttribute(`data-${prefix}-step`)) === selected)));
      byId(`${prefix}-content`).dataset.step = String(selected);
      if (byId(`${prefix}-progress`)) byId(`${prefix}-progress`).textContent = `Step ${selected + 1} of ${total}`;
      if (byId(`${prefix}-prev`)) byId(`${prefix}-prev`).disabled = selected === 0;
      if (byId(`${prefix}-next`)) byId(`${prefix}-next`).disabled = selected === total - 1;
      render(selected);
    }
    buttons.forEach((button) =>
      button.addEventListener("click", () => {
        selected = Number(button.getAttribute(`data-${prefix}-step`));
        refresh();
      })
    );
    for (const [suffix, direction] of [
      ["prev", -1],
      ["next", 1],
    ]) {
      if (byId(`${prefix}-${suffix}`))
        byId(`${prefix}-${suffix}`).addEventListener("click", () => {
          selected = Math.max(0, Math.min(total - 1, selected + direction));
          refresh();
        });
    }
    refresh();
    return { refresh };
  }

  function renderInput() {
    const mode = byId("input-mode").value;
    const textTokens = Number(byId("input-text-tokens").value);
    const hasVision = mode !== "text";
    const frames = mode === "video" ? Number(byId("input-frames").value) : 2;
    const visual = hasVision ? M.visualTokens(Number(byId("input-resolution").value), frames) : { patches: 0, merged: 0 };
    const tokens = textTokens + visual.merged;
    byId("input-resolution-control").hidden = !hasVision;
    byId("input-frames-control").hidden = mode !== "video";
    byId("visual-path").hidden = !hasVision;
    byId("input-token-strip").innerHTML = `<span class="token-pill"><strong>${integer(
      textTokens
    )} nonvisual token IDs</strong><small>Text and nonvisual special tokens</small></span>${
      hasVision
        ? `<span class="token-pill visual"><strong>${integer(
            visual.merged
          )} visual placeholder IDs</strong><small>One placeholder per merged visual vector</small></span>`
        : ""
    }<span aria-hidden="true">&rarr;</span><span class="token-pill"><strong>T = ${integer(
      tokens
    )}</strong><small>All decoder positions</small></span>`;
    byId("input-id-shape").textContent = `[${integer(tokens)}]`;
    byId("input-embedding-shape").textContent = `[${integer(tokens)}, 2560]`;
    byId("input-vision-shape").textContent = hasVision ? `[${integer(visual.merged)}, 2560]` : "No visual positions";
    byId("input-stream-shape").textContent = `[${integer(tokens)}, 4, 2560]`;
    byId("input-vision-description").textContent = hasVision
      ? "Replace matching placeholder embeddings with merged visual vectors. Original token IDs remain available separately."
      : "Text embeddings go directly to the four-stream copy.";
    for (const id of ["input-patch-shape", "input-encoder-shape"]) byId(id).textContent = `[${integer(visual.patches)}, 1152]`;
    byId("input-merge-shape").textContent = `[${integer(visual.merged)}, 2560]`;
    byId("input-accounting").textContent = `${
      hasVision
        ? `${visual.spatial} x ${visual.spatial} spatial patches x ${visual.temporal} temporal groups = ${integer(
            visual.patches
          )} vision positions; 2 x 2 spatial merging gives ${integer(visual.merged)} decoder positions. `
        : ""
    }At BF16, the logical [T,2560] vectors occupy ${bytes(tokens * 2560 * 2)}; one [T,4,2560] tensor occupies ${bytes(tokens * 4 * 2560 * 2)}.`;
  }
  for (const id of ["input-mode", "input-text-tokens", "input-resolution", "input-frames"]) byId(id).addEventListener("input", renderInput);
  renderInput();

  function residualBodies(e) {
    const weighted = e.normalized.map((row, stream) => row.map((value, channel) => value * e.readGates[stream][channel]));
    const updates = e.writeGates.map((gate) => e.update.map((value) => gate * value));
    return [
      `<h4>1. Carry four vectors for the same token</h4><p>Each row is one residual stream. Initially the four rows are copies; later layers have already written different updates. Flattening the stream and channel axes changes the layout, not the values.</p><div class="step-grid">${matrix(
        "Incoming H",
        e.hidden,
        "Toy [1,4,2] / release [T,4,2560]"
      )}${shapeCard(
        "Persistent residual state",
        "[T,4,2560] &harr; [T,10240]",
        "Keep the original H for the residual addition after this sublayer. The stream axis survives every block."
      )}</div>`,
      `<h4>2. Normalize each stream independently</h4><p>RMS statistics are computed over the channel dimension D of each row. The toy offset scales are zero, so the learned multiplier 1 + scale is 1. Both gate projections read these normalized streams.</p><div class="step-grid">${matrix(
        "Original H",
        e.hidden,
        "[1,4,2]"
      )}${matrix("Normalized H", e.normalized, "[1,4,2]")}</div>${code(
        "N_TSD = grouped_rms_norm(H_TSD)   # [T,4,2560]\nn_TC = N_TSD.flatten(-2)          # [T,10240]"
      )}`,
      `<h4>3. Predict gates from this token's hidden state</h4><p>The trained matrices are fixed during this forward. Their output gates vary with the input H. Read gates have a channel dimension; write gates have one scalar per stream.</p><div class="step-grid">${matrix(
        "Read gates",
        e.readGates,
        "Toy [1,4,2] / release [T,4,2560]"
      )}${matrix("Write gates", e.writeGates, "[1,4] / release [T,4]", ["t"], "s")}</div>${code(
        "z_TR = silu(linear_down(n_TC) / 4)     # [T,320]\nread_TSD = sigmoid(linear_up(z_TR))    # [T,4,2560]\nwrite_TS = 2 * sigmoid(linear_write(n_TC) / 4)  # [T,4]"
      )}<p class="note">PyTorch weight shapes: down [320,10240], up [10240,320], write [4,10240]. Toy write logits: ${vector(e.writeLogits)}.</p>`,
      `<h4>4. Multiply by the read gates, then average streams</h4><p>For each output channel, sum its four gated normalized values and divide by four. The result is one vector, which attention or MoE receives.</p><div class="step-grid">${matrix(
        "Read gates x normalized streams",
        weighted,
        "[1,4,2]"
      )}${matrix("Sublayer input", e.input, "Toy [1,2] / release [T,2560]")}</div>${code("x_TD = (read_TSD * N_TSD).mean(dim=1)  # [T,2560]")}`,
      `<h4>5. Compute a single sublayer update</h4><p>Attention or MoE computes at D = 2560. Its result has the same width as its input. This teaching example substitutes a fixed two-channel update so the write operation can be inspected directly.</p><div class="step-grid">${matrix(
        "Sublayer input",
        e.input,
        "[T,2560] in the model"
      )}${matrix("Illustrative sublayer update", e.update, "[T,2560] in the model")}</div>${code(
        "update_TD = attention(x_TD)  # or moe(x_TD)\n# The toy substitutes update = [0.8, -0.4]."
      )}`,
      `<h4>6. Broadcast one update, then add to the original streams</h4><p>Each write scalar multiplies every channel of the same update. The four scaled copies are added to the original H. These updated streams become the next sublayer's input.</p><div class="step-grid three">${matrix(
        "Original H",
        e.hidden,
        "[1,4,2]"
      )}${matrix("Write gates x update", updates, "[1,4,2]")}${matrix("Updated H", e.output, "[1,4,2]")}</div>${code(
        "scaled_TSD = write_TS[:,:,None] * update_TD[:,None,:]\nH_new_TSD = H_TSD + scaled_TSD  # [T,4,2560]"
      )}<p class="note">The next read uses this H_new. Attention and MoE have independent trained read/write matrices.</p>`,
    ];
  }
  const gr = stepper("gr", 6, (step) => {
    const bias = Number(byId("gr-write-bias").value);
    byId("gr-bias-label").textContent = bias.toFixed(1);
    byId("gr-content").innerHTML = residualBodies(M.residualExample(byId("gr-entry").value, bias))[step];
  });
  for (const id of ["gr-entry", "gr-write-bias"]) byId(id).addEventListener("input", gr.refresh);

  function renderDelta(step) {
    const decay = Number(byId("gdn-decay").value) / 100;
    const beta = Number(byId("gdn-beta").value) / 100;
    const e = M.deltaExample(decay, beta, byId("gdn-query").value);
    byId("gdn-decay-label").textContent = decay.toFixed(2);
    byId("gdn-beta-label").textContent = beta.toFixed(2);
    byId("gdn-inputs").innerHTML =
      matrix("Previous state", e.previous, "[K,V] = [2,2]") +
      matrix("Current key k", e.key, "[K] = [2]") +
      matrix("Current value v", e.value, "[V] = [2]") +
      matrix("Current query q", e.q, "[K] = [2]");
    const bodies = [
      `<h4>1. Decay the old state</h4><p>Multiply all entries by &alpha; = ${decay.toFixed(
        2
      )}. Smaller &alpha; forgets more old information. The matrix dimensions stay the same.</p><div class="step-grid">${matrix(
        "Previous state",
        e.previous,
        "[2,2]"
      )}${matrix("Decayed state", e.decayed, "[2,2]")}</div>`,
      `<h4>2. Predict the value already stored under this key</h4><p>The key [1,0] reads the first state row. Compare that prediction with the new value. The write will correct the difference.</p><div class="step-grid three">${matrix(
        "Prediction: k^T S_decayed",
        e.prediction,
        "[V] = [2]"
      )}${matrix("New value v", e.value, "[V] = [2]")}${matrix("Error: v - prediction", e.error, "[V] = [2]")}</div>`,
      `<h4>3. Add a gated outer-product correction</h4><p>Use &beta; = ${beta.toFixed(
        2
      )} to scale k &times; error. With this key, only the first row gets a corrective write. At &beta; = 1, reading along k recovers the new value exactly in this example.</p><div class="step-grid three">${matrix(
        "Decayed state",
        e.decayed,
        "[2,2]"
      )}${matrix("beta x key x error", e.correction, "[2,2]")}${matrix("Updated state", e.state, "[2,2]")}</div>`,
      `<h4>4. Read the updated state with the query</h4><p>The query controls which stored directions contribute to the output. Change its direction to read the updated first row, the other row, or a mixture.</p><div class="step-grid">${matrix(
        "Updated state",
        e.state,
        "[2,2]"
      )}${matrix(
        "Output: q^T S_new",
        e.output,
        "[V] = [2]"
      )}</div><p class="note">The model repeats this operation for 48 value heads, each producing 128 features. Output normalization, sigmoid gating, and a dense projection then return [T,2560].</p>`,
    ];
    byId("gdn-content").innerHTML = bodies[step];
  }
  const gdn = stepper("gdn", 4, renderDelta);
  for (const id of ["gdn-decay", "gdn-beta", "gdn-query"]) byId(id).addEventListener("input", gdn.refresh);

  const sequences = {
    city: { words: ["New", "York", "City"], ids: [10, 20, 30] },
    lake: { words: ["Salt", "Lake", "City"], ids: [40, 50, 30] },
    very: { words: ["very", "good"], ids: [60, 70] },
    not: { words: ["not", "good"], ids: [80, 70] },
  };
  function hashGrid(rows, withVectors) {
    return `<div class="hash-grid">${rows
      .map((entry) => {
        const preview = vector([0, 1, 2].map((channel) => Math.sin(entry.row * 0.01 + channel * 1.7)));
        return `<div class="hash-row${entry.order === 3 ? " trigram" : ""}" data-hash-head="${entry.head}" data-hash-row="${entry.row}"><strong>${
          entry.order === 2 ? "Bigram" : "Trigram"
        } head ${(entry.head % 8) + 1}</strong><code>${withVectors ? `table[${integer(entry.row)}]` : `row ${integer(entry.row)}`}</code>${
          withVectors
            ? `<small>${preview.slice(0, -1)}, ...]</small><small>160 learned features (synthetic preview)</small>`
            : `<small>bucket ${integer(entry.bucket)} + offset ${integer(entry.offset)}</small><small>${integer(
                M.lexicalHeadSizes[entry.head]
              )} buckets in this head</small>`
        }</div>`;
      })
      .join("")}</div>`;
  }
  function renderLexical(step) {
    const sequence = sequences[byId("ple-sequence").value];
    const offset = Number(byId("ple-position").value);
    const reset = byId("ple-boundary").checked;
    const positions = sequence.ids.map((_, index) => offset + index);
    if (reset) positions[positions.length - 1] = 0;
    const rows = M.lexicalHashes(sequence.ids, positions).at(-1);
    const words = reset ? ["<EOS>", "<EOS>", sequence.words.at(-1)] : ["<EOS>", ...sequence.words].slice(-3);
    const ids = reset ? [248044, 248044, sequence.ids.at(-1)] : [248044, ...sequence.ids].slice(-3);
    byId("ple-position-label").textContent = integer(offset);
    byId("ple-token-history").innerHTML = sequence.words
      .map(
        (word, index) =>
          `<span class="token-pill${
            index === sequence.words.length - 1 ? " current" : reset ? " excluded" : ""
          }"><strong>${word}</strong><small>demo ID ${sequence.ids[index]} &middot; position ${offset + index}${
            reset && index === sequence.words.length - 1 ? " / document position 0" : ""
          }</small></span>`
      )
      .join("");
    const bodies = [
      `<h4>1. Gather recent IDs for the current token</h4><p>The last token is the lookup's current token. The relative offsets choose its preceding IDs; the integer token values and their order define the pattern. ${
        reset
          ? "A new document starts here, so both preceding slots use EOS padding."
          : "Move the sequence later: its content and hash addresses remain the same."
      }</p><div class="step-grid">${shapeCard("Ordered bigram", escape(words.slice(-2).join(" | ")), `IDs [${ids.slice(-2).join(", ")}]`)}${shapeCard(
        "Ordered trigram",
        escape(words.join(" | ")),
        `IDs [${ids.join(", ")}]`
      )}</div>${code(
        "current  = input_ids[t]\nprevious = input_ids[t-1]  # EOS when outside this document\nolder    = input_ids[t-2]  # EOS when outside this document"
      )}`,
      `<h4>2. Compute sixteen integer table addresses</h4><p>Eight heads map the bigram into separate regions; eight more map the trigram. Fixed integer multipliers and XOR combine the IDs. Each head uses its own prime bucket count and global row offset.</p>${code(
        "pair   = (current * m0) XOR (previous * m1)\ntriple = pair XOR (older * m2)\nrow[head] = mixed % head_size[head] + head_offset[head]\n# int64 arithmetic; output shape [1,16]"
      )}${hashGrid(rows, false)}`,
      `<h4>3. Select sixteen rows from the learned table</h4><p>Each address selects a whole row of 160 numbers. The returned tensor groups those sixteen vectors under the current token. The table remains a model parameter with all its available rows.</p><div class="lookup-sizes"><div><small>Stored parameter</small><strong>320,001,536 rows</strong><code>table [rows,160]</code></div><div><small>Input addresses / one token</small><strong>16 integers</strong><code>hash_ids [1,16]</code></div><div><small>Returned features / one token</small><strong>16 &times; 160</strong><code>E [1,16,160]</code></div></div>${hashGrid(
        rows,
        true
      )}${code("E_TNF = table[hash_ids_TN]  # [T,16,160]\n# E contains the selected row vectors.")}`,
      `<h4>4. Concatenate the sixteen feature vectors</h4><p>For each token, lay its sixteen 160-feature vectors end to end. This produces 2560 lexical features. Tokens remain separate along T.</p><div class="step-grid">${shapeCard(
        "Lookup output",
        "E [T,16,160]",
        "Sixteen vectors for every current token."
      )}${shapeCard("Concatenated lexical vector", "e [T,2560]", "16 x 160 = 2560. This is an activation returned by the lookup.")}</div>${code(
        "e_TE = E_TNF.reshape(T, 16 * 160)  # [T,2560]"
      )}`,
      `<h4>5. Let the incoming hidden state gate the memory</h4><p>Project the lexical vector into four keys and one shared value. Independently normalize the four incoming H streams. Each stream's key/query similarity gives one sigmoid gate.</p><div class="step-grid three">${shapeCard(
        "Lexical keys",
        "[T,2560] &rarr; [T,10240]<br />&rarr; [T,4,2560]",
        "Key projection weight [10240,2560], followed by grouped RMSNorm."
      )}${shapeCard(
        "Hidden-state queries",
        "H [T,4,2560]<br />&rarr; normalized queries",
        "Grouped RMSNorm of H; no extra query projection."
      )}${shapeCard("Shared lexical value", "[T,2560] &rarr; [T,2560]", "Value projection weight [2560,2560].")}</div>${code(
        "a_TS = (key_TSD * query_TSD).sum(-1) / sqrt(2560)\ngate_TS = sigmoid(sign(a_TS) * sqrt(clamp(abs(a_TS), min=1e-6)))\n# One gate per token and stream: [T,4]"
      )}`,
      `<h4>6. Form a four-stream residual update</h4><p>Broadcast the shared value into four streams, using their contextual gates. Add a causal convolution refinement, then add the resulting delta to H. Layer 2's attention read follows.</p>${code(
        "u_TSD = gate_TS[:,:,None] * value_TD[:,None,:]  # [T,4,2560]\nrefined = silu(causal_conv(grouped_norm(u_TSD.flatten(-2))))\ndelta_TSD = u_TSD + refined.reshape(T,4,2560)\nH_TSD = H_TSD + delta_TSD"
      )}<div class="step-grid">${shapeCard(
        "Causal convolution",
        "weight [10240,1,4]",
        "One filter per carried channel, with taps at t, t-3, t-6, and t-9. Document boundaries mask unavailable history."
      )}${shapeCard(
        "PLE output",
        "delta [T,4,2560]",
        "The caller adds delta to the existing H. Its original IDs remain available as separate metadata."
      )}</div>`,
    ];
    byId("ple-content").innerHTML = bodies[step];
  }
  const ple = stepper("ple", 6, renderLexical);
  for (const id of ["ple-sequence", "ple-position", "ple-boundary"]) byId(id).addEventListener("input", ple.refresh);

  function renderExperts() {
    const tokens = Number(byId("expert-tokens").value);
    const result = M.trainingFootprint(tokens, 2, 1, false);
    byId("expert-assignments").textContent = `R = 10 x ${integer(tokens)} = ${integer(
      result.assignments
    )} assignments; their logical BF16 input payload is ${bytes(result.assignmentPayload)} across the expert group.`;
  }
  byId("expert-tokens").addEventListener("input", renderExperts);
  renderExperts();

  function renderTraining() {
    const result = M.trainingFootprint(
      Number(byId("training-tokens").value),
      Number(byId("training-dtype").value),
      Number(byId("training-tp").value),
      byId("training-sp").checked
    );
    const fields = [
      ["residual", "One logical four-stream boundary", bytes(result.residual), "[T,4,2560] before SP"],
      ["localResidual", "One carried boundary per TP rank", bytes(result.localResidual), `[${integer(result.localTokens)},4,2560]`],
      ["localSublayer", "D-wide input at that token partition", bytes(result.localSublayer), `[${integer(result.localTokens)},2560]`],
      ["assignments", "Logical routed assignments", integer(result.assignments), "R = 10T over the expert group"],
      ["assignmentPayload", "Logical expert input payload", bytes(result.assignmentPayload), "R x 2560 x element bytes"],
      ["lexicalRead", "Logical BF16 lexical row reads", bytes(result.lexicalRead), `${integer(result.lexicalRows)} lookups, before deduplication`],
    ];
    byId("training-footprint").innerHTML = fields
      .map(
        ([key, label, value, note]) =>
          `<div data-training-metric="${key}"><small>${label}</small><strong>${value}</strong><small>${note}</small></div>`
      )
      .join("");
    byId("training-shape-note").textContent = `${
      result.sequenceParallel
        ? `SP across TP${result.tp}: T_local = ${integer(result.tokens)} / ${result.tp} = ${integer(result.localTokens)}.`
        : `SP is off: each TP rank carries all ${integer(result.tokens)} tokens in this boundary scenario.`
    } Two full-T int64 arrays for original IDs and document positions occupy ${bytes(
      result.tokenMetadata
    )}. Sizes omit saved gates, recurrent/attention intermediates, weights, gradients, optimizer state, and allocator overhead.`;
  }
  for (const id of ["training-tokens", "training-dtype", "training-tp", "training-sp"]) byId(id).addEventListener("input", renderTraining);
  renderTraining();
})();
