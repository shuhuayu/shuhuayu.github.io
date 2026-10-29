---
layout: page
permalink: /blogs/
title: blogs
description: Notes and interactive diagrams on model architecture, memory, and training.
nav: true
nav_order: 3
studies:
  - title: Kimi K3
    url: /studies/kimi-k3.html
    description: KDA and MLA attention, Attention Residuals, and Stable LatentMoE.
  - title: DeepSeek V4
    url: /studies/deepseek-v4.html
    description: CSA and HCA attention, mHC residual mixing, MoE routing, and distributed Muon.
  - title: DeepSeek V4.1 Flash
    url: /studies/deepseek-v4-1-flash.html
    description: CSA2 sparse attention, shared KV memory, the causal encoder and decoder, and Engram.
  - title: Qwen3.8-Flash-Next
    url: /studies/qwen3-8-flash-next.html
    description: Recurrent states, sparse attention, gated residual streams, and lexical memory in a hybrid MoE decoder.
---

<ul class="post-list">
  {% for study in page.studies %}
    <li>
      <h2 class="h3">
        <a class="post-title" href="{{ study.url | relative_url }}">{{ study.title }}</a>
      </h2>
      <p>{{ study.description }}</p>
    </li>
  {% endfor %}
</ul>
