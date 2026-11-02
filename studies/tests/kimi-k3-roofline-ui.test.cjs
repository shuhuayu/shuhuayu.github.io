const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const A = require("../kimi-k3-accounting.js");
const R = require("../kimi-k3-roofline-model.js");

// Exercises controller behavior, not browser rendering or layout.
class Element {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.dataset = {};
    this.style = {};
    this.options = [];
    this.listeners = {};
    this.value = "";
    this.textContent = "";
    const classes = new Set();
    this.classList = {
      contains: (name) => classes.has(name),
      toggle: (name, force) => {
        const enabled = force === undefined ? !classes.has(name) : Boolean(force);
        if (enabled) classes.add(name);
        else classes.delete(name);
        return enabled;
      },
    };
  }
  appendChild(child) {
    child.parent = this;
    this.children.push(child);
    return child;
  }
  append(...children) {
    children.forEach((child) => this.appendChild(child));
  }
  replaceChildren(...children) {
    this.children = [];
    this.append(...children);
  }
  remove() {
    if (this.parent) this.parent.children = this.parent.children.filter((child) => child !== this);
  }
  setAttribute(name, value) {
    this[name] = value;
  }
  querySelector(selector) {
    assert.equal(selector, "tbody");
    if (!this.body) this.body = this.appendChild(new Element("tbody"));
    return this.body;
  }
  querySelectorAll(selector) {
    return this.children.filter((child) => selector.split(/,\s*/).includes(child.tag));
  }
  addEventListener(name, callback) {
    (this.listeners[name] ??= []).push(callback);
  }
  emit(name) {
    (this.listeners[name] || []).forEach((callback) => callback());
  }
  click() {
    this.emit("click");
  }
  checkValidity() {
    return this.valid !== false;
  }
}

function controller() {
  const html = readFileSync(join(__dirname, "..", "kimi-k3-roofline.html"), "utf8");
  const elements = new Map([...html.matchAll(/<([\w-]+)[^>]*\bid="([^"]+)"[^>]*>/g)].map((match) => [match[2], new Element(match[1])]));
  for (const match of html.matchAll(/<select\s+id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)) {
    const options = [...match[2].matchAll(/<option([^>]*)>([^<]+)<\/option>/g)].map((option) => ({
      value: option[1].match(/value="([^"]+)"/)?.[1] ?? option[2].trim(),
      disabled: false,
    }));
    // HTML collections are iterable but do not provide Array.find/filter.
    elements.get(match[1]).options = {
      length: options.length,
      item: (index) => options[index] ?? null,
      [Symbol.iterator]: () => options[Symbol.iterator](),
    };
  }
  const presets = [...html.matchAll(/data-preset="([^"]+)"/g)].map((match) => {
    const button = new Element("button");
    button.dataset.preset = match[1];
    return button;
  });
  const document = {
    getElementById: (id) => elements.get(id) ?? null,
    createElement: (tag) => new Element(tag),
    createElementNS: (_, tag) => new Element(tag),
    querySelectorAll: (selector) => {
      assert.equal(selector, "[data-preset]");
      return presets;
    },
    body: new Element("body"),
    documentElement: new Element("html"),
  };
  const downloads = [];
  vm.runInNewContext(readFileSync(join(__dirname, "..", "kimi-k3-roofline.js"), "utf8"), {
    window: { KimiK3Model: A, KimiK3Roofline: R },
    document,
    localStorage: { getItem: () => null, setItem() {} },
    matchMedia: () => ({ matches: false }),
    setTimeout: (callback) => callback(),
    Blob: class {
      constructor(parts, { type }) {
        this.content = parts.join("");
        this.type = type;
      }
    },
    URL: {
      createObjectURL: (blob) => {
        downloads.push(blob);
        return "blob:test";
      },
      revokeObjectURL() {},
    },
  });
  const get = (id) => elements.get(id);
  const change = (input) => {
    for (const [key, value] of Object.entries(input)) get(`roof-${key}`).value = String(value);
    get(`roof-${Object.keys(input).at(-1)}`).emit("change");
  };
  return { get, change, downloads, preset: (name) => presets.find((button) => button.dataset.preset === name).click() };
}

test("controller renders recipe results without a headline and updates the configured GPU count (DOM test double)", () => {
  const ui = controller();
  const initial = R.calculate();
  assert.equal(ui.get("headline-compute"), undefined);
  assert.equal(Number(ui.get("roof-recipe-throughput").dataset.throughput), initial.resourceTokensPerSecondPerGPU);
  assert.notEqual(Number(ui.get("roof-recipe-throughput").dataset.throughput), initial.computeTokensPerSecondPerGPU);
  ui.change({ world: 256, pp: 4, mbs: 32 });
  const expected = R.calculate({ world: 256, pp: 4, mbs: 32 });
  assert.match(ui.get("roof-shape").textContent, /256 = PP4 x CP1 x DP64/);
  assert.equal(Number(ui.get("roof-tokens").dataset.tokens), expected.globalTokens);
  assert.equal(Number(ui.get("roof-recipe-throughput").dataset.throughput), expected.resourceTokensPerSecondPerGPU);
  assert.equal(Number(ui.get("roof-recipe-tflops").dataset.tflops), expected.resourceModelTFLOPsPerGPU);
});

test("precision preset preserves lifecycle and communication inputs affect the recipe result (DOM test double)", () => {
  const ui = controller();
  ui.preset("mxfp8");
  assert.equal(ui.get("roof-precision").value, "mxfp8-experts");
  assert.equal(ui.get("roof-policy").value, R.defaults.policy);
  const mx = R.calculate({ precision: "mxfp8-experts" });
  assert.equal(Number(ui.get("roof-recipe-throughput").dataset.throughput), mx.resourceTokensPerSecondPerGPU);
  ui.change({ domain: 8, network: 5, routingMap: "hybridep" });
  assert.ok(Number(ui.get("roof-recipe-throughput").dataset.throughput) < mx.resourceTokensPerSecondPerGPU);
  ui.preset("baseline");
  assert.equal(Number(ui.get("roof-recipe-throughput").dataset.throughput), R.calculate().resourceTokensPerSecondPerGPU);
});

test("hardware selection loads reference rates without changing the configured recipe (DOM test double)", () => {
  const ui = controller();
  assert.equal(ui.get("roof-hardware").value, "gb200");
  assert.equal(ui.get("roof-hardware-note").dataset.computeBasis, "gb300-reference-proxy");
  assert.match(ui.get("roof-hardware-note").textContent, /uncalibrated GB200 proxy/);
  ui.change({ world: 256, pp: 4, mbs: 32, seq: 8192, precision: "mxfp8-experts", gemm: 1234.5, network: 50 });
  ui.change({ hardware: "gb300" });
  const expected = R.calculate({ hardware: "gb300", world: 256, pp: 4, mbs: 32, seq: 8192, precision: "mxfp8-experts" });
  assert.equal(Number(ui.get("roof-world").value), 256);
  assert.equal(Number(ui.get("roof-pp").value), 4);
  assert.equal(Number(ui.get("roof-mbs").value), 32);
  assert.equal(Number(ui.get("roof-seq").value), 8192);
  assert.equal(ui.get("roof-precision").value, "mxfp8-experts");
  for (const key of R.rateKeys) assert.equal(Number(ui.get(`roof-${key}`).value), R.hardwareProfiles.gb300.rates[key]);
  assert.equal(ui.get("roof-hardware-note").dataset.hardware, "gb300");
  assert.equal(ui.get("roof-hardware-note").dataset.computeBasis, "gb300-practical-target");
  assert.match(ui.get("roof-result-caption").textContent, /256 GB300 GPUs/);
  assert.match(ui.get("roof-shape").textContent, /GB300: 256 = PP4 x CP1 x DP64/);
  assert.equal(Number(ui.get("roof-recipe-throughput").dataset.throughput), expected.resourceTokensPerSecondPerGPU);
  ui.change({ hardware: "gb200" });
  assert.equal(ui.get("roof-hardware-note").dataset.computeBasis, "gb300-reference-proxy");
  assert.match(ui.get("roof-result-caption").textContent, /256 GB200 GPUs/);
  assert.equal(Number(ui.get("roof-recipe-throughput").dataset.throughput), expected.resourceTokensPerSecondPerGPU);
});

test("scenario buttons preserve hardware and custom rates while reset restores 512 GB200 (DOM test double)", () => {
  const ui = controller();
  ui.change({ hardware: "gb300" });
  ui.change({ world: 256, pp: 4, mbs: 32, gemm: 1234.5, hbm: 6.2, network: 50 });
  const input = { hardware: "gb300", world: 256, pp: 4, mbs: 32, gemm: 1234.5, hbm: 6.2, network: 50, precision: "mxfp8-experts" };
  ui.preset("mxfp8");
  for (const [key, value] of Object.entries(input)) assert.equal(String(ui.get(`roof-${key}`).value), String(value));
  assert.match(ui.get("roof-hardware-note").textContent, /Custom input overrides: gemm, hbm, network/);
  assert.match(ui.get("roof-hardware-note").textContent, /no calibration measurements attached/);
  assert.equal(ui.get("hardware-bf16-ridge").textContent, `${(input.gemm / input.hbm).toLocaleString("en-US", { maximumFractionDigits: 2 })} FLOP/B`);
  assert.equal(Number(ui.get("roof-recipe-throughput").dataset.throughput), R.calculate(input).resourceTokensPerSecondPerGPU);
  ui.preset("baseline");
  for (const [key, value] of Object.entries(R.defaults)) assert.equal(String(ui.get(`roof-${key}`).value), String(value));
  assert.match(ui.get("roof-result-caption").textContent, /512 GB200 GPUs/);
  assert.equal(Number(ui.get("roof-recipe-throughput").dataset.throughput), R.calculate().resourceTokensPerSecondPerGPU);
});

test("estimated VPP updates chunk placement, wraparound packets and results, and returning to VPP1 restores the original result (DOM test double)", () => {
  const ui = controller();
  ui.change({ hardware: "gb300" });
  ui.change({ world: 256, pp: 4, mbs: 32 });
  const before = Number(ui.get("roof-recipe-throughput").dataset.throughput);
  ui.preset("vpp2");
  const expected = R.calculate({ hardware: "gb300", world: 256, pp: 4, mbs: 32, vp: 2 });
  assert.equal(Number(ui.get("roof-vp").value), 2);
  assert.match(ui.get("roof-pipeline-mode").textContent, /Estimated VPP2.*interleaved 1F1B/);
  assert.equal(String(ui.get("roof-pipeline-mode").dataset.estimated), "true");
  assert.match(ui.get("roof-result-caption").textContent, /estimated VPP2/);
  assert.match(ui.get("roof-warning").textContent, /Residual-cache reuse, communication startup and schedule-specific waits are not modeled/);
  assert.match(ui.get("roof-stage-buttons").children[0].children[0].textContent, /C1: L1-11 \/ C5: L48-59/);
  assert.ok(ui.get("pp-packets").children.some((row) => row.children[0]?.textContent === "C4/PP3 -> C5/PP0"));
  assert.equal(Number(ui.get("roof-recipe-throughput").dataset.throughput), expected.resourceTokensPerSecondPerGPU);
  assert.match(ui.get("bubble-caption").textContent, /\(PP - 1\) \/ \(M \* VPP\)/);
  ui.change({ vp: 1 });
  assert.equal(String(ui.get("roof-pipeline-mode").dataset.estimated), "false");
  assert.equal(Number(ui.get("roof-recipe-throughput").dataset.throughput), before);
  assert.doesNotMatch(ui.get("roof-result-caption").textContent, /estimated/);
});

test("VPP selector disables empty-chunk splits and normalizes only unsupported VPP choices (DOM test double)", () => {
  const ui = controller();
  ui.change({ vp: 4 });
  assert.equal(Number(ui.get("roof-vp").value), 4);
  ui.change({ pp: 16 });
  assert.equal(Number(ui.get("roof-vp").value), 2);
  assert.equal([...ui.get("roof-vp").options].find((option) => option.value === "4").disabled, true);
  assert.match(ui.get("roof-warning").textContent, /VPP 4 -> 2 to keep decoder chunks nonempty/);
  const expected = R.calculate({ pp: 16, vp: 2, ep: 32, domain: 32 });
  assert.equal(Number(ui.get("roof-recipe-throughput").dataset.throughput), expected.resourceTokensPerSecondPerGPU);
});

test("JSON and CSV exports retain the configured world, estimated VPP and recipe-specific results (DOM test double)", () => {
  const ui = controller();
  ui.change({ hardware: "gb300" });
  ui.change({ world: 256, pp: 4, mbs: 32, precision: "mxfp8-experts", network: 50, vp: 2 });
  const expected = R.calculate({ hardware: "gb300", world: 256, pp: 4, mbs: 32, precision: "mxfp8-experts", network: 50, vp: 2 });
  ui.get("export-json").click();
  const json = JSON.parse(ui.downloads[0].content);
  assert.equal(json.config.world, 256);
  assert.equal(json.config.hardware, "gb300");
  assert.equal(json.config.vp, 2);
  assert.equal(json.pipeline.estimated, true);
  assert.equal(json.pipelineChunks.length, 8);
  assert.equal(json.pipelineEdges.length, 7);
  assert.ok(json.stages.every((stage) => stage.logicalChunks.length === 2));
  assert.equal(json.hardware.computeBasis, "gb300-practical-target");
  assert.deepEqual(json.hardware.overriddenRates, ["network"]);
  assert.deepEqual(json.hardware.referenceRates, R.hardwareProfiles.gb300.rates);
  assert.equal(json.derived.resourceModelTokensPerSecondPerGPU, expected.resourceTokensPerSecondPerGPU);
  assert.equal(json.derived.resourceModelTFLOPsPerGPU, expected.resourceModelTFLOPsPerGPU);
  ui.get("export-csv").click();
  const [header, ...rows] = ui.downloads[1].content
    .trim()
    .split("\n")
    .map((line) => line.split(","));
  assert.equal(rows.length, 4);
  for (const row of rows) {
    assert.equal(row.length, header.length);
    assert.equal(row[header.indexOf("gpu_count")], "256");
    assert.equal(row[header.indexOf("gpu_hardware")], "gb300");
    assert.equal(row[header.indexOf("compute_rate_basis")], "gb300-practical-target");
    assert.equal(row[header.indexOf("overridden_rates")], "network");
    assert.equal(row[header.indexOf("network_gb_s")], "50");
    assert.equal(row[header.indexOf("virtual_chunks_per_rank")], "2");
    assert.equal(row[header.indexOf("pipeline_schedule")], "interleaved-1f1b");
    assert.equal(row[header.indexOf("vpp_estimated")], "true");
    assert.equal(row[header.indexOf("logical_chunk_indices")], `${row[0]};${Number(row[0]) + 4}`);
  }
});
