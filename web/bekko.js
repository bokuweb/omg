import { decisionFor } from "./bekko-core.js";

// Pin weights and tokenizer together to the releases used by the upstream demo.
export const BEKKO_MODELS = Object.fromEntries([
  ["17m", "29 MB", "cf92c2f76214e764b9c4e3054125cdb146b0732e"],
  ["68m", "196 MB", "995dda388ac0f6735d8dc3cd4932d380c6fe06fc"],
  ["400m", "1.43 GB", "93c993e2fbf8ed480b49d410fe52347062616c6e"],
].map(([size, download, revision]) => {
  const id = `bekko-system-one-${size}`;
  return [id, { id, hub: `hotchpotch/bekko-system-one-v0-${size}`, revision, kind: "bekko", dtype: "int8/fp32", size: download, note: "Bekko: English decision model, CPU or WebGPU" }];
}));
export const bekkoBase = spec => `https://huggingface.co/${spec.hub}/resolve/${spec.revision}/onnx_browser/`;
// Only used for validation, option ordering and response assembly, not tokenization.
const LAYOUT = JSON.stringify({ layout: "pointer", state: "", question: "", opt: "", opt_end: "", decide: "" });

export async function loadBekko({ spec, device = "cpu", onProgress, modelBase, omg }) {
  if (!["cpu", "webgpu"].includes(device)) throw Error(`Unsupported Bekko device: ${device}`);
  const worker = new Worker(new URL("./bekko-worker.js", import.meta.url), { type: "module" });
  let nextId = 0, failed;
  const pending = new Map();
  const fail = error => {
    failed = error;
    for (const { reject } of pending.values()) reject(error);
    pending.clear();
    worker.terminate();
  };
  worker.onerror = event => fail(new Error(event.message || "Bekko worker failed to initialize"));
  worker.onmessageerror = () => fail(new Error("Could not read the Bekko worker response"));
  worker.onmessage = ({ data }) => {
    if (data.progress) { onProgress?.(data.progress); return; }
    const callback = pending.get(data.id);
    if (!callback) return;
    pending.delete(data.id);
    if (data.error) callback.reject(new Error(data.error));
    else callback.resolve(data.result);
  };
  const call = data => new Promise((resolve, reject) => {
    if (failed) { reject(failed); return; }
    const id = nextId++;
    pending.set(id, { resolve, reject });
    worker.postMessage({ ...data, id });
  });
  try {
    const base = modelBase ? new URL(`${modelBase.replace(/\/?$/, "/")}${spec.id}/`, location.href).href : bekkoBase(spec);
    await call({ type: "load", base, device });
  } catch (error) { fail(error); throw error; }
  return {
    model: spec.id, spec, device, labels: [],
    render: request => JSON.parse(omg.render(JSON.stringify(request), LAYOUT)),
    async answer(request, { temperature = 1 } = {}) {
      if (!Number.isFinite(temperature) || temperature <= 0) throw Error("Temperature must be positive and finite");
      const start = performance.now();
      const json = JSON.stringify(request);
      const { branches } = JSON.parse(omg.render(json, LAYOUT));
      // Validate every decision before starting inference (Bekko supports 2–64 options).
      const decisions = branches.map(branch => decisionFor(request.state, request.questions[branch.id], branch.keys));
      const { rows, inputTokens, forwards } = await call({ type: "answer", decisions });
      const out = JSON.parse(omg.answer(json, LAYOUT, 1, 0, JSON.stringify(rows), undefined, temperature, spec.id, inputTokens, undefined, undefined));
      return { ...out.response, diagnostics: out.diagnostics, usage: { ...out.response.usage, questions: branches.length, branches: branches.length, orders: 1, mode: "bekko", ms: performance.now() - start, forwards, passes: 1 } };
    },
  };
}
