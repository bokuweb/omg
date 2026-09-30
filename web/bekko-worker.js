import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.bundle.min.mjs";
import { Tokenizer } from "https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/+esm";
import { cachedFetch } from "./cache.js";
import { predict } from "./bekko-core.js";

// A worker keeps CPU inference and tokenizer/model initialization off the UI thread.
// One thread works on GitHub Pages without cross-origin-isolation headers.
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
let runtime;
const progress = info => self.postMessage({ progress: info });
async function dispatch(data) {
  if (data.type === "load") {
    if (data.device === "webgpu" && !await navigator.gpu?.requestAdapter()) throw Error("WebGPU is unavailable. Select CPU and try again.");
    const asset = name => cachedFetch(new URL(name, data.base).href, progress);
    const [manifest, config, tokenizerJson] = await Promise.all(["manifest.json", "tokenizer_config.json", "tokenizer.json"].map(async name => (await asset(name)).json()));
    const tokenizer = new Tokenizer(tokenizerJson, config);
    const bytes = await (await asset(manifest.model_file || "model.onnx")).arrayBuffer();
    progress({ status: "ready" });
    const session = await ort.InferenceSession.create(bytes, { executionProviders: data.device === "cpu" ? ["wasm"] : ["webgpu", "wasm"] });
    runtime = { manifest, tokenizer, session, ort };
    return {};
  }
  if (!runtime) throw Error("Load Bekko before running a request");
  const rows = [];
  let inputTokens = 0, forwards = 0;
  for (const decision of data.decisions) {
    const result = await predict(decision, runtime);
    rows.push({ logits: result.logits });
    inputTokens += result.inputTokens;
    forwards += result.forwards;
  }
  return { rows, inputTokens, forwards };
}
// ONNX sessions may not run concurrently; failures must not poison the queue.
let queue = Promise.resolve();
self.onmessage = ({ data }) => {
  queue = queue.then(async () => {
    try { self.postMessage({ id: data.id, result: await dispatch(data) }); }
    catch (error) { self.postMessage({ id: data.id, error: error.message }); }
  });
};
