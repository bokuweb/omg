import createBonsai from "./pkg/bonsai/bonsai.js";
import init, * as omg from "./pkg/omg.js";
import { BONSAI_MODEL as spec } from "./bonsai.js";
import { cachedFetch, idbCache } from "./cache.js";
import { preparePlan } from "./bonsai-core.js";

let wasm, labels, resident = null;
const layout = JSON.stringify(spec.layout);
const progress = info => self.postMessage({ progress: info });
function check(rc) {
  if (rc < 0) { resident = null; throw Error(wasm.ccall("bonsai_error", "string", [], [])); }
  return rc;
}
function tokenize(text, special) {
  const n = check(wasm.ccall("bonsai_tokenize", "number", ["string", "number", "number"], [text, new TextEncoder().encode(text).length, Number(special)]));
  const ptr = wasm._bonsai_tokens();
  return Array.from(wasm.HEAP32.subarray(ptr / 4, ptr / 4 + n));
}
function withIds(ids, fn) {
  const ptr = wasm._malloc(ids.length * 4);
  if (!ptr) throw Error("Bonsai ran out of browser memory");
  try { wasm.HEAP32.set(ids, ptr / 4); return fn(ptr); }
  finally { wasm._free(ptr); }
}
function rowsFor(plan) {
  // Validate every branch before spending CPU time on any of them.
  const { prefix, branches } = preparePlan(plan, tokenize, labels.length);
  const warm = resident && resident.length === prefix.length && resident.every((id, i) => id === prefix[i]);
  wasm._bonsai_forwards();
  withIds(prefix, ptr => check(wasm._bonsai_prefix(ptr, prefix.length)));
  resident = prefix;
  const rows = branches.map(({ ids, count }) => {
    withIds(ids, ptr => check(wasm._bonsai_branch(ptr, ids.length)));
    return withIds(labels.slice(0, count), ptr => {
      const row = wasm._bonsai_row(ptr, count) / 4;
      return { logits: Array.from(wasm.HEAPF32.subarray(row, row + count)), candidate_mass: wasm.HEAPF32[row + count] };
    });
  });
  return { rows, warm: Boolean(warm), state_tokens: prefix.length, tokens: prefix.length + branches.reduce((n, b) => n + b.ids.length, 0), forwards: wasm._bonsai_forwards() };
}
async function dispatch(data) {
  if (data.type === "load") {
    await init();
    const response = await cachedFetch(`${data.base}${spec.file}`, progress);
    const blob = await response.blob();
    if (blob.size !== spec.bytes) {
      await idbCache.delete(`${data.base}${spec.file}`).catch(() => {});
      throw Error(`Incomplete Bonsai model: expected ${spec.bytes} bytes, got ${blob.size}. Retry the download.`);
    }
    progress({ status: "ready" });
    wasm = await createBonsai();
    wasm.FS.mkdir("/model");
    wasm.FS.mount(wasm.WORKERFS, { blobs: [{ name: "bonsai.gguf", data: blob }] }, "/model");
    check(wasm.ccall("bonsai_load", "number", ["string"], ["/model/bonsai.gguf"]));
    labels = [..."ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"].map(label => {
      const ids = tokenize(label, false);
      if (ids.length !== 1) throw Error(`Bonsai label is not a single token: ${label}`);
      return ids[0];
    });
    return {};
  }
  if (!wasm || !labels) throw Error("Load Bonsai before running a request");
  const { request, options: { temperature = 1, calibrate = false, orders = 1 } } = data;
  if (!Number.isFinite(temperature) || temperature <= 0) throw Error("Temperature must be positive and finite");
  const start = performance.now(), json = JSON.stringify(request), cap = labels.length;
  const plan = JSON.parse(omg.plan(json, layout, orders, cap));
  const cfPrefix = calibrate ? JSON.parse(omg.plan(JSON.stringify({ ...request, state: typeof calibrate === "string" ? calibrate : "N/A" }), layout, orders, cap)).prefix : null;
  // Preflight both plans, including every question, before any inference.
  preparePlan(plan, tokenize, cap);
  if (cfPrefix) preparePlan({ prefix: cfPrefix, branches: plan.branches }, tokenize, cap);
  const base = cfPrefix ? rowsFor({ prefix: cfPrefix, branches: plan.branches }) : null;
  const r = rowsFor(plan), rows = JSON.stringify(r.rows), baseRows = base ? JSON.stringify(base.rows) : undefined;
  const second = JSON.parse(omg.plan_second(json, layout, orders, cap, rows, temperature, baseRows));
  const base2 = cfPrefix && second.branches.length ? rowsFor({ prefix: cfPrefix, branches: second.branches }) : null;
  const r2 = second.branches.length ? rowsFor({ prefix: plan.prefix, branches: second.branches }) : null;
  const tokens = r.tokens + (base?.tokens ?? 0) + (r2?.tokens ?? 0) + (base2?.tokens ?? 0);
  const forwards = r.forwards + (base?.forwards ?? 0) + (r2?.forwards ?? 0) + (base2?.forwards ?? 0);
  const out = JSON.parse(omg.answer(json, layout, orders, cap, rows, r2 ? JSON.stringify(r2.rows) : undefined, temperature, spec.id, tokens, baseRows, base2 ? JSON.stringify(base2.rows) : undefined));
  return { ...out.response, usage: { ...out.response.usage, ms: performance.now() - start, mode: "shared", state_tokens: r.state_tokens, state_resident: r.warm, state_source: r.warm ? "resident" : "cold",
    forwards, questions: Object.keys(request.questions).length, branches: r.rows.length + (r2?.rows.length ?? 0), orders, passes: 1 + Number(Boolean(r2)), ...(base ? { calibrated: "contextual", baseline_forwards: base.forwards + (base2?.forwards ?? 0) } : {}) },
    diagnostics: { ...out.diagnostics, rows: r.rows, ...(r2 ? { rows2: r2.rows } : {}), ...(base ? { baseline: base.rows } : {}) } };
}
// Heavy decode runs off the UI thread; one request owns the recurrent state at a time.
let queue = Promise.resolve();
self.onmessage = ({ data }) => {
  queue = queue.then(async () => {
    try { self.postMessage({ id: data.id, result: await dispatch(data) }); }
    catch (error) { self.postMessage({ id: data.id, error: error.message }); }
  });
};
