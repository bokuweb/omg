// Official PTQ weights and the Prism runtime are pinned together.
export const BONSAI_MODEL = {
  id: "bonsai-2-4b-wasm", kind: "bonsai", readout: "label", dtype: "PTQ1_0",
  hub: "prism-ml/Ternary-Bonsai-2-4B-gguf",
  revision: "8d6557168168f91ea6697a4a49ed2ef622b53adb",
  file: "Ternary-Bonsai-2-4B-PTQ1_0.gguf", bytes: 949549920,
  size: "950 MB", note: "Bonsai 2 4B: WebGPU / CPU, 2,048-token context",
  layout: { layout: "label", turn_start: "<|im_start|>", turn_end: "<|im_end|>", user: "user", model: "assistant", bos: false,
    assistant_prefix: [{ kind: "special", value: "<think>" }, { kind: "text", value: "\n\n" }, { kind: "special", value: "</think>" }, { kind: "text", value: "\n\n" }] },
};
export const bonsaiBase = spec => `https://huggingface.co/${spec.hub}/resolve/${spec.revision}/`;

export async function loadBonsai({ spec, onProgress, modelBase, omg, device = "webgpu" }) {
  const worker = new Worker(new URL("./bonsai-worker.js", import.meta.url), { type: "module" });
  let nextId = 0, failed;
  const pending = new Map();
  const fail = error => {
    failed = error;
    for (const p of pending.values()) p.reject(error);
    pending.clear();
    worker.terminate();
  };
  worker.onerror = event => fail(new Error(event.message || "Bonsai worker failed"));
  worker.onmessageerror = () => fail(new Error("Could not read the Bonsai worker response"));
  worker.onmessage = ({ data }) => {
    if (data.progress) { onProgress?.(data.progress); return; }
    const p = pending.get(data.id);
    if (!p) return;
    pending.delete(data.id);
    if (data.error) p.reject(new Error(data.error));
    else p.resolve(data.result);
  };
  const call = data => new Promise((resolve, reject) => {
    if (failed) return reject(failed);
    const id = nextId++;
    pending.set(id, { resolve, reject });
    try { worker.postMessage({ ...data, id }); }
    catch (error) { pending.delete(id); reject(error); }
  });
  try {
    const base = modelBase ? new URL(`${modelBase.replace(/\/?$/, "/")}${spec.id}/`, location.href).href : bonsaiBase(spec);
    await call({ type: "load", base, device });
  } catch (error) { fail(error); throw error; }
  return { model: spec.id, spec, device, labels: [..."ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"],
    render: request => JSON.parse(omg.render(JSON.stringify(request), JSON.stringify(spec.layout))),
    answer: (request, options = {}) => call({ type: "answer", request, options }),
  };
}
