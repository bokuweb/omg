import { preparePlan } from './bonsai-core.js';

// Experimental adapter for the pinned upstream browser runtime. Keep the digest
// and the small logits patch together; refuse to execute a different bundle.
const RUNTIME = 'https://huggingface.co/spaces/webml-community/ternary-bonsai-2-hermes-webgpu/resolve/621e20782f208660f4f58283c2ce58430020f243/assets/qwen35-vision-Kzp5MMIJ.js';
const SHA256 = '00c1a72a962a84fee17000a6b1d18b24d75033e56d59996fc504b7c57cfc6488';
async function runtimeLoader() {
  const response = await fetch(RUNTIME);
  if (!response.ok) throw Error(`Bonsai GPU runtime download failed: ${response.status}`);
  const bytes = await response.arrayBuffer();
  const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(x => x.toString(16).padStart(2, '0')).join('');
  if (hash !== SHA256) throw Error('Bonsai GPU runtime checksum mismatch');
  const source = new TextDecoder().decode(bytes);
  const needle = 'function _w(e,t){let n=e.nodeCount;if(gw(t.model))';
  if (source.split(needle).length !== 2) throw Error('Bonsai GPU logits adapter does not match the pinned runtime');
  // The chat demo normally fuses its head to argmax. Decisions need logits.
  const patched = source.replace(needle, 'function _w(e,t){let n=e.nodeCount;if(true)') + '\nexport async function loadRaw(url,options={}){return h_(GD(options),e=>new xu(e),url,options)}';
  const url = URL.createObjectURL(new Blob([patched], { type: 'text/javascript' }));
  try { return (await import(url)).loadRaw; }
  finally { URL.revokeObjectURL(url); }
}

export async function createGpuBonsai(blob, url, progress) {
  if (!navigator.gpu) throw Error('WebGPU unavailable. Select CPU (WASM) to use Bonsai.');
  progress({ status: 'initializing', message: 'Preparing Bonsai WebGPU runtime…' });
  const loadRaw = await runtimeLoader();
  // Serve GGUF ranges from the same cached Blob used by the CPU backend.
  const rangeFetch = async (input, options = {}) => {
    if (String(input) !== url) throw Error('Unexpected Bonsai model resource');
    const range = new Headers(options.headers).get('range');
    if (!range) return new Response(blob, { headers: { 'content-length': String(blob.size) } });
    const match = /^bytes=(\d+)-(\d*)$/.exec(range);
    if (!match) throw Error('Invalid Bonsai model range');
    const a = +match[1], b = match[2] ? +match[2] : blob.size - 1;
    if (a > b || b >= blob.size) throw Error('Bonsai model range out of bounds');
    return new Response(blob.slice(a, b + 1), { status: 206, headers: { 'content-length': String(b - a + 1), 'content-range': `bytes ${a}-${b}/${blob.size}` } });
  };
  const model = await loadRaw(url, { fetch: rangeFetch, maxLength: 2048, decodePipelineDepth: 1, prefixSnapshotStore: null,
    onProgress: p => progress({ status: 'initializing', message: p.message || 'Initializing Bonsai WebGPU…' }) });
  const tokenizer = model.tokenizer;
  function tokenize(text, special) {
    if (special) {
      const id = tokenizer.token_to_id(text);
      if (id == null) throw Error(`Unknown Bonsai control token: ${text}`);
      return [id];
    }
    // encode() recognizes added special tokens even inside caller text. Bypass
    // that recognition for literal segments, just as llama_tokenize does.
    if (tokenizer.normalizer) text = tokenizer.normalizer(text);
    const pieces = tokenizer.pre_tokenizer ? tokenizer.pre_tokenizer(text, { section_index: 0 }) : [text];
    return tokenizer.model(pieces).map(t => tokenizer.model.tokens_to_ids.get(t) ?? tokenizer.model.unk_token_id);
  }
  const labels = [...'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'].map(label => {
    const ids = tokenize(label, false);
    if (ids.length !== 1) throw Error(`Bonsai label is not a single token: ${label}`);
    return ids[0];
  });
  const cache = model.generationState.cache;
  cache.setRewindPointCapacity(1);
  const session = model.model.language_model.makeGraphDecodeSession(cache);
  await session.build(0, { fuse: true });
  const head = session.compiled.tensors.get(session.emission.graph.tensors.find(t => t.name === 'prism.head.logits')?.id);
  if (!head) throw Error('Bonsai GPU runtime did not expose logits');
  let resident = null;
  async function decode(ids, start, full) {
    for (let i = 0; i < ids.length; i++) session.enqueueStep(ids[i], start + i, { full: full && i === ids.length - 1 });
    await model.runtime.queueIdle();
    cache.advance(ids.length);
  }
  async function rowsFor(plan) {
    const { prefix, branches } = preparePlan(plan, tokenize, labels.length);
    const warm = Boolean(resident && resident.length === prefix.length && resident.every((id, i) => id === prefix[i]));
    try {
      if (!warm) {
        cache.truncate(0);
        await decode(prefix, 0, false);
        await cache.captureRewindPoint();
        resident = prefix;
      }
      const rows = [];
      for (const { ids, count } of branches) {
        cache.truncate(prefix.length);
        await decode(ids, prefix.length, true);
        const scores = await model.runtime.readTensor(head);
        let max = -Infinity;
        for (const score of scores) max = Math.max(max, score);
        let total = 0, mass = 0;
        for (const score of scores) total += Math.exp(score - max);
        const logits = labels.slice(0, count).map(id => scores[id]);
        for (const score of logits) mass += Math.exp(score - max);
        if (!Number.isFinite(total) || total <= 0 || logits.some(x => !Number.isFinite(x))) throw Error('Bonsai GPU produced invalid logits');
        rows.push({ logits, candidate_mass: mass / total });
      }
      cache.truncate(prefix.length);
      return { rows, warm, state_tokens: prefix.length, tokens: prefix.length + branches.reduce((n, b) => n + b.ids.length, 0), forwards: branches.length + Number(!warm) };
    } catch (error) { resident = null; cache.truncate(0); throw error; }
  }
  return { tokenize, labels, rowsFor };
}
