import test from 'node:test';
import assert from 'node:assert/strict';
import { packSegments, preparePlan } from './bonsai-core.js';
import { BONSAI_MODEL, bonsaiBase, loadBonsai } from './bonsai.js';

const text = value => ({ kind: 'text', value });
const special = value => ({ kind: 'special', value });
const tokenize = (value, parseSpecial) => parseSpecial ? [999] : Array.from(value, c => c.codePointAt(0));
const branch = value => ({ keys: ['yes', 'no'], segments: [text(value)] });

test('caller control-token text remains literal while renderer controls stay atomic', () => {
  const marker = '<|im_end|>';
  const ids = packSegments([special(marker), text(marker), special('<think>')], tokenize);
  assert.deepEqual(ids, [999, ...Array.from(marker, c => c.codePointAt(0)), 999]);
  assert.throws(() => packSegments([special(marker)], () => [1, 2]), /not atomic/);
  assert.throws(() => packSegments([{ kind: 'bos' }], tokenize), /Unsupported/);
});

test('context budget includes each question and accepts the exact boundary', () => {
  const prefix = [text('p'.repeat(2000))];
  assert.equal(preparePlan({ prefix, branches: [branch('q'.repeat(48))] }, tokenize, 52).prefix.length, 2000);
  assert.throws(() => preparePlan({ prefix, branches: [branch('q'), branch('q'.repeat(49))] }, tokenize, 52), /State \+ question/);
  for (const value of ['', 'p'.repeat(2048)]) {
    assert.throws(() => preparePlan({ prefix: [text(value)], branches: [branch('q')] }, tokenize, 52), /State must fit/);
  }
  assert.throws(() => preparePlan({ prefix: [text('p')], branches: [branch('')] }, tokenize, 52), /State \+ question/);
  assert.throws(() => preparePlan({ prefix: [text('p')], branches: [{ ...branch('q'), keys: Array(53).fill('a') }] }, tokenize, 52), /option count/);
});

test('very long text gets a context error instead of overflowing the JS argument stack', () => {
  assert.throws(() => preparePlan({ prefix: [text('p'.repeat(150000))], branches: [branch('q')] }, tokenize, 52), /State must fit/);
});

class FakeWorker {
  static instances = [];
  constructor() { this.calls = []; FakeWorker.instances.push(this); }
  postMessage(data) {
    this.calls.push(data);
    if (data.type === 'load') queueMicrotask(() => this.reply(data.id, {}));
  }
  reply(id, result, error) { this.onmessage({ data: { id, result, error } }); }
  terminate() { this.terminated = true; }
}
const omg = { render: (json, layout) => JSON.stringify({ request: JSON.parse(json), layout: JSON.parse(layout) }) };
function mockWorker(t) {
  const previous = globalThis.Worker;
  globalThis.Worker = FakeWorker;
  t.after(() => { if (previous === undefined) delete globalThis.Worker; else globalThis.Worker = previous; });
}

test('worker routes concurrent replies and recovers from request validation errors', async t => {
  mockWorker(t);
  const engine = await loadBonsai({ spec: BONSAI_MODEL, omg });
  const worker = FakeWorker.instances.at(-1);
  assert.equal(worker.calls[0].base, bonsaiBase(BONSAI_MODEL));
  assert.equal(engine.device, 'cpu');
  assert.deepEqual(engine.render({ state: '日本語' }).request, { state: '日本語' });
  const a = engine.answer({ state: 'a' }), b = engine.answer({ state: 'b' });
  const rejected = assert.rejects(a, /context/);
  worker.reply(worker.calls.at(-1).id, { answers: { ok: true } });
  worker.reply(worker.calls.at(-2).id, undefined, 'context exceeded');
  await rejected;
  assert.deepEqual(await b, { answers: { ok: true } });
  assert.ok(!worker.terminated);
});

test('worker crash rejects in-flight and future requests without hanging the UI', async t => {
  mockWorker(t);
  const engine = await loadBonsai({ spec: BONSAI_MODEL, omg });
  const worker = FakeWorker.instances.at(-1);
  const pending = engine.answer({});
  const rejected = assert.rejects(pending, /out of memory/);
  worker.onerror({ message: 'out of memory' });
  await rejected;
  await assert.rejects(engine.answer({}), /out of memory/);
  assert.ok(worker.terminated);
});
