import test from 'node:test';
import assert from 'node:assert/strict';
import { decisionFor, tokenize, feedsFor, predict } from './bekko-core.js';

const manifest = { query_length: 4096, document_length: 2048, cls_token_id: 50281, sep_token_id: 50282, pad_token_id: 50283, tasks: ['choice', 'noul', 'score'] };
const tokenizer = { encode: text => ({ ids: Array.from(text, c => c.charCodeAt(0)) }) };
const ort = { Tensor: class { constructor(type, data, dims) { Object.assign(this, { type, data, dims }); } } };
const choice = () => decisionFor({ review: 'Great' }, { type: 'choice', instructions: 'Sentiment?', criteria: { bad: 'Negative', good: 'Positive' } }, ['good', 'bad']);

test('choice preserves core branch order and upstream prompt format', () => {
  assert.deepEqual(choice(), {
    task: 'choice', instruction: 'Sentiment?', state: '{"review":"Great"}',
    candidates: [{ id: 'good', text: 'Candidate: good: Positive' }, { id: 'bad', text: 'Candidate: bad: Negative' }],
  });
  const tokens = tokenize(choice(), tokenizer, manifest);
  assert.equal(tokens.prefix_ids[0][0], 50281);
  assert.equal(tokens.prefix_ids[0].at(-1), 50282);
  assert.equal(String.fromCharCode(...tokens.prefix_ids[0].slice(1, -1)), 'Instruction: Sentiment?\nState: {"review":"Great"}');
  assert.equal(String.fromCharCode(...tokens.doc_ids[0].slice(0, -1)), 'Candidate: good: Positive');
});

test('noul wraps custom meanings and defaults without changing boolean order', () => {
  const custom = decisionFor('text', { type: 'noul', criteria: { true: 'Supported', false: 'Unsupported' } }, ['false', 'true']);
  assert.equal(custom.state, '{"noul":{"yes":"Supported","no":"Unsupported"},"state":"text"}');
  assert.deepEqual(custom.candidates.map(c => c.text), ['Candidate: false: Unsupported', 'Candidate: true: Supported']);
  const defaults = decisionFor(null, { type: 'noul' }, ['true', 'false']);
  assert.equal(defaults.instruction, 'Is the statement true?');
  assert.deepEqual(JSON.parse(defaults.state), { noul: { yes: 'Yes, the condition in the question holds.', no: 'No, the condition in the question does not hold.' }, state: null });
});

test('score uses core levels and serializes structured instructions/criteria', () => {
  const request = decisionFor('ok', { type: 'score', instructions: { rating: 'quality' }, criteria: { 0: { meaning: 'bad' }, 1: 'fine', 2: 'great' } }, ['0', '1', '2']);
  assert.equal(request.instruction, '{"rating":"quality"}');
  assert.equal(request.state, '"ok"');
  assert.deepEqual(request.candidates, [{ id: '0', text: 'Candidate: 0: {"meaning":"bad"}', value: 0 }, { id: '1', text: 'Candidate: 1: fine', value: 1 }, { id: '2', text: 'Candidate: 2: great', value: 2 }]);
  assert.deepEqual(decisionFor({}, { type: 'score' }, ['0', '1']).candidates.map(c => c.text), ['Candidate: 0: 0', 'Candidate: 1: 1']);
});

test('rejects unsupported option counts before inference', () => {
  for (const count of [1, 65]) assert.throws(() => decisionFor({}, { type: 'choice' }, Array.from({ length: count }, (_, i) => String(i))), /2–64/);
  assert.equal(decisionFor({}, { type: 'choice' }, Array.from({ length: 64 }, (_, i) => String(i))).candidates.length, 64);
});

test('truncation budgets preserve both instruction and state and terminal separators', () => {
  const request = { ...choice(), instruction: 'i'.repeat(100), state: 's'.repeat(100) };
  const tokens = tokenize(request, tokenizer, { ...manifest, query_length: 32, document_length: 8 });
  assert.equal(tokens.prefix_ids[0].length, 32);
  assert.equal(String.fromCharCode(...tokens.prefix_ids[0].slice(1, -1)), 'Instruction: iiiii\nState: ssss');
  assert.ok(tokens.doc_ids.every(row => row.length === 8 && row.at(-1) === 50282));
  const short = tokenize({ ...request, instruction: 'i' }, tokenizer, { ...manifest, query_length: 32 });
  assert.equal(String.fromCharCode(...short.prefix_ids[0].slice(1, -1)), 'Instruction: i\nState: ssssssss');
});

test('feeds use int64 IDs/owners, boolean masks and masked padding', () => {
  const feeds = feedsFor({ prefix_ids: [[50281, 1, 50282]], doc_ids: [[7, 50282], [8, 9, 50282]] }, ort, manifest);
  assert.equal(feeds.doc_ids.type, 'int64');
  assert.deepEqual(feeds.doc_ids.dims, [2, 3]);
  assert.deepEqual(Array.from(feeds.doc_ids.data), [7n, 50282n, 50283n, 8n, 9n, 50282n]);
  assert.equal(feeds.doc_mask.type, 'bool');
  assert.deepEqual(Array.from(feeds.doc_mask.data), [1, 1, 0, 1, 1, 1]);
  assert.deepEqual(Array.from(feeds.owners.data), [0n, 0n]);
});

test('each task reads its own head in manifest order', async () => {
  const reordered = { ...manifest, tasks: ['score', 'choice', 'noul'] };
  const session = { run: async () => ({ logits: { dims: [2, 3], data: Float32Array.from([11, 12, 13, 21, 22, 23]) } }) };
  const cases = [[choice(), [12, 22]], [decisionFor({}, { type: 'noul' }, ['true', 'false']), [13, 23]], [decisionFor({}, { type: 'score' }, ['0', '1']), [11, 21]]];
  for (const [request, expected] of cases) {
    const result = await predict(request, { tokenizer, ort, manifest: reordered, session });
    assert.deepEqual(result.logits, expected);
    assert.equal(result.forwards, 1);
    const tokens = tokenize(request, tokenizer, reordered);
    assert.equal(result.inputTokens, tokens.prefix_ids[0].length + tokens.doc_ids.flat().length);
  }
});

test('long candidates batch independently while preserving result order', async () => {
  const request = { ...choice(), instruction: 'i'.repeat(3000), state: 's'.repeat(3000), candidates: Array.from({ length: 3 }, (_, i) => ({ id: String(i), text: 'x'.repeat(3000) })) };
  let calls = 0;
  const session = { run: async feeds => {
    assert.equal(feeds.doc_ids.dims[0], 1);
    return { logits: { dims: [1, 3], data: [++calls, -1, -2] } };
  } };
  const result = await predict(request, { tokenizer, ort, manifest, session });
  assert.deepEqual(result, { logits: [1, 2, 3], forwards: 3, inputTokens: 3 * (4096 + 2048) });
});

test('invalid outputs fail explicitly', async () => {
  for (const logits of [{ dims: [2, 2], data: [1, 2, 3, 4] }, { dims: [2, 3], data: [NaN, 0, 0, 1, 0, 0] }]) {
    await assert.rejects(predict(choice(), { tokenizer, ort, manifest, session: { run: async () => ({ logits }) } }), /Invalid model logits/);
  }
});
