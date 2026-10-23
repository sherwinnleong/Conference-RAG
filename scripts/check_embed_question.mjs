import assert from 'node:assert/strict';

const env = { SUPABASE_URL: 'https://example.com', SUPABASE_ANON_KEY: 'public-test-key', GEMINI_API_KEY: 'fake-test-key' };
globalThis.Deno = { env: { get: key => env[key] }, serve: () => {} };
const { handler } = await import('../supabase/functions/embed-question/index.ts');
let calls = [];
let mode = 'success';
globalThis.fetch = async (url, options) => {
  calls.push({ url, options });
  if (url.endsWith('/auth/v1/user')) {
    return Response.json(mode === 'unauthorized' ? {} : { id: 'test-user' }, { status: mode === 'unauthorized' ? 401 : 200 });
  }
  if (mode === 'quota') return Response.json({}, { status: 429 });
  return Response.json({ embedding: { values: Array(mode === 'invalid-vector' ? 3 : 1536).fill(1) } });
};
const request = (body, token = true) => new Request('https://example.com/embed-question', {
  method: 'POST', headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: 'Bearer test-user-token' } : {}) }, body: JSON.stringify(body),
});
assert.equal((await handler(new Request('https://example.com', { method: 'OPTIONS' }))).status, 204);
assert.equal(calls.length, 0);
assert.equal((await handler(request({ question: 'test' }, false))).status, 401);
assert.equal(calls.length, 0);
mode = 'unauthorized';
assert.equal((await handler(request({ question: 'test' }))).status, 401);
assert.equal(calls.length, 1);
mode = 'success'; calls = [];
assert.equal((await handler(request({ question: '  ' }))).status, 400);
assert.equal(calls.length, 1);
calls = [];
const response = await handler(request({ question: 'How can I find peace?' }));
assert.equal(response.status, 200);
assert.equal(response.headers.get('Access-Control-Allow-Origin'), '*');
const { embedding } = await response.json();
assert.equal(embedding.length, 1536);
assert.ok(Math.abs(embedding.reduce((s,x) => s+x*x,0)-1) < 1e-10);
const sent = JSON.parse(calls[1].options.body);
assert.equal(sent.taskType, 'RETRIEVAL_QUERY');
assert.equal(sent.outputDimensionality, 1536);
mode = 'quota';
assert.equal((await handler(request({ question: 'test' }))).status, 429);
mode = 'invalid-vector';
assert.equal((await handler(request({ question: 'test' }))).status, 502);
console.log('Passed: CORS, authentication, input validation, Gemini request, normalized vector, quota and malformed-vector handling. No live services called.');
