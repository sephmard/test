import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createApp } from '../server.js';

let server, baseUrl;

before(async () => {
  server = createApp({ dbPath: ':memory:' });
  await new Promise((r) => server.listen(0, r));
  baseUrl = `http://localhost:${server.address().port}`;
});
after(() => server.close());

async function call(path, { method = 'GET', body, headers = {} } = {}) {
  const res = await fetch(baseUrl + path, {
    method,
    headers: { 'content-type': 'application/json', ...headers },
    body: body && JSON.stringify(body),
  });
  const text = await res.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = text;
  }
  return { status: res.status, data };
}

const newEvent = {
  title: 'Coffee <script>',
  organizer: 'Sam',
  timezone: 'Europe/London',
  dates: ['2026-11-02'],
  startMinute: 600,
  endMinute: 720,
  slotMinutes: 30,
  durationMinutes: 60,
};

test('full poll lifecycle', async () => {
  const created = await call('/api/events', { method: 'POST', body: newEvent });
  assert.equal(created.status, 201);
  const { id, adminToken } = created.data;

  let ev = (await call(`/api/events/${id}`)).data;
  assert.equal(ev.title, 'Coffee <script>');
  assert.equal(ev.slots.length, 4);
  assert.equal(ev.isAdmin, false);
  assert.equal(ev.adminHash, undefined);
  assert.equal((await call(`/api/events/${id}`, { headers: { 'x-admin-token': adminToken } })).data.isAdmin, true);

  const [s0, s1] = ev.slots;
  const ann = await call(`/api/events/${id}/participants`, { method: 'POST', body: { name: 'Ann', availability: { [s0]: 2, [s1]: 1 } } });
  assert.equal(ann.status, 201);
  const bob = await call(`/api/events/${id}/participants`, { method: 'POST', body: { name: 'Bob', availability: {} } });

  // Bob can't edit Ann's response, Ann can.
  const annPath = `/api/events/${id}/participants/${ann.data.id}`;
  assert.equal((await call(annPath, { method: 'PUT', body: { name: 'Ann', availability: {} }, headers: { 'x-edit-token': bob.data.editToken } })).status, 403);
  assert.equal((await call(annPath, { method: 'PUT', body: { name: 'Ann B', availability: { [s0]: 2 } }, headers: { 'x-edit-token': ann.data.editToken } })).status, 200);

  ev = (await call(`/api/events/${id}`)).data;
  assert.deepEqual(ev.participants.map((p) => p.name), ['Bob', 'Ann B']);
  assert.equal(ev.participants[0].editHash, undefined);
  assert.deepEqual(ev.participants[1].availability, { [s0]: 2 });

  // Finalize: organizer only, must be a real slot.
  const fin = `/api/events/${id}/finalize`;
  assert.equal((await call(fin, { method: 'POST', body: { start: s0 } })).status, 403);
  assert.equal((await call(fin, { method: 'POST', body: { start: 5 }, headers: { 'x-admin-token': adminToken } })).status, 400);
  assert.equal((await call(fin, { method: 'POST', body: { start: s0 }, headers: { 'x-admin-token': adminToken } })).status, 200);
  const ics = await call(`/api/events/${id}/event.ics`);
  assert.match(ics.data, /BEGIN:VEVENT/);
  assert.match(ics.data, /DTSTART:20261102T100000Z/);

  // Organizer can remove someone; then delete the poll.
  assert.equal((await call(`/api/events/${id}/participants/${bob.data.id}`, { method: 'DELETE', headers: { 'x-admin-token': adminToken } })).status, 200);
  assert.equal((await call(`/api/events/${id}`, { method: 'DELETE' })).status, 403);
  assert.equal((await call(`/api/events/${id}`, { method: 'DELETE', headers: { 'x-admin-token': adminToken } })).status, 200);
  assert.equal((await call(`/api/events/${id}`)).status, 404);
});

test('validation errors come back as 400', async () => {
  const r = await call('/api/events', { method: 'POST', body: { ...newEvent, dates: [] } });
  assert.equal(r.status, 400);
  assert.match(r.data.error, /date/);
  const created = await call('/api/events', { method: 'POST', body: newEvent });
  const p = await call(`/api/events/${created.data.id}/participants`, { method: 'POST', body: { name: '', availability: {} } });
  assert.equal(p.status, 400);
});

test('serves the app for client-side routes', async () => {
  const res = await fetch(`${baseUrl}/e/whatever`);
  assert.equal(res.status, 200);
  assert.match(await res.text(), /<main id="app"/);
  const js = await fetch(`${baseUrl}/core.js`);
  assert.match(js.headers.get('content-type'), /javascript/);
  const traversal = await fetch(`${baseUrl}/..%2f..%2fpackage.json`);
  assert.doesNotMatch(await traversal.text(), /"scripts"/);
});
