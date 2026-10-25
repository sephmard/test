// HTTP API built on web-standard Request/Response, shared by the Worker and the Node server.

import {
  LIMITS,
  ValidationError,
  buildIcs,
  buildSlots,
  normalizeAvailability,
  normalizeEventInput,
  normalizeName,
} from '../public/core.js';

const MAX_BODY = 200_000;
const ID_ALPHABET = 'abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789';

function randomId(len) {
  const bytes = crypto.getRandomValues(new Uint8Array(len));
  let s = '';
  for (const b of bytes) s += ID_ALPHABET[b % ID_ALPHABET.length];
  return s;
}

async function sha256(text) {
  const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

function timingSafeEqual(a, b) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

async function tokenMatches(token, hash) {
  return typeof token === 'string' && token.length > 0 && timingSafeEqual(await sha256(token), hash);
}

const json = (data, status = 200) =>
  new Response(JSON.stringify(data), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' },
  });

const error = (status, message) => json({ error: message }, status);

async function readJson(request) {
  const text = await request.text();
  if (text.length > MAX_BODY) throw new ValidationError('Request is too large');
  try {
    return JSON.parse(text || '{}');
  } catch {
    throw new ValidationError('Invalid JSON');
  }
}

function publicEvent(ev, participants, isAdmin) {
  const { adminHash, ...rest } = ev;
  return {
    ...rest,
    slots: buildSlots(ev),
    isAdmin,
    participants: participants.map(({ editHash, ...p }) => p),
  };
}

export async function handleApi(request, store, { now = Date.now } = {}) {
  const url = new URL(request.url);
  const parts = url.pathname.replace(/^\/api\/?/, '').split('/').filter(Boolean);
  const method = request.method;

  try {
    // POST /api/events
    if (parts.length === 1 && parts[0] === 'events' && method === 'POST') {
      const event = normalizeEventInput(await readJson(request));
      const id = randomId(10);
      const adminToken = randomId(24);
      await store.createEvent({ id, event, adminHash: await sha256(adminToken), now: now() });
      return json({ id, adminToken }, 201);
    }

    if (parts[0] !== 'events' || !parts[1]) return error(404, 'Not found');
    const ev = await store.getEvent(parts[1]);
    if (!ev) return error(404, 'This poll does not exist or was deleted');
    const isAdmin = await tokenMatches(request.headers.get('x-admin-token'), ev.adminHash);

    // GET|DELETE /api/events/:id
    if (parts.length === 2) {
      if (method === 'GET') return json(publicEvent(ev, await store.listParticipants(ev.id), isAdmin));
      if (method === 'DELETE') {
        if (!isAdmin) return error(403, 'Only the organizer can delete this poll');
        await store.deleteEvent(ev.id);
        return json({ ok: true });
      }
      return error(405, 'Method not allowed');
    }

    // GET /api/events/:id/event.ics
    if (parts.length === 3 && parts[2] === 'event.ics' && method === 'GET') {
      if (ev.finalStart == null) return error(404, 'No time has been picked yet');
      const body = buildIcs({
        id: ev.id,
        title: ev.title,
        description: ev.description,
        start: ev.finalStart,
        end: ev.finalStart + ev.durationMinutes * 60_000,
        url: `${url.origin}/e/${ev.id}`,
      });
      return new Response(body, {
        headers: {
          'content-type': 'text/calendar; charset=utf-8',
          'content-disposition': `attachment; filename="${ev.id}.ics"`,
        },
      });
    }

    // POST /api/events/:id/finalize  { start: number | null }
    if (parts.length === 3 && parts[2] === 'finalize' && method === 'POST') {
      if (!isAdmin) return error(403, 'Only the organizer can pick the final time');
      const { start } = await readJson(request);
      if (start !== null && !buildSlots(ev).includes(start)) throw new ValidationError('That time is not part of this poll');
      await store.setFinal(ev.id, start);
      return json({ ok: true, finalStart: start });
    }

    if (parts[2] !== 'participants') return error(404, 'Not found');
    const slots = buildSlots(ev);

    // POST /api/events/:id/participants
    if (parts.length === 3 && method === 'POST') {
      const body = await readJson(request);
      const name = normalizeName(body.name);
      const availability = normalizeAvailability(body.availability, slots);
      if ((await store.countParticipants(ev.id)) >= LIMITS.participants)
        throw new ValidationError('This poll has reached the participant limit');
      const id = randomId(10);
      const editToken = randomId(24);
      await store.insertParticipant({
        id,
        eventId: ev.id,
        name,
        availability,
        editHash: await sha256(editToken),
        now: now(),
      });
      return json({ id, editToken }, 201);
    }

    // PUT|DELETE /api/events/:id/participants/:pid
    if (parts.length === 4) {
      const p = await store.getParticipant(ev.id, parts[3]);
      if (!p) return error(404, 'Response not found');
      const isOwner = await tokenMatches(request.headers.get('x-edit-token'), p.editHash);
      if (method === 'PUT') {
        if (!isOwner) return error(403, 'You can only edit your own response');
        const body = await readJson(request);
        await store.updateParticipant({
          id: p.id,
          eventId: ev.id,
          name: normalizeName(body.name),
          availability: normalizeAvailability(body.availability, slots),
          now: now(),
        });
        return json({ ok: true });
      }
      if (method === 'DELETE') {
        if (!isOwner && !isAdmin) return error(403, 'You can only remove your own response');
        await store.deleteParticipant(ev.id, p.id);
        return json({ ok: true });
      }
      return error(405, 'Method not allowed');
    }

    return error(404, 'Not found');
  } catch (e) {
    if (e instanceof ValidationError) return error(400, e.message);
    console.error(e);
    return error(500, 'Something went wrong');
  }
}
