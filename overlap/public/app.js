import {
  YES,
  IF_NEEDED,
  MINUTE,
  addDays,
  availabilityFromBusy,
  googleCalendarUrl,
  outlookCalendarUrl,
  pad,
  parseIcsBusy,
  rankTimes,
  utcToZoned,
} from './core.js';

const app = document.getElementById('app');
const localTz = Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';

// ---------- small helpers ----------

const esc = (s) =>
  String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);

const $ = (sel, root = document) => root.querySelector(sel);

function saved(id) {
  try {
    return JSON.parse(localStorage.getItem(`overlap:${id}`)) || {};
  } catch {
    return {};
  }
}
function save(id, patch) {
  try {
    localStorage.setItem(`overlap:${id}`, JSON.stringify({ ...saved(id), ...patch }));
  } catch {}
}
function forget(id, ...keys) {
  const s = saved(id);
  for (const k of keys) delete s[k];
  try {
    localStorage.setItem(`overlap:${id}`, JSON.stringify(s));
  } catch {}
}
function lastName() {
  try {
    return localStorage.getItem('overlap:name') || '';
  } catch {
    return '';
  }
}
function rememberName(n) {
  try {
    localStorage.setItem('overlap:name', n);
  } catch {}
}

let toastTimer;
function toast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 2800);
}

async function api(path, { method = 'GET', body, headers = {} } = {}) {
  const res = await fetch(`/api/${path}`, {
    method,
    headers: { ...(body ? { 'content-type': 'application/json' } : {}), ...headers },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.error || `Request failed (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return data;
}

async function copy(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast('Link copied');
  } catch {
    prompt('Copy this link:', text);
  }
}

const todayStr = () => {
  const d = new Date();
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
};

function dateParts(dateStr) {
  const [y, m, d] = dateStr.split('-').map(Number);
  return new Date(Date.UTC(y, m - 1, d));
}
const fmtDow = new Intl.DateTimeFormat(undefined, { weekday: 'short', timeZone: 'UTC' });
const fmtMonthDay = new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric', timeZone: 'UTC' });
const fmtMinutes = new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit', timeZone: 'UTC' });
const minutesLabel = (m) => fmtMinutes.format(new Date(Date.UTC(2000, 0, 1, 0, m)));
const lengthLabel = (min) => (min < 60 ? `${min} min` : `${String(min / 60).replace('.5', '½')} hour${min > 60 ? 's' : ''}`);

function fmtRange(start, end, tz) {
  const day = new Intl.DateTimeFormat(undefined, { weekday: 'short', month: 'short', day: 'numeric', timeZone: tz }).format(start);
  const t = new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit', timeZone: tz });
  return `${day} · ${t.format(start)} – ${t.format(end)}`;
}

function tzOptions(selected) {
  let zones = [];
  try {
    zones = Intl.supportedValuesOf('timeZone');
  } catch {}
  for (const z of [selected, localTz, 'UTC']) if (z && !zones.includes(z)) zones.unshift(z);
  return zones.map((z) => `<option value="${esc(z)}"${z === selected ? ' selected' : ''}>${esc(z.replace(/_/g, ' '))}</option>`).join('');
}

// ---------- routing ----------

let cleanup = () => {};

function route() {
  cleanup();
  cleanup = () => {};
  const m = location.pathname.match(/^\/e\/([A-Za-z0-9]+)\/?$/);
  if (m) renderEvent(m[1]);
  else renderCreate();
}

function navigate(path) {
  history.pushState(null, '', path);
  route();
  window.scrollTo(0, 0);
}

document.addEventListener('click', (e) => {
  const a = e.target.closest('a[data-link]');
  if (a && !e.metaKey && !e.ctrlKey && !e.shiftKey) {
    e.preventDefault();
    navigate(a.getAttribute('href'));
  }
});
window.addEventListener('popstate', route);

// ---------- create page ----------

function renderCreate() {
  document.title = 'Overlap: find a time that works';
  const selected = new Set();
  let cursor = new Date();
  cursor = new Date(cursor.getFullYear(), cursor.getMonth(), 1);

  const hourOptions = (sel) =>
    Array.from({ length: 25 }, (_, h) => `<option value="${h * 60}"${h * 60 === sel ? ' selected' : ''}>${minutesLabel(h * 60 % 1440)}${h === 24 ? ' (midnight)' : ''}</option>`).join('');

  app.innerHTML = `
    <section class="hero">
      <h1>Find a time that works for everyone</h1>
      <p class="muted">Pick some dates, share one link, and everyone paints when they're free. Overlap shows you the best time right away. No accounts, no ads.</p>
    </section>
    <form id="create" class="create-grid" novalidate>
      <div class="card stack">
        <div>
          <label for="title">What are you planning?</label>
          <input id="title" type="text" maxlength="120" placeholder="Coffee chat, team sync, book club…" required>
        </div>
        <div>
          <label for="desc">Details <span class="muted">(optional)</span></label>
          <textarea id="desc" maxlength="1000" placeholder="Agenda, location or video link"></textarea>
        </div>
        <div>
          <label for="organizer">Your name <span class="muted">(optional)</span></label>
          <input id="organizer" type="text" maxlength="80" value="${esc(lastName())}">
        </div>
        <div class="row">
          <div><label for="start">From</label><select id="start">${hourOptions(9 * 60)}</select></div>
          <div><label for="end">Until</label><select id="end">${hourOptions(17 * 60)}</select></div>
        </div>
        <div class="row">
          <div>
            <label for="slot">Slot size</label>
            <select id="slot"><option value="15">15 min</option><option value="30" selected>30 min</option><option value="60">1 hour</option></select>
          </div>
          <div><label for="duration">Meeting length</label><select id="duration"></select></div>
        </div>
        <div>
          <label for="tz">Time zone</label>
          <select id="tz">${tzOptions(localTz)}</select>
          <p class="small muted" style="margin:6px 0 0">Everyone sees times converted to their own time zone.</p>
        </div>
      </div>
      <div class="card">
        <div class="cal-head">
          <button type="button" class="btn btn-sm" id="prev" aria-label="Previous month">‹</button>
          <h2 id="month" style="margin:0"></h2>
          <button type="button" class="btn btn-sm" id="next" aria-label="Next month">›</button>
        </div>
        <div class="cal-grid" id="cal" role="group" aria-label="Pick dates"></div>
        <div class="btn-row" style="margin-top:12px">
          <button type="button" class="btn btn-sm" id="weekdays">Next 5 weekdays</button>
          <button type="button" class="btn btn-sm" id="week">Next 7 days</button>
          <button type="button" class="btn btn-sm btn-ghost" id="clear">Clear</button>
        </div>
        <p class="small muted" id="count" style="margin:12px 0 0">Click or drag to pick dates.</p>
        <div class="btn-row" style="margin-top:16px">
          <button class="btn btn-primary" type="submit" id="submit">Create poll</button>
        </div>
        <div class="error" id="err" role="alert"></div>
      </div>
    </form>`;

  const cal = $('#cal');
  const today = todayStr();

  function renderCal() {
    $('#month').textContent = cursor.toLocaleDateString(undefined, { month: 'long', year: 'numeric' });
    const y = cursor.getFullYear(), m = cursor.getMonth();
    const first = new Date(y, m, 1).getDay();
    const days = new Date(y, m + 1, 0).getDate();
    let html = Array.from({ length: 7 }, (_, i) => `<div class="cal-dow">${fmtDow.format(new Date(Date.UTC(2023, 0, 1 + i)))}</div>`).join('');
    html += '<div class="cal-day blank"></div>'.repeat(first);
    for (let d = 1; d <= days; d++) {
      const ds = `${y}-${pad(m + 1)}-${pad(d)}`;
      const past = ds < today;
      html += `<button type="button" class="cal-day${selected.has(ds) ? ' selected' : ''}${ds === today ? ' today' : ''}" data-date="${ds}"${past ? ' disabled' : ''} aria-pressed="${selected.has(ds)}">${d}</button>`;
    }
    cal.innerHTML = html;
    const n = selected.size;
    $('#count').textContent = n ? `${n} date${n > 1 ? 's' : ''} selected` : 'Click or drag to pick dates.';
    $('#prev').disabled = y === new Date().getFullYear() && m === new Date().getMonth();
  }

  // drag to select/deselect a run of dates
  let dragMode = null;
  const apply = (btn) => {
    if (!btn || btn.disabled || !btn.dataset.date) return;
    if (dragMode === 'add') selected.add(btn.dataset.date);
    else selected.delete(btn.dataset.date);
    btn.classList.toggle('selected', dragMode === 'add');
    btn.setAttribute('aria-pressed', dragMode === 'add');
  };
  cal.addEventListener('pointerdown', (e) => {
    const btn = e.target.closest('.cal-day');
    if (!btn || btn.disabled) return;
    e.preventDefault();
    dragMode = selected.has(btn.dataset.date) ? 'remove' : 'add';
    apply(btn);
  });
  cal.addEventListener('pointermove', (e) => {
    if (dragMode) apply(document.elementFromPoint(e.clientX, e.clientY)?.closest('.cal-day'));
  });
  const endDrag = () => {
    if (dragMode) {
      dragMode = null;
      renderCal();
    }
  };
  window.addEventListener('pointerup', endDrag);
  cleanup = () => window.removeEventListener('pointerup', endDrag);
  cal.addEventListener('keydown', (e) => {
    const btn = e.target.closest('.cal-day');
    if (btn && (e.key === 'Enter' || e.key === ' ')) {
      e.preventDefault();
      dragMode = selected.has(btn.dataset.date) ? 'remove' : 'add';
      apply(btn);
      dragMode = null;
      renderCal();
      $(`[data-date="${btn.dataset.date}"]`, cal)?.focus();
    }
  });

  $('#prev').onclick = () => { cursor = new Date(cursor.getFullYear(), cursor.getMonth() - 1, 1); renderCal(); };
  $('#next').onclick = () => { cursor = new Date(cursor.getFullYear(), cursor.getMonth() + 1, 1); renderCal(); };
  $('#clear').onclick = () => { selected.clear(); renderCal(); };
  const quick = (n, weekdaysOnly) => {
    selected.clear();
    for (let d = today; selected.size < n; d = addDays(d, 1)) {
      const wd = dateParts(d).getUTCDay();
      if (!weekdaysOnly || (wd > 0 && wd < 6)) selected.add(d);
    }
    cursor = new Date(+today.slice(0, 4), +today.slice(5, 7) - 1, 1);
    renderCal();
  };
  $('#weekdays').onclick = () => quick(5, true);
  $('#week').onclick = () => quick(7, false);

  function renderDurations() {
    const slot = +$('#slot').value;
    const prev = +$('#duration').value || 60;
    const opts = [15, 30, 45, 60, 90, 120, 180, 240].filter((d) => d >= slot && d % slot === 0);
    const pick = opts.includes(prev) ? prev : opts.find((d) => d >= prev) ?? opts[0];
    $('#duration').innerHTML = opts
      .map((d) => `<option value="${d}"${d === pick ? ' selected' : ''}>${lengthLabel(d)}</option>`)
      .join('');
  }
  $('#slot').onchange = renderDurations;
  renderDurations();
  renderCal();

  $('#create').onsubmit = async (e) => {
    e.preventDefault();
    const err = $('#err');
    err.textContent = '';
    const title = $('#title').value.trim();
    if (!title) {
      err.textContent = 'Give your poll a name.';
      $('#title').focus();
      return;
    }
    if (!selected.size) {
      err.textContent = 'Pick at least one date.';
      return;
    }
    const organizer = $('#organizer').value.trim();
    if (organizer) rememberName(organizer);
    $('#submit').disabled = true;
    try {
      const { id, adminToken } = await api('events', {
        method: 'POST',
        body: {
          title,
          description: $('#desc').value,
          organizer,
          timezone: $('#tz').value,
          dates: [...selected],
          startMinute: +$('#start').value,
          endMinute: +$('#end').value,
          slotMinutes: +$('#slot').value,
          durationMinutes: +$('#duration').value,
        },
      });
      save(id, { adminToken, justCreated: true });
      navigate(`/e/${id}`);
    } catch (ex) {
      err.textContent = ex.message;
      $('#submit').disabled = false;
    }
  };
  $('#title').focus();
}

// ---------- event page ----------

async function renderEvent(id) {
  // Organizer and personal edit links carry their secret in the URL fragment,
  // which is never sent to the server. Move it to local storage and tidy the URL.
  const hash = new URLSearchParams(location.hash.slice(1));
  if (hash.get('admin')) save(id, { adminToken: hash.get('admin') });
  if (hash.get('edit')) {
    const [participantId, editToken] = hash.get('edit').split('.');
    if (participantId && editToken) save(id, { participantId, editToken });
  }
  if (location.hash) history.replaceState(null, '', location.pathname);

  app.innerHTML = '<div class="empty muted">Loading…</div>';

  const st = {
    id,
    ev: null,
    tz: saved(id).tz || localTz,
    mine: {},
    brush: YES,
    dirty: false,
    focus: null, // participant id highlighted in the group grid
    inspect: null, // slot being inspected in the group grid
  };

  const authHeaders = () => (saved(id).adminToken ? { 'x-admin-token': saved(id).adminToken } : {});

  async function load() {
    st.ev = await api(`events/${id}`, { headers: authHeaders() });
    if (!st.ev.isAdmin && saved(id).adminToken) forget(id, 'adminToken');
    const me = st.ev.participants.find((p) => p.id === saved(id).participantId);
    if (!me && saved(id).participantId) forget(id, 'participantId', 'editToken');
    return me;
  }

  try {
    const me = await load();
    if (me && !st.dirty) st.mine = { ...me.availability };
  } catch (e) {
    app.innerHTML = `<div class="card empty"><h1>Poll not found</h1><p class="muted">${esc(e.message)}</p><a class="btn btn-primary" href="/" data-link>Create a new poll</a></div>`;
    return;
  }

  document.title = `${st.ev.title} · Overlap`;
  const shareUrl = `${location.origin}/e/${id}`;
  const me = () => st.ev.participants.find((p) => p.id === saved(id).participantId);

  app.innerHTML = `
    <div id="event-root">
    <div class="event-head">
      <div>
        <h1 id="ev-title"></h1>
        <p class="muted" id="ev-meta" style="margin:0"></p>
      </div>
      <div class="share">
        <input type="text" readonly value="${esc(shareUrl)}" aria-label="Share link" onclick="this.select()">
        <button class="btn" id="copy">Copy link</button>
      </div>
    </div>
    <div id="banners"></div>
    <div class="toolbar">
      <div class="tz-picker small">
        <label for="tz" style="margin:0;font-weight:500" class="muted">Times in</label>
        <select id="tz">${tzOptions(st.tz)}</select>
      </div>
      <span class="small muted" id="tz-note"></span>
    </div>
    <div class="panels">
      <section class="card" aria-labelledby="mine-h">
        <h2 id="mine-h">Your availability</h2>
        <div class="stack">
          <div>
            <label for="name">Your name</label>
            <input id="name" type="text" maxlength="80" autocomplete="name">
          </div>
          <div class="btn-row" style="justify-content:space-between">
            <div class="segmented" role="group" aria-label="Brush">
              <button type="button" data-brush="${YES}"><span class="swatch swatch-yes"></span>Available</button>
              <button type="button" data-brush="${IF_NEEDED}"><span class="swatch swatch-maybe"></span>If needed</button>
              <button type="button" data-brush="0"><span class="swatch swatch-no"></span>Busy</button>
            </div>
            <div class="btn-row">
              <span class="btn btn-sm file-btn" title="Mark the times you're busy from a calendar export (.ics)">Import calendar<input type="file" id="ics" accept=".ics,text/calendar"></span>
              <button type="button" class="btn btn-sm btn-ghost" id="fill">All</button>
              <button type="button" class="btn btn-sm btn-ghost" id="none">None</button>
            </div>
          </div>
          <p class="small muted" style="margin:0">Click and drag to paint the times you can make it. On a phone, tap or drag.</p>
          <div class="grid-scroll"><div class="grid" id="mine" role="grid" aria-label="Your availability"></div></div>
        </div>
        <div class="save-bar">
          <span id="save-state" class="small"></span>
          <div class="btn-row">
            <button type="button" class="btn btn-ghost btn-danger btn-sm" id="leave" hidden>Remove my response</button>
            <button type="button" class="btn btn-primary" id="save">Save</button>
          </div>
        </div>
        <div class="error" id="save-err" role="alert"></div>
        <p class="small muted" id="edit-link" hidden style="margin:12px 0 0"></p>
      </section>
      <div class="stack">
        <section class="card" aria-labelledby="best-h">
          <h2 id="best-h">Best times</h2>
          <div id="best"></div>
        </section>
        <section class="card" aria-labelledby="group-h">
          <h2 id="group-h">Everyone</h2>
          <ul class="people" id="people"></ul>
          <div class="grid-scroll" style="margin-top:12px"><div class="grid readonly" id="group" aria-label="Group availability"></div></div>
          <div class="legend" id="legend"></div>
          <div class="hover-info" id="inspect"></div>
        </section>
      </div>
    </div>
    </div>`;

  $('#copy').onclick = () => copy(shareUrl);
  $('#name').value = me()?.name || lastName();
  $('#name').oninput = () => setDirty(true);
  $('#tz').onchange = (e) => {
    st.tz = e.target.value;
    save(id, { tz: st.tz });
    renderAll();
  };

  // ----- layout of slots in the viewer's time zone -----
  let L;
  function computeLayout() {
    const dates = new Set(), mins = new Set(), at = new Map();
    for (const s of st.ev.slots) {
      const z = utcToZoned(s, st.tz);
      dates.add(z.date);
      mins.add(z.minutes);
      at.set(`${z.date}|${z.minutes}`, s);
    }
    L = { dates: [...dates].sort(), mins: [...mins].sort((a, b) => a - b), at };
  }

  function gridHtml(cellClass) {
    const { dates, mins, at } = L;
    const labelEvery = st.ev.slotMinutes >= 60 ? st.ev.slotMinutes : 60;
    let h = '<div></div>';
    for (const d of dates) {
      const dt = dateParts(d);
      h += `<div class="day-label"><b>${fmtDow.format(dt)}</b>${fmtMonthDay.format(dt)}</div>`;
    }
    mins.forEach((m, r) => {
      h += `<div class="time-label">${m % labelEvery === 0 ? minutesLabel(m) : ''}</div>`;
      dates.forEach((d, c) => {
        const s = at.get(`${d}|${m}`);
        const hour = m % 60 === 0 && r > 0 ? ' hour' : '';
        if (s == null) h += `<div class="cell void${hour}"></div>`;
        else
          h += `<div class="cell${hour} ${cellClass(s)}" data-slot="${s}" data-r="${r}" data-c="${c}" role="gridcell" tabindex="-1" aria-label="${esc(fmtRange(s, s + st.ev.slotMinutes * MINUTE, st.tz))}"></div>`;
      });
    });
    return h;
  }

  const gridCols = () => `auto repeat(${L.dates.length}, minmax(38px, 1fr))`;
  const finalSet = () => {
    const set = new Set();
    const f = st.ev.finalStart;
    if (f != null) for (let t = f; t < f + st.ev.durationMinutes * MINUTE; t += st.ev.slotMinutes * MINUTE) set.add(t);
    return set;
  };

  // ----- my grid -----
  const mineEl = $('#mine');
  const mineClass = (s) => (st.mine[s] === YES ? 'yes' : st.mine[s] === IF_NEEDED ? 'maybe' : '');

  function renderMine() {
    mineEl.style.gridTemplateColumns = gridCols();
    mineEl.innerHTML = gridHtml((s) => mineClass(s) + (finalSet().has(s) ? ' final' : ''));
    const first = mineEl.querySelector('.cell[data-slot]');
    if (first) first.tabIndex = 0;
  }
  function refreshMine() {
    for (const el of mineEl.querySelectorAll('.cell[data-slot]')) {
      const v = st.mine[el.dataset.slot];
      el.classList.toggle('yes', v === YES);
      el.classList.toggle('maybe', v === IF_NEEDED);
    }
  }

  let drag = null;
  const cellAt = (x, y) => document.elementFromPoint(x, y)?.closest('#mine .cell[data-slot]');
  function paintRect(a, b) {
    st.mine = { ...drag.snapshot };
    const [r0, r1] = [Math.min(a.r, b.r), Math.max(a.r, b.r)];
    const [c0, c1] = [Math.min(a.c, b.c), Math.max(a.c, b.c)];
    for (let r = r0; r <= r1; r++)
      for (let c = c0; c <= c1; c++) {
        const s = L.at.get(`${L.dates[c]}|${L.mins[r]}`);
        if (s == null) continue;
        if (drag.value) st.mine[s] = drag.value;
        else delete st.mine[s];
      }
    refreshMine();
  }
  const pos = (el) => ({ r: +el.dataset.r, c: +el.dataset.c });

  mineEl.addEventListener('pointerdown', (e) => {
    const el = e.target.closest('.cell[data-slot]');
    if (!el || e.button > 0) return;
    e.preventDefault();
    const cur = st.mine[el.dataset.slot] || 0;
    drag = { anchor: pos(el), snapshot: { ...st.mine }, value: cur === st.brush ? 0 : st.brush };
    paintRect(drag.anchor, drag.anchor);
  });
  mineEl.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const el = cellAt(e.clientX, e.clientY);
    if (el) paintRect(drag.anchor, pos(el));
  });
  const endPaint = () => {
    if (!drag) return;
    drag = null;
    setDirty(true);
    renderGroup();
    renderBest();
  };
  window.addEventListener('pointerup', endPaint);
  window.addEventListener('pointercancel', endPaint);

  // keyboard: arrows move, space/enter paints with the current brush
  mineEl.addEventListener('keydown', (e) => {
    const el = e.target.closest('.cell[data-slot]');
    if (!el) return;
    let { r, c } = pos(el);
    const moves = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
    if (moves[e.key]) {
      e.preventDefault();
      for (let i = 0; i < 200; i++) {
        r += moves[e.key][0];
        c += moves[e.key][1];
        if (r < 0 || c < 0 || r >= L.mins.length || c >= L.dates.length) return;
        const next = mineEl.querySelector(`.cell[data-r="${r}"][data-c="${c}"]`);
        if (next) {
          el.tabIndex = -1;
          next.tabIndex = 0;
          next.focus();
          return;
        }
      }
    } else if (e.key === ' ' || e.key === 'Enter') {
      e.preventDefault();
      const s = el.dataset.slot;
      if ((st.mine[s] || 0) === st.brush) delete st.mine[s];
      else if (st.brush) st.mine[s] = st.brush;
      else delete st.mine[s];
      refreshMine();
      setDirty(true);
      renderGroup();
      renderBest();
    }
  });

  for (const b of app.querySelectorAll('[data-brush]')) {
    b.onclick = () => {
      st.brush = +b.dataset.brush;
      renderBrush();
    };
  }
  function renderBrush() {
    for (const b of app.querySelectorAll('[data-brush]')) b.setAttribute('aria-pressed', +b.dataset.brush === st.brush);
  }

  $('#fill').onclick = () => {
    st.mine = Object.fromEntries(st.ev.slots.map((s) => [s, YES]));
    refreshMine();
    setDirty(true);
  };
  $('#none').onclick = () => {
    st.mine = {};
    refreshMine();
    setDirty(true);
  };

  $('#ics').onchange = async (e) => {
    const file = e.target.files[0];
    e.target.value = '';
    if (!file) return;
    try {
      const slots = st.ev.slots;
      const slotMs = st.ev.slotMinutes * MINUTE;
      const busy = parseIcsBusy(await file.text(), slots[0], slots[slots.length - 1] + slotMs, st.tz);
      if (Object.keys(st.mine).length && !confirm('Replace what you have painted with the free times from your calendar?')) return;
      st.mine = availabilityFromBusy(slots, st.ev.slotMinutes, busy);
      refreshMine();
      setDirty(true);
      const free = Object.keys(st.mine).length;
      toast(`Found ${busy.length} busy event${busy.length === 1 ? '' : 's'}; marked ${free} free slot${free === 1 ? '' : 's'}. Review and save.`);
    } catch (ex) {
      console.error(ex);
      toast("Couldn't read that calendar file.");
    }
  };

  // ----- saving -----
  function setDirty(v) {
    st.dirty = v;
    const el = $('#save-state');
    if (v) {
      el.className = 'dirty';
      el.textContent = 'Unsaved changes';
    } else {
      el.className = 'small muted';
      el.textContent = me() ? 'Saved' : '';
    }
  }

  $('#save').onclick = async () => {
    const err = $('#save-err');
    err.textContent = '';
    const name = $('#name').value.trim();
    if (!name) {
      err.textContent = 'Add your name so others know who you are.';
      $('#name').focus();
      return;
    }
    rememberName(name);
    $('#save').disabled = true;
    try {
      const body = { name, availability: st.mine };
      const s = saved(id);
      let created = false;
      if (s.participantId) {
        try {
          await api(`events/${id}/participants/${s.participantId}`, { method: 'PUT', body, headers: { 'x-edit-token': s.editToken } });
        } catch (ex) {
          if (ex.status !== 404 && ex.status !== 403) throw ex;
          forget(id, 'participantId', 'editToken');
        }
      }
      if (!saved(id).participantId) {
        const r = await api(`events/${id}/participants`, { method: 'POST', body });
        save(id, { participantId: r.id, editToken: r.editToken });
        created = true;
      }
      await load();
      setDirty(false);
      renderAll();
      toast(created ? 'Saved! Others can now see your availability.' : 'Saved');
    } catch (ex) {
      err.textContent = ex.message;
    } finally {
      $('#save').disabled = false;
    }
  };

  $('#leave').onclick = async () => {
    if (!confirm('Remove your response from this poll?')) return;
    const s = saved(id);
    try {
      await api(`events/${id}/participants/${s.participantId}`, { method: 'DELETE', headers: { 'x-edit-token': s.editToken } });
    } catch (ex) {
      if (ex.status !== 404) return toast(ex.message);
    }
    forget(id, 'participantId', 'editToken');
    st.mine = {};
    await load();
    setDirty(false);
    renderAll();
  };

  const beforeUnload = (e) => {
    if (st.dirty) {
      e.preventDefault();
      e.returnValue = '';
    }
  };
  window.addEventListener('beforeunload', beforeUnload);

  // ----- group view -----
  const groupEl = $('#group');
  // Everyone's responses, with my unsaved edits shown live.
  const others = () => st.ev.participants.map((p) => (p.id === me()?.id ? { ...p, availability: st.mine } : p));

  function renderGroup() {
    const people = others();
    const n = people.length;
    const fs = finalSet();
    const focus = people.find((p) => p.id === st.focus);
    groupEl.style.gridTemplateColumns = gridCols();
    groupEl.innerHTML = gridHtml((s) => {
      let cls = fs.has(s) ? 'final' : '';
      if (String(s) === String(st.inspect)) cls += ' active';
      if (focus) cls += focus.availability[s] === YES ? ' yes' : focus.availability[s] === IF_NEEDED ? ' maybe' : '';
      return cls;
    });
    if (!focus) {
      for (const el of groupEl.querySelectorAll('.cell[data-slot]')) {
        const s = el.dataset.slot;
        let score = 0;
        for (const p of people) score += p.availability[s] === YES ? 1 : p.availability[s] === IF_NEEDED ? 0.5 : 0;
        if (score > 0) el.style.background = `rgba(var(--heat), ${0.15 + 0.85 * (score / n)})`;
      }
    }
    const scale = [0.2, 0.4, 0.6, 0.8, 1].map((a) => `<span style="background:rgba(var(--heat),${0.15 + 0.85 * a})"></span>`).join('');
    $('#legend').innerHTML = focus
      ? `<span>Showing <b>${esc(focus.name)}</b></span><span><span class="swatch swatch-yes"></span> available</span><span><span class="swatch swatch-maybe"></span> if needed</span><button class="btn btn-sm btn-ghost" id="unfocus">Show everyone</button>`
      : n
        ? `<span>0/${n}</span><span class="heat-scale">${scale}</span><span>${n}/${n} available</span>${st.ev.finalStart != null ? '<span><span class="swatch" style="outline:2px solid var(--final)"></span> picked time</span>' : ''}`
        : '';
    $('#unfocus') && ($('#unfocus').onclick = () => { st.focus = null; renderGroup(); renderPeople(); });
    renderInspect();
  }

  function renderInspect() {
    const box = $('#inspect');
    const people = others();
    if (!people.length) {
      box.innerHTML = '<p class="muted" style="margin:0">Nobody has responded yet. Share the link to get started.</p>';
      return;
    }
    if (st.inspect == null) {
      box.innerHTML = '<p class="muted" style="margin:0">Hover or tap a time to see who can make it. Tap a name to see just their availability.</p>';
      return;
    }
    const s = +st.inspect;
    const yes = people.filter((p) => p.availability[s] === YES);
    const maybe = people.filter((p) => p.availability[s] === IF_NEEDED);
    const no = people.filter((p) => !p.availability[s]);
    const chips = (list, cls) => list.map((p) => `<span class="chip ${cls}">${esc(p.name)}</span>`).join('');
    const canPick = st.ev.isAdmin && st.ev.slots.includes(s + st.ev.durationMinutes * MINUTE - st.ev.slotMinutes * MINUTE);
    box.innerHTML = `
      <div class="btn-row" style="justify-content:space-between">
        <b>${esc(fmtRange(s, s + st.ev.slotMinutes * MINUTE, st.tz))}</b>
        ${canPick ? `<button class="btn btn-sm" data-pick="${s}">Pick this start time</button>` : ''}
      </div>
      <div class="names">${chips(yes, 'yes')}${chips(maybe, 'maybe')}${chips(no, 'no')}</div>
      <p class="small muted" style="margin:6px 0 0">${yes.length} available · ${maybe.length} if needed · ${no.length} busy</p>`;
  }

  const inspectFrom = (e) => {
    const el = e.target.closest('.cell[data-slot]');
    if (!el || el.dataset.slot === String(st.inspect)) return;
    groupEl.querySelector('.cell.active')?.classList.remove('active');
    el.classList.add('active');
    st.inspect = el.dataset.slot;
    renderInspect();
  };
  groupEl.addEventListener('pointerover', inspectFrom);
  groupEl.addEventListener('click', inspectFrom);

  // ----- people -----
  function renderPeople() {
    const myId = me()?.id;
    $('#people').innerHTML = st.ev.participants
      .map((p) => {
        const canDelete = p.id === myId || st.ev.isAdmin;
        return `<li class="chip${p.id === myId ? ' me' : ''}${p.id === st.focus ? ' focus' : ''}" data-person="${esc(p.id)}" role="button" tabindex="0" aria-pressed="${p.id === st.focus}">${esc(p.name)}${p.id === myId ? ' (you)' : ''}${canDelete && p.id !== myId ? `<button data-remove="${esc(p.id)}" aria-label="Remove ${esc(p.name)}" title="Remove">×</button>` : ''}</li>`;
      })
      .join('');
  }
  $('#people').addEventListener('click', async (e) => {
    const rm = e.target.closest('[data-remove]');
    if (rm) {
      e.stopPropagation();
      const p = st.ev.participants.find((x) => x.id === rm.dataset.remove);
      if (!p || !confirm(`Remove ${p.name}'s response?`)) return;
      try {
        await api(`events/${id}/participants/${p.id}`, { method: 'DELETE', headers: authHeaders() });
        if (st.focus === p.id) st.focus = null;
        await load();
        renderAll();
      } catch (ex) {
        toast(ex.message);
      }
      return;
    }
    const chip = e.target.closest('[data-person]');
    if (chip) {
      st.focus = st.focus === chip.dataset.person ? null : chip.dataset.person;
      renderPeople();
      renderGroup();
    }
  });
  $('#people').addEventListener('keydown', (e) => {
    if ((e.key === 'Enter' || e.key === ' ') && e.target.matches('[data-person]')) {
      e.preventDefault();
      e.target.click();
    }
  });

  // ----- best times -----
  function renderBest() {
    const people = others();
    const best = rankTimes(st.ev, st.ev.slots, people, 5);
    const n = people.length;
    const lenText = lengthLabel(st.ev.durationMinutes);
    if (!n || !best.length) {
      $('#best').innerHTML = `<p class="muted" style="margin:0">${n ? 'No time works for anyone yet.' : `Best times for a ${lenText} meeting will show up here as people respond.`}</p>`;
      return;
    }
    $('#best').innerHTML = `
      <p class="small muted">For a ${lenText} meeting, ranked by how many people can make all of it.</p>
      <ol class="best">${best
        .map((b) => {
          const notes = [];
          if (b.ifNeeded.length) notes.push(`If needed: ${b.ifNeeded.map(esc).join(', ')}`);
          if (b.no.length) notes.push(`Can't: ${b.no.map(esc).join(', ')}`);
          const isFinal = st.ev.finalStart === b.start;
          return `<li>
            <div>
              <div class="when">${esc(fmtRange(b.start, b.end, st.tz))}${isFinal ? ' ✓' : ''}</div>
              <div class="who">${notes.join(' · ') || 'Everyone can make it'}</div>
            </div>
            <div class="btn-row" style="flex-wrap:nowrap">
              <span class="score" title="${b.yes.length} available, ${b.ifNeeded.length} if needed">${b.yes.length + b.ifNeeded.length}/${n}</span>
              ${st.ev.isAdmin && !isFinal ? `<button class="btn btn-sm" data-pick="${b.start}">Pick</button>` : ''}
            </div>
          </li>`;
        })
        .join('')}</ol>`;
  }

  // ----- finalize -----
  $('#event-root').addEventListener('click', async (e) => {
    const btn = e.target.closest('[data-pick], #unpick');
    if (!btn) return;
    const start = btn.id === 'unpick' ? null : +btn.dataset.pick;
    try {
      await api(`events/${id}/finalize`, { method: 'POST', body: { start }, headers: authHeaders() });
      await load();
      renderAll();
      toast(start == null ? 'Final time cleared' : 'Final time picked. Share the link so everyone can add it to their calendar.');
    } catch (ex) {
      toast(ex.message);
    }
  });

  function renderBanners() {
    const ev = st.ev;
    let h = '';
    if (ev.finalStart != null) {
      const end = ev.finalStart + ev.durationMinutes * MINUTE;
      const info = { title: ev.title, description: [ev.description, shareUrl].filter(Boolean).join('\n\n'), start: ev.finalStart, end };
      h += `<div class="banner banner-final">
        <strong>📅 ${esc(fmtRange(ev.finalStart, end, st.tz))}</strong>
        <div class="btn-row">
          <a class="btn btn-sm" href="${esc(googleCalendarUrl(info))}" target="_blank" rel="noopener">Google Calendar</a>
          <a class="btn btn-sm" href="${esc(outlookCalendarUrl(info))}" target="_blank" rel="noopener">Outlook</a>
          <a class="btn btn-sm" href="/api/events/${esc(id)}/event.ics">Apple / .ics</a>
          ${ev.isAdmin ? '<button class="btn btn-sm btn-ghost" id="unpick">Clear final time</button>' : ''}
        </div>
      </div>`;
    }
    if (ev.isAdmin) {
      const adminUrl = `${shareUrl}#admin=${saved(id).adminToken}`;
      const fresh = saved(id).justCreated;
      h += `<div class="banner banner-admin">
        <strong>${fresh ? 'Your poll is ready. Share the link above.' : "You're the organizer"}</strong>
        <p class="small" style="margin:0 0 8px">Pick the final time from "Best times" or by tapping any time in the group grid. To manage this poll from another device, bookmark your private organizer link, and don't share it.</p>
        <div class="btn-row">
          <button class="btn btn-sm" id="copy-admin">Copy organizer link</button>
          <button class="btn btn-sm btn-ghost btn-danger" id="delete-poll">Delete poll</button>
        </div>
      </div>`;
    }
    $('#banners').innerHTML = h;
    $('#copy-admin') && ($('#copy-admin').onclick = () => copy(`${shareUrl}#admin=${saved(id).adminToken}`));
    $('#delete-poll') &&
      ($('#delete-poll').onclick = async () => {
        if (!confirm('Delete this poll and all responses? This cannot be undone.')) return;
        try {
          await api(`events/${id}`, { method: 'DELETE', headers: authHeaders() });
          forget(id, 'adminToken', 'participantId', 'editToken');
          st.dirty = false;
          navigate('/');
          toast('Poll deleted');
        } catch (ex) {
          toast(ex.message);
        }
      });
  }

  function renderHeader() {
    const ev = st.ev;
    $('#ev-title').textContent = ev.title;
    const meta = [];
    if (ev.organizer) meta.push(`by ${ev.organizer}`);
    meta.push(`${lengthLabel(ev.durationMinutes)} meeting`);
    meta.push(`${ev.participants.length} response${ev.participants.length === 1 ? '' : 's'}`);
    $('#ev-meta').innerHTML = esc(meta.join(' · ')) + (ev.description ? `<br><span style="white-space:pre-wrap">${esc(ev.description)}</span>` : '');
    $('#tz-note').textContent = st.tz === ev.timezone ? '' : `Organizer's time zone: ${ev.timezone.replace(/_/g, ' ')}`;
  }

  function renderMeState() {
    const mine = me();
    $('#leave').hidden = !mine;
    $('#save').textContent = mine ? 'Save changes' : 'Save my availability';
    const s = saved(id);
    const link = $('#edit-link');
    if (mine) {
      link.hidden = false;
      link.innerHTML = `Responding as <b>${esc(mine.name)}</b>. Want to edit from another device? <a href="#" id="copy-edit">Copy your personal edit link</a>.`;
      $('#copy-edit').onclick = (e) => {
        e.preventDefault();
        copy(`${shareUrl}#edit=${s.participantId}.${s.editToken}`);
      };
    } else link.hidden = true;
  }

  function renderAll() {
    computeLayout();
    renderHeader();
    renderBanners();
    renderBrush();
    renderMine();
    renderPeople();
    renderGroup();
    renderBest();
    renderMeState();
    setDirty(st.dirty);
  }

  // Pick up other people's responses every 20s while the tab is visible.
  const poll = setInterval(async () => {
    if (document.hidden || drag) return;
    try {
      const before = JSON.stringify(st.ev.participants) + st.ev.finalStart;
      await load();
      if (JSON.stringify(st.ev.participants) + st.ev.finalStart !== before) {
        renderHeader();
        renderBanners();
        renderPeople();
        renderGroup();
        renderBest();
        renderMeState();
      }
    } catch {}
  }, 20_000);

  cleanup = () => {
    clearInterval(poll);
    window.removeEventListener('pointerup', endPaint);
    window.removeEventListener('pointercancel', endPaint);
    window.removeEventListener('beforeunload', beforeUnload);
  };

  renderAll();
  if (saved(id).justCreated) forget(id, 'justCreated');
}

route();
