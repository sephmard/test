import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  YES,
  IF_NEEDED,
  availabilityFromBusy,
  buildIcs,
  buildSlots,
  eventWindows,
  meetingEnd,
  normalizeAvailability,
  normalizeEventInput,
  parseIcsBusy,
  rankTimes,
  utcToZoned,
  zonedToUtc,
} from '../public/core.js';

const base = {
  title: 'Sync',
  timezone: 'America/New_York',
  dates: ['2026-11-02', '2026-11-03'],
  startMinute: 540,
  endMinute: 600,
  slotMinutes: 30,
  durationMinutes: 60,
};

test('zonedToUtc / utcToZoned round-trip, including DST', () => {
  assert.equal(zonedToUtc('2026-07-01', 540, 'America/New_York'), Date.UTC(2026, 6, 1, 13));
  assert.equal(zonedToUtc('2026-12-01', 540, 'America/New_York'), Date.UTC(2026, 11, 1, 14));
  assert.equal(zonedToUtc('2026-12-01', 540, 'Asia/Kolkata'), Date.UTC(2026, 11, 1, 3, 30));
  assert.deepEqual(utcToZoned(Date.UTC(2026, 11, 1, 3, 30), 'Asia/Kolkata'), { date: '2026-12-01', minutes: 540, weekday: 'Tue' });
});

test('buildSlots skips non-existent spring-forward times', () => {
  const slots = buildSlots({ ...base, dates: ['2026-03-08'], startMinute: 60, endMinute: 240, slotMinutes: 60 });
  // 2:00 doesn't exist in New York on 2026-03-08, so 1:00, 3:00 only.
  assert.deepEqual(slots.map((s) => utcToZoned(s, 'America/New_York').minutes), [60, 180]);
});

test('normalizeEventInput validates', () => {
  const ev = normalizeEventInput({ ...base, dates: ['2026-11-03', '2026-11-02', '2026-11-02'] });
  assert.deepEqual(ev.dates, ['2026-11-02', '2026-11-03']);
  assert.throws(() => normalizeEventInput({ ...base, title: ' ' }), /Title/);
  assert.throws(() => normalizeEventInput({ ...base, timezone: 'Mars/Base' }), /time zone/);
  assert.throws(() => normalizeEventInput({ ...base, dates: ['2026-02-30'] }), /Invalid date/);
  assert.throws(() => normalizeEventInput({ ...base, durationMinutes: 45 }), /multiple/);
  assert.throws(() => normalizeEventInput({ ...base, durationMinutes: 120 }), /longer/);
  assert.throws(() => normalizeEventInput({ ...base, endMinute: 540 }), /end time/);
});

test('normalizeAvailability drops unknown slots and rejects bad values', () => {
  const slots = buildSlots(base);
  const av = normalizeAvailability({ [slots[0]]: YES, [slots[1]]: IF_NEEDED, 123: YES, [slots[2]]: 0 }, slots);
  assert.deepEqual(av, { [slots[0]]: YES, [slots[1]]: IF_NEEDED });
  assert.throws(() => normalizeAvailability({ [slots[0]]: 7 }, slots));
});

test('rankTimes requires the whole meeting window and prefers yes over if-needed', () => {
  const slots = buildSlots(base); // 4 slots: 9:00, 9:30 on two days
  const [a, b, c, d] = slots;
  const people = [
    { name: 'Ann', availability: { [a]: YES, [b]: YES, [c]: YES, [d]: YES } },
    { name: 'Bob', availability: { [a]: YES, [c]: YES, [d]: IF_NEEDED } },
    { name: 'Cy', availability: { [c]: YES, [d]: YES } },
  ];
  const best = rankTimes(base, slots, people);
  // Only full-hour windows: 9:00 day 1 and 9:00 day 2 (9:30 has no second half-hour).
  assert.equal(best.length, 2);
  assert.equal(best[0].start, c);
  assert.deepEqual(best[0].yes, ['Ann', 'Cy']);
  assert.deepEqual(best[0].ifNeeded, ['Bob']);
  assert.deepEqual(best[1].yes, ['Ann']);
  assert.deepEqual(best[1].no, ['Bob', 'Cy']);
});

test('parseIcsBusy handles UTC, TZID, weekly recurrence, EXDATE and free/cancelled events', () => {
  const ics = [
    'BEGIN:VCALENDAR',
    'BEGIN:VEVENT',
    'UID:one',
    'DTSTART:20261102T143000Z',
    'DTEND:20261102T150000Z',
    'END:VEVENT',
    'BEGIN:VEVENT',
    'UID:weekly',
    'DTSTART;TZID=America/New_York:20261026T090000',
    'DURATION:PT30M',
    'RRULE:FREQ=WEEKLY;BYDAY=MO,TU;COUNT=6',
    'EXDATE;TZID=America/New_York:20261103T090000',
    'END:VEVENT',
    'BEGIN:VEVENT',
    'UID:free',
    'DTSTART:20261103T150000Z',
    'DTEND:20261103T160000Z',
    'TRANSP:TRANSPARENT',
    'END:VEVENT',
    'BEGIN:VEVENT',
    'UID:allday',
    'DTSTART;VALUE=DATE:20261102',
    'DTEND;VALUE=DATE:20261103',
    'END:VEVENT',
    'END:VCALENDAR',
  ].join('\r\n');
  const slots = buildSlots(base);
  const busy = parseIcsBusy(ics, slots[0], slots.at(-1) + 30 * 60_000, 'UTC');
  // Nov 2 9:00 ET (weekly), Nov 2 9:30 ET (one-off). Nov 3 is excluded via EXDATE.
  assert.deepEqual(busy.map((b) => b.start), [zonedToUtc('2026-11-02', 540, 'America/New_York'), Date.UTC(2026, 10, 2, 14, 30)]);
  const av = availabilityFromBusy(slots.map((s) => ({ start: s, end: s + 30 * 60_000 })), busy);
  assert.deepEqual(Object.keys(av).map(Number), slots.slice(2));
});

test('parseIcsBusy honours RECURRENCE-ID overrides and UNTIL', () => {
  const ics = [
    'BEGIN:VEVENT',
    'UID:daily',
    'DTSTART:20261102T140000Z',
    'DTEND:20261102T143000Z',
    'RRULE:FREQ=DAILY;UNTIL=20261104T000000Z',
    'END:VEVENT',
    'BEGIN:VEVENT',
    'UID:daily',
    'RECURRENCE-ID:20261103T140000Z',
    'DTSTART:20261103T180000Z',
    'DTEND:20261103T183000Z',
    'END:VEVENT',
  ].join('\n');
  const busy = parseIcsBusy(ics, Date.UTC(2026, 10, 1), Date.UTC(2026, 10, 10));
  assert.deepEqual(busy.map((b) => new Date(b.start).toISOString()), [
    '2026-11-02T14:00:00.000Z',
    '2026-11-03T18:00:00.000Z',
  ]);
});

test('buildIcs escapes text', () => {
  const ics = buildIcs({ id: 'x', title: 'A, B; C', description: 'line1\nline2', start: 0, end: 3_600_000 });
  assert.match(ics, /SUMMARY:A\\, B\\; C/);
  assert.match(ics, /DESCRIPTION:line1\\nline2/);
  assert.match(ics, /DTSTART:19700101T000000Z/);
});

const H = 3_600_000;
const t0 = Date.UTC(2026, 10, 3, 14);
const optionsPoll = {
  title: 'Lunch',
  timezone: 'UTC',
  mode: 'options',
  options: [
    { start: t0 + 24 * H, end: t0 + 25 * H },
    { start: t0, end: t0 + H },
    { start: t0 + 3 * H, end: t0 + 3.5 * H },
  ],
};

test('specific-times polls are validated and sorted', () => {
  const ev = normalizeEventInput(optionsPoll);
  assert.equal(ev.mode, 'options');
  assert.deepEqual(ev.options.map((o) => o.start), [t0, t0 + 3 * H, t0 + 24 * H]);
  assert.deepEqual(buildSlots(ev), [t0, t0 + 3 * H, t0 + 24 * H]);
  assert.equal(meetingEnd(ev, t0 + 3 * H), t0 + 3.5 * H);
  assert.throws(() => normalizeEventInput({ ...optionsPoll, options: [] }), /at least one/);
  assert.throws(() => normalizeEventInput({ ...optionsPoll, options: [{ start: t0, end: t0 }] }), /last between/);
  assert.throws(() => normalizeEventInput({ ...optionsPoll, options: [{ start: t0, end: t0 + H }, { start: t0, end: t0 + 2 * H }] }), /same moment/);
  assert.throws(() => normalizeEventInput({ ...optionsPoll, options: [{ start: 'x', end: 1 }] }), /Invalid/);
});

test('specific-times polls rank each option on its own', () => {
  const ev = normalizeEventInput(optionsPoll);
  const [a, b, c] = buildSlots(ev);
  const people = [
    { name: 'Ann', availability: { [a]: YES, [c]: YES } },
    { name: 'Bob', availability: { [a]: IF_NEEDED, [c]: YES } },
    { name: 'Cy', availability: { [b]: YES } },
  ];
  const best = rankTimes(ev, buildSlots(ev), people);
  assert.deepEqual(best.map((r) => r.start), [c, a, b]);
  assert.equal(best[0].end, c + H);
  assert.deepEqual(best[1].ifNeeded, ['Bob']);
});

test('calendar import checks each proposed time against busy events', () => {
  const ev = normalizeEventInput(optionsPoll);
  const busy = [{ start: t0 + 3.25 * H, end: t0 + 4 * H }];
  assert.deepEqual(Object.keys(availabilityFromBusy(eventWindows(ev), busy)).map(Number), [t0, t0 + 24 * H]);
});
