# Overlap

A free, open-source alternative to Doodle for finding a time to meet. Propose some times, share one link, and everyone marks what works. Overlap shows the best time straight away.

Two kinds of poll:

- **Propose specific times** (default): you suggest exact times, like "Tue 2pm, Wed 10am". Invitees answer each one with *Yes*, *If needed* or *No*, like a survey.
- **Open time range**: you pick days and an hours window, and people paint when they're free on a grid (like when2meet).

## Why it's better than Doodle

| | Doodle (free) | Overlap |
|---|---|---|
| Accounts | Organizer must sign up | None, for anyone |
| Ads / tracking | Yes | None |
| Proposing times | Type each option | Pick dates on a calendar, add times with one click, and copy a day's times to every day |
| Answering | Small checkboxes | Big Yes / If needed / No buttons per time with live tallies, or click-and-drag painting for range polls |
| "If needed" answers | Yes | Yes |
| Time zones | Confusing for participants | Every viewer sees times in their own zone, and can switch zones |
| Meeting length | You pre-pick fixed options | Ranks every start time where people can make the *whole* meeting |
| Your calendar | Paid feature | Check an `.ics` export against the proposed times, privately in your browser |
| Finalizing | Paid features for invites | Pick the time, then everyone gets Google / Outlook / Apple calendar links |
| Hosting | Their servers | Yours, free on Cloudflare or any Node host |

## How it works

1. **Create**: name the poll and pick dates on the calendar.
   - *Specific times*: add the times you want to propose for each date. "Use these times on every day" copies them across.
   - *Open range*: set the daily window, slot size (15/30/60 min) and meeting length.
2. **Share**: send the link. People enter their name and answer each time (or paint the grid). They can also check against a calendar file. They don't need an account.
3. **Decide**: **Best times** ranks options by how many people can make it, then by firm yeses. The **Everyone** table shows who answered what, Doodle-style, with the most popular times highlighted.
4. **Finalize**: the organizer picks a time. The poll then shows add-to-calendar buttons for everyone.

Secrets for the organizer and for each participant's edits live in the URL *fragment*, which is never sent to the server, and in the browser's local storage. Only their SHA-256 hashes are stored on the server. Personal "edit links" let people edit their response from another device.

## Run locally

Requires Node 22.13+. No dependencies to install.

```sh
cd overlap
npm start          # http://localhost:8787, data saved in ./data/overlap.db
npm test           # unit + API tests
```

`server.js` also works for self-hosting on any VPS (`PORT` and `DB_PATH` env vars).

## Deploy free on Cloudflare (Workers + D1)

The free tier is plenty for personal and small-team use.

```sh
cd overlap
npx wrangler login
npx wrangler d1 create overlap     # copy the database_id into wrangler.toml
npx wrangler deploy
```

Tables are created automatically on the first request. To try the Worker locally, run `npx wrangler dev`.

## Project layout

```
public/          static front end, no build step
  index.html
  app.js         UI: create page, availability grid, group heatmap, best times
  core.js        shared logic: time zones, slots, ranking, .ics import/export
  styles.css     light + dark themes
src/
  api.js         JSON API (web-standard Request/Response)
  store.js       SQL storage, adapters for D1 and node:sqlite
  worker.js      Cloudflare Worker entry
server.js        Node server (local / self-hosting)
test/            node:test suites
```

### API

| Method | Path | Notes |
|---|---|---|
| `POST` | `/api/events` | Create a poll → `{ id, adminToken }`. Specific times: `{ mode: "options", options: [{ start, end }] }` (UTC ms) |
| `GET` | `/api/events/:id` | Poll + responses (`x-admin-token` header sets `isAdmin`) |
| `DELETE` | `/api/events/:id` | Organizer only |
| `POST` | `/api/events/:id/participants` | `{ name, availability }` → `{ id, editToken }` |
| `PUT` / `DELETE` | `/api/events/:id/participants/:pid` | `x-edit-token` (delete also allows organizer) |
| `POST` | `/api/events/:id/finalize` | `{ start }` or `{ start: null }`, organizer only |
| `GET` | `/api/events/:id/event.ics` | Calendar file for the final time |

`availability` maps an option or slot start (UTC ms) → `2` (yes) or `1` (if needed). Anything missing means no.

## Ideas for next steps

- Email/webhook notifications when someone responds
- Recurring "general week" polls (days of week, not dates)
- Live Google/Microsoft calendar overlay via OAuth (currently `.ics` file import)
- Rate limiting (e.g. Cloudflare's rate-limit binding) before wide public use
