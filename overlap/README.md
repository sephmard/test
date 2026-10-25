# Overlap

A free, open-source alternative to Doodle for finding a time to meet. Create a poll, share one link, and everyone paints when they're free. Overlap shows the best time straight away.

## Why it's better than Doodle

| | Doodle (free) | Overlap |
|---|---|---|
| Accounts | Organizer must sign up | None, for anyone |
| Ads / tracking | Yes | None |
| Picking availability | One checkbox per option | Click-and-drag painting across days and hours (touch friendly) |
| "If needed" answers | Yes | Yes |
| Time zones | Confusing for participants | Every viewer sees times in their own zone, and can switch zones |
| Meeting length | You pre-pick fixed options | Ranks every start time where people can make the *whole* meeting |
| Your calendar | Paid feature | Import an `.ics` export to fill in your free time, privately in your browser |
| Finalizing | Paid features for invites | Pick the time, then everyone gets Google / Outlook / Apple calendar links |
| Hosting | Their servers | Yours, free on Cloudflare or any Node host |

## How it works

1. **Create**: name the poll, drag across a calendar to choose dates, set the daily window, slot size (15/30/60 min) and meeting length.
2. **Share**: send the link. People enter a name and paint *Available*, *If needed* or *Busy*. They can also import a calendar file.
3. **Decide**: the **Best times** list ranks start times by how many people can attend the full meeting. Hover a cell to see who's in. Tap a name to view just their availability.
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
| `POST` | `/api/events` | Create a poll → `{ id, adminToken }` |
| `GET` | `/api/events/:id` | Poll + responses (`x-admin-token` header sets `isAdmin`) |
| `DELETE` | `/api/events/:id` | Organizer only |
| `POST` | `/api/events/:id/participants` | `{ name, availability }` → `{ id, editToken }` |
| `PUT` / `DELETE` | `/api/events/:id/participants/:pid` | `x-edit-token` (delete also allows organizer) |
| `POST` | `/api/events/:id/finalize` | `{ start }` or `{ start: null }`, organizer only |
| `GET` | `/api/events/:id/event.ics` | Calendar file for the final time |

`availability` maps slot start (UTC ms) → `2` (available) or `1` (if needed).

## Ideas for next steps

- Email/webhook notifications when someone responds
- Recurring "general week" polls (days of week, not dates)
- Live Google/Microsoft calendar overlay via OAuth (currently `.ics` file import)
- Rate limiting (e.g. Cloudflare's rate-limit binding) before wide public use
