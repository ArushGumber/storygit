# `frontend`

Vite + React + TypeScript (strict), plain CSS, no UI framework. Five tabs.

## The visual contract

From the design brief, and treated as a contract rather than a suggestion: light
background, **one typeface**, a muted palette matching `docs/diagrams/style.tex`, no
gradients, no icons used as decoration, no emoji, no card shadows. It should look like a
serious writing tool, not a dashboard.

The palette values in `styles.css` are literally the ones in the TikZ style file, so a
diagram dropped into the Architecture tab belongs there rather than merely coexisting with
the page. `docs/diagrams/build.sh` copies the SVGs into `public/diagrams/` on every build,
so the two cannot drift.

Two accessibility decisions worth naming. **Status is never carried by colour alone** — the
plan tree shows a coloured dot *and* the word, because the palette is deliberately muted
enough that four statuses are close in hue. And **monospace appears only for identifiers**,
never for prose, so the one-typeface rule holds where it matters.

## The tabs, in the order the story is told

| Tab | What it is for |
|---|---|
| Problem & Decisions | The problem, Fabula and its findings, the thesis, and each decision with the alternative it rejected. |
| Architecture | The six diagrams, with a paragraph each, and the package map. |
| Live | The tool. Three panes. |
| Gallery | Recorded sessions, replayed step by step from real snapshots. |
| Eval | The numbers, with what each one demonstrates. |

Someone who has never heard of this project should be able to read left to right and
understand what was built, how it works, what it does, that it does it, and how well.

## The Live tab

**Left — the plan tree, and the story's history.** Status by dot and word, lock toggles,
stale badges whose tooltip is the reason, and the reason always names the beat that
established the fact that moved. The branch selector sits here: a branch is a pointer at a
snapshot, so exploring costs nothing. Below it, branching, comparing against `main`, and a
three-way merge that previews first and refuses to commit through a conflict; the
whole-story audit, which walks the graph rather than the last accept and names the threads
that have stopped moving; and the snapshot chain itself, which is the product's thesis made
visible.

**Centre — the work.** The selected node, an intent box, and the candidates. Every candidate
is a **diff**: what it would change in English, what it would invalidate, its axis label,
its rationale, and its continuity flags. Accept, edit in place, or reject with a reason.
Below them, the hand-write path for beats — the writer types, the system only reads.

**Right — the world.** The entities in this scene, the facts true here with who established
them and who has been told, open threads with how long they have gone untouched, hard
constraints, the authorship ratio, the dial, and the writer's own rules and criteria —
including the ones mined from their edits, each marked as mined and each with a control
that deletes it. Every entity carries a `merge` control, because alias resolution is
deliberately conservative and will leave a duplicate rather than risk welding two
characters together; that caution is only safe because this correction exists.

**Nothing is API-only.** A capability the writer cannot reach is a capability the product
does not have, so every backend capability has a control here — and
`test_no_client_method_is_unreachable_from_the_interface` fails if a method in `api.ts` is
called by no component.

## Two rules the implementation follows

**Nothing is mutated client-side.** Every action posts, and the panes reload from the
server. A client that optimistically updated its own copy would eventually show a story
that differs from the one in the snapshot — the exact failure this whole system exists to
prevent.

**Every failure is visible and recoverable.** A proposal that times out leaves an error and
a retry, never a blank pane. `ApiError` carries the backend's typed `kind` and, for a rate
limit, `retry_after` — so the interface can say *"try again in 45 seconds"* rather than
*"something went wrong"*.

## Running it

```bash
npm install
npm run dev        # 5173, proxying /api to 127.0.0.1:8000
npm run build      # tsc -b && vite build -> dist/
npm run typecheck  # tsc --noEmit
npm test           # vitest, over the client's error handling
```

The unit tests cover `src/api.ts` and nothing else, deliberately. It is the one piece of
frontend logic that is not obviously correct by inspection, and it is the piece a writer
notices when it is wrong: the difference between *"try again in 45 seconds"*, *"the server
is not running"*, and *"something went wrong"* is the difference between a tool that
recovers and one that just fails. Components are covered by the end-to-end smoke and the
screenshot audit, which is where component bugs actually show up.

In production `dist/` is served by the same uvicorn process as the API, on one port.

## Auditing the visual contract

`scripts/screenshots.py` boots a real server, builds a small story, asks for candidates,
and shoots every tab. The prohibition list is only checkable against pixels, so the
screenshots are the audit — they live in `arush/logs/chunk_6_screens/`.
