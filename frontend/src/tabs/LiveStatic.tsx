/**
 * The Live tab, as it appears on the published static site.
 *
 * The tool needs a running Python process and provider keys, neither of which a static
 * host has. Rather than ship a Live tab that shows an error, the static build replaces it
 * with screenshots of the real thing and the commands to run it.
 *
 * The rule this page follows: no caption may promise something the page can do. Every
 * verb here describes the local application, not this page.
 */

import { useState } from "react";

interface Shot {
  file: string;
  title: string;
  caption: string;
}

const SHOTS: Shot[] = [
  {
    file: "shots/live-tree.png",
    title: "The plan tree",
    caption:
      "Story, episodes, scenes, beats and prose, each node carrying its status as a word " +
      "rather than only a colour, with its lock, its staleness reason and its flag count.",
  },
  {
    file: "shots/live-candidates.png",
    title: "Candidates as diffs",
    caption:
      "Each candidate is shown as what it would change, under the axis it was generated " +
      "along, with its continuity flags and the number of downstream nodes it would mark.",
  },
  {
    file: "shots/live-world.png",
    title: "World state at a beat",
    caption:
      "What is true at the selected beat, who has been told it, the open threads with how " +
      "many beats each has gone untouched, and the bible diff from the last accept.",
  },
  {
    file: "shots/live-audit.png",
    title: "The whole-graph audit",
    caption:
      "Layers 1 and 2 over the entire story, rather than the incremental check that runs " +
      "on each accept. Flags cite the beat that established the fact they contradict.",
  },
];

const QUICKSTART = `git clone <repo-url> && cd storygit
make setup                 # venv, dependencies, frontend build
cp .env.example .env       # add provider keys for generation
make demo                  # http://127.0.0.1:8000`;

export function LiveStatic() {
  const [open, setOpen] = useState<Shot | null>(null);

  return (
    <section className="prose">
      <h1>Live</h1>
      <p>
        This is a static page. The tool itself is a local application: one Python process
        serving an API and the built frontend from the same origin, over a SQLite file you
        can copy. Below is what it looks like, and how to run it.
      </p>

      <div className="notice">
        <strong>Browsing needs no keys.</strong> The plan tree, world state, continuity
        audit, snapshot history, branches and the recorded Gallery sessions all work from
        the committed demo story with no <code>.env</code> at all. Only generation calls a
        model.
      </div>

      <h2>Running it</h2>
      <pre>
        <code>{QUICKSTART}</code>
      </pre>
      <p>
        <code>make demo</code> serves the finished two-episode story that a writer produced
        through this system, so the tree, the flags and the world state are populated on the
        first screen rather than empty.
      </p>

      <h2>The three panes</h2>
      <div className="shot-grid">
        {SHOTS.map((shot) => (
          <figure key={shot.file} className="shot">
            <button
              className="shot-button"
              onClick={() => setOpen(shot)}
              title="Enlarge"
              aria-label={`Enlarge: ${shot.title}`}
            >
              <img src={shot.file} alt={shot.title} loading="lazy" />
            </button>
            <figcaption>
              <strong>{shot.title}.</strong> {shot.caption}
            </figcaption>
          </figure>
        ))}
      </div>

      <p>
        The <strong>Gallery</strong> tab on this site is not a screenshot: those sessions
        replay from recorded JSON and run here exactly as they run locally, which is the
        closest this page gets to the tool moving.
      </p>

      {open && (
        <div
          className="lightbox"
          onClick={() => setOpen(null)}
          role="dialog"
          aria-modal="true"
          aria-label={open.title}
        >
          <img src={open.file} alt={open.title} />
          <p>{open.caption}</p>
        </div>
      )}
    </section>
  );
}
