/**
 * The Gallery tab: recorded sessions, replayed step by step.
 *
 * Every step names the snapshot it happened at, and the plan tree and world state shown
 * beside it are read back out of that snapshot. So the Gallery cannot show something the
 * system did not do, and replaying makes no model calls — which is what makes it a record
 * rather than a recording.
 */

import { useEffect, useState } from "react";
import { ApiError, api, type GalleryResponse, type GallerySessionIndex } from "../api";
import { FlagList } from "../components/Flags";

export function Gallery() {
  const [sessions, setSessions] = useState<GallerySessionIndex[]>([]);
  const [active, setActive] = useState<string | null>(null);
  const [session, setSession] = useState<GalleryResponse | null>(null);
  const [step, setStep] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      try {
        const response = await api.gallery();
        setSessions(response.sessions);
        setActive((current) => current ?? response.sessions[0]?.name ?? null);
      } catch (caught) {
        setError(caught instanceof ApiError ? caught.message : String(caught));
      }
    })();
  }, []);

  useEffect(() => {
    if (!active) return;
    void (async () => {
      try {
        setSession(await api.gallerySession(active));
        setStep(0);
      } catch (caught) {
        setError(caught instanceof ApiError ? caught.message : String(caught));
      }
    })();
  }, [active]);

  if (error) {
    return (
      <div className="prose-page">
        <h1>Gallery</h1>
        <div className="error">{error}</div>
      </div>
    );
  }

  if (sessions.length === 0) {
    return (
      <div className="prose-page">
        <h1>Gallery</h1>
        <p className="empty">
          No sessions recorded yet. Run <code>python -m eval.record_gallery</code>.
        </p>
      </div>
    );
  }

  const steps = session?.session.steps ?? [];
  const current = steps[step];
  const replayed = session?.replay?.[step];

  return (
    <div className="prose-page">
      <h1>Gallery</h1>
      <p className="lede">
        Recorded sessions, replayed from the snapshots they happened at. Nothing here is a
        mock-up — each step is a real commit, and replaying it makes no model calls.
      </p>

      <h2>Sessions</h2>
      <div className="stack">
        {sessions.map((entry) => (
          <div
            key={entry.name}
            className={`decision${entry.name === active ? "" : ""}`}
            style={{ borderColor: entry.name === active ? "var(--accent)" : undefined }}
          >
            <div className="row-between">
              <h3 style={{ margin: 0 }}>{entry.title}</h3>
              <button
                className={entry.name === active ? "primary" : ""}
                onClick={() => setActive(entry.name)}
              >
                {entry.name === active ? "showing" : `replay (${entry.steps} steps)`}
              </button>
            </div>
            <p style={{ marginBottom: 0 }}>{entry.summary}</p>
          </div>
        ))}
      </div>

      {session && current && (
        <>
          <h2>
            Step {step + 1} of {steps.length}: {current.title}
          </h2>
          <div className="inline">
            <button disabled={step === 0} onClick={() => setStep((s) => s - 1)}>
              previous
            </button>
            <button disabled={step >= steps.length - 1} onClick={() => setStep((s) => s + 1)}>
              next
            </button>
            {current.snapshot_id && (
              <span className="hint mono">snapshot {current.snapshot_id.slice(0, 12)}</span>
            )}
          </div>

          {current.note && <p>{current.note}</p>}

          {current.shown.length > 0 && (
            <>
              <h3>What the writer was shown</h3>
              {current.shown.map((candidate) => (
                <article
                  key={candidate.proposal_id}
                  className={`candidate ${candidate.selected ? "shortlisted" : "unselected"}`}
                >
                  <header>
                    <span className="axis">{candidate.axis_label}</span>
                    {!candidate.selected && <span className="badge">not shortlisted</span>}
                    <span className="scores">
                      quality {candidate.base_quality.toFixed(2)} · surprise{" "}
                      {candidate.surprise.toFixed(2)} · ranked{" "}
                      {candidate.effective_quality.toFixed(2)}
                    </span>
                  </header>
                  <ul className="delta">
                    {candidate.delta_summary.map((line, index) => (
                      <li key={index}>{line}</li>
                    ))}
                  </ul>
                  {candidate.proposal_id === current.chosen && (
                    <p className="hint">The writer took this one.</p>
                  )}
                </article>
              ))}
            </>
          )}

          {current.bible_diff.length > 0 && (
            <>
              <h3>What changed in the world</h3>
              <ul className="bible">
                {current.bible_diff.map((line, index) => (
                  <li key={index}>
                    <span
                      className={
                        line.startsWith("+") ? "add" : line.startsWith("~") ? "end" : "remove"
                      }
                    >
                      {line.slice(0, 1)}
                    </span>
                    <span>{line.slice(2)}</span>
                  </li>
                ))}
              </ul>
            </>
          )}

          {current.marks.length > 0 && (
            <>
              <h3>What was marked</h3>
              <ul className="facts">
                {current.marks.map((mark, index) => (
                  <li key={index}>
                    <div>{mark.reason}</div>
                    <div className="meta">
                      {mark.kind === "review"
                        ? "flagged for review — human prose is never staled"
                        : mark.kind === "maybe_affected"
                          ? "not a declared dependency"
                          : "stale"}
                    </div>
                  </li>
                ))}
              </ul>
            </>
          )}

          {current.flags.length > 0 && (
            <>
              <h3>Continuity</h3>
              <FlagList flags={current.flags} />
            </>
          )}

          {replayed && replayed.tree.length > 0 && (
            <>
              <h3>The story at this point</h3>
              <table>
                <thead>
                  <tr>
                    <th>node</th>
                    <th>status</th>
                    <th>why</th>
                  </tr>
                </thead>
                <tbody>
                  {replayed.tree
                    .filter((node) => node.type !== "story")
                    .map((node) => (
                      <tr key={node.id}>
                        <td>{node.title || node.type}</td>
                        <td>{node.status}</td>
                        <td>{node.stale_reason ?? ""}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
              {replayed.facts.length > 0 && (
                <>
                  <h3>What is true</h3>
                  <ul className="facts">
                    {replayed.facts.map((fact, index) => (
                      <li key={index}>{fact}</li>
                    ))}
                  </ul>
                </>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
