/**
 * Centre pane: the candidates.
 *
 * Every candidate is shown as a *diff* — what it would change, in English — rather than as
 * a wall of replacement text. That is the whole design argument made visible: the writer
 * authorizes changes, they do not adjudicate prose.
 *
 * Four things always appear on a candidate, and all four earn their place:
 *   - the axis label, so three options are three directions rather than three paragraphs;
 *   - the delta summary, so what it changes is readable before it happens;
 *   - the blast radius, so nothing is invalidated by surprise;
 *   - its flags, so a candidate that contradicts the bible says so before it is accepted.
 */

import { useState } from "react";
import type { Candidate } from "../api";
import { FlagList } from "./Flags";

export function CandidateCard(props: {
  candidate: Candidate;
  busy: boolean;
  onAccept: (id: string) => void;
  onReject: (id: string, reason: string) => void;
  onEdit: (id: string, text: string) => void;
  onNavigate: (nodeId: string) => void;
}) {
  const { candidate, busy, onAccept, onReject, onEdit, onNavigate } = props;
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(candidate.text);
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState("");

  const hard = candidate.flags.filter((f) => f.severity === "hard").length;

  return (
    <article className={`candidate ${candidate.selected ? "shortlisted" : "unselected"}`}>
      <header>
        <span className="axis">{candidate.axis_label || candidate.level}</span>
        {!candidate.selected && <span className="badge">not shortlisted</span>}
        {hard > 0 && <span className="badge stale">{hard} contradiction{hard > 1 ? "s" : ""}</span>}
        <span className="scores">
          quality {candidate.base_quality.toFixed(2)} · surprise {candidate.surprise.toFixed(2)}
        </span>
      </header>

      <ul className="delta">
        {candidate.delta_summary.map((line, index) => (
          <li key={index}>{line}</li>
        ))}
      </ul>

      {candidate.stale_preview > 0 && (
        <p className="hint">
          Accepting this would mark {candidate.stale_preview} downstream node
          {candidate.stale_preview > 1 ? "s" : ""} for review. Nothing is rewritten.
        </p>
      )}

      {candidate.rationale && <p className="rationale">{candidate.rationale}</p>}

      {candidate.notes.length > 0 && (
        <ul className="notes">
          {candidate.notes.map((note, index) => (
            <li key={index}>{note}</li>
          ))}
        </ul>
      )}

      <FlagList flags={candidate.flags} onNavigate={onNavigate} />

      {editing ? (
        <div className="stack">
          <textarea rows={8} value={draft} onChange={(event) => setDraft(event.target.value)} />
          <div className="actions">
            <button
              className="primary"
              disabled={busy}
              onClick={() => {
                setEditing(false);
                onEdit(candidate.proposal_id, draft);
              }}
            >
              accept my version
            </button>
            <button onClick={() => setEditing(false)}>cancel</button>
          </div>
          <p className="hint">
            Your edit is kept as a before/after pair — it is the strongest signal the system
            gets about how you want prose written.
          </p>
        </div>
      ) : rejecting ? (
        <div className="stack">
          <input
            type="text"
            placeholder="Why not? (this becomes a standing instruction you can delete)"
            value={reason}
            onChange={(event) => setReason(event.target.value)}
          />
          <div className="actions">
            <button
              disabled={busy}
              onClick={() => {
                setRejecting(false);
                onReject(candidate.proposal_id, reason);
              }}
            >
              reject
            </button>
            <button onClick={() => setRejecting(false)}>cancel</button>
          </div>
        </div>
      ) : (
        <div className="actions">
          <button className="primary" disabled={busy} onClick={() => onAccept(candidate.proposal_id)}>
            accept
          </button>
          <button
            disabled={busy}
            onClick={() => {
              setDraft(candidate.text);
              setEditing(true);
            }}
          >
            edit
          </button>
          <button disabled={busy} onClick={() => setRejecting(true)}>
            reject
          </button>
        </div>
      )}
    </article>
  );
}
