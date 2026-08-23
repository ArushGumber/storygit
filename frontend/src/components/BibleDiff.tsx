/**
 * What an accepted change did to the world, shown before the writer moves on.
 *
 * `+` added, `~` ended (true until here, not any more), `-` struck. The three are different
 * in kind and are rendered differently: adding a fact is routine, ending one has downstream
 * consequences, and striking one means it was never true.
 *
 * Stale marks appear here too, because they are the other half of the same answer to
 * "what did I just agree to".
 */

import type { ActionResponse } from "../api";
import { FlagList } from "./Flags";

export function BibleDiffModal(props: {
  result: ActionResponse;
  onClose: () => void;
  onNavigate: (nodeId: string) => void;
}) {
  const { result, onClose, onNavigate } = props;
  return (
    <div className="overlay" onClick={onClose}>
      <div className="modal" onClick={(event) => event.stopPropagation()}>
        <h2>What changed</h2>

        {result.bible_diff.length === 0 ? (
          <p className="empty">Nothing in the world state changed.</p>
        ) : (
          <ul className="bible">
            {result.bible_diff.map((line, index) => {
              const mark = line.slice(0, 1);
              const cls = mark === "+" ? "add" : mark === "~" ? "end" : "remove";
              return (
                <li key={index}>
                  <span className={cls}>{mark}</span>
                  <span>{line.slice(2)}</span>
                </li>
              );
            })}
          </ul>
        )}

        {result.extracted && (
          <p className="hint">
            Facts were read back out of the prose, so continuity checking sees what was
            actually written rather than what was planned.
          </p>
        )}

        {result.marks.length > 0 && (
          <>
            <h2>Marked for your attention</h2>
            <ul className="facts">
              {result.marks.map((mark, index) => (
                <li key={index}>
                  <div className="row-between">
                    <span>{mark.reason}</span>
                    <button className="quiet" onClick={() => onNavigate(mark.node_id)}>
                      go there
                    </button>
                  </div>
                  <div className="meta">
                    {mark.kind === "review"
                      ? "your prose — flagged, never staled"
                      : mark.kind === "maybe_affected"
                        ? "not a declared dependency — check if it still works"
                        : "stale — regenerate, edit, or dismiss"}
                  </div>
                </li>
              ))}
            </ul>
          </>
        )}

        {result.flags.length > 0 && (
          <>
            <h2>Continuity</h2>
            <FlagList flags={result.flags} onNavigate={onNavigate} />
          </>
        )}

        <div className="actions">
          <button className="primary" onClick={onClose}>
            got it
          </button>
        </div>
      </div>
    </div>
  );
}
