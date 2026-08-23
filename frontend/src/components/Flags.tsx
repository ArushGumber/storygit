/**
 * Continuity flags.
 *
 * Hard flags (deterministic contradictions) and soft flags (model opinions) are rendered
 * differently and are never mixed: a solid rule against a dashed one, full colour against
 * grey. A writer who cannot tell an opinion from a contradiction learns to distrust both.
 *
 * Every layer-1 and layer-2 flag carries the beat that established the fact it conflicts
 * with, and that citation is a link — the whole value of the flag is that you can go and
 * look.
 */

import type { Flag } from "../api";

export function FlagList(props: {
  flags: Flag[];
  onNavigate?: (nodeId: string) => void;
  empty?: string;
}) {
  const { flags, onNavigate, empty } = props;
  if (flags.length === 0) {
    return empty ? <p className="empty">{empty}</p> : null;
  }
  return (
    <div>
      {flags.map((flag, index) => (
        <div key={`${flag.kind}-${index}`} className={`flag ${flag.severity}`}>
          <div>{flag.message}</div>
          <div className="cite">
            {flag.severity === "hard" ? "contradiction" : "possible issue"} · layer {flag.layer}
            {flag.established_by && onNavigate && (
              <>
                {" · "}
                <a
                  href="#"
                  onClick={(event) => {
                    event.preventDefault();
                    onNavigate(flag.established_by as string);
                  }}
                >
                  go to where this was established
                </a>
              </>
            )}
            {flag.score !== null && ` · confidence ${flag.score.toFixed(2)}`}
          </div>
        </div>
      ))}
    </div>
  );
}
