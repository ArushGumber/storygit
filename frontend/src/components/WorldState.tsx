/**
 * Right pane: the world as it stands here.
 *
 * Facts are shown as sentences with their validity and the beat that established them,
 * because "Kael is at Ashfall, established in *Caught in the market*, still true" is
 * something a novelist can check, and a row of typed triples is not.
 *
 * Every fact can be struck. Striking is not an undo — it produces a diff that propagates
 * like any other change — so the button says what it does.
 */

import type {
  AuthorshipResponse,
  Entity,
  FactView,
  SliceResponse,
  Thread,
} from "../api";

function ThreadRow(props: { thread: Thread; sinceTouched?: number }) {
  const { thread, sinceTouched } = props;
  const stale = (sinceTouched ?? 0) >= 12;
  return (
    <li>
      <div>{thread.description}</div>
      <div className="meta">
        {thread.status}
        {sinceTouched !== undefined && ` · untouched for ${sinceTouched} beats`}
        {stale && " — worth paying off or dropping deliberately"}
      </div>
    </li>
  );
}

export function WorldState(props: {
  slice: SliceResponse | null;
  threadAges: Record<string, number>;
  authorship: AuthorshipResponse | null;
  onStrike: (fact: FactView) => void;
  onNavigate: (nodeId: string) => void;
  onMerge: (source: Entity) => void;
}) {
  const { slice, threadAges, authorship, onStrike, onNavigate, onMerge } =
    props;
  if (!slice)
    return <p className="empty">Select a node to see the world around it.</p>;

  const ratios = authorship?.overall ?? {};
  const human = ratios.human ?? 0;
  const edited = ratios.ai_edited_by_human ?? 0;
  const ai = ratios.ai ?? 0;

  return (
    <>
      <h2>In this scene</h2>
      {slice.entities.length === 0 ? (
        <p className="empty">No entities established here yet.</p>
      ) : (
        <div>
          {slice.entities.map((entity) => (
            <span key={entity.id} className="entity" title={entity.description}>
              {entity.name}
              {entity.aliases.length > 0 && ` (${entity.aliases.join(", ")})`}
              <button
                className="quiet"
                title="Already in the bible under another name? Fold this one into it."
                onClick={() => onMerge(entity)}
              >
                merge
              </button>
            </span>
          ))}
        </div>
      )}

      <h2>True right now</h2>
      {slice.facts.length === 0 ? (
        <p className="empty">Nothing established about these entities yet.</p>
      ) : (
        <ul className="facts">
          {slice.facts.map((view) => (
            <li key={view.fact.id}>
              <div className="row-between">
                <span>{view.sentence}</span>
                <button
                  className="quiet"
                  title="Strike this fact. It propagates like any other change."
                  onClick={() => onStrike(view)}
                >
                  strike
                </button>
              </div>
              <div className="meta">
                established in{" "}
                <a
                  href="#"
                  onClick={(event) => {
                    event.preventDefault();
                    onNavigate(view.fact.established_by_beat);
                  }}
                >
                  {view.established_in}
                </a>
                {view.fact.valid_until_beat
                  ? " · no longer true later"
                  : " · still true"}
                {view.known_by.length > 0 &&
                  ` · known by ${view.known_by.join(", ")}`}
                {view.known_by.length === 0 && " · nobody has been told"}
              </div>
            </li>
          ))}
        </ul>
      )}

      <h2>Open threads</h2>
      {slice.threads.length === 0 ? (
        <p className="empty">No open threads.</p>
      ) : (
        <ul className="facts">
          {slice.threads.map((thread) => (
            <ThreadRow
              key={thread.id}
              thread={thread}
              sinceTouched={threadAges[thread.id]}
            />
          ))}
        </ul>
      )}

      {slice.hard_constraints.length > 0 && (
        <>
          <h2>Hard constraints</h2>
          <ul className="facts">
            {slice.hard_constraints.map((line, index) => (
              <li key={index}>{line}</li>
            ))}
          </ul>
        </>
      )}

      <h2>Who wrote this</h2>
      {authorship && authorship.sentences > 0 ? (
        <>
          <div className="bar">
            <span className="human" style={{ width: `${human * 100}%` }} />
            <span
              className="ai_edited_by_human"
              style={{ width: `${edited * 100}%` }}
            />
            <span className="ai" style={{ width: `${ai * 100}%` }} />
          </div>
          <div className="legend">
            <span className="human">you {Math.round(human * 100)}%</span>
            <span className="edited">
              your edits {Math.round(edited * 100)}%
            </span>
            <span>ai {Math.round(ai * 100)}%</span>
          </div>
          <p className="hint">
            {authorship.sentences}{" "}
            {authorship.sentences === 1 ? "sentence" : "sentences"} written so
            far.
          </p>
        </>
      ) : (
        <p className="empty">No prose written yet.</p>
      )}
    </>
  );
}
