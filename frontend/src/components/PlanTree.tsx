/**
 * Left pane: the plan tree.
 *
 * Status is carried by a coloured dot *and* a word. Colour alone would fail for anyone who
 * reads it differently, and the palette is deliberately muted enough that the four statuses
 * are not far apart in hue.
 *
 * Stale nodes show their reason on hover, and the reason always names the beat that
 * established the fact that moved — that citation is the whole point of the mark, and
 * burying it behind a click would waste it.
 */

import type { NodeStatus, NodeSummary, TreeResponse } from "../api";

const DEPTH: Record<string, number> = { story: 0, episode: 1, scene: 2, beat: 3, prose: 4 };

function statusWord(node: NodeSummary): string {
  if (node.locked) return "locked";
  if (node.status === "stale") return "stale";
  if (node.stale_reason) return "review";
  return node.status;
}

export function PlanTree(props: {
  tree: TreeResponse | null;
  selected: string | null;
  onSelect: (id: string) => void;
  onToggleLock: (node: NodeSummary) => void;
}) {
  const { tree, selected, onSelect, onToggleLock } = props;
  if (!tree) return <p className="empty">Loading the story…</p>;
  if (tree.nodes.length === 0) return <p className="empty">Nothing here yet.</p>;

  return (
    <ul className="tree">
      {tree.nodes.map((node) => {
        const word = statusWord(node);
        const dotClass: NodeStatus | "review" = node.locked
          ? "locked"
          : node.status === "stale"
            ? "stale"
            : node.status;
        return (
          <li key={node.id}>
            <div
              className={`row depth-${DEPTH[node.node_type] ?? 0}`}
              aria-current={node.id === selected}
              onClick={() => onSelect(node.id)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") onSelect(node.id);
              }}
              role="button"
              tabIndex={0}
              title={node.stale_reason ?? undefined}
            >
              <span className={`dot ${dotClass}`} aria-hidden="true" />
              <span className="label">{node.title || node.node_type}</span>
              {node.flag_count > 0 && (
                <span className="badge stale" title={`${node.flag_count} continuity flag(s)`}>
                  {node.flag_count}
                </span>
              )}
              {(word === "stale" || word === "review" || word === "locked") && (
                <span className={`badge ${word}`}>{word}</span>
              )}
              {node.node_type !== "story" && (
                <button
                  className="quiet"
                  title={node.locked ? "Unlock this node" : "Lock this node"}
                  onClick={(event) => {
                    event.stopPropagation();
                    onToggleLock(node);
                  }}
                >
                  {node.locked ? "unlock" : "lock"}
                </button>
              )}
            </div>
          </li>
        );
      })}
    </ul>
  );
}
