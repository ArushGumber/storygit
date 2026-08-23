/**
 * The Live tab: three panes, exactly as the design contract specifies.
 *
 *   left   — the plan tree, with status, locks, stale badges, and the branch selector
 *   centre — the selected node, the intent box, k labelled candidates as diffs, the dial
 *   right  — the world state here, continuity flags, authorship, the writer ledger
 *
 * Two rules the implementation follows, worth stating because they are easy to break:
 *
 * Nothing is ever mutated client-side. Every action posts, and the panes reload from the
 * server afterwards. A client that optimistically updated its own copy would eventually
 * show a story that differs from the one in the snapshot — the exact failure this whole
 * system exists to prevent.
 *
 * Every failure is visible and recoverable. A proposal that times out leaves an error and
 * a retry, never a blank pane.
 */

import { useCallback, useEffect, useState } from "react";
import {
  ApiError,
  api,
  type ActionResponse,
  type AuthorshipResponse,
  type BranchesResponse,
  type Candidate,
  type FactView,
  type LedgerResponse,
  type NodeDetail,
  type NodeSummary,
  type SliceResponse,
  type TreeResponse,
} from "../api";
import { BibleDiffModal } from "../components/BibleDiff";
import { CandidateCard } from "../components/Candidates";
import { FlagList } from "../components/Flags";
import { PlanTree } from "../components/PlanTree";
import { WorldState } from "../components/WorldState";

const NEXT_LEVEL: Record<string, string> = {
  story: "episode",
  episode: "scene",
  scene: "beat",
  beat: "prose",
  prose: "prose",
};

export function Live() {
  const [tree, setTree] = useState<TreeResponse | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<NodeDetail | null>(null);
  const [slice, setSlice] = useState<SliceResponse | null>(null);
  const [ledger, setLedger] = useState<LedgerResponse | null>(null);
  const [authorship, setAuthorship] = useState<AuthorshipResponse | null>(null);
  const [branches, setBranches] = useState<BranchesResponse | null>(null);
  const [threadAges, setThreadAges] = useState<Record<string, number>>({});

  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [intent, setIntent] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [result, setResult] = useState<ActionResponse | null>(null);
  const [writing, setWriting] = useState(false);
  const [draft, setDraft] = useState("");

  const refresh = useCallback(async () => {
    try {
      const [nextTree, nextLedger, nextAuthorship, nextBranches, threads] = await Promise.all([
        api.tree(),
        api.ledger(),
        api.authorship(),
        api.branches(),
        api.threads(),
      ]);
      setTree(nextTree);
      setLedger(nextLedger);
      setAuthorship(nextAuthorship);
      setBranches(nextBranches);
      setThreadAges(
        Object.fromEntries(threads.threads.map((t) => [t.thread.id, t.beats_since_touched])),
      );
      // Default to the deepest node that can still be proposed *into*, which is where a
      // writer picking the tool back up wants to be. Landing on a prose node would show
      // them the end of what exists rather than the place to continue from.
      setSelected((current) => {
        if (current) return current;
        const openable = nextTree.nodes.filter((node) => node.node_type !== "prose");
        return openable.at(-1)?.id ?? nextTree.nodes.at(-1)?.id ?? null;
      });
    } catch (caught) {
      if (caught instanceof ApiError) setError(caught);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (!selected) return;
    void (async () => {
      try {
        const [nextDetail, nextSlice] = await Promise.all([api.node(selected), api.slice(selected)]);
        setDetail(nextDetail);
        setSlice(nextSlice);
      } catch (caught) {
        if (caught instanceof ApiError) setError(caught);
      }
    })();
  }, [selected, tree]);

  async function run<T>(label: string, work: () => Promise<T>): Promise<T | null> {
    setBusy(label);
    setError(null);
    try {
      return await work();
    } catch (caught) {
      setError(
        caught instanceof ApiError ? caught : new ApiError(0, "Unknown", String(caught), null),
      );
      return null;
    } finally {
      setBusy(null);
    }
  }

  const level = detail ? (NEXT_LEVEL[detail.node.node_type] ?? "beat") : "beat";

  async function propose() {
    const response = await run("propose", () => api.propose(selected, level, intent));
    if (response) setCandidates(response.candidates);
  }

  async function afterAction(response: ActionResponse | null) {
    if (!response) return;
    setResult(response);
    setCandidates([]);
    setIntent("");
    await refresh();
  }

  return (
    <div className="live">
      <section className="pane">
        <div className="row-between">
          <h2>Plan</h2>
          {branches && (
            <select
              value={branches.current}
              onChange={(event) =>
                void run("branch", async () => {
                  await api.switchBranch(event.target.value);
                  await refresh();
                })
              }
              title="Branch. Exploring costs nothing; a branch is a pointer at a snapshot."
            >
              {Object.keys(branches.branches).map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          )}
        </div>

        <PlanTree
          tree={tree}
          selected={selected}
          onSelect={setSelected}
          onToggleLock={(node: NodeSummary) =>
            void run("lock", async () => {
              await (node.locked ? api.unlock(node.id) : api.lock(node.id));
              await refresh();
            })
          }
        />

        {tree && (tree.stale_count > 0 || tree.review_count > 0) && (
          <p className="hint">
            {tree.stale_count} stale, {tree.review_count} flagged for review. Nothing was
            rewritten.
          </p>
        )}

        <h2>Branches</h2>
        <button
          onClick={() =>
            void run("branch", async () => {
              const name = window.prompt("Name this line of exploration");
              if (name) await api.createBranch(name);
              await refresh();
            })
          }
        >
          branch from here
        </button>
      </section>

      <section className="pane">
        {!detail ? (
          <p className="empty">Select something on the left.</p>
        ) : (
          <>
            <div className="row-between">
              <h2>
                {detail.node.node_type} · {detail.node.title}
              </h2>
              {detail.node.status === "stale" && (
                <div className="inline">
                  <button
                    onClick={() =>
                      void run("regen", async () => {
                        const response = await api.regenerate(detail.node.id, false);
                        setCandidates(response.candidates);
                      })
                    }
                  >
                    regenerate (previewed)
                  </button>
                  <button
                    onClick={() =>
                      void run("dismiss", async () => {
                        await api.dismissStale(detail.node.id);
                        await refresh();
                      })
                    }
                    title="It still works"
                  >
                    dismiss
                  </button>
                </div>
              )}
            </div>

            {detail.node.stale_reason && (
              <div className="flag">
                <div>{detail.node.stale_reason}</div>
                <div className="cite">
                  Regenerate it, edit it yourself, or dismiss the mark. Nothing has been
                  changed for you.
                </div>
              </div>
            )}

            {detail.what_happens && <p>{detail.what_happens}</p>}
            {detail.audience_learns && (
              <p className="hint">the audience learns: {detail.audience_learns}</p>
            )}
            {detail.episode && (
              <div className="stack">
                {["hook", "cliffhanger", "recap_of_previous"].map((key) =>
                  detail.episode?.[key] ? (
                    <p key={key} className="hint">
                      <strong>{key.replace(/_/g, " ")}:</strong> {String(detail.episode[key])}
                    </p>
                  ) : null,
                )}
              </div>
            )}

            {detail.prose && (
              <>
                <h2>Prose</h2>
                <p style={{ whiteSpace: "pre-wrap" }}>{detail.prose}</p>
              </>
            )}

            <FlagList flags={detail.flags} onNavigate={setSelected} />

            <h2>{level === "prose" ? "Write this beat" : `Ask for the next ${level}`}</h2>
            <div className="stack">
              <input
                type="text"
                placeholder={
                  level === "prose"
                    ? "How should this beat read?"
                    : `What should happen in the next ${level}?`
                }
                value={intent}
                onChange={(event) => setIntent(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !busy) void propose();
                }}
              />
              <div className="inline">
                <button className="primary" disabled={busy !== null} onClick={() => void propose()}>
                  {busy === "propose" ? "thinking…" : "propose"}
                </button>
                {detail.node.node_type === "beat" && (
                  <button disabled={busy !== null} onClick={() => setWriting((w) => !w)}>
                    write it myself
                  </button>
                )}
              </div>
            </div>

            {busy === "propose" && (
              <p className="working">
                Sampling six options along different directions, checking each against the
                bible, and picking three.
              </p>
            )}

            {error && (
              <div className="error">
                <strong>{error.kind}</strong> — {error.message}
                {error.retryAfter !== null && ` Try again in ${Math.ceil(error.retryAfter)}s.`}
                {error.retryable && (
                  <div className="actions">
                    <button onClick={() => void propose()}>retry</button>
                  </div>
                )}
              </div>
            )}

            {writing && (
              <div className="stack">
                <textarea
                  rows={10}
                  placeholder="Write it yourself. Facts are read back out of it, so continuity keeps working."
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                />
                <div className="inline">
                  <button
                    className="primary"
                    disabled={busy !== null || !draft.trim()}
                    onClick={() =>
                      void run("write", async () => {
                        const response = await api.write(detail.node.id, draft);
                        setDraft("");
                        setWriting(false);
                        await afterAction(response);
                      })
                    }
                  >
                    save
                  </button>
                  <button onClick={() => setWriting(false)}>cancel</button>
                </div>
              </div>
            )}

            {candidates.length > 0 && (
              <>
                <h2>{candidates.filter((c) => c.selected).length} options</h2>
                {candidates.map((candidate) => (
                  <CandidateCard
                    key={candidate.proposal_id}
                    candidate={candidate}
                    busy={busy !== null}
                    onNavigate={setSelected}
                    onAccept={(id) =>
                      void run("accept", async () => afterAction(await api.accept(id)))
                    }
                    onEdit={(id, text) =>
                      void run("edit", async () => afterAction(await api.edit(id, text)))
                    }
                    onReject={(id, reason) =>
                      void run("reject", async () => {
                        await api.reject(id, reason);
                        setCandidates((current) => current.filter((c) => c.proposal_id !== id));
                        await refresh();
                      })
                    }
                  />
                ))}
              </>
            )}
          </>
        )}
      </section>

      <section className="pane">
        <WorldState
          slice={slice}
          threadAges={threadAges}
          authorship={authorship}
          onNavigate={setSelected}
          onStrike={(fact: FactView) =>
            void run("strike", async () => afterAction(await api.strikeFact(fact.fact.id)))
          }
        />

        {ledger && (
          <>
            <h2>Your dial</h2>
            <div className="dial">
              <label>
                <span>coherent</span>
                <span>surprising</span>
              </label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={ledger.dial}
                onChange={(event) =>
                  void run("dial", async () => {
                    await api.setDial(Number(event.target.value));
                    await refresh();
                  })
                }
              />
              <p className="hint">
                At 0 the options are ranked on quality alone. At 1 they are ranked on how far
                they are from the obvious next move.
              </p>
            </div>

            <h2>Your rules</h2>
            {ledger.style_notes.length === 0 ? (
              <p className="empty">None yet.</p>
            ) : (
              <ul className="facts">
                {ledger.style_notes.map((note) => (
                  <li key={note.text}>
                    <div className="row-between">
                      <span>{note.text}</span>
                      <button
                        className="quiet"
                        title="Delete this rule. It stops reaching prompts immediately."
                        onClick={() =>
                          void run("note", async () => {
                            await api.removeStyleNote(note.text);
                            await refresh();
                          })
                        }
                      >
                        remove
                      </button>
                    </div>
                    <div className="meta">
                      {note.source === "mined"
                        ? `learned from your edits, seen ${note.count}×${
                            note.count < 2 ? " — not used yet" : ""
                          }`
                        : "yours"}
                    </div>
                  </li>
                ))}
              </ul>
            )}
            <div className="inline">
              <button
                onClick={() =>
                  void run("note", async () => {
                    const text = window.prompt("A rule for how prose should read");
                    if (text) await api.addStyleNote(text);
                    await refresh();
                  })
                }
              >
                add a rule
              </button>
              <button
                onClick={() =>
                  void run("mine", async () => {
                    await api.mineEdits();
                    await refresh();
                  })
                }
                title="Read your edits and turn them into rules you can see and delete"
              >
                learn from my edits
              </button>
            </div>

            <h2>Your criteria</h2>
            {ledger.criteria.length === 0 ? (
              <p className="empty">None yet. Adding one changes what the options are scored on.</p>
            ) : (
              <ul className="facts">
                {ledger.criteria.map((criterion) => (
                  <li key={criterion.name}>
                    <div className="row-between">
                      <span>{criterion.name}</span>
                      <button
                        className="quiet"
                        title="Stop scoring options on this."
                        onClick={() =>
                          void run("criterion", async () => {
                            await api.removeCriterion(criterion.name);
                            await refresh();
                          })
                        }
                      >
                        remove
                      </button>
                    </div>
                    <div className="meta">{criterion.description}</div>
                  </li>
                ))}
              </ul>
            )}
            <button
              onClick={() =>
                void run("criterion", async () => {
                  const name = window.prompt("Name the thing you care about (e.g. menace)");
                  if (!name) return;
                  const description = window.prompt("Describe it in your own words") ?? "";
                  await api.addCriterion(name, description);
                  await refresh();
                })
              }
            >
              add a criterion
            </button>

            {Object.keys(ledger.learned).length > 0 && (
              <>
                <h2>What it has learned</h2>
                <p className="hint">
                  {String(ledger.learned.pairs_seen ?? 0)} comparisons,{" "}
                  {String(ledger.learned.edits_seen ?? 0)} edits.
                  {Array.isArray(ledger.learned.top_weights) &&
                    ledger.learned.top_weights.length > 0 &&
                    ` Currently ranking most on ${(
                      ledger.learned.top_weights as Array<{ feature: string }>
                    )
                      .slice(0, 2)
                      .map((w) => w.feature.replace(/_/g, " "))
                      .join(" and ")}.`}
                </p>
              </>
            )}
          </>
        )}
      </section>

      {result && (
        <BibleDiffModal result={result} onClose={() => setResult(null)} onNavigate={setSelected} />
      )}
    </div>
  );
}
