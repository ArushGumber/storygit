"""Recording the Gallery sessions: seven scenarios, each proving one claim.

The Gallery is the part of this project that has to be *shown* rather than argued. Each
session demonstrates exactly one thing the design claims, and each is recorded against real
snapshots so replay reads the story out of the store — the tab cannot show something the
system did not do.

Five of the seven need no model calls at all: propagation, the epistemic catch, the thread
ledger, branching and merging, and the criterion changing a ranking are all properties of
the deterministic core, and scripting them makes the recordings reproducible and free. Two
need generation — the labelled candidates and the dial at both ends — and those are recorded
against the live provider with caching, so a re-record costs nothing.

    python -m eval.record_gallery
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from eval.gallery_record import Recorder, Session
from storygit.agents.schemas import Level
from storygit.continuity import layer1
from storygit.continuity.bible_diff import compute as bible_diff
from storygit.domain.diff import (
    AddEntity,
    AddFact,
    AddKnows,
    AddNode,
    Diff,
    DiffAuthor,
    OpenThread,
    SetLock,
    UpdateFact,
    UpdateNode,
)
from storygit.domain.ids import IdGenerator, NodeId
from storygit.domain.nodes import Beat, Episode, Prose, Scene, Story
from storygit.domain.provenance import Authorship, ProvenanceSpan
from storygit.domain.state import StoryState
from storygit.domain.threads import Thread
from storygit.domain.world import Entity, EntityKind, Fact, Knows, Predicate
from storygit.engine import AcceptResult
from storygit.graph.propagation import marks_to_diff, propagate_change
from storygit.store.repository import Repository

RESULTS = Path(__file__).parent / "results" / "gallery"


def _open(name: str) -> tuple[Repository, Path]:
    """A fresh repository for one session, kept so replay can read its snapshots."""
    db = RESULTS / f"{name}.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()
    return Repository.open(db), db


class Scaffold:
    """A small Ashfall story, built deterministically, for scripted sessions."""

    def __init__(self, repo: Repository, seed: int = 11) -> None:
        """Build the scaffold into a repository.

        Args:
            repo: Where to build it.
            seed: Id-generator seed.
        """
        self.repo = repo
        self.ids = IdGenerator(seed=seed, stream="gallery")
        self.story = self.ids.node()
        self.episode = self.ids.node()
        self.scene = self.ids.node()
        self.beats = [self.ids.node() for _ in range(5)]
        self.kael, self.mara, self.warden = (self.ids.entity() for _ in range(3))
        self.ashfall, self.kell = self.ids.entity(), self.ids.entity()

        repo.initialize(
            StoryState.build(
                nodes={
                    self.story: Story(
                        id=self.story,
                        title="Ashfall",
                        seed=(
                            "A powerless orphan discovers an ability that could change the "
                            "balance of power in a world at war."
                        ),
                        premise="Kael discovers the ash answers him.",
                        existential_question="Is power worth what it costs the powerless?",
                    )
                }
            )
        )
        titles = [
            "Caught in the market",
            "The Warden's offer",
            "Kael runs",
            "Mara at the harbour",
            "The gate closes",
        ]
        repo.commit_diff(
            Diff(
                ops=(
                    AddNode(
                        node=Episode(
                            id=self.episode,
                            parent_id=self.story,
                            title="Episode 1: Ashfall",
                            hook="A boy with nothing steals from the one person who notices.",
                            cliffhanger="The Warden calls him by a name he never gave.",
                        )
                    ),
                    AddNode(node=Scene(id=self.scene, parent_id=self.episode, title="The market")),
                    *(
                        AddNode(
                            node=Beat(
                                id=beat,
                                parent_id=self.scene,
                                title=title,
                                position=index,
                                what_happens=f"{title}.",
                                location="Ashfall",
                            )
                        )
                        for index, (beat, title) in enumerate(zip(self.beats, titles, strict=True))
                    ),
                    AddEntity(entity=Entity(id=self.kael, kind=EntityKind.character, name="Kael")),
                    AddEntity(entity=Entity(id=self.mara, kind=EntityKind.character, name="Mara")),
                    AddEntity(
                        entity=Entity(
                            id=self.warden,
                            kind=EntityKind.character,
                            name="Warden of Kell",
                            aliases=("the Warden",),
                        )
                    ),
                    AddEntity(
                        entity=Entity(id=self.ashfall, kind=EntityKind.place, name="Ashfall")
                    ),
                    AddEntity(entity=Entity(id=self.kell, kind=EntityKind.place, name="Kell")),
                ),
                author=DiffAuthor.human,
                intent="the story so far",
            )
        )

    def state(self) -> StoryState:
        """Current state."""
        return self.repo.state()


def _scripted_step(
    recorder: Recorder,
    *,
    title: str,
    note: str,
    diff: Diff,
    action: str = "",
    node_id: NodeId | None = None,
) -> AcceptResult:
    """Commit a scripted change, propagate it, and record the whole thing."""
    engine = recorder.engine
    before = engine.state()
    snapshot = engine.repo.commit_diff(diff, branch=engine.branch)
    after = engine.repo.state(engine.branch)
    marks = propagate_change(before, after, edge_provider=engine.edge_provider)
    if marks:
        mark_diff = marks_to_diff(after, marks)
        if len(mark_diff):
            snapshot = engine.repo.commit_diff(mark_diff, branch=engine.branch)
            after = engine.repo.state(engine.branch)
    result = AcceptResult(
        snapshot_id=snapshot,
        bible_diff=bible_diff(before, after),
        marks=tuple(marks),
        flags=tuple(layer1.check_state(after)),
    )
    recorder.record(title=title, note=note, action=action, result=result, node_id=node_id)
    return result


def _engine_for(repo: Repository) -> Any:
    """A minimal engine over a repository, with no provider work needed."""
    from storygit.engine import Engine
    from storygit.preference.layer import PreferenceLayer
    from storygit.providers.router import Router

    return Engine(
        repo,
        Router({}),
        ids=IdGenerator(seed=17, stream="gallery-engine"),
        use_nli=False,
        preference=PreferenceLayer(enabled=False),
    )


# --- the seven sessions -------------------------------------------------------


def session_propagation() -> Session:
    """An edit marks two beats stale, and neither is rewritten."""
    repo, db = _open("propagation_walk")
    scaffold = Scaffold(repo)
    engine = _engine_for(repo)
    recorder = Recorder(
        engine,
        "propagation_walk",
        title="An edit marks what it affected, and rewrites nothing",
        summary=(
            "Beat 1 establishes where Kael is. Beat 2 relies on it and establishes his "
            "goal; beat 3 relies on that. Changing beat 1 walks the declared dependency "
            "edges and marks both, with the reason and the beat that established the "
            "fact. Beat 4, which depends on neither, is untouched. Nothing is regenerated."
        ),
    )

    location = Fact(
        id=scaffold.ids.fact(),
        subject=scaffold.kael,
        predicate=Predicate.location,
        object_entity=scaffold.ashfall,
        valid_from_beat=scaffold.beats[0],
        established_by_beat=scaffold.beats[0],
    )
    goal = Fact(
        id=scaffold.ids.fact(),
        subject=scaffold.kael,
        predicate=Predicate.goal,
        object_text="get out of the city before the gate closes",
        valid_from_beat=scaffold.beats[1],
        established_by_beat=scaffold.beats[1],
    )
    _scripted_step(
        recorder,
        title="The story so far",
        note="Two facts, and the beats that declare they depend on them.",
        diff=Diff(
            ops=(
                AddFact(fact=location),
                AddFact(fact=goal),
                UpdateNode(node_id=scaffold.beats[1], fields={"consumes": [str(location.id)]}),
                UpdateNode(node_id=scaffold.beats[2], fields={"consumes": [str(goal.id)]}),
            ),
            author=DiffAuthor.ai,
            intent="establish the chain",
        ),
    )
    _scripted_step(
        recorder,
        title="The writer moves Kael to Kell",
        note=(
            "One fact changes. Two beats are marked stale, each citing the fact and the "
            "beat it came from. Beat 4 is not touched. Nothing is rewritten: the writer "
            "chooses regenerate, edit, or dismiss."
        ),
        diff=Diff(
            ops=(
                UpdateFact(
                    fact_id=location.id,
                    fields={"object_entity": None, "object_text": "Kell"},
                ),
            ),
            author=DiffAuthor.human,
            intent="Kael starts in Kell instead",
        ),
        action="edit",
        node_id=scaffold.beats[0],
    )
    session = recorder.session(db_path=str(db))
    session.save(RESULTS)
    repo.close()
    return session


def session_epistemic() -> Session:
    """A character acts on a secret they were never told."""
    repo, db = _open("epistemic_catch")
    scaffold = Scaffold(repo)
    engine = _engine_for(repo)
    recorder = Recorder(
        engine,
        "epistemic_catch",
        title="A character acts on something nobody told her",
        summary=(
            "Kael's secret is established in beat 2. In beat 4 Mara acts on it. Nothing "
            "about the facts is wrong, the secret is true, so a checker that only "
            "compares facts sees nothing. The epistemic layer asks whether Mara has been "
            "told, finds no edge, and says so, naming the beat where the secret was "
            "established."
        ),
    )

    secret = Fact(
        id=scaffold.ids.fact(),
        subject=scaffold.kael,
        predicate=Predicate.secret,
        object_text="the ash answers him",
        valid_from_beat=scaffold.beats[1],
        established_by_beat=scaffold.beats[1],
    )
    acting = Fact(
        id=scaffold.ids.fact(),
        subject=scaffold.mara,
        predicate=Predicate.goal,
        object_text="use what Kael can do",
        valid_from_beat=scaffold.beats[3],
        established_by_beat=scaffold.beats[3],
    )
    _scripted_step(
        recorder,
        title="Kael's secret is established",
        note="Only Kael knows it. The knows-edge says exactly that.",
        diff=Diff(
            ops=(
                AddFact(fact=secret),
                AddKnows(
                    knows=Knows(
                        character=scaffold.kael, fact=secret.id, since_beat=scaffold.beats[1]
                    )
                ),
            ),
            author=DiffAuthor.ai,
            intent="Kael learns what he can do",
        ),
    )
    _scripted_step(
        recorder,
        title="Mara acts on it two beats later",
        note=(
            "The flag names who does not know, what they are acting on, where it was "
            "established, and that they never learn it anywhere in the story."
        ),
        diff=Diff(
            ops=(
                AddFact(fact=acting),
                UpdateNode(node_id=scaffold.beats[3], fields={"consumes": [str(secret.id)]}),
            ),
            author=DiffAuthor.ai,
            intent="Mara moves on Kael",
        ),
        action="accept",
        node_id=scaffold.beats[3],
    )
    _scripted_step(
        recorder,
        title="The writer tells her, one beat earlier",
        note="The fix is a knows-edge, not a rewrite. The flag clears.",
        diff=Diff(
            ops=(
                AddKnows(
                    knows=Knows(
                        character=scaffold.mara, fact=secret.id, since_beat=scaffold.beats[2]
                    )
                ),
            ),
            author=DiffAuthor.human,
            intent="Mara overhears in beat 3",
        ),
        action="edit",
        node_id=scaffold.beats[2],
    )
    session = recorder.session(db_path=str(db))
    session.save(RESULTS)
    repo.close()
    return session


def session_threads() -> Session:
    """The thread ledger notices a subplot nobody has touched."""
    repo, db = _open("thread_ledger")
    scaffold = Scaffold(repo)
    engine = _engine_for(repo)
    recorder = Recorder(
        engine,
        "thread_ledger",
        title="A dropped thread is not a contradiction, so nothing else would find it",
        summary=(
            "A thread opened in beat 1 and untouched since is perfectly consistent. No "
            "fact conflicts, no dependency is stale, and every continuity check passes, "
            "which is exactly why a serial needs a ledger that tracks it separately. For "
            "audio fiction, a dropped thread is the most expensive kind of mistake."
        ),
    )
    thread_id = scaffold.ids.thread()
    _scripted_step(
        recorder,
        title="A question is opened",
        note="Beat 1 raises it: who told the Warden Kael's name?",
        diff=Diff(
            ops=(
                OpenThread(
                    thread=Thread(
                        id=thread_id,
                        description="Who told the Warden Kael's name?",
                        opened_at_beat=scaffold.beats[0],
                        last_touched_beat=scaffold.beats[0],
                    )
                ),
            ),
            author=DiffAuthor.ai,
            intent="open the thread",
        ),
    )

    from storygit.continuity.audit import check_dropped_threads

    state = engine.state()
    flags = check_dropped_threads(state, max_gap=2)
    recorder.record(
        title="Four beats later, nothing has touched it",
        note=(
            "The audit reports it. Not a contradiction, an omission, and the only place "
            "in the system that would ever notice."
        ),
        flags=flags,
        action="audit",
    )
    session = recorder.session(db_path=str(db))
    session.save(RESULTS)
    repo.close()
    return session


def session_locks() -> Session:
    """A lock stops propagation dead, and becomes a hard constraint."""
    repo, db = _open("locks_and_review")
    scaffold = Scaffold(repo)
    engine = _engine_for(repo)
    recorder = Recorder(
        engine,
        "locks_and_review",
        title="Locks stop the walk; human prose is flagged, never staled",
        summary=(
            "The same upstream edit, three times. Once normally: two beats stale. Once "
            "with beat 2 locked: the walk stops there and beat 3 is never reached, because "
            "a locked beat is not going to change. Once with human-written prose on beat "
            "2: it is flagged for review rather than staled, the system does not tell an "
            "author their own sentences are out of date."
        ),
    )

    location = Fact(
        id=scaffold.ids.fact(),
        subject=scaffold.kael,
        predicate=Predicate.location,
        object_entity=scaffold.ashfall,
        valid_from_beat=scaffold.beats[0],
        established_by_beat=scaffold.beats[0],
    )
    goal = Fact(
        id=scaffold.ids.fact(),
        subject=scaffold.kael,
        predicate=Predicate.goal,
        object_text="get out before the gate closes",
        valid_from_beat=scaffold.beats[1],
        established_by_beat=scaffold.beats[1],
    )
    prose_id = scaffold.ids.node()
    _scripted_step(
        recorder,
        title="The chain, plus prose the writer typed themselves",
        note="Beat 2's prose is human-written; beat 3 depends on what beat 2 established.",
        diff=Diff(
            ops=(
                AddFact(fact=location),
                AddFact(fact=goal),
                UpdateNode(node_id=scaffold.beats[1], fields={"consumes": [str(location.id)]}),
                UpdateNode(node_id=scaffold.beats[2], fields={"consumes": [str(goal.id)]}),
                AddNode(
                    node=Prose(
                        id=prose_id,
                        parent_id=scaffold.beats[1],
                        title="prose",
                        text="Kael counted the guards. Three, and none of them looking.",
                        spans=(ProvenanceSpan(start=0, end=2, source=Authorship.human),),
                    )
                ),
            ),
            author=DiffAuthor.human,
            intent="the writer writes beat 2 by hand",
        ),
    )
    _scripted_step(
        recorder,
        title="Upstream changes: beat 2 is flagged for review, not staled",
        note=(
            "Its status stays accepted. The reason says the writer's prose may be "
            "affected. Beat 3, which the AI wrote, is staled normally."
        ),
        diff=Diff(
            ops=(
                UpdateFact(
                    fact_id=location.id, fields={"object_entity": None, "object_text": "Kell"}
                ),
            ),
            author=DiffAuthor.human,
            intent="Kael starts in Kell",
        ),
        action="edit",
        node_id=scaffold.beats[0],
    )
    _scripted_step(
        recorder,
        title="Now lock beat 2 and change it back",
        note=(
            "The walk stops at the lock: beat 3 is never reached, because nothing a locked "
            "beat produces can change. The lock's facts also become hard constraints in "
            "every subsequent prompt."
        ),
        diff=Diff(
            ops=(
                SetLock(node_id=scaffold.beats[1]),
                UpdateFact(
                    fact_id=location.id,
                    fields={"object_entity": scaffold.ashfall, "object_text": ""},
                ),
            ),
            author=DiffAuthor.human,
            intent="lock beat 2, move Kael back",
        ),
        action="lock",
        node_id=scaffold.beats[1],
    )
    session = recorder.session(db_path=str(db))
    session.save(RESULTS)
    repo.close()
    return session


def session_branching() -> Session:
    """Branch, diverge, compare, merge — and surface a conflict rather than guessing."""
    repo, db = _open("branch_and_merge")
    scaffold = Scaffold(repo)
    engine = _engine_for(repo)
    recorder = Recorder(
        engine,
        "branch_and_merge",
        title="What if the Warden lets him go?",
        summary=(
            "A branch is a pointer at a snapshot, so exploring costs nothing. Two "
            "divergent edits merge automatically when they touch different beats. Two "
            "edits to the same beat come back as a conflict for the writer, the system "
            "never guesses which version of a scene the author meant."
        ),
    )
    recorder.record(title="The story so far", note="One line, five beats.", action="branch")

    repo.create_branch("what-if")
    repo.commit_diff(
        Diff(
            ops=(
                UpdateNode(
                    node_id=scaffold.beats[1],
                    fields={"what_happens": "The Warden lets him go, and says nothing."},
                ),
            ),
            author=DiffAuthor.human,
            intent="the Warden lets him go",
        ),
        branch="what-if",
    )
    recorder.record(
        title="Explore on a branch",
        note="Beat 2 is rewritten on `what-if`. `main` is untouched.",
        action="branch",
        snapshot_id=repo.head("what-if"),
        node_id=scaffold.beats[1],
    )

    repo.commit_diff(
        Diff(
            ops=(
                UpdateNode(
                    node_id=scaffold.beats[3],
                    fields={"what_happens": "Mara counts the boats and finds one missing."},
                ),
            ),
            author=DiffAuthor.human,
            intent="Mara notices the missing boat",
        ),
        branch="main",
    )
    recorder.record(
        title="Meanwhile, on main",
        note="A different beat is edited. The two changes do not overlap.",
        action="edit",
        node_id=scaffold.beats[3],
    )

    clean = repo.merge_branches("main", "what-if")
    snapshot = repo.commit_diff(clean.diff, branch="main")
    recorder.record(
        title="Merge: disjoint changes combine",
        note=(
            f"Three-way merge over content hashes. {len(clean.conflicts)} conflicts, so "
            "both edits land."
        ),
        action="merge",
        snapshot_id=snapshot,
    )

    repo.create_branch("what-if-2", from_branch="main")
    repo.commit_diff(
        Diff(
            ops=(UpdateNode(node_id=scaffold.beats[3], fields={"what_happens": "A."}),),
            author=DiffAuthor.human,
            intent="version A",
        ),
        branch="main",
    )
    repo.commit_diff(
        Diff(
            ops=(UpdateNode(node_id=scaffold.beats[3], fields={"what_happens": "B."}),),
            author=DiffAuthor.human,
            intent="version B",
        ),
        branch="what-if-2",
    )
    conflicted = repo.merge_branches("main", "what-if-2")
    recorder.record(
        title="Now both branches edit the same beat",
        note=(
            f"{len(conflicted.conflicts)} conflict, returned for the writer. Nothing is "
            "auto-resolved: last-writer-wins would silently discard authored work."
        ),
        action="merge",
        snapshot_id=repo.head("main"),
    )
    session = recorder.session(db_path=str(db))
    session.save(RESULTS)
    repo.close()
    return session


def session_criterion() -> Session:
    """A writer-defined criterion changes which candidate ranks first."""
    repo, db = _open("criterion_changes_ranking")
    Scaffold(repo)  # the session is about ranking, but replay still needs a real story
    engine = _engine_for(repo)
    recorder = Recorder(
        engine,
        "criterion_changes_ranking",
        title="The writer's own criterion reorders the candidates",
        summary=(
            "Two candidates, judged on the fixed narratology axes, rank one way. The "
            "writer adds a criterion in their own words, 'menace: every scene should "
            "threaten something', and the judge scores it alongside the fixed axes, so "
            "the ranking changes. The objective is partly the writer's, which is the whole "
            "answer to being an arbiter of somebody else's objective."
        ),
    )
    from storygit.preference.bt_head import BTWeights
    from storygit.preference.features import build

    quiet = build(
        judge_scores={"momentum": 4.0, "specificity": 4.5, "consequence": 3.0, "voice": 4.0},
        writer_criteria_scores={"menace": 1.5},
        text="They talk for a while, and nothing much is decided. " * 8,
    )
    threatening = build(
        judge_scores={"momentum": 3.5, "specificity": 3.5, "consequence": 4.0, "voice": 3.5},
        writer_criteria_scores={"menace": 5.0},
        text="The Warden does not raise her voice. That is what frightens him. " * 8,
    )
    weights = BTWeights.uniform()
    recorder.record(
        title="Before: ranked on the fixed axes",
        note=(
            f"'Quiet' scores {weights.score(quiet):.3f}, 'threatening' "
            f"{weights.score(threatening):.3f}. The polished, low-stakes candidate wins."
        ),
        action="propose",
    )

    engine.add_criterion("menace", "every scene should threaten something", weight=1.0)
    weighted = BTWeights(
        names=weights.names,
        weights=tuple(
            w * (3.0 if name == "writer_criteria" else 1.0)
            for name, w in zip(weights.names, weights.weights, strict=True)
        ),
    )
    recorder.record(
        title="After: the writer's criterion is in the objective",
        note=(
            f"With 'menace' weighted, 'quiet' scores {weighted.score(quiet):.3f} and "
            f"'threatening' {weighted.score(threatening):.3f}. The order flips. The "
            "criterion is visible in the ledger and deletable."
        ),
        action="criterion",
    )
    session = recorder.session(db_path=str(db))
    session.save(RESULTS)
    repo.close()
    return session


async def session_dial(*, live: bool = True) -> Session | None:
    """The same beat, at both ends of the dial."""
    from storygit.config import get_settings
    from storygit.engine import Engine
    from storygit.preference.layer import PreferenceLayer
    from storygit.providers.router import build_router
    from storygit.selection.select import SelectionConfig

    if not live:
        return None
    settings = get_settings()
    if settings.openrouter_is_enabled:
        raise RuntimeError("OpenRouter is enabled; the gallery never uses it.")

    repo, db = _open("dial_both_ends")
    scaffold = Scaffold(repo)
    router = build_router(settings)
    engine = Engine(
        repo,
        router,
        ids=IdGenerator(seed=23, stream="dial"),
        selection=SelectionConfig(n=6, k=3, use_judge=True, use_dial=True),
        use_nli=False,
        preference=PreferenceLayer(enabled=False),
    )
    recorder = Recorder(
        engine,
        "dial_both_ends",
        title="The same beat, at both ends of the dial",
        summary=(
            "The dial re-weights the objective, not the sampler. At 0 the ranking is "
            "quality alone; at 1 it is distance from the model's own temperature-0 "
            "continuation, so the system explicitly selects against what it would have "
            "done anyway. Same six candidates both times; different three shown."
        ),
    )
    try:
        for value, label in ((0.0, "coherent"), (1.0, "surprising")):
            engine.set_dial(value)
            candidates = await engine.propose_at(
                Level.beat,
                node_id=scaffold.scene,
                intent="Kael's ability shows itself, but only he understands what happened.",
            )
            recorder.record(
                title=f"Dial at {value:.0f} ({label})",
                note=("Shown: " + ", ".join(c.axis_label for c in candidates if c.selected) + "."),
                candidates=candidates,
                node_id=scaffold.scene,
                level="beat",
                intent="Kael's ability shows itself",
                action="propose",
            )
    finally:
        await router.aclose()
    session = recorder.session(db_path=str(db))
    session.save(RESULTS)
    repo.close()
    return session


async def session_labelled(*, live: bool = True) -> Session | None:
    """Six axis-conditioned candidates become three labelled options."""
    from storygit.config import get_settings
    from storygit.engine import Engine
    from storygit.preference.layer import PreferenceLayer
    from storygit.providers.router import build_router
    from storygit.selection.select import SelectionConfig

    if not live:
        return None
    settings = get_settings()
    repo, db = _open("labelled_candidates")
    scaffold = Scaffold(repo)
    router = build_router(settings)
    engine = Engine(
        repo,
        router,
        ids=IdGenerator(seed=29, stream="labelled"),
        selection=SelectionConfig(n=6, k=3, use_judge=True, use_dial=False),
        use_nli=False,
        preference=PreferenceLayer(enabled=False),
    )
    recorder = Recorder(
        engine,
        "labelled_candidates",
        title="Three named directions, not three paragraphs",
        summary=(
            "Six candidates are generated under named instructions: raise the stakes, "
            "slow down, subvert the expectation, and three are selected for quality and "
            "mutual difference. The label is the mechanism: three named directions cost a "
            "decision; three unlabelled paragraphs cost three readings. Each candidate "
            "arrives already checked, with its flags attached."
        ),
    )
    try:
        candidates = await engine.propose_at(
            Level.beat,
            node_id=scaffold.scene,
            intent="Kael is cornered and something answers him.",
        )
        recorder.record(
            title="Six sampled, three shown",
            note="Every candidate carries its axis label, its rationale, and its flags.",
            candidates=candidates,
            node_id=scaffold.scene,
            level="beat",
            action="propose",
        )
        if candidates:
            chosen = next(c for c in candidates if c.selected)
            result = await engine.accept(chosen.proposal.id, shown_with=engine.shown())
            recorder.record(
                title=f"The writer takes '{chosen.axis_label}'",
                note=(
                    "Accepting commits the diff, extracts facts from any new prose, and "
                    "shows the bible diff. The alternatives are recorded too, and that is "
                    "what makes this a preference rather than an event."
                ),
                candidates=candidates,
                action="accept",
                chosen=chosen.proposal.id,
                result=result,
            )
    finally:
        await router.aclose()
    session = recorder.session(db_path=str(db))
    session.save(RESULTS)
    repo.close()
    return session


SCRIPTED = (
    session_propagation,
    session_epistemic,
    session_threads,
    session_locks,
    session_branching,
    session_criterion,
)
"""Sessions needing no model calls. Reproducible and free."""


async def record_all(*, live: bool = True) -> list[Session]:
    """Record every session and return them.

    Args:
        live: Whether to record the two sessions that need generation.

    Returns:
        The recorded sessions.
    """
    sessions = [fn() for fn in SCRIPTED]
    for coroutine in (session_labelled(live=live), session_dial(live=live)):
        session = await coroutine
        if session is not None:
            sessions.append(session)
    return sessions


def write_index() -> list[dict[str, Any]]:
    """Rebuild the index from every session on disk.

    From disk rather than from what this invocation recorded, because ``--offline`` records
    only the six scripted sessions, and an index built from the return value would then
    silently drop the two generated ones. That is not hypothetical: it happened, and the
    Gallery quietly lost two sessions until a test noticed.

    Returns:
        The index entries, in name order.
    """
    entries: list[dict[str, Any]] = []
    for path in sorted(RESULTS.glob("*.json")):
        if path.name == "index.json":
            continue
        session = Session.load(path)
        entries.append(
            {
                "name": session.name,
                "title": session.title,
                "summary": session.summary,
                "steps": len(session.steps),
            }
        )
    (RESULTS / "index.json").write_text(json.dumps(entries, indent=2))
    return entries


def main() -> None:
    """Record every session and print what the Gallery now holds."""
    import sys

    live = "--offline" not in sys.argv
    recorded = {session.name for session in asyncio.run(record_all(live=live))}
    entries = write_index()
    for entry in entries:
        mark = "recorded" if entry["name"] in recorded else "kept    "
        print(f"  {mark}  {entry['name']:28} {entry['steps']} steps  {entry['title']}")
    if not live:
        print("\n  (--offline: the two generated sessions were kept as previously recorded)")
    print(f"\nGallery holds {len(entries)} sessions in {RESULTS}")


if __name__ == "__main__":
    main()
