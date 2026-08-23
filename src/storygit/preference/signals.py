"""Reading the signal store as preference data.

Chunk 2 wrote every writer action to the ``signals`` table. This module turns those rows
into the three shapes the learners actually consume:

* **Paired preferences** — "accepted A while B and C were on screen". This is the only
  honest way to read an accept. "The writer liked A" is weak; "the writer preferred A to B
  and C, having seen all three" is a comparison, and comparisons are what a Bradley-Terry
  model is defined on. Pairs are only formed *within one shown set*: comparing a candidate
  accepted on Monday against one rejected in a different scene on Friday would be
  comparing two different questions.
* **Edit pairs** — before and after. The most informative signal the system gets, because
  it says not just that the writer wanted something different but *what* different looks
  like.
* **Exemplar pools** — accepted and human-written prose as positives, rejected proposals
  as negatives, for retrieval into prompts and as training data for the voice model.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from storygit.domain.ids import NodeId, ProposalId
from storygit.store.signals import Signal, SignalKind, SignalStore


class PreferencePair(BaseModel):
    """One "chose A over B" comparison, drawn from a single shown set.

    Attributes:
        winner: The accepted (or edited-then-accepted) proposal.
        loser: A proposal shown alongside it that was not chosen.
        node_id: Where the choice was made.
        level: Which level of the plan tree it was at.
        was_edited: Whether the winner was edited before acceptance. An edited winner is
            a weaker endorsement than a clean accept, and chunk 4's head can weight it.
    """

    model_config = ConfigDict(frozen=True)

    winner: ProposalId
    loser: ProposalId
    node_id: NodeId | None = None
    level: str = ""
    was_edited: bool = False


class EditPair(BaseModel):
    """What the model wrote, and what the writer changed it to.

    Attributes:
        proposal_id: The proposal that was edited.
        before: The model's text.
        after: The writer's text.
        node_id: Where it happened.
    """

    model_config = ConfigDict(frozen=True)

    proposal_id: ProposalId
    before: str
    after: str
    node_id: NodeId | None = None

    @property
    def shortened(self) -> bool:
        """Whether the writer cut length. A crude but useful direction indicator."""
        return len(self.after.split()) < len(self.before.split())


class SignalReader:
    """Turns raw signals into the shapes the learners consume."""

    def __init__(self, store: SignalStore, *, branch: str | None = None) -> None:
        """Create a reader.

        Args:
            store: The signal store to read.
            branch: Restrict to one branch. Learned state is per branch, because a
                what-if branch may explore a direction the writer would reject on main.
        """
        self._store = store
        self._branch = branch

    def all(self) -> list[Signal]:
        """Every signal, oldest first."""
        return self._store.all(branch=self._branch)

    def preference_pairs(self) -> list[PreferencePair]:
        """Accepted-over-shown comparisons, within each shown set.

        Returns:
            One pair per (accepted, alternative) combination, in chronological order.
        """
        pairs: list[PreferencePair] = []
        edited = {
            s.proposal_id
            for s in self._store.all(kind=SignalKind.edit, branch=self._branch)
            if s.proposal_id
        }
        for signal in self._store.all(kind=SignalKind.accept, branch=self._branch):
            if signal.proposal_id is None:
                continue
            for alternative in signal.shown_with:
                if alternative == signal.proposal_id:
                    continue
                pairs.append(
                    PreferencePair(
                        winner=signal.proposal_id,
                        loser=alternative,
                        node_id=signal.node_id,
                        level=str(signal.payload.get("level", "")),
                        was_edited=signal.proposal_id in edited,
                    )
                )
        return pairs

    def edit_pairs(self) -> list[EditPair]:
        """Before/after text pairs from every edit.

        Returns:
            Pairs with non-empty before and after, in chronological order.
        """
        out: list[EditPair] = []
        for signal in self._store.all(kind=SignalKind.edit, branch=self._branch):
            before = str(signal.payload.get("before", ""))
            after = str(signal.payload.get("after", ""))
            if not before.strip() or not after.strip() or signal.proposal_id is None:
                continue
            out.append(
                EditPair(
                    proposal_id=signal.proposal_id,
                    before=before,
                    after=after,
                    node_id=signal.node_id,
                )
            )
        return out

    def accepted_ids(self) -> list[ProposalId]:
        """Proposals the writer accepted, oldest first."""
        return [
            s.proposal_id
            for s in self._store.all(kind=SignalKind.accept, branch=self._branch)
            if s.proposal_id
        ]

    def rejected_ids(self) -> list[ProposalId]:
        """Proposals the writer explicitly rejected, oldest first."""
        return [
            s.proposal_id
            for s in self._store.all(kind=SignalKind.reject, branch=self._branch)
            if s.proposal_id
        ]

    def unchosen_ids(self) -> list[ProposalId]:
        """Proposals shown alongside an accepted one but not chosen.

        These are weaker negatives than an explicit rejection — the writer may simply
        have liked another one more — but there are far more of them, and they are the
        only negatives that exist for a writer who never clicks reject.
        """
        chosen = set(self.accepted_ids())
        out: list[ProposalId] = []
        for signal in self._store.all(kind=SignalKind.accept, branch=self._branch):
            out.extend(p for p in signal.shown_with if p not in chosen)
        return out

    def counts(self) -> dict[str, int]:
        """How much data each learner currently has."""
        return {
            "signals": len(self.all()),
            "preference_pairs": len(self.preference_pairs()),
            "edit_pairs": len(self.edit_pairs()),
            "accepted": len(self.accepted_ids()),
            "rejected": len(self.rejected_ids()),
            "unchosen": len(self.unchosen_ids()),
        }
