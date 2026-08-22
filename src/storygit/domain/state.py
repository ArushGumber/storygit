"""``StoryState``: the whole story as one immutable in-memory aggregate.

Everything downstream — propagation, slicing, the continuity checker, prompt
construction — reads this object and nothing else. It is rebuilt, never mutated:
:func:`storygit.domain.apply.apply` returns a new aggregate, which is what lets the
store hash it and share unchanged pieces between snapshots.

The derived indices are the point. Layer 1 of the continuity checker has to answer
"what else is true about this subject and predicate right now" in constant time,
and propagation has to answer "which beats consume this fact"; both are dict
lookups here, never scans and never model calls.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace

from storygit.domain.ids import EntityId, FactId, NodeId, ThreadId
from storygit.domain.ledger import WriterLedger
from storygit.domain.nodes import BaseNode, Beat, NodeStatus, NodeType, Prose
from storygit.domain.threads import Thread
from storygit.domain.world import Entity, Fact, Knows, Predicate

KnowsKey = tuple[EntityId, FactId]


def normalize_name(name: str) -> str:
    """Fold a name for alias matching.

    Args:
        name: A name or alias as it appeared in text.

    Returns:
        Lowercased, whitespace-collapsed, article-stripped form.
    """
    folded = " ".join(name.strip().lower().split())
    for article in ("the ", "a ", "an "):
        if folded.startswith(article):
            folded = folded[len(article) :]
            break
    return folded


@dataclass(frozen=True)
class StoryState:
    """An immutable snapshot of the whole story.

    Attributes:
        nodes: Plan tree nodes by id.
        entities: World entities by id.
        facts: Facts by id.
        knows: Epistemic edges by ``(character, fact)``.
        threads: Open-thread ledger by id.
        ledger: The writer's standing instructions.
    """

    nodes: dict[NodeId, BaseNode] = field(default_factory=dict)
    entities: dict[EntityId, Entity] = field(default_factory=dict)
    facts: dict[FactId, Fact] = field(default_factory=dict)
    knows: dict[KnowsKey, Knows] = field(default_factory=dict)
    threads: dict[ThreadId, Thread] = field(default_factory=dict)
    ledger: WriterLedger = field(default_factory=WriterLedger)

    # --- derived indices, rebuilt on construction -------------------------------
    root_id: NodeId | None = field(default=None, compare=False)
    children: dict[NodeId, tuple[NodeId, ...]] = field(
        default_factory=dict, compare=False, repr=False
    )
    seq: dict[NodeId, int] = field(default_factory=dict, compare=False, repr=False)
    facts_by_key: dict[tuple[EntityId, Predicate], tuple[FactId, ...]] = field(
        default_factory=dict, compare=False, repr=False
    )
    producer_of: dict[FactId, NodeId] = field(default_factory=dict, compare=False, repr=False)
    consumers_of: dict[FactId, tuple[NodeId, ...]] = field(
        default_factory=dict, compare=False, repr=False
    )
    knows_by_character: dict[EntityId, tuple[FactId, ...]] = field(
        default_factory=dict, compare=False, repr=False
    )
    alias_index: dict[str, EntityId] = field(default_factory=dict, compare=False, repr=False)
    prose_of_beat: dict[NodeId, NodeId] = field(default_factory=dict, compare=False, repr=False)

    # --- construction -----------------------------------------------------------

    @staticmethod
    def build(
        *,
        nodes: dict[NodeId, BaseNode] | None = None,
        entities: dict[EntityId, Entity] | None = None,
        facts: dict[FactId, Fact] | None = None,
        knows: dict[KnowsKey, Knows] | None = None,
        threads: dict[ThreadId, Thread] | None = None,
        ledger: WriterLedger | None = None,
    ) -> StoryState:
        """Construct a state and compute every derived index.

        This is the only supported constructor; calling ``StoryState(...)`` directly
        would leave the indices empty.

        Args:
            nodes: Plan tree nodes by id.
            entities: Entities by id.
            facts: Facts by id.
            knows: Epistemic edges by ``(character, fact)``.
            threads: Threads by id.
            ledger: Writer ledger.

        Returns:
            A fully indexed state.
        """
        state = StoryState(
            nodes=dict(nodes or {}),
            entities=dict(entities or {}),
            facts=dict(facts or {}),
            knows=dict(knows or {}),
            threads=dict(threads or {}),
            ledger=ledger or WriterLedger(),
        )
        return state._reindex()

    def evolve(self, **changes: object) -> StoryState:
        """Return a copy with the given core collections replaced and reindexed.

        Args:
            **changes: Any of ``nodes``, ``entities``, ``facts``, ``knows``,
                ``threads``, ``ledger``.

        Returns:
            A new, fully indexed state.
        """
        return replace(self, **changes)._reindex()  # type: ignore[arg-type]

    def _reindex(self) -> StoryState:
        children: dict[NodeId, list[NodeId]] = {}
        root_id: NodeId | None = None
        for node in self.nodes.values():
            if node.parent_id is None:
                root_id = node.id
            else:
                children.setdefault(node.parent_id, []).append(node.id)
        ordered: dict[NodeId, tuple[NodeId, ...]] = {}
        for parent, kids in children.items():
            kids.sort(key=lambda nid: (self.nodes[nid].position, nid))
            ordered[parent] = tuple(kids)

        seq: dict[NodeId, int] = {}
        if root_id is not None:
            counter = 0
            stack: list[NodeId] = [root_id]
            while stack:
                current = stack.pop()
                seq[current] = counter
                counter += 1
                stack.extend(reversed(ordered.get(current, ())))

        facts_by_key: dict[tuple[EntityId, Predicate], list[FactId]] = {}
        for fact in self.facts.values():
            facts_by_key.setdefault(fact.key, []).append(fact.id)
        for ids in facts_by_key.values():
            ids.sort(key=lambda fid: (seq.get(self.facts[fid].valid_from_beat, 0), fid))

        producer_of: dict[FactId, NodeId] = {}
        consumers: dict[FactId, list[NodeId]] = {}
        prose_of_beat: dict[NodeId, NodeId] = {}
        for node in self.nodes.values():
            if isinstance(node, Beat):
                for fid in node.produces:
                    producer_of[fid] = node.id
                for fid in node.consumes:
                    consumers.setdefault(fid, []).append(node.id)
            elif isinstance(node, Prose) and node.parent_id is not None:
                prose_of_beat[node.parent_id] = node.id
        consumers_of = {
            fid: tuple(sorted(nids, key=lambda n: (seq.get(n, 0), n)))
            for fid, nids in consumers.items()
        }

        knows_by_character: dict[EntityId, list[FactId]] = {}
        for edge in self.knows.values():
            knows_by_character.setdefault(edge.character, []).append(edge.fact)

        alias_index: dict[str, EntityId] = {}
        for entity in self.entities.values():
            for name in entity.names:
                alias_index[normalize_name(name)] = entity.id

        return replace(
            self,
            root_id=root_id,
            children=ordered,
            seq=seq,
            facts_by_key={k: tuple(v) for k, v in facts_by_key.items()},
            producer_of=producer_of,
            consumers_of=consumers_of,
            knows_by_character={k: tuple(sorted(v)) for k, v in knows_by_character.items()},
            alias_index=alias_index,
            prose_of_beat=prose_of_beat,
        )

    # --- tree queries -----------------------------------------------------------

    def children_of(self, node_id: NodeId) -> tuple[BaseNode, ...]:
        """Direct children of a node, in reading order."""
        return tuple(self.nodes[nid] for nid in self.children.get(node_id, ()))

    def ancestors_of(self, node_id: NodeId) -> tuple[BaseNode, ...]:
        """The node's ancestors, root first."""
        chain: list[BaseNode] = []
        current = self.nodes[node_id].parent_id
        while current is not None:
            chain.append(self.nodes[current])
            current = self.nodes[current].parent_id
        return tuple(reversed(chain))

    def descendants_of(self, node_id: NodeId) -> Iterator[BaseNode]:
        """Every node beneath this one, in reading order."""
        for child_id in self.children.get(node_id, ()):
            yield self.nodes[child_id]
            yield from self.descendants_of(child_id)

    def nodes_of_type(self, node_type: NodeType) -> tuple[BaseNode, ...]:
        """All nodes of one level, in reading order."""
        return tuple(
            node
            for node in sorted(self.nodes.values(), key=lambda n: self.seq.get(n.id, 0))
            if node.node_type is node_type
        )

    def beats_in_order(self) -> tuple[Beat, ...]:
        """Every beat in reading order — the total order facts are dated against."""
        return tuple(n for n in self.nodes_of_type(NodeType.beat) if isinstance(n, Beat))

    def seq_of(self, node_id: NodeId) -> int:
        """Position of a node in the global depth-first reading order.

        Args:
            node_id: The node.

        Returns:
            Its sequence number.

        Raises:
            KeyError: If the node is not in the tree.
        """
        return self.seq[node_id]

    def is_locked(self, node_id: NodeId) -> bool:
        """Whether the writer has frozen this node."""
        return node_id in self.ledger.locks

    def prose_for(self, beat_id: NodeId) -> Prose | None:
        """The prose child of a beat, if it has one."""
        prose_id = self.prose_of_beat.get(beat_id)
        if prose_id is None:
            return None
        node = self.nodes[prose_id]
        return node if isinstance(node, Prose) else None

    # --- world queries ----------------------------------------------------------

    def valid_at(self, fact: Fact, beat_id: NodeId) -> bool:
        """Whether a fact holds at a given beat.

        Validity is the half-open interval ``[valid_from, valid_until)`` over the
        global beat sequence, so a fact that ends exactly where its replacement
        begins produces no overlap.

        Args:
            fact: The fact.
            beat_id: The beat to evaluate at.

        Returns:
            True when the fact holds at that beat.
        """
        if beat_id not in self.seq or fact.valid_from_beat not in self.seq:
            return False
        at = self.seq[beat_id]
        if at < self.seq[fact.valid_from_beat]:
            return False
        if fact.valid_until_beat is None:
            return True
        until = self.seq.get(fact.valid_until_beat)
        return until is None or at < until

    def facts_valid_at(
        self,
        beat_id: NodeId,
        *,
        subject: EntityId | None = None,
        predicate: Predicate | None = None,
    ) -> tuple[Fact, ...]:
        """Facts holding at a beat, optionally narrowed to a subject/predicate.

        Args:
            beat_id: Beat to evaluate at.
            subject: Restrict to this subject.
            predicate: Restrict to this predicate.

        Returns:
            Matching facts in establishment order.
        """
        if subject is not None and predicate is not None:
            candidates: Iterable[Fact] = (
                self.facts[fid] for fid in self.facts_by_key.get((subject, predicate), ())
            )
        else:
            candidates = self.facts.values()
        out = [
            fact
            for fact in candidates
            if self.valid_at(fact, beat_id)
            and (subject is None or fact.subject == subject)
            and (predicate is None or fact.predicate is predicate)
        ]
        out.sort(key=lambda f: (self.seq.get(f.valid_from_beat, 0), f.id))
        return tuple(out)

    def knows_at(self, character: EntityId, fact_id: FactId, beat_id: NodeId) -> bool:
        """Whether a character knows a fact by a given beat.

        Args:
            character: The character.
            fact_id: The fact.
            beat_id: The beat at which they would be acting on it.

        Returns:
            True when an epistemic edge exists dated at or before that beat.
        """
        edge = self.knows.get((character, fact_id))
        if edge is None:
            return False
        if edge.since_beat not in self.seq or beat_id not in self.seq:
            return False
        return self.seq[edge.since_beat] <= self.seq[beat_id]

    def entity_by_name(self, name: str) -> Entity | None:
        """Resolve a name or alias to an entity, or ``None`` if unknown."""
        entity_id = self.alias_index.get(normalize_name(name))
        return self.entities.get(entity_id) if entity_id is not None else None

    def entity_names(self) -> dict[EntityId, str]:
        """Display names for every entity, for rendering facts as sentences."""
        return {eid: entity.name for eid, entity in self.entities.items()}

    def stale_nodes(self) -> tuple[BaseNode, ...]:
        """Every node currently marked stale, in reading order."""
        return tuple(
            n
            for n in sorted(self.nodes.values(), key=lambda n: self.seq.get(n.id, 0))
            if n.status is NodeStatus.stale
        )
