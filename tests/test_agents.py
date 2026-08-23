"""Generation: schema output becomes typed diffs that apply cleanly."""

from __future__ import annotations

import json

import pytest
from tests.conftest import Fixture
from tests.mockprovider import MockProvider, canned

from storygit.agents.aliases import AliasResolver
from storygit.agents.parse import extract_json, parse_into
from storygit.agents.propose import Proposer
from storygit.agents.schemas import BeatProposal, Level
from storygit.domain.apply import apply
from storygit.domain.diff import AddEntity, OpenThread
from storygit.domain.ids import IdGenerator
from storygit.domain.nodes import Episode, NodeType
from storygit.domain.provenance import Authorship
from storygit.domain.world import EntityKind, Predicate
from storygit.providers.base import SchemaParseError
from storygit.providers.router import Router

BEAT_PAYLOAD = {
    "title": "The Warden's offer",
    "what_happens": "The Warden offers Kael passage out of Ashfall, for a price.",
    "audience_learns": "The Warden knows what Kael can do.",
    "audience_feels": "wary",
    "location": "Ashfall",
    "time": "nightfall",
    "produces": [
        {
            "subject": "Kael",
            "predicate": "secret",
            "object": "he can hear the ash",
            "object_is_entity": False,
            "known_by": ["Warden of Kell"],
        },
        {
            "subject": "Warden of Kell",
            "predicate": "goal",
            "object": "recruit Kael",
            "object_is_entity": False,
            "known_by": [],
        },
    ],
    "consumes": [],
    "threads_touched": [],
    "new_characters": [],
    "rationale": "Puts the two of them in a room and gives Kael something to refuse.",
    "delta_summary": ["the Warden makes Kael an offer"],
}


def router_with(
    responses: list[str] | None = None, **kwargs: object
) -> tuple[Router, MockProvider]:
    """A router whose every purpose is served by one mock provider."""
    provider = MockProvider(responses, **kwargs)  # type: ignore[arg-type]
    return Router({"gemini": provider, "groq": provider, "openrouter": provider}), provider


# --- parsing ------------------------------------------------------------------


def test_extract_json_survives_fences_and_commentary() -> None:
    assert extract_json('{"a": 1}') == '{"a": 1}'
    assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('Here you go:\n{"a": 1}\nHope that helps.') == '{"a": 1}'


def test_parse_into_raises_typed_error_with_the_raw_text() -> None:
    with pytest.raises(SchemaParseError) as excinfo:
        parse_into(BeatProposal, "not json at all")
    assert excinfo.value.raw == "not json at all"


async def test_one_bad_reply_is_repaired_and_two_is_an_error(fixture: Fixture) -> None:
    replies = ["this is not json", canned(BEAT_PAYLOAD)]
    router, provider = router_with(replies)
    proposer = Proposer(router, IdGenerator(seed=7))

    proposals = await proposer.propose(
        fixture.repo.state(), Level.beat, target_node_id=fixture.scene, k=1
    )
    assert len(proposals) == 1, "a malformed reply is repaired, not discarded"
    assert any(r.purpose.endswith(".repair") for r in provider.requests)

    router2, _ = router_with(["nope", "still nope"])
    proposer2 = Proposer(router2, IdGenerator(seed=7))
    assert await proposer2.propose(fixture.repo.state(), Level.beat, k=1) == []


# --- aliases ------------------------------------------------------------------


def test_alias_resolution_matches_known_names_and_flags_new_ones(fixture: Fixture) -> None:
    state = fixture.repo.state()
    resolver = AliasResolver(state, IdGenerator(seed=1))

    known = resolver.resolve("the Warden")
    assert known.entity_id == fixture.kell and known.is_new is False

    also_known = resolver.resolve("Kael")
    assert also_known.entity_id == fixture.kael

    fresh = resolver.resolve("Mara", kind=EntityKind.character)
    assert fresh.is_new is True
    assert fresh.note is not None and "Mara" in fresh.note

    again = resolver.resolve("Mara")
    assert again.entity_id == fresh.entity_id, "one entity per name, not one per mention"
    assert len(resolver.new_entities) == 1


def test_a_new_name_that_looks_like_an_existing_one_is_flagged(fixture: Fixture) -> None:
    resolver = AliasResolver(fixture.repo.state(), IdGenerator(seed=1))
    resolution = resolver.resolve("Warden of Kell the Elder")
    assert resolution.is_new is True, "a name that is not an exact alias creates an entity"
    assert resolution.note is not None
    assert "Warden of Kell" in resolution.note, "and points at what it might duplicate"
    assert "merge" in resolution.note


# --- proposals become diffs ---------------------------------------------------


async def test_beat_proposal_becomes_a_diff_that_applies(fixture: Fixture) -> None:
    router, _ = router_with([canned(BEAT_PAYLOAD)])
    proposer = Proposer(router, IdGenerator(seed=11))
    state = fixture.repo.state()

    proposals = await proposer.propose(state, Level.beat, target_node_id=fixture.scene, k=1)
    assert len(proposals) == 1
    proposal = proposals[0]

    after = apply(state, proposal.diff)
    new_beats = [b for b in after.beats_in_order() if b.id not in state.nodes]
    assert len(new_beats) == 1
    beat = new_beats[0]
    assert beat.title == "The Warden's offer"
    assert len(beat.produces) == 2, "produces is maintained from the declared facts"

    secret = next(f for f in after.facts.values() if f.predicate is Predicate.secret)
    assert secret.established_by_beat == beat.id
    assert after.knows_at(fixture.kell, secret.id, beat.id) is True
    assert after.knows_at(fixture.kael, secret.id, beat.id) is False

    assert proposal.rationale
    assert proposal.delta_summary
    assert proposal.raw_text


async def test_consumes_are_resolved_against_real_fact_ids(fixture: Fixture) -> None:
    payload = dict(BEAT_PAYLOAD, consumes=[str(fixture.fact_f), "f_does_not_exist"])
    router, _ = router_with([canned(payload)])
    proposer = Proposer(router, IdGenerator(seed=12))
    state = fixture.repo.state()

    proposal = (await proposer.propose(state, Level.beat, target_node_id=fixture.scene, k=1))[0]
    after = apply(state, proposal.diff)
    beat = next(b for b in after.beats_in_order() if b.id not in state.nodes)
    assert beat.consumes == (fixture.fact_f,), "invented fact ids are dropped, real ones kept"
    assert after.consumers_of[fixture.fact_f] == tuple(
        sorted({fixture.beat_b, beat.id}, key=lambda n: after.seq[n])
    )


async def test_a_new_character_produces_an_entity_and_a_review_note(fixture: Fixture) -> None:
    payload = dict(
        BEAT_PAYLOAD,
        new_characters=[{"name": "Mara", "kind": "character", "description": "a smuggler"}],
        produces=[
            {
                "subject": "Mara",
                "predicate": "goal",
                "object": "reach the harbour",
                "object_is_entity": False,
                "known_by": [],
            }
        ],
    )
    router, _ = router_with([canned(payload)])
    proposer = Proposer(router, IdGenerator(seed=13))
    state = fixture.repo.state()

    proposal = (await proposer.propose(state, Level.beat, target_node_id=fixture.scene, k=1))[0]
    assert any(isinstance(op, AddEntity) for op in proposal.diff.ops)
    assert any("Mara" in note for note in proposal.notes)

    after = apply(state, proposal.diff)
    mara = after.entity_by_name("Mara")
    assert mara is not None and mara.kind is EntityKind.character
    assert any(f.subject == mara.id for f in after.facts.values())


async def test_episode_proposal_populates_the_serial_fields(fixture: Fixture) -> None:
    payload = {
        "title": "Ashfall",
        "what_happens": "Kael is caught stealing and discovers what he can do.",
        "audience_learns": "the ash answers him",
        "audience_feels": "unsettled",
        "location": "Ashfall",
        "time": "winter",
        "hook": "A boy with nothing steals from the one person who would notice.",
        "cliffhanger": "The Warden calls him by a name he has never told anyone.",
        "recap_of_previous": "",
        "threads_opened": ["Who told the Warden Kael's name?"],
        "threads_touched": [],
        "threads_closed": [],
        "target_length": 2200,
        "rationale": "Opens on the seed's promise.",
        "delta_summary": ["opens the serial in Ashfall"],
    }
    router, _ = router_with([canned(payload)])
    proposer = Proposer(router, IdGenerator(seed=14))
    state = fixture.repo.state()

    proposal = (await proposer.propose(state, Level.episode, target_node_id=fixture.story, k=1))[0]
    assert any(isinstance(op, OpenThread) for op in proposal.diff.ops)

    after = apply(state, proposal.diff)
    episodes = [n for n in after.nodes_of_type(NodeType.episode) if n.id not in state.nodes]
    assert len(episodes) == 1
    episode = episodes[0]
    assert isinstance(episode, Episode)
    assert episode.hook and episode.cliffhanger
    assert episode.target_length == 2200
    assert len(episode.threads_out) == 1, "an open thread is carried out of the episode"
    assert after.threads[episode.threads_out[0]].description.startswith("Who told")


async def test_premise_proposal_updates_the_root_and_adds_the_cast(fixture: Fixture) -> None:
    payload = {
        "premise": "An orphan in a war-blasted city discovers the ash obeys him.",
        "existential_question": "Is power worth what it costs the powerless?",
        "title": "The Ashfall Orphan",
        "main_characters": [
            {"name": "Kael", "kind": "character", "description": "the orphan"},
            {"name": "Mara", "kind": "character", "description": "a smuggler"},
        ],
        "rationale": "Keeps the seed's asymmetry.",
        "delta_summary": ["develops the premise"],
    }
    router, _ = router_with([canned(payload)])
    proposer = Proposer(router, IdGenerator(seed=15))
    state = fixture.repo.state()

    proposal = (await proposer.propose(state, Level.premise, k=1))[0]
    after = apply(state, proposal.diff)
    root = after.nodes[after.root_id]  # type: ignore[index]
    assert getattr(root, "existential_question", "").startswith("Is power")
    assert after.entity_by_name("Mara") is not None
    assert len(after.entities) == len(state.entities) + 1, "Kael already existed; only Mara is new"


async def test_prose_proposal_is_attributed_to_the_ai(fixture: Fixture) -> None:
    payload = {
        "text": "The ash came down like flour. Kael did not look up. He counted three guards.",
        "rationale": "Short sentences, no interiority.",
        "delta_summary": ["writes the opening"],
    }
    router, _ = router_with([canned(payload)])
    proposer = Proposer(router, IdGenerator(seed=16))
    state = fixture.repo.state()

    proposal = (await proposer.propose(state, Level.prose, target_node_id=fixture.beat_a, k=1))[0]
    after = apply(state, proposal.diff)
    prose = after.prose_for(fixture.beat_a)
    assert prose is not None
    assert prose.text.startswith("The ash came down")
    assert prose.spans[0].source is Authorship.ai
    assert prose.spans[0].proposal_id == proposal.id


async def test_prompts_are_fed_slices_not_the_whole_state(fixture: Fixture) -> None:
    router, provider = router_with([canned(BEAT_PAYLOAD)])
    proposer = Proposer(router, IdGenerator(seed=17))
    state = fixture.repo.state()
    await proposer.propose(state, Level.beat, target_node_id=fixture.scene, k=1)

    prompt = provider.prompts_for("propose.beat")[0]
    assert "ESTABLISHED FACTS" in prompt
    assert str(fixture.fact_f) in prompt, "fact ids are shown so `consumes` can cite them"
    # The slice carries the path down to the insertion point, not the whole tree.
    assert "Beat D" in prompt, "the beat being written after is context"
    for unrelated in ("Beat A", "Beat B", "Beat C"):
        assert unrelated not in prompt, f"{unrelated} is not on the path and must not be sent"


async def test_k_samples_use_distinct_cache_slots(fixture: Fixture) -> None:
    router, provider = router_with(lambda _r: canned(BEAT_PAYLOAD))
    proposer = Proposer(router, IdGenerator(seed=18))
    proposals = await proposer.propose(
        fixture.repo.state(), Level.beat, target_node_id=fixture.scene, k=3
    )
    assert len(proposals) == 3
    assert {r.sample_index for r in provider.requests} == {0, 1, 2}
    assert len({p.id for p in proposals}) == 3


# --- extraction ---------------------------------------------------------------


async def test_extraction_lands_facts_knows_and_threads(fixture: Fixture) -> None:
    payload = {
        "facts": [
            {
                "subject": "Kael",
                "predicate": "injury",
                "object": "a burned left hand",
                "object_is_entity": False,
                "known_by": ["Kael"],
            }
        ],
        "new_characters": [],
        "threads_opened": ["What did Kael touch?"],
        "threads_touched": [],
    }
    router, provider = router_with([canned(payload)], name="groq")
    proposer = Proposer(router, IdGenerator(seed=19))
    state = fixture.repo.state()

    diff = await proposer.extract(state, fixture.beat_c, "He pulled his hand back, blistered.")
    after = apply(state, diff)

    injury = next(f for f in after.facts.values() if f.predicate is Predicate.injury)
    assert injury.source.value == "human", "extracted facts are attributed to the writer's prose"
    assert injury.established_by_beat == fixture.beat_c
    assert after.knows_at(fixture.kael, injury.id, fixture.beat_c) is True
    assert any(t.description.startswith("What did Kael") for t in after.threads.values())
    assert provider.requests[0].purpose == "extract.facts"
    assert provider.requests[0].temperature == 0.0


async def test_extraction_failure_is_an_empty_diff_not_a_crash(fixture: Fixture) -> None:
    router, _ = router_with(["garbage", "still garbage"])
    proposer = Proposer(router, IdGenerator(seed=20))
    diff = await proposer.extract(fixture.repo.state(), fixture.beat_c, "some prose")
    assert len(diff) == 0


SENTENCES = (
    "Ronnie leans on the bus shelter and lets the cast take his weight. "
    "Marguerite watches from the bench with the tin on her knees. "
    "Nobody says anything about Zurich. "
    "The 42 is late again, which is the only honest thing on the road. "
)


def test_over_long_prose_is_cut_at_a_sentence_rather_than_mid_word() -> None:
    """Gemini does not enforce maxLength, and a repair call per proposal is not free.

    Measured during the first evaluation run: roughly 80% of proposals were failing
    validation on a single over-long field and costing a whole extra call to recover. So
    the contract is truncate, not reject -- but the first version of it cut mid-word, and
    the fragment became permanent state that every later prompt read back as context.
    Roughly a third of one writer's candidates ended mid-clause.
    """
    from storygit.agents.schemas import PARAGRAPH, EpisodeProposal

    episode = EpisodeProposal.model_validate(
        {
            "title": "The Ballast Announcement",
            "what_happens": SENTENCES * 6,
            "threads_opened": [f"thread {i}" for i in range(30)],
            "rationale": SENTENCES * 4,
        }
    )
    assert len(episode.what_happens) <= PARAGRAPH
    assert episode.what_happens.endswith("."), "a bounded field ends where a sentence ends"
    assert "  " not in episode.what_happens
    assert SENTENCES.strip().split(". ")[0] in episode.what_happens, "the start survives"
    assert len(episode.threads_opened) == 4
    assert len(episode.rationale) <= PARAGRAPH


def test_a_field_with_no_sentence_inside_its_bound_is_re_asked_not_committed() -> None:
    """Every available cut is mid-clause, so there is nothing safe to commit.

    Raising here is what reaches ``complete_structured``, which repairs once by
    construction. A dropped candidate costs one sample out of six; a committed fragment
    costs every prompt that reads it afterwards.
    """
    import pytest
    from pydantic import ValidationError

    from storygit.agents.schemas import EpisodeProposal

    with pytest.raises(ValidationError, match="no sentence ending inside it"):
        EpisodeProposal.model_validate(
            {
                "title": "The Ballast Announcement",
                "what_happens": "and then " + "a very long unbroken clause that never ends " * 40,
            }
        )


def test_padding_and_keyboard_mash_are_re_asked_not_committed() -> None:
    """One field came back as literal mash padded to a minimum length, and was committed.

    Two cheap signals, neither of which touches real prose: too few distinct characters
    for the length, and one short chunk repeated to fill the space.
    """
    import pytest
    from pydantic import ValidationError

    from storygit.agents.schemas import EpisodeProposal

    for mash in (
        "asdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdf",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "beat beat beat beat beat beat beat beat beat beat beat beat.",
    ):
        with pytest.raises(ValidationError, match="padding rather than writing"):
            EpisodeProposal.model_validate(
                {"title": "The Ballast Announcement", "what_happens": mash}
            )


def test_ordinary_prose_is_never_mistaken_for_padding() -> None:
    """The guard has to be free on real writing, or it is worse than the bug."""
    from storygit.agents.schemas import EpisodeProposal

    episode = EpisodeProposal.model_validate(
        {"title": "The Hollow Knock", "what_happens": SENTENCES}
    )
    assert episode.what_happens == SENTENCES


def test_truncation_leaves_well_formed_output_alone() -> None:
    from storygit.agents.schemas import BeatProposal

    beat = BeatProposal.model_validate(BEAT_PAYLOAD)
    assert beat.title == BEAT_PAYLOAD["title"]
    assert beat.what_happens == BEAT_PAYLOAD["what_happens"]
    assert len(beat.produces) == 2


def test_the_prompt_states_the_bounds_because_the_provider_ignores_them(
    fixture: Fixture,
) -> None:
    from storygit.agents import prompts
    from storygit.agents.schemas import BeatProposal
    from storygit.graph.slices import StateSlice

    _, user = prompts.beat_prompt(StateSlice(), "go on", BeatProposal, "The market")
    assert "maxLength" in user.content
    assert "truncated" in user.content


def test_truncated_json_is_recovered_rather_than_thrown_away() -> None:
    """A model that runs out of tokens mid-string still produced a usable candidate.

    Measured on the live free tier: roughly a third of episode proposals were being lost
    this way, and once all six samples for one episode were lost, which stalled an
    evaluation run. Recovering costs nothing; the alternative is a repair call, or a gap.
    """
    from storygit.agents.parse import close_truncated_json
    from storygit.agents.schemas import EpisodeProposal

    truncated = (
        '{"title": "Echoes in the Grey", "what_happens": "Kael returns to the market and '
        'finds the stalls burned", "hook": "The ash is falling upwards, and only he'
    )
    recovered = parse_into(EpisodeProposal, truncated)
    assert recovered.title == "Echoes in the Grey"
    assert recovered.hook.startswith("The ash is falling upwards")

    # Nested structures close in the right order.
    assert json.loads(close_truncated_json('{"a": [1, {"b": "x') or "") == {"a": [1, {"b": "x"}]}
    # A dangling key is dropped rather than invented.
    assert json.loads(close_truncated_json('{"a": 1, "b":') or "") == {"a": 1}
    # Well-formed JSON is left alone.
    assert close_truncated_json('{"a": 1}') is None


def test_recovery_never_invents_a_required_field() -> None:
    """If the truncation lost something required, the candidate genuinely does not exist."""
    from storygit.agents.schemas import BeatProposal

    with pytest.raises(SchemaParseError):
        # `title` and `what_happens` are required and were never reached.
        parse_into(BeatProposal, '{"audience_feels": "wary and a little')


async def test_a_degenerate_field_costs_one_re_ask_and_then_the_candidate() -> None:
    """The whole contract, through the real call path: truncate, re-ask once, then drop.

    Rejecting outright is what the bounds used to do and it cost an 80% repair rate; not
    checking at all is what let a keyboard-mash field become permanent state. One re-ask is
    the middle, and it is cheap: a dropped candidate is one sample out of six, while a
    committed fragment is read back as context by every prompt after it.
    """
    from tests.mockprovider import MockProvider

    from storygit.agents.parse import complete_structured
    from storygit.agents.schemas import EpisodeProposal
    from storygit.providers.base import LLMRequest, SchemaParseError
    from storygit.providers.router import Router

    mash = json.dumps({"title": "The Ballast Announcement", "what_happens": "asdf" * 30})
    good = json.dumps({"title": "The Ballast Announcement", "what_happens": SENTENCES})

    # Mash, then a real answer: the re-ask rescues the candidate.
    rescued = MockProvider([mash, good])
    router = Router({"gemini": rescued})
    parsed, _ = await complete_structured(
        router,
        LLMRequest(messages=(), purpose="propose.episode"),
        EpisodeProposal,
    )
    assert parsed.what_happens == SENTENCES
    assert len(rescued.requests) == 2, "exactly one re-ask, not a retry loop"

    # Mash twice: the candidate is dropped rather than committed.
    hopeless = MockProvider([mash, mash])
    with pytest.raises(SchemaParseError, match="failed to validate twice"):
        await complete_structured(
            Router({"gemini": hopeless}),
            LLMRequest(messages=(), purpose="propose.episode"),
            EpisodeProposal,
        )
    assert len(hopeless.requests) == 2, "and it stops after one re-ask"
