"""Architectural invariants, tested structurally rather than by convention.

Each of these is a claim the design makes about itself, and a claim that a comment cannot
enforce. They are cheap to check by parsing the source, and they are exactly the properties
that would decay first under maintenance — so they get tests.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "storygit"


def modules(package: str) -> list[Path]:
    """Every Python file in a package."""
    return sorted((SRC / package).rglob("*.py"))


def imports_of(path: Path) -> set[str]:
    """Every module name imported by a file, including inside functions."""
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def calls_named(path: Path, name: str) -> list[ast.Call]:
    """Every call whose callee ends in ``name``."""
    tree = ast.parse(path.read_text())
    out: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Attribute) and func.attr == name) or (
            isinstance(func, ast.Name) and func.id == name
        ):
            out.append(node)
    return out


# --- the deterministic core stays deterministic -------------------------------


@pytest.mark.parametrize("package", ["domain", "graph", "store"])
def test_the_deterministic_core_cannot_reach_a_model(package: str) -> None:
    """domain/, graph/, and store/ must not import providers, agents, torch, or httpx.

    This is what makes "propagation cannot be wrong because a model was wrong" a structural
    property rather than a promise. It is also what lets the entire state layer be tested
    with no network and no ML dependency.
    """
    forbidden = (
        "providers",
        "agents",
        "selection",
        "continuity",
        "preference",
        "torch",
        "transformers",
        "httpx",
        "sentence_transformers",
    )
    # soft_edges.py is the one deliberate exception: it *is* the embedding ablation, and it
    # lives here so it can satisfy the EdgeProvider protocol. Its import is lazy, which the
    # next test asserts, so importing graph/ still costs nothing.
    for path in modules(package):
        if path.name == "soft_edges.py":
            continue
        for module in imports_of(path):
            assert not any(bad in module for bad in forbidden), (
                f"{path.relative_to(SRC)} imports {module}, which makes {package}/ "
                "non-deterministic"
            )


def test_the_embedding_ablation_loads_lazily() -> None:
    """Importing graph/ must not pull in an encoder.

    soft_edges.py is allowed to use embeddings -- it is the ablation -- but only inside a
    function. If the import moved to module scope, importing anything from graph/ would load
    a few hundred megabytes of model, and the deterministic core would stop being free.
    """
    tree = ast.parse((SRC / "graph" / "soft_edges.py").read_text())
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", "") or ""
            names = [alias.name for alias in node.names]
            assert "selection" not in module, "the encoder import must stay inside a function"
            assert not any("torch" in n or "sentence" in n for n in names)


def test_layer_one_of_the_checker_cannot_reach_a_model() -> None:
    """Layer 1's whole value is that it cannot be wrong for a subtle reason."""
    forbidden = ("providers", "agents", "router", "torch", "transformers", "httpx")
    for module in imports_of(SRC / "continuity" / "layer1.py"):
        assert not any(bad in module for bad in forbidden), f"continuity/layer1.py imports {module}"


# --- one door into state ------------------------------------------------------


def test_only_the_repository_writes_state() -> None:
    """`apply` is called outside the store only for previews, never to persist.

    The claim is that every change to the story goes through
    `Repository.commit_diff`, which is what makes the snapshot history a complete record.
    `apply` itself is pure, so calling it is harmless -- but a caller that applies and then
    stashes the result somewhere would have created state nobody can audit. The permitted
    callers are the ones that preview.
    """
    permitted = {
        "store/repository.py",  # commit_diff itself
        "store/branches.py",  # the structural-diff fixup pass
        "graph/propagation.py",  # preview and apply_marks
        "selection/select.py",  # checking a candidate on a scratch state
    }
    for path in SRC.rglob("*.py"):
        relative = str(path.relative_to(SRC))
        if relative in permitted:
            continue
        if calls_named(path, "apply"):
            source = path.read_text()
            assert "from storygit.domain.apply import apply" not in source, (
                f"{relative} calls apply() outside the permitted set; state must change "
                "through Repository.commit_diff"
            )


def test_the_api_never_touches_the_store_directly() -> None:
    """Routes go through the engine or the repository facade, never through apply()."""
    for path in modules("api"):
        source = path.read_text()
        assert "domain.apply" not in source, f"{path.name} imports apply()"
        assert "SnapshotStore" not in source, f"{path.name} reaches past the repository"


def test_the_library_does_not_import_its_own_test_harness() -> None:
    """A deliverable package that imports `eval/` breaks outside the repo root.

    It did, once: the API seeded an empty database by importing `eval.run.seed_story`, and
    the server failed to start from any other directory.
    """
    for path in SRC.rglob("*.py"):
        for module in imports_of(path):
            assert not module.startswith("eval"), f"{path.relative_to(SRC)} imports {module}"
            assert not module.startswith("tests"), f"{path.relative_to(SRC)} imports {module}"


# --- the metered provider stays locked ----------------------------------------


def test_only_the_openrouter_module_mentions_the_enable_flag() -> None:
    """Grep the tree for OPENROUTER: every path must go through the one lock.

    A second place that reads the flag is a second place that can get it wrong.
    """
    allowed = {"config.py", "openrouter.py"}
    for path in SRC.rglob("*.py"):
        if path.name in allowed:
            continue
        # Parse rather than grep: the flag's *name* appears in several docstrings, which is
        # documentation rather than a second code path. What must not exist is a second
        # place that reads it.
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "openrouter_enabled":
                # router.py may *transport* the raw flag into the provider's constructor.
                # What it must never do is decide, which the next assertion covers.
                assert path.name == "router.py", f"{path.name} reads the enable flag"
            if isinstance(node, ast.Constant) and node.value == "OPENROUTER_ENABLED":
                pytest.fail(f"{path.name} reads the environment variable directly")

    # And the transporter must not compare it: only openrouter.py decides.
    router = (SRC / "providers" / "router.py").read_text()
    for comparison in ('== "true"', '!= "true"', "is_enabled ="):
        assert comparison not in router, f"router.py decides the lock with {comparison}"


def test_the_lock_is_an_exact_string_comparison() -> None:
    """A lock that can be tripped by a typo is not a lock."""
    source = (SRC / "providers" / "openrouter.py").read_text()
    assert 'ENABLED_SENTINEL = "true"' in source
    assert "enabled_flag == ENABLED_SENTINEL" in source
    for sloppy in (".lower()", ".strip()", "in (", "bool("):
        assert f"enabled_flag{sloppy}" not in source, (
            f"the lock normalizes with {sloppy}, so a near-miss would unlock it"
        )


def test_no_module_hardcodes_a_key() -> None:
    """Keys come from settings and nowhere else."""
    for path in SRC.rglob("*.py"):
        source = path.read_text()
        for marker in ("AIza", "gsk_", "sk-or-v1"):
            assert marker not in source, f"{path.name} appears to contain a literal key"


# --- prompts see slices, not state --------------------------------------------


def test_prompt_builders_take_slices() -> None:
    """Every prompt builder's state argument is a StateSlice, never a StoryState.

    Sending the whole story would not fit, would cost a fortune, and measurably makes the
    output vaguer -- so the type is the guard.
    """
    tree = ast.parse((SRC / "agents" / "prompts.py").read_text())
    builders = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.endswith("_prompt")
    ]
    assert len(builders) >= 5, "expected one prompt builder per level"
    for builder in builders:
        annotations = [ast.unparse(arg.annotation) for arg in builder.args.args if arg.annotation]
        assert "StoryState" not in annotations, f"{builder.name} takes a whole StoryState"
        assert "StateSlice" in annotations, f"{builder.name} does not take a StateSlice"


def test_every_provider_request_carries_a_purpose_tag() -> None:
    """Routing, the cost report, and the call log all key on it, so it cannot be optional."""
    source = (SRC / "providers" / "base.py").read_text()
    tree = ast.parse(source)
    request = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LLMRequest"
    )
    purpose = next(
        node
        for node in request.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "purpose"
    )
    assert purpose.value is None, "purpose has a default, so a call could omit it"


def test_no_public_function_is_unreachable() -> None:
    """The code standard says no dead code, so check rather than trust.

    Route handlers are excluded: they are reached through a decorator, so a textual search
    would report every one of them as unused.
    """
    import re

    defined: dict[str, Path] = {}
    for path in SRC.rglob("*.py"):
        source = path.read_text()
        decorated = set(re.findall(r"@\w+\.(?:get|post|put|delete)\([^)]*\)\s*\ndef (\w+)", source))
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name.startswith("_") or node.name in decorated:
                    continue
                defined[node.name] = path

    roots = [SRC, Path(__file__).parent, SRC.parents[1] / "eval", SRC.parents[1] / "scripts"]
    used: set[str] = set()
    for root in roots:
        for path in root.rglob("*.py"):
            source = path.read_text()
            for name, home in defined.items():
                if source.count(name) > (1 if home == path else 0):
                    used.add(name)

    unused = sorted(set(defined) - used)
    assert unused == [], "public names nothing references: " + ", ".join(
        f"{n} ({defined[n].name})" for n in unused
    )


def test_no_todos_are_left_without_a_decision_record() -> None:
    """The code standard: no TODOs left without a note in DECISIONS.md.

    The simplest way to satisfy that is to leave none, which is what this asserts.
    """
    for root in (SRC, SRC.parents[1] / "eval", SRC.parents[1] / "scripts"):
        for path in root.rglob("*.py"):
            source = path.read_text()
            for marker in ("TODO", "FIXME", "XXX", "HACK"):
                assert marker not in source, f"{path.name} contains a {marker}"
