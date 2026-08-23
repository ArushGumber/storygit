"""Architectural invariants, tested structurally rather than by convention.

Each of these is a claim the design makes about itself, and a claim that a comment cannot
enforce. They are cheap to check by parsing the source, and they are exactly the properties
that would decay first under maintenance — so they get tests.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import get_args

import pytest

from storygit.domain.diff import Op

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


def test_the_prose_agrees_with_the_code_about_how_many_operations_there_are() -> None:
    """A document that miscounts the thing it is describing is a document nobody trusts.

    `presentable.tex` reads the count from a generated macro, so it cannot drift. The
    README states it in prose, so this is what stops it. It drifted once already, when
    the thirty-first operation was added.
    """
    count = len(get_args(get_args(Op)[0]))
    root = SRC.parents[1]
    prose = [root / "README.md", root / "frontend" / "src" / "tabs" / "Architecture.tsx"]
    for path in prose:
        stated = re.findall(r"(\d+) diff operations", path.read_text())
        assert stated, f"{path.name} no longer states the operation count"
        assert all(int(n) == count for n in stated), (
            f"{path.name} says {stated}, the union has {count}"
        )


def test_every_operation_can_actually_be_produced() -> None:
    """An operation nothing constructs is a capability the product does not have.

    `apply` handling an op proves only that the op *would* work. Four ops were defined,
    handled, and constructed by nothing when this was written — and two of them backed
    promises made in the interface and the paper: that a mined style rule can be deleted,
    and that a duplicate entity can be merged in one click. Both promises were false.

    `diff.py` defines them and `apply.py` handles them, so neither counts as a caller.
    """
    exempt = {SRC / "domain" / "diff.py", SRC / "domain" / "apply.py"}
    roots = [SRC, SRC.parents[1] / "eval", SRC.parents[1] / "scripts"]
    orphans = []
    for member in get_args(get_args(Op)[0]):
        name = member.__name__
        constructed = any(
            re.search(rf"\b{name}\s*\(", path.read_text())
            for root in roots
            for path in root.rglob("*.py")
            if path not in exempt
        )
        if not constructed:
            orphans.append(name)
    assert orphans == [], "operations nothing can produce: " + ", ".join(orphans)


def test_no_client_method_is_unreachable_from_the_interface() -> None:
    """The same dead-code standard, on the other side of the wire.

    `api.ts` is the only way the interface talks to the server, so a method there that no
    component calls is a capability the backend has and the writer cannot reach. Four were
    unreachable when this was written — the audit, the merge, the snapshot history, and the
    branch comparison — and the flags route's own docstring claimed the interface offered
    the audit as an explicit action.

    Checked from Python because this file is where the repo-wide standards live; it reads
    the TypeScript as text, which is all the check needs.
    """
    frontend = SRC.parents[1] / "frontend" / "src"
    client = (frontend / "api.ts").read_text()
    components = "\n".join(
        path.read_text() for path in frontend.rglob("*.tsx") if "__tests__" not in str(path)
    )
    methods = re.findall(r"^  (\w+):\s*(?:\(|=)", client, re.M)
    assert methods, "api.ts no longer looks like a method table"
    unreachable = [name for name in methods if f".{name}(" not in components]
    assert unreachable == [], "client methods no component calls: " + ", ".join(unreachable)


def test_every_figure_the_evaluation_produces_has_a_caption() -> None:
    """A caption says what a curve demonstrates; a filename says nothing.

    `bandit_selftest.svg` shipped with no caption, and the interface fell back to showing
    its filename under the figure. The captions live in the API rather than the frontend
    because they describe a property of the experiment, so this checks them against the
    figures the evaluation actually writes.
    """
    source = (SRC / "api" / "routers" / "artifacts.py").read_text()
    captioned = set(re.findall(r'"(\w+\.svg)":', source))
    produced = set(re.findall(r'"(\w+\.svg)"', (SRC.parents[1] / "eval" / "plots.py").read_text()))
    produced |= {
        name
        for path in (SRC.parents[1] / "eval").glob("*.py")
        for name in re.findall(r'/ "(\w+\.svg)"', path.read_text())
    }
    assert produced, "no figures found; this test has stopped checking anything"
    assert produced <= captioned, "figures with no caption: " + ", ".join(
        sorted(produced - captioned)
    )


def test_make_help_lists_every_target() -> None:
    """The README says `make help` shows "every regeneration command, named".

    It did not. The help target greps `^[a-zA-Z_-]+:` and `e2e` contains a digit, so the
    end-to-end smoke — one of the four things `make all` runs — was silently absent from
    the list of what this project can regenerate.
    """
    makefile = (SRC.parents[1] / "Makefile").read_text()
    phony = set(re.search(r"^\.PHONY:(.*)$", makefile, re.M).group(1).split())  # type: ignore[union-attr]
    documented = set(re.findall(r"^([a-zA-Z0-9_-]+):.*?## ", makefile, re.M))
    assert phony <= documented, "targets with no help text: " + ", ".join(
        sorted(phony - documented)
    )
    pattern = re.search(r"@grep -E '(\^[^']+)'", makefile).group(1)  # type: ignore[union-attr]
    listed = {name for name in documented if re.match(pattern.replace("$$", "$"), f"{name}: ## x")}
    assert listed == documented, "targets the help grep cannot match: " + ", ".join(
        sorted(documented - listed)
    )


def test_the_readme_example_actually_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The first code a reader tries is the one in the README.

    It is the cheapest possible broken promise and the most embarrassing one, so it is
    executed rather than eyeballed. The block is extracted by fence, so renaming a symbol
    it uses fails here rather than in front of whoever opened the repository first.
    """
    readme = (SRC.parents[1] / "README.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", readme, re.S)
    assert blocks, "the README no longer contains a Python example"
    monkeypatch.chdir(tmp_path)
    for block in blocks:
        exec(compile(block, "README.md", "exec"), {"__name__": "__readme__"})


def test_the_env_example_holds_no_values_that_could_be_keys() -> None:
    """A committed example file is the easiest place in a repository to leak a key.

    Every variable whose name says it holds a credential must be present and empty, so
    that filling one in locally and committing it fails here rather than in a git history
    that cannot be rewritten.
    """
    example = SRC.parents[1] / ".env.example"
    assert example.exists(), ".env.example is how a reader learns what to configure"
    for line in example.read_text().splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if any(word in name.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            assert value.strip() == "", f"{name} has a value in .env.example"
