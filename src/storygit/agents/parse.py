"""Getting structured output back out of a model, reliably.

Two defences, in order. The model is asked for JSON and given the schema. If what comes
back does not parse, one *repair* call is made: the model is shown its own output and
the validation error and asked to return corrected JSON only. If that fails too, a typed
:class:`~storygit.providers.base.SchemaParseError` is raised and the caller resamples.

The repair retry is worth its cost because the common failure is trivial — a trailing
comma, a missing field, prose wrapped around the JSON — and a resample throws away a
candidate that was probably fine.
"""

from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from storygit.providers.base import LLMRequest, LLMResponse, Message, Role, SchemaParseError
from storygit.providers.router import Router

T = TypeVar("T", bound=BaseModel)


def extract_json(text: str) -> str:
    """Pull the JSON object out of a model response.

    Handles the three things models do: return clean JSON, wrap it in a fenced code
    block, or surround it with a sentence of commentary.

    Args:
        text: Raw model output.

    Returns:
        The substring most likely to be the JSON object.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped
    start = min(
        (i for i in (stripped.find("{"), stripped.find("[")) if i >= 0),
        default=-1,
    )
    if start < 0:
        return stripped
    closer = "}" if stripped[start] == "{" else "]"
    end = stripped.rfind(closer)
    return stripped[start : end + 1] if end > start else stripped[start:]


def close_truncated_json(text: str) -> str | None:
    """Repair JSON that was cut off mid-generation, if it can be.

    A small model can fall into a repetition loop inside one string field and run until it
    hits the token cap, leaving a JSON object that is well formed right up to the point it
    stops. Throwing that away costs a whole extra call to recover a candidate whose only
    problem is one over-long field — and the schemas already truncate over-long fields, so
    the recovered object is usually exactly what the bound wanted anyway.

    Measured on the live free tier: this was happening on roughly a third of episode
    proposals, and *every* one of the six samples for one episode, which stalled a run.

    The repair is deliberately conservative. It closes an unterminated string, drops a
    dangling key or comma, and closes open brackets in the order they were opened. It never
    invents a value, so a required field that was never reached still fails validation —
    which is the correct outcome, because that candidate genuinely does not exist.

    Args:
        text: JSON text that failed to parse.

    Returns:
        Repaired JSON, or ``None`` if the text was not truncated in a way this can fix.
    """
    if not text.strip().startswith(("{", "[")):
        return None

    stack: list[str] = []
    in_string = False
    escaped = False
    last_structural = -1

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
                last_structural = index
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
            last_structural = index
        elif char in "}]":
            if not stack:
                return None
            stack.pop()
            last_structural = index
        elif char in ",:" or not char.isspace():
            last_structural = index

    if not stack and not in_string:
        return None  # not truncated; the failure is something else

    repaired = text
    if in_string:
        # Cut back to the last complete escape so a half-written \u does not break it.
        repaired = repaired.rstrip("\\")
        repaired += '"'
    else:
        repaired = repaired[: last_structural + 1]

    # A dangling key ("hook": ) or a trailing comma leaves the object unparseable.
    stripped = repaired.rstrip()
    while stripped.endswith((",", ":")):
        stripped = stripped[:-1].rstrip()
        if stripped.endswith('"'):
            # Drop the orphaned key as well, back to the comma or brace before it.
            cut = max(stripped.rfind(","), stripped.rfind("{"), stripped.rfind("["))
            if cut < 0:
                return None
            stripped = stripped[: cut + 1] if stripped[cut] in "{[" else stripped[:cut]
    repaired = stripped + "".join(reversed(stack))
    return repaired


def parse_into(model_cls: type[T], text: str) -> T:
    """Validate model output into a schema class.

    Args:
        model_cls: The expected response model.
        text: Raw model output.

    Returns:
        The parsed model.

    Raises:
        SchemaParseError: If the text is not valid JSON for that schema. The raw text is
            attached so the caller can attempt a repair.
    """
    payload = extract_json(text)
    try:
        return model_cls.model_validate_json(payload)
    except (ValidationError, json.JSONDecodeError) as exc:
        repaired = close_truncated_json(payload)
        if repaired is not None:
            try:
                return model_cls.model_validate_json(repaired)
            except (ValidationError, json.JSONDecodeError):
                pass  # the truncation lost something required; fall through and report
        raise SchemaParseError(
            f"{model_cls.__name__}: model output did not validate ({exc})", raw=text
        ) from exc


def repair_request(
    request: LLMRequest, model_cls: type[T], bad_text: str, error: str
) -> LLMRequest:
    """Build the one repair call.

    Args:
        request: The original request.
        model_cls: The expected response model.
        bad_text: What the model returned.
        error: The validation error.

    Returns:
        A new request that shows the model its output and the error.
    """
    schema = json.dumps(model_cls.model_json_schema(), indent=2)
    instruction = (
        "Your previous reply did not match the required JSON schema.\n\n"
        f"Schema:\n{schema}\n\n"
        f"Your reply:\n{bad_text[:2000]}\n\n"
        f"Validation error:\n{error}\n\n"
        "Return corrected JSON only. No commentary, no code fences."
    )
    return request.model_copy(
        update={
            "messages": (
                *request.messages,
                Message(role=Role.assistant, content=bad_text[:2000]),
                Message(role=Role.user, content=instruction),
            ),
            "temperature": 0.0,
            "sample_index": request.sample_index,
            "purpose": f"{request.purpose}.repair",
        }
    )


async def complete_structured(
    router: Router, request: LLMRequest, model_cls: type[T]
) -> tuple[T, LLMResponse]:
    """Run a call and return validated structured output, repairing once if needed.

    Args:
        router: The router to call through.
        request: The request. Its ``json_schema`` is filled from ``model_cls`` if unset.
        model_cls: The expected response model.

    Returns:
        ``(parsed, response)``, where ``response`` is the call that produced it.

    Raises:
        SchemaParseError: If both the original and the repair attempt fail to validate.
    """
    if request.json_schema is None:
        request = request.model_copy(update={"json_schema": model_cls.model_json_schema()})

    response = await router.complete(request)
    try:
        return parse_into(model_cls, response.text), response
    except SchemaParseError as first:
        repair = repair_request(request, model_cls, response.text, str(first))
        repaired = await router.complete(repair)
        try:
            return parse_into(model_cls, repaired.text), repaired
        except SchemaParseError as second:
            raise SchemaParseError(
                f"{model_cls.__name__}: output failed to validate twice "
                f"(first: {first}; after repair: {second})",
                raw=repaired.text,
            ) from second
