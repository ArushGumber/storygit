"""Typed errors raised by the domain layer.

Callers — and in chunk 6 the HTTP layer — branch on these, so every failure mode
that a writer could plausibly trigger has its own class rather than a bare
``ValueError``.
"""

from __future__ import annotations


class StoryGitError(Exception):
    """Base class for every error storygit raises."""


class ApplyError(StoryGitError):
    """A diff could not be applied to the given state."""


class UnknownNodeError(ApplyError):
    """An operation referenced a node that does not exist."""


class UnknownEntityError(ApplyError):
    """An operation referenced an entity that does not exist."""


class UnknownFactError(ApplyError):
    """An operation referenced a fact that does not exist."""


class UnknownThreadError(ApplyError):
    """An operation referenced a thread that does not exist."""


class DuplicateIdError(ApplyError):
    """An operation tried to create an object whose id is already taken."""


class InvalidStructureError(ApplyError):
    """An operation would break the shape of the plan tree."""


class CycleError(InvalidStructureError):
    """A move would make a node its own ancestor."""


class LockedNodeError(ApplyError):
    """An operation tried to modify a node the writer has locked."""


class SnapshotNotFoundError(StoryGitError):
    """A snapshot or branch reference does not resolve."""


class BranchExistsError(StoryGitError):
    """A branch name is already taken.

    Distinct from :class:`SnapshotNotFoundError` because the two mean opposite things and
    the API maps them to opposite statuses. Raising "not found" for "already exists" left
    one file catching the same class twelve lines apart and returning 409 in one place and
    404 in the other.
    """
