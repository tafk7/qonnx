# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of qonnx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Reasons a request is refused, and exceptions that carry those reasons.

A finding records a code, its public owner, a message and immutable details. A
refusal carries one or more findings. Families add the built-in exception that
callers already catch; a finding may also be carried without raising it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any, ClassVar, TypeVar

_F = TypeVar("_F", bound="Finding")
_R = TypeVar("_R", bound="Refusal")


def _freeze_detail(value: object) -> object:
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_detail(item) for item in value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("finding detail mapping keys must be strings")
        return tuple(sorted((key, _freeze_detail(item)) for key, item in value.items()))
    raise TypeError("finding details must contain primitive values, sequences or string mappings")


@dataclass(frozen=True, slots=True)
class Finding:
    """One reason: its code, public owner, message and frozen, sorted details."""

    code: str
    owner: str
    message: str
    details: tuple[tuple[str, object], ...] = ()
    SUBJECTS: ClassVar[tuple[str, ...]] = ("node", "tensor")

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("a finding requires a nonempty code")
        if any(not isinstance(key, str) for key, _ in self.details):
            raise TypeError("finding detail keys must be strings")
        details = tuple(sorted(((key, _freeze_detail(value)) for key, value in self.details), key=lambda pair: pair[0]))
        if len({key for key, _ in details}) != len(details):
            raise ValueError("finding detail keys must be unique")
        object.__setattr__(self, "details", details)

    def __str__(self) -> str:
        return f"{self.owner}: {self.code}: {self.message}"

    def subject_line(self) -> str:
        """Its line prefixed by the first subject, unless its message starts with it."""
        details = dict(self.details)
        subject = next((details[key] for key in self.SUBJECTS if key in details), None)
        if subject is None or self.message.startswith(f"{subject}:"):
            return str(self)
        return f"{subject}: {self}"

    def about(self: _F, **subjects: str) -> _F:
        """Name subjects not already stated, keeping the owner and existing subjects."""
        stated = dict(self.details)
        added = tuple((key, value) for key, value in subjects.items() if key not in stated)
        return replace(self, details=(*self.details, *added)) if added else self

    def record(self) -> dict[str, Any]:
        """JSON values for the code, owner, message and details."""
        return {"code": self.code, "owner": self.owner, "message": self.message, "details": dict(self.details)}


class Refusal(Exception):
    """A request refused, carrying at least one finding that explains why."""

    findings: tuple[Finding, ...]

    def __init__(self, findings: Iterable[Finding], message: str | None = None) -> None:
        self.findings = tuple(findings)
        if not self.findings or any(not isinstance(item, Finding) for item in self.findings):
            raise TypeError("a refusal carries at least one finding")
        self._default_message = message is None
        super().__init__("\n".join(map(str, self.findings)) if message is None else message)

    @classmethod
    def of(cls: type[_R], code: str, owner: str, message: str, **details: object) -> _R:
        """A refusal with one finding, its details frozen by the finding."""
        return cls((Finding(code, owner, message, tuple(details.items())),))

    def __str__(self) -> str:
        return str(self.args[0])

    def __reduce__(self) -> tuple[object, tuple[type[Refusal], tuple[Any, ...], dict[str, Any]]]:
        """Preserve type, args, findings and stored state across pickle and copy.

        As for Python exceptions, causes, contexts and tracebacks are not carried.
        Restore without calling a consumer subclass's constructor.
        """
        return _restored, (type(self), self.args, self.__dict__)


def _restored(cls: type[_R], args: tuple[Any, ...], state: dict[str, Any]) -> _R:
    error = cls.__new__(cls, *args)
    error.__dict__.update(state)
    return error


__all__ = ["Finding", "Refusal"]
