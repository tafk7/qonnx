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

"""Findings and refusal bases shared by qonnx's operation families.

Import Finding and Refusal here. Their standard-library-only implementation in
qonnx.core also travels with a standalone copy of the datatype module.
"""

from __future__ import annotations

import threading
from contextvars import ContextVar, Token
from types import TracebackType

from qonnx.core import Finding as Finding
from qonnx.core import Refusal as Refusal


class _Scope:
    def __init__(self, parent: _Scope | None, propagate: bool) -> None:
        self.found: list[Finding] = []
        self.parent = parent
        self.propagate = propagate
        self.closed = False
        self.lock = threading.Lock()

    def add(self, finding: Finding) -> bool:
        """Keep an equal finding once, unless this scope has already closed."""
        with self.lock:
            if self.closed:
                return False
            if finding not in self.found:
                self.found.append(finding)
            return True


_SCOPE: ContextVar[_Scope | None] = ContextVar("qonnx_findings", default=None)


def _open(scope: _Scope | None) -> _Scope | None:
    while scope is not None and scope.closed:
        scope = scope.parent
    return scope


def _report(scope: _Scope | None, finding: Finding) -> None:
    while scope is not None:
        if scope.add(finding):
            return
        scope = scope.parent


def report_finding(finding: Finding) -> None:
    """Report to this context's innermost open collector; without one, do nothing.

    A wrong Python type raises TypeError even when no collector is open.
    """
    if not isinstance(finding, Finding):
        raise TypeError("report_finding takes a Finding")
    _report(_SCOPE.get(), finding)


def collecting_findings() -> bool:
    """Whether a finding reported in this context would be kept by an open scope."""
    return _open(_SCOPE.get()) is not None


class collect_findings:
    """Collect reports from this block, including nested transforms, in report order.

    Equal findings are kept once. On return or exception, a closing scope passes
    its findings to an enclosing open scope; propagate=False keeps them with its
    own reader. A task or copied context reporting after this scope closes goes
    to the innermost scope still open around it, or drops its reports if none is.

    Do not hold a scope open across a yield: a suspended generator shares its
    caller's context and would collect the consumer's reports too. Work started
    in a copied context should preferably be joined before the block ends.
    A collector may be reused after closing; overlapping entries raise RuntimeError.
    """

    def __init__(self, *, propagate: bool = True) -> None:
        self._propagate = propagate
        self._scope: _Scope | None = None
        self._token: Token[_Scope | None] | None = None

    def __enter__(self) -> list[Finding]:
        if self._scope is not None and not self._scope.closed:
            raise RuntimeError("a findings collector is already open")
        self._scope = _Scope(_SCOPE.get(), self._propagate)
        self._token = _SCOPE.set(self._scope)
        return self._scope.found

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None
    ) -> None:
        assert self._scope is not None and self._token is not None
        scope = self._scope
        _SCOPE.reset(self._token)
        with scope.lock:
            scope.closed = True
            found = tuple(scope.found)
        if scope.propagate:
            for finding in found:
                _report(scope.parent, finding)


__all__ = ["Finding", "Refusal", "collect_findings", "collecting_findings", "report_finding"]
