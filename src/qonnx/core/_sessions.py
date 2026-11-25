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

"""ONNX Runtime sessions for execution, reused inside the reuse_sessions context of
qonnx.core.onnx_exec: one retained session of a stateless model, and both of ONNX
Runtime's thread pools capped at a given count. Outside the context every session is
constructed fresh with ONNX Runtime's default options.

A model is stateless where every node is a standard op whose schema states no seed and
no context-dependent function, whose function body (if it has one) and subgraphs are
stateless too, and which is not one of the random ops. A random or seeded op owns a
generator its session keeps, so it, and any node whose schema cannot be audited so,
runs a fresh session every time. The retained session is keyed by the model's bytes
and the thread count, and holds no inputs or results."""

from __future__ import annotations

import hashlib
import onnx
import onnxruntime as rt
from contextlib import contextmanager
from contextvars import ContextVar
from onnx import AttributeProto, ModelProto, NodeProto
from threading import Lock
from typing import Iterator, NamedTuple, Optional

_RANDOM = frozenset(
    {
        "RandomNormal",
        "RandomUniform",
        "RandomNormalLike",
        "RandomUniformLike",
        "Multinomial",
        "Bernoulli",
        "Dropout",
    }
)
_MAX_THREADS = 2_147_483_647


class _Reuse(NamedTuple):
    """The setting of a reuse_sessions context: the thread cap, or None for ONNX
    Runtime's default."""

    threads: Optional[int]


_REUSE: ContextVar[Optional[_Reuse]] = ContextVar("qonnx.core._sessions.reuse", default=None)
_lock = Lock()
_retained: Optional[tuple[bytes, Optional[int], rt.InferenceSession]] = None


@contextmanager
def reusing(threads: Optional[int]) -> Iterator[None]:
    """Sessions reused and capped at threads inside the context; the retained session
    is released on leaving the outermost one."""
    if threads is not None:
        if isinstance(threads, bool) or not isinstance(threads, int):
            raise TypeError(f"threads must be an int or None, not {type(threads).__name__}")
        if not 1 <= threads <= _MAX_THREADS:
            from qonnx.core.onnx_exec import ExecutionValueRefused

            raise ExecutionValueRefused.of(
                "setting-invalid",
                "reuse_sessions",
                f"threads must be in [1, {_MAX_THREADS}], not {threads}",
                threads=threads,
            )
    token = _REUSE.set(_Reuse(threads))
    try:
        yield
    finally:
        _REUSE.reset(token)
        if _REUSE.get() is None:
            release()


def release() -> None:
    """Drops the retained session."""
    global _retained
    with _lock:
        _retained = None


def _options(threads: Optional[int]) -> rt.SessionOptions:
    options = rt.SessionOptions()
    if threads is not None:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = threads
    return options


def _stateless(node: NodeProto, versions: dict[str, int], seen: frozenset[tuple[str, str, int]]) -> bool:
    """Whether a node keeps no state across runs, by its schema; False where the schema
    cannot show it."""
    if node.domain not in ("", "ai.onnx") or node.op_type in _RANDOM:
        return False
    version = versions.get(node.domain, versions.get("", 0))
    try:
        schema = onnx.defs.get_schema(node.op_type, version, "")
    except onnx.defs.SchemaError:
        return False
    if "seed" in schema.attributes:
        return False
    # A context-dependent function's expansion depends on facts only the runtime has.
    if getattr(schema, "has_context_dependent_function", None) is not False:
        return False
    has_function = getattr(schema, "has_function", None)
    if has_function is None:
        return False
    identity = (node.domain, node.op_type, schema.since_version)
    if has_function and identity not in seen:
        body = schema.function_body
        inner_versions = {**versions, **{item.domain: item.version for item in body.opset_import}}
        if not all(_stateless(inner, inner_versions, seen | {identity}) for inner in body.node):
            return False
    for attr in node.attribute:
        if attr.type == AttributeProto.GRAPH:
            graphs = [attr.g]
        elif attr.type == AttributeProto.GRAPHS:
            graphs = list(attr.graphs)
        else:
            continue
        if not all(_stateless(inner, versions, seen) for graph in graphs for inner in graph.node):
            return False
    return True


def stateless(model: ModelProto) -> bool:
    """Whether a model's session may be reused: no model-local functions, and every node
    stateless."""
    if model.functions:
        return False
    versions = {item.domain: item.version for item in model.opset_import}
    return all(_stateless(node, versions, frozenset()) for node in model.graph.node)


def session(model: ModelProto) -> rt.InferenceSession:
    """An ONNX Runtime session of model. Outside a reuse_sessions context, a fresh one
    with ONNX Runtime's default options. Inside, both thread pools capped at the
    context's threads; the retained session where it is of the same bytes and thread
    count, else a fresh one, retained where the model is stateless. At most one session
    is retained: it is released before another is constructed."""
    global _retained
    data = model.SerializeToString()
    reuse = _REUSE.get()
    if reuse is None:
        return rt.InferenceSession(data)
    key = hashlib.sha256(data).digest()
    with _lock:
        if _retained is not None and _retained[:2] == (key, reuse.threads):
            return _retained[2]
        _retained = None
        found = rt.InferenceSession(data, sess_options=_options(reuse.threads))
        if stateless(model):
            _retained = (key, reuse.threads, found)
        return found
