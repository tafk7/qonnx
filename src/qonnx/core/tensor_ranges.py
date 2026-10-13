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
# * Neither the name of AMD nor the names of its
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


"""A tensor's range on a model: where it is stored, how it is read and written, and
the index one inference pass reads it by. The range itself is ``qonnx.core.ranges``.

It is stored in the namespace ``qonnx.tensor`` (``qonnx.core.metadata``; it
follows its tensor into a subgraph cut from the graph, and the graph keeps its
own copy), key ``range``, in the tensor's ``quantization_annotation`` entry
beside ``finn_datatype`` (so ``rename_tensor`` renames it and
``RemoveUnusedTensors`` removes it with its tensor), as ``qonnx.core.ranges``'
text. Only a range that says more than the datatype is stored: the datatype's
full range is not. Nothing coerces: stored text in any other form, or a stored
range its datatype does not hold, is a ``MetadataError`` naming the entry and the
text.

A writer stating a range its own datatype does not hold contradicts itself: the
range is not stored, and a ``RangeContradiction`` warning names the tensor and
its writer (once per tensor in an inference pass, ``one_pass``).
"""

from __future__ import annotations

import hashlib
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from onnx import TensorAnnotation, TensorProto, numpy_helper
from typing import TYPE_CHECKING, Iterator, Optional

from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.metadata import Codec, Key, MetadataError, Namespace
from qonnx.core.metadata import clear as clear_namespace
from qonnx.core.metadata import read as read_namespace
from qonnx.core.metadata import write as write_namespace
from qonnx.core.ranges import RangeContradiction, TensorRange, _decode, _encode, full, holds
from qonnx.util.basic import get_by_name

if TYPE_CHECKING:
    from onnx import StringStringEntryProto
    from typing import MutableSequence

    from qonnx.core.modelwrapper import ModelWrapper

RANGE_CODEC: Codec[TensorRange] = Codec(
    'a range, {"nan": a bool, "range": [least, greatest]}, each end an int, "m*2^-k" (m odd, k >= 1), '
    '"inf" or "-inf", least <= greatest',
    lambda value: isinstance(value, TensorRange),
    _encode,
    _decode,
)

TENSOR = Namespace("qonnx.tensor", version=1, follow=True)
RANGE: Key[TensorRange] = TENSOR.key("range", RANGE_CODEC)


class _Pass:
    """What one inference pass looks up, indexed once: the tensors' annotations
    (indexed again when their count changes), initializers, value infos and
    producers; each initializer's content digest; and the tensors already named
    in a finding. The pass must not change the graph's nodes, initializers or
    value infos."""

    def __init__(self, model: ModelWrapper) -> None:
        graph = model.graph
        self.model = model
        self.annotations: dict[str, TensorAnnotation] = {}
        self.count = -1
        self.initializers = {item.name: item for item in graph.initializer}
        self.infos = {item.name: item for item in [*graph.input, *graph.output, *graph.value_info]}
        self.producers = {name: node for node in graph.node for name in node.output if name}
        self.digests: dict[str, bytes] = {}
        self.named: set[str] = set()

    def annotation(self, tensor: str) -> Optional[TensorAnnotation]:
        annotations = self.model.graph.quantization_annotation
        if len(annotations) != self.count:
            self.annotations = {item.tensor_name: item for item in annotations}
            self.count = len(annotations)
        return self.annotations.get(tensor)


_PASS: ContextVar[Optional[_Pass]] = ContextVar("qonnx_ranges_pass", default=None)


@contextmanager
def one_pass(model: ModelWrapper) -> Iterator[None]:
    """Index ``model`` for one inference pass (``InferDataTypes`` runs each pass in
    one): lookups are by dictionary instead of linear scans, each initializer is
    digested once, and a contradiction is named once per tensor. The pass must
    not change the graph's nodes, initializers or value infos."""
    token = _PASS.set(_Pass(model))
    try:
        yield
    finally:
        _PASS.reset(token)


def _pass(model: ModelWrapper) -> Optional[_Pass]:
    current = _PASS.get()
    return current if current is not None and current.model is model else None


def annotation(model: ModelWrapper, tensor: str) -> Optional[TensorAnnotation]:
    """``tensor``'s quantization annotation, by the pass's index inside ``one_pass``."""
    index = _pass(model)
    if index is None:
        return get_by_name(model.graph.quantization_annotation, tensor, "tensor_name")
    return index.annotation(tensor)


def initializer(model: ModelWrapper, tensor: str) -> Optional[TensorProto]:
    """``tensor``'s initializer proto, if it is one, by the pass's index inside ``one_pass``."""
    index = _pass(model)
    if index is None:
        return get_by_name(model.graph.initializer, tensor)
    return index.initializers.get(tensor)


def digest(model: ModelWrapper, tensor: str) -> Optional[bytes]:
    """A digest of ``tensor``'s values (element type, shape and data) if it is an
    initializer, else None; computed once per pass inside ``one_pass``."""
    index = _pass(model)
    if index is not None and tensor in index.digests:
        return index.digests[tensor]
    proto = initializer(model, tensor)
    if proto is None:
        return None
    data = proto.raw_data if proto.raw_data else numpy_helper.to_array(proto).tobytes()
    hashed = hashlib.sha256(b"%d:%s:" % (proto.data_type, ",".join(map(str, proto.dims)).encode()))
    hashed.update(data)
    found = hashed.digest()
    if index is not None:
        index.digests[tensor] = found
    return found


def datatype(model: ModelWrapper, tensor: str) -> BaseDataType:
    """``model.get_tensor_datatype(tensor)``, by the pass's index inside ``one_pass``."""
    found = annotation(model, tensor)
    if found is not None:
        for entry in found.quant_parameter_tensor_names:
            if entry.key == "finn_datatype":
                return DataType[entry.value]
    return model.get_tensor_datatype(tensor)


def stated(model: ModelWrapper, tensor: str) -> Optional[TensorRange]:
    """The range stored on ``tensor``, None if it stores none. MetadataError when
    it is stored malformed."""
    found = annotation(model, tensor)
    if found is None:
        return None
    return read_namespace(found.quant_parameter_tensor_names, TENSOR, tensor=True).get(RANGE.name)


def tensor_range(model: ModelWrapper, tensor: str) -> TensorRange:
    """The range ``tensor`` states, or its datatype's full range; MetadataError for
    a stored range its datatype does not hold. An initializer's is not read here
    (``values_range``)."""
    found = stated(model, tensor)
    stated_type = datatype(model, tensor)
    if found is None:
        return full(stated_type)
    if not holds(stated_type, found):
        raise MetadataError(
            f"{RANGE.entry} of tensor {tensor!r}: stored {found} is not held by its datatype "
            f"{stated_type.get_canonical_name()}"
        )
    return found


def _writer(model: ModelWrapper, tensor: str) -> str:
    index = _pass(model)
    node = index.producers.get(tensor) if index is not None else model.find_producer(tensor)
    if node is None:
        return "its writer"
    return f"{node.name} ({node.op_type})" if node.name else node.op_type


def contradiction(model: ModelWrapper, tensor: str, value: TensorRange, stated_type: BaseDataType) -> None:
    """Name a writer's range its own datatype does not hold, with a
    RangeContradiction warning: once per tensor inside ``one_pass``, else each time."""
    index = _pass(model)
    if index is not None:
        if tensor in index.named:
            return
        index.named.add(tensor)
    warnings.warn(
        f"{RANGE.entry} of tensor {tensor!r} not stated: {_writer(model, tensor)} states {value}, which its "
        f"datatype {stated_type.get_canonical_name()} does not hold",
        RangeContradiction,
        stacklevel=4,
    )


def write(
    model: ModelWrapper,
    tensor: str,
    props: MutableSequence[StringStringEntryProto],
    stated_type: BaseDataType,
    value: Optional[TensorRange],
) -> None:
    """Store ``value`` as the range of ``tensor`` (whose annotation entries are
    ``props``) beside its datatype ``stated_type``: none for None or the
    datatype's full range; none, named, for one the datatype does not hold.
    ``ModelWrapper.set_tensor_datatype`` calls it. MetadataError for a value that
    is not a TensorRange."""
    if value is not None:
        RANGE.encode(value)
    if value is not None and value != full(stated_type) and not holds(stated_type, value):
        contradiction(model, tensor, value, stated_type)
        value = None
    if value is None or value == full(stated_type):
        if any(prop.key.startswith(TENSOR.name + "/") for prop in props):
            clear_namespace(props, TENSOR)
        return
    text = _encode(value)
    for prop in props:
        if prop.key == RANGE.entry:
            if prop.value != text:
                prop.value = text
            return
    clear_namespace(props, TENSOR)
    write_namespace(props, RANGE, value, tensor=True)


__all__ = [
    "RANGE",
    "RANGE_CODEC",
    "TENSOR",
    "annotation",
    "contradiction",
    "datatype",
    "digest",
    "initializer",
    "one_pass",
    "stated",
    "tensor_range",
    "write",
]
