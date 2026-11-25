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

"""A tensor's range on a model: where it is stored, how it is read and written. The
range itself is ``qonnx.core.ranges``.

A range is stored in the namespace ``qonnx.tensor`` (``qonnx.core.metadata``: it
follows its tensor into a subgraph cut from the graph), key ``range``, in the tensor's
``quantization_annotation`` entry beside ``finn_datatype``, so ``rename_tensor``
renames it and ``RemoveUnusedTensors`` removes it with its tensor. Its text is
canonical JSON, ``{"nan":false,"range":[least,greatest]}``, each end an int,
``"m*2^-k"`` (m odd, k >= 1), ``"inf"`` or ``"-inf"``.

A range is one fact with its datatype: ``ModelWrapper.set_tensor_datatype`` writes
both, and a datatype written without a range clears the range. Only a range that says
more than the datatype is stored; the datatype's full range (``ranges.full``) is not.
Nothing coerces: stored text in any other form, or a stored range its datatype does
not hold, is a ``MetadataError`` naming the entry and the text. A writer stating a
range its own datatype does not hold contradicts itself: the range is not stored, and
a ``RangeContradiction`` warning names the tensor and its writer (once per tensor in
an inference pass, ``one_pass``).
"""

from __future__ import annotations

import json
import math
import re
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from fractions import Fraction
from onnx import TensorAnnotation
from typing import TYPE_CHECKING, Iterator, Optional, Union

from qonnx.core import metadata
from qonnx.core.datatype import BaseDataType
from qonnx.core.metadata import Codec, Key, MetadataError, Namespace
from qonnx.core.ranges import Endpoint, Range, full, holds
from qonnx.util.basic import get_by_name

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

__all__ = [
    "RANGE",
    "RangeContradiction",
    "TENSOR",
    "element_type",
    "inferring",
    "one_pass",
    "prepare",
    "stated",
    "tensor_range",
    "write",
]


class RangeContradiction(UserWarning):
    """A writer stated a range its own datatype does not hold; the range is not
    stored. The message names the tensor and its writer."""


def _text(value: Endpoint) -> Union[int, str]:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return "inf" if value > 0 else "-inf"
    return f"{value.numerator}*2^-{value.denominator.bit_length() - 1}"


_DYADIC = re.compile(r"(-?[1-9][0-9]*)\*2\^-([1-9][0-9]*)")


def _parse(item: object) -> Endpoint:
    if isinstance(item, int) and not isinstance(item, bool):
        return item
    if item == "inf":
        return math.inf
    if item == "-inf":
        return -math.inf
    found = _DYADIC.fullmatch(item) if isinstance(item, str) else None
    if found is None or int(found.group(1)) % 2 == 0:
        raise ValueError(item)
    return Fraction(int(found.group(1)), 2 ** int(found.group(2)))


def _encode(value: Range) -> str:
    return json.dumps(
        {"nan": value.nan, "range": [_text(value.least), _text(value.greatest)]},
        sort_keys=True,
        separators=(",", ":"),
    )


def _range_text(value: Range) -> str:
    """A range as its stored ends show it, in a diagnostic."""
    ends = f"[{_text(value.least)}, {_text(value.greatest)}]"
    return ends + (", or NaN" if value.nan else "")


def _decode(text: str) -> Range:
    def refuse(constant: str) -> None:
        raise ValueError(constant)

    data = json.loads(text, parse_constant=refuse)
    if not isinstance(data, dict) or set(data) != {"nan", "range"} or not isinstance(data["nan"], bool):
        raise ValueError(text)
    pair = data["range"]
    if not isinstance(pair, list) or len(pair) != 2:
        raise ValueError(text)
    return Range(_parse(pair[0]), _parse(pair[1]), data["nan"])


_CODEC: Codec[Range] = Codec(
    (
        'a range, {"nan": a bool, "range": [least, greatest]}, each end an int, "m*2^-k" (m odd, k >= 1), '
        '"inf" or "-inf", least <= greatest'
    ),
    lambda value: isinstance(value, Range),
    _encode,
    _decode,
)

TENSOR = Namespace("qonnx.tensor", version=1, follow=True)
RANGE: Key[Range] = TENSOR.key("range", _CODEC)


class _Pass:
    """One inference pass over a model: the tensors whose contradiction it named."""

    __slots__ = ("model", "named")

    def __init__(self, model: ModelWrapper) -> None:
        self.model = model
        self.named: set[str] = set()


_PASSES: ContextVar[tuple[_Pass, ...]] = ContextVar("qonnx_tensor_range_passes", default=())


def _pass(model: ModelWrapper) -> Optional[_Pass]:
    for found in _PASSES.get():
        if found.model is model:
            return found
    return None


@contextmanager
def one_pass(model: ModelWrapper) -> Iterator[None]:
    """One inference pass over ``model``: inside it, a writer's contradiction
    (``RangeContradiction``) is named once per tensor. Entered again inside itself for
    the same model, it is the same pass. It holds no index of the model: every read
    inside it reads the model as it is.

    It is also the signal for exact ranges (``inferring``): inside it, inference
    (``InferDataTypes(exact_ranges=True)`` opens one) and a custom op's
    ``infer_node_datatype`` state each output's range with its datatype. A hook may
    enter it around a node's datatype step for the same."""
    if _pass(model) is not None:
        yield
        return
    token = _PASSES.set(_PASSES.get() + (_Pass(model),))
    try:
        yield
    finally:
        _PASSES.reset(token)


def inferring(model: ModelWrapper) -> bool:
    """Whether an inference pass over ``model`` (``one_pass``) is open: where a writer
    states ranges with its datatypes. Outside one, the rules before exact ranges
    apply, and a custom op states no range."""
    return _pass(model) is not None


def element_type(model: ModelWrapper, tensor: str) -> Optional[int]:
    """``tensor``'s ONNX element type (``TensorProto.FLOAT``, ...), the one its values
    are held and computed in: its value info's, else its initializer's; None where
    neither states one."""
    info = model.get_tensor_valueinfo(tensor)
    if info is not None and info.type.tensor_type.elem_type:
        return int(info.type.tensor_type.elem_type)
    proto = get_by_name(model.graph.initializer, tensor)
    return int(proto.data_type) if proto is not None and proto.data_type else None


def _annotation(model: ModelWrapper, tensor: str) -> Optional[TensorAnnotation]:
    return get_by_name(model.graph.quantization_annotation, tensor, "tensor_name")


def stated(model: ModelWrapper, tensor: str) -> Optional[Range]:
    """The range stored on ``tensor``, None if it stores none. MetadataError when it
    is stored malformed."""
    found = _annotation(model, tensor)
    if found is None:
        return None
    return metadata.read(found.quant_parameter_tensor_names, TENSOR, tensor=True).get(RANGE.name)


def tensor_range(model: ModelWrapper, tensor: str, *, numeric_text: bool = False) -> Range:
    """The range ``tensor`` states, or its datatype's full range. MetadataError for a
    stored range that is malformed or that its datatype does not hold; the message
    shows the range as stored (``"3*2^-3"``), or as ``qonnx.core.ranges`` does
    (``3/8``) with ``numeric_text``. An initializer's values are not read here
    (``ModelWrapper.get_tensor_range``)."""
    found = stated(model, tensor)
    stated_type = model.get_tensor_datatype(tensor)
    if found is None:
        return full(stated_type)
    if not holds(stated_type, found):
        shown = str(found) if numeric_text else _range_text(found)
        raise MetadataError(
            f"{RANGE.entry} of tensor {tensor!r}: stored {shown} is not held by its datatype {stated_type.name}"
        )
    return found


def _writer(model: ModelWrapper, tensor: str) -> str:
    node = model.find_producer(tensor)
    if node is None:
        return "its writer"
    return f"{node.name} ({node.op_type})" if node.name else node.op_type


def _contradiction(model: ModelWrapper, tensor: str, value: Range, stated_type: BaseDataType) -> None:
    """Name a writer's range its own datatype does not hold, with a
    RangeContradiction warning: once per tensor inside ``one_pass``, else each time."""
    found = _pass(model)
    if found is not None and tensor in found.named:
        return
    warnings.warn(
        f"{RANGE.entry} of tensor {tensor!r} not stated: {_writer(model, tensor)} states {_range_text(value)}, "
        f"which its datatype {stated_type.name} does not hold",
        RangeContradiction,
        stacklevel=4,
    )
    # named once it was named: a warning raised as an error names it again on a retry
    if found is not None:
        found.named.add(tensor)


def prepare(model: ModelWrapper, tensor: str, datatype: BaseDataType, value: Range) -> Optional[Range]:
    """The range to store for ``tensor`` with ``datatype``, checked before either is
    written: ``value``; None for the datatype's full range; None, and a
    ``RangeContradiction`` naming it, for one the datatype does not hold.
    MetadataError for a value that is not a Range. It writes nothing, so a
    refusal, or the warning raised as an error, leaves the model as it was."""
    RANGE.encode(value)
    if value == full(datatype):
        return None
    if not holds(datatype, value):
        _contradiction(model, tensor, value, datatype)
        return None
    return value


def write(model: ModelWrapper, tensor: str, value: Optional[Range]) -> None:
    """Store ``value`` as the range of ``tensor``, beside the datatype it states:
    nothing for None or the datatype's full range (a range stored before is
    cleared); nothing, and a ``RangeContradiction`` naming it, for one the datatype
    does not hold. ``ModelWrapper.set_tensor_datatype`` calls it after writing the
    datatype, with a range ``prepare`` checked before. MetadataError for a value that is not a Range; ValueError for a range
    on a tensor that states no datatype."""
    found = _annotation(model, tensor)
    if value is not None:
        text = RANGE.encode(value)
        if not model.has_tensor_datatype(tensor):
            raise ValueError(f"{RANGE.entry} of tensor {tensor!r}: a range is stated with a datatype; it states none")
        stated_type = model.get_tensor_datatype(tensor)
        if value == full(stated_type):
            value = None
        elif not holds(stated_type, value):
            _contradiction(model, tensor, value, stated_type)
            value = None
    if found is None:
        return
    props = found.quant_parameter_tensor_names
    if value is None:
        metadata.clear(props, TENSOR)
        return
    try:
        current = metadata.read(props, TENSOR, tensor=True)
    except MetadataError:
        current = {}
    if RANGE.name in current:
        for prop in props:
            if prop.key == RANGE.entry and prop.value != text:
                prop.value = text
        return
    metadata.clear(props, TENSOR)
    metadata.write(props, RANGE, value, tensor=True)
