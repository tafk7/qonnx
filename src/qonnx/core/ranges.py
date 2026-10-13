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

"""A tensor's range: the values it may hold, as part of its datatype fact.

A tensor's datatype (``finn_datatype``) says how its values are represented; its
range narrows that to the values it may actually hold. The two are one fact,
written together by ``ModelWrapper.set_tensor_datatype(tensor, datatype,
range)``: setting a datatype without a range clears the range in the same call,
so a range never outlives the datatype it was stated with. A tensor that states
no range holds its datatype's full range (``full``); ``ModelWrapper.
get_tensor_range`` reads either.

A range (``TensorRange``) has two endpoints and a flag: whether the tensor may
hold NaN. Each endpoint is an exact dyadic number, m * 2^e (an int, the case
e = 0, or a ``fractions.Fraction`` whose denominator is a power of two: every
float, fixed-point or scaled-integer value is one exactly), or an infinity
(``math.inf``, ``-math.inf``). Integers are exact at any width.

It is stored in the namespace ``qonnx.tensor`` (``qonnx.core.metadata``; it
follows its tensor into a subgraph cut from the graph, and the graph keeps its
own copy), key ``range``, in the tensor's ``quantization_annotation`` entry
beside ``finn_datatype`` (so ``rename_tensor`` renames it and
``RemoveUnusedTensors`` removes it with its tensor). The stored text is
canonical JSON, ``{"nan":false,"range":[least,greatest]}``, each endpoint a
JSON integer for an integer, the string ``"m*2^-k"`` for a non-integer (m odd,
k >= 1: ``"3*2^-2"`` is 0.75), or ``"inf"`` or ``"-inf"``. Only a range that
says more than the datatype is stored: the datatype's full range is not.
Nothing coerces: stored text in any other form, or a stored range its datatype
does not hold, is a ``MetadataError`` naming the entry and the text.

A writer stating a range its own datatype does not hold contradicts itself: the
range is not stored, and a ``RangeContradiction`` warning names the tensor and
its writer (once per tensor in an inference pass, ``one_pass``).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from fractions import Fraction
from onnx import TensorAnnotation, TensorProto, numpy_helper
from typing import TYPE_CHECKING, Iterator, Optional, Union

from qonnx.core.datatype import BaseDataType, DataType, FixedPointType, ScaledIntType
from qonnx.core.metadata import Codec, Key, MetadataError, Namespace
from qonnx.core.metadata import clear as clear_namespace
from qonnx.core.metadata import read as read_namespace
from qonnx.core.metadata import write as write_namespace
from qonnx.util.basic import get_by_name

if TYPE_CHECKING:
    import numpy as np
    from onnx import StringStringEntryProto
    from typing import MutableSequence

    from qonnx.core.modelwrapper import ModelWrapper

# An endpoint: an int, a Fraction with a power-of-two denominator (not 1), or an infinity.
Endpoint = Union[int, Fraction, float]


class RangeContradiction(UserWarning):
    """A writer stated a range its own datatype does not hold; the range is not
    stored. The message names the tensor and its writer."""


def endpoint(value: Union[int, float, Fraction]) -> Endpoint:
    """``value`` as a canonical endpoint: an int if it is an integer, else a dyadic
    Fraction, else an infinity. ValueError for NaN or a non-dyadic Fraction."""
    if isinstance(value, bool):
        raise ValueError(f"an endpoint is a number, not {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("an endpoint is not NaN (a range states NaN by its flag)")
        if math.isinf(value):
            return value
        value = Fraction(value)
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return value.numerator
        if value.denominator & (value.denominator - 1):
            raise ValueError(f"an endpoint is dyadic (m * 2^e), not {value}")
        return value
    raise ValueError(f"an endpoint is an int, a dyadic Fraction or an infinity, not {value!r}")


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


@dataclass(frozen=True)
class TensorRange:
    """The values a tensor may hold: every number in [least, greatest], and NaN
    if ``nan``. Endpoints are canonical (``endpoint``) and least <= greatest."""

    least: Endpoint
    greatest: Endpoint
    nan: bool = False

    def __post_init__(self) -> None:
        least, greatest = endpoint(self.least), endpoint(self.greatest)
        if not least <= greatest:
            raise ValueError(f"a range's least end exceeds its greatest: [{least}, {greatest}]")
        if not isinstance(self.nan, bool):
            raise ValueError(f"a range's NaN flag is a bool, not {self.nan!r}")
        object.__setattr__(self, "least", least)
        object.__setattr__(self, "greatest", greatest)

    @property
    def span(self) -> tuple[Endpoint, Endpoint]:
        return self.least, self.greatest

    def is_integer(self) -> bool:
        """Whether every value it admits is an integer: integer endpoints, no NaN."""
        return isinstance(self.least, int) and isinstance(self.greatest, int) and not self.nan

    def __str__(self) -> str:
        ends = f"[{_text(self.least)}, {_text(self.greatest)}]"
        return ends + (", or NaN" if self.nan else "")


def unbounded(nan: bool) -> TensorRange:
    return TensorRange(-math.inf, math.inf, nan)


def hull(*ranges: TensorRange) -> TensorRange:
    """The least range holding each of ``ranges``."""
    return TensorRange(
        min(item.least for item in ranges), max(item.greatest for item in ranges), any(item.nan for item in ranges)
    )


def _encode(value: TensorRange) -> str:
    return json.dumps(
        {"nan": value.nan, "range": [_text(value.least), _text(value.greatest)]}, sort_keys=True, separators=(",", ":")
    )


def _decode(text: str) -> TensorRange:
    def refuse(constant: str) -> None:
        raise ValueError(constant)

    data = json.loads(text, parse_constant=refuse)
    if not isinstance(data, dict) or set(data) != {"nan", "range"} or not isinstance(data["nan"], bool):
        raise ValueError(text)
    pair = data["range"]
    if not isinstance(pair, list) or len(pair) != 2:
        raise ValueError(text)
    return TensorRange(_parse(pair[0]), _parse(pair[1]), data["nan"])


RANGE_CODEC: Codec[TensorRange] = Codec(
    'a range, {"nan": a bool, "range": [least, greatest]}, each end an int, "m*2^-k" (m odd, k >= 1), '
    '"inf" or "-inf", least <= greatest',
    lambda value: isinstance(value, TensorRange),
    _encode,
    _decode,
)

TENSOR = Namespace("qonnx.tensor", version=1, follow=True)
RANGE: Key[TensorRange] = TENSOR.key("range", RANGE_CODEC)


def _is_float(datatype: BaseDataType) -> bool:
    return not (datatype.is_integer() or datatype.is_fixed_point() or isinstance(datatype, ScaledIntType))


def full(datatype: BaseDataType) -> TensorRange:
    """The range of a tensor that states none: an integer or fixed-point datatype's
    bounds; a scaled integer's unbounded (its scale is not part of it), without
    NaN; a float's unbounded, with NaN."""
    if datatype.is_integer() or isinstance(datatype, FixedPointType):
        return TensorRange(endpoint(datatype.min()), endpoint(datatype.max()))
    if isinstance(datatype, ScaledIntType):
        return unbounded(False)
    return unbounded(True)


def holds(datatype: BaseDataType, value: TensorRange) -> bool:
    """Whether every value ``value`` admits is one of ``datatype``'s: an integer
    datatype holds integer ranges within its bounds; a fixed-point one, finite
    ranges within its bounds; a scaled integer, any range without NaN; a float,
    any range."""
    if _is_float(datatype):
        return True
    if value.nan:
        return False
    if isinstance(datatype, ScaledIntType):
        return True
    if datatype.is_integer() and not value.is_integer():
        return False
    bounds = full(datatype)
    return bounds.least <= value.least and value.greatest <= bounds.greatest


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


def values_range(values: np.ndarray) -> TensorRange:
    """The exact range of an array's values: its least and greatest number (each
    exactly as stored), NaN if it holds one."""
    import numpy as np

    flat = np.asarray(values).reshape(-1)
    nan = bool(flat.dtype.kind == "f" and np.isnan(flat).any())
    numbers = flat[~np.isnan(flat)] if nan else flat
    if numbers.size == 0:
        return TensorRange(0, 0, nan) if nan else TensorRange(0, 0)
    least, greatest = numbers.min(), numbers.max()
    if flat.dtype.kind in "iub":
        return TensorRange(int(least), int(greatest), nan)
    return TensorRange(endpoint(float(least)), endpoint(float(greatest)), nan)


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
    "Endpoint",
    "RANGE",
    "RANGE_CODEC",
    "RangeContradiction",
    "TENSOR",
    "TensorRange",
    "annotation",
    "contradiction",
    "datatype",
    "digest",
    "endpoint",
    "full",
    "holds",
    "hull",
    "initializer",
    "one_pass",
    "stated",
    "tensor_range",
    "unbounded",
    "values_range",
    "write",
]
