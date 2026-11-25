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

"""A tensor's facts read in one call, and the named ones written in one.

A tensor's facts are its shape and container (ONNX element type), from its value
info, else its initializer; its datatype (``element``), range and layout, from its
quantization annotation. ``read`` answers all of them, each None where the model
states none, validating what it reads; ``state`` writes only the facts it is given,
all or none of them; ``unstate`` removes one; ``forget`` removes a tensor's value
info and annotation. A refusal is a ``FactRefused``, which leaves the model as it
was.

``state`` and ModelWrapper's setters (``set_tensor_datatype``, ``set_tensor_shape``,
``set_tensor_layout``) share the writes below (``write_entry``, ``place_valueinfo``)
and keep two contracts. ``state`` admits only non-negative integer extents and a
shape and container its initializer agrees with, refuses the empty name, and
keeps the rest of a value info (its doc string and denotations): a graph input's or
output's updated in place, so a message the caller holds stays the graph's, an
intermediate's removed and appended after the others. The setters keep theirs: a
new value info replacing the old, symbolic and unknown extents, and any name."""

from __future__ import annotations

import ast
import copy
import numpy as np
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from onnx import StringStringEntryProto, TensorAnnotation, TensorProto, ValueInfoProto
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Optional, Union, cast

from qonnx.core import metadata
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.metadata import MetadataError
from qonnx.core.ranges import Range, full, holds
from qonnx.core.tensor_ranges import RANGE, TENSOR, _contradiction, stated
from qonnx.util.basic import get_by_name

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

__all__ = [
    "Fact",
    "FactRefused",
    "NonTensor",
    "NonTensorKind",
    "TensorFacts",
    "forget",
    "place_valueinfo",
    "read",
    "state",
    "unstate",
    "write_entry",
]

Shape = tuple[int, ...]
Layout = tuple[str, ...]
Fact = Literal["shape", "element", "range", "layout", "container"]
NonTensorKind = Literal["sequence_type", "map_type", "optional_type", "sparse_tensor_type", "opaque_type"]

_FACTS = ("shape", "element", "range", "layout", "container")
_DATATYPE = "finn_datatype"
_LAYOUT = "tensor_layout"


@dataclass(frozen=True)
class TensorFacts:
    """A tensor's facts, each None where the model states none: its shape (integer
    extents), datatype, range, layout (axis labels) and container (ONNX element type)."""

    shape: Optional[Shape]
    element: Optional[BaseDataType]
    range: Optional[Range]
    layout: Optional[Layout]
    container: Optional[int]


@dataclass(frozen=True)
class NonTensor:
    """A name whose value info declares no tensor: a sequence, map, optional, sparse
    tensor or opaque value."""

    kind: NonTensorKind


class FactRefused(ValueError):
    """A fact operation's refusal: the operation (``owner``), the tensor, a ``code``
    ("fact-invalid", "fact-conflict", "fact-non-tensor", "fact-source",
    "container-initializer"), the message, and details (such as the fact)."""

    def __init__(self, owner: str, tensor: str, code: str, message: str, details: dict[str, Any]) -> None:
        super().__init__(f"{owner} of tensor {tensor!r}: {message}")
        self.owner = owner
        self.tensor = tensor
        self.code = code
        self.message = message
        self.details = details


class _Omitted:
    """A fact not given to ``state``."""


_OMITTED = _Omitted()


def _refuse(owner: str, name: str, code: str, message: str, **details: Any) -> NoReturn:
    raise FactRefused(owner, name, code, message, details)


@contextmanager
def _refusing(owner: str, name: str) -> Iterator[None]:
    """Refuse the empty name, and translate what the reads below raise into a
    FactRefused with its text."""
    if not name:
        _refuse(owner, name, "fact-invalid", "an empty name identifies no tensor")
    try:
        yield
    except FactRefused:
        raise
    except KeyError as error:  # an unknown datatype name
        message = str(error.args[0]) if error.args else str(error)
        raise FactRefused(owner, name, "fact-invalid", message, {"source": "KeyError"}) from error
    except (MetadataError, ValueError) as error:
        raise FactRefused(owner, name, "fact-invalid", str(error), {"source": type(error).__name__}) from error


# -- the writes shared with the setters


def write_entry(model: ModelWrapper, name: str, key: str, value: Optional[str]) -> None:
    """Write ``value`` under ``key`` in the annotation of the tensor ``name``, or
    remove the entry for None: the entry is updated in place, or appended, and the
    annotation is appended to the graph's when the tensor has none (none is made to
    remove an entry). Upstream's setter writes, which state shares."""
    annotations = model.graph.quantization_annotation
    annotation = get_by_name(annotations, name, "tensor_name")
    if annotation is None:
        if value is None:
            return
        annotation = TensorAnnotation(tensor_name=name)
        _write_entry(annotation, key, value)
        annotations.append(annotation)
        return
    _write_entry(annotation, key, value)


def _write_entry(annotation: TensorAnnotation, key: str, value: Optional[str]) -> None:
    props = annotation.quant_parameter_tensor_names
    entry = get_by_name(props, key, "key")
    if entry is None:
        if value is not None:
            props.append(StringStringEntryProto(key=key, value=value))
    elif value is None:
        props.remove(entry)
    else:
        entry.value = value


def place_valueinfo(model: ModelWrapper, info: ValueInfoProto) -> None:
    """Put ``info`` in the place of the value info of its name: at its index among
    the graph's outputs, else its inputs; else last among the value infos, the old
    one removed. Upstream's set_tensor_shape placement, which state shares."""
    graph = model.graph
    name = info.name
    target = graph.value_info
    index = None
    if get_by_name(graph.input, name) is not None:
        target = graph.input
        index = [x.name for x in graph.input].index(name)
    if get_by_name(graph.output, name) is not None:
        target = graph.output
        index = [x.name for x in graph.output].index(name)
    old = get_by_name(target, name)
    if old is not None:
        target.remove(old)
    if index is None:
        target.append(info)
    else:
        target.insert(index, info)


# -- reading


def _info(model: ModelWrapper, name: str, owner: str) -> Optional[ValueInfoProto]:
    graph = model.graph
    found = [x for x in (*graph.input, *graph.output, *graph.value_info) if x.name == name]
    if len(found) > 1:
        _refuse(owner, name, "fact-conflict", "multiple ValueInfo declarations")
    return found[0] if found else None


def _source(model: ModelWrapper, name: str, owner: str) -> Optional[TensorProto]:
    found = [x for x in model.graph.initializer if x.name == name]
    if len(found) > 1:
        _refuse(owner, name, "fact-conflict", "multiple initializer declarations")
    return found[0] if found else None


def _kind(model: ModelWrapper, name: str, info: Optional[ValueInfoProto]) -> Optional[str]:
    kind = info.type.WhichOneof("value") if info is not None else None
    sparse = any(x.values.name == name for x in model.graph.sparse_initializer)
    return "sparse_tensor_type" if sparse and kind in (None, "tensor_type", "sparse_tensor_type") else kind


def _annotation(model: ModelWrapper, name: str, owner: str) -> Optional[TensorAnnotation]:
    found = [x for x in model.graph.quantization_annotation if x.tensor_name == name]
    if len(found) > 1:
        _refuse(owner, name, "fact-conflict", "multiple tensor annotations")
    return found[0] if found else None


def _entry_value(annotation: Optional[TensorAnnotation], key: str) -> Optional[str]:
    if annotation is None:
        return None
    entries = [x for x in annotation.quant_parameter_tensor_names if x.key == key]
    if len(entries) > 1:
        raise MetadataError(f"duplicate {key} annotation")
    return entries[0].value if entries else None


def _element(annotation: Optional[TensorAnnotation]) -> Optional[BaseDataType]:
    text = _entry_value(annotation, _DATATYPE)
    return None if text is None else DataType[text]


def _layout(model: ModelWrapper, name: str, annotation: Optional[TensorAnnotation]) -> Optional[Layout]:
    text = _entry_value(annotation, _LAYOUT)
    if text is None:
        return None
    try:
        labels = ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        labels = None
    if not isinstance(labels, list) or any(not isinstance(x, str) for x in labels):
        _refuse("facts", name, "fact-invalid", "layout must contain axis labels", fact="layout")
    return tuple(labels)


def read(model: ModelWrapper, name: str) -> Union[TensorFacts, NonTensor]:
    """The facts the model states of the tensor ``name`` (ModelWrapper.facts), or the
    kind of a value that is not a tensor. It changes nothing. FactRefused where they
    contradict each other (a shape or container its initializer does not have, a
    range its datatype does not hold), where one is stored more than once, or where
    one is malformed. A shape with a symbolic or unknown extent is None."""
    with _refusing("facts", name):
        info = _info(model, name, "facts")
        kind = _kind(model, name, info)
        if kind not in (None, "tensor_type"):
            return NonTensor(cast(NonTensorKind, kind))
        source = _source(model, name, "facts")
        tensor = info.type.tensor_type if info is not None else None
        shape: Optional[Shape] = None
        container: Optional[int] = None
        if tensor is not None and tensor.HasField("shape"):
            dimensions = tensor.shape.dim
            if all(x.WhichOneof("value") == "dim_value" for x in dimensions):
                shape = tuple(x.dim_value for x in dimensions)
                if any(x < 0 for x in shape):
                    _refuse("facts", name, "fact-invalid", "negative shape extent", fact="shape")
        elif source is not None:
            shape = tuple(source.dims)
        if tensor is not None and tensor.HasField("elem_type"):
            container = tensor.elem_type
        elif source is not None and source.HasField("data_type"):
            container = source.data_type
        if source is not None:
            if shape is not None and shape != tuple(source.dims):
                _refuse("facts", name, "fact-conflict", "shape contradicts initializer", fact="shape")
            if container is not None and container != source.data_type:
                _refuse("facts", name, "container-initializer", "container contradicts initializer", fact="container")
        annotation = _annotation(model, name, "facts")
        if annotation is not None:
            keys = [x.key for x in annotation.quant_parameter_tensor_names if x.key in (_DATATYPE, _LAYOUT, RANGE.entry)]
            if len(keys) != len(set(keys)):
                _refuse("facts", name, "fact-conflict", "duplicate fact annotation")
        element = _element(annotation)
        value_range = stated(model, name)
        if value_range is not None and element is not None and not holds(element, value_range):
            _refuse("facts", name, "fact-conflict", "range is not held by element", fact="range")
        return TensorFacts(shape, element, value_range, _layout(model, name, annotation), container)


# -- writing


def _structure(
    model: ModelWrapper, name: str, owner: str, patch: dict[str, Any], source: Optional[TensorProto]
) -> Optional[ValueInfoProto]:
    """The tensor's value info with the patch's shape and container, its other fields
    kept; None where nothing is to be written."""
    old = _info(model, name, owner)
    kind = _kind(model, name, old)
    if kind not in (None, "tensor_type"):
        _refuse(owner, name, "fact-non-tensor", "structural write to a non-tensor name", kind=kind)
    info = copy.deepcopy(old) if old is not None else ValueInfoProto(name=name)
    tensor = info.type.tensor_type
    if "container" in patch:
        container = patch["container"]
        if container is None:
            # absent, which is not UNDEFINED (0): a read falls back to the initializer's
            tensor.ClearField("elem_type")
        elif type(container) is not int or container not in TensorProto.DataType.values():
            _refuse(owner, name, "fact-invalid", "container must be an explicit ONNX element code", fact="container")
        else:
            tensor.elem_type = container
    elif old is None and patch.get("shape") is not None:
        tensor.elem_type = TensorProto.FLOAT
    if "shape" in patch:
        shape = patch["shape"]
        if shape is None:
            tensor.ClearField("shape")
        else:
            if not isinstance(shape, (tuple, list)) or any(
                isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 0 for x in shape
            ):
                _refuse(owner, name, "fact-invalid", "shape must contain nonnegative integer extents", fact="shape")
            denotations = [x.denotation for x in tensor.shape.dim]
            tensor.shape.ClearField("dim")
            tensor.shape.SetInParent()
            for i, extent in enumerate(shape):
                dim = tensor.shape.dim.add(dim_value=int(extent))
                if i < len(denotations) and denotations[i]:
                    dim.denotation = denotations[i]
            if source is not None and tuple(shape) != tuple(source.dims):
                _refuse(owner, name, "fact-conflict", "shape contradicts initializer", fact="shape")
    if source is not None and tensor.HasField("elem_type") and tensor.elem_type != source.data_type:
        _refuse(owner, name, "container-initializer", "container contradicts initializer", fact="container")
    if old is None and patch.get("shape") is None and patch.get("container") is None:
        return None
    return info


def _annotated(
    model: ModelWrapper, name: str, owner: str, patch: dict[str, Any]
) -> tuple[Optional[TensorAnnotation], Optional[tuple[Range, BaseDataType]]]:
    """The tensor's annotation with the patch's datatype, range and layout, its other
    entries kept (None where nothing is to be written), and a range its datatype does
    not hold, to be named."""
    contradiction = None
    old = _annotation(model, name, owner)
    annotation = copy.deepcopy(old) if old is not None else TensorAnnotation(tensor_name=name)
    if "element" in patch:
        element = patch["element"]
        if element is not None and not isinstance(element, BaseDataType):
            _refuse(owner, name, "fact-invalid", "element must be a qonnx DataType", fact="element")
        _entry_value(annotation, _DATATYPE)  # a duplicate entry is refused
        _write_entry(annotation, _DATATYPE, None if element is None else element.name)
    else:
        element = _element(annotation) if patch.get("range") is not None else None
    if "layout" in patch:
        layout = patch["layout"]
        if layout is not None and (not isinstance(layout, (tuple, list)) or any(not isinstance(x, str) for x in layout)):
            _refuse(owner, name, "fact-invalid", "layout must contain axis labels", fact="layout")
        _entry_value(annotation, _LAYOUT)
        _write_entry(annotation, _LAYOUT, None if layout is None else str(list(layout)))
    props = annotation.quant_parameter_tensor_names
    if "range" in patch:
        value = patch["range"]
        if value is not None and not isinstance(value, Range):
            _refuse(owner, name, "fact-invalid", "range must be a qonnx.core.ranges.Range", fact="range")
        if value is not None:
            RANGE.encode(value)
            if element is not None and not holds(element, value):
                contradiction = (value, element)
                value = None
        if element is not None and value == full(element):
            value = None
        if value is None:
            if any(x.key.startswith(TENSOR.name + "/") for x in props):
                metadata.clear(props, TENSOR)
        else:
            text = RANGE.encode(value)
            old_range = next((x for x in props if x.key == RANGE.entry), None)
            if old_range is not None:
                old_range.value = text
            else:
                metadata.clear(props, TENSOR)
                metadata.write(props, RANGE, value, tensor=True)
    elif "element" in patch and element is not None:
        previous = stated(model, name)
        if previous is not None and not holds(element, previous):
            _refuse(owner, name, "fact-conflict", "element does not hold unnamed range", fact="range")
    if old is None and not props:
        return None, contradiction
    return annotation, contradiction


def _publish(model: ModelWrapper, name: str, info: Optional[ValueInfoProto], annotation: Optional[TensorAnnotation]) -> None:
    """Write what _structure and _annotated prepared: nothing of the model changes
    before both are ready. A graph input's or output's value info is updated in
    place, so a message the caller holds stays the graph's; an intermediate's is
    replaced, the old one removed and the new one appended (place_valueinfo)."""
    if info is not None:
        interface = next((x for x in (*model.graph.input, *model.graph.output) if x.name == name), None)
        if interface is not None:
            interface.CopyFrom(info)
        else:
            place_valueinfo(model, info)
    if annotation is not None:
        old = get_by_name(model.graph.quantization_annotation, name, "tensor_name")
        if old is None:
            model.graph.quantization_annotation.append(annotation)
        else:
            old.CopyFrom(annotation)


def _prepare(
    model: ModelWrapper, name: str, owner: str, patch: dict[str, Any]
) -> tuple[Optional[ValueInfoProto], Optional[TensorAnnotation], Optional[tuple[Range, BaseDataType]]]:
    info = None
    if "shape" in patch or "container" in patch:
        info = _structure(model, name, owner, patch, _source(model, name, owner))
    annotation, contradiction = None, None
    if any(x in patch for x in ("element", "range", "layout")):
        annotation, contradiction = _annotated(model, name, owner, patch)
    return info, annotation, contradiction


def state(model: ModelWrapper, name: str, **patch: Any) -> None:
    """State the facts given (``shape``, ``element``, ``range``, ``layout``,
    ``container``; None removes one) of the tensor ``name``, keeping the others: all
    of them, or none and a FactRefused. A range its datatype does not hold is not
    stored, and is named (RangeContradiction), as by set_tensor_datatype; a datatype
    that does not hold the range the tensor states is refused. An annotation-only
    write makes no value info. The five fields ``read`` answers for a tensor are
    admitted as they are.

    A graph input's or output's value info is updated in place: a message the
    caller holds stays the graph's and shows the new facts. An intermediate's is
    replaced, removed and appended after the others, as the setters place it.

    ``container=None`` removes the value info's element type, which is not
    UNDEFINED (0); a later read falls back to the initializer's. A graph input or
    output without an element type does not pass ONNX's checker."""
    with _refusing("state", name):
        unknown = sorted(set(patch) - set(_FACTS))
        if unknown:
            _refuse("state", name, "fact-invalid", f"unknown facts {unknown}")
        info, annotation, contradiction = _prepare(model, name, "state", patch)
        if contradiction is not None:
            _contradiction(model, name, *contradiction)
        _publish(model, name, info, annotation)


def unstate(model: ModelWrapper, name: str, fact: Fact) -> None:
    """Remove one fact of the tensor ``name``, keeping the others."""
    with _refusing("unstate", name):
        if fact not in _FACTS:
            _refuse("unstate", name, "fact-invalid", "unknown fact")
        info, annotation, _ = _prepare(model, name, "unstate", {fact: None})
        _publish(model, name, info, annotation)


def forget(model: ModelWrapper, name: str) -> None:
    """Remove the value info and the annotation of the tensor ``name``; its
    initializer stays. FactRefused for a graph input or output."""
    with _refusing("forget", name):
        if any(x.name == name for x in (*model.graph.input, *model.graph.output)):
            _refuse("forget", name, "fact-source", "forget would remove a graph interface member")
        for info in list(model.graph.value_info):
            if info.name == name:
                model.graph.value_info.remove(info)
        for annotation in list(model.graph.quantization_annotation):
            if annotation.tensor_name == name:
                model.graph.quantization_annotation.remove(annotation)
