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
# * Neither the name of Xilinx nor the names of its
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

"""Representation-neutral facts about the values held by an ONNX tensor.

This module answers exactly one question: *what values does this tensor
contain, and what is their exact identity?* It deliberately knows nothing
about how a consumer might store, pack, or implement those values -- that
interpretation belongs to the consuming framework, not to QONNX.

The public value is :class:`TensorValueSummary`: small, immutable, hashable
and deterministic. Two tensors with identical dtype, shape and bytes always
produce equal summaries, whatever their names are; any change to dtype, shape
or bytes moves the content digest.

The value is named for what it describes and not for where the array came
from, so the same summary serves an initializer, a folded constant, or any
in-memory array. The model-level entry points are named for what they do
consult: initializers.
"""

from __future__ import annotations

import hashlib
import math
import numpy as np
import numpy.typing as npt
import re
from dataclasses import dataclass
from onnx import numpy_helper
from typing import TYPE_CHECKING, Any, Union, cast

from qonnx.core.datatype import BaseDataType, BipolarType, DataType, IntType, TernaryType, resolve_datatype

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

# digest domain separator; bump when the digested preimage changes
_DIGEST_SCHEMA = b"qonnx.tensor_value_summary.v1"

# sha256, rendered as lowercase hex
_DIGEST_LENGTH = 64
_DIGEST_PATTERN = re.compile("[0-9a-f]{%d}" % _DIGEST_LENGTH)

# NumPy dtype kinds whose values QONNX can summarize soundly without further
# restriction: signed integer, unsigned integer, boolean. Floating values need
# an explicit width check because extrema are stored as Python ``float``.
_SUPPORTED_NON_FLOAT_KINDS = frozenset("iub")
_SUPPORTED_FLOAT_ITEM_SIZES = frozenset((2, 4, 8))


class UnsupportedTensorValueError(TypeError):
    """Raised for a tensor whose values QONNX cannot summarize soundly.

    This is distinct from the absence of a tensor: an absent initializer is
    reported as ``None``, never as an exception and never as a fabricated
    zero-valued summary."""


@dataclass(frozen=True)
class TensorValueSummary:
    """Deterministic, representation-neutral facts about one tensor's values.

    * ``content_digest``: hex digest over dtype, shape and contiguous bytes.
      It is bit-level identity, so ``-0.0`` and ``0.0`` are different content.
    * ``element_count``: number of elements; ``0`` for an empty tensor.
    * ``minimum`` / ``maximum``: smallest/largest observed value, or ``None``
      when no comparable value was observed (an empty tensor, or a floating
      tensor whose every element is NaN). Absent is not zero: it must not
      authorize a zero-range storage optimization.
    * ``is_integral``: every element is finite and has an integral value.
      ``False`` if any NaN or infinity is present. Vacuously ``True`` for an
      empty tensor, which
      :func:`smallest_lossless_integer_datatype` still refuses.
    * ``contains_zero``: at least one element is exactly zero (``-0.0``
      counts). A range alone cannot answer this, and datatypes with a
      non-contiguous domain need it: ``BIPOLAR`` admits ``{-1, +1}`` and
      ``TERNARY`` admits ``{-1, 0, +1}``, which share the range ``[-1, +1]``.

    NaN values are excluded from ``minimum``/``maximum`` so that equal content
    yields equal summaries; their presence is reported by ``is_integral``
    being ``False``. Infinities are ordinary observed values and do appear in
    ``minimum``/``maximum``.

    Instances are validated on construction: a summary is a claim about one
    real tensor, so a malformed digest, a negative count, an inverted range,
    or a membership flag inconsistent with the range is a ``ValueError`` and
    not a constructible value.
    """

    content_digest: str
    element_count: int
    minimum: Union[int, float, None]
    maximum: Union[int, float, None]
    is_integral: bool
    contains_zero: bool

    def __post_init__(self) -> None:
        if not _DIGEST_PATTERN.fullmatch(self.content_digest):
            raise ValueError(
                "content_digest must be %d lowercase hex characters, got %r" % (_DIGEST_LENGTH, self.content_digest)
            )
        if type(self.element_count) is not int or self.element_count < 0:
            raise ValueError("element_count must be a non-negative int, got %r" % (self.element_count,))
        for flag_name in ("is_integral", "contains_zero"):
            if type(getattr(self, flag_name)) is not bool:
                raise ValueError("%s must be a bool, got %r" % (flag_name, getattr(self, flag_name)))
        minimum, maximum = self.minimum, self.maximum
        if (minimum is None) != (maximum is None):
            raise ValueError("minimum and maximum must be both present or both absent")
        if minimum is None or maximum is None:
            if self.contains_zero:
                raise ValueError("contains_zero cannot be True without an observed range")
            # an empty tensor is vacuously integral; a non-empty one with no
            # comparable value is all-NaN, which is not integral
            if self.element_count == 0 and not self.is_integral:
                raise ValueError("an empty tensor is vacuously integral, so is_integral must be True")
            if self.element_count > 0 and self.is_integral:
                raise ValueError("a non-empty tensor with no observed range is all-NaN, so is_integral must be False")
            return
        for bound_name, bound in (("minimum", minimum), ("maximum", maximum)):
            if type(bound) not in (int, float):
                raise ValueError("%s must be an int or float, got %r" % (bound_name, bound))
            if isinstance(bound, float) and math.isnan(bound):
                # the factory excludes NaN from the range, so a NaN bound
                # contradicts the contract rather than reporting a fact
                raise ValueError("%s must not be NaN" % bound_name)
        if self.element_count == 0:
            raise ValueError("an empty tensor cannot have an observed range")
        if minimum > maximum:
            raise ValueError("minimum %r exceeds maximum %r" % (minimum, maximum))
        if self.is_integral:
            if not (math.isfinite(minimum) and math.isfinite(maximum)):
                raise ValueError("is_integral cannot be True for a non-finite range")
            for bound_name, bound in (("minimum", minimum), ("maximum", maximum)):
                # the extrema are themselves observed values
                if bound != int(bound):
                    raise ValueError("is_integral is True but %s %r is not integral" % (bound_name, bound))
        if minimum > 0 or maximum < 0:
            if self.contains_zero:
                raise ValueError("contains_zero is True but zero lies outside [%r, %r]" % (minimum, maximum))
        elif minimum == 0 or maximum == 0:
            # zero is an extremum, so it was necessarily observed
            if not self.contains_zero:
                raise ValueError("contains_zero must be True when zero is an observed extremum")


def is_summarizable_dtype(dtype: npt.DTypeLike) -> bool:
    """Returns whether values of this numpy dtype can be summarized soundly.

    Supported: signed and unsigned integers, IEEE floating point with 16, 32,
    or 64 bits, and booleans. A platform type such as ``longdouble`` is
    supported only when it is an alias for one of those effective widths.

    Unsupported: wider floating types whose extrema cannot be represented
    exactly by this summary's Python ``float`` fields, complex, string/object,
    and every sub-byte or custom-encoded numeric type that
    ``onnx.numpy_helper`` returns as a *view* dtype with fields (bfloat16, the
    float8 variants, the 4-bit types). Their raw bytes are bit patterns rather
    than numbers, so a range taken over them would be meaningless rather than
    merely imprecise.

    ``INT4``/``UINT4`` are excluded here even though QONNX has datatypes of
    those names: the exclusion is a property of the packed bytes numpy hands
    back, not of the datatype being inexpressible. Decoding those encodings
    into ordinary arrays would lift the exclusion without changing any value
    in this module's contract."""
    resolved = np.dtype(dtype)
    if resolved.fields is not None or resolved.subdtype is not None:
        return False
    if resolved.kind == "f":
        return resolved.itemsize in _SUPPORTED_FLOAT_ITEM_SIZES
    return resolved.kind in _SUPPORTED_NON_FLOAT_KINDS


def summarize_tensor_values(array: npt.NDArray[Any]) -> TensorValueSummary:
    """Summarizes the values of an in-memory tensor.

    Raises :class:`UnsupportedTensorValueError` if the array's dtype is not
    supported by :func:`is_summarizable_dtype`."""
    array = np.asarray(array)
    if not is_summarizable_dtype(array.dtype):
        raise UnsupportedTensorValueError(
            "Cannot summarize tensor values of dtype %s: only integer, "
            "16/32/64-bit IEEE floating point and boolean tensors are supported." % str(array.dtype)
        )
    contiguous = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(_DIGEST_SCHEMA)
    hasher.update(str(contiguous.dtype.str).encode("utf-8"))
    hasher.update(str(contiguous.shape).encode("utf-8"))
    hasher.update(contiguous.tobytes())
    element_count = int(contiguous.size)
    minimum, maximum, is_integral, contains_zero = _observed_values(contiguous)
    return TensorValueSummary(
        content_digest=hasher.hexdigest(),
        element_count=element_count,
        minimum=minimum,
        maximum=maximum,
        is_integral=is_integral,
        contains_zero=contains_zero,
    )


def _observed_values(
    array: npt.NDArray[Any],
) -> tuple[Union[int, float, None], Union[int, float, None], bool, bool]:
    """Returns (minimum, maximum, is_integral, contains_zero) for an array."""
    if array.size == 0:
        # empty is not zero: no observed value, and vacuously integral
        return (None, None, True, False)
    if array.dtype.kind == "f":
        finite = np.isfinite(array)
        not_nan = ~np.isnan(array)
        if not not_nan.any():
            # every element is NaN: nothing comparable was observed
            return (None, None, False, False)
        comparable = array[not_nan]
        is_integral = bool(finite.all()) and bool(np.all(comparable == np.trunc(comparable)))
        contains_zero = bool((comparable == 0).any())
        return (float(comparable.min()), float(comparable.max()), is_integral, contains_zero)
    # integer and boolean values are exact and always integral
    return (int(array.min()), int(array.max()), True, bool((array == 0).any()))


def initializer_value_summary(model: "ModelWrapper", tensor_name: str) -> TensorValueSummary | None:
    """Summarizes the initializer values of one tensor in a model.

    Only initializers are consulted. ``None`` is returned both when the named
    tensor has no initializer and when no such tensor exists; absence is
    reported as absence, never as a fabricated zero summary. In particular a
    tensor produced by a ``Constant`` node has a static value but no
    initializer, so it reports ``None`` until the graph has been through
    ``FoldConstants``.

    Raises :class:`UnsupportedTensorValueError` when an initializer is present
    but its values cannot be summarized soundly."""
    array = cast(Union[npt.NDArray[Any], None], model.get_initializer(tensor_name))
    if array is None:
        return None
    return summarize_tensor_values(array)


def initializer_value_summaries(model: "ModelWrapper") -> dict[str, TensorValueSummary]:
    """Analysis pass: summarizes every initializer in the model in one pass.

    Returns a dict mapping tensor name to :class:`TensorValueSummary`. Use it
    as ``model.analysis(initializer_value_summaries)``. Tensors without an
    initializer are simply absent from the result; run ``FoldConstants`` first
    if ``Constant``-node values should be included. Unsupported initializers
    raise :class:`UnsupportedTensorValueError` rather than being skipped
    silently."""
    summaries: dict[str, TensorValueSummary] = {}
    for initializer in model.graph.initializer:
        # Traverse the initializer collection once and convert the TensorProto
        # already in hand. ModelWrapper.get_initializer rebuilds the complete
        # initializer-name list and searches it on every call, which would make
        # this model-level analysis quadratic in initializer count.
        array = numpy_helper.to_array(initializer)
        summaries[initializer.name] = summarize_tensor_values(array)
    return summaries


def _lossless_candidates() -> list[BaseDataType]:
    """Integer-valued candidate datatypes, genuinely narrowest first.

    ``DataType.get_accumulator_dt_cands()`` is grouped by family, not ordered
    by width: it lists every ``UINT`` before ``TERNARY`` and every ``INT``
    after it, so a linear scan can return a 2-bit ``TERNARY`` for a range that
    1-bit ``INT1`` already represents. Sort by bit width so that "smallest"
    means smallest, and break ties by the original order, which keeps QONNX's
    established preference for unsigned and for the small named types.

    Fixed-point, scaled-integer and arbitrary-precision float datatypes are
    not candidates: their losslessness depends on a scale factor or on
    individual mantissas, neither of which a value summary carries."""
    candidates = [resolve_datatype(name) for name in DataType.get_accumulator_dt_cands()]
    indexed = tuple(enumerate(candidates))
    return [datatype for _index, datatype in sorted(indexed, key=lambda item: (item[1].bitwidth(), item[0]))]


def _admits(datatype: BaseDataType, summary: TensorValueSummary) -> bool:
    """Whether an integer-valued candidate represents every observed value.

    Range containment alone is not the right test: ``BIPOLAR`` and ``TERNARY``
    share the range ``[-1, +1]`` but ``BIPOLAR`` excludes zero, so a tensor
    holding ``{-1, 0, +1}`` is inside ``BIPOLAR``'s range while being outside
    its domain.

    The caller has already established that the summary has a finite integral
    range, and every candidate is integer-valued, so the question is decidable
    here. It is not decidable for datatypes in general, which is why this stays
    private and narrow rather than becoming a public compatibility predicate."""
    minimum, maximum = summary.minimum, summary.maximum
    if minimum is None or maximum is None or not summary.is_integral:
        raise ValueError("_admits requires a summary with a finite integral range")
    if isinstance(datatype, BipolarType):
        return minimum >= -1 and maximum <= 1 and not summary.contains_zero
    if isinstance(datatype, TernaryType):
        return minimum >= -1 and maximum <= 1
    if isinstance(datatype, IntType):
        # every integer between an IntType's bounds is representable
        return bool(datatype.min() <= minimum and maximum <= datatype.max())
    raise ValueError("%s is not an integer-valued candidate datatype" % datatype)


def smallest_lossless_integer_datatype(summary: TensorValueSummary) -> BaseDataType | None:
    """Returns the narrowest integer-valued QONNX datatype that represents
    every observed value, or ``None`` when no such datatype exists.

    The answer is derived from observed values alone. It takes no declared
    datatype, because a summary cannot soundly validate one: a value can sit
    inside ``FLOAT16``'s range while needing more mantissa than ``FLOAT16``
    has, so range containment would prove nothing and trusting the declaration
    would smuggle an unchecked assumption into a value-derived fact. A
    declared datatype that cannot hold its own tensor is a source-model
    inconsistency for the consumer to detect, not a case for this helper.
    Consumers constrain or validate this answer against the declared logical
    datatype themselves.

    The name states the domain deliberately. A summary carries a range and a
    few membership facts, which is enough to prove losslessness for
    integer-valued datatypes and not enough for fractional ones.

    ``None`` -- an explicit refusal rather than an optimistic approximation --
    is returned when:

    * the tensor is empty, so no value was observed;
    * ``minimum``/``maximum`` are absent (every element is NaN);
    * the values are not all finite and integral, so no integer-valued
      datatype represents them; or
    * the observed values exceed the 64-bit candidates."""
    if summary.element_count == 0:
        return None
    if summary.minimum is None or summary.maximum is None:
        return None
    if not summary.is_integral:
        return None
    for candidate in _lossless_candidates():
        if _admits(candidate, summary):
            return candidate
    return None
