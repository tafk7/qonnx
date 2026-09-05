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
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union, cast

from qonnx.core.datatype import BaseDataType, DataType, resolve_datatype

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

# digest domain separator; bump when the digested preimage changes
_DIGEST_SCHEMA = b"qonnx.tensor_value_summary.v1"

# numpy dtype kinds whose values QONNX can summarize soundly:
# signed integer, unsigned integer, floating point, boolean
_SUPPORTED_KINDS = frozenset("iufb")


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
      empty tensor, which :func:`smallest_lossless_datatype` still refuses.

    NaN values are excluded from ``minimum``/``maximum`` so that equal content
    yields equal summaries; their presence is reported by ``is_integral``
    being ``False``. Infinities are ordinary observed values and do appear in
    ``minimum``/``maximum``.
    """

    content_digest: str
    element_count: int
    minimum: Union[int, float, None]
    maximum: Union[int, float, None]
    is_integral: bool


def is_summarizable_dtype(dtype: npt.DTypeLike) -> bool:
    """Returns whether values of this numpy dtype can be summarized soundly.

    Supported: signed and unsigned integers, IEEE floating point (float16,
    float32, float64) and booleans.

    Unsupported: complex, string/object, and every sub-byte or custom-encoded
    numeric type that ``onnx.numpy_helper`` returns as a *view* dtype with
    fields (bfloat16, the float8 variants, the 4-bit types). Their raw bytes
    are bit patterns rather than numbers, so a range taken over them would be
    meaningless rather than merely imprecise.

    ``INT4``/``UINT4`` are excluded here even though QONNX has datatypes of
    those names: the exclusion is a property of the packed bytes numpy hands
    back, not of the datatype being inexpressible. Decoding those encodings
    into ordinary arrays would lift the exclusion without changing any value
    in this module's contract."""
    resolved = np.dtype(dtype)
    if resolved.fields is not None or resolved.subdtype is not None:
        return False
    return resolved.kind in _SUPPORTED_KINDS


def summarize_tensor_values(array: npt.NDArray[Any]) -> TensorValueSummary:
    """Summarizes the values of an in-memory tensor.

    Raises :class:`UnsupportedTensorValueError` if the array's dtype is not
    supported by :func:`is_summarizable_dtype`."""
    array = np.asarray(array)
    if not is_summarizable_dtype(array.dtype):
        raise UnsupportedTensorValueError(
            "Cannot summarize tensor values of dtype %s: only integer, "
            "IEEE floating point and boolean tensors are supported." % str(array.dtype)
        )
    contiguous = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(_DIGEST_SCHEMA)
    hasher.update(str(contiguous.dtype.str).encode("utf-8"))
    hasher.update(str(contiguous.shape).encode("utf-8"))
    hasher.update(contiguous.tobytes())
    element_count = int(contiguous.size)
    minimum, maximum, is_integral = _observed_range(contiguous)
    return TensorValueSummary(
        content_digest=hasher.hexdigest(),
        element_count=element_count,
        minimum=minimum,
        maximum=maximum,
        is_integral=is_integral,
    )


def _observed_range(
    array: npt.NDArray[Any],
) -> tuple[Union[int, float, None], Union[int, float, None], bool]:
    """Returns (minimum, maximum, is_integral) for a supported array."""
    if array.size == 0:
        # empty is not zero: no observed value, and vacuously integral
        return (None, None, True)
    if array.dtype.kind == "f":
        finite = np.isfinite(array)
        not_nan = ~np.isnan(array)
        if not not_nan.any():
            # every element is NaN: nothing comparable was observed
            return (None, None, False)
        comparable = array[not_nan]
        is_integral = bool(finite.all()) and bool(np.all(comparable == np.trunc(comparable)))
        return (float(comparable.min()), float(comparable.max()), is_integral)
    # integer and boolean values are exact and always integral
    return (int(array.min()), int(array.max()), True)


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
        array = cast(Union[npt.NDArray[Any], None], model.get_initializer(initializer.name))
        assert array is not None, "initializer disappeared during analysis"
        summaries[initializer.name] = summarize_tensor_values(array)
    return summaries


def _lossless_candidate_names() -> list[str]:
    """Candidate datatypes for range-based lossless narrowing, smallest first.

    This is ``DataType.get_accumulator_dt_cands()`` without ``BIPOLAR``.
    BIPOLAR admits only ``{-1, +1}``; whether ``0`` occurs is not derivable
    from an observed range, so offering it here could silently drop zeros.
    TERNARY is sound because it admits every integer in ``[-1, +1]``.

    Fixed-point, scaled-integer and arbitrary-precision float datatypes are
    excluded: their losslessness depends on a scale factor or on individual
    mantissas, neither of which a min/max summary carries."""
    return [c for c in DataType.get_accumulator_dt_cands() if c != "BIPOLAR"]


def smallest_lossless_datatype(
    summary: TensorValueSummary, declared_datatype: BaseDataType | None = None
) -> BaseDataType | None:
    """Returns the smallest QONNX datatype that exactly represents the
    observed values, or ``None`` when no such datatype can be established.

    ``declared_datatype``, when given, is the datatype the summarized tensor
    is already declared to hold; the result is never wider than it. When the
    smallest candidate is not narrower than the declared type, the declared
    type is returned unchanged.

    ``None`` -- an explicit refusal rather than an optimistic approximation --
    is returned when:

    * the tensor is empty, so no value was observed;
    * ``minimum``/``maximum`` are absent (every element is NaN);
    * the values are not all finite and integral, so losslessness cannot be
      decided from a range summary (any float tensor with a fractional, NaN
      or infinite element);
    * the observed range exceeds the 64-bit integer candidates; or
    * ``declared_datatype`` is given and cannot itself represent the observed
      range, or does not define a range at all (e.g. ``SCALEDINT``)."""
    if summary.element_count == 0:
        return None
    if summary.minimum is None or summary.maximum is None:
        return None
    if not summary.is_integral:
        return None
    minimum, maximum = summary.minimum, summary.maximum
    if declared_datatype is not None:
        try:
            declared_min = declared_datatype.min()
            declared_max = declared_datatype.max()
        except Exception:
            # datatypes without a defined range cannot bound the result
            return None
        if minimum < declared_min or maximum > declared_max:
            return None
    for candidate_name in _lossless_candidate_names():
        candidate = resolve_datatype(candidate_name)
        if candidate.min() <= minimum and maximum <= candidate.max():
            if declared_datatype is not None:
                if candidate.bitwidth() >= declared_datatype.bitwidth():
                    return declared_datatype
            return candidate
    return None
