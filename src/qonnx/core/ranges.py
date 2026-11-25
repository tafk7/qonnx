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

"""A range: the values a tensor may hold, with each end exact.

A range (``Range``) has two ends and a flag: whether the tensor may hold NaN. Each end
is an exact dyadic number, ``m * 2**e`` (an int, or a ``Fraction`` whose denominator is
a power of two: every float, fixed-point or scaled-integer value is one exactly), or an
infinity. A datatype's own range is ``full``; ``holds`` says whether a datatype can
carry a range.
"""

from __future__ import annotations

import math
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from qonnx.core.datatype import (
    BaseDataType,
    FixedPointType,
    NumericValueRefused,
    ScaledIntType,
    _Endpoint,
    _endpoint,
    _exact,
)

Endpoint = _Endpoint
endpoint = _endpoint

__all__ = [
    "Endpoint",
    "Range",
    "endpoint",
    "full",
    "holds",
    "hull",
    "unbounded",
    "values_range",
]


def _text(value: Endpoint) -> str:
    if isinstance(value, float):
        return "inf" if value > 0 else "-inf"
    return str(value)


@dataclass(frozen=True)
class Range:
    """A bound on tensor values: the interval ``[least, greatest]``, and NaN if
    ``nan``. Ends are canonical (``endpoint``) and ``least <= greatest``. Invalid values raise
    ``ValueError``; a wrong Python type raises ``TypeError``."""

    least: Endpoint
    greatest: Endpoint
    nan: bool = False

    def __post_init__(self) -> None:
        least, greatest = endpoint(self.least), endpoint(self.greatest)
        if not least <= greatest:
            raise NumericValueRefused.of(
                "range-invalid", "Range", f"a range's least end exceeds its greatest: [{least}, {greatest}]"
            )
        if not isinstance(self.nan, bool):
            raise TypeError(f"a range's NaN flag is a bool, not {self.nan!r}")
        object.__setattr__(self, "least", least)
        object.__setattr__(self, "greatest", greatest)

    @property
    def span(self) -> tuple[Endpoint, Endpoint]:
        return self.least, self.greatest

    def is_integer(self) -> bool:
        """Whether its ends are integers and it excludes NaN. The datatype specifies the values between them."""
        return isinstance(self.least, int) and isinstance(self.greatest, int) and not self.nan

    def integers(self) -> tuple[int, int]:
        """The ends of an integer range (``is_integer``); ``ValueError`` for any other."""
        if not (isinstance(self.least, int) and isinstance(self.greatest, int) and not self.nan):
            raise NumericValueRefused.of("range-invalid", "Range.integers", f"{self} is no range of integers")
        return self.least, self.greatest

    def within(self, outer: Range) -> bool:
        """Every value it admits is one ``outer`` admits: its ends within ``outer``'s, and
        NaN only where ``outer`` admits it."""
        return outer.least <= self.least and self.greatest <= outer.greatest and (outer.nan or not self.nan)

    def __str__(self) -> str:
        ends = f"[{_text(self.least)}, {_text(self.greatest)}]"
        return ends + (", or NaN" if self.nan else "")


def unbounded(nan: bool) -> Range:
    """Every number, and NaN if ``nan``."""
    return Range(-math.inf, math.inf, nan)


def hull(*ranges: Range) -> Range:
    """The least range holding each of ``ranges`` (one at least)."""
    if any(not isinstance(item, Range) for item in ranges):
        raise TypeError("hull takes Range values")
    if not ranges:
        raise NumericValueRefused.of("range-invalid", "hull", "the hull of no range")
    return Range(
        min(item.least for item in ranges),
        max(item.greatest for item in ranges),
        any(item.nan for item in ranges),
    )


def _float(datatype: BaseDataType) -> bool:
    return not (datatype.is_integer() or isinstance(datatype, (FixedPointType, ScaledIntType)))


def full(datatype: BaseDataType) -> Range:
    """A datatype's own range: an integer or fixed-point datatype's bounds; a scaled
    integer's unbounded (its scale is not part of it), without NaN; a float's unbounded,
    with NaN."""
    if not isinstance(datatype, BaseDataType):
        raise TypeError("full takes a BaseDataType")
    if datatype.is_integer() or isinstance(datatype, FixedPointType):
        least, greatest = datatype.bounds()
        return Range(least, greatest)
    if isinstance(datatype, ScaledIntType):
        return unbounded(False)
    return unbounded(True)


def holds(datatype: BaseDataType, value: Range) -> bool:
    """Whether ``datatype`` can carry ``value``: a float any range; a scaled integer any
    range without NaN; an integer datatype a range of integers within its bounds; a fixed
    point a range without NaN within its bounds. A range bounds a tensor's values; which
    of its numbers the tensor holds, its datatype says."""
    if not isinstance(datatype, BaseDataType) or not isinstance(value, Range):
        raise TypeError("holds takes a BaseDataType and a Range")
    if _float(datatype):
        return True
    if value.nan:
        return False
    if isinstance(datatype, ScaledIntType):
        return True
    if datatype.is_integer() and not value.is_integer():
        return False
    bounds = full(datatype)
    return bounds.least <= value.least and value.greatest <= bounds.greatest


def values_range(values: npt.NDArray[Any]) -> Range:
    """The exact range of an array's values: its least and greatest number, each exactly
    as stored, and NaN if it holds one. An empty array's is ``[0, 0]``."""
    flat = np.asarray(values).reshape(-1)
    if flat.dtype.kind in "iub":
        if flat.size == 0:
            return Range(0, 0)
        return Range(int(flat.min()), int(flat.max()))
    if flat.dtype.kind == "f" and flat.dtype.itemsize <= 8:
        numbers = flat.astype(np.float64)
        nan = bool(np.isnan(numbers).any())
        numbers = numbers[~np.isnan(numbers)]
        if numbers.size == 0:
            return Range(0, 0, nan)
        return Range(endpoint(float(numbers.min())), endpoint(float(numbers.max())), nan)
    found = [_exact(value) for value in flat]
    if any(value is None for value in found):
        raise NumericValueRefused.of("range-invalid", "values_range", "a range is of numbers")
    nan = any(isinstance(value, float) and math.isnan(value) for value in found)
    exact_numbers = [
        value if isinstance(value, float) else Fraction(value)  # type: ignore[arg-type]
        for value in found
        if not (isinstance(value, float) and math.isnan(value))
    ]
    if not exact_numbers:
        return Range(0, 0, nan)
    return Range(endpoint(min(exact_numbers)), endpoint(max(exact_numbers)), nan)
