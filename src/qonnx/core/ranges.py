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

This module is the value and its rules alone: the range, a datatype's full
range and what a datatype holds, the exact range of an array's values, the
rounding of an exact value into a binary float format (``FloatFormat``), and
the range's stored text. It loads no ONNX, so a value layer that holds ranges (as it
holds datatypes, ``qonnx.core.datatype``) need not. Where a range is stored on a
model and read back, inside an inference pass, is ``qonnx.core.tensor_ranges``.

The stored text is canonical JSON, ``{"nan":false,"range":[least,greatest]}``,
each endpoint a JSON integer for an integer, the string ``"m*2^-k"`` for a
non-integer (m odd, k >= 1: ``"3*2^-2"`` is 0.75), or ``"inf"`` or ``"-inf"``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Union

from qonnx.core.datatype import BaseDataType, FixedPointType, ScaledIntType

if TYPE_CHECKING:
    import numpy as np

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


@dataclass(frozen=True)
class FloatFormat:
    """A binary floating-point format as IEEE 754 defines one: a sign, ``mantissa``
    fraction bits, normal exponents ``emin`` to ``emax``, subnormals below them and
    an infinity of each sign beyond its largest finite value."""

    mantissa: int
    emin: int
    emax: int

    @property
    def largest(self) -> Endpoint:
        """The largest finite value, (2 - 2^-mantissa) * 2^emax."""
        return endpoint((2 - Fraction(1, 2**self.mantissa)) * Fraction(2) ** self.emax)

    def nearest(self, value: Union[int, Fraction, float]) -> Endpoint:
        """``value``, an exact rational or an infinity, rounded once into the format
        as IEEE 754's round to nearest, ties to even, does: past the largest finite
        value by half its spacing or more, to the infinity of its sign. A zero
        result is 0 (a range does not tell -0 from +0)."""
        if isinstance(value, float):
            if math.isnan(value):
                raise ValueError("NaN is not rounded into a range")
            if math.isinf(value):
                return value
            value = Fraction(value)
        value = Fraction(value)
        if value == 0:
            return 0
        size = abs(value)
        # the exponent of size's leading bit, exactly: 2^exponent <= size < 2^(exponent + 1)
        exponent = size.numerator.bit_length() - size.denominator.bit_length()
        if size < Fraction(2) ** exponent:
            exponent -= 1
        # the spacing of the format's values at size: its normal binade's, or the subnormals'
        spacing = Fraction(2) ** (max(exponent, self.emin) - self.mantissa)
        rounded = round(size / spacing) * spacing  # round() of a Fraction: ties to even
        if rounded > self.largest:
            return math.inf if value > 0 else -math.inf
        return endpoint(rounded if value > 0 else -rounded)


# The formats of ONNX's float element types.
BINARY16 = FloatFormat(10, -14, 15)
BINARY32 = FloatFormat(23, -126, 127)
BINARY64 = FloatFormat(52, -1022, 1023)
BFLOAT16 = FloatFormat(7, -126, 127)


def is_float(datatype: BaseDataType) -> bool:
    """Whether ``datatype`` is a float: neither an integer, a fixed-point nor a
    scaled-integer datatype."""
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
    if is_float(datatype):
        return True
    if value.nan:
        return False
    if isinstance(datatype, ScaledIntType):
        return True
    if datatype.is_integer() and not value.is_integer():
        return False
    bounds = full(datatype)
    return bounds.least <= value.least and value.greatest <= bounds.greatest


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


__all__ = [
    "BFLOAT16",
    "BINARY16",
    "BINARY32",
    "BINARY64",
    "Endpoint",
    "FloatFormat",
    "RangeContradiction",
    "TensorRange",
    "endpoint",
    "full",
    "holds",
    "hull",
    "is_float",
    "unbounded",
    "values_range",
]
