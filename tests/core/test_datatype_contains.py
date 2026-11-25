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

"""contains(): exact membership, for every kind of input.

``contains`` is checked against ``datatype_oracle``, which decides membership in
``Fraction`` (an IEEE format's through ``struct``, a narrow float's by enumerating its bit
patterns), over every family, at each datatype's edges and at random, given as Python and
numpy scalars of every width, arrays of each, and object arrays.
"""

from __future__ import annotations

import pytest

import datatype_oracle as oracle
import math
import numpy as np
import random
from fractions import Fraction
from typing import Any

from qonnx.core.datatype import ArbPrecFloatType, DataType, FixedPointType, IntType, ScaledIntType

NAMES = [
    "BINARY",
    "BIPOLAR",
    "TERNARY",
    "INT1",
    "INT2",
    "UINT2",
    "INT8",
    "UINT8",
    "INT16",
    "UINT32",
    "INT53",
    "INT54",
    "UINT63",
    "INT64",
    "UINT64",
    "INT65",
    "UINT100",
    "FIXED<1,0>",
    "FIXED<8,4>",
    "FIXED<8,-2>",
    "FIXED<16,15>",
    "FIXED<32,16>",
    "FIXED<53,1>",
    "FIXED<54,27>",
    "FIXED<55,1>",
    "FIXED<64,32>",
    "FIXED<100,50>",
    "FIXED<1,-1073>",
    "FLOAT16",
    "FLOAT32",
    "FLOAT<5,10,15>",
    "FLOAT<4,3,7>",
    "FLOAT<5,2,15>",
    "FLOAT<3,1,1>",
]

INTEGER_TYPES = (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
FLOAT_TYPES = (np.float16, np.float32, np.float64, np.longdouble)
NOT_NUMBERS: tuple[object, ...] = (True, False, np.True_, "1", None, 1 + 0j, [1], b"1", object())


def numbers(name: str, rng: random.Random) -> list[Fraction | float]:
    """The values to ask about: each bound, one step past it, half a step, zero, signed
    zeros, infinities and NaN, and random values on the datatype's grid and off it."""
    item = oracle.spec(name)
    least, greatest = oracle.bounds(item)
    if item.kind == "fixed":
        step = Fraction(1, 2**item.fraction)
    elif item.kind in ("ieee", "mini"):
        step = Fraction(1, 2**item.mantissa_bits if item.kind == "mini" else 1)
    else:
        step = Fraction(1)
    found: list[Fraction | float] = [least, greatest, least - step, greatest + step, step / 2]
    found += [Fraction(0), step, -step, Fraction(1, 3), Fraction(1, 10), Fraction(-1)]
    found += [Fraction(2) ** 63, Fraction(2) ** 64, -(Fraction(2) ** 63) - 1, Fraction(2) ** 53 + 1]
    found += [0.0, -0.0, math.inf, -math.inf, math.nan]
    if item.kind in ("ieee", "mini"):
        found += [Fraction(65504), Fraction(65520), Fraction(2) ** -24, Fraction(2) ** -25]
        found += [Fraction(2) ** -149, Fraction(2) ** -150, Fraction(3, 2**25)]
    for _ in range(40):
        if item.kind in ("ieee", "mini"):
            found.append(Fraction(rng.uniform(-2, 2)) * Fraction(2) ** rng.randint(-30, 30))
        else:
            span = int((greatest - least) / step)
            found.append(least + step * rng.randint(0, span))
            found.append(least + step * rng.randint(0, span) + step / rng.choice((2, 3, 4)))
    return found


def forms(value: Fraction | float) -> list[object]:
    """``value`` as every kind of scalar that holds it: Python int, float and Fraction,
    and each numpy integer and float type (its own rounding, if any, is what is asked)."""
    found: list[object] = [value]
    if isinstance(value, float):
        found += [np.float16(value), np.float32(value), np.float64(value), np.longdouble(value)]
        return found
    found.append(float(value) if abs(value) < 2**1000 else math.inf)
    if value.denominator == 1:
        found.append(int(value))
        for integral in INTEGER_TYPES:
            info = np.iinfo(integral)
            if int(info.min) <= value <= int(info.max):
                found.append(integral(int(value)))
    for floating in FLOAT_TYPES:
        if abs(value) < 2**1000:
            with np.errstate(over="ignore", under="ignore"):
                found.append(floating(float(value)))
    return found


@pytest.mark.parametrize("name", NAMES)
def test_scalar_membership_is_exact(name: str) -> None:
    rng = random.Random(name)
    found, item = DataType[name], oracle.spec(name)
    for value in numbers(name, rng):
        for form in forms(value):
            answer = found.contains(form)
            assert type(answer) is bool, (name, form)
            assert answer is oracle.member(item, form), (name, form, type(form))
    for form in NOT_NUMBERS:
        assert found.contains(form) is False


@pytest.mark.parametrize("name", NAMES)
def test_array_membership_is_exact_for_every_element_type(name: str) -> None:
    rng = random.Random(name + "arrays")
    found, item = DataType[name], oracle.spec(name)
    values = numbers(name, rng)
    rationals = [value for value in values if isinstance(value, Fraction)]
    arrays: list[np.ndarray[Any, Any]] = []
    for integral in INTEGER_TYPES:
        info = np.iinfo(integral)
        low, high = int(info.min), int(info.max)
        fitting = [int(v) for v in rationals if v.denominator == 1 and low <= v <= high]
        arrays.append(np.array(fitting, dtype=integral))
    for floating in FLOAT_TYPES:
        with np.errstate(over="ignore", under="ignore"):
            finite = [v for v in values if not (isinstance(v, Fraction) and abs(v) >= 2**1000)]
            arrays.append(np.array([float(v) for v in finite], floating))
    mixed = [*values, *forms(Fraction(1)), *NOT_NUMBERS]
    arrays.append(np.empty(len(mixed), dtype=object))
    arrays[-1][:] = mixed
    arrays.append(np.array([True, False]))
    arrays.append(np.array(["1", "2"]))
    arrays.append(np.array([1 + 0j]))
    for array in arrays:
        shapes = (array, array.reshape(1, -1), *((array[:1].reshape(()),) if array.size else ()))
        for shaped in shapes:
            answer = found.contains(shaped)
            assert isinstance(answer, np.ndarray) and answer.dtype == np.bool_
            assert answer.shape == shaped.shape
            expected = [oracle.member(item, value) for value in shaped.flat]
            assert answer.reshape(-1).tolist() == expected, (name, shaped.dtype)


def test_named_cases() -> None:
    """Where membership and allowed() part, and the exact edges of a wide fixed point."""
    assert DataType["INT64"].contains(np.float64(2.0**63)) is False
    assert DataType["UINT64"].contains(np.float64(2.0**64)) is False
    assert DataType["FLOAT16"].contains(1e10) is False
    assert DataType["FLOAT32"].contains(0.1) is False and DataType["FLOAT32"].allowed(0.1)
    assert DataType["FLOAT32"].contains(np.float32(0.1)) is True
    assert DataType["INT8"].contains(True) is False and DataType["INT8"].allowed(True)
    assert DataType["FIXED<64,32>"].contains(Fraction(2**63 - 1, 2**32)) is True
    assert DataType["FIXED<64,32>"].contains(2147483648.0) is False


def test_infinities_and_nan_are_only_ieee_floats() -> None:
    for name in NAMES:
        found = DataType[name]
        ieee = name in ("FLOAT16", "FLOAT32")
        for value in (math.inf, -math.inf, math.nan, np.float32("nan")):
            assert found.contains(value) is ieee
        assert found.contains(np.array([math.inf, math.nan])).tolist() == [ieee, ieee]
        assert found.contains(-0.0) is found.contains(0)


@pytest.mark.parametrize(
    "datatype",
    [
        DataType["SCALEDINT<8>"],
        ScaledIntType(0),
        IntType(0, True),
        IntType(-3, False),
        IntType(5000, True),
        FixedPointType(1, -1074),
        ArbPrecFloatType(0, 3),
        ArbPrecFloatType(12, 3, 2047),
    ],
    ids=lambda datatype: type(datatype).__name__,
)
def test_a_datatype_without_exact_bounds_contains_nothing(datatype: Any) -> None:
    """A scaled integer has no value set; a nonsense width or one past the envelope has
    no exact bounds. Membership is False for each, and does not raise."""
    for value in (0, 1, -1, 0.5, math.inf):
        assert datatype.contains(value) is False
    assert datatype.contains(np.array([0, 1])).tolist() == [False, False]


@pytest.mark.parametrize(
    ("datatype", "plain"),
    (
        (FixedPointType(8, np.int64(-66)), FixedPointType(8, -66)),
        (FixedPointType(np.int64(64), np.int64(1)), FixedPointType(64, 1)),
        (IntType(np.int64(64), True), IntType(64, True)),
        (ArbPrecFloatType(np.int64(5), np.int64(10)), ArbPrecFloatType(5, 10)),
    ),
    ids=["FIXED<8,-66>", "FIXED<64,1>", "INT64", "FLOAT<5,10>"],
)
def test_numpy_integer_parameters_give_the_same_members(datatype: Any, plain: Any) -> None:
    least, greatest = plain.bounds()
    values = [0, least, greatest, Fraction(least) / 3, greatest + 1, 0.5, -1]
    assert [datatype.contains(value) for value in values] == [plain.contains(value) for value in values]
    assert datatype.contains(0)
