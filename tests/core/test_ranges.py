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

"""Exact range bounds compared with independently defined datatype bounds."""

from __future__ import annotations

import pytest

import datatype_oracle as oracle
import math
import numpy as np
from fractions import Fraction

from qonnx.core.datatype import DataType
from qonnx.core.ranges import Range, endpoint, full, holds, hull, unbounded, values_range

NAMES = [
    "BINARY",
    "BIPOLAR",
    "TERNARY",
    "INT8",
    "UINT8",
    "INT64",
    "UINT64",
    "INT100",
    "FIXED<8,4>",
    "FIXED<32,16>",
    "FIXED<53,1>",
    "FIXED<64,32>",
    "FIXED<100,-8>",
    "FLOAT16",
    "FLOAT32",
    "FLOAT<5,10,15>",
    "SCALEDINT<8>",
]


@pytest.mark.parametrize("name", NAMES)
def test_full_is_the_exact_bounds(name: str) -> None:
    found = full(DataType[name])
    item = oracle.spec(name)
    if item.kind in ("ieee", "mini"):
        assert found == unbounded(True)
    elif item.kind == "scaled":
        assert found == unbounded(False)
    else:
        assert found == Range(*oracle.bounds(item))
        assert (found.least, found.greatest) == oracle.bounds(item)


def test_full_of_a_wide_fixed_point_is_exact() -> None:
    assert full(DataType["FIXED<64,32>"]).greatest == Fraction(2**63 - 1, 2**32)


CASES = [
    Range(0, 0),
    Range(-128, 127),
    Range(-129, 127),
    Range(-1, 1),
    Range(0, 255),
    Range(0, 256),
    Range(-(2**63), 2**63 - 1),
    Range(0, 2**64),
    Range(Fraction(1, 2), 1),
    Range(Fraction(-1, 16), Fraction(15, 2)),
    Range(-8, Fraction(127, 16)),
    Range(-8, 8),
    Range(0, 1, nan=True),
    Range(-math.inf, math.inf),
    Range(-math.inf, math.inf, nan=True),
    Range(Fraction(-1, 2**200), 0),
]


@pytest.mark.parametrize("name", NAMES)
def test_holds_matches_the_closed_form(name: str) -> None:
    """A float carries any range; anything else no NaN and ends within its bounds,
    integer ends for an integer datatype; a scaled integer any range without NaN."""
    found, item = DataType[name], oracle.spec(name)
    for case in CASES:
        if item.kind in ("ieee", "mini"):
            expected = True
        elif case.nan:
            expected = False
        elif item.kind == "scaled":
            expected = True
        else:
            least, greatest = oracle.bounds(item)
            integral = all(Fraction(end).denominator == 1 for end in case.span if math.isfinite(end))
            expected = (item.kind == "fixed" or integral) and least <= case.least and case.greatest <= greatest
        assert holds(found, case) is expected, (name, case)


def test_ranges_are_exact_and_refuse_what_is_no_range() -> None:
    assert Range(Fraction(4, 2), 3).least == 2 and type(Range(Fraction(4, 2), 3).least) is int
    assert Range(0.5, 0.75).span == (Fraction(1, 2), Fraction(3, 4))
    for least, greatest, nan, error in (
        (math.nan, 1, False, ValueError),
        (Fraction(1, 3), 1, False, ValueError),
        (2, 1, False, ValueError),
        (0, 1, 1, TypeError),
        (True, 1, False, TypeError),
        ("0", 1, False, TypeError),
    ):
        with pytest.raises(error):
            Range(least, greatest, nan)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        endpoint(math.nan)
    with pytest.raises(ValueError):
        hull()


def test_integers_within_and_hull() -> None:
    assert Range(-3, 5).integers() == (-3, 5)
    for case in (Range(Fraction(1, 2), 1), Range(0, 1, nan=True), Range(0, math.inf)):
        with pytest.raises(ValueError):
            case.integers()
    assert Range(0, 3).within(Range(-1, 3)) and not Range(0, 4).within(Range(-1, 3))
    assert not Range(0, 3, nan=True).within(Range(0, 3))
    assert Range(0, 3).within(Range(0, 3, nan=True))
    assert hull(Range(0, 3), Range(-1, 1, nan=True)) == Range(-1, 3, nan=True)
    assert str(Range(Fraction(1, 2), math.inf, nan=True)) == "[1/2, inf], or NaN"


def test_values_range_is_exact() -> None:
    assert values_range(np.array([3, -2, 7], dtype=np.int8)) == Range(-2, 7)
    assert values_range(np.array([2**64 - 1, 0], dtype=np.uint64)) == Range(0, 2**64 - 1)
    assert values_range(np.array([0.1, -0.5, math.nan])) == Range(-0.5, 0.1, nan=True)
    assert values_range(np.array([0.1], dtype=np.float32)).least == Fraction(float(np.float32(0.1)))
    assert values_range(np.array([math.nan])) == Range(0, 0, nan=True)
    assert values_range(np.array([], dtype=np.int64)) == Range(0, 0)
    objects = np.empty(3, dtype=object)
    objects[:] = [Fraction(1, 4), 2**70, -3]
    assert values_range(objects) == Range(-3, 2**70)


@pytest.mark.parametrize("kind", [np.float16, np.float32, np.float64, np.longdouble])
def test_float_arrays_bound_the_values_as_stored(kind: type) -> None:
    values = np.array([0.1, -0.5, math.nan, math.inf], dtype=kind)
    low = Fraction(*values[1].as_integer_ratio())
    assert values_range(values) == Range(low, math.inf, nan=True)
    finite = values[:2]
    assert values_range(finite) == Range(low, Fraction(*finite[0].as_integer_ratio()))


@pytest.mark.parametrize("kind", [np.bool_, np.int8, np.uint64, np.float32, object])
def test_empty_arrays_and_shapes(kind: type) -> None:
    assert values_range(np.empty((0, 2), dtype=kind)) == Range(0, 0)
    assert values_range(np.array([[0, 1], [1, 0]], dtype=kind)) == Range(0, 1)
    assert values_range(np.array(1, dtype=kind)) == Range(1, 1)


@pytest.mark.parametrize("values", [np.array(["1"]), np.array([1j]), np.array([None], dtype=object)])
def test_an_array_of_non_numbers_has_no_range(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        values_range(values)


def test_object_arrays_and_all_nan_arrays() -> None:
    assert values_range(np.array([math.nan, math.nan])) == Range(0, 0, nan=True)
    assert values_range(np.array([math.nan, Fraction(-1, 4), 2**100], dtype=object)) == Range(
        Fraction(-1, 4), 2**100, nan=True
    )
    with pytest.raises(ValueError):
        values_range(np.array([Fraction(1, 3)], dtype=object))


def test_a_range_is_immutable_and_hashes_canonical_ends() -> None:
    from dataclasses import FrozenInstanceError

    value = Range(0.5, 2.0)
    assert value == Range(Fraction(1, 2), 2)
    assert hash(value) == hash(Range(Fraction(1, 2), 2))
    with pytest.raises(FrozenInstanceError):
        value.least = 0
