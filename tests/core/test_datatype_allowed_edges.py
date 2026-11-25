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

"""allowed() at the edges: its stated bugs fixed, its meaning kept.

For integer, bipolar, ternary and fixed-point datatypes, allowed() is checked against
``datatype_oracle``'s membership for numbers given in every Python and numpy form. Each
input that allowed() answered wrongly, or crashed on, is a named case. What allowed()
keeps (its answers for bools, strings and float datatypes, and its return types) is
checked as it was.
"""

from __future__ import annotations

import pytest

import datatype_oracle as oracle
import math
import numpy as np
import random
from fractions import Fraction

from qonnx.core.datatype import DataType, FixedPointType, IntType

NAMES = [
    "BINARY",
    "BIPOLAR",
    "TERNARY",
    "INT2",
    "UINT2",
    "INT8",
    "UINT8",
    "INT16",
    "UINT16",
    "INT32",
    "UINT32",
    "INT53",
    "INT54",
    "UINT63",
    "INT64",
    "UINT64",
    "INT65",
    "UINT100",
    "FIXED<1,0>",
    "FIXED<4,2>",
    "FIXED<8,-2>",
    "FIXED<16,15>",
    "FIXED<32,16>",
    "FIXED<53,1>",
    "FIXED<55,1>",
    "FIXED<64,32>",
    "FIXED<100,50>",
    "FIXED<1,-1073>",
]

INTEGER_TYPES = (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
FLOAT_TYPES = (np.float16, np.float32, np.float64, np.longdouble)


def numbers(name: str, rng: random.Random) -> list[Fraction | float]:
    """Each bound, a step past it, half a step, the 64-bit edges, infinities and NaN, and
    random values on the datatype's grid and off it."""
    item = oracle.spec(name)
    least, greatest = oracle.bounds(item)
    step = Fraction(1, 2**item.fraction) if item.kind == "fixed" else Fraction(1)
    found: list[Fraction | float] = [least, greatest, least - step, greatest + step, step / 2]
    found += [Fraction(0), step, -step, Fraction(1, 3), Fraction(-1), Fraction(1), Fraction(2)]
    found += [Fraction(2) ** 63, Fraction(2) ** 64, -(Fraction(2) ** 63) - 1, Fraction(2) ** 53 + 1]
    found += [Fraction(2) ** 31, Fraction(2) ** 54, Fraction(54940)]
    found += [0.0, -0.0, math.inf, -math.inf, math.nan]
    span = int((greatest - least) / step)
    for _ in range(30):
        found.append(least + step * rng.randint(0, span))
        found.append(least + step * rng.randint(0, span) + step / rng.choice((2, 3, 4)))
    return found


def forms(value: Fraction | float) -> list[object]:
    """``value`` as each Python and numpy scalar that holds it (a numpy float's own
    rounding, if any, is what is asked about)."""
    if isinstance(value, float):
        return [value, np.float16(value), np.float32(value), np.float64(value), np.longdouble(value)]
    found: list[object] = []
    if value.denominator == 1:
        found.append(int(value))
        for integral in INTEGER_TYPES:
            info = np.iinfo(integral)
            if int(info.min) <= value <= int(info.max):
                found.append(integral(int(value)))
    if abs(value) < 2**1000:
        found.append(float(value))
        with np.errstate(over="ignore", under="ignore"):
            found += [floating(float(value)) for floating in FLOAT_TYPES]
    return found


@pytest.mark.parametrize("name", NAMES)
def test_allowed_is_exact_membership_for_numbers(name: str) -> None:
    rng = random.Random(name)
    found, item = DataType[name], oracle.spec(name)
    for value in numbers(name, rng):
        for form in forms(value):
            answer = found.allowed(form)
            assert bool(answer) is oracle.member(item, form), (name, form, type(form))
            assert np.asarray(answer).shape == ()


@pytest.mark.parametrize("name", NAMES)
def test_allowed_is_exact_membership_for_arrays(name: str) -> None:
    rng = random.Random(name + "arrays")
    found, item = DataType[name], oracle.spec(name)
    values = numbers(name, rng)
    arrays = []
    for integral in INTEGER_TYPES:
        info = np.iinfo(integral)
        fitting = [int(v) for v in values if isinstance(v, Fraction) and v.denominator == 1]
        arrays.append(np.array([v for v in fitting if int(info.min) <= v <= int(info.max)], dtype=integral))
    for floating in FLOAT_TYPES:
        with np.errstate(over="ignore", under="ignore"):
            arrays.append(np.array([float(v) for v in values], dtype=floating))
    objects = np.empty(len(values), dtype=object)
    objects[:] = [v if isinstance(v, float) else (int(v) if v.denominator == 1 else v) for v in values]
    arrays.append(objects)
    for array in arrays:
        for shaped in (array, array.reshape(1, -1)):
            answer = found.allowed(shaped)
            assert isinstance(answer, np.ndarray) and answer.dtype == np.bool_
            assert answer.shape == shaped.shape
            expected = [oracle.member(item, value) for value in shaped.flat]
            assert answer.reshape(-1).tolist() == expected, (name, shaped.dtype)


def test_a_bound_is_not_rounded_into_the_inputs_float_type() -> None:
    """A float input was compared with bounds converted to its own type, so a value just
    past a bound, or an infinity past a bound that became one, was allowed."""
    assert not DataType["INT64"].allowed(np.float64(2**63))
    assert not DataType["UINT64"].allowed(np.float64(2**64))
    assert DataType["INT64"].allowed(np.array([2.0**63, 2.0**62, -(2.0**63)])).tolist() == [False, True, True]
    assert not DataType["INT32"].allowed(np.float32(2**31))
    assert not DataType["UINT16"].allowed(np.float16(math.inf))
    assert not DataType["INT64"].allowed(np.float16(-math.inf))
    # A fixed point's scaled value met the same rounding.
    assert not DataType["FIXED<64,32>"].allowed(np.float64(2**31))
    assert not DataType["FIXED<64,32>"].allowed(np.int64(2**31))
    assert DataType["FIXED<64,32>"].allowed(np.float64(2**31 - 2**-21))
    assert not DataType["FIXED<55,1>"].allowed(np.int64(1))
    assert DataType["FIXED<55,1>"].allowed(np.float64(1 - 2**-52))


def test_a_fixed_points_scaling_stays_exact() -> None:
    """A fixed point's value was scaled in the input's float type, which overflowed to an
    infinity (allowed) or underflowed to zero (0 / 0, not allowed)."""
    assert not DataType["FIXED<32,16>"].allowed(np.float16(54940.0))
    assert DataType["FIXED<32,16>"].allowed(np.float16(-32768.0))
    assert DataType["FIXED<64,32>"].allowed(np.float16(0.0))
    assert DataType["FIXED<1,-1073>"].allowed(np.float32(0.0))
    assert DataType["FIXED<64,32>"].allowed(np.array([0.0, 0.5, 54940.0], dtype=np.float16)).tolist() == [True] * 3


def test_crashes_become_answers() -> None:
    """Object arrays and Python ints past 64 bits (TypeError), numpy floats against an
    integer wider than 1024 bits (OverflowError) and unsigned 64-bit integers for BIPOLAR
    and TERNARY on numpy 1.x (IndexError) are answered."""
    int64 = DataType["INT64"]
    assert int64.allowed(np.array([2**63, 2**63 - 1], dtype=object)).tolist() == [False, True]
    assert int64.allowed(np.array([2.0**63], dtype=object)).tolist() == [False]
    assert DataType["FIXED<64,32>"].allowed(np.array([1.5, Fraction(1, 3)], dtype=object)).tolist() == [True, False]
    assert not DataType["UINT64"].allowed(2**64)
    assert not DataType["INT3"].allowed(2**70)
    assert DataType["INT65"].allowed(-(2**64))
    assert DataType["INT4096"].allowed(np.float64(1.0))
    assert not DataType["INT4096"].allowed(np.float64(0.5))
    assert DataType["BIPOLAR"].allowed(np.array([1, 0], dtype=np.uint64)).tolist() == [True, False]
    assert DataType["TERNARY"].allowed(np.array([1, 0, 2], dtype=np.uint64)).tolist() == [True, True, False]
    assert DataType["BIPOLAR"].allowed(np.uint64(2**63)).tolist() is False


def test_what_allowed_keeps() -> None:
    """Return types, bools, inputs that are no number, the float datatypes, and names past
    the envelope answer as before."""
    assert type(DataType["INT8"].allowed(5)) is np.bool_
    assert type(DataType["FIXED<8,4>"].allowed(0.5)) is np.bool_
    assert type(DataType["INT8"].allowed(np.array(5.0))) is np.bool_
    assert isinstance(DataType["BIPOLAR"].allowed(1), np.ndarray)
    assert isinstance(DataType["BIPOLAR"].allowed(np.uint8(1)), np.ndarray)
    assert DataType["INT8"].allowed(True) and DataType["BIPOLAR"].allowed(True)
    assert DataType["UINT8"].allowed(np.array([True, False])).tolist() == [True, True]
    assert DataType["INT8"].allowed(np.array([True, 1, 2**63, "1", None], dtype=object)).tolist() == [
        True,
        True,
        False,
        False,
        False,
    ]
    with pytest.raises(TypeError):
        DataType["INT8"].allowed("1")
    assert not DataType["BIPOLAR"].allowed("1")
    assert DataType["FLOAT32"].allowed(0.1) and DataType["FLOAT16"].allowed(1e10)
    assert bool(DataType["FLOAT<4,3>"].allowed(1.875)) and not bool(DataType["FLOAT<4,3>"].allowed(1.8))
    assert DataType["INT5000"].allowed(5)


class _Byte(IntType):
    """A datatype defined outside qonnx on IntType, stating its own range: 0 to 200 in
    eight bits."""

    def __init__(self) -> None:
        super().__init__(8, False)

    def max(self) -> int:
        return 200

    def get_canonical_name(self) -> str:
        return "BYTE200"


def test_a_subclass_keeps_its_own_range() -> None:
    """IntType.allowed() reads a subclass's min() and max(), as it always has."""
    byte = _Byte()
    assert byte.allowed(200) and not byte.allowed(201)
    assert byte.allowed(np.array([0, 200, 201, 255], dtype=np.uint8)).tolist() == [True, True, False, False]
    assert byte.allowed(np.array([200.0, 200.5, 201.0])).tolist() == [True, False, False]


# Upstream's answers for bools, which allowed() leaves to its original tests: True, False,
# np.True_, np.False_, then np.array([True, False]). A fixed point reads a bool scaled by
# 2**frac_bits, so FIXED<55,1> allows np.True_ (2**54 compared in float64) and not True.
BOOL_ANSWERS = {
    "INT8": ([True, True, True, True], [True, True]),
    "BIPOLAR": ([True, False, True, False], [True, False]),
    "TERNARY": ([True, True, True, True], [True, True]),
    "FIXED<8,4>": ([True, True, True, True], [True, True]),
    "FIXED<55,1>": ([False, True, True, True], [True, True]),
    "FIXED<64,1>": ([False, True, True, True], [True, True]),
    "FIXED<8,1>": ([False, True, False, True], [False, True]),
}


@pytest.mark.parametrize("name", sorted(BOOL_ANSWERS))
def test_bools_keep_upstreams_answers(name: str) -> None:
    datatype = DataType[name]
    scalars, array = BOOL_ANSWERS[name]
    assert [bool(datatype.allowed(value)) for value in (True, False, np.True_, np.False_)] == scalars
    assert datatype.allowed(np.array([True, False])).tolist() == array


class _HalfOpen(IntType):
    """A datatype defined outside qonnx on IntType whose min() and max() are not
    integers: its integers are 1 to 200."""

    def __init__(self) -> None:
        super().__init__(8, False)

    def min(self) -> Fraction:
        return Fraction(1, 2)

    def max(self) -> Fraction:
        return Fraction(401, 2)

    def get_canonical_name(self) -> str:
        return "HALFOPEN"


def test_a_subclass_range_that_ends_between_integers() -> None:
    """Upstream's answers: the integers within min() and max(), and no other."""
    half = _HalfOpen()
    assert half.allowed(np.array([0, 1, 200, 201], dtype=np.uint8)).tolist() == [False, True, True, False]
    assert not half.allowed(np.uint8(0)) and not half.allowed(0) and half.allowed(1)
    assert half.allowed(np.array([0.0, 1.0, 200.0, 200.5])).tolist() == [False, True, True, False]


class _BelowZero(IntType):
    """A datatype defined outside qonnx on IntType whose range ends between integers
    below zero: its integers are -200 to -1."""

    def __init__(self) -> None:
        super().__init__(8, True)

    def min(self) -> Fraction:
        return Fraction(-401, 2)

    def max(self) -> Fraction:
        return Fraction(-1, 2)

    def get_canonical_name(self) -> str:
        return "BELOWZERO"


def test_a_subclass_range_that_ends_between_integers_below_zero() -> None:
    """Upstream's answers, in integer arrays narrower and wider than the range, and in
    unsigned ones at their extremes."""
    below = _BelowZero()
    assert below.allowed(np.array([-128, -1, 0, 127], dtype=np.int8)).tolist() == [True, True, False, False]
    assert below.allowed(np.array([-201, -200, -1, 0], dtype=np.int16)).tolist() == [False, True, True, False]
    assert below.allowed(np.array([0, 1, 2**64 - 1], dtype=np.uint64)).tolist() == [False, False, False]
    assert below.allowed(-200) and below.allowed(-1) and not below.allowed(0) and not below.allowed(-201)


def test_a_long_double_array_is_decided_at_float64_speed_where_float64_holds_it() -> None:
    """Long doubles that float64 holds exactly are decided with it; the others one at a
    time, exactly."""
    values = np.ones(10, dtype=np.longdouble)
    values[1] = np.longdouble(1) + np.longdouble(2) ** -60
    values[2] = np.longdouble(np.inf)
    found = DataType["FIXED<32,16>"].allowed(values)
    assert found.tolist() == [True, np.finfo(np.longdouble).nmant < 60, False] + [True] * 7


def test_a_fixed_point_past_the_envelope_answers_as_before() -> None:
    """Upstream's answer: its original test, which cannot round an object array."""
    with pytest.warns(RuntimeWarning), pytest.raises(TypeError):
        FixedPointType(8, -2000).allowed(np.array([np.int64(1)], dtype=object))


def neighbours(value: float, dtype: type) -> list[object]:
    """``value`` in ``dtype`` and the three values of ``dtype`` on each side of it."""
    with np.errstate(over="ignore"):
        found = [dtype(value)]
    for direction in (np.inf, -np.inf):
        step = found[0]
        for _ in range(3):
            step = np.nextafter(step, dtype(direction))
            found.append(step)
    return found


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.longdouble])
@pytest.mark.parametrize(
    "name", ["INT8", "UINT16", "INT32", "UINT64", "FIXED<8,4>", "FIXED<32,16>", "FIXED<64,32>", "FIXED<100,3>"]
)
def test_arrays_near_the_bounds_and_the_grid_answer_as_their_elements(name: str, dtype: type) -> None:
    """An array answers element by element as each element does alone, which is decided
    exactly, at the bounds, at grid points and between them, in every float type."""
    datatype = DataType[name]
    least, greatest = datatype.bounds()
    step = Fraction(1, 2 ** datatype.frac_bits()) if name.startswith("FIXED") else 1
    values: list[object] = []
    for point in (least, greatest, 0, step, -step, step / 2, least + step, greatest - step):
        with np.errstate(over="ignore"):
            values += neighbours(float(point), dtype)
    values += [dtype(np.inf), dtype(-np.inf), dtype(np.nan), np.finfo(dtype).smallest_subnormal]
    array = np.array(values, dtype=dtype)
    assert datatype.allowed(array).tolist() == [bool(datatype.allowed(value)) for value in array]
