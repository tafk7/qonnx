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

"""Exact bounds: each datatype's least and greatest value against its closed form.

``bounds()`` is checked against ``datatype_oracle``, which reads a canonical name and
states its bounds without importing qonnx. ``min()`` and ``max()`` keep their answers:
where they answer, they are the floats nearest the exact bounds.
"""

from __future__ import annotations

import pytest

import datatype_oracle as oracle
import numpy as np
from fractions import Fraction

from qonnx.core.datatype import ArbPrecFloatType, BaseDataType, DataType, FixedPointType, IntType, ScaledIntType

INTEGERS = [
    f"{prefix}INT{bits}" for prefix in ("", "U") for bits in (*range(1, 66), 128, 4096) if (prefix, bits) != ("U", 1)
]
FIXED = [f"FIXED<{w},{i}>" for w in (1, 8, 32, 53, 54, 55, 64) for i in (-8, 0, 1, w // 2, w - 1) if i < w] + [
    "FIXED<1,-1073>",
    "FIXED<1075,1>",
    "FIXED<1025,1024>",
    "FIXED<2098,1024>",
]
FLOATS = ["FLOAT16", "FLOAT32", "FLOAT<5,10,15>", "FLOAT<4,3,7>", "FLOAT<5,2,15>"]
FLOATS += ["FLOAT<8,7,127>", "FLOAT<5,10,7>", "FLOAT<11,51,1024>", "FLOAT<1,52,-1022>"]
SPECIAL = ["BINARY", "BIPOLAR", "TERNARY"]
VALUED = SPECIAL + INTEGERS + FIXED + FLOATS

# Fixed-point datatypes whose min() and max() overflow a float today: their integer
# bound is past float64's range before it is scaled. bounds() is exact for them.
VIEW_OVERFLOWS = {"FIXED<1075,1>", "FIXED<1025,1024>", "FIXED<2098,1024>"}


@pytest.mark.parametrize("name", VALUED)
def test_bounds_are_exact_and_min_max_are_their_nearest_floats(name: str) -> None:
    found = DataType[name]
    least, greatest = oracle.bounds(oracle.spec(name))
    assert found.bounds() == (least, greatest)
    for end in found.bounds():
        assert type(end) is (int if Fraction(end).denominator == 1 else Fraction)
    if name in VIEW_OVERFLOWS:
        return
    if found.is_integer():
        assert (Fraction(found.min()), Fraction(found.max())) == (least, greatest)
        assert type(found.min()) is int and type(found.max()) is int
    else:
        # Python rounds a Fraction to the nearest float, ties to even.
        assert (found.min(), found.max()) == (float(least), float(greatest))
        assert type(found.min()) is float and type(found.max()) is float


def test_the_named_bound_of_the_module_docstring() -> None:
    wide = DataType["FIXED<64,32>"]
    assert wide.max() == 2147483648.0
    assert wide.bounds()[1] == Fraction(2**63 - 1, 2**32)


def test_the_float64_edge() -> None:
    """FLOAT<1,52,-1022>'s greatest value is float64's own; with one more mantissa bit it
    is float64's overflow midpoint, whose nearest float is an infinity: past the
    envelope."""
    edge = DataType["FLOAT<1,52,-1022>"]
    assert edge.bounds()[1] == 2**1024 - 2**971
    assert edge.max() == 1.7976931348623157e308 and edge.min() == -1.7976931348623157e308
    with pytest.raises(ValueError):
        DataType["FLOAT<1,53,-1022>"].bounds()


def test_a_scaled_integer_has_no_bounds() -> None:
    with pytest.raises(ValueError):
        DataType["SCALEDINT<8>"].bounds()


@pytest.mark.parametrize(
    "datatype",
    [
        IntType(0, True),
        IntType(0, False),
        IntType(-3, True),
        FixedPointType(0, -1),
        ScaledIntType(0),
        ArbPrecFloatType(0, 3),
        ArbPrecFloatType(5, -1),
        ArbPrecFloatType(5, 10, 1.5),
    ],
    ids=repr,
)
def test_no_bounds_for_a_width_that_holds_no_values(datatype: BaseDataType) -> None:
    with pytest.raises(ValueError):
        datatype.bounds()


PAST_THE_ENVELOPE = [
    IntType(4097, False),
    IntType(5000, True),
    IntType(2**64, True),
    FixedPointType(4097, 1),
    FixedPointType(1, -1074),
    FixedPointType(1026, 1025),
    FixedPointType(8, -(2**64) + 8),
    ArbPrecFloatType(12, 3, 2047),
    ArbPrecFloatType(11, 52, 1022),
    ArbPrecFloatType(5, 2000, 15),
    ArbPrecFloatType(3, 4095),
    ArbPrecFloatType(1, 53, -1022),
    ArbPrecFloatType(1, 100, -1022),
    ArbPrecFloatType(5, 10, 2**2000),
    ArbPrecFloatType(5, 10, -(2**2000)),
]


@pytest.mark.parametrize("datatype", PAST_THE_ENVELOPE, ids=lambda datatype: type(datatype).__name__)
def test_no_bounds_past_the_envelope(datatype: BaseDataType) -> None:
    """Refused before any huge number is built: none of these is slow."""
    with pytest.raises(ValueError):
        datatype.bounds()


class _Halves(BaseDataType):
    """A datatype defined outside qonnx, with no bounds() of its own: -1/2 to 3/2 in
    steps of a half."""

    def bitwidth(self) -> int:
        return 3

    def min(self) -> float:
        return -0.5

    def max(self) -> float:
        return 1.5

    def allowed(self, value: object) -> bool:
        return bool(np.all(np.asarray(value) * 2 == np.round(np.asarray(value) * 2)))

    def get_num_possible_values(self) -> int:
        return 5

    def is_integer(self) -> bool:
        return False

    def is_fixed_point(self) -> bool:
        return True

    def get_hls_datatype_str(self) -> str:
        return "ap_fixed<3,2>"

    def to_numpy_dt(self) -> np.dtype:
        return np.dtype(np.float32)

    def get_canonical_name(self) -> str:
        return "HALVES"


def test_a_subclass_without_bounds_reads_min_and_max_exactly() -> None:
    assert _Halves().bounds() == (Fraction(-1, 2), Fraction(3, 2))


class _Byte(IntType):
    """A datatype defined outside qonnx on IntType, stating its own range: 0 to 200 in
    eight bits."""

    def __init__(self) -> None:
        super().__init__(8, False)

    def max(self) -> int:
        return 200

    def get_canonical_name(self) -> str:
        return "BYTE200"


def test_a_subclass_that_states_its_own_range_has_it_as_bounds() -> None:
    assert _Byte().bounds() == (0, 200)


# Parameters given as numpy integers, as a caller reading them from an array does: the
# same datatype as the Python ints give (numpy shifts past 63 bits would wrap).
NUMPY_PARAMETERS = [
    (FixedPointType(8, np.int64(-66)), FixedPointType(8, -66)),
    (FixedPointType(np.int64(64), np.int64(1)), FixedPointType(64, 1)),
    (FixedPointType(np.int32(40), np.int32(-30)), FixedPointType(40, -30)),
    (IntType(np.int64(64), True), IntType(64, True)),
    (IntType(np.uint8(100), False), IntType(100, False)),
    (ArbPrecFloatType(np.int64(8), np.int64(23)), ArbPrecFloatType(8, 23)),
    (ArbPrecFloatType(np.int64(8), np.int64(23), np.int64(100)), ArbPrecFloatType(8, 23, 100)),
]


@pytest.mark.parametrize(
    ("datatype", "plain"),
    NUMPY_PARAMETERS,
    ids=["FIXED<8,-66>", "FIXED<64,1>", "FIXED<40,-30>", "INT64", "UINT100", "FLOAT<8,23>", "FLOAT<8,23,100>"],
)
def test_numpy_integer_parameters_give_the_same_bounds(datatype: BaseDataType, plain: BaseDataType) -> None:
    assert datatype.bounds() == plain.bounds()
