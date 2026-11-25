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

"""round_to(): IEEE 754's round to nearest, ties to even, applied once.

``round_to`` and ``FloatFormat.nearest`` are checked against the C library's
packing (``struct``'s ``e`` and ``f`` formats round a float64 to nearest even) and, for
exact values a float64 cannot hold, against binary16 by enumeration of its bit patterns
and binary32 at constructed ties.
"""

from __future__ import annotations

import pytest

import datatype_oracle as oracle
import math
import numpy as np
import random
import struct
from fractions import Fraction

from qonnx.core.datatype import BINARY16, BINARY32, DataType, round_to


def packed(bits: int, value: float) -> float:
    """``value`` rounded by the C library, or the infinity of its sign on overflow."""
    try:
        return float(struct.unpack("<e" if bits == 16 else "<f", struct.pack("<e" if bits == 16 else "<f", value))[0])
    except OverflowError:
        return math.copysign(math.inf, value)


def same(found: float, expected: float) -> bool:
    if math.isnan(expected):
        return math.isnan(found)
    return found == expected and math.copysign(1, found) == math.copysign(1, expected)


def float64s(rng: random.Random) -> list[float]:
    found = [0.0, -0.0, math.inf, -math.inf, math.nan, 0.1, 1 / 3, 65504.0, 65519.0, 65520.0]
    found += [2.0**-24, 2.0**-25, 2.0**-25 * 3, 2.0**-26, 2.0**-149, 2.0**-150, 3 * 2.0**-151]
    found += [float(np.finfo(np.float32).max), 3.4028235677973366e38, 3.402823669209385e38]
    found += [
        1 + 2.0**-11,
        1 + 2.0**-11 + 2.0**-40,
        1 + 3 * 2.0**-11,
        1 + 2.0**-24,
        1 + 3 * 2.0**-24,
    ]
    for _ in range(3000):
        found.append(rng.uniform(-1, 1) * 2.0 ** rng.randint(-160, 140))
        found.append(struct.unpack("<d", rng.getrandbits(64).to_bytes(8, "little"))[0])
    return found


@pytest.mark.parametrize("bits", (16, 32))
def test_round_to_is_the_c_librarys_rounding_of_a_float64(bits: int) -> None:
    rng = random.Random(bits)
    target = DataType[f"FLOAT{bits}"]
    values = float64s(rng)
    found = round_to(target, np.array(values))
    assert found.dtype == target.container()
    for value, rounded in zip(values, found.tolist()):
        assert same(rounded, packed(bits, value)), (bits, value, rounded)
    for value in values[:40]:  # one at a time, as Python floats
        assert same(float(round_to(target, value)), packed(bits, value)), value


def test_exact_values_round_once() -> None:
    """A value no float64 holds is rounded from its exact value, never through float64
    (which would round twice): binary16 against every one of its patterns; binary32 at
    ties between neighbours, a hair either side, and integers past 2**53."""
    rng = random.Random(16)
    half = DataType["FLOAT16"]
    exact = [Fraction(rng.randint(-(2**40), 2**40), 2 ** rng.randint(10, 70)) for _ in range(800)]
    exact += [Fraction(65504 + 16), Fraction(65504 + 16) - Fraction(1, 2**60), Fraction(1, 3)]
    exact += [Fraction(2**-25) + Fraction(1, 2**80), -Fraction(2**-25), Fraction(2**-26)]
    found = round_to(half, np.array(exact, dtype=object))
    for value, rounded in zip(exact, found.tolist()):
        assert same(rounded, oracle.nearest(16, value)) or (rounded == 0 and oracle.nearest(16, value) == 0), value
    single = DataType["FLOAT32"]
    for _ in range(300):
        pattern = rng.randrange(0x7F7FFFFF)
        low = Fraction(struct.unpack("<f", pattern.to_bytes(4, "little"))[0])
        high = Fraction(struct.unpack("<f", (pattern + 1).to_bytes(4, "little"))[0])
        even = low if pattern % 2 == 0 else high
        tie, nudge = (low + high) / 2, (high - low) / 2**40
        assert Fraction(float(round_to(single, tie))) == even
        assert Fraction(float(round_to(single, tie - nudge))) == low
        assert Fraction(float(round_to(single, tie + nudge))) == high
    for integer in (2**53 + 1, 2**60 + 2**36 + 1, -(2**62) - 2**38, 2**64 - 1):
        expected = float(oracle_round32(Fraction(integer)))
        assert float(round_to(single, integer)) == expected
        assert round_to(single, np.array([integer], dtype=object)).tolist() == [expected]
        if -(2**63) <= integer < 2**63:
            assert round_to(single, np.array([integer], dtype=np.int64)).tolist() == [expected]


def oracle_round32(value: Fraction) -> Fraction:
    """binary32 by its neighbours: the two adjacent float32 patterns around ``value`` from
    ``struct``, the nearer one, the even one at a tie."""
    guess = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    pattern = int.from_bytes(struct.pack("<f", abs(guess)), "little")
    neighbours = [Fraction(struct.unpack("<f", p.to_bytes(4, "little"))[0]) for p in (pattern - 1, pattern, pattern + 1)]
    size = abs(value)
    best = min(range(3), key=lambda i: (abs(neighbours[i] - size), (pattern - 1 + i) % 2))
    return neighbours[best] if value > 0 else -neighbours[best]


def test_a_zero_keeps_its_sign_and_nan_stays_nan() -> None:
    for bits in (16, 32):
        target = DataType[f"FLOAT{bits}"]
        assert math.copysign(1, round_to(target, -1e-300)) < 0
        assert math.copysign(1, round_to(target, np.array([-1e-300]))[0]) < 0
        assert math.copysign(1, round_to(target, -0.0)) < 0
        assert math.copysign(1, round_to(target, Fraction(-1, 2**200))) < 0
        assert math.isnan(round_to(target, math.nan))
        assert round_to(target, np.array([math.inf])).tolist() == [math.inf]


def test_only_an_ieee_float_is_a_target() -> None:
    for name in ("INT8", "FIXED<8,4>", "FLOAT<5,10,15>", "BIPOLAR", "SCALEDINT<8>"):
        with pytest.raises(ValueError):
            round_to(DataType[name], 1.0)
    with pytest.raises(TypeError):
        round_to(DataType["FLOAT32"], "1.0")


@pytest.mark.parametrize(("fmt", "bits"), ((BINARY16, 16), (BINARY32, 32)))
def test_float_format_nearest_is_ieees(fmt: object, bits: int) -> None:
    """``FloatFormat.nearest`` (a range's rounding) agrees with the C library, but for a
    zero, which a range does not sign."""
    rng = random.Random(bits + 1)
    for value in float64s(rng):
        if math.isnan(value):
            continue
        found = fmt.nearest(value)  # type: ignore[attr-defined]
        expected = packed(bits, value)
        assert found == (0 if expected == 0 else (expected if math.isinf(expected) else Fraction(expected)))


SOURCES = (np.float16, np.float32, np.float64, np.longdouble)


@pytest.mark.parametrize("source", SOURCES, ids=lambda kind: np.dtype(kind).name)
def test_every_float_source_type_rounds_once_as_scalar_and_array(source: type) -> None:
    """Midpoints between adjacent binary16 and binary32 values, and a hair either side by
    the source type's own precision, given as that type: the scalar and the array round
    alike, to the neighbour the patterns say (a tie to the even one)."""
    precision = int(np.finfo(source).nmant)
    rng = random.Random(f"source-{precision}")
    checked = 0
    for bits, pack, top in ((16, "<e", 0x7BFF), (32, "<f", 0x7F7FFFFF)):
        target = DataType[f"FLOAT{bits}"]
        for pattern in [0x3C00, 0x0001, 0x03FF] + [rng.randrange(top) for _ in range(60)]:
            low = Fraction(struct.unpack(pack, pattern.to_bytes(bits // 8, "little"))[0])
            high = Fraction(struct.unpack(pack, (pattern + 1).to_bytes(bits // 8, "little"))[0])
            tie = (low + high) / 2
            hair = Fraction(2) ** (math.floor(math.log2(tie)) - precision)
            cases = [(low, low), (high, high), (tie, low if pattern % 2 == 0 else high)]
            if hair < (high - low) / 2:
                cases += [(tie - hair, low), (tie + hair, high)]
            for value, expected in cases:
                with np.errstate(all="ignore"):
                    given = source(value.numerator) / source(value.denominator)
                if not np.isfinite(given) or Fraction(*given.as_integer_ratio()) != value:
                    continue  # not one of this type's values
                checked += 1
                scalar = round_to(target, given)
                array = round_to(target, np.array([given, -given], dtype=source))
                assert Fraction(float(scalar)) == expected, (source, bits, pattern, value)
                assert [Fraction(v) for v in array.tolist()] == [expected, -expected]
    assert checked >= 40, checked
