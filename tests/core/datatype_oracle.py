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

"""Each datatype's values, from its definition alone: the oracle the datatype tests
compare with.

Written apart from qonnx and importing none of it: a canonical name parsed here, its
bounds in closed form, membership in ``Fraction`` (an IEEE format's through ``struct``,
the C library's binary16 and binary32; a narrow float's by enumerating every bit
pattern), a code built bit by bit.
"""

from __future__ import annotations

import math
import numpy as np
import re
import struct
from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
from functools import cache

#: A value read exactly: a Fraction, or the strings "inf", "-inf", "nan"; None for no number.
Read = Fraction | str | None


@dataclass(frozen=True)
class Spec:
    """A datatype as its canonical name defines it."""

    name: str
    kind: str  # int, bipolar, ternary, fixed, ieee, mini, scaled
    bits: int
    signed: bool = True
    integer_bits: int = 0
    exponent_bits: int = 0
    mantissa_bits: int = 0
    bias: int = 0

    @property
    def fraction(self) -> int:
        return self.bits - self.integer_bits


def spec(name: str) -> Spec:
    """The definition a canonical name gives."""
    if name == "BINARY":
        return Spec(name, "int", 1, signed=False)
    if name == "BIPOLAR":
        return Spec(name, "bipolar", 1)
    if name == "TERNARY":
        return Spec(name, "ternary", 2)
    if name in ("FLOAT16", "FLOAT32"):
        return Spec(name, "ieee", int(name[5:]))
    found = re.fullmatch(r"(U?)INT(\d+)", name)
    if found:
        return Spec(name, "int", int(found.group(2)), signed=not found.group(1))
    found = re.fullmatch(r"FIXED<(\d+),(-?\d+)>", name)
    if found:
        return Spec(name, "fixed", int(found.group(1)), integer_bits=int(found.group(2)))
    found = re.fullmatch(r"FLOAT<(\d+),(\d+),(-?\d+)>", name)
    if found:
        e, m, b = (int(group) for group in found.groups())
        return Spec(name, "mini", 1 + e + m, exponent_bits=e, mantissa_bits=m, bias=b)
    found = re.fullmatch(r"SCALEDINT<(\d+)>", name)
    if found:
        return Spec(name, "scaled", int(found.group(1)))
    raise AssertionError(f"the oracle has no definition of {name}")


def bounds(item: Spec) -> tuple[Fraction, Fraction]:
    """The least and greatest value, in closed form."""
    if item.kind == "int":
        if item.signed:
            return Fraction(-(2 ** (item.bits - 1))), Fraction(2 ** (item.bits - 1) - 1)
        return Fraction(0), Fraction(2**item.bits - 1)
    if item.kind in ("bipolar", "ternary"):
        return Fraction(-1), Fraction(1)
    if item.kind == "fixed":
        scale = Fraction(2) ** item.fraction
        return -(2 ** (item.bits - 1)) / scale, (2 ** (item.bits - 1) - 1) / scale
    if item.kind == "ieee":
        m, emax = (10, 15) if item.bits == 16 else (23, 127)
    else:
        m, emax = item.mantissa_bits, 2**item.exponent_bits - 1 - item.bias
    largest = (2 - Fraction(1, 2**m)) * Fraction(2) ** emax
    return -largest, largest


def read(value: object) -> Read:
    """``value`` read exactly, as the contract lists the kinds a member may be given as."""
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "nan"
        if np.isinf(value):
            return "inf" if value > 0 else "-inf"
        return Fraction(*value.as_integer_ratio())
    return None


def _ieee(bits: int) -> str:
    return "<e" if bits == 16 else "<f"


def representable(bits: int, value: Fraction) -> bool:
    """Whether binary16 or binary32 holds ``value``: it survives float64 and the C
    library's packing unchanged."""
    if Fraction(float(value)) != value:
        return False
    try:
        packed = struct.pack(_ieee(bits), float(value))
    except OverflowError:
        return False
    return Fraction(struct.unpack(_ieee(bits), packed)[0]) == value


@cache
def mini_values(e: int, m: int, b: int) -> frozenset[Fraction]:
    """Every value of FLOAT<e,m,b>, one per bit pattern: no infinity, no NaN."""
    found: set[Fraction] = set()
    for exponent in range(2**e):
        for mantissa in range(2**m):
            if exponent == 0:
                magnitude = Fraction(mantissa, 2**m) * Fraction(2) ** (1 - b)
            else:
                magnitude = (1 + Fraction(mantissa, 2**m)) * Fraction(2) ** (exponent - b)
            found.update((magnitude, -magnitude))
    return frozenset(found)


def member(item: Spec, value: object) -> bool:
    """Whether ``value`` is one of the datatype's values."""
    number = read(value)
    if number is None:
        return False
    if isinstance(number, str):
        return item.kind == "ieee"
    least, greatest = bounds(item)
    if item.kind == "int":
        return number.denominator == 1 and least <= number <= greatest
    if item.kind == "bipolar":
        return number in (-1, 1)
    if item.kind == "ternary":
        return number in (-1, 0, 1)
    if item.kind == "fixed":
        return (number * 2**item.fraction).denominator == 1 and least <= number <= greatest
    if item.kind == "ieee":
        return representable(item.bits, number)
    if item.kind == "mini":
        return number in mini_values(item.exponent_bits, item.mantissa_bits, item.bias)
    raise AssertionError(f"{item.name} has no values")


def code(item: Spec, value: object) -> int:
    """A member's code, built as a bit string: two's complement for an integer, a fixed
    point (scaled) and TERNARY; BIPOLAR's -1 and +1 as 0 and 1; an IEEE float's pattern."""
    number = read(value)
    if item.kind == "ieee":
        packed = struct.pack(_ieee(item.bits), float(value))  # type: ignore[arg-type]
        return int.from_bytes(packed, "little")
    assert isinstance(number, Fraction)
    if item.kind == "bipolar":
        return int("1" if number > 0 else "0", 2)
    if item.kind == "fixed":
        number *= 2**item.fraction
    assert number.denominator == 1
    integer = int(number)
    pattern = format(integer + 2**item.bits if integer < 0 else integer, f"0{item.bits}b")
    assert len(pattern) == item.bits
    return int(pattern, 2)


def nearest(bits: int, value: Fraction) -> float:
    """``value`` rounded to binary16 by enumeration: the nearest of every finite pattern,
    a tie to the one whose pattern is even, past the largest by half a spacing to an
    infinity."""
    assert bits == 16
    finite = _binary16()
    largest = finite[-1][0]
    half = (largest - finite[-2][0]) / 2
    if abs(value) >= largest + half:
        return math.inf if value > 0 else -math.inf
    size = abs(value)
    at = bisect_right(_binary16_keys(), size)
    candidates = finite[max(at - 1, 0) : at + 1]
    best = min(candidates, key=lambda item: (abs(item[0] - size), item[1] & 1))
    magnitude = float(best[0])
    if value < 0:
        return -magnitude
    return magnitude


@cache
def _binary16() -> tuple[tuple[Fraction, int], ...]:
    found = []
    for pattern in range(0x7C00):  # the non-negative finite patterns
        found.append((Fraction(struct.unpack("<e", pattern.to_bytes(2, "little"))[0]), pattern))
    return tuple(found)


@cache
def _binary16_keys() -> tuple[Fraction, ...]:
    return tuple(item[0] for item in _binary16())
