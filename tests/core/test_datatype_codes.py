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

"""Codes and values: a value as the code the hardware carries, and back, exactly; and
each datatype's exact container.

``codes`` and ``values`` are checked against codes built bit by bit
(``datatype_oracle.code``: two's complement as a bit string, an IEEE pattern from
``struct``), the round trip ``values(codes(v)) == v`` in each container, FIXED's exact
decode, and the refusals.
"""

from __future__ import annotations

import pytest

import datatype_oracle as oracle
import math
import numpy as np
import numpy.typing as npt
import random
import struct
from fractions import Fraction
from typing import Any

from qonnx.core.datatype import DataType, FixedPointType, IntType

NAMES = [
    "BINARY",
    "BIPOLAR",
    "TERNARY",
    "INT1",
    "INT2",
    "UINT3",
    "INT8",
    "UINT8",
    "INT17",
    "UINT32",
    "INT53",
    "UINT63",
    "INT64",
    "UINT64",
    "FIXED<1,0>",
    "FIXED<8,4>",
    "FIXED<8,-2>",
    "FIXED<16,15>",
    "FIXED<32,16>",
    "FIXED<53,1>",
    "FIXED<54,27>",
    "FIXED<54,-1000>",
    "FLOAT16",
    "FLOAT32",
]
CARRIED = ["FIXED<55,1>", "FIXED<55,54>", "FIXED<64,32>", "FIXED<64,-900>"]


def members(name: str, rng: random.Random) -> list[Fraction | float]:
    """Members at both ends and at random, by the oracle's definition."""
    item = oracle.spec(name)
    least, greatest = oracle.bounds(item)
    if item.kind in ("bipolar", "ternary"):
        ends: list[Fraction | float] = [Fraction(-1), Fraction(1)]
        return ends + ([Fraction(0)] if item.kind == "ternary" else [])
    if item.kind == "ieee":
        pack = "<e" if item.bits == 16 else "<f"
        patterns = [0, 1, 0x7C00 if item.bits == 16 else 0x7F800000]
        patterns += [rng.getrandbits(item.bits) for _ in range(200)]
        found: list[Fraction | float] = [-0.0, math.inf, -math.inf]
        for pattern in patterns:
            value = struct.unpack(pack, pattern.to_bytes(item.bits // 8, "little"))[0]
            if not math.isnan(value):
                found.append(value if math.isinf(value) else Fraction(value))
        return found
    step = Fraction(1, 2**item.fraction) if item.kind == "fixed" else Fraction(1)
    span = int((greatest - least) / step)
    found = [least, greatest, Fraction(0)]
    found += [least + step * rng.randint(0, span) for _ in range(200)]
    return found


def as_container(name: str, values: list[Fraction | float]) -> npt.NDArray[Any]:
    found = DataType[name]
    return np.array([float(v) if found.container().kind == "f" else int(v) for v in values], found.container())


@pytest.mark.parametrize("name", NAMES)
def test_codes_are_the_hardwares_bit_patterns(name: str) -> None:
    rng = random.Random(name)
    found, item = DataType[name], oracle.spec(name)
    values = members(name, rng)
    expected = [oracle.code(item, v) for v in values]
    # As exact Python objects, one at a time and as an object array.
    for value, code in zip(values, expected):
        assert found.codes(value).tolist() == code, (name, value)
    objects = np.empty(len(values), dtype=object)
    objects[:] = values
    assert found.codes(objects).tolist() == expected
    # As the container's array, and back: exact in it.
    array = as_container(name, values)
    codes = found.codes(array)
    assert codes.dtype == np.uint64 and codes.tolist() == expected
    back = found.values(codes)
    assert back.dtype == found.container()
    assert [Fraction(v) if math.isfinite(v) else v for v in back.tolist()] == [
        Fraction(v) if math.isfinite(v) else v for v in array.tolist()
    ]
    # A negative zero keeps its sign in an IEEE float's pattern.
    if item.kind == "ieee":
        assert math.copysign(1, back[values.index(-0.0)]) < 0


@pytest.mark.parametrize("name", NAMES)
def test_values_of_every_code_round_trip(name: str) -> None:
    """Every code of a narrow datatype, and random codes of a wide one, read back to the
    value whose code it is (TERNARY's 2 is no code)."""
    rng = random.Random(name + "codes")
    found, item = DataType[name], oracle.spec(name)
    codes = (
        list(range(2**item.bits))
        if item.bits <= 10
        else [0, 2**item.bits - 1] + [rng.getrandbits(item.bits) for _ in range(300)]
    )
    if name == "TERNARY":
        codes.remove(2)
    values = found.values(np.array(codes, dtype=np.uint64))
    # Reinterpretation is bit for bit: in its own container a NaN keeps its payload.
    with np.errstate(all="raise"):
        assert found.codes(values).tolist() == codes
    for code, value in zip(codes, values.tolist()):
        if item.kind == "ieee" and math.isnan(value):
            continue  # a Python float cannot carry the pattern to the oracle
        assert oracle.code(item, value) == code, (name, code, value)
    nan_free = [v for v in values.tolist() if not (isinstance(v, float) and math.isnan(v))]
    assert all(found.contains(v) for v in nan_free)


@pytest.mark.parametrize("name", CARRIED)
def test_a_wide_fixed_point_is_carried_as_its_codes_both_ways(name: str) -> None:
    """Wider than 54 bits, a FIXED's container holds its codes (int64, its value times
    2**(w - i)), given to ``codes`` and returned by ``values`` alike."""
    rng = random.Random(name)
    found, item = DataType[name], oracle.spec(name)
    assert found.carried_as_codes() and found.container() == np.int64
    top = 2 ** (item.bits - 1)
    signed = [-top, top - 1, 0, -1] + [rng.randint(-top, top - 1) for _ in range(200)]
    codes = found.codes(np.array(signed, dtype=np.int64))
    assert codes.tolist() == [s % 2**item.bits for s in signed]
    assert found.values(codes).tolist() == signed
    assert found.codes(signed[0]).tolist() == signed[0] % 2**item.bits
    for outside in (top, -top - 1, 0.5, Fraction(1, 2)):
        with pytest.raises(ValueError):
            found.codes(outside)
    # Its values themselves are members, exactly: the greatest is (2**(w-1) - 1) / 2**f.
    assert found.contains(Fraction(top - 1, 2**item.fraction)) is True


def test_fixed_decodes_exactly() -> None:
    """FIXED<32,16> code 16777217 is exactly 256.0000152587890625 (float32, its
    to_numpy_dt(), holds only 256.0, which codes back to 16777216); FIXED<64,32>'s greatest
    code is itself."""
    narrow = DataType["FIXED<32,16>"]
    value = narrow.values(np.array([16777217], dtype=np.uint64))
    assert value.dtype == np.float64 and value.tolist() == [256.0000152587890625]
    assert Fraction(value[0]) == Fraction(16777217, 2**16)
    assert narrow.codes(value).tolist() == [16777217]
    wide = DataType["FIXED<64,32>"]
    assert wide.values(np.array([2**63 - 1], dtype=np.uint64)).tolist() == [2**63 - 1]
    rng = random.Random(54)
    for bits in range(2, 55):
        found = FixedPointType(bits, bits // 3)
        assert found == DataType[found.name]
        codes = [rng.getrandbits(bits) for _ in range(50)] + [2 ** (bits - 1) - 1, 2 ** (bits - 1)]
        for code, value in zip(codes, found.values(np.array(codes, dtype=np.uint64)).tolist()):
            signed = code - 2**bits if code >> (bits - 1) else code
            assert Fraction(value) == Fraction(signed, 2 ** found.frac_bits())


def test_refusals() -> None:
    int4 = DataType["INT4"]
    for value in (8, -9, 0.5, np.array([1, 8]), "1", None, True, math.nan, [1]):
        with pytest.raises(ValueError):
            int4.codes(value)
    for codes in (16, np.array([16], dtype=np.uint64), -1, np.array([-1]), 1.0, "1"):
        with pytest.raises(ValueError):
            int4.values(codes)
    with pytest.raises(ValueError):
        DataType["TERNARY"].values(2)
    with pytest.raises(ValueError):
        DataType["FLOAT32"].codes(0.1)
    for found in (DataType["INT65"], DataType["FIXED<65,1>"], DataType["FLOAT<5,10,15>"], DataType["SCALEDINT<8>"]):
        for operation in (found.codes, found.values):
            with pytest.raises(ValueError):
                operation(0)
    for operation in (IntType(5000, True).codes, IntType(5000, True).values):
        with pytest.raises(ValueError):
            operation(0)


def test_bipolar_and_ternary_codes() -> None:
    assert DataType["BIPOLAR"].codes(np.array([-1, 1, 1])).tolist() == [0, 1, 1]
    assert DataType["BIPOLAR"].values(np.array([0, 1])).tolist() == [-1, 1]
    assert DataType["TERNARY"].codes(np.array([-1, 0, 1])).tolist() == [3, 0, 1]
    assert DataType["TERNARY"].values(np.array([3, 0, 1])).tolist() == [-1, 0, 1]


@pytest.mark.parametrize("bits", (16, 32))
def test_nan_patterns_are_coded_by_their_bits(bits: int) -> None:
    """Signaling and quiet NaNs of either sign, given in the container (as an array, a
    numpy scalar, or in an object array), are coded as their own patterns, with no
    floating-point exception raised."""
    found = DataType[f"FLOAT{bits}"]
    pattern, container = (np.uint16, np.float16) if bits == 16 else (np.uint32, np.float32)
    exponent = 0x7C00 if bits == 16 else 0x7F800000
    sign = 1 << (bits - 1)
    nans = [exponent | 1, exponent | 3, sign | exponent | 1, exponent | (exponent >> 1) | 5]
    given = np.array(nans, dtype=pattern).view(container)
    objects = np.empty(len(nans), dtype=object)
    objects[:] = list(given)
    with np.errstate(all="raise"):
        assert found.codes(given).tolist() == nans
        assert [int(found.codes(value)) for value in given] == nans
        assert found.codes(objects).tolist() == nans
        assert found.values(np.array(nans, dtype=np.uint64)).view(pattern).tolist() == nans


CONTAINERS = {
    "BINARY": np.uint8,
    "UINT8": np.uint8,
    "UINT9": np.uint16,
    "INT1": np.int8,
    "INT16": np.int16,
    "INT17": np.int32,
    "UINT33": np.uint64,
    "INT64": np.int64,
    "BIPOLAR": np.int8,
    "TERNARY": np.int8,
    "FIXED<8,4>": np.float64,
    "FIXED<54,1>": np.float64,
    "FIXED<55,1>": np.int64,
    "FIXED<64,32>": np.int64,
    "FLOAT16": np.float16,
    "FLOAT32": np.float32,
    "FLOAT<5,10,15>": np.float32,
    "FLOAT<8,22,128>": np.float32,
}


@pytest.mark.parametrize(("name", "container"), CONTAINERS.items())
def test_containers(name: str, container: type) -> None:
    found = DataType[name]
    assert found.container() == np.dtype(container)
    assert found.carried_as_codes() is (name in ("FIXED<55,1>", "FIXED<64,32>"))


def test_to_numpy_dt_keeps_its_answers() -> None:
    """A fixed point's to_numpy_dt() stays float32; its exact container is beside it."""
    assert DataType["FIXED<8,4>"].to_numpy_dt() == np.float32
    assert DataType["FIXED<8,4>"].container() == np.float64
    assert DataType["FIXED<64,32>"].to_numpy_dt() == np.float32


@pytest.mark.parametrize(
    "name",
    ["INT65", "UINT128", "FIXED<65,1>", "SCALEDINT<8>", "FLOAT<8,24,127>", "FLOAT<9,7,255>"],
)
def test_no_container_where_none_is_exact(name: str) -> None:
    with pytest.raises(ValueError):
        DataType[name].container()
    with pytest.raises(ValueError):
        IntType(5000, True).container()


@pytest.mark.parametrize(
    ("datatype", "plain", "values"),
    (
        (IntType(np.int64(64), True), IntType(64, True), [-(2**63), -1, 0, 2**63 - 1]),
        (IntType(np.int64(64), False), IntType(64, False), [0, 1, 2**64 - 1]),
        (FixedPointType(np.int64(8), np.int64(4)), FixedPointType(8, 4), [-8.0, -0.0625, 0.0, 7.9375]),
        (FixedPointType(np.int64(64), np.int64(1)), FixedPointType(64, 1), [-(2**63), -1, 0, 2**63 - 1]),
    ),
    ids=["INT64", "UINT64", "FIXED<8,4>", "FIXED<64,1>"],
)
def test_numpy_integer_parameters_give_the_same_codes(datatype: Any, plain: Any, values: list[Any]) -> None:
    codes = datatype.codes(np.array(values, dtype=plain.container()))
    assert codes.tolist() == plain.codes(np.array(values, dtype=plain.container())).tolist()
    assert datatype.values(codes).tolist() == plain.values(codes).tolist()
