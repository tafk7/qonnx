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

"""The value rules, by brute force.

Algorithm proofs of the rules against enumeration: every operand vector of
small ranges and every column of small weights, each result's least and greatest (and
every partial sum's, which the rules also bound) found by summing; widths by searching
for the fewest bits whose bounds hold the range, at each power-of-two edge.
"""

from __future__ import annotations

import pytest

import itertools
from collections.abc import Callable
from fractions import Fraction

from qonnx.core.datatype import IntType
from qonnx.core.range_rules import column_range, dot_range, gemm_result, integer_width, operand_range, range_dtype
from qonnx.core.ranges import Range, holds


def fewest_bits(least: int, greatest: int, signed: bool) -> int:
    """The first width whose bounds hold both ends."""
    bits = 1
    while True:
        low, high = (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) if signed else (0, 2**bits - 1)
        if low <= least and greatest <= high:
            return bits
        bits += 1


def edges() -> list[tuple[int, int]]:
    found = [(least, greatest) for least in range(-40, 41) for greatest in range(least, 41)]
    for b in range(1, 71):
        p = 2**b
        found += [(0, p - 1), (0, p), (0, p + 1), (-p, 0), (-p - 1, 0), (-p + 1, 0)]
        found += [(-p, p - 1), (-p, p), (-p - 1, p - 1), (-p, -p), (-p - 1, -p), (p, p)]
    return found


def test_integer_width_is_the_fewest_bits() -> None:
    for least, greatest in edges():
        assert integer_width(least, greatest, True) == fewest_bits(least, greatest, True)
        if least >= 0:
            assert integer_width(least, greatest, False) == fewest_bits(least, greatest, False)


def test_range_dtype_is_the_smallest_integer_datatype() -> None:
    for least, greatest in edges():
        signed = least < 0
        bits = fewest_bits(least, greatest, signed)
        found = range_dtype(least, greatest)
        assert found == IntType(bits, signed)
        assert found.get_canonical_name() == (
            "BINARY" if (bits, signed) == (1, False) else f"{'INT' if signed else 'UINT'}{bits}"
        )
        assert holds(found, Range(least, greatest))
        assert found.contains(least) and found.contains(greatest)


def small_ranges() -> list[Range]:
    return [Range(lo, hi) for lo in range(-3, 4) for hi in range(lo, min(lo + 3, 4))]


def sums(terms: list[list[int]]) -> tuple[int, int]:
    """The least and greatest sum of any subset of the terms, each term one of its values
    (the empty subset among them: the sum before the first term)."""
    least = greatest = 0
    for choice in itertools.product(*terms):
        for size in range(len(choice) + 1):
            for subset in itertools.combinations(choice, size):
                least, greatest = min(least, sum(subset)), max(greatest, sum(subset))
    return least, greatest


def test_operand_range_widens_to_zero() -> None:
    for value in small_ranges():
        values = range(value.least, value.greatest + 1)  # type: ignore[arg-type]
        assert operand_range(value) == (min(0, *values), max(0, *values))


def test_dot_range_holds_every_dot_product_and_partial_sum() -> None:
    for x, w, k in itertools.product(small_ranges(), small_ranges(), (1, 2)):
        xs = range(x.least, x.greatest + 1)  # type: ignore[arg-type]
        ws = range(w.least, w.greatest + 1)  # type: ignore[arg-type]
        products = sorted({a * b for a in xs for b in ws})
        assert dot_range(k, x, w) == sums([products] * k), (x, w, k)
    for x, w in itertools.product(small_ranges()[::4], small_ranges()[::5]):
        xs = range(x.least, x.greatest + 1)  # type: ignore[arg-type]
        ws = range(w.least, w.greatest + 1)  # type: ignore[arg-type]
        products = sorted({a * b for a in xs for b in ws})
        assert dot_range(3, x, w) == sums([products] * 3), (x, w)


def test_column_range_holds_every_column_and_partial_sum() -> None:
    columns = list(itertools.product(range(-2, 3), repeat=3))
    for x in small_ranges():
        xs = list(range(x.least, x.greatest + 1))  # type: ignore[arg-type]
        found = [sums([[a * weight for a in xs] for weight in column]) for column in columns]
        sums_ = [(sum(v for v in column if v > 0), sum(v for v in column if v < 0)) for column in columns]
        for start in range(0, len(columns), 17):
            chosen = slice(start, start + 9)
            expected = (min(f[0] for f in found[chosen]), max(f[1] for f in found[chosen]))
            assert column_range(x, sums_[chosen]) == expected, (x, start)


def test_gemm_result_is_its_range_and_its_smallest_datatype() -> None:
    x, w = Range(-128, 127), Range(-1, 1)
    assert gemm_result(x, w, 4) == (IntType(11, True), Range(-512, 512))
    assert gemm_result(Range(0, 255), Range(0, 255), 1) == (IntType(16, False), Range(0, 65025))
    # Known weights: two columns, (2, 1, -1) and (0, -2, 0), against activations in [0, 3];
    # every dot product and partial sum enumerated, the width searched.
    weights = [(2, 1, -1), (0, -2, 0)]
    found = [sums([[a * w for a in range(4)] for w in column]) for column in weights]
    least, greatest = min(f[0] for f in found), max(f[1] for f in found)
    assert (least, greatest) == (-6, 9)
    columns = [(sum(w for w in c if w > 0), sum(w for w in c if w < 0)) for c in weights]
    assert gemm_result(Range(0, 3), columns, 3) == (
        IntType(fewest_bits(least, greatest, True), True),
        Range(least, greatest),
    )
    # A result on a power-of-two edge is given the exact datatype.
    assert gemm_result(Range(0, 1), Range(0, 255), 1)[0] == IntType(8, False)
    assert gemm_result(Range(0, 1), Range(0, 256), 1)[0] == IntType(9, False)
    assert gemm_result(Range(-1, 0), Range(0, 128), 1)[0] == IntType(8, True)
    assert gemm_result(Range(-1, 0), Range(0, 129), 1)[0] == IntType(9, True)


def test_rules_refuse_operands_outside_their_domain() -> None:
    cases: list[Callable[[], object]] = [
        lambda: integer_width(3, 2, True),
        lambda: integer_width(-1, 2, False),
        lambda: range_dtype(1, 0),
        lambda: operand_range(Range(Fraction(1, 2), 1)),
        lambda: operand_range(Range(0, 1, nan=True)),
        lambda: dot_range(0, Range(0, 1), Range(0, 1)),
        lambda: column_range(Range(0, 1), []),
        lambda: column_range(Range(0, 1), [(-1, 0)]),
        lambda: column_range(Range(0, 1), [(0, 1)]),
    ]
    for case in cases:
        with pytest.raises(ValueError):
            case()


@pytest.mark.parametrize("least,greatest", [(0, 2**200 - 1), (-(2**200), 2**200 - 1), (0, 2**4096)])
def test_widths_remain_exact_past_machine_integer_sizes(least: int, greatest: int) -> None:
    signed = least < 0
    bits = fewest_bits(least, greatest, signed)
    assert integer_width(least, greatest, signed) == bits
    if bits <= 4096:
        assert range_dtype(least, greatest) == IntType(bits, signed)
    else:
        with pytest.raises(ValueError, match="past the datatype envelope"):
            range_dtype(least, greatest)


@pytest.mark.parametrize("least,greatest", [(0, 2**4096 - 1), (-(2**4095), 2**4095 - 1)])
def test_datatype_selection_at_the_envelope_holds_its_range(least: int, greatest: int) -> None:
    expected = Range(least, greatest)
    found = range_dtype(least, greatest)
    assert found.bitwidth() == 4096
    assert holds(found, expected)
    assert found.contains(least) and found.contains(greatest)
    weights = [expected, [(greatest, least)]]
    for weight in weights:
        dtype, result = gemm_result(Range(0, 1), weight, 1)
        assert dtype == found and result == expected
        assert holds(dtype, result)
        assert dtype.contains(result.least) and dtype.contains(result.greatest)


@pytest.mark.parametrize("least,greatest", [(0, 2**4096), (-(2**4095) - 1, 0), (-(2**4095), 2**4095)])
def test_datatype_selection_refuses_the_first_excluded_width(least: int, greatest: int) -> None:
    assert integer_width(least, greatest, least < 0) == 4097
    cases: list[Callable[[], object]] = [
        lambda: range_dtype(least, greatest),
        lambda: gemm_result(Range(0, 1), Range(least, greatest), 1),
        lambda: gemm_result(Range(0, 1), [(greatest, least)], 1),
    ]
    for case in cases:
        with pytest.raises(ValueError) as refused:
            case()
        assert f"[{least}, {greatest}]" in str(refused.value)
        assert "past the datatype envelope of 4096 bits" in str(refused.value)


@pytest.mark.parametrize("length", [True, False, 1.5, "1", None, 0, -1])
@pytest.mark.parametrize("known", [False, True])
def test_gemm_length_is_validated_for_both_weight_forms(length: object, known: bool) -> None:
    weights = [(1, 0)] if known else Range(0, 1)
    refusal = ValueError if type(length) is int else TypeError
    with pytest.raises(refusal):
        gemm_result(Range(0, 1), weights, length)


@pytest.mark.parametrize("value", [None, 1, (0, 1), "range"])
def test_range_operands_require_a_range(value: object) -> None:
    cases: list[Callable[[], object]] = [
        lambda: operand_range(value),
        lambda: dot_range(1, value, Range(0, 1)),
        lambda: dot_range(1, Range(0, 1), value),
        lambda: column_range(value, [(1, 0)]),
        lambda: gemm_result(value, Range(0, 1), 1),
        lambda: gemm_result(value, [(1, 0)], 1),
    ]
    for case in cases:
        with pytest.raises(TypeError):
            case()


@pytest.mark.parametrize("columns", [None, 1, "columns"])
def test_known_columns_require_a_sequence(columns: object) -> None:
    with pytest.raises(TypeError):
        column_range(Range(0, 1), columns)
    with pytest.raises(TypeError):
        gemm_result(Range(0, 1), columns, 1)


def test_wide_dot_and_column_ranges() -> None:
    edge = 2**100
    x, w = Range(-edge, edge - 1), Range(-3, 2)
    assert dot_range(3, x, w) == (-9 * (edge - 1), 9 * edge)
    assert column_range(x, [(edge, -edge), (0, 0)]) == (-edge * (2 * edge - 1), edge * (2 * edge - 1))
    dtype, result = gemm_result(x, w, 3)
    assert result == Range(*dot_range(3, x, w))
    assert dtype == range_dtype(*result.integers())


@pytest.mark.parametrize("least,greatest", [(0.0, 1), (0, 1.0), (False, 1), (0, "1")])
def test_width_and_dtype_require_python_integer_ends(least: object, greatest: object) -> None:
    with pytest.raises(TypeError):
        integer_width(least, greatest, True)
    with pytest.raises(TypeError):
        range_dtype(least, greatest)


@pytest.mark.parametrize("length", [True, 1.5, "1"])
def test_dot_length_requires_a_python_integer(length: object) -> None:
    with pytest.raises(TypeError):
        dot_range(length, Range(0, 1), Range(0, 1))


@pytest.mark.parametrize("column", [(1.5, 0), (1, -0.5), (True, 0)])
def test_column_sums_require_python_integers(column: tuple[object, object]) -> None:
    with pytest.raises(TypeError):
        column_range(Range(0, 1), [column])


def test_signedness_is_a_boolean() -> None:
    with pytest.raises(TypeError):
        integer_width(0, 1, 1)
