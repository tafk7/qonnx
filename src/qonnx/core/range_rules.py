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

"""The value rules: what an operation's result holds, from what its operands hold.

Each is exact, in Python integers, at any width. They take ranges, integers and column
sums; a caller unpacks its own types into them. Datatype selection refuses a result
past the datatype envelope. Operands outside a rule's domain (an
empty range, a range that is not of integers, a negative range for an unsigned width)
raise ``ValueError``. Wrong Python types raise ``TypeError``.
"""

from __future__ import annotations

from collections.abc import Sequence

from qonnx.core.datatype import _MAX_BITS, IntType, NumericValueRefused
from qonnx.core.ranges import Range

__all__ = ["integer_width", "range_dtype", "operand_range", "dot_range", "column_range", "gemm_result"]


def _ends(least: int, greatest: int) -> None:
    if type(least) is not int or type(greatest) is not int:
        raise TypeError(f"a rule's ends are ints, not {least!r} and {greatest!r}")
    if least > greatest:
        raise NumericValueRefused.of("rule-domain", "integer_width", f"[{least}, {greatest}] is an empty range")


def integer_width(least: int, greatest: int, signed: bool) -> int:
    """The fewest bits an integer encoding of that signedness needs for every integer of
    ``[least, greatest]``; one at least."""
    _ends(least, greatest)
    if type(signed) is not bool:
        raise TypeError(f"a width's signedness is a bool, not {signed!r}")
    if signed:
        return 1 + max(
            greatest.bit_length() if greatest >= 0 else 0,
            (~least).bit_length() if least < 0 else 0,
        )
    if least < 0:
        raise NumericValueRefused.of(
            "rule-domain", "integer_width", f"[{least}, {greatest}] has negative values: no UINT holds it"
        )
    return max(1, greatest.bit_length())


def range_dtype(least: int, greatest: int) -> IntType:
    """The smallest integer datatype of every integer in ``[least, greatest]``: ``UINT``
    (``BINARY`` at one bit) when none is negative, otherwise ``INT``. Raises ``ValueError``
    when that range needs a width past the datatype envelope."""
    _ends(least, greatest)
    signed = least < 0
    bits = integer_width(least, greatest, signed)
    if bits > _MAX_BITS:
        raise NumericValueRefused.of(
            "beyond-envelope",
            "range_dtype",
            f"[{least}, {greatest}] needs {bits} bits, past the datatype envelope of {_MAX_BITS} bits",
        )
    return IntType(bits, signed)


def operand_range(value: Range) -> tuple[int, int]:
    """The least and greatest value an operand of ``value`` takes in a dot product: its
    integer range widened to hold 0, so every partial sum lies within the whole's."""
    if not isinstance(value, Range):
        raise TypeError(f"an operand's range is a Range, not {value!r}")
    least, greatest = value.integers()
    return min(least, 0), max(greatest, 0)


def _length(k: int) -> None:
    if type(k) is not int:
        raise TypeError(f"a dot product's length is an int, not {k!r}")
    if k < 1:
        raise NumericValueRefused.of("rule-domain", "dot_range", f"a dot product has a positive length, not {k!r}")


def dot_range(k: int, x: Range, w: Range) -> tuple[int, int]:
    """The least and greatest dot product of ``k`` values of ``x`` with ``k`` of ``w``
    (each its ``operand_range``)."""
    _length(k)
    activation, weight = operand_range(x), operand_range(w)
    products = tuple(a * b for a in activation for b in weight)
    return k * min(products), k * max(products)


def column_range(x: Range, columns: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """The least and greatest dot product of any column of known weights with values of
    ``x``, each column given by the sum ``P`` of its positive weights and ``N`` of its
    negative ones.

    Against values in ``[lo, hi]`` (``lo <= 0 <= hi``, ``operand_range``) a column spans
    ``[lo * P + hi * N, hi * P + lo * N]``, each term at its extremes independently; every
    partial sum lies within it too, since each term's range holds 0.
    """
    low, high = operand_range(x)
    if not isinstance(columns, Sequence) or isinstance(columns, (str, bytes)):
        raise TypeError(f"columns are a sequence of weight sums, not {columns!r}")
    if not columns:
        raise NumericValueRefused.of("rule-domain", "column_range", "no column to range over")
    for positive, negative in columns:
        if type(positive) is not int or type(negative) is not int:
            raise TypeError(f"a column's sums are ints, not {positive!r}, {negative!r}")
        if positive < 0 or negative > 0:
            raise NumericValueRefused.of(
                "rule-domain", "column_range", f"a column's sums are P >= 0 and N <= 0, not {positive}, {negative}"
            )
    return (
        min(low * positive + high * negative for positive, negative in columns),
        max(high * positive + low * negative for positive, negative in columns),
    )


def gemm_result(x: Range, w: Range | Sequence[tuple[int, int]], k: int) -> tuple[IntType, Range]:
    """A matrix product's result: its datatype (``range_dtype``) and its range. With
    known weights, given as their columns' sums, the columns' range (``column_range``);
    otherwise every dot product of ``k`` values of ``x`` and ``w`` (``dot_range``).
    In both cases ``k`` is a positive Python integer; datatype selection raises
    ``ValueError`` for a range past the datatype envelope."""
    _length(k)
    if isinstance(w, Range):
        least, greatest = dot_range(k, x, w)
    else:
        least, greatest = column_range(x, w)
    return range_dtype(least, greatest), Range(least, greatest)
