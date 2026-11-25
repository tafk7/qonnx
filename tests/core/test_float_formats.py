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

"""The four float formats round as IEEE 754 does, against their bit patterns.

``FloatFormat`` (``largest`` and ``nearest``) is checked for ``BINARY16``, ``BINARY32``,
``BINARY64`` and ``BFLOAT16``, each against its own encoding read by ``struct`` (bfloat16 as
the high half of a binary32 pattern): adjacent patterns' values, their midpoints (a tie
goes to the even pattern) and values a hair either side, subnormals and the overflow
threshold; and the format's domain.
"""

from __future__ import annotations

import pytest

import math
import random
import struct
from collections.abc import Callable
from fractions import Fraction

from qonnx.core.datatype import BFLOAT16, BINARY16, BINARY32, BINARY64, FloatFormat


def reader(code: str, width: int, shift: int = 0) -> Callable[[int], Fraction]:
    """A non-negative pattern's value, as ``struct`` decodes the format's bytes."""

    def value(pattern: int) -> Fraction:
        raw = (pattern << shift).to_bytes(width, "little")
        return Fraction(struct.unpack(code, raw)[0])

    return value


#: Each format, its pattern decoder, its greatest finite pattern and its least normal one.
FORMATS = {
    "binary16": (BINARY16, reader("<e", 2), 0x7BFF, 0x0400),
    "binary32": (BINARY32, reader("<f", 4), 0x7F7FFFFF, 0x00800000),
    "binary64": (BINARY64, reader("<d", 8), 0x7FEFFFFFFFFFFFFF, 0x0010000000000000),
    "bfloat16": (BFLOAT16, reader("<f", 4, 16), 0x7F7F, 0x0080),
}


@pytest.mark.parametrize("name", FORMATS)
def test_nearest_is_the_formats_own_rounding(name: str) -> None:
    fmt, value, greatest, normal = FORMATS[name]
    assert fmt.largest == value(greatest)
    assert Fraction(2) ** fmt.emin == value(normal)
    rng = random.Random(name)
    patterns = [0, 1, 2, 3, normal - 1, normal, normal + 1, greatest - 2, greatest - 1]
    patterns += [rng.randrange(greatest) for _ in range(400)]
    for pattern in patterns:
        low, high = value(pattern), value(pattern + 1)
        even = low if pattern % 2 == 0 else high
        tie, hair = (low + high) / 2, (high - low) / 2**70
        for sign in (1, -1):
            assert fmt.nearest(sign * low) == sign * low
            assert fmt.nearest(sign * tie) == sign * even, (name, pattern)
            assert fmt.nearest(sign * (tie - hair)) == sign * low
            assert fmt.nearest(sign * (tie + hair)) == sign * high
    spacing = value(greatest) - value(greatest - 1)
    threshold = value(greatest) + spacing / 2
    assert fmt.nearest(threshold) == math.inf and fmt.nearest(-threshold) == -math.inf
    assert fmt.nearest(threshold - Fraction(1, 2**70)) == value(greatest)
    assert fmt.nearest(value(1) / 2) == 0  # half the least subnormal: a tie to zero, even
    assert fmt.nearest(value(1) / 2 + Fraction(1, 2**2000)) == value(1)
    assert fmt.nearest(math.inf) == math.inf


def test_a_format_is_a_binary_float_format() -> None:
    for mantissa, emin, emax in ((-1, -14, 15), (10, 15, -14)):
        with pytest.raises(ValueError):
            FloatFormat(mantissa, emin, emax)
    with pytest.raises(TypeError):
        FloatFormat(10, -14.0, 15)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        FloatFormat(True, -14, 15)
    assert FloatFormat(0, 0, 0).largest == 1
