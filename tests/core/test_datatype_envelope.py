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

"""Datatype names: canonical names, source spellings, an explicit exponent bias of 0, and
the deprecation of names a later release refuses.

Each canonical name is checked against ``datatype_oracle``'s definition of it.
"""

from __future__ import annotations

import pytest

import datatype_oracle as oracle
import warnings
from fractions import Fraction

from qonnx.core.datatype import ArbPrecFloatType, DataType, DataTypeWarning, resolve_datatype

INTEGERS = [
    f"{prefix}INT{bits}" for prefix in ("", "U") for bits in (*range(1, 66), 128, 4096) if (prefix, bits) != ("U", 1)
]
FIXED = [f"FIXED<{w},{i}>" for w in (1, 8, 32, 53, 54, 55, 64) for i in (-8, 0, 1, w // 2, w - 1) if i < w] + [
    "FIXED<1,-1073>",
    "FIXED<1075,1>",
    "FIXED<1025,1024>",
    "FIXED<2098,1024>",
]
FLOATS = ["FLOAT16", "FLOAT32", "FLOAT<5,10,15>", "FLOAT<4,3,7>", "FLOAT<5,2,15>", "FLOAT<1,0,0>"]
FLOATS += ["FLOAT<8,7,127>", "FLOAT<5,10,7>", "FLOAT<5,10,0>", "FLOAT<11,51,1024>"]
NAMES = ["BINARY", "BIPOLAR", "TERNARY"] + INTEGERS + FIXED + FLOATS
NAMES += ["SCALEDINT<1>", "SCALEDINT<8>", "SCALEDINT<4096>"]

# Fixed-point datatypes whose signed() reads a min() that overflows a float today.
VIEW_OVERFLOWS = {"FIXED<1075,1>", "FIXED<1025,1024>", "FIXED<2098,1024>"}


@pytest.mark.parametrize("name", NAMES)
def test_a_canonical_name_resolves_to_itself_with_its_definition(name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataTypeWarning)
        found = resolve_datatype(name, canonical=True)
        assert DataType[name] == found and hash(DataType[name]) == hash(found)
    item = oracle.spec(name)
    assert found.get_canonical_name() == name == found.name == str(found) == repr(found)
    assert found.bitwidth() == item.bits
    if name not in VIEW_OVERFLOWS:
        assert found.signed() is (item.kind != "int" or item.signed)
    assert found.is_integer() is (item.kind in ("int", "bipolar", "ternary"))


SPELLINGS = {
    "UINT1": "BINARY",
    "FLOAT<5,10>": "FLOAT<5,10,15>",
    "FLOAT<8,7>": "FLOAT<8,7,127>",
    "FLOAT<1,0>": "FLOAT<1,0,0>",
    "FIXED<8, 4>": "FIXED<8,4>",
    "INT08": "INT8",
}


@pytest.mark.parametrize(("spelling", "name"), SPELLINGS.items())
def test_a_source_spelling_resolves_and_is_not_canonical(spelling: str, name: str) -> None:
    assert DataType[spelling] == resolve_datatype(name, canonical=True)
    assert DataType[spelling].get_canonical_name() == name
    with pytest.raises(KeyError):
        resolve_datatype(spelling, canonical=True)


def test_an_explicit_exponent_bias_of_zero_is_kept() -> None:
    """FLOAT<5,10,0> read back as FLOAT<5,10,15>: a bias of 0 was taken for "unset"."""
    zero = DataType["FLOAT<5,10,0>"]
    assert zero.exponent_bias() == 0 and zero.get_canonical_name() == "FLOAT<5,10,0>"
    assert zero != DataType["FLOAT<5,10>"]
    assert ArbPrecFloatType(4, 3, 0).get_canonical_name() == "FLOAT<4,3,0>"
    assert ArbPrecFloatType(4, 3).get_canonical_name() == "FLOAT<4,3,7>"
    for name in ("FLOAT<5,10,0>", "FLOAT<1,0,0>", "FLOAT<3,1,0>"):
        found, item = DataType[name], oracle.spec(name)
        assert found.bounds() == oracle.bounds(item)
        values = sorted(oracle.mini_values(item.exponent_bits, item.mantissa_bits, item.bias))
        probes = values[:40] + values[-40:] + values[:: max(1, len(values) // 200)]
        probes += [Fraction(1, 3), values[-1] * 2, values[len(values) // 2 + 1] / 3]
        assert [found.contains(v) for v in probes] == [oracle.member(item, v) for v in probes]


DEPRECATED = [
    # zero and negative widths and parameters
    "INT0",
    "UINT0",
    "SCALEDINT<0>",
    "FIXED<0,-1>",
    "INT-3",
    "UINT-3",
    "SCALEDINT<-1>",
    "FIXED<-8,-9>",
    "FLOAT<0,3>",
    "FLOAT<5,-1>",
    # past the envelope
    "FIXED<8,-18446744073709551608>",
    "INT5000",
    "UINT4097",
    "SCALEDINT<4097>",
    "FIXED<4097,1>",
    "FIXED<1,-1074>",
    "FIXED<1026,1025>",
    "FLOAT<12,3,2047>",
    "FLOAT<11,52,1022>",
    "FLOAT<5,2000,15>",
    "INT18446744073709551616",
    "FLOAT<3,4095>",
    "FLOAT<1,53,-1022>",
    "FLOAT<1,100,-1022>",
]


@pytest.mark.parametrize("name", DEPRECATED)
def test_a_deprecated_name_still_resolves_with_a_warning(name: str) -> None:
    with pytest.warns(DataTypeWarning):
        found = DataType[name]
    with pytest.warns(DataTypeWarning):
        assert type(resolve_datatype(name)) is type(found)
    with pytest.raises(KeyError):
        resolve_datatype(name, canonical=True)


def test_the_warning_is_a_future_warning_at_the_caller() -> None:
    """Shown by default (not hidden as a DeprecationWarning is), and attributed to the line
    that resolved the name."""
    assert issubclass(DataTypeWarning, FutureWarning)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        DataType["INT0"]
        resolve_datatype("UINT0")
    assert [w.category for w in caught] == [DataTypeWarning, DataTypeWarning]
    assert all(w.filename == __file__ for w in caught)


def test_names_within_the_envelope_resolve_without_a_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataTypeWarning)
        for name in NAMES + list(SPELLINGS) + ["FLOAT<1,52,-1022>", "FIXED<2098,1024>", "UINT4096"]:
            DataType[name]


@pytest.mark.parametrize("name", ["FLOAT64", "int8", "", "BIPOLAR2"])
def test_unknown_names_are_refused_as_before(name: str) -> None:
    for canonical in (False, True):
        with pytest.raises(KeyError):
            resolve_datatype(name, canonical=canonical)


@pytest.mark.parametrize("name", ["FIXED<8>", "FIXED<8,8>", "FIXED<8,9>", "FIXED<-8,4>", "INT 8", "INT٨"])
def test_a_canonical_lookup_refuses_a_malformed_name_with_key_error(name: str) -> None:
    with pytest.raises(KeyError):
        resolve_datatype(name, canonical=True)


def test_a_long_run_of_zeros_is_read_by_its_significant_digits() -> None:
    """Python refuses to read more than 4300 digits as an int; a name's leading zeros no
    longer reach that limit."""
    assert DataType["INT" + "0" * 5000 + "8"] == DataType["INT8"]
    assert DataType["FIXED<" + "0" * 5000 + "8," + "0" * 5000 + "4>"] == DataType["FIXED<8,4>"]
    assert DataType["FIXED<8,-" + "0" * 5000 + "4>"] == DataType["FIXED<8,-4>"]
    for name in ("INT" + "0" * 5000, "FIXED<" + "0" * 5000 + ",-1>"):
        with pytest.warns(DataTypeWarning):
            assert DataType[name].bitwidth() == 0
    with pytest.raises(ValueError):
        DataType["INT" + "1" * 5000]
