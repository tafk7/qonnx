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

"""Numeric refusals preserve lookup signals and describe unsupported values."""

from __future__ import annotations

import pytest

from collections.abc import Callable
from fractions import Fraction

import qonnx.core.datatype as datatype_module
from qonnx.core.datatype import (
    DataType,
    DataTypeNameMalformed,
    DataTypeNameUnknown,
    FloatFormat,
    NumericValueRefused,
    resolve_datatype,
    round_to,
)
from qonnx.core.findings import Finding
from qonnx.core.range_rules import column_range, dot_range, gemm_result, integer_width, range_dtype
from qonnx.core.ranges import Range, full, holds, hull


@pytest.mark.parametrize(
    "name,leaf,code",
    [
        ("FOO", DataTypeNameUnknown, "name-unknown"),
        ("FLOAT<1>", DataTypeNameUnknown, "name-unknown"),
        ("INTx", DataTypeNameMalformed, "name-malformed"),
        ("UINT", DataTypeNameMalformed, "name-malformed"),
        ("FIXED<8,x>", DataTypeNameMalformed, "name-malformed"),
        ("SCALEDINT<x>", DataTypeNameMalformed, "name-malformed"),
        ("FLOAT<a,b>", DataTypeNameMalformed, "name-malformed"),
        ("FIXED<8>", DataTypeNameMalformed, "name-malformed"),
        ("FIXED<8,8>", DataTypeNameMalformed, "name-malformed"),
        ("FIXED<8,9>", DataTypeNameMalformed, "name-malformed"),
        ("FIXED<-8,4>", DataTypeNameMalformed, "name-malformed"),
        ("FLOAT<1025,1>", DataTypeNameMalformed, "beyond-envelope"),
    ],
)
def test_name_refusals_keep_their_lookup_or_value_base(name: str, leaf: type, code: str) -> None:
    with pytest.raises(leaf) as caught:
        DataType[name]
    assert type(caught.value) is leaf
    assert caught.value.findings[0].code == code
    assert dict(caught.value.findings[0].details)["name"] == name
    assert not isinstance(caught.value, (AssertionError, IndexError, OverflowError))


@pytest.mark.parametrize(
    "name,reason",
    [
        ("FOO", "name-unknown"),
        ("INTx", "name-malformed"),
        ("FIXED<8>", "name-malformed"),
        ("FIXED<8,8>", "name-malformed"),
        ("FLOAT<1025,1>", "beyond-envelope"),
        ("INT0", "width-zero"),
        ("INT-3", "name-malformed"),
        ("INT4097", "beyond-envelope"),
        ("INT18446744073709551616", "beyond-envelope"),
        ("UINT1", "BINARY"),
    ],
)
def test_canonical_refusals_remain_key_errors_with_the_plain_reason(name: str, reason: str) -> None:
    with pytest.raises(DataTypeNameUnknown) as caught:
        resolve_datatype(name, canonical=True)
    assert isinstance(caught.value, KeyError) and not isinstance(caught.value, ValueError)
    assert caught.value.findings[0].code == "name-not-canonical"
    assert dict(caught.value.findings[0].details) == {"name": name, "reason": reason}
    if name != "UINT1":
        assert caught.value.__cause__ is not None


@pytest.mark.parametrize("name", ["FIXED<8,8>", "INTx"])
def test_resolved_name_refusals_use_the_public_owner_in_their_default_message(name: str) -> None:
    with pytest.raises(DataTypeNameMalformed) as caught:
        resolve_datatype(name)
    assert isinstance(caught.value, ValueError)
    assert caught.value.findings[0].owner == "resolve_datatype"
    assert str(caught.value) == str(caught.value.findings[0])


@pytest.mark.parametrize("explicit", [False, True])
def test_parser_owner_replacement_keeps_identity_and_explicit_messages(
    explicit: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    finding = Finding("name-malformed", "DataType", "bad parameters")
    # An explicit message may equal the original default; it still stays explicit.
    refused = DataTypeNameMalformed((finding,), str(finding) if explicit else None)

    def refuse(name: str) -> datatype_module.BaseDataType:
        raise refused

    monkeypatch.setattr(datatype_module, "_construct_unchecked", refuse)
    with pytest.raises(DataTypeNameMalformed) as caught:
        resolve_datatype("FIXED<8,8>")
    assert caught.value is refused and isinstance(refused, ValueError)
    assert refused.findings[0].owner == "resolve_datatype"
    assert str(refused) == (str(finding) if explicit else str(refused.findings[0]))


@pytest.mark.parametrize(
    "call,code,owner",
    [
        (lambda: DataType["SCALEDINT<8>"].bounds(), "undefined", "DataType.bounds"),
        (lambda: DataType["INT4"].codes(9), "not-member", "DataType.codes"),
        (lambda: DataType["TERNARY"].values(2), "code-outside", "DataType.values"),
        (lambda: DataType["INT4"].values(-1), "code-outside", "DataType.values"),
        (lambda: round_to(DataType["INT8"], 1), "undefined", "round_to"),
        (lambda: Range(2, 1), "range-invalid", "Range"),
        (lambda: Range(Fraction(1, 3), 1), "range-invalid", "endpoint"),
        (lambda: FloatFormat(-1, 0, 1), "format-invalid", "FloatFormat"),
        (lambda: integer_width(-1, 1, False), "rule-domain", "integer_width"),
        (lambda: dot_range(0, Range(1, 2), Range(1, 2)), "rule-domain", "dot_range"),
        (lambda: column_range(Range(0, 1), []), "rule-domain", "column_range"),
        (lambda: range_dtype(0, 1 << 4096), "beyond-envelope", "range_dtype"),
        (lambda: gemm_result(Range(0, 1 << 2048), Range(0, 1 << 2048), 1), "beyond-envelope", "range_dtype"),
        (lambda: DataType["INT65"].to_numpy_dt(), "undefined", "DataType.to_numpy_dt"),
        (lambda: DataType["SCALEDINT<8>"].min(), "undefined", "DataType.min"),
        (lambda: DataType["FLOAT32"].get_num_possible_values(), "undefined", "DataType.get_num_possible_values"),
        (lambda: DataType.get_smallest_possible(2**64), "undefined", "DataType.get_smallest_possible"),
        (lambda: DataType["FLOAT<5,10>"].get_hls_datatype_str(), "undefined", "DataType.get_hls_datatype_str"),
    ],
)
def test_numeric_value_refusals_carry_a_code_and_public_owner(call: Callable[[], object], code: str, owner: str) -> None:
    with pytest.raises(NumericValueRefused) as caught:
        call()
    assert isinstance(caught.value, ValueError)
    assert caught.value.findings[0].code == code and caught.value.findings[0].owner == owner


@pytest.mark.parametrize("call", [lambda: full("INT8"), lambda: holds(DataType["FLOAT32"], None), lambda: hull(None)])
def test_wrong_range_input_types_raise_type_error(call: Callable[[], object]) -> None:
    with pytest.raises(TypeError):
        call()
