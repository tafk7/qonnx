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

"""A tensor's range on a model (qonnx.core.tensor_ranges): its stored text, the
datatype fact it is part of, its refusals and diagnostics, and how it moves with
its tensor."""

from __future__ import annotations

import pytest

import math
import numpy as np
import warnings
from fractions import Fraction
from onnx import StringStringEntryProto, TensorProto, helper

from qonnx.core import metadata, tensor_ranges
from qonnx.core.datatype import DataType
from qonnx.core.metadata import MetadataError
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.ranges import Range
from qonnx.core.tensor_ranges import RANGE, TENSOR, RangeContradiction, one_pass
from qonnx.transformation.general import RemoveUnusedTensors
from qonnx.util.basic import get_by_name, qonnx_make_model

F = TensorProto.FLOAT
W = np.array([[1, -2], [3, 0], [-1, 1]], dtype=np.float32)


def chain() -> ModelWrapper:
    """x (INT4, 1x3) -> MatMul by w (INT3) -> m -> Add b (INT4) -> y."""
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "w"], ["m"], name="MatMul_0"),
            helper.make_node("Add", ["m", "b"], ["y"], name="Add_0"),
        ],
        "chain",
        [helper.make_tensor_value_info("x", F, [1, 3])],
        [helper.make_tensor_value_info("y", F, [1, 2])],
        value_info=[helper.make_tensor_value_info("m", F, [1, 2])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_initializer("w", W)
    model.set_initializer("b", np.array([[5, -5]], dtype=np.float32))
    for name, dtype in (("x", "INT4"), ("w", "INT3"), ("b", "INT4")):
        model.set_tensor_datatype(name, DataType[dtype])
    return model


def single() -> ModelWrapper:
    """x (FLOAT, [1]) as a graph with no node."""
    graph = helper.make_graph([], "annotations", [helper.make_tensor_value_info("x", F, [1])], [])
    return ModelWrapper(qonnx_make_model(graph))


def entries(model: ModelWrapper, tensor: str) -> dict[str, str]:
    annotation = get_by_name(model.graph.quantization_annotation, tensor, "tensor_name")
    return {entry.key: entry.value for entry in annotation.quant_parameter_tensor_names}


def props(model: ModelWrapper, tensor: str) -> list[tuple[str, str]]:
    annotation = get_by_name(model.graph.quantization_annotation, tensor, "tensor_name")
    return [(entry.key, entry.value) for entry in annotation.quant_parameter_tensor_names]


def store(model: ModelWrapper, tensor: str, text: str) -> None:
    """Store ``text`` as the tensor's range, past every writer's check."""
    annotation = get_by_name(model.graph.quantization_annotation, tensor, "tensor_name")
    for key, value in (("qonnx.tensor/@version", "1"), ("qonnx.tensor/@follow", "true"), ("qonnx.tensor/range", text)):
        annotation.quant_parameter_tensor_names.append(StringStringEntryProto(key=key, value=value))


# -- the stored text


@pytest.mark.parametrize(
    "value, text",
    [
        (Range(-(2**70), 2**70 + 1), '{"nan":false,"range":[-1180591620717411303424,1180591620717411303425]}'),
        (Range(Fraction(-3, 4), math.inf), '{"nan":false,"range":["-3*2^-2","inf"]}'),
        (Range(-math.inf, 0.5, nan=True), '{"nan":true,"range":["-inf","1*2^-1"]}'),
        (Range(-2.0, Fraction(6, 2)), '{"nan":false,"range":[-2,3]}'),
    ],
)
def test_a_range_is_stored_as_canonical_json_and_read_back_exactly(value: Range, text: str) -> None:
    model = chain()
    datatype = DataType["INT80"] if value.is_integer() else DataType["FLOAT32"]
    model.set_tensor_datatype("m", datatype, range=value)
    assert entries(model, "m") == {
        "finn_datatype": datatype.name,
        "qonnx.tensor/@version": "1",
        "qonnx.tensor/@follow": "true",
        "qonnx.tensor/range": text,
    }
    loaded = ModelWrapper(model.model.SerializeToString())
    assert loaded.get_tensor_range("m") == value
    assert loaded.get_stated_tensor_range("m") == value
    assert loaded.get_metadata(RANGE, tensor="m") == value


@pytest.mark.parametrize(
    "name, value, text",
    [
        ("INT8", Range(-3, 5), '{"nan":false,"range":[-3,5]}'),
        ("FLOAT32", Range(Fraction(-3, 8), Fraction(7, 16)), '{"nan":false,"range":["-3*2^-3","7*2^-4"]}'),
        ("FLOAT32", Range(0, float("inf"), True), '{"nan":true,"range":[0,"inf"]}'),
    ],
)
def test_a_saved_range_keeps_its_bytes_when_stated_again(name: str, value: Range, text: str) -> None:
    model = single()
    model.set_tensor_datatype("x", DataType[name], range=value)
    assert entries(model, "x")[RANGE.entry] == text
    saved = model.model.SerializeToString(deterministic=True)
    loaded = ModelWrapper(saved)
    assert loaded.get_stated_tensor_range("x") == value
    loaded.set_tensor_datatype("x", DataType[name], range=value)
    assert loaded.model.SerializeToString(deterministic=True) == saved


def test_a_range_stated_again_stays_in_place_among_other_annotations() -> None:
    model = chain()
    model.set_tensor_datatype("m", DataType["INT7"], range=Range(-39, 36))
    model.set_tensor_layout("m", ["N", "C"])
    before = props(model, "m")
    model.set_tensor_datatype("m", DataType["INT7"], range=Range(-39, 36))
    assert props(model, "m") == before
    model.set_tensor_datatype("m", DataType["INT7"], range=Range(-40, 36))
    assert props(model, "m") == [
        (key, '{"nan":false,"range":[-40,36]}' if key == RANGE.entry else value) for key, value in before
    ]


@pytest.mark.parametrize(
    "text",
    [
        "[-1, 2]",
        '{"nan": false, "range": [2, 1]}',
        '{"nan": false, "range": [1.0, 2]}',
        '{"nan":false,"range":[0.5,1]}',
        '{"nan": false, "range": [true, 2]}',
        '{"nan": "no", "range": [1, 2]}',
        '{"nan":0,"range":[0,1]}',
        '{"nan": false, "range": ["6*2^-2", 2]}',
        '{"nan":false,"range":["2*2^-2",1]}',
        '{"nan": false, "range": ["1*2^0", 2]}',
        '{"nan": false, "range": ["1/3", 2]}',
        '{"nan": false, "range": [1, "+inf"]}',
        '{"nan": false, "range": [1, 2, 3]}',
        '{"nan": false, "range": [1, 2], "extra": 1}',
        '{"nan": false, "range": [1, NaN]}',
        '{"nan": false, "range": [1, Infinity]}',
    ],
)
def test_a_malformed_range_is_a_metadata_error_naming_the_entry(text: str) -> None:
    model = chain()
    store(model, "x", text)
    loaded = ModelWrapper(model.model.SerializeToString())
    for read in (loaded.get_tensor_range, loaded.get_stated_tensor_range, loaded.get_tensor_annotation_range):
        with pytest.raises(MetadataError, match="qonnx.tensor/range: stored"):
            read("x")


def test_a_stored_range_its_datatype_does_not_hold_is_a_metadata_error() -> None:
    model = chain()
    model.set_metadata(RANGE, Range(-100, 100), tensor="x")  # x is INT4; written past the datatype
    with pytest.raises(MetadataError, match=r"tensor 'x': stored \[-100, 100\] is not held by its datatype INT4"):
        model.get_tensor_range("x")
    assert model.get_stated_tensor_range("x") == Range(-100, 100)


def test_a_range_that_is_not_a_range_is_refused_on_writing() -> None:
    model = chain()
    with pytest.raises(MetadataError, match="qonnx.tensor/range: cannot store"):
        model.set_tensor_datatype("m", DataType["INT8"], range=(1, 2))
    with pytest.raises(TypeError):
        model.set_tensor_datatype("m", DataType["INT8"], Range(1, 2))
    assert model.get_stated_tensor_range("m") is None


def test_a_range_without_a_datatype_is_refused() -> None:
    model = chain()
    before = model.model.SerializeToString()
    with pytest.raises(ValueError, match="a range is stated with a datatype"):
        model.set_tensor_datatype("m", None, range=Range(0, 1))
    with pytest.raises(ValueError, match="states none"):
        tensor_ranges.write(model, "m", Range(0, 1))
    assert model.model.SerializeToString() == before


# -- one fact with the datatype


def test_a_datatype_set_without_its_range_clears_the_range() -> None:
    model = chain()
    model.set_tensor_datatype("m", DataType["INT7"], range=Range(-39, 36))
    assert model.get_tensor_range("m") == Range(-39, 36)
    model.set_tensor_datatype("m", DataType["INT7"])
    assert model.get_stated_tensor_range("m") is None
    assert model.get_tensor_range("m") == Range(-64, 63)
    model.set_tensor_datatype("m", DataType["INT7"], range=Range(-39, 36))
    model.set_tensor_datatype("m", DataType["INT8"])
    assert model.get_stated_tensor_range("m") is None
    model.set_tensor_datatype("m", DataType["INT7"], range=Range(-39, 36))
    model.set_tensor_datatype("m", None)
    assert get_by_name(model.graph.quantization_annotation, "m", "tensor_name").quant_parameter_tensor_names == []


def test_a_datatype_written_over_a_malformed_range_replaces_it() -> None:
    model = chain()
    store(model, "x", '{"nan": false, "range": [2, 1]}')
    model.set_tensor_datatype("x", DataType["INT4"], range=Range(0, 3))
    assert model.get_stated_tensor_range("x") == Range(0, 3)
    store(model, "w", "malformed")
    model.set_tensor_datatype("w", DataType["INT3"])
    assert entries(model, "w") == {"finn_datatype": "INT3"}
    model.set_tensor_datatype("x", DataType["INT4"])
    assert entries(model, "x") == {"finn_datatype": "INT4"}


@pytest.mark.parametrize(
    "datatype, full",
    [
        ("INT7", Range(-64, 63)),
        ("BIPOLAR", Range(-1, 1)),
        ("FIXED<8,4>", Range(-8, Fraction(127, 16))),
        ("SCALEDINT<8>", Range(-math.inf, math.inf)),
        ("FLOAT32", Range(-math.inf, math.inf, nan=True)),
        ("FLOAT16", Range(-math.inf, math.inf, nan=True)),
    ],
)
def test_a_datatype_s_full_range_is_its_range_when_none_is_stated_and_is_not_stored(datatype: str, full: Range) -> None:
    model = chain()
    model.set_tensor_datatype("m", DataType[datatype], range=full)
    assert model.get_stated_tensor_range("m") is None
    assert entries(model, "m") == {"finn_datatype": DataType[datatype].name}
    assert model.get_tensor_range("m") == full
    assert model.get_tensor_annotation_range("m") == full


def test_without_a_range_the_annotation_is_upstream_s() -> None:
    """A datatype written, replaced or removed without a range leaves the bytes
    the setter wrote before ranges were stored with it."""
    model = chain()
    model.set_tensor_layout("m", ["N", "C"])
    model.set_tensor_datatype("m", DataType["INT7"])
    assert props(model, "m") == [("tensor_layout", "['N', 'C']"), ("finn_datatype", "INT7")]
    model.set_tensor_datatype("m", DataType["INT8"])
    assert props(model, "m") == [("tensor_layout", "['N', 'C']"), ("finn_datatype", "INT8")]
    model.set_tensor_datatype("m", None)
    assert props(model, "m") == [("tensor_layout", "['N', 'C']")]
    model.set_tensor_datatype("y", None)
    assert get_by_name(model.graph.quantization_annotation, "y", "tensor_name") is None


# -- a writer's contradiction


def test_a_writer_stating_a_range_its_datatype_does_not_hold_is_named_and_nothing_is_stored() -> None:
    model = chain()
    with pytest.warns(
        RangeContradiction,
        match=r"tensor 'm' not stated: MatMul_0 \(MatMul\) states \[0, 2\], which its datatype INT2 does not hold",
    ):
        model.set_tensor_datatype("m", DataType["INT2"], range=Range(0, 2))
    assert model.get_stated_tensor_range("m") is None
    assert model.get_tensor_datatype("m") == DataType["INT2"]


def test_a_contradiction_clears_the_range_stated_before() -> None:
    model = chain()
    model.set_tensor_datatype("m", DataType["INT7"], range=Range(-39, 36))
    with pytest.warns(RangeContradiction):
        model.set_tensor_datatype("m", DataType["INT2"], range=Range(0, 2))
    assert entries(model, "m") == {"finn_datatype": "INT2"}


def test_a_contradiction_is_named_once_per_tensor_in_a_pass() -> None:
    model = chain()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with one_pass(model):
            for _ in range(2):
                with one_pass(model):
                    model.set_tensor_datatype("m", DataType["INT2"], range=Range(0, 2))
                    model.set_tensor_datatype("y", DataType["INT2"], range=Range(0, 2))
        model.set_tensor_datatype("m", DataType["INT2"], range=Range(0, 2))
        model.set_tensor_datatype("m", DataType["INT2"], range=Range(0, 2))
    named = [str(item.message).split(" not stated")[0] for item in caught if item.category is RangeContradiction]
    assert (
        named
        == ["qonnx.tensor/range of tensor 'm'", "qonnx.tensor/range of tensor 'y'"]
        + ["qonnx.tensor/range of tensor 'm'"] * 2
    )
    assert all(item.filename == __file__ for item in caught)


def test_a_contradiction_raised_as_an_error_is_named_again_on_a_retry() -> None:
    model = chain()
    with warnings.catch_warnings(), one_pass(model):
        warnings.simplefilter("error", RangeContradiction)
        for _ in range(2):
            with pytest.raises(RangeContradiction):
                model.set_tensor_datatype("m", DataType["INT2"], range=Range(0, 2))


def test_a_range_refused_or_raised_as_an_error_leaves_the_model_unchanged() -> None:
    """The range is checked before the datatype is written: neither is written."""
    model = chain()
    model.set_tensor_datatype("m", DataType["INT16"], range=Range(-300, 300))
    before = model.model.SerializeToString(deterministic=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RangeContradiction)
        with pytest.raises(RangeContradiction):
            model.set_tensor_datatype("m", DataType["INT8"], range=Range(-300, 300))
    assert model.model.SerializeToString(deterministic=True) == before
    assert model.get_tensor_datatype("m") == DataType["INT16"]
    assert model.get_tensor_range("m") == Range(-300, 300)
    for name in ("m", "y"):
        before = model.model.SerializeToString(deterministic=True)
        with pytest.raises(MetadataError, match="qonnx.tensor/range: cannot store"):
            model.set_tensor_datatype(name, DataType["INT8"], range=(1, 2))
        assert model.model.SerializeToString(deterministic=True) == before
    assert get_by_name(model.graph.quantization_annotation, "y", "tensor_name") is None


def test_a_pass_is_one_per_model() -> None:
    first, second = chain(), chain()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with one_pass(first), one_pass(second):
            for model in (first, second, first, second):
                model.set_tensor_datatype("m", DataType["INT2"], range=Range(0, 2))
    assert len([item for item in caught if item.category is RangeContradiction]) == 2


# -- the stated range and an initializer's values


def test_stated_and_initializer_ranges_remain_distinct() -> None:
    model = single()
    model.set_tensor_datatype("x", DataType["INT8"], range=Range(-3, 5))
    model.set_initializer("x", np.array([2], dtype=np.float32))
    assert model.get_tensor_range("x") == Range(2, 2)
    assert model.get_tensor_annotation_range("x") == Range(-3, 5)
    assert model.get_stated_tensor_range("x") == Range(-3, 5)
    model.set_tensor_datatype("x", DataType["INT8"])
    assert model.get_stated_tensor_range("x") is None
    assert model.get_tensor_annotation_range("x") == Range(-128, 127)


def test_a_tensor_without_a_datatype_reads_its_container_s_range() -> None:
    model = chain()
    assert model.get_stated_tensor_range("m") is None
    assert model.get_tensor_range("m") == Range(-math.inf, math.inf, nan=True)
    assert model.get_tensor_range("w") == Range(-2, 3)


def test_saved_dyadic_range_diagnostics_keep_their_text() -> None:
    model = single()
    offered = Range(Fraction(3, 8), Fraction(65, 8))
    with pytest.warns(RangeContradiction) as caught:
        model.set_tensor_datatype("x", DataType["FIXED<8,4>"], range=offered)
    assert str(caught[0].message) == (
        "qonnx.tensor/range of tensor 'x' not stated: its writer states "
        "[3*2^-3, 65*2^-3], which its datatype FIXED<8,4> does not hold"
    )
    model.set_tensor_datatype("x", DataType["FIXED<8,4>"], range=Range(0, 1))
    annotation = model.graph.quantization_annotation[0].quant_parameter_tensor_names
    next(p for p in annotation if p.key == RANGE.entry).value = '{"nan":false,"range":["3*2^-3","65*2^-3"]}'
    with pytest.raises(MetadataError) as raw:
        model.get_tensor_range("x")
    assert str(raw.value) == (
        "qonnx.tensor/range of tensor 'x': stored [3*2^-3, 65*2^-3] is not held by its datatype FIXED<8,4>"
    )
    with pytest.raises(MetadataError) as shown:
        model.get_tensor_annotation_range("x")
    assert str(shown.value) == "qonnx.tensor/range of tensor 'x': stored [3/8, 65/8] is not held by its datatype FIXED<8,4>"


# -- the range with its tensor


def test_the_range_is_renamed_and_removed_with_its_tensor() -> None:
    model = chain()
    model.set_tensor_datatype("m", DataType["INT7"], range=Range(-39, 36))
    model.rename_tensor("m", "renamed")
    assert model.get_stated_tensor_range("renamed") == Range(-39, 36)
    assert model.get_stated_tensor_range("m") is None
    model.graph.value_info.append(helper.make_tensor_value_info("unused", F, [1]))
    model.set_tensor_datatype("unused", DataType["INT4"], range=Range(0, 1))
    after = model.transform(RemoveUnusedTensors())
    assert get_by_name(after.graph.quantization_annotation, "unused", "tensor_name") is None
    assert after.get_stated_tensor_range("renamed") == Range(-39, 36)


def test_the_range_follows_its_tensor_into_a_cut() -> None:
    model = chain()
    model.set_tensor_datatype("m", DataType["INT7"], range=Range(-39, 36))
    annotation = get_by_name(model.graph.quantization_annotation, "m", "tensor_name")
    kept = list(annotation.quant_parameter_tensor_names)
    metadata.cut(kept)
    assert [prop.key for prop in kept] == ["finn_datatype", "qonnx.tensor/@version", "qonnx.tensor/@follow", RANGE.entry]
    assert metadata.read(kept, TENSOR, tensor=True) == {"range": Range(-39, 36)}
