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
# * Neither the name of AMD nor the names of its
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

"""A tensor's range (qonnx.core.ranges): the format, the datatype fact it is part of,
how InferDataTypes computes it for every pass, and MultiThreshold's."""

import pytest

import math
import numpy as np
import subprocess
import sys
import warnings
from fractions import Fraction
from onnx import StringStringEntryProto, TensorProto, helper

import qonnx.transformation.infer_datatypes as infer_datatypes
from qonnx.core import metadata, ranges, tensor_ranges
from qonnx.core.datatype import DataType
from qonnx.core.metadata import MetadataError
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.ranges import RangeContradiction, TensorRange
from qonnx.core.tensor_ranges import RANGE
from qonnx.transformation.general import RemoveUnusedTensors
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import get_by_name, qonnx_make_model

F = TensorProto.FLOAT
W = np.array([[1, -2], [3, 0], [-1, 1]], dtype=np.float32)


def chain(weights=W):
    """x (INT4, 1x3) -> MatMul by w (INT3) -> m -> Add b (INT4, [-5, 5]) -> y.

    m's exact range is [-39, 36] (INT7); y reads it as [-44, 41] (INT7), where m's
    datatype alone, [-64, 63], would give [-69, 68] (INT8)."""
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
    model.set_initializer("w", weights)
    model.set_initializer("b", np.array([[5, -5]], dtype=np.float32))
    for name, dtype in (("x", "INT4"), ("w", "INT3"), ("b", "INT4")):
        model.set_tensor_datatype(name, DataType[dtype])
    return model


def inferred(model):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RangeContradiction)
        return model.transform(InferDataTypes())


def entries(model, tensor):
    annotation = get_by_name(model.graph.quantization_annotation, tensor, "tensor_name")
    return {entry.key: entry.value for entry in annotation.quant_parameter_tensor_names}


def single(op, x_type, attributes=None, inputs=(), shape=(1, 4), **initializers):
    """x (x_type) -> op(x, *inputs) -> y, each named initializer set."""
    graph = helper.make_graph(
        [helper.make_node(op, ["x", *inputs], ["y"], name=f"{op}_0", **(attributes or {}))],
        "single",
        [helper.make_tensor_value_info("x", F, list(shape))],
        [helper.make_tensor_value_info("y", F, list(shape))],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    for name, values in initializers.items():
        model.set_initializer(name, np.asarray(values, dtype=np.float32))
    model.set_tensor_datatype("x", DataType[x_type])
    return model


# -- the format


@pytest.mark.parametrize(
    "value, text",
    [
        (TensorRange(-(2**70), 2**70 + 1), '{"nan":false,"range":[-1180591620717411303424,1180591620717411303425]}'),
        (TensorRange(Fraction(-3, 4), math.inf), '{"nan":false,"range":["-3*2^-2","inf"]}'),
        (TensorRange(-math.inf, 0.5, nan=True), '{"nan":true,"range":["-inf","1*2^-1"]}'),
        (TensorRange(-2.0, Fraction(6, 2)), '{"nan":false,"range":[-2,3]}'),
    ],
)
def test_a_range_is_stored_as_canonical_json_and_read_back_exactly(value, text):
    model = chain()
    datatype = DataType["INT80"] if value.is_integer() else DataType["FLOAT32"]
    model.set_tensor_datatype("m", datatype, value)
    assert entries(model, "m") == {
        "finn_datatype": datatype.name,
        "qonnx.tensor/@version": "1",
        "qonnx.tensor/@follow": "true",
        "qonnx.tensor/range": text,
    }
    loaded = ModelWrapper(model.model.SerializeToString())
    assert loaded.get_tensor_range("m") == value
    assert loaded.get(RANGE, tensor="m") == value


@pytest.mark.parametrize("ends", [(2, 1), (Fraction(1, 3), 1), (math.nan, 1), (True, 2), ("1", 2)])
def test_a_range_s_ends_are_ordered_dyadic_numbers_or_infinities(ends):
    with pytest.raises(ValueError):
        TensorRange(*ends)


@pytest.mark.parametrize(
    "text",
    [
        "[-1, 2]",
        '{"nan": false, "range": [2, 1]}',
        '{"nan": false, "range": [1.0, 2]}',
        '{"nan": false, "range": [true, 2]}',
        '{"nan": "no", "range": [1, 2]}',
        '{"nan": false, "range": ["6*2^-2", 2]}',
        '{"nan": false, "range": ["1*2^0", 2]}',
        '{"nan": false, "range": ["1/3", 2]}',
        '{"nan": false, "range": [1, "+inf"]}',
        '{"nan": false, "range": [1, 2, 3]}',
        '{"nan": false, "range": [1, 2], "extra": 1}',
        '{"nan": false, "range": [1, NaN]}',
    ],
)
def test_a_malformed_range_is_a_metadata_error_naming_the_entry(text):
    """Read directly, or by inference reading the input it is stored on."""
    model = chain()
    annotation = get_by_name(model.graph.quantization_annotation, "x", "tensor_name")
    for key, value in (("qonnx.tensor/@version", "1"), ("qonnx.tensor/@follow", "true"), ("qonnx.tensor/range", text)):
        annotation.quant_parameter_tensor_names.append(StringStringEntryProto(key=key, value=value))
    with pytest.raises(MetadataError, match="qonnx.tensor/range: stored"):
        model.get_tensor_range("x")
    with pytest.raises(MetadataError, match="qonnx.tensor/range: stored"):
        model.transform(InferDataTypes())


def test_a_stored_range_its_datatype_does_not_hold_is_a_metadata_error():
    model = chain()
    model.set(RANGE, TensorRange(-100, 100), tensor="x")  # x is INT4; written past the datatype
    with pytest.raises(MetadataError, match=r"tensor 'x': stored \[-100, 100\] is not held by its datatype INT4"):
        model.get_tensor_range("x")


def test_a_range_that_is_not_a_tensor_range_is_refused_on_writing():
    with pytest.raises(MetadataError, match="qonnx.tensor/range: cannot store"):
        chain().set_tensor_datatype("m", DataType["INT8"], (1, 2))


# -- one fact with the datatype


def test_a_datatype_set_without_its_range_clears_the_range():
    model = chain()
    model.set_tensor_datatype("m", DataType["INT7"], TensorRange(-39, 36))
    assert model.get_tensor_range("m") == TensorRange(-39, 36)
    model.set_tensor_datatype("m", DataType["INT7"])
    assert tensor_ranges.stated(model, "m") is None
    assert model.get_tensor_range("m") == TensorRange(-64, 63)
    model.set_tensor_datatype("m", DataType["INT7"], TensorRange(-39, 36))
    model.set_tensor_datatype("m", DataType["INT8"])
    assert tensor_ranges.stated(model, "m") is None
    model.set_tensor_datatype("m", DataType["INT7"], TensorRange(-39, 36))
    model.set_tensor_datatype("m", None)
    assert get_by_name(model.graph.quantization_annotation, "m", "tensor_name").quant_parameter_tensor_names == []


@pytest.mark.parametrize(
    "datatype, full",
    [
        ("INT7", TensorRange(-64, 63)),
        ("BIPOLAR", TensorRange(-1, 1)),
        ("FIXED<8,4>", TensorRange(-8, Fraction(127, 16))),
        ("SCALEDINT<8>", TensorRange(-math.inf, math.inf)),
        ("FLOAT32", TensorRange(-math.inf, math.inf, nan=True)),
        ("FLOAT16", TensorRange(-math.inf, math.inf, nan=True)),
    ],
)
def test_a_datatype_s_full_range_is_its_range_when_none_is_stated_and_is_not_stored(datatype, full):
    model = chain()
    model.set_tensor_datatype("m", DataType[datatype], full)
    assert tensor_ranges.stated(model, "m") is None
    assert model.get_tensor_range("m") == full


def test_a_writer_stating_a_range_its_datatype_does_not_hold_is_named_and_nothing_is_stored():
    model = chain()
    with pytest.warns(
        RangeContradiction,
        match=r"tensor 'm' not stated: MatMul_0 \(MatMul\) states \[0, 2\], which its datatype INT2 does not hold",
    ):
        model.set_tensor_datatype("m", DataType["INT2"], TensorRange(0, 2))
    assert tensor_ranges.stated(model, "m") is None
    assert model.get_tensor_datatype("m") == DataType["INT2"]


@pytest.mark.parametrize(
    "datatype, value, held",
    [
        ("INT4", TensorRange(-8, 7), True),
        ("INT4", TensorRange(-8, 8), False),
        ("INT4", TensorRange(0, Fraction(1, 2)), False),
        ("INT4", TensorRange(0, 1, nan=True), False),
        ("FIXED<8,4>", TensorRange(Fraction(-1, 2), 7), True),
        ("FIXED<8,4>", TensorRange(0, math.inf), False),
        ("SCALEDINT<8>", TensorRange(-math.inf, 3), True),
        ("SCALEDINT<8>", TensorRange(0, 3, nan=True), False),
        ("FLOAT32", TensorRange(-math.inf, math.inf, nan=True), True),
        ("FLOAT<4,3,7>", TensorRange(0, 1), True),
    ],
)
def test_which_ranges_a_datatype_holds(datatype, value, held):
    assert ranges.holds(DataType[datatype], value) is held


# -- inference: integer rules


def test_inference_states_each_rule_s_exact_range_with_its_datatype():
    model = inferred(chain())
    assert (model.get_tensor_datatype("m"), model.get_tensor_range("m")) == (DataType["INT7"], TensorRange(-39, 36))
    assert (model.get_tensor_datatype("y"), model.get_tensor_range("y")) == (DataType["INT7"], TensorRange(-44, 41))
    assert tensor_ranges.stated(model, "x") is None


def test_a_narrowed_range_flows_forward_and_narrows_the_next_datatype():
    """y reads m's range [-39, 36], not INT7's [-64, 63]: INT7, not INT8."""
    model = inferred(chain())
    assert model.get_tensor_datatype("y") == DataType["INT7"]
    unstated = chain()
    unstated.set_tensor_datatype("m", DataType["INT7"])
    assert infer_datatypes._range_rules["Add"](unstated, unstated.graph.node[1]) == (-69, 68)


def test_a_pass_over_an_unchanged_graph_changes_nothing_and_its_bounds_come_from_the_cache():
    model = inferred(chain())
    infer_datatypes.bounds_counts.clear()
    unchanged, modified = InferDataTypes().apply(model)
    assert not modified
    assert infer_datatypes.bounds_counts["dot computed"] == 0 and infer_datatypes.bounds_counts["dot cached"] == 1


def test_an_edit_upstream_is_recomputed_on_the_next_pass_without_a_warning():
    model = inferred(chain())
    model.set_initializer("w", np.array([[1, 0], [1, 0], [1, 0]], dtype=np.float32))
    model.set_tensor_datatype("w", DataType["UINT1"])
    model = inferred(model)
    assert (model.get_tensor_datatype("m"), model.get_tensor_range("m")) == (DataType["INT6"], TensorRange(-24, 21))
    assert (model.get_tensor_datatype("y"), model.get_tensor_range("y")) == (DataType["INT6"], TensorRange(-29, 26))


def test_the_cache_is_keyed_by_the_input_s_range():
    model = chain()
    model.set_tensor_datatype("x", DataType["INT4"], TensorRange(0, 3))
    model = inferred(model)
    assert model.get_tensor_range("m") == TensorRange(-6, 12)
    model.set_tensor_datatype("x", DataType["INT4"])
    model = inferred(model)
    assert model.get_tensor_range("m") == TensorRange(-39, 36)


def test_a_graph_input_s_stated_range_is_read_as_stated():
    model = chain()
    model.set_tensor_datatype("x", DataType["INT4"], TensorRange(0, 3))
    model = inferred(model)
    assert model.get_tensor_range("x") == TensorRange(0, 3)
    assert (model.get_tensor_datatype("m"), model.get_tensor_range("m")) == (DataType["INT5"], TensorRange(-6, 12))


# -- inference: what passes through, and float arithmetic


def test_shape_ops_pass_the_range_through():
    model = single("Reshape", "INT4", inputs=["s"], s=np.array([4, 1], dtype=np.int64))
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 4
    model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 1
    model.set_tensor_datatype("x", DataType["INT4"], TensorRange(-1, 1))
    model = inferred(model)
    assert (model.get_tensor_datatype("y"), model.get_tensor_range("y")) == (DataType["INT4"], TensorRange(-1, 1))


@pytest.mark.parametrize(
    "op, attributes, inputs, initializers, x_range, expected",
    [
        ("Relu", {}, [], {}, None, TensorRange(0, math.inf, nan=True)),
        ("Neg", {}, [], {}, TensorRange(Fraction(-1, 2), 3), TensorRange(-3, Fraction(1, 2))),
        ("Clip", {}, ["lo", "hi"], {"lo": -1.5, "hi": 6.0}, None, TensorRange(Fraction(-3, 2), 6, nan=True)),
        ("Max", {}, ["c"], {"c": [0.25]}, TensorRange(-4, 2), TensorRange(Fraction(1, 4), 2)),
        ("Min", {}, ["c"], {"c": [0.25]}, TensorRange(-4, 2), TensorRange(-4, Fraction(1, 4))),
        ("Concat", {"axis": 0}, ["c"], {"c": [[9.0, 9.0, 9.0, 9.0]]}, TensorRange(-4, 2), TensorRange(-4, 9)),
        ("Mul", {}, ["c"], {"c": [0.5]}, TensorRange(-4, 3), TensorRange(-2, Fraction(3, 2))),
        ("Mul", {}, ["c"], {"c": [-4.0]}, TensorRange(-4, 3), TensorRange(-12, 16)),
    ],
)
def test_ops_that_do_not_round_pass_a_float_range_through(op, attributes, inputs, initializers, x_range, expected):
    model = single(op, "FLOAT32", attributes, inputs, **initializers)
    if x_range is not None:
        model.set_tensor_datatype("x", DataType["FLOAT32"], x_range)
    model = inferred(model)
    assert model.get_tensor_datatype("y") == DataType["FLOAT32"]
    assert model.get_tensor_range("y") == expected


def test_a_mul_by_a_power_of_two_scales_an_integer_range_into_its_float_result():
    model = inferred(single("Mul", "INT8", inputs=["c"], c=[0.125]))
    assert model.get_tensor_datatype("y") == DataType["FLOAT32"]
    assert model.get_tensor_range("y") == TensorRange(-16, Fraction(127, 8))


@pytest.mark.parametrize(
    "op, inputs, initializers, expected",
    [
        ("Mul", ["c"], {"c": [3.0]}, TensorRange(-math.inf, math.inf)),
        ("Add", ["c"], {"c": [0.1]}, TensorRange(-math.inf, math.inf)),
        ("MatMul", ["w"], {"w": np.full((4, 4), 0.1)}, TensorRange(-math.inf, math.inf, nan=True)),
    ],
)
def test_mixed_and_float_arithmetic_states_an_unbounded_range(op, inputs, initializers, expected):
    model = inferred(single(op, "INT8", inputs=inputs, **initializers))
    assert model.get_tensor_datatype("y") == DataType["FLOAT32"]
    assert model.get_tensor_range("y") == expected


def test_float_arithmetic_on_a_float_that_may_be_infinite_may_make_nan():
    model = inferred(single("Add", "FLOAT32", inputs=["c"], c=[1.0]))
    assert model.get_tensor_range("y") == TensorRange(-math.inf, math.inf, nan=True)
    assert tensor_ranges.stated(model, "y") is None  # FLOAT32's full range


def test_a_float_datatype_is_never_narrowed():
    model = single("Relu", "FLOAT32")
    model.set_tensor_datatype("x", DataType["FLOAT32"], TensorRange(0, 3))
    model = inferred(model)
    assert model.get_tensor_datatype("y") == DataType["FLOAT32"]
    assert model.get_tensor_range("y") == TensorRange(0, 3)


# -- MultiThreshold


def thresholded(out_dtype, steps=3, **attributes):
    """x (INT8, 1x2) -> MultiThreshold, 2 channels of ``steps`` thresholds -> y."""
    node = helper.make_node(
        "MultiThreshold", ["x", "thr"], ["y"], domain="qonnx.custom_op.general", out_dtype=out_dtype, **attributes
    )
    graph = helper.make_graph(
        [node],
        "thresholds",
        [helper.make_tensor_value_info("x", F, [1, 2])],
        [helper.make_tensor_value_info("y", F, [1, 2])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_initializer("thr", np.tile(np.arange(steps, dtype=np.float32), (2, 1)))
    model.set_tensor_datatype("x", DataType["INT8"])
    return model


@pytest.mark.parametrize(
    "out_dtype, attributes, expected",
    [
        ("UINT3", {}, (0, 3)),
        ("INT4", {"out_scale": 2.0, "out_bias": -3.0}, (-3, 3)),
        ("INT3", {"out_scale": -1.0}, (-3, 0)),
    ],
)
def test_multithreshold_states_out_bias_plus_out_scale_times_its_count(out_dtype, attributes, expected):
    model = inferred(thresholded(out_dtype, **attributes))
    assert model.get_tensor_datatype("y") == DataType[out_dtype]
    assert tensor_ranges.stated(model, "y") == TensorRange(*expected)


def test_multithreshold_whose_range_is_its_datatype_s_states_none():
    model = inferred(thresholded("UINT2"))
    assert tensor_ranges.stated(model, "y") is None and model.get_tensor_range("y") == TensorRange(0, 3)


def test_multithreshold_with_a_fractional_scale_states_no_range():
    model = inferred(thresholded("INT4", out_scale=0.5))
    assert model.get_tensor_datatype("y") == DataType["FLOAT32"]
    assert tensor_ranges.stated(model, "y") is None


def test_multithreshold_whose_out_dtype_cannot_hold_its_count_is_named_once_per_pass():
    model = thresholded("INT2", steps=2)  # computes [0, 2]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        InferDataTypes().apply(model)
        InferDataTypes().apply(model)
    assert [str(w.message) for w in caught if w.category is RangeContradiction] == [
        "qonnx.tensor/range of tensor 'y' not stated: MultiThreshold states [0, 2], which its datatype INT2 does not hold"
    ] * 2
    assert model.get_tensor_datatype("y") == DataType["INT2"] and tensor_ranges.stated(model, "y") is None


def test_a_narrow_range_out_of_multithreshold_narrows_the_matmul_after_it_through_a_reshape():
    """INT2 thresholds computing [-1, 1] (two thresholds, out_bias -1), reshaped: the
    MatMul after them reads [-1, 1], not INT2's [-2, 1]."""
    model = thresholded("INT2", steps=2, out_bias=-1.0)
    model.graph.output[0].name = "z"
    model.graph.node.append(helper.make_node("Reshape", ["y", "s"], ["r"], name="Reshape_0"))
    model.graph.node.append(helper.make_node("MatMul", ["r", "w"], ["z"], name="MatMul_0"))
    model.set_initializer("s", np.array([1, 2], dtype=np.int64))
    model.set_initializer("w", np.array([[3], [3]], dtype=np.float32))
    model.set_tensor_datatype("w", DataType["UINT2"])
    model = inferred(model)
    assert tensor_ranges.stated(model, "y") == TensorRange(-1, 1)
    assert tensor_ranges.stated(model, "r") == TensorRange(-1, 1)
    assert (model.get_tensor_datatype("z"), model.get_tensor_range("z")) == (DataType["INT4"], TensorRange(-6, 6))


# -- the range with its tensor


def test_the_range_is_renamed_and_removed_with_its_tensor():
    model = inferred(chain())
    m = tensor_ranges.stated(model, "m")
    model.rename_tensor("m", "renamed")
    assert tensor_ranges.stated(model, "renamed") == m and tensor_ranges.stated(model, "m") is None
    model.graph.value_info.append(helper.make_tensor_value_info("unused", F, [1]))
    model.set_tensor_datatype("unused", DataType["INT4"], TensorRange(0, 1))
    after = model.transform(RemoveUnusedTensors())
    assert get_by_name(after.graph.quantization_annotation, "unused", "tensor_name") is None
    assert tensor_ranges.stated(after, "renamed") == m


def test_the_range_follows_its_tensor_into_a_cut():
    model = inferred(chain())
    annotation = get_by_name(model.graph.quantization_annotation, "m", "tensor_name")
    props = list(annotation.quant_parameter_tensor_names)
    metadata.cut(props)
    assert {prop.key for prop in props} >= {"qonnx.tensor/range", "qonnx.tensor/@version", "qonnx.tensor/@follow"}


def test_the_range_module_loads_no_onnx():
    """qonnx.core.ranges is the value alone: a value layer holds ranges as it holds
    datatypes, without loading ONNX (the model half is qonnx.core.tensor_ranges)."""
    script = "\n".join(
        (
            "import sys",
            "import qonnx.core.ranges",
            "loaded = sorted(name for name in sys.modules if name == 'onnx' or name.startswith('onnx.'))",
            "raise SystemExit('loaded: ' + ', '.join(loaded) if loaded else 0)",
        )
    )
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
