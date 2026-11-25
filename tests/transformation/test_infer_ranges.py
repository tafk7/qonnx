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

"""Ranges inferred with datatypes (InferDataTypes(exact_ranges=True), and any call
inside an inference pass): what passes through, float arithmetic, Split, the bounds
cache, MultiThreshold, a custom op's contract and an absent optional output; the
default rules writing no range beside them."""

from __future__ import annotations

import pytest

import math
import numpy as np
import warnings
from fractions import Fraction
from onnx import TensorProto, helper
from typing import Any, Optional

import qonnx.transformation.infer_datatypes as infer_datatypes
from qonnx.core import ranges
from qonnx.core.datatype import ArbPrecFloatType, BaseDataType, DataType, Float16Type, FloatType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.core.ranges import Range
from qonnx.core.tensor_ranges import RangeContradiction, inferring, one_pass
from qonnx.custom_op.base import CustomOp
from qonnx.custom_op.registry import add_op_to_domain, getCustomOp
from qonnx.transformation.infer_datatypes import InferDataTypes, infer_node_datatype
from qonnx.util.basic import get_by_name, qonnx_make_model

F = TensorProto.FLOAT
INF = math.inf
W = np.array([[1, -2], [3, 0], [-1, 1]], dtype=np.float32)


def chain(weights: np.ndarray = W) -> ModelWrapper:
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


def inferred(model: ModelWrapper) -> ModelWrapper:
    """``model`` after InferDataTypes(exact_ranges=True); a contradiction fails."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RangeContradiction)
        return model.transform(InferDataTypes(exact_ranges=True))


def single(
    op: str,
    x_type: str,
    attributes: Optional[dict[str, Any]] = None,
    inputs: tuple[str, ...] = (),
    shape: tuple[int, ...] = (1, 4),
    **initializers: Any,
) -> ModelWrapper:
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


def is_float(datatype: BaseDataType) -> bool:
    return isinstance(datatype, (FloatType, Float16Type, ArbPrecFloatType))


# integer results: the range flows forward, and the bounds cache


def test_a_narrowed_range_flows_forward_and_narrows_the_next_datatype() -> None:
    """y reads m's range [-39, 36], not INT7's [-64, 63]: INT7, not INT8."""
    model = inferred(chain())
    assert (model.get_tensor_datatype("m"), model.get_tensor_range("m")) == (DataType["INT7"], Range(-39, 36))
    assert (model.get_tensor_datatype("y"), model.get_tensor_range("y")) == (DataType["INT7"], Range(-44, 41))
    assert model.get_stated_tensor_range("x") is None
    unstated = chain()
    unstated.set_tensor_datatype("m", DataType["INT7"])
    assert infer_datatypes._RANGE_RULES["Add"](unstated, unstated.graph.node[1]) == (-69, 68)


def test_a_pass_over_an_unchanged_graph_changes_nothing_and_its_bounds_come_from_the_cache() -> None:
    model = inferred(chain())
    infer_datatypes.bounds_counts.clear()
    _, modified = InferDataTypes(exact_ranges=True).apply(model)
    assert not modified
    assert infer_datatypes.bounds_counts["dot computed"] == 0 and infer_datatypes.bounds_counts["dot cached"] == 1


def test_an_edit_upstream_is_recomputed_on_the_next_pass_without_a_warning() -> None:
    model = inferred(chain())
    model.set_initializer("w", np.array([[1, 0], [1, 0], [1, 0]], dtype=np.float32))
    model.set_tensor_datatype("w", DataType["UINT1"])
    model = inferred(model)
    assert (model.get_tensor_datatype("m"), model.get_tensor_range("m")) == (DataType["INT6"], Range(-24, 21))
    assert (model.get_tensor_datatype("y"), model.get_tensor_range("y")) == (DataType["INT6"], Range(-29, 26))


def test_the_cache_is_keyed_by_the_input_s_range() -> None:
    model = chain()
    model.set_tensor_datatype("x", DataType["INT4"], range=Range(0, 3))
    model = inferred(model)
    assert model.get_tensor_range("x") == Range(0, 3)
    assert (model.get_tensor_datatype("m"), model.get_tensor_range("m")) == (DataType["INT5"], Range(-6, 12))
    model.set_tensor_datatype("x", DataType["INT4"])
    model = inferred(model)
    assert model.get_tensor_range("m") == Range(-39, 36)


def test_the_cache_is_keyed_by_the_weights_values() -> None:
    model = inferred(chain())
    infer_datatypes.bounds_counts.clear()
    model.set_initializer("w", -W)
    model = inferred(model)
    assert infer_datatypes.bounds_counts["dot computed"] == 1
    assert model.get_tensor_range("m") == Range(-36, 39)


# what passes through, and float arithmetic


def test_shape_ops_pass_the_range_through() -> None:
    model = single("Reshape", "INT4", inputs=("s",), s=np.array([4, 1], dtype=np.int64))
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 4
    model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 1
    model.set_tensor_datatype("x", DataType["INT4"], range=Range(-1, 1))
    model = inferred(model)
    assert (model.get_tensor_datatype("y"), model.get_tensor_range("y")) == (DataType["INT4"], Range(-1, 1))


@pytest.mark.parametrize(
    "op, attributes, inputs, initializers, x_range, expected",
    [
        ("Relu", {}, (), {}, None, Range(0, INF, nan=True)),
        ("Neg", {}, (), {}, Range(Fraction(-1, 2), 3), Range(-3, Fraction(1, 2))),
        ("Clip", {}, ("lo", "hi"), {"lo": -1.5, "hi": 6.0}, None, Range(Fraction(-3, 2), 6, nan=True)),
        ("Max", {}, ("c",), {"c": [0.25]}, Range(-4, 2), Range(Fraction(1, 4), 2)),
        ("Min", {}, ("c",), {"c": [0.25]}, Range(-4, 2), Range(-4, Fraction(1, 4))),
        ("Concat", {"axis": 0}, ("c",), {"c": [[9.0, 9.0, 9.0, 9.0]]}, Range(-4, 2), Range(-4, 9)),
        ("Mul", {}, ("c",), {"c": [0.5]}, Range(-4, 3), Range(-2, Fraction(3, 2))),
        ("Mul", {}, ("c",), {"c": [-4.0]}, Range(-4, 3), Range(-12, 16)),
    ],
)
def test_ops_that_do_not_round_pass_a_float_range_through(
    op: str,
    attributes: dict[str, Any],
    inputs: tuple[str, ...],
    initializers: dict[str, Any],
    x_range: Optional[Range],
    expected: Range,
) -> None:
    model = single(op, "FLOAT32", attributes, inputs, **initializers)
    if x_range is not None:
        model.set_tensor_datatype("x", DataType["FLOAT32"], range=x_range)
    model = inferred(model)
    assert model.get_tensor_datatype("y") == DataType["FLOAT32"]
    assert model.get_tensor_range("y") == expected


def test_a_mul_by_a_power_of_two_scales_an_integer_range_into_its_scaled_result() -> None:
    model = inferred(single("Mul", "INT8", inputs=("c",), c=[0.125]))
    assert model.get_tensor_datatype("y") == DataType["FLOAT32"]
    assert model.get_tensor_range("y") == Range(-16, Fraction(127, 8))


def f32(value: Any) -> ranges.Endpoint:
    """``value`` rounded into float32, as an endpoint."""
    return ranges.endpoint(float(np.float32(value)))


@pytest.mark.parametrize(
    "op, inputs, initializers, expected",
    [
        ("Mul", ("c",), {"c": [3.0]}, Range(-384, 381)),
        ("Div", ("c",), {"c": [4.0]}, Range(-32, Fraction(127, 4))),
        # each end rounded once into float32, as float32 arithmetic rounds it
        ("Add", ("c",), {"c": [0.1]}, Range(f32(np.float32(-128) + np.float32(0.1)), f32(127 + np.float32(0.1)))),
        ("Div", ("c",), {"c": [3.0]}, Range(f32(np.float32(-128) / 3), f32(np.float32(127) / 3))),
        # a sum rounds in an order ONNX leaves open: no bound
        ("MatMul", ("w",), {"w": np.full((4, 4), 0.1)}, Range(-INF, INF, nan=True)),
    ],
)
def test_mixed_arithmetic_states_its_exact_range_rounded_into_float32(
    op: str, inputs: tuple[str, ...], initializers: dict[str, Any], expected: Range
) -> None:
    model = inferred(single(op, "INT8", inputs=inputs, **initializers))
    assert model.get_tensor_datatype("y") == DataType["FLOAT32"]
    assert model.get_tensor_range("y") == expected


def test_float_arithmetic_on_a_float_that_may_be_infinite_may_make_nan() -> None:
    model = inferred(single("Add", "FLOAT32", inputs=("c",), c=[1.0]))
    assert model.get_tensor_range("y") == Range(-INF, INF, nan=True)
    assert model.get_stated_tensor_range("y") is None  # FLOAT32's full range


def binary(op: str, x_range: Range, z_range: Range, elem: int = F, datatype: str = "FLOAT32") -> Range:
    """x and z (``datatype``, their ranges stated) -> op -> y, all held as ``elem``;
    y's datatype and range inferred."""
    graph = helper.make_graph(
        [helper.make_node(op, ["x", "z"], ["y"], name=f"{op}_0")],
        "binary",
        [helper.make_tensor_value_info(name, elem, [4]) for name in ("x", "z")],
        [helper.make_tensor_value_info("y", elem, [4])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("x", DataType[datatype], range=x_range)
    model.set_tensor_datatype("z", DataType[datatype], range=z_range)
    model = inferred(model)
    assert is_float(model.get_tensor_datatype("y"))
    return model.get_tensor_range("y")


LARGEST32 = ranges.endpoint(float(np.finfo(np.float32).max))


@pytest.mark.parametrize(
    "op, x, z, expected",
    [
        # exact: no rounding
        ("Add", Range(1, 2), Range(Fraction(1, 4), Fraction(1, 2)), Range(Fraction(5, 4), Fraction(5, 2))),
        ("Sub", Range(1, 2), Range(-3, 5), Range(-4, 5)),
        ("Mul", Range(-2, 3), Range(-5, 4), Range(-15, 12)),
        ("Div", Range(-3, 6), Range(Fraction(1, 2), 4), Range(-6, 12)),
        # overflow: an end past float32's largest value rounds to infinity, as IEEE 754 does
        ("Mul", Range(0, 2**127), Range(0, 4), Range(0, INF)),
        # half float32's spacing past its largest value: to infinity (ties to even); less: to the largest
        ("Add", Range(LARGEST32, LARGEST32), Range(2**103, 2**103), Range(INF, INF)),
        ("Add", Range(LARGEST32, LARGEST32), Range(2**102, 2**102), Range(LARGEST32, LARGEST32)),
        ("Add", Range(1, INF), Range(1, INF), Range(2, INF)),
        ("Mul", Range(1, 2), Range(1, INF), Range(1, INF)),
        # NaN only where the intervals admit inf - inf, 0 * inf, 0 / 0 or inf / inf, or an input may be NaN
        ("Add", Range(0, INF), Range(-INF, 0), Range(-INF, INF, nan=True)),
        ("Sub", Range(0, INF), Range(0, INF), Range(-INF, INF, nan=True)),
        ("Mul", Range(0, 1), Range(1, INF), Range(0, INF, nan=True)),
        ("Div", Range(0, 1), Range(0, 1), Range(-INF, INF, nan=True)),
        ("Div", Range(1, INF), Range(1, INF), Range(0, INF, nan=True)),
        ("Add", Range(0, 1, nan=True), Range(0, 1), Range(0, 2, nan=True)),
        # a divisor interval holding 0: unbounded both ways (a range does not tell -0 from +0),
        # NaN only where 0 / 0 or inf / inf is admitted
        ("Div", Range(1, 2), Range(-1, 1), Range(-INF, INF)),
        ("Div", Range(1, INF), Range(0, 1), Range(-INF, INF)),
        ("Div", Range(1, INF), Range(0, INF), Range(-INF, INF, nan=True)),
    ],
)
def test_float_elementwise_arithmetic_rounds_its_exact_interval_once(op: str, x: Range, z: Range, expected: Range) -> None:
    assert binary(op, x, z) == expected


def test_float_elementwise_arithmetic_rounds_into_the_format_it_computes_in() -> None:
    x, z = Range(1, 1), Range(Fraction(1, 2**11), Fraction(1, 2**11))
    # 1 + 2^-11 is exact in float32, and a tie in float16, which rounds to even: the
    # format is the one the tensors are held and computed in
    assert binary("Add", x, z) == Range(Fraction(2**11 + 1, 2**11), Fraction(2**11 + 1, 2**11))
    assert binary("Add", x, z, TensorProto.FLOAT16, "FLOAT16") == Range(1, 1)
    # 300 * 300 overflows float16
    assert binary("Mul", Range(0, 300), Range(0, 300), TensorProto.FLOAT16, "FLOAT16") == Range(0, INF)


def test_the_float_rule_bounds_what_execution_computes() -> None:
    xs = np.array([-3.0, 0.1, 7.5, 1e-3], dtype=np.float32)
    zs = np.array([0.3, -7.0, 0.7, 3.0], dtype=np.float32)
    for op in ("Add", "Sub", "Mul", "Div"):
        graph = helper.make_graph(
            [helper.make_node(op, ["x", "z"], ["y"])],
            "binary",
            [helper.make_tensor_value_info(name, F, [4]) for name in ("x", "z")],
            [helper.make_tensor_value_info("y", F, [4])],
        )
        model = ModelWrapper(qonnx_make_model(graph))
        model.set_tensor_datatype("x", DataType["FLOAT32"], range=ranges.values_range(xs))
        model.set_tensor_datatype("z", DataType["FLOAT32"], range=ranges.values_range(zs))
        model = inferred(model)
        stated = model.get_tensor_range("y")
        computed = ranges.values_range(execute_onnx(model, {"x": xs, "z": zs})["y"])
        assert stated.least <= computed.least and computed.greatest <= stated.greatest and not computed.nan


def test_a_float_rule_is_not_applied_to_integers_held_in_integer_types() -> None:
    """An integer Div truncates: 7 / 2 is 3 on INT64 tensors, not 3.5."""
    graph = helper.make_graph(
        [helper.make_node("Div", ["a", "b"], ["y"])],
        "ints",
        [],
        [helper.make_tensor_value_info("y", TensorProto.INT64, [1])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_initializer("a", np.array([7], dtype=np.int64))
    model.set_initializer("b", np.array([2], dtype=np.int64))
    model = inferred(model)
    assert model.get_tensor_range("y") == Range(-INF, INF)


def test_a_float_datatype_is_never_narrowed() -> None:
    model = single("Relu", "FLOAT32")
    model.set_tensor_datatype("x", DataType["FLOAT32"], range=Range(0, 3))
    model = inferred(model)
    assert model.get_tensor_datatype("y") == DataType["FLOAT32"]
    assert model.get_tensor_range("y") == Range(0, 3)


def test_a_fixed_point_input_passes_its_exact_end_through() -> None:
    model = single("Relu", "FIXED<64,32>", shape=(2,))
    model = inferred(model)
    # (2**63 - 1) / 2**32, which float64 rounds up to 2**31
    assert model.get_tensor_datatype("y") == DataType["FLOAT32"]
    assert model.get_stated_tensor_range("y") == Range(0, Fraction(2**63 - 1, 2**32))
    annotation = get_by_name(model.graph.quantization_annotation, "y", "tensor_name")
    stored = {entry.key: entry.value for entry in annotation.quant_parameter_tensor_names}
    assert stored["qonnx.tensor/range"] == '{"nan":false,"range":[0,"9223372036854775807*2^-32"]}'


# Split


@pytest.mark.parametrize(
    "datatype, value",
    [("INT4", Range(0, 3)), ("FLOAT32", Range(Fraction(-1, 2), INF, nan=True))],
)
def test_each_output_of_a_split_keeps_its_input_s_datatype_and_range(datatype: str, value: Range) -> None:
    graph = helper.make_graph(
        [helper.make_node("Split", ["x", "split"], ["a", "b", "c"], axis=1)],
        "split",
        [helper.make_tensor_value_info("x", F, [1, 6])],
        [helper.make_tensor_value_info(name, F, [1, 2]) for name in ("a", "b", "c")],
    )
    model = ModelWrapper(qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 13)]))
    model.set_initializer("split", np.array([2, 2, 2], dtype=np.int64))
    model.set_tensor_datatype("x", DataType[datatype], range=value)
    for name in ("a", "b", "c"):
        # a stale fact on each output is replaced
        model.set_tensor_datatype(name, DataType["INT8"], range=Range(-100, 100))
    model = inferred(model)
    for name in ("a", "b", "c"):
        assert (model.get_tensor_datatype(name), model.get_tensor_range(name)) == (DataType[datatype], value)


# MultiThreshold


def thresholded(out_dtype: str, steps: int = 3, **attributes: Any) -> ModelWrapper:
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
def test_multithreshold_states_out_bias_plus_out_scale_times_its_count(
    out_dtype: str, attributes: dict[str, Any], expected: tuple[int, int]
) -> None:
    model = inferred(thresholded(out_dtype, **attributes))
    assert model.get_tensor_datatype("y") == DataType[out_dtype]
    assert model.get_stated_tensor_range("y") == Range(*expected)


def test_multithreshold_states_its_range_inside_a_pass_and_none_outside() -> None:
    """The signal is the pass, which a caller may enter around the op's own step."""
    model = thresholded("UINT3")
    op = getCustomOp(model.graph.node[0])
    op.infer_node_datatype(model)
    assert model.get_tensor_datatype("y") == DataType["UINT3"]
    assert model.get_stated_tensor_range("y") is None
    assert not inferring(model)
    with one_pass(model):
        assert inferring(model)
        op.infer_node_datatype(model)
    assert model.get_stated_tensor_range("y") == Range(0, 3)
    # outside again, a datatype stated alone clears it
    op.infer_node_datatype(model)
    assert model.get_stated_tensor_range("y") is None


def test_infer_node_datatype_inside_a_pass_is_exact_and_outside_is_the_default() -> None:
    model = thresholded("UINT3")
    node = model.graph.node[0]
    assert infer_node_datatype(model, node, False)
    assert model.get_stated_tensor_range("y") is None
    with one_pass(model):
        assert infer_node_datatype(model, node, False)
    assert model.get_stated_tensor_range("y") == Range(0, 3)
    # the default rules, as before exact ranges
    assert model.transform(InferDataTypes()).get_stated_tensor_range("y") is None
    model = chain()
    with one_pass(model):
        infer_node_datatype(model, model.graph.node[0], False)
    assert model.get_tensor_datatype("m") == DataType["INT7"]


def test_multithreshold_whose_range_is_its_datatype_s_states_none() -> None:
    model = inferred(thresholded("UINT2"))
    assert model.get_stated_tensor_range("y") is None and model.get_tensor_range("y") == Range(0, 3)


def test_multithreshold_with_a_fractional_scale_states_no_range() -> None:
    model = inferred(thresholded("INT4", out_scale=0.5))
    assert model.get_tensor_datatype("y") == DataType["FLOAT32"]
    assert model.get_stated_tensor_range("y") is None
    assert getCustomOp(model.graph.node[0]).output_range(model) is None


def test_multithreshold_whose_out_dtype_cannot_hold_its_count_is_named_once_per_pass() -> None:
    model = thresholded("INT2", steps=2)  # computes [0, 2]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        InferDataTypes(exact_ranges=True).apply(model)
        InferDataTypes(exact_ranges=True).apply(model)
    assert [str(w.message) for w in caught if w.category is RangeContradiction] == [
        "qonnx.tensor/range of tensor 'y' not stated: MultiThreshold states [0, 2], which its datatype INT2 does not hold"
    ] * 2
    assert model.get_tensor_datatype("y") == DataType["INT2"] and model.get_stated_tensor_range("y") is None


def test_a_narrow_range_out_of_multithreshold_narrows_the_matmul_after_it_through_a_reshape() -> None:
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
    assert model.get_stated_tensor_range("y") == Range(-1, 1)
    assert model.get_stated_tensor_range("r") == Range(-1, 1)
    assert (model.get_tensor_datatype("z"), model.get_tensor_range("z")) == (DataType["INT4"], Range(-6, 6))


# a custom op's contract


class RangedTestOp(CustomOp):
    """States UINT4 with [1, 5] inside a pass, as the contract asks; SCALEDINT<8>
    with [-2, 2] where its ``scaled`` attribute is set."""

    def get_nodeattr_types(self) -> dict[str, Any]:
        return {"scaled": ("i", False, 0)}

    def make_shape_compatible_op(self, model: ModelWrapper) -> Any:
        node = self.onnx_node
        return helper.make_node("Relu", [node.input[0]], [node.output[0]])

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        out = self.onnx_node.output[0]
        if self.get_nodeattr("scaled"):
            dtype, value = DataType["SCALEDINT<8>"], Range(-2, 2)
        else:
            dtype, value = DataType["UINT4"], Range(1, 5)
        model.set_tensor_datatype(out, dtype, range=value if inferring(model) else None)

    def execute_node(self, context: dict[str, Any], graph: Any) -> None:
        context[self.onnx_node.output[0]] = context[self.onnx_node.input[0]]

    def verify_node(self) -> None:
        pass


TEST_DOMAIN = "qonnx.custom_op.test_infer_ranges"
add_op_to_domain(TEST_DOMAIN, RangedTestOp)


def ranged(scaled: int = 0) -> ModelWrapper:
    """x (INT8) -> RangedTestOp -> m -> Add 1 -> y."""
    graph = helper.make_graph(
        [
            helper.make_node("RangedTestOp", ["x"], ["m"], domain=TEST_DOMAIN, scaled=scaled),
            helper.make_node("Add", ["m", "one"], ["y"]),
        ],
        "ranged",
        [helper.make_tensor_value_info("x", F, [4])],
        [helper.make_tensor_value_info("y", F, [4])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_initializer("one", np.ones(4, dtype=np.float32))
    model.set_tensor_datatype("one", DataType["UINT8"])
    model.set_tensor_datatype("x", DataType["INT8"])
    return model


def test_a_custom_op_s_range_is_stated_and_read_by_the_next_rule() -> None:
    model = inferred(ranged())
    assert (model.get_tensor_datatype("m"), model.get_stated_tensor_range("m")) == (DataType["UINT4"], Range(1, 5))
    assert (model.get_tensor_datatype("y"), model.get_stated_tensor_range("y")) == (DataType["UINT3"], Range(2, 6))
    default = ranged().transform(InferDataTypes())
    assert (default.get_tensor_datatype("m"), default.get_stated_tensor_range("m")) == (DataType["UINT4"], None)


def test_a_scaled_integer_replaced_by_float32_keeps_its_range() -> None:
    model = inferred(ranged(scaled=1))
    assert (model.get_tensor_datatype("m"), model.get_stated_tensor_range("m")) == (DataType["FLOAT32"], Range(-2, 2))
    allowed = ranged(scaled=1).transform(InferDataTypes(allow_scaledint_dtypes=True, exact_ranges=True))
    assert allowed.get_tensor_datatype("m") == DataType["SCALEDINT<8>"]
    assert allowed.get_stated_tensor_range("m") == Range(-2, 2)


# an absent optional output


def test_an_absent_optional_output_is_no_tensor() -> None:
    graph = helper.make_graph(
        [helper.make_node("Dropout", ["x"], ["y", ""], name="dropout")],
        "optional",
        [helper.make_tensor_value_info("x", F, [2])],
        [helper.make_tensor_value_info("y", F, [2])],
    )
    source = qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model = inferred(ModelWrapper(source))
    assert model.get_tensor_datatype("y") == DataType["FLOAT32"]
    assert all(annotation.tensor_name for annotation in model.graph.quantization_annotation)
    assert list(model.graph.node[0].output) == ["y", ""]
    # the default rules, as before exact ranges, annotate it
    default = ModelWrapper(source).transform(InferDataTypes())
    assert get_by_name(default.graph.quantization_annotation, "", "tensor_name") is not None


# an output no rule states


def test_an_output_no_rule_states_loses_its_range_from_before() -> None:
    graph = helper.make_graph(
        [helper.make_node("Resize", ["x", "", "s"], ["y"], name="Resize_0", mode="nearest")],
        "resize",
        [helper.make_tensor_value_info("x", F, [1, 1, 2, 2])],
        [helper.make_tensor_value_info("y", F, [1, 1, 4, 4])],
    )
    model = ModelWrapper(qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 13)]))
    model.set_initializer("s", np.array([1, 1, 2, 2], dtype=np.float32))
    model.set_tensor_datatype("x", DataType["INT4"], range=Range(1, 2))
    model = inferred(model)
    assert model.get_tensor_datatype("y") == DataType["INT4"]
    assert model.get_stated_tensor_range("y") == Range(1, 2)
    # cubic interpolation overshoots its inputs (1 and 2 give 0.89453125 and
    # 2.10546875), and no rule states its range
    get_by_name(model.graph.node[0].attribute, "mode").s = b"cubic"
    model, modified = InferDataTypes(exact_ranges=True).apply(model)
    assert modified
    # the datatype as the rules leave it, the range from before cleared
    assert model.get_tensor_datatype("y") == DataType["INT4"]
    assert model.get_stated_tensor_range("y") is None
    assert not InferDataTypes(exact_ranges=True).apply(model)[1]
