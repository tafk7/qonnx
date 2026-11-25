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

"""Integer results typed by their exact ranges: InferDataTypes(exact_ranges=True)
and infer_node_datatype, against interval arithmetic worked by hand and against
Python integer sums; the default rules unchanged beside them."""

from __future__ import annotations

import pytest

import itertools
import numpy as np
from collections.abc import Sequence
from fractions import Fraction
from onnx import TensorProto, helper, numpy_helper
from typing import Any, Optional, Union

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.ranges import Range
from qonnx.core.tensor_ranges import RangeContradiction
from qonnx.transformation.infer_datatypes import InferDataTypes, _infer_node_datatype, infer_node_datatype, int_type_holding
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

Input = Union[str, tuple[str, Any, Optional[str]]]


@pytest.mark.parametrize(
    "lo, hi, name",
    [
        (0, 127, "UINT7"),
        (-16, 14, "INT5"),
        (-8, 255, "INT9"),
        (0, 0, "BINARY"),
        (0, 1, "BINARY"),
        (-1, 0, "INT1"),
        (-1, -1, "INT1"),
        (-5, -1, "INT4"),
        (-(2**42), 2**42 - 1, "INT43"),
        (-(2**42), 2**42, "INT44"),
        (0, 2**70, "UINT71"),
    ],
)
def test_int_type_holding(lo: int, hi: int, name: str) -> None:
    assert int_type_holding(lo, hi) == DataType[name]


def test_int_type_holding_refuses_a_range_past_the_envelope() -> None:
    with pytest.raises(ValueError):
        int_type_holding(0, 2**4096)


def model_of(
    op_type: str, inputs: Sequence[Input], opset: int = 13, stale: Optional[str] = None, **attributes: Any
) -> ModelWrapper:
    """One ``op_type`` node, its output y. ``inputs``: in order, (name, shape,
    annotation) for a graph input, (name, values, annotation) for an initializer
    (annotation None: none), or "" for an omitted input. ``stale``: y's annotation
    before inference."""
    graph_inputs = [
        helper.make_tensor_value_info(item[0], TensorProto.FLOAT, item[1])
        for item in inputs
        if item and not isinstance(item[1], np.ndarray)
    ]
    names = [item[0] if item else "" for item in inputs]
    node = helper.make_node(op_type, names, ["y"], **attributes)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    graph = helper.make_graph([node], "g", graph_inputs, [y])
    model = ModelWrapper(qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", opset)]))
    for item in inputs:
        if item and isinstance(item[1], np.ndarray):
            model.set_initializer(item[0], item[1].astype(np.float32))
        if item and item[2] is not None:
            model.set_tensor_datatype(item[0], DataType[item[2]])
    model = model.transform(InferShapes())
    if stale is not None:
        model.set_tensor_datatype("y", DataType[stale])
    return model


def infer(op_type: str, inputs: Sequence[Input], opset: int = 13, stale: Optional[str] = None, **attributes: Any) -> Any:
    """The datatype exact-range inference gives y."""
    model = model_of(op_type, inputs, opset, stale, **attributes)
    return model.transform(InferDataTypes(exact_ranges=True)).get_tensor_datatype("y")


def test_relu_of_an_integer_is_its_non_negative_part() -> None:
    assert infer("Relu", [("x", [4], "INT8")]) == DataType["UINT7"]
    assert infer("Relu", [("x", [4], "UINT4")]) == DataType["UINT4"]
    assert infer("Relu", [("x", [4], "BIPOLAR")]) == DataType["BINARY"]


def test_a_stale_annotation_is_replaced() -> None:
    assert infer("Relu", [("x", [4], "INT8")], stale="INT20") == DataType["UINT7"]
    assert infer("Add", [("x", [4], "INT4"), ("z", [4], "INT4")], stale="INT32") == DataType["INT5"]
    assert infer("Concat", [("x", [4], "INT4"), ("z", [4], "UINT2")], stale="INT2", axis=0) == DataType["INT4"]


def test_float_inputs_keep_the_rules_before() -> None:
    assert infer("Relu", [("x", [4], None)]) == DataType["FLOAT32"]
    # the unknown op's rule keeps a non-FLOAT32 annotation, which cannot hold the float
    # range Relu passes: a contradiction, named
    with pytest.warns(RangeContradiction, match=r"Relu states \[0, inf\], or NaN, which its datatype INT20"):
        assert infer("Relu", [("x", [4], None)], stale="INT20") == DataType["INT20"]
    assert infer("Add", [("x", [4], "INT4"), ("z", [4], None)]) == DataType["FLOAT32"]
    assert infer("Mul", [("x", [4], "INT4"), ("s", np.array([0.5]), None)]) == DataType["FLOAT32"]
    assert infer("MatMul", [("x", [1, 4], "INT4"), ("w", np.ones((4, 2)), None)]) == DataType["FLOAT32"]
    # Concat and Clip keep input 0's type
    assert infer("Concat", [("x", [4], "FIXED<8,4>"), ("z", [4], "INT4")], axis=0) == DataType["FIXED<8,4>"]
    assert infer("Clip", [("x", [4], None), ("lo", np.array(0), None), ""]) == DataType["FLOAT32"]


def test_add_sub_mul_by_interval_arithmetic() -> None:
    assert infer("Add", [("x", [4], "INT4"), ("z", [4], "INT4")]) == DataType["INT5"]
    # [0 - 15, 15 - 0]
    assert infer("Sub", [("x", [4], "UINT4"), ("z", [4], "UINT4")]) == DataType["INT5"]
    # [-8 * 7, -8 * -8]
    assert infer("Mul", [("x", [4], "INT4"), ("z", [4], "INT4")]) == DataType["INT8"]
    # [0, 3 * 3]
    assert infer("Mul", [("x", [4], "UINT2"), ("z", [4], "UINT2")]) == DataType["UINT4"]
    # an initializer by its values: [-8 + 1, 7 + 2]
    assert infer("Add", [("x", [4], "INT4"), ("b", np.array([1, 2, 1, 2]), "INT4")]) == DataType["INT5"]
    # [0 + 1, 15 + 2]
    assert infer("Add", [("x", [4], "UINT4"), ("b", np.array([1, 2, 1, 2]), "INT4")]) == DataType["UINT5"]


def test_an_initializer_holding_a_non_integer_gives_float() -> None:
    assert infer("Add", [("x", [4], "INT4"), ("b", np.array([0.5]), "INT4")]) == DataType["FLOAT32"]
    assert infer("MatMul", [("x", [1, 2], "INT4"), ("w", np.array([[0.5], [1]]), "INT4")]) == DataType["FLOAT32"]


def test_matmul_from_types_and_the_reduction_depth() -> None:
    # (-2**15)**2 * 4096 = 2**42 is reached: INT43 holds at most 2**42 - 1
    a, b = ("a", [1, 4096], "INT16"), ("b", [4096, 8], "INT16")
    assert infer("MatMul", [a, b]) == DataType["INT44"]
    # depth 2: [2 * (-8 * 7), 2 * 64]
    assert infer("MatMul", [("a", [3, 2], "INT4"), ("b", [2, 5], "INT4")]) == DataType["INT9"]
    # batched operands, depth 4: [0, 4 * 3 * 3]
    assert infer("MatMul", [("a", [2, 3, 4], "UINT2"), ("b", [2, 4, 5], "UINT2")]) == DataType["UINT6"]


def test_matmul_from_the_weights_values() -> None:
    # INT16 weights without -2**15 over k 4096: |sum| at most 2**15 * (2**15 - 1) * 4096
    weights = np.full((4096, 2), 2**15 - 1)
    weights[:, 1] = -(2**15 - 1)
    assert infer("MatMul", [("a", [1, 4096], "INT16"), ("w", weights, "INT16")]) == DataType["INT43"]
    # per column: (1, 2) gives [-8 * 3, 7 * 3]; (-1, 3) gives [-8 * 3 + 7 * -1, 7 * 3 + -8 * -1]
    weights = np.array([[1, -1], [2, 3]])
    assert infer("MatMul", [("a", [1, 2], "INT4"), ("w", weights, "INT4")]) == DataType["INT6"]
    # the same weights by their type alone: [2 * -56, 2 * 64]
    assert infer("MatMul", [("a", [1, 2], "INT4"), ("w", [2, 2], "INT4")]) == DataType["INT9"]
    # bipolar weights (+1 and -1) over UINT2 inputs, k 3: [-3 * 3, 3 * 3]
    weights = np.array([[1, -1], [1, -1], [1, -1]])
    assert infer("MatMul", [("a", [1, 3], "UINT2"), ("w", weights, "BIPOLAR")]) == DataType["INT5"]
    # a vector of weights: [0, 3 * (1 + 2)]
    assert infer("MatMul", [("a", [1, 2], "UINT2"), ("w", np.array([1, 2]), "INT4")]) == DataType["UINT4"]
    # constant left operand: rows (1, 2) and (0, -1) against UINT2: [-3, 9]
    left = np.array([[1, 2], [0, -1]])
    assert infer("MatMul", [("w", left, "INT4"), ("b", [2, 3], "UINT2")]) == DataType["INT5"]


def test_gemm() -> None:
    # B (3, 2) transposed: rows (1, 1), (2, -1), (0, 0) against INT4 x; C [-1, 1]
    weights = np.array([[1, 1], [2, -1], [0, 0]])
    inputs = [("x", [1, 2], "INT4"), ("w", weights, "INT4"), ("c", np.array([1, -1, 0]), "INT4")]
    # (2, -1): [-8 * 2 + 7 * -1, 7 * 2 + -8 * -1] = [-23, 22]; plus [-1, 1]
    assert infer("Gemm", inputs, transB=1) == DataType["INT6"]
    # alpha 2, beta 3: [-46 - 3, 44 + 3]
    assert infer("Gemm", inputs, transB=1, alpha=2.0, beta=3.0) == DataType["INT7"]
    assert infer("Gemm", inputs, transB=1, alpha=0.5) == DataType["FLOAT32"]
    # from the types: k 2, [2 * -56, 2 * 64]
    assert infer("Gemm", [("x", [1, 2], "INT4"), ("w", [2, 3], "INT4")]) == DataType["INT9"]


def test_conv() -> None:
    # two output channels over a UINT4 input, 2x2 kernels: all +1 then all -1;
    # biases 10 and -10: [0 + 10, 4 * 15 + 10] and [-4 * 15 - 10, 0 - 10]
    weights = np.stack([np.ones((1, 2, 2)), -np.ones((1, 2, 2))])
    inputs = [("x", [1, 1, 4, 4], "UINT4"), ("w", weights, "INT2"), ("b", np.array([10, -10]), "INT5")]
    assert infer("Conv", inputs, kernel_shape=[2, 2]) == DataType["INT8"]
    assert infer("Conv", inputs[:2], kernel_shape=[2, 2]) == DataType["INT7"]
    # weights by their type alone: depth 4, [4 * 15 * -2, 4 * 15 * 1]
    inputs = [("x", [1, 1, 4, 4], "UINT4"), ("w", [2, 1, 2, 2], "INT2")]
    assert infer("Conv", inputs, kernel_shape=[2, 2]) == DataType["INT8"]
    # padding zeros: a bipolar input's sums may be 0, which its range already covers
    inputs = [("x", [1, 1, 4, 4], "BIPOLAR"), ("w", np.ones((1, 1, 3, 3)), "INT2")]
    assert infer("Conv", inputs, kernel_shape=[3, 3], pads=[1, 1, 1, 1]) == DataType["INT5"]


def test_max_and_min() -> None:
    assert infer("Max", [("x", [4], "INT4"), ("z", [4], "UINT2")]) == DataType["UINT3"]
    assert infer("Min", [("x", [4], "INT4"), ("z", [4], "UINT2")]) == DataType["INT4"]
    assert infer("Max", [("x", [4], "INT4"), ("z", np.array([0]), "INT4")]) == DataType["UINT3"]
    assert infer("Min", [("x", [4], "INT8"), ("v", [4], "INT4"), ("z", [4], "UINT2")]) == DataType["INT8"]


def test_neg() -> None:
    assert infer("Neg", [("x", [4], "INT8")]) == DataType["INT9"]
    assert infer("Neg", [("x", [4], "UINT4")]) == DataType["INT5"]


def test_clip_by_its_bounds() -> None:
    zero, six = ("lo", np.array(0), None), ("hi", np.array(6), None)
    assert infer("Clip", [("x", [4], "INT8"), zero, six]) == DataType["UINT3"]
    assert infer("Clip", [("x", [4], "INT8"), zero, ""]) == DataType["UINT7"]
    assert infer("Clip", [("x", [4], "INT8"), "", six]) == DataType["INT8"]
    # a bound beyond the input's range is never reached
    assert infer("Clip", [("x", [4], "INT4"), ("lo", np.array(-100.5), None), six]) == DataType["INT4"]
    # one within it may be the result
    assert infer("Clip", [("x", [4], "INT4"), ("lo", np.array(0.5), None), six]) == DataType["FLOAT32"]
    # a bound that is a graph input, by its annotation: [-128, min(127, 3)]
    assert infer("Clip", [("x", [4], "INT8"), "", ("hi", [], "UINT2")]) == DataType["INT8"]
    # [max(-8, 0), max(7, 3)]
    assert infer("Clip", [("x", [4], "INT4"), ("lo", [], "UINT2"), ""]) == DataType["UINT3"]
    # opset 6: bounds as attributes
    assert infer("Clip", [("x", [4], "INT8")], opset=6, min=-2.0, max=5.0) == DataType["INT4"]


def test_concat_is_the_union_of_its_inputs() -> None:
    assert infer("Concat", [("x", [4], "INT4"), ("z", [4], "UINT8")], axis=0) == DataType["INT9"]
    assert infer("Concat", [("x", [4], "UINT2"), ("z", [4], "UINT8")], axis=0) == DataType["UINT8"]


def test_identity_ops_keep_their_input_type() -> None:
    assert infer("Transpose", [("x", [2, 3], "INT4")], perm=[1, 0]) == DataType["INT4"]
    assert infer("Identity", [("x", [2, 3], "UINT7")]) == DataType["UINT7"]


def test_the_exact_range_is_stated_with_its_type() -> None:
    # [-8 + -8, 7 + 7]: INT5 holds [-16, 15]
    model = model_of("Add", [("x", [4], "INT4"), ("z", [4], "INT4")])
    model = model.transform(InferDataTypes(exact_ranges=True))
    assert model.get_tensor_datatype("y") == DataType["INT5"]
    assert model.get_stated_tensor_range("y") == Range(-16, 14)
    # a stated input range narrows the result: [-2 + -8, 3 + 7]
    model = model_of("Add", [("x", [4], "INT4"), ("z", [4], "INT4")])
    model.set_tensor_datatype("x", DataType["INT4"], range=Range(-2, 3))
    model = model.transform(InferDataTypes(exact_ranges=True))
    assert model.get_tensor_datatype("y") == DataType["INT5"]
    assert model.get_stated_tensor_range("y") == Range(-10, 10)
    # a range its datatype states in full is not stored
    model = model_of("Relu", [("x", [4], "UINT4")]).transform(InferDataTypes(exact_ranges=True))
    assert model.get_tensor_datatype("y") == DataType["UINT4"]
    assert model.get_stated_tensor_range("y") is None


@pytest.mark.parametrize(
    "op_type, inputs, attributes, default",
    [
        ("Add", [("x", [4], "INT4"), ("z", [4], "INT4")], {}, "INT32"),
        ("MatMul", [("a", [3, 2], "INT4"), ("b", [2, 5], "INT4")], {}, "INT32"),
        ("Conv", [("x", [1, 1, 4, 4], "UINT4"), ("w", [2, 1, 2, 2], "INT2")], {"kernel_shape": [2, 2]}, "INT32"),
        ("Mul", [("x", [4], "UINT2"), ("z", [4], "UINT2")], {}, "UINT32"),
        ("Relu", [("x", [4], "INT8")], {}, "FLOAT32"),
        ("Concat", [("x", [4], "INT4"), ("z", [4], "UINT8")], {"axis": 0}, "INT4"),
        ("Clip", [("x", [4], "INT8"), ("lo", np.array(0), None), ""], {}, "INT8"),
    ],
)
def test_the_default_keeps_the_rules_before(
    op_type: str, inputs: Sequence[Input], attributes: dict[str, Any], default: str
) -> None:
    """Without exact_ranges, the answers are those before it, and no range is stored."""
    model = model_of(op_type, inputs, **attributes)
    before = model.transform(InferDataTypes())
    assert before.get_tensor_datatype("y") == DataType[default]
    assert before.get_stated_tensor_range("y") is None
    model = model_of(op_type, inputs, **attributes)
    assert infer_node_datatype(model, model.graph.node[0], False) == (default != "FLOAT32")
    assert model.get_tensor_datatype("y") == DataType[default]
    model = model_of(op_type, inputs, **attributes)
    _infer_node_datatype(model, model.graph.node[0], False)
    assert model.get_tensor_datatype("y") == DataType[default]


def test_infer_node_datatype_opts_in_per_call() -> None:
    model = model_of("Add", [("x", [4], "INT4"), ("z", [4], "INT4")])
    node = model.graph.node[0]
    assert infer_node_datatype(model, node, False, exact_ranges=True)
    assert model.get_tensor_datatype("y") == DataType["INT5"]
    assert model.get_stated_tensor_range("y") == Range(-16, 14)
    # nothing changes on a second call
    assert not infer_node_datatype(model, node, False, exact_ranges=True)
    # the default rules replace both the datatype and the range
    assert infer_node_datatype(model, node, False)
    assert model.get_tensor_datatype("y") == DataType["INT32"]
    assert model.get_stated_tensor_range("y") is None


def test_a_range_change_alone_is_a_change() -> None:
    model = model_of("Add", [("x", [4], "INT4"), ("z", [4], "INT4")])
    node = model.graph.node[0]
    assert infer_node_datatype(model, node, False, exact_ranges=True)
    model.set_tensor_datatype("x", DataType["INT4"], range=Range(-7, 7))
    # [-7 + -8, 7 + 7]: still INT5, with another range
    assert infer_node_datatype(model, node, False, exact_ranges=True)
    assert model.get_tensor_datatype("y") == DataType["INT5"]
    assert model.get_stated_tensor_range("y") == Range(-15, 14)


def test_a_sum_past_the_envelope_is_refused() -> None:
    model = model_of("Add", [("x", [4], "INT4096"), ("z", [4], "INT4096")])
    with pytest.raises(ValueError):
        model.transform(InferDataTypes(exact_ranges=True))


# ReduceSum, checked against Python integer sums


def reduction(
    shape: Sequence[Union[int, str, None]],
    axes: Optional[tuple[int, ...]],
    *,
    opset: int = 13,
    noop: int = 0,
    keepdims: int = 0,
) -> ModelWrapper:
    attributes: dict[str, Any] = {"keepdims": keepdims}
    inputs = ["x"]
    if opset < 13:
        if axes is not None:
            attributes["axes"] = axes
    else:
        attributes["noop_with_empty_axes"] = noop
        if axes is not None:
            inputs.append("axes")
    graph = helper.make_graph(
        [helper.make_node("ReduceSum", inputs, ["y"], name="sum", **attributes)],
        "sum",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = ModelWrapper(qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", opset)]))
    model.set_tensor_datatype("x", DataType["INT4"])
    if axes is not None and opset >= 13:
        model.set_initializer("axes", np.asarray(axes, dtype=np.int64))
    return model


def inferred(model: ModelWrapper) -> ModelWrapper:
    return model.transform(InferDataTypes(exact_ranges=True), cleanup=False)


@pytest.mark.parametrize(
    ("dtype", "values", "expected_dtype", "expected_range"),
    [
        ("UINT24", [16777215, 1, 1], "UINT26", Range(0, 3 * (2**24 - 1))),
        ("INT16", [32767] * 1024 + [1], "INT27", Range(-32768 * 1025, 32767 * 1025)),
    ],
    ids=["UINT24-odd", "INT16-long"],
)
def test_an_integer_sum_above_float32_has_its_exact_type(
    dtype: str, values: list[int], expected_dtype: str, expected_range: Range
) -> None:
    model = reduction([1, len(values), 1], (0, 1))
    model.set_tensor_shape("y", [1])
    model.set_tensor_datatype("x", DataType[dtype])
    model = inferred(model)
    given = np.asarray(values, dtype=np.float32).reshape(1, len(values), 1)
    # every input is representable; only the sum needs the wider type
    assert [int(value) for value in given.flat] == values
    assert model.get_tensor_datatype("y") == DataType[expected_dtype]
    assert model.get_tensor_range("y") == expected_range
    assert DataType[expected_dtype].allowed(sum(values))


@pytest.mark.parametrize("opset", [11, 13])
@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("dtype", ["BINARY", "INT2", "UINT2", "BIPOLAR"])
def test_sum_range_and_smallest_type_hold_every_small_integer_vector(opset: int, keepdims: int, dtype: str) -> None:
    element = DataType[dtype]
    members = [value for value in range(int(element.min()), int(element.max()) + 1) if element.allowed(value)]
    totals = [sum(values) for values in itertools.product(members, repeat=3)]
    model = reduction([2, 3], (-1,), opset=opset, keepdims=keepdims)
    model.set_tensor_datatype("x", element)
    model = inferred(model)
    assert model.has_tensor_datatype("y")
    result = model.get_tensor_datatype("y")
    assert result.is_integer()
    assert model.get_tensor_range("y") == Range(min(totals), max(totals))
    assert all(result.allowed(value) for value in totals)
    # search candidate widths independently of the inference's bit-length rule
    bits = 1
    while True:
        low, high = (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) if min(totals) < 0 else (0, 2**bits - 1)
        if low <= min(totals) and max(totals) <= high:
            break
        bits += 1
    assert result.bitwidth() == bits
    assert result.signed() == (min(totals) < 0)


@pytest.mark.parametrize("axes", [None, (), (0, 1), (-3, -2)])
def test_multiple_or_default_axes_use_the_whole_reduced_extent(axes: Optional[tuple[int, ...]]) -> None:
    model = inferred(reduction([2, 3, 4], axes))
    positions = 24 if axes is None or not axes else 6
    assert model.get_tensor_range("y") == Range(-8 * positions, 7 * positions)


@pytest.mark.parametrize("axes", [None, ()])
def test_empty_axes_noop_keeps_the_input_integer_range(axes: Optional[tuple[int, ...]]) -> None:
    model = inferred(reduction(["N", 3], axes, noop=1))
    assert model.get_tensor_datatype("y") == DataType["INT4"]
    assert model.get_tensor_range("y") == Range(-8, 7)


def test_stated_input_range_sizes_the_sum() -> None:
    model = reduction([2, 3], (0,))
    model.set_tensor_datatype("x", DataType["INT4"], range=Range(-2, 3))
    model = inferred(model)
    assert model.get_tensor_datatype("y") == DataType["INT4"]
    assert model.get_tensor_range("y") == Range(-4, 6)


@pytest.mark.parametrize(("shape", "expected"), [([], Range(-8, 7)), ([0, 3], Range(0, 0))])
def test_scalar_and_empty_sums_have_their_exact_range(shape: list[int], expected: Range) -> None:
    model = inferred(reduction(shape, None))
    assert model.get_tensor_range("y") == expected


@pytest.mark.parametrize("dimension", ["N", None])
def test_unknown_reduced_extent_leaves_no_integer_guess(dimension: Optional[str]) -> None:
    model = inferred(reduction([dimension, 3], (0,)))
    assert not model.has_tensor_datatype("y")
    assert model.get_stated_tensor_range("y") is None


def test_unknown_unreduced_extent_does_not_hide_the_known_sum_extent() -> None:
    model = inferred(reduction(["N", 3], (1,)))
    assert model.get_tensor_datatype("y") == DataType["INT6"]
    assert model.get_tensor_range("y") == Range(-24, 21)


@pytest.mark.parametrize("dtype", ["INT4096", "UINT4096"])
def test_a_sum_beyond_the_envelope_leaves_no_integer_type(dtype: str) -> None:
    """Two terms of the widest integer need one bit more than any integer type holds."""
    model = reduction([1, 2], (0, 1))
    model.set_tensor_datatype("x", DataType[dtype])
    model = inferred(model)
    assert not model.has_tensor_datatype("y")
    assert model.get_stated_tensor_range("y") is None


def test_dynamic_or_overridable_axes_leave_no_integer_guess() -> None:
    for has_initializer in (False, True):
        model = reduction([2, 3], (0,))
        info = model.get_tensor_valueinfo("axes")
        if info is not None:
            model.graph.value_info.remove(info)
        model.graph.input.append(helper.make_tensor_value_info("axes", TensorProto.INT64, [1]))
        if not has_initializer:
            model.del_initializer("axes")
        model = inferred(model)
        assert not model.has_tensor_datatype("y")
        assert model.get_stated_tensor_range("y") is None


def test_reinference_drops_a_bound_when_the_reduced_extent_becomes_unknown() -> None:
    model = inferred(reduction([2, 3], (0,)))
    dimension = model.graph.input[0].type.tensor_type.shape.dim[0]
    dimension.dim_param = "N"
    model = inferred(model)
    assert not model.has_tensor_datatype("y")
    assert model.get_stated_tensor_range("y") is None


def test_float_input_does_not_acquire_an_integer_annotation() -> None:
    model = reduction([2, 3], (0,))
    model.set_tensor_datatype("x", DataType["FLOAT32"])
    model = inferred(model)
    assert model.get_tensor_datatype("y") == DataType["FLOAT32"]
    assert model.get_stated_tensor_range("y") is None


# FIXED<64,32>: its greatest value exactly


def test_a_fixed_point_input_keeps_its_exact_end() -> None:
    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"])],
        "relu",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )
    model = ModelWrapper(qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 13)]))
    model.set_tensor_datatype("x", DataType["FIXED<64,32>"])
    assert model.get_stated_tensor_range("x") is None
    # (2**63 - 1) / 2**32, which float64 rounds up to 2**31
    assert model.get_tensor_range("x").greatest == Fraction(2**63 - 1, 2**32)
    model = model.transform(InferDataTypes(exact_ranges=True))
    # a fixed-point input has no integer range rule
    assert model.get_tensor_datatype("y") == DataType["FLOAT32"]


# Integers at int64's ends, exactly


def int64_model(op_type: str, x_shape: list[int], constants: dict[str, np.ndarray], **attributes: Any) -> ModelWrapper:
    """One ``op_type`` node over an INT64 graph input x and int64 initializers
    (``constants``, in order; "" for an omitted input), its output y."""
    names = ["x"] + list(constants)
    graph = helper.make_graph(
        [helper.make_node(op_type, names, ["y"], **attributes)],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.INT64, x_shape)],
        [helper.make_tensor_value_info("y", TensorProto.INT64, None)],
        initializer=[numpy_helper.from_array(values, name) for name, values in constants.items() if name],
    )
    return ModelWrapper(qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 13)]))


def int64_dot(op_type: str, value: int, depth: int) -> ModelWrapper:
    """x (BINARY) times ``depth`` weights all equal to ``value``, by ``op_type``."""
    weights = np.full((depth, 1), value, dtype=np.int64)
    if op_type == "MatMul":
        return int64_model("MatMul", [1, depth], {"w": weights})
    if op_type == "Gemm":
        return int64_model("Gemm", [1, depth], {"w": weights})
    return int64_model("Conv", [1, depth, 1, 1], {"w": weights.reshape(1, depth, 1, 1)}, kernel_shape=[1, 1])


@pytest.mark.parametrize("op_type", ["MatMul", "Gemm", "Conv"])
@pytest.mark.parametrize(
    "value, depth",
    [
        # the signed minimum, whose fixed-width magnitude is itself
        (-(2**63), 2),
        (-(2**63), 1),
        (2**63 - 1, 3),
        # either side of where int64 sums stop being proved safe (2**62)
        (2**61 - 1, 2),
        (-(2**61 - 1), 2),
        (2**61, 2),
        (-(2**61), 2),
        # past int64 itself
        (2**62, 2),
        (-(2**62), 3),
    ],
)
def test_dot_products_of_weights_at_int64_ends_are_exact(op_type: str, value: int, depth: int) -> None:
    model = int64_dot(op_type, value, depth)
    model.set_tensor_datatype("x", DataType["BINARY"])
    model.set_tensor_datatype("w", DataType["INT64"])
    model = model.transform(InferDataTypes(exact_ranges=True))
    # BINARY x: every partial sum lies between 0 and depth * value
    expected = (min(0, depth * value), max(0, depth * value))
    assert model.get_tensor_datatype("y") == int_type_holding(*expected)
    assert model.get_tensor_range("y") == Range(*expected)


def test_the_signed_minimum_twice_is_int65() -> None:
    model = int64_dot("MatMul", -(2**63), 2)
    model.set_tensor_datatype("x", DataType["BINARY"])
    model.set_tensor_datatype("w", DataType["INT64"])
    model = model.transform(InferDataTypes(exact_ranges=True))
    assert model.get_tensor_datatype("y") == DataType["INT65"]
    assert model.get_stated_tensor_range("y") == Range(-(2**64), 0)


@pytest.mark.parametrize(
    "x_range, low, high, expected",
    [
        # 2**53 + 1 is not a float64: through float it would be 2**53
        (Range(2**53 + 1, 2**53 + 1), None, 2**53 + 1, Range(2**53 + 1, 2**53 + 1)),
        (None, 2**53 + 1, 2**53 + 1, Range(2**53 + 1, 2**53 + 1)),
        (None, -(2**53) - 1, None, Range(-(2**53) - 1, 2**63 - 1)),
        # int64's ends: through float the greatest would be 2**63
        (None, -(2**63), 2**63 - 1, None),
        (None, -(2**63) + 1, 2**63 - 2, Range(-(2**63) + 1, 2**63 - 2)),
        (Range(-(2**63), 0), None, -(2**53) - 1, Range(-(2**63), -(2**53) - 1)),
    ],
)
def test_integer_clip_bounds_are_read_exactly(
    x_range: Optional[Range], low: Optional[int], high: Optional[int], expected: Optional[Range]
) -> None:
    constants = {
        "lo" if low is not None else "": np.array(low if low is not None else 0, dtype=np.int64),
        "hi": np.array(high if high is not None else 2**63 - 1, dtype=np.int64),
    }
    model = int64_model("Clip", [1], constants)
    model.set_tensor_datatype("x", DataType["INT64"], range=x_range)
    model = model.transform(InferDataTypes(exact_ranges=True))
    full = Range(-(2**63), 2**63 - 1)
    assert model.get_tensor_datatype("y") == int_type_holding(*(expected or full).integers())
    assert model.get_tensor_range("y") == (expected or full)
