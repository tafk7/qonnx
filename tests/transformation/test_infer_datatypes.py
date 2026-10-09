# Copyright (c) 2020, Xilinx
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
# * Neither the name of QONNX nor the names of its
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

import pytest

import numpy as np
from onnx import TensorProto, helper
from pkgutil import get_data

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes, infer_mac_result_dtype, int_type_holding
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model
from qonnx.util.test import download_model


def test_infer_mac_dtype_result():
    # dtype prototypes
    is32 = DataType["INT32"]
    iu32 = DataType["UINT32"]
    f32 = DataType["FLOAT32"]
    is4 = DataType["INT4"]
    iu4 = DataType["UINT4"]
    fx4 = DataType["FIXED<4,2>"]
    si4 = DataType["SCALEDINT<4>"]
    si32 = DataType["SCALEDINT<32>"]
    # test several 2-input (e.g. weights, inputs) cases
    assert infer_mac_result_dtype([iu4, iu4], None, False) == iu32
    assert infer_mac_result_dtype([iu4, is4], None, False) == is32
    assert infer_mac_result_dtype([iu4, iu4], None, True) == is32
    assert infer_mac_result_dtype([iu4, fx4], None, False) == si32
    assert infer_mac_result_dtype([fx4, si4], None, False) == si32
    assert infer_mac_result_dtype([is4, si4], None, False) == si32
    assert infer_mac_result_dtype([f32, iu4], f32, False) == f32
    assert infer_mac_result_dtype([f32, si4], f32, False) == f32
    # test several 3-input (e.g. weights, inputs, biases) cases
    assert infer_mac_result_dtype([iu4, iu4, iu4], None, False) == iu32
    assert infer_mac_result_dtype([iu4, iu4, is4], None, False) == is32
    assert infer_mac_result_dtype([is4, iu4, fx4], None, False) == si32
    assert infer_mac_result_dtype([is4, iu4, f32], f32, False) == f32


def test_infer_datatypes():
    raw_m = get_data("qonnx.data", "onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    # this model has no DataType info, so add some DataType annotation
    # to make things a bit more exciting
    model.set_tensor_datatype("global_in", DataType["UINT8"])
    # Conv with int weights + inputs will have int output datatype: 25 weights
    # per output channel, +1 in the first, -1 in the others
    weights = -np.ones(model.get_initializer("Conv_0_param0").shape, dtype=np.float32)
    weights[0] = 1
    model.set_initializer("Conv_0_param0", weights)
    model.set_tensor_datatype("Conv_0_param0", DataType["INT4"])
    model = model.transform(InferDataTypes())
    assert model.get_tensor_datatype("global_in") == DataType["UINT8"]
    # [-25 * 255, 25 * 255]
    assert model.get_tensor_datatype("Conv_0_out0") == DataType["INT14"]
    assert model.get_tensor_datatype("Relu_0_out0") == DataType["FLOAT32"]
    assert model.get_tensor_datatype("global_out") == DataType["FLOAT32"]


def test_infer_datatypes_scaledint():
    orig_model = download_model("FINN-CNV_W2A2", do_cleanup=True, return_modelwrapper=True)
    model = orig_model.transform(InferDataTypes(allow_scaledint_dtypes=True))
    assert model.get_tensor_datatype("Quant_9_out0") == DataType["SCALEDINT<8>"]
    assert model.get_tensor_datatype("Conv_0_out0") == DataType["SCALEDINT<32>"]
    model = orig_model.transform(InferDataTypes(allow_scaledint_dtypes=False))
    assert model.get_tensor_datatype("Quant_9_out0") == DataType["FLOAT32"]
    assert model.get_tensor_datatype("Conv_0_out0") == DataType["FLOAT32"]
    # no dtypes should be inferred as SCALEDINT
    for tname in model.get_all_tensor_names():
        tensor_dt = model.get_tensor_datatype(tname)
        assert not ("SCALEDINT" in tensor_dt.get_canonical_name())


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
def test_int_type_holding(lo, hi, name):
    assert int_type_holding(lo, hi) == DataType[name]


def infer(op_type, inputs, opset=13, stale=None, **attributes):
    """The type inferred for the output of one ``op_type`` node. ``inputs``: in
    order, (name, shape, annotation) for a graph input, (name, values, annotation)
    for an initializer (annotation None: none), or "" for an omitted input.
    ``stale``: the output's annotation before inference."""
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
    return model.transform(InferDataTypes()).get_tensor_datatype("y")


def test_relu_of_an_integer_is_its_non_negative_part():
    assert infer("Relu", [("x", [4], "INT8")]) == DataType["UINT7"]
    assert infer("Relu", [("x", [4], "UINT4")]) == DataType["UINT4"]
    assert infer("Relu", [("x", [4], "BIPOLAR")]) == DataType["BINARY"]


def test_a_stale_annotation_is_replaced():
    assert infer("Relu", [("x", [4], "INT8")], stale="INT20") == DataType["UINT7"]
    assert infer("Add", [("x", [4], "INT4"), ("z", [4], "INT4")], stale="INT32") == DataType["INT5"]
    assert infer("Concat", [("x", [4], "INT4"), ("z", [4], "UINT2")], stale="INT2", axis=0) == DataType["INT4"]


def test_float_inputs_keep_the_rules_before():
    assert infer("Relu", [("x", [4], None)]) == DataType["FLOAT32"]
    # the unknown op's rule keeps a non-FLOAT32 annotation
    assert infer("Relu", [("x", [4], None)], stale="INT20") == DataType["INT20"]
    assert infer("Add", [("x", [4], "INT4"), ("z", [4], None)]) == DataType["FLOAT32"]
    assert infer("Mul", [("x", [4], "INT4"), ("s", np.array([0.5]), None)]) == DataType["FLOAT32"]
    assert infer("MatMul", [("x", [1, 4], "INT4"), ("w", np.ones((4, 2)), None)]) == DataType["FLOAT32"]
    # Concat and Clip keep input 0's type
    assert infer("Concat", [("x", [4], "FIXED<8,4>"), ("z", [4], "INT4")], axis=0) == DataType["FIXED<8,4>"]
    assert infer("Clip", [("x", [4], None), ("lo", np.array(0), None), ""]) == DataType["FLOAT32"]


def test_add_sub_mul_by_interval_arithmetic():
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


def test_an_initializer_holding_a_non_integer_gives_float():
    assert infer("Add", [("x", [4], "INT4"), ("b", np.array([0.5]), "INT4")]) == DataType["FLOAT32"]
    assert infer("MatMul", [("x", [1, 2], "INT4"), ("w", np.array([[0.5], [1]]), "INT4")]) == DataType["FLOAT32"]


def test_matmul_from_types_and_the_reduction_depth():
    # (-2**15)**2 * 4096 = 2**42 is reached: INT43 holds at most 2**42 - 1
    a, b = ("a", [1, 4096], "INT16"), ("b", [4096, 8], "INT16")
    assert infer("MatMul", [a, b]) == DataType["INT44"]
    # depth 2: [2 * (-8 * 7), 2 * 64]
    assert infer("MatMul", [("a", [3, 2], "INT4"), ("b", [2, 5], "INT4")]) == DataType["INT9"]
    # batched operands, depth 4: [0, 4 * 3 * 3]
    assert infer("MatMul", [("a", [2, 3, 4], "UINT2"), ("b", [2, 4, 5], "UINT2")]) == DataType["UINT6"]


def test_matmul_from_the_weights_values():
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


def test_gemm():
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


def test_conv():
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


def test_max_and_min():
    assert infer("Max", [("x", [4], "INT4"), ("z", [4], "UINT2")]) == DataType["UINT3"]
    assert infer("Min", [("x", [4], "INT4"), ("z", [4], "UINT2")]) == DataType["INT4"]
    assert infer("Max", [("x", [4], "INT4"), ("z", np.array([0]), "INT4")]) == DataType["UINT3"]
    assert infer("Min", [("x", [4], "INT8"), ("v", [4], "INT4"), ("z", [4], "UINT2")]) == DataType["INT8"]


def test_neg():
    assert infer("Neg", [("x", [4], "INT8")]) == DataType["INT9"]
    assert infer("Neg", [("x", [4], "UINT4")]) == DataType["INT5"]


def test_clip_by_its_bounds():
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


def test_concat_is_the_union_of_its_inputs():
    assert infer("Concat", [("x", [4], "INT4"), ("z", [4], "UINT8")], axis=0) == DataType["INT9"]
    assert infer("Concat", [("x", [4], "UINT2"), ("z", [4], "UINT8")], axis=0) == DataType["UINT8"]


def test_identity_ops_keep_their_input_type():
    assert infer("Transpose", [("x", [2, 3], "INT4")], perm=[1, 0]) == DataType["INT4"]
    assert infer("Identity", [("x", [2, 3], "UINT7")]) == DataType["UINT7"]
