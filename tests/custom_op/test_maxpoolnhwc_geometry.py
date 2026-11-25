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

import pytest

import copy
import numpy as np
from onnx import TensorProto, helper
from typing import Any

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model


def _model(
    element: int, attributes: dict[str, Any], shape: list[int], output_shape: list[int], custom: bool = True
) -> ModelWrapper:
    node = helper.make_node(
        "MaxPoolNHWC" if custom else "MaxPool",
        ["x"],
        ["y"],
        domain="qonnx.custom_op.general" if custom else "",
        **attributes
    )
    graph = helper.make_graph(
        [node],
        "pool",
        [helper.make_tensor_value_info("x", element, shape)],
        [helper.make_tensor_value_info("y", element, output_shape)],
    )
    return ModelWrapper(qonnx_make_model(graph))


@pytest.mark.parametrize(
    "element,container",
    [
        (TensorProto.FLOAT, np.float32),
        (TensorProto.DOUBLE, np.float64),
        (TensorProto.INT8, np.int8),
        (TensorProto.UINT8, np.uint8),
        (TensorProto.FLOAT16, np.float16),
    ],
)
@pytest.mark.parametrize("exact", [False, True])
def test_pool_uses_the_declared_element_type(element: int, container: type, exact: bool) -> None:
    model = _model(element, {"kernel_shape": [2, 2]}, [1, 4, 4, 1], [1, 3, 3, 1])
    values = np.arange(16, dtype=container).reshape(1, 4, 4, 1)
    original = model.graph.node[0].SerializeToString()
    result = execute_onnx(model, {"x": values}, exact_integers=exact)["y"]
    assert result.dtype == values.dtype
    np.testing.assert_array_equal(result, values[:, 1:, 1:, :])
    assert model.graph.node[0].SerializeToString() == original


@pytest.mark.parametrize(
    "attributes,expected",
    [
        ({"kernel_shape": [2, 2], "strides": [2, 2]}, [2, 3]),
        ({"kernel_shape": [2, 2], "dilations": [2, 2]}, [3, 4]),
        ({"kernel_shape": [2, 2], "strides": [2, 2], "pads": [0, 1, 1, 0]}, [3, 3]),
        ({"kernel_shape": [3, 3], "strides": [2, 2], "auto_pad": "SAME_UPPER"}, [3, 3]),
        ({"kernel_shape": [2, 2], "strides": [2, 2], "auto_pad": "SAME_LOWER"}, [3, 3]),
        ({"kernel_shape": [3, 3], "strides": [2, 2], "auto_pad": "VALID"}, [2, 2]),
        ({"kernel_shape": [2, 2], "strides": [2, 2], "pads": [1, 1, 1, 1], "ceil_mode": 1}, [3, 4]),
    ],
)
def test_shape_and_values_match_full_standard_pool_geometry(attributes: dict[str, Any], expected: list[int]) -> None:
    output = [1, *expected, 2]
    model = _model(TensorProto.FLOAT, attributes, [1, 5, 6, 2], output)
    original = model.graph.node[0].SerializeToString()
    inferred = copy.deepcopy(model)
    inferred.graph.output[0].type.tensor_type.ClearField("shape")
    inferred = inferred.transform(InferShapes())
    assert inferred.get_tensor_shape("y") == output
    values = np.arange(60, dtype=np.float32).reshape(1, 5, 6, 2)
    standard = _model(TensorProto.FLOAT, attributes, [1, 2, 5, 6], [1, 2, *expected], False)
    reference = execute_onnx(standard, {"x": values.transpose(0, 3, 1, 2)})["y"].transpose(0, 2, 3, 1)
    result = execute_onnx(model, {"x": values})["y"]
    assert result.shape == tuple(output)
    np.testing.assert_array_equal(result, reference)
    assert model.graph.node[0].SerializeToString() == original


def test_a_mismatched_integer_feed_keeps_the_standard_execution_contract() -> None:
    attrs = {"kernel_shape": [2, 2]}
    custom = _model(TensorProto.FLOAT, attrs, [1, 4, 4, 1], [1, 3, 3, 1])
    standard = _model(TensorProto.FLOAT, attrs, [1, 1, 4, 4], [1, 1, 3, 3], False)
    values = np.arange(16, dtype=np.int64).reshape(1, 4, 4, 1)
    original = custom.graph.node[0].SerializeToString()
    for model, feed in [(custom, values), (standard, values.transpose(0, 3, 1, 2))]:
        with pytest.raises(Exception, match="Unexpected input data type"):
            execute_onnx(model, {"x": feed}, exact_integers=False)
        result = execute_onnx(model, {"x": feed}, exact_integers=True)["y"]
        assert result.dtype == np.float32
    assert custom.graph.node[0].SerializeToString() == original


def test_failure_leaves_the_source_node_unchanged() -> None:
    model = _model(TensorProto.FLOAT, {"kernel_shape": [0, 2]}, [1, 4, 4, 1], [1, 3, 3, 1])
    original = model.graph.node[0].SerializeToString()
    with pytest.raises(Exception):
        execute_onnx(model, {"x": np.ones((1, 4, 4, 1), dtype=np.float32)})
    assert model.graph.node[0].SerializeToString() == original


def test_nested_exact_execution_keeps_integer_annotations() -> None:
    model = _model(TensorProto.FLOAT, {"kernel_shape": [2, 2]}, [1, 4, 4, 1], [1, 3, 3, 1])
    model.set_tensor_datatype("x", DataType["INT26"])
    model.set_tensor_datatype("y", DataType["INT26"])
    result = execute_onnx(model, {"x": np.full((1, 4, 4, 1), 16777217, dtype=np.int64)}, exact_integers=True)["y"]
    assert result.dtype == np.float64
    assert result.item(0) == 16777217


def test_missing_element_type_keeps_the_float_default() -> None:
    model = _model(TensorProto.FLOAT, {"kernel_shape": [2, 2]}, [1, 4, 4, 1], [1, 3, 3, 1])
    model.graph.input[0].type.tensor_type.ClearField("elem_type")
    context = {"x": np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1), "y": np.zeros((1, 3, 3, 1), dtype=np.float32)}
    getCustomOp(model.graph.node[0]).execute_node(context, model.graph)
    np.testing.assert_array_equal(context["y"], context["x"][:, 1:, 1:, :])


@pytest.mark.parametrize("exact", [False, True])
def test_int64_declarations_keep_the_standard_pool_schema_refusal(exact: bool) -> None:
    attributes = {"kernel_shape": [2, 2]}
    custom = _model(TensorProto.INT64, attributes, [1, 4, 4, 1], [1, 3, 3, 1])
    standard = _model(TensorProto.INT64, attributes, [1, 1, 4, 4], [1, 1, 3, 3], False)
    values = np.arange(16, dtype=np.int64).reshape(1, 4, 4, 1)
    original = custom.graph.node[0].SerializeToString()
    for model, feed in [(custom, values), (standard, values.transpose(0, 3, 1, 2))]:
        with pytest.raises(Exception, match="tensor\\(int64\\).*is invalid"):
            execute_onnx(model, {"x": feed}, exact_integers=exact)
    assert custom.graph.node[0].SerializeToString() == original


def test_existing_float_graph_keeps_opset11_values_and_serialized_bytes() -> None:
    attributes = {"kernel_shape": [2, 2], "strides": [2, 2], "pads": [1, 1, 1, 1], "ceil_mode": 1}
    custom = _model(TensorProto.FLOAT, attributes, [1, 4, 4, 1], [1, 3, 3, 1])
    standard = _model(TensorProto.FLOAT, attributes, [1, 1, 4, 4], [1, 1, 3, 3], False)
    assert custom.get_opset_imports()[""] == standard.get_opset_imports()[""] == 11
    original = custom.model.SerializeToString(deterministic=True)
    values = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1)
    reference = execute_onnx(standard, {"x": values.transpose(0, 3, 1, 2)})["y"].transpose(0, 2, 3, 1)
    result = execute_onnx(custom, {"x": values})["y"]
    np.testing.assert_array_equal(result, reference)
    assert custom.model.SerializeToString(deterministic=True) == original
