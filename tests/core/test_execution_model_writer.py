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

import numpy as np
from onnx import TensorProto, helper

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx_and_make_model
from qonnx.util.basic import qonnx_make_model


def _add_model() -> ModelWrapper:
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "one"], ["middle"]), helper.make_node("Identity", ["middle"], ["y"])],
        "record-execution",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
        value_info=[helper.make_tensor_value_info("middle", TensorProto.FLOAT, [2])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_initializer("one", np.ones(2, dtype=np.float32))
    return model


@pytest.mark.parametrize("exact", [False, True])
def test_ordinary_execution_records_only_named_tensor_values(exact: bool) -> None:
    model = _add_model()
    original = model.model.SerializeToString(deterministic=True)
    assert model.make_empty_exec_context()[""] is None
    values = np.asarray([2, 4], dtype=np.float32)
    recorded = execute_onnx_and_make_model(model, {"x": values}, exact_integers=exact)
    assert recorded is not model
    assert model.model.SerializeToString(deterministic=True) == original
    assert {tensor.name for tensor in recorded.graph.initializer} == {"x", "one", "middle", "y"}
    np.testing.assert_array_equal(recorded.get_initializer("x"), values)
    np.testing.assert_array_equal(recorded.get_initializer("one"), np.ones(2, dtype=np.float32))
    for name in ("middle", "y"):
        np.testing.assert_array_equal(recorded.get_initializer(name), values + 1)
    assert "middle" in {info.name for info in recorded.graph.output}
    assert all(info.name for info in (*recorded.graph.input, *recorded.graph.value_info, *recorded.graph.output))


@pytest.mark.parametrize("exact,expected,container", [(False, 16777216, np.float32), (True, 16777217, np.float64)])
def test_real_model_writer_forwards_the_exact_integer_setting(exact: bool, expected: int, container: type) -> None:
    model = _add_model()
    for name in ("x", "one", "middle", "y"):
        model.set_tensor_datatype(name, DataType["INT32"])
    recorded = execute_onnx_and_make_model(model, {"x": np.full(2, 16777216, dtype=np.float32)}, exact_integers=exact)
    result = recorded.get_initializer("y")
    assert result.dtype == container
    assert result.tolist() == [expected, expected]
    assert "" not in {tensor.name for tensor in recorded.graph.initializer}


def test_single_identity_records_values_without_optional_model_features() -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "identity",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    values = np.asarray([3], dtype=np.float32)
    recorded = execute_onnx_and_make_model(model, {"x": values})
    assert {tensor.name for tensor in recorded.graph.initializer} == {"x", "y"}
    np.testing.assert_array_equal(recorded.get_initializer("x"), values)
    np.testing.assert_array_equal(recorded.get_initializer("y"), values)
