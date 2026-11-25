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
import onnx
from typing import Callable

import qonnx.core.onnx_exec as oxe
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.util.basic import qonnx_make_model
from qonnx.util.cleanup import cleanup_model


def test_modelwrapper_rename_tensor_updates_all_node_inputs() -> None:
    shared = onnx.helper.make_tensor_value_info("shared", onnx.TensorProto.FLOAT, [1, 4])
    sum_output = onnx.helper.make_tensor_value_info("sum_output", onnx.TensorProto.FLOAT, [1, 4])
    product_output = onnx.helper.make_tensor_value_info("product_output", onnx.TensorProto.FLOAT, [1, 4])
    add_node = onnx.helper.make_node("Add", inputs=["shared", "shared"], outputs=["sum_output"])
    mul_node = onnx.helper.make_node("Mul", inputs=["shared", "sum_output"], outputs=["product_output"])
    graph = onnx.helper.make_graph(
        nodes=[add_node, mul_node],
        name="shared_input_graph",
        inputs=[shared],
        outputs=[product_output],
        value_info=[sum_output],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="rename-test"))

    model.rename_tensor("shared", "renamed_shared")

    assert model.graph.input[0].name == "renamed_shared"
    assert list(model.graph.node[0].input) == ["renamed_shared", "renamed_shared"]
    assert list(model.graph.node[1].input) == ["renamed_shared", "sum_output"]


def make_shared_input_model() -> ModelWrapper:
    graph_input = onnx.helper.make_tensor_value_info("B_source", onnx.TensorProto.FLOAT, [4])
    graph_output = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [4])
    shared_value = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.FLOAT, [4])
    source = onnx.helper.make_node("Identity", inputs=["B_source"], outputs=["B"], name="Source")
    add = onnx.helper.make_node("Add", inputs=["B", "B"], outputs=["Y"], name="AddSharedTwice")
    graph = onnx.helper.make_graph(
        nodes=[source, add],
        name="shared-input-model",
        inputs=[graph_input],
        outputs=[graph_output],
        value_info=[shared_value],
    )
    return ModelWrapper(qonnx_make_model(graph, producer_name="qonnx-tests"))


@pytest.mark.parametrize(
    "rename",
    [
        pytest.param(
            lambda model: model.transform(GiveReadableTensorNames()),
            id="readable-tensor-names",
        ),
        pytest.param(cleanup_model, id="cleanup-model"),
    ],
)
def test_renaming_preserves_repeated_input_graph(rename: Callable[[ModelWrapper], ModelWrapper]) -> None:
    values = np.asarray([3, 1, 7, 2], dtype=np.float32)
    model = make_shared_input_model().transform(GiveUniqueNodeNames())
    expected = next(iter(oxe.execute_onnx(model, {"B_source": values}).values()))

    renamed = rename(model)

    onnx.checker.check_model(renamed.model)
    renamed_input = renamed.graph.input[0].name
    actual = next(iter(oxe.execute_onnx(renamed, {renamed_input: values}).values()))
    assert np.array_equal(actual, expected)
    add = renamed.get_nodes_by_op_type("Add")[0]
    assert len(add.input) == 2
    assert add.input[0] == add.input[1]


def test_rename_updates_repeated_node_outputs_and_annotations() -> None:
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Custom", ["x", "x"], ["x", "x"])],
        "repeated-outputs",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
        [],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_layout("x", ["N"])
    model.rename_tensor("x", "renamed")
    assert list(model.graph.node[0].input) == ["renamed", "renamed"]
    assert list(model.graph.node[0].output) == ["renamed", "renamed"]
    assert model.get_tensor_layout("renamed") == ["N"]
    assert model.get_tensor_layout("x") is None
