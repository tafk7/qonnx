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
from onnx import TensorProto, checker, helper

import qonnx.core.data_layout as DataLayout
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import qonnx_make_model


def _dropout() -> ModelWrapper:
    graph = helper.make_graph(
        [helper.make_node("Dropout", ["x"], ["y", ""], name="dropout")],
        "optional",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
    )
    return ModelWrapper(qonnx_make_model(graph))


def test_layout_inference_omits_only_the_absent_output_annotation() -> None:
    model = _dropout()
    model.set_tensor_layout("x", DataLayout.NC)
    expected = copy.deepcopy(model)
    expected.set_tensor_layout("y", DataLayout.UNKNOWN)
    result = model.transform(InferDataLayouts())
    assert result.get_tensor_layout("y") == DataLayout.UNKNOWN
    assert result.get_tensor_layout("") is None
    assert result.model.SerializeToString(deterministic=True) == expected.model.SerializeToString(deterministic=True)
    checker.check_model(result.model)


@pytest.mark.parametrize("exact", [False, True])
def test_datatype_inference_omits_absent_output_in_both_settings(exact: bool) -> None:
    result = _dropout().transform(InferDataTypes(exact_ranges=exact))
    assert result.get_tensor_datatype("y") == DataType["FLOAT32"]
    assert all(annotation.tensor_name for annotation in result.graph.quantization_annotation)
    assert list(result.graph.node[0].output) == ["y", ""]
    checker.check_model(result.model)
