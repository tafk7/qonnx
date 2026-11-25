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

"""Exact execution keeps its arithmetic, unsupported and value error handlers."""

from __future__ import annotations

import pytest

import numpy as np
from onnx import TensorProto, helper

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import (
    ExecutionArithmeticRefused,
    ExecutionUnsupported,
    ExecutionValueRefused,
    execute_node,
    execute_onnx,
)
from qonnx.util.basic import qonnx_make_model


def standard(op_type: str) -> ModelWrapper:
    node = helper.make_node(op_type, ["x"], ["y"], name="exact-node")
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.DOUBLE, [1])],
        [helper.make_tensor_value_info("y", TensorProto.DOUBLE, [1])],
    )
    m = ModelWrapper(qonnx_make_model(graph))
    for name in ("x", "y"):
        m.set_tensor_datatype(name, DataType["INT64"])
    return m


@pytest.mark.parametrize(
    "op_type,leaf,legacy,code",
    [
        ("Abs", ExecutionArithmeticRefused, ArithmeticError, "exact-beyond-float64"),
        ("Sin", ExecutionUnsupported, NotImplementedError, "exact-unsupported"),
    ],
)
def test_exact_execution_retains_each_legacy_base(op_type: str, leaf: type, legacy: type, code: str) -> None:
    with pytest.raises(leaf) as caught:
        execute_onnx(standard(op_type), {"x": np.array([2**54], dtype=np.float64)}, exact_integers=True)
    assert isinstance(caught.value, legacy)
    assert caught.value.findings[0].code == code


def test_exact_execution_without_the_model_retains_value_error() -> None:
    m = standard("Abs")
    with pytest.raises(ExecutionValueRefused) as caught:
        execute_node(m.graph.node[0], {"x": np.ones(1), "y": np.zeros(1)}, m.graph, 13, exact_integers=True)
    assert isinstance(caught.value, ValueError)
    assert caught.value.findings[0].code == "exact-needs-model"


@pytest.mark.parametrize(
    "out_dtype,scale,bias,code",
    [
        ("INT2", 1.0, 0.0, "threshold-output-range"),
        ("INT64", 1.0, float(2**24), "threshold-output-container"),
    ],
)
def test_exact_multithreshold_output_refusals_keep_value_error(out_dtype: str, scale: float, bias: float, code: str) -> None:
    node = helper.make_node(
        "MultiThreshold",
        ["x", "t"],
        ["y"],
        domain="qonnx.custom_op.general",
        name="threshold",
        out_dtype=out_dtype,
        out_scale=scale,
        out_bias=bias,
    )
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])],
        initializer=[helper.make_tensor("t", TensorProto.FLOAT, [1, 2], [0, 1])],
    )
    proto = qonnx_make_model(graph)
    proto.opset_import.append(helper.make_opsetid("qonnx.custom_op.general", 1))
    m = ModelWrapper(proto)
    with pytest.raises(ExecutionValueRefused) as caught:
        execute_onnx(m, {"x": np.ones((1, 1), dtype=np.float32)}, exact_integers=True)
    assert isinstance(caught.value, ValueError)
    assert caught.value.findings[0].code == code
