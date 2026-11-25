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

"""Lookup failures stay unsupported; hooks' own errors reach their caller."""

from __future__ import annotations

import pytest

import numpy as np
import uuid
from onnx import GraphProto, NodeProto, TensorProto, helper
from typing import Any

from qonnx.core.execute_custom_node import execute_custom_node
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.base import CustomOp
from qonnx.custom_op.registry import CustomOpRefused, CustomOpUnsupported, add_op_to_domain
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model


class HookFault(CustomOp):
    """An extension that raises its own KeyError in one of its hooks."""

    hook: str = ""

    def get_nodeattr_types(self) -> dict[str, tuple]:
        return {}

    def make_shape_compatible_op(self, model: ModelWrapper) -> NodeProto:
        raise KeyError("hook's missing shape key")

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        raise KeyError("hook's missing datatype key")

    def execute_node(self, context: dict[str, Any], graph: GraphProto) -> None:
        raise KeyError("hook's missing execution key")

    def verify_node(self) -> None:
        pass


def make_model(domain: str, op_type: str) -> ModelWrapper:
    graph = helper.make_graph(
        [helper.make_node(op_type, ["x"], ["y"], domain=domain)],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    m = qonnx_make_model(graph)
    m.opset_import.append(helper.make_opsetid(domain, 1))
    return ModelWrapper(m)


def invoke(m: ModelWrapper, entry: str) -> None:
    if entry == "execute_custom_node":
        execute_custom_node(m.graph.node[0], {"x": np.zeros(1, dtype=np.float32)}, m.graph, 13, model=m)
    elif entry == "InferShapes":
        m.transform(InferShapes())
    else:
        m.transform(InferDataTypes(exact_ranges=entry == "InferDataTypes.exact"))


@pytest.mark.parametrize("entry", ["execute_custom_node", "InferShapes", "InferDataTypes", "InferDataTypes.exact"])
def test_a_missing_op_stays_an_exception_without_a_key_error_base(entry: str) -> None:
    m = make_model("qonnx.custom_op.general", "NoSuchCustomOp")
    with pytest.raises(CustomOpUnsupported) as caught:
        invoke(m, entry)
    assert isinstance(caught.value, Exception) and not isinstance(caught.value, KeyError)
    assert caught.value.findings[0].code == "op-unsupported"
    assert isinstance(caught.value.__cause__, CustomOpRefused)
    assert caught.value.__cause__.findings[0].code == "op-unknown"


@pytest.mark.parametrize("entry", ["execute_custom_node", "InferShapes", "InferDataTypes", "InferDataTypes.exact"])
def test_an_implemented_ops_own_key_error_is_not_called_unsupported(entry: str) -> None:
    domain = "qonnx_test_hook_fault_" + uuid.uuid4().hex
    add_op_to_domain(domain, HookFault)
    m = make_model(domain, "HookFault")
    with pytest.raises(KeyError, match="hook's missing") as caught:
        invoke(m, entry)
    assert type(caught.value) is KeyError
