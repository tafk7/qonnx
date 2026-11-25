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

"""Tensor and metadata refusals retain their family through public model calls."""

from __future__ import annotations

import pytest

from collections.abc import Callable
from onnx import GraphProto, NodeProto, TensorProto, helper

from qonnx.core import tensor_ranges
from qonnx.core.datatype import DataType, DataTypeNameUnknown
from qonnx.core.metadata import MetadataRefused, Namespace
from qonnx.core.modelwrapper import ModelWrapper, TensorFactRefused
from qonnx.core.onnx_exec import ExecutionValueRefused, reuse_sessions
from qonnx.core.ranges import Range
from qonnx.custom_op.base import CustomOp
from qonnx.custom_op.registry import CustomOpRefused, get_domain_opset_version
from qonnx.util.basic import qonnx_make_model


def model(shape: list[int] | None = None) -> ModelWrapper:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
    )
    return ModelWrapper(qonnx_make_model(graph))


@pytest.mark.parametrize(
    "call,leaf,code,owner",
    [
        (
            lambda m: m.set_tensor_datatype("x", None, range=Range(0, 1)),
            TensorFactRefused,
            "fact-invalid",
            "ModelWrapper.set_tensor_datatype",
        ),
        (lambda m: tensor_ranges.write(m, "x", Range(0, 1)), TensorFactRefused, "fact-invalid", "tensor_ranges.write"),
        (
            lambda m: m.inherit_metadata(Namespace("test.inherited", inherit=True)),
            MetadataRefused,
            "metadata-invalid",
            "ModelWrapper.inherit_metadata",
        ),
    ],
)
def test_refused_model_calls_keep_value_error(
    call: Callable[[ModelWrapper], object], leaf: type, code: str, owner: str
) -> None:
    m = model([1])
    before = m.model.SerializeToString()
    with pytest.raises(leaf) as caught:
        call(m)
    assert isinstance(caught.value, ValueError)
    assert type(caught.value) is leaf
    assert caught.value.findings[0].code == code and caught.value.findings[0].owner == owner
    assert m.model.SerializeToString() == before


@pytest.mark.parametrize("operation", ["facts", "state"])
def test_unknown_stored_datatype_keeps_its_key_error_through_facts_and_state(operation: str) -> None:
    m = model([1])
    annotation = m.graph.quantization_annotation.add(tensor_name="x")
    annotation.quant_parameter_tensor_names.add(key="finn_datatype", value="FOO")
    call = (lambda: m.facts("x")) if operation == "facts" else (lambda: m.state("x", range=Range(0, 1)))
    with pytest.raises(DataTypeNameUnknown) as caught:
        call()
    assert isinstance(caught.value, KeyError) and not isinstance(caught.value, ValueError)
    assert caught.value.findings[0].code == "name-unknown"
    assert dict(caught.value.findings[0].details)["tensor"] == "x"


@pytest.mark.parametrize("operation", ["facts", "state"])
def test_malformed_range_metadata_keeps_its_family_through_facts_and_state(operation: str) -> None:
    m = model([1])
    m.set_tensor_datatype("x", DataType["INT8"])
    annotation = m.graph.quantization_annotation[0]
    annotation.quant_parameter_tensor_names.add(key="qonnx.tensor/@version", value="1")
    annotation.quant_parameter_tensor_names.add(key="qonnx.tensor/range", value="not-json")
    call = (lambda: m.facts("x")) if operation == "facts" else (lambda: m.state("x", element=DataType["INT8"]))
    with pytest.raises(MetadataRefused) as caught:
        call()
    assert isinstance(caught.value, ValueError) and not isinstance(caught.value, TensorFactRefused)
    assert caught.value.findings[0].code == "metadata-invalid"
    assert dict(caught.value.findings[0].details)["tensor"] == "x"


class CopyFacts(CustomOp):
    """An op that uses the default all-facts implementation."""

    def get_nodeattr_types(self) -> dict:
        return {}

    def make_shape_compatible_op(self, model: ModelWrapper) -> NodeProto:
        return helper.make_node("Identity", list(self.onnx_node.input), list(self.onnx_node.output))

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        model.set_tensor_datatype(self.onnx_node.output[0], DataType["INT8"])

    def execute_node(self, context: dict, graph: GraphProto) -> None:
        context[self.onnx_node.output[0]] = context[self.onnx_node.input[0]]

    def verify_node(self) -> None:
        pass


def test_all_facts_missing_shape_names_the_public_custom_op_method() -> None:
    m = model()
    op = CopyFacts(m.graph.node[0])
    with pytest.raises(TensorFactRefused) as caught:
        op.infer_node_facts(m)
    assert isinstance(caught.value, ValueError)
    found = caught.value.findings[0]
    assert found.code == "fact-invalid" and found.owner == "CustomOp.infer_node_facts"
    assert dict(found.details)["fact"] == "shape"


def test_a_thread_count_outside_the_supported_range_keeps_value_error() -> None:
    with pytest.raises(ExecutionValueRefused) as caught:
        with reuse_sessions(0):
            pass
    assert isinstance(caught.value, ValueError)
    assert caught.value.findings[0].code == "setting-invalid"
    assert caught.value.findings[0].owner == "reuse_sessions"


def test_a_missing_domain_is_a_lookup_refusal_about_the_domain() -> None:
    domain = "qonnx_test_domain_that_is_absent"
    with pytest.raises(CustomOpRefused) as caught:
        get_domain_opset_version(domain)
    assert isinstance(caught.value, KeyError) and not isinstance(caught.value, ValueError)
    assert caught.value.findings[0].code == "domain-unknown"
    assert dict(caught.value.findings[0].details)["domain"] == domain
