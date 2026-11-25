# Copyright (c) 2020 Xilinx, Inc.
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
# * Neither the name of Xilinx nor the names of its
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

"""Worker failures retain their stack when transformation findings are collected."""

import pytest

import traceback
from contextlib import nullcontext
from onnx import NodeProto, TensorProto, helper

from qonnx.core.datatype import NumericValueRefused
from qonnx.core.findings import Finding, collect_findings, report_finding
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import NodeLocalTransformation


def _fail_in_helper(node: NodeProto, refusal: bool) -> tuple[NodeProto, bool]:
    if refusal:
        raise NumericValueRefused.of("undefined", "node-operation", "node refused", node=node.name)
    raise RuntimeError("worker failed in helper", 17)


class _FailingNodes(NodeLocalTransformation):
    def __init__(self, workers: int, refusal: bool) -> None:
        super().__init__(workers)
        self.refusal = refusal

    def applyNodeLocal(self, node: NodeProto) -> tuple[NodeProto, bool]:
        report_finding(Finding("visited", "node-operation", node.name))
        if node.name == "node1":
            return _fail_in_helper(node, self.refusal)
        return node, False


def _model() -> ModelWrapper:
    inputs = [helper.make_tensor_value_info("i", TensorProto.FLOAT, [1])]
    outputs = [helper.make_tensor_value_info("o", TensorProto.FLOAT, [1])]
    nodes = [
        helper.make_node("Relu", ["i"], ["a"], name="node0"),
        helper.make_node("Relu", ["a"], ["b"], name="node1"),
        helper.make_node("Relu", ["b"], ["o"], name="node2"),
    ]
    return ModelWrapper(helper.make_model(helper.make_graph(nodes, "graph", inputs, outputs)))


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("collecting", [False, True])
@pytest.mark.parametrize("refusal", [False, True])
def test_node_local_failure_retains_worker_frames_and_exception(workers: int, collecting: bool, refusal: bool) -> None:
    expected_type = NumericValueRefused if refusal else RuntimeError
    with collect_findings() if collecting else nullcontext() as found:
        with pytest.raises(expected_type) as caught:
            _model().transform(_FailingNodes(workers, refusal), cleanup=False)
    error = caught.value
    formatted = "".join(traceback.format_exception(error))
    assert "in _fail_in_helper" in formatted
    assert "in applyNodeLocal" in formatted
    assert type(error) is expected_type
    if refusal:
        expected = NumericValueRefused.of("undefined", "node-operation", "node refused", node="node1")
        assert error.args == expected.args
        assert error.findings == expected.findings
    else:
        assert error.args == ("worker failed in helper", 17)
    if workers == 1:
        assert error.__cause__ is None
    else:
        assert error.__cause__ is not None
        assert "in _fail_in_helper" in str(error.__cause__)
        assert "in applyNodeLocal" in str(error.__cause__)
    if collecting:
        assert [finding.message for finding in found] == ["node2", "node1"]
