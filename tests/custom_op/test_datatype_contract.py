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

"""The custom-op datatype contract (CustomOp.infer_node_datatype): an op writes
each output's datatype and range together, with model.set_tensor_datatype(tensor,
datatype, range); a datatype set alone clears the range the tensor stated
before."""

import pytest

from onnx import TensorProto, helper

from qonnx.core import tensor_ranges
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.ranges import TensorRange
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import qonnx_make_model


def states_its_range(self, model):
    model.set_tensor_datatype(self.onnx_node.output[0], DataType["INT4"], TensorRange(0, 3))


def states_a_datatype_alone(self, model):
    model.set_tensor_datatype(self.onnx_node.output[0], DataType["INT4"])


def model_of(domain, op_type):
    """x -> op -> y, y stating a stale fact: INT8, [-100, 100]."""
    graph = helper.make_graph(
        [helper.make_node(op_type, ["x"], ["y"], domain=domain)],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])],
    )
    model = ModelWrapper(
        qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(domain, 1)])
    )
    model.set_tensor_datatype("x", DataType["INT4"])
    model.set_tensor_datatype("y", DataType["INT8"], TensorRange(-100, 100))
    return model


@pytest.mark.parametrize(
    "body, stated, expected",
    [
        (states_its_range, TensorRange(0, 3), TensorRange(0, 3)),
        # INT4's full range: the stale [-100, 100] is not kept
        (states_a_datatype_alone, None, TensorRange(-8, 7)),
    ],
)
def test_a_custom_op_writes_each_output_s_datatype_and_range_together(make_op, make_domain, body, stated, expected):
    op = make_op("RangedOp", infer_node_datatype=body)
    model = model_of(make_domain(RangedOp=op), "RangedOp").transform(InferDataTypes())
    assert model.get_tensor_datatype("y") == DataType["INT4"]
    assert tensor_ranges.stated(model, "y") == stated
    assert model.get_tensor_range("y") == expected
