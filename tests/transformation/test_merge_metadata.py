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
from onnx import TensorProto, helper

from qonnx.core.metadata import MetadataRefused, Namespace
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.util.basic import qonnx_make_model


def make_relu_model(name: str) -> ModelWrapper:
    inp = helper.make_tensor_value_info(f"{name}_in", TensorProto.FLOAT, [1, 4])
    outp = helper.make_tensor_value_info(f"{name}_out", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph([helper.make_node("Relu", [f"{name}_in"], [f"{name}_out"])], name, [inp], [outp])
    return ModelWrapper(qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 13)]))


BOARD = Namespace("test.board", version=1)
PERIOD = BOARD.key("period_ns", float)
PART = BOARD.key("part", str)


def test_merge_onnx_models_keeps_both_graphs_metadata() -> None:
    pre, post = make_relu_model("pre"), make_relu_model("post")
    pre.set_metadata(PART, "xc7z020")
    pre.set_metadata_prop("only_pre", "a")
    pre.set_metadata_prop("both", "pre")
    post.set_metadata(PERIOD, 5.0)
    post.set_metadata(PART, "xc7z020")
    post.set_metadata_prop("both", "post")
    merged = post.transform(MergeONNXModels(pre))
    assert merged.metadata_namespace(BOARD) == {"period_ns": 5.0, "part": "xc7z020"}
    assert merged.get_metadata_prop("only_pre") == "a"
    assert merged.get_metadata_prop("both") == "post"  # the main model's untyped key wins
    # a pre model without metadata leaves the main model's
    assert post.transform(MergeONNXModels(make_relu_model("bare"))).get_metadata(PERIOD) == 5.0


def test_merge_onnx_models_refuses_a_typed_key_stated_differently() -> None:
    pre, post = make_relu_model("pre"), make_relu_model("post")
    pre.set_metadata(PERIOD, 4.0)
    post.set_metadata(PERIOD, 5.0)
    with pytest.raises(MetadataRefused, match="test.board/period_ns: the graphs merged state '5.0' and '4.0'"):
        post.transform(MergeONNXModels(pre))
