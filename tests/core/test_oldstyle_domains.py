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

"""qonnx's own old-style domain names, renamed when a model is loaded by the table
qonnx.custom_op.registry.RENAMED: each row's new domain and warnings."""

import pytest

import warnings
from onnx import helper

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import RENAMED
from qonnx.util.basic import qonnx_make_model

AGGREGATE = "Some old-style domain attributes were automatically converted to new-style"


def load(nodes, subgraph_nodes=()):
    """Load a model of the given top-level nodes (each reading x, writing its own
    output), the warnings it raised listed as (category, message)."""
    if subgraph_nodes:
        body = helper.make_graph(list(subgraph_nodes), "body", [], [helper.make_tensor_value_info("b", 1, [1])])
        nodes = [*nodes, helper.make_node("If", ["c"], ["z"], then_branch=body, else_branch=body)]
    graph = helper.make_graph(
        list(nodes),
        "g",
        [helper.make_tensor_value_info("x", 1, [1])],
        [helper.make_tensor_value_info(n.output[0], 1, [1]) for n in nodes],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = ModelWrapper(qonnx_make_model(graph))
    return model, [(w.category, str(w.message)) for w in caught]


def oldstyle(domain, output="y", **attributes):
    return helper.make_node("MultiThreshold", ["x"], [output], domain=domain, **attributes)


@pytest.mark.parametrize(
    "domain, attributes, renamed",
    [
        ("finn", {"backend": "fpgadataflow"}, "finn.custom_op.fpgadataflow"),
        ("finn", {}, "qonnx.custom_op.general"),
        ("finn.custom_op.general", {}, "qonnx.custom_op.general"),
    ],
)
def test_an_old_style_domain_is_renamed_with_one_warning(domain, attributes, renamed):
    model, caught = load([oldstyle(domain, **attributes)])
    assert model.graph.node[0].domain == renamed
    assert len(caught) == 1
    assert caught[0][0] is UserWarning and caught[0][1].startswith(AGGREGATE)


def test_a_finn_node_with_another_backend_keeps_its_domain_with_both_warnings():
    model, caught = load([oldstyle("finn", backend="other")])
    assert model.graph.node[0].domain == "finn"
    assert [category for category, _ in caught] == [UserWarning, UserWarning]
    assert caught[0][1].startswith("Can't fix domain for node ")
    assert caught[1][1].startswith(AGGREGATE)


def test_several_old_style_nodes_give_one_aggregate_warning():
    model, caught = load([oldstyle("finn", "y1"), oldstyle("finn.custom_op.general", "y2")])
    assert [n.domain for n in model.graph.node] == ["qonnx.custom_op.general"] * 2
    assert len(caught) == 1


def test_a_model_without_old_style_nodes_gives_no_warning():
    model, caught = load([oldstyle("qonnx.custom_op.general")])
    assert model.graph.node[0].domain == "qonnx.custom_op.general"
    assert caught == []


def test_only_top_level_nodes_are_renamed():
    model, caught = load([], subgraph_nodes=[oldstyle("finn", "b")])
    assert model.graph.node[0].attribute[0].g.node[0].domain == "finn"
    assert caught == []


def test_the_table_holds_qonnx_s_old_style_names():
    assert sorted(RENAMED) == ["finn", "finn.custom_op.general"]
    with pytest.raises(TypeError):
        RENAMED["other"] = lambda node: None
