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

"""CustomOp.infer_node_facts: each output's shape, from ONNX's inference of the op's
shape stand-in, then its datatype and range, from the op's datatype rule inside the
exact-range signal; in that order, for a pass that visits nodes in graph order."""

import pytest

import copy
import numpy as np
from onnx import TensorProto, helper

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper, TensorFactRefused
from qonnx.core.ranges import Range
from qonnx.custom_op.base import CustomOp
from qonnx.custom_op.registry import add_op_to_domain, getCustomOp
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

DOMAIN = "qonnx.custom_op.general"


class CopyOp(CustomOp):
    """A custom op as a third party writes one, knowing nothing of infer_node_facts:
    it copies its input to its output and states its output's datatype as its
    input's."""

    def get_nodeattr_types(self):
        return {}

    def make_shape_compatible_op(self, model):
        return helper.make_node("Identity", [self.onnx_node.input[0]], [self.onnx_node.output[0]])

    def infer_node_datatype(self, model):
        node = self.onnx_node
        model.set_tensor_datatype(node.output[0], model.get_tensor_datatype(node.input[0]))

    def execute_node(self, context, graph):
        context[self.onnx_node.output[0]] = context[self.onnx_node.input[0]]

    def verify_node(self):
        pass


def model_of(nodes, inputs, initializers=(), value_info=(), outputs=None):
    """A model of the given nodes, inputs as (name, element type, shape)."""
    graph = helper.make_graph(
        list(nodes),
        "g",
        [helper.make_tensor_value_info(name, elem, shape) for name, elem, shape in inputs],
        outputs or [helper.make_tensor_value_info(nodes[-1].output[0], TensorProto.FLOAT, None)],
        initializer=list(initializers),
        value_info=list(value_info),
    )
    return ModelWrapper(qonnx_make_model(graph), fix_missing_initializer_valueinfo=False)


def facts_of(model, name):
    info = model.get_tensor_valueinfo(name)
    return (
        model.get_tensor_shape(name),
        info.type.tensor_type.elem_type if info is not None else None,
        model.get_tensor_datatype(name),
        model.get_stated_tensor_range(name),
    )


def op_class(name, base=CopyOp, **body):
    return type(name, (base,), dict(body))


def registered(name, base=CopyOp, **body):
    domain = "qonnx_test_facts_" + name
    cls = op_class(name, base, **body)
    add_op_to_domain(domain, cls)
    return domain, cls


def test_a_third_party_op_gets_its_shape_and_datatype_from_its_own_hooks():
    domain, _ = registered("Copy")
    model = model_of([helper.make_node("Copy", ["x"], ["y"], domain=domain)], [("x", TensorProto.FLOAT, [2, 3])])
    model.set_tensor_datatype("x", DataType["INT4"])
    getCustomOp(model.graph.node[0]).infer_node_facts(model)
    assert facts_of(model, "y") == ([2, 3], TensorProto.FLOAT, DataType["INT4"], None)


def multithreshold_model(out_dtype="INT4", out_bias=-1.0):
    thresholds = np.array([[0.5, 1.5, 2.5]] * 4, dtype=np.float32)
    node = helper.make_node(
        "MultiThreshold", ["x", "t"], ["y"], domain=DOMAIN, out_dtype=out_dtype, out_bias=out_bias, out_scale=1.0
    )
    init = helper.make_tensor("t", TensorProto.FLOAT, [4, 3], thresholds.flatten())
    return model_of([node], [("x", TensorProto.FLOAT, [1, 4])], [init])


def test_multithreshold_states_its_range_through_infer_node_facts_only():
    model = multithreshold_model()
    getCustomOp(model.graph.node[0]).infer_node_facts(model)
    assert facts_of(model, "y") == ([1, 4], TensorProto.FLOAT, DataType["INT4"], Range(-1, 2))
    model = multithreshold_model().transform(InferShapes()).transform(InferDataTypes())
    assert model.get_tensor_datatype("y") == DataType["INT4"]
    assert model.get_stated_tensor_range("y") is None


def test_an_op_overriding_it_is_called_instead():
    calls = []

    def infer_node_facts(self, model):
        calls.append(self.onnx_node.name)
        model.set_tensor_shape(self.onnx_node.output[0], [7])

    domain, _ = registered("Whole", infer_node_facts=infer_node_facts)
    model = model_of([helper.make_node("Whole", ["x"], ["y"], domain=domain, name="w")], [("x", TensorProto.FLOAT, [2])])
    getCustomOp(model.graph.node[0]).infer_node_facts(model)
    assert calls == ["w"]
    assert model.get_tensor_shape("y") == [7]


@pytest.mark.parametrize("info", [None, helper.make_tensor_value_info("h", TensorProto.FLOAT, None)])
def test_an_input_without_a_shape_is_refused_and_nothing_is_written(info):
    domain, _ = registered("Copy")
    nodes = [helper.make_node("Copy", ["h"], ["y"], domain=domain, name="copy")]
    model = model_of(nodes, [("x", TensorProto.FLOAT, [2])], value_info=[info] if info is not None else [])
    model.set_tensor_datatype("y", DataType["INT2"])
    before = copy.deepcopy(model.model)
    with pytest.raises(TensorFactRefused) as caught:
        getCustomOp(model.graph.node[0]).infer_node_facts(model)
    assert isinstance(caught.value, ValueError)
    finding = caught.value.findings[0]
    assert finding.code == "fact-invalid" and finding.owner == "CustomOp.infer_node_facts"
    assert finding.message == "copy: h has no shape yet"
    assert model.model == before


def test_a_refusal_names_the_op_when_its_node_has_no_name():
    domain, _ = registered("Copy")
    model = model_of([helper.make_node("Copy", ["h"], ["y"], domain=domain)], [("x", TensorProto.FLOAT, [2])])
    with pytest.raises(TensorFactRefused) as caught:
        getCustomOp(model.graph.node[0]).infer_node_facts(model)
    assert isinstance(caught.value, ValueError)
    assert caught.value.findings[0].message == "Copy: h has no shape yet"


def test_a_stand_in_that_is_not_a_standard_op_leaves_the_shapes():
    def make_shape_compatible_op(self, model):
        return self.onnx_node

    domain, _ = registered("Opaque", make_shape_compatible_op=make_shape_compatible_op)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [5])
    model = model_of([helper.make_node("Opaque", ["x"], ["y"], domain=domain)], [("x", TensorProto.FLOAT, [2])], outputs=[y])
    model.set_tensor_datatype("x", DataType["UINT3"])
    getCustomOp(model.graph.node[0]).infer_node_facts(model)
    assert facts_of(model, "y") == ([5], TensorProto.FLOAT, DataType["UINT3"], None)


def test_an_output_ONNX_cannot_infer_fully_keeps_its_shape():
    def make_shape_compatible_op(self, model):
        return helper.make_node("Reshape", [self.onnx_node.input[0], "target"], [self.onnx_node.output[0]])

    domain, _ = registered("Dynamic", make_shape_compatible_op=make_shape_compatible_op)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2])
    nodes = [helper.make_node("Dynamic", ["x", "target"], ["y"], domain=domain)]
    inputs = [("x", TensorProto.FLOAT, [2, 3]), ("target", TensorProto.INT64, [2])]
    model = model_of(nodes, inputs, outputs=[y])
    getCustomOp(model.graph.node[0]).infer_node_facts(model)
    assert model.get_tensor_shape("y") == [3, 2]


def test_an_initializer_input_is_given_with_its_value():
    def make_shape_compatible_op(self, model):
        return helper.make_node("Reshape", [self.onnx_node.input[0], "target"], [self.onnx_node.output[0]])

    domain, _ = registered("Reshaping", make_shape_compatible_op=make_shape_compatible_op)
    target = helper.make_tensor("target", TensorProto.INT64, [2], [3, 2])
    nodes = [helper.make_node("Reshaping", ["x", "target"], ["y"], domain=domain)]
    model = model_of(nodes, [("x", TensorProto.FLOAT, [2, 3])], [target])
    getCustomOp(model.graph.node[0]).infer_node_facts(model)
    assert model.get_tensor_shape("y") == [3, 2]


def test_a_stand_in_in_the_ai_onnx_domain_is_a_standard_op():
    def make_shape_compatible_op(self, model):
        return helper.make_node("Transpose", [self.onnx_node.input[0]], [self.onnx_node.output[0]], domain="ai.onnx")

    domain, _ = registered("Transposing", make_shape_compatible_op=make_shape_compatible_op)
    model = model_of([helper.make_node("Transposing", ["x"], ["y"], domain=domain)], [("x", TensorProto.FLOAT, [2, 3])])
    getCustomOp(model.graph.node[0]).infer_node_facts(model)
    assert model.get_tensor_shape("y") == [3, 2]


def test_the_element_type_ONNX_infers_is_stated():
    def make_shape_compatible_op(self, model):
        return helper.make_node("Cast", [self.onnx_node.input[0]], [self.onnx_node.output[0]], to=TensorProto.INT64)

    domain, _ = registered("Casting", make_shape_compatible_op=make_shape_compatible_op)
    model = model_of([helper.make_node("Casting", ["x"], ["y"], domain=domain)], [("x", TensorProto.FLOAT, [4])])
    getCustomOp(model.graph.node[0]).infer_node_facts(model)
    info = model.get_tensor_valueinfo("y")
    assert info.type.tensor_type.elem_type == TensorProto.INT64
    assert model.get_tensor_shape("y") == [4]


def test_the_shape_is_stated_before_the_datatype_rule_runs():
    seen = []

    def infer_node_datatype(self, model):
        seen.append(model.get_tensor_shape(self.onnx_node.output[0]))
        model.set_tensor_datatype(self.onnx_node.output[0], DataType["INT8"])

    domain, _ = registered("Ordered", infer_node_datatype=infer_node_datatype)
    model = model_of([helper.make_node("Ordered", ["x"], ["y"], domain=domain)], [("x", TensorProto.FLOAT, [3])])
    getCustomOp(model.graph.node[0]).infer_node_facts(model)
    assert seen == [[3]]
    assert model.get_tensor_datatype("y") == DataType["INT8"]


def qonnx_op_models():
    """Single-node models of qonnx's own ops."""
    x = ("x", TensorProto.FLOAT, [1, 4, 4, 2])
    yield multithreshold_model()
    yield model_of(
        [
            helper.make_node(
                "MaxPoolNHWC", ["x"], ["y"], domain=DOMAIN, kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0]
            )
        ],
        [x],
    )
    yield model_of(
        [
            helper.make_node(
                "Im2Col",
                ["x"],
                ["y"],
                domain=DOMAIN,
                stride=[1, 1],
                kernel_size=[2, 2],
                input_shape="(1,4,4,2)",
                dilations=[1, 1],
            )
        ],
        [x],
    )
    scalars = [
        helper.make_tensor("scale", TensorProto.FLOAT, [], [0.5]),
        helper.make_tensor("zeropt", TensorProto.FLOAT, [], [0.0]),
        helper.make_tensor("bits", TensorProto.FLOAT, [], [4.0]),
    ]
    yield model_of(
        [helper.make_node("Quant", ["x", "scale", "zeropt", "bits"], ["y"], domain=DOMAIN, signed=1, narrow=0)],
        [x],
        scalars,
    )
    yield model_of(
        [helper.make_node("BipolarQuant", ["x", "scale"], ["y"], domain=DOMAIN)],
        [x],
        scalars[:1],
    )


@pytest.mark.parametrize("model", list(qonnx_op_models()), ids=lambda model: model.graph.node[0].op_type)
def test_qonnx_ops_state_what_the_passes_state(model):
    """The passes with exact ranges, scaled integers kept (InferDataTypes' replacement
    of them by FLOAT32 is the pass's, not the op's)."""
    model.set_tensor_datatype("x", DataType["INT4"])
    by_passes = copy.deepcopy(model).transform(InferShapes())
    by_passes = by_passes.transform(InferDataTypes(allow_scaledint_dtypes=True, exact_ranges=True))
    getCustomOp(model.graph.node[0]).infer_node_facts(model)
    assert facts_of(model, "y") == facts_of(by_passes, "y")
