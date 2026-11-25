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

"""The graph context of a custom op: an op that sets wants_model is given its model
wherever an op is built for a model, and every such place builds the op at the version
the model imports for the node's domain."""

import pytest

import json
import numpy as np
import os
import tempfile
import uuid
import warnings
from onnx import TensorProto, helper

import qonnx.core.data_layout as DataLayout
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.custom_op.base import CustomOp
from qonnx.custom_op.general.im2col import Im2Col
from qonnx.custom_op.general.multithreshold import MultiThreshold
from qonnx.custom_op.registry import add_op_to_domain, get_ops_in_domain, getCustomOp
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.rebalance_conv import RebalanceIm2Col
from qonnx.util.basic import qonnx_make_model
from qonnx.util.config import extract_model_config_to_json
from qonnx.util.range_analysis import calc_monotonic_range


class CopyOp(CustomOp):
    """A custom op as a third party writes one: it copies its input to its output,
    states its output's datatype as its input's, and says nothing of a model. Each
    class records the models attached to it and the versions it was built at."""

    attached: list = []
    built: list = []

    def __init__(self, onnx_node, onnx_opset_version=1):
        super().__init__(onnx_node, onnx_opset_version)
        type(self).built.append(onnx_opset_version)

    def attach_model(self, model):
        type(self).attached.append(model)
        return super().attach_model(model)

    def get_nodeattr_types(self):
        return {}

    def make_shape_compatible_op(self, model):
        return helper.make_node("Identity", [self.onnx_node.input[0]], [self.onnx_node.output[0]])

    def infer_node_datatype(self, model):
        node = self.onnx_node
        model.set_tensor_datatype(node.output[0], model.get_tensor_datatype(node.input[0]))

    def execute_node(self, context, graph):
        if self.wants_model:
            assert self._model is not None and self._model.graph is graph
        context[self.onnx_node.output[0]] = context[self.onnx_node.input[0]]

    def verify_node(self):
        pass


def make_op(name, base=CopyOp, **body):
    return type(name, (base,), {"attached": [], "built": [], **body})


def fresh_domain():
    return "qonnx_test_context_" + uuid.uuid4().hex[:8]


def make_model(domain, op_type="Thing", imported=1, x=None):
    """x (an initializer, so folding sees a constant) -> domain.op_type -> y -> Relu -> out."""
    nodes = [helper.make_node(op_type, ["x"], ["y"], domain=domain), helper.make_node("Relu", ["y"], ["out"])]
    inputs = [] if x is not None else [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])]
    graph = helper.make_graph(nodes, "g", inputs, [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4])])
    if x is not None:
        graph.initializer.append(helper.make_tensor("x", TensorProto.FLOAT, [1, 4], x.flatten()))
    imports = [helper.make_opsetid("", 13)]
    if imported is not None:
        imports.append(helper.make_opsetid(domain, imported))
    model = ModelWrapper(qonnx_make_model(graph, opset_imports=imports))
    model.set_tensor_datatype("x", DataType["INT4"])
    return model


def test_an_op_wants_no_model_by_default():
    assert CustomOp.wants_model is False
    assert CustomOp._model is None
    assert all(not cls.wants_model for _, cls in get_ops_in_domain("qonnx.custom_op.general"))


def test_attach_model_keeps_the_model_and_returns_the_op():
    domain = fresh_domain()
    Thing = make_op("Thing")
    add_op_to_domain(domain, Thing)
    model = make_model(domain)
    inst = getCustomOp(model.graph.node[0])
    assert inst._model is None
    assert inst.attach_model(model) is inst
    assert inst.attach_model(model) is inst
    assert inst._model is model


@pytest.mark.parametrize("wants_model", [False, True])
def test_get_customop_wrapper_attaches_only_an_op_that_wants_its_model(wants_model):
    domain = fresh_domain()
    Thing = make_op("Thing", wants_model=wants_model)
    add_op_to_domain(domain, Thing)
    model = make_model(domain)
    inst = model.get_customop_wrapper(model.graph.node[0])
    assert type(inst) is Thing
    assert inst._model is (model if wants_model else None)
    assert Thing.attached == ([model] if wants_model else [])


def test_a_bare_lookup_never_attaches():
    domain = fresh_domain()
    Thing = make_op("Thing", wants_model=True)
    add_op_to_domain(domain, Thing)
    model = make_model(domain)
    assert getCustomOp(model.graph.node[0])._model is None
    assert Thing.attached == []


def run_everything(model, attached=None):
    """Every place qonnx builds a custom op for a model: inference, execution, range
    analysis. Where ``attached`` (an op class's list of models) is given, each place
    must give the op its model."""

    def step(run):
        before = len(attached) if attached is not None else 0
        result = run()
        if attached is not None:
            assert len(attached) > before, run
        return result

    model = step(lambda: model.transform(InferShapes()))
    model = step(lambda: model.transform(InferDataTypes()))
    step(lambda: execute_onnx(model, {"x": np.ones((1, 4), dtype=np.float32)}))
    step(lambda: calc_monotonic_range(model.graph.node[0], model, {"x": (-1.0, 1.0)}))
    return model


@pytest.mark.parametrize("wants_model", [False, True])
def test_every_pass_attaches_only_an_op_that_wants_its_model(wants_model):
    domain = fresh_domain()
    Thing = make_op("Thing", wants_model=wants_model)
    add_op_to_domain(domain, Thing)
    run_everything(make_model(domain), Thing.attached if wants_model else None)
    before = len(Thing.attached)
    folded = make_model(domain, x=np.arange(4, dtype=np.float32)).transform(FoldConstants(exclude_op_types=["Relu"]))
    assert folded.get_initializer("y") is not None
    if wants_model:
        assert len(Thing.attached) > before
        assert all(isinstance(model, ModelWrapper) for model in Thing.attached)
    else:
        assert Thing.attached == []


def test_an_attached_op_answers_from_its_model_and_not_a_copy():
    domain = fresh_domain()
    Thing = make_op("Thing", wants_model=True)
    add_op_to_domain(domain, Thing)
    model = make_model(domain)
    inst = model.get_customop_wrapper(model.graph.node[0])
    copied = model.transform(InferShapes())
    assert copied is not model
    assert inst._model is model
    assert copied.get_customop_wrapper(copied.graph.node[0])._model is copied


def test_a_model_without_its_domain_import_infers_without_a_warning():
    domain = fresh_domain()
    add_op_to_domain(domain, make_op("Thing"))
    model = make_model(domain, imported=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        execute_onnx(model, {"x": np.ones((1, 4), dtype=np.float32)})
    assert model.get_tensor_datatype("y") == DataType["INT4"]
    with pytest.warns(
        UserWarning, match=f"Domain {domain} not found in model opset imports, using fallback_customop_version=1"
    ):
        model.get_customop_wrapper(model.graph.node[0])


def test_a_missing_op_of_an_imported_domain_is_missing_without_a_warning():
    domain = fresh_domain()
    add_op_to_domain(domain, make_op("Thing"))
    model = make_model(domain, op_type="Absent")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(KeyError, match="Absent"):
            model.get_customop_wrapper(model.graph.node[0])


def test_an_imported_version_below_every_version_is_refused_not_replaced_by_the_fallback():
    domain = fresh_domain()
    add_op_to_domain(domain, make_op("Thing_v2"))
    model = make_model(domain, imported=1)
    with pytest.raises(KeyError, match="No suitable version"):
        model.get_customop_wrapper(model.graph.node[0], fallback_customop_version=2)


def test_a_missing_op_of_an_unimported_domain_warns_and_is_missing_as_before():
    domain = fresh_domain()
    add_op_to_domain(domain, make_op("Thing"))
    model = make_model(domain, op_type="Absent", imported=None)
    with pytest.warns(UserWarning, match="not found in model opset imports"):
        with pytest.raises(KeyError, match="Absent"):
            model.get_customop_wrapper(model.graph.node[0])


class Negate_v2(CopyOp):
    """Version 2 of the op: negates its input, states INT8, and stands in for shape
    inference with a shape of its own; each answer differs from version 1's."""

    def make_shape_compatible_op(self, model):
        return self.make_const_shape_op([2, 2])

    def infer_node_datatype(self, model):
        model.set_tensor_datatype(self.onnx_node.output[0], DataType["INT8"])

    def execute_node(self, context, graph):
        context[self.onnx_node.output[0]] = -context[self.onnx_node.input[0]]


def test_the_imported_version_is_inferred_folded_and_executed():
    domain = fresh_domain()
    Negate = make_op("Negate")
    Negate_v2_class = make_op("Negate_v2", base=Negate_v2)
    add_op_to_domain(domain, Negate)
    add_op_to_domain(domain, Negate_v2_class)
    model = run_everything(make_model(domain, op_type="Negate", imported=1))
    ranges = {"x": (-1.0, 2.0)}
    calc_monotonic_range(model.graph.node[0], model, ranges)
    assert (ranges["y"][0] == -1.0).all() and (ranges["y"][1] == 2.0).all()
    assert model.get_tensor_shape("y") == [1, 4]
    assert model.get_tensor_datatype("y") == DataType["INT4"]
    x = np.array([[1.0, -2.0, 3.0, -4.0]], dtype=np.float32)
    assert (execute_onnx(model, {"x": x}, return_full_exec_context=True)["y"] == x).all()
    folded = make_model(domain, op_type="Negate", imported=1, x=x).transform(FoldConstants(exclude_op_types=["Relu"]))
    assert (folded.get_initializer("y") == x).all()
    assert set(Negate.built) == {1}
    assert Negate_v2_class.built == []
    model = make_model(domain, op_type="Negate", imported=2)
    model.set_tensor_shape("y", [1, 4])
    assert (execute_onnx(model, {"x": x}, return_full_exec_context=True)["y"] == -x).all()
    assert model.transform(InferDataTypes()).get_tensor_datatype("y") == DataType["INT8"]
    assert set(Negate_v2_class.built) == {2}


def make_trunc_model():
    """A Trunc node as version 1 states it (five inputs), in a model importing the
    standard domain at 11 and qonnx.custom_op.general at 1."""
    x = np.array([[0.3, 1.7, -2.6, 5.2]], dtype=np.float32)
    inits = [
        helper.make_tensor("x", TensorProto.FLOAT, [1, 4], x.flatten()),
        helper.make_tensor("scale", TensorProto.FLOAT, [], [1.0]),
        helper.make_tensor("zeropt", TensorProto.FLOAT, [], [0.0]),
        helper.make_tensor("in_bits", TensorProto.FLOAT, [], [8.0]),
        helper.make_tensor("out_bits", TensorProto.FLOAT, [], [4.0]),
    ]
    nodes = [
        helper.make_node(
            "Trunc",
            ["x", "scale", "zeropt", "in_bits", "out_bits"],
            ["y"],
            domain="qonnx.custom_op.general",
            rounding_mode="FLOOR",
        ),
        helper.make_node("Relu", ["y"], ["out"]),
    ]
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("in", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4])],
        initializer=inits,
        value_info=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
    )
    imports = [helper.make_opsetid("", 11), helper.make_opsetid("qonnx.custom_op.general", 1)]
    return ModelWrapper(qonnx_make_model(graph, opset_imports=imports))


def test_folding_a_version_1_trunc_computes_what_execution_computes():
    model = make_trunc_model()
    executed = execute_onnx(model, {"in": np.zeros(1, dtype=np.float32)}, return_full_exec_context=True)["y"]
    folded = make_trunc_model().transform(FoldConstants(exclude_op_types=["Relu"]))
    assert [n.op_type for n in folded.graph.node] == ["Relu"]
    assert (folded.get_initializer("y") == executed).all()


def watching(base, name, **body):
    """A subclass of one of qonnx's ops that wants its model and records each model
    attached, registered under the op's own type in a fresh domain."""
    cls = type(name, (base,), {"wants_model": True, "attached": [], **body})

    def attach_model(self, model):
        type(self).attached.append(model)
        return base.attach_model(self, model)

    cls.attach_model = attach_model
    domain = fresh_domain()
    add_op_to_domain(domain, cls, op_type=base.__name__)
    return domain, cls


def test_infer_data_layouts_gives_an_op_its_model():
    def get_nodeattr(self, name):
        # answers from its model: NHWC once it has one
        if name == "data_layout":
            return "NHWC" if self._model is not None else "NCHW"
        return MultiThreshold.get_nodeattr(self, name)

    domain, Watching = watching(MultiThreshold, "WatchingThreshold", get_nodeattr=get_nodeattr)
    thresholds = helper.make_tensor("t", TensorProto.FLOAT, [2, 1], [0.5, 0.5])
    node = helper.make_node("MultiThreshold", ["x", "t"], ["y"], domain=domain, out_dtype="UINT1")
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 3, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 3, 2])],
        initializer=[thresholds],
    )
    model = ModelWrapper(qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 13)]))
    model = model.transform(InferDataLayouts())
    assert model.get_tensor_layout("y") == DataLayout.NHWC
    assert Watching.attached


def test_rebalance_im2col_gives_an_op_its_model():
    domain, Watching = watching(Im2Col, "WatchingIm2Col")
    node = helper.make_node(
        "Im2Col", ["x"], ["y"], domain=domain, stride=[2, 2], kernel_size=[2, 2], input_shape="(1,4,4,1)"
    )
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 4, 1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 2, 4])],
    )
    model = ModelWrapper(qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 13)]))
    model.transform(RebalanceIm2Col(2))
    assert Watching.attached


def test_bipolar_matmul_conversion_gives_an_op_its_model():
    read_with_model = []

    def get_nodeattr(self, name):
        if name == "out_scale":
            # the conversion reads it from the op it builds
            read_with_model.append(self._model is not None)
        return MultiThreshold.get_nodeattr(self, name)

    domain, Watching = watching(MultiThreshold, "WatchingBipolar", get_nodeattr=get_nodeattr)
    thresholds = helper.make_tensor("t", TensorProto.FLOAT, [1, 1], [0.0])
    weights = helper.make_tensor("w", TensorProto.FLOAT, [4, 2], [1.0, -1.0] * 4)
    nodes = [
        helper.make_node(
            "MultiThreshold", ["x", "t"], ["b"], domain=domain, out_dtype="BIPOLAR", out_scale=2.0, out_bias=-1.0
        ),
        helper.make_node("MatMul", ["b", "w"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
        initializer=[thresholds, weights],
        value_info=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 4])],
    )
    model = ModelWrapper(qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 13)]))
    model.set_tensor_datatype("b", DataType["BIPOLAR"])
    model.set_tensor_datatype("w", DataType["BIPOLAR"])
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    assert read_with_model and all(read_with_model)
    assert model.get_tensor_datatype("b") == DataType["BINARY"]


def trunc_config(imported, version=1):
    """The extracted config of one Trunc node in version 1's form (five inputs) or
    version 2's (six), its domain imported at ``imported`` (None: not imported)."""
    inputs = ["x", "scale", "zeropt", "in_bits"] + (["out_scale"] if version == 2 else []) + ["out_bits"]
    node = helper.make_node("Trunc", inputs, ["y"], domain="qonnx.custom_op.general", name="t0", rounding_mode="FLOOR")
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    imports = [helper.make_opsetid("", 11)]
    if imported is not None:
        imports.append(helper.make_opsetid("qonnx.custom_op.general", imported))
    model = ModelWrapper(qonnx_make_model(graph, opset_imports=imports))
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "config.json")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            extract_model_config_to_json(model, path, ["rounding_mode", "narrow", "signed"])
        with open(path) as stream:
            return json.load(stream)


@pytest.mark.parametrize("imported", [1, None])
def test_a_version_1_trunc_reports_its_own_attributes_in_the_config(imported):
    assert trunc_config(imported) == {"Defaults": {}, "t0": {"rounding_mode": "FLOOR"}}


def test_a_version_2_trunc_reports_version_2_attributes_in_the_config():
    expected = {"Defaults": {}, "t0": {"rounding_mode": "FLOOR", "narrow": 0, "signed": 1}}
    assert trunc_config(2, version=2) == expected


def test_config_extraction_gives_an_op_its_model():
    domain = fresh_domain()
    Thing = make_op("Thing", wants_model=True)
    add_op_to_domain(domain, Thing)
    model = make_model(domain)
    model.graph.node.remove(model.graph.node[1])
    with tempfile.TemporaryDirectory() as folder:
        extract_model_config_to_json(model, os.path.join(folder, "config.json"), [])
    assert Thing.attached == [model]
