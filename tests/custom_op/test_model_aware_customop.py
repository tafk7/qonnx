# Copyright (c) 2025 Advanced Micro Devices, Inc.
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

"""Backward-compatibility + delivery tests for the opt-in model-aware custom-op
contract (CustomOp.wants_model / attach_model / get_customop_wrapper).

The whole product is backward compatibility: a classic op that does not opt in must
be indistinguishable from what getCustomOp returned before this contract existed. A
context-dependent op that opts in receives the ModelWrapper only through the
model-aware entry point (get_customop_wrapper), never through bare getCustomOp."""

import pytest

import json
import numpy as np
import onnx.parser as oprs
import warnings

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.custom_op.base import CustomOp
from qonnx.custom_op.registry import add_op_to_domain, getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.config import extract_model_config_to_json


class ClassicTestOp(CustomOp):
    """A context-free op: does not opt in, so it must be unchanged by the contract."""

    def get_nodeattr_types(self):
        return {"my_attr": ("i", True, 0)}

    def make_shape_compatible_op(self, model):
        ishape = model.get_tensor_shape(self.onnx_node.input[0])
        return super().make_const_shape_op(ishape)

    def infer_node_datatype(self, model):
        node = self.onnx_node
        model.set_tensor_datatype(node.output[0], model.get_tensor_datatype(node.input[0]))

    def execute_node(self, context, graph):
        node = self.onnx_node
        context[node.output[0]] = context[node.input[0]]

    def verify_node(self):
        pass


class ModelAwareTestOp(ClassicTestOp):
    """A context-dependent op: opts in via wants_model, and records the attach."""

    wants_model = True

    def attach_model(self, model):
        self.attach_calls = getattr(self, "attach_calls", 0) + 1
        return super().attach_model(model)

    def execute_node(self, context, graph):
        assert self._model.graph is graph
        super().execute_node(context, graph)


def _make_model(op_type, with_domain_import=True):
    ishp = (1, 10)
    ishp_str = str(list(ishp))
    domain_import = ', "qonnx.custom_op.general" : 1' if with_domain_import else ""
    input = f"""
    <
        ir_version: 7,
        opset_import: ["" : 9{domain_import}]
    >
    agraph (float{ishp_str} in0) => (float{ishp_str} out0)
    {{
        out0 = qonnx.custom_op.general.{op_type}<my_attr=3>(in0)
    }}
    """
    return ModelWrapper(oprs.parse_model(input))


def test_model_aware_op_receives_model_via_wrapper_only():
    """(a) A wants_model=True op gets the model via get_customop_wrapper (its _model
    is set, attach_model ran once) and NOT via bare getCustomOp (no _model)."""
    add_op_to_domain("qonnx.custom_op.general", ModelAwareTestOp)
    model = _make_model("ModelAwareTestOp")
    node = model.graph.node[0]

    # bare getCustomOp: no attach, no _model reference
    bare = getCustomOp(node)
    assert isinstance(bare, ModelAwareTestOp)
    assert getattr(bare, "_model", None) is None
    assert getattr(bare, "attach_calls", 0) == 0

    # model-aware path: the model is attached
    aware = model.get_customop_wrapper(node)
    assert isinstance(aware, ModelAwareTestOp)
    assert aware._model is model
    assert aware.attach_calls == 1


def test_classic_op_identical_from_both_paths():
    """(b) A wants_model=False op is byte-for-byte identical from getCustomOp and
    get_customop_wrapper: no _model attached, no behavior change."""
    add_op_to_domain("qonnx.custom_op.general", ClassicTestOp)
    model = _make_model("ClassicTestOp")
    node = model.graph.node[0]

    bare = getCustomOp(node)
    wrapped = model.get_customop_wrapper(node)

    assert type(bare) is type(wrapped) is ClassicTestOp
    assert getattr(bare, "_model", None) is None
    assert getattr(wrapped, "_model", None) is None
    assert bare.wants_model is False
    assert bare.get_nodeattr("my_attr") == wrapped.get_nodeattr("my_attr") == 3


def test_opset_version_selection_unchanged():
    """(c) Regression: get_customop_wrapper's opset-version selection is unchanged.
    The domain in the model's opset imports (v1) is honored, and a missing domain
    falls back to the requested fallback_customop_version."""
    add_op_to_domain("qonnx.custom_op.general", ClassicTestOp)
    model = _make_model("ClassicTestOp")
    node = model.graph.node[0]

    # domain present in opset imports (v1) -> that version selected
    inst = model.get_customop_wrapper(node)
    assert inst.onnx_opset_version == 1

    # domain absent from imports -> fallback_customop_version used (except branch)
    model_no_import = _make_model("ClassicTestOp", with_domain_import=False)
    node_no_import = model_no_import.graph.node[0]
    inst_fb = model_no_import.get_customop_wrapper(node_no_import, fallback_customop_version=1)
    assert inst_fb.onnx_opset_version == 1


def test_onnx_execution_attaches_model_to_model_aware_custom_op():
    add_op_to_domain("qonnx.custom_op.general", ModelAwareTestOp)
    model = _make_model("ModelAwareTestOp")
    input_value = np.arange(10, dtype=np.float32).reshape(1, 10)

    output = execute_onnx(model, {"in0": input_value})

    np.testing.assert_array_equal(output["out0"], input_value)


def test_unknown_op_in_imported_domain_raises_without_fallback_warning():
    """An op that does not exist raises the registry's KeyError; the opset fallback
    (and its warning) is only for a domain the model does not import."""
    model = _make_model("ClassicTestOp")
    node = model.graph.node[0]
    missing = type(node)()
    missing.CopyFrom(node)
    missing.op_type = "NoSuchTestOp"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(KeyError, match="NoSuchTestOp"):
            model.get_customop_wrapper(missing)


class RecordingModelAwareTestOp(ModelAwareTestOp):
    """Records every model it is attached to, across instances."""

    attached_to = []

    def attach_model(self, model):
        RecordingModelAwareTestOp.attached_to.append(model)
        return super().attach_model(model)


def test_model_config_extraction_attaches_model(tmp_path):
    """extract_model_config_to_json builds ops through get_customop_wrapper, so a
    model-aware op is attached to the model it reads attributes from."""
    add_op_to_domain("qonnx.custom_op.general", RecordingModelAwareTestOp)
    RecordingModelAwareTestOp.attached_to = []
    model = _make_model("RecordingModelAwareTestOp")
    model = model.transform(GiveUniqueNodeNames())
    cfg_file = tmp_path / "cfg.json"

    extract_model_config_to_json(model, str(cfg_file), ["my_attr"])

    assert RecordingModelAwareTestOp.attached_to == [model]
    cfg = json.loads(cfg_file.read_text())
    assert cfg[model.graph.node[0].name] == {"my_attr": 3}


def test_attached_instance_does_not_follow_a_transformed_copy():
    """Attach lifetime: an instance borrows the ModelWrapper it came from. After a
    transformation (a deep copy by default) it still answers from the old model;
    the transformed model gives a new instance attached to itself."""
    add_op_to_domain("qonnx.custom_op.general", ModelAwareTestOp)
    model = _make_model("ModelAwareTestOp")
    old_inst = model.get_customop_wrapper(model.graph.node[0])

    transformed = model.transform(GiveUniqueNodeNames())

    assert transformed is not model
    assert old_inst._model is model
    assert old_inst.onnx_node is model.graph.node[0]
    assert old_inst.onnx_node.name == ""
    new_inst = transformed.get_customop_wrapper(transformed.graph.node[0])
    assert new_inst._model is transformed
    assert new_inst.onnx_node.name != ""
