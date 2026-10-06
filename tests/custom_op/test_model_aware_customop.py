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

"""The opt-in model-aware custom-op contract (CustomOp.wants_model,
attach_model, ModelWrapper.get_customop_wrapper): an op that opts in receives
the ModelWrapper only through the model-aware entry point, never through bare
getCustomOp; an op that does not opt in is the same from both."""

import pytest

import json
import numpy as np
import onnx.parser as oprs
import warnings

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.config import extract_model_config_to_json


@pytest.fixture
def classic_op(make_op):
    """A context-free op: it does not opt in."""
    return make_op("ClassicOp")


@pytest.fixture
def model_aware_op(make_op):
    """A context-dependent op: it opts in via wants_model."""
    return make_op("ModelAwareOp", wants_model=True)


def _make_model(domain, op_type, with_domain_import=True):
    ishp_str = str([1, 10])
    domain_import = f', "{domain}" : 1' if with_domain_import else ""
    input = f"""
    <
        ir_version: 7,
        opset_import: ["" : 9{domain_import}]
    >
    agraph (float{ishp_str} in0) => (float{ishp_str} out0)
    {{
        out0 = {domain}.{op_type}<my_attr=3>(in0)
    }}
    """
    return ModelWrapper(oprs.parse_model(input))


def test_model_aware_op_receives_model_via_wrapper_only(make_domain, model_aware_op):
    """A wants_model=True op gets the model via get_customop_wrapper (attached
    once), and not via bare getCustomOp."""
    model = _make_model(make_domain(ModelAwareOp=model_aware_op), "ModelAwareOp")
    node = model.graph.node[0]

    bare = getCustomOp(node)
    assert isinstance(bare, model_aware_op)
    assert bare._model is None
    assert model_aware_op.attached_to == []

    aware = model.get_customop_wrapper(node)
    assert isinstance(aware, model_aware_op)
    assert aware._model is model
    assert model_aware_op.attached_to == [model]


def test_classic_op_identical_from_both_paths(make_domain, classic_op):
    """A wants_model=False op is the same from getCustomOp and
    get_customop_wrapper: nothing attached."""
    model = _make_model(make_domain(ClassicOp=classic_op), "ClassicOp")
    node = model.graph.node[0]

    bare = getCustomOp(node)
    wrapped = model.get_customop_wrapper(node)

    assert type(bare) is type(wrapped) is classic_op
    assert bare._model is None
    assert getattr(wrapped, "_model", None) is None
    assert classic_op.attached_to == []
    assert bare.get_nodeattr("my_attr") == wrapped.get_nodeattr("my_attr") == 3


def test_opset_version_from_the_import_or_the_fallback(make_domain, classic_op):
    """get_customop_wrapper takes the version from the model's import of the
    domain, and fallback_customop_version for a domain the model does not import."""
    domain = make_domain(ClassicOp=classic_op)
    model = _make_model(domain, "ClassicOp")
    assert model.get_customop_wrapper(model.graph.node[0]).onnx_opset_version == 1

    model_no_import = _make_model(domain, "ClassicOp", with_domain_import=False)
    with pytest.warns(UserWarning, match="not found in model opset imports"):
        inst_fb = model_no_import.get_customop_wrapper(model_no_import.graph.node[0], fallback_customop_version=1)
    assert inst_fb.onnx_opset_version == 1


def test_onnx_execution_attaches_model_to_model_aware_custom_op(make_domain, model_aware_op):
    model = _make_model(make_domain(ModelAwareOp=model_aware_op), "ModelAwareOp")
    input_value = np.arange(10, dtype=np.float32).reshape(1, 10)

    output = execute_onnx(model, {"in0": input_value})

    np.testing.assert_array_equal(output["out0"], input_value)
    assert model_aware_op.attached_to == [model]


def test_unknown_op_in_imported_domain_raises_without_fallback_warning(make_domain, classic_op):
    """An op that does not exist raises the registry's KeyError; the opset fallback
    (and its warning) is only for a domain the model does not import."""
    model = _make_model(make_domain(ClassicOp=classic_op), "ClassicOp")
    node = model.graph.node[0]
    missing = type(node)()
    missing.CopyFrom(node)
    missing.op_type = "NoSuchTestOp"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(KeyError, match="NoSuchTestOp"):
            model.get_customop_wrapper(missing)


def test_model_config_extraction_attaches_model(tmp_path, make_domain, model_aware_op):
    """extract_model_config_to_json builds ops through get_customop_wrapper, so a
    model-aware op is attached to the model it reads attributes from."""
    model = _make_model(make_domain(ModelAwareOp=model_aware_op), "ModelAwareOp")
    model = model.transform(GiveUniqueNodeNames())
    cfg_file = tmp_path / "cfg.json"

    extract_model_config_to_json(model, str(cfg_file), ["my_attr"])

    assert model_aware_op.attached_to == [model]
    cfg = json.loads(cfg_file.read_text())
    assert cfg[model.graph.node[0].name] == {"my_attr": 3}


def test_attached_instance_does_not_follow_a_transformed_copy(make_domain, model_aware_op):
    """Attach lifetime: an instance borrows the ModelWrapper it came from. After a
    transformation (a deep copy by default) it still answers from the old model;
    the transformed model gives a new instance attached to itself."""
    model = _make_model(make_domain(ModelAwareOp=model_aware_op), "ModelAwareOp")
    old_inst = model.get_customop_wrapper(model.graph.node[0])

    transformed = model.transform(GiveUniqueNodeNames())

    assert transformed is not model
    assert old_inst._model is model
    assert old_inst.onnx_node is model.graph.node[0]
    assert old_inst.onnx_node.name == ""
    new_inst = transformed.get_customop_wrapper(transformed.graph.node[0])
    assert new_inst._model is transformed
    assert new_inst.onnx_node.name != ""
