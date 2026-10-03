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

"""Op identity in the custom-op registry: one rule for versioned names, and an
op_type/op_version stated in a class's own body, not inherited."""

import pytest

import sys
import types
import uuid
import warnings
from onnx import helper

from qonnx.custom_op.base import CustomOp
from qonnx.custom_op.registry import (
    get_ops_in_domain,
    get_supported_versions,
    getCustomOp,
    op_identity,
    split_versioned_name,
)


class _Op(CustomOp):
    def get_nodeattr_types(self):
        return {}

    def make_shape_compatible_op(self, model):
        return helper.make_node("Identity", [self.onnx_node.input[0]], [self.onnx_node.output[0]])

    def infer_node_datatype(self, model):
        pass

    def execute_node(self, context, graph):
        context[self.onnx_node.output[0]] = context[self.onnx_node.input[0]]

    def verify_node(self):
        pass


def _domain(**members):
    """A fresh importable domain module exporting the given classes."""
    name = "qonnx_registry_probe_" + uuid.uuid4().hex[:8]
    module = types.ModuleType(name)
    for key, value in members.items():
        setattr(module, key, value)
    module.__all__ = [k for k in members if k != "opset_version"]
    sys.modules[name] = module
    return name


@pytest.mark.parametrize(
    "name, identity",
    [
        ("IntQuant", ("IntQuant", 1)),
        ("IntQuant_v2", ("IntQuant", 2)),
        ("BatchNormalization_v14", ("BatchNormalization", 14)),
        ("Thresholding_vitis", ("Thresholding_vitis", 1)),
        ("My_var_v2", ("My_var", 2)),
        ("Op_v0", ("Op_v0", 1)),
        ("Op_v02", ("Op_v02", 1)),
        ("Op_v2_rtl", ("Op_v2_rtl", 1)),
    ],
)
def test_one_rule_splits_versioned_names(name, identity):
    assert split_versioned_name(name) == identity


def test_a_name_containing_v_does_not_shadow_another_op():
    """Thresholding_vitis is its own op, and Thresholding stays reachable."""
    Thresholding = type("Thresholding", (_Op,), {})
    Thresholding_vitis = type("Thresholding_vitis", (_Op,), {})
    domain = _domain(Thresholding=Thresholding, Thresholding_vitis=Thresholding_vitis)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert type(getCustomOp(helper.make_node("Thresholding", ["x"], ["y"], domain=domain))) is Thresholding
        vitis = getCustomOp(helper.make_node("Thresholding_vitis", ["x"], ["y"], domain=domain))
    assert type(vitis) is Thresholding_vitis


def test_explicit_identity_is_not_inherited():
    """A kernel op states its op type and version; its backend subclass does not
    become the kernel op by inheritance."""
    MatMulOp = type("MatMulOp", (_Op,), {"op_type": "MatMul", "op_version": 6})
    MatMulOp_hls = type("MatMulOp_hls", (MatMulOp,), {})
    assert op_identity(MatMulOp) == ("MatMul", 6)
    assert op_identity(MatMulOp_hls) == ("MatMulOp_hls", 1)


@pytest.mark.parametrize("op_type, op_version", [("", 1), (None, 1), ("MatMul", 0), ("MatMul", "6"), ("MatMul", True)])
def test_a_stated_identity_must_be_valid(op_type, op_version):
    Bad = type("Bad", (_Op,), {"op_type": op_type, "op_version": op_version})
    with pytest.raises(ValueError):
        op_identity(Bad)


def test_stated_and_named_identities_in_one_domain():
    """A kernel op stated as MatMul 6, its backend subclass exported under its
    own name, and a named Trunc_v2 beside Trunc: each found under its own
    identity."""
    MatMulOp = type("MatMulOp", (_Op,), {"op_type": "MatMul", "op_version": 6})
    MatMulOp_hls = type("MatMulOp_hls", (MatMulOp,), {})
    Trunc = type("Trunc", (_Op,), {})
    Trunc_v2 = type("Trunc_v2", (_Op,), {})
    domain = _domain(MatMulOp=MatMulOp, MatMulOp_hls=MatMulOp_hls, Trunc=Trunc, Trunc_v2=Trunc_v2)
    assert dict(get_ops_in_domain(domain)) == {"MatMul": MatMulOp, "MatMulOp_hls": MatMulOp_hls, "Trunc": Trunc_v2}
    assert get_supported_versions(domain, "MatMul") == [6]
    assert get_supported_versions(domain, "Trunc") == [1, 2]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        inst = getCustomOp(helper.make_node("MatMul", ["x"], ["y"], domain=domain), onnx_opset_version=6)
    assert type(inst) is MatMulOp
    with pytest.raises(KeyError):
        getCustomOp(helper.make_node("MatMulOp", ["x"], ["y"], domain=domain))
