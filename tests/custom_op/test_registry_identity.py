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

"""Op identity in the custom-op registry: one rule for versioned names, an
op_type/op_version stated in a class's own body (not inherited), and registered
and exported versions merged, a duplicate refused; a domain's opset version,
and versions resolved from the model's opset import."""

import pytest

import sys
import types
import uuid
import warnings
from onnx import helper

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.base import CustomOp
from qonnx.custom_op.registry import (
    add_op_to_domain,
    get_domain_opset_version,
    get_ops_in_domain,
    get_supported_versions,
    getCustomOp,
    op_identity,
    split_versioned_name,
)
from qonnx.util.basic import qonnx_make_model


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


def _model(domain, op_type, imported_version=None):
    node = helper.make_node(op_type, ["x"], ["y"], domain=domain)
    x = helper.make_tensor_value_info("x", 1, [1, 4])
    y = helper.make_tensor_value_info("y", 1, [1, 4])
    graph = helper.make_graph([node], "g", [x], [y])
    opsets = [helper.make_opsetid("", 13)]
    if imported_version is not None:
        opsets.append(helper.make_opsetid(domain, imported_version))
    return ModelWrapper(qonnx_make_model(graph, opset_imports=opsets))


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


def test_registering_a_version_keeps_the_exported_ones():
    Foo = type("Foo", (_Op,), {})
    Foo_v2 = type("Foo_v2", (_Op,), {})
    Foo_v3 = type("Foo_v3", (_Op,), {})
    domain = _domain(Foo=Foo, Foo_v2=Foo_v2)
    add_op_to_domain(domain, Foo_v3)
    assert get_supported_versions(domain, "Foo") == [1, 2, 3]


def test_a_registered_class_replaces_an_exported_one_for_its_version_only():
    Foo = type("Foo", (_Op,), {})
    Foo_v2 = type("Foo_v2", (_Op,), {})
    Patched = type("Patched", (_Op,), {})
    domain = _domain(Foo=Foo, Foo_v2=Foo_v2)
    add_op_to_domain(domain, Patched, op_type="Foo", op_version=2)
    node = helper.make_node("Foo", ["x"], ["y"], domain=domain)
    assert type(getCustomOp(node, onnx_opset_version=1)) is Foo
    assert type(getCustomOp(node, onnx_opset_version=2)) is Patched
    with pytest.raises(ValueError):
        add_op_to_domain(domain, Patched, op_type="Foo", op_version=0)


def test_two_classes_exported_for_one_version_are_refused():
    A = type("Bar", (_Op,), {})
    B = type("Bar2", (_Op,), {"op_type": "Bar"})
    domain = _domain(Bar=A, Bar2=B)
    with pytest.raises(ValueError, match="exported twice"):
        get_supported_versions(domain, "Bar")


def test_one_class_exported_twice_is_not_a_duplicate():
    """__all__ and a legacy custom_op dict naming the same class, and the same
    class registered again at run time."""
    Baz = type("Baz", (_Op,), {})
    domain = _domain(Baz=Baz)
    sys.modules[domain].custom_op = {"Baz": Baz}
    add_op_to_domain(domain, Baz)
    add_op_to_domain(domain, Baz)
    assert get_supported_versions(domain, "Baz") == [1]


def test_a_bare_lookup_of_a_multi_version_op_warns():
    Qux = type("Qux", (_Op,), {})
    Qux_v2 = type("Qux_v2", (_Op,), {})
    Single = type("Single", (_Op,), {})
    domain = _domain(Qux=Qux, Qux_v2=Qux_v2, Single=Single)
    with pytest.warns(UserWarning, match="without the model's opset import"):
        assert type(getCustomOp(helper.make_node("Qux", ["x"], ["y"], domain=domain))) is Qux_v2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        getCustomOp(helper.make_node("Qux", ["x"], ["y"], domain=domain), onnx_opset_version=1)
        getCustomOp(helper.make_node("Single", ["x"], ["y"], domain=domain))


def test_domain_opset_version_is_stated_or_the_highest_since_version():
    MatMul = type("MatMul", (_Op,), {})
    MatMul_v6 = type("MatMul_v6", (_Op,), {})
    Thresholding = type("Thresholding", (_Op,), {})
    assert get_domain_opset_version(_domain(MatMul=MatMul, MatMul_v6=MatMul_v6, Thresholding=Thresholding)) == 6
    assert get_domain_opset_version(_domain(MatMul=MatMul, MatMul_v6=MatMul_v6, opset_version=7)) == 7
    with pytest.raises(ValueError, match="below"):
        get_domain_opset_version(_domain(MatMul=MatMul, MatMul_v6=MatMul_v6, opset_version=5))
    registered = _domain(MatMul=MatMul)
    add_op_to_domain(registered, MatMul_v6)
    assert get_domain_opset_version(registered) == 6


def test_a_kernel_family_version_is_resolved_from_the_model_opset_import():
    """A domain at opset 6 where MatMul changed at 6 and Thresholding never did:
    a model importing the domain at 1 gets MatMul 1, at 6 or later MatMul 6;
    Thresholding is version 1 in both. Bare getCustomOp has no model and takes
    the highest version, with a warning."""
    MatMul = type("MatMul", (_Op,), {"op_type": "MatMul", "op_version": 1})
    MatMulV6 = type("MatMulV6", (_Op,), {"op_type": "MatMul", "op_version": 6})
    Thresholding = type("Thresholding", (_Op,), {})
    domain = _domain(MatMul=MatMul, MatMulV6=MatMulV6, Thresholding=Thresholding)

    for imported, expected in [(1, MatMul), (5, MatMul), (6, MatMulV6), (9, MatMulV6)]:
        model = _model(domain, "MatMul", imported)
        inst = model.get_customop_wrapper(model.graph.node[0])
        assert type(inst) is expected
        assert inst.onnx_opset_version == op_identity(expected)[1]
        model_t = _model(domain, "Thresholding", imported)
        assert type(model_t.get_customop_wrapper(model_t.graph.node[0])) is Thresholding

    model = _model(domain, "MatMul", None)
    with pytest.warns(UserWarning, match="not found in model opset imports"):
        assert type(model.get_customop_wrapper(model.graph.node[0])) is MatMul
    with pytest.warns(UserWarning, match="without the model's opset"):
        assert type(getCustomOp(model.graph.node[0])) is MatMulV6
