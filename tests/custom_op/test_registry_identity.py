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

"""Op identity and versions in the custom op registry: a domain module's exports are
discovered by upstream's name rules, unchanged; a class states its own op type and
version only through the explicit registration paths; a domain has an opset version."""

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
    is_custom_op,
    op_identity,
)
from qonnx.util.basic import qonnx_make_model


class CopyOp(CustomOp):
    """A custom op as a third party writes one: it copies its input to its output and
    states nothing for the registry."""

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


def make_op(name, **body):
    """A new CopyOp class of that name, the body's members stated in its own body."""
    return type(name, (CopyOp,), dict(body))


@pytest.fixture
def make_domain(monkeypatch):
    """make_domain(**members): the name of a fresh importable domain module holding
    the given members. ``__all__`` lists every member except ``opset_version`` and
    ``custom_op`` unless the members give it."""

    def make(**members):
        name = "qonnx_test_domain_" + uuid.uuid4().hex[:8]
        module = types.ModuleType(name)
        for key, value in members.items():
            setattr(module, key, value)
        if "__all__" not in members:
            module.__all__ = [key for key in members if key not in ("opset_version", "custom_op")]
        monkeypatch.setitem(sys.modules, name, module)
        return name

    return make


def node(domain, op_type="Thing"):
    return helper.make_node(op_type, ["x"], ["y"], domain=domain)


def fresh_domain():
    return "qonnx_test_unimportable_" + uuid.uuid4().hex[:8]


def test_an_alias_export_takes_its_version_from_the_class_name(make_domain):
    Thing_v3 = make_op("Thing_v3")
    domain = make_domain(Thing=Thing_v3)
    assert get_supported_versions(domain, "Thing") == [3]
    assert type(getCustomOp(node(domain))) is Thing_v3


def test_a_legacy_dict_alias_takes_its_version_from_the_class_name(make_domain):
    Thing_v3 = make_op("Thing_v3")
    domain = make_domain(custom_op={"Thing": Thing_v3}, __all__=[])
    assert get_supported_versions(domain, "Thing") == [3]


def test_all_is_read_before_the_legacy_dict(make_domain):
    listed, legacy = make_op("Thing"), make_op("Thing")
    domain = make_domain(Thing=listed, custom_op={"Thing": legacy})
    assert type(getCustomOp(node(domain))) is listed


def test_a_leading_zero_version_reads_as_upstream_reads_it(make_domain):
    Op_v02 = make_op("Op_v02")
    domain = make_domain(Op_v02=Op_v02)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert get_supported_versions(domain, "Op") == [2]


def test_duplicate_exports_keep_the_warning_and_the_later_class(make_domain):
    first, second = make_op("Thing"), make_op("Thing")
    domain = make_domain(Thing=first, Thing_v1=second)
    with pytest.warns(UserWarning, match=f"Multiple classes found for {domain}.Thing version 1"):
        found = getCustomOp(node(domain))
    assert type(found) is second


def test_a_lazy_module_resolves_the_wanted_op_without_its_other_exports(make_domain):
    Wanted = make_op("Wanted")
    domain = make_domain(Wanted=Wanted, __all__=["Wanted", "Unavailable"])
    module = sys.modules[domain]

    def unavailable(name):
        raise ImportError(f"{name} needs a package that is not installed")

    module.__getattr__ = unavailable
    assert type(getCustomOp(node(domain, "Wanted"))) is Wanted


def test_a_discovered_class_body_is_never_read(make_domain):
    class Thing(CopyOp):
        @property
        def op_type(self):
            return "Elsewhere"

        op_version = "not a version"

    domain = make_domain(Thing=Thing)
    assert get_supported_versions(domain, "Thing") == [1]
    assert type(getCustomOp(node(domain))) is Thing


def test_a_bare_multi_version_lookup_takes_the_highest_silently(make_domain):
    Thing, Thing_v2 = make_op("Thing"), make_op("Thing_v2")
    domain = make_domain(Thing=Thing, Thing_v2=Thing_v2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        found = getCustomOp(node(domain))
    assert type(found) is Thing_v2
    assert found.onnx_opset_version == 2


def test_registration_over_a_module_that_fails_to_import(monkeypatch):
    domain = "qonnx_test_broken_" + uuid.uuid4().hex[:8]
    finder_calls = []

    class Broken:
        def find_spec(self, name, path=None, target=None):
            if name == domain:
                finder_calls.append(name)
                raise RuntimeError("this module cannot be imported")
            return None

    monkeypatch.setattr(sys, "meta_path", [Broken(), *sys.meta_path])
    Thing = make_op("Thing")
    add_op_to_domain(domain, Thing)
    assert type(getCustomOp(node(domain))) is Thing
    assert get_supported_versions(domain, "Thing") == [1]
    assert is_custom_op(domain, "Thing")
    assert finder_calls == []


def test_add_op_to_domain_by_keyword():
    domain = fresh_domain()
    Thing = make_op("Thing")
    add_op_to_domain(domain=domain, op_class=Thing)
    assert type(getCustomOp(node(domain))) is Thing


def test_add_op_to_domain_without_keywords_keeps_the_name_rule():
    domain = fresh_domain()
    Stated = make_op("Stated_v2", op_type="Other", op_version=5)
    add_op_to_domain(domain, Stated)
    assert get_supported_versions(domain, "Stated") == [2]
    assert not is_custom_op(domain, "Other")


def test_add_op_to_domain_with_keywords_overrides_the_name():
    domain = fresh_domain()
    Impl = make_op("Impl_v7")
    add_op_to_domain(domain, Impl, op_type="Thing", op_version=3)
    assert get_supported_versions(domain, "Thing") == [3]
    add_op_to_domain(domain, Impl, op_type="Other")
    assert get_supported_versions(domain, "Other") == [7]
    add_op_to_domain(domain, Impl, op_version=4)
    assert get_supported_versions(domain, "Impl") == [4]


@pytest.mark.parametrize(
    "keywords",
    [{"op_type": ""}, {"op_type": 3}, {"op_version": 0}, {"op_version": -1}, {"op_version": True}, {"op_version": "2"}],
)
def test_add_op_to_domain_refuses_a_malformed_keyword(keywords):
    domain = fresh_domain()
    with pytest.raises(ValueError, match="op_type must be a nonempty string|op_version must be a positive integer"):
        add_op_to_domain(domain, make_op("Thing"), **keywords)
    assert not is_custom_op(domain, "Thing")


def test_add_op_to_domain_replaces_its_own_version_only():
    domain = fresh_domain()
    first, second, other = make_op("Thing"), make_op("Thing"), make_op("Thing_v2")
    add_op_to_domain(domain, first)
    add_op_to_domain(domain, other)
    add_op_to_domain(domain, second)
    assert type(getCustomOp(node(domain), onnx_opset_version=1)) is second
    assert type(getCustomOp(node(domain), onnx_opset_version=2)) is other


def test_op_identity_reads_a_stated_identity():
    assert op_identity(make_op("Impl", op_type="MatMul", op_version=6)) == ("MatMul", 6)
    assert op_identity(make_op("Impl_v3", op_type="MatMul")) == ("MatMul", 3)
    assert op_identity(make_op("Impl", op_version=2)) == ("Impl", 2)


def test_op_identity_ignores_an_inherited_identity():
    Base = make_op("Base", op_type="MatMul", op_version=6)
    Backend = type("Backend_v2", (Base,), {})
    assert op_identity(Backend) == ("Backend", 2)


def test_op_identity_without_a_statement_is_the_name_rule():
    assert op_identity(make_op("Thing")) == ("Thing", 1)
    assert op_identity(make_op("Thing_v4")) == ("Thing", 4)


def test_op_identity_does_not_read_a_field_that_is_given():
    class HasProperty(CopyOp):
        @property
        def op_type(self):
            return "Elsewhere"

    assert op_identity(HasProperty, op_type="Explicit", op_version=4) == ("Explicit", 4)
    assert op_identity(make_op("Impl_v3", op_type="MatMul"), op_version=5) == ("MatMul", 5)
    with pytest.raises(ValueError, match="^HasProperty[.]op_type"):
        op_identity(HasProperty, op_version=4)
    with pytest.raises(ValueError, match="^op_version must be a positive integer"):
        op_identity(make_op("Impl"), op_version=0)


@pytest.mark.parametrize("body", [{"op_type": ""}, {"op_type": None}, {"op_version": 0}, {"op_version": 2.0}])
def test_op_identity_refuses_a_malformed_statement(body):
    with pytest.raises(ValueError, match="^Impl[.]op_"):
        op_identity(make_op("Impl", **body))


def test_a_domain_opset_version_is_the_highest_since_version(make_domain):
    domain = make_domain(MatMul=make_op("MatMul"), MatMul_v6=make_op("MatMul_v6"), Thresholding=make_op("Thresholding"))
    assert get_domain_opset_version(domain) == 6


def test_a_domain_opset_version_includes_registered_ops(make_domain):
    domain = make_domain(Thing=make_op("Thing"))
    add_op_to_domain(domain, make_op("Other"), op_version=9)
    assert get_domain_opset_version(domain) == 9


def test_a_domain_module_may_state_its_opset_version(make_domain):
    domain = make_domain(Thing_v2=make_op("Thing_v2"), opset_version=4)
    assert get_domain_opset_version(domain) == 4


@pytest.mark.parametrize("stated", [1, "4", 2.0, True])
def test_a_stated_opset_version_below_an_op_or_not_an_integer_is_refused(make_domain, stated):
    domain = make_domain(Thing_v2=make_op("Thing_v2"), opset_version=stated)
    with pytest.raises(ValueError, match="opset_version"):
        get_domain_opset_version(domain)


def test_a_domain_with_registered_ops_and_no_module():
    domain = fresh_domain()
    add_op_to_domain(domain, make_op("Thing_v3"))
    assert get_domain_opset_version(domain) == 3


def test_a_domain_with_neither_module_nor_ops_is_unknown():
    with pytest.raises(KeyError):
        get_domain_opset_version(fresh_domain())


def test_qonnx_general_ops_are_unchanged():
    assert get_supported_versions("qonnx.custom_op.general", "Trunc") == [1, 2]
    names = dict(get_ops_in_domain("qonnx.custom_op.general"))
    assert names["Trunc"].__name__ == "Trunc_v2"
    assert names["MultiThreshold"].__name__ == "MultiThreshold"


@pytest.mark.parametrize("imported", [1, 5, 6, 9, None])
def test_a_model_resolves_each_op_at_its_domain_import(make_domain, imported):
    MatMul, MatMul_v6, Thresholding = make_op("MatMul"), make_op("MatMul_v6"), make_op("Thresholding")
    domain = make_domain(MatMul=MatMul, MatMul_v6=MatMul_v6, Thresholding=Thresholding)
    nodes = [
        helper.make_node("MatMul", ["x"], ["y"], domain=domain),
        helper.make_node("Thresholding", ["y"], ["z"], domain=domain),
    ]
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("x", 1, [1])],
        [helper.make_tensor_value_info("z", 1, [1])],
    )
    imports = [helper.make_opsetid("", 13)] + ([helper.make_opsetid(domain, imported)] if imported else [])
    model = ModelWrapper(qonnx_make_model(graph, opset_imports=imports))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        matmul = model.get_customop_wrapper(model.graph.node[0])
        thresholding = model.get_customop_wrapper(model.graph.node[1])
    assert type(matmul) is (MatMul_v6 if (imported or 1) >= 6 else MatMul)
    assert type(thresholding) is Thresholding
    assert bool(caught) == (imported is None)


@pytest.mark.parametrize("exported, listed", [("Wanted_v2", "Impl"), ("Wanted_v5", "Wanted_v5")])
def test_a_registered_op_is_listed_at_its_registered_version(make_domain, exported, listed):
    domain = make_domain(**{exported: make_op(exported)})
    add_op_to_domain(domain, make_op("Impl"), op_type="Wanted", op_version=3)
    assert get_supported_versions(domain, "Wanted") == [3]
    assert [(op_type, cls.__name__) for op_type, cls in get_ops_in_domain(domain)] == [("Wanted", listed)]
