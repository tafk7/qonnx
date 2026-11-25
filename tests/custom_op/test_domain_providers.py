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

"""Custom op domains provided through the entry-point group qonnx.custom_op.domains:
a provider declares its domain's ops, opset version and retired names; a lookup loads
a domain's providers once, refuses a retired name and serves declared ops before the
domain's module; a failing provider leaves nothing and fails again."""

import pytest

import importlib.util
import numpy as np
import sys
import threading
import time
import types
import uuid
from importlib.metadata import EntryPoint
from onnx import helper

import qonnx.custom_op.registry as registry
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.custom_op.base import CustomOp
from qonnx.custom_op.registry import (
    PROVIDERS,
    CustomOpRefused,
    CustomOpUnsupported,
    Registration,
    add_op_to_domain,
    get_domain_opset_version,
    get_ops_in_domain,
    get_supported_versions,
    getCustomOp,
    is_custom_op,
    retired,
)
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model
from qonnx.util.range_analysis import calc_monotonic_range


class CopyOp(CustomOp):
    """A custom op as a third party writes one: it copies its input to its output."""

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
    return type(name, (CopyOp,), dict(body))


def fresh_domain():
    return "qonnx_test_provided_" + uuid.uuid4().hex[:8]


def node(domain, op_type="Thing", name=""):
    return helper.make_node(op_type, ["x"], ["y"], domain=domain, name=name)


class Providers:
    """Entry points of the providers' group, installed for one test: each a real
    EntryPoint naming a function of a module made for the test."""

    def __init__(self, monkeypatch):
        self.module_name = "qonnx_test_provider_module_" + uuid.uuid4().hex[:8]
        self.module = types.ModuleType(self.module_name)
        monkeypatch.setitem(sys.modules, self.module_name, self.module)
        self.entries = {}
        real = registry.entry_points

        def entry_points(**params):
            if params.get("group") == PROVIDERS:
                return list(self.entries.get(params.get("name"), []))
            return real(**params)

        monkeypatch.setattr(registry, "entry_points", entry_points)

    def add(self, domain, function, attribute=None, distribution=None):
        attribute = attribute or "provide_" + uuid.uuid4().hex[:8]
        setattr(self.module, attribute, function)
        entry = EntryPoint(name=domain, value=f"{self.module_name}:{attribute}", group=PROVIDERS)
        if distribution is not None:
            entry = entry._for(types.SimpleNamespace(name=distribution))
        self.entries.setdefault(domain, []).append(entry)
        return attribute


@pytest.fixture
def providers(monkeypatch):
    return Providers(monkeypatch)


@pytest.fixture
def no_import(monkeypatch):
    """no_import(domain): importing the domain's module fails the test."""

    def forbid(domain):
        class Forbidden:
            def find_spec(self, name, path=None, target=None):
                if name == domain:
                    raise AssertionError(f"{domain} was imported")
                return None

        monkeypatch.setattr(sys, "meta_path", [Forbidden(), *sys.meta_path])

    return forbid


def test_a_provider_declares_its_ops_version_and_retired_names(providers, no_import):
    domain = fresh_domain()
    Impl = make_op("Impl", op_type="Thing", op_version=2)
    calls = []

    def provide(registration):
        calls.append(registration.domain)
        registration.version(3)
        registration.op(Impl)
        registration.retire("use Thing", op_type="OldThing")

    providers.add(domain, provide)
    no_import(domain)
    assert type(getCustomOp(node(domain))) is Impl
    assert get_supported_versions(domain, "Thing") == [2]
    assert get_domain_opset_version(domain) == 3
    assert is_custom_op(domain) and is_custom_op(domain, "Thing") and is_custom_op(domain, "OldThing")
    assert retired(domain, "OldThing") == "use Thing"
    assert retired(domain, "Thing") is None
    assert calls == [domain]


def test_a_domain_lists_its_declared_ops_and_not_its_retired_names(providers, monkeypatch):
    domain = fresh_domain()
    Thing, Old = make_op("Thing"), make_op("OldThing")
    module = types.ModuleType(domain)
    module.OldThing = Old
    module.__all__ = ["OldThing"]
    monkeypatch.setitem(sys.modules, domain, module)

    def provide(registration):
        registration.op(Thing)
        registration.retire("use Thing", op_type="OldThing")

    providers.add(domain, provide)
    assert dict(get_ops_in_domain(domain)) == {"Thing": Thing}


def test_a_retired_name_is_refused_at_lookup(providers):
    domain = fresh_domain()
    providers.add(domain, lambda registration: registration.retire("rebuild the model", op_type="OldThing"))
    message = f"old: stored identity {domain}.OldThing is retired; rebuild the model"
    with pytest.raises(KeyError, match=message):
        getCustomOp(node(domain, "OldThing", name="old"))
    with pytest.raises(KeyError, match="is retired; rebuild the model"):
        get_supported_versions(domain, "OldThing")


def test_a_retired_domain_refuses_every_lookup(providers):
    domain = fresh_domain()
    providers.add(domain, lambda registration: registration.retire("this domain is gone"))
    assert is_custom_op(domain)
    assert is_custom_op(domain, "Anything")
    assert retired(domain, "Anything") == "this domain is gone"
    assert retired(domain) == "this domain is gone"
    lookups = [
        lambda: getCustomOp(node(domain, "Anything")),
        lambda: get_supported_versions(domain, "Anything"),
        lambda: get_ops_in_domain(domain),
        lambda: get_domain_opset_version(domain),
    ]
    for lookup in lookups:
        with pytest.raises(KeyError, match="is retired; this domain is gone"):
            lookup()
    with pytest.raises(CustomOpRefused) as caught:
        get_ops_in_domain(domain)
    assert caught.value.findings[0].owner == "get_ops_in_domain"
    assert caught.value.findings[0].code == "domain-retired"


def test_loading_a_model_with_a_retired_name_refuses_nothing(providers):
    domain = fresh_domain()
    providers.add(domain, lambda registration: registration.retire("this domain is gone"))
    graph = helper.make_graph(
        [node(domain, "Anything")],
        "g",
        [helper.make_tensor_value_info("x", 1, [1])],
        [helper.make_tensor_value_info("y", 1, [1])],
    )
    imports = [helper.make_opsetid("", 13), helper.make_opsetid(domain, 1)]
    model = ModelWrapper(qonnx_make_model(graph, opset_imports=imports))
    assert [retired(n.domain, n.op_type) for n in model.graph.node] == ["this domain is gone"]
    with pytest.raises(KeyError, match="is retired"):
        model.get_customop_wrapper(model.graph.node[0])


def test_a_provider_loads_once(providers):
    domain = fresh_domain()
    calls = []

    def provide(registration):
        calls.append(1)
        registration.op(make_op("Thing"))

    providers.add(domain, provide)
    getCustomOp(node(domain))
    getCustomOp(node(domain))
    is_custom_op(domain)
    retired(domain, "Thing")
    assert calls == [1]


def test_declared_ops_come_before_the_module_and_the_module_is_the_fallback(providers, monkeypatch):
    domain = fresh_domain()
    Declared, Exported, Other = make_op("Thing"), make_op("Thing"), make_op("Other")
    module = types.ModuleType(domain)
    module.Thing, module.Other = Exported, Other
    module.__all__ = ["Thing", "Other"]
    monkeypatch.setitem(sys.modules, domain, module)
    providers.add(domain, lambda registration: registration.op(Declared))
    assert type(getCustomOp(node(domain, "Thing"))) is Declared
    assert type(getCustomOp(node(domain, "Other"))) is Other
    with pytest.raises(KeyError, match=f"Op 'Absent' not found in domain '{domain}'"):
        getCustomOp(node(domain, "Absent"))


def test_a_domain_without_a_provider_or_module_is_named_in_the_refusal(providers):
    domain = fresh_domain()
    providers.add(fresh_domain(), lambda registration: registration.op(make_op("Thing")))
    with pytest.raises(KeyError, match=f"Op 'Thing' not found in domain '{domain}'"):
        getCustomOp(node(domain))
    assert not is_custom_op(domain)


def test_a_run_time_registration_wins_for_its_version(providers):
    domain = fresh_domain()
    Early, Declared, Declared_v2, Late = make_op("Thing"), make_op("Thing"), make_op("Thing_v2"), make_op("Thing")

    def provide(registration):
        registration.op(Declared)
        registration.op(Declared_v2)

    providers.add(domain, provide)
    add_op_to_domain(domain, Early)
    assert type(getCustomOp(node(domain), onnx_opset_version=1)) is Early
    assert type(getCustomOp(node(domain), onnx_opset_version=2)) is Declared_v2
    add_op_to_domain(domain, Late)
    assert type(getCustomOp(node(domain), onnx_opset_version=1)) is Late


def test_a_failing_provider_fails_again_and_leaves_nothing(providers, monkeypatch):
    domain, other = fresh_domain(), fresh_domain()
    calls = []
    imported_name = "qonnx_test_registering_module_" + uuid.uuid4().hex[:8]

    class Loader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            # a module whose body registers an op into the loading domain
            add_op_to_domain(domain, make_op("Imported"))

    class RegisteringFinder:
        def find_spec(self, name, path=None, target=None):
            return importlib.util.spec_from_loader(name, Loader()) if name == imported_name else None

    def provide(registration):
        calls.append(1)
        registration.op(make_op("Declared"))
        registration.version(4)
        registration.retire("gone", op_type="Gone")
        add_op_to_domain(domain, make_op("Direct"))
        importlib.import_module(imported_name)
        add_op_to_domain(other, make_op("Elsewhere"))
        raise RuntimeError("the provider is broken")

    monkeypatch.setattr(sys, "meta_path", [RegisteringFinder(), *sys.meta_path])
    providers.add(domain, provide)
    with pytest.raises(RuntimeError, match="^the provider is broken$"):
        getCustomOp(node(domain, "Declared"))
    with pytest.raises(RuntimeError, match="^the provider is broken$"):
        getCustomOp(node(domain, "Direct"))
    with pytest.raises(RuntimeError, match="^the provider is broken$"):
        retired(domain, "Gone")
    assert len(calls) == 3
    assert domain not in registry._PROVIDED
    assert domain not in registry._OP_REGISTRY
    assert domain not in registry._DECLARED
    assert not any(d == domain for d, _ in registry._RETIRED)
    # a registration for another domain made during the failed load stays
    assert get_supported_versions(other, "Elsewhere") == [1]


def test_a_provider_fixed_in_process_is_used_on_the_next_lookup(providers):
    domain = fresh_domain()
    Thing = make_op("Thing")

    def broken(registration):
        raise ImportError("a dependency is missing")

    attribute = providers.add(domain, broken)
    with pytest.raises(ImportError, match="a dependency is missing"):
        getCustomOp(node(domain))
    setattr(providers.module, attribute, lambda registration: registration.op(Thing))
    assert type(getCustomOp(node(domain))) is Thing


def test_a_provider_looking_up_its_own_domain_is_refused(providers):
    domain = fresh_domain()

    def provide(registration):
        registration.op(make_op("Thing"))
        is_custom_op(domain, "Thing")

    providers.add(domain, provide)
    with pytest.raises(RuntimeError, match=f"a provider of {domain} looked up {domain} while loading"):
        getCustomOp(node(domain))
    assert domain not in registry._PROVIDED
    assert domain not in registry._OP_REGISTRY


def run_two_threads(domain, entered, second_started):
    """Look the domain up on two threads, the second once the first is inside the
    provider; return what each got (an op or an exception)."""
    results = [None, None]

    def first():
        try:
            results[0] = getCustomOp(node(domain))
        except Exception as error:
            results[0] = error

    def second():
        assert entered.wait(10)
        second_started.set()
        try:
            results[1] = getCustomOp(node(domain))
        except Exception as error:
            results[1] = error

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert not any(thread.is_alive() for thread in threads)
    return results


def test_two_threads_meeting_a_failing_provider_each_get_the_failure(providers):
    domain = fresh_domain()
    entered, second_started = threading.Event(), threading.Event()
    calls = []

    def provide(registration):
        calls.append(1)
        registration.op(make_op("Thing"))
        if not entered.is_set():
            entered.set()
            assert second_started.wait(10)
            time.sleep(0.05)  # the second thread now waits for the registry
        assert domain not in registry._OP_REGISTRY
        assert domain not in registry._PROVIDED
        raise RuntimeError("the provider is broken")

    providers.add(domain, provide)
    results = run_two_threads(domain, entered, second_started)
    assert all(isinstance(result, RuntimeError) and str(result) == "the provider is broken" for result in results)
    assert len(calls) == 2
    assert domain not in registry._PROVIDED
    assert domain not in registry._OP_REGISTRY


def test_two_threads_meeting_a_working_provider_get_the_same_class(providers):
    domain = fresh_domain()
    entered, second_started = threading.Event(), threading.Event()
    Thing = make_op("Thing")
    calls = []

    def provide(registration):
        calls.append(1)
        registration.op(Thing)
        entered.set()
        assert second_started.wait(10)
        time.sleep(0.05)  # the second thread now waits for the registry
        # declared, and not yet in the registry
        assert domain not in registry._OP_REGISTRY
        assert domain not in registry._PROVIDED

    providers.add(domain, provide)
    results = run_two_threads(domain, entered, second_started)
    assert [type(result) for result in results] == [Thing, Thing]
    assert calls == [1]


def test_providers_load_in_a_fixed_order(providers):
    domain = fresh_domain()
    order = []
    providers.add(domain, lambda registration: order.append("b/z"), attribute="z", distribution="b")
    providers.add(domain, lambda registration: order.append("a/y"), attribute="y", distribution="a")
    providers.add(domain, lambda registration: order.append("b/x"), attribute="x", distribution="b")
    is_custom_op(domain)
    assert order == ["a/y", "b/x", "b/z"]


def test_two_providers_may_agree(providers):
    domain = fresh_domain()
    Thing = make_op("Thing")

    def provide(registration):
        registration.version(2)
        registration.op(Thing)
        registration.retire("gone", op_type="Gone")

    providers.add(domain, provide, attribute="a")
    providers.add(domain, provide, attribute="b")
    assert get_domain_opset_version(domain) == 2
    assert type(getCustomOp(node(domain))) is Thing
    assert retired(domain, "Gone") == "gone"


@pytest.mark.parametrize(
    "second, message",
    [
        (lambda registration: registration.version(3), "opset versions 2 and 3"),
        (lambda registration: registration.op(make_op("Thing")), "declared as both"),
        (lambda registration: registration.retire("other", op_type="Gone"), "retired for two reasons"),
    ],
)
def test_contradicting_providers_are_refused(providers, second, message):
    domain = fresh_domain()

    def first(registration):
        registration.version(2)
        registration.op(make_op("Thing"))
        registration.retire("gone", op_type="Gone")

    providers.add(domain, first, attribute="a")
    providers.add(domain, second, attribute="b")
    with pytest.raises(ValueError, match=message):
        is_custom_op(domain)
    assert domain not in registry._PROVIDED


@pytest.mark.parametrize("domain", ["finn", "finn.custom_op.general"])
def test_a_renamed_domain_cannot_be_retired(providers, domain):
    providers.add(domain, lambda registration: registration.retire("gone"))
    with pytest.raises(ValueError, match="renamed when a model is loaded"):
        retired(domain)
    assert domain not in registry._PROVIDED


def test_a_declared_version_below_an_op_is_refused(providers):
    domain = fresh_domain()

    def provide(registration):
        registration.version(1)
        registration.op(make_op("Thing_v3"))

    providers.add(domain, provide)
    with pytest.raises(ValueError, match="declared opset version 1 is below"):
        get_domain_opset_version(domain)


def test_registration_reads_a_stated_identity_and_not_an_inherited_one():
    registration = Registration("d")
    Base = make_op("Base", op_type="MatMul", op_version=6)
    registration.op(Base)
    registration.op(type("Backend_v2", (Base,), {}))
    registration.op(make_op("Impl"), op_type="Other", op_version=4)
    assert {t: sorted(v) for t, v in registration._ops.items()} == {"MatMul": [6], "Backend": [2], "Other": [4]}


@pytest.mark.parametrize(
    "declare",
    [
        lambda registration: registration.op(make_op("Impl", op_version=0)),
        lambda registration: registration.op(make_op("Impl", op_type="")),
        lambda registration: registration.op(make_op("Impl"), op_version="2"),
        lambda registration: registration.op(object),
        lambda registration: registration.version(0),
        lambda registration: registration.retire(""),
    ],
)
def test_registration_refuses_a_malformed_declaration(declare):
    with pytest.raises(ValueError):
        declare(Registration("d"))


def test_a_domain_without_providers_is_looked_up_as_before():
    Thing = make_op("Thing")
    domain = fresh_domain()
    add_op_to_domain(domain, Thing)
    assert type(getCustomOp(node(domain))) is Thing
    assert retired(domain, "Thing") is None
    assert retired("", "Relu") is None
    general = dict(get_ops_in_domain("qonnx.custom_op.general"))
    assert general["Trunc"].__name__ == "Trunc_v2"
    assert get_supported_versions("qonnx.custom_op.general", "Trunc") == [1, 2]


def retired_models(domain, op_type):
    """The same node, op_type of domain, in a model to run and one to fold."""

    def model(constant):
        x = helper.make_tensor_value_info("x", 1, [1, 2])
        graph = helper.make_graph(
            [helper.make_node(op_type, ["x"], ["y"], domain=domain), helper.make_node("Relu", ["y"], ["out"])],
            "g",
            [] if constant else [x],
            [helper.make_tensor_value_info("out", 1, [1, 2])],
            initializer=[helper.make_tensor("x", 1, [1, 2], [1.0, 2.0])] if constant else [],
            value_info=[helper.make_tensor_value_info("y", 1, [1, 2])],
        )
        imports = [helper.make_opsetid("", 13), helper.make_opsetid(domain, 1)]
        return ModelWrapper(qonnx_make_model(graph, opset_imports=imports))

    return model(False), model(True)


def through_every_route(domain, op_type):
    """What InferShapes, InferDataTypes, FoldConstants, calc_monotonic_range and
    execute_onnx raise for the node: (type, message) each."""
    running, folding = retired_models(domain, op_type)
    routes = {
        "InferShapes": lambda: running.transform(InferShapes()),
        "InferDataTypes": lambda: running.transform(InferDataTypes()),
        "FoldConstants": lambda: folding.transform(FoldConstants(exclude_op_types=[])),
        "calc_monotonic_range": lambda: calc_monotonic_range(running.graph.node[0], running, {"x": (0.0, 1.0)}),
        "execute_onnx": lambda: execute_onnx(running, {"x": np.ones((1, 2), dtype=np.float32)}),
    }
    raised = {}
    for name, route in routes.items():
        with pytest.raises(Exception) as caught:
            route()
        raised[name] = (caught.type, str(caught.value))
    return raised


@pytest.mark.parametrize("whole_domain", [False, True])
def test_a_retired_name_is_refused_with_its_reason_by_every_pass_and_execution(providers, whole_domain):
    domain = fresh_domain()

    def provide(registration):
        registration.op(make_op("Thing"))
        registration.retire("rebuild the model with Thing", op_type=None if whole_domain else "OldThing")

    providers.add(domain, provide)
    for name, (kind, message) in through_every_route(domain, "OldThing").items():
        assert kind is CustomOpRefused and issubclass(kind, KeyError), name
        assert f"stored identity {domain}" in message and "rebuild the model with Thing" in message, name


def test_an_unknown_op_is_refused_by_every_pass_and_execution_as_before():
    domain = fresh_domain()
    add_op_to_domain(domain, make_op("Thing"))
    raised = through_every_route(domain, "Absent")
    for name in ("InferShapes", "InferDataTypes"):
        kind, message = raised[name]
        assert kind is CustomOpUnsupported and issubclass(kind, Exception)
        assert not issubclass(kind, KeyError)
        assert message == f"{name}: op-unsupported: Custom op_type Absent is currently not supported."
    # execution sends an op its domain does not have to ONNX Runtime, which refuses it
    for name in ("FoldConstants", "calc_monotonic_range", "execute_onnx"):
        assert raised[name][0] is not KeyError and "INVALID_GRAPH" in raised[name][1], name


def test_an_explicit_identity_is_not_read_from_the_class():
    class HasProperty(CopyOp):
        @property
        def op_type(self):
            return "Elsewhere"

    registration = Registration("d")
    registration.op(HasProperty, op_type="Explicit", op_version=4)
    assert registration._ops == {"Explicit": {4: HasProperty}}


def test_a_registration_made_during_a_load_keeps_its_name_rule(providers):
    domain = fresh_domain()

    class OldProperty(CopyOp):
        @property
        def op_type(self):
            return "Elsewhere"

    def provide(registration):
        registration.op(make_op("Thing"))
        add_op_to_domain(domain, OldProperty)

    providers.add(domain, provide)
    assert type(getCustomOp(node(domain, "OldProperty"))) is OldProperty


def test_a_provider_op_is_listed_as_it_is_looked_up(providers, monkeypatch):
    domain = fresh_domain()
    Implementation, Wanted_v5 = make_op("Implementation"), make_op("Wanted_v5")
    module = types.ModuleType(domain)
    module.Wanted_v5 = Wanted_v5
    module.__all__ = ["Wanted_v5"]
    monkeypatch.setitem(sys.modules, domain, module)

    def provide(registration):
        registration.version(3)
        registration.op(Implementation, op_type="Wanted", op_version=3)

    providers.add(domain, provide)
    assert type(getCustomOp(node(domain, "Wanted"))) is Implementation
    assert dict(get_ops_in_domain(domain)) == {"Wanted": Implementation}
    assert get_supported_versions(domain, "Wanted") == [3]
    assert get_domain_opset_version(domain) == 3


def test_a_provider_op_without_a_declared_version_ignores_the_module_s_export(providers, monkeypatch):
    domain = fresh_domain()
    module = types.ModuleType(domain)
    module.Wanted_v5 = make_op("Wanted_v5")
    module.__all__ = ["Wanted_v5"]
    monkeypatch.setitem(sys.modules, domain, module)
    providers.add(domain, lambda registration: registration.op(make_op("Implementation"), op_type="Wanted", op_version=3))
    assert get_domain_opset_version(domain) == 3


def test_a_run_time_registration_above_a_provider_op_is_listed(providers):
    domain = fresh_domain()
    Declared, Later = make_op("Thing"), make_op("Thing_v4")
    providers.add(domain, lambda registration: registration.op(Declared))
    getCustomOp(node(domain))
    add_op_to_domain(domain, Later)
    assert dict(get_ops_in_domain(domain)) == {"Thing": Later}


def test_a_registration_without_a_provider_is_listed_as_before(monkeypatch):
    domain = fresh_domain()
    Registered, Wanted_v5 = make_op("Wanted"), make_op("Wanted_v5")
    module = types.ModuleType(domain)
    module.Wanted_v5 = Wanted_v5
    module.__all__ = ["Wanted_v5"]
    monkeypatch.setitem(sys.modules, domain, module)
    add_op_to_domain(domain, Registered)
    assert dict(get_ops_in_domain(domain)) == {"Wanted": Wanted_v5}
    assert type(getCustomOp(node(domain, "Wanted"))) is Registered


def write_registering_module(tmp_path, monkeypatch, domain, op_name):
    """A module file whose body registers op_name into domain, on sys.path."""
    name = "qonnx_test_registering_" + uuid.uuid4().hex[:8]
    (tmp_path / (name + ".py")).write_text(
        "from onnx import helper\n"
        "from qonnx.custom_op.base import CustomOp\n"
        "from qonnx.custom_op.registry import add_op_to_domain\n\n\n"
        f"class {op_name}(CustomOp):\n"
        "    def get_nodeattr_types(self):\n"
        "        return {}\n\n"
        "    def make_shape_compatible_op(self, model):\n"
        "        return helper.make_node('Identity', [self.onnx_node.input[0]], [self.onnx_node.output[0]])\n\n"
        "    def infer_node_datatype(self, model):\n"
        "        pass\n\n"
        "    def execute_node(self, context, graph):\n"
        "        pass\n\n"
        "    def verify_node(self):\n"
        "        pass\n\n\n"
        f"add_op_to_domain({domain!r}, {op_name})\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    return name


def test_a_registration_from_an_imported_module_survives_a_failed_load(providers, tmp_path, monkeypatch):
    domain = fresh_domain()
    module_name = write_registering_module(tmp_path, monkeypatch, domain, "Imported")
    calls, broken = [], [True]

    def provide(registration):
        calls.append(1)
        importlib.import_module(module_name)
        if broken[0]:
            raise RuntimeError("a transient failure")

    providers.add(domain, provide)
    with pytest.raises(RuntimeError, match="a transient failure"):
        getCustomOp(node(domain, "Imported"))
    assert module_name in sys.modules
    assert domain not in registry._OP_REGISTRY and domain not in registry._PROVIDED
    broken[0] = False
    assert type(getCustomOp(node(domain, "Imported"))).__name__ == "Imported"
    assert type(getCustomOp(node(domain, "Imported"))).__module__ == module_name
    assert calls == [1, 1]


def test_a_held_registration_wins_its_version_over_a_provider(providers):
    domain = fresh_domain()
    Declared, Registered = make_op("Thing"), make_op("Thing")

    def provide(registration):
        registration.op(Declared)
        add_op_to_domain(domain, Registered)

    providers.add(domain, provide)
    assert type(getCustomOp(node(domain))) is Registered


def test_a_later_registration_replaces_a_held_one(providers, tmp_path, monkeypatch):
    domain = fresh_domain()
    module_name = write_registering_module(tmp_path, monkeypatch, domain, "Thing")
    broken = [True]

    def provide(registration):
        importlib.import_module(module_name)
        if broken[0]:
            raise RuntimeError("a transient failure")

    providers.add(domain, provide)
    with pytest.raises(RuntimeError):
        getCustomOp(node(domain))
    Later = make_op("Thing")
    add_op_to_domain(domain, Later)
    broken[0] = False
    assert type(getCustomOp(node(domain))) is Later
