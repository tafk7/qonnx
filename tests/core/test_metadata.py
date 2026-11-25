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

from __future__ import annotations

import pytest

import copy
from enum import Enum
from onnx import GraphProto, ModelProto, NodeProto, StringStringEntryProto, TensorProto, helper
from pathlib import Path
from typing import Any, Sequence

from qonnx.core import metadata
from qonnx.core.metadata import JSON, MetadataError, Namespace
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name, qonnx_make_model


class Block(Enum):
    SMALL = 1
    LARGE = 2


def make_namespace(version: int = 1) -> tuple[Namespace, dict[str, metadata.Key[Any]]]:
    ns = Namespace("test.platform", version=version)
    keys = dict(
        block=ns.key("block", Block),
        period=ns.key("period", float, check=lambda v: v > 0, expect="a period > 0"),
        part=ns.key("part", str),
        ports=ns.key("ports", int),
        fast=ns.key("fast", bool),
        table=ns.key("table", JSON),
    )
    return ns, keys


def entries(graph: GraphProto) -> dict[str, str]:
    return {prop.key: prop.value for prop in graph.metadata_props}


def store(graph: GraphProto, **raw: str) -> None:
    for key, value in raw.items():
        graph.metadata_props.append(StringStringEntryProto(key=key, value=value))


def test_keys_round_trip_as_one_entry_each_with_the_namespace_version() -> None:
    ns, k = make_namespace()
    graph = GraphProto()
    values = dict(block=Block.LARGE, period=2.5, part="xc7z020", ports=4, fast=False, table={"b": [1, 2.0], "a": None})
    for name, value in values.items():
        metadata.write(graph.metadata_props, k[name], value)
    assert entries(graph) == {
        "test.platform/@version": "1",
        "test.platform/block": "LARGE",
        "test.platform/period": "2.5",
        "test.platform/part": "xc7z020",
        "test.platform/ports": "4",
        "test.platform/fast": "false",
        "test.platform/table": '{"a":null,"b":[1,2.0]}',
    }
    assert metadata.read(graph.metadata_props, ns) == values
    assert list(metadata.read(graph.metadata_props, ns)) == list(ns.keys)


def test_an_int_is_stored_as_a_float_where_a_float_is_declared() -> None:
    ns, k = make_namespace()
    graph = GraphProto()
    metadata.write(graph.metadata_props, k["period"], 5)
    assert entries(graph)["test.platform/period"] == "5.0"
    assert metadata.read(graph.metadata_props, ns)["period"] == 5.0


def test_writing_again_replaces_the_entry() -> None:
    ns, k = make_namespace()
    graph = GraphProto()
    metadata.write(graph.metadata_props, k["ports"], 1)
    metadata.write(graph.metadata_props, k["ports"], 2)
    assert [p.key for p in graph.metadata_props] == ["test.platform/@version", "test.platform/ports"]
    assert metadata.read(graph.metadata_props, ns) == {"ports": 2}


def test_an_absent_namespace_reads_empty() -> None:
    ns, _ = make_namespace()
    graph = GraphProto()
    store(graph, other="x", **{"test.platformx/@version": "1"})
    assert metadata.read(graph.metadata_props, ns) == {}


@pytest.mark.parametrize(
    "name, value",
    [
        ("block", "LARGE"),
        ("block", 2),
        ("period", "5.0"),
        ("period", 0.0),
        ("period", True),
        ("part", 3),
        ("ports", 2.0),
        ("ports", True),
        ("fast", 1),
        ("table", (1, 2)),
        ("table", {1: "a"}),
        ("table", float("nan")),
        ("table", object()),
    ],
)
def test_a_value_of_the_wrong_type_is_refused_on_writing(name: str, value: object) -> None:
    ns, k = make_namespace()
    graph = GraphProto()
    with pytest.raises(MetadataError, match=f"test.platform/{name}: cannot store"):
        metadata.write(graph.metadata_props, k[name], value)
    assert entries(graph) == {}


@pytest.mark.parametrize(
    "name, text, expectation",
    [
        ("block", "MEDIUM", "a Block (SMALL, LARGE)"),
        ("period", "fast", "a float, a period > 0"),
        ("period", "-1.0", "a float, a period > 0"),
        ("period", " 5.0", "a float, a period > 0"),
        ("ports", "4.0", "an int"),
        ("fast", "True", "a bool"),
        ("table", "{'a': 1}", "a JSON value"),
        ("table", "NaN", "a JSON value"),
    ],
)
def test_a_malformed_stored_value_is_refused_naming_entry_text_and_expectation(
    name: str, text: str, expectation: str
) -> None:
    ns, _ = make_namespace()
    graph = GraphProto()
    store(graph, **{"test.platform/@version": "1", f"test.platform/{name}": text})
    with pytest.raises(MetadataError) as refusal:
        metadata.read(graph.metadata_props, ns)
    assert str(refusal.value).startswith(f"test.platform/{name}: stored {text!r} is not {expectation}")


def test_an_undeclared_stored_key_is_refused() -> None:
    ns, _ = make_namespace()
    graph = GraphProto()
    store(graph, **{"test.platform/@version": "1", "test.platform/clock": "5"})
    with pytest.raises(MetadataError, match=r"stored keys \['clock'\] are not declared"):
        metadata.read(graph.metadata_props, ns)


def test_keys_without_a_version_and_malformed_versions_are_refused() -> None:
    ns, _ = make_namespace()
    graph = GraphProto()
    store(graph, **{"test.platform/part": "x"})
    with pytest.raises(MetadataError, match="without test.platform/@version"):
        metadata.read(graph.metadata_props, ns)
    graph = GraphProto()
    store(graph, **{"test.platform/@version": "v1"})
    with pytest.raises(MetadataError, match="is not a positive int"):
        metadata.read(graph.metadata_props, ns)


def test_an_entry_stored_twice_is_refused() -> None:
    ns, _ = make_namespace()
    graph = GraphProto()
    store(graph, **{"test.platform/@version": "1", "test.platform/part": "x"})
    store(graph, **{"test.platform/part": "y"})
    with pytest.raises(MetadataError, match="test.platform/part: stored twice"):
        metadata.read(graph.metadata_props, ns)


def test_a_version_the_reader_cannot_upgrade_from_is_refused() -> None:
    ns, _ = make_namespace(version=2)
    graph = GraphProto()
    store(graph, **{"test.platform/@version": "1", "test.platform/part": "x"})
    with pytest.raises(
        MetadataError, match=r"stored at version 1, which a reader of version 2 cannot upgrade from \(it has no upgrade\)"
    ):
        metadata.read(graph.metadata_props, ns)
    newer = GraphProto()
    store(newer, **{"test.platform/@version": "3", "test.platform/part": "x"})
    with pytest.raises(MetadataError, match="stored at version 3"):
        metadata.read(newer.metadata_props, ns)


def make_v3() -> tuple[Namespace, dict[str, metadata.Key[Any]]]:
    """Version 1 had `clock_mhz` (an int); version 2 replaced it by `period` (ns);
    version 3 renamed `name` to `part`."""
    ns, k = make_namespace(version=3)
    ns.upgrade(1, lambda e: {**{n: t for n, t in e.items() if n != "clock_mhz"}, "period": repr(1000 / int(e["clock_mhz"]))})
    ns.upgrade(2, lambda e: {("part" if n == "name" else n): t for n, t in e.items()})
    return ns, k


def test_a_reader_upgrades_an_earlier_version_through_each_step() -> None:
    ns, _ = make_v3()
    graph = GraphProto()
    store(graph, **{"test.platform/@version": "1", "test.platform/clock_mhz": "200", "test.platform/name": "xc7z020"})
    assert metadata.read(graph.metadata_props, ns) == {"period": 5.0, "part": "xc7z020"}
    assert entries(graph)["test.platform/@version"] == "1"  # reading changes nothing


def test_a_writer_rewrites_an_earlier_version_at_the_current_one() -> None:
    ns, k = make_v3()
    graph = GraphProto()
    store(graph, other="kept", **{"test.platform/@version": "2", "test.platform/name": "xc7z020"})
    metadata.write(graph.metadata_props, k["fast"], True)
    assert entries(graph) == {
        "other": "kept",
        "test.platform/@version": "3",
        "test.platform/part": "xc7z020",
        "test.platform/fast": "true",
    }


def test_upgrades_are_declared_once_from_an_earlier_version() -> None:
    ns, _ = make_namespace(version=2)
    with pytest.raises(ValueError, match="cannot upgrade from 2"):
        ns.upgrade(2, dict)
    ns.upgrade(1, dict)
    with pytest.raises(ValueError, match="twice"):
        ns.upgrade(1, dict)


def test_delete_removes_the_entry_and_the_version_with_the_last_key() -> None:
    ns, k = make_namespace()
    graph = GraphProto()
    store(graph, other="kept")
    metadata.write(graph.metadata_props, k["part"], "x")
    metadata.write(graph.metadata_props, k["ports"], 2)
    metadata.delete(graph.metadata_props, k["part"])
    assert metadata.read(graph.metadata_props, ns) == {"ports": 2}
    metadata.delete(graph.metadata_props, k["part"])  # absent: nothing to do
    metadata.delete(graph.metadata_props, k["ports"])
    assert entries(graph) == {"other": "kept"}


def test_declarations_are_checked() -> None:
    with pytest.raises(ValueError, match="namespace name"):
        Namespace("a/b")
    with pytest.raises(ValueError, match="positive int"):
        Namespace("a", version=0)
    ns = Namespace("a")
    ns.key("k", str)
    with pytest.raises(ValueError, match="twice"):
        ns.key("k", int)
    with pytest.raises(ValueError, match="key name"):
        ns.key("@version", int)
    with pytest.raises(TypeError, match="a metadata key's type"):
        ns.key("l", list)


def make_model() -> ModelWrapper:
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "g", [inp], [out])
    return ModelWrapper(qonnx_make_model(graph))


def test_modelwrapper_reads_and_writes_typed_keys_on_the_graph(tmp_path: Path) -> None:
    ns, k = make_namespace()
    model = make_model()
    assert model.get_metadata(k["period"]) is None and model.metadata_namespace(ns) == {}
    model.set_metadata(k["period"], 5.0)
    model.set_metadata(k["block"], Block.SMALL)
    model.set_metadata_prop("untyped", "kept")
    assert model.get_metadata(k["period"]) == 5.0
    assert model.metadata_namespace(ns) == {"block": Block.SMALL, "period": 5.0}
    assert model.get_metadata_prop("test.platform/period") == "5.0"
    assert len(model.model.metadata_props) == 0  # the ModelProto's stay untouched
    model.save(tmp_path / "m.onnx")
    loaded = ModelWrapper(str(tmp_path / "m.onnx"))
    assert loaded.metadata_namespace(ns) == {"block": Block.SMALL, "period": 5.0}
    assert copy.deepcopy(loaded).get_metadata(k["block"]) is Block.SMALL
    transformed = loaded.transform(InferShapes()).transform(GiveUniqueNodeNames())
    assert transformed.get_metadata(k["period"]) == 5.0
    loaded.delete_metadata(k["period"])
    assert loaded.metadata_namespace(ns) == {"block": Block.SMALL}
    assert loaded.get_metadata_prop("untyped") == "kept"


def test_modelwrapper_refuses_what_the_namespace_refuses() -> None:
    ns, k = make_namespace()
    model = make_model()
    with pytest.raises(MetadataError, match="cannot store 'fast'"):
        model.set_metadata(k["period"], "fast")
    model.set_metadata_prop("test.platform/@version", "1")
    model.set_metadata_prop("test.platform/period", "fast")
    with pytest.raises(MetadataError, match="test.platform/period: stored 'fast'"):
        model.get_metadata(k["part"])  # a namespace is read whole


SITE = Namespace("test.site", version=1, inherit=True)
CLOCK = SITE.key("clock_ns", float)
NAME = SITE.key("name", str)
LOCAL = Namespace("test.local", version=1)
NOTE = LOCAL.key("note", str)


def make_bodies_model(then_clock: float | None = None, nested: bool = False) -> ModelWrapper:
    """An outer graph with one node holding two bodies, `then` stating its own clock
    when then_clock is given, `else` stating none (and holding a body of its own
    when nested)."""

    def body(name: str, inner: Sequence[NodeProto] = ()) -> GraphProto:
        graph = helper.make_graph(list(inner), name, [], [])
        return graph

    inner = []
    if nested:
        inner = [helper.make_node("Holder", [], [], name="inner_holder", body=body("inner"))]
    then_branch, else_branch = body("then"), body("else", inner)
    if then_clock is not None:
        metadata.write(then_branch.metadata_props, CLOCK, then_clock)
    holder = helper.make_node("Holder", [], [], name="holder", then_branch=then_branch, else_branch=else_branch)
    model = ModelWrapper(qonnx_make_model(helper.make_graph([holder], "outer", [], [])))
    model.set_metadata(CLOCK, 5.0)
    model.set_metadata(NAME, "board")
    model.set_metadata(NOTE, "outer only")
    return model


def bodies(model: ModelWrapper) -> dict[str, ModelWrapper]:
    holder = model.graph.node[0]
    return {attr.name: model.make_subgraph_modelwrapper(attr.g) for attr in holder.attribute}


def test_a_body_reads_an_inheriting_key_it_does_not_state_from_its_parent() -> None:
    model = make_bodies_model(then_clock=4.0)
    b = bodies(model)
    assert b["then_branch"].get_metadata(CLOCK) == 4.0  # its own
    assert b["else_branch"].get_metadata(CLOCK) == 5.0  # inherited
    assert b["then_branch"].metadata_namespace(SITE) == {"clock_ns": 4.0, "name": "board"}
    assert b["else_branch"].get_metadata(NOTE) is None  # a namespace that does not inherit
    model.set_metadata(CLOCK, 6.0)
    assert b["else_branch"].get_metadata(CLOCK) == 6.0  # one source while the body is inside its parent
    b["else_branch"].delete_metadata(CLOCK)  # deletes nothing of the parent's
    assert b["else_branch"].get_metadata(CLOCK) == 6.0
    # a body of a body reads through both
    nested = make_bodies_model(nested=True)
    inner_holder = bodies(nested)["else_branch"]
    inner = inner_holder.make_subgraph_modelwrapper(inner_holder.graph.node[0].attribute[0].g)
    assert inner.get_metadata(CLOCK) == 5.0


def test_a_body_alone_reads_only_its_own() -> None:
    model = make_bodies_model(then_clock=4.0)
    else_graph = get_by_name(model.graph.node[0].attribute, "else_branch").g
    assert ModelWrapper(qonnx_make_model(else_graph)).get_metadata(CLOCK) is None


def test_a_body_stays_a_body_through_deepcopy_and_transformations() -> None:
    model = make_bodies_model()
    body = bodies(model)["else_branch"]
    clone = copy.deepcopy(body)
    model.set_metadata(CLOCK, 7.0)
    assert clone.get_metadata(CLOCK) == 7.0  # the parent is shared, not copied

    class Rewrap(Transformation):
        """Returns a new wrapper first, then records what each graph reads."""

        def __init__(self) -> None:
            super().__init__()
            self.seen = {}

        def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
            name = model.graph.name
            if name not in self.seen:
                self.seen[name] = None
                return ModelWrapper(model.model), True
            self.seen[name] = model.get_metadata(CLOCK)
            return model, False

    rewrap = Rewrap()
    model.transform(rewrap, apply_to_subgraphs=True)
    assert rewrap.seen == {"outer": 7.0, "then": 7.0, "else": 7.0}


def test_inherit_metadata_copies_what_a_standalone_body_must_carry(tmp_path: Path) -> None:
    model = make_bodies_model(then_clock=4.0)
    b = bodies(model)
    for name, body in b.items():
        body.inherit_metadata(SITE)
        body.save(tmp_path / f"{name}.onnx")
    then_alone = ModelWrapper(str(tmp_path / "then_branch.onnx"))
    else_alone = ModelWrapper(str(tmp_path / "else_branch.onnx"))
    assert then_alone.metadata_namespace(SITE) == {"clock_ns": 4.0, "name": "board"}  # its own kept
    assert else_alone.metadata_namespace(SITE) == {"clock_ns": 5.0, "name": "board"}
    assert else_alone.get_metadata(NOTE) is None
    # a model built apart from its parent names the parent
    extracted = ModelWrapper(qonnx_make_model(helper.make_graph([], "extracted", [], [])))
    extracted.inherit_metadata(SITE, parent=model)
    assert extracted.get_metadata(NAME) == "board"
    with pytest.raises(ValueError, match="does not inherit"):
        extracted.inherit_metadata(LOCAL, parent=model)
    with pytest.raises(ValueError, match="not opened as a subgraph body"):
        extracted.inherit_metadata(SITE)


class SlottedModel(ModelWrapper):
    __slots__ = ("payload",)


def test_plain_deepcopy_preserves_bytes_and_attributes() -> None:
    model = make_model()
    model.extra = {"labels": ["source"]}
    clone = copy.deepcopy(model)
    assert type(clone) is ModelWrapper and clone is not model
    assert clone.model.SerializeToString() == model.model.SerializeToString()
    assert clone.__dict__.keys() == model.__dict__.keys()
    assert clone.fix_float64 == model.fix_float64 and clone._parent is None
    assert clone.extra == model.extra and clone.extra is not model.extra
    clone.extra["labels"].append("clone")
    assert model.extra == {"labels": ["source"]}


@pytest.mark.parametrize("transform", [False, True])
def test_subclass_slots_survive_copy_and_default_transform(transform: bool) -> None:
    model = SlottedModel(make_model().model)
    model.payload = {"labels": ["source"]}
    clone = model.transform(GiveUniqueNodeNames(), cleanup=False) if transform else copy.deepcopy(model)
    assert type(clone) is SlottedModel
    assert clone.payload == model.payload and clone.payload is not model.payload
    clone.payload["labels"].append("clone")
    assert model.payload == {"labels": ["source"]}


class ReducedModel(ModelWrapper):
    def __reduce_ex__(self, protocol: int) -> tuple[Any, ...]:
        return _rebuild_reduced, (self.model, self.payload)


def _rebuild_reduced(proto: ModelProto, payload: list[str]) -> ReducedModel:
    model = ReducedModel(proto)
    model.payload = payload
    model.rebuilt = True
    return model


@pytest.mark.parametrize("transform", [False, True])
def test_a_subclass_own_reduction_is_used(transform: bool) -> None:
    model = ReducedModel(make_model().model)
    model.payload = ["source"]
    clone = model.transform(GiveUniqueNodeNames(), cleanup=False) if transform else copy.deepcopy(model)
    assert type(clone) is ReducedModel and clone.rebuilt is True
    assert clone.payload == model.payload and clone.payload is not model.payload
    assert not hasattr(model, "rebuilt")


class RegisteredModel(ModelWrapper):
    pass


def _rebuild_registered(proto: ModelProto, payload: list[str]) -> RegisteredModel:
    model = RegisteredModel(proto)
    model.payload = payload
    model.rebuilt = True
    return model


def _reduce_registered(model: RegisteredModel) -> tuple[Any, ...]:
    return _rebuild_registered, (model.model, model.payload)


@pytest.mark.parametrize("transform", [False, True])
def test_a_copyreg_reducer_is_used(monkeypatch: pytest.MonkeyPatch, transform: bool) -> None:
    import copyreg

    monkeypatch.setitem(copyreg.dispatch_table, RegisteredModel, _reduce_registered)
    copyreg.pickle(RegisteredModel, _reduce_registered)
    model = RegisteredModel(make_model().model)
    model.payload = ["source"]
    clone = model.transform(GiveUniqueNodeNames(), cleanup=False) if transform else copy.deepcopy(model)
    assert type(clone) is RegisteredModel and clone.rebuilt is True
    assert clone.payload == model.payload and clone.payload is not model.payload
    assert not hasattr(model, "rebuilt")


class StringReducedModel(ModelWrapper):
    def __reduce_ex__(self, protocol: int) -> str:
        return "STRING_REDUCED_MODEL"


STRING_REDUCED_MODEL = StringReducedModel(make_model().model)


def test_a_string_reduction_deepcopies_to_itself() -> None:
    assert copy.deepcopy(STRING_REDUCED_MODEL) is STRING_REDUCED_MODEL


def test_a_body_parent_stays_shared_on_shallow_and_deep_copy() -> None:
    model = make_bodies_model()
    body = bodies(model)["else_branch"]
    for clone in [copy.copy(body), copy.deepcopy(body)]:
        assert clone is not body
        assert clone._parent is body._parent
        assert clone._parent.value is model
        model.set_metadata(CLOCK, 9.0)
        assert clone.get_metadata(CLOCK) == 9.0


def test_merge_takes_main_then_other_and_refuses_a_namespace_at_two_versions() -> None:
    main, other = GraphProto(), GraphProto()
    store(main, a="main", **{"test.platform/@version": "1", "test.platform/part": "x"})
    store(other, a="other", b="other", **{"test.platform/@version": "1", "test.platform/ports": "2"})
    merged = metadata.merge(main.metadata_props, other.metadata_props)
    assert [(p.key, p.value) for p in merged] == [
        ("a", "main"),
        ("test.platform/@version", "1"),
        ("test.platform/part", "x"),
        ("b", "other"),
        ("test.platform/ports", "2"),
    ]
    newer = GraphProto()
    store(newer, **{"test.platform/@version": "2"})
    with pytest.raises(MetadataError, match="test.platform/@version: the graphs merged state '1' and '2'"):
        metadata.merge(main.metadata_props, newer.metadata_props)
