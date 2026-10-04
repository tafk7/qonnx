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
# * Neither the name of AMD nor the names of its
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

import pytest

from enum import Enum
from onnx import GraphProto, StringStringEntryProto

from qonnx.core import metadata
from qonnx.core.metadata import JSON, MetadataError, Namespace


class Block(Enum):
    SMALL = 1
    LARGE = 2


def make_namespace(version=1):
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


def entries(graph):
    return {prop.key: prop.value for prop in graph.metadata_props}


def store(graph, **raw):
    for key, value in raw.items():
        graph.metadata_props.append(StringStringEntryProto(key=key, value=value))


def test_keys_round_trip_as_one_entry_each_with_the_namespace_version():
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


def test_an_int_is_stored_as_a_float_where_a_float_is_declared():
    ns, k = make_namespace()
    graph = GraphProto()
    metadata.write(graph.metadata_props, k["period"], 5)
    assert entries(graph)["test.platform/period"] == "5.0"
    assert metadata.read(graph.metadata_props, ns)["period"] == 5.0


def test_writing_again_replaces_the_entry():
    ns, k = make_namespace()
    graph = GraphProto()
    metadata.write(graph.metadata_props, k["ports"], 1)
    metadata.write(graph.metadata_props, k["ports"], 2)
    assert [p.key for p in graph.metadata_props] == ["test.platform/@version", "test.platform/ports"]
    assert metadata.read(graph.metadata_props, ns) == {"ports": 2}


def test_an_absent_namespace_reads_empty():
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
def test_a_value_of_the_wrong_type_is_refused_on_writing(name, value):
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
def test_a_malformed_stored_value_is_refused_naming_entry_text_and_expectation(name, text, expectation):
    ns, _ = make_namespace()
    graph = GraphProto()
    store(graph, **{"test.platform/@version": "1", f"test.platform/{name}": text})
    with pytest.raises(MetadataError) as refusal:
        metadata.read(graph.metadata_props, ns)
    assert str(refusal.value).startswith(f"test.platform/{name}: stored {text!r} is not {expectation}")


def test_an_undeclared_stored_key_is_refused():
    ns, _ = make_namespace()
    graph = GraphProto()
    store(graph, **{"test.platform/@version": "1", "test.platform/clock": "5"})
    with pytest.raises(MetadataError, match=r"stored keys \['clock'\] are not declared"):
        metadata.read(graph.metadata_props, ns)


def test_keys_without_a_version_and_malformed_versions_are_refused():
    ns, _ = make_namespace()
    graph = GraphProto()
    store(graph, **{"test.platform/part": "x"})
    with pytest.raises(MetadataError, match="without test.platform/@version"):
        metadata.read(graph.metadata_props, ns)
    graph = GraphProto()
    store(graph, **{"test.platform/@version": "v1"})
    with pytest.raises(MetadataError, match="is not a positive int"):
        metadata.read(graph.metadata_props, ns)


def test_an_entry_stored_twice_is_refused():
    ns, _ = make_namespace()
    graph = GraphProto()
    store(graph, **{"test.platform/@version": "1", "test.platform/part": "x"})
    store(graph, **{"test.platform/part": "y"})
    with pytest.raises(MetadataError, match="test.platform/part: stored twice"):
        metadata.read(graph.metadata_props, ns)


def test_a_version_the_reader_cannot_upgrade_from_is_refused():
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


def make_v3():
    """Version 1 had `clock_mhz` (an int); version 2 replaced it by `period` (ns);
    version 3 renamed `name` to `part`."""
    ns, k = make_namespace(version=3)
    ns.upgrade(1, lambda e: {**{n: t for n, t in e.items() if n != "clock_mhz"}, "period": repr(1000 / int(e["clock_mhz"]))})
    ns.upgrade(2, lambda e: {("part" if n == "name" else n): t for n, t in e.items()})
    return ns, k


def test_a_reader_upgrades_an_earlier_version_through_each_step():
    ns, _ = make_v3()
    graph = GraphProto()
    store(graph, **{"test.platform/@version": "1", "test.platform/clock_mhz": "200", "test.platform/name": "xc7z020"})
    assert metadata.read(graph.metadata_props, ns) == {"period": 5.0, "part": "xc7z020"}
    assert entries(graph)["test.platform/@version"] == "1"  # reading changes nothing


def test_a_writer_rewrites_an_earlier_version_at_the_current_one():
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


def test_upgrades_are_declared_once_from_an_earlier_version():
    ns, _ = make_namespace(version=2)
    with pytest.raises(ValueError, match="cannot upgrade from 2"):
        ns.upgrade(2, dict)
    ns.upgrade(1, dict)
    with pytest.raises(ValueError, match="twice"):
        ns.upgrade(1, dict)


def test_delete_removes_the_entry_and_the_version_with_the_last_key():
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


def test_declarations_are_checked():
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
