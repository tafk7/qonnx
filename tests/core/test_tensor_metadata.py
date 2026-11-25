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

"""Typed metadata on a graph's tensors (qonnx.core.metadata, ModelWrapper's
get_metadata/set_metadata/delete_metadata/metadata_namespace/clear_metadata
with tensor=), what keeps it, and the cut rule."""

from __future__ import annotations

import pytest

import numpy as np
from enum import Enum
from onnx import StringStringEntryProto, TensorAnnotation, TensorProto, ValueInfoProto, helper
from pathlib import Path
from typing import Callable, Sequence

from qonnx.core import metadata
from qonnx.core.datatype import DataType
from qonnx.core.metadata import MetadataError, Namespace
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.create_generic_partitions import PartitionFromDict, PartitionFromLambda
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
    RemoveUnusedTensors,
    SortGraph,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name, qonnx_make_model
from qonnx.util.cleanup import cleanup_model


class Style(Enum):
    AUTO = 1
    BLOCK = 2


# follows a tensor into a cut
STREAM = Namespace("test.stream", version=1, follow=True)
DEPTH = STREAM.key("depth", int, check=lambda v: v > 0, expect="a depth > 0")
STYLE = STREAM.key("style", Style)
# does not
NOTES = Namespace("test.notes", version=1)
NOTE = NOTES.key("note", str)

TENSORS = ["x", "w", "h", "r", "y"]
KINDS = {"input": "x", "initializer": "w", "intermediate": "h", "output": "y"}


def make_model() -> ModelWrapper:
    """x (input) -> Mul by w (initializer) -> h -> Relu -> r -> Neg -> y (output)."""

    def vi(name: str) -> ValueInfoProto:
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 4])

    nodes = [
        helper.make_node("Mul", ["x", "w"], ["h"], name="mul"),
        helper.make_node("Relu", ["h"], ["r"], name="relu"),
        helper.make_node("Neg", ["r"], ["y"], name="neg"),
    ]
    graph = helper.make_graph(nodes, "g", [vi("x")], [vi("y")], value_info=[vi("h"), vi("r")])
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_initializer("w", np.full((1, 4), 2.0, dtype=np.float32))
    return model


def annotate(model: ModelWrapper, tensors: Sequence[str] = TENSORS) -> None:
    """Each tensor states both namespaces: depth by its position, its name as note."""
    for index, tensor in enumerate(tensors):
        model.set_metadata(DEPTH, index + 1, tensor=tensor)
        model.set_metadata(STYLE, Style.BLOCK, tensor=tensor)
        model.set_metadata(NOTE, tensor, tensor=tensor)


def stated(index: int, tensor: str) -> tuple[dict[str, int | Style], dict[str, str]]:
    return {"depth": index + 1, "style": Style.BLOCK}, {"note": tensor}


def entries(model: ModelWrapper, tensor: str) -> dict[str, str] | None:
    annotation = get_by_name(model.graph.quantization_annotation, tensor, "tensor_name")
    return None if annotation is None else {e.key: e.value for e in annotation.quant_parameter_tensor_names}


def store(model: ModelWrapper, tensor: str, **raw: str) -> None:
    """Entries written as text on a tensor, past the typed API."""
    annotation = get_by_name(model.graph.quantization_annotation, tensor, "tensor_name")
    if annotation is None:
        annotation = model.graph.quantization_annotation.add(tensor_name=tensor)
    for key, value in raw.items():
        annotation.quant_parameter_tensor_names.append(StringStringEntryProto(key=key, value=value))


def current_names(model: ModelWrapper) -> dict[str, str]:
    """Each original tensor's current name, found by the graph's structure."""
    (mul,) = model.get_nodes_by_op_type("Mul")
    (relu,) = model.get_nodes_by_op_type("Relu")
    return {
        "x": model.graph.input[0].name,
        "w": mul.input[1],
        "h": mul.output[0],
        "r": relu.output[0],
        "y": model.graph.output[0].name,
    }


def assert_annotated(model: ModelWrapper) -> None:
    names = current_names(model)
    for index, tensor in enumerate(TENSORS):
        stream, notes = stated(index, tensor)
        assert model.metadata_namespace(STREAM, tensor=names[tensor]) == stream, tensor
        assert model.metadata_namespace(NOTES, tensor=names[tensor]) == notes, tensor


@pytest.mark.parametrize("kind", KINDS)
def test_a_tensor_states_its_own_keys_apart_from_the_graph(kind: str) -> None:
    model = make_model()
    tensor = KINDS[kind]
    model.set_metadata(DEPTH, 4, tensor=tensor)
    model.set_metadata(STYLE, Style.AUTO, tensor=tensor)
    assert model.get_metadata(DEPTH, tensor=tensor) == 4
    assert model.metadata_namespace(STREAM, tensor=tensor) == {"depth": 4, "style": Style.AUTO}
    assert model.get_metadata(DEPTH) is None and model.metadata_namespace(STREAM) == {}
    assert all(model.metadata_namespace(STREAM, tensor=other) == {} for other in TENSORS if other != tensor)
    model.set_metadata(DEPTH, 9)
    assert model.get_metadata(DEPTH) == 9 and model.get_metadata(DEPTH, tensor=tensor) == 4
    # the tensor's version is stored as the graph's is, with the namespace's @follow
    assert entries(model, tensor) == {
        "test.stream/@version": "1",
        "test.stream/@follow": "true",
        "test.stream/depth": "4",
        "test.stream/style": "AUTO",
    }
    assert {p.key: p.value for p in model.graph.metadata_props} == {"test.stream/@version": "1", "test.stream/depth": "9"}
    model.set_metadata(NOTE, "n", tensor=tensor)
    assert entries(model, tensor)["test.notes/@version"] == "1" and "test.notes/@follow" not in entries(model, tensor)
    model.delete_metadata(DEPTH, tensor=tensor)
    assert model.metadata_namespace(STREAM, tensor=tensor) == {"style": Style.AUTO}
    assert model.get_metadata(DEPTH) == 9
    model.delete_metadata(DEPTH, tensor=tensor)  # absent: nothing to do
    model.delete_metadata(STYLE, tensor=tensor)
    model.delete_metadata(NOTE, tensor=tensor)
    assert entries(model, tensor) is None  # the emptied annotation goes
    model.delete_metadata(DEPTH)
    assert model.graph.metadata_props == [] and model.get_metadata(DEPTH) is None


def test_a_tensor_key_refuses_what_its_namespace_refuses() -> None:
    model = make_model()
    with pytest.raises(MetadataError, match="test.stream/depth: cannot store 0: expected an int, a depth > 0"):
        model.set_metadata(DEPTH, 0, tensor="h")
    assert entries(model, "h") is None  # nothing written, no annotation left
    store(model, "r", **{"test.stream/@version": "1", "test.stream/@follow": "true", "test.stream/style": "WIDE"})
    with pytest.raises(MetadataError, match="test.stream/style: stored 'WIDE' is not a Style"):
        model.get_metadata(DEPTH, tensor="r")  # a namespace is read whole


@pytest.mark.parametrize(
    "call",
    [
        lambda m: m.get_metadata(DEPTH, tensor="nope"),
        lambda m: m.set_metadata(DEPTH, 1, tensor="nope"),
        lambda m: m.delete_metadata(DEPTH, tensor="nope"),
        lambda m: m.metadata_namespace(STREAM, tensor="nope"),
        lambda m: m.clear_metadata(STREAM, tensor="nope"),
    ],
    ids=["get", "set", "delete", "namespace", "clear"],
)
def test_a_tensor_the_graph_does_not_have_is_refused_by_name(call: Callable[[ModelWrapper], object]) -> None:
    model = make_model()
    with pytest.raises(MetadataError, match="tensor 'nope': the graph has no tensor of that name"):
        call(model)
    assert len(model.graph.quantization_annotation) == 0


def test_a_tensor_without_value_info_is_a_tensor() -> None:
    model = make_model()
    model.graph.value_info.remove(model.get_tensor_valueinfo("h"))
    model.set_metadata(DEPTH, 2, tensor="h")
    assert model.get_metadata(DEPTH, tensor="h") == 2


def test_tensor_keys_sit_beside_the_datatype_and_layout_annotations() -> None:
    model = make_model()
    model.set_tensor_datatype("h", DataType["INT4"])
    model.set_metadata(DEPTH, 3, tensor="h")
    model.set_tensor_layout("h", ["N", "C"])
    model.set_tensor_datatype("w", DataType["INT2"])
    model.set_metadata(DEPTH, 1, tensor="w")
    assert model.get_tensor_datatype("h") == DataType["INT4"] and model.get_tensor_layout("h") == ["N", "C"]
    assert model.get_metadata(DEPTH, tensor="h") == 3
    model.delete_metadata(DEPTH, tensor="h")
    model.clear_metadata(STREAM, tensor="w")
    assert entries(model, "h") == {"finn_datatype": "INT4", "tensor_layout": "['N', 'C']"}
    assert entries(model, "w") == {"finn_datatype": "INT2"}


def test_clear_removes_a_namespace_from_one_tensor_or_from_every_tensor_as_stored() -> None:
    model = make_model()
    annotate(model)
    model.set_metadata(DEPTH, 7)
    assert model.tensors_with_metadata(STREAM) == TENSORS
    model.clear_metadata(STREAM, tensor="h")
    assert model.metadata_namespace(STREAM, tensor="h") == {} and model.get_metadata(NOTE, tensor="h") == "h"
    assert model.tensors_with_metadata(STREAM) == ["x", "w", "r", "y"]
    for tensor in model.tensors_with_metadata(STREAM):
        model.clear_metadata(STREAM, tensor=tensor)
    assert model.tensors_with_metadata(STREAM) == [] and model.tensors_with_metadata(NOTES) == TENSORS
    assert model.get_metadata(DEPTH) == 7  # the graph's own stays
    model.clear_metadata(STREAM)
    assert model.get_metadata(DEPTH) is None
    # a namespace stored malformed, or at a version the reader cannot read, is cleared too
    store(model, "r", **{"test.stream/@version": "5", "test.stream/width": "x"})
    with pytest.raises(MetadataError):
        model.metadata_namespace(STREAM, tensor="r")
    model.clear_metadata(STREAM, tensor="r")
    assert model.metadata_namespace(STREAM, tensor="r") == {}


def test_tensor_keys_survive_the_shape_and_initializer_setters() -> None:
    model = make_model()
    annotate(model)
    for tensor in ["x", "h", "r", "y"]:
        model.set_tensor_shape(tensor, [2, 4])
    model.set_initializer("w", np.zeros((1, 4), dtype=np.float32))
    model.set_tensor_shape("w", [1, 4], TensorProto.FLOAT)
    model.set_initializer("w2", np.zeros((1, 4), dtype=np.float32))  # a new one starts bare
    model.graph.node[0].input[1] = "w2"
    assert model.metadata_namespace(STREAM, tensor="w2") == {}
    model.graph.node[0].input[1] = "w"
    assert_annotated(model)


@pytest.mark.parametrize("kind", KINDS)
def test_a_tensor_key_is_renamed_with_its_tensor(kind: str) -> None:
    model = make_model()
    annotate(model)
    tensor = KINDS[kind]
    model.rename_tensor(tensor, "renamed")
    stream, notes = stated(TENSORS.index(tensor), tensor)
    assert (
        model.metadata_namespace(STREAM, tensor="renamed") == stream
        and model.metadata_namespace(NOTES, tensor="renamed") == notes
    )
    with pytest.raises(MetadataError, match=f"tensor '{tensor}'"):
        model.get_metadata(DEPTH, tensor=tensor)
    assert_annotated(model)


def reversed_then_sorted(model: ModelWrapper) -> ModelWrapper:
    nodes = list(model.graph.node)
    for node in nodes:
        model.graph.node.remove(node)
    model.graph.node.extend(reversed(nodes))
    return model.transform(SortGraph())


def saved_and_loaded(model: ModelWrapper, tmp_path: Path) -> ModelWrapper:
    model.save(tmp_path / "m.onnx")
    return ModelWrapper(str(tmp_path / "m.onnx"))


@pytest.mark.parametrize(
    "step",
    [
        lambda m, tmp: saved_and_loaded(m, tmp),
        lambda m, tmp: m.transform(InferShapes()),
        lambda m, tmp: m.transform(InferDataTypes()),
        lambda m, tmp: m.transform(GiveUniqueNodeNames()),
        lambda m, tmp: m.transform(GiveUniqueNodeNames()).transform(GiveReadableTensorNames()),
        lambda m, tmp: reversed_then_sorted(m),
        lambda m, tmp: m.cleanup(),
        lambda m, tmp: cleanup_model(m),
    ],
    ids=[
        "save-load",
        "InferShapes",
        "InferDataTypes",
        "GiveUniqueNodeNames",
        "GiveReadableTensorNames",
        "SortGraph",
        "ModelWrapper.cleanup",
        "cleanup_model",
    ],
)
def test_tensor_keys_survive_qonnx_steps(step: Callable[[ModelWrapper, Path], ModelWrapper], tmp_path: Path) -> None:
    model = make_model()
    annotate(model)
    after = step(model, tmp_path)
    assert_annotated(after)


def test_readable_names_rename_every_tensor_and_its_keys() -> None:
    model = make_model()
    annotate(model)
    after = cleanup_model(model)
    assert current_names(after) == {
        "x": "global_in",
        "w": "Mul_0_param0",
        "h": "Mul_0_out0",
        "r": "Relu_0_out0",
        "y": "global_out",
    }


def test_tensor_keys_are_removed_with_their_tensor() -> None:
    model = make_model()
    annotate(model)
    model.set_initializer("unused", np.zeros((1,), dtype=np.float32))
    model.set_metadata(DEPTH, 1, tensor="unused")
    after = model.transform(RemoveUnusedTensors())
    assert entries(after, "unused") is None
    assert after.tensors_with_metadata(STREAM) == TENSORS
    assert_annotated(after)


def test_a_shared_parameter_s_copy_carries_its_keys() -> None:
    model = make_model()
    model.graph.node.insert(1, helper.make_node("Mul", ["h", "w"], ["h2"], name="mul2"))
    model.graph.node[2].input[0] = "h2"
    model.set_tensor_shape("h2", [1, 4])
    model.set_tensor_datatype("w", DataType["INT2"])
    model.set_tensor_layout("w", ["N", "C"])
    model.set_tensor_sparsity("w", {"dw": {"kernel_shape": [1, 4]}})
    model.set_metadata(DEPTH, 5, tensor="w")
    model.set_metadata(NOTE, "shared", tensor="w")
    after = model.transform(GiveUniqueParameterTensors(), cleanup=False)
    first, second = after.get_nodes_by_op_type("Mul")
    copy = second.input[1]
    assert copy != "w" and first.input[1] == "w"
    assert after.metadata_namespace(STREAM, tensor=copy) == {"depth": 5}
    assert after.get_metadata(NOTE, tensor=copy) == "shared" and after.get_tensor_datatype(copy) == DataType["INT2"]
    assert after.get_tensor_layout(copy) == ["N", "C"]
    assert after.get_tensor_sparsity(copy) == {"dw": {"kernel_shape": [1, 4]}}


# versioning


def make_stream_v2() -> tuple[Namespace, metadata.Key[int]]:
    """Version 1 called `depth` `slots`; version 2 renamed it."""
    ns = Namespace("test.stream", version=2, follow=True)
    depth = ns.key("depth", int)
    ns.key("style", Style)
    ns.upgrade(1, lambda e: {("depth" if n == "slots" else n): t for n, t in e.items()})
    return ns, depth


def test_a_tensor_namespace_is_upgraded_on_reading_and_rewritten_on_writing() -> None:
    model = make_model()
    ns, depth = make_stream_v2()
    store(model, "h", **{"test.stream/@version": "1", "test.stream/@follow": "true", "test.stream/slots": "3"})
    assert model.metadata_namespace(ns, tensor="h") == {"depth": 3}
    assert entries(model, "h")["test.stream/@version"] == "1"  # reading changes nothing
    model.set_metadata(ns.keys["style"], Style.AUTO, tensor="h")
    assert entries(model, "h") == {
        "test.stream/@version": "2",
        "test.stream/@follow": "true",
        "test.stream/depth": "3",
        "test.stream/style": "AUTO",
    }


def test_a_version_may_change_whether_its_namespace_follows() -> None:
    model = make_model()
    store(model, "h", **{"test.notes/@version": "1", "test.notes/note": "n"})
    ns = Namespace("test.notes", version=2, follow=True)
    note = ns.key("note", str)
    ns.upgrade(1, dict)
    assert model.get_metadata(note, tensor="h") == "n"  # stored at version 1, which did not follow
    model.set_metadata(note, "m", tensor="h")
    assert entries(model, "h") == {"test.notes/@version": "2", "test.notes/@follow": "true", "test.notes/note": "m"}


@pytest.mark.parametrize(
    "raw, refusal",
    [
        (
            {"test.stream/@version": "1", "test.stream/@follow": "true", "test.stream/width": "3"},
            r"test.stream: stored keys \['width'\] are not declared",
        ),
        ({"test.stream/@follow": "true", "test.stream/depth": "3"}, r"stored without test.stream/@version"),
        ({"test.stream/@version": "2", "test.stream/@follow": "true"}, "stored at version 2, which a reader of version 1"),
        ({"test.stream/@version": "1", "test.stream/depth": "3"}, "test.stream/@follow: stored without it at version 1"),
        ({"test.stream/@version": "1", "test.stream/@follow": "yes"}, "test.stream/@follow: stored 'yes', not 'true'"),
    ],
    ids=["unknown key", "no version", "newer version", "follow missing", "follow malformed"],
)
def test_a_tensor_namespace_stored_unlike_its_declaration_is_refused(raw: dict[str, str], refusal: str) -> None:
    model = make_model()
    store(model, "r", **raw)
    with pytest.raises(MetadataError, match=refusal):
        model.metadata_namespace(STREAM, tensor="r")
    with pytest.raises(MetadataError, match=refusal):
        model.set_metadata(DEPTH, 1, tensor="r")


def test_follow_is_a_tensor_s_and_must_match_the_declaration() -> None:
    model = make_model()
    store(model, "r", **{"test.notes/@version": "1", "test.notes/@follow": "true", "test.notes/note": "n"})
    with pytest.raises(MetadataError, match="test.notes/@follow: stored with it at version 1, which declares follow=False"):
        model.get_metadata(NOTE, tensor="r")
    model.set_metadata_prop("test.stream/@version", "1")
    model.set_metadata_prop("test.stream/@follow", "true")
    with pytest.raises(MetadataError, match="test.stream/@follow: stored on a graph"):
        model.get_metadata(DEPTH)
    # a following namespace stored on the graph states no @follow
    model.clear_metadata(STREAM)
    model.set_metadata(DEPTH, 2)
    assert {p.key for p in model.graph.metadata_props} == {"test.stream/@version", "test.stream/depth"}


# the cut


def test_cut_keeps_the_namespaces_stored_as_following_and_untyped_entries() -> None:
    annotation = TensorAnnotation(tensor_name="t")
    raw = {
        "finn_datatype": "INT4",
        "test.stream/@version": "1",
        "test.stream/@follow": "true",
        "test.stream/depth": "2",
        "test.notes/@version": "1",
        "test.notes/note": "n",
        "loose/entry": "kept",
    }
    for key, value in raw.items():
        annotation.quant_parameter_tensor_names.append(StringStringEntryProto(key=key, value=value))
    metadata.cut(annotation.quant_parameter_tensor_names)
    assert {e.key: e.value for e in annotation.quant_parameter_tensor_names} == {
        "finn_datatype": "INT4",
        "test.stream/@version": "1",
        "test.stream/@follow": "true",
        "test.stream/depth": "2",
        "loose/entry": "kept",
    }


def partitioned(model: ModelWrapper, how: str, partition_dir: Path) -> tuple[ModelWrapper, ModelWrapper]:
    """The Mul and the Relu cut into one partition, the Neg left in the parent."""
    if how == "PartitionFromLambda":
        cut = PartitionFromLambda(lambda node: 0 if node.op_type in ["Mul", "Relu"] else -1, str(partition_dir))
    else:
        cut = PartitionFromDict({0: [0, 1]}, str(partition_dir))
    parent = model.transform(cut)
    (node,) = parent.get_nodes_by_op_type("GenericPartition")
    return parent, ModelWrapper(getCustomOp(node).get_nodeattr("model"))


@pytest.mark.parametrize("how", ["PartitionFromLambda", "PartitionFromDict"])
def test_a_tensor_key_follows_its_tensor_into_a_cut_as_its_namespace_states(how: str, tmp_path: Path) -> None:
    model = make_model().transform(InferShapes())
    annotate(model)
    model.set_tensor_datatype("h", DataType["INT8"])
    model.set_metadata(DEPTH, 8)
    model.set_metadata(NOTE, "graph")
    parent, body = partitioned(model, how, tmp_path)
    assert [i.name for i in body.graph.input] == ["x"] and [o.name for o in body.graph.output] == ["r"]
    # the body: each tensor's following namespace, not the other
    for index, tensor in enumerate(["x", "w", "h", "r"]):
        stream, _ = stated(index, tensor)
        assert body.metadata_namespace(STREAM, tensor=tensor) == stream, tensor
        assert body.metadata_namespace(NOTES, tensor=tensor) == {}, tensor
        assert not any(key.startswith("test.notes/") for key in entries(body, tensor)), tensor
    assert body.get_tensor_datatype("h") == DataType["INT8"]  # untyped annotations are copied as before
    assert body.get_metadata(DEPTH) == 8 and body.get_metadata(NOTE) == "graph"  # the graph's metadata, whole
    # the parent keeps its own copy of the boundary tensors', and its own tensors'
    for tensor in ["x", "r", "y"]:
        stream, notes = stated(TENSORS.index(tensor), tensor)
        assert (
            parent.metadata_namespace(STREAM, tensor=tensor) == stream
            and parent.metadata_namespace(NOTES, tensor=tensor) == notes
        )
    assert entries(parent, "w") is None and entries(parent, "h") is None
    assert parent.tensors_with_metadata(STREAM) == ["x", "r", "y"]
    # what a caller's cut needs to strip the parent: one tensor, or every tensor
    parent.clear_metadata(STREAM, tensor="x")
    assert parent.metadata_namespace(STREAM, tensor="x") == {} and parent.get_metadata(NOTE, tensor="x") == "x"
    for tensor in parent.tensors_with_metadata(STREAM):
        parent.clear_metadata(STREAM, tensor=tensor)
    assert parent.tensors_with_metadata(STREAM) == [] and parent.tensors_with_metadata(NOTES) == ["x", "r", "y"]


def test_a_body_s_tensor_keys_are_its_own() -> None:
    site = Namespace("test.site", version=1, inherit=True, follow=True)
    clock = site.key("clock", int)
    inner = helper.make_graph([helper.make_node("Relu", ["x"], ["z"])], "body", [], [])
    holder = helper.make_node("Holder", ["x"], [], name="holder", body=inner)
    graph = helper.make_graph([holder], "outer", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])], [])
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_metadata(clock, 5)
    model.set_metadata(clock, 3, tensor="x")
    body = model.make_subgraph_modelwrapper(holder.attribute[0].g)
    assert body.get_metadata(clock) == 5  # a graph key is inherited
    assert body.get_metadata(clock, tensor="x") is None  # a tensor's is not
