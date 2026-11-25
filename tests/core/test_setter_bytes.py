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

"""The setters' serialized bytes: set_tensor_datatype, set_tensor_shape and
set_tensor_layout, rebuilt on the writes they share with state(), write the bytes
qonnx's setters wrote before them, recorded from commit 4c1f9b4 (EXPECTED). The
one change is set_tensor_datatype(t, None), which removes the emptied datatype entry
rather than leaving an empty one. Where both state() and a setter admit a write and
the old value info has no side fields, the two write the same bytes (the agreement
cases); nowhere else is that claimed."""

# EXPECTED was recorded from a checkout of commit 4c1f9b4 (python 3.10, onnx
# 1.17), loading this file by its path PATH:
#
#     import importlib.util
#
#     spec = importlib.util.spec_from_file_location("setter_bytes", PATH)
#     cases = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(cases)
#     for name, case in cases.CASES.items():
#         print(name, case().model.SerializeToString(deterministic=True).hex())

from __future__ import annotations

import pytest

import numpy as np
from collections.abc import Callable
from onnx import ModelProto, TensorProto, helper, numpy_helper

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model


def graph_model(**fields: list) -> ModelWrapper:
    """An empty graph with ``fields`` (input, output, value_info, initializer), its
    initializers given no value info."""
    graph = helper.make_graph([], "setters", fields.get("input", []), fields.get("output", []))
    graph.value_info.extend(fields.get("value_info", []))
    graph.initializer.extend(fields.get("initializer", []))
    return ModelWrapper(qonnx_make_model(graph), fix_missing_initializer_valueinfo=False)


def chain() -> ModelWrapper:
    """x -> Identity -> a -> Identity -> b -> Identity -> y, a and b with value info."""
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["x"], ["a"]),
            helper.make_node("Identity", ["a"], ["b"]),
            helper.make_node("Identity", ["b"], ["y"]),
        ],
        "chain",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
        value_info=[helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 2]) for name in ("a", "b")],
    )
    return ModelWrapper(qonnx_make_model(graph))


def sided() -> ModelWrapper:
    """x, an input whose value info has a doc string, a type denotation and a
    dimension denotation."""
    info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    info.doc_string = "input docs"
    info.type.denotation = "TENSOR"
    info.type.tensor_type.shape.dim[0].denotation = "DATA_BATCH"
    return graph_model(input=[info], output=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])])


def optional() -> ModelWrapper:
    """Dropout with its optional mask output absent ("")."""
    graph = helper.make_graph(
        [helper.make_node("Dropout", ["x"], ["y", ""])],
        "optional",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )
    return ModelWrapper(qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 13)]))


def symbolic_and_unknown_extents() -> ModelWrapper:
    model = chain()
    model.set_tensor_shape("a", ["N", None, 3])
    model.set_tensor_shape("x", ["N", 2])
    return model


def side_fields_replaced() -> ModelWrapper:
    model = sided()
    model.set_tensor_shape("x", [3, 2])
    return model


def shape_before_its_initializer() -> ModelWrapper:
    model = graph_model(initializer=[numpy_helper.from_array(np.ones((2, 3), np.float32), name="w")])
    model.set_tensor_shape("w", [3, 2])
    model.set_initializer("w", np.zeros((3, 2), np.float64))
    return model


def intermediate_replaced() -> ModelWrapper:
    model = chain()
    model.set_tensor_shape("a", [2, 2], TensorProto.INT8)
    model.set_tensor_shape("x", [2, 2])
    model.set_tensor_shape("y", [2, 2])
    model.set_tensor_shape("new", [4])
    return model


def initializer_replaced() -> ModelWrapper:
    model = chain()
    model.set_initializer("a", np.array([[1, 2]], np.float32))
    model.set_initializer("a", np.array([[3, 4]], np.int8))
    return model


def absent_output_annotated() -> ModelWrapper:
    model = optional()
    model.set_tensor_datatype("", DataType["FLOAT32"])
    model.set_tensor_layout("", ["N"])
    return model


def datatype_and_layout_set() -> ModelWrapper:
    model = chain()
    model.set_tensor_datatype("a", DataType["INT8"])
    model.set_tensor_layout("a", ["N", "C"])
    model.set_tensor_layout("b", ["N", "C"])
    model.set_tensor_datatype("b", DataType["UINT4"])
    model.set_tensor_datatype("b", DataType["INT4"])
    model.set_tensor_layout("b", ["C", "N"])
    return model


def datatype_unset() -> ModelWrapper:
    """The one changed case (an emptied datatype entry removed)."""
    model = datatype_and_layout_set()
    model.set_tensor_datatype("a", None)
    model.set_tensor_datatype("x", DataType["INT8"])
    model.set_tensor_datatype("x", None)
    model.set_tensor_datatype("unannotated", None)
    return model


CASES: dict[str, Callable[[], ModelWrapper]] = {
    "symbolic_and_unknown_extents": symbolic_and_unknown_extents,
    "side_fields_replaced": side_fields_replaced,
    "shape_before_its_initializer": shape_before_its_initializer,
    "intermediate_replaced": intermediate_replaced,
    "initializer_replaced": initializer_replaced,
    "absent_output_annotated": absent_output_annotated,
    "datatype_and_layout_set": datatype_and_layout_set,
    "datatype_unset": datatype_unset,
}

EXPECTED: dict[str, str] = {
    "symbolic_and_unknown_extents": (
        "080a3a95010a100a017812016122084964656e746974790a100a016112016222084964656e746974790a100a016212017922"
        "084964656e746974791205636861696e5a140a0178120f0a0d080112090a0312014e0a02080262130a0179120e0a0c080112"
        "080a0208010a0208026a130a0162120e0a0c080112080a0208010a0208026a160a016112110a0f0801120b0a0312014e0a00"
        "0a02080342040a00100b421b0a17716f6e6e782e637573746f6d5f6f702e67656e6572616c1001"
    ),
    "side_fields_replaced": (
        "080a3a331207736574746572735a130a0178120e0a0c080112080a0208030a02080262130a0179120e0a0c080112080a0208"
        "010a02080242040a00100b421b0a17716f6e6e782e637573746f6d5f6f702e67656e6572616c1001"
    ),
    "shape_before_its_initializer": (
        "080a3a5b1207736574746572732a3b08030802100b4201774a30000000000000000000000000000000000000000000000000"
        "0000000000000000000000000000000000000000000000006a130a0177120e0a0c080b12080a0208030a02080242040a0010"
        "0b421b0a17716f6e6e782e637573746f6d5f6f702e67656e6572616c1001"
    ),
    "intermediate_replaced": (
        "080a3aa4010a100a017812016122084964656e746974790a100a016112016222084964656e746974790a100a016212017922"
        "084964656e746974791205636861696e5a130a0178120e0a0c080112080a0208020a02080262130a0179120e0a0c08011208"
        "0a0208020a0208026a130a0162120e0a0c080112080a0208010a0208026a130a0161120e0a0c080312080a0208020a020802"
        "6a110a036e6577120a0a08080112040a02080442040a00100b421b0a17716f6e6e782e637573746f6d5f6f702e67656e6572"
        "616c1001"
    ),
    "initializer_replaced": (
        "080a3aa0010a100a017812016122084964656e746974790a100a016112016222084964656e746974790a100a016212017922"
        "084964656e746974791205636861696e2a0d0801080210034201614a0203045a130a0178120e0a0c080112080a0208010a02"
        "080262130a0179120e0a0c080112080a0208010a0208026a130a0162120e0a0c080112080a0208010a0208026a130a016112"
        "0e0a0c080312080a0208010a02080242040a00100b421b0a17716f6e6e782e637573746f6d5f6f702e67656e6572616c1001"
    ),
    "absent_output_annotated": (
        "080a3a750a110a01781201791200220744726f706f757412086f7074696f6e616c5a0f0a0178120a0a08080112040a020802"
        "620f0a0179120a0a08080112040a02080272340a0012180a0d66696e6e5f64617461747970651207464c4f4154333212160a"
        "0d74656e736f725f6c61796f757412055b274e275d42040a00100d"
    ),
    "datatype_and_layout_set": (
        "080a3a83020a100a017812016122084964656e746974790a100a016112016222084964656e746974790a100a016212017922"
        "084964656e746974791205636861696e5a130a0178120e0a0c080112080a0208010a02080262130a0179120e0a0c08011208"
        "0a0208010a0208026a130a0161120e0a0c080112080a0208010a0208026a130a0162120e0a0c080112080a0208010a020802"
        "72370a016112150a0d66696e6e5f64617461747970651204494e5438121b0a0d74656e736f725f6c61796f7574120a5b274e"
        "272c202743275d72370a0162121b0a0d74656e736f725f6c61796f7574120a5b2743272c20274e275d12150a0d66696e6e5f"
        "64617461747970651204494e543442040a00100b421b0a17716f6e6e782e637573746f6d5f6f702e67656e6572616c1001"
    ),
    "datatype_unset": (
        "080a3af5010a100a017812016122084964656e746974790a100a016112016222084964656e746974790a100a016212017922"
        "084964656e746974791205636861696e5a130a0178120e0a0c080112080a0208010a02080262130a0179120e0a0c08011208"
        "0a0208010a0208026a130a0161120e0a0c080112080a0208010a0208026a130a0162120e0a0c080112080a0208010a020802"
        "72220a01611200121b0a0d74656e736f725f6c61796f7574120a5b274e272c202743275d72370a0162121b0a0d74656e736f"
        "725f6c61796f7574120a5b2743272c20274e275d12150a0d66696e6e5f64617461747970651204494e543472050a01781200"
        "42040a00100b421b0a17716f6e6e782e637573746f6d5f6f702e67656e6572616c1001"
    ),
}


def saved(model: ModelWrapper) -> bytes:
    return model.model.SerializeToString(deterministic=True)


@pytest.mark.parametrize("name", sorted(set(CASES) - {"datatype_unset"}))
def test_the_setters_write_the_bytes_they_wrote_before(name: str) -> None:
    assert saved(CASES[name]()).hex() == EXPECTED[name]


def test_unsetting_a_datatype_removes_the_emptied_entry() -> None:
    """The bytes before, with each emptied (empty-keyed) datatype entry removed."""
    before = ModelProto.FromString(bytes.fromhex(EXPECTED["datatype_unset"]))
    emptied = 0
    for annotation in before.graph.quantization_annotation:
        props = annotation.quant_parameter_tensor_names
        for entry in list(props):
            if not entry.key and not entry.value:
                props.remove(entry)
                emptied += 1
    assert emptied == 2
    assert saved(datatype_unset()) == before.SerializeToString(deterministic=True)


# where state() and a setter agree


@pytest.mark.parametrize("name", ["a", "x", "y", "new"])
def test_state_and_set_tensor_shape_agree_on_a_value_info_without_side_fields(name: str) -> None:
    by_setter, by_state = chain(), chain()
    by_setter.set_tensor_shape(name, [2, 2], TensorProto.INT8)
    by_state.state(name, shape=(2, 2), container=TensorProto.INT8)
    assert saved(by_state) == saved(by_setter)


def test_state_and_set_tensor_shape_agree_on_a_new_tensor() -> None:
    by_setter, by_state = chain(), chain()
    by_setter.set_tensor_shape("new", [3])
    by_state.state("new", shape=(3,))
    assert saved(by_state) == saved(by_setter)


def test_state_and_the_setters_agree_on_a_datatype_or_layout_written_or_removed() -> None:
    by_setter, by_state = chain(), chain()
    for model in (by_setter, by_state):
        model.set_tensor_layout("b", ["N", "C"])
    by_setter.set_tensor_datatype("a", DataType["INT8"])
    by_setter.set_tensor_layout("a", ["N", "C"])
    by_setter.set_tensor_datatype("b", DataType["INT4"])
    by_setter.set_tensor_datatype("x", DataType["INT8"])
    by_setter.set_tensor_datatype("x", None)
    by_state.state("a", element=DataType["INT8"])
    by_state.state("a", layout=("N", "C"))
    by_state.state("b", element=DataType["INT4"])
    by_state.state("x", element=DataType["INT8"])
    by_state.state("x", element=None)
    assert saved(by_state) == saved(by_setter)


def test_an_intermediate_replaced_by_either_goes_last() -> None:
    """The replacement order: an intermediate's new value info goes after the others."""
    by_setter, by_state = chain(), chain()
    by_setter.set_tensor_shape("a", [2])
    by_state.state("a", shape=(2,))
    assert [x.name for x in by_setter.graph.value_info] == ["b", "a"]
    assert saved(by_state) == saved(by_setter)


def test_state_keeps_side_fields_the_setter_replaces() -> None:
    by_state = sided()
    by_state.state("x", shape=(3, 2))
    info = by_state.get_tensor_valueinfo("x")
    assert (info.doc_string, info.type.denotation, info.type.tensor_type.shape.dim[0].denotation) == (
        "input docs",
        "TENSOR",
        "DATA_BATCH",
    )
    assert saved(by_state) != saved(side_fields_replaced())
