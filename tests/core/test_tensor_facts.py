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

"""A tensor's facts read and written as one: facts(), state(), unstate() and
forget(), each refusing what it cannot do with the model's bytes unchanged."""

from __future__ import annotations

import pytest

import copy
import dataclasses
import numpy as np
import onnx
import warnings
from fractions import Fraction
from onnx import TensorProto, ValueInfoProto, helper, numpy_helper

from qonnx.core import tensor_facts, tensor_ranges
from qonnx.core.datatype import DataType, DataTypeNameUnknown
from qonnx.core.metadata import MetadataRefused
from qonnx.core.modelwrapper import ModelWrapper, TensorFactRefused
from qonnx.core.ranges import Range
from qonnx.core.tensor_facts import NonTensor, TensorFacts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model


def model(*, infos: list[ValueInfoProto] | None = None) -> ModelWrapper:
    graph = helper.make_graph([], "facts", [], [], value_info=infos or [])
    return ModelWrapper(qonnx_make_model(graph), fix_missing_initializer_valueinfo=False)


def saved(value: ModelWrapper) -> bytes:
    return value.model.SerializeToString(deterministic=True)


@pytest.mark.parametrize(
    ("dims", "expected"),
    [(None, None), ([], ()), ([0], (0,)), (["N"], None), ([None], None), ([2, 3], (2, 3))],
)
def test_shape_presence_is_read_without_changing_the_model(
    dims: list[int | str | None] | None, expected: tuple[int, ...] | None
) -> None:
    m = model(infos=[helper.make_tensor_value_info("x", TensorProto.FLOAT, dims)])
    before = saved(m)
    facts = m.facts("x")
    assert isinstance(facts, TensorFacts)
    assert facts.shape == expected
    assert facts.container == TensorProto.FLOAT
    assert saved(m) == before


def test_unstated_and_explicit_zero_containers_are_distinct() -> None:
    absent = ValueInfoProto(name="absent")
    absent.type.tensor_type.SetInParent()
    explicit = helper.make_tensor_value_info("zero", 0, None)
    m = model(infos=[absent, explicit])
    before = saved(m)
    a, z = m.facts("absent"), m.facts("zero")
    assert isinstance(a, TensorFacts) and isinstance(z, TensorFacts)
    assert a.container is None
    assert z.container == 0
    assert saved(m) == before
    m.state("untyped", shape=(2, 3), container=0)
    vi = m.get_tensor_valueinfo("untyped")
    assert vi is not None and vi.type.tensor_type.HasField("elem_type")
    assert vi.type.tensor_type.elem_type == 0


def test_annotation_only_writes_do_not_create_value_info() -> None:
    m = model()
    m.state("x", element=DataType["INT8"], range=Range(-3, 5), layout=("N", "C"))
    assert m.get_tensor_valueinfo("x") is None
    m.state("x", shape=(2, 3))
    facts = m.facts("x")
    assert isinstance(facts, TensorFacts)
    assert facts == TensorFacts((2, 3), DataType["INT8"], Range(-3, 5), ("N", "C"), 1)
    m.state("x", element=DataType["INT16"])
    m.unstate("x", "layout")
    assert m.get_stated_tensor_range("x") == Range(-3, 5)
    assert m.get_tensor_layout("x") is None
    before = saved(m)
    m.unstate("missing", "shape")
    m.forget("missing")
    assert saved(m) == before


def test_fact_writes_keep_scalar_zero_extent_and_missing_shape_distinct() -> None:
    m = model()
    facts = m.facts("x")
    assert isinstance(facts, TensorFacts) and facts.shape is None
    for dims in ((), (0,)):
        m.state("x", shape=dims, element=DataType["INT8"])
        facts = m.facts("x")
        assert isinstance(facts, TensorFacts) and facts.shape == dims
    m.unstate("x", "shape")
    facts = m.facts("x")
    assert isinstance(facts, TensorFacts) and facts.shape is None


def test_the_reader_carries_datatype_and_metadata_refusals_with_the_tensor() -> None:
    m = model()
    annotation = m.graph.quantization_annotation.add(tensor_name="x")
    entry = annotation.quant_parameter_tensor_names.add(key="finn_datatype", value="unknown-type")
    with pytest.raises(DataTypeNameUnknown) as unknown:
        DataType[entry.value]
    with pytest.raises(DataTypeNameUnknown) as carried:
        m.facts("x")
    assert carried.value.findings[0] == unknown.value.findings[0].about(tensor="x")
    entry.value = "INT8"
    annotation.quant_parameter_tensor_names.add(key="qonnx.tensor/@version", value="1")
    annotation.quant_parameter_tensor_names.add(key="qonnx.tensor/range", value="malformed")
    with pytest.raises(MetadataRefused) as metadata:
        m.get_stated_tensor_range("x")
    with pytest.raises(MetadataRefused) as carried_metadata:
        m.facts("x")
    assert carried_metadata.value.findings[0] == metadata.value.findings[0].about(tensor="x")


def test_float16_chain_keeps_its_rounding_and_states_no_intermediate_value_info() -> None:
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "y"], ["z"]), helper.make_node("Identity", ["z"], ["out"])],
        "rounding",
        [helper.make_tensor_value_info(x, TensorProto.FLOAT16, [1]) for x in ("x", "y")],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT16, [1])],
    )
    m = ModelWrapper(qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 13)]))
    m.state("x", element=DataType["FLOAT32"], range=Range(1, 1))
    m.state("y", element=DataType["FLOAT32"], range=Range(Fraction(1, 2048), Fraction(1, 2048)))
    m = m.transform(InferDataTypes(exact_ranges=True))
    assert float(np.float16(1) + np.float16(1 / 2048)) == 1
    assert m.get_stated_tensor_range("z") == Range(1, 1)
    assert m.get_stated_tensor_range("out") == Range(1, 1)
    assert m.get_tensor_valueinfo("z") is None


def test_warning_as_error_leaves_bytes_and_does_not_suppress_retry() -> None:
    m = model()
    m.state("x", element=DataType["INT16"], range=Range(-3, 5))
    before = saved(m)
    with warnings.catch_warnings(), tensor_ranges.one_pass(m):
        warnings.simplefilter("error", tensor_ranges.RangeContradiction)
        for _ in range(2):
            with pytest.raises(tensor_ranges.RangeContradiction):
                m.state("x", element=DataType["INT8"], range=Range(-300, 300))
            assert saved(m) == before


def test_unnamed_range_cannot_be_cleared_by_conflicting_element() -> None:
    m = model()
    m.state("x", element=DataType["INT16"], range=Range(-300, 300))
    before = saved(m)
    with pytest.raises(TensorFactRefused) as error:
        m.state("x", element=DataType["INT8"])
    assert error.value.findings[0].code == "fact-conflict"
    assert saved(m) == before
    m.unstate("x", "element")
    assert m.get_stated_tensor_range("x") == Range(-300, 300)


def test_initializer_replacement_unstate_and_forget_keep_source_coherent() -> None:
    m = model()
    m.set_initializer("w", np.ones((2, 3), np.float32))
    before = saved(m)
    with pytest.raises(TensorFactRefused):
        m.state("w", shape=(3, 2))
    with pytest.raises(TensorFactRefused):
        m.state("w", container=TensorProto.DOUBLE)
    assert saved(m) == before
    m.set_initializer("w", np.zeros((3, 2), np.float64))
    facts = m.facts("w")
    assert isinstance(facts, TensorFacts) and facts.shape == (3, 2) and facts.container == 11
    m.unstate("w", "shape")
    vi = m.get_tensor_valueinfo("w")
    assert vi is not None and not vi.type.tensor_type.HasField("shape")
    assert m.facts("w") == TensorFacts((3, 2), None, None, None, 11)
    m.forget("w")
    assert m.get_tensor_valueinfo("w") is None
    source = m.get_initializer("w")
    assert source is not None
    assert np.array_equal(source, np.zeros((3, 2), np.float64))
    assert m.facts("w") == TensorFacts((3, 2), None, None, None, 11)


def test_interface_shape_can_be_unstated_but_membership_cannot_be_forgotten() -> None:
    m = model()
    m.graph.input.append(helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]))
    m.unstate("x", "shape")
    before = saved(m)
    with pytest.raises(TensorFactRefused) as error:
        m.forget("x")
    assert error.value.findings[0].owner == "ModelWrapper.forget"
    assert saved(m) == before
    assert [x.name for x in m.graph.input] == ["x"]


def test_shape_update_preserves_serialized_side_fields() -> None:
    vi = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    vi.doc_string = "input docs"
    vi.metadata_props.add(key="retained-key", value="retained-value")
    vi.type.denotation = "IMAGE"
    vi.type.tensor_type.shape.dim[0].denotation = "DATA_BATCH"
    m = model(infos=[vi])
    expected = copy.deepcopy(m.model)
    dims = expected.graph.value_info[0].type.tensor_type.shape.dim
    dims[0].dim_value, dims[1].dim_value = 3, 2
    # what set_tensor_shape writes: the value info made anew, its side fields gone
    replaced = copy.deepcopy(expected)
    replaced.graph.value_info[0].CopyFrom(helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 2]))
    by_setter = copy.deepcopy(m)
    by_setter.set_tensor_shape("x", [3, 2])
    assert saved(by_setter) == replaced.SerializeToString(deterministic=True)
    m.state("x", shape=(3, 2))
    assert saved(m) == expected.SerializeToString(deterministic=True)
    assert saved(m) != replaced.SerializeToString(deterministic=True)


@pytest.mark.parametrize("kind", ["sequence_type", "map_type", "optional_type", "sparse_tensor_type", "opaque_type"])
def test_structural_write_never_switches_a_non_tensor_oneof(kind: str) -> None:
    vi = ValueInfoProto(name="s")
    getattr(vi.type, kind).SetInParent()
    m = model(infos=[vi])
    before = saved(m)
    found = m.facts("s")
    assert isinstance(found, NonTensor) and found.kind == kind
    assert saved(m) == before
    with pytest.raises(TensorFactRefused):
        m.state("s", shape=(1,))
    with pytest.raises(TensorFactRefused):
        m.state("s", container=TensorProto.FLOAT, element=DataType["FLOAT32"])
    with pytest.raises(TensorFactRefused):
        m.unstate("s", "shape")
    assert saved(m) == before


def test_sequence_survives_actual_shape_inference_and_container_refusal() -> None:
    graph = helper.make_graph(
        [helper.make_node("SequenceEmpty", [], ["s"], dtype=TensorProto.FLOAT)],
        "sequence",
        [],
        [helper.make_tensor_sequence_value_info("s", TensorProto.FLOAT, None)],
    )
    m = ModelWrapper(qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 13)]))
    m = m.transform(InferShapes(), cleanup=False)
    onnx.checker.check_model(m.model)
    assert m.facts("s") == NonTensor("sequence_type")
    before = saved(m)
    with pytest.raises(TensorFactRefused):
        m.state("s", container=TensorProto.FLOAT)
    assert saved(m) == before


@pytest.mark.parametrize("tensor_vi", [False, True])
def test_sparse_source_is_not_read_as_a_dense_initializer_or_removed_by_forget(tensor_vi: bool) -> None:
    m = model()
    source = helper.make_sparse_tensor(
        numpy_helper.from_array(np.array([1], np.float32), name="s"),
        numpy_helper.from_array(np.array([2], np.int64)),
        [2, 3],
    )
    m.graph.sparse_initializer.append(source)
    if tensor_vi:
        m.graph.value_info.append(helper.make_tensor_value_info("s", TensorProto.FLOAT, [2, 3]))
    assert m.facts("s") == NonTensor("sparse_tensor_type")
    assert m.get_initializer("s") is None
    before = saved(m)
    with pytest.raises(TensorFactRefused):
        m.state("s", shape=(2, 3))
    assert saved(m) == before
    m.forget("s")
    assert m.get_tensor_valueinfo("s") is None
    assert m.graph.sparse_initializer[0].SerializeToString() == source.SerializeToString()
    assert m.facts("s") == NonTensor("sparse_tensor_type")


@pytest.mark.parametrize("has_range", [False, True])
def test_range_can_be_cleared_without_parsing_an_unnamed_malformed_element(has_range: bool) -> None:
    m = model()
    m.state("x", element=DataType["INT8"], range=Range(-3, 5) if has_range else None)
    annotation = next(x for x in m.graph.quantization_annotation if x.tensor_name == "x")
    entry = next(x for x in annotation.quant_parameter_tensor_names if x.key == "finn_datatype")
    entry.value = "NOT_A_DATATYPE"
    expected = copy.deepcopy(m.model)
    props = expected.graph.quantization_annotation[0].quant_parameter_tensor_names
    for prop in list(props):
        if prop.key.startswith("qonnx.tensor/"):
            props.remove(prop)
    m.unstate("x", "range")
    assert saved(m) == expected.SerializeToString(deterministic=True)
    assert entry.value == "NOT_A_DATATYPE"


@pytest.mark.parametrize("operation", ["facts", "state", "unstate", "forget"])
def test_the_empty_name_is_refused_where_the_setters_admit_it(operation: str) -> None:
    """An absent optional output is named "": the setters annotate it as upstream
    did, while the fact operations refuse it, the model unchanged."""
    m = model()
    m.set_tensor_datatype("", DataType["FLOAT32"])
    assert m.get_tensor_datatype("") == DataType["FLOAT32"]
    before = saved(m)
    call = {
        "facts": lambda: m.facts(""),
        "state": lambda: m.state("", element=DataType["INT8"]),
        "unstate": lambda: m.unstate("", "element"),
        "forget": lambda: m.forget(""),
    }[operation]
    with pytest.raises(TensorFactRefused) as error:
        call()
    assert (error.value.findings[0].owner, error.value.findings[0].code) == ("ModelWrapper." + operation, "fact-invalid")
    assert saved(m) == before


def test_state_refuses_an_unknown_fact() -> None:
    m = model()
    with pytest.raises(TypeError):
        m.state("x", dtype=DataType["INT8"])
    with pytest.raises(TensorFactRefused) as error:
        tensor_facts.state(m, "x", dtype=DataType["INT8"])
    assert (error.value.findings[0].owner, error.value.findings[0].code) == ("ModelWrapper.state", "fact-invalid")
    with pytest.raises(TensorFactRefused) as error:
        m.unstate("x", "dtype")
    assert (error.value.findings[0].owner, error.value.findings[0].code) == ("ModelWrapper.unstate", "fact-invalid")
    assert m.facts("x") == TensorFacts(None, None, None, None, None)


def test_a_container_is_removed_by_state_or_unstate_keeping_the_other_facts() -> None:
    for remove in (lambda m: m.state("x", container=None), lambda m: m.unstate("x", "container")):
        vi = helper.make_tensor_value_info("x", TensorProto.INT8, [2, 3])
        vi.doc_string = "kept"
        m = model(infos=[vi])
        m.state("x", element=DataType["INT4"], range=Range(-2, 3), layout=("N", "C"))
        remove(m)
        info = m.get_tensor_valueinfo("x")
        assert info is not None and not info.type.tensor_type.HasField("elem_type")
        assert info.doc_string == "kept"
        assert m.facts("x") == TensorFacts((2, 3), DataType["INT4"], Range(-2, 3), ("N", "C"), None)
        # absent is not UNDEFINED
        m.state("x", container=0)
        assert m.facts("x") == TensorFacts((2, 3), DataType["INT4"], Range(-2, 3), ("N", "C"), 0)


def test_a_removed_container_is_read_from_the_initializer_again() -> None:
    m = model()
    m.set_initializer("w", np.ones((2, 3), np.float64))
    m.state("w", shape=(2, 3), container=TensorProto.DOUBLE)
    m.state("w", container=None)
    info = m.get_tensor_valueinfo("w")
    assert info is not None and not info.type.tensor_type.HasField("elem_type")
    assert m.facts("w") == TensorFacts((2, 3), None, None, None, TensorProto.DOUBLE)


def test_removing_an_absent_container_writes_nothing() -> None:
    m = model()
    before = saved(m)
    m.state("x", container=None)
    m.unstate("x", "container")
    m.state("x", shape=None, container=None)
    assert saved(m) == before


def test_a_graph_input_without_a_container_does_not_pass_the_checker() -> None:
    m = model()
    m.graph.input.append(helper.make_tensor_value_info("x", TensorProto.FLOAT, [2]))
    onnx.checker.check_model(m.model)
    m.unstate("x", "container")
    with pytest.raises(onnx.checker.ValidationError):
        onnx.checker.check_model(m.model)


def facts_graph() -> ModelWrapper:
    """Tensors of each kind the reader answers: a graph input with a symbolic extent,
    an output with a datatype only, an intermediate with every fact, one without a
    container, one with UNDEFINED, an initializer alone and one with a value info
    stating no container, and a name the model does not know."""
    graph = helper.make_graph(
        [],
        "facts",
        [helper.make_tensor_value_info("i", TensorProto.FLOAT, ["N", 3])],
        [helper.make_tensor_value_info("o", TensorProto.INT8, [3])],
        value_info=[
            helper.make_tensor_value_info("m", TensorProto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("zero", 0, [2]),
        ],
        initializer=[
            numpy_helper.from_array(np.ones((2, 2), np.float32), "w"),
            numpy_helper.from_array(np.ones((3,), np.int64), "v"),
        ],
    )
    m = ModelWrapper(qonnx_make_model(graph), fix_missing_initializer_valueinfo=False)
    untyped = ValueInfoProto(name="untyped")
    untyped.type.tensor_type.shape.dim.add(dim_value=5)
    v = ValueInfoProto(name="v")
    v.type.tensor_type.SetInParent()
    m.graph.value_info.extend([untyped, v])
    m.set_tensor_datatype("o", DataType["INT4"])
    m.set_tensor_datatype("m", DataType["UINT8"], range=Range(1, 200))
    m.set_tensor_layout("m", ["N", "C"])
    return m


@pytest.mark.parametrize("name", ["i", "o", "m", "zero", "untyped", "w", "v", "absent"])
def test_state_admits_the_facts_the_reader_answers(name: str) -> None:
    m = facts_graph()
    facts = m.facts(name)
    assert isinstance(facts, TensorFacts)
    m.state(name, **{field.name: getattr(facts, field.name) for field in dataclasses.fields(facts)})
    assert m.facts(name) == facts


@pytest.mark.parametrize("interface", ["input", "output"])
def test_a_held_graph_input_or_output_stays_the_graph_s_and_shows_new_facts(interface: str) -> None:
    m = model()
    vi = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    vi.doc_string = "kept"
    getattr(m.graph, interface).extend([helper.make_tensor_value_info("a", TensorProto.FLOAT, [1]), vi])
    held = m.get_tensor_valueinfo("x")
    assert held is not None
    m.state("x", shape=(2,), container=TensorProto.INT8)
    members = getattr(m.graph, interface)
    assert [x.name for x in members] == ["a", "x"]
    assert held.type.tensor_type.shape.dim[0].dim_value == 2
    assert held.type.tensor_type.elem_type == TensorProto.INT8
    assert held.doc_string == "kept"
    m.unstate("x", "shape")
    assert not held.type.tensor_type.HasField("shape")
    # still the graph's: a change through it is the graph's
    held.doc_string = "changed"
    assert members[1].doc_string == "changed"


def test_an_intermediate_value_info_is_replaced_and_appended() -> None:
    m = model(infos=[helper.make_tensor_value_info(x, TensorProto.FLOAT, [1]) for x in ("x", "y")])
    m.state("x", shape=(2,))
    assert [x.name for x in m.graph.value_info] == ["y", "x"]
    assert m.facts("x").shape == (2,)
