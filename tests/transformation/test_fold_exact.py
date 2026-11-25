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

"""Constant folding by its own exact evaluation: a standard node is evaluated with exact
integers on whatever the caller's setting, left unfolded where exact integers refuse it
(a bound on its products and partial sums past 2**53, a Cast that loses a value, an
operator with no proof given or computing a value past what its container holds
exactly) or where its output's stated type cannot hold the value exactly, and folding
goes on past it. An operator whose attributes fix an element type folds in its own
types. Every other node folds as before (see also test_fold_operators)."""

from __future__ import annotations

import pytest

import numpy as np
from onnx import TensorProto, helper, numpy_helper
from typing import Any, Callable, Mapping

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import exact_integers, execute_onnx
from qonnx.transformation.base import Transformation
from qonnx.transformation.fold_constants import FoldConstants, FoldConstantsFiltered
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import qonnx_make_model

F = TensorProto.FLOAT
D = TensorProto.DOUBLE
FOLDS: list[Callable[[], Transformation]] = [FoldConstants, lambda: FoldConstantsFiltered(lambda model, node: True)]
SETTINGS = [False, True]


def model(
    nodes: list[Any],
    constants: Mapping[str, np.ndarray],
    *,
    annotation: str | None = None,
    kind: int = F,
    intermediates: Mapping[str, list[int]] | None = None,
) -> ModelWrapper:
    """A graph from x to y, every tensor held in kind, the constants initializers and
    every tensor annotated where annotation is given."""

    def tensor(name: str, shape: list[int]) -> Any:
        return helper.make_tensor_value_info(name, kind, shape)

    infos = {name: list(value.shape) for name, value in constants.items()}
    infos.update(intermediates or {"c": [1]})
    graph = helper.make_graph(
        nodes,
        "fold_exact",
        [tensor("x", [1])],
        [tensor("y", [1])],
        initializer=[numpy_helper.from_array(value, name) for name, value in constants.items()],
        value_info=[tensor(name, shape) for name, shape in infos.items()],
    )
    wrapper = ModelWrapper(qonnx_make_model(graph))
    if annotation is not None:
        for name in [*infos, "x", "y"]:
            wrapper.set_tensor_datatype(name, DataType[annotation])
    return wrapper


def constant_sum(a: int, b: int, annotation: str | None = "INT32", kind: int = F) -> ModelWrapper:
    """c = a + b, both constants, then y = x + c."""
    container = np.float64 if kind == D else np.float32
    return model(
        [
            helper.make_node("Add", ["a", "b"], ["c"], name="Add_0"),
            helper.make_node("Add", ["x", "c"], ["y"], name="Add_1"),
        ],
        {"a": np.array([a], container), "b": np.array([b], container)},
        annotation=annotation,
        kind=kind,
    )


def folded(graph: ModelWrapper, fold: Callable[[], Transformation], setting: bool) -> ModelWrapper:
    with exact_integers(setting):
        return graph.transform(fold())


@pytest.mark.parametrize("setting", SETTINGS)
@pytest.mark.parametrize("fold", FOLDS)
def test_a_sum_float32_cannot_hold_is_left_unfolded_and_folding_goes_on(fold: Any, setting: bool) -> None:
    """The INT32 sum 16777216 + 1 stays a node, with exact integers off and on; an
    independent constant node after it still folds."""
    graph = constant_sum(2**24, 1)
    graph.graph.node.append(helper.make_node("Add", ["a", "a"], ["d"], name="Add_2"))
    graph.graph.node.append(helper.make_node("Add", ["x", "d"], ["e"], name="Add_3"))
    graph.graph.value_info.append(helper.make_tensor_value_info("d", F, [1]))
    graph.graph.output.append(helper.make_tensor_value_info("e", F, [1]))
    result = folded(graph, fold, setting)
    assert sorted(n.name for n in result.graph.node) == ["Add_0", "Add_1", "Add_3"]
    assert result.get_initializer("c") is None
    assert result.get_initializer("d").item() == 2**25
    x = {"x": np.zeros(1, np.float32)}
    assert execute_onnx(result, x, exact_integers=True)["y"].item() == 2**24 + 1


@pytest.mark.parametrize("setting", SETTINGS)
@pytest.mark.parametrize("fold", FOLDS)
def test_a_sum_float32_holds_folds_to_the_same_bytes(fold: Any, setting: bool) -> None:
    for a, b in ((16777214, 1), (2**24, 2)):
        c = folded(constant_sum(a, b), fold, setting).get_initializer("c")
        assert c.dtype == np.float32 and c.tobytes() == np.array([a + b], np.float32).tobytes()


@pytest.mark.parametrize("setting", SETTINGS)
def test_a_matmul_float32_arithmetic_would_round_is_stored_exact(setting: bool) -> None:
    """A dot product of constants whose partial sums pass 2**24 but whose result float32
    holds: folded in float64, stored in float32, exact."""
    graph = model(
        [
            helper.make_node("MatMul", ["a", "b"], ["c"], name="MatMul_0"),
            helper.make_node("Add", ["x", "c"], ["y"], name="Add_0"),
        ],
        {"a": np.array([[2**24, 1, -(2**24)]], np.float32), "b": np.ones((3, 1), np.float32)},
        intermediates={"c": [1, 1]},
    )
    graph.set_tensor_shape("x", [1, 1])
    graph.set_tensor_shape("y", [1, 1])
    graph.set_tensor_datatype("a", DataType["INT26"])
    graph.set_tensor_datatype("b", DataType["INT2"])
    graph.set_tensor_datatype("x", DataType["INT8"])
    graph = graph.transform(InferDataTypes())
    assert graph.get_tensor_datatype("c").max() > 2**24
    # ONNX Runtime's float32 MatMul rounds 16777216 + 1 on the way.
    assert execute_onnx(graph, {"x": np.zeros((1, 1), np.float32)})["y"].item() != 1
    c = folded(graph, FoldConstants, setting).get_initializer("c")
    assert c.dtype == np.float32 and c.item() == 1


@pytest.mark.parametrize("fold", FOLDS)
@pytest.mark.parametrize("a,kind", [(2**53, F), (2**53 + 2, D)])
def test_a_node_past_float64s_exact_integers_is_left_unfolded(fold: Any, a: int, kind: int) -> None:
    """A sum whose bound passes 2**53: exact integers refuse the node, so its exactness
    is unproved; folding would store 2**53 for 2**53 + 1 in float32, and 2**53 + 4 for
    2**53 + 3 in float64."""
    graph = constant_sum(a, 1, "INT64", kind)
    result = folded(graph, fold, False)
    assert result.get_initializer("c") is None
    assert [n.name for n in result.graph.node] == ["Add_0", "Add_1"]


def product(op: str, a: list[list[int]], b: list[list[int]], annotation: str = "INT32") -> ModelWrapper:
    """c = op(a, b) of constants, then y = x + c; every tensor held in float32 and
    annotated."""
    graph = model(
        [helper.make_node(op, ["a", "b"], ["c"], name=f"{op}_0"), helper.make_node("Add", ["x", "c"], ["y"], name="Add_0")],
        {"a": np.array(a, np.float32), "b": np.array(b, np.float32)},
        annotation=annotation,
        intermediates={"c": [1, 1]},
    )
    graph.set_tensor_shape("x", [1, 1])
    graph.set_tensor_shape("y", [1, 1])
    return graph


@pytest.mark.parametrize("setting", SETTINGS)
@pytest.mark.parametrize("fold", FOLDS)
@pytest.mark.parametrize("op,a,b", [("Pow", 3, 40), ("Div", 2**24 + 2, 2), ("Div", 2**25, 2**25)])
def test_an_operator_without_a_proof_past_float32s_exact_integers_is_left_unfolded(
    op: str, a: int, b: int, fold: Any, setting: bool
) -> None:
    """A result (3**40) or a given value (2**24 + 2, 2**25) past 2**24; an independent
    constant node after it still folds."""
    graph = product(op, [[a]], [[b]])
    graph.graph.node.append(helper.make_node("Add", ["b", "b"], ["d"], name="Add_2"))
    graph.graph.value_info.append(helper.make_tensor_value_info("d", F, [1, 1]))
    graph.graph.node.append(helper.make_node("Add", ["x", "d"], ["e"], name="Add_3"))
    graph.graph.output.append(helper.make_tensor_value_info("e", F, [1, 1]))
    result = folded(graph, fold, setting)
    assert result.get_initializer("c") is None
    assert result.get_initializer("d").item() == 2 * b


@pytest.mark.parametrize("setting", SETTINGS)
@pytest.mark.parametrize("fold", FOLDS)
@pytest.mark.parametrize("op", ["MatMul", "Gemm"])
def test_a_cancellation_whose_products_pass_2_to_the_53_is_left_unfolded(op: str, fold: Any, setting: bool) -> None:
    """2**30 * 2**30 + 1 - 2**30 * 2**30 is 1, but float64 loses the 1 beside 2**60."""
    k = 2**30
    result = folded(product(op, [[k, 1, k]], [[k], [1], [-k]]), fold, setting)
    assert result.get_initializer("c") is None
    assert [n.name for n in result.graph.node] == [f"{op}_0", "Add_0"]


@pytest.mark.parametrize("setting", SETTINGS)
@pytest.mark.parametrize("fold", FOLDS)
@pytest.mark.parametrize("op", ["MatMul", "Gemm"])
def test_a_cancellation_whose_bound_float64_holds_folds_to_exact_bytes(op: str, fold: Any, setting: bool) -> None:
    k = 2**25
    c = folded(product(op, [[k, 1, k]], [[k], [1], [-k]]), fold, setting).get_initializer("c")
    assert c.dtype == np.float32 and c.tobytes().hex() == "0000803f"


def summed(op: str, values: list[int], kind: int) -> ModelWrapper:
    """c = values summed by op (a dot product by ones, or ReduceSum), then y = x + c;
    every tensor held in kind and annotated INT55."""
    container = np.float64 if kind == D else np.float32
    if op == "MatMul":
        node = helper.make_node("MatMul", ["a", "b"], ["c"], name="MatMul_0")
        constants = {"a": np.array(values, container), "b": np.ones((len(values), 1), container)}
    else:
        node = helper.make_node("ReduceSum", ["a"], ["c"], name="ReduceSum_0", axes=[0], keepdims=1)
        constants = {"a": np.array(values, container)}
    return model([node, helper.make_node("Add", ["x", "c"], ["y"], name="Add_0")], constants, annotation="INT55", kind=kind)


@pytest.mark.parametrize("fold", FOLDS)
@pytest.mark.parametrize("op", ["MatMul", "ReduceSum"])
@pytest.mark.parametrize(
    "values,kind,expected",
    [
        # A bound of 2**24: computed and stored in float32.
        ([2**23, 2**23], F, 2**24),
        # One past: computed in float64, stored in float32 (a float32 MatMul gives 2**24 - 1).
        ([2**23, 2**23, 1, -1], F, 2**24),
        # A bound of 2**53: computed in float64.
        ([2**52, 2**52 - 1, 1], D, 2**53),
        # Past it: refused, so left unfolded (a float64 MatMul gives 2**53 - 1).
        ([2**52, 2**52, 1, -1], D, None),
    ],
)
def test_both_sides_of_2_to_the_24_and_2_to_the_53(
    op: str, fold: Any, values: list[int], kind: int, expected: int | None
) -> None:
    c = folded(summed(op, values, kind), fold, False).get_initializer("c")
    if expected is None:
        assert c is None
    else:
        container = np.float64 if kind == D else np.float32
        assert c.dtype == container and c.tobytes() == np.array([expected], container).tobytes()


def cast(value: int, kind: int) -> ModelWrapper:
    """c = Cast(a, to=FLOAT) of a constant held in kind, then y = x + c; both Cast tensors
    annotated INT32."""
    graph = helper.make_graph(
        [
            helper.make_node("Cast", ["a"], ["c"], name="Cast_0", to=F),
            helper.make_node("Add", ["x", "c"], ["y"], name="Add_0"),
        ],
        "cast",
        [helper.make_tensor_value_info("x", F, [1])],
        [helper.make_tensor_value_info("y", F, [1])],
        initializer=[numpy_helper.from_array(np.array([value], helper.tensor_dtype_to_np_dtype(kind)), "a")],
        value_info=[helper.make_tensor_value_info("a", kind, [1]), helper.make_tensor_value_info("c", F, [1])],
    )
    wrapper = ModelWrapper(qonnx_make_model(graph))
    for name in ("a", "c"):
        wrapper.set_tensor_datatype(name, DataType["INT32"])
    return wrapper


@pytest.mark.parametrize("setting", SETTINGS)
@pytest.mark.parametrize("fold", FOLDS)
@pytest.mark.parametrize("kind", [F, TensorProto.INT32])
def test_a_cast_folds_in_its_own_types(kind: int, fold: Any, setting: bool) -> None:
    c = folded(cast(2**24, kind), fold, setting).get_initializer("c")
    assert c.dtype == np.float32 and c.tobytes().hex() == "0000804b"


@pytest.mark.parametrize("setting", SETTINGS)
@pytest.mark.parametrize("fold", FOLDS)
def test_a_cast_that_does_not_keep_every_value_is_left_unfolded(fold: Any, setting: bool) -> None:
    result = folded(cast(2**24 + 1, TensorProto.INT32), fold, setting)
    assert result.get_initializer("c") is None
    assert [n.name for n in result.graph.node] == ["Cast_0", "Add_0"]


@pytest.mark.parametrize("setting", SETTINGS)
@pytest.mark.parametrize("fold", FOLDS)
def test_a_constant_of_shape_folds_in_its_own_types(fold: Any, setting: bool) -> None:
    value = helper.make_tensor("value", F, [1], [2**24])
    graph = helper.make_graph(
        [
            helper.make_node("ConstantOfShape", ["shape"], ["c"], name="ConstantOfShape_0", value=value),
            helper.make_node("Add", ["x", "c"], ["y"], name="Add_0"),
        ],
        "fill",
        [helper.make_tensor_value_info("x", F, [1])],
        [helper.make_tensor_value_info("y", F, [1])],
        initializer=[numpy_helper.from_array(np.array([1], np.int64), "shape")],
        value_info=[helper.make_tensor_value_info("c", F, [1])],
    )
    wrapper = ModelWrapper(qonnx_make_model(graph))
    wrapper.set_tensor_datatype("c", DataType["INT32"])
    c = folded(wrapper, fold, setting).get_initializer("c")
    assert c.dtype == np.float32 and c.tobytes().hex() == "0000804b"


@pytest.mark.parametrize("fold", FOLDS)
def test_an_integer_annotated_node_with_a_subgraph_is_left_unfolded(fold: Any) -> None:
    """An If whose result, 2**25, passes what float32 holds exactly."""
    branch = helper.make_graph(
        [helper.make_node("Constant", [], ["value"], value=helper.make_tensor("value", F, [1], [2**25]))],
        "branch",
        [],
        [helper.make_tensor_value_info("value", F, [1])],
    )
    graph = model(
        [
            helper.make_node("If", ["condition"], ["c"], name="If_0", then_branch=branch, else_branch=branch),
            helper.make_node("Add", ["x", "c"], ["y"], name="Add_0"),
        ],
        {},
        annotation="INT26",
    )
    graph.set_initializer("condition", np.array(True))
    result = folded(graph, fold, False)
    assert result.get_initializer("c") is None
    assert [n.name for n in result.graph.node] == ["If_0", "Add_0"]


@pytest.mark.parametrize("fold", FOLDS)
@pytest.mark.parametrize("annotation", [None, "INT8", "INT25"])
def test_a_node_unannotated_or_within_float32_folds_as_before(fold: Any, annotation: str | None) -> None:
    """Unannotated float inputs, and integers float32 holds exactly however wide their
    annotation, fold as before, in float32 (16777216 + 1 rounded where nothing states
    integers)."""
    a, b = (2**24, 1) if annotation is None else (100, 27)
    c = folded(constant_sum(a, b, annotation), fold, False).get_initializer("c")
    assert c.dtype == np.float32 and c.tobytes() == (np.array([a], np.float32) + np.array([b], np.float32)).tobytes()


@pytest.mark.parametrize("fold", FOLDS)
def test_an_integer_container_folds_as_before(fold: Any) -> None:
    graph = model(
        [
            helper.make_node("Add", ["a", "b"], ["c"], name="Add_0"),
            helper.make_node("Add", ["x", "c"], ["y"], name="Add_1"),
        ],
        {"a": np.array([2**30], np.int32), "b": np.array([3], np.int32)},
        annotation="INT32",
        kind=TensorProto.INT32,
    )
    c = folded(graph, fold, False).get_initializer("c")
    assert c.dtype == np.int32 and c.item() == 2**30 + 3


def test_a_custom_node_folds_with_the_callers_setting() -> None:
    """A MultiThreshold whose out_dtype cannot hold every count folds with exact integers
    off, and is refused with them on."""
    node = helper.make_node(
        "MultiThreshold", ["a", "t"], ["c"], domain="qonnx.custom_op.general", name="threshold", out_dtype="UINT1"
    )
    graph = model(
        [node, helper.make_node("Add", ["x", "c"], ["y"], name="dynamic")],
        {"a": np.array([[-1]], np.float32), "t": np.zeros((1, 2), np.float32)},
        intermediates={"c": [1, 1]},
    )
    graph.set_tensor_shape("x", [1, 1])
    graph.set_tensor_shape("y", [1, 1])
    for fold in FOLDS:
        assert folded(graph, fold, False).get_initializer("c").tolist() == [[0]]
        with pytest.raises(ValueError, match="out_dtype BINARY cannot hold 2"):
            folded(graph, fold, True)


@pytest.mark.parametrize("value,expected", [(np.nan, "0000c07f"), (np.inf, "0000807f"), (-0.0, "00000080")])
def test_same_dtype_ieee_values_keep_their_stored_bytes(value: float, expected: str) -> None:
    graph = model(
        [
            helper.make_node("Identity", ["a"], ["c"], name="constant"),
            helper.make_node("Add", ["x", "c"], ["y"], name="dynamic"),
        ],
        {"a": np.array([value], np.float32)},
    )
    stored = graph.transform(FoldConstants()).get_initializer("c")
    assert stored.dtype == np.float32 and stored.tobytes().hex() == expected


def test_a_custom_output_keeps_float64_and_a_missing_value_info() -> None:
    for value, expected in (
        (np.array([16777217], np.float64), "0000001000007041"),
        (np.array([3], np.float32), "00004040"),
    ):
        marker = helper.make_node(
            "DebugMarker", ["a"], ["c"], domain="qonnx.custom_op.general", name="marker", export_debug_name="exported"
        )
        graph = model([marker, helper.make_node("Add", ["x", "c"], ["y"], name="dynamic")], {"a": value})
        if value.dtype == np.float64:
            graph.get_tensor_valueinfo("a").type.tensor_type.elem_type = D
        else:
            graph.graph.value_info.remove(graph.get_tensor_valueinfo("c"))
        stored = graph.transform(FoldConstants()).get_initializer("c")
        assert stored.dtype == value.dtype and stored.tobytes().hex() == expected


@pytest.mark.parametrize("excluded", [False, True])
def test_folding_keeps_dynamic_inputs_and_exclusions(excluded: bool) -> None:
    graph = model(
        [
            helper.make_node("Add", ["x", "a"], ["c"], name="dynamic"),
            helper.make_node("Add", ["c", "b"], ["y"], name="after"),
        ],
        {"a": np.array([3], np.float32), "b": np.array([1], np.float32)},
    )
    result = graph.transform(FoldConstants(exclude_op_types=["Add"] if excluded else []))
    assert result.get_initializer("c") is None and len(result.graph.node) == 2
    assert execute_onnx(result, {"x": np.array([7], np.float32)})["y"].item() == 11
    constant = model(
        [
            helper.make_node("Add", ["a", "b"], ["c"], name="constant"),
            helper.make_node("Add", ["x", "c"], ["y"], name="dynamic"),
        ],
        {"a": np.array([3], np.float32), "b": np.array([1], np.float32)},
    )
    result = constant.transform(FoldConstants(exclude_op_types=["Add"] if excluded else []))
    assert (result.get_initializer("c") is None) == excluded
    assert execute_onnx(result, {"x": np.array([7], np.float32)})["y"].item() == 11


@pytest.mark.parametrize("kind", ["DebugMarker", "Quant"])
@pytest.mark.parametrize("fold", [lambda: FoldConstants([]), FOLDS[1]])
def test_a_missing_secondary_custom_output_keeps_the_first_output_folding(fold: Any, kind: str) -> None:
    constants = {"a": np.array([3], np.float32)}
    if kind == "DebugMarker":
        node = helper.make_node(
            "DebugMarker", ["a"], ["c", "extra"], domain="qonnx.custom_op.general", name="marker", export_debug_name="x0"
        )
    else:
        constants.update({name: np.asarray(value, np.float32) for name, value in (("s", 1), ("zero", 0), ("bits", 4))})
        node = helper.make_node(
            "Quant",
            ["a", "s", "zero", "bits"],
            ["c", "extra"],
            domain="qonnx.custom_op.general",
            name="extra_output",
            signed=1,
            narrow=0,
            rounding_mode="ROUND",
        )
    result = model([node, helper.make_node("Add", ["x", "c"], ["y"], name="dynamic")], constants).transform(fold())
    stored = result.get_initializer("c")
    assert stored is not None and stored.tobytes().hex() == "00004040"
    assert result.get_initializer("extra") is None
    assert execute_onnx(result, {"x": np.zeros(1, np.float32)})["y"].item() == 3


@pytest.mark.parametrize("fold", FOLDS)
def test_a_dynamic_shape_is_folded_from_its_stated_dimensions(fold: Any) -> None:
    graph = helper.make_graph(
        [
            helper.make_node("Shape", ["x"], ["c"], name="shape"),
            helper.make_node("Add", ["z", "c"], ["y"], name="dynamic"),
        ],
        "constant_shape",
        [
            helper.make_tensor_value_info("x", F, [2, 3]),
            helper.make_tensor_value_info("z", TensorProto.INT64, [2]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.INT64, [2])],
        value_info=[helper.make_tensor_value_info("c", TensorProto.INT64, [2])],
    )
    result = ModelWrapper(qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 13)])).transform(fold())
    stored = result.get_initializer("c")
    assert stored is not None and stored.dtype == np.int64
    assert stored.tobytes().hex() == "02000000000000000300000000000000"


def test_filtered_folding_keeps_a_markers_publication_under_its_export_name() -> None:
    """The marker publishes a's value under c in the one context filtered folding shares,
    so the sum after it reads 3, not c's initializer 10, as execution of the graph does."""
    marker = helper.make_node(
        "DebugMarker", ["a"], ["marked"], domain="qonnx.custom_op.general", name="marker", export_debug_name="c"
    )
    graph = model(
        [
            marker,
            helper.make_node("Add", ["c", "d"], ["s"], name="sum"),
            helper.make_node("Add", ["x", "s"], ["y"], name="dynamic"),
        ],
        {name: np.array([value], np.float32) for name, value in (("a", 3), ("c", 10), ("d", 1))},
        intermediates={"marked": [1], "s": [1]},
    )
    assert execute_onnx(graph, {"x": np.zeros(1, np.float32)})["y"].item() == 4
    result = graph.transform(FOLDS[1]())
    stored = result.get_initializer("s")
    assert stored is not None and stored.tobytes().hex() == "00008040"
    assert graph.get_initializer("c").item() == 10
    assert execute_onnx(result, {"x": np.zeros(1, np.float32)})["y"].item() == 4
