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

"""Exact integer execution, an opt-in switched by the exact_integers context and the
exact_integers= arguments: a standard node whose float tensors are all integer-annotated,
given integers, computes by its operator and the values given (integer arithmetic shown
exact by a bound on its products and partial sums or refused, an operator that moves
values exact as its inputs, any other with no proof computed as without exact integers
within what its container holds exactly and refused past it), an operator whose
attributes fix an element type is never widened, and MultiThreshold computes its count
and affine result exactly, refusing an out_dtype or a container that cannot hold every
result."""

from __future__ import annotations

import pytest

import numpy as np
import onnxruntime as rt
from onnx import TensorProto, helper
from pathlib import Path
from typing import Any

import qonnx.core.onnx_exec as onnx_exec
from qonnx.core.datatype import DataType
from qonnx.core.execute_custom_node import execute_custom_node
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import (
    exact_integers,
    exact_integers_enabled,
    execute_node,
    execute_onnx,
    execute_onnx_and_make_model,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import qonnx_make_model

GENERAL = "qonnx.custom_op.general"
K = 2304
F = TensorProto.FLOAT


def single(node: Any, shape: list[int], output: list[int] | None = None, kind: int = F) -> ModelWrapper:
    """A model of one node from x to y, both held in kind (y in FLOAT)."""
    graph = helper.make_graph(
        [node],
        "single",
        [helper.make_tensor_value_info("x", kind, shape)],
        [helper.make_tensor_value_info("y", F, shape if output is None else output)],
    )
    return ModelWrapper(qonnx_make_model(graph))


def identity(annotation: str) -> ModelWrapper:
    """An Identity from x to y, both held in float32 and annotated."""
    model = single(helper.make_node("Identity", ["x"], ["y"], name="identity"), [1])
    for name in ("x", "y"):
        model.set_tensor_datatype(name, DataType[annotation])
    return model


def wide_weights() -> np.ndarray:
    """INT8 weights over k = 2304, spread over their whole range."""
    return np.random.default_rng(K).integers(-128, 128, size=(K, 3)).astype(np.float32)


def wide_input(weights: np.ndarray) -> np.ndarray:
    """An INT8 input whose first row's sums reach past 2**24, the second's first an odd
    integer there, which float32 cannot hold."""
    x = (127 * np.where(weights[:, :1].T > 0, 1, -1)).repeat(2, axis=0).astype(np.int64)
    x[1, 0] -= 1
    return x


def wide_layer(weights: np.ndarray | None = None) -> ModelWrapper:
    """A MatMul over k = 2304 of INT8 values, then a float Mul by 0.5; every tensor held in
    float32."""
    weights = wide_weights() if weights is None else weights
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "w"], ["s"], name="MatMul_0"),
            helper.make_node("Mul", ["s", "half"], ["scaled"], name="Mul_0"),
        ],
        "wide",
        [helper.make_tensor_value_info("x", F, [2, K])],
        [helper.make_tensor_value_info("s", F, [2, 3]), helper.make_tensor_value_info("scaled", F, [2, 3])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_initializer("w", weights)
    model.set_initializer("half", np.asarray(0.5, dtype=np.float32))
    model.set_tensor_datatype("x", DataType["INT8"])
    model.set_tensor_datatype("w", DataType["INT8"])
    return model.transform(InferDataTypes())


def partition(body: ModelWrapper, directory: Path) -> ModelWrapper:
    """A GenericPartition from x to y running body."""
    path = directory / "body.onnx"
    body.save(str(path))
    node = helper.make_node("GenericPartition", ["x"], ["y"], name="partition", domain=GENERAL, model=str(path))
    return single(node, [1])


def test_the_setting_follows_the_context_and_its_nesting() -> None:
    assert exact_integers_enabled() is False
    with exact_integers():
        assert exact_integers_enabled() is True
        with exact_integers(False):
            assert exact_integers_enabled() is False
            with exact_integers(True):
                assert exact_integers_enabled() is True
            assert exact_integers_enabled() is False
        assert exact_integers_enabled() is True
    assert exact_integers_enabled() is False
    with pytest.raises(RuntimeError):
        with exact_integers():
            raise RuntimeError("left by an exception")
    assert exact_integers_enabled() is False


@pytest.mark.parametrize("setting", [1, "yes", np.bool_(True)])
def test_the_setting_is_a_bool(setting: Any) -> None:
    with pytest.raises(TypeError, match="bool"):
        with exact_integers(setting):
            pass
    with pytest.raises(TypeError, match="bool"):
        execute_onnx(identity("INT8"), {"x": np.ones(1, np.float32)}, exact_integers=setting)
    assert exact_integers_enabled() is False


def test_a_wide_matmul_is_exact_with_the_setting_and_rounds_without() -> None:
    weights = wide_weights()
    model = wide_layer(weights)
    x = wide_input(weights)
    exact = x @ weights.astype(np.int64)
    assert np.abs(exact).max() > 2**24 and exact[1, 0] % 2 == 1
    assert model.get_tensor_datatype("s").is_integer()

    rounded = execute_onnx(model, {"x": x.astype(np.float32)})["s"]
    assert rounded.dtype == np.float32
    assert not np.array_equal(rounded.astype(np.int64), exact)

    widened = execute_onnx(model, {"x": x.astype(np.float32)}, exact_integers=True)["s"]
    assert widened.dtype == np.float64
    assert np.array_equal(widened.astype(np.int64), exact)
    with exact_integers():
        # The caller may give the integers as they are: the node casts what it reads.
        given = execute_onnx(model, {"x": x})["s"]
        assert np.array_equal(given.astype(np.int64), exact)
        # An argument of False switches them off inside the context.
        off = execute_onnx(model, {"x": x.astype(np.float32)}, exact_integers=False)["s"]
        assert off.dtype == np.float32 and np.array_equal(off, rounded)
    assert exact_integers_enabled() is False


def test_a_float_node_reads_a_widened_tensor_back_in_float32() -> None:
    weights = wide_weights()
    model = wide_layer(weights)
    x = wide_input(weights)
    exact = x @ weights.astype(np.int64)
    out = execute_onnx(model, {"x": x.astype(np.float32)}, return_full_exec_context=True, exact_integers=True)
    assert out["s"].dtype == np.float64
    # Mul is float arithmetic (0.5 is no integer): it reads the exact sums in float32.
    assert out["scaled"].dtype == np.float32
    assert np.array_equal(out["scaled"], exact.astype(np.float32) * np.float32(0.5))


def test_a_widened_tensor_is_read_back_in_float32_by_the_next_node() -> None:
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["x"], ["h"], name="widen"),
            helper.make_node("Add", ["h", "zero"], ["y"], name="narrow"),
        ],
        "widen_then_narrow",
        [helper.make_tensor_value_info("x", F, [1])],
        [helper.make_tensor_value_info("y", F, [1])],
        value_info=[helper.make_tensor_value_info("h", F, [1])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_initializer("zero", np.zeros(1, np.float32))
    for name in ("x", "h"):
        model.set_tensor_datatype(name, DataType["INT26"])
    context = execute_onnx(model, {"x": np.array([16777217], np.int64)}, True, exact_integers=True)
    assert context["h"].dtype == np.float64 and context["h"].item() == 16777217
    assert context["y"].dtype == np.float32 and context["y"].item() == 16777216


def arithmetic(
    op: str, arrays: dict[str, np.ndarray], annotation: str, output: list[int], kind: int = F, **attributes: Any
) -> ModelWrapper:
    """A model of one op over the named inputs, held in kind and given by the caller,
    to y; every tensor annotated."""
    graph = helper.make_graph(
        [helper.make_node(op, list(arrays), ["y"], name=f"{op}_0", **attributes)],
        op.lower(),
        [helper.make_tensor_value_info(name, kind, list(value.shape)) for name, value in arrays.items()],
        [helper.make_tensor_value_info("y", kind, output)],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    for name in [*arrays, "y"]:
        model.set_tensor_datatype(name, DataType[annotation])
    return model


k = 2**30
CANCELLING = {"a": np.array([[k, 1, k]], np.float32), "b": np.array([[k], [1], [-k]], np.float32)}


@pytest.mark.parametrize("op", ["MatMul", "Gemm"])
def test_a_cancellation_whose_products_pass_2_to_the_53_is_refused(op: str) -> None:
    model = arithmetic(op, CANCELLING, "INT32", [1, 1])
    with pytest.raises(ArithmeticError, match=rf"{op}_0 \({op}\): a bound .* {2**61 + 1}, passes 2\*\*53"):
        execute_onnx(model, CANCELLING, exact_integers=True)
    # Off, the products round as they always did.
    assert execute_onnx(model, CANCELLING)["y"].tolist() == [[0.0]]


def test_a_double_container_is_bounded_too() -> None:
    arrays = {"a": np.array([[2**52, 1, 2**52]], np.float64), "b": np.array([[1], [1], [-1]], np.float64)}
    model = arithmetic("MatMul", arrays, "INT54", [1, 1], kind=TensorProto.DOUBLE)
    with pytest.raises(ArithmeticError, match=r"passes 2\*\*53"):
        execute_onnx(model, arrays, exact_integers=True)


def dot(values: list[int]) -> tuple[ModelWrapper, dict[str, np.ndarray]]:
    """A MatMul of values by ones, INT55-annotated, and its inputs as integers."""
    arrays = {"a": np.array([values], np.int64), "b": np.ones((len(values), 1), np.int64)}
    model = arithmetic("MatMul", {name: value.astype(np.float32) for name, value in arrays.items()}, "INT55", [1, 1])
    return model, arrays


def total(values: list[int]) -> tuple[ModelWrapper, dict[str, np.ndarray]]:
    """A ReduceSum of values over their one axis, INT55-annotated, and its input."""
    arrays = {"x": np.array([values], np.int64)}
    model = arithmetic("ReduceSum", {"x": arrays["x"].astype(np.float32)}, "INT55", [1], keepdims=0)
    # Axes are an input from opset 13.
    model.model.opset_import[0].version = 13
    model.set_initializer("axes", np.array([1], np.int64))
    model.graph.node[0].input.append("axes")
    return model, arrays


@pytest.mark.parametrize("build", [dot, total], ids=["dot", "reduce_sum"])
def test_both_sides_of_2_to_the_24_and_2_to_the_53(build: Any) -> None:
    # A bound float32 holds: the node computes in it, exactly.
    model, arrays = build([2**23, 2**23])
    y = execute_onnx(model, arrays, exact_integers=True)["y"]
    assert y.dtype == np.float32 and y.reshape(-1).tolist() == [2**24]
    # One past: float64, exactly.
    model, arrays = build([2**23, 2**23 + 1])
    y = execute_onnx(model, arrays, exact_integers=True)["y"]
    assert y.dtype == np.float64 and y.reshape(-1).tolist() == [2**24 + 1]
    # A bound of 2**53 float64 holds.
    model, arrays = build([2**52, 2**52 - 1, 1])
    y = execute_onnx(model, arrays, exact_integers=True)["y"]
    assert y.dtype == np.float64 and int(y.reshape(-1)[0]) == 2**53
    # One past is refused, though the sum itself is small.
    model, arrays = build([2**52, 2**52 + 1, -(2**52)])
    with pytest.raises(ArithmeticError, match=rf"{2**53 + 2**52 + 1}, passes 2\*\*53"):
        execute_onnx(model, arrays, exact_integers=True)


@pytest.mark.parametrize("given", [[0.5, 2**30], [float("inf"), 1]], ids=["fraction", "infinity"])
def test_a_node_given_a_value_that_is_no_integer_computes_as_off(given: list[float]) -> None:
    # Products past 2**53 would be refused, were every value an integer.
    model, _ = dot([2**30, 2**30])
    inputs = {"a": np.array([given], np.float32), "b": np.array([[1], [2**30]], np.float32)}
    results = {}
    for setting in (False, True):
        context = model.make_empty_exec_context()
        context.update(inputs)
        execute_node(model.graph.node[0], context, model.graph, 13, model=model, exact_integers=setting)
        results[setting] = context["y"]
    assert results[True].dtype == np.float32 and results[True].tobytes() == results[False].tobytes()


def run(model: ModelWrapper, arrays: dict[str, np.ndarray], setting: bool) -> np.ndarray:
    """y of the model's one node given arrays, with exact integers switched by setting
    (no rounding to the annotations)."""
    context = model.make_empty_exec_context()
    context.update(arrays)
    execute_node(model.graph.node[0], context, model.graph, 13, model=model, exact_integers=setting)
    return context["y"]


SCALED = {"a": np.array([[2, 4]], np.float32), "b": np.array([[1], [1]], np.float32), "c": np.array([[1]], np.float32)}


@pytest.mark.parametrize("scale,expected", [({"alpha": 0.5}, 4.0), ({"beta": 0.5}, 6.5)], ids=["alpha", "beta"])
def test_a_gemm_scaled_by_a_fraction_has_no_proof(scale: dict[str, float], expected: float) -> None:
    # Within its container it computes as without exact integers.
    model = arithmetic("Gemm", SCALED, "INT32", [1, 1], **scale)
    on, off = run(model, SCALED, True), run(model, SCALED, False)
    assert on.dtype == np.float32 and on.tobytes() == off.tobytes() and on.item() == expected
    # Past it, it is refused.
    wide = {**CANCELLING, "c": np.array([[1]], np.float32)}
    with pytest.raises(NotImplementedError, match=r"Gemm_0 \(Gemm\): a is given a value past .*Gemm has no proof"):
        run(arithmetic("Gemm", wide, "INT32", [1, 1], **scale), wide, True)


def test_an_operator_without_a_proof_is_not_shown_exact_within_its_container() -> None:
    # Every value given and the answer, 1, are within 2**24; the products pass it and
    # cancel in float32, with exact integers as without.
    arrays = {"a": np.array([4097, 4096], np.float32), "b": np.array([4097, -4098], np.float32)}
    model = arithmetic("Einsum", arrays, "INT32", [], equation="i,i->")
    on, off = run(model, arrays, True), run(model, arrays, False)
    assert on.dtype == np.float32 and on.tobytes() == off.tobytes() and on.item() == 0.0
    # MatMul, with a bound, is exact on the same values.
    rows = {"a": arrays["a"].reshape(1, 2), "b": arrays["b"].reshape(2, 1)}
    assert run(arithmetic("MatMul", rows, "INT32", [1, 1]), rows, True).item() == 1.0


def test_an_operator_that_moves_values_is_exact_as_its_inputs() -> None:
    # Its own container holds the values given, however wide the annotation.
    for value in (3, 2**24 + 2, 2**53):
        y = execute_onnx(identity("UINT54"), {"x": np.array([value], np.int64)}, exact_integers=True)["y"]
        assert y.dtype == np.float32 and int(y.item()) == value
    # float64 where a value given needs it.
    for value in (2**24 + 1, 2**53 - 1):
        y = execute_onnx(identity("UINT54"), {"x": np.array([value], np.int64)}, exact_integers=True)["y"]
        assert y.dtype == np.float64 and int(y.item()) == value
    # A value float64 cannot hold is refused as the node reads it.
    with pytest.raises(ArithmeticError, match=r"identity \(Identity\): x holds a value that float64"):
        execute_onnx(identity("UINT54"), {"x": np.array([2**53 + 1], np.int64)}, exact_integers=True)


def unproved(op: str, value: int) -> ModelWrapper:
    """op of x by an initializer holding value, every tensor held in float32 and
    annotated INT32."""
    model = single(helper.make_node(op, ["x", "k"], ["y"], name=f"{op}_0"), [1])
    model.set_initializer("k", np.array([value], np.float32))
    for name in ("x", "k", "y"):
        model.set_tensor_datatype(name, DataType["INT32"])
    return model


def computed(model: ModelWrapper, x: int, setting: bool) -> np.ndarray:
    """y of the model's one node given x, with exact integers switched by setting (no
    rounding to the annotations)."""
    context = model.make_empty_exec_context()
    context["x"] = np.array([x], np.float32)
    execute_node(model.graph.node[0], context, model.graph, 11, model=model, exact_integers=setting)
    return context["y"]


@pytest.mark.parametrize("op,x,k", [("Div", 7, 2), ("Pow", 2, 3), ("Div", 2**24, 2**24)])
def test_an_operator_without_a_proof_computes_as_off_within_its_container(op: str, x: int, k: int) -> None:
    on, off = computed(unproved(op, k), x, True), computed(unproved(op, k), x, False)
    assert on.dtype == np.float32 and on.tobytes() == off.tobytes()


def test_an_operator_without_a_proof_is_refused_past_its_container() -> None:
    # A value given past 2**24, though float32 holds it.
    with pytest.raises(NotImplementedError, match=r"Div_0 \(Div\): x is given a value past .*Div has no proof"):
        computed(unproved("Div", 2), 2**24 + 2, True)
    # A result past 2**24, from values within it.
    with pytest.raises(NotImplementedError, match=r"Pow_0 \(Pow\): its result y holds a value past .*Pow has no proof"):
        computed(unproved("Pow", 40), 3, True)
    # Off, as it always did.
    assert computed(unproved("Pow", 40), 3, False).item() == np.float32(3.0) ** 40


def cast() -> ModelWrapper:
    """A Cast to FLOAT of an INT32 container, both tensors annotated INT32."""
    model = single(helper.make_node("Cast", ["x"], ["y"], name="Cast_0", to=F), [1], kind=TensorProto.INT32)
    for name in ("x", "y"):
        model.set_tensor_datatype(name, DataType["INT32"])
    return model


def test_a_cast_runs_in_its_own_types_and_must_keep_every_value() -> None:
    x = np.array([16777216], np.int32)
    off = execute_onnx(cast(), {"x": x})["y"]
    on = execute_onnx(cast(), {"x": x}, exact_integers=True)["y"]
    assert on.dtype == off.dtype == np.float32 and on.tobytes() == off.tobytes()
    odd = np.array([16777217], np.int32)
    with pytest.raises(ArithmeticError, match=r"Cast_0 \(Cast\): float32 does not hold every value of x"):
        execute_onnx(cast(), {"x": odd}, exact_integers=True)
    assert execute_onnx(cast(), {"x": odd})["y"].tolist() == [16777216.0]


def test_an_operator_whose_attribute_fixes_a_type_is_never_widened() -> None:
    node = helper.make_node(
        "ConstantOfShape", ["shape"], ["y"], name="ConstantOfShape_0", value=helper.make_tensor("value", F, [1], [16777216])
    )
    graph = helper.make_graph(
        [node],
        "fill",
        [helper.make_tensor_value_info("shape", TensorProto.INT64, [1])],
        [helper.make_tensor_value_info("y", F, [1])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("y", DataType["INT32"])
    shape = {"shape": np.array([1], np.int64)}
    on = execute_onnx(model, shape, exact_integers=True)["y"]
    assert on.dtype == np.float32 and on.tobytes() == execute_onnx(model, shape)["y"].tobytes()


def branched(value: int) -> ModelWrapper:
    """An If on x whose branches both give value, y held in float32, annotated INT27."""
    branch = helper.make_graph(
        [helper.make_node("Constant", [], ["value"], value=helper.make_tensor("value", F, [1], [value]))],
        "branch",
        [],
        [helper.make_tensor_value_info("value", F, [1])],
    )
    node = helper.make_node("If", ["x"], ["y"], name="If_0", then_branch=branch, else_branch=branch)
    model = single(node, [], [1], kind=TensorProto.BOOL)
    model.set_tensor_datatype("y", DataType["INT27"])
    return model


def test_a_node_with_a_subgraph_is_refused_where_its_result_passes_its_container() -> None:
    given = {"x": np.array(True)}
    assert execute_onnx(branched(1), given, exact_integers=True)["y"].item() == 1
    with pytest.raises(NotImplementedError, match=r"If_0 \(If\): its result y .*a node with a subgraph has no proof"):
        execute_onnx(branched(2**25), given, exact_integers=True)
    assert execute_onnx(branched(2**25), given)["y"].item() == 2**25


def test_a_widened_op_without_a_float64_kernel_raises_naming_it() -> None:
    node = helper.make_node("Conv", ["x", "w"], ["y"], name="Conv_0", kernel_shape=[1])
    model = single(node, [1, 1, 2])
    model.set_initializer("w", np.ones((1, 1, 1), np.float32))
    for name in ("x", "w", "y"):
        model.set_tensor_datatype(name, DataType["INT26"])
    # Values float32 holds, and a bound it holds: Conv computes in float32.
    y = execute_onnx(model, {"x": np.ones((1, 1, 2), np.float32)}, exact_integers=True)["y"]
    assert y.dtype == np.float32 and y.tolist() == [[[1, 1]]]
    # A bound past 2**24 widens Conv, which ONNX Runtime has no float64 kernel for.
    wide = {"x": np.array([[[2**24 + 1, 1]]], np.int64)}
    with pytest.raises(NotImplementedError, match=r"Conv_0 \(Conv\): ONNX Runtime has no float64 kernel for Conv") as caught:
        execute_onnx(model, wide, exact_integers=True)
    assert type(caught.value.__cause__).__name__ == "NotImplemented"


def test_off_by_default_and_within_float32_nothing_changes() -> None:
    weights = wide_weights()
    model = wide_layer(weights)
    x = wide_input(weights).astype(np.float32)
    default = execute_onnx(model, {"x": x}, return_full_exec_context=True)
    off = execute_onnx(model, {"x": x}, return_full_exec_context=True, exact_integers=False)
    for name, value in default.items():
        if value is not None:
            assert off[name].dtype == value.dtype and np.array_equal(off[name], value)
    # A layer whose sums float32 holds exactly runs with the setting on as it does off.
    narrow = wide_layer(weights[:16])
    narrow.set_tensor_shape("x", [2, 16])
    # 16 products of INT8 values reach 2**18 at most.
    narrow.set_tensor_datatype("s", DataType["INT20"])
    xs = x[:, :16]
    off = execute_onnx(narrow, {"x": xs}, return_full_exec_context=True)
    exact = execute_onnx(narrow, {"x": xs}, return_full_exec_context=True, exact_integers=True)
    for name, value in off.items():
        if value is not None:
            assert exact[name].dtype == value.dtype and np.array_equal(exact[name], value)


def test_execute_node_takes_the_setting_and_needs_the_model_for_it() -> None:
    model = identity("INT26")
    context = model.make_empty_exec_context()
    context["x"] = np.array([16777217], np.int64)
    node = model.graph.node[0]
    with pytest.raises(ValueError, match=r"identity \(Identity\).*pass model"):
        execute_node(node, context, model.graph, 13, exact_integers=True)
    with pytest.raises(rt.capi.onnxruntime_pybind11_state.InvalidArgument, match="tensor.int64"):
        execute_node(node, context, model.graph, 13, model=model)
    with exact_integers():
        with pytest.raises(rt.capi.onnxruntime_pybind11_state.InvalidArgument, match="tensor.int64"):
            execute_node(node, context, model.graph, 13, model=model, exact_integers=False)
        execute_node(node, context, model.graph, 13, model=model)
    assert context["y"].dtype == np.float64 and context["y"].item() == 16777217


def test_a_generic_partition_runs_its_body_with_the_setting(tmp_path: Path) -> None:
    model = partition(identity("INT26"), tmp_path)
    inputs = {"x": np.array([16777217], np.int64)}
    with pytest.raises(rt.capi.onnxruntime_pybind11_state.InvalidArgument, match="tensor.int64"):
        execute_onnx(model, inputs)
    y = execute_onnx(model, inputs, exact_integers=True)["y"]
    assert y.dtype == np.float64 and y.item() == 16777217
    with exact_integers():
        assert execute_onnx(model, inputs)["y"].item() == 16777217
        with pytest.raises(rt.capi.onnxruntime_pybind11_state.InvalidArgument):
            execute_onnx(model, inputs, exact_integers=False)


def test_a_wide_generic_partition_body_is_exact(tmp_path: Path) -> None:
    weights = wide_weights()
    body = wide_layer(weights)
    path = str(tmp_path / "body.onnx")
    body.save(path)
    graph = helper.make_graph(
        [helper.make_node("GenericPartition", ["x"], ["s", "scaled"], domain=GENERAL, model=path)],
        "parent",
        [helper.make_tensor_value_info("x", F, [2, K])],
        [helper.make_tensor_value_info("s", F, [2, 3]), helper.make_tensor_value_info("scaled", F, [2, 3])],
    )
    parent = ModelWrapper(qonnx_make_model(graph))
    x = wide_input(weights)
    s = execute_onnx(parent, {"x": x.astype(np.float32)}, exact_integers=True)["s"]
    assert np.array_equal(s.astype(np.int64), x @ weights.astype(np.int64))


def test_the_model_writer_takes_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    model = identity("INT26")
    inputs = {"x": np.array([16777217], np.int64)}
    with pytest.raises(rt.capi.onnxruntime_pybind11_state.InvalidArgument) as caught:
        execute_onnx_and_make_model(model, inputs)
    assert str(caught.value) == (
        "[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Unexpected input data type. "
        "Actual: (tensor(int64)) , expected: (tensor(float))"
    )
    # The writer's execution with the setting, its context's entry for the empty name
    # (an omitted optional input, None) left out.
    settings = []

    def execution(*args: Any, **kwargs: Any) -> dict[str, Any]:
        settings.append(kwargs.get("exact_integers"))
        with exact_integers(kwargs.pop("exact_integers")):
            context = execute_onnx(*args, **kwargs)
        return {name: value for name, value in context.items() if name}

    monkeypatch.setattr(onnx_exec, "execute_onnx", execution)
    written = execute_onnx_and_make_model(model, inputs, exact_integers=True)
    assert settings == [True]
    assert written.get_initializer("y").item() == 16777217


def test_quantavgpool2d_keeps_its_float32_computation() -> None:
    node = helper.make_node(
        "QuantAvgPool2d",
        ["x"],
        ["y"],
        name="pool",
        domain=GENERAL,
        kernel=2,
        stride=2,
        ibits=26,
        obits=28,
        signed=1,
    )
    model = single(node, [1, 1, 2, 2], [1, 1, 1, 1])
    model.set_tensor_datatype("x", DataType["INT26"])
    model.set_tensor_datatype("y", DataType["INT28"])
    x = np.full((1, 1, 2, 2), 16777217, np.float64)
    result = execute_onnx(model, {"x": x}, exact_integers=True)["y"]
    assert result.dtype == np.float32 and result.item() == 67108864


def threshold(
    count: int,
    dtype: str,
    scale: float = 1.0,
    bias: float = 0.0,
    input_type: int = TensorProto.INT8,
) -> ModelWrapper:
    """A MultiThreshold from x ([1, 1], held in input_type) over count zero thresholds."""
    node = helper.make_node(
        "MultiThreshold",
        ["x", "t"],
        ["y"],
        name="threshold",
        domain=GENERAL,
        out_dtype=dtype,
        out_scale=scale,
        out_bias=bias,
        data_layout="NC",
    )
    model = single(node, [1, 1], kind=input_type)
    model.set_initializer("t", np.zeros((1, count), dtype=np.float32))
    return model


@pytest.mark.parametrize("scale,bias,expected", [(1.0, -128.0, 2), (-2.0, 256.0, -4)])
def test_an_int8_count_and_an_integer_affine_result_do_not_wrap(scale: float, bias: float, expected: int) -> None:
    model = threshold(count=130, dtype="INT10", scale=scale, bias=bias)
    produced = execute_onnx(model, {"x": np.ones((1, 1), dtype=np.int8)}, exact_integers=True)["y"]
    assert produced.dtype == np.float64
    assert produced.tolist() == [[expected]]


def test_an_integer_affine_cancellation_beyond_int64_is_exact() -> None:
    model = threshold(count=1, dtype="UINT65", scale=-float(2**64), bias=float(2**64))
    produced = execute_onnx(model, {"x": np.ones((1, 1), dtype=np.int8)}, exact_integers=True)["y"]
    assert produced.dtype == np.float64
    assert produced.tolist() == [[0]]


def test_a_fractional_scale_and_bias_keep_numpy_promotion() -> None:
    model = threshold(count=130, dtype="FLOAT32", scale=0.5, bias=-63.5)
    produced = execute_onnx(model, {"x": np.ones((1, 1), dtype=np.int8)}, exact_integers=True)["y"]
    assert produced.dtype == np.float64
    assert produced.tolist() == [[1.5]]


@pytest.mark.parametrize(
    "count,dtype,scale,bias",
    [(2, "UINT1", 1.0, 0.0), (1, "UINT2", 1.0, -1.0), (2, "INT2", -2.0, 1.0)],
)
def test_out_dtype_must_hold_every_count_even_where_the_input_gives_one_it_holds(
    count: int, dtype: str, scale: float, bias: float
) -> None:
    model = threshold(count=count, dtype=dtype, scale=scale, bias=bias)
    x = {"x": -np.ones((1, 1), dtype=np.int8)}
    with pytest.raises(ValueError, match=f"out_dtype {DataType[dtype].name} cannot hold"):
        execute_onnx(model, x, exact_integers=True)
    # Without exact integers, MultiThreshold refuses nothing.
    execute_onnx(model, x)


def test_out_dtype_is_checked_at_interior_counts() -> None:
    model = threshold(count=2, dtype="BIPOLAR", bias=-1.0)
    with pytest.raises(ValueError, match="out_dtype BIPOLAR cannot hold 0 at count 1"):
        execute_onnx(model, {"x": -np.ones((1, 1), dtype=np.int8)}, exact_integers=True)


def test_an_exact_integer_result_must_fit_the_float32_container() -> None:
    model = threshold(count=1, dtype="UINT25", bias=float(2**24), input_type=F)
    x = {"x": -np.ones((1, 1), dtype=np.float32)}
    with pytest.raises(ValueError, match="the float32 container cannot hold exact result 16777217 at count 1"):
        execute_onnx(model, x, exact_integers=True)
    execute_onnx(model, x)


def test_a_custom_node_takes_the_setting() -> None:
    model = threshold(count=2, dtype="UINT1")
    context = model.make_empty_exec_context()
    context["x"] = -np.ones((1, 1), dtype=np.int8)
    node = model.graph.node[0]
    with pytest.raises(ValueError, match="out_dtype BINARY cannot hold 2"):
        execute_custom_node(node, context, model.graph, 1, model=model, exact_integers=True)
    with exact_integers():
        execute_custom_node(node, context, model.graph, 1, exact_integers=False)
        with pytest.raises(ValueError, match="out_dtype BINARY cannot hold 2"):
            execute_node(node, context, model.graph, 1)


@pytest.mark.parametrize(
    "input_type,numpy_type,intermediate,decision",
    [
        (TensorProto.INT8, np.int8, -(2**-27), 0),
        (TensorProto.FLOAT16, np.float16, -(2**-12), 0),
        (TensorProto.FLOAT, np.float32, 0.0, 1),
        (TensorProto.DOUBLE, np.float64, -(2**-27), 0),
    ],
)
def test_a_fractional_cancellation_keeps_the_next_integer_decision(
    input_type: int, numpy_type: Any, intermediate: float, decision: int
) -> None:
    model = threshold(count=3, dtype="FLOAT32", scale=0.1, bias=-0.3, input_type=input_type)
    model.graph.node[0].output[0] = "u"
    model.graph.value_info.append(helper.make_tensor_value_info("u", F, [1, 1]))
    model.graph.node.append(
        helper.make_node("MultiThreshold", ["u", "zero"], ["y"], domain=GENERAL, out_dtype="BINARY", data_layout="NC")
    )
    model.set_initializer("zero", np.zeros((1, 1), dtype=np.float32))
    context = execute_onnx(model, {"x": np.ones((1, 1), dtype=numpy_type)}, True, exact_integers=True)
    assert context["u"].tolist() == [[intermediate]]
    assert context["y"].tolist() == [[decision]]
    expected = np.dtype(np.float64 if numpy_type == np.int8 else numpy_type)
    assert context["u"].dtype == expected and context["y"].dtype == expected


@pytest.mark.parametrize("scale,bias", [(float("inf"), 0.0), (1.0, float("nan")), (0.5, float("nan"))])
def test_non_finite_attributes_keep_ieee_behaviour(scale: float, bias: float) -> None:
    model = threshold(count=1, dtype="FLOAT32", scale=scale, bias=bias, input_type=F)
    with np.errstate(all="ignore"):
        produced = execute_onnx(model, {"x": np.ones((1, 1), dtype=np.float32)}, exact_integers=True)["y"]
    assert produced.dtype == np.float32
    if np.isnan(bias):
        assert np.isnan(produced).all()
    else:
        assert np.isposinf(produced).all()
