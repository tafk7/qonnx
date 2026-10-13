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

"""Execution with exact integers (QONNX_EXACT_INTEGERS): a standard node whose float tensors
all hold integers, one past what its float container holds exactly, computes in float64;
constant folding stores such a node's value in its stated type where that type holds it,
and leaves the node unfolded where it does not."""

import pytest

import numpy as np
from onnx import TensorProto, helper

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import EXACT_INTEGERS, execute_node, execute_onnx, get_exact_integers
from qonnx.transformation.fold_constants import FoldConstants, FoldConstantsFiltered
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import qonnx_make_model

GENERAL = "qonnx.custom_op.general"
K = 2304
F = TensorProto.FLOAT


@pytest.fixture
def exact(monkeypatch):
    """Exact integers on for the test."""
    monkeypatch.setenv(EXACT_INTEGERS, "1")


@pytest.fixture
def rounding(monkeypatch):
    """Exact integers off for the test, as by default."""
    monkeypatch.delenv(EXACT_INTEGERS, raising=False)


def wide_weights():
    """INT8 weights over k = 2304, spread over their whole range."""
    return np.random.default_rng(K).integers(-128, 128, size=(K, 3)).astype(np.float32)


def wide_input(weights):
    """An INT8 input whose first row's sums reach past 2**24, the second's first an odd
    integer there, which float32 cannot hold."""
    x = (127 * np.where(weights[:, :1].T > 0, 1, -1)).repeat(2, axis=0).astype(np.int64)
    x[1, 0] -= 1
    return x


def wide_layer(weights=None):
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


def test_the_setting_is_read_from_the_environment(monkeypatch):
    monkeypatch.delenv(EXACT_INTEGERS, raising=False)
    assert get_exact_integers() is False
    monkeypatch.setenv(EXACT_INTEGERS, "0")
    assert get_exact_integers() is False
    monkeypatch.setenv(EXACT_INTEGERS, "1")
    assert get_exact_integers() is True
    monkeypatch.setenv(EXACT_INTEGERS, "yes")
    with pytest.raises(ValueError, match=EXACT_INTEGERS):
        get_exact_integers()


def test_a_wide_matmul_is_exact_with_the_setting_and_rounds_without(monkeypatch):
    monkeypatch.delenv(EXACT_INTEGERS, raising=False)
    weights = wide_weights()
    model = wide_layer(weights)
    x = wide_input(weights)
    exact = x @ weights.astype(np.int64)
    assert np.abs(exact).max() > 2**24 and exact[1, 0] % 2 == 1
    assert model.get_tensor_datatype("s").is_integer()

    rounded = execute_onnx(model, {"x": x.astype(np.float32)})["s"]
    assert rounded.dtype == np.float32
    assert not np.array_equal(rounded.astype(np.int64), exact)

    monkeypatch.setenv(EXACT_INTEGERS, "1")
    widened = execute_onnx(model, {"x": x.astype(np.float32)})["s"]
    assert widened.dtype == np.float64
    assert np.array_equal(widened.astype(np.int64), exact)
    # The caller may give the integers as they are: the node casts what it reads.
    given = execute_onnx(model, {"x": x})["s"]
    assert np.array_equal(given.astype(np.int64), exact)


def test_a_float_node_reads_a_widened_tensor_back_in_float32(exact):
    weights = wide_weights()
    model = wide_layer(weights)
    x = wide_input(weights)
    exact = x @ weights.astype(np.int64)
    out = execute_onnx(model, {"x": x.astype(np.float32)}, return_full_exec_context=True)
    assert out["s"].dtype == np.float64
    # Mul is float arithmetic (0.5 is no integer): it reads the exact sums in float32.
    assert out["scaled"].dtype == np.float32
    assert np.array_equal(out["scaled"], exact.astype(np.float32) * np.float32(0.5))


def test_a_node_widened_past_2_to_the_53_raises_naming_the_tensor(exact):
    model = wide_layer()
    model.set_tensor_datatype("s", DataType["INT60"])
    x = np.zeros((2, K), dtype=np.float32)
    with pytest.raises(ArithmeticError, match=r"MatMul_0 \(MatMul\): s .*INT60"):
        execute_onnx(model, {"x": x})


def test_a_widened_op_without_a_float64_kernel_raises_naming_it(exact):
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "w"], ["y"], name="Conv_0", kernel_shape=[1, 1])],
        "conv",
        [helper.make_tensor_value_info("x", F, [1, 2, 2, 2])],
        [helper.make_tensor_value_info("y", F, [1, 1, 2, 2])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_initializer("w", np.ones((1, 2, 1, 1), dtype=np.float32))
    model.set_tensor_datatype("x", DataType["INT26"])
    model.set_tensor_datatype("w", DataType["INT2"])
    model = model.transform(InferDataTypes())
    assert model.get_tensor_datatype("y").max() > 2**24
    with pytest.raises(NotImplementedError, match=r"Conv_0 \(Conv\).*float64"):
        execute_onnx(model, {"x": np.ones((1, 2, 2, 2), dtype=np.float32)})


def test_off_by_default_and_within_float32_nothing_changes(monkeypatch):
    weights = wide_weights()
    model = wide_layer(weights)
    x = wide_input(weights).astype(np.float32)
    monkeypatch.delenv(EXACT_INTEGERS, raising=False)
    default = execute_onnx(model, {"x": x}, return_full_exec_context=True)
    monkeypatch.setenv(EXACT_INTEGERS, "0")
    off = execute_onnx(model, {"x": x}, return_full_exec_context=True)
    for name, value in default.items():
        if value is not None:
            assert off[name].dtype == value.dtype and np.array_equal(off[name], value)
    # A layer whose sums float32 holds exactly runs with the setting on as it does off.
    narrow = wide_layer(weights[:16])
    narrow.set_tensor_shape("x", [2, 16])
    narrow = narrow.transform(InferDataTypes())
    assert narrow.get_tensor_datatype("s").max() <= 2**24
    xs = x[:, :16]
    off = execute_onnx(narrow, {"x": xs}, return_full_exec_context=True)
    monkeypatch.setenv(EXACT_INTEGERS, "1")
    exact = execute_onnx(narrow, {"x": xs}, return_full_exec_context=True)
    for name, value in off.items():
        if value is not None:
            assert exact[name].dtype == value.dtype and np.array_equal(exact[name], value)


def test_execute_node_needs_the_model_for_the_setting(exact):
    model = wide_layer()
    context = model.make_empty_exec_context()
    with pytest.raises(ValueError, match="MatMul_0"):
        execute_node(model.graph.node[0], context, model.graph, 13)


def test_maxpoolnhwc_computes_in_its_input_type(exact):
    d = TensorProto.DOUBLE
    graph = helper.make_graph(
        [
            helper.make_node(
                "MaxPoolNHWC",
                ["x"],
                ["y"],
                domain=GENERAL,
                kernel_shape=[2, 2],
                pads=[0, 0, 0, 0],
                strides=[2, 2],
                name="MaxPoolNHWC_0",
            )
        ],
        "maxpool",
        [helper.make_tensor_value_info("x", d, [1, 2, 2, 1])],
        [helper.make_tensor_value_info("y", d, [1, 1, 1, 1])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    x = np.array([2**24 + 1, 3, -5, 2**24 - 1], dtype=np.float64).reshape(1, 2, 2, 1)
    y = execute_onnx(model, {"x": x})["y"]
    assert y.dtype == np.float64 and int(y.item()) == 2**24 + 1


def test_quantavgpool2d_reads_a_float64_input(exact):
    graph = helper.make_graph(
        [
            helper.make_node(
                "QuantAvgPool2d",
                ["x"],
                ["y"],
                domain=GENERAL,
                stride=2,
                kernel=2,
                ibits=4,
                obits=4,
                signed=0,
                name="QuantAvgPool2d_0",
            )
        ],
        "avgpool",
        [helper.make_tensor_value_info("x", TensorProto.DOUBLE, [1, 1, 2, 2])],
        [helper.make_tensor_value_info("y", F, [1, 1, 1, 1])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    x = np.array([1, 2, 3, 6], dtype=np.float64).reshape(1, 1, 2, 2)
    y = execute_onnx(model, {"x": x})["y"]
    assert y.dtype == np.float32 and y.item() == 3


def test_a_generic_partition_runs_its_body_with_the_setting(exact, tmp_path):
    weights = wide_weights()
    body = wide_layer(weights)
    body_file = str(tmp_path / "body.onnx")
    body.save(body_file)
    graph = helper.make_graph(
        [helper.make_node("GenericPartition", ["x"], ["s", "scaled"], domain=GENERAL, model=body_file)],
        "parent",
        [helper.make_tensor_value_info("x", F, [2, K])],
        [helper.make_tensor_value_info("s", F, [2, 3]), helper.make_tensor_value_info("scaled", F, [2, 3])],
    )
    parent = ModelWrapper(qonnx_make_model(graph))
    x = wide_input(weights)
    exact = x @ weights.astype(np.int64)
    s = execute_onnx(parent, {"x": x.astype(np.float32)})["s"]
    assert np.array_equal(s.astype(np.int64), exact)


def constant_sum(a=2**24, b=1):
    """c = a + b, both constants, then y = x + c; every tensor INT32-annotated and held in
    float32."""
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["a", "b"], ["c"], name="Add_0"),
            helper.make_node("Add", ["x", "c"], ["y"], name="Add_1"),
        ],
        "constant_sum",
        [helper.make_tensor_value_info("x", F, [1])],
        [helper.make_tensor_value_info("y", F, [1])],
        value_info=[helper.make_tensor_value_info("c", F, [1])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_initializer("a", np.array([a], dtype=np.float32))
    model.set_initializer("b", np.array([b], dtype=np.float32))
    for name in ("a", "b", "c", "x", "y"):
        model.set_tensor_datatype(name, DataType["INT32"])
    return model


@pytest.mark.parametrize("fold", [FoldConstants(), FoldConstantsFiltered(lambda model, node: True)])
def test_a_sum_float32_cannot_hold_is_left_unfolded_and_executes_exactly(exact, fold):
    """The INT32 sum 16777216 + 1 stays a node; a constant node after it still folds."""
    model = constant_sum()
    model.graph.node.append(helper.make_node("Add", ["a", "a"], ["d"], name="Add_2"))
    model.graph.node.append(helper.make_node("Add", ["x", "d"], ["e"], name="Add_3"))
    model.graph.value_info.append(helper.make_tensor_value_info("d", F, [1]))
    model.graph.output.append(helper.make_tensor_value_info("e", F, [1]))
    folded = model.transform(fold)
    assert sorted(n.name for n in folded.graph.node) == ["Add_0", "Add_1", "Add_3"]
    assert folded.get_initializer("c") is None
    assert folded.get_initializer("d").item() == 2**25
    x = {"x": np.zeros(1, dtype=np.float32)}
    assert execute_onnx(folded, x)["y"].item() == 2**24 + 1


def test_without_the_setting_folding_rounds_the_sum(rounding):
    folded = constant_sum().transform(FoldConstants())
    c = folded.get_initializer("c")
    assert c.dtype == np.float32 and c.item() == 2**24
    assert execute_onnx(folded, {"x": np.zeros(1, dtype=np.float32)})["y"].item() == 2**24


def test_a_widened_fold_that_float32_holds_is_stored_as_without_the_setting(monkeypatch):
    monkeypatch.setenv(EXACT_INTEGERS, "1")
    exact = constant_sum(2**24, 2).transform(FoldConstants())
    monkeypatch.setenv(EXACT_INTEGERS, "0")
    off = constant_sum(2**24, 2).transform(FoldConstants())
    assert exact.get_initializer("c").dtype == np.float32
    assert exact.get_initializer("c").item() == 2**24 + 2
    assert exact.model == off.model


def test_a_widened_fold_is_exact_where_float32_arithmetic_is_not(exact):
    """A dot product of constants whose partial sums pass 2**24 but whose result float32
    holds: folded in float64, stored in float32, exact."""
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["a", "b"], ["c"], name="MatMul_0"),
            helper.make_node("Add", ["x", "c"], ["y"], name="Add_0"),
        ],
        "dot",
        [helper.make_tensor_value_info("x", F, [1, 1])],
        [helper.make_tensor_value_info("y", F, [1, 1])],
        value_info=[helper.make_tensor_value_info("c", F, [1, 1])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_initializer("a", np.array([[2**24, 1, -(2**24)]], dtype=np.float32))
    model.set_initializer("b", np.ones((3, 1), dtype=np.float32))
    model.set_tensor_datatype("a", DataType["INT26"])
    model.set_tensor_datatype("b", DataType["INT2"])
    model.set_tensor_datatype("x", DataType["INT8"])
    model = model.transform(InferDataTypes())
    assert model.get_tensor_datatype("c").max() > 2**24
    c = model.transform(FoldConstants()).get_initializer("c")
    assert c.dtype == np.float32 and c.item() == 1
