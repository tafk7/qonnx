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

"""Execution with exact_integers: a standard node whose float tensors all hold integers,
one past what its float container holds exactly, computes in float64."""

import pytest

import numpy as np
from onnx import TensorProto, helper

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_node, execute_onnx
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import qonnx_make_model

GENERAL = "qonnx.custom_op.general"
K = 2304
F = TensorProto.FLOAT


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


def test_a_wide_matmul_is_exact_with_the_option_and_rounds_without():
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
    # The caller may give the integers as they are: the node casts what it reads.
    given = execute_onnx(model, {"x": x}, exact_integers=True)["s"]
    assert np.array_equal(given.astype(np.int64), exact)


def test_a_float_node_reads_a_widened_tensor_back_in_float32():
    weights = wide_weights()
    model = wide_layer(weights)
    x = wide_input(weights)
    exact = x @ weights.astype(np.int64)
    out = execute_onnx(model, {"x": x.astype(np.float32)}, exact_integers=True, return_full_exec_context=True)
    assert out["s"].dtype == np.float64
    # Mul is float arithmetic (0.5 is no integer): it reads the exact sums in float32.
    assert out["scaled"].dtype == np.float32
    assert np.array_equal(out["scaled"], exact.astype(np.float32) * np.float32(0.5))


def test_a_node_widened_past_2_to_the_53_raises_naming_the_tensor():
    model = wide_layer()
    model.set_tensor_datatype("s", DataType["INT60"])
    x = np.zeros((2, K), dtype=np.float32)
    with pytest.raises(ArithmeticError, match=r"MatMul_0 \(MatMul\): s .*INT60"):
        execute_onnx(model, {"x": x}, exact_integers=True)


def test_a_widened_op_without_a_float64_kernel_raises_naming_it():
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
        execute_onnx(model, {"x": np.ones((1, 2, 2, 2), dtype=np.float32)}, exact_integers=True)


def test_off_by_default_and_within_float32_nothing_changes():
    weights = wide_weights()
    model = wide_layer(weights)
    x = wide_input(weights).astype(np.float32)
    default = execute_onnx(model, {"x": x}, return_full_exec_context=True)
    off = execute_onnx(model, {"x": x}, return_full_exec_context=True, exact_integers=False)
    for name, value in default.items():
        if value is not None:
            assert off[name].dtype == value.dtype and np.array_equal(off[name], value)
    # A layer whose sums float32 holds exactly runs as it does today with the option on.
    narrow = wide_layer(weights[:16])
    narrow.set_tensor_shape("x", [2, 16])
    narrow = narrow.transform(InferDataTypes())
    assert narrow.get_tensor_datatype("s").max() <= 2**24
    xs = x[:, :16]
    today = execute_onnx(narrow, {"x": xs}, return_full_exec_context=True)
    exact = execute_onnx(narrow, {"x": xs}, return_full_exec_context=True, exact_integers=True)
    for name, value in today.items():
        if value is not None:
            assert exact[name].dtype == value.dtype and np.array_equal(exact[name], value)


def test_execute_node_needs_the_model_for_the_option():
    model = wide_layer()
    context = model.make_empty_exec_context()
    with pytest.raises(ValueError, match="MatMul_0"):
        execute_node(model.graph.node[0], context, model.graph, 13, exact_integers=True)


def test_maxpoolnhwc_computes_in_its_input_type():
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
    y = execute_onnx(model, {"x": x}, exact_integers=True)["y"]
    assert y.dtype == np.float64 and int(y.item()) == 2**24 + 1


def test_quantavgpool2d_reads_a_float64_input():
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
    y = execute_onnx(model, {"x": x}, exact_integers=True)["y"]
    assert y.dtype == np.float32 and y.item() == 3


def test_a_generic_partition_runs_its_body_with_the_option(tmp_path):
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
    s = execute_onnx(parent, {"x": x.astype(np.float32)}, exact_integers=True)["s"]
    assert np.array_equal(s.astype(np.int64), exact)
