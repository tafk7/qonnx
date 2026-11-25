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

import pytest

import numpy as np
from onnx import TensorProto, helper

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.custom_op.general.multithreshold import multithreshold
from qonnx.util.basic import qonnx_make_model


@pytest.mark.parametrize("container", [np.int8, np.uint8, np.float16, np.float32, np.float64])
@pytest.mark.parametrize("scale,bias", [(1.0, 0.0), (0.5, -63.5), (-2.0, 256.0)])
def test_count_does_not_narrow_to_the_input(container: type, scale: float, bias: float) -> None:
    count = 300 if container == np.uint8 else 200
    x = np.ones((1, 1), dtype=container)
    result = multithreshold(x, np.zeros((1, count), dtype=np.float32), scale, bias)
    count_container = np.int64 if container in (np.int8, np.uint8) else container
    expected_type = np.dtype((scale * np.zeros(1, dtype=count_container) + bias).dtype)
    assert result.dtype == expected_type
    assert result.item() == scale * count + bias


@pytest.mark.parametrize("channels_last", [False, True])
def test_each_channel_counts_thresholds_in_its_original_layout(channels_last: bool) -> None:
    values = np.asarray([[[1, -1], [2, 0]]], dtype=np.int8)
    if channels_last:
        values = values.swapaxes(1, 2)
    result = multithreshold(values, np.zeros((2, 200), dtype=np.float32), channels_last=channels_last)
    np.testing.assert_array_equal(result, np.where(values >= 0, 200, 0))


def test_default_node_execution_keeps_the_count_exact() -> None:
    node = helper.make_node("MultiThreshold", ["x", "t"], ["y"], domain="qonnx.custom_op.general", out_dtype="UINT8")
    graph = helper.make_graph(
        [node],
        "count",
        [helper.make_tensor_value_info("x", TensorProto.INT8, [1, 1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_initializer("t", np.zeros((1, 200), dtype=np.float32))
    result = execute_onnx(model, {"x": np.ones((1, 1), dtype=np.int8)})["y"]
    assert result.dtype == np.float64
    assert result.item() == 200


@pytest.mark.parametrize("container", [np.float16, np.float32, np.float64])
def test_fractional_affine_keeps_its_existing_precision(container: type) -> None:
    values = np.ones((1, 1), dtype=container)
    result = multithreshold(values, np.zeros((1, 3), dtype=np.float32), 0.1, -0.3)
    expected = 0.1 * np.asarray([[3]], dtype=container) - 0.3
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "scale,bias,expected",
    [(0.1, np.float64(0.0), 0.2998046875), (np.float16(0.1), np.float32(0.0), 0.2998046875)],
)
def test_affine_preserves_upstream_mixed_numpy_promotion(scale: float, bias: float, expected: float) -> None:
    result = multithreshold(np.ones((1, 1), dtype=np.float16), np.zeros((1, 3)), scale, bias)
    upstream = np.asarray([[expected]], dtype=type(bias))
    assert result.dtype == upstream.dtype
    assert result.tobytes() == upstream.tobytes()


def test_integer_multiplication_overflow_is_only_corrected_by_exact_execution() -> None:
    values = np.ones((1, 1), dtype=np.int8)
    thresholds = np.zeros((1, 100), dtype=np.float32)
    # The public helper's Python integer scale multiplies in int8 before the
    # floating bias promotes the result. Its count already fits in int8.
    result = multithreshold(values, thresholds, out_scale=2, out_bias=0.0)
    assert result.dtype == np.float64
    assert result.tobytes() == np.asarray([[-56.0]], dtype=np.float64).tobytes()

    # ONNX FLOAT attributes carry the same integer-valued scale to the exact
    # affine path through the public execution entry point.
    node = helper.make_node(
        "MultiThreshold",
        ["x", "t"],
        ["y"],
        domain="qonnx.custom_op.general",
        out_dtype="UINT8",
        out_scale=2.0,
        out_bias=0.0,
    )
    graph = helper.make_graph(
        [node],
        "integer-affine",
        [helper.make_tensor_value_info("x", TensorProto.INT8, [1, 1])],
        [helper.make_tensor_value_info("y", TensorProto.DOUBLE, [1, 1])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_initializer("t", thresholds)
    exact = execute_onnx(model, {"x": values}, exact_integers=True)["y"]
    assert exact.dtype == np.float64
    assert exact.item() == 200


@pytest.mark.parametrize(
    "container,count,expected_type",
    [
        (np.int8, 127, np.float64),
        (np.int8, 128, np.float64),
        (np.uint8, 255, np.float64),
        (np.uint8, 256, np.float64),
        (np.bool_, 1, np.float64),
        (np.bool_, 2, np.float64),
        (np.float16, 2048, np.float16),
        (np.float16, 2049, np.float64),
    ],
)
def test_count_container_at_its_consecutive_integer_boundary(container: type, count: int, expected_type: type) -> None:
    result = multithreshold(np.ones((1, 1), dtype=container), np.zeros((1, count)))
    assert result.dtype == expected_type
    assert result.item() == count
