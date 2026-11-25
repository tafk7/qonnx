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
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.util.basic import qonnx_make_model


def _pool_model(shape: tuple[int, ...], layout: str, ibits: int = 4, obits: int = 6) -> ModelWrapper:
    node = helper.make_node(
        "QuantAvgPool2d",
        ["x"],
        ["y"],
        domain="qonnx.custom_op.general",
        kernel=2,
        stride=2,
        ibits=ibits,
        obits=obits,
        signed=0,
        data_layout=layout,
    )
    graph = helper.make_graph(
        [node],
        "average-pool",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 1, 1])],
    )
    return ModelWrapper(qonnx_make_model(graph))


@pytest.mark.parametrize("container", [np.float16, np.float32, np.float64, np.longdouble])
@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
@pytest.mark.parametrize("exact", [False, True])
def test_average_pool_accepts_floating_containers_as_float32(container: type, layout: str, exact: bool) -> None:
    values = np.asarray([[[[1, 2], [3, 4]]]], dtype=container)
    if layout == "NHWC":
        values = values.transpose(0, 2, 3, 1)
    model = _pool_model(values.shape, layout)
    before = values.copy()
    result = execute_onnx(model, {"x": values}, exact_integers=exact)["y"]
    assert result.dtype == np.float32
    assert result.item() == 10
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("container", [np.int8, np.uint8, np.int64])
@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
def test_average_pool_retains_upstream_integer_feed_refusal(container: type, layout: str) -> None:
    values = np.asarray([[[[1, 2], [3, 4]]]], dtype=container)
    if layout == "NHWC":
        values = values.transpose(0, 2, 3, 1)
    model = _pool_model(values.shape, layout)
    before = values.copy()
    with pytest.raises(InvalidArgument, match="Unexpected input data type"):
        execute_onnx(model, {"x": values})
    np.testing.assert_array_equal(values, before)


def test_average_pool_refuses_wide_integer_feed_instead_of_rounding_it() -> None:
    values = np.full((1, 1, 2, 2), 16777219, dtype=np.int64)
    model = _pool_model(values.shape, "NCHW", ibits=25, obits=24)
    with pytest.raises(InvalidArgument, match="Unexpected input data type"):
        execute_onnx(model, {"x": values})
