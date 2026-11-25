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

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.util.basic import qonnx_make_model


def _model(datatype: str, pad: int, padding: bool = True) -> ModelWrapper:
    extent = 4 if padding else 2
    node = helper.make_node(
        "Im2Col",
        ["x"],
        ["y"],
        domain="qonnx.custom_op.general",
        kernel_size=[1, 1],
        stride=[1, 1],
        input_shape="(1,2,2,1)",
        pad_amount=[1, 1, 1, 1] if padding else [0, 0, 0, 0],
        pad_value=pad,
    )
    graph = helper.make_graph(
        [node],
        "padding",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 2, 1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, extent, extent, 1])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("x", DataType[datatype])
    return model


@pytest.mark.parametrize(
    "datatype,container,pad",
    [
        ("FLOAT32", np.float32, 16777217),
        ("FLOAT16", np.float16, 2049),
        ("FLOAT<8,23,127>", np.float32, 16777217),
        ("INT32", np.float32, 16777217),
        ("INT8", np.int8, 128),
        ("INT32", np.int8, 128),
        ("INT64", np.float16, 65536),
    ],
)
def test_padding_refuses_a_value_the_datatype_or_container_cannot_hold(datatype: str, container: type, pad: int) -> None:
    with pytest.raises(AssertionError, match="^Im2Col dtype must allow pad_val$"):
        execute_onnx(_model(datatype, pad), {"x": np.ones((1, 2, 2, 1), dtype=container)})


@pytest.mark.parametrize(
    "datatype,container,pad",
    [
        ("FLOAT32", np.float32, 16777216),
        ("FLOAT16", np.float16, 2048),
        ("FLOAT<8,23,127>", np.float32, 16777216),
        ("INT32", np.float64, 16777217),
        ("INT8", np.int8, -128),
    ],
)
def test_representable_padding_is_stored_exactly(datatype: str, container: type, pad: int) -> None:
    values = np.ones((1, 2, 2, 1), dtype=container)
    result = execute_onnx(_model(datatype, pad), {"x": values})["y"]
    assert result.dtype == values.dtype
    assert result[0, 0, 0, 0].item() == pad
    np.testing.assert_array_equal(result[:, 1:3, 1:3, :], values)


def test_unused_padding_value_is_not_checked() -> None:
    values = np.ones((1, 2, 2, 1), dtype=np.float32)
    np.testing.assert_array_equal(execute_onnx(_model("INT8", 16777217, False), {"x": values})["y"], values)


@pytest.mark.parametrize(
    "shape,kernel,stride,pads,dilations",
    [
        ([1, 2, 2, 1], [2, 2], [2, 2], [0, 0, 1, 1], [1, 1]),
        ([1, 3, 3, 1], [2, 2], [3, 3], [0, 0, 1, 1], [2, 2]),
    ],
)
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_exact_checks_only_apply_to_padding_sampled_by_a_window(
    shape: list[int], kernel: list[int], stride: list[int], pads: list[int], dilations: list[int], axis: int | None
) -> None:
    # Reduce one stride so its next window reaches the otherwise unused pad.
    if axis is not None:
        stride = stride.copy()
        stride[axis] = 1
    out_h = (shape[1] + pads[0] + pads[2] - (kernel[0] - 1) * dilations[0] - 1) // stride[0] + 1
    out_w = (shape[2] + pads[1] + pads[3] - (kernel[1] - 1) * dilations[1] - 1) // stride[1] + 1
    model = _model("FLOAT32", 16777217)
    model.set_tensor_shape("x", shape)
    model.set_tensor_shape("y", [1, out_h, out_w, kernel[0] * kernel[1]])
    node = model.graph.node[0]
    for key, value in [
        ("input_shape", str(tuple(shape))),
        ("kernel_size", kernel),
        ("stride", stride),
        ("pad_amount", pads),
        ("dilations", dilations),
    ]:
        for attr in list(node.attribute):
            if attr.name == key:
                node.attribute.remove(attr)
        node.attribute.append(helper.make_attribute(key, value))
    values = np.arange(1, shape[1] * shape[2] + 1, dtype=np.float32).reshape(shape)
    if axis is not None:
        with pytest.raises(AssertionError, match="^Im2Col dtype must allow pad_val$"):
            execute_onnx(model, {"x": values})
    else:
        result = execute_onnx(model, {"x": values})["y"]
        expected = values[:, :: dilations[0], :: dilations[1], :].reshape(1, 1, 1, -1)
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == values.dtype


def test_unused_requested_padding_keeps_the_upstream_datatype_guard() -> None:
    model = _model("INT8", 128)
    node = model.graph.node[0]
    for attr in node.attribute:
        if attr.name == "stride":
            attr.ints[:] = [4, 4]
        elif attr.name == "pad_amount":
            attr.ints[:] = [0, 0, 1, 1]
    with pytest.raises(AssertionError, match="^Im2Col dtype must allow pad_val$"):
        execute_onnx(model, {"x": np.ones((1, 2, 2, 1), dtype=np.int8)})
