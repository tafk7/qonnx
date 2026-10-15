# Copyright (c) 2024 Advanced Micro Devices, Inc.
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

import math
import numpy as np
import warnings
from onnx import TensorProto, helper

from qonnx.core import ranges, tensor_ranges
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.core.ranges import RangeContradiction, TensorRange
from qonnx.custom_op.general.floatquant import compute_max_val
from qonnx.custom_op.general.floatquant import float_quant as qonnx_float_quant
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import qonnx_make_model


def test_compute_max_val():
    # reference max normal values from OCP MX 1.0 standard
    assert compute_max_val(2, 3) == 7.5  # FP6 E2M3
    assert compute_max_val(3, 2) == 28.0  # FP6 E3M2
    assert compute_max_val(2, 1) == 6.0  # FP4 E2M1


def test_float_quantize():
    zero_tensor = np.zeros((2, 2))
    unit_scale = np.asarray([1.0], dtype=np.float32)
    assert np.all(qonnx_float_quant(zero_tensor, unit_scale, 2, 3) == zero_tensor)
    testcase_a = np.asarray([1.5], dtype=np.float32)
    testcase_b = np.asarray([3.25], dtype=np.float32)
    testcase_c = np.asarray([8.0], dtype=np.float32)
    testcase_d = np.asarray([28.2], dtype=np.float32)
    testcase_e = np.asarray([6.1], dtype=np.float32)
    assert np.all(qonnx_float_quant(testcase_a, unit_scale, 2, 3) == testcase_a)
    assert np.all(qonnx_float_quant(testcase_b, unit_scale, 2, 3) == testcase_b)
    assert np.all(qonnx_float_quant(testcase_c, unit_scale, 2, 3) == compute_max_val(2, 3))
    assert np.all(qonnx_float_quant(testcase_d, unit_scale, 3, 2) == compute_max_val(3, 2))
    assert np.all(qonnx_float_quant(testcase_e, unit_scale, 2, 1) == compute_max_val(2, 1))


INF = math.inf


def quantizer(x_range=None, scale=1.0, max_val=448.0, constant=("s", "e", "m", "b", "mv"), **attributes):
    """x (FLOAT32, ``x_range`` stated) -> FloatQuant to FP8 E4M3 (bias 7) -> y; the
    inputs named in ``constant`` initializers, the others graph inputs."""
    attributes = {"has_inf": 0, "has_nan": 0, "has_subnormal": 1, "saturation": 1, **attributes}
    names = ["x", "s", "e", "m", "b", "mv"]
    node = helper.make_node("FloatQuant", names, ["y"], domain="qonnx.custom_op.general", **attributes)
    values = {"s": scale, "e": 4.0, "m": 3.0, "b": 7.0, "mv": max_val}
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])]
    inputs += [helper.make_tensor_value_info(name, TensorProto.FLOAT, []) for name in names[1:] if name not in constant]
    graph = helper.make_graph([node], "fq", inputs, [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])])
    model = ModelWrapper(qonnx_make_model(graph))
    for name in constant:
        model.set_initializer(name, np.asarray(values[name], dtype=np.float32))
    if x_range is not None:
        model.set_tensor_datatype("x", DataType["FLOAT32"], x_range)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RangeContradiction)
        return model.transform(InferDataTypes())


def test_a_saturating_floatquant_states_its_clip_value():
    model = quantizer(TensorRange(-1000, 1000))
    assert model.get_tensor_datatype("y") == DataType["FLOAT<4,3,7>"]
    assert model.get_tensor_range("y") == TensorRange(-448, 448)
    # without max_val's clip, the format's largest value (no code reserved): 480
    assert quantizer(TensorRange(-1000, 1000), max_val=1000.0).get_tensor_range("y") == TensorRange(-480, 480)


def test_a_floatquant_outputs_nan_only_where_its_execution_can():
    # an open FLOAT32 input may be NaN or infinite, and execution makes NaN of an infinity
    assert quantizer().get_tensor_range("y") == TensorRange(-448, 448, nan=True)
    assert quantizer(TensorRange(0, INF)).get_tensor_range("y") == TensorRange(-448, 448, nan=True)
    assert quantizer(TensorRange(-1, 1, nan=True)).get_tensor_range("y") == TensorRange(-448, 448, nan=True)
    # x / scale may overflow float32
    big = ranges.endpoint(3e38)
    assert quantizer(TensorRange(-big, big), scale=0.5).get_tensor_range("y").nan
    assert not quantizer(TensorRange(-big, big), scale=2.0).get_tensor_range("y").nan
    # a scale holding 0 divides by it
    assert quantizer(TensorRange(-1, 1), scale=[0.0, 1.0]).get_tensor_range("y").nan


def test_a_floatquant_s_range_is_its_clip_value_times_its_scale_as_executed():
    # a non-unit scale: FLOAT32, and 448 * 0.1 rounded into float32, as the product is
    model = quantizer(TensorRange(-1000, 1000), scale=0.1)
    assert model.get_tensor_datatype("y") == DataType["FLOAT32"]
    end = ranges.endpoint(float(np.float32(448) * np.float32(0.1)))
    assert model.get_tensor_range("y") == TensorRange(-end, end)
    # per channel: the largest magnitude, of either sign
    assert quantizer(TensorRange(-1, 1), scale=[0.5, -2.0]).get_tensor_range("y") == TensorRange(-896, 896)


@pytest.mark.parametrize(
    "attributes, expected",
    [
        ({"saturation": 0, "has_inf": 1}, TensorRange(-INF, INF)),
        ({"saturation": 0, "has_inf": 1, "has_nan": 1}, TensorRange(-INF, INF)),
        ({"saturation": 0, "has_nan": 1}, TensorRange(-448, 448, nan=True)),
        # has_inf and has_nan matter only when not saturating
        ({"saturation": 1, "has_inf": 1, "has_nan": 1}, TensorRange(-448, 448)),
    ],
)
def test_a_floatquant_that_does_not_saturate_states_infinity_or_nan_beyond_its_clip_value(attributes, expected):
    model = quantizer(TensorRange(-1000, 1000), **attributes)
    assert model.get_tensor_range("y") == expected


@pytest.mark.parametrize(
    "x_range, attributes",
    [
        (TensorRange(-1000, 1000), {}),
        (None, {}),
        (TensorRange(-1000, 1000), {"saturation": 0, "has_inf": 1}),
        (TensorRange(-1000, 1000), {"saturation": 0, "has_nan": 1}),
    ],
)
def test_a_floatquant_s_stated_range_holds_what_it_executes(x_range, attributes):
    for scale in (1.0, 0.1, -0.375):
        model = quantizer(x_range, scale=scale, **attributes)
        stated = model.get_tensor_range("y")
        xs = np.array([-1000.0, -448.5, 1e-9, 1000.0], dtype=np.float32)
        if x_range is None:
            xs = np.array([np.inf, -np.inf, np.nan, 3e38], dtype=np.float32)
        with np.errstate(all="ignore"):
            computed = ranges.values_range(execute_onnx(model, {"x": xs})["y"])
        assert stated.least <= computed.least and computed.greatest <= stated.greatest
        assert stated.nan or not computed.nan


def test_a_floatquant_whose_parameters_are_not_constants_states_no_range():
    model = quantizer(TensorRange(-1, 1), constant=("s", "e", "m", "b"))
    assert model.get_tensor_datatype("y") == DataType["FLOAT32"]
    assert tensor_ranges.stated(model, "y") is None
