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

"""FloatQuant's output range (get_output_range), stated with its datatype inside an
inference pass, against what execute_node computes."""

from __future__ import annotations

import pytest

import math
import numpy as np
import warnings
from fractions import Fraction
from onnx import TensorProto, helper
from typing import Any, Optional

from qonnx.core import ranges
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.core.ranges import Range
from qonnx.core.tensor_ranges import RangeContradiction
from qonnx.custom_op.general.floatquant import clip_value, compute_max_val
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import qonnx_make_model

INF = math.inf


def quantizer(
    x_range: Optional[Range] = None,
    scale: Any = 1.0,
    max_val: float = 448.0,
    constant: tuple[str, ...] = ("s", "e", "m", "b", "mv"),
    exact_ranges: bool = True,
    exponent: float = 4.0,
    mantissa: float = 3.0,
    bias: float = 7.0,
    **attributes: int,
) -> ModelWrapper:
    """x (FLOAT32, ``x_range`` stated) -> FloatQuant to FP8 E4M3 (bias 7), or the
    given widths and bias, -> y; the inputs named in ``constant`` initializers, the
    others graph inputs."""
    attributes = {"has_inf": 0, "has_nan": 0, "has_subnormal": 1, "saturation": 1, **attributes}
    names = ["x", "s", "e", "m", "b", "mv"]
    node = helper.make_node("FloatQuant", names, ["y"], domain="qonnx.custom_op.general", **attributes)
    values = {"s": scale, "e": exponent, "m": mantissa, "b": bias, "mv": max_val}
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])]
    inputs += [helper.make_tensor_value_info(name, TensorProto.FLOAT, []) for name in names[1:] if name not in constant]
    graph = helper.make_graph([node], "fq", inputs, [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])])
    model = ModelWrapper(qonnx_make_model(graph))
    for name in constant:
        model.set_initializer(name, np.asarray(values[name], dtype=np.float32))
    if x_range is not None:
        model.set_tensor_datatype("x", DataType["FLOAT32"], range=x_range)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RangeContradiction)
        return model.transform(InferDataTypes(exact_ranges=exact_ranges))


def test_clip_value_is_the_format_s_largest_or_max_val_where_less() -> None:
    assert clip_value(4, 3, 7) == compute_max_val(4, 3, 7) == 480
    assert clip_value(4, 3, 7, np.float32(448)) == 448
    assert clip_value(4, 3, 7, np.float32(1000)) == 480


def test_a_saturating_floatquant_states_its_clip_value() -> None:
    model = quantizer(Range(-1000, 1000))
    assert model.get_tensor_datatype("y") == DataType["FLOAT<4,3,7>"]
    assert model.get_tensor_range("y") == Range(-448, 448)
    # without max_val's clip, the format's largest value (no code reserved): 480
    assert quantizer(Range(-1000, 1000), max_val=1000.0).get_tensor_range("y") == Range(-480, 480)


def test_the_default_states_no_range() -> None:
    model = quantizer(Range(-1000, 1000), exact_ranges=False)
    assert model.get_tensor_datatype("y") == DataType["FLOAT<4,3,7>"]
    assert model.get_stated_tensor_range("y") is None


def test_a_floatquant_outputs_nan_only_where_its_execution_can() -> None:
    # an open FLOAT32 input may be NaN or infinite, and execution makes NaN of an infinity
    assert quantizer().get_tensor_range("y") == Range(-448, 448, nan=True)
    assert quantizer(Range(0, INF)).get_tensor_range("y") == Range(-448, 448, nan=True)
    assert quantizer(Range(-1, 1, nan=True)).get_tensor_range("y") == Range(-448, 448, nan=True)
    # x / scale may overflow float32
    big = ranges.endpoint(3e38)
    assert quantizer(Range(-big, big), scale=0.5).get_tensor_range("y").nan
    assert not quantizer(Range(-big, big), scale=2.0).get_tensor_range("y").nan
    # a scale holding 0 divides by it
    assert quantizer(Range(-1, 1), scale=[0.0, 1.0]).get_tensor_range("y").nan


def test_a_floatquant_s_range_is_its_clip_value_times_its_scale_as_executed() -> None:
    # a non-unit scale: FLOAT32, and 448 * 0.1 rounded into float32, as the product is
    model = quantizer(Range(-1000, 1000), scale=0.1)
    assert model.get_tensor_datatype("y") == DataType["FLOAT32"]
    end = ranges.endpoint(float(np.float32(448) * np.float32(0.1)))
    assert model.get_tensor_range("y") == Range(-end, end)
    # per channel: the largest magnitude, of either sign
    assert quantizer(Range(-1, 1), scale=[0.5, -2.0]).get_tensor_range("y") == Range(-896, 896)


@pytest.mark.parametrize(
    "attributes, expected",
    [
        ({"saturation": 0, "has_inf": 1}, Range(-INF, INF)),
        ({"saturation": 0, "has_inf": 1, "has_nan": 1}, Range(-INF, INF)),
        ({"saturation": 0, "has_nan": 1}, Range(-448, 448, nan=True)),
        # has_inf and has_nan matter only when not saturating
        ({"saturation": 1, "has_inf": 1, "has_nan": 1}, Range(-448, 448)),
    ],
)
def test_a_floatquant_that_does_not_saturate_states_infinity_or_nan_beyond_its_clip_value(
    attributes: dict[str, int], expected: Range
) -> None:
    model = quantizer(Range(-1000, 1000), **attributes)
    assert model.get_tensor_range("y") == expected


@pytest.mark.parametrize(
    "x_range, attributes",
    [
        (Range(-1000, 1000), {}),
        (None, {}),
        (Range(-1000, 1000), {"saturation": 0, "has_inf": 1}),
        (Range(-1000, 1000), {"saturation": 0, "has_nan": 1}),
    ],
)
def test_a_floatquant_s_stated_range_holds_what_it_executes(x_range: Optional[Range], attributes: dict[str, int]) -> None:
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


def test_a_floatquant_whose_parameters_are_not_constants_states_no_range() -> None:
    model = quantizer(Range(-1, 1), constant=("s", "e", "m", "b"))
    assert model.get_tensor_datatype("y") == DataType["FLOAT32"]
    assert model.get_stated_tensor_range("y") is None


# NaN from float_quant's internal scale, 2 ** max(floor(log2(|x| + tiny)) - m, 1 - bias - m)

FMAX = ranges.endpoint(float(np.finfo(np.float32).max))
TINY = Fraction(1, 2**149)


@pytest.mark.parametrize(
    "x_range, widths, nan, executes_nan",
    [
        # log2 of float32's largest rounds up to 128: 2**128 is inf, and inf * 0 NaN
        (Range(FMAX, FMAX), (4.0, 0.0, 7.0, 256.0), True, True),
        (Range(-FMAX, FMAX), (4.0, 0.0, 7.0, 256.0), True, True),
        # conservative: 2**127 executes to 256, but its log2 is not proved not to round up
        (Range(-(2**127), 2**127), (4.0, 0.0, 7.0, 256.0), True, False),
        (Range(-(2**126), 2**126), (4.0, 0.0, 7.0, 256.0), False, False),
        # at 0, 2**(-126 - 24) underflows to 0: 0 / 0, then 0 * NaN
        (Range(0, 1), (8.0, 24.0, 127.0, 2.0**100), True, True),
        (Range(-TINY, TINY), (8.0, 23.0, 127.0, 2.0**100), False, False),
        (Range(0, 1), (8.0, 23.0, 127.0, 2.0**100), False, False),
        (Range(0, 0), (8.0, 30.0, 200.0, 2.0**40), True, True),
        (Range(-1000, 1000), (4.0, 3.0, 7.0, 448.0), False, False),
    ],
)
def test_nan_is_stated_unless_the_internal_scale_is_proved_finite_and_nonzero(
    x_range: Range, widths: tuple[float, float, float, float], nan: bool, executes_nan: bool
) -> None:
    exponent, mantissa, bias, max_val = widths
    model = quantizer(x_range, max_val=max_val, exponent=exponent, mantissa=mantissa, bias=bias)
    # the ends are the clip value's, NaN or not
    end = ranges.endpoint(float(clip_value(exponent, mantissa, bias, np.float32(max_val))))
    assert model.get_tensor_range("y") == Range(-end, end, nan)
    # execution at the range's ends, float32's largest, 0 and the smallest subnormal
    points = [x_range.least, x_range.greatest, -FMAX, FMAX, 0, -TINY, TINY]
    xs = np.resize(np.array([float(v) for v in points if x_range.least <= v <= x_range.greatest], np.float32), 4)
    with np.errstate(all="ignore"):
        computed = execute_onnx(model, {"x": xs})["y"]
    assert bool(np.isnan(computed).any()) == executes_nan
    held = computed[~np.isnan(computed)]
    assert all(-end <= ranges.endpoint(float(value)) <= end for value in held)
