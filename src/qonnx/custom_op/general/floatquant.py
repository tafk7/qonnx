# Copyright (c) 2024 Nicolo Ghielmetti
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

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from fractions import Fraction
from onnx import TensorProto, helper
from typing import TYPE_CHECKING, Any, Optional, Union

from qonnx.core import tensor_ranges
from qonnx.core.datatype import BINARY32, BINARY64, DataType
from qonnx.core.ranges import Range, endpoint, unbounded, values_range
from qonnx.custom_op.base import CustomOp
from qonnx.custom_op.general.quant import resolve_rounding_mode

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper


def compute_default_exponent_bias(exponent_bitwidth):
    return (2.0 ** (exponent_bitwidth - 1)) - 1


def compute_max_val(exponent_bitwidth, mantissa_bitwidth, exponent_bias=None):
    if exponent_bias is None:
        exponent_bias = compute_default_exponent_bias(exponent_bitwidth)
    max_exponent = (2.0**exponent_bitwidth) - 1.0 - exponent_bias
    max_mantissa = np.sum((2.0 ** np.arange(0, -1.0 * mantissa_bitwidth - 1.0, -1.0)))
    max_val = max_mantissa * (2**max_exponent)
    return max_val


def clip_value(
    exponent_bitwidth: npt.ArrayLike,
    mantissa_bitwidth: npt.ArrayLike,
    exponent_bias: Optional[npt.ArrayLike] = None,
    max_val: Optional[npt.ArrayLike] = None,
) -> Union[float, npt.NDArray[Any]]:
    """The largest magnitude float_quant outputs before its scale: the format's
    largest value (compute_max_val), or max_val where that is less."""
    max_value = compute_max_val(exponent_bitwidth, mantissa_bitwidth, exponent_bias)
    return max_value if max_val is None else np.minimum(max_value, max_val)


def float_quant(
    X: npt.NDArray[np.floating[Any]],
    scale: npt.ArrayLike,
    exponent_bitwidth: npt.ArrayLike,
    mantissa_bitwidth: npt.ArrayLike,
    exponent_bias: Optional[npt.ArrayLike] = None,
    signed: bool = True,
    max_val: Optional[npt.ArrayLike] = None,
    has_inf: bool = False,
    has_nan: bool = False,
    has_subnormal: bool = False,
    rounding_mode: str = "ROUND",
    saturation: bool = True,
) -> Union[npt.NDArray[np.floating[Any]], np.floating[Any]]:
    # the comments are left to track the correspondence with the brevitas code
    # np version of brevitas function
    def inf_nan_clamp(
        X: npt.NDArray[np.floating[Any]],
        inf_mask: npt.NDArray[np.bool_],
        p_max_val_mask: npt.NDArray[np.bool_],
        n_max_val_mask: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.floating[Any]]:
        if has_inf:
            X[p_max_val_mask] = np.inf
            X[n_max_val_mask] = -np.inf
        elif has_nan:
            full_max_val_mask = np.logical_or(p_max_val_mask, n_max_val_mask)
            X[full_max_val_mask] = np.nan
            X[inf_mask] = np.nan
        else:
            raise RuntimeError("Clamping is not saturating, but neither `inf_values` nor `nan_values` is specified")
        return X

    # consistency check
    # if bit_width != exponent_bitwidth + mantissa_bitwidth + int(signed):
    #         raise RuntimeError("Mismatch between total bit-width, exponent, mantissa and sign.")

    # x = self.input_view_impl(x) # assuming input_view_impl is Identity

    # the following lines (up to max_value assignment) implements the float_internal_scale function from brevitas using numpy
    # internal_scale = float_internal_scale(
    #     scaled_x, self.mantissa_bit_width(), self.fp_internal_scale_min(), self.eps)
    if exponent_bias is None:
        exponent_bias = compute_default_exponent_bias(exponent_bitwidth)
    X = X / scale

    eps = np.finfo(X.dtype).tiny  # the datatype used here and in brevitas must be the same to have the same eps
    fp_internal_scale_min = 1.0 - exponent_bias - mantissa_bitwidth

    internal_scale = np.floor(np.log2(np.abs(X) + eps)) - mantissa_bitwidth
    internal_scale = np.maximum(
        internal_scale, fp_internal_scale_min
    )  # np version of: internal_scale = torch.ok(internal_scale, fp_internal_scale_min)
    internal_scale = np.exp2(internal_scale)

    x_q = internal_scale * resolve_rounding_mode(rounding_mode)(
        X / internal_scale
    )  # self.float_to_int_impl(x / internal_scale)

    max_value = clip_value(exponent_bitwidth, mantissa_bitwidth, exponent_bias, max_val)
    min_value = 0.0 if not signed else -max_value

    # Compute masks
    inf_mask = np.isinf(x_q)
    p_max_val_mask = x_q > max_value
    n_max_val_mask = x_q < min_value

    # first clamp everything to  [min_value,max_value], basically the saturating case
    x_q = np.clip(x_q, min_value, max_value)  # self.saturating_clamp(x_q, max_value, min_value)

    if not saturation:
        x_q = inf_nan_clamp(x_q, inf_mask, p_max_val_mask, n_max_val_mask)

    return x_q * scale  # , self.saturating, self.inf_values, self.nan_values


# For float32 and float64, the exponents of the smallest subnormal, the smallest
# normal and the largest power of two.
_EXPONENTS = ((-149, -126, 127), (-1074, -1022, 1023))


def _floor_log2(value: Fraction) -> int:
    """floor(log2(value)) for value > 0, exactly."""
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    return exponent - 1 if Fraction(2) ** exponent > value else exponent


def _internal_scale_finite(largest: Fraction, mantissa_bitwidth: npt.ArrayLike, exponent_bias: npt.ArrayLike) -> bool:
    """Whether float_quant's internal scale, 2 ** max(floor(log2(|x| + tiny)) - m,
    1 - bias - m), is finite and nonzero for every |x| up to ``largest`` (the input
    over its scale), computed in float32 or in float64. Where it is not, execution
    makes NaN: inf * 0 when it overflows, 0 / 0 and 0 * inf when it underflows.

    Conservative: the computed log2 may round up to the next integer, |x| + tiny is
    at least tiny, and a mantissa width or bias that is not an integer keeps a
    margin of one binade at each end. A width or bias that is not finite is not
    proved."""
    with np.errstate(all="ignore"):
        widths = np.asarray(mantissa_bitwidth, dtype=np.float64)
        biases = np.asarray(exponent_bias, dtype=np.float64)
    if widths.size == 0 or biases.size == 0 or not (np.isfinite(widths).all() and np.isfinite(biases).all()):
        return False
    margin = 0 if (widths == np.round(widths)).all() and (biases == np.round(biases)).all() else 1
    m_least, m_greatest = float(widths.min()), float(widths.max())
    b_least, b_greatest = float(biases.min()), float(biases.max())
    for subnormal, normal, top in _EXPONENTS:
        floor_greatest = normal + 1 if largest == 0 else max(_floor_log2(largest) + 1, normal + 1)
        least = max(normal - m_greatest, 1 - b_greatest - m_greatest)
        greatest = max(floor_greatest - m_least, 1 - b_least - m_least)
        if least < subnormal + margin or greatest > top - margin:
            return False
    return True


class FloatQuant(CustomOp):
    """Floating point quantization operation for QONNX.

    The output is a tensor of the same shape as the input tensor, with quantized
    values.
    """

    def get_nodeattr_types(self):
        return {
            # Integer value interpreted as boolean, defines whether the representation supports signed values.
            # This attribute has no effect on the execution of this operation and is intended purely to inform backends.
            # "signed": ("i", True, 1),
            # Defines how rounding should be applied during quantization.
            # Currently available modes are: "ROUND", "CEIL" and "FLOOR". Here "ROUND" implies a round-to-even operation.
            # Lowercase variants for the rounding mode string are also supported: "round", "ceil", "floor".
            "rounding_mode": ("s", False, "ROUND"),
            # Integer value interpreted as boolean, defines whether the representation supports infinity values.
            # The ability to represent infinity values will decrease the representable numerical range.
            # This attribute has no effect on the execution of this operation and is intended purely to inform backends.
            "has_inf": ("i", True, 0),
            # Integer value interpreted as boolean, defines whether the representation supports not-a-number (NaN) values.
            # The ability to represent NaN values will decrease the representable numerical range.
            # This attribute has no effect on the execution of this operation and is intended purely to inform backends.
            "has_nan": ("i", True, 0),
            # # Integer value interpreted as boolean, defines whether the representation supports subnormal values.
            # Subnormal values have an exponent value of 0 and
            # are interpreted to have a leading significand digit of zero rather than one.
            # Supporting subnormals will increase the complexity of the required arithmetic datapath.
            # This attribute has no effect on the execution of this operation and is intended purely to inform backends.
            "has_subnormal": ("i", True, 0),
            # Integer value interpreted as boolean, defines whether the representation will saturate during arithmetic.
            # This attribute has no effect on the execution of this operation and is intended purely to inform backends.
            "saturation": ("i", True, 1),
        }

    def execute_node(self, context, graph):
        node = self.onnx_node
        # save inputs
        inp_tensor = context[node.input[0]]
        scale = context[node.input[1]]
        exponent_bitwidth = context[node.input[2]]
        mantissa_bitwidth = context[node.input[3]]
        exponent_bias = context[node.input[4]]
        max_val = context[node.input[5]]
        # save attributes
        # signed = self.get_nodeattr("signed")
        signed = True
        rounding_mode = self.get_nodeattr("rounding_mode")
        has_inf = self.get_nodeattr("has_inf")
        has_nan = self.get_nodeattr("has_nan")
        has_subnormal = self.get_nodeattr("has_subnormal")  # not supported in Brevitas, so not supported for the moment
        saturation = self.get_nodeattr("saturation")

        # calculate output
        ret = float_quant(
            inp_tensor,
            scale,
            exponent_bitwidth,
            mantissa_bitwidth,
            exponent_bias,
            signed,
            max_val,
            has_inf,
            has_nan,
            has_subnormal,
            rounding_mode,
            saturation,
        )  # signed, max_val, rounding_mode, has_inf, has_nan, saturating)
        # ensure output is ndarray (even if 0d)
        # since numpy silently flattens 0d arrays to scalars
        # more: https://github.com/numpy/numpy/issues/13105
        if not isinstance(ret, np.ndarray):
            ret = np.asarray(ret, dtype=np.float32)
        if not ret.dtype == np.float32:
            ret = ret.astype(np.float32)
        # set context according to output name
        context[node.output[0]] = ret

    def make_shape_compatible_op(self, model):
        """Returns a standard ONNX op which is compatible with this CustomOp
        for performing shape inference."""
        return helper.make_node(
            "Cast",
            inputs=[self.onnx_node.input[0]],
            outputs=[self.onnx_node.output[0]],
            to=int(TensorProto.FLOAT),
        )

    def get_output_dtype(self, model):
        node = self.onnx_node
        # scale, zero-point and bitwidth must be read from initializers
        scale = model.get_initializer(node.input[1])
        exponent_bitwidth = model.get_initializer(node.input[2])
        mantissa_bitwidth = model.get_initializer(node.input[3])
        expoent_bias = model.get_initializer(node.input[4])
        max_val = model.get_initializer(node.input[5])
        assert scale is not None, "Found unspecified scale for FloatQuant node: " + str(node)
        assert exponent_bitwidth is not None, "Found unspecified exponent width for FloatQuant node: " + str(node)
        assert mantissa_bitwidth is not None, "Found unspecified mantissa width for FloatQuant node: " + str(node)
        assert expoent_bias is not None, "Found unspecified exponent bias for FloatQuant node: " + str(node)
        assert max_val is not None, "Found unspecified maximum value for FloatQuant node: " + str(node)
        # extract the exponent and mantissa widths (assume scalar)
        assert exponent_bitwidth.ndim == 0, "Exponent width must be scalar for FloatQuant node: " + str(node)
        assert mantissa_bitwidth.ndim == 0, "Mantissa width must be scalar for FloatQuant node: " + str(node)
        exponent_bitwidth = exponent_bitwidth.item()
        mantissa_bitwidth = mantissa_bitwidth.item()
        assert int(exponent_bitwidth) == exponent_bitwidth, "Exponent width must be integer for FloatQuant node: " + str(
            node
        )
        assert int(mantissa_bitwidth) == mantissa_bitwidth, "Mantissa width must be integer for FloatQuant node: " + str(
            node
        )
        exponent_bitwidth = int(exponent_bitwidth)
        mantissa_bitwidth = int(mantissa_bitwidth)
        # extract the exponent bias (assume scalar)
        assert expoent_bias.ndim == 0, "Exponent bias must be scalar for FloatQuant node: " + str(node)
        expoent_bias = expoent_bias.item()
        assert int(expoent_bias) == expoent_bias, "Exponent bias must be integer for FloatQuant node: " + str(node)
        expoent_bias = int(expoent_bias)
        # extract the maximum value (assume scalar)
        assert max_val.ndim == 0, "Maximum value must be scalar for FloatQuant node: " + str(node)
        max_val = max_val.item()
        # ensure unit scale
        unit_scale = np.all(scale == 1.0)
        assert unit_scale, "Only scale=1 FloatQuant nodes supported for now"
        # determine the FINN DataType
        finn_dt = DataType[f"FLOAT<{exponent_bitwidth},{mantissa_bitwidth},{expoent_bias}>"]
        return finn_dt

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """The output's datatype is FLOAT<e,m,b> for a unit scale (get_output_dtype),
        else FLOAT32. Inside an inference pass (qonnx.core.tensor_ranges.inferring), its
        range is stated with it (get_output_range)."""
        try:
            finn_dt = self.get_output_dtype(model)
        except AssertionError:
            finn_dt = DataType["FLOAT32"]
        node = self.onnx_node
        value = self.get_output_range(model) if tensor_ranges.inferring(model) else None
        model.set_tensor_datatype(node.output[0], finn_dt, range=value)

    def get_output_range(self, model: ModelWrapper) -> Optional[Range]:
        """The values the node outputs, as execute_node computes them in float32.

        M, the clip value (clip_value of the exponent and mantissa widths, the
        exponent bias and max_val), bounds each value before its scale, and the
        output is that value times the scale, rounded into float32. So:

        - saturating: [-M * s, M * s], s the scale's largest magnitude, the ends
          rounded into float32 as the product is;
        - not saturating, with has_inf: unbounded, its values beyond M becoming
          infinities;
        - not saturating, with has_nan only: as saturating, NaN possible (its
          values beyond M becoming NaN).

        NaN is possible where the input may be NaN or infinite, or the input over
        the scale may overflow float32 or divide by 0 (execution makes NaN of an
        infinity), or float_quant's internal scale may overflow or underflow over
        the input's range (_internal_scale_finite: unless proved finite and nonzero,
        NaN is possible). None (no range stated) where the scale, the widths, the bias or
        max_val is not an initializer, where M or the scale is not exact in
        float32, where the input is held in another type than float32 or
        float64, or where execution raises (neither saturating, has_inf nor
        has_nan)."""
        node = self.onnx_node
        constants = [model.get_initializer(name) if name else None for name in node.input[1:6]]
        if len(constants) < 5 or any(value is None for value in constants):
            return None
        scale, exponent_bitwidth, mantissa_bitwidth, exponent_bias, max_val = constants
        saturation, has_inf, has_nan = (self.get_nodeattr(key) for key in ("saturation", "has_inf", "has_nan"))
        if not (saturation or has_inf or has_nan):
            return None
        held = tensor_ranges.element_type(model, node.input[0])
        if held not in (None, TensorProto.FLOAT, TensorProto.DOUBLE) or scale.dtype != np.float32:
            return None
        with np.errstate(all="ignore"):
            clip = np.abs(np.asarray(clip_value(exponent_bitwidth, mantissa_bitwidth, exponent_bias, max_val)))
        scales = values_range(np.abs(scale))
        if np.isnan(clip).any() or scales.nan or isinstance(scales.greatest, float):
            return unbounded(True)
        bound = endpoint(float(clip.max()))
        if BINARY32.nearest(bound) != bound:
            return None
        x = model.get_tensor_range(node.input[0])
        divided = BINARY64 if held == TensorProto.DOUBLE else BINARY32
        largest = max(abs(x.least), abs(x.greatest))
        nan = (
            x.nan
            or isinstance(largest, float)
            or scales.least == 0
            or isinstance(divided.nearest(Fraction(largest) / Fraction(scales.least)), float)
            or not _internal_scale_finite(Fraction(largest) / Fraction(scales.least), mantissa_bitwidth, exponent_bias)
        )
        if scales.greatest == 0 or bound == 0:
            end = 0
        elif isinstance(bound, float):
            end = bound
        else:
            end = BINARY32.nearest(Fraction(bound) * Fraction(scales.greatest))
        if saturation:
            return Range(-end, end, nan)
        if has_inf:
            return unbounded(nan)
        return Range(-end, end, True)

    def verify_node(self) -> None:
        pass
