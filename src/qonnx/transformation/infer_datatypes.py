# Copyright (c) 2020 Xilinx, Inc.
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
# * Neither the name of Xilinx nor the names of its
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

import math
import numpy as np
from collections import Counter, OrderedDict
from fractions import Fraction
from onnx import NodeProto, TensorProto, helper
from typing import TYPE_CHECKING, Any, Callable, Hashable, Optional

from qonnx.core import ranges, tensor_ranges
from qonnx.core.datatype import BaseDataType, DataType, ScaledIntType
from qonnx.core.ranges import TensorRange
from qonnx.custom_op.registry import is_custom_op
from qonnx.transformation.base import Transformation
from qonnx.transformation.qcdq_to_qonnx import extract_elem_type
from qonnx.util.basic import get_by_name

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

# The integers a tensor may hold, both ends included.
Range = tuple[int, int]

# The bounds computed over initializers (a MatMul's, Gemm's or Conv's sums over its
# weights; an initializer's own range), kept in the process by what they read: the
# rule, the weights' content digest and the input's range. Least recently used first.
_BOUNDS: OrderedDict[Hashable, Any] = OrderedDict()
_BOUNDS_KEPT = 4096
bounds_counts: Counter[str] = Counter()
"""How often each kind of bound was computed, and answered from the cache ("<kind> computed",
"<kind> cached")."""


def _cached(key: tuple[Hashable, ...], compute: Callable[[], Any]) -> Any:
    try:
        value = _BOUNDS[key]
    except KeyError:
        bounds_counts[f"{key[0]} computed"] += 1
        value = compute()
        _BOUNDS[key] = value
        if len(_BOUNDS) > _BOUNDS_KEPT:
            _BOUNDS.popitem(last=False)
        return value
    _BOUNDS.move_to_end(key)
    bounds_counts[f"{key[0]} cached"] += 1
    return value


def int_type_holding(lo: int, hi: int) -> BaseDataType:
    """The smallest INTn or UINTn holding every integer in [lo, hi]: unsigned when
    lo is not negative."""
    if lo >= 0:
        return DataType["UINT%d" % max(1, hi.bit_length())]
    return DataType["INT%d" % (max((-lo - 1).bit_length(), max(hi, 0).bit_length()) + 1)]


def _integer_values(values: np.ndarray) -> Optional[Range]:
    if values.size == 0 or not np.all(np.isfinite(values)) or not np.all(values == np.round(values)):
        return None
    return int(values.min()), int(values.max())


def _range(model: ModelWrapper, name: str) -> Optional[Range]:
    """The integers ``name`` may hold: an initializer's by its values, any other
    tensor's by its range (qonnx.core.ranges: the one it states with its
    datatype, else its datatype's). None for an initializer holding a
    non-integer."""
    key = tensor_ranges.digest(model, name)
    if key is None:
        found = tensor_ranges.tensor_range(model, name)
        if found.is_integer():
            return found.least, found.greatest  # type: ignore[return-value]
        dtype = tensor_ranges.datatype(model, name)
        return int(dtype.min()), int(dtype.max())
    return _cached(("integers", key), lambda: _integer_values(model.get_initializer(name)))


def _tensor_range(model: ModelWrapper, name: str) -> TensorRange:
    """The values ``name`` may hold, for any datatype: an initializer's by its
    values, any other tensor's range."""
    key = tensor_ranges.digest(model, name)
    if key is None:
        return tensor_ranges.tensor_range(model, name)
    return _cached(("values", key), lambda: ranges.values_range(model.get_initializer(name)))


def _sum(x: Range, y: Range) -> Range:
    return x[0] + y[0], x[1] + y[1]


def _product(x: Range, y: Range) -> Range:
    corners = [x[0] * y[0], x[0] * y[1], x[1] * y[0], x[1] * y[1]]
    return min(corners), max(corners)


def _integers(values: np.ndarray, count: int) -> np.ndarray:
    """Integral ``values`` as integers whose sums of ``count`` terms cannot overflow:
    int64 where that holds, Python integers otherwise."""
    if values.size == 0 or int(np.abs(values).max()) * max(count, 1) < 2**62:
        return values.astype(np.int64)
    return np.array([int(v) for v in values.flat], dtype=object).reshape(values.shape)


def _dot_bounds(weights: np.ndarray, axes: tuple[int, ...], x: Range) -> tuple[np.ndarray, np.ndarray]:
    """For each position of ``weights`` outside ``axes``, the least and the greatest
    sum over ``axes`` of weight * x, x anywhere in its range: a positive weight's
    term is greatest at x's top, a negative one's at x's bottom."""
    weights = _integers(weights, math.prod(weights.shape[axis] for axis in axes))
    positive = np.where(weights > 0, weights, 0).sum(axis=axes).astype(object)
    negative = np.where(weights < 0, weights, 0).sum(axis=axes).astype(object)
    return x[0] * positive + x[1] * negative, x[1] * positive + x[0] * negative


def _span(lows: np.ndarray, highs: np.ndarray) -> Range:
    return int(np.min(lows)), int(np.max(highs))


def _depth(model: ModelWrapper, name: str, axis: int) -> Optional[int]:
    """The length of ``name``'s ``axis``, None while it is unknown."""
    shape = model.get_tensor_shape(name)
    if not shape or not -len(shape) <= axis < len(shape) or shape[axis] <= 0:
        return None
    return shape[axis]


def _dot(model: ModelWrapper, x: str, w: str, axis: int, depth: Optional[int]) -> Optional[Range]:
    """The range of sums of ``depth`` products of x and w: from w's values summed
    over its ``axis`` when w is an initializer (cached by the weights' digest,
    the axis and x's range), else from both ranges; None while the depth is
    unknown, or when w holds a non-integer."""
    x_range, w_range = _range(model, x), _range(model, w)
    if x_range is None or w_range is None:
        return None
    key = tensor_ranges.digest(model, w)
    if key is not None:

        def compute() -> Range:
            weights = model.get_initializer(w)
            return _span(*_dot_bounds(weights, (axis % weights.ndim,), x_range))

        return _cached(("dot", key, axis, x_range), compute)
    if depth is None:
        return None
    lo, hi = _product(x_range, w_range)
    return depth * lo, depth * hi


def _matmul(model: ModelWrapper, node: NodeProto) -> Optional[Range]:
    a, b = node.input
    if tensor_ranges.initializer(model, b) is None and tensor_ranges.initializer(model, a) is not None:
        # a constant left operand: each row's sum over its last axis
        return _dot(model, b, a, -1, None)
    weights = tensor_ranges.initializer(model, b)
    axis = 0 if weights is not None and len(weights.dims) == 1 else -2
    return _dot(model, a, b, axis, _depth(model, a, -1) or _depth(model, b, axis))


def _gemm(model: ModelWrapper, node: NodeProto) -> Optional[Range]:
    """alpha * A' B' + beta * C, A' and B' A and B transposed as the node says."""
    a, b = node.input[:2]
    attribute = {item.name: item for item in node.attribute}
    alpha = attribute["alpha"].f if "alpha" in attribute else 1.0
    beta = attribute["beta"].f if "beta" in attribute else 1.0
    a_axis = 0 if "transA" in attribute and attribute["transA"].i else 1
    b_axis = 1 if "transB" in attribute and attribute["transB"].i else 0
    if not (alpha.is_integer() and beta.is_integer()):
        return None
    if tensor_ranges.initializer(model, b) is None and tensor_ranges.initializer(model, a) is not None:
        product = _dot(model, b, a, a_axis, None)
    else:
        product = _dot(model, a, b, b_axis, _depth(model, a, a_axis) or _depth(model, b, b_axis))
    if len(node.input) < 3 or not node.input[2]:
        return _scaled(product, int(alpha))
    return _plus(_scaled(product, int(alpha)), _scaled(_range(model, node.input[2]), int(beta)))


def _scaled(x: Optional[Range], factor: int) -> Optional[Range]:
    return None if x is None else _product(x, (factor, factor))


def _plus(x: Optional[Range], y: Optional[Range]) -> Optional[Range]:
    return None if x is None or y is None else _sum(x, y)


def _conv(model: ModelWrapper, node: NodeProto) -> Optional[Range]:
    """Per output channel, sums over its weights (of input channels and kernel
    positions), its bias added."""
    x, w = node.input[:2]
    bias = node.input[2] if len(node.input) > 2 and node.input[2] else None
    x_range, w_range = _range(model, x), _range(model, w)
    if x_range is None or w_range is None or (bias is not None and _range(model, bias) is None):
        return None
    # padding adds zeros to the sums
    x_range = (min(x_range[0], 0), max(x_range[1], 0))
    key = tensor_ranges.digest(model, w)
    if key is not None:
        bias_key = None if bias is None else tensor_ranges.digest(model, bias)

        def compute() -> Range:
            weights = model.get_initializer(w)
            lows, highs = _dot_bounds(weights, tuple(range(1, weights.ndim)), x_range)
            if bias_key is not None:
                offsets = _integers(model.get_initializer(bias), 1).astype(object)
                return _span(lows + offsets, highs + offsets)
            return _span(lows, highs)

        result = _cached(("conv", key, bias_key, x_range), compute)
        if bias is None or bias_key is not None:
            return result
    else:
        shape = model.get_tensor_shape(w)
        if not shape or len(shape) < 2 or min(shape[1:]) <= 0:
            return None
        lo, hi = _product(x_range, w_range)
        result = (math.prod(shape[1:]) * lo, math.prod(shape[1:]) * hi)
    return result if bias is None else _plus(result, _range(model, bias))


def _elementwise(combine: Callable[[Range, Range], Range]) -> Callable[[ModelWrapper, NodeProto], Optional[Range]]:
    """A rule combining the inputs' ranges, first to last."""

    def rule(model: ModelWrapper, node: NodeProto) -> Optional[Range]:
        ranges = [_range(model, name) for name in node.input]
        result = ranges[0]
        for other in ranges[1:]:
            result = None if result is None or other is None else combine(result, other)
        return result

    return rule


def _unary(apply: Callable[[Range], Range]) -> Callable[[ModelWrapper, NodeProto], Optional[Range]]:
    def rule(model: ModelWrapper, node: NodeProto) -> Optional[Range]:
        x = _range(model, node.input[0])
        return None if x is None else apply(x)

    return rule


def _clip(model: ModelWrapper, node: NodeProto) -> Optional[Range]:
    """min(max(x, low), high) grows with each argument, so its range's ends are
    its values at the arguments' ends. A constant bound is read by its value, a
    non-integer one included: the result is an integer where it never reaches it."""
    x = _range(model, node.input[0])
    if x is None:
        return None
    bounds = []
    for index, key, default in ((1, "min", -math.inf), (2, "max", math.inf)):
        name = node.input[index] if len(node.input) > index else ""
        attribute = get_by_name(node.attribute, key)
        values = model.get_initializer(name) if name else None
        if values is not None:
            bounds.append((float(np.min(values)), float(np.max(values))))
        elif name:
            bounds.append(_range(model, name))
        elif attribute is not None:
            bounds.append((attribute.f, attribute.f))
        else:
            bounds.append((default, default))
    low, high = bounds
    ends = [min(max(x[0], low[0]), high[0]), min(max(x[1], low[1]), high[1])]
    if not all(float(end).is_integer() for end in ends):
        return None
    return int(ends[0]), int(ends[1])


# The ops whose output range follows from their integer inputs' ranges (None: the
# result may be a non-integer).
_range_rules: dict[str, Callable[[ModelWrapper, NodeProto], Optional[Range]]] = {
    "MatMul": _matmul,
    "Gemm": _gemm,
    "Conv": _conv,
    "Add": _elementwise(_sum),
    "Sub": _elementwise(lambda x, y: (x[0] - y[1], x[1] - y[0])),
    "Mul": _elementwise(_product),
    "Max": _elementwise(lambda x, y: (max(x[0], y[0]), max(x[1], y[1]))),
    "Min": _elementwise(lambda x, y: (min(x[0], y[0]), min(x[1], y[1]))),
    "Concat": _elementwise(lambda x, y: (min(x[0], y[0]), max(x[1], y[1]))),
    "Relu": _unary(lambda x: (max(x[0], 0), max(x[1], 0))),
    "Neg": _unary(lambda x: (-x[1], -x[0])),
    "Clip": _clip,
}


def _integer_inputs(model: ModelWrapper, node: NodeProto) -> bool:
    """Whether the inputs a range rule reads are annotated integers (Clip's
    constant bounds are read by their values)."""
    names = [name for name in node.input if name]
    if node.op_type == "Clip":
        names = names[:1] + [name for name in names[1:] if tensor_ranges.initializer(model, name) is None]
    return all(tensor_ranges.datatype(model, name).is_integer() for name in names)


def is_scaled_int(x):
    # can treat both integer, fixed point and scaled int as scaled int
    return x.is_integer() or x.is_fixed_point() or isinstance(x, ScaledIntType)


def infer_mac_result_dtype(idtypes, odtype_orig, possible_negation):
    # will default to original output dtype unless specific cases detected
    ret = odtype_orig
    # result may be signed if:
    # - any of the operands are signed
    # - the operator itself may induce negation (like subtraction)
    maybe_signed = possible_negation or any([x.signed() for x in idtypes])
    if all([x.is_integer() for x in idtypes]):
        ret = DataType["INT32"] if maybe_signed else DataType["UINT32"]
    elif all([is_scaled_int(x) for x in idtypes]):
        ret = DataType["SCALEDINT<32>"]
    elif any(["FLOAT" in x.name for x in idtypes]):
        # default to float32 if any inps are float
        # TODO use output container dtype instead?
        ret = DataType["FLOAT32"]
    return ret


# The ops that pass their inputs' values through without rounding: the output
# holds only values of its inputs (and a Pad's constant).
_FIRST_INPUT = {
    "Reshape",
    "Transpose",
    "Flatten",
    "Slice",
    "Gather",
    "GatherElements",
    "GatherND",
    "Identity",
    "Expand",
    "MaxPool",
    "GlobalMaxPool",
    "Squeeze",
    "Unsqueeze",
    "Tile",
    "Resize",
    "Upsample",
}


def _pad_value(model: ModelWrapper, node: NodeProto) -> TensorRange:
    if len(node.input) > 2 and node.input[2]:
        return _tensor_range(model, node.input[2])
    value = get_by_name(node.attribute, "value")
    return TensorRange(ranges.endpoint(value.f if value is not None else 0.0), ranges.endpoint(value.f if value else 0.0))


def _bound(model: ModelWrapper, node: NodeProto, index: int, key: str, default: float) -> TensorRange:
    name = node.input[index] if len(node.input) > index else ""
    if name:
        return _tensor_range(model, name)
    attribute = get_by_name(node.attribute, key)
    value = ranges.endpoint(attribute.f if attribute is not None else default)
    return TensorRange(value, value)


def _passed(model: ModelWrapper, node: NodeProto) -> Optional[TensorRange]:
    """The range of a node that passes values through without rounding, for any
    datatype; None for any other node."""
    op = node.op_type
    if op in _FIRST_INPUT:
        return _tensor_range(model, node.input[0])
    if op == "Pad":
        x = _tensor_range(model, node.input[0])
        mode = get_by_name(node.attribute, "mode")
        if mode is not None and mode.s not in (b"", b"constant"):
            return x
        return ranges.hull(x, _pad_value(model, node))
    if op in ("Scatter", "ScatterElements", "ScatterND"):
        return ranges.hull(_tensor_range(model, node.input[0]), _tensor_range(model, node.input[2]))
    inputs = [_tensor_range(model, name) for name in node.input if name]
    nan = any(item.nan for item in inputs)
    if op == "Concat":
        return ranges.hull(*inputs)
    if op == "Max":
        return TensorRange(max(item.least for item in inputs), max(item.greatest for item in inputs), nan)
    if op == "Min":
        return TensorRange(min(item.least for item in inputs), min(item.greatest for item in inputs), nan)
    x = inputs[0]
    if op == "Relu":
        return TensorRange(max(x.least, 0), max(x.greatest, 0), x.nan)
    if op == "Neg":
        return TensorRange(-x.greatest, -x.least, x.nan)
    if op == "Clip":
        low, high = _bound(model, node, 1, "min", -math.inf), _bound(model, node, 2, "max", math.inf)
        least = min(max(x.least, low.least), high.least)
        greatest = min(max(x.greatest, low.greatest), high.greatest)
        return TensorRange(least, greatest, x.nan)
    return None


def _power_of_two(value: Fraction) -> bool:
    size = abs(value)
    return size != 0 and (size.numerator == 1 or size.denominator == 1) and (size.numerator & (size.numerator - 1)) == 0


def _scaled_by_power_of_two(model: ModelWrapper, node: NodeProto) -> Optional[TensorRange]:
    """A Mul by a constant power of two (every element the same +-2^k) into a
    scaled integer: its input's range scaled exactly; else None."""
    for scale, other in ((1, 0), (0, 1)):
        if tensor_ranges.initializer(model, node.input[scale]) is None:
            continue
        values = model.get_initializer(node.input[scale])
        if values.size == 0 or not np.all(np.isfinite(values)) or not np.all(values == values.flat[0]):
            continue
        factor = Fraction(float(values.flat[0]))
        if not _power_of_two(factor):
            continue
        x = _tensor_range(model, node.input[other])
        ends = sorted((x.least * factor, x.greatest * factor))
        return TensorRange(ends[0], ends[1], x.nan)
    return None


# The formats ONNX's float element types compute in.
_FLOAT_FORMATS = {
    TensorProto.FLOAT: ranges.BINARY32,
    TensorProto.FLOAT16: ranges.BINARY16,
    TensorProto.BFLOAT16: ranges.BFLOAT16,
    TensorProto.DOUBLE: ranges.BINARY64,
}


def _computed_in(model: ModelWrapper, node: NodeProto, datatype: BaseDataType) -> Optional[ranges.FloatFormat]:
    """The float format a node's output is computed and rounded in: its element
    type's (the output's, else the first input's that states one), else the
    container of ``datatype``. None for an element type that is not one of
    ONNX's binary floats (an integer one included)."""
    for name in (*node.output[:1], *node.input):
        found = tensor_ranges.element_type(model, name) if name else None
        if found is not None:
            return _FLOAT_FORMATS.get(found)
    return _FLOAT_FORMATS.get(helper.np_dtype_to_tensor_dtype(datatype.to_numpy_dt()))


def _sign(value: ranges.Endpoint) -> int:
    return (value > 0) - (value < 0)


def _exact(op: str, a: ranges.Endpoint, b: ranges.Endpoint) -> Optional[ranges.Endpoint]:
    """``a op b`` exactly (a rational, or an infinity), None where IEEE 754 makes
    NaN (inf - inf, 0 * inf, inf / inf). A divisor is never 0."""
    if op == "Sub":
        op, b = "Add", -b
    infinite = isinstance(a, float) or isinstance(b, float)
    if op == "Add":
        if not infinite:
            return a + b
        if isinstance(a, float) and isinstance(b, float) and a != b:
            return None
        return a if isinstance(a, float) else b
    if op == "Mul":
        if not infinite:
            return Fraction(a) * Fraction(b)
        sign = _sign(a) * _sign(b)
        return None if sign == 0 else sign * math.inf
    if isinstance(a, float) and isinstance(b, float):
        return None
    if isinstance(a, float):
        return _sign(a) * _sign(b) * math.inf
    return 0 if isinstance(b, float) else Fraction(a) / Fraction(b)


def _contains_zero(value: TensorRange) -> bool:
    return value.least <= 0 <= value.greatest


def _infinite(value: TensorRange) -> bool:
    return isinstance(value.least, float) or isinstance(value.greatest, float)


def _float_elementwise(op: str, x: TensorRange, y: TensorRange, form: ranges.FloatFormat) -> TensorRange:
    """The range of a float Add, Sub, Mul or Div of x and y, computed in ``form``:
    the exact interval of its results, each end rounded once into ``form`` (to
    nearest, as IEEE 754 rounds each result; rounding is monotone, so the
    rounded ends bound every rounded result, and an end past the largest finite
    value is an infinity). NaN where an input may be NaN, or where the intervals
    admit inf - inf, 0 * inf, 0 / 0 or inf / inf. A Div by an interval holding
    0 is unbounded both ways: a range does not tell -0 from +0."""
    nan = x.nan or y.nan
    if op == "Div" and _contains_zero(y):
        nan = nan or _contains_zero(x) or (_infinite(x) and _infinite(y))
        return ranges.unbounded(nan)
    if op == "Mul":
        nan = nan or (_contains_zero(x) and _infinite(y)) or (_contains_zero(y) and _infinite(x))
    # each op is monotone in each argument where it is a number, so its extremes
    # over the two intervals are at their ends; inf - inf and inf / inf are too
    corners = [_exact(op, a, b) for a in x.span for b in y.span]
    results = [value for value in corners if value is not None]
    nan = nan or len(results) < len(corners)
    if not results:
        return ranges.unbounded(True)
    return TensorRange(form.nearest(min(results)), form.nearest(max(results)), nan)


def _arithmetic(model: ModelWrapper, node: NodeProto, datatype: BaseDataType) -> TensorRange:
    """A mixed or float arithmetic op's range (MatMul, Gemm, Conv, Add, Sub, Mul,
    Div).

    For a float result, an Add, Sub, Mul or Div of two inputs computed in one of
    ONNX's binary float formats states the exact interval of its results, its
    ends rounded once into that format (_float_elementwise). A MatMul, Gemm or
    Conv, whose sums round in an order ONNX leaves open, states an unbounded
    range, NaN possible: its partial sums may reach both infinities. Any other
    float result is unbounded, NaN possible where an input may be NaN or
    infinite.

    A scaled-integer result has no NaN and no infinity, and is unbounded but for
    a Mul by a power of two, which scales its input's range."""
    if not ranges.is_float(datatype):
        if node.op_type == "Mul":
            scaled = _scaled_by_power_of_two(model, node)
            if scaled is not None:
                return scaled
        return ranges.unbounded(False)
    inputs = [_tensor_range(model, name) for name in node.input if name]
    if node.op_type in ("Add", "Sub", "Mul", "Div") and len(inputs) == 2:
        form = _computed_in(model, node, datatype)
        if form is not None:
            return _float_elementwise(node.op_type, inputs[0], inputs[1], form)
    nan = node.op_type in ("MatMul", "Gemm", "Conv") or any(item.nan or _infinite(item) for item in inputs)
    return ranges.unbounded(nan)


def _fact(model: ModelWrapper, name: str) -> tuple[str, Optional[str]]:
    """A tensor's datatype (as get_tensor_datatype reads it) and stored range text."""
    annotation = tensor_ranges.annotation(model, name)
    text = None
    if annotation is not None:
        for entry in annotation.quant_parameter_tensor_names:
            if entry.key == tensor_ranges.RANGE.entry:
                text = entry.value
    return tensor_ranges.datatype(model, name).get_canonical_name(), text


def infer_node_datatype(model: ModelWrapper, node: NodeProto, allow_scaledint_dtypes: bool) -> bool:
    """Infer the datatype and range (qonnx.core.ranges) of a node's outputs.
    Returns True if any changed.

    A standard op with a range rule (MatMul, Gemm, Conv, Add, Sub, Mul, Max, Min,
    Concat, Relu, Neg, Clip) whose inputs are annotated integers is typed by the
    exact range of its result: interval arithmetic over the inputs' ranges (each
    input's stated range, else its datatype's), an initializer's read by its
    values (a MatMul's, Gemm's or Conv's weights per output column, cached by
    their digest), the result the smallest INTn or UINTn holding it, and that
    range stated with it. An initializer or Clip bound holding a non-integer
    makes the result FLOAT32. Other inputs keep the rules below, with ranges:

    - an op that passes values through without rounding (Relu, Neg, Max, Min,
      Clip, Concat, Pad, Scatter and the identity and shape ops) states the range
      of what it passes, for any datatype;
    - a float Add, Sub, Mul or Div states the exact interval of its results, its
      ends rounded once into the float format it computes in, NaN possible only
      where an input may be NaN or the intervals admit inf - inf, 0 * inf, 0 / 0
      or inf / inf; a float MatMul, Gemm or Conv states an unbounded range, NaN
      possible (its rounding depends on its summation order); a scaled-integer
      result is unbounded, but for a Mul by a constant power of two, which
      scales its input's range;
    - any other op states none. A custom op states its own
      (CustomOp.infer_node_datatype).

    The datatype and the range are written together (set_tensor_datatype); an
    output's range is never kept from before. A float datatype is never
    narrowed."""
    dt_identity_optypes = [
        "Reshape",
        "Transpose",
        "Flatten",
        "Slice",
        "Gather",
        "GatherElements",
        "GatherND",
        "Identity",
        "Expand",
        "Flatten",
        "MaxPool",
        "GlobalMaxPool",
        "Scatter",
        "ScatterElements",
        "ScatterND",
        "Squeeze",
        "Unsqueeze",
        "Tile",
        "Pad",
        "Concat",
        "Clip",
    ]
    # the rules below are for ops without a range rule or inputs that are not all
    # integers: Concat and Clip then keep input 0's type, Max, Min, Relu and Neg
    # take the unknown op's rule
    mac_like_optypes = ["MatMul", "Gemm", "Conv", "Add", "Sub", "Mul"]
    before = [_fact(model, name) for name in node.output]
    op_type = node.op_type
    if is_custom_op(node.domain):
        # handle DataType inference for CustomOp
        try:
            # model-aware instantiation: ops declaring wants_model=True (those that
            # derive datatypes from the graph around them) receive the model;
            # classic ops are unaffected (safe superset of getCustomOp).
            inst = model.get_customop_wrapper(node)
            inst.infer_node_datatype(model)
        except KeyError:
            # exception if op_type is not supported
            raise Exception("Custom op_type %s is currently not supported." % op_type)
        facts = {}
        for out in node.output:
            if not allow_scaledint_dtypes and "SCALEDINT" in tensor_ranges.datatype(model, out).get_canonical_name():
                facts[out] = (DataType["FLOAT32"], tensor_ranges.stated(model, out))
    else:
        facts = _standard_facts(model, node, dt_identity_optypes, mac_like_optypes, allow_scaledint_dtypes)
    for out, (dtype, value) in facts.items():
        model.set_tensor_datatype(out, dtype, value)
    # compare old and new output datatypes and ranges to see if anything changed
    return [_fact(model, name) for name in node.output] != before


def _standard_facts(
    model: ModelWrapper,
    node: NodeProto,
    dt_identity_optypes: list[str],
    mac_like_optypes: list[str],
    allow_scaledint_dtypes: bool,
) -> dict[str, tuple[BaseDataType, Optional[TensorRange]]]:
    """Each output's datatype and range for a standard op (infer_node_datatype)."""
    idtypes = [tensor_ranges.datatype(model, name) for name in node.input]
    out = node.output[0]

    def final(dtype: BaseDataType) -> BaseDataType:
        # if scaled-int dtype inference is disabled, replace those with FLOAT32
        if not allow_scaledint_dtypes and "SCALEDINT" in dtype.get_canonical_name():
            return DataType["FLOAT32"]
        return dtype

    if node.op_type in _range_rules and _integer_inputs(model, node):
        span = _range_rules[node.op_type](model, node)
        if span is not None:
            return {out: (int_type_holding(*span), TensorRange(*span))}
        dtype = DataType["FLOAT32"]
        return {out: (dtype, _arithmetic(model, node, dtype))}
    if node.op_type == "Sign":
        # always produces bipolar outputs
        return {out: (DataType["BIPOLAR"], None)}
    if node.op_type in mac_like_optypes:
        possible_negation = node.op_type in ["Sub"]
        odtype_orig = tensor_ranges.datatype(model, out)
        dtype = final(infer_mac_result_dtype(idtypes, odtype_orig, possible_negation=possible_negation))
        return {out: (dtype, _arithmetic(model, node, dtype))}
    if node.op_type in ["Resize", "Upsample"]:
        mode = get_by_name(node.attribute, "mode").s
        if mode is None:
            mode = "nearest"
        else:
            mode = mode.decode("UTF-8")
        if mode == "nearest":
            # set output dtype = input dtype
            return {out: (final(idtypes[0]), _passed(model, node))}
        return {}
    if node.op_type in dt_identity_optypes:
        # set output dtype = input dtype
        return {out: (final(idtypes[0]), _passed(model, node))}
    if node.op_type == "QuantizeLinear":
        # retrieve from output tensor dtype
        ovi = model.get_tensor_valueinfo(out)
        (bitwidth, signed, _) = extract_elem_type(ovi.type.tensor_type.elem_type)
        prefix = "INT" if signed else "UINT"
        return {out: (DataType["%s%d" % (prefix, bitwidth)], None)}
    if node.op_type == "DequantizeLinear":
        # retrieve from input tensor dtype
        ivi = model.get_tensor_valueinfo(node.input[0])
        (bitwidth, signed, _) = extract_elem_type(ivi.type.tensor_type.elem_type)
        return {out: (final(DataType["SCALEDINT<%d>" % (bitwidth)]), None)}
    # unknown, assume node produces float32 outputs
    facts: dict[str, tuple[BaseDataType, Optional[TensorRange]]] = {}
    passed = _passed(model, node) if node.op_type in ("Relu", "Neg", "Max", "Min") else None
    for o in node.output:
        # check if output datatype is already set to a value != FLOAT32
        odtype = tensor_ranges.datatype(model, o)
        dtype = final(odtype if odtype is not None and odtype != DataType["FLOAT32"] else DataType["FLOAT32"])
        if node.op_type == "Div" and ranges.is_float(dtype):
            facts[o] = (dtype, _arithmetic(model, node, dtype))
        else:
            facts[o] = (dtype, passed)
    return facts


class InferDataTypes(Transformation):
    """Infer QONNX DataType info, with each tensor's range (qonnx.core.ranges), for
    all intermediate/output tensors based on inputs and node type. Each pass
    recomputes every node's; it stops changing the graph where a node's
    recomputed datatypes and ranges equal what is stored."""

    def __init__(self, allow_scaledint_dtypes=False):
        super().__init__()
        self.allow_scaledint_dtypes = allow_scaledint_dtypes

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        with tensor_ranges.one_pass(model):
            for node in graph.node:
                graph_modified |= infer_node_datatype(model, node, self.allow_scaledint_dtypes)
        return (model, graph_modified)
