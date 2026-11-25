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

import hashlib
import math
import numpy as np
from collections import Counter, OrderedDict
from fractions import Fraction
from onnx import TensorProto, helper, numpy_helper
from typing import TYPE_CHECKING, Any, Callable, Hashable, Optional, Union

from qonnx.core import ranges
from qonnx.core.datatype import (
    BFLOAT16,
    BINARY16,
    BINARY32,
    BINARY64,
    ArbPrecFloatType,
    BaseDataType,
    DataType,
    Float16Type,
    FloatFormat,
    FloatType,
    IntType,
    ScaledIntType,
)
from qonnx.core.modelwrapper import _custom_op_for
from qonnx.core.range_rules import column_range, dot_range, range_dtype
from qonnx.core.ranges import Range
from qonnx.core.tensor_ranges import RANGE, element_type, inferring, one_pass, stated, tensor_range
from qonnx.custom_op.registry import is_custom_op
from qonnx.transformation.base import Transformation
from qonnx.transformation.qcdq_to_qonnx import extract_elem_type
from qonnx.util.basic import get_by_name

if TYPE_CHECKING:
    from onnx import NodeProto

    from qonnx.core.modelwrapper import ModelWrapper


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


def int_type_holding(lo: int, hi: int) -> IntType:
    """The smallest INTn or UINTn holding every integer in [lo, hi]: unsigned when lo
    is not negative (UINT1 is BINARY). ValueError for a range past the datatype
    envelope (qonnx.core.range_rules.range_dtype)."""
    return range_dtype(lo, hi)


_Span = tuple[int, int]
_Rule = Callable[["ModelWrapper", "NodeProto"], Optional[_Span]]

# The bounds computed over initializers (an initializer's own range; a MatMul's, Gemm's
# or Conv's sums over its weights), kept in the process by what they read: the rule,
# the initializer's content digest and the input's range. Least recently used first.
_BOUNDS: OrderedDict[Hashable, Any] = OrderedDict()
_BOUNDS_KEPT = 4096
bounds_counts: Counter[str] = Counter()
"""How often each kind of bound was computed, and answered from the cache
("<kind> computed", "<kind> cached")."""


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


def _digest(model: ModelWrapper, name: str) -> Optional[bytes]:
    """A digest of ``name``'s values (element type, shape and data) if it is an
    initializer, else None."""
    proto = get_by_name(model.graph.initializer, name)
    if proto is None:
        return None
    data = proto.raw_data if proto.raw_data else numpy_helper.to_array(proto).tobytes()
    hashed = hashlib.sha256(b"%d:%s:" % (proto.data_type, ",".join(map(str, proto.dims)).encode()))
    hashed.update(data)
    return hashed.digest()


def _is_initializer(model: ModelWrapper, name: str) -> bool:
    return get_by_name(model.graph.initializer, name) is not None


def _integer_values(values: np.ndarray) -> Optional[_Span]:
    if values.size == 0 or not np.all(np.isfinite(values)) or not np.all(values == np.round(values)):
        return None
    return int(values.min()), int(values.max())


def _span(model: ModelWrapper, name: str) -> Optional[_Span]:
    """The integers ``name`` may hold: an initializer's by its values (None where one
    is not an integer), any other tensor's by its range (the one it states with its
    datatype, else its datatype's). An initializer's is cached by its digest."""
    key = _digest(model, name)
    if key is None:
        return tensor_range(model, name).integers()
    return _cached(("integers", key), lambda: _integer_values(model.get_initializer(name)))


def _tensor_range(model: ModelWrapper, name: str) -> Range:
    """The values ``name`` may hold, for any datatype: an initializer's by its values
    (cached by its digest), any other tensor's range."""
    key = _digest(model, name)
    if key is None:
        return tensor_range(model, name)
    return _cached(("values", key), lambda: ranges.values_range(model.get_initializer(name)))


def _sum(x: _Span, y: _Span) -> _Span:
    return x[0] + y[0], x[1] + y[1]


def _product(x: _Span, y: _Span) -> _Span:
    corners = [x[0] * y[0], x[0] * y[1], x[1] * y[0], x[1] * y[1]]
    return min(corners), max(corners)


def _scaled(x: Optional[_Span], factor: int) -> Optional[_Span]:
    return None if x is None else _product(x, (factor, factor))


def _plus(x: Optional[_Span], y: Optional[_Span]) -> Optional[_Span]:
    return None if x is None or y is None else _sum(x, y)


def _integers(values: np.ndarray, count: int) -> np.ndarray:
    """Integral ``values`` as integers whose sums of ``count`` terms cannot overflow:
    int64 where every such sum is below 2**62 in magnitude, Python integers
    otherwise. The bound is taken over Python integers (no fixed-width abs, which
    wraps at the signed minimum)."""
    if values.size == 0:
        return values.astype(np.int64)
    magnitude = max(abs(int(values.min())), abs(int(values.max())))
    if magnitude * max(count, 1) < 2**62:
        return values.astype(np.int64)
    return np.array([int(v) for v in values.flat], dtype=object).reshape(values.shape)


def _columns(weights: np.ndarray, axes: tuple[int, ...]) -> list[tuple[int, int]]:
    """For each position of ``weights`` outside ``axes``, the sums of its positive and
    of its negative weights over ``axes`` (qonnx.core.range_rules.column_range)."""
    weights = _integers(weights, math.prod(weights.shape[axis] for axis in axes))
    positive = np.where(weights > 0, weights, 0).sum(axis=axes)
    negative = np.where(weights < 0, weights, 0).sum(axis=axes)
    return [(int(p), int(n)) for p, n in zip(np.ravel(positive), np.ravel(negative))]


def _depth(model: ModelWrapper, name: str, axis: int) -> Optional[int]:
    """The length of ``name``'s ``axis``, None while it is unknown."""
    shape = model.get_tensor_shape(name)
    if not shape or not -len(shape) <= axis < len(shape) or not isinstance(shape[axis], int) or shape[axis] <= 0:
        return None
    return shape[axis]


def _dot(model: ModelWrapper, x: str, w: str, axis: int, depth: Optional[int]) -> Optional[_Span]:
    """A range holding every sum of products of x and w over w's ``axis``, partial
    sums included: from w's columns
    when w is an initializer (cached by its digest, the axis and x's range), else
    from both ranges over ``depth`` terms; None while the depth is unknown, or when x
    or w holds a non-integer."""
    x_span, w_span = _span(model, x), _span(model, w)
    if x_span is None or w_span is None:
        return None
    key = _digest(model, w)
    if key is not None:

        def compute() -> Optional[_Span]:
            weights = model.get_initializer(w)
            if weights.ndim == 0:
                return None
            columns = _columns(weights, (axis % weights.ndim,))
            return column_range(Range(*x_span), columns) if columns else None

        return _cached(("dot", key, axis, x_span), compute)
    if depth is None:
        return None
    return dot_range(depth, Range(*x_span), Range(*w_span))


def _matmul(model: ModelWrapper, node: NodeProto) -> Optional[_Span]:
    a, b = node.input
    if not _is_initializer(model, b) and _is_initializer(model, a):
        # a constant left operand: each row's sum over its last axis
        return _dot(model, b, a, -1, None)
    weights = model.get_initializer(b)
    axis = 0 if weights is not None and weights.ndim == 1 else -2
    return _dot(model, a, b, axis, _depth(model, a, -1) or _depth(model, b, axis))


def _gemm(model: ModelWrapper, node: NodeProto) -> Optional[_Span]:
    """alpha * A' B' + beta * C, A' and B' A and B transposed as the node says."""
    a, b = node.input[:2]
    attribute = {item.name: item for item in node.attribute}
    alpha = attribute["alpha"].f if "alpha" in attribute else 1.0
    beta = attribute["beta"].f if "beta" in attribute else 1.0
    a_axis = 0 if "transA" in attribute and attribute["transA"].i else 1
    b_axis = 1 if "transB" in attribute and attribute["transB"].i else 0
    if not (alpha.is_integer() and beta.is_integer()):
        return None
    if not _is_initializer(model, b) and _is_initializer(model, a):
        product = _dot(model, b, a, a_axis, None)
    else:
        product = _dot(model, a, b, b_axis, _depth(model, a, a_axis) or _depth(model, b, b_axis))
    if len(node.input) < 3 or not node.input[2]:
        return _scaled(product, int(alpha))
    return _plus(_scaled(product, int(alpha)), _scaled(_span(model, node.input[2]), int(beta)))


def _conv(model: ModelWrapper, node: NodeProto) -> Optional[_Span]:
    """Per output channel, sums over its weights (of input channels and kernel
    positions; padding adds zeros, which every sum's range holds), its bias added;
    cached by the weights' and the bias's digests and x's range."""
    x, w = node.input[:2]
    bias = node.input[2] if len(node.input) > 2 and node.input[2] else None
    x_span, w_span = _span(model, x), _span(model, w)
    bias_span = None if bias is None else _span(model, bias)
    if x_span is None or w_span is None or (bias is not None and bias_span is None):
        return None
    key = _digest(model, w)
    if key is None:
        shape = model.get_tensor_shape(w)
        if not shape or len(shape) < 2 or not all(isinstance(d, int) and d > 0 for d in shape[1:]):
            return None
        return _plus(dot_range(math.prod(shape[1:]), Range(*x_span), Range(*w_span)), bias_span or (0, 0))
    bias_key = None if bias is None else _digest(model, bias)

    def compute() -> Optional[_Span]:
        weights = model.get_initializer(w)
        if weights.ndim < 2 or weights.shape[0] == 0:
            return None
        columns = _columns(weights, tuple(range(1, weights.ndim)))
        offsets = None if bias_key is None else model.get_initializer(bias)
        if offsets is None or offsets.size != len(columns):
            return _plus(column_range(Range(*x_span), columns), bias_span or (0, 0))
        spans = [
            _sum(column_range(Range(*x_span), [column]), (int(offset), int(offset)))
            for column, offset in zip(columns, offsets.flat)
        ]
        return min(span[0] for span in spans), max(span[1] for span in spans)

    return _cached(("conv", key, bias_key, x_span, bias_span), compute)


def _reduce_sum(model: ModelWrapper, node: NodeProto) -> Optional[_Span]:
    """Each output sums the input's integer range over its fixed reduced extent.
    Unknown or overridable axes and unknown reduced dimensions give no bound, as
    does a sum no integer type within the datatype envelope holds. Symbolic
    dimensions outside the reduction do not affect its extent."""
    x = node.input[0]
    span = _span(model, x)
    if span is None:
        return None
    attribute = {item.name: item for item in node.attribute}
    if model.get_opset_imports().get(node.domain, 1) >= 13:
        axes_name = node.input[1] if len(node.input) > 1 else ""
        if axes_name:
            if any(info.name == axes_name for info in model.graph.input):
                return None
            axes_values = model.get_initializer(axes_name)
            if axes_values is None or axes_values.ndim != 1 or axes_values.dtype != np.int64:
                return None
            axes = tuple(int(value) for value in axes_values)
        else:
            axes = ()
        if not axes and "noop_with_empty_axes" in attribute and attribute["noop_with_empty_axes"].i:
            return span
    else:
        axes = tuple(attribute["axes"].ints) if "axes" in attribute else ()
    info = model.get_tensor_valueinfo(x)
    if info is not None:
        tensor = info.type.tensor_type
        if not tensor.HasField("shape"):
            return None
        shape = [dim.dim_value if dim.HasField("dim_value") else None for dim in tensor.shape.dim]
    else:
        values = model.get_initializer(x)
        if values is None:
            return None
        shape = list(values.shape)
    rank = len(shape)
    if any(not -rank <= axis < rank for axis in axes):
        return None
    reduced = {axis % rank for axis in axes} if axes else set(range(rank))
    extent = 1
    for axis in reduced:
        dimension = shape[axis]
        if dimension is None:
            return None
        extent *= dimension
    total = (extent * span[0], extent * span[1])
    try:
        int_type_holding(*total)
    except ValueError:
        return None
    return total


def _elementwise(combine: Callable[[_Span, _Span], _Span]) -> _Rule:
    """A rule combining the inputs' ranges, first to last."""

    def rule(model: ModelWrapper, node: NodeProto) -> Optional[_Span]:
        spans = [_span(model, name) for name in node.input]
        result = spans[0]
        for other in spans[1:]:
            result = None if result is None or other is None else combine(result, other)
        return result

    return rule


def _unary(apply: Callable[[_Span], _Span]) -> _Rule:
    def rule(model: ModelWrapper, node: NodeProto) -> Optional[_Span]:
        x = _span(model, node.input[0])
        return None if x is None else apply(x)

    return rule


def _clip(model: ModelWrapper, node: NodeProto) -> Optional[_Span]:
    """min(max(x, low), high) grows with each argument, so its range's ends are its
    values at the arguments' ends. A constant bound is read by its value, a
    non-integer one included: the result is an integer where it never reaches it.
    An integer bound is read as a Python int, never through float."""
    x = _span(model, node.input[0])
    if x is None:
        return None
    bounds: list[tuple[Union[int, float], Union[int, float]]] = []
    for index, key, default in ((1, "min", -math.inf), (2, "max", math.inf)):
        name = node.input[index] if len(node.input) > index else ""
        attribute = get_by_name(node.attribute, key)
        values = model.get_initializer(name) if name else None
        if values is not None and np.issubdtype(values.dtype, np.integer):
            bounds.append((int(np.min(values)), int(np.max(values))))
        elif values is not None:
            bounds.append((float(np.min(values)), float(np.max(values))))
        elif name:
            found = _span(model, name)
            if found is None:
                return None
            bounds.append(found)
        elif attribute is not None:
            bounds.append((attribute.f, attribute.f))
        else:
            bounds.append((default, default))
    low, high = bounds
    ends = [min(max(x[0], low[0]), high[0]), min(max(x[1], low[1]), high[1])]
    if not all(isinstance(end, int) or float(end).is_integer() for end in ends):
        return None
    return int(ends[0]), int(ends[1])


# The ops whose output range follows from their integer inputs' ranges (None: the
# result may be a non-integer, or, for ReduceSum, is not known).
_RANGE_RULES: dict[str, _Rule] = {
    "MatMul": _matmul,
    "Gemm": _gemm,
    "Conv": _conv,
    "ReduceSum": _reduce_sum,
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
    """Whether the inputs a range rule reads are integers by their datatypes
    (ReduceSum's axes are shape information; Clip's constant bounds are read by
    their values)."""
    names = [name for name in node.input if name]
    if node.op_type == "ReduceSum":
        names = names[:1]
    elif node.op_type == "Clip":
        names = names[:1] + [name for name in names[1:] if not _is_initializer(model, name)]
    return all(model.get_tensor_datatype(name).is_integer() for name in names)


# The ops that pass their inputs' values through without rounding: the output holds
# only values of its inputs (and a Pad's constant).
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


def _pad_value(model: ModelWrapper, node: NodeProto) -> Range:
    if len(node.input) > 2 and node.input[2]:
        return _tensor_range(model, node.input[2])
    value = get_by_name(node.attribute, "value")
    end = ranges.endpoint(value.f if value is not None else 0.0)
    return Range(end, end)


def _bound(model: ModelWrapper, node: NodeProto, index: int, key: str, default: float) -> Range:
    name = node.input[index] if len(node.input) > index else ""
    if name:
        return _tensor_range(model, name)
    attribute = get_by_name(node.attribute, key)
    value = ranges.endpoint(attribute.f if attribute is not None else default)
    return Range(value, value)


def _passed(model: ModelWrapper, node: NodeProto) -> Optional[Range]:
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
        return Range(max(item.least for item in inputs), max(item.greatest for item in inputs), nan)
    if op == "Min":
        return Range(min(item.least for item in inputs), min(item.greatest for item in inputs), nan)
    x = inputs[0]
    if op == "Relu":
        return Range(max(x.least, 0), max(x.greatest, 0), x.nan)
    if op == "Neg":
        return Range(-x.greatest, -x.least, x.nan)
    if op == "Clip":
        low, high = _bound(model, node, 1, "min", -math.inf), _bound(model, node, 2, "max", math.inf)
        least = min(max(x.least, low.least), high.least)
        greatest = min(max(x.greatest, low.greatest), high.greatest)
        return Range(least, greatest, x.nan)
    return None


def _power_of_two(value: Fraction) -> bool:
    size = abs(value)
    return size != 0 and (size.numerator == 1 or size.denominator == 1) and (size.numerator & (size.numerator - 1)) == 0


def _scaled_by_power_of_two(model: ModelWrapper, node: NodeProto) -> Optional[Range]:
    """A Mul by a constant power of two (every element the same +-2^k) into a scaled
    integer: its input's range scaled exactly; else None."""
    for scale, other in ((1, 0), (0, 1)):
        values = model.get_initializer(node.input[scale])
        if values is None:
            continue
        if values.size == 0 or not np.all(np.isfinite(values)) or not np.all(values == values.flat[0]):
            continue
        factor = Fraction(float(values.flat[0]))
        if not _power_of_two(factor):
            continue
        x = _tensor_range(model, node.input[other])
        ends = sorted((x.least * factor, x.greatest * factor))
        return Range(ranges.endpoint(ends[0]), ranges.endpoint(ends[1]), x.nan)
    return None


# The formats ONNX's float element types compute in.
_FLOAT_FORMATS = {
    TensorProto.FLOAT: BINARY32,
    TensorProto.FLOAT16: BINARY16,
    TensorProto.BFLOAT16: BFLOAT16,
    TensorProto.DOUBLE: BINARY64,
}


def _is_float(datatype: BaseDataType) -> bool:
    return isinstance(datatype, (FloatType, Float16Type, ArbPrecFloatType))


def _computed_in(model: ModelWrapper, node: NodeProto, datatype: BaseDataType) -> Optional[FloatFormat]:
    """The float format a node's output is computed and rounded in: its element type's
    (the output's, else the first input's that states one), else the container of
    ``datatype``. None for an element type that is not one of ONNX's binary floats (an
    integer one included)."""
    for name in (*node.output[:1], *node.input):
        found = element_type(model, name) if name else None
        if found is not None:
            return _FLOAT_FORMATS.get(found)
    return _FLOAT_FORMATS.get(helper.np_dtype_to_tensor_dtype(datatype.to_numpy_dt()))


def _sign(value: ranges.Endpoint) -> int:
    return (value > 0) - (value < 0)


def _exact(op: str, a: ranges.Endpoint, b: ranges.Endpoint) -> Optional[ranges.Endpoint]:
    """``a op b`` exactly (a rational, or an infinity), None where IEEE 754 makes NaN
    (inf - inf, 0 * inf, inf / inf). A divisor is never 0."""
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


def _contains_zero(value: Range) -> bool:
    return value.least <= 0 <= value.greatest


def _infinite(value: Range) -> bool:
    return isinstance(value.least, float) or isinstance(value.greatest, float)


def _float_elementwise(op: str, x: Range, y: Range, form: FloatFormat) -> Range:
    """The range of a float Add, Sub, Mul or Div of x and y, computed in ``form``: the
    exact interval of its results, each end rounded once into ``form`` (to nearest, as
    IEEE 754 rounds each result; rounding is monotone, so the rounded ends bound every
    rounded result, and an end past the largest finite value is an infinity). NaN where
    an input may be NaN, or where the intervals admit inf - inf, 0 * inf, 0 / 0 or
    inf / inf. A Div by an interval holding 0 is unbounded both ways: a range does not
    tell -0 from +0."""
    nan = x.nan or y.nan
    if op == "Div" and _contains_zero(y):
        nan = nan or _contains_zero(x) or (_infinite(x) and _infinite(y))
        return ranges.unbounded(nan)
    if op == "Mul":
        nan = nan or (_contains_zero(x) and _infinite(y)) or (_contains_zero(y) and _infinite(x))
    # each op is monotone in each argument where it is a number, so its extremes over
    # the two intervals are at their ends; inf - inf and inf / inf are too
    corners = [_exact(op, a, b) for a in x.span for b in y.span]
    results = [value for value in corners if value is not None]
    nan = nan or len(results) < len(corners)
    if not results:
        return ranges.unbounded(True)
    return Range(form.nearest(min(results)), form.nearest(max(results)), nan)


def _arithmetic(model: ModelWrapper, node: NodeProto, datatype: BaseDataType) -> Range:
    """A mixed or float arithmetic op's range (MatMul, Gemm, Conv, Add, Sub, Mul, Div).

    For a float result, an Add, Sub, Mul or Div of two inputs computed in one of ONNX's
    binary float formats states the exact interval of its results, its ends rounded
    once into that format (_float_elementwise). A MatMul, Gemm or Conv, whose sums
    round in an order ONNX leaves open, states an unbounded range, NaN possible: its
    partial sums may reach both infinities. Any other float result is unbounded, NaN
    possible where an input may be NaN or infinite.

    A scaled-integer result has no NaN and no infinity, and is unbounded but for a Mul
    by a power of two, which scales its input's range."""
    if not _is_float(datatype):
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
    annotation = get_by_name(model.graph.quantization_annotation, name, "tensor_name")
    text = None
    if annotation is not None:
        entry = get_by_name(annotation.quant_parameter_tensor_names, RANGE.entry, "key")
        text = None if entry is None else entry.value
    return model.get_tensor_datatype(name).name, text


def infer_node_datatype(
    model: ModelWrapper, node: NodeProto, allow_scaledint_dtypes: bool, exact_ranges: bool = False
) -> bool:
    """Infer output datatype(s) for a particular node. Returns True if any
    changes were made.

    With ``exact_ranges``, or inside an inference pass over the model
    (qonnx.core.tensor_ranges.one_pass, the signal for exact ranges), each output's
    range (qonnx.core.ranges) is inferred with its datatype and stated with it
    (ModelWrapper.set_tensor_datatype), and a True return includes a changed range;
    the call is one pass. Then:

    - a standard op with a range rule (MatMul, Gemm, Conv, ReduceSum, Add, Sub, Mul,
      Max, Min, Concat, Relu, Neg, Clip) whose inputs are integers by their datatypes
      is typed by a range of its result computed in exact integer arithmetic over the
      inputs' ranges (each input's stated range, else its datatype's; an
      initializer's by its values), the result the smallest INTn or UINTn holding it
      (int_type_holding). For the elementwise ops and ReduceSum the range is the
      tightest interval of the results. For MatMul, Gemm and Conv it holds every
      result and every partial sum of the dot products (qonnx.core.range_rules: a
      constant operand's weights per output column, cached by their digest:
      bounds_counts; else both ranges over the reduction depth; each operand's range
      widened to include 0), which an accumulator needs; it need not be the tightest
      range of the results alone. An initializer or a
      reachable Clip bound holding a non-integer, or a reduction depth not known,
      gives FLOAT32. ReduceSum multiplies its input's range by the reduced extent;
      unknown axes or reduced dimensions, or a sum past the datatype envelope, leave
      its output's datatype and range unstated; any other result past it is a
      ValueError;
    - other ops and inputs take the datatypes of the rules below (Concat and Clip
      keep input 0's; Max, Min, Relu and Neg take the unknown op's), with ranges: an
      op that passes values through without rounding (Relu, Neg, Max, Min, Clip,
      Concat, Pad, Scatter and the identity and shape ops) states the range of what
      it passes, for any datatype; each of a Split's outputs keeps its input's
      datatype and range; a float Add, Sub, Mul or Div states the exact interval of
      its results, its ends rounded once into the float format it computes in, NaN
      possible only where an input may be NaN or the intervals admit inf - inf,
      0 * inf, 0 / 0 or inf / inf; a float MatMul, Gemm or Conv states an unbounded
      range, NaN possible (its rounding depends on its summation order); a
      scaled-integer result is unbounded, but for a Mul by a constant power of two,
      which scales its input's range; any other op states none;
    - a custom op states its own (CustomOp.infer_node_datatype);
    - an output's range is never kept from before, a float datatype is never
      narrowed, and an absent optional output ("") is left alone.

    Without them, the rules are those before exact ranges: a MAC op's integer result
    is INT32 or UINT32, and no range is written."""
    if exact_ranges or inferring(model):
        with one_pass(model):
            return _exact_node_datatype(model, node, allow_scaledint_dtypes)
    return _node_datatype(model, node, allow_scaledint_dtypes)


_Facts = dict[str, tuple[Optional[BaseDataType], Optional[Range]]]


def _exact_node_datatype(model: ModelWrapper, node: NodeProto, allow_scaledint_dtypes: bool) -> bool:
    """infer_node_datatype with exact ranges."""
    before = [_fact(model, name) for name in node.output]
    if is_custom_op(node.domain):
        try:
            # the op at the version the model imports for its domain, given the model
            # where it wants it
            inst = _custom_op_for(model, node)
            inst.infer_node_datatype(model)
        except KeyError:
            # exception if op_type is not supported
            raise Exception("Custom op_type %s is currently not supported." % node.op_type)
        facts: _Facts = {}
        for out in node.output:
            if out and not allow_scaledint_dtypes and "SCALEDINT" in model.get_tensor_datatype(out).get_canonical_name():
                facts[out] = (DataType["FLOAT32"], stated(model, out))
    else:
        facts = _standard_facts(model, node, allow_scaledint_dtypes)
        for out in node.output:
            # an output no rule states: its datatype as it is, its range from before cleared
            if out and out not in facts and stated(model, out) is not None:
                facts[out] = (model.get_tensor_datatype(out), None)
    for out, (dtype, value) in facts.items():
        if out:  # an absent optional output is no tensor
            model.set_tensor_datatype(out, dtype, range=value)
    return [_fact(model, name) for name in node.output] != before


def _standard_facts(model: ModelWrapper, node: NodeProto, allow_scaledint_dtypes: bool) -> _Facts:
    """Each output's datatype and range for a standard op, with exact ranges: the
    datatypes of _node_datatype's rules, but where a range rule applies."""
    idtypes = [model.get_tensor_datatype(name) for name in node.input]
    out = node.output[0]

    def final(dtype: BaseDataType) -> BaseDataType:
        # if scaled-int dtype inference is disabled, replace those with FLOAT32
        if not allow_scaledint_dtypes and "SCALEDINT" in dtype.get_canonical_name():
            return DataType["FLOAT32"]
        return dtype

    if node.op_type in _RANGE_RULES and _integer_inputs(model, node):
        span = _RANGE_RULES[node.op_type](model, node)
        if span is not None:
            return {out: (int_type_holding(*span), Range(*span))}
        if node.op_type == "ReduceSum":
            return {out: (None, None)}
        dtype = DataType["FLOAT32"]
        return {out: (dtype, _arithmetic(model, node, dtype))}
    if node.op_type == "Sign":
        # always produces bipolar outputs
        return {out: (DataType["BIPOLAR"], None)}
    if node.op_type in _MAC_LIKE_OPTYPES:
        possible_negation = node.op_type in ["Sub"]
        odtype_orig = model.get_tensor_datatype(out)
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
    if node.op_type in _DT_IDENTITY_OPTYPES:
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
        return {out: (final(DataType["SCALEDINT<%d>" % bitwidth]), None)}
    if node.op_type == "Split":
        # every output holds values of input 0
        passed = _tensor_range(model, node.input[0])
        return {name: (final(idtypes[0]), passed) for name in node.output}
    # unknown, assume node produces float32 outputs
    facts: _Facts = {}
    passed = _passed(model, node) if node.op_type in ("Relu", "Neg", "Max", "Min") else None
    for name in node.output:
        # check if output datatype is already set to a value != FLOAT32
        odtype = model.get_tensor_datatype(name)
        dtype = final(odtype if odtype is not None and odtype != DataType["FLOAT32"] else DataType["FLOAT32"])
        if node.op_type == "Div" and _is_float(dtype):
            facts[name] = (dtype, _arithmetic(model, node, dtype))
        else:
            facts[name] = (dtype, passed)
    return facts


# The ops whose output takes input 0's datatype, and the multiply-accumulate ops
# (_node_datatype's rules).
_DT_IDENTITY_OPTYPES = [
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
_MAC_LIKE_OPTYPES = ["MatMul", "Gemm", "Conv", "Add", "Sub", "Mul"]


def _node_datatype(model: ModelWrapper, node: NodeProto, allow_scaledint_dtypes: bool) -> bool:
    """Infer output datatype(s) for a particular node. Returns True if any
    changes were made."""
    dt_identity_optypes = _DT_IDENTITY_OPTYPES
    mac_like_optypes = _MAC_LIKE_OPTYPES
    idtypes = list(map(lambda x: model.get_tensor_datatype(x), node.input))
    odtypes = list(map(lambda x: model.get_tensor_datatype(x), node.output))
    op_type = node.op_type
    if is_custom_op(node.domain):
        # handle DataType inference for CustomOp
        try:
            # the op at the version the model imports for its domain, given the model
            # where it wants it
            inst = _custom_op_for(model, node)
            inst.infer_node_datatype(model)
        except KeyError:
            # exception if op_type is not supported
            raise Exception("Custom op_type %s is currently not supported." % op_type)
    else:
        if node.op_type == "Sign":
            # always produces bipolar outputs
            model.set_tensor_datatype(node.output[0], DataType["BIPOLAR"])
        elif node.op_type in mac_like_optypes:
            possible_negation = node.op_type in ["Sub"]
            odtype_orig = model.get_tensor_datatype(node.output[0])
            odtype = infer_mac_result_dtype(idtypes, odtype_orig, possible_negation=possible_negation)
            model.set_tensor_datatype(node.output[0], odtype)
        elif node.op_type in ["Resize", "Upsample"]:
            mode = get_by_name(node.attribute, "mode").s
            if mode is None:
                mode = "nearest"
            else:
                mode = mode.decode("UTF-8")
            if mode == "nearest":
                # set output dtype = input dtype
                idtype = model.get_tensor_datatype(node.input[0])
                model.set_tensor_datatype(node.output[0], idtype)
        elif node.op_type in dt_identity_optypes:
            # set output dtype = input dtype
            idtype = model.get_tensor_datatype(node.input[0])
            model.set_tensor_datatype(node.output[0], idtype)
        elif node.op_type == "QuantizeLinear":
            # retrieve from output tensor dtype
            ovi = model.get_tensor_valueinfo(node.output[0])
            (bitwidth, signed, _) = extract_elem_type(ovi.type.tensor_type.elem_type)
            prefix = "INT" if signed else "UINT"
            ret = DataType["%s%d" % (prefix, bitwidth)]
            model.set_tensor_datatype(node.output[0], ret)
        elif node.op_type == "DequantizeLinear":
            # retrieve from input tensor dtype
            ivi = model.get_tensor_valueinfo(node.input[0])
            (bitwidth, signed, _) = extract_elem_type(ivi.type.tensor_type.elem_type)
            ret = DataType["SCALEDINT<%d>" % (bitwidth)]
            model.set_tensor_datatype(node.output[0], ret)
        else:
            # unknown, assume node produces float32 outputs
            for o in node.output:
                if not o:
                    continue
                # check if output datatype is already set to a value != FLOAT32
                odtype = model.get_tensor_datatype(o)
                if odtype is not None and odtype != DataType["FLOAT32"]:
                    # don't change data type
                    model.set_tensor_datatype(o, odtype)
                else:
                    model.set_tensor_datatype(o, DataType["FLOAT32"])
    # if scaled-int dtype inference is disabled, replace those with FLOAT32
    if not allow_scaledint_dtypes:
        for out in node.output:
            if out and "SCALEDINT" in model.get_tensor_datatype(out).get_canonical_name():
                model.set_tensor_datatype(out, DataType["FLOAT32"])
    # compare old and new output dtypes to see if anything changed
    new_odtypes = list(map(lambda x: model.get_tensor_datatype(x), node.output))
    graph_modified = new_odtypes != odtypes
    return graph_modified


# the name before infer_node_datatype was public
_infer_node_datatype = infer_node_datatype


class InferDataTypes(Transformation):
    """Infer QONNX DataType info for all intermediate/output tensors based on
    inputs and node type. With ``exact_ranges``, each tensor's range is inferred with
    its datatype and stated with it, and integer results are typed by ranges
    computed in exact integer arithmetic (infer_node_datatype: for MatMul, Gemm and
    Conv, a range holding every result and partial sum); each application is one pass
    (qonnx.core.tensor_ranges.one_pass), which recomputes every node's outputs and
    stops changing the graph where they equal what is stored."""

    def __init__(self, allow_scaledint_dtypes: bool = False, exact_ranges: bool = False) -> None:
        super().__init__()
        self.allow_scaledint_dtypes = allow_scaledint_dtypes
        self.exact_ranges = exact_ranges

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        graph = model.graph
        graph_modified = False
        if self.exact_ranges:
            with one_pass(model):
                for node in graph.node:
                    graph_modified |= infer_node_datatype(model, node, self.allow_scaledint_dtypes, True)
            return (model, graph_modified)
        for node in graph.node:
            graph_modified |= _infer_node_datatype(model, node, self.allow_scaledint_dtypes)
        return (model, graph_modified)
