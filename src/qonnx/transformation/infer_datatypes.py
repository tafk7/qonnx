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
from typing import TYPE_CHECKING, Callable, Optional, Union

import qonnx.custom_op.registry as registry
from qonnx.core.datatype import DataType, IntType, ScaledIntType
from qonnx.core.range_rules import column_range, dot_range, range_dtype
from qonnx.core.ranges import Range
from qonnx.core.tensor_ranges import RANGE, one_pass, tensor_range
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


def _is_initializer(model: ModelWrapper, name: str) -> bool:
    return get_by_name(model.graph.initializer, name) is not None


def _integer_values(values: np.ndarray) -> Optional[_Span]:
    if values.size == 0 or not np.all(np.isfinite(values)) or not np.all(values == np.round(values)):
        return None
    return int(values.min()), int(values.max())


def _span(model: ModelWrapper, name: str) -> Optional[_Span]:
    """The integers ``name`` may hold: an initializer's by its values (None where one
    is not an integer), any other tensor's by its range (the one it states with its
    datatype, else its datatype's)."""
    values = model.get_initializer(name)
    if values is not None:
        return _integer_values(values)
    return tensor_range(model, name).integers()


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
    when w is an initializer, else from both ranges over ``depth`` terms; None while
    the depth is unknown, or when x or w holds a non-integer."""
    x_span, w_span = _span(model, x), _span(model, w)
    if x_span is None or w_span is None:
        return None
    weights = model.get_initializer(w)
    if weights is not None:
        if weights.ndim == 0:
            return None
        columns = _columns(weights, (axis % weights.ndim,))
        return column_range(Range(*x_span), columns) if columns else None
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
    positions; padding adds zeros, which every sum's range holds), its bias added."""
    x, w = node.input[:2]
    bias = node.input[2] if len(node.input) > 2 and node.input[2] else None
    x_span, w_span = _span(model, x), _span(model, w)
    bias_span = None if bias is None else _span(model, bias)
    if x_span is None or w_span is None or (bias is not None and bias_span is None):
        return None
    weights = model.get_initializer(w)
    if weights is None:
        shape = model.get_tensor_shape(w)
        if not shape or len(shape) < 2 or not all(isinstance(d, int) and d > 0 for d in shape[1:]):
            return None
        return _plus(dot_range(math.prod(shape[1:]), Range(*x_span), Range(*w_span)), bias_span or (0, 0))
    if weights.ndim < 2 or weights.shape[0] == 0:
        return None
    columns = _columns(weights, tuple(range(1, weights.ndim)))
    offsets = None if bias is None else model.get_initializer(bias)
    if offsets is None or offsets.size != len(columns):
        return _plus(column_range(Range(*x_span), columns), bias_span or (0, 0))
    spans = [
        _sum(column_range(Range(*x_span), [column]), (int(offset), int(offset)))
        for column, offset in zip(columns, offsets.flat)
    ]
    return min(span[0] for span in spans), max(span[1] for span in spans)


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

    With ``exact_ranges``, a standard op with a range rule (MatMul, Gemm, Conv,
    ReduceSum, Add, Sub, Mul, Max, Min, Concat, Relu, Neg, Clip) whose inputs are
    integers by their datatypes is typed by a range of its result computed in exact
    integer arithmetic over the inputs' ranges (qonnx.core.ranges: each input's
    stated range, else its datatype's; an initializer's by its values), the result
    the smallest INTn or UINTn holding that range (int_type_holding), and the range
    stated with it (ModelWrapper.set_tensor_datatype). For the elementwise ops and
    ReduceSum it is the tightest interval of the results over the inputs' ranges.
    For MatMul, Gemm and Conv it is the range holding every result and every partial
    sum of its dot products (qonnx.core.range_rules: a constant operand's weights
    per output column, else both ranges over the reduction depth, each operand's
    range widened to include 0), which an accumulator needs; it need not be the
    tightest range of the results alone. An initializer or a reachable Clip bound
    holding a non-integer, or a reduction depth not known, gives FLOAT32. ReduceSum
    multiplies its input's range by the reduced extent; unknown axes or reduced
    dimensions, or a sum past the datatype envelope, leave its output's datatype
    unstated; any other result past it is a ValueError. Other ops keep the rules
    below (Concat and Clip keep input 0's datatype; Max, Min, Relu and Neg take the
    unknown op's), and the call is one pass (qonnx.core.tensor_ranges.one_pass).
    Without it, the rules are those before exact ranges: a MAC op's integer result
    is INT32 or UINT32."""
    if exact_ranges:
        with one_pass(model):
            return _exact_node_datatype(model, node, allow_scaledint_dtypes)
    return _node_datatype(model, node, allow_scaledint_dtypes)


def _exact_node_datatype(model: ModelWrapper, node: NodeProto, allow_scaledint_dtypes: bool) -> bool:
    """infer_node_datatype with exact ranges."""
    before = [_fact(model, name) for name in node.output]
    if not is_custom_op(node.domain) and node.op_type in _RANGE_RULES and _integer_inputs(model, node):
        span = _RANGE_RULES[node.op_type](model, node)
        out = node.output[0]
        if span is not None:
            model.set_tensor_datatype(out, int_type_holding(*span), range=Range(*span))
        elif node.op_type == "ReduceSum":
            model.set_tensor_datatype(out, None)
        else:
            model.set_tensor_datatype(out, DataType["FLOAT32"])
    else:
        _node_datatype(model, node, allow_scaledint_dtypes)
    return [_fact(model, name) for name in node.output] != before


def _node_datatype(model: ModelWrapper, node: NodeProto, allow_scaledint_dtypes: bool) -> bool:
    """Infer output datatype(s) for a particular node. Returns True if any
    changes were made."""
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
    mac_like_optypes = ["MatMul", "Gemm", "Conv", "Add", "Sub", "Mul"]
    idtypes = list(map(lambda x: model.get_tensor_datatype(x), node.input))
    odtypes = list(map(lambda x: model.get_tensor_datatype(x), node.output))
    op_type = node.op_type
    if is_custom_op(node.domain):
        # handle DataType inference for CustomOp
        try:
            # lookup op_type in registry of CustomOps
            inst = registry.getCustomOp(node)
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
            if "SCALEDINT" in model.get_tensor_datatype(out).get_canonical_name():
                model.set_tensor_datatype(out, DataType["FLOAT32"])
    # compare old and new output dtypes to see if anything changed
    new_odtypes = list(map(lambda x: model.get_tensor_datatype(x), node.output))
    graph_modified = new_odtypes != odtypes
    return graph_modified


# the name before infer_node_datatype was public
_infer_node_datatype = infer_node_datatype


class InferDataTypes(Transformation):
    """Infer QONNX DataType info for all intermediate/output tensors based on
    inputs and node type. With ``exact_ranges``, integer results are typed by ranges
    computed in exact integer arithmetic, stated with them (infer_node_datatype:
    for MatMul, Gemm and Conv, a range holding every result and partial sum); each
    application is one pass (qonnx.core.tensor_ranges.one_pass)."""

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
