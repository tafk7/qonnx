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
from onnx import NodeProto
from typing import TYPE_CHECKING, Callable, Optional

from qonnx.core.datatype import BaseDataType, DataType, ScaledIntType
from qonnx.custom_op.registry import is_custom_op
from qonnx.transformation.base import Transformation
from qonnx.transformation.qcdq_to_qonnx import extract_elem_type
from qonnx.util.basic import get_by_name

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

# The integers a tensor may hold, both ends included.
Range = tuple[int, int]


def int_type_holding(lo: int, hi: int) -> BaseDataType:
    """The smallest INTn or UINTn holding every integer in [lo, hi]: unsigned when
    lo is not negative."""
    if lo >= 0:
        return DataType["UINT%d" % max(1, hi.bit_length())]
    return DataType["INT%d" % (max((-lo - 1).bit_length(), max(hi, 0).bit_length()) + 1)]


def _range(model: ModelWrapper, name: str) -> Optional[Range]:
    """The integers ``name`` may hold: an initializer's by its values, any other
    tensor's by its annotation. None for an initializer holding a non-integer."""
    values = model.get_initializer(name)
    if values is None:
        dtype = model.get_tensor_datatype(name)
        return int(dtype.min()), int(dtype.max())
    if values.size == 0 or not np.all(np.isfinite(values)) or not np.all(values == np.round(values)):
        return None
    return int(values.min()), int(values.max())


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
    over its ``axis`` when w is an initializer, else from both ranges; None while
    the depth is unknown, or when w holds a non-integer."""
    x_range, w_range = _range(model, x), _range(model, w)
    if x_range is None or w_range is None:
        return None
    weights = model.get_initializer(w)
    if weights is not None:
        return _span(*_dot_bounds(weights, (axis % weights.ndim,), x_range))
    if depth is None:
        return None
    lo, hi = _product(x_range, w_range)
    return depth * lo, depth * hi


def _matmul(model: ModelWrapper, node: NodeProto) -> Optional[Range]:
    a, b = node.input
    if model.get_initializer(b) is None and model.get_initializer(a) is not None:
        # a constant left operand: each row's sum over its last axis
        return _dot(model, b, a, -1, None)
    weights = model.get_initializer(b)
    axis = 0 if weights is not None and weights.ndim == 1 else -2
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
    if model.get_initializer(b) is None and model.get_initializer(a) is not None:
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
    weights = model.get_initializer(w)
    if weights is not None:
        lows, highs = _dot_bounds(weights, tuple(range(1, weights.ndim)), x_range)
        offsets = None if bias is None else model.get_initializer(bias)
        if offsets is not None:
            offsets = _integers(offsets, 1).astype(object)
            return _span(lows + offsets, highs + offsets)
        result = _span(lows, highs)
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
        names = names[:1] + [name for name in names[1:] if model.get_initializer(name) is None]
    return all(model.get_tensor_datatype(name).is_integer() for name in names)


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


def infer_node_datatype(model: ModelWrapper, node: NodeProto, allow_scaledint_dtypes: bool) -> bool:
    """Infer output datatype(s) for a particular node. Returns True if any
    changes were made.

    A standard op with a range rule (MatMul, Gemm, Conv, Add, Sub, Mul, Max, Min,
    Concat, Relu, Neg, Clip) whose inputs are annotated integers is typed by the
    exact range of its result: interval arithmetic over the inputs' ranges, an
    initializer's read by its values (a MatMul's, Gemm's or Conv's weights per
    output column), the result the smallest INTn or UINTn holding it. Its output's
    annotation is replaced, whatever it was. An initializer or Clip bound holding a
    non-integer makes the result FLOAT32. Other inputs keep the rules below."""
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
    idtypes = list(map(lambda x: model.get_tensor_datatype(x), node.input))
    odtypes = list(map(lambda x: model.get_tensor_datatype(x), node.output))
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
    else:
        if node.op_type in _range_rules and _integer_inputs(model, node):
            span = _range_rules[node.op_type](model, node)
            odtype = DataType["FLOAT32"] if span is None else int_type_holding(*span)
            model.set_tensor_datatype(node.output[0], odtype)
        elif node.op_type == "Sign":
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


class InferDataTypes(Transformation):
    """Infer QONNX DataType info for all intermediate/output tensors based on
    inputs and node type."""

    def __init__(self, allow_scaledint_dtypes=False):
        super().__init__()
        self.allow_scaledint_dtypes = allow_scaledint_dtypes

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for node in graph.node:
            graph_modified |= infer_node_datatype(model, node, self.allow_scaledint_dtypes)
        return (model, graph_modified)
