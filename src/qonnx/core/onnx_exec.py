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

import copy
import numpy as np
import onnx.helper as helper
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from onnx import AttributeProto, GraphProto, NodeProto, TensorProto, ValueInfoProto
from onnxruntime.capi.onnxruntime_pybind11_state import NotImplemented as RuntimeNotImplemented
from typing import TYPE_CHECKING, Any, Iterator, Literal, Mapping, overload

import qonnx.analysis.topology as ta
import qonnx.core.execute_custom_node as ex_cu_node
from qonnx.core import _sessions
from qonnx.custom_op.registry import is_custom_op
from qonnx.util.basic import get_preferred_qonnx_opset, get_sanitize_quant_tensors, qonnx_make_model, sanitize_quant_values

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

# Whether integers are computed exactly: set by the exact_integers context and the
# exact_integers= arguments, read by every node executed inside them.
_EXACT_INTEGERS: ContextVar[bool] = ContextVar("qonnx.core.onnx_exec.exact_integers", default=False)


def exact_integers_enabled() -> bool:
    """Whether integers are computed exactly here: the setting of the innermost
    exact_integers context (or exact_integers= argument) around the caller, else off."""
    return _EXACT_INTEGERS.get()


@contextmanager
def _switched(enabled: bool) -> Iterator[None]:
    if not isinstance(enabled, bool):
        raise TypeError(f"exact integers are switched by a bool, not {type(enabled).__name__}")
    token = _EXACT_INTEGERS.set(enabled)
    try:
        yield
    finally:
        _EXACT_INTEGERS.reset(token)


@contextmanager
def exact_integers(enabled: bool = True) -> Iterator[None]:
    """Computes integers exactly (enabled=True), or as the graph's float containers
    hold them (enabled=False), in every execution the block runs, nested ones
    included. Leaving the block restores the setting around it; off by default.

    With exact integers on, a standard node whose float tensors are all
    integer-annotated, given only integers, computes by its operator and the values
    given (execute_node). Integer arithmetic with a bound on its products and partial
    sums computes a result shown exact or refuses: in its own container or float64,
    whichever holds the bound, raising ArithmeticError past 2**53. An operator that
    only moves or selects values computes a result exact as its inputs, in float64
    where its container does not hold a value given. Any other operator (a Gemm whose
    alpha or beta is not an integer among them) has no proof: it computes as without
    exact integers where the values given and its result are within what its
    container holds exactly, and raises NotImplementedError past that. Within that,
    nothing shows its result exact: an Einsum of [4097, 4096] by [4097, -4098] gives
    0, not 1, its products passing 2**24 and cancelling. A node given a value that is
    not an integer computes as without exact integers. Operators whose attributes fix
    an element type (Cast, Constant and the like) are never widened, and a Cast into a
    float type raises ArithmeticError where that type loses a value. MultiThreshold
    computes its count and affine result exactly, refusing an out_dtype or a container
    that cannot hold every result; other custom ops compute as they do without exact
    integers (QuantAvgPool2d in float32), a graph one runs (a generic partition's
    body) under the setting.

    The setting belongs to the current context, as any context variable's does: each
    thread and each asyncio task has its own, a generator does not. A generator that
    yields inside the block leaves the setting in force for its caller until it is
    resumed past the block or closed; yielding inside the block is unsupported."""
    with _switched(enabled):
        yield


@contextmanager
def reuse_sessions(threads: int | None = None) -> Iterator[None]:
    """Reuses ONNX Runtime sessions in every execution the block runs, nested ones
    included. By default each standard node runs a fresh session, with ONNX Runtime's
    default thread pools; inside the block, the session of a stateless node model is
    retained and reused while the same model runs again, and where threads is given,
    every session the execution constructs (custom ops' own included) caps both of
    ONNX Runtime's thread pools at threads.

    One session is retained in all, released before another is constructed and on
    leaving an outermost block; concurrent blocks (other threads, other tasks) share
    it. So reuse pays only for repeated runs of one node model: a graph of several
    distinct node models constructs a session at each node, as it does without the
    block. The thread cap applies to every session either way. Random and seeded ops
    (Dropout among them), and any node whose schema cannot show it keeps no state, run
    a fresh session every time. threads must be an int in [1, 2147483647] (TypeError,
    ValueError).

    The setting belongs to the current context, as any context variable's does: each
    thread and each asyncio task has its own, a generator does not. A generator that
    yields inside the block leaves the setting in force, and the session retained, for
    its caller until it is resumed past the block or closed; yielding inside the block
    is unsupported."""
    with _sessions.reusing(threads):
        yield


def _np_type(elem_type: int) -> np.dtype | None:
    """The numpy type of an ONNX element type, or None where numpy has none."""
    try:
        return np.dtype(helper.tensor_dtype_to_np_dtype(elem_type))
    except (KeyError, TypeError, ValueError):
        return None


def _exact_limit(elem_type: int) -> int | None:
    """The largest magnitude up to which a float element type holds every integer
    exactly, 2**(mantissa bits + 1); None for an element type that is not a float."""
    np_type = _np_type(elem_type)
    if np_type is None or not np.issubdtype(np_type, np.floating):
        return None
    return 2 ** (int(np.finfo(np_type).nmant) + 1)


# float64 holds every integer up to 2**53 exactly.
_DOUBLE = 2**53

# Operators whose output elements are input elements or constants, and the rounding
# operators, exact in IEEE arithmetic: their result is exact whenever their inputs are
# held exactly. (ScatterElements and ScatterND only without a reduction.)
_MOVING = frozenset(
    {
        "ArgMax",
        "ArgMin",
        "CenterCropPad",
        "Ceil",
        "Clip",
        "Compress",
        "Concat",
        "DepthToSpace",
        "Equal",
        "Expand",
        "Flatten",
        "Floor",
        "Gather",
        "GatherElements",
        "GatherND",
        "GlobalMaxPool",
        "Greater",
        "GreaterOrEqual",
        "Hardmax",
        "Identity",
        "IsInf",
        "IsNaN",
        "Less",
        "LessOrEqual",
        "Max",
        "MaxPool",
        "MaxUnpool",
        "Min",
        "NonZero",
        "Pad",
        "ReduceMax",
        "ReduceMin",
        "Relu",
        "Reshape",
        "ReverseSequence",
        "Round",
        "Scatter",
        "ScatterElements",
        "ScatterND",
        "Shape",
        "Sign",
        "Size",
        "Slice",
        "SpaceToDepth",
        "Split",
        "Squeeze",
        "ThresholdedRelu",
        "Tile",
        "TopK",
        "Transpose",
        "Trilu",
        "Unique",
        "Unsqueeze",
        "Where",
    }
)

# Integer arithmetic whose exactness a bound on every product and partial sum proves.
_BOUNDED = frozenset({"Abs", "Add", "Conv", "Gemm", "MatMul", "Mul", "Neg", "ReduceSum", "Sub", "Sum"})

# Operators whose attributes fix an element type: they run in their own types.
_TYPED = frozenset(
    {
        "Bernoulli",
        "Cast",
        "CastLike",
        "Constant",
        "ConstantOfShape",
        "EyeLike",
        "Multinomial",
        "RandomNormal",
        "RandomNormalLike",
        "RandomUniform",
        "RandomUniformLike",
    }
)
_TYPE_ATTRIBUTES = frozenset({"dtype", "output_dtype", "to"})
_VALUE_ATTRIBUTES = frozenset(
    {
        AttributeProto.TENSOR,
        AttributeProto.TENSORS,
        AttributeProto.SPARSE_TENSOR,
        AttributeProto.SPARSE_TENSORS,
        AttributeProto.TYPE_PROTO,
        AttributeProto.TYPE_PROTOS,
    }
)


def _typed(node: NodeProto) -> bool:
    """Whether the node's attributes fix an element type (Cast's to, a dtype, a tensor
    value): such a node is never widened."""
    return node.op_type in _TYPED or any(
        attr.name in _TYPE_ATTRIBUTES or attr.type in _VALUE_ATTRIBUTES for attr in node.attribute
    )


def _bounded(node: NodeProto) -> bool:
    """Whether the node is integer arithmetic with a bound: a Gemm scaled by an alpha or
    beta that is not an integer is not, and has no proof."""
    if node.op_type not in _BOUNDED:
        return False
    scales = [helper.get_attribute_value(attr) for attr in node.attribute if attr.name in ("alpha", "beta")]
    return all(float(scale).is_integer() for scale in scales)


def _moving(node: NodeProto) -> bool:
    """Whether the node only moves, selects or rounds values."""
    if node.op_type not in _MOVING:
        return False
    reduction = [helper.get_attribute_value(attr) for attr in node.attribute if attr.name == "reduction"]
    return all(value in (b"", b"none") for value in reduction)


def _all_integers(value: Any) -> bool:
    """Whether every value of an array is a finite integer."""
    array = np.asarray(value)
    if array.dtype.kind in "biu":
        return True
    if array.dtype.kind != "f":
        return False
    return bool(np.isfinite(array).all() and np.array_equal(array, np.trunc(array)))


def _top(array: np.ndarray | None) -> int:
    """The largest magnitude of an array of integers, 0 for an absent or empty one."""
    if array is None or array.size == 0:
        return 0
    return max(abs(int(array.min())), abs(int(array.max())))


def _magnitudes(array: np.ndarray, wide: bool) -> np.ndarray:
    """The magnitudes of an array of integers, exactly: int64, or Python integers where
    wide (a bound past int64 may be reached)."""
    if wide:
        return np.asarray(np.frompyfunc(lambda value: abs(int(value)), 1, 1)(array), dtype=object)
    return np.abs(array.astype(np.int64))


def _reduced_axes(node: NodeProto, operands: list[np.ndarray | None], opset_version: int) -> list[int] | None:
    """ReduceSum's axes, normalized; None where it reduces nothing (an empty axes list
    with noop_with_empty_axes)."""
    attributes = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
    if opset_version >= 13:
        given = operands[1] if len(operands) > 1 else None
        axes = [] if given is None else [int(axis) for axis in np.asarray(given).reshape(-1)]
    else:
        axes = [int(axis) for axis in attributes.get("axes", [])]
    rank = operands[0].ndim
    if not axes:
        return None if attributes.get("noop_with_empty_axes", 0) else list(range(rank))
    return sorted({axis % rank for axis in axes})


def _bound(node: NodeProto, operands: list[np.ndarray | None], opset_version: int, enough: int) -> int:
    """A bound, in Python integers, on the magnitude of every operand, every product and
    every partial sum the node computes, in any order of accumulation, from the integers
    it is given:
    a coarse bound from the operands' largest magnitudes, and where that passes enough,
    the bound taken element by element."""
    op = node.op_type
    present = [operand for operand in operands if operand is not None]
    tops = [_top(operand) for operand in present]
    if op in ("Abs", "Neg"):
        return max(tops)
    if op in ("Add", "Sub", "Sum"):
        coarse = sum(tops)

        def tight(wide: bool) -> Any:
            total: Any = 0
            for operand in present:
                total = total + _magnitudes(operand, wide)
            return np.max(total)

    elif op == "Mul":
        coarse = tops[0] * tops[1]

        def tight(wide: bool) -> Any:
            return np.max(_magnitudes(present[0], wide) * _magnitudes(present[1], wide))

    elif op == "MatMul":
        coarse = present[0].shape[-1] * tops[0] * tops[1]

        def tight(wide: bool) -> Any:
            return np.max(np.matmul(_magnitudes(present[0], wide), _magnitudes(present[1], wide)))

    elif op == "Gemm":
        attributes = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        alpha, beta = float(attributes.get("alpha", 1.0)), float(attributes.get("beta", 1.0))
        a = operands[0].T if attributes.get("transA", 0) else operands[0]
        b = operands[1].T if attributes.get("transB", 0) else operands[1]
        scale, offset = max(1, abs(int(alpha))), max(1, abs(int(beta))) * _top(operands[2] if len(operands) > 2 else None)
        coarse = scale * a.shape[1] * _top(a) * _top(b) + offset

        def tight(wide: bool) -> Any:
            return scale * int(np.max(np.matmul(_magnitudes(a, wide), _magnitudes(b, wide)))) + offset

    elif op == "Conv":
        x, w = operands[0], operands[1]
        bias = operands[2] if len(operands) > 2 else None
        coarse = int(np.prod(w.shape[1:])) * _top(x) * _top(w) + _top(bias)

        def tight(wide: bool) -> Any:
            per_channel = _magnitudes(w, wide).reshape(w.shape[0], -1).sum(axis=1) * _top(x)
            if bias is not None:
                per_channel = per_channel + _magnitudes(bias, wide)
            return np.max(per_channel)

    elif op == "ReduceSum":
        x = operands[0]
        axes = _reduced_axes(node, operands, opset_version)
        if axes is None or x.size == 0:
            return max(tops)
        coarse = int(np.prod([x.shape[axis] for axis in axes])) * tops[0]

        def tight(wide: bool) -> Any:
            return np.max(_magnitudes(x, wide).sum(axis=tuple(axes)))

    else:
        raise AssertionError(op)
    largest = max(tops, default=0)
    if max(coarse, largest) <= enough:
        return max(coarse, largest)
    return max(int(tight(coarse >= 2**63)), largest)


def _kept(given: np.ndarray, read: np.ndarray) -> bool:
    """Whether read, given in a float type, holds every value of given exactly."""
    if given.dtype.kind == "f":
        return bool(np.array_equal(read.astype(given.dtype), given, equal_nan=True))
    limit = 2 ** (int(np.finfo(read.dtype).nmant) + 1)
    return _top(given) <= limit or bool(np.all(read.astype(object) == given.astype(object)))


def _past(value: Any, limit: int) -> bool:
    """Whether an array of numbers holds a value whose magnitude passes limit (an
    infinity among them; not a NaN)."""
    array = np.asarray(value)
    if array.size == 0 or array.dtype.kind not in "biuf":
        return False
    if array.dtype.kind == "f":
        return bool(np.any(np.abs(array) > limit))
    return _top(array) > limit


def _widened(
    node: NodeProto,
    infos: list[ValueInfoProto],
    model: ModelWrapper,
    context: Mapping[str, Any],
    opset_version: int,
) -> tuple[set[str], bool, dict[str, int]]:
    """For a standard node: the tensors exact integers make it compute in float64
    (empty where it computes in the graph's own element types), whether exact integers
    concern it, and the float outputs whose values its result must keep within what
    their containers hold exactly, with that limit. Raises where it refuses the node.

    Exact integers concern a node whose float tensors are all integer-annotated and
    every value given to it is an integer. Any other node, and a node whose attributes
    fix an element type (Cast, Constant and the like, _typed), computes as it does
    without them. A node they concern, by the values given to it:

    - Arithmetic with a bound (_bounded): a bound on every operand, product and partial
      sum is taken from the integers given. Where its own container holds the bound
      exactly, the node computes in it; where float64 does (2**53), in float64; past
      2**53, it raises ArithmeticError, a DOUBLE container included.
    - An operator that moves, selects or rounds values (_moving): exact as its inputs
      are; it computes in its own container where that holds every value given, in
      float64 where a given value needs it.
    - Any other operator (a node with a subgraph, and a Gemm scaled by a fraction,
      among them): no proof, so it computes as it does without exact integers where
      every value given, and every value of its result, is within what its container
      holds exactly (float32: 2**24 in magnitude; float64: 2**53); nothing shows that
      result exact. A given value past that raises NotImplementedError here; a result
      value past it, once computed (the returned limits)."""
    held = {info.name: info.type.tensor_type.elem_type for info in infos}
    names = dict.fromkeys(name for name in (*node.input, *node.output) if name in held)
    limits = {name: _exact_limit(held[name]) for name in names}
    floats = [name for name in names if limits[name] is not None]
    given = {name: context[name] for name in node.input if name and context.get(name) is not None}
    concerned = (
        bool(floats)
        and all(model.get_tensor_datatype(name).is_integer() for name in floats)
        and all(_all_integers(value) for value in given.values())
    )
    if not concerned or _typed(node):
        return set(), concerned, {}
    if _bounded(node):
        operands = [np.asarray(given[name]) if name in given else None for name in node.input]
        container = min(limits[name] for name in floats)
        bound = _bound(node, operands, opset_version, container)
        if bound <= container:
            return set(), True, {}
        if bound > _DOUBLE:
            raise ArithmeticError(
                f"{node.name} ({node.op_type}): a bound on its operands, products and partial sums, {bound}, "
                "passes 2**53, beyond which float64 does not hold every integer exactly"
            )
        return set(floats), True, {}
    inputs = [name for name in floats if name in given]
    if _moving(node):
        own = [_kept(np.asarray(given[name]), np.asarray(given[name]).astype(_np_type(held[name]))) for name in inputs]
        return (set() if all(own) else set(floats)), True, {}
    past = [name for name in inputs if _past(given[name], limits[name])]
    if past:
        raise NotImplementedError(_unproved(node, f"{past[0]} is given a value"))
    return set(), True, {name: limits[name] for name in floats if name in node.output}


def _unproved(node: NodeProto, holder: str) -> str:
    """The message refusing a node with no proof that it computes integers exactly."""
    what = (
        "a node with a subgraph"
        if any(attr.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS) for attr in node.attribute)
        else node.op_type
    )
    return (
        f"{node.name} ({node.op_type}): {holder} past what its container holds exactly, "
        f"and {what} has no proof that it computes integers exactly"
    )


def _read(node: NodeProto, name: str, value: Any, np_type: np.dtype, checked: bool) -> np.ndarray:
    """value given in np_type, the element type the node reads it in; where checked (an
    input of a node exact integers concern, read in a float type), refused
    (ArithmeticError) if that type does not hold every value."""
    given = np.asarray(value)
    read = given.astype(np_type, copy=False)
    if checked and read.dtype != given.dtype and given.dtype.kind in "biuf":
        if not _kept(given, read):
            raise ArithmeticError(
                f"{node.name} ({node.op_type}): {name} holds a value that {np_type}, the type the node reads it in, "
                "does not hold exactly"
            )
    return read


def _cast_exact(node: NodeProto, given: Any, produced: np.ndarray, concerned: bool) -> None:
    """Refuses (ArithmeticError) a Cast or CastLike into a float container that has not
    kept every value: from an integer container always, from a float container where the
    node's float tensors are all integer-annotated."""
    given = np.asarray(given)
    if produced.dtype.kind != "f" or not (given.dtype.kind in "biu" or (concerned and given.dtype.kind == "f")):
        return
    if not _kept(given, produced):
        raise ArithmeticError(f"{node.name} ({node.op_type}): {produced.dtype} does not hold every value of {node.input[0]}")


def _as_double(info: ValueInfoProto) -> ValueInfoProto:
    """A copy of a value info, its element type DOUBLE."""
    double = ValueInfoProto()
    double.CopyFrom(info)
    double.type.tensor_type.elem_type = TensorProto.DOUBLE
    return double


def execute_node(
    node: NodeProto,
    context: dict[str, Any],
    graph: GraphProto,
    opset_version: int,
    return_full_exec_context: bool = False,
    *,
    model: ModelWrapper | None = None,
    exact_integers: bool | None = None,
) -> None:
    """Executes a single node by using onnxruntime or with a custom function.

    Input/output provided via context.

    exact_integers switches exact integers for this node (see the exact_integers
    context); None, the default, keeps the setting around the call, off unless set.
    With them on, a standard node whose float tensors are all integer-annotated, and
    every value given to which is an integer, computes by its operator and the values
    given:

    - Add, Sub, Mul, Neg, Abs, Sum, ReduceSum, MatMul, Gemm and Conv compute a result
      shown exact or refuse. They take a bound on every operand, product and partial
      sum from the integers given. Where the narrowest of its float containers holds
      the bound exactly (float32: 2**24) the node computes in its element types; where
      float64 does (2**53), in float64; past 2**53 it raises ArithmeticError, a DOUBLE
      container included.
    - An operator that only moves, selects, compares or rounds values (Reshape,
      Transpose, Concat, Gather, Pad, Max, Min, Relu, Clip, MaxPool, Identity and the
      like) computes a result exact as its inputs: in its element types where they
      hold every value given, and in float64 where a value given needs it.
    - Any other operator (a node with a subgraph, and a Gemm whose alpha or beta is
      not an integer, among them) has no proof. It computes as without exact integers
      where every value given and every value of its result is within what its
      container holds exactly (float32: 2**24 in magnitude; float64: 2**53), and
      raises NotImplementedError past that. Within that range nothing shows its result
      exact: an Einsum of [4097, 4096] by [4097, -4098] gives 0, not 1, as without
      exact integers, its products passing 2**24 and cancelling.

    A node given a value that is not an integer (or not finite) computes as without
    exact integers.

    Computing in float64, a node's float tensors are stated DOUBLE, its inputs given
    as float64, and its outputs stay float64; ONNX Runtime having no float64 kernel
    for it raises NotImplementedError. An operator whose attributes fix an element
    type (Cast, CastLike, Constant, ConstantOfShape, EyeLike, a dtype, a tensor
    value) is never widened. Every node not widened computes in the graph's element
    types, its inputs cast to them. Where exact integers concern the node, an input
    read in a float type that does not hold its values raises ArithmeticError. A Cast
    into a float type raises ArithmeticError where it does not keep every value of an
    integer input (or of a float one, where exact integers concern it). Integer
    containers compute as ONNX Runtime does, wrapping. The annotations are read from model, the ModelWrapper the node belongs
    to, which exact integers need for a standard node (ValueError without it). A
    custom op is given model and runs under the same setting: MultiThreshold computes
    exactly; others compute as they do without exact integers (QuantAvgPool2d in
    float32), a graph one runs under the setting."""

    if exact_integers is not None:
        with _switched(exact_integers):
            return execute_node(node, context, graph, opset_version, return_full_exec_context, model=model)
    if is_custom_op(node.domain, node.op_type):
        ex_cu_node.execute_custom_node(node, context, graph, onnx_opset_version=opset_version, model=model)
    else:
        # onnxruntime unfortunately does not implement run_node as defined by ONNX,
        # it can only execute entire models -- so we create a model which solely
        # consists of our current node.
        # note: ensure that the same ValueInfo does not appear both in
        # graph.value_info as well as graph.output or graph.input
        # nodes with multiple outputs that are a mix of value_info and
        # input/outputs may get them reordered below
        # note: a node's input may (also) be a top-level input or output
        node_inputs = list(filter(lambda x: x.name in node.input, graph.input))
        node_inputs += list(filter(lambda x: x.name in node.input, graph.output))
        node_inputs += list(filter(lambda x: x.name in node.input, graph.value_info))
        node_outputs = list(filter(lambda x: x.name in node.output, graph.output))
        node_outputs += list(filter(lambda x: x.name in node.output, graph.value_info))
        for attr in node.attribute:
            if attr.type == 5:
                subgraph = attr.g
                for subgraph_node in subgraph.node:
                    subgraph_node_inputs = list(filter(lambda x: x.name in subgraph_node.input, graph.value_info))
                    new_inps = list(filter(lambda x: x not in node_inputs, subgraph_node_inputs))
                    node_inputs += new_inps
        exact = exact_integers_enabled()
        widened: set[str] = set()
        concerned = False
        judged: dict[str, int] = {}
        if exact:
            if model is None:
                raise ValueError(f"{node.name} ({node.op_type}): exact integers read the model's annotations; pass model")
            widened, concerned, judged = _widened(node, node_inputs + node_outputs, model, context, opset_version)
            node_inputs = [_as_double(x) if x.name in widened else x for x in node_inputs]
            node_outputs = [_as_double(x) if x.name in widened else x for x in node_outputs]
        node_graph = helper.make_graph(
            nodes=[node],
            name="single-node-exec",
            inputs=node_inputs,
            outputs=node_outputs,
        )
        node_model = qonnx_make_model(node_graph)
        node_model.opset_import[0].version = opset_version
        # With exact integers, each input is given in the element type the node reads it
        # in; for a node they concern, a float type only where that holds the value.
        read_as = {x.name: _np_type(x.type.tensor_type.elem_type) for x in node_inputs} if exact else {}
        input_dict = dict()
        for inp in node.input:
            value = context[inp]
            np_type = read_as.get(inp)
            if value is not None and np_type is not None:
                checked = concerned and np_type.kind == "f"
                value = _read(node, inp, value, np_type, checked)
            input_dict[inp] = value

        try:
            sess = _sessions.session(node_model)
        except RuntimeNotImplemented as error:
            if not widened:
                raise
            raise NotImplementedError(
                f"{node.name} ({node.op_type}): ONNX Runtime has no float64 kernel for {node.op_type}, "
                "which this node computes in to keep its integers exact"
            ) from error
        output_list = sess.run(None, input_dict)
        if exact and node.op_type in ("Cast", "CastLike"):
            _cast_exact(node, input_dict[node.input[0]], output_list[0], concerned)
        for info, value in zip(node_outputs, output_list):
            if info.name in judged and _past(value, judged[info.name]):
                raise NotImplementedError(_unproved(node, f"its result {info.name} holds a value"))

        for output_ind in range(len(node.output)):
            # get the name of the target buffer from node.output
            outp = node.output[output_ind]

            # retrieve the index of that name in node_outputs
            list_ind = None
            for i in range(len(node_outputs)):
                if outp == node_outputs[i].name:
                    list_ind = i

            # use that index to index output_list
            if list_ind is None:
                raise Exception("Output %s not found in node outputs." % outp)
            if output_list[list_ind].shape != context[outp].shape:
                warnings.warn(
                    """Output shapes disagree after node %s execution:
                    found %s vs expected %s"""
                    % (str(node), str(output_list[list_ind].shape), str(context[outp].shape))
                )
            context[outp] = output_list[list_ind]


@overload
def execute_onnx(
    model: ModelWrapper,
    input_dict: Mapping[str, np.ndarray],
    return_full_exec_context: Literal[False] = False,
    start_node: NodeProto | None = None,
    end_node: NodeProto | None = None,
    *,
    exact_integers: bool | None = None,
) -> dict[str, np.ndarray]:
    ...


@overload
def execute_onnx(
    model: ModelWrapper,
    input_dict: Mapping[str, np.ndarray],
    return_full_exec_context: Literal[True],
    start_node: NodeProto | None = None,
    end_node: NodeProto | None = None,
    *,
    exact_integers: bool | None = None,
) -> dict[str, np.ndarray | None]:
    ...


@overload
def execute_onnx(
    model: ModelWrapper,
    input_dict: Mapping[str, np.ndarray],
    return_full_exec_context: bool,
    start_node: NodeProto | None = None,
    end_node: NodeProto | None = None,
    *,
    exact_integers: bool | None = None,
) -> dict[str, np.ndarray] | dict[str, np.ndarray | None]:
    ...


def execute_onnx(
    model: ModelWrapper,
    input_dict: Mapping[str, np.ndarray],
    return_full_exec_context: bool = False,
    start_node: NodeProto | None = None,
    end_node: NodeProto | None = None,
    *,
    exact_integers: bool | None = None,
) -> dict[str, np.ndarray] | dict[str, np.ndarray | None]:
    """Executes given ONNX ModelWrapper with given named inputs.

    If return_full_exec_context is False, a dict of named outputs is returned
    as indicated by the model.graph.output.

    If return return_full_exec_context is True, the full set of tensors used by
    the execution (including inputs, weights, activations and final outputs)
    will be returned as a dict; its entry for the empty name (an omitted
    optional input) is None.

    When start_node and end_node are set to None, the whole graph is executed.
    If they are set to particular ONNX nodes, only the subgraph between (and
    including) those nodes is executed.

    exact_integers switches exact integers for this execution and every one nested in
    it (see the exact_integers context); None, the default, keeps the setting around
    the call, off unless set. A node computed in float64 for them leaves its outputs in
    float64, graph outputs included (execute_node).
    """

    if exact_integers is not None:
        with _switched(exact_integers):
            return execute_onnx(model, input_dict, return_full_exec_context, start_node, end_node)
    if not model.check_all_tensor_shapes_specified():
        raise Exception("Found unspecified tensor shapes, try infer_shapes")
    ret = model.analysis(ta.nodes_topologically_sorted)
    assert (
        ret["nodes_topologically_sorted"] is True
    ), """Nodes must be
    topologically sorted."""

    graph = model.graph
    # first, we need to make sure that every variable required by the graph has
    # some buffer associated with it. this includes graph inputs (which includes
    # the input data as well as the trained parameters) and the graph ValueInfo
    # (intermediate tensors between layers)
    # this is provided by the execution_context, which is a dict of np.ndarray
    execution_context = model.make_empty_exec_context()
    # fill in any inputs provided to this function
    for inp_name in input_dict.keys():
        if inp_name in execution_context:
            if execution_context[inp_name].shape == input_dict[inp_name].shape:
                execution_context[inp_name] = input_dict[inp_name]
            else:
                raise Exception(
                    "Shape mismatch for provided input %s: found %s expected %s "
                    % (
                        inp_name,
                        str(execution_context[inp_name].shape),
                        str(input_dict[inp_name].shape),
                    )
                )
        # else:
        # raise Exception("Provided input not found in graph context: %s" % inp_name)

    # check if model has an execution mode set
    # if None, execute model node by node using execute_node()
    model_exec_mode = model.get_metadata_prop("exec_mode")
    if (model_exec_mode is None) or (model_exec_mode == ""):
        # extract opset version for node-by-node execution
        opset_imports = model.get_opset_imports()
        # execute the model node by node
        # we can simply walk down the list since the ONNX spec guarantees that it is
        # topologically sorted
        subgraph = []
        if start_node is None:
            start_node = model.graph.node[0]
        if end_node is None:
            end_node = model.graph.node[-1]
        # select the nodes between specified start/end nodes
        start_ind = model.get_node_index(start_node)
        end_ind = model.get_node_index(end_node) + 1
        assert end_ind >= start_ind, "Start/end nodes must define valid subgraph"
        subgraph = graph.node[start_ind:end_ind]
        for node in subgraph:
            if get_sanitize_quant_tensors() != 0:
                # round input values to match quantization annotation
                execution_context = sanitize_quant_values(model, node.input, execution_context)
            if node.domain in opset_imports:
                opset_version = opset_imports[node.domain]
            else:
                opset_version = get_preferred_qonnx_opset()
            execute_node(node, execution_context, graph, opset_version, return_full_exec_context, model=model)
            if get_sanitize_quant_tensors() != 0:
                # round output values to quantization annotation
                execution_context = sanitize_quant_values(model, node.output, execution_context)
    else:
        raise Exception('Metadata property "exec_mode" is set to an unknown value.')

    if return_full_exec_context:
        return execution_context
    else:
        # provide outputs as dict
        output_dict = dict()
        for out_tensor in graph.output:
            out_name = out_tensor.name
            output_dict[out_name] = execution_context[out_name]
        return output_dict


def execute_onnx_and_make_model(
    model: ModelWrapper, input_dict: Mapping[str, Any], *, exact_integers: bool | None = None
) -> ModelWrapper:
    """Executes given ONNX ModelWrapper with given named inputs and return a new
    ModelWrapper where an initializer is provided for each tensor as taken from
    the execution. This new model is useful for debugging, since it contains
    all the intermediate activation values.

    exact_integers is execute_onnx's."""

    # retrieve the full execution context
    execution_context = execute_onnx(model, input_dict, True, exact_integers=exact_integers)
    new_model = copy.deepcopy(model)
    # create value_info entries and initializers for everything
    for name, value in execution_context.items():
        if name and value is not None:
            new_model.set_initializer(name, value)
    for vi in new_model.graph.value_info:
        new_model.graph.output.append(vi)
    return new_model


def compare_execution(
    model_a,
    model_b,
    input_dict,
    compare_fxn=lambda x, y: np.isclose(x, y, atol=1e-3).all(),
):
    """Executes two ONNX models and compare their outputs using given function.

    compare_fxn should take in two tensors and return a Boolean"""
    # compare values from first output tensors produced
    res_a = list(execute_onnx(model_a, input_dict).items())[0][1]
    res_b = list(execute_onnx(model_b, input_dict).items())[0][1]
    return compare_fxn(res_a, res_b)
