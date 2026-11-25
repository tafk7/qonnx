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

"""Constant folding of small integers: every standard operator but those whose
attributes fix an element type (the bounded integer arithmetic, a Gemm scaled by a
fraction, the operators that move, select or round values, and those with no proof),
its float tensors annotated INT32 and given small integers, folds to the bytes ONNX
Runtime computes for its one-node model, as folding did before it evaluated exactly. So
do integer-annotated nodes given values that are not integers."""

from __future__ import annotations

import pytest

import numpy as np
import onnx
import onnxruntime as rt
from onnx import TensorProto, helper, numpy_helper
from typing import Any, Callable

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.fold_constants import FoldConstants, FoldConstantsFiltered

OPSET = 21
SHAPE = (1, 2, 3, 3)
FOLDS: list[Callable[[], Transformation]] = [FoldConstants, lambda: FoldConstantsFiltered(lambda model, node: True)]


def data(shape: tuple[int, ...] = SHAPE, offset: int = 0) -> np.ndarray:
    """Small positive integers, 1 to 4, as float32."""
    return ((np.arange(int(np.prod(shape))) + offset) % 4 + 1).astype(np.float32).reshape(shape)


def ints(*values: int) -> np.ndarray:
    return np.array(values, np.int64)


def floats(*values: float) -> np.ndarray:
    return np.array(values, np.float32)


def bytes_(*values: int) -> np.ndarray:
    return np.array(values, np.uint8)


def scalar(value: Any, kind: Any) -> np.ndarray:
    return np.array(value, kind)


def info(name: str, kind: int, shape: list[int]) -> Any:
    return helper.make_tensor_value_info(name, kind, shape)


X = data()
POOL = {"kernel_shape": [2, 2]}
BRANCH = helper.make_graph(
    [helper.make_node("Constant", [], ["b"], value=numpy_helper.from_array(data((2, 2)), "b"))],
    "branch",
    [],
    [info("b", TensorProto.FLOAT, [2, 2])],
)
LOOP_BODY = helper.make_graph(
    [
        helper.make_node("Identity", ["cond"], ["cond_out"]),
        helper.make_node("Add", ["carried", "carried"], ["carried_out"]),
    ],
    "loop_body",
    [info("i", TensorProto.INT64, []), info("cond", TensorProto.BOOL, []), info("carried", TensorProto.FLOAT, [2])],
    [info("cond_out", TensorProto.BOOL, []), info("carried_out", TensorProto.FLOAT, [2])],
)
SCAN_BODY = helper.make_graph(
    [
        helper.make_node("Add", ["state", "element"], ["state_out"]),
        helper.make_node("Identity", ["state_out"], ["scanned"]),
    ],
    "scan_body",
    [info("state", TensorProto.FLOAT, [2]), info("element", TensorProto.FLOAT, [2])],
    [info("state_out", TensorProto.FLOAT, [2]), info("scanned", TensorProto.FLOAT, [2])],
)
RECURRENT = {"hidden_size": 2}

# Inputs by name, where the generic choice from the schema does not fit.
NAMED = {
    "axes": ints(1),
    "shape": ints(1, 2, 3, 3),
    "repeats": ints(1, 1, 1, 1),
    "pads": ints(0, 0, 1, 1, 0, 0, 1, 1),
    "starts": ints(0),
    "ends": ints(1),
    "steps": ints(1),
    "k": ints(0),
    "indices": ints(0, 1),
}

# Operators whose case the schema does not give: inputs in order, attributes, and the
# number of outputs where it is not the schema's least.
CASES: dict[str, dict[str, Any]] = {
    "AffineGrid": {"inputs": [floats(1, 0, 0, 0, 1, 0).reshape(1, 2, 3), ints(1, 1, 2, 2)]},
    # Atanh(1) is infinite, past what float32 holds exactly: exact integers refuse it.
    "Atanh": {"inputs": [np.zeros(SHAPE, np.float32)]},
    "AveragePool": {"inputs": [X], "attrs": POOL},
    "BatchNormalization": {"inputs": [X, floats(1, 2), floats(1, 1), floats(0, 1), floats(1, 1)]},
    "BitShift": {"inputs": [bytes_(1, 2, 3), bytes_(1, 1, 2)], "attrs": {"direction": "LEFT"}},
    "BlackmanWindow": {"inputs": [scalar(4, np.int64)]},
    "Col2Im": {"inputs": [data((1, 4, 4)), ints(3, 3), ints(2, 2)]},
    "Concat": {"inputs": [X, data(offset=1)], "attrs": {"axis": 1}},
    "ConvInteger": {"inputs": [bytes_(*range(9)).reshape(1, 1, 3, 3), bytes_(1, 2, 3, 4).reshape(1, 1, 2, 2)]},
    "ConvTranspose": {"inputs": [X, data((2, 1, 2, 2))]},
    "DFT": {"inputs": [data((1, 4, 1))]},
    "DepthToSpace": {"inputs": [data((1, 4, 2, 2))], "attrs": {"blocksize": 2}},
    "DequantizeLinear": {"inputs": [bytes_(1, 2, 3), floats(2), bytes_(1)]},
    "DynamicQuantizeLinear": {"inputs": [floats(1, 2, 3, 4)], "outputs": 3},
    "Einsum": {"inputs": [data((2, 3)), data((3, 2), 1)], "attrs": {"equation": "ij,jk->ik"}},
    "GRU": {"inputs": [data((1, 1, 2)), data((1, 6, 2)), data((1, 6, 2), 1)], "attrs": RECURRENT, "outputs": 1},
    "Gather": {"inputs": [X, ints(0)], "attrs": {"axis": 1}},
    "GatherElements": {"inputs": [X, np.zeros(SHAPE, np.int64)], "attrs": {"axis": 1}},
    "GridSample": {"inputs": [X, np.zeros((1, 2, 2, 2), np.float32)]},
    "GroupNormalization": {"inputs": [X, floats(1, 1), floats(0, 0)], "attrs": {"num_groups": 2}, "typed": SHAPE},
    "HammingWindow": {"inputs": [scalar(4, np.int64)]},
    "HannWindow": {"inputs": [scalar(4, np.int64)]},
    "If": {"inputs": [scalar(True, bool)], "attrs": {"then_branch": BRANCH, "else_branch": BRANCH}},
    "InstanceNormalization": {"inputs": [X, floats(1, 1), floats(0, 0)]},
    "LRN": {"inputs": [X], "attrs": {"size": 3}},
    "LSTM": {"inputs": [data((1, 1, 2)), data((1, 8, 2)), data((1, 8, 2), 1)], "attrs": RECURRENT, "outputs": 1},
    "LayerNormalization": {"inputs": [X, data((3,))]},
    "Loop": {"inputs": [ints(2), scalar(True, bool), floats(1, 2)], "attrs": {"body": LOOP_BODY}},
    "LpPool": {"inputs": [X], "attrs": POOL},
    "MatMulInteger": {"inputs": [bytes_(1, 2, 3, 4).reshape(2, 2), bytes_(1, 2, 3, 4).reshape(2, 2)]},
    "MaxPool": {"inputs": [X], "attrs": POOL},
    "MaxRoiPool": {"inputs": [X, floats(0, 0, 0, 2, 2).reshape(1, 5)], "attrs": {"pooled_shape": [2, 2]}},
    "MaxUnpool": {
        "inputs": [data((1, 1, 2, 2)), ints(0, 3, 12, 15).reshape(1, 1, 2, 2)],
        "attrs": {**POOL, "strides": [2, 2]},
    },
    "MelWeightMatrix": {
        "inputs": [
            scalar(2, np.int64),
            scalar(8, np.int64),
            scalar(16, np.int64),
            scalar(0, np.float32),
            scalar(8, np.float32),
        ]
    },
    "Mod": {"inputs": [X, data(offset=1)], "attrs": {"fmod": 1}},
    "NegativeLogLikelihoodLoss": {"inputs": [data((2, 3)), ints(0, 2)]},
    "NonMaxSuppression": {"inputs": [floats(0, 0, 1, 1, 0, 0, 2, 2).reshape(1, 2, 4), floats(1, 2).reshape(1, 1, 2)]},
    "OneHot": {"inputs": [ints(0, 2), ints(3), floats(0, 1)]},
    "QLinearConv": {
        "inputs": [
            bytes_(*range(9)).reshape(1, 1, 3, 3),
            floats(1),
            bytes_(0),
            bytes_(1, 2, 3, 4).reshape(1, 1, 2, 2),
            floats(1),
            bytes_(0),
            floats(4),
            bytes_(0),
        ]
    },
    "QLinearMatMul": {
        "inputs": [
            bytes_(1, 2, 3, 4).reshape(2, 2),
            floats(1),
            bytes_(0),
            bytes_(1, 2, 3, 4).reshape(2, 2),
            floats(1),
            bytes_(0),
            floats(4),
            bytes_(0),
        ]
    },
    "QuantizeLinear": {"inputs": [floats(1, 2, 3), floats(2), bytes_(1)]},
    "RNN": {"inputs": [data((1, 1, 2)), data((1, 2, 2)), data((1, 2, 2), 1)], "attrs": RECURRENT, "outputs": 1},
    "Range": {"inputs": [scalar(1, np.float32), scalar(9, np.float32), scalar(2, np.float32)]},
    "Resize": {"inputs": [X, floats(), floats(1, 1, 2, 2)]},
    "ReverseSequence": {"inputs": [data((3, 2)), ints(2, 3)], "attrs": {"batch_axis": 1, "time_axis": 0}},
    "RoiAlign": {"inputs": [X, floats(0, 0, 2, 2).reshape(1, 4), ints(0)], "attrs": {"output_height": 2, "output_width": 2}},
    "STFT": {"inputs": [data((1, 8, 1)), scalar(4, np.int64), floats(1, 1, 1, 1), scalar(4, np.int64)]},
    "Scan": {"inputs": [floats(0, 0), data((3, 2))], "attrs": {"body": SCAN_BODY, "num_scan_inputs": 1}, "outputs": 2},
    "Scatter": {"inputs": [X, np.zeros(SHAPE, np.int64), data(offset=1)], "attrs": {"axis": 1}},
    "ScatterElements": {"inputs": [X, np.zeros(SHAPE, np.int64), data(offset=1)], "attrs": {"axis": 1}},
    "ScatterND": {"inputs": [X, ints(0, 1).reshape(1, 2), data((1, 3, 3), 1)]},
    "SoftmaxCrossEntropyLoss": {"inputs": [data((2, 3)), ints(0, 2)]},
    "SpaceToDepth": {"inputs": [data((1, 1, 4, 4))], "attrs": {"blocksize": 2}},
    "Split": {"inputs": [X, ints(1, 1)], "attrs": {"axis": 1}, "outputs": 2},
    "TfIdfVectorizer": {
        "inputs": [ints(1, 2, 1, 3)],
        "attrs": {
            "mode": "TF",
            "min_gram_length": 1,
            "max_gram_length": 1,
            "max_skip_count": 0,
            "ngram_counts": [0],
            "ngram_indexes": [0, 1],
            "pool_int64s": [1, 2],
        },
    },
    "TopK": {"inputs": [X, ints(2)], "attrs": {"axis": 3}, "outputs": 2},
    "Upsample": {"inputs": [X, floats(1, 1, 2, 2)]},
    "Where": {"inputs": [np.ones(SHAPE, bool), X, data(offset=1)]},
}
GEMM = [data((2, 3)), data((3, 2), 1), data((2, 2), 2)]
CASES["Gemm"] = {"inputs": GEMM}
# A Gemm scaled by a fraction has no proof; named op/attribute, built as op.
SCALED = {
    "Gemm/alpha=0.5": {"inputs": GEMM, "attrs": {"alpha": 0.5}},
    "Gemm/beta=0.5": {"inputs": GEMM, "attrs": {"beta": 0.5}},
}
CASES.update(SCALED)

# Operators with no case: sequences, optionals, strings and images hold no tensor of
# numbers to fold, and ONNX Runtime has no DeformConv kernel.
UNBUILT = frozenset(
    {
        "ConcatFromSequence",
        "DeformConv",
        "ImageDecoder",
        "Optional",
        "OptionalGetElement",
        "OptionalHasElement",
        "RegexFullMatch",
        "SequenceAt",
        "SequenceConstruct",
        "SequenceEmpty",
        "SequenceErase",
        "SequenceInsert",
        "SequenceLength",
        "SequenceMap",
        "SplitToSequence",
        "StringConcat",
        "StringNormalizer",
        "StringSplit",
    }
)


def schema(op: str) -> Any:
    """op's schema at OPSET, or its last before it was deprecated."""
    found = onnx.defs.get_schema(op, OPSET)
    return onnx.defs.get_schema(op, found.since_version - 1) if found.deprecated else found


def generic(op: str) -> dict[str, Any]:
    """op's required inputs from its schema: a named input where NAMED has it, else
    small integers in float32 where the input takes them, else int64 or bool."""
    allowed = {constraint.type_param_str: list(constraint.allowed_type_strs) for constraint in schema(op).type_constraints}
    inputs: list[np.ndarray] = []
    for index, formal in enumerate(schema(op).inputs):
        if formal.option == onnx.defs.OpSchema.FormalParameterOption.Optional:
            break
        types = allowed.get(formal.type_str, [formal.type_str])
        named = NAMED.get(formal.name)
        if named is not None and "tensor(int64)" in types:
            inputs.append(named)
        elif "tensor(float)" in types:
            inputs.append(data(offset=index))
        elif "tensor(int64)" in types:
            inputs.append(ints(1))
        elif "tensor(bool)" in types:
            inputs.append(np.ones(SHAPE, bool))
        else:
            raise AssertionError(f"{op}: no generic case for {formal.name}")
        if formal.option == onnx.defs.OpSchema.FormalParameterOption.Variadic:
            inputs.append(data(offset=index + 1))
    return {"inputs": inputs}


def build(op: str) -> onnx.ModelProto:
    """A model of op over constants c0, c1, ..., its outputs typed by inference, at
    OPSET (an operator deprecated by then at its last opset before); op may name a
    case of SCALED."""
    case = CASES.get(op) or generic(op)
    op = op.split("/")[0]
    found = schema(op)
    opset = found.since_version if onnx.defs.get_schema(op, OPSET).deprecated else OPSET
    names = [f"c{index}" for index in range(len(case["inputs"]))]
    outputs = [f"y{index}" for index in range(case.get("outputs", found.min_output))]
    node = helper.make_node(op, names, outputs, name=f"{op}_0", **case.get("attrs", {}))
    graph = helper.make_graph(
        [node],
        "operator",
        [],
        [],
        initializer=[numpy_helper.from_array(value, name) for name, value in zip(names, case["inputs"])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    inferred = {value.name: value for value in onnx.shape_inference.infer_shapes(model).graph.value_info}
    for name in outputs:
        if "typed" in case:
            model.graph.output.append(info(name, TensorProto.FLOAT, list(case["typed"])))
        else:
            model.graph.output.append(inferred[name])
    return model


def computed(model: onnx.ModelProto) -> np.ndarray:
    """The first output ONNX Runtime computes for the model."""
    return rt.InferenceSession(model.SerializeToString()).run(None, {})[0]


def consumed(model: onnx.ModelProto) -> ModelWrapper:
    """The model, its first output consumed with a graph input x (Add, or Xor for bool)
    into the graph output z, so that folding keeps it; every float tensor annotated
    INT32."""
    graph = onnx.GraphProto()
    graph.CopyFrom(model.graph)
    first = graph.output[0]
    kind = first.type.tensor_type.elem_type
    consumer = "Xor" if kind == TensorProto.BOOL else "Add"
    graph.node.append(helper.make_node(consumer, [first.name, "x"], ["z"], name=f"{consumer}_1"))
    graph.input.append(info("x", kind, [1]))
    graph.value_info.extend(graph.output)
    del graph.output[:]
    graph.output.append(helper.make_tensor_value_info("z", kind, None))
    wrapper = ModelWrapper(helper.make_model(graph, opset_imports=model.opset_import))
    for tensor in [*graph.initializer, *graph.value_info]:
        kind = tensor.data_type if isinstance(tensor, onnx.TensorProto) else tensor.type.tensor_type.elem_type
        if kind == TensorProto.FLOAT:
            wrapper.set_tensor_datatype(tensor.name, DataType["INT32"])
    return wrapper


def folded(model: onnx.ModelProto, fold: Callable[[], Transformation]) -> np.ndarray | None:
    """The model's first output as folding stores it, every float tensor annotated
    INT32; None where folding leaves it unfolded."""
    return consumed(model).transform(fold()).get_initializer(model.graph.output[0].name)


def swept() -> list[str]:
    """Every standard operator but those named as fixing an element type (_TYPED), and
    the scaled Gemm cases."""
    from qonnx.core.onnx_exec import _BOUNDED, _TYPED

    standard = {found.name for found in onnx.defs.get_all_schemas_with_history() if found.domain == ""}
    assert _BOUNDED <= standard
    return sorted(standard - _TYPED) + list(SCALED)


def test_every_swept_operator_but_the_unbuilt_has_a_case() -> None:
    assert UNBUILT <= set(swept())
    for op in swept():
        if op not in UNBUILT:
            build(op)


# Operators whose output's size depends on values warn that it disagrees with the
# stated shape, as they always did.
@pytest.mark.filterwarnings("ignore:Output shapes disagree")
@pytest.mark.parametrize("fold", FOLDS, ids=["FoldConstants", "FoldConstantsFiltered"])
@pytest.mark.parametrize("op", [op for op in swept() if op not in UNBUILT])
def test_a_small_integer_case_folds_to_the_bytes_onnx_runtime_computes(op: str, fold: Any) -> None:
    model = build(op)
    expected = computed(model)
    value = folded(model, fold)
    assert value is not None, f"{op} was left unfolded"
    assert value.dtype == expected.dtype and value.shape == expected.shape
    assert value.tobytes() == expected.tobytes()


def pair(op: str, a: float, b: float, annotation: str | None) -> ModelWrapper:
    """c = op(a, b) of float32 constants, then y = x + c, at opset 13; a, b and c
    annotated where annotation is given."""
    graph = helper.make_graph(
        [helper.make_node(op, ["a", "b"], ["c"], name=f"{op}_0"), helper.make_node("Add", ["x", "c"], ["y"], name="Add_1")],
        "pair",
        [info("x", TensorProto.FLOAT, [1])],
        [info("y", TensorProto.FLOAT, [1])],
        initializer=[numpy_helper.from_array(floats(a), "a"), numpy_helper.from_array(floats(b), "b")],
        value_info=[info("c", TensorProto.FLOAT, [1])],
    )
    wrapper = ModelWrapper(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]))
    if annotation is not None:
        for name in ("a", "b", "c"):
            wrapper.set_tensor_datatype(name, DataType[annotation])
    return wrapper


@pytest.mark.parametrize("fold", FOLDS, ids=["FoldConstants", "FoldConstantsFiltered"])
@pytest.mark.parametrize(
    "op,a,b,annotation,expected",
    [
        ("Div", 7, 2, "INT32", 3.5),
        ("Pow", 2, 3, "INT32", 8),
        ("Add", 0.5, 0.25, "INT8", 0.75),
        ("Mul", 1.5, 2, "INT8", 3),
        ("Div", 7, 2, "INT8", 3.5),
        ("Add", 3, 4, "INT8", 7),
        ("Add", 0.5, 0.25, None, 0.75),
    ],
)
def test_an_integer_annotated_constant_folds_as_before(
    op: str, a: float, b: float, annotation: str | None, expected: float, fold: Any
) -> None:
    """A result within float32's exact integers, and a node given a value that is not an
    integer, fold to the bytes they always did."""
    c = pair(op, a, b, annotation).transform(fold()).get_initializer("c")
    assert c is not None and c.dtype == np.float32 and c.tobytes() == floats(expected).tobytes()
