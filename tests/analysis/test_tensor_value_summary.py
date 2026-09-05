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

import pytest

import numpy as np
import onnx.helper as oh
from onnx import TensorProto

from qonnx.analysis.tensor_value_summary import (
    TensorValueSummary,
    UnsupportedTensorValueError,
    initializer_value_summaries,
    initializer_value_summary,
    is_summarizable_dtype,
    smallest_lossless_datatype,
    summarize_tensor_values,
)
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.util.basic import qonnx_make_model


def make_model_with_initializers(initializers):
    """Builds a two-Add model whose parameter tensors carry initializers.

    initializers: dict of tensor name -> numpy array."""
    top_in = oh.make_tensor_value_info("top_in", TensorProto.FLOAT, [2])
    top_out = oh.make_tensor_value_info("top_out", TensorProto.FLOAT, [2])
    param_names = list(initializers.keys())
    nodes = []
    value_info = []
    previous = "top_in"
    for index, name in enumerate(param_names):
        value_info.append(oh.make_tensor_value_info(name, TensorProto.FLOAT, [2]))
        is_last = index == len(param_names) - 1
        output = "top_out" if is_last else "middle%d" % index
        nodes.append(oh.make_node("Add", [previous, name], [output]))
        previous = output
    modelproto = qonnx_make_model(
        oh.make_graph(
            name="test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=value_info,
            nodes=nodes,
        )
    )
    model = ModelWrapper(modelproto)
    for name, array in initializers.items():
        model.set_initializer(name, array)
    return model


# --- value-level summaries ---------------------------------------------------


def test_summary_of_integer_valued_tensor():
    summary = summarize_tensor_values(np.array([[-3, 0, 7], [1, 2, 3]], dtype=np.int8))
    assert summary.element_count == 6
    assert summary.minimum == -3
    assert summary.maximum == 7
    assert summary.is_integral is True
    assert isinstance(summary.minimum, int)
    assert len(summary.content_digest) == 64


def test_summary_is_deterministic_and_content_addressed():
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    first = summarize_tensor_values(values)
    second = summarize_tensor_values(values.copy())
    assert first == second
    assert first.content_digest == second.content_digest
    # equal values are the same fact however the array is laid out in memory
    non_contiguous = np.asfortranarray(values)
    assert summarize_tensor_values(non_contiguous) == first


def test_digest_moves_on_shape_dtype_or_byte_change():
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    baseline = summarize_tensor_values(values).content_digest
    reshaped = summarize_tensor_values(values.reshape(4, 3)).content_digest
    retyped = summarize_tensor_values(values.astype(np.float64)).content_digest
    changed = values.copy()
    changed[0, 0] = 100.0
    rebytes = summarize_tensor_values(changed).content_digest
    assert len({baseline, reshaped, retyped, rebytes}) == 4


def test_negative_zero_is_different_content():
    positive = summarize_tensor_values(np.array([0.0], dtype=np.float32))
    negative = summarize_tensor_values(np.array([-0.0], dtype=np.float32))
    assert positive.content_digest != negative.content_digest
    assert positive.minimum == negative.minimum == 0.0


def test_empty_tensor_has_absent_range():
    summary = summarize_tensor_values(np.zeros((0, 4), dtype=np.float32))
    assert summary.element_count == 0
    assert summary.minimum is None
    assert summary.maximum is None
    # empty is not zero and must not authorize a zero-range optimization
    assert smallest_lossless_datatype(summary) is None


def test_empty_tensors_of_different_shape_differ():
    first = summarize_tensor_values(np.zeros((0, 4), dtype=np.float32))
    second = summarize_tensor_values(np.zeros((0, 5), dtype=np.float32))
    assert first.element_count == second.element_count == 0
    assert first.content_digest != second.content_digest


def test_boolean_tensor_is_summarized_as_zero_one():
    summary = summarize_tensor_values(np.array([True, False, True]))
    assert (summary.minimum, summary.maximum) == (0, 1)
    assert summary.is_integral is True
    assert smallest_lossless_datatype(summary) == DataType["BINARY"]


def test_fractional_float_tensor_is_not_integral():
    summary = summarize_tensor_values(np.array([0.0, 0.5, 1.0], dtype=np.float32))
    assert summary.minimum == 0.0
    assert summary.maximum == 1.0
    assert summary.is_integral is False
    # a range alone cannot prove losslessness for fractional values
    assert smallest_lossless_datatype(summary) is None


def test_nan_is_excluded_from_range_but_refuses_narrowing():
    summary = summarize_tensor_values(np.array([1.0, np.nan, 3.0], dtype=np.float32))
    assert (summary.minimum, summary.maximum) == (1.0, 3.0)
    assert summary.is_integral is False
    assert smallest_lossless_datatype(summary) is None
    # equal content still compares equal, which NaN in the range would break
    assert summary == summarize_tensor_values(np.array([1.0, np.nan, 3.0], dtype=np.float32))


def test_all_nan_tensor_reports_absent_range():
    summary = summarize_tensor_values(np.array([np.nan, np.nan], dtype=np.float32))
    assert summary.element_count == 2
    assert summary.minimum is None
    assert summary.maximum is None
    assert summary.is_integral is False
    assert smallest_lossless_datatype(summary) is None


def test_infinity_is_an_observed_value_but_refuses_narrowing():
    summary = summarize_tensor_values(np.array([1.0, np.inf], dtype=np.float32))
    assert summary.minimum == 1.0
    assert summary.maximum == np.inf
    assert summary.is_integral is False
    assert smallest_lossless_datatype(summary) is None


def test_wide_integer_tensors_are_exact():
    values = np.array([-(2**62), 2**62], dtype=np.int64)
    summary = summarize_tensor_values(values)
    assert summary.minimum == -(2**62)
    assert summary.maximum == 2**62
    values = np.array([2**64 - 1], dtype=np.uint64)
    assert summarize_tensor_values(values).maximum == 2**64 - 1


# --- unsupported categories --------------------------------------------------


def test_string_tensor_is_refused():
    with pytest.raises(UnsupportedTensorValueError):
        summarize_tensor_values(np.array(["a", "b"]))
    with pytest.raises(UnsupportedTensorValueError):
        summarize_tensor_values(np.array([object()], dtype=object))


def test_complex_tensor_is_refused():
    with pytest.raises(UnsupportedTensorValueError):
        summarize_tensor_values(np.array([1 + 2j], dtype=np.complex64))


def test_custom_encoded_dtypes_are_refused():
    # bfloat16, the float8 variants and the sub-byte integers arrive from
    # onnx.numpy_helper as view dtypes over raw bit patterns; a range over
    # those bytes is meaningless rather than merely imprecise
    bfloat16_view = np.dtype((np.uint16, [("bfloat16", "<u2")]))
    float8_view = np.dtype((np.uint8, [("e4m3fn", "u1")]))
    int4_view = np.dtype((np.int8, [("int4", "i1")]))
    assert not is_summarizable_dtype(bfloat16_view)
    assert not is_summarizable_dtype(float8_view)
    assert not is_summarizable_dtype(int4_view)
    with pytest.raises(UnsupportedTensorValueError):
        summarize_tensor_values(np.zeros(4, dtype=bfloat16_view))


def test_supported_dtype_predicate():
    for dtype in [np.int8, np.uint8, np.int64, np.uint64, np.float16, np.float32, np.float64, bool]:
        assert is_summarizable_dtype(np.dtype(dtype))
    for dtype in [np.complex64, np.complex128, object, np.str_]:
        assert not is_summarizable_dtype(np.dtype(dtype))


# --- model-level analysis ----------------------------------------------------


def test_absent_initializer_is_absent_not_zero():
    model = make_model_with_initializers({"p0": np.array([1.0, 2.0], dtype=np.float32)})
    assert initializer_value_summary(model, "top_in") is None
    assert initializer_value_summary(model, "no_such_tensor") is None
    assert initializer_value_summary(model, "p0") is not None


def test_constant_node_value_needs_folding_first():
    # a Constant node's output has a static value but no initializer, so only
    # initializers are summarized until the value has been folded into one
    top_in = oh.make_tensor_value_info("top_in", TensorProto.FLOAT, [2])
    top_out = oh.make_tensor_value_info("top_out", TensorProto.FLOAT, [2])
    const_value = oh.make_tensor("value", TensorProto.FLOAT, [2], [1.0, 2.0])
    modelproto = qonnx_make_model(
        oh.make_graph(
            name="test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=[oh.make_tensor_value_info("c0", TensorProto.FLOAT, [2])],
            nodes=[
                oh.make_node("Constant", [], ["c0"], value=const_value),
                oh.make_node("Add", ["top_in", "c0"], ["top_out"]),
            ],
        )
    )
    model = ModelWrapper(modelproto)
    assert initializer_value_summary(model, "c0") is None
    assert initializer_value_summaries(model) == {}
    model = model.transform(FoldConstants())
    folded = initializer_value_summary(model, "c0")
    assert folded == summarize_tensor_values(np.array([1.0, 2.0], dtype=np.float32))


@pytest.mark.parametrize("onnx_dtype", ["BFLOAT16", "FLOAT8E4M3FN", "FLOAT8E5M2", "INT4", "UINT4"])
def test_unsupported_initializer_is_reported_not_summarized(onnx_dtype):
    # an unsupported initializer must not be confused with an absent one.
    # INT4/UINT4 are excluded even though QONNX has INT4/UINT4 datatypes: the
    # exclusion is about the packed bytes numpy_helper hands back, not about
    # the datatype being inexpressible.
    tensor_proto_dtype = getattr(TensorProto, onnx_dtype, None)
    if tensor_proto_dtype is None:
        pytest.skip("%s not available in this onnx version" % onnx_dtype)
    model = make_model_with_initializers({"p0": np.array([1.0, 2.0], dtype=np.float32)})
    model.graph.initializer[0].CopyFrom(oh.make_tensor("p0", tensor_proto_dtype, [2], [1, 1]))
    with pytest.raises(UnsupportedTensorValueError):
        initializer_value_summary(model, "p0")
    with pytest.raises(UnsupportedTensorValueError):
        model.analysis(initializer_value_summaries)


def test_equal_content_under_two_names_is_one_fact():
    values = np.array([1.0, -2.0], dtype=np.float32)
    model = make_model_with_initializers({"p0": values, "p1": values.copy()})
    assert initializer_value_summary(model, "p0") == initializer_value_summary(model, "p1")


def test_renamed_tensor_keeps_its_summary():
    values = np.array([1.0, -2.0], dtype=np.float32)
    before = initializer_value_summary(make_model_with_initializers({"p0": values}), "p0")
    after = initializer_value_summary(make_model_with_initializers({"weights": values}), "weights")
    assert before == after


def test_changed_content_changes_the_summary():
    model = make_model_with_initializers({"p0": np.array([1.0, 2.0], dtype=np.float32)})
    before = initializer_value_summary(model, "p0")
    model.set_initializer("p0", np.array([1.0, 3.0], dtype=np.float32))
    after = initializer_value_summary(model, "p0")
    assert before != after
    assert before.content_digest != after.content_digest


def test_repeated_reads_observe_one_equal_summary():
    model = make_model_with_initializers({"p0": np.array([1.0, 2.0], dtype=np.float32)})
    observations = [initializer_value_summary(model, "p0") for _ in range(3)]
    assert observations[0] == observations[1] == observations[2]


def test_analysis_pass_summarizes_every_initializer():
    model = make_model_with_initializers(
        {
            "p0": np.array([1.0, 2.0], dtype=np.float32),
            "p1": np.array([-1.0, 0.0], dtype=np.float32),
        }
    )
    summaries = model.analysis(initializer_value_summaries)
    assert set(summaries.keys()) == {"p0", "p1"}
    assert summaries["p0"] == initializer_value_summary(model, "p0")
    assert all(isinstance(s, TensorValueSummary) for s in summaries.values())


# --- smallest lossless datatype ----------------------------------------------


@pytest.mark.parametrize(
    "values, expected",
    [
        ([0, 1], "BINARY"),
        ([0, 0], "BINARY"),
        ([-1, 0, 1], "TERNARY"),
        ([-1, -1, 1], "TERNARY"),
        ([-2, 1], "INT2"),
        ([-8, 7], "INT4"),
        ([-7, 7], "INT4"),
        ([0, 15], "UINT4"),
        ([0, 255], "UINT8"),
        ([-128, 127], "INT8"),
        ([-129, 127], "INT9"),
    ],
)
def test_smallest_lossless_datatype_for_ranges(values, expected):
    summary = summarize_tensor_values(np.array(values, dtype=np.float32))
    assert smallest_lossless_datatype(summary) == DataType[expected]


def test_bipolar_is_never_offered_from_a_range():
    # {-1, +1} and {-1, 0, +1} share one range; only TERNARY is sound for both
    summary = summarize_tensor_values(np.array([-1.0, 1.0], dtype=np.float32))
    assert smallest_lossless_datatype(summary) == DataType["TERNARY"]


def test_declared_datatype_bounds_the_result():
    summary = summarize_tensor_values(np.array([-3.0, 3.0], dtype=np.float32))
    # narrowing below a wide declared type is the useful case
    assert smallest_lossless_datatype(summary, DataType["FLOAT32"]) == DataType["INT3"]
    assert smallest_lossless_datatype(summary, DataType["INT8"]) == DataType["INT3"]
    # never wider than declared: an equal-or-wider candidate keeps the declared type
    assert smallest_lossless_datatype(summary, DataType["INT3"]) == DataType["INT3"]
    # a declared type that cannot hold the observed values is refused
    assert smallest_lossless_datatype(summary, DataType["UINT8"]) is None
    assert smallest_lossless_datatype(summary, DataType["INT2"]) is None


def test_datatype_without_a_range_is_refused():
    summary = summarize_tensor_values(np.array([1.0, 2.0], dtype=np.float32))
    assert smallest_lossless_datatype(summary, DataType["SCALEDINT<8>"]) is None


def test_out_of_range_integral_values_are_refused():
    summary = summarize_tensor_values(np.array([2.0**70], dtype=np.float64))
    assert summary.is_integral is True
    assert smallest_lossless_datatype(summary) is None


def test_narrowness_question_is_derivable_from_the_summary():
    # the consumer-side question "does this tensor exclude its datatype
    # minimum?" is answered from the summary, not by rescanning the array
    declared = DataType["INT4"]
    narrow = summarize_tensor_values(np.array([-7.0, 7.0], dtype=np.float32))
    full = summarize_tensor_values(np.array([-8.0, 7.0], dtype=np.float32))
    assert narrow.minimum > declared.min()
    assert not (full.minimum > declared.min())


def test_summaries_are_comparable_without_a_model():
    # a composition-level consumer can combine several summaries on their own
    first = summarize_tensor_values(np.array([-3.0, 2.0], dtype=np.float32))
    second = summarize_tensor_values(np.array([0.0, 9.0], dtype=np.float32))
    combined_min = min(first.minimum, second.minimum)
    combined_max = max(first.maximum, second.maximum)
    shared = TensorValueSummary(
        content_digest="",
        element_count=first.element_count + second.element_count,
        minimum=combined_min,
        maximum=combined_max,
        is_integral=first.is_integral and second.is_integral,
    )
    assert smallest_lossless_datatype(shared) == DataType["INT5"]
