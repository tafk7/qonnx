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

import dataclasses
import numpy as np
import onnx.helper as oh
from onnx import TensorProto

from qonnx.analysis.tensor_value_summary import (
    TensorValueSummary,
    UnsupportedTensorValueError,
    initializer_value_summaries,
    initializer_value_summary,
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


def test_empty_tensors_of_different_shape_differ():
    first = summarize_tensor_values(np.zeros((0, 4), dtype=np.float32))
    second = summarize_tensor_values(np.zeros((0, 5), dtype=np.float32))
    assert first.element_count == second.element_count == 0
    assert first.content_digest != second.content_digest


def test_boolean_tensor_is_summarized_as_zero_one():
    summary = summarize_tensor_values(np.array([True, False, True]))
    assert (summary.minimum, summary.maximum) == (0, 1)
    assert summary.is_integral is True


def test_fractional_float_tensor_is_not_integral():
    summary = summarize_tensor_values(np.array([0.0, 0.5, 1.0], dtype=np.float32))
    assert summary.minimum == 0.0
    assert summary.maximum == 1.0
    assert summary.is_integral is False


def test_nan_is_excluded_from_range_and_not_integral():
    summary = summarize_tensor_values(np.array([1.0, np.nan, 3.0], dtype=np.float32))
    assert (summary.minimum, summary.maximum) == (1.0, 3.0)
    assert summary.is_integral is False
    # equal content still compares equal, which NaN in the range would break
    assert summary == summarize_tensor_values(np.array([1.0, np.nan, 3.0], dtype=np.float32))


def test_all_nan_tensor_reports_absent_range():
    summary = summarize_tensor_values(np.array([np.nan, np.nan], dtype=np.float32))
    assert summary.element_count == 2
    assert summary.minimum is None
    assert summary.maximum is None
    assert summary.is_integral is False


def test_infinity_is_an_observed_value_and_not_integral():
    summary = summarize_tensor_values(np.array([1.0, np.inf], dtype=np.float32))
    assert summary.minimum == 1.0
    assert summary.maximum == np.inf
    assert summary.is_integral is False


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
    for view in (bfloat16_view, float8_view, int4_view):
        with pytest.raises(UnsupportedTensorValueError):
            summarize_tensor_values(np.zeros(4, dtype=view))


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int64, np.uint64, np.float16, np.float32, np.float64, bool])
def test_supported_dtypes_are_summarized(dtype):
    assert summarize_tensor_values(np.zeros(2, dtype=dtype)).element_count == 2


def test_distinct_wider_floating_dtype_is_refused():
    dtype = np.dtype(np.longdouble)
    if dtype.itemsize <= np.dtype(np.float64).itemsize:
        pytest.skip("longdouble is not wider than float64 on this platform")

    assert dtype.kind == "f"
    with pytest.raises(UnsupportedTensorValueError, match=str(dtype)):
        summarize_tensor_values(np.array([np.finfo(dtype).max], dtype=dtype))


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


def test_analysis_pass_does_not_repeat_name_based_initializer_lookup(monkeypatch):
    model = make_model_with_initializers(
        {
            "p0": np.array([1.0, 2.0], dtype=np.float32),
            "p1": np.array([-1.0, 0.0], dtype=np.float32),
            "p2": np.array([3, 4], dtype=np.int8),
        }
    )

    def refuse_name_based_lookup(*_args, **_kwargs):
        raise AssertionError("bulk analysis must convert each encountered TensorProto directly")

    monkeypatch.setattr(model, "get_initializer", refuse_name_based_lookup)
    summaries = initializer_value_summaries(model)

    assert set(summaries) == {"p0", "p1", "p2"}
    assert summaries["p2"] == summarize_tensor_values(np.array([3, 4], dtype=np.int8))


def test_narrowness_question_is_derivable_from_the_summary():
    # the consumer-side question "does this tensor exclude its datatype
    # minimum?" is answered from the summary, not by rescanning the array
    declared = DataType["INT4"]
    narrow = summarize_tensor_values(np.array([-7.0, 7.0], dtype=np.float32))
    full = summarize_tensor_values(np.array([-8.0, 7.0], dtype=np.float32))
    assert narrow.minimum > declared.min()
    assert not (full.minimum > declared.min())


def test_summaries_are_comparable_without_a_model():
    # a composition-level consumer can compare and combine several summaries
    # on its own, with no model and nothing imported from a consumer framework
    first = summarize_tensor_values(np.array([-3.0, 2.0], dtype=np.float32))
    second = summarize_tensor_values(np.array([0.0, 9.0], dtype=np.float32))
    assert min(first.minimum, second.minimum) == -3.0
    assert max(first.maximum, second.maximum) == 9.0
    assert first.is_integral and second.is_integral
    assert first.contains_zero is False and second.contains_zero is True
    assert first.content_digest != second.content_digest


def test_a_combined_range_is_not_a_tensor_summary():
    # combining two tensors' ranges does not produce a third tensor, so it
    # must not be expressible as a TensorValueSummary with a fabricated
    # identity: the digest is the exact identity of one real tensor
    first = summarize_tensor_values(np.array([-3.0, 2.0], dtype=np.float32))
    second = summarize_tensor_values(np.array([0.0, 9.0], dtype=np.float32))
    with pytest.raises(ValueError, match="content_digest"):
        TensorValueSummary(
            content_digest="",
            element_count=first.element_count + second.element_count,
            minimum=min(first.minimum, second.minimum),
            maximum=max(first.maximum, second.maximum),
            is_integral=True,
            contains_zero=True,
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"content_digest": "abc"}, "content_digest"),
        ({"content_digest": "A" * 64}, "content_digest"),
        ({"element_count": -1}, "element_count"),
        ({"is_integral": 1}, "is_integral"),
        ({"contains_zero": 1}, "contains_zero"),
        ({"minimum": None}, "both present or both absent"),
        ({"minimum": 5.0, "maximum": 1.0}, "exceeds maximum"),
        ({"element_count": 0}, "empty tensor cannot have an observed range"),
        ({"minimum": 1.0, "contains_zero": True}, "zero lies outside"),
        ({"minimum": float("-inf"), "is_integral": True}, "non-finite range"),
        # zero is an observed value whenever it is an extremum
        ({"contains_zero": False}, "zero is an observed extremum"),
        ({"minimum": -3.0, "maximum": 0.0, "contains_zero": False}, "zero is an observed extremum"),
        # extrema are observed values, so integral values have integral extrema
        ({"minimum": 0.5, "maximum": 1.5, "contains_zero": False}, "not integral"),
        # NaN is excluded from the range by construction
        ({"minimum": float("nan"), "is_integral": False, "contains_zero": False}, "must not be NaN"),
        ({"maximum": float("nan"), "is_integral": False}, "must not be NaN"),
        # no observed range means empty (vacuously integral) or all-NaN (not)
        (
            {"minimum": None, "maximum": None, "contains_zero": False},
            "all-NaN, so is_integral must be False",
        ),
        (
            {"element_count": 0, "minimum": None, "maximum": None, "is_integral": False, "contains_zero": False},
            "vacuously integral",
        ),
    ],
)
def test_malformed_summaries_are_rejected(kwargs, message):
    # a summary is a claim about one real tensor; inconsistent claims are not
    # constructible values
    valid = dict(
        content_digest="0" * 64,
        element_count=2,
        minimum=0.0,
        maximum=2.0,
        is_integral=True,
        contains_zero=True,
    )
    with pytest.raises(ValueError, match=message):
        TensorValueSummary(**{**valid, **kwargs})


def test_well_formed_edge_case_summaries_are_constructible():
    # empty and all-NaN summaries must remain expressible
    TensorValueSummary(
        content_digest="0" * 64, element_count=0, minimum=None, maximum=None, is_integral=True, contains_zero=False
    )
    TensorValueSummary(
        content_digest="0" * 64, element_count=2, minimum=None, maximum=None, is_integral=False, contains_zero=False
    )
    # infinity is an observed value, and is_integral is False alongside it
    TensorValueSummary(
        content_digest="0" * 64,
        element_count=2,
        minimum=1.0,
        maximum=float("inf"),
        is_integral=False,
        contains_zero=False,
    )
    # zero strictly inside the range may or may not have been observed
    for observed in (True, False):
        TensorValueSummary(
            content_digest="0" * 64,
            element_count=2,
            minimum=-1.0,
            maximum=1.0,
            is_integral=True,
            contains_zero=observed,
        )


def test_every_produced_summary_satisfies_its_own_invariants():
    # the factory and the validator must agree: reconstructing each summary
    # from its own fields is the cheapest way to keep them from drifting
    arrays = [
        np.zeros((0, 3), dtype=np.float32),
        np.array([np.nan, np.nan], dtype=np.float32),
        np.array([1.0, np.inf], dtype=np.float32),
        np.array([0.0, 0.5], dtype=np.float32),
        np.array([-1.0, 1.0], dtype=np.float32),
        np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        np.array([0, 255], dtype=np.uint8),
        np.array([-3, 0, 7], dtype=np.int8),
        np.array([True, False]),
        np.array([-0.0], dtype=np.float32),
    ]
    for array in arrays:
        summary = summarize_tensor_values(array)
        assert dataclasses.replace(summary) == summary
