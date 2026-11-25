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

import pytest

import numpy as np
import os
from pkgutil import get_data

import qonnx.core.data_layout as DataLayout
from qonnx.core.modelwrapper import ModelWrapper


def test_modelwrapper_annotation_round_trip_and_literals_only() -> None:
    """Layout and sparsity annotations round-trip as Python literals (including an
    empty set), and reading one never evaluates code stored in the model."""
    raw_m = get_data("qonnx.data", "onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)
    tname = model.graph.input[0].name
    for layout in [DataLayout.NCHW, DataLayout.NHWC, DataLayout.NC, DataLayout.UNKNOWN]:
        model.set_tensor_layout(tname, layout)
        assert model.get_tensor_layout(tname) == layout
    sparsity = {1: {0, 2, 3}, 0: set(), "dw": {"kernel_shape": [3, 3]}}
    model.set_tensor_sparsity(tname, sparsity)
    assert model.get_tensor_sparsity(tname) == sparsity

    probe = "QONNX_ANNOTATION_EVAL_PROBE"
    os.environ.pop(probe, None)
    code = f"[__import__('os').environ.setdefault('{probe}', 'ran')]"
    for key, getter in [("tensor_layout", model.get_tensor_layout), ("tensor_sparsity", model.get_tensor_sparsity)]:
        annotation = next(a for a in model.graph.quantization_annotation if a.tensor_name == tname)
        entry = next(e for e in annotation.quant_parameter_tensor_names if e.key == key)
        entry.value = code
        with pytest.raises(ValueError, match=f"{key} annotation of tensor {tname}"):
            getter(tname)
        assert probe not in os.environ


@pytest.mark.parametrize("text", ["[1,", "not_a_literal", "{**{1: 2}}"])
@pytest.mark.parametrize("key", ["tensor_layout", "tensor_sparsity"])
def test_invalid_annotation_names_the_tensor_and_key(text: str, key: str) -> None:
    raw_m = get_data("qonnx.data", "onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)
    name = model.graph.input[0].name
    model.set_tensor_layout(name, [])
    model.set_tensor_sparsity(name, {})
    annotation = next(a for a in model.graph.quantization_annotation if a.tensor_name == name)
    entry = next(e for e in annotation.quant_parameter_tensor_names if e.key == key)
    entry.value = text
    getter = model.get_tensor_layout if key == "tensor_layout" else model.get_tensor_sparsity
    with pytest.raises(ValueError, match=f"{key} annotation of tensor {name}"):
        getter(name)


@pytest.mark.parametrize("branch", ["new_tensor", "new_entry", "existing_entry"])
def test_numpy_nonzero_mask_round_trip_stores_plain_python_numbers(branch: str) -> None:
    model = ModelWrapper(get_data("qonnx.data", "onnx/mnist-conv/model.onnx"))
    name = model.graph.input[0].name
    if branch == "new_entry":
        model.set_tensor_layout(name, DataLayout.NCHW)
    elif branch == "existing_entry":
        model.set_tensor_sparsity(name, {})
    indices = np.nonzero(np.asarray([1, 0, 1]))[0]
    mask = {np.int64(1): set(indices)}
    model.set_tensor_sparsity(name, mask)
    annotation = next(a for a in model.graph.quantization_annotation if a.tensor_name == name)
    entry = next(e for e in annotation.quant_parameter_tensor_names if e.key == "tensor_sparsity")
    assert entry.value == "{1: {0, 2}}"
    assert model.get_tensor_sparsity(name) == {1: {0, 2}}
    assert all(isinstance(index, np.integer) for index in mask[np.int64(1)])


def test_sparsity_writer_normalizes_nested_float_and_bool_scalars() -> None:
    model = ModelWrapper(get_data("qonnx.data", "onnx/mnist-conv/model.onnx"))
    name = model.graph.input[0].name
    value = {"counts": [np.int32(3), np.float64(0.5)], "flags": (np.bool_(True), np.longdouble(0.5))}
    model.set_tensor_sparsity(name, value)
    annotation = next(a for a in model.graph.quantization_annotation if a.tensor_name == name)
    entry = next(e for e in annotation.quant_parameter_tensor_names if e.key == "tensor_sparsity")
    assert entry.value == "{'counts': [3, 0.5], 'flags': (True, 0.5)}"
    assert model.get_tensor_sparsity(name) == {"counts": [3, 0.5], "flags": (True, 0.5)}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("np.int8(-3)", -3),
        ("np.int16(3)", 3),
        ("np.int32(3)", 3),
        ("np.int64(3)", 3),
        ("np.uint8(3)", 3),
        ("np.uint16(3)", 3),
        ("np.uint32(3)", 3),
        ("np.uint64(3)", 3),
        ("np.float16(0.5)", 0.5),
        ("np.float32(0.5)", 0.5),
        ("np.float64(0.5)", 0.5),
        ("np.bool_(True)", True),
        ("np.True_", True),
        ("np.False_", False),
        ("np.longdouble('0.5')", 0.5),
        ("np.float128('-1.25e2')", -125.0),
        ("np.float96('+1.25E-2')", 0.0125),
        ("np.longfloat('.5')", 0.5),
        ("np.longdouble(0.5)", 0.5),
    ],
)
def test_stored_upstream_numpy_scalar_sparsity_reads_as_plain_numbers(text: str, expected: int | float | bool) -> None:
    model = ModelWrapper(get_data("qonnx.data", "onnx/mnist-conv/model.onnx"))
    name = model.graph.input[0].name
    model.set_tensor_sparsity(name, {})
    annotation = next(a for a in model.graph.quantization_annotation if a.tensor_name == name)
    entry = next(e for e in annotation.quant_parameter_tensor_names if e.key == "tensor_sparsity")
    entry.value = "{1: {" + text + "}}"
    result = next(iter(model.get_tensor_sparsity(name)[1]))
    assert type(result) is type(expected)
    assert result == expected


@pytest.mark.parametrize(
    "text",
    [
        "np.array([3])",
        "other.int64(3)",
        "np.int64(1 + 2)",
        "np.int64(np.int64(3))",
        "np.int64(value=3)",
        "np.int64(3, 4)",
        "np.int64('3')",
        "np.float64('0.5')",
        "np.int64([3])",
        "np.int64",
        "np.pi",
        "np.True_.item()",
        "np.float64(np.inf)",
        "np.float64(np.nan)",
        "np.float64(1e999)",
        "np.longdouble('inf')",
        "np.longdouble('nan')",
        "np.longdouble('1e999')",
        "np.longdouble('0x1p0')",
        "np.longdouble('__import__(\"os\")')",
    ],
)
def test_numpy_sparsity_reader_refuses_other_expressions(text: str) -> None:
    model = ModelWrapper(get_data("qonnx.data", "onnx/mnist-conv/model.onnx"))
    name = model.graph.input[0].name
    model.set_tensor_sparsity(name, {})
    annotation = next(a for a in model.graph.quantization_annotation if a.tensor_name == name)
    entry = next(e for e in annotation.quant_parameter_tensor_names if e.key == "tensor_sparsity")
    entry.value = "{1: {" + text + "}}"
    with pytest.raises(ValueError, match=f"tensor_sparsity annotation of tensor {name}"):
        model.get_tensor_sparsity(name)


def test_numpy_looking_text_inside_a_literal_string_is_unchanged() -> None:
    model = ModelWrapper(get_data("qonnx.data", "onnx/mnist-conv/model.onnx"))
    name = model.graph.input[0].name
    model.set_tensor_layout(name, ["np.int64(3)", "np.True_", "np.longdouble('0.5')"])
    assert model.get_tensor_layout(name) == ["np.int64(3)", "np.True_", "np.longdouble('0.5')"]


@pytest.mark.parametrize("key", ["tensor_layout", "tensor_sparsity"])
def test_literal_annotation_preserves_leading_whitespace(key: str) -> None:
    model = ModelWrapper(get_data("qonnx.data", "onnx/mnist-conv/model.onnx"))
    name = model.graph.input[0].name
    model.set_tensor_layout(name, [])
    model.set_tensor_sparsity(name, {})
    annotation = next(a for a in model.graph.quantization_annotation if a.tensor_name == name)
    entry = next(e for e in annotation.quant_parameter_tensor_names if e.key == key)
    if key == "tensor_layout":
        entry.value = ' \t["N", "C"] '
        assert model.get_tensor_layout(name) == ["N", "C"]
    else:
        entry.value = " \t{1: {0, 2}} "
        assert model.get_tensor_sparsity(name) == {1: {0, 2}}
