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

"""A Quant extension's malformed datatype is not an assertion fallback."""

from __future__ import annotations

import pytest

import uuid
from onnx import TensorProto, helper

from qonnx.core.datatype import BaseDataType, DataType, DataTypeNameMalformed
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.intquant import IntQuant
from qonnx.custom_op.registry import add_op_to_domain
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import qonnx_make_model


class MalformedFixedQuant(IntQuant):
    """A Quant subclass that resolves a fixed-point output datatype."""

    def get_output_dtype(self, model: ModelWrapper) -> BaseDataType:
        return DataType["FIXED<8,8>"]


class OwnAssertQuant(IntQuant):
    """The op's own assertion still selects Quant's FLOAT32 fallback."""

    def get_output_dtype(self, model: ModelWrapper) -> BaseDataType:
        raise AssertionError("the Quant extension's own unsupported output")


def model_for(cls: type[IntQuant]) -> ModelWrapper:
    domain = "qonnx_test_quant_" + uuid.uuid4().hex
    add_op_to_domain(domain, cls)
    graph = helper.make_graph(
        [helper.make_node(cls.__name__, ["x", "s", "z", "b"], ["y"], domain=domain)],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        initializer=[
            helper.make_tensor(name, TensorProto.FLOAT, [], [value]) for name, value in [("s", 1), ("z", 0), ("b", 8)]
        ],
    )
    m = qonnx_make_model(graph)
    m.opset_import.append(helper.make_opsetid(domain, 1))
    return ModelWrapper(m)


def test_a_quant_subclasss_malformed_fixed_name_propagates_as_a_value_refusal() -> None:
    m = model_for(MalformedFixedQuant)
    with pytest.raises(DataTypeNameMalformed) as caught:
        m.transform(InferDataTypes())
    assert isinstance(caught.value, ValueError) and not isinstance(caught.value, AssertionError)
    assert caught.value.findings[0].code == "name-malformed"


def test_quants_own_assertions_keep_the_float32_fallback() -> None:
    m = model_for(OwnAssertQuant).transform(InferDataTypes())
    assert m.get_tensor_datatype("y") == DataType["FLOAT32"]
