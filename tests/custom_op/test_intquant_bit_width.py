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


"""Quant reads its bit width as a scalar whether the initializer has shape () or (1,).
Exporters write both; numpy 2.4 and later refuse int() on a shape-(1,) array."""

import numpy as np
import pytest
from onnx import TensorProto, helper

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model


def quant_model(bit_width, signed):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node(
        "Quant", ["x", "scale", "zeropt", "bits"], ["y"], domain="qonnx.custom_op.general", signed=signed, narrow=0
    )
    model = ModelWrapper(qonnx_make_model(helper.make_graph([node], "quant", [x], [y])))
    model.set_initializer("scale", np.array(1.0, dtype=np.float32))
    model.set_initializer("zeropt", np.array(0.0, dtype=np.float32))
    model.set_initializer("bits", bit_width)
    return model


@pytest.mark.parametrize("shape", [(), (1,)])
@pytest.mark.parametrize("bits, signed, expected", [(4, 1, "INT4"), (4, 0, "UINT4"), (1, 1, "BIPOLAR"), (1, 0, "BINARY")])
def test_quant_bit_width_scalar_or_one_element(shape, bits, signed, expected):
    model = quant_model(np.full(shape, bits, dtype=np.float32), signed)
    op = getCustomOp(model.graph.node[0])
    assert op.get_integer_datatype(model) == DataType[expected]
    assert op.get_scaled_integer_datatype(model) == DataType["SCALEDINT<%d>" % bits]
