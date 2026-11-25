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

from pkgutil import get_data

import qonnx.core.data_layout as DataLayout
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper


def test_modelwrapper_has_tensor_datatype() -> None:
    """An annotated FLOAT32 and an unannotated tensor read the same through
    get_tensor_datatype; has_tensor_datatype tells them apart."""
    raw_m = get_data("qonnx.data", "onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)
    tname = model.graph.node[0].output[0]
    assert model.has_tensor_datatype(tname) is False
    assert model.get_tensor_datatype(tname) == DataType["FLOAT32"]
    model.set_tensor_layout(tname, DataLayout.NCHW)
    assert model.has_tensor_datatype(tname) is False
    model.set_tensor_datatype(tname, DataType["FLOAT32"])
    assert model.has_tensor_datatype(tname) is True
    assert model.get_tensor_datatype(tname) == DataType["FLOAT32"]


def test_modelwrapper_unsetting_a_datatype_removes_its_annotation() -> None:
    raw_m = get_data("qonnx.data", "onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)
    tname = model.graph.node[0].output[0]
    model.set_tensor_layout(tname, DataLayout.NCHW)
    model.set_tensor_datatype(tname, DataType["INT4"])
    model.set_tensor_datatype(tname, None)
    assert model.has_tensor_datatype(tname) is False
    assert model.get_tensor_datatype(tname) == DataType["FLOAT32"]
    assert model.get_tensor_layout(tname) == DataLayout.NCHW
    annotation = next(a for a in model.graph.quantization_annotation if a.tensor_name == tname)
    assert [e.key for e in annotation.quant_parameter_tensor_names] == ["tensor_layout"]
