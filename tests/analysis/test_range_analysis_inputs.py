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

import onnx.parser as oprs
from pathlib import Path

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.range_analysis import range_analysis


def _two_input_add() -> ModelWrapper:
    model = ModelWrapper(
        oprs.parse_model(
            """
            <ir_version: 8, opset_import: ["" : 13]>
            g (float[1,4] a, float[1,4] b) => (float[1,4] s)
            {
              s = Add(a, b)
            }
            """
        )
    )
    model.set_tensor_datatype("a", DataType["INT8"])
    model.set_tensor_datatype("b", DataType["UINT2"])
    return model


def test_range_analysis_each_input_takes_its_own_datatype_range() -> None:
    """Without irange, every graph input starts from its own datatype's range,
    not the first input's."""
    ret = range_analysis(_two_input_add(), report_mode="range")
    assert ret["a"] == (-128, 127)
    assert ret["b"] == (0, 3)
    assert ret["s"] == (-128, 130)


def test_range_analysis_irange_applies_to_every_input() -> None:
    ret = range_analysis(_two_input_add(), irange=(-1.0, 1.0), report_mode="range")
    assert ret["a"] == (-1.0, 1.0)
    assert ret["b"] == (-1.0, 1.0)
    assert ret["s"] == (-2.0, 2.0)


def test_command_line_accepts_filename_and_irange_as_text(tmp_path: Path) -> None:
    from clize import Clize

    filename = tmp_path / "add.onnx"
    _two_input_add().save(str(filename))
    report = Clize(range_analysis)("range-analysis", str(filename), "--irange=(-1.0,1.0)", "--report-mode=range")
    assert report["a"] == (-1.0, 1.0)
    assert report["b"] == (-1.0, 1.0)
    assert report["s"] == (-2.0, 2.0)
