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

"""Throwaway custom-op domains for the registry and model-aware tests.

These tests register their ops into a domain of their own (make_domain), not
into a real one such as qonnx.custom_op.general: the registry is process-wide,
so an op registered into a real domain stays visible to every later test in the
worker."""

import pytest

import sys
import types
import uuid
from onnx import helper

from qonnx.custom_op.base import CustomOp


class CopyOp(CustomOp):
    """The test op: copies its input to its output, with one integer attribute
    (my_attr). Every model it is attached to is appended to its class's
    attached_to; make_op gives each op class a list of its own."""

    attached_to: list = []

    def get_nodeattr_types(self):
        return {"my_attr": ("i", False, 0)}

    def attach_model(self, model):
        type(self).attached_to.append(model)
        return super().attach_model(model)

    def make_shape_compatible_op(self, model):
        return helper.make_node("Identity", [self.onnx_node.input[0]], [self.onnx_node.output[0]])

    def infer_node_datatype(self, model):
        node = self.onnx_node
        model.set_tensor_datatype(node.output[0], model.get_tensor_datatype(node.input[0]))

    def execute_node(self, context, graph):
        if self.wants_model:
            assert self._model.graph is graph
        context[self.onnx_node.output[0]] = context[self.onnx_node.input[0]]

    def verify_node(self):
        pass


@pytest.fixture
def make_op():
    """make_op(name, **body): a new CopyOp class of that name, the body's
    attributes (op_type, op_version, wants_model...) stated in its own body."""

    def make(name, **body):
        return type(name, (CopyOp,), {"attached_to": [], **body})

    return make


@pytest.fixture
def make_domain(monkeypatch):
    """make_domain(**members): the name of a fresh importable domain module
    exporting the given members (an ``opset_version`` member is the module's
    stated version, not an export). The module leaves sys.modules after the
    test; its uniquely named registry entries are never looked up again."""

    def make(**members):
        name = "qonnx_test_domain_" + uuid.uuid4().hex[:8]
        module = types.ModuleType(name)
        for key, value in members.items():
            setattr(module, key, value)
        module.__all__ = [key for key in members if key != "opset_version"]
        monkeypatch.setitem(sys.modules, name, module)
        return name

    return make
