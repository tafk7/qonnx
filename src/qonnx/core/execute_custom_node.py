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

from onnx import GraphProto, NodeProto
from typing import TYPE_CHECKING, Any

import qonnx.custom_op.registry as registry
from qonnx.core.modelwrapper import _custom_op_for

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper


def execute_custom_node(
    node: NodeProto,
    context: dict[str, Any],
    graph: GraphProto,
    onnx_opset_version: int,
    *,
    model: ModelWrapper | None = None,
    exact_integers: bool | None = None,
) -> None:
    """Call custom implementation to execute a single custom node.
    Input/output provided via context.

    model is the ModelWrapper the node belongs to, where the caller has it. Given it,
    the custom op is built at the version the model imports for the node's domain,
    else at the preferred qonnx opset, and onnx_opset_version is not read; an op that
    sets wants_model is given the model. Without it, the op is built at
    onnx_opset_version. exact_integers switches exact integers for
    this node (see qonnx.core.onnx_exec.exact_integers); None, the default, keeps the
    setting around the call."""
    if exact_integers is not None:
        import qonnx.core.onnx_exec as onnx_exec

        with onnx_exec.exact_integers(exact_integers):
            return execute_custom_node(node, context, graph, onnx_opset_version, model=model)
    op_type = node.op_type
    try:
        if model is None:
            # lookup op_type in registry of CustomOps
            inst = registry.getCustomOp(node, onnx_opset_version=onnx_opset_version)
        else:
            inst = _custom_op_for(model, node)
        inst.execute_node(context, graph)
    except KeyError:
        # exception if op_type is not supported
        raise Exception("Custom op_type %s is currently not supported." % op_type)
