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

import numpy as np
from onnx import NodeProto
from typing import TYPE_CHECKING, Any, Optional

import qonnx.core.onnx_exec as oxe
from qonnx.custom_op.registry import is_custom_op
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_shapes import InferShapes

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper


def _folded(model: ModelWrapper, node: NodeProto, context: dict[str, Any], opset_version: int) -> Optional[np.ndarray]:
    """The value constant folding stores for node's first output, or None where the node
    is left unfolded.

    A standard node is evaluated with exact integers on, whatever the caller's setting:
    where its float tensors are all integer-annotated and the values given to it are
    integers, it computes by its operator and those values (onnx_exec.execute_node):
    integer arithmetic with a bound and an operator that moves values compute a result
    shown exact or refuse; any other operator has no proof, computes as without exact
    integers within what its container holds exactly, and refuses past it. Where exact
    integers refuse the node (ArithmeticError: a
    bound past 2**53, a value its type does not keep; NotImplementedError: an operator
    with no proof given or computing a value past what its container holds exactly, no
    float64 kernel), its exactness is unproved and it is left unfolded. Its value is
    stored in the element type the graph states for the output, where that type holds
    it exactly; where it does not, the node is left unfolded, for an initializer would
    round it. A custom node is evaluated with the caller's setting, and its value
    stored as computed."""
    name = node.output[0]
    if is_custom_op(node.domain, node.op_type):
        oxe.execute_node(node, context, model.graph, opset_version=opset_version, model=model)
        return context[name]
    try:
        oxe.execute_node(node, context, model.graph, opset_version=opset_version, model=model, exact_integers=True)
    except (ArithmeticError, NotImplementedError):
        return None
    value = context[name]
    info = model.get_tensor_valueinfo(name)
    if info is None:
        return value
    stated = oxe._np_type(info.type.tensor_type.elem_type)
    if stated is None or value.dtype == stated:
        return value
    held = value.astype(stated)
    if not np.array_equal(held, value):
        return None
    return held


class FoldConstantsFiltered(Transformation):
    """Replace the output of a node with const-only inputs with a precomputed
    result. Use the match_filter_fxn(model, node) function to decide which nodes
    can be eligible for const folding.

    A node is folded only where exact integers do not refuse it and its output's stated
    type holds its value; any other node is left as it is (see FoldConstants)."""

    def __init__(self, match_filter_fxn):
        super().__init__()
        self.match_filter_fxn = match_filter_fxn

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        opset_version = model.model.opset_import[0].version
        graph = model.graph
        node_ind = 0
        graph_modified = False
        execution_context = model.make_empty_exec_context()
        nodes_to_remove = []
        for n in graph.node:
            node_ind += 1
            node_inp_inits = list(map(lambda x: model.get_initializer(x), n.input))
            node_inp_dyn = list(filter(lambda x: x is None, node_inp_inits))
            node_out = n.output[0]
            is_all_constant_inputs = len(node_inp_dyn) == 0
            ishape = model.get_tensor_shape(n.input[0])
            is_const_shape = (n.op_type == "Shape") and (ishape is not None)
            eligible = self.match_filter_fxn(model, n)
            if (is_all_constant_inputs or is_const_shape) and eligible:
                # this node has no dynamic inputs, only constant ones -- so we can
                # do constant folding.
                folded = _folded(model, n, execution_context, opset_version)
                if folded is None:
                    continue
                # use the execution result as an initializer
                model.set_initializer(node_out, folded)
                # remove old node
                nodes_to_remove.append(n)
                graph_modified = True
        for node in nodes_to_remove:
            model.graph.node.remove(node)
        if graph_modified:
            model = model.transform(InferShapes())
        return (model, graph_modified)


class FoldConstants(Transformation):
    """Replace the output of a node with const-only inputs with a precomputed
    result. Skip any op types given in exclude_op_types.

    A standard node is evaluated with exact integers on (qonnx.core.onnx_exec's
    exact_integers), whatever the caller's setting: one whose float tensors are all
    integer-annotated, given integers, computes by its operator and the values given:
    integer arithmetic by a bound on its products and partial sums (in its own
    container or float64), and an operator that moves values, compute a result shown
    exact or refuse; any other operator has no proof and computes as without exact
    integers within what its container holds exactly, its result not shown exact (an
    Einsum whose products pass 2**24 and cancel rounds as before). A node given a value
    that is not an integer folds as before. It is left unfolded where exact integers refuse it (a
    bound past 2**53, a Cast that does not keep every value, an operator with no proof
    given or computing a value past what its container holds exactly, no float64
    kernel for its op), and where its output's stated type does not hold the value
    exactly; folding goes on past it. An operator whose attributes fix an element type
    (Cast, ConstantOfShape and the like) is never widened, and folds as it does
    without exact integers. A custom node is evaluated with the caller's setting, its
    value stored as computed."""

    def __init__(self, exclude_op_types=["Quant", "BipolarQuant"]):
        super().__init__()
        self.exclude_op_types = exclude_op_types

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        opset_version = model.model.opset_import[0].version
        node_ind = 0
        graph_modified = False
        execution_context = model.make_empty_exec_context()
        nodes_to_remove = []
        for n in model.graph.node:
            node_ind += 1
            node_inp_inits = list(map(lambda x: model.get_initializer(x), n.input))
            node_inp_dyn = list(filter(lambda x: x is None, node_inp_inits))
            node_out = n.output[0]
            is_all_constant_inputs = len(node_inp_dyn) == 0
            if len(n.input) > 0:
                ishape = model.get_tensor_shape(n.input[0])
                is_const_shape = (n.op_type == "Shape") and (ishape is not None)
                is_no_input = False
            else:
                is_no_input = True
            exclude = n.op_type in self.exclude_op_types
            if (is_all_constant_inputs or is_const_shape or is_no_input) and not exclude:
                # this node has no (dynamic) inputs, only constant ones -- so we can
                # do constant folding. to ensure any missing ValueInfos from initializers
                # are populated, we 'touch' the shape of all inputs first below.
                for inp in n.input:
                    model.get_tensor_shape(inp, fix_missing_init_shape=True)
                folded = _folded(model, n, execution_context, opset_version)
                if folded is None:
                    continue
                # use the execution result as an initializer
                model.set_initializer(node_out, folded)
                # remove old node
                nodes_to_remove.append(n)
                graph_modified = True
                # Exit the loop here, after changing a single node. The
                # ModelWrapper ensures to repeat this transformatin as long as
                # there are suitable nodes available.
                # See https://github.com/fastmachinelearning/qonnx/issues/104
                break
        for node in nodes_to_remove:
            model.graph.node.remove(node)
        if graph_modified:
            model = model.transform(InferShapes())
        return (model, graph_modified)
