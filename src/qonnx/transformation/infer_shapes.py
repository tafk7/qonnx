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

import onnx.shape_inference as si
from onnx import NodeProto, TensorProto, TypeProto, defs, helper

from qonnx.core.modelwrapper import ModelWrapper, TensorFactRefused, _custom_op_for
from qonnx.custom_op.registry import CustomOpUnsupported, is_custom_op, retired
from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_by_name, get_preferred_onnx_opset


def _make_shape_compatible_op(node: NodeProto, model: ModelWrapper) -> NodeProto:
    """Return a shape-compatible non-QONNX op for a given QONNX op. Used for
    shape inference with custom ops."""
    assert is_custom_op(node.domain), "Node domain is not a registered custom op domain"
    op_type = node.op_type
    try:
        # the op at the version the model imports for its domain, given the model
        # where it wants it
        inst = _custom_op_for(model, node)
    except KeyError as error:
        if retired(node.domain, op_type) is not None:
            # a name the domain's provider retired: its refusal, with the reason
            raise
        # exception if op_type is not supported
        raise CustomOpUnsupported.of(
            "op-unsupported", "InferShapes", "Custom op_type %s is currently not supported." % op_type
        ) from error
    return inst.make_shape_compatible_op(model)


def _node_output_shapes(
    model: ModelWrapper, node: NodeProto, *, owner: str = "CustomOp.infer_node_facts"
) -> dict[str, tuple[list[int], int]]:
    """The outputs of the standard ONNX node ``node`` whose shapes ONNX's inference of
    that one node knows fully, each with its shape and element type (0 where ONNX
    states none). The node's inputs are read from ``model``: an initializer with its
    value (a Reshape's target shape), any other input from its value info. It is
    inferred at the version the model imports for the node's domain. ValueError for
    an input the model states no shape of; empty for a node ONNX has no schema of."""
    imports = model.get_opset_imports()
    if node.domain in ("", "ai.onnx"):
        version = imports.get("", imports.get("ai.onnx", get_preferred_onnx_opset()))
    else:
        version = imports.get(node.domain, 1)
    try:
        schema = defs.get_schema(node.op_type, version, "" if node.domain == "ai.onnx" else node.domain)
    except defs.SchemaError:
        return {}
    types: dict[str, TypeProto] = {}
    values: dict[str, TensorProto] = {}
    for name in node.input:
        if not name:
            continue
        initializer = get_by_name(model.graph.initializer, name)
        if initializer is not None:
            values[name] = initializer
            types[name] = helper.make_tensor_type_proto(initializer.data_type, list(initializer.dims))
            continue
        info = model.get_tensor_valueinfo(name)
        if info is None or not (info.type.HasField("tensor_type") and info.type.tensor_type.HasField("shape")):
            raise TensorFactRefused.of(
                "fact-invalid",
                owner,
                f"{node.name or node.op_type}: {name} has no shape yet",
                node=node.name or node.op_type,
                tensor=name,
                fact="shape",
            )
        types[name] = info.type
    inferred = si.infer_node_outputs(schema, node, types, values, opset_imports=list(model.model.opset_import))
    shapes = {}
    for name in node.output:
        found = inferred.get(name)
        if not name or found is None or not found.HasField("tensor_type"):
            continue
        tensor = found.tensor_type
        if not tensor.HasField("shape") or not all(dim.HasField("dim_value") for dim in tensor.shape.dim):
            continue
        shapes[name] = ([dim.dim_value for dim in tensor.shape.dim], tensor.elem_type)
    return shapes


def _hide_finn_ops(model):
    """Replace any QONNX ops by shape-compatible ones, and return a dict that
    can be used to map the string representations of the new (shape-compatible)
    ops back to the old ops."""
    hidden_ops = {}
    node_ind = 0
    for node in model.graph.node:
        node_ind += 1
        if is_custom_op(node.domain):
            new_node = _make_shape_compatible_op(node, model)
            # keep old node name to help debug shape inference issues
            new_node.name = node.name
            hidden_ops[str(new_node)] = node
            model.graph.node.insert(node_ind, new_node)
            model.graph.node.remove(node)
    return hidden_ops


def _restore_finn_ops(model, hidden_ops):
    """Replace any shape-compatible ops with the QONNX ops that originally
    generated them."""
    node_ind = 0
    for node in model.graph.node:
        node_ind += 1
        try:
            old_node = hidden_ops[str(node)]
            model.graph.node.insert(node_ind, old_node)
            model.graph.node.remove(node)
        except KeyError:
            pass


class InferShapes(Transformation):
    """Ensure every tensor in the model has a specified shape (ValueInfo)."""

    def apply(self, model):
        # hide your riches!
        hidden_ops = _hide_finn_ops(model)
        # call regular ONNX shape inference
        model = ModelWrapper(si.infer_shapes(model.model))
        # bring back hidden ops
        _restore_finn_ops(model, hidden_ops)
        return (model, False)
