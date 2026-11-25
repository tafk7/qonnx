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

# TODO deprecate in favor of the channels_last MaxPool

import copy
import numpy as np
from onnx import GraphProto, NodeProto, TensorProto, defs, helper, shape_inference
from typing import Any, Optional

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.base import CustomOp
from qonnx.util.basic import qonnx_make_model


def compute_pool_output_dim(ifm_dim, k, stride, pad=0, ceil_mode=0):
    "Return spatial output dimension size for pooling with given params."
    if ceil_mode:
        out_dim = int(np.ceil(((ifm_dim + 2 * pad - k) / stride) + 1))
        # per the ONNX spec (Operators.md, MaxPool since opset 22) and to match
        # PyTorch's pooling behavior, the last pooling window must not start
        # inside the right/bottom padding region. If the naive ceil-based
        # formula produces such a window, drop it.
        # See: https://github.com/onnx/onnx/pull/5741
        if (out_dim - 1) * stride >= ifm_dim + pad:
            out_dim -= 1
        return out_dim
    else:
        return int(np.floor(((ifm_dim + 2 * pad - k) / stride) + 1))


class MaxPoolNHWC(CustomOp):
    # a MaxPool node, but using the NHWC data layout

    def get_nodeattr_types(self) -> dict[str, tuple[Any, ...]]:
        # no specific attributes for MaxPoolNHWC
        # attributes below are identical to the standard ONNX MaxPool op:
        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool
        return {
            "kernel_shape": ("ints", True, []),
            "pads": ("ints", False, [0, 0, 0, 0]),
            "strides": ("ints", False, [1, 1]),
            "ceil_mode": ("i", False, 0),
            "auto_pad": ("s", False, "NOTSET"),
            "dilations": ("ints", False, [1, 1]),
        }

    def _output_shape(self, model: ModelWrapper) -> Optional[tuple[int, ...]]:
        """Infer full ONNX MaxPool geometry between NHWC and NCHW.

        Unknown extents or malformed attributes leave the output shape unstated.
        """
        info = model.get_tensor_valueinfo(self.onnx_node.input[0])
        if info is None or not info.type.tensor_type.HasField("shape"):
            return None
        dims = info.type.tensor_type.shape.dim
        if len(dims) != 4 or any(dim.WhichOneof("value") != "dim_value" for dim in dims):
            return None
        pads = next((attr for attr in self.onnx_node.attribute if attr.name == "pads"), None)
        if pads is not None and (pads.type != pads.INTS or len(pads.ints) != 4 or any(value < 0 for value in pads.ints)):
            return None
        for name in ("kernel_shape", "strides", "dilations"):
            pair = next((attr for attr in self.onnx_node.attribute if attr.name == name), None)
            if pair is None:
                if name == "kernel_shape":
                    return None
            elif pair.type != pair.INTS or len(pair.ints) != 2 or any(value < 1 for value in pair.ints):
                return None
        n, h, w, c = (dim.dim_value for dim in dims)
        node = copy.deepcopy(self.onnx_node)
        node.domain, node.op_type = "", "MaxPool"
        input_info = helper.make_tensor_value_info(
            node.input[0], info.type.tensor_type.elem_type or TensorProto.FLOAT, [n, c, h, w]
        )
        try:
            outputs = shape_inference.infer_node_outputs(
                defs.get_schema("MaxPool", 13), node, {node.input[0]: input_info.type}
            )
        except shape_inference.InferenceError:
            return None
        output = outputs.get(node.output[0])
        if output is None or not output.tensor_type.HasField("shape"):
            return None
        dimensions = output.tensor_type.shape.dim
        if len(dimensions) != 4 or any(dim.WhichOneof("value") != "dim_value" for dim in dimensions):
            return None
        result = [dimensions[index].dim_value for index in (0, 2, 3, 1)]
        # ONNX's ceil-mode rule also excludes a window starting entirely in
        # the right/bottom padding. The ONNX shape inferrer used here omits
        # this final guard; ONNX Runtime applies it, as the schema requires.
        attributes = {attr.name: attr for attr in node.attribute}
        ceil = attributes.get("ceil_mode")
        auto = attributes.get("auto_pad")
        mode = b"NOTSET" if auto is None else auto.s
        if ceil is not None and ceil.i == 1 and mode in (b"NOTSET", b"VALID"):
            stride_attr = attributes.get("strides")
            strides = (1, 1) if stride_attr is None else tuple(stride_attr.ints)
            starts = (0, 0, 0, 0) if pads is None or mode == b"VALID" else tuple(pads.ints)
            for axis, extent in ((1, h), (2, w)):
                if (result[axis] - 1) * strides[axis - 1] >= extent + starts[axis - 1]:
                    result[axis] -= 1
        return tuple(result)

    def make_shape_compatible_op(self, model: ModelWrapper) -> NodeProto:
        shape = self._output_shape(model)
        return copy.deepcopy(self.onnx_node) if shape is None else super().make_const_shape_op(shape)

    def infer_node_datatype(self, model):
        node = self.onnx_node
        # data type stays the same
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def execute_node(self, context: dict[str, np.ndarray], graph: GraphProto) -> None:
        node = copy.deepcopy(self.onnx_node)
        inp_name = node.input[0]
        out_name = node.output[0]
        inp = context[inp_name]
        dummy_out = context[out_name]
        # convert i/o NHWC -> NCHW
        inp = np.transpose(inp, (0, 3, 1, 2))
        dummy_out = np.transpose(dummy_out, (0, 3, 1, 2))
        # execute as regular MaxPool
        node.domain = ""
        node.op_type = "MaxPool"
        infos = (*graph.input, *graph.value_info, *graph.output)
        input_info = next((info for info in infos if info.name == inp_name), None)
        elem_type = input_info.type.tensor_type.elem_type if input_info is not None else TensorProto.FLOAT
        elem_type = elem_type or TensorProto.FLOAT
        inp_vi = helper.make_tensor_value_info(inp_name, elem_type, inp.shape)
        out_vi = helper.make_tensor_value_info(out_name, elem_type, dummy_out.shape)
        tmp_graph = helper.make_graph(nodes=[node], name="tmp_graph", inputs=[inp_vi], outputs=[out_vi])
        tmp_graph.quantization_annotation.extend(
            annotation for annotation in graph.quantization_annotation if annotation.tensor_name in (inp_name, out_name)
        )
        tmp_model = qonnx_make_model(tmp_graph, opset_imports=[helper.make_opsetid("", 13)])
        tmp_model = ModelWrapper(tmp_model)
        new_ctx = {inp_name: inp}
        from qonnx.core.onnx_exec import execute_onnx

        ret = execute_onnx(tmp_model, new_ctx)
        outp = ret[out_name]
        # convert output NCHW -> NHWC
        outp = np.transpose(outp, (0, 2, 3, 1))
        context[out_name] = outp

    def verify_node(self):
        info_messages = []
        return info_messages
