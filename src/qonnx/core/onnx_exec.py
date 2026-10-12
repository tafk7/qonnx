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

import copy
import numpy as np
import onnx.helper as helper
import onnxruntime as rt
import warnings
from contextvars import ContextVar
from onnx import AttributeProto, NodeProto, TensorProto, ValueInfoProto
from onnxruntime.capi.onnxruntime_pybind11_state import NotImplemented as RuntimeNotImplemented
from typing import TYPE_CHECKING, Literal, Mapping, overload

import qonnx.analysis.topology as ta
import qonnx.core.execute_custom_node as ex_cu_node
from qonnx.custom_op.registry import is_custom_op
from qonnx.util.basic import get_preferred_qonnx_opset, get_sanitize_quant_tensors, qonnx_make_model, sanitize_quant_values

if TYPE_CHECKING:
    from qonnx.core.datatype import BaseDataType
    from qonnx.core.modelwrapper import ModelWrapper

# Whether the custom op execute_node is running was asked for exact integers: a custom
# op that executes a model of its own reads it (exact_integers_running).
_exact_integers: ContextVar[bool] = ContextVar("qonnx.core.onnx_exec.exact_integers", default=False)


def exact_integers_running() -> bool:
    """Whether the custom op being executed was asked for exact integers (execute_node's
    exact_integers). A custom op that executes a model of its own passes it on."""
    return _exact_integers.get()


def _np_type(elem_type: int) -> np.dtype | None:
    """The numpy type of an ONNX element type, or None where numpy has none."""
    try:
        return np.dtype(helper.tensor_dtype_to_np_dtype(elem_type))
    except (KeyError, TypeError, ValueError):
        return None


def _exact_limit(elem_type: int) -> int | None:
    """The largest magnitude up to which a float element type holds every integer
    exactly, 2**(mantissa bits + 1); None for an element type that is not a float."""
    np_type = _np_type(elem_type)
    if np_type is None or not np.issubdtype(np_type, np.floating):
        return None
    return 2 ** (np.finfo(np_type).nmant + 1)


def _largest(datatype: BaseDataType) -> int:
    return max(abs(int(datatype.min())), abs(int(datatype.max())))


def _widened(node: NodeProto, infos: list[ValueInfoProto], model: ModelWrapper) -> set[str]:
    """The tensors of a standard node that exact integers make it compute in float64:
    its float tensors, where every one of them is integer-annotated and one annotation
    passes what its container holds exactly. Empty where the node computes in the
    graph's own element types.

    A node that would widen raises where an annotation passes float64's exact integers
    (2**53), and where it carries a subgraph (its subgraph's types are not restated)."""
    held = {info.name: info.type.tensor_type.elem_type for info in infos}
    names = dict.fromkeys(name for name in (*node.input, *node.output) if name in held)
    limits = {name: _exact_limit(held[name]) for name in names}
    floats = [name for name in names if limits[name] is not None]
    annotations = {name: model.get_tensor_datatype(name) for name in floats}
    if not floats or not all(datatype.is_integer() for datatype in annotations.values()):
        return set()
    if all(_largest(annotations[name]) <= limits[name] for name in floats):
        return set()
    double = _exact_limit(TensorProto.DOUBLE)
    for name in floats:
        if _largest(annotations[name]) > double:
            raise ArithmeticError(
                f"{node.name} ({node.op_type}): {name} is annotated {annotations[name].name}, "
                "past 2**53, beyond which float64 does not hold every integer exactly"
            )
    if any(attr.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS) for attr in node.attribute):
        raise NotImplementedError(
            f"{node.name} ({node.op_type}): a node with a subgraph cannot compute in float64 for exact integers"
        )
    return set(floats)


def _as_double(info: ValueInfoProto) -> ValueInfoProto:
    double = ValueInfoProto()
    double.CopyFrom(info)
    double.type.tensor_type.elem_type = TensorProto.DOUBLE
    return double


def execute_node(
    node,
    context,
    graph,
    opset_version,
    return_full_exec_context=False,
    model=None,
    exact_integers=False,
):
    """Executes one node, its inputs read from context and its outputs written there:
    a custom op by its own execute_node, a standard op as an ONNX Runtime session of a
    model holding the node alone.

    exact_integers: integers are computed exactly where the graph's float containers
    would round them. A standard node whose float tensors (those the graph holds in a
    float element type) are all integer-annotated, one of the annotations past what its
    container holds exactly (float32: 2**24), computes in float64: its tensors are
    stated DOUBLE and its inputs given as float64, and its outputs stay float64. Every
    other standard node computes in the graph's element types, its inputs cast to them.
    A node that would widen raises where an annotation passes 2**53, or where ONNX
    Runtime has no float64 kernel for its op. The annotations are read from model,
    which the option needs. A custom op computes in whatever it is given; one that
    executes a model of its own reads the option with exact_integers_running."""

    if is_custom_op(node.domain, node.op_type):
        token = _exact_integers.set(exact_integers)
        try:
            ex_cu_node.execute_custom_node(
                node,
                context,
                graph,
                onnx_opset_version=opset_version,
                model=model,
            )
        finally:
            _exact_integers.reset(token)
    else:
        # ONNX Runtime executes whole models only, so the node runs as a model of its
        # own. Its inputs and outputs take their value infos from the graph: a tensor
        # may be a graph input or output too, and must have one value info only. The
        # outputs may come out of another order than the node's, which the loop below
        # undoes by name. A subgraph's reads of the outer graph become inputs too.
        node_inputs = list(filter(lambda x: x.name in node.input, graph.input))
        node_inputs += list(filter(lambda x: x.name in node.input, graph.output))
        node_inputs += list(filter(lambda x: x.name in node.input, graph.value_info))
        node_outputs = list(filter(lambda x: x.name in node.output, graph.output))
        node_outputs += list(filter(lambda x: x.name in node.output, graph.value_info))
        for attr in node.attribute:
            if attr.type == 5:
                subgraph = attr.g
                for subgraph_node in subgraph.node:
                    subgraph_node_inputs = list(filter(lambda x: x.name in subgraph_node.input, graph.value_info))
                    new_inps = list(filter(lambda x: x not in node_inputs, subgraph_node_inputs))
                    node_inputs += new_inps
        widened = set()
        if exact_integers:
            if model is None:
                raise ValueError(f"{node.name} ({node.op_type}): exact_integers reads the model's annotations; pass model")
            widened = _widened(node, node_inputs + node_outputs, model)
            node_inputs = [_as_double(x) if x.name in widened else x for x in node_inputs]
            node_outputs = [_as_double(x) if x.name in widened else x for x in node_outputs]
        node_graph = helper.make_graph(
            nodes=[node],
            name="single-node-exec",
            inputs=node_inputs,
            outputs=node_outputs,
        )
        node_model = qonnx_make_model(node_graph)
        node_model.opset_import[0].version = opset_version
        # With exact_integers, each input is given in the element type the node
        # reads it in.
        read_as = {x.name: _np_type(x.type.tensor_type.elem_type) for x in node_inputs} if exact_integers else {}
        input_dict = dict()
        for inp in node.input:
            value = context[inp]
            if value is not None and read_as.get(inp) is not None:
                value = np.asarray(value).astype(read_as[inp], copy=False)
            input_dict[inp] = value

        try:
            sess = rt.InferenceSession(node_model.SerializeToString())
        except RuntimeNotImplemented as error:
            if not widened:
                raise
            raise NotImplementedError(
                f"{node.name} ({node.op_type}): ONNX Runtime has no float64 kernel for {node.op_type}, "
                "which this node computes in to keep its integers exact"
            ) from error
        output_list = sess.run(None, input_dict)

        for output_ind in range(len(node.output)):
            # each output by its name, wherever node_outputs lists it
            outp = node.output[output_ind]
            list_ind = None
            for i in range(len(node_outputs)):
                if outp == node_outputs[i].name:
                    list_ind = i

            if list_ind is None:
                raise Exception("Output %s not found in node outputs." % outp)
            if output_list[list_ind].shape != context[outp].shape:
                warnings.warn(
                    """Output shapes disagree after node %s execution:
                    found %s vs expected %s"""
                    % (str(node), str(output_list[list_ind].shape), str(context[outp].shape))
                )
            context[outp] = output_list[list_ind]


@overload
def execute_onnx(
    model: ModelWrapper,
    input_dict: Mapping[str, np.ndarray],
    return_full_exec_context: Literal[False] = False,
    start_node: NodeProto | None = None,
    end_node: NodeProto | None = None,
    exact_integers: bool = False,
) -> dict[str, np.ndarray]:
    ...


@overload
def execute_onnx(
    model: ModelWrapper,
    input_dict: Mapping[str, np.ndarray],
    return_full_exec_context: Literal[True],
    start_node: NodeProto | None = None,
    end_node: NodeProto | None = None,
    exact_integers: bool = False,
) -> dict[str, np.ndarray | None]:
    ...


def execute_onnx(
    model: ModelWrapper,
    input_dict: Mapping[str, np.ndarray],
    return_full_exec_context: bool = False,
    start_node: NodeProto | None = None,
    end_node: NodeProto | None = None,
    exact_integers: bool = False,
) -> dict[str, np.ndarray] | dict[str, np.ndarray | None]:
    """Executes given ONNX ModelWrapper with given named inputs.

    If return_full_exec_context is False, a dict of named outputs is returned
    as indicated by the model.graph.output.

    If return return_full_exec_context is True, the full set of tensors used by
    the execution (including inputs, weights, activations and final outputs)
    will be returned as a dict; its entry for the empty name (an omitted
    optional input) is None.

    When start_node and end_node are set to None, the whole graph is executed.
    If they are set to particular ONNX nodes, only the subgraph between (and
    including) those nodes is executed.

    exact_integers: integers are computed exactly where the graph's float
    containers would round them (execute_node). A node computed so leaves its
    outputs in float64, graph outputs included.
    """

    if not model.check_all_tensor_shapes_specified():
        raise Exception("Found unspecified tensor shapes, try infer_shapes")
    ret = model.analysis(ta.nodes_topologically_sorted)
    assert (
        ret["nodes_topologically_sorted"] is True
    ), """Nodes must be
    topologically sorted."""

    graph = model.graph
    # first, we need to make sure that every variable required by the graph has
    # some buffer associated with it. this includes graph inputs (which includes
    # the input data as well as the trained parameters) and the graph ValueInfo
    # (intermediate tensors between layers)
    # this is provided by the execution_context, which is a dict of np.ndarray
    execution_context = model.make_empty_exec_context()
    # fill in any inputs provided to this function
    for inp_name in input_dict.keys():
        if inp_name in execution_context:
            if execution_context[inp_name].shape == input_dict[inp_name].shape:
                execution_context[inp_name] = input_dict[inp_name]
            else:
                raise Exception(
                    "Shape mismatch for provided input %s: found %s expected %s "
                    % (
                        inp_name,
                        str(execution_context[inp_name].shape),
                        str(input_dict[inp_name].shape),
                    )
                )
        # else:
        # raise Exception("Provided input not found in graph context: %s" % inp_name)

    # check if model has an execution mode set
    # if None, execute model node by node using execute_node()
    model_exec_mode = model.get_metadata_prop("exec_mode")
    if (model_exec_mode is None) or (model_exec_mode == ""):
        # extract opset version for node-by-node execution
        opset_imports = model.get_opset_imports()
        # execute the model node by node
        # we can simply walk down the list since the ONNX spec guarantees that it is
        # topologically sorted
        subgraph = []
        if start_node is None:
            start_node = model.graph.node[0]
        if end_node is None:
            end_node = model.graph.node[-1]
        # select the nodes between specified start/end nodes
        start_ind = model.get_node_index(start_node)
        end_ind = model.get_node_index(end_node) + 1
        assert end_ind >= start_ind, "Start/end nodes must define valid subgraph"
        subgraph = graph.node[start_ind:end_ind]
        for node in subgraph:
            if get_sanitize_quant_tensors() != 0:
                # round input values to match quantization annotation
                execution_context = sanitize_quant_values(model, node.input, execution_context)
            if node.domain in opset_imports:
                opset_version = opset_imports[node.domain]
            else:
                opset_version = get_preferred_qonnx_opset()
            execute_node(
                node,
                execution_context,
                graph,
                opset_version,
                return_full_exec_context,
                model=model,
                exact_integers=exact_integers,
            )
            if get_sanitize_quant_tensors() != 0:
                # round output values to quantization annotation
                execution_context = sanitize_quant_values(model, node.output, execution_context)
    else:
        raise Exception('Metadata property "exec_mode" is set to an unknown value.')

    if return_full_exec_context:
        return execution_context
    else:
        # provide outputs as dict
        output_dict = dict()
        for out_tensor in graph.output:
            out_name = out_tensor.name
            output_dict[out_name] = execution_context[out_name]
        return output_dict


def execute_onnx_and_make_model(model, input_dict):
    """Executes given ONNX ModelWrapper with given named inputs and return a new
    ModelWrapper where an initializer is provided for each tensor as taken from
    the execution. This new model is useful for debugging, since it contains
    all the intermediate activation values."""

    # retrieve the full execution context
    execution_context = execute_onnx(model, input_dict, True)
    new_model = copy.deepcopy(model)
    # create value_info entries and initializers for everything
    for i in execution_context.keys():
        new_model.set_initializer(i, execution_context[i])
    for vi in new_model.graph.value_info:
        new_model.graph.output.append(vi)
    return new_model


def compare_execution(
    model_a,
    model_b,
    input_dict,
    compare_fxn=lambda x, y: np.isclose(x, y, atol=1e-3).all(),
):
    """Executes two ONNX models and compare their outputs using given function.

    compare_fxn should take in two tensors and return a Boolean"""
    # compare values from first output tensors produced
    res_a = list(execute_onnx(model_a, input_dict).items())[0][1]
    res_b = list(execute_onnx(model_b, input_dict).items())[0][1]
    return compare_fxn(res_a, res_b)
