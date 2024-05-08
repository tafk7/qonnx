# Copyright (c) 2023 Advanced Micro Devices, Inc.
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

import clize
import dataclasses as dc
import itertools
import numpy as np
import pprint
from warnings import warn

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_node
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name
from qonnx.util.cleanup import cleanup_model
from qonnx.util.onnx import valueinfo_to_tensor

# walk the graph to deduce range information about each tensor
# assumptions:
# - layout and shape inference already completed
# - range info is generated per-channel (tuple of 1D arrays) or per-tensor (tuple of scalars)


@dc.dataclass
class RangeInfo:
    range: tuple = None
    int_range: tuple = None
    scale: np.ndarray = None
    bias: np.ndarray = None
    is_initializer: bool = False

    def has_integer_info(self) -> bool:
        integer_props = [self.int_range, self.scale, self.bias]
        return all([x is not None for x in integer_props])


def calculate_matvec_accumulator_extremum(matrix: np.ndarray, vec_min, vec_max):
    """Calculate the minimum and maximum possible result (accumulator) values for a dot product A*x,
    given matrix A of dims (MH, MW), and vector (MW) with range (vec_min, vec_max). vec_min and
    vec_max are either scalars, or 1D arrays of length MW.
    Returns (acc_min, acc_max) where acc_min and acc_max are 1D arrays of length MH."""
    max_vectors = np.where(matrix > 0, vec_max, vec_min)
    min_vectors = np.where(matrix > 0, vec_min, vec_max)
    max_values = (matrix * max_vectors).sum(axis=1)
    min_values = (matrix * min_vectors).sum(axis=1)
    return (min_values, max_values)


def calc_gemm_range(node, model, range_dict):
    alpha = get_by_name(node.attribute, "alpha").f
    beta = get_by_name(node.attribute, "beta").f
    transA = get_by_name(node.attribute, "transA")
    if transA is not None:
        transA = transA.i
    else:
        transA = 0
    transB = get_by_name(node.attribute, "transB")
    if transB is not None:
        transB = transB.i
    else:
        transB = 1
    assert (not transA) and transB
    iname = node.input[0]
    wname = node.input[1]
    bname = None
    if len(node.input) > 2:
        bname = node.input[2]
    oname = node.output[0]

    irange = range_dict[iname].range
    imin, imax = irange
    weight_range_info = range_dict[wname]
    assert weight_range_info.is_initializer, "Uninitialized Gemm weights"
    assert weight_range_info.range[0].ndim == 2, "Malformed Gemm weights in range info"
    assert (weight_range_info.range[0] == weight_range_info.range[1]).all(), "Non-constant Gemm weights in range info"
    weights = weight_range_info.range[0]
    if type(imin) is np.ndarray:
        assert len(imin) == weights.shape[1], "Dot product length mismatch, np broadcast may be wrong"
    pmin, pmax = calculate_matvec_accumulator_extremum(weights, imin, imax)
    # apply Gemm scale factors to matrix multiply output
    pmin *= alpha
    pmax *= alpha
    # if there is a bias, apply it to the range
    if bname is not None:
        bias_range_info = range_dict[bname]
        assert bias_range_info.is_initializer, "Uninitialized Gemm bias"
        assert bias_range_info.range[0].ndim == 1, "Malformed Gemm bias in range info"
        assert (bias_range_info.range[0] == bias_range_info.range[1]).all(), "Non-constant Gemm bias in range info"
        bias = bias_range_info.range[0]
        pmin += beta * bias
        pmax += beta * bias
    ret = (pmin, pmax)
    range_dict[oname].range = ret


def calc_matmul_range(node, model, range_dict):
    iname = node.input[0]
    wname = node.input[1]
    oname = node.output[0]
    irange = range_dict[iname].range
    imin, imax = irange
    weight_range_info = range_dict[wname]
    assert weight_range_info.is_initializer, "Uninitialized MatMul weights"
    assert weight_range_info.range[0].ndim == 2, "Malformed MatMul weights in range info"
    assert (weight_range_info.range[0] == weight_range_info.range[1]).all(), "Non-constant MatMul weights in range info"
    weights = weight_range_info.range[0]
    # util function expects (mh, mw) so transpose
    weights = weights.transpose()
    if type(imin) is np.ndarray:
        assert len(imin) == weights.shape[1], "Dot product length mismatch, np broadcast may be wrong"
    ret = calculate_matvec_accumulator_extremum(weights, imin, imax)
    range_dict[oname].range = ret


def calc_conv_range(node, model, range_dict):
    iname = node.input[0]
    wname = node.input[1]
    assert len(node.input) == 2, "Found unsupported Conv with bias"
    oname = node.output[0]
    irange = range_dict[iname].range
    imin, imax = irange
    weight_range_info = range_dict[wname]
    assert weight_range_info.is_initializer, "Uninitialized Conv weights"
    assert weight_range_info.range[0].ndim >= 2, "Malformed Conv weights in range info"
    assert (weight_range_info.range[0] == weight_range_info.range[1]).all(), "Non-constant Conv weights in range info"
    weights = weight_range_info.range[0]
    # do weight reshaping to treat Conv similar to MatMul
    # (mh, mw) = (ofm, (ifm x k0 x k1 x ...))
    conv_ofm = weights.shape[0]
    conv_ifm = weights.shape[1]
    weights = weights.reshape(conv_ofm, -1)
    k_total = weights.shape[1] // conv_ifm
    groups = get_by_name(node.attribute, "group")
    if groups is None:
        # default to dense convs
        groups = 1
    else:
        groups = groups.i
    # TODO smarter check, other kinds of grouped convs out there..
    is_depthwise = groups > 1
    # need to construct specialzed input range vectors for Conv
    if is_depthwise:
        conv_ifm = conv_ofm
    if type(imin) is np.ndarray:
        imin_rep = np.repeat(imin, k_total)
        imax_rep = np.repeat(imax, k_total)
    else:
        imin_rep = imin
        imax_rep = imax
    dw_ret_min = []
    dw_ret_max = []
    for i in range(conv_ofm):
        w_slice = weights[i, :].reshape(1, -1)
        if is_depthwise and type(imin_rep) is np.ndarray:
            dw_ret = calculate_matvec_accumulator_extremum(
                w_slice, imin_rep[i * k_total : (i + 1) * k_total], imax_rep[i * k_total : (i + 1) * k_total]
            )
        else:
            dw_ret = calculate_matvec_accumulator_extremum(w_slice, imin_rep, imax_rep)
        dw_ret_min.append(dw_ret[0].item())
        dw_ret_max.append(dw_ret[1].item())
    ret = (np.asarray(dw_ret_min), np.asarray(dw_ret_max))
    range_dict[oname].range = ret


def calc_convtranspose_range(node, model, range_dict):
    iname = node.input[0]
    wname = node.input[1]
    assert len(node.input) == 2, "Found unsupported ConvTranspose with bias"
    oname = node.output[0]
    irange = range_dict[iname].range
    imin, imax = irange
    weight_range_info = range_dict[wname]
    assert weight_range_info.is_initializer, "Uninitialized ConvTranspose weights"
    assert weight_range_info.range[0].ndim >= 2, "Malformed ConvTranspose weights in range info"
    assert (
        weight_range_info.range[0] == weight_range_info.range[1]
    ).all(), "Non-constant ConvTranspose weights in range info"
    weights = weight_range_info.range[0]
    groups = get_by_name(node.attribute, "group")
    if groups is None:
        # default to dense convs
        groups = 1
    else:
        groups = groups.i
    assert groups == 1, "Only dense (non-grouped) ConvTranspose is supported"
    # do weight reshaping to treat Conv similar to MatMul
    # (mh, mw) = (ofm, (ifm x k0 x k1 x ...))
    conv_ofm = weights.shape[1]
    conv_ifm = weights.shape[0]
    weights = weights.transpose(1, 0, 2, 3).reshape(conv_ofm, -1)
    k_total = weights.shape[1] // conv_ifm
    if type(imin) is np.ndarray:
        imin_rep = np.repeat(imin, k_total)
        imax_rep = np.repeat(imax, k_total)
    else:
        imin_rep = imin
        imax_rep = imax
    dw_ret_min = []
    dw_ret_max = []
    for i in range(conv_ofm):
        w_slice = weights[i, :].reshape(1, -1)
        dw_ret = calculate_matvec_accumulator_extremum(w_slice, imin_rep, imax_rep)
        dw_ret_min.append(dw_ret[0].item())
        dw_ret_max.append(dw_ret[1].item())
    ret = (np.asarray(dw_ret_min), np.asarray(dw_ret_max))
    range_dict[oname].range = ret


def get_minmax_prototype_tensors(irange, ishp, inp_vi, i_channel_axis=1):
    proto_min = valueinfo_to_tensor(inp_vi)
    proto_max = valueinfo_to_tensor(inp_vi)
    if type(irange[0]) in [float, int, np.float16, np.float32, np.float64, np.uint8, np.int8]:
        imin, imax = irange
        proto_min[...] = imin
        proto_max[...] = imax
    elif type(irange[0]) is np.ndarray:
        # irange is [(min_ch0, max_ch0), (min_ch1, max_ch1) ...]
        n_ch = ishp[i_channel_axis]
        proto_min = np.moveaxis(proto_min, i_channel_axis, 0)
        proto_max = np.moveaxis(proto_max, i_channel_axis, 0)
        for ch in range(n_ch):
            proto_min[ch, ...] = irange[0][ch]
            proto_max[ch, ...] = irange[1][ch]
        proto_min = np.moveaxis(proto_min, 0, i_channel_axis)
        proto_max = np.moveaxis(proto_max, 0, i_channel_axis)
    else:
        assert False, "Unknown range type"
    return (proto_min, proto_max)


def is_dyn_input(x, model):
    return model.get_initializer(x) is None and x != ""


def calc_monotonic_range(node, model, range_dict, i_channel_axis=1):
    opset_version = model.model.opset_import[0].version
    oname = node.output[0]
    dyn_inps = [x for x in node.input if is_dyn_input(x, model)]
    n_dyn_inp = len(dyn_inps)
    # create context for single-node execution
    ctx = {x: model.get_initializer(x) for x in node.input}
    for oname in node.output:
        ctx[oname] = valueinfo_to_tensor(model.get_tensor_valueinfo(oname))
    if n_dyn_inp == 0:
        # special case: all inputs were constants (e.g. quantized for trained weights)
        # so there is no proto vectors to operate over really - just need a single eval
        execute_node(node, ctx, model.graph, opset_version=opset_version)
        # grab new output and keep the entire thing as the range
        for oname in node.output:
            range_dict[oname].range = (ctx[oname], ctx[oname])
            range_dict[oname].is_initializer = True
        return
    # going beyond this point we are sure we have at least one dynamic input
    # generate min-max prototype vectors for each dynamic input
    proto_vectors = []
    for inp in dyn_inps:
        irange = range_dict[inp].range
        ishp = model.get_tensor_shape(inp)
        inp_vi = model.get_tensor_valueinfo(inp)
        proto_vectors.append(get_minmax_prototype_tensors(irange, ishp, inp_vi, i_channel_axis))
    # process all combinations of prototype vectors for dynamic inputs
    running_min = [None for i in range(len(node.output))]
    running_max = [None for i in range(len(node.output))]
    # assume all outputs are homogenous wrt data layout (e.g. channel axis
    # always lives in the same position)
    axes_to_min = [i for i in range(ctx[oname].ndim)]
    axes_to_min.remove(i_channel_axis)
    axes_to_min = tuple(axes_to_min)
    for inps in itertools.product(*proto_vectors):
        for i in range(n_dyn_inp):
            ctx[dyn_inps[i]] = inps[i]
        execute_node(node, ctx, model.graph, opset_version=opset_version)
        for oind, oname in enumerate(node.output):
            # grab new output and update running min/max
            out = ctx[oname]
            if len(axes_to_min) != 0:
                chanwise_min = out.min(axis=axes_to_min).flatten()
                chanwise_max = out.max(axis=axes_to_min).flatten()
            else:
                # for certain cases (e.g. quantizer for a 1D vector) the axes_to_min may be empty
                # then we don't do any more min/max reduction and just take the vector as-is
                chanwise_max = out.flatten()
                chanwise_min = out.flatten()
            running_min[oind] = (
                np.minimum(chanwise_min, running_min[oind]).flatten() if running_min[oind] is not None else chanwise_min
            )
            running_max[oind] = (
                np.maximum(chanwise_max, running_max[oind]).flatten() if running_max[oind] is not None else chanwise_max
            )
    for oind, oname in enumerate(node.output):
        range_dict[oname].range = (running_min[oind], running_max[oind])


def calc_range_outdtype(node, model, range_dict):
    oname = node.output[0]
    odt = model.get_tensor_datatype(oname)
    assert odt is not None, "Cannot infer %s range, dtype annotation is missing" % oname
    range_dict[oname].range = (odt.min(), odt.max())


def calc_range_all_initializers(model, range_dict):
    all_tensor_names = model.get_all_tensor_names()
    for tensor_name in all_tensor_names:
        tensor_init = model.get_initializer(tensor_name)
        if tensor_init is not None:
            range_dict[tensor_name] = RangeInfo(range=(tensor_init, tensor_init), is_initializer=True)
            # use % 1 == 0 to identify integers
            if ((tensor_init % 1) == 0).all():
                range_dict[tensor_name].int_range = (tensor_init, tensor_init)
                range_dict[tensor_name].scale = np.asarray([1.0], dtype=np.float32)
                range_dict[tensor_name].bias = np.asarray([0.0], dtype=np.float32)


optype_to_range_calc = {
    "Transpose": calc_monotonic_range,
    "MatMul": calc_matmul_range,
    "Conv": calc_conv_range,
    "ConvTranspose": calc_convtranspose_range,
    "QuantMaxNorm": calc_range_outdtype,
    "Flatten": calc_monotonic_range,
    "Reshape": calc_monotonic_range,
    "Quant": calc_monotonic_range,
    "BipolarQuant": calc_monotonic_range,
    "Mul": calc_monotonic_range,
    "Sub": calc_monotonic_range,
    "Div": calc_monotonic_range,
    "Add": calc_monotonic_range,
    "BatchNormalization": calc_monotonic_range,
    "Relu": calc_monotonic_range,
    "Pad": calc_monotonic_range,
    "AveragePool": calc_monotonic_range,
    "Trunc": calc_range_outdtype,
    "MaxPool": calc_monotonic_range,
    "Resize": calc_monotonic_range,
    "Upsample": calc_monotonic_range,
    "GlobalAveragePool": calc_monotonic_range,
    "Gemm": calc_gemm_range,
    "QuantizeLinear": calc_monotonic_range,
    "DequantizeLinear": calc_monotonic_range,
    "Clip": calc_monotonic_range,
    "Sigmoid": calc_monotonic_range,
    "Concat": calc_monotonic_range,
    "Split": calc_monotonic_range,
}


def simplify_range(range):
    """Where possible, simplify a range that is expressed as channelwise ranges
    back to a scalar range if all channels' ranges were equal."""
    rmin = range[0]
    rmax = range[1]
    if type(rmin) is np.ndarray and type(rmax) is np.ndarray:
        rmin_eq = all(rmin == rmin[0])
        rmax_eq = all(rmax == rmax[0])
        if rmin_eq and rmax_eq:
            return (rmin[0], rmax[0])
        else:
            return range
    else:
        return range


REPORT_MODE_RANGE = "range"
REPORT_MODE_STUCKCHANNEL = "stuck_channel"
REPORT_MODE_ZEROSTUCKCHANNEL = "zerostuck_channel"

report_modes = {REPORT_MODE_RANGE, REPORT_MODE_STUCKCHANNEL, REPORT_MODE_ZEROSTUCKCHANNEL}

report_mode_options = clize.parameters.mapped(
    [
        (REPORT_MODE_RANGE, [REPORT_MODE_RANGE], "Report ranges"),
        (REPORT_MODE_STUCKCHANNEL, [REPORT_MODE_STUCKCHANNEL], "Report stuck channels"),
        (REPORT_MODE_ZEROSTUCKCHANNEL, [REPORT_MODE_ZEROSTUCKCHANNEL], "Report 0-stuck channels"),
    ]
)


# assumptions for intrange calculations:
# * "normal" ranges (.range) for inputs and outputs have been already computed & present in range_dict
# * other range info (.int_range, .scale, .bias) is present for inputs in range_dict, if relevant


def calc_intrange_quant(node, model, range_dict):
    orange_inf = range_dict[node.output[0]]
    # get quantizer parameters
    q_bitwidth = model.get_initializer(node.input[3])
    assert not (q_bitwidth is None)
    assert q_bitwidth.ndim <= 1
    q_zeropt = model.get_initializer(node.input[2])
    assert not (q_zeropt is None)
    q_scale = model.get_initializer(node.input[1])
    assert not (q_scale is None)
    # TODO can we use input/output range info instead of all this
    # node-specific behavior, using one of the other existing handlers?
    # we need to do a little style conversion for the scale/bias:
    # intrange calculations here represent quant tensors as Mx+N (x: int tensor, M: scale, N: bias)
    # whereas Quant nodes represent them as S(x-Z) (x: int tensor, S: scale, Z: zeropoint)
    # it follows that M = S and N = -SZ
    orange_inf.scale = q_scale
    orange_inf.bias = -(q_scale * q_zeropt)
    if orange_inf.is_initializer:
        # if the quantizer output was a constant, we can derive the entire
        # integer tensor component by recovering it using the scale/bias info
        assert (orange_inf.range[0] == orange_inf.range[1]).all()
        q_out = orange_inf.range[0]
        q_out_int_cand = (q_out / q_scale) + q_zeropt
        q_out_int = np.round(q_out_int_cand)
        # TODO ensure that rounding error introduced here is smaller than scale? how does zeropt come into this?
        orange_inf.int_range = (q_out_int, q_out_int)
    else:
        # input is not constant so we can only reason about the "dtype range"
        # as implemented by the generic settings of the quantizer:
        # output int range is decided by bitwidth & signedness of quantization
        qnt_node_inst = getCustomOp(node)
        odt_int_type = qnt_node_inst.get_integer_datatype(model)
        if qnt_node_inst.get_nodeattr("narrow") and qnt_node_inst.get_nodeattr("signed"):
            narrow_range_adj = 1
        else:
            narrow_range_adj = 0
        orange_inf.int_range = (odt_int_type.min() + narrow_range_adj, odt_int_type.max())

    range_dict[node.output[0]] = orange_inf


def check_int_inputs(node, range_dict):
    inp_int_info = [range_dict[x].has_integer_info() for x in node.input]
    return inp_int_info


def calc_intrange_relu(node, model, range_dict):
    # try to propagate integer range and scale/bias info for ReLU
    inp_int_info = check_int_inputs(node, range_dict)
    if not any(inp_int_info):
        # must have at least one input with integer info, otherwise no point
        warn(node.name + " has no integer info on inputs, cannot propagate")
        return
    irange_inf = range_dict[node.input[inp_int_info.index(True)]]
    orange_inf = range_dict[node.output[0]]
    # we'll use the ReLU output range to infer the integer parts
    # * output range can only come from the ReLU identity part (input > 0)
    # * scale and bias are always left unchanged, unless stuck channel
    # range_max = S*int_range_max + B
    # range_min = S*int_range_min + B
    # S and B are identical between input and output
    scale = irange_inf.scale
    bias = irange_inf.bias
    # int_range_min = (range_min - B) / S
    # int_range_max = (range_max - B) / S
    int_range_0 = (orange_inf.range[0] - bias) / scale
    int_range_1 = (orange_inf.range[1] - bias) / scale
    int_range_min = np.round(int_range_0)
    int_range_max = np.round(int_range_1)
    range_dict[node.output[0]].scale = scale
    range_dict[node.output[0]].bias = bias
    range_dict[node.output[0]].int_range = (int_range_min, int_range_max)


def calc_intrange_linear(node, model, range_dict):
    # try to propagate integer range and scale/bias info
    inp_int_info = check_int_inputs(node, range_dict)
    if all(inp_int_info):
        # use own handler when all inputs have integer info available
        return calc_intrange_linear_allint(node, model, range_dict)
    if not any(inp_int_info):
        # must have at least one input with integer info, otherwise no point
        warn(node.name + " has no integer info on inputs, cannot propagate")
        return
    irange_inf = range_dict[node.input[inp_int_info.index(True)]]
    # remaining cases are mix of integer and non-integer inputs
    # e.g. scaled-int dynamic input times channelwise float scales
    # assumption: int part remains untouched, scale/bias gets updated
    range_dict[node.output[0]].int_range = irange_inf.int_range
    orange_inf = range_dict[node.output[0]]
    # range_max = S*int_range_max + B
    # range_min = S*int_range_min + B
    # so S = (range_max - range_min) / (int_range_max - int_range_min)
    # and afterwards, B = range_max - S*int_range_max
    # TODO scale and bias may contain NaN's when channels are stuck
    # how best to deal with this? leave as is? set to 1/0?
    # try to recover in some other way? (perturb the actual range before calling range_calc_fxn)
    scale = (orange_inf.range[1] - orange_inf.range[0]) / (orange_inf.int_range[1] - orange_inf.int_range[0])
    bias = orange_inf.range[1] - scale * orange_inf.int_range[1]
    range_dict[node.output[0]].scale = scale
    range_dict[node.output[0]].bias = bias


def calc_intrange_linear_allint(node, model, range_dict):
    for node_in in node.input:
        irange_inf = range_dict[node_in]
        if not irange_inf.has_integer_info():
            # integer range info is missing in at least one of the inputs
            # cannot infer anything about the output int range info
            return
        # be extra conservative for now: no negative scales, no biases
        assert (irange_inf.scale >= 0).all(), "Need nonnegative scale for inputs"
        assert (irange_inf.bias == 0).all(), "Need zero bias for weights"
    orange_inf = range_dict[node.output[0]]
    int_range_dict = {}
    for node_out in node.output:
        int_range_dict[node_out] = RangeInfo()
    # use integer components of input ranges for new range computation
    for node_in in node.input:
        int_range_dict[node_in] = RangeInfo(
            range=range_dict[node_in].int_range, is_initializer=range_dict[node_in].is_initializer
        )
    range_calc_fxn = optype_to_range_calc[node.op_type]
    range_calc_fxn(node, model, int_range_dict)
    int_orange_inf = int_range_dict[node.output[0]]
    # now deduce the output scale factor and bias from all available info
    # range_max = S*int_range_max + B
    # range_min = S*int_range_min + B
    # so S = (range_max - range_min) / (int_range_max - int_range_min)
    # and afterwards, B = range_max - S*int_range_max
    # TODO scale and bias may contain NaN's when channels are stuck
    # how best to deal with this? leave as is? set to 1/0?
    # try to recover in some other way? (perturb the actual range before calling range_calc_fxn)
    scale = (orange_inf.range[1] - orange_inf.range[0]) / (int_orange_inf.range[1] - int_orange_inf.range[0])
    bias = orange_inf.range[1] - scale * int_orange_inf.range[1]
    range_dict[node.output[0]].scale = scale
    range_dict[node.output[0]].bias = bias
    range_dict[node.output[0]].int_range = int_orange_inf.range


def calc_intrange_identity(node, model, range_dict):
    n_dyn_inps = [(model.get_initializer(x) is None) for x in node.input].count(True)
    assert n_dyn_inps == 1, "Identity int range prop needs a single dynamic input"
    irange_inf = range_dict[node.input[0]]
    for o in node.output:
        range_dict[o].scale = irange_inf.scale
        range_dict[o].bias = irange_inf.bias
        range_dict[o].int_range = irange_inf.int_range


def calc_intrange(model, range_dict):
    intrange_mapping = {
        "Conv": calc_intrange_linear,
        "MatMul": calc_intrange_linear,
        "BatchNormalization": calc_intrange_linear,
        "Add": calc_intrange_linear,
        "Relu": calc_intrange_relu,
        "Quant": calc_intrange_quant,
        "Pad": calc_intrange_identity,
        "MaxPool": calc_intrange_identity,
        "Reshape": calc_intrange_identity,
    }

    # now walk the graph node by node and propagate scaled-int range info
    for node in model.graph.node:
        op_ok = node.op_type in intrange_mapping.keys()
        if op_ok:
            range_calc_fxn = intrange_mapping[node.op_type]
            range_calc_fxn(node, model, range_dict)
            # range_dict[node.output[0]].int_range = simplify_range(range_dict[node.output[0]].int_range)
        else:
            warn("Skipping %s : op_ok? (%s) %s" % (node.name, node.op_type, str(op_ok)))


def range_analysis(
    model_filename_or_wrapper,
    *,
    irange="",
    key_filter: str = "",
    report_mode: report_mode_options = REPORT_MODE_STUCKCHANNEL,
    prettyprint=False,
    do_cleanup=False,
    strip_initializers_from_report=True,
    scaled_int=False
):
    assert report_mode in report_modes, "Unrecognized report_mode, must be " + str(report_modes)
    if isinstance(model_filename_or_wrapper, ModelWrapper):
        model = model_filename_or_wrapper
    else:
        model = ModelWrapper(model_filename_or_wrapper)
    if isinstance(irange, str):
        if irange == "":
            range_min = None
            range_max = None
        else:
            irange = eval(irange)
            range_min, range_max = irange
            if isinstance(range_min, list):
                range_min = np.asarray(range_min, dtype=np.float32)
            if isinstance(range_max, list):
                range_max = np.asarray(range_max, dtype=np.float32)
    elif isinstance(irange, tuple):
        range_min, range_max = irange
    elif isinstance(irange, RangeInfo):
        pass
    else:
        assert False, "Unknown irange type"
    if do_cleanup:
        model = cleanup_model(model)
    # call constant folding & shape inference, this preserves weight quantizers
    # (but do not do extra full cleanup, in order to preserve node/tensor naming)
    # TODO is this redundant? remove?
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(InferDataTypes())
    range_dict = {}
    stuck_chans = {}

    # start by calculating/annotating range info for input tensors
    for inp in model.graph.input:
        iname = inp.name
        if isinstance(irange, RangeInfo):
            range_dict[iname] = irange
        else:
            if range_min is None or range_max is None:
                # use idt annotation
                idt = model.get_tensor_datatype(iname)
                assert idt is not None, "Could not infer irange, please specify"
                range_min = idt.min()
                range_max = idt.max()
            range_dict[iname] = RangeInfo(range=(range_min, range_max))

    # add range info for all tensors with initializers
    calc_range_all_initializers(model, range_dict)

    # now walk the graph node by node and propagate range info
    for node in model.graph.node:
        dyn_inputs = [x for x in node.input if is_dyn_input(x, model)]
        inprange_ok = all([x in range_dict.keys() for x in dyn_inputs])
        op_ok = node.op_type in optype_to_range_calc.keys()
        if inprange_ok and op_ok:
            # create entries in range_dict with RangeInfo type for all outputs
            # since range analysis functions will be assigning to the .range member of
            # this RangeInfo directly later on
            for node_out in node.output:
                range_dict[node_out] = RangeInfo()
            range_calc_fxn = optype_to_range_calc[node.op_type]
            range_calc_fxn(node, model, range_dict)
            if not range_dict[node.output[0]].is_initializer:
                # only consider non-initializer (dynamic) tensors for range simplification
                # and stuck channel analysis
                out_range = range_dict[node.output[0]].range
                tensor_stuck_chans = np.nonzero(out_range[0] == out_range[1])[0]
                if len(tensor_stuck_chans) > 0:
                    list_stuck_chans = list(tensor_stuck_chans)
                    list_stuck_values = list(out_range[0][tensor_stuck_chans])
                    stuck_chans[node.output[0]] = list(zip(list_stuck_chans, list_stuck_values))
                range_dict[node.output[0]].range = simplify_range(out_range)
        else:
            warn("Skipping %s : inp_range? %s op_ok? (%s) %s" % (node.name, str(inprange_ok), node.op_type, str(op_ok)))

    # if scaled-int range prop is enabled, call as postproc
    if scaled_int:
        calc_intrange(model, range_dict)

    # range dict is now complete, apply filters and formatting
    if report_mode in [REPORT_MODE_ZEROSTUCKCHANNEL, REPORT_MODE_STUCKCHANNEL]:
        ret = stuck_chans
    else:
        ret = range_dict
        if strip_initializers_from_report:
            # exclude all initializer ranges for reporting
            ret = {k: v for (k, v) in ret.items() if not v.is_initializer}

    # only keep tensors (keys) where filter appears in the name
    if key_filter != "":
        ret = {k: v for (k, v) in ret.items() if key_filter in k}
    # only keep tensors (keys) where filter appears in the name
    if key_filter != "":
        ret = {k: v for (k, v) in ret.items() if key_filter in k}

    if report_mode == REPORT_MODE_RANGE:
        # TODO convert ranges in report to regular Python lists for nicer printing
        pass
    elif report_mode == REPORT_MODE_ZEROSTUCKCHANNEL:
        # only leave channels that are stuck at zero
        # value info removed since implicitly 0
        new_ret = {}
        for tname, schans in ret.items():
            schans_only_zero = set([x[0] for x in schans if x[1] == 0])
            if len(schans_only_zero) > 0:
                new_ret[tname] = schans_only_zero
        ret = new_ret
    if prettyprint:
        ret = pprint.pformat(ret, sort_dicts=False)
    return ret


def main():
    clize.run(range_analysis)


if __name__ == "__main__":
    main()
