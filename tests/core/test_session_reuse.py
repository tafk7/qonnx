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

"""Session reuse, an opt-in switched by the reuse_sessions context: one retained session
of a stateless node model in all, both of ONNX Runtime's thread pools capped where
threads is given, random and seeded ops always run fresh, and the setting held by the
current context (a generator suspended inside the block holds it until closed)."""

from __future__ import annotations

import pytest

import numpy as np
import onnxruntime as rt
import weakref
from onnx import TensorProto, helper
from typing import Any, Callable, Iterator

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import exact_integers, exact_integers_enabled, execute_onnx, reuse_sessions
from qonnx.util.basic import qonnx_make_model

F = TensorProto.FLOAT


def identity(name: str, *, invalid: bool = False) -> ModelWrapper:
    """An Identity from x to y, held in float32; with invalid, an attribute Identity does
    not have, which ONNX Runtime refuses."""
    node = helper.make_node("Identity", ["x"], ["y"], name=name)
    if invalid:
        node.attribute.append(helper.make_attribute("not_an_identity_attribute", 1))
    graph = helper.make_graph(
        [node], "reuse", [helper.make_tensor_value_info("x", F, [1])], [helper.make_tensor_value_info("y", F, [1])]
    )
    return ModelWrapper(qonnx_make_model(graph))


def quant_avg_pool() -> ModelWrapper:
    node = helper.make_node(
        "QuantAvgPool2d",
        ["x"],
        ["y"],
        domain="qonnx.custom_op.general",
        kernel=2,
        stride=2,
        ibits=4,
        obits=6,
        signed=0,
    )
    graph = helper.make_graph(
        [node],
        "pool",
        [helper.make_tensor_value_info("x", F, [1, 1, 2, 2])],
        [helper.make_tensor_value_info("y", F, [1, 1, 1, 1])],
    )
    return ModelWrapper(qonnx_make_model(graph))


def channels_last_max_pool() -> ModelWrapper:
    node = helper.make_node(
        "MaxPool",
        ["x"],
        ["y"],
        domain="qonnx.custom_op.channels_last",
        kernel_shape=[2, 2],
        strides=[2, 2],
        pads=[0, 0, 0, 0],
    )
    graph = helper.make_graph(
        [node],
        "channels_last_pool",
        [helper.make_tensor_value_info("x", F, [1, 2, 2, 1])],
        [helper.make_tensor_value_info("y", F, [1, 1, 1, 1])],
    )
    return ModelWrapper(qonnx_make_model(graph))


def random_model(kind: str) -> tuple[ModelWrapper, dict[str, Any]]:
    """A model of one random or seeded op (or an If whose branches draw), and its feeds."""
    shape = [32]
    inputs = [helper.make_tensor_value_info("x", F, shape)]
    feeds: dict[str, Any] = {"x": np.full(shape, 0.5, np.float32)}
    output_type = F
    if kind in {"RandomNormal", "RandomUniform"}:
        inputs, feeds = [], {}
        node = helper.make_node(kind, [], ["y"], name="draw", shape=shape, seed=23.0)
    elif kind == "Multinomial":
        inputs = [helper.make_tensor_value_info("x", F, [1, 2])]
        feeds = {"x": np.zeros((1, 2), np.float32)}
        shape, output_type = [1, 32], TensorProto.INT64
        node = helper.make_node(kind, ["x"], ["y"], name="draw", sample_size=32, dtype=output_type, seed=23.0)
    elif kind == "Dropout":
        node = helper.make_node(kind, ["x", "ratio", "training"], ["y"], name="draw", seed=23)
    elif kind == "If":
        inputs = [helper.make_tensor_value_info("x", TensorProto.BOOL, [])]
        feeds = {"x": np.array(True)}
        branch = helper.make_graph(
            [helper.make_node("RandomNormal", [], ["value"], seed=23.0, shape=shape)],
            "random_branch",
            [],
            [helper.make_tensor_value_info("value", F, shape)],
        )
        node = helper.make_node(kind, ["x"], ["y"], name="draw", then_branch=branch, else_branch=branch)
    else:
        node = helper.make_node(kind, ["x"], ["y"], name="draw", seed=23.0)
    graph = helper.make_graph([node], "random", inputs, [helper.make_tensor_value_info("y", output_type, shape)])
    model = ModelWrapper(qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))
    if kind == "Dropout":
        model.set_initializer("ratio", np.array(0.5, dtype=np.float32))
        model.set_initializer("training", np.array(True, dtype=np.bool_))
    return model, feeds


def observe(monkeypatch: pytest.MonkeyPatch, record: Callable[[rt.InferenceSession, dict[str, Any]], None]) -> None:
    """Calls record with every session ONNX Runtime constructs, and its keywords."""
    constructor = rt.InferenceSession

    def observed(data: bytes, **kwargs: Any) -> rt.InferenceSession:
        found = constructor(data, **kwargs)
        record(found, kwargs)
        return found

    monkeypatch.setattr(rt, "InferenceSession", observed)


def test_by_default_every_node_runs_a_fresh_session_with_default_options(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[dict[str, Any]] = []
    observe(monkeypatch, lambda found, kwargs: received.append(kwargs))
    model = identity("default")
    for value in (1, 2):
        assert execute_onnx(model, {"x": np.array([value], np.float32)})["y"].item() == value
    assert received == [{}, {}]


def test_a_stateless_repetition_reuses_its_session(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[rt.InferenceSession] = []
    observe(monkeypatch, lambda found, kwargs: received.append(found))
    model = identity("repeated")
    model.set_tensor_datatype("x", DataType["INT4"])
    with reuse_sessions():
        first = execute_onnx(model, {"x": np.ones(1, np.float32)})
        # Inputs and settings outside the node's model do not construct another session.
        second = execute_onnx(model, {"x": np.full(1, 3, np.float32)}, exact_integers=True)
    assert first["y"].item() == 1 and second["y"].item() == 3
    assert len(received) == 1


@pytest.mark.parametrize("threads", [None, 3])
@pytest.mark.parametrize("kind", ["retained", "fresh", "quant-average-pool", "channels-last"])
def test_every_session_honours_the_thread_cap(threads: int | None, kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[rt.SessionOptions] = []
    observe(monkeypatch, lambda found, kwargs: received.extend((kwargs["sess_options"], found.get_session_options())))
    if kind == "fresh":
        model, feeds = random_model("RandomNormal")
    elif kind == "quant-average-pool":
        model, feeds = quant_avg_pool(), {"x": np.full((1, 1, 2, 2), 7, np.float32)}
    elif kind == "channels-last":
        model, feeds = channels_last_max_pool(), {"x": np.arange(4, dtype=np.float32).reshape(1, 2, 2, 1)}
    else:
        model, feeds = identity(f"threads-{threads}"), {"x": np.array([7], np.float32)}
    with reuse_sessions(threads):
        output = execute_onnx(model, feeds)["y"]
    if kind != "fresh":
        assert output.item() == {"retained": 7, "quant-average-pool": 28, "channels-last": 3}[kind]
    assert len(received) == 2
    defaults = rt.SessionOptions()
    for options in received:
        expected = defaults.intra_op_num_threads if threads is None else threads
        assert options.intra_op_num_threads == expected
        assert options.inter_op_num_threads == (defaults.inter_op_num_threads if threads is None else threads)
        assert options.execution_mode == defaults.execution_mode
        assert options.graph_optimization_level == defaults.graph_optimization_level


def test_a_changed_thread_cap_constructs_a_session_of_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[int] = []
    observe(monkeypatch, lambda found, kwargs: received.append(found.get_session_options().intra_op_num_threads))
    model = identity("changing-threads")
    with reuse_sessions():
        for threads in (1, 3, 3, 1):
            with reuse_sessions(threads):
                assert execute_onnx(model, {"x": np.array([7], np.float32)})["y"].item() == 7
    assert received == [1, 3, 1]


@pytest.mark.parametrize(
    "threads,error",
    [(0, ValueError), (-1, ValueError), (2**31, ValueError), ("3", TypeError), (1.0, TypeError), (True, TypeError)],
)
def test_an_invalid_thread_cap_is_refused(threads: Any, error: type[Exception]) -> None:
    with pytest.raises(error, match="threads"):
        with reuse_sessions(threads):
            pass


def test_one_idle_session_is_retained_and_released(monkeypatch: pytest.MonkeyPatch) -> None:
    constructor = rt.InferenceSession
    references: list[weakref.ReferenceType[rt.InferenceSession]] = []
    idle_at_construction: list[int] = []

    def alive() -> int:
        return sum(reference() is not None for reference in references)

    def observed(data: bytes, **kwargs: Any) -> rt.InferenceSession:
        idle_at_construction.append(alive())
        found = constructor(data, **kwargs)
        references.append(weakref.ref(found))
        return found

    monkeypatch.setattr(rt, "InferenceSession", observed)
    a, b = identity("lifetime-a"), identity("lifetime-b")
    with reuse_sessions():
        for model, value in ((a, 1), (b, 2), (a, 3)):
            assert execute_onnx(model, {"x": np.array([value], np.float32)})["y"].item() == value
            assert alive() == 1
        assert idle_at_construction == [0, 0, 0]
        # A random op runs fresh, and releases the retained session first.
        execute_onnx(random_model("RandomNormal")[0], {})
        assert idle_at_construction == [0, 0, 0, 0] and alive() == 0
        execute_onnx(a, {"x": np.ones(1, np.float32)})
        assert alive() == 1
        # A construction that fails leaves none retained.
        with pytest.raises(rt.capi.onnxruntime_pybind11_state.InvalidGraph):
            execute_onnx(identity("lifetime-failure", invalid=True), {"x": np.ones(1, np.float32)})
        assert idle_at_construction == [0] * 6 and alive() == 0
        execute_onnx(a, {"x": np.full(1, 4, np.float32)})
        assert execute_onnx(a, {"x": np.full(1, 5, np.float32)})["y"].item() == 5
        assert idle_at_construction == [0] * 7 and alive() == 1
    # Leaving the context releases it.
    assert alive() == 0


@pytest.mark.parametrize(
    "kind",
    ["RandomNormal", "RandomUniform", "RandomNormalLike", "RandomUniformLike", "Multinomial", "Bernoulli", "Dropout", "If"],
)
def test_every_seeded_op_and_nested_graph_keeps_fresh_session_answers(kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    model, feeds = random_model(kind)
    expected = rt.InferenceSession(model.model.SerializeToString()).run(None, feeds)[0]
    constructed: list[rt.InferenceSession] = []
    observe(monkeypatch, lambda found, kwargs: constructed.append(found))
    with reuse_sessions(1):
        for _ in range(3):
            assert np.array_equal(execute_onnx(model, feeds)["y"], expected)
    assert len(constructed) == 3


def test_a_generator_suspended_inside_both_blocks_holds_them_until_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    references: list[weakref.ReferenceType[rt.InferenceSession]] = []
    observe(monkeypatch, lambda found, kwargs: references.append(weakref.ref(found)))
    model = identity("suspended")

    def suspended() -> Iterator[None]:
        with exact_integers(), reuse_sessions(1):
            execute_onnx(model, {"x": np.ones(1, np.float32)})
            yield

    generator = suspended()
    next(generator)
    # Yielding inside the blocks is unsupported: the settings stay in force for the
    # caller, and the session retained, until the generator is closed.
    assert exact_integers_enabled() is True
    assert len(references) == 1 and references[0]() is not None
    generator.close()
    assert exact_integers_enabled() is False
    assert references[0]() is None
