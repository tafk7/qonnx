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

"""Finding records, handler selection and complete refusal serialization."""

from __future__ import annotations

import pytest

import asyncio
import contextvars
import copy
import json
import multiprocessing.pool
import numpy as np
import pickle
import shutil
import subprocess
import sys
import threading
import warnings
from collections.abc import Iterator
from dataclasses import FrozenInstanceError, dataclass
from onnx import GraphProto, NodeProto, TensorProto, helper, numpy_helper
from pathlib import Path
from typing import Any

import qonnx.core
import qonnx.core.datatype
from qonnx.core.datatype import (
    DataType,
    DataTypeNameMalformed,
    DataTypeNameRefused,
    DataTypeNameUnknown,
    NumericRefused,
    NumericValueRefused,
)
from qonnx.core.findings import Finding, Refusal, collect_findings, collecting_findings, report_finding
from qonnx.core.metadata import MetadataRefused
from qonnx.core.modelwrapper import ModelWrapper, TensorFactRefused
from qonnx.core.onnx_exec import ExecutionArithmeticRefused, ExecutionRefused, ExecutionUnsupported, ExecutionValueRefused
from qonnx.custom_op.registry import CustomOpRefused, CustomOpUnsupported
from qonnx.transformation.base import NodeLocalTransformation, Transformation
from qonnx.transformation.fold_constants import FoldConstants, FoldConstantsFiltered
from qonnx.util.basic import qonnx_make_model


class LocatedRefusal(Refusal):
    """A consumer's refusal with a constructor different from Exception's."""

    def __init__(self, site: str, found: Finding, count: int) -> None:
        super().__init__((found,), message="a chosen message")
        self.site = site
        self.count = count


def test_findings_have_frozen_details_and_json_records() -> None:
    values = [1, {"value": "yes"}]
    found = Finding("bad-value", "operation", "change the value", (("z", values), ("node", "n")))
    values.append(2)
    assert found.details == (("node", "n"), ("z", (1, (("value", "yes"),))))
    assert str(found) == "operation: bad-value: change the value"
    assert found.subject_line() == "n: operation: bad-value: change the value"
    assert json.loads(json.dumps(found.record())) == {
        "code": "bad-value",
        "owner": "operation",
        "message": "change the value",
        "details": {"node": "n", "z": [1, [["value", "yes"]]]},
    }
    with pytest.raises(FrozenInstanceError):
        setattr(found, "owner", "other")
    assert found.about(node="other") is found
    assert dict(found.about(tensor="x").details)["tensor"] == "x"
    assert pickle.loads(pickle.dumps(found)) == found


@pytest.mark.parametrize(
    "details,base", [((("x", 1), ("x", 2)), ValueError), (((1, 2),), TypeError), ((("x", object()),), TypeError)]
)
def test_invalid_detail_records_are_programming_errors(details: tuple, base: type[Exception]) -> None:
    with pytest.raises(base):
        Finding("code", "operation", "message", details)


def test_a_refusal_needs_findings_and_has_no_code_shortcut() -> None:
    with pytest.raises(TypeError):
        Refusal(())
    with pytest.raises(ValueError):
        Finding("", "owner", "message")
    found = Finding("bad-value", "operation", "message")
    refused = Refusal((found, found.about(tensor="x")))
    assert refused.findings == (found, found.about(tensor="x"))
    assert str(refused) == "\n".join(map(str, refused.findings))
    assert not hasattr(refused, "code")


@pytest.mark.parametrize("name,expected", [("FOO", "KeyError"), ("INTx", "ValueError")])
@pytest.mark.parametrize("key_first", [True, False])
def test_datatype_handlers_select_the_same_clause_in_either_order(name: str, expected: str, key_first: bool) -> None:
    selected = ""
    if key_first:
        try:
            DataType[name]
        except KeyError:
            selected = "KeyError"
        except ValueError:
            selected = "ValueError"
    else:
        try:
            DataType[name]
        except ValueError:
            selected = "ValueError"
        except KeyError:
            selected = "KeyError"
    assert selected == expected


@pytest.mark.parametrize(
    "leaf,family,legacy",
    [
        (NumericValueRefused, NumericRefused, ValueError),
        (DataTypeNameUnknown, DataTypeNameRefused, KeyError),
        (DataTypeNameMalformed, DataTypeNameRefused, ValueError),
        (CustomOpRefused, Refusal, KeyError),
        (CustomOpUnsupported, Refusal, Exception),
        (TensorFactRefused, Refusal, ValueError),
        (MetadataRefused, Refusal, ValueError),
        (ExecutionRefused, Refusal, Exception),
        (ExecutionArithmeticRefused, ExecutionRefused, ArithmeticError),
        (ExecutionUnsupported, ExecutionRefused, NotImplementedError),
        (ExecutionValueRefused, ExecutionRefused, ValueError),
    ],
)
def test_every_leaf_keeps_its_family_and_legacy_handler(
    leaf: type[Refusal], family: type[Refusal], legacy: type[Exception]
) -> None:
    refused = leaf.of("code", "operation", "message", node="n")
    assert isinstance(refused, (family, legacy))
    assert isinstance(refused, family) and isinstance(refused, legacy)
    assert leaf.mro().index(Refusal) < leaf.mro().index(legacy)
    assert str(refused) == "operation: code: message"
    restored = pickle.loads(pickle.dumps(refused))
    assert type(restored) is leaf
    assert restored.args == refused.args and restored.__dict__ == refused.__dict__
    assert str(restored) == str(refused)
    assert not isinstance(refused, (AssertionError, IndexError))
    if legacy is KeyError:
        assert not isinstance(refused, ValueError)
    if legacy is ValueError:
        assert not isinstance(refused, KeyError)


def test_pickle_does_not_call_a_consumers_constructor() -> None:
    refused = LocatedRefusal("one", Finding("code", "owner", "message"), 7)
    refused.__cause__ = ValueError("the original reason")
    restored = pickle.loads(pickle.dumps(refused))
    assert type(restored) is LocatedRefusal
    assert restored.args == refused.args
    assert restored.__dict__ == refused.__dict__
    assert restored.site == "one" and restored.count == 7
    assert restored.__cause__ is None
    for copied in (copy.copy(refused), copy.deepcopy(refused)):
        assert type(copied) is LocatedRefusal
        assert copied.args == refused.args and copied.__dict__ == refused.__dict__
        assert copied.__cause__ is None


def test_the_documented_findings_module_exports_the_real_core_types() -> None:
    assert Finding is qonnx.core.Finding and Refusal is qonnx.core.Refusal
    assert Finding.__module__ == Refusal.__module__ == "qonnx.core"


def test_copied_datatype_refusals_are_key_errors_and_pickle(tmp_path: Path) -> None:
    core = tmp_path / "qonnx" / "core"
    core.mkdir(parents=True)
    source = Path(qonnx.core.datatype.__file__).parent
    for name in ("datatype.py", "__init__.py"):
        shutil.copyfile(source / name, core / name)
    probe = r"""
import importlib.abc
import importlib.machinery
import pickle
import sys
from pathlib import Path

copy = Path(sys.argv[1])
class CopyOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] != "qonnx":
            return None
        found = importlib.machinery.PathFinder.find_spec(name, [str(copy)] if name == "qonnx" else path)
        if found is None:
            raise ModuleNotFoundError(name, name=name)
        return found
sys.meta_path.insert(0, CopyOnly())
from qonnx.core import Refusal
from qonnx.core.datatype import DataType
try:
    DataType["FOO"]
except KeyError as error:
    assert isinstance(error, Refusal) and len(error.findings) == 1
    assert error.findings[0].code == "name-unknown"
    restored = pickle.loads(pickle.dumps(error))
    assert type(restored) is type(error)
    assert restored.args == error.args and restored.findings == error.findings
    assert Path(sys.modules["qonnx.core.datatype"].__file__).parent == copy / "qonnx" / "core"
else:
    raise AssertionError("an unknown name was accepted")
assert "qonnx.core.findings" not in sys.modules
"""
    done = subprocess.run([sys.executable, "-c", probe, str(tmp_path)], text=True, capture_output=True)
    assert done.returncode == 0, done.stderr


# Scenarios for findings reported while transformations retain their ordinary return.
def _route_model() -> ModelWrapper:
    nodes = [helper.make_node("Relu", [f"t{i}" if i else "x"], [f"t{i + 1}"], name=f"node{i}") for i in range(3)]
    graph = helper.make_graph(
        nodes,
        "route",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("t3", TensorProto.FLOAT, [1])],
    )
    return ModelWrapper(qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 13)]))


def _route_finding(code: str, node: str = "") -> Finding:
    return Finding(code, "reporter", code, (("node", node),) if node else ())


class _ReportOnce(Transformation):
    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        for node in model.graph.node:
            report_finding(_route_finding("seen", node.name))
        return model, False


class _RepeatReports(Transformation):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        self.count += 1
        report_finding(_route_finding("same"))
        report_finding(_route_finding(str(self.count)))
        return model, self.count < 3


class _NestedReports(Transformation):
    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        model = model.transform(_ReportOnce(), cleanup=False)
        report_finding(_route_finding("outer"))
        return model, False


@pytest.mark.parametrize("make_deepcopy", [False, True])
def test_a_one_shot_pass_returns_a_model_and_reports_in_node_order(make_deepcopy: bool) -> None:
    model = _route_model()
    with collect_findings() as found:
        result = model.transform(_ReportOnce(), make_deepcopy=make_deepcopy, cleanup=False)
    assert isinstance(result, ModelWrapper)
    assert (result is model) is (not make_deepcopy)
    assert [dict(f.details)["node"] for f in found] == ["node0", "node1", "node2"]


def test_repeat_and_nested_transforms_report_without_forwarding_code() -> None:
    repeating = _RepeatReports()
    with collect_findings() as found:
        _route_model().transform(repeating, cleanup=False)
    assert repeating.count == 3
    assert [f.code for f in found] == ["same", "1", "2", "3"]
    with collect_findings() as nested:
        _route_model().transform(_NestedReports(), cleanup=False)
    assert [f.code for f in nested] == ["seen", "seen", "seen", "outer"]


@dataclass(frozen=True, slots=True)
class _TaggedFinding(Finding):
    tag: str = ""


def test_scopes_propagate_once_by_complete_record_equality() -> None:
    first = _route_finding("first")
    with collect_findings() as outer:
        assert collecting_findings()
        report_finding(first)
        with collect_findings() as inner:
            report_finding(first)
            report_finding(_TaggedFinding("tagged", "reporter", "tagged", tag="a"))
            report_finding(_TaggedFinding("tagged", "reporter", "tagged", tag="b"))
        assert outer == inner
        with collect_findings(propagate=False) as trial:
            report_finding(_route_finding("discarded"))
        assert [f.code for f in trial] == ["discarded"]
    assert [f.code for f in outer] == ["first", "tagged", "tagged"]
    assert not collecting_findings()


def test_overlapping_entries_refuse_before_changing_the_active_scope() -> None:
    collector = collect_findings()
    with pytest.raises(RuntimeError, match="already open"):
        with collector as found:
            report_finding(_route_finding("before-overlap"))
            with collector:
                pytest.fail("overlapping entry was accepted")
    assert not collecting_findings()
    report_finding(_route_finding("after-close"))
    assert found == [_route_finding("before-overlap")]
    with collector as reused:
        report_finding(_route_finding("reused"))
    assert reused == [_route_finding("reused")]
    assert not collecting_findings()


class _ReportsThenRefuses(Transformation):
    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        report_finding(_route_finding("before-refusal"))
        raise NumericValueRefused.of("undefined", "reporter", "refused")


def test_reports_before_a_refusal_survive_and_pass_up() -> None:
    with collect_findings() as outer:
        with pytest.raises(NumericValueRefused):
            with collect_findings() as inner:
                _route_model().transform(_ReportsThenRefuses(), cleanup=False)
    assert outer == inner == [_route_finding("before-refusal")]


class _LocalReports(NodeLocalTransformation):
    def __init__(self, workers: int, fail_at: str | None = None) -> None:
        super().__init__(workers)
        self.fail_at = fail_at

    def applyNodeLocal(self, node: NodeProto) -> tuple[NodeProto, bool]:
        report_finding(_route_finding("local", node.name))
        if node.name == self.fail_at:
            raise NumericValueRefused.of("undefined", "local", "node refused", node=node.name)
        return node, False


class _LocalUnpicklable(NodeLocalTransformation):
    def applyNodeLocal(self, node: NodeProto) -> tuple[NodeProto, bool]:
        class LocalFinding(Finding):
            pass

        report_finding(LocalFinding("local", "reporter", node.name))
        return node, False


@pytest.mark.parametrize("workers", [1, 2])
def test_node_local_reports_are_carried_in_processing_order(workers: int) -> None:
    before = _route_model()
    with collect_findings() as found:
        after = before.transform(_LocalReports(workers), cleanup=False)
    assert after.model == before.model
    assert [dict(f.details)["node"] for f in found] == ["node2", "node1", "node0"]


@pytest.mark.parametrize("workers", [1, 2])
def test_node_local_failure_carries_reports_up_to_the_failing_node(workers: int) -> None:
    with collect_findings() as found:
        with pytest.raises(NumericValueRefused) as caught:
            _route_model().transform(_LocalReports(workers, "node1"), cleanup=False)
    assert [dict(f.details)["node"] for f in found] == ["node2", "node1"]
    assert dict(caught.value.findings[0].details)["node"] == "node1"


@pytest.mark.parametrize("workers", [1, 2])
def test_no_collector_keeps_node_local_transport_unchanged(workers: int) -> None:
    before = _route_model()
    after = before.transform(_LocalUnpicklable(workers), cleanup=False)
    assert after.model == before.model
    with pytest.raises(NumericValueRefused):
        before.transform(_LocalReports(workers, "node1"), cleanup=False)


def test_a_collector_makes_unpicklable_worker_findings_fail_visibly() -> None:
    with collect_findings():
        with pytest.raises(multiprocessing.pool.MaybeEncodingError):
            _route_model().transform(_LocalUnpicklable(2), cleanup=False)


def test_plain_threads_drop_reports_and_copied_contexts_carry_them_once() -> None:
    with collect_findings() as found:
        plain = threading.Thread(target=report_finding, args=(_route_finding("plain"),))
        plain.start()
        plain.join()
        threads = [
            threading.Thread(target=contextvars.copy_context().run, args=(report_finding, _route_finding("copied")))
            for _ in range(8)
        ]
        for worker in threads:
            worker.start()
        for worker in threads:
            worker.join()
    assert found == [_route_finding("copied")]


def test_a_copied_context_reports_to_the_open_parent_after_close() -> None:
    with collect_findings() as outer:
        with collect_findings(propagate=False) as inner:
            copied = contextvars.copy_context()
            report_finding(_route_finding("inside"))
        copied.run(report_finding, _route_finding("late"))
    assert inner == [_route_finding("inside")]
    assert outer == [_route_finding("late")]
    copied.run(report_finding, _route_finding("dropped"))
    assert outer == [_route_finding("late")]


def test_closing_in_a_different_context_refuses_without_closing_the_scope() -> None:
    collector = collect_findings()
    with collector as found:
        with pytest.raises(ValueError):
            contextvars.copy_context().run(collector.__exit__, None, None, None)
        assert collecting_findings()
        report_finding(_route_finding("still-open"))
    assert found == [_route_finding("still-open")]
    assert not collecting_findings()


def test_concurrent_copied_context_reports_survive_scope_close() -> None:
    ready = threading.Barrier(5)

    def report_many(worker: int) -> None:
        ready.wait(timeout=10)
        for number in range(100):
            report_finding(_route_finding(f"{worker}:{number}"))

    with collect_findings() as outer:
        with collect_findings() as inner:
            threads = [
                threading.Thread(target=contextvars.copy_context().run, args=(report_many, worker)) for worker in range(4)
            ]
            for thread in threads:
                thread.start()
            ready.wait(timeout=10)
        closed = tuple(inner)
        for thread in threads:
            thread.join(timeout=10)
            assert not thread.is_alive()
        assert tuple(inner) == closed
    assert {finding.code for finding in outer} == {f"{worker}:{number}" for worker in range(4) for number in range(100)}
    assert len(outer) == 400


def test_late_async_tasks_report_to_an_open_parent_or_drop() -> None:
    async def scenario() -> None:
        ready = asyncio.Event()

        async def report_later() -> None:
            await ready.wait()
            report_finding(_route_finding("late-task"))

        with collect_findings() as outer:
            with collect_findings() as inner:
                task = asyncio.create_task(report_later())
            ready.set()
            await task
        assert inner == [] and outer == [_route_finding("late-task")]
        ready.clear()
        with collect_findings() as closed:
            dropped = asyncio.create_task(report_later())
        ready.set()
        await dropped
        assert closed == []

    asyncio.run(scenario())


def test_a_generator_cannot_revive_a_closed_scope() -> None:
    def scenario() -> None:
        def suspended() -> Iterator[list[Finding]]:
            with collect_findings() as inner:
                yield inner

        outer = collect_findings()
        outer.__enter__()
        generator = suspended()
        inner = next(generator)
        outer.__exit__(None, None, None)
        generator.close()
        assert not collecting_findings()
        report_finding(_route_finding("dropped"))
        assert inner == []

    contextvars.copy_context().run(scenario)


def test_the_documented_yield_limit_collects_consumer_reports() -> None:
    def suspended() -> Iterator[list[Finding]]:
        with collect_findings(propagate=False) as inner:
            yield inner

    with collect_findings() as outer:
        generator = suspended()
        inner = next(generator)
        report_finding(_route_finding("consumer"))
        generator.close()
    assert inner == [_route_finding("consumer")] and outer == []


def test_no_scope_keeps_no_output_and_wrong_types_still_raise(capsys: pytest.CaptureFixture[str]) -> None:
    with warnings.catch_warnings(record=True) as warned:
        report_finding(_route_finding("discarded"))
    assert not warned
    assert capsys.readouterr() == ("", "")
    with pytest.raises(TypeError):
        report_finding(1)  # type: ignore[arg-type]


def _branch_model() -> ModelWrapper:
    def branch(name: str) -> GraphProto:
        node = helper.make_node(
            "Constant", [], ["y"], name="same", value=helper.make_tensor("c", TensorProto.FLOAT, [1], [1.0])
        )
        return helper.make_graph([node], name, [], [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])

    node = helper.make_node("If", ["cond"], ["y"], name="if", then_branch=branch("then"), else_branch=branch("else"))
    graph = helper.make_graph(
        [node],
        "outer",
        [helper.make_tensor_value_info("cond", TensorProto.BOOL, [])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    return ModelWrapper(qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 13)]))


class _ReportGraphs(Transformation):
    def __init__(self, qualify: bool) -> None:
        super().__init__()
        self.qualify = qualify

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        for node in model.graph.node:
            details = {"node": node.name}
            if self.qualify:
                details["graph"] = model.graph.name
            report_finding(Finding("seen", "reporter", "seen", tuple(details.items())))
        return model, False


@pytest.mark.parametrize("preorder", [False, True])
@pytest.mark.parametrize("qualify", [False, True])
def test_subgraphs_report_and_graph_details_preserve_distinct_subjects(preorder: bool, qualify: bool) -> None:
    with collect_findings() as found:
        _branch_model().transform(
            _ReportGraphs(qualify), cleanup=False, apply_to_subgraphs=True, use_preorder_traversal=preorder
        )
    assert len(found) == (3 if qualify else 2)
    if qualify:
        assert {dict(f.details)["graph"] for f in found} == {"outer", "then", "else"}


def _unfoldable_sum(unproved: bool) -> ModelWrapper:
    kind = TensorProto.DOUBLE if unproved else TensorProto.FLOAT
    value = 2**53 if unproved else 2**24
    dtype = np.float64 if unproved else np.float32
    graph = helper.make_graph(
        [helper.make_node("Add", ["a", "b"], ["y"], name="sum")],
        "fold_report",
        [],
        [helper.make_tensor_value_info("y", kind, [1])],
        initializer=[
            numpy_helper.from_array(np.array([value], dtype=dtype), "a"),
            numpy_helper.from_array(np.array([1], dtype=dtype), "b"),
        ],
    )
    model = ModelWrapper(qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 13)]))
    for name in ("a", "b", "y"):
        model.set_tensor_datatype(name, DataType["INT64" if unproved else "INT26"])
    return model


@pytest.mark.parametrize("unproved,code", [(False, "fold-unrepresentable"), (True, "fold-unproved")])
def test_folding_reports_why_it_keeps_the_node(unproved: bool, code: str) -> None:
    with collect_findings() as found:
        result = _unfoldable_sum(unproved).transform(FoldConstants(), cleanup=False)
    assert [node.name for node in result.graph.node] == ["sum"]
    assert len(found) == 1
    finding = found[0]
    assert finding.code == code and finding.owner == "FoldConstants"
    details = dict(finding.details)
    assert (details["graph"], details["node"], details["tensor"]) == ("fold_report", "sum", "y")
    if unproved:
        assert details["refusal_code"] == "exact-beyond-float64"
        assert "passes 2**53" in str(details["refusal_message"])
    else:
        assert details["element_type"] == TensorProto.FLOAT and details["container"] == "float32"


def test_filtered_folding_reports_its_own_public_owner() -> None:
    with collect_findings() as found:
        result = _unfoldable_sum(False).transform(FoldConstantsFiltered(lambda model, node: True), cleanup=False)
    assert [node.name for node in result.graph.node] == ["sum"]
    assert len(found) == 1 and found[0].owner == "FoldConstantsFiltered"
    assert found[0].code == "fold-unrepresentable"


@pytest.mark.parametrize("unproved", [False, True])
def test_folding_without_a_collector_keeps_the_same_node_without_output(
    unproved: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    model = _unfoldable_sum(unproved)
    before = model.model.SerializeToString()
    with warnings.catch_warnings(record=True) as warned:
        result = model.transform(FoldConstants(), cleanup=False)
    assert result.model.SerializeToString() == before
    assert not warned and capsys.readouterr() == ("", "")


@pytest.mark.parametrize("filtered", [False, True])
@pytest.mark.parametrize("collecting", [False, True])
@pytest.mark.parametrize("unfoldable", [False, True])
def test_folding_reads_the_public_owner_only_when_it_reports(filtered: bool, collecting: bool, unfoldable: bool) -> None:
    reads: list[str] = []

    class OwnerAudit(type(Transformation)):
        def __getattribute__(cls, name: str) -> Any:
            if name == "__name__":
                assert collecting_findings(), "owner read without a collector"
                reads.append(name)
            return super().__getattribute__(name)

    base = FoldConstantsFiltered if filtered else FoldConstants

    class Audited(base, metaclass=OwnerAudit):
        pass

    transform = Audited(lambda model, node: True) if filtered else Audited()
    model = _unfoldable_sum(False)
    if not unfoldable:
        model.set_initializer("b", np.array([0], dtype=np.float32))
    if collecting:
        with collect_findings() as found:
            result = model.transform(transform, cleanup=False)
        assert len(found) == int(unfoldable)
        if found:
            assert found[0].owner == "Audited"
    else:
        result = model.transform(transform, cleanup=False)
    assert len(result.graph.node) == int(unfoldable)
    assert len(reads) == int(collecting and unfoldable)


@pytest.mark.parametrize("filtered", [False, True])
def test_collecting_a_fold_report_does_not_repeat_a_deprecated_datatype_warning(filtered: bool) -> None:
    model = _unfoldable_sum(False)
    for annotation in model.graph.quantization_annotation:
        if annotation.tensor_name == "y":
            for entry in annotation.quant_parameter_tensor_names:
                if entry.key == "finn_datatype":
                    entry.value = "INT0"
    transform = FoldConstantsFiltered(lambda model, node: True) if filtered else FoldConstants()
    with warnings.catch_warnings(record=True) as plain:
        warnings.simplefilter("always")
        uncollected = model.transform(transform, cleanup=False)
    with warnings.catch_warnings(record=True) as collected:
        warnings.simplefilter("always")
        with collect_findings() as found:
            result = model.transform(transform, cleanup=False)
    assert result.model.SerializeToString() == uncollected.model.SerializeToString()
    assert len(plain) == 1
    assert [(warning.category, str(warning.message)) for warning in collected] == [
        (warning.category, str(warning.message)) for warning in plain
    ]
    assert len(found) == 1 and found[0].code == "fold-unrepresentable"
    assert dict(found[0].details)["element_type"] == TensorProto.FLOAT
