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

import copy
import json
import pickle
import shutil
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

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
from qonnx.core.findings import Finding, Refusal
from qonnx.core.metadata import MetadataRefused
from qonnx.core.modelwrapper import TensorFactRefused
from qonnx.core.onnx_exec import ExecutionArithmeticRefused, ExecutionRefused, ExecutionUnsupported, ExecutionValueRefused
from qonnx.custom_op.registry import CustomOpRefused, CustomOpUnsupported


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
