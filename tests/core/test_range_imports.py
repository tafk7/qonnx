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

"""The numeric range modules import only the standard library, numpy and their refusal bases."""

from __future__ import annotations

import pytest

import subprocess
import sys


def check_import(module: str, extra_source: str = "") -> None:
    code = """
import ast
import importlib
import importlib.abc
import importlib.util
import sys
from pathlib import Path

numeric = {"qonnx.core.datatype", "qonnx.core.ranges", "qonnx.core.range_rules"}
refusals = {"qonnx.core.findings", "qonnx.core.Refusal", "qonnx.core.Finding"}
blocked: list[str] = []

class Blocked(importlib.abc.MetaPathFinder):
    def find_spec(self, name: str, path: object = None, target: object = None) -> None:
        if name.split(".")[0] in ("onnx", "onnxruntime", "onnxscript", "google"):
            blocked.append(name)
            raise ImportError(f"{name} is blocked")
        if name.startswith("qonnx.") and name not in numeric | refusals | {"qonnx.core"}:
            blocked.append(name)
            raise ImportError(f"non-numeric module {name} is blocked")
        return None

sys.meta_path.insert(0, Blocked())
before = set(sys.modules)
module = importlib.import_module(sys.argv[1])
exec(sys.argv[2], vars(module))
assert not blocked, f"blocked import attempts: {blocked}"
roots = {name.split(".")[0] for name in set(sys.modules) - before}
cython = {root for root in roots if root == "cython_runtime" or root.startswith("_cython_")}
foreign = sorted(root for root in roots - cython if root not in sys.stdlib_module_names and root not in ("numpy", "qonnx"))
assert not foreign, foreign

# Inspect every import statement, including imports deferred inside functions.
for name in sorted((numeric | {"qonnx.core", "qonnx.core.findings"}) & set(sys.modules)):
    item = sys.modules[name]
    source = Path(item.__file__).read_text()
    if item is module:
        source += chr(10) + sys.argv[2]
    for node in ast.walk(ast.parse(source)):
        imports = []
        if isinstance(node, ast.Import):
            imports = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = importlib.util.resolve_name("." * node.level + base, item.__package__)
            imports = [base + "." + alias.name for alias in node.names] if base in ("qonnx", "qonnx.core") else [base]
        for imported in imports:
            root = imported.split(".")[0]
            if root == "qonnx":
                allowed = imported in numeric | refusals
            else:
                allowed = root in sys.stdlib_module_names or root == "numpy"
            assert allowed, f"non-numeric import statement: {imported}"
"""
    subprocess.run([sys.executable, "-c", code, module, extra_source], check=True, capture_output=True, text=True)


def test_ranges_are_numpy_only() -> None:
    check_import("qonnx.core.ranges")


@pytest.mark.parametrize("dependency", ["onnx", "onnxruntime", "qonnx.core.modelwrapper"])
def test_an_optional_import_cannot_hide_a_dependency(dependency: str) -> None:
    source = f"try:\n    import {dependency}\nexcept ImportError:\n    pass\n"
    with pytest.raises(subprocess.CalledProcessError) as refused:
        check_import("qonnx.core.ranges", source)
    assert "blocked import attempts:" in refused.value.stderr
    assert dependency in refused.value.stderr


def test_a_deferred_import_cannot_hide_a_dependency() -> None:
    with pytest.raises(subprocess.CalledProcessError) as refused:
        check_import("qonnx.core.ranges", "def deferred():\n    import onnx\n")
    assert "non-numeric import statement: onnx" in refused.value.stderr


def test_range_rules_are_numpy_only() -> None:
    check_import("qonnx.core.range_rules")
