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

"""qonnx.core.datatype imports the standard library and numpy alone, and works on its own.

A process with no onnx, onnxruntime or onnxscript to import loads it, and every module it
brings in is the standard library's, numpy's or qonnx's own. A copy of it beside
qonnx/core/__init__.py and nothing else of qonnx, the layout a generated PYNQ driver
carries, imports and answers in a fresh interpreter.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import qonnx.core.datatype

BLOCKED = """
import importlib.abc
import sys

class Blocked(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("onnx", "onnxruntime", "onnxscript"):
            raise ImportError(f"{name} is blocked")
        return None

sys.meta_path.insert(0, Blocked())
before = set(sys.modules)
import qonnx.core.datatype

roots = {name.split(".")[0] for name in set(sys.modules) - before}
# numpy 1.x's compiled modules register the Cython runtime they were built with.
cython = {root for root in roots if root == "cython_runtime" or root.startswith("_cython_")}
foreign = sorted(root for root in roots - cython if root not in sys.stdlib_module_names and root not in ("numpy", "qonnx"))
assert not foreign, foreign
"""

# qonnx is found only in the copy: no installed qonnx, editable or not, can answer for it.
ALONE = """
import importlib.abc
import importlib.machinery
import sys
from pathlib import Path

copy = Path(sys.argv[1])

class CopyOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] != "qonnx":
            return None
        found = importlib.machinery.PathFinder.find_spec(name, [str(copy)] if name == "qonnx" else path)
        if found is None:
            raise ModuleNotFoundError(f"{name} is not in the copy", name=name)
        return found

sys.meta_path.insert(0, CopyOnly())
before = set(sys.modules)
from qonnx.core.datatype import DataType

loaded = sorted(name for name in set(sys.modules) - before if name.split(".")[0] == "qonnx")
assert loaded == ["qonnx", "qonnx.core", "qonnx.core.datatype"], loaded
assert Path(sys.modules["qonnx.core.datatype"].__file__).resolve().parent == (copy / "qonnx" / "core").resolve()

int4 = DataType["INT4"]
assert int4.bitwidth() == 4 and int4.signed() and int4.min() == -8 and int4.max() == 7
assert int4.allowed(7) and not int4.allowed(8)
assert int4.bounds() == (-8, 7)
assert DataType["FIXED<8,4>"].bounds()[1] * 16 == 127
assert DataType["BIPOLAR"].allowed(-1) and not DataType["BIPOLAR"].allowed(0)
assert DataType["FLOAT32"].to_numpy_dt().itemsize == 4
assert DataType.get_smallest_possible(-3) == DataType["INT3"]
"""


def test_the_datatype_module_imports_numpy_alone() -> None:
    done = subprocess.run([sys.executable, "-c", BLOCKED], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_the_datatype_module_works_alone(tmp_path: Path) -> None:
    source = Path(qonnx.core.datatype.__file__).parent
    core = tmp_path / "qonnx" / "core"
    core.mkdir(parents=True)
    shutil.copy(source / "datatype.py", core / "datatype.py")
    shutil.copy(source / "__init__.py", core / "__init__.py")
    done = subprocess.run([sys.executable, "-c", ALONE, str(tmp_path)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
