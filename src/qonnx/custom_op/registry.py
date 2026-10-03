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

import importlib
import inspect
import re
import warnings
from threading import RLock
from typing import Dict, List, Optional, Tuple, Type
from onnx import NodeProto
from qonnx.custom_op.base import CustomOp

# Nested registry for O(1) lookups: domain -> op_type -> version -> CustomOp class
# Uses "since version" semantics: version N covers opset N until a higher version exists
_OP_REGISTRY: Dict[str, Dict[str, Dict[int, Type[CustomOp]]]] = {}

_REGISTRY_LOCK = RLock()

# Maps ONNX domain names to Python module paths (used for imports only)
_DOMAIN_ALIASES: Dict[str, str] = {
    "onnx.brevitas": "qonnx.custom_op.general",
}


def add_domain_alias(domain: str, module_path: str) -> None:
    """Map a domain name to a different module path.

    Args:
        domain: The ONNX domain name (e.g., "finn.custom_op.fpgadataflow")
        module_path: The Python module path to use instead (e.g., "finn_custom_ops.fpgadataflow")
    """
    with _REGISTRY_LOCK:
        _DOMAIN_ALIASES[domain] = module_path


def resolve_domain(domain: str) -> str:
    """Resolve a domain to its actual module path, handling aliases.

    Args:
        domain: The ONNX domain name

    Returns:
        Resolved module path
    """
    return _DOMAIN_ALIASES.get(domain, domain)


# The one rule for a versioned name: OpType_vN, N a positive integer without
# leading zeros; any other name is an op type of version 1 (Thresholding_vitis)
_VERSIONED_NAME = re.compile(r"(?P<op_type>.+)_v(?P<version>[1-9][0-9]*)")


def split_versioned_name(name: str) -> Tuple[str, int]:
    """Split a registered name into (op_type, since-version).

    "IntQuant_v2" is ("IntQuant", 2); "IntQuant", "Thresholding_vitis" and
    "Op_v02" are op types of version 1.
    """
    match = _VERSIONED_NAME.fullmatch(name)
    if match is None:
        return name, 1
    return match["op_type"], int(match["version"])


def op_identity(cls: Type[CustomOp], exported_as: Optional[str] = None) -> Tuple[str, int]:
    """The (op_type, since-version) a CustomOp class is registered under.

    A class may state either in its own body (``op_type = "MatMul"``,
    ``op_version = 6``). A stated identity is not inherited, so a backend
    subclass of a kernel op is not registered as the kernel op. Otherwise both
    come from the name the domain exports the class under (``exported_as``,
    default the class name), split by split_versioned_name.

    Raises:
        ValueError: If a stated op_type is not a nonempty string or a stated
            op_version not a positive integer
    """
    own = vars(cls)
    name_op_type, name_version = split_versioned_name(exported_as or cls.__name__)
    op_type = own.get("op_type", name_op_type)
    op_version = own.get("op_version", name_version)
    if not isinstance(op_type, str) or not op_type:
        raise ValueError(f"{cls.__name__}.op_type must be a nonempty string, not {op_type!r}")
    if type(op_version) is not int or op_version < 1:
        raise ValueError(f"{cls.__name__}.op_version must be a positive integer, not {op_version!r}")
    return op_type, op_version


def _get_op_type_for_class(cls: Type[CustomOp]) -> str:
    """The op_type of a CustomOp class (see op_identity)."""
    return op_identity(cls)[0]


def _get_op_version_for_class(cls: Type[CustomOp]) -> int:
    """The since-version of a CustomOp class (see op_identity)."""
    return op_identity(cls)[1]


def _exported_classes(module) -> List[Tuple[str, Type[CustomOp]]]:
    """(exported name, class) for every CustomOp class a domain module exports.

    Its __all__ (and, for backward compatibility, a legacy ``custom_op`` dict
    beside it), else the legacy dict alone, else every class in the module.
    The legacy dict pattern:
        custom_op = dict()
        custom_op["IntQuant"] = IntQuant
        custom_op["IntQuant_v2"] = IntQuant_v2
    """
    legacy = getattr(module, "custom_op", None)
    if not isinstance(legacy, dict):
        legacy = None
    if hasattr(module, "__all__"):
        pairs = [(name, getattr(module, name, None)) for name in module.__all__]
        if legacy is not None:
            pairs += list(legacy.items())
    elif legacy is not None:
        pairs = list(legacy.items())
    else:
        pairs = inspect.getmembers(module, inspect.isclass)
    return [(name, obj) for name, obj in pairs if inspect.isclass(obj) and issubclass(obj, CustomOp) and obj is not CustomOp]


def _discover_custom_op_versions(domain: str, op_type: str) -> Dict[int, Type[CustomOp]]:
    """All versions of one custom op that a domain's module exports.

    Every exported class is identified by op_identity; those whose op_type
    matches are returned by since-version.

    Args:
        domain: The ONNX domain name
        op_type: The specific op type to discover

    Returns:
        Dict mapping version -> CustomOp class
    """
    module_path = resolve_domain(domain)
    versions: Dict[int, Type[CustomOp]] = {}
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        return versions
    for name, obj in _exported_classes(module):
        cls_op_type, version = op_identity(obj, exported_as=name)
        if cls_op_type != op_type:
            continue
        if version in versions and versions[version] is not obj:
            warnings.warn(
                f"Multiple classes found for {domain}.{op_type} version {version}: "
                f"{versions[version].__name__} and {obj.__name__}. Using {obj.__name__}."
            )
        versions[version] = obj
    return versions


def _resolve_version(
    available_versions: Dict[int, Type[CustomOp]], requested_version: Optional[int]
) -> Tuple[int, Type[CustomOp]]:
    """Resolve which version to use given available and requested versions.

    Uses "since version" semantics: highest version <= requested is selected.

    Resolution strategy:
    1. If requested is None, use highest available version
    2. Try exact match
    3. Use highest version <= requested
    4. Raise KeyError if no suitable version

    Args:
        available_versions: Dict of available versions -> CustomOp classes
        requested_version: Requested opset version, or None for highest

    Returns:
        Tuple of (resolved_version, CustomOp_class)

    Raises:
        KeyError: If no suitable version found
    """
    if not available_versions:
        raise KeyError("No versions available")

    # Strategy 1: If no specific version requested, use highest
    if requested_version is None:
        highest = max(available_versions.keys())
        return highest, available_versions[highest]

    # Strategy 2: Try exact match
    if requested_version in available_versions:
        return requested_version, available_versions[requested_version]

    # Strategy 3: Use highest version <= requested (since version semantics)
    suitable = [v for v in available_versions.keys() if v <= requested_version]
    if suitable:
        selected = max(suitable)
        return selected, available_versions[selected]

    # Strategy 4: No suitable version found
    available_list = sorted(available_versions.keys())
    raise KeyError(
        f"No suitable version found. Requested: {requested_version}, "
        f"Available: {available_list}. Lowest available version is {available_list[0]}."
    )


def add_op_to_domain(domain: str, op_class: Type[CustomOp]) -> None:
    """Register a custom op directly to a domain at runtime.

    The op_type and version are automatically derived from the class name.
    Useful for testing and experimentation. For production, define CustomOps
    in the appropriate module file.

    Args:
        domain: ONNX domain name (e.g., "qonnx.custom_op.general")
        op_class: CustomOp subclass (version inferred from name)

    Example:
        add_op_to_domain("qonnx.custom_op.general", MyTestOp)      # v1
        add_op_to_domain("qonnx.custom_op.general", MyTestOp_v2)  # v2
    """
    if not issubclass(op_class, CustomOp):
        raise ValueError(f"{op_class} must be a subclass of CustomOp")

    op_type = _get_op_type_for_class(op_class)
    op_version = _get_op_version_for_class(op_class)

    with _REGISTRY_LOCK:
        # Ensure nested dict structure exists
        if domain not in _OP_REGISTRY:
            _OP_REGISTRY[domain] = {}
        if op_type not in _OP_REGISTRY[domain]:
            _OP_REGISTRY[domain][op_type] = {}

        _OP_REGISTRY[domain][op_type][op_version] = op_class


def getCustomOp(node: NodeProto, onnx_opset_version: int | None = None) -> CustomOp:
    """Get a custom op instance for an ONNX node.

    Uses "since version" semantics: selects highest version <= requested opset.
    Lazy loads only the requested op_type using __all__ for efficiency.

    Args:
        node: ONNX node with domain and op_type attributes
        onnx_opset_version: Opset version from model's opset_import, or None for highest

    Returns:
        CustomOp instance for the node

    Raises:
        KeyError: If op_type not found in domain or no suitable version available
    """
    op_type = node.op_type
    domain = node.domain

    with _REGISTRY_LOCK:
        # O(1) nested dict lookup to check cache
        if domain in _OP_REGISTRY and op_type in _OP_REGISTRY[domain]:
            cached_versions = _OP_REGISTRY[domain][op_type]
        else:
            # Cache miss: discover THIS op only (lazy, uses __all__ for speed)
            cached_versions = _discover_custom_op_versions(domain, op_type)

            if not cached_versions:
                module_path = resolve_domain(domain)
                raise KeyError(
                    f"Op '{op_type}' not found in domain '{domain}' (module: {module_path}). "
                    f"Ensure it's defined in the module with proper naming (OpName or OpName_vN)."
                )

            # Cache it in nested structure
            if domain not in _OP_REGISTRY:
                _OP_REGISTRY[domain] = {}
            _OP_REGISTRY[domain][op_type] = cached_versions

        # Resolve which version to use
        resolved_version, op_class = _resolve_version(cached_versions, onnx_opset_version)

        # Instantiate and return
        return op_class(node, onnx_opset_version=resolved_version)


def get_supported_versions(domain: str, op_type: str) -> List[int]:
    """Get list of supported opset versions for a custom op.

    Returns all "since versions" where the operator was introduced or changed.

    Args:
        domain: ONNX domain name
        op_type: Operation type name

    Returns:
        Sorted list of opset versions

    Raises:
        KeyError: If op not found
    """
    with _REGISTRY_LOCK:
        # O(1) check if cached
        if domain in _OP_REGISTRY and op_type in _OP_REGISTRY[domain]:
            return sorted(_OP_REGISTRY[domain][op_type].keys())

        # Not cached: discover this op
        versions_dict = _discover_custom_op_versions(domain, op_type)

        if not versions_dict:
            raise KeyError(f"Op '{op_type}' not found in domain '{domain}'")

        # Cache discovered versions
        if domain not in _OP_REGISTRY:
            _OP_REGISTRY[domain] = {}
        _OP_REGISTRY[domain][op_type] = versions_dict

        return sorted(versions_dict.keys())


def is_custom_op(domain: str, op_type: Optional[str] = None) -> bool:
    """Check if a custom op exists or if a domain has any custom ops.

    Args:
        domain: The ONNX domain name
        op_type: Optional operation type name. If None, checks if domain has any ops.

    Returns:
        True if the specific op exists (when op_type given) or
        if any ops exist for the domain (when op_type=None), False otherwise
    """
    # Empty domain means standard ONNX op
    if not domain:
        return False

    with _REGISTRY_LOCK:
        if op_type is not None:
            # Check for specific op - O(1) with nested dict
            if domain in _OP_REGISTRY and op_type in _OP_REGISTRY[domain]:
                return True
            # Try to discover
            versions = _discover_custom_op_versions(domain, op_type)
            return len(versions) > 0
        else:
            # Check if domain has any registered ops
            if domain in _OP_REGISTRY and _OP_REGISTRY[domain]:
                return True
            # Try to import the domain module as fallback
            module_path = resolve_domain(domain)
            try:
                importlib.import_module(module_path)
                return True
            except (ModuleNotFoundError, ValueError):
                return False


def hasCustomOp(domain: str, op_type: str) -> bool:
    """Deprecated: Use is_custom_op instead.

    Check if a custom op exists.

    Args:
        domain: The ONNX domain name
        op_type: The operation type name

    Returns:
        True if the op exists, False otherwise
    """
    warnings.warn(
        "hasCustomOp is deprecated and will be removed in QONNX v1.0. " "Use is_custom_op instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return is_custom_op(domain, op_type)


def get_ops_in_domain(domain: str) -> List[Tuple[str, Type[CustomOp]]]:
    """Get all CustomOp classes available in a domain.

    Note: Returns unique op_types. If multiple versions exist, returns the highest version.
    This function eagerly loads all ops in the domain.

    Args:
        domain: ONNX domain name (e.g., "qonnx.custom_op.general")

    Returns:
        List of (op_type, op_class) tuples

    Example:
        ::

            ops = get_ops_in_domain("qonnx.custom_op.general")
            for op_name, op_class in ops:
                print(f"{op_name}: {op_class}")

    """
    module_path = resolve_domain(domain)
    ops_dict = {}
    highest: Dict[str, int] = {}

    with _REGISTRY_LOCK:
        # Strategy 1: Get cached ops (fast path) - use highest version
        if domain in _OP_REGISTRY:
            for op_type, versions in _OP_REGISTRY[domain].items():
                if versions:
                    highest[op_type] = max(versions.keys())
                    ops_dict[op_type] = versions[highest[op_type]]

        # Strategy 2: Discover from module (for uncached ops), keeping the
        # highest version of each op type
        try:
            module = importlib.import_module(module_path)
            for name, obj in _exported_classes(module):
                op_type, version = op_identity(obj, exported_as=name)
                if version > highest.get(op_type, 0):
                    highest[op_type] = version
                    ops_dict[op_type] = obj
        except ModuleNotFoundError:
            pass  # Domain doesn't exist as module, return cached ops only

    return list(ops_dict.items())
