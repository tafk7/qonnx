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
# * Neither the name of AMD nor the names of its
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

"""Typed, namespaced graph metadata.

A namespace (``Namespace("finn.platform", version=1)``) declares keys, each
with a type: ``str``, ``int``, ``float``, ``bool``, an ``enum.Enum`` subclass
(stored by member name) or ``JSON`` (a JSON value). Each key is one entry of a
graph's ``metadata_props``, named ``<namespace>/<key>``, its value the type's
canonical text; the namespace's version is the entry ``<namespace>/@version``,
written with its keys. A graph stores each namespace at one version.

Nothing coerces. A stored value that does not parse as its key's type, a value
of the wrong type on writing, a stored key the namespace does not declare, a
namespace stored without a version or at a version the reader cannot upgrade
from raise ``MetadataError``, naming the entry, the text and the expectation.

Versions are upgraded, never downgraded: a namespace at version N may register
an upgrade from each earlier version (a function from the entries stored at
that version, key name to text, to those of the next). A reader upgrades what it
reads; a writer rewrites the namespace at the current version before writing.

The functions here operate on a ``metadata_props`` field (a graph's); the
``ModelWrapper`` methods ``get``, ``set``, ``delete`` and ``namespace`` are
the API. The untyped ``get_metadata_prop``/``set_metadata_prop`` keep working
for keys outside any namespace.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from onnx import StringStringEntryProto
from typing import Any, Callable, Generic, Iterable, MutableSequence, TypeVar

T = TypeVar("T")

SEPARATOR = "/"
VERSION = "@version"


class MetadataError(ValueError):
    """A metadata entry that cannot be read or written as its namespace declares."""


@dataclass(frozen=True)
class Codec(Generic[T]):
    """A key type: which values it admits, their canonical text, and how text is
    read back (``decode`` raises ``ValueError`` or ``KeyError`` on text that is
    not one of its values)."""

    expect: str
    admits: Callable[[object], bool]
    encode: Callable[[T], str]
    decode: Callable[[str], T]


def _decode_int(text: str) -> int:
    if not re.fullmatch(r"-?[0-9]+", text):
        raise ValueError(text)
    return int(text)


def _decode_float(text: str) -> float:
    if text != text.strip():
        raise ValueError(text)
    return float(text)


def _decode_bool(text: str) -> bool:
    return {"true": True, "false": False}[text]


def _is_json(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(name, str) and _is_json(item) for name, item in value.items())
    return False


def _decode_json(text: str) -> Any:
    def refuse(constant: str) -> Any:
        raise ValueError(constant)

    return json.loads(text, parse_constant=refuse)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


STR: Codec[str] = Codec("a str", lambda value: isinstance(value, str), str, str)
INT: Codec[int] = Codec("an int", lambda value: isinstance(value, int) and not isinstance(value, bool), str, _decode_int)
# An int is admitted where a float is declared (it is written as a float, and reads back as one).
FLOAT: Codec[float] = Codec("a float", _is_number, lambda value: repr(float(value)), _decode_float)
BOOL: Codec[bool] = Codec(
    "a bool (stored as true or false)",
    lambda value: isinstance(value, bool),
    lambda value: "true" if value else "false",
    _decode_bool,
)
# A JSON value: None, str, bool, int, finite float, and lists and str-keyed dicts of
# them (not tuples, which would read back as lists). Stored compact, keys sorted.
JSON: Codec[Any] = Codec(
    "a JSON value (None, str, bool, int, float, list, dict with str keys)",
    _is_json,
    lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False),
    _decode_json,
)


def enumeration(kind: type[Enum]) -> Codec[Any]:
    """The codec of an enumeration: a member, stored by its name."""
    return Codec(
        f"a {kind.__name__} ({', '.join(kind.__members__)})",
        lambda value: isinstance(value, kind),
        lambda value: value.name,
        lambda text: kind[text],
    )


_CODECS: dict[Any, Codec[Any]] = {str: STR, int: INT, float: FLOAT, bool: BOOL}


def _codec(kind: Any) -> Codec[Any]:
    if isinstance(kind, Codec):
        return kind
    if isinstance(kind, type) and issubclass(kind, Enum):
        return enumeration(kind)
    if kind in _CODECS:
        return _CODECS[kind]
    raise TypeError(f"a metadata key's type is str, int, float, bool, an Enum subclass or a Codec (JSON), not {kind!r}")


def _check_name(name: object, what: str) -> None:
    if not isinstance(name, str) or not name or SEPARATOR in name or name.startswith("@"):
        raise ValueError(f"a {what} name is a nonempty str without '{SEPARATOR}' that does not start with '@': {name!r}")


@dataclass(frozen=True, eq=False)
class Key(Generic[T]):
    """A declared key: its namespace, name and type. Made by ``Namespace.key``."""

    namespace: Namespace
    name: str
    codec: Codec[T]
    check: Callable[[T], bool] | None = None
    expect: str | None = None

    @property
    def entry(self) -> str:
        """The name of its ``metadata_props`` entry, ``<namespace>/<key>``."""
        return f"{self.namespace.name}{SEPARATOR}{self.name}"

    def _expectation(self) -> str:
        return self.codec.expect if self.check is None else f"{self.codec.expect}, {self.expect or 'admitted by its check'}"

    def encode(self, value: T) -> str:
        """The canonical text of a value; MetadataError if the key does not admit it."""
        if not self.codec.admits(value) or (self.check is not None and not self.check(value)):
            raise MetadataError(f"{self.entry}: cannot store {value!r}: expected {self._expectation()}")
        return self.codec.encode(value)

    def decode(self, text: str) -> T:
        """The value stored as ``text``; MetadataError if it is not one of the key's values."""
        try:
            value = self.codec.decode(text)
        except (ValueError, KeyError):
            raise MetadataError(f"{self.entry}: stored {text!r} is not {self._expectation()}") from None
        if self.check is not None and not self.check(value):
            raise MetadataError(f"{self.entry}: stored {text!r} is not {self._expectation()}")
        return value


class Namespace:
    """A versioned set of typed keys. ``inherit=True`` lets a subgraph body read a
    key it does not state itself from the graph it is a body of (see
    ``ModelWrapper.make_subgraph_modelwrapper``)."""

    def __init__(self, name: str, version: int = 1, inherit: bool = False) -> None:
        _check_name(name, "namespace")
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise ValueError(f"namespace {name}: a version is a positive int, not {version!r}")
        self.name = name
        self.version = version
        self.inherit = inherit
        self.keys: dict[str, Key[Any]] = {}
        self._upgrades: dict[int, Callable[[dict[str, str]], dict[str, str]]] = {}

    def __repr__(self) -> str:
        return f"Namespace({self.name!r}, version={self.version}, inherit={self.inherit})"

    def key(self, name: str, kind: Any, check: Callable[[Any], bool] | None = None, expect: str | None = None) -> Key[Any]:
        """Declare a key: its name, its type (``str``, ``int``, ``float``, ``bool``, an
        ``Enum`` subclass, or ``JSON``) and optionally a check on its values with
        ``expect`` describing what the check admits."""
        _check_name(name, "key")
        if name in self.keys:
            raise ValueError(f"namespace {self.name} declares key {name!r} twice")
        declared: Key[Any] = Key(self, name, _codec(kind), check, expect)
        self.keys[name] = declared
        return declared

    def upgrade(self, version: int, step: Callable[[dict[str, str]], dict[str, str]]) -> None:
        """Register the upgrade from ``version`` to ``version + 1``: a function from
        the entries stored at ``version`` (key name to text) to those of the next."""
        if not isinstance(version, int) or not 1 <= version < self.version:
            raise ValueError(f"namespace {self.name} (version {self.version}) cannot upgrade from {version!r}")
        if version in self._upgrades:
            raise ValueError(f"namespace {self.name} registers the upgrade from version {version} twice")
        self._upgrades[version] = step

    def _current(self, version: int, entries: dict[str, str]) -> dict[str, str]:
        """Stored entries brought from ``version`` to the current version."""
        if version > self.version or any(v not in self._upgrades for v in range(version, self.version)):
            raise MetadataError(
                f"{self.name}: stored at version {version}, which a reader of version {self.version} cannot upgrade from"
                + (f" (it upgrades from {sorted(self._upgrades)})" if self._upgrades else " (it has no upgrade)")
            )
        for v in range(version, self.version):
            entries = self._upgrades[v](dict(entries))
        return entries

    def _decoded(self, entries: dict[str, str]) -> dict[str, Any]:
        undeclared = sorted(set(entries) - set(self.keys))
        if undeclared:
            raise MetadataError(
                f"{self.name}: stored keys {undeclared} are not declared (version {self.version} declares {list(self.keys)})"
            )
        return {name: key.decode(entries[name]) for name, key in self.keys.items() if name in entries}


Props = MutableSequence[StringStringEntryProto]


def _stored(props: Iterable[StringStringEntryProto], namespace: Namespace) -> tuple[int | None, dict[str, str]]:
    """A namespace's stored version and entries (key name to text), as stored."""
    prefix = namespace.name + SEPARATOR
    version_text = None
    entries: dict[str, str] = {}
    for prop in props:
        if not prop.key.startswith(prefix):
            continue
        name = prop.key[len(prefix) :]
        if name == VERSION:
            if version_text is not None:
                raise MetadataError(f"{prop.key}: stored twice")
            version_text = prop.value
        elif name in entries:
            raise MetadataError(f"{prop.key}: stored twice")
        else:
            entries[name] = prop.value
    if version_text is None:
        if entries:
            raise MetadataError(f"{namespace.name}: keys {sorted(entries)} are stored without {prefix}{VERSION}")
        return None, entries
    if not re.fullmatch(r"[1-9][0-9]*", version_text):
        raise MetadataError(f"{prefix}{VERSION}: stored {version_text!r} is not a positive int")
    return int(version_text), entries


def read(props: Iterable[StringStringEntryProto], namespace: Namespace) -> dict[str, Any]:
    """The keys of ``namespace`` stored in ``props``, decoded (upgraded from an
    earlier stored version), in declaration order; MetadataError as the module
    describes."""
    version, entries = _stored(props, namespace)
    if version is None:
        return {}
    return namespace._decoded(namespace._current(version, entries))


def _rewrite(props: Props, namespace: Namespace, entries: dict[str, str]) -> None:
    """Replace the namespace's stored entries by ``entries`` at the current version
    (none at all when ``entries`` is empty)."""
    prefix = namespace.name + SEPARATOR
    for prop in [prop for prop in props if prop.key.startswith(prefix)]:
        props.remove(prop)
    if entries:
        props.append(StringStringEntryProto(key=prefix + VERSION, value=str(namespace.version)))
        for name, key in namespace.keys.items():
            if name in entries:
                props.append(StringStringEntryProto(key=key.entry, value=entries[name]))


def _current_entries(props: Props, namespace: Namespace) -> dict[str, str]:
    version, entries = _stored(props, namespace)
    if version is None:
        return {}
    entries = namespace._current(version, entries)
    namespace._decoded(entries)  # refuse a malformed namespace rather than rewrite it
    return entries


def write(props: Props, key: Key[T], value: T) -> None:
    """Store ``value`` under ``key`` (the namespace rewritten at the current version
    if it was stored at an earlier one)."""
    text = key.encode(value)
    namespace = key.namespace
    version, stored = _stored(props, namespace)
    if version == namespace.version:
        namespace._decoded(stored)
        for prop in props:
            if prop.key == key.entry:
                prop.value = text
                return
        props.append(StringStringEntryProto(key=key.entry, value=text))
        return
    entries = _current_entries(props, namespace)
    entries[key.name] = text
    _rewrite(props, namespace, entries)


def delete(props: Props, key: Key[Any]) -> None:
    """Remove ``key``'s entry, and the namespace's version with its last key."""
    entries = _current_entries(props, key.namespace)
    if key.name in entries:
        del entries[key.name]
        _rewrite(props, key.namespace, entries)


def merge(main: Iterable[StringStringEntryProto], other: Iterable[StringStringEntryProto]) -> list[StringStringEntryProto]:
    """The entries of two graphs merged into one: all of ``main``'s, then those of
    ``other`` that ``main`` does not state. A key both state with different text is
    taken from ``main``, unless it belongs to a typed namespace (one either graph
    stores with a version): then MetadataError, as the two graphs disagree on a fact
    (a namespace stored at two versions disagrees on its version entry)."""
    main, other = list(main), list(other)
    typed = {prop.key[: -len(SEPARATOR + VERSION)] for prop in main + other if prop.key.endswith(SEPARATOR + VERSION)}
    stated = {prop.key: prop.value for prop in main}
    merged = [StringStringEntryProto(key=prop.key, value=prop.value) for prop in main]
    for prop in other:
        if prop.key not in stated:
            merged.append(StringStringEntryProto(key=prop.key, value=prop.value))
            stated[prop.key] = prop.value
        elif stated[prop.key] != prop.value and prop.key.split(SEPARATOR, 1)[0] in typed:
            raise MetadataError(f"{prop.key}: the graphs merged state {stated[prop.key]!r} and {prop.value!r}")
    return merged


__all__ = [
    "BOOL",
    "Codec",
    "FLOAT",
    "INT",
    "JSON",
    "Key",
    "MetadataError",
    "Namespace",
    "STR",
    "delete",
    "enumeration",
    "merge",
    "read",
    "write",
]
