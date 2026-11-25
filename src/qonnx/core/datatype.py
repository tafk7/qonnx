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

"""QONNX datatypes: the value sets a tensor's ``finn_datatype`` annotation names.

Bounds
------
``bounds()`` is a datatype's exact least and greatest finite value: an ``int``, or a
``Fraction`` whose denominator is a power of two. ``min()`` and ``max()`` keep their
answers; for a fixed-point or float datatype they are floats, which may be rounded:
``FIXED<64,32>.max()`` is ``2147483648.0``, while ``FIXED<64,32>.bounds()[1]`` is
``Fraction(2**63 - 1, 2**32)``. A caller that needs exactness reads ``bounds()``.

``bounds()`` raises ``ValueError`` where a datatype has no exact bounds: a scaled integer,
which has no value set; a zero or negative width; and a datatype past these limits, the
envelope:

- ``INT<n>``, ``UINT<n>`` and ``SCALEDINT<n>``: at most 4096 bits;
- ``FIXED<w,i>``: at most 4096 bits, at most 1074 fraction bits (``w - i``) and at most
  1024 integer bits (``i``);
- ``FLOAT<e,m,b>``: at most 4096 bits, exponents within float64's
  (``2**e - 1 - b <= 1023`` and ``1 - b - m >= -1074``), and a greatest value whose
  nearest float64 is finite.

Float formats
-------------
``FloatFormat(mantissa, emin, emax)`` is a binary floating-point format as IEEE 754
defines one: its ``largest`` finite value, and ``nearest(value)``, an exact value rounded
once into the format by IEEE 754's round to nearest, ties to even. ``BINARY16``,
``BINARY32``, ``BINARY64`` and ``BFLOAT16`` are the formats of ONNX's float element types.
FLOAT16's and FLOAT32's bounds come from their formats' ``largest``.

allowed
-------
``allowed(value)`` keeps its meaning: for an integer, bipolar, ternary or fixed-point
datatype, whether a number is one of the datatype's values. For these datatypes it is
decided exactly for Python and numpy integers and floats of every width, and for arrays
of them, object arrays included. In an object array a bool counts as the integer it
equals, as it does on its own, and an element that is no number is not allowed. The float
datatypes' ``allowed`` is unchanged: FLOAT32 and FLOAT16 allow every value, and
FLOAT<e,m,b> keeps its float32-based test. Past the envelope, and for any other kind of
input, ``allowed`` answers as it always has.

Membership
----------
``contains(values)`` says whether each value is one of the datatype's, decided exactly
for Python ``int``, ``float`` and ``Fraction``, numpy integers and floats of every width,
and arrays of them, object arrays included. A scalar gives a ``bool``, an array a bool
array of its shape. Anything else (a ``bool``, a string, ``None``, a complex number, a
list) is not a member, and nothing raises. For an IEEE float datatype, membership is
representability: ``FLOAT32`` does not contain ``0.1``, and does contain each infinity and
NaN. No other datatype contains an infinity or NaN. A negative zero is the value zero. A
datatype without exact bounds (a scaled integer, a nonsense width, a datatype past the
envelope) contains nothing. ``allowed`` keeps its own answers, which differ from these for
the float datatypes and for bools.

Codes and values
----------------
``codes(values)`` gives each value as the ``bitwidth()``-wide pattern the hardware carries,
as ``uint64``; ``values(codes)`` gives each code's value back. Both are exact:
``codes`` raises ``ValueError`` for a value the datatype does not contain, and ``values``
for a code wider than the datatype or one no value has (TERNARY's 2);
``values(codes(v))`` equals ``v`` for every member ``v``. An integer's code is its two's
complement in ``bitwidth()`` bits; BINARY's is its value; BIPOLAR's -1 and +1 are 0 and
1; TERNARY's values are two's complement in two bits; a fixed point's code is its value
times ``2**(w - i)`` in two's complement; an IEEE float's is its bit pattern, a negative
zero's sign included. An IEEE float value given in its own container is coded by its bits
as they are, a NaN's payload included; one given in another type is converted first,
exactly for every value but a NaN, which becomes a NaN of its sign. Codes exist for
integer, bipolar, ternary and fixed-point datatypes of at most 64 bits and for FLOAT16 and
FLOAT32; elsewhere both methods raise ``ValueError``.

Containers
----------
``container()`` is the numpy type ``codes`` takes values in and ``values`` returns them in,
exact for every value; ``ValueError`` where there is none. An integer's is the smallest
numpy integer type of at most 64 bits that holds it; BIPOLAR's and TERNARY's ``int8``;
FLOAT16's and FLOAT32's their own; a FLOAT<e,m,b>'s ``float32`` where float32 holds every
value. A FIXED<w,i> of at most 54 bits is ``float64``, exact because its codes fit
float64's significand and its scale fits float64's exponents. A wider one, to 64 bits, is
carried as its codes (``carried_as_codes()``): ``int64`` holding the value times
``2**(w - i)``, given to ``codes`` and returned by ``values`` alike. ``to_numpy_dt()``
keeps its answers (``float32`` for every fixed point); ``FIXED<32,16>.values(16777217)``
is ``256.0000152587890625``, where float32 holds only ``256.0``.

Rounded conversion
------------------
``round_to(target, values)`` rounds each value to the nearest one an IEEE float datatype
(FLOAT16 or FLOAT32) holds, by IEEE 754's round to nearest, ties to even, applied once to
the exact value. A magnitude at least the largest finite value plus half its spacing
becomes the infinity of its sign; an infinity stays; a NaN stays a NaN, its payload not
kept; a zero keeps its sign. Only an IEEE float datatype is a target. Membership and codes
never round; rounding is only ever asked for.

Names
-----
``resolve_datatype(name)`` and ``DataType[name]`` read a name as a source spells it:
``UINT1`` is ``BINARY``, ``FLOAT<5,10>`` is ``FLOAT<5,10,15>`` and ``FIXED<8, 4>`` is
``FIXED<8,4>``. A parameter's leading zeros are not significant, however many there are.
An explicit exponent bias of 0 is kept: ``FLOAT<5,10,0>`` is not ``FLOAT<5,10,15>``.

A name with a zero or negative width or parameter (``INT0``, ``INT-3``, ``FLOAT<0,3>``),
or one past the envelope (``INT5000``; see "Bounds"), still resolves, with a
``DataTypeWarning``. It is a ``FutureWarning``, shown by default, because such names
reach users through their model files rather than their code. A later release refuses
them.

``resolve_datatype(name, canonical=True)`` resolves only the canonical name of a datatype
within the envelope, the one ``get_canonical_name()`` gives, as a stored annotation holds
it. Any other name raises ``KeyError``: another spelling (``UINT1``, ``FLOAT<5,10>``), a
nonsense width, a name past the envelope, or a malformed name.
"""

import math
import numpy as np
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum, EnumMeta
from fractions import Fraction
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Optional, Protocol, Union

from qonnx.core import Refusal


class DataTypeWarning(FutureWarning):
    """A datatype name was resolved that a later release refuses: a zero or negative
    width or parameter, or a datatype past the envelope (see the module's docstring)."""


# The envelope: the widest INT, UINT, FIXED, SCALEDINT or FLOAT<e,m,b>, and a FIXED's
# most fraction bits and integer bits (float64's exponents).
_MAX_BITS = 4096
_MAX_FRACTION_BITS = 1074
_MAX_INTEGER_BITS = 1024


# Exact numbers: a number read as an exact rational, a range's endpoint, a binary
# floating-point format and its rounding, and the float64 tests that decide membership on
# arrays without leaving float64.

# A range's end: an int, a ``Fraction`` whose denominator is a power of two (not 1), or an
# infinity.
_Endpoint = Union[int, Fraction, float]

# A number read exactly: an int, a non-integer ``Fraction``, or a float infinity or NaN.


class NumericRefused(Refusal):
    """The family of datatype, name and numeric-value refusals."""


class NumericValueRefused(NumericRefused, ValueError):
    """A value outside a numeric operation's domain."""


class DataTypeNameRefused(NumericRefused):
    """The family of datatype-name refusals."""


class DataTypeNameUnknown(DataTypeNameRefused, KeyError):
    """A datatype name whose family is unknown."""


class DataTypeNameMalformed(DataTypeNameRefused, ValueError):
    """A datatype name whose parameters cannot be read."""


_Exact = Union[int, Fraction, float]


def _rational(value: Union[int, Fraction]) -> Union[int, Fraction]:
    """``value`` as an int when it is an integer."""
    if isinstance(value, Fraction) and value.denominator == 1:
        return value.numerator
    return value


def _exact(value: object) -> Optional[_Exact]:
    """``value`` read exactly: a Python or numpy integer or float, or a ``Fraction``.
    ``None`` for anything else (a ``bool``, a string, ``None``, a complex number)."""
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return math.nan
        if np.isinf(value):
            return math.inf if value > 0 else -math.inf
        numerator, denominator = value.as_integer_ratio()
        return _rational(Fraction(numerator, denominator))
    if isinstance(value, Fraction):
        return _rational(value)
    return None


def _finite(value: Optional[_Exact]) -> bool:
    """Whether an exactly read value is a finite number."""
    return isinstance(value, (int, Fraction))


def _endpoint(value: Union[int, float, Fraction]) -> _Endpoint:
    """``value`` as a canonical endpoint: an int if it is an integer, else a dyadic
    ``Fraction``, else an infinity. ``ValueError`` for NaN or a non-dyadic ``Fraction``;
    ``TypeError`` for anything that is no number."""
    if isinstance(value, bool):
        raise TypeError(f"an endpoint is a number, not {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            raise NumericValueRefused.of("range-invalid", "endpoint", "an endpoint is not NaN")
        if math.isinf(value):
            return value
        value = Fraction(value)
    if isinstance(value, Fraction):
        if value.denominator & (value.denominator - 1):
            raise NumericValueRefused.of("range-invalid", "endpoint", f"an endpoint is dyadic (m * 2**e), not {value}")
        return _rational(value)
    raise TypeError(f"an endpoint is an int, a dyadic Fraction or an infinity, not {value!r}")


@dataclass(frozen=True)
class FloatFormat:
    """A binary floating-point format as IEEE 754 defines one: a sign, ``mantissa``
    fraction bits, normal exponents ``emin`` to ``emax``, subnormals below them and an
    infinity of each sign beyond its largest finite value."""

    mantissa: int
    emin: int
    emax: int

    def __post_init__(self) -> None:
        for value in (self.mantissa, self.emin, self.emax):
            if type(value) is not int:
                raise TypeError(f"a float format's parameters are ints, not {value!r}")
        if self.mantissa < 0 or self.emin > self.emax:
            raise NumericValueRefused.of(
                "format-invalid",
                "FloatFormat",
                f"no binary float format has {self.mantissa} fraction bits and normal "
                f"exponents {self.emin} to {self.emax}",
            )

    @property
    def largest(self) -> Union[int, Fraction]:
        """The largest finite value, (2 - 2**-mantissa) * 2**emax."""
        return _rational((2 - Fraction(1, 2**self.mantissa)) * Fraction(2) ** self.emax)

    def nearest(self, value: Union[int, Fraction, float]) -> _Endpoint:
        """``value``, an exact rational or an infinity, rounded once into the format as
        IEEE 754's round to nearest, ties to even, does: past the largest finite value by
        half its spacing or more, to the infinity of its sign. A zero result is 0 (an
        endpoint does not tell -0 from +0). ``ValueError`` for NaN."""
        if isinstance(value, float):
            if math.isnan(value):
                raise NumericValueRefused.of("format-invalid", "FloatFormat.nearest", "NaN has no nearest value in a format")
            if math.isinf(value):
                return value
        value = Fraction(value)
        if value == 0:
            return 0
        size = abs(value)
        exponent = size.numerator.bit_length() - size.denominator.bit_length()
        if size < Fraction(2) ** exponent:
            exponent -= 1
        spacing = Fraction(2) ** (max(exponent, self.emin) - self.mantissa)
        rounded = round(size / spacing) * spacing  # round() of a Fraction: ties to even
        if rounded > self.largest:
            return math.inf if value > 0 else -math.inf
        return _endpoint(rounded if value > 0 else -rounded)


# The formats of ONNX's float element types.
BINARY16 = FloatFormat(10, -14, 15)
BINARY32 = FloatFormat(23, -126, 127)
BINARY64 = FloatFormat(52, -1022, 1023)
BFLOAT16 = FloatFormat(7, -126, 127)


def _at_most(bound: Union[int, Fraction]) -> float:
    """The greatest float64 not above ``bound`` (``-inf`` when there is none)."""
    try:
        found = float(bound)
    except OverflowError:
        found = math.inf if bound > 0 else -math.inf
    while found > -math.inf and (found == math.inf or Fraction(found) > bound):
        found = math.nextafter(found, -math.inf)
    return found


def _at_least(bound: Union[int, Fraction]) -> float:
    """The least float64 not below ``bound`` (``inf`` when there is none)."""
    return -_at_most(-bound)


@lru_cache(maxsize=4096)
def _at_most_in(dtype: np.dtype, bound: Union[int, Fraction]) -> np.floating:
    """The greatest value of the float type ``dtype`` (at most 64 bits) not above ``bound``
    (``-inf`` when there is none). Kept per type and bound: allowed() asks for the same
    ones on every call."""
    wide = _at_most(bound)
    with np.errstate(over="ignore"):
        found = dtype.type(wide)
    if float(found) > wide:
        found = np.nextafter(found, dtype.type(-np.inf))
    return found


def _at_least_in(dtype: np.dtype, bound: Union[int, Fraction]) -> np.floating:
    """The least value of the float type ``dtype`` (at most 64 bits) not below ``bound``
    (``inf`` when there is none)."""
    return -_at_most_in(dtype, -bound)


def _float64_exact(bound: Union[int, Fraction]) -> bool:
    """Whether float64 holds ``bound`` exactly (and so does every wider float type)."""
    try:
        return float(bound) == bound
    except OverflowError:
        return False


@lru_cache(maxsize=4096)
def _fixed_point_bounds(bits: int, fraction: int) -> tuple[Union[int, Fraction], Union[int, Fraction]]:
    """FIXED's exact bounds for a width and its fraction bits, kept: building them costs
    more than allowed() on a small array."""
    top, scale = 1 << (bits - 1), 1 << fraction
    return _rational(Fraction(-top, scale)), _rational(Fraction(top - 1, scale))


@lru_cache(maxsize=4096)
def _fixed_point_limits(dtype: np.dtype, bits: int, fraction: int) -> tuple[np.floating, np.floating]:
    """FIXED's bounds in the float type ``dtype``, kept per type and FIXED."""
    least, greatest = _fixed_point_bounds(bits, fraction)
    return _at_least_in(dtype, least), _at_most_in(dtype, greatest)


@lru_cache(maxsize=4096)
def _scale_in(dtype: np.dtype, bits: int, fraction: int) -> Optional[np.floating]:
    """2**fraction in the float type ``dtype``, when it is a normal value there and the
    codes of a ``bits``-wide FIXED are all finite in it; else None."""
    info = np.finfo(dtype)
    if bits >= info.maxexp or not info.minexp <= fraction < info.maxexp:
        return None
    return dtype.type(2.0**fraction)


def _multiples(values: np.ndarray, power: Union[int, np.ndarray]) -> np.ndarray:
    """Whether each finite float64 is an integer multiple of ``2**power``.

    A float is ``m * 2**e`` with ``m`` in [0.5, 1) of 53 bits (``frexp``): a multiple of
    ``2**q`` exactly when ``m * 2**(e - q)`` is an integer, which it is from ``e - q =
    53`` and is not at ``e - q <= 0`` unless ``m`` is 0. No product leaves float64.
    """
    mantissa, exponent = np.frexp(values)
    shift = np.clip(exponent.astype(np.int64) - power, 0, 53)
    scaled = np.ldexp(mantissa, shift.astype(np.int32))
    return np.floor(scaled) == scaled


def _leading_exponents(values: np.ndarray) -> np.ndarray:
    """Each finite nonzero float64's leading bit's exponent: ``2**k <= |x| < 2**(k + 1)``."""
    return np.frexp(values)[1].astype(np.int64) - 1


def _valued_width(datatype: "BaseDataType", bits: int) -> int:
    """``bits`` when it is a width the envelope holds; ``ValueError`` otherwise. (The
    message names the class and width alone: a huge width's name is costly to build.)"""
    family = type(datatype).__name__
    if bits < 1:
        raise NumericValueRefused.of("undefined", "DataType.bounds", f"{family} with a width of {bits} bits has no values")
    if bits > _MAX_BITS:
        raise NumericValueRefused.of(
            "beyond-envelope",
            "DataType.bounds",
            f"{family} with a width of {bits} bits is past the envelope: wider than {_MAX_BITS} bits",
        )
    return int(bits)


def _keeps_range(datatype: "BaseDataType", family: type) -> bool:
    """Whether a datatype keeps its family's min() and max(). A subclass defined elsewhere
    that redefines them states its own range, and its bounds() reads them exactly."""
    return type(datatype).min is family.min and type(datatype).max is family.max


def _each(values: np.ndarray, test: Callable[[object], bool]) -> np.ndarray:
    """``test`` of each element, as a bool array of ``values``' shape."""
    found = np.fromiter((test(value) for value in values.flat), np.bool_, values.size)
    return found.reshape(values.shape)


def _between(values: np.ndarray, least: Union[int, Fraction], greatest: Union[int, Fraction]) -> np.ndarray:
    """Whether each element of an integer array lies in ``[least, greatest]``, exactly:
    both ends are first rounded inward to integers and brought within the array's own
    integer type."""
    info = np.iinfo(values.dtype)
    least, greatest = max(math.ceil(least), int(info.min)), min(math.floor(greatest), int(info.max))
    if least > greatest:
        return np.zeros(values.shape, dtype=np.bool_)
    kind = values.dtype.type
    # An end at the type's own limit holds for every element: no comparison for it.
    if least == info.min:
        if greatest == info.max:
            return np.ones(values.shape, dtype=np.bool_)
        return values <= kind(greatest)
    if greatest == info.max:
        return values >= kind(least)
    return (values >= kind(least)) & (values <= kind(greatest))


def _allowed_read(value: object) -> Optional[_Exact]:
    """An element of an object array as ``allowed`` reads it: a bool as the integer it
    equals, a number exactly, anything else as no number (None)."""
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    return _exact(value)


def _exact_allowed(datatype: "BaseDataType", value: object) -> Optional[np.ndarray]:
    """``allowed``'s answer decided exactly, as a bool array of ``value``'s shape (0-d for
    a scalar). None for what is left to the original test: a datatype past the envelope,
    and any input but a Python or numpy integer or float, or an array of them or of
    objects."""
    try:
        datatype.bounds()
    except ValueError:
        return None
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            return _each(value, lambda element: datatype._holds(_allowed_read(element)))
        if value.dtype.kind in "iuf":
            return np.asarray(datatype._contains_array(value))
        return None
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.asarray(datatype._holds(_exact(value)))
    return None


def _rounded_allowed(least: Union[int, float], greatest: Union[int, float], value: object) -> object:
    """``IntType.allowed``'s original test of an integer range, for a fixed point's inputs
    that are not decided exactly: rounding leaves the value as it is, and it lies within
    the range as numpy compares them."""
    value_is_integer = np.round(value) == value
    value_is_bounded = np.logical_and(least <= value, value <= greatest)
    return np.logical_and(value_is_integer, value_is_bounded)


def _is_integral(value: object) -> bool:
    """Whether ``value`` is a Python or numpy integer (not a bool) or an integer array:
    the inputs ``np.isin`` mishandles for an unsigned 64-bit type on numpy 1.x."""
    if isinstance(value, np.ndarray):
        return value.dtype.kind in "iu"
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _in_format(fmt: FloatFormat, value: Optional[_Exact]) -> bool:
    """Whether an exactly read value is a finite value of a binary float format."""
    return _finite(value) and fmt.nearest(value) == value


def _representable(fmt: FloatFormat, values: np.ndarray) -> np.ndarray:
    """Whether each element of a float64 array is a finite value of a binary float
    format: within its largest value, and a multiple of the spacing at its exponent."""
    spacing = np.maximum(_leading_exponents(values), fmt.emin) - fmt.mantissa
    return np.isfinite(values) & (np.abs(values) <= _at_most(fmt.largest)) & _multiples(values, spacing)


def _float_integers(datatype: "BaseDataType", values: np.ndarray) -> np.ndarray:
    """Membership of each element of an integer array in a float datatype: through float64
    where every element converts exactly, else element by element."""
    if values.size and max(-int(values.min()), int(values.max())) > 1 << 53:
        return _each(values, lambda value: datatype._holds(_exact(value)))
    with np.errstate(invalid="ignore", over="ignore"):
        return datatype._floats(values.astype(np.float64))


def _as_array(values: object) -> np.ndarray:
    """``values`` as an array: an array as it is, a numpy scalar as a 0-d array of its
    type, anything else as a 0-d object array holding it."""
    if isinstance(values, np.ndarray):
        return values
    if isinstance(values, np.generic):
        return np.asarray(values)
    found = np.empty((), dtype=object)
    found[()] = values
    return found


def _code_array(codes: object) -> np.ndarray:
    """Codes as uint64: integers from 0 below 2**64, of any integer type; ValueError for
    anything else."""
    array = _as_array(codes)
    if array.dtype.kind == "u":
        return array.astype(np.uint64)
    if array.dtype.kind == "i":
        if (array < 0).any():
            raise NumericValueRefused.of("code-outside", "DataType.values", "a code is not negative")
        return array.astype(np.uint64)
    if array.dtype.kind == "O":
        found = np.empty(array.shape, dtype=np.uint64)
        for index, code in np.ndenumerate(array):
            value = _exact(code)
            if not isinstance(value, int) or isinstance(code, (float, np.floating, Fraction)):
                raise NumericValueRefused.of("code-outside", "DataType.values", f"a code is an integer, not {code!r}")
            if not 0 <= value < 1 << 64:
                raise NumericValueRefused.of("code-outside", "DataType.values", f"{code} is no 64-bit code")
            found[index] = value
        return found
    raise NumericValueRefused.of("code-outside", "DataType.values", f"codes are integers, not {array.dtype}")


def _signed_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """``bits``-wide codes read as two's complement, as int64."""
    if bits == 64:
        return codes.view(np.int64)
    values = codes.astype(np.int64)
    return values - ((values >> (bits - 1)) << bits)


def _from_ints(values: np.ndarray, code: Callable[[Union[int, Fraction]], int]) -> np.ndarray:
    """The codes of an array of members given as objects, one at a time."""
    found = np.empty(values.shape, dtype=np.uint64)
    for index, value in np.ndenumerate(values):
        found[index] = code(_exact(value))
    return found


def _integer_codes(values: np.ndarray, bits: int, signed: bool) -> np.ndarray:
    """The two's complement codes, ``bits`` wide, of an array of integer members."""
    mask = (1 << bits) - 1
    kind = values.dtype.kind
    if kind == "u":
        return values.astype(np.uint64) & np.uint64(mask)
    if kind == "i" or (kind == "f" and values.dtype.itemsize <= 8 and signed):
        return values.astype(np.int64).view(np.uint64) & np.uint64(mask)
    if kind == "f" and values.dtype.itemsize <= 8:
        return values.astype(np.uint64)
    return _from_ints(values, lambda value: int(value) & mask)


def _ieee_codes(container: np.dtype, values: np.ndarray) -> np.ndarray:
    """An IEEE float's codes: a value given in the container by its bits, a NaN's payload
    and a zero's sign included; one given otherwise converted, exactly for a member, a NaN
    to a NaN of its sign."""
    pattern = np.uint16 if container == np.float16 else np.uint32
    if values.dtype == container:
        return values.view(pattern).astype(np.uint64)
    if values.dtype.kind in "iuf" and values.dtype.itemsize <= 8:
        with np.errstate(invalid="ignore"):
            converted = values.astype(np.float64).astype(container)
        return converted.view(pattern).astype(np.uint64)
    found = np.empty(values.shape, dtype=np.uint64)
    for index, value in np.ndenumerate(values):
        if isinstance(value, np.floating) and value.dtype == container:
            found[index] = int(np.asarray(value).view(pattern))
            continue
        number = float(value) if isinstance(value, (float, np.floating)) else float(_exact(value))
        with np.errstate(invalid="ignore"):
            found[index] = int(np.asarray(number).astype(container).view(pattern))
    return found


def _ieee_values(container: np.dtype, codes: np.ndarray) -> np.ndarray:
    """An IEEE float's values of its bit patterns."""
    pattern = np.uint16 if container == np.float16 else np.uint32
    return codes.astype(pattern).view(container)


class BaseDataType(ABC):
    "Base class for QONNX data types."

    def signed(self) -> bool:
        "Returns whether this DataType can represent negative numbers."
        return self.min() < 0

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BaseDataType):
            return self.get_canonical_name() == other.get_canonical_name()
        elif isinstance(other, str):
            return self.get_canonical_name() == other
        else:
            return NotImplemented

    def __hash__(self) -> int:
        return hash(self.get_canonical_name())

    @property
    def name(self) -> str:
        return self.get_canonical_name()

    def __repr__(self) -> str:
        return self.get_canonical_name()

    def __str__(self) -> str:
        return self.get_canonical_name()

    @abstractmethod
    def bitwidth(self) -> int:
        "Returns the number of bits required for this DataType."
        pass

    @abstractmethod
    def min(self) -> Union[int, float]:
        "Returns the smallest possible value allowed by this DataType."
        pass

    @abstractmethod
    def max(self) -> Union[int, float]:
        "Returns the largest possible value allowed by this DataType."
        pass

    def bounds(self) -> tuple[Union[int, Fraction], Union[int, Fraction]]:
        """Returns the exact least and greatest finite value: an int, or a Fraction whose
        denominator is a power of two. Raises ValueError where the datatype has no exact
        bounds (see the module's docstring). A subclass that does not define its own
        reads min() and max() exactly."""
        least, greatest = _exact(self.min()), _exact(self.max())
        if not (_finite(least) and _finite(greatest)):
            raise NumericValueRefused.of("undefined", "DataType.bounds", f"{self} has no finite bounds")
        return least, greatest

    def contains(self, values: object) -> Union[bool, np.ndarray]:
        """Returns whether each value is one of this datatype's values, decided exactly: a
        bool for a scalar, a bool array of an array's shape. A value that is no number is
        not a member, and nothing raises (see the module's docstring). A subclass defined
        elsewhere states its membership by overriding this method."""
        try:
            self.bounds()
        except ValueError:
            return np.zeros(values.shape, dtype=np.bool_) if isinstance(values, np.ndarray) else False
        if isinstance(values, np.ndarray):
            return np.asarray(self._contains_array(values), dtype=np.bool_).reshape(values.shape)
        return bool(self._holds(_exact(values)))

    def codes(self, values: object) -> np.ndarray:
        """Returns each value as the bitwidth()-wide pattern the hardware carries, as uint64
        (see the module's docstring). Raises ValueError for a value this datatype does not
        contain, and for a datatype that has no codes."""
        self._require_codes()
        array = _as_array(values)
        members = self._container_members(array)
        if not members.all():
            bad = array.flat[int(np.flatnonzero(~members.reshape(-1))[0])]
            raise NumericValueRefused.of("not-member", "DataType.codes", f"{bad!r} is not a {self} value")
        return self._encode(array)

    def values(self, codes: object) -> np.ndarray:
        """Returns each bitwidth()-wide code's value, in container(). Raises ValueError for
        a code wider than bitwidth() or one no value has, and for a datatype that has no
        codes."""
        self._require_codes()
        raw = _code_array(codes)
        bits = int(self.bitwidth())
        if bits < 64 and (raw >> np.uint64(bits)).any():
            raise NumericValueRefused.of("code-outside", "DataType.values", f"a code wider than {self}'s {bits} bits")
        return self._decode(raw)

    def container(self) -> np.dtype:
        """Returns the numpy type codes() takes values in and values() returns them in,
        exact for every value. Raises ValueError where there is none. to_numpy_dt() keeps
        its answers."""
        raise NumericValueRefused.of("undefined", "DataType.container", f"{self} has no exact numpy container")

    def carried_as_codes(self) -> bool:
        """Returns whether container() holds this datatype's codes, not its values (a
        FIXED wider than 54 bits): codes() takes and values() gives its values so."""
        return False

    def _require_codes(self) -> None:
        """Raises ValueError for a datatype that has no codes."""
        raise NumericValueRefused.of("undefined", "DataType.codes", f"{self} has no codes")

    def _container_members(self, values: np.ndarray) -> np.ndarray:
        """Whether each element is a value codes() takes."""
        return np.asarray(self.contains(values))

    def _encode(self, values: np.ndarray) -> np.ndarray:
        """The codes of an array of members."""
        raise NotImplementedError

    def _decode(self, codes: np.ndarray) -> np.ndarray:
        """The values of an array of uint64 codes, each at most bitwidth() bits wide."""
        raise NotImplementedError

    # Exact membership: the private tests allowed() runs on. A family defines _holds, and
    # may define _integers and _floats to decide whole arrays without leaving numpy.

    def _holds(self, value: Optional[_Exact]) -> bool:
        """Whether one exactly read value (None for no number) is a member."""
        raise NotImplementedError(f"{type(self).__name__} decides no membership exactly")

    def _contains_array(self, values: np.ndarray) -> np.ndarray:
        """Whether each element of an array is a member, exactly, as a bool array of its
        shape: integer and float arrays as numbers, an object array element by element;
        an array of any other kind holds no member."""
        kind = values.dtype.kind
        if kind in "iu":
            return self._integers(values)
        if kind == "f":
            with np.errstate(invalid="ignore", over="ignore", under="ignore"):
                if values.dtype.itemsize <= 8:
                    # float16 arithmetic is slow in numpy; float32 holds its values.
                    if values.dtype.itemsize < 4:
                        values = values.astype(np.float32)
                    return np.asarray(self._floats(values), dtype=np.bool_)
                return np.asarray(self._wide_floats(values), dtype=np.bool_)
        if kind == "O":
            return _each(values, lambda value: self._holds(_exact(value)))
        return np.zeros(values.shape, dtype=np.bool_)

    def _integers(self, values: np.ndarray) -> np.ndarray:
        """Membership of each element of an integer array."""
        return _each(values, lambda value: self._holds(_exact(value)))

    def _floats(self, values: np.ndarray) -> np.ndarray:
        """Membership of each element of a float array of at most 64 bits."""
        return _each(values, lambda value: self._holds(_exact(value)))

    def _wide_floats(self, values: np.ndarray) -> np.ndarray:
        """Membership of each element of a float array wider than 64 bits (longdouble):
        the elements float64 holds exactly are decided with it, the rest one at a time."""
        narrow = values.astype(np.float64)
        found = np.asarray(self._floats(narrow), dtype=np.bool_)
        kept = (narrow.astype(values.dtype) == values) | np.isnan(narrow)
        if not kept.all():
            found = found.copy()
            found[~kept] = _each(values[~kept], lambda value: self._holds(_exact(value)))
        return found

    @abstractmethod
    def allowed(self, value: Union[int, float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Check whether given value is allowed for this DataType.

        * value (float32): value to be checked
        * returns: bool if input is scalar, ndarray of bools if input is ndarray"""
        pass

    @abstractmethod
    def get_num_possible_values(self) -> int:
        """Returns the number of possible values this DataType can take. Only
        implemented for integer types for now."""
        pass

    @abstractmethod
    def is_integer(self) -> bool:
        "Returns whether this DataType represents integer values only."
        pass

    @abstractmethod
    def is_fixed_point(self) -> bool:
        "Returns whether this DataType represent fixed-point values only."
        pass

    @abstractmethod
    def get_hls_datatype_str(self) -> str:
        "Returns the corresponding Vivado HLS datatype name."
        pass

    @abstractmethod
    def to_numpy_dt(self) -> np.dtype:
        "Return an appropriate numpy datatype that can represent this QONNX DataType."
        pass

    @abstractmethod
    def get_canonical_name(self) -> str:
        "Return a canonical string representation of this QONNX DataType."


class FloatType(BaseDataType):
    def bitwidth(self) -> int:
        return 32

    def min(self) -> float:
        return float(np.finfo(np.float32).min)

    def max(self) -> float:
        return float(np.finfo(np.float32).max)

    def bounds(self) -> tuple[Union[int, Fraction], Union[int, Fraction]]:
        largest = BINARY32.largest
        return -largest, largest

    def _holds(self, value: Optional[_Exact]) -> bool:
        return isinstance(value, float) or _in_format(BINARY32, value)

    def _integers(self, values: np.ndarray) -> np.ndarray:
        return _float_integers(self, values)

    def _floats(self, values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float64, copy=False)
        return ~np.isfinite(values) | _representable(BINARY32, values)

    def container(self) -> np.dtype:
        return np.dtype(np.float32)

    def _require_codes(self) -> None:
        pass

    def _encode(self, values: np.ndarray) -> np.ndarray:
        return _ieee_codes(self.container(), values)

    def _decode(self, codes: np.ndarray) -> np.ndarray:
        return _ieee_values(self.container(), codes)

    def allowed(self, value: Union[int, float, np.ndarray]) -> Union[bool, np.ndarray]:
        if isinstance(value, np.ndarray):
            return np.ones(value.shape, dtype=bool)
        return True

    def get_num_possible_values(self) -> int:
        raise NumericValueRefused.of("undefined", "DataType.get_num_possible_values", "Undefined for FloatType")

    def is_integer(self) -> bool:
        return False

    def is_fixed_point(self) -> bool:
        return False

    def get_hls_datatype_str(self) -> str:
        return "float"

    def to_numpy_dt(self) -> np.dtype:
        return np.dtype(np.float32)

    def get_canonical_name(self) -> str:
        return "FLOAT32"


class ArbPrecFloatType(BaseDataType):
    def __init__(
        self,
        exponent_bits: int,
        mantissa_bits: int,
        exponent_bias: Union[int, float, None] = None,
    ) -> None:
        self._exponent_bits = exponent_bits
        self._mantissa_bits = mantissa_bits

        if exponent_bias is None:
            # default (IEEE-style) exponent bias; an explicit 0 is a bias of 0
            exponent_bias = (2.0 ** (exponent_bits - 1)) - 1
        self._exponent_bias = exponent_bias

    def signed(self) -> bool:
        "Returns whether this DataType can represent negative numbers."
        return True

    def bitwidth(self) -> int:
        # sign bit + exponent bits + mantissa bits
        return 1 + self.exponent_bits() + self.mantissa_bits()

    def exponent_bits(self) -> int:
        return self._exponent_bits

    def mantissa_bits(self) -> int:
        return self._mantissa_bits

    def exponent_bias(self) -> Union[int, float]:
        return self._exponent_bias

    def min(self) -> float:
        return -1 * self.max()

    def max(self) -> float:
        # note: assumes no bits reserved for NaN/inf etc.
        exponent_bias = self.exponent_bias()
        exponent_bitwidth = self.exponent_bits()
        mantissa_bitwidth = self.mantissa_bits()
        max_exponent = (2.0**exponent_bitwidth) - 1.0 - exponent_bias
        max_mantissa = np.sum(
            (2.0 ** np.arange(0, -1.0 * mantissa_bitwidth - 1.0, -1.0))
        )
        max_val = max_mantissa * (2**max_exponent)
        return float(max_val)
    
    def bounds(self) -> tuple[Union[int, Fraction], Union[int, Fraction]]:
        largest = self._format().largest
        return -largest, largest

    def _format(self) -> FloatFormat:
        """The binary float format of this datatype's values, every bit pattern a finite
        value; ValueError for parameters that denote none, or past the envelope."""
        exponent_bits, mantissa_bits = int(self.exponent_bits()), int(self.mantissa_bits())
        bias = self.exponent_bias()
        if isinstance(bias, float) and bias.is_integer():
            bias = int(bias)
        if exponent_bits < 1 or mantissa_bits < 0 or not isinstance(bias, (int, np.integer)):
            raise NumericValueRefused.of(
                "undefined",
                "DataType.bounds",
                f"{self} has no values: exponent bits >= 1, mantissa bits >= 0, an integer bias",
            )
        _valued_width(self, self.bitwidth())
        bias = int(bias)
        emax = (1 << exponent_bits) - 1 - bias
        if emax > 1023 or 1 - bias - mantissa_bits < -1074:
            raise NumericValueRefused.of(
                "beyond-envelope", "DataType.bounds", f"{self} is past the envelope: its exponents are beyond float64's"
            )
        found = FloatFormat(int(mantissa_bits), 1 - bias, emax)
        try:
            float(found.largest)
        except OverflowError:
            raise NumericValueRefused.of(
                "beyond-envelope",
                "DataType.bounds",
                f"{self} is past the envelope: its greatest value rounds past float64's",
            ) from None
        return found

    def _holds(self, value: Optional[_Exact]) -> bool:
        return _in_format(self._format(), value)

    def _integers(self, values: np.ndarray) -> np.ndarray:
        return _float_integers(self, values)

    def _floats(self, values: np.ndarray) -> np.ndarray:
        return _representable(self._format(), values.astype(np.float64, copy=False))

    def container(self) -> np.dtype:
        fmt = self._format()
        if fmt.mantissa <= 23 and fmt.emax <= 127 and fmt.emin - fmt.mantissa >= -149:
            return np.dtype(np.float32)
        raise NumericValueRefused.of(
            "undefined", "DataType.container", f"{self} has no exact numpy container: float32 does not hold every value"
        )

    def allowed(self, value: Union[int, float, np.ndarray]) -> Union[bool, np.ndarray]:
        # fp32 format parameters
        fp32_exponent_bias = 127
        fp32_mantissa_bitwidth = 23
        fp32_nrm_mantissa_bitwidth = fp32_mantissa_bitwidth + 1  # width of normalized mantissa with implicit 1
        # minifloat format parameters
        exponent_bias = self.exponent_bias()
        min_exponent = -exponent_bias + 1  # minimum exponent if IEEE-style denormals are supported
        mantissa_bitwidth = self.mantissa_bits()
        nrm_mantissa_bitwidth = mantissa_bitwidth + 1  # width of normalized mantissa with implicit 1
        # extract fields from fp32 representation
        bin_val = np.float32(value).view(np.uint32)
        exp = (bin_val & 0b01111111100000000000000000000000) >> fp32_mantissa_bitwidth
        mant = bin_val & 0b00000000011111111111111111111111
        exp_biased = np.array(exp).astype(int) - fp32_exponent_bias  # bias the extracted raw exponent (assume not denormal)
        mant_normalized = mant + np.array((2**fp32_mantissa_bitwidth) * (exp != 0)).astype(int)  # append implicit 1
        # for this value to be representable as this ArbPrecFloatType:
        # the value must be within the representable range
        range_ok = np.logical_and(value <= self.max(), value >= self.min())
        # the mantissa must be within representable range:
        # no set bits in the mantissa beyond the allowed number of bits (assume value is not denormal in fp32)
        # compute bits of precision lost to tapered precision if denormal, clamp to: 0 <= dnm_shift <= nrm_mantissa_bitwidth
        dnm_shift = np.array(np.minimum(np.maximum(0, min_exponent - exp_biased), nrm_mantissa_bitwidth)).astype(int)
        available_bits = nrm_mantissa_bitwidth - dnm_shift  # number of bits of precision available
        mantissa_mask = (1 << (fp32_nrm_mantissa_bitwidth - available_bits)) - 1
        mantissa_ok = (mant_normalized & mantissa_mask) == 0
        return np.logical_and(mantissa_ok, range_ok)

    def is_integer(self) -> bool:
        return False

    def is_fixed_point(self) -> bool:
        return False

    def get_hls_datatype_str(self) -> str:
        raise NumericValueRefused.of(
            "undefined", "DataType.get_hls_datatype_str", "get_hls_datatype_str() not yet implemented for ArbPrecFloatType"
        )

    def to_numpy_dt(self) -> np.dtype:
        return np.dtype(np.float32)

    def get_canonical_name(self) -> str:
        return "FLOAT<%d,%d,%d>" % (
            self.exponent_bits(),
            self.mantissa_bits(),
            self.exponent_bias(),
        )

    def get_num_possible_values(self) -> int:
        # TODO: consider -0 and +0 as different values?
        # also assumes no special symbols like NaN, inf etc
        return 2 ** self.bitwidth()


class Float16Type(BaseDataType):
    def bitwidth(self) -> int:
        return 16

    def min(self) -> float:
        return float(np.finfo(np.float16).min)

    def max(self) -> float:
        return float(np.finfo(np.float16).max)

    def bounds(self) -> tuple[Union[int, Fraction], Union[int, Fraction]]:
        largest = BINARY16.largest
        return -largest, largest

    def _holds(self, value: Optional[_Exact]) -> bool:
        return isinstance(value, float) or _in_format(BINARY16, value)

    def _integers(self, values: np.ndarray) -> np.ndarray:
        return _float_integers(self, values)

    def _floats(self, values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float64, copy=False)
        return ~np.isfinite(values) | _representable(BINARY16, values)

    def container(self) -> np.dtype:
        return np.dtype(np.float16)

    def _require_codes(self) -> None:
        pass

    def _encode(self, values: np.ndarray) -> np.ndarray:
        return _ieee_codes(self.container(), values)

    def _decode(self, codes: np.ndarray) -> np.ndarray:
        return _ieee_values(self.container(), codes)

    def allowed(self, value: Union[int, float, np.ndarray]) -> Union[bool, np.ndarray]:
        if isinstance(value, np.ndarray):
            return np.ones(value.shape, dtype=bool)
        return True

    def get_num_possible_values(self) -> int:
        raise Exception("Undefined for Float16Type")

    def is_integer(self) -> bool:
        return False

    def is_fixed_point(self) -> bool:
        return False

    def get_hls_datatype_str(self) -> str:
        return "half"

    def to_numpy_dt(self) -> np.dtype:
        return np.dtype(np.float16)

    def get_canonical_name(self) -> str:
        return "FLOAT16"


class IntType(BaseDataType):
    def __init__(self, bitwidth: int, signed: bool) -> None:
        super().__init__()
        self._bitwidth = bitwidth
        self._signed = signed

    def bitwidth(self) -> int:
        return self._bitwidth

    def min(self) -> Union[int, float]:
        unsigned_min = 0
        signed_min = -(2 ** (self.bitwidth() - 1))
        return signed_min if self._signed else unsigned_min

    def max(self) -> Union[int, float]:
        unsigned_max = (2 ** (self.bitwidth())) - 1
        signed_max = (2 ** (self.bitwidth() - 1)) - 1
        return signed_max if self._signed else unsigned_max

    def bounds(self) -> tuple[Union[int, Fraction], Union[int, Fraction]]:
        if not _keeps_range(self, IntType):
            return BaseDataType.bounds(self)
        bits = _valued_width(self, self.bitwidth())
        if self._signed:
            return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        return 0, (1 << bits) - 1

    def allowed(self, value: Union[int, float, np.ndarray]) -> Union[bool, np.ndarray]:
        found = _exact_allowed(self, value)
        if found is not None:
            return found[()] if found.ndim == 0 else found
        value_is_integer = np.round(value) == value
        value_is_bounded = np.logical_and(self.min() <= value, value <= self.max())
        return np.logical_and(value_is_integer, value_is_bounded)

    def _holds(self, value: Optional[_Exact]) -> bool:
        least, greatest = self.bounds()
        return isinstance(value, int) and least <= value <= greatest

    def _integers(self, values: np.ndarray) -> np.ndarray:
        return _between(values, *self.bounds())

    def _floats(self, values: np.ndarray) -> np.ndarray:
        # In the array's own float type: its bounds there are exact for its values.
        least, greatest = self.bounds()
        low, high = _at_least_in(values.dtype, least), _at_most_in(values.dtype, greatest)
        found = (np.floor(values) == values) & (values >= low) & (values <= high)
        if not (np.isfinite(low) and np.isfinite(high)):
            found &= np.isfinite(values)
        return found

    def _wide_floats(self, values: np.ndarray) -> np.ndarray:
        # In the array's own type where float64 holds both bounds: then so does it.
        least, greatest = self.bounds()
        if not (_float64_exact(least) and _float64_exact(greatest)):
            return super()._wide_floats(values)
        low, high = values.dtype.type(float(least)), values.dtype.type(float(greatest))
        return (np.rint(values) == values) & (values >= low) & (values <= high)

    def container(self) -> np.dtype:
        bits = _valued_width(self, self.bitwidth())
        for width in (8, 16, 32, 64):
            if bits <= width:
                return np.dtype(f"{'i' if self._signed else 'u'}{width // 8}")
        raise NumericValueRefused.of(
            "undefined", "DataType.container", f"{self} has no exact numpy container: wider than 64 bits"
        )

    def _require_codes(self) -> None:
        if _valued_width(self, self.bitwidth()) > 64:
            raise NumericValueRefused.of("undefined", "DataType.codes", f"{self} has no codes: wider than 64 bits")

    def _encode(self, values: np.ndarray) -> np.ndarray:
        return _integer_codes(values, int(self.bitwidth()), self._signed)

    def _decode(self, codes: np.ndarray) -> np.ndarray:
        if self._signed:
            return _signed_codes(codes, int(self.bitwidth())).astype(self.container())
        return codes.astype(self.container())

    def get_num_possible_values(self) -> int:
        return int(abs(self.min()) + abs(self.max()) + 1)

    def is_integer(self) -> bool:
        return True

    def is_fixed_point(self) -> bool:
        return False

    def get_hls_datatype_str(self) -> str:
        if self.signed():
            return "ap_int<%d>" % self.bitwidth()
        else:
            return "ap_uint<%d>" % self.bitwidth()

    def to_numpy_dt(self) -> np.dtype:
        if self.bitwidth() <= 8:
            return np.dtype(np.int8 if self.signed() else np.uint8)
        elif self.bitwidth() <= 16:
            return np.dtype(np.int16 if self.signed() else np.uint16)
        elif self.bitwidth() <= 32:
            return np.dtype(np.int32 if self.signed() else np.uint32)
        elif self.bitwidth() <= 64:
            return np.dtype(np.int64 if self.signed() else np.uint64)
        else:
            raise NumericValueRefused.of("undefined", "DataType.to_numpy_dt", "Unknown numpy dtype for " + str(self))

    def get_canonical_name(self) -> str:
        if self.bitwidth() == 1 and (not self.signed()):
            return "BINARY"
        else:
            prefix = "INT" if self.signed() else "UINT"
            return prefix + str(self.bitwidth())


class BipolarType(BaseDataType):
    def bitwidth(self) -> int:
        return 1

    def min(self) -> int:
        return -1

    def max(self) -> int:
        return +1

    def bounds(self) -> tuple[Union[int, Fraction], Union[int, Fraction]]:
        return -1, 1

    def allowed(self, value: Union[int, float, np.ndarray]) -> Union[bool, np.ndarray]:
        if _is_integral(value):
            return _exact_allowed(self, value)
        return np.isin(value, [-1, +1])

    def _holds(self, value: Optional[_Exact]) -> bool:
        return isinstance(value, int) and value in (-1, 1)

    def _integers(self, values: np.ndarray) -> np.ndarray:
        return _between(values, -1, 1) & (values != 0)

    def _floats(self, values: np.ndarray) -> np.ndarray:
        return (values == -1.0) | (values == 1.0)

    def container(self) -> np.dtype:
        return np.dtype(np.int8)

    def _require_codes(self) -> None:
        pass

    def _encode(self, values: np.ndarray) -> np.ndarray:
        if values.dtype.kind in "iuf":
            return (values > 0).astype(np.uint64)
        return _from_ints(values, lambda value: int(value > 0))

    def _decode(self, codes: np.ndarray) -> np.ndarray:
        return np.where(codes == 1, 1, -1).astype(np.int8)

    def get_num_possible_values(self) -> int:
        return 2

    def is_integer(self) -> bool:
        return True

    def is_fixed_point(self) -> bool:
        return False

    def get_hls_datatype_str(self) -> str:
        return "ap_int<1>"

    def to_numpy_dt(self) -> np.dtype:
        return np.dtype(np.int8)

    def get_canonical_name(self) -> str:
        return "BIPOLAR"


class TernaryType(BaseDataType):
    def bitwidth(self) -> int:
        return 2

    def min(self) -> int:
        return -1

    def max(self) -> int:
        return +1

    def bounds(self) -> tuple[Union[int, Fraction], Union[int, Fraction]]:
        return -1, 1

    def allowed(self, value: Union[int, float, np.ndarray]) -> Union[bool, np.ndarray]:
        if _is_integral(value):
            return _exact_allowed(self, value)
        return np.isin(value, [-1, 0, +1])

    def _holds(self, value: Optional[_Exact]) -> bool:
        return isinstance(value, int) and value in (-1, 0, 1)

    def _integers(self, values: np.ndarray) -> np.ndarray:
        return _between(values, -1, 1)

    def _floats(self, values: np.ndarray) -> np.ndarray:
        return (values == -1.0) | (values == 0.0) | (values == 1.0)

    def container(self) -> np.dtype:
        return np.dtype(np.int8)

    def _require_codes(self) -> None:
        pass

    def _encode(self, values: np.ndarray) -> np.ndarray:
        return _integer_codes(values, 2, True)

    def _decode(self, codes: np.ndarray) -> np.ndarray:
        if (codes == 2).any():
            raise NumericValueRefused.of("code-outside", "DataType.values", "code 2 is no TERNARY value")
        return _signed_codes(codes, 2).astype(np.int8)

    def get_num_possible_values(self) -> int:
        return 3

    def is_integer(self) -> bool:
        return True

    def is_fixed_point(self) -> bool:
        return False

    def get_hls_datatype_str(self) -> str:
        return "ap_int<2>"

    def to_numpy_dt(self) -> np.dtype:
        return np.dtype(np.int8)

    def get_canonical_name(self) -> str:
        return "TERNARY"


class FixedPointType(IntType):
    def __init__(self, bitwidth: int, intwidth: int) -> None:
        super().__init__(bitwidth=bitwidth, signed=True)
        if intwidth >= bitwidth:
            raise DataTypeNameMalformed.of(
                "name-malformed",
                "DataType",
                "FixedPointType violates intwidth < bitwidth",
                bitwidth=int(bitwidth),
                intwidth=int(intwidth),
            )
        self._intwidth = intwidth

    def int_bits(self) -> int:
        return self._intwidth

    def frac_bits(self) -> int:
        return self.bitwidth() - self.int_bits()

    def scale_factor(self) -> float:
        return 2 ** -(self.frac_bits())

    def _fraction(self) -> int:
        """frac_bits() as a Python int, for the exact methods: a width given as a numpy
        integer would wrap in a shift."""
        return int(self.frac_bits())

    def min(self) -> float:
        return super().min() * self.scale_factor()

    def max(self) -> float:
        return super().max() * self.scale_factor()

    def bounds(self) -> tuple[Union[int, Fraction], Union[int, Fraction]]:
        if not _keeps_range(self, FixedPointType):
            return BaseDataType.bounds(self)
        bits = _valued_width(self, self.bitwidth())
        fraction = self._fraction()
        if fraction > _MAX_FRACTION_BITS or self.int_bits() > _MAX_INTEGER_BITS:
            raise NumericValueRefused.of(
                "beyond-envelope",
                "DataType.bounds",
                f"{self} is past the envelope: at most {_MAX_FRACTION_BITS} fraction bits "
                f"and {_MAX_INTEGER_BITS} integer bits",
            )
        return _fixed_point_bounds(bits, fraction)

    def allowed(self, value: Union[int, float, np.ndarray]) -> Union[bool, np.ndarray]:
        found = _exact_allowed(self, value)
        if found is not None:
            return found[()] if found.ndim == 0 else found
        int_value = value / self.scale_factor()
        codes = IntType(self._bitwidth, True)
        return _rounded_allowed(codes.min(), codes.max(), int_value)

    def _holds(self, value: Optional[_Exact]) -> bool:
        if not _finite(value):
            return False
        least, greatest = self.bounds()
        number = Fraction(value)
        return least <= number <= greatest and (number * (1 << self._fraction())).denominator == 1

    def _integers(self, values: np.ndarray) -> np.ndarray:
        least, greatest = self.bounds()
        return _between(values, math.ceil(least), math.floor(greatest))

    def _floats(self, values: np.ndarray) -> np.ndarray:
        least, greatest = self.bounds()
        bits, fraction = int(self.bitwidth()), self._fraction()
        scale = _scale_in(values.dtype, bits, fraction)
        if scale is None:
            if values.dtype.itemsize < 8:
                return self._floats(values.astype(np.float64))
            on_grid = np.isfinite(values) & _multiples(values, -fraction)
            return on_grid & (values >= _at_least(least)) & (values <= _at_most(greatest))
        # In the array's own float type. 2**fraction is a normal value of it, and every
        # code within the bounds is finite in it, so scaling a value within the bounds by
        # 2**fraction is exact, but where it underflows: only a nonzero value scaled
        # below 1 can, which is no code unless it rounds to 0.
        if _keeps_range(self, FixedPointType):
            low, high = _fixed_point_limits(values.dtype, bits, fraction)
        else:
            low, high = _at_least_in(values.dtype, least), _at_most_in(values.dtype, greatest)
        codes = values * scale
        found = (np.floor(codes) == codes) & (values >= low) & (values <= high)
        if fraction < 0:
            found &= (codes != 0) | (values == 0)
        return found

    def _wide_floats(self, values: np.ndarray) -> np.ndarray:
        # In the array's own type where float64 holds both bounds, and so it does: its
        # exponents reach past every code within them, so scaling is exact but where a
        # nonzero value scaled below 1 underflows to 0.
        least, greatest = self.bounds()
        if not (_float64_exact(least) and _float64_exact(greatest)):
            return super()._wide_floats(values)
        low, high = values.dtype.type(float(least)), values.dtype.type(float(greatest))
        codes = values * np.ldexp(values.dtype.type(1), self._fraction())
        found = (np.rint(codes) == codes) & (values >= low) & (values <= high)
        if self._fraction() < 0:
            found &= (codes != 0) | (values == 0)
        return found

    def carried_as_codes(self) -> bool:
        return self.bitwidth() > 54

    def container(self) -> np.dtype:
        self.bounds()
        if self.bitwidth() <= 54:
            return np.dtype(np.float64)
        if self.bitwidth() <= 64:
            return np.dtype(np.int64)
        raise NumericValueRefused.of(
            "undefined", "DataType.container", f"{self} has no exact numpy container: wider than 64 bits"
        )

    def _require_codes(self) -> None:
        self.bounds()
        if self.bitwidth() > 64:
            raise NumericValueRefused.of("undefined", "DataType.codes", f"{self} has no codes: wider than 64 bits")

    def _container_members(self, values: np.ndarray) -> np.ndarray:
        if not self.carried_as_codes():
            return super()._container_members(values)
        top = 1 << (int(self.bitwidth()) - 1)
        least, greatest = -top, top - 1
        if values.dtype.kind in "iu":
            return _between(values, least, greatest)
        return _each(
            values,
            lambda value: (
                isinstance(value, (int, np.integer))
                and not isinstance(value, (bool, np.bool_))
                and least <= int(value) <= greatest
            ),
        )

    def _encode(self, values: np.ndarray) -> np.ndarray:
        bits, fraction = int(self.bitwidth()), self._fraction()
        mask = (1 << bits) - 1
        if self.carried_as_codes():
            if values.dtype.kind in "iu":
                return values.astype(np.int64).view(np.uint64) & np.uint64(mask)
            return _from_ints(values, lambda value: int(value) & mask)
        if values.dtype.kind == "f" and values.dtype.itemsize <= 8:
            scaled = np.ldexp(values.astype(np.float64), fraction)
            return scaled.astype(np.int64).view(np.uint64) & np.uint64(mask)
        return _from_ints(values, lambda value: int(Fraction(value) * (1 << fraction)) & mask)

    def _decode(self, codes: np.ndarray) -> np.ndarray:
        signed = _signed_codes(codes, int(self.bitwidth()))
        if self.carried_as_codes():
            return signed
        return np.ldexp(signed.astype(np.float64), -self._fraction())

    def is_integer(self) -> bool:
        return False

    def is_fixed_point(self) -> bool:
        return True

    def get_hls_datatype_str(self) -> str:
        return "ap_fixed<%d, %d>" % (self.bitwidth(), self.int_bits())

    def to_numpy_dt(self) -> np.dtype:
        return np.dtype(np.float32)

    def get_canonical_name(self) -> str:
        return "FIXED<%d,%d>" % (self.bitwidth(), self.int_bits())


class ScaledIntType(IntType):
    # scaled integer datatype, only intended for
    # inference cost calculations, many of the
    # member methods are not implemented
    def __init__(self, bitwidth: int) -> None:
        super().__init__(bitwidth=bitwidth, signed=True)

    def min(self) -> int:
        raise NumericValueRefused.of("undefined", "DataType.min", "Undefined for ScaledIntType")

    def max(self) -> int:
        raise Exception("Undefined for ScaledIntType")

    def bounds(self) -> tuple[Union[int, Fraction], Union[int, Fraction]]:
        raise NumericValueRefused.of(
            "undefined", "DataType.bounds", f"{self} has no value set: a scaled integer's scale is not stated"
        )

    def container(self) -> np.dtype:
        raise NumericValueRefused.of(
            "undefined", "DataType.container", f"{self} has no exact numpy container: it has no value set"
        )

    def _require_codes(self) -> None:
        raise NumericValueRefused.of("undefined", "DataType.codes", f"{self} has no codes: it has no value set")

    def allowed(self, value: Union[int, float, np.ndarray]) -> Union[bool, np.ndarray]:
        raise Exception("Undefined for ScaledIntType")

    def is_integer(self) -> bool:
        return False

    def is_fixed_point(self) -> bool:
        return False

    def get_hls_datatype_str(self) -> str:
        raise Exception("Undefined for ScaledIntType")

    def to_numpy_dt(self) -> np.dtype:
        return np.dtype(np.float32)

    def signed(self) -> bool:
        "Returns whether this DataType can represent negative numbers."
        return True

    def get_canonical_name(self) -> str:
        return "SCALEDINT<%d>" % (self.bitwidth())


def _parameter(text: str) -> int:
    """``int(text)``; a parameter with more digits than ``int`` reads is read by its
    significant digits, its leading zeros dropped."""
    try:
        return int(text)
    except ValueError:
        digits = text.strip()
        sign = digits[:1] if digits[:1] in ("+", "-") else ""
        digits = digits[len(sign) :]
        significant = digits.lstrip("0") or "0"
        if not (digits.isascii() and digits.isdigit() and len(significant) < len(digits)):
            raise
    return int(sign + significant)


def _construct_unchecked(name: str) -> BaseDataType:
    """The datatype a name spells, parsed as resolve_datatype always has."""
    _special_types = {
        "BINARY": IntType(1, False),
        "BIPOLAR": BipolarType(),
        "TERNARY": TernaryType(),
        "FLOAT32": FloatType(),
        "FLOAT16": Float16Type(),
    }
    if name in _special_types.keys():
        return _special_types[name]
    elif name.startswith("UINT"):
        bitwidth = _parameter(name.replace("UINT", ""))
        return IntType(bitwidth, False)
    elif name.startswith("INT"):
        bitwidth = _parameter(name.replace("INT", ""))
        return IntType(bitwidth, True)
    elif name.startswith("FIXED"):
        name = name.replace("FIXED<", "")
        name = name.replace(">", "")
        nums = name.split(",")
        bitwidth = _parameter(nums[0].strip())
        intwidth = _parameter(nums[1].strip())
        return FixedPointType(bitwidth, intwidth)
    elif name.startswith("SCALEDINT"):
        name = name.replace("SCALEDINT<", "")
        name = name.replace(">", "")
        nums = name.split(",")
        bitwidth = _parameter(nums[0].strip())
        return ScaledIntType(bitwidth)
    elif name.startswith("FLOAT<"):
        name = name.replace("FLOAT<", "")
        name = name.replace(">", "")
        nums = name.split(",")
        if len(nums) == 2:
            exp_bits = _parameter(nums[0].strip())
            mant_bits = _parameter(nums[1].strip())
            return ArbPrecFloatType(exp_bits, mant_bits)
        elif len(nums) == 3:
            exp_bits = _parameter(nums[0].strip())
            mant_bits = _parameter(nums[1].strip())
            exp_bias = _parameter(nums[2].strip())
            return ArbPrecFloatType(exp_bits, mant_bits, exp_bias)
        else:
            raise KeyError("Could not resolve DataType " + name)
    else:
        raise KeyError("Could not resolve DataType " + name)


def _construct(name: str, owner: str = "DataType") -> BaseDataType:
    """Read a source spelling, preserving unknown-name and malformed-name signals."""
    try:
        return _construct_unchecked(name)
    except DataTypeNameRefused as error:
        error.findings = tuple(replace(found.about(name=name), owner=owner) for found in error.findings)
        if error._default_message:
            error.args = ("\n".join(map(str, error.findings)),)
        raise
    except KeyError as error:
        raise DataTypeNameUnknown.of("name-unknown", owner, str(error.args[0]), name=name) from error
    except (ValueError, IndexError) as error:
        raise DataTypeNameMalformed.of("name-malformed", owner, str(error), name=name) from error
    except OverflowError as error:
        raise DataTypeNameMalformed.of(
            "beyond-envelope", owner, f"{name} is past the envelope: its default exponent bias overflows", name=name
        ) from error


def _nonsense_refusal(datatype: BaseDataType, name: str) -> DataTypeNameMalformed:
    """The name refusal a deprecated value set will give after its warning period."""
    try:
        if isinstance(datatype, ScaledIntType):
            _valued_width(datatype, datatype.bitwidth())
        else:
            datatype.bounds()
    except NumericValueRefused as error:
        code = error.findings[0].code
        if datatype.bitwidth() == 0:
            code = "width-zero"
        elif code != "beyond-envelope":
            code = "name-malformed"
        return DataTypeNameMalformed.of(code, "DataType", error.findings[0].message, name=name)
    raise AssertionError("a deprecated datatype has a reason")


def _nonsense(datatype: BaseDataType) -> Optional[str]:
    """Why a resolved datatype is one a later release refuses (a zero or negative width
    or parameter, or past the envelope); None for any other."""
    try:
        if isinstance(datatype, ScaledIntType):
            _valued_width(datatype, datatype.bitwidth())
        else:
            datatype.bounds()
    except NumericValueRefused as reason:
        return reason.findings[0].message
    return None


@lru_cache(maxsize=4096)
def _nonsense_name(name: str) -> Optional[str]:
    """_nonsense of the datatype a name spells, kept per name: resolution is frequent."""
    return _nonsense(_construct(name))


def _resolve(name: str, canonical: bool, stacklevel: int, owner: str = "DataType") -> BaseDataType:
    """The one resolution path of resolve_datatype and DataType[name]."""
    if not isinstance(name, str):
        raise TypeError(f"Input 'name' must be of type 'str', but got type '{type(name).__name__}'")
    if not canonical:
        found = _construct(name, owner)
        reason = _nonsense_name(name)
        if reason is not None:
            warnings.warn(
                f"DataType {name!r} is deprecated ({reason}); a later release refuses it",
                DataTypeWarning,
                stacklevel=stacklevel,
            )
        return found
    try:
        found = _construct(name, owner)
    except DataTypeNameRefused as error:
        raise DataTypeNameUnknown.of(
            "name-not-canonical",
            "resolve_datatype",
            f"{name!r} is no canonical DataType name",
            name=name,
            reason=error.findings[0].code,
        ) from error
    reason = _nonsense_name(name)
    if reason is not None:
        plain = _nonsense_refusal(found, name)
        raise DataTypeNameUnknown.of(
            "name-not-canonical",
            "resolve_datatype",
            f"{name!r} is no canonical DataType name: {reason}",
            name=name,
            reason=plain.findings[0].code,
        ) from plain
    if found.get_canonical_name() != name:
        raise DataTypeNameUnknown.of(
            "name-not-canonical",
            "resolve_datatype",
            f"{name!r} is not a canonical DataType name; it denotes {found.get_canonical_name()}",
            name=name,
            reason=found.get_canonical_name(),
        )
    return found


def resolve_datatype(name: str, canonical: bool = False) -> BaseDataType:
    """Returns the datatype a name denotes. By default the name is read as a source
    spells it (UINT1 is BINARY, FLOAT<5,10> is FLOAT<5,10,15>), and a name a later release
    refuses resolves with a DataTypeWarning. With canonical=True only the canonical name of
    a datatype within the envelope resolves, as a stored annotation holds it; any other
    name raises KeyError (see the module's docstring)."""
    return _resolve(name, canonical, stacklevel=3, owner="resolve_datatype")


def _ieee_format(target: BaseDataType) -> Optional[FloatFormat]:
    """The binary format of an IEEE float datatype; None for any other datatype."""
    if isinstance(target, FloatType):
        return BINARY32
    if isinstance(target, Float16Type):
        return BINARY16
    return None


def round_to(target: BaseDataType, values: object) -> np.ndarray:
    """Returns each value rounded once, from its exact value, to the nearest value of the
    IEEE float datatype ``target`` (FLOAT16 or FLOAT32), in ``target.container()``: IEEE
    754's round to nearest, ties to even. A magnitude at least the largest finite value
    plus half its spacing becomes the infinity of its sign; an infinity stays; a NaN stays
    a NaN, its payload not kept; a zero keeps its sign. Raises ValueError for a target that
    is no IEEE float datatype, and TypeError for a value that is no number."""
    fmt = _ieee_format(target)
    if fmt is None:
        raise NumericValueRefused.of("undefined", "round_to", f"{target} is no IEEE float datatype to round to")
    container = target.container()
    array = values if isinstance(values, np.ndarray) else np.asarray(values, dtype=object)
    kind = array.dtype.kind
    if kind == "f" and array.dtype.itemsize <= 8:
        # float16 and float32 widen to float64 exactly; numpy rounds float64 to the
        # target once. A wider float (longdouble) is rounded from its exact value below.
        with np.errstate(over="ignore"):
            return array.astype(np.float64).astype(container)
    if kind in "iu" and (not array.size or max(-int(array.min()), int(array.max())) <= 1 << 53):
        with np.errstate(over="ignore"):
            return array.astype(np.float64).astype(container)
    if kind not in "iufO":
        raise TypeError(f"round_to takes numbers, not {array.dtype}")
    found = np.empty(array.shape, dtype=container)
    for index, value in np.ndenumerate(array):
        found[index] = _nearest(fmt, value)
    return found


def _nearest(fmt: FloatFormat, value: object) -> float:
    """One value rounded into an IEEE format, as a float (its sign kept on a zero)."""
    number = _exact(value)
    if number is None:
        raise TypeError(f"round_to takes numbers, not {value!r}")
    if isinstance(number, float):  # an infinity or NaN
        return number
    rounded = fmt.nearest(number)
    if rounded == 0:
        negative = number < 0 or (isinstance(value, (float, np.floating)) and math.copysign(1, value) < 0)
        return -0.0 if negative else 0.0
    return float(Fraction(rounded)) if not isinstance(rounded, float) else rounded


class DataTypeMeta(EnumMeta):
    def __getitem__(self, name: str) -> BaseDataType:
        return _resolve(name, False, stacklevel=3)


if TYPE_CHECKING:

    class _DataTypeLookup(Protocol):
        """The datatype values and accumulator helpers available through DataType."""

        def __getitem__(self, name: str) -> BaseDataType:
            ...

        def get_accumulator_dt_cands(self) -> list[str]:
            ...

        def get_smallest_possible(self, value: Union[int, float]) -> BaseDataType:
            ...

    DataType: _DataTypeLookup
else:

    class DataType(Enum, metaclass=DataTypeMeta):
        """Enum class that contains QONNX data types to set the quantization annotation.
        ONNX does not support data types smaller than 8-bit integers, whereas in QONNX we are
        interested in smaller integers down to ternary and bipolar."""

        @staticmethod
        def get_accumulator_dt_cands() -> list[str]:
            cands = ["BINARY"]
            cands += ["UINT%d" % (x + 1) for x in range(64)]
            cands += ["BIPOLAR", "TERNARY"]
            cands += ["INT%d" % (x + 1) for x in range(64)]
            return cands

        @staticmethod
        def get_smallest_possible(value: Union[int, float]) -> BaseDataType:
            """Returns smallest (fewest bits) possible DataType that can represent
            value. Prefers unsigned integers where possible."""
            if not int(value) == value:
                return DataType["FLOAT32"]
            cands = DataType.get_accumulator_dt_cands()
            for cand in cands:
                dt = DataType[cand]
                if (dt.min() <= value) and (value <= dt.max()):
                    return dt
            raise NumericValueRefused.of(
                "undefined", "DataType.get_smallest_possible", "Could not find a suitable int datatype for " + str(value)
            )
