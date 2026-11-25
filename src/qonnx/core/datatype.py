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
"""

import math
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, EnumMeta
from fractions import Fraction
from typing import Optional, Union

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
            raise ValueError("an endpoint is not NaN")
        if math.isinf(value):
            return value
        value = Fraction(value)
    if isinstance(value, Fraction):
        if value.denominator & (value.denominator - 1):
            raise ValueError(f"an endpoint is dyadic (m * 2**e), not {value}")
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
            raise ValueError(
                f"no binary float format has {self.mantissa} fraction bits and normal "
                f"exponents {self.emin} to {self.emax}"
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
                raise ValueError("NaN has no nearest value in a format")
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
        raise ValueError(f"{family} with a width of {bits} bits has no values")
    if bits > _MAX_BITS:
        raise ValueError(f"{family} with a width of {bits} bits is past the envelope: wider than {_MAX_BITS} bits")
    return int(bits)


def _keeps_range(datatype: "BaseDataType", family: type) -> bool:
    """Whether a datatype keeps its family's min() and max(). A subclass defined elsewhere
    that redefines them states its own range, and its bounds() reads them exactly."""
    return type(datatype).min is family.min and type(datatype).max is family.max


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
            raise ValueError(f"{self} has no finite bounds")
        return least, greatest

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

    def allowed(self, value: Union[int, float, np.ndarray]) -> Union[bool, np.ndarray]:
        if isinstance(value, np.ndarray):
            return np.ones(value.shape, dtype=bool)
        return True

    def get_num_possible_values(self) -> int:
        raise Exception("Undefined for FloatType")

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

        if not exponent_bias:
            # default (IEEE-style) exponent bias
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
            raise ValueError(f"{self} has no values: exponent bits >= 1, mantissa bits >= 0, an integer bias")
        _valued_width(self, self.bitwidth())
        bias = int(bias)
        emax = (1 << exponent_bits) - 1 - bias
        if emax > 1023 or 1 - bias - mantissa_bits < -1074:
            raise ValueError(f"{self} is past the envelope: its exponents are beyond float64's")
        found = FloatFormat(int(mantissa_bits), 1 - bias, emax)
        try:
            float(found.largest)
        except OverflowError:
            raise ValueError(f"{self} is past the envelope: its greatest value rounds past float64's") from None
        return found

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
        assert False, "get_hls_datatype_str() not yet implemented for ArbPrecFloatType"

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
        value_is_integer = np.round(value) == value
        value_is_bounded = np.logical_and(self.min() <= value, value <= self.max())
        return np.logical_and(value_is_integer, value_is_bounded)

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
            raise Exception("Unknown numpy dtype for " + str(self))

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
        return np.isin(value, [-1, +1])

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
        return np.isin(value, [-1, 0, +1])

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
        assert intwidth < bitwidth, "FixedPointType violates intwidth < bitwidth"
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
            raise ValueError(
                f"{self} is past the envelope: at most {_MAX_FRACTION_BITS} fraction bits "
                f"and {_MAX_INTEGER_BITS} integer bits"
            )
        top, scale = 1 << (bits - 1), 1 << fraction
        return _rational(Fraction(-top, scale)), _rational(Fraction(top - 1, scale))

    def allowed(self, value: Union[int, float, np.ndarray]) -> Union[bool, np.ndarray]:
        int_value = value / self.scale_factor()
        return IntType(self._bitwidth, True).allowed(int_value)

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
        raise Exception("Undefined for ScaledIntType")

    def max(self) -> int:
        raise Exception("Undefined for ScaledIntType")

    def bounds(self) -> tuple[Union[int, Fraction], Union[int, Fraction]]:
        raise ValueError(f"{self} has no value set: a scaled integer's scale is not stated")

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


def resolve_datatype(name: str) -> BaseDataType:
    if not isinstance(name, str):
        raise TypeError(
            f"Input 'name' must be of type 'str', but got type '{type(name).__name__}'"
        )

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
        bitwidth = int(name.replace("UINT", ""))
        return IntType(bitwidth, False)
    elif name.startswith("INT"):
        bitwidth = int(name.replace("INT", ""))
        return IntType(bitwidth, True)
    elif name.startswith("FIXED"):
        name = name.replace("FIXED<", "")
        name = name.replace(">", "")
        nums = name.split(",")
        bitwidth = int(nums[0].strip())
        intwidth = int(nums[1].strip())
        return FixedPointType(bitwidth, intwidth)
    elif name.startswith("SCALEDINT"):
        name = name.replace("SCALEDINT<", "")
        name = name.replace(">", "")
        nums = name.split(",")
        bitwidth = int(nums[0].strip())
        return ScaledIntType(bitwidth)
    elif name.startswith("FLOAT<"):
        name = name.replace("FLOAT<", "")
        name = name.replace(">", "")
        nums = name.split(",")
        if len(nums) == 2:
            exp_bits = int(nums[0].strip())
            mant_bits = int(nums[1].strip())
            return ArbPrecFloatType(exp_bits, mant_bits)
        elif len(nums) == 3:
            exp_bits = int(nums[0].strip())
            mant_bits = int(nums[1].strip())
            exp_bias = int(nums[2].strip())
            return ArbPrecFloatType(exp_bits, mant_bits, exp_bias)
        else:
            raise KeyError("Could not resolve DataType " + name)
    else:
        raise KeyError("Could not resolve DataType " + name)


class DataTypeMeta(EnumMeta):
    def __getitem__(self, name: str) -> BaseDataType:
        return resolve_datatype(name)


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
        raise Exception("Could not find a suitable int datatype for " + str(value))
