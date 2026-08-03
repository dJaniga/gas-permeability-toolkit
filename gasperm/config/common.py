"""Building blocks shared by the hardware, sample and run schemas."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gasperm import units

__all__ = [
    "ConfigError",
    "PressureUnit",
    "LinearCalibration",
    "UncertaintySpec",
    "validated_pressure_unit",
    "_Base",
]


class ConfigError(Exception):
    """Raised for unusable configuration, with an operator-readable message."""


def validated_pressure_unit(value: str) -> str:
    """Normalise and validate a pressure unit, raising with the allowed set."""
    return units.normalize_pressure_unit(value)


PressureUnit = Annotated[str, Field(description="One of units.SUPPORTED_PRESSURE_UNITS")]


class _Base(BaseModel):
    """Strict base: unknown keys are errors, assignment is re-validated."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class LinearCalibration(_Base):
    """Two-point linear map from transducer volts to a physical value.

    ``value = value_min + (volts - volts_min) * span_value / span_volts``
    """

    volts_min: float = 0.0
    volts_max: float = 5.0
    value_min: float = 0.0
    value_max: float = 1000.0

    @model_validator(mode="after")
    def _non_degenerate(self) -> LinearCalibration:
        if self.volts_min == self.volts_max:
            raise ValueError(
                "volts_min and volts_max must differ; a zero-width voltage span "
                "cannot define a calibration"
            )
        if self.value_min == self.value_max:
            raise ValueError(
                "value_min and value_max must differ; the channel would report a "
                "constant regardless of input voltage"
            )
        return self

    def apply(self, volts: float) -> float:
        """Map a raw voltage to the calibrated physical value (config units)."""
        span_volts = self.volts_max - self.volts_min
        span_value = self.value_max - self.value_min
        return self.value_min + (volts - self.volts_min) * span_value / span_volts

    def invert(self, value: float) -> float:
        """Inverse of :meth:`apply` -- physical value back to volts. Test aid."""
        span_volts = self.volts_max - self.volts_min
        span_value = self.value_max - self.value_min
        return self.volts_min + (value - self.value_min) * span_volts / span_value

    @property
    def full_scale(self) -> float:
        """Magnitude of the calibrated span, for percent-of-full-scale specs."""
        return abs(self.value_max - self.value_min)


class UncertaintySpec(_Base):
    """A Type B uncertainty, as instrument specifications actually state it.

    GUM (ISO/IEC Guide 98-3) section 4.3: a specification quotes a half-width
    ``a`` together with an implied distribution, and the standard uncertainty
    is ``a`` divided by that distribution's factor.

    ``kind`` says how the half-width is computed from the reading:

    ``percent_full_scale``
        ``a = value/100 * full_scale`` -- how pressure transducers and most
        analog instruments are specified.
    ``percent_reading``
        ``a = value/100 * |reading|`` -- how thermal mass flowmeters are
        usually specified over their upper range.
    ``absolute``
        ``a = value``, in the channel's own configured unit.
    ``none``
        No contribution. Use only when a term is genuinely negligible, and say
        why in ``source``.
    """

    kind: Literal["percent_full_scale", "percent_reading", "absolute", "none"] = (
        "percent_full_scale"
    )
    value: float = Field(default=0.0, ge=0.0)
    #: Divisor applied to the half-width. Rectangular is the GUM default when
    #: a specification quotes limits with no distribution stated.
    distribution: Literal["rectangular", "triangular", "normal"] = "rectangular"
    #: Only used when ``distribution == "normal"``: the k the certificate's
    #: expanded uncertainty was reported at.
    coverage_factor: float = Field(default=2.0, gt=0.0)
    #: Degrees of freedom for Welch-Satterthwaite. ``null`` means infinite,
    #: which is right for a specification limit taken as exact.
    degrees_of_freedom: float | None = Field(default=None, gt=0.0)
    #: Where the number came from -- certificate number, datasheet, estimate.
    source: str = ""

    @property
    def divisor(self) -> float:
        """The factor the half-width is divided by, per GUM 4.3."""
        if self.distribution == "rectangular":
            return math.sqrt(3.0)
        if self.distribution == "triangular":
            return math.sqrt(6.0)
        return self.coverage_factor

    def half_width(self, reading: float, full_scale: float) -> float:
        """The specification limit ``a`` for this reading, in the channel's unit."""
        if self.kind == "none":
            return 0.0
        if self.kind == "percent_full_scale":
            return self.value / 100.0 * abs(full_scale)
        if self.kind == "percent_reading":
            return self.value / 100.0 * abs(reading)
        return self.value

    def standard_uncertainty(self, reading: float, full_scale: float) -> float:
        """Standard uncertainty ``u(x) = a / divisor``, in the channel's unit."""
        return self.half_width(reading, full_scale) / self.divisor

    @property
    def dof(self) -> float:
        """Degrees of freedom, with ``null`` meaning infinite."""
        return math.inf if self.degrees_of_freedom is None else self.degrees_of_freedom

    @field_validator("value")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("uncertainty value must be finite")
        return value
