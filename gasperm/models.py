"""Structured data passed between modules.

Everything here is a Pydantic model rather than a raw dict, so a field renamed
in one module fails loudly at the boundary instead of silently producing a
``None``.

Unit convention: any field whose name ends in a unit suffix (``_atm``,
``_cm3_s``, ``_cp``, ``_darcy``, ``_cm``) is already in **internal CGS-Darcy
units**. Conversion happens at the calibration boundary on the way in and at
the display/storage boundary on the way out -- never in between.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SampleGeometry(BaseModel):
    """Physical geometry of the core plug under test."""

    model_config = ConfigDict(frozen=True)

    sample_id: str
    length_cm: float = Field(gt=0.0)
    diameter_cm: float = Field(gt=0.0)
    description: str = ""
    porosity_fraction: float | None = Field(default=None, ge=0.0, le=1.0)

    @property
    def area_cm2(self) -> float:
        """Cross-sectional area of the plug, cm^2."""
        from gasperm.units import circle_area_cm2

        return circle_area_cm2(self.diameter_cm)


class GasState(BaseModel):
    """Thermophysical state of the working gas at one instant.

    Resolved by :mod:`gasperm.gas_properties` from the live temperature and the
    mean pore pressure, then handed to :mod:`gasperm.permeability` as a plain
    number.
    """

    model_config = ConfigDict(frozen=True)

    gas_name: str
    temperature_k: float
    pressure_pa: float
    viscosity_cp: float
    density_kg_m3: float | None = None
    compressibility_z: float | None = None
    #: ``"coolprop"`` for a live lookup, ``"fixed"`` when the config bypassed it.
    source: Literal["coolprop", "fixed"] = "coolprop"


class Reading(BaseModel):
    """One acquisition sample: raw voltages, calibrated values, derived result.

    Both the raw voltages and the derived quantities are kept so a run can be
    re-processed after the fact with a corrected calibration without repeating
    the experiment.
    """

    model_config = ConfigDict(frozen=True)

    index: int
    timestamp: datetime
    elapsed_s: float

    # --- raw hardware ---
    inlet_voltage: float
    outlet_voltage: float
    flow_voltage: float
    #: Raw serial line as received, kept verbatim when parsing failed.
    temperature_raw: str | None = None

    # --- calibrated, absolute, internal CGS ---
    inlet_pressure_atm: float
    outlet_pressure_atm: float
    #: P2 actually used in the Darcy equation (may be the configured
    #: atmospheric reference rather than the measured outlet transducer).
    downstream_pressure_atm: float
    #: Mean pore pressure, (P1 + P2) / 2, absolute.
    mean_pressure_atm: float
    #: Flow rate as measured, converted to cm^3/s but still at the meter's
    #: own reference state.
    flow_cm3_s: float
    #: Flow rate paired with :attr:`flow_reference_pressure_atm` for the Darcy
    #: equation; the product ``flow * reference pressure`` is the invariant.
    flow_reference_cm3_s: float
    flow_reference_pressure_atm: float

    temperature_c: float
    #: False when the serial link produced no fresh value for this sample.
    temperature_ok: bool = True
    #: True when the temperature is a carried-over last-known-good value.
    temperature_stale: bool = False

    viscosity_cp: float
    #: Apparent gas permeability for this sample; ``None`` when the pressure
    #: differential is too small or non-physical to invert.
    permeability_darcy: float | None = None
    #: Rolling-window mean over ``run.averaging_window_s``.
    permeability_darcy_avg: float | None = None
    #: Populated when a sample could not yield a permeability.
    note: str | None = None

    @property
    def delta_pressure_atm(self) -> float:
        """P1 - P2 (absolute), atm."""
        return self.inlet_pressure_atm - self.downstream_pressure_atm


class RunSummary(BaseModel):
    """Aggregate result of a completed ``collect`` run."""

    model_config = ConfigDict(frozen=True)

    sample_id: str
    gas_name: str
    started_at: datetime
    ended_at: datetime
    duration_s: float
    sample_count: int
    #: Steady-state mean over the trailing averaging window.
    mean_pressure_atm: float
    permeability_darcy: float
    permeability_stddev_darcy: float
    mean_temperature_c: float
    mean_flow_cm3_s: float
    #: How many trailing samples the steady-state means were taken over.
    averaged_samples: int
    csv_path: str | None = None
    #: Non-fatal problems seen during the run (serial dropouts, bad samples).
    warnings: list[str] = Field(default_factory=list)


#: Historical/alternate name for :class:`RunSummary`.
RunResult = RunSummary


class KlinkenbergPoint(BaseModel):
    """One (mean pressure, apparent permeability) pair feeding the regression."""

    model_config = ConfigDict(frozen=True)

    mean_pressure_atm: float = Field(gt=0.0)
    apparent_permeability_darcy: float
    label: str = ""
    #: Present when the point came from a ``collect`` run rather than a CSV.
    source_path: str | None = None
    sample_id: str | None = None

    @property
    def inverse_mean_pressure(self) -> float:
        """1 / P_mean -- the regression's independent variable, 1/atm."""
        return 1.0 / self.mean_pressure_atm


class KlinkenbergResult(BaseModel):
    """Outcome of the Klinkenberg regression ``k_g = k_L + (k_L * b) * (1/P)``."""

    model_config = ConfigDict(frozen=True)

    #: Liquid-equivalent (Klinkenberg-corrected) permeability = y-intercept.
    liquid_permeability_darcy: float
    #: Gas slippage factor b, atm. Equals slope / intercept.
    slippage_factor_atm: float
    #: Fitted slope, k_L * b (darcy*atm).
    slope: float
    #: Fitted intercept, k_L (darcy).
    intercept: float
    r_squared: float
    #: Standard error of the intercept, i.e. of k_L itself.
    intercept_stderr: float | None = None
    slope_stderr: float | None = None
    points: list[KlinkenbergPoint]
    warnings: list[str] = Field(default_factory=list)

    @property
    def point_count(self) -> int:
        """Number of (P_mean, k_g) pairs used in the fit."""
        return len(self.points)

    def predict_darcy(self, mean_pressure_atm: float) -> float:
        """Apparent permeability the fit predicts at ``mean_pressure_atm``."""
        if mean_pressure_atm <= 0.0:
            raise ValueError("mean pressure must be positive")
        return self.intercept + self.slope / mean_pressure_atm


def is_finite(value: float | None) -> bool:
    """True when ``value`` is a real, usable number (not ``None``/NaN/inf)."""
    return value is not None and math.isfinite(value)
