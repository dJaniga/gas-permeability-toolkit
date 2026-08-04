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
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SampleGeometry(BaseModel):
    """Physical geometry of the core plug, with its measurement uncertainty.

    The caliper uncertainties are carried here rather than looked up later
    because area enters the Darcy equation as ``d^2`` -- a 1% diameter error is
    a 2% permeability error, which is usually the largest single term in the
    budget.
    """

    model_config = ConfigDict(frozen=True)

    sample_id: str
    length_cm: float = Field(gt=0.0)
    diameter_cm: float = Field(gt=0.0)
    description: str = ""
    porosity_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    #: Standard uncertainty of the length measurement, cm.
    length_uncertainty_cm: float = Field(default=0.0, ge=0.0)
    #: Standard uncertainty of the diameter measurement, cm.
    diameter_uncertainty_cm: float = Field(default=0.0, ge=0.0)

    @property
    def area_cm2(self) -> float:
        """Cross-sectional area of the plug, cm^2."""
        from gasperm.units import circle_area_cm2

        return circle_area_cm2(self.diameter_cm)

    @property
    def relative_length_uncertainty(self) -> float:
        """u(L)/L, dimensionless."""
        return self.length_uncertainty_cm / self.length_cm

    @property
    def relative_area_uncertainty(self) -> float:
        """u(A)/A = 2 u(d)/d, since ``A = pi d^2 / 4``."""
        return 2.0 * self.diameter_uncertainty_cm / self.diameter_cm


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
    #: Relative standard uncertainty of the viscosity model itself, from
    #: config. Feeds the GUM budget as a Type B component.
    relative_viscosity_uncertainty: float = 0.0


# --------------------------------------------------------------------------
# Steady state
# --------------------------------------------------------------------------


class SignalStability(BaseModel):
    """Stationarity diagnostics for one monitored signal over one window."""

    model_config = ConfigDict(frozen=True)

    name: str
    sample_count: int
    mean: float
    stddev: float
    #: Coefficient of variation, ``stddev / |mean|``.
    relative_stddev: float
    #: OLS slope of the signal against elapsed time, per second.
    slope_per_s: float
    #: Fractional change the fitted slope predicts across the whole window --
    #: the drift criterion, and the one that catches a slow ramp that a pure
    #: scatter test would pass.
    relative_drift: float
    #: Student-t statistic for H0: slope == 0. ``None`` with too few points.
    slope_t_statistic: float | None = None
    slope_p_value: float | None = None
    passed: bool = False
    failures: list[str] = Field(default_factory=list)


class SteadyStateStatus(BaseModel):
    """Whether the rig is currently at steady state, and why (or why not)."""

    model_config = ConfigDict(frozen=True)

    is_steady: bool
    #: Consecutive windows that have passed every criterion.
    consecutive_passes: int
    required_passes: int
    window_s: float
    elapsed_s: float
    signals: list[SignalStability] = Field(default_factory=list)
    #: Elapsed time at which steady state was first confirmed.
    reached_at_elapsed_s: float | None = None
    #: Short human-readable explanation for the console.
    summary: str = ""

    @property
    def progress(self) -> str:
        """``"2/3"`` style progress towards confirmation."""
        return f"{self.consecutive_passes}/{self.required_passes}"


class SteadyStateWindow(BaseModel):
    """The span of a run that was confirmed steady, i.e. the measurement."""

    model_config = ConfigDict(frozen=True)

    start_elapsed_s: float
    end_elapsed_s: float
    sample_count: int
    #: Index of the first reading inside the window, into the run's readings.
    start_index: int
    end_index: int

    @property
    def duration_s(self) -> float:
        """Length of the steady window, seconds."""
        return self.end_elapsed_s - self.start_elapsed_s


# --------------------------------------------------------------------------
# Uncertainty (ISO/IEC Guide 98-3, "GUM")
# --------------------------------------------------------------------------


class UncertaintyComponent(BaseModel):
    """One input quantity's contribution to the combined uncertainty.

    Follows GUM notation: ``x_i`` with standard uncertainty ``u(x_i)``,
    sensitivity coefficient ``c_i = d f / d x_i``, and contribution
    ``|c_i| u(x_i)`` to the combined standard uncertainty of the measurand.

    Because the Darcy equation is a product of powers, the *relative*
    sensitivity ``(x_i / k)(dk / dx_i)`` is the exponent of that input, which
    is what :attr:`relative_sensitivity` reports -- 1 for flow and viscosity,
    -2 for diameter, and a pressure-dependent value for P1 and P2.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    symbol: str
    #: GUM evaluation type: A (statistical, from the run) or B (from a
    #: specification, certificate or handbook).
    evaluation_type: Literal["A", "B"]
    value: float
    unit: str
    standard_uncertainty: float
    #: u(x_i) / |x_i|.
    relative_standard_uncertainty: float
    #: Dimensionless log-derivative, i.e. the exponent of this input.
    relative_sensitivity: float
    #: |c_i * u_rel|, the term that is squared and summed.
    relative_contribution: float
    degrees_of_freedom: float = float("inf")
    source: str = ""

    @property
    def variance_share(self) -> float:
        """This component's squared relative contribution."""
        return self.relative_contribution**2


class UncertaintyBudget(BaseModel):
    """Full GUM budget for a measured permeability.

    ``value +/- expanded_uncertainty`` is the reported result at
    :attr:`coverage_probability`.
    """

    model_config = ConfigDict(frozen=True)

    measurand: str = "apparent gas permeability"
    value_darcy: float
    combined_standard_uncertainty_darcy: float
    relative_combined_standard_uncertainty: float
    #: Welch-Satterthwaite effective degrees of freedom.
    effective_degrees_of_freedom: float
    coverage_factor: float
    coverage_probability: float
    expanded_uncertainty_darcy: float
    components: list[UncertaintyComponent] = Field(default_factory=list)
    #: Covariance contribution from correlated inputs (P1 and P2 sharing a
    #: transducer model or an atmospheric reference), as a relative variance.
    correlation_relative_variance: float = 0.0
    notes: list[str] = Field(default_factory=list)

    @property
    def interval_darcy(self) -> tuple[float, float]:
        """``(low, high)`` of the expanded coverage interval, darcy."""
        return (
            self.value_darcy - self.expanded_uncertainty_darcy,
            self.value_darcy + self.expanded_uncertainty_darcy,
        )

    @property
    def relative_expanded_uncertainty(self) -> float:
        """U/k, dimensionless."""
        if self.value_darcy == 0.0:
            return math.inf
        return self.expanded_uncertainty_darcy / abs(self.value_darcy)

    def dominant_components(self, limit: int = 3) -> list[UncertaintyComponent]:
        """The largest contributors, biggest first -- what to improve next."""
        return sorted(self.components, key=lambda c: c.variance_share, reverse=True)[:limit]


# --------------------------------------------------------------------------
# Readings and runs
# --------------------------------------------------------------------------


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
    #: P1 in the Darcy equation, from the inlet transducer.
    inlet_pressure_atm: float
    #: What the outlet transducer read. Always recorded, even when a supplied
    #: value is used as P2 -- it is then the cross-check that the declared
    #: downstream pressure is really what the rig is doing.
    outlet_pressure_atm: float
    #: P2 as actually used in the Darcy equation: the outlet transducer, or the
    #: value supplied via ``run.downstream_pressure``. Equal to
    #: :attr:`outlet_pressure_atm` under the default.
    downstream_pressure_atm: float
    #: Mean pore pressure, (P1 + P2) / 2, absolute -- computed from P2 as used.
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
    compressibility_z: float | None = None
    #: Apparent gas permeability for this sample; ``None`` when the pressure
    #: differential is too small or non-physical to invert.
    permeability_darcy: float | None = None
    #: Rolling-window mean over ``run.averaging_window_s``.
    permeability_darcy_avg: float | None = None
    #: True once the steady-state detector has confirmed stationarity. Only
    #: these readings are representative of the sample's permeability.
    steady_state: bool = False
    #: Consecutive passing windows at the time of this sample.
    steady_state_passes: int = 0
    #: Populated when a sample could not yield a permeability.
    note: str | None = None

    @property
    def delta_pressure_atm(self) -> float:
        """P1 - P2 (absolute), atm, using P2 as it entered the equation."""
        return self.inlet_pressure_atm - self.downstream_pressure_atm


class ExperimentMetadata(BaseModel):
    """Who ran what, on which sample, under what conditions.

    Snapshotted into every run so a result is traceable without the config
    files that produced it.
    """

    model_config = ConfigDict(frozen=True)

    operator: str = ""
    institution: str = ""
    project: str = ""
    experiment_id: str = ""
    notes: str = ""

    sample_id: str = ""
    sample_description: str = ""
    lithology: str = ""
    formation: str = ""
    well: str = ""
    depth: float | None = None
    depth_unit: str = "m"
    porosity_fraction: float | None = None
    porosity_method: str = ""
    grain_density_g_cm3: float | None = None
    bulk_density_g_cm3: float | None = None
    prepared_by: str = ""
    prepared_on: date | None = None

    length_cm: float | None = None
    diameter_cm: float | None = None

    gas_name: str = ""
    confining_pressure: float | None = None
    confining_pressure_unit: str = "MPa"

    #: Which flowmeter was active. Runs on the same plug routinely differ only
    #: in this, so it has to be on the record.
    flowmeter: str = ""
    flowmeter_channel: str = ""
    flowmeter_range: str = ""


class RunSummary(BaseModel):
    """Aggregate result of a completed ``collect`` run.

    A result is only representative when :attr:`steady_state_reached` is true:
    permeability measured while the rig is still equilibrating reflects the
    transient, not the sample.
    """

    model_config = ConfigDict(frozen=True)

    sample_id: str
    gas_name: str
    started_at: datetime
    ended_at: datetime
    duration_s: float
    sample_count: int

    #: True when the detector confirmed stationarity at some point in the run.
    steady_state_reached: bool
    #: The span the result was taken over. ``None`` when steady state was
    #: never reached.
    steady_state_window: SteadyStateWindow | None = None

    #: Steady-state means. Only meaningful when ``steady_state_reached``.
    mean_pressure_atm: float
    permeability_darcy: float
    permeability_stddev_darcy: float
    mean_temperature_c: float
    mean_flow_cm3_s: float
    #: How many samples the steady-state means were taken over.
    averaged_samples: int

    uncertainty: UncertaintyBudget | None = None
    metadata: ExperimentMetadata | None = None
    csv_path: str | None = None
    #: Non-fatal problems seen during the run (serial dropouts, bad samples).
    warnings: list[str] = Field(default_factory=list)

    @property
    def is_representative(self) -> bool:
        """Whether this run may be used as a measurement of the sample."""
        return self.steady_state_reached and self.permeability_darcy > 0.0


#: Historical/alternate name for :class:`RunSummary`.
RunResult = RunSummary


# --------------------------------------------------------------------------
# Klinkenberg
# --------------------------------------------------------------------------


class KlinkenbergPoint(BaseModel):
    """One (mean pressure, apparent permeability) pair feeding the regression."""

    model_config = ConfigDict(frozen=True)

    mean_pressure_atm: float = Field(gt=0.0)
    apparent_permeability_darcy: float
    label: str = ""
    #: Present when the point came from a ``collect`` run rather than a CSV.
    source_path: str | None = None
    sample_id: str | None = None
    #: Standard uncertainty of this point's permeability, darcy. Used to
    #: weight the regression when every point carries one.
    standard_uncertainty_darcy: float | None = None
    #: False when the source run never reached steady state.
    steady_state: bool = True
    #: Canonical key for how P2 was obtained (``"measured"`` or
    #: ``"fixed:<atm>"``). ``None`` when the source run did not record it.
    #: Regressing runs that disagree mixes two different x-axes, since mean
    #: pressure is computed from P2.
    downstream_convention: str | None = None

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
    #: Expanded uncertainty of k_L, propagated from the fit (and from the
    #: per-point uncertainties when they were supplied).
    liquid_permeability_expanded_uncertainty_darcy: float | None = None
    slippage_factor_standard_uncertainty_atm: float | None = None
    coverage_factor: float | None = None
    coverage_probability: float | None = None
    #: True when per-point uncertainties were used to weight the fit.
    weighted: bool = False
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
