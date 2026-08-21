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

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    #: Isothermal compressibility ``-1/V (dV/dP)_T``, in **1/atm**. Distinct
    #: from :attr:`compressibility_z`, which is the dimensionless real-gas
    #: factor. Pulse decay measures permeability through this quantity, so it
    #: must track the pore pressure -- near-ideal gases give ``c ~ 1/P``, which
    #: changes sixfold across a 5-30 atm Klinkenberg series.
    isothermal_compressibility_per_atm: float | None = None
    #: ``"coolprop"`` for a live lookup, ``"fixed"`` when the config bypassed it.
    source: Literal["coolprop", "fixed"] = "coolprop"
    #: Relative standard uncertainty of the viscosity model itself, from
    #: config. Feeds the GUM budget as a Type B component.
    relative_viscosity_uncertainty: float = 0.0
    #: Relative standard uncertainty of the compressibility. Its own field
    #: because an EOS pressure-derivative is not as well determined as the
    #: viscosity correlation it sits beside.
    relative_compressibility_uncertainty: float = 0.0


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
# Pulse decay
# --------------------------------------------------------------------------


class DecayFit(BaseModel):
    """The fitted differential-pressure decay: everything the fit determined.

    ``dP(t) = amplitude * exp(-decay_rate * (t - start)) + offset``. The decay
    rate is the measurand's whole content -- permeability follows from it by a
    closed-form expression -- so the quality diagnostics beside it are not
    decoration: they are how an operator knows whether the number is real.
    """

    model_config = ConfigDict(frozen=True)

    #: alpha, 1/s. Permeability is proportional to it.
    decay_rate_per_s: float
    #: u(alpha) from the fit, which becomes the budget's Type A term. ``None``
    #: when the covariance could not be estimated.
    decay_rate_standard_uncertainty_per_s: float | None = None
    degrees_of_freedom: float = float("inf")
    #: dP extrapolated back to the fit window's start, atm.
    amplitude_atm: float
    #: The fitted constant, atm: two independent transducers have a zero
    #: mismatch, and leaving it out biases a log-linear fit low. ``None`` when
    #: the offset was not fitted.
    offset_atm: float | None = None
    #: Coefficient of determination on dP itself (not log dP), so the two fit
    #: models are directly comparable.
    r_squared: float
    start_elapsed_s: float
    end_elapsed_s: float
    #: Points actually fitted, after binning.
    sample_count: int
    #: Points before binning, so the reduction is visible.
    raw_sample_count: int
    model: Literal["exponential_offset", "log_linear"] = "exponential_offset"
    #: Lag-1 autocorrelation of the residuals. Consecutive DAQ samples are not
    #: independent; a high value after binning means u(alpha) is optimistic.
    residual_autocorrelation: float | None = None
    #: Correlation between the fitted amplitude and offset. When the decay does
    #: not get far, the two trade off and u(alpha) inflates.
    amplitude_offset_correlation: float | None = None

    @property
    def time_constant_s(self) -> float:
        """1/alpha, the time to fall to 1/e of the pulse."""
        return math.inf if self.decay_rate_per_s == 0.0 else 1.0 / self.decay_rate_per_s

    @property
    def relative_standard_uncertainty(self) -> float | None:
        """u(alpha)/alpha, the fit's own precision."""
        u = self.decay_rate_standard_uncertainty_per_s
        if u is None or self.decay_rate_per_s == 0.0:
            return None
        return abs(u / self.decay_rate_per_s)


class PulseDecayStatus(BaseModel):
    """Live view of a decay in progress -- the analogue of a steady-state status.

    A pulse-decay run takes hours, so the operator needs to see where it is and
    when it will finish, not just whether it has.
    """

    model_config = ConfigDict(frozen=True)

    #: ``waiting`` before the pulse, ``transient`` while it is still rising,
    #: ``decaying`` once past the peak, ``complete`` at the stop fraction.
    phase: Literal["waiting", "transient", "decaying", "complete"] = "waiting"
    elapsed_s: float = 0.0
    delta_pressure_atm: float = 0.0
    pulse_at_elapsed_s: float | None = None
    pulse_amplitude_atm: float | None = None
    #: dP/dP0 -- how far the decay has gone.
    decay_fraction: float | None = None
    #: Running log-linear estimate, refined by the one-shot fit at the end.
    decay_rate_per_s: float | None = None
    time_constant_s: float | None = None
    projected_complete_elapsed_s: float | None = None
    fit_sample_count: int = 0
    #: dP rose again after the peak: a leak, a reopened valve, or a thermal
    #: ramp. Otherwise silent, and it invalidates the fit.
    reversed_since_peak: bool = False
    summary: str = "waiting for the pulse"


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
    #: ``None`` in a pulse-decay run, where no flowmeter is read at all. It is
    #: deliberately not 0.0 and not NaN: a zero draws a flat trace that reads
    #: as a dead meter, and NaN slips past the ``f is None or f <= 0`` guards
    #: downstream (``NaN <= 0`` is False) and makes the flowmeter-offset
    #: warning fire on a run that had no flowmeter.
    flow_voltage: float | None = None
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
    #: own reference state. ``None`` in pulse decay -- see :attr:`flow_voltage`.
    flow_cm3_s: float | None = None
    #: Flow rate paired with :attr:`flow_reference_pressure_atm` for the Darcy
    #: equation; the product ``flow * reference pressure`` is the invariant.
    flow_reference_cm3_s: float | None = None
    flow_reference_pressure_atm: float | None = None

    temperature_c: float
    #: False when the serial link produced no fresh value for this sample.
    temperature_ok: bool = True
    #: True when the temperature is a carried-over last-known-good value.
    temperature_stale: bool = False
    #: Seconds since the probe produced this temperature. A sensor slower than
    #: the sample rate is held between conversions, so a small age is normal;
    #: a growing one means the probe has stopped answering.
    temperature_age_s: float | None = None

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
    #: ``dP/dP0`` for the current pulse. ``None`` outside a pulse-decay run, and
    #: before the pulse has been applied.
    decay_fraction: float | None = None
    #: Populated when a sample could not yield a permeability.
    note: str | None = None

    @property
    def delta_pressure_atm(self) -> float:
        """P1 - P2 (absolute), atm, using P2 as it entered the equation.

        In a pulse-decay run this **is** the measured signal: the mode requires
        ``downstream_pressure: measured``, so P2 is the downstream transducer
        and this property is exactly the differential whose decay is fitted.
        """
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


class PulseDecayResult(BaseModel):
    """What a pulse-decay run determined, beyond the permeability itself.

    Nested rather than flattened into :class:`RunSummary`, following the
    ``uncertainty`` precedent: eighteen mode-specific fields do not belong in a
    summary that also describes steady-state runs.
    """

    model_config = ConfigDict(frozen=True)

    # -- the fit ----------------------------------------------------------
    decay_rate_per_s: float
    decay_rate_standard_uncertainty_per_s: float | None = None
    degrees_of_freedom: float = float("inf")
    pulse_amplitude_atm: float
    pulse_at_elapsed_s: float
    #: The two vessel pressures at the instant the pulse was applied -- the
    #: **setup condition**, and the thing to reproduce when the plug is
    #: re-measured. Distinct from the run's mean inlet and outlet: the upstream
    #: vessel decays toward the downstream throughout, so averaging it over the
    #: fit window returns very nearly the pore pressure and describes nothing
    #: anyone could set a regulator to. Optional because sidecars written
    #: before they existed do not carry them.
    initial_upstream_pressure_atm: float | None = None
    initial_downstream_pressure_atm: float | None = None
    fitted_offset_atm: float | None = None
    r_squared: float
    fit_start_elapsed_s: float
    fit_end_elapsed_s: float
    fit_sample_count: int
    fit_model: Literal["exponential_offset", "log_linear"] = "exponential_offset"
    residual_autocorrelation: float | None = None

    # -- the rig it was measured on ---------------------------------------
    #: V1 as assembled for this run: vessel + dead volume + the spacer stack.
    upstream_volume_cm3: float
    downstream_volume_cm3: float
    #: The spacers fitted upstream, as ``"type:length"`` strings. Recorded
    #: because they are part of V1, and because a series measured at different
    #: stack heights is otherwise indistinguishable from one measured at the
    #: same -- the record is what makes that checkable afterwards.
    upstream_spacers: list[str] = Field(default_factory=list)
    spacer_volume_cm3: float = 0.0
    #: ``V_pore / V1`` and ``V_pore / V2``. ``None`` when porosity is unrecorded
    #: and the zero-storage form was used.
    upstream_storage_ratio: float | None = None
    downstream_storage_ratio: float | None = None
    #: ``theta_1``, the first root of the storage equation.
    storage_root: float | None = None
    storage_correction: Literal["brace", "dicker_smits"] = "brace"
    gas_compressibility_per_atm: float

    # -- the rig's own contribution ---------------------------------------
    #: Decay rate the blanked rig produced in its leak test, 1/s. ``None`` when
    #: no leak test was found, which is itself worth knowing.
    leak_rate_per_s: float | None = None
    #: Equivalent permeability of that leak: what the apparatus alone would
    #: report with no sample in it. The number to compare against
    #: :attr:`RunSummary.permeability_darcy` -- if they are close, the run is
    #: measuring the bench.
    leak_equivalent_permeability_darcy: float | None = None
    #: Where that leak test came from, for traceability.
    leak_test_source: str | None = None
    #: True when the leak rate was taken off the measured one rather than
    #: merely compared against it.
    leak_subtracted: bool = False

    @property
    def leak_fraction(self) -> float | None:
        """The share of the measured decay that came from the rig."""
        if self.leak_rate_per_s is None or self.decay_rate_per_s == 0.0:
            return None
        # Against the *uncorrected* rate, so the figure means the same thing
        # whether or not the correction was applied.
        total = self.decay_rate_per_s + (
            self.leak_rate_per_s if self.leak_subtracted else 0.0
        )
        return abs(self.leak_rate_per_s / total) if total else None

    @property
    def time_constant_s(self) -> float:
        """1/alpha -- how long the decay takes to fall to 1/e."""
        return math.inf if self.decay_rate_per_s == 0.0 else 1.0 / self.decay_rate_per_s

    @property
    def relative_standard_uncertainty(self) -> float | None:
        """u(alpha)/alpha, which is the measurement's own precision."""
        u = self.decay_rate_standard_uncertainty_per_s
        if u is None or self.decay_rate_per_s == 0.0:
            return None
        return abs(u / self.decay_rate_per_s)


class RunSummary(BaseModel):
    """Aggregate result of a completed ``collect`` run.

    A result is only representative when :attr:`measurement_confirmed` is true:
    a steady-state permeability measured while the rig is still equilibrating
    reflects the transient rather than the sample, and a pulse-decay run whose
    fit was rejected has not measured anything either.
    """

    model_config = ConfigDict(frozen=True)

    sample_id: str
    gas_name: str
    started_at: datetime
    ended_at: datetime
    duration_s: float
    sample_count: int
    #: Which method produced this result.
    method: Literal["steady_state", "pulse_decay"] = "steady_state"
    #: Whether this run measured the sample or characterised the rig. A
    #: ``leak_test`` reports the equivalent permeability the apparatus alone
    #: would fake, and is never a Klinkenberg point.
    purpose: Literal["measurement", "leak_test"] = "measurement"

    #: True when the detector confirmed stationarity at some point in the run.
    #: Always False for a pulse-decay run -- there is no steady state to reach,
    #: and saying otherwise in a field with this name would be a lie.
    steady_state_reached: bool
    #: The span the result was taken over. ``None`` when steady state was
    #: never reached, and always ``None`` in pulse decay.
    steady_state_window: SteadyStateWindow | None = None
    #: Whether the run met its **own** method's criterion for a usable
    #: measurement: a confirmed steady window, or an accepted decay fit.
    #: Defaulted from :attr:`steady_state_reached` when absent, so sidecars
    #: written before pulse decay existed still read correctly.
    measurement_confirmed: bool | None = None

    #: Steady-state means. Only meaningful when the measurement was confirmed.
    mean_pressure_atm: float
    #: The two pressures behind :attr:`mean_pressure_atm`, averaged over the
    #: same window, so ``P_mean`` is exactly their midpoint and a summary table
    #: showing all three reads consistently. ``mean_downstream_pressure_atm``
    #: is the P2 the equation *used* -- the transducer, or the declared number
    #: when ``downstream_pressure`` is supplied -- which is why it is named for
    #: the convention rather than for the outlet port.
    #:
    #: Optional because sidecars written before these existed do not carry
    #: them, and they cannot be recovered from a mean: such a run reports them
    #: as unknown rather than inventing a split.
    mean_inlet_pressure_atm: float | None = None
    mean_downstream_pressure_atm: float | None = None
    permeability_darcy: float
    permeability_stddev_darcy: float
    mean_temperature_c: float
    #: ``None`` in pulse decay, where no flowmeter is read.
    mean_flow_cm3_s: float | None = None
    #: How many samples the steady-state means were taken over.
    averaged_samples: int

    #: The decay and the vessels behind it. ``None`` for a steady-state run.
    pulse_decay: PulseDecayResult | None = None

    uncertainty: UncertaintyBudget | None = None
    metadata: ExperimentMetadata | None = None
    csv_path: str | None = None
    #: Non-fatal problems seen during the run (serial dropouts, bad samples).
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _default_measurement_confirmed(self) -> RunSummary:
        if self.measurement_confirmed is None:
            object.__setattr__(
                self, "measurement_confirmed", self.steady_state_reached
            )
        return self

    @property
    def is_representative(self) -> bool:
        """Whether this run may be used as a measurement of the sample."""
        return bool(self.measurement_confirmed) and self.permeability_darcy > 0.0


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
    #: Mean flow behind this point, cm3/s. ``None`` for points from a CSV or
    #: from a sidecar written before this was recorded. Flow that barely moves
    #: while pressure moves a lot is the signature of a meter reporting its own
    #: zero offset rather than the sample.
    flow_cm3_s: float | None = None
    #: Canonical key for how P2 was obtained (``"measured"`` or
    #: ``"fixed:<atm>"``). ``None`` when the source run did not record it.
    #: Regressing runs that disagree mixes two different x-axes, since mean
    #: pressure is computed from P2.
    downstream_convention: str | None = None
    #: Which method produced this point. A sibling of
    #: :attr:`downstream_convention` rather than a value of it: a pulse-decay
    #: run's honest answer to "how was P2 obtained" is "measured", so folding
    #: the method into that field would produce a nonsense refusal message.
    method: str | None = None

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


# --------------------------------------------------------------------------
# Comparing two measurements
# --------------------------------------------------------------------------


class ComponentPairing(BaseModel):
    """One input quantity, matched across two measurements.

    The audit trail for a cancellation claim. When ``shared`` is true the same
    physical error produced both readings, so only the *difference* of their
    contributions survives into the ratio -- see :mod:`gasperm.comparison`.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    name: str
    #: Whether one physical error produced both readings.
    shared: bool
    #: Why it was or was not treated as shared, in words.
    reason: str
    #: GUM Type A on either side, i.e. statistical scatter. Kept because a
    #: comparison of two *fitted* quantities already carries the scatter through
    #: the fit's own standard error, so counting these again would double it.
    type_a: bool = False
    relative_contribution_before: float
    relative_contribution_after: float
    #: What this input contributes to the **ratio's** relative variance: the
    #: squared difference of contributions when shared, their squared sum when
    #: independent.
    variance_contribution: float
    degrees_of_freedom_before: float = float("inf")
    degrees_of_freedom_after: float = float("inf")

    @property
    def degrees_of_freedom(self) -> float:
        """The limiting dof of this input, for display."""
        return min(self.degrees_of_freedom_before, self.degrees_of_freedom_after)

    @property
    def welch_terms(self) -> list[tuple[float, float]]:
        """``(variance, dof)`` terms this input adds to Welch-Satterthwaite.

        An **independent** pairing contributes *two* terms, one per side, each
        with its own degrees of freedom -- collapsing them into a single term
        at the smaller dof understates the effective degrees of freedom and so
        inflates the coverage factor. A **shared** pairing is one physical
        quantity and therefore one term.
        """
        if self.shared:
            return [(self.variance_contribution, self.degrees_of_freedom)]
        return [
            (self.relative_contribution_before**2, self.degrees_of_freedom_before),
            (self.relative_contribution_after**2, self.degrees_of_freedom_after),
        ]

    @property
    def cancelled_fraction(self) -> float:
        """How much of this input's contribution the pairing removed, in [0, 1]."""
        independent = (
            self.relative_contribution_before**2 + self.relative_contribution_after**2
        )
        if independent <= 0.0:
            return 0.0
        return max(0.0, 1.0 - self.variance_contribution / independent)


class QuantityChange(BaseModel):
    """A before/after pair, its change, and whether that change is real.

    ``significant`` is the whole point: a percentage on its own reads as a
    finding whether or not the measurement could have resolved it.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    symbol: str
    unit: str
    before: float
    after: float
    difference: float
    ratio: float
    percent_change: float
    #: Of the **ratio**, after the shared inputs have cancelled.
    relative_standard_uncertainty: float
    relative_expanded_uncertainty: float
    standard_uncertainty: float
    coverage_factor: float
    coverage_probability: float
    effective_degrees_of_freedom: float
    #: The smallest change this comparison could have resolved, in percent. A
    #: null result means nothing without it.
    minimum_detectable_percent: float
    significant: bool
    #: Whether the same-plug treatment was applied.
    paired: bool
    notes: list[str] = Field(default_factory=list)

    @property
    def verdict(self) -> str:
        """One line an operator can read without decoding the numbers."""
        if not math.isfinite(self.percent_change):
            return "not computable"
        if not math.isfinite(self.relative_expanded_uncertainty):
            return f"{self.percent_change:+.2f}% (no uncertainty available)"
        direction = "increased" if self.percent_change > 0 else "decreased"
        if not self.significant:
            return (
                f"{self.percent_change:+.2f}% +/- "
                f"{self.relative_expanded_uncertainty * 100.0:.2f}% -- NOT "
                "distinguishable from no change"
            )
        return (
            f"{direction} {abs(self.percent_change):.2f}% +/- "
            f"{self.relative_expanded_uncertainty * 100.0:.2f}% -- SIGNIFICANT"
        )


class ConditionCheck(BaseModel):
    """One thing that had to match for a difference to mean anything."""

    model_config = ConfigDict(frozen=True)

    key: str
    label: str
    before: str
    after: str
    matched: bool
    #: A mismatch that makes the comparison meaningless rather than noisier.
    blocking: bool = False
    #: Budget symbols whose cancellation this mismatch invalidates.
    voids_symbols: list[str] = Field(default_factory=list)
    advice: str = ""


class GroupSummary(BaseModel):
    """What one side of a comparison consisted of."""

    model_config = ConfigDict(frozen=True)

    label: str
    sample_id: str | None
    run_count: int
    mean_pressures_atm: list[float] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    gases: list[str] = Field(default_factory=list)
    flowmeters: list[str] = Field(default_factory=list)
    liquid_permeability_darcy: float | None = None
    slippage_factor_atm: float | None = None
    r_squared: float | None = None
    porosity_fraction: float | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None


class ComparisonResult(BaseModel):
    """Everything that differs between two sets of runs, with its uncertainty."""

    model_config = ConfigDict(frozen=True)

    before: GroupSummary
    after: GroupSummary
    #: Whether both sides measured the same core plug, which decides whether
    #: the plug's own inputs cancel.
    paired: bool
    coverage_probability: float
    changes: list[QuantityChange] = Field(default_factory=list)
    conditions: list[ConditionCheck] = Field(default_factory=list)
    component_pairings: list[ComponentPairing] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def change(self, symbol: str) -> QuantityChange | None:
        """The first reported change for ``symbol``, if there is one."""
        return next((c for c in self.changes if c.symbol == symbol), None)

    @property
    def mismatched_conditions(self) -> list[ConditionCheck]:
        """Conditions that differed between the two campaigns."""
        return [c for c in self.conditions if not c.matched]

    @property
    def cancelled(self) -> list[ComponentPairing]:
        """Inputs whose error the pairing removed from the comparison."""
        return [p for p in self.component_pairings if p.shared]


def is_finite(value: float | None) -> bool:
    """True when ``value`` is a real, usable number (not ``None``/NaN/inf)."""
    return value is not None and math.isfinite(value)
