"""The experiment: who is running it, on what gas, at what confining pressure.

File: ``run.yaml``. This is the file that changes most often -- a new pressure
step, a different working gas, a different operator -- which is exactly why it
is separate from the rig description and the sample description.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from gasperm import units
from gasperm.screens import WindowMode
from gasperm.config.common import (
    PressureUnit,
    UncertaintySpec,
    _Base,
    validated_pressure_unit,
)

__all__ = [
    "GasConfig",
    "SteadyStateConfig",
    "UncertaintyReportConfig",
    "PulseDecayConfig",
    "SpacerFitting",
    "LivePlotConfig",
    "PLOT_PANELS",
    "MEASUREMENT_METHODS",
    "RunConfig",
]

#: The two measurement methods. ``steady_state`` drives gas through the plug at
#: a fixed differential and measures the flow; ``pulse_decay`` watches a small
#: differential decay between two closed vessels and measures no flow at all,
#: which is what makes it usable below about ten microdarcy.
MEASUREMENT_METHODS: tuple[str, ...] = ("steady_state", "pulse_decay")

#: Every quantity the live plot can give a panel to, in their natural order.
#: Each gets its **own** stacked axes -- pressures are not overlaid, because
#: the point of the live view is watching one signal settle at a time.
PLOT_PANELS: tuple[str, ...] = (
    "inlet_pressure",
    "outlet_pressure",
    "delta_pressure",
    "decay_fraction",
    "flow",
    "temperature",
    "permeability",
)

PlotPanel = Literal[
    "inlet_pressure",
    "outlet_pressure",
    "delta_pressure",
    "decay_fraction",
    "flow",
    "temperature",
    "permeability",
]

#: Panels that only make sense for one method. The live plot filters by
#: ``run.method`` so a default panel list works for both without asking the
#: operator to curate it per run.
METHOD_ONLY_PANELS: dict[str, str] = {
    "flow": "steady_state",
    "delta_pressure": "pulse_decay",
    "decay_fraction": "pulse_decay",
}


class GasConfig(_Base):
    """Working gas and where its thermophysical properties come from."""

    #: Any CoolProp fluid name: "Nitrogen", "Air", "CarbonDioxide", "Methane"...
    name: str = "Nitrogen"
    properties_source: Literal["coolprop", "fixed"] = "coolprop"
    #: Only consulted when ``properties_source == "fixed"``. Setting this
    #: bypasses the live (T, P) lookup; record why in ``fixed_reason``.
    fixed_viscosity_cp: float | None = Field(default=None, gt=0.0)
    fixed_reason: str = ""
    #: Relative standard uncertainty of the viscosity model, dimensionless.
    #: CoolProp's transport correlations for common gases near ambient are
    #: typically quoted at a few tenths of a percent to ~1%; 0.01 is a safe
    #: default. Enters the GUM budget as a Type B component.
    viscosity_relative_uncertainty: float = Field(default=0.01, ge=0.0)
    #: The same for the isothermal compressibility, which only pulse-decay runs
    #: consume. Its own field because an EOS pressure-derivative is not as well
    #: determined as the viscosity correlation.
    compressibility_relative_uncertainty: float = Field(default=0.01, ge=0.0)
    #: Divide the reference-state flow by Z when pairing ``Q_ref * P_ref``.
    #: The pairing is exact for an ideal gas; at a few atm and ambient
    #: temperature Z differs from 1 by well under a percent for N2, but for
    #: CO2 or high pressures it matters.
    real_gas_correction: bool = False

    @model_validator(mode="after")
    def _fixed_needs_value(self) -> GasConfig:
        if self.properties_source == "fixed" and self.fixed_viscosity_cp is None:
            raise ValueError(
                "gas.properties_source is 'fixed' but gas.fixed_viscosity_cp is not "
                "set; either provide a viscosity in cP or switch back to 'coolprop'"
            )
        return self


class SteadyStateConfig(_Base):
    """When the rig counts as equilibrated.

    Permeability measured while pressures and flow are still settling
    describes the transient, not the rock, so the detector gates what gets
    reported. Two independent criteria must hold on every monitored signal,
    over ``required_windows`` consecutive windows:

    * **scatter** -- the coefficient of variation within the window is at or
      below ``relative_stddev_tolerance``;
    * **drift** -- the fractional change an OLS line predicts across the
      window is at or below ``relative_drift_tolerance``.

    The drift test is the one that matters: a slowly ramping signal has small
    scatter within any short window and would pass a scatter-only test
    indefinitely.
    """

    enabled: bool = True
    #: Trailing window each test is evaluated over.
    window_s: float = Field(default=30.0, gt=0.0)
    #: Consecutive passing windows required before steady state is declared.
    #: More than one guards against a momentarily flat patch in a slow ramp.
    required_windows: int = Field(default=3, ge=1)
    #: Minimum samples inside a window for the tests to be meaningful.
    min_samples: int = Field(default=10, ge=3)
    #: Ignore this much of the start of a run outright -- the initial pressure
    #: build-up is never steady and testing it just wastes windows.
    settling_time_s: float = Field(default=0.0, ge=0.0)
    relative_stddev_tolerance: float = Field(default=0.02, gt=0.0)
    relative_drift_tolerance: float = Field(default=0.01, gt=0.0)
    #: Also require the OLS slope to be statistically indistinguishable from
    #: zero at this significance. Set to ``null`` to use the drift bound alone.
    slope_significance: float | None = Field(default=None, gt=0.0, lt=1.0)
    #: Which signals must be stationary. Permeability alone is not enough:
    #: it can be momentarily flat while pressure and flow drift together.
    signals: list[Literal["permeability", "inlet_pressure", "flow", "temperature"]] = Field(
        default_factory=lambda: ["permeability", "inlet_pressure", "flow"]
    )
    #: Give up waiting after this long and end the run unsteady rather than
    #: running forever. ``null`` waits indefinitely.
    max_wait_s: float | None = Field(default=None, gt=0.0)
    #: Warn when the steady window is shorter than this many characteristic
    #: pressure-equilibration times ``t ~ phi mu L^2 / (k P_mean)``. Tight rock
    #: takes hours to equilibrate -- around two at 1 uD and 5 atm mean pressure
    #: -- while these criteria can declare steady state in ninety seconds, so a
    #: plateau can be real and the rig still not equilibrated. ``null``
    #: disables the check, which also happens when the sample's porosity is
    #: unrecorded.
    equilibration_factor: float | None = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def _at_least_one_signal(self) -> SteadyStateConfig:
        if self.enabled and not self.signals:
            raise ValueError(
                "steady_state.signals is empty, so nothing would be tested. List at "
                "least one signal, or set steady_state.enabled: false."
            )
        return self


class UncertaintyReportConfig(_Base):
    """How the GUM budget is evaluated and reported."""

    enabled: bool = True
    #: Level of confidence for the expanded uncertainty ``U = k * u_c``.
    coverage_probability: float = Field(default=0.95, gt=0.0, lt=1.0)
    #: Include the Type A term from the scatter of the steady-state window.
    #: Leave on: it is the only term that reflects the rig's actual behaviour
    #: rather than its datasheets.
    include_type_a: bool = True
    #: Override the coverage factor instead of deriving it from Student-t at
    #: the effective degrees of freedom. ``null`` derives it (GUM annex G).
    fixed_coverage_factor: float | None = Field(default=None, gt=0.0)
    #: Say so when one input's relative contribution ``|c_i| u(x_i)/x_i``
    #: exceeds this. Not a share of the budget -- a healthy budget can easily
    #: have one term at half the variance -- but its absolute size: a single
    #: term worth 25 % of the result means the result is not a measurement.
    #: The usual cause is a flowmeter running at a fraction of a percent of
    #: full scale. ``null`` disables the check.
    max_component_contribution: float | None = Field(default=0.25, gt=0.0)


class SpacerFitting(_Base):
    """One hollow spacer actually fitted upstream of the core, for this run.

    The ``type`` names a bore defined in ``hardware.reservoirs.spacer_types``;
    the ``length`` is that particular spacer's, in the type's own
    ``dimension_unit``. Two measurements, split the way they are actually
    established: the bore is a property of a set of parts, the length is a
    property of the one you just put in.
    """

    type: str
    length: float = Field(gt=0.0)

    def __str__(self) -> str:  # for messages and the console
        return f"{self.type}:{self.length:g}"


class PulseDecayConfig(_Base):
    """When a pulse-decay run is finished, and which part of it is the answer.

    There is no ``enabled`` flag here: the mode switch is :attr:`RunConfig.method`.

    The defaults are set by what the fit actually buys. Simulating the decay of
    a 1 uD plug on this rig's vessels, the precision of alpha against how far
    the decay was allowed to run:

    =================  ==========  ========
    stop at dP/dP0     run length  u(alpha)
    =================  ==========  ========
    0.9                1.5 h       7.4 %
    0.7                5.0 h       2.1 %
    0.5                9.7 h       0.9 %
    0.3                16.8 h      0.5 %
    0.05               41.8 h      0.7 %
    =================  ==========  ========

    Past about half a decade the samples are noise-dominated, so they add
    scatter rather than information -- running to the textbook 5 % costs four
    times the time and is slightly *worse*. Hence a fit window of 0.90 to 0.50
    and a stop at 0.40, not 0.05.
    """

    # -- the rig, as assembled for THIS run --------------------------------
    #: The hollow spacers stacked upstream of the core face, in order, each as
    #: ``{type, length}``. Their internal volume is part of V1, and V1 is in
    #: the equation, so a stack recorded wrongly moves the result.
    #:
    #: This lives here rather than in hardware.yaml because the stack is made
    #: up to suit the plug in the holder and changes between runs without the
    #: bench changing. The bores are in ``hardware.reservoirs.spacer_types``.
    #: Override per run with ``--spacer TYPE:LENGTH`` (repeat it to stack).
    upstream_spacers: list[SpacerFitting] = Field(default_factory=list)

    # -- the pulse --------------------------------------------------------
    #: Smallest differential that counts as a pulse having been applied. Below
    #: this the monitor is still waiting, so transducer noise cannot be
    #: mistaken for the operator opening the valve.
    min_pulse_pressure: float = Field(default=20.0, gt=0.0)
    pulse_pressure_unit: PressureUnit = "kPa"
    #: Largest ``dP0/P_mean`` the small-pulse linearisation tolerates. Above
    #: this the compressibility and viscosity are not constant across the decay
    #: and a single exponential no longer describes it.
    max_pulse_fraction: float = Field(default=0.10, gt=0.0, le=1.0)
    #: End the run when ``dP/dP0`` falls below this -- the analogue of
    #: ``run.stop_after_steady_s``. Deliberately below ``fit_end_fraction``, so
    #: there is data past the fit window to pin the fitted offset.
    stop_below_fraction: float = Field(default=0.40, gt=0.0, lt=1.0)
    #: Give up after this long rather than running forever, the counterpart of
    #: ``steady_state.max_wait_s``. ``null`` waits indefinitely.
    max_decay_s: float | None = Field(default=None, gt=0.0)

    # -- the fit ----------------------------------------------------------
    #: Fit between these fractions of the pulse. The top is skipped because of
    #: the valve transient AND because the single exponential is asymptotic --
    #: higher decay modes are still dying, and including them biases alpha high.
    fit_start_fraction: float = Field(default=0.90, gt=0.0, le=1.0)
    fit_end_fraction: float = Field(default=0.50, gt=0.0, lt=1.0)
    #: Bin the decay to this cadence before fitting. Consecutive DAQ samples are
    #: not independent -- thermal drift and 1/f noise -- so an unbinned fit
    #: reports an optimistic u(alpha) and an absurd effective dof. ``null``
    #: fits every sample.
    fit_bin_s: float | None = Field(default=1.0, gt=0.0)
    #: Fit the constant offset. Two independent transducers have a zero
    #: mismatch that a log-linear fit turns into a low alpha; leave this on.
    fit_offset: bool = True
    min_fit_samples: int = Field(default=20, ge=5)
    #: Below this R^2 the decay is not a single exponential and the run is
    #: reported as unconfirmed.
    min_r_squared: float = Field(default=0.98, gt=0.0, le=1.0)
    #: Warn above this lag-1 residual autocorrelation after binning: it means
    #: the residuals still have structure and u(alpha) is understated.
    max_residual_autocorrelation: float = Field(default=0.5, gt=0.0, lt=1.0)

    # -- the model --------------------------------------------------------
    #: ``auto`` applies the Dicker-Smits storage correction when the sample's
    #: porosity is recorded and falls back to Brace with a warning when it is
    #: not. ``brace`` and ``dicker_smits`` force one.
    storage_correction: Literal["auto", "brace", "dicker_smits"] = "auto"
    #: Warn when the pore volume exceeds this fraction of either vessel while
    #: the zero-storage form is in use.
    max_storage_ratio: float = Field(default=0.05, gt=0.0)

    # -- the leak test ----------------------------------------------------
    #: How long a ``--leak-test`` run records for. A leak test is a *fixed
    #: observation*, not a decay to be waited out: on a tight rig the ideal
    #: outcome is that nothing happens, so there is no completion signal to
    #: stop on and it must be given a duration. ``null`` falls back to
    #: ``run.duration_s``, and a leak test with neither is refused.
    leak_test_duration_s: float | None = Field(default=3600.0, gt=0.0)
    #: Largest share of the measured decay that may come from the rig rather
    #: than the rock before the measurement stops being about the sample.
    #: Compared as an equivalent permeability, so it reads directly against k.
    max_leak_fraction: float = Field(default=0.05, gt=0.0, le=1.0)
    #: What to do with a recorded leak rate.
    #:
    #: ``off``
    #:     Compare and warn only. The default, deliberately: a leak that
    #:     changed between the test and the run would silently corrupt a
    #:     subtracted result, and the correction is worth less than knowing
    #:     the leak is small.
    #: ``subtract``
    #:     Take the leak rate off the measured one. Only defensible when the
    #:     leak is linear and stable, and it is still reported both ways.
    leak_correction: Literal["off", "subtract"] = "off"
    #: Warn when the leak test's mean pressure differs from the run's by more
    #: than this. Leak conductance is pressure-dependent, so a test done at a
    #: different charge does not describe this run.
    leak_pressure_tolerance: float = Field(default=0.2, gt=0.0)

    # -- planning ---------------------------------------------------------
    #: Roughly expected permeability, used ONLY to predict the run's duration at
    #: startup. Never enters a result.
    expected_permeability: float | None = Field(default=None, gt=0.0)
    expected_permeability_unit: str = "uD"

    @field_validator("expected_permeability_unit")
    @classmethod
    def _check_expected_unit(cls, value: str) -> str:
        units.darcy_to(1.0, value)
        return value

    @field_validator("pulse_pressure_unit")
    @classmethod
    def _check_pulse_unit(cls, value: str) -> str:
        return validated_pressure_unit(value)

    @model_validator(mode="after")
    def _fit_window_is_ordered(self) -> PulseDecayConfig:
        if self.fit_end_fraction >= self.fit_start_fraction:
            raise ValueError(
                f"pulse_decay.fit_end_fraction ({self.fit_end_fraction}) must be "
                f"below fit_start_fraction ({self.fit_start_fraction}): the decay "
                "is fitted from the higher fraction down to the lower one."
            )
        if self.stop_below_fraction > self.fit_end_fraction:
            raise ValueError(
                f"pulse_decay.stop_below_fraction ({self.stop_below_fraction}) is "
                f"above fit_end_fraction ({self.fit_end_fraction}), so the run "
                "would end before the fit window closed. Stop at or below the end "
                "of the window -- ideally a little below, so there is data past it "
                "to pin the fitted offset."
            )
        return self

    @property
    def min_pulse_pressure_atm(self) -> float:
        """The pulse threshold in the internal unit."""
        return units.to_atm(self.min_pulse_pressure, self.pulse_pressure_unit)

    @property
    def expected_permeability_darcy(self) -> float | None:
        """The planning permeability in darcy, or ``None`` when unset."""
        if self.expected_permeability is None:
            return None
        return units.darcy_from(
            self.expected_permeability, self.expected_permeability_unit
        )


class LivePlotConfig(_Base):
    """The optional ``--plot`` window.

    Display only: nothing here touches a reading, a CSV or a reported result,
    and the whole window can fail without disturbing the run.
    """

    #: One stacked panel per entry, drawn top to bottom in this order.
    panels: list[PlotPanel] = Field(default_factory=lambda: list(PLOT_PANELS))
    #: Trailing seconds to display. ``null`` shows the whole run from t0.
    #: A window is what you want while waiting for a plateau; from t0 is what
    #: you want to see how far the rig has come since pressure was applied.
    window_s: float | None = Field(default=None, gt=0.0)
    #: Draw the steady-state criterion bands and the fitted drift line.
    show_criteria: bool = True
    #: Print each panel's most recent value in its top-right corner. The trace
    #: shows the shape and the axis gives the scale, but reading a number off a
    #: plot by eye is guesswork -- and it is the number, not the shape, that
    #: gets written in the lab book.
    show_last_value: bool = True
    #: Minimum wall-clock gap between redraws. The acquisition loop only ever
    #: appends to a buffer; this is what keeps matplotlib off its critical path.
    redraw_interval_s: float = Field(default=0.5, gt=0.0)
    #: Points held per series. The trailing-window view keeps every sample up
    #: to this many; the from-t0 view decimates to stay inside it, so a
    #: multi-hour run still spans the whole x-axis without unbounded memory.
    max_points: int = Field(default=3600, gt=1)
    #: Which monitor to open the window on, 1-based, or ``null`` to leave it
    #: wherever the desktop puts it. A rig bench usually has the console on one
    #: screen and the live plot left running for hours on the other.
    monitor: int | None = Field(default=None, ge=1)
    #: How much of that screen to take. ``fullscreen`` covers the monitor
    #: including the taskbar and drops the title bar; ``maximised`` fills the
    #: work area and keeps it, which is usually what you want for a window you
    #: may need to close.
    window: WindowMode = "normal"

    @field_validator("window", mode="before")
    @classmethod
    def _accept_either_spelling(cls, value: str) -> str:
        return "maximised" if value == "maximized" else value

    @field_validator("panels")
    @classmethod
    def _panels_are_usable(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError(
                "plot.panels is empty; list at least one of: " + ", ".join(PLOT_PANELS)
            )
        duplicates = {name for name in value if value.count(name) > 1}
        if duplicates:
            raise ValueError(
                "plot.panels repeats " + ", ".join(sorted(duplicates)) + "; each "
                "quantity gets exactly one panel."
            )
        return value


class RunConfig(_Base):
    """Experiment metadata plus everything about how ``collect`` executes.

    File: ``run.yaml``.
    """

    # -- who and what -----------------------------------------------------
    operator: str = ""
    institution: str = ""
    project: str = ""
    experiment_id: str = ""
    notes: str = ""

    # -- test conditions --------------------------------------------------
    #: How this run measures permeability.
    #:
    #: ``steady_state``
    #:     Drive gas through the plug at a fixed differential and measure the
    #:     flow: the compressible Darcy equation. Needs a flowmeter that can
    #:     resolve the flow, which below ~10 uD it generally cannot.
    #: ``pulse_decay``
    #:     Apply a small pulse to a closed upstream vessel and watch the
    #:     differential decay into a closed downstream one. **No flow is
    #:     measured at all**, which is what makes it usable at a microdarcy.
    #:     Requires ``downstream_pressure: measured`` and the two vessel
    #:     volumes in ``hardware.reservoirs``.
    method: Literal["steady_state", "pulse_decay"] = "steady_state"
    #: What this run is for.
    #:
    #: ``measurement``
    #:     A permeability measurement, the normal case.
    #: ``leak_test``
    #:     The pre-step: the plug blanked or bypassed, the same pulse applied,
    #:     and the differential watched for a fixed time. Whatever decays there
    #:     is the rig -- leaks and thermal drift -- and it sets the floor below
    #:     which a sample's decay cannot be distinguished from the apparatus.
    #:     Reported as the equivalent permeability the rig alone would fake, so
    #:     it compares directly against the k you are trying to measure.
    #:
    #:     Excluded from ``klinkenberg`` discovery: it is a property of the
    #:     bench, not a point on the sample's curve.
    purpose: Literal["measurement", "leak_test"] = "measurement"
    gas: GasConfig = Field(default_factory=GasConfig)
    #: Which meter in ``hardware.flowmeters`` this run uses. ``null`` takes the
    #: rig's ``default_flowmeter``. This lives here, not in hardware.yaml,
    #: because swapping to the high-range meter for a high-flow pressure step
    #: is an experiment decision -- the bench has not changed.
    flowmeter: str | None = None
    #: Confining/overburden pressure the plug is held at. A run-level quantity:
    #: the same plug is routinely measured at several confining pressures.
    #: Typically MPa-scale while pore pressure is kPa-scale, hence its own unit.
    confining_pressure: float | None = None
    confining_pressure_unit: PressureUnit = "MPa"

    # -- downstream pressure (P2) -----------------------------------------
    #: What to use as P2 in the Darcy equation.
    #:
    #: ``"measured"`` (the default) reads the outlet transducer, which is what
    #: a normally-plumbed rig wants. A **number** overrides it with a value the
    #: operator supplies, in ``downstream_pressure_unit`` -- for a rig whose
    #: outlet vents to atmosphere, where the transducer reads noise around zero
    #: or is not fitted at all.
    #:
    #: This is not a cosmetic choice: P2 sets the apparent permeability through
    #: ``P1^2 - P2^2`` *and* the mean pore pressure, which is the Klinkenberg
    #: regression's own abscissa. Runs that used different conventions are
    #: refused by ``klinkenberg`` unless explicitly allowed.
    downstream_pressure: float | Literal["measured"] = "measured"
    downstream_pressure_unit: PressureUnit = "kPa"
    #: Uncertainty of a supplied downstream pressure, in
    #: ``downstream_pressure_unit``. Deliberately its own spec rather than
    #: sharing ``atmospheric_pressure_uncertainty``: that one also feeds the
    #: gauge-to-absolute conversion of *both* transducers, so widening this
    #: figure to admit "the back-pressure number is an estimate" must not
    #: silently widen P1 as well.
    downstream_pressure_uncertainty: UncertaintySpec = Field(
        default_factory=lambda: UncertaintySpec(
            kind="absolute", value=0.1, source="operator-supplied downstream pressure"
        )
    )

    # -- ambient reference ------------------------------------------------
    #: Local atmospheric pressure. This is *not* P2 -- see
    #: ``downstream_pressure`` for that. It converts **gauge** transducer
    #: readings to the absolute pressures the Darcy equation needs, and serves
    #: as the flowmeter's reference when ``flowmeter.actual_pressure_source``
    #: is ``atmospheric``. With absolute transducers on both ports and a
    #: standard-basis meter it is unused.
    atmospheric_pressure: float = Field(default=101.325, gt=0.0)
    atmospheric_pressure_unit: PressureUnit = "kPa"
    #: Uncertainty of that ambient value, in ``atmospheric_pressure_unit``.
    #: Contributes only through the gauge-to-absolute conversion above.
    atmospheric_pressure_uncertainty: UncertaintySpec = Field(
        default_factory=lambda: UncertaintySpec(
            kind="absolute", value=0.1, source="local barometric reading"
        )
    )

    # -- analysis ---------------------------------------------------------
    steady_state: SteadyStateConfig = Field(default_factory=SteadyStateConfig)
    #: Only consulted when ``method`` is ``pulse_decay``.
    pulse_decay: PulseDecayConfig = Field(default_factory=PulseDecayConfig)
    uncertainty: UncertaintyReportConfig = Field(default_factory=UncertaintyReportConfig)
    #: Rolling window for the live permeability display. The reported result
    #: comes from the detected steady-state window, not from this.
    averaging_window_s: float = Field(default=5.0, gt=0.0)

    # -- output -----------------------------------------------------------
    output_dir: str = "./runs"
    #: Display-only units. Independent of calibration units and of the
    #: internal CGS calculation.
    display_pressure_unit: PressureUnit = "kPa"
    display_permeability_unit: str = "mD"
    display_flow_unit: str = "sccm"
    #: The optional live window. Only consulted when ``--plot`` is passed.
    plot: LivePlotConfig = Field(default_factory=LivePlotConfig)

    # -- stop conditions --------------------------------------------------
    #: ``null`` on both means run until Ctrl+C (or until steady state has held
    #: for ``stop_after_steady_s``, if that is set).
    duration_s: float | None = Field(default=None, gt=0.0)
    max_samples: int | None = Field(default=None, gt=0)
    #: Seconds of *confirmed* steady state to record before ending the run.
    #: ``null`` runs until Ctrl+C.
    #:
    #: The clock starts when the detector declares steady state -- not when the
    #: plateau began, which is only known in hindsight -- so this is the soak
    #: time on top of whatever established steadiness. ``0`` stops the moment
    #: it is confirmed. If the rig leaves steady state the clock resets, since
    #: a hold that was interrupted did not last.
    stop_after_steady_s: float | None = Field(default=None, ge=0.0)
    #: Flush the CSV every N samples so a crash cannot lose the whole run.
    flush_every_n: int = Field(default=20, gt=0)

    @field_validator(
        "atmospheric_pressure_unit",
        "display_pressure_unit",
        "confining_pressure_unit",
        "downstream_pressure_unit",
    )
    @classmethod
    def _check_pressure_unit(cls, value: str) -> str:
        return validated_pressure_unit(value)

    @field_validator("display_permeability_unit")
    @classmethod
    def _check_perm_unit(cls, value: str) -> str:
        units.darcy_to(1.0, value)  # raises ValueError on an unknown unit
        return value

    @field_validator("display_flow_unit")
    @classmethod
    def _check_flow_unit(cls, value: str) -> str:
        units.flow_to_cm3_s(1.0, value)
        return value

    @model_validator(mode="after")
    def _downstream_pressure_is_usable(self) -> RunConfig:
        """A supplied P2 must be positive and absolute.

        Caught here rather than at the first sample: a mistyped ``0`` would
        otherwise make *every* reading unusable and only surface as "no sample
        produced a usable permeability" minutes into the run.
        """
        if not isinstance(self.downstream_pressure, str) and self.downstream_pressure <= 0.0:
            raise ValueError(
                f"downstream_pressure must be a positive absolute pressure, got "
                f"{self.downstream_pressure}. Use 'measured' to read the outlet "
                "transducer instead."
            )
        return self

    @model_validator(mode="after")
    def _stop_after_steady_needs_detection(self) -> RunConfig:
        if self.stop_after_steady_s is not None and not self.steady_state.enabled:
            raise ValueError(
                "run.stop_after_steady_s is set but run.steady_state.enabled is false, "
                "so steady state would never be detected and the run would never stop"
            )
        return self

    @model_validator(mode="after")
    def _pulse_decay_needs_a_closed_downstream(self) -> RunConfig:
        """A declared P2 and pulse decay contradict each other.

        ``downstream_pressure: <number>`` is the escape hatch for an outlet that
        **vents** -- it asserts the downstream side is held open at a constant
        pressure. Pulse decay measures the differential decaying into a
        **closed** vessel, so under a declared constant P2 the differential
        would decay to a fixed number rather than to zero, and the fitted rate
        would describe the assumption rather than the rock.
        """
        if self.method == "pulse_decay" and not self.downstream_is_measured:
            raise ValueError(
                "run.method is 'pulse_decay' but run.downstream_pressure is the "
                f"supplied value {self.downstream_pressure} "
                f"{self.downstream_pressure_unit}. Pulse decay watches P1 - P2 decay "
                "into a CLOSED downstream vessel; a declared constant P2 asserts the "
                "outlet is open to something. Set downstream_pressure: measured."
            )
        return self

    @model_validator(mode="after")
    def _leak_test_is_a_pulse_decay_step(self) -> RunConfig:
        """A leak test only means something for pulse decay.

        Steady-state Darcy reads a flow, and a blanked plug passes none -- the
        run would simply report no flow, which says nothing about the rig's
        leak rate. The pre-step exists because pulse decay infers permeability
        from a *rate*, and a leak produces a rate indistinguishable from a slow
        sample.
        """
        if self.purpose == "leak_test" and self.method != "pulse_decay":
            raise ValueError(
                f"run.purpose is 'leak_test' but run.method is {self.method!r}. A "
                "leak test measures the differential decay of a blanked rig, which "
                "is a pulse-decay observation; a steady-state run on a blanked plug "
                "would just report no flow. Set method: pulse_decay."
            )
        return self

    @model_validator(mode="after")
    def _pulse_decay_has_no_steady_soak(self) -> RunConfig:
        if self.method == "pulse_decay" and self.stop_after_steady_s is not None:
            raise ValueError(
                "run.stop_after_steady_s has no meaning in pulse decay -- there is no "
                "steady state to soak in. Use pulse_decay.stop_below_fraction to say "
                "how far the decay should run before the measurement ends."
            )
        return self

    @property
    def atmospheric_pressure_atm(self) -> float:
        """Configured atmospheric pressure, atm."""
        return units.to_atm(self.atmospheric_pressure, self.atmospheric_pressure_unit)

    @property
    def fixed_downstream_pressure_atm(self) -> float | None:
        """The supplied P2 in atm, or ``None`` when it is measured."""
        if isinstance(self.downstream_pressure, str):
            return None
        return units.to_atm(self.downstream_pressure, self.downstream_pressure_unit)

    @property
    def downstream_is_measured(self) -> bool:
        """Whether P2 comes from the outlet transducer."""
        return isinstance(self.downstream_pressure, str)

    @property
    def confining_pressure_atm(self) -> float | None:
        """Confining pressure in atm, or ``None`` if unspecified."""
        if self.confining_pressure is None:
            return None
        return units.to_atm(self.confining_pressure, self.confining_pressure_unit)
