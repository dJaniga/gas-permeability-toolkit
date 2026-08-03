"""The experiment: who is running it, on what gas, at what confining pressure.

File: ``run.yaml``. This is the file that changes most often -- a new pressure
step, a different working gas, a different operator -- which is exactly why it
is separate from the rig description and the sample description.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from gasperm import units
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
    "RunConfig",
]


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
    gas: GasConfig = Field(default_factory=GasConfig)
    #: Confining/overburden pressure the plug is held at. A run-level quantity:
    #: the same plug is routinely measured at several confining pressures.
    #: Typically MPa-scale while pore pressure is kPa-scale, hence its own unit.
    confining_pressure: float | None = None
    confining_pressure_unit: PressureUnit = "MPa"

    # -- ambient reference ------------------------------------------------
    #: Local atmospheric pressure. P1 and P2 both come from their transducers
    #: on the DAQ, so this is *not* a stand-in for either: it is used only to
    #: convert **gauge** transducer readings to the absolute pressures the
    #: Darcy equation needs, and as the flowmeter's reference pressure when
    #: ``flowmeter.actual_pressure_source`` is ``atmospheric``. With absolute
    #: transducers on both ports and a standard-basis meter it is unused.
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

    # -- stop conditions --------------------------------------------------
    #: ``null`` on both means run until Ctrl+C (or until steady state, if
    #: ``stop_when_steady`` is set).
    duration_s: float | None = Field(default=None, gt=0.0)
    max_samples: int | None = Field(default=None, gt=0)
    #: End the run once steady state has been confirmed and held for
    #: ``steady_state.window_s``. Useful unattended; off by default so an
    #: operator can watch the plateau hold.
    stop_when_steady: bool = False
    #: Flush the CSV every N samples so a crash cannot lose the whole run.
    flush_every_n: int = Field(default=20, gt=0)

    @field_validator(
        "atmospheric_pressure_unit",
        "display_pressure_unit",
        "confining_pressure_unit",
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
    def _stop_when_steady_needs_detection(self) -> RunConfig:
        if self.stop_when_steady and not self.steady_state.enabled:
            raise ValueError(
                "run.stop_when_steady is true but run.steady_state.enabled is false, "
                "so steady state would never be detected and the run would not stop"
            )
        return self

    @property
    def atmospheric_pressure_atm(self) -> float:
        """Configured atmospheric pressure, atm."""
        return units.to_atm(self.atmospheric_pressure, self.atmospheric_pressure_unit)

    @property
    def confining_pressure_atm(self) -> float | None:
        """Confining pressure in atm, or ``None`` if unspecified."""
        if self.confining_pressure is None:
            return None
        return units.to_atm(self.confining_pressure, self.confining_pressure_unit)
