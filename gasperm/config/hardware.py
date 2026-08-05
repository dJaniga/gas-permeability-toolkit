"""Rig configuration: DAQ, transducer calibrations, flowmeter, probe.

This is the file that describes the *bench*, not the experiment. It changes
when the rig is rewired or recalibrated -- not when a new core plug is loaded
or a new pressure step is run. Keeping it separate is what makes
``sample.yaml`` and ``run.yaml`` short enough to review before every run.
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from gasperm import units
from gasperm.config.common import (
    LinearCalibration,
    PressureUnit,
    UncertaintySpec,
    _Base,
    validated_pressure_unit,
)

__all__ = [
    "DaqConfig",
    "PressureChannelConfig",
    "PressureCalibrationConfig",
    "FlowmeterConfig",
    "TemperatureConfig",
    "InstrumentUncertaintyConfig",
    "HardwareConfig",
    "FLOWMETER_CHANNEL_HINTS",
]

#: NI USB-6421 analog inputs usable for the flowmeter. A hint for ``init``;
#: other ``aiN`` names are still accepted since the rig may be rewired.
FLOWMETER_CHANNEL_HINTS: tuple[str, ...] = ("ai2", "ai3")


class DaqConfig(_Base):
    """NI-DAQmx device and the two pressure channels."""

    device_name: str = "Dev1"
    inlet_pressure_channel: str = "ai0"
    outlet_pressure_channel: str = "ai1"
    sample_rate_hz: float = Field(default=10.0, gt=0.0, le=1000.0)
    #: NI terminal configuration; RSE is typical for single-ended transducers.
    terminal_config: Literal["DEFAULT", "RSE", "NRSE", "DIFF", "PSEUDO_DIFF"] = "DIFF"

    @field_validator("inlet_pressure_channel", "outlet_pressure_channel")
    @classmethod
    def _bare_channel_name(cls, value: str) -> str:
        name = value.strip()
        if "/" in name:
            # Accept "Dev1/ai0" but store the bare channel; device_name owns
            # the prefix so a device rename is a one-line config change.
            name = name.rsplit("/", 1)[-1]
        if not name.startswith("ai"):
            raise ValueError(f"expected an analog-input channel like 'ai0', got {value!r}")
        return name

    @model_validator(mode="after")
    def _distinct_channels(self) -> DaqConfig:
        if self.inlet_pressure_channel == self.outlet_pressure_channel:
            raise ValueError(
                "inlet and outlet pressure channels must differ "
                f"(both are {self.inlet_pressure_channel!r})"
            )
        return self


class PressureChannelConfig(LinearCalibration):
    """Calibration for one pressure transducer, with its own independent unit.

    The shipped default is the rig's transducer: 0-5 V spanning 0-68.95 MPa,
    i.e. a nominal 10 000 psi unit. (10 000 psi is 68.9476 MPa exactly; the
    rounded figure is the nameplate range, so replace it with the calibration
    certificate's actual span before it matters.)

    Note what that range implies for the uncertainty budget. A
    percent-of-full-scale specification does not shrink with the reading: at
    0.25 % FS the limit is 0.172 MPa, so the standard uncertainty is
    0.172/sqrt(3) = 0.0995 MPa, or 0.98 atm, *whatever pressure is applied*.
    That is fine at MPa-scale pore pressure and hopeless near ambient --
    around 5 MPa inlet it contributes ~4 % to the permeability, and at 0.3 MPa
    it swamps everything else. ``collect`` ranks the budget by contribution, so
    this shows up explicitly rather than hiding in a confident-looking number.
    """

    #: Pressure at ``volts_min`` / ``volts_max``, in :attr:`unit`.
    value_min: float = 0.0
    value_max: float = 68.95
    unit: PressureUnit = "MPa"
    #: Whether the transducer reads absolute or gauge pressure. The Darcy
    #: equation needs **absolute** P1/P2, so gauge readings get the configured
    #: atmospheric pressure added -- explicitly, at the calibration boundary.
    reading_type: Literal["absolute", "gauge"] = "absolute"
    #: Type B uncertainty of this channel, in this channel's unit.
    uncertainty: UncertaintySpec = Field(
        default_factory=lambda: UncertaintySpec(
            kind="percent_full_scale", value=0.25, source="transducer datasheet"
        )
    )

    @field_validator("unit")
    @classmethod
    def _check_unit(cls, value: str) -> str:
        return validated_pressure_unit(value)


class PressureCalibrationConfig(_Base):
    """Inlet and outlet transducer calibrations. Units are independent."""

    inlet: PressureChannelConfig = Field(default_factory=PressureChannelConfig)
    outlet: PressureChannelConfig = Field(default_factory=PressureChannelConfig)
    #: Correlation coefficient between the inlet and outlet pressure errors,
    #: in [-1, 1]. Two channels of the same transducer model, calibrated
    #: against the same reference, share a systematic error; GUM equation (13)
    #: requires the covariance term when they do. Because P1 and P2 enter the
    #: Darcy equation with opposite signs, positive correlation *reduces* the
    #: combined uncertainty -- ignoring it is conservative but wrong.
    correlation: float = Field(default=0.0, ge=-1.0, le=1.0)


class FlowmeterConfig(LinearCalibration):
    """One flowmeter wired to the rig.

    Several may be defined -- a low-range and a high-range meter is the usual
    arrangement -- but exactly **one** is active per run, chosen by name in
    ``run.yaml``. The other meters' analog inputs are never added to the DAQ
    task. Selection stays a config-time decision, never autodetected.

    ``reading_basis`` documents *what thermodynamic state the reported volume
    refers to*, which is the most common source of silent error on these rigs:

    ``standard``
        The meter is mass-based (thermal MFM) and reports volume referenced to
        its own standard state (``standard_temperature_c`` /
        ``standard_pressure``). The Darcy equation then pairs this flow with
        the meter's standard pressure as ``P_ref``.
    ``actual``
        The meter reports volume at the line conditions where it sits. The
        Darcy equation pairs it with the pressure at that location, chosen by
        ``actual_pressure_source``.
    """

    channel: str = "ai2"
    #: Free-text note, e.g. serial number or "0-500 sccm thermal MFM".
    description: str = ""
    volts_min: float = 0.0
    volts_max: float = 10.0
    #: Flow at ``volts_min`` / ``volts_max``, in :attr:`unit`.
    value_min: float = Field(default=0.0, alias="flow_min")
    value_max: float = Field(default=500.0, alias="flow_max")
    unit: str = "sccm"

    reading_basis: Literal["standard", "actual"] = "standard"
    standard_temperature_c: float = 0.0
    standard_pressure: float = 101.325
    standard_pressure_unit: PressureUnit = "kPa"
    #: Only used when ``reading_basis == "actual"``: which pressure the
    #: reported volume is at.
    actual_pressure_source: Literal["outlet", "atmospheric", "inlet"] = "outlet"
    #: Type B uncertainty, in this meter's flow unit.
    uncertainty: UncertaintySpec = Field(
        default_factory=lambda: UncertaintySpec(
            kind="percent_reading", value=1.0, source="mass flowmeter datasheet"
        )
    )

    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, populate_by_name=True
    )

    @field_validator("channel")
    @classmethod
    def _bare_channel(cls, value: str) -> str:
        name = value.strip().rsplit("/", 1)[-1]
        if not name.startswith("ai"):
            raise ValueError(f"expected an analog-input channel like 'ai2', got {value!r}")
        return name

    @field_validator("unit")
    @classmethod
    def _check_flow_unit(cls, value: str) -> str:
        units.flow_to_cm3_s(1.0, value)  # raises ValueError on an unknown unit
        return value

    @field_validator("standard_pressure_unit")
    @classmethod
    def _check_std_unit(cls, value: str) -> str:
        return validated_pressure_unit(value)

    @property
    def flow_min(self) -> float:
        """Flow at ``volts_min``, in :attr:`unit`."""
        return self.value_min

    @property
    def flow_max(self) -> float:
        """Full-scale flow at ``volts_max``, in :attr:`unit`."""
        return self.value_max

    @property
    def standard_pressure_atm(self) -> float:
        """The meter's standard reference pressure, atm."""
        return units.to_atm(self.standard_pressure, self.standard_pressure_unit)

    @property
    def summary(self) -> str:
        """One-line description for the console and the run metadata."""
        detail = f" -- {self.description}" if self.description else ""
        return f"{self.channel}, {self.flow_min:g}-{self.flow_max:g} {self.unit}{detail}"


def _default_flowmeters() -> dict[str, FlowmeterConfig]:
    """The two meters the documented rig has wired, on ai2 and ai3.

    Both are placeholders for full scale: replace them with your meters' actual
    ranges. Defining both here and selecting one per run is what lets a
    Klinkenberg series switch meters between pressure steps without touching
    the rig description.
    """
    return {
        "low_range": FlowmeterConfig(channel="ai2", value_max=500.0),
        "high_range": FlowmeterConfig(channel="ai3", value_max=5000.0),
    }


class TemperatureConfig(_Base):
    """Arduino temperature probe on a USB serial port.

    This link is independent of the DAQ and may fail on its own. ``required``
    controls whether ``collect`` refuses to start without it; once running, a
    dropout never aborts an otherwise-healthy acquisition loop.
    """

    port: str = "COM4"
    baud_rate: int = Field(default=9600, gt=0)
    #: Line format. ``{value}`` marks the number. Set to ``null`` to accept the
    #: first float found anywhere on the line (handles bare ``23.4`` and CSV).
    parse_pattern: str | None = "T:{value}"
    timeout_s: float = Field(default=2.0, gt=0.0)
    units: Literal["C", "K", "F"] = "C"
    #: How long the sensor takes to produce a reading. A DS18B20 needs 750 ms
    #: at 12-bit resolution (187 ms at 9-bit). Acquisition normally samples
    #: faster than this, so each value is *held* for several samples -- which
    #: is correct, not a fault: temperature moves far more slowly than the
    #: pressures. Used to judge whether the probe is keeping up.
    conversion_time_s: float = Field(default=0.75, gt=0.0)
    #: How long ``collect`` waits at startup for the probe's first reading, so
    #: that no sample falls back to a guessed temperature.
    warmup_timeout_s: float = Field(default=5.0, gt=0.0)
    #: Readings outside this band are discarded and the last good value kept.
    #: The default excludes the two DS18B20 sentinels that would otherwise pass
    #: as ordinary numbers: -127 means the sensor did not answer, and 85 is its
    #: power-on reset value. Widen it for a genuinely hot rig.
    plausible_min_c: float = -20.0
    plausible_max_c: float = 60.0
    #: Refuse to start ``collect`` if the port cannot be opened, or if the
    #: probe opens but never speaks within ``warmup_timeout_s``.
    required: bool = True
    #: Used only when the probe never produced a reading and ``required`` is
    #: False -- keeps viscosity lookups possible on a degraded run.
    fallback_temperature_c: float = 20.0
    #: How long a last-known-good reading may be reused before it is flagged
    #: stale in the log.
    stale_after_s: float = Field(default=10.0, gt=0.0)
    #: Type B uncertainty of the probe, in kelvin/degC (the interval is the
    #: same size in both).
    uncertainty: UncertaintySpec = Field(
        default_factory=lambda: UncertaintySpec(
            kind="absolute", value=0.5, source="probe datasheet, degC"
        )
    )

    @model_validator(mode="after")
    def _plausible_band_is_ordered(self) -> TemperatureConfig:
        if self.plausible_min_c >= self.plausible_max_c:
            raise ValueError(
                f"temperature.plausible_min_c ({self.plausible_min_c}) must be below "
                f"plausible_max_c ({self.plausible_max_c})"
            )
        return self


class InstrumentUncertaintyConfig(_Base):
    """Uncertainty terms that belong to the rig rather than to any one channel."""

    #: Relative standard uncertainty of the DAQ's own voltage measurement,
    #: dimensionless. Usually far below the transducer term; kept explicit so
    #: it can be shown to be negligible rather than assumed to be.
    daq_relative: float = Field(default=0.0002, ge=0.0)
    #: Extra relative uncertainty from bench repeatability not otherwise
    #: captured (sleeve seating, valve position, and so on).
    repeatability_relative: float = Field(default=0.0, ge=0.0)
    notes: str = ""


class HardwareConfig(_Base):
    """Everything about the physical rig.

    File: ``hardware.yaml``.
    """

    rig_name: str = "gas-permeameter"
    description: str = ""
    daq: DaqConfig = Field(default_factory=DaqConfig)
    pressure_calibration: PressureCalibrationConfig = Field(
        default_factory=PressureCalibrationConfig
    )
    #: Every flowmeter wired to the rig, by name. One is active per run,
    #: selected in ``run.yaml`` -- which meter suits a given pressure step is
    #: an experiment decision, not a change to the bench.
    flowmeters: dict[str, FlowmeterConfig] = Field(default_factory=_default_flowmeters)
    #: Meter used when ``run.flowmeter`` is unset. May be ``null`` when exactly
    #: one meter is defined.
    default_flowmeter: str | None = "low_range"
    temperature: TemperatureConfig = Field(default_factory=TemperatureConfig)
    uncertainty: InstrumentUncertaintyConfig = Field(
        default_factory=InstrumentUncertaintyConfig
    )
    #: Free-form provenance: who calibrated the rig and when.
    calibrated_by: str = ""
    calibrated_on: str = ""

    @model_validator(mode="after")
    def _channels_do_not_collide(self) -> HardwareConfig:
        if not self.flowmeters:
            raise ValueError(
                "hardware.flowmeters is empty; define at least one meter, e.g.\n"
                "  flowmeters:\n    low_range:\n      channel: ai2\n      flow_max: 500.0"
            )

        pressure_channels = {
            self.daq.inlet_pressure_channel,
            self.daq.outlet_pressure_channel,
        }
        seen: dict[str, str] = {}
        for name, meter in self.flowmeters.items():
            if not name.strip():
                raise ValueError("flowmeter names must not be blank")
            if meter.channel in pressure_channels:
                raise ValueError(
                    f"flowmeters.{name}.channel {meter.channel!r} is already assigned to "
                    "a pressure transducer; each flowmeter needs its own analog input "
                    "(typically ai2 or ai3)"
                )
            if meter.channel in seen:
                raise ValueError(
                    f"flowmeters.{name} and flowmeters.{seen[meter.channel]} are both on "
                    f"{meter.channel!r}; two meters cannot share one analog input"
                )
            seen[meter.channel] = name

        if self.default_flowmeter is not None:
            if self.default_flowmeter not in self.flowmeters:
                raise ValueError(
                    f"default_flowmeter {self.default_flowmeter!r} is not a defined "
                    f"meter. Available: {', '.join(sorted(self.flowmeters))}"
                )
        elif len(self.flowmeters) > 1:
            raise ValueError(
                "default_flowmeter is null but several meters are defined "
                f"({', '.join(sorted(self.flowmeters))}); name the one to use when a run "
                "does not choose"
            )
        return self

    def resolve_flowmeter(self, name: str | None) -> tuple[str, FlowmeterConfig]:
        """The named meter, or the rig default when ``name`` is ``None``.

        Returns:
            ``(name, meter)``.

        Raises:
            ValueError: the name is not a defined meter -- with the available
                names, because a typo here silently changes the calibration
                that every flow reading passes through.
        """
        if name is None:
            if self.default_flowmeter is not None:
                name = self.default_flowmeter
            elif len(self.flowmeters) == 1:
                name = next(iter(self.flowmeters))
            else:  # pragma: no cover - the model validator rules this out
                raise ValueError("no flowmeter selected and no default configured")
        try:
            return name, self.flowmeters[name]
        except KeyError:
            raise ValueError(
                f"run.flowmeter {name!r} is not defined in hardware.yaml. Available "
                f"meters: {', '.join(sorted(self.flowmeters))}"
            ) from None

    def daq_channel_path(self, channel: str) -> str:
        """Fully-qualified NI channel name, e.g. ``Dev1/ai0``."""
        return f"{self.daq.device_name}/{channel}"
