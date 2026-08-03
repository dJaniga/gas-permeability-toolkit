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
    """Calibration for one pressure transducer, with its own independent unit."""

    unit: PressureUnit = "kPa"
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
    """The single active flowmeter for this run.

    Only one of ``ai2``/``ai3`` is read; the other analog input is never added
    to the DAQ task. Which one is a config-time decision, never autodetected.

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
    #: Refuse to start ``collect`` if the port cannot be opened.
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
    flowmeter: FlowmeterConfig = Field(default_factory=FlowmeterConfig)
    temperature: TemperatureConfig = Field(default_factory=TemperatureConfig)
    uncertainty: InstrumentUncertaintyConfig = Field(
        default_factory=InstrumentUncertaintyConfig
    )
    #: Free-form provenance: who calibrated the rig and when.
    calibrated_by: str = ""
    calibrated_on: str = ""

    @model_validator(mode="after")
    def _channels_do_not_collide(self) -> HardwareConfig:
        pressure_channels = {
            self.daq.inlet_pressure_channel,
            self.daq.outlet_pressure_channel,
        }
        if self.flowmeter.channel in pressure_channels:
            raise ValueError(
                f"flowmeter.channel {self.flowmeter.channel!r} is already assigned to a "
                "pressure transducer; the flowmeter needs its own analog input "
                "(typically ai2 or ai3)"
            )
        return self

    def daq_channel_path(self, channel: str) -> str:
        """Fully-qualified NI channel name, e.g. ``Dev1/ai0``."""
        return f"{self.daq.device_name}/{channel}"
