"""Pydantic config schema plus YAML load/save.

Validation policy (see CLAUDE.md):

* Structural problems -- unknown units, ``volts_min == volts_max``,
  non-positive geometry -- are rejected at **load** time.
* Environment problems -- serial port missing, DAQ device absent, CoolProp not
  recognising the gas -- are checked at ``collect`` **startup** by
  :func:`validate_for_collect`, not at ``init``. A rig is often configured
  before it is fully wired up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from gasperm import units
from gasperm.models import SampleGeometry

DEFAULT_CONFIG_FILENAME = "gasperm_config.yaml"

#: NI USB-6421 analog inputs usable for the flowmeter (see CLAUDE.md wiring
#: table). Kept as a hint for ``init``; other ``aiN`` names are still accepted
#: since the rig may be rewired.
FLOWMETER_CHANNEL_HINTS: tuple[str, ...] = ("ai2", "ai3")


class ConfigError(Exception):
    """Raised for unusable configuration, with an operator-readable message."""


def _validated_pressure_unit(value: str) -> str:
    return units.normalize_pressure_unit(value)


PressureUnit = Annotated[str, Field(description="One of units.SUPPORTED_PRESSURE_UNITS")]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------
# DAQ
# --------------------------------------------------------------------------


class DaqConfig(_Base):
    """NI-DAQmx device and the two pressure channels."""

    device_name: str = "Dev1"
    inlet_pressure_channel: str = "ai0"
    outlet_pressure_channel: str = "ai1"
    sample_rate_hz: float = Field(default=10.0, gt=0.0, le=1000.0)
    #: NI terminal configuration; RSE is typical for single-ended transducers.
    terminal_config: Literal["DEFAULT", "RSE", "NRSE", "DIFF", "PSEUDO_DIFF"] = "DEFAULT"

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


# --------------------------------------------------------------------------
# Pressure calibration
# --------------------------------------------------------------------------


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


class PressureChannelConfig(LinearCalibration):
    """Calibration for one pressure transducer, with its own independent unit."""

    unit: PressureUnit = "kPa"
    #: Whether the transducer reads absolute or gauge pressure. The Darcy
    #: equation needs **absolute** P1/P2, so gauge readings get the configured
    #: atmospheric pressure added -- explicitly, at the calibration boundary.
    reading_type: Literal["absolute", "gauge"] = "absolute"

    @field_validator("unit")
    @classmethod
    def _check_unit(cls, value: str) -> str:
        return _validated_pressure_unit(value)


class PressureCalibrationConfig(_Base):
    """Inlet and outlet transducer calibrations. Units are independent."""

    inlet: PressureChannelConfig = Field(default_factory=PressureChannelConfig)
    outlet: PressureChannelConfig = Field(default_factory=PressureChannelConfig)


# --------------------------------------------------------------------------
# Flowmeter
# --------------------------------------------------------------------------


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
        the meter's standard pressure as ``P_atm``.
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
        return _validated_pressure_unit(value)

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


# --------------------------------------------------------------------------
# Temperature (Arduino over serial)
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Gas
# --------------------------------------------------------------------------


class GasConfig(_Base):
    """Working gas and where its thermophysical properties come from."""

    #: Any CoolProp fluid name: "Nitrogen", "Air", "CarbonDioxide", "Methane"...
    name: str = "Nitrogen"
    properties_source: Literal["coolprop", "fixed"] = "coolprop"
    #: Only consulted when ``properties_source == "fixed"``. Setting this
    #: bypasses the live (T, P) lookup; record why in ``fixed_reason``.
    fixed_viscosity_cp: float | None = Field(default=None, gt=0.0)
    fixed_reason: str = ""

    @model_validator(mode="after")
    def _fixed_needs_value(self) -> GasConfig:
        if self.properties_source == "fixed" and self.fixed_viscosity_cp is None:
            raise ValueError(
                "gas.properties_source is 'fixed' but gas.fixed_viscosity_cp is not "
                "set; either provide a viscosity in cP or switch back to 'coolprop'"
            )
        return self


# --------------------------------------------------------------------------
# Sample
# --------------------------------------------------------------------------


class SampleConfig(_Base):
    """Core plug geometry and the confining pressure it is held at."""

    id: str = "core-001"
    description: str = ""
    length_cm: float = Field(default=5.0, gt=0.0)
    diameter_cm: float = Field(default=2.54, gt=0.0)
    #: Informational only -- not used by the Darcy calculation.
    porosity_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    #: Confining/overburden pressure target. Typically MPa-scale while pore
    #: pressure is kPa-scale, hence its own independent unit.
    confining_pressure: float | None = None
    confining_pressure_unit: PressureUnit = "MPa"

    @field_validator("confining_pressure_unit")
    @classmethod
    def _check_unit(cls, value: str) -> str:
        return _validated_pressure_unit(value)

    def geometry(self) -> SampleGeometry:
        """Convert to the hardware-free geometry model used by the physics."""
        return SampleGeometry(
            sample_id=self.id,
            description=self.description,
            length_cm=self.length_cm,
            diameter_cm=self.diameter_cm,
            porosity_fraction=self.porosity_fraction,
        )

    @property
    def confining_pressure_atm(self) -> float | None:
        """Confining pressure in atm, or ``None`` if unspecified."""
        if self.confining_pressure is None:
            return None
        return units.to_atm(self.confining_pressure, self.confining_pressure_unit)


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


class RunConfig(_Base):
    """Everything about how a ``collect`` run executes and reports."""

    #: Which pressure to use as P2 in the Darcy equation:
    #: ``"atmospheric"`` -- the configured atmospheric reference;
    #: ``"measured"``    -- the outlet transducer reading;
    #: a number         -- a fixed value in ``outlet_pressure_reference_unit``.
    outlet_pressure_reference: float | Literal["atmospheric", "measured"] = "atmospheric"
    outlet_pressure_reference_unit: PressureUnit = "kPa"

    atmospheric_pressure: float = Field(default=101.325, gt=0.0)
    atmospheric_pressure_unit: PressureUnit = "kPa"

    #: Rolling window for the "live" permeability display and for the
    #: steady-state summary. Also reused by ``klinkenberg``.
    averaging_window_s: float = Field(default=5.0, gt=0.0)
    output_dir: str = "./runs"

    #: Display-only units. Independent of calibration units and of the
    #: internal CGS calculation.
    display_pressure_unit: PressureUnit = "kPa"
    display_permeability_unit: str = "mD"
    display_flow_unit: str = "sccm"

    #: Stop conditions. ``null`` on both means run until Ctrl+C.
    duration_s: float | None = Field(default=None, gt=0.0)
    max_samples: int | None = Field(default=None, gt=0)
    #: Flush the CSV every N samples so a crash cannot lose the whole run.
    flush_every_n: int = Field(default=20, gt=0)

    @field_validator(
        "atmospheric_pressure_unit",
        "display_pressure_unit",
        "outlet_pressure_reference_unit",
    )
    @classmethod
    def _check_pressure_unit(cls, value: str) -> str:
        return _validated_pressure_unit(value)

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

    @property
    def atmospheric_pressure_atm(self) -> float:
        """Configured atmospheric pressure, atm."""
        return units.to_atm(self.atmospheric_pressure, self.atmospheric_pressure_unit)

    @property
    def fixed_outlet_pressure_atm(self) -> float | None:
        """P2 when ``outlet_pressure_reference`` is a fixed number, else None."""
        if isinstance(self.outlet_pressure_reference, str):
            return None
        return units.to_atm(
            self.outlet_pressure_reference, self.outlet_pressure_reference_unit
        )


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------


class GaspermConfig(_Base):
    """Complete rig configuration -- the object every command works from."""

    daq: DaqConfig = Field(default_factory=DaqConfig)
    pressure_calibration: PressureCalibrationConfig = Field(
        default_factory=PressureCalibrationConfig
    )
    flowmeter: FlowmeterConfig = Field(default_factory=FlowmeterConfig)
    temperature: TemperatureConfig = Field(default_factory=TemperatureConfig)
    gas: GasConfig = Field(default_factory=GasConfig)
    sample: SampleConfig = Field(default_factory=SampleConfig)
    run: RunConfig = Field(default_factory=RunConfig)

    @model_validator(mode="after")
    def _channels_do_not_collide(self) -> GaspermConfig:
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

    # -- convenience accessors used by acquisition / storage / plotting ----

    def geometry(self) -> SampleGeometry:
        """The core plug geometry."""
        return self.sample.geometry()

    def daq_channel_path(self, channel: str) -> str:
        """Fully-qualified NI channel name, e.g. ``Dev1/ai0``."""
        return f"{self.daq.device_name}/{channel}"

    @classmethod
    def example(cls) -> GaspermConfig:
        """A fully-populated config matching the shipped rig defaults."""
        return cls()


# --------------------------------------------------------------------------
# YAML I/O
# --------------------------------------------------------------------------


def _format_validation_error(exc: ValidationError, source: str) -> str:
    lines = [f"{source} is not a valid gasperm config ({exc.error_count()} problem(s)):"]
    for err in exc.errors():
        location = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  - {location}: {err['msg']}")
    return "\n".join(lines)


def load_config(path: str | Path) -> GaspermConfig:
    """Load and validate a YAML config.

    Raises:
        ConfigError: file missing, malformed YAML, or failing validation --
            always with a message naming the offending field.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(
            f"Config file not found: {config_path}. Run 'gasperm init' to create one."
        )
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{config_path} is not valid YAML:\n  {exc}") from exc

    if raw is None:
        raise ConfigError(f"{config_path} is empty.")
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{config_path} must contain a YAML mapping at the top level, "
            f"got {type(raw).__name__}."
        )
    try:
        return GaspermConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc, str(config_path))) from exc


def config_to_dict(config: GaspermConfig) -> dict[str, Any]:
    """Plain-dict form of the config, suitable for YAML/JSON serialisation."""
    return config.model_dump(mode="json", by_alias=True)


def save_config(config: GaspermConfig, path: str | Path, *, overwrite: bool = False) -> Path:
    """Write ``config`` to ``path`` as commented YAML.

    Raises:
        ConfigError: if the file exists and ``overwrite`` is False.
    """
    target = Path(path)
    if target.exists() and not overwrite:
        raise ConfigError(f"{target} already exists. Pass --force to overwrite.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_config_yaml(config), encoding="utf-8")
    return target


def render_config_yaml(config: GaspermConfig) -> str:
    """Render the config as YAML with the explanatory comments operators need.

    Written by hand rather than dumped, because the comments (which unit each
    field is in, which channel does what) are the part that prevents a silently
    wrong permeability.
    """
    data = config_to_dict(config)
    daq = data["daq"]
    inlet = data["pressure_calibration"]["inlet"]
    outlet = data["pressure_calibration"]["outlet"]
    flow = data["flowmeter"]
    temp = data["temperature"]
    gas = data["gas"]
    sample = data["sample"]
    run = data["run"]

    def y(value: Any) -> str:
        """YAML scalar form of ``value`` (``None`` -> ``null``)."""
        return yaml.safe_dump(value, default_flow_style=True).strip().rstrip("...").strip()

    supported = ", ".join(sorted(units.SUPPORTED_PRESSURE_UNITS))

    return f"""\
# gasperm configuration
#
# Rig: NI USB-6421 DAQ (pressures + flow) plus an Arduino temperature probe on
# USB serial. Every pressure-bearing field below carries its OWN unit; nothing
# forces a single global pressure unit on the config.
#
# Supported pressure units: {supported}
# All internal physics runs in CGS-Darcy (atm, cm, cP, cm3/s) regardless of
# what you set here.

daq:
  device_name: {y(daq["device_name"])}          # NI-DAQmx device name, from NI MAX
  inlet_pressure_channel: {y(daq["inlet_pressure_channel"])}      # 0-5 V transducer
  outlet_pressure_channel: {y(daq["outlet_pressure_channel"])}     # 0-5 V transducer
  sample_rate_hz: {y(daq["sample_rate_hz"])}
  terminal_config: {y(daq["terminal_config"])}     # DEFAULT | RSE | NRSE | DIFF | PSEUDO_DIFF

pressure_calibration:
  # Each channel is added to the DAQ task with its own min_val/max_val, so the
  # 0-5 V transducers and the 0-10 V flowmeter never share a range.
  inlet:
    volts_min: {y(inlet["volts_min"])}
    volts_max: {y(inlet["volts_max"])}
    value_min: {y(inlet["value_min"])}            # physical pressure at volts_min
    value_max: {y(inlet["value_max"])}         # physical pressure at volts_max
    unit: {y(inlet["unit"])}
    reading_type: {y(inlet["reading_type"])}    # absolute | gauge (gauge gets run.atmospheric_pressure added)
  outlet:
    volts_min: {y(outlet["volts_min"])}
    volts_max: {y(outlet["volts_max"])}
    value_min: {y(outlet["value_min"])}
    value_max: {y(outlet["value_max"])}
    unit: {y(outlet["unit"])}              # independent of inlet: this may be 'bar' while inlet is 'kPa'
    reading_type: {y(outlet["reading_type"])}

flowmeter:
  # Exactly ONE flowmeter is active per run. The other analog input is never
  # added to the DAQ task.
  channel: {y(flow["channel"])}               # ai2 or ai3
  volts_min: {y(flow["volts_min"])}
  volts_max: {y(flow["volts_max"])}             # flow channel is 0-10 V, unlike the pressure channels
  flow_min: {y(flow["flow_min"])}
  flow_max: {y(flow["flow_max"])}            # full-scale flow at volts_max, for THIS meter's range
  unit: {y(flow["unit"])}
  # What state the reported volume refers to. This is the single most common
  # source of a silently wrong permeability on this kind of rig.
  #   standard -> mass-based meter, volume referenced to the standard state below
  #   actual   -> volume at line conditions, paired with actual_pressure_source
  reading_basis: {y(flow["reading_basis"])}
  standard_temperature_c: {y(flow["standard_temperature_c"])}
  standard_pressure: {y(flow["standard_pressure"])}
  standard_pressure_unit: {y(flow["standard_pressure_unit"])}
  actual_pressure_source: {y(flow["actual_pressure_source"])}   # only used when reading_basis == actual

temperature:
  # Separate device from the DAQ; a dropout here degrades the run, never aborts it.
  port: {y(temp["port"])}
  baud_rate: {y(temp["baud_rate"])}
  parse_pattern: {y(temp["parse_pattern"])}     # '{{value}}' marks the number; null = first float on the line
  timeout_s: {y(temp["timeout_s"])}
  units: {y(temp["units"])}                  # C | K | F
  required: {y(temp["required"])}              # refuse to start collect if the port will not open
  fallback_temperature_c: {y(temp["fallback_temperature_c"])}
  stale_after_s: {y(temp["stale_after_s"])}

gas:
  name: {y(gas["name"])}           # any CoolProp fluid: Air, CarbonDioxide, Methane, ...
  properties_source: {y(gas["properties_source"])}   # coolprop (recommended) | fixed
  fixed_viscosity_cp: {y(gas["fixed_viscosity_cp"])}      # only when properties_source == fixed
  fixed_reason: {y(gas["fixed_reason"])}            # why the live (T,P) lookup is being bypassed

sample:
  id: {y(sample["id"])}
  description: {y(sample["description"])}
  length_cm: {y(sample["length_cm"])}
  diameter_cm: {y(sample["diameter_cm"])}
  porosity_fraction: {y(sample["porosity_fraction"])}     # informational; not used by the Darcy calc
  confining_pressure: {y(sample["confining_pressure"])}
  confining_pressure_unit: {y(sample["confining_pressure_unit"])}   # usually MPa-scale while pore pressure is kPa-scale

run:
  outlet_pressure_reference: {y(run["outlet_pressure_reference"])}   # atmospheric | measured | a number
  outlet_pressure_reference_unit: {y(run["outlet_pressure_reference_unit"])}      # used only if the above is a number
  atmospheric_pressure: {y(run["atmospheric_pressure"])}
  atmospheric_pressure_unit: {y(run["atmospheric_pressure_unit"])}
  averaging_window_s: {y(run["averaging_window_s"])}              # rolling window for live k and the run summary
  output_dir: {y(run["output_dir"])}
  display_pressure_unit: {y(run["display_pressure_unit"])}       # console/plot only
  display_permeability_unit: {y(run["display_permeability_unit"])}      # mD | D | uD | um2 | m2
  display_flow_unit: {y(run["display_flow_unit"])}
  duration_s: {y(run["duration_s"])}                # null = run until Ctrl+C
  max_samples: {y(run["max_samples"])}
  flush_every_n: {y(run["flush_every_n"])}
"""


def validate_for_collect(config: GaspermConfig) -> list[str]:
    """Environment checks run at ``collect`` startup, before opening the DAQ.

    Fails loudly and specifically here rather than three minutes into a run.

    Returns:
        Non-fatal warnings. Fatal problems raise :class:`ConfigError`.
    """
    problems: list[str] = []
    warnings: list[str] = []

    # -- gas name must be one CoolProp recognises ------------------------
    if config.gas.properties_source == "coolprop":
        from gasperm.gas_properties import CoolPropUnavailable, validate_gas_name

        try:
            validate_gas_name(config.gas.name)
        except CoolPropUnavailable as exc:
            problems.append(
                f"gas.properties_source is 'coolprop' but CoolProp is unusable: {exc}. "
                "Install CoolProp, or set gas.properties_source: fixed with a "
                "fixed_viscosity_cp."
            )
        except ValueError as exc:
            problems.append(str(exc))

    # -- serial port ------------------------------------------------------
    from gasperm.hardware.temperature import serial_port_exists

    port_state = serial_port_exists(config.temperature.port)
    if port_state is False:
        message = (
            f"temperature.port {config.temperature.port!r} was not found among the "
            "ports this machine reports."
        )
        if config.temperature.required:
            problems.append(
                message + " Fix the port, or set temperature.required: false to run "
                "without the probe."
            )
        else:
            warnings.append(
                message + " temperature.required is false, so the run will proceed "
                f"using fallback_temperature_c = {config.temperature.fallback_temperature_c}."
            )

    # -- geometry sanity --------------------------------------------------
    if config.sample.length_cm > 100.0:
        warnings.append(
            f"sample.length_cm = {config.sample.length_cm} is unusually long for a core "
            "plug; check the unit."
        )
    if config.sample.diameter_cm > 30.0:
        warnings.append(
            f"sample.diameter_cm = {config.sample.diameter_cm} is unusually wide for a "
            "core plug; check the unit."
        )

    # -- pressure plausibility -------------------------------------------
    inlet_max_atm = units.to_atm(
        config.pressure_calibration.inlet.value_max, config.pressure_calibration.inlet.unit
    )
    if inlet_max_atm < config.run.atmospheric_pressure_atm:
        warnings.append(
            "The inlet transducer's full-scale reading is below atmospheric pressure; "
            "check pressure_calibration.inlet.unit."
        )

    if problems:
        raise ConfigError(
            "Configuration cannot be used for a collect run:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )
    return warnings
