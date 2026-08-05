"""Configuration: three files, three concerns.

``hardware.yaml``
    The bench. DAQ device, transducer calibrations, flowmeter, temperature
    probe, instrument uncertainties. Changes on rewiring or recalibration.
``sample.yaml``
    The rock. Identity, geometry (the only part the physics uses), lithology
    and petrophysical provenance. Changes when a new plug is loaded.
``run.yaml``
    The experiment. Operator, working gas, confining pressure, steady-state
    criteria, output settings. Changes most often -- every pressure step.

They are combined into a single :class:`GaspermConfig` in memory. Validation
policy is unchanged: structural problems (unknown units, degenerate
calibrations, non-positive geometry) are rejected at load; environment
problems (missing serial port, unknown gas) are checked at ``collect``
startup by :func:`validate_for_collect`, because a rig is often configured
before it is fully wired up.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from pydantic import Field, ValidationError, model_validator

from gasperm import units
from gasperm.config._yamldoc import render as _render_yaml
from gasperm.config.common import (
    ConfigError,
    LinearCalibration,
    PressureUnit,
    UncertaintySpec,
    _Base,
    validated_pressure_unit,
)
from gasperm.config.hardware import (
    FLOWMETER_CHANNEL_HINTS,
    DaqConfig,
    FlowmeterConfig,
    HardwareConfig,
    InstrumentUncertaintyConfig,
    PressureCalibrationConfig,
    PressureChannelConfig,
    TemperatureConfig,
)
from gasperm.config.run import (
    GasConfig,
    RunConfig,
    SteadyStateConfig,
    UncertaintyReportConfig,
)
from gasperm.config.sample import SampleConfig
from gasperm.models import ExperimentMetadata, SampleGeometry

__all__ = [
    "ConfigError",
    "ConfigPaths",
    "GaspermConfig",
    "HardwareConfig",
    "SampleConfig",
    "RunConfig",
    "GasConfig",
    "SteadyStateConfig",
    "UncertaintyReportConfig",
    "InstrumentUncertaintyConfig",
    "DaqConfig",
    "PressureChannelConfig",
    "PressureCalibrationConfig",
    "FlowmeterConfig",
    "TemperatureConfig",
    "LinearCalibration",
    "UncertaintySpec",
    "PressureUnit",
    "validated_pressure_unit",
    "FLOWMETER_CHANNEL_HINTS",
    "HARDWARE_FILENAME",
    "SAMPLE_FILENAME",
    "RUN_FILENAME",
    "SECTION_NAMES",
    "load_config",
    "load_run_config",
    "load_sample_config",
    "save_config",
    "config_to_dict",
    "experiment_metadata",
    "validate_for_collect",
    "render_hardware_yaml",
    "render_sample_yaml",
    "render_run_yaml",
]

HARDWARE_FILENAME = "hardware.yaml"
SAMPLE_FILENAME = "sample.yaml"
RUN_FILENAME = "run.yaml"

#: The three sections, in file order.
SECTION_NAMES: tuple[str, ...] = ("hardware", "sample", "run")

#: What to tell an operator when a section's file is missing. The sample file
#: is not written by ``init`` -- it belongs to a core plug, not to the rig --
#: so it points somewhere different from the other two.
_MISSING_FILE_HINT: dict[str, str] = {
    "hardware": "Run 'gasperm init' to create it.",
    "run": "Run 'gasperm init' to create it.",
    "sample": (
        "A sample file describes one core plug, so it is not created by 'init'. "
        "Make one with 'gasperm new-sample <id>' and point at it with --sample."
    ),
}

#: A pre-split single-file config carried the rig sections *and* the sample or
#: experiment sections in one document. The new hardware.yaml legitimately has
#: the rig sections at top level, so the combination is what identifies legacy.
_LEGACY_RIG_KEYS = frozenset({"daq", "pressure_calibration", "flowmeter"})
_LEGACY_OTHER_KEYS = frozenset({"sample", "run", "gas"})


class GaspermConfig(_Base):
    """The three config files, combined."""

    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    sample: SampleConfig = Field(default_factory=SampleConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    #: Folder the config was loaded from. Relative paths inside the config are
    #: resolved against it, so a rig folder can be moved or worked in from any
    #: directory without its runs moving too. Not serialised: it describes
    #: where the files were found, not what they say.
    config_dir: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _selected_flowmeter_exists(self) -> GaspermConfig:
        """Catch a bad meter name at load, not three minutes into a run."""
        self.hardware.resolve_flowmeter(self.run.flowmeter)
        return self

    # -- convenience accessors used across the package --------------------

    def resolved_output_dir(self) -> Path:
        """Where this rig's runs go, as an absolute-or-cwd-relative path.

        ``run.output_dir`` is relative to the config folder, not to whatever
        directory the command happened to be invoked from. Otherwise ``collect``
        and ``klinkenberg`` would disagree about where the runs are the moment
        anyone ran one of them from somewhere else.
        """
        output = Path(self.run.output_dir)
        if output.is_absolute() or self.config_dir is None:
            return output
        return self.config_dir / output

    def geometry(self) -> SampleGeometry:
        """The core plug geometry, with its caliper uncertainties."""
        return self.sample.geometry()

    @property
    def daq(self) -> DaqConfig:
        """Shorthand for ``config.hardware.daq``."""
        return self.hardware.daq

    @property
    def flowmeter(self) -> FlowmeterConfig:
        """The meter active for this run, resolved from ``run.flowmeter``."""
        return self.hardware.resolve_flowmeter(self.run.flowmeter)[1]

    @property
    def flowmeter_name(self) -> str:
        """Name of the active meter, for the console and the run metadata."""
        return self.hardware.resolve_flowmeter(self.run.flowmeter)[0]

    @property
    def pressure_calibration(self) -> PressureCalibrationConfig:
        """Shorthand for ``config.hardware.pressure_calibration``."""
        return self.hardware.pressure_calibration

    @property
    def temperature(self) -> TemperatureConfig:
        """Shorthand for ``config.hardware.temperature``."""
        return self.hardware.temperature

    @property
    def gas(self) -> GasConfig:
        """Shorthand for ``config.run.gas``."""
        return self.run.gas

    def daq_channel_path(self, channel: str) -> str:
        """Fully-qualified NI channel name, e.g. ``Dev1/ai0``."""
        return self.hardware.daq_channel_path(channel)

    @classmethod
    def example(cls) -> GaspermConfig:
        """A fully-populated config matching the shipped rig defaults."""
        return cls()


@dataclass(frozen=True)
class ConfigPaths:
    """Where the three config files live."""

    hardware: Path
    sample: Path
    run: Path

    @classmethod
    def in_directory(cls, directory: str | Path) -> ConfigPaths:
        """The default file names inside ``directory``."""
        base = Path(directory)
        return cls(
            hardware=base / HARDWARE_FILENAME,
            sample=base / SAMPLE_FILENAME,
            run=base / RUN_FILENAME,
        )

    def as_tuple(self) -> tuple[Path, Path, Path]:
        """``(hardware, sample, run)``."""
        return (self.hardware, self.sample, self.run)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _read_mapping(path: Path, section: str) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(
            f"{section.capitalize()} config file not found: {path}. "
            + _MISSING_FILE_HINT[section]
        )
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML:\n  {exc}") from exc
    if raw is None:
        raise ConfigError(f"{path} is empty.")
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{path} must contain a YAML mapping at the top level, got "
            f"{type(raw).__name__}."
        )
    return raw


def _reject_legacy(raw: Mapping[str, Any], path: Path) -> None:
    """Refuse an outdated hardware file, pointing at the replacement."""
    keys = set(raw)
    if "hardware" in keys:
        return

    # Most serious first: a whole pre-split config is a bigger problem than a
    # hardware file that only needs its flowmeter section renamed.
    overlap = _LEGACY_OTHER_KEYS & keys
    if _LEGACY_RIG_KEYS & keys and overlap:
        raise ConfigError(
            f"{path} looks like a pre-split single-file config: it mixes rig sections "
            f"({', '.join(sorted(_LEGACY_RIG_KEYS & keys))}) with "
            f"{', '.join(sorted(overlap))} in one file. Configuration is now three "
            f"files -- {HARDWARE_FILENAME} (the rig), {SAMPLE_FILENAME} (the core plug) "
            f"and {RUN_FILENAME} (the experiment). Run 'gasperm init' to generate them, "
            "then copy your calibration numbers across."
        )

    if "flowmeter" in keys and "flowmeters" not in keys:
        raise ConfigError(
            f"{path} has a single 'flowmeter:' section. A rig can have more than one "
            "meter wired, so they are now named under 'flowmeters:' and each run picks "
            "one by name. Rename the section, e.g.\n"
            "  flowmeters:\n"
            "    low_range:\n"
            "      channel: ai2\n"
            "      ...\n"
            "  default_flowmeter: low_range\n"
            "then set 'flowmeter: <name>' in run.yaml (or leave it null for the default)."
        )


def _unwrap(raw: dict[str, Any], key: str) -> dict[str, Any]:
    """Accept a file either wrapped in its section key or written bare."""
    if set(raw) == {key} and isinstance(raw[key], dict):
        return raw[key]
    return raw


def _format_validation_error(exc: ValidationError, source: str) -> str:
    lines = [f"{source} is not valid ({exc.error_count()} problem(s)):"]
    for err in exc.errors():
        location = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  - {location}: {err['msg']}")
    return "\n".join(lines)


def _validate_section(model, data: Mapping[str, Any], path: Path):
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc, str(path))) from exc


def load_config(
    directory: str | Path | None = None,
    *,
    hardware: str | Path | None = None,
    sample: str | Path | None = None,
    run: str | Path | None = None,
) -> GaspermConfig:
    """Load and validate the three config files.

    Args:
        directory: Where the default file names live. Defaults to the current
            directory. Ignored for any section given an explicit path.
        hardware: Explicit path to the hardware file.
        sample: Explicit path to the sample file.
        run: Explicit path to the run file.

    Returns:
        The combined, validated configuration.

    Raises:
        ConfigError: a file is missing, malformed, fails validation, or is a
            pre-split single-file config.
    """
    defaults = ConfigPaths.in_directory(directory if directory is not None else ".")
    paths = ConfigPaths(
        hardware=Path(hardware) if hardware is not None else defaults.hardware,
        sample=Path(sample) if sample is not None else defaults.sample,
        run=Path(run) if run is not None else defaults.run,
    )

    hardware_raw = _read_mapping(paths.hardware, "hardware")
    _reject_legacy(hardware_raw, paths.hardware)
    sample_raw = _read_mapping(paths.sample, "sample")
    run_raw = _read_mapping(paths.run, "run")

    hardware = _validate_section(
        HardwareConfig, _unwrap(hardware_raw, "hardware"), paths.hardware
    )
    sample = _validate_section(SampleConfig, _unwrap(sample_raw, "sample"), paths.sample)
    run = _validate_section(RunConfig, _unwrap(run_raw, "run"), paths.run)

    try:
        # The run file's folder anchors relative paths: it is the file that
        # names output_dir, so it is the one whose location that path means.
        return GaspermConfig(
            hardware=hardware, sample=sample, run=run, config_dir=paths.run.parent
        )
    except ValidationError as exc:
        # Cross-file checks -- a run naming a meter the rig does not define, for
        # instance. Neither file is wrong on its own, so name both.
        raise ConfigError(
            _format_validation_error(exc, f"{paths.run} together with {paths.hardware}")
        ) from exc


def load_run_config(path: str | Path) -> RunConfig:
    """Load and validate ``run.yaml`` alone.

    ``klinkenberg`` needs only ``output_dir`` to find a plug's runs, and a rig
    folder legitimately has no ``sample.yaml`` -- plugs live in ``samples/`` --
    so the three-file :func:`load_config` would fail on a perfectly good rig.

    Raises:
        ConfigError: the file is missing, malformed, or fails validation.
    """
    run_path = Path(path)
    return _validate_section(
        RunConfig, _unwrap(_read_mapping(run_path, "run"), "run"), run_path
    )


def load_sample_config(path: str | Path) -> SampleConfig:
    """Load and validate one core plug's file on its own.

    Raises:
        ConfigError: the file is missing, malformed, or fails validation.
    """
    sample_path = Path(path)
    return _validate_section(
        SampleConfig, _unwrap(_read_mapping(sample_path, "sample"), "sample"), sample_path
    )


def config_to_dict(config: GaspermConfig) -> dict[str, Any]:
    """Plain-dict form of the config, suitable for YAML/JSON serialisation."""
    return config.model_dump(mode="json", by_alias=True)


def experiment_metadata(config: GaspermConfig) -> ExperimentMetadata:
    """Flatten the metadata every run should be traceable by."""
    sample = config.sample
    run = config.run
    meter_name, meter = config.hardware.resolve_flowmeter(run.flowmeter)
    return ExperimentMetadata(
        flowmeter=meter_name,
        flowmeter_channel=meter.channel,
        flowmeter_range=f"{meter.flow_min:g}-{meter.flow_max:g} {meter.unit}",
        operator=run.operator,
        institution=run.institution,
        project=run.project,
        experiment_id=run.experiment_id,
        notes=run.notes,
        sample_id=sample.id,
        sample_description=sample.description,
        lithology=sample.lithology,
        formation=sample.formation,
        well=sample.well,
        depth=sample.depth,
        depth_unit=sample.depth_unit,
        porosity_fraction=sample.porosity_fraction,
        porosity_method=sample.porosity_method,
        grain_density_g_cm3=sample.grain_density_g_cm3,
        bulk_density_g_cm3=sample.bulk_density_g_cm3,
        prepared_by=sample.prepared_by,
        prepared_on=sample.prepared_on,
        length_cm=sample.length_cm,
        diameter_cm=sample.diameter_cm,
        gas_name=run.gas.name,
        confining_pressure=run.confining_pressure,
        confining_pressure_unit=run.confining_pressure_unit,
    )


# --------------------------------------------------------------------------
# Rendering the commented templates
# --------------------------------------------------------------------------

_SUPPORTED_PRESSURE = ", ".join(sorted(units.SUPPORTED_PRESSURE_UNITS))


def render_hardware_yaml(config: GaspermConfig) -> str:
    """``hardware.yaml`` with the comments that prevent a wrong calibration."""
    data = config.hardware.model_dump(mode="json", by_alias=True)
    header = (
        "gasperm -- HARDWARE configuration (the bench)\n"
        "\n"
        "NI USB-6421 DAQ for pressures and flow, plus an Arduino temperature probe\n"
        "on USB serial. Change this file when the rig is rewired or recalibrated.\n"
        "The core plug lives in sample.yaml; the experiment lives in run.yaml.\n"
        "\n"
        f"Supported pressure units: {_SUPPORTED_PRESSURE}\n"
        "Each pressure-bearing field carries its OWN unit. All internal physics\n"
        "runs in CGS-Darcy (atm, cm, cP, cm3/s) regardless of what you set here."
    )
    notes = {
        "pressure_calibration": (
            "Each channel is added to the DAQ task with its own min_val/max_val, so\n"
            "the 0-5 V transducers and the 0-10 V flowmeter never share a range."
        ),
        "pressure_calibration.inlet.uncertainty": (
            "Type B uncertainty (GUM 4.3). 'a' is the specification limit; the\n"
            "standard uncertainty is a/sqrt(3) for a rectangular distribution."
        ),
        "pressure_calibration.correlation": (
            "Covariance between the two pressure channels. Same transducer model\n"
            "calibrated against the same reference => correlated systematic error.\n"
            "P1 and P2 enter the Darcy equation with opposite signs, so a positive\n"
            "correlation REDUCES the combined uncertainty."
        ),
        "flowmeters": (
            "Every meter wired to the rig, by name. A run selects ONE by name in\n"
            "run.yaml; the others' analog inputs are never added to the DAQ task.\n"
            "Define them once here -- swapping meters between pressure steps is an\n"
            "experiment decision, not a change to the bench."
        ),
        "temperature": (
            "Separate device from the DAQ; a dropout degrades the run, never aborts it."
        ),
        "uncertainty": "Rig-level terms that belong to no single channel.",
    }
    comments = {
        "daq.device_name": "NI-DAQmx device name, from NI MAX",
        "daq.inlet_pressure_channel": "0-5 V transducer",
        "daq.outlet_pressure_channel": "0-5 V transducer",
        "daq.terminal_config": "DEFAULT | RSE | NRSE | DIFF | PSEUDO_DIFF",
        "pressure_calibration.inlet.value_min": "pressure at volts_min",
        "pressure_calibration.inlet.value_max": "pressure at volts_max",
        "pressure_calibration.inlet.reading_type": "absolute | gauge",
        "pressure_calibration.inlet.uncertainty.kind": (
            "percent_full_scale | percent_reading | absolute | none"
        ),
        "pressure_calibration.inlet.uncertainty.distribution": (
            "rectangular | triangular | normal"
        ),
        "pressure_calibration.outlet.unit": "independent of inlet: may be 'bar' here",
        "pressure_calibration.outlet.reading_type": "gauge adds run.atmospheric_pressure",
        "default_flowmeter": "used when run.yaml leaves 'flowmeter' null",
        "temperature.parse_pattern": "'{value}' marks the number; null = first float",
        "temperature.conversion_time_s": "DS18B20: 0.75 s at 12-bit, 0.19 s at 9-bit",
        "temperature.warmup_timeout_s": "startup wait for the probe's first reading",
        "temperature.plausible_min_c": "readings outside the band are discarded;",
        "temperature.plausible_max_c": "excludes the DS18B20 -127 and 85 sentinels",
        "temperature.units": "C | K | F",
        "temperature.required": "refuse to start collect if the port will not open",
        "temperature.uncertainty.value": "degC",
        "uncertainty.daq_relative": "relative, dimensionless",
        "uncertainty.repeatability_relative": "bench repeatability not otherwise captured",
    }

    # Per-meter comments, keyed by whatever the meters are actually called.
    for name in config.hardware.flowmeters:
        comments.update(
            {
                f"flowmeters.{name}.channel": "its own analog input (ai2, ai3, ...)",
                f"flowmeters.{name}.volts_max": "0-10 V, unlike the pressure channels",
                f"flowmeters.{name}.flow_max": "full-scale flow, for THIS meter",
                f"flowmeters.{name}.reading_basis": "standard | actual -- see below",
                f"flowmeters.{name}.actual_pressure_source": (
                    "only used when reading_basis == actual"
                ),
                f"flowmeters.{name}.uncertainty.kind": (
                    "flowmeters are usually spec'd % of reading"
                ),
            }
        )
        notes[f"flowmeters.{name}.reading_basis"] = (
            "What state this meter's reported volume refers to -- the single most\n"
            "common source of a silently wrong permeability on this kind of rig.\n"
            "  standard -> mass-based meter, referenced to the standard state below\n"
            "  actual   -> volume at line conditions, paired with actual_pressure_source"
        )
    return _render_yaml(data, header=header, notes=notes, comments=comments)


def render_sample_yaml(config: GaspermConfig) -> str:
    """``sample.yaml`` -- the core plug."""
    data = config.sample.model_dump(mode="json", by_alias=True)
    header = (
        "gasperm -- SAMPLE configuration (the core plug)\n"
        "\n"
        "Change this file when a new plug is loaded. Confining pressure and working\n"
        "gas are NOT here: the same plug is routinely measured at several of each,\n"
        "so they belong to run.yaml.\n"
        "\n"
        "Only length_cm and diameter_cm enter the Darcy calculation. Everything else\n"
        "is provenance, carried into every run so a number stays traceable to a rock."
    )
    supported_length = ", ".join(sorted(units.SUPPORTED_LENGTH_UNITS))
    unit = config.sample.dimension_unit
    notes = {
        "dimension_unit": (
            "Geometry -- the only part of this file the physics uses. Every dimension\n"
            "below is in dimension_unit; the calculation converts to cm internally.\n"
            "Area goes as diameter^2, so the diameter uncertainty enters the budget\n"
            "doubled -- it is usually the largest single term."
        ),
        "porosity_fraction": "Petrophysics. Informational; not used by the Darcy calc.",
        "prepared_by": "Provenance.",
    }
    comments = {
        "depth_unit": "m | ft",
        "dimension_unit": supported_length,
        "length": unit,
        "diameter": f"{unit} (38.1 mm = 1.5 in)",
        "length_uncertainty": f"standard uncertainty, {unit} (caliper)",
        "diameter_uncertainty": f"standard uncertainty, {unit} -- counts double",
        "porosity_method": "helium pycnometry, MICP, image analysis, ...",
        "prepared_on": "YYYY-MM-DD",
    }
    return _render_yaml(data, header=header, notes=notes, comments=comments)


def render_run_yaml(config: GaspermConfig) -> str:
    """``run.yaml`` -- the experiment."""
    data = config.run.model_dump(mode="json", by_alias=True)
    header = (
        "gasperm -- RUN configuration (the experiment)\n"
        "\n"
        "The file that changes most often: a new pressure step, a different gas, a\n"
        "different operator. The rig is in hardware.yaml; the plug is in sample.yaml.\n"
        "\n"
        f"Supported pressure units: {_SUPPORTED_PRESSURE}"
    )
    available = ", ".join(sorted(config.hardware.flowmeters))
    notes = {
        "operator": "Who ran it. Copied into every run's metadata for traceability.",
        "gas": "Working gas and where its properties come from.",
        "flowmeter": (
            "Which meter from hardware.yaml this run uses. Defined meters:\n"
            f"  {available}\n"
            "null takes hardware.default_flowmeter. Only this meter's analog input\n"
            "is read; the others are never added to the DAQ task."
        ),
        "confining_pressure": (
            "Confining/overburden pressure for this run. Usually MPa-scale while\n"
            "pore pressure is kPa-scale, hence its own independent unit."
        ),
        "downstream_pressure": (
            "P2 in the Darcy equation. 'measured' reads the outlet transducer, which\n"
            "is what a normally-plumbed rig wants. A number overrides it with a value\n"
            "you supply -- for an outlet that vents to atmosphere, where the\n"
            "transducer reads noise around zero or is not fitted.\n"
            "P2 sets the apparent permeability AND the mean pressure, which is the\n"
            "Klinkenberg regression's x-axis, so 'klinkenberg' refuses to mix runs\n"
            "that used different conventions."
        ),
        "atmospheric_pressure": (
            "Ambient reference -- NOT P2 (see downstream_pressure above). It converts\n"
            "GAUGE transducer readings to the absolute pressures the Darcy equation\n"
            "needs, and serves as the flowmeter reference when actual_pressure_source\n"
            "is 'atmospheric'."
        ),
        "steady_state": (
            "Permeability is only representative once the rig has equilibrated, so\n"
            "the reported result is taken from the detected steady-state window.\n"
            "Two criteria must hold on every listed signal, over consecutive windows:\n"
            "  scatter -- coefficient of variation <= relative_stddev_tolerance\n"
            "  drift   -- fractional change of an OLS line across the window\n"
            "             <= relative_drift_tolerance\n"
            "The drift test is the important one: a slow ramp has small scatter in\n"
            "any short window and would pass a scatter-only test forever."
        ),
        "uncertainty": (
            "ISO/IEC Guide 98-3 (GUM) evaluation. Type A comes from the scatter of\n"
            "the steady-state window; Type B from the specs in hardware.yaml and\n"
            "the caliper figures in sample.yaml."
        ),
        "output_dir": "Output.",
        "duration_s": "Stop conditions.",
    }
    comments = {
        "flowmeter": f"one of: {available} (null = rig default)",
        "downstream_pressure": "measured | a number in downstream_pressure_unit",
        "downstream_pressure_unit": "only used when the above is a number",
        "gas.name": "any CoolProp fluid: Air, CarbonDioxide, Methane, ...",
        "gas.properties_source": "coolprop (recommended) | fixed",
        "gas.fixed_viscosity_cp": "only when properties_source == fixed",
        "gas.viscosity_relative_uncertainty": "relative, for the GUM budget",
        "gas.real_gas_correction": "divide reference flow by Z",
        "steady_state.window_s": "trailing window each test runs over",
        "steady_state.required_windows": "consecutive passes before declaring steady",
        "steady_state.settling_time_s": "ignore this much of the run start outright",
        "steady_state.slope_significance": "null = use the drift bound alone",
        "steady_state.max_wait_s": "null = wait indefinitely",
        "uncertainty.coverage_probability": "0.95 -> k ~ 2 for large dof",
        "uncertainty.fixed_coverage_factor": "null = derive from Student-t",
        "averaging_window_s": "live display only; the result uses the steady window",
        "display_pressure_unit": "console/plot only",
        "display_permeability_unit": "mD | D | uD | um2 | m2",
        "duration_s": "null = run until Ctrl+C",
        "stop_when_steady": "end the run once steady state is confirmed",
    }
    return _render_yaml(data, header=header, notes=notes, comments=comments)


def save_config(
    config: GaspermConfig,
    directory: str | Path = ".",
    *,
    overwrite: bool = False,
    sections: Sequence[str] = SECTION_NAMES,
) -> ConfigPaths:
    """Write config files into ``directory``.

    Args:
        config: The configuration to render.
        directory: Where the files go.
        overwrite: Replace files that already exist.
        sections: Which of ``hardware``/``sample``/``run`` to write. ``init``
            writes only the rig and the experiment: a sample file describes one
            core plug and is created per plug by ``new-sample``.

    Returns:
        The locations of all three files, whether or not each was written.

    Raises:
        ConfigError: a target being written exists and ``overwrite`` is False,
            or an unknown section name was given.
    """
    unknown = set(sections) - set(SECTION_NAMES)
    if unknown:
        raise ConfigError(
            f"Unknown config section(s): {', '.join(sorted(unknown))}. "
            f"Valid sections: {', '.join(SECTION_NAMES)}"
        )

    paths = ConfigPaths.in_directory(directory)
    renderers = {
        "hardware": (paths.hardware, render_hardware_yaml),
        "sample": (paths.sample, render_sample_yaml),
        "run": (paths.run, render_run_yaml),
    }
    selected = [renderers[name] for name in SECTION_NAMES if name in sections]

    existing = [path for path, _ in selected if path.exists()]
    if existing and not overwrite:
        raise ConfigError(
            "Refusing to overwrite: "
            + ", ".join(str(p) for p in existing)
            + ". Pass --force to replace them."
        )

    paths.hardware.parent.mkdir(parents=True, exist_ok=True)
    for path, render in selected:
        path.write_text(render(config), encoding="utf-8")
    return paths


# --------------------------------------------------------------------------
# Startup validation
# --------------------------------------------------------------------------


def validate_for_collect(config: GaspermConfig) -> list[str]:
    """Environment checks run at ``collect`` startup, before opening the DAQ.

    Fails loudly and specifically here rather than three minutes into a run.

    Returns:
        Non-fatal warnings. Fatal problems raise :class:`ConfigError`.
    """
    problems: list[str] = []
    warnings: list[str] = []

    # -- gas name must be one CoolProp recognises ------------------------
    if config.run.gas.properties_source == "coolprop":
        from gasperm.gas_properties import CoolPropUnavailable, validate_gas_name

        try:
            validate_gas_name(config.run.gas.name)
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

    port_state = serial_port_exists(config.hardware.temperature.port)
    if port_state is False:
        message = (
            f"temperature.port {config.hardware.temperature.port!r} was not found among "
            "the ports this machine reports."
        )
        if config.hardware.temperature.required:
            problems.append(
                message + " Fix the port, or set temperature.required: false to run "
                "without the probe."
            )
        else:
            warnings.append(
                message + " temperature.required is false, so the run will proceed using "
                f"fallback_temperature_c = "
                f"{config.hardware.temperature.fallback_temperature_c}."
            )

    # -- temperature probe cadence ----------------------------------------
    probe = config.hardware.temperature
    sample_interval_s = 1.0 / config.hardware.daq.sample_rate_hz
    if probe.conversion_time_s > sample_interval_s:
        held = probe.conversion_time_s / sample_interval_s
        warnings.append(
            f"The temperature probe converts every {probe.conversion_time_s:g} s while "
            f"the DAQ samples every {sample_interval_s:.3g} s, so each temperature is "
            f"held for about {held:.0f} samples. That is expected -- temperature moves "
            "far more slowly than the pressures -- and the run summary reports if the "
            "probe falls further behind than that."
        )
    if probe.stale_after_s <= probe.conversion_time_s:
        warnings.append(
            f"temperature.stale_after_s ({probe.stale_after_s:g} s) is not longer than "
            f"conversion_time_s ({probe.conversion_time_s:g} s), so every reading will "
            "be flagged stale as a matter of course. Raise it above the conversion time."
        )

    # -- geometry sanity --------------------------------------------------
    sample = config.sample
    if sample.length_cm > 100.0:
        warnings.append(
            f"sample.length = {sample.length} {sample.dimension_unit} "
            f"({sample.length_cm:.1f} cm) is unusually long for a core plug; check "
            "sample.dimension_unit."
        )
    if sample.diameter_cm > 30.0:
        warnings.append(
            f"sample.diameter = {sample.diameter} {sample.dimension_unit} "
            f"({sample.diameter_cm:.1f} cm) is unusually wide for a core plug; check "
            "sample.dimension_unit."
        )

    # -- pressure plausibility -------------------------------------------
    inlet = config.hardware.pressure_calibration.inlet
    inlet_full_scale_atm = units.to_atm(inlet.value_max, inlet.unit)
    if inlet_full_scale_atm < config.run.atmospheric_pressure_atm:
        warnings.append(
            "The inlet transducer's full-scale reading is below atmospheric pressure; "
            "check pressure_calibration.inlet.unit."
        )

    supplied_p2 = config.run.fixed_downstream_pressure_atm
    if supplied_p2 is not None and supplied_p2 >= inlet_full_scale_atm:
        warnings.append(
            f"run.downstream_pressure ({config.run.downstream_pressure:g} "
            f"{config.run.downstream_pressure_unit}) is at or above the inlet "
            "transducer's full scale, so no reading could ever show a positive "
            "differential. Check the value and its unit."
        )

    # The active meter, for the checks below. Which one is in use is reported
    # prominently by ``collect`` itself, so it needs no warning here.
    _, meter = config.hardware.resolve_flowmeter(config.run.flowmeter)

    # -- analysis settings ------------------------------------------------
    if not config.run.steady_state.enabled:
        warnings.append(
            "run.steady_state.enabled is false. Permeability will be reported from the "
            "trailing averaging window without any check that the rig had equilibrated, "
            "which is not a representative measurement of the sample."
        )
    elif config.run.duration_s is not None:
        needed = config.run.steady_state.settling_time_s + config.run.steady_state.window_s * (
            config.run.steady_state.required_windows
        )
        if config.run.duration_s < needed:
            warnings.append(
                f"run.duration_s ({config.run.duration_s} s) is shorter than the "
                f"{needed:g} s the steady-state criteria need at minimum, so the run "
                "will almost certainly end before steady state can be confirmed."
            )

    if config.run.uncertainty.enabled:
        specs = [
            config.hardware.pressure_calibration.inlet.uncertainty,
            config.hardware.pressure_calibration.outlet.uncertainty,
            meter.uncertainty,
        ]
        if all(spec.kind == "none" or spec.value == 0.0 for spec in specs):
            warnings.append(
                "Every instrument uncertainty in hardware.yaml is zero, so the GUM "
                "budget would report only the Type A scatter and understate the real "
                "uncertainty. Fill in the transducer and flowmeter specifications."
            )

    if problems:
        raise ConfigError(
            "Configuration cannot be used for a collect run:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )
    return warnings
