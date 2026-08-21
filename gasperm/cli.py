"""Command-line entry point: ``gasperm init | preview | collect | klinkenberg``."""

from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

import typer

from gasperm import __version__, units
from gasperm.config import (
    HARDWARE_FILENAME,
    MEASUREMENT_METHODS,
    PLOT_PANELS,
    RUN_FILENAME,
    ConfigError,
    GaspermConfig,
    config_to_dict,
    load_config,
    render_hardware_yaml,
    render_run_yaml,
    render_sample_yaml,
    save_config,
    validate_for_collect,
)

logger = logging.getLogger("gasperm")

#: ``init`` describes the bench and the experiment. The sample file belongs to
#: a core plug, so it is created per plug by ``new-sample``.
INIT_SECTIONS: tuple[str, ...] = ("hardware", "run")

app = typer.Typer(
    add_completion=False,
    help=(
        "Measure gas permeability of core samples on an NI USB-6421 rig.\n\n"
        "Configuration is three files: hardware.yaml (the bench) and run.yaml (the "
        "experiment), both from 'init'; plus one sample file per core plug, from "
        "'new-sample'.\n\n"
        "Typical order: init once per rig, new-sample per plug, collect once per "
        "mean pressure, then klinkenberg across those runs. 'preview' is the "
        "signal check you run in between, which measures and stores nothing."
    ),
)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _parse_spacers(values: list[str], config: GaspermConfig) -> list:
    """Turn ``--spacer TYPE:LENGTH`` arguments into fittings.

    Repeat the flag to stack: ``--spacer wide:50 --spacer wide:50`` is two of
    them. The single word ``none`` declares an empty holder, which is how you
    override a stack that ``run.yaml`` defines by default -- without it there
    would be no way to say "no spacers today" on the command line.

    Raises:
        ValueError: a malformed argument, an unknown bore, or a bad length.
            All caught before the DAQ is opened.
    """
    from gasperm.config import SpacerFitting

    if len(values) == 1 and values[0].strip().lower() == "none":
        return []

    fittings: list[SpacerFitting] = []
    for raw in values:
        text = raw.strip()
        if ":" not in text:
            raise ValueError(
                f"--spacer {raw!r} should be TYPE:LENGTH, e.g. 'wide:50'. Bores "
                "available: "
                + (", ".join(sorted(config.hardware.reservoirs.spacer_types)) or "(none)")
            )
        name, _, length_text = text.partition(":")
        name = name.strip()
        # Resolve the bore now so a typo fails here, naming the alternatives,
        # rather than surfacing later as a quietly wrong V1.
        config.hardware.reservoirs.resolve_spacer_type(name)
        try:
            length = float(length_text)
        except ValueError:
            raise ValueError(
                f"--spacer {raw!r}: {length_text!r} is not a length."
            ) from None
        if length <= 0.0:
            raise ValueError(f"--spacer {raw!r}: the length must be positive.")
        fittings.append(SpacerFitting(type=name, length=length))
    return fittings


def _fail(message: str) -> None:
    """Print an error to stderr and exit non-zero."""
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _configure_logging(verbose: bool, log_file: Path | None = None) -> None:
    """Console logging, plus a per-run file handler once a run directory exists."""
    root = logging.getLogger("gasperm")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers.clear()

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        root.addHandler(file_handler)


def _load_or_fail(
    config_dir: Path,
    hardware: Path | None,
    sample: Path | None,
    run: Path | None,
) -> GaspermConfig:
    try:
        return load_config(config_dir, hardware=hardware, sample=sample, run=run)
    except ConfigError as exc:
        _fail(str(exc))
        raise  # unreachable; keeps type checkers happy


def _set_dotted(data: dict[str, Any], dotted_key: str, raw_value: str) -> None:
    """Apply a ``section.field=value`` override to a plain config dict.

    Paths are rooted at the three sections, e.g. ``sample.length``,
    ``hardware.daq.device_name``, ``run.gas.name``. Values are parsed as YAML
    so numbers, booleans and ``null`` arrive as the right type.
    """
    import yaml

    parts = dotted_key.split(".")
    target = data
    for depth, part in enumerate(parts[:-1]):
        nested = target.get(part)
        if not isinstance(nested, dict):
            available = ", ".join(
                sorted(k for k, v in target.items() if isinstance(v, dict))
            )
            raise ConfigError(
                f"--set {dotted_key}: {'.'.join(parts[: depth + 1])!r} is not a config "
                f"section. Sections available at that level: {available or '(none)'}"
            )
        target = nested
    try:
        target[parts[-1]] = yaml.safe_load(raw_value)
    except yaml.YAMLError:
        # A value YAML cannot parse is taken literally rather than refused.
        # Some perfectly ordinary unit strings are YAML syntax: a bare "%" is a
        # directive indicator, so `--set sample.porosity_unit=%` would fail on a
        # value the schema accepts happily. Anything genuinely wrong is still
        # caught by the model a moment later, with a better message than a YAML
        # scanner can give.
        target[parts[-1]] = raw_value


# Unit menus, taken from the canonical sets in units.py so a prompt can never
# offer something the config would then reject.
PRESSURE_UNITS = ", ".join(sorted(units.SUPPORTED_PRESSURE_UNITS))
FLOW_UNITS = ", ".join(sorted(units.SUPPORTED_FLOW_UNITS))
PERMEABILITY_UNITS = ", ".join(sorted(units.SUPPORTED_PERMEABILITY_UNITS))
LENGTH_UNITS = ", ".join(sorted(units.SUPPORTED_LENGTH_UNITS))

#: How many times a unit prompt re-asks before giving up and keeping the default.
_UNIT_PROMPT_ATTEMPTS = 5


def _prompt_unit(label: str, default: str, options: str, check) -> str:
    """Prompt for a unit, listing the allowed values and re-asking on a typo.

    Validating here rather than at the end of the interview matters: ``init``
    asks a few dozen questions, and letting a mistyped unit surface only at
    final validation would throw away every answer.
    """
    for _ in range(_UNIT_PROMPT_ATTEMPTS):
        value = typer.prompt(f"{label} ({options})", default=default).strip()
        try:
            check(value)
        except ValueError as exc:
            typer.secho(f"      {exc}", fg=typer.colors.YELLOW)
            continue
        return value
    typer.secho(
        f"      Keeping the default {default!r}; edit the file afterwards if needed.",
        fg=typer.colors.YELLOW,
    )
    return default


def _prompt_pressure_unit(label: str, default: str) -> str:
    """Prompt for one of the supported pressure units."""
    return _prompt_unit(label, default, PRESSURE_UNITS, units.normalize_pressure_unit)


def _prompt_flow_unit(label: str, default: str) -> str:
    """Prompt for one of the supported volumetric flow units."""
    return _prompt_unit(
        label, default, FLOW_UNITS, lambda value: units.flow_to_cm3_s(1.0, value)
    )


def _apply_overrides(config: GaspermConfig, assignments: list[str]) -> GaspermConfig:
    data = config_to_dict(config)
    for assignment in assignments:
        if "=" not in assignment:
            raise ConfigError(f"--set {assignment!r} is not in SECTION.FIELD=VALUE form.")
        key, value = assignment.split("=", 1)
        _set_dotted(data, key.strip(), value.strip())
    return GaspermConfig.model_validate(data)


# --------------------------------------------------------------------------
# init
# --------------------------------------------------------------------------


def _prompt_hardware(data: dict[str, Any], defaults: GaspermConfig) -> None:
    typer.secho("\n== hardware.yaml : the bench ==", fg=typer.colors.CYAN, bold=True)
    hardware = data["hardware"]
    hardware["rig_name"] = typer.prompt("  Rig name", default=hardware["rig_name"])

    typer.secho("  DAQ (NI USB-6421)", bold=True)
    daq = hardware["daq"]
    daq["device_name"] = typer.prompt("    Device name (from NI MAX)", default=daq["device_name"])
    daq["inlet_pressure_channel"] = typer.prompt(
        "    Inlet pressure channel", default=daq["inlet_pressure_channel"]
    )
    daq["outlet_pressure_channel"] = typer.prompt(
        "    Outlet pressure channel", default=daq["outlet_pressure_channel"]
    )
    daq["sample_rate_hz"] = typer.prompt(
        "    Sample rate (Hz)", default=daq["sample_rate_hz"], type=float
    )

    typer.secho(f"  Pressure transducers  (units: {PRESSURE_UNITS})", bold=True)
    for side in ("inlet", "outlet"):
        section = hardware["pressure_calibration"][side]
        typer.secho(f"    {side}:", bold=True)
        section["volts_min"] = typer.prompt("      Volts at zero", default=section["volts_min"], type=float)
        section["volts_max"] = typer.prompt("      Volts at full scale", default=section["volts_max"], type=float)
        section["unit"] = _prompt_pressure_unit("      Pressure unit", section["unit"])
        section["value_min"] = typer.prompt(
            f"      Pressure at {section['volts_min']} V ({section['unit']})",
            default=section["value_min"], type=float,
        )
        section["value_max"] = typer.prompt(
            f"      Pressure at {section['volts_max']} V ({section['unit']})",
            default=section["value_max"], type=float,
        )
        section["reading_type"] = typer.prompt(
            "      Reading type (absolute/gauge)", default=section["reading_type"]
        )
        section["uncertainty"]["value"] = typer.prompt(
            "      Accuracy (% of full scale)",
            default=section["uncertainty"]["value"], type=float,
        )
    hardware["pressure_calibration"]["correlation"] = typer.prompt(
        "    Correlation between the two transducers' errors (0 = independent)",
        default=hardware["pressure_calibration"]["correlation"], type=float,
    )

    typer.secho(
        f"  Flowmeters  (define every meter wired; a run picks one)\n"
        f"              (units: {FLOW_UNITS})",
        bold=True,
    )
    meters: dict[str, Any] = {}
    for name, flow in list(hardware["flowmeters"].items()):
        if not typer.confirm(f"    Is a '{name}' meter wired?", default=True):
            continue
        new_name = typer.prompt("      Name", default=name).strip() or name
        flow["channel"] = typer.prompt("      Channel", default=flow["channel"])
        flow["description"] = typer.prompt(
            "      Description (serial, make)", default="", show_default=False
        )
        flow["volts_min"] = typer.prompt(
            "      Volts at zero flow", default=flow["volts_min"], type=float
        )
        flow["volts_max"] = typer.prompt(
            "      Volts at full scale", default=flow["volts_max"], type=float
        )
        flow["unit"] = _prompt_flow_unit("      Flow unit", flow["unit"])
        flow["flow_min"] = typer.prompt(
            f"      Flow at {flow['volts_min']} V ({flow['unit']})",
            default=flow["flow_min"], type=float,
        )
        flow["flow_max"] = typer.prompt(
            f"      Full-scale flow at {flow['volts_max']} V ({flow['unit']})",
            default=flow["flow_max"], type=float,
        )
        flow["reading_basis"] = typer.prompt(
            "      Reading basis (standard = mass-based / actual = at line conditions)",
            default=flow["reading_basis"],
        )
        flow["uncertainty"]["value"] = typer.prompt(
            "      Accuracy (% of reading)", default=flow["uncertainty"]["value"], type=float
        )
        meters[new_name] = flow

    if not meters:
        typer.secho(
            "    No meter defined; keeping the first default so the config stays valid.",
            fg=typer.colors.YELLOW,
        )
        first = next(iter(hardware["flowmeters"]))
        meters = {first: hardware["flowmeters"][first]}
    hardware["flowmeters"] = meters
    hardware["default_flowmeter"] = (
        next(iter(meters))
        if len(meters) == 1
        else typer.prompt(
            f"    Default meter ({', '.join(meters)})", default=next(iter(meters))
        )
    )

    typer.secho("  Temperature probe (Arduino, USB serial)", bold=True)
    temperature = hardware["temperature"]
    temperature["port"] = typer.prompt("    Serial port", default=temperature["port"])
    temperature["baud_rate"] = typer.prompt("    Baud rate", default=temperature["baud_rate"], type=int)
    temperature["parse_pattern"] = typer.prompt(
        "    Line format ('{value}' marks the number; '-' for any number on the line)",
        default=temperature["parse_pattern"],
    )
    if temperature["parse_pattern"] in {"-", ""}:
        temperature["parse_pattern"] = None
    temperature["units"] = typer.prompt("    Probe reports in (C/K/F)", default=temperature["units"])
    temperature["uncertainty"]["value"] = typer.prompt(
        "    Probe accuracy (degC)", default=temperature["uncertainty"]["value"], type=float
    )

    hardware["calibrated_by"] = typer.prompt(
        "  Calibrated by", default="", show_default=False
    )
    hardware["calibrated_on"] = typer.prompt(
        "  Calibrated on (YYYY-MM-DD)", default="", show_default=False
    )


#: Fields that describe the *core*, not the individual plug, so they are the
#: ones ``--from`` may sensibly carry over to a sibling plug.
SHARED_SAMPLE_FIELDS: tuple[str, ...] = (
    "lithology",
    "formation",
    "well",
    "depth",
    "depth_unit",
    "porosity_method",
    "grain_density_g_cm3",
    "prepared_by",
)


def _optional_float(label: str) -> float | None:
    """Prompt for a number that may be left blank, re-asking on a bad one.

    ``typer.prompt(type=float)`` re-asks by itself, but it cannot express
    "blank means unset". Doing it by hand means doing the retry by hand too --
    otherwise one mistyped optional field aborts the whole interview.
    """
    for _ in range(_UNIT_PROMPT_ATTEMPTS):
        answer = typer.prompt(label, default="", show_default=False).strip()
        if not answer:
            return None
        try:
            return float(answer)
        except ValueError:
            typer.secho(f"      {answer!r} is not a number.", fg=typer.colors.YELLOW)
    typer.secho("      Leaving it unset.", fg=typer.colors.YELLOW)
    return None


def _prompt_sample(sample: dict[str, Any], *, inherited: bool = False) -> None:
    """Ask for one core plug's details, in place.

    ``inherited`` means the shared, core-level fields already came from a
    template, so only the per-plug measurements are asked for. Geometry is
    always asked: every plug is cut and measured individually, and silently
    reusing another plug's length or diameter would put a wrong number straight
    into the Darcy equation.
    """
    if not inherited:
        typer.secho("  Core", bold=True)
        sample["lithology"] = typer.prompt("    Lithology", default="", show_default=False)
        sample["formation"] = typer.prompt("    Formation", default="", show_default=False)
        sample["well"] = typer.prompt("    Well", default="", show_default=False)
        depth = _optional_float("    Depth (blank to skip)")
        if depth is not None:
            sample["depth"] = depth
            sample["depth_unit"] = typer.prompt(
                "    Depth unit (m, ft)", default=sample["depth_unit"]
            )
        sample["prepared_by"] = typer.prompt("    Prepared by", default="", show_default=False)

    typer.secho("  This plug", bold=True)
    sample["description"] = typer.prompt("    Description", default="", show_default=False)

    typer.secho("  Geometry (measured per plug)", bold=True)
    sample["dimension_unit"] = _prompt_unit(
        "    Dimension unit",
        sample["dimension_unit"],
        LENGTH_UNITS,
        lambda value: units.length_to_cm(1.0, value),
    )
    unit = sample["dimension_unit"]
    sample["length"] = typer.prompt(
        f"    Length ({unit})", default=sample["length"], type=float
    )
    sample["length_uncertainty"] = typer.prompt(
        f"    Length uncertainty ({unit})",
        default=sample["length_uncertainty"], type=float,
    )
    sample["diameter"] = typer.prompt(
        f"    Diameter ({unit})", default=sample["diameter"], type=float
    )
    sample["diameter_uncertainty"] = typer.prompt(
        f"    Diameter uncertainty ({unit}, counts double in the budget)",
        default=sample["diameter_uncertainty"], type=float,
    )

    typer.secho("  Petrophysics (optional, per plug)", bold=True)
    # Ask for the unit before the number. A pycnometer reports percentage
    # points and every equation wants the fraction, so asking "porosity?" alone
    # invites a silent factor of 100 -- and the two are only distinguishable by
    # eye when the value happens to be above 1.
    porosity_unit = _prompt_unit(
        "    Porosity unit", sample.get("porosity_unit") or "fraction",
        "fraction | %", units.normalize_porosity_unit,
    )
    sample["porosity_unit"] = porosity_unit
    porosity = _optional_float(f"    Porosity ({porosity_unit}, blank to skip)")
    if porosity is not None:
        sample["porosity"] = porosity
        uncertainty = _optional_float(
            f"    Porosity uncertainty ({porosity_unit}, blank to skip)"
        )
        if uncertainty is not None:
            sample["porosity_uncertainty"] = uncertainty
        if not inherited or not sample.get("porosity_method"):
            sample["porosity_method"] = typer.prompt(
                "    Porosity method", default=sample.get("porosity_method") or "",
                show_default=False,
            )
    if not inherited:
        grain = _optional_float("    Grain density (g/cm3, blank to skip)")
        if grain is not None:
            sample["grain_density_g_cm3"] = grain
    bulk = _optional_float("    Bulk density (g/cm3, blank to skip)")
    if bulk is not None:
        sample["bulk_density_g_cm3"] = bulk
    sample["notes"] = typer.prompt("    Notes", default="", show_default=False)


def _prompt_run(data: dict[str, Any]) -> None:
    typer.secho("\n== run.yaml : the experiment ==", fg=typer.colors.CYAN, bold=True)
    run = data["run"]
    meters = list(data["hardware"]["flowmeters"])
    if len(meters) > 1:
        run["flowmeter"] = typer.prompt(
            f"  Flowmeter for this run ({', '.join(meters)})",
            default=data["hardware"]["default_flowmeter"],
        )
    run["operator"] = typer.prompt("  Operator", default="", show_default=False)
    run["institution"] = typer.prompt("  Institution", default="", show_default=False)
    run["project"] = typer.prompt("  Project", default="", show_default=False)
    run["experiment_id"] = typer.prompt("  Experiment id", default="", show_default=False)

    typer.secho("  Gas", bold=True)
    run["gas"]["name"] = typer.prompt(
        "    CoolProp fluid (Nitrogen, Air, CarbonDioxide, Methane, ...)",
        default=run["gas"]["name"],
    )
    run["gas"]["viscosity_relative_uncertainty"] = typer.prompt(
        "    Viscosity model relative uncertainty",
        default=run["gas"]["viscosity_relative_uncertainty"], type=float,
    )

    typer.secho(f"  Conditions  (pressure units: {PRESSURE_UNITS})", bold=True)
    run["confining_pressure_unit"] = _prompt_pressure_unit(
        "    Confining pressure unit", run["confining_pressure_unit"]
    )
    confining = typer.prompt(
        f"    Confining pressure ({run['confining_pressure_unit']}, blank to skip)",
        default="", show_default=False,
    )
    if confining.strip():
        run["confining_pressure"] = float(confining)
    # P1 and P2 both come from the DAQ; the ambient value is only the
    # gauge-to-absolute reference (and the flowmeter's, when it sits at ambient).
    run["atmospheric_pressure_unit"] = _prompt_pressure_unit(
        "    Ambient pressure unit", run["atmospheric_pressure_unit"]
    )
    run["atmospheric_pressure"] = typer.prompt(
        f"    Ambient pressure, for gauge->absolute "
        f"({run['atmospheric_pressure_unit']})",
        default=run["atmospheric_pressure"], type=float,
    )

    typer.secho("  Steady state (gates what gets reported)", bold=True)
    steady = run["steady_state"]
    steady["window_s"] = typer.prompt("    Window (s)", default=steady["window_s"], type=float)
    steady["required_windows"] = typer.prompt(
        "    Consecutive windows required", default=steady["required_windows"], type=int
    )
    steady["relative_stddev_tolerance"] = typer.prompt(
        "    Scatter tolerance (fraction)",
        default=steady["relative_stddev_tolerance"], type=float,
    )
    steady["relative_drift_tolerance"] = typer.prompt(
        "    Drift tolerance (fraction over the window)",
        default=steady["relative_drift_tolerance"], type=float,
    )
    steady["settling_time_s"] = typer.prompt(
        "    Settling time to ignore at the start (s)",
        default=steady["settling_time_s"], type=float,
    )

    typer.secho("  Output  (display only -- never affects the calculation)", bold=True)
    run["output_dir"] = typer.prompt("    Output directory", default=run["output_dir"])
    run["display_pressure_unit"] = _prompt_pressure_unit(
        "    Display pressure unit", run["display_pressure_unit"]
    )
    run["display_flow_unit"] = _prompt_flow_unit(
        "    Display flow unit", run["display_flow_unit"]
    )
    run["display_permeability_unit"] = _prompt_unit(
        "    Display permeability unit",
        run["display_permeability_unit"],
        PERMEABILITY_UNITS,
        lambda value: units.darcy_to(1.0, value),
    )
    run["uncertainty"]["coverage_probability"] = typer.prompt(
        "    Uncertainty coverage probability",
        default=run["uncertainty"]["coverage_probability"], type=float,
    )


#: Runs live inside the rig folder. Written relative to that folder, not to the
#: caller's working directory, so the rig stays self-contained and relocatable
#: -- ``resolved_output_dir()`` anchors it to wherever run.yaml is found.
DEFAULT_OUTPUT_DIR = "./runs"


@app.command("init")
def init_command(
    directory: Path = typer.Argument(
        ...,
        metavar="FOLDER",
        help="Folder to create for this rig. The config files are written inside it.",
    ),
    non_interactive: bool = typer.Option(
        False, "--non-interactive", "-n",
        help="Skip the prompts and write the default templates. Combine with --set.",
    ),
    set_values: list[str] = typer.Option(
        [], "--set", metavar="SECTION.FIELD=VALUE",
        help=(
            "Override a field, e.g. --set sample.length=48.7 --set run.gas.name=Air "
            "--set hardware.daq.device_name=Dev2. Repeatable."
        ),
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files."),
    print_only: bool = typer.Option(
        False, "--print", help="Write nothing; print the three files to stdout."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Create a folder for one rig, holding hardware.yaml and run.yaml.

    FOLDER is created if it does not exist, along with a ``samples/``
    subdirectory for the core plugs. Deliberately writes no sample file: that
    describes one plug, and a rig measures many. Add plugs with
    ``gasperm new-sample``.

    Interactive by default, prompting with the shipped NI USB-6421 defaults.
    ``--non-interactive`` plus ``--set`` covers scripted setup.
    """
    _configure_logging(verbose)

    defaults = GaspermConfig()
    data = config_to_dict(defaults)
    data["run"]["output_dir"] = DEFAULT_OUTPUT_DIR
    if not non_interactive:
        try:
            _prompt_hardware(data, defaults)
            _prompt_run(data)
        except (ValueError, TypeError) as exc:
            _fail(f"Could not read that answer: {exc}")
            return

    try:
        config = GaspermConfig.model_validate(data)
        if set_values:
            config = _apply_overrides(config, set_values)
    except ConfigError as exc:
        _fail(str(exc))
        return
    except Exception as exc:  # noqa: BLE001 - pydantic validation of prompted input
        _fail(f"Those answers do not make a valid config:\n  {exc}")
        return

    if print_only:
        for name, text in (
            (HARDWARE_FILENAME, render_hardware_yaml(config)),
            (RUN_FILENAME, render_run_yaml(config)),
        ):
            typer.secho(f"\n# ===== {name} =====", fg=typer.colors.CYAN, bold=True)
            typer.echo(text)
        return

    try:
        paths = save_config(config, directory, overwrite=force, sections=INIT_SECTIONS)
    except ConfigError as exc:
        _fail(str(exc))
        return

    # A home for the plugs, created up front so the intended layout is visible
    # rather than implied by the documentation.
    samples_dir = directory / "samples"
    try:
        samples_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        typer.secho(f"warning: could not create {samples_dir}: {exc}", fg=typer.colors.YELLOW)

    typer.secho(f"Created {directory}/", fg=typer.colors.GREEN)
    for path in (paths.hardware, paths.run):
        typer.secho(f"  {path.name}", fg=typer.colors.GREEN)
    typer.secho("  samples/        (core plugs go here)", fg=typer.colors.GREEN)
    typer.secho(f"  runs go to {config.run.output_dir}", fg=typer.colors.BRIGHT_BLACK)

    # Environment checks are advisory at init time: a rig is often configured
    # before it is fully wired up. Either way the next steps still apply, so
    # they are printed regardless -- an unwired rig is exactly when knowing
    # what comes next is most useful.
    try:
        for warning in validate_for_collect(config):
            typer.secho(f"note: {warning}", fg=typer.colors.YELLOW)
    except ConfigError as exc:
        typer.secho("\nNot ready for a collect run yet:\n" + str(exc), fg=typer.colors.YELLOW)

    samples = samples_dir.as_posix()
    location = "" if directory == Path(".") else f" --config-dir {directory}"
    typer.echo("\nNext, add a core plug and measure it:")
    typer.echo(f"  gasperm new-sample <id> --dir {samples}")
    typer.echo(f"  gasperm collect{location} --sample {samples}/<id>.yaml")


@app.command("new-sample")
def new_sample_command(
    sample_id: Optional[str] = typer.Argument(
        None, help="Id of the new core plug, e.g. core-042. Asked for if omitted."
    ),
    directory: Path = typer.Option(
        Path("samples"), "--dir", "-d", help="Where to write the sample file."
    ),
    template: Optional[Path] = typer.Option(
        None, "--from",
        help=(
            "Carry the core-level fields over from an existing sample file "
            "(lithology, formation, well, depth, grain density). Geometry and the "
            "per-plug measurements are always asked for."
        ),
    ),
    non_interactive: bool = typer.Option(
        False, "--non-interactive", "-n", help="Skip the prompts. Combine with --set."
    ),
    set_values: list[str] = typer.Option(
        [], "--set", metavar="FIELD=VALUE",
        help="Override a sample field, e.g. --set length=48.7. Repeatable.",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite an existing file."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Add another core plug without touching the rig or run configuration.

    Writes ``<dir>/<sample_id>.yaml``, which ``collect --sample`` then points
    at. Measuring many plugs on one rig means one hardware.yaml, one run.yaml
    and a file per plug -- not a full ``init`` each time.
    """
    import yaml as _yaml

    from gasperm.config.sample import SampleConfig

    _configure_logging(verbose)

    if sample_id is None:
        if non_interactive:
            _fail("No sample id given. Pass one as an argument, e.g. 'gasperm new-sample core-042'.")
            return
        sample_id = typer.prompt("Sample id").strip()
    sample_id = sample_id.strip()
    if not sample_id:
        _fail("The sample id must not be blank.")
        return

    inherited: tuple[str, ...] = ()
    if template is not None:
        try:
            raw = _yaml.safe_load(Path(template).read_text(encoding="utf-8"))
        except (OSError, _yaml.YAMLError) as exc:
            _fail(f"Could not read the template {template}: {exc}")
            return
        if isinstance(raw, dict) and set(raw) == {"sample"}:
            raw = raw["sample"]
        try:
            source = SampleConfig.model_validate(raw)
        except Exception as exc:  # noqa: BLE001
            _fail(f"{template} is not a valid sample file:\n  {exc}")
            return

        # Carry over what describes the *core*. Everything else -- the id, the
        # geometry, the per-plug densities and porosity -- is measured on this
        # plug, and inheriting it would put another plug's numbers straight into
        # the Darcy equation.
        carried = source.model_dump(mode="json", include=set(SHARED_SAMPLE_FIELDS))
        sample = SampleConfig(id=sample_id, **carried)
        inherited = tuple(
            name for name in SHARED_SAMPLE_FIELDS if carried.get(name) not in (None, "")
        )
    else:
        sample = SampleConfig(id=sample_id)

    data = sample.model_dump(mode="json")
    if not non_interactive:
        typer.secho(f"\n== new sample: {sample_id} ==", fg=typer.colors.CYAN, bold=True)
        if inherited:
            typer.secho(
                f"  inherited from {Path(template).name}: {', '.join(inherited)}",
                fg=typer.colors.BRIGHT_BLACK,
            )
        try:
            _prompt_sample(data, inherited=bool(template))
        except (ValueError, TypeError) as exc:
            _fail(f"Could not read that answer: {exc}")
            return
    data["id"] = sample_id

    try:
        for assignment in set_values:
            if "=" not in assignment:
                raise ConfigError(f"--set {assignment!r} is not in FIELD=VALUE form.")
            key, value = assignment.split("=", 1)
            _set_dotted(data, key.strip(), value.strip())
        sample = SampleConfig.model_validate(data)
    except ConfigError as exc:
        _fail(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        _fail(f"Those answers do not make a valid sample:\n  {exc}")
        return

    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in sample_id)
    target = Path(directory) / f"{safe}.yaml"
    if target.exists() and not force:
        _fail(f"{target} already exists. Pass --force to overwrite.")
        return

    config = GaspermConfig(sample=sample)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_sample_yaml(config), encoding="utf-8")

    typer.secho(f"Wrote {target}", fg=typer.colors.GREEN)
    typer.echo(f"\nNext: gasperm collect --sample {target}")


# --------------------------------------------------------------------------
# collect
# --------------------------------------------------------------------------


def _open_temperature_source(config: GaspermConfig):
    """Open the probe, or fall back when it is not required."""
    from gasperm.hardware.temperature import (
        SerialTemperatureReader,
        StaticTemperatureSource,
    )

    settings = config.hardware.temperature
    reader = SerialTemperatureReader(
        settings.port,
        settings.baud_rate,
        parse_pattern=settings.parse_pattern,
        timeout_s=settings.timeout_s,
        unit=settings.units,
        stale_after_s=settings.stale_after_s,
        plausible_min_c=settings.plausible_min_c,
        plausible_max_c=settings.plausible_max_c,
    )
    try:
        reader.open()
    except OSError as exc:
        if settings.required:
            raise
        logger.warning(
            "%s Continuing without the probe because temperature.required is false.", exc
        )
        return StaticTemperatureSource(
            settings.fallback_temperature_c,
            note=f"{settings.port} could not be opened",
        )

    # Wait out the sensor's conversion time once, so that no sample falls back
    # to a guessed temperature. A DS18B20 needs 750 ms; the run is minutes.
    typer.secho(
        f"Waiting for the temperature probe on {settings.port}...",
        fg=typer.colors.CYAN,
        nl=False,
    )
    started = time.monotonic()
    if reader.wait_for_first_reading(settings.warmup_timeout_s):
        typer.secho(f" {time.monotonic() - started:.1f} s", fg=typer.colors.CYAN)
        return reader

    typer.echo("")
    reader.close()
    message = (
        f"The port {settings.port} opened but the probe sent no usable reading within "
        f"{settings.warmup_timeout_s:g} s. Check the baud rate ({settings.baud_rate}), "
        f"that the sketch is running, and that temperature.parse_pattern matches what "
        "it prints."
    )
    if settings.required:
        raise OSError(message)
    logger.warning("%s Continuing on the fallback temperature.", message)
    return StaticTemperatureSource(
        settings.fallback_temperature_c, note=f"{settings.port} never answered"
    )


@app.command("collect")
def collect_command(
    config_dir: Path = typer.Option(
        Path("."), "--config-dir", "-c", help="Directory holding the three config files."
    ),
    hardware: Optional[Path] = typer.Option(None, "--hardware", help="Override hardware.yaml."),
    sample: Optional[Path] = typer.Option(None, "--sample", help="Override sample.yaml."),
    run_file: Optional[Path] = typer.Option(None, "--run", help="Override run.yaml."),
    method: Optional[str] = typer.Option(
        None, "--method", "-m", metavar="steady_state|pulse_decay",
        help="Measurement method for this run. pulse_decay reads no flowmeter "
             "and requires a closed downstream vessel.",
    ),
    leak_test: bool = typer.Option(
        False, "--leak-test",
        help="Run the pulse-decay PRE-STEP instead of a measurement: blank or "
             "bypass the plug, apply the same pulse, and record what the rig "
             "alone does. Implies --method pulse_decay.",
    ),
    spacer: Optional[list[str]] = typer.Option(
        None, "--spacer", metavar="TYPE:LENGTH",
        help="One hollow spacer fitted upstream, e.g. 'wide:50'. Repeat to "
             "stack. Bores are defined in hardware.reservoirs.spacer_types; "
             "the length is in that type's dimension_unit. Replaces run.yaml's "
             "list entirely, so '--spacer none' declares an empty holder.",
    ),
    plot: bool = typer.Option(
        False, "--plot",
        help="Open a live window: one stacked panel per parameter, with the "
             "steady-state criteria drawn on it.",
    ),
    plot_window: Optional[float] = typer.Option(
        None, "--plot-window", metavar="SECONDS",
        help="Live plot shows only this trailing window. Default is run.yaml's "
             "plot.window_s. Implies --plot.",
    ),
    plot_from_start: bool = typer.Option(
        False, "--plot-from-start",
        help="Live plot spans the whole run from t0, overriding any configured "
             "window. Implies --plot.",
    ),
    plot_panels: Optional[str] = typer.Option(
        None, "--plot-panels", metavar="A,B,...",
        help="Comma-separated panels to stack, overriding run.yaml. One of: "
             + ", ".join(PLOT_PANELS) + ". Implies --plot.",
    ),
    duration: Optional[float] = typer.Option(
        None, "--duration", "-d", metavar="SECONDS", help="Stop after this long."
    ),
    samples: Optional[int] = typer.Option(
        None, "--samples", "-n", help="Stop after this many samples."
    ),
    flowmeter: Optional[str] = typer.Option(
        None, "--flowmeter", "-F",
        help="Flowmeter to use for this run, by name from hardware.yaml.",
    ),
    downstream_pressure: Optional[str] = typer.Option(
        None, "--downstream-pressure", "--outlet-pressure", metavar="VALUE|measured",
        help=(
            "Use a supplied pressure as P2 instead of the outlet transducer, for a rig "
            "venting to atmosphere. 'measured' forces the transducer back."
        ),
    ),
    downstream_pressure_unit: Optional[str] = typer.Option(
        None, "--downstream-pressure-unit",
        help="Unit of --downstream-pressure. Defaults to run.yaml's setting.",
    ),
    stop_after_steady: Optional[float] = typer.Option(
        None, "--stop-after-steady", metavar="SECONDS",
        help=(
            "End the run after steady state has held this long. The clock starts when "
            "steady state is confirmed; 0 stops immediately on confirmation."
        ),
    ),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Override run.output_dir."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Acquire pressure, flow and temperature, and compute permeability live.

    The reported result comes from the detected steady-state window: a
    permeability measured while the rig is still equilibrating describes the
    transient, not the rock. Runs until Ctrl+C unless a stop condition is set.
    """
    from gasperm.acquisition import (
        AcquisitionLoop,
        PulseDecayLoop,
        PulseProcessor,
        SampleProcessor,
        console_header,
        format_pulse_reading_line,
        format_reading_line,
        pulse_console_header,
    )
    from gasperm.gas_properties import build_provider
    from gasperm.hardware.daq import DaqError, open_analog_input
    from gasperm.live import run_with_display
    from gasperm.preview import ConsoleThrottle
    from gasperm.storage import RunWriter

    _configure_logging(verbose)
    config = _load_or_fail(config_dir, hardware, sample, run_file)

    if duration is not None:
        config.run.duration_s = duration
    if samples is not None:
        config.run.max_samples = samples
    if output_dir is not None:
        config.run.output_dir = str(output_dir)
    if stop_after_steady is not None:
        config.run.stop_after_steady_s = stop_after_steady
    if downstream_pressure_unit is not None:
        try:
            config.run.downstream_pressure_unit = downstream_pressure_unit
        except ValueError as exc:
            _fail(str(exc))
            return
    if downstream_pressure is not None:
        text = downstream_pressure.strip()
        if text.lower() == "measured":
            config.run.downstream_pressure = "measured"
        else:
            try:
                config.run.downstream_pressure = float(text)
            except ValueError:
                _fail(
                    f"--downstream-pressure {downstream_pressure!r} is neither a number "
                    "nor 'measured'."
                )
                return
            except Exception as exc:  # noqa: BLE001 - pydantic rejects non-positive
                _fail(str(exc))
                return

    # Applied AFTER --downstream-pressure, deliberately. With validate_assignment
    # on, switching to pulse_decay while a supplied P2 is still set raises at the
    # assignment -- correct, but it would reject a command line that sets both
    # and is perfectly consistent once both have landed.
    if method is not None:
        if method not in MEASUREMENT_METHODS:
            _fail(
                f"--method {method!r} is not a measurement method. Available: "
                f"{', '.join(MEASUREMENT_METHODS)}."
            )
            return
        try:
            config.run.method = method
        except Exception as exc:  # noqa: BLE001 - pydantic enforces the pairing
            _fail(str(exc))
            return

    if leak_test:
        # Implies the method, since a leak test is a pulse-decay observation --
        # asking for both would be redundant and forgetting --method would be a
        # confusing refusal.
        try:
            config.run.method = "pulse_decay"
            config.run.purpose = "leak_test"
        except Exception as exc:  # noqa: BLE001 - pydantic enforces the pairing
            _fail(str(exc))
            return
        if config.run.pulse_decay.leak_test_duration_s is None and (
            config.run.duration_s is None and config.run.max_samples is None
        ):
            _fail(
                "A leak test needs a duration: on a tight rig nothing decays, so "
                "there is no completion signal to stop on and the run would never "
                "end. Set pulse_decay.leak_test_duration_s, or pass --duration."
            )
            return

    if spacer:
        try:
            config.run.pulse_decay.upstream_spacers = _parse_spacers(spacer, config)
        except ValueError as exc:
            _fail(str(exc))
            return

    if flowmeter is not None:
        if flowmeter not in config.hardware.flowmeters:
            _fail(
                f"--flowmeter {flowmeter!r} is not defined in hardware.yaml. Available "
                f"meters: {', '.join(sorted(config.hardware.flowmeters))}"
            )
            return
        config.run.flowmeter = flowmeter

    if plot_window is not None and plot_from_start:
        _fail(
            "--plot-window and --plot-from-start ask for opposite views. Pass one: a "
            "trailing window, or the whole run from t0."
        )
        return
    if plot_panels is not None:
        names = [name.strip() for name in plot_panels.split(",") if name.strip()]
        unknown = [name for name in names if name not in PLOT_PANELS]
        if unknown:
            _fail(
                f"--plot-panels does not recognise {', '.join(unknown)}. Available "
                f"panels: {', '.join(PLOT_PANELS)}."
            )
            return
        try:
            config.run.plot.panels = names
        except ValueError as exc:
            _fail(str(exc))
            return
    # Asking for any plot detail means asking for the plot.
    plot = plot or plot_from_start or plot_window is not None or plot_panels is not None

    # Fail loudly and specifically BEFORE opening the DAQ.
    try:
        startup_warnings = validate_for_collect(config)
    except ConfigError as exc:
        _fail(str(exc))
        return
    for warning in startup_warnings:
        typer.secho(f"warning: {warning}", fg=typer.colors.YELLOW, err=True)

    try:
        gas_provider = build_provider(config.run.gas)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Could not set up gas properties: {exc}")
        return

    try:
        temperature_source = _open_temperature_source(config)
    except OSError as exc:
        _fail(
            f"{exc}\nFix temperature.port, or set temperature.required: false to run "
            "without the probe."
        )
        return

    try:
        analog_source = open_analog_input(config)
    except DaqError as exc:
        temperature_source.close()
        _fail(str(exc))
        return

    writer = RunWriter(config)
    try:
        writer.open()
    except OSError as exc:
        analog_source.close()
        temperature_source.close()
        _fail(f"Could not open the run directory {writer.directory}: {exc}")
        return
    _configure_logging(verbose, log_file=writer.log_path)
    logger.info(
        "Sample %s, gas %s, %.3f cm x %.3f cm dia, %.4g Hz, flowmeter %s (%s), operator %s",
        config.sample.id,
        config.run.gas.name,
        config.sample.length_cm,
        config.sample.diameter_cm,
        config.hardware.daq.sample_rate_hz,
        config.flowmeter_name,
        config.flowmeter.summary,
        config.run.operator or "(unset)",
    )

    live_plot = None
    if plot:
        from gasperm.plotting import LivePlot, PlottingUnavailable

        try:
            live_plot = LivePlot(
                config, window_s=plot_window, from_start=plot_from_start
            ).open()
        except PlottingUnavailable as exc:
            typer.secho(f"warning: live plot unavailable: {exc}", fg=typer.colors.YELLOW)
            live_plot = None

    pulse_mode = config.run.method == "pulse_decay"
    processor = (
        PulseProcessor(config, gas_provider)
        if pulse_mode
        else SampleProcessor(config, gas_provider)
    )

    # `ConsoleThrottle` is defined next to `preview`'s console loop, which needs
    # the same thing for the same reason: this callback runs
    # inside the sample slot, and writing a line to a Windows console is slow
    # enough that echoing every sample at 10 Hz starves the loop -- while
    # scrolling far too fast to read. The CSV keeps every sample regardless;
    # only what reaches the terminal is thinned. A rate slower than the throttle
    # prints every sample, since `due()` is true whenever the interval has
    # passed.
    console = ConsoleThrottle()
    printed_index: int | None = None

    def show(reading) -> None:
        nonlocal printed_index
        if pulse_mode:
            typer.echo(format_pulse_reading_line(reading, loop.status, config))
        else:
            typer.echo(format_reading_line(reading, config))
        printed_index = reading.index

    def on_reading(reading) -> None:
        writer.write(reading)
        if console.due(time.monotonic()):
            show(reading)
        if live_plot is not None:
            # Buffering only -- O(1), and safe to call from the acquisition
            # thread. The drawing itself belongs to whoever owns the main
            # thread; see `run_with_display`.
            #
            # The detector's verdict is what the criterion lines are drawn
            # from; it is already updated for this reading by the time the
            # loop calls back. A pulse run has no steady-state status, so the
            # plot draws no criteria -- which its own panels already reflect.
            live_plot.add(reading, None if pulse_mode else loop.status)

    loop_class = PulseDecayLoop if pulse_mode else AcquisitionLoop
    loop = loop_class(
        config, processor, analog_source, temperature_source, on_reading=on_reading
    )

    typer.secho(f"\nRecording to {writer.directory}   (Ctrl+C to stop)", fg=typer.colors.CYAN)
    meter_text = (
        "no flowmeter (pulse decay measures no flow)"
        if pulse_mode
        else f"flowmeter {config.flowmeter_name} ({config.flowmeter.summary})"
    )
    typer.secho(
        f"Sample {config.sample.id}   gas {config.run.gas.name}   {meter_text}",
        fg=typer.colors.CYAN,
    )
    if not config.run.downstream_is_measured:
        typer.secho(
            f"P2 is the supplied {config.run.downstream_pressure:g} "
            f"{config.run.downstream_pressure_unit}, not the outlet transducer.",
            fg=typer.colors.YELLOW,
        )
    if pulse_mode:
        _print_pulse_criteria(config, processor)
    elif config.run.steady_state.enabled:
        criteria = config.run.steady_state
        typer.secho(
            f"Steady state: {criteria.required_windows} x {criteria.window_s:g} s windows, "
            f"scatter <= {criteria.relative_stddev_tolerance:.2%}, "
            f"drift <= {criteria.relative_drift_tolerance:.2%}, "
            f"on {', '.join(criteria.signals)}",
            fg=typer.colors.CYAN,
        )
    else:
        typer.secho(
            "Steady-state detection is DISABLED -- the result will not be a "
            "representative permeability.",
            fg=typer.colors.YELLOW,
        )
    typer.echo("")
    typer.echo(pulse_console_header(config) if pulse_mode else console_header(config))

    exit_code = 0
    try:
        if live_plot is None:
            loop.run()
        else:
            # With a window open the loop moves to a worker and the drawing
            # keeps the main thread -- the only order matplotlib's GUI backends
            # allow, and the only one that keeps a 0.15 s frame out of a 0.1 s
            # sample slot. Without a window there is nothing to draw and the
            # loop stays where it is, on the simpler path.
            live_plot.on_own_thread = True
            run_with_display(loop, live_plot)
    except DaqError as exc:
        logger.error("%s", exc)
        typer.secho(f"\nAcquisition stopped: {exc}", fg=typer.colors.RED, err=True)
        exit_code = 1
    except KeyboardInterrupt:  # pragma: no cover - the handler normally catches this
        logger.info("Interrupted.")
    finally:
        writer.close()
        # The console is throttled, so the last line on screen is whichever
        # sample happened to land on a tick. Show the final one unconditionally
        # -- a run that ends between ticks would otherwise close on a stale
        # reading, and on a pulse run that line is the decay's last word.
        if loop.readings and loop.readings[-1].index != printed_index:
            show(loop.readings[-1])
        if live_plot is not None:
            live_plot.close()

    if not loop.readings:
        writer.write_metadata()
        _fail("No samples were recorded.")
        return

    summary = None
    try:
        summary = loop.summarize(csv_path=str(writer.readings_path))
    except ValueError as exc:
        typer.secho(f"\n{exc}", fg=typer.colors.YELLOW)
    writer.write_metadata(summary)

    if summary is not None and summary.pulse_decay is not None:
        # Free at the end of a multi-hour run, and the single most diagnostic
        # artefact it produces: a leak or a thermal ramp shows as structure in
        # the residuals long before it shows in R^2.
        try:
            from gasperm.plotting import plot_pulse_decay

            saved = plot_pulse_decay(
                summary.pulse_decay,
                loop.readings,
                path=writer.directory / "decay_fit.png",
                pressure_unit=config.run.display_pressure_unit,
            )
            if saved is not None:
                typer.secho(f"Decay fit plotted to {saved}", fg=typer.colors.CYAN)
        except Exception as exc:  # noqa: BLE001 - a plot must never fail a run
            logger.debug("Could not plot the decay fit: %s", exc)

    typer.echo("")
    if summary is not None:
        _print_run_summary(summary, config)
    typer.secho(f"\nRun written to {writer.directory}", fg=typer.colors.GREEN)
    if loop.stop_reason:
        typer.echo(f"Stopped: {loop.stop_reason}")
    if loop.warnings:
        typer.secho(
            f"{len(loop.warnings)} note(s) logged to {writer.log_path}",
            fg=typer.colors.YELLOW,
        )

    try:
        _print_collect_next_steps(config, writer, config_dir, output_dir is not None)
    except OSError as exc:
        # A convenience footer must never turn a good run into a traceback, nor
        # swallow the exit code that says the run produced no measurement.
        logger.debug("Could not summarise sibling runs: %s", exc)

    # Exit 2 keeps its meaning across both methods -- "this run did not produce a
    # confirmed measurement" -- which is a confirmed steady window for one and an
    # accepted decay fit for the other.
    if summary is not None and not summary.measurement_confirmed:
        exit_code = exit_code or 2
    if exit_code:
        raise typer.Exit(code=exit_code)


@app.command("preview")
def preview_command(
    config_dir: Path = typer.Option(
        Path("."), "--config-dir", "-c", help="Directory holding hardware.yaml and run.yaml."
    ),
    hardware: Optional[Path] = typer.Option(None, "--hardware", help="Override hardware.yaml."),
    run_file: Optional[Path] = typer.Option(None, "--run", help="Override run.yaml."),
    signal: Optional[list[str]] = typer.Option(
        None, "--signal", "-s", metavar="NAME[:UNIT]",
        help="Signal to preview, e.g. 'inlet_pressure' or 'inlet_pressure:bar'. "
             "Repeat for more. 'pulse' selects both transducers a pulse-decay "
             "run would read -- the dedicated pair when the rig has one, the "
             "steady-state pair when it does not -- plus 'pulse_dp', their "
             "difference, which is what the method measures. A bare channel "
             "name (ai7) previews an uncalibrated input as raw volts. Default: "
             "every signal this rig defines. See --list.",
    ),
    list_signals: bool = typer.Option(
        False, "--list",
        help="Print what this rig can preview, with each signal's channel, "
             "range and calibration, then exit. Touches no hardware.",
    ),
    volts: bool = typer.Option(
        False, "--volts",
        help="Show raw voltages instead of calibrated values -- what the wire "
             "is doing, before any calibration has an opinion about it. A "
             "differential like pulse_dp has no wire of its own and stays in "
             "its pressure unit.",
    ),
    plot: bool = typer.Option(
        False, "--plot", help="Open a live window, one stacked panel per signal."
    ),
    plot_window: Optional[float] = typer.Option(
        None, "--plot-window", metavar="SECONDS",
        help="Live plot shows only this trailing window. Implies --plot.",
    ),
    plot_from_start: bool = typer.Option(
        False, "--plot-from-start",
        help="Live plot spans the whole session from t0. Implies --plot.",
    ),
    rate: Optional[float] = typer.Option(
        None, "--rate", metavar="HZ",
        help="Sampling rate. Defaults to daq.sample_rate_hz.",
    ),
    duration: Optional[float] = typer.Option(
        None, "--duration", "-d", metavar="SECONDS", help="Stop after this long."
    ),
    samples: Optional[int] = typer.Option(
        None, "--samples", "-n", help="Stop after this many samples."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Watch the rig's raw signals. Computes nothing, stores nothing.

    A diagnostic view for checking that a transducer reads what you think it
    does, and how noisy it is right now -- with no plug in the holder and no
    run directory created. Only the channels you select are opened, which is
    what lets you look at the flowmeter a run is *not* using, or at a bare
    input, without editing a config file. Runs until Ctrl+C unless stopped.
    """
    from gasperm.config import load_bench_config
    from gasperm.hardware.daq import DaqError, NiDaqAnalogInput
    from gasperm.live import run_with_display
    from gasperm.preview import (
        GROUPS,
        ConsoleThrottle,
        PreviewError,
        PreviewLoop,
        available_signals,
        describe_signals,
        format_preview_line,
        preview_channel_specs,
        preview_header,
        pulse_transducer_pair,
        resolve_signals,
    )

    _configure_logging(verbose)
    try:
        config = load_bench_config(config_dir, hardware=hardware, run=run_file)
    except ConfigError as exc:
        _fail(str(exc))
        return

    if list_signals:
        catalogue = list(available_signals(config).values())
        typer.secho(f"Signals available on {config.daq.device_name}:", fg=typer.colors.CYAN)
        for line in describe_signals(catalogue, volts=False):
            typer.echo(line)
        typer.echo("\nGroups, selectable with one --signal:")
        for name, members in GROUPS.items():
            typer.echo(f"  {name:<10}  {', '.join(members)}")
        typer.echo("  flow        whichever meter run.yaml selects")
        if not pulse_transducer_pair(config)[2]:
            typer.secho(
                "\nThis rig has NO dedicated pulse transducers, so 'pulse' resolves to "
                "the\nsteady-state pair -- which is also what a pulse-decay run would "
                "read. Set\nhardware.pulse_transducers if you have a lower-range pair "
                "wired.",
                fg=typer.colors.YELLOW,
            )
        typer.echo(
            "\nAny bare channel name (ai0, ai7, ...) also works, as raw volts.\n"
            "Add ':UNIT' to a signal to change its unit, e.g. --signal inlet_pressure:bar."
        )
        return

    try:
        signals = resolve_signals(config, signal)
    except PreviewError as exc:
        _fail(str(exc))
        return

    if plot_window is not None and plot_from_start:
        _fail(
            "--plot-window and --plot-from-start ask for opposite views. Pass one: a "
            "trailing window, or the whole session from t0."
        )
        return
    plot = plot or plot_from_start or plot_window is not None

    rate_hz = rate if rate is not None else config.hardware.daq.sample_rate_hz
    if rate_hz <= 0.0:
        _fail(f"--rate must be positive, got {rate_hz:g}.")
        return

    # Only what was selected is opened -- the whole point of the command.
    specs = preview_channel_specs(signals)
    wants_probe = any(s.from_probe for s in signals)

    temperature_source = None
    if wants_probe:
        try:
            temperature_source = _open_temperature_source(config)
        except OSError as exc:
            if signal:
                # Explicitly asked for, so a silent drop would leave the
                # operator watching a column that is never going to appear.
                _fail(f"{exc}\nDrop '--signal temperature' to preview without the probe.")
                return
            typer.secho(f"warning: {exc}", fg=typer.colors.YELLOW, err=True)
            typer.secho("Previewing without the temperature probe.", fg=typer.colors.YELLOW)
            signals = [s for s in signals if not s.from_probe]
            if not signals:
                _fail("Nothing left to preview.")
                return

    analog_source = None
    if specs:
        try:
            analog_source = NiDaqAnalogInput(
                config.daq.device_name,
                specs,
                terminal_config=config.daq.terminal_config,
            ).open()
        except DaqError as exc:
            if temperature_source is not None:
                temperature_source.close()
            _fail(str(exc))
            return

    live_plot = None
    if plot:
        from gasperm.plotting import PlottingUnavailable, PreviewPlot

        try:
            live_plot = PreviewPlot(
                signals,
                volts=volts,
                window_s=plot_window,
                from_start=plot_from_start,
                max_points=config.run.plot.max_points,
                redraw_interval_s=config.run.plot.redraw_interval_s,
                device_name=config.daq.device_name,
                plot_config=config.run.plot,
            ).open()
        except PlottingUnavailable as exc:
            typer.secho(f"warning: live plot unavailable: {exc}", fg=typer.colors.YELLOW)
            live_plot = None

    typer.secho(
        f"\nPreviewing {len(signals)} signal(s) at {rate_hz:g} Hz   (Ctrl+C to stop)",
        fg=typer.colors.CYAN,
    )
    for line in describe_signals(signals, volts=volts):
        typer.secho(line, fg=typer.colors.CYAN)
    typer.secho(
        "Nothing is computed and nothing is written -- this is a signal check, "
        "not a measurement.",
        fg=typer.colors.CYAN,
    )
    typer.echo("")
    typer.echo(preview_header(signals, volts=volts))

    # The DAQ is sampled at the full rate so the plot and any judgement about
    # noise see the real signal; the console is throttled because ten updates a
    # second is not readable.
    throttle = ConsoleThrottle()
    live = sys.stdout.isatty()

    def show(sample) -> None:
        line = format_preview_line(sample, signals, volts=volts)
        if live:
            # Rewrite one line in place: a preview is watched, not read back.
            typer.echo(f"\r{line}", nl=False)
        else:
            typer.echo(line)

    def on_sample(sample) -> None:
        if live_plot is not None:
            # Buffering only; the drawing belongs to the main thread. See
            # `run_with_display`.
            live_plot.add(sample)
        if throttle.due(time.monotonic()):
            show(sample)

    loop = PreviewLoop(
        signals,
        analog_source,
        temperature_source,
        rate_hz=rate_hz,
        duration_s=duration,
        max_samples=samples,
        on_sample=on_sample,
    )

    exit_code = 0
    try:
        if live_plot is None:
            loop.run()
        else:
            live_plot.on_own_thread = True
            run_with_display(loop, live_plot)
    except DaqError as exc:
        logger.error("%s", exc)
        typer.secho(f"\nPreview stopped: {exc}", fg=typer.colors.RED, err=True)
        exit_code = 1
    except KeyboardInterrupt:  # pragma: no cover - the handler normally catches this
        pass
    finally:
        # The console is throttled, so the last thing on screen is whatever
        # sample happened to land on a tick. Show the final one unconditionally
        # -- a short or fast preview would otherwise end on a stale line.
        if loop.latest is not None:
            show(loop.latest)
        if live:
            typer.echo("")
        if live_plot is not None:
            live_plot.close()

    typer.secho(
        f"\n{loop.sample_count} sample(s) previewed. Nothing was written.",
        fg=typer.colors.GREEN,
    )
    if loop.stop_reason:
        typer.echo(f"Stopped: {loop.stop_reason}")
    if exit_code:
        raise typer.Exit(code=exit_code)


def _klinkenberg_command_line(
    sample_id: str, config_dir: Path, runs_dir: Path | None = None
) -> str:
    """The ready-to-paste regression command for this plug."""
    parts = ["gasperm klinkenberg"]
    if config_dir != Path("."):
        parts.append(f"-c {config_dir}")
    parts.append(f"--sample {sample_id}")
    if runs_dir is not None:
        # run.yaml no longer names where this run went, so spell it out.
        parts.append(f"--runs-dir {runs_dir}")
    parts.append("--plot")
    return " ".join(parts)


def _print_collect_next_steps(
    config: GaspermConfig, writer, config_dir: Path, output_dir_overridden: bool
) -> None:
    """Say how many runs this plug has now, and how to regress them.

    Deliberately a plain count: how many points a Klinkenberg series needs is
    the operator's call, so nothing here suggests a target or a next pressure.
    """
    from gasperm.storage import find_runs, runs_for_sample

    # The writer's own parent, not run.output_dir, so the count and the command
    # can never disagree with where this run actually landed.
    runs_dir = writer.directory.parent
    recorded = runs_for_sample(find_runs(runs_dir), config.sample.id)

    typer.echo("")
    typer.secho(
        f"{len(recorded)} run{'s' if len(recorded) != 1 else ''} recorded for "
        f"{config.sample.id} in {runs_dir}.",
        fg=typer.colors.CYAN,
    )
    typer.echo("\nRegress them:")
    typer.echo(
        "  "
        + _klinkenberg_command_line(
            config.sample.id, config_dir, runs_dir if output_dir_overridden else None
        )
    )


def _print_pulse_criteria(config: GaspermConfig, processor) -> None:
    """Startup banner for a pulse-decay run: vessels, criteria, expected length."""
    pulse = config.run.pulse_decay
    reservoirs = config.hardware.reservoirs
    if config.run.purpose == "leak_test":
        typer.secho(
            "LEAK TEST -- this run measures the APPARATUS, not the sample.",
            fg=typer.colors.YELLOW,
            bold=True,
        )
        typer.secho(
            "Blank or bypass the plug, charge both vessels to the pressure you will "
            "measure at, and apply the same pulse. Whatever decays is the rig.",
            fg=typer.colors.YELLOW,
        )
    spacers = pulse.upstream_spacers
    typer.secho(
        f"Pulse decay: {reservoirs.describe(spacers)}"
        f"   effective {reservoirs.effective_volume_cm3(spacers):.4g} cm3, "
        f"{processor.storage_correction.replace('_', '-')} model",
        fg=typer.colors.CYAN,
    )
    if spacers:
        effect = (
            reservoirs.effective_volume_cm3(spacers)
            / reservoirs.effective_volume_cm3()
            - 1.0
        )
        stack = ", ".join(str(fitting) for fitting in spacers)
        typer.secho(
            f"Confirm {len(spacers)} spacer{'s' if len(spacers) != 1 else ''} "
            f"[{stack}] {'are' if len(spacers) != 1 else 'is'} actually fitted: "
            f"they raise the reported k by {effect:.2%}.",
            fg=typer.colors.YELLOW,
        )
    if config.run.purpose == "leak_test":
        duration = pulse.leak_test_duration_s or config.run.duration_s
        typer.secho(
            f"Apply a pulse of at least {pulse.min_pulse_pressure:g} "
            f"{pulse.pulse_pressure_unit} and let it sit for "
            f"{duration:g} s. Nothing decaying is the result you want.",
            fg=typer.colors.CYAN,
        )
    else:
        typer.secho(
            f"Apply a pulse of at least {pulse.min_pulse_pressure:g} "
            f"{pulse.pulse_pressure_unit}; the run ends at dP/dP0 = "
            f"{pulse.stop_below_fraction:g}, fitting "
            f"{pulse.fit_start_fraction:g} down to {pulse.fit_end_fraction:g}.",
            fg=typer.colors.CYAN,
        )


def _print_pulse_decay_result(summary, config: GaspermConfig) -> None:
    """The decay-specific block of a pulse-decay run summary."""
    result = summary.pulse_decay
    unit = config.run.display_pressure_unit
    if result is None:
        if summary.purpose == "leak_test":
            # For a leak test the absence of a decay is the pass condition, so
            # the same state reads the opposite way round.
            typer.secho(
                "  purpose             LEAK TEST -- the apparatus, not the sample",
                fg=typer.colors.YELLOW,
                bold=True,
            )
            typer.secho(
                "  leak                NONE MEASURABLE -- the blanked rig held its "
                "differential for the whole run",
                fg=typer.colors.GREEN,
                bold=True,
            )
            return
        typer.secho(
            "  decay fit           REJECTED -- no decay could be fitted, so this run "
            "did not measure the sample",
            fg=typer.colors.RED,
            bold=True,
        )
        return

    correction = result.storage_correction.replace("_", " & ").title()
    if summary.purpose == "leak_test":
        typer.secho(
            "  purpose             LEAK TEST -- the apparatus, not the sample",
            fg=typer.colors.YELLOW,
            bold=True,
        )
    typer.echo(f"  method              pulse decay -- {correction} model")
    amplitude = units.from_atm(result.pulse_amplitude_atm, unit)
    fraction = (
        result.pulse_amplitude_atm / summary.mean_pressure_atm
        if summary.mean_pressure_atm
        else 0.0
    )
    typer.echo(
        f"  pulse               dP0 = {amplitude:.4g} {unit} at "
        f"t = {result.pulse_at_elapsed_s:.1f} s   ({fraction:.2%} of P_mean)"
    )
    typer.secho(
        f"  decay fit           "
        f"{'ACCEPTED' if summary.measurement_confirmed else 'REJECTED'}   "
        f"{result.fit_start_elapsed_s:.1f}-{result.fit_end_elapsed_s:.1f} s, "
        f"{result.fit_sample_count} pts",
        fg=typer.colors.GREEN if summary.measurement_confirmed else typer.colors.RED,
        bold=not summary.measurement_confirmed,
    )
    relative = result.relative_standard_uncertainty
    relative_text = f" +/- {relative:.2%}" if relative is not None else ""
    typer.echo(
        f"                      alpha = {result.decay_rate_per_s:.4e} 1/s"
        f"{relative_text},  tau = {result.time_constant_s:.0f} s"
    )
    offset_text = (
        f",  offset = {units.from_atm(result.fitted_offset_atm, unit):+.4g} {unit}"
        if result.fitted_offset_atm is not None
        else ""
    )
    autocorrelation = (
        f",  rho_1 = {result.residual_autocorrelation:.2f}"
        if result.residual_autocorrelation is not None
        else ""
    )
    typer.echo(
        f"                      R^2 = {result.r_squared:.6f}{offset_text}{autocorrelation}"
        f"  [{result.fit_model}]"
    )
    vessels = (
        f"  volumes             V1 = {result.upstream_volume_cm3:g} cm3"
    )
    if result.upstream_spacers:
        vessels += (
            f" (incl. {len(result.upstream_spacers)} spacer"
            f"{'s' if len(result.upstream_spacers) != 1 else ''} = "
            f"{result.spacer_volume_cm3:g} cm3: "
            f"{', '.join(result.upstream_spacers)})"
        )
    vessels += f", V2 = {result.downstream_volume_cm3:g} cm3"
    if result.upstream_storage_ratio is not None:
        vessels += (
            f"    a1 = {result.upstream_storage_ratio:.3f}, "
            f"a2 = {result.downstream_storage_ratio:.3f}"
        )
    typer.echo(vessels)
    if result.leak_rate_per_s is not None:
        unit = config.run.display_permeability_unit
        equivalent = result.leak_equivalent_permeability_darcy
        fraction = result.leak_fraction
        text = (
            f"  leak test           {result.leak_test_source}: "
            f"{result.leak_rate_per_s:.4e} 1/s"
        )
        if equivalent:
            text += f" = {units.darcy_to(equivalent, unit):.4g} {unit}"
        if fraction is not None:
            text += f"   ({fraction:.1%} of this decay)"
        if result.leak_subtracted:
            text += "  [SUBTRACTED]"
        typer.secho(
            text,
            fg=(
                typer.colors.GREEN
                if fraction is not None
                and fraction <= config.run.pulse_decay.max_leak_fraction
                else typer.colors.YELLOW
            ),
        )
    if result.storage_root is not None:
        ratios = result.upstream_storage_ratio + result.downstream_storage_ratio
        understated = ratios / result.storage_root**2
        typer.echo(
            f"                      theta_1 = {result.storage_root:.4f}   "
            f"(the zero-storage form would read {1.0 - 1.0 / understated:.1%} low)"
        )


def _print_run_summary(summary, config: GaspermConfig) -> None:
    run = config.run
    pressure_unit = run.display_pressure_unit
    permeability_unit = run.display_permeability_unit

    k_display = units.darcy_to(summary.permeability_darcy, permeability_unit)
    p_display = units.from_atm(summary.mean_pressure_atm, pressure_unit)

    typer.secho("Run summary", bold=True)
    typer.echo(f"  sample              {summary.sample_id}  ({summary.gas_name})")
    if summary.metadata and summary.metadata.flowmeter:
        typer.echo(
            f"  flowmeter           {summary.metadata.flowmeter} "
            f"({summary.metadata.flowmeter_range})"
        )
    if summary.metadata and summary.metadata.operator:
        typer.echo(f"  operator            {summary.metadata.operator}")
    if summary.metadata and summary.metadata.lithology:
        typer.echo(f"  lithology           {summary.metadata.lithology}")
    if summary.metadata and summary.metadata.confining_pressure is not None:
        typer.echo(
            f"  confining pressure  {summary.metadata.confining_pressure:g} "
            f"{summary.metadata.confining_pressure_unit}"
        )
    typer.echo(
        f"  duration            {summary.duration_s:.1f} s over {summary.sample_count} samples"
    )

    if summary.method == "pulse_decay":
        _print_pulse_decay_result(summary, config)
    elif summary.steady_state_reached and summary.steady_state_window is not None:
        window = summary.steady_state_window
        typer.secho(
            f"  steady state        CONFIRMED, {window.start_elapsed_s:.1f}-"
            f"{window.end_elapsed_s:.1f} s ({window.sample_count} samples)",
            fg=typer.colors.GREEN,
        )
    else:
        typer.secho(
            "  steady state        NOT REACHED -- the figures below are not a "
            "representative measurement",
            fg=typer.colors.RED,
            bold=True,
        )

    typer.echo(f"  mean pressure       {p_display:.4g} {pressure_unit}")
    typer.echo(f"  mean temperature    {summary.mean_temperature_c:.2f} C")
    if summary.mean_flow_cm3_s is not None:
        typer.echo(
            f"  mean flow           "
            f"{units.flow_from_cm3_s(summary.mean_flow_cm3_s, run.display_flow_unit):.4g} "
            f"{run.display_flow_unit}"
        )

    good = bool(summary.measurement_confirmed)
    if summary.purpose == "leak_test" and summary.pulse_decay is None:
        # Nothing decayed, so there is no permeability to report -- and a bare
        # "0" would read as a measurement of zero rather than as no signal.
        return
    # A leak test's "permeability" is what the apparatus alone would fake, not
    # a property of any rock, so it must not be labelled as one.
    label = "leak equivalent" if summary.purpose == "leak_test" else "apparent k_g"
    budget = summary.uncertainty
    if budget is not None:
        expanded = units.darcy_to(budget.expanded_uncertainty_darcy, permeability_unit)
        typer.secho(
            f"  {label:<18}  {k_display:.5g} +/- {expanded:.3g} {permeability_unit}"
            f"  ({budget.relative_expanded_uncertainty:.2%}, k = {budget.coverage_factor:.2f},"
            f" {budget.coverage_probability:.0%})",
            fg=typer.colors.GREEN if good else typer.colors.YELLOW,
            bold=True,
        )
        _print_budget(budget)
    else:
        stddev = units.darcy_to(summary.permeability_stddev_darcy, permeability_unit)
        typer.secho(
            f"  {label:<18}  {k_display:.5g} +/- {stddev:.3g} {permeability_unit} (1 sd)",
            fg=typer.colors.GREEN if good else typer.colors.YELLOW,
            bold=True,
        )

    for warning in summary.warnings[-4:]:
        typer.secho(f"  ! {warning}", fg=typer.colors.YELLOW)


def _print_budget(budget) -> None:
    typer.secho("\n  Uncertainty budget (ISO/IEC Guide 98-3)", bold=True)
    typer.echo(
        f"    {'quantity':<24} {'type':>4} {'u(x)/x':>10} {'c_i':>8} "
        f"{'|c*u|':>10} {'share':>7}"
    )
    total_variance = sum(component.variance_share for component in budget.components)
    for component in sorted(
        budget.components, key=lambda c: c.variance_share, reverse=True
    ):
        share = component.variance_share / total_variance if total_variance else 0.0
        typer.echo(
            f"    {component.name[:24]:<24} {component.evaluation_type:>4} "
            f"{component.relative_standard_uncertainty:>10.3%} "
            f"{component.relative_sensitivity:>8.3f} "
            f"{component.relative_contribution:>10.3%} {share:>7.1%}"
        )
    dof = budget.effective_degrees_of_freedom
    dof_text = "inf" if not math.isfinite(dof) else f"{dof:.0f}"
    typer.echo(
        f"    combined u_c/k = {budget.relative_combined_standard_uncertainty:.3%}, "
        f"v_eff = {dof_text}"
    )
    for note in budget.notes:
        typer.secho(f"    note: {note}", fg=typer.colors.BRIGHT_BLACK)


# --------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------


@app.command("summarize")
def summarize_command(
    sample: Optional[str] = typer.Argument(
        None, metavar="[SAMPLE]",
        help="The plug to summarise, by id or by its sample file. Omit to list "
             "every plug the runs directory holds.",
    ),
    config_dir: Path = typer.Option(
        Path("."), "--config-dir", "-c", help="Rig folder holding run.yaml."
    ),
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Where the runs are. Defaults to run.yaml's output_dir."
    ),
    window: Optional[float] = typer.Option(
        None, "--window", metavar="SECONDS", help="Override the stored averaging window."
    ),
    allow_unsteady: bool = typer.Option(
        False, "--allow-unsteady",
        help="Count runs that never confirmed a measurement. Off by default: "
             "such a run describes the transient, not the rock.",
    ),
    allow_mixed_methods: bool = typer.Option(
        False, "--allow-mixed-methods",
        help="Let the Klinkenberg fit mix steady-state and pulse-decay points.",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write the summary to a YAML file."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Everything one core plug has been through, on one page.

    Its geometry and provenance, every run with its result and uncertainty, the
    Klinkenberg fit across them, any leak tests behind them -- and, more useful
    than the table, **what is missing**: a run that never confirmed, a series
    one pressure short of a fit, two meters where there should be one, a
    pulse-decay campaign with no leak test behind it.

    A re-derived run supersedes the original it came from, so one measurement is
    never counted twice.

    With no plug named, lists every one the runs directory holds.
    """
    from gasperm.klinkenberg import fit_klinkenberg
    from gasperm.storage import (
        drop_superseded,
        find_runs,
        runs_for_sample,
        summary_from_run,
        write_sample_summary,
    )
    from gasperm.summary import build_report

    _configure_logging(verbose)
    resolved_runs_dir = _resolve_runs_dir(runs_dir, config_dir)
    everything = find_runs(resolved_runs_dir)
    if not everything:
        _fail(f"No runs found under {resolved_runs_dir}.")
        return

    if sample is None:
        _print_plug_roster(everything, resolved_runs_dir)
        return

    sample_id = _resolve_sample_id(sample)
    records = runs_for_sample(everything, sample_id)
    if not records:
        present = sorted({r.sample_id for r in everything if r.sample_id})
        _fail(
            f"No runs found for {sample_id!r} in {resolved_runs_dir}."
            + (f" Plugs recorded there: {', '.join(present)}." if present else "")
        )
        return

    records, superseded = drop_superseded(records)
    records = sorted(records, key=lambda r: (r.started_at is None, r.started_at))
    summaries = [summary_from_run(record.directory) for record in records]

    fit = None
    usable = [
        record
        for record, summary in zip(records, summaries)
        if record.purpose != "leak_test"
        and (allow_unsteady or (summary and summary.measurement_confirmed))
    ]
    if len(usable) >= 2:
        points, _ = _discover_points(
            usable, window=window, allow_unsteady=allow_unsteady
        )
        if len(points) >= 2:
            try:
                fit = fit_klinkenberg(points, allow_mixed_methods=allow_mixed_methods)
            except ValueError as exc:
                logger.debug("No Klinkenberg fit for %s: %s", sample_id, exc)

    report = build_report(sample_id, records, summaries, klinkenberg=fit)
    _print_sample_summary(
        report, resolved_runs_dir, superseded, _summary_pressure_unit(config_dir)
    )

    if output is not None:
        saved = write_sample_summary(report, output)
        typer.secho(f"\nSummary written to {saved}", fg=typer.colors.GREEN)


def _print_plug_roster(records, runs_dir: Path) -> None:
    """One line per plug: what exists, so a plug can be named to look at it."""
    from gasperm.storage import drop_superseded

    by_plug: dict[str, list] = {}
    for record in records:
        by_plug.setdefault(record.sample_id or "(unknown)", []).append(record)

    typer.secho(f"\n{len(by_plug)} plug(s) in {runs_dir}", bold=True)
    typer.echo(
        f"  {'plug':<20} {'runs':>5} {'confirmed':>10} {'leak':>5}  "
        f"{'first':<11} {'last':<11}  methods"
    )
    for plug in sorted(by_plug):
        kept, _ = drop_superseded(by_plug[plug])
        measurements = [r for r in kept if r.purpose != "leak_test"]
        confirmed = [r for r in measurements if r.measurement_confirmed]
        leaks = [r for r in kept if r.purpose == "leak_test"]
        stamps = sorted(r.started_at for r in kept if r.started_at)
        methods = sorted({r.method or "steady_state" for r in measurements})
        typer.echo(
            f"  {plug[:20]:<20} {len(measurements):>5} {len(confirmed):>10} "
            f"{len(leaks):>5}  "
            f"{(stamps[0].date().isoformat() if stamps else '--'):<11} "
            f"{(stamps[-1].date().isoformat() if stamps else '--'):<11}  "
            f"{', '.join(methods)}"
        )
    typer.secho(
        "\nName a plug to see its full history: gasperm summarize <plug>",
        fg=typer.colors.CYAN,
    )


def _print_sample_summary(
    report, runs_dir: Path, superseded, pressure_unit: str = "atm"
) -> None:
    """The report. Identity, then the runs, then the result, then the gaps."""
    typer.secho(f"\n{report.sample_id}", bold=True)

    identity = []
    if report.length_cm and report.diameter_cm:
        identity.append(f"{report.length_cm:.3f} x {report.diameter_cm:.3f} cm")
    if report.porosity_fraction is not None:
        method = f" ({report.porosity_method})" if report.porosity_method else ""
        identity.append(f"porosity {report.porosity_fraction:.4g}{method}")
    for label, value in (
        ("lithology", report.lithology), ("formation", report.formation),
        ("well", report.well),
    ):
        if value:
            identity.append(f"{label} {value}")
    if report.depth is not None:
        identity.append(f"depth {report.depth:g} {report.depth_unit}".strip())
    if identity:
        typer.secho("  " + "   ".join(identity), fg=typer.colors.CYAN)
    if report.description:
        typer.secho(f"  {report.description}", fg=typer.colors.CYAN)

    span = ""
    if report.first_run and report.last_run:
        span = (
            f"   {report.first_run.date().isoformat()} to "
            f"{report.last_run.date().isoformat()}"
        )
    typer.secho(
        f"  {report.run_count} confirmed run(s)"
        + (f", {len(report.excluded)} not" if report.excluded else "")
        + (f", {len(report.leak_tests)} leak test(s)" if report.leak_tests else "")
        + span,
        fg=typer.colors.CYAN,
    )

    rows = (*report.measurements, *report.excluded, *report.leak_tests)
    if rows:
        # dP0 belongs to pulse decay alone, so a plug measured only in steady
        # state gets no permanently empty column -- the same rule the live plot
        # follows when it drops the flow panel from a pulse run.
        with_pulse = any(line.method == "pulse_decay" for line in rows)
        typer.secho("\n  Runs", bold=True, nl=False)
        # The units, once, rather than repeated in seven column headings: three
        # pressures share one unit, and a heading wide enough to carry it would
        # push the table past the width of a terminal.
        caption = f"   pressures in {pressure_unit}, permeability in mD"
        if with_pulse:
            # P_in and P_out mean something different on a pulse row, and an
            # operator setting the rig up again is reading exactly those cells.
            caption += ";  pulse rows: P at the pulse"
        typer.secho(caption, fg=typer.colors.BRIGHT_BLACK)
        header = (
            f"    {'run':<24} {'date':<11} {'method':<12} "
            f"{'P_in':>9} {'P_out':>9} {'P_mean':>9} "
        )
        if with_pulse:
            header += f"{'dP0':>9} "
        header += f"{'k':>9} {'U(k)':>9}  {'meter':<11}"
        typer.echo(header)
        for line in rows:
            _print_run_line(line, pressure_unit, with_pulse)

    if report.klinkenberg is not None:
        fit = report.klinkenberg
        typer.secho("\n  Klinkenberg correction", bold=True)
        k_l = units.darcy_to(fit.liquid_permeability_darcy, "mD")
        expanded = fit.liquid_permeability_expanded_uncertainty_darcy
        text = f"    k_L = {k_l:.6g} mD"
        if expanded is not None:
            text += (
                f" +/- {units.darcy_to(expanded, 'mD'):.4g}"
                f" (k = {fit.coverage_factor:.2f})"
            )
        typer.echo(text)
        typer.echo(
            f"    b   = {fit.slippage_factor_atm:.4g} atm"
            f"    R^2 = {fit.r_squared:.4f}"
            f"    {fit.point_count} points"
            + ("   weighted" if fit.weighted else "")
        )
        for warning in fit.warnings:
            typer.secho(f"    ! {warning}", fg=typer.colors.YELLOW)

    for record, reason in superseded:
        typer.secho(f"\n  {record.name}: {reason}", fg=typer.colors.BRIGHT_BLACK)

    if report.findings:
        typer.secho("\n  Findings", bold=True)
        for finding in report.findings:
            typer.secho(f"    - {finding}", fg=typer.colors.YELLOW)


def _pressure_cell(value_atm: float | None, unit: str) -> str:
    """One pressure, in the operator's unit rather than the internal atm."""
    if value_atm is None:
        return "--"
    return f"{units.from_atm(value_atm, unit):.5g}"


def _print_run_line(line, pressure_unit: str, with_pulse: bool) -> None:
    permeability = (
        f"{units.darcy_to(line.permeability_darcy, 'mD'):.5g}"
        if line.permeability_darcy is not None
        else "--"
    )
    expanded = (
        f"{units.darcy_to(line.expanded_uncertainty_darcy, 'mD'):.4g}"
        if line.expanded_uncertainty_darcy is not None
        else "--"
    )
    if line.purpose == "leak_test":
        colour, tail = typer.colors.BRIGHT_BLACK, "  LEAK TEST"
    elif not line.confirmed:
        colour, tail = typer.colors.YELLOW, f"  {line.excluded_reason}"
    else:
        colour, tail = None, ""
    text = (
        f"    {line.name[:24]:<24} "
        f"{(line.started_at.date().isoformat() if line.started_at else '--'):<11} "
        f"{line.method[:12]:<12} "
        # Per method: the measured means for steady state, the pressures at the
        # pulse for pulse decay. See RunLine.reported_inlet_pressure_atm.
        f"{_pressure_cell(line.reported_inlet_pressure_atm, pressure_unit):>9} "
        f"{_pressure_cell(line.reported_downstream_pressure_atm, pressure_unit):>9} "
        f"{_pressure_cell(line.mean_pressure_atm, pressure_unit):>9} "
    )
    if with_pulse:
        # Blank rather than "--" on a steady-state row: there is no pulse to
        # report, which is different from a pulse whose amplitude went missing.
        cell = (
            _pressure_cell(line.pulse_amplitude_atm, pressure_unit)
            if line.method == "pulse_decay"
            else ""
        )
        text += f"{cell:>9} "
    text += f"{permeability:>9} {expanded:>9}  {(line.flowmeter or '--')[:11]:<11}{tail}"
    typer.secho(text, fg=colour)


# --------------------------------------------------------------------------
# reprocess
# --------------------------------------------------------------------------


def _stored_config(record) -> GaspermConfig:
    """The configuration a run was recorded under, from its own sidecar.

    The snapshot rather than the current files: reprocessing has to start from
    what the run actually used, or the "before" half of the comparison would be
    a result nobody ever produced.
    """
    from gasperm.storage import read_run_metadata

    if record.metadata_path is None:
        raise ConfigError(
            f"{record.name} has no metadata sidecar, so the configuration it ran "
            "under is unknown and it cannot be reprocessed."
        )
    stored = read_run_metadata(record.metadata_path)
    snapshot = stored.get("config") if isinstance(stored, dict) else None
    if not isinstance(snapshot, dict):
        raise ConfigError(
            f"{record.name} has no stored config snapshot. It was recorded before "
            "runs became self-describing, so there is nothing to reprocess against."
        )
    try:
        return GaspermConfig.model_validate(snapshot)
    except Exception as exc:  # noqa: BLE001 - pydantic's own hierarchy
        raise ConfigError(
            f"{record.name}: its stored config no longer validates ({exc}). The "
            "schema has moved on since it was recorded."
        ) from exc


@app.command("reprocess")
def reprocess_command(
    targets: Optional[list[str]] = typer.Argument(
        None, metavar="[RUN...]",
        help="Run directories to re-derive. Omit and use --sample to take every "
             "run recorded for one plug.",
    ),
    sample: Optional[str] = typer.Option(
        None, "--sample", "-s",
        help="Re-derive every run for this plug -- the usual case, since a "
             "corrected uncertainty applies to a whole campaign, not one run.",
    ),
    set_values: Optional[list[str]] = typer.Option(
        None, "--set", metavar="KEY=VALUE",
        help="Override a config field, e.g. --set sample.porosity_uncertainty=0.005. "
             "Repeatable. Applied to each run's own stored snapshot.",
    ),
    from_config: bool = typer.Option(
        False, "--from-config",
        help="Start from the config files currently on disk instead of each run's "
             "stored snapshot. For when the rig file itself has been corrected.",
    ),
    sample_file: Optional[Path] = typer.Option(
        None, "--sample-file",
        help="Replace the stored sample section with this file, e.g. after "
             "re-measuring a plug's porosity.",
    ),
    write: bool = typer.Option(
        False, "--write",
        help="Write each re-derived run to a NEW run directory beside the "
             "original, which is never modified. Without this, nothing is "
             "written and the change is only reported.",
    ),
    config_dir: Path = typer.Option(
        Path("."), "--config-dir", "-c", help="Rig folder holding run.yaml."
    ),
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Where the runs are. Defaults to run.yaml's output_dir."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Re-derive stored runs from their raw voltages under a changed config.

    Every run keeps its raw voltages and probe readings, so a measurement can be
    **re-costed** -- a calibration certificate arrives, a porosity is finally
    measured with a stated uncertainty -- without repeating an experiment that
    may have taken hours.

    Changes fall into three classes, and the command says which you made.
    Touching an *uncertainty* field moves ``U(k)`` and leaves ``k`` exactly
    where it was; touching a *result* field (geometry, calibration, gas, vessel
    volumes) moves ``k`` itself, which is a correction rather than a re-costing;
    metadata moves neither. The prediction is checked against the arithmetic,
    and a disagreement is reported rather than trusted.

    Reports only, unless --write. The raw record is never modified.
    """
    from gasperm.reprocess import (
        ReprocessError,
        ReprocessResult,
        diff_configs,
        reprocess_run,
    )
    from gasperm.storage import (
        _record_from_directory,
        find_runs,
        runs_for_sample,
        summary_from_run,
    )

    _configure_logging(verbose)

    records = []
    if targets:
        for target in targets:
            path = Path(target)
            if not path.is_dir():
                _fail(f"No such run directory: {target}")
            records.append(_record_from_directory(path))
    if sample is not None:
        resolved_runs = _resolve_runs_dir(runs_dir, config_dir)
        sample_id = _resolve_sample_id(sample)
        found = runs_for_sample(find_runs(resolved_runs), sample_id)
        if not found:
            _fail(f"No runs found for {sample_id!r} under {resolved_runs}.")
        records.extend(found)
    if not records:
        _fail(
            "Nothing to reprocess. Give one or more run directories, or --sample "
            "to take every run for a plug."
        )
        return

    overrides = list(set_values or [])
    replacement_sample = None
    if sample_file is not None:
        from gasperm.config import load_sample_config

        try:
            replacement_sample = load_sample_config(sample_file)
        except ConfigError as exc:
            _fail(str(exc))
            return

    on_disk = None
    if from_config:
        on_disk = _load_or_fail(config_dir, None, None, None)

    results: list[ReprocessResult] = []
    failures: list[tuple[str, str]] = []
    for record in records:
        try:
            base = on_disk if on_disk is not None else _stored_config(record)
        except ConfigError as exc:
            failures.append((record.name, str(exc)))
            continue

        before_config = config_to_dict(base)
        data = config_to_dict(base)
        if replacement_sample is not None:
            data["sample"] = replacement_sample.model_dump(mode="json", by_alias=True)
        try:
            for assignment in overrides:
                key, _, raw = assignment.partition("=")
                if not raw:
                    raise ConfigError(
                        f"--set {assignment!r} should be KEY=VALUE, e.g. "
                        "sample.porosity_uncertainty=0.005"
                    )
                _set_dotted(data, key.strip(), raw)
            updated = GaspermConfig.model_validate(data)
        except (ConfigError, ValueError) as exc:
            _fail(str(exc))
            return

        changes = tuple(diff_configs(before_config, config_to_dict(updated)))
        original = summary_from_run(record.directory)
        try:
            redone = reprocess_run(
                record.directory, updated,
                started_at=original.started_at if original else None,
                ended_at=original.ended_at if original else None,
            )
        except (ReprocessError, ValueError, KeyError) as exc:
            failures.append((record.name, str(exc)))
            continue

        results.append(
            ReprocessResult(
                directory=record.directory, before=original, after=redone,
                changes=changes,
            )
        )

    for name, reason in failures:
        typer.secho(f"skipped {name}: {reason}", fg=typer.colors.YELLOW, err=True)
    if not results:
        _fail("Nothing could be reprocessed.")
        return

    _print_reprocess(results, wrote=write)

    if write:
        for result in results:
            base = _stored_config(_record_from_directory(result.directory))
            data = config_to_dict(base)
            if replacement_sample is not None:
                data["sample"] = replacement_sample.model_dump(mode="json", by_alias=True)
            for assignment in overrides:
                key, _, raw = assignment.partition("=")
                _set_dotted(data, key.strip(), raw)
            updated = GaspermConfig.model_validate(data)
            updated.config_dir = base.config_dir
            saved = _write_reprocessed(result, updated)
            typer.secho(f"  wrote {saved}", fg=typer.colors.GREEN)


def _write_reprocessed(result, config: GaspermConfig) -> Path:
    """Copy the raw record into a new run directory beside the original.

    A new directory, never an edit in place: the original is the record of a
    measurement, and a tool that silently rewrote one would make every report
    already issued from it unreproducible.

    The name is the original's plus ``_reprocessed`` rather than a fresh
    timestamp. A derived run must be obvious at a glance in a directory
    listing, and stamping it with the moment of *recomputation* would put it in
    the wrong place chronologically -- it describes an experiment that happened
    when its parent did.

    The copy carries the same raw CSV, so it is as self-describing as its
    parent, plus a ``derived_from`` block naming what it came from and exactly
    what was changed.
    """
    import shutil
    from datetime import datetime, timezone

    import yaml

    from gasperm.config import experiment_metadata
    from gasperm.storage import (
        METADATA_FILENAME,
        READINGS_FILENAME,
        _unique_directory,
    )

    target = _unique_directory(
        result.directory.with_name(f"{result.directory.name}_reprocessed")
    )
    target.mkdir(parents=True, exist_ok=True)
    source_csv = result.directory / READINGS_FILENAME
    shutil.copy2(source_csv, target / READINGS_FILENAME)

    payload = {
        "gasperm_run": {
            "started_at": result.after.started_at.isoformat(),
            "readings_csv": READINGS_FILENAME,
            "rows": result.after.sample_count,
        },
        "metadata": experiment_metadata(config).model_dump(mode="json"),
        "config": config_to_dict(config),
        "summary": result.after.model_dump(mode="json"),
        "derived_from": {
            "run": result.directory.name,
            "path": str(result.directory),
            "reprocessed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "changes": [
                {
                    "field": change.key,
                    "before": change.before,
                    "after": change.after,
                    "predicted": change.predicted,
                }
                for change in result.changes
            ],
            "permeability_moved": result.permeability_moved,
            "uncertainty_moved": result.uncertainty_moved,
        },
    }
    (target / METADATA_FILENAME).write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return target


def _print_reprocess(results, *, wrote: bool) -> None:
    """What changed, grouped by whether it moved the answer or only its cost."""
    from gasperm.reprocess import summarise_changes

    changes = results[0].changes
    typer.secho(f"\nReprocessing {len(results)} run(s) from raw voltages", bold=True)
    if not changes:
        typer.secho(
            "  No configuration change -- this is a re-derivation check, and every "
            "value below should be unchanged.",
            fg=typer.colors.CYAN,
        )
    grouped = summarise_changes(changes)
    labels = {
        "result": ("moves k -- this is a CORRECTION, not a re-costing", typer.colors.RED),
        "uncertainty": ("moves U(k) only -- k is untouched", typer.colors.CYAN),
        "metadata": ("moves neither", typer.colors.BRIGHT_BLACK),
    }
    for kind in ("result", "uncertainty", "metadata"):
        entries = grouped.get(kind, [])
        if not entries:
            continue
        text, color = labels[kind]
        typer.secho(f"  {kind}: {text}", fg=color)
        for change in entries:
            typer.echo(f"    {change.describe()}")

    typer.echo("")
    typer.echo(
        f"  {'run':<34} {'k (mD)':>22}  {'U(k) (mD)':>22}   verdict"
    )
    for result in results:
        before, after = result.before, result.after
        k_before = (
            f"{units.darcy_to(before.permeability_darcy, 'mD'):.6g}" if before else "--"
        )
        k_after = f"{units.darcy_to(after.permeability_darcy, 'mD'):.6g}"
        u_before = _expanded_text(before)
        u_after = _expanded_text(after)

        if result.permeability_moved:
            verdict, color = (
                f"k moved {(result.permeability_ratio - 1.0) * 100.0:+.3f}%",
                typer.colors.RED,
            )
        elif result.uncertainty_moved:
            verdict, color = "k unchanged, U re-costed", typer.colors.GREEN
        else:
            verdict, color = "unchanged", typer.colors.BRIGHT_BLACK
        typer.secho(
            f"  {result.directory.name[:34]:<34} "
            f"{k_before:>10} -> {k_after:>10}  "
            f"{u_before:>10} -> {u_after:>10}   {verdict}",
            fg=color,
        )
        if result.surprise:
            typer.secho(f"    ! {result.surprise}", fg=typer.colors.RED)
        elif result.note:
            typer.secho(f"    {result.note}", fg=typer.colors.BRIGHT_BLACK)

    if not wrote:
        typer.secho(
            "\n  Nothing was written. Pass --write to save each re-derived run to a "
            "new directory beside its original.",
            fg=typer.colors.YELLOW,
        )


def _expanded_text(summary) -> str:
    if summary is None or summary.uncertainty is None:
        return "--"
    return f"{units.darcy_to(summary.uncertainty.expanded_uncertainty_darcy, 'mD'):.4g}"


# --------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------


def _looks_like_a_run(path: Path) -> bool:
    from gasperm.storage import METADATA_FILENAME, READINGS_FILENAME

    return (path / METADATA_FILENAME).is_file() or (path / READINGS_FILENAME).is_file()


def _records_for_selector(selector: str, runs_dir: Path) -> tuple[list, str]:
    """Runs named by one side of a comparison, plus a label for it.

    A selector is a run directory, a directory of runs, or a sample id -- the
    same three things an operator already has in hand, rather than a fourth
    syntax to learn.
    """
    from gasperm.storage import (
        _record_from_directory,
        drop_superseded,
        find_runs,
        runs_for_sample,
    )

    path = Path(selector)
    if path.is_dir() and _looks_like_a_run(path):
        return [_record_from_directory(path)], path.name
    if path.is_dir():
        records = find_runs(path)
        if not records:
            _fail(f"No runs found under {path}.")
        return records, path.name

    sample_id = _resolve_sample_id(selector)
    records, _ = drop_superseded(runs_for_sample(find_runs(runs_dir), sample_id))
    if not records:
        _fail(
            f"No runs found for {sample_id!r} under {runs_dir}. Use --runs-dir if they "
            "are somewhere else."
        )
    return records, sample_id


def _split_records(records: list, split: str) -> tuple[list, list]:
    """Divide one plug's runs at an instant: everything before it, everything after.

    The natural discriminator for a before/after study when the two campaigns
    are the same plug under the same id -- which is the ordinary case, since a
    plug does not get a new name because it spent a month in hydrogen.
    """
    from datetime import datetime, timezone

    text = split.strip()
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        _fail(
            f"--split {split!r} is not a date or timestamp. Use an ISO form such as "
            "2026-06-01 or 2026-06-01T12:00:00."
        )
        raise
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)

    undated = [r for r in records if r.started_at is None]
    if undated:
        _fail(
            f"{len(undated)} run(s) have no recorded start time, so --split cannot "
            "place them: " + ", ".join(r.name for r in undated[:5])
        )
    before = [r for r in records if r.started_at < moment]
    after = [r for r in records if r.started_at >= moment]
    if not before or not after:
        _fail(
            f"--split {split} puts {len(before)} run(s) before and {len(after)} after. "
            "Both sides need at least one."
        )
    return before, after


def _build_group(
    records: list,
    label: str,
    *,
    window: float | None,
    allow_unsteady: bool,
    allow_mixed_methods: bool,
):
    """Assemble one side: its confirmed runs, their budgets, and a fit if possible."""
    from gasperm.comparison import MeasurementGroup
    from gasperm.klinkenberg import fit_klinkenberg
    from pydantic import ValidationError

    from gasperm.config.sample import SampleConfig
    from gasperm.storage import read_run_metadata, summary_from_run

    usable = [r for r in records if r.purpose != "leak_test"]
    dropped_leak = len(records) - len(usable)

    summaries, conventions, skipped, kept = [], [], [], []
    porosity = porosity_uncertainty = None
    for record in usable:
        summary = summary_from_run(record.directory)
        if summary is None:
            skipped.append((record.name, "no stored summary"))
            continue
        if not summary.measurement_confirmed and not allow_unsteady:
            skipped.append((record.name, "never confirmed a measurement"))
            continue
        summaries.append(summary)
        kept.append(record)
        conventions.append(record.downstream_convention or "measured")
        if porosity is None and summary.metadata is not None:
            porosity = summary.metadata.porosity_fraction
        if porosity_uncertainty is None and record.metadata_path is not None:
            stored = read_run_metadata(record.metadata_path)
            sample = (stored.get("config") or {}).get("sample") or {}
            # Through the schema rather than off the raw key: the stored value
            # is in that run's own porosity_unit, which may be percentage
            # points, and the comparison works in fractions throughout.
            try:
                porosity_uncertainty = SampleConfig.model_validate(
                    sample
                ).porosity_uncertainty_fraction
            except ValidationError:
                porosity_uncertainty = sample.get("porosity_uncertainty")

    if not summaries:
        _fail(
            f"{label}: none of its {len(records)} run(s) produced a usable "
            "measurement. Pass --allow-unsteady to include runs that never "
            "confirmed one."
        )

    # A fit needs spread in mean pressure; a single-pressure campaign is a
    # perfectly good thing to compare, just at the k_g level rather than k_L.
    fit = None
    if len({round(s.mean_pressure_atm, 6) for s in summaries}) >= 2:
        points, _ = _discover_points(
            kept, window=window, allow_unsteady=allow_unsteady
        )
        if len(points) >= 2:
            try:
                fit = fit_klinkenberg(points, allow_mixed_methods=allow_mixed_methods)
            except ValueError as exc:
                logger.debug("No Klinkenberg fit for %s: %s", label, exc)

    group = MeasurementGroup(
        label=label,
        sample_id=summaries[0].sample_id,
        summaries=tuple(summaries),
        klinkenberg=fit,
        porosity_fraction=porosity,
        porosity_uncertainty=porosity_uncertainty,
        downstream_conventions=tuple(conventions),
    )
    return group, skipped, dropped_leak


@app.command("compare")
def compare_command(
    before: str = typer.Argument(
        ..., metavar="BEFORE",
        help="Baseline: a sample id, a run directory, or a directory of runs.",
    ),
    after: Optional[str] = typer.Argument(
        None, metavar="AFTER",
        help="What to compare against it. Omit and pass --split to divide one "
             "plug's own runs into a before and an after.",
    ),
    split: Optional[str] = typer.Option(
        None, "--split", metavar="DATE",
        help="Split BEFORE's runs at this instant instead of taking a second "
             "selector, e.g. --split 2026-06-01. For a plug measured, treated, "
             "and measured again under the same id.",
    ),
    config_dir: Path = typer.Option(
        Path("."), "--config-dir", "-c", help="Rig folder holding run.yaml."
    ),
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Where the runs are. Defaults to run.yaml's output_dir."
    ),
    label_before: Optional[str] = typer.Option(
        None, "--label-before", help="Name for the baseline in the report."
    ),
    label_after: Optional[str] = typer.Option(
        None, "--label-after", help="Name for the comparison in the report."
    ),
    paired: Optional[bool] = typer.Option(
        None, "--paired/--unpaired",
        help="Force the same-plug treatment on or off. By default it is decided "
             "from the sample ids, which is right unless you have renamed a plug "
             "between campaigns.",
    ),
    allow_mismatched_conditions: bool = typer.Option(
        False, "--allow-mismatched-conditions",
        help="Report a comparison whose campaigns were not run alike. The "
             "mismatch is still shown, and the affected uncertainties stop "
             "cancelling.",
    ),
    allow_mixed_methods: bool = typer.Option(
        False, "--allow-mixed-methods",
        help="Permit each side's own Klinkenberg fit to mix methods.",
    ),
    allow_unsteady: bool = typer.Option(
        False, "--allow-unsteady", help="Include runs that never confirmed a measurement."
    ),
    coverage: float = typer.Option(
        0.95, "--coverage", min=0.5, max=0.999,
        help="Level of confidence for every expanded uncertainty.",
    ),
    pressure_tolerance: float = typer.Option(
        0.05, "--pressure-tolerance", min=0.0, max=1.0,
        help="How close two runs' mean pressures must be to be treated as the "
             "same point, as a fraction.",
    ),
    window: Optional[float] = typer.Option(
        None, "--window", metavar="SECONDS", help="Override the stored averaging window."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write the full comparison to a YAML file."
    ),
    plot: bool = typer.Option(False, "--plot", help="Plot both fits and the change."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Compare two sets of runs and report every difference with its uncertainty.

    The measurand is the **change**, not either value, and that is what makes
    this worth a command of its own. Errors common to both measurements -- the
    same plug's geometry, the same transducer, the same meter, the same
    viscosity model -- move both results the same way and cancel out of their
    ratio. What survives is the scatter, which is usually far smaller. A rig
    reporting 20% on each of two permeabilities can still resolve a 5% change
    between them.

    Every cancellation is itemised, because a claim that an uncertainty went
    away is the load-bearing part of the result.
    """
    from gasperm.comparison import ComparisonError, compare_groups
    from gasperm.storage import write_comparison_result

    _configure_logging(verbose)
    resolved_runs_dir = _resolve_runs_dir(runs_dir, config_dir)

    if split is not None and after is not None:
        _fail(
            "Pass either a second selector or --split, not both: they are two ways "
            "of saying which runs form the 'after' side."
        )
        return
    if split is None and after is None:
        _fail(
            "Nothing to compare against. Give a second selector "
            "('gasperm compare core-041 core-042'), or --split DATE to divide one "
            "plug's runs into a before and an after."
        )
        return

    if split is not None:
        records, name = _records_for_selector(before, resolved_runs_dir)
        before_records, after_records = _split_records(records, split)
        before_name, after_name = f"{name} before {split}", f"{name} from {split}"
    else:
        before_records, before_name = _records_for_selector(before, resolved_runs_dir)
        after_records, after_name = _records_for_selector(after, resolved_runs_dir)

    before_group, before_skipped, before_leaks = _build_group(
        before_records, label_before or before_name,
        window=window, allow_unsteady=allow_unsteady,
        allow_mixed_methods=allow_mixed_methods,
    )
    after_group, after_skipped, after_leaks = _build_group(
        after_records, label_after or after_name,
        window=window, allow_unsteady=allow_unsteady,
        allow_mixed_methods=allow_mixed_methods,
    )

    for label, skipped, leaks in (
        (before_group.label, before_skipped, before_leaks),
        (after_group.label, after_skipped, after_leaks),
    ):
        if leaks:
            typer.secho(
                f"{label}: {leaks} leak test(s) excluded -- a leak belongs to the rig, "
                "not to any plug.",
                fg=typer.colors.BRIGHT_BLACK,
            )
        for name, reason in skipped:
            typer.secho(f"{label}: skipped {name} ({reason})", fg=typer.colors.YELLOW)

    try:
        result = compare_groups(
            before_group, after_group,
            coverage_probability=coverage,
            paired=paired,
            allow_mismatched_conditions=allow_mismatched_conditions,
            pressure_tolerance=pressure_tolerance,
        )
    except ComparisonError as exc:
        _fail(str(exc))
        return

    _print_comparison(result)

    if output is not None:
        saved = write_comparison_result(result, output)
        typer.secho(f"\nComparison written to {saved}", fg=typer.colors.GREEN)

    if plot:
        try:
            from gasperm.plotting import PlottingUnavailable, plot_comparison

            plot_config = _plot_config_for(config_dir)
            path = (output.with_suffix(".png") if output else None)
            saved_plot = plot_comparison(
                result, before_group.klinkenberg, after_group.klinkenberg, path=path,
                show=path is None, plot_config=plot_config,
            )
            if saved_plot is not None:
                typer.secho(f"Plot written to {saved_plot}", fg=typer.colors.GREEN)
        except PlottingUnavailable as exc:
            typer.secho(f"warning: {exc}", fg=typer.colors.YELLOW)

    # Exit 2 when nothing measurable changed: a script driving a screening study
    # can then branch on "this plug moved" without parsing the report.
    headline = result.change("k_L") or result.change("k_g")
    if headline is not None and not headline.significant:
        raise typer.Exit(code=2)


def _print_comparison(result) -> None:
    """The report. Ordered so the answer comes first and its basis follows."""
    typer.secho(
        f"\nComparison   {result.before.label}   ->   {result.after.label}", bold=True
    )
    kind = "PAIRED (same plug)" if result.paired else "UNPAIRED (different plugs)"
    typer.secho(
        f"  {kind}   coverage {result.coverage_probability:.0%}", fg=typer.colors.CYAN
    )

    for side, tag in ((result.before, "before"), (result.after, "after ")):
        detail = f"{side.run_count} run(s)"
        if side.mean_pressures_atm:
            detail += "   P_mean " + ", ".join(
                f"{p:.3g}" for p in sorted(side.mean_pressures_atm)
            ) + " atm"
        if side.liquid_permeability_darcy is not None:
            detail += (
                f"   k_L {units.darcy_to(side.liquid_permeability_darcy, 'mD'):.4g} mD"
                f"   b {side.slippage_factor_atm:.3g} atm   R2 {side.r_squared:.4f}"
            )
        typer.echo(f"  {tag}  {side.sample_id or '(unknown)'}   {detail}")

    typer.secho("\n  Conditions", bold=True)
    for check in result.conditions:
        mark = "match" if check.matched else ("BLOCKING" if check.blocking else "DIFFER")
        color = (
            typer.colors.GREEN if check.matched
            else (typer.colors.RED if check.blocking else typer.colors.YELLOW)
        )
        typer.secho(
            f"    {check.label:<24} {check.before[:22]:<22} -> "
            f"{check.after[:22]:<22} {mark}",
            fg=color,
        )
        if check.advice:
            typer.secho(f"      {check.advice}", fg=typer.colors.BRIGHT_BLACK)

    typer.secho("\n  Changes", bold=True)
    for change in result.changes:
        _print_change(change)

    cancelled = [p for p in result.component_pairings if p.shared]
    retained = [p for p in result.component_pairings if not p.shared]
    if cancelled:
        typer.secho("\n  What cancelled between the two measurements", bold=True)
        for pairing in sorted(cancelled, key=lambda p: -p.cancelled_fraction):
            typer.secho(
                f"    {pairing.symbol:<8} {pairing.name[:26]:<26} "
                f"{pairing.cancelled_fraction:>6.1%} removed   {pairing.reason}",
                fg=typer.colors.GREEN,
            )
    if retained:
        typer.secho("\n  What did not, and therefore sets the detection limit", bold=True)
        for pairing in sorted(retained, key=lambda p: -p.variance_contribution):
            typer.echo(
                f"    {pairing.symbol:<8} {pairing.name[:26]:<26} "
                f"{math.sqrt(pairing.variance_contribution):>6.2%} of the ratio   "
                f"{pairing.reason}"
            )

    for warning in result.warnings:
        typer.secho(f"\n  note: {warning}", fg=typer.colors.YELLOW)


def _print_change(change) -> None:
    scale, unit = (
        (units.darcy_to(1.0, "mD"), "mD") if change.unit == "darcy" else (1.0, change.unit)
    )
    color = typer.colors.GREEN if change.significant else typer.colors.YELLOW
    typer.secho(f"    {change.symbol}   {change.name}", bold=True)
    typer.echo(
        f"        {change.before * scale:.6g} -> {change.after * scale:.6g} {unit}"
        f"   (delta {change.difference * scale:+.4g})"
    )
    typer.secho(f"        {change.verdict}", fg=color)
    if math.isfinite(change.minimum_detectable_percent):
        dof = change.effective_degrees_of_freedom
        dof_text = "inf" if not math.isfinite(dof) else f"{dof:.1f}"
        # Both uncertainties, because they can move in opposite directions: an
        # input that stops cancelling raises u_c but, being Type B, also raises
        # v_eff -- which lowers k and can shrink U. Showing only U would make
        # that read as an improvement.
        typer.secho(
            f"        u_c = {change.relative_standard_uncertainty * 100.0:.2f}%,"
            f" k = {change.coverage_factor:.2f} (v_eff = {dof_text});"
            f" smallest change this could resolve: "
            f"{change.minimum_detectable_percent:.2f}%",
            fg=typer.colors.BRIGHT_BLACK,
        )
    for note in change.notes:
        typer.secho(f"        {note}", fg=typer.colors.BRIGHT_BLACK)


# --------------------------------------------------------------------------
# klinkenberg
# --------------------------------------------------------------------------


def _resolve_sample_id(value: str) -> str:
    """A sample id from either a bare id or a path to a sample file.

    ``collect --sample`` takes a path, so an operator will reasonably paste the
    same value here. A value that was clearly *meant* as a path but does not
    exist is an error rather than a literal id -- otherwise a typo'd path turns
    into a baffling "no runs found for samples/core-041.yaml".
    """
    from gasperm.config import load_sample_config

    candidate = Path(value)
    if candidate.is_file():
        try:
            return load_sample_config(candidate).id
        except ConfigError as exc:
            _fail(str(exc))
    looks_like_a_path = (
        "/" in value or "\\" in value or value.lower().endswith((".yaml", ".yml"))
    )
    if looks_like_a_path:
        _fail(f"No such sample file: {value}")
    return value


def _plot_config_for(config_dir: Path):
    """``run.yaml``'s plot section, for window placement on a chosen monitor.

    ``klinkenberg`` and ``compare`` work over stored runs and do not otherwise
    need a rig config, so this is best-effort: no run.yaml means no placement,
    which is the same as not asking for any.
    """
    from gasperm.config import RUN_FILENAME, load_run_config

    try:
        return load_run_config(config_dir / RUN_FILENAME).plot
    except ConfigError:
        return None


def _resolve_runs_dir(runs_dir: Path | None, config_dir: Path) -> Path:
    """Where a plug's runs live: an explicit path, or run.yaml's output_dir."""
    from gasperm.config import RUN_FILENAME, load_run_config

    if runs_dir is not None:
        return runs_dir

    run_file = config_dir / RUN_FILENAME
    try:
        run_config = load_run_config(run_file)
    except ConfigError as exc:
        _fail(
            f"{exc}\n\nCannot tell where the runs are. Pass --runs-dir <dir>, or "
            "--config-dir pointing at the rig folder that holds run.yaml."
        )
        raise  # unreachable
    output = Path(run_config.output_dir)
    return output if output.is_absolute() else config_dir / output


def _summary_pressure_unit(config_dir: Path) -> str:
    """The unit ``summarize`` reports pressures in.

    ``run.yaml``'s display unit, so a summary agrees with what ``collect``
    printed on the console and what the live plot showed for the same rig.
    Pressures are stored in atm because that is what the physics runs in, and
    a table that reported the internal unit would be asking the operator to
    convert numbers that the rest of the package already converts for them.

    Falls back to ``atm`` when there is no ``run.yaml`` to read -- a
    ``--runs-dir`` pointed at a folder of runs with no rig beside it. That is
    the stored unit itself, and the header names it, which beats picking a
    display unit nobody configured.
    """
    from gasperm.config import RUN_FILENAME, load_run_config

    try:
        return load_run_config(config_dir / RUN_FILENAME).display_pressure_unit
    except (ConfigError, OSError):
        return "atm"


def _discover_points(records, *, window, allow_unsteady):
    """Reduce discovered runs to points, skipping the ones that cannot be used.

    ``--sample`` means "every run for this plug", which will legitimately
    include aborted and exploratory ones. Failing the whole regression over a
    single bad run is exactly the friction this command removes, so they are
    skipped and reported. Nothing is silent.
    """
    from gasperm.storage import point_from_run

    points, skipped = [], []
    for record in records:
        try:
            points.append(
                point_from_run(
                    record.directory,
                    averaging_window_s=window,
                    allow_unsteady=allow_unsteady,
                )
            )
        except (ValueError, FileNotFoundError) as exc:
            skipped.append((record, _short_reason(exc, record)))
    return points, skipped


def _short_reason(exc: Exception, record) -> str:
    """The gist of a skip, without the path the operator can already see.

    These messages lead with the CSV's full path, which is both redundant in a
    listing keyed by run name and full of dots -- so naive sentence-splitting
    truncates mid-path instead of mid-sentence.
    """
    reason = str(exc)
    for noise in (str(record.readings_csv), str(record.directory)):
        reason = reason.replace(noise, "")
    reason = reason.strip().lstrip(":").strip()
    # Keep the claim, drop the advice that follows it.
    for separator in (", so ", ". "):
        reason = reason.split(separator)[0]
    return reason.strip() or "could not be used"


def _print_discovered_runs(sample_id, runs_dir, records, skipped) -> None:
    """Say what was found, and what was left out and why."""
    from gasperm.storage import describe_convention

    reasons = {record.name: reason for record, reason in skipped}
    # Show P2's provenance only when it varies -- a uniform column is noise,
    # but a mismatch should be visible here rather than only in the refusal.
    conventions = {r.downstream_convention for r in records if r.downstream_convention}
    show_convention = len(conventions) > 1
    typer.secho(
        f"\nFound {len(records)} run{'s' if len(records) != 1 else ''} for "
        f"{sample_id} in {runs_dir}",
        bold=True,
    )
    for record in records:
        when = (
            record.started_at.strftime("%Y-%m-%d %H:%MZ") if record.started_at else "unknown"
        )
        line = f"  {record.name:<32} {when}"
        if show_convention:
            line += f"   P2 {describe_convention(record.downstream_convention)}"
        if record.name in reasons:
            line += f"   skipped: {reasons[record.name]}"
        elif not record.has_summary:
            line += "   (no summary recorded)"
        typer.echo(line)
    if skipped:
        typer.secho(
            f"{len(skipped)} run{'s' if len(skipped) != 1 else ''} skipped. "
            "Pass --allow-unsteady to include them.",
            fg=typer.colors.YELLOW,
        )


def _default_klinkenberg_output(base: Path, sample_ids: set[str]) -> Path:
    """Results path for a regression, named per plug.

    Without the sample id in the name, measuring a second plug silently
    overwrites the first plug's results.
    """
    from gasperm.storage import safe_sample_id

    if len(sample_ids) == 1:
        return base / f"klinkenberg_{safe_sample_id(next(iter(sample_ids)))}.yaml"
    return base / "klinkenberg.yaml"


@app.command("klinkenberg")
def klinkenberg_command(
    runs: list[Path] = typer.Argument(
        None, metavar="[RUN]...",
        help="Run directories (or readings.csv paths) from previous collect runs.",
    ),
    sample: Optional[str] = typer.Option(
        None, "--sample", metavar="ID|FILE",
        help=(
            "Regress every recorded run for this core plug. Takes a sample id "
            "(core-041) or a sample file (samples/core-041.yaml)."
        ),
    ),
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir",
        help="Where the run directories are. Defaults to run.yaml's output_dir.",
    ),
    config_dir: Path = typer.Option(
        Path("."), "--config-dir", "-c",
        help="Rig folder holding run.yaml. Only read when --sample is used.",
    ),
    csv_path: Optional[Path] = typer.Option(
        None, "--csv",
        help="Standalone CSV of mean-pressure / apparent-permeability pairs instead of runs.",
    ),
    csv_pressure_unit: Optional[str] = typer.Option(
        None, "--csv-pressure-unit", help="Unit of the CSV's mean-pressure column."
    ),
    csv_permeability_unit: Optional[str] = typer.Option(
        None, "--csv-permeability-unit", help="Unit of the CSV's permeability column."
    ),
    window: Optional[float] = typer.Option(
        None, "--window", "-w", metavar="SECONDS",
        help="Override each run's steady-state detection window.",
    ),
    allow_unsteady: bool = typer.Option(
        False, "--allow-unsteady",
        help="Accept runs that never reached steady state. They do not measure the sample.",
    ),
    allow_mixed_samples: bool = typer.Option(
        False, "--allow-mixed-samples",
        help="Permit runs from more than one core plug in a single regression.",
    ),
    allow_mixed_conditions: bool = typer.Option(
        False, "--allow-mixed-conditions",
        help="Permit runs that obtained P2 differently (measured vs supplied).",
    ),
    allow_mixed_methods: bool = typer.Option(
        False, "--allow-mixed-methods",
        help="Permit steady-state and pulse-decay runs in the same regression.",
    ),
    coverage: float = typer.Option(
        0.95, "--coverage", help="Level of confidence for the uncertainty on k_L."
    ),
    permeability_unit: str = typer.Option(
        "mD", "--unit", "-u", help="Unit for the reported permeabilities."
    ),
    pressure_unit: str = typer.Option(
        "atm", "--pressure-unit", help="Unit for the reported pressures and b."
    ),
    plot: bool = typer.Option(False, "--plot", help="Save a regression plot beside the results."),
    show: bool = typer.Option(False, "--show", help="Open the regression plot in a window."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write the results YAML here."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Recover liquid-equivalent permeability from runs at different mean pressures.

    Regresses apparent permeability against ``1 / P_mean``; the intercept is
    ``k_L`` and ``slope / intercept`` is the slippage factor ``b``. Each run is
    reduced over its detected steady-state window, and the fit is weighted by
    the runs' uncertainties when they carry them.
    """
    from gasperm.klinkenberg import MIN_POINTS, fit_klinkenberg, load_points_from_csv
    from gasperm.storage import (
        collect_points,
        drop_superseded,
        find_runs,
        runs_for_sample,
        write_klinkenberg_result,
    )

    _configure_logging(verbose)

    runs = [r for r in (runs or []) if r is not None]
    sources = sum(1 for source in (runs, csv_path, sample) if source)
    if sources == 0:
        _fail(
            "Nothing to regress. Pass --sample core-041 to use every recorded run for "
            "that plug, or name run directories explicitly, or --csv with a file of "
            "mean-pressure / apparent-permeability pairs."
        )
        return
    if sources > 1:
        _fail("Pass exactly one of --sample, run directories, or --csv.")
        return

    try:
        # Reject a bad display unit now, not after the regression has run.
        units.darcy_to(1.0, permeability_unit)
        units.from_atm(1.0, pressure_unit)
    except ValueError as exc:
        _fail(str(exc))
        return

    discovered_dir: Path | None = None
    try:
        if csv_path is not None:
            points = load_points_from_csv(
                csv_path,
                pressure_unit=csv_pressure_unit,
                permeability_unit=csv_permeability_unit,
            )
        elif sample is not None:
            sample_id = _resolve_sample_id(sample)
            discovered_dir = _resolve_runs_dir(runs_dir, config_dir)
            matching = runs_for_sample(find_runs(discovered_dir), sample_id)
            # A reprocessed run and its parent describe the SAME measurement.
            # Regressing both would use one experiment twice, and at the same
            # mean pressure it would also make the fit look better determined.
            matching, superseded = drop_superseded(matching)
            for record, reason in superseded:
                typer.secho(f"  {record.name}: {reason}", fg=typer.colors.BRIGHT_BLACK)
            if not matching:
                present = sorted(
                    {r.sample_id for r in find_runs(discovered_dir) if r.sample_id}
                )
                _fail(
                    f"No runs found for {sample_id!r} in {discovered_dir}."
                    + (
                        f" Plugs recorded there: {', '.join(present)}."
                        if present
                        else " That directory holds no runs yet."
                    )
                )
                return
            points, skipped = _discover_points(
                matching, window=window, allow_unsteady=allow_unsteady
            )
            _print_discovered_runs(sample_id, discovered_dir, matching, skipped)
            if len(points) < MIN_POINTS:
                _fail(
                    f"\nOnly {len(points)} of {len(matching)} runs could be used, and the "
                    f"regression needs at least {MIN_POINTS}."
                    + (" Pass --allow-unsteady to include the skipped ones." if skipped else "")
                )
                return
        else:
            points = collect_points(
                runs, averaging_window_s=window, allow_unsteady=allow_unsteady
            )
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))
        return

    try:
        result = fit_klinkenberg(
            points,
            coverage_probability=coverage,
            allow_mixed_samples=allow_mixed_samples,
            allow_mixed_conditions=allow_mixed_conditions,
            allow_mixed_methods=allow_mixed_methods,
        )
    except ValueError as exc:
        _fail(str(exc))
        return

    _print_klinkenberg(result, permeability_unit, pressure_unit)

    destination = output
    if destination is None and csv_path is None:
        base = discovered_dir if discovered_dir is not None else Path(runs[0]).parent
        destination = _default_klinkenberg_output(
            base, {p.sample_id for p in result.points if p.sample_id}
        )
    if destination is not None:
        written = write_klinkenberg_result(result, destination)
        typer.secho(f"\nResults written to {written}", fg=typer.colors.GREEN)

    if plot or show:
        from gasperm.plotting import PlottingUnavailable, plot_klinkenberg

        image_path = None
        if plot:
            base = destination if destination is not None else Path("klinkenberg.yaml")
            image_path = base.with_suffix(".png")
        plot_config = _plot_config_for(config_dir)
        try:
            saved = plot_klinkenberg(
                result, path=image_path, show=show,
                permeability_unit=permeability_unit, pressure_unit=pressure_unit,
                plot_config=plot_config,
            )
        except PlottingUnavailable as exc:
            typer.secho(f"warning: {exc}", fg=typer.colors.YELLOW)
        else:
            if saved is not None:
                typer.secho(f"Plot written to {saved}", fg=typer.colors.GREEN)


def _print_klinkenberg(result, permeability_unit: str, pressure_unit: str) -> None:
    typer.secho("\nKlinkenberg regression  k_g = k_L + (k_L * b) / P_mean", bold=True)
    typer.echo(
        f"  {'run':<28} {'P_mean':>12} {'1/P_mean':>12} {'k_g':>14} {'u(k_g)':>12} {'ss':>4}"
    )
    typer.echo(
        f"  {'':<28} {f'({pressure_unit})':>12} {f'(1/{pressure_unit})':>12} "
        f"{f'({permeability_unit})':>14} {f'({permeability_unit})':>12}"
    )
    for point in result.points:
        # 1/P is reported in the same unit as P, so it is the reciprocal of the
        # already-converted pressure.
        pressure = units.from_atm(point.mean_pressure_atm, pressure_unit)
        uncertainty = (
            f"{units.darcy_to(point.standard_uncertainty_darcy, permeability_unit):12.4g}"
            if point.standard_uncertainty_darcy is not None
            else f"{'--':>12}"
        )
        typer.echo(
            f"  {(point.label or '-')[:28]:<28} {pressure:12.5g} {1.0 / pressure:12.5g} "
            f"{units.darcy_to(point.apparent_permeability_darcy, permeability_unit):14.6g} "
            f"{uncertainty} {'ok' if point.steady_state else 'NO':>4}"
        )

    k_l = units.darcy_to(result.liquid_permeability_darcy, permeability_unit)
    b_display = units.from_atm(result.slippage_factor_atm, pressure_unit)
    typer.echo("")

    expanded = result.liquid_permeability_expanded_uncertainty_darcy
    if expanded is not None:
        typer.secho(
            f"  k_L (liquid-equivalent)  {k_l:.6g} +/- "
            f"{units.darcy_to(expanded, permeability_unit):.4g} {permeability_unit}"
            f"  (k = {result.coverage_factor:.2f}, {result.coverage_probability:.0%})",
            fg=typer.colors.GREEN, bold=True,
        )
    else:
        typer.secho(
            f"  k_L (liquid-equivalent)  {k_l:.6g} {permeability_unit}",
            fg=typer.colors.GREEN, bold=True,
        )

    b_uncertainty = result.slippage_factor_standard_uncertainty_atm
    if b_uncertainty is not None:
        typer.echo(
            f"  b   (slippage factor)    {b_display:.6g} +/- "
            f"{units.from_atm(b_uncertainty, pressure_unit):.3g} {pressure_unit} (1 u)"
        )
    else:
        typer.echo(f"  b   (slippage factor)    {b_display:.6g} {pressure_unit}")

    fit_kind = "weighted by u(k_g)" if result.weighted else "unweighted"
    typer.echo(
        f"  R^2                      {result.r_squared:.6f}   "
        f"({result.point_count} points, {fit_kind})"
    )

    for warning in result.warnings:
        typer.secho(f"  warning: {warning}", fg=typer.colors.YELLOW)


# --------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", help="Print the version and exit.", is_eager=True
    ),
) -> None:
    """gasperm -- gas permeability of core samples."""
    if version:
        typer.echo(f"gasperm {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


if __name__ == "__main__":  # pragma: no cover
    app()
