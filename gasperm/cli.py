"""Command-line entry point: ``gasperm init | collect | klinkenberg``."""

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
        "mean pressure, then klinkenberg across those runs."
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
    except yaml.YAMLError as exc:
        raise ConfigError(f"--set {dotted_key}: could not parse {raw_value!r}: {exc}") from exc


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
    porosity = _optional_float("    Porosity fraction (blank to skip)")
    if porosity is not None:
        sample["porosity_fraction"] = porosity
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

    def on_reading(reading) -> None:
        writer.write(reading)
        if pulse_mode:
            typer.echo(format_pulse_reading_line(reading, loop.status, config))
        else:
            typer.echo(format_reading_line(reading, config))
        if live_plot is not None:
            # The detector's verdict is what the criterion lines are drawn
            # from; it is already updated for this reading by the time the
            # loop calls back. A pulse run has no steady-state status, so the
            # plot draws no criteria -- which its own panels already reflect.
            live_plot.add(reading, None if pulse_mode else loop.status)
            live_plot.maybe_redraw()

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
        loop.run()
    except DaqError as exc:
        logger.error("%s", exc)
        typer.secho(f"\nAcquisition stopped: {exc}", fg=typer.colors.RED, err=True)
        exit_code = 1
    except KeyboardInterrupt:  # pragma: no cover - the handler normally catches this
        logger.info("Interrupted.")
    finally:
        writer.close()
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
        try:
            saved = plot_klinkenberg(
                result, path=image_path, show=show,
                permeability_unit=permeability_unit, pressure_unit=pressure_unit,
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
