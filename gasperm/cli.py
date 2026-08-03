"""Command-line entry point: ``gasperm init | collect | klinkenberg``."""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any, Optional

import typer

from gasperm import __version__, units
from gasperm.config import (
    HARDWARE_FILENAME,
    RUN_FILENAME,
    SAMPLE_FILENAME,
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

app = typer.Typer(
    add_completion=False,
    help=(
        "Measure gas permeability of core samples on an NI USB-6421 rig.\n\n"
        "Configuration is three files: hardware.yaml (the bench), sample.yaml (the "
        "core plug) and run.yaml (the experiment).\n\n"
        "Typical order: init once per rig, collect once per mean pressure, then "
        "klinkenberg across those runs."
    ),
)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


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

    Paths are rooted at the three sections, e.g. ``sample.length_cm``,
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
        "    Inlet pressure channel (0-5 V)", default=daq["inlet_pressure_channel"]
    )
    daq["outlet_pressure_channel"] = typer.prompt(
        "    Outlet pressure channel (0-5 V)", default=daq["outlet_pressure_channel"]
    )
    daq["sample_rate_hz"] = typer.prompt(
        "    Sample rate (Hz)", default=daq["sample_rate_hz"], type=float
    )

    supported = ", ".join(sorted(units.SUPPORTED_PRESSURE_UNITS))
    typer.secho(f"  Pressure transducers  (units: {supported})", bold=True)
    for side in ("inlet", "outlet"):
        section = hardware["pressure_calibration"][side]
        typer.secho(f"    {side}:", bold=True)
        section["volts_min"] = typer.prompt("      Volts at zero", default=section["volts_min"], type=float)
        section["volts_max"] = typer.prompt("      Volts at full scale", default=section["volts_max"], type=float)
        section["unit"] = typer.prompt("      Pressure unit", default=section["unit"])
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

    typer.secho("  Flowmeter  (one meter per run: ai2 or ai3)", bold=True)
    flow = hardware["flowmeter"]
    flow["channel"] = typer.prompt("    Channel", default=flow["channel"])
    flow["volts_min"] = typer.prompt("    Volts at zero flow", default=flow["volts_min"], type=float)
    flow["volts_max"] = typer.prompt("    Volts at full scale", default=flow["volts_max"], type=float)
    flow["unit"] = typer.prompt("    Flow unit", default=flow["unit"])
    flow["flow_min"] = typer.prompt(
        f"    Flow at {flow['volts_min']} V ({flow['unit']})", default=flow["flow_min"], type=float
    )
    flow["flow_max"] = typer.prompt(
        f"    Full-scale flow at {flow['volts_max']} V ({flow['unit']})",
        default=flow["flow_max"], type=float,
    )
    flow["reading_basis"] = typer.prompt(
        "    Reading basis (standard = mass-based meter / actual = at line conditions)",
        default=flow["reading_basis"],
    )
    flow["uncertainty"]["value"] = typer.prompt(
        "    Accuracy (% of reading)", default=flow["uncertainty"]["value"], type=float
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


def _prompt_sample(data: dict[str, Any]) -> None:
    typer.secho("\n== sample.yaml : the core plug ==", fg=typer.colors.CYAN, bold=True)
    sample = data["sample"]
    sample["id"] = typer.prompt("  Sample id", default=sample["id"])
    sample["description"] = typer.prompt("  Description", default="", show_default=False)
    sample["lithology"] = typer.prompt("  Lithology", default="", show_default=False)
    sample["formation"] = typer.prompt("  Formation", default="", show_default=False)
    sample["well"] = typer.prompt("  Well", default="", show_default=False)
    depth = typer.prompt("  Depth (blank to skip)", default="", show_default=False)
    if depth.strip():
        sample["depth"] = float(depth)
        sample["depth_unit"] = typer.prompt("  Depth unit", default=sample["depth_unit"])

    typer.secho("  Geometry", bold=True)
    sample["length_cm"] = typer.prompt("    Length (cm)", default=sample["length_cm"], type=float)
    sample["length_uncertainty_cm"] = typer.prompt(
        "    Length uncertainty (cm)", default=sample["length_uncertainty_cm"], type=float
    )
    sample["diameter_cm"] = typer.prompt("    Diameter (cm)", default=sample["diameter_cm"], type=float)
    sample["diameter_uncertainty_cm"] = typer.prompt(
        "    Diameter uncertainty (cm, counts double in the budget)",
        default=sample["diameter_uncertainty_cm"], type=float,
    )

    typer.secho("  Petrophysics (optional)", bold=True)
    porosity = typer.prompt("    Porosity fraction (blank to skip)", default="", show_default=False)
    if porosity.strip():
        sample["porosity_fraction"] = float(porosity)
        sample["porosity_method"] = typer.prompt(
            "    Porosity method", default="", show_default=False
        )
    grain = typer.prompt("    Grain density (g/cm3, blank to skip)", default="", show_default=False)
    if grain.strip():
        sample["grain_density_g_cm3"] = float(grain)
    bulk = typer.prompt("    Bulk density (g/cm3, blank to skip)", default="", show_default=False)
    if bulk.strip():
        sample["bulk_density_g_cm3"] = float(bulk)
    sample["prepared_by"] = typer.prompt("  Prepared by", default="", show_default=False)


def _prompt_run(data: dict[str, Any]) -> None:
    typer.secho("\n== run.yaml : the experiment ==", fg=typer.colors.CYAN, bold=True)
    run = data["run"]
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

    typer.secho("  Conditions", bold=True)
    run["confining_pressure_unit"] = typer.prompt(
        "    Confining pressure unit", default=run["confining_pressure_unit"]
    )
    confining = typer.prompt(
        f"    Confining pressure ({run['confining_pressure_unit']}, blank to skip)",
        default="", show_default=False,
    )
    if confining.strip():
        run["confining_pressure"] = float(confining)
    run["outlet_pressure_reference"] = typer.prompt(
        "    Downstream pressure P2 (atmospheric / measured / a number)",
        default=run["outlet_pressure_reference"],
    )
    if run["outlet_pressure_reference"] not in {"atmospheric", "measured"}:
        try:
            run["outlet_pressure_reference"] = float(run["outlet_pressure_reference"])
        except ValueError:
            typer.secho("    Not a number; using 'atmospheric'.", fg=typer.colors.YELLOW)
            run["outlet_pressure_reference"] = "atmospheric"
        else:
            run["outlet_pressure_reference_unit"] = typer.prompt(
                "    ... in which unit", default=run["outlet_pressure_reference_unit"]
            )
    run["atmospheric_pressure_unit"] = typer.prompt(
        "    Atmospheric pressure unit", default=run["atmospheric_pressure_unit"]
    )
    run["atmospheric_pressure"] = typer.prompt(
        f"    Atmospheric pressure ({run['atmospheric_pressure_unit']})",
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

    typer.secho("  Output", bold=True)
    run["output_dir"] = typer.prompt("    Output directory", default=run["output_dir"])
    run["display_pressure_unit"] = typer.prompt(
        "    Display pressure unit", default=run["display_pressure_unit"]
    )
    run["display_permeability_unit"] = typer.prompt(
        "    Display permeability unit (mD/D/uD/um2/m2)",
        default=run["display_permeability_unit"],
    )
    run["uncertainty"]["coverage_probability"] = typer.prompt(
        "    Uncertainty coverage probability",
        default=run["uncertainty"]["coverage_probability"], type=float,
    )


@app.command("init")
def init_command(
    directory: Path = typer.Argument(
        Path("."), help="Directory to write the three config files into."
    ),
    non_interactive: bool = typer.Option(
        False, "--non-interactive", "-n",
        help="Skip the prompts and write the default templates. Combine with --set.",
    ),
    set_values: list[str] = typer.Option(
        [], "--set", metavar="SECTION.FIELD=VALUE",
        help=(
            "Override a field, e.g. --set sample.length_cm=4.2 --set run.gas.name=Air "
            "--set hardware.daq.device_name=Dev2. Repeatable."
        ),
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files."),
    print_only: bool = typer.Option(
        False, "--print", help="Write nothing; print the three files to stdout."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Create hardware.yaml, sample.yaml and run.yaml.

    Interactive by default, prompting with the shipped NI USB-6421 defaults.
    ``--non-interactive`` plus ``--set`` covers scripted setup.
    """
    _configure_logging(verbose)

    defaults = GaspermConfig()
    data = config_to_dict(defaults)
    if not non_interactive:
        try:
            _prompt_hardware(data, defaults)
            _prompt_sample(data)
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
            (SAMPLE_FILENAME, render_sample_yaml(config)),
            (RUN_FILENAME, render_run_yaml(config)),
        ):
            typer.secho(f"\n# ===== {name} =====", fg=typer.colors.CYAN, bold=True)
            typer.echo(text)
        return

    try:
        paths = save_config(config, directory, overwrite=force)
    except ConfigError as exc:
        _fail(str(exc))
        return

    typer.secho("Wrote:", fg=typer.colors.GREEN)
    for path in paths.as_tuple():
        typer.secho(f"  {path}", fg=typer.colors.GREEN)

    # Environment checks are advisory at init time: a rig is often configured
    # before it is fully wired up.
    try:
        warnings = validate_for_collect(config)
    except ConfigError as exc:
        typer.secho("\nNot ready for a collect run yet:\n" + str(exc), fg=typer.colors.YELLOW)
    else:
        for warning in warnings:
            typer.secho(f"note: {warning}", fg=typer.colors.YELLOW)
        location = "" if directory == Path(".") else f" --config-dir {directory}"
        typer.echo(f"\nNext: gasperm collect{location}")


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
    )
    try:
        return reader.open()
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


@app.command("collect")
def collect_command(
    config_dir: Path = typer.Option(
        Path("."), "--config-dir", "-c", help="Directory holding the three config files."
    ),
    hardware: Optional[Path] = typer.Option(None, "--hardware", help="Override hardware.yaml."),
    sample: Optional[Path] = typer.Option(None, "--sample", help="Override sample.yaml."),
    run_file: Optional[Path] = typer.Option(None, "--run", help="Override run.yaml."),
    plot: bool = typer.Option(False, "--plot", help="Also open a live matplotlib window."),
    duration: Optional[float] = typer.Option(
        None, "--duration", "-d", metavar="SECONDS", help="Stop after this long."
    ),
    samples: Optional[int] = typer.Option(
        None, "--samples", "-n", help="Stop after this many samples."
    ),
    stop_when_steady: bool = typer.Option(
        False, "--stop-when-steady", help="End the run once steady state is confirmed."
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
        SampleProcessor,
        console_header,
        format_reading_line,
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
    if stop_when_steady:
        config.run.stop_when_steady = True

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
        "Sample %s, gas %s, %.3f cm x %.3f cm dia, %.4g Hz, operator %s",
        config.sample.id,
        config.run.gas.name,
        config.sample.length_cm,
        config.sample.diameter_cm,
        config.hardware.daq.sample_rate_hz,
        config.run.operator or "(unset)",
    )

    live_plot = None
    if plot:
        from gasperm.plotting import LivePlot, PlottingUnavailable

        try:
            live_plot = LivePlot(config).open()
        except PlottingUnavailable as exc:
            typer.secho(f"warning: live plot unavailable: {exc}", fg=typer.colors.YELLOW)
            live_plot = None

    processor = SampleProcessor(config, gas_provider)

    def on_reading(reading) -> None:
        writer.write(reading)
        typer.echo(format_reading_line(reading, config))
        if live_plot is not None:
            live_plot.add(reading)
            live_plot.maybe_redraw()

    loop = AcquisitionLoop(
        config, processor, analog_source, temperature_source, on_reading=on_reading
    )

    typer.secho(f"\nRecording to {writer.directory}   (Ctrl+C to stop)", fg=typer.colors.CYAN)
    if config.run.steady_state.enabled:
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
    typer.echo(console_header(config))

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
    if summary is not None and not summary.steady_state_reached:
        exit_code = exit_code or 2
    if exit_code:
        raise typer.Exit(code=exit_code)


def _print_run_summary(summary, config: GaspermConfig) -> None:
    run = config.run
    pressure_unit = run.display_pressure_unit
    permeability_unit = run.display_permeability_unit

    k_display = units.darcy_to(summary.permeability_darcy, permeability_unit)
    p_display = units.from_atm(summary.mean_pressure_atm, pressure_unit)

    typer.secho("Run summary", bold=True)
    typer.echo(f"  sample              {summary.sample_id}  ({summary.gas_name})")
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

    if summary.steady_state_reached and summary.steady_state_window is not None:
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
    typer.echo(
        f"  mean flow           "
        f"{units.flow_from_cm3_s(summary.mean_flow_cm3_s, run.display_flow_unit):.4g} "
        f"{run.display_flow_unit}"
    )

    budget = summary.uncertainty
    if budget is not None:
        expanded = units.darcy_to(budget.expanded_uncertainty_darcy, permeability_unit)
        typer.secho(
            f"  apparent k_g        {k_display:.5g} +/- {expanded:.3g} {permeability_unit}"
            f"  ({budget.relative_expanded_uncertainty:.2%}, k = {budget.coverage_factor:.2f},"
            f" {budget.coverage_probability:.0%})",
            fg=typer.colors.GREEN if summary.steady_state_reached else typer.colors.YELLOW,
            bold=True,
        )
        _print_budget(budget)
    else:
        stddev = units.darcy_to(summary.permeability_stddev_darcy, permeability_unit)
        typer.secho(
            f"  apparent k_g        {k_display:.5g} +/- {stddev:.3g} {permeability_unit} (1 sd)",
            fg=typer.colors.GREEN if summary.steady_state_reached else typer.colors.YELLOW,
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


@app.command("klinkenberg")
def klinkenberg_command(
    runs: list[Path] = typer.Argument(
        None, metavar="[RUN]...",
        help="Run directories (or readings.csv paths) from previous collect runs.",
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
    from gasperm.klinkenberg import fit_klinkenberg, load_points_from_csv
    from gasperm.storage import collect_points, write_klinkenberg_result

    _configure_logging(verbose)

    runs = [r for r in (runs or []) if r is not None]
    if not runs and csv_path is None:
        _fail(
            "Nothing to regress. Pass two or more collect run directories, or --csv with "
            "a file of mean-pressure / apparent-permeability pairs."
        )
        return
    if runs and csv_path is not None:
        _fail("Pass either run directories or --csv, not both.")
        return

    try:
        # Reject a bad display unit now, not after the regression has run.
        units.darcy_to(1.0, permeability_unit)
        units.from_atm(1.0, pressure_unit)
    except ValueError as exc:
        _fail(str(exc))
        return

    try:
        if csv_path is not None:
            points = load_points_from_csv(
                csv_path,
                pressure_unit=csv_pressure_unit,
                permeability_unit=csv_permeability_unit,
            )
        else:
            points = collect_points(
                runs, averaging_window_s=window, allow_unsteady=allow_unsteady
            )
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))
        return

    try:
        result = fit_klinkenberg(points, coverage_probability=coverage)
    except ValueError as exc:
        _fail(str(exc))
        return

    _print_klinkenberg(result, permeability_unit, pressure_unit)

    destination = output
    if destination is None and csv_path is None and runs:
        destination = Path(runs[0]).parent / "klinkenberg.yaml"
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
