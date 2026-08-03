"""Command-line entry point: ``gasperm init | collect | klinkenberg``."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import typer

from gasperm import __version__, units
from gasperm.config import (
    DEFAULT_CONFIG_FILENAME,
    ConfigError,
    GaspermConfig,
    config_to_dict,
    load_config,
    render_config_yaml,
    save_config,
    validate_for_collect,
)

logger = logging.getLogger("gasperm")

app = typer.Typer(
    add_completion=False,
    help=(
        "Measure gas permeability of core samples on an NI USB-6421 rig.\n\n"
        "Typical order: init to describe the rig, collect once per mean pressure, "
        "then klinkenberg across those runs."
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
    """Console logging, plus a per-run file handler when a run directory exists.

    Hardware disconnects and other mid-run problems land in both, timestamped,
    rather than being silently swallowed.
    """
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


def _load_or_fail(config_path: Path) -> GaspermConfig:
    try:
        return load_config(config_path)
    except ConfigError as exc:
        _fail(str(exc))
        raise  # unreachable; keeps type checkers happy


def _set_dotted(data: dict[str, Any], dotted_key: str, raw_value: str) -> None:
    """Apply a ``section.field=value`` override to a plain config dict.

    Values are parsed as YAML so numbers, booleans and ``null`` come through as
    the right type instead of strings.
    """
    import yaml

    parts = dotted_key.split(".")
    target = data
    for part in parts[:-1]:
        nested = target.get(part)
        if not isinstance(nested, dict):
            raise ConfigError(
                f"--set {dotted_key}: {part!r} is not a config section. Sections are: "
                f"{', '.join(sorted(k for k, v in data.items() if isinstance(v, dict)))}"
            )
        target = nested
    try:
        target[parts[-1]] = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"--set {dotted_key}: could not parse {raw_value!r}: {exc}") from exc


# --------------------------------------------------------------------------
# init
# --------------------------------------------------------------------------


def _prompt_config() -> GaspermConfig:
    """Walk the operator through the rig, defaulting to the shipped wiring."""
    defaults = GaspermConfig()
    data = config_to_dict(defaults)

    typer.secho("\nDAQ (NI USB-6421)", fg=typer.colors.CYAN, bold=True)
    data["daq"]["device_name"] = typer.prompt(
        "  Device name (from NI MAX)", default=defaults.daq.device_name
    )
    data["daq"]["inlet_pressure_channel"] = typer.prompt(
        "  Inlet pressure channel (0-5 V)", default=defaults.daq.inlet_pressure_channel
    )
    data["daq"]["outlet_pressure_channel"] = typer.prompt(
        "  Outlet pressure channel (0-5 V)", default=defaults.daq.outlet_pressure_channel
    )
    data["daq"]["sample_rate_hz"] = typer.prompt(
        "  Sample rate (Hz)", default=defaults.daq.sample_rate_hz, type=float
    )

    supported = ", ".join(sorted(units.SUPPORTED_PRESSURE_UNITS))
    typer.secho(f"\nPressure transducers  (units: {supported})", fg=typer.colors.CYAN, bold=True)
    for side in ("inlet", "outlet"):
        section = data["pressure_calibration"][side]
        typer.secho(f"  {side}:", bold=True)
        section["volts_min"] = typer.prompt("    Volts at zero", default=section["volts_min"], type=float)
        section["volts_max"] = typer.prompt("    Volts at full scale", default=section["volts_max"], type=float)
        section["unit"] = typer.prompt("    Pressure unit", default=section["unit"])
        section["value_min"] = typer.prompt(
            f"    Pressure at {section['volts_min']} V ({section['unit']})",
            default=section["value_min"],
            type=float,
        )
        section["value_max"] = typer.prompt(
            f"    Pressure at {section['volts_max']} V ({section['unit']})",
            default=section["value_max"],
            type=float,
        )
        section["reading_type"] = typer.prompt(
            "    Reading type (absolute/gauge)", default=section["reading_type"]
        )

    typer.secho("\nFlowmeter  (one meter per run: ai2 or ai3)", fg=typer.colors.CYAN, bold=True)
    flow = data["flowmeter"]
    flow["channel"] = typer.prompt("  Channel", default=flow["channel"])
    flow["volts_min"] = typer.prompt("  Volts at zero flow", default=flow["volts_min"], type=float)
    flow["volts_max"] = typer.prompt("  Volts at full scale", default=flow["volts_max"], type=float)
    flow["unit"] = typer.prompt("  Flow unit", default=flow["unit"])
    flow["flow_min"] = typer.prompt(
        f"  Flow at {flow['volts_min']} V ({flow['unit']})", default=flow["flow_min"], type=float
    )
    flow["flow_max"] = typer.prompt(
        f"  Full-scale flow at {flow['volts_max']} V ({flow['unit']})",
        default=flow["flow_max"],
        type=float,
    )
    flow["reading_basis"] = typer.prompt(
        "  Reading basis (standard = mass-based meter / actual = at line conditions)",
        default=flow["reading_basis"],
    )

    typer.secho("\nTemperature probe (Arduino, USB serial)", fg=typer.colors.CYAN, bold=True)
    temperature = data["temperature"]
    temperature["port"] = typer.prompt("  Serial port", default=temperature["port"])
    temperature["baud_rate"] = typer.prompt("  Baud rate", default=temperature["baud_rate"], type=int)
    temperature["parse_pattern"] = typer.prompt(
        "  Line format ('{value}' marks the number; '-' for any number on the line)",
        default=temperature["parse_pattern"],
    )
    if temperature["parse_pattern"] in {"-", ""}:
        temperature["parse_pattern"] = None
    temperature["units"] = typer.prompt("  Probe reports in (C/K/F)", default=temperature["units"])

    typer.secho("\nGas", fg=typer.colors.CYAN, bold=True)
    data["gas"]["name"] = typer.prompt(
        "  CoolProp fluid name (Nitrogen, Air, CarbonDioxide, Methane, ...)",
        default=defaults.gas.name,
    )

    typer.secho("\nSample", fg=typer.colors.CYAN, bold=True)
    sample = data["sample"]
    sample["id"] = typer.prompt("  Sample id", default=sample["id"])
    sample["description"] = typer.prompt("  Description", default="", show_default=False)
    sample["length_cm"] = typer.prompt("  Length (cm)", default=sample["length_cm"], type=float)
    sample["diameter_cm"] = typer.prompt("  Diameter (cm)", default=sample["diameter_cm"], type=float)
    sample["confining_pressure_unit"] = typer.prompt(
        "  Confining pressure unit", default=sample["confining_pressure_unit"]
    )
    confining = typer.prompt(
        f"  Confining pressure ({sample['confining_pressure_unit']}, blank to skip)",
        default="",
        show_default=False,
    )
    sample["confining_pressure"] = float(confining) if confining.strip() else None

    typer.secho("\nRun", fg=typer.colors.CYAN, bold=True)
    run = data["run"]
    run["outlet_pressure_reference"] = typer.prompt(
        "  Downstream pressure P2 (atmospheric / measured / a number)",
        default=run["outlet_pressure_reference"],
    )
    if run["outlet_pressure_reference"] not in {"atmospheric", "measured"}:
        try:
            run["outlet_pressure_reference"] = float(run["outlet_pressure_reference"])
        except ValueError:
            typer.secho(
                "  Not a number; falling back to 'atmospheric'.", fg=typer.colors.YELLOW
            )
            run["outlet_pressure_reference"] = "atmospheric"
        else:
            run["outlet_pressure_reference_unit"] = typer.prompt(
                "  ... in which unit", default=run["outlet_pressure_reference_unit"]
            )
    run["atmospheric_pressure_unit"] = typer.prompt(
        "  Atmospheric pressure unit", default=run["atmospheric_pressure_unit"]
    )
    run["atmospheric_pressure"] = typer.prompt(
        f"  Atmospheric pressure ({run['atmospheric_pressure_unit']})",
        default=run["atmospheric_pressure"],
        type=float,
    )
    run["averaging_window_s"] = typer.prompt(
        "  Averaging window (s)", default=run["averaging_window_s"], type=float
    )
    run["output_dir"] = typer.prompt("  Output directory", default=run["output_dir"])
    run["display_pressure_unit"] = typer.prompt(
        "  Display pressure unit", default=run["display_pressure_unit"]
    )
    run["display_permeability_unit"] = typer.prompt(
        "  Display permeability unit (mD/D/uD/um2/m2)",
        default=run["display_permeability_unit"],
    )

    return GaspermConfig.model_validate(data)


@app.command("init")
def init_command(
    output: Path = typer.Option(
        Path(DEFAULT_CONFIG_FILENAME), "--output", "-o", help="Where to write the config."
    ),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        "-n",
        help="Skip the prompts and write the default template. Combine with --set.",
    ),
    set_values: list[str] = typer.Option(
        [],
        "--set",
        metavar="SECTION.FIELD=VALUE",
        help="Override a field, e.g. --set sample.length_cm=4.2 --set gas.name=Air. Repeatable.",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite an existing file."),
    print_only: bool = typer.Option(
        False, "--print", help="Write nothing; print the config to stdout."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Create a rig configuration file.

    Interactive by default, prompting with the shipped NI USB-6421 defaults.
    ``--non-interactive`` plus ``--set`` covers scripted setup.
    """
    _configure_logging(verbose)

    try:
        config = GaspermConfig() if non_interactive else _prompt_config()
    except ConfigError as exc:
        _fail(str(exc))
        return
    except Exception as exc:  # noqa: BLE001 - pydantic validation of prompted input
        _fail(f"Could not build a valid config from those answers:\n  {exc}")
        return

    if set_values:
        data = config_to_dict(config)
        try:
            for assignment in set_values:
                if "=" not in assignment:
                    raise ConfigError(
                        f"--set {assignment!r} is not in SECTION.FIELD=VALUE form."
                    )
                key, value = assignment.split("=", 1)
                _set_dotted(data, key.strip(), value.strip())
            config = GaspermConfig.model_validate(data)
        except ConfigError as exc:
            _fail(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            _fail(f"Overrides produced an invalid config:\n  {exc}")
            return

    if print_only:
        typer.echo(render_config_yaml(config))
        return

    try:
        written = save_config(config, output, overwrite=force)
    except ConfigError as exc:
        _fail(str(exc))
        return

    typer.secho(f"Wrote {written}", fg=typer.colors.GREEN)

    # Environment checks are advisory at init time: a rig is often configured
    # before it is fully wired up.
    try:
        warnings = validate_for_collect(config)
    except ConfigError as exc:
        typer.secho(
            "\nThis config is not ready for a collect run yet:\n" + str(exc),
            fg=typer.colors.YELLOW,
        )
    else:
        for warning in warnings:
            typer.secho(f"note: {warning}", fg=typer.colors.YELLOW)
        typer.echo(f"\nNext: gasperm collect --config {output}")


# --------------------------------------------------------------------------
# collect
# --------------------------------------------------------------------------


def _open_temperature_source(config: GaspermConfig):
    """Open the probe, or fall back when it is not required."""
    from gasperm.hardware.temperature import (
        SerialTemperatureReader,
        StaticTemperatureSource,
    )

    reader = SerialTemperatureReader(
        config.temperature.port,
        config.temperature.baud_rate,
        parse_pattern=config.temperature.parse_pattern,
        timeout_s=config.temperature.timeout_s,
        unit=config.temperature.units,
        stale_after_s=config.temperature.stale_after_s,
    )
    try:
        return reader.open()
    except OSError as exc:
        if config.temperature.required:
            raise
        logger.warning(
            "%s Continuing without the probe because temperature.required is false.", exc
        )
        return StaticTemperatureSource(
            config.temperature.fallback_temperature_c,
            note=f"{config.temperature.port} could not be opened",
        )


@app.command("collect")
def collect_command(
    config_path: Path = typer.Option(
        Path(DEFAULT_CONFIG_FILENAME), "--config", "-c", help="Rig configuration file."
    ),
    plot: bool = typer.Option(False, "--plot", help="Also open a live matplotlib window."),
    duration: Optional[float] = typer.Option(
        None, "--duration", "-d", metavar="SECONDS", help="Stop after this long."
    ),
    samples: Optional[int] = typer.Option(
        None, "--samples", "-n", help="Stop after this many samples."
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", help="Override run.output_dir."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Acquire pressure, flow and temperature, and compute permeability live.

    Runs until Ctrl+C unless ``--duration``/``--samples`` is given. Readings
    stream to a timestamped run directory as they are taken.
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
    config = _load_or_fail(config_path)

    if duration is not None:
        config.run.duration_s = duration
    if samples is not None:
        config.run.max_samples = samples
    if output_dir is not None:
        config.run.output_dir = str(output_dir)

    # Fail loudly and specifically BEFORE opening the DAQ -- never discover a
    # bad config three minutes into a run.
    try:
        startup_warnings = validate_for_collect(config)
    except ConfigError as exc:
        _fail(str(exc))
        return
    for warning in startup_warnings:
        typer.secho(f"warning: {warning}", fg=typer.colors.YELLOW, err=True)

    # Open every device BEFORE creating the run directory, so a rig that is not
    # ready does not litter runs/ with empty directories.
    try:
        gas_provider = build_provider(config.gas)
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
        "Sample %s, gas %s, %.3f cm x %.3f cm dia, %.4g Hz",
        config.sample.id,
        config.gas.name,
        config.sample.length_cm,
        config.sample.diameter_cm,
        config.daq.sample_rate_hz,
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
        config,
        processor,
        analog_source,
        temperature_source,
        on_reading=on_reading,
    )

    typer.secho(
        f"\nRecording to {writer.directory}   (Ctrl+C to stop)\n", fg=typer.colors.CYAN
    )
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
    typer.secho(f"Run written to {writer.directory}", fg=typer.colors.GREEN)
    if loop.warnings:
        typer.secho(f"{len(loop.warnings)} warning(s) logged to {writer.log_path}", fg=typer.colors.YELLOW)
    if exit_code:
        raise typer.Exit(code=exit_code)


def _print_run_summary(summary, config: GaspermConfig) -> None:
    run = config.run
    pressure_unit = run.display_pressure_unit
    permeability_unit = run.display_permeability_unit

    k_display = units.darcy_to(summary.permeability_darcy, permeability_unit)
    k_stddev = units.darcy_to(summary.permeability_stddev_darcy, permeability_unit)
    p_display = units.from_atm(summary.mean_pressure_atm, pressure_unit)

    typer.secho("Run summary", bold=True)
    typer.echo(f"  sample              {summary.sample_id}  ({summary.gas_name})")
    typer.echo(f"  duration            {summary.duration_s:.1f} s over {summary.sample_count} samples")
    typer.echo(
        f"  steady-state window last {run.averaging_window_s:g} s "
        f"({summary.averaged_samples} samples)"
    )
    typer.echo(f"  mean pressure       {p_display:.4g} {pressure_unit}")
    typer.echo(f"  mean temperature    {summary.mean_temperature_c:.2f} C")
    typer.echo(
        f"  mean flow           "
        f"{units.flow_from_cm3_s(summary.mean_flow_cm3_s, run.display_flow_unit):.4g} "
        f"{run.display_flow_unit}"
    )
    typer.secho(
        f"  apparent k_g        {k_display:.5g} +/- {k_stddev:.3g} {permeability_unit}",
        fg=typer.colors.GREEN,
        bold=True,
    )


# --------------------------------------------------------------------------
# klinkenberg
# --------------------------------------------------------------------------


@app.command("klinkenberg")
def klinkenberg_command(
    runs: list[Path] = typer.Argument(
        None,
        metavar="[RUN]...",
        help="Run directories (or readings.csv paths) from previous collect runs.",
    ),
    csv_path: Optional[Path] = typer.Option(
        None,
        "--csv",
        help="Standalone CSV of mean-pressure / apparent-permeability pairs instead of runs.",
    ),
    csv_pressure_unit: Optional[str] = typer.Option(
        None, "--csv-pressure-unit", help="Unit of the CSV's mean-pressure column."
    ),
    csv_permeability_unit: Optional[str] = typer.Option(
        None, "--csv-permeability-unit", help="Unit of the CSV's permeability column."
    ),
    window: Optional[float] = typer.Option(
        None,
        "--window",
        "-w",
        metavar="SECONDS",
        help="Steady-state averaging window per run. Defaults to each run's own setting.",
    ),
    permeability_unit: str = typer.Option(
        "mD", "--unit", "-u", help="Unit for the reported permeabilities."
    ),
    pressure_unit: str = typer.Option(
        "atm", "--pressure-unit", help="Unit for the reported pressures and b."
    ),
    plot: bool = typer.Option(False, "--plot", help="Save a regression plot beside the results."),
    show: bool = typer.Option(False, "--show", help="Open the regression plot in a window."),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write the results YAML here."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Recover liquid-equivalent permeability from runs at different mean pressures.

    Regresses apparent permeability against ``1 / P_mean``; the intercept is
    ``k_L`` and ``slope / intercept`` is the slippage factor ``b``.
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
            points = collect_points(runs, averaging_window_s=window)
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))
        return

    try:
        result = fit_klinkenberg(points)
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
                result,
                path=image_path,
                show=show,
                permeability_unit=permeability_unit,
                pressure_unit=pressure_unit,
            )
        except PlottingUnavailable as exc:
            typer.secho(f"warning: {exc}", fg=typer.colors.YELLOW)
        else:
            if saved is not None:
                typer.secho(f"Plot written to {saved}", fg=typer.colors.GREEN)


def _print_klinkenberg(result, permeability_unit: str, pressure_unit: str) -> None:
    typer.secho("\nKlinkenberg regression  k_g = k_L + (k_L * b) / P_mean", bold=True)
    header = (
        f"  {'run':<28} {'P_mean':>12} {'1/P_mean':>12} {'k_g':>14}"
    )
    typer.echo(header)
    typer.echo(
        f"  {'':<28} {f'({pressure_unit})':>12} {f'(1/{pressure_unit})':>12} "
        f"{f'({permeability_unit})':>14}"
    )
    for point in result.points:
        # 1/P is reported in the same unit as P, so it is simply the reciprocal
        # of the already-converted pressure.
        pressure = units.from_atm(point.mean_pressure_atm, pressure_unit)
        typer.echo(
            f"  {(point.label or '-')[:28]:<28} {pressure:12.5g} {1.0 / pressure:12.5g} "
            f"{units.darcy_to(point.apparent_permeability_darcy, permeability_unit):14.6g}"
        )

    k_l = units.darcy_to(result.liquid_permeability_darcy, permeability_unit)
    b_display = units.from_atm(result.slippage_factor_atm, pressure_unit)
    typer.echo("")
    typer.secho(
        f"  k_L (liquid-equivalent)  {k_l:.6g} {permeability_unit}",
        fg=typer.colors.GREEN,
        bold=True,
    )
    typer.echo(f"  b   (slippage factor)    {b_display:.6g} {pressure_unit}")
    typer.echo(f"  R^2                      {result.r_squared:.6f}   ({result.point_count} points)")
    if result.intercept_stderr is not None:
        stderr_display = units.darcy_to(result.intercept_stderr, permeability_unit)
        typer.echo(f"  std. error on k_L        {stderr_display:.4g} {permeability_unit}")

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
