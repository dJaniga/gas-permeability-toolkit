"""Run output: streaming CSV writer, metadata sidecar, and readers.

Layout of a run directory::

    runs/
      core-001_20260803T141530Z/
        readings.csv        # one row per sample, streamed and flushed periodically
        run_metadata.yaml   # config snapshot + experiment metadata + summary
        run.log             # per-run log file

The sidecar is ``run_metadata.yaml`` rather than ``run.yaml`` so it is never
confused with the *run configuration* file of that name.

CSV column names carry their unit as a suffix (``inlet_pressure_atm``,
``permeability_D``) and are always in **internal CGS-Darcy units**, never in
display units: a stored run must mean the same thing regardless of what the
operator happened to be looking at. Raw voltages are stored alongside, so a run
can be reprocessed against a corrected calibration without repeating the
experiment.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Sequence

import yaml

from gasperm import units
from gasperm.config import GaspermConfig, config_to_dict, experiment_metadata
from gasperm.config.run import SteadyStateConfig
from gasperm.models import KlinkenbergPoint, Reading, RunSummary
from gasperm.steady_state import detect_steady_window

logger = logging.getLogger(__name__)

__all__ = [
    "READING_COLUMNS",
    "RunWriter",
    "read_readings_csv",
    "read_run_metadata",
    "run_directory_name",
    "resolve_run_paths",
    "point_from_run",
    "collect_points",
    "write_klinkenberg_result",
]

READINGS_FILENAME = "readings.csv"
METADATA_FILENAME = "run_metadata.yaml"
LOG_FILENAME = "run.log"

#: CSV columns, in order. Unit suffixes are part of the contract -- the
#: klinkenberg reader and any external analysis depend on them.
READING_COLUMNS: tuple[str, ...] = (
    "index",
    "timestamp_utc",
    "elapsed_s",
    "inlet_voltage_V",
    "outlet_voltage_V",
    "flow_voltage_V",
    "inlet_pressure_atm",
    "outlet_pressure_atm",
    "mean_pressure_atm",
    "flow_cm3_s",
    "flow_reference_cm3_s",
    "flow_reference_pressure_atm",
    "temperature_C",
    "temperature_ok",
    "temperature_stale",
    "viscosity_cP",
    "compressibility_Z",
    "permeability_D",
    "permeability_avg_D",
    "steady_state",
    "steady_state_passes",
    "note",
    "temperature_raw",
)


def run_directory_name(sample_id: str, started_at: datetime | None = None) -> str:
    """Timestamped directory name for a run, e.g. ``core-001_20260803T141530Z``."""
    moment = (started_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in sample_id) or "sample"
    return f"{safe_id}_{moment.strftime('%Y%m%dT%H%M%SZ')}"


def _unique_directory(preferred: Path) -> Path:
    """``preferred``, or the next free ``-2``/``-3``... variant.

    The directory name is stamped to the second, so two short runs started
    inside the same second would otherwise land in the same directory and the
    later one would overwrite the earlier -- silently destroying a completed
    measurement.
    """
    if not preferred.exists():
        return preferred
    for suffix in range(2, 1000):
        candidate = preferred.with_name(f"{preferred.name}-{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a free run directory next to {preferred}.")


def _optional(value: float | None, spec: str = ".8g") -> str:
    return "" if value is None else format(value, spec)


def _reading_row(reading: Reading) -> dict[str, Any]:
    return {
        "index": reading.index,
        "timestamp_utc": reading.timestamp.astimezone(timezone.utc).isoformat(),
        "elapsed_s": f"{reading.elapsed_s:.4f}",
        "inlet_voltage_V": f"{reading.inlet_voltage:.6f}",
        "outlet_voltage_V": f"{reading.outlet_voltage:.6f}",
        "flow_voltage_V": f"{reading.flow_voltage:.6f}",
        "inlet_pressure_atm": f"{reading.inlet_pressure_atm:.8g}",
        "outlet_pressure_atm": f"{reading.outlet_pressure_atm:.8g}",
        "mean_pressure_atm": f"{reading.mean_pressure_atm:.8g}",
        "flow_cm3_s": f"{reading.flow_cm3_s:.8g}",
        "flow_reference_cm3_s": f"{reading.flow_reference_cm3_s:.8g}",
        "flow_reference_pressure_atm": f"{reading.flow_reference_pressure_atm:.8g}",
        "temperature_C": f"{reading.temperature_c:.4f}",
        "temperature_ok": int(reading.temperature_ok),
        "temperature_stale": int(reading.temperature_stale),
        "viscosity_cP": f"{reading.viscosity_cp:.8g}",
        "compressibility_Z": _optional(reading.compressibility_z),
        "permeability_D": _optional(reading.permeability_darcy),
        "permeability_avg_D": _optional(reading.permeability_darcy_avg),
        "steady_state": int(reading.steady_state),
        "steady_state_passes": reading.steady_state_passes,
        "note": reading.note or "",
        "temperature_raw": reading.temperature_raw or "",
    }


class RunWriter:
    """Streams readings to CSV and writes the metadata sidecar at the end."""

    def __init__(self, config: GaspermConfig, *, started_at: datetime | None = None) -> None:
        self.config = config
        self.started_at = started_at or datetime.now(timezone.utc)
        self.directory = _unique_directory(
            Path(config.run.output_dir) / run_directory_name(config.sample.id, self.started_at)
        )
        self.readings_path = self.directory / READINGS_FILENAME
        self.metadata_path = self.directory / METADATA_FILENAME
        self.log_path = self.directory / LOG_FILENAME
        self._handle = None
        self._writer: csv.DictWriter | None = None
        self._since_flush = 0
        self.rows_written = 0

    def open(self) -> RunWriter:
        """Create the run directory and start the CSV."""
        self.directory.mkdir(parents=True, exist_ok=True)
        self._handle = self.readings_path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=list(READING_COLUMNS))
        self._writer.writeheader()
        self._handle.flush()
        logger.info("Writing run to %s", self.directory)
        return self

    def __enter__(self) -> RunWriter:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def write(self, reading: Reading) -> None:
        """Append one reading, flushing periodically."""
        if self._writer is None or self._handle is None:
            raise RuntimeError("RunWriter is not open; call open() first.")
        self._writer.writerow(_reading_row(reading))
        self.rows_written += 1
        self._since_flush += 1
        if self._since_flush >= self.config.run.flush_every_n:
            self._handle.flush()
            self._since_flush = 0

    def close(self) -> None:
        """Flush and close the CSV. Safe to call more than once."""
        handle, self._handle = self._handle, None
        self._writer = None
        if handle is None:
            return
        try:
            handle.flush()
        finally:
            handle.close()

    def write_metadata(self, summary: RunSummary | None = None) -> Path:
        """Write the sidecar: config snapshot, experiment metadata, summary.

        The full config is snapshotted so a stored run is self-describing --
        reprocessing it later never depends on the config files still existing
        or still saying the same thing.
        """
        payload: dict[str, Any] = {
            "gasperm_run": {
                "started_at": self.started_at.isoformat(),
                "readings_csv": READINGS_FILENAME,
                "rows": self.rows_written,
            },
            "metadata": experiment_metadata(self.config).model_dump(mode="json"),
            "config": config_to_dict(self.config),
        }
        if summary is not None:
            payload["summary"] = summary.model_dump(mode="json")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        return self.metadata_path


# --------------------------------------------------------------------------
# Reading runs back
# --------------------------------------------------------------------------


def _float_or_none(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_readings_csv(path: str | Path) -> list[dict[str, Any]]:
    """Read a run's ``readings.csv`` into typed dicts.

    The returned rows carry both the stored column names and the signal keys
    the steady-state detector expects (``permeability``, ``inlet_pressure``,
    ``flow``, ``temperature`` in kelvin), so a stored run can be replayed
    through exactly the same detector the live run used.

    Raises:
        ValueError: the file is missing the columns a run CSV must have.
    """
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header row.")
        required = {"elapsed_s", "mean_pressure_atm", "permeability_D"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{csv_path} is missing required column(s): {', '.join(sorted(missing))}. "
                "Is this a gasperm run CSV?"
            )
        rows: list[dict[str, Any]] = []
        for row in reader:
            temperature_c = _float_or_none(row.get("temperature_C"))
            parsed = {
                "elapsed_s": _float_or_none(row.get("elapsed_s")),
                "mean_pressure_atm": _float_or_none(row.get("mean_pressure_atm")),
                "inlet_pressure_atm": _float_or_none(row.get("inlet_pressure_atm")),
                "outlet_pressure_atm": _float_or_none(row.get("outlet_pressure_atm")),
                "permeability_D": _float_or_none(row.get("permeability_D")),
                "temperature_C": temperature_c,
                "flow_cm3_s": _float_or_none(row.get("flow_cm3_s")),
                "steady_state": _float_or_none(row.get("steady_state")),
            }
            # Detector signal aliases.
            parsed["permeability"] = parsed["permeability_D"]
            parsed["inlet_pressure"] = parsed["inlet_pressure_atm"]
            parsed["flow"] = parsed["flow_cm3_s"]
            parsed["temperature"] = (
                units.celsius_to_kelvin(temperature_c) if temperature_c is not None else None
            )
            rows.append(parsed)
    return rows


def read_run_metadata(path: str | Path) -> dict[str, Any]:
    """Read a run's metadata sidecar, or return ``{}`` when absent."""
    metadata_path = Path(path)
    if not metadata_path.is_file():
        return {}
    try:
        data = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        logger.warning("Could not parse %s: %s", metadata_path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def resolve_run_paths(target: str | Path) -> tuple[Path, Path | None]:
    """Resolve a run directory or CSV path to ``(readings_csv, metadata)``.

    Accepts either the run directory or the ``readings.csv`` inside it, since
    operators reasonably point at both.

    Raises:
        FileNotFoundError: neither form resolves to a readable CSV.
    """
    candidate = Path(target)
    if candidate.is_dir():
        readings = candidate / READINGS_FILENAME
        if not readings.is_file():
            raise FileNotFoundError(
                f"{candidate} does not contain {READINGS_FILENAME}. Point at a gasperm "
                "run directory or directly at a readings CSV."
            )
        metadata = candidate / METADATA_FILENAME
        return readings, metadata if metadata.is_file() else None
    if candidate.is_file():
        metadata = candidate.parent / METADATA_FILENAME
        return candidate, metadata if metadata.is_file() else None
    raise FileNotFoundError(f"No such run directory or file: {candidate}")


def _steady_config_from_metadata(
    metadata: dict[str, Any], override_window_s: float | None
) -> SteadyStateConfig:
    """The steady-state criteria a stored run was recorded under."""
    stored = ((metadata.get("config") or {}).get("run") or {}).get("steady_state")
    config = SteadyStateConfig.model_validate(stored) if stored else SteadyStateConfig()
    if override_window_s is not None:
        config = config.model_copy(update={"window_s": override_window_s})
    return config


def point_from_run(
    target: str | Path,
    *,
    averaging_window_s: float | None = None,
    allow_unsteady: bool = False,
) -> KlinkenbergPoint:
    """Reduce a stored ``collect`` run to one Klinkenberg point.

    The reduction is the run's **steady-state window**, detected by replaying
    the stored readings through the same detector the live run used -- so a
    point derived here matches what ``collect`` reported. When the run's
    metadata already carries a steady summary, that is used directly, which
    also brings its uncertainty across for a weighted regression.

    Args:
        target: Run directory or ``readings.csv``.
        averaging_window_s: Override the stored detector window.
        allow_unsteady: Accept a run that never reached steady state. Off by
            default: such a run does not measure the sample.

    Raises:
        ValueError: the run has no usable samples, or never reached steady
            state and ``allow_unsteady`` is false.
    """
    readings_path, metadata_path = resolve_run_paths(target)
    metadata = read_run_metadata(metadata_path) if metadata_path else {}
    stored_config = metadata.get("config", {}) if isinstance(metadata, dict) else {}
    sample_id = (stored_config.get("sample") or {}).get("id")
    label = Path(readings_path).parent.name

    summary = metadata.get("summary") if isinstance(metadata, dict) else None
    if (
        summary
        and averaging_window_s is None
        and summary.get("steady_state_reached")
        and summary.get("permeability_darcy")
    ):
        budget = summary.get("uncertainty") or {}
        return KlinkenbergPoint(
            mean_pressure_atm=float(summary["mean_pressure_atm"]),
            apparent_permeability_darcy=float(summary["permeability_darcy"]),
            standard_uncertainty_darcy=(
                float(budget["combined_standard_uncertainty_darcy"])
                if budget.get("combined_standard_uncertainty_darcy")
                else None
            ),
            label=label,
            source_path=str(readings_path),
            sample_id=sample_id or summary.get("sample_id"),
            steady_state=True,
        )

    rows = read_readings_csv(readings_path)
    usable = [
        row
        for row in rows
        if row["permeability_D"] is not None
        and row["mean_pressure_atm"] is not None
        and row["elapsed_s"] is not None
    ]
    if not usable:
        raise ValueError(
            f"{readings_path} contains no sample with a usable permeability, so it "
            "cannot contribute a Klinkenberg point."
        )

    steady_config = _steady_config_from_metadata(metadata, averaging_window_s)
    window = detect_steady_window(usable, steady_config) if steady_config.enabled else None

    if window is None:
        if not allow_unsteady:
            raise ValueError(
                f"{readings_path} never reached steady state under its own criteria, so "
                "it does not measure this sample's permeability. Re-run it to a "
                "plateau, or pass --allow-unsteady to use it anyway and accept that the "
                "result is not representative."
            )
        selected = usable
    else:
        selected = [
            row
            for row in usable
            if window.start_elapsed_s <= row["elapsed_s"] <= window.end_elapsed_s
        ]

    permeabilities = [row["permeability_D"] for row in selected]
    pressures = [row["mean_pressure_atm"] for row in selected]
    mean_k = sum(permeabilities) / len(permeabilities)
    mean_p = sum(pressures) / len(pressures)
    if mean_p <= 0.0:
        raise ValueError(
            f"{readings_path}: the steady window has a non-positive mean pressure."
        )

    return KlinkenbergPoint(
        mean_pressure_atm=mean_p,
        apparent_permeability_darcy=mean_k,
        label=label,
        source_path=str(readings_path),
        sample_id=sample_id,
        steady_state=window is not None,
    )


def collect_points(
    targets: Sequence[str | Path],
    *,
    averaging_window_s: float | None = None,
    allow_unsteady: bool = False,
) -> list[KlinkenbergPoint]:
    """Reduce several stored runs to Klinkenberg points, in order."""
    return [
        point_from_run(
            target, averaging_window_s=averaging_window_s, allow_unsteady=allow_unsteady
        )
        for target in targets
    ]


def write_klinkenberg_result(result, path: str | Path) -> Path:
    """Write a Klinkenberg result to YAML, including the points it used."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "klinkenberg": {
            "liquid_permeability_D": result.liquid_permeability_darcy,
            "liquid_permeability_mD": units.darcy_to(result.liquid_permeability_darcy, "mD"),
            "expanded_uncertainty_D": result.liquid_permeability_expanded_uncertainty_darcy,
            "coverage_factor": result.coverage_factor,
            "coverage_probability": result.coverage_probability,
            "slippage_factor_atm": result.slippage_factor_atm,
            "slippage_factor_standard_uncertainty_atm": (
                result.slippage_factor_standard_uncertainty_atm
            ),
            "slope_D_atm": result.slope,
            "intercept_D": result.intercept,
            "r_squared": result.r_squared,
            "intercept_stderr_D": result.intercept_stderr,
            "slope_stderr": result.slope_stderr,
            "weighted": result.weighted,
            "point_count": result.point_count,
            "warnings": result.warnings,
        },
        "points": [
            {
                "label": point.label,
                "sample_id": point.sample_id,
                "steady_state": point.steady_state,
                "mean_pressure_atm": point.mean_pressure_atm,
                "inverse_mean_pressure_per_atm": point.inverse_mean_pressure,
                "apparent_permeability_D": point.apparent_permeability_darcy,
                "apparent_permeability_mD": units.darcy_to(
                    point.apparent_permeability_darcy, "mD"
                ),
                "standard_uncertainty_D": point.standard_uncertainty_darcy,
                "source_path": point.source_path,
            }
            for point in result.points
        ],
    }
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return output_path
