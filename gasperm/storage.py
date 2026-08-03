"""Run output: streaming CSV writer, metadata sidecar, and readers.

Layout of a run directory::

    runs/
      core-001_20260803T141530Z/
        readings.csv       # one row per sample, streamed and periodically flushed
        run.yaml           # config snapshot + summary, written at the end
        run.log            # per-run log file

CSV column names carry their unit as a suffix (``inlet_pressure_atm``,
``permeability_D``) and are always in **internal CGS-Darcy units**, never in
display units. Display units are a console/plot concern only; a stored run must
mean the same thing regardless of what the operator happened to be looking at.
Raw voltages are stored alongside, so a run can be reprocessed against a
corrected calibration without repeating the experiment.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Sequence

import yaml

from gasperm.acquisition import steady_state_stats
from gasperm.config import GaspermConfig, config_to_dict
from gasperm.models import KlinkenbergPoint, Reading, RunSummary

logger = logging.getLogger(__name__)

__all__ = [
    "READING_COLUMNS",
    "RunWriter",
    "read_readings_csv",
    "read_run_metadata",
    "run_directory_name",
    "point_from_run",
    "collect_points",
    "write_klinkenberg_result",
]

READINGS_FILENAME = "readings.csv"
METADATA_FILENAME = "run.yaml"
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
    "downstream_pressure_atm",
    "mean_pressure_atm",
    "flow_cm3_s",
    "flow_reference_pressure_atm",
    "temperature_C",
    "temperature_ok",
    "temperature_stale",
    "viscosity_cP",
    "permeability_D",
    "permeability_avg_D",
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
        "downstream_pressure_atm": f"{reading.downstream_pressure_atm:.8g}",
        "mean_pressure_atm": f"{reading.mean_pressure_atm:.8g}",
        "flow_cm3_s": f"{reading.flow_cm3_s:.8g}",
        "flow_reference_pressure_atm": f"{reading.flow_reference_pressure_atm:.8g}",
        "temperature_C": f"{reading.temperature_c:.4f}",
        "temperature_ok": int(reading.temperature_ok),
        "temperature_stale": int(reading.temperature_stale),
        "viscosity_cP": f"{reading.viscosity_cp:.8g}",
        "permeability_D": (
            "" if reading.permeability_darcy is None else f"{reading.permeability_darcy:.8g}"
        ),
        "permeability_avg_D": (
            ""
            if reading.permeability_darcy_avg is None
            else f"{reading.permeability_darcy_avg:.8g}"
        ),
        "note": reading.note or "",
        "temperature_raw": reading.temperature_raw or "",
    }


class RunWriter:
    """Streams readings to CSV and writes the metadata sidecar at the end.

    Flushes every ``run.flush_every_n`` samples so a crash costs at most that
    many rows rather than the whole run.
    """

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
        """Write ``run.yaml``: the config snapshot plus the run summary.

        The full config is snapshotted so a stored run is self-describing --
        reprocessing it later never depends on the config file still existing
        or still saying the same thing.
        """
        payload: dict[str, Any] = {
            "gasperm_run": {
                "started_at": self.started_at.isoformat(),
                "readings_csv": READINGS_FILENAME,
                "rows": self.rows_written,
            },
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

    Kept as dicts rather than :class:`Reading` models because a stored run may
    come from an older column set; the consumers here only need the few columns
    they name.

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
            rows.append(
                {
                    "elapsed_s": _float_or_none(row.get("elapsed_s")),
                    "mean_pressure_atm": _float_or_none(row.get("mean_pressure_atm")),
                    "permeability_D": _float_or_none(row.get("permeability_D")),
                    "temperature_C": _float_or_none(row.get("temperature_C")),
                    "flow_cm3_s": _float_or_none(row.get("flow_cm3_s")),
                }
            )
    return rows


def read_run_metadata(path: str | Path) -> dict[str, Any]:
    """Read a run's ``run.yaml``, or return ``{}`` when absent."""
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
    """Resolve a run directory or CSV path to ``(readings_csv, run_yaml)``.

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


def point_from_run(
    target: str | Path, *, averaging_window_s: float | None = None
) -> KlinkenbergPoint:
    """Reduce a stored ``collect`` run to one Klinkenberg point.

    Uses the same trailing-window reduction as the live run summary
    (:func:`gasperm.acquisition.steady_state_stats`) rather than reinventing
    "steady state", so a point derived here matches what ``collect`` printed.

    Args:
        target: Run directory or ``readings.csv``.
        averaging_window_s: Overrides the window stored in the run's metadata.

    Raises:
        ValueError: the run has no usable permeability samples.
    """
    readings_path, metadata_path = resolve_run_paths(target)
    metadata = read_run_metadata(metadata_path) if metadata_path else {}
    stored_config = metadata.get("config", {}) if isinstance(metadata, dict) else {}
    sample_id = (stored_config.get("sample") or {}).get("id")

    window_s = averaging_window_s
    if window_s is None:
        window_s = (stored_config.get("run") or {}).get("averaging_window_s")
    if window_s is None:
        window_s = 5.0

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

    timestamps = [row["elapsed_s"] for row in usable]
    mean_k, _, _ = steady_state_stats(
        timestamps, [row["permeability_D"] for row in usable], float(window_s)
    )
    mean_p, _, _ = steady_state_stats(
        timestamps, [row["mean_pressure_atm"] for row in usable], float(window_s)
    )
    if mean_k is None or mean_p is None or mean_p <= 0.0:
        raise ValueError(
            f"{readings_path}: could not reduce the trailing {window_s} s to a usable "
            "(mean pressure, permeability) pair."
        )

    return KlinkenbergPoint(
        mean_pressure_atm=float(mean_p),
        apparent_permeability_darcy=float(mean_k),
        label=Path(readings_path).parent.name,
        source_path=str(readings_path),
        sample_id=sample_id,
    )


def collect_points(
    targets: Sequence[str | Path], *, averaging_window_s: float | None = None
) -> list[KlinkenbergPoint]:
    """Reduce several stored runs to Klinkenberg points, in order."""
    return [
        point_from_run(target, averaging_window_s=averaging_window_s) for target in targets
    ]


def write_klinkenberg_result(result, path: str | Path) -> Path:
    """Write a Klinkenberg result to YAML, including the points it used."""
    from gasperm import units

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "klinkenberg": {
            "liquid_permeability_D": result.liquid_permeability_darcy,
            "liquid_permeability_mD": units.darcy_to(
                result.liquid_permeability_darcy, "mD"
            ),
            "slippage_factor_atm": result.slippage_factor_atm,
            "slope_D_atm": result.slope,
            "intercept_D": result.intercept,
            "r_squared": result.r_squared,
            "intercept_stderr_D": result.intercept_stderr,
            "slope_stderr": result.slope_stderr,
            "point_count": result.point_count,
            "warnings": result.warnings,
        },
        "points": [
            {
                "label": point.label,
                "sample_id": point.sample_id,
                "mean_pressure_atm": point.mean_pressure_atm,
                "inverse_mean_pressure_per_atm": point.inverse_mean_pressure,
                "apparent_permeability_D": point.apparent_permeability_darcy,
                "apparent_permeability_mD": units.darcy_to(
                    point.apparent_permeability_darcy, "mD"
                ),
                "source_path": point.source_path,
            }
            for point in result.points
        ],
    }
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return output_path
