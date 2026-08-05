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
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Mapping, Sequence

import yaml

from gasperm import units
from gasperm.config import GaspermConfig, config_to_dict, experiment_metadata
from gasperm.config.run import SteadyStateConfig
from gasperm.models import KlinkenbergPoint, Reading, RunSummary
from gasperm.steady_state import detect_steady_window

logger = logging.getLogger(__name__)

__all__ = [
    "READING_COLUMNS",
    "RunRecord",
    "RunWriter",
    "find_runs",
    "read_readings_csv",
    "read_run_metadata",
    "run_directory_name",
    "runs_for_sample",
    "resolve_run_paths",
    "point_from_run",
    "collect_points",
    "describe_convention",
    "downstream_convention",
    "safe_sample_id",
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
    "downstream_pressure_atm",
    "mean_pressure_atm",
    "flow_cm3_s",
    "flow_reference_cm3_s",
    "flow_reference_pressure_atm",
    "temperature_C",
    "temperature_ok",
    "temperature_stale",
    "temperature_age_s",
    "viscosity_cP",
    "compressibility_Z",
    "permeability_D",
    "permeability_avg_D",
    "steady_state",
    "steady_state_passes",
    "note",
    "temperature_raw",
)


def safe_sample_id(sample_id: str) -> str:
    """Filesystem-safe form of a sample id, as used in run directory names.

    Lossy on purpose: ``core/041`` and ``core_041`` both become ``core_041``.
    The metadata sidecar is therefore authoritative for a run's sample id, and
    the directory prefix is only a fallback for a run whose sidecar was lost.
    """
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in sample_id) or "sample"


def run_directory_name(sample_id: str, started_at: datetime | None = None) -> str:
    """Timestamped directory name for a run, e.g. ``core-001_20260803T141530Z``."""
    moment = (started_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return f"{safe_sample_id(sample_id)}_{moment.strftime('%Y%m%dT%H%M%SZ')}"


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
        "downstream_pressure_atm": f"{reading.downstream_pressure_atm:.8g}",
        "mean_pressure_atm": f"{reading.mean_pressure_atm:.8g}",
        "flow_cm3_s": f"{reading.flow_cm3_s:.8g}",
        "flow_reference_cm3_s": f"{reading.flow_reference_cm3_s:.8g}",
        "flow_reference_pressure_atm": f"{reading.flow_reference_pressure_atm:.8g}",
        "temperature_C": f"{reading.temperature_c:.4f}",
        "temperature_ok": int(reading.temperature_ok),
        "temperature_stale": int(reading.temperature_stale),
        "temperature_age_s": _optional(reading.temperature_age_s, ".3f"),
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
            config.resolved_output_dir()
            / run_directory_name(config.sample.id, self.started_at)
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


def _summary_float(value: Any) -> float | None:
    """Coerce one number out of a metadata sidecar, tolerating age and junk.

    Sidecars written by older versions simply lack the newer keys, which must
    read as "unknown" rather than raise.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
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
                # Absent from runs recorded before P2 became overridable.
                "downstream_pressure_atm": _float_or_none(
                    row.get("downstream_pressure_atm")
                ),
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


# --------------------------------------------------------------------------
# Discovering runs
# --------------------------------------------------------------------------

#: ``<sample>_<YYYYMMDDTHHMMSSZ>`` with an optional ``-2`` collision suffix.
_RUN_DIR_PATTERN = re.compile(r"^(?P<sample>.+)_(?P<stamp>\d{8}T\d{6}Z)(?:-\d+)?$")

#: Sort floor for a run whose start time cannot be determined at all.
_UNKNOWN_START = datetime(1, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class RunRecord:
    """One discovered ``collect`` run, summarised without reading its CSV.

    Discovery has to stay cheap over a directory holding hundreds of runs, so
    everything here comes from the metadata sidecar. The measured fields are
    ``None`` for a run that never got one; :func:`point_from_run` reads the CSV
    later, and only for the runs actually used.
    """

    directory: Path
    readings_csv: Path
    metadata_path: Path | None
    sample_id: str | None
    #: False when the id was recovered from the directory name, which is lossy.
    sample_id_from_metadata: bool
    started_at: datetime | None
    has_summary: bool
    mean_pressure_atm: float | None = None
    permeability_darcy: float | None = None
    steady_state_reached: bool | None = None
    flowmeter: str | None = None
    #: How this run obtained P2; see :func:`downstream_convention`.
    downstream_convention: str | None = None

    @property
    def name(self) -> str:
        """The run directory's name, which is what the operator sees."""
        return self.directory.name


def downstream_convention(stored_config: Mapping[str, Any] | None) -> str | None:
    """Canonical key for how a stored run obtained P2.

    ``"measured"``, or ``"fixed:<atm>"`` so that the same physical pressure
    written as ``101.8 kPa`` and as ``1.018 bar`` compares equal.

    Derived from the *presence of the run block*, not of the key: a sidecar
    written before P2 became overridable was measured by definition. Returning
    ``None`` for those would let a set of old runs plus one supplied-P2 run
    collapse to a single distinct convention and slip past the very check this
    exists for. ``None`` means genuinely unknown -- no sidecar at all.
    """
    if not stored_config:
        return None
    run_block = stored_config.get("run")
    if run_block is None:
        return None

    value = run_block.get("downstream_pressure", "measured")
    if isinstance(value, str):
        return "measured"
    unit = run_block.get("downstream_pressure_unit", "kPa")
    try:
        return f"fixed:{units.to_atm(float(value), unit):.6g}"
    except (TypeError, ValueError):
        return None


def describe_convention(key: str | None) -> str:
    """Human-readable form of a convention key, for listings and messages."""
    if key is None:
        return "unknown"
    if key == "measured":
        return "measured"
    atm = float(key.split(":", 1)[1])
    return f"fixed {units.from_atm(atm, 'kPa'):.4g} kPa"


def _parse_started_at(metadata: dict[str, Any], directory_name: str) -> datetime | None:
    """Start time from the sidecar, falling back to the directory stamp."""
    stamp = (metadata.get("gasperm_run") or {}).get("started_at")
    if isinstance(stamp, str) and stamp:
        text = stamp[:-1] + "+00:00" if stamp.endswith("Z") else stamp
        try:
            moment = datetime.fromisoformat(text)
        except ValueError:
            moment = None
        if moment is not None:
            # A hand-edited sidecar can be naive; make every record comparable
            # so sorting cannot raise.
            return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)

    match = _RUN_DIR_PATTERN.match(directory_name)
    if match:
        try:
            return datetime.strptime(match["stamp"], "%Y%m%dT%H%M%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            pass
    return None


def _record_from_directory(directory: Path) -> RunRecord:
    """Summarise one run directory that is known to hold a readings CSV."""
    metadata_path = directory / METADATA_FILENAME
    metadata = read_run_metadata(metadata_path) if metadata_path.is_file() else {}

    stored_config = metadata.get("config") or {}
    experiment = metadata.get("metadata") or {}
    summary = metadata.get("summary") or {}

    sample_id = (
        (stored_config.get("sample") or {}).get("id")
        or experiment.get("sample_id")
        or summary.get("sample_id")
    )
    from_metadata = bool(sample_id)
    if not sample_id:
        match = _RUN_DIR_PATTERN.match(directory.name)
        sample_id = match["sample"] if match else None

    return RunRecord(
        directory=directory,
        readings_csv=directory / READINGS_FILENAME,
        metadata_path=metadata_path if metadata_path.is_file() else None,
        sample_id=sample_id,
        sample_id_from_metadata=from_metadata,
        started_at=_parse_started_at(metadata, directory.name),
        has_summary=bool(summary),
        mean_pressure_atm=summary.get("mean_pressure_atm"),
        permeability_darcy=summary.get("permeability_darcy"),
        steady_state_reached=summary.get("steady_state_reached"),
        flowmeter=experiment.get("flowmeter"),
        downstream_convention=downstream_convention(stored_config),
    )


def find_runs(runs_dir: str | Path) -> list[RunRecord]:
    """Every ``collect`` run directly under ``runs_dir``, oldest first.

    A run is an immediate subdirectory containing ``readings.csv``. That test
    also excludes the ``klinkenberg_*.yaml`` and ``.png`` result files that sit
    alongside the runs. A run whose sidecar is missing or corrupt is still
    listed -- the CSV is the measurement, and hiding it would be worse than
    showing it with unknown metadata.

    Raises:
        FileNotFoundError: ``runs_dir`` does not exist or is not a directory.
            An empty list would not distinguish "wrong path" from "nothing
            recorded yet", and those need different fixes.
    """
    base = Path(runs_dir)
    if not base.is_dir():
        raise FileNotFoundError(
            f"No such runs directory: {base}. Nothing has been recorded there yet, or "
            "the path is wrong -- check run.output_dir, or pass --runs-dir."
        )

    records = [
        _record_from_directory(child)
        for child in base.iterdir()
        if child.is_dir() and (child / READINGS_FILENAME).is_file()
    ]
    records.sort(key=lambda record: (record.started_at or _UNKNOWN_START, record.name))
    return records


def runs_for_sample(records: Sequence[RunRecord], sample_id: str) -> list[RunRecord]:
    """The records belonging to one core plug.

    Ids that came from a sidecar are compared exactly. Ids recovered from a
    directory name are compared against the sanitised form, because that is all
    the name preserves.
    """
    wanted_safe = safe_sample_id(sample_id)
    return [
        record
        for record in records
        if (
            record.sample_id == sample_id
            if record.sample_id_from_metadata
            else record.sample_id == wanted_safe
        )
    ]


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
    convention = downstream_convention(stored_config)

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
            downstream_convention=convention,
            flow_cm3_s=_summary_float(summary.get("mean_flow_cm3_s")),
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
    flows = [row["flow_cm3_s"] for row in selected if row.get("flow_cm3_s") is not None]

    return KlinkenbergPoint(
        mean_pressure_atm=mean_p,
        apparent_permeability_darcy=mean_k,
        label=label,
        source_path=str(readings_path),
        sample_id=sample_id,
        steady_state=window is not None,
        downstream_convention=convention,
        flow_cm3_s=sum(flows) / len(flows) if flows else None,
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
                "downstream_pressure": describe_convention(point.downstream_convention),
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
