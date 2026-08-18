"""Re-derive a stored run from its **raw** record, under a changed configuration.

Every run's CSV keeps the raw voltages and the raw probe temperature alongside
the derived values, precisely so this is possible: a measurement can be costed
again -- or corrected -- without repeating an experiment that may have taken
fourteen hours.

Nothing here re-reads a device. The raw record is replayed through the same
``SampleProcessor`` / ``PulseProcessor`` and the same summarisers the live run
used, so a reprocessed result is reached by the identical code path and cannot
quietly diverge from what ``collect`` would have produced.

**Three classes of configuration change, and only two of them move the answer.**
That distinction is the point of the command:

``raw``
    The voltages and the probe reading. Never changed by anything here; they
    are the measurement.
``result``
    Inputs the physics consumes -- geometry, calibration constants, the gas,
    atmospheric pressure, the vessel volumes, the fit window. Change one and
    ``k`` itself moves. This is a *correction*, and the original stays on disk
    beside it.
``uncertainty``
    Inputs only the GUM budget consumes -- ``porosity_uncertainty``, every
    ``*.uncertainty`` spec, the coverage probability. ``k`` is untouched and
    only ``U(k)`` moves. This is a *re-costing*, and it is the ordinary reason
    to reach for this command: a calibration certificate arrives, a porosity is
    finally measured with a stated uncertainty, and every run for that plug
    should say so.
``metadata``
    Operator, notes, lithology, display units. Neither moves.

:func:`classify_change` predicts which class a field belongs to, and the
prediction is then **checked against the arithmetic**. A field predicted
``uncertainty`` that turns out to move ``k`` is reported loudly rather than
trusted, because that means either the table below is wrong or the field is
coupled to the physics in a way nobody noticed. The table is advisory; the
recomputation is authoritative.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from gasperm.config import GaspermConfig
from gasperm.models import Reading, RunSummary

logger = logging.getLogger(__name__)

__all__ = [
    "ChangeClass",
    "ConfigChange",
    "RawSample",
    "ReprocessError",
    "ReprocessResult",
    "classify_change",
    "diff_configs",
    "read_raw_samples",
    "rebuild_readings",
    "reprocess_run",
]

ChangeClass = Literal["raw", "result", "uncertainty", "metadata"]


class ReprocessError(ValueError):
    """A stored run that cannot be re-derived, with the reason."""


#: Dotted config prefixes whose values the **physics** consumes. Changing one
#: is a correction to the result, not a re-costing of it.
_RESULT_PREFIXES: tuple[str, ...] = (
    "sample.length",
    "sample.diameter",
    "sample.dimension_unit",
    # Porosity is an input to the Dicker-Smits storage correction, so on a
    # pulse-decay run it moves k. On a steady-state run it moves nothing.
    # Classified as result-bearing because the conservative error is to warn
    # about a change that turns out to be harmless.
    "sample.porosity_fraction",
    "hardware.daq",
    "hardware.pressure_calibration",
    "hardware.pulse_transducers",
    "hardware.flowmeters",
    "hardware.reservoirs",
    "hardware.temperature.units",
    "hardware.temperature.fallback_temperature_c",
    "run.method",
    "run.flowmeter",
    "run.gas.name",
    "run.gas.properties_source",
    "run.gas.fixed_viscosity_cp",
    "run.atmospheric_pressure",
    "run.downstream_pressure",
    "run.averaging_window_s",
    "run.steady_state",
    "run.pulse_decay",
)

#: Prefixes and suffixes that mark a field the **budget** consumes and the
#: physics does not.
_UNCERTAINTY_MARKERS: tuple[str, ...] = ("uncertainty", "_uncertainty")


def classify_change(dotted_key: str) -> ChangeClass:
    """Predict whether changing ``dotted_key`` moves ``k``, ``U(k)``, or neither.

    Uncertainty is tested first and wins: ``sample.porosity_uncertainty`` sits
    under a result-bearing prefix in spelling only, and
    ``hardware.pressure_calibration.inlet.uncertainty.value`` sits under one in
    fact -- yet neither touches a pressure the equation sees.
    """
    parts = dotted_key.split(".")
    if any(
        part == "uncertainty" or part.endswith("_uncertainty") for part in parts
    ):
        return "uncertainty"
    for prefix in _RESULT_PREFIXES:
        if dotted_key == prefix or dotted_key.startswith(prefix + "."):
            return "result"
    return "metadata"


@dataclass(frozen=True)
class ConfigChange:
    """One field that differs between the stored config and the new one."""

    key: str
    before: Any
    after: Any
    predicted: ChangeClass

    def describe(self) -> str:
        return f"{self.key}: {self.before!r} -> {self.after!r}"


def _flatten(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def diff_configs(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> list[ConfigChange]:
    """Every leaf field that differs, each with its predicted class."""
    flat_before, flat_after = _flatten(before), _flatten(after)
    changes = []
    for key in sorted(set(flat_before) | set(flat_after)):
        old, new = flat_before.get(key), flat_after.get(key)
        if old == new:
            continue
        changes.append(
            ConfigChange(key=key, before=old, after=new, predicted=classify_change(key))
        )
    return changes


# --------------------------------------------------------------------------
# The raw record
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RawSample:
    """One sample as the instruments produced it, before any calibration.

    This -- and only this -- is what a reprocess is entitled to start from.
    Everything else in the CSV is a derived value that a changed config may
    legitimately overturn.
    """

    index: int
    elapsed_s: float
    inlet_voltage: float
    outlet_voltage: float
    #: ``None`` for a pulse-decay run, which reads no meter.
    flow_voltage: float | None
    temperature_c: float
    temperature_ok: bool = True
    temperature_stale: bool = False
    temperature_age_s: float | None = None
    temperature_raw: str | None = None


def _optional_float(value: Any) -> float | None:
    text = (value or "").strip() if isinstance(value, str) else value
    if text in (None, ""):
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def read_raw_samples(csv_path: str | Path) -> list[RawSample]:
    """Read the raw voltages and probe readings out of a run's CSV.

    Distinct from :func:`gasperm.storage.read_readings_csv`, which returns the
    *derived* values a detector replay needs. A reprocess must not see those:
    starting from a stored pressure would silently keep the old calibration.

    Raises:
        ReprocessError: the CSV predates raw-voltage storage, or has no rows.
    """
    import csv as csv_module

    path = Path(csv_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv_module.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        required = {"inlet_voltage_V", "outlet_voltage_V", "temperature_C", "elapsed_s"}
        missing = required - fieldnames
        if missing:
            raise ReprocessError(
                f"{path} cannot be reprocessed: it is missing {', '.join(sorted(missing))}. "
                "Only the derived values were stored, so re-deriving them would "
                "start from the old calibration rather than from the measurement."
            )
        samples: list[RawSample] = []
        for position, row in enumerate(reader):
            inlet = _optional_float(row.get("inlet_voltage_V"))
            outlet = _optional_float(row.get("outlet_voltage_V"))
            temperature = _optional_float(row.get("temperature_C"))
            elapsed = _optional_float(row.get("elapsed_s"))
            if None in (inlet, outlet, temperature, elapsed):
                continue
            samples.append(
                RawSample(
                    index=int(_optional_float(row.get("index")) or position),
                    elapsed_s=elapsed,
                    inlet_voltage=inlet,
                    outlet_voltage=outlet,
                    flow_voltage=_optional_float(row.get("flow_voltage_V")),
                    temperature_c=temperature,
                    temperature_ok=str(row.get("temperature_ok", "1")).strip() in ("1", "True", "true"),
                    temperature_stale=str(row.get("temperature_stale", "0")).strip() in ("1", "True", "true"),
                    temperature_age_s=_optional_float(row.get("temperature_age_s")),
                    temperature_raw=(row.get("temperature_raw") or "") or None,
                )
            )
    if not samples:
        raise ReprocessError(f"{path} holds no usable samples.")
    return samples


# --------------------------------------------------------------------------
# Replaying them
# --------------------------------------------------------------------------


def rebuild_readings(
    samples: Sequence[RawSample], config: GaspermConfig, gas_provider
) -> list[Reading]:
    """Drive the raw record back through the live processors.

    The stored voltage columns are keyed by **role** -- inlet, outlet, flow --
    while a processor asks for them by channel name. They are mapped through the
    new config's channels, so a run whose transducer was moved to a different
    analog input still reprocesses correctly: the column records which
    instrument produced the voltage, not which socket it was in.
    """
    from gasperm.acquisition import PulseProcessor, SampleProcessor
    from gasperm.hardware.daq import _pressure_channels
    from gasperm.hardware.temperature import TemperatureSample

    pulse_mode = config.run.method == "pulse_decay"
    processor = (
        PulseProcessor(config, gas_provider)
        if pulse_mode
        else SampleProcessor(config, gas_provider)
    )
    (_, inlet_channel, _), (_, outlet_channel, _) = _pressure_channels(config)
    flow_channel = None if pulse_mode else config.flowmeter.channel

    readings: list[Reading] = []
    for sample in samples:
        voltages = {
            inlet_channel: sample.inlet_voltage,
            outlet_channel: sample.outlet_voltage,
        }
        if flow_channel is not None:
            if sample.flow_voltage is None:
                raise ReprocessError(
                    f"Sample {sample.index} has no flow voltage, but the "
                    f"configuration asks for a steady-state run, which measures flow. "
                    "Was this recorded as a pulse-decay run?"
                )
            voltages[flow_channel] = sample.flow_voltage
        temperature = TemperatureSample(
            temperature_c=sample.temperature_c if sample.temperature_ok else None,
            received_at=None,
            raw_line=sample.temperature_raw,
            stale=sample.temperature_stale,
            age_s=sample.temperature_age_s,
        )
        readings.append(
            processor.process(
                index=sample.index,
                elapsed_s=sample.elapsed_s,
                voltages=voltages,
                temperature=temperature,
            )
        )
    return readings


def _fit_decay(readings: Sequence[Reading], config: GaspermConfig):
    """Re-fit a rebuilt decay, mirroring ``PulseDecayLoop.fit`` exactly."""
    from gasperm.pulse_decay import (
        PulseDecayInputError,
        find_pulse,
        fit_decay_rate,
        fit_window,
    )

    pulse = config.run.pulse_decay
    if len(readings) < 3:
        return None, ["too few samples to fit a decay"]
    times = [r.elapsed_s for r in readings]
    deltas = [r.delta_pressure_atm for r in readings]
    try:
        peak_index, peak_value = find_pulse(times, deltas)
        if peak_value < pulse.min_pulse_pressure_atm:
            return None, ["no pulse above pulse_decay.min_pulse_pressure was found"]
        start, end = fit_window(
            times, deltas, peak_index=peak_index, peak_value=peak_value,
            start_fraction=pulse.fit_start_fraction,
            end_fraction=pulse.fit_end_fraction,
        )
        if end - start < pulse.min_fit_samples:
            return None, [
                f"the fit window holds {end - start} samples, fewer than "
                f"pulse_decay.min_fit_samples ({pulse.min_fit_samples})"
            ]
        return (
            fit_decay_rate(
                times[start:end], deltas[start:end],
                fit_offset=pulse.fit_offset, bin_s=pulse.fit_bin_s,
            ),
            [],
        )
    except PulseDecayInputError as exc:
        return None, [f"could not fit the decay: {exc}"]


def reprocess_run(
    directory: str | Path,
    config: GaspermConfig,
    *,
    gas_provider=None,
    started_at=None,
    ended_at=None,
) -> RunSummary:
    """Re-derive one stored run's summary under ``config``.

    Args:
        directory: The run directory, or its ``readings.csv``.
        config: The configuration to re-derive under -- normally the run's own
            stored snapshot with a field or two changed.
        gas_provider: Property source. Built from ``config`` when omitted.
        started_at: Preserved from the original run, so a re-derived summary
            still carries when the *measurement* happened rather than when it
            was recomputed.

    Raises:
        ReprocessError: the run has no raw record, or cannot be replayed under
            this configuration.
    """
    from gasperm.acquisition import summarize_pulse_decay_run, summarize_run
    from gasperm.gas_properties import build_provider
    from gasperm.steady_state import detect_steady_window
    from gasperm.storage import resolve_run_paths

    readings_path, _ = resolve_run_paths(directory)
    samples = read_raw_samples(readings_path)
    provider = gas_provider if gas_provider is not None else build_provider(config.run.gas)

    readings = rebuild_readings(samples, config, provider)

    if config.run.method == "pulse_decay":
        fit, warnings = _fit_decay(readings, config)
        return summarize_pulse_decay_run(
            readings, config, fit=fit, started_at=started_at, ended_at=ended_at,
            csv_path=str(readings_path), warnings=warnings,
        )

    # Re-detect the steady window rather than reusing the stored one: the
    # criteria are configuration too, and a run reprocessed under different
    # ones must be reduced over the window those criteria actually select.
    window = detect_steady_window(
        (
            {
                "elapsed_s": r.elapsed_s,
                "permeability": r.permeability_darcy,
                "inlet_pressure": r.inlet_pressure_atm,
                "flow": r.flow_cm3_s,
                "temperature": (
                    r.temperature_c + 273.15 if r.temperature_c is not None else None
                ),
            }
            for r in readings
        ),
        config.run.steady_state,
    )
    return summarize_run(
        readings, config, steady_window=window, gas_provider=provider,
        started_at=started_at, ended_at=ended_at, csv_path=str(readings_path),
    )


# --------------------------------------------------------------------------
# Reporting the outcome
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReprocessResult:
    """One run, before and after, plus whether the prediction held."""

    directory: Path
    before: RunSummary | None
    after: RunSummary
    changes: tuple[ConfigChange, ...]

    @property
    def permeability_moved(self) -> bool:
        """Whether ``k`` actually changed, beyond re-derivation noise.

        Not an exact comparison. A steady-state run re-derives to the last bit,
        but a pulse-decay run re-*fits* its exponential, and a nonlinear
        least-squares optimum is not an arithmetic identity -- it reproduces to
        around a part in 10^7. :data:`_MOVED_TOLERANCE` sits above that and far
        below anything a configuration change could do: the smallest interesting
        correction moves ``k`` by a fraction of a percent, four orders larger.
        """
        if self.before is None:
            return False
        return not _close(
            self.before.permeability_darcy, self.after.permeability_darcy
        )

    @property
    def uncertainty_moved(self) -> bool:
        old, new = _expanded(self.before), _expanded(self.after)
        if old is None or new is None:
            return old is not new
        return not _close(old, new)

    @property
    def permeability_ratio(self) -> float:
        if self.before is None or not self.before.permeability_darcy:
            return math.nan
        return self.after.permeability_darcy / self.before.permeability_darcy

    @property
    def predicted_classes(self) -> set[ChangeClass]:
        return {change.predicted for change in self.changes}

    @property
    def surprise(self) -> str:
        """A genuine disagreement between the prediction and the arithmetic.

        The recomputation is authoritative. When it disagrees with the table in
        this module, the table is what needs fixing -- so the disagreement is
        surfaced rather than resolved silently in favour of either one.
        """
        if not self.changes or self.before is None:
            return ""
        classes = self.predicted_classes
        if self.permeability_moved and "result" not in classes:
            listed = ", ".join(sorted(classes)) or "none"
            return (
                f"k moved, but every change was predicted to be {listed}-only. "
                "Either a field is coupled to the physics in a way "
                "reprocess.classify_change does not know about, or the run was "
                "not deterministic. Treat the new value with suspicion."
            )
        return ""

    @property
    def note(self) -> str:
        """A benign explanation for a change that did nothing.

        Not every field applies to every run. Porosity enters the budget only
        through the Dicker-Smits storage correction, so changing its
        uncertainty on a **steady-state** run is a no-op -- correct, but
        indistinguishable from a typo unless it is said out loud.
        """
        if not self.changes or self.before is None:
            return ""
        classes = self.predicted_classes
        if "uncertainty" in classes and not self.uncertainty_moved:
            fields = ", ".join(
                c.key for c in self.changes if c.predicted == "uncertainty"
            )
            return (
                f"U(k) did not move: {fields} is not an input to this run's budget. "
                f"A {self.after.method} run does not use it."
            )
        if "result" in classes and not self.permeability_moved:
            fields = ", ".join(c.key for c in self.changes if c.predicted == "result")
            return (
                f"k did not move: {fields} does not apply to a "
                f"{self.after.method} run."
            )
        return ""


def _expanded(summary: RunSummary | None) -> float | None:
    if summary is None or summary.uncertainty is None:
        return None
    return summary.uncertainty.expanded_uncertainty_darcy


#: Relative threshold for calling a re-derived value "changed". See
#: :attr:`ReprocessResult.permeability_moved` for why it is not zero.
_MOVED_TOLERANCE = 1e-6


def _close(left: float, right: float, tolerance: float = _MOVED_TOLERANCE) -> bool:
    scale = max(abs(left), abs(right))
    if scale == 0.0:
        return True
    return abs(left - right) <= tolerance * scale


def summarise_changes(changes: Iterable[ConfigChange]) -> dict[ChangeClass, list[ConfigChange]]:
    """Group changes by predicted class, for reporting."""
    grouped: dict[ChangeClass, list[ConfigChange]] = {}
    for change in changes:
        grouped.setdefault(change.predicted, []).append(change)
    return grouped
