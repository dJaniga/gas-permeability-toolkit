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
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence, TypeVar

from gasperm.config import GaspermConfig
from gasperm.models import Reading, RunSummary

logger = logging.getLogger(__name__)

__all__ = [
    "ChangeClass",
    "ConfigChange",
    "RawSample",
    "ReprocessError",
    "ReprocessJob",
    "ReprocessResult",
    "VerifyReport",
    "classify_change",
    "diff_configs",
    "read_raw_samples",
    "rebuild_readings",
    "reprocess_batch",
    "reprocess_run",
    "resolve_workers",
    "verify_batch",
    "verify_run",
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

#: How far two window bounds may sit apart and still be the same window, in
#: seconds. The CSV stores ``elapsed_s`` to four decimals, so a replayed bound
#: is up to 5e-5 s from the full-precision one the live run held; anything
#: larger than that means a different sample, not a different rounding. Kept
#: well under one sample interval at any rate this rig runs.
_WINDOW_TOLERANCE_S = 2e-4


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


# --------------------------------------------------------------------------
# Verifying that a replay reproduces its original
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifyReport:
    """Where a no-change replay of one run agrees with its stored summary, and
    where it does not.

    A no-change reprocess **must** reproduce the original. When it does not,
    the useful question is not "by how much" but "at which stage", because the
    three stages fail for entirely different reasons:

    ``sample_drift``
        The per-sample derivation. Non-zero means the replay is computing
        different pressures, flows or viscosities from the same raw voltages --
        a calibration or property-lookup problem.
    ``window``
        Which samples get averaged. The per-sample values can be exact while
        the reduction still moves, because the two paths disagreed about where
        the measurement was.
    ``summary``
        What came out -- ``k`` and ``U(k)``. Moves if either of the above did,
        and on its own means the reduction arithmetic differs.

    Every numeric verdict is relative to :attr:`tolerance`, which is carried on
    the report rather than read from a module constant: a pass or a fail is
    meaningless without the threshold it was judged at, and anything reporting
    one has to be able to state the other.
    """

    directory: Path
    #: Largest relative difference between the stored per-sample permeability
    #: and the rebuilt one. ``None`` when the CSV stored none to compare.
    sample_drift: float | None
    #: ``(stored, replayed)`` steady-window bounds in seconds, or ``None``
    #: where a run has no window (pulse decay, or one that never settled).
    stored_window: tuple[float, float] | None
    replayed_window: tuple[float, float] | None
    stored_permeability_darcy: float | None
    replayed_permeability_darcy: float
    stored_expanded_darcy: float | None
    replayed_expanded_darcy: float | None
    #: Relative difference treated as reproduction. Not applied to the window,
    #: which is compared in **seconds** against the CSV's stored precision --
    #: a different question, and not one an operator should be tuning.
    tolerance: float = _MOVED_TOLERANCE

    @property
    def permeability_ratio(self) -> float | None:
        if not self.stored_permeability_darcy:
            return None
        return self.replayed_permeability_darcy / self.stored_permeability_darcy

    @property
    def uncertainty_ratio(self) -> float | None:
        if not self.stored_expanded_darcy or self.replayed_expanded_darcy is None:
            return None
        return self.replayed_expanded_darcy / self.stored_expanded_darcy

    @property
    def samples_agree(self) -> bool:
        return self.sample_drift is None or self.sample_drift <= self.tolerance

    @property
    def uncertainty_agrees(self) -> bool:
        """Whether ``U(k)`` reproduced.

        Checked separately from ``k`` because the two move independently: a
        re-costing bug leaves ``k`` exactly where it was and moves only the
        budget, which a check on ``k`` alone would pass.
        """
        if self.stored_expanded_darcy is None or self.replayed_expanded_darcy is None:
            return self.stored_expanded_darcy == self.replayed_expanded_darcy
        return _close(
            self.stored_expanded_darcy, self.replayed_expanded_darcy, self.tolerance
        )

    @property
    def windows_agree(self) -> bool:
        """Whether the two windows cover the same span.

        Compared with an **absolute** tolerance in seconds, not a relative one:
        a stored bound is a full-precision in-memory float while the replayed
        one comes back through the CSV's four-decimal ``elapsed_s``, so they
        differ by up to 5e-5 s for reasons that have nothing to do with the
        measurement -- and near ``t = 0`` a relative tolerance turns that into
        a false alarm.
        """
        if self.stored_window is None or self.replayed_window is None:
            return self.stored_window == self.replayed_window
        return all(
            abs(a - b) <= _WINDOW_TOLERANCE_S
            for a, b in zip(self.stored_window, self.replayed_window)
        )

    @property
    def reproduces(self) -> bool:
        ratio = self.permeability_ratio
        return (
            self.samples_agree
            and self.windows_agree
            and self.uncertainty_agrees
            and (ratio is None or abs(ratio - 1.0) <= self.tolerance)
        )

    def diagnosis(self) -> str:
        """One line naming the stage that broke, for a run that did not reproduce."""
        if self.reproduces:
            return "reproduces its stored result"
        if not self.samples_agree:
            return (
                "the per-sample derivation differs -- the same voltages are "
                "producing different values, so the calibration or a property "
                "lookup is not being reproduced"
            )
        if not self.windows_agree:
            return (
                "the per-sample values are exact but the averaged window is not "
                "the stored one, so the reduction covers different samples"
            )
        ratio = self.permeability_ratio
        if ratio is not None and abs(ratio - 1.0) > self.tolerance:
            return (
                "samples and window both match, so the reduction arithmetic itself "
                "differs -- or the stored summary was written by different code"
            )
        return (
            "k reproduces but U(k) does not, so the budget is being rebuilt from "
            "something the raw record does not pin down"
        )


def verify_run(
    directory: str | Path,
    config: GaspermConfig,
    *,
    tolerance: float = _MOVED_TOLERANCE,
) -> VerifyReport:
    """Replay one run with **no change** and compare it against what is stored.

    This is the invariant every other use of ``reprocess`` rests on: if a no-op
    replay moves the answer, no reported change can be attributed to the field
    that was actually edited. It is separated from :func:`reprocess_run` so the
    check can be run over a whole directory without writing anything.

    Args:
        directory: The run directory, or its ``readings.csv``.
        config: The run's **own** stored snapshot -- verifying against anything
            else is a different question, and not this one.
        tolerance: Relative difference treated as reproduction. The default is
            tight enough to catch a real defect and loose enough to absorb a
            pulse-decay refit, which lands within about 1e-7. Loosen it to hunt
            outliers on a rig whose replay is known to differ slightly; the
            report carries the value so a verdict is never quoted without it.

    Raises:
        ValueError: ``tolerance`` is not positive. A zero tolerance would fail
            every run on the last bit of a float and report nothing useful.
    """
    if tolerance <= 0.0:
        raise ValueError(
            f"tolerance must be positive, got {tolerance!r}. Floating-point "
            "arithmetic does not reproduce exactly across a CSV round trip, so "
            "a zero tolerance fails every run and localises nothing."
        )
    import csv as csv_module

    from gasperm.gas_properties import build_provider
    from gasperm.storage import resolve_run_paths, summary_from_run

    readings_path, _ = resolve_run_paths(directory)
    stored = summary_from_run(directory)
    samples = read_raw_samples(readings_path)
    provider = build_provider(config.run.gas)
    readings = rebuild_readings(samples, config, provider)

    # Per-sample: the stored derived column against the rebuilt one, keyed by
    # index so a skipped row cannot silently shift the comparison by one.
    rebuilt = {r.index: r.permeability_darcy for r in readings}
    drift: float | None = None
    with Path(readings_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv_module.DictReader(handle):
            was = _optional_float(row.get("permeability_D"))
            index = _optional_float(row.get("index"))
            if was is None or index is None or not was:
                continue
            now = rebuilt.get(int(index))
            if now is None:
                continue
            drift = max(drift or 0.0, abs(now / was - 1.0))

    replayed = reprocess_run(
        directory, config,
        started_at=stored.started_at if stored else None,
        ended_at=stored.ended_at if stored else None,
    )

    def bounds(summary):
        window = getattr(summary, "steady_state_window", None) if summary else None
        if window is None:
            return None
        return (window.start_elapsed_s, window.end_elapsed_s)

    return VerifyReport(
        directory=Path(readings_path).parent,
        sample_drift=drift,
        stored_window=bounds(stored),
        replayed_window=bounds(replayed),
        stored_permeability_darcy=stored.permeability_darcy if stored else None,
        replayed_permeability_darcy=replayed.permeability_darcy,
        stored_expanded_darcy=_expanded(stored),
        replayed_expanded_darcy=_expanded(replayed),
        tolerance=tolerance,
    )


# --------------------------------------------------------------------------
# Doing it to a whole campaign at once
# --------------------------------------------------------------------------
#
# A reprocess is dominated by :func:`rebuild_readings`, and that cost is
# per-*sample*: every reading is pushed back through the live processor, which
# looks the gas properties up at that reading's own (T, P) and, for a pulse
# run, solves the storage equation for ``theta_1``. A fourteen-hour run at
# 10 Hz is half a million samples, so one run takes tens of seconds and a
# rig-level ``--all`` over a season's work takes an hour.
#
# The saving grace is that **runs do not interact**. Each re-derives from its
# own stored snapshot -- that is what makes ``--all`` safe in the first place
# (see the module docstring) -- so the batch is embarrassingly parallel at the
# granularity of one run.
#
# Processes, not threads: the time is spent inside CoolProp's ``PropsSI`` and
# SciPy's ``brentq``, neither of which releases the GIL for the scalar calls
# made here, so threads would serialise onto exactly the same core. Nothing is
# shared between workers -- a job carries its own config and its own directory,
# and the worker builds its own property provider -- so there is no state to
# guard and no ordering to preserve inside a run.
#
# Two invariants the parallel path must not break, because both would fail
# silently:
#
# - **The results stay in the caller's order.** They are reported as a table
#   keyed by run, and completion order is whatever the scheduler decided.
# - **A worker's exception is returned, not raised.** The serial loop reported
#   a run that could not be replayed as a skip and carried on with the rest;
#   losing forty good runs to one unreadable CSV would be a regression.
#
# And one thing it has to get right to be worth doing: the **longest run is
# submitted first**. Runs on one bench differ by two orders of magnitude in
# length, so a fourteen-hour decay handed out last leaves every worker idle
# waiting on it, and the batch takes as long as that one run plus everything
# queued ahead of it.

#: Raw record below which the batch stays in this process, in bytes summed over
#: every run in it.
#:
#: A worker is not free: it pays a fresh interpreter start plus CoolProp and
#: SciPy imports, a second or two on Windows. Run *count* is the wrong thing to
#: weigh that against, because runs differ by two orders of magnitude in length
#: -- three fourteen-hour decays are worth spreading and thirty short bursts are
#: not. CSV size is the honest proxy, since the cost is per sample and a sample
#: is a row. At roughly three seconds of re-derivation per megabyte, this is a
#: batch that would take about half a minute serially -- comfortably more than
#: the workers cost to start.
_PARALLEL_MIN_BYTES = 8 * 1024 * 1024

_Payload = TypeVar("_Payload")
_Outcome = TypeVar("_Outcome")


@dataclass(frozen=True)
class ReprocessJob:
    """One run to re-derive, and the configuration to re-derive it under.

    Carries everything a worker needs, because a worker shares nothing with the
    parent: the config is the run's **own** snapshot with the requested edits
    already applied, and the timestamps are the original measurement's, so a
    re-derived summary still says when the experiment happened rather than when
    it was recomputed.
    """

    directory: Path
    config: GaspermConfig
    started_at: datetime | None = None
    ended_at: datetime | None = None


def _record_size(job: ReprocessJob) -> int:
    """Size of the CSV this job will replay, as a stand-in for what it will cost.

    The re-derivation is per sample and a sample is a row, so bytes track work
    closely enough for the two things this is used for: deciding whether workers
    are worth starting, and deciding what to start first.

    A run whose size cannot be read counts zero rather than raising. This is a
    scheduling hint; a missing or unreadable record is a failure to report from
    the replay, with a message about the record rather than about a stat call.
    """
    from gasperm.storage import resolve_run_paths

    try:
        readings, _ = resolve_run_paths(job.directory)
        return Path(readings).stat().st_size
    except (OSError, ValueError):
        return 0


def resolve_workers(requested: int | None, jobs: Sequence[ReprocessJob]) -> int:
    """How many processes to use for this batch.

    ``requested`` is the operator's ``--jobs``; ``None`` means decide. There is
    no reason to hold a core back -- nothing else is running and the answer is
    being waited on -- so the default is one worker per CPU, capped at the
    number of runs, and a batch with less raw record than
    ``_PARALLEL_MIN_BYTES`` stays serial rather than spend longer starting
    workers than re-deriving.

    A **negative** ``requested`` counts back from the CPU count, the convention
    joblib and scikit-learn use and the one people arrive with: ``-1`` is every
    CPU, ``-2`` every CPU but one -- the useful form, since it leaves the
    machine usable while the batch runs. Below one worker it clamps rather than
    raising: ``-32`` on an eight-core box means "as few as possible", and
    refusing it would be pedantry about a number nobody typed deliberately.

    An explicit ``--jobs`` is honoured whatever the size: someone who asked for
    workers gets them, and someone who asked for one gets a single process and a
    readable traceback. It is also the lever for **memory**, which is the one
    resource this cannot size itself against: a worker holds its whole run in
    memory as ``Reading`` objects, on the order of twenty times the CSV, so a
    rig whose records run to tens of megabytes wants fewer workers than it has
    cores. There is no portable way to ask how much memory is free, so the
    default sizes against CPUs and the operator lowers ``-j`` if the machine
    complains.

    Raises:
        ValueError: ``requested`` is zero. Every other integer names a worker
            count; zero names none, and no workers is not a slower reprocess,
            it is no reprocess.
    """
    cpus = os.cpu_count() or 1
    if requested == 0:
        raise ValueError(
            "jobs must not be 0. A positive count is that many worker "
            f"processes; a negative one counts back from the {cpus} CPU(s), so "
            "-1 is all of them and -2 all but one."
        )
    count = len(jobs)
    if count <= 0:
        return 1
    if requested is not None:
        resolved = requested if requested > 0 else max(1, cpus + 1 + requested)
        return min(resolved, count)
    if count < 2 or sum(_record_size(job) for job in jobs) < _PARALLEL_MIN_BYTES:
        return 1
    return max(1, min(cpus, count))


def _map_jobs(
    worker: Callable[[_Payload], _Outcome],
    payloads: Sequence[_Payload],
    *,
    workers: int,
    order: Sequence[int] | None = None,
    on_done: Callable[[int, int], None] | None = None,
) -> list[_Outcome | BaseException]:
    """Apply ``worker`` to each payload, returning results **in payload order**.

    Each entry is either the worker's return value or the exception it raised;
    deciding which exceptions are survivable belongs to the caller, since that
    is the layer that knows what it promised the operator.

    ``order`` is the sequence to *submit* in, longest job first. It changes when
    the batch finishes, not what it produces: a fourteen-hour decay handed out
    last would still be running long after every short burst had finished and
    the pool had gone idle, so the batch would take as long as its worst run
    plus everything queued ahead of it. Results still come back in payload
    order.

    Falls back to the serial path if the pool cannot be started or breaks
    part-way, finishing only the runs that had not come back. A batch that takes
    longer is a much better outcome than a batch that dies, and a broken pool
    means a worker was killed -- which says nothing about whether the work can
    be done.
    """
    results: list[Any] = [None] * len(payloads)
    done = 0

    def record(index: int, outcome: Any) -> None:
        nonlocal done
        results[index] = outcome
        done += 1
        if on_done is not None:
            on_done(done, len(payloads))

    def serially(indices: Iterable[int]) -> None:
        for index in indices:
            try:
                record(index, worker(payloads[index]))
            except Exception as exc:  # noqa: BLE001 -- handed back to the caller
                record(index, exc)

    if workers <= 1 or len(payloads) <= 1:
        serially(range(len(payloads)))
        return results

    pending = set(range(len(payloads)))
    submission = range(len(payloads)) if order is None else order
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(worker, payloads[index]): index for index in submission
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    record(index, future.result())
                except Exception as exc:  # noqa: BLE001 -- handed back to the caller
                    record(index, exc)
                pending.discard(index)
    except Exception as exc:  # noqa: BLE001 -- the pool failed, not a job
        logger.warning(
            "Parallel reprocessing failed (%s); finishing the remaining %d "
            "run(s) in this process.", exc, len(pending),
        )
        serially(sorted(pending))
    return results


def _longest_first(jobs: Sequence[ReprocessJob]) -> list[int]:
    """Job indices ordered by descending raw record -- the order to submit in."""
    return sorted(range(len(jobs)), key=lambda index: -_record_size(jobs[index]))


def _reprocess_job(job: ReprocessJob) -> RunSummary:
    """Worker body. Module-level, so it is picklable under ``spawn``."""
    return reprocess_run(
        job.directory, job.config,
        started_at=job.started_at, ended_at=job.ended_at,
    )


def _verify_job(payload: tuple[ReprocessJob, float]) -> VerifyReport:
    """Worker body for ``--verify``. See :func:`_reprocess_job`."""
    job, tolerance = payload
    return verify_run(job.directory, job.config, tolerance=tolerance)


def reprocess_batch(
    jobs: Sequence[ReprocessJob],
    *,
    workers: int | None = None,
    on_done: Callable[[int, int], None] | None = None,
) -> list[RunSummary | BaseException]:
    """Re-derive many runs at once, one worker process per core by default.

    Args:
        jobs: What to re-derive, each carrying its own configuration.
        workers: Process count, or ``None`` to decide from the CPU count and
            the size of the batch. ``1`` keeps everything in this process.
        on_done: Called ``(completed, total)`` as each run finishes, for a
            progress line. Completion order is not payload order, so it says
            how many are done and never which.

    Returns:
        One entry per job, **in the order given**: the re-derived summary, or
        the exception that run raised.
    """
    jobs = list(jobs)
    return _map_jobs(
        _reprocess_job, jobs,
        workers=resolve_workers(workers, jobs),
        order=_longest_first(jobs), on_done=on_done,
    )


def verify_batch(
    jobs: Sequence[ReprocessJob],
    *,
    tolerance: float = _MOVED_TOLERANCE,
    workers: int | None = None,
    on_done: Callable[[int, int], None] | None = None,
) -> list[VerifyReport | BaseException]:
    """Verify many runs at once. The parallel form of :func:`verify_run`.

    Each job's ``config`` must be that run's **own** stored snapshot -- verifying
    against anything else is a different question, and not this one. Returns one
    entry per job in the order given: the report, or the exception it raised.
    """
    jobs = list(jobs)
    return _map_jobs(
        _verify_job, [(job, tolerance) for job in jobs],
        workers=resolve_workers(workers, jobs),
        order=_longest_first(jobs), on_done=on_done,
    )
