"""Automatic steady-state detection.

Darcy's law describes *steady* flow. A permeability computed while the rig is
still filling, while the sleeve is still seating, or while the regulator is
still hunting describes the transient, not the rock -- and it will happily
produce a confident-looking number. So detection is not a convenience here: it
is what makes a reported permeability representative.

Two criteria are applied to every monitored signal, over a trailing window:

**Scatter** -- the coefficient of variation ``s / |mean|`` must be at or below
``relative_stddev_tolerance``. Catches noise and instability.

**Drift** -- the fractional change an ordinary-least-squares line predicts
across the window, ``|slope| * window / |mean|``, must be at or below
``relative_drift_tolerance``. This is the criterion that matters: a slowly
ramping signal has small scatter inside *any* short window and would pass a
scatter-only test forever, which is the classic way these rigs report a
too-high permeability from a still-pressurising system.

Optionally a Student-t test on the slope is added, so that a drift small in
absolute terms but statistically unambiguous also fails.

Windows are **non-overlapping**: one evaluation per ``window_s`` of elapsed
time. Requiring N consecutive passes therefore means N genuinely independent
windows, not N adjacent samples that necessarily agree.

Hardware-free and directly testable against synthetic signals.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from typing import Iterable, Mapping, Sequence

from gasperm import units
from gasperm.config.run import SteadyStateConfig
from gasperm.models import Reading, SignalStability, SteadyStateStatus, SteadyStateWindow

__all__ = [
    "SIGNAL_LABELS",
    "assess_signal",
    "signals_from_reading",
    "SteadyStateDetector",
    "detect_steady_window",
]

#: Human-readable names for the console.
SIGNAL_LABELS: dict[str, str] = {
    "permeability": "permeability",
    "inlet_pressure": "inlet pressure",
    "flow": "flow rate",
    "temperature": "temperature",
}


def signals_from_reading(reading: Reading) -> dict[str, float | None]:
    """Extract the monitored signals from a reading, in internal units.

    Temperature is taken in **kelvin** rather than degC so that a run near
    0 degC does not divide by a near-zero mean when the relative criteria are
    evaluated.
    """
    return {
        "permeability": reading.permeability_darcy,
        "inlet_pressure": reading.inlet_pressure_atm,
        "flow": reading.flow_cm3_s,
        "temperature": units.celsius_to_kelvin(reading.temperature_c),
    }


def _ols_slope(times: Sequence[float], values: Sequence[float]) -> tuple[float, float | None, float | None]:
    """``(slope, t_statistic, p_value)`` for ``values`` against ``times``.

    Uses a plain closed-form OLS rather than scipy so this module stays cheap
    enough to call on every sample of a fast run.
    """
    n = len(times)
    mean_t = statistics.fmean(times)
    mean_v = statistics.fmean(values)
    s_tt = sum((t - mean_t) ** 2 for t in times)
    if s_tt <= 0.0:
        return 0.0, None, None
    s_tv = sum((t - mean_t) * (v - mean_v) for t, v in zip(times, values))
    slope = s_tv / s_tt

    if n <= 2:
        return slope, None, None

    intercept = mean_v - slope * mean_t
    residual_ss = sum((v - (intercept + slope * t)) ** 2 for t, v in zip(times, values))
    dof = n - 2
    residual_variance = residual_ss / dof
    if residual_variance <= 0.0:
        # A perfect fit: the slope is exactly determined. Zero slope means no
        # evidence of drift; any other slope is infinitely significant.
        return slope, (0.0 if slope == 0.0 else math.inf), (1.0 if slope == 0.0 else 0.0)
    slope_stderr = math.sqrt(residual_variance / s_tt)
    t_statistic = slope / slope_stderr
    p_value = _two_sided_t_p_value(abs(t_statistic), dof)
    return slope, t_statistic, p_value


def _two_sided_t_p_value(t_abs: float, dof: int) -> float:
    """Two-sided p-value for Student's t, via the regularised incomplete beta."""
    if not math.isfinite(t_abs):
        return 0.0
    x = dof / (dof + t_abs * t_abs)
    return _betainc(dof / 2.0, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b), by continued fraction.

    Lifted from the standard Lentz algorithm. Present so that a slope
    significance test does not drag scipy into the acquisition loop.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(log_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_cf(a, b, x) / a
    return 1.0 - front * _beta_cf(b, a, 1.0 - x) / b


def _beta_cf(a: float, b: float, x: float, max_iterations: int = 200) -> float:
    tiny = 1e-30
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for m in range(1, max_iterations + 1):
        two_m = 2 * m
        numerator = m * (b - m) * x / ((a + two_m - 1.0) * (a + two_m))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c

        numerator = -(a + m) * (a + b + m) * x / ((a + two_m) * (a + two_m + 1.0))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return result


def assess_signal(
    name: str,
    times: Sequence[float],
    values: Sequence[float],
    config: SteadyStateConfig,
) -> SignalStability:
    """Evaluate one signal's stationarity over one window.

    Args:
        name: Signal key, used in the report.
        times: Elapsed seconds, ascending.
        values: The signal, in internal units.
        config: Tolerances and options.

    Returns:
        Diagnostics, including which criteria failed and by how much.
    """
    n = len(values)
    if n < config.min_samples:
        return SignalStability(
            name=name,
            sample_count=n,
            mean=statistics.fmean(values) if values else 0.0,
            stddev=0.0,
            relative_stddev=math.inf,
            slope_per_s=0.0,
            relative_drift=math.inf,
            passed=False,
            failures=[f"only {n} samples in the window, need {config.min_samples}"],
        )

    mean = statistics.fmean(values)
    stddev = statistics.stdev(values)
    span = times[-1] - times[0]
    slope, t_statistic, p_value = _ols_slope(times, values)

    failures: list[str] = []
    if mean == 0.0:
        # A dead channel is not a steady measurement, it is no measurement.
        relative_stddev = math.inf
        relative_drift = math.inf
        failures.append("mean is zero -- the signal looks dead")
    else:
        relative_stddev = stddev / abs(mean)
        relative_drift = abs(slope) * span / abs(mean)
        if relative_stddev > config.relative_stddev_tolerance:
            failures.append(
                f"scatter {relative_stddev:.3%} > {config.relative_stddev_tolerance:.3%}"
            )
        if relative_drift > config.relative_drift_tolerance:
            failures.append(
                f"drift {relative_drift:.3%} over {span:.1f} s > "
                f"{config.relative_drift_tolerance:.3%}"
            )
        if (
            config.slope_significance is not None
            and p_value is not None
            and p_value < config.slope_significance
        ):
            failures.append(
                f"slope is significant (p = {p_value:.4f} < {config.slope_significance})"
            )

    return SignalStability(
        name=name,
        sample_count=n,
        mean=mean,
        stddev=stddev,
        relative_stddev=relative_stddev,
        slope_per_s=slope,
        relative_drift=relative_drift,
        slope_t_statistic=t_statistic,
        slope_p_value=p_value,
        passed=not failures,
        failures=failures,
    )


class SteadyStateDetector:
    """Streaming detector: feed it samples, ask whether the rig has settled.

    Used identically live (from the acquisition loop) and offline (replaying a
    stored run), so ``collect`` and ``klinkenberg`` can never disagree about
    what steady state means.
    """

    def __init__(self, config: SteadyStateConfig) -> None:
        self.config = config
        self._buffer: deque[tuple[float, dict[str, float | None]]] = deque()
        self._consecutive_passes = 0
        self._next_evaluation_at: float | None = None
        self._steady_since: float | None = None
        self._last_status: SteadyStateStatus | None = None
        self._first_reached_at: float | None = None

    # -- state ------------------------------------------------------------

    @property
    def is_steady(self) -> bool:
        """Whether the most recent evaluation left the rig steady."""
        return self._consecutive_passes >= self.config.required_windows

    @property
    def steady_since_elapsed_s(self) -> float | None:
        """Start of the *current* uninterrupted steady stretch, if any.

        Cleared if the rig destabilises, so a run that wobbles late does not
        get credit for an earlier plateau.
        """
        return self._steady_since if self.is_steady else None

    @property
    def status(self) -> SteadyStateStatus:
        """The most recent status, or a not-yet-evaluated placeholder."""
        if self._last_status is not None:
            return self._last_status
        return SteadyStateStatus(
            is_steady=False,
            consecutive_passes=0,
            required_passes=self.config.required_windows,
            window_s=self.config.window_s,
            elapsed_s=0.0,
            summary="waiting for the first window",
        )

    # -- feeding ----------------------------------------------------------

    def update(self, elapsed_s: float, values: Mapping[str, float | None]) -> SteadyStateStatus:
        """Add one sample and, when a window closes, re-evaluate.

        Args:
            elapsed_s: Seconds since the run started.
            values: Signal name -> value, in internal units. ``None`` marks a
                signal that could not be computed for this sample.

        Returns:
            The current status. Between window boundaries this is the previous
            evaluation with the elapsed time updated, so callers can display it
            every sample without re-testing.
        """
        if not self.config.enabled:
            return SteadyStateStatus(
                is_steady=False,
                consecutive_passes=0,
                required_passes=self.config.required_windows,
                window_s=self.config.window_s,
                elapsed_s=elapsed_s,
                summary="steady-state detection disabled",
            )

        self._buffer.append((elapsed_s, dict(values)))
        cutoff = elapsed_s - self.config.window_s
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

        if self._next_evaluation_at is None:
            self._next_evaluation_at = self.config.settling_time_s + self.config.window_s

        if elapsed_s < self._next_evaluation_at:
            status = self._carry_forward(elapsed_s)
            self._last_status = status
            return status

        # A window has closed: evaluate it, then schedule the next
        # non-overlapping one.
        self._next_evaluation_at += self.config.window_s
        status = self._evaluate(elapsed_s)
        self._last_status = status
        return status

    def _carry_forward(self, elapsed_s: float) -> SteadyStateStatus:
        previous = self._last_status
        summary = previous.summary if previous else "collecting the first window"
        return SteadyStateStatus(
            is_steady=self.is_steady,
            consecutive_passes=self._consecutive_passes,
            required_passes=self.config.required_windows,
            window_s=self.config.window_s,
            elapsed_s=elapsed_s,
            signals=list(previous.signals) if previous else [],
            reached_at_elapsed_s=self.steady_since_elapsed_s,
            summary=summary,
        )

    def _evaluate(self, elapsed_s: float) -> SteadyStateStatus:
        times = [t for t, _ in self._buffer]
        reports: list[SignalStability] = []
        all_passed = True

        for name in self.config.signals:
            series = [row.get(name) for _, row in self._buffer]
            usable = [(t, v) for t, v in zip(times, series) if v is not None and math.isfinite(v)]
            if len(usable) < len(series):
                missing = len(series) - len(usable)
                reports.append(
                    SignalStability(
                        name=name,
                        sample_count=len(usable),
                        mean=0.0,
                        stddev=0.0,
                        relative_stddev=math.inf,
                        slope_per_s=0.0,
                        relative_drift=math.inf,
                        passed=False,
                        failures=[
                            f"{missing} of {len(series)} samples had no usable value"
                        ],
                    )
                )
                all_passed = False
                continue
            report = assess_signal(
                name, [t for t, _ in usable], [v for _, v in usable], self.config
            )
            reports.append(report)
            all_passed = all_passed and report.passed

        if all_passed:
            self._consecutive_passes += 1
            if self._steady_since is None:
                # Credit the window that started this stretch.
                self._steady_since = max(0.0, elapsed_s - self.config.window_s)
        else:
            self._consecutive_passes = 0
            self._steady_since = None

        if self.is_steady and self._first_reached_at is None:
            self._first_reached_at = elapsed_s

        if all_passed and self.is_steady:
            summary = f"steady ({self._consecutive_passes}/{self.config.required_windows})"
        elif all_passed:
            summary = f"settling ({self._consecutive_passes}/{self.config.required_windows})"
        else:
            worst = next((r for r in reports if not r.passed), None)
            detail = f"{SIGNAL_LABELS.get(worst.name, worst.name)}: {worst.failures[0]}" if worst else ""
            summary = f"not steady -- {detail}" if detail else "not steady"

        return SteadyStateStatus(
            is_steady=self.is_steady,
            consecutive_passes=self._consecutive_passes,
            required_passes=self.config.required_windows,
            window_s=self.config.window_s,
            elapsed_s=elapsed_s,
            signals=reports,
            reached_at_elapsed_s=self.steady_since_elapsed_s,
            summary=summary,
        )


def detect_steady_window(
    samples: Iterable[Mapping[str, float | None]],
    config: SteadyStateConfig,
    *,
    time_key: str = "elapsed_s",
) -> SteadyStateWindow | None:
    """Replay a completed run and return its final steady stretch.

    Args:
        samples: Rows with an elapsed-time key and the monitored signals.
        config: The same criteria the live run used.
        time_key: Name of the elapsed-time field.

    Returns:
        The steady window, or ``None`` if the run never settled.
    """
    detector = SteadyStateDetector(config)
    rows = list(samples)
    steady_since: float | None = None
    steady_until: float | None = None

    for row in rows:
        elapsed = row.get(time_key)
        if elapsed is None:
            continue
        signals = {name: row.get(name) for name in config.signals}
        detector.update(float(elapsed), signals)
        # The plateau, captured while the detector is *in* it. Reading
        # `steady_since_elapsed_s` after the final row instead loses the whole
        # window for a run that was still drifting when it stopped -- the
        # detector has left steady state by then and reports nothing. That run
        # has a perfectly good plateau, it is what the live loop reports, and
        # summarising the drifting tail in its place moves k.
        if detector.is_steady:
            steady_since = detector.steady_since_elapsed_s
            steady_until = float(elapsed)

    if steady_since is None or steady_until is None:
        return None

    indices = [
        index
        for index, row in enumerate(rows)
        if row.get(time_key) is not None
        and steady_since <= float(row[time_key]) <= steady_until
        # Samples that yielded no permeability are trimmed from the ends the
        # same way the live loop trims them, so a replay reduces over exactly
        # the samples the original run reduced over.
        and row.get("permeability", 0.0) is not None
    ]
    if not indices:
        return None
    start, end = indices[0], indices[-1]
    return SteadyStateWindow(
        start_elapsed_s=float(rows[start][time_key]),
        end_elapsed_s=float(rows[end][time_key]),
        sample_count=len(indices),
        start_index=start,
        end_index=end,
    )
