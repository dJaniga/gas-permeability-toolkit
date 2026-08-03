"""Steady-state detection against synthetic signals with known behaviour.

The cases that matter are the ones a naive detector gets wrong: a slow ramp
(small scatter in any window, but not steady) and a noisy plateau (large
scatter, but genuinely steady on average).
"""

from __future__ import annotations

import math
import random

import pytest

from gasperm.config.run import SteadyStateConfig
from gasperm.steady_state import (
    SteadyStateDetector,
    assess_signal,
    detect_steady_window,
    signals_from_reading,
)


def criteria(**overrides) -> SteadyStateConfig:
    """Fast criteria suited to synthetic 1 Hz series."""
    base = dict(
        window_s=10.0,
        required_windows=2,
        min_samples=5,
        relative_stddev_tolerance=0.02,
        relative_drift_tolerance=0.01,
        signals=["permeability"],
    )
    base.update(overrides)
    return SteadyStateConfig(**base)


def series(values, start=0.0, step=1.0):
    """``(times, values)`` at a fixed cadence."""
    return [start + i * step for i in range(len(values))], list(values)


class TestAssessSignal:
    def test_a_flat_signal_passes(self):
        times, values = series([5.0] * 20)
        report = assess_signal("permeability", times, values, criteria())
        assert report.passed
        assert report.relative_stddev == pytest.approx(0.0)
        assert report.relative_drift == pytest.approx(0.0)

    def test_a_slow_ramp_fails_on_drift_despite_tiny_scatter(self):
        """The case a scatter-only test passes forever."""
        # 2% rise across the window: the coefficient of variation is only
        # ~0.6%, well inside the scatter tolerance.
        times, values = series([5.0 * (1.0 + 0.02 * i / 19) for i in range(20)])
        report = assess_signal("permeability", times, values, criteria())
        assert report.relative_stddev < criteria().relative_stddev_tolerance
        assert not report.passed
        assert any("drift" in f for f in report.failures)

    def test_a_noisy_plateau_fails_on_scatter(self):
        rng = random.Random(7)
        times, values = series([5.0 + rng.uniform(-0.5, 0.5) for _ in range(40)])
        report = assess_signal("permeability", times, values, criteria())
        assert not report.passed
        assert any("scatter" in f for f in report.failures)

    def test_too_few_samples_fails_explicitly(self):
        times, values = series([5.0, 5.0])
        report = assess_signal("permeability", times, values, criteria(min_samples=5))
        assert not report.passed
        assert "only 2 samples" in report.failures[0]

    def test_a_dead_channel_never_counts_as_steady(self):
        """Zero flow is perfectly flat, and perfectly not a measurement."""
        times, values = series([0.0] * 20)
        report = assess_signal("flow", times, values, criteria())
        assert not report.passed
        assert any("dead" in f for f in report.failures)

    def test_slope_significance_can_reject_a_small_but_certain_drift(self):
        # A tiny, perfectly linear ramp: inside the drift bound, but the slope
        # is unambiguous.
        times, values = series([5.0 * (1.0 + 0.002 * i / 19) for i in range(20)])
        lenient = criteria(relative_drift_tolerance=0.05)
        assert assess_signal("permeability", times, values, lenient).passed

        strict = criteria(relative_drift_tolerance=0.05, slope_significance=0.05)
        report = assess_signal("permeability", times, values, strict)
        assert not report.passed
        assert any("significant" in f for f in report.failures)

    def test_slope_statistics_are_reported(self):
        rng = random.Random(3)
        times, values = series([5.0 + rng.gauss(0, 0.01) for _ in range(30)])
        report = assess_signal("permeability", times, values, criteria())
        assert report.slope_t_statistic is not None
        assert 0.0 <= report.slope_p_value <= 1.0


class TestDetector:
    def _feed(self, detector, values, step=1.0, signal="permeability"):
        status = None
        for index, value in enumerate(values):
            status = detector.update(index * step, {signal: value})
        return status

    def test_a_steady_signal_is_confirmed_after_the_required_windows(self):
        detector = SteadyStateDetector(criteria(window_s=10.0, required_windows=2))
        status = self._feed(detector, [5.0] * 40)
        assert status.is_steady
        assert status.consecutive_passes >= 2
        assert detector.steady_since_elapsed_s is not None

    def test_one_window_is_not_enough_when_two_are_required(self):
        detector = SteadyStateDetector(criteria(window_s=10.0, required_windows=2))
        status = self._feed(detector, [5.0] * 15)
        assert not status.is_steady
        assert status.consecutive_passes == 1

    def test_windows_are_non_overlapping(self):
        """N consecutive passes must mean N independent windows, not N samples."""
        detector = SteadyStateDetector(criteria(window_s=10.0, required_windows=3))
        # 25 s of data closes windows at 10, 20 -- two evaluations, not 25.
        status = self._feed(detector, [5.0] * 25)
        assert status.consecutive_passes == 2
        assert not status.is_steady

    def test_a_ramp_never_becomes_steady(self):
        detector = SteadyStateDetector(criteria())
        status = self._feed(detector, [1.0 + 0.05 * i for i in range(60)])
        assert not status.is_steady
        assert detector.steady_since_elapsed_s is None

    def test_settling_time_is_ignored(self):
        detector = SteadyStateDetector(
            criteria(window_s=10.0, required_windows=1, settling_time_s=20.0)
        )
        # Wild for the first 20 s, then flat.
        values = [1.0, 9.0] * 10 + [5.0] * 20
        status = self._feed(detector, values)
        assert status.is_steady

    def test_destabilising_clears_the_steady_stretch(self):
        detector = SteadyStateDetector(criteria(window_s=10.0, required_windows=2))
        self._feed(detector, [5.0] * 40)
        assert detector.is_steady
        # A step change breaks it.
        for index in range(40, 60):
            detector.update(float(index), {"permeability": 9.0})
        assert not detector.is_steady
        assert detector.steady_since_elapsed_s is None

    def test_a_missing_value_fails_the_window(self):
        detector = SteadyStateDetector(criteria(window_s=10.0, required_windows=1))
        values = [5.0] * 8 + [None] + [5.0] * 8
        status = self._feed(detector, values)
        assert not status.is_steady
        assert any("no usable value" in f for r in status.signals for f in r.failures)

    def test_every_listed_signal_must_pass(self):
        """A flat permeability with a drifting pressure is not steady."""
        detector = SteadyStateDetector(
            criteria(signals=["permeability", "inlet_pressure"], required_windows=1)
        )
        status = None
        for index in range(25):
            status = detector.update(
                float(index),
                {"permeability": 5.0, "inlet_pressure": 3.0 + 0.05 * index},
            )
        assert not status.is_steady
        assert any(r.name == "inlet_pressure" and not r.passed for r in status.signals)

    def test_disabled_detection_never_reports_steady(self):
        detector = SteadyStateDetector(criteria(enabled=False))
        status = self._feed(detector, [5.0] * 60)
        assert not status.is_steady
        assert "disabled" in status.summary

    def test_status_carries_progress_between_windows(self):
        detector = SteadyStateDetector(criteria(window_s=10.0, required_windows=3))
        status = self._feed(detector, [5.0] * 23)
        assert status.progress == "2/3"


class TestDetectSteadyWindow:
    def test_finds_the_plateau_after_a_transient(self):
        rows = [
            {"elapsed_s": float(i), "permeability": (1.0 + 0.4 * i if i < 15 else 6.6)}
            for i in range(60)
        ]
        window = detect_steady_window(rows, criteria(window_s=10.0, required_windows=2))
        assert window is not None
        # The plateau starts at 15 s; detection cannot confirm it before then.
        assert window.start_elapsed_s >= 15.0
        assert window.end_elapsed_s == 59.0
        assert window.sample_count == pytest.approx(
            window.end_elapsed_s - window.start_elapsed_s + 1, abs=1
        )

    def test_returns_none_for_a_run_that_never_settles(self):
        rows = [{"elapsed_s": float(i), "permeability": 1.0 + 0.1 * i} for i in range(60)]
        assert detect_steady_window(rows, criteria()) is None

    def test_live_and_offline_detection_agree(self):
        """The same criteria over the same data must give the same verdict."""
        values = [1.0 + 0.4 * i if i < 15 else 6.6 for i in range(60)]
        rows = [{"elapsed_s": float(i), "permeability": v} for i, v in enumerate(values)]
        config = criteria(window_s=10.0, required_windows=2)

        live = SteadyStateDetector(config)
        for row in rows:
            live.update(row["elapsed_s"], {"permeability": row["permeability"]})

        offline = detect_steady_window(rows, config)
        assert offline is not None
        assert offline.start_elapsed_s == pytest.approx(live.steady_since_elapsed_s, abs=1.0)


class TestSignalExtraction:
    def test_temperature_is_taken_in_kelvin(self, base_config, fixed_gas_provider):
        """Celsius would divide by a near-zero mean on a cold-room run."""
        from gasperm.acquisition import SampleProcessor
        from gasperm.hardware.temperature import TemperatureSample

        base_config.run.outlet_pressure_reference = "measured"
        processor = SampleProcessor(base_config, fixed_gas_provider)
        reading = processor.process(
            index=0,
            elapsed_s=0.0,
            voltages={"ai0": 2.5, "ai1": 0.5, "ai2": 4.0},
            temperature=TemperatureSample(0.0, 0.0, None, False),
        )
        signals = signals_from_reading(reading)
        assert signals["temperature"] == pytest.approx(273.15)
        assert math.isfinite(signals["permeability"])
