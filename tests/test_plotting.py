"""The live ``--plot`` window: panels, time windows and criterion lines.

Rendering is exercised through a real matplotlib figure on the Agg backend --
no window ever opens, and nothing here needs a display. The point is that the
buffering, the two display modes and the criterion geometry are ordinary
testable logic, not something only visible by eye on a running rig.
"""

from __future__ import annotations

import math

import matplotlib
import pytest

matplotlib.use("Agg")  # noqa: E402 - must precede any pyplot import

from gasperm.config import GaspermConfig  # noqa: E402
from gasperm.models import Reading, SignalStability, SteadyStateStatus  # noqa: E402
from gasperm.plotting import (  # noqa: E402
    LivePlot,
    _History,
    _limits_with_band,
    _panels_for,
    _steady_spans,
)


def reading(
    index: int = 0,
    elapsed_s: float = 0.0,
    *,
    inlet_atm: float = 5.0,
    outlet_atm: float = 1.0,
    flow_cm3_s: float = 3.0,
    temperature_c: float = 22.0,
    permeability: float | None = 0.005,
    steady: bool = False,
    decay_fraction: float | None = None,
) -> Reading:
    return Reading(
        index=index,
        timestamp="2026-08-06T10:00:00+00:00",
        elapsed_s=elapsed_s,
        inlet_voltage=1.0,
        outlet_voltage=0.2,
        flow_voltage=4.0,
        inlet_pressure_atm=inlet_atm,
        outlet_pressure_atm=outlet_atm,
        downstream_pressure_atm=outlet_atm,
        mean_pressure_atm=(inlet_atm + outlet_atm) / 2,
        flow_cm3_s=flow_cm3_s,
        flow_reference_cm3_s=flow_cm3_s,
        flow_reference_pressure_atm=1.0,
        temperature_c=temperature_c,
        viscosity_cp=0.0178,
        permeability_darcy=permeability,
        permeability_darcy_avg=permeability,
        steady_state=steady,
        decay_fraction=decay_fraction,
    )


def status_with(
    signal: str = "inlet_pressure",
    *,
    mean: float = 5.0,
    relative_stddev: float = 0.001,
    relative_drift: float = 0.0005,
    slope_per_s: float = 0.0,
    passed: bool = True,
    elapsed_s: float = 100.0,
) -> SteadyStateStatus:
    return SteadyStateStatus(
        is_steady=passed,
        consecutive_passes=3 if passed else 0,
        required_passes=3,
        window_s=30.0,
        elapsed_s=elapsed_s,
        signals=[
            SignalStability(
                name=signal,
                sample_count=50,
                mean=mean,
                stddev=mean * relative_stddev,
                relative_stddev=relative_stddev,
                slope_per_s=slope_per_s,
                relative_drift=relative_drift,
                passed=passed,
            )
        ],
        summary="steady (3/3)" if passed else "not steady",
    )


class TestHistory:
    def test_the_archive_still_spans_the_whole_run_once_decimated(self):
        """A from-t0 view of a long run must start at t0, not part way in."""
        history = _History(["flow"], max_points=50)
        for index in range(5000):
            history.append(index * 0.1, {"flow": float(index)}, False)
        assert len(history) == 5000
        assert len(history.archive_times) <= 50
        assert history.archive_times[0] == pytest.approx(0.0)
        # Within one stride of the true end.
        assert history.archive_times[-1] > 490.0

    def test_the_recent_buffer_keeps_full_detail_and_stays_bounded(self):
        history = _History(["flow"], max_points=50)
        for index in range(5000):
            history.append(index * 0.1, {"flow": float(index)}, False)
        assert len(history.recent_times) <= 50
        # Consecutive samples, i.e. not decimated.
        gaps = [
            b - a for a, b in zip(history.recent_times, history.recent_times[1:])
        ]
        assert all(gap == pytest.approx(0.1) for gap in gaps)
        assert history.recent_times[-1] == pytest.approx(499.9)

    def test_a_window_is_served_from_the_full_rate_buffer(self):
        history = _History(["flow"], max_points=500)
        for index in range(1000):
            history.append(index * 0.1, {"flow": float(index)}, False)
        times, channels, _ = history.view(10.0)
        assert times[-1] - times[0] == pytest.approx(10.0, abs=0.11)
        assert len(times) == pytest.approx(101, abs=1)
        assert len(channels["flow"]) == len(times)

    def test_a_window_longer_than_the_buffer_falls_back_to_the_archive(self):
        """Coarser data beats a silently truncated axis."""
        history = _History(["flow"], max_points=50)
        for index in range(5000):
            history.append(index * 0.1, {"flow": float(index)}, False)
        times, _, _ = history.view(400.0)
        assert times[-1] - times[0] > 300.0

    def test_from_t0_ignores_any_window(self):
        history = _History(["flow"], max_points=500)
        for index in range(1000):
            history.append(index * 0.1, {"flow": float(index)}, False)
        times, _, _ = history.view(None)
        assert times[0] == pytest.approx(0.0)


class TestSteadySpans:
    def test_a_contiguous_stretch_becomes_one_span(self):
        times = [0.0, 1.0, 2.0, 3.0, 4.0]
        assert _steady_spans(times, [False, False, True, True, True]) == [(2.0, 4.0)]

    def test_a_stretch_that_ended_closes_at_the_first_unsteady_sample(self):
        times = [0.0, 1.0, 2.0, 3.0]
        assert _steady_spans(times, [True, True, False, False]) == [(0.0, 2.0)]

    def test_separate_stretches_stay_separate(self):
        times = [0.0, 1.0, 2.0, 3.0, 4.0]
        spans = _steady_spans(times, [True, False, True, True, False])
        assert spans == [(0.0, 1.0), (2.0, 4.0)]

    def test_no_steady_samples_means_no_shading(self):
        assert _steady_spans([0.0, 1.0], [False, False]) == []


class TestBandLimits:
    def test_a_band_near_the_data_is_included(self):
        limits, clipped = _limits_with_band((10.0, 12.0), 9.5, 12.5)
        assert limits == (9.5, 12.5)
        assert clipped is False

    def test_a_band_far_wider_than_the_data_is_left_off_scale(self):
        """Otherwise a settled trace is compressed into a flat line."""
        limits, clipped = _limits_with_band((99.9, 100.1), 98.0, 102.0)
        assert limits == (99.9, 100.1)
        assert clipped is True

    def test_a_band_narrower_than_the_data_never_clips(self):
        """Early in a run the signal is outside the band; both must show."""
        limits, clipped = _limits_with_band((0.0, 100.0), 49.0, 51.0)
        assert limits == (0.0, 100.0)
        assert clipped is False


class TestPanels:
    def test_the_configured_order_is_the_drawing_order(self):
        config = GaspermConfig()
        config.run.plot.panels = ["permeability", "flow", "inlet_pressure"]
        assert [p.key for p in _panels_for(config)] == [
            "permeability", "flow", "inlet_pressure"
        ]

    def test_every_parameter_gets_its_own_panel_by_default(self):
        panels = _panels_for(GaspermConfig())
        assert [p.key for p in panels] == [
            "inlet_pressure", "outlet_pressure", "flow", "temperature", "permeability"
        ]
        # Pressures are not overlaid: one trace each, on separate axes.
        assert len(panels[0].traces) == 1
        assert len(panels[1].traces) == 1

    def test_the_outlet_carries_no_criteria_because_it_is_not_monitored(self):
        panels = {p.key: p for p in _panels_for(GaspermConfig())}
        assert panels["outlet_pressure"].signal is None
        assert panels["inlet_pressure"].signal == "inlet_pressure"

    def test_the_permeability_panel_shows_instant_and_averaged(self):
        """The detector tests the instantaneous value; the console reports the mean."""
        panel = {p.key: p for p in _panels_for(GaspermConfig())}["permeability"]
        assert [t.channel for t in panel.traces] == ["permeability", "permeability_avg"]

    def test_a_supplied_downstream_pressure_adds_the_transducer_cross_check(self):
        config = GaspermConfig()
        config.run.downstream_pressure = 101.325
        panel = {p.key: p for p in _panels_for(config)}["outlet_pressure"]
        channels = [t.channel for t in panel.traces]
        assert channels == ["outlet_pressure", "downstream_pressure"]

    def test_temperature_converts_out_of_the_detector_kelvin(self):
        """The detector works in K so it never divides by a near-zero mean."""
        panel = {p.key: p for p in _panels_for(GaspermConfig())}["temperature"]
        assert panel.to_display(295.15) == pytest.approx(22.0)


class TestPulseAndLeakPanels:
    """A pulse-decay window, and how a leak test differs from a measurement."""

    def pulse(self, *, leak_test: bool = False) -> GaspermConfig:
        config = GaspermConfig()
        config.run.method = "pulse_decay"
        if leak_test:
            config.run.purpose = "leak_test"
        return config

    def test_the_decay_panels_replace_the_flow_panel(self):
        keys = [panel.key for panel in _panels_for(self.pulse())]
        assert "flow" not in keys
        assert {"delta_pressure", "decay_fraction"} <= set(keys)

    def test_the_decay_fraction_axis_spans_the_meaningful_range(self):
        """Otherwise a rig that is holding reads as violent noise.

        dP/dP0 sits at 1.0 to within a few parts in 1e5 on a tight rig, and a
        log axis autoscaled to that turns the flat trace the operator wants to
        see into apparent chaos.
        """
        config = self.pulse(leak_test=True)
        panel = next(p for p in _panels_for(config) if p.key == "decay_fraction")
        low, high = panel.y_range
        assert low == pytest.approx(config.run.pulse_decay.stop_below_fraction * 0.8)
        assert high >= 1.0

    def test_a_flat_decay_fraction_is_not_magnified(self):
        config = self.pulse(leak_test=True)
        config.run.plot.panels = ["decay_fraction"]
        plot = LivePlot(config)
        plot.open()
        for index in range(40):
            # A rig that is not leaking: 1.0 give or take a few parts in 1e5.
            plot.add(reading(index, index * 0.1, decay_fraction=1.0 + 1e-5 * (index % 3)))
        plot.maybe_redraw(now=1000.0)
        low, high = plot._axes[0].get_ylim()
        assert low <= config.run.pulse_decay.stop_below_fraction
        assert high >= 1.0
        plot.close()

    def test_the_permeability_panel_says_whose_permeability_it_is(self):
        """On a leak test the trace is the APPARATUS, not the rock."""
        measurement = next(
            p for p in _panels_for(self.pulse()) if p.key == "permeability"
        )
        leak = next(
            p for p in _panels_for(self.pulse(leak_test=True)) if p.key == "permeability"
        )
        assert measurement.ylabel.startswith("k (")
        assert leak.ylabel.startswith("leak equiv.")
        assert all("leak equiv." in trace.label for trace in leak.traces)

    def test_the_title_names_a_leak_test(self):
        plot = LivePlot(self.pulse(leak_test=True))
        plot.open()
        plot.add(reading())
        plot.maybe_redraw(now=1000.0)
        assert "LEAK TEST" in plot._figure._suptitle.get_text()
        plot.close()

    def test_a_measurement_title_does_not(self):
        plot = LivePlot(self.pulse())
        plot.open()
        plot.add(reading())
        plot.maybe_redraw(now=1000.0)
        assert "LEAK TEST" not in plot._figure._suptitle.get_text()
        plot.close()

    def test_no_steady_state_note_appears_in_pulse_mode(self):
        """No detector is running, so it would be true of every panel."""
        config = self.pulse()
        config.run.plot.panels = ["delta_pressure", "temperature"]
        plot = LivePlot(config)
        plot.open()
        for index in range(20):
            plot.add(reading(index, index * 0.1))
        plot.maybe_redraw(now=1000.0)
        for axis in plot._axes:
            texts = [t.get_text() for t in axis.texts]
            assert not any("steady-state signal" in t for t in texts)
        plot.close()

    def test_steady_state_runs_keep_the_note(self):
        config = GaspermConfig()
        config.run.plot.panels = ["outlet_pressure"]
        plot = LivePlot(config)
        plot.open()
        for index in range(20):
            plot.add(reading(index, index * 0.1), status_with())
        plot.maybe_redraw(now=1000.0)
        texts = [t.get_text() for t in plot._axes[0].texts]
        assert any("not a steady-state signal" in t for t in texts)
        plot.close()


class TestWindowResolution:
    def test_the_configured_window_is_used_by_default(self):
        config = GaspermConfig()
        config.run.plot.window_s = 120.0
        assert LivePlot(config).window_s == 120.0

    def test_no_configured_window_means_from_t0(self):
        assert LivePlot(GaspermConfig()).window_s is None

    def test_an_explicit_window_overrides_the_config(self):
        config = GaspermConfig()
        config.run.plot.window_s = 120.0
        assert LivePlot(config, window_s=30.0).window_s == 30.0

    def test_from_start_overrides_a_configured_window(self):
        config = GaspermConfig()
        config.run.plot.window_s = 120.0
        assert LivePlot(config, from_start=True).window_s is None


class TestRendering:
    """Drive a real figure headless; assert on what ended up on the axes."""

    def _plot(self, config, **kwargs) -> LivePlot:
        plot = LivePlot(config, **kwargs)
        plot.open()
        return plot

    def test_one_axes_per_configured_panel(self):
        config = GaspermConfig()
        config.run.plot.panels = ["inlet_pressure", "flow"]
        plot = self._plot(config)
        assert len(plot._axes) == 2
        plot.close()

    def test_a_single_panel_still_renders(self):
        """squeeze=False matters: one panel must not collapse to a bare axes."""
        config = GaspermConfig()
        config.run.plot.panels = ["permeability"]
        plot = self._plot(config)
        for index in range(20):
            plot.add(reading(index, index * 0.1))
        assert plot.maybe_redraw(now=1000.0) is True
        plot.close()

    def test_the_x_axis_starts_at_zero_from_t0(self):
        plot = self._plot(GaspermConfig())
        for index in range(60):
            plot.add(reading(index, 100.0 + index * 0.1))
        plot.maybe_redraw(now=1000.0)
        assert plot._axes[-1].get_xlim()[0] == pytest.approx(0.0)
        plot.close()

    def test_the_x_axis_is_exactly_the_window_when_one_is_set(self):
        plot = self._plot(GaspermConfig(), window_s=5.0)
        for index in range(200):
            plot.add(reading(index, index * 0.1))
        plot.maybe_redraw(now=1000.0)
        low, high = plot._axes[-1].get_xlim()
        assert high - low == pytest.approx(5.0)
        assert high == pytest.approx(19.9)
        plot.close()

    def test_a_window_wider_than_the_run_so_far_is_clamped_at_t0(self):
        """Never draw axis before the run started."""
        plot = self._plot(GaspermConfig(), window_s=60.0)
        for index in range(50):
            plot.add(reading(index, index * 0.1))
        plot.maybe_redraw(now=1000.0)
        low, high = plot._axes[-1].get_xlim()
        assert low == pytest.approx(0.0)
        assert high == pytest.approx(4.9)
        plot.close()

    def test_the_steady_stretch_is_shaded(self):
        plot = self._plot(GaspermConfig())
        for index in range(60):
            plot.add(reading(index, index * 0.1, steady=index >= 30))
        plot.maybe_redraw(now=1000.0)
        # One axvspan patch per panel, on top of the traces.
        patches = plot._axes[0].patches
        assert len(patches) == 1
        plot.close()

    def test_an_unsteady_run_is_not_shaded(self):
        plot = self._plot(GaspermConfig())
        for index in range(60):
            plot.add(reading(index, index * 0.1))
        plot.maybe_redraw(now=1000.0)
        assert len(plot._axes[0].patches) == 0
        plot.close()

    def test_criterion_lines_are_drawn_on_a_monitored_panel(self):
        config = GaspermConfig()
        config.run.plot.panels = ["inlet_pressure"]
        plot = self._plot(config)
        for index in range(60):
            plot.add(reading(index, index * 0.1), status_with("inlet_pressure"))
        plot.maybe_redraw(now=1000.0)
        # The trace, plus mean + two tolerance bounds + the drift segment.
        assert len(plot._axes[0].lines) == 5
        plot.close()

    def test_an_unmonitored_panel_says_why_it_has_no_criteria(self):
        config = GaspermConfig()
        config.run.plot.panels = ["outlet_pressure"]
        plot = self._plot(config)
        for index in range(60):
            plot.add(reading(index, index * 0.1), status_with())
        plot.maybe_redraw(now=1000.0)
        texts = [t.get_text() for t in plot._axes[0].texts]
        assert "not a steady-state signal" in texts
        assert len(plot._axes[0].lines) == 1  # the trace alone
        plot.close()

    def test_a_signal_dropped_from_the_run_criteria_also_says_so(self):
        """`signals` omits temperature by default; that is not 'no data yet'."""
        config = GaspermConfig()
        config.run.plot.panels = ["temperature"]
        assert "temperature" not in config.run.steady_state.signals
        plot = self._plot(config)
        for index in range(60):
            plot.add(reading(index, index * 0.1), status_with())
        plot.maybe_redraw(now=1000.0)
        assert "not a steady-state signal" in [t.get_text() for t in plot._axes[0].texts]
        plot.close()

    def test_the_criteria_can_be_switched_off(self):
        config = GaspermConfig()
        config.run.plot.panels = ["inlet_pressure"]
        config.run.plot.show_criteria = False
        plot = self._plot(config)
        for index in range(60):
            plot.add(reading(index, index * 0.1), status_with("inlet_pressure"))
        plot.maybe_redraw(now=1000.0)
        assert len(plot._axes[0].lines) == 1
        plot.close()

    def test_the_corner_note_reports_both_criteria_against_tolerance(self):
        config = GaspermConfig()
        config.run.plot.panels = ["inlet_pressure"]
        plot = self._plot(config)
        for index in range(60):
            plot.add(
                reading(index, index * 0.1),
                status_with("inlet_pressure", relative_stddev=0.004, relative_drift=0.002),
            )
        plot.maybe_redraw(now=1000.0)
        note = " ".join(t.get_text() for t in plot._axes[0].texts)
        assert "scatter 0.40%/2.0%" in note
        assert "drift 0.20%/1.0%" in note
        plot.close()

    def test_an_infinite_scatter_does_not_break_the_note(self):
        """The detector reports inf for a dead channel; that must still format."""
        config = GaspermConfig()
        config.run.plot.panels = ["inlet_pressure"]
        plot = self._plot(config)
        for index in range(60):
            plot.add(
                reading(index, index * 0.1),
                status_with(
                    "inlet_pressure", relative_stddev=math.inf,
                    relative_drift=math.inf, passed=False,
                ),
            )
        plot.maybe_redraw(now=1000.0)
        assert "scatter --/2.0%" in " ".join(t.get_text() for t in plot._axes[0].texts)
        plot.close()

    def test_a_sample_with_no_permeability_leaves_a_gap(self):
        """NaN, not a dropped point: a missing k is part of the picture."""
        plot = self._plot(GaspermConfig())
        plot.add(reading(0, 0.0, permeability=None))
        values = plot._history.recent["permeability"]
        assert math.isnan(values[0])
        plot.close()

    def test_the_title_names_the_display_mode(self):
        plot = self._plot(GaspermConfig(), window_s=45.0)
        for index in range(20):
            plot.add(reading(index, index * 0.1), status_with())
        plot.maybe_redraw(now=1000.0)
        title = plot._figure._suptitle.get_text()
        assert "last 45 s" in title
        assert "steady (3/3)" in title
        plot.close()


class TestRepeatedRedraw:
    """A live plot redraws hundreds of times; the first frame proves nothing.

    The traces are created once and thereafter only fed new data -- clearing
    the axes per frame is what made a redraw cost ~310 ms whatever it drew,
    because it destroyed the ticks and the next draw rebuilt every one. What
    that buys has to be paid for here: the annotations *do* change every frame,
    so they are stripped and redrawn, and nothing may accumulate.
    """

    def _plot(self, config, **kwargs) -> LivePlot:
        plot = LivePlot(config, **kwargs)
        plot.open()
        return plot

    def feed(self, plot, count, start=0, *, steady=False, status=None):
        for index in range(start, start + count):
            plot.add(reading(index, index * 0.1, steady=steady), status)

    def test_a_trace_is_the_same_artist_frame_after_frame(self):
        """If it is not, the ticks are being rebuilt too and the cost is back."""
        config = GaspermConfig()
        config.run.plot.panels = ["inlet_pressure"]
        plot = self._plot(config)
        self.feed(plot, 20)
        plot.maybe_redraw(now=1000.0)
        first = plot._axes[0].lines[0]
        self.feed(plot, 20, start=20)
        plot.maybe_redraw(now=2000.0)
        assert plot._axes[0].lines[0] is first
        plot.close()

    def test_the_trace_still_follows_the_data(self):
        config = GaspermConfig()
        config.run.plot.panels = ["inlet_pressure"]
        plot = self._plot(config)
        self.feed(plot, 20)
        plot.maybe_redraw(now=1000.0)
        self.feed(plot, 20, start=20)
        plot.maybe_redraw(now=2000.0)
        xdata = plot._axes[0].lines[0].get_xdata()
        assert len(xdata) == 40
        assert xdata[-1] == pytest.approx(3.9)
        plot.close()

    def test_criterion_lines_do_not_accumulate(self):
        """They are redrawn every frame, so they must be stripped every frame."""
        config = GaspermConfig()
        config.run.plot.panels = ["inlet_pressure"]
        plot = self._plot(config)
        self.feed(plot, 60, status=status_with("inlet_pressure"))
        plot.maybe_redraw(now=1000.0)
        after_one = len(plot._axes[0].lines)
        for frame in range(2, 8):
            self.feed(plot, 10, start=60 * frame, status=status_with("inlet_pressure"))
            plot.maybe_redraw(now=1000.0 * frame)
        assert len(plot._axes[0].lines) == after_one == 5
        plot.close()

    def test_the_shaded_span_does_not_accumulate(self):
        plot = self._plot(GaspermConfig())
        self.feed(plot, 60, steady=True)
        plot.maybe_redraw(now=1000.0)
        for frame in range(2, 8):
            self.feed(plot, 10, start=60 * frame, steady=True)
            plot.maybe_redraw(now=1000.0 * frame)
        assert len(plot._axes[0].patches) == 1
        plot.close()

    def test_the_corner_note_does_not_accumulate(self):
        config = GaspermConfig()
        config.run.plot.panels = ["outlet_pressure"]
        plot = self._plot(config)
        for frame in range(1, 6):
            self.feed(plot, 20, start=20 * frame, status=status_with())
            plot.maybe_redraw(now=1000.0 * frame)
        notes = [t.get_text() for t in plot._axes[0].texts]
        assert notes.count("not a steady-state signal") == 1
        plot.close()

    def test_the_y_axis_still_follows_the_data_after_a_criterion_band(self):
        """A band's set_ylim turns autoscale off; the next frame must re-enable it.

        Without that the panel freezes at the first frame's limits and a signal
        walking out of range slides off the top of a plot that looks fine --
        exactly the drift the criterion lines are there to make visible.
        """
        config = GaspermConfig()
        config.run.plot.panels = ["inlet_pressure"]
        plot = self._plot(config)
        for index in range(60):
            plot.add(reading(index, index * 0.1, inlet_atm=5.0), status_with("inlet_pressure"))
        plot.maybe_redraw(now=1000.0)
        settled = plot._axes[0].get_ylim()

        # The inlet climbs well clear of the old window.
        for index in range(60, 120):
            plot.add(reading(index, index * 0.1, inlet_atm=9.0), status_with("inlet_pressure"))
        plot.maybe_redraw(now=2000.0)
        assert plot._axes[0].get_ylim()[1] > settled[1]
        plot.close()

    def test_a_pinned_range_survives_repeated_frames(self):
        """The decay panel's floor is not a first-frame-only courtesy."""
        config = GaspermConfig()
        config.run.method = "pulse_decay"
        config.run.plot.panels = ["decay_fraction"]
        plot = self._plot(config)
        for frame in range(1, 5):
            for index in range(20 * frame, 20 * frame + 20):
                plot.add(reading(index, index * 0.1, decay_fraction=1.0 - 1e-5 * index))
            plot.maybe_redraw(now=1000.0 * frame)
        low, high = plot._axes[0].get_ylim()
        floor = config.run.pulse_decay.stop_below_fraction * 0.8
        assert low <= floor and high >= 1.05
        plot.close()


class TestRedrawBackoff:
    """Drawing happens on the acquisition thread, so it has to budget itself."""

    def test_a_fast_redraw_keeps_the_configured_interval(self):
        plot = LivePlot(GaspermConfig(), redraw_interval_s=0.5)
        plot._redraw_cost_s = 0.002
        assert plot._effective_interval_s() == pytest.approx(0.5)

    def test_a_slow_redraw_stretches_it(self):
        """A 0.15 s redraw every 0.5 s spends a third of the run not sampling."""
        from gasperm.plotting import MAX_REDRAW_DUTY

        plot = LivePlot(GaspermConfig(), redraw_interval_s=0.5)
        plot._redraw_cost_s = 0.15
        assert plot._effective_interval_s() == pytest.approx(0.15 / MAX_REDRAW_DUTY)
        assert plot._effective_interval_s() > 0.5

    def test_the_interval_is_never_shortened(self):
        """It is a floor the operator set, not a target to be optimised past."""
        plot = LivePlot(GaspermConfig(), redraw_interval_s=5.0)
        plot._redraw_cost_s = 0.001
        assert plot._effective_interval_s() == pytest.approx(5.0)

    def test_an_unmeasured_plot_uses_the_configured_interval(self):
        plot = LivePlot(GaspermConfig(), redraw_interval_s=0.5)
        assert plot._redraw_cost_s is None
        assert plot._effective_interval_s() == pytest.approx(0.5)

    def test_the_cost_is_measured_from_a_real_redraw(self):
        plot = LivePlot(GaspermConfig())
        plot.open()
        plot.add(reading())
        plot.maybe_redraw(now=100.0)
        assert plot._redraw_cost_s is not None and plot._redraw_cost_s > 0.0
        plot.close()

    def test_backoff_defers_the_next_frame(self):
        """The measured cost, not the configured interval, gates the next one."""
        plot = LivePlot(GaspermConfig(), redraw_interval_s=0.5)
        plot.open()
        plot.add(reading())
        assert plot.maybe_redraw(now=100.0) is True
        plot._redraw_cost_s = 0.15  # a slow backend
        assert plot.maybe_redraw(now=100.6) is False
        assert plot.maybe_redraw(now=102.0) is True
        plot.close()


class TestLoopSafety:
    """The plot must never be able to take a run down with it."""

    def test_redraw_is_rate_limited(self):
        plot = LivePlot(GaspermConfig(), redraw_interval_s=10.0)
        plot.open()
        plot.add(reading())
        assert plot.maybe_redraw(now=100.0) is True
        assert plot.maybe_redraw(now=105.0) is False
        assert plot.maybe_redraw(now=111.0) is True
        plot.close()

    def test_nothing_is_drawn_before_the_first_sample(self):
        plot = LivePlot(GaspermConfig())
        plot.open()
        assert plot.maybe_redraw(now=100.0) is False
        plot.close()

    def test_a_failing_redraw_disables_the_plot_rather_than_raising(self):
        plot = LivePlot(GaspermConfig())
        plot.open()
        plot.add(reading())
        plot._figure = _Exploding()
        assert plot.maybe_redraw(now=100.0) is False
        assert plot._disabled is True
        # And it stays quiet afterwards.
        plot.add(reading(1, 1.0))
        assert plot.maybe_redraw(now=200.0) is False

    def test_a_disabled_plot_buffers_nothing(self):
        plot = LivePlot(GaspermConfig())
        plot._disabled = True
        plot.add(reading())
        assert len(plot._history) == 0

    def test_close_is_safe_without_a_figure(self):
        LivePlot(GaspermConfig()).close()


class _Exploding:
    """Stands in for a figure whose window the operator closed mid-run."""

    def __getattr__(self, name):
        raise RuntimeError("the window is gone")
