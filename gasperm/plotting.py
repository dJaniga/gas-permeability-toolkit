"""Optional matplotlib views: the live ``--plot`` window and the Klinkenberg fit.

Plotting is strictly additive. The console output is the primary display and
``--plot`` only adds a window on top of it; nothing here is required for a run
to produce correct data.

**The live plot must never slow the acquisition loop.** Points go into a
bounded buffer on the loop's thread -- an O(1) append -- and the figure is
redrawn on a timer (``redraw_interval_s``), not once per sample. At 10 Hz a
per-sample redraw would spend more time in matplotlib than in the DAQ. If the
redraw itself fails (window closed by the operator, no display available), the
error is swallowed and the run continues.

Each monitored quantity gets its **own** stacked panel rather than sharing
axes, and the panels the detector actually watches carry its criteria drawn on
them: the trailing window's mean, the scatter tolerance band around it, and the
fitted drift line. Watching a signal approach its band is the whole point of a
live view on a rig where a plateau can take hours.

matplotlib is imported lazily so the package works headless.
"""

from __future__ import annotations

import logging
import math
import time
from bisect import bisect_left
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

from gasperm import screens, units
from gasperm.config import GaspermConfig
from gasperm.config.run import METHOD_ONLY_PANELS
from gasperm.models import KlinkenbergResult, Reading, SteadyStateStatus

logger = logging.getLogger(__name__)

__all__ = [
    "LivePlot",
    "PreviewPlot",
    "plot_comparison",
    "plot_klinkenberg",
    "plot_pulse_decay",
    "PlottingUnavailable",
]

#: How many points the live window keeps per series. Bounded so a multi-hour
#: run cannot grow the figure's memory without limit.
DEFAULT_MAX_POINTS = 3600

#: Steady state is only ever declared after ``required_windows`` consecutive
#: windows, so a steady stretch is minutes long at the shipped criteria. The
#: from-t0 view can therefore decimate its steady flags by plain sampling
#: without any risk of dropping a span: losing one would take a run roughly
#: ``max_points * required_windows * window_s`` long.
_STEADY_COLOR = "tab:green"

#: How far the criterion band may stretch a panel beyond its data before it is
#: left off-screen instead. A band comfortably wider than the signal means the
#: signal is well inside tolerance, and letting it set the y-axis would flatten
#: the trace into a line -- which is exactly the shape the operator is watching.
#: The numbers are still reported in the corner annotation either way.
_MAX_BAND_ZOOM = 4.0


class PlottingUnavailable(RuntimeError):
    """matplotlib is not installed or no usable backend is available."""


def place_figure(figure: Any, plot_config: Any) -> str:
    """Move a figure's window onto the configured monitor.

    Best-effort in every direction: no manager, no window, an unsupported
    backend or an absent monitor all leave the window where it was. A plot is
    additive to a run and placement is additive to the plot -- neither may take
    anything down with it.

    Returns what was done, or ``""``, so the caller can log it once.
    """
    if plot_config is None:
        return ""
    monitor = getattr(plot_config, "monitor", None)
    mode = getattr(plot_config, "window", "normal")
    if monitor is None and mode == "normal":
        return ""

    screen, complaint = screens.choose_screen(screens.list_screens(), monitor)
    if complaint:
        logger.warning("%s", complaint)
    try:
        window = figure.canvas.manager.window
    except AttributeError:
        logger.debug("This backend has no window to place.")
        return ""
    return screens.place_window(window, screen, mode)


def _pyplot(interactive: bool):
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise PlottingUnavailable(
            "matplotlib is not installed. Install it with 'pip install matplotlib', or "
            "drop the --plot flag."
        ) from exc
    if not interactive:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


class _Trace(NamedTuple):
    """One line on a panel."""

    channel: str
    label: str
    color: str
    linewidth: float = 1.4
    alpha: float = 1.0


class _Panel(NamedTuple):
    """One stacked axes: a quantity, its unit, and how it is drawn.

    ``signal`` is the steady-state detector's name for this quantity, or
    ``None`` when the detector does not watch it -- an unwatched panel gets no
    criterion lines, which is itself worth seeing.
    """

    key: str
    ylabel: str
    traces: tuple[_Trace, ...]
    signal: str | None
    #: Detector-internal units to display units. Affine, not merely scaled:
    #: the detector works in kelvin so it never divides by a near-zero mean.
    to_display: Callable[[float], float]
    #: ``"log"`` for the decay fraction, where a straightening line is the whole
    #: visual confirmation the operator is waiting for. Defaulted, so no
    #: existing construction site changes.
    yscale: str = "linear"
    #: A range the axis must span at least, ``(low, high)``. For a quantity
    #: with a known meaningful scale this stops autoscale magnifying noise on a
    #: flat trace into what looks like violent movement -- which is exactly
    #: what happens to dP/dP0 on a rig that is not leaking, i.e. the case the
    #: operator most needs to read as "nothing is happening".
    y_range: tuple[float, float] | None = None


def _channel_values(reading: Reading, run) -> dict[str, float]:
    """Every plottable quantity of one reading, already in display units."""

    def darcy(value: float | None) -> float:
        # NaN rather than a dropped point: matplotlib leaves a gap, which is
        # the honest picture of a sample that yielded no permeability.
        if value is None:
            return float("nan")
        return units.darcy_to(value, run.display_permeability_unit)

    return {
        "inlet_pressure": units.from_atm(
            reading.inlet_pressure_atm, run.display_pressure_unit
        ),
        "outlet_pressure": units.from_atm(
            reading.outlet_pressure_atm, run.display_pressure_unit
        ),
        "downstream_pressure": units.from_atm(
            reading.downstream_pressure_atm, run.display_pressure_unit
        ),
        "delta_pressure": units.from_atm(
            reading.delta_pressure_atm, run.display_pressure_unit
        ),
        # NaN rather than a dropped point, as for permeability: before the pulse
        # there is no fraction, and matplotlib leaves the gap.
        "decay_fraction": (
            reading.decay_fraction if reading.decay_fraction is not None else float("nan")
        ),
        "flow": (
            units.flow_from_cm3_s(reading.flow_cm3_s, run.display_flow_unit)
            if reading.flow_cm3_s is not None
            else float("nan")
        ),
        "temperature": reading.temperature_c,
        "permeability": darcy(reading.permeability_darcy),
        "permeability_avg": darcy(reading.permeability_darcy_avg),
    }


def _k_label(run) -> str:
    """What the permeability trace is, which depends on what the run is for."""
    return "leak equiv." if run.purpose == "leak_test" else "k"


def _panels_for(config: GaspermConfig) -> list[_Panel]:
    """Build the requested panels, in the configured order."""
    run = config.run
    pressure = run.display_pressure_unit
    supplied_p2 = not run.downstream_is_measured

    outlet_traces = [
        _Trace("outlet_pressure", "transducer", "tab:orange"),
    ]
    if supplied_p2:
        # A declared P2 is what the equation used; the transducer is then the
        # cross-check, so both belong on the panel and must be told apart.
        outlet_traces.append(
            _Trace("downstream_pressure", "P2 used (supplied)", "tab:red", 1.2, 0.9)
        )

    available = {
        "inlet_pressure": _Panel(
            key="inlet_pressure",
            ylabel=f"inlet P1 ({pressure})",
            traces=(_Trace("inlet_pressure", "P1", "tab:blue"),),
            signal="inlet_pressure",
            to_display=lambda atm: units.from_atm(atm, pressure),
        ),
        "outlet_pressure": _Panel(
            key="outlet_pressure",
            ylabel=f"outlet P2 ({pressure})",
            traces=tuple(outlet_traces),
            # The detector watches the inlet, not the outlet.
            signal=None,
            to_display=lambda atm: units.from_atm(atm, pressure),
        ),
        "delta_pressure": _Panel(
            key="delta_pressure",
            ylabel=f"dP = P1-P2 ({pressure})",
            traces=(_Trace("delta_pressure", "dP", "tab:red"),),
            signal=None,
            to_display=lambda atm: units.from_atm(atm, pressure),
        ),
        "decay_fraction": _Panel(
            key="decay_fraction",
            ylabel="dP / dP0",
            traces=(_Trace("decay_fraction", "dP/dP0", "tab:red"),),
            signal=None,
            to_display=lambda value: value,
            # The decay only means anything between the stop fraction and 1, so
            # pin that span: a rig that is holding its differential then reads
            # as a flat line at the top rather than as magnified noise.
            y_range=(run.pulse_decay.stop_below_fraction * 0.8, 1.05),
            # Log: an exponential decay straightens into a line here, which is
            # the fastest visual check that the plug -- and not a leak or a
            # thermal ramp -- is what the differential is doing.
            yscale="log",
        ),
        "flow": _Panel(
            key="flow",
            ylabel=f"flow ({run.display_flow_unit})",
            traces=(_Trace("flow", "Q", "tab:green"),),
            signal="flow",
            to_display=lambda cm3_s: units.flow_from_cm3_s(cm3_s, run.display_flow_unit),
        ),
        "temperature": _Panel(
            key="temperature",
            ylabel="temperature (C)",
            traces=(_Trace("temperature", "T", "tab:purple"),),
            signal="temperature",
            to_display=units.kelvin_to_celsius,
        ),
        "permeability": _Panel(
            key="permeability",
            # On a leak test this trace is the permeability the APPARATUS would
            # fake, not the rock's. Same number, entirely different meaning.
            ylabel=(
                f"leak equiv. ({run.display_permeability_unit})"
                if run.purpose == "leak_test"
                else f"k ({run.display_permeability_unit})"
            ),
            traces=(
                # The detector tests the instantaneous value, so that is what
                # the criterion band belongs around; the rolling mean is drawn
                # over it because that is the number the console reports.
                _Trace("permeability", f"{_k_label(run)} (instant)", "tab:red", 0.9, 0.35),
                _Trace("permeability_avg", f"{_k_label(run)} (averaged)", "tab:red", 1.6),
            ),
            signal="permeability",
            to_display=lambda d: units.darcy_to(d, run.display_permeability_unit),
        ),
    }
    # Filter by method rather than making the operator curate the list per run:
    # `panels` defaults to every panel, and a flow trace on a pulse run (or a
    # decay trace on a steady one) would be a permanently empty axes.
    return [
        available[name]
        for name in run.plot.panels
        if METHOD_ONLY_PANELS.get(name, run.method) == run.method
    ]


class _History:
    """Bounded storage for every plotted channel.

    Two views, because the two display modes want different things:

    ``recent``
        Every sample, up to ``max_points``. Backs the trailing-window view,
        which needs full detail over a short span.
    ``archive``
        Decimated by a stride that doubles whenever it fills. Backs the from-t0
        view, so a multi-hour run spans the whole x-axis at fixed memory
        instead of either exhausting it or silently starting the axis late.
    """

    def __init__(self, channels: Sequence[str], max_points: int) -> None:
        self._channels = list(channels)
        self.max_points = max_points
        self.recent_times: list[float] = []
        self.recent: dict[str, list[float]] = {name: [] for name in self._channels}
        self.recent_steady: list[bool] = []
        self.archive_times: list[float] = []
        self.archive: dict[str, list[float]] = {name: [] for name in self._channels}
        self.archive_steady: list[bool] = []
        self._stride = 1
        self._seen = 0

    def __len__(self) -> int:
        return self._seen

    def append(self, elapsed_s: float, values: dict[str, float], steady: bool) -> None:
        self.recent_times.append(elapsed_s)
        for name in self._channels:
            self.recent[name].append(values[name])
        self.recent_steady.append(steady)
        if len(self.recent_times) > self.max_points:
            # Drop the oldest in one slice rather than per-append: amortised
            # O(1) and it keeps the lists contiguous for matplotlib.
            keep = self.max_points // 2
            self.recent_times = self.recent_times[-keep:]
            for name in self._channels:
                self.recent[name] = self.recent[name][-keep:]
            self.recent_steady = self.recent_steady[-keep:]

        if self._seen % self._stride == 0:
            self.archive_times.append(elapsed_s)
            for name in self._channels:
                self.archive[name].append(values[name])
            self.archive_steady.append(steady)
            if len(self.archive_times) > self.max_points:
                self._decimate()
        self._seen += 1

    def _decimate(self) -> None:
        """Halve the archive and double the stride, covering the same span."""
        self.archive_times = self.archive_times[::2]
        for name in self._channels:
            self.archive[name] = self.archive[name][::2]
        self.archive_steady = self.archive_steady[::2]
        self._stride *= 2

    def view(self, window_s: float | None) -> tuple[list[float], dict[str, list[float]], list[bool]]:
        """The series to draw for the requested display mode.

        A trailing window is served from ``recent`` when that still reaches
        back far enough, and from the archive otherwise -- asking for a window
        longer than the buffer holds degrades to coarser data rather than to a
        silently truncated axis.
        """
        if window_s is None:
            return self.archive_times, self.archive, self.archive_steady

        covers = bool(self.recent_times) and (
            self.recent_times[0] <= self.recent_times[-1] - window_s
            or len(self.recent_times) == self._seen
        )
        times, channels, steady = (
            (self.recent_times, self.recent, self.recent_steady)
            if covers
            else (self.archive_times, self.archive, self.archive_steady)
        )
        if not times:
            return times, channels, steady
        start = bisect_left(times, times[-1] - window_s)
        return (
            times[start:],
            {name: series[start:] for name, series in channels.items()},
            steady[start:],
        )


class _StackedPlot:
    """Shared machinery for every stacked, time-axis live window.

    Figure lifecycle, the bounded buffer, the redraw timer and the two display
    modes are identical whether the window is showing a ``collect`` run or a
    ``preview`` of raw signals. What differs is only what gets *drawn on top*
    of the traces, so subclasses override the two annotation hooks rather than
    reimplementing the loop.

    Every method is safe to call when the backend turns out to be unusable --
    the plot degrades to a no-op rather than taking the run down with it.
    """

    def __init__(
        self,
        panels: Sequence[_Panel],
        *,
        window_s: float | None,
        max_points: int,
        redraw_interval_s: float,
        window_title: str,
        plot_config: Any = None,
    ) -> None:
        self._panels = list(panels)
        #: Supplies the monitor and window mode; ``None`` places nothing.
        self._plot_config = plot_config
        self.window_s = window_s
        self.redraw_interval_s = redraw_interval_s
        self._window_title = window_title
        channels = sorted({trace.channel for panel in self._panels for trace in panel.traces})
        self._history = _History(channels, max_points)
        self._last_redraw = 0.0
        self._figure: Any = None
        self._axes: Any = None
        self._plt: Any = None
        self._disabled = False

    # -- lifecycle --------------------------------------------------------

    def open(self) -> Any:
        """Create the figure. Raises :class:`PlottingUnavailable` if it cannot."""
        plt = _pyplot(interactive=True)
        plt.ion()
        count = len(self._panels)
        figure, axes = plt.subplots(
            count, 1, sharex=True, figsize=(9.5, max(3.0, 1.55 * count + 1.4)), squeeze=False
        )
        axes = [row[0] for row in axes]
        try:
            figure.canvas.manager.set_window_title(self._window_title)
        except Exception as exc:  # noqa: BLE001 - some backends have no manager
            logger.debug("Could not set the window title: %s", exc)
        for panel, axis in zip(self._panels, axes):
            axis.set_ylabel(panel.ylabel, fontsize="small")
            axis.grid(True, alpha=0.3)
        axes[-1].set_xlabel("elapsed (s)")
        figure.tight_layout()
        # Reserve the strip the status line is written into. Done once, here,
        # rather than re-running tight_layout on every redraw.
        figure.subplots_adjust(top=1.0 - 0.62 / figure.get_figheight())
        placed = place_figure(figure, self._plot_config)
        if placed:
            logger.info("Plot window %s.", placed)
        self._figure = figure
        self._axes = axes
        self._plt = plt
        return self

    def __enter__(self) -> Any:
        try:
            return self.open()
        except PlottingUnavailable as exc:
            logger.warning("Live plot disabled: %s", exc)
            self._disabled = True
            return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()

    def close(self) -> None:
        """Close the figure, if one was ever created."""
        if self._figure is None:
            return
        try:
            self._plt.close(self._figure)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Error closing the live plot: %s", exc)
        finally:
            self._figure = None

    # -- data flow --------------------------------------------------------

    def maybe_redraw(self, now: float | None = None) -> bool:
        """Redraw if ``redraw_interval_s`` has elapsed. Returns whether it did.

        Called from the loop after the per-sample append; the interval check is
        what keeps plotting off the critical path.
        """
        if self._disabled or self._figure is None or not len(self._history):
            return False
        moment = now if now is not None else time.monotonic()
        if moment - self._last_redraw < self.redraw_interval_s:
            return False
        self._last_redraw = moment
        try:
            self._redraw()
        except Exception as exc:  # noqa: BLE001 - the window may have been closed
            logger.warning("Live plot failed and has been disabled: %s", exc)
            self._disabled = True
            return False
        return True

    # -- drawing ----------------------------------------------------------

    def _redraw(self) -> None:
        times, channels, flags = self._history.view(self.window_s)
        if not times:
            return
        spans = self._spans(times, flags)

        for panel, axis in zip(self._panels, self._axes):
            axis.clear()
            axis.grid(True, alpha=0.3)
            axis.set_ylabel(panel.ylabel, fontsize="small")
            if panel.yscale != "linear":
                axis.set_yscale(panel.yscale)
            for trace in panel.traces:
                axis.plot(
                    times,
                    channels[trace.channel],
                    color=trace.color,
                    linewidth=trace.linewidth,
                    alpha=trace.alpha,
                    label=trace.label,
                )
            # The trace's own y-range, captured before any annotation gets a
            # vote on the autoscale.
            data_limits = axis.get_ylim()
            if panel.y_range is not None:
                data_limits = (
                    min(data_limits[0], panel.y_range[0]),
                    max(data_limits[1], panel.y_range[1]),
                )
                axis.set_ylim(*data_limits)
            for start, end in spans:
                axis.axvspan(start, end, color=_STEADY_COLOR, alpha=0.12, zorder=0)
            self._annotate(panel, axis, data_limits)
            if len(panel.traces) > 1:
                axis.legend(loc="best", fontsize="x-small", framealpha=0.6)

        self._axes[-1].set_xlabel("elapsed (s)")
        self._axes[-1].set_xlim(*self._xlim(times))
        self._figure.suptitle(self._title(), fontsize="medium")
        self._figure.canvas.draw_idle()
        self._figure.canvas.flush_events()

    def _xlim(self, times: Sequence[float]) -> tuple[float, float]:
        """From t0, or the trailing window -- never reaching before the run.

        A window wider than the run so far is clamped at t0 rather than drawing
        empty axis at negative elapsed time, so the view grows into its window
        and then scrolls.
        """
        end = max(times[-1], 1e-6)
        if self.window_s is None:
            return 0.0, end
        return max(0.0, end - self.window_s), end

    # -- hooks ------------------------------------------------------------

    def _spans(
        self, times: Sequence[float], flags: Sequence[bool]
    ) -> list[tuple[float, float]]:
        """Stretches of the x-axis to shade. Nothing, unless a subclass says so."""
        return []

    def _annotate(self, panel: _Panel, axis: Any, data_limits: tuple[float, float]) -> None:
        """Draw anything that goes on top of the traces. A no-op by default."""

    def _title(self) -> str:
        return self._window_title


class LivePlot(_StackedPlot):
    """Live per-parameter view of a ``collect`` run.

    One stacked panel per configured quantity, the steady-state criteria drawn
    on the panels the detector watches, and the confirmed steady stretch
    shaded. Usable as a context manager.
    """

    def __init__(
        self,
        config: GaspermConfig,
        *,
        max_points: int | None = None,
        redraw_interval_s: float | None = None,
        window_s: float | None = None,
        from_start: bool = False,
    ) -> None:
        """Args:
        config: Supplies the panels, display units and criteria; the plot
            never shows CGS.
        max_points: Override ``run.plot.max_points``.
        redraw_interval_s: Override ``run.plot.redraw_interval_s``.
        window_s: Trailing seconds to show, overriding ``run.plot.window_s``.
        from_start: Force the whole-run view, overriding a configured window.
        """
        plot_config = config.run.plot
        self.config = config
        self._status: SteadyStateStatus | None = None
        if from_start:
            resolved_window: float | None = None
        elif window_s is not None:
            resolved_window = window_s
        else:
            resolved_window = plot_config.window_s
        super().__init__(
            _panels_for(config),
            window_s=resolved_window,
            max_points=max_points if max_points is not None else plot_config.max_points,
            redraw_interval_s=(
                redraw_interval_s
                if redraw_interval_s is not None
                else plot_config.redraw_interval_s
            ),
            window_title=f"gasperm - {config.sample.id} ({config.gas.name})",
            plot_config=plot_config,
        )

    def open(self) -> LivePlot:
        """Create the figure. Raises :class:`PlottingUnavailable` if it cannot."""
        super().open()
        return self

    def __enter__(self) -> LivePlot:
        super().__enter__()
        return self

    # -- data flow --------------------------------------------------------

    def add(self, reading: Reading, status: SteadyStateStatus | None = None) -> None:
        """Buffer a reading. O(1), called from the acquisition loop.

        This is the only method the loop calls per sample; it does no drawing.

        Args:
            reading: The sample just taken.
            status: The detector's current verdict, which carries the per-signal
                means and tolerances the criterion lines are drawn from.
        """
        if self._disabled:
            return
        self._history.append(
            reading.elapsed_s,
            _channel_values(reading, self.config.run),
            reading.steady_state,
        )
        if status is not None:
            self._status = status

    # -- drawing ----------------------------------------------------------

    def _spans(
        self, times: Sequence[float], flags: Sequence[bool]
    ) -> list[tuple[float, float]]:
        """Shade the stretch the detector has confirmed steady -- the part of
        the run that will actually be reported."""
        return _steady_spans(times, flags)

    def _annotate(self, panel: _Panel, axis: Any, data_limits: tuple[float, float]) -> None:
        if self.config.run.plot.show_criteria:
            self._draw_criteria(panel, axis, data_limits)

    def _draw_criteria(
        self, panel: _Panel, axis: Any, data_limits: tuple[float, float]
    ) -> None:
        """Draw what the detector is testing this signal against.

        The mean of the trailing window, the scatter tolerance either side of
        it, and the fitted drift line over the window the slope was measured
        on. A signal creeping out of its band is the thing worth catching by
        eye, minutes before the console says anything.

        A panel the detector does not watch says so, rather than silently
        looking like one whose criteria have not arrived yet.
        """
        status = self._status
        criteria = self.config.run.steady_state
        # Unwatched either because the quantity has no criteria at all (the
        # outlet) or because this run's `signals` list leaves it out.
        if self.config.run.method != "steady_state":
            # No detector is running, so "not a steady-state signal" would be
            # true of every panel and mean nothing on any of them.
            return
        if panel.signal is None or panel.signal not in criteria.signals:
            _corner_note(axis, "not a steady-state signal", "0.45")
            return
        if status is None:
            return
        report = next((s for s in status.signals if s.name == panel.signal), None)
        if report is None or not report.sample_count or not math.isfinite(report.mean):
            return

        color = _STEADY_COLOR if report.passed else "tab:orange"
        mean = panel.to_display(report.mean)
        axis.axhline(mean, color=color, linewidth=1.0, alpha=0.8, zorder=1)

        tolerance = criteria.relative_stddev_tolerance
        band = [panel.to_display(report.mean * (1.0 + s * tolerance)) for s in (-1.0, 1.0)]
        for bound in band:
            axis.axhline(
                bound, color=color, linewidth=0.9, linestyle="--", alpha=0.7, zorder=1
            )
        limits, clipped = _limits_with_band(data_limits, min(band), max(band))
        axis.set_ylim(*limits)

        # The OLS line the drift criterion is computed from, over the window it
        # was fitted on. It passes through the window mean at the midpoint.
        window_end = status.elapsed_s
        window_start = window_end - criteria.window_s
        midpoint = (window_start + window_end) / 2.0
        axis.plot(
            [window_start, window_end],
            [
                panel.to_display(report.mean + report.slope_per_s * (window_start - midpoint)),
                panel.to_display(report.mean + report.slope_per_s * (window_end - midpoint)),
            ],
            color=color, linewidth=1.2, linestyle=":", alpha=0.9, zorder=2,
        )

        note = (
            f"scatter {_percent(report.relative_stddev)}/{tolerance:.1%}   "
            f"drift {_percent(report.relative_drift)}/{criteria.relative_drift_tolerance:.1%}"
        )
        if clipped:
            # The band is off-scale by design; say so, so its absence reads as
            # "comfortably inside tolerance" rather than "not drawn yet".
            note += "   (band off-scale)"
        _corner_note(axis, note, color)

    def _title(self) -> str:
        span = "from t0" if self.window_s is None else f"last {self.window_s:g} s"
        head = f"{self.config.sample.id}   {self.config.gas.name}   [{span}]"
        if self.config.run.purpose == "leak_test":
            # An operator glancing at the window must not mistake the pre-step
            # for a measurement.
            head = f"LEAK TEST (the apparatus, not the sample)   {head}"
        if self._status is None:
            return head
        # status.summary already carries the "(n/m)" progress, so it is not
        # repeated here.
        return f"{head}\n{self._status.summary}"


class PreviewPlot(_StackedPlot):
    """Live view of raw rig signals -- ``gasperm preview --plot``.

    One panel per selected signal, in the order they were selected, and
    **nothing drawn on top of them**: no criterion bands, no steady shading, no
    fitted line. Preview runs no detector and computes no result, so every
    annotation ``LivePlot`` adds would be an assertion about something that was
    never tested.
    """

    def __init__(
        self,
        signals: Sequence[Any],
        *,
        volts: bool = False,
        window_s: float | None = None,
        from_start: bool = False,
        max_points: int = DEFAULT_MAX_POINTS,
        redraw_interval_s: float = 0.5,
        device_name: str = "",
        plot_config: Any = None,
    ) -> None:
        """Args:
        signals: :class:`gasperm.preview.PreviewSignal` objects to stack.
        volts: Show the raw voltage rather than the calibrated value.
        window_s: Trailing seconds to show. ``None`` spans the whole session.
        from_start: Force the whole-session view over any window.
        device_name: Named in the window title, so two previews of two rigs
            are told apart.
        """
        from gasperm.preview import preview_color

        self.signals = list(signals)
        self.volts = volts
        panels = [
            _Panel(
                key=signal.key,
                ylabel=f"{signal.label} ({'V' if volts or signal.raw_only else signal.unit})",
                traces=(_Trace(signal.key, signal.label, preview_color(index)),),
                signal=None,
                to_display=lambda value: value,
            )
            for index, signal in enumerate(self.signals)
        ]
        super().__init__(
            panels,
            window_s=None if from_start else window_s,
            max_points=max_points,
            redraw_interval_s=redraw_interval_s,
            window_title=f"gasperm preview{f' - {device_name}' if device_name else ''}",
            plot_config=plot_config,
        )

    def open(self) -> PreviewPlot:
        """Create the figure. Raises :class:`PlottingUnavailable` if it cannot."""
        super().open()
        return self

    def __enter__(self) -> PreviewPlot:
        super().__enter__()
        return self

    def add(self, sample: Any) -> None:
        """Buffer one :class:`gasperm.preview.PreviewSample`. O(1)."""
        if self._disabled:
            return
        values = {}
        for signal in self.signals:
            source = sample.raw if (self.volts or signal.raw_only) else sample.values
            # NaN rather than a dropped point: a probe that said nothing this
            # sample leaves a visible gap instead of a straight line drawn
            # across the silence.
            values[signal.key] = source.get(signal.key, float("nan"))
        self._history.append(sample.elapsed_s, values, False)

    def _title(self) -> str:
        span = "whole session" if self.window_s is None else f"last {self.window_s:g} s"
        mode = "raw volts" if self.volts else "calibrated"
        return f"{self._window_title}   {mode}   [{span}]"


def _corner_note(axis: Any, text: str, color: str) -> None:
    """Bottom-right annotation, legible over whatever the trace is doing."""
    axis.text(
        0.995, 0.04, text,
        transform=axis.transAxes, ha="right", va="bottom",
        fontsize="x-small", color=color,
        bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1.5},
    )


def _limits_with_band(
    data_limits: tuple[float, float], band_low: float, band_high: float
) -> tuple[tuple[float, float], bool]:
    """Fit the criterion band into a panel, unless it would flatten the trace.

    Including the band gives the reading context -- how much headroom is left
    before the signal fails, which is what you want early in a run when the
    signal is still outside it. But once settled the band is often tens of
    times wider than the signal, and letting it drive the y-axis would compress
    the trace into a flat line and hide the drift that matters most.

    Returns the limits and whether the band was left off-scale, so the caller
    can say so rather than leaving the operator to wonder where it went.
    """
    low, high = data_limits
    data_span = high - low
    merged = (min(low, band_low), max(high, band_high))
    if data_span <= 0.0:
        return merged, False
    if (merged[1] - merged[0]) / data_span > _MAX_BAND_ZOOM:
        return data_limits, True
    return merged, False


def _percent(value: float) -> str:
    """Format a ratio that the detector may legitimately report as infinite."""
    return "--" if not math.isfinite(value) else f"{value:.2%}"


def _steady_spans(
    times: Sequence[float], steady: Sequence[bool]
) -> list[tuple[float, float]]:
    """Contiguous ``(start, end)`` stretches flagged steady."""
    spans: list[tuple[float, float]] = []
    start: float | None = None
    for moment, is_steady in zip(times, steady):
        if is_steady and start is None:
            start = moment
        elif not is_steady and start is not None:
            spans.append((start, moment))
            start = None
    if start is not None and times:
        spans.append((start, times[-1]))
    return spans


def plot_klinkenberg(
    result: KlinkenbergResult,
    *,
    path: str | Path | None = None,
    show: bool = False,
    permeability_unit: str = "mD",
    pressure_unit: str = "atm",
    plot_config: Any = None,
) -> Path | None:
    """Plot ``k_g`` against ``1 / P_mean`` with the fitted line.

    The x-axis is extended to zero so the intercept -- which *is* ``k_L`` --
    is visible rather than implied.

    Args:
        result: The fit to draw.
        path: Where to save the PNG. ``None`` skips saving.
        show: Open an interactive window as well.
        permeability_unit: Display unit for the y-axis.
        pressure_unit: Pressure unit the ``1/P`` axis is expressed in.

    Returns:
        The saved path, or ``None`` when nothing was saved.
    """
    plt = _pyplot(interactive=show)

    # 1/P in the display unit: 1/atm -> 1/unit means dividing by the number of
    # display units per atm, which is exactly from_atm(1.0, unit).
    units_per_atm = units.from_atm(1.0, pressure_unit)
    x_values = [point.inverse_mean_pressure / units_per_atm for point in result.points]
    y_values = [
        units.darcy_to(point.apparent_permeability_darcy, permeability_unit)
        for point in result.points
    ]

    figure, axis = plt.subplots(figsize=(7.5, 5))

    # Error bars where the runs carried an uncertainty budget. Without them the
    # eye weights every point equally, which is exactly what the weighted fit
    # is there to avoid.
    y_errors = [
        units.darcy_to(point.standard_uncertainty_darcy, permeability_unit)
        if point.standard_uncertainty_darcy is not None
        else 0.0
        for point in result.points
    ]
    if any(y_errors):
        axis.errorbar(
            x_values, y_values, yerr=y_errors, fmt="none", ecolor="tab:blue",
            elinewidth=1.2, capsize=4, zorder=2,
        )
    axis.scatter(x_values, y_values, color="tab:blue", zorder=3, label="runs")

    x_max = max(x_values) * 1.08 if x_values else 1.0
    fit_x = [0.0, x_max]
    fit_y = [
        units.darcy_to(result.intercept + result.slope * x * units_per_atm, permeability_unit)
        for x in fit_x
    ]
    intercept_display = units.darcy_to(result.liquid_permeability_darcy, permeability_unit)
    axis.plot(
        fit_x,
        fit_y,
        color="tab:red",
        linestyle="--",
        label=(
            f"fit: k_L = {intercept_display:.4g} {permeability_unit}, "
            f"b = {units.from_atm(result.slippage_factor_atm, pressure_unit):.4g} "
            f"{pressure_unit}  (R^2 = {result.r_squared:.4f})"
        ),
    )
    intercept_label = "k_L (1/P -> 0)"
    expanded = result.liquid_permeability_expanded_uncertainty_darcy
    if expanded is not None:
        expanded_display = units.darcy_to(expanded, permeability_unit)
        axis.errorbar(
            [0.0], [intercept_display], yerr=[expanded_display], fmt="none",
            ecolor="tab:red", elinewidth=1.5, capsize=6, zorder=4,
        )
        intercept_label += (
            f" +/- {expanded_display:.3g} (k = {result.coverage_factor:.2f})"
        )
    axis.scatter(
        [0.0], [intercept_display], color="tab:red", marker="*", s=140, zorder=5,
        label=intercept_label,
    )

    for point, x, y in zip(result.points, x_values, y_values):
        if point.label:
            axis.annotate(
                point.label, (x, y), textcoords="offset points", xytext=(6, 5),
                fontsize="x-small",
            )

    axis.set_xlim(left=0.0)
    axis.set_xlabel(f"1 / mean pressure (1/{pressure_unit})")
    axis.set_ylabel(f"apparent gas permeability ({permeability_unit})")
    sample_ids = {p.sample_id for p in result.points if p.sample_id}
    title = "Klinkenberg correction"
    if len(sample_ids) == 1:
        title += f" - {next(iter(sample_ids))}"
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize="small")
    figure.tight_layout()

    saved: Path | None = None
    if path is not None:
        saved = Path(path)
        saved.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(saved, dpi=150)
    if show:
        # Only an interactive figure has a window to place; a saved-only one
        # runs on Agg and has none.
        place_figure(figure, plot_config)
        plt.show()
    else:
        plt.close(figure)
    return saved


def plot_comparison(
    result,
    before: KlinkenbergResult | None = None,
    after: KlinkenbergResult | None = None,
    *,
    path: str | Path | None = None,
    show: bool = False,
    permeability_unit: str = "mD",
    pressure_unit: str = "atm",
    plot_config: Any = None,
) -> Path | None:
    """Plot two campaigns against each other: the fits, and the change.

    Two stacked axes:

    **Top** -- both Klinkenberg series on one ``1/P_mean`` axis, extended to
    zero so the two intercepts are visible side by side. Reading the change off
    the plot is then the same operation as reading either ``k_L``.

    **Bottom** -- the change itself, per matched pressure point, as a percentage
    with its expanded uncertainty. A zero line is drawn: bars that straddle it
    did not resolve a change, and the picture says so before any number is
    read. Points are ordered as they are reported, so the leftmost bar is the
    headline ``k_L``.

    Args:
        result: A :class:`~gasperm.models.ComparisonResult`.
        before: The baseline fit, if there was one.
        after: The comparison fit, if there was one.
        path: Where to save the PNG. ``None`` skips saving.
        show: Open an interactive window as well.
        permeability_unit: Display unit for the permeability axis.
        pressure_unit: Pressure unit the ``1/P`` axis is expressed in.
    """
    plt = _pyplot(interactive=show)
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.5, 8.0), gridspec_kw={"height_ratios": [3, 2]}
    )

    units_per_atm = units.from_atm(1.0, pressure_unit)
    x_max = 0.0
    for fit, label, color in (
        (before, result.before.label, "tab:blue"),
        (after, result.after.label, "tab:red"),
    ):
        if fit is None:
            continue
        xs = [p.inverse_mean_pressure / units_per_atm for p in fit.points]
        ys = [
            units.darcy_to(p.apparent_permeability_darcy, permeability_unit)
            for p in fit.points
        ]
        errors = [
            units.darcy_to(p.standard_uncertainty_darcy, permeability_unit)
            if p.standard_uncertainty_darcy is not None
            else 0.0
            for p in fit.points
        ]
        if any(errors):
            top.errorbar(
                xs, ys, yerr=errors, fmt="none", ecolor=color, elinewidth=1.1,
                capsize=3, zorder=2,
            )
        top.scatter(xs, ys, color=color, zorder=3, label=f"{label} runs")
        x_max = max(x_max, max(xs, default=0.0))

    x_max = x_max * 1.08 if x_max else 1.0
    for fit, label, color in (
        (before, result.before.label, "tab:blue"),
        (after, result.after.label, "tab:red"),
    ):
        if fit is None:
            continue
        intercept = units.darcy_to(fit.liquid_permeability_darcy, permeability_unit)
        top.plot(
            [0.0, x_max],
            [
                intercept,
                units.darcy_to(
                    fit.intercept + fit.slope * x_max * units_per_atm, permeability_unit
                ),
            ],
            color=color, linestyle="--",
            label=f"{label}: k_L = {intercept:.4g} {permeability_unit}",
        )
        expanded = fit.liquid_permeability_expanded_uncertainty_darcy
        if expanded is not None:
            top.errorbar(
                [0.0], [intercept],
                yerr=[units.darcy_to(expanded, permeability_unit)],
                fmt="none", ecolor=color, elinewidth=1.6, capsize=6, zorder=4,
            )
        top.scatter([0.0], [intercept], color=color, marker="*", s=140, zorder=5)

    top.set_xlim(left=0.0)
    top.set_xlabel(f"1 / mean pressure (1/{pressure_unit})")
    top.set_ylabel(f"apparent gas permeability ({permeability_unit})")
    top.set_title(f"{result.before.label}   ->   {result.after.label}")
    top.grid(True, alpha=0.3)
    if before is not None or after is not None:
        top.legend(loc="best", fontsize="x-small")
    else:
        _corner_note(top, "no Klinkenberg fit on either side", "0.45")

    # -- the change itself -------------------------------------------------
    changes = [c for c in result.changes if math.isfinite(c.percent_change)]
    if changes:
        positions = list(range(len(changes)))
        values = [c.percent_change for c in changes]
        errors = [
            c.relative_expanded_uncertainty * 100.0
            if math.isfinite(c.relative_expanded_uncertainty)
            else 0.0
            for c in changes
        ]
        colors = [
            "tab:green" if c.significant else "tab:orange" for c in changes
        ]
        bottom.bar(positions, values, color=colors, alpha=0.55, zorder=2)
        bottom.errorbar(
            positions, values, yerr=errors, fmt="none", ecolor="black",
            elinewidth=1.3, capsize=5, zorder=3,
        )
        # No change is the reference, not zero-on-an-arbitrary-axis: a bar whose
        # interval crosses this line did not resolve anything.
        bottom.axhline(0.0, color="black", linewidth=1.0, zorder=1)
        bottom.set_xticks(positions)
        bottom.set_xticklabels(
            [c.symbol if len(changes) > 6 else f"{c.symbol}\n{_short_label(c)}"
             for c in changes],
            fontsize="x-small",
        )
        bottom.set_ylabel("change (%)")
        bottom.grid(True, axis="y", alpha=0.3)
        _corner_note(
            bottom,
            "green = resolved   amber = inside the uncertainty",
            "0.35",
        )
    else:
        _corner_note(bottom, "nothing comparable", "0.45")

    figure.tight_layout()
    saved: Path | None = None
    if path is not None:
        saved = Path(path)
        saved.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(saved, dpi=150)
    if show:
        # Only an interactive figure has a window to place; a saved-only one
        # runs on Agg and has none.
        place_figure(figure, plot_config)
        plt.show()
    else:
        plt.close(figure)
    return saved


def _short_label(change) -> str:
    """A tick label for one change: the pressure for a k_g point, else the unit."""
    if change.symbol == "k_g" and " at " in change.name:
        return change.name.split(" at ", 1)[1]
    return change.unit


def plot_pulse_decay(
    result,
    readings: Sequence[Reading],
    *,
    path: str | Path | None = None,
    show: bool = False,
    pressure_unit: str = "kPa",
    plot_config: Any = None,
) -> Path | None:
    """Plot the fitted decay and its residuals.

    Two stacked axes. The top shows dP against time on a **log** y-axis with the
    fitted curve over it and the fit window shaded: on a log axis a single
    exponential is a straight line, so a curve that bends is telling you the
    model does not fit. The bottom shows the residuals, which is where a thermal
    ramp or a leak appears as structure rather than as noise -- the two failures
    that a good R^2 will happily hide.

    Args:
        result: A :class:`~gasperm.models.PulseDecayResult`.
        readings: The run's readings, for the measured differential.
        path: Where to save the PNG. ``None`` skips saving.
        show: Open an interactive window as well.
        pressure_unit: Display unit for the differential.

    Returns:
        The saved path, or ``None`` when nothing was saved.
    """
    plt = _pyplot(interactive=show)

    times = [r.elapsed_s for r in readings]
    deltas = [units.from_atm(r.delta_pressure_atm, pressure_unit) for r in readings]
    offset = units.from_atm(result.fitted_offset_atm or 0.0, pressure_unit)
    amplitude = units.from_atm(
        result.pulse_amplitude_atm, pressure_unit
    )

    figure, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 6.5), height_ratios=(3, 1)
    )

    top.plot(times, deltas, color="tab:blue", linewidth=1.0, label="measured dP")
    inside = [
        (t, d)
        for t, d in zip(times, deltas)
        if result.fit_start_elapsed_s <= t <= result.fit_end_elapsed_s
    ]
    if inside:
        fitted_amplitude = units.from_atm(
            result.pulse_amplitude_atm, pressure_unit
        )
        # Re-evaluate the fitted model over the window, from its own start.
        curve_t = [t for t, _ in inside]
        start = result.fit_start_elapsed_s
        measured_start = inside[0][1]
        curve = [
            (measured_start - offset) * math.exp(-result.decay_rate_per_s * (t - start))
            + offset
            for t in curve_t
        ]
        top.plot(
            curve_t, curve, color="tab:red", linestyle="--", linewidth=1.6,
            label=(
                f"fit: alpha = {result.decay_rate_per_s:.4e} 1/s, "
                f"tau = {result.time_constant_s:.0f} s  "
                f"(R^2 = {result.r_squared:.5f})"
            ),
        )
        top.axvspan(
            result.fit_start_elapsed_s, result.fit_end_elapsed_s,
            color="tab:green", alpha=0.10, zorder=0, label="fit window",
        )
        residuals = [d - c for (_, d), c in zip(inside, curve)]
        bottom.axhline(0.0, color="0.6", linewidth=0.8)
        bottom.plot(curve_t, residuals, color="tab:red", linewidth=0.9)
        bottom.set_ylabel(f"residual ({pressure_unit})", fontsize="small")
        _ = fitted_amplitude

    if result.fitted_offset_atm is not None:
        top.axhline(
            offset, color="tab:orange", linestyle=":", linewidth=1.2,
            label=f"fitted offset = {offset:+.4g} {pressure_unit}",
        )

    # Log y so a single exponential reads as a straight line. Only the positive
    # part can be shown; the offset makes late samples cross zero, and that is
    # precisely what the residual panel below is for.
    if any(d > 0 for d in deltas):
        top.set_yscale("log")
    top.set_ylabel(f"dP ({pressure_unit})")
    top.grid(True, alpha=0.3, which="both")
    top.legend(loc="upper right", fontsize="small")
    top.set_title(
        f"Pulse decay - dP0 = {amplitude:.4g} {pressure_unit}, "
        f"{result.storage_correction.replace('_', '-')} model"
    )
    bottom.grid(True, alpha=0.3)
    bottom.set_xlabel("elapsed (s)")
    figure.tight_layout()

    saved: Path | None = None
    if path is not None:
        saved = Path(path)
        saved.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(saved, dpi=150)
    if show:
        # Only an interactive figure has a window to place; a saved-only one
        # runs on Agg and has none.
        place_figure(figure, plot_config)
        plt.show()
    else:
        plt.close(figure)
    return saved
