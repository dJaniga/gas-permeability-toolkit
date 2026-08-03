"""Optional matplotlib views: the live ``--plot`` window and the Klinkenberg fit.

Plotting is strictly additive. The console output is the primary display and
``--plot`` only adds a window on top of it; nothing here is required for a run
to produce correct data.

**The live plot must never slow the acquisition loop.** Points go into a
bounded deque on the loop's thread -- an O(1) append -- and the figure is
redrawn on a timer (``redraw_interval_s``), not once per sample. At 10 Hz a
per-sample redraw would spend more time in matplotlib than in the DAQ. If the
redraw itself fails (window closed by the operator, no display available), the
error is swallowed and the run continues.

matplotlib is imported lazily so the package works headless.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

from gasperm import units
from gasperm.config import GaspermConfig
from gasperm.models import KlinkenbergResult, Reading

logger = logging.getLogger(__name__)

__all__ = ["LivePlot", "plot_klinkenberg", "PlottingUnavailable"]

#: How many points the live window keeps. Bounded so a multi-hour run cannot
#: grow the figure's memory without limit.
DEFAULT_MAX_POINTS = 3600


class PlottingUnavailable(RuntimeError):
    """matplotlib is not installed or no usable backend is available."""


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


class LivePlot:
    """Live view of pressures, flow and permeability during a ``collect`` run.

    Usable as a context manager. Every method is safe to call when the backend
    turns out to be unusable -- the plot degrades to a no-op rather than taking
    the run down with it.
    """

    def __init__(
        self,
        config: GaspermConfig,
        *,
        max_points: int = DEFAULT_MAX_POINTS,
        redraw_interval_s: float = 0.5,
    ) -> None:
        """Args:
        config: Supplies the display units; the plot never shows CGS.
        max_points: Bound on the deque backing each series.
        redraw_interval_s: Minimum wall-clock gap between redraws.
        """
        self.config = config
        self.redraw_interval_s = redraw_interval_s
        self._times: deque[float] = deque(maxlen=max_points)
        self._inlet: deque[float] = deque(maxlen=max_points)
        self._outlet: deque[float] = deque(maxlen=max_points)
        self._flow: deque[float] = deque(maxlen=max_points)
        self._permeability: deque[float] = deque(maxlen=max_points)
        self._steady: deque[bool] = deque(maxlen=max_points)
        self._last_redraw = 0.0
        self._figure: Any = None
        self._axes: Any = None
        self._disabled = False

    # -- lifecycle --------------------------------------------------------

    def open(self) -> LivePlot:
        """Create the figure. Raises :class:`PlottingUnavailable` if it cannot."""
        plt = _pyplot(interactive=True)
        plt.ion()
        figure, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 7))
        figure.canvas.manager.set_window_title(
            f"gasperm - {self.config.sample.id} ({self.config.gas.name})"
        )
        run = self.config.run
        axes[0].set_ylabel(f"pressure ({run.display_pressure_unit})")
        axes[1].set_ylabel(f"flow ({run.display_flow_unit})")
        axes[2].set_ylabel(f"k ({run.display_permeability_unit})")
        axes[2].set_xlabel("elapsed (s)")
        for axis in axes:
            axis.grid(True, alpha=0.3)
        figure.tight_layout()
        self._figure = figure
        self._axes = axes
        self._plt = plt
        return self

    def __enter__(self) -> LivePlot:
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

    def add(self, reading: Reading) -> None:
        """Buffer a reading. O(1), called from the acquisition loop.

        This is the only method the loop calls per sample; it does no drawing.
        """
        if self._disabled:
            return
        run = self.config.run
        self._times.append(reading.elapsed_s)
        self._inlet.append(units.from_atm(reading.inlet_pressure_atm, run.display_pressure_unit))
        self._outlet.append(
            units.from_atm(reading.downstream_pressure_atm, run.display_pressure_unit)
        )
        self._flow.append(units.flow_from_cm3_s(reading.flow_cm3_s, run.display_flow_unit))
        value = reading.permeability_darcy_avg
        self._permeability.append(
            units.darcy_to(value, run.display_permeability_unit)
            if value is not None
            else float("nan")
        )
        self._steady.append(reading.steady_state)

    def maybe_redraw(self, now: float | None = None) -> bool:
        """Redraw if ``redraw_interval_s`` has elapsed. Returns whether it did.

        Called from the loop after :meth:`add`; the interval check is what
        keeps plotting off the critical path.
        """
        if self._disabled or self._figure is None or not self._times:
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

    def _redraw(self) -> None:
        times = list(self._times)
        pressure_axis, flow_axis, permeability_axis = self._axes
        for axis in self._axes:
            axis.clear()
            axis.grid(True, alpha=0.3)

        run = self.config.run
        pressure_axis.plot(times, list(self._inlet), label="inlet (P1)")
        pressure_axis.plot(times, list(self._outlet), label="outlet (P2)")
        pressure_axis.set_ylabel(f"pressure ({run.display_pressure_unit})")
        pressure_axis.legend(loc="upper left", fontsize="small")

        flow_axis.plot(times, list(self._flow), color="tab:green")
        flow_axis.set_ylabel(f"flow ({run.display_flow_unit})")

        permeability_axis.plot(times, list(self._permeability), color="tab:red")
        permeability_axis.set_ylabel(f"k ({run.display_permeability_unit})")
        permeability_axis.set_xlabel("elapsed (s)")

        # Shade the stretch the detector has confirmed steady -- the part of
        # the run that will actually be reported.
        for start, end in self._steady_spans(times):
            for axis in self._axes:
                axis.axvspan(start, end, color="tab:green", alpha=0.12, zorder=0)
        if any(self._steady):
            permeability_axis.legend(
                handles=[
                    self._plt.Line2D(
                        [], [], color="tab:green", alpha=0.35, linewidth=8,
                        label="steady state (reported)",
                    )
                ],
                loc="lower right",
                fontsize="small",
            )

        self._figure.canvas.draw_idle()
        self._figure.canvas.flush_events()

    def _steady_spans(self, times: list[float]) -> list[tuple[float, float]]:
        """Contiguous ``(start, end)`` stretches flagged steady."""
        spans: list[tuple[float, float]] = []
        start: float | None = None
        for moment, steady in zip(times, self._steady):
            if steady and start is None:
                start = moment
            elif not steady and start is not None:
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
        plt.show()
    else:
        plt.close(figure)
    return saved
