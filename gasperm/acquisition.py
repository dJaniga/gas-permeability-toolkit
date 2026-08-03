"""Real-time sampling loop, rolling averages, and live permeability.

Split deliberately in two:

* :class:`SampleProcessor` -- pure computation. Takes raw voltages and a
  temperature, returns a :class:`~gasperm.models.Reading`. No device imports,
  no I/O, no clock. This is where every unit conversion and the Darcy call
  happen, and it is directly unit-testable.
* :class:`AcquisitionLoop` -- the timing, the sources, the CSV writer and the
  plot queue. Talks to its inputs through the
  :class:`~gasperm.hardware.daq.AnalogInputSource` and
  :class:`~gasperm.hardware.temperature.TemperatureSource` protocols, so the
  tests drive it with fakes.
"""

from __future__ import annotations

import logging
import math
import signal
import statistics
import time
from collections import deque
from datetime import datetime, timezone
from typing import Callable, Sequence

from gasperm import units
from gasperm.config import GaspermConfig
from gasperm.gas_properties import GasPropertyProvider
from gasperm.hardware.daq import AnalogInputSource
from gasperm.hardware.flowmeter import FlowChannel
from gasperm.hardware.pressure import PressureChannel
from gasperm.hardware.temperature import TemperatureSample, TemperatureSource
from gasperm.models import Reading, RunSummary
from gasperm.permeability import (
    PermeabilityInputError,
    compute_gas_permeability,
    mean_pressure,
)

logger = logging.getLogger(__name__)

__all__ = [
    "RollingWindow",
    "SampleProcessor",
    "AcquisitionLoop",
    "trailing_window_mean",
    "steady_state_stats",
]


# --------------------------------------------------------------------------
# Rolling average
# --------------------------------------------------------------------------


class RollingWindow:
    """Time-based rolling window over ``(time, value)`` pairs.

    Used for the live permeability display, for the run summary, and -- via
    :func:`steady_state_stats` -- by ``klinkenberg`` when reducing a stored run
    to a single point, so all three share one definition of "the trailing
    N seconds".
    """

    def __init__(self, window_s: float) -> None:
        if window_s <= 0.0:
            raise ValueError(f"window_s must be positive, got {window_s}")
        self.window_s = window_s
        self._points: deque[tuple[float, float]] = deque()

    def add(self, timestamp_s: float, value: float) -> None:
        """Add a sample, dropping anything older than the window.

        Non-finite values are ignored rather than poisoning the mean; a run
        with a few unusable samples still reports a sensible average.
        """
        if not math.isfinite(value):
            return
        self._points.append((timestamp_s, value))
        cutoff = timestamp_s - self.window_s
        while self._points and self._points[0][0] < cutoff:
            self._points.popleft()

    def clear(self) -> None:
        """Drop every buffered point."""
        self._points.clear()

    @property
    def values(self) -> list[float]:
        """Values currently inside the window."""
        return [value for _, value in self._points]

    @property
    def count(self) -> int:
        """How many samples are inside the window."""
        return len(self._points)

    def mean(self) -> float | None:
        """Arithmetic mean of the window, or ``None`` when empty."""
        values = self.values
        return statistics.fmean(values) if values else None

    def stddev(self) -> float | None:
        """Sample standard deviation, or ``None`` with fewer than two points."""
        values = self.values
        return statistics.stdev(values) if len(values) > 1 else None


def trailing_window_mean(
    timestamps: Sequence[float], values: Sequence[float], window_s: float
) -> float | None:
    """Mean of ``values`` over the last ``window_s`` seconds of ``timestamps``."""
    window = RollingWindow(window_s)
    for timestamp, value in zip(timestamps, values):
        window.add(timestamp, value)
    return window.mean()


def steady_state_stats(
    timestamps: Sequence[float], values: Sequence[float], window_s: float
) -> tuple[float | None, float | None, int]:
    """``(mean, stddev, n)`` over the trailing ``window_s`` seconds.

    The same reduction ``collect`` uses for its run summary and ``klinkenberg``
    uses to turn a stored run into one regression point -- deliberately shared
    rather than reimplemented, so the two never disagree about what
    "steady-state" means.
    """
    window = RollingWindow(window_s)
    for timestamp, value in zip(timestamps, values):
        window.add(timestamp, value)
    return window.mean(), window.stddev(), window.count


# --------------------------------------------------------------------------
# Pure per-sample computation
# --------------------------------------------------------------------------


class SampleProcessor:
    """Turns raw voltages plus a temperature into a :class:`Reading`.

    Holds the calibrated channels and the gas property provider; owns every
    unit conversion between the hardware boundary and the CGS physics call.
    """

    def __init__(self, config: GaspermConfig, gas_provider: GasPropertyProvider) -> None:
        self.config = config
        self.gas_provider = gas_provider

        atmospheric_atm = config.run.atmospheric_pressure_atm
        self.atmospheric_pressure_atm = atmospheric_atm
        self.inlet = PressureChannel.from_config(
            "inlet",
            config.daq.inlet_pressure_channel,
            config.pressure_calibration.inlet,
            atmospheric_atm,
        )
        self.outlet = PressureChannel.from_config(
            "outlet",
            config.daq.outlet_pressure_channel,
            config.pressure_calibration.outlet,
            atmospheric_atm,
        )
        self.flow = FlowChannel.from_config(config.flowmeter)

        geometry = config.geometry()
        self.length_cm = geometry.length_cm
        self.area_cm2 = geometry.area_cm2

        self._window = RollingWindow(config.run.averaging_window_s)

    # -- helpers ----------------------------------------------------------

    def resolve_downstream_pressure_atm(self, measured_outlet_atm: float) -> float:
        """P2 for the Darcy equation, per ``run.outlet_pressure_reference``.

        ``"measured"`` trusts the outlet transducer; ``"atmospheric"`` uses the
        configured ambient (correct for a rig venting to atmosphere, where the
        outlet transducer may be absent or reading noise around zero); a
        number pins it to a fixed back-pressure.
        """
        reference = self.config.run.outlet_pressure_reference
        if reference == "measured":
            return measured_outlet_atm
        if reference == "atmospheric":
            return self.atmospheric_pressure_atm
        fixed = self.config.run.fixed_outlet_pressure_atm
        assert fixed is not None  # guaranteed by the config model
        return fixed

    def reset_window(self) -> None:
        """Clear the rolling average, e.g. between runs."""
        self._window.clear()

    @property
    def rolling_count(self) -> int:
        """Samples currently inside the rolling window."""
        return self._window.count

    # -- the per-sample computation ---------------------------------------

    def process(
        self,
        *,
        index: int,
        elapsed_s: float,
        voltages: dict[str, float],
        temperature: TemperatureSample,
        timestamp: datetime | None = None,
    ) -> Reading:
        """Compute one :class:`Reading` from one set of raw voltages.

        Args:
            index: Zero-based sample number.
            elapsed_s: Seconds since the run started -- also the rolling
                window's time base.
            voltages: ``{channel_name: volts}`` covering both pressure
                channels and the active flow channel.
            temperature: Latest probe state; may be missing or stale.
            timestamp: Wall-clock time for the log. Defaults to now (UTC).

        Returns:
            A fully-populated reading. ``permeability_darcy`` is ``None`` with
            an explanatory ``note`` when the sample cannot be inverted (no
            differential yet, transducer noise below zero, and so on) -- that
            is normal early in a run and must not abort it.

        Raises:
            KeyError: a configured channel is missing from ``voltages``.
        """
        inlet_volts = self._require(voltages, self.inlet.channel, "inlet pressure")
        outlet_volts = self._require(voltages, self.outlet.channel, "outlet pressure")
        flow_volts = self._require(voltages, self.flow.channel, "flow")

        inlet_atm = self.inlet.volts_to_absolute_atm(inlet_volts)
        measured_outlet_atm = self.outlet.volts_to_absolute_atm(outlet_volts)
        downstream_atm = self.resolve_downstream_pressure_atm(measured_outlet_atm)
        mean_atm = mean_pressure(inlet_atm, downstream_atm)

        flow_cm3_s = self.flow.volts_to_cm3_s(flow_volts)
        reference_pressure_atm = self.flow.reference_pressure_atm(
            inlet_pressure_atm=inlet_atm,
            outlet_pressure_atm=measured_outlet_atm,
            atmospheric_atm=self.atmospheric_pressure_atm,
        )

        temperature_c = temperature.temperature_c
        temperature_ok = temperature_c is not None
        if temperature_c is None:
            temperature_c = self.config.temperature.fallback_temperature_c

        # Viscosity is evaluated per reading at the mean pore pressure and the
        # current temperature -- both drift during a run, especially before the
        # rig settles. See gas_properties for why the mean pressure is the
        # documented choice.
        gas_state = self.gas_provider.state_at_cgs(temperature_c, max(mean_atm, 1e-9))

        permeability: float | None = None
        note: str | None = None
        try:
            permeability = compute_gas_permeability(
                flow_rate_cm3_s=flow_cm3_s,
                reference_pressure_atm=reference_pressure_atm,
                viscosity_cp=gas_state.viscosity_cp,
                length_cm=self.length_cm,
                area_cm2=self.area_cm2,
                inlet_pressure_atm=inlet_atm,
                outlet_pressure_atm=downstream_atm,
            )
        except PermeabilityInputError as exc:
            note = str(exc).split(".")[0]

        if permeability is not None:
            self._window.add(elapsed_s, permeability)

        return Reading(
            index=index,
            timestamp=timestamp or datetime.now(timezone.utc),
            elapsed_s=elapsed_s,
            inlet_voltage=inlet_volts,
            outlet_voltage=outlet_volts,
            flow_voltage=flow_volts,
            temperature_raw=temperature.raw_line,
            inlet_pressure_atm=inlet_atm,
            outlet_pressure_atm=measured_outlet_atm,
            downstream_pressure_atm=downstream_atm,
            mean_pressure_atm=mean_atm,
            flow_cm3_s=flow_cm3_s,
            flow_reference_cm3_s=flow_cm3_s,
            flow_reference_pressure_atm=reference_pressure_atm,
            temperature_c=temperature_c,
            temperature_ok=temperature_ok,
            temperature_stale=temperature.stale or not temperature_ok,
            viscosity_cp=gas_state.viscosity_cp,
            permeability_darcy=permeability,
            permeability_darcy_avg=self._window.mean(),
            note=note,
        )

    @staticmethod
    def _require(voltages: dict[str, float], channel: str, role: str) -> float:
        try:
            return voltages[channel]
        except KeyError as exc:
            raise KeyError(
                f"No voltage for the {role} channel {channel!r}; the DAQ returned "
                f"{sorted(voltages)}."
            ) from exc


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


class AcquisitionLoop:
    """Drives sampling at ``daq.sample_rate_hz`` until stopped.

    Stops on Ctrl+C, on ``run.duration_s``, or on ``run.max_samples``,
    whichever comes first, and always closes its sources.
    """

    def __init__(
        self,
        config: GaspermConfig,
        processor: SampleProcessor,
        analog_source: AnalogInputSource,
        temperature_source: TemperatureSource,
        *,
        on_reading: Callable[[Reading], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Args:
        config: Validated rig configuration.
        processor: The pure per-sample computation.
        analog_source: Anything satisfying ``AnalogInputSource``.
        temperature_source: Anything satisfying ``TemperatureSource``.
        on_reading: Called with each reading -- console output, CSV writer,
            plot queue. Exceptions raised here are logged and swallowed so a
            display problem cannot end a run.
        clock: Monotonic time source. Injectable for deterministic tests.
        sleep: Sleep function. Injectable for deterministic tests.
        """
        self.config = config
        self.processor = processor
        self.analog_source = analog_source
        self.temperature_source = temperature_source
        self.on_reading = on_reading
        self._clock = clock
        self._sleep = sleep

        self.readings: list[Reading] = []
        self.warnings: list[str] = []
        self._stop_requested = False
        self.started_at: datetime | None = None
        self.ended_at: datetime | None = None

    def request_stop(self) -> None:
        """Ask the loop to finish after the current sample."""
        self._stop_requested = True

    # -- main loop --------------------------------------------------------

    def run(self, *, install_signal_handler: bool = True) -> list[Reading]:
        """Sample until a stop condition is met.

        Args:
            install_signal_handler: Install a SIGINT handler that requests a
                clean stop instead of unwinding mid-write. Disabled in tests
                and when not on the main thread.

        Returns:
            Every reading taken, in order.
        """
        interval_s = 1.0 / self.config.daq.sample_rate_hz
        duration_s = self.config.run.duration_s
        max_samples = self.config.run.max_samples

        previous_handler = None
        if install_signal_handler:
            previous_handler = self._install_sigint()

        self.started_at = datetime.now(timezone.utc)
        start = self._clock()
        index = 0
        try:
            while not self._stop_requested:
                if max_samples is not None and index >= max_samples:
                    break
                target = start + index * interval_s
                elapsed = self._clock() - start
                if duration_s is not None and elapsed >= duration_s:
                    break

                reading = self._sample_once(index, elapsed)
                if reading is not None:
                    self.readings.append(reading)
                    self._emit(reading)
                index += 1

                # Sleep to the next slot rather than a fixed interval, so a
                # slow sample does not make the whole run drift late.
                remaining = (target + interval_s) - self._clock()
                if remaining > 0:
                    self._sleep(remaining)
        finally:
            self.ended_at = datetime.now(timezone.utc)
            if previous_handler is not None:
                signal.signal(signal.SIGINT, previous_handler)
            self.close()
        return self.readings

    def _sample_once(self, index: int, elapsed: float) -> Reading | None:
        """One acquisition step. Returns ``None`` if the sample was unusable."""
        try:
            voltages = self.analog_source.read()
        except Exception as exc:  # noqa: BLE001 - DaqError or a driver error
            # A DAQ failure is fatal in a way a serial dropout is not: without
            # pressures and flow there is nothing to record.
            self._record_warning(f"DAQ read failed at sample {index}: {exc}")
            raise

        temperature = self.temperature_source.latest()
        if temperature.temperature_c is None:
            self._record_warning_once(
                "temperature-missing",
                "No temperature reading available; using "
                f"temperature.fallback_temperature_c = "
                f"{self.config.temperature.fallback_temperature_c} degC for viscosity.",
            )
        elif temperature.stale:
            self._record_warning_once(
                "temperature-stale",
                f"Temperature probe has gone quiet; reusing the last value "
                f"({temperature.temperature_c:.2f} degC).",
            )

        return self.processor.process(
            index=index,
            elapsed_s=elapsed,
            voltages=voltages,
            temperature=temperature,
        )

    def _emit(self, reading: Reading) -> None:
        if self.on_reading is None:
            return
        try:
            self.on_reading(reading)
        except Exception as exc:  # noqa: BLE001 - display must never kill a run
            logger.warning("Reading handler failed at sample %d: %s", reading.index, exc)

    def _install_sigint(self):
        def handler(signum, frame):  # noqa: ANN001, ARG001
            logger.info("Stop requested; finishing the current sample.")
            self.request_stop()

        try:
            return signal.signal(signal.SIGINT, handler)
        except ValueError:
            # Not on the main thread; the caller can still use request_stop().
            return None

    def close(self) -> None:
        """Close both sources, reporting but not re-raising failures."""
        for name, source in (
            ("temperature source", self.temperature_source),
            ("analog source", self.analog_source),
        ):
            try:
                source.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing the %s: %s", name, exc)

    # -- warnings ---------------------------------------------------------

    def _record_warning(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.warnings.append(f"[{stamp}] {message}")
        logger.warning("%s", message)

    def _record_warning_once(self, key: str, message: str) -> None:
        """Log a recurring condition once, so a long run's log stays readable."""
        seen = getattr(self, "_warned_keys", None)
        if seen is None:
            seen = set()
            self._warned_keys = seen
        if key in seen:
            return
        seen.add(key)
        self._record_warning(message)

    # -- summary ----------------------------------------------------------

    def summarize(self, csv_path: str | None = None) -> RunSummary:
        """Reduce the run to its steady-state result.

        The trailing ``run.averaging_window_s`` is treated as steady state --
        the same reduction ``klinkenberg`` applies to a stored run.

        Raises:
            ValueError: no reading produced a usable permeability.
        """
        usable = [r for r in self.readings if r.permeability_darcy is not None]
        if not usable:
            raise ValueError(
                "No sample produced a usable permeability. Check that inlet pressure "
                "exceeded outlet pressure and that gas was flowing."
            )

        timestamps = [r.elapsed_s for r in usable]
        permeabilities = [r.permeability_darcy for r in usable]
        mean_k, stddev_k, n = steady_state_stats(
            timestamps, permeabilities, self.config.run.averaging_window_s
        )
        mean_p, _, _ = steady_state_stats(
            timestamps, [r.mean_pressure_atm for r in usable],
            self.config.run.averaging_window_s,
        )
        mean_t, _, _ = steady_state_stats(
            timestamps, [r.temperature_c for r in usable],
            self.config.run.averaging_window_s,
        )
        mean_q, _, _ = steady_state_stats(
            timestamps, [r.flow_cm3_s for r in usable],
            self.config.run.averaging_window_s,
        )

        started = self.started_at or usable[0].timestamp
        ended = self.ended_at or usable[-1].timestamp
        return RunSummary(
            sample_id=self.config.sample.id,
            gas_name=self.config.gas.name,
            started_at=started,
            ended_at=ended,
            duration_s=usable[-1].elapsed_s,
            sample_count=len(self.readings),
            mean_pressure_atm=float(mean_p or 0.0),
            permeability_darcy=float(mean_k or 0.0),
            permeability_stddev_darcy=float(stddev_k or 0.0),
            mean_temperature_c=float(mean_t or 0.0),
            mean_flow_cm3_s=float(mean_q or 0.0),
            averaged_samples=n,
            csv_path=csv_path,
            warnings=list(self.warnings),
        )


def format_reading_line(reading: Reading, config: GaspermConfig) -> str:
    """One console line for a reading, in the configured **display** units.

    Display units are decoupled from both the calibration units and the
    internal CGS calculation, so changing ``display_pressure_unit`` never
    touches a number that feeds the physics.
    """
    run = config.run
    pressure_unit = run.display_pressure_unit
    permeability_unit = run.display_permeability_unit
    flow_unit = run.display_flow_unit

    p1 = units.from_atm(reading.inlet_pressure_atm, pressure_unit)
    p2 = units.from_atm(reading.downstream_pressure_atm, pressure_unit)
    flow = units.flow_from_cm3_s(reading.flow_cm3_s, flow_unit)

    if reading.permeability_darcy_avg is not None:
        k_display = units.darcy_to(reading.permeability_darcy_avg, permeability_unit)
        k_text = f"{k_display:>11.4g} {permeability_unit}"
    else:
        k_text = f"{'--':>11} {permeability_unit}"

    temperature_flag = "" if reading.temperature_ok and not reading.temperature_stale else "*"
    line = (
        f"{reading.elapsed_s:7.1f}s  "
        f"P1 {p1:9.3f}  P2 {p2:9.3f} {pressure_unit}  "
        f"Q {flow:9.3f} {flow_unit}  "
        f"T {reading.temperature_c:6.2f}{temperature_flag:1}C  "
        f"mu {reading.viscosity_cp:.5f} cP  "
        f"k {k_text}"
    )
    if reading.note:
        line += f"   [{reading.note}]"
    return line


def console_header(config: GaspermConfig) -> str:
    """Header line matching :func:`format_reading_line`'s column layout."""
    return (
        f"{'time':>8}  {'inlet':>12} {'outlet':>12} "
        f"({config.run.display_pressure_unit})  "
        f"{'flow':>11} ({config.run.display_flow_unit})  "
        f"{'temp':>8}  {'viscosity':>11}  "
        f"{'permeability':>14} ({config.run.display_permeability_unit})"
    )
