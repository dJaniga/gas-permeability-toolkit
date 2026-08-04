"""Real-time sampling loop, steady-state gating, and live permeability.

Split deliberately in three:

* :class:`SampleProcessor` -- pure computation. Raw voltages plus a temperature
  in, one :class:`~gasperm.models.Reading` out. No device imports, no I/O, no
  clock. Every unit conversion and the Darcy call live here.
* :class:`AcquisitionLoop` -- timing, sources, the CSV writer, the plot queue,
  and the steady-state detector. Talks to its inputs through the
  ``AnalogInputSource`` / ``TemperatureSource`` protocols, so tests drive it
  with fakes.
* :func:`summarize_run` -- reduces a finished run to its reported result,
  taken from the **detected steady-state window** and accompanied by a GUM
  uncertainty budget.

The reported permeability comes only from the steady window: a value measured
while the rig is still equilibrating describes the transient, not the rock.
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
from gasperm.config import GaspermConfig, experiment_metadata
from gasperm.gas_properties import GasPropertyProvider
from gasperm.hardware.daq import AnalogInputSource
from gasperm.hardware.flowmeter import FlowChannel
from gasperm.hardware.pressure import PressureChannel
from gasperm.hardware.temperature import TemperatureSample, TemperatureSource
from gasperm.models import (
    Reading,
    RunSummary,
    SteadyStateStatus,
    SteadyStateWindow,
    UncertaintyBudget,
)
from gasperm.permeability import (
    PermeabilityInputError,
    compute_gas_permeability,
    mean_pressure,
)
from gasperm.steady_state import SteadyStateDetector, signals_from_reading
from gasperm.uncertainty import MeasurementPoint, build_budget

logger = logging.getLogger(__name__)

#: How far a supplied downstream pressure may sit from the outlet transducer
#: before the run summary says so. Loose enough to tolerate a transducer that is
#: uncalibrated near ambient, tight enough to catch a closed valve.
DOWNSTREAM_MISMATCH_TOLERANCE = 0.05

__all__ = [
    "RollingWindow",
    "SampleProcessor",
    "AcquisitionLoop",
    "summarize_run",
    "trailing_window_mean",
    "steady_state_stats",
    "format_reading_line",
    "console_header",
]


# --------------------------------------------------------------------------
# Rolling average
# --------------------------------------------------------------------------


class RollingWindow:
    """Time-based rolling window over ``(time, value)`` pairs.

    Drives the live display. The *reported* result does not come from here --
    it comes from the steady-state window -- but a rolling mean is what makes
    the console readable while the rig settles.
    """

    def __init__(self, window_s: float) -> None:
        if window_s <= 0.0:
            raise ValueError(f"window_s must be positive, got {window_s}")
        self.window_s = window_s
        self._points: deque[tuple[float, float]] = deque()

    def add(self, timestamp_s: float, value: float) -> None:
        """Add a sample, dropping anything older than the window.

        Non-finite values are ignored rather than poisoning the mean.
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
    """``(mean, stddev, n)`` over the trailing ``window_s`` seconds."""
    window = RollingWindow(window_s)
    for timestamp, value in zip(timestamps, values):
        window.add(timestamp, value)
    return window.mean(), window.stddev(), window.count


def _mean_stddev(values: Sequence[float]) -> tuple[float, float]:
    """``(mean, sample stddev)``; stddev is 0 for fewer than two values."""
    if not values:
        return 0.0, 0.0
    mean = statistics.fmean(values)
    return mean, (statistics.stdev(values) if len(values) > 1 else 0.0)


# --------------------------------------------------------------------------
# Pure per-sample computation
# --------------------------------------------------------------------------


class SampleProcessor:
    """Turns raw voltages plus a temperature into a :class:`Reading`."""

    def __init__(self, config: GaspermConfig, gas_provider: GasPropertyProvider) -> None:
        self.config = config
        self.gas_provider = gas_provider

        atmospheric_atm = config.run.atmospheric_pressure_atm
        self.atmospheric_pressure_atm = atmospheric_atm
        calibration = config.hardware.pressure_calibration
        self.inlet = PressureChannel.from_config(
            "inlet", config.hardware.daq.inlet_pressure_channel, calibration.inlet, atmospheric_atm
        )
        self.outlet = PressureChannel.from_config(
            "outlet",
            config.hardware.daq.outlet_pressure_channel,
            calibration.outlet,
            atmospheric_atm,
        )
        # The one meter this run selected; the others are never read.
        self.flowmeter_name = config.flowmeter_name
        self.flow = FlowChannel.from_config(config.flowmeter)

        geometry = config.geometry()
        self.geometry = geometry
        self.length_cm = geometry.length_cm
        self.area_cm2 = geometry.area_cm2

        self._window = RollingWindow(config.run.averaging_window_s)

    # -- helpers ----------------------------------------------------------

    def resolve_downstream_pressure_atm(self, measured_outlet_atm: float) -> float:
        """P2 for the Darcy equation: the transducer, or the supplied value.

        ``run.downstream_pressure`` is ``"measured"`` by default, which is what
        a normally-plumbed rig wants. A supplied number is for an outlet that
        vents to atmosphere, where the transducer reads noise around zero.
        """
        fixed = self.config.run.fixed_downstream_pressure_atm
        return measured_outlet_atm if fixed is None else fixed

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
        steady_state: bool = False,
        steady_state_passes: int = 0,
    ) -> Reading:
        """Compute one :class:`Reading` from one set of raw voltages.

        Args:
            index: Zero-based sample number.
            elapsed_s: Seconds since the run started.
            voltages: ``{channel_name: volts}`` for both pressure channels and
                the active flow channel.
            temperature: Latest probe state; may be missing or stale.
            timestamp: Wall-clock time for the log. Defaults to now (UTC).
            steady_state: Whether the detector had confirmed steady state as of
                this sample.
            steady_state_passes: Consecutive passing windows at this sample.

        Returns:
            A fully-populated reading. ``permeability_darcy`` is ``None`` with
            an explanatory ``note`` when the sample cannot be inverted, which
            is normal early in a run and must not abort it.

        Raises:
            KeyError: a configured channel is missing from ``voltages``.
        """
        inlet_volts = self._require(voltages, self.inlet.channel, "inlet pressure")
        outlet_volts = self._require(voltages, self.outlet.channel, "outlet pressure")
        flow_volts = self._require(voltages, self.flow.channel, "flow")

        inlet_atm = self.inlet.volts_to_absolute_atm(inlet_volts)
        outlet_atm = self.outlet.volts_to_absolute_atm(outlet_volts)
        # P2 is the transducer unless the run supplies a value. Both are kept:
        # the measured one is the only evidence that a declared downstream
        # pressure matches what the rig is actually doing.
        downstream_atm = self.resolve_downstream_pressure_atm(outlet_atm)
        mean_atm = mean_pressure(inlet_atm, downstream_atm)

        flow_cm3_s = self.flow.volts_to_cm3_s(flow_volts)
        # The meter is given the resolved pressure too: declaring the outlet
        # line to be at a value and then telling a meter sitting on that line
        # something different would be two beliefs about one pressure.
        # A meter genuinely elsewhere has actual_pressure_source for that.
        reference_pressure_atm = self.flow.reference_pressure_atm(
            inlet_pressure_atm=inlet_atm,
            outlet_pressure_atm=downstream_atm,
            atmospheric_atm=self.atmospheric_pressure_atm,
        )

        temperature_c = temperature.temperature_c
        temperature_ok = temperature_c is not None
        if temperature_c is None:
            temperature_c = self.config.hardware.temperature.fallback_temperature_c

        # Viscosity is evaluated per reading at the mean pore pressure and the
        # current temperature -- both drift during a run, especially before the
        # rig settles.
        gas_state = self.gas_provider.state_at_cgs(temperature_c, max(mean_atm, 1e-9))

        # The Darcy equation's Q_ref * P_ref pairing is exact for an ideal gas.
        # Dividing the reference flow by Z restores the molar-flow invariant
        # when the gas is not ideal (CO2, or several tens of atm).
        reference_flow_cm3_s = flow_cm3_s
        z_factor = gas_state.compressibility_z
        if self.config.run.gas.real_gas_correction and z_factor:
            reference_flow_cm3_s = flow_cm3_s / z_factor

        permeability: float | None = None
        note: str | None = None
        try:
            permeability = compute_gas_permeability(
                flow_rate_cm3_s=reference_flow_cm3_s,
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
            outlet_pressure_atm=outlet_atm,
            downstream_pressure_atm=downstream_atm,
            mean_pressure_atm=mean_atm,
            flow_cm3_s=flow_cm3_s,
            flow_reference_cm3_s=reference_flow_cm3_s,
            flow_reference_pressure_atm=reference_pressure_atm,
            temperature_c=temperature_c,
            temperature_ok=temperature_ok,
            temperature_stale=temperature.stale or not temperature_ok,
            viscosity_cp=gas_state.viscosity_cp,
            compressibility_z=z_factor,
            permeability_darcy=permeability,
            permeability_darcy_avg=self._window.mean(),
            steady_state=steady_state,
            steady_state_passes=steady_state_passes,
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

    Stops on Ctrl+C, on ``run.duration_s``, on ``run.max_samples``, on
    ``run.stop_when_steady`` once steady state is confirmed, or on
    ``steady_state.max_wait_s`` if it never is -- whichever comes first. Always
    closes its sources.
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
        self.config = config
        self.processor = processor
        self.analog_source = analog_source
        self.temperature_source = temperature_source
        self.on_reading = on_reading
        self._clock = clock
        self._sleep = sleep

        self.readings: list[Reading] = []
        self.warnings: list[str] = []
        self.detector = SteadyStateDetector(config.run.steady_state)
        self.status: SteadyStateStatus = self.detector.status
        #: Bounds of the last confirmed steady stretch, preserved even if the
        #: rig destabilises afterwards -- a late wobble should not erase a good
        #: plateau, it should be reported alongside it.
        self.steady_start_s: float | None = None
        self.steady_end_s: float | None = None
        self.ended_unsteady = False

        self._stop_requested = False
        self._stop_reason = ""
        self.started_at: datetime | None = None
        self.ended_at: datetime | None = None

    def request_stop(self, reason: str = "requested") -> None:
        """Ask the loop to finish after the current sample."""
        self._stop_requested = True
        self._stop_reason = reason

    @property
    def stop_reason(self) -> str:
        """Why the loop ended."""
        return self._stop_reason

    @property
    def steady_state_reached(self) -> bool:
        """Whether the run ever confirmed steady state."""
        return self.steady_start_s is not None

    # -- main loop --------------------------------------------------------

    def run(self, *, install_signal_handler: bool = True) -> list[Reading]:
        """Sample until a stop condition is met."""
        interval_s = 1.0 / self.config.hardware.daq.sample_rate_hz
        run_config = self.config.run
        duration_s = run_config.duration_s
        max_samples = run_config.max_samples
        max_wait_s = run_config.steady_state.max_wait_s

        previous_handler = None
        if install_signal_handler:
            previous_handler = self._install_sigint()

        self.started_at = datetime.now(timezone.utc)
        start = self._clock()
        index = 0
        try:
            while not self._stop_requested:
                if max_samples is not None and index >= max_samples:
                    self._stop_reason = f"reached max_samples ({max_samples})"
                    break
                target = start + index * interval_s
                elapsed = self._clock() - start
                if duration_s is not None and elapsed >= duration_s:
                    self._stop_reason = f"reached duration_s ({duration_s} s)"
                    break
                if (
                    max_wait_s is not None
                    and not self.steady_state_reached
                    and elapsed >= max_wait_s
                ):
                    self._record_warning(
                        f"Gave up waiting for steady state after {max_wait_s} s "
                        f"(steady_state.max_wait_s)."
                    )
                    self._stop_reason = "timed out waiting for steady state"
                    break

                reading = self._sample_once(index, elapsed)
                if reading is not None:
                    self.readings.append(reading)
                    self._emit(reading)
                    if (
                        run_config.stop_when_steady
                        and self.detector.is_steady
                        and self.steady_end_s is not None
                        and self.steady_start_s is not None
                        and (self.steady_end_s - self.steady_start_s)
                        >= run_config.steady_state.window_s
                    ):
                        self._stop_reason = "steady state confirmed (stop_when_steady)"
                        break
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

        if self.steady_state_reached and not self.detector.is_steady:
            self.ended_unsteady = True
            self._record_warning(
                "The rig left steady state before the run ended; the reported result "
                f"comes from the plateau at {self.steady_start_s:.1f}-"
                f"{self.steady_end_s:.1f} s."
            )
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
                f"{self.config.hardware.temperature.fallback_temperature_c} degC for "
                "viscosity.",
            )
        elif temperature.stale:
            self._record_warning_once(
                "temperature-stale",
                f"Temperature probe has gone quiet; reusing the last value "
                f"({temperature.temperature_c:.2f} degC).",
            )

        # Compute the reading first, then let the detector see it, then stamp
        # the resulting verdict onto the reading. One pass, no double work.
        provisional = self.processor.process(
            index=index,
            elapsed_s=elapsed,
            voltages=voltages,
            temperature=temperature,
        )
        was_steady = self.detector.is_steady
        self.status = self.detector.update(elapsed, signals_from_reading(provisional))

        if self.detector.is_steady:
            if not was_steady:
                self._record_warning(
                    f"Steady state confirmed at {elapsed:.1f} s "
                    f"({self.status.progress} windows)."
                )
            self.steady_start_s = self.detector.steady_since_elapsed_s
            self.steady_end_s = elapsed

        return provisional.model_copy(
            update={
                "steady_state": self.detector.is_steady,
                "steady_state_passes": self.status.consecutive_passes,
            }
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
            self.request_stop("interrupted")

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

    def steady_window(self) -> SteadyStateWindow | None:
        """The confirmed steady stretch, as indices into :attr:`readings`."""
        if self.steady_start_s is None or self.steady_end_s is None:
            return None
        indices = [
            i
            for i, reading in enumerate(self.readings)
            if self.steady_start_s <= reading.elapsed_s <= self.steady_end_s
            and reading.permeability_darcy is not None
        ]
        if not indices:
            return None
        return SteadyStateWindow(
            start_elapsed_s=self.readings[indices[0]].elapsed_s,
            end_elapsed_s=self.readings[indices[-1]].elapsed_s,
            sample_count=len(indices),
            start_index=indices[0],
            end_index=indices[-1],
        )

    def summarize(self, csv_path: str | None = None) -> RunSummary:
        """Reduce the run to its reported result. See :func:`summarize_run`."""
        return summarize_run(
            self.readings,
            self.config,
            steady_window=self.steady_window(),
            gas_provider=self.processor.gas_provider,
            started_at=self.started_at,
            ended_at=self.ended_at,
            csv_path=csv_path,
            warnings=list(self.warnings),
        )


# --------------------------------------------------------------------------
# Reduction
# --------------------------------------------------------------------------


def summarize_run(
    readings: Sequence[Reading],
    config: GaspermConfig,
    *,
    steady_window: SteadyStateWindow | None,
    gas_provider: GasPropertyProvider | None = None,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
    csv_path: str | None = None,
    warnings: Sequence[str] = (),
) -> RunSummary:
    """Reduce a finished run to its reported result.

    The result is taken from ``steady_window`` when there is one. When there is
    not, the run is summarised over its trailing ``averaging_window_s`` purely
    so the operator can see what happened -- but ``steady_state_reached`` is
    false and the summary must not be treated as a measurement of the sample.

    Args:
        readings: Every reading taken.
        config: The run's configuration.
        steady_window: The detected steady stretch, or ``None``.
        gas_provider: Used for the viscosity/temperature sensitivity in the
            uncertainty budget. ``None`` omits the temperature term.
        started_at: Wall-clock start. Defaults to the first reading.
        ended_at: Wall-clock end. Defaults to the last reading.
        csv_path: Recorded in the summary for traceability.
        warnings: Non-fatal problems seen during the run.

    Returns:
        The summary, with a GUM budget attached when uncertainty is enabled and
        the run reached steady state.

    Raises:
        ValueError: no reading produced a usable permeability at all.
    """
    usable = [r for r in readings if r.permeability_darcy is not None]
    if not usable:
        raise ValueError(
            "No sample produced a usable permeability. Check that inlet pressure "
            "exceeded outlet pressure and that gas was flowing."
        )

    collected_warnings = list(warnings)

    if steady_window is not None:
        window_readings = [
            r
            for r in usable
            if steady_window.start_elapsed_s <= r.elapsed_s <= steady_window.end_elapsed_s
        ]
    else:
        cutoff = usable[-1].elapsed_s - config.run.averaging_window_s
        window_readings = [r for r in usable if r.elapsed_s >= cutoff]
        collected_warnings.append(
            "Steady state was never confirmed. The figures below describe the trailing "
            f"{config.run.averaging_window_s:g} s of an unsettled run and are NOT a "
            "representative permeability for this sample."
        )
    if not window_readings:
        window_readings = usable[-1:]

    permeabilities = [r.permeability_darcy for r in window_readings]
    mean_k, stddev_k = _mean_stddev(permeabilities)
    mean_p, _ = _mean_stddev([r.mean_pressure_atm for r in window_readings])
    mean_t, _ = _mean_stddev([r.temperature_c for r in window_readings])
    mean_q, _ = _mean_stddev([r.flow_cm3_s for r in window_readings])
    mean_inlet, _ = _mean_stddev([r.inlet_pressure_atm for r in window_readings])
    mean_outlet, _ = _mean_stddev([r.outlet_pressure_atm for r in window_readings])
    mean_downstream, _ = _mean_stddev(
        [r.downstream_pressure_atm for r in window_readings]
    )
    mean_reference_p, _ = _mean_stddev(
        [r.flow_reference_pressure_atm for r in window_readings]
    )
    mean_reference_q, _ = _mean_stddev([r.flow_reference_cm3_s for r in window_readings])
    mean_mu, _ = _mean_stddev([r.viscosity_cp for r in window_readings])

    # A declared downstream pressure is an assertion about the rig. The outlet
    # transducer is still being read, so check it: a shut valve would otherwise
    # scale every reported permeability with nothing to show for it.
    supplied = config.run.fixed_downstream_pressure_atm
    if supplied is not None and mean_outlet > 0.0:
        disagreement = abs(mean_outlet - supplied) / supplied
        if disagreement > DOWNSTREAM_MISMATCH_TOLERANCE:
            collected_warnings.append(
                f"The supplied downstream pressure "
                f"({units.from_atm(supplied, config.run.downstream_pressure_unit):.4g} "
                f"{config.run.downstream_pressure_unit}) disagrees with the outlet "
                f"transducer, which read "
                f"{units.from_atm(mean_outlet, config.run.downstream_pressure_unit):.4g} "
                f"{config.run.downstream_pressure_unit} over the same window "
                f"({disagreement:.1%}). Either the declared value is wrong, or the "
                "outlet is not actually open to it."
            )

    budget: UncertaintyBudget | None = None
    if config.run.uncertainty.enabled and mean_k > 0.0:
        count = len(permeabilities)
        type_a_relative = (
            stddev_k / math.sqrt(count) / mean_k if count > 1 and mean_k else None
        )
        exponent = 0.0
        if gas_provider is not None:
            exponent = gas_provider.viscosity_temperature_exponent(
                units.celsius_to_kelvin(mean_t), mean_p * units.ATM_IN_PA
            )
        try:
            budget = build_budget(
                MeasurementPoint(
                    permeability_darcy=mean_k,
                    inlet_pressure_atm=mean_inlet,
                    downstream_pressure_atm=mean_downstream,
                    flow_cm3_s=mean_reference_q,
                    reference_pressure_atm=mean_reference_p,
                    viscosity_cp=mean_mu,
                    temperature_c=mean_t,
                ),
                config.geometry(),
                config.hardware,
                config.run,
                type_a_relative=type_a_relative,
                type_a_dof=float(count - 1) if count > 1 else math.inf,
                viscosity_temperature_exponent=exponent,
            )
        except ValueError as exc:
            collected_warnings.append(f"Could not evaluate the uncertainty budget: {exc}")

    first, last = usable[0], usable[-1]
    return RunSummary(
        sample_id=config.sample.id,
        gas_name=config.run.gas.name,
        started_at=started_at or first.timestamp,
        ended_at=ended_at or last.timestamp,
        duration_s=last.elapsed_s,
        sample_count=len(readings),
        steady_state_reached=steady_window is not None,
        steady_state_window=steady_window,
        mean_pressure_atm=mean_p,
        permeability_darcy=mean_k,
        permeability_stddev_darcy=stddev_k,
        mean_temperature_c=mean_t,
        mean_flow_cm3_s=mean_q,
        averaged_samples=len(window_readings),
        uncertainty=budget,
        metadata=experiment_metadata(config),
        csv_path=csv_path,
        warnings=collected_warnings,
    )


# --------------------------------------------------------------------------
# Console rendering
# --------------------------------------------------------------------------


def format_reading_line(reading: Reading, config: GaspermConfig) -> str:
    """One console line for a reading, in the configured **display** units."""
    run = config.run
    pressure_unit = run.display_pressure_unit
    permeability_unit = run.display_permeability_unit
    flow_unit = run.display_flow_unit

    p1 = units.from_atm(reading.inlet_pressure_atm, pressure_unit)
    # The P2 actually in use, marked when it is a supplied value rather than a
    # reading -- otherwise a constant column looks like a suspiciously steady
    # transducer.
    p2 = units.from_atm(reading.downstream_pressure_atm, pressure_unit)
    p2_flag = "" if reading.downstream_pressure_atm == reading.outlet_pressure_atm else "*"
    flow = units.flow_from_cm3_s(reading.flow_cm3_s, flow_unit)

    if reading.permeability_darcy_avg is not None:
        k_display = units.darcy_to(reading.permeability_darcy_avg, permeability_unit)
        k_text = f"{k_display:>11.4g} {permeability_unit}"
    else:
        k_text = f"{'--':>11} {permeability_unit}"

    temperature_flag = "" if reading.temperature_ok and not reading.temperature_stale else "*"
    marker = "STEADY" if reading.steady_state else f"settling {reading.steady_state_passes}"
    line = (
        f"{reading.elapsed_s:7.1f}s  "
        f"P1 {p1:9.3f}  P2 {p2:9.3f}{p2_flag:1} {pressure_unit}  "
        f"Q {flow:9.3f} {flow_unit}  "
        f"T {reading.temperature_c:6.2f}{temperature_flag:1}C  "
        f"k {k_text}  {marker:>11}"
    )
    if reading.note:
        line += f"   [{reading.note}]"
    return line


def console_header(config: GaspermConfig) -> str:
    """Header line matching :func:`format_reading_line`'s column layout."""
    return (
        f"{'time':>8}  {'P1':>12} {'P2':>12} "
        f"({config.run.display_pressure_unit})  "
        f"{'flow':>11} ({config.run.display_flow_unit})  "
        f"{'temp':>8}  "
        f"{'permeability':>14} ({config.run.display_permeability_unit})  {'state':>11}"
    )
