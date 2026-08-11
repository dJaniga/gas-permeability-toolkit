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
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Sequence

from gasperm import units
from gasperm.config import GaspermConfig, experiment_metadata
from gasperm.gas_properties import GasPropertyProvider, build_provider
from gasperm.hardware.daq import AnalogInputSource, _pressure_channels
from gasperm.hardware.flowmeter import FlowChannel
from gasperm.hardware.pressure import PressureChannel
from gasperm.hardware.temperature import TemperatureSample, TemperatureSource
from gasperm.models import (
    DecayFit,
    PulseDecayResult,
    PulseDecayStatus,
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
from gasperm.pulse_decay import (
    PulseDecayInputError,
    PulseDecayMonitor,
    brace_permeability_darcy,
    dicker_smits_permeability_darcy,
    find_pulse,
    first_storage_root,
    fit_decay_rate,
    fit_window,
    pore_volume_cm3,
    storage_ratios,
)
from gasperm.steady_state import SteadyStateDetector, signals_from_reading
from gasperm.uncertainty import (
    MeasurementPoint,
    PulseDecayPoint,
    build_budget,
    build_pulse_decay_budget,
)

logger = logging.getLogger(__name__)

#: How far a supplied downstream pressure may sit from the outlet transducer
#: before the run summary says so. Loose enough to tolerate a transducer that is
#: uncalibrated near ambient, tight enough to catch a closed valve.
DOWNSTREAM_MISMATCH_TOLERANCE = 0.05

#: How many conversion times a temperature may age before the probe counts as
#: having missed a beat. Two is a hold across one skipped conversion, which is
#: the point at which "slower than the sample rate" becomes "not answering".
MISSED_CONVERSIONS = 3

#: Porosity to fall back on when the sample sheet does not record one, used
#: only to bound the equilibration time from below -- consolidated rock is
#: rarely tighter than this, so the true time can only be longer. It decides
#: whether the missing datum is worth mentioning, never what the answer is.
LOW_POROSITY_BOUND = 0.05

__all__ = [
    "RollingWindow",
    "SampleProcessor",
    "PulseProcessor",
    "AcquisitionLoop",
    "PulseDecayLoop",
    "summarize_run",
    "summarize_pulse_decay_run",
    "format_pulse_reading_line",
    "pulse_console_header",
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


class _ChannelProcessor:
    """What both measurement methods do identically to one raw sample.

    Pressure calibration, plug geometry, the temperature fallback and the gas
    state lookup are the same work whether the run measures flow through the
    plug or watches a differential decay across it. Only what is *derived* from
    them differs, so only that is left to the subclasses.
    """

    def __init__(self, config: GaspermConfig, gas_provider: GasPropertyProvider) -> None:
        self.config = config
        self.gas_provider = gas_provider

        atmospheric_atm = config.run.atmospheric_pressure_atm
        self.atmospheric_pressure_atm = atmospheric_atm
        # Which physical channels this run reads is a DAQ-layer decision -- a
        # pulse-decay run on a rig with a dedicated transducer pair reads
        # different inputs entirely -- so take it from the same helper the task
        # builder uses rather than deriving it twice.
        (_, upstream_channel, upstream_config), (
            _,
            downstream_channel,
            downstream_config,
        ) = _pressure_channels(config)
        self.inlet = PressureChannel.from_config(
            "inlet", upstream_channel, upstream_config, atmospheric_atm
        )
        self.outlet = PressureChannel.from_config(
            "outlet", downstream_channel, downstream_config, atmospheric_atm
        )

        geometry = config.geometry()
        self.geometry = geometry
        self.length_cm = geometry.length_cm
        self.area_cm2 = geometry.area_cm2

    def read_pressures(self, voltages: dict[str, float]) -> tuple[float, float, float, float]:
        """``(inlet_volts, outlet_volts, inlet_atm, outlet_atm)``, absolute."""
        inlet_volts = self._require(voltages, self.inlet.channel, "inlet pressure")
        outlet_volts = self._require(voltages, self.outlet.channel, "outlet pressure")
        return (
            inlet_volts,
            outlet_volts,
            self.inlet.volts_to_absolute_atm(inlet_volts),
            self.outlet.volts_to_absolute_atm(outlet_volts),
        )

    def resolve_temperature(self, temperature: TemperatureSample) -> tuple[float, bool]:
        """``(temperature_c, ok)`` -- the fallback applied when the probe is mute."""
        ok = temperature.temperature_c is not None
        value = (
            temperature.temperature_c
            if ok
            else self.config.hardware.temperature.fallback_temperature_c
        )
        return value, ok

    @staticmethod
    def _require(voltages: dict[str, float], channel: str, role: str) -> float:
        try:
            return voltages[channel]
        except KeyError as exc:
            raise KeyError(
                f"no voltage for the {role} channel {channel!r}; the DAQ task "
                f"returned {sorted(voltages)}"
            ) from exc


class SampleProcessor(_ChannelProcessor):
    """Turns raw voltages plus a temperature into a :class:`Reading`.

    The steady-state path: flow through the plug at a fixed differential, into
    the compressible Darcy equation.
    """

    def __init__(self, config: GaspermConfig, gas_provider: GasPropertyProvider) -> None:
        super().__init__(config, gas_provider)
        # The one meter this run selected; the others are never read.
        self.flowmeter_name = config.flowmeter_name
        self.flow = FlowChannel.from_config(config.flowmeter)
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
        inlet_volts, outlet_volts, inlet_atm, outlet_atm = self.read_pressures(voltages)
        flow_volts = self._require(voltages, self.flow.channel, "flow")

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

        temperature_c, temperature_ok = self.resolve_temperature(temperature)

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
            temperature_age_s=temperature.age_s,
            viscosity_cp=gas_state.viscosity_cp,
            compressibility_z=z_factor,
            permeability_darcy=permeability,
            permeability_darcy_avg=self._window.mean(),
            steady_state=steady_state,
            steady_state_passes=steady_state_passes,
            note=note,
        )

class PulseProcessor(_ChannelProcessor):
    """Turns raw voltages into a :class:`Reading` for a pulse-decay run.

    No flowmeter is read at all -- the flow fields stay ``None``, which is the
    honest record of an instrument that was not part of the measurement. The
    permeability written per sample is a *running* estimate from the live
    monitor's log-linear slope; the reported result comes from the one-shot
    nonlinear fit at the end, exactly as the steady-state rolling mean relates
    to the final steady window.
    """

    def __init__(self, config: GaspermConfig, gas_provider: GasPropertyProvider) -> None:
        super().__init__(config, gas_provider)
        pulse_config = config.run.pulse_decay
        self.monitor = PulseDecayMonitor(
            pulse_config, min_pulse_atm=pulse_config.min_pulse_pressure_atm
        )
        reservoirs = config.hardware.reservoirs
        # The spacer stack is part of V1 and is decided per run, so the total
        # is composed once here rather than read off the vessel.
        self.upstream_spacers = list(pulse_config.upstream_spacers)
        self.upstream_volume_cm3 = reservoirs.upstream_volume_cm3(self.upstream_spacers)
        self.downstream_volume_cm3 = reservoirs.downstream_volume_cm3()
        self.porosity_fraction = config.sample.porosity_fraction
        self.storage_correction = self._resolve_storage_correction()

    def _resolve_storage_correction(self) -> str:
        """Which model this run will use, decided once at startup."""
        requested = self.config.run.pulse_decay.storage_correction
        if requested == "dicker_smits":
            return "dicker_smits"
        if requested == "brace":
            return "brace"
        # auto: apply the correction whenever the plug's porosity is known.
        return "dicker_smits" if self.porosity_fraction else "brace"

    def permeability_from_rate(self, decay_rate_per_s: float, gas_state) -> float | None:
        """Permeability for a decay rate, by whichever model this run uses.

        Returns ``None`` rather than raising when the inputs are not yet
        physical -- early in a run the running rate is noise, and a failed
        sample must not abort a measurement that takes hours.
        """
        compressibility = gas_state.isothermal_compressibility_per_atm
        if not decay_rate_per_s or not compressibility:
            return None
        shared = dict(
            decay_rate_per_s=decay_rate_per_s,
            viscosity_cp=gas_state.viscosity_cp,
            gas_compressibility_per_atm=compressibility,
            length_cm=self.length_cm,
            area_cm2=self.area_cm2,
            upstream_volume_cm3=self.upstream_volume_cm3,
            downstream_volume_cm3=self.downstream_volume_cm3,
        )
        try:
            if self.storage_correction == "dicker_smits" and self.porosity_fraction:
                return dicker_smits_permeability_darcy(
                    porosity_fraction=self.porosity_fraction, **shared
                )
            return brace_permeability_darcy(**shared)
        except PulseDecayInputError:
            return None

    def process(
        self,
        *,
        index: int,
        elapsed_s: float,
        voltages: dict[str, float],
        temperature: TemperatureSample,
        timestamp: datetime | None = None,
    ) -> Reading:
        """Compute one pulse-decay :class:`Reading` and advance the monitor.

        Raises:
            KeyError: a configured channel is missing from ``voltages``.
        """
        inlet_volts, outlet_volts, inlet_atm, outlet_atm = self.read_pressures(voltages)
        # Pulse decay requires downstream_pressure: measured (enforced at config
        # load), so P2 is the transducer and delta_pressure_atm IS the signal.
        mean_atm = mean_pressure(inlet_atm, outlet_atm)
        delta_atm = inlet_atm - outlet_atm

        temperature_c, temperature_ok = self.resolve_temperature(temperature)
        gas_state = self.gas_provider.state_at_cgs(temperature_c, max(mean_atm, 1e-9))

        status = self.monitor.update(elapsed_s, delta_atm)
        permeability = self.permeability_from_rate(status.decay_rate_per_s, gas_state)

        return Reading(
            index=index,
            timestamp=timestamp or datetime.now(timezone.utc),
            elapsed_s=elapsed_s,
            inlet_voltage=inlet_volts,
            outlet_voltage=outlet_volts,
            temperature_raw=temperature.raw_line,
            inlet_pressure_atm=inlet_atm,
            outlet_pressure_atm=outlet_atm,
            downstream_pressure_atm=outlet_atm,
            mean_pressure_atm=mean_atm,
            temperature_c=temperature_c,
            temperature_ok=temperature_ok,
            temperature_stale=temperature.stale,
            temperature_age_s=temperature.age_s,
            viscosity_cp=gas_state.viscosity_cp,
            compressibility_z=gas_state.compressibility_z,
            permeability_darcy=permeability,
            permeability_darcy_avg=permeability,
            decay_fraction=status.decay_fraction,
            note=None if status.phase != "waiting" else "waiting for the pulse",
        )


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


class _LoopBase:
    """Sampling plumbing that has nothing to do with which method is running.

    Timing, the sources and their disposal, the interrupt handler, the warning
    log and the reading callback are identical whether the run is measuring
    steady flow or a decaying differential. The stop conditions are **not** --
    ``stop_after_steady_s`` has no meaning in pulse decay and
    ``stop_below_fraction`` none in steady state -- so each subclass owns its
    own ``run()`` rather than sharing a parameterised one that would have to
    understand both.
    """

    def __init__(
        self,
        config: GaspermConfig,
        analog_source: AnalogInputSource,
        temperature_source: TemperatureSource,
        *,
        on_reading: Callable[[Reading], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.analog_source = analog_source
        self.temperature_source = temperature_source
        self.on_reading = on_reading
        self._clock = clock
        self._sleep = sleep

        self.readings: list[Reading] = []
        self.warnings: list[str] = []
        self._stop_requested = False
        self._stop_reason = ""
        self.started_at: datetime | None = None
        self.ended_at: datetime | None = None

    # -- control ----------------------------------------------------------

    def request_stop(self, reason: str = "requested") -> None:
        """Ask the loop to finish after the current sample."""
        self._stop_requested = True
        self._stop_reason = reason

    @property
    def stop_reason(self) -> str:
        """Why the loop ended."""
        return self._stop_reason

    # -- sources ----------------------------------------------------------

    def _read_sources(self, index: int) -> tuple[dict[str, float], TemperatureSample]:
        """Read the DAQ and the probe, warning about the probe but not the DAQ.

        A serial dropout degrades to a held temperature; a DAQ failure leaves
        nothing to record at all, so it is logged and re-raised.
        """
        try:
            voltages = self.analog_source.read()
        except Exception as exc:  # noqa: BLE001 - DaqError or a driver error
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
        return voltages, temperature

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


class AcquisitionLoop(_LoopBase):
    """Drives steady-state sampling at ``daq.sample_rate_hz`` until stopped.

    Stops on Ctrl+C, on ``run.duration_s``, on ``run.max_samples``, on
    ``run.stop_after_steady_s`` once steady state has held that long, or on
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
        super().__init__(
            config,
            analog_source,
            temperature_source,
            on_reading=on_reading,
            clock=clock,
            sleep=sleep,
        )
        self.processor = processor
        self.detector = SteadyStateDetector(config.run.steady_state)
        self.status: SteadyStateStatus = self.detector.status
        #: Bounds of the last confirmed steady stretch, preserved even if the
        #: rig destabilises afterwards -- a late wobble should not erase a good
        #: plateau, it should be reported alongside it.
        self.steady_start_s: float | None = None
        self.steady_end_s: float | None = None
        #: When the detector last *declared* steady state. Distinct from
        #: :attr:`steady_start_s`, which is where the plateau turns out to have
        #: begun; this is when we became sure of it, and it is what the soak
        #: time is measured from. Cleared if the rig leaves steady state.
        self.steady_confirmed_at_s: float | None = None
        self.ended_unsteady = False

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
                    hold_s = run_config.stop_after_steady_s
                    if (
                        hold_s is not None
                        and self.steady_confirmed_at_s is not None
                        and (elapsed - self.steady_confirmed_at_s) >= hold_s
                    ):
                        self._stop_reason = (
                            f"steady state held for {hold_s:g} s "
                            "(run.stop_after_steady_s)"
                        )
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
        voltages, temperature = self._read_sources(index)

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
                self.steady_confirmed_at_s = elapsed
                self._record_warning(
                    f"Steady state confirmed at {elapsed:.1f} s "
                    f"({self.status.progress} windows)."
                )
            self.steady_start_s = self.detector.steady_since_elapsed_s
            self.steady_end_s = elapsed
        elif was_steady:
            # The hold has to be continuous: an interrupted soak did not last,
            # so the clock restarts from the next confirmation. The plateau
            # bounds are left alone -- a late wobble should not erase them.
            self.steady_confirmed_at_s = None

        return provisional.model_copy(
            update={
                "steady_state": self.detector.is_steady,
                "steady_state_passes": self.status.consecutive_passes,
            }
        )

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


class PulseDecayLoop(_LoopBase):
    """Drives a pulse-decay run: wait for the pulse, then watch it decay.

    Stops on Ctrl+C, on ``run.duration_s``, on ``run.max_samples``, on
    ``pulse_decay.max_decay_s``, or -- normally -- once the differential has
    fallen below ``pulse_decay.stop_below_fraction`` of the pulse. Always
    closes its sources.
    """

    def __init__(
        self,
        config: GaspermConfig,
        processor: PulseProcessor,
        analog_source: AnalogInputSource,
        temperature_source: TemperatureSource,
        *,
        on_reading: Callable[[Reading], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(
            config,
            analog_source,
            temperature_source,
            on_reading=on_reading,
            clock=clock,
            sleep=sleep,
        )
        self.processor = processor
        self.status: PulseDecayStatus = processor.monitor.status

    @property
    def monitor(self) -> PulseDecayMonitor:
        """The live decay monitor, for the console and the plot."""
        return self.processor.monitor

    @property
    def pulse_seen(self) -> bool:
        """Whether a pulse was ever applied."""
        return self.status.pulse_at_elapsed_s is not None

    def run(self, *, install_signal_handler: bool = True) -> list[Reading]:
        """Sample until the decay completes or a stop condition is met."""
        interval_s = 1.0 / self.config.hardware.daq.sample_rate_hz
        run_config = self.config.run
        pulse_config = run_config.pulse_decay
        # A leak test is a *fixed observation*, not a decay to be waited out:
        # on a tight rig the ideal outcome is that nothing happens, so there is
        # no completion signal to stop on. It runs for its configured time and
        # ignores the decay-fraction stop entirely.
        leak_test = run_config.purpose == "leak_test"
        duration_s = run_config.duration_s
        if leak_test:
            duration_s = pulse_config.leak_test_duration_s or duration_s
        max_samples = run_config.max_samples
        max_decay_s = pulse_config.max_decay_s

        previous_handler = self._install_sigint() if install_signal_handler else None
        self.started_at = datetime.now(timezone.utc)
        start = self._clock()
        index = 0

        try:
            while not self._stop_requested:
                elapsed = self._clock() - start

                if max_samples is not None and index >= max_samples:
                    self._stop_reason = f"reached max_samples ({max_samples})"
                    break
                if duration_s is not None and elapsed >= duration_s:
                    self._stop_reason = f"reached duration_s ({duration_s} s)"
                    break
                # The pulse clock, not the run clock: waiting for the operator
                # to open the valve must not count against the decay's budget.
                pulse_at = self.status.pulse_at_elapsed_s
                if (
                    max_decay_s is not None
                    and not leak_test
                    and pulse_at is not None
                    and elapsed - pulse_at >= max_decay_s
                ):
                    self._record_warning(
                        f"The decay did not reach dP/dP0 = "
                        f"{pulse_config.stop_below_fraction:g} within "
                        f"pulse_decay.max_decay_s ({max_decay_s:g} s); it reached "
                        f"{self.status.decay_fraction:.3f}. A longer run, a smaller "
                        "vessel or a higher pore pressure would finish it."
                    )
                    self._stop_reason = "timed out waiting for the decay"
                    break

                reading = self._sample_once(index, elapsed)
                if reading is not None:
                    self.readings.append(reading)
                    self._emit(reading)

                if self.monitor.is_complete and not leak_test:
                    self._stop_reason = (
                        f"decay reached dP/dP0 = {self.status.decay_fraction:.3f}"
                    )
                    break

                index += 1
                target = start + index * interval_s
                remaining = (target + interval_s) - self._clock()
                if remaining > 0:
                    self._sleep(remaining)
        finally:
            self.ended_at = datetime.now(timezone.utc)
            if previous_handler is not None:
                signal.signal(signal.SIGINT, previous_handler)
            self.close()

        if leak_test and not self.pulse_seen:
            self._record_warning(
                "No pulse was detected during the leak test, so nothing was watched "
                "decaying and the test bounds nothing. Apply the same pulse you will "
                "use for the measurement, with the plug blanked or bypassed."
            )
        elif not self.pulse_seen:
            self._record_warning(
                "No pulse was ever detected: the differential never reached "
                f"pulse_decay.min_pulse_pressure "
                f"({pulse_config.min_pulse_pressure:g} "
                f"{pulse_config.pulse_pressure_unit}). Check that the valve was "
                "opened, and that the threshold is not above the pulse you applied."
            )
        elif self.status.reversed_since_peak:
            self._record_warning(
                "The differential rose again after the pulse peaked. That is a leak, "
                "a reopened valve, or a thermal ramp -- not a decay through the plug. "
                "The fitted rate below describes whichever of those it was."
            )
        return self.readings

    def _sample_once(self, index: int, elapsed: float) -> Reading | None:
        voltages, temperature = self._read_sources(index)
        reading = self.processor.process(
            index=index,
            elapsed_s=elapsed,
            voltages=voltages,
            temperature=temperature,
        )
        self.status = self.monitor.status
        return reading

    def fit(self) -> DecayFit | None:
        """Fit the recorded decay over the configured window.

        Returns ``None`` when there was no pulse, or too little of a decay to
        fit -- both of which are reported as warnings rather than exceptions,
        because a run that has already cost hours should still produce its CSV.
        """
        pulse_config = self.config.run.pulse_decay
        if len(self.readings) < 3:
            return None
        times = [r.elapsed_s for r in self.readings]
        deltas = [r.delta_pressure_atm for r in self.readings]
        try:
            peak_index, peak_value = find_pulse(times, deltas)
            if peak_value < pulse_config.min_pulse_pressure_atm:
                return None
            start, end = fit_window(
                times,
                deltas,
                peak_index=peak_index,
                peak_value=peak_value,
                start_fraction=pulse_config.fit_start_fraction,
                end_fraction=pulse_config.fit_end_fraction,
            )
            if end - start < pulse_config.min_fit_samples:
                self._record_warning(
                    f"The fit window holds {end - start} samples, fewer than "
                    f"pulse_decay.min_fit_samples ({pulse_config.min_fit_samples}). "
                    "The decay did not run far enough, or the sample rate is too low."
                )
                return None
            return fit_decay_rate(
                times[start:end],
                deltas[start:end],
                fit_offset=pulse_config.fit_offset,
                bin_s=pulse_config.fit_bin_s,
            )
        except PulseDecayInputError as exc:
            self._record_warning(f"Could not fit the decay: {exc}")
            return None

    def summarize(self, csv_path: str | None = None) -> RunSummary:
        """Reduce the run to its reported result."""
        return summarize_pulse_decay_run(
            self.readings,
            self.config,
            fit=self.fit(),
            processor=self.processor,
            started_at=self.started_at,
            ended_at=self.ended_at,
            csv_path=csv_path,
            warnings=list(self.warnings),
        )


# --------------------------------------------------------------------------
# Reduction
# --------------------------------------------------------------------------


def _temperature_lag_warnings(
    window_readings: Sequence[Reading], config: GaspermConfig
) -> list[str]:
    """Whether the probe kept up over the window the result was taken from.

    Holding a value between conversions is normal for a slow sensor; a value
    held for many conversions is the probe having stopped, which the operator
    can otherwise only infer from the console. Shared by both methods, because
    a stale temperature corrupts viscosity either way.
    """
    conversion_time_s = config.hardware.temperature.conversion_time_s
    ages = [r.temperature_age_s for r in window_readings if r.temperature_age_s is not None]
    if not ages:
        return []
    missed = sum(1 for age in ages if age > MISSED_CONVERSIONS * conversion_time_s)
    if not missed:
        return []
    return [
        f"The temperature probe fell behind on {missed} of {len(ages)} samples in the "
        f"reported window: the value was older than {MISSED_CONVERSIONS} x "
        f"conversion_time_s ({MISSED_CONVERSIONS * conversion_time_s:.2f} s), peaking "
        f"at {max(ages):.2f} s. Viscosity was computed from a held temperature."
    ]


def _dominant_component_warnings(
    budget: UncertaintyBudget, config: GaspermConfig, mean_flow_cm3_s: float
) -> list[str]:
    """Say so when one input is large enough to negate the result.

    A budget is allowed to be dominated by one term -- viscosity often is. What
    is not allowed is a term worth a quarter of the answer on its own, which
    means the instrument reporting it cannot resolve what it is being asked to
    measure.

    Only the worst offender is reported, however many exceed the threshold. At
    a low differential the two pressures blow up together with the flow, and
    three near-identical warnings would crowd out the ones the operator has not
    already been told; the printed budget ranks every component anyway.
    """
    threshold = config.run.uncertainty.max_component_contribution
    if threshold is None or not budget.components:
        return []

    offenders = [
        c
        for c in budget.dominant_components(len(budget.components))
        if c.relative_contribution > threshold
    ]
    if not offenders:
        return []

    component = offenders[0]
    message = (
        f"{component.name} contributes {component.relative_contribution:.0%} of the "
        f"permeability on its own (|c| = {abs(component.relative_sensitivity):.3g}, "
        f"u/x = {component.relative_standard_uncertainty:.1%})."
    )
    if component.symbol == "Q":
        meter = config.flowmeter
        reading = units.flow_from_cm3_s(mean_flow_cm3_s, meter.unit)
        full_scale = abs(meter.value_max - meter.value_min)
        if full_scale > 0:
            message += (
                f" The meter is at {reading / full_scale:.2%} of its "
                f"{meter.value_max:g} {meter.unit} full scale, where its specification "
                "exceeds the signal. This is not a measurement of the sample -- fit a "
                "meter sized for this flow, or raise the pore pressure."
            )
    if len(offenders) > 1:
        others = ", ".join(c.name for c in offenders[1:])
        message += f" {len(offenders) - 1} other input(s) also exceed it: {others}."
    return [message]


def _equilibration_warnings(
    config: GaspermConfig,
    permeability_darcy: float,
    mean_pressure_atm: float,
    viscosity_cp: float,
    window_readings: Sequence[Reading],
) -> list[str]:
    """Compare the run's length against the rock's own equilibration time.

    Pressure diffuses through a plug on a timescale ``t ~ phi mu L^2 / (k P)``.
    For tight rock that is hours, while the detector's criteria can confirm a
    plateau in ninety seconds -- so a signal can be genuinely flat while the
    core is still filling. The permeability is then measuring the transient.

    What matters is how long the plug has had since pressure was applied, which
    is the elapsed time at the end of the averaging window -- not the width of
    that window. A run that held pressure for an hour has equilibrated whether
    its plateau was averaged over thirty seconds or three hundred.
    """
    geometry = config.geometry()
    factor = config.run.steady_state.equilibration_factor
    porosity = geometry.porosity_fraction
    if factor is None or permeability_darcy <= 0.0 or not window_readings:
        return []

    # Strict SI, since the expression mixes k with pressure and viscosity.
    k_m2 = units.darcy_to(permeability_darcy, "m2")
    length_m = geometry.length_cm / 100.0
    equilibration_s = (
        (porosity if porosity is not None else LOW_POROSITY_BOUND)
        * units.cp_to_pa_s(viscosity_cp)
        * length_m**2
        / (k_m2 * mean_pressure_atm * units.ATM_IN_PA)
    )
    elapsed_s = window_readings[-1].elapsed_s
    if elapsed_s >= factor * equilibration_s:
        # With an unrecorded porosity this used the low-end bound, so passing
        # here means the true time is longer -- but the check is a lower bound
        # on t, and clearing it does not settle the question. Stay quiet
        # anyway: nagging every run for optional metadata trains people to
        # ignore the warnings that matter.
        return []
    if porosity is None:
        return [
            f"This run lasted {elapsed_s:.0f} s, and pressure equilibration through this "
            f"plug would take at least {equilibration_s:.0f} s even at a porosity of "
            f"{LOW_POROSITY_BOUND:.0%} -- longer at anything realistic. "
            "sample.porosity_fraction is unrecorded, so this cannot be checked properly; "
            "record it. The signal can be flat while the core is still filling."
        ]
    return [
        f"This run lasted {elapsed_s:.0f} s but pressure equilibration through this plug "
        f"takes about {equilibration_s:.0f} s "
        f"(t ~ phi mu L^2 / (k P_mean), phi = {porosity:g}). The signal can be flat while "
        "the core is still filling, so this may be measuring the transient rather than "
        "the rock. Equilibration scales as 1/P_mean, so a higher pore pressure shortens "
        "it proportionally."
    ]


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

    collected_warnings.extend(_temperature_lag_warnings(window_readings, config))

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

    if budget is not None:
        collected_warnings.extend(_dominant_component_warnings(budget, config, mean_q))

    collected_warnings.extend(
        _equilibration_warnings(
            config, mean_k, mean_p, mean_mu, window_readings
        )
    )

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


def _thermal_drift_warnings(
    window_readings: Sequence[Reading],
    mean_pressure_atm: float,
    fit: DecayFit,
    config: GaspermConfig,
) -> list[str]:
    """Catch room temperature masquerading as a decay.

    A closed vessel obeys ``dP/P = dT/T``, so a 0.1 K room swing moves the
    pressure by 0.34 kPa at 10 atm -- comparable to the late-time signal of a
    small pulse, and drifting on hour timescales, which looks exactly like a
    slow exponential. The fitted offset absorbs a *constant* thermal bias but
    not a ramp, so a drift comparable to the fit's own residual scatter is
    worth saying out loud. This is the pulse-decay counterpart of the
    dominated-budget check.
    """
    inside = [
        r
        for r in window_readings
        if fit.start_elapsed_s <= r.elapsed_s <= fit.end_elapsed_s
    ]
    if len(inside) < 2:
        return []
    temperatures = [r.temperature_c for r in inside]
    swing_k = max(temperatures) - min(temperatures)
    if swing_k <= 0.0:
        return []
    mean_k_abs = units.celsius_to_kelvin(sum(temperatures) / len(temperatures))
    # The pressure a closed vessel would move by, purely from the room.
    thermal_atm = mean_pressure_atm * swing_k / mean_k_abs
    if thermal_atm <= 0.1 * fit.amplitude_atm:
        return []
    unit = config.run.display_pressure_unit
    return [
        f"Room temperature moved {swing_k:.2f} K across the fit window. In a closed "
        f"vessel that alone shifts the pressure by about "
        f"{units.from_atm(thermal_atm, unit):.3g} {unit}, which is "
        f"{thermal_atm / fit.amplitude_atm:.0%} of the "
        f"{units.from_atm(fit.amplitude_atm, unit):.3g} {unit} pulse. The fitted "
        "offset absorbs a constant thermal bias but not a ramp, so part of the decay "
        "reported here may be the room rather than the rock. Insulate the vessels, or "
        "run when the room is stable."
    ]


@dataclass(frozen=True)
class _ResolvedLeak:
    """A leak test found for a measurement run, and how it is being used.

    ``rate_per_s`` is ``None`` when the test was done and **nothing decayed** --
    which is the outcome you want, and is quite different from no test having
    been done at all. Collapsing the two would tell an operator who did the
    right thing that they had not.
    """

    rate_per_s: float | None
    source: str
    mean_pressure_atm: float | None
    subtracted: bool


def find_recorded_leak_test(config: GaspermConfig) -> _ResolvedLeak | None:
    """The most recent leak test recorded for this rig, if any.

    Looked up by rig rather than by plug: the apparatus leaked the same
    whichever core was in it. Returns ``None`` when there is no runs directory
    yet, none has been done, or the stored one has no usable rate -- all of
    which the caller reports rather than treats as an error.
    """
    from gasperm.storage import find_leak_test, find_runs, read_run_metadata

    try:
        records = find_runs(config.resolved_output_dir())
    except (FileNotFoundError, OSError):
        return None
    record = find_leak_test(records)
    if record is None or record.metadata_path is None:
        return None

    summary = read_run_metadata(record.metadata_path).get("summary") or {}
    decay = summary.get("pulse_decay") or {}
    rate = decay.get("decay_rate_per_s")
    usable = isinstance(rate, (int, float)) and rate > 0.0
    return _ResolvedLeak(
        # A test with no fitted decay passed: it bounds the leak below what the
        # pulse and duration could resolve. Reported as None, not as absent.
        rate_per_s=float(rate) if usable else None,
        source=record.name,
        mean_pressure_atm=summary.get("mean_pressure_atm"),
        subtracted=(
            usable and config.run.pulse_decay.leak_correction == "subtract"
        ),
    )


def _leak_warnings(
    leak: _ResolvedLeak | None,
    config: GaspermConfig,
    *,
    measured_rate_per_s: float,
    mean_pressure_atm: float,
    permeability_darcy: float,
    leak_permeability_darcy: float | None,
) -> list[str]:
    """Compare a measurement against the rig's own decay rate.

    The leak test is what separates "this rock passes gas slowly" from "this
    apparatus does". Without one there is no way to tell them apart, which is
    why its absence is itself a warning rather than silence.
    """
    pulse = config.run.pulse_decay
    unit = config.run.display_permeability_unit
    if leak is None:
        return [
            "No leak test has been recorded for this rig, so nothing separates the "
            "sample's decay from the apparatus's. Blank or bypass the plug and run "
            "'collect --method pulse_decay --leak-test' at this pore pressure; "
            "whatever decays then is the floor below which a measurement means "
            "nothing."
        ]

    warnings: list[str] = []

    # Checked first, and regardless of whether anything decayed: leak
    # conductance grows with pressure, so a test that found nothing at a lower
    # charge is not evidence of tightness at this one.
    if leak.mean_pressure_atm:
        difference = abs(leak.mean_pressure_atm - mean_pressure_atm) / mean_pressure_atm
        if difference > pulse.leak_pressure_tolerance:
            display = config.run.display_pressure_unit
            warnings.append(
                f"The leak test ({leak.source}) was done at "
                f"{units.from_atm(leak.mean_pressure_atm, display):.4g} {display} but "
                f"this run is at {units.from_atm(mean_pressure_atm, display):.4g} "
                f"({difference:.0%} apart). Leak conductance depends on pressure, so "
                "that test does not describe this run -- repeat it at this charge."
            )

    if leak.rate_per_s is None:
        warnings.append(
            f"The leak test ({leak.source}) found no measurable decay, so the "
            "apparatus is not contributing anything this run could resolve. Nothing "
            "to correct for."
        )
        return warnings

    total = measured_rate_per_s + (leak.rate_per_s if leak.subtracted else 0.0)
    fraction = abs(leak.rate_per_s / total) if total else math.inf
    if fraction > pulse.max_leak_fraction:
        equivalent = (
            f"{units.darcy_to(leak_permeability_darcy, unit):.4g} {unit}"
            if leak_permeability_darcy
            else "an unknown permeability"
        )
        warnings.append(
            f"The rig's own decay is {fraction:.1%} of the one measured here, above "
            f"pulse_decay.max_leak_fraction ({pulse.max_leak_fraction:.0%}). The leak "
            f"test ({leak.source}) alone would report {equivalent}, against this run's "
            f"{units.darcy_to(permeability_darcy, unit):.4g} {unit}. Find the leak "
            "before trusting this number -- at this ratio you are largely measuring "
            "the apparatus."
        )

    if leak.subtracted:
        warnings.append(
            f"The leak rate {leak.rate_per_s:.4e} 1/s was SUBTRACTED from the measured "
            "one (pulse_decay.leak_correction: subtract). That is only sound if the "
            "leak is linear and has not changed since the test; if it has, the "
            "correction moves the result without any sign that it did."
        )
    return warnings


def _leak_test_verdict(
    config: GaspermConfig, permeability_darcy: float, fit: DecayFit
) -> list[str]:
    """State what a completed leak test bounds, in the units of the decision.

    The useful form is not a decay rate but the permeability the apparatus
    alone would report: that is the number the next measurement has to stand
    clear of, and it is directly comparable with the k you are chasing.
    """
    unit = config.run.display_permeability_unit
    equivalent = units.darcy_to(permeability_darcy, unit)
    ceiling = permeability_darcy / config.run.pulse_decay.max_leak_fraction
    return [
        f"LEAK TEST: the blanked rig decays at {fit.decay_rate_per_s:.4e} 1/s, which "
        f"is what a sample of {equivalent:.4g} {unit} would look like. At "
        f"pulse_decay.max_leak_fraction ({config.run.pulse_decay.max_leak_fraction:.0%}) "
        f"that puts the floor for a trustworthy measurement at about "
        f"{units.darcy_to(ceiling, unit):.4g} {unit} -- below that you would mostly be "
        "measuring this apparatus. Runs from here on are compared against it "
        "automatically."
    ]


def summarize_pulse_decay_run(
    readings: Sequence[Reading],
    config: GaspermConfig,
    *,
    fit: DecayFit | None,
    processor: PulseProcessor | None = None,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
    csv_path: str | None = None,
    warnings: Sequence[str] = (),
) -> RunSummary:
    """Reduce a pulse-decay run to its reported permeability.

    Unlike the steady-state path there is no averaging window: the whole fitted
    decay *is* the measurement, and the permeability comes from its rate. The
    means reported alongside are taken over the fit window, because that is the
    state the decay actually happened at.

    Args:
        readings: Every sample of the run.
        config: The configuration it ran under.
        fit: The fitted decay, or ``None`` when there was nothing to fit.
        processor: Supplies the vessel volumes and the storage model. Rebuilt
            from ``config`` when omitted, so a stored run can be re-reduced.
        started_at / ended_at: Wall-clock bounds, defaulting to the readings'.
        csv_path: Recorded in the summary for traceability.
        warnings: Warnings already collected by the loop.

    Raises:
        ValueError: the run produced no usable sample at all.
    """
    if not readings:
        raise ValueError("No samples were recorded, so there is nothing to summarise.")

    collected_warnings = list(warnings)
    if processor is None:
        processor = PulseProcessor(config, build_provider(config.run.gas))

    first, last = readings[0], readings[-1]
    if fit is None:
        # No fit. For a measurement that means nothing was measured. For a leak
        # test it means the opposite -- the differential did not decay, which is
        # the outcome you want -- so the same condition is reported as a pass.
        leak_test_no_decay = config.run.purpose == "leak_test"
        if leak_test_no_decay:
            collected_warnings.append(
                "No decay could be fitted, which for a leak test is the result you "
                "want: the blanked rig held its differential for the whole run. The "
                "leak is below what this pulse and duration can resolve."
            )
        else:
            collected_warnings.append(
                "No decay could be fitted, so this run did NOT measure the sample's "
                "permeability. The readings are recorded for inspection."
            )
        window = list(readings)
        mean_p, _ = _mean_stddev([r.mean_pressure_atm for r in window])
        mean_t, _ = _mean_stddev([r.temperature_c for r in window])
        return RunSummary(
            sample_id=config.sample.id,
            gas_name=config.run.gas.name,
            method="pulse_decay",
            purpose=config.run.purpose,
            started_at=started_at or first.timestamp,
            ended_at=ended_at or last.timestamp,
            duration_s=last.elapsed_s,
            sample_count=len(readings),
            steady_state_reached=False,
            # A leak test that found nothing has done its job.
            measurement_confirmed=leak_test_no_decay,
            mean_pressure_atm=mean_p,
            permeability_darcy=0.0,
            permeability_stddev_darcy=0.0,
            mean_temperature_c=mean_t,
            averaged_samples=len(window),
            metadata=experiment_metadata(config),
            csv_path=csv_path,
            warnings=collected_warnings,
        )

    window = [
        r for r in readings if fit.start_elapsed_s <= r.elapsed_s <= fit.end_elapsed_s
    ] or list(readings)
    mean_p, _ = _mean_stddev([r.mean_pressure_atm for r in window])
    mean_t, _ = _mean_stddev([r.temperature_c for r in window])
    mean_mu, _ = _mean_stddev([r.viscosity_cp for r in window])

    gas_state = processor.gas_provider.state_at_cgs(mean_t, max(mean_p, 1e-9))
    compressibility = gas_state.isothermal_compressibility_per_atm or (
        1.0 / max(mean_p, 1e-9)
    )

    # A leak test characterises the bench, so it looks for no prior test of its
    # own; a measurement compares itself against the most recent one.
    leak_test_run = config.run.purpose == "leak_test"
    leak = None if leak_test_run else find_recorded_leak_test(config)

    decay_rate = fit.decay_rate_per_s
    if leak is not None and leak.subtracted and leak.rate_per_s is not None:
        # The leak path is in parallel with the plug, so for a linear leak the
        # two rates add and the sample's is the difference.
        decay_rate = max(decay_rate - leak.rate_per_s, 0.0)
        if decay_rate <= 0.0:
            collected_warnings.append(
                f"Subtracting the leak rate ({leak.rate_per_s:.4e} 1/s) left nothing "
                f"of the measured {fit.decay_rate_per_s:.4e} 1/s. This run detected no "
                "decay the apparatus does not already account for."
            )

    permeability = processor.permeability_from_rate(decay_rate, gas_state)
    if permeability is None or permeability <= 0.0:
        collected_warnings.append(
            f"The fitted decay rate ({fit.decay_rate_per_s:.6g} 1/s) did not yield a "
            "positive permeability. Check the vessel volumes and the gas."
        )
        permeability = 0.0

    geometry = config.geometry()
    storage_ratio_up = storage_ratio_down = theta = None
    if processor.storage_correction == "dicker_smits" and processor.porosity_fraction:
        pore = pore_volume_cm3(
            area_cm2=geometry.area_cm2,
            length_cm=geometry.length_cm,
            porosity_fraction=processor.porosity_fraction,
        )
        storage_ratio_up, storage_ratio_down = storage_ratios(
            pore_volume_cm3=pore,
            upstream_volume_cm3=processor.upstream_volume_cm3,
            downstream_volume_cm3=processor.downstream_volume_cm3,
        )
        theta = first_storage_root(storage_ratio_up, storage_ratio_down)

    result = PulseDecayResult(
        decay_rate_per_s=decay_rate,
        decay_rate_standard_uncertainty_per_s=fit.decay_rate_standard_uncertainty_per_s,
        degrees_of_freedom=fit.degrees_of_freedom,
        pulse_amplitude_atm=processor.monitor.pulse_amplitude_atm or fit.amplitude_atm,
        pulse_at_elapsed_s=processor.monitor.status.pulse_at_elapsed_s or 0.0,
        fitted_offset_atm=fit.offset_atm,
        r_squared=fit.r_squared,
        fit_start_elapsed_s=fit.start_elapsed_s,
        fit_end_elapsed_s=fit.end_elapsed_s,
        fit_sample_count=fit.sample_count,
        fit_model=fit.model,
        residual_autocorrelation=fit.residual_autocorrelation,
        upstream_volume_cm3=processor.upstream_volume_cm3,
        downstream_volume_cm3=processor.downstream_volume_cm3,
        upstream_spacers=[str(f) for f in processor.upstream_spacers],
        spacer_volume_cm3=config.hardware.reservoirs.spacer_volume_cm3(
            processor.upstream_spacers
        ),
        upstream_storage_ratio=storage_ratio_up,
        downstream_storage_ratio=storage_ratio_down,
        storage_root=theta,
        storage_correction=processor.storage_correction,
        gas_compressibility_per_atm=compressibility,
        leak_rate_per_s=leak.rate_per_s if leak else None,
        leak_equivalent_permeability_darcy=(
            processor.permeability_from_rate(leak.rate_per_s, gas_state)
            if leak and leak.rate_per_s is not None
            else None
        ),
        leak_test_source=leak.source if leak else None,
        leak_subtracted=bool(leak and leak.subtracted),
    )

    # -- quality gates ----------------------------------------------------
    pulse_config = config.run.pulse_decay
    confirmed = permeability > 0.0 and fit.r_squared >= pulse_config.min_r_squared
    if fit.r_squared < pulse_config.min_r_squared:
        collected_warnings.append(
            f"The decay fit's R^2 is {fit.r_squared:.5f}, below "
            f"pulse_decay.min_r_squared ({pulse_config.min_r_squared:g}). The "
            "differential is not decaying as a single exponential -- suspect a leak, "
            "a thermal ramp, or a pulse too large for the linearisation."
        )
    if fit.model == "log_linear":
        collected_warnings.append(
            "The offset fit did not converge, so a log-linear fit was used instead. "
            "Any zero mismatch between the two transducers biases that rate LOW, and "
            "with it the permeability."
        )
    if (
        fit.residual_autocorrelation is not None
        and fit.residual_autocorrelation > pulse_config.max_residual_autocorrelation
    ):
        collected_warnings.append(
            f"The fit residuals are still correlated after binning "
            f"(lag-1 = {fit.residual_autocorrelation:.2f} > "
            f"{pulse_config.max_residual_autocorrelation:g}), which means they carry "
            "structure rather than noise and u(alpha) below is understated. A longer "
            "pulse_decay.fit_bin_s, or a look at the residual plot, will say which."
        )
    if fit.amplitude_atm > 0.0 and mean_p > 0.0:
        pulse_fraction = fit.amplitude_atm / mean_p
        if pulse_fraction > pulse_config.max_pulse_fraction:
            collected_warnings.append(
                f"The pulse was {pulse_fraction:.1%} of the mean pore pressure, above "
                f"pulse_decay.max_pulse_fraction ({pulse_config.max_pulse_fraction:.0%}). "
                "The small-pulse linearisation assumes the gas compressibility is "
                "constant across the decay, and at this amplitude it is not."
            )
    collected_warnings.extend(_temperature_lag_warnings(window, config))
    collected_warnings.extend(
        _thermal_drift_warnings(window, mean_p, fit, config)
    )
    if leak_test_run:
        collected_warnings.extend(
            _leak_test_verdict(config, permeability, fit)
        )
    else:
        collected_warnings.extend(
            _leak_warnings(
                leak,
                config,
                measured_rate_per_s=decay_rate,
                mean_pressure_atm=mean_p,
                permeability_darcy=permeability,
                leak_permeability_darcy=result.leak_equivalent_permeability_darcy,
            )
        )

    budget: UncertaintyBudget | None = None
    if config.run.uncertainty.enabled and permeability > 0.0:
        try:
            budget = build_pulse_decay_budget(
                PulseDecayPoint(
                    permeability_darcy=permeability,
                    decay_rate_per_s=fit.decay_rate_per_s,
                    mean_pressure_atm=mean_p,
                    viscosity_cp=mean_mu,
                    gas_compressibility_per_atm=compressibility,
                    temperature_c=mean_t,
                    upstream_volume_cm3=processor.upstream_volume_cm3,
                    downstream_volume_cm3=processor.downstream_volume_cm3,
                    upstream_spacers=processor.upstream_spacers,
                    porosity_fraction=(
                        processor.porosity_fraction
                        if processor.storage_correction == "dicker_smits"
                        else None
                    ),
                    storage_root=theta,
                ),
                geometry,
                config.hardware,
                config.run,
                decay_rate_relative_uncertainty=(
                    fit.relative_standard_uncertainty or 0.0
                ),
                decay_rate_dof=fit.degrees_of_freedom,
                viscosity_temperature_exponent=(
                    processor.gas_provider.viscosity_temperature_exponent(
                        units.celsius_to_kelvin(mean_t), mean_p * units.ATM_IN_PA
                    )
                ),
                compressibility_pressure_exponent=(
                    processor.gas_provider.compressibility_pressure_exponent(
                        units.celsius_to_kelvin(mean_t), mean_p * units.ATM_IN_PA
                    )
                ),
            )
        except ValueError as exc:
            collected_warnings.append(f"Could not evaluate the uncertainty budget: {exc}")

    if budget is not None:
        collected_warnings.extend(
            _dominant_component_warnings(budget, config, mean_flow_cm3_s=0.0)
        )

    return RunSummary(
        sample_id=config.sample.id,
        gas_name=config.run.gas.name,
        method="pulse_decay",
        purpose=config.run.purpose,
        started_at=started_at or first.timestamp,
        ended_at=ended_at or last.timestamp,
        duration_s=last.elapsed_s,
        sample_count=len(readings),
        steady_state_reached=False,
        measurement_confirmed=confirmed,
        mean_pressure_atm=mean_p,
        permeability_darcy=permeability,
        permeability_stddev_darcy=0.0,
        mean_temperature_c=mean_t,
        averaged_samples=len(window),
        pulse_decay=result,
        uncertainty=budget,
        metadata=experiment_metadata(config),
        csv_path=csv_path,
        warnings=collected_warnings,
    )


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
    flow = (
        units.flow_from_cm3_s(reading.flow_cm3_s, flow_unit)
        if reading.flow_cm3_s is not None
        else float("nan")
    )

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


def format_pulse_reading_line(
    reading: Reading, status: PulseDecayStatus, config: GaspermConfig
) -> str:
    """One console line for a pulse-decay sample.

    A sibling of :func:`format_reading_line` rather than a branch inside it:
    the columns are genuinely different -- no flow, but a differential, a decay
    fraction and a time constant -- and both functions hard-code their widths
    to match their own header.
    """
    run = config.run
    pressure_unit = run.display_pressure_unit
    permeability_unit = run.display_permeability_unit

    p1 = units.from_atm(reading.inlet_pressure_atm, pressure_unit)
    p2 = units.from_atm(reading.outlet_pressure_atm, pressure_unit)
    delta = units.from_atm(reading.delta_pressure_atm, pressure_unit)
    fraction = (
        f"{reading.decay_fraction:7.3f}" if reading.decay_fraction is not None else f"{'--':>7}"
    )
    tau = f"{status.time_constant_s:8.0f}" if status.time_constant_s else f"{'--':>8}"

    if reading.permeability_darcy is not None:
        k_display = units.darcy_to(reading.permeability_darcy, permeability_unit)
        k_text = f"{k_display:>11.4g} {permeability_unit}"
    else:
        k_text = f"{'--':>11} {permeability_unit}"

    temperature_flag = "" if reading.temperature_ok and not reading.temperature_stale else "*"
    line = (
        f"{reading.elapsed_s:7.1f}s  "
        f"P1 {p1:9.3f}  P2 {p2:9.3f} {pressure_unit}  "
        f"dP {delta:8.3f}  dP/dP0 {fraction}  "
        f"T {reading.temperature_c:6.2f}{temperature_flag:1}C  "
        f"tau {tau} s  k {k_text}  {status.phase:>10}"
    )
    if status.reversed_since_peak:
        line += "   [RISING]"
    return line


def pulse_console_header(config: GaspermConfig) -> str:
    """Header matching :func:`format_pulse_reading_line`'s columns."""
    unit = config.run.display_pressure_unit
    return (
        f"{'time':>8}  {'P1':>12} {'P2':>12} ({unit})  "
        f"{'dP':>11}  {'dP/dP0':>14}  {'temp':>8}  {'tau':>12}  "
        f"{'permeability':>14} ({config.run.display_permeability_unit})  {'phase':>10}"
    )
