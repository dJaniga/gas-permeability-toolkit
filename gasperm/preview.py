"""``gasperm preview``: watch the rig's raw signals, and nothing else.

This is a **diagnostic view, not a measurement**. Nothing here computes a
permeability, opens a run directory, writes a CSV or consults CoolProp. It
reads the channels you ask for, applies their configured calibration, and
shows the result -- on the console and, with ``--plot``, on a stacked live
window. When the command ends, nothing has been stored.

That restraint is the feature. The question preview answers is "is this
transducer reading what I think it is, and how noisy is it right now" --
asked while nobody is measuring anything, often with the plug not even in the
holder. A command that also derived a permeability would have to invent a
sample, a gas and a geometry to do it, and would then write a run directory
full of numbers that describe nothing.

Three consequences of that, all deliberate:

* **No sample file is loaded.** Preview describes a rig, not an experiment, so
  it reads ``hardware.yaml`` and ``run.yaml`` (for display units and the plot
  defaults) via :func:`gasperm.config.load_bench_config` and never asks for a
  plug. You can preview a bench with no core in it.
* **Only the selected channels are opened.** ``collect`` reads a fixed set
  determined by the method; preview reads exactly what was asked for. That is
  what lets you look at the *inactive* flowmeter, or at a bare ``ai7``, without
  editing a config file.
* **Samples are not accumulated.** :class:`PreviewLoop` holds one sample at a
  time. An operator leaves a preview running for an hour while chasing a
  wiring fault, and a loop that kept every sample would grow without bound for
  no purpose -- there is no summary at the end to need them.
"""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Mapping, Sequence

from gasperm import units
from gasperm.config import GaspermConfig
from gasperm.hardware.daq import ChannelSpec

logger = logging.getLogger(__name__)

__all__ = [
    "PreviewError",
    "PreviewSignal",
    "PreviewSample",
    "PreviewLoop",
    "ConsoleThrottle",
    "available_signals",
    "default_selection",
    "pulse_transducer_pair",
    "resolve_signals",
    "GROUPS",
    "preview_channel_specs",
    "describe_signals",
    "format_preview_line",
    "preview_header",
    "RAW_CHANNEL_RANGE_V",
    "DEFAULT_CONSOLE_INTERVAL_S",
]

#: Voltage range used for a bare ``aiN`` asked for by name -- a channel with no
#: calibration in the config, previewed purely as volts. The widest range a
#: USB-6421 analog input supports, because the point of naming an unconfigured
#: channel is to find out what is on it: a narrower guess would clip the very
#: signal being looked for and show a flat line at the rail.
RAW_CHANNEL_RANGE_V: tuple[float, float] = (-10.0, 10.0)

#: How often the console line is refreshed. The DAQ is sampled at the full
#: configured rate -- the plot and any noise judgement want that -- but ten
#: console updates a second is unreadable, so the text is throttled and the
#: rest of the samples go to the plot only.
DEFAULT_CONSOLE_INTERVAL_S: float = 0.5

#: Panel colours, cycled in selection order.
_COLORS: tuple[str, ...] = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:purple",
    "tab:red",
    "tab:brown",
    "tab:olive",
    "tab:cyan",
)


class PreviewError(ValueError):
    """An unknown signal, or a unit that does not belong to it."""


@dataclass(frozen=True)
class PreviewSignal:
    """One previewable quantity: where it comes from and how it is displayed.

    ``convert`` maps the source's raw value -- volts for a DAQ channel, degC
    for the probe -- to ``unit``. It is the *only* place a preview number is
    derived, and it is always the config's own calibration, never a constant
    written here.
    """

    key: str
    #: Short column heading, e.g. ``P1``.
    label: str
    #: Display unit, or ``"V"`` when this signal is being previewed raw.
    unit: str
    #: Bare DAQ channel, or ``None`` for the temperature probe.
    channel: str | None
    #: The range this channel must be added to the DAQ task at.
    volts_range: tuple[float, float] | None
    #: Raw source value -> display value.
    convert: Callable[[float], float]
    #: One line for the startup banner and ``--list``.
    detail: str
    #: ``True`` when this signal has no calibration and can only be volts.
    raw_only: bool = False

    @property
    def from_probe(self) -> bool:
        """Whether this signal comes from the serial probe rather than the DAQ."""
        return self.channel is None


@dataclass(frozen=True)
class PreviewSample:
    """One pass over the selected signals.

    A plain frozen dataclass rather than a pydantic model, matching
    :class:`~gasperm.hardware.temperature.TemperatureSample`: this is transient
    display data produced at the sample rate and never stored, so per-sample
    validation would cost more than it could catch.
    """

    index: int
    elapsed_s: float
    #: Display-unit value per signal key.
    values: Mapping[str, float]
    #: Raw source value per signal key -- volts, or degC for the probe.
    raw: Mapping[str, float] = field(default_factory=dict)
    #: False when the probe had nothing to give and its column is a gap.
    temperature_ok: bool = True


# --------------------------------------------------------------------------
# The catalogue
# --------------------------------------------------------------------------


def _range_text(low: float, high: float) -> str:
    """A voltage range an operator can read at a glance.

    ``0-5 V`` for the usual unipolar input, but a bipolar one written the same
    way comes out as ``-10-10 V``, where the separator and the sign are the
    same character and the range is unreadable.
    """
    if low < 0.0:
        return f"{low:g} to {high:+g} V"
    return f"{low:g}-{high:g} V"


def _pressure_signal(
    key: str, label: str, channel: str, channel_config, atmospheric_atm: float, unit: str
) -> PreviewSignal:
    """A calibrated transducer, shown as **absolute** pressure.

    Absolute, not gauge, because that is the number every other part of the
    package works in -- a preview that disagreed with what ``collect`` would
    use for the same voltage would be worse than no preview at all. The banner
    says so when the transducer is a gauge type, and ``--volts`` is the better
    tool for checking a zero anyway.
    """
    from gasperm.hardware.pressure import PressureChannel

    transducer = PressureChannel.from_config(key, channel, channel_config, atmospheric_atm)
    low, high = transducer.voltage_range
    sense = "gauge +atm" if transducer.is_gauge else "absolute"
    return PreviewSignal(
        key=key,
        label=label,
        unit=unit,
        channel=channel,
        volts_range=(low, high),
        convert=lambda volts: units.from_atm(transducer.volts_to_absolute_atm(volts), unit),
        detail=(
            f"{channel}  {_range_text(low, high)} -> "
            f"{channel_config.value_min:g}-{channel_config.value_max:g} "
            f"{channel_config.unit} ({sense})"
        ),
    )


def _flow_signal(key: str, name: str, meter, unit: str) -> PreviewSignal:
    from gasperm.hardware.flowmeter import FlowChannel

    channel = FlowChannel.from_config(meter)
    low, high = channel.voltage_range
    return PreviewSignal(
        key=key,
        label=f"Q:{name}" if key != "flow" else "Q",
        unit=unit,
        channel=meter.channel,
        volts_range=(low, high),
        convert=lambda volts: units.flow_from_cm3_s(channel.volts_to_cm3_s(volts), unit),
        detail=(
            f"{meter.channel}  {_range_text(low, high)} -> "
            f"{meter.flow_min:g}-{meter.flow_max:g} {meter.unit}  (meter {name!r})"
        ),
    )


def _temperature_signal(config: GaspermConfig, unit: str) -> PreviewSignal:
    probe = config.hardware.temperature
    return PreviewSignal(
        key="temperature",
        label="T",
        unit=unit,
        channel=None,
        volts_range=None,
        convert=lambda celsius: units.temperature_from_kelvin(
            units.celsius_to_kelvin(celsius), unit
        ),
        detail=(
            f"{probe.port} @ {probe.baud_rate} baud  (serial probe, not the DAQ)"
        ),
    )


def _default_unit(family: str, config: GaspermConfig) -> str:
    return {
        "pressure": config.run.display_pressure_unit,
        "flow": config.run.display_flow_unit,
        "temperature": "C",
    }[family]


def _validate_unit(key: str, family: str, unit: str) -> str:
    """Check a requested unit against the family the signal belongs to."""
    try:
        if family == "pressure":
            return units.normalize_pressure_unit(unit)
        if family == "flow":
            units.flow_to_cm3_s(1.0, unit)
            return unit
        units.temperature_to_kelvin(0.0, unit)
        return unit
    except ValueError as exc:
        raise PreviewError(f"{key}: {exc}") from exc


#: Which unit family each built-in signal belongs to, so a requested unit can
#: be checked against the right set rather than against all of them.
_FAMILIES: dict[str, str] = {
    "inlet_pressure": "pressure",
    "outlet_pressure": "pressure",
    "pulse_upstream": "pressure",
    "pulse_downstream": "pressure",
    "temperature": "temperature",
}


def signal_family(key: str) -> str:
    """The unit family of a signal key. Flow meters are keyed ``flow`` or ``flow.*``."""
    if key == "flow" or key.startswith("flow."):
        return "flow"
    return _FAMILIES.get(key, "voltage")


def pulse_transducer_pair(config: GaspermConfig):
    """The two transducers a pulse-decay run would read, and whether they are its own.

    Returns ``((up_channel, up_config), (down_channel, down_config), dedicated)``.

    A rig may have a **dedicated** low-range pair on their own analog inputs --
    which is what makes pulse decay work, since a 0-68.95 MPa transducer cannot
    resolve a 100 kPa pulse -- or it may not, in which case pulse decay falls
    back to the same inlet/outlet transducers steady state uses. Preview must
    resolve that the same way ``daq._pressure_channels`` does for the
    measurement, or "check the pulse sensors" would mean one thing here and a
    different thing three minutes into a fourteen-hour run.

    Unlike the measurement path this does **not** consult ``run.method``:
    checking the dedicated pair is something you do on a rig whose run.yaml
    still says ``steady_state``, and often the reason you are checking is that
    you are about to switch it.
    """
    pulse = config.hardware.pulse_transducers
    if pulse is not None:
        return (
            (pulse.upstream.channel, pulse.upstream),
            (pulse.downstream.channel, pulse.downstream),
            True,
        )
    calibration = config.hardware.pressure_calibration
    return (
        (config.hardware.daq.inlet_pressure_channel, calibration.inlet),
        (config.hardware.daq.outlet_pressure_channel, calibration.outlet),
        False,
    )


def available_signals(
    config: GaspermConfig, *, unit_overrides: Mapping[str, str] | None = None
) -> dict[str, PreviewSignal]:
    """Every signal this rig defines, in a sensible display order.

    Built entirely from the config: the steady-state pressure pair on their
    configured channels, the pulse-decay pair -- **always** offered, resolved
    exactly as a pulse-decay run would resolve it, so it names the dedicated
    transducers when the rig has them and the steady-state pair when it does
    not -- **every** flowmeter defined rather than only the selected one, and
    the serial probe.
    """
    overrides = dict(unit_overrides or {})

    def unit_for(key: str, family: str) -> str:
        requested = overrides.get(key)
        if requested is None:
            return _default_unit(family, config)
        return _validate_unit(key, family, requested)

    hardware = config.hardware
    atmospheric_atm = config.run.atmospheric_pressure_atm
    calibration = hardware.pressure_calibration

    signals: dict[str, PreviewSignal] = {
        "inlet_pressure": _pressure_signal(
            "inlet_pressure", "P1", hardware.daq.inlet_pressure_channel,
            calibration.inlet, atmospheric_atm, unit_for("inlet_pressure", "pressure"),
        ),
        "outlet_pressure": _pressure_signal(
            "outlet_pressure", "P2", hardware.daq.outlet_pressure_channel,
            calibration.outlet, atmospheric_atm, unit_for("outlet_pressure", "pressure"),
        ),
    }

    (up_channel, up_config), (down_channel, down_config), dedicated = (
        pulse_transducer_pair(config)
    )
    for key, label, channel, channel_config in (
        ("pulse_upstream", "pP1", up_channel, up_config),
        ("pulse_downstream", "pP2", down_channel, down_config),
    ):
        signal_ = _pressure_signal(
            key, label, channel, channel_config, atmospheric_atm,
            unit_for(key, "pressure"),
        )
        # Say which it is. A pulse pair that silently turned out to be the
        # steady-state transducers is the failure this whole method exists to
        # avoid -- they cannot resolve a pulse, and the run would look fine.
        note = (
            "dedicated pulse transducer"
            if dedicated
            else "NO dedicated pulse pair -- falls back to the steady-state transducer"
        )
        signals[key] = replace(signal_, detail=f"{signal_.detail}  [{note}]")

    for name, meter in hardware.flowmeters.items():
        key = f"flow.{name}"
        signals[key] = _flow_signal(key, name, meter, unit_for(key, "flow"))

    signals["temperature"] = _temperature_signal(config, unit_for("temperature", "temperature"))
    return signals


def _selected_flow_key(config: GaspermConfig) -> str | None:
    """``flow.<name>`` for the meter this rig's run.yaml selects, if any."""
    try:
        name, _ = config.hardware.resolve_flowmeter(config.run.flowmeter)
    except ValueError:
        return None
    return f"flow.{name}"


#: Signals that stand for a *pair*, expanded by :func:`resolve_signals`. A
#: pulse-decay rig is checked two channels at a time -- the differential is the
#: measurement -- so naming them one at a time is busywork.
GROUPS: dict[str, tuple[str, ...]] = {
    "pulse": ("pulse_upstream", "pulse_downstream"),
    "pressure": ("inlet_pressure", "outlet_pressure"),
}


def default_selection(config: GaspermConfig) -> list[str]:
    """What ``preview`` shows when no ``--signal`` is given.

    Everything the rig is wired for, with **one** flowmeter -- the one
    ``run.yaml`` selects. Opening both by default would put a second meter's
    channel into the DAQ task on a rig that only has one physically connected,
    which fails at open time rather than showing anything.

    The pulse pair is included only when it is **dedicated**. On a rig without
    one it resolves to the steady-state transducers, and listing it too would
    draw the same two channels on four panels.
    """
    available = available_signals(config)
    dedicated = pulse_transducer_pair(config)[2]
    keys = [
        key
        for key in available
        if not key.startswith("flow.")
        and (dedicated or key not in GROUPS["pulse"])
    ]
    selected_flow = _selected_flow_key(config)
    if selected_flow is not None:
        # Between the pressures and the probe, matching the console order the
        # collect table uses.
        keys.insert(keys.index("temperature"), selected_flow)
    return keys


def _raw_channel_signal(channel: str) -> PreviewSignal:
    """A bare ``aiN`` with no calibration: volts, and only volts."""
    low, high = RAW_CHANNEL_RANGE_V
    return PreviewSignal(
        key=channel,
        label=channel,
        unit="V",
        channel=channel,
        volts_range=(low, high),
        convert=lambda volts: volts,
        detail=f"{channel}  {_range_text(low, high)}  (uncalibrated -- raw volts only)",
        raw_only=True,
    )


def _looks_like_a_channel(text: str) -> bool:
    return len(text) > 2 and text[:2] == "ai" and text[2:].isdigit()


def resolve_signals(
    config: GaspermConfig, requested: Sequence[str] | None
) -> list[PreviewSignal]:
    """Turn ``--signal`` arguments into the signals to read.

    Each argument is ``KEY`` or ``KEY:UNIT``. Three conveniences, all resolved
    here so a typo fails before the DAQ is opened:

    * ``pulse`` and ``pressure`` expand to a **pair** -- the two transducers a
      pulse-decay run would read, and the steady-state pair, respectively.
    * ``flow`` is whichever meter ``run.yaml`` selects.
    * a bare ``aiN`` the config does not describe is an uncalibrated,
      volts-only channel.

    Raises:
        PreviewError: an unknown key, a duplicate, or a unit that does not
            belong to the signal's family. All of it before the DAQ is opened.
    """
    if not requested:
        keys, overrides = default_selection(config), {}
    else:
        keys, overrides = [], {}
        for raw in requested:
            text = raw.strip()
            if not text:
                continue
            key, _, unit = text.partition(":")
            key = key.strip()
            if key in GROUPS:
                expanded = list(GROUPS[key])
            elif key == "flow":
                resolved = _selected_flow_key(config)
                if resolved is None:
                    raise PreviewError(
                        "'flow' means the meter run.yaml selects, and this rig "
                        "defines none. Name one explicitly, e.g. --signal flow.low_range."
                    )
                expanded = [resolved]
            else:
                expanded = [key]
            keys.extend(expanded)
            if unit.strip():
                # A unit on a group applies to every member: the two halves of
                # a differential in different units would be unreadable.
                for member in expanded:
                    overrides[member] = unit.strip()

    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise PreviewError(
            "--signal repeats " + ", ".join(duplicates) + "; each signal gets one panel."
        )

    catalogue = available_signals(config, unit_overrides=overrides)
    resolved: list[PreviewSignal] = []
    for key in keys:
        if key in catalogue:
            resolved.append(catalogue[key])
            continue
        if _looks_like_a_channel(key):
            if key in overrides:
                raise PreviewError(
                    f"{key} has no calibration in hardware.yaml, so it can only be "
                    f"previewed as volts -- {overrides[key]!r} has nothing to convert from."
                )
            resolved.append(_raw_channel_signal(key))
            continue
        raise PreviewError(
            f"--signal {key!r} is not a signal on this rig. Available: "
            + ", ".join(catalogue)
            + "; the pairs " + ", ".join(GROUPS) + " and flow. "
            + "Any bare channel name (ai0, ai7, ...) also works, as raw volts."
        )
    return resolved


def preview_channel_specs(signals: Iterable[PreviewSignal]) -> list[ChannelSpec]:
    """The DAQ task for a preview: exactly the selected channels, nothing else.

    Two signals may share a channel -- ``flow.low_range`` and a bare ``ai2``,
    say -- and NI-DAQmx rejects a task that names one input twice, so the first
    occurrence wins and the second reads the same voltage.
    """
    specs: list[ChannelSpec] = []
    seen: set[str] = set()
    for signal_ in signals:
        if signal_.channel is None or signal_.channel in seen:
            continue
        seen.add(signal_.channel)
        low, high = signal_.volts_range or RAW_CHANNEL_RANGE_V
        specs.append(ChannelSpec(signal_.channel, low, high, role=signal_.key))
    return specs


def describe_signals(signals: Sequence[PreviewSignal], *, volts: bool) -> list[str]:
    """Banner lines: what each selected signal is and where it comes from."""
    width = max((len(s.key) for s in signals), default=0)
    lines = []
    for signal_ in signals:
        unit = "V" if volts or signal_.raw_only else signal_.unit
        lines.append(f"  {signal_.key:<{width}}  {unit:>7}   {signal_.detail}")
    return lines


# --------------------------------------------------------------------------
# Console
# --------------------------------------------------------------------------

#: Minimum width of one signal column: enough for a value with four
#: significant figures, an exponent and a sign.
_MIN_COLUMN = 15


def _column_heading(signal_: PreviewSignal, *, volts: bool) -> str:
    unit = "V" if volts or signal_.raw_only else signal_.unit
    return f"{signal_.label} ({unit})"


def _column_widths(signals: Sequence[PreviewSignal], *, volts: bool) -> list[int]:
    """Per-signal column widths, computed once for both the header and the rows.

    A heading like ``Q:high_range (cm3/min)`` is wider than any value under it,
    and the signals are chosen at the command line, so the widths cannot be
    literals the way ``format_reading_line``'s are. Deriving both from this
    keeps the header sitting over its own column.
    """
    return [
        max(len(_column_heading(s, volts=volts)) + 2, _MIN_COLUMN) for s in signals
    ]


def preview_header(signals: Sequence[PreviewSignal], *, volts: bool = False) -> str:
    """Header matching :func:`format_preview_line`'s columns."""
    widths = _column_widths(signals, volts=volts)
    columns = "".join(
        f"{_column_heading(s, volts=volts):>{width}}"
        for s, width in zip(signals, widths)
    )
    return f"{'time':>9}{columns}"


def format_preview_line(
    sample: PreviewSample, signals: Sequence[PreviewSignal], *, volts: bool = False
) -> str:
    """One console line: the elapsed time and one column per signal."""
    parts = [f"{sample.elapsed_s:8.1f}s"]
    for signal_, width in zip(signals, _column_widths(signals, volts=volts)):
        source = sample.raw if (volts or signal_.raw_only) else sample.values
        value = source.get(signal_.key)
        text = "--" if value is None else f"{value:.4g}"
        parts.append(f"{text:>{width}}")
    line = "".join(parts)
    if not sample.temperature_ok:
        # The probe column would otherwise just look like a steady reading.
        line += "   [no probe reading]"
    return line


class ConsoleThrottle:
    """Rate-limits the console without rate-limiting the acquisition.

    Preview samples at the configured rate so the plot is smooth and the
    scatter on screen is the real scatter; the text would be unreadable at
    that rate, so it updates on its own slower clock.
    """

    def __init__(self, interval_s: float = DEFAULT_CONSOLE_INTERVAL_S) -> None:
        self.interval_s = interval_s
        self._last: float | None = None

    def due(self, now: float) -> bool:
        """Whether enough time has passed to print again. The first call is always due."""
        if self._last is not None and (now - self._last) < self.interval_s:
            return False
        self._last = now
        return True


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


class PreviewLoop:
    """Samples the selected signals until stopped, keeping nothing.

    A sibling of :class:`~gasperm.acquisition.AcquisitionLoop` rather than
    another subclass of ``_LoopBase``: that base exists to accumulate
    ``Reading`` objects and the warnings a run summary is built from, and
    preview is *defined* by producing neither. What it does share -- pacing to
    the next slot rather than sleeping a fixed interval, so a slow sample does
    not make the whole preview drift late -- is a dozen lines, and copying them
    is cheaper than a base class that has to understand a loop with no output.
    """

    def __init__(
        self,
        signals: Sequence[PreviewSignal],
        analog_source,
        temperature_source=None,
        *,
        rate_hz: float,
        duration_s: float | None = None,
        max_samples: int | None = None,
        on_sample: Callable[[PreviewSample], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Args:
        signals: What to read, in display order.
        analog_source: An ``AnalogInputSource``, or ``None`` when nothing
            selected comes from the DAQ.
        temperature_source: A ``TemperatureSource``, or ``None`` when the probe
            was not selected.
        rate_hz: Sampling rate.
        duration_s: Stop after this long. ``None`` runs until interrupted.
        max_samples: Stop after this many samples.
        on_sample: Called with each sample; exceptions in it are logged, never
            propagated -- a display fault must not end the preview.
        """
        if rate_hz <= 0.0:
            raise ValueError(f"rate_hz must be positive, got {rate_hz}")
        self.signals = list(signals)
        self.analog_source = analog_source
        self.temperature_source = temperature_source
        self.rate_hz = rate_hz
        self.duration_s = duration_s
        self.max_samples = max_samples
        self.on_sample = on_sample
        self._clock = clock
        self._sleep = sleep

        self.sample_count = 0
        #: The most recent sample, and the only one held. See the module docstring.
        self.latest: PreviewSample | None = None
        self._stop_requested = False
        self._stop_reason = ""

    # -- control ----------------------------------------------------------

    def request_stop(self, reason: str = "requested") -> None:
        """Ask the loop to finish after the current sample."""
        self._stop_requested = True
        self._stop_reason = reason

    @property
    def stop_reason(self) -> str:
        """Why the loop ended."""
        return self._stop_reason

    # -- sampling ---------------------------------------------------------

    def sample_once(self, index: int, elapsed_s: float) -> PreviewSample:
        """Read every source once and convert. No state is kept but ``latest``."""
        voltages: dict[str, float] = {}
        if self.analog_source is not None:
            voltages = self.analog_source.read()

        celsius: float | None = None
        if self.temperature_source is not None:
            celsius = self.temperature_source.latest().temperature_c

        values: dict[str, float] = {}
        raw: dict[str, float] = {}
        for signal_ in self.signals:
            if signal_.from_probe:
                if celsius is None:
                    continue
                raw[signal_.key] = celsius
                values[signal_.key] = signal_.convert(celsius)
                continue
            try:
                volts = voltages[signal_.channel]
            except KeyError:
                # The task was built from these same signals, so this means the
                # device returned a different set than it was configured with.
                raise KeyError(
                    f"no voltage for {signal_.key} on channel {signal_.channel!r}; "
                    f"the DAQ task returned {sorted(voltages)}"
                ) from None
            raw[signal_.key] = volts
            values[signal_.key] = signal_.convert(volts)

        sample = PreviewSample(
            index=index,
            elapsed_s=elapsed_s,
            values=values,
            raw=raw,
            temperature_ok=self.temperature_source is None or celsius is not None,
        )
        self.latest = sample
        self.sample_count = index + 1
        return sample

    def run(self, *, install_signal_handler: bool = True) -> int:
        """Sample until stopped. Returns how many samples were taken."""
        interval_s = 1.0 / self.rate_hz
        previous_handler = None
        if install_signal_handler:
            previous_handler = self._install_sigint()

        start = self._clock()
        index = 0
        try:
            while not self._stop_requested:
                if self.max_samples is not None and index >= self.max_samples:
                    self._stop_reason = f"reached {self.max_samples} samples"
                    break
                target = start + index * interval_s
                elapsed = self._clock() - start
                if self.duration_s is not None and elapsed >= self.duration_s:
                    self._stop_reason = f"reached {self.duration_s:g} s"
                    break

                sample = self.sample_once(index, elapsed)
                self._emit(sample)
                index += 1

                remaining = (target + interval_s) - self._clock()
                if remaining > 0:
                    self._sleep(remaining)
        finally:
            if previous_handler is not None:
                signal.signal(signal.SIGINT, previous_handler)
            self.close()
        return self.sample_count

    def _emit(self, sample: PreviewSample) -> None:
        if self.on_sample is None:
            return
        try:
            self.on_sample(sample)
        except Exception as exc:  # noqa: BLE001 - display must never kill the loop
            logger.warning("Preview handler failed at sample %d: %s", sample.index, exc)

    def _install_sigint(self):
        def handler(signum, frame):  # noqa: ANN001, ARG001
            self.request_stop("interrupted")

        try:
            return signal.signal(signal.SIGINT, handler)
        except ValueError:
            # Not on the main thread; request_stop() still works.
            return None

    def close(self) -> None:
        """Close both sources, reporting but not re-raising failures."""
        for name, source in (
            ("temperature source", self.temperature_source),
            ("analog source", self.analog_source),
        ):
            if source is None:
                continue
            try:
                source.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing the %s: %s", name, exc)


def preview_color(index: int) -> str:
    """Trace colour for the ``index``-th selected signal."""
    return _COLORS[index % len(_COLORS)]
