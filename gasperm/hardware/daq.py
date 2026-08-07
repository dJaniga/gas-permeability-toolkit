"""NI USB-6421 analog input wrapper.

One of only two modules allowed to import a device driver.

The design point that matters: **each channel is added to the task
individually with its own ``min_val``/``max_val``**. The pressure transducers
are 0-5 V and the flowmeter is 0-10 V; a single multi-channel
``add_ai_voltage_chan`` call with a shared range would quietly halve the
pressure channels' effective resolution (or clip the flow channel), and
nothing would raise.

``nidaqmx`` is imported lazily so the package imports, and the whole physics
test suite runs, on a machine with no NI-DAQmx driver installed.
"""

from __future__ import annotations

import logging
from types import TracebackType
from typing import Any, Protocol, Sequence

logger = logging.getLogger(__name__)

__all__ = ["AnalogInputSource", "ChannelSpec", "DaqError", "NiDaqAnalogInput"]


class DaqError(RuntimeError):
    """DAQ could not be opened or read, with an operator-readable message."""


class AnalogInputSource(Protocol):
    """What :mod:`gasperm.acquisition` needs from an analog input device.

    Anything satisfying this works -- the real DAQ, a mock in the tests, or a
    replay source. The acquisition loop never imports ``nidaqmx`` itself.
    """

    def read(self) -> dict[str, float]:
        """One sample from every configured channel, keyed by bare name."""
        ...

    def close(self) -> None:
        """Release the device."""
        ...


class ChannelSpec:
    """One analog input channel and the voltage range it must be read at."""

    def __init__(self, name: str, min_volts: float, max_volts: float, role: str = "") -> None:
        """Args:
        name: Bare channel name, e.g. ``ai0``.
        min_volts: Lower end of this channel's input range.
        max_volts: Upper end of this channel's input range.
        role: Label used in log messages ("inlet pressure", "flow", ...).
        """
        if min_volts >= max_volts:
            raise ValueError(
                f"channel {name}: min_volts ({min_volts}) must be below max_volts "
                f"({max_volts})"
            )
        self.name = name
        self.min_volts = min_volts
        self.max_volts = max_volts
        self.role = role or name

    def __repr__(self) -> str:
        return (
            f"ChannelSpec({self.name!r}, {self.min_volts}, {self.max_volts}, "
            f"role={self.role!r})"
        )


def _terminal_config(name: str) -> Any:
    """Map the config's terminal-config string to the nidaqmx enum."""
    from nidaqmx.constants import TerminalConfiguration

    return {
        "DEFAULT": TerminalConfiguration.DEFAULT,
        "RSE": TerminalConfiguration.RSE,
        "NRSE": TerminalConfiguration.NRSE,
        "DIFF": TerminalConfiguration.DIFF,
        "PSEUDO_DIFF": TerminalConfiguration.PSEUDO_DIFF,
    }[name]


class NiDaqAnalogInput:
    """A live NI-DAQmx analog input task over a fixed set of channels.

    Usable as a context manager; :meth:`close` is idempotent so a failure
    partway through ``open`` still leaves the device released.
    """

    def __init__(
        self,
        device_name: str,
        channels: Sequence[ChannelSpec],
        *,
        terminal_config: str = "DEFAULT",
    ) -> None:
        """Args:
        device_name: NI-DAQmx device name from NI MAX, e.g. ``Dev1``.
        channels: Channels to read, in the order they will be returned.
        terminal_config: DEFAULT / RSE / NRSE / DIFF / PSEUDO_DIFF.
        """
        if not channels:
            raise ValueError("at least one channel is required")
        duplicates = {c.name for c in channels if [x.name for x in channels].count(c.name) > 1}
        if duplicates:
            raise ValueError(
                f"channel(s) {', '.join(sorted(duplicates))} listed more than once"
            )
        self.device_name = device_name
        self.channels = list(channels)
        self.terminal_config = terminal_config
        self._task: Any = None

    # -- lifecycle --------------------------------------------------------

    def open(self) -> NiDaqAnalogInput:
        """Create the DAQ task and add every channel with its own range.

        Raises:
            DaqError: the driver is missing, or the device/channel does not
                exist. The message names the device so a wrong ``device_name``
                is obvious.
        """
        try:
            import nidaqmx
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise DaqError(
                "The 'nidaqmx' package is not installed. Install it with "
                "'pip install nidaqmx'. Note it also needs the NI-DAQmx driver, which "
                "is a separate system-level install from National Instruments."
            ) from exc

        try:
            task = nidaqmx.Task()
        except Exception as exc:  # noqa: BLE001 - driver raises its own hierarchy
            raise DaqError(f"Could not create an NI-DAQmx task: {exc}") from exc

        try:
            terminal = _terminal_config(self.terminal_config)
            # One add_ai_voltage_chan call PER CHANNEL, each with its own
            # min_val/max_val -- see the module docstring.
            for spec in self.channels:
                physical = f"{self.device_name}/{spec.name}"
                logger.debug(
                    "Adding %s (%s) at %.3f..%.3f V",
                    physical,
                    spec.role,
                    spec.min_volts,
                    spec.max_volts,
                )
                task.ai_channels.add_ai_voltage_chan(
                    physical,
                    min_val=spec.min_volts,
                    max_val=spec.max_volts,
                    terminal_config=terminal,
                )
        except Exception as exc:  # noqa: BLE001
            try:
                task.close()
            except Exception:  # noqa: BLE001 - already failing; don't mask it
                pass
            raise DaqError(
                f"Could not configure device {self.device_name!r}: {exc}. Check the "
                "device name in NI MAX and that the channels exist on this model."
            ) from exc

        self._task = task
        return self

    def __enter__(self) -> NiDaqAnalogInput:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the task. Safe to call more than once."""
        task, self._task = self._task, None
        if task is None:
            return
        try:
            task.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error closing the DAQ task: %s", exc)

    # -- reading ----------------------------------------------------------

    def read(self) -> dict[str, float]:
        """Read one sample from every channel.

        Returns:
            ``{channel_name: volts}`` for every configured channel.

        Raises:
            DaqError: the task is not open, or the read failed.
        """
        if self._task is None:
            raise DaqError("DAQ task is not open; call open() first.")
        try:
            values = self._task.read()
        except Exception as exc:  # noqa: BLE001
            raise DaqError(
                f"Read from {self.device_name} failed: {exc}. The device may have been "
                "unplugged mid-run."
            ) from exc

        # A single-channel task returns a bare float rather than a list.
        if not isinstance(values, (list, tuple)):
            values = [values]
        if len(values) != len(self.channels):
            raise DaqError(
                f"Expected {len(self.channels)} values from {self.device_name}, got "
                f"{len(values)}."
            )
        return {spec.name: float(value) for spec, value in zip(self.channels, values)}


def build_channel_specs(config) -> list[ChannelSpec]:
    """The analog inputs this run actually reads, each with its own volt range.

    **Steady state** opens three: the two pressure transducers and the one
    selected flowmeter. The unused flowmeter input (``ai3`` when ``ai2`` is
    configured, or vice versa) is deliberately absent -- meter selection is a
    config-time decision and ``collect`` never touches the other input.

    **Pulse decay** opens two, and no flow channel at all: the method measures
    no flow, which is the whole reason it works below a microdarcy. When the rig
    has a dedicated ``pulse_transducers`` pair those are used, typically a
    lower-range pair on their own inputs; otherwise it falls back to the
    steady-state inlet/outlet channels.

    Args:
        config: A :class:`gasperm.config.GaspermConfig`.
    """
    specs = []
    for role, channel, channel_config in _pressure_channels(config):
        low, high = sorted((channel_config.volts_min, channel_config.volts_max))
        specs.append(ChannelSpec(channel, low, high, role=role))

    if config.run.method != "pulse_decay":
        flow = config.flowmeter
        low, high = sorted((flow.volts_min, flow.volts_max))
        specs.append(ChannelSpec(flow.channel, low, high, role="flow"))
    return specs


def _pressure_channels(config):
    """``(role, channel, calibration)`` for the pressure pair this run reads.

    Shared by the DAQ task builder and the acquisition processors, so the two
    can never disagree about which transducer a voltage came from.
    """
    pulse = config.hardware.pulse_transducers
    if config.run.method == "pulse_decay" and pulse is not None:
        return (
            ("upstream pressure", pulse.upstream.channel, pulse.upstream),
            ("downstream pressure", pulse.downstream.channel, pulse.downstream),
        )
    calibration = config.pressure_calibration
    return (
        ("inlet pressure", config.daq.inlet_pressure_channel, calibration.inlet),
        ("outlet pressure", config.daq.outlet_pressure_channel, calibration.outlet),
    )


def open_analog_input(config) -> NiDaqAnalogInput:
    """Open the DAQ task described by ``config``."""
    return NiDaqAnalogInput(
        config.daq.device_name,
        build_channel_specs(config),
        terminal_config=config.daq.terminal_config,
    ).open()
