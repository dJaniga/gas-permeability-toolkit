"""Arduino temperature probe over USB serial.

The second and last module allowed to import a device driver (``pyserial``).

Two properties matter here, both because this link is a **separate device from
the DAQ** and fails independently:

1. **Never block the acquisition loop.** A background reader thread drains the
   serial buffer; the loop asks for the latest value and gets an answer
   immediately, even if no new line has arrived.
2. **Never abort a healthy run.** An unplugged probe, a wrong baud rate or a
   line that does not parse degrades the run -- last-known-good value, marked
   stale, with a timestamped warning in the log -- rather than killing a
   multi-minute acquisition that is otherwise fine.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol

from gasperm import units

logger = logging.getLogger(__name__)

__all__ = [
    "TemperatureSource",
    "TemperatureSample",
    "SerialTemperatureReader",
    "build_line_parser",
    "parse_temperature_line",
    "serial_port_exists",
    "list_serial_ports",
]

#: Matches a signed decimal, optionally in exponent form.
_NUMBER_PATTERN = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


@dataclass(frozen=True)
class TemperatureSample:
    """The probe's most recent state, as seen by the acquisition loop."""

    #: degC, or ``None`` when the probe has never produced a usable reading.
    temperature_c: float | None
    #: Monotonic timestamp of the reading, or ``None``.
    received_at: float | None
    #: The raw line, kept verbatim -- including when it failed to parse.
    raw_line: str | None
    #: True when this value has been carried over past ``stale_after_s``.
    stale: bool = False


class TemperatureSource(Protocol):
    """What :mod:`gasperm.acquisition` needs from a temperature probe."""

    def latest(self) -> TemperatureSample:
        """Most recent sample, non-blocking."""
        ...

    def close(self) -> None:
        """Release the port."""
        ...


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def build_line_parser(pattern: str | None) -> re.Pattern[str]:
    """Compile a ``parse_pattern`` such as ``"T:{value}"`` into a regex.

    Everything except the ``{value}`` marker is matched literally, so a pattern
    containing regex metacharacters (``[``, ``.``, ``+``) behaves the way an
    operator writing a serial format would expect.

    ``None`` yields a permissive parser that takes the first number anywhere on
    the line, which handles bare ``23.4`` and CSV lines like ``23.4,55.1``.

    Raises:
        ValueError: the pattern contains no ``{value}`` marker.
    """
    if pattern is None:
        return re.compile(f"({_NUMBER_PATTERN})")
    if "{value}" not in pattern:
        raise ValueError(
            f"temperature.parse_pattern {pattern!r} contains no '{{value}}' marker, so "
            "there is nothing to read the number from. Use e.g. 'T:{value}', or null to "
            "accept the first number on the line."
        )
    literal_parts = [re.escape(part) for part in pattern.split("{value}")]
    return re.compile(f"({_NUMBER_PATTERN})".join(literal_parts))


def parse_temperature_line(
    line: str, parser: re.Pattern[str], *, unit: str = "C"
) -> float | None:
    """Extract a temperature in **degC** from one serial line.

    Returns ``None`` rather than raising when the line does not match -- the
    caller logs the raw line and carries on, because a boot banner or a partial
    first line is normal on an Arduino that has just been opened.
    """
    match = parser.search(line.strip())
    if match is None:
        return None
    try:
        value = float(match.group(1))
    except (ValueError, IndexError):
        return None
    if unit == "C":
        return value
    return units.kelvin_to_celsius(units.temperature_to_kelvin(value, unit))


# --------------------------------------------------------------------------
# Port discovery
# --------------------------------------------------------------------------


def list_serial_ports() -> list[str] | None:
    """Serial port names this machine reports, or ``None`` if unknowable."""
    try:
        from serial.tools import list_ports
    except ImportError:  # pragma: no cover - environment-dependent
        return None
    try:
        return [port.device for port in list_ports.comports()]
    except Exception as exc:  # noqa: BLE001 - enumeration is best-effort
        logger.debug("Could not enumerate serial ports: %s", exc)
        return None


def serial_port_exists(port: str) -> bool | None:
    """Whether ``port`` is present.

    Returns ``None`` when enumeration is unavailable, so callers can tell
    "definitely absent" from "cannot tell" and avoid blocking a run over the
    latter.
    """
    ports = list_serial_ports()
    if ports is None:
        return None
    return port in ports


# --------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------


class SerialTemperatureReader:
    """Background-threaded reader for an Arduino emitting temperature lines.

    The thread owns the port and keeps only the newest parsed value, so the
    acquisition loop always reads current data instead of draining a backlog
    that built up while it was busy.
    """

    def __init__(
        self,
        port: str,
        baud_rate: int = 9600,
        *,
        parse_pattern: str | None = "T:{value}",
        timeout_s: float = 2.0,
        unit: str = "C",
        stale_after_s: float = 10.0,
        max_parse_warnings: int = 5,
    ) -> None:
        """Args:
        port: Serial port, e.g. ``COM4``.
        baud_rate: Must match the Arduino sketch.
        parse_pattern: See :func:`build_line_parser`.
        timeout_s: Per-read serial timeout. Bounds how long the *reader
            thread* blocks; the acquisition loop never blocks at all.
        unit: Unit the probe reports in (``C``/``K``/``F``).
        stale_after_s: How long a value may be reused before being flagged.
        max_parse_warnings: Log at most this many unparseable lines at
            WARNING, then drop to DEBUG so a chatty probe cannot flood a
            long run's log.
        """
        self.port = port
        self.baud_rate = baud_rate
        self.timeout_s = timeout_s
        self.unit = unit
        self.stale_after_s = stale_after_s
        self.max_parse_warnings = max_parse_warnings
        self._parser = build_line_parser(parse_pattern)

        self._serial = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._sample = TemperatureSample(None, None, None)
        self._parse_failures = 0
        self._read_errors = 0
        self.warnings: list[str] = []

    # -- lifecycle --------------------------------------------------------

    def open(self) -> SerialTemperatureReader:
        """Open the port and start the reader thread.

        Raises:
            OSError: the port could not be opened. ``collect`` decides whether
                that is fatal, based on ``temperature.required``.
        """
        try:
            import serial
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise OSError(
                "The 'pyserial' package is not installed. Install it with "
                "'pip install pyserial'."
            ) from exc

        try:
            self._serial = serial.Serial(
                self.port, self.baud_rate, timeout=self.timeout_s
            )
        except Exception as exc:  # noqa: BLE001 - pyserial raises SerialException
            available = list_serial_ports()
            hint = (
                f" Ports this machine reports: {', '.join(available) or '(none)'}."
                if available is not None
                else ""
            )
            raise OSError(
                f"Could not open temperature probe on {self.port} at {self.baud_rate} "
                f"baud: {exc}.{hint}"
            ) from exc

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._read_loop, name=f"gasperm-temp-{self.port}", daemon=True
        )
        self._thread.start()
        logger.info("Temperature probe open on %s at %d baud", self.port, self.baud_rate)
        return self

    def __enter__(self) -> SerialTemperatureReader:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Stop the thread and close the port. Safe to call more than once."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            # Slightly longer than one read timeout so the thread can finish
            # its in-flight readline instead of being abandoned.
            thread.join(timeout=self.timeout_s + 1.0)
        serial_port, self._serial = self._serial, None
        if serial_port is not None:
            try:
                serial_port.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing %s: %s", self.port, exc)

    # -- reader thread ----------------------------------------------------

    def _read_loop(self) -> None:
        """Drain the port, keeping only the newest parsed value."""
        while not self._stop.is_set():
            try:
                raw = self._serial.readline()
            except Exception as exc:  # noqa: BLE001
                # The probe was unplugged, or the port went away. Record it and
                # back off; the acquisition loop keeps running on the last
                # known value.
                self._read_errors += 1
                if self._read_errors <= self.max_parse_warnings:
                    message = f"Temperature serial read failed on {self.port}: {exc}"
                    logger.warning("%s", message)
                    self._record_warning(message)
                self._stop.wait(1.0)
                continue

            if not raw:
                continue  # readline timed out; normal between probe updates

            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            value = parse_temperature_line(line, self._parser, unit=self.unit)
            if value is None:
                self._parse_failures += 1
                message = f"Unparseable temperature line on {self.port}: {line!r}"
                if self._parse_failures <= self.max_parse_warnings:
                    logger.warning("%s", message)
                    self._record_warning(message)
                else:
                    logger.debug("%s", message)
                with self._lock:
                    # Keep the raw line for diagnostics without disturbing the
                    # last good value.
                    self._sample = TemperatureSample(
                        self._sample.temperature_c,
                        self._sample.received_at,
                        line,
                        self._sample.stale,
                    )
                continue

            with self._lock:
                self._sample = TemperatureSample(value, time.monotonic(), line, False)

    def _record_warning(self, message: str) -> None:
        with self._lock:
            self.warnings.append(message)

    # -- consumer API -----------------------------------------------------

    def latest(self) -> TemperatureSample:
        """Most recent sample, non-blocking.

        Marks the sample stale when it has aged past ``stale_after_s``, so the
        acquisition loop can flag the affected rows in the CSV rather than
        silently logging an hour-old temperature.
        """
        with self._lock:
            sample = self._sample
        if sample.temperature_c is None or sample.received_at is None:
            return sample
        age = time.monotonic() - sample.received_at
        if age > self.stale_after_s and not sample.stale:
            return TemperatureSample(
                sample.temperature_c, sample.received_at, sample.raw_line, True
            )
        return sample

    @property
    def parse_failure_count(self) -> int:
        """How many lines failed to parse over the life of the reader."""
        return self._parse_failures


class StaticTemperatureSource:
    """A constant temperature, for running without a probe attached.

    Used when ``temperature.required`` is false and the port cannot be opened,
    so the rest of the rig still produces usable numbers. Every reading is
    flagged stale, which propagates into the CSV.
    """

    def __init__(self, temperature_c: float, *, note: str = "") -> None:
        self.temperature_c = temperature_c
        self.note = note
        self.warnings: list[str] = [
            f"Running with a fixed temperature of {temperature_c} degC"
            + (f" ({note})" if note else "")
        ]

    def latest(self) -> TemperatureSample:
        """The configured constant, always flagged stale."""
        return TemperatureSample(self.temperature_c, time.monotonic(), None, True)

    def close(self) -> None:
        """No-op; there is nothing to release."""
