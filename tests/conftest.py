"""Shared fixtures and hardware fakes.

Nothing in this suite touches a real device. ``nidaqmx`` and ``serial`` are
replaced in ``sys.modules`` for the tests that exercise
:mod:`gasperm.hardware.daq` / :mod:`gasperm.hardware.temperature`; everything
else drives the acquisition loop through the ``AnalogInputSource`` /
``TemperatureSource`` protocols with plain Python objects.
"""

from __future__ import annotations

import csv
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from gasperm.config import GaspermConfig
from gasperm.gas_properties import FixedPropertyProvider
from gasperm.hardware.temperature import TemperatureSample

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@pytest.fixture
def base_config() -> GaspermConfig:
    """The shipped defaults: ai0/ai1 pressure at 0-5 V, ai2 flow at 0-10 V."""
    return GaspermConfig()


@pytest.fixture
def quick_steady_config(base_config: GaspermConfig) -> GaspermConfig:
    """Defaults retuned so a handful of synthetic samples can reach steady state.

    The shipped criteria (3 x 30 s windows) are right for a real rig and far
    too slow for a test, so the windows are shrunk rather than the criteria
    weakened -- the same tests still exercise the real detector.
    """
    base_config.run.steady_state.window_s = 0.2
    base_config.run.steady_state.required_windows = 2
    base_config.run.steady_state.min_samples = 3
    base_config.hardware.daq.sample_rate_hz = 1000.0
    return base_config


def write_fake_run(
    runs_dir,
    sample_id: str,
    started_at: datetime,
    *,
    mean_pressure_atm: float = 2.0,
    permeability_darcy: float = 0.005,
    steady: bool = True,
    sidecar: bool = True,
    uncertainty_darcy: float | None = 1e-4,
    flowmeter: str = "low_range",
) -> Path:
    """Write a run directory without driving the acquisition loop.

    Enough for discovery and reduction because ``point_from_run`` short-circuits
    on a stored steady summary and never reads the CSV. Omitting the summary
    (``steady=False`` or ``sidecar=False``) makes it replay the two-row CSV
    instead, which cannot satisfy the steady-state criteria -- so those runs are
    genuinely unsteady rather than merely labelled so.
    """
    import yaml

    from gasperm.storage import (
        METADATA_FILENAME,
        READING_COLUMNS,
        READINGS_FILENAME,
        run_directory_name,
    )

    directory = Path(runs_dir) / run_directory_name(sample_id, started_at)
    directory.mkdir(parents=True, exist_ok=True)

    row = {name: "" for name in READING_COLUMNS}
    row.update(
        elapsed_s="0.0", mean_pressure_atm=f"{mean_pressure_atm:g}",
        inlet_pressure_atm=f"{mean_pressure_atm * 1.5:g}",
        outlet_pressure_atm=f"{mean_pressure_atm * 0.5:g}",
        permeability_D=f"{permeability_darcy:g}", temperature_C="22.0",
        flow_cm3_s="1.5", steady_state="1" if steady else "0",
    )
    with (directory / READINGS_FILENAME).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(READING_COLUMNS))
        writer.writeheader()
        for index in (0, 1):
            writer.writerow({**row, "index": index, "elapsed_s": f"{index:.1f}"})

    if sidecar:
        payload = {
            "gasperm_run": {
                "started_at": started_at.isoformat(),
                "readings_csv": READINGS_FILENAME,
                "rows": 2,
            },
            "metadata": {"sample_id": sample_id, "flowmeter": flowmeter},
            "config": {"sample": {"id": sample_id}},
        }
        if steady:
            payload["summary"] = {
                "sample_id": sample_id,
                "steady_state_reached": True,
                "mean_pressure_atm": mean_pressure_atm,
                "permeability_darcy": permeability_darcy,
                "uncertainty": (
                    {"combined_standard_uncertainty_darcy": uncertainty_darcy}
                    if uncertainty_darcy is not None
                    else None
                ),
            }
        (directory / METADATA_FILENAME).write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )
    return directory


@pytest.fixture
def fake_run_writer():
    """Factory for :func:`write_fake_run`."""
    return write_fake_run


@pytest.fixture
def fixed_gas_provider() -> FixedPropertyProvider:
    """A constant 0.0178 cP -- nitrogen at room temperature, to 3 figures.

    Using a fixed provider keeps physics assertions exact; CoolProp itself is
    verified separately in ``test_gas_properties.py``.
    """
    return FixedPropertyProvider("Nitrogen", 0.0178, reason="test fixture")


# --------------------------------------------------------------------------
# Fake NI-DAQmx
# --------------------------------------------------------------------------


class FakeAiChannelCollection:
    """Records every ``add_ai_voltage_chan`` call verbatim."""

    def __init__(self) -> None:
        self.added: list[dict[str, Any]] = []

    def add_ai_voltage_chan(
        self, physical_channel: str, *, min_val: float, max_val: float, terminal_config: Any = None
    ) -> None:
        self.added.append(
            {
                "physical_channel": physical_channel,
                "min_val": min_val,
                "max_val": max_val,
                "terminal_config": terminal_config,
            }
        )


class FakeTask:
    """Stand-in for ``nidaqmx.Task``."""

    #: Every task created during a test, so assertions can inspect them.
    instances: list["FakeTask"] = []
    #: ``{bare channel name: volts}`` the next read() should report.
    voltages: dict[str, float] = {}
    #: Set to an exception instance to make the next read() raise.
    read_error: BaseException | None = None
    #: Set to an exception to make add_ai_voltage_chan raise.
    configure_error: BaseException | None = None

    def __init__(self) -> None:
        self.ai_channels = FakeAiChannelCollection()
        self.closed = False
        FakeTask.instances.append(self)

    @property
    def channel_names(self) -> list[str]:
        """Bare channel names in the order they were added."""
        return [
            entry["physical_channel"].rsplit("/", 1)[-1] for entry in self.ai_channels.added
        ]

    def read(self) -> list[float]:
        if FakeTask.read_error is not None:
            raise FakeTask.read_error
        return [FakeTask.voltages.get(name, 0.0) for name in self.channel_names]

    def close(self) -> None:
        self.closed = True


class _FakeTerminalConfiguration:
    DEFAULT = "DEFAULT"
    RSE = "RSE"
    NRSE = "NRSE"
    DIFF = "DIFF"
    PSEUDO_DIFF = "PSEUDO_DIFF"


@pytest.fixture
def fake_nidaqmx(monkeypatch: pytest.MonkeyPatch):
    """Install a fake ``nidaqmx`` package for the duration of a test."""
    FakeTask.instances = []
    FakeTask.voltages = {}
    FakeTask.read_error = None
    FakeTask.configure_error = None

    module = types.ModuleType("nidaqmx")

    def _task_factory():
        task = FakeTask()
        if FakeTask.configure_error is not None:

            def failing(*args, **kwargs):  # noqa: ANN002, ANN003
                raise FakeTask.configure_error

            task.ai_channels.add_ai_voltage_chan = failing  # type: ignore[method-assign]
        return task

    module.Task = _task_factory  # type: ignore[attr-defined]

    constants = types.ModuleType("nidaqmx.constants")
    constants.TerminalConfiguration = _FakeTerminalConfiguration  # type: ignore[attr-defined]
    module.constants = constants  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "nidaqmx", module)
    monkeypatch.setitem(sys.modules, "nidaqmx.constants", constants)
    return FakeTask


# --------------------------------------------------------------------------
# Fake pyserial
# --------------------------------------------------------------------------


class FakeSerial:
    """Stand-in for ``serial.Serial`` that replays a scripted set of lines."""

    #: Lines the next opened port will emit, as bytes.
    lines: list[bytes] = []
    #: Set to an exception to make construction fail (unplugged probe).
    open_error: BaseException | None = None
    #: Set to an exception raised by readline() after the scripted lines.
    read_error: BaseException | None = None

    def __init__(self, port: str, baudrate: int, timeout: float | None = None) -> None:
        if FakeSerial.open_error is not None:
            raise FakeSerial.open_error
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.closed = False
        self._queue = list(FakeSerial.lines)

    def readline(self) -> bytes:
        if self._queue:
            return self._queue.pop(0)
        if FakeSerial.read_error is not None:
            raise FakeSerial.read_error
        return b""  # timed out, which is normal between probe updates

    def close(self) -> None:
        self.closed = True


class _FakePort:
    def __init__(self, device: str) -> None:
        self.device = device


@pytest.fixture
def fake_serial(monkeypatch: pytest.MonkeyPatch):
    """Install a fake ``serial`` package, including ``serial.tools.list_ports``."""
    FakeSerial.lines = []
    FakeSerial.open_error = None
    FakeSerial.read_error = None

    module = types.ModuleType("serial")
    module.Serial = FakeSerial  # type: ignore[attr-defined]

    class SerialException(Exception):
        pass

    module.SerialException = SerialException  # type: ignore[attr-defined]

    tools = types.ModuleType("serial.tools")
    list_ports = types.ModuleType("serial.tools.list_ports")
    list_ports.available = ["COM4"]  # type: ignore[attr-defined]
    list_ports.comports = lambda: [  # type: ignore[attr-defined]
        _FakePort(name) for name in list_ports.available  # type: ignore[attr-defined]
    ]
    tools.list_ports = list_ports  # type: ignore[attr-defined]
    module.tools = tools  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "serial", module)
    monkeypatch.setitem(sys.modules, "serial.tools", tools)
    monkeypatch.setitem(sys.modules, "serial.tools.list_ports", list_ports)
    return FakeSerial


# --------------------------------------------------------------------------
# Protocol-level fakes for the acquisition loop
# --------------------------------------------------------------------------


class FakeAnalogSource:
    """An ``AnalogInputSource`` that replays voltages. No driver involved."""

    def __init__(self, samples: list[dict[str, float]] | dict[str, float]) -> None:
        """Args:
        samples: One voltage dict repeated forever, or a list replayed in
            order (the last entry repeats once exhausted).
        """
        self._samples = [samples] if isinstance(samples, dict) else list(samples)
        self.read_count = 0
        self.closed = False
        self.fail_after: int | None = None

    def read(self) -> dict[str, float]:
        if self.fail_after is not None and self.read_count >= self.fail_after:
            from gasperm.hardware.daq import DaqError

            raise DaqError("simulated DAQ unplug")
        index = min(self.read_count, len(self._samples) - 1)
        self.read_count += 1
        return dict(self._samples[index])

    def close(self) -> None:
        self.closed = True


class FakeTemperatureSource:
    """A ``TemperatureSource`` returning a scripted sample."""

    def __init__(
        self, temperature_c: float | None = 22.0, *, stale: bool = False, raw: str | None = None
    ) -> None:
        self.temperature_c = temperature_c
        self.stale = stale
        self.raw = raw
        self.closed = False

    def latest(self) -> TemperatureSample:
        return TemperatureSample(self.temperature_c, 0.0, self.raw, self.stale)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_analog_source():
    """Factory for :class:`FakeAnalogSource`."""
    return FakeAnalogSource


@pytest.fixture
def fake_temperature_source():
    """Factory for :class:`FakeTemperatureSource`."""
    return FakeTemperatureSource
