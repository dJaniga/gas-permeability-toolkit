"""Shared fixtures and hardware fakes.

Nothing in this suite touches a real device. ``nidaqmx`` and ``serial`` are
replaced in ``sys.modules`` for the tests that exercise
:mod:`gasperm.hardware.daq` / :mod:`gasperm.hardware.temperature`; everything
else drives the acquisition loop through the ``AnalogInputSource`` /
``TemperatureSource`` protocols with plain Python objects.
"""

from __future__ import annotations

import csv
import math
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


@pytest.fixture
def pulse_config(base_config: GaspermConfig) -> GaspermConfig:
    """Defaults retuned so a short synthetic decay can be fitted.

    Follows the ``quick_steady_config`` idiom: shrink the windows, never weaken
    the criteria, so the same real fitter is exercised. The vessels are shrunk
    too, which is what makes a synthetic decay finish in a test rather than in
    the fourteen hours the shipped 400/75 cm3 pair would take at a microdarcy.
    """
    base_config.run.method = "pulse_decay"
    base_config.hardware.daq.sample_rate_hz = 1000.0
    base_config.hardware.reservoirs.upstream.vessel = 8.0
    base_config.hardware.reservoirs.upstream.dead = 0.0
    base_config.hardware.reservoirs.downstream.vessel = 8.0
    base_config.hardware.reservoirs.downstream.dead = 0.0
    base_config.sample.porosity = 0.10
    base_config.run.pulse_decay.fit_bin_s = None
    base_config.run.pulse_decay.min_fit_samples = 10
    return base_config


def decay_voltages(
    config: GaspermConfig,
    *,
    decay_rate_per_s: float,
    mean_pressure_atm: float = 10.0,
    pulse_atm: float = 0.5,
    offset_atm: float = 0.0,
    pulse_at_s: float = 1.0,
    duration_s: float = 30.0,
    step_s: float = 0.1,
) -> list[dict[str, float]]:
    """A scripted decay as raw volts, through the configured calibration.

    Inverts the real ``PressureChannelConfig`` rather than fabricating volts, so
    a test that replays these drives the same calibration path a rig would --
    otherwise it would be asserting that a number survives being multiplied by
    one.
    """
    from gasperm import units
    from gasperm.hardware.daq import _pressure_channels

    (_, up_channel, up_config), (_, down_channel, down_config) = _pressure_channels(
        config
    )
    frames: list[dict[str, float]] = []
    steps = max(int(duration_s / step_s), 1)
    for index in range(steps):
        elapsed = index * step_s
        delta = (
            offset_atm
            if elapsed < pulse_at_s
            else pulse_atm * math.exp(-decay_rate_per_s * (elapsed - pulse_at_s))
            + offset_atm
        )
        frames.append(
            {
                up_channel: up_config.invert(
                    units.from_atm(mean_pressure_atm + delta / 2.0, up_config.unit)
                ),
                down_channel: down_config.invert(
                    units.from_atm(mean_pressure_atm - delta / 2.0, down_config.unit)
                ),
            }
        )
    return frames


@pytest.fixture
def pulse_voltages():
    """Factory for :func:`decay_voltages`."""
    return decay_voltages


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
    downstream_pressure: float | str = "measured",
    mean_flow_cm3_s: float | None = None,
    method: str = "steady_state",
    purpose: str = "measurement",
    pulse_decay: dict | None = None,
) -> Path:
    """Write a run directory without driving the acquisition loop.

    Enough for discovery and reduction because ``point_from_run`` short-circuits
    on a stored steady summary and never reads the CSV. Omitting the summary
    (``steady=False`` or ``sidecar=False``) makes it replay the two-row CSV
    instead, which cannot satisfy the steady-state criteria -- so those runs are
    genuinely unsteady rather than merely labelled so.

    ``mean_flow_cm3_s`` left at ``None`` omits the key entirely, which is what
    a sidecar written before that field existed looks like.
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
        downstream_pressure_atm=f"{mean_pressure_atm * 0.5:g}",
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
            "config": {
                "sample": {"id": sample_id},
                "run": {
                    "method": method,
                    "purpose": purpose,
                    "downstream_pressure": downstream_pressure,
                    "downstream_pressure_unit": "kPa",
                },
            },
        }
        if steady:
            payload["summary"] = {
                "sample_id": sample_id,
                "method": method,
                "purpose": purpose,
                "steady_state_reached": method == "steady_state",
                "measurement_confirmed": True,
                "mean_pressure_atm": mean_pressure_atm,
                "permeability_darcy": permeability_darcy,
                "uncertainty": (
                    {"combined_standard_uncertainty_darcy": uncertainty_darcy}
                    if uncertainty_darcy is not None
                    else None
                ),
            }
            if mean_flow_cm3_s is not None:
                payload["summary"]["mean_flow_cm3_s"] = mean_flow_cm3_s
            if pulse_decay is not None:
                payload["summary"]["pulse_decay"] = pulse_decay
        (directory / METADATA_FILENAME).write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )
    return directory


@pytest.fixture
def fake_run_writer():
    """Factory for :func:`write_fake_run`."""
    return write_fake_run


#: A plausible steady-state budget: two plug-geometry terms, three rig terms and
#: one Type A. ``(symbol, name, u_rel, sensitivity, type, source, dof)``.
BUDGET_TEMPLATE = (
    ("L", "sample length", 0.004, 1.0, "B", "caliper specification", math.inf),
    ("d", "sample diameter", 0.002, -2.0, "B", "caliper specification", math.inf),
    ("Q", "gas flow rate", 0.010, 1.0, "B", "flowmeter specification", math.inf),
    ("mu", "gas viscosity", 0.010, 1.0, "B", "coolprop viscosity for Nitrogen", math.inf),
    ("P1", "inlet pressure", 0.006, 2.0, "B", "inlet transducer", math.inf),
    ("rep", "repeatability", 0.008, 1.0, "A", "scatter of the steady window", 9.0),
)


def build_budget(permeability_darcy: float, *, geometry=None, template=BUDGET_TEMPLATE):
    """A real :class:`UncertaintyBudget` with components, for comparison tests.

    Built through the model rather than as a dict, so a fixture can never drift
    into a shape the product would reject.

    ``geometry`` overrides the recorded *values* of the plug terms, which is how
    a test says "this plug was measured again between campaigns" -- the values
    are what :mod:`gasperm.comparison` reads to decide whether caliper error
    still cancels.
    """
    from gasperm.models import UncertaintyBudget, UncertaintyComponent

    values = {"L": 5.0, "d": 3.81, "Q": 1.5, "mu": 0.0178, "P1": 3.0, "rep": 1.0}
    values.update(geometry or {})

    components = []
    for symbol, name, u_rel, sensitivity, kind, source, dof in template:
        value = values.get(symbol, 1.0)
        components.append(
            UncertaintyComponent(
                name=name, symbol=symbol, evaluation_type=kind, value=value, unit="",
                standard_uncertainty=abs(u_rel * value),
                relative_standard_uncertainty=u_rel,
                relative_sensitivity=sensitivity,
                relative_contribution=abs(sensitivity * u_rel),
                degrees_of_freedom=dof, source=source,
            )
        )
    variance = sum(c.relative_contribution**2 for c in components)
    u_rel_total = math.sqrt(variance)
    return UncertaintyBudget(
        value_darcy=permeability_darcy,
        combined_standard_uncertainty_darcy=u_rel_total * permeability_darcy,
        relative_combined_standard_uncertainty=u_rel_total,
        effective_degrees_of_freedom=math.inf,
        coverage_factor=2.0,
        coverage_probability=0.95,
        expanded_uncertainty_darcy=2.0 * u_rel_total * permeability_darcy,
        components=components,
    )


def write_measured_run(
    runs_dir,
    sample_id: str,
    started_at: datetime,
    *,
    mean_pressure_atm: float,
    permeability_darcy: float,
    gas_name: str = "Nitrogen",
    flowmeter: str = "low_range",
    method: str = "steady_state",
    purpose: str = "measurement",
    downstream_pressure: float | str = "measured",
    porosity_fraction: float | None = None,
    porosity_uncertainty: float | None = None,
    #: Porosity as a sample file states it, with its unit -- the pair a summary
    #: reads back. Leave ``None`` for a run that kept only the fraction.
    porosity: float | None = None,
    porosity_unit: str = "fraction",
    bulk_density_g_cm3: float | None = None,
    length_cm: float = 5.0,
    diameter_cm: float = 3.81,
    budget=None,
    pulse_amplitude_atm: float | None = None,
    #: Pass ``False`` for a pulse run recorded before the setup condition was
    #: kept, i.e. one whose sidecar has dP0 but no pressures at the pulse.
    initial_upstream_pressure_atm: bool = True,
) -> Path:
    """A run directory carrying a **complete** :class:`RunSummary`.

    :func:`write_fake_run` writes only the keys run *discovery* reads, which is
    all that Klinkenberg needs. A comparison needs the individual components of
    each run's uncertainty budget -- deciding what cancels is a per-component
    question -- so this writes the whole model.
    """
    import yaml

    from gasperm.models import ExperimentMetadata, PulseDecayResult, RunSummary
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
        mean_pressure_atm=f"{mean_pressure_atm:g}",
        inlet_pressure_atm=f"{mean_pressure_atm * 1.5:g}",
        outlet_pressure_atm=f"{mean_pressure_atm * 0.5:g}",
        downstream_pressure_atm=f"{mean_pressure_atm * 0.5:g}",
        permeability_D=f"{permeability_darcy:g}", temperature_C="22.0",
        flow_cm3_s="1.5", steady_state="1",
    )
    with (directory / READINGS_FILENAME).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(READING_COLUMNS))
        writer.writeheader()
        for index in (0, 1):
            writer.writerow({**row, "index": index, "elapsed_s": f"{index:.1f}"})

    summary = RunSummary(
        sample_id=sample_id,
        gas_name=gas_name,
        started_at=started_at,
        ended_at=started_at,
        duration_s=120.0,
        sample_count=1200,
        method=method,
        purpose=purpose,
        steady_state_reached=method == "steady_state",
        measurement_confirmed=True,
        mean_pressure_atm=mean_pressure_atm,
        # The same pair the CSV rows above carry, so their midpoint really is
        # mean_pressure_atm and a summary table showing all three is consistent.
        mean_inlet_pressure_atm=mean_pressure_atm * 1.5,
        mean_downstream_pressure_atm=mean_pressure_atm * 0.5,
        permeability_darcy=permeability_darcy,
        permeability_stddev_darcy=permeability_darcy * 0.005,
        mean_temperature_c=22.0,
        mean_flow_cm3_s=1.5,
        averaged_samples=50,
        pulse_decay=(
            None
            if pulse_amplitude_atm is None
            else PulseDecayResult(
                decay_rate_per_s=0.02,
                pulse_amplitude_atm=pulse_amplitude_atm,
                pulse_at_elapsed_s=1.0,
                # The setup condition: both vessels at the pore pressure, the
                # upstream one dP0 above it. dP0 is exactly their difference.
                initial_upstream_pressure_atm=(
                    None
                    if initial_upstream_pressure_atm is False
                    else mean_pressure_atm + pulse_amplitude_atm
                ),
                initial_downstream_pressure_atm=(
                    None if initial_upstream_pressure_atm is False else mean_pressure_atm
                ),
                r_squared=0.999,
                fit_start_elapsed_s=1.0,
                fit_end_elapsed_s=100.0,
                fit_sample_count=990,
                upstream_volume_cm3=8.0,
                downstream_volume_cm3=8.0,
                gas_compressibility_per_atm=1.0 / mean_pressure_atm,
            )
        ),
        uncertainty=budget if budget is not None else build_budget(
            permeability_darcy, geometry={"L": length_cm, "d": diameter_cm}
        ),
        metadata=ExperimentMetadata(
            flowmeter=flowmeter,
            sample_id=sample_id,
            gas_name=gas_name,
            length_cm=length_cm,
            diameter_cm=diameter_cm,
            porosity_fraction=porosity_fraction,
            porosity=porosity,
            porosity_unit=porosity_unit,
            porosity_uncertainty=porosity_uncertainty if porosity is not None else None,
            bulk_density_g_cm3=bulk_density_g_cm3,
        ),
    )

    payload = {
        "gasperm_run": {
            "started_at": started_at.isoformat(),
            "readings_csv": READINGS_FILENAME,
            "rows": 2,
        },
        "metadata": {"sample_id": sample_id, "flowmeter": flowmeter},
        "config": {
            "sample": {
                "id": sample_id,
                "porosity_fraction": porosity_fraction,
                "porosity_uncertainty": porosity_uncertainty,
                "bulk_density_g_cm3": bulk_density_g_cm3,
            },
            "run": {
                "method": method,
                "purpose": purpose,
                "downstream_pressure": downstream_pressure,
                "downstream_pressure_unit": "kPa",
            },
        },
        "summary": summary.model_dump(mode="json"),
    }
    (directory / METADATA_FILENAME).write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    return directory


@pytest.fixture
def measured_run_writer():
    """Factory for :func:`write_measured_run`."""
    return write_measured_run


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
        self, temperature_c: float | None = 22.0, *, stale: bool = False,
        raw: str | None = None, age_s: float | None = 0.0,
    ) -> None:
        self.temperature_c = temperature_c
        self.stale = stale
        self.raw = raw
        self.age_s = age_s
        self.closed = False

    def latest(self) -> TemperatureSample:
        return TemperatureSample(
            self.temperature_c, 0.0, self.raw, self.stale, self.age_s
        )

    def wait_for_first_reading(self, timeout_s: float) -> bool:
        return self.temperature_c is not None

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
