"""``gasperm preview``: the signal catalogue, the loop, and what it refuses to do.

Preview is defined as much by what it does *not* do -- no permeability, no run
directory, no sample file, no channel opened that was not asked for -- so a fair
share of these assert absences. The rest check that a previewed number really
did come through the config's own calibration, since a preview that disagreed
with what ``collect`` computes from the same voltage would be worse than none.
"""

from __future__ import annotations

import math
import re

import matplotlib
import pytest

matplotlib.use("Agg")  # noqa: E402 - must precede any pyplot import

from typer.testing import CliRunner  # noqa: E402

from gasperm import units  # noqa: E402
from gasperm.cli import app  # noqa: E402
from gasperm.config import ConfigError, GaspermConfig, load_bench_config  # noqa: E402
from gasperm.preview import (  # noqa: E402
    ConsoleThrottle,
    PreviewError,
    PreviewLoop,
    PreviewSample,
    available_signals,
    default_selection,
    format_preview_line,
    preview_channel_specs,
    preview_header,
    resolve_signals,
)

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def init_rig_with_pulse_pair(directory, upstream="ai4", downstream="ai5"):
    """A rig with a dedicated low-range pulse pair on its own inputs.

    Written into hardware.yaml directly: the section is absent by default, so
    there is nothing for ``init --set`` to descend into.
    """
    import yaml

    from gasperm.config import HARDWARE_FILENAME, PulseTransducersConfig

    rig = init_rig(directory)
    path = rig / HARDWARE_FILENAME
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    pair = PulseTransducersConfig()
    pair.upstream.channel = upstream
    pair.downstream.channel = downstream
    data["pulse_transducers"] = pair.model_dump(mode="json", by_alias=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return rig


def init_rig(directory, *overrides: str):
    args = ["init", str(directory), "--non-interactive", "--force"]
    for override in overrides:
        args += ["--set", override]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return directory


class TestCatalogue:
    """What a rig offers to preview, built from its config and nothing else."""

    def test_the_shipped_rig_offers_its_pressures_flow_and_probe(self):
        keys = list(available_signals(GaspermConfig()))
        assert "inlet_pressure" in keys
        assert "outlet_pressure" in keys
        assert "temperature" in keys
        assert any(key.startswith("flow.") for key in keys)

    def test_every_flowmeter_is_offered_not_only_the_selected_one(self):
        """Checking the meter a run is *not* using is the normal case here."""
        config = GaspermConfig()
        assert len(config.hardware.flowmeters) > 1
        offered = {
            key for key in available_signals(config) if key.startswith("flow.")
        }
        assert offered == {f"flow.{name}" for name in config.hardware.flowmeters}

    def test_a_dedicated_pulse_pair_is_taken_when_the_rig_has_one(self):
        from gasperm.config import PulseTransducersConfig

        config = GaspermConfig()
        config.hardware.pulse_transducers = PulseTransducersConfig()
        catalogue = available_signals(config)
        assert catalogue["pulse_upstream"].channel == "ai4"
        assert catalogue["pulse_downstream"].channel == "ai5"
        assert "dedicated" in catalogue["pulse_upstream"].detail

    def test_without_one_the_pulse_pair_falls_back_to_the_steady_state_pair(self):
        """The same fallback the measurement takes, so the two cannot disagree."""
        config = GaspermConfig()
        assert config.hardware.pulse_transducers is None
        catalogue = available_signals(config)
        assert catalogue["pulse_upstream"].channel == config.daq.inlet_pressure_channel
        assert catalogue["pulse_downstream"].channel == config.daq.outlet_pressure_channel

    def test_the_fallback_is_stated_rather_than_silent(self):
        """A pulse run on 0-68.95 MPa transducers cannot resolve its own pulse."""
        detail = available_signals(GaspermConfig())["pulse_upstream"].detail
        assert "NO dedicated pulse pair" in detail

    def test_preview_resolves_the_pair_the_way_the_daq_task_does(self):
        """Same channels as `collect --method pulse_decay` would open."""
        from gasperm.config import PulseTransducersConfig
        from gasperm.hardware.daq import _pressure_channels

        for equipped in (False, True):
            config = GaspermConfig()
            config.run.method = "pulse_decay"
            if equipped:
                config.hardware.pulse_transducers = PulseTransducersConfig()
            (_, up, _), (_, down, _) = _pressure_channels(config)
            catalogue = available_signals(config)
            assert catalogue["pulse_upstream"].channel == up
            assert catalogue["pulse_downstream"].channel == down

    def test_the_pulse_pair_ignores_the_configured_method(self):
        """You check the dedicated pair on a rig still set to steady state."""
        from gasperm.config import PulseTransducersConfig

        config = GaspermConfig()
        config.hardware.pulse_transducers = PulseTransducersConfig()
        assert config.run.method == "steady_state"
        assert available_signals(config)["pulse_upstream"].channel == "ai4"

    def test_the_default_selection_takes_one_meter(self):
        """Opening both would put an unwired input into the task and fail."""
        config = GaspermConfig()
        chosen = [key for key in default_selection(config) if key.startswith("flow.")]
        assert chosen == [f"flow.{config.flowmeter_name}"]

    def test_signals_carry_their_channel_and_range(self):
        catalogue = available_signals(GaspermConfig())
        assert catalogue["inlet_pressure"].channel == "ai0"
        assert catalogue["inlet_pressure"].volts_range == (0.0, 5.0)
        # The flowmeter is a 0-10 V input; a shared range would clip it.
        meter = catalogue["flow.low_range"]
        assert meter.volts_range == (0.0, 10.0)


def midscale_pressure(config, channel="inlet"):
    """``(volts, pressure in the channel's own unit)`` halfway up its range."""
    calibration = getattr(config.hardware.pressure_calibration, channel)
    volts = (calibration.volts_min + calibration.volts_max) / 2.0
    return volts, (calibration.value_min + calibration.value_max) / 2.0


class TestConversion:
    """A previewed number must be the config's calibration, not a constant."""

    def test_a_pressure_is_shown_in_the_configured_display_unit(self):
        config = GaspermConfig()
        config.run.display_pressure_unit = "kPa"
        volts, expected_in_own_unit = midscale_pressure(config)
        signal = available_signals(config)["inlet_pressure"]
        assert signal.unit == "kPa"
        assert signal.convert(volts) == pytest.approx(
            units.from_atm(
                units.to_atm(
                    expected_in_own_unit,
                    config.hardware.pressure_calibration.inlet.unit,
                ),
                "kPa",
            )
        )

    def test_a_requested_unit_overrides_the_configured_one(self):
        config = GaspermConfig()
        volts, _ = midscale_pressure(config)
        in_kpa = available_signals(config)["inlet_pressure"].convert(volts)
        in_bar = available_signals(
            config, unit_overrides={"inlet_pressure": "bar"}
        )["inlet_pressure"]
        assert in_bar.unit == "bar"
        assert in_bar.convert(volts) == pytest.approx(
            units.from_atm(units.to_atm(in_kpa, "kPa"), "bar")
        )

    def test_a_gauge_transducer_is_previewed_as_absolute(self):
        """The same number collect would use, so the two can never disagree."""
        config = GaspermConfig()
        volts, _ = midscale_pressure(config)
        as_absolute = available_signals(config)["inlet_pressure"].convert(volts)

        config.hardware.pressure_calibration.inlet.reading_type = "gauge"
        signal = available_signals(config)["inlet_pressure"]
        expected = as_absolute + units.from_atm(
            config.run.atmospheric_pressure_atm, "kPa"
        )
        assert signal.convert(volts) == pytest.approx(expected)
        assert "gauge" in signal.detail

    def test_a_flow_signal_goes_through_the_meter_calibration(self):
        config = GaspermConfig()
        meter = config.hardware.flowmeters["low_range"]
        signal = available_signals(config)["flow.low_range"]
        midpoint = (meter.volts_min + meter.volts_max) / 2.0
        expected_in_meter_unit = (meter.flow_min + meter.flow_max) / 2.0
        expected = units.flow_from_cm3_s(
            units.flow_to_cm3_s(expected_in_meter_unit, meter.unit), signal.unit
        )
        assert signal.convert(midpoint) == pytest.approx(expected)

    def test_temperature_can_be_previewed_in_fahrenheit(self):
        config = GaspermConfig()
        signal = available_signals(config, unit_overrides={"temperature": "F"})[
            "temperature"
        ]
        assert signal.convert(100.0) == pytest.approx(212.0)


class TestResolution:
    """Turning --signal arguments into signals, and failing before the DAQ opens."""

    def test_no_argument_selects_the_rig_default(self):
        config = GaspermConfig()
        assert [s.key for s in resolve_signals(config, None)] == default_selection(config)

    def test_flow_is_an_alias_for_the_selected_meter(self):
        config = GaspermConfig()
        config.run.flowmeter = "high_range"
        assert [s.key for s in resolve_signals(config, ["flow"])] == ["flow.high_range"]

    def test_pulse_selects_both_halves_of_the_differential_and_the_difference(self):
        from gasperm.config import PulseTransducersConfig

        config = GaspermConfig()
        config.hardware.pulse_transducers = PulseTransducersConfig()
        signals = resolve_signals(config, ["pulse"])
        assert [s.key for s in signals] == [
            "pulse_upstream", "pulse_downstream", "pulse_dp"
        ]
        assert [s.channel for s in signals] == ["ai4", "ai5", "ai4"]
        assert preview_channel_specs(signals) and [
            spec.name for spec in preview_channel_specs(signals)
        ] == ["ai4", "ai5"]

    def test_pulse_works_on_a_rig_with_no_dedicated_pair(self):
        signals = resolve_signals(GaspermConfig(), ["pulse"])
        assert [s.channel for s in signals] == ["ai0", "ai1", "ai0"]

    def test_a_unit_on_a_group_applies_to_every_member(self):
        """Two halves of one differential in different units is unreadable."""
        signals = resolve_signals(GaspermConfig(), ["pulse:bar"])
        assert [s.unit for s in signals] == ["bar", "bar", "bar"]

    def test_pressure_is_a_pair_too(self):
        signals = resolve_signals(GaspermConfig(), ["pressure"])
        assert [s.key for s in signals] == ["inlet_pressure", "outlet_pressure"]

    def test_the_same_pair_twice_is_refused(self):
        with pytest.raises(PreviewError, match="repeats"):
            resolve_signals(GaspermConfig(), ["pulse", "pulse"])

    def test_a_fallback_pulse_pair_alongside_the_steady_state_one_reads_one_input(self):
        """They are the same transducer under two names, which is the point of
        the banner note -- but the DAQ task must still name each input once."""
        signals = resolve_signals(GaspermConfig(), ["pulse", "inlet_pressure"])
        assert [s.key for s in signals] == [
            "pulse_upstream", "pulse_downstream", "pulse_dp", "inlet_pressure"
        ]
        assert [spec.name for spec in preview_channel_specs(signals)] == ["ai0", "ai1"]

    def test_the_default_selection_skips_a_pulse_pair_that_is_the_same_hardware(self):
        """Otherwise two channels would be drawn on four panels."""
        config = GaspermConfig()
        assert "pulse_upstream" not in default_selection(config)

        from gasperm.config import PulseTransducersConfig

        config.hardware.pulse_transducers = PulseTransducersConfig()
        assert "pulse_upstream" in default_selection(config)

    def test_a_unit_suffix_is_applied(self):
        signals = resolve_signals(GaspermConfig(), ["inlet_pressure:MPa"])
        assert signals[0].unit == "MPa"

    def test_a_unit_from_the_wrong_family_is_refused(self):
        """'sccm' is a real unit, just not one a transducer can be shown in."""
        with pytest.raises(PreviewError, match="pressure unit"):
            resolve_signals(GaspermConfig(), ["inlet_pressure:sccm"])

    def test_an_unknown_signal_names_the_alternatives(self):
        with pytest.raises(PreviewError) as excinfo:
            resolve_signals(GaspermConfig(), ["p1"])
        message = str(excinfo.value)
        assert "inlet_pressure" in message and "temperature" in message

    def test_a_repeated_signal_is_refused(self):
        with pytest.raises(PreviewError, match="repeats"):
            resolve_signals(GaspermConfig(), ["inlet_pressure", "inlet_pressure"])

    def test_a_bare_channel_becomes_raw_volts(self):
        """The wiring check: look at an input the config says nothing about."""
        signals = resolve_signals(GaspermConfig(), ["ai7"])
        assert signals[0].raw_only is True
        assert signals[0].unit == "V"
        assert signals[0].volts_range == (-10.0, 10.0)
        assert signals[0].convert(3.3) == 3.3

    def test_a_unit_on_an_uncalibrated_channel_is_refused(self):
        with pytest.raises(PreviewError, match="no calibration"):
            resolve_signals(GaspermConfig(), ["ai7:bar"])

    def test_flow_with_no_meter_defined_says_so(self):
        """A rig built only for pulse decay legitimately has no meter at all."""
        from gasperm.config import HardwareConfig, RunConfig

        config = GaspermConfig(
            hardware=HardwareConfig(flowmeters={}, default_flowmeter=None),
            run=RunConfig(method="pulse_decay"),
        )
        with pytest.raises(PreviewError, match="defines none"):
            resolve_signals(config, ["flow"])


class TestChannelSpecs:
    """Only what was asked for is opened."""

    def test_one_signal_opens_one_channel(self):
        signals = resolve_signals(GaspermConfig(), ["inlet_pressure"])
        specs = preview_channel_specs(signals)
        assert [spec.name for spec in specs] == ["ai0"]
        assert (specs[0].min_volts, specs[0].max_volts) == (0.0, 5.0)

    def test_each_channel_keeps_its_own_range(self):
        signals = resolve_signals(GaspermConfig(), ["inlet_pressure", "flow"])
        ranges = {s.name: (s.min_volts, s.max_volts) for s in preview_channel_specs(signals)}
        assert ranges == {"ai0": (0.0, 5.0), "ai2": (0.0, 10.0)}

    def test_the_probe_adds_no_channel(self):
        signals = resolve_signals(GaspermConfig(), ["temperature"])
        assert preview_channel_specs(signals) == []

    def test_two_signals_on_one_channel_open_it_once(self):
        """NI-DAQmx rejects a task that names the same input twice."""
        signals = resolve_signals(GaspermConfig(), ["flow.low_range", "ai2"])
        assert [spec.name for spec in preview_channel_specs(signals)] == ["ai2"]


class _Clock:
    """A clock that advances a fixed step per call.

    The loop reads it more than once per sample, so a per-call tick would
    desynchronise elapsed time from the sample index. This one is driven
    explicitly by the test instead.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class TestLoop:
    def make(self, config, keys, source, **kwargs):
        clock = _Clock()
        signals = resolve_signals(config, keys)
        loop = PreviewLoop(
            signals, source, kwargs.pop("temperature_source", None),
            rate_hz=kwargs.pop("rate_hz", 10.0),
            clock=clock, sleep=clock.sleep, **kwargs,
        )
        return loop, signals

    def test_it_stops_after_the_requested_samples(self, fake_analog_source):
        loop, _ = self.make(
            GaspermConfig(), ["inlet_pressure"], fake_analog_source({"ai0": 2.5}),
            max_samples=5,
        )
        assert loop.run(install_signal_handler=False) == 5
        assert "5 samples" in loop.stop_reason

    def test_it_stops_after_the_requested_duration(self, fake_analog_source):
        loop, _ = self.make(
            GaspermConfig(), ["inlet_pressure"], fake_analog_source({"ai0": 2.5}),
            duration_s=1.0, rate_hz=10.0,
        )
        loop.run(install_signal_handler=False)
        assert loop.sample_count == 10
        assert "1 s" in loop.stop_reason

    def test_nothing_is_accumulated(self, fake_analog_source):
        """A preview left running for an hour must not grow without bound."""
        loop, _ = self.make(
            GaspermConfig(), ["inlet_pressure"], fake_analog_source({"ai0": 2.5}),
            max_samples=50,
        )
        loop.run(install_signal_handler=False)
        assert not hasattr(loop, "readings")
        assert loop.latest is not None
        assert loop.latest.index == 49

    def test_both_sources_are_closed(self, fake_analog_source, fake_temperature_source):
        source = fake_analog_source({"ai0": 2.5})
        probe = fake_temperature_source(21.0)
        loop, _ = self.make(
            GaspermConfig(), ["inlet_pressure", "temperature"], source,
            temperature_source=probe, max_samples=2,
        )
        loop.run(install_signal_handler=False)
        assert source.closed and probe.closed

    def test_a_sample_carries_both_the_value_and_the_volts(self, fake_analog_source):
        config = GaspermConfig()
        volts, _ = midscale_pressure(config)
        expected = available_signals(config)["inlet_pressure"].convert(volts)
        loop, _ = self.make(
            config, ["inlet_pressure"], fake_analog_source({"ai0": volts}),
            max_samples=1,
        )
        loop.run(install_signal_handler=False)
        assert loop.latest.raw["inlet_pressure"] == volts
        assert loop.latest.values["inlet_pressure"] == pytest.approx(expected)

    def test_a_mute_probe_leaves_a_gap_rather_than_a_number(
        self, fake_analog_source, fake_temperature_source
    ):
        loop, signals = self.make(
            GaspermConfig(), ["inlet_pressure", "temperature"],
            fake_analog_source({"ai0": 2.5}),
            temperature_source=fake_temperature_source(None), max_samples=1,
        )
        loop.run(install_signal_handler=False)
        assert "temperature" not in loop.latest.values
        assert loop.latest.temperature_ok is False
        # And the console says so rather than showing a blank column.
        assert "no probe reading" in format_preview_line(loop.latest, signals)

    def test_a_failing_display_does_not_end_the_preview(self, fake_analog_source):
        def explode(sample):
            raise RuntimeError("the window went away")

        loop, _ = self.make(
            GaspermConfig(), ["inlet_pressure"], fake_analog_source({"ai0": 2.5}),
            max_samples=3, on_sample=explode,
        )
        assert loop.run(install_signal_handler=False) == 3

    def test_a_daq_failure_stops_the_preview(self, fake_analog_source):
        from gasperm.hardware.daq import DaqError

        source = fake_analog_source({"ai0": 2.5})
        source.fail_after = 2
        loop, _ = self.make(GaspermConfig(), ["inlet_pressure"], source, max_samples=10)
        with pytest.raises(DaqError):
            loop.run(install_signal_handler=False)
        assert source.closed

    def test_a_probe_only_preview_needs_no_daq(self, fake_temperature_source):
        signals = resolve_signals(GaspermConfig(), ["temperature"])
        clock = _Clock()
        loop = PreviewLoop(
            signals, None, fake_temperature_source(23.5), rate_hz=10.0,
            max_samples=2, clock=clock, sleep=clock.sleep,
        )
        loop.run(install_signal_handler=False)
        assert loop.latest.values["temperature"] == pytest.approx(23.5)

    def test_a_non_positive_rate_is_refused(self):
        with pytest.raises(ValueError, match="must be positive"):
            PreviewLoop([], None, None, rate_hz=0.0)


def rig_with_pulse_pair():
    """The shipped rig plus its dedicated 0-100 bar pair on ai4/ai5, 0-10 V."""
    from gasperm.config import PulseTransducersConfig

    config = GaspermConfig()
    config.hardware.pulse_transducers = PulseTransducersConfig()
    return config


class TestPulseDifferential:
    """``pulse_dp``: the quantity pulse decay measures, which neither panel shows.

    A pulse is a fraction of a percent of either absolute trace, so it stays
    invisible until the difference gets its own panel. These check that the
    number on that panel really is the difference of the two above it -- and
    that it never pretends to be a voltage, because no wire carries it.
    """

    def dp_from(self, config, source_factory, *, up, down, channels=("ai4", "ai5")):
        """One sample through the real loop; returns its dP."""
        signals = resolve_signals(config, ["pulse_dp"])
        clock = _Clock()
        loop = PreviewLoop(
            signals,
            source_factory(dict(zip(channels, (up, down)))),
            None, rate_hz=10.0, max_samples=1, clock=clock, sleep=clock.sleep,
        )
        loop.run(install_signal_handler=False)
        return loop.latest

    def test_it_is_offered_next_to_the_pair_it_subtracts(self):
        catalogue = available_signals(rig_with_pulse_pair())
        keys = list(catalogue)
        assert keys.index("pulse_dp") == keys.index("pulse_downstream") + 1
        signal = catalogue["pulse_dp"]
        assert signal.label == "dP"
        assert signal.is_differential
        assert (signal.channel, signal.subtracted.channel) == ("ai4", "ai5")

    def test_it_is_the_difference_of_the_two_panels_above_it(self):
        """0-10 V over 0-100 bar: 6 V against 5 V is 10 bar, hand-checkable."""
        config = rig_with_pulse_pair()
        config.run.display_pressure_unit = "kPa"
        catalogue = available_signals(config)
        difference = (
            catalogue["pulse_upstream"].convert(6.0)
            - catalogue["pulse_downstream"].convert(5.0)
        )
        assert catalogue["pulse_dp"].unit == "kPa"
        assert difference == pytest.approx(units.from_atm(units.to_atm(10.0, "bar"), "kPa"))

    def test_the_loop_computes_it_from_both_channels(self, fake_analog_source):
        config = rig_with_pulse_pair()
        config.run.display_pressure_unit = "bar"
        sample = self.dp_from(config, fake_analog_source, up=6.0, down=5.0)
        assert sample.values["pulse_dp"] == pytest.approx(10.0)

    def test_it_shows_the_zero_mismatch_a_holding_rig_still_has(
        self, fake_analog_source
    ):
        """What the pulse-decay fit's free offset exists to absorb; see it first."""
        config = rig_with_pulse_pair()
        config.run.display_pressure_unit = "bar"
        sample = self.dp_from(config, fake_analog_source, up=5.0, down=5.02)
        assert sample.values["pulse_dp"] == pytest.approx(-0.2)

    def test_a_gauge_pair_does_not_add_atmospheric_twice(self, fake_analog_source):
        """dP is a difference, so the atmospheric term cancels out of it."""
        config = rig_with_pulse_pair()
        absolute = self.dp_from(config, fake_analog_source, up=6.0, down=5.0)
        for end in ("upstream", "downstream"):
            getattr(config.hardware.pulse_transducers, end).reading_type = "gauge"
        gauge = self.dp_from(config, fake_analog_source, up=6.0, down=5.0)
        assert gauge.values["pulse_dp"] == pytest.approx(absolute.values["pulse_dp"])

    def test_it_carries_no_raw_value(self, fake_analog_source):
        """A difference of two voltages is not one, so there is nothing to store."""
        sample = self.dp_from(
            rig_with_pulse_pair(), fake_analog_source, up=6.0, down=5.0
        )
        assert "pulse_dp" not in sample.raw
        assert "pulse_dp" in sample.values

    def test_selecting_it_alone_still_opens_both_channels(self):
        signals = resolve_signals(rig_with_pulse_pair(), ["pulse_dp"])
        specs = preview_channel_specs(signals)
        assert [spec.name for spec in specs] == ["ai4", "ai5"]
        assert all((s.min_volts, s.max_volts) == (0.0, 10.0) for s in specs)

    def test_the_whole_group_still_opens_each_input_once(self):
        signals = resolve_signals(rig_with_pulse_pair(), ["pulse"])
        assert [spec.name for spec in preview_channel_specs(signals)] == ["ai4", "ai5"]

    def test_it_takes_its_own_unit_not_the_upstream_panels(self):
        """`pulse_upstream:bar` and `pulse_dp:kPa` must not cross over."""
        signals = resolve_signals(
            rig_with_pulse_pair(), ["pulse_upstream:bar", "pulse_dp:kPa"]
        )
        dp = signals[1]
        assert dp.unit == "kPa"
        assert dp.convert(6.0) - dp.subtracted.convert(5.0) == pytest.approx(
            units.from_atm(units.to_atm(10.0, "bar"), "kPa")
        )

    def test_a_unit_from_the_wrong_family_is_refused(self):
        with pytest.raises(PreviewError, match="pressure unit"):
            resolve_signals(rig_with_pulse_pair(), ["pulse_dp:sccm"])

    def test_it_stays_in_its_unit_under_volts(self):
        signals = resolve_signals(rig_with_pulse_pair(), ["pulse_dp"])
        dp = signals[0]
        assert dp.calibrated_only is True
        assert dp.shown_as_volts(volts=True) is False
        assert "dP (kPa)" in preview_header(signals, volts=True)
        sample = PreviewSample(0, 0.0, {"pulse_dp": 1000.0}, {})
        assert "1000" in format_preview_line(sample, signals, volts=True)

    def test_the_default_selection_includes_it_with_a_dedicated_pair(self):
        assert "pulse_dp" in default_selection(rig_with_pulse_pair())
        assert "pulse_dp" not in default_selection(GaspermConfig())

    def test_it_can_still_be_asked_for_on_a_rig_without_a_dedicated_pair(self):
        """The steady-state differential is worth watching too."""
        signals = resolve_signals(GaspermConfig(), ["pulse_dp"])
        assert (signals[0].channel, signals[0].subtracted.channel) == ("ai0", "ai1")
        assert "NO dedicated pulse pair" in signals[0].detail

    def test_it_gets_its_own_plot_panel_in_its_own_unit(self):
        from gasperm.plotting import PreviewPlot

        signals = resolve_signals(rig_with_pulse_pair(), ["pulse"])
        plot = PreviewPlot(signals, volts=True).open()
        try:
            assert [p.key for p in plot._panels] == [
                "pulse_upstream", "pulse_downstream", "pulse_dp"
            ]
            # --volts relabels the two transducers but not their difference.
            assert [p.ylabel for p in plot._panels] == ["pP1 (V)", "pP2 (V)", "dP (kPa)"]
            plot.add(
                PreviewSample(
                    0, 0.0,
                    {"pulse_upstream": 6e5, "pulse_downstream": 5e5, "pulse_dp": 1e5},
                    {"pulse_upstream": 6.0, "pulse_downstream": 5.0},
                )
            )
            assert plot._history.recent["pulse_upstream"] == [6.0]
            assert plot._history.recent["pulse_dp"] == [1e5]
        finally:
            plot.close()


class TestConsole:
    def sample(self, **values):
        return PreviewSample(
            index=0, elapsed_s=1.5,
            values=values, raw={key: 1.0 for key in values},
        )

    def test_the_header_matches_the_line_width(self):
        signals = resolve_signals(GaspermConfig(), ["inlet_pressure", "flow"])
        header = preview_header(signals)
        line = format_preview_line(
            self.sample(inlet_pressure=500.0, **{"flow.low_range": 12.5}), signals
        )
        assert len(header) == len(line)

    def test_the_header_names_the_unit(self):
        signals = resolve_signals(GaspermConfig(), ["inlet_pressure:bar"])
        assert "P1 (bar)" in preview_header(signals)

    def test_volts_mode_shows_the_raw_value_and_says_so(self):
        signals = resolve_signals(GaspermConfig(), ["inlet_pressure"])
        sample = PreviewSample(
            index=0, elapsed_s=1.5,
            values={"inlet_pressure": 34475.0}, raw={"inlet_pressure": 2.5},
        )
        assert "P1 (V)" in preview_header(signals, volts=True)
        assert "P1 (kPa)" in preview_header(signals, volts=False)
        assert "2.5" in format_preview_line(sample, signals, volts=True)
        assert "34475" not in format_preview_line(sample, signals, volts=True)
        assert "3.448e+04" in format_preview_line(sample, signals, volts=False)

    def test_an_uncalibrated_channel_is_always_volts(self):
        signals = resolve_signals(GaspermConfig(), ["ai7"])
        assert "ai7 (V)" in preview_header(signals)

    def test_a_missing_value_is_a_dash_not_a_zero(self):
        signals = resolve_signals(GaspermConfig(), ["inlet_pressure"])
        assert "--" in format_preview_line(self.sample(), signals)


class TestThrottle:
    def test_the_first_update_is_always_due(self):
        assert ConsoleThrottle(0.5).due(100.0) is True

    def test_updates_inside_the_interval_are_skipped(self):
        throttle = ConsoleThrottle(0.5)
        throttle.due(100.0)
        assert throttle.due(100.2) is False
        assert throttle.due(100.6) is True


class TestPreviewPlot:
    def plot_for(self, keys, **kwargs):
        from gasperm.plotting import PreviewPlot

        signals = resolve_signals(GaspermConfig(), keys)
        return PreviewPlot(signals, **kwargs).open(), signals

    def test_one_panel_per_signal_in_selection_order(self):
        plot, _ = self.plot_for(["flow", "inlet_pressure"])
        assert [p.key for p in plot._panels] == ["flow.low_range", "inlet_pressure"]
        assert len(plot._axes) == 2
        plot.close()

    def test_the_panel_is_labelled_with_the_signal_unit(self):
        plot, _ = self.plot_for(["inlet_pressure:bar"])
        assert plot._panels[0].ylabel == "P1 (bar)"
        plot.close()

    def test_volts_mode_relabels_and_plots_the_raw_value(self):
        plot, _ = self.plot_for(["inlet_pressure"], volts=True)
        assert plot._panels[0].ylabel == "P1 (V)"
        plot.add(PreviewSample(0, 0.0, {"inlet_pressure": 500.0}, {"inlet_pressure": 2.5}))
        assert plot._history.recent["inlet_pressure"] == [2.5]
        plot.close()

    def test_no_criteria_are_drawn_on_top_of_the_traces(self):
        """No detector runs, so bands or shading would assert an untested claim.

        The corner readout is not one of those: it restates the sample that was
        just plotted, which is the one thing preview is certain of.
        """
        plot, _ = self.plot_for(["inlet_pressure"])
        for index in range(20):
            plot.add(
                PreviewSample(index, index * 0.1, {"inlet_pressure": 500.0}, {"inlet_pressure": 2.5})
            )
        plot.maybe_redraw(now=1000.0)
        axis = plot._axes[0]
        assert list(axis.patches) == []      # no steady shading
        assert len(axis.lines) == 1          # the trace, and no criterion lines
        texts = [t.get_text() for t in axis.texts]
        assert texts == ["500 kPa"]          # the readout, and nothing else
        plot.close()

    def test_the_readout_shows_the_latest_value(self):
        plot, _ = self.plot_for(["inlet_pressure"])
        for index, value in enumerate((500.0, 512.5, 523.75)):
            plot.add(PreviewSample(index, index * 0.1, {"inlet_pressure": value}, {}))
        plot.maybe_redraw(now=1000.0)
        assert [t.get_text() for t in plot._axes[0].texts] == ["523.8 kPa"]
        plot.close()

    def test_the_readout_follows_volts_mode(self):
        """It is the number written down from a wiring check; it must not lie."""
        plot, _ = self.plot_for(["inlet_pressure"], volts=True)
        plot.add(PreviewSample(0, 0.0, {"inlet_pressure": 500.0}, {"inlet_pressure": 2.5}))
        plot.maybe_redraw(now=1000.0)
        assert [t.get_text() for t in plot._axes[0].texts] == ["2.5 V"]
        plot.close()

    def test_a_signal_with_no_reading_reads_as_a_gap(self):
        """Same '--' the console shows; never the last value it happened to have."""
        plot, _ = self.plot_for(["temperature"])
        plot.add(PreviewSample(0, 0.0, {"temperature": 21.5}, {}))
        plot.add(PreviewSample(1, 0.1, {}, {}, temperature_ok=False))
        plot.maybe_redraw(now=1000.0)
        assert [t.get_text() for t in plot._axes[0].texts] == ["--"]
        plot.close()

    def test_a_missing_signal_leaves_a_gap(self):
        plot, _ = self.plot_for(["temperature"])
        plot.add(PreviewSample(0, 0.0, {}, {}, temperature_ok=False))
        assert math.isnan(plot._history.recent["temperature"][0])
        plot.close()

    def test_the_title_says_which_view_and_which_mode(self):
        plot, _ = self.plot_for(["inlet_pressure"], volts=True, window_s=30.0)
        plot.add(PreviewSample(0, 0.0, {"inlet_pressure": 1.0}, {"inlet_pressure": 1.0}))
        plot.maybe_redraw(now=1000.0)
        title = plot._figure._suptitle.get_text()
        assert "raw volts" in title and "last 30 s" in title
        plot.close()


class TestBenchConfig:
    """Preview describes a rig, so it must load without a plug."""

    def test_it_loads_hardware_and_run_with_no_sample_file(self, tmp_path):
        rig = init_rig(tmp_path / "rig")
        assert not (rig / "sample.yaml").exists()
        config = load_bench_config(rig)
        assert config.hardware.daq.device_name
        assert config.run.display_pressure_unit

    def test_a_missing_hardware_file_still_fails_clearly(self, tmp_path):
        with pytest.raises(ConfigError, match="hardware"):
            load_bench_config(tmp_path)


class TestPreviewCommand:
    """The command end to end, against the fake driver."""

    def run_preview(self, rig, *args, **kwargs):
        return runner.invoke(app, ["preview", "-c", str(rig), *args], **kwargs)

    def test_list_needs_no_hardware_at_all(self, tmp_path):
        rig = init_rig(tmp_path / "rig")
        result = self.run_preview(rig, "--list")
        assert result.exit_code == 0, result.output
        output = strip_ansi(result.output)
        assert "inlet_pressure" in output
        assert "ai0" in output
        assert "raw volts" in output

    def test_it_samples_and_stores_nothing(self, tmp_path, fake_nidaqmx):
        rig = init_rig(tmp_path / "rig")
        volts, _ = midscale_pressure(load_bench_config(rig))
        expected = available_signals(load_bench_config(rig))["inlet_pressure"].convert(
            volts
        )
        fake_nidaqmx.voltages = {"ai0": volts}
        result = self.run_preview(
            rig, "--signal", "inlet_pressure", "-n", "3", "--rate", "50"
        )
        assert result.exit_code == 0, result.output
        output = strip_ansi(result.output)
        assert "3 sample(s) previewed" in output
        assert "Nothing was written" in output
        assert f"{expected:.4g}" in output
        # No run directory, no CSV, no sidecar -- anywhere.
        assert list(rig.rglob("*.csv")) == []
        assert not (rig / "runs").exists()

    def test_only_the_selected_channel_is_opened(self, tmp_path, fake_nidaqmx):
        rig = init_rig(tmp_path / "rig")
        fake_nidaqmx.voltages = {"ai0": 1.0}
        result = self.run_preview(rig, "-s", "inlet_pressure", "-n", "1", "--rate", "50")
        assert result.exit_code == 0, result.output
        assert fake_nidaqmx.instances[-1].channel_names == ["ai0"]

    def test_the_inactive_flowmeter_can_be_watched(self, tmp_path, fake_nidaqmx):
        """The reason preview picks its own channels rather than the run's."""
        rig = init_rig(tmp_path / "rig")
        fake_nidaqmx.voltages = {"ai3": 0.02}
        result = self.run_preview(rig, "-s", "flow.high_range", "-n", "1", "--rate", "50")
        assert result.exit_code == 0, result.output
        assert fake_nidaqmx.instances[-1].channel_names == ["ai3"]

    def test_a_bare_channel_is_read_as_volts(self, tmp_path, fake_nidaqmx):
        rig = init_rig(tmp_path / "rig")
        fake_nidaqmx.voltages = {"ai7": 3.3}
        result = self.run_preview(rig, "-s", "ai7", "-n", "1", "--rate", "50")
        assert result.exit_code == 0, result.output
        added = fake_nidaqmx.instances[-1].ai_channels.added[0]
        assert (added["min_val"], added["max_val"]) == (-10.0, 10.0)
        assert "3.3" in strip_ansi(result.output)

    def test_volts_mode_reports_the_wire(self, tmp_path, fake_nidaqmx):
        rig = init_rig(tmp_path / "rig")
        config = load_bench_config(rig)
        volts, _ = midscale_pressure(config)
        calibrated = available_signals(config)["inlet_pressure"].convert(volts)
        fake_nidaqmx.voltages = {"ai0": volts}
        result = self.run_preview(
            rig, "-s", "inlet_pressure", "--volts", "-n", "1", "--rate", "50"
        )
        output = strip_ansi(result.output)
        assert "P1 (V)" in output
        assert f"{volts:.4g}" in output
        assert f"{calibrated:.4g}" not in output

    def test_a_unit_can_be_chosen_per_signal(self, tmp_path, fake_nidaqmx):
        rig = init_rig(tmp_path / "rig")
        fake_nidaqmx.voltages = {"ai0": 2.5}
        result = self.run_preview(
            rig, "-s", "inlet_pressure:bar", "-n", "1", "--rate", "50"
        )
        assert result.exit_code == 0, result.output
        assert "P1 (bar)" in strip_ansi(result.output)

    def test_an_unknown_signal_fails_before_the_daq_is_touched(
        self, tmp_path, fake_nidaqmx
    ):
        rig = init_rig(tmp_path / "rig")
        result = self.run_preview(rig, "-s", "p1", "-n", "1")
        assert result.exit_code == 1
        assert "not a signal on this rig" in strip_ansi(result.output)
        assert fake_nidaqmx.instances == []

    def test_opposite_plot_views_are_refused(self, tmp_path, fake_nidaqmx):
        rig = init_rig(tmp_path / "rig")
        result = self.run_preview(rig, "--plot-window", "10", "--plot-from-start")
        assert result.exit_code == 1
        assert "opposite views" in strip_ansi(result.output)

    def test_the_plot_flag_opens_a_window_and_closes_it(
        self, tmp_path, fake_nidaqmx, monkeypatch
    ):
        rig = init_rig(tmp_path / "rig")
        fake_nidaqmx.voltages = {"ai0": 2.5}
        opened = []

        from gasperm import plotting

        original = plotting.PreviewPlot

        class Recording(original):
            def open(self):
                opened.append(self)
                return self

            def close(self):
                self.closed = True

        monkeypatch.setattr(plotting, "PreviewPlot", Recording)
        result = self.run_preview(
            rig, "-s", "inlet_pressure", "--plot", "-n", "2", "--rate", "50"
        )
        assert result.exit_code == 0, result.output
        assert len(opened) == 1
        assert opened[0].closed is True
        assert len(opened[0]._history) == 2

    def test_the_probe_is_only_opened_when_it_is_wanted(
        self, tmp_path, fake_nidaqmx, monkeypatch
    ):
        rig = init_rig(tmp_path / "rig")
        fake_nidaqmx.voltages = {"ai0": 2.5}

        from gasperm import cli

        def refuse(config):
            raise AssertionError("the probe must not be opened for a pressure preview")

        monkeypatch.setattr(cli, "_open_temperature_source", refuse)
        result = self.run_preview(rig, "-s", "inlet_pressure", "-n", "1", "--rate", "50")
        assert result.exit_code == 0, result.output

    def test_an_explicitly_asked_for_probe_that_fails_is_fatal(
        self, tmp_path, fake_nidaqmx, monkeypatch
    ):
        rig = init_rig(tmp_path / "rig")

        from gasperm import cli

        def fail(config):
            raise OSError("COM4 could not be opened")

        monkeypatch.setattr(cli, "_open_temperature_source", fail)
        result = self.run_preview(rig, "-s", "temperature", "-n", "1")
        assert result.exit_code == 1
        assert "COM4" in strip_ansi(result.output)

    def test_a_default_probe_that_fails_only_drops_its_column(
        self, tmp_path, fake_nidaqmx, monkeypatch
    ):
        """The DAQ half of a default preview is still worth watching."""
        rig = init_rig(tmp_path / "rig")
        fake_nidaqmx.voltages = {"ai0": 2.5, "ai1": 0.5, "ai2": 1.0}

        from gasperm import cli

        def fail(config):
            raise OSError("COM4 could not be opened")

        monkeypatch.setattr(cli, "_open_temperature_source", fail)
        result = self.run_preview(rig, "-n", "1", "--rate", "50")
        assert result.exit_code == 0, result.output
        output = strip_ansi(result.output)
        assert "Previewing without the temperature probe" in output
        assert "P1 (kPa)" in output

    def test_the_console_is_throttled_but_ends_on_the_last_sample(
        self, tmp_path, fake_nidaqmx
    ):
        """The DAQ runs at full rate; the text does not, and must still be current."""

        class Ramp(dict):
            """A different voltage every read, so each sample is identifiable."""

            def __init__(self) -> None:
                super().__init__()
                self.reads = 0

            def get(self, key, default=0.0):  # noqa: ANN001
                if key != "ai0":
                    return default
                self.reads += 1
                return 0.1 * self.reads

        rig = init_rig(tmp_path / "rig")
        fake_nidaqmx.voltages = Ramp()
        config = load_bench_config(rig)
        convert = available_signals(config)["inlet_pressure"].convert

        result = self.run_preview(rig, "-s", "inlet_pressure", "-n", "8", "--rate", "400")
        assert result.exit_code == 0, result.output
        output = strip_ansi(result.output)

        data_lines = [
            line for line in output.splitlines() if re.match(r"\s*[\d.]+s\s", line)
        ]
        # Eight samples in 20 ms, against a 0.5 s console interval: one tick,
        # plus the final line.
        assert 0 < len(data_lines) < 8
        assert f"{convert(0.1):.4g}" in output   # the first, printed on the first tick
        assert f"{convert(0.8):.4g}" in output   # the last, printed unconditionally

    def test_the_pulse_pair_is_opened_by_one_flag(self, tmp_path, fake_nidaqmx):
        """The case that matters: check both pulse transducers, whichever they are."""
        rig = init_rig_with_pulse_pair(tmp_path / "rig")
        fake_nidaqmx.voltages = {"ai4": 1.0, "ai5": 0.99}
        result = self.run_preview(rig, "-s", "pulse", "-n", "1", "--rate", "50")
        assert result.exit_code == 0, result.output
        assert fake_nidaqmx.instances[-1].channel_names == ["ai4", "ai5"]
        output = strip_ansi(result.output)
        assert "pP1" in output and "pP2" in output
        assert "dedicated pulse transducer" in output

    def test_without_a_dedicated_pair_it_opens_the_steady_state_one_and_says_so(
        self, tmp_path, fake_nidaqmx
    ):
        rig = init_rig(tmp_path / "rig")
        fake_nidaqmx.voltages = {"ai0": 1.0, "ai1": 0.99}
        result = self.run_preview(rig, "-s", "pulse", "-n", "1", "--rate", "50")
        assert result.exit_code == 0, result.output
        assert fake_nidaqmx.instances[-1].channel_names == ["ai0", "ai1"]
        assert "NO dedicated pulse pair" in strip_ansi(result.output)

    def test_list_warns_when_the_rig_has_no_dedicated_pulse_pair(self, tmp_path):
        rig = init_rig(tmp_path / "rig")
        output = strip_ansi(self.run_preview(rig, "--list").output)
        assert "NO dedicated pulse transducers" in output
        assert "hardware.pulse_transducers" in output

    def test_a_missing_rig_says_which_file(self, tmp_path):
        result = self.run_preview(tmp_path / "nowhere", "--list")
        assert result.exit_code == 1
        assert "hardware.yaml" in strip_ansi(result.output)

    def test_no_sample_file_is_required(self, tmp_path, fake_nidaqmx):
        """You preview a bench with nothing in the holder."""
        rig = init_rig(tmp_path / "rig")
        fake_nidaqmx.voltages = {"ai0": 2.5}
        for stray in rig.rglob("*.yaml"):
            assert stray.name != "sample.yaml"
        result = self.run_preview(rig, "-s", "inlet_pressure", "-n", "1", "--rate", "50")
        assert result.exit_code == 0, result.output

    def test_the_flags_are_documented_in_help(self):
        result = runner.invoke(app, ["preview", "--help"], env={"COLUMNS": "200"})
        output = strip_ansi(result.output)
        for flag in ("--signal", "--list", "--volts", "--plot", "--rate"):
            assert flag in output
