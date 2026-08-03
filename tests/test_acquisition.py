"""Per-sample computation and the acquisition loop, driven by fakes.

The end-to-end unit-invariance test here is the one that would catch a
conversion factor going wrong at the calibration boundary: two configs
describing the *same physical rig* in different units must produce the same
permeability from the same voltages.
"""

from __future__ import annotations

import math

import pytest

from gasperm import units
from gasperm.acquisition import (
    AcquisitionLoop,
    RollingWindow,
    SampleProcessor,
    console_header,
    format_reading_line,
    steady_state_stats,
)
from gasperm.config import GaspermConfig
from gasperm.gas_properties import FixedPropertyProvider
from gasperm.hardware.daq import DaqError
from gasperm.hardware.temperature import TemperatureSample
from gasperm.permeability import compute_gas_permeability

VOLTAGES = {"ai0": 2.5, "ai1": 0.5, "ai2": 4.0}


def make_processor(config: GaspermConfig, viscosity_cp: float = 0.0178) -> SampleProcessor:
    return SampleProcessor(config, FixedPropertyProvider("Nitrogen", viscosity_cp))


def sample(temperature_c: float | None = 22.0, *, stale: bool = False) -> TemperatureSample:
    return TemperatureSample(temperature_c, 0.0, None, stale)


class FakeClock:
    """Monotonic clock advancing a fixed step per call."""

    def __init__(self, step: float = 0.1) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


# --------------------------------------------------------------------------
# Rolling window
# --------------------------------------------------------------------------


class TestRollingWindow:
    def test_mean_of_everything_inside_the_window(self):
        window = RollingWindow(5.0)
        for t, value in enumerate([1.0, 2.0, 3.0]):
            window.add(float(t), value)
        assert window.mean() == pytest.approx(2.0)

    def test_points_older_than_the_window_are_dropped(self):
        window = RollingWindow(2.0)
        window.add(0.0, 100.0)
        window.add(1.0, 2.0)
        window.add(3.0, 4.0)
        assert window.count == 2
        assert window.mean() == pytest.approx(3.0)

    def test_empty_window_has_no_mean(self):
        assert RollingWindow(1.0).mean() is None
        assert RollingWindow(1.0).stddev() is None

    def test_non_finite_values_are_ignored(self):
        window = RollingWindow(10.0)
        window.add(0.0, 1.0)
        window.add(1.0, math.nan)
        window.add(2.0, 3.0)
        assert window.count == 2
        assert window.mean() == pytest.approx(2.0)

    def test_stddev_needs_two_points(self):
        window = RollingWindow(10.0)
        window.add(0.0, 1.0)
        assert window.stddev() is None
        window.add(1.0, 3.0)
        assert window.stddev() == pytest.approx(math.sqrt(2.0))

    def test_non_positive_window_is_rejected(self):
        with pytest.raises(ValueError):
            RollingWindow(0.0)

    def test_steady_state_stats_uses_only_the_trailing_window(self):
        times = [0.0, 1.0, 2.0, 8.0, 9.0, 10.0]
        values = [99.0, 99.0, 99.0, 1.0, 2.0, 3.0]
        mean, stddev, n = steady_state_stats(times, values, 5.0)
        assert n == 3
        assert mean == pytest.approx(2.0)
        assert stddev == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Per-sample computation
# --------------------------------------------------------------------------


class TestSampleProcessor:
    def test_matches_a_hand_calculation_through_every_conversion(self, base_config):
        """Voltages -> calibration -> CGS -> Darcy, checked end to end."""
        base_config.run.outlet_pressure_reference = "measured"
        processor = make_processor(base_config, viscosity_cp=0.0178)
        reading = processor.process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )

        # ai0 = 2.5 V of 0-5 V -> 500 kPa; ai1 = 0.5 V -> 100 kPa.
        assert reading.inlet_pressure_atm == pytest.approx(units.kpa_to_atm(500.0))
        assert reading.outlet_pressure_atm == pytest.approx(units.kpa_to_atm(100.0))
        # ai2 = 4.0 V of 0-10 V -> 200 sccm -> 200/60 cm^3/s.
        assert reading.flow_cm3_s == pytest.approx(200.0 / 60.0)

        expected = compute_gas_permeability(
            flow_rate_cm3_s=200.0 / 60.0,
            reference_pressure_atm=units.kpa_to_atm(101.325),
            viscosity_cp=0.0178,
            length_cm=5.0,
            area_cm2=units.circle_area_cm2(2.54),
            inlet_pressure_atm=units.kpa_to_atm(500.0),
            outlet_pressure_atm=units.kpa_to_atm(100.0),
        )
        assert reading.permeability_darcy == pytest.approx(expected, rel=1e-12)

    def test_mean_pressure_uses_the_downstream_reference(self, base_config):
        base_config.run.outlet_pressure_reference = "measured"
        reading = make_processor(base_config).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        assert reading.mean_pressure_atm == pytest.approx(
            (reading.inlet_pressure_atm + reading.downstream_pressure_atm) / 2.0
        )

    def test_atmospheric_reference_overrides_the_outlet_transducer(self, base_config):
        base_config.run.outlet_pressure_reference = "atmospheric"
        reading = make_processor(base_config).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        assert reading.downstream_pressure_atm == pytest.approx(1.0, rel=1e-12)
        # The measured value is still recorded, just not used as P2.
        assert reading.outlet_pressure_atm == pytest.approx(units.kpa_to_atm(100.0))

    def test_a_fixed_back_pressure_is_honoured(self, base_config):
        base_config.run.outlet_pressure_reference = 2.0
        base_config.run.outlet_pressure_reference_unit = "bar"
        reading = make_processor(base_config).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        assert reading.downstream_pressure_atm == pytest.approx(units.bar_to_atm(2.0))

    def test_no_differential_yields_no_permeability_but_still_a_reading(self, base_config):
        base_config.run.outlet_pressure_reference = "measured"
        flat = {"ai0": 1.0, "ai1": 1.0, "ai2": 4.0}
        reading = make_processor(base_config).process(
            index=0, elapsed_s=0.0, voltages=flat, temperature=sample()
        )
        assert reading.permeability_darcy is None
        assert reading.note is not None
        assert reading.inlet_pressure_atm == pytest.approx(reading.outlet_pressure_atm)

    def test_a_missing_channel_names_the_role(self, base_config):
        with pytest.raises(KeyError, match="flow channel"):
            make_processor(base_config).process(
                index=0, elapsed_s=0.0, voltages={"ai0": 2.5, "ai1": 0.5}, temperature=sample()
            )

    def test_a_missing_temperature_falls_back_and_is_flagged(self, base_config):
        base_config.temperature.fallback_temperature_c = 18.0
        reading = make_processor(base_config).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample(None)
        )
        assert reading.temperature_c == 18.0
        assert reading.temperature_ok is False
        assert reading.temperature_stale is True

    def test_a_stale_temperature_is_marked(self, base_config):
        reading = make_processor(base_config).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample(21.0, stale=True)
        )
        assert reading.temperature_ok is True
        assert reading.temperature_stale is True

    def test_rolling_average_builds_over_samples(self, base_config):
        base_config.run.outlet_pressure_reference = "measured"
        processor = make_processor(base_config)
        first = processor.process(index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample())
        louder = {**VOLTAGES, "ai2": 2.0}
        second = processor.process(index=1, elapsed_s=0.1, voltages=louder, temperature=sample())
        assert second.permeability_darcy < first.permeability_darcy
        assert second.permeability_darcy_avg == pytest.approx(
            (first.permeability_darcy + second.permeability_darcy) / 2.0
        )

    def test_gauge_transducers_shift_both_pressures(self, base_config):
        base_config.run.outlet_pressure_reference = "measured"
        absolute_reading = make_processor(base_config).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        base_config.pressure_calibration.inlet.reading_type = "gauge"
        base_config.pressure_calibration.outlet.reading_type = "gauge"
        gauge_reading = make_processor(base_config).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        assert gauge_reading.inlet_pressure_atm == pytest.approx(
            absolute_reading.inlet_pressure_atm + 1.0
        )
        # Treating gauge as absolute overstates permeability, which is exactly
        # the silent error the explicit flag exists to prevent.
        assert gauge_reading.permeability_darcy < absolute_reading.permeability_darcy


class TestUnitInvariance:
    """The same physical rig described in different units must agree."""

    @staticmethod
    def _metric_config() -> GaspermConfig:
        return GaspermConfig.model_validate(
            {
                "pressure_calibration": {
                    "inlet": {"volts_max": 5.0, "value_max": 1000.0, "unit": "kPa"},
                    "outlet": {"volts_max": 5.0, "value_max": 1000.0, "unit": "kPa"},
                },
                "flowmeter": {"flow_max": 500.0, "unit": "sccm"},
                "sample": {
                    "length_cm": 5.0,
                    "diameter_cm": 2.54,
                    "confining_pressure": 20.0,
                    "confining_pressure_unit": "MPa",
                },
                "run": {
                    "outlet_pressure_reference": "measured",
                    "atmospheric_pressure": 101.325,
                    "atmospheric_pressure_unit": "kPa",
                    "display_pressure_unit": "kPa",
                },
            }
        )

    @staticmethod
    def _mixed_config() -> GaspermConfig:
        """Identical rig: inlet in bar, outlet in psi, ambient in Pa, flow in slpm."""
        return GaspermConfig.model_validate(
            {
                "pressure_calibration": {
                    "inlet": {"volts_max": 5.0, "value_max": 10.0, "unit": "bar"},
                    "outlet": {
                        "volts_max": 5.0,
                        "value_max": units.from_atm(units.kpa_to_atm(1000.0), "psi"),
                        "unit": "psi",
                    },
                },
                "flowmeter": {"flow_max": 0.5, "unit": "slpm"},
                "sample": {
                    "length_cm": 5.0,
                    "diameter_cm": 2.54,
                    "confining_pressure": 2900.75,
                    "confining_pressure_unit": "psi",
                },
                "run": {
                    "outlet_pressure_reference": "measured",
                    "atmospheric_pressure": 101_325.0,
                    "atmospheric_pressure_unit": "Pa",
                    "display_pressure_unit": "MPa",
                },
            }
        )

    def test_pressures_agree(self):
        metric = make_processor(self._metric_config()).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        mixed = make_processor(self._mixed_config()).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        assert mixed.inlet_pressure_atm == pytest.approx(metric.inlet_pressure_atm, rel=1e-9)
        assert mixed.outlet_pressure_atm == pytest.approx(metric.outlet_pressure_atm, rel=1e-9)

    def test_flow_agrees(self):
        metric = make_processor(self._metric_config()).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        mixed = make_processor(self._mixed_config()).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        assert mixed.flow_cm3_s == pytest.approx(metric.flow_cm3_s, rel=1e-12)

    def test_permeability_is_unit_invariant(self):
        metric = make_processor(self._metric_config()).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        mixed = make_processor(self._mixed_config()).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        assert mixed.permeability_darcy == pytest.approx(metric.permeability_darcy, rel=1e-8)

    def test_confining_pressure_agrees_despite_different_units(self):
        # 20 MPa == 2900.75 psi to five figures.
        assert self._mixed_config().sample.confining_pressure_atm == pytest.approx(
            self._metric_config().sample.confining_pressure_atm, rel=1e-4
        )

    def test_display_units_do_not_affect_the_physics(self):
        config_a = self._metric_config()
        config_b = self._metric_config()
        config_b.run.display_pressure_unit = "psi"
        config_b.run.display_permeability_unit = "um2"
        reading_a = make_processor(config_a).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        reading_b = make_processor(config_b).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        assert reading_b.permeability_darcy == pytest.approx(reading_a.permeability_darcy)


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


class TestAcquisitionLoop:
    def _loop(self, config, analog, temperature, **kwargs):
        return AcquisitionLoop(
            config,
            make_processor(config),
            analog,
            temperature,
            sleep=lambda _s: None,
            **kwargs,
        )

    def test_stops_after_max_samples(
        self, base_config, fake_analog_source, fake_temperature_source
    ):
        base_config.run.max_samples = 5
        base_config.daq.sample_rate_hz = 1000.0
        analog = fake_analog_source(VOLTAGES)
        loop = self._loop(base_config, analog, fake_temperature_source())
        readings = loop.run(install_signal_handler=False)
        assert len(readings) == 5
        assert [r.index for r in readings] == [0, 1, 2, 3, 4]

    def test_stops_after_the_configured_duration(
        self, base_config, fake_analog_source, fake_temperature_source
    ):
        base_config.run.duration_s = 1.0
        base_config.daq.sample_rate_hz = 10.0
        loop = self._loop(
            base_config,
            fake_analog_source(VOLTAGES),
            fake_temperature_source(),
            clock=FakeClock(step=0.25),
        )
        readings = loop.run(install_signal_handler=False)
        assert 0 < len(readings) < 10

    def test_every_reading_reaches_the_callback(
        self, base_config, fake_analog_source, fake_temperature_source
    ):
        base_config.run.max_samples = 4
        base_config.daq.sample_rate_hz = 1000.0
        seen = []
        loop = self._loop(
            base_config,
            fake_analog_source(VOLTAGES),
            fake_temperature_source(),
            on_reading=seen.append,
        )
        loop.run(install_signal_handler=False)
        assert len(seen) == 4

    def test_a_failing_callback_does_not_stop_the_run(
        self, base_config, fake_analog_source, fake_temperature_source
    ):
        base_config.run.max_samples = 3
        base_config.daq.sample_rate_hz = 1000.0

        def explode(_reading):
            raise RuntimeError("the plot window was closed")

        loop = self._loop(
            base_config,
            fake_analog_source(VOLTAGES),
            fake_temperature_source(),
            on_reading=explode,
        )
        assert len(loop.run(install_signal_handler=False)) == 3

    def test_request_stop_ends_the_loop(
        self, base_config, fake_analog_source, fake_temperature_source
    ):
        base_config.daq.sample_rate_hz = 1000.0
        analog = fake_analog_source(VOLTAGES)
        loop = self._loop(base_config, analog, fake_temperature_source())

        def stop_after_two(reading):
            if reading.index >= 1:
                loop.request_stop()

        loop.on_reading = stop_after_two
        readings = loop.run(install_signal_handler=False)
        assert len(readings) == 2

    def test_a_daq_failure_propagates_and_still_closes_the_sources(
        self, base_config, fake_analog_source, fake_temperature_source
    ):
        base_config.daq.sample_rate_hz = 1000.0
        analog = fake_analog_source(VOLTAGES)
        analog.fail_after = 3
        temperature = fake_temperature_source()
        loop = self._loop(base_config, analog, temperature)
        with pytest.raises(DaqError):
            loop.run(install_signal_handler=False)
        assert analog.closed and temperature.closed
        assert any("DAQ read failed" in w for w in loop.warnings)

    def test_a_missing_temperature_warns_once_not_every_sample(
        self, base_config, fake_analog_source, fake_temperature_source
    ):
        base_config.run.max_samples = 10
        base_config.daq.sample_rate_hz = 1000.0
        loop = self._loop(
            base_config, fake_analog_source(VOLTAGES), fake_temperature_source(None)
        )
        loop.run(install_signal_handler=False)
        assert sum("No temperature reading" in w for w in loop.warnings) == 1

    def test_sources_are_closed_even_on_a_clean_finish(
        self, base_config, fake_analog_source, fake_temperature_source
    ):
        base_config.run.max_samples = 2
        base_config.daq.sample_rate_hz = 1000.0
        analog = fake_analog_source(VOLTAGES)
        temperature = fake_temperature_source()
        self._loop(base_config, analog, temperature).run(install_signal_handler=False)
        assert analog.closed and temperature.closed


class TestSummary:
    def test_summary_reports_the_trailing_window(
        self, base_config, fake_analog_source, fake_temperature_source
    ):
        base_config.run.max_samples = 20
        base_config.run.outlet_pressure_reference = "measured"
        base_config.daq.sample_rate_hz = 1000.0
        loop = AcquisitionLoop(
            base_config,
            make_processor(base_config),
            fake_analog_source(VOLTAGES),
            fake_temperature_source(),
            sleep=lambda _s: None,
            clock=FakeClock(step=0.05),
        )
        loop.run(install_signal_handler=False)
        summary = loop.summarize(csv_path="somewhere.csv")

        assert summary.sample_id == base_config.sample.id
        assert summary.gas_name == base_config.gas.name
        assert summary.sample_count == len(loop.readings)
        assert summary.permeability_darcy > 0.0
        # Constant voltages -> no scatter.
        assert summary.permeability_stddev_darcy == pytest.approx(0.0, abs=1e-12)
        assert summary.csv_path == "somewhere.csv"

    def test_a_run_with_no_usable_sample_is_reported_clearly(
        self, base_config, fake_analog_source, fake_temperature_source
    ):
        base_config.run.max_samples = 3
        base_config.run.outlet_pressure_reference = "measured"
        base_config.daq.sample_rate_hz = 1000.0
        flat = {"ai0": 1.0, "ai1": 1.0, "ai2": 4.0}
        loop = AcquisitionLoop(
            base_config,
            make_processor(base_config),
            fake_analog_source(flat),
            fake_temperature_source(),
            sleep=lambda _s: None,
        )
        loop.run(install_signal_handler=False)
        with pytest.raises(ValueError, match="No sample produced"):
            loop.summarize()


class TestConsoleFormatting:
    def test_line_uses_the_configured_display_units(self, base_config):
        base_config.run.display_pressure_unit = "bar"
        base_config.run.display_permeability_unit = "mD"
        reading = make_processor(base_config).process(
            index=0, elapsed_s=1.5, voltages=VOLTAGES, temperature=sample()
        )
        line = format_reading_line(reading, base_config)
        assert "bar" in line and "mD" in line
        assert "1.5s" in line

    def test_a_stale_temperature_is_marked_in_the_line(self, base_config):
        reading = make_processor(base_config).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample(21.0, stale=True)
        )
        assert "*" in format_reading_line(reading, base_config)

    def test_an_unusable_sample_shows_its_note(self, base_config):
        base_config.run.outlet_pressure_reference = "measured"
        flat = {"ai0": 1.0, "ai1": 1.0, "ai2": 4.0}
        reading = make_processor(base_config).process(
            index=0, elapsed_s=0.0, voltages=flat, temperature=sample()
        )
        line = format_reading_line(reading, base_config)
        assert "[" in line and "--" in line

    def test_header_names_the_display_units(self, base_config):
        header = console_header(base_config)
        assert base_config.run.display_pressure_unit in header
        assert base_config.run.display_permeability_unit in header
