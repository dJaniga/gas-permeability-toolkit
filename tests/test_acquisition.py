"""Per-sample computation, the acquisition loop, and steady-state gating.

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
    summarize_run,
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


def in_kpa(config: GaspermConfig) -> GaspermConfig:
    """Pin both transducers to 0-1000 kPa.

    The hand calculations below are worked in kPa, so they must not depend on
    whichever unit the rig happens to ship as its default.
    """
    for channel in (
        config.hardware.pressure_calibration.inlet,
        config.hardware.pressure_calibration.outlet,
    ):
        channel.unit = "kPa"
        channel.value_min = 0.0
        channel.value_max = 1000.0
    return config


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

    def test_non_positive_window_is_rejected(self):
        with pytest.raises(ValueError):
            RollingWindow(0.0)

    def test_steady_state_stats_uses_only_the_trailing_window(self):
        times = [0.0, 1.0, 2.0, 8.0, 9.0, 10.0]
        values = [99.0, 99.0, 99.0, 1.0, 2.0, 3.0]
        mean, stddev, n = steady_state_stats(times, values, 5.0)
        assert (n, mean, stddev) == (3, pytest.approx(2.0), pytest.approx(1.0))


# --------------------------------------------------------------------------
# Per-sample computation
# --------------------------------------------------------------------------


class TestSampleProcessor:
    def test_matches_a_hand_calculation_through_every_conversion(self, base_config):
        # Pin the geometry too, in cm, so the whole line below stays a genuine
        # hand calculation rather than tracking the shipped plug defaults.
        base_config.sample.dimension_unit = "cm"
        base_config.sample.length = 5.0
        base_config.sample.diameter = 2.54
        processor = make_processor(in_kpa(base_config), viscosity_cp=0.0178)
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

    def test_p2_comes_from_the_outlet_transducer_by_default(self, base_config):
        """The default is 'measured'; configuration does not substitute for it."""
        base_config = in_kpa(base_config)
        reading = make_processor(base_config).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        # ai1 = 0.5 V of 0-5 V -> 100 kPa, which is NOT the 101.325 kPa ambient.
        assert reading.outlet_pressure_atm == pytest.approx(units.kpa_to_atm(100.0))
        assert reading.downstream_pressure_atm == reading.outlet_pressure_atm
        assert reading.outlet_pressure_atm != pytest.approx(
            base_config.run.atmospheric_pressure_atm
        )

    def test_the_outlet_voltage_moves_p2(self, base_config):
        processor = make_processor(base_config)
        low = processor.process(
            index=0, elapsed_s=0.0, voltages={**VOLTAGES, "ai1": 0.5}, temperature=sample()
        )
        high = processor.process(
            index=1, elapsed_s=0.1, voltages={**VOLTAGES, "ai1": 1.5}, temperature=sample()
        )
        assert high.outlet_pressure_atm > low.outlet_pressure_atm
        # A higher back-pressure narrows the differential, so k rises.
        assert high.permeability_darcy > low.permeability_darcy

    def test_the_ambient_value_is_unused_with_absolute_transducers(self, base_config):
        """It is the gauge-to-absolute reference, not a stand-in for P2."""
        baseline = make_processor(base_config).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        base_config.run.atmospheric_pressure = 95.0
        shifted = make_processor(base_config).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        assert shifted.outlet_pressure_atm == pytest.approx(baseline.outlet_pressure_atm)
        assert shifted.permeability_darcy == pytest.approx(baseline.permeability_darcy)

    def test_no_differential_yields_no_permeability_but_still_a_reading(self, base_config):
        flat = {"ai0": 1.0, "ai1": 1.0, "ai2": 4.0}
        reading = make_processor(base_config).process(
            index=0, elapsed_s=0.0, voltages=flat, temperature=sample()
        )
        assert reading.permeability_darcy is None
        assert reading.note is not None

    def test_a_missing_channel_names_the_role(self, base_config):
        with pytest.raises(KeyError, match="flow channel"):
            make_processor(base_config).process(
                index=0, elapsed_s=0.0, voltages={"ai0": 2.5, "ai1": 0.5}, temperature=sample()
            )

    def test_a_missing_temperature_falls_back_and_is_flagged(self, base_config):
        base_config.hardware.temperature.fallback_temperature_c = 18.0
        reading = make_processor(base_config).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample(None)
        )
        assert reading.temperature_c == 18.0
        assert reading.temperature_ok is False
        assert reading.temperature_stale is True

    def test_gauge_transducers_shift_both_pressures(self, base_config):
        absolute_reading = make_processor(base_config).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        base_config.hardware.pressure_calibration.inlet.reading_type = "gauge"
        base_config.hardware.pressure_calibration.outlet.reading_type = "gauge"
        gauge_reading = make_processor(base_config).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        assert gauge_reading.inlet_pressure_atm == pytest.approx(
            absolute_reading.inlet_pressure_atm + 1.0
        )
        assert gauge_reading.permeability_darcy < absolute_reading.permeability_darcy

    def test_steady_state_flags_are_carried_onto_the_reading(self, base_config):
        reading = make_processor(base_config).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample(),
            steady_state=True, steady_state_passes=3,
        )
        assert reading.steady_state is True
        assert reading.steady_state_passes == 3


class TestSuppliedDownstreamPressure:
    """P2 given by the operator, for a rig whose outlet vents to atmosphere."""

    @staticmethod
    def _config(base_config, value=101.325):
        config = in_kpa(base_config)
        config.run.downstream_pressure = value
        config.run.downstream_pressure_unit = "kPa"
        return config

    def _read(self, base_config, voltages=VOLTAGES, **kwargs):
        processor = make_processor(self._config(base_config, **kwargs))
        return processor.process(
            index=0, elapsed_s=0.0, voltages=voltages, temperature=sample()
        )

    def test_the_supplied_value_becomes_p2(self, base_config):
        reading = self._read(base_config)
        assert reading.downstream_pressure_atm == pytest.approx(
            units.kpa_to_atm(101.325)
        )

    def test_the_transducer_is_still_recorded(self, base_config):
        """It is the only evidence the declared pressure matches the rig."""
        reading = self._read(base_config)
        assert reading.outlet_pressure_atm == pytest.approx(units.kpa_to_atm(100.0))
        assert reading.outlet_pressure_atm != reading.downstream_pressure_atm

    def test_mean_pressure_uses_the_supplied_value(self, base_config):
        reading = self._read(base_config)
        assert reading.mean_pressure_atm == pytest.approx(
            (reading.inlet_pressure_atm + units.kpa_to_atm(101.325)) / 2.0
        )

    def test_the_outlet_voltage_no_longer_moves_the_permeability(self, base_config):
        low = self._read(base_config, {**VOLTAGES, "ai1": 0.5})
        high = self._read(base_config, {**VOLTAGES, "ai1": 2.5})
        assert high.outlet_pressure_atm > low.outlet_pressure_atm
        assert high.permeability_darcy == pytest.approx(low.permeability_darcy)

    def test_the_darcy_result_matches_a_hand_calculation(self, base_config):
        config = self._config(base_config)
        config.sample.dimension_unit = "cm"
        config.sample.length = 5.0
        config.sample.diameter = 2.54
        reading = make_processor(config, viscosity_cp=0.0178).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        expected = compute_gas_permeability(
            flow_rate_cm3_s=200.0 / 60.0,
            reference_pressure_atm=units.kpa_to_atm(101.325),
            viscosity_cp=0.0178,
            length_cm=5.0,
            area_cm2=units.circle_area_cm2(2.54),
            inlet_pressure_atm=units.kpa_to_atm(500.0),
            outlet_pressure_atm=units.kpa_to_atm(101.325),
        )
        assert reading.permeability_darcy == pytest.approx(expected, rel=1e-12)

    def test_a_meter_at_the_outlet_is_referenced_to_the_supplied_value(self, base_config):
        """Declaring the outlet line's pressure applies to a meter sitting on it."""
        config = self._config(base_config)
        config.hardware.flowmeters["low_range"].reading_basis = "actual"
        reading = make_processor(config).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        assert reading.flow_reference_pressure_atm == pytest.approx(
            units.kpa_to_atm(101.325)
        )

    def test_measured_can_be_restored(self, base_config):
        config = self._config(base_config)
        config.run.downstream_pressure = "measured"
        reading = make_processor(config).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        assert reading.downstream_pressure_atm == reading.outlet_pressure_atm

    def test_the_console_marks_a_supplied_p2(self, base_config):
        supplied = self._read(base_config)
        assert "*" in format_reading_line(supplied, self._config(base_config))


class TestDownstreamCrossCheck:
    """A declared P2 is an assertion; the transducer is the evidence."""

    def _summary(self, config, analog, temperature, samples=80):
        config.run.max_samples = samples
        loop = build_loop(config, analog(VOLTAGES), temperature(), clock=FakeClock(step=0.05))
        loop.run(install_signal_handler=False)
        return loop.summarize()

    def test_a_large_disagreement_is_reported(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        config = in_kpa(quick_steady_config)
        # ai1 reads 100 kPa; declaring 300 kPa means something is shut.
        config.run.downstream_pressure = 300.0
        summary = self._summary(config, fake_analog_source, fake_temperature_source)
        assert any("disagrees with the outlet transducer" in w for w in summary.warnings)

    def test_a_matching_value_is_silent(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        config = in_kpa(quick_steady_config)
        config.run.downstream_pressure = 100.0  # exactly what ai1 reads
        summary = self._summary(config, fake_analog_source, fake_temperature_source)
        assert not any("disagrees" in w for w in summary.warnings)

    def test_a_measured_run_is_never_checked(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        summary = self._summary(
            in_kpa(quick_steady_config), fake_analog_source, fake_temperature_source
        )
        assert not any("disagrees" in w for w in summary.warnings)


class TestTemperatureCadence:
    """A probe slower than the sample rate holds its value; that is normal."""

    def test_a_held_value_carries_its_age(self, base_config):
        reading = make_processor(base_config).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES,
            temperature=TemperatureSample(22.0, 0.0, None, False, 0.4),
        )
        assert reading.temperature_age_s == pytest.approx(0.4)
        assert reading.temperature_ok is True
        assert reading.temperature_stale is False

    def test_a_probe_keeping_up_is_not_reported(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        quick_steady_config.run.max_samples = 80
        loop = build_loop(
            quick_steady_config, fake_analog_source(VOLTAGES), fake_temperature_source(),
            clock=FakeClock(step=0.05),
        )
        loop.run(install_signal_handler=False)
        assert not any("fell behind" in w for w in loop.summarize().warnings)

    def test_a_probe_that_falls_behind_is_reported(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        """Holding across several conversions means it has stopped answering."""
        quick_steady_config.run.max_samples = 80
        quick_steady_config.hardware.temperature.conversion_time_s = 0.75
        stalled = fake_temperature_source()
        stalled.age_s = 30.0  # far past 3 x 0.75 s
        loop = build_loop(
            quick_steady_config, fake_analog_source(VOLTAGES), stalled,
            clock=FakeClock(step=0.05),
        )
        loop.run(install_signal_handler=False)
        warnings = loop.summarize().warnings
        assert any("fell behind" in w for w in warnings)
        assert any("held temperature" in w for w in warnings)


class TestRealGasCorrection:
    def test_off_by_default(self, base_config):
        assert base_config.run.gas.real_gas_correction is False
        reading = make_processor(base_config).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        assert reading.flow_reference_cm3_s == pytest.approx(reading.flow_cm3_s)

    def test_divides_the_reference_flow_by_z(self, base_config):
        pytest.importorskip("CoolProp")
        from gasperm.gas_properties import CoolPropProvider

        base_config.run.gas.real_gas_correction = True
        # Pin to kPa so the mean pore pressure stays near ambient, where CO2's
        # departure from ideality is the few tenths of a percent asserted below.
        processor = SampleProcessor(in_kpa(base_config), CoolPropProvider("CarbonDioxide"))
        reading = processor.process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        assert reading.compressibility_z is not None
        assert reading.flow_reference_cm3_s == pytest.approx(
            reading.flow_cm3_s / reading.compressibility_z
        )
        # CO2 near ambient is measurably non-ideal, so the correction bites.
        assert reading.compressibility_z < 0.999


class TestUnitInvariance:
    """The same physical rig described in different units must agree."""

    @staticmethod
    def _metric_config() -> GaspermConfig:
        return GaspermConfig.model_validate(
            {
                "hardware": {
                    "pressure_calibration": {
                        "inlet": {"volts_max": 5.0, "value_max": 1000.0, "unit": "kPa"},
                        "outlet": {"volts_max": 5.0, "value_max": 1000.0, "unit": "kPa"},
                    },
                    "flowmeters": {"m": {"flow_max": 500.0, "unit": "sccm"}},
                    "default_flowmeter": "m",
                },
                "sample": {
                    "dimension_unit": "cm",
                    "length": 5.0,
                    "diameter": 2.54,
                },
                "run": {
                    "atmospheric_pressure": 101.325,
                    "atmospheric_pressure_unit": "kPa",
                    "confining_pressure": 20.0,
                    "confining_pressure_unit": "MPa",
                    "display_pressure_unit": "kPa",
                },
            }
        )

    @staticmethod
    def _mixed_config() -> GaspermConfig:
        """Identical rig: inlet in bar, outlet in psi, ambient in Pa, flow in slpm."""
        return GaspermConfig.model_validate(
            {
                "hardware": {
                    "pressure_calibration": {
                        "inlet": {"volts_max": 5.0, "value_max": 10.0, "unit": "bar"},
                        "outlet": {
                            "volts_max": 5.0,
                            "value_max": units.from_atm(units.kpa_to_atm(1000.0), "psi"),
                            "unit": "psi",
                        },
                    },
                    "flowmeters": {"m": {"flow_max": 0.5, "unit": "slpm"}},
                    "default_flowmeter": "m",
                },
                "sample": {
                    "dimension_unit": "cm",
                    "length": 5.0,
                    "diameter": 2.54,
                },
                "run": {
                    "atmospheric_pressure": 101_325.0,
                    "atmospheric_pressure_unit": "Pa",
                    "confining_pressure": 2900.75,
                    "confining_pressure_unit": "psi",
                    "display_pressure_unit": "MPa",
                },
            }
        )

    def _readings(self):
        metric = make_processor(self._metric_config()).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        mixed = make_processor(self._mixed_config()).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        return metric, mixed

    def test_pressures_agree(self):
        metric, mixed = self._readings()
        assert mixed.inlet_pressure_atm == pytest.approx(metric.inlet_pressure_atm, rel=1e-9)
        assert mixed.outlet_pressure_atm == pytest.approx(metric.outlet_pressure_atm, rel=1e-9)

    def test_flow_agrees(self):
        metric, mixed = self._readings()
        assert mixed.flow_cm3_s == pytest.approx(metric.flow_cm3_s, rel=1e-12)

    def test_permeability_is_unit_invariant(self):
        metric, mixed = self._readings()
        assert mixed.permeability_darcy == pytest.approx(metric.permeability_darcy, rel=1e-8)

    def test_confining_pressure_agrees_despite_different_units(self):
        assert self._mixed_config().run.confining_pressure_atm == pytest.approx(
            self._metric_config().run.confining_pressure_atm, rel=1e-4
        )

    def test_display_units_do_not_affect_the_physics(self):
        config_a = self._metric_config()
        config_b = self._metric_config()
        config_b.run.display_pressure_unit = "psi"
        config_b.run.display_permeability_unit = "um2"
        a = make_processor(config_a).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        b = make_processor(config_b).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        assert b.permeability_darcy == pytest.approx(a.permeability_darcy)


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def build_loop(config, analog, temperature, **kwargs):
    return AcquisitionLoop(
        config, make_processor(config), analog, temperature, sleep=lambda _s: None, **kwargs
    )


class TestAcquisitionLoop:
    def test_stops_after_max_samples(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        quick_steady_config.run.max_samples = 5
        loop = build_loop(
            quick_steady_config, fake_analog_source(VOLTAGES), fake_temperature_source()
        )
        readings = loop.run(install_signal_handler=False)
        assert len(readings) == 5
        assert "max_samples" in loop.stop_reason

    def test_stops_after_the_configured_duration(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        quick_steady_config.run.duration_s = 1.0
        quick_steady_config.hardware.daq.sample_rate_hz = 10.0
        loop = build_loop(
            quick_steady_config, fake_analog_source(VOLTAGES), fake_temperature_source(),
            clock=FakeClock(step=0.25),
        )
        readings = loop.run(install_signal_handler=False)
        assert 0 < len(readings) < 10
        assert "duration_s" in loop.stop_reason

    def test_every_reading_reaches_the_callback(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        quick_steady_config.run.max_samples = 4
        seen = []
        loop = build_loop(
            quick_steady_config, fake_analog_source(VOLTAGES), fake_temperature_source(),
            on_reading=seen.append,
        )
        loop.run(install_signal_handler=False)
        assert len(seen) == 4

    def test_a_failing_callback_does_not_stop_the_run(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        quick_steady_config.run.max_samples = 3

        def explode(_reading):
            raise RuntimeError("the plot window was closed")

        loop = build_loop(
            quick_steady_config, fake_analog_source(VOLTAGES), fake_temperature_source(),
            on_reading=explode,
        )
        assert len(loop.run(install_signal_handler=False)) == 3

    def test_a_daq_failure_propagates_and_still_closes_the_sources(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        analog = fake_analog_source(VOLTAGES)
        analog.fail_after = 3
        temperature = fake_temperature_source()
        loop = build_loop(quick_steady_config, analog, temperature)
        with pytest.raises(DaqError):
            loop.run(install_signal_handler=False)
        assert analog.closed and temperature.closed
        assert any("DAQ read failed" in w for w in loop.warnings)

    def test_a_missing_temperature_warns_once_not_every_sample(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        quick_steady_config.run.max_samples = 10
        loop = build_loop(
            quick_steady_config, fake_analog_source(VOLTAGES), fake_temperature_source(None)
        )
        loop.run(install_signal_handler=False)
        assert sum("No temperature reading" in w for w in loop.warnings) == 1

    def test_sources_are_closed_on_a_clean_finish(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        quick_steady_config.run.max_samples = 2
        analog = fake_analog_source(VOLTAGES)
        temperature = fake_temperature_source()
        build_loop(quick_steady_config, analog, temperature).run(install_signal_handler=False)
        assert analog.closed and temperature.closed


class TestSteadyStateGating:
    def test_a_stable_rig_reaches_steady_state_and_flags_its_readings(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        quick_steady_config.run.max_samples = 60
        loop = build_loop(
            quick_steady_config, fake_analog_source(VOLTAGES), fake_temperature_source(),
            clock=FakeClock(step=0.05),
        )
        readings = loop.run(install_signal_handler=False)
        assert loop.steady_state_reached
        assert any(r.steady_state for r in readings)
        assert loop.steady_start_s is not None

    def test_a_drifting_rig_never_reaches_steady_state(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        quick_steady_config.run.max_samples = 60
        ramp = [{**VOLTAGES, "ai0": 1.0 + 0.05 * i} for i in range(60)]
        loop = build_loop(
            quick_steady_config, fake_analog_source(ramp), fake_temperature_source(),
            clock=FakeClock(step=0.05),
        )
        loop.run(install_signal_handler=False)
        assert not loop.steady_state_reached

    def test_a_zero_soak_stops_on_confirmation(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        quick_steady_config.run.stop_after_steady_s = 0.0
        quick_steady_config.run.max_samples = 500
        loop = build_loop(
            quick_steady_config, fake_analog_source(VOLTAGES), fake_temperature_source(),
            clock=FakeClock(step=0.05),
        )
        readings = loop.run(install_signal_handler=False)
        assert loop.steady_state_reached
        assert len(readings) < 500
        assert "stop_after_steady_s" in loop.stop_reason
        # Stopped on the very sample that confirmed it.
        assert readings[-1].elapsed_s == pytest.approx(loop.steady_confirmed_at_s)


class TestSteadySoakTime:
    """`stop_after_steady_s` records N seconds of *confirmed* steady state."""

    def _run(self, config, analog, temperature, *, soak, step=0.05, samples=2000):
        config.run.stop_after_steady_s = soak
        config.run.max_samples = samples
        loop = build_loop(config, analog, temperature, clock=FakeClock(step=step))
        loop.run(install_signal_handler=False)
        return loop

    def test_the_run_continues_for_the_soak_after_confirmation(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        loop = self._run(
            quick_steady_config, fake_analog_source(VOLTAGES),
            fake_temperature_source(), soak=1.0,
        )
        assert loop.steady_state_reached
        assert loop.steady_confirmed_at_s is not None
        held = loop.readings[-1].elapsed_s - loop.steady_confirmed_at_s
        assert held >= 1.0
        assert "held for 1 s" in loop.stop_reason

    def test_a_longer_soak_records_a_longer_run(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        short = self._run(
            quick_steady_config.model_copy(deep=True), fake_analog_source(VOLTAGES),
            fake_temperature_source(), soak=0.5,
        )
        long = self._run(
            quick_steady_config.model_copy(deep=True), fake_analog_source(VOLTAGES),
            fake_temperature_source(), soak=2.0,
        )
        assert len(long.readings) > len(short.readings)

    def test_the_soak_is_counted_from_confirmation_not_the_plateau_start(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        """The plateau is already window_s old when it is declared."""
        loop = self._run(
            quick_steady_config, fake_analog_source(VOLTAGES),
            fake_temperature_source(), soak=1.0,
        )
        # Measured from the plateau start the span is longer than the soak,
        # which is exactly the distinction the two readings differ on.
        assert loop.steady_start_s < loop.steady_confirmed_at_s
        assert (loop.readings[-1].elapsed_s - loop.steady_start_s) > 1.0

    def test_losing_steady_state_restarts_the_clock(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        """An interrupted hold did not last, so it must not count."""
        # Steady, then a step change, then steady again.
        settled = [VOLTAGES] * 40
        disturbed = [{**VOLTAGES, "ai0": 4.0}] * 20
        recovered = [{**VOLTAGES, "ai0": 4.0}] * 400
        loop = self._run(
            quick_steady_config, fake_analog_source(settled + disturbed + recovered),
            fake_temperature_source(), soak=1.0,
        )
        assert loop.steady_state_reached
        # Steady state was declared twice: the first hold was interrupted, so
        # the clock restarted rather than the run stopping on the first plateau.
        confirmations = [w for w in loop.warnings if "Steady state confirmed" in w]
        assert len(confirmations) == 2
        # The soak is measured from the second confirmation, not the first.
        held = loop.readings[-1].elapsed_s - loop.steady_confirmed_at_s
        assert held == pytest.approx(1.0, abs=0.3)

    def test_without_a_soak_the_run_is_not_stopped_by_steadiness(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        quick_steady_config.run.max_samples = 120
        loop = build_loop(
            quick_steady_config, fake_analog_source(VOLTAGES), fake_temperature_source(),
            clock=FakeClock(step=0.05),
        )
        readings = loop.run(install_signal_handler=False)
        assert loop.steady_state_reached
        assert len(readings) == 120
        assert "max_samples" in loop.stop_reason

    def test_max_wait_gives_up_on_a_rig_that_never_settles(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        quick_steady_config.run.steady_state.max_wait_s = 1.0
        quick_steady_config.run.max_samples = 500
        ramp = [{**VOLTAGES, "ai0": 1.0 + 0.02 * i} for i in range(500)]
        loop = build_loop(
            quick_steady_config, fake_analog_source(ramp), fake_temperature_source(),
            clock=FakeClock(step=0.05),
        )
        loop.run(install_signal_handler=False)
        assert not loop.steady_state_reached
        assert "timed out" in loop.stop_reason
        assert any("Gave up waiting" in w for w in loop.warnings)


class TestSummary:
    def _steady_loop(self, config, analog_factory, temperature_factory, samples=80):
        config.run.max_samples = samples
        loop = build_loop(
            config, analog_factory(VOLTAGES), temperature_factory(), clock=FakeClock(step=0.05)
        )
        loop.run(install_signal_handler=False)
        return loop

    def test_a_steady_run_reports_from_the_steady_window(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        loop = self._steady_loop(
            quick_steady_config, fake_analog_source, fake_temperature_source
        )
        summary = loop.summarize(csv_path="somewhere.csv")
        assert summary.steady_state_reached
        assert summary.is_representative
        assert summary.steady_state_window is not None
        assert summary.averaged_samples == summary.steady_state_window.sample_count
        assert summary.permeability_darcy > 0.0
        assert summary.csv_path == "somewhere.csv"

    def test_an_unsteady_run_is_marked_not_representative(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        quick_steady_config.run.max_samples = 60
        ramp = [{**VOLTAGES, "ai0": 1.0 + 0.05 * i} for i in range(60)]
        loop = build_loop(
            quick_steady_config, fake_analog_source(ramp), fake_temperature_source(),
            clock=FakeClock(step=0.05),
        )
        loop.run(install_signal_handler=False)
        summary = loop.summarize()
        assert not summary.steady_state_reached
        assert not summary.is_representative
        assert any("NOT a representative permeability" in w for w in summary.warnings)

    def test_the_summary_carries_experiment_metadata(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        quick_steady_config.run.operator = "Damian"
        quick_steady_config.sample.lithology = "sandstone"
        quick_steady_config.run.confining_pressure = 15.0
        loop = self._steady_loop(
            quick_steady_config, fake_analog_source, fake_temperature_source
        )
        metadata = loop.summarize().metadata
        assert metadata.operator == "Damian"
        assert metadata.lithology == "sandstone"
        assert metadata.confining_pressure == 15.0

    def test_the_summary_carries_an_uncertainty_budget(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        loop = self._steady_loop(
            quick_steady_config, fake_analog_source, fake_temperature_source
        )
        budget = loop.summarize().uncertainty
        assert budget is not None
        assert budget.relative_combined_standard_uncertainty > 0.0
        assert budget.expanded_uncertainty_darcy > budget.combined_standard_uncertainty_darcy
        assert {"Q", "P1", "P2", "L", "d", "mu"} <= {c.symbol for c in budget.components}

    def test_the_budget_can_be_disabled(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        quick_steady_config.run.uncertainty.enabled = False
        loop = self._steady_loop(
            quick_steady_config, fake_analog_source, fake_temperature_source
        )
        assert loop.summarize().uncertainty is None

    def test_a_run_with_no_usable_sample_is_reported_clearly(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        quick_steady_config.run.max_samples = 3
        flat = {"ai0": 1.0, "ai1": 1.0, "ai2": 4.0}
        loop = build_loop(
            quick_steady_config, fake_analog_source(flat), fake_temperature_source()
        )
        loop.run(install_signal_handler=False)
        with pytest.raises(ValueError, match="No sample produced"):
            loop.summarize()

    def test_summarize_run_is_usable_standalone(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        loop = self._steady_loop(
            quick_steady_config, fake_analog_source, fake_temperature_source
        )
        direct = summarize_run(
            loop.readings, quick_steady_config, steady_window=loop.steady_window()
        )
        assert direct.permeability_darcy == pytest.approx(
            loop.summarize().permeability_darcy
        )


class TestDominatedBudgetWarning:
    """A term worth a quarter of the answer means there is no answer.

    The failure this guards against is a thermal flowmeter parked near its
    zero: perfectly stable, so the steady-state detector confirms it, and the
    budget is the only thing that knows the number is meaningless.
    """

    def _summary(self, config, analog_factory, temperature_factory, voltages):
        config.run.max_samples = 80
        loop = build_loop(
            config, analog_factory(voltages), temperature_factory(), clock=FakeClock(step=0.05)
        )
        loop.run(install_signal_handler=False)
        return loop.summarize()

    def test_a_meter_near_its_zero_is_called_out(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        # 0.02 V on a 0-10 V, 0-500 sccm meter is 1 sccm: 0.2 % of full scale,
        # where the +/-0.5 % FS specification is 144 % of the reading.
        summary = self._summary(
            quick_steady_config, fake_analog_source, fake_temperature_source,
            {**VOLTAGES, "ai2": 0.02},
        )
        offending = [w for w in summary.warnings if "flow rate contributes" in w]
        assert len(offending) == 1
        assert "0.20% of its 500 sccm full scale" in offending[0]
        assert "not a measurement of the sample" in offending[0]

    def test_only_the_worst_offender_is_reported(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        """One message, however many terms blow up together."""
        quick_steady_config.run.uncertainty.max_component_contribution = 1e-6
        summary = self._summary(
            quick_steady_config, fake_analog_source, fake_temperature_source, VOLTAGES
        )
        offending = [w for w in summary.warnings if "contributes" in w]
        assert len(offending) == 1
        assert "other input(s) also exceed it" in offending[0]

    def test_a_meter_in_its_working_range_is_not(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        # 4.0 V is 200 sccm, 40 % of full scale -> the spec is 1.25 % of it.
        summary = self._summary(
            quick_steady_config, fake_analog_source, fake_temperature_source, VOLTAGES
        )
        assert not any("contributes" in w for w in summary.warnings)

    def test_the_threshold_is_configurable(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        quick_steady_config.run.uncertainty.max_component_contribution = 0.001
        summary = self._summary(
            quick_steady_config, fake_analog_source, fake_temperature_source, VOLTAGES
        )
        assert any("contributes" in w for w in summary.warnings)

    def test_the_check_can_be_switched_off(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        quick_steady_config.run.uncertainty.max_component_contribution = None
        summary = self._summary(
            quick_steady_config, fake_analog_source, fake_temperature_source,
            {**VOLTAGES, "ai2": 0.02},
        )
        assert not any("contributes" in w for w in summary.warnings)


class TestEquilibrationWarning:
    """A plateau is not equilibrium.

    ``t ~ phi mu L^2 / (k P_mean)`` is hours for tight rock, and the detector
    can confirm a plateau in ninety seconds. The two facts are compatible: the
    signal is flat because the core is still filling at a steady rate.
    """

    def _summary(self, config, analog_factory, temperature_factory):
        config.run.max_samples = 80
        loop = build_loop(
            config, analog_factory(VOLTAGES), temperature_factory(), clock=FakeClock(step=0.05)
        )
        loop.run(install_signal_handler=False)
        return loop.summarize()

    def permeable(self, config):
        """A 2.2 mD plug on a 1 cm core: t is well under the run's length."""
        in_kpa(config)
        config.sample.diameter = 10.0
        return config

    def test_a_tight_plug_run_too_briefly_is_flagged(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        # The shipped 0-68.95 MPa transducers put ai0 = 2.5 V at 340 atm, and
        # 200 sccm through this plug at that differential is 0.47 uD -- so the
        # default test rig really does describe a microdarcy sample.
        quick_steady_config.sample.porosity = 0.15
        summary = self._summary(
            quick_steady_config, fake_analog_source, fake_temperature_source
        )
        flagged = [w for w in summary.warnings if "equilibration" in w]
        assert len(flagged) == 1
        assert "phi = 0.15" in flagged[0]
        assert "measuring the transient" in flagged[0]

    def test_a_permeable_plug_is_left_alone(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        config = self.permeable(quick_steady_config)
        config.sample.porosity = 0.15
        summary = self._summary(config, fake_analog_source, fake_temperature_source)
        assert not any("equilibration" in w for w in summary.warnings)

    def test_an_unrecorded_porosity_is_bounded_rather_than_guessed(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        assert quick_steady_config.sample.porosity_fraction is None
        summary = self._summary(
            quick_steady_config, fake_analog_source, fake_temperature_source
        )
        flagged = [w for w in summary.warnings if "equilibration" in w]
        assert len(flagged) == 1
        assert "porosity_fraction is unrecorded" in flagged[0]
        assert "even at a porosity of 5%" in flagged[0]

    def test_an_unrecorded_porosity_alone_is_not_worth_mentioning(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        """Silence when the bound clears: optional metadata must not nag."""
        config = self.permeable(quick_steady_config)
        assert config.sample.porosity_fraction is None
        summary = self._summary(config, fake_analog_source, fake_temperature_source)
        assert not any("porosity_fraction" in w for w in summary.warnings)

    def test_the_check_can_be_switched_off(
        self, quick_steady_config, fake_analog_source, fake_temperature_source
    ):
        quick_steady_config.sample.porosity = 0.15
        quick_steady_config.run.steady_state.equilibration_factor = None
        summary = self._summary(
            quick_steady_config, fake_analog_source, fake_temperature_source
        )
        assert not any("equilibration" in w for w in summary.warnings)


class TestConsoleFormatting:
    def test_line_uses_the_configured_display_units(self, base_config):
        base_config.run.display_pressure_unit = "bar"
        reading = make_processor(base_config).process(
            index=0, elapsed_s=1.5, voltages=VOLTAGES, temperature=sample()
        )
        line = format_reading_line(reading, base_config)
        assert "bar" in line and "mD" in line and "1.5s" in line

    def test_the_line_shows_the_steady_state_verdict(self, base_config):
        processor = make_processor(base_config)
        settling = processor.process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample()
        )
        steady = processor.process(
            index=1, elapsed_s=0.1, voltages=VOLTAGES, temperature=sample(),
            steady_state=True, steady_state_passes=3,
        )
        assert "settling" in format_reading_line(settling, base_config)
        assert "STEADY" in format_reading_line(steady, base_config)

    def test_a_stale_temperature_is_marked(self, base_config):
        reading = make_processor(base_config).process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES, temperature=sample(21.0, stale=True)
        )
        assert "*" in format_reading_line(reading, base_config)

    def test_header_names_the_display_units(self, base_config):
        header = console_header(base_config)
        assert base_config.run.display_pressure_unit in header
        assert base_config.run.display_permeability_unit in header
