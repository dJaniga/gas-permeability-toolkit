"""Hardware wrappers: calibration maths, DAQ channel setup, serial parsing.

No physical device is involved anywhere in this file. ``nidaqmx`` and
``serial`` are replaced with the fakes from ``conftest.py``.
"""

from __future__ import annotations

import time

import pytest

from gasperm import units
from gasperm.config import (
    FlowmeterConfig,
    PressureChannelConfig,
)
from gasperm.hardware.daq import (
    ChannelSpec,
    DaqError,
    NiDaqAnalogInput,
    build_channel_specs,
    open_analog_input,
)
from gasperm.hardware.flowmeter import FlowChannel
from gasperm.hardware.pressure import PressureChannel
from gasperm.hardware.temperature import (
    SerialTemperatureReader,
    StaticTemperatureSource,
    build_line_parser,
    parse_temperature_line,
    serial_port_exists,
)

ATMOSPHERIC_ATM = 1.0


def pressure_channel(**overrides) -> PressureChannel:
    config = PressureChannelConfig(**overrides)
    return PressureChannel.from_config("inlet", "ai0", config, ATMOSPHERIC_ATM)


# --------------------------------------------------------------------------
# Pressure calibration
# --------------------------------------------------------------------------


class TestPressureCalibration:
    def test_endpoints_in_the_configured_unit(self):
        channel = pressure_channel(volts_min=0.0, volts_max=5.0, value_min=0.0, value_max=1000.0)
        assert channel.volts_to_pressure(0.0) == pytest.approx(0.0)
        assert channel.volts_to_pressure(5.0) == pytest.approx(1000.0)
        assert channel.volts_to_pressure(2.5) == pytest.approx(500.0)

    def test_conversion_to_atm_uses_the_channel_unit(self):
        channel = pressure_channel(unit="kPa", value_max=1000.0)
        # 5 V -> 1000 kPa -> 1000/101.325 atm
        assert channel.volts_to_absolute_atm(5.0) == pytest.approx(units.kpa_to_atm(1000.0))

    def test_the_same_physical_pressure_in_different_units_agrees(self):
        """A rig calibrated in bar and one in kPa must produce the same atm."""
        in_kpa = pressure_channel(unit="kPa", value_min=0.0, value_max=1000.0)
        in_bar = pressure_channel(unit="bar", value_min=0.0, value_max=10.0)
        assert in_kpa.volts_to_absolute_atm(3.7) == pytest.approx(
            in_bar.volts_to_absolute_atm(3.7), rel=1e-12
        )

    def test_psi_calibration(self):
        channel = pressure_channel(unit="psi", value_min=0.0, value_max=145.0377)
        assert channel.volts_to_absolute_atm(5.0) == pytest.approx(units.bar_to_atm(10.0), rel=1e-5)

    def test_gauge_readings_get_atmospheric_added(self):
        gauge = pressure_channel(reading_type="gauge", unit="kPa", value_max=1000.0)
        absolute = pressure_channel(reading_type="absolute", unit="kPa", value_max=1000.0)
        difference = gauge.volts_to_absolute_atm(2.0) - absolute.volts_to_absolute_atm(2.0)
        assert difference == pytest.approx(ATMOSPHERIC_ATM)

    def test_a_gauge_transducer_at_zero_reads_one_atmosphere(self):
        gauge = pressure_channel(reading_type="gauge")
        assert gauge.volts_to_absolute_atm(0.0) == pytest.approx(ATMOSPHERIC_ATM)

    def test_inverse_round_trips_for_both_reading_types(self):
        for reading_type in ("absolute", "gauge"):
            channel = pressure_channel(reading_type=reading_type)
            for volts in (0.0, 1.3, 5.0):
                assert channel.absolute_atm_to_volts(
                    channel.volts_to_absolute_atm(volts)
                ) == pytest.approx(volts)

    def test_voltage_range_is_reported_ascending(self):
        channel = pressure_channel(volts_min=5.0, volts_max=0.0, value_min=1000.0, value_max=0.0)
        assert channel.voltage_range == (0.0, 5.0)

    def test_inverted_calibration_still_maps_correctly(self):
        """Some transducers fall with pressure; the two-point form handles it."""
        channel = pressure_channel(volts_min=0.0, volts_max=5.0, value_min=1000.0, value_max=0.0)
        assert channel.volts_to_pressure(0.0) == pytest.approx(1000.0)
        assert channel.volts_to_pressure(5.0) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Flowmeter calibration
# --------------------------------------------------------------------------


class TestFlowCalibration:
    def test_endpoints(self):
        flow = FlowChannel(FlowmeterConfig(volts_min=0.0, volts_max=10.0, flow_min=0.0, flow_max=500.0))
        assert flow.volts_to_flow(0.0) == pytest.approx(0.0)
        assert flow.volts_to_flow(10.0) == pytest.approx(500.0)
        assert flow.volts_to_flow(5.0) == pytest.approx(250.0)

    def test_sccm_becomes_cm3_per_second(self):
        flow = FlowChannel(FlowmeterConfig(unit="sccm", flow_max=600.0))
        # Full scale 600 sccm = 10 cm^3/s.
        assert flow.volts_to_cm3_s(10.0) == pytest.approx(10.0)

    def test_slpm_meter_agrees_with_an_equivalent_sccm_meter(self):
        in_sccm = FlowChannel(FlowmeterConfig(unit="sccm", flow_max=1000.0))
        in_slpm = FlowChannel(FlowmeterConfig(unit="slpm", flow_max=1.0))
        assert in_sccm.volts_to_cm3_s(6.4) == pytest.approx(in_slpm.volts_to_cm3_s(6.4), rel=1e-12)

    def test_inverse_round_trips(self):
        flow = FlowChannel(FlowmeterConfig())
        assert flow.cm3_s_to_volts(flow.volts_to_cm3_s(3.3)) == pytest.approx(3.3)

    def test_flow_channel_uses_a_ten_volt_range(self):
        assert FlowChannel(FlowmeterConfig()).voltage_range == (0.0, 10.0)


class TestFlowReferenceState:
    """The pairing of reported volume with the pressure it was measured at."""

    def test_standard_basis_uses_the_meters_own_standard_pressure(self):
        flow = FlowChannel(
            FlowmeterConfig(
                reading_basis="standard", standard_pressure=100.0, standard_pressure_unit="kPa"
            )
        )
        reference = flow.reference_pressure_atm(
            inlet_pressure_atm=3.0, outlet_pressure_atm=1.0, atmospheric_atm=1.0
        )
        assert reference == pytest.approx(units.kpa_to_atm(100.0))

    def test_actual_basis_defaults_to_the_outlet_pressure(self):
        flow = FlowChannel(FlowmeterConfig(reading_basis="actual"))
        reference = flow.reference_pressure_atm(
            inlet_pressure_atm=3.0, outlet_pressure_atm=1.4, atmospheric_atm=1.0
        )
        assert reference == pytest.approx(1.4)

    def test_actual_basis_can_reference_the_inlet_or_ambient(self):
        upstream = FlowChannel(
            FlowmeterConfig(reading_basis="actual", actual_pressure_source="inlet")
        )
        ambient = FlowChannel(
            FlowmeterConfig(reading_basis="actual", actual_pressure_source="atmospheric")
        )
        args = dict(inlet_pressure_atm=3.0, outlet_pressure_atm=1.4, atmospheric_atm=0.98)
        assert upstream.reference_pressure_atm(**args) == pytest.approx(3.0)
        assert ambient.reference_pressure_atm(**args) == pytest.approx(0.98)

    def test_reference_temperature_follows_the_basis(self):
        standard = FlowChannel(FlowmeterConfig(reading_basis="standard", standard_temperature_c=0.0))
        actual = FlowChannel(FlowmeterConfig(reading_basis="actual"))
        assert standard.reference_temperature_c(23.5) == 0.0
        assert actual.reference_temperature_c(23.5) == 23.5


# --------------------------------------------------------------------------
# DAQ task construction
# --------------------------------------------------------------------------


class TestDaqTask:
    def test_each_channel_is_added_with_its_own_voltage_range(self, fake_nidaqmx, base_config):
        """The whole point of the per-channel design: 0-5 V and 0-10 V coexist."""
        source = open_analog_input(base_config)
        task = fake_nidaqmx.instances[-1]
        added = {entry["physical_channel"]: entry for entry in task.ai_channels.added}

        assert set(added) == {"Dev1/ai0", "Dev1/ai1", "Dev1/ai2"}
        assert (added["Dev1/ai0"]["min_val"], added["Dev1/ai0"]["max_val"]) == (0.0, 5.0)
        assert (added["Dev1/ai1"]["min_val"], added["Dev1/ai1"]["max_val"]) == (0.0, 5.0)
        assert (added["Dev1/ai2"]["min_val"], added["Dev1/ai2"]["max_val"]) == (0.0, 10.0)
        source.close()

    def test_channels_are_added_one_call_each(self, fake_nidaqmx, base_config):
        open_analog_input(base_config).close()
        assert len(fake_nidaqmx.instances[-1].ai_channels.added) == 3

    def test_only_the_selected_meter_reaches_the_daq(self, fake_nidaqmx, base_config):
        """Both meters are wired; the run selects one and the other is untouched."""
        base_config.run.flowmeter = "high_range"
        open_analog_input(base_config).close()
        names = fake_nidaqmx.instances[-1].channel_names
        assert "ai3" in names  # high_range
        assert "ai2" not in names  # low_range, wired but not selected

    def test_the_default_meter_is_used_when_the_run_does_not_choose(
        self, fake_nidaqmx, base_config
    ):
        assert base_config.run.flowmeter is None
        open_analog_input(base_config).close()
        names = fake_nidaqmx.instances[-1].channel_names
        assert "ai2" in names and "ai3" not in names

    def test_the_selected_meter_sets_the_channel_voltage_range(
        self, fake_nidaqmx, base_config
    ):
        base_config.hardware.flowmeters["high_range"].volts_max = 5.0
        base_config.run.flowmeter = "high_range"
        open_analog_input(base_config).close()
        added = {
            entry["physical_channel"]: entry
            for entry in fake_nidaqmx.instances[-1].ai_channels.added
        }
        assert (added["Dev1/ai3"]["min_val"], added["Dev1/ai3"]["max_val"]) == (0.0, 5.0)

    def test_device_name_prefixes_every_channel(self, fake_nidaqmx, base_config):
        base_config.daq.device_name = "cDAQ1Mod2"
        open_analog_input(base_config).close()
        for entry in fake_nidaqmx.instances[-1].ai_channels.added:
            assert entry["physical_channel"].startswith("cDAQ1Mod2/")

    def test_terminal_config_is_passed_through(self, fake_nidaqmx, base_config):
        base_config.daq.terminal_config = "RSE"
        open_analog_input(base_config).close()
        assert all(
            entry["terminal_config"] == "RSE"
            for entry in fake_nidaqmx.instances[-1].ai_channels.added
        )

    def test_read_maps_values_back_to_channel_names(self, fake_nidaqmx, base_config):
        fake_nidaqmx.voltages = {"ai0": 2.5, "ai1": 0.5, "ai2": 4.0}
        source = open_analog_input(base_config)
        assert source.read() == {"ai0": 2.5, "ai1": 0.5, "ai2": 4.0}
        source.close()

    def test_read_before_open_is_rejected(self, fake_nidaqmx, base_config):
        source = NiDaqAnalogInput("Dev1", build_channel_specs(base_config))
        with pytest.raises(DaqError, match="not open"):
            source.read()

    def test_a_failed_read_is_reported_as_a_daq_error(self, fake_nidaqmx, base_config):
        source = open_analog_input(base_config)
        fake_nidaqmx.read_error = RuntimeError("device removed")
        with pytest.raises(DaqError, match="unplugged"):
            source.read()
        source.close()

    def test_a_configuration_failure_closes_the_task(self, fake_nidaqmx, base_config):
        fake_nidaqmx.configure_error = RuntimeError("no such channel")
        with pytest.raises(DaqError, match="Dev1"):
            open_analog_input(base_config)
        assert fake_nidaqmx.instances[-1].closed

    def test_close_is_idempotent(self, fake_nidaqmx, base_config):
        source = open_analog_input(base_config)
        source.close()
        source.close()

    def test_context_manager_closes_the_task(self, fake_nidaqmx, base_config):
        with NiDaqAnalogInput("Dev1", build_channel_specs(base_config)):
            pass
        assert fake_nidaqmx.instances[-1].closed

    def test_duplicate_channels_are_rejected(self):
        with pytest.raises(ValueError, match="more than once"):
            NiDaqAnalogInput(
                "Dev1", [ChannelSpec("ai0", 0.0, 5.0), ChannelSpec("ai0", 0.0, 10.0)]
            )

    def test_an_inverted_range_is_rejected(self):
        with pytest.raises(ValueError, match="min_volts"):
            ChannelSpec("ai0", 5.0, 0.0)


# --------------------------------------------------------------------------
# Serial temperature parsing
# --------------------------------------------------------------------------


class TestTemperatureParsing:
    def test_the_documented_arduino_format(self):
        parser = build_line_parser("T:{value}")
        assert parse_temperature_line("T:23.4", parser) == pytest.approx(23.4)

    def test_negative_and_exponent_forms(self):
        parser = build_line_parser("T:{value}")
        assert parse_temperature_line("T:-5.25", parser) == pytest.approx(-5.25)
        assert parse_temperature_line("T:2.34e1", parser) == pytest.approx(23.4)

    def test_surrounding_whitespace_and_line_endings(self):
        parser = build_line_parser("T:{value}")
        assert parse_temperature_line("  T:23.4\r\n", parser) == pytest.approx(23.4)

    def test_a_pattern_with_regex_metacharacters_is_matched_literally(self):
        parser = build_line_parser("[temp]={value}C")
        assert parse_temperature_line("[temp]=21.7C", parser) == pytest.approx(21.7)
        assert parse_temperature_line("Xtemp]=21.7C", parser) is None

    def test_null_pattern_takes_the_first_number(self):
        parser = build_line_parser(None)
        assert parse_temperature_line("23.4", parser) == pytest.approx(23.4)
        assert parse_temperature_line("23.4,55.1", parser) == pytest.approx(23.4)

    def test_an_unmatched_line_returns_none_rather_than_raising(self):
        parser = build_line_parser("T:{value}")
        assert parse_temperature_line("Arduino ready", parser) is None
        assert parse_temperature_line("", parser) is None

    def test_probe_units_are_converted_to_celsius(self):
        parser = build_line_parser("T:{value}")
        assert parse_temperature_line("T:68.0", parser, unit="F") == pytest.approx(20.0)
        assert parse_temperature_line("T:293.15", parser, unit="K") == pytest.approx(20.0)

    def test_a_pattern_without_a_value_marker_is_rejected(self):
        with pytest.raises(ValueError, match="value"):
            build_line_parser("T:")


class TestSerialReader:
    def test_reads_and_parses_scripted_lines(self, fake_serial):
        fake_serial.lines = [b"T:21.0\n", b"T:21.5\n"]
        reader = SerialTemperatureReader("COM4", 9600).open()
        try:
            sample = _wait_for_reading(reader)
            assert sample.temperature_c == pytest.approx(21.5)
            assert sample.stale is False
        finally:
            reader.close()

    def test_an_unparseable_line_keeps_the_last_good_value(self, fake_serial):
        fake_serial.lines = [b"T:21.0\n", b"Arduino booting...\n"]
        reader = SerialTemperatureReader("COM4", 9600).open()
        try:
            # The bad line arrives after the good one, so wait for the reader
            # to have actually processed it before asserting.
            _wait_until(lambda: reader.parse_failure_count >= 1)
            assert reader.latest().temperature_c == pytest.approx(21.0)
        finally:
            reader.close()

    def test_a_serial_read_failure_does_not_raise_to_the_consumer(self, fake_serial):
        fake_serial.lines = [b"T:21.0\n"]
        fake_serial.read_error = OSError("probe unplugged")
        reader = SerialTemperatureReader("COM4", 9600).open()
        try:
            # Likewise: the failure happens on the readline *after* the good
            # line, so poll for the warning rather than racing it.
            _wait_until(lambda: any("read failed" in w for w in reader.warnings))
            assert reader.latest().temperature_c == pytest.approx(21.0)
        finally:
            reader.close()

    def test_a_stale_value_is_flagged(self, fake_serial):
        fake_serial.lines = [b"T:21.0\n"]
        reader = SerialTemperatureReader("COM4", 9600, stale_after_s=0.0).open()
        try:
            _wait_for_reading(reader)
            time.sleep(0.01)
            assert reader.latest().stale is True
        finally:
            reader.close()

    def test_an_unopenable_port_raises_with_the_available_ports(self, fake_serial):
        import sys

        fake_serial.open_error = OSError("access denied")
        sys.modules["serial.tools.list_ports"].available = ["COM7"]
        reader = SerialTemperatureReader("COM4", 9600)
        with pytest.raises(OSError, match="COM7"):
            reader.open()

    def test_close_is_idempotent(self, fake_serial):
        reader = SerialTemperatureReader("COM4", 9600).open()
        reader.close()
        reader.close()

    def test_no_reading_yet_reports_none_rather_than_blocking(self, fake_serial):
        reader = SerialTemperatureReader("COM4", 9600).open()
        try:
            assert reader.latest().temperature_c is None
        finally:
            reader.close()


class TestSlowSensor:
    """A DS18B20 converts in 750 ms while the DAQ samples every 100 ms."""

    def test_a_value_is_held_between_conversions(self, fake_serial):
        """The gap between conversions must not read as 'unavailable'."""
        fake_serial.lines = [b"T:21.0\n"]
        reader = SerialTemperatureReader("COM4", 9600).open()
        try:
            _wait_for_reading(reader)
            # No further lines arrive, as between two conversions.
            for _ in range(5):
                sample = reader.latest()
                assert sample.temperature_c == pytest.approx(21.0)
                assert sample.stale is False
        finally:
            reader.close()

    def test_the_age_of_a_held_value_is_reported(self, fake_serial):
        fake_serial.lines = [b"T:21.0\n"]
        reader = SerialTemperatureReader("COM4", 9600).open()
        try:
            _wait_for_reading(reader)
            time.sleep(0.05)
            sample = reader.latest()
            assert sample.age_s is not None and sample.age_s >= 0.04
        finally:
            reader.close()

    def test_waiting_for_the_first_reading_succeeds(self, fake_serial):
        fake_serial.lines = [b"T:21.0\n"]
        reader = SerialTemperatureReader("COM4", 9600).open()
        try:
            assert reader.wait_for_first_reading(2.0) is True
            assert reader.latest().temperature_c == pytest.approx(21.0)
        finally:
            reader.close()

    def test_waiting_times_out_when_the_probe_never_speaks(self, fake_serial):
        """A port that opens but says nothing -- wrong baud, or no sketch."""
        fake_serial.lines = []
        reader = SerialTemperatureReader("COM4", 9600).open()
        try:
            assert reader.wait_for_first_reading(0.2) is False
            assert reader.latest().temperature_c is None
        finally:
            reader.close()

    def test_an_unparseable_line_does_not_satisfy_the_wait(self, fake_serial):
        fake_serial.lines = [b"Arduino ready\n"]
        reader = SerialTemperatureReader("COM4", 9600).open()
        try:
            assert reader.wait_for_first_reading(0.2) is False
        finally:
            reader.close()

    def test_a_static_source_never_waits(self):
        assert StaticTemperatureSource(20.0).wait_for_first_reading(99.0) is True


class TestImplausibleReadings:
    """The DS18B20 sentinels parse as ordinary numbers and must not pass."""

    @pytest.mark.parametrize("sentinel", [b"T:-127.00\n", b"T:85.00\n"])
    def test_a_sentinel_is_discarded_and_the_last_value_kept(self, fake_serial, sentinel):
        fake_serial.lines = [b"T:21.0\n", sentinel]
        reader = SerialTemperatureReader("COM4", 9600).open()
        try:
            _wait_until(lambda: reader.implausible_count >= 1)
            assert reader.latest().temperature_c == pytest.approx(21.0)
        finally:
            reader.close()

    def test_the_rejection_is_reported(self, fake_serial):
        fake_serial.lines = [b"T:21.0\n", b"T:-127.00\n"]
        reader = SerialTemperatureReader("COM4", 9600).open()
        try:
            _wait_until(lambda: reader.implausible_count >= 1)
            assert any("Implausible temperature" in w for w in reader.warnings)
        finally:
            reader.close()

    def test_a_sentinel_first_does_not_satisfy_the_wait(self, fake_serial):
        """Otherwise a dead sensor would look like a working one at startup."""
        fake_serial.lines = [b"T:-127.00\n"]
        reader = SerialTemperatureReader("COM4", 9600).open()
        try:
            assert reader.wait_for_first_reading(0.2) is False
        finally:
            reader.close()

    def test_the_band_is_configurable(self, fake_serial):
        fake_serial.lines = [b"T:85.00\n"]
        reader = SerialTemperatureReader(
            "COM4", 9600, plausible_min_c=-40.0, plausible_max_c=125.0
        ).open()
        try:
            assert reader.wait_for_first_reading(2.0) is True
            assert reader.latest().temperature_c == pytest.approx(85.0)
        finally:
            reader.close()

    def test_an_ordinary_reading_is_untouched(self, fake_serial):
        fake_serial.lines = [b"T:21.0\n"]
        reader = SerialTemperatureReader("COM4", 9600).open()
        try:
            _wait_for_reading(reader)
            assert reader.implausible_count == 0
        finally:
            reader.close()


class TestPortDiscovery:
    def test_a_present_port_is_found(self, fake_serial):
        import sys

        sys.modules["serial.tools.list_ports"].available = ["COM3", "COM4"]
        assert serial_port_exists("COM4") is True

    def test_an_absent_port_is_reported_absent(self, fake_serial):
        import sys

        sys.modules["serial.tools.list_ports"].available = ["COM3"]
        assert serial_port_exists("COM4") is False


class TestStaticTemperatureSource:
    def test_returns_the_constant_and_flags_it_stale(self):
        source = StaticTemperatureSource(19.5, note="no probe")
        sample = source.latest()
        assert sample.temperature_c == 19.5
        assert sample.stale is True
        source.close()


def _wait_until(predicate, timeout_s: float = 2.0, *, what: str = "condition"):
    """Poll ``predicate`` until it holds.

    The reader runs on its own thread, so any assertion about what it has
    *already* processed has to wait for it rather than assume an ordering.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError(f"the reader thread never satisfied the {what} within {timeout_s}s")


def _wait_for_reading(reader, timeout_s: float = 2.0):
    """Poll until the background thread has parsed at least one line."""
    _wait_until(
        lambda: reader.latest().temperature_c is not None,
        timeout_s,
        what="first temperature reading",
    )
    return reader.latest()
