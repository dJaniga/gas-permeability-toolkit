"""Config schema, YAML round-tripping, and startup validation."""

from __future__ import annotations

import pytest
import yaml

from gasperm import units
from gasperm.config import (
    ConfigError,
    DaqConfig,
    FlowmeterConfig,
    GaspermConfig,
    LinearCalibration,
    PressureChannelConfig,
    RunConfig,
    SampleConfig,
    config_to_dict,
    load_config,
    render_config_yaml,
    save_config,
    validate_for_collect,
)


class TestDefaults:
    def test_shipped_defaults_match_the_documented_wiring(self, base_config: GaspermConfig):
        assert base_config.daq.device_name == "Dev1"
        assert base_config.daq.inlet_pressure_channel == "ai0"
        assert base_config.daq.outlet_pressure_channel == "ai1"
        assert base_config.flowmeter.channel == "ai2"
        assert base_config.temperature.port == "COM4"
        assert base_config.temperature.baud_rate == 9600
        assert base_config.gas.name == "Nitrogen"

    def test_pressure_channels_default_to_0_to_5_volts(self, base_config: GaspermConfig):
        for channel in (
            base_config.pressure_calibration.inlet,
            base_config.pressure_calibration.outlet,
        ):
            assert (channel.volts_min, channel.volts_max) == (0.0, 5.0)

    def test_flow_channel_defaults_to_0_to_10_volts(self, base_config: GaspermConfig):
        assert (base_config.flowmeter.volts_min, base_config.flowmeter.volts_max) == (0.0, 10.0)


class TestValidation:
    def test_zero_width_voltage_span_is_rejected(self):
        with pytest.raises(ValueError, match="volts_min and volts_max must differ"):
            PressureChannelConfig(volts_min=5.0, volts_max=5.0)

    def test_zero_width_value_span_is_rejected(self):
        with pytest.raises(ValueError, match="value_min and value_max must differ"):
            PressureChannelConfig(value_min=100.0, value_max=100.0)

    @pytest.mark.parametrize("field", ["length_cm", "diameter_cm"])
    def test_non_positive_geometry_is_rejected(self, field: str):
        with pytest.raises(ValueError):
            SampleConfig(**{field: 0.0})

    def test_negative_geometry_is_rejected(self):
        with pytest.raises(ValueError):
            SampleConfig(length_cm=-1.0)

    def test_unknown_pressure_unit_is_rejected(self):
        with pytest.raises(ValueError, match="torr"):
            PressureChannelConfig(unit="torr")

    def test_unknown_flow_unit_is_rejected(self):
        with pytest.raises(ValueError, match="gallons"):
            FlowmeterConfig(unit="gallons/hour")

    def test_unknown_permeability_display_unit_is_rejected(self):
        with pytest.raises(ValueError, match="furlongs"):
            RunConfig(display_permeability_unit="furlongs")

    def test_identical_pressure_channels_are_rejected(self):
        with pytest.raises(ValueError, match="must differ"):
            DaqConfig(inlet_pressure_channel="ai0", outlet_pressure_channel="ai0")

    def test_flowmeter_cannot_share_a_pressure_channel(self):
        with pytest.raises(ValueError, match="already assigned"):
            GaspermConfig(flowmeter=FlowmeterConfig(channel="ai0"))

    def test_non_analog_channel_names_are_rejected(self):
        with pytest.raises(ValueError, match="analog-input channel"):
            DaqConfig(inlet_pressure_channel="port0")

    def test_device_qualified_channel_names_are_accepted_and_stripped(self):
        assert DaqConfig(inlet_pressure_channel="Dev1/ai0").inlet_pressure_channel == "ai0"

    def test_unknown_fields_are_rejected(self):
        """A typo'd key must fail loudly, not be silently ignored."""
        with pytest.raises(ValueError):
            SampleConfig(lenght_cm=5.0)

    def test_ai3_is_a_valid_flowmeter_channel(self):
        assert GaspermConfig(flowmeter=FlowmeterConfig(channel="ai3")).flowmeter.channel == "ai3"


class TestIndependentUnits:
    def test_inlet_and_outlet_units_are_independent(self):
        config = GaspermConfig()
        config.pressure_calibration.inlet.unit = "bar"
        config.pressure_calibration.outlet.unit = "kPa"
        assert config.pressure_calibration.inlet.unit == "bar"
        assert config.pressure_calibration.outlet.unit == "kPa"

    def test_confining_pressure_has_its_own_unit(self):
        sample = SampleConfig(confining_pressure=20.0, confining_pressure_unit="MPa")
        assert sample.confining_pressure_atm == pytest.approx(units.mpa_to_atm(20.0))

    def test_confining_pressure_is_optional(self):
        assert SampleConfig().confining_pressure_atm is None

    def test_display_units_are_independent_of_calibration_units(self):
        config = GaspermConfig()
        config.pressure_calibration.inlet.unit = "psi"
        config.run.display_pressure_unit = "MPa"
        assert config.pressure_calibration.inlet.unit == "psi"
        assert config.run.display_pressure_unit == "MPa"

    @pytest.mark.parametrize("unit", sorted(units.SUPPORTED_PRESSURE_UNITS))
    def test_every_supported_unit_is_accepted_everywhere_a_pressure_lives(self, unit: str):
        config = GaspermConfig(
            pressure_calibration={"inlet": {"unit": unit}, "outlet": {"unit": unit}},
            sample={"confining_pressure": 1.0, "confining_pressure_unit": unit},
            run={
                "atmospheric_pressure": units.from_atm(1.0, unit),
                "atmospheric_pressure_unit": unit,
                "display_pressure_unit": unit,
            },
            flowmeter={"standard_pressure": units.from_atm(1.0, unit), "standard_pressure_unit": unit},
        )
        assert config.run.atmospheric_pressure_atm == pytest.approx(1.0, rel=1e-12)
        assert config.flowmeter.standard_pressure_atm == pytest.approx(1.0, rel=1e-12)


class TestLinearCalibration:
    def test_endpoints_map_exactly(self):
        calibration = LinearCalibration(volts_min=1.0, volts_max=5.0, value_min=0.0, value_max=200.0)
        assert calibration.apply(1.0) == pytest.approx(0.0)
        assert calibration.apply(5.0) == pytest.approx(200.0)

    def test_midpoint_is_linear(self):
        calibration = LinearCalibration(volts_min=1.0, volts_max=5.0, value_min=0.0, value_max=200.0)
        assert calibration.apply(3.0) == pytest.approx(100.0)

    def test_extrapolates_below_the_low_endpoint(self):
        """Transducer noise around zero must not be clamped away silently."""
        calibration = LinearCalibration(volts_min=0.0, volts_max=5.0, value_min=0.0, value_max=100.0)
        assert calibration.apply(-0.01) == pytest.approx(-0.2)

    def test_inverse_round_trips(self):
        calibration = LinearCalibration(volts_min=0.5, volts_max=4.5, value_min=-10.0, value_max=90.0)
        for volts in (0.5, 1.7, 4.5):
            assert calibration.invert(calibration.apply(volts)) == pytest.approx(volts)

    def test_offset_calibration(self):
        """A 1-5 V transducer: 1 V is zero pressure, not 0 V."""
        calibration = LinearCalibration(volts_min=1.0, volts_max=5.0, value_min=0.0, value_max=1000.0)
        assert calibration.apply(1.0) == pytest.approx(0.0)
        assert calibration.apply(2.0) == pytest.approx(250.0)


class TestFlowmeterAliases:
    def test_flow_min_max_aliases_populate_the_calibration(self):
        flow = FlowmeterConfig.model_validate({"flow_min": 0.0, "flow_max": 250.0})
        assert flow.value_max == 250.0
        assert flow.flow_max == 250.0

    def test_yaml_round_trip_preserves_the_alias_spelling(self):
        rendered = config_to_dict(GaspermConfig())
        assert "flow_max" in rendered["flowmeter"]
        assert GaspermConfig.model_validate(rendered).flowmeter.flow_max == 500.0


class TestOutletReference:
    def test_atmospheric_is_the_default(self):
        assert RunConfig().outlet_pressure_reference == "atmospheric"
        assert RunConfig().fixed_outlet_pressure_atm is None

    def test_measured_is_accepted(self):
        assert RunConfig(outlet_pressure_reference="measured").fixed_outlet_pressure_atm is None

    def test_a_fixed_value_is_converted_from_its_own_unit(self):
        run = RunConfig(outlet_pressure_reference=2.0, outlet_pressure_reference_unit="bar")
        assert run.fixed_outlet_pressure_atm == pytest.approx(units.bar_to_atm(2.0))

    def test_a_nonsense_string_is_rejected(self):
        with pytest.raises(ValueError):
            RunConfig(outlet_pressure_reference="ambientish")


class TestYamlIo:
    def test_rendered_template_is_valid_yaml_and_reloads(self, tmp_path, base_config):
        path = tmp_path / "gasperm_config.yaml"
        save_config(base_config, path)
        reloaded = load_config(path)
        assert config_to_dict(reloaded) == config_to_dict(base_config)

    def test_rendered_template_carries_explanatory_comments(self, base_config):
        text = render_config_yaml(base_config)
        assert "NI-DAQmx device name" in text
        assert "ai2 or ai3" in text
        assert "CGS-Darcy" in text

    def test_parse_pattern_survives_the_yaml_round_trip(self, tmp_path, base_config):
        """'T:{value}' contains YAML-significant characters."""
        path = tmp_path / "c.yaml"
        save_config(base_config, path)
        assert load_config(path).temperature.parse_pattern == "T:{value}"

    def test_existing_file_is_not_overwritten_without_force(self, tmp_path, base_config):
        path = tmp_path / "c.yaml"
        save_config(base_config, path)
        with pytest.raises(ConfigError, match="--force"):
            save_config(base_config, path)
        save_config(base_config, path, overwrite=True)

    def test_missing_file_suggests_init(self, tmp_path):
        with pytest.raises(ConfigError, match="gasperm init"):
            load_config(tmp_path / "nope.yaml")

    def test_malformed_yaml_names_the_file(self, tmp_path):
        path = tmp_path / "broken.yaml"
        path.write_text("daq: [unclosed\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="not valid YAML"):
            load_config(path)

    def test_empty_file_is_rejected(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ConfigError, match="empty"):
            load_config(path)

    def test_validation_error_names_the_offending_field(self, tmp_path, base_config):
        data = config_to_dict(base_config)
        data["sample"]["length_cm"] = -3.0
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        with pytest.raises(ConfigError, match="sample.length_cm"):
            load_config(path)

    def test_shipped_example_config_loads(self):
        from pathlib import Path

        example = Path(__file__).resolve().parents[1] / "examples" / "sample_config.yaml"
        if not example.is_file():  # pragma: no cover - only when examples/ is pruned
            pytest.skip("examples/sample_config.yaml not present")
        assert load_config(example).flowmeter.channel == "ai2"


class TestValidateForCollect:
    def test_a_healthy_config_passes(self, base_config, fake_serial, monkeypatch):
        import sys

        sys.modules["serial.tools.list_ports"].available = ["COM4"]
        assert validate_for_collect(base_config) == []

    def test_a_missing_required_port_is_fatal(self, base_config, fake_serial):
        import sys

        sys.modules["serial.tools.list_ports"].available = ["COM7"]
        with pytest.raises(ConfigError, match="COM4"):
            validate_for_collect(base_config)

    def test_a_missing_optional_port_only_warns(self, base_config, fake_serial):
        import sys

        sys.modules["serial.tools.list_ports"].available = ["COM7"]
        base_config.temperature.required = False
        warnings = validate_for_collect(base_config)
        assert any("COM4" in w for w in warnings)

    def test_an_unknown_gas_is_fatal(self, base_config, fake_serial):
        pytest.importorskip("CoolProp")
        base_config.gas.name = "Unobtainium"
        with pytest.raises(ConfigError, match="Unobtainium"):
            validate_for_collect(base_config)

    def test_implausible_geometry_warns(self, base_config, fake_serial):
        base_config.sample.length_cm = 500.0
        warnings = validate_for_collect(base_config)
        assert any("unusually long" in w for w in warnings)

    def test_a_sub_atmospheric_full_scale_warns_about_the_unit(self, base_config, fake_serial):
        # A transducer configured in Pa instead of kPa reads 1000 Pa full scale.
        base_config.pressure_calibration.inlet.unit = "Pa"
        warnings = validate_for_collect(base_config)
        assert any("below atmospheric" in w for w in warnings)
