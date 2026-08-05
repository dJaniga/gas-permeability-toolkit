"""Config schema, the three-file split, YAML round-tripping, startup validation."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from gasperm import units
from gasperm.config import (
    HARDWARE_FILENAME,
    RUN_FILENAME,
    SAMPLE_FILENAME,
    ConfigError,
    ConfigPaths,
    DaqConfig,
    FlowmeterConfig,
    GasConfig,
    GaspermConfig,
    HardwareConfig,
    LinearCalibration,
    PressureChannelConfig,
    RunConfig,
    SampleConfig,
    SteadyStateConfig,
    config_to_dict,
    experiment_metadata,
    load_config,
    load_run_config,
    load_sample_config,
    render_hardware_yaml,
    render_run_yaml,
    render_sample_yaml,
    save_config,
    validate_for_collect,
)


class TestDefaults:
    def test_shipped_defaults_match_the_documented_wiring(self, base_config: GaspermConfig):
        assert base_config.hardware.daq.device_name == "Dev1"
        assert base_config.hardware.daq.inlet_pressure_channel == "ai0"
        assert base_config.hardware.daq.outlet_pressure_channel == "ai1"
        assert base_config.hardware.flowmeters["low_range"].channel == "ai2"
        assert base_config.hardware.flowmeters["high_range"].channel == "ai3"
        assert base_config.hardware.temperature.port == "COM4"
        assert base_config.run.gas.name == "Nitrogen"

    def test_pressure_channels_default_to_0_to_5_volts(self, base_config: GaspermConfig):
        calibration = base_config.hardware.pressure_calibration
        for channel in (calibration.inlet, calibration.outlet):
            assert (channel.volts_min, channel.volts_max) == (0.0, 5.0)

    def test_flow_channel_defaults_to_0_to_10_volts(self, base_config: GaspermConfig):
        flow = base_config.hardware.flowmeters["low_range"]
        assert (flow.volts_min, flow.volts_max) == (0.0, 10.0)

    def test_steady_state_and_uncertainty_are_on_by_default(self, base_config):
        assert base_config.run.steady_state.enabled
        assert base_config.run.uncertainty.enabled


class TestSplitConcerns:
    """Each file owns exactly one concern."""

    def test_the_rig_lives_in_hardware(self):
        fields = set(HardwareConfig.model_fields)
        assert {"daq", "pressure_calibration", "flowmeters", "temperature"} <= fields
        assert "gas" not in fields and "confining_pressure" not in fields

    def test_the_plug_lives_in_sample_without_test_conditions(self):
        fields = set(SampleConfig.model_fields)
        assert {"id", "lithology", "length", "diameter", "porosity_fraction"} <= fields
        # The same plug is measured at several confining pressures and gases,
        # so neither belongs to the sample.
        assert "confining_pressure" not in fields
        assert "gas" not in fields

    def test_the_experiment_lives_in_run(self):
        fields = set(RunConfig.model_fields)
        assert {"operator", "gas", "confining_pressure", "steady_state"} <= fields
        assert "daq" not in fields

    def test_shorthand_accessors_reach_through(self, base_config):
        assert base_config.daq is base_config.hardware.daq
        assert base_config.flowmeter is base_config.hardware.flowmeters[
            base_config.hardware.default_flowmeter
        ]
        assert base_config.gas is base_config.run.gas
        assert base_config.temperature is base_config.hardware.temperature


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
            HardwareConfig(
                flowmeters={"m": FlowmeterConfig(channel="ai0")},
                default_flowmeter="m",
            )

    def test_unknown_fields_are_rejected(self):
        with pytest.raises(ValueError):
            SampleConfig(lenght_cm=5.0)

    def test_bulk_density_above_grain_density_is_rejected(self):
        with pytest.raises(ValueError, match="impossible"):
            SampleConfig(grain_density_g_cm3=2.65, bulk_density_g_cm3=2.80)

    def test_stop_when_steady_needs_detection_enabled(self):
        with pytest.raises(ValueError, match="would never be detected"):
            RunConfig(stop_when_steady=True, steady_state=SteadyStateConfig(enabled=False))

    def test_steady_state_needs_at_least_one_signal(self):
        with pytest.raises(ValueError, match="nothing would be tested"):
            SteadyStateConfig(signals=[])

    def test_fixed_gas_source_needs_a_viscosity(self):
        with pytest.raises(ValueError, match="fixed_viscosity_cp"):
            GasConfig(properties_source="fixed")


class TestFlowmeterSelection:
    """Several meters are wired once; each run picks one by name."""

    def test_the_rig_ships_with_both_documented_meters(self, base_config):
        assert set(base_config.hardware.flowmeters) == {"low_range", "high_range"}

    def test_selecting_a_meter_is_a_run_decision_not_a_rig_change(self):
        fields = set(RunConfig.model_fields)
        assert "flowmeter" in fields
        # The meter definitions stay in hardware.yaml; only the choice moves.
        assert "flowmeters" not in fields

    def test_the_run_selects_which_meter_is_active(self, base_config):
        assert base_config.flowmeter_name == "low_range"
        base_config.run.flowmeter = "high_range"
        assert base_config.flowmeter_name == "high_range"
        assert base_config.flowmeter.channel == "ai3"

    def test_an_unset_selection_falls_back_to_the_rig_default(self, base_config):
        base_config.hardware.default_flowmeter = "high_range"
        assert base_config.run.flowmeter is None
        assert base_config.flowmeter_name == "high_range"

    def test_a_single_meter_needs_no_default(self):
        config = GaspermConfig(
            hardware={
                "flowmeters": {"only": {"channel": "ai2"}},
                "default_flowmeter": None,
            }
        )
        assert config.flowmeter_name == "only"

    def test_an_unknown_meter_name_is_rejected_with_the_available_ones(self):
        with pytest.raises(ValueError, match="high_range, low_range"):
            GaspermConfig(run={"flowmeter": "middle_range"})

    def test_resolution_names_the_available_meters(self, base_config):
        with pytest.raises(ValueError, match="high_range, low_range"):
            base_config.hardware.resolve_flowmeter("middle_range")

    def test_an_unknown_name_is_caught_at_load(self, tmp_path, base_config):
        save_config(base_config, tmp_path)
        data = base_config.run.model_dump(mode="json")
        data["flowmeter"] = "nonexistent"
        (tmp_path / RUN_FILENAME).write_text(yaml.safe_dump(data), encoding="utf-8")
        with pytest.raises(ConfigError, match="nonexistent"):
            load_config(tmp_path)

    def test_two_meters_cannot_share_an_analog_input(self):
        with pytest.raises(ValueError, match="cannot share one analog input"):
            HardwareConfig(
                flowmeters={
                    "a": FlowmeterConfig(channel="ai2"),
                    "b": FlowmeterConfig(channel="ai2"),
                },
                default_flowmeter="a",
            )

    def test_an_empty_meter_set_is_rejected(self):
        with pytest.raises(ValueError, match="define at least one meter"):
            HardwareConfig(flowmeters={}, default_flowmeter=None)

    def test_a_bad_default_is_rejected(self):
        with pytest.raises(ValueError, match="not a defined meter"):
            HardwareConfig(default_flowmeter="middle_range")

    def test_several_meters_need_a_named_default(self):
        with pytest.raises(ValueError, match="name the one to use"):
            HardwareConfig(default_flowmeter=None)

    def test_each_meter_keeps_its_own_range_and_specification(self, base_config):
        base_config.hardware.flowmeters["low_range"].value_max = 100.0
        base_config.hardware.flowmeters["high_range"].value_max = 5000.0
        assert base_config.flowmeter.flow_max == 100.0
        base_config.run.flowmeter = "high_range"
        assert base_config.flowmeter.flow_max == 5000.0

    def test_the_active_meter_is_recorded_in_the_metadata(self, base_config):
        base_config.run.flowmeter = "high_range"
        metadata = experiment_metadata(base_config)
        assert metadata.flowmeter == "high_range"
        assert metadata.flowmeter_channel == "ai3"
        assert "sccm" in metadata.flowmeter_range

    def test_a_legacy_singular_flowmeter_section_is_rejected(self, tmp_path, base_config):
        save_config(base_config, tmp_path)
        data = base_config.hardware.model_dump(mode="json", by_alias=True)
        data["flowmeter"] = data.pop("flowmeters")["low_range"]
        (tmp_path / HARDWARE_FILENAME).write_text(yaml.safe_dump(data), encoding="utf-8")
        with pytest.raises(ConfigError, match="single 'flowmeter:' section"):
            load_config(tmp_path)

    def test_the_run_template_lists_the_available_meters(self, base_config):
        run_yaml = render_run_yaml(base_config)
        assert "low_range" in run_yaml and "high_range" in run_yaml


class TestSampleDimensions:
    """Plug geometry carries its own unit, like every pressure field does."""

    def test_the_default_plug_is_one_and_a_half_inches_in_mm(self, base_config):
        sample = base_config.sample
        assert sample.dimension_unit == "mm"
        assert sample.diameter == 38.1
        assert sample.length == 50.0

    def test_the_default_diameter_is_exactly_1_5_inches(self, base_config):
        # 25.4 mm/in exactly, so 1.5 in is 38.1 mm with nothing left over.
        assert base_config.sample.diameter_cm == pytest.approx(2.54 * 1.5)
        assert units.length_from_cm(
            base_config.sample.diameter_cm, "in"
        ) == pytest.approx(1.5)

    def test_dimensions_convert_to_cm_for_the_physics(self, base_config):
        assert base_config.sample.length_cm == pytest.approx(5.0)
        assert base_config.sample.diameter_cm == pytest.approx(3.81)
        assert base_config.geometry().area_cm2 == pytest.approx(11.4009, abs=1e-4)

    def test_uncertainties_convert_too(self, base_config):
        base_config.sample.length_uncertainty = 0.1
        base_config.sample.diameter_uncertainty = 0.2
        geometry = base_config.geometry()
        assert geometry.length_uncertainty_cm == pytest.approx(0.01)
        assert geometry.diameter_uncertainty_cm == pytest.approx(0.02)

    @pytest.mark.parametrize(
        "unit,length,diameter",
        [("mm", 50.0, 38.1), ("cm", 5.0, 3.81), ("in", 5.0 / 2.54, 1.5)],
    )
    def test_the_same_plug_in_any_unit_gives_the_same_physics(
        self, unit: str, length: float, diameter: float
    ):
        sample = SampleConfig(dimension_unit=unit, length=length, diameter=diameter)
        assert sample.length_cm == pytest.approx(5.0)
        assert sample.diameter_cm == pytest.approx(3.81)
        assert sample.geometry().area_cm2 == pytest.approx(11.4009, abs=1e-4)

    def test_an_unknown_dimension_unit_is_rejected(self):
        with pytest.raises(ValueError, match="furlong"):
            SampleConfig(dimension_unit="furlong")

    @pytest.mark.parametrize("unit", sorted(units.SUPPORTED_LENGTH_UNITS))
    def test_every_supported_length_unit_is_accepted(self, unit: str):
        assert SampleConfig(dimension_unit=unit).dimension_unit == unit

    def test_the_unit_survives_the_yaml_round_trip(self, tmp_path, base_config):
        base_config.sample.dimension_unit = "in"
        base_config.sample.diameter = 1.5
        save_config(base_config, tmp_path)
        reloaded = load_config(tmp_path)
        assert reloaded.sample.dimension_unit == "in"
        assert reloaded.sample.diameter_cm == pytest.approx(3.81)

    def test_the_template_names_the_unit_on_each_dimension(self, base_config):
        rendered = render_sample_yaml(base_config)
        assert "dimension_unit: mm" in rendered
        assert "1.5 in" in rendered


class TestSampleMetadata:
    def test_petrophysical_fields_round_trip(self):
        sample = SampleConfig(
            id="core-042",
            lithology="fine-grained quartz arenite",
            formation="Rotliegend",
            well="A-12",
            depth=2145.5,
            depth_unit="m",
            porosity_fraction=0.18,
            porosity_method="helium pycnometry",
            grain_density_g_cm3=2.65,
            bulk_density_g_cm3=2.17,
            prepared_by="DJ",
            prepared_on=date(2026, 7, 15),
        )
        restored = SampleConfig.model_validate(sample.model_dump(mode="json"))
        assert restored.lithology == "fine-grained quartz arenite"
        assert restored.prepared_on == date(2026, 7, 15)

    def test_porosity_from_densities_cross_checks_the_stated_value(self):
        sample = SampleConfig(
            porosity_fraction=0.18, grain_density_g_cm3=2.65, bulk_density_g_cm3=2.17
        )
        assert sample.porosity_from_densities == pytest.approx(0.1811, abs=1e-3)

    def test_porosity_cross_check_needs_both_densities(self):
        assert SampleConfig(grain_density_g_cm3=2.65).porosity_from_densities is None

    def test_geometry_carries_the_caliper_uncertainties(self):
        geometry = SampleConfig(
            dimension_unit="cm", length=5.0, diameter=2.5,
            length_uncertainty=0.02, diameter_uncertainty=0.01,
        ).geometry()
        assert geometry.relative_length_uncertainty == pytest.approx(0.004)
        assert geometry.relative_area_uncertainty == pytest.approx(2.0 * 0.01 / 2.5)

    def test_experiment_metadata_flattens_both_files(self, base_config):
        base_config.sample.lithology = "sandstone"
        base_config.run.operator = "Damian"
        base_config.run.confining_pressure = 20.0
        metadata = experiment_metadata(base_config)
        assert metadata.operator == "Damian"
        assert metadata.lithology == "sandstone"
        assert metadata.confining_pressure == 20.0
        assert metadata.gas_name == "Nitrogen"
        assert metadata.sample_id == base_config.sample.id


class TestIndependentUnits:
    def test_inlet_and_outlet_units_are_independent(self, base_config):
        base_config.hardware.pressure_calibration.inlet.unit = "bar"
        base_config.hardware.pressure_calibration.outlet.unit = "kPa"
        assert base_config.hardware.pressure_calibration.inlet.unit == "bar"
        assert base_config.hardware.pressure_calibration.outlet.unit == "kPa"

    def test_confining_pressure_has_its_own_unit(self):
        run = RunConfig(confining_pressure=20.0, confining_pressure_unit="MPa")
        assert run.confining_pressure_atm == pytest.approx(units.mpa_to_atm(20.0))

    def test_confining_pressure_is_optional(self):
        assert RunConfig().confining_pressure_atm is None

    @pytest.mark.parametrize("unit", sorted(units.SUPPORTED_PRESSURE_UNITS))
    def test_every_supported_unit_is_accepted_wherever_a_pressure_lives(self, unit: str):
        config = GaspermConfig(
            hardware={
                "pressure_calibration": {"inlet": {"unit": unit}, "outlet": {"unit": unit}},
                "flowmeters": {
                    "m": {
                        "standard_pressure": units.from_atm(1.0, unit),
                        "standard_pressure_unit": unit,
                    }
                },
                "default_flowmeter": "m",
            },
            run={
                "confining_pressure": 1.0,
                "confining_pressure_unit": unit,
                "atmospheric_pressure": units.from_atm(1.0, unit),
                "atmospheric_pressure_unit": unit,
                "display_pressure_unit": unit,
            },
        )
        assert config.run.atmospheric_pressure_atm == pytest.approx(1.0, rel=1e-12)
        assert config.flowmeter.standard_pressure_atm == pytest.approx(1.0, rel=1e-12)


class TestLinearCalibration:
    def test_endpoints_map_exactly(self):
        calibration = LinearCalibration(volts_min=1.0, volts_max=5.0, value_min=0.0, value_max=200.0)
        assert calibration.apply(1.0) == pytest.approx(0.0)
        assert calibration.apply(5.0) == pytest.approx(200.0)
        assert calibration.apply(3.0) == pytest.approx(100.0)

    def test_extrapolates_below_the_low_endpoint(self):
        calibration = LinearCalibration(volts_min=0.0, volts_max=5.0, value_min=0.0, value_max=100.0)
        assert calibration.apply(-0.01) == pytest.approx(-0.2)

    def test_inverse_round_trips(self):
        calibration = LinearCalibration(volts_min=0.5, volts_max=4.5, value_min=-10.0, value_max=90.0)
        for volts in (0.5, 1.7, 4.5):
            assert calibration.invert(calibration.apply(volts)) == pytest.approx(volts)

    def test_full_scale_is_the_span_magnitude(self):
        assert LinearCalibration(value_min=0.0, value_max=1000.0).full_scale == 1000.0
        assert LinearCalibration(value_min=1000.0, value_max=0.0).full_scale == 1000.0


class TestDownstreamPressure:
    """P2 is the transducer by default, or a value the operator supplies."""

    def test_measured_is_the_default(self):
        run = RunConfig()
        assert run.downstream_pressure == "measured"
        assert run.downstream_is_measured
        assert run.fixed_downstream_pressure_atm is None

    def test_a_supplied_value_converts_from_its_own_unit(self):
        run = RunConfig(downstream_pressure=1.0, downstream_pressure_unit="bar")
        assert run.fixed_downstream_pressure_atm == pytest.approx(units.bar_to_atm(1.0))
        assert not run.downstream_is_measured

    def test_a_nonsense_keyword_is_rejected(self):
        with pytest.raises(ValueError):
            RunConfig(downstream_pressure="ambient")

    @pytest.mark.parametrize("bad", [0.0, -101.325])
    def test_a_non_positive_value_is_rejected_at_load(self, bad: float):
        """A mistyped zero would otherwise make every reading unusable."""
        with pytest.raises(ValueError, match="positive absolute pressure"):
            RunConfig(downstream_pressure=bad)

    def test_the_unit_is_validated(self):
        with pytest.raises(ValueError, match="torr"):
            RunConfig(downstream_pressure_unit="torr")

    @pytest.mark.parametrize("value", ["measured", 101.325])
    def test_both_forms_round_trip(self, tmp_path, base_config, value):
        base_config.run.downstream_pressure = value
        save_config(base_config, tmp_path)
        assert load_config(tmp_path).run.downstream_pressure == value

    def test_the_ambient_value_is_separate_from_p2(self):
        """Ambient still exists, for gauge-to-absolute conversion."""
        run = RunConfig(atmospheric_pressure=1.0, atmospheric_pressure_unit="bar")
        assert run.atmospheric_pressure_atm == pytest.approx(units.bar_to_atm(1.0))
        assert run.downstream_is_measured

    def test_the_uncertainty_spec_is_its_own(self, base_config):
        """Widening it must not widen P1 through the ambient reference."""
        assert (
            base_config.run.downstream_pressure_uncertainty
            is not base_config.run.atmospheric_pressure_uncertainty
        )

    def test_the_template_explains_it(self, base_config):
        rendered = render_run_yaml(base_config)
        assert "downstream_pressure: measured" in rendered
        assert "measured | a number" in rendered


class TestThreeFileIo:
    def test_writes_three_files(self, tmp_path, base_config):
        paths = save_config(base_config, tmp_path)
        assert paths.hardware.name == HARDWARE_FILENAME
        assert paths.sample.name == SAMPLE_FILENAME
        assert paths.run.name == RUN_FILENAME
        for path in paths.as_tuple():
            assert path.is_file()

    def test_round_trips_exactly(self, tmp_path, base_config):
        save_config(base_config, tmp_path)
        assert config_to_dict(load_config(tmp_path)) == config_to_dict(base_config)

    def test_round_trips_with_metadata_populated(self, tmp_path):
        config = GaspermConfig()
        config.sample.lithology = "shale"
        config.sample.prepared_on = date(2026, 3, 1)
        config.run.operator = "Damian"
        config.run.confining_pressure = 12.5
        save_config(config, tmp_path)
        reloaded = load_config(tmp_path)
        assert reloaded.sample.lithology == "shale"
        assert reloaded.sample.prepared_on == date(2026, 3, 1)
        assert reloaded.run.operator == "Damian"
        assert reloaded.run.confining_pressure == 12.5

    def test_templates_carry_explanatory_comments(self, base_config):
        hardware = render_hardware_yaml(base_config)
        assert "NI-DAQmx device name" in hardware
        assert "selects ONE by name" in hardware
        assert "CGS-Darcy" in hardware
        sample = render_sample_yaml(base_config)
        assert "counts double" in sample
        run = render_run_yaml(base_config)
        assert "drift" in run and "GUM" in run

    def test_parse_pattern_survives_the_round_trip(self, tmp_path, base_config):
        """'T:{value}' contains YAML-significant characters."""
        save_config(base_config, tmp_path)
        assert load_config(tmp_path).hardware.temperature.parse_pattern == "T:{value}"

    def test_signals_list_survives_the_round_trip(self, tmp_path, base_config):
        base_config.run.steady_state.signals = ["permeability", "flow"]
        save_config(base_config, tmp_path)
        assert load_config(tmp_path).run.steady_state.signals == ["permeability", "flow"]

    def test_existing_files_are_not_overwritten_without_force(self, tmp_path, base_config):
        save_config(base_config, tmp_path)
        with pytest.raises(ConfigError, match="--force"):
            save_config(base_config, tmp_path)
        save_config(base_config, tmp_path, overwrite=True)

    def test_explicit_paths_override_the_directory(self, tmp_path, base_config):
        save_config(base_config, tmp_path)
        elsewhere = tmp_path / "other"
        elsewhere.mkdir()
        other_sample = elsewhere / "plug.yaml"
        base_config.sample.id = "core-999"
        other_sample.write_text(render_sample_yaml(base_config), encoding="utf-8")
        loaded = load_config(tmp_path, sample=other_sample)
        assert loaded.sample.id == "core-999"
        assert loaded.hardware.daq.device_name == "Dev1"

    def test_a_file_may_be_wrapped_in_its_section_key(self, tmp_path, base_config):
        save_config(base_config, tmp_path)
        data = base_config.sample.model_dump(mode="json")
        (tmp_path / SAMPLE_FILENAME).write_text(
            yaml.safe_dump({"sample": data}), encoding="utf-8"
        )
        assert load_config(tmp_path).sample.id == base_config.sample.id

    def test_a_missing_file_names_it_and_suggests_init(self, tmp_path):
        with pytest.raises(ConfigError, match="gasperm init"):
            load_config(tmp_path)

    def test_malformed_yaml_names_the_file(self, tmp_path, base_config):
        save_config(base_config, tmp_path)
        (tmp_path / RUN_FILENAME).write_text("gas: [unclosed\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="not valid YAML"):
            load_config(tmp_path)

    def test_empty_file_is_rejected(self, tmp_path, base_config):
        save_config(base_config, tmp_path)
        (tmp_path / SAMPLE_FILENAME).write_text("", encoding="utf-8")
        with pytest.raises(ConfigError, match="empty"):
            load_config(tmp_path)

    def test_validation_error_names_the_offending_field_and_file(self, tmp_path, base_config):
        save_config(base_config, tmp_path)
        data = base_config.sample.model_dump(mode="json")
        data["length_cm"] = -3.0
        (tmp_path / SAMPLE_FILENAME).write_text(yaml.safe_dump(data), encoding="utf-8")
        with pytest.raises(ConfigError, match="length_cm"):
            load_config(tmp_path)

    def test_shipped_example_configs_load(self):
        """examples/ mirrors the real layout: rig + run, and a plug in samples/."""
        from pathlib import Path

        examples = Path(__file__).resolve().parents[1] / "examples"
        if not (examples / HARDWARE_FILENAME).is_file():  # pragma: no cover
            pytest.skip("examples/ not present")
        # init writes no sample file, so the example set must not have one here.
        assert not (examples / SAMPLE_FILENAME).exists()
        config = load_config(examples, sample=examples / "samples" / "core-001.yaml")
        assert config.flowmeter.channel == "ai2"
        assert config.sample.id == "core-001"

    def test_config_paths_in_directory(self, tmp_path):
        paths = ConfigPaths.in_directory(tmp_path)
        assert paths.hardware == tmp_path / HARDWARE_FILENAME
        assert len(paths.as_tuple()) == 3


class TestSingleSectionLoading:
    """`klinkenberg` needs run.yaml alone; a rig folder has no sample.yaml."""

    def test_run_config_loads_from_a_rig_folder_without_a_sample(self, tmp_path, base_config):
        save_config(base_config, tmp_path, sections=("hardware", "run"))
        assert not (tmp_path / SAMPLE_FILENAME).exists()
        assert load_run_config(tmp_path / RUN_FILENAME).output_dir == base_config.run.output_dir

    def test_a_wrapped_run_file_also_loads(self, tmp_path, base_config):
        data = {"run": base_config.run.model_dump(mode="json")}
        path = tmp_path / RUN_FILENAME
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        assert load_run_config(path).output_dir == base_config.run.output_dir

    def test_a_missing_run_file_names_the_path(self, tmp_path):
        with pytest.raises(ConfigError, match=RUN_FILENAME):
            load_run_config(tmp_path / RUN_FILENAME)

    def test_sample_config_loads_on_its_own(self, tmp_path, base_config):
        base_config.sample.id = "core-042"
        path = tmp_path / "core-042.yaml"
        path.write_text(render_sample_yaml(base_config), encoding="utf-8")
        assert load_sample_config(path).id == "core-042"

    def test_an_invalid_sample_file_is_rejected(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("length: -5\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="length"):
            load_sample_config(path)


class TestResolvedOutputDir:
    """Relative paths in the config mean 'relative to the config folder'."""

    def test_relative_output_is_anchored_to_the_config_folder(self, tmp_path, base_config):
        save_config(base_config, tmp_path)
        loaded = load_config(tmp_path)
        assert loaded.run.output_dir == "./runs"
        assert loaded.resolved_output_dir().resolve() == (tmp_path / "runs").resolve()

    def test_an_absolute_output_is_left_alone(self, tmp_path, base_config):
        base_config.run.output_dir = (tmp_path / "elsewhere").as_posix()
        save_config(base_config, tmp_path)
        assert load_config(tmp_path).resolved_output_dir() == tmp_path / "elsewhere"

    def test_an_unloaded_config_falls_back_to_the_raw_path(self, base_config):
        """A config built in memory has no folder to anchor against."""
        assert base_config.config_dir is None
        assert base_config.resolved_output_dir() == Path(base_config.run.output_dir)

    def test_the_anchor_is_not_serialised(self, tmp_path, base_config):
        save_config(base_config, tmp_path)
        assert "config_dir" not in config_to_dict(load_config(tmp_path))


class TestLegacyRejection:
    """Clean break: a pre-split single-file config must not load silently."""

    def test_a_legacy_single_file_is_rejected_with_instructions(self, tmp_path, base_config):
        save_config(base_config, tmp_path)
        legacy = {
            "daq": {"device_name": "Dev1"},
            "flowmeter": {"channel": "ai2"},
            "gas": {"name": "Nitrogen"},
            "sample": {"id": "core-001"},
            "run": {"output_dir": "./runs"},
        }
        (tmp_path / HARDWARE_FILENAME).write_text(yaml.safe_dump(legacy), encoding="utf-8")
        with pytest.raises(ConfigError, match="pre-split single-file config"):
            load_config(tmp_path)

    def test_the_message_names_the_three_replacement_files(self, tmp_path, base_config):
        save_config(base_config, tmp_path)
        legacy = {"daq": {}, "sample": {}, "run": {}}
        (tmp_path / HARDWARE_FILENAME).write_text(yaml.safe_dump(legacy), encoding="utf-8")
        with pytest.raises(ConfigError) as info:
            load_config(tmp_path)
        for name in (HARDWARE_FILENAME, SAMPLE_FILENAME, RUN_FILENAME):
            assert name in str(info.value)

    def test_a_valid_hardware_file_is_not_mistaken_for_legacy(self, tmp_path, base_config):
        """hardware.yaml legitimately has daq/flowmeter at top level."""
        save_config(base_config, tmp_path)
        assert load_config(tmp_path).hardware.daq.device_name == "Dev1"


class TestValidateForCollect:
    def test_a_healthy_config_raises_nothing(self, base_config, fake_serial):
        import sys

        sys.modules["serial.tools.list_ports"].available = ["COM4"]
        warnings = validate_for_collect(base_config)
        # The shipped probe (0.75 s) is slower than the shipped sample rate
        # (10 Hz), so the informational cadence note is expected. Nothing else.
        assert len(warnings) == 1
        assert "held for about" in warnings[0]

    def test_a_faster_probe_than_the_sample_rate_says_nothing(
        self, base_config, fake_serial
    ):
        import sys

        sys.modules["serial.tools.list_ports"].available = ["COM4"]
        base_config.hardware.temperature.conversion_time_s = 0.01
        assert validate_for_collect(base_config) == []

    def test_a_missing_required_port_is_fatal(self, base_config, fake_serial):
        import sys

        sys.modules["serial.tools.list_ports"].available = ["COM7"]
        with pytest.raises(ConfigError, match="COM4"):
            validate_for_collect(base_config)

    def test_a_missing_optional_port_only_warns(self, base_config, fake_serial):
        import sys

        sys.modules["serial.tools.list_ports"].available = ["COM7"]
        base_config.hardware.temperature.required = False
        assert any("COM4" in w for w in validate_for_collect(base_config))

    def test_an_unknown_gas_is_fatal(self, base_config, fake_serial):
        pytest.importorskip("CoolProp")
        base_config.run.gas.name = "Unobtainium"
        with pytest.raises(ConfigError, match="Unobtainium"):
            validate_for_collect(base_config)

    def test_implausible_geometry_warns(self, base_config, fake_serial):
        base_config.sample.length = 5000.0
        assert any("unusually long" in w for w in validate_for_collect(base_config))

    def test_disabled_steady_state_warns_that_the_result_is_not_representative(
        self, base_config, fake_serial
    ):
        base_config.run.steady_state.enabled = False
        assert any(
            "not a representative measurement" in w
            for w in validate_for_collect(base_config)
        )

    def test_a_duration_too_short_for_the_criteria_warns(self, base_config, fake_serial):
        base_config.run.duration_s = 10.0  # criteria need 3 x 30 s
        assert any("shorter than" in w for w in validate_for_collect(base_config))

    def test_all_zero_instrument_specs_warn_about_the_budget(self, base_config, fake_serial):
        from gasperm.config.common import UncertaintySpec

        none_spec = UncertaintySpec(kind="none")
        base_config.hardware.pressure_calibration.inlet.uncertainty = none_spec
        base_config.hardware.pressure_calibration.outlet.uncertainty = none_spec
        base_config.flowmeter.uncertainty = none_spec
        assert any("understate the real uncertainty" in w for w in validate_for_collect(base_config))
