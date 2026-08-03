"""Run output: CSV streaming, metadata, and reduction back to Klinkenberg points."""

from __future__ import annotations

import csv

import pytest
import yaml

from gasperm.acquisition import AcquisitionLoop, SampleProcessor
from gasperm.config import GaspermConfig
from gasperm.gas_properties import FixedPropertyProvider
from gasperm.klinkenberg import fit_klinkenberg
from gasperm.storage import (
    READING_COLUMNS,
    RunWriter,
    collect_points,
    point_from_run,
    read_readings_csv,
    read_run_metadata,
    resolve_run_paths,
    run_directory_name,
    write_klinkenberg_result,
)

VOLTAGES = {"ai0": 2.5, "ai1": 0.5, "ai2": 4.0}


def run_once(config: GaspermConfig, analog_source, temperature_source, samples: int = 6):
    """Drive a short run and return ``(loop, writer)`` with the CSV closed."""
    config.run.max_samples = samples
    config.daq.sample_rate_hz = 1000.0
    writer = RunWriter(config).open()
    loop = AcquisitionLoop(
        config,
        SampleProcessor(config, FixedPropertyProvider("Nitrogen", 0.0178)),
        analog_source,
        temperature_source,
        on_reading=writer.write,
        sleep=lambda _s: None,
    )
    loop.run(install_signal_handler=False)
    writer.close()
    return loop, writer


@pytest.fixture
def run_config(base_config: GaspermConfig, tmp_path) -> GaspermConfig:
    base_config.run.output_dir = str(tmp_path / "runs")
    base_config.run.outlet_pressure_reference = "measured"
    return base_config


class TestRunDirectory:
    def test_name_carries_the_sample_id_and_a_utc_stamp(self):
        from datetime import datetime, timezone

        moment = datetime(2026, 8, 3, 14, 15, 30, tzinfo=timezone.utc)
        assert run_directory_name("core-001", moment) == "core-001_20260803T141530Z"

    def test_unsafe_characters_are_replaced(self):
        name = run_directory_name("core/001 A")
        assert "/" not in name and " " not in name


class TestRunWriter:
    def test_writes_a_header_and_one_row_per_sample(
        self, run_config, fake_analog_source, fake_temperature_source
    ):
        _, writer = run_once(
            run_config, fake_analog_source(VOLTAGES), fake_temperature_source(), samples=6
        )
        with writer.readings_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 6
        assert list(rows[0]) == list(READING_COLUMNS)

    def test_stores_cgs_units_not_display_units(
        self, run_config, fake_analog_source, fake_temperature_source
    ):
        """A stored run must mean the same thing whatever the console showed."""
        run_config.run.display_pressure_unit = "psi"
        run_config.run.display_permeability_unit = "um2"
        loop, writer = run_once(
            run_config, fake_analog_source(VOLTAGES), fake_temperature_source()
        )
        rows = read_readings_csv(writer.readings_path)
        assert rows[0]["mean_pressure_atm"] == pytest.approx(
            loop.readings[0].mean_pressure_atm, rel=1e-6
        )
        assert rows[0]["permeability_D"] == pytest.approx(
            loop.readings[0].permeability_darcy, rel=1e-6
        )

    def test_keeps_raw_voltages_for_reprocessing(
        self, run_config, fake_analog_source, fake_temperature_source
    ):
        _, writer = run_once(run_config, fake_analog_source(VOLTAGES), fake_temperature_source())
        with writer.readings_path.open(encoding="utf-8", newline="") as handle:
            first = next(csv.DictReader(handle))
        assert float(first["inlet_voltage_V"]) == pytest.approx(2.5)
        assert float(first["flow_voltage_V"]) == pytest.approx(4.0)

    def test_flushes_periodically_so_a_crash_loses_little(
        self, run_config, fake_analog_source, fake_temperature_source
    ):
        run_config.run.flush_every_n = 2
        run_config.run.max_samples = 5
        run_config.daq.sample_rate_hz = 1000.0
        writer = RunWriter(run_config).open()
        processor = SampleProcessor(run_config, FixedPropertyProvider("Nitrogen", 0.0178))
        loop = AcquisitionLoop(
            run_config,
            processor,
            fake_analog_source(VOLTAGES),
            fake_temperature_source(),
            on_reading=writer.write,
            sleep=lambda _s: None,
        )
        loop.run(install_signal_handler=False)
        # Readable before close(), because of the periodic flush.
        partial = writer.readings_path.read_text(encoding="utf-8")
        assert partial.count("\n") >= 5
        writer.close()

    def test_an_unusable_sample_leaves_the_permeability_cell_blank(
        self, run_config, fake_analog_source, fake_temperature_source
    ):
        flat = {"ai0": 1.0, "ai1": 1.0, "ai2": 4.0}
        _, writer = run_once(run_config, fake_analog_source(flat), fake_temperature_source())
        with writer.readings_path.open(encoding="utf-8", newline="") as handle:
            first = next(csv.DictReader(handle))
        assert first["permeability_D"] == ""
        assert first["note"]

    def test_writing_before_open_is_rejected(self, run_config):
        writer = RunWriter(run_config)
        processor = SampleProcessor(run_config, FixedPropertyProvider("Nitrogen", 0.0178))
        from gasperm.hardware.temperature import TemperatureSample

        reading = processor.process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES,
            temperature=TemperatureSample(22.0, 0.0, None, False),
        )
        with pytest.raises(RuntimeError, match="not open"):
            writer.write(reading)


class TestMetadata:
    def test_snapshots_the_whole_config(
        self, run_config, fake_analog_source, fake_temperature_source
    ):
        loop, writer = run_once(
            run_config, fake_analog_source(VOLTAGES), fake_temperature_source()
        )
        writer.write_metadata(loop.summarize(csv_path=str(writer.readings_path)))
        data = yaml.safe_load(writer.metadata_path.read_text(encoding="utf-8"))
        assert data["config"]["sample"]["id"] == run_config.sample.id
        assert data["config"]["flowmeter"]["channel"] == run_config.flowmeter.channel
        assert data["summary"]["permeability_darcy"] > 0.0

    def test_stored_config_reloads_as_a_valid_config(
        self, run_config, fake_analog_source, fake_temperature_source
    ):
        """A run must be self-describing without the original config file."""
        _, writer = run_once(run_config, fake_analog_source(VOLTAGES), fake_temperature_source())
        writer.write_metadata()
        data = read_run_metadata(writer.metadata_path)
        assert GaspermConfig.model_validate(data["config"]).sample.id == run_config.sample.id

    def test_metadata_can_be_written_without_a_summary(
        self, run_config, fake_analog_source, fake_temperature_source
    ):
        _, writer = run_once(run_config, fake_analog_source(VOLTAGES), fake_temperature_source())
        writer.write_metadata()
        assert "summary" not in read_run_metadata(writer.metadata_path)

    def test_a_missing_metadata_file_reads_as_empty(self, tmp_path):
        assert read_run_metadata(tmp_path / "absent.yaml") == {}


class TestReadingBack:
    def test_a_run_directory_resolves_to_its_csv(
        self, run_config, fake_analog_source, fake_temperature_source
    ):
        _, writer = run_once(run_config, fake_analog_source(VOLTAGES), fake_temperature_source())
        writer.write_metadata()
        readings, metadata = resolve_run_paths(writer.directory)
        assert readings == writer.readings_path
        assert metadata == writer.metadata_path

    def test_a_csv_path_resolves_directly(
        self, run_config, fake_analog_source, fake_temperature_source
    ):
        _, writer = run_once(run_config, fake_analog_source(VOLTAGES), fake_temperature_source())
        readings, _ = resolve_run_paths(writer.readings_path)
        assert readings == writer.readings_path

    def test_a_directory_without_a_csv_is_rejected(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="readings.csv"):
            resolve_run_paths(tmp_path / "empty")

    def test_a_foreign_csv_is_rejected_by_name(self, tmp_path):
        path = tmp_path / "other.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required column"):
            read_readings_csv(path)


class TestPointFromRun:
    def test_reduces_a_run_to_its_steady_state_point(
        self, run_config, fake_analog_source, fake_temperature_source
    ):
        loop, writer = run_once(
            run_config, fake_analog_source(VOLTAGES), fake_temperature_source(), samples=8
        )
        writer.write_metadata(loop.summarize())
        point = point_from_run(writer.directory)
        assert point.sample_id == run_config.sample.id
        assert point.apparent_permeability_darcy == pytest.approx(
            loop.readings[-1].permeability_darcy, rel=1e-6
        )
        assert point.mean_pressure_atm == pytest.approx(
            loop.readings[-1].mean_pressure_atm, rel=1e-6
        )

    def test_matches_the_live_run_summary(
        self, run_config, fake_analog_source, fake_temperature_source
    ):
        """collect and klinkenberg must agree on what 'steady state' means."""
        loop, writer = run_once(
            run_config, fake_analog_source(VOLTAGES), fake_temperature_source(), samples=8
        )
        summary = loop.summarize()
        writer.write_metadata(summary)
        point = point_from_run(writer.directory)
        assert point.apparent_permeability_darcy == pytest.approx(
            summary.permeability_darcy, rel=1e-6
        )

    def test_a_run_with_no_usable_sample_is_rejected(
        self, run_config, fake_analog_source, fake_temperature_source
    ):
        flat = {"ai0": 1.0, "ai1": 1.0, "ai2": 4.0}
        _, writer = run_once(run_config, fake_analog_source(flat), fake_temperature_source())
        writer.write_metadata()
        with pytest.raises(ValueError, match="no sample with a usable permeability"):
            point_from_run(writer.directory)

    def test_several_runs_regress_end_to_end(
        self, base_config, tmp_path, fake_analog_source, fake_temperature_source
    ):
        """collect -> storage -> klinkenberg, with no hardware anywhere."""
        directories = []
        # Three inlet voltages give three different mean pressures.
        for inlet_volts in (1.5, 2.5, 4.0):
            config = base_config.model_copy(deep=True)
            config.run.output_dir = str(tmp_path / "runs")
            config.run.outlet_pressure_reference = "measured"
            config.sample.id = "core-001"
            _, writer = run_once(
                config,
                fake_analog_source({**VOLTAGES, "ai0": inlet_volts}),
                fake_temperature_source(),
                samples=6,
            )
            writer.write_metadata()
            directories.append(writer.directory)

        points = collect_points(directories)
        assert len(points) == 3
        assert {p.sample_id for p in points} == {"core-001"}
        result = fit_klinkenberg(points)
        assert result.point_count == 3
        assert 0.0 <= result.r_squared <= 1.0


class TestKlinkenbergOutput:
    def test_writes_results_and_the_points_used(self, tmp_path):
        from gasperm.models import KlinkenbergPoint

        points = [
            KlinkenbergPoint(
                mean_pressure_atm=p,
                apparent_permeability_darcy=0.5 * (1.0 + 0.2 / p),
                label=f"run{i}",
                sample_id="core-001",
            )
            for i, p in enumerate((1.0, 2.0, 4.0))
        ]
        result = fit_klinkenberg(points)
        path = write_klinkenberg_result(result, tmp_path / "klinkenberg.yaml")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))

        assert data["klinkenberg"]["liquid_permeability_D"] == pytest.approx(0.5, rel=1e-8)
        assert data["klinkenberg"]["liquid_permeability_mD"] == pytest.approx(500.0, rel=1e-8)
        assert data["klinkenberg"]["slippage_factor_atm"] == pytest.approx(0.2, rel=1e-8)
        assert len(data["points"]) == 3
        assert data["points"][0]["label"] == "run0"
