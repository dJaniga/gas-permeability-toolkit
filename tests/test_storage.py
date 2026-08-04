"""Run output: CSV streaming, metadata, and reduction back to Klinkenberg points."""

from __future__ import annotations

import csv
from datetime import datetime, timezone

import pytest
import yaml

from gasperm.acquisition import AcquisitionLoop, SampleProcessor
from gasperm.config import GaspermConfig
from gasperm.gas_properties import FixedPropertyProvider
from gasperm.klinkenberg import fit_klinkenberg
from gasperm.storage import (
    METADATA_FILENAME,
    READING_COLUMNS,
    RunWriter,
    collect_points,
    describe_convention,
    downstream_convention,
    find_runs,
    point_from_run,
    read_readings_csv,
    read_run_metadata,
    resolve_run_paths,
    run_directory_name,
    runs_for_sample,
    safe_sample_id,
    write_klinkenberg_result,
)

VOLTAGES = {"ai0": 2.5, "ai1": 0.5, "ai2": 4.0}


class FakeClock:
    """Monotonic clock advancing a fixed step per call."""

    def __init__(self, step: float = 0.05) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


def run_once(config: GaspermConfig, analog_source, temperature_source, samples: int = 80):
    """Drive a short run to steady state; returns ``(loop, writer)``, CSV closed."""
    config.run.max_samples = samples
    writer = RunWriter(config).open()
    loop = AcquisitionLoop(
        config,
        SampleProcessor(config, FixedPropertyProvider("Nitrogen", 0.0178)),
        analog_source,
        temperature_source,
        on_reading=writer.write,
        sleep=lambda _s: None,
        clock=FakeClock(),
    )
    loop.run(install_signal_handler=False)
    writer.close()
    return loop, writer


@pytest.fixture
def run_config(quick_steady_config: GaspermConfig, tmp_path) -> GaspermConfig:
    quick_steady_config.run.output_dir = str(tmp_path / "runs")
    return quick_steady_config


class TestRunDirectory:
    def test_name_carries_the_sample_id_and_a_utc_stamp(self):
        from datetime import datetime, timezone

        moment = datetime(2026, 8, 3, 14, 15, 30, tzinfo=timezone.utc)
        assert run_directory_name("core-001", moment) == "core-001_20260803T141530Z"

    def test_unsafe_characters_are_replaced(self):
        name = run_directory_name("core/001 A")
        assert "/" not in name and " " not in name

    def test_runs_started_in_the_same_second_do_not_collide(self, run_config):
        first = RunWriter(run_config).open()
        second = RunWriter(run_config, started_at=first.started_at).open()
        assert first.directory != second.directory
        first.close()
        second.close()


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

    def test_records_both_the_measured_and_the_used_downstream_pressure(
        self, run_config, fake_analog_source, fake_temperature_source
    ):
        run_config.run.downstream_pressure = 101.325
        loop, writer = run_once(
            run_config, fake_analog_source(VOLTAGES), fake_temperature_source()
        )
        rows = read_readings_csv(writer.readings_path)
        assert rows[0]["outlet_pressure_atm"] == pytest.approx(
            loop.readings[0].outlet_pressure_atm, rel=1e-6
        )
        assert rows[0]["downstream_pressure_atm"] == pytest.approx(
            loop.readings[0].downstream_pressure_atm, rel=1e-6
        )
        assert rows[0]["outlet_pressure_atm"] != pytest.approx(
            rows[0]["downstream_pressure_atm"]
        )

    def test_a_csv_without_the_downstream_column_still_loads(self, tmp_path):
        """Runs recorded before P2 became overridable."""
        path = tmp_path / "old.csv"
        path.write_text(
            "elapsed_s,mean_pressure_atm,permeability_D\n0.0,2.0,0.005\n",
            encoding="utf-8",
        )
        rows = read_readings_csv(path)
        assert rows[0]["downstream_pressure_atm"] is None
        assert rows[0]["mean_pressure_atm"] == pytest.approx(2.0)

    def test_keeps_raw_voltages_for_reprocessing(
        self, run_config, fake_analog_source, fake_temperature_source
    ):
        _, writer = run_once(run_config, fake_analog_source(VOLTAGES), fake_temperature_source())
        with writer.readings_path.open(encoding="utf-8", newline="") as handle:
            first = next(csv.DictReader(handle))
        assert float(first["inlet_voltage_V"]) == pytest.approx(2.5)
        assert float(first["flow_voltage_V"]) == pytest.approx(4.0)

    def test_records_the_steady_state_verdict_per_sample(
        self, run_config, fake_analog_source, fake_temperature_source
    ):
        loop, writer = run_once(
            run_config, fake_analog_source(VOLTAGES), fake_temperature_source()
        )
        assert loop.steady_state_reached
        with writer.readings_path.open(encoding="utf-8", newline="") as handle:
            flags = [int(row["steady_state"]) for row in csv.DictReader(handle)]
        assert flags[0] == 0  # cannot be steady before the first window closes
        assert flags[-1] == 1

    def test_flushes_periodically_so_a_crash_loses_little(
        self, run_config, fake_analog_source, fake_temperature_source
    ):
        run_config.run.flush_every_n = 2
        run_config.run.max_samples = 5
        writer = RunWriter(run_config).open()
        loop = AcquisitionLoop(
            run_config,
            SampleProcessor(run_config, FixedPropertyProvider("Nitrogen", 0.0178)),
            fake_analog_source(VOLTAGES),
            fake_temperature_source(),
            on_reading=writer.write,
            sleep=lambda _s: None,
        )
        loop.run(install_signal_handler=False)
        # Readable before close(), because of the periodic flush.
        assert writer.readings_path.read_text(encoding="utf-8").count("\n") >= 5
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
        from gasperm.hardware.temperature import TemperatureSample

        writer = RunWriter(run_config)
        processor = SampleProcessor(run_config, FixedPropertyProvider("Nitrogen", 0.0178))
        reading = processor.process(
            index=0, elapsed_s=0.0, voltages=VOLTAGES,
            temperature=TemperatureSample(22.0, 0.0, None, False),
        )
        with pytest.raises(RuntimeError, match="not open"):
            writer.write(reading)


class TestMetadata:
    def test_the_sidecar_is_not_named_like_the_run_config_file(self):
        """run.yaml is the experiment config; the sidecar must not shadow it."""
        assert METADATA_FILENAME == "run_metadata.yaml"

    def test_snapshots_the_whole_config_and_the_metadata(
        self, run_config, fake_analog_source, fake_temperature_source
    ):
        run_config.run.operator = "Damian"
        run_config.sample.lithology = "sandstone"
        loop, writer = run_once(
            run_config, fake_analog_source(VOLTAGES), fake_temperature_source()
        )
        writer.write_metadata(loop.summarize(csv_path=str(writer.readings_path)))
        data = yaml.safe_load(writer.metadata_path.read_text(encoding="utf-8"))

        assert data["config"]["sample"]["id"] == run_config.sample.id
        assert data["config"]["hardware"]["flowmeters"]["low_range"]["channel"] == "ai2"
        assert data["metadata"]["flowmeter"] == "low_range"
        assert data["metadata"]["operator"] == "Damian"
        assert data["metadata"]["lithology"] == "sandstone"
        assert data["summary"]["permeability_darcy"] > 0.0
        assert data["summary"]["steady_state_reached"] is True
        assert data["summary"]["uncertainty"]["expanded_uncertainty_darcy"] > 0.0

    def test_stored_config_reloads_as_a_valid_config(
        self, run_config, fake_analog_source, fake_temperature_source
    ):
        """A run must be self-describing without the original config files."""
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
        assert resolve_run_paths(writer.readings_path)[0] == writer.readings_path

    def test_a_directory_without_a_csv_is_rejected(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="readings.csv"):
            resolve_run_paths(tmp_path / "empty")

    def test_a_foreign_csv_is_rejected_by_name(self, tmp_path):
        path = tmp_path / "other.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required column"):
            read_readings_csv(path)

    def test_rows_carry_the_detector_signal_aliases(
        self, run_config, fake_analog_source, fake_temperature_source
    ):
        """So a stored run replays through the same detector the live run used."""
        _, writer = run_once(run_config, fake_analog_source(VOLTAGES), fake_temperature_source())
        row = read_readings_csv(writer.readings_path)[0]
        assert row["permeability"] == row["permeability_D"]
        assert row["inlet_pressure"] == row["inlet_pressure_atm"]
        assert row["flow"] == row["flow_cm3_s"]
        assert row["temperature"] == pytest.approx(row["temperature_C"] + 273.15)


class TestPointFromRun:
    def test_reduces_a_steady_run_to_one_point(
        self, run_config, fake_analog_source, fake_temperature_source
    ):
        loop, writer = run_once(
            run_config, fake_analog_source(VOLTAGES), fake_temperature_source()
        )
        summary = loop.summarize()
        writer.write_metadata(summary)
        point = point_from_run(writer.directory)
        assert point.sample_id == run_config.sample.id
        assert point.steady_state is True
        assert point.apparent_permeability_darcy == pytest.approx(
            summary.permeability_darcy, rel=1e-6
        )

    def test_the_stored_summary_brings_its_uncertainty_across(
        self, run_config, fake_analog_source, fake_temperature_source
    ):
        loop, writer = run_once(
            run_config, fake_analog_source(VOLTAGES), fake_temperature_source()
        )
        writer.write_metadata(loop.summarize())
        point = point_from_run(writer.directory)
        assert point.standard_uncertainty_darcy is not None
        assert point.standard_uncertainty_darcy > 0.0

    def test_replaying_the_csv_agrees_with_the_live_summary(
        self, run_config, fake_analog_source, fake_temperature_source
    ):
        """collect and klinkenberg must agree on what steady state means."""
        loop, writer = run_once(
            run_config, fake_analog_source(VOLTAGES), fake_temperature_source()
        )
        summary = loop.summarize()
        writer.write_metadata(summary)
        # Forcing a window override bypasses the stored summary and replays.
        replayed = point_from_run(
            writer.directory, averaging_window_s=run_config.run.steady_state.window_s
        )
        assert replayed.apparent_permeability_darcy == pytest.approx(
            summary.permeability_darcy, rel=1e-3
        )

    def test_a_run_that_never_settled_is_refused_by_default(
        self, run_config, fake_analog_source, fake_temperature_source
    ):
        ramp = [{**VOLTAGES, "ai0": 1.0 + 0.05 * i} for i in range(60)]
        loop, writer = run_once(
            run_config, fake_analog_source(ramp), fake_temperature_source(), samples=60
        )
        writer.write_metadata(loop.summarize())
        with pytest.raises(ValueError, match="never reached steady state"):
            point_from_run(writer.directory)

    def test_an_unsteady_run_can_be_forced_and_is_flagged(
        self, run_config, fake_analog_source, fake_temperature_source
    ):
        ramp = [{**VOLTAGES, "ai0": 1.0 + 0.05 * i} for i in range(60)]
        loop, writer = run_once(
            run_config, fake_analog_source(ramp), fake_temperature_source(), samples=60
        )
        writer.write_metadata(loop.summarize())
        point = point_from_run(writer.directory, allow_unsteady=True)
        assert point.steady_state is False

    def test_a_run_with_no_usable_sample_is_rejected(
        self, run_config, fake_analog_source, fake_temperature_source
    ):
        flat = {"ai0": 1.0, "ai1": 1.0, "ai2": 4.0}
        loop, writer = run_once(run_config, fake_analog_source(flat), fake_temperature_source())
        writer.write_metadata()
        with pytest.raises(ValueError, match="no sample with a usable permeability"):
            point_from_run(writer.directory)

    def test_several_runs_regress_end_to_end(
        self, quick_steady_config, tmp_path, fake_analog_source, fake_temperature_source
    ):
        """collect -> storage -> klinkenberg, with no hardware anywhere."""
        directories = []
        for inlet_volts in (1.5, 2.5, 4.0):
            config = quick_steady_config.model_copy(deep=True)
            config.run.output_dir = str(tmp_path / "runs")
            loop, writer = run_once(
                config,
                fake_analog_source({**VOLTAGES, "ai0": inlet_volts}),
                fake_temperature_source(),
            )
            writer.write_metadata(loop.summarize())
            directories.append(writer.directory)

        points = collect_points(directories)
        assert len(points) == 3
        assert all(p.steady_state for p in points)
        assert {p.sample_id for p in points} == {"core-001"}

        result = fit_klinkenberg(points)
        assert result.point_count == 3
        assert result.weighted is True
        assert 0.0 <= result.r_squared <= 1.0


class TestDownstreamConvention:
    """How a stored run obtained P2, recovered from its sidecar."""

    def test_measured_runs(self):
        assert downstream_convention({"run": {"downstream_pressure": "measured"}}) == "measured"

    def test_a_supplied_value_is_keyed_in_atm(self):
        key = downstream_convention(
            {"run": {"downstream_pressure": 101.325, "downstream_pressure_unit": "kPa"}}
        )
        assert key.startswith("fixed:")

    def test_the_same_pressure_in_two_units_compares_equal(self):
        """101.325 kPa and 1.01325 bar are one convention, not two."""
        in_kpa = downstream_convention(
            {"run": {"downstream_pressure": 101.325, "downstream_pressure_unit": "kPa"}}
        )
        in_bar = downstream_convention(
            {"run": {"downstream_pressure": 1.01325, "downstream_pressure_unit": "bar"}}
        )
        assert in_kpa == in_bar

    def test_a_run_block_without_the_key_was_measured_by_definition(self):
        """The hole this closes: it must not read as 'unknown'.

        A set of runs recorded before P2 became overridable, plus one supplied
        run, would otherwise collapse to a single distinct convention and slip
        past the mixed-convention refusal.
        """
        assert downstream_convention({"run": {"output_dir": "./runs"}}) == "measured"

    def test_no_run_block_is_genuinely_unknown(self):
        assert downstream_convention({"sample": {"id": "core-001"}}) is None
        assert downstream_convention({}) is None
        assert downstream_convention(None) is None

    def test_descriptions_are_readable(self):
        assert describe_convention("measured") == "measured"
        assert describe_convention(None) == "unknown"
        assert "kPa" in describe_convention("fixed:1")

    def test_a_record_carries_it(self, tmp_path, fake_run_writer):
        runs = tmp_path / "runs"
        fake_run_writer(runs, "core-041", datetime(2026, 8, 3, 9, tzinfo=timezone.utc))
        fake_run_writer(
            runs, "core-041", datetime(2026, 8, 3, 10, tzinfo=timezone.utc),
            downstream_pressure=101.325,
        )
        conventions = [r.downstream_convention for r in find_runs(runs)]
        assert conventions[0] == "measured"
        assert conventions[1].startswith("fixed:")

    def test_a_point_carries_it(self, tmp_path, fake_run_writer):
        runs = tmp_path / "runs"
        directory = fake_run_writer(
            runs, "core-041", datetime(2026, 8, 3, 9, tzinfo=timezone.utc),
            downstream_pressure=101.325,
        )
        assert point_from_run(directory).downstream_convention.startswith("fixed:")


class TestFindRuns:
    """Discovery over a runs directory, without reading any CSV."""

    def _tree(self, tmp_path, writer):
        runs = tmp_path / "runs"
        writer(runs, "core-041", datetime(2026, 8, 3, 14, 15, tzinfo=timezone.utc),
               mean_pressure_atm=2.07)
        writer(runs, "core-041", datetime(2026, 8, 3, 15, 22, tzinfo=timezone.utc),
               mean_pressure_atm=4.64)
        writer(runs, "core-042", datetime(2026, 8, 3, 16, 1, tzinfo=timezone.utc),
               mean_pressure_atm=3.11)
        return runs

    def test_finds_every_run(self, tmp_path, fake_run_writer):
        runs = self._tree(tmp_path, fake_run_writer)
        assert len(find_runs(runs)) == 3

    def test_finds_runs_written_by_the_real_writer(
        self, run_config, fake_analog_source, fake_temperature_source
    ):
        _, writer = run_once(run_config, fake_analog_source(VOLTAGES), fake_temperature_source())
        writer.write_metadata()
        records = find_runs(writer.directory.parent)
        assert [r.directory for r in records] == [writer.directory]
        assert records[0].sample_id == run_config.sample.id

    def test_ignores_things_that_are_not_runs(self, tmp_path, fake_run_writer):
        runs = self._tree(tmp_path, fake_run_writer)
        (runs / "klinkenberg_core-041.yaml").write_text("x: 1\n", encoding="utf-8")
        (runs / "klinkenberg_core-041.png").write_bytes(b"")
        (runs / "notes").mkdir()
        assert len(find_runs(runs)) == 3

    def test_sorted_oldest_first(self, tmp_path, fake_run_writer):
        runs = tmp_path / "runs"
        fake_run_writer(runs, "core-041", datetime(2026, 8, 3, 16, 0, tzinfo=timezone.utc))
        fake_run_writer(runs, "core-041", datetime(2026, 8, 3, 9, 0, tzinfo=timezone.utc))
        fake_run_writer(runs, "core-041", datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc))
        hours = [r.started_at.hour for r in find_runs(runs)]
        assert hours == [9, 12, 16]

    def test_carries_the_summary_fields(self, tmp_path, fake_run_writer):
        runs = self._tree(tmp_path, fake_run_writer)
        record = find_runs(runs)[0]
        assert record.has_summary
        assert record.mean_pressure_atm == pytest.approx(2.07)
        assert record.permeability_darcy == pytest.approx(0.005)
        assert record.steady_state_reached is True
        assert record.flowmeter == "low_range"
        assert record.sample_id_from_metadata is True

    def test_a_run_without_a_sidecar_is_still_found(self, tmp_path, fake_run_writer):
        runs = tmp_path / "runs"
        fake_run_writer(runs, "core-041", datetime(2026, 8, 3, 14, 0, tzinfo=timezone.utc),
                        sidecar=False)
        record = find_runs(runs)[0]
        assert record.sample_id == "core-041"
        assert record.sample_id_from_metadata is False
        assert record.has_summary is False
        assert record.started_at.hour == 14  # recovered from the directory stamp

    def test_a_corrupt_sidecar_does_not_hide_the_run(self, tmp_path, fake_run_writer):
        runs = tmp_path / "runs"
        directory = fake_run_writer(
            runs, "core-041", datetime(2026, 8, 3, 14, 0, tzinfo=timezone.utc)
        )
        (directory / METADATA_FILENAME).write_text("{[not yaml\n", encoding="utf-8")
        record = find_runs(runs)[0]
        assert record.directory == directory
        assert record.has_summary is False

    def test_a_collision_suffixed_directory_is_understood(self, tmp_path, fake_run_writer):
        runs = tmp_path / "runs"
        directory = fake_run_writer(
            runs, "core-041", datetime(2026, 8, 3, 14, 0, tzinfo=timezone.utc), sidecar=False
        )
        directory.rename(directory.with_name(directory.name + "-2"))
        record = find_runs(runs)[0]
        assert record.sample_id == "core-041"
        assert record.started_at.hour == 14

    def test_a_missing_directory_is_an_error_not_an_empty_list(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No such runs directory"):
            find_runs(tmp_path / "nope")

    def test_an_empty_directory_is_simply_empty(self, tmp_path):
        (tmp_path / "runs").mkdir()
        assert find_runs(tmp_path / "runs") == []


class TestRunsForSample:
    def test_filters_to_one_plug(self, tmp_path, fake_run_writer):
        runs = tmp_path / "runs"
        fake_run_writer(runs, "core-041", datetime(2026, 8, 3, 14, 0, tzinfo=timezone.utc))
        fake_run_writer(runs, "core-041", datetime(2026, 8, 3, 15, 0, tzinfo=timezone.utc))
        fake_run_writer(runs, "core-042", datetime(2026, 8, 3, 16, 0, tzinfo=timezone.utc))
        records = find_runs(runs)
        assert len(runs_for_sample(records, "core-041")) == 2
        assert len(runs_for_sample(records, "core-042")) == 1
        assert runs_for_sample(records, "core-099") == []

    def test_a_sidecarless_run_matches_on_the_sanitised_id(self, tmp_path, fake_run_writer):
        """The directory name is all that is left, and it is lossy."""
        runs = tmp_path / "runs"
        fake_run_writer(runs, "core/041", datetime(2026, 8, 3, 14, 0, tzinfo=timezone.utc),
                        sidecar=False)
        records = find_runs(runs)
        assert records[0].sample_id == "core_041"
        assert len(runs_for_sample(records, "core/041")) == 1

    def test_safe_sample_id_matches_the_directory_prefix(self):
        name = run_directory_name("core/041 A", datetime(2026, 8, 3, tzinfo=timezone.utc))
        assert name.startswith(safe_sample_id("core/041 A"))


class TestKlinkenbergOutput:
    def test_writes_results_points_and_uncertainties(self, tmp_path):
        from gasperm.models import KlinkenbergPoint

        points = [
            KlinkenbergPoint(
                mean_pressure_atm=p,
                apparent_permeability_darcy=0.5 * (1.0 + 0.2 / p),
                standard_uncertainty_darcy=0.001,
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
        assert data["klinkenberg"]["weighted"] is True
        assert data["klinkenberg"]["expanded_uncertainty_D"] is not None
        assert len(data["points"]) == 3
        assert data["points"][0]["standard_uncertainty_D"] == pytest.approx(0.001)
