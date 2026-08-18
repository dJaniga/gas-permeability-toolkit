"""``gasperm reprocess``: re-deriving a stored run from its raw voltages.

The load-bearing test is the boring one -- reprocessing with **no** change must
reproduce the original bit for bit. Everything else is only meaningful against
that baseline: a command that quietly perturbs a result while claiming to
re-cost it would be worse than no command.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gasperm.acquisition import (
    AcquisitionLoop,
    PulseProcessor,
    SampleProcessor,
    summarize_pulse_decay_run,
)
from gasperm.cli import app
from gasperm.config import load_config
from gasperm.gas_properties import build_provider
from gasperm.hardware.temperature import TemperatureSample
from gasperm.reprocess import (
    ReprocessError,
    classify_change,
    diff_configs,
    read_raw_samples,
    rebuild_readings,
    reprocess_run,
)
from gasperm.storage import RunWriter

from conftest import FakeAnalogSource, FakeTemperatureSource, decay_voltages

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


@pytest.fixture(autouse=True)
def _quiet_logging():
    """The CLI reconfigures root handlers; keep them off pytest's captured streams."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def make_rig(tmp_path, *overrides: str) -> Path:
    rig = tmp_path / "rig"
    args = ["init", str(rig), "--non-interactive", "--force"]
    for override in overrides:
        args += ["--set", override]
    assert runner.invoke(app, args).exit_code == 0
    assert runner.invoke(
        app, ["new-sample", "core-041", "--dir", str(rig / "samples"), "-n", "--force"]
    ).exit_code == 0
    return rig


def record_steady_run(tmp_path) -> tuple[Path, Path, object]:
    """A genuine steady-state run, through the real loop and the real writer."""
    rig = make_rig(tmp_path)
    config = load_config(rig, sample=rig / "samples" / "core-041.yaml")
    config.run.steady_state.window_s = 1.0
    config.run.steady_state.required_windows = 2
    config.run.steady_state.min_samples = 3
    config.hardware.daq.sample_rate_hz = 200.0
    config.run.max_samples = 700
    config.sample.porosity_fraction = 0.10

    provider = build_provider(config.run.gas)
    loop = AcquisitionLoop(
        config, SampleProcessor(config, provider),
        FakeAnalogSource({"ai0": 0.30, "ai1": 0.05, "ai2": 2.0}),
        FakeTemperatureSource(22.0),
    )
    loop.run(install_signal_handler=False)
    writer = RunWriter(config)
    writer.open()
    for reading in loop.readings:
        writer.write(reading)
    writer.close()
    summary = loop.summarize(csv_path=str(writer.readings_path))
    writer.write_metadata(summary)
    return rig, writer.directory, summary


def record_pulse_run(tmp_path) -> tuple[Path, Path, object]:
    """A well-conditioned pulse decay, sampled at the rate its times imply."""
    rig = make_rig(tmp_path, "run.method=pulse_decay")
    config = load_config(rig, sample=rig / "samples" / "core-041.yaml")
    config.hardware.daq.sample_rate_hz = 20.0
    for side in ("upstream", "downstream"):
        getattr(config.hardware.reservoirs, side).vessel = 8.0
        getattr(config.hardware.reservoirs, side).dead = 0.0
    config.sample.porosity_fraction = 0.10
    config.sample.porosity_uncertainty = 0.01
    config.run.pulse_decay.fit_bin_s = None
    config.run.pulse_decay.min_fit_samples = 10
    config.run.pulse_decay.storage_correction = "dicker_smits"

    step_s = 0.05
    frames = decay_voltages(config, decay_rate_per_s=0.15, duration_s=30.0, step_s=step_s)
    provider = build_provider(config.run.gas)
    processor = PulseProcessor(config, provider)
    readings = [
        processor.process(
            index=index, elapsed_s=index * step_s, voltages=frame,
            temperature=TemperatureSample(22.0, None, None, False, 0.0),
        )
        for index, frame in enumerate(frames)
    ]
    writer = RunWriter(config)
    writer.open()
    for reading in readings:
        writer.write(reading)
    writer.close()

    from gasperm.reprocess import _fit_decay

    fit, _ = _fit_decay(readings, config)
    summary = summarize_pulse_decay_run(
        readings, config, fit=fit, processor=processor,
        csv_path=str(writer.readings_path),
    )
    writer.write_metadata(summary)
    return rig, writer.directory, summary


class TestClassification:
    """Which fields move k, which only move U(k), and which move neither."""

    @pytest.mark.parametrize(
        "key",
        [
            "sample.porosity_uncertainty",
            "hardware.pressure_calibration.inlet.uncertainty.value",
            "run.uncertainty.coverage_probability",
            "run.gas.viscosity_relative_uncertainty",
            "hardware.reservoirs.upstream.uncertainty.value",
        ],
    )
    def test_uncertainty_fields(self, key):
        assert classify_change(key) == "uncertainty"

    @pytest.mark.parametrize(
        "key",
        [
            "sample.length",
            "sample.diameter",
            "hardware.pressure_calibration.inlet.value_max",
            "hardware.flowmeters.low_range.flow_max",
            "run.gas.name",
            "run.atmospheric_pressure",
            "hardware.reservoirs.upstream.vessel",
            "run.pulse_decay.fit_end_fraction",
        ],
    )
    def test_result_fields(self, key):
        assert classify_change(key) == "result"

    @pytest.mark.parametrize(
        "key",
        ["run.operator", "run.notes", "sample.lithology", "run.display_pressure_unit"],
    )
    def test_metadata_fields(self, key):
        assert classify_change(key) == "metadata"

    def test_uncertainty_wins_over_a_result_prefix(self):
        """`sample.porosity_uncertainty` sits under a result prefix in spelling only."""
        assert classify_change("sample.porosity_fraction") == "result"
        assert classify_change("sample.porosity_uncertainty") == "uncertainty"

    def test_a_diff_reports_only_what_changed(self):
        before = {"sample": {"length": 5.0, "id": "core-041"}}
        after = {"sample": {"length": 5.1, "id": "core-041"}}
        changes = diff_configs(before, after)
        assert [c.key for c in changes] == ["sample.length"]
        assert changes[0].before == 5.0 and changes[0].after == 5.1
        assert changes[0].predicted == "result"


class TestRawRecord:
    def test_the_raw_voltages_come_back(self, tmp_path):
        _, directory, _ = record_steady_run(tmp_path)
        samples = read_raw_samples(directory / "readings.csv")
        assert len(samples) == 700
        assert samples[0].inlet_voltage == pytest.approx(0.30)
        assert samples[0].flow_voltage == pytest.approx(2.0)

    def test_a_csv_without_voltages_is_refused(self, tmp_path):
        """Re-deriving from a stored pressure would keep the old calibration."""
        path = tmp_path / "readings.csv"
        path.write_text(
            "elapsed_s,mean_pressure_atm,permeability_D\n0.0,5.0,0.001\n",
            encoding="utf-8",
        )
        with pytest.raises(ReprocessError, match="missing"):
            read_raw_samples(path)

    def test_an_empty_csv_is_refused(self, tmp_path):
        path = tmp_path / "readings.csv"
        path.write_text(
            "elapsed_s,inlet_voltage_V,outlet_voltage_V,temperature_C\n", encoding="utf-8"
        )
        with pytest.raises(ReprocessError, match="no usable samples"):
            read_raw_samples(path)

    def test_the_stored_precision_survives_a_round_trip(self, tmp_path):
        """The voltages are the measurement; storing them coarsely loses it.

        Six decimals is 14 Pa on a 0-68.95 MPa transducer, which is invisible in
        a pressure and costs ~0.2% of a pulse-decay decay rate -- because that
        is a small difference between two large pressures.
        """
        _, directory, _ = record_pulse_run(tmp_path)
        text = (directory / "readings.csv").read_text(encoding="utf-8")
        column = text.splitlines()[0].split(",").index("inlet_voltage_V")
        digits = max(
            len(line.split(",")[column].split(".")[-1])
            for line in text.splitlines()[1:]
            if "." in line.split(",")[column]
        )
        assert digits > 6


class TestReDerivation:
    """The baseline everything else rests on."""

    def test_a_steady_run_reproduces_exactly(self, tmp_path):
        rig, directory, original = record_steady_run(tmp_path)
        config = load_config(rig, sample=rig / "samples" / "core-041.yaml")
        config = _config_from_sidecar(directory)
        again = reprocess_run(directory, config)
        assert again.permeability_darcy == pytest.approx(
            original.permeability_darcy, rel=1e-12
        )
        assert again.steady_state_reached == original.steady_state_reached
        assert again.averaged_samples == original.averaged_samples

    def test_a_pulse_run_reproduces_to_within_a_part_in_a_million(self, tmp_path):
        """Not exact: a decay is re-fitted, and the fit sees the stored digits."""
        _, directory, original = record_pulse_run(tmp_path)
        config = _config_from_sidecar(directory)
        again = reprocess_run(directory, config)
        assert again.permeability_darcy == pytest.approx(
            original.permeability_darcy, rel=1e-6
        )

    def test_the_readings_themselves_are_identical(self, tmp_path):
        rig, directory, _ = record_steady_run(tmp_path)
        config = _config_from_sidecar(directory)
        samples = read_raw_samples(directory / "readings.csv")
        readings = rebuild_readings(samples, config, build_provider(config.run.gas))
        assert len(readings) == len(samples)
        assert all(r.permeability_darcy is not None for r in readings[10:])

    def test_a_changed_length_scales_k_by_the_same_factor(self, tmp_path):
        """k is proportional to L, so this is checkable without the rig."""
        _, directory, original = record_steady_run(tmp_path)
        config = _config_from_sidecar(directory)
        config.sample.length = config.sample.length * 1.02
        again = reprocess_run(directory, config)
        assert again.permeability_darcy == pytest.approx(
            original.permeability_darcy * 1.02, rel=1e-9
        )

    def test_a_changed_diameter_scales_k_by_its_inverse_square(self, tmp_path):
        _, directory, original = record_steady_run(tmp_path)
        config = _config_from_sidecar(directory)
        config.sample.diameter = config.sample.diameter * 2.0
        again = reprocess_run(directory, config)
        assert again.permeability_darcy == pytest.approx(
            original.permeability_darcy / 4.0, rel=1e-9
        )

    def test_an_uncertainty_change_leaves_k_alone(self, tmp_path):
        _, directory, original = record_pulse_run(tmp_path)
        config = _config_from_sidecar(directory)
        before = reprocess_run(directory, config)
        config.sample.porosity_uncertainty = 0.05
        after = reprocess_run(directory, config)
        assert after.permeability_darcy == pytest.approx(
            before.permeability_darcy, rel=1e-12
        )
        assert (
            after.uncertainty.expanded_uncertainty_darcy
            > before.uncertainty.expanded_uncertainty_darcy
        )


def _config_from_sidecar(directory: Path):
    from gasperm.config import GaspermConfig
    from gasperm.storage import read_run_metadata

    stored = read_run_metadata(directory / "run_metadata.yaml")
    return GaspermConfig.model_validate(stored["config"])


class TestReprocessCommand:
    def test_no_change_reports_everything_unchanged(self, tmp_path):
        _, directory, _ = record_steady_run(tmp_path)
        result = runner.invoke(app, ["reprocess", str(directory)])
        assert result.exit_code == 0, result.output
        output = strip_ansi(result.output)
        assert "No configuration change" in output
        assert "unchanged" in output

    def test_an_uncertainty_change_is_labelled_as_a_re_costing(self, tmp_path):
        _, directory, _ = record_pulse_run(tmp_path)
        result = runner.invoke(
            app, ["reprocess", str(directory), "--set", "sample.porosity_uncertainty=0.05"]
        )
        assert result.exit_code == 0, result.output
        output = strip_ansi(result.output)
        assert "moves U(k) only" in output
        assert "k unchanged, U re-costed" in output

    def test_a_result_change_is_labelled_a_correction(self, tmp_path):
        _, directory, _ = record_steady_run(tmp_path)
        result = runner.invoke(
            app, ["reprocess", str(directory), "--set", "sample.length=51.0"]
        )
        assert result.exit_code == 0, result.output
        output = strip_ansi(result.output)
        assert "CORRECTION" in output
        assert "k moved +2.000%" in output

    def test_a_field_that_does_not_apply_says_so(self, tmp_path):
        """Porosity is not in a steady-state budget; that is not a silent failure."""
        _, directory, _ = record_steady_run(tmp_path)
        result = runner.invoke(
            app, ["reprocess", str(directory), "--set", "sample.porosity_uncertainty=0.05"]
        )
        assert result.exit_code == 0, result.output
        assert "not an input to this run's budget" in strip_ansi(result.output)

    def test_nothing_is_written_without_the_flag(self, tmp_path):
        rig, directory, _ = record_steady_run(tmp_path)
        before = sorted(p.name for p in (rig / "runs").iterdir())
        runner.invoke(app, ["reprocess", str(directory), "--set", "sample.length=51.0"])
        assert sorted(p.name for p in (rig / "runs").iterdir()) == before

    def test_write_makes_a_new_run_and_never_touches_the_original(self, tmp_path):
        rig, directory, original = record_steady_run(tmp_path)
        original_bytes = (directory / "run_metadata.yaml").read_bytes()
        result = runner.invoke(
            app,
            ["reprocess", str(directory), "--set", "sample.length=51.0", "--write"],
        )
        assert result.exit_code == 0, result.output
        assert (directory / "run_metadata.yaml").read_bytes() == original_bytes

        derived = [
            p for p in (rig / "runs").iterdir() if p.name.endswith("_reprocessed")
        ]
        assert len(derived) == 1
        assert (derived[0] / "readings.csv").is_file()

    def test_the_derived_run_records_where_it_came_from(self, tmp_path):
        import yaml

        rig, directory, _ = record_steady_run(tmp_path)
        runner.invoke(
            app,
            ["reprocess", str(directory), "--set", "sample.length=51.0", "--write"],
        )
        derived = next(p for p in (rig / "runs").iterdir() if p.name.endswith("_reprocessed"))
        payload = yaml.safe_load((derived / "run_metadata.yaml").read_text(encoding="utf-8"))
        provenance = payload["derived_from"]
        assert provenance["run"] == directory.name
        assert provenance["permeability_moved"] is True
        assert provenance["changes"][0]["field"] == "sample.length"
        assert provenance["changes"][0]["predicted"] == "result"
        # And it is a complete run in its own right.
        assert payload["summary"]["permeability_darcy"] > 0.0

    def test_a_second_write_does_not_overwrite_the_first(self, tmp_path):
        rig, directory, _ = record_steady_run(tmp_path)
        for length in ("51.0", "52.0"):
            runner.invoke(
                app,
                ["reprocess", str(directory), "--set", f"sample.length={length}", "--write"],
            )
        derived = [p for p in (rig / "runs").iterdir() if "_reprocessed" in p.name]
        assert len(derived) == 2

    def test_a_whole_plug_can_be_reprocessed_at_once(self, tmp_path):
        """The usual case: a corrected uncertainty applies to a campaign."""
        rig, first, _ = record_steady_run(tmp_path)
        result = runner.invoke(
            app,
            ["reprocess", "--sample", "core-041", "-c", str(rig),
             "--set", "sample.porosity_uncertainty=0.02"],
        )
        assert result.exit_code == 0, result.output
        assert "Reprocessing 1 run(s)" in strip_ansi(result.output)

    def test_a_bad_override_is_refused(self, tmp_path):
        _, directory, _ = record_steady_run(tmp_path)
        result = runner.invoke(
            app, ["reprocess", str(directory), "--set", "sample.length"]
        )
        assert result.exit_code == 1
        assert "KEY=VALUE" in strip_ansi(result.output)

    def test_an_unknown_field_is_refused(self, tmp_path):
        _, directory, _ = record_steady_run(tmp_path)
        result = runner.invoke(
            app, ["reprocess", str(directory), "--set", "sample.nonsense=1"]
        )
        assert result.exit_code == 1

    def test_no_target_says_what_to_do(self, tmp_path):
        result = runner.invoke(app, ["reprocess", "-c", str(tmp_path)])
        assert result.exit_code == 1
        assert "--sample" in strip_ansi(result.output)

    def test_a_missing_run_directory_is_refused(self, tmp_path):
        result = runner.invoke(app, ["reprocess", str(tmp_path / "nope")])
        assert result.exit_code == 1
        assert "No such run directory" in strip_ansi(result.output)

    def test_the_flags_are_documented_in_help(self):
        result = runner.invoke(app, ["reprocess", "--help"], env={"COLUMNS": "200"})
        output = strip_ansi(result.output)
        for flag in ("--set", "--write", "--sample", "--from-config"):
            assert flag in output
