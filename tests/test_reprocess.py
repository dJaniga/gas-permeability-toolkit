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
from typing import ClassVar
from unittest import mock

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
    verify_run,
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
    config.sample.porosity = 0.10

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


def add_plug(rig: Path, sample_id: str, *overrides: str) -> Path:
    """A second (or third) core plug in an existing rig folder."""
    args = ["new-sample", sample_id, "--dir", str(rig / "samples"), "-n", "--force"]
    for override in overrides:
        args += ["--set", override]
    assert runner.invoke(app, args).exit_code == 0, sample_id
    return rig / "samples" / f"{sample_id}.yaml"


def record_run_for(rig: Path, sample_path: Path, *, started_at=None) -> Path:
    """One steady-state run for one plug, into the rig's own runs directory."""
    from datetime import datetime, timezone

    config = load_config(rig, sample=sample_path)
    config.run.steady_state.window_s = 1.0
    config.run.steady_state.required_windows = 2
    config.run.steady_state.min_samples = 3
    config.hardware.daq.sample_rate_hz = 200.0
    config.run.max_samples = 700
    config.sample.porosity = 0.10

    provider = build_provider(config.run.gas)
    loop = AcquisitionLoop(
        config, SampleProcessor(config, provider),
        FakeAnalogSource({"ai0": 0.30, "ai1": 0.05, "ai2": 2.0}),
        FakeTemperatureSource(22.0),
    )
    loop.run(install_signal_handler=False)
    writer = RunWriter(
        config, started_at=started_at or datetime.now(timezone.utc)
    )
    writer.open()
    for reading in loop.readings:
        writer.write(reading)
    writer.close()
    writer.write_metadata(loop.summarize(csv_path=str(writer.readings_path)))
    return writer.directory


def record_two_plugs(tmp_path, second_length: float = 50.0) -> Path:
    """A rig whose runs directory holds one run each for two different plugs."""
    from datetime import datetime, timedelta, timezone

    rig = make_rig(tmp_path)
    base = datetime(2026, 5, 1, 9, tzinfo=timezone.utc)
    record_run_for(rig, rig / "samples" / "core-041.yaml", started_at=base)
    # `new-sample --set` keys are relative to the sample file, which *is* the
    # sample section -- unlike `reprocess --set`, which addresses a whole config.
    second = add_plug(rig, "core-042", f"length={second_length}")
    record_run_for(rig, second, started_at=base + timedelta(hours=1))
    return rig


def record_pulse_run(tmp_path) -> tuple[Path, Path, object]:
    """A well-conditioned pulse decay, sampled at the rate its times imply."""
    rig = make_rig(tmp_path, "run.method=pulse_decay")
    config = load_config(rig, sample=rig / "samples" / "core-041.yaml")
    config.hardware.daq.sample_rate_hz = 20.0
    for side in ("upstream", "downstream"):
        getattr(config.hardware.reservoirs, side).vessel = 8.0
        getattr(config.hardware.reservoirs, side).dead = 0.0
    config.sample.porosity = 0.10
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
        for flag in ("--set", "--write", "--sample", "--from-config", "--all"):
            assert flag in output


class TestNoChangeReproduces:
    """A re-derivation with nothing changed must reproduce the original.

    This is the baseline every other use of reprocess rests on: if a no-op
    replay moves the answer, no reported change can be attributed to the field
    that was actually edited. Both methods reconstruct *derived* state -- the
    steady window, the pulse -- and both used to rebuild it from an object that
    had never seen the run.
    """

    def replay(self, directory, live):
        from gasperm.cli import _stored_config
        from gasperm.storage import _record_from_directory

        return reprocess_run(
            directory, _stored_config(_record_from_directory(directory)),
            started_at=live.started_at, ended_at=live.ended_at,
        )

    def drifting_run(self, tmp_path):
        """A run that is still moving when it stops -- warned about, not rare."""
        rig = make_rig(tmp_path)
        config = load_config(rig, sample=rig / "samples" / "core-041.yaml")
        config.run.steady_state.window_s = 1.0
        config.run.steady_state.required_windows = 2
        config.run.steady_state.min_samples = 3
        config.hardware.daq.sample_rate_hz = 20.0
        frames = [{"ai0": 0.30, "ai1": 0.05, "ai2": 2.0} for _ in range(400)]
        frames += [
            {"ai0": 0.30, "ai1": 0.05, "ai2": 2.0 * (1 + 0.002 * (i + 1))}
            for i in range(200)
        ]
        config.run.max_samples = len(frames)
        loop = AcquisitionLoop(
            config, SampleProcessor(config, build_provider(config.run.gas)),
            FakeAnalogSource(frames), FakeTemperatureSource(22.0),
        )
        loop.run(install_signal_handler=False)
        writer = RunWriter(config)
        writer.open()
        for reading in loop.readings:
            writer.write(reading)
        writer.close()
        live = loop.summarize(csv_path=str(writer.readings_path))
        writer.write_metadata(live)
        return loop, writer.directory, live

    def test_a_run_that_ended_while_drifting_keeps_its_plateau(self, tmp_path):
        """The replay must not summarise the drifting tail in the plateau's place.

        Reading the detector's state *after* the last row loses the window
        entirely for such a run -- it has left steady state by then -- and the
        fallback averages the trailing seconds, which are exactly the samples
        the live run excluded.
        """
        loop, directory, live = self.drifting_run(tmp_path)
        assert loop.ended_unsteady, "the fixture must actually leave steady state"
        again = self.replay(directory, live)
        assert again.steady_state_window is not None
        assert (
            again.steady_state_window.start_index,
            again.steady_state_window.end_index,
        ) == (
            live.steady_state_window.start_index,
            live.steady_state_window.end_index,
        )
        assert again.permeability_darcy == pytest.approx(
            live.permeability_darcy, rel=1e-9
        )

    def test_such_a_run_does_not_lose_its_confirmation(self, tmp_path):
        """The more damaging half: an unconfirmed re-derivation supersedes its
        parent and is then excluded from klinkenberg and counted as a failure."""
        _, directory, live = self.drifting_run(tmp_path)
        assert live.measurement_confirmed
        assert self.replay(directory, live).measurement_confirmed

    def test_a_pulse_run_keeps_its_pulse(self, tmp_path):
        """dP0 and the pulse instant come from the recorded differential now.

        They used to come from the live monitor, which on a replay has seen no
        samples -- so they fell back to the fit's extrapolated amplitude and to
        t = 0.
        """
        _, directory, live = record_pulse_run(tmp_path)
        again = self.replay(directory, live)
        before, after = live.pulse_decay, again.pulse_decay
        assert after.pulse_amplitude_atm == pytest.approx(
            before.pulse_amplitude_atm, rel=1e-6
        )
        assert after.pulse_at_elapsed_s == pytest.approx(before.pulse_at_elapsed_s)

    def test_a_pulse_run_keeps_its_setup_condition(self, tmp_path):
        """With t = 0 this read the *pre-pulse* equilibrium -- both vessels at
        one pressure and no pulse -- which is what an operator would have tried
        to set the rig back to."""
        _, directory, live = record_pulse_run(tmp_path)
        after = self.replay(directory, live).pulse_decay
        before = live.pulse_decay
        assert after.initial_upstream_pressure_atm == pytest.approx(
            before.initial_upstream_pressure_atm, rel=1e-6
        )
        assert after.initial_downstream_pressure_atm == pytest.approx(
            before.initial_downstream_pressure_atm, rel=1e-6
        )
        # And they still bracket the pulse, rather than collapsing together.
        span = (
            after.initial_upstream_pressure_atm - after.initial_downstream_pressure_atm
        )
        assert span == pytest.approx(after.pulse_amplitude_atm, rel=0.02)

    def test_the_command_reports_no_movement(self, tmp_path):
        """What the operator actually sees for a no-change reprocess."""
        _, directory, _ = self.drifting_run(tmp_path)
        result = runner.invoke(app, ["reprocess", str(directory)])
        assert result.exit_code == 0, result.output
        output = strip_ansi(result.output)
        assert "No configuration change" in output
        assert "k moved" not in output


class TestVerify:
    """``--verify``: does a run re-derive to what is stored, and if not, where?

    Three bugs in this area reached a rig before being noticed, each one a
    no-change replay quietly moving ``k``. The size of the difference was never
    the useful part -- the *stage* it appeared at was, because the stages fail
    for unrelated reasons and only one of them is about the physics.
    """

    def stored_config(self, directory):
        from gasperm.cli import _stored_config
        from gasperm.storage import _record_from_directory

        return _stored_config(_record_from_directory(directory))

    def test_a_healthy_steady_run_reproduces(self, tmp_path):
        _, directory, _ = record_steady_run(tmp_path)
        report = verify_run(directory, self.stored_config(directory))
        assert report.reproduces, report.diagnosis()

    def test_a_healthy_pulse_run_reproduces(self, tmp_path):
        _, directory, _ = record_pulse_run(tmp_path)
        report = verify_run(directory, self.stored_config(directory))
        assert report.reproduces, report.diagnosis()

    def test_a_run_that_ended_drifting_reproduces(self, tmp_path):
        """The case that did not, before the window fix."""
        _, directory, _ = TestNoChangeReproduces().drifting_run(tmp_path)
        report = verify_run(directory, self.stored_config(directory))
        assert report.reproduces, report.diagnosis()

    def test_csv_rounding_is_not_reported_as_drift(self, tmp_path):
        """A stored bound is a full-precision float; a replayed one comes back
        through four decimals of elapsed_s. Comparing those relatively -- near
        t = 0 especially -- reports every healthy run as broken."""
        _, directory, _ = record_steady_run(tmp_path)
        report = verify_run(directory, self.stored_config(directory))
        assert report.stored_window != report.replayed_window   # they do differ
        assert report.windows_agree                             # ...but not meaningfully

    def test_a_lost_window_is_named_as_the_window(self, tmp_path):
        """Localisation is the feature: exact samples, wrong reduction."""
        _, directory, _ = TestNoChangeReproduces().drifting_run(tmp_path)

        def lost(samples, config, *, time_key="elapsed_s"):
            return None

        with mock.patch("gasperm.steady_state.detect_steady_window", lost):
            report = verify_run(directory, self.stored_config(directory))
        assert not report.reproduces
        assert report.samples_agree
        assert not report.windows_agree
        assert "averaged window is not the stored one" in report.diagnosis()

    def test_a_per_sample_difference_is_named_as_the_samples(self, tmp_path):
        """The other stage, so a real calibration drift is not blamed on the
        window -- they need entirely different fixes."""
        _, directory, _ = record_steady_run(tmp_path)
        config = self.stored_config(directory)
        config.sample.length = config.sample.length * 1.05  # moves every sample's k
        report = verify_run(directory, config)
        assert not report.reproduces
        assert not report.samples_agree
        assert "per-sample derivation differs" in report.diagnosis()

    def test_the_command_exits_two_when_a_run_does_not_reproduce(self, tmp_path):
        """A distinct code, so a scripted check over a directory can gate on it."""
        _, directory, _ = TestNoChangeReproduces().drifting_run(tmp_path)

        with mock.patch(
            "gasperm.steady_state.detect_steady_window",
            lambda *a, **k: None,
        ):
            result = runner.invoke(app, ["reprocess", "--verify", str(directory)])
        assert result.exit_code == 2
        output = strip_ansi(result.output)
        assert "DOES NOT REPRODUCE" in output
        assert "averaged window" in output

    def test_the_command_is_quiet_and_clean_when_all_reproduce(self, tmp_path):
        _, directory, _ = record_steady_run(tmp_path)
        result = runner.invoke(app, ["reprocess", "--verify", str(directory)])
        assert result.exit_code == 0, result.output
        assert "All runs reproduce" in strip_ansi(result.output)

    def test_it_writes_nothing(self, tmp_path):
        rig, directory, _ = record_steady_run(tmp_path)
        before = sorted(p.name for p in (rig / "runs").iterdir())
        runner.invoke(app, ["reprocess", "--verify", str(directory)])
        assert sorted(p.name for p in (rig / "runs").iterdir()) == before

    def drifted_by(self, directory, relative):
        """A config that moves every sample's k by ``relative``, via geometry."""
        config = self.stored_config(directory)
        config.sample.length = config.sample.length * (1 + relative)
        return config

    def test_the_threshold_decides_the_verdict(self, tmp_path):
        """The point of making it settable: the same drift, judged two ways."""
        _, directory, _ = record_steady_run(tmp_path)
        config = self.drifted_by(directory, 3e-6)
        assert not verify_run(directory, config, tolerance=1e-6).reproduces
        assert verify_run(directory, config, tolerance=1e-5).reproduces

    def test_the_report_carries_the_threshold_it_was_judged_at(self, tmp_path):
        """A pass means nothing without it, so nothing has to look it up."""
        _, directory, _ = record_steady_run(tmp_path)
        assert verify_run(directory, self.stored_config(directory),
                          tolerance=1e-5).tolerance == 1e-5

    def test_the_default_is_unchanged(self, tmp_path):
        from gasperm.reprocess import _MOVED_TOLERANCE

        _, directory, _ = record_steady_run(tmp_path)
        report = verify_run(directory, self.stored_config(directory))
        assert report.tolerance == _MOVED_TOLERANCE

    def test_a_non_positive_threshold_is_refused(self, tmp_path):
        """It would fail every run on the last bit of a float and localise nothing."""
        _, directory, _ = record_steady_run(tmp_path)
        for bad in (0.0, -1e-6):
            with pytest.raises(ValueError, match="must be positive"):
                verify_run(directory, self.stored_config(directory), tolerance=bad)

    def test_the_command_states_the_threshold(self, tmp_path):
        _, directory, _ = record_steady_run(tmp_path)
        result = runner.invoke(
            app, ["reprocess", "--verify", str(directory), "--tolerance", "1e-5"]
        )
        assert result.exit_code == 0, result.output
        assert "tolerance 1e-05" in strip_ansi(result.output)

    def test_the_command_refuses_a_non_positive_threshold(self, tmp_path):
        _, directory, _ = record_steady_run(tmp_path)
        result = runner.invoke(
            app, ["reprocess", "--verify", str(directory), "--tolerance", "0"]
        )
        assert result.exit_code == 1
        assert "must be positive" in strip_ansi(result.output)

    def test_the_threshold_without_verify_is_refused(self, tmp_path):
        """Rather than silently doing nothing, which is how a check gets trusted
        that was never applied."""
        _, directory, _ = record_steady_run(tmp_path)
        result = runner.invoke(
            app, ["reprocess", str(directory), "--tolerance", "1e-5"]
        )
        assert result.exit_code == 1
        assert "only applies to --verify" in strip_ansi(result.output)

    def test_a_moved_uncertainty_is_caught_and_named(self, tmp_path):
        """k and U(k) move independently: a re-costing bug leaves k exactly
        where it was, and a check on k alone would pass it."""
        _, directory, _ = record_pulse_run(tmp_path)
        config = self.stored_config(directory)
        config.sample.porosity_uncertainty = (config.sample.porosity_uncertainty or 0.01) * 4
        report = verify_run(directory, config)
        assert not report.reproduces
        assert report.samples_agree
        assert not report.uncertainty_agrees
        assert "U(k) does not" in report.diagnosis()

    def test_it_verifies_a_whole_directory(self, tmp_path):
        rig = record_two_plugs(tmp_path)
        result = runner.invoke(
            app, ["reprocess", "--verify", "--all", "-c", str(rig)]
        )
        assert result.exit_code == 0, result.output
        assert "Verifying 2 run(s)" in strip_ansi(result.output)


class TestReprocessAll:
    """``--all``: the whole runs directory, for a rig-level correction.

    A recalibrated transducer or a re-measured vessel applies to everything the
    bench recorded, not to one plug -- and reprocessing plug by plug is both
    tedious and easy to leave half-done.
    """

    def run(self, rig, *args):
        return runner.invoke(app, ["reprocess", "--all", "-c", str(rig), *args])

    def derived(self, rig):
        return sorted(
            p.name for p in (rig / "runs").iterdir() if "_reprocessed" in p.name
        )

    def test_it_takes_every_plug(self, tmp_path):
        rig = record_two_plugs(tmp_path)
        result = self.run(rig)
        assert result.exit_code == 0, result.output
        output = strip_ansi(result.output)
        assert "2 run(s) across 2 plug(s)" in output
        assert "Reprocessing 2 run(s)" in output

    def test_each_run_keeps_its_own_plug(self, tmp_path):
        """The reason --all is safe: every run re-derives from its own snapshot.

        A batch that applied one plug's geometry to the rest would silently
        corrupt every core but the first, and each result would still look
        internally consistent.
        """
        import yaml

        rig = record_two_plugs(tmp_path, second_length=51.0)
        assert self.run(rig, "--write").exit_code == 0
        lengths = {}
        for name in self.derived(rig):
            payload = yaml.safe_load(
                (rig / "runs" / name / "run_metadata.yaml").read_text(encoding="utf-8")
            )
            lengths[payload["config"]["sample"]["id"]] = payload["config"]["sample"]["length"]
        assert lengths == {"core-041": 50.0, "core-042": 51.0}

    def test_write_derives_one_run_per_original(self, tmp_path):
        rig = record_two_plugs(tmp_path)
        assert self.run(rig, "--set", "sample.porosity_uncertainty=0.01", "--write").exit_code == 0
        assert len(self.derived(rig)) == 2

    def test_a_second_pass_skips_what_it_already_superseded(self, tmp_path):
        """Otherwise one parent gets two children, and `drop_superseded` keeps
        both -- putting one experiment into a regression twice, which is the
        exact thing supersession exists to prevent."""
        rig = record_two_plugs(tmp_path)
        assert self.run(rig, "--set", "sample.porosity_uncertainty=0.01", "--write").exit_code == 0
        first = self.derived(rig)

        result = self.run(rig, "--set", "sample.porosity_uncertainty=0.02", "--write")
        assert result.exit_code == 0, result.output
        output = strip_ansi(result.output)
        assert "superseded by" in output
        # Two more derived runs, each a child of a child -- not of the originals.
        assert len(self.derived(rig)) == 4
        assert all(name in self.derived(rig) for name in first)
        assert "Reprocessing 2 run(s)" in output

    def test_every_experiment_still_reduces_once_afterwards(self, tmp_path):
        """The property the skipping protects, asserted where it is observable."""
        from gasperm.storage import drop_superseded, find_runs

        rig = record_two_plugs(tmp_path)
        for value in ("0.01", "0.02"):
            assert self.run(
                rig, "--set", f"sample.porosity_uncertainty={value}", "--write"
            ).exit_code == 0
        kept, _ = drop_superseded(find_runs(rig / "runs"))
        assert len(kept) == 2
        assert {record.sample_id for record in kept} == {"core-041", "core-042"}

    def test_a_change_that_hit_only_some_runs_says_so(self, tmp_path):
        """Each run diffs against its own snapshot, so a batch's changes vary."""
        rig = record_two_plugs(tmp_path, second_length=51.0)
        result = self.run(rig, "--set", "sample.length=51.0")
        assert result.exit_code == 0, result.output
        assert "(1 of 2 runs)" in strip_ansi(result.output)

    def test_a_change_that_hit_every_run_is_not_annotated(self, tmp_path):
        rig = record_two_plugs(tmp_path)
        result = self.run(rig, "--set", "sample.length=52.0")
        output = strip_ansi(result.output)
        assert "sample.length" in output
        assert "of 2 runs" not in output

    def test_per_plug_overrides_are_warned_about(self, tmp_path):
        """A rig-level correction is the point; a plug-level one is a mistake."""
        rig = record_two_plugs(tmp_path)
        output = strip_ansi(self.run(rig, "--set", "sample.length=52.0").output)
        assert "will be applied to all 2 plug(s)" in output

    def test_a_rig_level_override_is_not_warned_about(self, tmp_path):
        rig = record_two_plugs(tmp_path)
        output = strip_ansi(
            self.run(rig, "--set", "run.uncertainty.coverage_factor=2.5").output
        )
        assert "will be applied to all" not in output

    def test_a_batch_wide_sample_field_is_not_warned_about(self, tmp_path):
        """`porosity_uncertainty` describes the *method*, not the core, and
        applying it across a batch measured the same way is what --all is for.
        A warning that fires on the correct case is one people click past."""
        rig = record_two_plugs(tmp_path)
        output = strip_ansi(
            self.run(rig, "--set", "sample.porosity_uncertainty=0.01").output
        )
        assert "will be applied to all" not in output

    def test_it_refuses_a_sample_file(self, tmp_path):
        """It replaces the whole sample section -- one plug's id onto every core."""
        rig = record_two_plugs(tmp_path)
        result = self.run(rig, "--sample-file", str(rig / "samples" / "core-041.yaml"))
        assert result.exit_code == 1
        assert "describes one plug" in strip_ansi(result.output)

    def test_it_refuses_a_competing_scope(self, tmp_path):
        rig = record_two_plugs(tmp_path)
        for extra in (["--sample", "core-041"], [str(rig / "runs")]):
            result = runner.invoke(
                app, ["reprocess", "--all", "-c", str(rig), *extra]
            )
            assert result.exit_code == 1
            assert "already means every run" in strip_ansi(result.output)

    def test_an_empty_runs_directory_says_so(self, tmp_path):
        rig = make_rig(tmp_path)
        (rig / "runs").mkdir(exist_ok=True)
        result = self.run(rig)
        assert result.exit_code == 1
        assert "No runs found" in strip_ansi(result.output)


class _RecordingExecutor:
    """A pool that runs everything here, and remembers the submission order.

    Stands in for ``ProcessPoolExecutor`` so the batch layer's own promises --
    payload order out, exceptions carried rather than raised, longest run
    submitted first -- can be pinned without paying for real interpreters. The
    one thing it cannot exercise is pickling, which the end-to-end ``-j`` tests
    do.
    """

    #: Payloads in the order they were handed out, newest batch only.
    submitted: ClassVar[list] = []

    def __init__(self, max_workers=None):
        self.max_workers = max_workers
        type(self).submitted = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def submit(self, fn, *args):
        from concurrent.futures import Future

        type(self).submitted.append(args[0])
        future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as exc:  # noqa: BLE001 -- exactly what a worker would do
            future.set_exception(exc)
        return future


@pytest.fixture
def in_process_pool(monkeypatch):
    """Run the parallel path in this process, completing in **reverse** order.

    Reverse specifically: results must come back in the caller's order, and a
    stand-in that happened to complete in order would let a reversed one
    through.
    """
    from gasperm import reprocess as module

    monkeypatch.setattr(module, "ProcessPoolExecutor", _RecordingExecutor)
    monkeypatch.setattr(module, "as_completed", lambda futures: reversed(list(futures)))
    return _RecordingExecutor


def _job_for(directory: Path):
    from gasperm.reprocess import ReprocessJob

    return ReprocessJob(directory=directory, config=_config_from_sidecar(directory))


class TestWorkerCount:
    """How many processes a batch gets, and why it is sometimes one.

    A worker costs an interpreter start plus CoolProp and SciPy imports. Paying
    that to re-derive a handful of short runs makes the command slower, which is
    the opposite of the point.
    """

    def test_a_small_batch_stays_in_this_process(self, tmp_path):
        from gasperm.reprocess import resolve_workers

        _, directory, _ = record_steady_run(tmp_path)
        job = _job_for(directory)
        assert resolve_workers(None, [job, job]) == 1

    def test_a_large_batch_spreads_over_the_cpus(self, tmp_path, monkeypatch):
        import os

        from gasperm import reprocess as module

        _, directory, _ = record_steady_run(tmp_path)
        monkeypatch.setattr(
            module, "_record_size", lambda job: module._PARALLEL_MIN_BYTES
        )
        jobs = [_job_for(directory)] * 4
        assert module.resolve_workers(None, jobs) == min(os.cpu_count() or 1, 4)

    def test_it_never_asks_for_more_workers_than_runs(self, tmp_path):
        from gasperm.reprocess import resolve_workers

        _, directory, _ = record_steady_run(tmp_path)
        job = _job_for(directory)
        assert resolve_workers(64, [job, job]) == 2

    def test_an_explicit_count_is_honoured_whatever_the_size(self, tmp_path):
        """--jobs is an instruction, not a hint: a small batch still gets it."""
        from gasperm.reprocess import resolve_workers

        _, directory, _ = record_steady_run(tmp_path)
        job = _job_for(directory)
        assert resolve_workers(2, [job, job]) == 2

    def test_minus_one_means_every_cpu(self, tmp_path):
        """joblib's convention, because it is the one people arrive with."""
        import os

        from gasperm.reprocess import resolve_workers

        _, directory, _ = record_steady_run(tmp_path)
        jobs = [_job_for(directory)] * 64
        assert resolve_workers(-1, jobs) == (os.cpu_count() or 1)

    def test_minus_two_holds_one_core_back(self, tmp_path):
        """The useful form: the machine stays usable while the batch runs."""
        import os

        from gasperm.reprocess import resolve_workers

        _, directory, _ = record_steady_run(tmp_path)
        jobs = [_job_for(directory)] * 64
        assert resolve_workers(-2, jobs) == max(1, (os.cpu_count() or 1) - 1)

    def test_a_negative_past_the_core_count_clamps(self, tmp_path):
        """-32 on an eight-core box means "as few as possible", not an error."""
        from gasperm.reprocess import resolve_workers

        _, directory, _ = record_steady_run(tmp_path)
        assert resolve_workers(-999, [_job_for(directory)] * 4) == 1

    def test_no_jobs_at_all_is_refused(self):
        """Zero is the one integer that names no workers rather than some."""
        from gasperm.reprocess import resolve_workers

        with pytest.raises(ValueError, match="must not be 0"):
            resolve_workers(0, [])

    def test_the_command_refuses_no_jobs(self, tmp_path):
        _, directory, _ = record_steady_run(tmp_path)
        result = runner.invoke(app, ["reprocess", str(directory), "-j", "0"])
        assert result.exit_code == 1
        assert "must not be 0" in strip_ansi(result.output)

    def test_the_command_takes_a_negative_count(self, tmp_path):
        _, directory, _ = record_steady_run(tmp_path)
        result = runner.invoke(app, ["reprocess", str(directory), "-j", "-2"])
        assert result.exit_code == 0, result.output
        assert "Reprocessing 1 run(s)" in strip_ansi(result.output)


class TestBatchScheduling:
    """What the batch layer promises whatever order the pool works in."""

    def test_results_come_back_in_the_order_given(self, tmp_path, in_process_pool):
        """Completion order is the scheduler's; the report is keyed by run."""
        from gasperm.reprocess import reprocess_batch
        from gasperm.storage import find_runs

        rig = record_two_plugs(tmp_path, second_length=51.0)
        records = sorted(find_runs(rig / "runs"), key=lambda record: record.name)
        jobs = [_job_for(record.directory) for record in records]
        results = reprocess_batch(jobs, workers=2)
        assert [result.sample_id for result in results] == [
            job.config.sample.id for job in jobs
        ]

    def test_the_longest_run_is_submitted_first(self, tmp_path, monkeypatch):
        """Otherwise the pool goes idle waiting on the run it started last."""
        from gasperm import reprocess as module

        _, directory, _ = record_steady_run(tmp_path)
        config = _config_from_sidecar(directory)
        jobs = [
            module.ReprocessJob(directory=Path(name), config=config)
            for name in ("short", "long", "middling")
        ]
        sizes = {"short": 10, "long": 300, "middling": 100}
        monkeypatch.setattr(
            module, "_record_size", lambda job: sizes[job.directory.name]
        )
        assert module._longest_first(jobs) == [1, 2, 0]

    def test_that_order_is_the_order_it_submits_in(self, tmp_path, monkeypatch, in_process_pool):
        from gasperm import reprocess as module

        _, directory, _ = record_steady_run(tmp_path)
        config = _config_from_sidecar(directory)
        jobs = [
            module.ReprocessJob(directory=Path(name), config=config)
            for name in ("short", "long", "middling")
        ]
        sizes = {"short": 10, "long": 300, "middling": 100}
        monkeypatch.setattr(
            module, "_record_size", lambda job: sizes[job.directory.name]
        )
        module.reprocess_batch(jobs, workers=2)
        assert [job.directory.name for job in in_process_pool.submitted] == [
            "long", "middling", "short",
        ]

    def test_a_run_that_fails_does_not_cost_the_others(self, tmp_path, in_process_pool):
        """One unreadable CSV must not take forty good runs with it."""
        from gasperm.reprocess import ReprocessJob, reprocess_batch

        _, directory, _ = record_steady_run(tmp_path)
        config = _config_from_sidecar(directory)
        jobs = [
            _job_for(directory),
            ReprocessJob(directory=tmp_path / "nowhere", config=config),
            _job_for(directory),
        ]
        results = reprocess_batch(jobs, workers=2)
        assert results[0].permeability_darcy is not None
        assert isinstance(results[1], (ValueError, OSError))
        assert results[2].permeability_darcy is not None

    def test_a_broken_pool_finishes_the_batch_here(self, tmp_path, monkeypatch):
        """A killed worker says nothing about whether the work can be done."""
        from gasperm import reprocess as module

        class _Broken:
            def __init__(self, max_workers=None):
                raise OSError("no processes available")

        monkeypatch.setattr(module, "ProcessPoolExecutor", _Broken)
        monkeypatch.setattr(
            module, "_record_size", lambda job: module._PARALLEL_MIN_BYTES
        )
        _, directory, _ = record_steady_run(tmp_path)
        job = _job_for(directory)
        results = module.reprocess_batch([job, job])
        assert all(result.permeability_darcy is not None for result in results)

    def test_progress_counts_every_run(self, tmp_path, in_process_pool):
        from gasperm.reprocess import reprocess_batch

        _, directory, _ = record_steady_run(tmp_path)
        job = _job_for(directory)
        seen = []
        reprocess_batch(
            [job, job, job], workers=2,
            on_done=lambda done, total: seen.append((done, total)),
        )
        assert seen == [(1, 3), (2, 3), (3, 3)]


class TestParallelMatchesSerial:
    """The only thing that finally matters: workers change speed, not answers.

    Run through real worker processes rather than the in-process stand-in,
    because what these catch is a config or a summary that does not survive the
    trip to a worker and back.
    """

    def test_the_report_is_the_same_either_way(self, tmp_path):
        rig = record_two_plugs(tmp_path, second_length=51.0)
        serial = runner.invoke(app, ["reprocess", "--all", "-c", str(rig), "-j", "1"])
        parallel = runner.invoke(app, ["reprocess", "--all", "-c", str(rig), "-j", "2"])
        assert serial.exit_code == 0, serial.output
        assert parallel.exit_code == 0, parallel.output
        assert strip_ansi(parallel.output) == strip_ansi(serial.output)

    def test_verify_is_the_same_either_way(self, tmp_path):
        rig = record_two_plugs(tmp_path, second_length=51.0)
        serial = runner.invoke(
            app, ["reprocess", "--verify", "--all", "-c", str(rig), "-j", "1"]
        )
        parallel = runner.invoke(
            app, ["reprocess", "--verify", "--all", "-c", str(rig), "-j", "2"]
        )
        assert serial.exit_code == 0, serial.output
        assert strip_ansi(parallel.output) == strip_ansi(serial.output)
