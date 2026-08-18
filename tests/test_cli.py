"""CLI surface: the multi-plug and multi-meter workflows.

Exercises the real commands through typer's runner. Hardware is never touched:
``collect`` is covered by the acquisition tests, so these focus on the parts an
operator drives between runs -- adding plugs and choosing a meter.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest
import yaml
from typer.testing import CliRunner

from gasperm import cli, units
from gasperm.cli import app
from gasperm.config import (
    HARDWARE_FILENAME,
    RUN_FILENAME,
    SAMPLE_FILENAME,
    ConfigError,
    load_config,
)
from gasperm.config.sample import SampleConfig

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Drop colour escapes so assertions see the words, not the styling."""
    return _ANSI.sub("", text)


def init_config(directory, *overrides: str):
    """Write the rig and experiment files into ``directory``."""
    args = ["init", str(directory), "--non-interactive", "--force"]
    for override in overrides:
        args += ["--set", override]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return result


def add_sample(directory, sample_id: str = "core-001", *overrides: str):
    """Create one core plug's file and return its path."""
    args = ["new-sample", sample_id, "--dir", str(directory), "-n", "--force"]
    for override in overrides:
        args += ["--set", override]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return Path(directory) / f"{sample_id}.yaml"


def load(directory, sample_path=None):
    """Load a full config: rig and run from ``directory``, plus a sample."""
    if sample_path is None:
        sample_path = add_sample(Path(directory) / "samples")
    return load_config(directory, sample=sample_path)


def read_sample(path) -> SampleConfig:
    return SampleConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


class TestInitFolder:
    """The folder is named by the caller and created by init."""

    def test_creates_the_named_folder(self, tmp_path):
        rig = tmp_path / "tight-gas-rig"
        assert not rig.exists()
        init_config(rig)
        assert rig.is_dir()
        assert (rig / HARDWARE_FILENAME).is_file()
        assert (rig / RUN_FILENAME).is_file()

    def test_creates_nested_parents(self, tmp_path):
        rig = tmp_path / "lab" / "benches" / "rig-2"
        init_config(rig)
        assert (rig / HARDWARE_FILENAME).is_file()

    def test_creates_a_home_for_the_plugs(self, tmp_path):
        rig = tmp_path / "rig"
        init_config(rig)
        assert (rig / "samples").is_dir()

    def test_runs_default_inside_the_rig_folder(self, tmp_path):
        """A bench's config, plugs and measurements stay together."""
        rig = tmp_path / "rig"
        init_config(rig)
        # Written relative to the rig folder, not to whatever directory init
        # was invoked from -- the folder name must not be baked in.
        assert load(rig).run.output_dir == cli.DEFAULT_OUTPUT_DIR == "./runs"

    def test_the_runs_directory_does_not_move_with_the_working_directory(self, tmp_path):
        """The bug this replaced: './rig/runs' became 'rig/rig/runs' after a cd."""
        rig = tmp_path / "rig"
        init_config(rig)
        expected = (rig / "runs").resolve()
        assert load(rig).resolved_output_dir().resolve() == expected

        import os

        previous = os.getcwd()
        os.chdir(rig)
        try:
            assert load(Path(".")).resolved_output_dir().resolve() == expected
        finally:
            os.chdir(previous)

    def test_an_absolute_output_dir_is_left_alone(self, tmp_path):
        rig = tmp_path / "rig"
        absolute = (tmp_path / "elsewhere").as_posix()
        init_config(rig, f"run.output_dir={absolute}")
        assert load(rig).resolved_output_dir() == Path(absolute)

    def test_an_explicit_output_dir_still_wins(self, tmp_path):
        rig = tmp_path / "rig"
        init_config(rig, "run.output_dir=/data/runs")
        assert load(rig).run.output_dir == "/data/runs"

    def test_the_folder_name_is_required(self, tmp_path):
        result = runner.invoke(app, ["init", "--non-interactive"])
        assert result.exit_code != 0
        assert "FOLDER" in result.output or "Missing argument" in result.output

    def test_print_only_creates_nothing(self, tmp_path):
        rig = tmp_path / "rig"
        result = runner.invoke(
            app, ["init", str(rig), "--non-interactive", "--print"]
        )
        assert result.exit_code == 0, result.output
        assert not rig.exists()

    def test_the_next_steps_name_the_folder(self, tmp_path):
        rig = tmp_path / "rig"
        result = init_config(rig)
        assert f"{rig.as_posix()}/samples" in result.output

    def test_re_running_into_an_existing_folder_needs_force(self, tmp_path):
        rig = tmp_path / "rig"
        init_config(rig)
        result = runner.invoke(app, ["init", str(rig), "--non-interactive"])
        assert result.exit_code == 1
        assert "--force" in result.output


class TestInit:
    def test_writes_the_rig_and_the_experiment(self, tmp_path):
        init_config(tmp_path)
        assert (tmp_path / HARDWARE_FILENAME).is_file()
        assert (tmp_path / RUN_FILENAME).is_file()

    def test_does_not_write_a_sample_file(self, tmp_path):
        """A sample describes one plug; a rig measures many, so init stays out."""
        init_config(tmp_path)
        assert not (tmp_path / SAMPLE_FILENAME).exists()

    def test_points_at_new_sample_next(self, tmp_path):
        result = init_config(tmp_path)
        assert "new-sample" in result.output

    def test_overrides_reach_the_rig_and_the_run(self, tmp_path):
        init_config(
            tmp_path, "hardware.daq.device_name=Dev3", "run.operator=Damian"
        )
        config = load(tmp_path)
        assert config.hardware.daq.device_name == "Dev3"
        assert config.run.operator == "Damian"

    def test_a_typo_in_an_override_fails_loudly(self, tmp_path):
        result = runner.invoke(
            app,
            ["init", str(tmp_path), "--non-interactive", "--force", "--set", "run.oprator=X"],
        )
        assert result.exit_code == 1
        assert "not a valid" in result.output or "oprator" in result.output

    def test_refuses_to_overwrite_without_force(self, tmp_path):
        init_config(tmp_path)
        result = runner.invoke(app, ["init", str(tmp_path), "--non-interactive"])
        assert result.exit_code == 1
        assert "--force" in result.output

    def test_an_existing_sample_file_is_not_disturbed(self, tmp_path):
        """init --force must not clobber a plug's file that happens to sit there."""
        sample = add_sample(tmp_path, "core-007")
        before = sample.read_text(encoding="utf-8")
        init_config(tmp_path)
        assert sample.read_text(encoding="utf-8") == before


class TestMissingSample:
    def test_the_error_points_at_new_sample_not_init(self, tmp_path):
        init_config(tmp_path)
        with pytest.raises(ConfigError, match="new-sample"):
            load_config(tmp_path)

    def test_the_error_names_the_missing_file(self, tmp_path):
        init_config(tmp_path)
        with pytest.raises(ConfigError, match=SAMPLE_FILENAME):
            load_config(tmp_path)


class TestNewSample:
    """Adding plug number two must not mean redoing the rig configuration."""

    def test_writes_one_file_named_after_the_plug(self, tmp_path):
        target = add_sample(tmp_path, "core-042")
        assert target.is_file()
        assert read_sample(target).id == "core-042"

    def test_does_not_touch_the_rig_or_run_configuration(self, tmp_path):
        init_config(tmp_path)
        before = {
            name: (tmp_path / name).read_text(encoding="utf-8")
            for name in (HARDWARE_FILENAME, RUN_FILENAME)
        }
        add_sample(tmp_path / "samples", "core-042")
        after = {
            name: (tmp_path / name).read_text(encoding="utf-8")
            for name in (HARDWARE_FILENAME, RUN_FILENAME)
        }
        assert before == after

    def test_overrides_set_the_geometry(self, tmp_path):
        target = add_sample(
            tmp_path, "core-042",
            "length=42.0", "diameter=25.0", "lithology=shale",
        )
        sample = read_sample(target)
        assert (sample.length, sample.diameter) == (42.0, 25.0)
        assert sample.lithology == "shale"

    def test_an_unsafe_id_produces_a_safe_filename(self, tmp_path):
        result = runner.invoke(
            app, ["new-sample", "core/42 A", "--dir", str(tmp_path), "-n"]
        )
        assert result.exit_code == 0, result.output
        written = list(tmp_path.glob("*.yaml"))
        assert len(written) == 1
        assert "/" not in written[0].name and " " not in written[0].name

    def test_refuses_to_overwrite_without_force(self, tmp_path):
        runner.invoke(app, ["new-sample", "core-042", "--dir", str(tmp_path), "-n"])
        result = runner.invoke(app, ["new-sample", "core-042", "--dir", str(tmp_path), "-n"])
        assert result.exit_code == 1
        assert "--force" in result.output

    def test_a_bad_template_is_reported(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("length_cm: -5\n", encoding="utf-8")
        result = runner.invoke(
            app,
            ["new-sample", "core-042", "--dir", str(tmp_path), "-n", "--from", str(bad)],
        )
        assert result.exit_code == 1
        assert "not a valid sample" in result.output

    def test_the_written_file_loads_as_a_sample_for_collect(self, tmp_path):
        init_config(tmp_path)
        target = add_sample(tmp_path / "samples", "core-042", "length=42.0")
        config = load_config(tmp_path, sample=target)
        assert config.sample.id == "core-042"
        assert config.sample.length_cm == pytest.approx(4.2)
        # The rig and the experiment came from the shared files.
        assert config.hardware.daq.device_name == "Dev1"

    def test_the_id_is_asked_for_when_omitted(self, tmp_path):
        result = runner.invoke(
            app, ["new-sample", "--dir", str(tmp_path)], input="core-077\n" + "\n" * 40
        )
        assert result.exit_code == 0, result.output
        assert "Sample id" in result.output
        assert read_sample(tmp_path / "core-077.yaml").id == "core-077"

    def test_a_missing_id_is_an_error_when_not_interactive(self, tmp_path):
        result = runner.invoke(app, ["new-sample", "--dir", str(tmp_path), "-n"])
        assert result.exit_code == 1
        assert "No sample id" in result.output

    def test_a_blank_id_is_rejected(self, tmp_path):
        result = runner.invoke(
            app, ["new-sample", "--dir", str(tmp_path)], input="   \n"
        )
        assert result.exit_code == 1
        assert "must not be blank" in result.output


class TestTemplateInheritance:
    """``--from`` carries the core, never this plug's own measurements."""

    @staticmethod
    def _template(tmp_path):
        return add_sample(
            tmp_path, "core-041",
            "lithology=sandstone",
            "formation=Rotliegend",
            "well=A-12",
            "depth=2145.5",
            "grain_density_g_cm3=2.65",
            "porosity_method=helium pycnometry",
            "prepared_by=DJ",
            "length=50.2",
            "diameter=38.1",
            "porosity_fraction=0.18",
            "bulk_density_g_cm3=2.17",
            "notes=first plug",
        )

    def _derive(self, tmp_path, **kwargs):
        template = self._template(tmp_path)
        args = [
            "new-sample", "core-042", "--dir", str(tmp_path), "-n",
            "--from", str(template),
        ]
        for key, value in kwargs.items():
            args += ["--set", f"{key}={value}"]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        return read_sample(tmp_path / "core-042.yaml")

    def test_core_level_fields_are_carried_over(self, tmp_path):
        sample = self._derive(tmp_path)
        assert sample.lithology == "sandstone"
        assert sample.formation == "Rotliegend"
        assert sample.well == "A-12"
        assert sample.depth == 2145.5
        assert sample.grain_density_g_cm3 == 2.65
        assert sample.porosity_method == "helium pycnometry"
        assert sample.prepared_by == "DJ"

    def test_the_identity_is_never_inherited(self, tmp_path):
        sample = self._derive(tmp_path)
        assert sample.id == "core-042"
        assert sample.notes == ""
        assert sample.description == ""

    def test_geometry_is_never_inherited(self, tmp_path):
        """Every plug is cut and measured individually."""
        sample = self._derive(tmp_path)
        defaults = SampleConfig(id="x")
        assert sample.length == defaults.length
        assert sample.diameter == defaults.diameter
        assert sample.length != 50.2

    def test_per_plug_measurements_are_never_inherited(self, tmp_path):
        sample = self._derive(tmp_path)
        assert sample.porosity_fraction is None
        assert sample.bulk_density_g_cm3 is None

    def test_the_new_plug_can_set_its_own_geometry(self, tmp_path):
        sample = self._derive(tmp_path, length=48.7, diameter=38.0)
        assert (sample.length, sample.diameter) == (48.7, 38.0)
        assert sample.lithology == "sandstone"

    def test_interactively_it_asks_for_the_geometry_and_reports_what_it_inherited(
        self, tmp_path
    ):
        template = self._template(tmp_path)
        result = runner.invoke(
            app,
            ["new-sample", "core-042", "--dir", str(tmp_path), "--from", str(template)],
            # description, length, length uncertainty, diameter, ...
            input="\n\n48.7\n\n38.0\n" + "\n" * 40,
        )
        assert result.exit_code == 0, result.output
        assert "inherited from" in result.output
        assert "lithology" in result.output
        assert "Length (mm)" in result.output
        sample = read_sample(tmp_path / "core-042.yaml")
        assert sample.length == 48.7
        assert sample.diameter == 38.0
        assert sample.lithology == "sandstone"

    def test_the_core_level_questions_are_skipped_when_inherited(self, tmp_path):
        template = self._template(tmp_path)
        result = runner.invoke(
            app,
            ["new-sample", "core-042", "--dir", str(tmp_path), "--from", str(template)],
            input="\n" * 40,
        )
        assert result.exit_code == 0, result.output
        assert "Lithology" not in result.output
        assert "Formation" not in result.output

class TestFlowmeterSelection:
    def test_the_run_file_can_name_the_meter(self, tmp_path):
        init_config(tmp_path, "run.flowmeter=high_range")
        config = load(tmp_path)
        assert config.flowmeter_name == "high_range"
        assert config.flowmeter.channel == "ai3"

    def test_an_unknown_meter_in_the_run_file_is_reported_clearly(self, tmp_path):
        result = runner.invoke(
            app,
            ["init", str(tmp_path), "--non-interactive", "--force",
             "--set", "run.flowmeter=middle_range"],
        )
        assert result.exit_code == 1
        assert "middle_range" in result.output


class TestInteractiveUnitPrompts:
    """Every unit prompt must list what it accepts, and reject what it doesn't.

    A mistyped unit used to surface only at final validation, throwing away the
    whole interview -- and the flow prompt offered no menu at all.
    """

    @staticmethod
    def _interview(tmp_path, answers: str):
        return runner.invoke(app, ["init", str(tmp_path), "--force"], input=answers)

    def test_pressure_and_flow_menus_are_both_offered(self, tmp_path):
        # Accept every default by sending blank lines.
        result = self._interview(tmp_path, "\n" * 200)
        assert result.exit_code == 0, result.output
        for unit in units.SUPPORTED_PRESSURE_UNITS:
            assert unit in result.output
        for unit in units.SUPPORTED_FLOW_UNITS:
            assert unit in result.output, f"flow unit {unit} was never offered"

    def test_the_permeability_menu_is_offered(self, tmp_path):
        result = self._interview(tmp_path, "\n" * 200)
        for unit in units.SUPPORTED_PERMEABILITY_UNITS:
            assert unit in result.output

    def test_accepting_the_defaults_produces_a_loadable_config(self, tmp_path):
        result = self._interview(tmp_path, "\n" * 200)
        assert result.exit_code == 0, result.output
        assert load(tmp_path).hardware.daq.device_name == "Dev1"

    def test_a_mistyped_unit_does_not_discard_the_interview(self, tmp_path):
        """A typo must be caught at the prompt, not after every other answer."""
        # Prompts before the inlet transducer's unit, in order: rig name,
        # device name, inlet channel, outlet channel, sample rate, volts_min,
        # volts_max -- seven. Then the unit: a typo, then a good value.
        answers = [""] * 7 + ["torr", "bar"] + [""] * 200
        result = self._interview(tmp_path, "\n".join(answers))
        assert result.exit_code == 0, result.output
        assert "torr" in result.output  # the complaint was shown inline
        assert load(tmp_path).hardware.pressure_calibration.inlet.unit == "bar"


class TestUnitPromptHelper:
    """The retry behaviour, tested directly rather than by counting prompts."""

    @staticmethod
    def _answers(monkeypatch, replies):
        given = iter(replies)
        asked: list[str] = []

        def fake_prompt(text, default=None, **kwargs):
            asked.append(text)
            return next(given)

        monkeypatch.setattr(cli.typer, "prompt", fake_prompt)
        return asked

    def test_a_good_flow_unit_is_accepted_first_time(self, monkeypatch):
        asked = self._answers(monkeypatch, ["slpm"])
        assert cli._prompt_flow_unit("Flow unit", "sccm") == "slpm"
        assert len(asked) == 1

    def test_the_flow_menu_is_shown(self, monkeypatch):
        asked = self._answers(monkeypatch, ["sccm"])
        cli._prompt_flow_unit("Flow unit", "sccm")
        for unit in units.SUPPORTED_FLOW_UNITS:
            assert unit in asked[0]

    def test_a_bad_flow_unit_is_re_asked(self, monkeypatch):
        asked = self._answers(monkeypatch, ["gallons/hour", "sccm"])
        assert cli._prompt_flow_unit("Flow unit", "sccm") == "sccm"
        assert len(asked) == 2

    def test_a_bad_pressure_unit_is_re_asked(self, monkeypatch):
        asked = self._answers(monkeypatch, ["torr", "MPa"])
        assert cli._prompt_pressure_unit("Pressure unit", "kPa") == "MPa"
        assert len(asked) == 2

    def test_persistent_nonsense_falls_back_to_the_default(self, monkeypatch):
        """Rather than looping forever or returning something invalid."""
        self._answers(monkeypatch, ["nope"] * cli._UNIT_PROMPT_ATTEMPTS)
        assert cli._prompt_pressure_unit("Pressure unit", "kPa") == "kPa"

    def test_a_bad_optional_number_is_re_asked(self, monkeypatch):
        """One mistyped optional field must not abort the interview either."""
        asked = self._answers(monkeypatch, ["A-12", "2145.5"])
        assert cli._optional_float("Depth") == 2145.5
        assert len(asked) == 2

    def test_a_blank_optional_number_means_unset(self, monkeypatch):
        self._answers(monkeypatch, ["  "])
        assert cli._optional_float("Depth") is None

    def test_persistent_nonsense_leaves_an_optional_number_unset(self, monkeypatch):
        self._answers(monkeypatch, ["nope"] * cli._UNIT_PROMPT_ATTEMPTS)
        assert cli._optional_float("Depth") is None

    def test_surrounding_whitespace_is_tolerated(self, monkeypatch):
        self._answers(monkeypatch, ["  bar  "])
        assert cli._prompt_pressure_unit("Pressure unit", "kPa") == "bar"

    def test_every_offered_unit_is_actually_accepted(self, monkeypatch):
        """The menu is built from units.py, so it cannot drift from reality."""
        for unit in units.SUPPORTED_FLOW_UNITS:
            self._answers(monkeypatch, [unit])
            assert cli._prompt_flow_unit("Flow unit", "sccm") == unit
        for unit in units.SUPPORTED_PRESSURE_UNITS:
            self._answers(monkeypatch, [unit])
            assert cli._prompt_pressure_unit("Pressure unit", "kPa") == unit


class TestKlinkenbergDiscovery:
    """`--sample` replaces typing every run directory."""

    @staticmethod
    def _rig(tmp_path, writer, *, plugs=("core-041",), runs_per_plug=3):
        rig = tmp_path / "rig"
        init_config(rig)
        runs = rig / "runs"
        hour = 9
        for plug in plugs:
            for step in range(runs_per_plug):
                writer(
                    runs, plug, datetime(2026, 8, 3, hour, 0, tzinfo=timezone.utc),
                    mean_pressure_atm=2.0 + step,
                    permeability_darcy=0.005 * (1.0 + 0.2 / (2.0 + step)),
                )
                hour += 1
        return rig, runs

    def test_regresses_every_run_for_the_plug(self, tmp_path, fake_run_writer):
        rig, _ = self._rig(tmp_path, fake_run_writer)
        result = runner.invoke(app, ["klinkenberg", "--sample", "core-041", "-c", str(rig)])
        assert result.exit_code == 0, result.output
        assert "Found 3 runs for core-041" in result.output
        assert "k_L (liquid-equivalent)" in result.output

    def test_results_are_named_per_plug(self, tmp_path, fake_run_writer):
        rig, runs = self._rig(tmp_path, fake_run_writer)
        runner.invoke(app, ["klinkenberg", "--sample", "core-041", "-c", str(rig)])
        assert (runs / "klinkenberg_core-041.yaml").is_file()
        assert not (runs / "klinkenberg.yaml").exists()

    def test_a_second_plug_does_not_overwrite_the_first(self, tmp_path, fake_run_writer):
        """The collision this change fixes."""
        rig, runs = self._rig(tmp_path, fake_run_writer, plugs=("core-041", "core-042"))
        for plug in ("core-041", "core-042"):
            result = runner.invoke(app, ["klinkenberg", "--sample", plug, "-c", str(rig)])
            assert result.exit_code == 0, result.output

        first = yaml.safe_load((runs / "klinkenberg_core-041.yaml").read_text(encoding="utf-8"))
        second = yaml.safe_load((runs / "klinkenberg_core-042.yaml").read_text(encoding="utf-8"))
        assert {p["sample_id"] for p in first["points"]} == {"core-041"}
        assert {p["sample_id"] for p in second["points"]} == {"core-042"}

    def test_only_the_named_plugs_runs_are_used(self, tmp_path, fake_run_writer):
        rig, _ = self._rig(tmp_path, fake_run_writer, plugs=("core-041", "core-042"))
        result = runner.invoke(app, ["klinkenberg", "--sample", "core-041", "-c", str(rig)])
        assert "Found 3 runs" in result.output
        assert "core-042" not in result.output

    def test_a_sample_file_works_the_same_as_an_id(self, tmp_path, fake_run_writer):
        rig, _ = self._rig(tmp_path, fake_run_writer)
        sample_file = add_sample(rig / "samples", "core-041")
        by_id = runner.invoke(app, ["klinkenberg", "--sample", "core-041", "-c", str(rig)])
        by_file = runner.invoke(app, ["klinkenberg", "--sample", str(sample_file), "-c", str(rig)])
        assert by_file.exit_code == 0, by_file.output
        assert "Found 3 runs for core-041" in by_file.output
        assert by_id.exit_code == by_file.exit_code

    def test_a_missing_sample_file_is_not_treated_as_an_id(self, tmp_path, fake_run_writer):
        rig, _ = self._rig(tmp_path, fake_run_writer)
        result = runner.invoke(
            app, ["klinkenberg", "--sample", "samples/nope.yaml", "-c", str(rig)]
        )
        assert result.exit_code == 1
        assert "No such sample file" in result.output

    def test_an_unknown_plug_lists_the_ones_present(self, tmp_path, fake_run_writer):
        rig, _ = self._rig(tmp_path, fake_run_writer, plugs=("core-041", "core-042"))
        result = runner.invoke(app, ["klinkenberg", "--sample", "core-999", "-c", str(rig)])
        assert result.exit_code == 1
        assert "core-041" in result.output and "core-042" in result.output

    def test_runs_dir_works_without_any_run_yaml(self, tmp_path, fake_run_writer):
        runs = tmp_path / "loose"
        for step in range(3):
            fake_run_writer(
                runs, "core-041", datetime(2026, 8, 3, 9 + step, tzinfo=timezone.utc),
                mean_pressure_atm=2.0 + step,
            )
        result = runner.invoke(
            app, ["klinkenberg", "--sample", "core-041", "--runs-dir", str(runs)]
        )
        assert result.exit_code == 0, result.output
        assert (runs / "klinkenberg_core-041.yaml").is_file()

    def test_no_runs_dir_and_no_run_yaml_says_so(self, tmp_path):
        result = runner.invoke(
            app, ["klinkenberg", "--sample", "core-041", "-c", str(tmp_path)]
        )
        assert result.exit_code == 1
        assert "--runs-dir" in result.output

    def test_an_unsteady_run_is_skipped_with_a_reason(self, tmp_path, fake_run_writer):
        rig, runs = self._rig(tmp_path, fake_run_writer)
        fake_run_writer(
            runs, "core-041", datetime(2026, 8, 3, 20, tzinfo=timezone.utc), steady=False
        )
        result = runner.invoke(app, ["klinkenberg", "--sample", "core-041", "-c", str(rig)])
        assert result.exit_code == 0, result.output
        assert "Found 4 runs" in result.output
        assert "skipped" in result.output
        assert "(3 points" in result.output

    def test_the_skip_reason_is_readable_not_a_file_path(self, tmp_path, fake_run_writer):
        """The message leads with the CSV path, which is redundant and full of dots."""
        rig, runs = self._rig(tmp_path, fake_run_writer)
        fake_run_writer(
            runs, "core-041", datetime(2026, 8, 3, 20, tzinfo=timezone.utc), steady=False
        )
        result = runner.invoke(app, ["klinkenberg", "--sample", "core-041", "-c", str(rig)])
        skipped = next(line for line in result.output.splitlines() if "skipped:" in line)
        reason = skipped.split("skipped:", 1)[1].strip()
        assert reason.startswith("never reached steady state")
        assert "readings" not in reason
        assert str(runs) not in reason

    def test_allow_unsteady_includes_it(self, tmp_path, fake_run_writer):
        rig, runs = self._rig(tmp_path, fake_run_writer)
        fake_run_writer(
            runs, "core-041", datetime(2026, 8, 3, 20, tzinfo=timezone.utc), steady=False
        )
        result = runner.invoke(
            app, ["klinkenberg", "--sample", "core-041", "-c", str(rig), "--allow-unsteady"]
        )
        assert result.exit_code == 0, result.output
        assert "(4 points" in result.output

    def test_too_few_usable_runs_fails_clearly(self, tmp_path, fake_run_writer):
        rig = tmp_path / "rig"
        init_config(rig)
        fake_run_writer(
            rig / "runs", "core-041", datetime(2026, 8, 3, 9, tzinfo=timezone.utc)
        )
        result = runner.invoke(app, ["klinkenberg", "--sample", "core-041", "-c", str(rig)])
        assert result.exit_code == 1
        assert "at least" in result.output

    def test_sample_cannot_be_combined_with_paths_or_csv(self, tmp_path, fake_run_writer):
        rig, runs = self._rig(tmp_path, fake_run_writer)
        both = runner.invoke(
            app, ["klinkenberg", str(runs), "--sample", "core-041", "-c", str(rig)]
        )
        assert both.exit_code == 1
        assert "exactly one" in both.output

    def test_nothing_at_all_mentions_sample(self, tmp_path):
        result = runner.invoke(app, ["klinkenberg"])
        assert result.exit_code == 1
        assert "--sample" in result.output

    def test_positional_mode_also_gets_a_per_plug_filename(self, tmp_path, fake_run_writer):
        runs = tmp_path / "runs"
        directories = [
            fake_run_writer(
                runs, "core-041", datetime(2026, 8, 3, 9 + step, tzinfo=timezone.utc),
                mean_pressure_atm=2.0 + step,
            )
            for step in range(3)
        ]
        result = runner.invoke(app, ["klinkenberg", *[str(d) for d in directories]])
        assert result.exit_code == 0, result.output
        assert (runs / "klinkenberg_core-041.yaml").is_file()

    def test_plot_lands_beside_the_results(self, tmp_path, fake_run_writer):
        pytest.importorskip("matplotlib")
        rig, runs = self._rig(tmp_path, fake_run_writer)
        result = runner.invoke(
            app, ["klinkenberg", "--sample", "core-041", "-c", str(rig), "--plot"]
        )
        assert result.exit_code == 0, result.output
        assert (runs / "klinkenberg_core-041.png").is_file()


class TestMixedDownstreamConventions:
    """`--sample` sweeps up every run, so a convention mismatch could hide."""

    @staticmethod
    def _rig(tmp_path, writer, *, conventions):
        rig = tmp_path / "rig"
        init_config(rig)
        runs = rig / "runs"
        for hour, convention in enumerate(conventions, start=9):
            writer(
                runs, "core-041", datetime(2026, 8, 3, hour, tzinfo=timezone.utc),
                mean_pressure_atm=2.0 + hour - 9,
                downstream_pressure=convention,
            )
        return rig, runs

    def test_a_uniform_series_regresses(self, tmp_path, fake_run_writer):
        rig, _ = self._rig(
            tmp_path, fake_run_writer, conventions=["measured"] * 3
        )
        result = runner.invoke(app, ["klinkenberg", "--sample", "core-041", "-c", str(rig)])
        assert result.exit_code == 0, result.output

    def test_a_mixed_series_is_refused(self, tmp_path, fake_run_writer):
        rig, _ = self._rig(
            tmp_path, fake_run_writer, conventions=["measured", "measured", 101.325]
        )
        result = runner.invoke(app, ["klinkenberg", "--sample", "core-041", "-c", str(rig)])
        assert result.exit_code == 1
        assert "did not all obtain the downstream" in result.output

    def test_the_listing_shows_the_mismatch_before_the_refusal(
        self, tmp_path, fake_run_writer
    ):
        rig, _ = self._rig(
            tmp_path, fake_run_writer, conventions=["measured", "measured", 101.325]
        )
        result = runner.invoke(app, ["klinkenberg", "--sample", "core-041", "-c", str(rig)])
        assert "P2 measured" in result.output
        assert "P2 fixed" in result.output

    def test_it_can_be_forced(self, tmp_path, fake_run_writer):
        rig, _ = self._rig(
            tmp_path, fake_run_writer, conventions=["measured", "measured", 101.325]
        )
        result = runner.invoke(
            app,
            ["klinkenberg", "--sample", "core-041", "-c", str(rig),
             "--allow-mixed-conditions"],
        )
        assert result.exit_code == 0, result.output
        assert "downstream pressure differently" in result.output


class TestCollectDownstreamFlag:
    """The flag itself, checked without opening a DAQ."""

    def test_the_config_field_accepts_a_value_and_a_unit(self, tmp_path):
        rig = tmp_path / "rig"
        init_config(
            rig, "run.downstream_pressure=1.0", "run.downstream_pressure_unit=bar"
        )
        run = load(rig).run
        assert run.downstream_pressure == 1.0
        assert run.fixed_downstream_pressure_atm == pytest.approx(
            units.bar_to_atm(1.0)
        )

    def test_a_non_positive_value_is_refused_at_init(self, tmp_path):
        result = runner.invoke(
            app,
            ["init", str(tmp_path / "rig"), "--non-interactive", "--force",
             "--set", "run.downstream_pressure=0"],
        )
        assert result.exit_code == 1
        assert "positive absolute pressure" in result.output

    @pytest.mark.parametrize("flag", ["--downstream-pressure", "--outlet-pressure"])
    def test_both_spellings_are_accepted(self, tmp_path, flag):
        """The user's own wording is 'outlet'; the config field says 'downstream'."""
        result = runner.invoke(
            app, ["collect", flag, "101.8", "--config-dir", str(tmp_path)]
        )
        # It gets past option parsing and fails on the missing rig instead.
        assert "No such option" not in result.output
        assert "hardware.yaml" in result.output

    def test_a_value_that_is_neither_a_number_nor_measured_is_rejected(self, tmp_path):
        rig = tmp_path / "rig"
        init_config(rig)
        add_sample(rig / "samples", "core-041")
        result = runner.invoke(
            app,
            ["collect", "-c", str(rig), "--sample", str(rig / "samples" / "core-041.yaml"),
             "--downstream-pressure", "ambient"],
        )
        assert result.exit_code == 1
        assert "neither a number nor 'measured'" in result.output


class TestPorosityUnitOnTheCommandLine:
    """A percentage porosity, end to end through new-sample."""

    def test_a_bare_percent_sign_is_accepted(self, tmp_path):
        """`%` is a YAML directive indicator, so --set must not parse it as YAML."""
        rig = tmp_path / "rig"
        init_config(rig)
        path = add_sample(
            rig / "samples", "core-041",
            "porosity_unit=%", "porosity=10.4", "porosity_uncertainty=0.5",
        )
        sample = read_sample(path)
        assert sample.porosity_unit == "%"
        assert sample.porosity_fraction == pytest.approx(0.104)
        assert sample.porosity_uncertainty_fraction == pytest.approx(0.005)

    def test_a_fraction_is_still_the_default(self, tmp_path):
        rig = tmp_path / "rig"
        init_config(rig)
        path = add_sample(rig / "samples", "core-042", "porosity=0.104")
        assert read_sample(path).porosity_fraction == pytest.approx(0.104)

    def test_a_percentage_left_labelled_a_fraction_is_refused(self, tmp_path):
        rig = tmp_path / "rig"
        init_config(rig)
        result = runner.invoke(
            app,
            ["new-sample", "core-043", "--dir", str(rig / "samples"), "-n", "--force",
             "--set", "porosity=10.4"],
        )
        assert result.exit_code == 1
        assert "more than the whole rock" in strip_ansi(result.output)

    def test_the_template_documents_both_spellings(self, tmp_path):
        rig = tmp_path / "rig"
        init_config(rig)
        path = add_sample(rig / "samples", "core-044")
        text = path.read_text(encoding="utf-8")
        assert "fraction | %" in text
        assert "half a percentage point" in text


class TestCollectFooter:
    """The count printed after a run. Deliberately says nothing about targets."""

    def test_the_command_line_for_the_current_folder(self):
        assert cli._klinkenberg_command_line("core-041", Path(".")) == (
            "gasperm klinkenberg --sample core-041 --plot"
        )

    def test_a_rig_folder_is_passed_through(self):
        line = cli._klinkenberg_command_line("core-041", Path("tight-gas-rig"))
        assert "-c tight-gas-rig" in line and "--sample core-041" in line

    def test_an_overridden_output_dir_is_spelled_out(self):
        line = cli._klinkenberg_command_line("core-041", Path("."), Path("/data/runs"))
        assert "--runs-dir" in line

    def test_the_footer_counts_the_plugs_runs(self, tmp_path, fake_run_writer, base_config):
        runs = tmp_path / "runs"
        for step in range(3):
            fake_run_writer(runs, "core-041", datetime(2026, 8, 3, 9 + step, tzinfo=timezone.utc))
        base_config.sample.id = "core-041"

        class _Writer:
            directory = runs / "core-041_20260803T090000Z"

        printed = []
        with mock.patch.object(cli.typer, "echo", lambda *a, **k: printed.append(str(a[0] if a else ""))), \
             mock.patch.object(cli.typer, "secho", lambda *a, **k: printed.append(str(a[0] if a else ""))):
            cli._print_collect_next_steps(base_config, _Writer(), Path("."), False)
        text = "\n".join(printed)
        assert "3 runs recorded for core-041" in text
        assert "gasperm klinkenberg --sample core-041" in text

    def test_the_footer_offers_no_target_or_coaching(self, tmp_path, fake_run_writer, base_config):
        """The user asked to manage the point count themselves."""
        runs = tmp_path / "runs"
        fake_run_writer(runs, "core-041", datetime(2026, 8, 3, 9, tzinfo=timezone.utc))
        base_config.sample.id = "core-041"

        class _Writer:
            directory = runs / "core-041_20260803T090000Z"

        printed = []
        with mock.patch.object(cli.typer, "echo", lambda *a, **k: printed.append(str(a[0] if a else ""))), \
             mock.patch.object(cli.typer, "secho", lambda *a, **k: printed.append(str(a[0] if a else ""))):
            cli._print_collect_next_steps(base_config, _Writer(), Path("."), False)
        text = "\n".join(printed).lower()
        for banned in (" of 3", "target", "suggest", "at least", "only ", "more point"):
            assert banned not in text, f"footer should not coach the operator: {banned!r}"


class TestLivePlotFlags:
    """The plot flags are rejected before any hardware is opened.

    Each of these fails on the flags alone, so none of them needs a DAQ -- the
    point being that a mistyped plot option cannot cost an operator a run that
    has already started.
    """

    def _collect(self, tmp_path, *flags):
        init_config(tmp_path)
        sample = add_sample(tmp_path / "samples", "core-041")
        return runner.invoke(
            app, ["collect", "-c", str(tmp_path), "--sample", str(sample), *flags]
        )

    def test_a_window_and_from_start_together_are_refused(self, tmp_path):
        result = self._collect(tmp_path, "--plot-window", "60", "--plot-from-start")
        assert result.exit_code == 1
        assert "opposite views" in result.output

    def test_an_unknown_panel_name_is_refused_and_lists_the_real_ones(self, tmp_path):
        result = self._collect(tmp_path, "--plot-panels", "flow,viscosity")
        assert result.exit_code == 1
        assert "viscosity" in result.output
        assert "inlet_pressure" in result.output

    def test_a_repeated_panel_is_refused(self, tmp_path):
        result = self._collect(tmp_path, "--plot-panels", "flow,flow")
        assert result.exit_code == 1
        assert "repeats flow" in result.output

    def test_the_flags_are_documented_in_help(self):
        # Typer renders help through rich, which both wraps to the terminal
        # width and interleaves colour escapes *inside* the flag names -- so a
        # raw substring check silently becomes a test of the window size and
        # of whether colour happens to be on. Pin the width and strip the
        # escapes, and the assertion is about the help text again.
        result = runner.invoke(app, ["collect", "--help"], env={"COLUMNS": "200"})
        assert result.exit_code == 0
        plain = strip_ansi(result.output)
        for flag in ("--plot", "--plot-window", "--plot-from-start", "--plot-panels"):
            assert flag in plain


class TestPulseDecayFlags:
    """The method switch and its refusals, all before any hardware is opened."""

    def _collect(self, tmp_path, *flags, **overrides):
        init_config(tmp_path, *overrides.pop("init", ()))
        sample = add_sample(tmp_path / "samples", "core-041")
        return runner.invoke(
            app, ["collect", "-c", str(tmp_path), "--sample", str(sample), *flags]
        )

    def test_an_unknown_method_is_refused_and_lists_the_real_ones(self, tmp_path):
        result = self._collect(tmp_path, "--method", "pulse-decay")
        assert result.exit_code == 1
        assert "steady_state" in result.output
        assert "pulse_decay" in result.output

    def test_pulse_decay_with_a_supplied_p2_is_refused(self, tmp_path):
        result = self._collect(
            tmp_path, "--downstream-pressure", "101.325", "--method", "pulse_decay"
        )
        assert result.exit_code == 1
        assert "CLOSED downstream vessel" in strip_ansi(result.output)

    def test_the_override_order_lets_a_consistent_pair_through(self, tmp_path):
        """--downstream-pressure measured then --method must not self-refuse."""
        result = self._collect(
            tmp_path,
            "--downstream-pressure", "measured",
            "--method", "pulse_decay",
            "--samples", "1",
        )
        # It gets past config entirely and fails on the absent DAQ/serial rig,
        # which is the point: the refusal was not a config one.
        assert "CLOSED downstream vessel" not in strip_ansi(result.output)

    def test_a_malformed_spacer_is_refused(self, tmp_path):
        result = self._collect(tmp_path, "--method", "pulse_decay", "--spacer", "wide")
        assert result.exit_code == 1
        assert "TYPE:LENGTH" in strip_ansi(result.output)

    def test_an_unknown_bore_is_refused_and_lists_the_real_ones(self, tmp_path):
        result = self._collect(
            tmp_path, "--method", "pulse_decay", "--spacer", "enormous:50"
        )
        assert result.exit_code == 1
        plain = strip_ansi(result.output)
        assert "enormous" in plain
        assert "wide" in plain and "narrow" in plain

    def test_a_bad_length_is_refused(self, tmp_path):
        result = self._collect(
            tmp_path, "--method", "pulse_decay", "--spacer", "wide:abc"
        )
        assert result.exit_code == 1
        assert "not a length" in strip_ansi(result.output)

    def test_the_stack_reaches_the_config(self, tmp_path):
        """The stack changes per run, so it is a run-level list."""
        init_config(tmp_path, "run.method=pulse_decay")
        config = load_config(tmp_path, sample=add_sample(tmp_path / "samples", "core-041"))
        from gasperm.config import SpacerFitting

        config.run.pulse_decay.upstream_spacers = [
            SpacerFitting(type="wide", length=50.0),
            SpacerFitting(type="narrow", length=25.0),
        ]
        reservoirs = config.hardware.reservoirs
        expected = reservoirs.spacer_types["wide"].volume_cm3(50.0) + reservoirs.spacer_types[
            "narrow"
        ].volume_cm3(25.0)
        assert reservoirs.upstream_volume_cm3(
            config.run.pulse_decay.upstream_spacers
        ) == pytest.approx(400.0 + expected)

    def test_the_bores_survive_a_config_round_trip(self, tmp_path):
        init_config(tmp_path, "run.method=pulse_decay")
        config = load_config(tmp_path, sample=add_sample(tmp_path / "samples", "core-041"))
        types = config.hardware.reservoirs.spacer_types
        assert set(types) == {"wide", "narrow"}
        assert types["wide"].internal_diameter == pytest.approx(25.4)
        assert types["narrow"].internal_diameter == pytest.approx(12.7)

    def test_leak_test_implies_pulse_decay(self, tmp_path):
        """A leak test IS a pulse-decay observation, so asking for both is noise."""
        init_config(tmp_path)
        sample = add_sample(tmp_path / "samples", "core-041")
        config = load_config(tmp_path, sample=sample)
        assert config.run.method == "steady_state"
        result = runner.invoke(
            app,
            ["collect", "-c", str(tmp_path), "--sample", str(sample), "--leak-test"],
        )
        # It gets past config and fails on the absent rig, not on the pairing.
        plain = strip_ansi(result.output)
        assert "pulse-decay observation" not in plain

    def test_a_leak_test_without_a_duration_is_refused(self, tmp_path):
        """Nothing decays on a tight rig, so there is no signal to stop on."""
        init_config(
            tmp_path,
            "run.method=pulse_decay",
            "run.pulse_decay.leak_test_duration_s=null",
        )
        sample = add_sample(tmp_path / "samples", "core-041")
        result = runner.invoke(
            app,
            ["collect", "-c", str(tmp_path), "--sample", str(sample), "--leak-test"],
        )
        assert result.exit_code == 1
        assert "needs a duration" in strip_ansi(result.output)

    def test_the_flags_are_documented_in_help(self):
        result = runner.invoke(app, ["collect", "--help"], env={"COLUMNS": "200"})
        plain = strip_ansi(result.output)
        assert "--method" in plain
        assert "--spacer" in plain
        assert "--leak-test" in plain
        result = runner.invoke(app, ["klinkenberg", "--help"], env={"COLUMNS": "200"})
        assert "--allow-mixed-methods" in strip_ansi(result.output)

    def test_a_pulse_run_config_round_trips_through_init(self, tmp_path):
        init_config(tmp_path, "run.method=pulse_decay")
        sample = add_sample(tmp_path / "samples", "core-041")
        config = load_config(tmp_path, sample=sample)
        assert config.run.method == "pulse_decay"
        reservoirs = config.hardware.reservoirs
        assert reservoirs.upstream_volume_cm3() == pytest.approx(400.0)
        assert reservoirs.downstream_volume_cm3() == pytest.approx(75.0)
        # Split into the parts that are separately measured.
        assert reservoirs.upstream.vessel == pytest.approx(380.0)
        assert reservoirs.upstream.dead == pytest.approx(20.0)


class TestVersion:
    def test_version_prints_and_exits(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "gasperm" in result.output

    def test_bare_invocation_shows_help(self):
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "new-sample" in result.output
        assert "klinkenberg" in result.output
