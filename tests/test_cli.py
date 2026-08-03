"""CLI surface: the multi-plug and multi-meter workflows.

Exercises the real commands through typer's runner. Hardware is never touched:
``collect`` is covered by the acquisition tests, so these focus on the parts an
operator drives between runs -- adding plugs and choosing a meter.
"""

from __future__ import annotations

from pathlib import Path

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
        assert load(rig).run.output_dir == f"{rig.as_posix()}/runs"

    def test_a_relative_folder_keeps_a_relative_output_dir(self):
        """And an absolute one is not mangled with a leading './'."""
        assert cli._default_output_dir(Path("tight-gas-rig")) == "./tight-gas-rig/runs"
        assert cli._default_output_dir(Path("./rig")) == "./rig/runs"
        absolute = Path.cwd() / "rig"
        assert cli._default_output_dir(absolute) == f"{absolute.as_posix()}/runs"
        assert not cli._default_output_dir(absolute).startswith("./")

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
            "length_cm=4.2", "diameter_cm=2.5", "lithology=shale",
        )
        sample = read_sample(target)
        assert (sample.length_cm, sample.diameter_cm) == (4.2, 2.5)
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
        target = add_sample(tmp_path / "samples", "core-042", "length_cm=4.2")
        config = load_config(tmp_path, sample=target)
        assert config.sample.id == "core-042"
        assert config.sample.length_cm == 4.2
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
            "length_cm=5.02",
            "diameter_cm=2.54",
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
        assert sample.length_cm == defaults.length_cm
        assert sample.diameter_cm == defaults.diameter_cm
        assert sample.length_cm != 5.02

    def test_per_plug_measurements_are_never_inherited(self, tmp_path):
        sample = self._derive(tmp_path)
        assert sample.porosity_fraction is None
        assert sample.bulk_density_g_cm3 is None

    def test_the_new_plug_can_set_its_own_geometry(self, tmp_path):
        sample = self._derive(tmp_path, length_cm=4.87, diameter_cm=2.53)
        assert (sample.length_cm, sample.diameter_cm) == (4.87, 2.53)
        assert sample.lithology == "sandstone"

    def test_interactively_it_asks_for_the_geometry_and_reports_what_it_inherited(
        self, tmp_path
    ):
        template = self._template(tmp_path)
        result = runner.invoke(
            app,
            ["new-sample", "core-042", "--dir", str(tmp_path), "--from", str(template)],
            # description, length, length uncertainty, diameter, ...
            input="\n4.87\n\n2.53\n" + "\n" * 40,
        )
        assert result.exit_code == 0, result.output
        assert "inherited from" in result.output
        assert "lithology" in result.output
        assert "Length (cm)" in result.output
        sample = read_sample(tmp_path / "core-042.yaml")
        assert sample.length_cm == 4.87
        assert sample.diameter_cm == 2.53
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
