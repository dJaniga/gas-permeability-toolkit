"""``gasperm summarize``: one plug's whole history, and what is missing from it.

Most of these assert on the **findings** rather than on the table. A summary
that only restated what is on disk would leave the operator to notice that a
run never confirmed, that two meters were used, or that a pulse-decay series
has no leak test behind it -- and those are the reasons to run it.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from gasperm.cli import app
from gasperm.summary import detect_campaign_split

from conftest import write_measured_run

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


@pytest.fixture(autouse=True)
def _quiet_logging():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def make_rig(tmp_path) -> Path:
    rig = tmp_path / "rig"
    assert runner.invoke(
        app, ["init", str(rig), "--non-interactive", "--force"]
    ).exit_code == 0
    return rig


def add_series(
    rig: Path,
    sample_id: str = "core-041",
    *,
    start: datetime | None = None,
    factor: float = 1.0,
    pressures=(5.0, 10.0, 20.0),
    **kwargs,
):
    """A Klinkenberg series with a planted k_L of 0.5 mD and b of 4 atm."""
    start = start or datetime(2026, 1, 10, 9, tzinfo=timezone.utc)
    for index, pressure in enumerate(pressures):
        write_measured_run(
            rig / "runs", sample_id, start + timedelta(hours=index),
            mean_pressure_atm=pressure,
            permeability_darcy=0.5e-3 * factor * (1.0 + 4.0 / pressure),
            **kwargs,
        )


def unconfirm(directory: Path) -> Path:
    """Turn a written run into one that never confirmed a measurement."""
    path = directory / "run_metadata.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["summary"]["measurement_confirmed"] = False
    payload["summary"]["steady_state_reached"] = False
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return directory


def summarize(rig, *args):
    return runner.invoke(app, ["summarize", *args, "-c", str(rig)])


class TestCampaignSplit:
    """Telling one series from two, which decides whether `compare` applies."""

    def moments(self, *days):
        return [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=d) for d in days]

    def test_one_continuous_series_has_no_split(self):
        """Pressure steps hours apart must never read as two campaigns."""
        assert detect_campaign_split(self.moments(0, 0.1, 0.2, 0.3)) is None

    def test_a_long_quiet_stretch_splits(self):
        stamps = self.moments(0, 0.1, 0.2, 150, 150.1, 150.2)
        split = detect_campaign_split(stamps)
        assert split == stamps[3]

    def test_a_short_gap_does_not_split(self):
        """Absolute as well as relative: a day off is not an exposure."""
        assert detect_campaign_split(self.moments(0, 0.1, 0.2, 1.0, 1.1)) is None

    def test_too_few_runs_to_call_it_two_campaigns(self):
        assert detect_campaign_split(self.moments(0, 150)) is None

    def test_a_lone_late_run_is_not_a_campaign(self):
        """One run on the far side is a straggler, not a second campaign."""
        assert detect_campaign_split(self.moments(0, 0.1, 0.2, 0.3, 200)) is None

    def test_evenly_spaced_runs_never_split(self):
        """Monthly monitoring is one series, however long it runs."""
        assert detect_campaign_split(self.moments(0, 30, 60, 90, 120)) is None


class TestRoster:
    def test_it_lists_every_plug(self, tmp_path):
        rig = make_rig(tmp_path)
        add_series(rig, "core-041")
        add_series(rig, "core-042", start=datetime(2026, 2, 1, tzinfo=timezone.utc))
        result = summarize(rig)
        assert result.exit_code == 0, result.output
        output = strip_ansi(result.output)
        assert "2 plug(s)" in output
        assert "core-041" in output and "core-042" in output

    def test_it_says_how_to_go_deeper(self, tmp_path):
        rig = make_rig(tmp_path)
        add_series(rig)
        assert "gasperm summarize <plug>" in strip_ansi(summarize(rig).output)

    def test_an_empty_runs_directory_is_refused(self, tmp_path):
        rig = make_rig(tmp_path)
        (rig / "runs").mkdir(exist_ok=True)
        result = summarize(rig)
        assert result.exit_code == 1
        assert "No runs found" in strip_ansi(result.output)


class TestSampleReport:
    def test_it_lists_the_runs_and_fits_them(self, tmp_path):
        rig = make_rig(tmp_path)
        add_series(rig)
        result = summarize(rig, "core-041")
        assert result.exit_code == 0, result.output
        output = strip_ansi(result.output)
        assert "3 confirmed run(s)" in output
        assert "Klinkenberg correction" in output
        assert "k_L = 0.5" in output
        assert "b   = 4" in output

    def test_it_shows_the_plug_identity(self, tmp_path):
        rig = make_rig(tmp_path)
        add_series(rig, porosity_fraction=0.123)
        output = strip_ansi(summarize(rig, "core-041").output)
        assert "5.000 x 3.810 cm" in output
        assert "porosity 0.123" in output

    def test_an_unconfirmed_run_is_excluded_and_named(self, tmp_path):
        rig = make_rig(tmp_path)
        add_series(rig)
        unconfirm(
            write_measured_run(
                rig / "runs", "core-041",
                datetime(2026, 1, 10, 15, tzinfo=timezone.utc),
                mean_pressure_atm=12.0, permeability_darcy=9e-4,
            )
        )
        output = strip_ansi(summarize(rig, "core-041").output)
        assert "3 confirmed run(s), 1 not" in output
        assert "never confirmed a measurement" in output

    def test_a_single_pressure_says_why_there_is_no_fit(self, tmp_path):
        rig = make_rig(tmp_path)
        add_series(rig, pressures=(10.0,))
        output = strip_ansi(summarize(rig, "core-041").output)
        assert "No Klinkenberg fit" in output
        assert "distinct mean pressure" in output

    def test_a_two_point_fit_says_it_cannot_be_checked(self, tmp_path):
        rig = make_rig(tmp_path)
        add_series(rig, pressures=(5.0, 20.0))
        output = strip_ansi(summarize(rig, "core-041").output)
        assert "define a line exactly" in output

    def test_two_meters_are_flagged(self, tmp_path):
        rig = make_rig(tmp_path)
        add_series(rig, pressures=(5.0, 10.0))
        write_measured_run(
            rig / "runs", "core-041", datetime(2026, 1, 10, 14, tzinfo=timezone.utc),
            mean_pressure_atm=20.0, permeability_darcy=6e-4, flowmeter="high_range",
        )
        output = strip_ansi(summarize(rig, "core-041").output)
        assert "More than one flowmeter" in output
        assert "does not cancel" in output

    def test_a_pulse_series_without_a_leak_test_is_flagged(self, tmp_path):
        rig = make_rig(tmp_path)
        for index, pressure in enumerate((5.0, 10.0, 20.0)):
            write_measured_run(
                rig / "runs", "core-041",
                datetime(2026, 1, 10, 9 + index, tzinfo=timezone.utc),
                mean_pressure_atm=pressure, permeability_darcy=1e-6,
                method="pulse_decay",
            )
        output = strip_ansi(summarize(rig, "core-041").output)
        assert "No leak test is recorded" in output

    def test_a_leak_test_is_listed_but_not_counted_as_a_measurement(self, tmp_path):
        rig = make_rig(tmp_path)
        add_series(rig)
        write_measured_run(
            rig / "runs", "core-041", datetime(2026, 1, 9, tzinfo=timezone.utc),
            mean_pressure_atm=10.0, permeability_darcy=1e-9,
            method="pulse_decay", purpose="leak_test",
        )
        output = strip_ansi(summarize(rig, "core-041").output)
        assert "3 confirmed run(s)" in output
        assert "1 leak test(s)" in output
        assert "LEAK TEST" in output

    def test_two_campaigns_are_noticed_and_point_at_compare(self, tmp_path):
        rig = make_rig(tmp_path)
        add_series(rig, start=datetime(2026, 1, 10, 9, tzinfo=timezone.utc))
        add_series(
            rig, start=datetime(2026, 6, 14, 9, tzinfo=timezone.utc), factor=1.09
        )
        output = strip_ansi(summarize(rig, "core-041").output)
        assert "fall into two groups" in output
        assert "gasperm compare core-041 --split 2026-06-14" in output

    def test_one_campaign_says_nothing_about_splitting(self, tmp_path):
        rig = make_rig(tmp_path)
        add_series(rig)
        assert "two groups" not in strip_ansi(summarize(rig, "core-041").output)

    def test_an_unknown_plug_names_the_ones_present(self, tmp_path):
        rig = make_rig(tmp_path)
        add_series(rig, "core-041")
        result = summarize(rig, "core-999")
        assert result.exit_code == 1
        assert "core-041" in strip_ansi(result.output)


class TestRunTableColumns:
    """What the runs table reports, and in whose units.

    Pressures are *stored* in atm, because that is what the physics runs in.
    A table that printed the internal unit -- and printed it unlabelled --
    would be asking the operator to convert numbers the rest of the package
    already converts for them, and to guess what they were converting from.
    """

    def table(self, rig, sample_id="core-041"):
        """The run rows, without the header or the surrounding report.

        Matched on the run *directory* name (``<plug>_<timestamp>``) so the
        report's own title line, which is the bare plug id, is not mistaken
        for a row.
        """
        lines = strip_ansi(summarize(rig, sample_id).output).splitlines()
        return [line for line in lines if line.strip().startswith(f"{sample_id}_")]

    def header(self, rig, sample_id="core-041"):
        lines = strip_ansi(summarize(rig, sample_id).output).splitlines()
        return next(line for line in lines if "P_mean" in line)

    def caption(self, rig, sample_id="core-041"):
        """The units line above the table."""
        lines = strip_ansi(summarize(rig, sample_id).output).splitlines()
        return next(line for line in lines if line.strip().startswith("Runs"))

    def test_the_units_are_stated_once_above_the_table(self, tmp_path):
        """Three pressures share a unit, and a heading wide enough to repeat it
        on each would push the table past the width of a terminal."""
        rig = make_rig(tmp_path)
        add_series(rig)
        caption = self.caption(rig)
        assert "pressures in kPa" in caption
        assert "permeability in mD" in caption

    def test_every_column_is_headed(self, tmp_path):
        rig = make_rig(tmp_path)
        add_series(rig)
        header = self.header(rig)
        for column in ("P_in", "P_out", "P_mean", "k", "U(k)"):
            assert column in header

    def widest(self, rig):
        return max(len(row) for row in (self.header(rig), *self.table(rig)))

    def test_a_steady_state_table_fits_a_normal_terminal(self, tmp_path):
        """A wrapped table is unreadable, and this one grew by two columns."""
        rig = make_rig(tmp_path)
        add_series(rig)
        assert self.widest(rig) <= 120, self.widest(rig)

    def test_the_pulse_column_does_not_blow_it_open(self, tmp_path):
        """dP0 costs ten more columns, and only on plugs that ran pulse decay."""
        rig = make_rig(tmp_path)
        add_series(rig)
        write_measured_run(
            rig / "runs", "core-041",
            datetime(2026, 1, 11, 9, tzinfo=timezone.utc),
            mean_pressure_atm=30.0, permeability_darcy=1.2e-5,
            method="pulse_decay", pulse_amplitude_atm=0.5,
        )
        assert self.widest(rig) <= 130, self.widest(rig)

    def test_pressures_are_shown_in_the_rigs_display_unit(self, tmp_path):
        """Not atm: the summary must agree with what collect printed."""
        from gasperm import units

        rig = make_rig(tmp_path)
        add_series(rig, pressures=(10.0,))
        row = self.table(rig)[0]
        # write_measured_run puts the pair at 1.5x and 0.5x the mean.
        assert f"{units.from_atm(15.0, 'kPa'):.5g}" in row
        assert f"{units.from_atm(5.0, 'kPa'):.5g}" in row
        assert f"{units.from_atm(10.0, 'kPa'):.5g}" in row
        assert " 10 " not in row  # ...and not the raw atm value

    def test_another_display_unit_is_followed(self, tmp_path):
        from gasperm import units
        from gasperm.config import RUN_FILENAME

        rig = make_rig(tmp_path)
        add_series(rig, pressures=(10.0,))
        path = rig / RUN_FILENAME
        text = path.read_text(encoding="utf-8")
        path.write_text(
            text.replace('display_pressure_unit: "kPa"', 'display_pressure_unit: "bar"')
            .replace("display_pressure_unit: kPa", "display_pressure_unit: bar"),
            encoding="utf-8",
        )
        assert "pressures in bar" in self.caption(rig)
        assert f"{units.from_atm(10.0, 'bar'):.5g}" in self.table(rig)[0]

    def test_p_mean_is_the_midpoint_of_the_pair_it_is_shown_with(self, tmp_path):
        """The property that makes the three columns checkable at a glance."""
        rig = make_rig(tmp_path)
        add_series(rig, pressures=(10.0,))
        row = self.table(rig)[0].split()
        p_in, p_out, p_mean = (float(row[3]), float(row[4]), float(row[5]))
        assert (p_in + p_out) / 2 == pytest.approx(p_mean, rel=1e-4)

    def test_a_run_recorded_before_the_pair_existed_reads_as_unknown(self, tmp_path):
        """It cannot be recovered from a mean, so it is not guessed at."""
        rig = make_rig(tmp_path)
        directory = write_measured_run(
            rig / "runs", "core-041",
            datetime(2026, 1, 10, 9, tzinfo=timezone.utc),
            mean_pressure_atm=10.0, permeability_darcy=9e-4,
        )
        path = directory / "run_metadata.yaml"
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        del payload["summary"]["mean_inlet_pressure_atm"]
        del payload["summary"]["mean_downstream_pressure_atm"]
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        row = self.table(rig)[0]
        assert "--" in row
        # The mean it does have is still reported.
        assert f"{1013.25:.5g}" in row

    def test_no_dp0_column_on_a_plug_measured_only_in_steady_state(self, tmp_path):
        """An always-empty column is the same waste as an always-empty panel."""
        rig = make_rig(tmp_path)
        add_series(rig)
        assert "dP0" not in self.header(rig)

    def test_a_pulse_run_reports_the_pulse_it_started_from(self, tmp_path):
        from gasperm import units

        rig = make_rig(tmp_path)
        write_measured_run(
            rig / "runs", "core-041",
            datetime(2026, 1, 10, 9, tzinfo=timezone.utc),
            mean_pressure_atm=30.0, permeability_darcy=1.2e-5,
            method="pulse_decay", pulse_amplitude_atm=0.5,
        )
        assert "dP0" in self.header(rig)
        assert f"{units.from_atm(0.5, 'kPa'):.5g}" in self.table(rig)[0]

    def test_a_steady_row_leaves_the_dp0_cell_blank(self, tmp_path):
        """Blank, not '--': there is no pulse, which is not a missing pulse."""
        from gasperm import units

        rig = make_rig(tmp_path)
        add_series(rig, pressures=(10.0,))
        write_measured_run(
            rig / "runs", "core-041",
            datetime(2026, 1, 11, 9, tzinfo=timezone.utc),
            mean_pressure_atm=30.0, permeability_darcy=1.2e-5,
            method="pulse_decay", pulse_amplitude_atm=0.5,
            flowmeter="low_range",
        )
        rows = {row.split()[2]: row for row in self.table(rig)}
        assert "--" not in rows["steady_state"].split()[6:7]
        assert rows["pulse_decay"].split()[6] == f"{units.from_atm(0.5, 'kPa'):.5g}"


class TestSupersession:
    """A re-derived run replaces its parent; one measurement, counted once."""

    def derive(self, rig: Path, original_name: str) -> Path:
        """What `reprocess --write` leaves behind."""
        import shutil

        source = rig / "runs" / original_name
        target = source.with_name(source.name + "_reprocessed")
        shutil.copytree(source, target)
        payload = yaml.safe_load(
            (target / "run_metadata.yaml").read_text(encoding="utf-8")
        )
        payload["derived_from"] = {"run": original_name}
        (target / "run_metadata.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )
        return target

    def test_the_parent_is_not_counted_twice(self, tmp_path):
        rig = make_rig(tmp_path)
        add_series(rig)
        original = sorted((rig / "runs").iterdir())[0].name
        self.derive(rig, original)
        output = strip_ansi(summarize(rig, "core-041").output)
        assert "3 confirmed run(s)" in output
        assert "superseded by" in output

    def test_klinkenberg_does_not_regress_it_twice(self, tmp_path):
        """The bug this exists to prevent: one experiment, two points."""
        rig = make_rig(tmp_path)
        add_series(rig)
        self.derive(rig, sorted((rig / "runs").iterdir())[0].name)
        result = runner.invoke(
            app, ["klinkenberg", "--sample", "core-041", "-c", str(rig)]
        )
        output = strip_ansi(result.output)
        assert "superseded by" in output
        assert "3 points" in output or "Found 3 runs" in output

    def test_the_derived_run_is_the_one_kept(self, tmp_path):
        rig = make_rig(tmp_path)
        add_series(rig)
        original = sorted((rig / "runs").iterdir())[0].name
        self.derive(rig, original)
        output = strip_ansi(summarize(rig, "core-041").output)
        assert f"{original}_reprocessed" in output
        assert "was re-derived from" in output


class TestSummaryFile:
    def test_it_writes_the_history_and_the_findings(self, tmp_path):
        rig = make_rig(tmp_path)
        add_series(rig, porosity_fraction=0.11)
        target = tmp_path / "core-041.yaml"
        result = summarize(rig, "core-041", "--output", str(target))
        assert result.exit_code == 0, result.output

        payload = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert payload["sample"]["id"] == "core-041"
        assert payload["sample"]["porosity_fraction"] == pytest.approx(0.11)
        assert payload["history"]["confirmed_runs"] == 3
        assert payload["history"]["distinct_mean_pressures"] == 3
        assert payload["klinkenberg"]["liquid_permeability_mD"] == pytest.approx(0.5, rel=1e-3)
        assert len(payload["runs"]) == 3
        # The findings are the part a script can act on.
        assert isinstance(payload["findings"], list)

    def test_the_written_runs_carry_the_pressure_pair(self, tmp_path):
        """The file is what a script reads; it must not be thinner than the table.

        Written in atm, not the display unit: a file is parsed rather than read,
        and the stored unit is the one every other number in it is already in.
        """
        rig = make_rig(tmp_path)
        add_series(rig, pressures=(10.0,))
        target = tmp_path / "core-041.yaml"
        assert summarize(rig, "core-041", "--output", str(target)).exit_code == 0
        run = yaml.safe_load(target.read_text(encoding="utf-8"))["runs"][0]
        assert run["inlet_pressure_atm"] == pytest.approx(15.0)
        assert run["downstream_pressure_atm"] == pytest.approx(5.0)
        assert run["mean_pressure_atm"] == pytest.approx(10.0)
        assert run["pulse_amplitude_atm"] is None

    def test_a_written_pulse_run_carries_its_dp0(self, tmp_path):
        rig = make_rig(tmp_path)
        write_measured_run(
            rig / "runs", "core-041",
            datetime(2026, 1, 10, 9, tzinfo=timezone.utc),
            mean_pressure_atm=30.0, permeability_darcy=1.2e-5,
            method="pulse_decay", pulse_amplitude_atm=0.5,
        )
        target = tmp_path / "core-041.yaml"
        assert summarize(rig, "core-041", "--output", str(target)).exit_code == 0
        run = yaml.safe_load(target.read_text(encoding="utf-8"))["runs"][0]
        assert run["pulse_amplitude_atm"] == pytest.approx(0.5)

    def test_an_excluded_run_is_written_with_its_reason(self, tmp_path):
        rig = make_rig(tmp_path)
        add_series(rig)
        unconfirm(
            write_measured_run(
                rig / "runs", "core-041",
                datetime(2026, 1, 10, 15, tzinfo=timezone.utc),
                mean_pressure_atm=12.0, permeability_darcy=9e-4,
            )
        )
        target = tmp_path / "core-041.yaml"
        summarize(rig, "core-041", "--output", str(target))
        payload = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert len(payload["excluded"]) == 1
        assert payload["excluded"][0]["excluded_reason"]

    def test_the_flags_are_documented_in_help(self):
        result = runner.invoke(app, ["summarize", "--help"], env={"COLUMNS": "200"})
        output = strip_ansi(result.output)
        for flag in ("--output", "--allow-unsteady", "--runs-dir"):
            assert flag in output
