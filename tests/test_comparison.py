"""``gasperm compare``: the paired-difference statistics and what they refuse.

The arithmetic here is checked against numbers worked out by hand rather than
against itself, because the whole value of a paired comparison rests on one
claim -- that some uncertainties cancel -- and a round-trip test would assert
that claim by assuming it.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone

import matplotlib
import pytest

matplotlib.use("Agg")  # noqa: E402 - must precede any pyplot import

from typer.testing import CliRunner  # noqa: E402

from gasperm.cli import app  # noqa: E402
from gasperm.comparison import (  # noqa: E402
    ComparisonError,
    MeasurementGroup,
    combine_pairings,
    compare_groups,
    compare_quantity,
    klinkenberg_uncertainty_split,
    pair_components,
    slippage_explained_ratio,
)
from gasperm.models import (  # noqa: E402
    KlinkenbergPoint,
    KlinkenbergResult,
    UncertaintyBudget,
    UncertaintyComponent,
)

from conftest import build_budget, write_measured_run  # noqa: E402

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def component(
    symbol, u_rel, *, sensitivity=1.0, kind="B", source="instrument", value=1.0,
    dof=math.inf, name=None,
):
    return UncertaintyComponent(
        name=name or symbol, symbol=symbol, evaluation_type=kind, value=value,
        unit="", standard_uncertainty=abs(u_rel * value),
        relative_standard_uncertainty=u_rel, relative_sensitivity=sensitivity,
        relative_contribution=abs(sensitivity * u_rel), degrees_of_freedom=dof,
        source=source,
    )


def budget(components, value=1.0):
    variance = sum(c.relative_contribution**2 for c in components)
    return UncertaintyBudget(
        value_darcy=value,
        combined_standard_uncertainty_darcy=math.sqrt(variance) * value,
        relative_combined_standard_uncertainty=math.sqrt(variance),
        effective_degrees_of_freedom=math.inf, coverage_factor=2.0,
        coverage_probability=0.95,
        expanded_uncertainty_darcy=2 * math.sqrt(variance) * value,
        components=components,
    )


class TestCancellation:
    """Which inputs drop out of a ratio, and by exactly how much."""

    def test_a_shared_systematic_cancels_completely(self):
        """Same plug, same calipers, same recorded number: one error, not two."""
        left = budget([component("L", 0.03, source="calipers", value=5.0)])
        right = budget([component("L", 0.03, source="calipers", value=5.0)])
        u, _ = combine_pairings(pair_components([left], [right], same_sample=True))
        assert u == pytest.approx(0.0, abs=1e-12)

    def test_type_a_never_cancels(self):
        """Scatter is an independent draw each run, however alike the runs are."""
        left = budget([component("rep", 0.01, kind="A", source="scatter")])
        right = budget([component("rep", 0.01, kind="A", source="scatter")])
        u, _ = combine_pairings(pair_components([left], [right], same_sample=True))
        assert u == pytest.approx(math.hypot(0.01, 0.01))

    def test_the_ratio_beats_either_absolute_uncertainty(self):
        """The point of the whole exercise, stated as a number."""
        components = [
            component("L", 0.03, source="calipers", value=5.0),
            component("rep", 0.01, kind="A", source="scatter"),
        ]
        left, right = budget(list(components)), budget(list(components))
        absolute = left.relative_combined_standard_uncertainty
        paired, _ = combine_pairings(pair_components([left], [right], same_sample=True))
        assert absolute == pytest.approx(math.hypot(0.03, 0.01))
        assert paired == pytest.approx(math.hypot(0.01, 0.01))
        assert paired < absolute / 2

    def test_a_shared_input_read_at_different_values_leaves_a_residue(self):
        """Not a special case: the difference of contributions IS the formula."""
        left = budget([component("P", 0.10, source="transducer", value=10.0)])
        right = budget([component("P", 0.08, source="transducer", value=12.5)])
        u, _ = combine_pairings(pair_components([left], [right], same_sample=True))
        assert u == pytest.approx(abs(0.10 - 0.08))

    def test_re_measured_geometry_stops_cancelling(self):
        """Two caliper readings are two independent errors, not one shared one."""
        left = budget([component("L", 0.03, source="calipers", value=5.0)])
        right = budget([component("L", 0.03, source="calipers", value=5.02)])
        pairing = pair_components([left], [right], same_sample=True)[0]
        assert pairing.shared is False
        assert "measured again" in pairing.reason
        assert combine_pairings([pairing])[0] == pytest.approx(math.hypot(0.03, 0.03))

    def test_plug_inputs_do_not_cancel_between_different_plugs(self):
        left = budget([component("L", 0.03, source="calipers", value=5.0)])
        right = budget([component("L", 0.03, source="calipers", value=5.0)])
        pairing = pair_components([left], [right], same_sample=False)[0]
        assert pairing.shared is False
        assert "different plug" in pairing.reason

    def test_rig_inputs_still_cancel_between_different_plugs(self):
        """Two plugs on one bench share its transducer, whatever else differs."""
        left = budget([component("Q", 0.02, source="flowmeter spec", value=1.5)])
        right = budget([component("Q", 0.02, source="flowmeter spec", value=1.5)])
        pairing = pair_components([left], [right], same_sample=False)[0]
        assert pairing.shared is True

    def test_a_different_instrument_does_not_cancel(self):
        left = budget([component("Q", 0.02, source="low_range meter", value=1.5)])
        right = budget([component("Q", 0.02, source="high_range meter", value=1.5)])
        pairing = pair_components([left], [right], same_sample=True)[0]
        assert pairing.shared is False
        assert "different source" in pairing.reason

    def test_a_voided_symbol_is_forced_independent(self):
        left = budget([component("Q", 0.02, source="meter", value=1.5)])
        right = budget([component("Q", 0.02, source="meter", value=1.5)])
        pairing = pair_components(
            [left], [right], same_sample=True, force_independent=["Q"]
        )[0]
        assert pairing.shared is False
        assert "conditions differed" in pairing.reason


class TestDegreesOfFreedom:
    """Welch-Satterthwaite over what survives, not over what went in."""

    def test_two_independent_type_a_terms_give_both_their_dof(self):
        """Collapsing them into one term at the smaller dof would inflate k."""
        left = budget([component("rep", 0.01, kind="A", source="scatter", dof=9)])
        right = budget([component("rep", 0.01, kind="A", source="scatter", dof=9)])
        _, dof = combine_pairings(pair_components([left], [right], same_sample=True))
        assert dof == pytest.approx(18.0)

    def test_asymmetric_dof_follows_welch(self):
        left = budget([component("rep", 0.01, kind="A", source="scatter", dof=9)])
        right = budget([component("rep", 0.02, kind="A", source="scatter", dof=4)])
        _, dof = combine_pairings(pair_components([left], [right], same_sample=True))
        variance_a, variance_b = 0.01**2, 0.02**2
        expected = (variance_a + variance_b) ** 2 / (
            variance_a**2 / 9 + variance_b**2 / 4
        )
        assert dof == pytest.approx(expected)

    def test_a_cancelled_input_contributes_no_degrees_of_freedom(self):
        """It contributes no variance either; both follow from it not being there."""
        shared = component("L", 0.03, source="calipers", value=5.0)
        random = component("rep", 0.01, kind="A", source="scatter", dof=9)
        left, right = budget([shared, random]), budget([shared, random])
        _, dof = combine_pairings(pair_components([left], [right], same_sample=True))
        assert dof == pytest.approx(18.0)


class TestVerdict:
    def test_a_change_inside_its_uncertainty_is_not_significant(self):
        change = compare_quantity(
            "k", "k_L", "darcy", 1.0, 1.02,
            relative_uncertainty=0.05, effective_dof=math.inf, paired=True,
        )
        assert change.significant is False
        assert "NOT distinguishable" in change.verdict

    def test_a_change_beyond_it_is(self):
        change = compare_quantity(
            "k", "k_L", "darcy", 1.0, 1.30,
            relative_uncertainty=0.05, effective_dof=math.inf, paired=True,
        )
        assert change.significant is True
        assert "SIGNIFICANT" in change.verdict
        assert "increased 30" in change.verdict

    def test_the_minimum_detectable_change_is_reported(self):
        """A null result means nothing without the limit it was null against."""
        change = compare_quantity(
            "k", "k_L", "darcy", 1.0, 1.01,
            relative_uncertainty=0.04, effective_dof=math.inf, paired=True,
        )
        assert change.minimum_detectable_percent == pytest.approx(
            change.coverage_factor * 4.0
        )
        assert change.minimum_detectable_percent > abs(change.percent_change)

    def test_a_decrease_is_named_as_one(self):
        change = compare_quantity(
            "k", "k_L", "darcy", 1.0, 0.5,
            relative_uncertainty=0.01, effective_dof=math.inf, paired=True,
        )
        assert "decreased 50" in change.verdict


class TestKlinkenbergSplit:
    def fit(self, *, weighted, stderr, intercept=1.0, points=4):
        return KlinkenbergResult(
            liquid_permeability_darcy=intercept, slippage_factor_atm=4.0,
            slope=4.0 * intercept, intercept=intercept, r_squared=0.999,
            intercept_stderr=stderr, weighted=weighted,
            points=[
                KlinkenbergPoint(
                    mean_pressure_atm=float(index + 2),
                    apparent_permeability_darcy=intercept,
                )
                for index in range(points)
            ],
        )

    def test_an_unweighted_fit_keeps_its_standard_error(self):
        """Its residuals are already a purely random estimate."""
        budgets = [budget([
            component("L", 0.03, source="calipers", value=5.0),
            component("rep", 0.01, kind="A", source="scatter"),
        ])]
        fit_rel, systematic = klinkenberg_uncertainty_split(
            self.fit(weighted=False, stderr=0.02), budgets
        )
        assert fit_rel == pytest.approx(0.02)
        assert systematic == pytest.approx(0.03)

    def test_a_weighted_fit_is_scaled_to_its_random_share(self):
        """It propagated the systematic too; counting it twice would double it."""
        budgets = [budget([
            component("L", 0.04, source="calipers", value=5.0),
            component("rep", 0.03, kind="A", source="scatter"),
        ])]
        fit_rel, systematic = klinkenberg_uncertainty_split(
            self.fit(weighted=True, stderr=0.02), budgets
        )
        assert systematic == pytest.approx(0.04)
        # random share = 0.03 / hypot(0.03, 0.04) = 0.6
        assert fit_rel == pytest.approx(0.02 * 0.6)

    def test_a_zero_intercept_is_not_divided_by(self):
        fit_rel, systematic = klinkenberg_uncertainty_split(
            self.fit(weighted=False, stderr=0.02, intercept=0.0), []
        )
        assert math.isinf(fit_rel) and math.isinf(systematic)


class TestKlinkenbergComparisonArithmetic:
    """The k_L ratio's uncertainty, worked out by hand and pinned.

    Two ways to double-count converge here, and neither changes the verdict on
    an obvious change -- so only the number itself can catch them.
    """

    L_UNCERTAINTY = 0.04      # Type B, shared: same plug, same calipers
    REP_UNCERTAINTY = 0.03    # Type A, independent every run
    INTERCEPT_STDERR = 0.02   # from a weighted fit, so it carries both

    def group_with_fit(self, sample_id, intercept):
        from gasperm.models import ExperimentMetadata, RunSummary

        components = [
            component("L", self.L_UNCERTAINTY, source="calipers", value=5.0),
            component("rep", self.REP_UNCERTAINTY, kind="A", source="scatter", dof=9),
        ]
        summaries = [
            RunSummary(
                sample_id=sample_id, gas_name="Nitrogen",
                started_at=datetime(2026, 1, 1 + index, tzinfo=timezone.utc),
                ended_at=datetime(2026, 1, 1 + index, tzinfo=timezone.utc),
                duration_s=60.0, sample_count=600, steady_state_reached=True,
                measurement_confirmed=True, mean_pressure_atm=float(index + 2),
                permeability_darcy=intercept, permeability_stddev_darcy=0.0,
                mean_temperature_c=22.0, averaged_samples=50,
                uncertainty=budget(list(components), value=intercept),
                metadata=ExperimentMetadata(
                    flowmeter="low_range", sample_id=sample_id,
                    length_cm=5.0, diameter_cm=3.81,
                ),
            )
            for index in range(4)
        ]
        fit = KlinkenbergResult(
            liquid_permeability_darcy=intercept, slippage_factor_atm=4.0,
            slope=4.0 * intercept, intercept=intercept, r_squared=0.999,
            intercept_stderr=self.INTERCEPT_STDERR * intercept,
            slippage_factor_standard_uncertainty_atm=0.2, weighted=True,
            points=[
                KlinkenbergPoint(
                    mean_pressure_atm=float(index + 2),
                    apparent_permeability_darcy=intercept,
                )
                for index in range(4)
            ],
        )
        return MeasurementGroup(
            label=sample_id, sample_id=sample_id, summaries=tuple(summaries),
            klinkenberg=fit, downstream_conventions=("measured",) * 4,
        )

    def expected_relative_uncertainty(self) -> float:
        """Hand-worked, stated as the arithmetic rather than as a constant.

        The weighted fit's intercept standard error propagated *both* halves of
        each point's budget, so it is scaled to the random share; the shared
        systematic then cancels to nothing; and the repeatability is NOT added
        again, because the scaled fit term already is the repeatability
        propagated into the intercept.
        """
        random_share = self.REP_UNCERTAINTY / math.hypot(
            self.REP_UNCERTAINTY, self.L_UNCERTAINTY
        )
        per_side = self.INTERCEPT_STDERR * random_share
        return math.hypot(per_side, per_side)

    def test_the_ratio_uncertainty_matches_the_hand_calculation(self):
        result = compare_groups(
            self.group_with_fit("core-041", 1.0e-3),
            self.group_with_fit("core-041", 1.12e-3),
        )
        change = result.change("k_L")
        assert change.relative_standard_uncertainty == pytest.approx(
            self.expected_relative_uncertainty(), rel=1e-9
        )

    def test_the_repeatability_is_not_charged_twice(self):
        """Once through the fit's own standard error is once."""
        result = compare_groups(
            self.group_with_fit("core-041", 1.0e-3),
            self.group_with_fit("core-041", 1.12e-3),
        )
        change = result.change("k_L")
        double_counted = math.hypot(
            self.expected_relative_uncertainty(),
            math.hypot(self.REP_UNCERTAINTY, self.REP_UNCERTAINTY),
        )
        assert change.relative_standard_uncertainty < double_counted / 2

    def test_the_shared_systematic_is_absent_from_the_ratio(self):
        """It is present in each absolute value and gone from their ratio."""
        result = compare_groups(
            self.group_with_fit("core-041", 1.0e-3),
            self.group_with_fit("core-041", 1.12e-3),
        )
        change = result.change("k_L")
        assert change.relative_standard_uncertainty < self.L_UNCERTAINTY

    def test_between_different_plugs_the_geometry_is_charged(self):
        result = compare_groups(
            self.group_with_fit("core-041", 1.0e-3),
            self.group_with_fit("core-042", 1.12e-3),
        )
        change = result.change("k_L")
        expected = math.hypot(
            self.expected_relative_uncertainty(),
            math.hypot(self.L_UNCERTAINTY, self.L_UNCERTAINTY),
        )
        assert change.relative_standard_uncertainty == pytest.approx(expected, rel=1e-9)

    def test_the_slippage_factor_is_compared_too(self):
        result = compare_groups(
            self.group_with_fit("core-041", 1.0e-3),
            self.group_with_fit("core-041", 1.12e-3),
        )
        assert result.change("b") is not None


class TestSlippageExplained:
    def test_no_pressure_change_explains_nothing(self):
        assert slippage_explained_ratio(4.0, 10.0, 10.0) == pytest.approx(1.0)

    def test_a_higher_mean_pressure_lowers_apparent_permeability(self):
        """Less slip at higher pressure -- with the rock entirely unchanged."""
        assert slippage_explained_ratio(4.0, 10.0, 12.0) < 1.0

    def test_it_matches_the_klinkenberg_form(self):
        ratio = slippage_explained_ratio(4.0, 10.0, 20.0)
        assert ratio == pytest.approx((1.0 + 4.0 / 20.0) / (1.0 + 4.0 / 10.0))

    def test_no_slippage_means_no_pressure_dependence(self):
        assert slippage_explained_ratio(0.0, 5.0, 30.0) == pytest.approx(1.0)


def group(label, sample_id, *, permeabilities, pressures, **kwargs):
    """A MeasurementGroup built from summaries, without touching a filesystem."""
    from gasperm.models import ExperimentMetadata, RunSummary

    summaries = []
    for index, (k, p) in enumerate(zip(permeabilities, pressures)):
        summaries.append(
            RunSummary(
                sample_id=sample_id, gas_name=kwargs.get("gas", "Nitrogen"),
                started_at=datetime(2026, 1, 1 + index, tzinfo=timezone.utc),
                ended_at=datetime(2026, 1, 1 + index, tzinfo=timezone.utc),
                duration_s=60.0, sample_count=600,
                method=kwargs.get("method", "steady_state"),
                steady_state_reached=True, measurement_confirmed=True,
                mean_pressure_atm=p, permeability_darcy=k,
                permeability_stddev_darcy=k * 0.01, mean_temperature_c=22.0,
                averaged_samples=50,
                uncertainty=build_budget(
                    k, geometry={"L": kwargs.get("length_cm", 5.0), "d": 3.81}
                ),
                metadata=ExperimentMetadata(
                    flowmeter=kwargs.get("flowmeter", "low_range"),
                    sample_id=sample_id, gas_name=kwargs.get("gas", "Nitrogen"),
                    length_cm=kwargs.get("length_cm", 5.0), diameter_cm=3.81,
                    porosity_fraction=kwargs.get("porosity"),
                ),
            )
        )
    return MeasurementGroup(
        label=label, sample_id=sample_id, summaries=tuple(summaries),
        porosity_fraction=kwargs.get("porosity"),
        porosity_uncertainty=kwargs.get("porosity_uncertainty"),
        downstream_conventions=tuple(
            kwargs.get("convention", "measured") for _ in summaries
        ),
    )


class TestCompareGroups:
    def test_a_planted_change_is_recovered_exactly(self):
        before = group("before", "core-041", permeabilities=[1e-3], pressures=[10.0])
        after = group("after", "core-041", permeabilities=[1.12e-3], pressures=[10.0])
        result = compare_groups(before, after)
        change = result.change("k_g")
        assert change.percent_change == pytest.approx(12.0)
        assert change.significant is True
        assert result.paired is True

    def test_matching_plugs_are_detected_without_being_told(self):
        before = group("a", "core-041", permeabilities=[1e-3], pressures=[10.0])
        after = group("b", "core-041", permeabilities=[1e-3], pressures=[10.0])
        assert compare_groups(before, after).paired is True

    def test_different_plugs_are_unpaired_and_said_to_be(self):
        before = group("a", "core-041", permeabilities=[1e-3], pressures=[10.0])
        after = group("b", "core-042", permeabilities=[1e-3], pressures=[10.0])
        result = compare_groups(before, after)
        assert result.paired is False
        assert any("different plugs" in w for w in result.warnings)

    def test_pairing_can_be_forced(self):
        before = group("a", "core-041", permeabilities=[1e-3], pressures=[10.0])
        after = group("b", "core-041-post", permeabilities=[1e-3], pressures=[10.0])
        assert compare_groups(before, after, paired=True).paired is True

    def test_mixing_methods_is_refused(self):
        """The two methods carry a systematic offset relative to each other."""
        before = group("a", "core-041", permeabilities=[1e-3], pressures=[10.0])
        after = group(
            "b", "core-041", permeabilities=[1e-3], pressures=[10.0],
            method="pulse_decay",
        )
        with pytest.raises(ComparisonError, match="comparable conditions"):
            compare_groups(before, after)

    def test_mixing_gases_is_refused(self):
        before = group("a", "core-041", permeabilities=[1e-3], pressures=[10.0])
        after = group(
            "b", "core-041", permeabilities=[1e-3], pressures=[10.0], gas="Helium"
        )
        with pytest.raises(ComparisonError, match="comparable conditions"):
            compare_groups(before, after)

    def test_a_refusal_can_be_overridden_and_stays_flagged(self):
        before = group("a", "core-041", permeabilities=[1e-3], pressures=[10.0])
        after = group(
            "b", "core-041", permeabilities=[1e-3], pressures=[10.0], gas="Helium"
        )
        result = compare_groups(before, after, allow_mismatched_conditions=True)
        assert any(c.blocking for c in result.mismatched_conditions)

    def test_a_changed_flowmeter_voids_its_cancellation(self):
        """Survivable, but the meter is now charged to the comparison in full."""
        before = group("a", "core-041", permeabilities=[1e-3], pressures=[10.0])
        after = group(
            "b", "core-041", permeabilities=[1e-3], pressures=[10.0],
            flowmeter="high_range",
        )
        result = compare_groups(before, after)
        flow = next(p for p in result.component_pairings if p.symbol == "Q")
        assert flow.shared is False
        assert any(c.key == "flowmeter" and not c.matched for c in result.conditions)

    def test_a_changed_p2_convention_is_refused(self):
        before = group("a", "core-041", permeabilities=[1e-3], pressures=[10.0])
        after = group(
            "b", "core-041", permeabilities=[1e-3], pressures=[10.0],
            convention="fixed:1.0",
        )
        with pytest.raises(ComparisonError, match="comparable conditions"):
            compare_groups(before, after)

    def test_unmatched_pressures_are_reported_not_silently_paired(self):
        before = group("a", "core-041", permeabilities=[1e-3], pressures=[10.0])
        after = group("b", "core-041", permeabilities=[1e-3], pressures=[30.0])
        result = compare_groups(before, after)
        assert any("no counterpart" in w for w in result.warnings)
        assert result.change("k_g") is None

    def test_porosity_is_compared_when_both_sides_have_it(self):
        before = group(
            "a", "core-041", permeabilities=[1e-3], pressures=[10.0],
            porosity=0.10, porosity_uncertainty=0.002,
        )
        after = group(
            "b", "core-041", permeabilities=[1e-3], pressures=[10.0],
            porosity=0.12, porosity_uncertainty=0.002,
        )
        change = compare_groups(before, after).change("phi")
        assert change.percent_change == pytest.approx(20.0)
        assert change.significant is True

    def test_an_empty_side_is_refused(self):
        before = group("a", "core-041", permeabilities=[1e-3], pressures=[10.0])
        empty = MeasurementGroup(label="b", sample_id="core-041", summaries=())
        with pytest.raises(ComparisonError, match="at least one confirmed run"):
            compare_groups(before, empty)


def make_rig(tmp_path, *, change=1.12, pressures=(5.0, 10.0, 20.0), **after_kwargs):
    """A rig folder with two campaigns on one plug, split by date."""
    rig = tmp_path / "rig"
    result = runner.invoke(
        app, ["init", str(rig), "--non-interactive", "--force"]
    )
    assert result.exit_code == 0, result.output
    for day, factor in ((10, 1.0), (20, change)):
        for index, pressure in enumerate(pressures):
            extra = dict(after_kwargs) if day == 20 else {}
            write_measured_run(
                rig / "runs", "core-041",
                datetime(2026, 1, day, 9 + index, tzinfo=timezone.utc),
                mean_pressure_atm=pressure,
                permeability_darcy=0.5e-3 * factor * (1.0 + 4.0 / pressure),
                **extra,
            )
    return rig


class TestCompareCommand:
    def compare(self, rig, *args):
        return runner.invoke(
            app, ["compare", "core-041", "--split", "2026-01-15", "-c", str(rig), *args]
        )

    def test_it_finds_a_planted_change(self, tmp_path):
        result = self.compare(make_rig(tmp_path, change=1.12))
        assert result.exit_code == 0, result.output
        output = strip_ansi(result.output)
        assert "increased 12.00%" in output
        assert "SIGNIFICANT" in output

    def test_a_change_below_the_detection_limit_exits_2(self, tmp_path):
        """So a screening script can branch on 'this plug moved' without parsing."""
        result = self.compare(make_rig(tmp_path, change=1.005))
        assert result.exit_code == 2
        assert "NOT distinguishable" in strip_ansi(result.output)

    def test_the_report_itemises_what_cancelled(self, tmp_path):
        result = self.compare(make_rig(tmp_path))
        output = strip_ansi(result.output)
        assert "What cancelled between the two measurements" in output
        assert "sample length" in output
        assert "What did not" in output
        assert "repeatability" in output

    def test_re_measured_geometry_is_caught_and_explained(self, tmp_path):
        result = self.compare(make_rig(tmp_path, length_cm=5.02))
        output = strip_ansi(result.output)
        assert "plug geometry" in output and "DIFFER" in output
        assert "measured again" in output

    def test_a_changed_meter_is_flagged(self, tmp_path):
        result = self.compare(make_rig(tmp_path, flowmeter="high_range"))
        output = strip_ansi(result.output)
        assert "flowmeter" in output
        assert "no longer cancels" in output

    def test_both_uncertainties_are_shown(self, tmp_path):
        """u_c and U can move in opposite directions; only U would mislead."""
        result = self.compare(make_rig(tmp_path))
        output = strip_ansi(result.output)
        assert "u_c =" in output and "v_eff" in output

    def test_two_plugs_compare_without_a_split(self, tmp_path):
        rig = make_rig(tmp_path)
        for index, pressure in enumerate((5.0, 10.0, 20.0)):
            write_measured_run(
                rig / "runs", "core-042",
                datetime(2026, 2, 1, 9 + index, tzinfo=timezone.utc),
                mean_pressure_atm=pressure,
                permeability_darcy=0.9e-3 * (1.0 + 4.0 / pressure),
            )
        result = runner.invoke(
            app, ["compare", "core-041", "core-042", "-c", str(rig)]
        )
        assert result.exit_code in (0, 2), result.output
        output = strip_ansi(result.output)
        assert "UNPAIRED" in output
        assert "different plugs" in output

    def test_a_split_with_nothing_on_one_side_is_refused(self, tmp_path):
        result = self.compare(make_rig(tmp_path), "--split", "2020-01-01")
        assert result.exit_code == 1

    def test_giving_both_a_selector_and_a_split_is_refused(self, tmp_path):
        rig = make_rig(tmp_path)
        result = runner.invoke(
            app, ["compare", "core-041", "core-042", "--split", "2026-01-15",
                  "-c", str(rig)]
        )
        assert result.exit_code == 1
        assert "not both" in strip_ansi(result.output)

    def test_giving_neither_says_what_to_do(self, tmp_path):
        rig = make_rig(tmp_path)
        result = runner.invoke(app, ["compare", "core-041", "-c", str(rig)])
        assert result.exit_code == 1
        assert "--split" in strip_ansi(result.output)

    def test_a_bad_split_date_is_refused(self, tmp_path):
        result = self.compare(make_rig(tmp_path), "--split", "last tuesday")
        assert result.exit_code == 1
        assert "not a date" in strip_ansi(result.output)

    def test_leak_tests_are_excluded(self, tmp_path):
        """A leak belongs to the rig, not to any plug."""
        rig = make_rig(tmp_path)
        write_measured_run(
            rig / "runs", "core-041",
            datetime(2026, 1, 21, tzinfo=timezone.utc),
            mean_pressure_atm=10.0, permeability_darcy=1e-9,
            method="pulse_decay", purpose="leak_test",
        )
        result = self.compare(rig)
        assert result.exit_code in (0, 2), result.output
        assert "leak test(s) excluded" in strip_ansi(result.output)

    def test_the_result_can_be_written_out(self, tmp_path):
        import yaml

        target = tmp_path / "comparison.yaml"
        result = self.compare(make_rig(tmp_path), "--output", str(target))
        assert result.exit_code in (0, 2), result.output
        payload = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert payload["comparison"]["paired"] is True
        assert any(c["symbol"] == "k_L" for c in payload["changes"])
        # The evidence for the cancellation claim, not just the conclusion.
        assert any(p["shared"] for p in payload["uncertainty_pairing"])

    def test_the_written_file_has_no_unparseable_infinities(self, tmp_path):
        import yaml

        target = tmp_path / "comparison.yaml"
        self.compare(make_rig(tmp_path), "--output", str(target))
        payload = yaml.safe_load(target.read_text(encoding="utf-8"))
        for change in payload["changes"]:
            for key, value in change.items():
                assert not isinstance(value, str) or value not in (".inf", "-.inf", ".nan")

    def test_the_plot_is_written_beside_the_output(self, tmp_path):
        target = tmp_path / "comparison.yaml"
        result = self.compare(
            make_rig(tmp_path), "--output", str(target), "--plot"
        )
        assert result.exit_code in (0, 2), result.output
        assert target.with_suffix(".png").is_file()

    def test_the_flags_are_documented_in_help(self):
        result = runner.invoke(app, ["compare", "--help"], env={"COLUMNS": "200"})
        output = strip_ansi(result.output)
        for flag in ("--split", "--paired", "--allow-mismatched-conditions", "--output"):
            assert flag in output
