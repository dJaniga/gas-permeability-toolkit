"""Klinkenberg regression against synthetic data from a known k_L and b."""

from __future__ import annotations


import pytest

from gasperm.klinkenberg import (
    POOR_FIT_R_SQUARED,
    fit_klinkenberg,
    load_points_from_csv,
)
from gasperm.models import KlinkenbergPoint

#: Ground truth for the synthetic points below.
TRUE_K_L = 0.5  # darcy
TRUE_B = 0.2  # atm


def synthetic_points(pressures_atm=(1.0, 2.0, 4.0, 8.0), *, k_l=TRUE_K_L, b=TRUE_B):
    """Exact ``k_g = k_L * (1 + b / P_mean)`` points -- no noise."""
    return [
        KlinkenbergPoint(
            mean_pressure_atm=p,
            apparent_permeability_darcy=k_l * (1.0 + b / p),
            label=f"P={p}",
            sample_id="core-001",
        )
        for p in pressures_atm
    ]


def _mixed_sample_points():
    """Three points where the middle one came from a different core plug."""
    points = list(synthetic_points((1.0, 2.0, 4.0)))
    points[1] = KlinkenbergPoint(
        mean_pressure_atm=points[1].mean_pressure_atm,
        apparent_permeability_darcy=points[1].apparent_permeability_darcy,
        label="run-b",
        sample_id="core-002",
    )
    return points


class TestExactRecovery:
    def test_recovers_k_l_and_b_from_noise_free_points(self):
        result = fit_klinkenberg(synthetic_points())
        assert result.liquid_permeability_darcy == pytest.approx(TRUE_K_L, rel=1e-9)
        assert result.slippage_factor_atm == pytest.approx(TRUE_B, rel=1e-9)
        assert result.r_squared == pytest.approx(1.0, abs=1e-12)
        assert result.warnings == []

    def test_slope_is_k_l_times_b(self):
        result = fit_klinkenberg(synthetic_points())
        assert result.slope == pytest.approx(TRUE_K_L * TRUE_B, rel=1e-9)
        assert result.intercept == pytest.approx(result.liquid_permeability_darcy)

    def test_prediction_reproduces_the_inputs(self):
        result = fit_klinkenberg(synthetic_points())
        for point in result.points:
            assert result.predict_darcy(point.mean_pressure_atm) == pytest.approx(
                point.apparent_permeability_darcy, rel=1e-9
            )

    @pytest.mark.parametrize(
        "k_l,b", [(0.001, 0.5), (1.5, 0.05), (0.02, 1.2), (250.0, 0.3)]
    )
    def test_recovery_across_magnitudes(self, k_l: float, b: float):
        result = fit_klinkenberg(
            synthetic_points((0.5, 1.0, 2.0, 5.0, 10.0), k_l=k_l, b=b)
        )
        assert result.liquid_permeability_darcy == pytest.approx(k_l, rel=1e-8)
        assert result.slippage_factor_atm == pytest.approx(b, rel=1e-8)

    def test_two_points_fit_exactly(self):
        result = fit_klinkenberg(synthetic_points((1.0, 4.0)))
        assert result.liquid_permeability_darcy == pytest.approx(TRUE_K_L, rel=1e-9)
        assert result.slippage_factor_atm == pytest.approx(TRUE_B, rel=1e-9)


class TestWarnings:
    def test_two_points_warn_that_r_squared_is_meaningless(self):
        result = fit_klinkenberg(synthetic_points((1.0, 4.0)))
        assert any("Two points fit an exact line" in w for w in result.warnings)

    def test_three_clean_points_do_not_warn(self):
        result = fit_klinkenberg(synthetic_points((1.0, 2.0, 4.0)))
        assert result.warnings == []

    def test_poor_fit_is_flagged(self):
        points = synthetic_points((1.0, 2.0, 4.0, 8.0))
        scattered = list(points)
        # Push one run badly off the line, as a non-steady-state run would be.
        scattered[2] = KlinkenbergPoint(
            mean_pressure_atm=scattered[2].mean_pressure_atm,
            apparent_permeability_darcy=scattered[2].apparent_permeability_darcy * 3.0,
            sample_id="core-001",
        )
        result = fit_klinkenberg(scattered)
        assert result.r_squared < POOR_FIT_R_SQUARED
        assert any("Poor linear fit" in w for w in result.warnings)

    def test_mixed_samples_are_refused(self):
        """On a rig running many plugs, mixing runs is an easy silent mistake."""
        mixed = _mixed_sample_points()
        with pytest.raises(ValueError, match="only valid within a single sample"):
            fit_klinkenberg(mixed)

    def test_the_refusal_names_the_offending_runs(self):
        with pytest.raises(ValueError) as info:
            fit_klinkenberg(_mixed_sample_points())
        assert "core-001" in str(info.value) and "core-002" in str(info.value)

    def test_mixing_can_be_forced_and_is_then_warned_about(self):
        result = fit_klinkenberg(_mixed_sample_points(), allow_mixed_samples=True)
        assert any("more than one sample id" in w for w in result.warnings)

    def test_negative_intercept_is_flagged(self):
        # An implausibly steep slippage trend extrapolates to a negative
        # intercept, which is non-physical -- usually too narrow a pressure
        # range, or one outlying run.
        points = [
            KlinkenbergPoint(mean_pressure_atm=1.0, apparent_permeability_darcy=1.0),
            KlinkenbergPoint(mean_pressure_atm=2.0, apparent_permeability_darcy=0.3),
            KlinkenbergPoint(mean_pressure_atm=4.0, apparent_permeability_darcy=0.1),
        ]
        result = fit_klinkenberg(points)
        assert result.liquid_permeability_darcy < 0.0
        assert any("not positive" in w for w in result.warnings)

    def test_a_negative_slope_is_flagged(self):
        """Apparent permeability must fall, not rise, with mean pressure."""
        points = [
            KlinkenbergPoint(mean_pressure_atm=1.0, apparent_permeability_darcy=0.1),
            KlinkenbergPoint(mean_pressure_atm=2.0, apparent_permeability_darcy=0.5),
            KlinkenbergPoint(mean_pressure_atm=4.0, apparent_permeability_darcy=0.9),
        ]
        result = fit_klinkenberg(points)
        assert result.slope < 0.0
        assert any("slope is negative" in w for w in result.warnings)


class TestUncertainty:
    def test_an_unweighted_fit_reports_the_intercept_standard_error(self):
        result = fit_klinkenberg(synthetic_points())
        assert result.weighted is False
        assert result.intercept_stderr is not None
        # Noise-free points fit exactly, so the standard error collapses.
        assert result.intercept_stderr == pytest.approx(0.0, abs=1e-12)

    def test_scatter_produces_a_finite_uncertainty_on_k_l(self):
        points = synthetic_points((1.0, 2.0, 4.0, 8.0))
        noisy = [
            KlinkenbergPoint(
                mean_pressure_atm=p.mean_pressure_atm,
                apparent_permeability_darcy=p.apparent_permeability_darcy * factor,
            )
            for p, factor in zip(points, (1.01, 0.99, 1.005, 0.995))
        ]
        result = fit_klinkenberg(noisy)
        assert result.intercept_stderr > 0.0
        assert result.liquid_permeability_expanded_uncertainty_darcy > result.intercept_stderr

    def test_the_coverage_factor_uses_n_minus_two_degrees_of_freedom(self):
        """Three points leave one degree of freedom, so t is large."""
        points = synthetic_points((1.0, 2.0, 4.0))
        noisy = [
            KlinkenbergPoint(
                mean_pressure_atm=p.mean_pressure_atm,
                apparent_permeability_darcy=p.apparent_permeability_darcy * factor,
            )
            for p, factor in zip(points, (1.01, 0.99, 1.005))
        ]
        result = fit_klinkenberg(noisy, coverage_probability=0.95)
        assert result.coverage_factor == pytest.approx(12.706, rel=1e-3)

    def test_two_points_cannot_support_an_uncertainty(self):
        result = fit_klinkenberg(synthetic_points((1.0, 4.0)))
        assert result.liquid_permeability_expanded_uncertainty_darcy is None
        assert result.coverage_factor is None

    def test_slippage_uncertainty_is_reported(self):
        points = synthetic_points((1.0, 2.0, 4.0, 8.0))
        noisy = [
            KlinkenbergPoint(
                mean_pressure_atm=p.mean_pressure_atm,
                apparent_permeability_darcy=p.apparent_permeability_darcy * factor,
            )
            for p, factor in zip(points, (1.01, 0.99, 1.005, 0.995))
        ]
        result = fit_klinkenberg(noisy)
        assert result.slippage_factor_standard_uncertainty_atm > 0.0


class TestWeighting:
    @staticmethod
    def _weighted_points(uncertainties):
        points = synthetic_points((1.0, 2.0, 4.0, 8.0))
        return [
            KlinkenbergPoint(
                mean_pressure_atm=p.mean_pressure_atm,
                apparent_permeability_darcy=p.apparent_permeability_darcy,
                standard_uncertainty_darcy=u,
                label=p.label,
            )
            for p, u in zip(points, uncertainties)
        ]

    def test_uncertainties_on_every_point_trigger_a_weighted_fit(self):
        result = fit_klinkenberg(self._weighted_points([0.001] * 4))
        assert result.weighted is True
        assert result.liquid_permeability_darcy == pytest.approx(TRUE_K_L, rel=1e-8)

    def test_equal_weights_reproduce_the_unweighted_answer(self):
        weighted = fit_klinkenberg(self._weighted_points([0.002] * 4))
        unweighted = fit_klinkenberg(synthetic_points())
        assert weighted.liquid_permeability_darcy == pytest.approx(
            unweighted.liquid_permeability_darcy, rel=1e-9
        )

    def test_a_badly_determined_point_is_down_weighted(self):
        """A low-differential run must not drag the intercept around."""
        points = self._weighted_points([0.001, 0.001, 0.001, 0.001])
        # Corrupt the point at the lowest pressure and mark it as poorly known.
        corrupted = list(points)
        corrupted[0] = KlinkenbergPoint(
            mean_pressure_atm=points[0].mean_pressure_atm,
            apparent_permeability_darcy=points[0].apparent_permeability_darcy * 1.5,
            standard_uncertainty_darcy=1.0,  # enormous, so nearly ignored
        )
        result = fit_klinkenberg(corrupted)
        assert result.liquid_permeability_darcy == pytest.approx(TRUE_K_L, rel=1e-3)

        # The same corruption with equal weights pulls k_L badly off.
        equal = list(corrupted)
        equal[0] = KlinkenbergPoint(
            mean_pressure_atm=corrupted[0].mean_pressure_atm,
            apparent_permeability_darcy=corrupted[0].apparent_permeability_darcy,
            standard_uncertainty_darcy=0.001,
        )
        assert fit_klinkenberg(equal).liquid_permeability_darcy != pytest.approx(
            TRUE_K_L, rel=0.05
        )

    def test_partial_uncertainties_fall_back_to_unweighted_with_a_warning(self):
        points = self._weighted_points([0.001, None, 0.001, 0.001])
        result = fit_klinkenberg(points)
        assert result.weighted is False
        assert any("Not every point carried" in w for w in result.warnings)

    def test_the_warning_names_the_points_that_lacked_one(self):
        points = self._weighted_points([0.001, None, 0.001, 0.001])
        warning = next(w for w in fit_klinkenberg(points).warnings if "Not every" in w)
        assert points[1].label in warning

    def test_two_fully_specified_points_do_not_claim_a_missing_uncertainty(self):
        """Two points cannot be weighted, but that is not the same as missing data."""
        points = [
            KlinkenbergPoint(
                mean_pressure_atm=p.mean_pressure_atm,
                apparent_permeability_darcy=p.apparent_permeability_darcy,
                standard_uncertainty_darcy=0.001,
            )
            for p in synthetic_points((1.0, 4.0))
        ]
        result = fit_klinkenberg(points)
        assert result.weighted is False
        assert not any("Not every point carried" in w for w in result.warnings)


class TestSteadyStateAwareness:
    def test_points_from_unsteady_runs_are_flagged(self):
        points = [
            KlinkenbergPoint(
                mean_pressure_atm=p.mean_pressure_atm,
                apparent_permeability_darcy=p.apparent_permeability_darcy,
                label=f"run-{i}",
                steady_state=(i != 1),
            )
            for i, p in enumerate(synthetic_points((1.0, 2.0, 4.0)))
        ]
        result = fit_klinkenberg(points)
        assert any("never reached steady state" in w for w in result.warnings)
        assert any("run-1" in w for w in result.warnings)

    def test_all_steady_points_produce_no_such_warning(self):
        result = fit_klinkenberg(synthetic_points((1.0, 2.0, 4.0)))
        assert not any("steady state" in w for w in result.warnings)


def _mixed_convention_points():
    """Three points where one run supplied P2 and the others measured it."""
    points = list(synthetic_points((1.0, 2.0, 4.0)))
    return [
        KlinkenbergPoint(
            mean_pressure_atm=p.mean_pressure_atm,
            apparent_permeability_darcy=p.apparent_permeability_darcy,
            label=f"run-{index}",
            sample_id="core-001",
            downstream_convention="fixed:1" if index == 1 else "measured",
        )
        for index, p in enumerate(points)
    ]


class TestMixedDownstreamConventions:
    """P2 sets the mean pressure, which is this regression's own x-axis."""

    def test_mixed_conventions_are_refused(self):
        with pytest.raises(ValueError, match="did not all obtain the downstream"):
            fit_klinkenberg(_mixed_convention_points())

    def test_the_refusal_names_the_runs_and_their_conventions(self):
        with pytest.raises(ValueError) as info:
            fit_klinkenberg(_mixed_convention_points())
        message = str(info.value)
        assert "run-0" in message and "run-1" in message
        assert "measured" in message and "fixed" in message

    def test_it_can_be_forced_and_is_then_warned_about(self):
        result = fit_klinkenberg(_mixed_convention_points(), allow_mixed_conditions=True)
        assert any("downstream pressure differently" in w for w in result.warnings)

    def test_one_convention_throughout_is_fine(self):
        points = [
            KlinkenbergPoint(
                mean_pressure_atm=p.mean_pressure_atm,
                apparent_permeability_darcy=p.apparent_permeability_darcy,
                downstream_convention="measured",
            )
            for p in synthetic_points((1.0, 2.0, 4.0))
        ]
        result = fit_klinkenberg(points)
        assert not any("downstream" in w for w in result.warnings)

    def test_an_unknown_convention_alone_does_not_trigger_it(self):
        """A run with no sidecar says nothing about how P2 was obtained."""
        points = list(synthetic_points((1.0, 2.0, 4.0)))
        mixed = [
            KlinkenbergPoint(
                mean_pressure_atm=p.mean_pressure_atm,
                apparent_permeability_darcy=p.apparent_permeability_darcy,
                downstream_convention=None if index == 1 else "measured",
            )
            for index, p in enumerate(points)
        ]
        assert fit_klinkenberg(mixed).point_count == 3


class TestRejectedInputs:
    def test_single_point_is_rejected(self):
        with pytest.raises(ValueError, match="at least 2 points"):
            fit_klinkenberg(synthetic_points((1.0,)))

    def test_identical_pressures_are_rejected(self):
        points = [
            KlinkenbergPoint(mean_pressure_atm=2.0, apparent_permeability_darcy=0.6),
            KlinkenbergPoint(mean_pressure_atm=2.0, apparent_permeability_darcy=0.7),
        ]
        with pytest.raises(ValueError, match="same mean pressure"):
            fit_klinkenberg(points)

    def test_non_positive_pressure_is_rejected_by_the_model(self):
        with pytest.raises(ValueError):
            KlinkenbergPoint(mean_pressure_atm=0.0, apparent_permeability_darcy=0.5)


class TestCsvInput:
    def test_reads_atm_and_millidarcy_by_default(self, tmp_path):
        path = tmp_path / "points.csv"
        path.write_text(
            "mean_pressure,apparent_permeability\n"
            "1.0,600\n"
            "2.0,550\n"
            "4.0,525\n",
            encoding="utf-8",
        )
        points = load_points_from_csv(path)
        assert len(points) == 3
        assert points[0].mean_pressure_atm == pytest.approx(1.0)
        # 600 mD == 0.6 darcy
        assert points[0].apparent_permeability_darcy == pytest.approx(0.6)
        result = fit_klinkenberg(points)
        assert result.liquid_permeability_darcy == pytest.approx(TRUE_K_L, rel=1e-9)

    def test_infers_units_from_column_name_suffixes(self, tmp_path):
        path = tmp_path / "points.csv"
        # Same physical points as above, expressed in kPa and darcy.
        path.write_text(
            "p_mean_kPa,k_g_D\n"
            "101.325,0.6\n"
            "202.650,0.55\n"
            "405.300,0.525\n",
            encoding="utf-8",
        )
        points = load_points_from_csv(path)
        assert points[0].mean_pressure_atm == pytest.approx(1.0, rel=1e-9)
        assert points[0].apparent_permeability_darcy == pytest.approx(0.6)
        result = fit_klinkenberg(points)
        assert result.liquid_permeability_darcy == pytest.approx(TRUE_K_L, rel=1e-6)

    def test_explicit_units_override_the_header(self, tmp_path):
        path = tmp_path / "points.csv"
        path.write_text("mean_pressure,k_g\n1.01325,600\n", encoding="utf-8")
        points = load_points_from_csv(path, pressure_unit="bar", permeability_unit="mD")
        assert points[0].mean_pressure_atm == pytest.approx(1.0, rel=1e-6)

    def test_missing_columns_name_what_was_expected(self, tmp_path):
        path = tmp_path / "wrong.csv"
        path.write_text("pressure_bar,flow\n1,2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mean-pressure column"):
            load_points_from_csv(path)

    def test_unparseable_row_names_the_line(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("mean_pressure,k_g\n1.0,600\n2.0,oops\n", encoding="utf-8")
        with pytest.raises(ValueError, match="line 3"):
            load_points_from_csv(path)

    def test_blank_rows_are_tolerated(self, tmp_path):
        path = tmp_path / "gaps.csv"
        path.write_text(
            "mean_pressure,k_g\n1.0,600\n,\n2.0,550\n4.0,525\n", encoding="utf-8"
        )
        assert len(load_points_from_csv(path)) == 3

    def test_empty_file_is_rejected(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("mean_pressure,k_g\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no data rows"):
            load_points_from_csv(path)
