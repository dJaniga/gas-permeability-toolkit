"""GUM uncertainty propagation, checked against hand-worked cases.

The sensitivity coefficients are exact and independently derivable, so they are
pinned to hand-calculated values rather than to whatever the code produces.
"""

from __future__ import annotations

import math

import pytest

from gasperm.config import GaspermConfig
from gasperm.config.common import UncertaintySpec
from gasperm.models import SampleGeometry
from gasperm.uncertainty import (
    MeasurementPoint,
    build_budget,
    coverage_factor,
    pressure_sensitivities,
)


def geometry(**overrides) -> SampleGeometry:
    base = dict(
        sample_id="core-001",
        length_cm=5.0,
        diameter_cm=2.54,
        length_uncertainty_cm=0.01,
        diameter_uncertainty_cm=0.01,
    )
    base.update(overrides)
    return SampleGeometry(**base)


def point(**overrides) -> MeasurementPoint:
    base = dict(
        permeability_darcy=0.005,
        inlet_pressure_atm=3.0,
        outlet_pressure_atm=1.0,
        flow_cm3_s=1.6667,
        reference_pressure_atm=1.0,
        viscosity_cp=0.0178,
        temperature_c=22.0,
    )
    base.update(overrides)
    return MeasurementPoint(**base)


def config_with(**run_overrides) -> GaspermConfig:
    config = GaspermConfig()
    for key, value in run_overrides.items():
        setattr(config.run, key, value)
    return config


class TestTypeBSpecs:
    """GUM 4.3: a specification limit divided by its distribution factor."""

    def test_rectangular_divides_by_root_three(self):
        spec = UncertaintySpec(kind="absolute", value=3.0, distribution="rectangular")
        assert spec.standard_uncertainty(0.0, 0.0) == pytest.approx(3.0 / math.sqrt(3.0))

    def test_triangular_divides_by_root_six(self):
        spec = UncertaintySpec(kind="absolute", value=3.0, distribution="triangular")
        assert spec.standard_uncertainty(0.0, 0.0) == pytest.approx(3.0 / math.sqrt(6.0))

    def test_normal_divides_by_the_stated_coverage_factor(self):
        spec = UncertaintySpec(
            kind="absolute", value=3.0, distribution="normal", coverage_factor=2.0
        )
        assert spec.standard_uncertainty(0.0, 0.0) == pytest.approx(1.5)

    def test_percent_of_full_scale_ignores_the_reading(self):
        spec = UncertaintySpec(kind="percent_full_scale", value=0.5)
        assert spec.half_width(10.0, 1000.0) == pytest.approx(5.0)
        assert spec.half_width(900.0, 1000.0) == pytest.approx(5.0)

    def test_percent_of_reading_scales_with_the_reading(self):
        spec = UncertaintySpec(kind="percent_reading", value=1.0)
        assert spec.half_width(200.0, 500.0) == pytest.approx(2.0)

    def test_none_contributes_nothing(self):
        spec = UncertaintySpec(kind="none", value=99.0)
        assert spec.standard_uncertainty(1.0, 1.0) == 0.0

    def test_missing_degrees_of_freedom_means_infinite(self):
        assert UncertaintySpec().dof == math.inf
        assert UncertaintySpec(degrees_of_freedom=9).dof == 9.0


class TestSensitivities:
    def test_hand_calculated_pressure_coefficients(self):
        # P1 = 3, P2 = 1 -> P1^2 - P2^2 = 8
        # c_P1 = -2*9/8 = -2.25 ; c_P2 = +2*1/8 = +0.25
        c_p1, c_p2 = pressure_sensitivities(3.0, 1.0)
        assert c_p1 == pytest.approx(-2.25)
        assert c_p2 == pytest.approx(0.25)

    def test_coefficients_diverge_as_the_differential_closes(self):
        wide = pressure_sensitivities(5.0, 1.0)[0]
        narrow = pressure_sensitivities(1.05, 1.0)[0]
        assert abs(narrow) > 10 * abs(wide)

    def test_equal_pressures_are_rejected(self):
        with pytest.raises(ValueError, match="undefined"):
            pressure_sensitivities(2.0, 2.0)

    def test_coefficients_match_a_numerical_derivative(self):
        """Cross-check the analytic exponents against finite differences."""
        from gasperm.permeability import compute_gas_permeability

        args = dict(
            flow_rate_cm3_s=1.5, reference_pressure_atm=1.0, viscosity_cp=0.018,
            length_cm=5.0, area_cm2=5.067, outlet_pressure_atm=1.0,
        )
        p1 = 3.0
        base = compute_gas_permeability(inlet_pressure_atm=p1, **args)
        step = 1e-6
        numerical = (
            compute_gas_permeability(inlet_pressure_atm=p1 + step, **args) - base
        ) / step
        relative = numerical * p1 / base
        assert relative == pytest.approx(pressure_sensitivities(p1, 1.0)[0], rel=1e-5)


class TestCoverageFactor:
    def test_large_dof_approaches_the_normal_quantile(self):
        assert coverage_factor(math.inf, 0.95) == pytest.approx(1.96, abs=0.005)

    def test_small_dof_inflates_the_factor(self):
        assert coverage_factor(4.0, 0.95) == pytest.approx(2.776, abs=0.01)

    def test_higher_confidence_gives_a_larger_factor(self):
        assert coverage_factor(math.inf, 0.99) > coverage_factor(math.inf, 0.95)

    def test_invalid_probability_is_rejected(self):
        with pytest.raises(ValueError, match="coverage_probability"):
            coverage_factor(10.0, 1.5)


class TestBudget:
    def test_relative_sensitivities_are_the_exponents(self):
        config = config_with()
        budget = build_budget(point(), geometry(), config.hardware, config.run)
        by_symbol = {c.symbol: c for c in budget.components}
        assert by_symbol["Q"].relative_sensitivity == pytest.approx(1.0)
        assert by_symbol["mu"].relative_sensitivity == pytest.approx(1.0)
        assert by_symbol["L"].relative_sensitivity == pytest.approx(1.0)
        assert by_symbol["d"].relative_sensitivity == pytest.approx(-2.0)
        assert by_symbol["P1"].relative_sensitivity == pytest.approx(-2.25)
        assert by_symbol["P2"].relative_sensitivity == pytest.approx(0.25)

    def test_combined_is_the_quadrature_sum_of_contributions(self):
        config = config_with()
        budget = build_budget(point(), geometry(), config.hardware, config.run)
        expected = math.sqrt(sum(c.variance_share for c in budget.components))
        assert budget.relative_combined_standard_uncertainty == pytest.approx(expected)
        assert budget.combined_standard_uncertainty_darcy == pytest.approx(
            expected * 0.005
        )

    def test_expanded_is_the_coverage_factor_times_the_combined(self):
        config = config_with()
        budget = build_budget(point(), geometry(), config.hardware, config.run)
        assert budget.expanded_uncertainty_darcy == pytest.approx(
            budget.coverage_factor * budget.combined_standard_uncertainty_darcy
        )
        low, high = budget.interval_darcy
        assert low < budget.value_darcy < high

    def test_diameter_enters_doubled(self):
        """u(A)/A = 2 u(d)/d, so the diameter term is twice the raw caliper term."""
        config = config_with()
        budget = build_budget(point(), geometry(diameter_uncertainty_cm=0.02), config.hardware, config.run)
        diameter = next(c for c in budget.components if c.symbol == "d")
        assert diameter.relative_standard_uncertainty == pytest.approx(0.02 / 2.54)
        assert diameter.relative_contribution == pytest.approx(2.0 * 0.02 / 2.54)

    def test_a_narrow_differential_inflates_the_budget(self):
        """The quantitative version of 'do not measure at a small dP'."""
        config = config_with()
        wide = build_budget(
            point(inlet_pressure_atm=5.0), geometry(), config.hardware, config.run
        )
        narrow = build_budget(
            point(inlet_pressure_atm=1.05), geometry(), config.hardware, config.run
        )
        assert (
            narrow.relative_combined_standard_uncertainty
            > 5 * wide.relative_combined_standard_uncertainty
        )

    def test_type_a_is_included_and_labelled(self):
        config = config_with()
        budget = build_budget(
            point(), geometry(), config.hardware, config.run,
            type_a_relative=0.004, type_a_dof=19.0,
        )
        type_a = [c for c in budget.components if c.evaluation_type == "A"]
        assert len(type_a) == 1
        assert type_a[0].relative_standard_uncertainty == pytest.approx(0.004)
        assert type_a[0].degrees_of_freedom == 19.0

    def test_type_a_can_be_switched_off(self):
        config = config_with()
        config.run.uncertainty.include_type_a = False
        budget = build_budget(
            point(), geometry(), config.hardware, config.run, type_a_relative=0.004
        )
        assert not any(c.evaluation_type == "A" for c in budget.components)

    def test_welch_satterthwaite_reduces_dof_when_type_a_dominates(self):
        config = config_with()
        budget = build_budget(
            point(), geometry(), config.hardware, config.run,
            type_a_relative=0.5, type_a_dof=4.0,
        )
        # A dominant term with 4 dof drags the effective dof towards 4.
        assert budget.effective_degrees_of_freedom < 10.0
        assert budget.coverage_factor > 2.2

    def test_all_type_b_gives_infinite_dof_and_k_near_two(self):
        config = config_with()
        budget = build_budget(point(), geometry(), config.hardware, config.run)
        assert budget.effective_degrees_of_freedom == math.inf
        assert budget.coverage_factor == pytest.approx(1.96, abs=0.01)

    def test_a_negligible_type_a_term_reports_infinite_dof_not_a_huge_number(self):
        """v_eff of 1e14 is infinity with a misleading number of digits."""
        config = config_with()
        budget = build_budget(
            point(), geometry(), config.hardware, config.run,
            type_a_relative=1e-7, type_a_dof=50.0,
        )
        assert budget.effective_degrees_of_freedom == math.inf

    def test_a_fixed_coverage_factor_overrides_the_derivation(self):
        config = config_with()
        config.run.uncertainty.fixed_coverage_factor = 3.0
        budget = build_budget(point(), geometry(), config.hardware, config.run)
        assert budget.coverage_factor == 3.0

    def test_temperature_enters_only_through_viscosity(self):
        config = config_with()
        without = build_budget(point(), geometry(), config.hardware, config.run)
        with_temperature = build_budget(
            point(), geometry(), config.hardware, config.run,
            viscosity_temperature_exponent=0.7,
        )
        assert not any(c.symbol == "T" for c in without.components)
        temperature = next(c for c in with_temperature.components if c.symbol == "T")
        assert temperature.relative_sensitivity == pytest.approx(0.7)
        assert (
            with_temperature.relative_combined_standard_uncertainty
            > without.relative_combined_standard_uncertainty
        )

    def test_gauge_transducers_use_the_gauge_reading_for_percent_of_reading(self):
        config = config_with()
        config.hardware.pressure_calibration.inlet.reading_type = "gauge"
        config.hardware.pressure_calibration.inlet.uncertainty = UncertaintySpec(
            kind="percent_reading", value=1.0
        )
        budget = build_budget(point(), geometry(), config.hardware, config.run)
        inlet = next(c for c in budget.components if c.symbol == "P1")
        # The spec applies to what the transducer reads, i.e. the gauge value
        # 3 - 1 = 2 atm. 1% of that is a half-width of 0.02 atm; rectangular so
        # u = 0.02/sqrt(3). Making it absolute adds the ambient reference, so
        # that value's own uncertainty joins in quadrature. Then relative to the
        # 3 atm ABSOLUTE pressure, plus the DAQ term.
        u_transducer = 0.02 / math.sqrt(3.0)
        u_ambient = (0.1 / math.sqrt(3.0)) / 101.325  # 0.1 kPa rectangular, in atm
        expected = math.hypot(math.hypot(u_transducer, u_ambient) / 3.0, 0.0002)
        assert inlet.relative_standard_uncertainty == pytest.approx(expected, rel=1e-9)

    def test_an_absolute_transducer_does_not_pick_up_the_ambient_uncertainty(self):
        """Nothing is added to an absolute reading, so nothing propagates in."""
        config = config_with()
        config.hardware.pressure_calibration.inlet.uncertainty = UncertaintySpec(
            kind="percent_reading", value=1.0
        )
        strict = build_budget(point(), geometry(), config.hardware, config.run)
        config.run.atmospheric_pressure_uncertainty = UncertaintySpec(
            kind="absolute", value=50.0  # absurdly bad barometer
        )
        loose = build_budget(point(), geometry(), config.hardware, config.run)
        inlet_strict = next(c for c in strict.components if c.symbol == "P1")
        inlet_loose = next(c for c in loose.components if c.symbol == "P1")
        assert inlet_loose.relative_standard_uncertainty == pytest.approx(
            inlet_strict.relative_standard_uncertainty
        )

    def test_an_absolute_transducer_uses_the_absolute_reading(self):
        config = config_with()
        config.hardware.pressure_calibration.inlet.uncertainty = UncertaintySpec(
            kind="percent_reading", value=1.0
        )
        budget = build_budget(point(), geometry(), config.hardware, config.run)
        inlet = next(c for c in budget.components if c.symbol == "P1")
        # 1% of 3 atm, rectangular, relative to 3 atm -> just 1%/sqrt(3).
        expected = math.hypot(0.01 / math.sqrt(3.0), 0.0002)
        assert inlet.relative_standard_uncertainty == pytest.approx(expected, rel=1e-6)

    def test_dominant_components_are_ranked(self):
        config = config_with()
        budget = build_budget(point(), geometry(diameter_uncertainty_cm=0.2), config.hardware, config.run)
        assert budget.dominant_components(1)[0].symbol == "d"


class TestCorrelation:
    def test_positive_correlation_reduces_the_combined_uncertainty(self):
        """P1 and P2 enter with opposite signs, so shared error partly cancels."""
        independent = config_with()
        correlated = config_with()
        correlated.hardware.pressure_calibration.correlation = 0.9

        a = build_budget(point(), geometry(), independent.hardware, independent.run)
        b = build_budget(point(), geometry(), correlated.hardware, correlated.run)
        assert b.correlation_relative_variance < 0.0
        assert (
            b.relative_combined_standard_uncertainty
            < a.relative_combined_standard_uncertainty
        )

    def test_zero_correlation_adds_no_term(self):
        config = config_with()
        budget = build_budget(point(), geometry(), config.hardware, config.run)
        assert budget.correlation_relative_variance == 0.0

    def test_correlation_always_applies_because_p2_is_always_a_transducer(self):
        """Both pressures come from the DAQ, so the covariance term is never skipped."""
        config = config_with()
        config.hardware.pressure_calibration.correlation = 0.5
        budget = build_budget(point(), geometry(), config.hardware, config.run)
        assert budget.correlation_relative_variance != 0.0
        assert any("covariance term" in n for n in budget.notes)

    def test_out_of_range_correlation_is_rejected_by_the_config(self):
        config = GaspermConfig()
        with pytest.raises(ValueError):
            config.hardware.pressure_calibration.correlation = 1.5


class TestFlowReferenceHandling:
    def test_a_standard_basis_meter_contributes_no_reference_pressure_term(self):
        config = config_with()
        budget = build_budget(point(), geometry(), config.hardware, config.run)
        assert not any(c.symbol == "P_ref" for c in budget.components)
        assert any("defined standard state" in n for n in budget.notes)

    def test_an_actual_basis_meter_adds_a_reference_pressure_term(self):
        config = config_with()
        config.hardware.flowmeter.reading_basis = "actual"
        budget = build_budget(point(), geometry(), config.hardware, config.run)
        reference = next(c for c in budget.components if c.symbol == "P_ref")
        assert reference.relative_sensitivity == pytest.approx(1.0)
        assert any("errs high" in n for n in budget.notes)
