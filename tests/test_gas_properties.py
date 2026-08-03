"""CoolProp lookups against published reference values.

The reference numbers below come from the NIST/CoolProp equations of state at
standard conditions; the tolerances are loose enough to survive a CoolProp
version bump but tight enough to catch a unit error (which would be off by
1000x, not 1%).
"""

from __future__ import annotations

import math

import pytest

from gasperm import units
from gasperm.config import GasConfig
from gasperm.gas_properties import (
    CoolPropProvider,
    FixedPropertyProvider,
    build_provider,
    compressibility_factor,
    density_kg_m3,
    dynamic_viscosity_pa_s,
    validate_gas_name,
)

CoolProp = pytest.importorskip("CoolProp", reason="CoolProp is required for these tests")

STP_K = 298.15
STP_PA = 101_325.0


class TestReferenceValues:
    def test_nitrogen_viscosity_at_298k_and_1_atm(self):
        """N2 at 25 C, 1 atm: ~17.8 uPa*s (NIST)."""
        viscosity = dynamic_viscosity_pa_s("Nitrogen", STP_K, STP_PA)
        assert viscosity == pytest.approx(1.78e-5, rel=0.01)

    def test_air_viscosity_at_20c(self):
        """Air at 20 C, 1 atm: ~18.2 uPa*s."""
        assert dynamic_viscosity_pa_s("Air", 293.15, STP_PA) == pytest.approx(1.82e-5, rel=0.01)

    def test_carbon_dioxide_viscosity_at_25c(self):
        """CO2 at 25 C, 1 atm: ~14.9 uPa*s."""
        assert dynamic_viscosity_pa_s("CarbonDioxide", STP_K, STP_PA) == pytest.approx(
            1.49e-5, rel=0.01
        )

    def test_nitrogen_density_matches_the_ideal_gas_estimate(self):
        """rho = PM/RT = 101325 * 0.0280134 / (8.31446 * 298.15) = 1.1453 kg/m^3."""
        ideal = STP_PA * 0.028_0134 / (8.314_462_618 * STP_K)
        assert density_kg_m3("Nitrogen", STP_K, STP_PA) == pytest.approx(ideal, rel=0.005)

    def test_nitrogen_is_nearly_ideal_at_ambient(self):
        assert compressibility_factor("Nitrogen", STP_K, STP_PA) == pytest.approx(1.0, abs=0.002)


class TestUnitBoundary:
    def test_provider_returns_centipoise_not_pascal_seconds(self):
        """The SI -> CGS conversion is the single most dangerous step here.

        1.78e-5 Pa*s is 0.0178 cP. A missed conversion would be off by 1000x
        and would silently scale every permeability by the same factor.
        """
        provider = CoolPropProvider("Nitrogen")
        state = provider.state_at(STP_K, STP_PA)
        assert state.viscosity_cp == pytest.approx(0.0178, rel=0.01)
        assert state.viscosity_cp == pytest.approx(
            units.pa_s_to_cp(dynamic_viscosity_pa_s("Nitrogen", STP_K, STP_PA)), rel=1e-12
        )

    def test_state_at_cgs_accepts_celsius_and_atm(self):
        provider = CoolPropProvider("Nitrogen")
        from_si = provider.state_at(STP_K, STP_PA)
        provider_2 = CoolPropProvider("Nitrogen")
        from_cgs = provider_2.state_at_cgs(25.0, 1.0)
        assert from_cgs.viscosity_cp == pytest.approx(from_si.viscosity_cp, rel=1e-9)

    def test_viscosity_rises_with_temperature_for_a_gas(self):
        """Unlike a liquid, gas viscosity increases with temperature."""
        provider = CoolPropProvider("Nitrogen")
        cold = provider.viscosity_cp_at(273.15, STP_PA)
        hot = CoolPropProvider("Nitrogen").viscosity_cp_at(373.15, STP_PA)
        assert hot > cold


class TestValidation:
    def test_known_fluids_pass(self):
        for name in ("Nitrogen", "Air", "CarbonDioxide", "Methane", "Helium"):
            validate_gas_name(name)

    def test_unknown_fluid_is_rejected_with_suggestions(self):
        with pytest.raises(ValueError, match="Nitrogen"):
            validate_gas_name("NotAGas")


class TestCaching:
    def test_repeat_lookups_at_the_same_state_hit_the_cache(self):
        provider = CoolPropProvider("Nitrogen")
        for _ in range(20):
            provider.state_at(STP_K, STP_PA)
        assert provider.lookup_count == 1
        assert provider.cache_hits == 19

    def test_a_meaningful_temperature_change_invalidates_the_cache(self):
        provider = CoolPropProvider("Nitrogen", temperature_tolerance_k=0.05)
        provider.state_at(STP_K, STP_PA)
        provider.state_at(STP_K + 0.01, STP_PA)  # within tolerance
        assert provider.lookup_count == 1
        provider.state_at(STP_K + 5.0, STP_PA)  # well outside
        assert provider.lookup_count == 2

    def test_a_meaningful_pressure_change_invalidates_the_cache(self):
        provider = CoolPropProvider("Nitrogen", pressure_tolerance_frac=0.002)
        provider.state_at(STP_K, STP_PA)
        provider.state_at(STP_K, STP_PA * 1.0005)
        assert provider.lookup_count == 1
        provider.state_at(STP_K, STP_PA * 2.0)
        assert provider.lookup_count == 2

    def test_cache_tolerance_does_not_change_the_answer_materially(self):
        """Whatever the cache serves must be within measurement noise."""
        exact = CoolPropProvider("Nitrogen").viscosity_cp_at(STP_K + 0.05, STP_PA)
        cached_provider = CoolPropProvider("Nitrogen")
        cached_provider.state_at(STP_K, STP_PA)
        served = cached_provider.viscosity_cp_at(STP_K + 0.05, STP_PA)
        assert served == pytest.approx(exact, rel=2e-4)


class TestViscosityTemperatureSensitivity:
    """d(ln mu)/d(ln T): the only route by which temperature enters the budget."""

    def test_nitrogen_exponent_is_near_the_kinetic_theory_value(self):
        # Gas viscosity rises roughly as T^0.7 near ambient (Sutherland).
        exponent = CoolPropProvider("Nitrogen").viscosity_temperature_exponent(
            STP_K, STP_PA
        )
        assert exponent == pytest.approx(0.72, abs=0.08)

    def test_the_exponent_is_positive_for_gases(self):
        """Unlike liquids, gases get more viscous when heated."""
        for gas in ("Nitrogen", "Air", "CarbonDioxide", "Methane"):
            assert CoolPropProvider(gas).viscosity_temperature_exponent(STP_K, STP_PA) > 0.0

    def test_it_matches_a_coarse_finite_difference(self):
        provider = CoolPropProvider("Nitrogen")
        cold = dynamic_viscosity_pa_s("Nitrogen", 290.0, STP_PA)
        hot = dynamic_viscosity_pa_s("Nitrogen", 310.0, STP_PA)
        coarse = (math.log(hot) - math.log(cold)) / (math.log(310.0) - math.log(290.0))
        assert provider.viscosity_temperature_exponent(300.0, STP_PA) == pytest.approx(
            coarse, rel=0.02
        )

    def test_probing_the_derivative_does_not_disturb_the_cache(self):
        provider = CoolPropProvider("Nitrogen")
        provider.state_at(STP_K, STP_PA)
        assert provider.lookup_count == 1
        provider.viscosity_temperature_exponent(STP_K, STP_PA)
        provider.state_at(STP_K, STP_PA)
        # The acquisition loop's cached state must still be served.
        assert provider.cache_hits == 1

    def test_a_fixed_provider_has_no_temperature_sensitivity(self):
        provider = FixedPropertyProvider("Nitrogen", 0.0178)
        assert provider.viscosity_temperature_exponent(STP_K, STP_PA) == 0.0


class TestUncertaintyPassthrough:
    def test_the_configured_viscosity_uncertainty_reaches_the_state(self):
        config = GasConfig(viscosity_relative_uncertainty=0.02)
        state = build_provider(config).state_at(STP_K, STP_PA)
        assert state.relative_viscosity_uncertainty == 0.02


class TestFixedProvider:
    def test_returns_the_configured_constant(self):
        provider = FixedPropertyProvider("Nitrogen", 0.0178, reason="no CoolProp here")
        assert provider.viscosity_cp_at(200.0, 1e5) == 0.0178
        assert provider.viscosity_cp_at(400.0, 5e6) == 0.0178

    def test_records_that_the_live_lookup_was_bypassed(self):
        provider = FixedPropertyProvider("Nitrogen", 0.0178)
        assert provider.state_at(STP_K, STP_PA).source == "fixed"

    def test_reports_the_state_it_was_asked_about(self):
        provider = FixedPropertyProvider("Nitrogen", 0.0178)
        state = provider.state_at(310.0, 200_000.0)
        assert state.temperature_k == 310.0
        assert state.pressure_pa == 200_000.0

    def test_non_positive_viscosity_is_rejected(self):
        with pytest.raises(ValueError):
            FixedPropertyProvider("Nitrogen", 0.0)


class TestBuildProvider:
    def test_coolprop_is_the_default(self):
        assert isinstance(build_provider(GasConfig()), CoolPropProvider)

    def test_fixed_source_builds_a_fixed_provider(self):
        config = GasConfig(
            name="Nitrogen",
            properties_source="fixed",
            fixed_viscosity_cp=0.02,
            fixed_reason="offline rig",
        )
        provider = build_provider(config)
        assert isinstance(provider, FixedPropertyProvider)
        assert provider.viscosity_cp == 0.02

    def test_fixed_source_without_a_value_is_rejected_at_config_time(self):
        with pytest.raises(ValueError, match="fixed_viscosity_cp"):
            GasConfig(properties_source="fixed")
