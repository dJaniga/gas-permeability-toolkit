"""Unit conversions checked against hand-computed factors, not just round-trips.

A round-trip test passes even if a factor is wrong in both directions, which is
precisely the failure mode this package cannot tolerate -- so every conversion
is also pinned to an independently-known number.
"""

from __future__ import annotations


import pytest

from gasperm import units

# Hand-checked: pascal per unit, from the SI definitions.
PASCAL_PER_UNIT = {
    "Pa": 1.0,
    "kPa": 1_000.0,
    "MPa": 1_000_000.0,
    "bar": 100_000.0,
    "psi": 6_894.757_293_168_361,
    "atm": 101_325.0,
}


class TestPressure:
    def test_every_supported_unit_has_a_hand_checked_factor(self):
        assert set(units.SUPPORTED_PRESSURE_UNITS) == set(PASCAL_PER_UNIT)

    @pytest.mark.parametrize("unit,pascal", sorted(PASCAL_PER_UNIT.items()))
    def test_to_pa_matches_the_definition(self, unit: str, pascal: float):
        assert units.to_pa(1.0, unit) == pytest.approx(pascal, rel=1e-12)

    @pytest.mark.parametrize("unit", sorted(PASCAL_PER_UNIT))
    def test_to_atm_matches_the_definition(self, unit: str):
        expected = PASCAL_PER_UNIT[unit] / 101_325.0
        assert units.to_atm(1.0, unit) == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("unit", sorted(PASCAL_PER_UNIT))
    def test_round_trip_through_atm(self, unit: str):
        value = 37.5
        assert units.from_atm(units.to_atm(value, unit), unit) == pytest.approx(value, rel=1e-12)

    def test_known_landmarks(self):
        # 1 standard atmosphere in the usual engineering units.
        assert units.to_atm(101.325, "kPa") == pytest.approx(1.0, rel=1e-12)
        assert units.to_atm(1.013_25, "bar") == pytest.approx(1.0, rel=1e-12)
        assert units.to_atm(14.695_948_775_513_2, "psi") == pytest.approx(1.0, rel=1e-9)
        # 1 bar is very slightly under 1 atm.
        assert units.bar_to_atm(1.0) == pytest.approx(0.986_923_266_7, rel=1e-9)
        assert units.mpa_to_atm(1.0) == pytest.approx(9.869_232_667, rel=1e-9)
        assert units.psi_to_atm(1.0) == pytest.approx(0.068_045_963_9, rel=1e-8)
        assert units.kpa_to_atm(1.0) == pytest.approx(0.009_869_232_667, rel=1e-9)
        assert units.pa_to_atm(101_325.0) == pytest.approx(1.0, rel=1e-12)

    def test_named_helpers_agree_with_the_generic_function(self):
        for value, unit, helper in (
            (250.0, "kPa", units.kpa_to_atm),
            (3.0, "MPa", units.mpa_to_atm),
            (2.5, "bar", units.bar_to_atm),
            (100.0, "psi", units.psi_to_atm),
            (5000.0, "Pa", units.pa_to_atm),
        ):
            assert helper(value) == pytest.approx(units.to_atm(value, unit), rel=1e-15)

    def test_unit_names_are_case_insensitive(self):
        assert units.to_atm(1.0, "KPA") == units.to_atm(1.0, "kPa")
        assert units.normalize_pressure_unit(" mpa ") == "MPa"

    def test_unknown_unit_is_rejected_by_name(self):
        with pytest.raises(ValueError, match="torr"):
            units.to_atm(1.0, "torr")


class TestFlow:
    def test_sccm_is_per_minute(self):
        assert units.sccm_to_cm3_s(60.0) == pytest.approx(1.0)
        assert units.slpm_to_cm3_s(1.0) == pytest.approx(1000.0 / 60.0)

    @pytest.mark.parametrize(
        "unit,expected_cm3_s",
        [
            ("cm3/s", 1.0),
            ("cm3/min", 1.0 / 60.0),
            ("sccm", 1.0 / 60.0),
            ("mL/s", 1.0),
            ("mL/min", 1.0 / 60.0),
            ("L/s", 1000.0),
            ("L/min", 1000.0 / 60.0),
            ("slpm", 1000.0 / 60.0),
            ("m3/h", 1_000_000.0 / 3600.0),
        ],
    )
    def test_flow_factors(self, unit: str, expected_cm3_s: float):
        assert units.flow_to_cm3_s(1.0, unit) == pytest.approx(expected_cm3_s, rel=1e-12)

    @pytest.mark.parametrize("unit", sorted(units.SUPPORTED_FLOW_UNITS))
    def test_flow_round_trip(self, unit: str):
        assert units.flow_from_cm3_s(units.flow_to_cm3_s(4.25, unit), unit) == pytest.approx(4.25)

    def test_unknown_flow_unit_is_rejected(self):
        with pytest.raises(ValueError, match="gallons"):
            units.flow_to_cm3_s(1.0, "gallons/s")


class TestLengthAndArea:
    @pytest.mark.parametrize(
        "unit,expected_cm", [("cm", 1.0), ("mm", 0.1), ("m", 100.0), ("in", 2.54), ("ft", 30.48)]
    )
    def test_length_factors(self, unit: str, expected_cm: float):
        assert units.length_to_cm(1.0, unit) == pytest.approx(expected_cm, rel=1e-12)

    def test_circle_area_of_a_one_inch_plug(self):
        # A 1" plug: A = pi * (2.54/2)^2 = 5.0670747 cm^2.
        assert units.circle_area_cm2(2.54) == pytest.approx(5.067_074_79, rel=1e-8)

    def test_non_positive_diameter_is_rejected(self):
        with pytest.raises(ValueError):
            units.circle_area_cm2(0.0)


class TestVolume:
    """Pulse-decay reservoir volumes. Internal unit is cm^3."""

    @pytest.mark.parametrize(
        "unit,expected_cm3",
        [("cm3", 1.0), ("mL", 1.0), ("L", 1000.0), ("m3", 1.0e6), ("in3", 16.387_064)],
    )
    def test_volume_factors(self, unit: str, expected_cm3: float):
        assert units.volume_to_cm3(1.0, unit) == pytest.approx(expected_cm3, rel=1e-12)

    def test_a_millilitre_is_a_cubic_centimetre(self):
        """Exact since the 1964 redefinition of the litre, not an approximation."""
        assert units.volume_to_cm3(1.0, "mL") == units.volume_to_cm3(1.0, "cm3")

    def test_a_cubic_inch_is_two_point_five_four_cubed(self):
        assert units.volume_to_cm3(1.0, "in3") == pytest.approx(2.54**3, rel=1e-15)

    @pytest.mark.parametrize("unit", sorted(units.SUPPORTED_VOLUME_UNITS))
    def test_round_trip(self, unit: str):
        value = 76.5
        assert units.volume_from_cm3(
            units.volume_to_cm3(value, unit), unit
        ) == pytest.approx(value, rel=1e-12)

    @pytest.mark.parametrize("alias", ["cc", "CM3", "cm^3", "litre", "in^3"])
    def test_aliases_are_accepted(self, alias: str):
        assert units.volume_to_cm3(1.0, alias) > 0.0

    def test_unknown_unit_is_rejected(self):
        with pytest.raises(ValueError, match="gallon"):
            units.volume_to_cm3(1.0, "gallon")

    def test_the_rig_vessels_convert(self):
        """400 cm3 upstream and 75 cm3 downstream, however they are written."""
        assert units.volume_to_cm3(0.4, "L") == pytest.approx(400.0)
        assert units.volume_to_cm3(75.0, "mL") == pytest.approx(75.0)


class TestCompressibility:
    """Reciprocal pressure. Internal unit is 1/atm."""

    def test_per_pa_to_per_atm_is_the_atm_in_pa_factor(self):
        # c = 1e-5 /Pa is 1e-5 * 101325 = 1.01325 /atm.
        assert units.per_pa_to_per_atm(1e-5) == pytest.approx(1.013_25, rel=1e-12)

    def test_per_atm_is_the_identity(self):
        assert units.compressibility_to_per_atm(0.1, "1/atm") == pytest.approx(0.1)

    @pytest.mark.parametrize(
        "unit,expected_per_atm",
        [
            ("1/Pa", 101_325.0),
            ("1/kPa", 101.325),
            ("1/MPa", 0.101_325),
            ("1/bar", 1.013_25),
            ("1/atm", 1.0),
        ],
    )
    def test_factors(self, unit: str, expected_per_atm: float):
        assert units.compressibility_to_per_atm(1.0, unit) == pytest.approx(
            expected_per_atm, rel=1e-12
        )

    @pytest.mark.parametrize("unit", sorted(units.SUPPORTED_COMPRESSIBILITY_UNITS))
    def test_round_trip(self, unit: str):
        value = 0.0975
        assert units.compressibility_from_per_atm(
            units.compressibility_to_per_atm(value, unit), unit
        ) == pytest.approx(value, rel=1e-12)

    @pytest.mark.parametrize("spelling", ["1/kPa", "kPa^-1", "kPa-1", "per_kPa", "1/KPA"])
    def test_spellings_agree(self, spelling: str):
        assert units.compressibility_to_per_atm(1.0, spelling) == pytest.approx(101.325)

    def test_a_bare_pressure_unit_is_rejected(self):
        """'atm' and '1/atm' differ by 1e5 at a typical pore pressure."""
        with pytest.raises(ValueError, match="reciprocal pressure"):
            units.compressibility_to_per_atm(1.0, "atm")

    def test_unknown_unit_is_rejected(self):
        with pytest.raises(ValueError, match="torr"):
            units.compressibility_to_per_atm(1.0, "1/torr")

    def test_the_family_tracks_the_pressure_units(self):
        """No second constant table: adding a pressure unit extends this one."""
        assert units.SUPPORTED_COMPRESSIBILITY_UNITS == {
            f"1/{unit}" for unit in units.SUPPORTED_PRESSURE_UNITS
        }

    def test_ideal_gas_compressibility_at_ten_atm(self):
        """c = 1/P for an ideal gas: 10 atm -> 0.1 /atm, however written."""
        assert units.compressibility_to_per_atm(
            1.0 / units.to_pa(10.0, "atm"), "1/Pa"
        ) == pytest.approx(0.1, rel=1e-12)


class TestViscosity:
    def test_pa_s_to_cp(self):
        # Water at 20 C is ~1.002e-3 Pa*s, i.e. ~1.002 cP -- the definition of
        # the centipoise as a practical unit.
        assert units.pa_s_to_cp(1.002e-3) == pytest.approx(1.002, rel=1e-12)
        assert units.cp_to_pa_s(1.0) == pytest.approx(1e-3, rel=1e-12)

    def test_round_trip(self):
        assert units.cp_to_pa_s(units.pa_s_to_cp(1.78e-5)) == pytest.approx(1.78e-5, rel=1e-15)


class TestPermeability:
    def test_millidarcy(self):
        assert units.darcy_to(1.0, "mD") == pytest.approx(1000.0)
        assert units.darcy_from(1000.0, "mD") == pytest.approx(1.0)

    def test_darcy_in_si(self):
        # 1 darcy = 9.869233e-13 m^2 = 0.9869233 um^2.
        assert units.darcy_to(1.0, "m2") == pytest.approx(9.869_233e-13, rel=1e-9)
        assert units.darcy_to(1.0, "um2") == pytest.approx(0.986_923_3, rel=1e-7)

    @pytest.mark.parametrize("unit", sorted(units.SUPPORTED_PERMEABILITY_UNITS))
    def test_round_trip(self, unit: str):
        assert units.darcy_from(units.darcy_to(0.25, unit), unit) == pytest.approx(0.25)

    def test_unknown_unit_is_rejected(self):
        with pytest.raises(ValueError, match="furlongs"):
            units.darcy_to(1.0, "furlongs")


class TestTemperature:
    def test_celsius_kelvin(self):
        assert units.celsius_to_kelvin(0.0) == pytest.approx(273.15)
        assert units.celsius_to_kelvin(25.0) == pytest.approx(298.15)
        assert units.kelvin_to_celsius(298.15) == pytest.approx(25.0)

    def test_fahrenheit(self):
        assert units.fahrenheit_to_celsius(32.0) == pytest.approx(0.0)
        assert units.fahrenheit_to_celsius(212.0) == pytest.approx(100.0)
        assert units.fahrenheit_to_celsius(-40.0) == pytest.approx(-40.0)

    @pytest.mark.parametrize("unit", ["C", "K", "F"])
    def test_round_trip_through_kelvin(self, unit: str):
        value = 21.5
        kelvin = units.temperature_to_kelvin(value, unit)
        assert units.temperature_from_kelvin(kelvin, unit) == pytest.approx(value, rel=1e-12)

    def test_probe_unit_conversion(self):
        assert units.temperature_to_kelvin(68.0, "F") == pytest.approx(293.15)

    def test_unknown_unit_is_rejected(self):
        with pytest.raises(ValueError, match="rankine"):
            units.temperature_to_kelvin(1.0, "rankine")
