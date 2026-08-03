"""Physics correctness for the compressible-gas Darcy equation.

Reference values are hand-calculated from the equation rather than captured
from a previous run, so a sign or factor error cannot be blessed by the test.
"""

from __future__ import annotations

import math

import pytest

from gasperm.permeability import (
    PermeabilityInputError,
    compute_gas_permeability,
    mean_pressure,
)

#: Unit-cell inputs: every quantity equal to 1 in CGS-Darcy units, so the
#: expected result falls out of the equation by hand.
UNIT_CELL = dict(
    flow_rate_cm3_s=1.0,
    reference_pressure_atm=1.0,
    viscosity_cp=1.0,
    length_cm=1.0,
    area_cm2=1.0,
    inlet_pressure_atm=2.0,
    outlet_pressure_atm=1.0,
)


class TestKnownValues:
    def test_unit_cell(self):
        # k = 2*1*1*1*1 / (1 * (2^2 - 1^2)) = 2/3 darcy.
        assert compute_gas_permeability(**UNIT_CELL) == pytest.approx(2.0 / 3.0, rel=1e-12)

    def test_realistic_core_plug(self):
        # 1" x 2" plug, nitrogen at 0.0178 cP, 100 sccm through a 2 atm
        # differential, flow referenced to 1 atm.
        #   A = pi*(2.54/2)^2       = 5.06707479 cm^2
        #   L = 5.08 cm
        #   Q = 100/60              = 1.6666667 cm^3/s
        #   P1^2 - P2^2 = 9 - 1     = 8 atm^2
        #   k = 2*1.6666667*1*0.0178*5.08 / (5.06707479 * 8)
        expected = (2.0 * (100.0 / 60.0) * 1.0 * 0.0178 * 5.08) / (
            math.pi * (2.54 / 2.0) ** 2 * (3.0**2 - 1.0**2)
        )
        result = compute_gas_permeability(
            flow_rate_cm3_s=100.0 / 60.0,
            reference_pressure_atm=1.0,
            viscosity_cp=0.0178,
            length_cm=5.08,
            area_cm2=math.pi * (2.54 / 2.0) ** 2,
            inlet_pressure_atm=3.0,
            outlet_pressure_atm=1.0,
        )
        assert result == pytest.approx(expected, rel=1e-12)
        # 0.30141333 / 40.5365983 = 0.00743565 darcy, i.e. ~7.44 mD.
        assert result == pytest.approx(0.007_435_65, rel=1e-5)

    def test_approaches_the_incompressible_form_at_small_differential(self):
        """With P1 -> P2, the compressible form must collapse to k = Q*mu*L/(A*dP).

        This is the strongest available check that the ``2 * Q * P_ref`` factor
        is right: get it wrong by any constant and the two forms diverge.
        """
        p2 = 1.0
        p1 = 1.001
        compressible = compute_gas_permeability(
            flow_rate_cm3_s=1.0,
            reference_pressure_atm=p2,
            viscosity_cp=1.0,
            length_cm=1.0,
            area_cm2=1.0,
            inlet_pressure_atm=p1,
            outlet_pressure_atm=p2,
        )
        incompressible = 1.0 * 1.0 * 1.0 / (1.0 * (p1 - p2))
        assert compressible == pytest.approx(incompressible, rel=1e-3)


class TestScaling:
    def test_linear_in_flow_rate(self):
        base = compute_gas_permeability(**UNIT_CELL)
        doubled = compute_gas_permeability(**{**UNIT_CELL, "flow_rate_cm3_s": 2.0})
        assert doubled == pytest.approx(2.0 * base)

    def test_linear_in_viscosity(self):
        base = compute_gas_permeability(**UNIT_CELL)
        doubled = compute_gas_permeability(**{**UNIT_CELL, "viscosity_cp": 2.0})
        assert doubled == pytest.approx(2.0 * base)

    def test_linear_in_length(self):
        base = compute_gas_permeability(**UNIT_CELL)
        doubled = compute_gas_permeability(**{**UNIT_CELL, "length_cm": 2.0})
        assert doubled == pytest.approx(2.0 * base)

    def test_inverse_in_area(self):
        base = compute_gas_permeability(**UNIT_CELL)
        doubled = compute_gas_permeability(**{**UNIT_CELL, "area_cm2": 2.0})
        assert doubled == pytest.approx(base / 2.0)

    def test_reference_pressure_and_flow_trade_off_exactly(self):
        """Only the product ``Q_ref * P_ref`` matters.

        This is why the flowmeter's reading basis must pair the reported volume
        with the right pressure: halving one and doubling the other is a no-op,
        but getting only one of them wrong scales the answer.
        """
        base = compute_gas_permeability(**UNIT_CELL)
        traded = compute_gas_permeability(
            **{**UNIT_CELL, "flow_rate_cm3_s": 0.5, "reference_pressure_atm": 2.0}
        )
        assert traded == pytest.approx(base, rel=1e-12)


class TestRejectedInputs:
    def test_equal_pressures_are_rejected(self):
        with pytest.raises(PermeabilityInputError, match="must exceed outlet"):
            compute_gas_permeability(**{**UNIT_CELL, "inlet_pressure_atm": 1.0})

    def test_reversed_pressures_are_rejected(self):
        with pytest.raises(PermeabilityInputError, match="must exceed outlet"):
            compute_gas_permeability(
                **{**UNIT_CELL, "inlet_pressure_atm": 1.0, "outlet_pressure_atm": 2.0}
            )

    def test_gauge_outlet_mistaken_for_absolute_is_rejected(self):
        """A gauge transducer venting to atmosphere reads 0, not 1 atm."""
        with pytest.raises(PermeabilityInputError, match="absolute, not gauge"):
            compute_gas_permeability(**{**UNIT_CELL, "outlet_pressure_atm": 0.0})

    @pytest.mark.parametrize(
        "field", ["length_cm", "area_cm2", "viscosity_cp", "reference_pressure_atm"]
    )
    def test_non_positive_scalars_are_rejected(self, field: str):
        with pytest.raises(PermeabilityInputError, match=field):
            compute_gas_permeability(**{**UNIT_CELL, field: 0.0})

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_inputs_are_rejected(self, bad: float):
        with pytest.raises(PermeabilityInputError, match="finite"):
            compute_gas_permeability(**{**UNIT_CELL, "flow_rate_cm3_s": bad})


class TestMeanPressure:
    def test_arithmetic_mean(self):
        assert mean_pressure(3.0, 1.0) == pytest.approx(2.0)

    def test_order_does_not_matter(self):
        assert mean_pressure(1.0, 3.0) == mean_pressure(3.0, 1.0)
