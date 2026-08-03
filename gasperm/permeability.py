"""Core physics: steady-state compressible-gas (Darcy) permeability.

Hardware-free by construction. Every function here takes plain floats already
in **CGS-Darcy units** and returns a plain float, so it is unit-testable
against hand-calculated reference values with no DAQ, no serial port and no
CoolProp object in sight. The caller (:mod:`gasperm.acquisition`) resolves the
viscosity via :mod:`gasperm.gas_properties` and converts units via
:mod:`gasperm.units` *before* calling in here.
"""

from __future__ import annotations

import math

__all__ = [
    "compute_gas_permeability",
    "mean_pressure",
    "PermeabilityInputError",
]


class PermeabilityInputError(ValueError):
    """Raised when inputs cannot yield a physically meaningful permeability."""


def compute_gas_permeability(
    *,
    flow_rate_cm3_s: float,
    reference_pressure_atm: float,
    viscosity_cp: float,
    length_cm: float,
    area_cm2: float,
    inlet_pressure_atm: float,
    outlet_pressure_atm: float,
) -> float:
    """Apparent gas permeability in **darcy**.

    Implements the steady-state compressible-gas form of Darcy's law::

        k_g = (2 * Q_ref * P_ref * mu * L) / (A * (P1^2 - P2^2))

    The ``2 * Q_ref * P_ref`` numerator is what makes this the *compressible*
    form: gas expands as it travels down the plug, so the volumetric rate is
    only meaningful when paired with the pressure it was measured at. The
    product ``Q_ref * P_ref`` is the invariant (proportional to molar flow for
    an ideal gas at fixed temperature), which is why the caller must pass a
    matched pair -- a standard-referenced flow with the meter's standard
    pressure, or a line-conditions flow with that line's pressure.

    This equation returns darcy **only** in CGS-Darcy units. Every argument is
    named with its unit for that reason; there are no conversions inside this
    function.

    Args:
        flow_rate_cm3_s: Volumetric flow rate Q_ref, cm^3/s, at
            ``reference_pressure_atm``.
        reference_pressure_atm: P_ref, absolute atm -- the pressure the flow
            rate is referenced to.
        viscosity_cp: Gas dynamic viscosity mu, cP, at the test conditions.
        length_cm: Plug length L, cm.
        area_cm2: Plug cross-sectional area A, cm^2.
        inlet_pressure_atm: P1, **absolute** atm.
        outlet_pressure_atm: P2, **absolute** atm.

    Returns:
        Apparent gas permeability, darcy. Convert to mD/um^2 at the display
        boundary with :func:`gasperm.units.darcy_to`.

    Raises:
        PermeabilityInputError: on non-positive geometry, non-positive
            viscosity, non-finite inputs, or a pressure differential that is
            zero, reversed, or non-physical.
    """
    arguments = {
        "flow_rate_cm3_s": flow_rate_cm3_s,
        "reference_pressure_atm": reference_pressure_atm,
        "viscosity_cp": viscosity_cp,
        "length_cm": length_cm,
        "area_cm2": area_cm2,
        "inlet_pressure_atm": inlet_pressure_atm,
        "outlet_pressure_atm": outlet_pressure_atm,
    }
    for name, value in arguments.items():
        if not math.isfinite(value):
            raise PermeabilityInputError(f"{name} must be a finite number, got {value!r}")

    if length_cm <= 0.0:
        raise PermeabilityInputError(f"length_cm must be positive, got {length_cm}")
    if area_cm2 <= 0.0:
        raise PermeabilityInputError(f"area_cm2 must be positive, got {area_cm2}")
    if viscosity_cp <= 0.0:
        raise PermeabilityInputError(f"viscosity_cp must be positive, got {viscosity_cp}")
    if reference_pressure_atm <= 0.0:
        raise PermeabilityInputError(
            f"reference_pressure_atm must be positive (absolute), got "
            f"{reference_pressure_atm}"
        )
    if outlet_pressure_atm <= 0.0:
        raise PermeabilityInputError(
            f"outlet_pressure_atm must be positive (absolute, not gauge), got "
            f"{outlet_pressure_atm}"
        )
    if inlet_pressure_atm <= outlet_pressure_atm:
        raise PermeabilityInputError(
            f"inlet pressure ({inlet_pressure_atm} atm) must exceed outlet pressure "
            f"({outlet_pressure_atm} atm); with no differential there is no flow to "
            "invert. Check for a stalled regulator, swapped transducer channels, or "
            "gauge readings not converted to absolute."
        )

    pressure_squared_difference = inlet_pressure_atm**2 - outlet_pressure_atm**2
    numerator = 2.0 * flow_rate_cm3_s * reference_pressure_atm * viscosity_cp * length_cm
    denominator = area_cm2 * pressure_squared_difference
    return numerator / denominator


def mean_pressure(inlet_pressure_atm: float, outlet_pressure_atm: float) -> float:
    """Mean pore pressure ``(P1 + P2) / 2``, absolute atm.

    This is the pressure the Klinkenberg correction regresses against, and --
    by the documented modelling choice in :mod:`gasperm.gas_properties` -- the
    pressure at which viscosity is evaluated.
    """
    return (inlet_pressure_atm + outlet_pressure_atm) / 2.0
