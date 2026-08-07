"""Centralized unit conversions.

This module is the **only** place in the package where a unit-conversion
constant is allowed to live. Every other module (``hardware/*``,
``permeability``, ``gas_properties``, ``plotting``, ``storage``) converts by
calling through the functions here.

Internal convention
-------------------
All physics in :mod:`gasperm.permeability` is done in strict **CGS-Darcy**
units, because the compressible-gas Darcy equation only yields permeability in
darcy when its inputs are:

===================== ==========
quantity              unit
===================== ==========
pressure              atm
length                cm
area                  cm^2
volumetric flow rate  cm^3/s
dynamic viscosity     cP
permeability          darcy
===================== ==========

CoolProp, by contrast, speaks strict SI (K, Pa, Pa*s, kg/m^3). The SI<->CGS
boundary lives here and nowhere else -- see :func:`pa_s_to_cp`,
:func:`to_pa`, :func:`to_atm`.
"""

from __future__ import annotations

import math

__all__ = [
    "SUPPORTED_PRESSURE_UNITS",
    "SUPPORTED_FLOW_UNITS",
    "SUPPORTED_LENGTH_UNITS",
    "SUPPORTED_VOLUME_UNITS",
    "SUPPORTED_COMPRESSIBILITY_UNITS",
    "SUPPORTED_PERMEABILITY_UNITS",
    "SUPPORTED_TEMPERATURE_UNITS",
    "ATM_IN_PA",
    "to_pa",
    "from_pa",
    "to_atm",
    "from_atm",
    "pa_to_atm",
    "kpa_to_atm",
    "mpa_to_atm",
    "bar_to_atm",
    "psi_to_atm",
    "flow_to_cm3_s",
    "flow_from_cm3_s",
    "sccm_to_cm3_s",
    "slpm_to_cm3_s",
    "length_to_cm",
    "length_from_cm",
    "volume_to_cm3",
    "volume_from_cm3",
    "compressibility_to_per_atm",
    "compressibility_from_per_atm",
    "per_pa_to_per_atm",
    "per_atm_to_per_pa",
    "pa_s_to_cp",
    "cp_to_pa_s",
    "darcy_to",
    "darcy_from",
    "temperature_to_kelvin",
    "temperature_from_kelvin",
    "celsius_to_kelvin",
    "kelvin_to_celsius",
    "fahrenheit_to_celsius",
    "circle_area_cm2",
    "normalize_pressure_unit",
]


# --------------------------------------------------------------------------
# Pressure
# --------------------------------------------------------------------------

#: The canonical set of pressure units accepted anywhere a pressure unit is
#: configured (inlet/outlet calibration, confining pressure, atmospheric
#: reference, flowmeter standard pressure, display). Every pressure-bearing
#: config field carries its own independent unit drawn from this set.
SUPPORTED_PRESSURE_UNITS: frozenset[str] = frozenset(
    {"Pa", "kPa", "MPa", "bar", "psi", "atm"}
)

#: Standard atmosphere, exact by definition.
ATM_IN_PA: float = 101_325.0

# Exact / CODATA-consistent factors. 1 psi = 1 lbf/in^2 with
# lbf = 4.448_221_615_260_5 N (exact) and in^2 = 6.451_6e-4 m^2 (exact).
_PRESSURE_TO_PA: dict[str, float] = {
    "Pa": 1.0,
    "kPa": 1.0e3,
    "MPa": 1.0e6,
    "bar": 1.0e5,
    "psi": 6_894.757_293_168_361,
    "atm": ATM_IN_PA,
}

# Case-insensitive alias table -> canonical spelling.
_PRESSURE_ALIASES: dict[str, str] = {u.lower(): u for u in _PRESSURE_TO_PA}
_PRESSURE_ALIASES.update(
    {
        "pascal": "Pa",
        "kilopascal": "kPa",
        "megapascal": "MPa",
        "bars": "bar",
        "psia": "psi",
        "atmosphere": "atm",
    }
)


def normalize_pressure_unit(unit: str) -> str:
    """Return the canonical spelling of ``unit`` (e.g. ``"KPA"`` -> ``"kPa"``).

    Raises:
        ValueError: if ``unit`` is not a supported pressure unit.
    """
    canonical = _PRESSURE_ALIASES.get(unit.strip().lower())
    if canonical is None:
        supported = ", ".join(sorted(SUPPORTED_PRESSURE_UNITS))
        raise ValueError(
            f"Unsupported pressure unit {unit!r}. Supported units: {supported}."
        )
    return canonical


def to_pa(value: float, unit: str) -> float:
    """Convert ``value`` expressed in ``unit`` to pascal (the SI boundary unit)."""
    return value * _PRESSURE_TO_PA[normalize_pressure_unit(unit)]


def from_pa(value_pa: float, unit: str) -> float:
    """Convert a pascal value out to ``unit``."""
    return value_pa / _PRESSURE_TO_PA[normalize_pressure_unit(unit)]


def to_atm(value: float, unit: str) -> float:
    """Convert ``value`` in ``unit`` to atm (CGS reference unit used internally)."""
    return to_pa(value, unit) / ATM_IN_PA


def from_atm(value_atm: float, unit: str) -> float:
    """Convert an internal atm value out to ``unit``, for display/storage."""
    return from_pa(value_atm * ATM_IN_PA, unit)


# Named single-unit helpers. These exist so call sites read as prose and so a
# grep for a bare constant like ``101.325`` outside this module is a bug.
def pa_to_atm(value: float) -> float:
    """Pascal -> atm."""
    return to_atm(value, "Pa")


def kpa_to_atm(value: float) -> float:
    """Kilopascal -> atm."""
    return to_atm(value, "kPa")


def mpa_to_atm(value: float) -> float:
    """Megapascal -> atm."""
    return to_atm(value, "MPa")


def bar_to_atm(value: float) -> float:
    """Bar -> atm."""
    return to_atm(value, "bar")


def psi_to_atm(value: float) -> float:
    """Pounds per square inch -> atm."""
    return to_atm(value, "psi")


# --------------------------------------------------------------------------
# Volumetric flow rate
# --------------------------------------------------------------------------

#: Supported volumetric flow units. Note that ``sccm``/``slpm`` denote flow
#: *referenced to standard conditions*; the numeric conversion below is purely
#: volumetric (cm^3/min -> cm^3/s). Which thermodynamic state that volume is
#: referenced to is a separate, explicit config decision -- see
#: ``FlowmeterConfig.reading_basis`` and :mod:`gasperm.acquisition`.
SUPPORTED_FLOW_UNITS: frozenset[str] = frozenset(
    {"cm3/s", "cm3/min", "sccm", "mL/s", "mL/min", "L/s", "L/min", "slpm", "m3/h"}
)

_FLOW_TO_CM3_S: dict[str, float] = {
    "cm3/s": 1.0,
    "cm3/min": 1.0 / 60.0,
    "sccm": 1.0 / 60.0,  # standard cm^3 per minute
    "mL/s": 1.0,  # 1 mL == 1 cm^3 exactly
    "mL/min": 1.0 / 60.0,
    "L/s": 1000.0,
    "L/min": 1000.0 / 60.0,
    "slpm": 1000.0 / 60.0,  # standard litres per minute
    "m3/h": 1.0e6 / 3600.0,
}

_FLOW_ALIASES: dict[str, str] = {u.lower(): u for u in _FLOW_TO_CM3_S}
_FLOW_ALIASES.update(
    {
        "ccm": "cm3/min",
        "ml/min": "mL/min",
        "ml/s": "mL/s",
        "sl/min": "slpm",
        "slm": "slpm",
        "lpm": "L/min",
        "cc/s": "cm3/s",
        "cc/min": "cm3/min",
    }
)


def _normalize_flow_unit(unit: str) -> str:
    canonical = _FLOW_ALIASES.get(unit.strip().lower())
    if canonical is None:
        supported = ", ".join(sorted(SUPPORTED_FLOW_UNITS))
        raise ValueError(
            f"Unsupported flow unit {unit!r}. Supported units: {supported}."
        )
    return canonical


def flow_to_cm3_s(value: float, unit: str) -> float:
    """Convert a volumetric flow rate in ``unit`` to cm^3/s."""
    return value * _FLOW_TO_CM3_S[_normalize_flow_unit(unit)]


def flow_from_cm3_s(value_cm3_s: float, unit: str) -> float:
    """Convert a cm^3/s flow rate out to ``unit``."""
    return value_cm3_s / _FLOW_TO_CM3_S[_normalize_flow_unit(unit)]


def sccm_to_cm3_s(value: float) -> float:
    """Standard cm^3/min -> cm^3/s (at the meter's standard state)."""
    return flow_to_cm3_s(value, "sccm")


def slpm_to_cm3_s(value: float) -> float:
    """Standard L/min -> cm^3/s (at the meter's standard state)."""
    return flow_to_cm3_s(value, "slpm")


# --------------------------------------------------------------------------
# Length / area
# --------------------------------------------------------------------------

SUPPORTED_LENGTH_UNITS: frozenset[str] = frozenset({"cm", "mm", "m", "in", "ft"})

_LENGTH_TO_CM: dict[str, float] = {
    "cm": 1.0,
    "mm": 0.1,
    "m": 100.0,
    "in": 2.54,  # exact
    "ft": 30.48,  # exact
}


def _normalize_length_unit(unit: str) -> str:
    canonical = unit.strip().lower()
    if canonical == "inch":
        canonical = "in"
    if canonical not in _LENGTH_TO_CM:
        supported = ", ".join(sorted(SUPPORTED_LENGTH_UNITS))
        raise ValueError(
            f"Unsupported length unit {unit!r}. Supported units: {supported}."
        )
    return canonical


def length_to_cm(value: float, unit: str) -> float:
    """Convert a length in ``unit`` to cm."""
    return value * _LENGTH_TO_CM[_normalize_length_unit(unit)]


def length_from_cm(value_cm: float, unit: str) -> float:
    """Convert a cm length out to ``unit``."""
    return value_cm / _LENGTH_TO_CM[_normalize_length_unit(unit)]


def circle_area_cm2(diameter_cm: float) -> float:
    """Cross-sectional area (cm^2) of a cylindrical plug of the given diameter."""
    if diameter_cm <= 0.0:
        raise ValueError(f"diameter must be positive, got {diameter_cm}")
    return math.pi * (diameter_cm / 2.0) ** 2


# --------------------------------------------------------------------------
# Volume
# --------------------------------------------------------------------------

#: Units for the pulse-decay reservoir volumes. cm^3 is the internal unit,
#: matching the CGS-Darcy system the physics is worked in.
SUPPORTED_VOLUME_UNITS: frozenset[str] = frozenset({"cm3", "mL", "L", "m3", "in3"})

_VOLUME_TO_CM3: dict[str, float] = {
    "cm3": 1.0,
    "mL": 1.0,  # 1 mL == 1 cm^3 exactly, since the 1964 redefinition of the litre
    "L": 1.0e3,
    "m3": 1.0e6,
    "in3": 16.387_064,  # 2.54^3, exact
}

_VOLUME_ALIASES: dict[str, str] = {u.lower(): u for u in _VOLUME_TO_CM3}
_VOLUME_ALIASES.update(
    {
        "cc": "cm3",
        "cm^3": "cm3",
        "ccm": "cm3",
        "millilitre": "mL",
        "milliliter": "mL",
        "litre": "L",
        "liter": "L",
        "m^3": "m3",
        "in^3": "in3",
        "cu_in": "in3",
        "cubic_inch": "in3",
    }
)


def _normalize_volume_unit(unit: str) -> str:
    canonical = _VOLUME_ALIASES.get(unit.strip().lower())
    if canonical is None:
        supported = ", ".join(sorted(SUPPORTED_VOLUME_UNITS))
        raise ValueError(
            f"Unsupported volume unit {unit!r}. Supported units: {supported}."
        )
    return canonical


def volume_to_cm3(value: float, unit: str) -> float:
    """Convert a volume in ``unit`` to cm^3 (the internal unit)."""
    return value * _VOLUME_TO_CM3[_normalize_volume_unit(unit)]


def volume_from_cm3(value_cm3: float, unit: str) -> float:
    """Convert a cm^3 volume out to ``unit``, for display or storage."""
    return value_cm3 / _VOLUME_TO_CM3[_normalize_volume_unit(unit)]


# --------------------------------------------------------------------------
# Compressibility (reciprocal pressure)
# --------------------------------------------------------------------------

#: Isothermal gas compressibility is a reciprocal pressure, so its units are
#: exactly the pressure units inverted. The internal unit is 1/atm, which is
#: what makes the pulse-decay equation come out in darcy with no extra factor.
#:
#: There is deliberately no separate constant table: the factors are derived
#: from :data:`_PRESSURE_TO_PA`, so adding a pressure unit extends this family
#: automatically. That is what "units.py owns every constant" means here.
SUPPORTED_COMPRESSIBILITY_UNITS: frozenset[str] = frozenset(
    f"1/{unit}" for unit in SUPPORTED_PRESSURE_UNITS
)


def _normalize_compressibility_unit(unit: str) -> str:
    """Canonicalise ``1/kPa``, ``kPa^-1`` or ``per_kPa`` to ``1/kPa``.

    A bare pressure unit is rejected rather than assumed: ``atm`` and ``1/atm``
    differ by six orders of magnitude at a typical pore pressure, and silently
    guessing which was meant is exactly the class of error this module exists
    to prevent.
    """
    text = unit.strip()
    lowered = text.lower()
    for prefix in ("1/", "per_", "per "):
        if lowered.startswith(prefix):
            base = text[len(prefix) :]
            break
    else:
        if lowered.endswith("^-1"):
            base = text[:-3]
        elif lowered.endswith("-1"):
            base = text[:-2]
        else:
            supported = ", ".join(sorted(SUPPORTED_COMPRESSIBILITY_UNITS))
            raise ValueError(
                f"Unsupported compressibility unit {unit!r}: it must be a reciprocal "
                f"pressure, written like '1/kPa'. Supported units: {supported}."
            )
    return f"1/{normalize_pressure_unit(base)}"


def compressibility_to_per_atm(value: float, unit: str) -> float:
    """Convert a compressibility in ``unit`` to 1/atm (the internal unit)."""
    canonical = _normalize_compressibility_unit(unit)[2:]
    return value * ATM_IN_PA / _PRESSURE_TO_PA[canonical]


def compressibility_from_per_atm(value_per_atm: float, unit: str) -> float:
    """Convert a 1/atm compressibility out to ``unit``."""
    canonical = _normalize_compressibility_unit(unit)[2:]
    return value_per_atm * _PRESSURE_TO_PA[canonical] / ATM_IN_PA


def per_pa_to_per_atm(value: float) -> float:
    """Convert a 1/Pa compressibility to 1/atm.

    Named because it is the boundary CoolProp returns at, mirroring
    :func:`pa_to_atm` for pressures themselves.
    """
    return value * ATM_IN_PA


def per_atm_to_per_pa(value: float) -> float:
    """Convert a 1/atm compressibility back to 1/Pa."""
    return value / ATM_IN_PA


# --------------------------------------------------------------------------
# Viscosity
# --------------------------------------------------------------------------

#: 1 Pa*s = 10 P = 1000 cP, exact.
_PA_S_IN_CP: float = 1000.0


def pa_s_to_cp(value_pa_s: float) -> float:
    """Dynamic viscosity Pa*s (CoolProp/SI) -> cP (CGS-Darcy)."""
    return value_pa_s * _PA_S_IN_CP


def cp_to_pa_s(value_cp: float) -> float:
    """Dynamic viscosity cP -> Pa*s."""
    return value_cp / _PA_S_IN_CP


# --------------------------------------------------------------------------
# Permeability
# --------------------------------------------------------------------------

SUPPORTED_PERMEABILITY_UNITS: frozenset[str] = frozenset({"D", "mD", "uD", "um2", "m2"})

#: 1 darcy = 9.869233e-13 m^2 = 0.9869233 um^2.
_DARCY_IN_M2: float = 9.869_233e-13

_PER_DARCY: dict[str, float] = {
    "D": 1.0,
    "mD": 1.0e3,
    "uD": 1.0e6,
    "um2": _DARCY_IN_M2 * 1.0e12,  # m^2 -> um^2
    "m2": _DARCY_IN_M2,
}

_PERMEABILITY_ALIASES: dict[str, str] = {u.lower(): u for u in _PER_DARCY}
_PERMEABILITY_ALIASES.update(
    {
        "darcy": "D",
        "millidarcy": "mD",
        "md": "mD",
        "microdarcy": "uD",
        "ud": "uD",
        "µm2": "um2",
        "um^2": "um2",
        "m^2": "m2",
    }
)


def _normalize_permeability_unit(unit: str) -> str:
    canonical = _PERMEABILITY_ALIASES.get(unit.strip().lower())
    if canonical is None:
        supported = ", ".join(sorted(SUPPORTED_PERMEABILITY_UNITS))
        raise ValueError(
            f"Unsupported permeability unit {unit!r}. Supported units: {supported}."
        )
    return canonical


def darcy_to(value_darcy: float, unit: str) -> float:
    """Convert an internal darcy value out to ``unit`` (display/storage)."""
    return value_darcy * _PER_DARCY[_normalize_permeability_unit(unit)]


def darcy_from(value: float, unit: str) -> float:
    """Convert a permeability expressed in ``unit`` back to darcy."""
    return value / _PER_DARCY[_normalize_permeability_unit(unit)]


# --------------------------------------------------------------------------
# Temperature
# --------------------------------------------------------------------------

SUPPORTED_TEMPERATURE_UNITS: frozenset[str] = frozenset({"C", "K", "F"})

_TEMPERATURE_ALIASES: dict[str, str] = {
    "c": "C",
    "degc": "C",
    "celsius": "C",
    "k": "K",
    "kelvin": "K",
    "f": "F",
    "degf": "F",
    "fahrenheit": "F",
}

#: 0 degC in kelvin, exact.
_ZERO_C_IN_K: float = 273.15


def _normalize_temperature_unit(unit: str) -> str:
    canonical = _TEMPERATURE_ALIASES.get(unit.strip().lower())
    if canonical is None:
        supported = ", ".join(sorted(SUPPORTED_TEMPERATURE_UNITS))
        raise ValueError(
            f"Unsupported temperature unit {unit!r}. Supported units: {supported}."
        )
    return canonical


def celsius_to_kelvin(value_c: float) -> float:
    """degC -> K."""
    return value_c + _ZERO_C_IN_K


def kelvin_to_celsius(value_k: float) -> float:
    """K -> degC."""
    return value_k - _ZERO_C_IN_K


def fahrenheit_to_celsius(value_f: float) -> float:
    """degF -> degC."""
    return (value_f - 32.0) * 5.0 / 9.0


def temperature_to_kelvin(value: float, unit: str) -> float:
    """Convert a temperature in ``unit`` to kelvin (the CoolProp/SI boundary)."""
    canonical = _normalize_temperature_unit(unit)
    if canonical == "K":
        return value
    if canonical == "C":
        return celsius_to_kelvin(value)
    return celsius_to_kelvin(fahrenheit_to_celsius(value))


def temperature_from_kelvin(value_k: float, unit: str) -> float:
    """Convert a kelvin temperature out to ``unit``."""
    canonical = _normalize_temperature_unit(unit)
    if canonical == "K":
        return value_k
    if canonical == "C":
        return kelvin_to_celsius(value_k)
    return kelvin_to_celsius(value_k) * 9.0 / 5.0 + 32.0
