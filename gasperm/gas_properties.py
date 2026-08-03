"""Gas thermophysical properties from CoolProp.

Viscosity is *not* a static config constant -- it is looked up live for the
gas's actual (T, P) at each reading, because both drift during a run,
especially before the rig reaches steady state.

**Unit boundary.** CoolProp's ``PropsSI`` takes and returns strict SI
(K, Pa, Pa*s, kg/m^3). This module is where SI enters and leaves; conversion
to the internal CGS-Darcy convention happens through :mod:`gasperm.units` and
nowhere else. Nothing downstream of :class:`GasPropertyProvider` ever sees a
pascal.

**Modelling choice.** Viscosity is evaluated at the *mean pore pressure*
``(P1 + P2) / 2`` (absolute) and the current measured temperature. That is a
deliberate decision, not a CoolProp default: the gas spans P1..P2 across the
plug, and the mean is the standard single-point representative used with the
mean-pressure form of the Darcy equation.
"""

from __future__ import annotations

import logging
import math
from typing import Final

from gasperm import units
from gasperm.models import GasState

logger = logging.getLogger(__name__)

__all__ = [
    "CoolPropUnavailable",
    "dynamic_viscosity_pa_s",
    "density_kg_m3",
    "compressibility_factor",
    "validate_gas_name",
    "GasPropertyProvider",
    "CoolPropProvider",
    "FixedPropertyProvider",
    "build_provider",
]

#: Standard-ish conditions used to smoke-test a fluid name at startup.
_VALIDATION_TEMPERATURE_K: Final[float] = 298.15
_VALIDATION_PRESSURE_PA: Final[float] = units.ATM_IN_PA


class CoolPropUnavailable(RuntimeError):
    """CoolProp is not importable or its lookup failed at the C++ layer."""


def _props_si():
    """Import ``PropsSI`` lazily so the package imports without CoolProp."""
    try:
        from CoolProp.CoolProp import PropsSI
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise CoolPropUnavailable(
            "CoolProp is not installed. Install it with 'pip install CoolProp', or set "
            "gas.properties_source: fixed in the config."
        ) from exc
    return PropsSI


def dynamic_viscosity_pa_s(gas_name: str, temperature_k: float, pressure_pa: float) -> float:
    """Dynamic viscosity (Pa*s) of ``gas_name`` at the given T, P."""
    return float(_props_si()("V", "T", temperature_k, "P", pressure_pa, gas_name))


def density_kg_m3(gas_name: str, temperature_k: float, pressure_pa: float) -> float:
    """Mass density (kg/m^3) of ``gas_name`` at the given T, P."""
    return float(_props_si()("D", "T", temperature_k, "P", pressure_pa, gas_name))


def compressibility_factor(gas_name: str, temperature_k: float, pressure_pa: float) -> float:
    """Compressibility factor Z (dimensionless) of ``gas_name`` at T, P."""
    return float(_props_si()("Z", "T", temperature_k, "P", pressure_pa, gas_name))


def validate_gas_name(gas_name: str) -> None:
    """Confirm CoolProp recognises ``gas_name``, by attempting a lookup at STP.

    Called at ``init``/``collect`` time so a typo surfaces immediately rather
    than deep into a multi-minute run.

    Raises:
        ValueError: the name is not a CoolProp fluid.
        CoolPropUnavailable: CoolProp itself could not be used.
    """
    props_si = _props_si()
    try:
        viscosity = props_si(
            "V", "T", _VALIDATION_TEMPERATURE_K, "P", _VALIDATION_PRESSURE_PA, gas_name
        )
    except (ValueError, RuntimeError) as exc:
        raise ValueError(
            f"gas.name {gas_name!r} is not a fluid CoolProp recognises (or has no "
            f"viscosity model at standard conditions). CoolProp said: {exc}. Try one of "
            "'Nitrogen', 'Air', 'CarbonDioxide', 'Methane', 'Helium', 'Argon'."
        ) from exc
    if not viscosity or viscosity <= 0.0:
        raise ValueError(
            f"CoolProp returned a non-physical viscosity ({viscosity}) for "
            f"gas.name {gas_name!r} at standard conditions."
        )


class GasPropertyProvider:
    """Resolves a :class:`~gasperm.models.GasState` from (T, P).

    Subclasses implement :meth:`_lookup`; this base adds the tolerance-based
    cache described in CLAUDE.md.
    """

    def __init__(
        self,
        gas_name: str,
        *,
        temperature_tolerance_k: float = 0.05,
        pressure_tolerance_frac: float = 0.002,
        relative_uncertainty: float = 0.0,
    ) -> None:
        """Args:
        gas_name: CoolProp fluid name (or a label, for the fixed provider).
        temperature_tolerance_k: Reuse the cached state while T has moved
            less than this. 0.05 K changes gas viscosity by well under
            0.02%, far below transducer noise.
        pressure_tolerance_frac: Same idea for pressure, as a fraction of
            the cached pressure. Gas viscosity is nearly pressure-independent
            near ambient, so 0.2% is conservative.
        relative_uncertainty: Relative standard uncertainty of the viscosity
            model, carried on every state for the GUM budget.
        """
        self.gas_name = gas_name
        self.temperature_tolerance_k = temperature_tolerance_k
        self.pressure_tolerance_frac = pressure_tolerance_frac
        self.relative_uncertainty = relative_uncertainty
        self._cached: GasState | None = None
        self.lookup_count = 0
        self.cache_hits = 0

    # -- public API -------------------------------------------------------

    def state_at(self, temperature_k: float, pressure_pa: float) -> GasState:
        """Gas state at (T, P), reusing the cache when the change is negligible.

        The cache is deliberately *not* time-based: it invalidates on the
        physical quantities that actually change viscosity, so a fast-drifting
        rig re-looks-up every sample while a settled one does not.
        """
        cached = self._cached
        if cached is not None and self._within_tolerance(cached, temperature_k, pressure_pa):
            self.cache_hits += 1
            return cached
        state = self._lookup(temperature_k, pressure_pa)
        self.lookup_count += 1
        self._cached = state
        return state

    def viscosity_cp_at(self, temperature_k: float, pressure_pa: float) -> float:
        """Convenience: dynamic viscosity in **cP**, ready for the Darcy equation."""
        return self.state_at(temperature_k, pressure_pa).viscosity_cp

    def state_at_cgs(self, temperature_c: float, pressure_atm: float) -> GasState:
        """Same as :meth:`state_at` but taking the internal CGS-ish units.

        Callers in the acquisition loop hold degC and atm; this keeps the
        SI conversion inside the module that owns the SI boundary.
        """
        return self.state_at(
            units.celsius_to_kelvin(temperature_c),
            pressure_atm * units.ATM_IN_PA,
        )

    def viscosity_temperature_exponent(
        self, temperature_k: float, pressure_pa: float, *, delta_k: float = 0.5
    ) -> float:
        """``d ln mu / d ln T`` at the given state, by central difference.

        This is the sensitivity coefficient the GUM budget needs to propagate
        the temperature measurement's uncertainty into the permeability -- the
        only route by which temperature enters the Darcy equation at all. For
        gases it is around +0.7 (viscosity *rises* with temperature, unlike a
        liquid), so it is not negligible.

        The cache is restored afterwards so probing the derivative never leaves
        a neighbouring state cached for the acquisition loop.
        """
        saved = self._cached
        try:
            low = self._lookup(temperature_k - delta_k, pressure_pa).viscosity_cp
            high = self._lookup(temperature_k + delta_k, pressure_pa).viscosity_cp
        except (ValueError, CoolPropUnavailable):
            # A derivative is a diagnostic, not a measurement: if the fluid
            # model will not evaluate slightly off-state, report no sensitivity
            # rather than failing the whole budget.
            return 0.0
        finally:
            self._cached = saved

        if low <= 0.0 or high <= 0.0:
            return 0.0
        return (math.log(high) - math.log(low)) / (
            math.log(temperature_k + delta_k) - math.log(temperature_k - delta_k)
        )

    # -- hooks ------------------------------------------------------------

    def _lookup(self, temperature_k: float, pressure_pa: float) -> GasState:
        raise NotImplementedError

    def _within_tolerance(
        self, cached: GasState, temperature_k: float, pressure_pa: float
    ) -> bool:
        if abs(cached.temperature_k - temperature_k) > self.temperature_tolerance_k:
            return False
        allowed = self.pressure_tolerance_frac * max(cached.pressure_pa, 1.0)
        return abs(cached.pressure_pa - pressure_pa) <= allowed


class CoolPropProvider(GasPropertyProvider):
    """Live CoolProp lookups -- the default and recommended path."""

    def __init__(self, gas_name: str, *, include_derived: bool = True, **kwargs) -> None:
        """Args:
        gas_name: CoolProp fluid name.
        include_derived: Also fetch density and Z. They are logged with each
            reading for diagnostics but are not used by the Darcy equation.
        """
        super().__init__(gas_name, **kwargs)
        self.include_derived = include_derived

    def _lookup(self, temperature_k: float, pressure_pa: float) -> GasState:
        try:
            viscosity_pa_s = dynamic_viscosity_pa_s(self.gas_name, temperature_k, pressure_pa)
            density = (
                density_kg_m3(self.gas_name, temperature_k, pressure_pa)
                if self.include_derived
                else None
            )
            z_factor = (
                compressibility_factor(self.gas_name, temperature_k, pressure_pa)
                if self.include_derived
                else None
            )
        except CoolPropUnavailable:
            raise
        except (ValueError, RuntimeError) as exc:
            raise ValueError(
                f"CoolProp could not evaluate {self.gas_name!r} at "
                f"T = {temperature_k:.2f} K, P = {pressure_pa:.0f} Pa: {exc}"
            ) from exc

        return GasState(
            gas_name=self.gas_name,
            temperature_k=temperature_k,
            pressure_pa=pressure_pa,
            viscosity_cp=units.pa_s_to_cp(viscosity_pa_s),
            density_kg_m3=density,
            compressibility_z=z_factor,
            source="coolprop",
            relative_viscosity_uncertainty=self.relative_uncertainty,
        )


class FixedPropertyProvider(GasPropertyProvider):
    """Escape hatch: a constant viscosity, no (T, P) dependence at all.

    Only for constrained deployments without CoolProp, or a deliberate
    reference-condition comparison. The reason is carried through into the run
    log so a result produced this way is never mistaken for a live lookup.
    """

    def __init__(
        self,
        gas_name: str,
        viscosity_cp: float,
        *,
        reason: str = "",
        relative_uncertainty: float = 0.0,
    ) -> None:
        super().__init__(gas_name, relative_uncertainty=relative_uncertainty)
        if viscosity_cp <= 0.0:
            raise ValueError(f"fixed viscosity must be positive, got {viscosity_cp}")
        self.viscosity_cp = viscosity_cp
        self.reason = reason
        logger.warning(
            "Using a FIXED viscosity of %.6g cP for %s -- the live CoolProp (T, P) "
            "lookup is bypassed.%s",
            viscosity_cp,
            gas_name,
            f" Reason: {reason}" if reason else "",
        )

    def _lookup(self, temperature_k: float, pressure_pa: float) -> GasState:
        return GasState(
            gas_name=self.gas_name,
            temperature_k=temperature_k,
            pressure_pa=pressure_pa,
            viscosity_cp=self.viscosity_cp,
            source="fixed",
            relative_viscosity_uncertainty=self.relative_uncertainty,
        )

    def _within_tolerance(self, cached, temperature_k, pressure_pa) -> bool:
        # A constant is always "in tolerance", but the returned state still
        # records the T/P it was requested at, so never serve a stale one.
        return False


def build_provider(gas_config) -> GasPropertyProvider:
    """Construct the provider named by ``gas.properties_source`` in the config.

    Args:
        gas_config: A :class:`gasperm.config.GasConfig`.
    """
    if gas_config.properties_source == "fixed":
        if gas_config.fixed_viscosity_cp is None:
            raise ValueError(
                "gas.properties_source is 'fixed' but gas.fixed_viscosity_cp is unset"
            )
        return FixedPropertyProvider(
            gas_config.name,
            gas_config.fixed_viscosity_cp,
            reason=gas_config.fixed_reason,
            relative_uncertainty=gas_config.viscosity_relative_uncertainty,
        )
    return CoolPropProvider(
        gas_config.name,
        relative_uncertainty=gas_config.viscosity_relative_uncertainty,
    )
