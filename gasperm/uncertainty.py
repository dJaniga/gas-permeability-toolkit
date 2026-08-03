"""Uncertainty propagation per ISO/IEC Guide 98-3 (the GUM).

The measurand is the apparent gas permeability::

    k = 2 Q P_ref mu L / (A (P1^2 - P2^2)),   A = pi d^2 / 4

Because that is a product of powers, the GUM law of propagation of uncertainty
(equation 10) takes a particularly clean form: the *relative* sensitivity
coefficient of each input is simply its exponent,

    c_i^rel = (x_i / k) (dk / dx_i)

so ``u_c(k)/k`` is a quadrature sum of ``|c_i^rel| u(x_i)/x_i``. The exponents
are +1 for Q, P_ref, mu and L; -2 for the diameter (area goes as d^2, which is
why the caliper term is usually the largest in the budget); and, for the two
pressures,

    c_P1 = -2 P1^2 / (P1^2 - P2^2)      c_P2 = +2 P2^2 / (P1^2 - P2^2)

which blow up as the differential closes -- the quantitative statement of why
a low-differential measurement is a bad measurement.

**Correlation.** P1 and P2 usually come from two units of the same transducer
model calibrated against the same reference, so their errors are correlated.
GUM equation (13) requires the covariance term, and because the two enter with
opposite signs a positive correlation *reduces* the combined uncertainty.
Ignoring it is conservative but wrong; ``pressure_calibration.correlation``
makes it explicit.

**Type A vs Type B.** The Type A term is the standard deviation of the mean of
the steady-state window -- the run's own random scatter. The Type B terms come
from instrument specifications and represent systematic calibration error.
Treating them as independent is the standard treatment and is what the
separation of the two evaluation types is for; it does mean the random part of
a transducer's specification is counted twice, which errs high.

**Neglected.** Viscosity's weak pressure dependence: for common gases a few
atm from ambient, ``d ln mu / d ln P`` is order 1e-3, far below the viscosity
model's own uncertainty. Temperature's effect *is* included, through
``d ln mu / d ln T``.

Hardware-free: everything here takes plain numbers and config objects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from gasperm import units
from gasperm.config.hardware import HardwareConfig, PressureChannelConfig
from gasperm.config.run import RunConfig
from gasperm.models import SampleGeometry, UncertaintyBudget, UncertaintyComponent

#: Above this, the Welch-Satterthwaite result is reported as infinite: the
#: Student-t factor is already within rounding of the normal quantile.
_EFFECTIVE_DOF_INFINITE = 1.0e6

__all__ = [
    "MeasurementPoint",
    "pressure_sensitivities",
    "coverage_factor",
    "build_budget",
]


@dataclass(frozen=True)
class MeasurementPoint:
    """The steady-state means the budget is evaluated at, in internal units."""

    permeability_darcy: float
    inlet_pressure_atm: float
    outlet_pressure_atm: float
    flow_cm3_s: float
    reference_pressure_atm: float
    viscosity_cp: float
    temperature_c: float


def pressure_sensitivities(
    inlet_pressure_atm: float, outlet_pressure_atm: float
) -> tuple[float, float]:
    """Relative sensitivity coefficients of ``k`` to P1 and P2.

    Returns:
        ``(c_P1, c_P2)``, dimensionless. ``c_P1`` is negative and ``c_P2``
        positive; both diverge as ``P1 -> P2``.

    Raises:
        ValueError: the pressures are equal, so ``k`` is not defined.
    """
    denominator = inlet_pressure_atm**2 - outlet_pressure_atm**2
    if denominator == 0.0:
        raise ValueError(
            "inlet and outlet pressures are equal, so the sensitivity coefficients "
            "are undefined"
        )
    return (
        -2.0 * inlet_pressure_atm**2 / denominator,
        2.0 * outlet_pressure_atm**2 / denominator,
    )


def coverage_factor(effective_dof: float, coverage_probability: float) -> float:
    """Coverage factor ``k`` for a level of confidence, per GUM annex G.

    Student-t at the effective degrees of freedom, falling back to the normal
    quantile when the degrees of freedom are infinite (all-Type-B budgets).
    """
    if not 0.0 < coverage_probability < 1.0:
        raise ValueError(
            f"coverage_probability must be in (0, 1), got {coverage_probability}"
        )
    from scipy import stats

    tail = 1.0 - (1.0 - coverage_probability) / 2.0
    if not math.isfinite(effective_dof) or effective_dof > 1e6:
        return float(stats.norm.ppf(tail))
    # Below 1 dof the t distribution has no usable variance; clamp so a budget
    # built from a single sample still reports something finite and obviously
    # large rather than raising.
    return float(stats.t.ppf(tail, max(effective_dof, 1.0)))


def _pressure_relative_uncertainty(
    channel: PressureChannelConfig,
    absolute_pressure_atm: float,
    run: RunConfig,
    daq_relative: float,
) -> tuple[float, float]:
    """``(u_rel, dof)`` for one pressure channel at the given absolute pressure.

    The specification applies to what the *transducer* reads. For a gauge
    channel that is the gauge value, and the absolute pressure the equation
    needs is that reading plus the ambient reference -- so the ambient value's
    own uncertainty joins in quadrature. (If both channels are gauge they share
    that ambient error; express the resulting correlation through
    ``pressure_calibration.correlation``.) The relative uncertainty is taken
    against the absolute pressure, which is what enters the equation.
    """
    reading_atm = absolute_pressure_atm
    if channel.reading_type == "gauge":
        reading_atm = absolute_pressure_atm - run.atmospheric_pressure_atm
    reading_in_unit = units.from_atm(reading_atm, channel.unit)

    u_in_unit = channel.uncertainty.standard_uncertainty(reading_in_unit, channel.full_scale)
    # Pressure units are pure scalings, so a difference converts like a value.
    u_atm = units.to_atm(u_in_unit, channel.unit)

    if channel.reading_type == "gauge":
        ambient_spec = run.atmospheric_pressure_uncertainty
        u_ambient_atm = units.to_atm(
            ambient_spec.standard_uncertainty(
                run.atmospheric_pressure, abs(run.atmospheric_pressure)
            ),
            run.atmospheric_pressure_unit,
        )
        u_atm = math.hypot(u_atm, u_ambient_atm)

    if absolute_pressure_atm == 0.0:
        return math.inf, channel.uncertainty.dof
    u_relative = u_atm / abs(absolute_pressure_atm)
    # The DAQ's own voltage error rides on the same signal.
    return math.hypot(u_relative, daq_relative), channel.uncertainty.dof


def _component(
    *,
    name: str,
    symbol: str,
    evaluation_type: str,
    value: float,
    unit: str,
    relative_uncertainty: float,
    sensitivity: float,
    dof: float,
    source: str,
) -> UncertaintyComponent:
    return UncertaintyComponent(
        name=name,
        symbol=symbol,
        evaluation_type=evaluation_type,
        value=value,
        unit=unit,
        standard_uncertainty=abs(value) * relative_uncertainty,
        relative_standard_uncertainty=relative_uncertainty,
        relative_sensitivity=sensitivity,
        relative_contribution=abs(sensitivity * relative_uncertainty),
        degrees_of_freedom=dof,
        source=source,
    )


def build_budget(
    point: MeasurementPoint,
    geometry: SampleGeometry,
    hardware: HardwareConfig,
    run: RunConfig,
    *,
    type_a_relative: float | None = None,
    type_a_dof: float = math.inf,
    viscosity_temperature_exponent: float = 0.0,
) -> UncertaintyBudget:
    """Evaluate the full GUM budget for one measured permeability.

    Args:
        point: Steady-state means, in internal CGS units.
        geometry: Plug geometry with its caliper uncertainties.
        hardware: Instrument specifications.
        run: Coverage settings and how P2 was determined.
        type_a_relative: Relative standard uncertainty of the mean from the
            steady-state window's scatter, ``s / (sqrt(n) |mean|)``. ``None``
            omits the Type A term.
        type_a_dof: Degrees of freedom for that term, normally ``n - 1``.
        viscosity_temperature_exponent: ``d ln mu / d ln T`` at the test
            conditions, from :mod:`gasperm.gas_properties`. Zero omits the
            temperature term.

    Returns:
        The budget, with every component and the expanded uncertainty.

    Raises:
        ValueError: the pressures are equal, so no budget can be formed.
    """
    notes: list[str] = []
    components: list[UncertaintyComponent] = []

    daq_relative = hardware.uncertainty.daq_relative
    calibration = hardware.pressure_calibration

    c_p1, c_p2 = pressure_sensitivities(
        point.inlet_pressure_atm, point.outlet_pressure_atm
    )

    # -- flow rate --------------------------------------------------------
    flow = hardware.flowmeter
    flow_in_unit = units.flow_from_cm3_s(point.flow_cm3_s, flow.unit)
    flow_full_scale = abs(flow.value_max - flow.value_min)
    u_flow = flow.uncertainty.standard_uncertainty(flow_in_unit, flow_full_scale)
    flow_relative = (
        math.hypot(u_flow / abs(flow_in_unit), daq_relative) if flow_in_unit else math.inf
    )
    components.append(
        _component(
            name="flow rate",
            symbol="Q",
            evaluation_type="B",
            value=point.flow_cm3_s,
            unit="cm3/s",
            relative_uncertainty=flow_relative,
            sensitivity=1.0,
            dof=flow.uncertainty.dof,
            source=flow.uncertainty.source or "flowmeter specification",
        )
    )

    # -- inlet pressure ---------------------------------------------------
    u_p1_rel, p1_dof = _pressure_relative_uncertainty(
        calibration.inlet, point.inlet_pressure_atm, run, daq_relative
    )
    components.append(
        _component(
            name="inlet pressure",
            symbol="P1",
            evaluation_type="B",
            value=point.inlet_pressure_atm,
            unit="atm",
            relative_uncertainty=u_p1_rel,
            sensitivity=c_p1,
            dof=p1_dof,
            source=calibration.inlet.uncertainty.source or "inlet transducer",
        )
    )

    # -- outlet pressure --------------------------------------------------
    u_p2_rel, p2_dof = _pressure_relative_uncertainty(
        calibration.outlet, point.outlet_pressure_atm, run, daq_relative
    )
    components.append(
        _component(
            name="outlet pressure",
            symbol="P2",
            evaluation_type="B",
            value=point.outlet_pressure_atm,
            unit="atm",
            relative_uncertainty=u_p2_rel,
            sensitivity=c_p2,
            dof=p2_dof,
            source=calibration.outlet.uncertainty.source or "outlet transducer",
        )
    )

    # -- reference pressure the flow is paired with -----------------------
    if flow.reading_basis == "standard":
        # A defined standard state, not a measurement.
        notes.append(
            "The flow reference pressure is the meter's defined standard state, so it "
            "contributes no measurement uncertainty."
        )
    else:
        source_channel = (
            calibration.inlet
            if flow.actual_pressure_source == "inlet"
            else calibration.outlet
        )
        u_ref_rel, ref_dof = _pressure_relative_uncertainty(
            source_channel, point.reference_pressure_atm, run, daq_relative
        )
        components.append(
            _component(
                name="flow reference pressure",
                symbol="P_ref",
                evaluation_type="B",
                value=point.reference_pressure_atm,
                unit="atm",
                relative_uncertainty=u_ref_rel,
                sensitivity=1.0,
                dof=ref_dof,
                source=f"{flow.actual_pressure_source} transducer",
            )
        )
        notes.append(
            "flowmeter.reading_basis is 'actual', so the flow reference pressure is a "
            "transducer reading and is correlated with that channel; the budget treats "
            "it as independent, which errs high."
        )

    # -- viscosity model --------------------------------------------------
    components.append(
        _component(
            name="viscosity model",
            symbol="mu",
            evaluation_type="B",
            value=point.viscosity_cp,
            unit="cP",
            relative_uncertainty=run.gas.viscosity_relative_uncertainty,
            sensitivity=1.0,
            dof=math.inf,
            source=f"{run.gas.properties_source} transport correlation",
        )
    )

    # -- temperature, through the viscosity -------------------------------
    if viscosity_temperature_exponent:
        temperature_k = units.celsius_to_kelvin(point.temperature_c)
        u_temperature = hardware.temperature.uncertainty.standard_uncertainty(
            point.temperature_c, abs(point.temperature_c) or 1.0
        )
        components.append(
            _component(
                name="temperature",
                symbol="T",
                evaluation_type="B",
                value=temperature_k,
                unit="K",
                relative_uncertainty=u_temperature / temperature_k,
                sensitivity=viscosity_temperature_exponent,
                dof=hardware.temperature.uncertainty.dof,
                source="probe specification, via d(ln mu)/d(ln T)",
            )
        )

    # -- geometry ---------------------------------------------------------
    components.append(
        _component(
            name="sample length",
            symbol="L",
            evaluation_type="B",
            value=geometry.length_cm,
            unit="cm",
            relative_uncertainty=geometry.relative_length_uncertainty,
            sensitivity=1.0,
            dof=math.inf,
            source="caliper",
        )
    )
    components.append(
        _component(
            name="sample diameter",
            symbol="d",
            evaluation_type="B",
            value=geometry.diameter_cm,
            unit="cm",
            relative_uncertainty=geometry.diameter_uncertainty_cm / geometry.diameter_cm,
            sensitivity=-2.0,
            dof=math.inf,
            source="caliper, area goes as d^2",
        )
    )

    # -- bench repeatability ----------------------------------------------
    if hardware.uncertainty.repeatability_relative > 0.0:
        components.append(
            _component(
                name="bench repeatability",
                symbol="rep",
                evaluation_type="B",
                value=point.permeability_darcy,
                unit="D",
                relative_uncertainty=hardware.uncertainty.repeatability_relative,
                sensitivity=1.0,
                dof=math.inf,
                source=hardware.uncertainty.notes or "estimated from repeat loadings",
            )
        )

    # -- Type A: the run's own scatter ------------------------------------
    if type_a_relative is not None and run.uncertainty.include_type_a:
        components.append(
            _component(
                name="steady-state scatter",
                symbol="s/sqrt(n)",
                evaluation_type="A",
                value=point.permeability_darcy,
                unit="D",
                relative_uncertainty=type_a_relative,
                sensitivity=1.0,
                dof=type_a_dof,
                source="standard deviation of the mean over the steady-state window",
            )
        )

    # -- combine ----------------------------------------------------------
    variance = sum(component.variance_share for component in components)

    correlation_variance = 0.0
    if calibration.correlation != 0.0:
        correlation_variance = (
            2.0 * c_p1 * c_p2 * calibration.correlation * u_p1_rel * u_p2_rel
        )
        variance += correlation_variance
        notes.append(
            f"Included the P1/P2 covariance term at r = {calibration.correlation:g}; "
            "because the two pressures enter with opposite signs this "
            f"{'reduces' if correlation_variance < 0 else 'increases'} the combined "
            "uncertainty."
        )
        if variance <= 0.0:
            variance = sum(component.variance_share for component in components)
            correlation_variance = 0.0
            notes.append(
                "The covariance term drove the combined variance non-positive, which "
                "means the stated correlation is too strong to be physical here; it was "
                "dropped and the uncorrelated result is reported instead."
            )

    relative_combined = math.sqrt(variance)
    combined = relative_combined * abs(point.permeability_darcy)

    # -- Welch-Satterthwaite ----------------------------------------------
    denominator = 0.0
    for component in components:
        if math.isfinite(component.degrees_of_freedom) and component.degrees_of_freedom > 0:
            denominator += component.variance_share**2 / component.degrees_of_freedom
    effective_dof = math.inf if denominator == 0.0 else variance**2 / denominator
    if effective_dof > _EFFECTIVE_DOF_INFINITE:
        # A finite-dof term with a negligible variance share produces an
        # astronomically large v_eff. Anything past this is indistinguishable
        # from infinity for the coverage factor, and reporting the raw number
        # implies a precision that is not there.
        effective_dof = math.inf

    factor = (
        run.uncertainty.fixed_coverage_factor
        if run.uncertainty.fixed_coverage_factor is not None
        else coverage_factor(effective_dof, run.uncertainty.coverage_probability)
    )

    return UncertaintyBudget(
        value_darcy=point.permeability_darcy,
        combined_standard_uncertainty_darcy=combined,
        relative_combined_standard_uncertainty=relative_combined,
        effective_degrees_of_freedom=effective_dof,
        coverage_factor=factor,
        coverage_probability=run.uncertainty.coverage_probability,
        expanded_uncertainty_darcy=factor * combined,
        components=components,
        correlation_relative_variance=correlation_variance,
        notes=notes,
    )
