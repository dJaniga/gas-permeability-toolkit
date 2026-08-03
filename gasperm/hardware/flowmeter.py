"""Flowmeter calibration: volts -> cm^3/s, plus the reference-state pairing.

Pure maths, no device imports.

Only **one** flowmeter is active per run. Which analog input it sits on
(``ai2`` or ``ai3``) and what full-scale flow it reports are config-time
decisions; nothing here autodetects anything, and the inactive channel is
never added to the DAQ task.

The subtle part is not the volts-to-flow line -- it is *what state the
reported volume is at*. See :meth:`FlowChannel.reference_pressure_atm`.
"""

from __future__ import annotations

from dataclasses import dataclass

from gasperm import units
from gasperm.config import FlowmeterConfig

__all__ = ["FlowChannel"]


@dataclass(frozen=True)
class FlowChannel:
    """The single active gas flowmeter."""

    config: FlowmeterConfig

    @property
    def channel(self) -> str:
        """Bare DAQ channel name, ``ai2`` or ``ai3``."""
        return self.config.channel

    @property
    def unit(self) -> str:
        """The configured flow unit (``sccm``, ``slpm``, ...)."""
        return self.config.unit

    @property
    def voltage_range(self) -> tuple[float, float]:
        """``(min_val, max_val)`` in volts -- typically 0-10 V, unlike the
        0-5 V pressure channels. This is why each channel is added to the DAQ
        task individually instead of in one shared-range call."""
        low, high = self.config.volts_min, self.config.volts_max
        return (low, high) if low <= high else (high, low)

    def volts_to_flow(self, volts: float) -> float:
        """Raw volts -> flow rate in the meter's **configured unit**."""
        return self.config.apply(volts)

    def volts_to_cm3_s(self, volts: float) -> float:
        """Raw volts -> flow rate in cm^3/s, still at the meter's own state."""
        return units.flow_to_cm3_s(self.volts_to_flow(volts), self.config.unit)

    def cm3_s_to_volts(self, flow_cm3_s: float) -> float:
        """Inverse of :meth:`volts_to_cm3_s`. Used to build test fixtures."""
        return self.config.invert(units.flow_from_cm3_s(flow_cm3_s, self.config.unit))

    def reference_pressure_atm(
        self, *, inlet_pressure_atm: float, outlet_pressure_atm: float, atmospheric_atm: float
    ) -> float:
        """The pressure that the flow reading must be **paired with**.

        The compressible Darcy equation uses the product ``Q_ref * P_ref``,
        which is proportional to molar flow. So the reading is only correct
        when paired with the pressure of the state it was reported at:

        * ``reading_basis == "standard"`` -- a mass-based (thermal) meter
          reports volume at its own standard state, so ``P_ref`` is the
          meter's ``standard_pressure``, *not* atmospheric and *not* the line
          pressure. On a meter standardised at 1 atm these coincide; on one
          standardised at 14.696 psia at 70 degF they do not.
        * ``reading_basis == "actual"`` -- the meter reports volume at the line
          conditions where it physically sits, so ``P_ref`` is the pressure at
          that point: downstream of the core (``outlet``, the usual placement),
          upstream (``inlet``), or vented to ambient (``atmospheric``).

        Getting this wrong scales the permeability by the pressure ratio and
        raises nothing -- hence the explicit config field.

        Args:
            inlet_pressure_atm: Absolute P1.
            outlet_pressure_atm: Absolute P2 as measured.
            atmospheric_atm: The configured atmospheric reference.

        Returns:
            The reference pressure to pass to
            :func:`gasperm.permeability.compute_gas_permeability`, atm.
        """
        if self.config.reading_basis == "standard":
            return self.config.standard_pressure_atm
        source = self.config.actual_pressure_source
        if source == "outlet":
            return outlet_pressure_atm
        if source == "inlet":
            return inlet_pressure_atm
        return atmospheric_atm

    def reference_temperature_c(self, measured_temperature_c: float) -> float:
        """The temperature the reported volume is at.

        For a standard-basis meter that is the meter's own standard
        temperature; for an actual-basis meter it is the measured line
        temperature. Reported for the run log -- the Darcy equation itself is
        isothermal in the reference state and needs only ``Q_ref * P_ref``.
        """
        if self.config.reading_basis == "standard":
            return self.config.standard_temperature_c
        return measured_temperature_c

    @classmethod
    def from_config(cls, config: FlowmeterConfig) -> FlowChannel:
        """Build the active flow channel from its config section."""
        return cls(config=config)
