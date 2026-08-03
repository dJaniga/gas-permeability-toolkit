"""Pressure transducer calibration: volts -> absolute atm.

Pure maths, no device imports. The calibration constants live in the config,
never here -- baking a specific transducer's scale factor into acquisition
code is exactly the silent-error class this package is built to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass

from gasperm import units
from gasperm.config import PressureChannelConfig

__all__ = ["PressureChannel"]


@dataclass(frozen=True)
class PressureChannel:
    """One calibrated pressure transducer.

    Converts a raw voltage to **absolute atm** in a single step, applying, in
    order: the two-point linear calibration, the gauge-to-absolute offset when
    the transducer is a gauge type, and the config-unit-to-atm conversion.

    Attributes:
        name: Human label used in errors and CSV headers ("inlet"/"outlet").
        channel: Bare DAQ channel name, e.g. ``ai0``.
        config: The calibration and unit for this channel.
        atmospheric_pressure_atm: Added to gauge readings to make them
            absolute. The Darcy equation needs absolute P1/P2.
    """

    name: str
    channel: str
    config: PressureChannelConfig
    atmospheric_pressure_atm: float

    @property
    def voltage_range(self) -> tuple[float, float]:
        """``(min_val, max_val)`` for this channel's DAQ task entry, volts.

        Ordered ascending regardless of how the calibration was written, since
        NI-DAQmx rejects an inverted range.
        """
        low, high = self.config.volts_min, self.config.volts_max
        return (low, high) if low <= high else (high, low)

    @property
    def unit(self) -> str:
        """The configured pressure unit for this channel."""
        return self.config.unit

    @property
    def is_gauge(self) -> bool:
        """True when the transducer reads gauge rather than absolute pressure."""
        return self.config.reading_type == "gauge"

    def volts_to_pressure(self, volts: float) -> float:
        """Raw volts -> pressure in this channel's **configured unit**.

        Still gauge if the transducer is a gauge type -- see
        :meth:`volts_to_absolute_atm` for the value the physics wants.
        """
        return self.config.apply(volts)

    def volts_to_absolute_atm(self, volts: float) -> float:
        """Raw volts -> **absolute** pressure in atm, ready for the physics.

        The gauge-to-absolute step is explicit and unconditional on the
        ``reading_type`` flag: a gauge transducer reading 0 V means "ambient",
        not "vacuum", and treating it as absolute understates P1^2 - P2^2 and
        overstates permeability without raising anything.
        """
        pressure_in_config_unit = self.volts_to_pressure(volts)
        pressure_atm = units.to_atm(pressure_in_config_unit, self.config.unit)
        if self.is_gauge:
            pressure_atm += self.atmospheric_pressure_atm
        return pressure_atm

    def absolute_atm_to_volts(self, pressure_atm: float) -> float:
        """Inverse of :meth:`volts_to_absolute_atm`. Used to build test fixtures."""
        if self.is_gauge:
            pressure_atm = pressure_atm - self.atmospheric_pressure_atm
        return self.config.invert(units.from_atm(pressure_atm, self.config.unit))

    @classmethod
    def from_config(
        cls,
        name: str,
        channel: str,
        config: PressureChannelConfig,
        atmospheric_pressure_atm: float,
    ) -> PressureChannel:
        """Build a channel from its config section."""
        return cls(
            name=name,
            channel=channel,
            config=config,
            atmospheric_pressure_atm=atmospheric_pressure_atm,
        )
