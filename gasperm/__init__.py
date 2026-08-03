"""gasperm -- gas permeability measurement for core samples.

A hardware-in-the-loop lab tool for an NI USB-6421 DAQ (inlet/outlet pressure
and gas flow) plus an Arduino temperature probe on USB serial.

Three commands:

``gasperm init``
    Write a YAML config describing the rig, its calibrations and the sample.
``gasperm collect``
    Sample in real time, compute apparent gas permeability live, stream to the
    console and a timestamped run directory.
``gasperm klinkenberg``
    Regress two or more runs at different mean pressures to recover the
    liquid-equivalent permeability ``k_L`` and the slippage factor ``b``.

All internal physics is done in CGS-Darcy units (atm, cm, cP, cm^3/s); see
:mod:`gasperm.units`, which owns every conversion in the package.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
