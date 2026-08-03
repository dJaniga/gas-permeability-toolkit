"""Hardware boundary.

**Isolation rule (see CLAUDE.md).** This subpackage is the *only* place
allowed to import ``nidaqmx`` or ``serial``. Everything outside it --
``permeability``, ``klinkenberg``, ``gas_properties``, and the maths inside
``acquisition`` -- works on plain floats, which is what lets the whole physics
test suite run in CI with nothing plugged in.

Within the subpackage the split is deliberate too: :mod:`pressure` and
:mod:`flowmeter` are pure calibration maths with no device imports at all, so
they are directly unit-testable; :mod:`daq` and :mod:`temperature` are the thin
wrappers that actually touch drivers, and are tested against mocks.
"""

from gasperm.hardware.flowmeter import FlowChannel
from gasperm.hardware.pressure import PressureChannel

__all__ = ["FlowChannel", "PressureChannel"]
