"""Core plug identity, geometry and petrophysical metadata.

File: ``sample.yaml``. Changes when a new plug is loaded, and only then --
confining pressure and working gas live in ``run.yaml`` because the same plug
is routinely measured at several of each.

Only ``length_cm`` and ``diameter_cm`` enter the Darcy calculation. Everything
else is provenance, and is carried into every run's metadata so a number can
be traced back to a rock years later.
"""

from __future__ import annotations

from datetime import date

from pydantic import Field, model_validator

from gasperm.config.common import _Base
from gasperm.models import SampleGeometry

__all__ = ["SampleConfig"]


class SampleConfig(_Base):
    """Identity, geometry and petrophysical description of one core plug."""

    # -- identity ---------------------------------------------------------
    id: str = "core-001"
    description: str = ""
    #: Rock type, e.g. "fine-grained quartz arenite".
    lithology: str = ""
    formation: str = ""
    well: str = ""
    #: Sampled depth, in ``depth_unit``.
    depth: float | None = None
    depth_unit: str = "m"

    # -- geometry (the only fields the physics uses) ----------------------
    length_cm: float = Field(default=5.0, gt=0.0)
    diameter_cm: float = Field(default=2.54, gt=0.0)
    #: Standard uncertainty of the length measurement, cm. Default is a
    #: typical digital-caliper figure; measure your own if it matters.
    length_uncertainty_cm: float = Field(default=0.01, ge=0.0)
    #: Standard uncertainty of the diameter, cm. This one is worth getting
    #: right: area goes as d^2, so it enters the budget doubled.
    diameter_uncertainty_cm: float = Field(default=0.01, ge=0.0)

    # -- petrophysics (informational) -------------------------------------
    porosity_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    #: How porosity was obtained: helium pycnometry, MICP, image analysis...
    porosity_method: str = ""
    grain_density_g_cm3: float | None = Field(default=None, gt=0.0)
    bulk_density_g_cm3: float | None = Field(default=None, gt=0.0)

    # -- provenance -------------------------------------------------------
    prepared_by: str = ""
    prepared_on: date | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _densities_are_consistent(self) -> SampleConfig:
        """Bulk density cannot exceed grain density for a porous solid."""
        if (
            self.grain_density_g_cm3 is not None
            and self.bulk_density_g_cm3 is not None
            and self.bulk_density_g_cm3 > self.grain_density_g_cm3
        ):
            raise ValueError(
                f"bulk_density_g_cm3 ({self.bulk_density_g_cm3}) exceeds "
                f"grain_density_g_cm3 ({self.grain_density_g_cm3}), which is impossible "
                "for a porous sample -- check whether the two are swapped"
            )
        return self

    @property
    def porosity_from_densities(self) -> float | None:
        """Porosity implied by the densities, ``1 - rho_bulk / rho_grain``.

        A cross-check on ``porosity_fraction``, not a replacement: returns
        ``None`` unless both densities are given.
        """
        if self.grain_density_g_cm3 is None or self.bulk_density_g_cm3 is None:
            return None
        return 1.0 - self.bulk_density_g_cm3 / self.grain_density_g_cm3

    def geometry(self) -> SampleGeometry:
        """Convert to the hardware-free geometry model used by the physics."""
        return SampleGeometry(
            sample_id=self.id,
            description=self.description,
            length_cm=self.length_cm,
            diameter_cm=self.diameter_cm,
            porosity_fraction=self.porosity_fraction,
            length_uncertainty_cm=self.length_uncertainty_cm,
            diameter_uncertainty_cm=self.diameter_uncertainty_cm,
        )
