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

from pydantic import Field, field_validator, model_validator

from gasperm import units
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
    #: Unit for every dimension below. Calipers read in mm, so that is the
    #: default; the physics converts to cm internally either way.
    dimension_unit: str = "mm"
    #: Plug length, in :attr:`dimension_unit`.
    length: float = Field(default=50.0, gt=0.0)
    #: Plug diameter, in :attr:`dimension_unit`. The default is a 1.5 in plug
    #: (38.1 mm exactly, at 25.4 mm/in).
    diameter: float = Field(default=38.1, gt=0.0)
    #: Standard uncertainty of the length measurement. Default is a typical
    #: digital-caliper figure; measure your own if it matters.
    length_uncertainty: float = Field(default=0.1, ge=0.0)
    #: Standard uncertainty of the diameter. This one is worth getting right:
    #: area goes as d^2, so it enters the budget doubled.
    diameter_uncertainty: float = Field(default=0.1, ge=0.0)

    # -- petrophysics (informational) -------------------------------------
    #: Whether :attr:`porosity` and :attr:`porosity_uncertainty` are written as
    #: a fraction (0.104) or a percentage (10.4). Porosity is dimensionless, so
    #: this only says where the decimal point is -- but a helium pycnometer
    #: reports percentage points while every equation wants the fraction, and
    #: transcribing between them by hand is a silent factor of 100.
    porosity_unit: str = "fraction"
    #: Porosity, in :attr:`porosity_unit`. Accepts the older spelling
    #: ``porosity_fraction`` as an input alias, which always meant a fraction
    #: and still does -- so a config written before this field existed loads
    #: unchanged.
    porosity: float | None = Field(default=None, ge=0.0)
    #: Standard uncertainty of :attr:`porosity`, **in the same unit**. So 0.5
    #: against a porosity in ``%`` is half a percentage point, not half a
    #: percent of the reading. Only enters a budget when the pulse-decay storage
    #: correction is in use, where porosity is an input to the measurement
    #: rather than metadata; ``null`` omits the term with a note saying so.
    porosity_uncertainty: float | None = Field(default=None, ge=0.0)
    #: How porosity was obtained: helium pycnometry, MICP, image analysis...
    porosity_method: str = ""
    grain_density_g_cm3: float | None = Field(default=None, gt=0.0)
    bulk_density_g_cm3: float | None = Field(default=None, gt=0.0)

    # -- provenance -------------------------------------------------------
    prepared_by: str = ""
    prepared_on: date | None = None
    notes: str = ""

    @model_validator(mode="before")
    @classmethod
    def _accept_the_older_porosity_spelling(cls, data):
        """Read ``porosity_fraction`` as ``porosity``, so old files still load.

        A ``before`` validator rather than a field alias because an alias only
        consumes its key when the canonical one is *absent*: a template dict
        carrying ``porosity: None`` alongside a supplied ``porosity_fraction``
        would leave the latter over as a forbidden extra. This handles every
        caller the same way -- a stored sidecar, a ``--set`` override, a
        hand-written YAML.
        """
        if not isinstance(data, dict) or "porosity_fraction" not in data:
            return data
        data = dict(data)
        legacy = data.pop("porosity_fraction")
        if legacy is None:
            return data
        if data.get("porosity") is not None and data["porosity"] != legacy:
            raise ValueError(
                f"porosity ({data['porosity']}) and porosity_fraction ({legacy}) "
                "are both set and disagree. porosity_fraction is the older "
                "spelling of the same field; keep one."
            )
        unit = data.get("porosity_unit", "fraction")
        if units.normalize_porosity_unit(unit) != "fraction":
            raise ValueError(
                f"porosity_fraction is set alongside porosity_unit: {unit!r}. The "
                "older spelling always meant a fraction, so the pair is "
                "ambiguous -- rename it to 'porosity', which is read in "
                "porosity_unit."
            )
        data["porosity"] = legacy
        return data

    @field_validator("dimension_unit")
    @classmethod
    def _check_dimension_unit(cls, value: str) -> str:
        units.length_to_cm(1.0, value)  # raises ValueError on an unknown unit
        return value

    @field_validator("porosity_unit")
    @classmethod
    def _check_porosity_unit(cls, value: str) -> str:
        return units.normalize_porosity_unit(value)

    @model_validator(mode="after")
    def _porosity_is_a_real_porosity(self) -> SampleConfig:
        """Range-check the **converted** value, and catch a mislabelled unit.

        The check has to happen after conversion, because 10.4 is out of range
        as a fraction and perfectly ordinary as a percentage. That also makes it
        the one place able to catch the mistake that matters: a percentage left
        labelled ``fraction``, which would otherwise put a porosity of 1040 %
        into the storage correction and read the permeability wildly low.
        """
        for name, value in (
            ("porosity", self.porosity),
            ("porosity_uncertainty", self.porosity_uncertainty),
        ):
            if value is None:
                continue
            as_fraction = units.porosity_to_fraction(value, self.porosity_unit)
            if as_fraction > 1.0:
                hint = (
                    f" Did you mean porosity_unit: '%'? {value:g} is not a "
                    "fraction."
                    if self.porosity_unit == "fraction"
                    else ""
                )
                raise ValueError(
                    f"{name} is {value:g} {self.porosity_unit}, i.e. "
                    f"{as_fraction:.4g} as a fraction, which is more than the "
                    f"whole rock.{hint}"
                )
        return self

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
    def porosity_fraction(self) -> float | None:
        """Porosity as a fraction, whatever unit it was entered in.

        The canonical form, and the only one anything downstream sees -- the
        same boundary :attr:`length_cm` draws for the dimensions.
        """
        if self.porosity is None:
            return None
        return units.porosity_to_fraction(self.porosity, self.porosity_unit)

    @property
    def porosity_uncertainty_fraction(self) -> float | None:
        """``u(phi)`` as a fraction, whatever unit it was entered in."""
        if self.porosity_uncertainty is None:
            return None
        return units.porosity_to_fraction(
            self.porosity_uncertainty, self.porosity_unit
        )

    @property
    def porosity_from_densities(self) -> float | None:
        """Porosity implied by the densities, ``1 - rho_bulk / rho_grain``.

        A cross-check on ``porosity_fraction``, not a replacement: returns
        ``None`` unless both densities are given.
        """
        if self.grain_density_g_cm3 is None or self.bulk_density_g_cm3 is None:
            return None
        return 1.0 - self.bulk_density_g_cm3 / self.grain_density_g_cm3

    def _to_cm(self, value: float) -> float:
        """One dimension in the configured unit -> cm, the internal unit."""
        return units.length_to_cm(value, self.dimension_unit)

    @property
    def length_cm(self) -> float:
        """Plug length in cm, whatever unit it was entered in."""
        return self._to_cm(self.length)

    @property
    def diameter_cm(self) -> float:
        """Plug diameter in cm, whatever unit it was entered in."""
        return self._to_cm(self.diameter)

    def geometry(self) -> SampleGeometry:
        """Convert to the hardware-free geometry model used by the physics.

        This is the one boundary where the configured dimension unit is left
        behind: everything downstream works in cm, because that is what the
        Darcy equation was derived in.
        """
        return SampleGeometry(
            sample_id=self.id,
            description=self.description,
            length_cm=self.length_cm,
            diameter_cm=self.diameter_cm,
            porosity_fraction=self.porosity_fraction,
            length_uncertainty_cm=self._to_cm(self.length_uncertainty),
            diameter_uncertainty_cm=self._to_cm(self.diameter_uncertainty),
        )
