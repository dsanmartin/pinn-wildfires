from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .lsm import LSMConfig, LSM
from .asensio import AsensioConfig, Asensio
from .mass_conservation import MassConservation, MassConservationConfig


@dataclass
class PDEConfig:
    model_type: str = "lsm"  # "lsm", "asensio", or "mass_conservation"
    lsm_config: LSMConfig | None = field(default_factory=LSMConfig)
    asensio_config: AsensioConfig | None = None
    mass_conservation_config: MassConservationConfig | None = None


class PDE:
    """
    Wrapper for PDE models (LSM or Asensio).
    Delegates to the appropriate model implementation.
    """

    def __init__(self, config: PDEConfig | None = None):
        self.config = config or PDEConfig()

        model_type = self.config.model_type.lower()
        if model_type == "asensio":
            self.model = Asensio(self.config.asensio_config or AsensioConfig())
        elif model_type == "mass_conservation":
            self.model = MassConservation(self.config.mass_conservation_config or MassConservationConfig())
        else:  # default to LSM
            self.model = LSM(self.config.lsm_config or LSMConfig())

    def residual(self, xyt: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        return self.model.residual(xyt, phi)

    def initial_condition(self, xyt: torch.Tensor) -> torch.Tensor:
        return self.model.initial_condition(xyt)

    def boundary_condition(self, xyt: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        return self.model.boundary_condition(xyt, phi)
