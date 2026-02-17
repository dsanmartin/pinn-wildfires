from __future__ import annotations

from dataclasses import dataclass

import torch

from .utils import gradient, split_xyz, split_xyz_t


@dataclass
class MassConservationConfig:
    a1: float = 0.1
    a2: float = 0.1
    l: float = 1.0
    z0: float = 10.0
    U0: float = 1.0
    V0: float = 0.0
    W0: float = 0.0
    bc_type: str = "dirichlet"
    topo_height: float = 5.0
    topo_center: tuple[float, float] = (5, 5)
    topo_sigma_x: float = 2.0
    topo_sigma_y: float = 2.0
    topo_base: float = 0.0


class MassConservation:
    r"""
    PDE definition.
    At the moment, solving the level set equation for wildfire spread

    Level Set equation: phi_t + U \cdot grad(phi) = 0.
    """

    def __init__(self, config: MassConservationConfig | None = None):
        self.config = config or MassConservationConfig()

    def residual(self, xyzt: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        xyzt.requires_grad_(True)
        phi = phi(xyzt)
        u = phi[:, 0:1]
        v = phi[:, 1:2]
        w = phi[:, 2:3]
        ux = gradient(u, xyzt)[:, 0:1]
        vy = gradient(v, xyzt)[:, 1:2]
        wz = gradient(w, xyzt)[:, 2:3]
        u0, v0, w0 = self.initial_condition(xyzt).split(1, dim=1)
        return self.config.a1 ** 2 * (u - u0) ** 2 + self.config.a1 ** 2 * (v - v0) ** 2 + self.config.a2 ** 2 * (w - w0) ** 2 + self.config.l ** 2 * (ux + vy + wz) ** 2

    def topography(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cx, cy = self.config.topo_center
        sx = self.config.topo_sigma_x
        sy = self.config.topo_sigma_y
        height = self.config.topo_height
        base = self.config.topo_base
        return base + height * torch.exp(-(((x - cx) ** 2) / (2 * sx**2) + ((y - cy) ** 2) / (2 * sy**2)))

    def initial_condition(self, xyzt: torch.Tensor) -> torch.Tensor:
        if xyzt.shape[1] == 3:
            x, y, z = split_xyz(xyzt)
        else:
            x, y, z, _ = split_xyz_t(xyzt)
        topo = self.topography(x, y)
        topo_mask = z <= topo
        # Power law initial wind profile
        u0 = self.config.U0 * (z / self.config.z0) ** (1 / 7)
        v0 = self.config.V0 * torch.ones_like(u0)
        w0 = self.config.W0 * torch.ones_like(u0)
        ic = torch.cat([u0, v0, w0], dim=1)
        topo_values = torch.zeros_like(ic)
        return torch.where(topo_mask, topo_values, ic)

    def boundary_condition(self, xyzt: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        if xyzt.shape[1] == 3:
            x, y, z = split_xyz(xyzt)
        else:
            x, y, z, _ = split_xyz_t(xyzt)
        topo = self.topography(x, y)
        topo_mask = z <= topo

        bc_type = self.config.bc_type.lower().strip()
        if bc_type == "periodic":
            # Periodic boundary condition: enforce phi(x_min, y, t) == phi(x_max, y, t)
            # and phi(x, y_min, t) == phi(x, y_max, t).
            if xyzt.shape[1] == 3:
                t = None
            else:
                x, y, z, t = split_xyz_t(xyzt)
            x_min = x.min()
            x_max = x.max()
            y_min = y.min()
            y_max = y.max()
            x_periodic = torch.where(
                torch.isclose(x, x_min),
                x_max,
                torch.where(torch.isclose(x, x_max), x_min, x),
            )
            y_periodic = torch.where(
                torch.isclose(y, y_min),
                y_max,
                torch.where(torch.isclose(y, y_max), y_min, y),
            )
            if t is None:
                xyz_periodic = torch.cat([x_periodic, y_periodic, z], dim=1)
                base_bc = phi(xyz_periodic)
            else:
                xyzt_periodic = torch.cat([x_periodic, y_periodic, z, t], dim=1)
                base_bc = phi(xyzt_periodic)
        elif bc_type == "neumann":
            # Zero Neumann boundary condition is enforced in the loss via normal derivatives.
            # Return zeros here to keep API consistent when called directly.
            base_bc = torch.zeros_like(self.initial_condition(xyzt))
        else:
            # Dirichlet boundary condition: use initial condition values on boundary.
            base_bc = self.initial_condition(xyzt)

        topo_values = torch.zeros_like(base_bc)
        return torch.where(topo_mask, topo_values, base_bc)
