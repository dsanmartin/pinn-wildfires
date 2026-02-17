from __future__ import annotations

from dataclasses import dataclass

import torch

from .utils import gradient, split_xyz_t


# Physical constants
STEFAN_BOLTZMANN = 5.67e-8  # Stefan-Boltzmann constant [W/(m^2 K^4)]


@dataclass
class AsensioConfigBK:
    T0: float = 300.0 # Ambient temperature
    T_inf: float = 1200.0 # Peak temperature
    center: tuple[float, float] = (0.5, 0.5) # Center of initial heat source
    sx: float = 0.5 # Gaussian width in x-direction
    sy: float = 0.5 # Gaussian width in y-direction
    vx: float = 0.0 # Wind velocity in x-direction
    vy: float = 0.0 # Wind velocity in y-direction
    k: float = 0.02476  # Thermal conductivity
    T_act: float = 18000.0  # Activation temperature
    Y_f: float = 0.1    # Fuel consumption rate
    delta: float = 0.1  # Thickness of the combustible layer
    T_ign: float = 500.0  # Ignition temperature
    H_C: float = 21e6  # Heat of combustion
    A: float = 1e9  # Pre-exponential factor
    bc_type: str = "periodic"

@dataclass
class AsensioConfig:
    T0: float = 1.0 # Ambient temperature
    T_inf: float = 4 # Peak temperature
    T_ign: float = 2  # Ignition temperature
    T_act: float = 1  # Activation temperature
    center: tuple[float, float] = (5, 5) # Center of initial heat source
    sx: float = 2 # Gaussian width in x-direction
    sy: float = 2 # Gaussian width in y-direction
    vx: float = 0.0 # Wind velocity in x-direction
    vy: float = 0.0 # Wind velocity in y-direction
    k: float = 1  # Thermal conductivity
    E: float = 1  # Activation energy
    R: float = 1    # Universal gas constant
    Y_f: float = 1   # Fuel consumption rate
    delta: float = 1  # Thickness of the combustible layer
    H_C: float = 1  # Heat of combustion
    A: float = 1  # Pre-exponential factor
    bc_type: str = "periodic"


class Asensio:
    r"""
    PDE definition.
    Implement the following simplified wildfire model from Asensio et al. (2020):

    Temperature:
    T_t + U \cdot grad(T) = div((k + 4 * delta * SIGMA * T^3) grad(T)) + S(T, Y)
    Y_t = - Y_f * Y * H(T - T_ign) exp(-E / (R * T)) 
    """

    def __init__(self, config: AsensioConfig | None = None):
        self.config = config or AsensioConfig()

    def residual(self, xyt: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        xyt.requires_grad_(True)
        output = phi(xyt)
        T = output[:, 0:1]
        Y = output[:, 1:2]
        Y = torch.clamp(Y, min=0.0, max=1.0)  # Ensure fuel fraction stays in [0, 1]
        grads_T = gradient(T, xyt)
        grads_Y = gradient(Y, xyt)
        T_t = grads_T[:, 2:3]
        Y_t = grads_Y[:, 2:3]
        grad_T = grads_T[:, 0:2]
        # Use smooth Heaviside approximation: sigmoid with steep slope to approximate step function
        heaviside_approx = torch.sigmoid(100 * (T - self.config.T_ign))
        # Clamp T to avoid extreme values in exponential
        T_safe = torch.clamp(T, min=300.0, max=2000.0)
        reaction_rate = torch.exp(-self.config.T_act / T_safe) * heaviside_approx * self.config.H_C * self.config.A
        # Diffusion term with radiative heating
        diffusion = torch.sum((self.config.k + 4 * self.config.delta * STEFAN_BOLTZMANN * T**3) * grad_T, dim=1, keepdim=True)
        T_new = T_t + self.config.vx * grad_T[:, 0:1] + self.config.vy * grad_T[:, 1:2] - diffusion - self.config.Y_f * Y * reaction_rate
        Y_new = Y_t + self.config.Y_f * Y * reaction_rate
        # Clamp Y_new to ensure fuel stays in [0, 1]
        Y_new = torch.clamp(Y_new, min=0.0, max=1.0)
        return torch.cat([T_new, Y_new], dim=1)

    def initial_condition(self, xyt: torch.Tensor) -> torch.Tensor:
        # Gaussian function as initial condition for T, constant for Y
        x, y, _ = split_xyz_t(xyt)
        cx, cy = self.config.center
        T0 = self.config.T0
        T_inf = self.config.T_inf
        sx = self.config.sx
        sy = self.config.sy
        T_ic = T0 + (T_inf - T0) * torch.exp(-(((x - cx) ** 2) / (2 * sx**2) + ((y - cy) ** 2) / (2 * sy**2)))
        Y_ic = torch.ones_like(T_ic) * 0
        return torch.cat([T_ic, Y_ic], dim=1)

    def boundary_condition(self, xyt: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        bc_type = self.config.bc_type.lower().strip()
        if bc_type == "periodic":
            # Periodic boundary condition: return values from opposite side
            x, y, t = split_xyz_t(xyt)
            x_min = x.min()
            x_max = x.max()
            y_min = y.min()
            y_max = y.max()
            
            # Create periodic coordinates
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
            xyt_periodic = torch.cat([x_periodic, y_periodic, t], dim=1)
            return phi(xyt_periodic)
        elif bc_type == "dirichlet":
            # Dirichlet boundary condition: enforce phi = T0 (ambient) for T and Y=1 at boundaries
            output = phi(xyt)
            T0 = self.config.T0
            T_bc = torch.full_like(output[:, 0:1], T0)
            Y_bc = torch.ones_like(output[:, 1:2])
            return torch.cat([T_bc, Y_bc], dim=1)
        else:
            raise ValueError(f"Unknown boundary condition type: {self.config.bc_type}")