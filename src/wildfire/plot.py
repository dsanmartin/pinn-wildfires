from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from .pde import PDE, PDEConfig
from .utils import Domain


@dataclass
class PlotConfig:
    resolution: int = 200
    cmap: str = "coolwarm"


def _grid(domain: Domain, resolution: int, t: float, device: torch.device) -> torch.Tensor:
    xs = torch.linspace(domain.x_min, domain.x_max, resolution, device=device)
    ys = torch.linspace(domain.y_min, domain.y_max, resolution, device=device)
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    tt = torch.full_like(xx, t)
    xyt = torch.stack([xx, yy, tt], dim=-1).reshape(-1, 3)
    return xyt


def _grid_xz_at_y(domain: Domain, resolution: int, y_value: float, device: torch.device) -> torch.Tensor:
    xs = torch.linspace(domain.x_min, domain.x_max, resolution, device=device)
    zs = torch.linspace(domain.z_min, domain.z_max, resolution, device=device)
    xx, zz = torch.meshgrid(xs, zs, indexing="xy")
    yy = torch.full_like(xx, y_value)
    xyz = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    return xyz


def _grid_yz_at_x(domain: Domain, resolution: int, x_value: float, device: torch.device) -> torch.Tensor:
    ys = torch.linspace(domain.y_min, domain.y_max, resolution, device=device)
    zs = torch.linspace(domain.z_min, domain.z_max, resolution, device=device)
    yy, zz = torch.meshgrid(ys, zs, indexing="xy")
    xx = torch.full_like(yy, x_value)
    xyz = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    return xyz


def _plot_field(
    field: np.ndarray,
    domain: Domain,
    title: str,
    path: Path | None,
    cfg: PlotConfig,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        field,
        origin="lower",
        extent=[domain.x_min, domain.x_max, domain.y_min, domain.y_max],
        cmap=cfg.cmap,
        aspect="auto",
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def _plot_zero_level_set(
    field: np.ndarray,
    domain: Domain,
    title: str,
    path: Path | None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.linspace(domain.x_min, domain.x_max, field.shape[1])
    y = np.linspace(domain.y_min, domain.y_max, field.shape[0])
    xx, yy = np.meshgrid(x, y, indexing="xy")
    ax.contour(xx, yy, field, levels=[0.0], colors="black", linewidths=2.0)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(title)
    ax.set_xlim(domain.x_min, domain.x_max)
    ax.set_ylim(domain.y_min, domain.y_max)
    ax.set_aspect("equal", adjustable="box")

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_initial_condition(
    domain: Domain,
    path: str | Path | None = None,
    cfg: PlotConfig | None = None,
    lsm: PDEConfig | None = None,
) -> None:
    cfg = cfg or PlotConfig()
    lsm = lsm or PDEConfig()
    pde = PDE(lsm)
    device = torch.device("cpu")
    xyt = _grid(domain, cfg.resolution, domain.t_min, device)
    phi = pde.initial_condition(xyt).reshape(cfg.resolution, cfg.resolution)
    field = phi.detach().cpu().numpy()
    _plot_field(field, domain, "Initial condition", Path(path) if path else None, cfg)


def plot_final_result(
    model: torch.nn.Module,
    domain: Domain,
    path: str | Path | None = None,
    cfg: PlotConfig | None = None,
) -> None:
    cfg = cfg or PlotConfig()
    device = next(model.parameters()).device
    xyt = _grid(domain, cfg.resolution, domain.t_max, device)
    with torch.no_grad():
        phi = model(xyt).reshape(cfg.resolution, cfg.resolution)
    field = phi.detach().cpu().numpy()
    _plot_field(field, domain, "Final result", Path(path) if path else None, cfg)


def plot_initial_perimeter(
    domain: Domain,
    path: str | Path | None = None,
    cfg: PlotConfig | None = None,
    lsm: PDEConfig | None = None,
) -> None:
    cfg = cfg or PlotConfig()
    lsm = lsm or PDEConfig()
    pde = PDE(lsm)
    device = torch.device("cpu")
    xyt = _grid(domain, cfg.resolution, domain.t_min, device)
    phi = pde.initial_condition(xyt).reshape(cfg.resolution, cfg.resolution)
    field = phi.detach().cpu().numpy()
    _plot_zero_level_set(field, domain, "Initial perimeter (phi=0)", Path(path) if path else None)


def plot_final_perimeter(
    model: torch.nn.Module,
    domain: Domain,
    path: str | Path | None = None,
    cfg: PlotConfig | None = None,
) -> None:
    cfg = cfg or PlotConfig()
    device = next(model.parameters()).device
    xyt = _grid(domain, cfg.resolution, domain.t_max, device)
    with torch.no_grad():
        phi = model(xyt).reshape(cfg.resolution, cfg.resolution)
    field = phi.detach().cpu().numpy()
    _plot_zero_level_set(field, domain, "Final perimeter (phi=0)", Path(path) if path else None)


def plot_perimeter_comparison(
    model: torch.nn.Module,
    domain: Domain,
    path: str | Path | None = None,
    cfg: PlotConfig | None = None,
    lsm: PDEConfig | None = None,
) -> None:
    cfg = cfg or PlotConfig()
    lsm = lsm or PDEConfig()
    pde = PDE(lsm)
    device = next(model.parameters()).device

    xyt_init = _grid(domain, cfg.resolution, domain.t_min, torch.device("cpu"))
    phi_init = pde.initial_condition(xyt_init).reshape(cfg.resolution, cfg.resolution)

    xyt_final = _grid(domain, cfg.resolution, domain.t_max, device)
    with torch.no_grad():
        phi_final = model(xyt_final).reshape(cfg.resolution, cfg.resolution)

    field_init = phi_init.detach().cpu().numpy()
    field_final = phi_final.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.linspace(domain.x_min, domain.x_max, field_init.shape[1])
    y = np.linspace(domain.y_min, domain.y_max, field_init.shape[0])
    xx, yy = np.meshgrid(x, y, indexing="xy")
    ax.contour(xx, yy, field_init, levels=[0.0], colors="blue", linewidths=2.0)
    ax.contour(xx, yy, field_final, levels=[0.0], colors="red", linewidths=2.0)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(r"Perimeter comparison ($\phi=0$)")
    ax.set_xlim(domain.x_min, domain.x_max)
    ax.set_ylim(domain.y_min, domain.y_max)
    ax.set_aspect("equal", adjustable="box")
    handles = [
        Line2D([0], [0], color="blue", lw=2, label="Initial"),
        Line2D([0], [0], color="red", lw=2, label="Final"),
    ]
    ax.legend(handles=handles, loc="upper right")

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_phi_comparison(
    model: torch.nn.Module,
    domain: Domain,
    path: str | Path | None = None,
    cfg: PlotConfig | None = None,
    lsm: PDEConfig | None = None,
) -> None:
    cfg = cfg or PlotConfig()
    lsm = lsm or PDEConfig()
    pde = PDE(lsm)
    device = next(model.parameters()).device

    xyt_init = _grid(domain, cfg.resolution, domain.t_min, torch.device("cpu"))
    phi_init = pde.initial_condition(xyt_init).reshape(cfg.resolution, cfg.resolution)

    xyt_final = _grid(domain, cfg.resolution, domain.t_max, device)
    with torch.no_grad():
        phi_final = model(xyt_final).reshape(cfg.resolution, cfg.resolution)

    field_init = phi_init.detach().cpu().numpy()
    field_final = phi_final.detach().cpu().numpy()
    vmin = float(min(field_init.min(), field_final.min()))
    vmax = float(max(field_init.max(), field_final.max()))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    im0 = axes[0].imshow(
        field_init,
        origin="lower",
        extent=[domain.x_min, domain.x_max, domain.y_min, domain.y_max],
        cmap=cfg.cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title(r"$\phi(x,y,0)$")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$y$")

    im1 = axes[1].imshow(
        field_final,
        origin="lower",
        extent=[domain.x_min, domain.x_max, domain.y_min, domain.y_max],
        cmap=cfg.cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title(r"$\phi(x,y,t_{\text{max}})$")
    axes[1].set_xlabel(r"$x$")

    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.85)

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_temperature_comparison(
    model: torch.nn.Module,
    domain: Domain,
    path: str | Path | None = None,
    cfg: PlotConfig | None = None,
    pde_cfg: PDEConfig | None = None,
) -> None:
    """Plot initial and final temperature fields for Asensio model."""
    cfg = cfg or PlotConfig()
    pde_cfg = pde_cfg or PDEConfig()
    pde = PDE(pde_cfg)
    device = next(model.parameters()).device

    xyt_init = _grid(domain, cfg.resolution, domain.t_min, torch.device("cpu"))
    ic = pde.initial_condition(xyt_init)
    T_init = ic[:, 0].reshape(cfg.resolution, cfg.resolution)

    xyt_final = _grid(domain, cfg.resolution, domain.t_max, device)
    with torch.no_grad():
        model_output = model(xyt_final)
        T_final = model_output[:, 0].reshape(cfg.resolution, cfg.resolution)

    field_init = T_init.detach().cpu().numpy()
    field_final = T_final.detach().cpu().numpy()
    vmin = float(min(field_init.min(), field_final.min()))
    vmax = float(max(field_init.max(), field_final.max()))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    im0 = axes[0].imshow(
        field_init,
        origin="lower",
        extent=[domain.x_min, domain.x_max, domain.y_min, domain.y_max],
        cmap=cfg.cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title(r"$T(x,y,0)$ [K]")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$y$")

    im1 = axes[1].imshow(
        field_final,
        origin="lower",
        extent=[domain.x_min, domain.x_max, domain.y_min, domain.y_max],
        cmap=cfg.cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title(r"$T(x,y,t_{\text{max}})$ [K]")
    axes[1].set_xlabel(r"$x$")

    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.85)

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_fuel_fraction_comparison(
    model: torch.nn.Module,
    domain: Domain,
    path: str | Path | None = None,
    cfg: PlotConfig | None = None,
    pde_cfg: PDEConfig | None = None,
) -> None:
    """Plot initial and final fuel fraction fields for Asensio model."""
    cfg = cfg or PlotConfig()
    pde_cfg = pde_cfg or PDEConfig()
    pde = PDE(pde_cfg)
    device = next(model.parameters()).device

    # For Asensio, get initial fuel fraction from initial condition
    xyt_init = _grid(domain, cfg.resolution, domain.t_min, torch.device("cpu"))
    ic = pde.initial_condition(xyt_init)
    Y_init = ic[:, 1].reshape(cfg.resolution, cfg.resolution).detach().cpu().numpy()

    xyt_final = _grid(domain, cfg.resolution, domain.t_max, device)
    with torch.no_grad():
        model_output = model(xyt_final)
        Y_final = model_output[:, 1].reshape(cfg.resolution, cfg.resolution)

    field_init = Y_init
    field_final = Y_final.detach().cpu().numpy()
    vmin = float(min(field_init.min(), field_final.min()))
    vmax = float(max(field_init.max(), field_final.max()))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    im0 = axes[0].imshow(
        field_init,
        origin="lower",
        extent=[domain.x_min, domain.x_max, domain.y_min, domain.y_max],
        cmap=cfg.cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title(r"$Y(x,y,0)$ [Fuel Fraction]")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$y$")

    im1 = axes[1].imshow(
        field_final,
        origin="lower",
        extent=[domain.x_min, domain.x_max, domain.y_min, domain.y_max],
        cmap=cfg.cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title(r"$Y(x,y,t_{\text{max}})$ [Fuel Fraction]")
    axes[1].set_xlabel(r"$x$")

    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.85)

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_mass_conservation_comparison(
    model: torch.nn.Module,
    domain: Domain,
    out_dir: str | Path,
    cfg: PlotConfig | None = None,
    pde_cfg: PDEConfig | None = None,
) -> None:
    cfg = cfg or PlotConfig()
    pde_cfg = pde_cfg or PDEConfig(model_type="mass_conservation")
    pde = PDE(pde_cfg)
    device = next(model.parameters()).device
    out_dir = Path(out_dir)

    y_fixed = domain.y_max - domain.y_min
    x_fixed = domain.x_max - domain.x_min

    def plot_field(field_name: str, field_index: int, plane: str, fixed_value: float) -> None:
        if plane == "xz":
            xyz_init = _grid_xz_at_y(domain, cfg.resolution, fixed_value, torch.device("cpu"))
            xyz_final = _grid_xz_at_y(domain, cfg.resolution, fixed_value, device)
            x_vals = np.linspace(domain.x_min, domain.x_max, cfg.resolution)
            y_vals = np.linspace(domain.z_min, domain.z_max, cfg.resolution)
            xlabel = r"$x$"
            ylabel = r"$z$"
        else:
            xyz_init = _grid_yz_at_x(domain, cfg.resolution, fixed_value, torch.device("cpu"))
            xyz_final = _grid_yz_at_x(domain, cfg.resolution, fixed_value, device)
            x_vals = np.linspace(domain.y_min, domain.y_max, cfg.resolution)
            y_vals = np.linspace(domain.z_min, domain.z_max, cfg.resolution)
            xlabel = r"$y$"
            ylabel = r"$z$"

        ic = pde.initial_condition(xyz_init)
        init_field = ic[:, field_index].reshape(cfg.resolution, cfg.resolution).detach().cpu().numpy()
        with torch.no_grad():
            final_out = model(xyz_final)
            final_field = final_out[:, field_index].reshape(cfg.resolution, cfg.resolution).detach().cpu().numpy()

        vmin = float(min(init_field.min(), final_field.min()))
        vmax = float(max(init_field.max(), final_field.max()))

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
        xx, yy = np.meshgrid(x_vals, y_vals, indexing="xy")

        im0 = axes[0].contourf(xx, yy, init_field, levels=50, cmap=cfg.cmap, vmin=vmin, vmax=vmax)
        axes[0].set_title(f"{field_name}(initial)")
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)

        im1 = axes[1].contourf(xx, yy, final_field, levels=50, cmap=cfg.cmap, vmin=vmin, vmax=vmax)
        axes[1].set_title(f"{field_name}(final)")
        axes[1].set_xlabel(xlabel)

        fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.85)
        out_path = out_dir / f"mass_{field_name.lower()}_comparison.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    plot_field("u", 0, "xz", y_fixed)
    plot_field("v", 1, "yz", x_fixed)
    plot_field("w", 2, "xz", y_fixed)


def plot_training_loss(
    loss_history: dict[str, list[float]],
    path: str | Path | None = None,
) -> None:
    """Plot training loss components: total, PDE, BC, and IC losses.
    
    Args:
        loss_history: Dictionary containing loss history with keys 'loss', 'loss_pde', 'loss_bc', 'loss_ic'
        path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    # Total loss
    axes[0, 0].semilogy(loss_history["loss"], linewidth=2, color="black")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].grid(True, alpha=0.3)
    
    # PDE loss
    axes[0, 1].semilogy(loss_history["loss_pde"], linewidth=2, color="blue")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("PDE Residual Loss")
    axes[0, 1].grid(True, alpha=0.3)
    
    # BC loss
    axes[1, 0].semilogy(loss_history["loss_bc"], linewidth=2, color="green")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_title("Boundary Condition Loss")
    axes[1, 0].grid(True, alpha=0.3)
    
    # IC loss
    axes[1, 1].semilogy(loss_history["loss_ic"], linewidth=2, color="red")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_title("Initial Condition Loss")
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)