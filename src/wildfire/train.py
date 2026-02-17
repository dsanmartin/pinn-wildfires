from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .pde import PDE, PDEConfig
from .lsm import LSMConfig
from .pinn import MLPConfig, PINN
from .utils import Domain, TrainConfig


@dataclass
class ExperimentConfig:
    domain: Domain = field(default_factory=Domain)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: MLPConfig = field(default_factory=MLPConfig)
    pde: PDEConfig = field(default_factory=PDEConfig)


def loss_terms(model: PINN, pde: PDE, domain: Domain, cfg: TrainConfig, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    model_type = pde.config.model_type.lower()
    if model_type == "mass_conservation":
        coords_interior = domain.sample_interior_xyz(cfg.n_interior, device)
        n_face = max(1, cfg.n_boundary // 2)
        n_topo = max(0, cfg.n_boundary - n_face)
        coords_boundary_faces = domain.sample_boundary_xyz(n_face, device)
        if n_topo > 0:
            x = torch.rand(n_topo, 1, device=device) * (domain.x_max - domain.x_min) + domain.x_min
            y = torch.rand(n_topo, 1, device=device) * (domain.y_max - domain.y_min) + domain.y_min
            z = pde.model.topography(x, y)
            coords_topo = torch.cat([x, y, z], dim=1)
            coords_boundary = torch.cat([coords_boundary_faces, coords_topo], dim=0)
        else:
            coords_topo = None
            coords_boundary = coords_boundary_faces
        coords_initial = domain.sample_initial_xyz(cfg.n_initial, device)
    else:
        coords_interior = domain.sample_interior(cfg.n_interior, device)
        coords_boundary = domain.sample_boundary(cfg.n_boundary, device)
        coords_initial = domain.sample_initial(cfg.n_initial, device)

    res = pde.residual(coords_interior, model)
    loss_pde = torch.mean(res**2)

    bc_type = pde.model.config.bc_type.lower().strip() if hasattr(pde.model.config, "bc_type") else "dirichlet"
    if bc_type == "neumann":
        if model_type == "mass_conservation":
            coords_boundary_faces.requires_grad_(True)
            phi_b = model(coords_boundary_faces)
            n_face = coords_boundary_faces.shape[0] // 6
            nx = torch.cat(
                [
                    torch.full((n_face, 1), -1.0, device=device),
                    torch.full((n_face, 1), 1.0, device=device),
                    torch.zeros((n_face, 1), device=device),
                    torch.zeros((n_face, 1), device=device),
                    torch.zeros((n_face, 1), device=device),
                    torch.zeros((n_face, 1), device=device),
                ],
                dim=0,
            )
            ny = torch.cat(
                [
                    torch.zeros((n_face, 1), device=device),
                    torch.zeros((n_face, 1), device=device),
                    torch.full((n_face, 1), -1.0, device=device),
                    torch.full((n_face, 1), 1.0, device=device),
                    torch.zeros((n_face, 1), device=device),
                    torch.zeros((n_face, 1), device=device),
                ],
                dim=0,
            )
            nz = torch.cat(
                [
                    torch.zeros((n_face, 1), device=device),
                    torch.zeros((n_face, 1), device=device),
                    torch.zeros((n_face, 1), device=device),
                    torch.zeros((n_face, 1), device=device),
                    torch.full((n_face, 1), -1.0, device=device),
                    torch.full((n_face, 1), 1.0, device=device),
                ],
                dim=0,
            )
            normals = torch.cat([nx, ny, nz], dim=1)
        else:
            coords_boundary.requires_grad_(True)
            phi_b = model(coords_boundary)
            n_side = coords_boundary.shape[0] // 4
            nx = torch.cat(
                [
                    torch.full((n_side, 1), -1.0, device=device),
                    torch.full((n_side, 1), 1.0, device=device),
                    torch.zeros((n_side, 1), device=device),
                    torch.zeros((n_side, 1), device=device),
                ],
                dim=0,
            )
            ny = torch.cat(
                [
                    torch.zeros((n_side, 1), device=device),
                    torch.zeros((n_side, 1), device=device),
                    torch.full((n_side, 1), -1.0, device=device),
                    torch.full((n_side, 1), 1.0, device=device),
                ],
                dim=0,
            )
            nt = torch.zeros_like(nx)
            normals = torch.cat([nx, ny, nt], dim=1)

        grads = []
        for c in range(phi_b.shape[1]):
            grad_c = torch.autograd.grad(
                phi_b[:, c : c + 1],
                coords_boundary_faces if model_type == "mass_conservation" else coords_boundary,
                grad_outputs=torch.ones_like(phi_b[:, c : c + 1]),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            grads.append(grad_c)
        grads_b = torch.stack(grads, dim=1)
        dn = torch.sum(grads_b * normals[:, None, :], dim=2)
        loss_bc = torch.mean(dn**2)
        if model_type == "mass_conservation" and coords_topo is not None:
            phi_topo = model(coords_topo)
            loss_bc = loss_bc + torch.mean(phi_topo**2)
    else:
        phi_b = model(coords_boundary)
        target_b = pde.boundary_condition(coords_boundary, model)
        loss_bc = torch.mean((phi_b - target_b) ** 2)

    phi_i = model(coords_initial)
    target_i = pde.initial_condition(coords_initial)
    loss_ic = torch.mean((phi_i - target_i) ** 2)
    
    # Separate loss tracking for Asensio model (temperature and fuel)
    metrics = {
        "loss": 0.0,
        "loss_pde": float(loss_pde.detach().cpu()),
        "loss_bc": float(loss_bc.detach().cpu()),
        "loss_ic": float(loss_ic.detach().cpu()),
    }
    
    if pde.config.model_type.lower() == "asensio" and phi_i.shape[1] == 2:
        # Asensio model has temperature (col 0) and fuel (col 1)
        T_pred = phi_i[:, 0:1]
        Y_pred = phi_i[:, 1:2]
        T_target = target_i[:, 0:1]
        Y_target = target_i[:, 1:2]
        
        loss_ic_temp = torch.mean((T_pred - T_target) ** 2)
        loss_ic_fuel = torch.mean((Y_pred - Y_target) ** 2)
        
        metrics["loss_ic_temp"] = float(loss_ic_temp.detach().cpu())
        metrics["loss_ic_fuel"] = float(loss_ic_fuel.detach().cpu())
        metrics["T_pred_mean"] = float(T_pred.mean().detach().cpu())
        metrics["T_target_mean"] = float(T_target.mean().detach().cpu())
        metrics["Y_pred_mean"] = float(Y_pred.mean().detach().cpu())
        metrics["Y_target_mean"] = float(Y_target.mean().detach().cpu())

    loss = cfg.weight_pde * loss_pde + cfg.weight_bc * loss_bc + cfg.weight_ic * loss_ic
    metrics["loss"] = float(loss.detach().cpu())
    
    return loss, metrics


def train(cfg: ExperimentConfig, device: torch.device | None = None) -> tuple[PINN, dict[str, list[float]]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PINN(cfg.model).to(device)
    pde = PDE(cfg.pde)
    
    # Initialize loss history tracking
    loss_history = {
        "loss": [],
        "loss_pde": [],
        "loss_bc": [],
        "loss_ic": [],
    }
    
    # Get training parameters for each optimizer
    n_adam = cfg.train.epochs_adam
    n_lbfgs = cfg.train.epochs_lbfgs
    lr_adam = cfg.train.lr_adam
    lr_lbfgs = cfg.train.lr_lbfgs
    
    # Phase 1: Adam optimizer
    if n_adam > 0:
        print("\n=== Adam Training ===")
        optimizer_adam = torch.optim.Adam(model.parameters(), lr=lr_adam)
        scheduler_adam = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_adam,
            mode="min",
            factor=cfg.train.lr_factor,
            patience=cfg.train.lr_patience_adam,
            min_lr=cfg.train.lr_min,
        )
        best_loss_adam = float("inf")
        patience_adam = 0
        
        for epoch in range(1, n_adam + 1):
            optimizer_adam.zero_grad()
            loss, metrics = loss_terms(model, pde, cfg.domain, cfg.train, device)
            loss.backward()
            optimizer_adam.step()
            
            # Track loss at every epoch
            loss_history["loss"].append(metrics["loss"])
            loss_history["loss_pde"].append(metrics["loss_pde"])
            loss_history["loss_bc"].append(metrics["loss_bc"])
            loss_history["loss_ic"].append(metrics["loss_ic"])

            scheduler_adam.step(metrics["loss"])
            if metrics["loss"] < best_loss_adam - cfg.train.early_stop_min_delta:
                best_loss_adam = metrics["loss"]
                patience_adam = 0
            else:
                patience_adam += 1
                if cfg.train.early_stop_patience_adam > 0 and patience_adam >= cfg.train.early_stop_patience_adam:
                    print(f"Early stopping Adam at epoch {epoch:05d}.")
                    break

            if epoch % 500 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:05d} | loss={metrics['loss']:.4e} "
                    f"pde={metrics['loss_pde']:.4e} bc={metrics['loss_bc']:.4e} ic={metrics['loss_ic']:.4e}"
                )
    
    # Phase 2: L-BFGS optimizer
    if n_lbfgs > 0:
        print("\n=== L-BFGS Training ===")
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=lr_lbfgs,
            max_iter=20,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe"
        )
        scheduler_lbfgs = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_lbfgs,
            mode="min",
            factor=cfg.train.lr_factor,
            patience=cfg.train.lr_patience_lbfgs,
            min_lr=cfg.train.lr_min,
        )
        best_loss_lbfgs = float("inf")
        patience_lbfgs = 0
        
        for epoch in range(1, n_lbfgs + 1):
            def closure():
                optimizer_lbfgs.zero_grad()
                loss, _ = loss_terms(model, pde, cfg.domain, cfg.train, device)
                
                # Check for NaN
                if torch.isnan(loss):
                    print(f"Warning: NaN detected in loss at epoch {n_adam + epoch}. Stopping L-BFGS.")
                    return loss
                
                loss.backward()
                
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                return loss
            
            optimizer_lbfgs.step(closure)
            
            # Compute metrics for tracking and display
            _, metrics = loss_terms(model, pde, cfg.domain, cfg.train, device)
            
            # Check for NaN in metrics and stop if detected
            if not torch.isfinite(torch.tensor(metrics["loss"])):
                print(f"Warning: NaN/Inf detected at epoch {n_adam + epoch}. Stopping L-BFGS training.")
                break

            scheduler_lbfgs.step(metrics["loss"])
            if metrics["loss"] < best_loss_lbfgs - cfg.train.early_stop_min_delta:
                best_loss_lbfgs = metrics["loss"]
                patience_lbfgs = 0
            else:
                patience_lbfgs += 1
                if cfg.train.early_stop_patience_lbfgs > 0 and patience_lbfgs >= cfg.train.early_stop_patience_lbfgs:
                    print(f"Early stopping L-BFGS at epoch {n_adam + epoch:05d}.")
                    break
            
            loss_history["loss"].append(metrics["loss"])
            loss_history["loss_pde"].append(metrics["loss_pde"])
            loss_history["loss_bc"].append(metrics["loss_bc"])
            loss_history["loss_ic"].append(metrics["loss_ic"])
            
            if epoch % 100 == 0 or epoch == 1:
                print(
                    f"Epoch {n_adam + epoch:05d} | loss={metrics['loss']:.4e} "
                    f"pde={metrics['loss_pde']:.4e} bc={metrics['loss_bc']:.4e} ic={metrics['loss_ic']:.4e}"
                )

    return model, loss_history


def save_model(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)
