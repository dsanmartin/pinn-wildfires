from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .train import ExperimentConfig


def build_config_log(cfg: ExperimentConfig) -> str:
    """Build configuration log with common sections and model-specific parameters."""
    model_type = cfg.pde.model_type.lower() if hasattr(cfg.pde, "model_type") else "lsm"
    
    # Common header and sections
    log = f"""{'='*60}
Experiment Configuration
{'='*60}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Domain:
  x: [{cfg.domain.x_min}, {cfg.domain.x_max}]
  y: [{cfg.domain.y_min}, {cfg.domain.y_max}]
  z: [{cfg.domain.z_min}, {cfg.domain.z_max}]
  t: [{cfg.domain.t_min}, {cfg.domain.t_max}]

PINN Model:
  layers: {cfg.model.layers}
  activation: {cfg.model.activation}

Training:
  n_interior: {cfg.train.n_interior}
  n_boundary: {cfg.train.n_boundary}
  n_initial: {cfg.train.n_initial}
  lr_adam: {cfg.train.lr_adam}
  lr_lbfgs: {cfg.train.lr_lbfgs}
  epochs_adam: {cfg.train.epochs_adam}
  epochs_lbfgs: {cfg.train.epochs_lbfgs}
  weight_pde: {cfg.train.weight_pde}
  weight_bc: {cfg.train.weight_bc}
  weight_ic: {cfg.train.weight_ic}

"""
    
    # Model-specific parameters
    if model_type == "asensio" and cfg.pde.asensio_config:
        asensio = cfg.pde.asensio_config
        log += f"""Asensio Model Parameters:
  Initial condition:
    center: ({asensio.center[0]}, {asensio.center[1]})
    sx: {asensio.sx}
    sy: {asensio.sy}
  Temperature:
    T0: {asensio.T0}
    T_inf: {asensio.T_inf}
    T_ign: {asensio.T_ign}
    T_act: {asensio.T_act}
  Properties:
    k: {asensio.k}
    Y_f: {asensio.Y_f}
    delta: {asensio.delta}
    H_C: {asensio.H_C}
    A: {asensio.A}
  Boundary conditions: {asensio.bc_type}
  Velocity: ({asensio.vx}, {asensio.vy})
"""
    elif model_type == "mass_conservation" and cfg.pde.mass_conservation_config:
        mass = cfg.pde.mass_conservation_config
        log += f"""Mass Conservation Parameters:
  Coefficients:
    a1: {mass.a1}
    a2: {mass.a2}
    l: {mass.l}
  Reference profile:
    z0: {mass.z0}
    U0: {mass.U0}
    V0: {mass.V0}
    W0: {mass.W0}
  Topography:
    height: {mass.topo_height}
    center: ({mass.topo_center[0]}, {mass.topo_center[1]})
    sigma_x: {mass.topo_sigma_x}
    sigma_y: {mass.topo_sigma_y}
    base: {mass.topo_base}
  Boundary conditions: {mass.bc_type}
"""
    else:
        lsm = cfg.pde.lsm_config if cfg.pde.lsm_config else None
        if lsm:
            log += f"""LSM Model Parameters:
  Initial condition:
    center: ({lsm.center[0]}, {lsm.center[1]})
    radius: {lsm.radius}
  PDE Parameters:
    speed: {lsm.speed}
    epsilon: {lsm.epsilon}
    r0: {lsm.r0}
    cf: {lsm.cf}
  Boundary conditions: {lsm.bc_type}
  Velocity: ({lsm.vx}, {lsm.vy})
"""
    
    log += f"{'='*60}\n"
    return log


def write_config_log(cfg: ExperimentConfig, out_dir: Path) -> Path:
    config_text = build_config_log(cfg)
    print(config_text)
    log_path = out_dir / "config.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(config_text)
    print(f"Configuration saved to: {log_path}\n")
    return log_path
