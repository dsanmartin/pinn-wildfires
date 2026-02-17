from __future__ import annotations

from argparse import ArgumentParser
from configparser import ConfigParser
from datetime import datetime
from pathlib import Path

from wildfire.plot import plot_perimeter_comparison, plot_phi_comparison, plot_temperature_comparison, plot_fuel_fraction_comparison, plot_training_loss, plot_mass_conservation_comparison
from wildfire.logging_utils import write_config_log
from wildfire.pde import PDEConfig
from wildfire.lsm import LSMConfig
from wildfire.asensio import AsensioConfig
from wildfire.train import ExperimentConfig, train
from wildfire.pinn import MLPConfig
from wildfire.utils import Domain, TrainConfig
from wildfire.mass_conservation import MassConservationConfig


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train PINN for Level Set Method and plot perimeters.")

    parser.add_argument("--config", type=str, default=None, help="Path to INI config file.")

    # Domain parameters
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument("--z-min", type=float, default=None)
    parser.add_argument("--z-max", type=float, default=None)
    parser.add_argument("--t-min", type=float, default=None)
    parser.add_argument("--t-max", type=float, default=None)

    # Training parameters
    parser.add_argument("--n-interior", type=int, default=None)
    parser.add_argument("--n-boundary", type=int, default=None)
    parser.add_argument("--n-initial", type=int, default=None)
    parser.add_argument("--lr-adam", type=float, default=None)
    parser.add_argument("--lr-lbfgs", type=float, default=None)
    parser.add_argument("--epochs-adam", type=int, default=None)
    parser.add_argument("--epochs-lbfgs", type=int, default=None)
    parser.add_argument("--weight-pde", type=float, default=None)
    parser.add_argument("--weight-bc", type=float, default=None)
    parser.add_argument("--weight-ic", type=float, default=None)

    # Initial condition parameters
    parser.add_argument("--center-x", type=float, default=None)
    parser.add_argument("--center-y", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--sx", type=float, default=None)
    parser.add_argument("--sy", type=float, default=None)

    # Velocity parameters
    parser.add_argument("--vx", type=float, default=None)
    parser.add_argument("--vy", type=float, default=None)
    return parser


def _read_config(path: str | None) -> ConfigParser | None:
    if not path:
        return None
    cfg = ConfigParser()
    cfg.read(path)
    return cfg


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    base_domain = Domain()
    base_train = TrainConfig()
    base_lsm_config = LSMConfig()  # Default LSM config
    base_asensio_config = AsensioConfig()  # Default Asensio config
    base_mass_config = MassConservationConfig()  # Default Mass Conservation config
    cfg_file = _read_config(args.config)

    if cfg_file is not None and cfg_file.has_section("domain"):
        base_domain = Domain(
            x_min=cfg_file.getfloat("domain", "x_min", fallback=base_domain.x_min),
            x_max=cfg_file.getfloat("domain", "x_max", fallback=base_domain.x_max),
            y_min=cfg_file.getfloat("domain", "y_min", fallback=base_domain.y_min),
            y_max=cfg_file.getfloat("domain", "y_max", fallback=base_domain.y_max),
            z_min=cfg_file.getfloat("domain", "z_min", fallback=base_domain.z_min),
            z_max=cfg_file.getfloat("domain", "z_max", fallback=base_domain.z_max),
            t_min=cfg_file.getfloat("domain", "t_min", fallback=base_domain.t_min),
            t_max=cfg_file.getfloat("domain", "t_max", fallback=base_domain.t_max),
        )

    if cfg_file is not None and cfg_file.has_section("train"):
        base_train = TrainConfig(
            n_interior=cfg_file.getint("train", "n_interior", fallback=base_train.n_interior),
            n_boundary=cfg_file.getint("train", "n_boundary", fallback=base_train.n_boundary),
            n_initial=cfg_file.getint("train", "n_initial", fallback=base_train.n_initial),
            lr_adam=cfg_file.getfloat("train", "lr_adam", fallback=base_train.lr_adam),
            lr_factor=cfg_file.getfloat("train", "lr_factor", fallback=base_train.lr_factor),
            lr_min=cfg_file.getfloat("train", "lr_min", fallback=base_train.lr_min),
            lr_patience_adam=cfg_file.getint("train", "lr_patience_adam", fallback=base_train.lr_patience_adam),
            lr_patience_lbfgs=cfg_file.getint("train", "lr_patience_lbfgs", fallback=base_train.lr_patience_lbfgs),
            early_stop_patience_adam=cfg_file.getint("train", "early_stop_patience_adam", fallback=base_train.early_stop_patience_adam),
            early_stop_patience_lbfgs=cfg_file.getint("train", "early_stop_patience_lbfgs", fallback=base_train.early_stop_patience_lbfgs),
            early_stop_min_delta=cfg_file.getfloat("train", "early_stop_min_delta", fallback=base_train.early_stop_min_delta),
            epochs_adam=cfg_file.getint("train", "epochs_adam", fallback=base_train.epochs_adam),
            epochs_lbfgs=cfg_file.getint("train", "epochs_lbfgs", fallback=base_train.epochs_lbfgs),
            lr_lbfgs=cfg_file.getfloat("train", "lr_lbfgs", fallback=base_train.lr_lbfgs),
            weight_pde=cfg_file.getfloat("train", "weight_pde", fallback=base_train.weight_pde),
            weight_bc=cfg_file.getfloat("train", "weight_bc", fallback=base_train.weight_bc),
            weight_ic=cfg_file.getfloat("train", "weight_ic", fallback=base_train.weight_ic),
        )

    if cfg_file is not None and cfg_file.has_section("initial"):
        base_lsm_config = LSMConfig(
            center=(
                cfg_file.getfloat("initial", "center_x", fallback=base_lsm_config.center[0]),
                cfg_file.getfloat("initial", "center_y", fallback=base_lsm_config.center[1]),
            ),
            radius=cfg_file.getfloat("initial", "radius", fallback=base_lsm_config.radius),
            speed=base_lsm_config.speed,
            epsilon=base_lsm_config.epsilon,
            vx=base_lsm_config.vx,
            vy=base_lsm_config.vy,
            r0=base_lsm_config.r0,
            cf=base_lsm_config.cf,
            bc_type=base_lsm_config.bc_type,
        )

    if cfg_file is not None and cfg_file.has_section("pde"):
        model_type = cfg_file.get("pde", "model_type", fallback="lsm")

        print(model_type)
        
        if model_type.lower() == "asensio":
            fallback_radius = cfg_file.getfloat("initial", "radius", fallback=None)
            sx = cfg_file.getfloat("initial", "sx", fallback=base_asensio_config.sx if fallback_radius is None else fallback_radius)
            sy = cfg_file.getfloat("initial", "sy", fallback=base_asensio_config.sy if fallback_radius is None else fallback_radius)
            asensio_cfg = AsensioConfig(
                T0=cfg_file.getfloat("pde", "T0", fallback=base_asensio_config.T0),
                T_inf=cfg_file.getfloat("pde", "T_inf", fallback=base_asensio_config.T_inf),
                center=(
                    cfg_file.getfloat("initial", "center_x", fallback=base_asensio_config.center[0]),
                    cfg_file.getfloat("initial", "center_y", fallback=base_asensio_config.center[1]),
                ),
                sx=sx,
                sy=sy,
                vx=cfg_file.getfloat("pde", "vx", fallback=base_asensio_config.vx),
                vy=cfg_file.getfloat("pde", "vy", fallback=base_asensio_config.vy),
                k=cfg_file.getfloat("pde", "k", fallback=base_asensio_config.k),
                T_act=cfg_file.getfloat("pde", "T_act", fallback=base_asensio_config.T_act),
                Y_f=cfg_file.getfloat("pde", "Y_f", fallback=base_asensio_config.Y_f),
                delta=cfg_file.getfloat("pde", "delta", fallback=base_asensio_config.delta),
                T_ign=cfg_file.getfloat("pde", "T_ign", fallback=base_asensio_config.T_ign),
                H_C=cfg_file.getfloat("pde", "H_C", fallback=base_asensio_config.H_C),
                bc_type=cfg_file.get("pde", "bc_type", fallback=base_asensio_config.bc_type),
            )
            base_lsm = PDEConfig(model_type="asensio", asensio_config=asensio_cfg)
        elif model_type.lower() == "mass_conservation":
            initial_section = "initial" if cfg_file.has_section("initial") else "pde"
            mass_cfg = MassConservationConfig(
                a1=cfg_file.getfloat("pde", "a1", fallback=base_mass_config.a1),
                a2=cfg_file.getfloat("pde", "a2", fallback=base_mass_config.a2),
                l=cfg_file.getfloat("pde", "l", fallback=base_mass_config.l),
                z0=cfg_file.getfloat(initial_section, "z0", fallback=cfg_file.getfloat("pde", "z0", fallback=base_mass_config.z0)),
                U0=cfg_file.getfloat(initial_section, "U0", fallback=cfg_file.getfloat("pde", "U0", fallback=base_mass_config.U0)),
                V0=cfg_file.getfloat(initial_section, "V0", fallback=cfg_file.getfloat("pde", "V0", fallback=base_mass_config.V0)),
                W0=cfg_file.getfloat(initial_section, "W0", fallback=cfg_file.getfloat("pde", "W0", fallback=base_mass_config.W0)),
                topo_height=cfg_file.getfloat(initial_section, "topo_height", fallback=cfg_file.getfloat("pde", "topo_height", fallback=base_mass_config.topo_height)),
                topo_sigma_x=cfg_file.getfloat(initial_section, "topo_sigma_x", fallback=cfg_file.getfloat("pde", "topo_sigma_x", fallback=base_mass_config.topo_sigma_x)),
                topo_sigma_y=cfg_file.getfloat(initial_section, "topo_sigma_y", fallback=cfg_file.getfloat("pde", "topo_sigma_y", fallback=base_mass_config.topo_sigma_y)),
                topo_base=cfg_file.getfloat(initial_section, "topo_base", fallback=cfg_file.getfloat("pde", "topo_base", fallback=base_mass_config.topo_base)),
                topo_center=(
                    cfg_file.getfloat(initial_section, "topo_center_x", fallback=cfg_file.getfloat("pde", "topo_center_x", fallback=base_mass_config.topo_center[0])),
                    cfg_file.getfloat(initial_section, "topo_center_y", fallback=cfg_file.getfloat("pde", "topo_center_y", fallback=base_mass_config.topo_center[1])),
                ),
                bc_type=cfg_file.get("pde", "bc_type", fallback=base_mass_config.bc_type),
            )
            base_lsm = PDEConfig(model_type="mass_conservation", mass_conservation_config=mass_cfg)
        else:
            lsm_cfg = LSMConfig(
                speed=cfg_file.getfloat("pde", "speed", fallback=base_lsm_config.speed),
                epsilon=cfg_file.getfloat("pde", "epsilon", fallback=base_lsm_config.epsilon),
                center=(
                    cfg_file.getfloat("initial", "center_x", fallback=base_lsm_config.center[0]),
                    cfg_file.getfloat("initial", "center_y", fallback=base_lsm_config.center[1]),
                ),
                radius=cfg_file.getfloat("initial", "radius", fallback=base_lsm_config.radius),
                vx=cfg_file.getfloat("pde", "vx", fallback=base_lsm_config.vx),
                vy=cfg_file.getfloat("pde", "vy", fallback=base_lsm_config.vy),
                r0=cfg_file.getfloat("pde", "r0", fallback=base_lsm_config.r0),
                cf=cfg_file.getfloat("pde", "cf", fallback=base_lsm_config.cf),
                bc_type=cfg_file.get("pde", "bc_type", fallback=base_lsm_config.bc_type),
            )
            base_lsm = PDEConfig(model_type="lsm", lsm_config=lsm_cfg)
    else:
        base_lsm = PDEConfig(model_type="lsm", lsm_config=base_lsm_config)

    domain = Domain(
        x_min=base_domain.x_min if args.x_min is None else args.x_min,
        x_max=base_domain.x_max if args.x_max is None else args.x_max,
        y_min=base_domain.y_min if args.y_min is None else args.y_min,
        y_max=base_domain.y_max if args.y_max is None else args.y_max,
        z_min=base_domain.z_min if args.z_min is None else args.z_min,
        z_max=base_domain.z_max if args.z_max is None else args.z_max,
        t_min=base_domain.t_min if args.t_min is None else args.t_min,
        t_max=base_domain.t_max if args.t_max is None else args.t_max,
    )
    train_cfg = TrainConfig(
        n_interior=base_train.n_interior if args.n_interior is None else args.n_interior,
        n_boundary=base_train.n_boundary if args.n_boundary is None else args.n_boundary,
        n_initial=base_train.n_initial if args.n_initial is None else args.n_initial,
        lr_adam=base_train.lr_adam if args.lr_adam is None else args.lr_adam,
        lr_factor=base_train.lr_factor,
        lr_min=base_train.lr_min,
        lr_patience_adam=base_train.lr_patience_adam,
        lr_patience_lbfgs=base_train.lr_patience_lbfgs,
        early_stop_patience_adam=base_train.early_stop_patience_adam,
        early_stop_patience_lbfgs=base_train.early_stop_patience_lbfgs,
        early_stop_min_delta=base_train.early_stop_min_delta,
        epochs_adam=base_train.epochs_adam if args.epochs_adam is None else args.epochs_adam,
        epochs_lbfgs=base_train.epochs_lbfgs if args.epochs_lbfgs is None else args.epochs_lbfgs,
        lr_lbfgs=base_train.lr_lbfgs if args.lr_lbfgs is None else args.lr_lbfgs,
        weight_pde=base_train.weight_pde if args.weight_pde is None else args.weight_pde,
        weight_bc=base_train.weight_bc if args.weight_bc is None else args.weight_bc,
        weight_ic=base_train.weight_ic if args.weight_ic is None else args.weight_ic,
    )
    pde_cfg = base_lsm
    if pde_cfg.model_type.lower() == "asensio" and pde_cfg.asensio_config:
        if args.sx is not None:
            pde_cfg.asensio_config.sx = args.sx
        if args.sy is not None:
            pde_cfg.asensio_config.sy = args.sy

    # Set PINN output size based on model type
    # LSM: 1 output (level set function)
    # Asensio: 2 outputs (temperature and fuel fraction)
    # Mass conservation: 3 outputs (u, v, w)
    model_type = pde_cfg.model_type.lower()
    if model_type == "asensio":
        output_size = 2
    elif model_type == "mass_conservation":
        output_size = 3
    else:
        output_size = 1
    model_cfg = MLPConfig(layers=(3, 64, 64, 64, output_size))

    cfg = ExperimentConfig(domain=domain, train=train_cfg, model=model_cfg, pde=pde_cfg)
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    base_out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir = base_out_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    write_config_log(cfg, out_dir)

    model, loss_history = train(cfg)
    
    # Save training loss plot
    plot_training_loss(loss_history, out_dir / "training_loss.png")
    
    if cfg.pde.model_type.lower() == "asensio":
        plot_temperature_comparison(model, cfg.domain, out_dir / "temperature_comparison.png", pde_cfg=cfg.pde)
        plot_fuel_fraction_comparison(model, cfg.domain, out_dir / "fuel_fraction_comparison.png", pde_cfg=cfg.pde)
    elif cfg.pde.model_type.lower() == "mass_conservation":
        plot_mass_conservation_comparison(model, cfg.domain, out_dir, pde_cfg=cfg.pde)
    else:
        plot_phi_comparison(model, cfg.domain, out_dir / "phi_comparison.png", lsm=cfg.pde)
        plot_perimeter_comparison(model, cfg.domain, out_dir / "perimeter_comparison.png", lsm=cfg.pde)
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
