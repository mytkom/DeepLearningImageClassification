import accelerate
import tyro
import yaml
import json

import wandb

from configs import Config
from engine import build_engine
from utils.config_merge import merge_configs


def sweep(cfg: Config):
    project_config = accelerate.utils.ProjectConfiguration(
        project_dir=cfg.project_dir,
        logging_dir=cfg.log_dir,
    )
    accelerator = accelerate.Accelerator(
        log_with=cfg.project_tracker,
        project_config=project_config,
        gradient_accumulation_steps=cfg.training.accum_iter,
        mixed_precision=cfg.mixed_precision,
    )

    accelerate.utils.set_seed(cfg.seed, device_specific=True)
    engine = build_engine(cfg.training.engine)(accelerator, cfg)
    engine.train()
    engine.close()


if __name__ == "__main__":
    cfg = tyro.cli(Config)

    # if config path specified, override CLI config only
    # fields specified in json config
    if cfg.config is not None:
        with open(cfg.config, "r") as f:
            json_cfg = json.load(f)
        if cfg.model.resume_path is not None:
            json_cfg["model"]["resume_path"] = cfg.model.resume_path

    merge_configs(cfg, json_cfg)

    with open(cfg.sweep.config) as f:
        sweep_configuration = yaml.safe_load(f)

    if cfg.sweep.name:
        sweep_configuration["name"] = cfg.sweep.name

    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project=cfg.sweep.project_name,
    )
    wandb.agent(sweep_id, function=lambda: sweep(cfg))
