import accelerate
import tyro
import yaml

import wandb

from configs import Config
from engine import build_engine


def sweep():
    cfg = tyro.cli(Config)

    if cfg.config is not None:
        with open(cfg.config, "r") as f:
            json_cfg = Config.from_json(f.read())
        if cfg.model.resume_path is not None:
            json_cfg.model.resume_path = cfg.model.resume_path
        cfg = json_cfg

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
    with open("configuration/cinic/sweep_config.yaml") as f:
        sweep_configuration = yaml.safe_load(f)

    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project="sweep_test",
    )
    wandb.agent(sweep_id, function=sweep)
