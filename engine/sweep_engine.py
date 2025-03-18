import wandb

import dataclasses

from engine.train_engine import Engine


def merge_configs(self_cfg, wandb_cfg):
    """Recursively merge wandb config into self config, preserving defaults."""
    for key, value in wandb_cfg.items():
        if hasattr(self_cfg, key):
            sub_cfg = getattr(self_cfg, key)
            if dataclasses.is_dataclass(sub_cfg) and isinstance(value, dict):
                merge_configs(sub_cfg, value)
            else:
                setattr(self_cfg, key, value)


class SweepEngine(Engine):
    def setup_training(self):
        self.accelerator.wait_for_everyone()

        merge_configs(self.cfg, wandb.config)

        super().setup_training()
