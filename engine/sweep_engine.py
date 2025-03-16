import wandb

import dataclasses

from engine.train_engine import Engine


def merge_configs(self_cfg, wandb_cfg):
    """Merge wandb config into self config, preserving defaults."""
    for key, value in wandb_cfg.items():
        if hasattr(self_cfg, key):
            sub_cfg = getattr(self_cfg, key)
            if dataclasses.is_dataclass(sub_cfg):
                for sub_key, sub_value in value.items():
                    if hasattr(sub_cfg, sub_key):
                        setattr(sub_cfg, sub_key, sub_value)
            else:
                setattr(self_cfg, key, value)


class SweepEngine(Engine):
    def setup_training(self):
        super().setup_training()
        self.accelerator.wait_for_everyone()

        merge_configs(self.cfg, wandb.config)
